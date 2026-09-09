from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from langchain_core.messages import HumanMessage, ToolMessage

from core.config import AgentConfig
from core.errors import ErrorType, format_error
from core.fast_copy import copy_jsonish
from core.message_utils import compact_text, is_error_text, stringify_content
from core.policy_engine import classify_shell_command
from core.self_correction_engine import repair_fingerprint
from core.tool_issues import build_tool_issue, enrich_tool_issue_details, merge_tool_issues
from core.tool_policy import ToolMetadata
from core.tool_results import ToolExecutionResult, parse_tool_execution_result
from core.tool_output_compressor import ToolOutputCompressor
from core.validation import validate


LogRunEvent = Callable[..., None]
MetadataForTool = Callable[[str], ToolMetadata]
WorkspaceBoundaryChecker = Callable[[str, Dict[str, Any]], bool]


@dataclass(frozen=True)
class ToolExecutionOutcome:
    tool_message: ToolMessage
    parsed_result: ToolExecutionResult
    had_error: bool
    issue: Dict[str, Any] | None
    content: str


class ToolExecutor:
    def __init__(
        self,
        *,
        config: AgentConfig,
        metadata_for_tool: MetadataForTool,
        log_run_event: LogRunEvent,
        workspace_boundary_violated: WorkspaceBoundaryChecker,
    ) -> None:
        self.config = config
        self._metadata_for_tool = metadata_for_tool
        self._log_run_event = log_run_event
        self._workspace_boundary_violated = workspace_boundary_violated
        self._output_compressor = ToolOutputCompressor(
            enabled=getattr(config, "enable_headroom_compression", False),
        )

    @staticmethod
    def _result_log_payload(parsed_result: ToolExecutionResult) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"ok": bool(parsed_result.ok)}
        if parsed_result.error_type:
            payload["error_type"] = parsed_result.error_type
        summary = compact_text(
            str(parsed_result.message or parsed_result.raw or "").strip(),
            240,
        )
        if summary:
            payload["summary"] = summary
        if parsed_result.retryable:
            payload["retryable"] = True
        return payload

    @staticmethod
    def _latest_user_query(state: Dict[str, Any] | None) -> str:
        """Latest human turn, used by headroom for relevance-aware compression."""
        if not state:
            return ""
        messages = state.get("messages") or []
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return compact_text(stringify_content(message.content), 2000)
        return ""

    def merge_issues(self, issues: list[Dict[str, Any]], *, current_turn_id: int) -> Dict[str, Any] | None:
        return merge_tool_issues(issues, current_turn_id=current_turn_id)

    def handle_result(
        self,
        *,
        state: Dict[str, Any] | None,
        current_turn_id: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: str,
        content: str,
        issue_details: Dict[str, Any] | None = None,
        tool_duration_seconds: float | None = None,
        apply_validation: bool = True,
        had_error: bool = False,
    ) -> ToolExecutionOutcome:
        if apply_validation:
            validation_error = validate(
                content,
                {
                    "tool_name": tool_name,
                    "args": copy_jsonish(tool_args) if isinstance(tool_args, dict) else {},
                },
            )
            if validation_error:
                if str(content or "").strip():
                    content = f"{validation_error}\n\nTool output:\n{content}"
                else:
                    content = validation_error
                had_error = True

        limit = self.config.safety.max_tool_output
        is_mcp = self._metadata_for_tool(tool_name).source == "mcp"
        compressed = self._output_compressor.compress(
            content=content,
            tool_name=tool_name,
            tool_args=tool_args if isinstance(tool_args, dict) else None,
            limit=limit,
            is_mcp=is_mcp,
            user_query=self._latest_user_query(state),
        )
        content = self._output_compressor.reduce_to_limit(
            content=compressed if compressed is not None else content,
            tool_name=tool_name,
            limit=limit,
            is_mcp=is_mcp,
        )
        parsed_result = parse_tool_execution_result(content)
        if not parsed_result.ok:
            had_error = True
        self._log_interrupted_tool_result_if_needed(
            state=state,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            parsed_result=parsed_result,
        )
        self._log_run_event(
            state,
            "tool_call_end",
            run_id="" if state is None else state.get("run_id", ""),
            tool_name=tool_name,
            tool_args=tool_args,
            result=self._result_log_payload(parsed_result),
        )

        issue = None
        if not parsed_result.ok:
            issue = self._build_open_tool_issue(
                state=state,
                current_turn_id=current_turn_id,
                tool_name=tool_name,
                tool_args=tool_args,
                parsed_result=parsed_result,
                content=content,
                issue_details=issue_details,
            )

        return ToolExecutionOutcome(
            tool_message=self._build_tool_message(
                content=content,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_duration_seconds=tool_duration_seconds,
                parsed_result=parsed_result,
            ),
            parsed_result=parsed_result,
            had_error=had_error,
            issue=issue,
            content=content,
        )

    def build_not_allowed_result(
        self,
        *,
        state: Dict[str, Any] | None,
        current_turn_id: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: str,
        allowed_tool_names: list[str],
    ) -> ToolExecutionOutcome:
        allowed_label = ", ".join(allowed_tool_names) if allowed_tool_names else "none"
        content = format_error(
            ErrorType.VALIDATION,
            f"Tool '{tool_name}' is not allowed for this request. Allowed tools: {allowed_label}.",
        )
        self._log_run_event(
            state,
            "tool_call_not_available_blocked",
            run_id="" if state is None else state.get("run_id", ""),
            tool_name=tool_name,
            tool_args=tool_args,
            allowed_tool_names=allowed_tool_names,
        )
        return self.handle_result(
            state=state,
            current_turn_id=current_turn_id,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            content=content,
            issue_details={"allowed_tool_names": allowed_tool_names},
            apply_validation=False,
            had_error=True,
        )

    def build_denied_result(
        self,
        *,
        state: Dict[str, Any] | None,
        current_turn_id: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: str,
        policy: Dict[str, Any],
    ) -> ToolExecutionOutcome:
        content = format_error(
            ErrorType.ACCESS_DENIED,
            f"Execution of '{tool_name}' was cancelled by approval policy.",
        )
        self._log_run_event(
            state,
            "tool_call_denied",
            run_id="" if state is None else state.get("run_id", ""),
            tool_name=tool_name,
            tool_args=tool_args,
            policy=policy,
        )
        return self.handle_result(
            state=state,
            current_turn_id=current_turn_id,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            content=content,
            issue_details={"approval_denied": True, "needs_external_input": True},
            apply_validation=False,
            had_error=True,
        )

    def build_missing_required_result(
        self,
        *,
        state: Dict[str, Any] | None,
        current_turn_id: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: str,
        missing_required: list[str],
    ) -> ToolExecutionOutcome:
        content = format_error(
            ErrorType.VALIDATION,
            f"Missing required field(s): {', '.join(missing_required)}.",
        )
        self._log_run_event(
            state,
            "tool_call_preflight_validation_failed",
            run_id="" if state is None else state.get("run_id", ""),
            tool_name=tool_name,
            tool_args=tool_args,
            missing_required=missing_required,
        )
        return self.handle_result(
            state=state,
            current_turn_id=current_turn_id,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            content=content,
            issue_details={"missing_required_fields": missing_required},
            apply_validation=False,
            had_error=True,
        )

    def _tool_error_requires_recovery_gate(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        parsed_result: ToolExecutionResult,
    ) -> bool:
        if parsed_result.ok:
            return False

        error_type = str(parsed_result.error_type or "").strip().upper()
        if error_type == "ACCESS_DENIED":
            return True

        metadata = self._metadata_for_tool(tool_name)
        if metadata.requires_approval or metadata.destructive or metadata.mutating:
            return True

        if tool_name == "cli_exec":
            command = str((tool_args or {}).get("command", "") or "")
            profile = classify_shell_command(command)
            if profile.get("inspect_only") and not profile.get("long_running_service"):
                return False
            if (
                profile.get("mutating")
                or profile.get("destructive")
                or profile.get("long_running_service")
            ):
                return True

        if metadata.read_only:
            return False

        return True

    def _build_open_tool_issue(
        self,
        *,
        state: Dict[str, Any] | None,
        current_turn_id: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        parsed_result: ToolExecutionResult,
        content: str,
        issue_details: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        if not self._tool_error_requires_recovery_gate(tool_name, tool_args, parsed_result):
            self._log_run_event(
                state,
                "tool_error_returned_to_agent",
                run_id="" if state is None else state.get("run_id", ""),
                tool_name=tool_name,
                tool_args=tool_args,
                error_type=parsed_result.error_type,
                summary=compact_text(parsed_result.message or content, 220),
            )
            return None

        details = enrich_tool_issue_details(
            tool_name,
            tool_args,
            parsed_result,
            issue_details=issue_details,
            workspace_boundary_violated=self._workspace_boundary_violated,
        )
        metadata = self._metadata_for_tool(tool_name)
        details.setdefault("retryable", bool(parsed_result.retryable))
        details.setdefault("tool_read_only", bool(metadata.read_only))
        details.setdefault("tool_mutating", bool(metadata.mutating))
        details.setdefault("tool_destructive", bool(metadata.destructive))
        details.setdefault("tool_requires_approval", bool(metadata.requires_approval))
        issue_kind = "approval_denied" if details.get("approval_denied") else "tool_error"
        issue_source = "approval" if issue_kind == "approval_denied" else "tools"
        return build_tool_issue(
            current_turn_id=current_turn_id,
            kind=issue_kind,
            summary=parsed_result.message or content,
            tool_names=[tool_name],
            tool_args=tool_args,
            source=issue_source,
            error_type=parsed_result.error_type,
            fingerprint=repair_fingerprint(tool_name, tool_args, parsed_result.error_type),
            progress_fingerprint=repair_fingerprint(tool_name, tool_args, parsed_result.error_type),
            details=details,
        )

    @staticmethod
    def _build_tool_message(
        *,
        content: str,
        tool_call_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_duration_seconds: float | None = None,
        parsed_result: ToolExecutionResult | None = None,
    ) -> ToolMessage:
        parsed = parsed_result or parse_tool_execution_result(content)
        additional_kwargs: Dict[str, Any] = {
            "tool_args": copy_jsonish(tool_args) if isinstance(tool_args, dict) else {}
        }
        if tool_duration_seconds is not None:
            additional_kwargs["tool_duration_seconds"] = float(tool_duration_seconds)
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
            additional_kwargs=additional_kwargs,
            status="error" if not parsed.ok else "success",
        )

    def _log_interrupted_tool_result_if_needed(
        self,
        *,
        state: Dict[str, Any] | None,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: str,
        parsed_result: ToolExecutionResult,
    ) -> None:
        if parsed_result.ok:
            return

        message = str(parsed_result.message or "").lower()
        reason = ""
        if parsed_result.error_type == "TIMEOUT":
            reason = "timeout"
        elif "interactive prompt detected" in message:
            reason = "interactive_prompt"
        elif "execution interrupted" in message:
            reason = "execution_interrupted"

        if not reason:
            return

        self._log_run_event(
            state,
            "tool_call_interrupted",
            run_id="" if state is None else state.get("run_id", ""),
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            reason=reason,
        )
