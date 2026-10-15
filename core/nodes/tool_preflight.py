from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from core.message_utils import recent_tool_calls
from core.tool_policy import ToolMetadata, default_tool_metadata
from core.state import OpenToolIssue, RecoveryState
from core.tool_args import (
    canonicalize_tool_args,
    inspect_tool_args_payload,
    tool_loop_budget,
)
from core.tool_issues import build_tool_issue
from core.policy_engine import classify_shell_command, shell_command_requires_approval, tool_requires_approval
from core.self_correction_engine import normalize_tool_args, repair_fingerprint

logger = logging.getLogger("agent")

# Sentinel for "tool not found" in the required-fields cache.
_REQUIRED_FIELDS_NOT_FOUND: tuple[str, ...] = ()


class ToolPreflightMixin:
    """Tool metadata, JSON-schema introspection, validation, and loop detection."""

    def _metadata_for_tool(self, tool_name: str) -> ToolMetadata:
        return self.tool_metadata.get(tool_name, default_tool_metadata(tool_name))

    def _normalize_tool_name(self, tool_name: str) -> str:
        return str(tool_name or "").strip().lower()

    @staticmethod
    def _resolve_local_json_schema_ref(
        json_schema: Dict[str, Any],
        ref: str,
    ) -> Dict[str, Any] | None:
        if not isinstance(json_schema, dict) or not isinstance(ref, str) or not ref.startswith("#/"):
            return None

        cursor: Any = json_schema
        for raw_segment in ref[2:].split("/"):
            segment = raw_segment.replace("~1", "/").replace("~0", "~")
            if not isinstance(cursor, dict) or segment not in cursor:
                return None
            cursor = cursor[segment]

        return cursor if isinstance(cursor, dict) else None

    def _top_level_object_json_schema(self, json_schema: Dict[str, Any]) -> Dict[str, Any] | None:
        if not isinstance(json_schema, dict):
            return None

        current = json_schema
        seen_refs: set[str] = set()
        while isinstance(current, dict):
            if current.get("type") == "object":
                return current

            ref = current.get("$ref")
            if isinstance(ref, str):
                if ref in seen_refs:
                    return None
                seen_refs.add(ref)
                resolved = self._resolve_local_json_schema_ref(json_schema, ref)
                if not isinstance(resolved, dict):
                    return None
                current = resolved
                continue

            composed_schema = current.get("allOf")
            if isinstance(composed_schema, list) and len(composed_schema) == 1 and isinstance(composed_schema[0], dict):
                current = composed_schema[0]
                continue

            return None

        return None

    def _required_tool_fields(self, tool_name: str) -> List[str]:
        cached = self._required_fields_cache.get(tool_name)
        if cached is not None:
            return list(cached)

        tool = self.tools_map.get(tool_name)
        if not tool:
            self._required_fields_cache[tool_name] = _REQUIRED_FIELDS_NOT_FOUND
            return []
        try:
            schema = tool.get_input_schema()
        except Exception:
            logger.debug("Failed to get input schema for tool '%s'.", tool_name, exc_info=True)
            self._required_fields_cache[tool_name] = _REQUIRED_FIELDS_NOT_FOUND
            return []

        try:
            json_schema = schema.model_json_schema()
        except Exception:
            logger.debug("Failed to serialize JSON schema for tool '%s'.", tool_name, exc_info=True)
            json_schema = {}

        object_schema = self._top_level_object_json_schema(json_schema)
        if isinstance(object_schema, dict):
            required_fields = object_schema.get("required")
            if isinstance(required_fields, list):
                result = tuple(str(field_name) for field_name in required_fields if str(field_name).strip())
                self._required_fields_cache[tool_name] = result
                return list(result)
            self._required_fields_cache[tool_name] = _REQUIRED_FIELDS_NOT_FOUND
            return []

        if getattr(schema, "__pydantic_root_model__", False):
            self._required_fields_cache[tool_name] = _REQUIRED_FIELDS_NOT_FOUND
            return []

        fields = getattr(schema, "model_fields", {}) or {}
        required_list: List[str] = []
        for field_name, field_info in fields.items():
            try:
                if field_info.is_required():
                    required_list.append(str(field_name))
            except Exception:
                logger.debug("Failed to check field '%s' on tool '%s'.", field_name, tool_name, exc_info=True)
                continue
        result = tuple(required_list)
        self._required_fields_cache[tool_name] = result
        return list(result)

    def _missing_required_tool_fields(self, tool_name: str, tool_args: Dict[str, Any]) -> List[str]:
        normalized_args = canonicalize_tool_args(tool_args)
        tool = self.tools_map.get(tool_name)
        if tool:
            try:
                schema = tool.get_input_schema()
                validated = schema.model_validate(normalized_args)
                payload = validated.model_dump() if hasattr(validated, "model_dump") else normalized_args
                if isinstance(payload, dict):
                    normalized_args = payload
            except ValidationError as exc:
                missing_from_schema: List[str] = []
                for error in exc.errors():
                    if str(error.get("type") or "").strip() != "missing":
                        continue
                    loc = error.get("loc") or ()
                    if not loc:
                        continue
                    field_name = str(loc[-1]).strip()
                    if field_name and field_name not in missing_from_schema:
                        missing_from_schema.append(field_name)
                if missing_from_schema:
                    return missing_from_schema
            except Exception:
                pass

        if self._normalize_tool_name(tool_name) == "write_file":
            content = normalized_args.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                return ["content"]

        required = self._required_tool_fields(tool_name)
        if not required:
            return []
        missing: List[str] = []
        for field_name in required:
            value = normalized_args.get(field_name)
            if value is None:
                missing.append(field_name)
                continue
            if isinstance(value, str) and not value.strip():
                missing.append(field_name)
        return missing

    def _normalize_tool_args_for_preflight(
        self,
        tool_name: str,
        raw_tool_args: Any,
        *,
        current_task: str = "",
    ) -> Dict[str, Any]:
        tool_args, _ = inspect_tool_args_payload(raw_tool_args)
        normalized_args, _ = normalize_tool_args(
            tool_name,
            tool_args,
            current_task=current_task,
        )
        return normalized_args

    def _tool_call_requires_ready_approval(
        self,
        tool_name: str,
        raw_tool_args: Any,
        *,
        current_task: str = "",
    ) -> bool:
        tool_args = self._normalize_tool_args_for_preflight(
            tool_name,
            raw_tool_args,
            current_task=current_task,
        )
        return (
            self._tool_requires_approval(tool_name, tool_args)
            and not self._missing_required_tool_fields(tool_name, tool_args)
        )

    def _tool_is_read_only(self, tool_name: str) -> bool:
        metadata = self._metadata_for_tool(tool_name)
        return metadata.read_only and not metadata.mutating and not metadata.destructive

    def _effective_tool_metadata(self, tool_name: str, tool_args: Dict[str, Any] | None = None) -> ToolMetadata:
        metadata = self._metadata_for_tool(tool_name)
        if self._normalize_tool_name(tool_name) != "cli_exec":
            return metadata

        command = str(((tool_args or {}).get("command")) or "").strip()
        if not command:
            return metadata

        profile = classify_shell_command(command)
        approval_required = shell_command_requires_approval(command)
        inspect_only = bool(profile.get("inspect_only") and not profile.get("long_running_service"))
        return ToolMetadata(
            name=metadata.name,
            read_only=inspect_only,
            mutating=bool(profile.get("mutating") or profile.get("long_running_service")),
            destructive=bool(profile.get("destructive")),
            requires_approval=approval_required,
            networked=bool(metadata.networked or profile.get("network_diagnostic") or profile.get("http_probe")),
            source=metadata.source,
        )

    def _tool_requires_approval(self, tool_name: str, tool_args: Dict[str, Any] | None = None) -> bool:
        metadata = self._effective_tool_metadata(tool_name, tool_args)
        return tool_requires_approval(
            tool_name,
            tool_args,
            metadata=metadata,
            approvals_enabled=self.config.enable_approvals,
        )

    def tool_calls_require_approval(self, tool_calls: List[Dict[str, Any]]) -> bool:
        return any(
            self._tool_call_requires_ready_approval(
                (tool_call.get("name") or "unknown_tool"),
                tool_call.get("args"),
            )
            for tool_call in tool_calls
            if tool_call.get("name") != "request_user_input"
        )

    def _iter_path_like_targets(self, tool_args: Dict[str, Any]) -> List[Tuple[str, str]]:
        if not isinstance(tool_args, dict):
            return []
        targets: List[Tuple[str, str]] = []
        for key in ("path", "file_path", "dir_path", "source", "destination", "cwd"):
            raw_value = tool_args.get(key)
            if isinstance(raw_value, str) and raw_value.strip():
                targets.append((key, raw_value.strip()))
        return targets

    def _tool_mutates_workspace(self, tool_name: str) -> bool:
        metadata = self._metadata_for_tool(tool_name)
        return metadata.mutating or metadata.destructive

    def _workspace_boundary_violated(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        if not self._tool_mutates_workspace(tool_name):
            return False

        if tool_name == "run_background_process":
            try:
                from tools import process_tools

                process_tools._resolve_cwd(str((tool_args or {}).get("cwd") or "."))
                return False
            except Exception:
                return True

        path_targets = self._iter_path_like_targets(tool_args)
        if not path_targets:
            return False

        try:
            from tools.filesystem import resolve_workspace_path
        except Exception:
            return False

        for _, raw_value in path_targets:
            try:
                resolve_workspace_path(raw_value)
            except Exception:
                return True
        return False

    def _merge_open_tool_issues(
        self,
        issues: List[Dict[str, Any]],
        current_turn_id: int,
    ) -> Dict[str, Any] | None:
        return self.tool_executor.merge_issues(issues, current_turn_id=current_turn_id)

    def _preflight_recovery_loop_issue(
        self,
        messages: List[Any],
        *,
        current_turn_id: int,
        open_tool_issue: OpenToolIssue | None,
        recovery_state: RecoveryState | None,
    ) -> OpenToolIssue | None:
        if not isinstance(open_tool_issue, dict):
            return None
        if not isinstance(recovery_state, dict):
            return None

        active_strategy = recovery_state.get("active_strategy")
        if not isinstance(active_strategy, dict):
            return None

        tool_name = str(
            active_strategy.get("suggested_tool_name")
            or active_strategy.get("tool_name")
            or ""
        ).strip()
        if not tool_name:
            return None

        patched_args = active_strategy.get("patched_args")
        if not isinstance(patched_args, dict):
            return None

        loop_count, loop_limit = tool_loop_budget(
            recent_tool_calls(messages, self.config.effective_tool_loop_window),
            tool_name=tool_name,
            tool_args=patched_args,
            readonly_tool_names=self.READ_ONLY_LOOP_TOLERANT_TOOL_NAMES,
            readonly_limit=self.config.effective_tool_loop_limit_readonly,
            mutating_limit=self.config.effective_tool_loop_limit_mutating,
        )
        if loop_count < loop_limit:
            return None

        return build_tool_issue(
            current_turn_id=current_turn_id,
            kind=str(open_tool_issue.get("kind") or "tool_error"),
            summary=(
                f"Recovery strategy for '{tool_name}' would repeat the same tool call that already hit the loop budget. "
                "Replanning without another LLM round."
            ),
            tool_names=[tool_name],
            tool_args=patched_args,
            source="recovery",
            error_type="LOOP_DETECTED",
            fingerprint=repair_fingerprint(tool_name, patched_args, "LOOP_DETECTED"),
            progress_fingerprint=repair_fingerprint(tool_name, patched_args, "LOOP_DETECTED"),
            details={
                "loop_detected": True,
                "preflight_blocked": True,
                "loop_count": loop_count,
                "loop_limit": loop_limit,
                "strategy_id": str(active_strategy.get("id") or "").strip(),
            },
        )
