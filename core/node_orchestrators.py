from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Protocol

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp

from core.errors import ErrorType, format_error
from core.message_utils import compact_text
from core.node_errors import EmptyLLMResponseError
from core.self_correction_engine import normalize_tool_args
from core.tool_args import canonicalize_tool_args, inspect_tool_args_payload
from core.tool_results import parse_tool_execution_result
from core.state import transcript_message_delta
from core.turn_outcomes import (
    TURN_OUTCOME_CONTINUE_AGENT,
    TURN_OUTCOME_FINISH_TURN,
    TURN_OUTCOME_RECOVER_AGENT,
    TURN_OUTCOME_RUN_TOOLS,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool

    from core.config import AgentConfig
    from core.context_builder import ContextBuilder
    from core.recovery_manager import RecoveryManager
    from core.state import AgentState, OpenToolIssue, RecoveryState
    from core.tool_executor import ToolExecutor
    from core.tool_policy import ToolMetadata


class NodeOrchestratorOwner(Protocol):
    def _log_run_event(self, state: AgentState | None, event_type: str, **payload: Any) -> None: ...

    def _log_node_start(self, state: AgentState | None, node: str, **payload: Any) -> float: ...

    def _log_node_end(
        self,
        state: AgentState | None,
        node: str,
        started_at: float,
        **payload: Any,
    ) -> None: ...

    def _current_turn_id(self, state: AgentState, messages: list[BaseMessage]) -> int: ...


class AgentTurnOwner(NodeOrchestratorOwner, Protocol):
    context_builder: ContextBuilder

    def _log_node_error(
        self,
        state: AgentState | None,
        node: str,
        started_at: float,
        error: Exception,
        **payload: Any,
    ) -> None: ...

    def _resolve_current_task(self, state: AgentState | None, messages: list[BaseMessage]) -> str: ...

    def _get_active_open_tool_issue(
        self,
        state: AgentState,
        messages: list[BaseMessage],
        current_turn_id: int | None = None,
    ) -> OpenToolIssue | None: ...

    def _get_recovery_state(self, state: AgentState, *, current_turn_id: int) -> RecoveryState: ...

    def _active_tools_for_turn(
        self,
        state: AgentState,
        messages: list[BaseMessage],
    ) -> tuple[list[BaseTool], list[str]]: ...

    def _select_llm_for_active_tools(
        self,
        active_tools: list[BaseTool],
        active_tool_names: list[str],
    ) -> BaseChatModel: ...

    def _current_turn_has_completed_user_choice(self, messages: list[BaseMessage]) -> bool: ...

    def _preflight_recovery_loop_issue(
        self,
        messages: list[Any],
        *,
        current_turn_id: int,
        open_tool_issue: OpenToolIssue | None,
        recovery_state: RecoveryState | None,
    ) -> OpenToolIssue | None: ...

    def _build_protocol_open_tool_issue(
        self,
        *,
        current_turn_id: int,
        summary: str,
        reason: str,
        source: str,
        tool_names: list[str] | None = None,
        tool_args: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        response_preview: str = "",
    ) -> dict[str, Any]: ...

    def _summarize_history_tool_mismatch(self, history_issue: dict[str, Any]) -> str: ...

    def _build_agent_context(
        self,
        messages: list[BaseMessage],
        summary: str,
        current_task: str,
        tools_available: bool,
        active_tool_names: list[str],
        open_tool_issue: OpenToolIssue | None,
        recovery_state: RecoveryState | None = None,
        state: AgentState | None = None,
        user_choice_locked: bool = False,
    ) -> list[BaseMessage]: ...

    def _assert_provider_safe_agent_context(
        self,
        context: list[BaseMessage],
        state: AgentState | None = None,
    ) -> None: ...

    async def _invoke_llm_with_retry(
        self,
        llm: BaseChatModel,
        context: list[Any],
        state: AgentState | None = None,
        node_name: str = "",
    ) -> AIMessage: ...

    def _build_agent_result(
        self,
        response: AIMessage,
        current_task: str,
        tools_available: bool,
        turn_id: int,
        messages: list[Any],
        open_tool_issue: OpenToolIssue | None = None,
        recovery_state: RecoveryState | None = None,
        allowed_tool_names: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def _normalize_system_prefix_for_provider(
        self,
        context: list[BaseMessage],
    ) -> list[BaseMessage]: ...


class RecoveryTurnOwner(NodeOrchestratorOwner, Protocol):
    config: AgentConfig
    recovery_manager: RecoveryManager

    def _resolve_current_task(self, state: AgentState | None, messages: list[BaseMessage]) -> str: ...

    def _get_active_open_tool_issue(
        self,
        state: AgentState,
        messages: list[BaseMessage],
        current_turn_id: int | None = None,
    ) -> OpenToolIssue | None: ...

    def _get_last_ai_message(self, messages: list[BaseMessage]) -> AIMessage | None: ...

    def _get_recovery_state(self, state: AgentState, *, current_turn_id: int) -> RecoveryState: ...

    def _hard_loop_ceiling(self) -> int: ...

    def _successful_tool_stagnation_limit(self, tool_name: str) -> int: ...


class ToolBatchOwner(NodeOrchestratorOwner, Protocol):
    config: AgentConfig
    recovery_manager: RecoveryManager
    tool_executor: ToolExecutor
    READ_ONLY_LOOP_TOLERANT_TOOL_NAMES: frozenset[str]
    _all_tool_names: tuple[str, ...]

    def _log_node_error(
        self,
        state: AgentState | None,
        node: str,
        started_at: float,
        error: Exception,
        **payload: Any,
    ) -> None: ...

    def _check_invariants(self, state: AgentState) -> None: ...

    def _get_last_pending_ai_with_tool_calls(self, messages: list[BaseMessage]) -> AIMessage | None: ...

    def _active_tools_for_turn(
        self,
        state: AgentState,
        messages: list[BaseMessage],
    ) -> tuple[list[BaseTool], list[str]]: ...

    def _partition_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...

    def _tool_call_is_parallel_safe(self, tool_call: dict[str, Any]) -> bool: ...

    async def _process_tool_call(
        self,
        tool_call: dict[str, Any],
        recent_calls: list[dict[str, Any]],
        state: AgentState,
        approval_state: dict[str, Any],
        current_turn_id: int,
        allowed_tool_names: list[str] | None = None,
    ) -> tuple[ToolMessage, bool, dict[str, Any] | None]: ...

    def _merge_open_tool_issues(
        self,
        issues: list[dict[str, Any]],
        current_turn_id: int,
    ) -> dict[str, Any] | None: ...

    def _effective_tool_metadata(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
    ) -> ToolMetadata: ...

    def _tool_is_allowed_for_turn(
        self,
        tool_name: str,
        allowed_tool_names: list[str] | None = None,
    ) -> bool: ...

    def _tool_requires_approval(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
    ) -> bool: ...

    def _tool_call_is_approved(
        self,
        tool_call_id: str,
        approval_state: dict[str, Any],
    ) -> bool: ...

    def _missing_required_tool_fields(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> list[str]: ...

    async def _execute_tool(
        self,
        name: str,
        args: dict[str, Any],
        state: AgentState | None = None,
        tool_call_id: str = "",
    ) -> str: ...


class AgentTurnOrchestrator:
    def __init__(self, owner: AgentTurnOwner) -> None:
        self.owner = owner

    async def run(self, state):
        owner = self.owner
        node_timer = owner._log_node_start(
            state,
            "agent",
            message_count=len(state.get("messages") or []),
            has_summary=bool(state.get("summary")),
        )
        messages = state["messages"]
        summary = state.get("summary", "")
        current_task = owner._resolve_current_task(state, messages)
        current_turn_id = owner._current_turn_id(state, messages)
        open_tool_issue = owner._get_active_open_tool_issue(state, messages, current_turn_id)
        recovery_state = owner._get_recovery_state(state, current_turn_id=current_turn_id)
        active_tools, active_tool_names = owner._active_tools_for_turn(
            state,
            messages,
        )
        llm_for_turn = owner._select_llm_for_active_tools(active_tools, active_tool_names)
        tools_available = bool(active_tool_names)
        user_choice_locked = owner._current_turn_has_completed_user_choice(messages)
        owner._log_run_event(
            state,
            "tool_context_selected",
            run_id=state.get("run_id", ""),
            active_tool_count=len(active_tool_names),
            user_choice_locked=user_choice_locked,
        )
        try:
            validation_handoff_reason = ""
            preflight_loop_issue = owner._preflight_recovery_loop_issue(
                messages,
                current_turn_id=current_turn_id,
                open_tool_issue=open_tool_issue,
                recovery_state=recovery_state,
            )
            if preflight_loop_issue:
                validation_handoff_reason = "recovery_strategy_loop_blocked"
                owner._log_run_event(
                    state,
                    "recovery_strategy_loop_blocked",
                    run_id=state.get("run_id", ""),
                    issue=preflight_loop_issue,
                )
                owner._log_node_end(
                    state,
                    "agent",
                    node_timer,
                    tool_calls=0,
                    tools_available=tools_available,
                    active_tool_count=len(active_tool_names),
                    validation_handoff_reason=validation_handoff_reason,
                    has_open_tool_issue=True,
                )
                return {
                    "current_task": current_task,
                    "turn_id": current_turn_id,
                    "turn_outcome": TURN_OUTCOME_RECOVER_AGENT,
                    "recovery_state": recovery_state,
                    "pending_approval": None,
                    "open_tool_issue": preflight_loop_issue,
                    "last_tool_error": str(preflight_loop_issue.get("summary") or ""),
                    "last_tool_result": "",
                }
            history_issue = owner.context_builder.detect_tool_history_mismatch(messages)
            if history_issue:
                validation_handoff_reason = "history_tool_mismatch"
                pending_tool_calls = history_issue.get("pending_tool_calls") or []
                first_pending_tool_call = pending_tool_calls[0] if pending_tool_calls else {}
                protocol_issue = owner._build_protocol_open_tool_issue(
                    current_turn_id=current_turn_id,
                    summary=owner._summarize_history_tool_mismatch(history_issue),
                    reason="history_tool_mismatch",
                    source="history",
                    tool_names=[
                        str(item.get("name") or "").strip()
                        for item in pending_tool_calls
                        if str(item.get("name") or "").strip()
                    ],
                    tool_args=canonicalize_tool_args(first_pending_tool_call.get("args")) if first_pending_tool_call else {},
                    details=history_issue,
                )
                owner._log_run_event(
                    state,
                    "history_tool_mismatch_detected",
                    run_id=state.get("run_id", ""),
                    issue=protocol_issue,
                )
                owner._log_node_end(
                    state,
                    "agent",
                    node_timer,
                    tool_calls=0,
                    tools_available=tools_available,
                    active_tool_count=len(active_tool_names),
                    validation_handoff_reason=validation_handoff_reason,
                    has_open_tool_issue=True,
                )
                return {
                    "current_task": current_task,
                    "turn_id": current_turn_id,
                    "turn_outcome": TURN_OUTCOME_RECOVER_AGENT,
                    "recovery_state": recovery_state,
                    "pending_approval": None,
                    "open_tool_issue": protocol_issue,
                    "last_tool_error": str(protocol_issue.get("summary") or ""),
                    "last_tool_result": "",
                }

            full_context = owner._build_agent_context(
                messages,
                summary,
                current_task,
                tools_available,
                active_tool_names,
                open_tool_issue,
                recovery_state,
                state=state,
                user_choice_locked=user_choice_locked,
            )
            owner._assert_provider_safe_agent_context(full_context, state)
            response = await owner._invoke_llm_with_retry(
                llm_for_turn,
                full_context,
                state=state,
                node_name="agent",
            )
            result = owner._build_agent_result(
                response,
                current_task,
                tools_available,
                current_turn_id,
                messages,
                open_tool_issue=open_tool_issue,
                recovery_state=recovery_state,
                allowed_tool_names=active_tool_names,
            )
            if result.pop("_retry_user_input_turn", False):
                owner._log_run_event(
                    state,
                    "user_input_reask_suppressed",
                    run_id=state.get("run_id", ""),
                    step=state.get("steps", 0),
                    current_task=current_task,
                )
                retry_context = owner._normalize_system_prefix_for_provider(
                    [
                        *full_context,
                        SystemMessage(
                            content=(
                                "USER INPUT ALREADY PROVIDED IN THIS TURN. "
                                "Do not call request_user_input again. "
                                "Use the latest request_user_input ToolMessage as the user's final choice and continue."
                            )
                        ),
                    ]
                )
                owner._assert_provider_safe_agent_context(retry_context, state)
                response = await owner._invoke_llm_with_retry(
                    llm_for_turn,
                    retry_context,
                    state=state,
                    node_name="agent_retry_after_user_choice",
                )
                result = owner._build_agent_result(
                    response,
                    current_task,
                    tools_available,
                    current_turn_id,
                    messages,
                    open_tool_issue=open_tool_issue,
                    recovery_state=recovery_state,
                    allowed_tool_names=active_tool_names,
                )
                result.pop("_retry_user_input_turn", None)
            result_issue = result.get("open_tool_issue")
            if (
                isinstance(result_issue, dict)
                and str(result_issue.get("kind") or "").strip().lower() == "protocol_error"
            ):
                validation_handoff_reason = str(
                    (result_issue.get("details") or {}).get("protocol_reason")
                    or "tool_protocol_error"
                )
                owner._log_run_event(
                    state,
                    "protocol_recovery_requested",
                    run_id=state.get("run_id", ""),
                    issue=result_issue,
                )
            tool_calls_count = len(getattr(response, "tool_calls", []) or [])
            owner._log_node_end(
                state,
                "agent",
                node_timer,
                tool_calls=tool_calls_count,
                tools_available=tools_available,
                active_tool_count=len(active_tool_names),
                validation_handoff_reason=validation_handoff_reason,
                has_open_tool_issue=bool(open_tool_issue),
            )
            return result
        except EmptyLLMResponseError as exc:
            owner._log_node_error(
                state,
                "agent",
                node_timer,
                exc,
                tools_available=tools_available,
                has_open_tool_issue=bool(open_tool_issue),
                handled=True,
            )
            empty_response_messages = [
                AIMessage(
                    content=(
                        "The model returned an empty response after repeated attempts. "
                        "I did not take any additional actions; please retry the request or clarify the wording."
                    )
                )
            ]
            return {
                "messages": empty_response_messages,
                "transcript_messages": transcript_message_delta(empty_response_messages),
                "current_task": current_task,
                "turn_id": current_turn_id,
                "turn_outcome": TURN_OUTCOME_FINISH_TURN,
                "recovery_state": recovery_state,
                "pending_approval": None,
                "open_tool_issue": None,
                "last_tool_error": str(exc),
                "last_tool_result": "",
            }
        except Exception as exc:
            owner._log_node_error(
                state,
                "agent",
                node_timer,
                exc,
                tools_available=tools_available,
                has_open_tool_issue=bool(open_tool_issue),
            )
            raise

class RecoveryTurnOrchestrator:
    def __init__(self, owner: RecoveryTurnOwner) -> None:
        self.owner = owner

    async def run(self, state):
        owner = self.owner
        node_timer = owner._log_node_start(
            state,
            "recovery",
            has_recovery_state=bool(state.get("recovery_state")),
            has_open_tool_issue=bool(state.get("open_tool_issue")),
        )
        messages = state.get("messages", [])
        current_turn_id = owner._current_turn_id(state, messages)
        current_task = owner._resolve_current_task(state, messages).strip()
        open_tool_issue = owner._get_active_open_tool_issue(state, messages, current_turn_id)
        last_ai = owner._get_last_ai_message(messages)
        last_message = messages[-1] if messages else None
        step_count = int(state.get("steps", 0) or 0)
        recovery_state = owner._get_recovery_state(state, current_turn_id=current_turn_id)
        # _hard_loop_ceiling() already returns 0 when self-correction is disabled,
        # so a single value drives both the stagnation ceiling and the repair budget.
        self_correction_limit = owner._hard_loop_ceiling()

        result = owner.recovery_manager.plan_recovery(
            state=state,
            messages=messages,
            current_task=current_task,
            current_turn_id=current_turn_id,
            open_tool_issue=open_tool_issue,
            recovery_state=recovery_state,
            last_ai=last_ai,
            last_message=last_message,
            step_count=step_count,
            max_loops=int(owner.config.max_loops or 0),
            hard_loop_ceiling=self_correction_limit,
            max_auto_repairs=self_correction_limit,
            successful_tool_stagnation_limit=owner._successful_tool_stagnation_limit(
                str(getattr(last_message, "name", "") or "")
            ),
        )

        outbound_messages: list[BaseMessage] = []
        if (
            result["drop_trailing_tool_call"]
            and last_ai
            and getattr(last_ai, "tool_calls", None)
            and getattr(last_ai, "id", None)
        ):
            outbound_messages.append(RemoveMessage(id=last_ai.id))
        if result["turn_outcome"] == TURN_OUTCOME_FINISH_TURN and result["handoff_message"]:
            handoff_kind = (
                "loop_budget_handoff"
                if str(result["completion_reason"]).startswith("loop_budget_exhausted")
                else "tool_issue_handoff"
            )
            outbound_messages.append(
                AIMessage(
                    content=result["handoff_message"],
                    additional_kwargs={
                        "agent_internal": {
                            "kind": handoff_kind,
                            "turn_id": current_turn_id,
                            "visible_in_ui": False,
                            "ui_notice": owner.recovery_manager.build_internal_ui_notice(
                                str(result["completion_reason"])
                            ),
                        }
                    },
                )
            )

        turn_outcome = TURN_OUTCOME_FINISH_TURN
        next_recovery_state = result["recovery_state"]
        if result["turn_outcome"] == TURN_OUTCOME_RECOVER_AGENT:
            turn_outcome = TURN_OUTCOME_RECOVER_AGENT
            active_strategy = next_recovery_state.get("active_strategy") if isinstance(next_recovery_state, dict) else {}
            owner._log_run_event(
                state,
                "recovery_prepared",
                run_id=state.get("run_id", ""),
                turn_id=current_turn_id,
                strategy_id=str((active_strategy or {}).get("id") or ""),
                strategy=str((active_strategy or {}).get("strategy") or ""),
                suggested_tool=str((active_strategy or {}).get("suggested_tool_name") or ""),
            )
        elif result["turn_outcome"] == TURN_OUTCOME_CONTINUE_AGENT:
            turn_outcome = TURN_OUTCOME_CONTINUE_AGENT

        owner._log_run_event(
            state,
            "recovery_verdict",
            run_id=state.get("run_id", ""),
            outcome=turn_outcome,
            reason=result["completion_reason"],
            retry_count=int(result["recovery_state"].get("retry_count", 0) or 0),
            has_open_tool_issue=bool(open_tool_issue),
            loop_budget_reached=result["loop_budget_reached"],
            had_pending_tool_calls=result["had_pending_tool_calls"],
        )
        owner._log_node_end(
            state,
            "recovery",
            node_timer,
            outcome=turn_outcome,
            reason=result["completion_reason"],
            turn_id=current_turn_id,
        )
        payload = {
            "turn_outcome": turn_outcome,
            "recovery_state": next_recovery_state,
            "open_tool_issue": result["open_tool_issue"],
            "last_tool_error": result.get("last_tool_error", state.get("last_tool_error", "")),
            "last_tool_result": result.get("last_tool_result", state.get("last_tool_result", "")),
        }
        if (
            turn_outcome == TURN_OUTCOME_FINISH_TURN
            and result["handoff_message"]
            and not str(result["completion_reason"]).startswith("loop_budget_exhausted")
        ):
            payload["steps"] = int(state.get("steps", 0) or 0) + 1
        if outbound_messages:
            payload["messages"] = outbound_messages
            payload["transcript_messages"] = transcript_message_delta(outbound_messages)
        return payload


class ToolBatchCoordinator:
    def __init__(self, owner: ToolBatchOwner) -> None:
        self.owner = owner

    async def run(self, state):
        owner = self.owner
        node_timer = owner._log_node_start(
            state,
            "tools",
            message_count=len(state.get("messages") or []),
            has_pending_approval=bool(state.get("pending_approval")),
        )
        owner._check_invariants(state)

        messages = state["messages"]
        last_msg = owner._get_last_pending_ai_with_tool_calls(messages)
        current_turn_id = owner._current_turn_id(state, messages)

        if not last_msg:
            owner._log_node_end(
                state,
                "tools",
                node_timer,
                outcome="skipped",
                reason="no_tool_calls",
            )
            return {}

        final_messages: list[ToolMessage] = []
        has_error = False
        last_error = ""
        last_result = ""
        tool_issues: list[dict[str, Any]] = []
        approval_state = state.get("pending_approval") or {}
        _, active_tool_names = owner._active_tools_for_turn(
            state,
            messages,
        )

        recent_calls = []
        history_window = owner.config.effective_tool_loop_window
        history_slice = messages[-(history_window + 1):-1] if history_window > 0 else messages[:-1]
        for message in reversed(history_slice):
            if isinstance(message, AIMessage) and message.tool_calls:
                recent_calls.extend(message.tool_calls)

        tool_calls = list(last_msg.tool_calls)

        parallel_calls, sequential_calls = owner._partition_tool_calls(tool_calls)
        max_parallel_calls = owner.config.max_parallel_tool_calls
        parallel_mode = (
            "sequential" if max_parallel_calls == 1
            else self._parallel_mode_label(parallel_calls, sequential_calls)
        )
        self._emit_live_tool_batch_started(tool_calls)
        owner._log_run_event(
            state,
            "tools_node_start",
            run_id=state.get("run_id", ""),
            tool_call_count=len(tool_calls),
            tool_names=[(tool_call.get("name") or "unknown_tool") for tool_call in tool_calls],
            parallel_mode=parallel_mode,
            parallel_count=len(parallel_calls),
            sequential_count=len(sequential_calls),
            max_parallel_tool_calls=max_parallel_calls,
        )
        try:
            # Results are collected positionally, then re-assembled in the
            # original tool_calls order so that the LLM receives ToolMessages
            # in the same sequence it emitted tool_calls.
            results: dict[int, tuple[ToolMessage, bool, dict[str, Any] | None]] = {}

            async def process_tool_call(
                tool_call: dict[str, Any],
                *,
                convert_exceptions: bool = False,
            ) -> tuple[ToolMessage, bool, dict[str, Any] | None]:
                try:
                    result = await owner._process_tool_call(
                        tool_call,
                        recent_calls,
                        state,
                        approval_state,
                        current_turn_id,
                        active_tool_names,
                    )
                except (asyncio.CancelledError, GraphBubbleUp):
                    raise
                except Exception as exc:
                    if not convert_exceptions:
                        raise
                    result = self._build_exception_result(
                        tool_call=tool_call,
                        exception=exc,
                        state=state,
                        current_turn_id=current_turn_id,
                    )
                self._emit_live_tool_result(result[0])
                return result

            async def run_parallel_group(group: list[tuple[int, dict[str, Any]]]) -> None:
                # Only create tasks for available slots, not one task per queued
                # call. Refill as soon as any tool finishes (no chunking barrier).
                remaining = iter(group)
                pending: dict[asyncio.Task, int] = {}

                def fill_slots() -> None:
                    while len(pending) < max_parallel_calls:
                        item = next(remaining, None)
                        if item is None:
                            break
                        index, tool_call = item
                        task = asyncio.create_task(process_tool_call(tool_call, convert_exceptions=True))
                        pending[task] = index

                try:
                    fill_slots()
                    while pending:
                        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                        # Propagate control exceptions before starting more work.
                        for task in done:
                            index = pending.pop(task)
                            results[index] = task.result()
                        fill_slots()
                finally:
                    # gather alone leaves siblings running when a child raises or
                    # cancels itself. Cancel and drain before leaving this group.
                    for task in pending:
                        if not task.done():
                            task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)

            pending_parallel: list[tuple[int, dict[str, Any]]] = []
            for index, tool_call in enumerate(tool_calls):
                if owner._tool_call_is_parallel_safe(tool_call):
                    pending_parallel.append((index, tool_call))
                    continue

                await run_parallel_group(pending_parallel)
                pending_parallel = []
                results[index] = await process_tool_call(tool_call)

            await run_parallel_group(pending_parallel)

            # Re-assemble in original order.
            for index in range(len(tool_calls)):
                tool_msg, had_error, issue = results[index]
                final_messages.append(tool_msg)
                has_error = has_error or had_error
                if issue:
                    tool_issues.append(issue)
                parsed = parse_tool_execution_result(tool_msg.content)
                if parsed.ok:
                    last_result = parsed.message
                else:
                    last_error = parsed.message
            merged_issue = owner._merge_open_tool_issues(tool_issues, current_turn_id)
            owner._log_run_event(
                state,
                "tools_node_end",
                run_id=state.get("run_id", ""),
                tool_result_count=len(final_messages),
                has_error=has_error,
                issue_kind="" if not merged_issue else merged_issue.get("kind", ""),
                issue_source="" if not merged_issue else merged_issue.get("source", ""),
            )
            owner._log_node_end(
                state,
                "tools",
                node_timer,
                tool_call_count=len(tool_calls),
                tool_result_count=len(final_messages),
                parallel_mode=parallel_mode,
                has_error=has_error,
                has_open_tool_issue=bool(merged_issue),
            )
            payload = {
                "messages": final_messages,
                "transcript_messages": transcript_message_delta(final_messages),
                "turn_id": current_turn_id,
                "turn_outcome": TURN_OUTCOME_RUN_TOOLS,
                "pending_approval": None,
                "open_tool_issue": merged_issue,
                "last_tool_error": last_error,
                "last_tool_result": last_result,
            }
            if not merged_issue:
                payload.update(
                    {
                        "recovery_state": owner.recovery_manager.reset_after_success(
                            state.get("recovery_state"),
                            current_turn_id=current_turn_id,
                            successful_evidence=last_result,
                        ),
                    }
                )
            return payload
        except (asyncio.CancelledError, GraphBubbleUp):
            # Graph control flow is not a tool execution failure.
            raise
        except Exception as exc:
            owner._log_node_error(
                state,
                "tools",
                node_timer,
                exc,
                tool_call_count=len(tool_calls),
                parallel_mode=parallel_mode,
            )
            raise

    @staticmethod
    def _emit_live_tool_batch_started(tool_calls: list[dict[str, Any]]) -> None:
        try:
            writer = get_stream_writer()
            writer(
                {
                    "type": "tool_batch_started",
                    "tool_calls": [
                        {
                            "id": str(tool_call.get("id") or ""),
                            "name": str(tool_call.get("name") or "unknown_tool"),
                            "args": canonicalize_tool_args(tool_call.get("args")),
                        }
                        for tool_call in tool_calls
                    ],
                }
            )
        except RuntimeError:
            return

    @staticmethod
    def _emit_live_tool_result(message: ToolMessage) -> None:
        try:
            writer = get_stream_writer()
            writer(
                {
                    "type": "tool_result",
                    "message": {
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                        "name": message.name,
                        "additional_kwargs": dict(message.additional_kwargs or {}),
                        "status": message.status,
                    },
                }
            )
        except RuntimeError:
            return

    @staticmethod
    def _parallel_mode_label(
        parallel_calls: list[dict[str, Any]],
        sequential_calls: list[dict[str, Any]],
    ) -> str:
        """Return a human-readable execution mode for logging.

        - ``"all"``       — every call is parallel-safe (bounded task pool)
        - ``"mixed"``     — some calls parallel, some sequential
        - ``"sequential"`` — no parallel-safe calls, all run one-by-one
        """
        if parallel_calls and not sequential_calls:
            return "all"
        if parallel_calls and sequential_calls:
            return "mixed"
        return "sequential"

    def _build_exception_result(
        self,
        *,
        tool_call: dict[str, Any],
        exception: Exception,
        state,
        current_turn_id: int,
    ) -> tuple[ToolMessage, bool, dict[str, Any] | None]:
        owner = self.owner
        tool_name = str(tool_call.get("name") or "unknown_tool")
        tool_args = canonicalize_tool_args(tool_call.get("args"))
        tool_call_id = str(tool_call.get("id") or "").strip() or f"call_missing_{uuid.uuid4().hex[:8]}"
        owner._log_run_event(
            state,
            "tool_call_parallel_exception_captured",
            run_id=state.get("run_id", ""),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            error_type=type(exception).__name__,
            error=compact_text(str(exception), 400),
        )
        outcome = owner.tool_executor.handle_result(
            state=state,
            current_turn_id=current_turn_id,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            content=format_error(
                ErrorType.EXECUTION,
                f"Unhandled exception while executing '{tool_name}': {exception}",
            ),
            issue_details={"parallel_batch_exception": type(exception).__name__},
            apply_validation=False,
            had_error=True,
        )
        return outcome.tool_message, outcome.had_error, outcome.issue

    async def process_tool_call(
        self,
        tool_call: dict[str, Any],
        recent_calls: list[dict[str, Any]],
        state,
        approval_state: dict[str, Any],
        current_turn_id: int,
        allowed_tool_names: list[str] | None = None,
    ):
        owner = self.owner
        tool_name = tool_call.get("name") or "unknown_tool"
        raw_tool_args = tool_call.get("args")
        tool_args, args_payload_kind = inspect_tool_args_payload(raw_tool_args)
        if args_payload_kind == "json_string":
            owner._log_run_event(
                state,
                "tool_call_args_canonicalized",
                run_id=state.get("run_id", ""),
                tool_name=tool_name,
                tool_call_id=str(tool_call.get("id") or ""),
                source_kind=args_payload_kind,
                arg_keys=sorted(tool_args.keys()),
                raw_preview=compact_text(str(raw_tool_args), 220),
            )
        elif args_payload_kind not in {"mapping", "missing", "empty_string"}:
            owner._log_run_event(
                state,
                "tool_call_args_unparsed",
                run_id=state.get("run_id", ""),
                tool_name=tool_name,
                tool_call_id=str(tool_call.get("id") or ""),
                source_kind=args_payload_kind,
                raw_preview=compact_text(str(raw_tool_args), 220),
            )
        normalized_args, normalized_changes = normalize_tool_args(
            tool_name,
            tool_args,
            current_task=str(state.get("current_task") or ""),
        )
        if normalized_changes:
            owner._log_run_event(
                state,
                "tool_call_args_repaired",
                run_id=state.get("run_id", ""),
                tool_name=tool_name,
                original_args=tool_args,
                patched_args=normalized_args,
                changes=normalized_changes,
            )
            tool_args = normalized_args

        tool_call_id = tool_call.get("id")
        if not tool_call_id:
            tool_call_id = f"call_missing_{uuid.uuid4().hex[:8]}"

        had_error = False
        tool_duration_seconds: float | None = None
        metadata = owner._effective_tool_metadata(tool_name, tool_args)
        active_tool_names = (
            list(allowed_tool_names)
            if allowed_tool_names is not None
            else (list(owner._all_tool_names) if owner.config.model_supports_tools else [])
        )

        if not owner._tool_is_allowed_for_turn(tool_name, allowed_tool_names):
            outcome = owner.tool_executor.build_not_allowed_result(
                state=state,
                current_turn_id=current_turn_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
                allowed_tool_names=active_tool_names,
            )
            return outcome.tool_message, outcome.had_error, outcome.issue

        if owner._tool_requires_approval(tool_name, tool_args) and not owner._tool_call_is_approved(
            tool_call_id, approval_state
        ):
            outcome = owner.tool_executor.build_denied_result(
                state=state,
                current_turn_id=current_turn_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
                policy=metadata.to_dict(),
            )
            return outcome.tool_message, outcome.had_error, outcome.issue

        missing_required = owner._missing_required_tool_fields(tool_name, tool_args)
        if missing_required:
            outcome = owner.tool_executor.build_missing_required_result(
                state=state,
                current_turn_id=current_turn_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
                missing_required=missing_required,
            )
            return outcome.tool_message, outcome.had_error, outcome.issue

        loop_count = sum(
            1
            for recent_call in recent_calls
            if recent_call.get("name") == tool_name and canonicalize_tool_args(recent_call.get("args")) == tool_args
        )
        loop_limit = (
            owner.config.effective_tool_loop_limit_readonly
            if tool_name in owner.READ_ONLY_LOOP_TOLERANT_TOOL_NAMES
            else owner.config.effective_tool_loop_limit_mutating
        )
        if loop_count >= loop_limit:
            content = format_error(
                ErrorType.LOOP_DETECTED,
                f"Loop detected. You have called '{tool_name}' with these exact arguments {loop_limit} times in the recent history. Please try a different approach.",
            )
            had_error = True
            owner._log_run_event(
                state,
                "tool_call_loop_blocked",
                run_id=state.get("run_id", ""),
                tool_name=tool_name,
                tool_args=tool_args,
                loop_count=loop_count,
                loop_limit=loop_limit,
            )
        else:
            owner._log_run_event(
                state,
                "tool_call_start",
                run_id=state.get("run_id", ""),
                tool_name=tool_name,
                tool_args=tool_args,
                policy=metadata.to_dict(),
            )
            started_at = time.perf_counter()
            content = await owner._execute_tool(tool_name, tool_args, state=state, tool_call_id=tool_call_id)
            tool_duration_seconds = max(0.0, time.perf_counter() - started_at)

        outcome = owner.tool_executor.handle_result(
            state=state,
            current_turn_id=current_turn_id,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            content=content,
            tool_duration_seconds=tool_duration_seconds,
            had_error=had_error,
            issue_details=(
                {
                    "loop_detected": True,
                    "loop_count": loop_count,
                    "loop_limit": loop_limit,
                }
                if loop_count >= loop_limit
                else None
            ),
        )
        return outcome.tool_message, outcome.had_error, outcome.issue
