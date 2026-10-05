from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, RemoveMessage, ToolMessage

from core.state import AgentState
from core.message_utils import (
    iter_message_tool_calls,
    compact_text,
    is_internal_retry_message,
    is_user_turn_message,
    stringify_content,
)

logger = logging.getLogger("agent")


class BaseMixin:
    """Base logging, state introspection, and common helpers."""

    def _log_run_event(self, state: AgentState | None, event_type: str, **payload: Any) -> None:
        if not self.run_logger:
            return
        # Guard against accidental duplicate keyword with the positional session_id
        # argument of JsonlRunLogger.log_event(...).
        payload.pop("session_id", None)
        session_id = None if state is None else state.get("session_id")
        self.run_logger.log_event(session_id, event_type, **payload)

    def _state_log_context(self, state: AgentState | None) -> Dict[str, Any]:
        if not state:
            return {}
        return {
            "run_id": state.get("run_id", ""),
            "step": state.get("steps", 0),
            "turn_id": state.get("turn_id", 0),
            "state_session_id": state.get("session_id", ""),
        }

    def _log_node_start(self, state: AgentState | None, node: str, **payload: Any) -> float:
        event_payload: Dict[str, Any] = {"node": node, **self._state_log_context(state)}
        event_payload.update(payload)
        self._log_run_event(state, "node_start", **event_payload)
        return time.perf_counter()

    def _log_node_end(self, state: AgentState | None, node: str, started_at: float, **payload: Any) -> None:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        event_payload: Dict[str, Any] = {
            "node": node,
            "duration_ms": duration_ms,
            **self._state_log_context(state),
        }
        event_payload.update(payload)
        self._log_run_event(state, "node_end", **event_payload)

    def _log_node_error(
        self,
        state: AgentState | None,
        node: str,
        started_at: float,
        error: Exception,
        **payload: Any,
    ) -> None:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        event_payload: Dict[str, Any] = {
            "node": node,
            "duration_ms": duration_ms,
            "error_type": type(error).__name__,
            "error": compact_text(str(error), 400),
            **self._state_log_context(state),
        }
        event_payload.update(payload)
        self._log_run_event(state, "node_error", **event_payload)

    def _is_internal_retry_message(self, message: BaseMessage) -> bool:
        return is_internal_retry_message(message)

    def _current_turn_id(self, state: AgentState, messages: List[BaseMessage]) -> int:
        derived_turn_id = 0
        for message in messages:
            if is_user_turn_message(message):
                derived_turn_id += 1
        return max(int(state.get("turn_id", 0) or 0), derived_turn_id)

    def _get_active_open_tool_issue(
        self,
        state: AgentState,
        messages: List[BaseMessage],
        current_turn_id: int | None = None,
    ) -> Dict[str, Any] | None:
        issue = state.get("open_tool_issue")
        if not isinstance(issue, dict):
            return None

        active_turn_id = current_turn_id if current_turn_id is not None else self._current_turn_id(state, messages)
        issue_turn_id = int(issue.get("turn_id", 0) or 0)
        summary = str(issue.get("summary", "")).strip()
        if issue_turn_id != active_turn_id or not summary:
            return None

        tool_names = [str(name) for name in (issue.get("tool_names") or []) if str(name).strip()]
        return {
            "turn_id": issue_turn_id,
            "kind": str(issue.get("kind") or "tool_error"),
            "summary": summary,
            "tool_names": tool_names,
            "tool_args": dict(issue.get("tool_args") or {}),
            "source": str(issue.get("source") or "tools"),
            "error_type": str(issue.get("error_type") or ""),
            "fingerprint": str(issue.get("fingerprint") or ""),
            "progress_fingerprint": str(issue.get("progress_fingerprint") or issue.get("fingerprint") or ""),
            "details": dict(issue.get("details") or {}),
        }

    def _get_recovery_state(self, state: AgentState, *, current_turn_id: int) -> Dict[str, Any]:
        return self.recovery_manager.get_recovery_state(
            state.get("recovery_state"),
            current_turn_id=current_turn_id,
            legacy_state=state,
        )

    def _collect_internal_retry_removals(self, messages: List[BaseMessage]) -> List[RemoveMessage]:
        removals: List[RemoveMessage] = []
        for message in reversed(messages):
            if not self._is_internal_retry_message(message):
                break
            if message.id:
                removals.append(RemoveMessage(id=message.id))
        removals.reverse()
        return removals

    def _derive_current_task(self, messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if is_user_turn_message(message):
                return stringify_content(message.content).strip()
        return ""

    def _is_low_information_follow_up(self, content: str) -> bool:
        normalized = " ".join(str(content or "").strip().lower().split()).strip(" .,!?:;…")
        return normalized in {
            "continue",
            "continue please",
            "go on",
            "keep going",
            "proceed",
            "продолжай",
            "продолжай пожалуйста",
            "дальше",
            "далее",
            "продолжи",
            "продолжи пожалуйста",
        }

    def _derive_specific_task(self, messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if is_user_turn_message(message):
                content = stringify_content(message.content).strip()
                if not self._is_low_information_follow_up(content):
                    return content
        return ""

    def _resolve_current_task(self, state: AgentState | None, messages: List[BaseMessage]) -> str:
        derived_task = self._derive_current_task(messages).strip()
        specific_task = self._derive_specific_task(messages).strip()
        stored_task = str((state or {}).get("current_task") or "").strip()
        if derived_task and derived_task == specific_task:
            if stored_task and stored_task != derived_task:
                self._log_run_event(
                    state,
                    "current_task_overridden_by_latest_user_message",
                    run_id=None if state is None else state.get("run_id", ""),
                    previous_task=compact_text(stored_task, 180),
                    latest_user_message=compact_text(derived_task, 180),
                )
            return derived_task
        if stored_task:
            if derived_task and derived_task != stored_task and self._is_low_information_follow_up(derived_task):
                self._log_run_event(
                    state,
                    "current_task_preserved_from_state",
                    run_id=None if state is None else state.get("run_id", ""),
                    preserved_task=compact_text(stored_task, 180),
                    latest_user_message=compact_text(derived_task, 180),
                )
            return stored_task
        if specific_task:
            return specific_task
        return stored_task

    def _active_tools_for_turn(
        self,
        state: AgentState,
        messages: List[BaseMessage],
    ) -> Tuple[List[Any], List[str]]:
        if state.get("chat_only", False) or not self.config.model_supports_tools:
            return [], []

        active_tools = list(
            self.active_tools_provider() if self.active_tools_provider is not None else self.tools
        )

        if self._current_turn_has_completed_user_choice(messages):
            active_tools = [
                tool
                for tool in active_tools
                if self._normalize_tool_name(tool.name) != "request_user_input"
            ]

        return active_tools, [tool.name for tool in active_tools]

    def _current_turn_has_completed_user_choice(self, messages: List[BaseMessage]) -> bool:
        human_indexes = self.message_context.non_internal_human_indexes(
            messages,
            self._is_internal_retry_message,
        )
        if not human_indexes:
            return False

        start_idx = human_indexes[-1] + 1
        for message in messages[start_idx:]:
            if isinstance(message, ToolMessage):
                if self._normalize_tool_name(message.name or "") == "request_user_input":
                    return True
        return False

    def _current_turn_has_user_input_request(self, messages: List[BaseMessage]) -> bool:
        human_indexes = self.message_context.non_internal_human_indexes(
            messages,
            self._is_internal_retry_message,
        )
        if not human_indexes:
            return False

        start_idx = human_indexes[-1] + 1
        for message in messages[start_idx:]:
            if isinstance(message, ToolMessage):
                if self._normalize_tool_name(message.name or "") == "request_user_input":
                    return True
                continue
            if isinstance(message, (AIMessage, AIMessageChunk)):
                for tool_call in iter_message_tool_calls(message):
                    if self._normalize_tool_name(tool_call.get("name") or "") == "request_user_input":
                        return True
        return False

    def _successful_tool_stagnation_limit(self, tool_name: str) -> int:
        return 3 if self._tool_is_read_only(tool_name) else 2

    def _get_last_ai_message(self, messages: List[BaseMessage]) -> Optional[AIMessage]:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    def _get_last_pending_ai_with_tool_calls(self, messages: List[BaseMessage]) -> Optional[AIMessage]:
        """
        Returns the latest AI message that still has at least one unresolved tool call.
        This is robust to provider id collisions where the newest AI message may replace
        an older message and stop being the last list item.
        """
        resolved_tool_call_ids: set[str] = {
            str(message.tool_call_id or "").strip()
            for message in messages
            if isinstance(message, ToolMessage) and str(message.tool_call_id or "").strip()
        }

        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            tool_calls = list(iter_message_tool_calls(message))
            if not tool_calls:
                continue
            unresolved_calls = [
                call for call in tool_calls
                if not str(call.get("id") or "").strip()
                or str(call.get("id") or "").strip() not in resolved_tool_call_ids
            ]
            if unresolved_calls:
                return message
        return None

    def _get_base_prompt(self) -> str:
        """Lazily load and cache the prompt to avoid repeated disk I/O."""
        if self._cached_base_prompt is None:
            prompt_path = self.config.prompt_path.absolute()
            if self.config.prompt_path.exists():
                try:
                    self._cached_base_prompt = self.config.prompt_path.read_text("utf-8")
                    logger.info("Loaded prompt from file: %s", prompt_path)
                except Exception as e:
                    logger.warning(
                        "Failed to read prompt file %s: %s. Using built-in prompt.",
                        prompt_path,
                        e,
                    )
                    self._cached_base_prompt = (
                        "You are an autonomous AI agent.\n"
                        "Reason in English, Reply in Russian.\n"
                        "Date: {{current_date}}"
                    )
            else:
                logger.info("Prompt file not found at %s. Using built-in prompt.", prompt_path)
                self._cached_base_prompt = (
                    "You are an autonomous AI agent.\n"
                    "Reason in English, Reply in Russian.\n"
                    "Date: {{current_date}}"
                )
        return self._cached_base_prompt

    def _check_invariants(self, state: AgentState):
        if not self.config.debug:
            return
        steps = state.get("steps", 0)
        if steps < 0:
            logger.error(f"INVARIANT VIOLATION: steps ({steps}) < 0")
