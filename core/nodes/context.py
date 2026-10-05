from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from core.state import AgentState, OpenToolIssue, RecoveryState
from core.node_errors import ProviderContextError

logger = logging.getLogger("agent")


class ContextMixin:
    """Context building and provider-specific normalization helpers."""

    def _build_agent_context(
        self,
        messages: List[BaseMessage],
        summary: str,
        current_task: str,
        tools_available: bool,
        active_tool_names: List[str],
        open_tool_issue: OpenToolIssue | None,
        recovery_state: RecoveryState | None = None,
        state: AgentState | None = None,
        user_choice_locked: bool = False,
    ) -> List[BaseMessage]:
        return self.context_builder.build(
            messages,
            state,
            summary=summary,
            current_task=current_task,
            tools_available=tools_available,
            active_tool_names=active_tool_names,
            open_tool_issue=open_tool_issue,
            recovery_state=recovery_state,
            user_choice_locked=user_choice_locked,
            mcp_tool_groups=self._mcp_tool_groups_for_names(active_tool_names),
        )

    def _mcp_tool_groups_for_names(self, active_tool_names: List[str]) -> tuple:
        """Resolve server -> tools mapping so the prompt can list the MCP tool names."""
        provider = getattr(self, "mcp_tool_groups_provider", None)
        if provider is None or not active_tool_names:
            return ()
        try:
            return tuple(provider(list(active_tool_names)))
        except Exception:
            logger.debug("Failed to resolve MCP tool groups for the prompt", exc_info=True)
            return ()

    def _normalize_system_prefix_for_provider(
        self,
        context: List[BaseMessage],
    ) -> List[BaseMessage]:
        return self.context_builder.normalize_system_prefix(context)

    def _message_role_for_provider(self, message: BaseMessage) -> str:
        return self.context_builder._message_role_for_provider(message)

    def _normalize_tool_call_id_for_provider(self, tool_call_id: str, *, used_ids: set[str]) -> str:
        return self.context_builder._normalize_tool_call_id_for_provider(tool_call_id, used_ids=used_ids)

    def _sanitize_messages_for_model(
        self,
        messages: List[BaseMessage],
        state: AgentState | None = None,
    ) -> List[BaseMessage]:
        return self.context_builder.sanitize_messages(messages, state=state)

    def _assert_provider_safe_agent_context(
        self,
        context: List[BaseMessage],
        state: AgentState | None = None,
    ) -> None:
        try:
            self.context_builder.assert_provider_safe_context(context, state=state)
        except RuntimeError as exc:
            raise ProviderContextError(str(exc)) from exc
