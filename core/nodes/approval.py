from __future__ import annotations

from typing import Any, Dict, List, Optional

from langgraph.types import interrupt

from core.state import AgentState


class ApprovalMixin:
    """Approval node: asks the user to approve mutating/destructive tool calls."""

    async def approval_node(self, state: AgentState):
        node_timer = self._log_node_start(
            state,
            "approval",
            message_count=len(state.get("messages") or []),
        )
        messages = state.get("messages", [])
        if not messages:
            self._log_node_end(
                state,
                "approval",
                node_timer,
                outcome="skipped",
                reason="no_messages",
            )
            return {"pending_approval": None}

        pending_ai_with_tools = self._get_last_pending_ai_with_tool_calls(messages)
        if not pending_ai_with_tools:
            self._log_node_end(
                state,
                "approval",
                node_timer,
                outcome="skipped",
                reason="no_protected_tool_calls",
            )
            return {"pending_approval": None}

        protected_calls = []
        current_task = str(state.get("current_task") or "")
        for tool_call in pending_ai_with_tools.tool_calls:
            tool_name = tool_call.get("name") or "unknown_tool"
            if not self._tool_call_requires_ready_approval(
                tool_name,
                tool_call.get("args"),
                current_task=current_task,
            ):
                continue
            tool_args = self._normalize_tool_args_for_preflight(
                tool_name,
                tool_call.get("args"),
                current_task=current_task,
            )
            metadata = self._effective_tool_metadata(tool_name, tool_args)
            protected_calls.append(
                {
                    "id": tool_call.get("id") or "",
                    "name": tool_name,
                    "args": tool_args,
                    "policy": metadata.to_dict(),
                }
            )

        if not protected_calls:
            self._log_node_end(
                state,
                "approval",
                node_timer,
                outcome="skipped",
                reason="all_tools_readonly",
            )
            return {"pending_approval": None}

        payload = {
            "kind": "tool_approval",
            "message": "Approve protected tool execution?",
            "tools": protected_calls,
            "run_id": state.get("run_id", ""),
            "session_id": state.get("session_id", ""),
        }
        self._log_run_event(
            state,
            "approval_requested",
            run_id=state.get("run_id", ""),
            tool_names=[tool["name"] for tool in protected_calls],
        )
        decision = interrupt(payload)
        approved = self._approval_decision_is_approved(decision)
        approval_state = {
            "approved": approved,
            "decision": decision,
            "tool_call_ids": [tool["id"] for tool in protected_calls if tool["id"]],
            "tool_names": [tool["name"] for tool in protected_calls],
        }
        self._log_run_event(
            state,
            "approval_resolved",
            run_id=state.get("run_id", ""),
            approved=approved,
            tool_names=approval_state["tool_names"],
        )
        self._log_node_end(
            state,
            "approval",
            node_timer,
            outcome="resolved",
            approved=approved,
            protected_count=len(protected_calls),
        )
        return {"pending_approval": approval_state}

    def _approval_decision_is_approved(self, decision: Any) -> bool:
        if isinstance(decision, bool):
            return decision
        if isinstance(decision, dict):
            if "approved" in decision:
                return decision["approved"] is True
            action = str(decision.get("action", "")).strip().lower()
            return action in {"approve", "approved", "yes", "y"}
        # Неизвестный формат (строка, число, None и т.п.) — отказ.
        # Для approval дефолт должен быть безопасным; угадывание интента
        # свободного текста недопустимо (bool("no") == True и т.п.).
        return False
