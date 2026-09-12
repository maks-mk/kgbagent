from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from core.state import AgentState
from core.errors import format_error, ErrorType
from core.tool_results import parse_tool_execution_result
from core.tool_executor import ToolExecutor


class ToolsMixin:
    """Tools node: dispatches tool execution via ToolBatchCoordinator."""

    async def tools_node(self, state: AgentState):
        return await self.tool_batch.run(state)

    def _tool_call_is_parallel_safe(self, tool_call: Dict[str, Any]) -> bool:
        """Allow registered read-only tools and the existing shell exception to overlap."""
        name = tool_call.get("name") or "unknown_tool"
        if name not in self._all_tool_names or name == "request_user_input":
            return False
        if name == "cli_exec":
            return True
        if name not in self.PARALLEL_SAFE_TOOL_NAMES and name not in self.tool_metadata:
            return False
        return self._tool_is_read_only(name)

    def _partition_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split tool calls into (parallel_safe, sequential) groups.

        Each call is independently classified.  The original ordering within
        each group is preserved for diagnostics. Execution uses contiguous
        parallel-safe groups separated by sequential barriers, so reads cannot
        overtake writes. Registered read-only tools and cli_exec may overlap.
        """
        parallel: List[Dict[str, Any]] = []
        sequential: List[Dict[str, Any]] = []
        for tc in tool_calls:
            if self._tool_call_is_parallel_safe(tc):
                parallel.append(tc)
            else:
                sequential.append(tc)
        return parallel, sequential

    def _tool_is_allowed_for_turn(
        self,
        tool_name: str,
        allowed_tool_names: List[str] | None = None,
    ) -> bool:
        if allowed_tool_names is not None:
            return self._normalize_tool_name(tool_name) in {
                self._normalize_tool_name(name) for name in allowed_tool_names
            }
        return (
            bool(self.config.model_supports_tools)
            and bool(self._all_tool_names)
            and self._normalize_tool_name(tool_name) in {
                self._normalize_tool_name(name) for name in self._all_tool_names
            }
        )

    async def _process_tool_call(
        self,
        tool_call: Dict[str, Any],
        recent_calls: List[Dict[str, Any]],
        state: AgentState,
        approval_state: Dict[str, Any],
        current_turn_id: int,
        allowed_tool_names: List[str] | None = None,
    ) -> Tuple[ToolMessage, bool, Optional[Dict[str, Any]]]:
        return await self.tool_batch.process_tool_call(
            tool_call,
            recent_calls,
            state,
            approval_state,
            current_turn_id,
            allowed_tool_names,
        )

    def _tool_call_is_approved(self, tool_call_id: str, approval_state: Dict[str, Any]) -> bool:
        if not self.config.enable_approvals:
            return True
        if not approval_state:
            return False
        if not approval_state.get("approved"):
            return False
        approved_ids = set(approval_state.get("tool_call_ids") or [])
        return not approved_ids or tool_call_id in approved_ids

    async def _execute_tool(
        self,
        name: str,
        args: dict,
        state: AgentState | None = None,
        tool_call_id: str = "",
    ) -> str:
        # Fast O(1) lookup
        tool = self.tools_map.get(name)
        if not tool:
            self._log_run_event(
                state,
                "tool_call_missing",
                run_id=None if state is None else state.get("run_id", ""),
                tool_name=name,
                tool_args=args,
            )
            return format_error(ErrorType.NOT_FOUND, f"Tool '{name}' not found.")
        try:
            invoke_scope = nullcontext()
            if name == "cli_exec" and tool_call_id:
                try:
                    from tools.local_shell import cli_output_context

                    invoke_scope = cli_output_context(tool_call_id)
                except Exception:
                    invoke_scope = nullcontext()

            with invoke_scope:
                raw_result = await tool.ainvoke(args)
            if isinstance(raw_result, (dict, list)):
                try:
                    content = json.dumps(raw_result, ensure_ascii=False)
                except (TypeError, ValueError):
                    content = str(raw_result)
            else:
                content = str(raw_result)
            if not content.strip():
                self._log_run_event(
                    state,
                    "tool_call_empty_result",
                    run_id=None if state is None else state.get("run_id", ""),
                    tool_name=name,
                    tool_args=args,
                )
                return format_error(ErrorType.EXECUTION, "Tool returned empty response.")
            return content
        except asyncio.CancelledError:
            self._log_run_event(
                state,
                "tool_call_cancelled",
                run_id=None if state is None else state.get("run_id", ""),
                tool_name=name,
                tool_args=args,
                tool_call_id=tool_call_id,
                reason="task_cancelled",
            )
            raise
        except GraphBubbleUp:
            raise
        except Exception as e:
            self._log_run_event(
                state,
                "tool_call_exception",
                run_id=None if state is None else state.get("run_id", ""),
                tool_name=name,
                tool_args=args,
                error_type=type(e).__name__,
                error=str(e)[:400],
            )
            return format_error(ErrorType.EXECUTION, str(e))
