import logging
from typing import Any, Callable

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from core.tool_args import canonicalize_tool_args

logger = logging.getLogger("agent")
HANDOFF_MARKERS_SKIP_REPAIR = frozenset({"loop_budget_handoff"})
INCOMPLETE_TOOL_CALL_REPAIR_MESSAGE = (
    "ERROR[NETWORK]: The provider or aggregator stream ended before this tool returned a result. "
    "This is usually a transient upstream disconnect, not a user stop. Please retry or continue."
)


async def repair_session_if_needed(
    agent_app: Any,
    thread_id: str,
    notifier: Callable[[str], None] | None = None,
    event_logger: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[str]:
    notices: list[str] = []

    def _notify(message: str) -> None:
        notices.append(message)
        if notifier:
            notifier(message)

    try:
        config = {"configurable": {"thread_id": thread_id}}
        async_get_state = getattr(agent_app, "aget_state", None)
        if callable(async_get_state):
            current_state = await async_get_state(config)
        else:
            current_state = agent_app.get_state(config)

        if not current_state or not current_state.values:
            return notices

        messages = current_state.values.get("messages", [])
        if not messages:
            return notices

        pending_tool_calls: dict[str, dict[str, Any]] = {}
        pending_order: list[str] = []
        pending_segment_start_idx: int | None = None

        for index, message in enumerate(messages):
            if isinstance(message, (AIMessage, AIMessageChunk)) and getattr(message, "tool_calls", None):
                if pending_segment_start_idx is None:
                    pending_segment_start_idx = index
                for tool_call in message.tool_calls:
                    tool_call_id = str(tool_call.get("id") or "").strip()
                    tool_name = str(tool_call.get("name") or "").strip()
                    if not tool_call_id or not tool_name:
                        logger.warning(
                            "Session repair skipped malformed tool call without id/name in thread %s",
                            thread_id,
                        )
                        continue
                    pending_tool_calls[tool_call_id] = {
                        "id": tool_call_id,
                        "name": tool_name,
                        "args": canonicalize_tool_args(tool_call.get("args")),
                    }
                    if tool_call_id not in pending_order:
                        pending_order.append(tool_call_id)
                continue

            if not isinstance(message, ToolMessage):
                continue

            tool_call_id = str(message.tool_call_id or "").strip()
            if not tool_call_id:
                continue
            pending_tool_calls.pop(tool_call_id, None)
            pending_order = [item for item in pending_order if item != tool_call_id]
            if not pending_tool_calls:
                pending_segment_start_idx = None

        missing_tool_calls = [
            pending_tool_calls[tool_call_id]
            for tool_call_id in pending_order
            if tool_call_id in pending_tool_calls
        ]

        if not missing_tool_calls:
            return notices

        # If the run already produced an explicit internal handoff after the
        # dangling tool-call message, do not inject synthetic tool outputs.
        handoff_after_tool_call = False
        repair_window_start = (pending_segment_start_idx + 1) if pending_segment_start_idx is not None else 0
        for index in range(repair_window_start, len(messages)):
            message = messages[index]
            if not isinstance(message, (AIMessage, AIMessageChunk)):
                continue
            metadata = getattr(message, "additional_kwargs", {}) or {}
            internal = metadata.get("agent_internal") if isinstance(metadata, dict) else None
            if isinstance(internal, dict) and str(internal.get("kind") or "") in HANDOFF_MARKERS_SKIP_REPAIR:
                handoff_after_tool_call = True
                break
        if handoff_after_tool_call:
            return notices

        _notify(
            f"Detected {len(missing_tool_calls)} incomplete tool execution(s) after a stream interruption. "
            "Repairing history automatically."
        )
        tool_messages = [
            ToolMessage(
                tool_call_id=tool_call["id"],
                content=INCOMPLETE_TOOL_CALL_REPAIR_MESSAGE,
                name=tool_call["name"],
                additional_kwargs={
                    "tool_args": canonicalize_tool_args(tool_call.get("args")),
                    "agent_internal": {
                        "kind": "repaired_interrupted_tool_call",
                        "visible_in_ui": False,
                        "ui_notice": (
                            "Provider stream interrupted while a tool was running. "
                            "History was repaired automatically."
                        ),
                    },
                },
                status="error",
            )
            for tool_call in missing_tool_calls
        ]

        if event_logger:
            for tool_call in missing_tool_calls:
                event_logger(
                    "tool_repair_inserted",
                    {
                        "tool_name": str(tool_call.get("name") or ""),
                        "tool_call_id": str(tool_call.get("id") or ""),
                        "tool_args": canonicalize_tool_args(tool_call.get("args")),
                        "reason": "missing_tool_output_after_stream_interruption",
                    },
                )

        async_update_state = getattr(agent_app, "aupdate_state", None)
        update = {"messages": tool_messages, "transcript_messages": tool_messages}
        if callable(async_update_state):
            await async_update_state(config, update, as_node="tools")
        else:
            agent_app.update_state(config, update, as_node="tools")

        _notify("History repaired. The restored session is ready for a new request.")
    except Exception as exc:
        logger.warning("Session repair skipped due to error: %s", exc)

    return notices
