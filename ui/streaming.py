import asyncio
import difflib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

from langchain_core.messages import AIMessage, AIMessageChunk, RemoveMessage, ToolMessage

from core.api_key_rotation import describe_provider_error
from core.message_utils import is_tool_message_error, stringify_content
from core.text_utils import (
    TokenTracker,
    build_tool_ui_labels,
    build_mcp_tool_ui_labels,
    classify_tool_args_state,
    format_elapsed_seconds,
    format_tool_display,
    format_tool_output,
    has_visible_text,
)
from core.reasoning_debug import debug_event, elapsed_since, log_unknown_fields, now, preview_value
from core.tool_args import canonicalize_tool_args
from ui.tool_message_utils import extract_tool_args, extract_tool_duration
from ui.visibility import get_internal_ui_notice, is_hidden_internal_message

DIFF_REGEX = re.compile(r"```diff\r?\n(.*?)```", re.DOTALL)
_INLINE_THOUGHT_BLOCK_RE = re.compile(r"<(think|thought)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_INLINE_THOUGHT_OPEN_RE = re.compile(r"<(think|thought)\b[^>]*>", re.IGNORECASE)
_INLINE_THOUGHT_CLOSE_PREFIX_RE = re.compile(r"^.*?</(think|thought)>\s*", re.IGNORECASE | re.DOTALL)
_INLINE_THOUGHT_UNCLOSED_RE = re.compile(r"<(think|thought)\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_TEXT_TOOL_CALL_START_RE = re.compile(r"call:[A-Za-z_][\w.-]*\s*\{", re.DOTALL)
_TEXT_TOOL_CALL_END = "<tool_call|>"
_AGENT_WORKING_LABEL = "Working..."
_AGENT_THINKING_LABEL = "Thinking..."
logger = logging.getLogger("agent")
reasoning_logger = logging.getLogger("agent.reasoning_debug")


INTERRUPTED_TOOL_MESSAGES = {
    "cancelled": "ERROR[CANCELLED]: The run was stopped before this tool returned a result.",
    "stream_error": (
        "ERROR[NETWORK]: The provider or aggregator stream ended before this tool returned a result. "
        "This is usually a transient upstream disconnect, not a user stop. Please retry or continue."
    ),
}
DEFAULT_INTERRUPTED_TOOL_MESSAGE = (
    "ERROR[INTERRUPTED]: The run ended before this tool returned a result. Please retry or continue."
)


def _strip_inline_thought_content(text: str) -> str:
    cleaned = _INLINE_THOUGHT_BLOCK_RE.sub("", text)
    cleaned = _INLINE_THOUGHT_CLOSE_PREFIX_RE.sub("", cleaned)
    return _INLINE_THOUGHT_UNCLOSED_RE.sub("", cleaned)


def _clip_debug_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…(+{len(text) - limit} chars)"


def _describe_thinking_signal(content: Any) -> str:
    if isinstance(content, str):
        if _INLINE_THOUGHT_OPEN_RE.search(content):
            return "inline_thought_tag"
        return ""
    if content is None:
        return ""
    if isinstance(content, list):
        for item in content:
            signal = _describe_thinking_signal(item)
            if signal:
                return signal
        return ""
    if isinstance(content, dict):
        item_type = str(content.get("type") or "").strip().lower()
        if bool(content.get("thought")):
            return "thought_flag"
        if item_type in {"thinking", "thought", "reasoning", "reasoning_content", "reasoning_delta", "reasoning_summary", "redacted_thinking"}:
            return f"type:{item_type}"
        if item_type.startswith(("reasoning.", "thinking.", "thought.", "analysis.")):
            return f"type:{item_type}"
        for key in (
            "analysis",
            "analysis_content",
            "reasoning_content",
            "reasoning",
            "reasoning_delta",
            "reasoning_summary",
            "reasoning_details",
            "thinking",
            "thinking_content",
        ):
            if key in content and content.get(key):
                nested_signal = _describe_thinking_signal(content.get(key))
                return f"key:{key}" + (f"/{nested_signal}" if nested_signal else "")
        for key in ("parts", "items", "content", "content_blocks", "message", "messages", "data"):
            if key in content:
                signal = _describe_thinking_signal(content.get(key))
                if signal:
                    return f"key:{key}/{signal}"
        return ""
    for key in ("content", "content_blocks", "text", "additional_kwargs"):
        try:
            value = getattr(content, key)
        except Exception:
            continue
        if callable(value) or value is None:
            continue
        signal = _describe_thinking_signal(value)
        if signal:
            return f"attr:{key}/{signal}"
    return ""


def _reasoning_type_from_signal(signal: str) -> str:
    normalized = str(signal or "").lower()
    if not normalized:
        return "unknown"
    if "inline_thought_tag" in normalized:
        return "think_tag"
    if "reasoning.delta" in normalized:
        return "reasoning_delta"
    if "reasoning" in normalized:
        return "reasoning_field"
    if "thinking" in normalized:
        return "thinking_block"
    if "analysis" in normalized:
        return "analysis_field"
    if "thought" in normalized:
        return "thinking_block"
    return "unknown"


def _normalize_stream_chunk(chunk: Any) -> tuple[str, Any]:
    if isinstance(chunk, dict):
        return str(chunk["type"]), chunk["data"]
    if isinstance(chunk, tuple) and len(chunk) == 2:
        mode, payload = chunk
        return str(mode), payload
    raise TypeError(f"Unsupported stream chunk format: {type(chunk).__name__}")


@dataclass(frozen=True)
class StreamEvent:
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamProcessResult:
    stats: Optional[str]
    interrupt: Optional[Dict[str, Any]] = None
    cancelled: bool = False
    failed: bool = False
    error_message: str = ""
    error_details: Dict[str, Any] = field(default_factory=dict)
    cancelled_tools: list[Dict[str, Any]] = field(default_factory=list)
    events: list[StreamEvent] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class StreamProcessor:
    __slots__ = (
        "emit_event",
        "tracker",
        "full_text",
        "clean_full",
        "printed_tool_ids",
        "tool_buffer",
        "tool_start_times",
        "start_time",
        "pending_interrupt",
        "active_node",
        "events",
        "_last_status",
        "_text_max_chars",
        "_events_max",
        "_tool_buffer_max",
        "_completed_tool_ids",
        "_tool_id_aliases",
        "_tool_index_to_id",
        "_tool_id_to_index",
        "_tool_chunk_accumulators",
        "_synthetic_tool_counter",
        "_base_elapsed_seconds",
        "_last_internal_notice",
        "_agent_is_thinking",
        "_first_token_at",
        "_first_reasoning_at",
        "_messages_text_seen",
        "_suppressing_text_tool_call",
        "_visible_full_text",
        "_deferred_assistant_delta",
        "_assistant_delta_sequence",
        "_messages_replay_cursor",
        "_tool_sources",
        "_mcp_tool_servers",
        "_previous_assistant_section_text",
        "_post_tool_replay_text",
        "_post_tool_text_buffer",
        "_post_tool_needs_boundary",
    )

    def __init__(
        self,
        emit_event: Callable[[StreamEvent], None] | None = None,
        *,
        text_max_chars: int = 120000,
        events_max: int = 400,
        tool_buffer_max: int = 128,
        base_elapsed_seconds: float = 0.0,
        token_tracker: TokenTracker | None = None,
        tool_sources: Dict[str, str] | None = None,
        mcp_tool_servers: Dict[str, str] | None = None,
    ):
        self.emit_event = emit_event
        # Keep request usage and message-ID deduplication across stream segments.
        self.tracker = token_tracker if token_tracker is not None else TokenTracker()
        self.tracker.advance_step()
        self.full_text = ""
        self.clean_full = ""
        self.printed_tool_ids: Set[str] = set()
        self.tool_buffer: Dict[str, Dict[str, Any]] = {}
        self.tool_start_times: Dict[str, float] = {}
        self.start_time = time.perf_counter()
        self.pending_interrupt: Optional[Dict[str, Any]] = None
        self.active_node = "agent"
        self.events: list[StreamEvent] = []
        self._last_status: tuple[str, str] | None = None
        self._text_max_chars = max(1, int(text_max_chars))
        self._events_max = max(1, int(events_max))
        self._tool_buffer_max = max(1, int(tool_buffer_max))
        self._completed_tool_ids: dict[str, None] = {}
        self._tool_id_aliases: Dict[str, str] = {}
        self._tool_index_to_id: Dict[int, str] = {}
        self._tool_id_to_index: Dict[str, int] = {}
        self._tool_chunk_accumulators: Dict[str, Dict[int, AIMessageChunk]] = {}
        self._synthetic_tool_counter = 0
        self._base_elapsed_seconds = max(0.0, float(base_elapsed_seconds or 0.0))
        self._last_internal_notice = ""
        self._agent_is_thinking = False
        self._first_token_at: float | None = None
        self._first_reasoning_at: float | None = None
        self._messages_text_seen = False
        self._suppressing_text_tool_call = False
        self._visible_full_text = ""
        self._deferred_assistant_delta = False
        self._assistant_delta_sequence = 0
        self._messages_replay_cursor: int | None = None
        self._tool_sources = dict(tool_sources or {})
        self._mcp_tool_servers = dict(mcp_tool_servers or {})
        self._previous_assistant_section_text = ""
        self._post_tool_replay_text = ""
        self._post_tool_text_buffer = ""
        self._post_tool_needs_boundary = False

    def _elapsed_seconds(self) -> float:
        return self._base_elapsed_seconds + max(0.0, time.perf_counter() - self.start_time)

    async def process_stream(self, stream) -> StreamProcessResult:
        self._emit_status(force=True)
        try:
            async for chunk in stream:
                debug_event(
                    "raw_stream_chunk",
                    source="langgraph_stream",
                    elapsed=elapsed_since(self.start_time),
                    chunk_preview=preview_value(chunk),
                )
                log_unknown_fields("langgraph_stream_chunk", chunk)
                mode, payload = _normalize_stream_chunk(chunk)
                self._handle_stream_event(mode, payload)
                if self.pending_interrupt is not None:
                    break
        except (KeyboardInterrupt, asyncio.CancelledError):
            cancelled_tools = self._emit_interrupted_tool_results(reason="cancelled")
            return StreamProcessResult(
                stats=None,
                cancelled=True,
                cancelled_tools=cancelled_tools,
                events=list(self.events),
                elapsed_seconds=self._elapsed_seconds(),
            )
        except Exception as exc:
            error_details = describe_provider_error(exc)
            logger.warning(
                "Stream processing failed: type=%s http_status=%s url_host=%s "
                "request_error_id=%s root_cause_type=%s message=%s root_cause=%s",
                error_details["error_type"],
                error_details["http_status"] or "-",
                error_details["url_host"] or "-",
                error_details["request_error_id"] or "-",
                error_details["root_cause_type"],
                error_details["message"],
                error_details["root_cause_message"],
            )
            self._emit_interrupted_tool_results(reason="stream_error")
            self._emit("run_failed", {"message": error_details["message"], **error_details})
            return StreamProcessResult(
                stats=None,
                failed=True,
                error_message=error_details["message"],
                error_details=error_details,
                events=list(self.events),
                elapsed_seconds=self._elapsed_seconds(),
            )

        if self.pending_interrupt is not None:
            interrupt_payload = self.pending_interrupt
            self.pending_interrupt = None
            return StreamProcessResult(
                stats=None,
                interrupt=interrupt_payload,
                events=list(self.events),
                elapsed_seconds=self._elapsed_seconds(),
            )

        self._flush_deferred_assistant_delta()
        self._emit_assistant_commit()
        # A well-behaved run resolves every announced tool call, but a flaky model or
        # provider can end the stream with a tool card still in the "preparing"/running
        # state (announced via tool_started, never followed by a ToolMessage result).
        # Finalize those orphans so their cards do not hang as if still executing.
        self._emit_interrupted_tool_results(reason="incomplete")
        duration = self._elapsed_seconds()
        debug_event(
            "stream_timing",
            time_to_first_token=(self._first_token_at - self.start_time) if self._first_token_at is not None else None,
            time_to_first_reasoning=(self._first_reasoning_at - self.start_time)
            if self._first_reasoning_at is not None
            else None,
            time_to_final_response=duration,
        )
        stats = self.tracker.render(duration)
        self._emit("run_finished", {"stats": stats, "duration": duration})
        return StreamProcessResult(
            stats=stats,
            events=list(self.events),
            elapsed_seconds=duration,
        )

    def _emit(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        event = StreamEvent(event_type, payload or {})
        self.events.append(event)
        if len(self.events) > self._events_max:
            overflow = len(self.events) - self._events_max
            del self.events[:overflow]
        if self.emit_event:
            self.emit_event(event)

    def _trim_text_buffers(self) -> None:
        if len(self.full_text) > self._text_max_chars:
            self.full_text = "… " + self.full_text[-(self._text_max_chars - 2) :]
        if len(self.clean_full) > self._text_max_chars:
            self.clean_full = "… " + self.clean_full[-(self._text_max_chars - 2) :]

    def _trim_tool_buffers(self) -> None:
        while len(self.tool_buffer) > self._tool_buffer_max:
            oldest_id = next(iter(self.tool_buffer), None)
            if oldest_id is None:
                break
            self.tool_buffer.pop(oldest_id, None)
            self.tool_start_times.pop(oldest_id, None)
            self.printed_tool_ids.discard(oldest_id)
        while len(self.tool_start_times) > self._tool_buffer_max:
            oldest_id = next(iter(self.tool_start_times), None)
            if oldest_id is None:
                break
            self.tool_start_times.pop(oldest_id, None)
            self.printed_tool_ids.discard(oldest_id)

        while len(self._completed_tool_ids) > self._tool_buffer_max:
            oldest_id = next(iter(self._completed_tool_ids), None)
            if oldest_id is None:
                break
            self._completed_tool_ids.pop(oldest_id, None)

        active_ids = set(self.tool_buffer) | set(self.tool_start_times)
        for alias_id, target_id in list(self._tool_id_aliases.items()):
            if alias_id in active_ids or target_id in active_ids:
                continue
            self._tool_id_aliases.pop(alias_id, None)
            self._tool_id_to_index.pop(alias_id, None)

    def _handle_stream_event(self, mode: str, payload: Any) -> None:
        if mode == "updates":
            self._handle_updates(payload)
        elif mode == "messages":
            self._handle_messages(payload)
        elif mode == "custom":
            self._handle_custom(payload)

    def _handle_custom(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("type") == "api_key_rotated":
            self._handle_api_key_rotated(payload)
            return
        if payload.get("type") == "tool_batch_started":
            tool_calls = payload.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                return
            self.active_node = "tools"
            self._agent_is_thinking = False
            self._emit_status(force=True)
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    self._remember_tool_call(tool_call)
                    self._emit_tool_started(tool_call, force=True)
            return
        if payload.get("type") == "tool_result":
            message_payload = payload.get("message")
            if not isinstance(message_payload, dict):
                return
            try:
                message = ToolMessage(**message_payload)
            except (TypeError, ValueError):
                logger.debug("Ignored malformed custom tool result payload.", exc_info=True)
                return
            self._handle_tool_result(message)
            return
        if payload.get("type") != "status_changed":
            return

        label = str(payload.get("label") or "").strip()
        if not label:
            return
        node = str(payload.get("node") or self.active_node or "agent")
        self.active_node = node
        self._last_status = (node, label)
        status_payload = {key: value for key, value in payload.items() if key != "type"}
        status_payload.setdefault("node", node)
        status_payload.setdefault("elapsed", self._elapsed_seconds())
        status_payload.setdefault("elapsed_text", self._status_elapsed_text())
        status_payload.setdefault("phase", self._status_phase())
        self._emit("status_changed", status_payload)

    def _handle_api_key_rotated(self, payload: Dict[str, Any]) -> None:
        """Restart the assistant section after an API-key rotation mid-run.

        Tokens already streamed by the failed key belong to an aborted response.
        The retry on the next key streams a fresh answer, so the partial text
        must be closed as its own section instead of being concatenated with
        the replayed response.
        """
        error_kind = str(payload.get("error_kind") or "").strip()
        notice = "Provider key limit reached; switching to the next API key and retrying."
        if error_kind:
            notice = f"Provider key limit reached ({error_kind}); switching to the next API key and retrying."
        self._emit(
            "summary_notice",
            {
                "message": notice,
                "kind": "api_key_rotated",
                "level": "warning",
            },
        )
        self._begin_assistant_stream_section()
        # The next key regenerates the answer from scratch; its opening may
        # legitimately match the aborted partial text, so the previous-section
        # replay guard must not strip it.
        self._previous_assistant_section_text = ""

    def _emit_cache_hit_delta(self, tokens: int) -> None:
        delta = max(0, int(tokens or 0))
        if delta:
            self._emit("cache_hit", {"tokens": delta})

    def _handle_updates(self, payload: Dict[str, Any]) -> None:
        if "__interrupt__" in payload:
            self.active_node = "approval"
            self._emit_status(force=True)
            interrupt_entries = payload.get("__interrupt__") or []
            if interrupt_entries:
                interrupt_value = getattr(interrupt_entries[0], "value", interrupt_entries[0])
                if isinstance(interrupt_value, dict):
                    self.pending_interrupt = interrupt_value
                else:
                    self.pending_interrupt = {"value": interrupt_value}
            return

        self._emit_cache_hit_delta(self.tracker.update_from_node_update(payload))

        summarize_payload = payload.get("summarize") or {}
        if summarize_payload:
            self._handle_summarize_update(summarize_payload)

        agent_payload = payload.get("agent") or {}
        messages = agent_payload.get("messages", [])
        if not isinstance(messages, list):
            messages = [messages]

        last_message = messages[-1] if messages else None
        if isinstance(last_message, (AIMessage, AIMessageChunk)):
            self._handle_agent_message(last_message, source="updates_agent")

        for node_name, node_payload in payload.items():
            if node_name in {"agent", "summarize", "__interrupt__"}:
                continue
            if not isinstance(node_payload, dict):
                continue
            node_messages = node_payload.get("messages", [])
            if not isinstance(node_messages, list):
                node_messages = [node_messages]
            for message in node_messages:
                if not isinstance(message, (AIMessage, AIMessageChunk, ToolMessage)):
                    continue
                self._emit_cache_hit_delta(self.tracker.update_from_message(message))
                if isinstance(message, (AIMessage, AIMessageChunk)):
                    self._handle_agent_message(message, source=f"updates_{node_name}")
                elif isinstance(message, ToolMessage):
                    self._handle_tool_result(message)

    def _handle_summarize_update(self, payload: Dict[str, Any]) -> None:
        summary_text = payload.get("summary")
        removed_messages = payload.get("messages") or []
        remove_count = sum(1 for item in removed_messages if isinstance(item, RemoveMessage))
        if not summary_text:
            return
        rendered_count = remove_count if remove_count > 0 else len(removed_messages)
        self._emit(
            "summary_notice",
            {
                "message": f"Context compressed automatically ({rendered_count} message(s) summarized).",
                "count": rendered_count,
                "kind": "auto_summary",
            },
        )

    def _handle_messages(self, payload: tuple[Any, Dict[str, Any]]) -> None:
        message, metadata = payload
        node = metadata.get("langgraph_node")
        self._emit_cache_hit_delta(self.tracker.update_from_message(message))

        if node == "tools" and isinstance(message, ToolMessage):
            self._agent_is_thinking = False
            self._handle_tool_result(message)
            return

        if node:
            if node != "agent":
                self._agent_is_thinking = False
            self.active_node = "agent" if node == "agent" else node
            if node == "agent" and isinstance(message, (AIMessage, AIMessageChunk)):
                self._agent_is_thinking = self._agent_is_thinking or self._has_thinking_content(message)
                if self._agent_is_thinking and self._first_reasoning_at is None:
                    self._first_reasoning_at = now()
                reasoning_logger.debug(
                    "status parse source=messages node=agent message_type=%s thinking=%s signal=%s content_type=%s",
                    type(message).__name__,
                    self._agent_is_thinking,
                    _describe_thinking_signal(message),
                    type(getattr(message, "content", None)).__name__,
                )
            self._emit_status()

        if node == "agent" and isinstance(message, (AIMessage, AIMessageChunk)):
            self._handle_agent_message(message, source="messages")

    def _handle_agent_message(
        self,
        message: AIMessage | AIMessageChunk,
        *,
        source: str = "messages",
    ) -> None:
        self._handle_tool_calls_from_message(message, source=source)
        message_has_tool_calls = self._message_has_tool_calls(message)
        if is_hidden_internal_message(message):
            notice = get_internal_ui_notice(message)
            if notice and notice != self._last_internal_notice:
                self._last_internal_notice = notice
                self._emit(
                    "summary_notice",
                    {
                        "message": notice,
                        "kind": "agent_internal_notice",
                        "level": "warning",
                    },
                )
            return

        if self.active_node == "agent":
            self._agent_is_thinking = self._agent_is_thinking or self._has_thinking_content(message)
            if self._agent_is_thinking and self._first_reasoning_at is None:
                self._first_reasoning_at = now()
            signal = _describe_thinking_signal(message)
            debug_event(
                "reasoning_detected",
                detected=self._agent_is_thinking,
                reasoning_type=_reasoning_type_from_signal(signal),
                source_field=signal,
                source=source,
            )
            reasoning_logger.debug(
                "status parse source=%s node=agent message_type=%s thinking=%s signal=%s content_type=%s",
                source,
                type(message).__name__,
                self._agent_is_thinking,
                signal,
                type(getattr(message, "content", None)).__name__,
            )
            self._emit_status()

        chunk = self._extract_text_content(message.content)
        if not chunk:
            chunk = self._extract_text_content(message)
        reasoning_logger.debug(
            "thought text extraction source=%s message_type=%s visible_chunk_len=%s thinking_signal=%s",
            source,
            type(message).__name__,
            len(chunk or ""),
            _describe_thinking_signal(message),
        )
        if chunk and self._is_reasoning_only_visible_text(message, chunk):
            reasoning_logger.debug(
                "suppressed reasoning-only visible chunk source=%s chunk_preview=%s",
                source,
                _clip_debug_text(chunk),
            )
            return
        if not chunk:
            logger.debug(
                "Stream empty assistant chunk content_type=%s additional_kwargs=%s response_metadata=%s content_preview=%s",
                type(message.content).__name__,
                sorted((getattr(message, "additional_kwargs", {}) or {}).keys()),
                sorted((getattr(message, "response_metadata", {}) or {}).keys()),
                _clip_debug_text(message.content),
            )
            return
        if not has_visible_text(chunk) and (
            self._post_tool_needs_boundary or not has_visible_text(self.full_text)
        ):
            logger.debug(
                "Stream ignored invisible assistant chunk at section boundary content_preview=%s",
                _clip_debug_text(chunk),
            )
            return
        chunk = self._filter_visible_text_tool_call_chunk(chunk)
        if not chunk:
            return
        chunk = self._strip_previous_section_replay(chunk)
        if not chunk:
            return
        guarded_chunk = self._guard_post_tool_replay(
            chunk,
            source=source,
            message_has_tool_calls=message_has_tool_calls,
        )
        if guarded_chunk is None:
            return
        chunk = guarded_chunk
        if not chunk:
            return
        if source == "messages" and self._consume_messages_replay_chunk(chunk):
            logger.debug(
                "Stream suppressed messages replay after updates_agent chunk_len=%s preview=%s",
                len(chunk),
                _clip_debug_text(chunk),
            )
            return
        if self._messages_text_seen and (
            (source == "updates_agent" and not self._updates_agent_text_has_new_visible_content(chunk))
            or (message_has_tool_calls and not self._updates_agent_text_has_new_visible_content(chunk))
        ):
            logger.debug(
                "Stream suppressed assistant text replay chunk_len=%s source=%s tool_calls=%s preview=%s",
                len(chunk),
                source,
                message_has_tool_calls,
                _clip_debug_text(chunk),
            )
            return
        if self._first_token_at is None:
            self._first_token_at = now()

        if self._post_tool_needs_boundary:
            self._post_tool_needs_boundary = False
            self._begin_assistant_stream_section(flush_deferred=False)

        merged_text = self._merge_assistant_text(chunk, source=source)
        if merged_text is None:
            return
        if source == "messages":
            self._messages_text_seen = True
            self._messages_replay_cursor = None

        self.full_text = merged_text
        self.clean_full = self.full_text
        logger.debug(
            "Stream assistant chunk source=%s chunk_len=%s full_len=%s chunk_preview=%s",
            source,
            len(chunk),
            len(self.full_text),
            _clip_debug_text(chunk),
        )

        self._trim_text_buffers()

        if self._should_defer_assistant_text(message_has_tool_calls=message_has_tool_calls):
            self._deferred_assistant_delta = True
            logger.debug(
                "Stream deferred assistant chunk until active tools finish source=%s active_tools=%s full_len=%s",
                source,
                len(self.tool_start_times),
                len(self.full_text),
            )
            return

        self._emit_assistant_delta(chunk)

    def _emit_assistant_delta(self, chunk: str) -> None:
        self._emit_status()
        visible_text = self.clean_full.strip()
        self._assistant_delta_sequence += 1
        logger.debug(
            "Stream emit_assistant_delta visible_len=%s visible_preview=%s",
            len(visible_text),
            _clip_debug_text(visible_text),
        )
        self._emit(
            "assistant_delta",
            {
                "text": chunk,
                # Keep the stream processor focused on text assembly. Markdown
                # normalization is intentionally deferred to the coalesced UI
                # render so bursts of provider tokens do not repeatedly scan the
                # complete response before most intermediate states are dropped.
                "full_text": visible_text,
                "sequence": self._assistant_delta_sequence,
            },
        )
        self._visible_full_text = self.full_text
        self._deferred_assistant_delta = False

    def _emit_assistant_commit(self) -> None:
        """Commit the last visible text before the run-finished UI transition."""
        visible_text = self.clean_full.strip()
        if not visible_text or visible_text != self._visible_full_text.strip():
            return
        self._emit(
            "assistant_commit",
            {
                "full_text": visible_text,
                "sequence": self._assistant_delta_sequence,
            },
        )

    def _consume_messages_replay_chunk(self, incoming_text: str) -> bool:
        if self._messages_text_seen or not self.full_text:
            self._messages_replay_cursor = None
            return False

        text = str(incoming_text or "")
        if not text:
            return False

        current = self.full_text
        if self._messages_replay_cursor is None:
            self._messages_replay_cursor = 0

        cursor = self._messages_replay_cursor
        if cursor < len(current) and current.startswith(text, cursor):
            self._messages_replay_cursor = cursor + len(text)
            return True

        self._messages_replay_cursor = None
        return False

    def _flush_deferred_assistant_delta(self) -> None:
        if not self._deferred_assistant_delta or self.tool_start_times:
            return
        if not self.full_text or self.full_text == self._visible_full_text:
            self._deferred_assistant_delta = False
            return
        if self.full_text.startswith(self._visible_full_text):
            chunk = self.full_text[len(self._visible_full_text) :]
        else:
            chunk = self.full_text
        logger.debug(
            "Stream flushing deferred assistant delta full_len=%s visible_len=%s",
            len(self.full_text),
            len(self._visible_full_text),
        )
        self._emit_assistant_delta(chunk)

    def _should_defer_assistant_text(self, *, message_has_tool_calls: bool) -> bool:
        if message_has_tool_calls:
            return False
        return bool(self.tool_start_times)

    def _updates_agent_text_has_new_visible_content(self, incoming_text: str) -> bool:
        incoming = str(incoming_text or "").strip()
        if not incoming:
            return False
        # Compare against _visible_full_text only (text actually emitted to the UI),
        # NOT full_text — full_text may contain deferred text that was never shown.
        # Using full_text as fallback would falsely suppress updates_agent text
        # that duplicates deferred-but-not-yet-visible messages text.
        visible = str(self._visible_full_text or "").strip()
        if not visible:
            return True
        if incoming == visible or incoming.startswith(visible) or visible.startswith(incoming):
            return False
        if incoming in visible or self._normalized_text(incoming) in self._normalized_text(visible):
            return False
        for paragraph in self._nonempty_paragraphs(visible):
            if self._texts_are_same_replay(paragraph, incoming):
                return False
        if min(len(incoming), len(visible)) >= 20:
            similarity = difflib.SequenceMatcher(None, visible, incoming).quick_ratio()
            if similarity >= 0.92:
                return False
        if self._is_fuzzy_substring_of(incoming, visible):
            return False
        replay_merge = self._merge_with_replay_guard(visible, incoming)
        if replay_merge is not None:
            return replay_merge.strip() != visible
        return True

    @staticmethod
    def _is_fuzzy_substring_of(incoming: str, visible: str) -> bool:
        """Check if *incoming* is approximately contained within *visible*.

        This catches the case where ``updates_agent`` replays only a portion of
        the already-visible ``messages`` text with minor typos.  The overall
        ``quick_ratio`` is too low in this scenario because *visible* is much
        longer than *incoming*, so we measure the **containment ratio** instead:
        what fraction of *incoming*'s characters appear (in order) inside
        *visible*.
        """
        incoming_norm = StreamProcessor._normalized_text(incoming)
        visible_norm = StreamProcessor._normalized_text(visible)
        if len(incoming_norm) < 80 or len(visible_norm) <= len(incoming_norm):
            return False
        matcher = difflib.SequenceMatcher(None, visible_norm, incoming_norm, autojunk=False)
        total_matched = sum(block.size for block in matcher.get_matching_blocks())
        return total_matched / len(incoming_norm) >= 0.85

    def _merge_assistant_text(
        self,
        incoming_text: str,
        *,
        source: str,
    ) -> str | None:
        text = str(incoming_text or "")
        if not text:
            return None

        current = self.full_text
        if not current:
            return text

        if text == current or text.strip() == current.strip():
            # If the text was never emitted to the UI (deferred), allow re-emission
            # so updates_agent with tool_calls can show it. Only suppress if it's
            # already visible.
            if self._visible_full_text and self._visible_full_text.strip() == current.strip():
                return None
            return current

        if text.startswith(current):
            merged = self._merge_current_prefix_replay(current, text)
            return merged

        merged = self._merge_with_replay_guard(current, text)
        if merged is not None:
            return merged

        if source == "messages":
            return current + text

        if source == "updates_agent":
            return self._append_distinct_assistant_text(current, text)

        return current + text

    def _merge_current_prefix_replay(self, current: str, incoming: str) -> str | None:
        remainder = incoming[len(current) :]
        if not remainder:
            return None

        if current.rstrip().endswith(remainder.strip()):
            logger.debug(
                "Stream suppressed duplicate assistant replay after full prefix current_len=%s incoming_len=%s",
                len(current),
                len(incoming),
            )
            return None

        return incoming

    def _merge_with_replay_guard(self, current: str, incoming: str) -> str | None:
        replay = self._drop_replayed_tail(current, incoming)
        if replay is not None:
            return replay

        overlap = self._suffix_prefix_overlap(current, incoming)
        if overlap <= 0:
            return None
        merged = current + incoming[overlap:]
        if merged == current or merged.strip() == current.strip():
            return None
        logger.debug(
            "Stream merged assistant text by suffix/prefix overlap overlap=%s current_len=%s incoming_len=%s",
            overlap,
            len(current),
            len(incoming),
        )
        return merged

    def _drop_replayed_tail(self, current: str, incoming: str) -> str | None:
        current_trimmed = current.rstrip()
        incoming_left = incoming.lstrip()
        incoming_trimmed = incoming_left.strip()
        if not current_trimmed or not incoming_trimmed:
            return None

        if current_trimmed.endswith(incoming_trimmed):
            logger.debug(
                "Stream suppressed assistant text replay matching already visible tail incoming_len=%s",
                len(incoming),
            )
            return None

        tail = self._last_nonempty_paragraph(current_trimmed)
        if not tail:
            return None

        if incoming_left.startswith(tail):
            remainder = incoming_left[len(tail) :]
        else:
            boundary = self._similar_prefix_boundary(tail, incoming_left)
            if boundary is None:
                return None
            remainder = incoming_left[boundary:]

        if not remainder.strip():
            logger.debug(
                "Stream suppressed assistant text replay matching last paragraph incoming_len=%s",
                len(incoming),
            )
            return None

        logger.debug(
            "Stream dropped replayed assistant tail paragraph tail_len=%s incoming_len=%s",
            len(tail),
            len(incoming),
        )
        return current + remainder

    @classmethod
    def _similar_prefix_boundary(cls, current_tail: str, incoming: str) -> int | None:
        tail_normalized = cls._normalized_text(current_tail)
        if len(tail_normalized) < 24:
            return None

        best_index: int | None = None
        for match in re.finditer(r"\S+", incoming):
            prefix = incoming[: match.end()]
            prefix_normalized = cls._normalized_text(prefix)
            if not prefix_normalized:
                continue
            if tail_normalized.startswith(prefix_normalized):
                best_index = match.end()
                continue
            if prefix_normalized.startswith(tail_normalized):
                return best_index if best_index is not None else match.end()
            if len(prefix_normalized) >= min(len(tail_normalized), 80):
                similarity = difflib.SequenceMatcher(None, tail_normalized, prefix_normalized).quick_ratio()
                if similarity >= 0.94:
                    best_index = match.end()

        return best_index

    @staticmethod
    def _last_nonempty_paragraph(text: str) -> str:
        paragraphs = StreamProcessor._nonempty_paragraphs(text)
        return paragraphs[-1] if paragraphs else ""

    @staticmethod
    def _nonempty_paragraphs(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n\s*\n", str(text or "")) if part.strip()]

    @staticmethod
    def _normalized_text(text: str) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _texts_are_same_replay(current: str, incoming: str) -> bool:
        current_normalized = StreamProcessor._normalized_text(current)
        incoming_normalized = StreamProcessor._normalized_text(incoming)
        if not current_normalized or not incoming_normalized:
            return False
        if current_normalized == incoming_normalized:
            return True
        if incoming_normalized in current_normalized:
            return True
        if min(len(current_normalized), len(incoming_normalized)) < 80:
            return False
        return difflib.SequenceMatcher(None, current_normalized, incoming_normalized).quick_ratio() >= 0.94

    @staticmethod
    def _append_distinct_assistant_text(current: str, incoming: str) -> str:
        if not current:
            return incoming
        separator = "" if current.endswith(("\n", " ")) or incoming.startswith(("\n", " ")) else "\n\n"
        return current + separator + incoming

    @staticmethod
    def _suffix_prefix_overlap(current: str, incoming: str) -> int:
        max_len = min(len(current), len(incoming))
        for size in range(max_len, 0, -1):
            if current.endswith(incoming[:size]):
                return size
        return 0

    def _filter_visible_text_tool_call_chunk(self, text: str) -> str:
        raw_text = str(text or "")
        if not raw_text:
            return ""

        if self._suppressing_text_tool_call:
            marker_index = raw_text.find(_TEXT_TOOL_CALL_END)
            if marker_index < 0:
                return ""
            self._suppressing_text_tool_call = False
            return raw_text[marker_index + len(_TEXT_TOOL_CALL_END) :]

        match = _TEXT_TOOL_CALL_START_RE.search(raw_text)
        if not match:
            return raw_text

        start = match.start(0)
        before = raw_text[:start]
        marker_index = raw_text.find(_TEXT_TOOL_CALL_END, start)
        if marker_index < 0:
            self._suppressing_text_tool_call = True
            return before
        after = raw_text[marker_index + len(_TEXT_TOOL_CALL_END) :]
        return before + after

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return _strip_inline_thought_content(content)
        if content is None:
            return ""
        if isinstance(content, list):
            return "".join(StreamProcessor._extract_text_content(item) for item in content)
        if isinstance(content, dict):
            item_type = str(content.get("type") or "").strip().lower()
            if bool(content.get("thought")) or item_type in {
                "thinking",
                "thought",
                "reasoning",
                "reasoning_content",
                "reasoning_delta",
                "reasoning_summary",
                "analysis",
                "analysis_content",
                "summary_text",
                "redacted_thinking",
            } or item_type.startswith(("reasoning.", "thinking.", "thought.", "analysis.")):
                return ""
            for key in ("text", "output_text", "content", "answer", "response", "final", "final_text"):
                if key in content:
                    text = StreamProcessor._extract_text_content(content.get(key))
                    if text:
                        return text
            return "".join(
                StreamProcessor._extract_text_content(content.get(key))
                for key in ("parts", "items", "content_blocks", "message", "messages", "data")
                if key in content
            )
        for key in ("content", "content_blocks", "text"):
            try:
                value = getattr(content, key)
            except Exception:
                continue
            if callable(value) or value is None:
                continue
            text = StreamProcessor._extract_text_content(value)
            if text:
                return text
        try:
            additional_kwargs = getattr(content, "additional_kwargs")
        except Exception:
            additional_kwargs = None
        if isinstance(additional_kwargs, dict):
            return StreamProcessor._extract_text_content(additional_kwargs.get("content_blocks"))
        return ""

    @staticmethod
    def _is_reasoning_only_visible_text(message: AIMessage | AIMessageChunk, chunk: str) -> bool:
        signal = _describe_thinking_signal(message)
        if not signal:
            return False
        metadata = getattr(message, "additional_kwargs", {}) or {}
        if not isinstance(metadata, dict):
            return False
        reasoning_values = []
        for key in (
            "reasoning_content",
            "reasoning",
            "reasoning_delta",
            "analysis",
            "analysis_content",
            "thinking",
            "thinking_content",
        ):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                reasoning_values.append(value)
        if not reasoning_values:
            return False
        normalized_chunk = str(chunk or "").strip()
        return any(normalized_chunk == value.strip() for value in reasoning_values)

    @staticmethod
    def _has_thinking_content(content: Any) -> bool:
        if isinstance(content, str):
            return bool(_INLINE_THOUGHT_BLOCK_RE.search(content) or _INLINE_THOUGHT_OPEN_RE.search(content))
        if content is None:
            return False
        if isinstance(content, list):
            return any(StreamProcessor._has_thinking_content(item) for item in content)
        if isinstance(content, dict):
            item_type = str(content.get("type") or "").strip().lower()
            if bool(content.get("thought")) or item_type in {
                "thinking",
                "thought",
                "reasoning",
                "reasoning_content",
                "reasoning_summary",
                "analysis",
                "analysis_content",
                "redacted_thinking",
            } or item_type.startswith(("reasoning.", "thinking.", "thought.", "analysis.")):
                return True
            for key in (
                "analysis",
                "analysis_content",
                "reasoning_content",
                "reasoning",
                "reasoning_delta",
                "reasoning_summary",
                "reasoning_details",
                "thinking",
                "thinking_content",
            ):
                if key in content and content.get(key):
                    if isinstance(content.get(key), str):
                        return True
                    if StreamProcessor._has_thinking_content(content.get(key)):
                        return True
                    return True
            return any(
                StreamProcessor._has_thinking_content(content.get(key))
                for key in ("parts", "items", "content", "content_blocks", "message", "messages", "data")
                if key in content
            )
        for key in ("content", "content_blocks", "text", "additional_kwargs"):
            try:
                value = getattr(content, key)
            except Exception:
                continue
            if callable(value) or value is None:
                continue
            if StreamProcessor._has_thinking_content(value):
                return True
        return False

    @staticmethod
    def _message_has_tool_calls(message: AIMessage | AIMessageChunk) -> bool:
        return bool(
            list(getattr(message, "tool_calls", []) or [])
            or list(getattr(message, "tool_call_chunks", []) or [])
            or list(getattr(message, "invalid_tool_calls", []) or [])
        )

    def _begin_assistant_stream_section(self, *, flush_deferred: bool = True) -> None:
        """Close one assistant response before the next agent invocation."""
        if flush_deferred:
            self._flush_deferred_assistant_delta()
        previous_text = self.full_text or self._visible_full_text
        self._previous_assistant_section_text = previous_text.strip()
        self._emit("assistant_boundary", {})
        self.full_text = ""
        self.clean_full = ""
        self._visible_full_text = ""
        self._messages_text_seen = False
        self._messages_replay_cursor = None
        self._deferred_assistant_delta = False

    def _begin_post_tool_assistant_section(self) -> str:
        """Start a fresh assistant section after tools and return deferred tail text."""
        visible_before_tool = self._visible_full_text or ""
        full_before_boundary = self.full_text or ""
        deferred_tail = ""
        if self._deferred_assistant_delta and visible_before_tool:
            if full_before_boundary.startswith(visible_before_tool):
                deferred_tail = full_before_boundary[len(visible_before_tool) :]
            elif full_before_boundary.strip() != visible_before_tool.strip():
                deferred_tail = full_before_boundary
        elif self._deferred_assistant_delta:
            deferred_tail = full_before_boundary

        previous_section_text = visible_before_tool or full_before_boundary
        if deferred_tail.strip():
            self._begin_assistant_stream_section(flush_deferred=False)
        elif previous_section_text.strip():
            self._post_tool_needs_boundary = True
        return deferred_tail

    def _strip_previous_section_replay(self, incoming_text: str) -> str:
        """Remove a cumulative replay after a section boundary.

        Some providers occasionally start the next agent response with the
        complete commentary from the preceding step.  The transcript has
        already closed that step, so forwarding the cumulative text creates a
        second copy above the next tool card.
        """
        previous = self._previous_assistant_section_text
        incoming = str(incoming_text or "")
        if not previous or not incoming.strip():
            return incoming

        self._previous_assistant_section_text = ""
        if incoming.startswith(previous):
            remainder = incoming[len(previous) :].lstrip()
            logger.debug(
                "Stream removed previous section replay previous_len=%s incoming_len=%s",
                len(previous),
                len(incoming),
            )
            return remainder

        boundary = self._similar_prefix_boundary(previous, incoming)
        previous_normalized = self._normalized_text(previous)
        replayed_prefix = self._normalized_text(incoming[:boundary]) if boundary is not None else ""
        significant_replay = bool(replayed_prefix) and (
            len(replayed_prefix) >= min(len(previous_normalized), 48)
            or len(replayed_prefix) >= int(len(previous_normalized) * 0.8)
        )
        if boundary is not None and significant_replay:
            remainder = incoming[boundary:].lstrip()
            logger.debug(
                "Stream removed approximate previous section replay boundary=%s incoming_len=%s",
                boundary,
                len(incoming),
            )
            return remainder
        return incoming

    def _guard_post_tool_replay(
        self,
        incoming_text: str,
        *,
        source: str,
        message_has_tool_calls: bool,
    ) -> str | None:
        """Buffer the first post-tool tokens until they are known to be new.

        Final node updates can replace duplicated text, but that happens too
        late for a live transcript.  Holding only the ambiguous prefix prevents
        a replayed comment from flashing in the UI while distinct commentary
        still appears as soon as it diverges from the previous paragraph.
        """
        replay = self._post_tool_replay_text
        if not replay:
            return incoming_text

        incoming = str(incoming_text or "")
        if source == "messages":
            self._post_tool_text_buffer += incoming
            candidate = self._post_tool_text_buffer
        else:
            candidate = incoming

        replay_normalized = self._normalized_text(replay)
        candidate_normalized = self._normalized_text(candidate)
        if not candidate_normalized:
            return None

        if replay_normalized.startswith(candidate_normalized):
            if message_has_tool_calls or source == "updates_agent":
                self._post_tool_replay_text = ""
                self._post_tool_text_buffer = ""
                return ""
            return None

        remainder = ""
        if candidate.startswith(replay):
            remainder = candidate[len(replay) :]
        else:
            boundary = self._similar_prefix_boundary(replay, candidate)
            replayed_prefix = self._normalized_text(candidate[:boundary]) if boundary is not None else ""
            significant_replay = bool(replayed_prefix) and (
                len(replayed_prefix) >= min(len(replay_normalized), 48)
                or len(replayed_prefix) >= int(len(replay_normalized) * 0.8)
            )
            if boundary is not None and significant_replay:
                remainder = candidate[boundary:]
            else:
                remainder = candidate

        self._post_tool_replay_text = ""
        self._post_tool_text_buffer = ""
        if remainder != candidate:
            logger.debug(
                "Stream removed live post-tool commentary replay replay_len=%s candidate_len=%s",
                len(replay),
                len(candidate),
            )
        return remainder

    def _handle_tool_calls_from_message(self, message: AIMessage | AIMessageChunk, *, source: str) -> None:
        if isinstance(message, AIMessageChunk):
            chunk_calls = list(getattr(message, "tool_call_chunks", []) or [])
            if chunk_calls:
                self._remember_tool_call_chunk_ids(chunk_calls)
                accumulator_key = source or "messages"
                accumulators = self._tool_chunk_accumulators.setdefault(accumulator_key, {})
                processed_any = False
                has_arg_fragment = False
                is_last_chunk = getattr(message, "chunk_position", None) == "last"
                for chunk in chunk_calls:
                    chunk_name = str(chunk.get("name") or "").strip() if isinstance(chunk, dict) else str(getattr(chunk, "name", "") or "").strip()
                    raw_index = chunk.get("index") if isinstance(chunk, dict) else getattr(chunk, "index", None)
                    try:
                        index = int(raw_index)
                    except (TypeError, ValueError):
                        continue
                    raw_id = chunk.get("id") if isinstance(chunk, dict) else getattr(chunk, "id", None)
                    raw_args = chunk.get("args") if isinstance(chunk, dict) else getattr(chunk, "args", None)
                    if str(raw_args or "").strip():
                        has_arg_fragment = True
                    previous = accumulators.get(index)
                    current_chunk = AIMessageChunk(content="", tool_call_chunks=[self._tool_call_chunk_payload(chunk)])
                    if previous is None:
                        gathered = current_chunk
                    else:
                        try:
                            gathered = previous + current_chunk
                        except Exception:
                            gathered = current_chunk
                    accumulators[index] = gathered
                    tool_calls = list(getattr(gathered, "tool_calls", []) or [])
                    if not tool_calls:
                        continue
                    processed_any = True
                    self._process_tool_calls(
                        [
                            dict(tool_call, index=index, name=chunk_name or tool_call.get("name"))
                            for tool_call in tool_calls
                        ],
                        index_order=None,
                        require_args_before_start=True,
                    )
                if not processed_any:
                    self._process_tool_calls(
                        list(getattr(message, "tool_calls", []) or []),
                        index_order=None,
                        require_args_before_start=True,
                    )
                if getattr(message, "chunk_position", None) == "last":
                    self._tool_chunk_accumulators.pop(accumulator_key, None)
                return

        self._process_tool_calls(list(getattr(message, "tool_calls", []) or []), index_order=None)

    @staticmethod
    def _tool_call_chunk_payload(chunk: Any) -> Dict[str, Any]:
        if isinstance(chunk, dict):
            return {
                "name": chunk.get("name"),
                "args": chunk.get("args"),
                "id": chunk.get("id"),
                "index": chunk.get("index"),
            }
        return {
            "name": getattr(chunk, "name", None),
            "args": getattr(chunk, "args", None),
            "id": getattr(chunk, "id", None),
            "index": getattr(chunk, "index", None),
        }

    def _remember_tool_call_chunk_ids(self, chunks: list[Any]) -> None:
        for chunk in chunks:
            if isinstance(chunk, dict):
                raw_index = chunk.get("index")
                raw_id = chunk.get("id")
                raw_name = chunk.get("name")
            else:
                raw_index = getattr(chunk, "index", None)
                raw_id = getattr(chunk, "id", None)
                raw_name = getattr(chunk, "name", None)
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            tool_id = str(raw_id or "").strip()
            self._tool_id_for_index(index, tool_id, str(raw_name or "").strip())

    def _process_tool_calls(
        self,
        tool_calls: list[Dict[str, Any]],
        *,
        index_order: list[int] | None,
        require_args_before_start: bool = False,
    ) -> None:
        for position, tool_call in enumerate(tool_calls):
            normalized = dict(tool_call)
            if index_order is not None and position < len(index_order):
                normalized["index"] = index_order[position]
            self._remember_tool_call(normalized)
            self._emit_tool_started(normalized, require_args=require_args_before_start)

    def _resolve_tool_id(self, tool_id: Any) -> str:
        current = str(tool_id or "").strip()
        seen: set[str] = set()
        while current and current in self._tool_id_aliases and current not in seen:
            seen.add(current)
            current = str(self._tool_id_aliases.get(current) or "").strip()
        return current

    def _synthetic_tool_id(self, index: int) -> str:
        self._synthetic_tool_counter += 1
        return f"stream-tool-{index}-{self._synthetic_tool_counter}"

    def _tool_id_for_index(self, index: int, incoming_id: str, tool_name: str = "") -> str:
        incoming = self._resolve_tool_id(incoming_id)
        existing = self._tool_index_to_id.get(index)
        if existing and existing in self._completed_tool_ids:
            self._tool_index_to_id.pop(index, None)
            self._tool_id_to_index.pop(existing, None)
            existing = ""
        if existing:
            if incoming and incoming != existing:
                self._tool_id_aliases[incoming] = existing
                self._tool_id_to_index[incoming] = index
            return existing
        tool_id = incoming or self._synthetic_tool_id(index)
        self._tool_index_to_id[index] = tool_id
        self._tool_id_to_index[tool_id] = index
        return tool_id

    def _normalize_tool_call_id(self, tool_call: Dict[str, Any]) -> str:
        raw_id = str(tool_call.get("id") or "").strip()
        raw_index = tool_call.get("index")
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            return self._resolve_tool_id(raw_id)
        return self._tool_id_for_index(index, raw_id, str(tool_call.get("name") or "").strip())

    def _mark_tool_completed(self, tool_id: str, raw_tool_id: str = "") -> None:
        completed_indexes: set[int] = set()
        for completed_id in {tool_id, raw_tool_id, self._resolve_tool_id(raw_tool_id)}:
            if completed_id:
                self._completed_tool_ids.pop(completed_id, None)
                self._completed_tool_ids[completed_id] = None
                index = self._tool_id_to_index.get(completed_id)
                if index is not None:
                    completed_indexes.add(index)
        for completed_id in {tool_id, raw_tool_id}:
            index = self._tool_id_to_index.pop(completed_id, None)
            if index is not None and self._tool_index_to_id.get(index) == tool_id:
                self._tool_index_to_id.pop(index, None)
                completed_indexes.add(index)
        for index in completed_indexes:
            self._clear_tool_chunk_accumulators_for_index(index)

    def _clear_tool_chunk_accumulators_for_index(self, index: int) -> None:
        for accumulator_key, accumulators in list(self._tool_chunk_accumulators.items()):
            accumulators.pop(index, None)
            if not accumulators:
                self._tool_chunk_accumulators.pop(accumulator_key, None)

    def _is_active_tool_id(self, tool_id: str) -> bool:
        return bool(tool_id) and (
            tool_id in self.tool_buffer
            or tool_id in self.tool_start_times
            or tool_id in self.printed_tool_ids
        )

    @staticmethod
    def _tool_args_compatible(candidate_args: Any, result_args: Any) -> bool:
        candidate = canonicalize_tool_args(candidate_args)
        result = canonicalize_tool_args(result_args)
        if not candidate or not result:
            return True
        matched = False
        for key, value in candidate.items():
            if key not in result:
                return False
            matched = True
            if result.get(key) != value:
                return False
        return matched

    def _match_active_tool_result(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        normalized_name = str(tool_name or "").strip()
        ordered_ids = list(dict.fromkeys([*self.tool_start_times.keys(), *self.tool_buffer.keys()]))
        candidates: list[str] = []
        for candidate_id in ordered_ids:
            if candidate_id in self._completed_tool_ids:
                continue
            candidate = self.tool_buffer.get(candidate_id, {})
            candidate_name = str(candidate.get("name") or "").strip()
            if normalized_name and candidate_name and normalized_name != candidate_name:
                continue
            if not normalized_name and len(ordered_ids) != 1:
                continue
            if not self._tool_args_compatible(candidate.get("args", {}), tool_args):
                continue
            candidates.append(candidate_id)
        return candidates[0] if candidates else ""

    def _resolve_tool_result_id(self, raw_tool_id: str, tool_name: str, tool_args: Dict[str, Any]) -> str:
        tool_id = self._resolve_tool_id(raw_tool_id)
        if self._is_active_tool_id(tool_id):
            return tool_id
        matched_id = self._match_active_tool_result(tool_name, tool_args)
        if matched_id and raw_tool_id:
            self._tool_id_aliases[raw_tool_id] = matched_id
            index = self._tool_id_to_index.get(matched_id)
            if index is not None:
                self._tool_id_to_index[raw_tool_id] = index
        return matched_id or tool_id

    def _remember_tool_call(self, tool_call: Dict[str, Any]) -> None:
        tool_id = self._normalize_tool_call_id(tool_call)
        if not tool_id:
            return
        if tool_id in self._completed_tool_ids:
            return
        existing = self.tool_buffer.get(tool_id, {})
        existing_name = str(existing.get("name") or "").strip()
        existing_args = existing.get("args", {})
        incoming_args = tool_call.get("args", {})
        merged_args = self._merge_tool_args(existing_args, incoming_args)
        tool_name = (
            str(tool_call.get("name") or "").strip()
            or existing_name
            or "unknown_tool"
        )
        self.tool_buffer[tool_id] = {
            "name": tool_name,
            "args": merged_args,
        }
        self._trim_tool_buffers()

        # If the card is already visible and we learned richer args later (stream races),
        # push a lightweight refresh so UI updates command/args immediately.
        if tool_id in self.printed_tool_ids and (merged_args != existing_args or tool_name != existing_name):
            payload = self._build_tool_event_payload(tool_id, tool_name, merged_args, phase="preparing")
            payload["refresh"] = True
            self._emit("tool_started", payload)

    def _merge_tool_args(self, current_args: Any, incoming_args: Any) -> Dict[str, Any]:
        current = canonicalize_tool_args(current_args)
        incoming = canonicalize_tool_args(incoming_args)
        if not incoming:
            return current

        merged = dict(current)
        for key, value in incoming.items():
            if value is None:
                continue
            if isinstance(value, str):
                if not value.strip():
                    continue
                merged[key] = value
                continue
            if isinstance(value, dict):
                base = merged.get(key)
                merged[key] = self._merge_tool_args(base if isinstance(base, dict) else {}, value)
                continue
            if isinstance(value, list):
                if not value:
                    continue
                merged[key] = value
                continue
            merged[key] = value

        return merged

    def _emit_tool_started(
        self,
        tool_call: Dict[str, Any],
        *,
        require_args: bool = False,
        force: bool = False,
    ) -> None:
        tool_id = self._normalize_tool_call_id(tool_call)
        if not tool_id or tool_id in self.printed_tool_ids or tool_id in self._completed_tool_ids:
            return

        tool_info = self.tool_buffer.get(tool_id, {})
        tool_name = (
            str(tool_call.get("name") or "").strip()
            or str(tool_info.get("name") or "").strip()
            or "unknown_tool"
        )
        if self._normalize_tool_name(tool_name) == "request_user_input":
            return
        tool_args = self._merge_tool_args(tool_info.get("args", {}), tool_call.get("args", {}))
        if require_args and not force and classify_tool_args_state(tool_name, tool_args) != "complete":
            self.tool_buffer[tool_id] = {"name": tool_name, "args": tool_args}
            return
        self.tool_buffer[tool_id] = {"name": tool_name, "args": tool_args}

        self.tool_start_times[tool_id] = time.perf_counter()
        self.printed_tool_ids.add(tool_id)
        self._emit("tool_started", self._build_tool_event_payload(tool_id, tool_name, tool_args, phase="preparing"))

    def _handle_tool_result(self, message: ToolMessage) -> None:
        if is_hidden_internal_message(message):
            notice = get_internal_ui_notice(message)
            if notice and notice != self._last_internal_notice:
                self._last_internal_notice = notice
                self._emit(
                    "summary_notice",
                    {
                        "message": notice,
                        "kind": "agent_internal_notice",
                        "level": "warning",
                    },
                )
            return

        raw_tool_id = str(message.tool_call_id or "").strip()
        message_args = extract_tool_args(message)
        message_name = str(message.name or "").strip() or "unknown_tool"
        if self._normalize_tool_name(message_name) == "request_user_input":
            return
        tool_id = self._resolve_tool_result_id(raw_tool_id, message_name, message_args)
        if tool_id and tool_id in self._completed_tool_ids:
            return
        if raw_tool_id and raw_tool_id in self._completed_tool_ids:
            return

        if tool_id and message_args:
            self._remember_tool_call(
                {
                    "id": tool_id,
                    "name": message_name,
                    "args": message_args,
                }
            )
        if tool_id in self.tool_buffer and tool_id not in self.printed_tool_ids:
            self._emit_tool_started({"id": tool_id, **self.tool_buffer[tool_id]}, force=True)

        tool_info = self.tool_buffer.get(tool_id, {})
        tool_name = (
            str(tool_info.get("name") or "").strip()
            or message_name
            or "unknown_tool"
        )
        tool_args = self._merge_tool_args(tool_info.get("args", {}), message_args)
        if not tool_args:
            self._emit(
                "tool_args_missing",
                {
                    "tool_id": tool_id,
                    "name": tool_name,
                    "message": "No canonical tool args were available when tool result arrived.",
                },
            )

        content_str = stringify_content(message.content)
        is_error = is_tool_message_error(message)
        summary = format_tool_output(tool_name, content_str, is_error)

        elapsed = extract_tool_duration(message)
        start_time = self.tool_start_times.pop(tool_id, None)
        if elapsed is None and start_time is not None:
            elapsed = max(0.0, time.perf_counter() - start_time)

        diff_blocks = [match.group(1).strip() for match in DIFF_REGEX.finditer(content_str)]
        is_repaired_stream_interruption = self._is_repaired_stream_interruption(message, content_str)
        payload = {
            "content": content_str,
            "summary": summary,
            "is_error": is_error,
            "duration": elapsed,
            "diff": diff_blocks[0] if diff_blocks else "",
            "diff_blocks": diff_blocks,
        }
        payload.update(self._build_tool_event_payload(tool_id, tool_name, tool_args, phase="finished", is_error=is_error))
        if is_repaired_stream_interruption:
            payload.update(
                {
                    "display": "Provider stream interrupted",
                    "subtitle": f"{tool_name} did not return a result before the upstream stream ended.",
                    "raw_display": content_str,
                    "interrupted": True,
                    "interruption_reason": "stream_error",
                }
            )
        self._emit("tool_finished", payload)
        if tool_id:
            self.tool_buffer.pop(tool_id, None)
            self._mark_tool_completed(tool_id, raw_tool_id)
            self._trim_tool_buffers()
        if diff_blocks:
            self._emit("tool_diff", {"tool_id": tool_id, "diff": diff_blocks[0], "diff_blocks": diff_blocks})

        if self.tool_start_times:
            self.active_node = "tools"
            self._emit_status()
            self._flush_deferred_assistant_delta()
            return

        self.active_node = "agent"
        self._emit_status(force=True)
        visible = self._visible_full_text or self.full_text
        self._post_tool_replay_text = self._last_nonempty_paragraph(visible).strip()
        deferred_tail = self._begin_post_tool_assistant_section()
        self._post_tool_text_buffer = ""
        if deferred_tail.strip():
            self._handle_agent_message(AIMessage(content=deferred_tail), source="post_tool_deferred")

    @staticmethod
    def _normalize_tool_name(tool_name: Any) -> str:
        return str(tool_name or "").strip().lower()

    @staticmethod
    def _is_repaired_stream_interruption(message: ToolMessage, content: str) -> bool:
        metadata = getattr(message, "additional_kwargs", {}) or {}
        internal = metadata.get("agent_internal") if isinstance(metadata, dict) else None
        if isinstance(internal, dict) and str(internal.get("kind") or "") == "repaired_interrupted_tool_call":
            return True
        return False

    def _emit_interrupted_tool_results(self, reason: str) -> list[Dict[str, Any]]:
        interrupted_payloads: list[Dict[str, Any]] = []
        active_tool_ids = list(self.tool_start_times.keys()) or list(self.tool_buffer.keys())
        for tool_id in active_tool_ids:
            tool_info = self.tool_buffer.get(tool_id, {})
            tool_name = str(tool_info.get("name") or "unknown_tool")
            tool_args = self._merge_tool_args(tool_info.get("args", {}), {})
            elapsed = None
            start_time = self.tool_start_times.pop(tool_id, None)
            if start_time is not None:
                elapsed = time.perf_counter() - start_time
            content = INTERRUPTED_TOOL_MESSAGES.get(reason, DEFAULT_INTERRUPTED_TOOL_MESSAGE)
            payload = {
                "content": content,
                "summary": format_tool_output(tool_name, content, True),
                "is_error": True,
                "duration": elapsed,
                "diff": "",
                "diff_blocks": [],
                "interrupted": True,
                "interruption_reason": reason,
            }
            payload.update(self._build_tool_event_payload(tool_id, tool_name, tool_args, phase="finished", is_error=True))
            self._emit("tool_finished", payload)
            interrupted_payloads.append(payload)
            self.tool_buffer.pop(tool_id, None)
            self.printed_tool_ids.discard(tool_id)
            if tool_id:
                self._completed_tool_ids.pop(tool_id, None)
                self._completed_tool_ids[tool_id] = None
                self._trim_tool_buffers()

        if interrupted_payloads:
            self.active_node = "agent"
            self._emit_status(force=True)
            self._flush_deferred_assistant_delta()
        return interrupted_payloads

    def _status_label(self) -> str:
        agent_label = (
            _AGENT_THINKING_LABEL
            if self._agent_is_thinking
            else _AGENT_WORKING_LABEL
        )
        node_labels = {
            "agent": agent_label,
            "recovery": "Reviewing results",
            "tools": "Running tools",
            "summarize": "Compressing context",
            "approval": "Waiting for approval",
        }
        return node_labels.get(self.active_node, "")

    def _status_phase(self) -> str:
        phase_map = {
            "agent": "working",
            "recovery": "reviewing",
            "tools": "active",
            "summarize": "system",
            "approval": "waiting",
        }
        return phase_map.get(self.active_node, "working")

    def _status_elapsed_text(self) -> str:
        elapsed = self._elapsed_seconds()
        if elapsed < 4.0:
            return ""
        if elapsed < 10.0:
            return f"{elapsed:.1f}s"
        return format_elapsed_seconds(round(elapsed))

    def _build_tool_event_payload(
        self,
        tool_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        *,
        phase: str,
        is_error: bool = False,
    ) -> Dict[str, Any]:
        args_state = classify_tool_args_state(tool_name, tool_args)
        source_kind = self._tool_sources.get(tool_name, "")
        if source_kind == "mcp":
            labels = build_mcp_tool_ui_labels(
                tool_name,
                tool_args,
                phase=phase,
                is_error=is_error,
                server_name=self._mcp_tool_servers.get(tool_name, ""),
            )
        else:
            labels = build_tool_ui_labels(tool_name, tool_args, phase=phase, is_error=is_error)
        display_state = "finished" if phase == "finished" else ("resolved" if args_state == "complete" else "preview")
        if phase == "preparing" and args_state == "complete":
            phase = "running"
        status_hint = labels.get("subtitle") or ("Preparing tool call…" if display_state == "preview" else "")
        return {
            "tool_id": tool_id,
            "name": tool_name,
            "args": tool_args,
            "display": labels.get("title") or tool_name,
            "subtitle": labels.get("subtitle", ""),
            "raw_display": labels.get("raw_display") or format_tool_display(tool_name, tool_args),
            "args_state": args_state,
            "display_state": display_state,
            "phase": phase,
            "status_hint": status_hint,
            "source_kind": labels.get("source_kind", "tool"),
        }

    def _emit_status(self, force: bool = False) -> None:
        label = self._status_label()
        if not label:
            return
        current = (self.active_node, label)
        if not force and current == self._last_status:
            return
        self._last_status = current
        reasoning_logger.debug(
            "status emitted node=%s label=%s thinking=%s phase=%s",
            self.active_node,
            label,
            self._agent_is_thinking,
            self._status_phase(),
        )
        self._emit(
            "status_changed",
            {
                "node": self.active_node,
                "label": label,
                "elapsed": self._elapsed_seconds(),
                "elapsed_text": self._status_elapsed_text(),
                "phase": self._status_phase(),
            },
        )
