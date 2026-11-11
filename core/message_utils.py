import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from core.constants import REFLECTION_PROMPT


_STRUCTURED_TOOL_ERROR_RE = re.compile(r"^\s*ERROR\[[A-Z_]+\]:", re.IGNORECASE)
_LEGACY_INTERRUPTED_TOOL_ERROR_RE = re.compile(
    r"^\s*Error:\s*Execution interrupted\b",
    re.IGNORECASE,
)
_TOOL_MESSAGE_ERROR_STATUS = "error"
_TOOL_MESSAGE_SUCCESS_STATUS = "success"


def recent_tool_calls(
    messages: list[Any], history_window: int, *, exclude_latest: bool = False
) -> list[dict[str, Any]]:
    """Collect recent calls, optionally excluding the pending assistant message."""
    end = max(0, len(messages) - int(exclude_latest))
    start = max(0, end - history_window) if history_window > 0 else 0
    history = messages[start:end]
    return [
        tool_call
        for message in reversed(history)
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]


def iter_message_tool_calls(message: Any):
    """Yield structured calls from LangChain fields and native content blocks."""
    seen = set()
    for call in getattr(message, "tool_calls", None) or []:
        if isinstance(call, dict):
            normalized = dict(call)
            call_id = str(normalized.get("id") or normalized.get("call_id") or "").strip()
            if call_id and call_id in seen:
                continue
            if call_id:
                seen.add(call_id)
            yield normalized
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") not in {"function_call", "tool_use", "tool_call"}:
            continue
        call_id = str(block.get("call_id") or block.get("id") or "").strip()
        if call_id and call_id in seen:
            continue
        arguments = block.get("arguments", block.get("input", block.get("args", {})))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                pass
        if call_id:
            seen.add(call_id)
        yield {"id": call_id, "name": str(block.get("name") or ""), "args": arguments}


def _stringify_content_item(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if "text" in item:
            return str(item.get("text") or "")
        if "refusal" in item:
            return str(item.get("refusal") or "")
        if "content" in item:
            return _stringify_content_item(item.get("content"))
        return ""
    if isinstance(item, list):
        return "".join(_stringify_content_item(part) for part in item)
    return str(item)


def stringify_content(content: Any) -> str:
    return _stringify_content_item(content)


# Reasoning/thinking blocks are provider-private: they stay in the transcript for
# protocol round-trips but they are never user-visible output.
_NON_VISIBLE_CONTENT_TYPES = frozenset(
    {
        "analysis",
        "analysis_content",
        "reasoning",
        "reasoning_content",
        "reasoning_delta",
        "reasoning_summary",
        "reasoning_text",
        "redacted_thinking",
        "summary_text",
        "thinking",
        "thinking_content",
        "thought",
    }
)
_NON_VISIBLE_CONTENT_TYPE_PREFIXES = ("analysis.", "reasoning.", "thinking.", "thought.")

_THINK_TAG_PATTERN = r"(think|thought|thinking)"
INLINE_THOUGHT_BLOCK_RE = re.compile(rf"<{_THINK_TAG_PATTERN}\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
INLINE_THOUGHT_OPEN_RE = re.compile(rf"<{_THINK_TAG_PATTERN}\b[^>]*>", re.IGNORECASE)
INLINE_THOUGHT_CLOSE_RE = re.compile(rf"</{_THINK_TAG_PATTERN}\s*>", re.IGNORECASE)
_INLINE_THOUGHT_CLOSE_PREFIX_RE = re.compile(rf"^.*?</{_THINK_TAG_PATTERN}\s*>\s*", re.IGNORECASE | re.DOTALL)
_INLINE_THOUGHT_UNCLOSED_RE = re.compile(rf"<{_THINK_TAG_PATTERN}\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)


def strip_inline_thought_blocks(text: str) -> str:
    """Strip complete inline thoughts after joining text; streams need a buffer."""
    cleaned = INLINE_THOUGHT_BLOCK_RE.sub("", text)
    cleaned = _INLINE_THOUGHT_CLOSE_PREFIX_RE.sub("", cleaned)
    return _INLINE_THOUGHT_UNCLOSED_RE.sub("", cleaned)


def _visible_text_content_item(item: Any, *, include_refusal: bool) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)):
        return "".join(_visible_text_content_item(part, include_refusal=include_refusal) for part in item)
    if isinstance(item, BaseMessage):
        # LangChain wraps unrecognised provider blocks as non_standard/value.
        # The same traversal handles those blocks and legacy raw payloads.
        text = _visible_text_content_item(item.content_blocks, include_refusal=include_refusal)
        if text:
            return text
        return _visible_text_content_item(item.additional_kwargs.get("content_blocks"), include_refusal=include_refusal)
    if isinstance(item, dict):
        item_type = str(item.get("type") or "").strip().lower()
        if bool(item.get("thought")) or item_type in _NON_VISIBLE_CONTENT_TYPES:
            return ""
        if item_type.startswith(_NON_VISIBLE_CONTENT_TYPE_PREFIXES):
            return ""
        if item_type == "refusal" and not include_refusal:
            return ""
        for key in ("text", "output_text", "refusal", "content", "answer", "response", "final", "final_text"):
            if key in item and (key != "refusal" or include_refusal):
                text = _visible_text_content_item(item[key], include_refusal=include_refusal)
                if text:
                    return text
        return "".join(
            _visible_text_content_item(item[key], include_refusal=include_refusal)
            for key in ("parts", "items", "content_blocks", "message", "messages", "data", "value")
            if key in item
        )
    for key in ("content_blocks", "content", "text"):
        try:
            value = getattr(item, key)
        except Exception:
            continue
        if value is not None and not callable(value):
            text = _visible_text_content_item(value, include_refusal=include_refusal)
            if text:
                return text
    metadata = getattr(item, "additional_kwargs", None)
    if isinstance(metadata, dict):
        return _visible_text_content_item(metadata.get("content_blocks"), include_refusal=include_refusal)
    return ""


def visible_text_content(content: Any, *, strip_inline: bool = False, include_refusal: bool = True) -> str:
    """Return user-visible content; title generation can exclude refusal blocks."""
    text = _visible_text_content_item(content, include_refusal=include_refusal)
    return strip_inline_thought_blocks(text) if strip_inline else text


def compact_text(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    marker = "... [truncated]"
    if limit <= len(marker):
        # Too small to fit the marker; hard-truncate so the result never exceeds the limit.
        return compact[: max(0, limit)]
    return compact[: limit - len(marker)] + marker


def is_error_text(text: Any) -> bool:
    normalized = stringify_content(text).strip()
    return bool(
        _STRUCTURED_TOOL_ERROR_RE.match(normalized)
        or _LEGACY_INTERRUPTED_TOOL_ERROR_RE.match(normalized)
    )


def tool_message_status(message: ToolMessage) -> str:
    status = str(getattr(message, "status", "") or "").strip().lower()
    if _STRUCTURED_TOOL_ERROR_RE.match(stringify_content(message.content).strip()):
        return _TOOL_MESSAGE_ERROR_STATUS
    if status in {_TOOL_MESSAGE_SUCCESS_STATUS, _TOOL_MESSAGE_ERROR_STATUS}:
        return status
    return _TOOL_MESSAGE_ERROR_STATUS if is_error_text(message.content) else _TOOL_MESSAGE_SUCCESS_STATUS


def is_tool_message_error(message: ToolMessage) -> bool:
    return tool_message_status(message) == _TOOL_MESSAGE_ERROR_STATUS


def is_internal_retry_message(message: BaseMessage) -> bool:
    """True for internal retry scaffolding injected by recovery, not a real user turn."""
    if not isinstance(message, HumanMessage):
        return False
    metadata = getattr(message, "additional_kwargs", {}) or {}
    internal = metadata.get("agent_internal")
    return isinstance(internal, dict) and internal.get("kind") == "retry_instruction"


def is_user_turn_message(message: BaseMessage) -> bool:
    """True for a real user turn: a non-empty human message that is not internal scaffolding.

    Internal retry hints are stripped from the outbound model context, so they never
    qualify as a turn boundary.
    """
    if not isinstance(message, HumanMessage) or is_internal_retry_message(message):
        return False
    content = stringify_content(message.content).strip()
    return bool(content) and content != REFLECTION_PROMPT


def current_turn_id(state: dict[str, Any], messages: list[BaseMessage]) -> int:
    derived_turn_id = 0
    for message in messages:
        if is_user_turn_message(message):
            derived_turn_id += 1
    return max(int(state.get("turn_id", 0) or 0), derived_turn_id)


def last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    return next((message for message in reversed(messages) if isinstance(message, AIMessage)), None)
