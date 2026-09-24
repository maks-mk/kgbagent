import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from core.constants import REFLECTION_PROMPT


_STRUCTURED_TOOL_ERROR_RE = re.compile(r"^\s*ERROR\[[A-Z_]+\]:", re.IGNORECASE)
_LEGACY_INTERRUPTED_TOOL_ERROR_RE = re.compile(
    r"^\s*Error:\s*Execution interrupted\b",
    re.IGNORECASE,
)
_TOOL_MESSAGE_ERROR_STATUS = "error"
_TOOL_MESSAGE_SUCCESS_STATUS = "success"


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
