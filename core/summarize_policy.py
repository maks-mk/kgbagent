from __future__ import annotations

import json
import logging
from typing import Callable, List, Optional

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from core.message_utils import is_user_turn_message, stringify_content


IsInternalRetry = Callable[[BaseMessage], bool]

logger = logging.getLogger("agent")

# ---------------------------------------------------------------------------
# Tiktoken — lazy initialization.
# The encoder is created once and reused for the lifetime of the process.
# cl100k_base is the GPT-4/GPT-3.5 encoding. It is not ideal for Gemini, but
# its accuracy is within about ±15%, which is far better than a char heuristic.
# ---------------------------------------------------------------------------

_TIKTOKEN_ENCODER = None
_TIKTOKEN_AVAILABLE: Optional[bool] = None  # None = not checked yet


def _get_encoder():
    global _TIKTOKEN_ENCODER, _TIKTOKEN_AVAILABLE
    if _TIKTOKEN_AVAILABLE is True:
        return _TIKTOKEN_ENCODER
    if _TIKTOKEN_AVAILABLE is False:
        return None
    # First call — try to initialize it
    try:
        import tiktoken
        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        _TIKTOKEN_AVAILABLE = True
        logger.debug("tiktoken encoder initialised (cl100k_base).")
    except Exception as exc:
        _TIKTOKEN_AVAILABLE = False
        logger.warning(
            "tiktoken unavailable, falling back to char-based token estimate: %s", exc
        )
    return _TIKTOKEN_ENCODER


# Overhead for each message in the Chat Completions API:
# role token + separators = ~4 tokens (per the OpenAI tiktoken spec).
_MESSAGE_OVERHEAD_TOKENS = 4


def _count_tokens_tiktoken(text: str) -> int:
    """Count tokens with tiktoken. Called only when the encoder is available."""
    enc = _get_encoder()
    if enc is None:
        return 0
    try:
        return len(enc.encode(text))
    except Exception:
        return 0


def _count_tokens_fallback(text: str) -> int:
    """Character heuristic: about 3 characters per token.
    3 is more accurate than 2: a compromise between ru (~2 chars/token) and en (~4 chars/token)."""
    return max(1, len(text) // 3)


# ---------------------------------------------------------------------------
# Per-message token cache.
#
# Messages are immutable after creation (LangChain guarantees this for
# AIMessage, HumanMessage, ToolMessage).  The cache key combines the message
# id with a cheap hash of the stringified content + tool_calls, so that even
# if a message is replaced with a new instance bearing the same id (e.g. after
# recovery rewriting), the cache will miss and recompute correctly.
#
# The cache is bounded to avoid unbounded memory growth in very long sessions.
# ---------------------------------------------------------------------------

_MESSAGE_TOKEN_CACHE: dict[tuple[str | None, int], int] = {}
_MESSAGE_TOKEN_CACHE_LIMIT = 512


def _message_cache_key(message: BaseMessage) -> tuple[str | None, int]:
    content_str = stringify_content(message.content)
    tool_calls = getattr(message, "tool_calls", None) or []
    tool_calls_str = str(tool_calls) if tool_calls else ""
    combined = content_str + "\x00" + tool_calls_str
    return (getattr(message, "id", None), hash(combined))


def _count_single_message_tokens(message: BaseMessage, *, use_tiktoken: bool) -> int:
    key = _message_cache_key(message)
    cached = _MESSAGE_TOKEN_CACHE.get(key)
    if cached is not None:
        return cached

    count_fn = _count_tokens_tiktoken if use_tiktoken else _count_tokens_fallback
    total = 0
    content = stringify_content(message.content)
    total += count_fn(content)

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        total += count_fn(str(tool_calls))

    if use_tiktoken:
        total += _MESSAGE_OVERHEAD_TOKENS

    if len(_MESSAGE_TOKEN_CACHE) >= _MESSAGE_TOKEN_CACHE_LIMIT:
        # Evict oldest entry (dict preserves insertion order in Python 3.7+).
        _MESSAGE_TOKEN_CACHE.pop(next(iter(_MESSAGE_TOKEN_CACHE)))
    _MESSAGE_TOKEN_CACHE[key] = total
    return total


def estimate_tokens(messages: List[BaseMessage]) -> int:
    """Estimate the total token count for a list of messages.

    Algorithm:
    - If tiktoken is available, use cl100k_base + per-message overhead.
    - Otherwise, use a character heuristic with a divisor of 3.
    - Per-message results are cached to avoid recomputation for unchanged
      messages across turns.
    """
    use_tiktoken = _get_encoder() is not None

    total = 0
    for message in messages:
        total += _count_single_message_tokens(message, use_tiktoken=use_tiktoken)

    return total


def estimate_summary_tokens(summary: str) -> int:
    summary_text = str(summary or "").strip()
    if not summary_text:
        return 0
    return estimate_tokens([SystemMessage(content=f"<memory>\n{summary_text}\n</memory>")])


_MEMORY_TRUNCATION_MARKER = "... [memory truncated]"


def _hard_cut_summary(text: str, max_tokens: int) -> str:
    """Shrink a single oversized memory item until it fits the budget."""
    cut = str(text or "").strip()
    while cut and estimate_summary_tokens(f"{cut} {_MEMORY_TRUNCATION_MARKER}") > max_tokens:
        cut = cut[: int(len(cut) * 0.8)].rstrip()
    return f"{cut} {_MEMORY_TRUNCATION_MARKER}" if cut else _MEMORY_TRUNCATION_MARKER


def truncate_summary_to_token_budget(summary: str, max_tokens: int) -> str:
    """Drop trailing memory items until the memory estimate fits ``max_tokens``.

    Memory is written as importance-ordered bullets, so cutting from the end keeps the
    active task, blockers, and next steps. Used as the deterministic guarantee behind the
    model-driven memory fold, which can overshoot the requested size.
    """
    text = str(summary or "").strip()
    budget = int(max_tokens or 0)
    if not text or budget <= 0 or estimate_summary_tokens(text) <= budget:
        return text

    kept: List[str] = []
    for line in (line for line in text.splitlines() if line.strip()):
        candidate = "\n".join(kept + [line, _MEMORY_TRUNCATION_MARKER])
        if estimate_summary_tokens(candidate) > budget:
            break
        kept.append(line)
    if kept:
        return "\n".join(kept + [_MEMORY_TRUNCATION_MARKER])
    first_line = next((line for line in text.splitlines() if line.strip()), text)
    return _hard_cut_summary(first_line, budget)


def estimate_context_tokens(messages: List[BaseMessage], *, reserved_tokens: int = 0) -> int:
    """Estimate the model context budget used by message history plus fixed runtime overhead.

    The message list does not include system/developer prompts, tool schemas, and provider
    wrapper fields. The reserve keeps auto-summary progress closer to provider-reported
    prompt/input tokens without pretending to know every provider tokenizer exactly.
    """
    try:
        reserve = max(0, int(reserved_tokens or 0))
    except (TypeError, ValueError):
        reserve = 0
    if not messages:
        return 0
    return estimate_tokens(messages) + reserve


# ---------------------------------------------------------------------------
# The rest is unchanged
# ---------------------------------------------------------------------------

def _soft_summary_margin(threshold: int, *, has_summary: bool) -> int:
    base = max(800, int(threshold * 0.15))
    if has_summary:
        return max(base, int(threshold * 0.35))
    return base


def summary_trigger_tokens(threshold: int, *, has_summary: bool = False) -> int:
    """Context estimate at which compaction is no longer delayed by the soft guards."""
    threshold = int(threshold or 0)
    if threshold <= 0:
        return 0
    return threshold + _soft_summary_margin(threshold, has_summary=has_summary)


def summary_progress_ratio(
    estimated_tokens: int,
    *,
    threshold: int,
    baseline_tokens: int = 0,
    has_summary: bool = False,
) -> float:
    """Share of the compactable context budget already used.

    ``baseline_tokens`` (fixed reserve plus compressed memory) survives every compaction,
    so it is excluded from both sides of the ratio: right after a compaction the indicator
    reflects the retained history instead of staying pegged at the fixed overhead.
    """
    trigger = summary_trigger_tokens(threshold, has_summary=has_summary)
    if trigger <= 0:
        return 0.0
    baseline = max(0, min(int(baseline_tokens or 0), trigger))
    span = trigger - baseline
    if span <= 0:
        return 1.0
    used = max(0, int(estimated_tokens or 0) - baseline)
    return max(0.0, min(1.0, used / span))


def summary_remaining_ratio(estimated_tokens: int, *, threshold: int) -> float:
    """Return the fraction of the configured summary threshold that remains."""
    threshold = int(threshold or 0)
    if threshold <= 0:
        return 0.0
    estimated = max(0, int(estimated_tokens or 0))
    return max(0.0, min(1.0, 1.0 - (estimated / threshold)))


def should_summarize(
    messages: List[BaseMessage],
    *,
    threshold: int,
    keep_last: int,
    has_summary: bool = False,
    reserved_tokens: int = 0,
    allow_tool_round_boundaries: bool = False,
) -> bool:
    threshold = int(threshold or 0)
    if threshold <= 0:
        return False

    estimated = estimate_context_tokens(messages, reserved_tokens=reserved_tokens)
    if estimated <= threshold:
        return False

    boundary = choose_summary_boundary(
        messages,
        keep_last=keep_last,
        threshold=threshold,
        reserved_tokens=reserved_tokens,
        allow_tool_round_boundaries=allow_tool_round_boundaries,
    )
    summarizable = messages[:boundary]
    if not summarizable:
        return False

    summarizable_human_turns = sum(1 for message in summarizable if isinstance(message, HumanMessage))
    soft_threshold = summary_trigger_tokens(threshold, has_summary=has_summary)
    min_summarizable_messages = max(6, int(keep_last or 0) + 2)

    if estimated < soft_threshold:
        if len(summarizable) < min_summarizable_messages:
            return False
        if summarizable_human_turns < 2:
            return False

    return True


def _tool_round_boundaries(messages: List[BaseMessage], *, user_boundaries: set[int]) -> List[int]:
    """Indexes that start a completed tool round: an AI message with tool_calls whose
    results are already present. Everything before such an index is finished work, so it
    can be compacted mid-run without orphaning a tool result (the AI/tool-call pair and its
    ToolMessage stay intact on the retained side)."""
    result: List[int] = []
    for index, message in enumerate(messages):
        if index in user_boundaries:
            continue
        if not isinstance(message, (AIMessage, AIMessageChunk)):
            continue
        tool_calls = list(getattr(message, "tool_calls", []) or [])
        if not tool_calls:
            continue
        call_ids = {str(tc.get("id") or "").strip() for tc in tool_calls if str(tc.get("id") or "").strip()}
        if not call_ids:
            continue
        # Every tool call must already have a matching ToolMessage before this round is safe to cut.
        present_ids = {
            str(getattr(m, "tool_call_id", "") or "").strip()
            for m in messages[index + 1 :]
            if getattr(m, "tool_call_id", None) is not None
        }
        if call_ids.issubset(present_ids):
            result.append(index)
    return result


def choose_summary_boundary(
    messages: List[BaseMessage],
    *,
    keep_last: int,
    threshold: int = 0,
    reserved_tokens: int = 0,
    allow_tool_round_boundaries: bool = False,
) -> int:
    """Pick the compaction cut: ``messages[:boundary]`` is summarized and removed.

    Only real user turns are valid cut points, so the remaining history always starts
    on a user message and never keeps a tool result whose tool call was removed.

    When ``allow_tool_round_boundaries`` is set (mid-run compaction between tool
    results and the next model comment), the start of a completed tool round — an
    AI message with tool_calls whose results are already present — is also a valid
    cut point before the active user turn: everything before it is finished, and
    AI/tool-call pairs stay intact. The active user message remains as the anchor
    for the final visible transcript.

    Preference order:
    1. the newest user turn that still keeps at least ``keep_last`` messages;
    2. later user turns, when the retained history would still exceed ``threshold``;
    3. ``0`` (no compaction) when no user turn can be cut — a growing context is safer
       than a history that violates the provider tool-call contract.
    """
    user_indexes = [index for index, message in enumerate(messages) if is_user_turn_message(message)]
    boundaries = [index for index in user_indexes if index > 0]
    if allow_tool_round_boundaries:
        # The latest real user message belongs to the active turn. Removing it
        # leaves the retained AI/tool messages without a transcript turn, so a
        # final session refresh appears to erase the whole chat. Mid-run
        # compaction may only cut before the active user message.
        active_user_index = user_indexes[-1] if user_indexes else len(messages)
        tool_boundaries = _tool_round_boundaries(messages, user_boundaries=set(boundaries))
        # A user boundary at ``active_user_index`` removes only older turns and
        # is therefore safe. Tool-round boundaries at or after that index belong
        # to the active turn and would remove its user-message anchor.
        tool_boundaries = [index for index in tool_boundaries if index < active_user_index]
        boundaries = sorted(set(boundaries + tool_boundaries))
    if not boundaries:
        return 0

    keep_from = max(0, len(messages) - max(0, int(keep_last or 0)))
    within_keep_last = [index for index in boundaries if index <= keep_from]
    candidates = (
        [index for index in boundaries if index >= within_keep_last[-1]]
        if within_keep_last
        else boundaries
    )

    threshold = int(threshold or 0)
    if threshold > 0:
        for index in candidates:
            retained_tokens = estimate_context_tokens(messages[index:], reserved_tokens=reserved_tokens)
            if retained_tokens <= threshold:
                return index
        return candidates[-1]
    return candidates[0]


def _compact_for_summary(text: str, *, limit: int = 500) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"


def _format_tool_calls_for_summary(message: BaseMessage) -> str:
    if not isinstance(message, (AIMessage, AIMessageChunk)):
        return ""
    raw_tool_calls = list(getattr(message, "tool_calls", []) or [])
    if not raw_tool_calls:
        return ""

    parts: List[str] = []
    for tool_call in raw_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_name = str(tool_call.get("name") or "tool").strip() or "tool"
        tool_args = tool_call.get("args")
        try:
            rendered_args = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
        except TypeError:
            rendered_args = str(tool_args)
        parts.append(f"{tool_name}({rendered_args})")
    return _compact_for_summary("; ".join(parts), limit=320)


def format_history_for_summary(
    messages: List[BaseMessage],
    *,
    is_internal_retry: IsInternalRetry,
) -> str:
    parts: List[str] = []
    for message in messages:
        if isinstance(message, HumanMessage) and is_internal_retry(message):
            continue
        rendered = _compact_for_summary(stringify_content(message.content), limit=500)
        tool_call_text = _format_tool_calls_for_summary(message)
        if isinstance(message, ToolMessage):
            tool_name = str(getattr(message, "name", "") or "tool").strip() or "tool"
            header = f"{message.type}({tool_name})"
        else:
            header = message.type

        segments: List[str] = []
        if tool_call_text:
            segments.append(f"tool_calls={tool_call_text}")
        if rendered:
            segments.append(f"content={rendered}")
        if not segments:
            segments.append("<empty>")
        parts.append(f"{header}: {' | '.join(segments)}")
    return "\n".join(parts)
