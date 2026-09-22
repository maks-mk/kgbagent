from __future__ import annotations

import json
import logging
from typing import Callable, List, Optional

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from core.logging_config import SensitiveDataFilter
from core.message_utils import is_user_turn_message, stringify_content
from core.tool_output_compressor import iter_diagnostic_lines


IsInternalRetry = Callable[[BaseMessage], bool]

logger = logging.getLogger("agent")

# Encoders are cached per model; unknown/non-OpenAI models use an approximation.
_ENCODERS: dict[str, object] = {}


def token_model_name(config) -> str:
    provider = getattr(config, "provider", "")
    return str(getattr(config, f"{provider}_model", "") or "") if provider == "openai" else ""


def _get_encoder(model_name: str = ""):
    if model_name in _ENCODERS:
        return _ENCODERS[model_name]
    try:
        import tiktoken
        try:
            encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoder = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        logger.warning("tiktoken unavailable; using character estimate: %s", type(exc).__name__)
        encoder = None
    _ENCODERS[model_name] = encoder
    return encoder


_MESSAGE_OVERHEAD_TOKENS = 4


def estimate_text_tokens(text: str, model_name: str = "") -> int:
    encoder = _get_encoder(model_name)
    if encoder is not None:
        try:
            # User/tool text may contain literal special-token spellings.
            return len(encoder.encode_ordinary(text))
        except Exception:
            logger.debug("Tokenization failed; using character estimate.")
    return _count_tokens_fallback(text)


def _count_tokens_fallback(text: str) -> int:
    """Character heuristic: about 3 characters per token.
    3 is more accurate than 2: a compromise between ru (~2 chars/token) and en (~4 chars/token)."""
    return (len(text) + 2) // 3


# ---------------------------------------------------------------------------
# Per-message token cache.
#
# Include the current content and tokenizer so edits and model switches invalidate
# cached estimates. LangChain messages can be modified after creation.
#
# The cache is bounded to avoid unbounded memory growth in very long sessions.
# ---------------------------------------------------------------------------

_MESSAGE_TOKEN_CACHE: dict[tuple[str | None, int, str, bool], int] = {}
_MESSAGE_TOKEN_CACHE_LIMIT = 512


def _message_cache_key(message: BaseMessage) -> tuple[str | None, int]:
    content_str = stringify_content(message.content)
    tool_calls = getattr(message, "tool_calls", None) or []
    tool_calls_str = str(tool_calls) if tool_calls else ""
    combined = content_str + "\x00" + tool_calls_str
    return (getattr(message, "id", None), hash(combined))


def _count_single_message_tokens(message: BaseMessage, *, use_tiktoken: bool, model_name: str = "") -> int:
    key = (*_message_cache_key(message), model_name, use_tiktoken)
    cached = _MESSAGE_TOKEN_CACHE.get(key)
    if cached is not None:
        return cached

    count_fn = (lambda text: estimate_text_tokens(text, model_name)) if use_tiktoken else _count_tokens_fallback
    total = 0
    content = stringify_content(message.content)
    total += count_fn(content)

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        total += count_fn(str(tool_calls))

    total += _MESSAGE_OVERHEAD_TOKENS

    if len(_MESSAGE_TOKEN_CACHE) >= _MESSAGE_TOKEN_CACHE_LIMIT:
        # Evict oldest entry (dict preserves insertion order in Python 3.7+).
        _MESSAGE_TOKEN_CACHE.pop(next(iter(_MESSAGE_TOKEN_CACHE)))
    _MESSAGE_TOKEN_CACHE[key] = total
    return total


def estimate_tokens(messages: List[BaseMessage], *, model_name: str = "") -> int:
    """Estimate the total token count for a list of messages.

    Algorithm:
    - Use the model encoding (cl100k_base for unknown models) + message overhead.
    - Otherwise, use a character heuristic with a divisor of 3.
    - Per-message results are cached to avoid recomputation for unchanged
      messages across turns.
    """
    use_tiktoken = _get_encoder(model_name) is not None

    total = 0
    for message in messages:
        total += _count_single_message_tokens(message, use_tiktoken=use_tiktoken, model_name=model_name)

    return total


def estimate_summary_tokens(summary: str, *, model_name: str = "") -> int:
    summary_text = str(summary or "").strip()
    if not summary_text:
        return 0
    return estimate_tokens([SystemMessage(content=f"<memory>\n{summary_text}\n</memory>")], model_name=model_name)


_MEMORY_TRUNCATION_MARKER = "... [memory truncated]"


def _hard_cut_summary(text: str, max_tokens: int, *, model_name: str = "") -> str:
    """Shrink a single oversized memory item until it fits the budget."""
    cut = str(text or "").strip()
    while cut and estimate_summary_tokens(f"{cut} {_MEMORY_TRUNCATION_MARKER}", model_name=model_name) > max_tokens:
        cut = cut[: int(len(cut) * 0.8)].rstrip()
    return f"{cut} {_MEMORY_TRUNCATION_MARKER}" if cut else _MEMORY_TRUNCATION_MARKER


def truncate_summary_to_token_budget(summary: str, max_tokens: int, *, model_name: str = "") -> str:
    """Drop trailing memory items until the memory estimate fits ``max_tokens``.

    Memory is written as importance-ordered bullets, so cutting from the end keeps the
    active task, blockers, and next steps. Used as the deterministic guarantee behind the
    model-driven memory fold, which can overshoot the requested size.
    """
    text = str(summary or "").strip()
    budget = int(max_tokens or 0)
    if not text or budget <= 0 or estimate_summary_tokens(text, model_name=model_name) <= budget:
        return text

    kept: List[str] = []
    for line in (line for line in text.splitlines() if line.strip()):
        candidate = "\n".join(kept + [line, _MEMORY_TRUNCATION_MARKER])
        if estimate_summary_tokens(candidate, model_name=model_name) > budget:
            break
        kept.append(line)
    if kept:
        return "\n".join(kept + [_MEMORY_TRUNCATION_MARKER])
    first_line = next((line for line in text.splitlines() if line.strip()), text)
    return _hard_cut_summary(first_line, budget, model_name=model_name)


def estimate_context_tokens(messages: List[BaseMessage], *, reserved_tokens: int = 0, model_name: str = "") -> int:
    """Estimate history plus the caller-supplied non-history budget.

    The caller includes locally counted instructions, tool schemas and memory,
    plus a configurable safety margin for provider-specific formatting.
    """
    try:
        reserve = max(0, int(reserved_tokens or 0))
    except (TypeError, ValueError):
        reserve = 0
    if not messages:
        return 0
    return estimate_tokens(messages, model_name=model_name) + reserve


# ---------------------------------------------------------------------------
# Summary trigger and compaction boundaries
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


def summary_fill_ratio(estimated_tokens: int, *, threshold: int, baseline_tokens: int = 0) -> float:
    """Used share of the compactable budget, scaled to the hard summarization threshold.

    ``baseline_tokens`` (fixed prompt/tool overhead plus compressed memory) is excluded
    from both the used amount and the budget: it survives every compaction, so keeping it
    out makes the indicator drop right after a compaction and reach ``1.0`` exactly when
    ``should_summarize`` would fire (``estimated_tokens`` reaches ``threshold``).
    """
    threshold = int(threshold or 0)
    if threshold <= 0:
        return 0.0
    baseline = max(0, min(int(baseline_tokens or 0), threshold))
    span = threshold - baseline
    if span <= 0:
        # No room left for history once the fixed overhead is subtracted.
        return 1.0
    used = max(0, int(estimated_tokens or 0) - baseline)
    return max(0.0, min(1.0, used / span))


def should_summarize(
    messages: List[BaseMessage],
    *,
    threshold: int,
    keep_last: int,
    has_summary: bool = False,
    reserved_tokens: int = 0,
    allow_tool_round_boundaries: bool = False,
    model_name: str = "",
) -> bool:
    threshold = int(threshold or 0)
    if threshold <= 0:
        return False

    estimated = estimate_context_tokens(messages, reserved_tokens=reserved_tokens, model_name=model_name)
    if estimated <= threshold:
        return False

    boundary = choose_summary_boundary(
        messages,
        keep_last=keep_last,
        threshold=threshold,
        reserved_tokens=reserved_tokens,
        allow_tool_round_boundaries=allow_tool_round_boundaries,
        model_name=model_name,
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
    model_name: str = "",
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
            retained_tokens = estimate_context_tokens(messages[index:], reserved_tokens=reserved_tokens, model_name=model_name)
            if retained_tokens <= threshold:
                return index
        return candidates[-1]
    return candidates[0]


def _compact_for_summary(text: str, *, limit: int = 500, preserve_tail: bool = False) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    if preserve_tail:
        marker = " ... [truncated] ... "
        if limit <= len(marker):
            return normalized[:max(0, limit)]
        available = limit - len(marker)
        # Tool outcomes and diagnostics often follow lengthy progress output.
        head = available * 3 // 10
        tail = available - head
        return normalized[:head] + marker + normalized[-tail:]
    return normalized[:limit] + "... [truncated]"


def _render_summary_excerpt(text: str, spans: list[tuple[int, int]]) -> str:
    """Render verbatim source spans in order; every gap is explicitly marked."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        start, end = max(0, start), min(len(text), end)
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    marker = "\n... [truncated] ...\n"
    parts = []
    cursor = 0
    for start, end in merged:
        if start > cursor:
            parts.append(marker)
        parts.append(text[start:end])
        cursor = end
    if cursor < len(text):
        parts.append(marker)
    return "".join(parts)


def _format_tool_content_for_summary(text: str, *, tool_name: str, limit: int = 500) -> str:
    # Only shell output is treated as diagnostics. ERROR in a file or document
    # may be an example, not an observed execution failure.
    diagnostics = list(iter_diagnostic_lines(text, include_outcomes=True)) if tool_name == "cli_exec" else []
    if not diagnostics:
        return _compact_for_summary(text, limit=limit, preserve_tail=True)
    if len(text) <= limit:
        return text

    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    # Reserve 40% for operation context, split like the head/tail fallback;
    # diagnostics and omission markers get the rest before any extra context.
    context_budget = max(0, limit) * 2 // 5
    head_budget = context_budget * 3 // 10
    tail_budget = context_budget - head_budget
    spans = [(0, min(head_budget, len(lines[0]))), (len(text) - min(tail_budget, len(lines[-1])), len(text))]
    seen = set()
    kept_lines = []
    for index, line in diagnostics:
        if line.strip() in seen:
            continue
        seen.add(line.strip())
        candidate = spans + [(offsets[index], offsets[index + 1])]
        if len(_render_summary_excerpt(text, candidate)) <= limit:
            spans = candidate
            kept_lines.append(index)
    if not kept_lines:
        return _compact_for_summary(text, limit=limit, preserve_tail=True)

    # Once the diagnostic lines fit, retain adjacent source context (notably
    # traceback frames) when there is room, without flattening indentation.
    for index in kept_lines:
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(lines):
                candidate = spans + [(offsets[neighbor], offsets[neighbor + 1])]
                if len(_render_summary_excerpt(text, candidate)) <= limit:
                    spans = candidate

    # Spend the remaining budget on context, retaining the original head/tail
    # preference. Re-rendering merges overlaps rather than duplicating facts.
    remaining = max(0, limit - len(_render_summary_excerpt(text, spans)))
    head_extra = remaining * 3 // 10
    spans += [(0, spans[0][1] + head_extra), (spans[1][0] - (remaining - head_extra), len(text))]
    rendered = _render_summary_excerpt(text, spans)
    return rendered if len(rendered) <= limit else _compact_for_summary(text, limit=limit, preserve_tail=True)


_SUMMARY_ARG_PRIORITY = ("command", "path", "query", "queries", "url", "urls", "pattern")
_SUMMARY_BODY_ARGS = frozenset({"content", "old_string", "new_string"})


def _summary_safe_args(value, *, key: str = ""):
    # Reuse the runtime's masking rules before selection/truncation. Do not
    # partially expose explicitly sensitive fields, including nested values.
    if key.strip().lower() in SensitiveDataFilter.SENSITIVE_FIELD_NAMES:
        return "<redacted>"
    if isinstance(value, dict):
        return {k: _summary_safe_args(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summary_safe_args(item) for item in value]
    if isinstance(value, str):
        return SensitiveDataFilter._sanitize_string(value)
    return value


def _format_tool_calls_for_summary(message: BaseMessage, refs: list[str]) -> str:
    if not isinstance(message, (AIMessage, AIMessageChunk)):
        return ""
    calls = [call for call in (message.tool_calls or []) if isinstance(call, dict)]
    if not calls:
        return ""

    headers = []
    arguments = []
    for call, ref in zip(calls, refs):
        name = str(call.get("name") or "tool").strip() or "tool"
        headers.append(f"{name}[{ref}]")
        args = _summary_safe_args(call.get("args"))
        if isinstance(args, dict):
            keys = [key for key in _SUMMARY_ARG_PRIORITY if key in args]
            keys += sorted((key for key in args if key not in keys), key=lambda key: (key in _SUMMARY_BODY_ARGS, key))
            args = {key: args[key] for key in keys}
        try:
            arguments.append(json.dumps(args, ensure_ascii=False))
        except (TypeError, ValueError):
            arguments.append(SensitiveDataFilter._sanitize_string(str(args)))

    limit = 320
    marker = "... [truncated]"
    fixed = sum(len(header) + 2 for header in headers) + 2 * (len(headers) - 1)
    budgets = [min(len(marker), len(arg)) for arg in arguments]
    if fixed + sum(budgets) > limit:
        # An arbitrarily large batch cannot fit all arguments (or even names).
        # Keep explicit references and say how many calls were omitted.
        parts = []
        for index, header in enumerate(headers):
            suffix = f"; ... [{len(headers) - index - 1} calls omitted]"
            candidate = "; ".join(parts + [header + "(" + marker + ")"])
            if len(candidate + suffix) > limit:
                break
            parts.append(header + "(" + marker + ")")
        return "; ".join(parts + [f"... [{len(headers) - len(parts)} calls omitted]"])

    # Fair allocation prevents one large write_file body hiding later calls.
    remaining = limit - fixed - sum(budgets)
    while remaining and any(budget < len(arg) for budget, arg in zip(budgets, arguments)):
        for index, arg in enumerate(arguments):
            if remaining and budgets[index] < len(arg):
                budgets[index] += 1
                remaining -= 1
    parts = []
    for header, args, budget in zip(headers, arguments, budgets):
        args = " ".join(args.split())
        if len(args) > budget:
            args = args[:budget - len(marker)] + marker
        parts.append(f"{header}({args})")
    return "; ".join(parts)


def format_history_for_summary(
    messages: List[BaseMessage],
    *,
    is_internal_retry: IsInternalRetry,
) -> str:
    parts: List[str] = []
    call_refs: dict[str, Optional[str]] = {}
    call_number = 0
    for message in messages:
        if isinstance(message, HumanMessage) and is_internal_retry(message):
            continue
        refs = []
        if isinstance(message, (AIMessage, AIMessageChunk)):
            batch_ids = set()
            for call in message.tool_calls or []:
                if not isinstance(call, dict):
                    continue
                call_number += 1
                ref = f"c{call_number}"
                refs.append(ref)
                call_id = str(call.get("id") or "")
                if call_id:
                    call_refs[call_id] = None if call_id in batch_ids else ref
                    batch_ids.add(call_id)
        tool_call_text = _format_tool_calls_for_summary(message, refs)
        text = stringify_content(message.content)
        if isinstance(message, ToolMessage):
            tool_name = str(getattr(message, "name", "") or "tool").strip() or "tool"
            rendered = _format_tool_content_for_summary(text, tool_name=tool_name)
            ref = call_refs.get(str(message.tool_call_id or "")) or "unmatched"
            header = f"{message.type}({tool_name})[{ref}]"
        else:
            rendered = _compact_for_summary(text)
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
