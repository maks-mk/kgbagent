"""Shared text comparisons for streamed and restored assistant responses."""

import difflib
import re

REPLAY_SIMILARITY = 0.94


def normalized_replay_text(text: str) -> str:
    return " ".join(str(text or "").split())


def common_prefix_length(first: str, second: str) -> int:
    index = 0
    limit = min(len(first), len(second))
    while index < limit and first[index] == second[index]:
        index += 1
    return index


def _partial_word_prefix(previous: str, incoming: str, boundary: int) -> bool:
    return bool(previous) and boundary == len(previous) and boundary < len(incoming) and (
        not previous[-1].isspace() and not incoming[boundary].isspace()
    )


def _significant_prefix(previous: str, prefix: str) -> bool:
    return bool(prefix) and (
        len(prefix) >= min(len(previous), 48) or len(prefix) >= int(len(previous) * 0.8)
    )


def similar_prefix_boundary(previous: str, incoming: str, *, min_length: int = 0) -> int | None:
    """Find a matching prefix at a word boundary, preserving partial new words."""
    previous = normalized_replay_text(previous)
    if not previous or len(previous) < min_length:
        return None
    best_index = None
    for match in re.finditer(r"\S+", incoming):
        prefix = normalized_replay_text(incoming[:match.end()])
        if previous.startswith(prefix):
            best_index = match.end()
            continue
        if prefix.startswith(previous):
            if _partial_word_prefix(previous, prefix, len(previous)):
                return best_index
            return best_index if best_index is not None else match.end()
        if len(prefix) >= min(len(previous), 80):
            if difflib.SequenceMatcher(None, previous, prefix).quick_ratio() >= REPLAY_SIMILARITY:
                best_index = match.end()
    return best_index


def assistant_replay_remainder(previous: str, incoming: str) -> str | None:
    """Return new text, or None when the prefix is not a sufficiently clear replay."""
    previous_normalized = normalized_replay_text(previous)
    incoming_normalized = normalized_replay_text(incoming)
    if not previous_normalized or not incoming_normalized:
        return None
    if previous_normalized == incoming_normalized:
        return ""
    prefix_len = common_prefix_length(previous, incoming)
    if prefix_len and (incoming.startswith(previous) or _significant_prefix(previous, incoming[:prefix_len])):
        if not _partial_word_prefix(previous, incoming, prefix_len):
            return incoming[prefix_len:].lstrip()
    boundary = similar_prefix_boundary(previous, incoming)
    if boundary is not None:
        prefix = incoming[:boundary]
        # A shared first word can begin a new comment rather than replay one.
        if len(prefix.split()) >= 2 and _significant_prefix(previous_normalized, normalized_replay_text(prefix)):
            return incoming[boundary:].lstrip()
    return None
