from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping
from typing import Any


def canonicalize_tool_args(raw_args: Any) -> dict[str, Any]:
    """Return canonical tool args from dict-like or JSON-string payloads."""
    parsed, _kind = inspect_tool_args_payload(raw_args)
    return parsed


def normalize_tool_args_for_repeat_detection(raw_args: Any) -> dict[str, Any]:
    """Normalize args so cosmetic differences still count as the same call.

    Loop detection compares tool arguments, so a retry that only differs in
    whitespace, key order, argument spelling or (on Windows) path case must be
    recognized as the same call.  Without this, a single trailing space or a
    second letter changed in a filename makes every destructive retry look like a
    brand-new call and the loop limit never triggers.
    """
    canonical = canonicalize_tool_args(raw_args)
    normalized: dict[str, Any] = {}
    for key, value in canonical.items():
        # Argument names are schema-defined, so ``path``/``Path``/``PATH`` are
        # the same argument and must fold together.
        normalized_key = _collapse_whitespace(str(key)).strip().casefold()
        normalized[normalized_key] = _normalize_arg_value(value)
    return dict(sorted(normalized.items()))


def tool_loop_budget(
    recent_calls: Iterable[Mapping[str, Any]],
    *,
    tool_name: str,
    tool_args: Any,
    readonly_tool_names: Collection[str],
    readonly_limit: int,
    mutating_limit: int,
) -> tuple[int, int]:
    """Return repeat count and limit using the same policy at both guard boundaries."""
    normalized_name = str(tool_name or "").strip()
    normalized_args = normalize_tool_args_for_repeat_detection(tool_args)
    count = sum(
        1
        for call in recent_calls
        if str(call.get("name") or "").strip() == normalized_name
        and normalize_tool_args_for_repeat_detection(call.get("args")) == normalized_args
    )
    limit = readonly_limit if normalized_name in readonly_tool_names else mutating_limit
    return count, limit


def _collapse_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def _normalize_arg_value(value: Any) -> Any:
    if isinstance(value, str):
        collapsed = _collapse_whitespace(value).strip()
        # Paths are case-insensitive on Windows; fold them so ``A.txt`` and
        # ``a.txt`` count as the same target.
        if "\\" in collapsed or "/" in collapsed:
            return collapsed.replace("\\", "/").casefold()
        return collapsed.casefold()
    if isinstance(value, Mapping):
        return {
            _collapse_whitespace(str(key)).strip().casefold(): _normalize_arg_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_arg_value(item) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _collapse_whitespace(str(value)).strip().casefold()


def inspect_tool_args_payload(raw_args: Any) -> tuple[dict[str, Any], str]:
    """Return canonical args plus the original payload shape for diagnostics."""
    value = raw_args
    for _ in range(2):
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}, "mapping"
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}, "empty_string"
            try:
                value = json.loads(text)
            except Exception:
                return {}, "invalid_json_string"
            if isinstance(value, Mapping):
                return {str(key): item for key, item in value.items()}, "json_string"
            continue
        if value is None:
            return {}, "missing"
        return {}, type(value).__name__
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}, "mapping"
    if value is None:
        return {}, "missing"
    return {}, "json_non_object"
