"""Model capability rules for Anthropic thinking and reasoning effort."""

from __future__ import annotations

import re

_MANUAL_THINKING_MODELS = (
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
)
_ADAPTIVE_THINKING_MODELS = (
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
)
_ALWAYS_ON_THINKING_MODELS = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
)
_XHIGH_EFFORT_MODELS = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-fable-5",
    "claude-mythos-5",
)
_MAX_EFFORT_MODELS = (
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
)
_EFFORT_MODELS = (
    "claude-opus-4-5",
    *_MAX_EFFORT_MODELS,
)
_NO_SAMPLING_MODELS = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
)
_BASE_EFFORTS = ("low", "medium", "high")


def _model_basename(model: str | None) -> str:
    return str(model or "").strip().lower().rsplit("/", 1)[-1]


def _matches_family(model: str | None, families: tuple[str, ...]) -> bool:
    normalized = _model_basename(model)
    if normalized.startswith("claude-4.8-opus"):
        normalized = "claude-opus-4-8" + normalized[len("claude-4.8-opus") :]
    # Anthropic aliases some versions with "." (e.g. "claude-opus-5.5") and
    # others with "-" (e.g. "claude-opus-5-5"). Normalise both separators so
    # every family entry covers both spellings without listing each twice.
    normalized = re.sub(r"[._]", "-", normalized)
    return any(normalized == family or normalized.startswith(f"{family}-") for family in families)


def is_claude_model(model: str | None) -> bool:
    """Whether *model* uses Anthropic's Claude model naming family.

    Anthropic-compatible aggregators can expose non-Claude models through the
    Messages API. Those models must not receive Claude-specific thinking or
    reasoning parameters merely because the selected transport is Anthropic.
    """
    return _model_basename(model).startswith("claude-")


def anthropic_model_uses_manual_thinking(model: str | None) -> bool:
    return _matches_family(model, _MANUAL_THINKING_MODELS)


def anthropic_model_uses_adaptive_thinking(model: str | None) -> bool:
    return _matches_family(model, _ADAPTIVE_THINKING_MODELS)


def anthropic_model_requires_thinking(model: str | None) -> bool:
    return _matches_family(model, _ALWAYS_ON_THINKING_MODELS)


def anthropic_model_reasoning_efforts(model: str | None) -> tuple[str, ...]:
    if not _matches_family(model, _EFFORT_MODELS):
        return ()
    efforts = _BASE_EFFORTS
    if _matches_family(model, _MAX_EFFORT_MODELS):
        efforts = (*efforts, "max")
    if _matches_family(model, _XHIGH_EFFORT_MODELS):
        efforts = (*efforts, "xhigh")
    return efforts


def anthropic_model_disallows_sampling(model: str | None) -> bool:
    return _matches_family(model, _NO_SAMPLING_MODELS)
