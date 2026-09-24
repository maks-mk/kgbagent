"""Provider-aware reasoning controls used by the model-profile UI.
The UI stores a small, provider-neutral ``reasoning`` object on a profile and
this module translates it to the configuration fields consumed by the native
provider adapters. OpenAI-compatible options deliberately come from the
provider registry, because the accepted vocabulary differs by endpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.constants import BASE_DIR
from core.provider_registry import ProviderRegistry
from core.anthropic_capabilities import (
    anthropic_model_reasoning_efforts,
    anthropic_model_requires_thinking,
    anthropic_model_uses_adaptive_thinking,
    anthropic_model_uses_manual_thinking,
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _title(value: str) -> str:
    return value.replace("xhigh", "X-High").replace("_", " ").title()


def _option(value: str, label: str, config: dict[str, Any]) -> dict[str, Any]:
    return {"value": value, "label": label, "config": config}


def _normalized_gemini_model_name(model_name: str) -> str:
    normalized = _clean_text(model_name).lower()
    return normalized[len("models/") :] if normalized.startswith("models/") else normalized


def _gemini_model_supports_thinking_budget(model_name: str) -> bool:
    normalized = _normalized_gemini_model_name(model_name)
    return normalized.startswith("gemini-2.5") or normalized in {
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-pro-latest",
    }


def _gemini_model_supports_thinking_level(model_name: str) -> bool:
    normalized = _normalized_gemini_model_name(model_name)
    return normalized.startswith("gemini-3") or normalized.startswith("gemma-4")


def reasoning_options_for_profile(
    profile: Mapping[str, Any] | None,
    *,
    registry_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return only the reasoning controls supported by *profile*."""
    data = profile if isinstance(profile, Mapping) else {}
    provider = _clean_text(data.get("provider")).lower()
    model = _clean_text(data.get("model"))
    if provider == "openai":
        try:
            registry = ProviderRegistry.from_path(registry_path or (BASE_DIR / "provider_registry.json"))
            provider_config = registry.match(_clean_text(data.get("base_url")), model)
        except Exception:
            return []
        if not isinstance(provider_config, Mapping):
            return []
        if provider_config.get("mode") == "toggle":
            return [
                _option("off", "Off", {"enabled": False}),
                _option("true", "On", {"enabled": True}),
            ]

        # ``values`` maps input effort -> provider value; expose the distinct
        # provider-side levels (deduplicated, first-occurrence order) as the
        # selectable reasoning options.
        values = provider_config.get("values", {}) if isinstance(provider_config, Mapping) else {}
        levels: list[str] = []
        for resolved in values.values():
            label = _clean_text(resolved).lower()
            if label and label not in levels:
                levels.append(label)
        return [_option(value, _title(value), {"enabled": True, "effort": value}) for value in levels]

    off = _option("off", "Off", {"enabled": False})
    if provider == "gemini":
        if _gemini_model_supports_thinking_level(model):
            normalized_model = _normalized_gemini_model_name(model)
            if normalized_model.startswith("gemma-4"):
                values = ("minimal", "high")
            elif normalized_model.startswith(("gemini-3.1-pro", "gemini-3-pro")):
                values = ("low", "medium", "high") if "3.1" in normalized_model else ("low", "high")
            elif normalized_model.startswith("gemini-3.1-flash-lite-image"):
                values = ("minimal", "high")
            else:
                values = ("minimal", "low", "medium", "high")
            return [_option(value, _title(value), {"enabled": True, "effort": value}) for value in values]
        if _gemini_model_supports_thinking_budget(model):
            budgets = (1024, 4096, 8192)
            return [
                off,
                *[
                    _option(f"budget:{budget}", f"Thinking: {budget:,}", {"enabled": True, "thinking_budget": budget})
                    for budget in budgets
                ],
            ]
        return []

    if provider == "anthropic":
        efforts = anthropic_model_reasoning_efforts(model)
        off_options = [] if anthropic_model_requires_thinking(model) else [off]
        if efforts:
            return [*off_options, *[_option(value, _title(value), {"enabled": True, "effort": value}) for value in efforts]]
        if anthropic_model_uses_adaptive_thinking(model):
            return [*off_options, _option("adaptive", "Adaptive", {"enabled": True, "mode": "adaptive"})]
        if not anthropic_model_uses_manual_thinking(model):
            return []
        budgets = (1024, 4096, 8192)
        return [
            off,
            *[
                _option(f"budget:{budget}", f"Thinking: {budget:,}", {"enabled": True, "thinking_budget": budget})
                for budget in budgets
            ],
        ]

    return []


def normalize_profile_reasoning(value: Any) -> dict[str, Any]:
    """Keep the persisted profile reasoning payload small and well-formed."""
    raw = value if isinstance(value, Mapping) else {}
    if not raw:
        return {}
    result: dict[str, Any] = {"enabled": bool(raw.get("enabled", True))}
    mode = _clean_text(raw.get("mode")).lower()
    if mode == "adaptive":
        result["mode"] = mode
    effort = _clean_text(raw.get("effort")).lower()
    if effort in {"true", "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        result["effort"] = effort
    try:
        budget = int(raw.get("thinking_budget"))
    except (TypeError, ValueError):
        budget = 0
    if budget > 0:
        result["thinking_budget"] = budget
    return result


def profile_reasoning_overrides(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    """Translate a persisted profile setting into ``AgentConfig`` overrides."""
    data = profile if isinstance(profile, Mapping) else {}
    reasoning = normalize_profile_reasoning(data.get("reasoning"))
    if not reasoning:
        return {}
    provider = _clean_text(data.get("provider")).lower()
    enabled = bool(reasoning.get("enabled", True))
    overrides: dict[str, Any] = {"enable_model_reasoning": enabled}
    effort = _clean_text(reasoning.get("effort")).lower()
    if provider == "gemini" and not enabled:
        # Gemini 3 treats an omitted thinking config as its model default, not as
        # disabled. Preserve the meaning of old saved "Off" choices explicitly.
        overrides["enable_model_reasoning"] = True
        model = _clean_text(data.get("model"))
        if _gemini_model_supports_thinking_level(model):
            overrides["model_reasoning_effort"] = "minimal"
        elif _gemini_model_supports_thinking_budget(model):
            overrides["gemini_thinking_budget"] = 0
        return overrides
    if provider in {"openai", "gemini"} and effort:
        overrides["model_reasoning_effort"] = effort
    if provider == "gemini" and "thinking_budget" in reasoning:
        overrides["gemini_thinking_budget"] = int(reasoning["thinking_budget"])
    if provider == "anthropic":
        if not enabled:
            overrides["anthropic_reasoning"] = "off"
        else:
            if reasoning.get("mode") == "adaptive":
                overrides["anthropic_reasoning"] = "adaptive"
            elif effort:
                overrides["anthropic_reasoning"] = effort
            if "thinking_budget" in reasoning:
                overrides["anthropic_thinking_budget"] = int(reasoning["thinking_budget"])
    return overrides
