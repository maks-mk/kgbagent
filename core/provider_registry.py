from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
import json

from core.reasoning_debug import debug_event

MATCH_TYPES = {"exact", "suffix"}
MODEL_MATCH_FIELDS = {"exact", "prefix", "contains"}
RULE_MODES = {"effort", "toggle"}


class PathConflictError(ValueError):
    def __init__(self, path: str, conflict_at: str):
        self.path = path
        self.conflict_at = conflict_at
        super().__init__(f'Cannot set nested path "{path}": conflict at "{conflict_at}" (not an object)')


class RegistryValidationError(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Invalid provider registry: {reason}")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _ensure_str_list(value: Any, *, field: str, provider_id: str = "") -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        owner = f' for provider "{provider_id}"' if provider_id else ""
        raise RegistryValidationError(f"{field} must be a non-empty string array{owner}")
    return [item.strip().lower() for item in value]


def _hostname_from_base_url(base_url: str) -> str:
    normalized = _clean_text(base_url).lower()
    if not normalized:
        return "api.openai.com"
    if "://" not in normalized:
        normalized = f"https://{normalized}"
    parsed = urlparse(normalized)
    return str(parsed.hostname or "").strip().lower()


def set_nested(obj: dict[str, Any], path: str, value: Any) -> None:
    parts = [_clean_text(part) for part in _clean_text(path).split(".")]
    if not parts or any(not part for part in parts):
        raise RegistryValidationError(f'invalid path "{path}"')

    current = obj
    traversed: list[str] = []
    for part in parts[:-1]:
        traversed.append(part)
        existing = current.get(part)
        if existing is None:
            child: dict[str, Any] = {}
            current[part] = child
            current = child
            continue
        if not isinstance(existing, dict):
            raise PathConflictError(path, ".".join(traversed))
        current = existing
    current[parts[-1]] = value


def _host_matches(hostname: str, patterns: list[str], match_type: str) -> bool:
    for pattern in patterns:
        if match_type == "exact" and hostname == pattern:
            return True
        if match_type == "suffix" and (hostname == pattern or hostname.endswith(f".{pattern}")):
            return True
    return False


def _model_matches(model_name: str | None, models: Mapping[str, Any] | None) -> bool:
    if models is None:
        return True
    normalized = _clean_text(model_name).lower()
    if not normalized:
        return False
    if normalized in models.get("exact", []):
        return True
    if any(normalized.startswith(prefix) for prefix in models.get("prefix", [])):
        return True
    return any(marker in normalized for marker in models.get("contains", []))


class ProviderRegistry:
    def __init__(self, registry: Mapping[str, Any]):
        self._registry = dict(registry)
        self._providers = self._validate_registry(self._registry)

    @classmethod
    def from_path(cls, path: str | Path) -> "ProviderRegistry":
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as exc:
            raise RegistryValidationError(f"cannot read registry file: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RegistryValidationError(f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise RegistryValidationError("registry root must be an object")
        return cls(payload)

    def match(self, base_url: str | None, model_name: str | None = None) -> dict[str, Any] | None:
        """Resolve the first rule whose host and model match, or ``None``.

        The first provider whose host matches is committed to (mirroring how the
        v2 registry keeps one object per host set); the rules inside it are then
        scanned top-to-bottom and the first one matching the model wins. A host
        match with no matching rule (or a provider without rules) resolves to
        ``None`` so no reasoning payload is sent.
        """
        hostname = _hostname_from_base_url(base_url or "")
        if not hostname:
            debug_event("provider_registry_match_skipped", base_url=base_url, hostname="")
            return None

        for provider in self._providers:
            if not provider.get("enabled", True):
                continue
            if not _host_matches(hostname, provider["hosts"], provider["match_type"]):
                continue
            for rule in provider["rules"]:
                if not _model_matches(model_name, rule.get("models")):
                    continue
                resolved = {
                    "id": provider["id"],
                    "mode": rule["mode"],
                    "param": rule["param"],
                    "extra": dict(rule.get("extra") or {}),
                    "notes": _clean_text(rule.get("notes")),
                }
                if rule["mode"] == "toggle":
                    resolved["toggle"] = dict(rule["toggle"])
                else:
                    resolved["values"] = dict(rule["values"])
                debug_event(
                    "provider_registry_matched",
                    base_url=base_url,
                    hostname=hostname,
                    provider_id=provider["id"],
                    match_type=provider["match_type"],
                    mode=rule["mode"],
                    param=rule["param"],
                )
                return resolved
            debug_event(
                "provider_registry_no_rule",
                base_url=base_url,
                hostname=hostname,
                provider_id=provider["id"],
                model=model_name,
            )
            return None
        debug_event("provider_registry_no_match", base_url=base_url, hostname=hostname)
        return None

    @staticmethod
    def _validate_registry(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
        if "schema_version" not in registry:
            raise RegistryValidationError("schema_version is required")
        if "data_version" not in registry:
            raise RegistryValidationError("data_version is required")
        raw_providers = registry.get("providers")
        if not isinstance(raw_providers, list):
            raise RegistryValidationError("providers must be an array")

        seen_ids: set[str] = set()
        providers: list[dict[str, Any]] = []
        for raw in raw_providers:
            if not isinstance(raw, dict):
                raise RegistryValidationError("provider entries must be objects")
            provider = dict(raw)
            provider_id = _clean_text(provider.get("id"))
            if not provider_id:
                raise RegistryValidationError("provider id is required")
            if provider_id in seen_ids:
                raise RegistryValidationError(f'duplicate provider id "{provider_id}"')
            seen_ids.add(provider_id)
            provider["id"] = provider_id
            provider["enabled"] = bool(provider.get("enabled", True))
            provider["hosts"] = _ensure_str_list(provider.get("hosts"), field="hosts", provider_id=provider_id)
            match_type = _clean_text(provider.get("match_type")) or "exact"
            if match_type not in MATCH_TYPES:
                raise RegistryValidationError(f'invalid match_type for provider "{provider_id}"')
            provider["match_type"] = match_type
            provider["rules"] = _validate_rules(provider_id, provider.get("rules"))
            providers.append(provider)
        return providers


def _validate_models(provider_id: str, models: Any) -> dict[str, list[str]]:
    if not isinstance(models, dict):
        raise RegistryValidationError(f'rule.models must be an object for provider "{provider_id}"')
    unknown_fields = set(models) - MODEL_MATCH_FIELDS
    if unknown_fields:
        raise RegistryValidationError(
            f'unsupported rule.models field(s) for provider "{provider_id}": {", ".join(sorted(unknown_fields))}'
        )
    if not any(field in models for field in MODEL_MATCH_FIELDS):
        raise RegistryValidationError(f'rule.models must define at least one matcher for provider "{provider_id}"')
    normalized: dict[str, list[str]] = {}
    for field in MODEL_MATCH_FIELDS:
        if field in models:
            normalized[field] = _ensure_str_list(models.get(field), field=f"models.{field}", provider_id=provider_id)
    return normalized


def _validate_rules(provider_id: str, raw_rules: Any) -> list[dict[str, Any]]:
    if raw_rules is None:
        return []
    if not isinstance(raw_rules, list):
        raise RegistryValidationError(f'rules must be an array for provider "{provider_id}"')

    rules: list[dict[str, Any]] = []
    for raw in raw_rules:
        if not isinstance(raw, dict):
            raise RegistryValidationError(f'rule entries must be objects for provider "{provider_id}"')
        rule = dict(raw)
        param = _clean_text(rule.get("param"))
        if not param or param.startswith(".") or param.endswith("."):
            raise RegistryValidationError(f'invalid rule.param for provider "{provider_id}"')
        rule["param"] = param
        mode = _clean_text(rule.get("mode")) or "effort"
        if mode not in RULE_MODES:
            raise RegistryValidationError(f'invalid rule.mode "{mode}" for provider "{provider_id}"')
        rule["mode"] = mode

        models = rule.get("models")
        if models is not None:
            rule["models"] = _validate_models(provider_id, models)

        if mode == "toggle":
            toggle = rule.get("toggle")
            if not isinstance(toggle, dict) or "on" not in toggle or "off" not in toggle:
                raise RegistryValidationError(f'rule.toggle must define "on" and "off" for provider "{provider_id}"')
            rule["toggle"] = {"on": toggle["on"], "off": toggle["off"]}
        else:
            values = rule.get("values")
            if not isinstance(values, dict) or not values:
                raise RegistryValidationError(f'rule.values must be a non-empty object for provider "{provider_id}"')
            normalized_values = {_clean_text(key).lower(): value for key, value in values.items() if _clean_text(key)}
            if not normalized_values:
                raise RegistryValidationError(f'rule.values must define at least one effort for provider "{provider_id}"')
            rule["values"] = normalized_values

        extra = rule.get("extra")
        if extra is not None and not isinstance(extra, dict):
            raise RegistryValidationError(f'rule.extra must be an object for provider "{provider_id}"')
        rule["extra"] = dict(extra) if isinstance(extra, dict) else {}

        rules.append(rule)
    return rules


def build_reasoning_kwargs(
    kwargs: dict[str, Any],
    config: Mapping[str, Any] | None,
    effort_value: str,
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """Apply the resolved reasoning rule to *kwargs* in place.

    ``config`` is the object returned by :meth:`ProviderRegistry.match`. Effort
    rules add the payload only when *effort_value* is one of the rule's declared
    ``values`` keys; anything else (including ``none`` when not declared) is
    silently skipped. Toggle rules send ``toggle.on`` when *enabled* and
    ``toggle.off`` otherwise.
    """
    if not isinstance(config, Mapping) or not _clean_text(config.get("param")):
        debug_event(
            "reasoning_kwargs_skipped",
            provider_id=config.get("id") if isinstance(config, Mapping) else None,
            reason="no_provider",
        )
        return kwargs

    provider_id = config.get("id")
    param = _clean_text(config.get("param"))
    mode = _clean_text(config.get("mode")) or "effort"
    extra = config.get("extra") if isinstance(config.get("extra"), Mapping) else {}

    if mode == "toggle":
        toggle = config.get("toggle") if isinstance(config.get("toggle"), Mapping) else {}
        key = "on" if enabled else "off"
        if key not in toggle:
            debug_event("reasoning_kwargs_skipped", provider_id=provider_id, reason=f"toggle_{key}_missing")
            return kwargs
        resolved_value = toggle[key]
        set_nested(kwargs, param, resolved_value)
        applied_paths = [param]
        if enabled:
            for extra_path, extra_value in extra.items():
                set_nested(kwargs, str(extra_path), extra_value)
                applied_paths.append(str(extra_path))
        debug_event(
            "reasoning_kwargs_applied",
            provider_id=provider_id,
            input_effort=effort_value,
            resolved_effort=resolved_value,
            paths=applied_paths,
        )
        return kwargs

    # Effort mode.
    if not enabled:
        debug_event("reasoning_kwargs_skipped", provider_id=provider_id, reason="effort_disabled")
        return kwargs
    values = config.get("values") if isinstance(config.get("values"), Mapping) else {}
    key = _clean_text(effort_value).lower()
    if key not in values:
        debug_event(
            "reasoning_kwargs_skipped",
            provider_id=provider_id,
            reason="effort_not_supported",
            input_effort=effort_value,
        )
        return kwargs
    resolved_value = values[key]
    set_nested(kwargs, param, resolved_value)
    applied_paths = [param]
    for extra_path, extra_value in extra.items():
        set_nested(kwargs, str(extra_path), extra_value)
        applied_paths.append(str(extra_path))
    debug_event(
        "reasoning_kwargs_applied",
        provider_id=provider_id,
        input_effort=effort_value,
        resolved_effort=resolved_value,
        paths=applied_paths,
    )
    return kwargs
