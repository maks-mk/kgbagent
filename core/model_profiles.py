from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from core.reasoning_controls import normalize_profile_reasoning

ALLOWED_PROVIDERS = {"openai", "gemini", "anthropic"}
_ID_ALLOWED_RE = re.compile(r"[^a-z0-9_-]+")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_api_key_list(raw_keys: Any, *, fallback: Any = "") -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    def _append(candidate: Any) -> None:
        text = _clean_text(candidate)
        if not text or text in seen:
            return
        seen.add(text)
        values.append(text)

    if isinstance(raw_keys, str):
        raw_items = raw_keys.splitlines() if any(ch in raw_keys for ch in ("\n", "\r")) else [raw_keys]
    elif isinstance(raw_keys, (list, tuple, set)):
        raw_items = list(raw_keys)
    else:
        raw_items = []

    for item in raw_items:
        _append(item)

    fallback_text = _clean_text(fallback)
    if fallback_text and fallback_text not in seen:
        values.insert(0, fallback_text)
    return values


def _normalize_api_key_index(raw_index: Any, api_keys: list[str], *, preferred_key: str = "") -> tuple[int, str]:
    if not api_keys:
        return 0, ""

    if preferred_key and preferred_key in api_keys:
        index = api_keys.index(preferred_key)
    else:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = 0
        index = max(0, min(index, len(api_keys) - 1))
    return index, api_keys[index]


def _next_circular_key_index(api_keys: list[str], current_index: int) -> int | None:
    """Return the next key index in the circular pool, or None if pool has ≤1 key."""
    if len(api_keys) <= 1:
        return None
    return (current_index + 1) % len(api_keys)


def _normalized_rotation_fields(raw_profile: Mapping[str, Any]) -> tuple[list[str], int, list[str], str, dict[str, float]]:
    api_keys = normalize_api_key_list(raw_profile.get("api_keys"), fallback=raw_profile.get("api_key"))
    invalid_api_keys: list[str] = []
    api_key_index, active_api_key = _normalize_api_key_index(
        raw_profile.get("api_key_index"),
        api_keys,
        preferred_key=_clean_text(raw_profile.get("api_key")),
    )
    key_error_timestamps: dict[str, float] = {}
    return api_keys, api_key_index, invalid_api_keys, active_api_key, key_error_timestamps
    
def _normalize_provider(value: Any) -> str:
    normalized = _clean_text(value).lower()
    return normalized if normalized in ALLOWED_PROVIDERS else ""


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _clean_text(value).lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return bool(value)


def sanitize_profile_id(value: Any) -> str:
    raw = _clean_text(value)
    segments = raw.split("/", 1)
    normalized_segments: list[str] = []
    for segment in segments:
        normalized = _ID_ALLOWED_RE.sub("-", segment.lower().replace(" ", "-"))
        normalized = re.sub(r"-{2,}", "-", normalized).strip("-_")
        if normalized:
            normalized_segments.append(normalized)
    return "/".join(normalized_segments)


def _ensure_unique_id(base_id: str, used_ids: set[str]) -> str:
    base = sanitize_profile_id(base_id) or "profile"
    candidate = base
    suffix = 1
    while candidate in used_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def _api_url_prefix(api_url: Any) -> str:
    raw_url = _clean_text(api_url)
    if not raw_url:
        return ""
    parsed = urlsplit(raw_url if "://" in raw_url else f"//{raw_url}")
    hostname = str(parsed.hostname or "").strip(".").lower()
    if not hostname:
        return ""
    labels = hostname.split(".")
    domain_label = next((label for label in reversed(labels[:-1]) if label not in {"api", "www"}), labels[0])
    letters = "".join(char for char in domain_label if "a" <= char <= "z")
    if not letters:
        return ""
    return letters[0] + "".join(char for char in letters[1:] if char not in "aeiouy")


def ensure_unique_profile_id(profile_id: Any, used_ids: set[str]) -> str:
    return _ensure_unique_id(_clean_text(profile_id), used_ids)


def generate_profile_id(model_name: Any, used_ids: set[str], api_url: Any = "") -> str:
    model_text = _clean_text(model_name)
    model_id = sanitize_profile_id(model_text.rsplit("/", 1)[-1] if "/" in model_text else model_text)
    prefix = _api_url_prefix(api_url)
    base = f"{prefix}/{model_id}" if prefix and model_id else model_id
    return _ensure_unique_id(base or "profile", used_ids)


def normalize_profiles_payload(payload: Any) -> dict[str, Any]:
    raw_payload = payload if isinstance(payload, dict) else {}
    raw_profiles = raw_payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raw_profiles = []

    used_ids: set[str] = set()
    seen_profiles: set[tuple[str, str, tuple[str, ...], str, bool, bool, str]] = set()
    profiles: list[dict[str, Any]] = []
    for raw in raw_profiles:
        if not isinstance(raw, dict):
            continue
        provider = _normalize_provider(raw.get("provider"))
        model_name = _clean_text(raw.get("model"))
        if not provider or not model_name:
            continue
        api_keys, api_key_index, invalid_api_keys, api_key, key_error_timestamps = _normalized_rotation_fields(raw)
        base_url = _clean_text(raw.get("base_url")) if provider in {"openai", "anthropic"} else ""
        supports_image_input = _normalize_bool(raw.get("supports_image_input"))
        enabled = _normalize_bool(raw.get("enabled")) if "enabled" in raw else True
        reasoning = normalize_profile_reasoning(raw.get("reasoning"))

        # Deduplicate exact same profile payloads to keep env bootstrap/import idempotent.
        profile_fingerprint = (
            provider,
            model_name,
            tuple(api_keys),
            base_url,
            supports_image_input,
            enabled,
            json.dumps(reasoning, sort_keys=True),
        )
        if profile_fingerprint in seen_profiles:
            continue
        seen_profiles.add(profile_fingerprint)

        requested_id = sanitize_profile_id(raw.get("id"))
        fallback_id = model_name.rsplit("/", 1)[-1] if "/" in model_name else model_name
        profile_id = _ensure_unique_id(requested_id or fallback_id, used_ids)
        profile = {
            "id": profile_id,
            "provider": provider,
            "model": model_name,
            "api_key": api_key,
            "api_keys": api_keys,
            "api_key_index": api_key_index,
            "invalid_api_keys": invalid_api_keys,
            "key_error_timestamps": key_error_timestamps,
            "base_url": base_url,
            "supports_image_input": supports_image_input,
            "enabled": enabled,
        }
        if reasoning:
            profile["reasoning"] = reasoning
        profiles.append(profile)

    active_raw = raw_payload.get("active_profile")
    active_profile = _clean_text(active_raw) if active_raw is not None else ""
    enabled_ids = [item["id"] for item in profiles if bool(item.get("enabled", True))]
    if active_profile not in enabled_ids:
        active_profile = enabled_ids[0] if enabled_ids else ""

    normalized: dict[str, Any] = {
        "active_profile": active_profile or None,
        "profiles": profiles,
    }

    # Preserve non-profile UI settings stored in config.json (e.g. SESSION_SIZE
    # managed by the Settings dialog) so profile rewrites never drop them.
    raw_session_size = raw_payload.get("session_size")
    if raw_session_size is not None:
        normalized["session_size"] = raw_session_size

    return normalized


def bootstrap_profiles_from_env(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    source = env or os.environ
    provider_hint = _normalize_provider(source.get("PROVIDER", ""))
    generic_model = _clean_text(source.get("MODEL"))

    if generic_model and provider_hint:
        profile_payload = {
            "active_profile": None,
            "profiles": [
                {
                    "id": "",
                    "provider": provider_hint,
                    "model": generic_model,
                    "api_key": _clean_text(source.get("API_KEY")),
                    "base_url": _clean_text(source.get("BASE_URL")),
                }
            ],
        }
        return normalize_profiles_payload(profile_payload)

    openai_model = _clean_text(source.get("OPENAI_MODEL"))
    gemini_model = _clean_text(source.get("GEMINI_MODEL"))
    anthropic_model = _clean_text(source.get("ANTHROPIC_MODEL"))

    chosen_provider = ""
    chosen_model = ""
    chosen_api_key = ""
    chosen_base_url = ""

    if provider_hint == "openai" and openai_model:
        chosen_provider = "openai"
        chosen_model = openai_model
        chosen_api_key = _clean_text(source.get("OPENAI_API_KEY"))
        chosen_base_url = _clean_text(source.get("OPENAI_BASE_URL"))
    elif provider_hint == "gemini" and gemini_model:
        chosen_provider = "gemini"
        chosen_model = gemini_model
        chosen_api_key = _clean_text(source.get("GEMINI_API_KEY"))
    elif provider_hint == "anthropic" and anthropic_model:
        chosen_provider = "anthropic"
        chosen_model = anthropic_model
        chosen_api_key = _clean_text(source.get("ANTHROPIC_API_KEY"))
        chosen_base_url = _clean_text(source.get("ANTHROPIC_BASE_URL"))
    elif openai_model:
        chosen_provider = "openai"
        chosen_model = openai_model
        chosen_api_key = _clean_text(source.get("OPENAI_API_KEY"))
        chosen_base_url = _clean_text(source.get("OPENAI_BASE_URL"))
    elif gemini_model:
        chosen_provider = "gemini"
        chosen_model = gemini_model
        chosen_api_key = _clean_text(source.get("GEMINI_API_KEY"))
    elif anthropic_model:
        chosen_provider = "anthropic"
        chosen_model = anthropic_model
        chosen_api_key = _clean_text(source.get("ANTHROPIC_API_KEY"))
        chosen_base_url = _clean_text(source.get("ANTHROPIC_BASE_URL"))

    if chosen_provider and chosen_model:
        return normalize_profiles_payload(
            {
                "active_profile": None,
                "profiles": [
                    {
                        "id": "",
                        "provider": chosen_provider,
                        "model": chosen_model,
                        "api_key": chosen_api_key,
                        "base_url": chosen_base_url,
                    }
                ],
            }
        )

    return {"active_profile": None, "profiles": []}


def find_active_profile(payload: dict[str, Any]) -> dict[str, Any] | None:
    normalized = normalize_profiles_payload(payload)
    active_id = _clean_text(normalized.get("active_profile"))
    for profile in normalized.get("profiles", []) or []:
        if isinstance(profile, dict) and _clean_text(profile.get("id")) == active_id:
            return dict(profile)
    return None


def find_profile_by_id(payload: Any, profile_id: Any) -> dict[str, Any] | None:
    normalized = normalize_profiles_payload(payload)
    target_id = _clean_text(profile_id)
    if not target_id:
        return None
    for profile in normalized.get("profiles", []) or []:
        if isinstance(profile, dict) and _clean_text(profile.get("id")) == target_id:
            return dict(profile)
    return None


def _profile_merge_key(profile: Mapping[str, Any]) -> tuple[str, str, str]:
    provider = _normalize_provider(profile.get("provider"))
    model_name = _clean_text(profile.get("model"))
    base_url = _clean_text(profile.get("base_url")) if provider in {"openai", "anthropic"} else ""
    return provider, model_name, base_url


def merge_profiles_with_env(existing_payload: Any, env_payload: Any) -> dict[str, Any]:
    current = normalize_profiles_payload(existing_payload)
    env_normalized = normalize_profiles_payload(env_payload)
    env_profiles = [item for item in env_normalized.get("profiles", []) if isinstance(item, dict)]
    if not env_profiles:
        return current

    merged_profiles: list[dict[str, str]] = [dict(item) for item in current.get("profiles", [])]

    for env_profile in env_profiles:
        env_key = _profile_merge_key(env_profile)
        if not all(env_key[:2]):
            continue
        match_index = next(
            (
                index
                for index, existing in enumerate(merged_profiles)
                if _profile_merge_key(existing) == env_key
            ),
            None,
        )
        if match_index is None:
            merged_profiles.append(dict(env_profile))
            continue

        # Keep user-edited values, but backfill missing credentials from env.
        existing_profile = dict(merged_profiles[match_index])
        env_api_key = _clean_text(env_profile.get("api_key"))
        if env_api_key and not _clean_text(existing_profile.get("api_key")):
            existing_profile["api_key"] = env_api_key
        if env_key[0] in {"openai", "anthropic"}:
            env_base_url = _clean_text(env_profile.get("base_url"))
            if env_base_url and not _clean_text(existing_profile.get("base_url")):
                existing_profile["base_url"] = env_base_url
        merged_profiles[match_index] = existing_profile

    active_profile = _clean_text(current.get("active_profile"))
    if not active_profile:
        active_profile = _clean_text(env_normalized.get("active_profile"))

    merged_payload: dict[str, Any] = {
        "active_profile": active_profile or None,
        "profiles": merged_profiles,
    }
    # Keep the UI-saved SESSION_SIZE override when merging env bootstrap
    # data back into the stored payload.
    if "session_size" in current:
        merged_payload["session_size"] = current["session_size"]
    return normalize_profiles_payload(merged_payload)


class ModelProfileStore:
    _lock_registry_guard = threading.Lock()
    _lock_registry: dict[str, threading.RLock] = {}

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize(payload: Any) -> dict[str, Any]:
        return normalize_profiles_payload(payload)

    def _path_lock(self) -> threading.RLock:
        key = str(self.path.resolve())
        with self._lock_registry_guard:
            lock = self._lock_registry.get(key)
            if lock is None:
                lock = threading.RLock()
                self._lock_registry[key] = lock
            return lock

    def _read_raw_payload(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return raw if isinstance(raw, dict) else None

    def _write_normalized_payload(self, payload: Any) -> dict[str, Any]:
        normalized = self.normalize(payload)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        serialized = json.dumps(normalized, ensure_ascii=False, indent=2)
        tmp_path.write_text(serialized, encoding="utf-8")
        try:
            tmp_path.replace(self.path)
        except PermissionError:
            # Some Windows environments temporarily block atomic replace;
            # fall back to direct write instead of failing profile updates.
            self.path.write_text(serialized, encoding="utf-8")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        return normalized

    def load(self) -> dict[str, Any]:
        with self._path_lock():
            raw = self._read_raw_payload()
            if raw is None:
                return {"active_profile": None, "profiles": []}
            normalized = self.normalize(raw)
            if normalized != raw:
                return self._write_normalized_payload(normalized)
            return normalized

    def load_or_initialize(self, env: Mapping[str, str] | None = None) -> dict[str, Any]:
        with self._path_lock():
            env_bootstrap = bootstrap_profiles_from_env(env)
            raw = self._read_raw_payload()
            if raw is None:
                return self._write_normalized_payload(env_bootstrap)
            normalized = self.normalize(raw)
            merged = merge_profiles_with_env(normalized, env_bootstrap)
            if merged != normalized:
                return self._write_normalized_payload(merged)
            if normalized != raw:
                return self._write_normalized_payload(normalized)
            return normalized

    def save(self, payload: Any) -> dict[str, Any]:
        with self._path_lock():
            return self._write_normalized_payload(payload)

    def get_api_key_state(self, profile_id: str) -> dict[str, Any]:
        with self._path_lock():
            profile = find_profile_by_id(self.load(), profile_id)
            if profile is None:
                return {
                    "profile_id": _clean_text(profile_id),
                    "api_keys": [],
                    "invalid_api_keys": [],
                    "api_key_index": 0,
                    "current_key": "",
                    "available_keys": [],
                    "key_error_timestamps": {},
                }
            api_keys = list(profile.get("api_keys") or [])
            current_key = _clean_text(profile.get("api_key"))
            # Get error timestamps if they exist
            key_error_timestamps = dict(profile.get("key_error_timestamps") or {})
            return {
                "profile_id": _clean_text(profile.get("id")),
                "api_keys": api_keys,
                "invalid_api_keys": [],
                "api_key_index": int(profile.get("api_key_index") or 0),
                "current_key": current_key,
                "available_keys": api_keys,
                "key_error_timestamps": key_error_timestamps,
            }

    def rotate_api_key(self, profile_id: str, failed_key: str) -> dict[str, Any]:
        with self._path_lock():
            payload = self.load()
            target_id = _clean_text(profile_id)
            failed_key = _clean_text(failed_key)
            profiles = [dict(item) for item in payload.get("profiles", []) or []]
            profile_index = next(
                (
                    index
                    for index, profile in enumerate(profiles)
                    if _clean_text(profile.get("id")) == target_id
                ),
                None,
            )
            if profile_index is None:
                return self.get_api_key_state(target_id)

            profile = dict(profiles[profile_index])
            api_keys = list(profile.get("api_keys") or [])

            current_index, current_key = _normalize_api_key_index(
                profile.get("api_key_index"),
                api_keys,
                preferred_key=profile.get("api_key"),
            )
            rotation_changed = False
            if current_key == failed_key:
                next_index = _next_circular_key_index(api_keys, current_index)
                if next_index is not None and next_index != current_index:
                    current_index = next_index
                    current_key = api_keys[current_index]
                    rotation_changed = True

            profile["api_keys"] = api_keys
            profile["api_key_index"] = current_index
            profile["invalid_api_keys"] = []
            profile["api_key"] = current_key
            profile["key_error_timestamps"] = {}
            profiles[profile_index] = profile
            payload["profiles"] = profiles
            if rotation_changed:
                payload = self._write_normalized_payload(payload)
            else:
                payload = self.normalize(payload)
            return self.get_api_key_state(target_id)
            
