from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.constants import BASE_DIR

logger = logging.getLogger("agent")

DEFAULT_HEADERS_FILE = "codex_headers.json"
DEFAULT_HEADERS_SECTION = "default_headers"


def _configured_headers_file() -> str:
    """Resolve the headers file name from the process env or from configuration.

    A real environment variable wins over ``.env`` / ``.agent_state/config.json``,
    matching the priority of :class:`core.config.AgentConfig`. The config import
    is local so this module stays free of import cycles.
    """
    override = os.getenv("HEADERS_FILE")
    if override and override.strip():
        return override.strip()
    try:
        from core.config import AgentConfig  # local import avoids an import cycle

        value = AgentConfig().headers_file
    except Exception as exc:  # pragma: no cover - config must never break headers
        logger.warning("Could not read HEADERS_FILE from configuration: %s", exc)
        return DEFAULT_HEADERS_FILE
    return value.strip() if value and value.strip() else DEFAULT_HEADERS_FILE


def _default_headers_path() -> Path:
    """Resolve the provider headers file location.

    The file name is configurable via the ``HEADERS_FILE`` environment variable
    or the matching ``.env`` entry and is resolved relative to :data:`BASE_DIR`,
    which points to the executable directory when frozen (PyInstaller) and to the
    project root otherwise. Absolute values are honored as-is.
    """
    configured = _configured_headers_file()
    candidate = Path(configured)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / configured


def load_provider_headers(path: Path | None = None) -> dict[str, str]:
    """Load editable headers for provider requests without failing startup.

    When the headers file is absent, an empty mapping is returned so the
    underlying SDK sends its own standard headers (no spoofing). When the file
    exists, only its string-valued entries are applied as overrides. A flat
    JSON object is used as-is; a sectioned file such as ``codex_headers.json``
    contributes its ``default_headers`` section only, because its auth,
    cookie and transport sections carry per-session values that would
    override real credentials.

    Used by both OpenAI-compatible and Anthropic providers to allow custom
    ``User-Agent`` / ``x-*`` header overrides from the headers file.
    """
    headers_path = path or _default_headers_path()
    try:
        payload: Any = json.loads(headers_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Could not load provider headers from %s: %s", headers_path, exc)
        return {}

    if not isinstance(payload, dict):
        logger.warning("Ignoring provider headers from %s: expected a JSON object", headers_path)
        return {}

    section = payload.get(DEFAULT_HEADERS_SECTION)
    if section is not None and not isinstance(section, dict):
        logger.warning(
            "Ignoring provider headers from %s: section %r is not a JSON object",
            headers_path,
            DEFAULT_HEADERS_SECTION,
        )
        return {}
    headers = payload if section is None else section

    return {
        key: value
        for key, value in headers.items()
        if isinstance(key, str) and key.strip() and isinstance(value, str)
    }


def load_openai_headers(path: Path | None = None) -> dict[str, str]:
    """Backward-compatible alias for :func:`load_provider_headers`."""
    return load_provider_headers(path)
