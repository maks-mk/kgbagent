from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.constants import BASE_DIR

logger = logging.getLogger("agent")


def _default_headers_path() -> Path:
    """Resolve the provider headers file location.

    The file name is configurable via the ``HEADERS_FILE`` environment
    variable and is resolved relative to :data:`BASE_DIR`, which points to the
    executable directory when frozen (PyInstaller) and to the project root
    otherwise. Absolute values are honored as-is.
    """
    return BASE_DIR / os.getenv("HEADERS_FILE", "headers.json")


def load_provider_headers(path: Path | None = None) -> dict[str, str]:
    """Load editable headers for provider requests without failing startup.

    When the headers file is absent, an empty mapping is returned so the
    underlying SDK sends its own standard headers (no spoofing). When the file
    exists, only its string-valued entries are applied as overrides.

    Used by both OpenAI-compatible and Anthropic providers to allow custom
    ``User-Agent`` / ``x-*`` header overrides from ``headers.json``.
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

    return {
        key: value
        for key, value in payload.items()
        if isinstance(key, str) and key.strip() and isinstance(value, str)
    }


def load_openai_headers(path: Path | None = None) -> dict[str, str]:
    """Backward-compatible alias for :func:`load_provider_headers`."""
    return load_provider_headers(path)
