"""Restore real standard streams in windowed PyInstaller builds.

PyInstaller's ``--windowed`` mode runs the app without a console, so
``sys.stdin``/``sys.stdout``/``sys.stderr`` are ``None``. Libraries that spawn
subprocesses then hand an invalid stderr handle down to the child.

The MCP stdio transport is one such case: ``mcp.client.stdio.stdio_client``
captures ``sys.stderr`` as its default ``errlog`` when the module is imported,
so a ``None`` value makes the spawned server exit immediately and surfaces as
``McpError: Connection closed``. Installing real streams before the MCP stack is
imported keeps stdio servers working in the packaged executable.
"""

from __future__ import annotations

import os
import sys

_STANDARD_STREAM_MODES = {"stdin": "r", "stdout": "w", "stderr": "w"}


def ensure_standard_streams() -> None:
    """Replace ``None`` standard streams with valid null devices.

    Idempotent: streams that already exist (normal console runs) are left
    untouched, so this is safe to call from any entrypoint.
    """
    for name, mode in _STANDARD_STREAM_MODES.items():
        if getattr(sys, name, None) is not None:
            continue
        try:
            setattr(sys, name, open(os.devnull, mode, encoding="utf-8"))
        except OSError:
            continue
