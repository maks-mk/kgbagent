"""Bounded output capture for cli_exec with full-output persistence.

Inline text keeps the head+tail shape cli_exec always returned, while the
complete stream is mirrored into ``<workspace>/.agent_state/cli_exec`` as soon
as it outgrows ``max_raw_tool_output``.  The model then receives a
``<persisted-output>`` envelope pointing at the artifact, so oversized output is
no longer silently lost.  Persistence policy and envelope wording follow ZCode's
Bash tool (``.bash_tool/bash_tool/streams.py``, ``model_content.py``).

Artifacts live inside the workspace on purpose: the filesystem tools can only
read paths inside the workspace jail, so a state directory next to the
application would produce a path the model cannot open.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

__all__ = [
    "ARTIFACT_DIR_ENV",
    "DEFAULT_KEEP_ARTIFACTS",
    "DEFAULT_MAX_PERSISTED_CHARS",
    "MAX_PERSISTED_CHARS_ENV",
    "PERSISTED_OUTPUT_CLOSE_TAG",
    "PERSISTED_OUTPUT_OPEN_TAG",
    "ShellOutputCapture",
    "build_persisted_output_envelope",
    "format_character_size",
    "format_persisted_output_pointer",
    "is_persisted_output",
    "prune_artifacts",
    "resolve_artifact_directory",
    "resolve_max_persisted_chars",
]

PERSISTED_OUTPUT_OPEN_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSE_TAG = "</persisted-output>"
ARTIFACT_DIR_ENV = "CLI_EXEC_RESULTS_DIR"
MAX_PERSISTED_CHARS_ENV = "CLI_EXEC_MAX_PERSISTED_CHARS"
DEFAULT_MAX_PERSISTED_CHARS = 5_000_000
DEFAULT_KEEP_ARTIFACTS = 20
ARTIFACT_SUBDIRECTORY = (".agent_state", "cli_exec")
_ARTIFACT_NAME_PREFIX = "cli-exec-"
_ARTIFACT_NAME_RE = re.compile(r"^cli-exec-\d{8}-\d{6}-[0-9a-f]{8}\.log$")
_TRUNCATION_MARKER = "\n[ARTIFACT TRUNCATED: output exceeded {limit} characters]\n"
_READ_HINT = "Use read_file with offset/limit on that path to inspect the rest."


def resolve_artifact_directory(working_directory: str) -> Path:
    """Directory for persisted cli_exec output, inside the workspace by default."""

    base = Path(working_directory or os.getcwd()).resolve()
    override = os.environ.get(ARTIFACT_DIR_ENV, "").strip()
    if override:
        candidate = Path(override).expanduser()
        resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            # The model can only read inside the workspace, so a pointer to an
            # outside path would be useless; keep the workspace default.
            return base.joinpath(*ARTIFACT_SUBDIRECTORY)
        return resolved
    return base.joinpath(*ARTIFACT_SUBDIRECTORY)


def resolve_max_persisted_chars() -> int:
    """Upper bound for one persisted artifact (characters)."""

    return _parse_positive_int(os.environ.get(MAX_PERSISTED_CHARS_ENV)) or DEFAULT_MAX_PERSISTED_CHARS


def prune_artifacts(directory: Path, keep: int) -> None:
    """Delete the oldest persisted output files, keeping ``keep`` of them."""

    if keep < 1:
        return
    try:
        candidates = sorted(
            (
                entry
                for entry in directory.iterdir()
                if entry.is_file() and _ARTIFACT_NAME_RE.match(entry.name)
            ),
            key=lambda entry: (entry.stat().st_mtime, entry.name),
        )
    except OSError:
        # Pruning is housekeeping; never fail a tool call because of it.
        return
    for entry in candidates[: max(0, len(candidates) - keep)]:
        try:
            entry.unlink()
        except OSError:
            continue


def format_character_size(char_count: int) -> str:
    """Human-readable size for character counts (decimal units)."""

    count = max(0, int(char_count))
    if count < 1_000:
        return f"{count} chars"
    if count < 1_000_000:
        return f"{count / 1_000:.1f}K chars"
    if count < 1_000_000_000:
        return f"{count / 1_000_000:.1f}M chars"
    return f"{count / 1_000_000_000:.1f}G chars"


def build_persisted_output_envelope(
    *,
    body: str,
    original_chars: int,
    persisted_path: str,
    artifact_truncated: bool = False,
) -> str:
    """Wrap an already-truncated preview into a pointer to the full artifact."""

    lines = [
        PERSISTED_OUTPUT_OPEN_TAG,
        f"Output too large ({format_character_size(original_chars)}). "
        f"Full output saved to: {persisted_path}",
    ]
    if artifact_truncated:
        lines.append("The saved file is capped as well; narrow the command output if you need the rest.")
    lines.append(_READ_HINT)
    lines.append("")
    lines.append(body)
    lines.append(PERSISTED_OUTPUT_CLOSE_TAG)
    return "\n".join(lines)


def is_persisted_output(content: str) -> bool:
    """True when ``content`` is a persisted-output envelope."""

    text = str(content or "")
    return text.startswith(PERSISTED_OUTPUT_OPEN_TAG) and PERSISTED_OUTPUT_CLOSE_TAG in text


def format_persisted_output_pointer(
    *,
    original_chars: int,
    persisted_path: str,
    artifact_truncated: bool = False,
) -> str:
    """Trailing pointer for results that must keep their first line intact.

    Error results start with ``ERROR[TYPE]:`` and are parsed by that prefix
    (``core/tool_results.py``), so they cannot be wrapped in the envelope.
    """

    pointer = (
        f"Full output saved to: {persisted_path} "
        f"({format_character_size(original_chars)})."
    )
    if artifact_truncated:
        pointer += " The saved file is capped as well."
    return f"{pointer} {_READ_HINT}"


class _InlineBuffer:
    """Retains the head and the diagnostic tail of one stream."""

    def __init__(self, head_limit: int, tail_limit: int):
        self.head_limit = max(0, int(head_limit))
        self.tail_limit = max(0, int(tail_limit))
        self.head = ""
        self.tail = ""
        self.total = 0

    def append(self, text: str) -> None:
        self.total += len(text)
        needed = self.head_limit - len(self.head)
        if needed > 0:
            self.head += text[:needed]
            text = text[needed:]
        if self.tail_limit:
            self.tail = (self.tail + text)[-self.tail_limit:]

    def raw_view(self) -> str:
        return self.head + self.tail

    def render(self) -> str:
        omitted = self.total - len(self.head) - len(self.tail)
        if omitted > 0:
            return self.head + f"\n[TRUNCATED: {omitted} characters omitted]\n" + self.tail
        return self.head + self.tail


class ShellOutputCapture:
    """Bound inline memory per stream while mirroring everything to an artifact.

    The artifact is opened lazily on the first chunk that pushes the combined
    output above ``inline_limit``; chunks seen before that point are replayed
    from a bounded history, so the file always starts at the beginning of the
    command output while tiny outputs never create a file at all.
    """

    def __init__(
        self,
        *,
        inline_limit: int,
        artifact_directory: Optional[Path] = None,
        max_persisted_chars: int = DEFAULT_MAX_PERSISTED_CHARS,
        keep_artifacts: int = DEFAULT_KEEP_ARTIFACTS,
    ) -> None:
        self.inline_limit = max(0, int(inline_limit))
        self.artifact_directory = artifact_directory
        self.max_persisted_chars = max(0, int(max_persisted_chars))
        self.keep_artifacts = max(1, int(keep_artifacts))
        self.total_chars = 0
        self.persisted_path: Optional[str] = None
        self.artifact_truncated = False
        self._buffers = {
            "stdout": _InlineBuffer(self.inline_limit, self.inline_limit),
            "stderr": _InlineBuffer(self.inline_limit, self.inline_limit),
        }
        self._handle = None
        self._persisted_chars = 0
        self._history: Optional[list[str]] = [] if self.artifact_directory else None

    @property
    def stdout(self) -> str:
        return self._buffers["stdout"].render()

    @property
    def stderr(self) -> str:
        return self._buffers["stderr"].render()

    def append(self, stream: str, text: str) -> None:
        """Record one decoded chunk of ``stdout`` or ``stderr``."""

        if not text:
            return
        buffer = self._buffers.get(str(stream))
        if buffer is None:
            return
        buffer.append(text)
        self.total_chars += len(text)
        if self._history is None:
            self._write(text)
            return
        if self.total_chars <= self.inline_limit:
            self._history.append(text)
            return
        self._replay_history()
        self._write(text)

    def close(self) -> None:
        """Flush and close the artifact, if one was opened."""

        handle = self._handle
        self._handle = None
        self._history = None
        if handle is None:
            return
        try:
            handle.flush()
            handle.close()
        except OSError:
            return

    def _replay_history(self) -> None:
        history = self._history
        self._history = None
        for chunk in history or ():
            self._write(chunk)

    def _write(self, text: str) -> None:
        if self.artifact_directory is None or self.max_persisted_chars <= 0:
            return
        if self._handle is None and not self._open_artifact():
            return
        remaining = self.max_persisted_chars - self._persisted_chars
        if remaining <= 0:
            self._note_artifact_truncated()
            return
        payload = text if len(text) <= remaining else text[:remaining]
        try:
            self._handle.write(payload)
        except OSError:
            self._disable_persistence()
            return
        self._persisted_chars += len(payload)
        if len(payload) < len(text):
            self._note_artifact_truncated()

    def _open_artifact(self) -> bool:
        directory = self.artifact_directory
        if directory is None:
            return False
        try:
            directory.mkdir(parents=True, exist_ok=True)
            prune_artifacts(directory, self.keep_artifacts)
            path = directory / (
                f"{_ARTIFACT_NAME_PREFIX}{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.log"
            )
            self._handle = path.open("w", encoding="utf-8", errors="replace", newline="")
        except OSError:
            self._disable_persistence()
            return False
        self.persisted_path = str(path)
        return True

    def _note_artifact_truncated(self) -> None:
        if self.artifact_truncated:
            return
        self.artifact_truncated = True
        try:
            self._handle.write(_TRUNCATION_MARKER.format(limit=self.max_persisted_chars))
        except (AttributeError, OSError):
            return

    def _disable_persistence(self) -> None:
        """Fall back to plain inline truncation when the artifact cannot be used."""

        handle = self._handle
        self._handle = None
        self._history = None
        self.artifact_directory = None
        self.persisted_path = None
        if handle is None:
            return
        try:
            handle.close()
        except OSError:
            return


def _parse_positive_int(value: Optional[str]) -> Optional[int]:
    if value is None or not str(value).strip():
        return None
    try:
        parsed = int(str(value).strip(), 10)
    except ValueError:
        return None
    return parsed if parsed > 0 else None
