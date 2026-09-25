"""Return-code semantics for cli_exec commands.

A command line is split into executable segments (``;``, ``&&``, ``||``, ``|``,
quotes, escaping) instead of being probed with one flat regex, so a command name
only counts when it is the executable of a segment and not a mere word inside an
argument list (``echo grep`` no longer looks like a ``grep`` run).
Ported from ZCode's Bash tool (``.bash_tool/bash_tool/semantics.py``).

The neutral set below intentionally stays the one cli_exec already treated as a
protocol exit: widening it (``find``, ``test``, ``[``, ...) would silently hide
real failures, and the parser fix alone removes the false positives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "EXIT_CODE_NEUTRAL_MESSAGES",
    "CommandSegment",
    "analyze_command",
    "interpret_non_error_exit",
]

_OPERATORS = ("&&", "||", ";", "|")

# Exit code 1 is part of the protocol for these commands: their output is the
# result the agent needs, not an execution failure.
EXIT_CODE_NEUTRAL_MESSAGES: dict[str, str] = {
    "diff": "Files differ",
    "egrep": "No matches found",
    "fgrep": "No matches found",
    "findstr": "No matches found",
    "grep": "No matches found",
    "pytest": "Tests failed",
    "rg": "No matches found",
    "select-string": "No matches found",
    "vulture": "Dead code found",
}
_GIT_NEUTRAL_MESSAGES = {"diff": "Files differ", "grep": "No matches found"}
# ``sudo time grep x`` still runs grep; assignments like ``VAR=1 rg x`` do too.
_WRAPPER_COMMANDS = frozenset({"command", "nohup", "sudo", "time"})
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_EXECUTABLE_SUFFIXES = (".exe", ".cmd", ".bat", ".ps1")
_FALLBACK_NEUTRAL_RE = re.compile(
    r"(?:^|[;&|()\s])(?P<name>"
    + "|".join(sorted((re.escape(name) for name in EXIT_CODE_NEUTRAL_MESSAGES), key=len, reverse=True))
    + r")(?=$|[;&|()\s])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CommandSegment:
    """One command inside a composed command line."""

    name: str
    argv: tuple[str, ...]
    operator_before: str | None = None


def analyze_command(command: str) -> list[CommandSegment] | None:
    """Split a command line into executable segments.

    Returns ``None`` when the line cannot be tokenized (unbalanced quotes), so
    the caller can fall back to boundary matching.
    """

    segments: list[CommandSegment] = []
    for operator, raw_segment in _split_segments(str(command or "")):
        argv = _tokenize(raw_segment)
        if argv is None:
            return None
        if not argv:
            continue
        name = _executable_name(argv[0])
        if not name:
            continue
        segments.append(CommandSegment(name=name, argv=tuple(argv), operator_before=operator))
    return segments


def interpret_non_error_exit(command: str, exit_code: int) -> str | None:
    """Describe a non-zero exit code that is part of a command's protocol.

    Returns ``None`` when the exit code must be reported as a real failure.
    """

    if exit_code != 1:
        return None

    segments = analyze_command(command)
    if segments is None:
        return _interpret_unparsable(command)
    for argv in _candidate_argv(segments):
        message = _interpret_argv(argv)
        if message:
            return message
    return None


def _candidate_argv(segments: Sequence[CommandSegment]) -> list[tuple[str, ...]]:
    """Argv lists whose status can be the exit code the shell reports.

    Pipelines report the status of any native leaf, ``&&``/``||`` stop at the
    first failure, and ``;`` reports only the last command — so the last
    pipeline is always considered, and the groups before it only while they are
    chained with ``&&``/``||``.
    """

    groups: list[list[tuple[str, ...]]] = []
    group_operators: list[str | None] = []
    for segment in segments:
        if not groups or segment.operator_before != "|":
            groups.append([])
            group_operators.append(segment.operator_before)
        groups[-1].append(segment.argv)

    candidates = list(groups[-1])
    index = len(groups) - 1
    while index > 0 and group_operators[index] in ("&&", "||"):
        index -= 1
        candidates = list(groups[index]) + candidates
    return candidates


def _interpret_argv(argv: Sequence[str]) -> str | None:
    tokens = _strip_leading_wrappers(argv)
    if not tokens:
        return None
    name = _executable_name(tokens[0])
    if name == "git":
        subcommand = _first_non_flag(tokens[1:], skip_next_after=("-C", "-c"))
        return _GIT_NEUTRAL_MESSAGES.get(subcommand or "")
    return EXIT_CODE_NEUTRAL_MESSAGES.get(name)


def _strip_leading_wrappers(argv: Sequence[str]) -> list[str]:
    tokens = list(argv)
    while tokens:
        if _ASSIGNMENT_RE.match(tokens[0]) or _executable_name(tokens[0]) in _WRAPPER_COMMANDS:
            tokens.pop(0)
            continue
        break
    return tokens


def _executable_name(token: str) -> str:
    raw = str(token or "").strip().strip("\"'")
    if not raw:
        return ""
    # PowerShell call operator (``& "C:\\tools\\rg.exe"``) and path separators.
    raw = raw.lstrip("&").strip().replace("\\", "/").rstrip("/")
    name = raw.rsplit("/", 1)[-1].lower()
    for suffix in _EXECUTABLE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _first_non_flag(argv: Sequence[str], skip_next_after: Iterable[str]) -> str | None:
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("-"):
            if arg in skip_next_after:
                skip_next = True
            continue
        return arg
    return None


def _interpret_unparsable(command: str) -> str | None:
    """Boundary fallback for lines the parser rejects (unbalanced quotes)."""

    matches = list(_FALLBACK_NEUTRAL_RE.finditer(str(command or "")))
    if not matches:
        return None
    return EXIT_CODE_NEUTRAL_MESSAGES.get(matches[-1].group("name").lower())


def _split_segments(command: str) -> list[tuple[str | None, str]]:
    """Split on ``;``, ``&&``, ``||``, ``|`` outside quotes and escapes."""

    segments: list[tuple[str | None, str]] = []
    buffer: list[str] = []
    operator_before: str | None = None
    quote: str | None = None
    index = 0
    length = len(command)

    while index < length:
        char = command[index]
        if quote is None and char == "\\":
            buffer.append(char)
            index += 1
            if index < length:
                buffer.append(command[index])
                index += 1
            continue
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            buffer.append(char)
            index += 1
            continue
        if quote is None:
            matched = next((op for op in _OPERATORS if command.startswith(op, index)), None)
            if matched is not None:
                segments.append((operator_before, "".join(buffer)))
                operator_before = matched
                buffer = []
                index += len(matched)
                continue
        buffer.append(char)
        index += 1

    segments.append((operator_before, "".join(buffer)))
    return segments


def _tokenize(segment: str) -> list[str] | None:
    """Split one segment into argv tokens; ``None`` when quotes are unbalanced."""

    tokens: list[str] = []
    buffer: list[str] = []
    quote: str | None = None
    index = 0
    length = len(segment)
    has_token = False

    while index < length:
        char = segment[index]
        if quote is None and char == "\\":
            index += 1
            if index < length:
                buffer.append(segment[index])
                has_token = True
            index += 1
            continue
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            else:
                buffer.append(char)
            has_token = True
            index += 1
            continue
        if quote is None and char.isspace():
            if has_token:
                tokens.append("".join(buffer))
                buffer = []
                has_token = False
            index += 1
            continue
        buffer.append(char)
        has_token = True
        index += 1

    if quote is not None:
        return None
    if has_token:
        tokens.append("".join(buffer))
    return tokens
