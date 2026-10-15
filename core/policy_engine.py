from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List

from core.tool_policy import ToolMetadata


# ---------------------------------------------------------------------------
# Combined regex patterns — one alternation per category instead of N
# individual patterns.  This reduces classify_shell_command from ~50
# sequential re.search calls to a single re.search per category.
# ---------------------------------------------------------------------------

# Inspect-only commands are matched against the FIRST token of the command only.
# Substring matching (e.g. "ss" inside "node ss.js") previously let arbitrary
# executables masquerade as read-only inspection commands.
_INSPECT_ONLY_COMMAND_NAMES = frozenset({
    "get-process", "tasklist", "where-object", "select-object", "findstr",
    "get-childitem", "get-content", "select-string", "dir", "type", "where",
    "netstat", "ss", "ps",
})
_NETWORK_DIAGNOSTIC_COMMAND_RE = re.compile(
    r"\b(?:ping(?:\.exe)?|test-netconnection|resolve-dnsname|"
    r"nslookup(?:\.exe)?|tracert(?:\.exe)?|pathping(?:\.exe)?)\b",
    re.IGNORECASE,
)
_MUTATING_COMMAND_RE = re.compile(
    r"\b(?:taskkill|stop-process|remove-item|rm|del|rmdir|move-item|rename-item|"
    r"copy-item|set-content|add-content|out-file|new-item|npm\s+install|npm\s+uninstall|"
    r"pip\s+install|python\s+-m\s+pip\s+install|git\s+checkout|git\s+reset)\b"
    r"|python\s+-c\b.*\b(?:write_text|write_bytes|open\s*\([^)]*,\s*['\"]?[wax])",
    re.IGNORECASE,
)
_DESTRUCTIVE_COMMAND_RE = re.compile(
    r"\b(?:taskkill|stop-process|remove-item|rm|del|rmdir)\b",
    re.IGNORECASE,
)
_LONG_RUNNING_SERVICE_RE = re.compile(
    r"(?:^|[;&|()])\s*(?:python(?:3(?:\.\d+)?)?\s+-m\s+http\.server|http-server|npm\s+start|"
    r"npm\s+run\s+dev|npm\s+exec\b.*\bhttp-server\b|npx\b.*\bhttp-server\b|"
    r"uvicorn|flask\s+run|webpack(?:\.cmd)?\s+serve|serve(?:\.cmd|\.ps1|\.exe)?)(?=$|[;&|()\s])",
    re.IGNORECASE,
)
_HTTP_PROBE_COMMAND_RE = re.compile(
    r"\b(?:curl(?:\.exe)?|invoke-webrequest|iwr|invoke-restmethod|irm)\b",
    re.IGNORECASE,
)
_HTTP_WRITE_METHOD_RE = re.compile(
    r"(?:--request|-x|\b-method\b)\s*[:=]?[\'\"]?(post|put|patch|delete)\b",
    re.IGNORECASE,
)
_HTTP_WRITE_FLAG_RE = re.compile(
    r"(?:^|[\s;|&])-(?:d|f|t)\b"
    r"|--(?:data(?:-raw|-binary|-ascii|-urlencode)?|form|string|upload-file|json)\b",
    re.IGNORECASE,
)
# HTTP clients that write the response body to disk (download-to-file). These are
# mutating even without an explicit write method (POST/PUT/...).
_HTTP_OUTPUT_FLAG_RE = re.compile(
    r"(?:^|[\s;|&])-o(?:[=\s]|$)"
    r"|(?:^|[\s;|&])--(?:output(?:-dir)?|remote-name)\b"
    r"|(?:^|[\s;|&])-OutFile\b",
    re.IGNORECASE,
)
_RIPGREP_COMMAND_NAMES = {"rg", "rg.exe"}

# PowerShell built-in aliases resolved against the real shell (verified with
# ``Get-Alias``).  The classifier used to look at the first token only, so a
# harmless read-only first segment (``Get-Content a.txt | ...``) whitelisted the
# WHOLE command and hid mutating aliases such as ``| ac b.txt`` (Add-Content),
# ``| ri a.txt`` (Remove-Item) or ``| mv a b`` behind ``inspect_only``.
_SHELL_ALIAS_TARGETS = {
    "ac": "add-content",
    "cat": "get-content",
    "cpi": "copy-item",
    "clc": "clear-content",
    "clv": "clear-variable",
    "clear": "clear-host",
    "compare": "compare-object",
    "copy": "copy-item",
    "cp": "copy-item",
    "del": "remove-item",
    "echo": "write-output",
    "erase": "remove-item",
    "ft": "format-table",
    "fw": "format-wide",
    "gci": "get-childitem",
    "gc": "get-content",
    "gi": "get-item",
    "group": "group-object",
    "gv": "get-variable",
    "icm": "invoke-command",
    "iex": "invoke-expression",
    "ls": "get-childitem",
    "man": "help",
    "measure": "measure-object",
    "mount": "new-psdrive",
    "mv": "move-item",
    "ni": "new-item",
    "nmo": "new-module",
    "pwd": "get-location",
    "ren": "rename-item",
    "rename": "rename-item",
    "ri": "remove-item",
    "rm": "remove-item",
    "rmdir": "remove-item",
    "rd": "remove-item",
    "saps": "start-process",
    "sc": "set-content",
    "select": "select-object",
    "sls": "select-string",
    "sort": "sort-object",
    "start": "start-process",
    "tee": "tee-object",
    "where": "where-object",
    "write": "write-output",
    "?": "where-object",
    "%": "foreach-object",
}
# Aliases that mutate the workspace or spawn processes.  Only consulted for the
# LEADING token of a segment, so ``rg cp`` stays a cheap read-only search.
_MUTATING_ALIAS_TARGETS = frozenset({
    "add-content",
    "clear-content",
    "clear-variable",
    "copy-item",
    "invoke-expression",
    "move-item",
    "new-item",
    "new-module",
    "new-psdrive",
    "remove-item",
    "rename-item",
    "set-content",
    "start-process",
    "tee-object",
})
_DESTRUCTIVE_ALIAS_TARGETS = frozenset({"remove-item"})
# Redirection characters: the remainder of the redirect target is consumed with
# the segment, so ``2>$null`` / ``2>&1`` never splits a read-only command.
_REDIRECT_LEADING_CHARS = "<>"


def _is_http_probe_command(command: str) -> bool:
    return bool(_HTTP_PROBE_COMMAND_RE.search(command))


def _is_http_write_command(command: str) -> bool:
    if not _is_http_probe_command(command):
        return False
    if _HTTP_WRITE_METHOD_RE.search(command):
        return True
    if _HTTP_OUTPUT_FLAG_RE.search(command):
        return True
    return bool(_HTTP_WRITE_FLAG_RE.search(command))


def _has_unquoted_shell_operator(command: str) -> bool:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(command):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in ("'", '"'):
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if quote is not None:
            continue
        if char in (";", "|", "<", ">"):
            return True
        if char == "&":
            stripped_before = command[:index].strip()
            if stripped_before:
                return True
        if char == "`":
            return True
    return False


# Redirection targets that are not files: discarding stderr into ``$null`` or
# merging streams (``2>&1``) does not write anything to disk.
_NON_FILE_REDIRECT_TARGET_RE = re.compile(r"^\s*(?:\$null\b|[&$]\s*[\d*]?\s*$)")


def _has_unquoted_write_redirect(command: str) -> bool:
    """True when the command contains an output redirection (``>`` / ``>>``) outside quotes.

    Stream redirects (``2>&1``) and discards (``> $null``) are ignored: they do
    not create or modify a file, so they must not escalate a read-only command.
    """
    quote: str | None = None
    escaped = False
    index = 0
    text = str(command or "")
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote == '"':
            escaped = True
            index += 1
            continue
        if char in ("'", '"'):
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            index += 1
            continue
        if quote is not None:
            index += 1
            continue
        if char != ">":
            index += 1
            continue
        # Skip ``>>`` and consume the target that follows the redirect operator.
        index += 1
        if text[index : index + 1] == ">":
            index += 1
        target = text[index:].lstrip()
        if _NON_FILE_REDIRECT_TARGET_RE.match(target):
            index = len(text)
            continue
        return True
    return False


def _first_command_token(command: str) -> str:
    raw = command.strip()
    if raw.startswith("&"):
        raw = raw[1:].lstrip()
    if not raw:
        return ""

    quote: str | None = None
    token_chars: list[str] = []
    for char in raw:
        if char in ("'", '"'):
            if quote == char:
                quote = None
                continue
            if quote is None:
                quote = char
                continue
        if quote is None and char.isspace():
            break
        token_chars.append(char)
    return "".join(token_chars).strip()


def _strip_trailing_redirection(command: str) -> str:
    """Drop trailing stream redirects (``2>$null``, ``2>&1``) from a segment.

    Segment splitting already isolates redirect targets, so a read-only segment
    may carry a redirect suffix.  Non-file redirects are removed; file redirects
    are kept so the write check still sees them.
    """
    text = str(command or "").strip()
    while text:
        match = re.search(r"(?<=[^\d&])\d*\s*>>?\s*(?P<target>\S+)\s*$", text)
        if match and _NON_FILE_REDIRECT_TARGET_RE.match(match.group("target")):
            text = text[: match.start()].strip()
            continue
        return text
    return text


def _is_ripgrep_read_only_command(command: str) -> bool:
    normalized = _strip_trailing_redirection(command).strip()
    if not normalized or _has_unquoted_shell_operator(normalized):
        return False
    if "$(" in normalized or "%{" in normalized:
        return False

    token = _first_command_token(normalized).strip('"\'')
    if not token:
        return False
    executable = token.replace("/", "\\").rsplit("\\", 1)[-1].lower()
    return executable in _RIPGREP_COMMAND_NAMES


def _leading_command_executable(command: str) -> str:
    token = _first_command_token(command).strip('"\'')
    if not token:
        return ""
    executable = token.replace("/", "\\").rsplit("\\", 1)[-1].lower()
    if executable.endswith(".exe"):
        executable = executable[:-4]
    return executable


def _resolve_alias_executable(executable: str) -> str:
    """Map a PowerShell built-in alias to the cmdlet it stands for."""
    normalized = str(executable or "").strip().lower()
    return _SHELL_ALIAS_TARGETS.get(normalized, normalized)


def _is_inspect_only_leading_command(command: str) -> bool:
    return _resolve_alias_executable(_leading_command_executable(command)) in _INSPECT_ONLY_COMMAND_NAMES


def _split_command_segments(command: str) -> List[str]:
    """Split a command line into the individual commands the shell will run.

    Splits on unquoted ``;``, ``|``, ``&`` and newlines while keeping
    redirections (``2>&1``, ``2>$null``) attached to the command that owns them,
    so a redirect never manufactures a bogus extra segment.
    """
    segments: List[str] = []
    current: List[str] = []
    quote: str | None = None
    escaped = False
    redirect_open = False
    for char in str(command or ""):
        if escaped:
            current.append(char)
            escaped = False
            redirect_open = False
            continue
        if char == "\\" and quote == '"':
            current.append(char)
            escaped = True
            redirect_open = False
            continue
        if char == "`":
            current.append(char)
            escaped = True
            redirect_open = False
            continue
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            current.append(char)
            continue
        if char in _REDIRECT_LEADING_CHARS:
            redirect_open = True
            current.append(char)
            continue
        if char == "&" and redirect_open:
            # ``2>&1`` / ``>&2``: the redirection belongs to this command.
            redirect_open = False
            current.append(char)
            continue
        if char in (";", "|", "&", "\n", "\r"):
            segments.append("".join(current))
            current = []
            redirect_open = False
            continue
        current.append(char)
        redirect_open = False
    segments.append("".join(current))
    return [segment.strip() for segment in segments if segment.strip()]


def _script_block_bodies(command: str) -> List[str]:
    """Return the bodies of unquoted ``{ ... }`` script blocks inside a command."""
    bodies: List[str] = []
    text = str(command or "")
    quote: str | None = None
    escaped = False
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char == "`":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "{":
            if depth == 0:
                start = index + 1
            depth += 1
            continue
        if char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    bodies.append(text[start:index])
                    start = -1
    return bodies


def _is_trivial_expression(text: str) -> bool:
    """True for literals and property/value access that cannot invoke a command.

    Covers ``1``, ``'a b'``, ``$x``, ``$_.Length``, ``$x.Count`` and quoted
    strings, so ``$x = 1`` or ``$x = $_.Name`` stay read-only.  Anything with a
    real command token (``Remove-Item``, ``Get-Process``) is not trivial and is
    classified normally.
    """
    stripped = str(text or "").strip()
    if not stripped:
        return True
    if stripped[0] in ("'", '"') and stripped[-1] == stripped[0] and len(stripped) >= 2:
        return True
    if re.fullmatch(r"[\w.]+", stripped):
        return True
    if re.fullmatch(r"\$[\w:]+\s*(?:=[^=]\S*)?", stripped):
        return True
    if re.fullmatch(r"\$\w[\w:]*\.\w+", stripped):
        return True
    try:
        float(stripped)
        return True
    except ValueError:
        return False


def _is_expression_segment(segment: str) -> bool:
    """True when the segment is a PowerShell expression rather than a command.

    ``$_.Length -gt 0`` and ``(Get-Process).Count`` evaluate an expression and
    never invoke a cmdlet on their own, so they must not be treated as unknown
    commands.  Anything they DO invoke lives in an assignment or a
    sub-expression, both of which are extracted separately.
    """
    stripped = str(segment or "").lstrip()
    return stripped.startswith(("$", "(", "@("))


def _split_assignment_rhs(segment: str) -> str:
    """Return the right-hand side of a top-level ``$var = ...`` assignment."""
    text = str(segment or "")
    quote: str | None = None
    escaped = False
    depth = 0
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char == "`":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char in "([{":
            depth += 1
            continue
        if char in ")]}":
            depth = max(0, depth - 1)
            continue
        if depth:
            continue
        if char == "=" and text[index - 1 : index] not in ("!", "<", ">", "="):
            return text[index + 1 :].strip()
    return ""


def _subexpression_bodies(segment: str) -> List[str]:
    """Return bodies of unquoted ``$( )``, ``@( )`` and ``( )`` sub-expressions."""
    bodies: List[str] = []
    text = str(segment or "")
    quote: str | None = None
    escaped = False
    depths: List[int] = []
    start = -1
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char == "`":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "(":
            if not depths:
                start = index + 1
            depths.append(1)
            continue
        if char == ")" and depths:
            depths[-1] -= 1
            if not depths[-1]:
                bodies.append(text[start:index])
                start = -1
                depths.pop()
    return bodies


def _iter_effective_commands(command: str, _depth: int = 0) -> Iterator[str]:
    """Yield every command the shell will actually execute.

    Pipeline segments, assignments and nested script-block/sub-expression bodies
    are separate executions, so each one is classified on its own; a read-only
    segment must not vouch for the rest of the line.
    """
    for segment in _split_command_segments(command):
        if _is_expression_segment(segment):
            # Pure expressions execute nothing on their own; recurse into the
            # commands they embed (assignments and sub-expressions).
            rhs = _split_assignment_rhs(segment)
            if rhs and _depth < 4:
                for nested in _iter_effective_commands(rhs, _depth + 1):
                    yield nested
                if not _is_trivial_expression(rhs):
                    yield rhs
            for body in _subexpression_bodies(segment):
                nested = body.strip()
                if nested and _depth < 4:
                    yield from _iter_effective_commands(nested, _depth + 1)
            continue
        if _is_trivial_expression(segment) and not _is_read_only_segment(segment):
            # A bare literal that reached the top level runs nothing.  Known
            # read-only commands (``ls``, ``dir``, ``Get-Process``) are exempt so
            # their argument-less form keeps working.
            continue
        yield segment
        if _depth >= 4:
            continue
        for body in _script_block_bodies(segment):
            nested = body.strip()
            if nested:
                yield from _iter_effective_commands(nested, _depth + 1)


def _segment_is_destructive(segment: str) -> bool:
    executable = _resolve_alias_executable(_leading_command_executable(segment))
    return executable in _DESTRUCTIVE_ALIAS_TARGETS


def _segment_is_mutating(segment: str) -> bool:
    executable = _resolve_alias_executable(_leading_command_executable(segment))
    return executable in _MUTATING_ALIAS_TARGETS


def _is_read_only_segment(segment: str) -> bool:
    """Deny-by-default: a segment is read-only only when positively recognized."""
    normalized = str(segment or "").strip()
    if not normalized:
        return False
    if (
        _segment_is_mutating(normalized)
        or _segment_is_destructive(normalized)
        or _has_unquoted_write_redirect(normalized)
        or _is_http_write_command(normalized)
        or bool(_LONG_RUNNING_SERVICE_RE.search(normalized))
        or bool(_MUTATING_COMMAND_RE.search(normalized))
        or bool(_DESTRUCTIVE_COMMAND_RE.search(normalized))
    ):
        return False
    executable = _resolve_alias_executable(_leading_command_executable(normalized))
    if executable in _INSPECT_ONLY_COMMAND_NAMES:
        return True
    if _is_ripgrep_read_only_command(normalized):
        return True
    if bool(_NETWORK_DIAGNOSTIC_COMMAND_RE.search(normalized)):
        return True
    return _is_http_probe_command(normalized)


def classify_shell_command(command: str) -> Dict[str, Any]:
    normalized = str(command or "").strip()
    is_long_running_service = bool(_LONG_RUNNING_SERVICE_RE.search(normalized))
    is_destructive = bool(_DESTRUCTIVE_COMMAND_RE.search(normalized))
    is_http_write = _is_http_write_command(normalized)
    is_redirect_write = _has_unquoted_write_redirect(normalized)
    is_mutating = (
        is_long_running_service
        or is_destructive
        or is_http_write
        or is_redirect_write
        or bool(_MUTATING_COMMAND_RE.search(normalized))
    )
    is_network_diagnostic = bool(_NETWORK_DIAGNOSTIC_COMMAND_RE.search(normalized))

    # Every command the shell will run is classified on its own, so a read-only
    # first segment cannot vouch for a mutating alias later in the pipeline.
    effective_commands = list(_iter_effective_commands(normalized))
    # A line that executes nothing but literals/assignments (``$x = 1``) cannot
    # mutate anything, while an empty command is never whitelisted.
    all_segments_read_only = bool(normalized) and all(
        _is_read_only_segment(segment) for segment in effective_commands
    )
    if not all_segments_read_only:
        is_mutating = True
        is_destructive = is_destructive or any(
            _segment_is_destructive(segment) for segment in effective_commands
        )
    is_inspect_only = all_segments_read_only and not is_long_running_service
    return {
        "normalized": normalized,
        "inspect_only": is_inspect_only,
        "mutating": is_mutating,
        "destructive": is_destructive,
        "long_running_service": is_long_running_service,
        "network_diagnostic": is_network_diagnostic,
        "http_probe": _is_http_probe_command(normalized),
    }


def shell_command_requires_approval(command: str) -> bool:
    profile = classify_shell_command(command)
    if profile.get("inspect_only") and not profile.get("long_running_service"):
        return False
    return True


def tool_requires_approval(
    tool_name: str,
    tool_args: Dict[str, Any] | None = None,
    *,
    metadata: ToolMetadata | None = None,
    approvals_enabled: bool = True,
) -> bool:
    if not approvals_enabled:
        return False

    normalized_name = str(tool_name or "").strip().lower()
    effective_metadata = metadata or ToolMetadata(name=str(tool_name or "unknown_tool"))
    if normalized_name == "cli_exec":
        command = str(((tool_args or {}).get("command")) or "").strip()
        if not command:
            return True
        return shell_command_requires_approval(command)

    return bool(
        effective_metadata.requires_approval
        or effective_metadata.destructive
        or effective_metadata.mutating
    )
