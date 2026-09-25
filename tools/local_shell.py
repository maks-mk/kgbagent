import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import base64
import locale
import os
import re
import shutil
import subprocess
from typing import Annotated, Any, Callable, Iterator, Optional
from langchain_core.tools import tool
from pydantic import Field

from core.utils import truncate_output
from core.errors import format_error, ErrorType
from core.safety_policy import SafetyPolicy
from core.policy_engine import classify_shell_command
from tools.shell_output import (
    ShellOutputCapture,
    build_persisted_output_envelope,
    format_persisted_output_pointer,
    resolve_artifact_directory,
    resolve_max_persisted_chars,
)
from tools.shell_semantics import interpret_non_error_exit

# Constants
DEFAULT_TIMEOUT = 120
# Upper bound parity with the reference Bash tool (bash-timeout-policy.ts:
# default 120000 ms, max 600000 ms), overridable through the env var below.
BASH_TOOL_MAX_TIMEOUT_MS = 600_000
MAX_TIMEOUT_ENV = "CLI_EXEC_MAX_TIMEOUT_MS"

# Global settings
_SAFETY_POLICY: Optional[SafetyPolicy] = None
_WORKING_DIRECTORY: str = os.getcwd()  # Default to the current process working directory
_CLI_OUTPUT_EMITTER: Optional[Callable[[dict[str, str]], None]] = None
_CLI_TOOL_ID: ContextVar[str] = ContextVar("cli_tool_id", default="")

_WINDOWS_COMMAND_HINTS = {
    "cat": "type <file> (or Get-Content <file>)",
    "ls": "dir (or Get-ChildItem)",
    "pwd": "cd (or Get-Location)",
    "cp": "copy (or Copy-Item)",
    "mv": "move (or Move-Item)",
    "rm": "del (or Remove-Item)",
    "grep": "findstr <pattern> <file> (or Select-String ...)",
    "head": "Get-Content <file> -TotalCount N",
    "tail": "Get-Content <file> -Tail N",
    "which": "where <command> (or Get-Command <name>)",
    "clear": "cls (or Clear-Host)",
}

_WINDOWS_PYTHON_HEREDOC_RE = re.compile(
    r"^(?P<prefix>[\s\S]*?)(?P<exe>python(?:3(?:\.\d+)?)?|py(?:\.exe)?)"
    r"(?P<flags>(?:[ \t]+-[^<\r\n]+)*)[ \t]+-[ \t]*<<[ \t]*"
    r"(?P<quote>['\"]?)(?P<tag>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)[ \t]*\r?\n"
    r"(?P<body>[\s\S]*?)\r?\n(?P<closing>[ \t]*)(?P=tag)[ \t]*(?:\r?\n|$)"
    r"(?P<suffix>[\s\S]*)$",
    re.IGNORECASE,
)
_WINDOWS_POWERSHELL_WRAPPER_RE = re.compile(
    r"^\s*(?:pwsh|powershell(?:\.exe)?)\s+(?:(?:-(?:NoProfile|NoLogo|NonInteractive))\s+)*-Command\s+(?P<body>[\s\S]+?)\s*$",
    re.IGNORECASE,
)
_WINDOWS_NULL_OUTPUT_FLAG_RE = re.compile(
    r"(?i)(?P<flag>--output|-o)\s+(?P<quote>['\"]?)/dev/null(?P=quote)"
)
_WINDOWS_NULL_OUTPUT_FLAG_EQ_RE = re.compile(
    r"(?i)(?P<flag>--output=)(?P<quote>['\"]?)/dev/null(?P=quote)"
)
_WINDOWS_NULL_REDIRECT_RE = re.compile(
    r"(?i)(?P<redir>\d?>)\s*(?P<quote>['\"]?)/dev/null(?P=quote)"
)
_NPM_LIKE_COMMAND_RE = re.compile(r"(^|[;&|()\s])(?:npm|npx)(?=$|[;&|()\s])", re.IGNORECASE)
_INTERACTIVE_PROMPT_COMMAND_RE = re.compile(
    r"(^|[;&|()\s])(?:npm|npx|pnpm|yarn)(?=$|[;&|()\s])",
    re.IGNORECASE,
)
_INSPECT_ONLY_COMMAND_PATTERNS = (
    re.compile(r"\bget-process\b"),
    re.compile(r"\btasklist\b"),
    re.compile(r"\bwhere-object\b"),
    re.compile(r"\bselect-object\b"),
    re.compile(r"\bfindstr\b"),
    re.compile(r"\bget-childitem\b"),
    re.compile(r"\bget-content\b"),
    re.compile(r"\bselect-string\b"),
    re.compile(r"\bdir\b"),
    re.compile(r"\btype\b"),
    re.compile(r"\bwhere\b"),
    re.compile(r"\bnetstat\b"),
    re.compile(r"\bss\b"),
    re.compile(r"\bps\b"),
)
_MUTATING_COMMAND_PATTERNS = (
    re.compile(r"\btaskkill\b"),
    re.compile(r"\bstop-process\b"),
    re.compile(r"\bremove-item\b"),
    re.compile(r"\brm\b"),
    re.compile(r"\bdel\b"),
    re.compile(r"\brmdir\b"),
    re.compile(r"\bmove-item\b"),
    re.compile(r"\brename-item\b"),
    re.compile(r"\bcopy-item\b"),
    re.compile(r"\bset-content\b"),
    re.compile(r"\badd-content\b"),
    re.compile(r"\bnpm\s+install\b"),
    re.compile(r"\bnpm\s+uninstall\b"),
    re.compile(r"\bpip\s+install\b"),
    re.compile(r"\bgit\s+checkout\b"),
)
_DESTRUCTIVE_COMMAND_PATTERNS = (
    re.compile(r"\btaskkill\b"),
    re.compile(r"\bstop-process\b"),
    re.compile(r"\bremove-item\b"),
    re.compile(r"\brm\b"),
    re.compile(r"\bdel\b"),
    re.compile(r"\brmdir\b"),
)
# Commands that use a non-zero exit code as part of their normal protocol
# (grep/rg → 1 = no matches, vulture → 1 = dead code found, pytest → 1 = test
# failures, diff → 1 = files differ) are interpreted by tools.shell_semantics,
# which matches real executable segments instead of words inside arguments.

_LONG_RUNNING_SERVICE_PATTERNS = (
    re.compile(r"\bpython(?:3(?:\.\d+)?)?\s+-m\s+http\.server\b"),
    re.compile(r"\bhttp-server\b"),
    re.compile(r"\bnpm\s+start\b"),
    re.compile(r"\bnpm\s+run\s+dev\b"),
    re.compile(r"\bnpm\s+exec\b.*\bhttp-server\b"),
    re.compile(r"\bnpx\b.*\bhttp-server\b"),
    re.compile(r"\buvicorn\b"),
    re.compile(r"\bflask\s+run\b"),
    re.compile(r"\bwebpack(?:\.cmd)?\s+serve\b"),
    re.compile(r"\bserve\b"),
)


def _get_windows_command_hint(command: str, stderr: str) -> str:
    """Return a friendly Windows-specific hint for common Unix commands."""
    if os.name != "nt":
        return ""

    lower_stderr = stderr.lower()
    if "is not recognized as an internal or external command" not in lower_stderr:
        return ""

    parts = command.strip().split()
    if not parts:
        return ""

    first_token = parts[0].strip("\"'").lower()
    suggestion = _WINDOWS_COMMAND_HINTS.get(first_token)
    if not suggestion:
        return ""
    return f"\nHint (Windows): command '{first_token}' was not found. Try: {suggestion}."


def _normalize_windows_python_heredoc(command: str) -> str:
    """Converts bash-style python heredoc into a PowerShell-compatible command body on Windows."""
    if os.name != "nt":
        return command

    match = _WINDOWS_PYTHON_HEREDOC_RE.match(command.strip())
    if not match:
        return command

    prefix = match.group("prefix")
    suffix = match.group("suffix")
    exe = match.group("exe")
    flags = match.group("flags")
    body = match.group("body").replace("\r\n", "\n").replace("\r", "\n")
    closing = match.group("closing")
    if closing:
        # Bash commonly permits an indented terminator in generated snippets;
        # preserve Python's relative indentation by removing the same margin.
        lines = body.split("\n")
        nonempty = [line for line in lines if line.strip()]
        if nonempty:
            margin = min(len(line) - len(line.lstrip(" \t")) for line in nonempty)
            body = "\n".join(line[margin:] if line.strip() else line for line in lines)

    # Base64 prevents PowerShell from interpreting Python quotes, $, backticks,
    # braces, or here-string-looking text before it reaches Python.
    payload = base64.b64encode(body.encode("utf-8")).decode("ascii")
    converted = f"[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{payload}')) | {exe}{flags} -"
    return f"{prefix}{converted}{suffix}"


def _strip_nested_windows_powershell_wrapper(command: str) -> str:
    """
    cli_exec already runs PowerShell on Windows.
    If the model wraps the command in `powershell -Command ...`, unwrap it so
    `$vars` are not expanded by an outer shell layer.
    """
    if os.name != "nt":
        return command
    raw = str(command or "")
    match = _WINDOWS_POWERSHELL_WRAPPER_RE.match(raw.strip())
    if not match:
        return command
    body = str(match.group("body") or "").strip()
    if len(body) >= 2 and body[0] == body[-1] and body[0] in {'"', "'"}:
        body = body[1:-1]
    return body or command


def _normalize_windows_null_device(command: str) -> str:
    """Maps POSIX null sink paths to Windows `NUL` for common CLI patterns."""
    if os.name != "nt":
        return command

    raw = str(command or "")
    if "/dev/null" not in raw.lower():
        return raw

    normalized = _WINDOWS_NULL_OUTPUT_FLAG_RE.sub(
        lambda match: f"{match.group('flag')} {match.group('quote')}NUL{match.group('quote')}",
        raw,
    )
    normalized = _WINDOWS_NULL_OUTPUT_FLAG_EQ_RE.sub(
        lambda match: f"{match.group('flag')}{match.group('quote')}NUL{match.group('quote')}",
        normalized,
    )
    normalized = _WINDOWS_NULL_REDIRECT_RE.sub(
        lambda match: f"{match.group('redir')} {match.group('quote')}NUL{match.group('quote')}",
        normalized,
    )
    return normalized


def _prepare_shell_env(command: str) -> dict[str, str]:
    env = os.environ.copy()
    # Best-effort non-interactive mode for CLI tools that support CI semantics.
    env.setdefault("CI", "1")

    if os.name == "nt":
        # Keep Python child processes on the same UTF-8 protocol as the shell.
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")

    if _NPM_LIKE_COMMAND_RE.search(command or ""):
        # Prevent npm/npx confirmation prompts such as "Ok to proceed? (y)".
        env.setdefault("npm_config_yes", "true")
        env.setdefault("npm_config_audit", "false")
        env.setdefault("npm_config_fund", "false")
    return env


def _should_detect_interactive_prompts(command: str) -> bool:
    """Limit prompt heuristics to package managers that commonly ask for confirmation."""
    return bool(_INTERACTIVE_PROMPT_COMMAND_RE.search(command or ""))


def _detect_interactive_prompt(text: str) -> str | None:
    sample = str(text or "")
    if not sample:
        return None
    prompt_markers = (
        "ok to proceed? (y)",
        "do you want to continue? [y/n]",
        "do you want to continue? [yes/no]",
        "[y/n]",
        "[yes/no]",
        "[n/y]",
        "[no/yes]",
    )
    lowered = sample.lower()
    for marker in prompt_markers:
        if marker in lowered:
            return marker
    return None


def classify_cli_command(command: str) -> dict[str, Any]:
    # Shared classification policy for both routing and execution layers.
    return classify_shell_command(command)


def _powershell_executable() -> str:
    return shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"


def _windows_powershell_command(command: str) -> str:
    """Run PowerShell with UTF-8 input/output for reliable pipe transport."""
    return (
        "$utf8 = [System.Text.UTF8Encoding]::new($false); "
        "$OutputEncoding = $utf8; "
        "if ($null -ne $PSStyle) { $PSStyle.OutputRendering = 'PlainText' }; "
        "[Console]::InputEncoding = $utf8; "
        "[Console]::OutputEncoding = $utf8; "
        f"& {{ {command} }}"
    )


def _windows_output_encodings() -> tuple[str, ...]:
    """Return legacy Windows encodings used by commands that ignore UTF-8 settings."""
    if os.name != "nt":
        return ("utf-8",)
    candidates = ["utf-8"]
    try:
        # Python UTF-8 mode makes getpreferredencoding(False) return utf-8;
        # getencoding() still exposes the Windows ANSI code page.
        candidates.append(locale.getencoding())
    except (AttributeError, LookupError):
        pass
    try:
        candidates.append(locale.getpreferredencoding(False))
    except Exception:
        pass
    candidates.extend(("cp866", "cp1251"))
    return tuple(dict.fromkeys(name.lower() for name in candidates if name))


class _CliOutputDecoder:
    """Decode UTF-8 streams incrementally, with a Windows legacy-codepage fallback."""

    def __init__(self) -> None:
        self._encodings = _windows_output_encodings()
        self._encoding = self._encodings[0]
        self._pending = bytearray()

    def decode(self, chunk: bytes, *, final: bool = False) -> str:
        if not chunk and not final:
            return ""
        self._pending.extend(chunk)
        raw = bytes(self._pending)
        try:
            text = raw.decode(self._encoding, errors="strict")
        except UnicodeDecodeError as exc:
            # Keep an incomplete UTF-8 character for the next pipe chunk.
            if self._encoding == "utf-8" and not final and exc.end == len(raw):
                return ""
            prefix = raw[:exc.start].decode(self._encoding, errors="strict")
            suffix = raw[exc.start:]
            self._encoding = next(
                (encoding for encoding in self._encodings[1:] if _can_decode(suffix, encoding)),
                "utf-8",
            )
            text = prefix + suffix.decode(self._encoding, errors="replace")
        self._pending.clear()
        return text


def _can_decode(data: bytes, encoding: str) -> bool:
    try:
        data.decode(encoding, errors="strict")
    except (LookupError, UnicodeDecodeError):
        return False
    return True


def _windows_subprocess_kwargs() -> dict[str, Any]:
    if os.name != "nt":
        return {}
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if not create_no_window:
        return {}
    return {"creationflags": create_no_window}


async def _terminate_process_tree(process: Any) -> None:
    pid = getattr(process, "pid", None)
    if os.name == "nt" and pid:
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                **_windows_subprocess_kwargs(),
            )
            await asyncio.wait_for(killer.wait(), timeout=3)
        except Exception:
            try:
                process.kill()
            except OSError:
                pass
    else:
        try:
            process.kill()
        except OSError:
            pass

    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except Exception:
        return


def set_safety_policy(policy: SafetyPolicy):
    """Sets the global safety policy for shell execution."""
    global _SAFETY_POLICY
    _SAFETY_POLICY = policy


def _limit_raw_output(content: str, capture: Optional[ShellOutputCapture] = None) -> str:
    limit = _SAFETY_POLICY.max_raw_tool_output if _SAFETY_POLICY else 100000
    limited = truncate_output(content, limit, source="shell-raw")
    if capture is None or not capture.persisted_path:
        return limited
    # Only success/neutral results may be wrapped: error results must keep the
    # leading ERROR[TYPE]: marker that core/tool_results.py parses.
    return build_persisted_output_envelope(
        body=limited,
        original_chars=capture.total_chars,
        persisted_path=capture.persisted_path,
        artifact_truncated=capture.artifact_truncated,
    )


def _append_persisted_pointer(content: str, capture: ShellOutputCapture) -> str:
    if not capture.persisted_path:
        return content
    pointer = format_persisted_output_pointer(
        original_chars=capture.total_chars,
        persisted_path=capture.persisted_path,
        artifact_truncated=capture.artifact_truncated,
    )
    return f"{content}\n\n{pointer}"


def _max_timeout_seconds() -> int:
    raw = (os.environ.get(MAX_TIMEOUT_ENV) or "").strip()
    try:
        configured_ms = int(raw, 10) if raw else 0
    except ValueError:
        configured_ms = 0
    limit_ms = configured_ms if configured_ms > 0 else BASH_TOOL_MAX_TIMEOUT_MS
    return max(1, limit_ms // 1000)


def _resolve_timeout(requested: int) -> tuple[int, bool]:
    """Clamp one tool-call timeout to the configured policy.

    Returns the effective timeout in seconds and whether it was clamped.
    """

    limit = _max_timeout_seconds()
    value = max(1, int(requested))
    if value > limit:
        return limit, True
    return value, False


def set_working_directory(cwd: str):
    """
    Syncs the shell's working directory with the FilesystemManager's workspace.
    Call this when initializing the agent to ensure tools look at the same folders.
    """
    global _WORKING_DIRECTORY
    _WORKING_DIRECTORY = cwd


def set_cli_output_emitter(emitter: Optional[Callable[[dict[str, str]], None]]) -> None:
    """Registers callback that receives streaming CLI chunks for UI rendering."""
    global _CLI_OUTPUT_EMITTER
    _CLI_OUTPUT_EMITTER = emitter


@contextmanager
def cli_output_context(tool_id: str) -> Iterator[None]:
    """Binds current tool call id so streaming chunks can be routed to the right widget."""
    token = _CLI_TOOL_ID.set(str(tool_id or "").strip())
    try:
        yield
    finally:
        _CLI_TOOL_ID.reset(token)


def _emit_cli_output(data: str, stream: str) -> None:
    if not data:
        return
    emitter = _CLI_OUTPUT_EMITTER
    tool_id = _CLI_TOOL_ID.get()
    if emitter is None or not tool_id:
        return
    try:
        emitter({"tool_id": tool_id, "data": data, "stream": stream})
    except Exception:
        # Streaming is best-effort and must never fail tool execution.
        return

@tool("cli_exec")
async def cli_exec(
    command: str,
    timeout: Annotated[
        int,
        Field(
            gt=0,
            description=(
                "Maximum execution time in seconds. Values above the "
                f"{MAX_TIMEOUT_ENV} limit (600 seconds by default) are clamped."
            ),
        ),
    ] = DEFAULT_TIMEOUT,
) -> str:
    """Run one non-interactive shell command in the workspace. Stateless: include cd/chains in the same command. Supports pipes, redirects, &&. Use run_background_process for servers/watchers; avoid prompts and interactive TUI commands."""
    if _SAFETY_POLICY and not _SAFETY_POLICY.allow_shell:
        return format_error(ErrorType.ACCESS_DENIED, "Shell execution is disabled by SafetyPolicy.")

    if not command.strip():
        return format_error(ErrorType.VALIDATION, "Command cannot be empty.")

    timeout_seconds, timeout_clamped = _resolve_timeout(timeout)

    normalized_command = _normalize_windows_python_heredoc(command)
    normalized_command = _strip_nested_windows_powershell_wrapper(normalized_command)
    normalized_command = _normalize_windows_null_device(normalized_command)
    command_profile = classify_cli_command(normalized_command)
    if command_profile["long_running_service"]:
        return format_error(
            ErrorType.VALIDATION,
            "Foreground service/server commands are not supported in cli_exec. Use run_background_process instead.",
        )
    command_env = _prepare_shell_env(normalized_command)
    detect_interactive_prompts = _should_detect_interactive_prompts(normalized_command)

    try:
        if os.name == "nt":
            process = await asyncio.create_subprocess_exec(
                _powershell_executable(),
                "-NoProfile",
                "-Command",
                _windows_powershell_command(normalized_command),

                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=_WORKING_DIRECTORY,
                env=command_env,
                **_windows_subprocess_kwargs(),
            )
        else:
            process = await asyncio.create_subprocess_shell(
                normalized_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=_WORKING_DIRECTORY,
                env=command_env,
            )
        # Bound memory per stream while keeping head+tail diagnostics.
        # The capture limit sits at max_raw_tool_output so _limit_raw_output
        # stays the single canonical truncation point of the inline text; any
        # output beyond it is mirrored into a workspace artifact instead of
        # being lost.
        raw_limit = _SAFETY_POLICY.max_raw_tool_output if _SAFETY_POLICY else 100000
        capture = ShellOutputCapture(
            inline_limit=raw_limit,
            artifact_directory=resolve_artifact_directory(_WORKING_DIRECTORY),
            max_persisted_chars=resolve_max_persisted_chars(),
        )

        chunk_queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()
        interactive_prompt: str = ""
        interactive_prompt_sample: str = ""

        async def _read_stream(stream_name: str, reader: asyncio.StreamReader | None) -> None:
            if reader is None:
                await chunk_queue.put((stream_name, None))
                return
            # Decode UTF-8 incrementally; on Windows, fall back to the
            # system code page for older commands that ignore UTF-8 settings.
            decoder = _CliOutputDecoder()
            try:
                while True:
                    chunk = await reader.read(8192)
                    if not chunk:
                        break
                    text = decoder.decode(chunk)
                    if not text:
                        continue
                    await chunk_queue.put((stream_name, text))
                tail = decoder.decode(b"", final=True)

                if tail:
                    await chunk_queue.put((stream_name, tail))
            finally:
                await chunk_queue.put((stream_name, None))

        async def _collect_stream_output() -> None:
            nonlocal interactive_prompt, interactive_prompt_sample
            completed_readers = 0
            prompt_window = ""
            while completed_readers < 2:
                stream_name, chunk = await chunk_queue.get()
                if chunk is None:
                    completed_readers += 1
                    continue
                if stream_name == "stdout":
                    capture.append("stdout", chunk)
                else:
                    capture.append("stderr", chunk)
                _emit_cli_output(chunk, stream_name)

                if detect_interactive_prompts and not interactive_prompt:
                    prompt_window = (prompt_window + chunk)[-1200:]
                    detected_prompt = _detect_interactive_prompt(prompt_window)
                    if detected_prompt:
                        interactive_prompt = detected_prompt
                        interactive_prompt_sample = prompt_window[-320:].strip()
                        await _terminate_process_tree(process)

        stdout_reader_task = asyncio.create_task(_read_stream("stdout", process.stdout))
        stderr_reader_task = asyncio.create_task(_read_stream("stderr", process.stderr))
        collector_task = asyncio.create_task(_collect_stream_output())

        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            timed_out = True
            await _terminate_process_tree(process)
        except asyncio.CancelledError:
            await _terminate_process_tree(process)
            raise
        finally:
            await asyncio.gather(stdout_reader_task, stderr_reader_task, return_exceptions=True)
            await collector_task

        capture.close()
        stdout = capture.stdout.strip()
        stderr = capture.stderr.strip()

        output_parts =[]
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"[stderr]\n{stderr}")

        output = "\n".join(output_parts)

        if interactive_prompt:
            details = f"\nOutput tail:\n{interactive_prompt_sample}" if interactive_prompt_sample else ""
            return _append_persisted_pointer(
                format_error(
                    ErrorType.EXECUTION,
                    "Interactive prompt detected in cli_exec output "
                    f"('{interactive_prompt}'). Run command in non-interactive mode "
                    "(for npm/npx add -y/--yes)."
                    f"{details}",
                ),
                capture,
            )

        if timed_out:
            details = f"\nPartial output:\n{output}" if output else ""
            timeout_reason = f"Command timed out after {timeout_seconds} seconds."
            if timeout_clamped:
                timeout_reason += (
                    f" The requested timeout ({timeout} seconds) was clamped by"
                    f" {MAX_TIMEOUT_ENV} to {timeout_seconds} seconds."
                )
            return _append_persisted_pointer(
                _limit_raw_output(
                    format_error(
                        ErrorType.TIMEOUT,
                        f"{timeout_reason} Did you run an interactive command (like nano/vim) or a blocking server?{details}",
                    )
                ),
                capture,
            )

        if process.returncode != 0:
            # Some commands use a non-zero exit code as part of their normal
            # protocol (grep/rg → 1 = no matches, vulture → 1 = dead code found,
            # pytest → 1 = test failures, diff → 1 = files differ).  For these,
            # the output is the result — do NOT mark it as an error.
            interpretation = interpret_non_error_exit(normalized_command, process.returncode)
            if interpretation:
                neutral_parts = [f"Exit Code: {process.returncode} ({interpretation})"]
                if output:
                    neutral_parts.append(output)
                result = "\n".join(neutral_parts)
                return _limit_raw_output(result, capture)

            error_msg = f"Command failed with Exit Code {process.returncode}."
            cmd_hint = _get_windows_command_hint(command, stderr)
            if os.name == "nt" and "<< was unexpected at this time." in stderr:
                cmd_hint += (
                    "\nHint (Windows): bash-style heredoc (`python - <<'PY'`) is not supported through cmd.exe. "
                    "Use a PowerShell here-string: @' ... '@ | python -"
                )
            if output:
                error_msg += f"\nOutput:\n{output}"
            else:
                error_msg += " (No output)"
            if cmd_hint:
                error_msg += cmd_hint
            return _append_persisted_pointer(
                _limit_raw_output(format_error(ErrorType.EXECUTION, error_msg)),
                capture,
            )

        if not output:
            output = "Command executed successfully (no output)."

        return _limit_raw_output(output, capture)

    except Exception as e:
        return format_error(ErrorType.EXECUTION, f"Error executing command: {str(e)}")
