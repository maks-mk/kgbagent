import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import base64
import locale
import os
import re
import shutil
import subprocess
import threading
import logging
import time
from typing import Annotated, Any, Callable, Iterator, Optional
from langchain_core.tools import tool
from pydantic import Field

from core.utils import truncate_output
from core.errors import format_error, ErrorType
from core.safety_policy import SafetyPolicy
from core.policy_engine import classify_shell_command
from tools import background_processes
from tools.shell_output import (
    PersistedOutputWriter,
    ShellOutputCapture,
    build_persisted_output_envelope,
    format_persisted_output_pointer,
    new_background_log_path,
    prune_background_logs,
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

# A command may exit while a process it spawned keeps the inherited stdout/stderr
# handles open; the pipe readers then never see EOF. Bound that wait so one tool
# call can never block the agent forever.
STREAM_DRAIN_TIMEOUT = 2.0
_STREAMS_ABANDONED_NOTE = (
    "[note] Output capture stopped because another process kept the command's "
    "stdout/stderr open (a background process was likely started without "
    "background=true). Start long-running commands with background=true to get a "
    "PID and a log file instead of a hanging call."
)

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

    # The agent itself typically runs inside its own venv; a child shell that
    # inherits VIRTUAL_ENV/PATH silently resolves bare `python`/`pip` to the
    # agent's venv instead of the user's project venv. When the workspace has
    # its own venv, prefer its interpreter by prepending its bin/Scripts dir to
    # PATH and repointing VIRTUAL_ENV (and the Windows-style PYLAUNCHER envs).
    workspace_venv = detect_workspace_venv(_WORKING_DIRECTORY)
    agent_venv = os.environ.get("VIRTUAL_ENV") or ""
    if workspace_venv and not _same_venv(workspace_venv, agent_venv):
        bin_dir = venv_scripts_dir(workspace_venv)
        if bin_dir:
            path_value = env.get("PATH", "")
            env["PATH"] = f"{bin_dir}{os.pathsep}{path_value}" if path_value else bin_dir
        env["VIRTUAL_ENV"] = str(workspace_venv)
        if os.name == "nt":
            for launcher_env in ("PYLAUNCHER_ALLOW_INSTALL", "PYLAUNCHER_ALWAYS_INSTALL"):
                env.pop(launcher_env, None)

    if _NPM_LIKE_COMMAND_RE.search(command or ""):
        # Prevent npm/npx confirmation prompts such as "Ok to proceed? (y)".
        env.setdefault("npm_config_yes", "true")
        env.setdefault("npm_config_audit", "false")
        env.setdefault("npm_config_fund", "false")
    return env


def detect_workspace_venv(workspace: str) -> Path | None:
    """Returns the project venv directory (``venv``/``.venv``) of ``workspace``.

    Detection requires an actual interpreter inside the directory so stray
    folders or venvs of a parent project are not picked up accidentally.
    """
    candidates = ("venv", ".venv")
    try:
        for name in candidates:
            candidate = Path(workspace) / name
            if venv_scripts_dir(candidate) is not None:
                return candidate
    except (OSError, ValueError):
        # Unreadable workspace paths must not break shell execution.
        return None
    return None


def venv_scripts_dir(venv_dir: Path) -> Path | None:
    """Returns the bin/Scripts directory of a venv if it holds a Python interpreter."""
    scripts = "Scripts" if os.name == "nt" else "bin"
    interpreter = "python.exe" if os.name == "nt" else "python"
    try:
        candidate = venv_dir / scripts
        if (candidate / interpreter).is_file():
            return candidate
    except (OSError, ValueError):
        return None
    return None


def _same_venv(left: Path, right: str) -> bool:
    """Case-insensitive (Windows) comparison of two venv paths, best-effort."""
    try:
        return str(Path(left).resolve()).lower() == str(Path(right).resolve()).lower()
    except (OSError, ValueError):
        return False


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
            # Keep only an incomplete UTF-8 suffix, not the decoded prefix.
            if (
                self._encoding == "utf-8" and not final
                and exc.reason == "unexpected end of data"
            ):
                self._pending[:] = raw[exc.start:]
                return raw[:exc.start].decode("utf-8", errors="strict")
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


def _background_process_kwargs() -> dict[str, Any]:
    """Start a child that keeps running after the tool call returns."""

    if os.name != "nt":
        return {"start_new_session": True}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    return {"creationflags": flags} if flags else {}


def _workspace_relative_path(path: Path) -> str:
    """Workspace-relative path when possible, so read_file can open it directly."""

    try:
        return str(path.relative_to(Path(_WORKING_DIRECTORY).resolve()))
    except (OSError, ValueError):
        return str(path)


def _capture_background_output(process: subprocess.Popen, log_file, writer: PersistedOutputWriter) -> None:
    """Drain the pipe even after truncation or a disk error, until all writers exit."""

    decoder = _CliOutputDecoder()
    writing = True
    try:
        while True:
            chunk = process.stdout.read1(65536)
            text = decoder.decode(chunk, final=not chunk) if not writer.truncated and writing else ""
            if writing:
                try:
                    writer.append(text.replace("\x00", "\ufffd"))
                    log_file.flush()
                except OSError:
                    writing = False
                    logging.getLogger(__name__).warning("Cannot write background log for PID %s.", process.pid)
            if not chunk:
                break
    finally:
        for handle in (process.stdout, log_file):
            try:
                handle.close()
            except OSError:
                logging.getLogger(__name__).warning("Cannot close background output for PID %s.", process.pid)


def _background_stop_hint(pid: int) -> str:
    command = f"taskkill /PID {pid} /T /F" if os.name == "nt" else f"kill -TERM -{pid}"
    return f'cli_exec(command="{command}")'


def _start_background_command(
    command: str, command_env: dict[str, str], *, started_processes: list[subprocess.Popen] | None = None
) -> str:
    """Start ``command`` detached; return at once with its PID and log path."""

    limit = _SAFETY_POLICY.max_background_processes if _SAFETY_POLICY else 5
    process = None
    log_file = None
    try:
        with background_processes.reserve_slot(limit) as register:
            artifact_directory = resolve_artifact_directory(_WORKING_DIRECTORY)
            artifact_directory.mkdir(parents=True, exist_ok=True)
            prune_background_logs(artifact_directory)
            log_path = new_background_log_path(artifact_directory)
            log_file = log_path.open("w", encoding="utf-8", errors="replace", newline="")
            writer = PersistedOutputWriter(log_file, resolve_max_persisted_chars())
            header = f"# cli_exec background ({time.strftime('%Y-%m-%d %H:%M:%S')}): {command}\n"
            writer.append(header.replace("\x00", "\ufffd"))
            log_file.flush()
            if os.name == "nt":
                argv = [
                    _powershell_executable(), "-NoProfile", "-NonInteractive", "-Command",
                    _windows_powershell_command(command),
                ]
            else:
                argv = ["/bin/sh", "-c", command]
            process = subprocess.Popen(
                argv,
                cwd=_WORKING_DIRECTORY,
                env=command_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                **_background_process_kwargs(),
            )
            pid = register(process)
            if started_processes is not None:
                started_processes.append(process)
            reader = threading.Thread(
                target=_capture_background_output, args=(process, log_file, writer),
                name=f"cli-exec-output-{pid}", daemon=True,
            )
            reader.start()
            process._cli_output_reader = reader
            log_file = None  # The reader owns the file from here on.
    except background_processes.BackgroundProcessLimitError as exc:
        return format_error(ErrorType.LIMIT_EXCEEDED, f"{exc} Stop a running background process with cli_exec before starting another.")
    except Exception as exc:
        stop_hint = ""
        if process is not None:
            try:
                background_processes.terminate(process.pid)
            except (OSError, subprocess.SubprocessError):
                logging.getLogger(__name__).warning("Cannot stop failed background startup for PID %s.", process.pid)
            finally:
                if process.stdout is not None:
                    process.stdout.close()
            if process.poll() is None:
                stop_hint = f" Process {process.pid} is still running and registered; stop it with {_background_stop_hint(process.pid)}."
        return format_error(ErrorType.EXECUTION, f"Error starting the background command: {exc}{stop_hint}")
    finally:
        if log_file is not None:
            try:
                log_file.close()
            except OSError:
                logging.getLogger(__name__).warning("Cannot close the log after failed background startup.")

    return (
        f"Success: Started background command with PID {pid}.\n"
        f"Log: {_workspace_relative_path(log_path)}\n"
        "The command keeps running after this call returns; do not start it again. "
        "Read the log to follow its output. "
        f"Stop it with {_background_stop_hint(pid)}."
    )


async def _start_background_async(command: str, command_env: dict[str, str]) -> str:
    """Keep startup off the event loop and stop any child created during cancellation."""

    started_processes: list[subprocess.Popen] = []

    def stop_started_processes() -> None:
        for process in started_processes:
            try:
                stopped = background_processes.terminate(process.pid)
            except (OSError, subprocess.SubprocessError):
                stopped = False
            if not stopped and background_processes.get(process.pid) is not None:
                logging.getLogger(__name__).warning(
                    "Cancelled startup left PID %s running and registered.", process.pid
                )

    starter = asyncio.create_task(asyncio.to_thread(
        _start_background_command, command, command_env, started_processes=started_processes
    ))
    try:
        return await asyncio.shield(starter)
    except asyncio.CancelledError:
        # to_thread cannot interrupt Popen: wait for registration before cleanup.
        while not starter.done():
            try:
                await asyncio.shield(starter)
            except asyncio.CancelledError:
                continue
        starter.result()
        cleanup = asyncio.create_task(asyncio.to_thread(stop_started_processes))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        cleanup.result()
        raise


async def _drain_output_streams(
    reader_tasks: list[asyncio.Task],
    collector_task: asyncio.Task,
    *,
    timeout: float | None = None,
) -> bool:
    """Finish pipe reading without ever blocking the tool call forever.

    The readers stop on EOF, but a process spawned by the command can inherit
    stdout/stderr and keep them open after the command itself exited. After
    ``timeout`` the readers are cancelled; they emit their sentinels on
    cancellation, so the collector finishes as well.

    Returns True when a reader had to be cancelled.
    """

    wait_timeout = STREAM_DRAIN_TIMEOUT if timeout is None else timeout
    abandoned = False
    try:
        _, pending = await asyncio.wait(reader_tasks, timeout=wait_timeout)
        if pending:
            abandoned = True
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        _, pending_collector = await asyncio.wait([collector_task], timeout=wait_timeout)
        if pending_collector:
            collector_task.cancel()
            await asyncio.gather(collector_task, return_exceptions=True)
        elif not collector_task.cancelled():
            # Surface collector failures exactly like the previous direct await.
            collector_task.result()
    except asyncio.CancelledError:
        for task in (*reader_tasks, collector_task):
            task.cancel()
        await asyncio.gather(*reader_tasks, collector_task, return_exceptions=True)
        raise
    return abandoned


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
    background: Annotated[
        bool,
        Field(
            description=(
                "Start the command detached and return immediately with its PID and a log "
                "file. Use for servers, watchers and any command that never exits."
            ),
        ),
    ] = False,
) -> str:
    """Run one non-interactive shell command in the workspace; stateless, chain cd in the same call. Foreground waits for exit; servers/watchers need background=true (returns PID and log). Never `&`/Start-Process. Avoid interactive TUI. Windows runs PowerShell; bare python/pip use venv/.venv."""
    if _SAFETY_POLICY and not _SAFETY_POLICY.allow_shell:
        return format_error(ErrorType.ACCESS_DENIED, "Shell execution is disabled by SafetyPolicy.")

    if not command.strip():
        return format_error(ErrorType.VALIDATION, "Command cannot be empty.")

    timeout_seconds, timeout_clamped = _resolve_timeout(timeout)

    normalized_command = _normalize_windows_python_heredoc(command)
    normalized_command = _strip_nested_windows_powershell_wrapper(normalized_command)
    normalized_command = _normalize_windows_null_device(normalized_command)
    if background:
        # Background mode exists exactly for the commands rejected right below.
        return await _start_background_async(
            normalized_command, _prepare_shell_env(normalized_command)
        )
    command_profile = classify_cli_command(normalized_command)
    if command_profile["long_running_service"]:
        return format_error(
            ErrorType.VALIDATION,
            "Foreground service/server commands are not supported in cli_exec because they "
            "never exit. Re-run the same command with background=true: it starts detached and "
            "returns the PID and a log path immediately.",
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
            streams_abandoned = await _drain_output_streams(
                [stdout_reader_task, stderr_reader_task], collector_task
            )
            capture.close()

        stdout = capture.stdout.strip()
        stderr = capture.stderr.strip()

        output_parts = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"[stderr]\n{stderr}")

        output = "\n".join(output_parts)
        if streams_abandoned:
            output = f"{output}\n{_STREAMS_ABANDONED_NOTE}".strip()

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
            timeout_reason = f"Command timed out after {timeout_seconds} seconds and was terminated."
            if timeout_clamped:
                timeout_reason += (
                    f" The requested timeout ({timeout} seconds) was clamped by"
                    f" {MAX_TIMEOUT_ENV} to {timeout_seconds} seconds."
                )
            return _append_persisted_pointer(
                _limit_raw_output(
                    format_error(
                        ErrorType.TIMEOUT,
                        f"{timeout_reason} Did you run an interactive command (like nano/vim) or a blocking server? "
                        "For servers, watchers and other long-running commands use background=true: "
                        "cli_exec starts them detached and returns the PID and a log path immediately."
                        f"{details}",
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
