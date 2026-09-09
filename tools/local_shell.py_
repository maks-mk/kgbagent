import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import codecs
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

# Constants
DEFAULT_TIMEOUT = 120

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
    r"^\s*(?P<exe>python(?:3(?:\.\d+)?)?)\s*-\s*<<\s*['\"]?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\r?\n(?P<body>[\s\S]*?)\r?\n(?P=tag)\s*$",
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
# (e.g. grep/rg return 1 when no matches found, vulture returns 1 when dead
# code is found, pytest returns 1 on test failures).  For these, a non-zero
# exit code does NOT indicate an execution error — the output itself is the
# result the agent needs.
_EXIT_CODE_NEUTRAL_COMMAND_RE = re.compile(
    r"(?:^|[;&|()\s])(?:vulture|grep|rg|findstr|select-string|diff|pytest)(?=$|[;&|()\s])",
    re.IGNORECASE,
)

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

    exe = match.group("exe")
    body = match.group("body").replace("\r\n", "\n")
    # PowerShell single-quoted here-string terminator must be at line start.
    # Indent-breaking sequences are extremely unlikely for generated scripts; if present,
    # fallback to original command so the model sees a direct error and can correct manually.
    if "\n'@\n" in f"\n{body}\n":
        return command

    return f"@'\n{body}\n'@ | {exe} -"


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


def _limit_raw_output(content: str) -> str:
    limit = _SAFETY_POLICY.max_raw_tool_output if _SAFETY_POLICY else 100000
    return truncate_output(content, limit, source="shell-raw")


class _OutputBuffer:
    """Bound memory while retaining the beginning and the diagnostic tail."""

    def __init__(self, limit: int):
        self.head_limit = max(0, limit // 2)
        self.tail_limit = max(0, limit - self.head_limit)
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

    def getvalue(self) -> str:
        omitted = self.total - len(self.head) - len(self.tail)
        if omitted > 0:
            return self.head + f"\n[TRUNCATED: {omitted} characters omitted]\n" + self.tail
        return self.head + self.tail


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
        Field(gt=0, description="Maximum execution time in seconds."),
    ] = DEFAULT_TIMEOUT,
) -> str:
    """Run one non-interactive shell command in the workspace. Stateless: include cd/chains in the same command. Supports pipes, redirects, &&. Use run_background_process for servers/watchers; avoid prompts and interactive TUI commands."""
    if _SAFETY_POLICY and not _SAFETY_POLICY.allow_shell:
        return format_error(ErrorType.ACCESS_DENIED, "Shell execution is disabled by SafetyPolicy.")

    if not command.strip():
        return format_error(ErrorType.VALIDATION, "Command cannot be empty.")

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
                normalized_command,
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
        # The buffer limit sits above max_raw_tool_output so that _limit_raw_output
        # stays the single canonical truncation point in the normal range; the
        # buffer only kicks in for runaway multi-megabyte output.
        raw_limit = _SAFETY_POLICY.max_raw_tool_output if _SAFETY_POLICY else 100000
        buffer_limit = max(raw_limit * 2, raw_limit + 1024)
        stdout_buffer = _OutputBuffer(buffer_limit)
        stderr_buffer = _OutputBuffer(buffer_limit)
        chunk_queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()
        interactive_prompt: str = ""
        interactive_prompt_sample: str = ""

        async def _read_stream(stream_name: str, reader: asyncio.StreamReader | None) -> None:
            if reader is None:
                await chunk_queue.put((stream_name, None))
                return
            # Incremental decoder: multi-byte UTF-8 characters split across chunk
            # boundaries are reassembled instead of being replaced with U+FFFD.
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
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
                    stdout_buffer.append(chunk)
                else:
                    stderr_buffer.append(chunk)
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
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await _terminate_process_tree(process)
        except asyncio.CancelledError:
            await _terminate_process_tree(process)
            raise
        finally:
            await asyncio.gather(stdout_reader_task, stderr_reader_task, return_exceptions=True)
            await collector_task

        stdout = stdout_buffer.getvalue().strip()
        stderr = stderr_buffer.getvalue().strip()

        output_parts =[]
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"[stderr]\n{stderr}")

        output = "\n".join(output_parts)

        if interactive_prompt:
            details = f"\nOutput tail:\n{interactive_prompt_sample}" if interactive_prompt_sample else ""
            return format_error(
                ErrorType.EXECUTION,
                "Interactive prompt detected in cli_exec output "
                f"('{interactive_prompt}'). Run command in non-interactive mode "
                "(for npm/npx add -y/--yes)."
                f"{details}",
            )

        if timed_out:
            details = f"\nPartial output:\n{output}" if output else ""
            return _limit_raw_output(
                format_error(
                    ErrorType.TIMEOUT,
                    f"Command timed out after {timeout} seconds. Did you run an interactive command (like nano/vim) or a blocking server?{details}",
                )
            )

        
        if process.returncode != 0:
            # Some commands use a non-zero exit code as part of their normal
            # protocol (grep/rg → 1 = no matches, vulture → 1 = dead code found,
            # pytest → 1 = test failures, diff → 1 = files differ).  For these,
            # the output is the result — do NOT mark it as an error.
            if _EXIT_CODE_NEUTRAL_COMMAND_RE.search(normalized_command):
                neutral_parts = [f"Exit Code: {process.returncode}"]
                if output:
                    neutral_parts.append(output)
                result = "\n".join(neutral_parts)
                return _limit_raw_output(result)

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
            return _limit_raw_output(format_error(ErrorType.EXECUTION, error_msg))

        if not output:
            output = "Command executed successfully (no output)."
        
        return _limit_raw_output(output)

    except Exception as e:
        return format_error(ErrorType.EXECUTION, f"Error executing command: {str(e)}")
