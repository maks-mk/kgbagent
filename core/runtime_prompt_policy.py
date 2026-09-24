from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import platform
from typing import Iterable, List, Sequence

from langchain_core.messages import SystemMessage

from core.config import AgentConfig
from core.message_utils import compact_text


@dataclass(frozen=True)
class RuntimePromptContext:
    current_task: str
    tools_available: bool
    active_tool_names: Sequence[str]
    user_choice_locked: bool = False


@dataclass(frozen=True)
class RuntimeExecutionEnvironment:
    os_family: str
    shell_family: str
    path_style: str
    workspace_root: str
    current_working_directory: str
    timezone_name: str
    utc_offset: str


class RuntimePromptPolicyBuilder:
    REQUEST_USER_INPUT_TOOL_NAME = "request_user_input"
    REQUEST_USER_INPUT_POLICY_TEXT = (
        "REQUEST_USER_INPUT POLICY:\n"
        "Use request_user_input only when the next step is genuinely blocked by one concrete user decision "
        "or one missing external value that cannot be recovered from repository state, current messages, or tools.\n"
        "Do not use request_user_input for approvals of risky actions; the approval flow handles that separately.\n"
        "Do not use request_user_input for open-ended brainstorming, optional style preferences, or questions "
        "you can answer yourself from context.\n"
        "Ask exactly one short question.\n"
        "Provide 2 to 5 short mutually exclusive options. Options must be concise labels, not explanations.\n"
        "If one option is best, set `recommended` to the exact option text.\n"
        "Make the request_user_input tool call by itself. Never batch multiple request_user_input calls.\n"
        "After resume, treat the latest request_user_input ToolMessage content as the user's final answer and continue "
        "without asking again in the same turn."
    )
    TOOL_INTENT_REQUIREMENT_TEXT = (
        "TOOL INTENT REQUIREMENT:\n"
        "Always start each new group of tool calls with a 2 to 4 sentence comment in content describing what you are doing "
        "and why, written in the same language as the user's current request. This applies even to single or trivial actions "
        "such as committing.\n"
        "Do not comment on every individual call. For a series of calls serving the same objective, one comment at the "
        "start of the group is enough; add another only when the work reaches a meaningful new stage or an important "
        "result is discovered.\n"
        "A logical group consists of consecutive tool calls serving one immediate objective, even across multiple "
        "messages. The opening comment MUST be in the SAME assistant message as the group's opening tool_calls.\n"
        "Changing a tool, file or command, retrying, or fetching more results alone does not start a new group.\n"
        "Write a new comment before calls serving a different immediate objective, or before a risky action not covered "
        "by the current comment. For risky actions, briefly state the intended effect.\n"
        "Do not narrate individual tool results or repeat comments unnecessarily. When the task is complete, summarize "
        "verified outcomes, actionable blockers, decisions, and next steps.\n"
        "Keep comments concrete and brief. Avoid generic filler and do not expose internal reasoning."
    )

    def __init__(self, *, config: AgentConfig) -> None:
        self.config = config
        self._execution_environment = self._detect_execution_environment()

    def build_messages(self, context: RuntimePromptContext) -> List[SystemMessage]:
        messages: List[SystemMessage] = [
            SystemMessage(content=self._build_runtime_contract(context)),
        ]

        strict_mode_message = self._build_strict_mode_message()
        if strict_mode_message:
            messages.append(SystemMessage(content=strict_mode_message))

        tool_access_message = self._build_tool_access_message(context)
        if tool_access_message:
            messages.append(SystemMessage(content=tool_access_message))

        request_user_input_message = self._build_request_user_input_policy(context)
        if request_user_input_message:
            messages.append(SystemMessage(content=request_user_input_message))

        if context.user_choice_locked:
            messages.append(
                SystemMessage(
                    content=(
                        "USER CHOICE ALREADY COLLECTED IN THIS TURN.\n"
                        "Do not call request_user_input again.\n"
                        "Use the selected value from the latest request_user_input ToolMessage as the user's final answer and continue."
                    )
                )
            )

        if context.tools_available:
            messages.append(SystemMessage(content=self.TOOL_INTENT_REQUIREMENT_TEXT))
        
        return messages

    def _build_runtime_contract(self, context: RuntimePromptContext) -> str:
        environment = self._execution_environment_for_prompt()
        location_lines = [f"Workspace: {environment.workspace_root}"]
        if environment.current_working_directory != environment.workspace_root:
            location_lines.append(f"Working directory: {environment.current_working_directory}")
        lines = [
            "RUNTIME CONTRACT:",
            "CLI only; no GUI.",
            self._build_execution_environment_line(environment),
            *location_lines,
            f"Local time: {environment.timezone_name} ({environment.utc_offset}); date={datetime.now().strftime('%Y-%m-%d')}.",
        ]
        current_task = compact_text(str(context.current_task or "").strip(), 240)
        if current_task:
            lines.append(f"Current task: {current_task}")
        return "\n".join(lines)

    def _build_execution_environment_line(self, environment: RuntimeExecutionEnvironment | None = None) -> str:
        environment = environment or self._detect_execution_environment()
        return (
            "Execution environment: "
            f"os={environment.os_family}; "
            f"shell={environment.shell_family}; "
            f"paths={environment.path_style}."
        )

    def _build_strict_mode_message(self) -> str:
        if not self.config.strict_mode:
            return ""
        return (
            "STRICT MODE: Be precise. No guessing.\n"
            "State material uncertainty, failed checks, and skipped verification explicitly. "
            "If a fact is not confirmed by repository state, tool output, or the user, say so instead of assuming it."
        )

    def _build_tool_access_message(self, context: RuntimePromptContext) -> str:
        if not context.tools_available:
            return (
                "TOOLS:\n"
                "No tools are available in this runtime. Do not claim tool access."
            )

        names = self._normalized_tool_names(context.active_tool_names)
        if not names:
            return (
                "TOOLS:\n"
                "Tools are available in this runtime. Call a tool only when it serves the current objective; "
                "answer directly from already-known information otherwise. Do not invent unavailable tools."
            )
        if len(names) <= 4:
            return (
                "TOOLS:\n"
                "Available tools: "
                + ", ".join(names)
                + ". Do not invent unavailable tools."
            )
        return (
            "TOOLS:\n"
            "Multiple tools are available in this runtime. Do not invent unavailable tools. "
            "If unsure which tool fits, prefer the read-only inspection tool over a mutating one."
        )
    def _build_request_user_input_policy(self, context: RuntimePromptContext) -> str:
        if self.REQUEST_USER_INPUT_TOOL_NAME not in self._normalized_tool_names(context.active_tool_names):
            return ""
        return self.REQUEST_USER_INPUT_POLICY_TEXT

    def _detect_execution_environment(self) -> RuntimeExecutionEnvironment:
        os_family = self._detect_os_family()
        workspace_root = str(Path.cwd().resolve())
        now = datetime.now().astimezone()
        return RuntimeExecutionEnvironment(
            os_family=os_family,
            shell_family=self._detect_shell_family(os_family=os_family),
            path_style="windows" if os_family == "windows" else "unix" if os_family in {"linux", "mac"} else "unknown",
            workspace_root=workspace_root,
            current_working_directory=workspace_root,
            timezone_name=self._detect_timezone_name(now),
            utc_offset=self._format_utc_offset(now),
        )

    def _execution_environment_for_prompt(self) -> RuntimeExecutionEnvironment:
        current_directory = str(Path.cwd().resolve())
        if current_directory == self._execution_environment.current_working_directory:
            return self._execution_environment
        return RuntimeExecutionEnvironment(
            os_family=self._execution_environment.os_family,
            shell_family=self._execution_environment.shell_family,
            path_style=self._execution_environment.path_style,
            workspace_root=current_directory,
            current_working_directory=current_directory,
            timezone_name=self._execution_environment.timezone_name,
            utc_offset=self._execution_environment.utc_offset,
        )

    @staticmethod
    def _detect_os_family() -> str:
        raw_name = platform.system().strip().casefold()
        if raw_name == "windows":
            return "windows"
        if raw_name == "linux":
            return "linux"
        if raw_name == "darwin":
            return "mac"
        return "unknown"

    @staticmethod
    def _detect_shell_family(*, os_family: str) -> str:
        shell_candidates = [
            os.environ.get("SHELL", ""),
            os.environ.get("COMSPEC", ""),
            os.environ.get("TERM_SHELL", ""),
        ]
        if "PSModulePath" in os.environ:
            shell_candidates.insert(0, "powershell")

        for candidate in shell_candidates:
            normalized = str(candidate or "").replace("\\", "/").casefold().strip()
            if not normalized:
                continue
            if "pwsh" in normalized or "powershell" in normalized:
                return "powershell"
            if normalized.endswith("cmd.exe") or normalized.endswith("/cmd") or normalized == "cmd":
                return "cmd"
            if normalized.endswith("/bash") or normalized == "bash":
                return "bash"
            if normalized.endswith("/zsh") or normalized == "zsh":
                return "zsh"
            if normalized.endswith("/fish") or normalized == "fish":
                return "fish"
            if normalized.endswith("/sh") or normalized == "sh":
                return "sh"

        if os_family in {"linux", "mac"}:
            return "sh"
        return "unknown"

    @staticmethod
    def _detect_timezone_name(now: datetime) -> str:
        zone_key = str(getattr(now.tzinfo, "key", "") or "").strip()
        if zone_key:
            return zone_key
        tz_name = str(now.tzname() or "").strip()
        if tz_name and all(char.isascii() and (char.isalnum() or char in "/_+-:") for char in tz_name):
            return tz_name
        return RuntimePromptPolicyBuilder._format_utc_offset(now)

    @staticmethod
    def _format_utc_offset(now: datetime) -> str:
        offset = now.utcoffset()
        if offset is None:
            return "UTC?"
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        absolute_minutes = abs(total_minutes)
        hours, minutes = divmod(absolute_minutes, 60)
        return f"UTC{sign}{hours:02d}:{minutes:02d}"

    @staticmethod
    def _normalized_tool_names(tool_names: Iterable[str]) -> List[str]:
        return [str(name).strip() for name in tool_names if str(name).strip()]
