from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from core.policy_engine import classify_shell_command


@dataclass(frozen=True)
class RepairPlan:
    strategy: str
    reason: str
    fingerprint: str
    tool_name: str
    suggested_tool_name: str
    original_args: Dict[str, Any]
    patched_args: Dict[str, Any]
    notes: str = ""
    terminal_reason: str = ""
    max_auto_repairs: int = 2
    needs_external_input: bool = False
    safety_violation: bool = False
    progress_fingerprint: str = ""
    llm_guidance: str = ""

    @property
    def retryable(self) -> bool:
        return not self.terminal_reason and self.strategy != "external_block"

_PATH_LIKE_FIELDS = frozenset({"path", "file_path", "dir_path", "source", "destination", "cwd"})
_EDIT_TARGET_FIELDS = frozenset({"old_string", "new_string"})
_CANONICAL_PATH_TOOL_NAMES = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "safe_delete_file",
        "safe_delete_directory",
    }
)


def repair_fingerprint(tool_name: str, tool_args: Dict[str, Any], error_type: str = "") -> str:
    payload = {
        "tool_name": str(tool_name or "").strip().lower(),
        "tool_args": tool_args if isinstance(tool_args, dict) else {},
        "error_type": str(error_type or "").strip().upper(),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _normalize_args_dict(tool_args: Any) -> Dict[str, Any]:
    if not isinstance(tool_args, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for key, value in tool_args.items():
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                normalized[str(key)] = stripped
            continue
        normalized[str(key)] = value
    return normalized


def _split_command(command: str) -> List[str]:
    raw = str(command or "").strip()
    if not raw:
        return []
    try:
        parts = shlex.split(raw, posix=False)
        if parts:
            return [str(part) for part in parts]
    except Exception:
        pass
    return [part for part in re.split(r"\s+", raw) if part]


_NONINTERACTIVE_PATCHERS: Tuple[Tuple[re.Pattern[str], Any], ...] = (
    (re.compile(r"^npx\s+", re.IGNORECASE), lambda raw: "npx --yes " + raw[4:]),
    (re.compile(r"^npm\s+exec\b", re.IGNORECASE), lambda raw: re.sub(r"^npm\s+exec\b", "npm exec --yes", raw, count=1)),
    (re.compile(r"^npm\s+install\b", re.IGNORECASE), lambda raw: raw + " --yes"),
    (re.compile(r"^pnpm\s+(?:add|install)\b", re.IGNORECASE), lambda raw: raw + " --yes"),
    (re.compile(r"^yarn\s+add\b", re.IGNORECASE), lambda raw: raw + " --non-interactive"),
)


def _inject_yes_flag(command: str) -> str:
    raw = str(command or "").strip()
    if not raw:
        return raw
    lowered = raw.lower()
    if "--yes" in lowered or "--non-interactive" in lowered or re.search(r"(^|\s)-y(\s|$)", lowered):
        return raw
    for pattern, patcher in _NONINTERACTIVE_PATCHERS:
        if pattern.search(raw):
            return patcher(raw)
    return raw


def _issue_details(open_tool_issue: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(open_tool_issue, dict):
        return {}
    details = open_tool_issue.get("details") or {}
    return dict(details) if isinstance(details, dict) else {}


def _missing_required_fields(details: Dict[str, Any]) -> Tuple[str, ...]:
    fields = details.get("missing_required_fields") or []
    normalized = sorted({str(field).strip() for field in fields if str(field).strip()})
    return tuple(normalized)


def _missing_required_fields_from_text(text: str) -> Tuple[str, ...]:
    match = re.search(r"missing required field(?:\(s\))?:\s*([A-Za-z0-9_, .-]+)", str(text or ""), re.IGNORECASE)
    if not match:
        return ()
    fields = [
        item.strip(" .`'\"")
        for item in re.split(r"[,/]", match.group(1))
        if item.strip(" .`'\"")
    ]
    return tuple(sorted(set(fields)))


def _choose_context_refresh_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    *,
    missing_fields: Tuple[str, ...],
) -> str:
    if any(field in _PATH_LIKE_FIELDS for field in missing_fields):
        return "list_directory"

    if tool_name == "write_file" and "content" in missing_fields:
        return "write_file"

    if tool_name == "edit_file":
        path_known = bool(str(tool_args.get("path") or "").strip())
        return "read_file" if path_known else "list_directory"

    if tool_name in {"read_file", "write_file"}:
        return "list_directory"

    return tool_name


def _build_recoverable_validation_notes(
    tool_name: str,
    tool_args: Dict[str, Any],
    missing_fields: Tuple[str, ...],
) -> Tuple[str, str]:
    suggested_tool = _choose_context_refresh_tool(
        tool_name,
        tool_args,
        missing_fields=missing_fields,
    )
    if tool_name == "write_file" and "content" in missing_fields and not _has_path_like_gap(missing_fields):
        return (
            suggested_tool,
            "Draft the full file contents from repository and task context, then retry write_file with both "
            "path and complete content in the same call.",
        )
    if tool_name == "edit_file":
        if missing_fields:
            return (
                suggested_tool,
                "Refresh the file context, infer the missing edit target from the repository and task history, "
                "then retry edit_file with a complete exact patch.",
            )
        if all(str(tool_args.get(field) or "").strip() for field in _EDIT_TARGET_FIELDS):
            return (
                suggested_tool,
                "Read the current file contents first, then retry edit_file with an exact existing block instead "
                "of repeating the same patch unchanged.",
            )
    return (
        suggested_tool,
        "The validation error is model-recoverable. Review the latest tool failure, fix the arguments, "
        "or choose a better tool instead of repeating the same call unchanged.",
    )


def _has_path_like_gap(missing_fields: Tuple[str, ...]) -> bool:
    return any(field in _PATH_LIKE_FIELDS for field in missing_fields)


def _repair_plan(
    *,
    strategy: str,
    reason: str,
    fingerprint: str,
    tool_name: str,
    suggested_tool_name: str,
    original_args: Dict[str, Any],
    patched_args: Dict[str, Any],
    notes: str,
    max_auto_repairs: int,
    terminal_reason: str = "",
    needs_external_input: bool = False,
    safety_violation: bool = False,
    llm_guidance: str = "",
) -> RepairPlan:
    return RepairPlan(
        strategy=strategy,
        reason=reason,
        fingerprint=fingerprint,
        tool_name=tool_name,
        suggested_tool_name=suggested_tool_name,
        original_args=original_args,
        patched_args=patched_args,
        notes=notes,
        terminal_reason=terminal_reason,
        max_auto_repairs=max(1, int(max_auto_repairs)),
        needs_external_input=needs_external_input,
        safety_violation=safety_violation,
        progress_fingerprint=fingerprint,
        llm_guidance=llm_guidance,
    )


def normalize_tool_args(tool_name: str, tool_args: Dict[str, Any], current_task: str = "") -> Tuple[Dict[str, Any], List[str]]:
    _ = current_task
    normalized = _normalize_args_dict(tool_args)
    changes: List[str] = []
    name = str(tool_name or "").strip()

    if name in _CANONICAL_PATH_TOOL_NAMES and "path" not in normalized:
        for alias in ("file_path", "filepath", "dir_path", "directory_path"):
            candidate = normalized.get(alias)
            if isinstance(candidate, str) and candidate.strip():
                normalized["path"] = candidate
                changes.append(f"{alias}->path")
                break

    if name in {"find_process_by_port"}:
        port_value = normalized.get("port")
        if isinstance(port_value, str) and port_value.isdigit():
            normalized["port"] = int(port_value)
            changes.append("port:str->int")

    if name in {"stop_background_process"}:
        pid_value = normalized.get("pid")
        if isinstance(pid_value, str) and pid_value.isdigit():
            normalized["pid"] = int(pid_value)
            changes.append("pid:str->int")

    if name == "run_background_process":
        command = normalized.get("command")
        if isinstance(command, str):
            normalized["command"] = _split_command(command)
            changes.append("command:str->list")
        if normalized.get("cwd", "") == "":
            normalized.pop("cwd", None)
            changes.append("cwd:empty_removed")

    if name == "edit_file":
        alias_map = (
            ("old_text", "old_string"),
            ("find_text", "old_string"),
            ("search_text", "old_string"),
            ("new_text", "new_string"),
            ("replacement", "new_string"),
            ("replace_text", "new_string"),
        )
        for old_key, new_key in alias_map:
            if new_key in normalized:
                continue
            candidate = normalized.get(old_key)
            if isinstance(candidate, str) and candidate.strip():
                normalized[new_key] = candidate
                changes.append(f"{old_key}->{new_key}")
        for old_key, _ in alias_map:
            if old_key in normalized:
                normalized.pop(old_key, None)

    if name == "write_file":
        path_value = normalized.get("path")
        if isinstance(path_value, str):
            cleaned = path_value.strip().rstrip(",;")
            if cleaned != path_value:
                normalized["path"] = cleaned
                changes.append("path:trimmed")

    if name == "cli_exec":
        command = normalized.get("command")
        if isinstance(command, str):
            cleaned = command.strip()
            if cleaned != command:
                normalized["command"] = cleaned
                changes.append("command:trimmed")

    return normalized, changes


def build_repair_plan(
    open_tool_issue: Dict[str, Any] | None,
    *,
    current_task: str,
    max_auto_repairs: int = 2,
) -> RepairPlan | None:
    if not isinstance(open_tool_issue, dict):
        return None

    tool_names = [str(name).strip() for name in (open_tool_issue.get("tool_names") or []) if str(name).strip()]
    details = _issue_details(open_tool_issue)
    protocol_reason = str(details.get("protocol_reason") or "").strip().lower()
    if not tool_names:
        fallback_tool_name = str(details.get("suggested_tool_name") or "unknown_tool").strip() or "unknown_tool"
        tool_names = [fallback_tool_name]
    tool_name = tool_names[0]

    summary = str(open_tool_issue.get("summary") or "").strip()
    error_type = str(open_tool_issue.get("error_type") or "").strip().upper()
    original_args = _normalize_args_dict(open_tool_issue.get("tool_args") or {})
    explicit_missing_fields = _missing_required_fields(details)
    inferred_missing_fields = _missing_required_fields_from_text(summary) if not explicit_missing_fields else ()
    missing_fields = explicit_missing_fields or inferred_missing_fields
    patched_args, normalized_changes = normalize_tool_args(tool_name, original_args, current_task=current_task)
    fingerprint = str(open_tool_issue.get("fingerprint") or "").strip() or repair_fingerprint(
        tool_name,
        original_args,
        error_type,
    )
    issue_kind = str(open_tool_issue.get("kind") or "").strip().lower()

    if issue_kind == "approval_denied":
        return _repair_plan(
            strategy="external_block",
            reason="approval_denied",
            fingerprint=fingerprint,
            tool_name=tool_name,
            suggested_tool_name=tool_name,
            original_args=original_args,
            patched_args=original_args,
            notes="The user rejected the protected action. Wait for a different instruction.",
            max_auto_repairs=max_auto_repairs,
            terminal_reason="approval_denied",
            needs_external_input=True,
        )

    if issue_kind == "protocol_error":
        guidance = (
            "Rebuild the next step from repository state and issue a valid structured tool call. "
            "Do not narrate intended actions as plain text."
        )
        notes = "Protocol mismatch detected."
        if protocol_reason == "history_tool_mismatch":
            notes = "Tool-call history is inconsistent. Resume from the last confirmed repository state."
            guidance = (
                "Do not continue from the broken tool chain. Inspect confirmed repository state and issue a fresh valid tool call."
            )
        elif protocol_reason == "tool_not_allowed_for_turn":
            notes = "The model requested a tool outside the active tool set."
            guidance = (
                "Use only currently allowed tools and emit one fresh valid tool call."
            )
        elif protocol_reason == "tool_protocol_error":
            notes = "The model emitted malformed tool-calling payload."
        return _repair_plan(
            strategy="llm_replan",
            reason=protocol_reason or "tool_protocol_error",
            fingerprint=fingerprint,
            tool_name=tool_name,
            suggested_tool_name=tool_name,
            original_args=original_args,
            patched_args=original_args,
            notes=notes,
            max_auto_repairs=max_auto_repairs,
            llm_guidance=guidance,
        )

    if bool(details.get("safety_violation")):
        return _repair_plan(
            strategy="external_block",
            reason="workspace_boundary_violation",
            fingerprint=fingerprint,
            tool_name=tool_name,
            suggested_tool_name=tool_name,
            original_args=original_args,
            patched_args=original_args,
            notes="Mutation targets must stay inside the active workspace.",
            max_auto_repairs=max_auto_repairs,
            terminal_reason="workspace_boundary_violation",
            safety_violation=True,
        )

    if tool_name == "cli_exec" and str(original_args.get("command") or "").strip():
        command = str(original_args.get("command") or "").strip()
        profile = classify_shell_command(command)
        if profile.get("long_running_service"):
            run_bg_args = {"command": _split_command(command)} if command else {}
            return _repair_plan(
                strategy="switch_tool",
                reason="cli_exec_foreground_service",
                fingerprint=fingerprint,
                tool_name=tool_name,
                suggested_tool_name="run_background_process",
                original_args=original_args,
                patched_args=run_bg_args,
                notes="Use run_background_process for long-running services.",
                max_auto_repairs=max_auto_repairs,
                llm_guidance="Switch to run_background_process with argv-style command arguments.",
            )

        command = str(original_args.get("command") or "").strip()
        patched_command = _inject_yes_flag(command)
        if patched_command and patched_command != command:
            fixed_args = dict(original_args)
            fixed_args["command"] = patched_command
            return _repair_plan(
                strategy="resume_after_transient_failure",
                reason="cli_exec_noninteractive_retry",
                fingerprint=fingerprint,
                tool_name=tool_name,
                suggested_tool_name=tool_name,
                original_args=original_args,
                patched_args=fixed_args,
                notes="Retry cli_exec in non-interactive mode.",
                max_auto_repairs=max_auto_repairs,
                llm_guidance="Retry the same command in non-interactive mode and preserve verification after it finishes.",
            )

    if normalized_changes:
        return _repair_plan(
            strategy="normalize_args",
            reason="args_normalization",
            fingerprint=fingerprint,
            tool_name=tool_name,
            suggested_tool_name=tool_name,
            original_args=original_args,
            patched_args=patched_args,
            notes="Apply deterministic argument normalization before retry.",
            max_auto_repairs=max_auto_repairs,
            llm_guidance="Use the normalized arguments instead of the original malformed payload.",
        )

    if error_type == "VALIDATION" and tool_name == "write_file" and "content" in missing_fields and not _has_path_like_gap(missing_fields):
        return _repair_plan(
            strategy="llm_replan",
            reason="validation_missing_write_content",
            fingerprint=fingerprint,
            tool_name=tool_name,
            suggested_tool_name="write_file",
            original_args=original_args,
            patched_args=original_args,
            notes=(
                "The write target is known, but the file body is missing. "
                "Compose the complete file contents first, then retry write_file once with the full payload."
            ),
            max_auto_repairs=max_auto_repairs,
            llm_guidance=(
                "Do not call write_file with path only. Use repository context to draft the full file body, "
                "then emit one valid write_file call containing both path and complete content."
            ),
        )

    if error_type == "VALIDATION" and inferred_missing_fields and _has_path_like_gap(inferred_missing_fields):
        return _repair_plan(
            strategy="external_block",
            reason="validation_missing_path",
            fingerprint=fingerprint,
            tool_name=tool_name,
            suggested_tool_name=tool_name,
            original_args=original_args,
            patched_args=original_args,
            notes="A required path-like field is missing; do not guess mutation targets.",
            max_auto_repairs=max_auto_repairs,
            terminal_reason="validation_missing_path",
            needs_external_input=True,
        )

    if error_type == "VALIDATION" and missing_fields:
        suggested_tool_name, notes = _build_recoverable_validation_notes(
            tool_name,
            original_args,
            missing_fields,
        )
        return _repair_plan(
            strategy="refresh_context",
            reason="validation_missing_fields",
            fingerprint=fingerprint,
            tool_name=tool_name,
            suggested_tool_name=suggested_tool_name,
            original_args=original_args,
            patched_args=original_args,
            notes=notes,
            max_auto_repairs=max_auto_repairs,
            needs_external_input=False,
            llm_guidance="Refresh repository context, infer the missing fields, then continue the same task without asking the user unless inference is impossible.",
        )

    if error_type in {"VALIDATION", "LOOP_DETECTED"}:
        suggested_tool_name, notes = _build_recoverable_validation_notes(
            tool_name,
            original_args,
            missing_fields,
        )
        return _repair_plan(
            strategy="repair_then_rerun",
            reason="validation_recoverable",
            fingerprint=fingerprint,
            tool_name=tool_name,
            suggested_tool_name=suggested_tool_name,
            original_args=original_args,
            patched_args=original_args,
            notes=notes,
            max_auto_repairs=max_auto_repairs,
            llm_guidance="Inspect the latest context, repair the tool call or pick the better tool, then rerun and verify the result.",
        )

    if (
        not bool(details.get("retryable"))
        and bool(details.get("tool_mutating") or details.get("tool_destructive") or details.get("tool_requires_approval"))
        and tool_name not in {"cli_exec", "edit_file", "write_file"}
    ):
        return _repair_plan(
            strategy="llm_replan",
            reason="non_retryable_mutating_tool_error",
            fingerprint=fingerprint,
            tool_name=tool_name,
            suggested_tool_name=tool_name,
            original_args=original_args,
            patched_args=original_args,
            notes=(
                "The mutating tool failed in a way that must not be auto-retried. "
                "Return control to the model so it can explain the failure or adapt the approach."
            ),
            max_auto_repairs=max_auto_repairs,
            llm_guidance=(
                "Do not repeat the failed call unchanged. Inspect the error output, fix the arguments or switch to a "
                "safer verification path, and tell the user what failed and what happens next."
            ),
        )

    if error_type in {"TIMEOUT", "NETWORK"}:
        return _repair_plan(
            strategy="resume_after_transient_failure",
            reason="transient_tool_failure",
            fingerprint=fingerprint,
            tool_name=tool_name,
            suggested_tool_name=tool_name,
            original_args=original_args,
            patched_args=original_args,
            notes="Retry with adjusted arguments or an alternate tool if the same attempt keeps failing.",
            max_auto_repairs=max_auto_repairs,
            llm_guidance="Retry after checking partial output and prefer alternate verification if the same command keeps timing out.",
        )

    return _repair_plan(
        strategy="llm_replan",
        reason="no_safe_auto_repair",
        fingerprint=fingerprint,
        tool_name=tool_name,
        suggested_tool_name=tool_name,
        original_args=original_args,
        patched_args=original_args,
        notes="No deterministic auto-repair available.",
        max_auto_repairs=max_auto_repairs,
        llm_guidance="Replan using repository state, recent tool output, and an alternative tool path instead of repeating the failed step unchanged.",
    )
