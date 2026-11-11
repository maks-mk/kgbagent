from __future__ import annotations

from typing import Any, Callable, Dict, List

from core.fast_copy import copy_jsonish
from core.message_utils import compact_text
from core.state import OpenToolIssue


ToolIssue = OpenToolIssue
WorkspaceBoundaryChecker = Callable[[str, Dict[str, Any]], bool]


def build_tool_issue(
    *,
    current_turn_id: int,
    kind: str,
    summary: str,
    tool_names: List[str],
    tool_args: Dict[str, Any] | None,
    source: str,
    error_type: str = "",
    fingerprint: str = "",
    details: Dict[str, Any] | None = None,
    progress_fingerprint: str = "",
) -> ToolIssue:
    return {
        "turn_id": current_turn_id,
        "kind": str(kind or "tool_error").strip() or "tool_error",
        "summary": compact_text(str(summary or "").strip(), 320),
        "tool_names": [str(name).strip() for name in (tool_names or []) if str(name).strip()],
        "tool_args": copy_jsonish(tool_args) if isinstance(tool_args, dict) else {},
        "source": str(source or "tools").strip() or "tools",
        "error_type": str(error_type or "").strip().upper(),
        "fingerprint": str(fingerprint or "").strip(),
        "progress_fingerprint": str(progress_fingerprint or fingerprint or "").strip(),
        "details": copy_jsonish(details) if isinstance(details, dict) else {},
    }


def enrich_tool_issue_details(
    tool_name: str,
    tool_args: Dict[str, Any],
    parsed_result: Any,
    *,
    issue_details: Dict[str, Any] | None = None,
    workspace_boundary_violated: WorkspaceBoundaryChecker | None = None,
) -> Dict[str, Any]:
    details = copy_jsonish(issue_details) if isinstance(issue_details, dict) else {}
    missing_fields = [
        str(field).strip()
        for field in (details.get("missing_required_fields") or [])
        if str(field).strip()
    ]
    if missing_fields:
        details["missing_required_fields"] = missing_fields
    if str(getattr(parsed_result, "error_type", "") or "").strip().upper() == "LOOP_DETECTED":
        details["loop_detected"] = True
    details["safety_violation"] = bool(
        details.get("safety_violation")
        or (
            callable(workspace_boundary_violated)
            and workspace_boundary_violated(tool_name, tool_args if isinstance(tool_args, dict) else {})
        )
    )
    return details


def merge_tool_issues(
    issues: List[ToolIssue],
    *,
    current_turn_id: int,
) -> ToolIssue | None:
    if not issues:
        return None

    summaries: List[str] = []
    tool_names: List[str] = []
    tool_args: Dict[str, Any] = {}
    kind = "tool_error"
    source = "tools"
    error_type = ""
    fingerprint = ""
    progress_fingerprint = ""
    merged_details: Dict[str, Any] = {}

    for issue in issues:
        summary = str(issue.get("summary") or "").strip()
        if summary and summary not in summaries:
            summaries.append(summary)

        for tool_name in issue.get("tool_names") or []:
            normalized_name = str(tool_name).strip()
            if normalized_name and normalized_name not in tool_names:
                tool_names.append(normalized_name)

        if not tool_args and isinstance(issue.get("tool_args"), dict):
            tool_args = copy_jsonish(issue.get("tool_args"))

        issue_error_type = str(issue.get("error_type") or "").strip().upper()
        if issue_error_type and not error_type:
            error_type = issue_error_type

        issue_kind = str(issue.get("kind") or "").strip().lower()
        if issue_kind == "approval_denied":
            kind = "approval_denied"
            source = "approval"

        if not fingerprint:
            fingerprint = str(issue.get("fingerprint") or "").strip()
        if not progress_fingerprint:
            progress_fingerprint = str(
                issue.get("progress_fingerprint") or issue.get("fingerprint") or ""
            ).strip()

        details = issue.get("details") if isinstance(issue.get("details"), dict) else {}
        merged_details.update(copy_jsonish(details))

    return build_tool_issue(
        current_turn_id=current_turn_id,
        kind=kind,
        summary=" | ".join(summaries),
        tool_names=tool_names,
        tool_args=tool_args,
        source=source,
        error_type=error_type,
        fingerprint=fingerprint,
        progress_fingerprint=progress_fingerprint,
        details=merged_details,
    )


def active_open_tool_issue(state: dict[str, Any], current_turn_id: int) -> ToolIssue | None:
    issue = state.get("open_tool_issue")
    if not isinstance(issue, dict):
        return None

    issue_turn_id = int(issue.get("turn_id", 0) or 0)
    summary = str(issue.get("summary", "")).strip()
    if issue_turn_id != current_turn_id or not summary:
        return None

    tool_names = [str(name) for name in (issue.get("tool_names") or []) if str(name).strip()]
    return {
        "turn_id": issue_turn_id,
        "kind": str(issue.get("kind") or "tool_error"),
        "summary": summary,
        "tool_names": tool_names,
        "tool_args": dict(issue.get("tool_args") or {}),
        "source": str(issue.get("source") or "tools"),
        "error_type": str(issue.get("error_type") or ""),
        "fingerprint": str(issue.get("fingerprint") or ""),
        "progress_fingerprint": str(issue.get("progress_fingerprint") or issue.get("fingerprint") or ""),
        "details": dict(issue.get("details") or {}),
    }
