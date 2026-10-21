from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

from core.errors import ErrorType, format_error
from core.tool_args import canonicalize_tool_args

_FILE_ARG_NAMES = ("path", "file_path", "dir_path")


class ValidationContext(TypedDict, total=False):
    tool_name: str
    args: Dict[str, Any]


def _extract_path(args: Dict[str, Any]) -> Optional[str]:
    for name in _FILE_ARG_NAMES:
        value = args.get(name)
        if value:
            return str(value)
    return None


def _resolve_workspace_path(path: str) -> Path:
    try:
        from tools.filesystem import resolve_workspace_path

        return resolve_workspace_path(path)
    except Exception:
        return Path(path).resolve()


def validate(result: str, context: ValidationContext) -> Optional[str]:
    """Validates the side effects of tool execution."""
    tool_name = str(context.get("tool_name") or "").strip()
    args = canonicalize_tool_args(context.get("args"))
    if result.startswith("ERROR"):
        return None

    try:
        if tool_name in ("safe_delete_file", "safe_delete_directory"):
            path = _extract_path(args)
            if path and _resolve_workspace_path(path).exists():
                target_type = "File" if "file" in tool_name else "Directory"
                return format_error(ErrorType.VALIDATION, f"{target_type} {path} still exists after deletion.")

    except Exception as e:
        return format_error(ErrorType.VALIDATION, f"Validation check failed: {e}")

    return None


def validate_tool_result(tool_name: str, args: Dict[str, Any], result: str) -> Optional[str]:
    return validate(result, {"tool_name": tool_name, "args": args})
