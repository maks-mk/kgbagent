"""
Compatibility facade for filesystem tools.
Tool names and imports stay stable while the implementation lives in smaller internal modules.
"""

import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional, Protocol, cast
from urllib.parse import unquote, urlsplit

import aiofiles
import httpx
from langchain_core.tools import tool
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from core.config import DEFAULT_MAX_FILE_SIZE
from core.errors import ErrorType, format_error
from core.safety_policy import SafetyPolicy
from tools.filesystem_impl.manager import FilesystemManager as _FilesystemManager

logger = logging.getLogger(__name__)


class FilesystemBackend(Protocol):
    cwd: Path
    safety_policy: SafetyPolicy | None

    def set_policy(self, policy: SafetyPolicy) -> None: ...
    def read_file(self, path: str, offset: int = 0, limit: int = 2000, show_line_numbers: bool = False) -> str: ...
    def write_file(self, path: str, content: str) -> str: ...
    def edit_file(self, path: str, old_string: str, new_string: str, *, replace_all: bool = False) -> str: ...
    def list_files(self, path: str = ".", include_hidden: bool = False) -> str: ...
    def delete_file(self, path: str) -> str: ...
    def delete_directory(self, path: str, recursive: bool = False) -> str: ...


FilesystemManager = _FilesystemManager

_WORKING_DIRECTORY: Path = Path.cwd().resolve()


def create_filesystem_backend(*, virtual_mode: bool = True) -> FilesystemBackend:
    return FilesystemManager(root_dir=_WORKING_DIRECTORY, virtual_mode=virtual_mode)


fs_manager: FilesystemBackend = create_filesystem_backend(virtual_mode=True)


def _sync_backend_working_directory() -> Path:
    cwd = Path(_WORKING_DIRECTORY).resolve()
    if fs_manager.cwd != cwd:
        fs_manager.cwd = cwd
    return cwd


def _get_synced_backend() -> FilesystemBackend:
    _sync_backend_working_directory()
    return fs_manager


def set_safety_policy(policy: SafetyPolicy):
    fs_manager.set_policy(policy)


def set_working_directory(cwd: str):
    global _WORKING_DIRECTORY
    _WORKING_DIRECTORY = Path(cwd).resolve()
    fs_manager.cwd = _WORKING_DIRECTORY


def set_read_only_roots(roots):
    """Register additional roots that read_file/list_files may reach outside the
    workspace (read-only). Used for the bundled skills folder so a skill's
    auxiliary files stay readable when the workspace is a different project."""
    setter = getattr(fs_manager, "set_read_only_roots", None)
    if callable(setter):
        setter(tuple(roots))


def resolve_workspace_path(path: str) -> Path:
    manager = cast(Any, fs_manager)
    manager.cwd = _sync_backend_working_directory()
    return manager._resolve_path(path)


def max_filesystem_file_size() -> int:
    policy = getattr(fs_manager, "safety_policy", None)
    return policy.max_file_size if policy else DEFAULT_MAX_FILE_SIZE


_EDIT_FILE_ALIAS_MAP = {
    "old_string": ("old_text", "search_text", "find_text", "target_text", "old", "before", "from"),
    "new_string": ("new_text", "replace_text", "replacement", "replace_with", "new", "after", "to"),
}

_PATH_ALIASES = ("file_path", "filepath")
_WRITE_FILE_CONTENT_ALIASES = ("text", "body", "contents")


def _cleanup_edit_path(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.strip("\"'")
    text = text.splitlines()[0].strip()
    text = re.sub(r"^(?:path|file_path|filepath)\s*[:=]\s*", "", text, flags=re.IGNORECASE)

    # Defensive cleanup for malformed LLM arguments where browser UA lands in `path`.
    for marker in (" Mozilla/", " AppleWebKit/", " Safari/", " Chrome/", " Firefox/", " Edg/"):
        if marker in text:
            text = text.split(marker, 1)[0].rstrip(" ,;")
            break

    return text or None


def _resolve_payload_path(data: dict[str, Any]) -> str | None:
    for key in ("path", *_PATH_ALIASES, "dir_path", "directory_path"):
        if key not in data:
            continue
        cleaned = _cleanup_edit_path(data.get(key))
        if cleaned:
            return cleaned
    return None


def _resolve_payload_value(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key not in data:
            continue
        value = data.get(key)
        if value is None:
            continue
        return str(value)
    return None


class EditFileInput(BaseModel):
    """Input for edit_file; aliases accepted for old/new text."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("path", "file_path", "filepath"),
        description="File path.",
    )
    old_string: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "old_string", "old_text", "search_text", "find_text", "target_text", "old", "before", "from_", "from"
        ),
        description="Exact text to replace.",
    )
    new_string: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "new_string", "new_text", "replace_text", "replacement", "replace_with", "new", "after", "to"
        ),
        description="Replacement text.",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all exact matches instead of requiring a unique one.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)

        for field_name, aliases in _EDIT_FILE_ALIAS_MAP.items():
            if data.get(field_name) is not None:
                continue
            for alias in aliases:
                if data.get(alias) is not None:
                    data[field_name] = data[alias]
                    break

        data["path"] = _resolve_payload_path(data)

        for key in ("old_string", "new_string"):
            if data.get(key) is None:
                continue
            text = str(data[key]).replace("\r\n", "\n")
            data[key] = text

        return data


class WriteFileInput(BaseModel):
    """Input for write_file with explicit guidance for full-file writes."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("path", "file_path", "filepath"),
        description="Workspace-relative file path to create or overwrite.",
    )
    content: str = Field(
        default="",
        validation_alias=AliasChoices("content", "text", "body", "contents"),
        description=(
            "Complete final file contents to write. Provide the full body of the file, "
            "not a summary, placeholder, or filename."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data["path"] = _resolve_payload_path(data)
        if data.get("content") in {None, ""}:
            resolved_content = _resolve_payload_value(data, ("content", *_WRITE_FILE_CONTENT_ALIASES))
            if resolved_content is not None:
                data["content"] = resolved_content
        if data.get("content") is not None:
            data["content"] = str(data["content"]).replace("\r\n", "\n")
        return data


class DeleteFileInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("path", "file_path", "filepath"),
        description="Workspace-relative file path to delete.",
    )


class DeleteDirectoryInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("path", "dir_path", "directory_path"),
        description="Workspace-relative directory path to delete.",
    )
    recursive: bool = Field(default=False, description="Delete non-empty directory recursively when true.")


@tool("read_file")
def read_file_tool(path: str, offset: int = 0, limit: int = 2000, show_line_numbers: bool = False) -> str:
    """Read a file with pagination."""
    return _get_synced_backend().read_file(path, offset, limit, show_line_numbers)


@tool("write_file", args_schema=WriteFileInput)
def write_file_tool(
    path: str | None = None,
    content: str | None = None,
    file_path: str | None = None,
    filepath: str | None = None,
    text: str | None = None,
    body: str | None = None,
    contents: str | None = None,
    **kwargs: Any,
) -> str:
    """Create or overwrite a file using the exact full content provided."""
    resolved_path = _resolve_payload_path(
        {"path": path, "file_path": file_path, "filepath": filepath, **kwargs}
    )
    if not resolved_path:
        return format_error(ErrorType.VALIDATION, "Missing required field: path.")
    if content in {None, ""}:
        content = _resolve_payload_value(
            {
                "content": content,
                "text": text,
                "body": body,
                "contents": contents,
                **kwargs,
            },
            ("content", *_WRITE_FILE_CONTENT_ALIASES),
        )
    if content is None or not str(content).strip():
        return format_error(ErrorType.VALIDATION, "Missing required field: content.")
    return _get_synced_backend().write_file(resolved_path, str(content))


@tool("edit_file", args_schema=EditFileInput)
def edit_file_tool(
    path: str | None = None,
    old_string: str | None = None,
    new_string: str | None = None,
    replace_all: bool = False,
    file_path: str | None = None,
    filepath: str | None = None,
    old_text: str | None = None,
    search_text: str | None = None,
    find_text: str | None = None,
    target_text: str | None = None,
    old: str | None = None,
    before: str | None = None,
    from_: str | None = None,
    new_text: str | None = None,
    replace_text: str | None = None,
    replacement: str | None = None,
    replace_with: str | None = None,
    new: str | None = None,
    after: str | None = None,
    to: str | None = None,
    **kwargs: Any,
) -> str:
    """Replace text in a file."""
    resolved_path = _resolve_payload_path(
        {"path": path, "file_path": file_path, "filepath": filepath, **kwargs}
    )
    if not resolved_path:
        return format_error(ErrorType.VALIDATION, "Missing required field: path.")
    if old_string is None:
        old_string = _resolve_payload_value(
            {
                "old_string": old_string,
                "old_text": old_text,
                "search_text": search_text,
                "find_text": find_text,
                "target_text": target_text,
                "old": old,
                "before": before,
                "from": from_,
                **kwargs,
            },
            ("old_string", *_EDIT_FILE_ALIAS_MAP["old_string"]),
        )
    if old_string is None or not str(old_string).strip():
        return format_error(
            ErrorType.VALIDATION,
            "Missing required field: old_string. Accepted aliases: old_text, search_text, find_text.",
        )
    if new_string is None:
        new_string = _resolve_payload_value(
            {
                "new_string": new_string,
                "new_text": new_text,
                "replace_text": replace_text,
                "replacement": replacement,
                "replace_with": replace_with,
                "new": new,
                "after": after,
                "to": to,
                **kwargs,
            },
            ("new_string", *_EDIT_FILE_ALIAS_MAP["new_string"]),
        )
    if new_string is None:
        return format_error(
            ErrorType.VALIDATION,
            "Missing required field: new_string. Accepted aliases: new_text, replace_text, replacement.",
        )
    backend = _get_synced_backend()
    if replace_all:
        return backend.edit_file(resolved_path, old_string, new_string, replace_all=True)
    return backend.edit_file(resolved_path, old_string, new_string)


@tool("list_directory")
def list_directory_tool(path: str = ".", include_hidden: bool = False) -> str:
    """List files and directories."""
    return _get_synced_backend().list_files(path, include_hidden)


@tool("safe_delete_file", args_schema=DeleteFileInput)
async def safe_delete_file(path: str | None = None, **kwargs: Any) -> str:
    """Delete a workspace file."""
    resolved_path = _resolve_payload_path({"path": path, **kwargs})
    if not resolved_path:
        return format_error(ErrorType.VALIDATION, "Missing required field: path.")
    backend = _get_synced_backend()
    return await asyncio.to_thread(backend.delete_file, resolved_path)


@tool("safe_delete_directory", args_schema=DeleteDirectoryInput)
async def safe_delete_directory(path: str | None = None, recursive: bool = False, **kwargs: Any) -> str:
    """Delete a workspace directory."""
    resolved_path = _resolve_payload_path({"path": path, **kwargs})
    if not resolved_path:
        return format_error(ErrorType.VALIDATION, "Missing required field: path.")
    backend = _get_synced_backend()
    return await asyncio.to_thread(backend.delete_directory, resolved_path, recursive)


_DOWNLOAD_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}
_DOWNLOAD_CLIENT: Optional[httpx.AsyncClient] = None


def _get_download_client() -> httpx.AsyncClient:
    global _DOWNLOAD_CLIENT
    if _DOWNLOAD_CLIENT is None:
        _DOWNLOAD_CLIENT = httpx.AsyncClient(timeout=15, follow_redirects=True)
    return _DOWNLOAD_CLIENT


async def close_download_client() -> None:
    global _DOWNLOAD_CLIENT
    if _DOWNLOAD_CLIENT is None:
        return
    client, _DOWNLOAD_CLIENT = _DOWNLOAD_CLIENT, None
    await client.aclose()


def _cleanup_partial_download(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _format_download_http_error(exc: httpx.HTTPStatusError) -> str:
    status_code = exc.response.status_code
    reason = exc.response.reason_phrase or "Unknown error"
    if status_code == 403:
        return format_error(
            ErrorType.ACCESS_DENIED,
            "HTTP 403 - Forbidden. Remote host blocked the download or requires browser-only access.",
        )
    if status_code == 404:
        return format_error(
            ErrorType.NOT_FOUND,
            "HTTP 404 - Not Found. Check that the URL points to the direct file, not a landing page.",
        )
    return format_error(ErrorType.NETWORK, f"HTTP {status_code} - {reason}")


def _format_download_request_error(exc: httpx.RequestError) -> str:
    detail = str(exc).strip() or type(exc).__name__
    return format_error(ErrorType.NETWORK, f"Network request failed ({type(exc).__name__}): {detail}")


@tool("download_file")
async def download_file(url: str, filename: Optional[str] = None) -> str:
    """Download HTTP(S) content atomically; partial files never replace the destination."""
    temp_destination: Optional[Path] = None
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return format_error(ErrorType.VALIDATION, "A valid HTTP(S) URL is required.")
        if not filename:
            name = unquote(parsed.path.rsplit("/", 1)[-1]).replace("\\", "/").rsplit("/", 1)[-1]
            filename = name if name not in {"", ".", ".."} else "downloaded_file"

        try:
            destination = resolve_workspace_path(filename)
        except ValueError as exc:
            return format_error(ErrorType.VALIDATION, str(exc))
        if destination.is_dir():
            return format_error(ErrorType.VALIDATION, "Download destination is a directory.")

        client = _get_download_client()
        logger.info("Downloading to %s", destination)
        async with client.stream("GET", url, follow_redirects=True, headers=_DOWNLOAD_HEADERS) as response:
            response.raise_for_status()
            max_size = max_filesystem_file_size()
            length = response.headers.get("content-length")
            if length:
                try:
                    declared = int(length)
                except ValueError:
                    declared = None
                if declared is not None and declared > max_size:
                    return format_error(ErrorType.LIMIT_EXCEEDED, f"File too large (>{max_size} bytes). Download aborted.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part", dir=destination.parent)
            os.close(fd)
            temp_destination = Path(name)
            async with aiofiles.open(temp_destination, "wb") as handle:
                downloaded = 0
                async for chunk in response.aiter_bytes():
                    downloaded += len(chunk)
                    if downloaded > max_size:
                        return format_error(ErrorType.LIMIT_EXCEEDED, f"File exceeded max size {max_size}. Aborted.")
                    await handle.write(chunk)
        temp_destination.replace(destination)
        return f"Success: File downloaded to {destination}"
    except httpx.HTTPStatusError as exc:
        return _format_download_http_error(exc)
    except httpx.RequestError as exc:
        return _format_download_request_error(exc)
    except Exception as exc:
        return format_error(ErrorType.EXECUTION, f"Error downloading file: {exc}")
    finally:
        # Runs after handles close, also on cancellation/size errors on Windows.
        if temp_destination is not None:
            _cleanup_partial_download(temp_destination)


__all__ = [
    "FilesystemManager",
    "FilesystemBackend",
    "create_filesystem_backend",
    "fs_manager",
    "set_safety_policy",
    "set_working_directory",
    "set_read_only_roots",
    "resolve_workspace_path",
    "max_filesystem_file_size",
    "read_file_tool",
    "write_file_tool",
    "edit_file_tool",
    "list_directory_tool",
    "safe_delete_file",
    "safe_delete_directory",
    "download_file",
    "close_download_client",
    "_DOWNLOAD_HEADERS",
    "_format_download_http_error",
]
