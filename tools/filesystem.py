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
from typing import Any, Literal, Optional, Protocol, cast
from urllib.parse import unquote, urlsplit

import aiofiles
import httpx
from langchain_core.tools import tool
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from core.config import DEFAULT_MAX_FILE_SIZE
from core.errors import ErrorType, format_error
from core.multimodal import can_read_image_file, image_dimensions, image_mime_type_for_file
from core.safety_policy import SafetyPolicy
from tools.filesystem_impl.manager import FilesystemManager as _FilesystemManager

logger = logging.getLogger(__name__)


class FilesystemBackend(Protocol):
    cwd: Path
    safety_policy: SafetyPolicy | None

    def set_policy(self, policy: SafetyPolicy) -> None: ...
    def read_file(self, path: str, offset: int = 0, limit: int = 2000, show_line_numbers: bool = False) -> str: ...
    def read_files(self, requests: list[dict[str, Any]], limit: int = 2000) -> str: ...
    def write_file(self, path: str, content: str) -> str: ...
    def edit_file(self, path: str, old_string: str | None = None, new_string: str | None = None, *, replace_all: bool = False, match_mode: str = "exact", edits: list[dict[str, Any]] | None = None) -> str: ...
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


def resolve_readable_path(path: str) -> Path:
    """Resolve a path the way read-only tools do (read-only roots stay reachable)."""
    manager = cast(Any, fs_manager)
    manager.cwd = _sync_backend_working_directory()
    return manager._resolve_path(path, for_read=True)


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


class FileEdit(BaseModel):
    """One ordered replacement; paths belong to the outer edit_file call."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    old_string: str = Field(min_length=1, validation_alias=AliasChoices("old_string", *_EDIT_FILE_ALIAS_MAP["old_string"], "from_"), description="Exact current text, including whitespace; omit read_file line numbers/metadata.")
    new_string: str = Field(validation_alias=AliasChoices("new_string", *_EDIT_FILE_ALIAS_MAP["new_string"]), description="Replacement text; empty string deletes the match.")
    replace_all: bool = Field(default=False, description="Replace all matches, otherwise require a unique match.")
    match_mode: Literal["exact", "indentation"] = Field(default="exact", description="Only opt into indentation matching intentionally; internal/trailing whitespace stays exact.")


class ReadFileRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    path: str = Field(min_length=1, validation_alias=AliasChoices("path", *_PATH_ALIASES), description="Text file path.")
    offset: int = Field(default=0, ge=0, description="Zero-based starting line; omit to inherit the top-level offset.")
    limit: int = Field(default=2000, ge=1, description="Maximum lines; shares the call's line/output budget.")
    show_line_numbers: bool = Field(default=False, description="Prefix lines with their one-based numbers; omit to inherit the top-level setting.")


class ReadFileInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    path: str | None = Field(default=None, validation_alias=AliasChoices("path", *_PATH_ALIASES), description="Single text file path; omit or use null when using requests.")
    offset: int = Field(default=0, ge=0, description="Zero-based starting line for a single file, or default offset for requests. Explicit per-item offsets take precedence.")
    limit: int = Field(default=2000, ge=1, description="Maximum lines for a single file or shared across all requests.")
    show_line_numbers: bool = Field(default=False, description="Show one-based line numbers for a single file, or by default for requests. Explicit per-item settings take precedence.")
    requests: list[ReadFileRequest] | None = Field(default=None, min_length=1, max_length=8, description="Read 1-8 known file ranges in order with one shared output/line budget. Use instead of path; per-item offset/show_line_numbers override top-level defaults.")


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
        description="Exact current text, including whitespace; omit read_file line numbers/metadata.",
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
        description="Replace all matches instead of requiring a unique one.",
    )
    match_mode: Literal["exact", "indentation"] = Field(
        default="exact", description="Exact by default. Explicit indentation mode allows only leading-indentation adjustment.",
    )
    edits: list[FileEdit] | None = Field(
        default=None, min_length=1, max_length=20,
        description="1-20 ordered replacements in this file, each seeing the preceding edits. Use instead of top-level old_string/new_string/replace_all/match_mode. All must validate before one write.",
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


class ReadImagesInput(BaseModel):
    """Input for read_images; one path or a list of paths is accepted."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    paths: str | list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("paths", "path", *_PATH_ALIASES, "files", "images"),
        description="Image file path, or a list of image file paths, to show to the model.",
    )
    max_images: int | None = Field(
        default=None,
        validation_alias=AliasChoices("max_images", "max_files", "limit"),
        description="Optional lower cap on how many images this call returns.",
    )


@tool("read_file", args_schema=ReadFileInput)
def read_file_tool(
    path: str | None = None, offset: int = 0, limit: int = 2000, show_line_numbers: bool = False,
    requests: list[ReadFileRequest] | None = None,
) -> str:
    """Read one text file with pagination, or up to 8 file ranges with a shared budget and optional defaults."""
    backend = _get_synced_backend()
    if requests is not None:
        if path is not None and path.strip():
            return format_error(ErrorType.VALIDATION, "Provide either path or requests, not both. For batch reads, put every path inside requests and omit the top-level path (or set it to null).")
        ranges = []
        for item in requests:
            # Keep explicit 0/False overrides while inheriting only omitted fields.
            request = item.model_dump(exclude_unset=True) if isinstance(item, BaseModel) else item
            if isinstance(request, dict):
                request = {"offset": offset, "show_line_numbers": show_line_numbers, **request}
            ranges.append(request)
        return backend.read_files(ranges, limit=limit)
    if not path or not path.strip():
        return format_error(ErrorType.VALIDATION, "Provide path or requests.")
    return backend.read_file(path, offset, limit, show_line_numbers)


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
    match_mode: str = "exact",
    edits: list[FileEdit] | None = None,
    **kwargs: Any,
) -> str:
    """Replace exact text in one file; ordered edits validate together before a single write."""
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
    if edits is not None:
        supplied_new = _resolve_payload_value(
            {"new_string": new_string, "new_text": new_text, "replace_text": replace_text,
             "replacement": replacement, "replace_with": replace_with, "new": new, "after": after, "to": to, **kwargs},
            ("new_string", *_EDIT_FILE_ALIAS_MAP["new_string"]),
        )
        if old_string is not None or supplied_new is not None or replace_all or match_mode != "exact":
            return format_error(ErrorType.VALIDATION, "Do not mix edits with top-level old_string/new_string/replace_all/match_mode.")
        operations = [item.model_dump() if isinstance(item, BaseModel) else item for item in edits]
        return _get_synced_backend().edit_file(resolved_path, edits=operations)
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
    if match_mode != "exact":
        return backend.edit_file(resolved_path, old_string, new_string, replace_all=replace_all, match_mode=match_mode)
    if replace_all:
        return backend.edit_file(resolved_path, old_string, new_string, replace_all=True)
    return backend.edit_file(resolved_path, old_string, new_string)


@tool("list_directory")
def list_directory_tool(path: str = ".", include_hidden: bool = False) -> str:
    """List files and directories."""
    return _get_synced_backend().list_files(path, include_hidden)


# Each attached image costs vision tokens and stays in the conversation, so one
# call is bounded; the model can call again for the remaining paths.
_IMAGE_READ_LIMIT = 4
# Providers reject vector formats as inline images; failing here is clearer than
# an opaque provider error.
_UNSUPPORTED_IMAGE_MIME_TYPES = {"image/svg+xml"}


def _normalize_image_paths(raw: Any) -> list[str]:
    if isinstance(raw, str):
        items: list[Any] = [raw]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        return []

    paths: list[str] = []
    for item in items:
        cleaned = _cleanup_edit_path(item)
        if cleaned:
            paths.append(cleaned)
    return paths


def _read_image_block(path_field: str) -> tuple[dict[str, Any] | None, str, ErrorType]:
    """Validate one requested path and return (image block, note, error type)."""
    try:
        target = resolve_readable_path(path_field)
    except ValueError as exc:
        return None, f"{path_field}: {exc}", ErrorType.ACCESS_DENIED
    except Exception as exc:
        return None, f"{path_field}: {exc}", ErrorType.EXECUTION

    if not target.exists():
        return None, f"{path_field}: file not found.", ErrorType.NOT_FOUND
    if not target.is_file():
        return None, f"{path_field}: not a file.", ErrorType.VALIDATION

    mime_type = image_mime_type_for_file(target)
    if mime_type in _UNSUPPORTED_IMAGE_MIME_TYPES:
        return (
            None,
            f"{path_field}: {mime_type} cannot be attached as an image; convert it to PNG or JPEG first.",
            ErrorType.VALIDATION,
        )
    if not can_read_image_file(target):
        return None, f"{path_field}: not a readable image file.", ErrorType.VALIDATION

    try:
        size = target.stat().st_size
    except OSError as exc:
        return None, f"{path_field}: {exc}", ErrorType.EXECUTION
    if size == 0:
        return None, f"{path_field}: file is empty.", ErrorType.VALIDATION

    max_size = max_filesystem_file_size()
    if size > max_size:
        return (
            None,
            f"{path_field}: file is too large ({size} bytes). Max: {max_size}.",
            ErrorType.LIMIT_EXCEEDED,
        )

    block: dict[str, Any] = {
        "type": "image",
        "path": str(target),
        "mime_type": mime_type,
        "file_name": target.name,
        "size_bytes": int(size),
    }
    width, height = image_dimensions(target)
    if width and height:
        block["width"] = width
        block["height"] = height
    return block, "", ErrorType.VALIDATION


@tool("read_images", args_schema=ReadImagesInput)
def read_images_tool(
    paths: str | list[str] | None = None,
    max_images: int | None = None,
    path: str | list[str] | None = None,
    file_path: str | list[str] | None = None,
    filepath: str | list[str] | None = None,
    files: str | list[str] | None = None,
    images: str | list[str] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]] | str:
    """Read image files (PNG, JPEG, GIF, WEBP, BMP, TIFF) from disk and attach them to the conversation so they become visible. Accepts one path or a list of paths, up to 4 images per call."""
    requested: list[str] = []
    for candidate in (paths, path, file_path, filepath, files, images):
        requested = _normalize_image_paths(candidate)
        if requested:
            break
    if not requested:
        return format_error(
            ErrorType.VALIDATION,
            "Missing required field: paths (an image file path or a list of image file paths).",
        )

    limit = _IMAGE_READ_LIMIT
    if isinstance(max_images, int) and not isinstance(max_images, bool) and max_images > 0:
        limit = min(limit, max_images)
    selected = requested[:limit]

    blocks: list[dict[str, Any]] = []
    notes: list[str] = []
    failures: list[ErrorType] = []
    for raw_path in selected:
        block, note, error_type = _read_image_block(raw_path)
        if block is None:
            notes.append(note)
            failures.append(error_type)
            continue
        blocks.append(block)

    if not blocks:
        error_type = failures[0] if all(item == failures[0] for item in failures) else ErrorType.VALIDATION
        return format_error(error_type, f"No images attached. {' '.join(notes)}")

    summary_lines = [f"Attached {len(blocks)} image(s) to this tool result; they are visible now."]
    for block in blocks:
        dimensions = f"{block['width']}x{block['height']}" if block.get("width") and block.get("height") else ""
        details = ", ".join(
            part for part in (block["mime_type"], dimensions, f"{block['size_bytes']} bytes") if part
        )
        summary_lines.append(f"- {block['path']} ({details})")
    if notes:
        summary_lines.append(f"Skipped {len(notes)} path(s): {' '.join(notes)}")
    if len(requested) > len(selected):
        summary_lines.append(
            f"{len(requested) - len(selected)} more image(s) were not included; call read_images again with the remaining paths."
        )

    return [{"type": "text", "text": "\n".join(summary_lines)}, *blocks]


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
