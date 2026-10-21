import errno
import itertools
import json
import os
from pathlib import Path
from typing import Optional, Union

from core.config import DEFAULT_MAX_FILE_SIZE, DEFAULT_READ_LIMIT
from core.errors import ErrorType, format_error
from core.safety_policy import SafetyPolicy
from core.utils import truncate_output

from .editing import edit_text_file
from .pathing import (
    atomic_write_text,
    count_file_lines,
    delete_directory_path,
    is_binary_path,
    resolve_existing_path,
    resolve_path,
)


class FilesystemManager:
    __slots__ = ("cwd", "virtual_mode", "safety_policy", "read_only_roots")

    def __init__(self, root_dir: Union[str, Path] = None, virtual_mode: bool = True):
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode
        self.safety_policy: Optional[SafetyPolicy] = None
        # Additional roots that read-only operations (read_file, list_files) may
        # reach even when they live outside the workspace cwd — e.g. the bundled
        # skills folder next to the executable. Never consulted for write/edit/
        # delete, which stay confined to the workspace.
        self.read_only_roots: tuple[Path, ...] = ()

    def set_policy(self, policy: SafetyPolicy):
        self.safety_policy = policy

    def set_read_only_roots(self, roots: tuple[Path, ...]):
        self.read_only_roots = tuple(Path(root).resolve() for root in roots)

    def _is_binary(self, path: Union[str, Path]) -> bool:
        return is_binary_path(str(path))

    def _count_lines(self, path: Path) -> int:
        return count_file_lines(path)

    def _resolve_path(self, path_str: str, *, for_read: bool = False) -> Path:
        extra = self.read_only_roots if for_read else ()
        return resolve_path(self.cwd, self.virtual_mode, path_str, extra)

    def _resolve_existing(self, path: str, expected: str, *, follow_final_symlink: bool = True) -> Path:
        return resolve_existing_path(self.cwd, self.virtual_mode, path, expected, follow_final_symlink=follow_final_symlink)

    def _tool_limit(self) -> int:
        return self.safety_policy.max_tool_output if self.safety_policy else 5000

    def _truncate(self, output: str, source: str = "filesystem") -> str:
        return truncate_output(output, self._tool_limit(), source=source)

    def delete_file(self, path: str) -> str:
        try:
            # follow_final_symlink=False: unlink the link itself, never its target.
            target = self._resolve_existing(path, "file", follow_final_symlink=False)
            target.unlink()
            return f"Success: File {path} deleted."
        except FileNotFoundError:
            return format_error(ErrorType.NOT_FOUND, f"File not found: {path}")
        except IsADirectoryError:
            return format_error(ErrorType.VALIDATION, f"{path} is a directory. Use safe_delete_directory.")
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, str(exc))

    def delete_directory(self, path: str, recursive: bool = False) -> str:
        try:
            target = self._resolve_existing(path, "dir", follow_final_symlink=False)
            if self.cwd.resolve().is_relative_to(target):
                return format_error(ErrorType.VALIDATION, "Refusing to delete the workspace root or its ancestor.")
            delete_directory_path(target, recursive)
            if recursive:
                return f"Success: Directory {path} deleted recursively."
            return f"Success: Empty directory {path} deleted."
        except FileNotFoundError:
            return format_error(ErrorType.NOT_FOUND, f"Directory not found: {path}")
        except NotADirectoryError:
            return format_error(ErrorType.VALIDATION, f"{path} is not a directory; use safe_delete_file for symlinks.")
        except OSError as exc:
            if exc.errno == errno.ENOTEMPTY:
                return format_error(ErrorType.VALIDATION, "Directory is not empty. Set recursive=True to delete it.")
            return format_error(ErrorType.EXECUTION, str(exc))
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, str(exc))

    def _read_file_page(self, path: str, offset: int, limit: int, show_line_numbers: bool, char_budget: int) -> dict:
        """Read a bounded page; metadata is kept separate so truncation cannot hide it."""
        try:
            if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
                raise ValueError("offset must be a non-negative integer.")
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                raise ValueError("limit must be a positive integer.")
            if not isinstance(show_line_numbers, bool):
                raise ValueError("show_line_numbers must be a boolean.")
            target = self._resolve_path(path, for_read=True)
            if not target.exists():
                return {"error": format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")}
            if not target.is_file():
                raise ValueError(f"'{path}' is not a file.")
            stats = target.stat()
            max_size = self.safety_policy.max_file_size if self.safety_policy else DEFAULT_MAX_FILE_SIZE
            if stats.st_size > max_size:
                return {"error": format_error(ErrorType.LIMIT_EXCEEDED, f"File is too large ({stats.st_size} bytes). Max: {max_size}.")}
            if self._is_binary(target):
                raise ValueError("File binary or unknown encoding.")
            total_lines = self._count_lines(target)
            if offset >= total_lines and total_lines > 0:
                raise ValueError(f"Line offset {offset} exceeds file length ({total_lines} lines).")
            policy_limit = self.safety_policy.max_read_lines if self.safety_policy else DEFAULT_READ_LIMIT
            limit = min(limit, policy_limit)
            result = []
            chars_used = 0
            lines_read = 0
            truncated_line = False
            with open(target, "r", encoding="utf-8") as file_obj:
                for index, line in enumerate(itertools.islice(file_obj, offset, offset + limit)):
                    clean_line = line.rstrip("\n").rstrip("\r")
                    formatted = f"{offset + index + 1:6}  {clean_line}" if show_line_numbers else clean_line
                    line_chars = len(formatted) + (1 if result else 0)
                    if chars_used + line_chars > char_budget:
                        if not result and char_budget > 0:
                            result.append(formatted[:char_budget])
                            lines_read = 1
                            truncated_line = True
                        break
                    result.append(formatted)
                    chars_used += line_chars
                    lines_read += 1
            return {"content": "\n".join(result), "offset": offset, "end": offset + lines_read if total_lines else 0,
                    "total": total_lines, "size": stats.st_size, "limit": limit,
                    "truncated_line": truncated_line, "lines_read": lines_read}
        except UnicodeDecodeError:
            return {"error": format_error(ErrorType.VALIDATION, "File is not valid UTF-8. Convert it to UTF-8 before reading.")}
        except ValueError as exc:
            return {"error": format_error(ErrorType.VALIDATION, str(exc))}
        except Exception as exc:
            return {"error": format_error(ErrorType.EXECUTION, f"Error reading file: {exc}")}


    def read_file(self, path: str, offset: int = 0, limit: int = DEFAULT_READ_LIMIT, show_line_numbers: bool = False) -> str:
        budget = max(0, self._tool_limit())
        page = self._read_file_page(path, offset, limit, show_line_numbers, budget)
        if "error" in page:
            return self._truncate(page["error"], source="read_file")
        if page["size"] == 0:
            message = "System reminder: File exists but has empty contents."
            if len(message) <= budget:
                return message
            return self._truncate(format_error(ErrorType.LIMIT_EXCEEDED, "Increase max_tool_output."), source="read_file")

        def footer() -> str:
            metadata = ""
            if page["truncated_line"]:
                metadata += (f"\n\n[TRUNCATED LINE] Line {offset + 1}: only its prefix is shown. "
                             "The remainder of this line is omitted; use cli_exec to read character "
                             "ranges or increase max_tool_output.")
            end_index = page["end"]
            if page["total"] > end_index:
                metadata += (f"\n\n[TRUNCATED] Showing lines {offset + 1}-{end_index} of {page['total']} "
                             f"({page['size']} bytes total). "
                             f"To continue: read_file(path={path!r}, offset={end_index}, "
                             f"limit={page['limit']}, show_line_numbers={show_line_numbers})")
            elif (offset or show_line_numbers) and not page["truncated_line"]:
                metadata += f"\n\n[EOF] Lines {offset + 1}-{end_index} of {page['total']} ({page['size']} bytes)."
            return metadata

        # Trim content, never metadata, and preserve complete line boundaries first.
        content = page["content"]
        metadata = footer()
        while len(content) + len(metadata) > budget:
            if page["lines_read"] > 1:
                content = content.rsplit("\n", 1)[0]
                page["lines_read"] -= 1
                page["end"] -= 1
            else:
                page["truncated_line"] = True
                metadata = footer()
                allowance = budget - len(metadata)
                if allowance < 1:
                    return self._truncate(format_error(ErrorType.LIMIT_EXCEEDED, "Increase max_tool_output to retain read metadata."), source="read_file")
                content = content[:allowance]
            metadata = footer()
        return content + metadata


    def read_files(self, requests: list[dict], limit: int = DEFAULT_READ_LIMIT) -> str:
        """Internal batch implementation of read_file, with one shared output/line budget."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            return self._truncate(format_error(ErrorType.VALIDATION, "limit must be a positive integer."), source="read_file")
        if not isinstance(requests, list) or not 1 <= len(requests) <= 8:
            return self._truncate(format_error(ErrorType.VALIDATION, "requests must contain 1-8 file ranges."), source="read_file")
        for index, request in enumerate(requests, 1):
            if not isinstance(request, dict) or set(request) - {"path", "offset", "limit", "show_line_numbers"}:
                return self._truncate(format_error(ErrorType.VALIDATION, f"Request {index}: invalid fields."), source="read_file")
            if not isinstance(request.get("path"), str) or not request["path"].strip():
                return self._truncate(format_error(ErrorType.VALIDATION, f"Request {index}: path is required."), source="read_file")
        labels = [json.dumps(request["path"], ensure_ascii=False) for request in requests]
        # Reserve headers, errors, totals, and continuation for every item, before IO.
        numeric_width = len(str(self.safety_policy.max_file_size if self.safety_policy else DEFAULT_MAX_FILE_SIZE))
        reserves = [
            400 + 2 * len(label) + 3 * numeric_width
            + sum(len(str(request.get(key, 0))) for key in ("offset", "limit") if isinstance(request.get(key, 0), int))
            for request, label in zip(requests, labels)
        ]
        remaining_chars = self._tool_limit() - sum(reserves) - 200
        if remaining_chars < len(requests):
            return self._truncate(format_error(ErrorType.LIMIT_EXCEEDED, "Batch metadata exceeds max_tool_output; use fewer requests or shorter paths."), source="read_file")
        remaining_lines = min(limit, self.safety_policy.max_read_lines if self.safety_policy else DEFAULT_READ_LIMIT)
        parts = []
        failures = []
        incomplete = False
        for index, (request, label) in enumerate(zip(requests, labels), 1):
            offset = request.get("offset", 0)
            limit = request.get("limit", DEFAULT_READ_LIMIT)
            numbered = request.get("show_line_numbers", False)
            pending = len(requests) - index + 1
            allowance = remaining_chars // pending
            line_allowance = remaining_lines // pending
            header = f"--- Request {index}: path={label} ---\n"
            if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
                page = {"error": format_error(ErrorType.VALIDATION, "offset must be a non-negative integer.")}
            elif not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                page = {"error": format_error(ErrorType.VALIDATION, "limit must be a positive integer.")}
            elif not isinstance(numbered, bool):
                page = {"error": format_error(ErrorType.VALIDATION, "show_line_numbers must be a boolean.")}
            elif allowance < 1 or line_allowance < 1:
                incomplete = True
                parts.append(header + f"status=SKIPPED; shared budget exhausted. Retry read_file(path={label}, offset={offset})."
                             + f"\n--- End request {index} ---")
                continue
            else:
                page = self._read_file_page(request["path"], offset, min(limit, line_allowance), numbered, allowance)
            if "error" in page:
                error = page["error"]
                failures.append(error.split("]: ", 1)[0].removeprefix("ERROR["))
                # Bound the whole error item, including its truncation marker and footer.
                prefix = header + "status=ERROR\n"
                footer = f"\n--- End request {index} ---"
                error_budget = reserves[index - 1] - len(prefix) - len(footer)
                if len(error) > error_budget:
                    marker = "... [TRUNCATED ERROR]"
                    error = error[:error_budget - len(marker)] + marker
                parts.append(prefix + error + footer)
                continue
            remaining_lines -= page["lines_read"]
            remaining_chars -= len(page["content"])
            partial = page["truncated_line"] or page["end"] < page["total"]
            incomplete |= partial
            status = "TRUNCATED" if partial else "OK"
            metadata = (f"status={status}; lines={offset + 1 if page['total'] else 0}-{page['end']} "
                        f"of {page['total']}; bytes={page['size']}\n")
            footer = ""
            if page["truncated_line"]:
                footer += (f"\n[TRUNCATED LINE] Line {offset + 1}: prefix only; its remainder is omitted. "
                           "Read character ranges with cli_exec or increase max_tool_output.")
            if page["end"] < page["total"]:
                footer += (f"\nTo continue: read_file(path={label}, offset={page['end']}, "
                           f"limit={limit}, show_line_numbers={numbered})")
            body = page["content"] if page["size"] else "System reminder: File exists but has empty contents."
            parts.append(header + metadata + body + footer + f"\n--- End request {index} ---")
        if failures:
            # Preserve existing result parsing: partial failures are not reported as success.
            error_type = failures[0] if len(set(failures)) == 1 else ErrorType.EXECUTION
            summary = format_error(error_type, f"Batch read: {len(failures)} of {len(requests)} requests failed; successful pages follow.")
        else:
            summary = f"Batch read: {'TRUNCATED' if incomplete else 'OK'}; {len(requests)} request(s)."
        output = summary + "\n\n" + "\n\n".join(parts)
        if len(output) > self._tool_limit():
            return self._truncate(format_error(ErrorType.LIMIT_EXCEEDED, "Batch output metadata exceeds max_tool_output; retry with fewer requests."), source="read_file")
        return output


    def write_file(self, path: str, content: str) -> str:
        try:
            target = self._resolve_path(path)
            if not isinstance(content, str):
                return format_error(ErrorType.VALIDATION, "content must be a string.")
            atomic_write_text(target, content)
            return f"Success: File '{path}' saved ({len(content)} chars)."
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, f"Error writing file: {exc}")

    def edit_file(
        self, path: str, old_string: str | None = None, new_string: str | None = None,
        *, replace_all: bool = False, match_mode: str = "exact", edits: list[dict] | None = None,
    ) -> str:
        try:
            return self._truncate(edit_text_file(
                self._resolve_path(path), path, old_string, new_string,
                replace_all=replace_all, match_mode=match_mode, edits=edits,
            ))
        except ValueError as exc:
            return format_error(ErrorType.VALIDATION, str(exc))
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, f"Error editing file: {exc}")

    def list_files(self, path: str, include_hidden: bool = False) -> str:
        try:
            target = self._resolve_path(path, for_read=True)
            if not target.exists():
                return format_error(ErrorType.NOT_FOUND, f"Path '{path}' not found.")
            if target.is_file():
                return f"[FILE] {target.name} ({target.stat().st_size} bytes)"

            results = []
            try:
                with os.scandir(target) as entries:
                    ordered = sorted(list(entries), key=lambda entry: (not entry.is_dir(follow_symlinks=False), entry.name.lower()))
                    for entry in ordered:
                        if entry.name.startswith(".") and not include_hidden:
                            continue
                        if entry.name in {".DS_Store", "__pycache__"}:
                            continue
                        prefix = "[DIR] " if entry.is_dir(follow_symlinks=False) else "[FILE]"
                        results.append(f"{prefix} {entry.name}")
            except PermissionError:
                return format_error(ErrorType.EXECUTION, "Permission denied while accessing directory.")

            output = f"Directory '{path}':\n" + "\n".join(results) + f"\n\n(Total {len(results)} items)"
            return self._truncate(output)
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, f"Error listing directory: {exc}")
