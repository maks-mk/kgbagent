import errno
import itertools
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

    # FIX: show_line_numbers default changed from True to False.
    #
    # Root cause of indentation corruption:
    #   With show_line_numbers=True the model receives lines prefixed as "     1  code",
    #   copies them into old_string/new_string, exact-match fails (prefix not in file),
    #   trim/aggressive fallback finds the block anyway, and new_string (without proper
    #   indentation) is written verbatim — corrupting indentation.
    #
    #   Changing the default to False eliminates the prefix entirely, making exact-match
    #   succeed in the normal case.  The secondary defence (indentation realignment in
    #   editing.py) handles any residual cases where indentation still drifts.
    #
    # Callers that explicitly need line numbers can still pass show_line_numbers=True.
    def read_file(self, path: str, offset: int = 0, limit: int = DEFAULT_READ_LIMIT, show_line_numbers: bool = False) -> str:
        try:
            target = self._resolve_path(path, for_read=True)
            if not target.exists():
                return format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")
            if not target.is_file():
                return format_error(ErrorType.VALIDATION, f"'{path}' is not a file.")

            stats = target.stat()
            max_size = self.safety_policy.max_file_size if self.safety_policy else DEFAULT_MAX_FILE_SIZE
            if stats.st_size > max_size:
                return format_error(ErrorType.LIMIT_EXCEEDED, f"File is too large ({stats.st_size} bytes). Max: {max_size}.")
            if self._is_binary(target):
                return format_error(ErrorType.VALIDATION, "File binary or unknown encoding.")
            if stats.st_size == 0:
                return "System reminder: File exists but has empty contents."

            total_lines = self._count_lines(target)
            policy_limit = self.safety_policy.max_read_lines if self.safety_policy else DEFAULT_READ_LIMIT
            limit = min(limit, policy_limit)
            if offset >= total_lines and total_lines > 0:
                return format_error(ErrorType.VALIDATION, f"Line offset {offset} exceeds file length ({total_lines} lines).")

            char_budget = max(500, self._tool_limit() - 120)
            result: list[str] = []
            chars_used = 0
            lines_read = 0
            long_line_warning = ""
            try:
                with open(target, "r", encoding="utf-8", errors="replace") as file_obj:
                    for index, line in enumerate(itertools.islice(file_obj, offset, offset + limit)):
                        clean_line = line.rstrip("\n").rstrip("\r")
                        formatted = f"{offset + index + 1:6}  {clean_line}" if show_line_numbers else clean_line
                        line_chars = len(formatted) + 1
                        if chars_used + line_chars > char_budget:
                            if not result:
                                # A single line longer than the whole budget: show its
                                # prefix instead of returning an empty page.
                                result.append(formatted[:char_budget])
                                lines_read = 1
                                long_line_warning = (
                                    f"\n\n[TRUNCATED LINE] Line {offset + 1}: only its prefix is shown. "
                                    "The remainder of this line is omitted; use cli_exec to read character "
                                    "ranges or increase max_tool_output."
                                )
                            break
                        result.append(formatted)
                        chars_used += line_chars
                        lines_read += 1
            except Exception as exc:
                return format_error(ErrorType.EXECUTION, f"Error reading file stream: {exc}")

            output = "\n".join(result) + long_line_warning
            end_index = offset + lines_read
            if total_lines > end_index:
                output += (
                    f"\n\n[TRUNCATED] Showing lines {offset + 1}-{end_index} of {total_lines} "
                    f"({stats.st_size} bytes total). "
                    f"To continue: read_file(path='{path}', offset={end_index}, "
                    f"limit={limit}, show_line_numbers={show_line_numbers})"
                )
            else:
                if offset == 0 and not show_line_numbers and end_index >= total_lines:
                    return self._truncate(output, source="read_file")
                output += f"\n\n[EOF] Lines {offset + 1}-{end_index} of {total_lines} ({stats.st_size} bytes)."
            return self._truncate(output, source="read_file")
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, f"Error reading file: {exc}")

    def write_file(self, path: str, content: str) -> str:
        try:
            target = self._resolve_path(path)
            if not isinstance(content, str):
                return format_error(ErrorType.VALIDATION, "content must be a string.")
            atomic_write_text(target, content)
            return f"Success: File '{path}' saved ({len(content)} chars)."
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, f"Error writing file: {exc}")

    def edit_file(self, path: str, old_string: str, new_string: str, *, replace_all: bool = False) -> str:
        try:
            return self._truncate(edit_text_file(self._resolve_path(path), path, old_string, new_string, replace_all=replace_all))
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
