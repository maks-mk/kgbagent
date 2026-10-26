import difflib
import json
import re
from pathlib import Path

from core.errors import ErrorType, format_error

from .pathing import atomic_write_text

_READ_FILE_LINE_NUMBER_RE = re.compile(r"^( *[1-9][0-9]*) {2}(.*)$")


def _strip_read_file_line_numbers(text: str) -> tuple[str, bool]:
    """Accept only complete, consecutive prefixes in read_file's exact format."""
    lines = text.split("\n")
    previous_number = None
    stripped_lines: list[str] = []
    for index, line in enumerate(lines):
        if not line and index == len(lines) - 1:
            stripped_lines.append(line)
            continue
        match = _READ_FILE_LINE_NUMBER_RE.fullmatch(line)
        if not match:
            return text, False
        number = int(match.group(1))
        if match.group(1) != f"{number:6d}" or (previous_number is not None and number != previous_number + 1):
            return text, False
        previous_number = number
        stripped_lines.append(match.group(2))
    return "\n".join(stripped_lines), previous_number is not None


def _exact_text_pattern(text: str) -> re.Pattern[str]:
    return re.compile(r"(?:\r\n|(?<!\r)\n)".join(re.escape(line) for line in text.split("\n")))


def _match_failure_hint(content: str, old_text: str) -> str:
    """Locate a possible block for diagnostics only; never use it to edit text."""
    guidance = (
        "Read the current file with read_file(show_line_numbers=false); copy only file text, "
        "not headers/footers. Preserve whitespace and do not reuse stale or truncated text."
    )
    file_lines = content.replace("\r\n", "\n").split("\n")
    search_lines = old_text.split("\n")
    # Index all non-blank search anchors in one file pass. The longest line may
    # itself be stale; another unchanged line can still locate useful context.
    # Whitespace is ignored for diagnostics only, never for applying an edit.
    whitespace = str.maketrans("", "", " \t")
    anchors: dict[str, tuple[int, list[int]]] = {}
    for index, line in enumerate(search_lines):
        anchor = line.translate(whitespace)
        if anchor:
            anchors.setdefault(anchor, (index, []))
    for index, line in enumerate(file_lines):
        candidate = anchors.get(line.translate(whitespace))
        if candidate is not None:
            anchor_index, locations = candidate
            if index >= anchor_index and len(locations) < 3:
                locations.append(index - anchor_index)
    available = [candidate for anchor, candidate in anchors.items() if candidate[1]]
    if not available:
        return guidance
    _, starts = max(available, key=lambda candidate: len(search_lines[candidate[0]].strip()))
    if len(starts) > 1:
        locations = ", ".join(str(start + 1) for start in starts)
        return f"Possible target blocks start at lines {locations} (up to 3 shown); no block was selected. {guidance}"

    start = starts[0]
    hint = f"Possible target starts at line {start + 1}. "
    for index, expected in enumerate(search_lines):
        actual = file_lines[start + index] if start + index < len(file_lines) else None
        if expected != actual:
            column = 0
            if actual is not None:
                while column < min(len(expected), len(actual)) and expected[column] == actual[column]:
                    column += 1
            preview_start = max(0, column - 60)

            def preview(line: str | None) -> str:
                if line is None:
                    return "<EOF>"
                return (
                    ("..." if preview_start else "") + repr(line[preview_start:preview_start + 160])
                    + ("..." if len(line) > preview_start + 160 else "")
                )

            def character(line: str | None) -> str:
                if line is None:
                    return "<EOF>"
                if column >= len(line):
                    return "<end of line>"
                return f"{ascii(line[column])} (U+{ord(line[column]):04X})"

            hint += (
                f"First differing line {start + index + 1}: old_string={preview(expected)}; file={preview(actual)}. "
                f"First differing column {column + 1}: old_string={character(expected)}; file={character(actual)}. "
            )
            if actual is None:
                hint += "The requested block extends beyond EOF (possibly a missing final newline). "
            elif expected.translate(whitespace) == actual.translate(whitespace):
                hint += "Spaces/tabs differ; leading, internal and trailing whitespace remain significant. "
            break
    return hint + guidance


def _realign_indentation(
    file_lines: list[str],
    match_idx: int,
    search_len: int,
    new_lines: list[str],
) -> list[str]:
    """Shift a replacement's base indentation to the matched block's base.

    Used only in explicitly requested indentation mode. Relative indentation,
    internal whitespace, and trailing whitespace are not normalized.
    """
    # Base indentation = indentation of the first non-empty line in the matched file block.
    file_base = ""
    for line in file_lines[match_idx : match_idx + search_len]:
        if line.strip():
            file_base = line[: len(line) - len(line.lstrip(" \t"))]
            break

    # Base indentation = indentation of the first non-empty line in new_lines.
    new_base = ""
    for line in new_lines:
        if line.strip():
            new_base = line[: len(line) - len(line.lstrip(" \t"))]
            break

    # Nothing to adjust.
    if file_base == new_base:
        return new_lines

    result: list[str] = []
    for line in new_lines:
        if not line.strip():
            # Preserve blank / whitespace-only lines as-is.
            result.append(line)
            continue

        current_indent = line[: len(line) - len(line.lstrip(" \t"))]
        content = line.lstrip(" \t")

        if current_indent.startswith(new_base):
            # Strip the model's base indent and prepend the file's base indent,
            # keeping any extra indentation (e.g. nested blocks) intact.
            extra = current_indent[len(new_base):]
            result.append(file_base + extra + content)
        elif new_base.startswith(current_indent):
            # A dedent must remove the same whitespace suffix from the file base.
            removed_indent = new_base[len(current_indent):]
            if not file_base.endswith(removed_indent):
                raise ValueError("Cannot preserve relative indentation in this block.")
            result.append(file_base[:-len(removed_indent)] + content)
        else:
            raise ValueError("Mixed indentation cannot be realigned safely.")

    return result


def _apply_text_edit(content: str, old_string: str, new_string: str, *, replace_all: bool, match_mode: str) -> tuple[str, int, list[str]]:
    if not isinstance(old_string, str) or not old_string.strip():
        raise ValueError("old_string must be a non-empty string.")
    if not isinstance(new_string, str):
        raise ValueError("new_string must be a string (use an empty string to delete text).")
    if not isinstance(replace_all, bool):
        raise ValueError("replace_all must be a boolean.")
    if match_mode not in {"exact", "indentation"}:
        raise ValueError("match_mode must be 'exact' or 'indentation'.")

    old_text = old_string.replace("\r\n", "\n")
    new_text = new_string.replace("\r\n", "\n")
    warnings = []

    # Treat LF/CRLF as equivalent in the search, but preserve untouched bytes,
    # including mixed line endings and a UTF-8 BOM. Literal text takes priority:
    # numeric data must never be mistaken for read_file's presentation prefixes.
    pattern = _exact_text_pattern(old_text)
    if not pattern.search(content):
        stripped_text, stripped_old = _strip_read_file_line_numbers(old_text)
        if stripped_old:
            if not stripped_text.strip():
                raise ValueError("old_string is empty after removing line-number prefixes.")
            old_text = stripped_text
            # Only a numbered old_string gives us grounds to strip new_string.
            new_text, _ = _strip_read_file_line_numbers(new_text)
            pattern = _exact_text_pattern(old_text)
            warnings.append("read_file line-number prefixes were removed before applying the edit. Verify the diff carefully.")
    count = 0
    previous_end = 0
    for candidate in re.finditer(f"(?=({pattern.pattern}))", content):
        if candidate.start() < previous_end:
            raise ValueError("Exact matches overlap; provide a non-overlapping target block.")
        previous_end = candidate.end(1)
        count += 1
    if count:
        if count > 1 and not replace_all:
            raise ValueError(
                f"Found {count} identical occurrences of the exact target text. "
                "Provide more context to uniquely identify the block, or set replace_all=true."
            )

        def replacement(match):
            newline = "\r\n" if "\r\n" in match.group() else ("\r\n" if "\r\n" in content else "\n")
            return new_text.replace("\n", newline)

        return pattern.sub(replacement, content, count=0 if replace_all else 1), count, warnings

    if match_mode == "exact":
        raise ValueError(
            "Could not find an exact match for old_string. " + _match_failure_hint(content, old_text)
            + " Use match_mode='indentation' only for an intentional leading-indentation adjustment."
        )

    file_lines = content.replace("\r\n", "\n").split("\n")
    search_lines = old_text.split("\n")
    trailing_newline = old_text.endswith("\n")
    if trailing_newline:
        search_lines.pop()
    search_len = len(search_lines)
    matches = []
    for index in range(len(file_lines) - search_len + 1):
        block = file_lines[index:index + search_len]
        # No trimming of trailing or internal whitespace, and no independent
        # re-indentation of nested lines: only one base-indent shift is allowed.
        if [line.lstrip(" \t") for line in block] != [line.lstrip(" \t") for line in search_lines]:
            continue
        try:
            aligned_search = _realign_indentation(file_lines, index, search_len, search_lines)
        except ValueError:
            continue
        if aligned_search != block:
            continue
        if trailing_newline and index + search_len >= len(file_lines):
            continue
        matches.append(index)
    if not matches:
        raise ValueError(
            "Could not find a leading-indentation match for old_string. "
            "Internal/trailing whitespace and relative indentation must match exactly. "
            + _match_failure_hint(content, old_text)
        )
    if len(matches) > 1 and not replace_all:
        raise ValueError(f"Found {len(matches)} occurrences of the target block. Provide more context or set replace_all=true.")
    if any(right < left + search_len for left, right in zip(matches, matches[1:])):
        raise ValueError("Indentation matches overlap; provide a non-overlapping target block.")

    raw_lines = content.split("\n")
    starts = []
    cursor = 0
    for line in raw_lines:
        starts.append(cursor)
        cursor += len(line) + 1
    updated = content
    for index in reversed(matches):
        end_line = index + search_len - 1
        end = starts[end_line] + len(raw_lines[end_line])
        if trailing_newline:
            end += 1
        elif end_line + 1 < len(raw_lines) and raw_lines[end_line].endswith("\r"):
            end -= 1
        begin = starts[index]
        matched = content[begin:end]
        newline = "\r\n" if "\r\n" in matched else ("\r\n" if "\r\n" in content else "\n")
        replacement_lines = _realign_indentation(file_lines, index, search_len, new_text.split("\n"))
        updated = updated[:begin] + "\n".join(replacement_lines).replace("\n", newline) + updated[end:]
    warnings.append("Explicit indentation match used; replacement indentation was realigned. Verify the diff carefully.")
    return updated, len(matches), warnings


def edit_text_file(
    target: Path, path_label: str, old_string: str | None = None, new_string: str | None = None,
    *, replace_all: bool = False, match_mode: str = "exact", edits: list[dict] | None = None,
) -> str:
    """Apply ordered edits in memory, validate the final result, then write once."""
    try:
        if not target.exists():
            return format_error(ErrorType.NOT_FOUND, f"File '{path_label}' not found.")
        if not target.is_file():
            return format_error(ErrorType.VALIDATION, f"'{path_label}' is not a file.")
        if edits is not None:
            if old_string is not None or new_string is not None or replace_all or match_mode != "exact":
                raise ValueError("Do not mix edits with top-level old_string/new_string/replace_all/match_mode.")
            if not isinstance(edits, list) or not 1 <= len(edits) <= 20:
                raise ValueError("edits must contain 1-20 replacements for the same file.")
            operations = edits
        else:
            operations = [{"old_string": old_string, "new_string": new_string,
                           "replace_all": replace_all, "match_mode": match_mode}]
        for index, operation in enumerate(operations, 1):
            if not isinstance(operation, dict) or set(operation) - {"old_string", "new_string", "replace_all", "match_mode"}:
                raise ValueError(f"Edit {index}: only old_string, new_string, replace_all and match_mode are allowed.")

        original_bytes = target.read_bytes()
        try:
            content = original_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            return format_error(ErrorType.VALIDATION, f"File '{path_label}' is not valid UTF-8 (byte {exc.start}). No changes were written.")
        new_content = content
        replacements = 0
        warnings = []
        for index, operation in enumerate(operations, 1):
            try:
                new_content, count, edit_warnings = _apply_text_edit(
                    new_content, operation.get("old_string"), operation.get("new_string"),
                    replace_all=operation.get("replace_all", False), match_mode=operation.get("match_mode", "exact"),
                )
            except ValueError as exc:
                context = " Diagnostics refer to content after preceding edits in memory." if index > 1 else ""
                raise ValueError(f"Edit {index}: {exc}{context} No changes were written.") from exc
            replacements += count
            warnings.extend(warning for warning in edit_warnings if warning not in warnings)
        if new_content == content:
            return "No changes: replacement is identical to the existing text."
        if target.suffix.lower() == ".json":
            try:
                json.loads(new_content.removeprefix("\ufeff"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Edit would produce invalid JSON in '{path_label}' (line {exc.lineno}, column {exc.colno}): {exc.msg}. No changes were written.") from exc

        atomic_write_text(target, new_content, expected_content=original_bytes)
        diff = difflib.unified_diff(
            content.splitlines(), new_content.splitlines(), fromfile=f"a/{path_label}", tofile=f"b/{path_label}", lineterm="",
        )
        warning_text = "".join(f"\n\nWarning: {warning}" for warning in warnings)
        diff_text = "\n".join(diff)
        return f"Success: File edited. ({replacements} replacement(s)){warning_text}\n\nDiff:\n```diff\n{diff_text}\n```"
    except ValueError as exc:
        return format_error(ErrorType.VALIDATION, str(exc))
    except Exception as exc:
        return format_error(ErrorType.EXECUTION, f"Error editing file: {exc}")
