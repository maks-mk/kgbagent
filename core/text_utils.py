import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from langchain_core.messages import AIMessage, AIMessageChunk

_CLEAN_MD_RE = re.compile(r"\n{3,}")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_SIMPLE_LATEX_INLINE_RE = re.compile(r"\$\s*(\\[A-Za-z]+)\s*\$")
_MARKDOWN_FENCE_RE = re.compile(r"^(?P<indent>[ \t]{0,3})(?P<fence>`{3,}|~{3,})(?P<info>[^`~\r\n]*)[ \t]*(?:\r?\n)?$")
_INVISIBLE_TEXT_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})


def has_visible_text(value: Any) -> bool:
    """Return whether text contains a character that can produce visible output."""
    return any(
        not char.isspace() and unicodedata.category(char) not in _INVISIBLE_TEXT_CATEGORIES
        for char in str(value or "")
    )


def format_elapsed_seconds(seconds: int) -> str:
    whole_seconds = max(0, int(seconds))
    if whole_seconds <= 60:
        return f"{whole_seconds}s"
    minutes, remaining_seconds = divmod(whole_seconds, 60)
    return f"{minutes}m {remaining_seconds}s"


def format_compact_tokens(tokens: int) -> str:
    value = max(0, int(tokens))
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if value >= threshold:
            compact = value / threshold
            rendered = f"{compact:.1f}" if compact < 100 else f"{compact:.0f}"
            if "." in rendered:
                rendered = rendered.rstrip("0").rstrip(".")
            return f"{rendered}{suffix}"
    return str(value)


_CACHE_HIT_EXACT_PATHS = (
    ("usage", "prompt_tokens_details", "cached_tokens"),
    ("usage", "input_tokens_details", "cached_tokens"),
    ("usage", "cached_tokens"),
    ("usage", "prompt_cache_hit_tokens"),
    ("usage", "cache_read_input_tokens"),
    ("usageMetadata", "cachedContentTokenCount"),
    ("usage", "cached_prompt_text_tokens"),
    ("prompt_tokens_details", "cached_tokens"),
    ("input_tokens_details", "cached_tokens"),
    ("cached_tokens",),
    ("prompt_cache_hit_tokens",),
    ("cache_read_input_tokens",),
    ("cachedContentTokenCount",),
    ("cached_prompt_text_tokens",),
)
# LangChain reports cache reads in ``usage_metadata["input_token_details"]``, where
# the key carries a service-tier prefix for non-default tiers (OpenAI's "priority"
# and "flex" tiers yield ``priority_cache_read`` / ``flex_cache_read``).
_CACHE_HIT_DETAIL_PATHS = (("input_token_details",),)
_CACHE_HIT_DETAIL_KEY = "cache_read"
_CACHE_HIT_FALLBACK_NAMES = frozenset(
    {
        "cached_tokens",
        "prompt_cache_hit_tokens",
        "cache_read_input_tokens",
        "cachedContentTokenCount",
        "cached_prompt_text_tokens",
    }
)


def _reported_token_count(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value)) if value.is_integer() else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _mapping_value(value: Any, key: str) -> tuple[bool, Any]:
    if isinstance(value, dict):
        return (key in value, value.get(key))
    try:
        return (hasattr(value, key), getattr(value, key, None))
    except Exception:
        return (False, None)


def _resolve_path(payload: Any, path: tuple[str, ...]) -> tuple[bool, Any]:
    current = payload
    for key in path:
        found, current = _mapping_value(current, key)
        if not found:
            return (False, None)
    return (True, current)


def _cache_hit_from_details(details: Any) -> int | None:
    found, value = _mapping_value(details, _CACHE_HIT_DETAIL_KEY)
    if found:
        reported = _reported_token_count(value)
        if reported is not None:
            return reported
    if not isinstance(details, dict):
        return None
    for key, value in details.items():
        if isinstance(key, str) and key.endswith(f"_{_CACHE_HIT_DETAIL_KEY}"):
            reported = _reported_token_count(value)
            if reported is not None:
                return reported
    return None


def extract_cache_hit_tokens(payload: Any) -> int | None:
    if payload is None:
        return None

    for path in _CACHE_HIT_EXACT_PATHS:
        found, current = _resolve_path(payload, path)
        if found:
            reported = _reported_token_count(current)
            if reported is not None:
                return reported

    for path in _CACHE_HIT_DETAIL_PATHS:
        found, current = _resolve_path(payload, path)
        if found:
            reported = _cache_hit_from_details(current)
            if reported is not None:
                return reported

    visited: set[int] = set()

    def _fallback(value: Any) -> int | None:
        if isinstance(value, dict):
            identity = id(value)
            if identity in visited:
                return None
            visited.add(identity)
            for key, nested in value.items():
                if key in _CACHE_HIT_FALLBACK_NAMES:
                    reported = _reported_token_count(nested)
                    if reported is not None:
                        return reported
            for nested in value.values():
                reported = _fallback(nested)
                if reported is not None:
                    return reported
        elif isinstance(value, (list, tuple)):
            for nested in value:
                reported = _fallback(nested)
                if reported is not None:
                    return reported
        return None

    return _fallback(payload)


@dataclass(frozen=True)
class MarkdownSegment:
    kind: str
    text: str
    raw: str
    language: str = ""
    closed: bool = True


def split_markdown_segments(text: str) -> list[MarkdownSegment]:
    if text == "":
        return [MarkdownSegment("markdown", "", "")]

    segments: list[MarkdownSegment] = []
    markdown_lines: list[str] = []
    code_lines: list[str] = []
    open_fence = ""
    fence_marker = ""
    language = ""
    in_fence = False

    def _flush_markdown() -> None:
        markdown_text = "".join(markdown_lines)
        if markdown_text:
            segments.append(MarkdownSegment("markdown", markdown_text, markdown_text))
        markdown_lines.clear()

    def _fence_match(line: str) -> re.Match | None:
        return _MARKDOWN_FENCE_RE.match(line)

    def _is_closing_fence(line: str) -> bool:
        match = _fence_match(line)
        if not match:
            return False
        marker = match.group("fence")
        return bool(
            fence_marker
            and not match.group("info").strip()
            and marker[0] == fence_marker[0]
            and len(marker) >= len(fence_marker)
        )

    for line in text.splitlines(keepends=True):
        fence_match = _fence_match(line)
        if fence_match:
            if in_fence:
                if _is_closing_fence(line):
                    code_text = "".join(code_lines)
                    segments.append(
                        MarkdownSegment(
                            "code",
                            code_text,
                            f"{open_fence}{code_text}{line}",
                            language=language,
                            closed=True,
                        )
                    )
                    code_lines.clear()
                    open_fence = ""
                    fence_marker = ""
                    language = ""
                    in_fence = False
                else:
                    code_lines.append(line)
            else:
                _flush_markdown()
                open_fence = line
                fence_marker = fence_match.group("fence")
                language = fence_match.group("info").strip()
                in_fence = True
            continue

        if in_fence:
            code_lines.append(line)
        else:
            markdown_lines.append(line)

    if in_fence:
        code_text = "".join(code_lines)
        segments.append(
            MarkdownSegment(
                "code",
                code_text,
                f"{open_fence}{code_text}",
                language=language,
                closed=False,
            )
        )
    else:
        _flush_markdown()

    return segments or [MarkdownSegment("markdown", "", "")]

# Filename detection for assistant-answer highlighting. Matches:
#   - explicit paths with a known extension: src/ui/theme.py, ./docs/README.md
#   - bare names with a known extension: theme.py, README.md
#   - Windows/UNC paths: core\text_utils.py, D:\project\main.py
# Spaces are deliberately excluded from name characters: allowing them lets a
# sentence like "see the docs. The file main.py" match as one giant token.
_FILENAME_TOKEN_RE = re.compile(
    r"(?<![\w./\\-])"
    r"(?:[A-Za-z]:)?(?:[\w.\-]*[/\\])*"
    r"[\w.\-]+\.(?:py|pyw|js|mjs|cjs|ts|tsx|jsx|json|jsonl|md|markdown|txt|rst|"
    r"yml|yaml|toml|ini|cfg|conf|env|bat|ps1|psm1|sh|bash|zsh|fish|"
    r"c|h|cpp|hpp|cc|cxx|cs|go|rs|java|kt|kts|rb|php|swift|m|mm|"
    r"html|htm|css|scss|sass|less|vue|svelte|"
    r"xml|sql|graphql|gql|proto|tf|tfvars|hcl|"
    r"png|jpg|jpeg|gif|bmp|webp|svg|ico|tiff|pdf|zip|gz|tgz|tar|7z|rar|exe|dll|so|dylib|"
    r"csv|tsv|xlsx|xls|docx|doc|pptx|ppt|ipynb|lock|log)"
    r"(?:_[A-Za-z0-9]+)*(?!\w)(?!\.[A-Za-z0-9])"
)
# Extension-less filenames are too ambiguous in prose; only well-known project
# files are highlighted.
_BARE_FILENAME_RE = re.compile(
    r"(?<![\w./\\-])(?:Dockerfile|Makefile|LICENSE|CHANGELOG|README)(?!\w)(?!\.[A-Za-z0-9])"
)


def find_filename_spans(text: str) -> list[tuple[int, int]]:
    """Return merged (start, end) spans of filename-like tokens in plain text.

    Used by the UI to tint filenames in assistant answers. Overlapping matches
    from both patterns are merged so a token is never highlighted twice.
    """
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    for regex in (_FILENAME_TOKEN_RE, _BARE_FILENAME_RE):
        for match in regex.finditer(text):
            start, end = match.span()
            if end > start:
                spans.append((start, end))
    if not spans:
        return []
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
            continue
        merged.append((start, end))
    return merged

# Matches Markdown links whose href is a local file path (not http/https/ftp/mailto).
# Rich URL-encodes non-ASCII hrefs, which breaks Cyrillic filenames in output.
# Capture groups: 1=link text, 2=href
_LOCAL_LINK_RE = re.compile(
    r"\[([^\]]+)\]\((?!https?://|ftp://|mailto:)([^)]+)\)",
    re.IGNORECASE,
)
_SIMPLE_LATEX_SYMBOLS = {
    r"\to": "→",
    r"\rightarrow": "→",
    r"\gets": "←",
    r"\leftarrow": "←",
    r"\leftrightarrow": "↔",
    r"\Rightarrow": "⇒",
    r"\Leftarrow": "⇐",
    r"\Leftrightarrow": "⇔",
    r"\uparrow": "↑",
    r"\downarrow": "↓",
    r"\ge": "≥",
    r"\geq": "≥",
    r"\le": "≤",
    r"\leq": "≤",
    r"\neq": "≠",
    r"\pm": "±",
    r"\times": "×",
    r"\cdot": "·",
    r"\approx": "≈",
    r"\infty": "∞",
}


def _rewrite_outside_code(text: str, replacer: Callable[[str], str]) -> str:
    parts: list[str] = []
    for segment in split_markdown_segments(text):
        if segment.kind == "code":
            parts.append(segment.raw)
        else:
            parts.append(_rewrite_outside_inline_code(segment.text, replacer))
    return "".join(parts)


def _rewrite_outside_inline_code(text: str, replacer: Callable[[str], str]) -> str:
    parts: list[str] = []
    last = 0
    for block in _INLINE_CODE_RE.finditer(text):
        parts.append(replacer(text[last:block.start()]))
        parts.append(block.group(0))
        last = block.end()
    parts.append(replacer(text[last:]))
    return "".join(parts)


def _normalize_simple_latex_inline(text: str) -> str:
    def _replace_segment(segment: str) -> str:
        def _replace_match(match: re.Match) -> str:
            command = str(match.group(1) or "").strip()
            return _SIMPLE_LATEX_SYMBOLS.get(command, match.group(0))

        return _SIMPLE_LATEX_INLINE_RE.sub(_replace_match, segment)

    return _rewrite_outside_code(text, _replace_segment)


_ESCAPED_MARKDOWN_MARKERS_RE = re.compile(r"\\([\*_`#>!|\[\]\(\){}+\-.])")


def _unescape_common_markdown_markers(text: str) -> str:
    """Undo provider-overescaped Markdown markers outside code spans/blocks."""
    return _rewrite_outside_code(text, lambda segment: _ESCAPED_MARKDOWN_MARKERS_RE.sub(r"\1", segment))


def _rewrite_local_file_links(text: str) -> str:
    """Convert Markdown local-file links to inline code to prevent Rich URL-encoding.

    [filename.md](filename.md)          →  `filename.md`
    [label](path/to/file.py)  →  `path/to/file.py` (label)
    """
    def _replace(m: re.Match) -> str:
        label: str = m.group(1).strip()
        href: str = m.group(2).strip()
        # If the href itself is the label (auto-link), just emit inline code
        if label == href or label.lower() == href.lower():
            return f"`{href}`"
        # Otherwise emit: `href` (label)
        return f"`{href}` ({label})"

    return _rewrite_outside_code(text, lambda segment: _LOCAL_LINK_RE.sub(_replace, segment))

def truncate_value(value: str, max_length: int = 60) -> str:
    if len(value) > max_length:
        return value[:max_length] + "..."
    return value


def _single_line_preview(value: Any) -> str:
    text = str(value or "")
    # Tool card headers should stay one-line even when command/query contains newlines.
    return " ".join(text.split())


def _first_non_empty_item(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return next((str(item) for item in value if str(item).strip()), "")
    return str(value or "")


def abbreviate_path(path_str: str, max_length: int = 60) -> str:
    try:
        path = Path(path_str)
        if len(path.parts) == 1:
            return path_str

        try:
            rel_str = str(path.relative_to(Path.cwd()))
            if len(rel_str) < len(path_str) and len(rel_str) <= max_length:
                return rel_str
        except (ValueError, OSError):
            pass

        if len(path_str) <= max_length:
            return path_str
    except Exception:
        pass

    return truncate_value(path_str, max_length)


def _format_path_tool(tool_name: str, tool_args: Dict[str, Any]) -> str | None:
    path_value = tool_args.get("file_path") or tool_args.get("path") or tool_args.get("dir_path")
    if path_value:
        return f"{tool_name}({abbreviate_path(str(path_value))})"
    return None


def _format_query_tool(tool_name: str, tool_args: Dict[str, Any]) -> str | None:
    query = tool_args.get("queries") if tool_name == "batch_web_search" else tool_args.get("query")
    query_text = _first_non_empty_item(query)
    if query_text:
        return f'{tool_name}("{truncate_value(_single_line_preview(query_text), 80)}")'
    return None


def _format_pattern_tool(tool_name: str, tool_args: Dict[str, Any]) -> str | None:
    pattern_val = tool_args.get("pattern") or tool_args.get("name_pattern")
    if pattern_val is not None:
        return f'{tool_name}("{truncate_value(_single_line_preview(pattern_val), 70)}")'
    return None


def _format_command_tool(tool_name: str, tool_args: Dict[str, Any]) -> str | None:
    command = tool_args.get("command")
    if command is not None:
        return f'{tool_name}("{truncate_value(_single_line_preview(command), 100)}")'
    return None


def _format_list_tool(tool_name: str, tool_args: Dict[str, Any]) -> str | None:
    path = tool_args.get("path")
    return f"{tool_name}({abbreviate_path(str(path))})" if path else f"{tool_name}()"


def _format_url_tool(tool_name: str, tool_args: Dict[str, Any]) -> str | None:
    url_val = tool_args.get("url") or tool_args.get("urls")
    url_text = _first_non_empty_item(url_val)
    if url_text:
        return f'{tool_name}("{truncate_value(_single_line_preview(url_text), 80)}")'
    return None


DISPLAY_RULES: tuple[tuple[set[str], Callable[[str, Dict[str, Any]], str | None]], ...] = (
    (
        {
            "read_file",
            "write_file",
            "edit_file",
            "safe_delete_file",
            "safe_delete_directory",
            "Read",
            "Write",
            "SearchReplace",
        },
        _format_path_tool,
    ),
    ({"batch_web_search"}, _format_query_tool),
    ({"grep", "Grep", "glob", "Glob"}, _format_pattern_tool),
    ({"execute", "RunCommand", "cli_exec"}, _format_command_tool),
    ({"ls", "LS", "list_directory"}, _format_list_tool),
    ({"fetch_url", "WebFetch", "fetch_content", "crawl_site", "download_file"}, _format_url_tool),
)


def format_tool_display(tool_name: str, tool_args: Dict[str, Any]) -> str:
    for names, formatter in DISPLAY_RULES:
        if tool_name in names:
            formatted = formatter(tool_name, tool_args)
            if formatted:
                return formatted
            break

    args_str = ", ".join(f"{k}={truncate_value(_single_line_preview(v), 50)}" for k, v in tool_args.items())
    return f"{tool_name}({args_str})"


def classify_tool_args_state(tool_name: str, tool_args: Dict[str, Any]) -> str:
    args = dict(tool_args or {})
    if not args:
        return "pending"

    anchor_keys: tuple[str, ...] = ()
    if tool_name in {
        "read_file",
        "write_file",
        "edit_file",
        "safe_delete_file",
        "safe_delete_directory",
        "Read",
        "Write",
        "SearchReplace",
    }:
        anchor_keys = ("path", "file_path", "dir_path")
    elif tool_name == "batch_web_search":
        anchor_keys = ("queries",)
    elif tool_name in {"grep", "Grep", "glob", "Glob"}:
        anchor_keys = ("pattern", "name_pattern")
    elif tool_name in {"execute", "RunCommand", "cli_exec"}:
        anchor_keys = ("command",)
    elif tool_name in {"fetch_url", "WebFetch", "fetch_content", "crawl_site", "download_file"}:
        anchor_keys = ("url", "urls")

    if anchor_keys and not any(args.get(key) for key in anchor_keys):
        return "partial"
    return "complete"


def tool_source_kind(tool_name: str) -> str:
    normalized = str(tool_name or "").strip().lower()
    if normalized == "cli_exec":
        return "cli"
    if ":" in normalized:
        return "mcp"
    return "tool"


def _humanize_tool_name(tool_name: str) -> str:
    words = str(tool_name or "").replace(":", " ").replace("-", " ").replace("_", " ").strip().split()
    return " ".join(word.upper() if word.casefold() == "id" else word.capitalize() for word in words) or "Tool"


def _mcp_target_summary(tool_args: Dict[str, Any]) -> str:
    preferred_keys = (
        "query", "topic", "question", "library_name", "libraryName", "library_id", "libraryId",
        "path", "file_path", "url", "uri", "name", "id",
    )
    for key in preferred_keys:
        value = tool_args.get(key)
        if value not in (None, "", [], {}):
            return truncate_value(_single_line_preview(value), 100)
    for value in tool_args.values():
        if value not in (None, "", [], {}):
            return truncate_value(_single_line_preview(value), 100)
    return ""


def build_mcp_tool_ui_labels(
    tool_name: str,
    tool_args: Dict[str, Any],
    *,
    phase: str = "running",
    is_error: bool = False,
    server_name: str = "",
) -> Dict[str, str]:
    human_name = _humanize_tool_name(tool_name)
    server = _humanize_tool_name(server_name) if server_name else "MCP"
    if is_error:
        title = f"{server}: {human_name} failed"
    elif phase == "finished":
        title = f"{server}: {human_name} completed"
    else:
        title = f"{server}: {human_name}"
    args = dict(tool_args or {})
    return {
        "title": title,
        "subtitle": _mcp_target_summary(args),
        "raw_display": format_tool_display(tool_name, args),
        "args_state": "complete" if args else "pending",
        "source_kind": "mcp",
    }


def tool_title_case(tool_name: str) -> str:
    words = str(tool_name or "").replace(":", " ").replace("_", " ").strip().split()
    if not words:
        return "Tool"
    return " ".join(word[:1].upper() + word[1:] for word in words)


def tool_target_summary(tool_name: str, tool_args: Dict[str, Any]) -> str:
    args = dict(tool_args or {})
    normalized_name = str(tool_name or "").strip()

    if normalized_name in {
        "read_file",
        "write_file",
        "edit_file",
        "safe_delete_file",
        "safe_delete_directory",
        "Read",
        "Write",
        "SearchReplace",
    }:
        path_value = args.get("file_path") or args.get("path") or args.get("dir_path")
        return abbreviate_path(str(path_value)) if path_value else ""
    if normalized_name in {"ls", "LS", "list_directory"}:
        path_value = args.get("path")
        return abbreviate_path(str(path_value)) if path_value else "current directory"
    if normalized_name == "batch_web_search":
        query = _first_non_empty_item(args.get("queries"))
        return truncate_value(_single_line_preview(query), 80) if query else ""
    if normalized_name in {"grep", "Grep", "glob", "Glob"}:
        pattern_val = args.get("pattern") or args.get("name_pattern")
        return truncate_value(_single_line_preview(pattern_val), 70) if pattern_val else ""
    if normalized_name in {"execute", "RunCommand", "cli_exec"}:
        command = args.get("command")
        return truncate_value(_single_line_preview(command), 100) if command else ""
    if normalized_name in {"fetch_url", "WebFetch", "fetch_content", "crawl_site", "download_file"}:
        url_text = _first_non_empty_item(args.get("url") or args.get("urls"))
        return truncate_value(_single_line_preview(url_text), 80) if url_text else ""
    return ""


def build_tool_ui_labels(
    tool_name: str,
    tool_args: Dict[str, Any],
    *,
    phase: str = "running",
    is_error: bool = False,
) -> Dict[str, str]:
    normalized_name = str(tool_name or "").strip() or "unknown_tool"
    args = dict(tool_args or {})
    args_state = classify_tool_args_state(normalized_name, args)
    target = tool_target_summary(normalized_name, args)

    action_map = {
        "read_file": "Reading file",
        "write_file": "Writing file",
        "edit_file": "Editing file",
        "safe_delete_file": "Deleting file",
        "safe_delete_directory": "Deleting directory",
        "Read": "Reading file",
        "Write": "Writing file",
        "SearchReplace": "Editing file",
        "ls": "Listing directory",
        "LS": "Listing directory",
        "list_directory": "Listing directory",
        "batch_web_search": "Searching web",
        "grep": "Searching files",
        "Grep": "Searching files",
        "glob": "Finding files",
        "Glob": "Finding files",
        "fetch_url": "Fetching URL",
        "WebFetch": "Fetching URL",
        "fetch_content": "Fetching content",
        "crawl_site": "Crawling site",
        "download_file": "Downloading file",
        "execute": "Running command",
        "RunCommand": "Running command",
        "cli_exec": "Running command",
    }
    preparing_map = {
        "read_file": "Preparing file read",
        "write_file": "Preparing file write",
        "edit_file": "Preparing edit",
        "safe_delete_file": "Preparing file deletion",
        "safe_delete_directory": "Preparing directory deletion",
        "Read": "Preparing file read",
        "Write": "Preparing file write",
        "SearchReplace": "Preparing edit",
        "ls": "Preparing directory listing",
        "LS": "Preparing directory listing",
        "list_directory": "Preparing directory listing",
        "batch_web_search": "Preparing search",
        "grep": "Preparing search",
        "Grep": "Preparing search",
        "glob": "Preparing file search",
        "Glob": "Preparing file search",
        "fetch_url": "Preparing fetch",
        "WebFetch": "Preparing fetch",
        "fetch_content": "Preparing fetch",
        "crawl_site": "Preparing crawl",
        "download_file": "Preparing download",
        "execute": "Preparing command",
        "RunCommand": "Preparing command",
        "cli_exec": "Preparing command",
    }
    base_title = action_map.get(normalized_name, tool_title_case(normalized_name))

    if is_error:
        title = f"{base_title} failed"
    elif phase == "finished":
        title = base_title
    elif phase == "preparing":
        title = preparing_map.get(normalized_name, f"Preparing {base_title.lower()}")
    else:
        title = base_title

    if args_state == "pending":
        subtitle = "Waiting for arguments…"
    elif args_state == "partial":
        subtitle = target or "Resolving arguments…"
    else:
        subtitle = target

    raw_display = format_tool_display(normalized_name, args) if args_state == "complete" else normalized_name
    return {
        "title": title,
        "subtitle": subtitle,
        "raw_display": raw_display,
        "args_state": args_state,
        "source_kind": tool_source_kind(normalized_name),
    }


def _collapse_non_code_markdown(text: str) -> str:
    parts: list[str] = []
    for segment in split_markdown_segments(text):
        if segment.kind == "code":
            parts.append(segment.raw)
        else:
            parts.append(_CLEAN_MD_RE.sub("\n\n", segment.text))
    return "".join(parts)


def clean_markdown_text(text: str) -> str:
    if not text:
        return text
    return _collapse_non_code_markdown(text)


def prepare_markdown_for_render(text: str) -> str:
    text = _normalize_simple_latex_inline(text)
    text = _unescape_common_markdown_markers(text)
    text = _rewrite_local_file_links(text)
    # Do not infer fenced code while text is streaming. The inference can change
    # its mind as a line grows, which makes ordinary prose flash as a code block.
    # Explicit Markdown fences remain fully supported by the renderer.
    return clean_markdown_text(text)


def _hint_for_error(content: str) -> str:
    lower_content = content.lower()
    if "401" in lower_content or "unauthorized" in lower_content:
        return " (Hint: Check your API keys in .env)"
    if "not found" in lower_content and ("file" in lower_content or "dir" in lower_content):
        return " (Hint: Check path relative to workspace)"
    if "disabled" in lower_content:
        return " (Hint: Check .env configuration)"
    if "connection" in lower_content or "timeout" in lower_content:
        return " (Hint: Network issue, try again)"
    return ""


def _format_search_output(content: str) -> str:
    count = content.count("http")
    return f"Found {count} results" if count > 0 else "No results found"


def _format_cli_output(content: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return "Command executed (no output)"

    first_line = lines[0].replace("[stderr]", "").strip()
    preview = truncate_value(first_line, 60)
    if len(lines) > 1:
        return f"{preview} (+{len(lines) - 1} lines)"
    return preview


def _format_list_output(content: str) -> str:
    lines = content.splitlines()
    count = len(lines)
    preview = ", ".join(line.strip() for line in lines[:3])
    if count > 3:
        return f"Listed {count} items: {preview}, ..."
    return f"Listed {count} items: {preview}"


OUTPUT_RULES: tuple[tuple[Callable[[str], bool], Callable[[str], str]], ...] = (
    (lambda name: name == "batch_web_search", _format_search_output),
    (lambda name: "cli_exec" in name or "shell" in name, _format_cli_output),
    (lambda name: "list" in name and "directory" in name, _format_list_output),
    (lambda name: "read" in name, lambda content: f"Read {len(content.splitlines())} lines ({len(content)} chars)"),
    (lambda name: "write" in name or "save" in name, lambda content: "File saved successfully"),
    (lambda name: "edit_file" in name, lambda content: "File edited successfully"),
    (lambda name: "delete" in name, lambda content: "Deleted successfully"),
    (lambda name: "fetch" in name or "download" in name, lambda content: f"Fetched content ({len(content)} chars)"),
)


def format_tool_output(name: str, content: str, is_error: bool) -> str:
    content = str(content).strip()

    if is_error:
        lower_content = content.lower()
        if "error[access_denied]" in lower_content or "cancelled by approval policy" in lower_content:
            return "Skipped"
        summary = truncate_value(content, 120)
        return f"{summary}{_hint_for_error(content)}"

    name_lower = name.lower()
    for predicate, formatter in OUTPUT_RULES:
        if predicate(name_lower):
            return formatter(content)

    return truncate_value(content, 150)


def format_exception_friendly(e: Exception) -> str:
    err_str = str(e)
    err_type = type(e).__name__

    if "429" in err_str or "RateLimit" in err_type or "QuotaExceeded" in err_type or "ResourceExhausted" in err_type:
        return "Rate Limit Exceeded (429). Please wait a moment or check your API quota."

    if "401" in err_str or "403" in err_str or "Authentication" in err_type:
        return "Authentication Failed. Check your API KEY in .env."

    if "402" in err_str or "insufficient_balance" in err_str or "Insufficient account balance" in err_str:
        return "Insufficient account balance (402). Top up the provider account or switch model/provider."

    if "context_length_exceeded" in err_str or "too many tokens" in err_str.lower():
        return "Context Limit Reached. Use 'reset' to start fresh."

    if "ConnectError" in err_type or "Timeout" in err_type or "ReadTimeout" in err_type:
        return "Network Error. Connection failed or timed out."

    if len(err_str) > 300:
        return f"Error ({err_type}): {err_str[:300]}...[truncated]"

    return f"Error ({err_type}): {err_str}"


class TokenTracker:
    __slots__ = (
        "_streaming_len",
        "_step_usage",
        "_unkeyed_step_index",
        "_active_step_has_data",
        "_aliases",
        "_active_response_id",
    )

    def __init__(self):
        self._streaming_len = 0
        self._step_usage: dict[str, tuple[int, int, int]] = {}
        self._unkeyed_step_index = 0
        self._active_step_has_data = False
        # alias key (e.g. ``lc_run--...``) -> canonical key (e.g. ``resp_...``)
        self._aliases: dict[str, str] = {}
        self._active_response_id: str | None = None

    @property
    def total_input(self) -> int:
        return sum(in_t for in_t, _, _ in self._step_usage.values())

    @property
    def total_output(self) -> int:
        return sum(out_t for _, out_t, _ in self._step_usage.values())

    @property
    def total_cache_hit(self) -> int:
        return sum(cache_hit for _, _, cache_hit in self._step_usage.values())

    def advance_step(self):
        if self._active_step_has_data:
            self._unkeyed_step_index += 1
            self._active_step_has_data = False
        # A new stream segment starts a new provider response; stale response
        # IDs must not capture unrelated ``lc_run--`` keys of later invocations.
        self._active_response_id = None

    def update_from_message(self, msg: Any) -> int:
        if isinstance(msg, (AIMessage, AIMessageChunk)):
            content = msg.content
            chunk_len = 0
            if isinstance(content, str):
                chunk_len = len(content)
            elif isinstance(content, list):
                chunk_len = sum(len(x.get("text", "")) for x in content if isinstance(x, dict))

            if isinstance(msg, AIMessageChunk):
                self._streaming_len += chunk_len
            elif self._streaming_len == 0:
                self._streaming_len = chunk_len

        msg_id = getattr(msg, "id", None)
        # Streamed chunks report additive usage: LangChain merges them with
        # ``add_usage``, so providers that send already-accumulated counts (Gemini)
        # publish a per-chunk delta. Complete messages and node updates instead
        # report the cumulative usage of the whole response.
        is_delta = isinstance(msg, AIMessageChunk)
        self._register_response_alias(msg, is_delta=is_delta)
        usage_candidates = (
            getattr(msg, "usage_metadata", None),
            getattr(msg, "response_metadata", None),
            getattr(msg, "additional_kwargs", None),
        )
        for usage in usage_candidates:
            if isinstance(usage, dict) and usage:
                reported_cache_hit = extract_cache_hit_tokens(usage)
                delta = self._apply_metadata(
                    usage,
                    msg_id=msg_id,
                    source="message",
                    accumulate=is_delta,
                )
                if reported_cache_hit is not None:
                    return delta
                if is_delta and any(self._usage_token_counts(usage, allow_negative=True)):
                    # A chunk publishes its usage once; adding a second candidate
                    # of the same chunk would inflate the totals.
                    return 0
        return 0

    def update_from_node_update(self, update: Dict) -> int:
        if not isinstance(update, dict):
            return 0

        applied_any = False
        cache_hit_delta = 0
        for node_payload in update.values():
            if not isinstance(node_payload, dict):
                continue

            node_messages = node_payload.get("messages", [])
            if not isinstance(node_messages, list):
                node_messages = [node_messages]

            msg_ids = [
                str(getattr(m, "id", "")).strip()
                for m in node_messages
                if str(getattr(m, "id", "")).strip()
            ]
            # ``token_usage`` always belongs to the AI response a node appends last.
            # Keying it by an unrelated leading message (e.g. a RemoveMessage emitted
            # for internal-retry cleanup) creates a second key for usage already
            # tracked from the streamed chunks, double-counting cache hits and tokens.
            ai_msg_ids = [
                str(getattr(m, "id", "")).strip()
                for m in node_messages
                if isinstance(m, (AIMessage, AIMessageChunk)) and str(getattr(m, "id", "")).strip()
            ]
            primary_msg_id = ai_msg_ids[-1] if ai_msg_ids else (msg_ids[0] if msg_ids else None)

            usage = node_payload.get("token_usage")
            if isinstance(usage, dict) and usage:
                fallback_id = primary_msg_id or self._active_response_id
                cache_hit_delta += self._apply_metadata(usage, msg_id=fallback_id, source="update")
                applied_any = True

        if applied_any:
            self.advance_step()
        return cache_hit_delta

    @staticmethod
    def _coerce_token_int(value: Any, *, allow_negative: bool = False) -> int:
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return 0
        return coerced if allow_negative else max(0, coerced)

    def _register_response_alias(self, msg: Any, *, is_delta: bool) -> None:
        """Link a streamed chunk ID to the provider response it belongs to.

        The Responses API announces the canonical ``resp_...`` response ID on the
        first streamed chunk (``response_metadata["id"]``), while usage arrives on
        later chunks whose LangChain-assigned IDs are ``lc_run--...``. The final
        message and the node update are keyed by the ``resp_...`` ID again.
        Without linking, one provider call is counted under two keys and every
        token bucket (input/output/cache) doubles.
        """
        msg_id = getattr(msg, "id", None)
        msg_key = str(msg_id).strip() if msg_id else ""
        response_metadata = getattr(msg, "response_metadata", None)
        response_id = ""
        if isinstance(response_metadata, dict):
            raw_response_id = response_metadata.get("id")
            response_id = str(raw_response_id).strip() if raw_response_id else ""
        if response_id:
            if msg_key and msg_key != response_id:
                self._aliases[msg_key] = response_id
            if is_delta:
                # Only the currently streaming response may capture new chunk IDs;
                # replayed or unrelated messages must not rebind the active alias.
                self._active_response_id = response_id
            return
        if not is_delta or not msg_key or not self._active_response_id:
            return
        # Usage chunks of the Responses API carry the ``lc_run--...`` run ID and
        # no response ID in their metadata; bind them to the response announced
        # by the opening chunk. Only chunks that actually report usage create
        # tracker keys, so only those need the alias.
        usage_metadata = getattr(msg, "usage_metadata", None)
        if isinstance(usage_metadata, dict) and usage_metadata:
            self._aliases[msg_key] = self._active_response_id

    def _resolve_key(self, msg_key: str) -> str:
        seen: set[str] = set()
        key = msg_key
        while key in self._aliases and key not in seen:
            seen.add(key)
            key = self._aliases[key]
        return key

    @classmethod
    def _extract_output_tokens(cls, usage: Dict[str, Any], *, allow_negative: bool = False) -> int:
        for key in (
            "output_tokens",
            "completion_tokens",
            "completion_token_count",
            "output_token_count",
            "candidates_token_count",
        ):
            if key in usage and usage.get(key) is not None:
                return cls._coerce_token_int(usage.get(key), allow_negative=allow_negative)
        return 0

    @classmethod
    def _extract_input_tokens(
        cls,
        usage: Dict[str, Any],
        output_tokens: int,
        *,
        allow_negative: bool = False,
    ) -> int:
        for key in (
            "input_tokens",
            "prompt_tokens",
            "prompt_token_count",
            "input_token_count",
        ):
            if key in usage and usage.get(key) is not None:
                return cls._coerce_token_int(usage.get(key), allow_negative=allow_negative)
        total_tokens = cls._coerce_token_int(usage.get("total_tokens"))
        if total_tokens > 0:
            return max(0, total_tokens - output_tokens)
        return 0

    @classmethod
    def _usage_token_counts(
        cls,
        usage: Dict[str, Any],
        *,
        allow_negative: bool = False,
    ) -> tuple[int, int]:
        out_t = cls._extract_output_tokens(usage, allow_negative=allow_negative)
        in_t = cls._extract_input_tokens(usage, out_t, allow_negative=allow_negative)
        return in_t, out_t

    def _apply_metadata(
        self,
        usage: Dict[str, Any],
        msg_id: str | None = None,
        source: str = "",
        accumulate: bool = False,
    ) -> int:
        if not isinstance(usage, dict) or not usage:
            return 0

        in_t, out_t = self._usage_token_counts(usage, allow_negative=accumulate)
        reported_cache_hit = extract_cache_hit_tokens(usage)
        cache_hit = reported_cache_hit if reported_cache_hit is not None else 0

        if in_t == 0 and out_t == 0 and reported_cache_hit is None:
            return 0

        msg_key = str(msg_id).strip() if msg_id else ""
        if not msg_key:
            msg_key = f"_unkeyed_step_{self._unkeyed_step_index}"
        else:
            # One provider call must land under a single key: streamed chunks
            # (``lc_run--...``) and the final message/node update (``resp_...``)
            # describe the same response.
            msg_key = self._resolve_key(msg_key)

        existing_in, existing_out, existing_cache_hit = self._step_usage.get(msg_key, (0, 0, 0))
        if accumulate:
            # A negative delta compensates an earlier over-reported chunk count
            # (Gemini 2.0 lowers the cumulative prompt count in the final chunk).
            new_in = max(0, existing_in + in_t)
            new_out = max(0, existing_out + out_t)
            new_cache_hit = existing_cache_hit + cache_hit
        else:
            new_in = max(existing_in, in_t)
            new_out = max(existing_out, out_t)
            new_cache_hit = max(existing_cache_hit, cache_hit)
        self._step_usage[msg_key] = (new_in, new_out, new_cache_hit)
        self._active_step_has_data = True
        return new_cache_hit - existing_cache_hit

    def render(self, duration: float) -> str:
        in_display = str(self.total_input if self.total_input > 0 else 0)
        if duration <= 60:
            duration_display = f"{duration:.1f}s"
        else:
            minutes, remaining_seconds = divmod(round(duration), 60)
            duration_display = f"{minutes}m {remaining_seconds}s"
        return f"{duration_display}  ↓ {in_display}  ↑ {self.total_output}"
