"""
tools/skills.py — Agent Skills support (Anthropic Claude Skills style).

Wiring into the existing architecture:
- ``SkillsIndex`` scans ``SKILLS_DIR`` for ``*/SKILL.md``, parses the YAML
  frontmatter (``name`` + ``description`` + optional ``when_to_use``) and renders
  the ``<available_skills>`` block that ``core/context_builder.py`` injects into
  the system prompt as a dedicated layer (after Runtime, before Safety).
- ``read_skills`` is a read-only tool registered in ``tools/tool_registry.py``
  next to ``read_file``/``write_file``. It only loads full ``SKILL.md`` content;
  files a skill references (``REFERENCE.md``, ``scripts/``) are read afterwards
  through the existing ``read_file`` tool using the folder path ``read_skills``
  returns.

Config is read from ``core.config.AgentConfig`` (never raw ``os.environ`` when a
config is wired), with an ``os.environ`` fallback for direct/tool-only use:
  SKILLS_DIR         default "skills"  (relative → resolved against the agent root)
  ENABLE_SKILLS_TOOL default "true"

Metadata (registered in ``tool_registry.py`` via ``ToolMetadata``):
read_only=True, mutating=False, destructive=False, requires_approval=False —
no approval, eligible for parallel read-only batching under
``MAX_PARALLEL_TOOL_CALLS``. The result is excluded from the tool output
compressor's deterministic truncation (see
``core/tool_output_compressor.py``) so full ``SKILL.md`` content is never cut.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.tools import tool

from core.constants import BASE_DIR

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

_DEFAULT_SKILLS_DIR = "skills"
# Soft cap for a single description inside the prompt block. Full text always
# stays available through read_skills; this only protects the system prompt
# from one verbose skill.
_DESCRIPTION_PROMPT_LIMIT = 500
# Fallback budget used only when no AgentConfig is wired (mirrors
# AgentConfig.max_tool_output_length default).
_DEFAULT_MAX_TOOL_OUTPUT = 4000


# --------------------------------------------------------------------------
# 1. Frontmatter parsing
# --------------------------------------------------------------------------

@dataclass
class SkillMeta:
    name: str
    description: str
    path: Path  # absolute path to the SKILL.md file
    when_to_use: str = ""


def _parse_frontmatter(text: str) -> dict:
    """Minimal parser for flat ``key: value`` frontmatter pairs.

    Sufficient for Skill frontmatter (name/description/when_to_use). Kept
    dependency-free on purpose; the project does not require PyYAML for tools.
    """
    # Strip a UTF-8 BOM and leading blank lines so a SKILL.md authored on
    # Windows (saved as UTF-8 with BOM) or with a leading empty line is not
    # silently skipped by the ``^---`` anchor.
    text = text.lstrip("\ufeff \t\r\n")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}

    raw = match.group(1)
    data: dict[str, str] = {}
    current_key: Optional[str] = None

    for line in raw.splitlines():
        if not line.strip():
            continue
        if line.startswith((" ", "\t")) and current_key:
            data[current_key] += " " + line.strip()
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value in (">", "|", ">-", "|-"):
                data[key] = ""
                current_key = key
            else:
                data[key] = value
                current_key = key
    return data


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


# --------------------------------------------------------------------------
# 2. Index
# --------------------------------------------------------------------------

class SkillsIndex:
    """Scans ``SKILLS_DIR`` once (lazily). Call :meth:`reload` to rescan after
    the folder or configuration changes."""

    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, SkillMeta] = {}
        self._scanned = False

    def ensure_scanned(self) -> None:
        if not self._scanned:
            self._scan()

    def reload(self) -> None:
        self._skills.clear()
        self._scanned = False
        self._scan()

    def _scan(self) -> None:
        self._scanned = True
        self._skills.clear()
        if not self.skills_dir.exists():
            logger.info("Skills: directory %s does not exist; 0 skills loaded.", self.skills_dir)
            return

        loaded = 0
        skipped: list[str] = []
        # sorted() keeps a deterministic order; on duplicate names the last one
        # in alphabetical order wins (a warning is logged).
        for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
            try:
                text = skill_md.read_text(encoding="utf-8-sig")
            except OSError:
                skipped.append(f"{skill_md.parent.name} (unreadable)")
                continue
            meta = _parse_frontmatter(text)
            if not meta:
                skipped.append(f"{skill_md.parent.name} (no frontmatter)")
                continue
            name = str(meta.get("name") or skill_md.parent.name).strip()
            description = str(meta.get("description") or "").strip()
            if not description:
                skipped.append(f"{skill_md.parent.name} (no description)")
                continue
            if name in self._skills:
                logger.warning(
                    "Skills: duplicate name '%s' in %s overrides %s.",
                    name,
                    skill_md.parent,
                    self._skills[name].path.parent,
                )
            self._skills[name] = SkillMeta(
                name=name,
                description=description,
                path=skill_md.resolve(),
                when_to_use=str(meta.get("when_to_use") or "").strip(),
            )
            loaded += 1

        if skipped:
            logger.info(
                "Skills: loaded %d skill(s) from %s; skipped %d (%s).",
                loaded,
                self.skills_dir,
                len(skipped),
                "; ".join(skipped),
            )
        else:
            logger.info("Skills: loaded %d skill(s) from %s.", loaded, self.skills_dir)

    def get(self, name: str) -> Optional[SkillMeta]:
        self.ensure_scanned()
        return self._skills.get(name)

    def names(self) -> list[str]:
        self.ensure_scanned()
        return sorted(self._skills)

    def as_prompt_block(self) -> str:
        """Rendered for injection into the system prompt. Only name+description
        (with optional ``when_to_use``) live here — full instructions load lazily
        through the ``read_skills`` tool."""
        self.ensure_scanned()
        if not self._skills:
            return ""

        lines = ["<available_skills>"]
        for name in sorted(self._skills):
            skill = self._skills[name]
            description = _clip(skill.description, _DESCRIPTION_PROMPT_LIMIT)
            if skill.when_to_use:
                description = f"{description} — {skill.when_to_use}"
                description = _clip(description, _DESCRIPTION_PROMPT_LIMIT)
            lines.append(
                "  <skill>\n"
                f"    <name>{skill.name}</name>\n"
                f"    <description>{description}</description>\n"
                "  </skill>"
            )
        lines.append("</available_skills>")
        lines.append(
            "\nBefore writing code or producing output for a task that plausibly "
            "matches one or more of the descriptions above, call read_skills with "
            "ALL plausibly relevant skill names in a single call, and follow their "
            "instructions before starting the task. If a SKILL.md references other "
            "files in its own folder (e.g. REFERENCE.md, scripts/), read those with "
            "read_file, using the folder path read_skills gives you, only when the "
            "current subtask actually needs them."
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 3. Process-wide runtime (config + workspace root + lazy index)
# --------------------------------------------------------------------------
# The prompt block builder and the tool below must see the same scan. The index
# is built lazily on first use, so the file system is never touched at startup.

class _SkillsRuntime:
    # Agent installation root. Uses core.constants.BASE_DIR, which is
    # frozen-aware: next to the executable in a PyInstaller build, or the
    # project root when running from source. Bundled skills ship with the
    # agent, so a relative SKILLS_DIR must anchor here and NOT to the runtime
    # cwd, which follows the user's project workspace.
    _AGENT_ROOT: Path = Path(BASE_DIR).resolve()

    def __init__(self) -> None:
        self.config: Any = None
        self.workspace_root: Path = Path.cwd().resolve()
        self._index: Optional[SkillsIndex] = None

    def reset_index(self) -> None:
        self._index = None

    def resolve_skills_dir(self) -> Path:
        raw: Any = None
        if self.config is not None:
            raw = getattr(self.config, "skills_dir", None)
        if raw in (None, ""):
            raw = os.environ.get("SKILLS_DIR", _DEFAULT_SKILLS_DIR)
        path = Path(raw)
        if path.is_absolute():
            return path.resolve()
        # Relative SKILLS_DIR is resolved against the agent root so bundled
        # skills are found regardless of the user's working directory.
        return (self._AGENT_ROOT / path).resolve()

    def get_index(self) -> SkillsIndex:
        if self._index is None:
            self._index = SkillsIndex(self.resolve_skills_dir())
        return self._index

    def max_tool_output(self) -> int:
        config = self.config
        if config is not None:
            safety = getattr(config, "safety", None)
            limit = getattr(safety, "max_tool_output", None)
            if isinstance(limit, int) and limit > 0:
                return limit
            limit = getattr(config, "max_tool_output_length", None)
            if isinstance(limit, int) and limit > 0:
                return limit
        return _DEFAULT_MAX_TOOL_OUTPUT


_RUNTIME = _SkillsRuntime()


def set_runtime_config(config: Any) -> None:
    """Wire the AgentConfig (called from tool_registry configure)."""
    _RUNTIME.config = config
    _RUNTIME.reset_index()


def set_working_directory(cwd: str | Path) -> None:
    """Propagate the workspace root so skill folder paths and SKILLS_DIR resolve
    against the same root as read_file/write_file."""
    _RUNTIME.workspace_root = Path(cwd).resolve()
    _RUNTIME.reset_index()


def get_skills_index() -> SkillsIndex:
    return _RUNTIME.get_index()


def get_skills_dir() -> Path:
    """Absolute path of the resolved skills directory (agent root by default).
    Used to register the folder as a read-only root for read_file/list_files so a
    skill's auxiliary files stay readable when the workspace is a different
    project."""
    return _RUNTIME.resolve_skills_dir()


def skills_tool_enabled() -> bool:
    if _RUNTIME.config is not None:
        return bool(getattr(_RUNTIME.config, "enable_skills_tool", True))
    return os.environ.get("ENABLE_SKILLS_TOOL", "true").strip().lower() not in {"false", "0", "no", "off"}


def build_skills_prompt_block() -> str:
    """Build the ``<available_skills>`` prompt layer.

    Returns ``""`` when the feature flag is off or no valid skills exist, so the
    caller can append it unconditionally.
    """
    if not skills_tool_enabled():
        return ""
    return get_skills_index().as_prompt_block()


# --------------------------------------------------------------------------
# 4. The tool
# --------------------------------------------------------------------------

def _relative_folder(path: Path) -> str:
    """Folder of a SKILL.md as a portable relative path (fewer tokens, no local
    absolute paths leaked into context). Prefers a path relative to the workspace
    root; for skills bundled outside the workspace (e.g. next to the executable)
    it falls back to a path relative to the skills directory (``<skills_dir_name>/
    <skill>``). Only when the skill lives outside both does it return the absolute
    path."""
    try:
        return path.parent.relative_to(_RUNTIME.workspace_root).as_posix()
    except ValueError:
        pass
    try:
        skills_dir = _RUNTIME.resolve_skills_dir()
        return (Path(skills_dir.name) / path.parent.relative_to(skills_dir)).as_posix()
    except (ValueError, OSError):
        return path.parent.as_posix()


def _coerce_names(names: Any) -> List[str]:
    if names is None:
        return []
    if isinstance(names, str):
        names = [names]
    result: List[str] = []
    for name in names:
        text = str(name).strip()
        if text:
            result.append(text)
    return result


@tool
def read_skills(names: list[str]) -> str:
    """Read the full SKILL.md instructions for one or more skills.

    Pass every plausibly relevant skill name from <available_skills> in one call
    (e.g. names=["pdf", "design"]) so they load together. Returns each skill's
    complete SKILL.md content, labeled with its name and folder path (relative to
    the project root). If a skill's instructions reference other files in that
    same folder, read those separately with read_file — the folder path given
    here is what you need to build those paths.
    """
    index = get_skills_index()
    requested = _coerce_names(names)
    parts: list[str] = []
    missing: list[str] = []

    for name in requested:
        # Names are resolved through the index only; an arbitrary path in the
        # argument (e.g. "../../etc/passwd") cannot be opened — path traversal is
        # impossible by design, such a name simply falls into "Not found".
        meta = index.get(name)
        if meta is None:
            missing.append(name)
            continue
        try:
            content = meta.path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            parts.append(f"=== Skill: {name} (folder: {_relative_folder(meta.path)}) ===\n[Unreadable: {exc}]")
            continue
        parts.append(
            f"=== Skill: {name} (folder: {_relative_folder(meta.path)}) ===\n{content}"
        )

    if missing:
        available = ", ".join(index.names()) or "(none)"
        parts.append(
            f"[Not found: {', '.join(missing)}. Available skills: {available}]"
        )

    if not parts:
        return "No skills found."

    result = "\n\n".join(parts)

    # Skill instructions must never be silently truncated. read_skills is
    # excluded from the tool output compressor, so the full text is delivered;
    # when it exceeds the output budget we say so explicitly instead of cutting
    # the middle of a SKILL.md.
    budget = _RUNTIME.max_tool_output()
    if len(result) > budget:
        result += (
            f"\n\n[NOTE: combined skills output is {len(result)} chars, above the "
            f"MAX_TOOL_OUTPUT budget of {budget}. Returned in full to preserve "
            f"complete instructions. If this crowds the context, request fewer "
            f"skills per call.]"
        )
    return result
