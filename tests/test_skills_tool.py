import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from core.config import AgentConfig
from core.tool_output_compressor import COMPRESSIBLE_TOOL_NAMES
from tools import skills as skills_module
from tools.skills import (
    SkillsIndex,
    build_skills_prompt_block,
    read_skills,
    set_runtime_config,
    set_working_directory,
    skills_tool_enabled,
)
from tools.tool_registry import ToolRegistry


VALID_SKILL = """---
name: pdf-processing
description: Извлечение и заполнение PDF. Использовать при работе с PDF-файлами.
when_to_use: когда нужно читать или заполнять PDF
---

# PDF Processing

Инструкции по обработке PDF.
"""

SECOND_SKILL = """---
name: docx-writing
description: Генерация DOCX-документов.
---

# DOCX Writing

Инструкции по DOCX.
"""

NO_FRONTMATTER = """# Broken

Нет frontmatter, скилл должен быть пропущен.
"""

NO_DESCRIPTION = """---
name: broken-no-desc
---

# Broken

Нет description, скилл должен быть пропущен.
"""


class SkillsToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path.cwd() / ".tmp_tests" / uuid4().hex
        self.skills_dir = self.root / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        # Isolate the process-wide skills runtime between tests.
        self.addCleanup(lambda: set_runtime_config(None))
        self.addCleanup(lambda: set_working_directory(Path.cwd()))

    def _write_skill(self, folder: str, content: str) -> None:
        skill_dir = self.skills_dir / folder
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    def _make_config(self, **overrides):
        defaults = {
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "PROMPT_PATH": Path(__file__).resolve().parents[1] / "prompt.txt",
            "MCP_CONFIG_PATH": Path(__file__).resolve().parents[1] / "tests" / "missing_mcp.json",
            "ENABLE_SEARCH_TOOLS": False,
            "ENABLE_PROCESS_TOOLS": False,
            "ENABLE_SHELL_TOOL": False,
            "SKILLS_DIR": str(self.skills_dir),
        }
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def _wire_runtime(self, **overrides):
        config = self._make_config(**overrides)
        set_working_directory(self.root)
        set_runtime_config(config)
        return config

    # --- scanning -------------------------------------------------------

    def test_index_scans_valid_and_skips_invalid_skills(self):
        self._write_skill("pdf-processing", VALID_SKILL)
        self._write_skill("docx-writing", SECOND_SKILL)
        self._write_skill("broken-no-frontmatter", NO_FRONTMATTER)
        self._write_skill("broken-no-desc", NO_DESCRIPTION)

        index = SkillsIndex(self.skills_dir)
        self.assertEqual(index.names(), ["docx-writing", "pdf-processing"])
        self.assertIsNone(index.get("broken-no-frontmatter"))
        self.assertIsNone(index.get("broken-no-desc"))

    def test_index_parses_skill_with_utf8_bom(self):
        # A SKILL.md saved on Windows as UTF-8 with BOM (or with a leading blank
        # line) must still be indexed, not silently skipped by the ^--- anchor.
        skill_dir = self.skills_dir / "pdf-processing"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_bytes(
            b"\xef\xbb\xbf" + VALID_SKILL.encode("utf-8")
        )

        index = SkillsIndex(self.skills_dir)
        self.assertEqual(index.names(), ["pdf-processing"])
        meta = index.get("pdf-processing")
        self.assertIsNotNone(meta)
        # The name must not carry a stray BOM character.
        self.assertEqual(meta.name, "pdf-processing")

    def test_prompt_block_renders_available_skills(self):
        self._write_skill("pdf-processing", VALID_SKILL)
        self._wire_runtime()

        block = build_skills_prompt_block()
        self.assertIn("<available_skills>", block)
        self.assertIn("<name>pdf-processing</name>", block)
        # when_to_use is appended to the description.
        self.assertIn("когда нужно читать или заполнять PDF", block)
        self.assertIn("read_skills", block)

    def test_prompt_block_empty_when_no_skills(self):
        self._wire_runtime()
        self.assertEqual(build_skills_prompt_block(), "")

    def test_prompt_block_empty_when_flag_disabled(self):
        self._write_skill("pdf-processing", VALID_SKILL)
        self._wire_runtime(ENABLE_SKILLS_TOOL=False)
        self.assertFalse(skills_tool_enabled())
        self.assertEqual(build_skills_prompt_block(), "")

    # --- read_skills tool ----------------------------------------------

    def test_read_skills_returns_content_and_relative_path(self):
        self._write_skill("pdf-processing", VALID_SKILL)
        self._wire_runtime()

        result = read_skills.invoke({"names": ["pdf-processing"]})
        self.assertIn("# PDF Processing", result)
        # Folder path must be relative to the workspace root, POSIX style.
        self.assertIn("folder: skills/pdf-processing", result)
        self.assertNotIn(str(self.root), result)

    def test_read_skills_folder_relative_to_skills_dir_when_outside_workspace(self):
        # When the workspace is unrelated to the skills directory (bundled next
        # to the executable), the folder path must stay portable — relative to
        # the skills dir (skills/<name>), never a leaked local absolute path.
        self._write_skill("pdf-processing", VALID_SKILL)
        config = self._make_config()
        elsewhere = self.root / "some" / "other" / "workspace"
        elsewhere.mkdir(parents=True, exist_ok=True)
        set_working_directory(elsewhere)
        set_runtime_config(config)

        result = read_skills.invoke({"names": ["pdf-processing"]})
        self.assertIn("# PDF Processing", result)
        self.assertIn("folder: skills/pdf-processing", result)
        self.assertNotIn(str(self.skills_dir), result)
        self.assertNotIn(str(elsewhere), result)

    def test_read_skills_reports_missing_names(self):
        self._write_skill("pdf-processing", VALID_SKILL)
        self._wire_runtime()

        result = read_skills.invoke({"names": ["nope"]})
        self.assertIn("Not found: nope", result)
        self.assertIn("pdf-processing", result)

    def test_read_skills_rejects_path_traversal(self):
        self._write_skill("pdf-processing", VALID_SKILL)
        self._wire_runtime()

        result = read_skills.invoke({"names": ["../../etc/passwd"]})
        self.assertIn("Not found", result)
        self.assertNotIn("root:", result)

    def test_read_skills_never_truncates_large_skill(self):
        # A skill larger than the compression threshold must come back whole.
        big_body = "Строка инструкции скилла.\n" * 400
        big_skill = f"---\nname: big-skill\ndescription: Большой скилл.\n---\n\n{big_body}"
        self._write_skill("big-skill", big_skill)
        self._wire_runtime()

        self.assertGreater(len(big_body), 2000)
        result = read_skills.invoke({"names": ["big-skill"]})
        # Every instruction line is preserved (no silent middle truncation).
        self.assertEqual(result.count("Строка инструкции скилла."), 400)
        self.assertNotIn("[TRUNCATED", result)

    def test_read_skills_excluded_from_output_compressor(self):
        self.assertNotIn("read_skills", COMPRESSIBLE_TOOL_NAMES)

    # --- auxiliary files via read_file read-only root ------------------

    def test_skills_folder_readable_outside_workspace_but_not_writable(self):
        import tools.filesystem as filesystem

        self._write_skill("pdf-processing", VALID_SKILL)
        aux = self.skills_dir / "pdf-processing" / "REFERENCE.md"
        aux.write_text("Справочные детали.", encoding="utf-8")

        # Workspace is a different directory; the skills folder lives outside it.
        other_workspace = self.root / "project"
        other_workspace.mkdir(parents=True, exist_ok=True)
        filesystem.set_working_directory(str(other_workspace))
        self.addCleanup(lambda: filesystem.set_working_directory(str(Path.cwd())))
        self.addCleanup(lambda: filesystem.set_read_only_roots(()))
        filesystem.set_read_only_roots((self.skills_dir,))

        # Read of a skill's auxiliary file (outside cwd) is allowed.
        read_result = filesystem.read_file_tool.invoke({"path": str(aux)})
        self.assertIn("Справочные детали.", read_result)

        # Writing into the read-only root is still denied.
        write_result = filesystem.write_file_tool.invoke(
            {"path": str(aux), "content": "changed"}
        )
        self.assertIn("ACCESS DENIED", write_result)
        self.assertEqual(aux.read_text(encoding="utf-8"), "Справочные детали.")

        # Arbitrary files outside both cwd and the read-only root stay denied.
        outside = self.root / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        denied = filesystem.read_file_tool.invoke({"path": str(outside)})
        self.assertIn("ACCESS DENIED", denied)

    # --- registration / feature flag -----------------------------------

    async def test_read_skills_registered_when_enabled(self):
        registry = ToolRegistry(self._make_config())
        await registry.load_all()
        self.assertIn("read_skills", {tool.name for tool in registry.tools})

    async def test_read_skills_not_registered_when_disabled(self):
        registry = ToolRegistry(self._make_config(ENABLE_SKILLS_TOOL=False))
        await registry.load_all()
        self.assertNotIn("read_skills", {tool.name for tool in registry.tools})

    async def test_read_skills_metadata_is_read_only_no_approval(self):
        registry = ToolRegistry(self._make_config())
        await registry.load_all()
        metadata = registry.tool_metadata.get("read_skills")
        self.assertIsNotNone(metadata)
        self.assertTrue(metadata.read_only)
        self.assertFalse(metadata.mutating)
        self.assertFalse(metadata.destructive)
        self.assertFalse(metadata.requires_approval)


if __name__ == "__main__":
    unittest.main()
