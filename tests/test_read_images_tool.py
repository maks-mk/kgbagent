import asyncio
import base64
import os
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PySide6.QtGui import QImage

from core.config import AgentConfig
from core.context_builder import ContextBuilder
from core.multimodal import (
    IMAGE_CONTENT_OMITTED_PLACEHOLDER,
    TOOL_RESULT_IMAGE_OMITTED_PLACEHOLDER,
    TOOL_RESULT_IMAGE_UNREADABLE_TEMPLATE,
    materialize_tool_result_image_blocks,
    provider_supports_tool_result_images,
    strip_tool_result_image_blocks,
)
from core.safety_policy import SafetyPolicy
from core.self_correction_engine import build_repair_plan
from core.text_utils import build_tool_ui_labels
from tools import filesystem
from tools.tool_registry import ToolRegistry
from ui.widgets.tools import TOOL_ROLE_ICONS


def _write_png(path: Path, width: int = 4, height: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = QImage(width, height, QImage.Format_ARGB32)
    image.fill(0xFF3366FF)
    if not image.save(str(path), "PNG"):
        raise RuntimeError(f"failed to write test image: {path}")
    return path


class ReadImagesToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path.cwd() / ".tmp_tests" / uuid4().hex
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self._previous_cwd = filesystem.fs_manager.cwd
        self._previous_policy = filesystem.fs_manager.safety_policy
        self._previous_read_only_roots = filesystem.fs_manager.read_only_roots
        self.addCleanup(self._restore_filesystem_state)
        filesystem.set_safety_policy(SafetyPolicy())
        filesystem.set_working_directory(str(self.tmp))

    def _restore_filesystem_state(self):
        filesystem.set_working_directory(str(self._previous_cwd))
        filesystem.set_safety_policy(self._previous_policy)
        filesystem.fs_manager.set_read_only_roots(self._previous_read_only_roots)

    def test_attaches_image_block_with_mime_type_and_dimensions(self):
        _write_png(self.tmp / "shot.png", width=6, height=5)

        result = filesystem.read_images_tool.invoke({"paths": "shot.png"})

        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["type"], "text")
        self.assertIn("Attached 1 image(s)", result[0]["text"])
        block = result[1]
        self.assertEqual(block["type"], "image")
        self.assertEqual(block["mime_type"], "image/png")
        self.assertEqual(block["file_name"], "shot.png")
        self.assertEqual((block["width"], block["height"]), (6, 5))
        self.assertEqual(Path(block["path"]), (self.tmp / "shot.png").resolve())
        self.assertGreater(block["size_bytes"], 0)
        self.assertNotIn("base64", block)

    def test_single_path_alias_is_accepted(self):
        _write_png(self.tmp / "aliased.png")

        result = filesystem.read_images_tool.invoke({"path": "aliased.png"})

        self.assertIsInstance(result, list)
        self.assertEqual(result[1]["file_name"], "aliased.png")

    def test_list_of_paths_is_capped_and_remaining_paths_are_reported(self):
        names = [f"frame_{index}.png" for index in range(6)]
        for name in names:
            _write_png(self.tmp / name)

        result = filesystem.read_images_tool.invoke({"paths": names})

        attached = [block for block in result if block["type"] == "image"]
        self.assertEqual(len(attached), 4)
        self.assertIn("2 more image(s) were not included", result[0]["text"])

    def test_max_images_lowers_the_per_call_cap(self):
        for index in range(3):
            _write_png(self.tmp / f"page_{index}.png")

        result = filesystem.read_images_tool.invoke(
            {"paths": ["page_0.png", "page_1.png", "page_2.png"], "max_images": 1}
        )

        attached = [block for block in result if block["type"] == "image"]
        self.assertEqual(len(attached), 1)
        self.assertEqual(attached[0]["file_name"], "page_0.png")

    def test_missing_paths_field_is_a_validation_error(self):
        result = filesystem.read_images_tool.invoke({})

        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("ERROR[VALIDATION]"), result)

    def test_text_file_is_rejected_although_it_exists(self):
        target = self.tmp / "notes.txt"
        target.write_text("not an image", encoding="utf-8")

        result = filesystem.read_images_tool.invoke({"paths": "notes.txt"})

        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("ERROR[VALIDATION]"), result)
        self.assertIn("not a readable image file", result)

    def test_missing_image_reports_not_found(self):
        result = filesystem.read_images_tool.invoke({"paths": "missing.png"})

        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("ERROR[NOT_FOUND]"), result)

    def test_workspace_escape_is_denied(self):
        outside = Path.cwd() / ".tmp_tests" / f"{uuid4().hex}.png"
        _write_png(outside)
        self.addCleanup(lambda: outside.unlink(missing_ok=True))

        result = filesystem.read_images_tool.invoke({"paths": str(outside)})

        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("ERROR[ACCESS_DENIED]"), result)

    def test_read_only_root_makes_skill_images_reachable(self):
        root = Path.cwd() / ".tmp_tests" / uuid4().hex
        _write_png(root / "diagram.png")
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        filesystem.set_read_only_roots((root,))

        result = filesystem.read_images_tool.invoke({"paths": str(root / "diagram.png")})

        self.assertIsInstance(result, list)
        self.assertEqual(result[1]["file_name"], "diagram.png")

    def test_oversized_image_is_rejected_with_limit_error(self):
        _write_png(self.tmp / "big.png")
        filesystem.set_safety_policy(SafetyPolicy(max_file_size=16))

        result = filesystem.read_images_tool.invoke({"paths": "big.png"})

        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("ERROR[LIMIT_EXCEEDED]"), result)

    def test_partial_failure_keeps_the_readable_image_and_notes_the_skip(self):
        _write_png(self.tmp / "ok.png")

        result = filesystem.read_images_tool.invoke({"paths": ["ok.png", "gone.png"]})

        attached = [block for block in result if block["type"] == "image"]
        self.assertEqual(len(attached), 1)
        self.assertIn("Skipped 1 path(s)", result[0]["text"])
        self.assertIn("gone.png", result[0]["text"])


class ReadImagesRegistryTests(unittest.TestCase):
    def _make_config(self, **overrides) -> AgentConfig:
        defaults = {
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "PROMPT_PATH": Path(__file__).resolve().parents[1] / "prompt.txt",
            "MCP_CONFIG_PATH": Path(__file__).resolve().parents[1] / "tests" / "missing_mcp.json",
            "ENABLE_SEARCH_TOOLS": False,
            "ENABLE_PROCESS_TOOLS": False,
            "ENABLE_SHELL_TOOL": False,
        }
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def test_read_images_is_registered_as_read_only_filesystem_tool(self):
        async def run() -> ToolRegistry:
            registry = ToolRegistry(self._make_config(ENABLE_FILESYSTEM_TOOLS=True))
            await registry.load_all()
            return registry

        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: filesystem.set_working_directory(str(original_cwd)))

        registry = asyncio.run(run())

        self.assertIn("read_images", {tool.name for tool in registry.tools})
        metadata = registry.tool_metadata["read_images"]
        self.assertTrue(metadata.read_only)
        self.assertFalse(metadata.mutating)
        self.assertFalse(metadata.requires_approval)


class ReadImagesContentBlockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path.cwd() / ".tmp_tests" / uuid4().hex
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def _image_block(self, *, name: str = "chart.png", path: str | None = None) -> dict:
        target = _write_png(self.tmp / name)
        return {
            "type": "image",
            "path": path if path is not None else str(target),
            "mime_type": "image/png",
            "file_name": name,
        }

    def test_provider_support_gate(self):
        self.assertFalse(provider_supports_tool_result_images("openai", "chat"))
        self.assertTrue(provider_supports_tool_result_images("openai", "responses"))
        self.assertTrue(provider_supports_tool_result_images("anthropic", "chat"))
        self.assertTrue(provider_supports_tool_result_images("gemini", "chat"))
        self.assertFalse(provider_supports_tool_result_images("ollama", "chat"))

    def test_materialize_uses_anthropic_base64_source(self):
        block = self._image_block()
        expected = base64.b64encode(Path(block["path"]).read_bytes()).decode("ascii")

        materialized = materialize_tool_result_image_blocks(
            [{"type": "text", "text": "attached"}, block],
            provider="anthropic",
        )

        self.assertEqual(materialized[0], {"type": "text", "text": "attached"})
        self.assertEqual(materialized[1]["type"], "image")
        self.assertEqual(materialized[1]["source"]["type"], "base64")
        self.assertEqual(materialized[1]["source"]["media_type"], "image/png")
        self.assertEqual(materialized[1]["source"]["data"], expected)

    def test_materialize_uses_openai_data_uri(self):
        block = self._image_block()

        materialized = materialize_tool_result_image_blocks([block], provider="openai")

        self.assertEqual(materialized[0]["type"], "image_url")
        self.assertTrue(
            materialized[0]["image_url"]["url"].startswith("data:image/png;base64,"),
            materialized[0]["image_url"]["url"][:40],
        )

    def test_materialize_replaces_a_vanished_file_with_text(self):
        block = self._image_block()
        Path(block["path"]).unlink()

        materialized = materialize_tool_result_image_blocks([block], provider="anthropic")

        self.assertEqual(materialized[0]["type"], "text")
        self.assertEqual(
            materialized[0]["text"],
            TOOL_RESULT_IMAGE_UNREADABLE_TEMPLATE.format(path=block["path"]),
        )

    def test_materialize_leaves_text_only_content_untouched(self):
        content = [{"type": "text", "text": "nothing to attach"}]

        self.assertIs(materialize_tool_result_image_blocks(content, provider="anthropic"), content)

    def test_strip_keeps_text_and_states_the_omission(self):
        stripped, removed = strip_tool_result_image_blocks(
            [{"type": "text", "text": "attached 1 image(s)"}, self._image_block()]
        )

        self.assertEqual(removed, 1)
        self.assertEqual(stripped[0], {"type": "text", "text": "attached 1 image(s)"})
        self.assertEqual(stripped[1], {"type": "text", "text": TOOL_RESULT_IMAGE_OMITTED_PLACEHOLDER})

    def test_strip_of_image_only_content_returns_the_placeholder(self):
        stripped, removed = strip_tool_result_image_blocks([self._image_block()])

        self.assertEqual(removed, 1)
        self.assertEqual(stripped, TOOL_RESULT_IMAGE_OMITTED_PLACEHOLDER)

    def test_strip_ignores_text_only_content(self):
        content = [{"type": "text", "text": "attached 0 image(s)"}]

        stripped, removed = strip_tool_result_image_blocks(content)

        self.assertEqual(removed, 0)
        self.assertIs(stripped, content)


class ReadImagesContextBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path.cwd() / ".tmp_tests" / uuid4().hex
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def _make_config(self, **overrides) -> AgentConfig:
        defaults = {
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "ANTHROPIC_API_KEY": "test-key",
            "PROMPT_PATH": Path(__file__).resolve().parents[1] / "prompt.txt",
            "MCP_CONFIG_PATH": Path(__file__).resolve().parents[1] / "tests" / "missing_mcp.json",
            "ENABLE_SEARCH_TOOLS": False,
            "ENABLE_PROCESS_TOOLS": False,
            "ENABLE_SHELL_TOOL": False,
        }
        defaults.update(overrides)
        return AgentConfig(**defaults)

    @staticmethod
    def _text_of(message) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        return "\n".join(item.get("text", "") for item in content if isinstance(item, dict))

    @staticmethod
    def _block_types(message) -> list:
        content = message.content
        if not isinstance(content, list):
            return []
        return [item.get("type") for item in content if isinstance(item, dict)]

    def _builder(self, config: AgentConfig, *, image_input_supported: bool = True) -> ContextBuilder:
        return ContextBuilder(
            config=config,
            model_capabilities={"image_input_supported": image_input_supported},
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

    def _tool_turn(self) -> list:
        block = {
            "type": "image",
            "path": str(_write_png(self.tmp / "turn.png")),
            "mime_type": "image/png",
            "file_name": "turn.png",
        }
        return [
            AIMessage(content="", tool_calls=[{"id": "call-000001", "name": "read_images", "args": {}}]),
            ToolMessage(
                name="read_images",
                tool_call_id="call-000001",
                content=[{"type": "text", "text": "Attached 1 image(s)"}, block],
            ),
            block,
        ]

    def test_openai_chat_mode_replaces_tool_image_blocks_with_a_note(self):
        ai_message, tool_message, _block = self._tool_turn()
        builder = self._builder(self._make_config(PROVIDER="openai", LLM_API_MODE="chat"))

        sanitized = builder.sanitize_messages([ai_message, tool_message])

        output = sanitized[1]
        self.assertIsInstance(output, ToolMessage)
        # Chat completions cannot carry images in tool output; the blocks are
        # forwarded to a synthetic user message instead of being dropped.
        self.assertNotIn("data:image", self._text_of(output))

        carrier = sanitized[2]
        self.assertIsInstance(carrier, HumanMessage)
        image_blocks = [item for item in carrier.content if isinstance(item, dict) and item.get("type") == "image_url"]
        self.assertEqual(len(image_blocks), 1)
        self.assertTrue(image_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_openai_chat_mode_forwards_tool_images_with_the_carrier_note(self):
        ai_message, tool_message, _block = self._tool_turn()
        builder = self._builder(self._make_config(PROVIDER="openai", LLM_API_MODE="chat"))

        sanitized = builder.sanitize_messages([ai_message, tool_message])

        self.assertIn("attached to this message", self._text_of(sanitized[1]))

    def test_openai_chat_mode_strips_tool_images_for_text_only_models(self):
        ai_message, tool_message, _block = self._tool_turn()
        builder = self._builder(
            self._make_config(PROVIDER="openai", LLM_API_MODE="chat"),
            image_input_supported=False,
        )

        sanitized = builder.sanitize_messages([ai_message, tool_message])

        self.assertEqual(len(sanitized), 2)
        output = sanitized[1]
        self.assertIsInstance(output, ToolMessage)
        self.assertIn(IMAGE_CONTENT_OMITTED_PLACEHOLDER, self._text_of(output))
        self.assertNotIn("data:image", self._text_of(output))

    def test_openai_chat_mode_merges_forwarded_tool_images_into_the_next_user_message(self):
        ai_message, tool_message, _block = self._tool_turn()
        user_message = HumanMessage(content="what is on the image?")
        builder = self._builder(self._make_config(PROVIDER="openai", LLM_API_MODE="chat"))

        sanitized = builder.sanitize_messages([ai_message, tool_message, user_message])

        self.assertEqual(len(sanitized), 3)
        carrier = sanitized[2]
        self.assertIsInstance(carrier, HumanMessage)
        self.assertEqual(carrier.content[0]["type"], "image_url")
        self.assertEqual(carrier.content[-1], {"type": "text", "text": "what is on the image?"})

    def test_openai_responses_mode_materializes_tool_images_as_data_uri(self):
        ai_message, tool_message, _block = self._tool_turn()
        builder = self._builder(self._make_config(PROVIDER="openai", LLM_API_MODE="responses"))

        sanitized = builder.sanitize_messages([ai_message, tool_message])

        output = sanitized[1]
        self.assertIsInstance(output, ToolMessage)
        image_blocks = [item for item in output.content if item["type"] == "image_url"]
        self.assertEqual(len(image_blocks), 1)
        self.assertTrue(image_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_anthropic_materializes_tool_images_as_base64(self):
        ai_message, tool_message, block = self._tool_turn()
        expected = base64.b64encode(Path(block["path"]).read_bytes()).decode("ascii")
        builder = self._builder(self._make_config(PROVIDER="anthropic"))

        sanitized = builder.sanitize_messages([ai_message, tool_message])

        output = sanitized[1]
        self.assertIsInstance(output, ToolMessage)
        image_blocks = [item for item in output.content if item["type"] == "image"]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0]["source"]["data"], expected)
        self.assertEqual(image_blocks[0]["source"]["media_type"], "image/png")

    def test_text_only_model_loses_tool_images_with_a_note(self):
        ai_message, tool_message, _block = self._tool_turn()
        builder = self._builder(self._make_config(PROVIDER="anthropic"), image_input_supported=False)

        sanitized = builder.sanitize_messages([ai_message, tool_message])

        output = sanitized[1]
        self.assertIsInstance(output, ToolMessage)
        self.assertNotIn("image", self._block_types(output))
        self.assertEqual(output.content[-1]["text"], IMAGE_CONTENT_OMITTED_PLACEHOLDER)

    def test_text_only_tool_result_survives_untouched(self):
        tool_message = ToolMessage(
            name="read_file",
            tool_call_id="call-000002",
            content="plain text result",
        )
        builder = self._builder(self._make_config(PROVIDER="anthropic"))

        sanitized = builder.sanitize_messages([tool_message])

        self.assertEqual(sanitized[0].content, "plain text result")


class ReadImagesPresentationTests(unittest.TestCase):
    """UI labels and recovery guidance must treat read_images as a read tool."""

    def test_ui_role_is_read_in_both_tool_widgets(self):
        from ui.widgets.tool_group import ToolGroupWidget
        from ui.widgets.tools import ToolCardWidget

        card = SimpleNamespace(payload={"name": "read_images"})
        self.assertEqual(ToolCardWidget._tool_role("read_images"), "read")
        self.assertEqual(ToolGroupWidget._tool_role(card), "read")

    def test_card_title_uses_read_role_titles(self):
        from ui.widgets.tools import ToolCardWidget

        payload = {"name": "read_images", "args": {"paths": ["shot.png"]}, "phase": "finished"}
        self.assertEqual(ToolCardWidget._localized_title("read_images", payload), "Read")
        self.assertEqual(ToolCardWidget._tool_icon_name("read_images"), TOOL_ROLE_ICONS["read"])

    def test_card_renders_first_path_instead_of_raw_list(self):
        from PySide6.QtWidgets import QApplication

        from ui.widgets.tools import ToolCardWidget

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)

        card = ToolCardWidget(
            {"name": "read_images", "args": {"paths": ["shots/one.png", "shots/two.png"]}, "phase": "running"}
        )
        self.addCleanup(card.deleteLater)

        self.assertEqual(card.action_label.full_text(), "Reading shots/one.png, shots/two.png")

    def test_group_header_counts_every_requested_image(self):
        from PySide6.QtWidgets import QApplication

        from ui.widgets.tool_group import ToolGroupWidget
        from ui.widgets.tools import ToolCardWidget

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)

        group = ToolGroupWidget()
        self.addCleanup(group.deleteLater)
        card = ToolCardWidget(
            {"name": "read_images", "args": {"paths": ["shots/one.png", "shots/two.png"]}, "phase": "finished"},
            parent=group.container,
        )
        group.add_tool(card)
        group.refresh_completion()

        self.assertEqual(group.header_btn.text(), "Read 2 files")

    def test_build_tool_ui_labels_renders_every_path(self):
        labels = build_tool_ui_labels("read_images", {"paths": ["shots/one.png", "shots/two.png"]})

        self.assertEqual(labels["subtitle"], "shots/one.png, shots/two.png")
        self.assertEqual(labels["raw_display"], "read_images(shots/one.png, shots/two.png)")
        self.assertEqual(labels["title"], "Reading images")
        self.assertEqual(labels["args_state"], "complete")

    def test_single_path_alias_still_renders_one_target(self):
        labels = build_tool_ui_labels("read_images", {"path": "shots/one.png"})

        self.assertEqual(labels["subtitle"], "shots/one.png")

    def test_missing_paths_shows_the_placeholder_instead_of_none(self):
        labels = build_tool_ui_labels("read_images", {"max_images": 2})

        self.assertNotIn("None", labels["subtitle"])
        self.assertEqual(labels["args_state"], "partial")

    def test_missing_paths_suggests_list_directory(self):
        issue = {
            "tool_names": ["read_images"],
            "tool_args": {},
            "summary": "Missing required field(s): paths.",
            "error_type": "VALIDATION",
            "details": {"missing_required_fields": ["paths"]},
        }

        plan = build_repair_plan(issue, current_task="Покажи картинку", max_auto_repairs=2)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan.retryable)
        self.assertEqual(plan.suggested_tool_name, "list_directory")


if __name__ == "__main__":
    unittest.main()
