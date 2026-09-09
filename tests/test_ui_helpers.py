import os
import shutil
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from ui.main_window_state import StreamEventRouter
from ui.streaming import StreamEvent
from ui.theme import build_stylesheet
from ui.widgets.attachments import ImageAttachmentChipWidget
from ui.widgets.composer import ComposerTextEdit
from ui.widgets.foundation import SummaryProgressRing
from ui.widgets.tool_group import ToolGroupWidget


class UiHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_stream_event_router_dispatches_known_events(self):
        handler = mock.Mock()
        router = StreamEventRouter({"status_changed": handler})

        router.dispatch(StreamEvent("status_changed", {"label": "Working"}))
        router.dispatch(StreamEvent("ignored", {"label": "noop"}))

        handler.assert_called_once_with({"label": "Working"})

    def test_build_stylesheet_is_cached(self):
        first = build_stylesheet()
        second = build_stylesheet()

        self.assertIs(first, second)
        self.assertIn("QStatusBar", first)

    def test_model_image_checkbox_indicator_has_visible_border(self):
        stylesheet = build_stylesheet()

        self.assertIn("QCheckBox#ModelSupportsImagesCheckbox::indicator", stylesheet)
        self.assertIn("QCheckBox#ModelSupportsImagesCheckbox::indicator:checked", stylesheet)
        self.assertIn("border: 1px solid", stylesheet)
        self.assertNotIn("checkbox-check.svg", stylesheet)

    def test_enabled_tool_switch_uses_white_track(self):
        stylesheet = build_stylesheet()
        selector = "QCheckBox#ToolAvailabilitySwitch:checked"
        start = stylesheet.index(selector)
        checked_rule = stylesheet[start:stylesheet.index("}", start)]

        self.assertIn("background: #FFFFFF;", checked_rule)
        self.assertIn("border: 1px solid #FFFFFF;", checked_rule)

    def test_enabled_model_switch_uses_white_track(self):
        stylesheet = build_stylesheet()
        selector = "QCheckBox#ModelProfileEnabledSwitch:checked"
        start = stylesheet.index(selector)
        checked_rule = stylesheet[start:stylesheet.index("}", start)]
        indicator_selector = "QCheckBox#ModelProfileEnabledSwitch::indicator:checked"
        indicator_start = stylesheet.index(indicator_selector)
        indicator_rule = stylesheet[indicator_start:stylesheet.index("}", indicator_start)]

        self.assertIn("background: #FFFFFF;", checked_rule)
        self.assertIn("border: 1px solid #FFFFFF;", checked_rule)
        self.assertIn("background: #1E1D1B;", indicator_rule)

    def test_tool_card_meta_labels_have_transparent_background(self):
        stylesheet = build_stylesheet()
        selector = "QLabel#MetaText"
        start = stylesheet.index(selector)
        meta_rule = stylesheet[start:stylesheet.index("}", start)]
        error_selector = 'QLabel#MetaText[severity="error"]'
        error_start = stylesheet.index(error_selector)
        error_rule = stylesheet[error_start:stylesheet.index("}", error_start)]

        self.assertIn("background: transparent;", meta_rule)
        self.assertNotIn("background:", error_rule)

    def test_empty_attachment_path_uses_placeholder_without_file_load(self):
        with mock.patch("ui.widgets.attachments.QPixmap") as pixmap_cls:
            placeholder = mock.Mock()
            pixmap_cls.return_value = placeholder

            result = ImageAttachmentChipWidget._load_pixmap({"path": ""}, 40)

        pixmap_cls.assert_called_once_with(40, 40)
        placeholder.fill.assert_called_once()
        self.assertIs(result, placeholder)

    def test_summary_progress_ring_updates_tooltip_from_payload(self):
        ring = SummaryProgressRing()
        self.addCleanup(ring.deleteLater)

        ring.set_summary_progress(
            {
                "estimated_tokens": 6400,
                "threshold": 8000,
                "trigger_tokens": 10800,
                "remaining_tokens": 4400,
                "reserved_tokens": 3000,
                "summary_tokens": 850,
                "provider_input_tokens": 229094,
                "progress": 0.2,
                "will_summarize": False,
            }
        )

        self.assertTrue(ring.isVisible())
        self.assertEqual(ring.toolTip(), "20% left until auto-summary.")
        self.assertNotIn("tokens", ring.toolTip())

    def test_summary_progress_ring_falls_back_to_threshold_without_trigger(self):
        ring = SummaryProgressRing()
        self.addCleanup(ring.deleteLater)

        ring.set_summary_progress({"estimated_tokens": 6400, "threshold": 8000})

        self.assertEqual(ring.toolTip(), "20% left until auto-summary.")

    def test_summary_progress_ring_reports_imminent_auto_summary(self):
        ring = SummaryProgressRing()
        self.addCleanup(ring.deleteLater)

        ring.set_summary_progress(
            {"estimated_tokens": 8100, "threshold": 8000, "progress": 0.0, "will_summarize": True}
        )

        self.assertEqual(ring.toolTip(), "Auto-summary will run on the next step.")

    def test_summary_progress_ring_marks_disabled_without_threshold(self):
        ring = SummaryProgressRing()
        self.addCleanup(ring.deleteLater)

        ring.set_summary_progress({"estimated_tokens": 6400})

        self.assertEqual(ring.toolTip(), "Auto-summary is disabled.")
        self.assertFalse(ring.isVisible())

    def test_tool_group_animates_expand_and_collapse(self):
        group = ToolGroupWidget()
        group.inner.addWidget(QLabel("Tool details", group.container))
        group.show()
        group.resize(240, 120)
        self.app.processEvents()
        self.addCleanup(group.deleteLater)

        group.collapse()
        self.assertTrue(group.container.isVisible())
        self.assertGreater(group._container_animation.duration(), 0)
        group._container_animation.setCurrentTime(group._container_animation.duration())
        self.app.processEvents()
        self.assertTrue(group.container.isHidden())
        self.assertEqual(group.container.maximumHeight(), 0)

        group.expand()
        self.assertTrue(group.container.isVisible())
        group._container_animation.setCurrentTime(group._container_animation.duration())
        self.app.processEvents()
        self.assertTrue(group.container.isVisible())
        self.assertEqual(group.container.maximumHeight(), 16777215)

    def test_composer_file_index_refreshes_empty_mention_after_directory_change(self):
        composer = ComposerTextEdit()
        self.addCleanup(composer.deleteLater)
        temp_root = Path.cwd() / ".tmp_tests" / f"composer-index-{time.time_ns()}"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: temp_root.exists() and shutil.rmtree(temp_root, ignore_errors=True))

        with mock.patch("ui.widgets.composer.Path.cwd", return_value=temp_root):
            (temp_root / "first.py").write_text("print('first')", encoding="utf-8")
            composer._ensure_file_index(force_refresh=True)
            self.assertEqual([row["relative"] for row in composer._filter_mention_candidates("")], ["first.py"])

            (temp_root / "late.py").write_text("print('late')", encoding="utf-8")
            composer._on_file_index_directory_changed(str(temp_root))
            rows = composer._filter_mention_candidates("")

        self.assertIn("first.py", [row["relative"] for row in rows])
        self.assertIn("late.py", [row["relative"] for row in rows])


class FilenameSpanTests(unittest.TestCase):
    def test_finds_plain_filenames_and_paths(self):
        from core.text_utils import find_filename_spans

        cases = {
            "Смотри файл main.py в папке core": ["main.py"],
            "Изменил src/ui/theme.py и core\\text_utils.py": ["src/ui/theme.py", "core\\text_utils.py"],
            "Windows: D:\\project\\main.py открылся": ["D:\\project\\main.py"],
            "Отредактированы Dockerfile и Makefile": ["Dockerfile", "Makefile"],
            "картинка image.png и архив data.tar.gz": ["image.png", "data.tar.gz"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                spans = find_filename_spans(text)
                self.assertEqual([text[start:end] for start, end in spans], expected)

    def test_ignores_versions_and_non_filenames(self):
        from core.text_utils import find_filename_spans

        cases = [
            "Версия 1.2.3 и main.py2 не подсвечиваются",
            "Обычный текст без файлов",
            "см. раздел 3.14 главы",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(find_filename_spans(text), [])

    def test_recognizes_extensions_with_dotless_suffixes(self):
        from core.text_utils import find_filename_spans

        text = "файл headers.json_codex"
        self.assertEqual(
            [text[start:end] for start, end in find_filename_spans(text)],
            ["headers.json_codex"],
        )
        text = "config.json_codex и config.json_bak"
        self.assertEqual(
            [text[start:end] for start, end in find_filename_spans(text)],
            ["config.json_codex", "config.json_bak"],
        )

    def test_empty_input_returns_no_spans(self):
        from core.text_utils import find_filename_spans

        self.assertEqual(find_filename_spans(""), [])
        self.assertEqual(find_filename_spans(None), [])
