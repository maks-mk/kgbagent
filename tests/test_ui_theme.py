import os
import re
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QPalette, QTextDocument
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from ui.theme import build_stylesheet
from ui.widgets.composer import _ComposerMentionItemWidget
from ui.widgets.dialogs import ModelSettingsDialog
from ui.widgets.foundation import CodeHighlighter
from ui.widgets.messages import NoticeWidget


class UiThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def assert_monochrome(self, color):
        self.assertEqual(color.red(), color.green(), color.name())
        self.assertEqual(color.green(), color.blue(), color.name())

    def assert_monochrome_icon(self, pixmap):
        self.assertFalse(pixmap.isNull())
        image = pixmap.toImage()
        visible_pixels = 0
        for y in range(image.height()):
            for x in range(image.width()):
                color = image.pixelColor(x, y)
                if color.alpha():
                    visible_pixels += 1
                    self.assert_monochrome(color)
        self.assertGreater(visible_pixels, 0)

    def test_stylesheet_colors_are_monochrome_except_diff_content(self):
        content_colors = {
            "QLabel#DiffHeaderPath": "#6FC8FF",
            "QLabel#DiffStatAdded": "#63D882",
            "QLabel#DiffStatRemoved": "#FF7B72",
        }
        colored_rules = {}
        for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", build_stylesheet()):
            for color in re.findall(r"#[0-9A-Fa-f]{6}\b", body):
                if color[1:3].lower() != color[3:5].lower() or color[3:5].lower() != color[5:7].lower():
                    colored_rules[selectors.strip()] = color.upper()
        self.assertEqual(colored_rules, content_colors)

    def test_top_status_colors_are_monochrome_in_every_state(self):
        chip = QLabel("Ready")
        self.addCleanup(chip.deleteLater)
        chip.setObjectName("TopStatusChip")
        chip.setStyleSheet(build_stylesheet())
        for state in ("idle", "busy", "success", "error"):
            with self.subTest(state=state):
                chip.setProperty("statusState", state)
                chip.style().unpolish(chip)
                chip.style().polish(chip)
                self.assert_monochrome(chip.palette().color(QPalette.Window))
                self.assert_monochrome(chip.palette().color(QPalette.WindowText))

    def test_save_and_saved_have_distinct_monochrome_colors(self):
        with mock.patch.object(ModelSettingsDialog, "_schedule_fetch"):
            dialog = ModelSettingsDialog({"active_profile": None, "profiles": []})
        self.addCleanup(dialog.close)
        dialog.setStyleSheet(build_stylesheet())
        button = dialog.save_button
        button.setEnabled(True)
        button.ensurePolished()
        save_background = button.palette().color(QPalette.Button)
        self.assertEqual(save_background.name(), "#ffffff")
        self.assertEqual(button.palette().color(QPalette.ButtonText).name(), "#000000")
        self.assert_monochrome_icon(button.icon().pixmap(11, 11))

        dialog._show_saved_button_state()
        saved_background = button.palette().color(QPalette.Button)
        self.assert_monochrome(saved_background)
        self.assertNotEqual(saved_background, save_background)
        self.assertEqual(button.palette().color(QPalette.ButtonText).name(), "#ffffff")
        self.assert_monochrome_icon(button.icon().pixmap(11, 11))

    def test_notice_and_file_picker_icons_are_monochrome(self):
        notice = NoticeWidget("Status")
        owner = QWidget()
        self.addCleanup(notice.deleteLater)
        self.addCleanup(owner.deleteLater)
        for level in ("info", "success", "warning", "error"):
            with self.subTest(level=level):
                notice.set_level(level)
                self.assert_monochrome_icon(notice.icon_label.pixmap())
        for path in ("folder", "settings.toml", "main.py", "readme.md", "image.png", "archive.zip", "document.pdf", "table.xlsx", "other.bin"):
            with self.subTest(path=path):
                icon = _ComposerMentionItemWidget._icon_for_entry(owner, relative=path, is_dir=path == "folder")
                self.assert_monochrome_icon(icon.pixmap(13, 13))

    def test_code_highlighting_retains_original_content_colors(self):
        document = QTextDocument()
        document.setPlainText('def demo():\n    return "value" # comment')
        highlighter = CodeHighlighter(document, "python")
        highlighter.rehighlight()
        first_block_colors = {item.format.foreground().color().name() for item in document.firstBlock().layout().formats()}
        second_block_colors = {item.format.foreground().color().name() for item in document.lastBlock().layout().formats()}
        self.assertEqual(first_block_colors, {"#8ab4f8"})
        self.assertEqual(second_block_colors, {"#8ab4f8", "#63d882", "#929292"})
