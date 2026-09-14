import unittest
from pathlib import Path

from PySide6.QtGui import QColor, QIcon

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox, QToolButton

from ui.theme import TEXT_PRIMARY, TEXT_SECONDARY, build_stylesheet
from ui.widgets.window_chrome import install_dialog_chrome


class DialogChromeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_dialog_chrome(cls.app)

    def test_standard_dialogs_receive_chrome_and_theme(self):
        for dialog in (QMessageBox(), QFileDialog()):
            with self.subTest(dialog=type(dialog).__name__):
                dialog.setWindowTitle("Test dialog")
                dialog.ensurePolished()
                self.assertTrue(dialog.windowFlags() & Qt.WindowType.FramelessWindowHint)
                self.assertEqual(dialog.styleSheet(), build_stylesheet())
                self.assertIsNotNone(dialog.layout().menuBar())
                self.assertIsNotNone(dialog.findChild(QToolButton, "SidebarGhostButton"))
                dialog.close()
                dialog.deleteLater()

    def test_file_dialog_navigation_icons_have_visible_normal_and_disabled_colors(self):
        dialog = QFileDialog()
        dialog.ensurePolished()
        try:
            for name in ("backButton", "forwardButton", "toParentButton"):
                button = dialog.findChild(QToolButton, name)
                self.assertIsNotNone(button)
                for mode, color in ((QIcon.Mode.Normal, TEXT_PRIMARY),
                                    (QIcon.Mode.Disabled, TEXT_SECONDARY)):
                    with self.subTest(button=name, mode=mode):
                        image = button.icon().pixmap(16, 16, mode).toImage()
                        opaque = [image.pixelColor(x, y)
                                  for x in range(image.width())
                                  for y in range(image.height())
                                  if image.pixelColor(x, y).alpha() > 240]
                        self.assertTrue(opaque, "Navigation glyph must not be empty")
                        expected = QColor(color)
                        self.assertTrue(any(
                            max(abs(pixel.red() - expected.red()),
                                abs(pixel.green() - expected.green()),
                                abs(pixel.blue() - expected.blue())) <= 2
                            for pixel in opaque
                        ), "Navigation glyph must retain its theme color")
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_file_dialog_back_forward_keep_navigation_and_enabled_states(self):
        root = Path(__file__).resolve().parents[1]
        dialog = QFileDialog(directory=str(root))
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        back = dialog.findChild(QToolButton, "backButton")
        forward = dialog.findChild(QToolButton, "forwardButton")
        initial_states = (back.isEnabled(), forward.isEnabled())
        dialog.ensurePolished()
        try:
            # Qt may restore directory history from settings. Styling must not
            # override either state, regardless of that persisted history.
            self.assertEqual((back.isEnabled(), forward.isEnabled()), initial_states)
            dialog.setDirectory(str(root / "tests"))
            self.assertTrue(back.isEnabled())
            back.click()
            self.assertEqual(Path(dialog.directory().absolutePath()), root)
            self.assertTrue(forward.isEnabled())
            forward.click()
            self.assertEqual(Path(dialog.directory().absolutePath()), root / "tests")
            self.assertFalse(forward.isEnabled())
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_main_window_and_existing_frameless_dialog_are_unchanged(self):
        main = QMainWindow()
        popup = QDialog()
        popup.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        for widget in (main, popup):
            flags = widget.windowFlags()
            widget.ensurePolished()
            self.assertEqual(widget.windowFlags(), flags)
            self.assertIsNone(widget.property("applicationDialog"))
            widget.deleteLater()

    def test_message_box_close_does_not_confirm_action(self):
        dialog = QMessageBox()
        dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dialog.setDefaultButton(QMessageBox.No)
        dialog.ensurePolished()
        dialog.show()
        dialog.findChild(QToolButton, "SidebarGhostButton").click()
        self.assertFalse(dialog.isVisible())
        self.assertNotEqual(dialog.result(), QMessageBox.Yes)
        dialog.deleteLater()

    def test_installation_is_idempotent_and_disables_native_dialogs(self):
        previous = self.app._dialog_chrome_filter
        install_dialog_chrome(self.app)
        self.assertIs(previous, self.app._dialog_chrome_filter)
        self.assertTrue(self.app.testAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs))


class MainWindowTitleBarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_window_controls_and_title(self):
        from PySide6.QtWidgets import QLabel
        from PySide6.QtTest import QTest
        from ui.widgets.window_chrome import MainWindowTitleBar

        window = QMainWindow()
        window.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        bar = MainWindowTitleBar(window)
        window.setMenuWidget(bar)
        window.show()
        try:
            window.setWindowTitle("Updated title")
            self.assertEqual(bar.findChild(QLabel, "DialogWindowTitle").text(), "Updated title")
            bar.maximize_button.click()
            self.assertTrue(window.isMaximized())
            self.assertEqual(bar.maximize_button.accessibleName(), "Restore window")
            bar.maximize_button.click()
            self.assertFalse(window.isMaximized())
            QTest.mouseDClick(bar, Qt.MouseButton.LeftButton)
            self.assertTrue(window.isMaximized())
            bar.minimize_button.click()
            self.assertTrue(window.isMinimized())
            window.showNormal()
            bar.findChild(QToolButton, "WindowCloseButton").click()
            self.assertFalse(window.isVisible())
        finally:
            window.close()
            window.deleteLater()
