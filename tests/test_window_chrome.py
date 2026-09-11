import unittest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QMessageBox, QToolButton

from ui.theme import build_stylesheet
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
