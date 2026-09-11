from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QToolButton, QWidget,
)

from ui.theme import build_stylesheet


class _DialogTitleBar(QWidget):
    """Client-side window controls for the application's secondary dialogs."""

    def __init__(self, dialog: QDialog) -> None:
        super().__init__(dialog)
        self.setObjectName("DialogTitleBar")
        self._dialog = dialog
        self._drag_offset: QPoint | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 6, 6)
        title = QLabel(dialog.windowTitle())
        title.setObjectName("DialogWindowTitle")
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        dialog.windowTitleChanged.connect(title.setText)
        layout.addWidget(title, 1)
        close = QToolButton()
        close.setObjectName("SidebarGhostButton")
        close.setText("×")
        close.setAccessibleName("Close dialog")
        close.setToolTip("Close")
        close.clicked.connect(dialog.close)
        layout.addWidget(close)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self._dialog.windowHandle()
            if handle is None or not handle.startSystemMove():
                self._drag_offset = event.globalPosition().toPoint() - self._dialog.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._dialog.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)


class _DialogChromeFilter(QObject):
    def eventFilter(self, watched, event) -> bool:
        # Polish runs after construction and before first display. Changing
        # flags on Show would hide a dialog that is already being displayed.
        if (
            event.type() == QEvent.Type.Polish
            and isinstance(watched, QDialog)
            and watched.isWindow()
            and not watched.windowFlags() & Qt.WindowType.FramelessWindowHint
        ):
            watched.setWindowFlag(Qt.WindowType.FramelessWindowHint)
            watched.setProperty("applicationDialog", True)
            watched.setStyleSheet(build_stylesheet())
            layout = watched.layout()
            if layout is not None and layout.menuBar() is None:
                layout.setMenuBar(_DialogTitleBar(watched))
            if isinstance(watched, QFileDialog):
                watched.setSizeGripEnabled(True)
        return False


def install_dialog_chrome(app: QApplication) -> None:
    """Include static QMessageBox/QFileDialog helpers without replacing their APIs."""
    if getattr(app, "_dialog_chrome_filter", None) is not None:
        return
    app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs)
    event_filter = _DialogChromeFilter(app)
    app.installEventFilter(event_filter)
    app._dialog_chrome_filter = event_filter
