from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QStyle, QToolButton, QVBoxLayout, QWidget,
)

from ui.theme import build_stylesheet


class _DialogTitleBar(QWidget):
    """Client-side window controls for the application's secondary dialogs."""

    def __init__(self, dialog: QWidget) -> None:
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


class MainWindowTitleBar(_DialogTitleBar):
    """Frameless main-window caption with the usual three window controls."""

    def __init__(self, window: QWidget) -> None:
        super().__init__(window)
        self.setObjectName("MainWindowTitleBar")
        layout = self.layout()
        close = self.findChild(QToolButton, "SidebarGhostButton")
        close.setObjectName("WindowCloseButton")
        close.setText("")
        close.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton))
        close.setAccessibleName("Close window")
        self.minimize_button = QToolButton()
        self.minimize_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMinButton))
        self.minimize_button.setAccessibleName("Minimize window")
        self.minimize_button.setToolTip("Minimize")
        self.minimize_button.clicked.connect(window.showMinimized)
        self.maximize_button = QToolButton()
        self.maximize_button.clicked.connect(self._toggle_maximized)
        layout.insertWidget(layout.count() - 1, self.minimize_button)
        layout.insertWidget(layout.count() - 1, self.maximize_button)
        for button in (self.minimize_button, self.maximize_button, close):
            button.setFixedSize(42, 28)
        window.installEventFilter(self)
        self._sync_window_state()

    def _toggle_maximized(self) -> None:
        if self._dialog.isMaximized():
            self._dialog.showNormal()
        else:
            self._dialog.showMaximized()

    def _sync_window_state(self) -> None:
        maximized = self._dialog.isMaximized()
        icon = QStyle.StandardPixmap.SP_TitleBarNormalButton if maximized else QStyle.StandardPixmap.SP_TitleBarMaxButton
        self.maximize_button.setIcon(self.style().standardIcon(icon))
        label = "Restore window" if maximized else "Maximize window"
        self.maximize_button.setAccessibleName(label)
        self.maximize_button.setToolTip(label)

    def eventFilter(self, watched, event) -> bool:
        if watched is self._dialog and event.type() == QEvent.Type.WindowStateChange:
            self._sync_window_state()
        return False

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = None
            self._toggle_maximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


def build_main_window_header(window: QWidget, menu: QWidget) -> QWidget:
    header = QWidget(window)
    layout = QVBoxLayout(header)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(MainWindowTitleBar(window))
    layout.addWidget(menu)
    return header


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
