from __future__ import annotations

from shiboken6 import isValid

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QWidget,
)


class ComposerFocusController(QObject):
    """Return idle chat focus to the composer after Qt finishes an interaction."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._restore_focus)
        self.attach()
        window.destroyed.connect(self.detach)

    def attach(self) -> None:
        QApplication.instance().installEventFilter(self)

    def detach(self) -> None:
        QApplication.instance().removeEventFilter(self)
        self._timer.stop()

    def schedule(self) -> None:
        if not self._timer.isActive():
            self._timer.start(0)

    def eventFilter(self, watched, event) -> bool:
        if not isValid(self.window):
            return False
        if isinstance(watched, QWidget) and (watched is self.window or self.window.isAncestorOf(watched)):
            event_type = event.type()
            if event_type in (QEvent.FocusIn, QEvent.MouseButtonRelease, QEvent.WindowActivate, QEvent.Hide):
                # Keyboard navigation should still allow controls to receive focus.
                if event_type != QEvent.FocusIn or event.reason() not in (Qt.TabFocusReason, Qt.BacktabFocusReason):
                    self.schedule()
            elif event_type == QEvent.Show and watched is self.window:
                self.schedule()
        return super().eventFilter(watched, event)

    def _restore_focus(self) -> None:
        if not isValid(self.window):
            return
        composer = self.window.composer
        if (
            not self.window.isActiveWindow()
            or not composer.isVisible()
            or not composer.isEnabled()
            or composer.hasFocus()
            or QApplication.activePopupWidget() is not None
            or QApplication.activeModalWidget() is not None
            or QApplication.mouseButtons() != Qt.NoButton
        ):
            return

        focused = QApplication.focusWidget()
        if focused is not None:
            if focused.window() is not self.window or self.window.inspector_container.isAncestorOf(focused):
                return
            # Preserve typing in other editors and copying selected chat text.
            if isinstance(focused, (QLineEdit, QSpinBox, QDoubleSpinBox)):
                return
            if isinstance(focused, QComboBox) and focused.isEditable():
                return
            if isinstance(focused, (QPlainTextEdit, QTextEdit)):
                if not focused.isReadOnly() or focused.textCursor().hasSelection():
                    return
            if isinstance(focused, QLabel) and focused.hasSelectedText():
                return
        composer.setFocus(Qt.OtherFocusReason)
