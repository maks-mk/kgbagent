from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenuBar,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from ui.theme import ERROR_RED, TEXT_PRIMARY
from ui.widgets import _fa_icon


@dataclass(frozen=True)
class MenuBuildResult:
    toggle_sidebar_action: QAction
    new_session_action: QAction
    settings_action: QAction
    info_action: QAction
    quit_action: QAction
    status_text: QLabel
    status_meta: QLabel
    top_status_chip: QLabel
    new_project_button: QToolButton
    sidebar_toggle_button: QToolButton
    new_session_button: QToolButton
    settings_button: QToolButton
    info_button: QToolButton
    menu_widget: QWidget


class MenuBuilder:
    def __init__(self, window) -> None:
        self.window = window

    def build(self) -> MenuBuildResult:
        toggle_sidebar_action = QAction(_fa_icon("fa5s.columns", color=TEXT_PRIMARY, size=14), "Toggle Sidebar", self.window)
        new_session_action = QAction(_fa_icon("fa5s.plus", color=TEXT_PRIMARY, size=14), "New Session", self.window)
        settings_action = QAction(_fa_icon("fa5s.cog", color=TEXT_PRIMARY, size=14), "Settings", self.window)
        info_action = QAction(_fa_icon("fa5s.sliders-h", color=TEXT_PRIMARY, size=14), "Toggle Inspector", self.window)
        quit_action = QAction(_fa_icon("fa5s.sign-out-alt", color=ERROR_RED, size=14), "Quit", self.window)

        toggle_sidebar_action.setShortcut("Ctrl+B")
        new_session_action.setShortcut("Ctrl+N")
        info_action.setShortcut("Ctrl+I")

        for action, tooltip in (
            (toggle_sidebar_action, "Show or hide chat history (Ctrl+B)"),
            (new_session_action, "Start a new session (Ctrl+N)"),
            (settings_action, "Open Settings"),
            (info_action, "Show or hide the inspector (Ctrl+I)"),
            (quit_action, "Quit"),
        ):
            action.setToolTip(tooltip)
            action.setStatusTip(tooltip)

        actual_menu = QMenuBar()
        actual_menu.setNativeMenuBar(False)
        actual_menu.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        menu = actual_menu.addMenu("File")
        menu.addAction(new_session_action)
        menu.addSeparator()
        menu.addAction(quit_action)

        view_menu = actual_menu.addMenu("View")
        view_menu.addAction(toggle_sidebar_action)
        view_menu.addAction(settings_action)
        view_menu.addAction(info_action)

        status_text = QLabel("Initializing runtime…")
        status_text.setObjectName("StatusLabel")
        status_meta = QLabel("")
        status_meta.setObjectName("MetaText")
        top_status_chip = QLabel("Initializing runtime…")
        top_status_chip.setObjectName("TopStatusChip")
        top_status_chip.setAccessibleName("Run status")
        top_status_chip.setAccessibleDescription("Displays the current agent run status")

        new_project_button = self._build_tool_button(
            "fa5s.folder-open",
            tooltip="Open new project folder",
            status_tip="Select a new working directory for the agent",
        )
        sidebar_toggle_button = self._build_tool_button(
            icon=toggle_sidebar_action.icon(),
            tooltip=toggle_sidebar_action.toolTip(),
            status_tip=toggle_sidebar_action.statusTip(),
        )
        new_session_button = self._build_tool_button(
            icon=new_session_action.icon(),
            tooltip=new_session_action.toolTip(),
            status_tip=new_session_action.statusTip(),
        )
        settings_button = self._build_tool_button(
            icon=settings_action.icon(),
            tooltip=settings_action.toolTip(),
            status_tip=settings_action.statusTip(),
        )
        info_button = self._build_tool_button(
            icon=info_action.icon(),
            tooltip=info_action.toolTip(),
            status_tip=info_action.statusTip(),
        )

        right_buttons = QWidget()
        right_buttons.setObjectName("MainWindowTopBarSection")
        right_layout = QHBoxLayout(right_buttons)
        right_layout.setContentsMargins(0, 2, 4, 2)
        right_layout.setSpacing(6)
        right_layout.addWidget(top_status_chip, 0, Qt.AlignVCenter)
        right_layout.addWidget(new_project_button)
        right_layout.addWidget(new_session_button)
        right_layout.addWidget(settings_button)
        right_layout.addWidget(info_button)

        left_controls = QWidget()
        left_controls.setObjectName("MainWindowTopBarSection")
        left_layout = QHBoxLayout(left_controls)
        left_layout.setContentsMargins(0, 2, 4, 2)
        left_layout.setSpacing(4)
        left_layout.addWidget(sidebar_toggle_button)
        left_layout.addWidget(actual_menu)

        top_bar = QWidget()
        top_bar.setObjectName("MainWindowTopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        top_layout.addWidget(left_controls, 0, Qt.AlignVCenter)
        top_layout.addStretch(1)
        top_layout.addWidget(right_buttons, 0, Qt.AlignVCenter)

        return MenuBuildResult(
            toggle_sidebar_action=toggle_sidebar_action,
            new_session_action=new_session_action,
            settings_action=settings_action,
            info_action=info_action,
            quit_action=quit_action,
            status_text=status_text,
            status_meta=status_meta,
            top_status_chip=top_status_chip,
            new_project_button=new_project_button,
            sidebar_toggle_button=sidebar_toggle_button,
            new_session_button=new_session_button,
            settings_button=settings_button,
            info_button=info_button,
            menu_widget=top_bar,
        )

    def _build_tool_button(self, icon=None, *, tooltip: str, status_tip: str) -> QToolButton:
        button = QToolButton()
        button.setObjectName("MainWindowToolButton")
        if isinstance(icon, str):
            button.setIcon(_fa_icon(icon, color=TEXT_PRIMARY, size=14))
        elif icon is not None:
            button.setIcon(icon)
        button.setIconSize(QSize(14, 14))
        button.setAutoRaise(False)
        button.setToolTip(tooltip)
        button.setStatusTip(status_tip)
        return button
