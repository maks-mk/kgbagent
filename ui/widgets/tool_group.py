from __future__ import annotations

from collections import Counter

from PySide6.QtCore import QAbstractAnimation, QEasingCurve, QPropertyAnimation, QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .foundation import _fa_icon
from .tools import ToolCardWidget
from ui.theme import ERROR_RED, SUCCESS_GREEN, TEXT_MUTED


class ToolGroupWidget(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ToolGroupFrame")
        self.setFrameShape(QFrame.NoFrame)
        self._tools: list[ToolCardWidget] = []
        self._collapsed = False
        self._completed = False
        self._completion_announced = False
        self._animation_target_expanded = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 2)
        layout.setSpacing(2)

        self.header_row = QWidget(self)
        self.header_row.setObjectName("ToolGroupHeaderRow")
        header_layout = QHBoxLayout(self.header_row)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        self.header_btn = QPushButton(self)
        self.header_btn.setObjectName("ToolGroupHeaderButton")
        self.header_btn.setCheckable(True)
        self.header_btn.setFlat(True)
        self.header_btn.setChecked(True)
        self.header_btn.setCursor(Qt.PointingHandCursor)
        self.header_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.header_btn.setMinimumWidth(0)
        self.header_btn.setIconSize(QSize(9, 9))
        self.header_btn.setAccessibleName("Tool results group")
        self.header_btn.setAccessibleDescription("Expand or collapse the tool results for this turn")
        self.header_btn.clicked.connect(self._toggle)
        header_layout.addWidget(self.header_btn, 0)

        self.error_icon_label = QLabel(self.header_row)
        self.error_icon_label.setObjectName("MetaText")
        self.error_icon_label.setPixmap(_fa_icon("fa5s.times-circle", color=ERROR_RED, size=9).pixmap(9, 9))
        self.error_icon_label.setVisible(False)
        header_layout.addWidget(self.error_icon_label, 0, Qt.AlignVCenter)

        self.error_count_label = QLabel("", self.header_row)
        self.error_count_label.setObjectName("MetaText")
        self.error_count_label.setProperty("severity", "error")
        self.error_count_label.setVisible(False)
        header_layout.addWidget(self.error_count_label, 0, Qt.AlignVCenter)

        self.expand_button = QPushButton(self.header_row)
        self.expand_button.setObjectName("ToolCallButton")
        self.expand_button.setFlat(True)
        self.expand_button.setFixedSize(16, 18)
        self.expand_button.setIconSize(QSize(8, 8))
        self.expand_button.setCursor(Qt.PointingHandCursor)
        self.expand_button.setAccessibleName("Expand or collapse tool results")
        self.expand_button.clicked.connect(self.header_btn.click)
        header_layout.insertWidget(1, self.expand_button, 0, Qt.AlignVCenter)
        header_layout.addStretch(1)

        layout.addWidget(self.header_row)

        self.container = QWidget(self)
        self.container.setObjectName("ToolGroupContainer")
        self.inner = QVBoxLayout(self.container)
        self.inner.setContentsMargins(10, 1, 0, 0)
        self.inner.setSpacing(0)
        layout.addWidget(self.container)

        self._container_animation = QPropertyAnimation(self.container, b"maximumHeight", self)
        self._container_animation.setDuration(180)
        self._container_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._container_animation.finished.connect(self._finish_container_animation)

        self._sync_header()

    @staticmethod
    def _pluralize(value: int, singular: str, plural: str) -> str:
        return singular if abs(int(value)) == 1 else plural

    @staticmethod
    def _tool_role(card: ToolCardWidget) -> str:
        name = str(card.payload.get("name", "") or "").strip()
        if name in {"write_file", "Write"}:
            return "write"
        if name in {"edit_file", "SearchReplace"}:
            return "edit"
        if name in {"read_file", "Read"}:
            return "read"
        if name in {"execute", "RunCommand", "cli_exec"}:
            return "command"
        if name in {"grep", "Grep", "glob", "Glob"}:
            return "search"
        if name == "batch_web_search":
            return "network"
        if name in {"fetch_url", "WebFetch", "fetch_content"}:
            return "fetch"
        if name == "crawl_site":
            return "crawl"
        if name == "download_file":
            return "download"
        if name in {"ls", "LS", "list_directory"}:
            return "list"
        if name in {"safe_delete_file", "safe_delete_directory"}:
            return "delete"
        if name == "run_background_process":
            return "start_process"
        if name == "stop_background_process":
            return "stop_process"
        if name == "find_process_by_port":
            return "find_process"
        if name == "request_user_input":
            return "input"
        return "tool"

    def _group_role(self) -> str:
        roles = Counter(self._tool_role(tool) for tool in self._tools)
        if not roles:
            return "tool"
        if len(roles) == 1:
            return next(iter(roles))
        if roles.get("write", 0) + roles.get("edit", 0) == len(self._tools):
            return "edit"
        if roles.get("command", 0) == len(self._tools):
            return "command"
        return "tool"

    def _header_text(self, *, completed: bool) -> str:
        total = len(self._tools)
        if total <= 0:
            return "Running"
        role = self._group_role()
        role_titles = {
            "write": ("Writing", "Wrote", "file", "files"),
            "edit": ("Editing", "Edited", "file", "files"),
            "read": ("Reading", "Read", "file", "files"),
            "command": ("Running", "Ran", "command", "commands"),
        }
        titles = role_titles.get(role)
        if titles:
            action_title, completed_title, singular, plural = titles
            title = completed_title if completed else action_title
            noun = self._pluralize(total, singular, plural)
            return f"{title} {total} {noun}"
        action_titles = {
            "search": "Searching",
            "network": "Searching",
            "fetch": "Fetching",
            "crawl": "Crawling",
            "download": "Downloading",
            "list": "Listing",
            "delete": "Deleting",
            "start_process": "Starting process",
            "stop_process": "Stopping process",
            "find_process": "Finding process",
            "input": "Requesting input",
        }
        completed_titles = {
            "search": "Searched",
            "network": "Searched",
            "fetch": "Fetched",
            "crawl": "Crawled",
            "download": "Downloaded",
            "list": "Listed",
            "delete": "Deleted",
            "start_process": "Started process",
            "stop_process": "Stopped process",
            "find_process": "Found process",
            "input": "Requested input",
        }
        title = (completed_titles if completed else action_titles).get(role)
        if title:
            return title
        noun = self._pluralize(total, "tool", "tools")
        return f"Completed {total} {noun}" if completed else f"Running {total} {noun}"

    def _error_header_text(self, errors: int) -> str:
        total = len(self._tools)
        role = self._group_role()
        if total == 1:
            return {
                "write": "Writing failed",
                "edit": "Editing failed",
                "read": "Reading failed",
                "command": "Running failed",
                "search": "Searching failed",
                "network": "Searching failed",
                "fetch": "Fetching failed",
                "crawl": "Crawling failed",
                "download": "Downloading failed",
                "list": "Listing failed",
                "delete": "Deleting failed",
                "start_process": "Starting process failed",
                "stop_process": "Stopping process failed",
                "find_process": "Finding process failed",
                "input": "Requesting input failed",
            }.get(role, "Tool failed")
        return f"Completed {total} tools with {errors} errors"

    def _set_header_state(self, *, state: str) -> None:
        if self.header_btn.property("state") == state:
            return
        self.header_btn.setProperty("state", state)
        style = self.header_btn.style()
        if style is not None:
            style.unpolish(self.header_btn)
            style.polish(self.header_btn)

    def add_tool(self, card: ToolCardWidget) -> None:
        if card in self._tools:
            return
        self._tools.append(card)
        self.inner.addWidget(card)
        if self._completed:
            self._completed = False
            self._completion_announced = False
            self.expand()
        else:
            self._sync_header()

    @staticmethod
    def _tool_is_finished(card: ToolCardWidget) -> bool:
        return str(card.payload.get("phase", "running") or "running") == "finished"

    def refresh_completion(self, *, auto_collapse: bool = False) -> None:
        self._completed = bool(self._tools) and all(self._tool_is_finished(tool) for tool in self._tools)
        if not self._completed:
            self._completion_announced = False
        if self._completed and auto_collapse:
            self._completion_announced = True
            self._collapsed = True
            self._set_container_expanded(False)
            self.header_btn.setChecked(False)
        elif self._completed:
            self._completion_announced = True
        self._sync_header()

    def _set_container_expanded(self, expanded: bool, *, animated: bool = True) -> None:
        target_height = self.container.sizeHint().height()
        if target_height <= 0:
            self.container.setMaximumHeight(0 if not expanded else 16777215)
            self.container.setVisible(expanded)
            return

        animation = self._container_animation
        if animation.state() != QAbstractAnimation.Stopped:
            animation.stop()

        was_visible = self.container.isVisible()
        current_height = self.container.height()
        self._animation_target_expanded = expanded
        if not animated or not self.isVisible():
            self.container.setMaximumHeight(16777215 if expanded else 0)
            self.container.setVisible(expanded)
            return

        if expanded:
            self.container.setVisible(True)
            start_height = max(0, current_height) if was_visible else 0
            self.container.setMaximumHeight(start_height)
            end_height = target_height
        else:
            start_height = max(0, current_height)
            end_height = 0
            self.container.setMaximumHeight(start_height)

        animation.setStartValue(start_height)
        animation.setEndValue(end_height)
        animation.start()

    def _finish_container_animation(self) -> None:
        expanded = self._animation_target_expanded
        self.container.setMaximumHeight(16777215 if expanded else 0)
        self.container.setVisible(expanded)

    def collapse(self) -> None:
        self._completed = bool(self._tools) and all(self._tool_is_finished(tool) for tool in self._tools)
        if self._completed:
            self._completion_announced = True
        if self._collapsed:
            self._sync_header()
            return
        self._collapsed = True
        self._set_container_expanded(False)
        self.header_btn.setChecked(False)
        self._sync_header()

    def expand(self) -> None:
        self._collapsed = False
        self._set_container_expanded(True)
        self.header_btn.setChecked(True)
        self._sync_header()

    def _toggle(self, checked: bool = False) -> None:
        self._collapsed = not checked
        self._set_container_expanded(checked)
        if self._collapsed and self._completed:
            self._completion_announced = True
        self._sync_header()

    def _sync_header(self) -> None:
        expanded = not self._collapsed
        self.expand_button.setIcon(
            _fa_icon("fa5s.chevron-down" if expanded else "fa5s.chevron-right", color=TEXT_MUTED, size=8)
        )
        if self._completion_announced:
            total = len(self._tools)
            errors = sum(1 for tool in self._tools if tool.payload.get("is_error", False))
            self.error_icon_label.setVisible(errors > 0)
            self.error_count_label.setText(str(errors) if errors > 0 else "")
            self.error_count_label.setVisible(errors > 0)
            self._set_header_state(state="error" if errors > 0 else "complete")
            if errors > 0:
                self.header_btn.setIcon(_fa_icon("fa5s.check-circle", color=SUCCESS_GREEN, size=9))
                error_title = self._error_header_text(errors)
                self.header_btn.setText(error_title)
            else:
                self.header_btn.setIcon(_fa_icon("fa5s.check-circle", color=SUCCESS_GREEN, size=9))
                self.header_btn.setText(self._header_text(completed=True))
            return
        self.error_icon_label.setVisible(False)
        self.error_count_label.setVisible(False)
        self.error_count_label.clear()
        self._set_header_state(state="active")
        self.header_btn.setIcon(QIcon())
        self.header_btn.setText(self._header_text(completed=False))
