from __future__ import annotations

import json
import re
from typing import Any

from PySide6.QtCore import QSize, QTimer, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget
from shiboken6 import isValid

from core.text_utils import build_tool_ui_labels, format_tool_display
from ui.theme import ERROR_RED, FILENAME_BLUE, SUCCESS_GREEN, TEXT_MUTED
from .foundation import (
    CodeBlockWidget,
    CollapsibleSection,
    CopySafePlainTextEdit,
    DiffBlockWidget,
    ElidedLabel,
    _fa_icon,
    _make_mono_font,
    _sync_plain_text_height,
)

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
FILE_ACCENT_BLUE = FILENAME_BLUE  # Backwards-compatible alias; canonical color lives in ui.theme
TOOL_ICON_SIZE = 12
MCP_TOOL_ICON = "fa5s.cubes"
TOOL_ROLE_ICONS = {
    "write": "fa5s.file-signature",
    "edit": "fa5s.edit",
    "read": "fa5s.file-alt",
    "command": "fa5s.terminal",
    "search": "fa5s.search",
    "list": "fa5s.folder-open",
    "network": "fa5s.globe",
    "fetch": "fa5s.link",
    "crawl": "fa5s.spider",
    "download": "fa5s.download",
    "delete": "fa5s.trash-alt",
    "start_process": "fa5s.play",
    "stop_process": "fa5s.stop",
    "find_process": "fa5s.network-wired",
    "input": "fa5s.comment-dots",
    "tool": "fa5s.tools",
}


def _strip_ansi_for_display(text: Any) -> str:
    return _ANSI_ESCAPE_RE.sub("", str(text or ""))


class CliExecWidget(QFrame):
    def __init__(self, command: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CliExecPanel")
        self.setFrameShape(QFrame.NoFrame)
        self._programmatic_scroll = False
        self._auto_follow_enabled = True
        self._pending_output: list[str] = []
        self._output_flush_timer = QTimer(self)
        self._output_flush_timer.setSingleShot(True)
        self._output_flush_timer.setInterval(0)
        self._output_flush_timer.timeout.connect(self._flush_pending_output)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(10, 8, 10, 6)
        header_row.setSpacing(6)

        self.command_label = ElidedLabel(self, elide_mode=Qt.ElideRight)
        self.command_label.setObjectName("CliExecHeader")
        self.command_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.command_label.setWordWrap(False)
        self.command_label.setMinimumWidth(0)
        header_row.addWidget(self.command_label, 1)

        self.meta_label = QLabel("", self)
        self.meta_label.setObjectName("MetaText")
        self.meta_label.setVisible(False)
        header_row.addWidget(self.meta_label, 0, Qt.AlignRight | Qt.AlignVCenter)
        layout.addLayout(header_row)

        self.output_view = CopySafePlainTextEdit(self)
        self.output_view.setObjectName("CliExecOutput")
        self.output_view.setReadOnly(True)
        self.output_view.setFont(_make_mono_font())
        self.output_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.output_view.setMinimumHeight(96)
        self.output_view.setMaximumHeight(220)
        self.output_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.output_view.document().setMaximumBlockCount(1200)
        self.output_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.output_view.verticalScrollBar().valueChanged.connect(self._on_scroll_value_changed)
        layout.addWidget(self.output_view)

        self.set_command(command)
        self._sync_output_height()

    def set_command(self, command: str) -> None:
        raw = str(command or "")
        stripped = raw.strip()
        compact = " ".join(stripped.split())
        rendered = f"$ {compact}" if compact else "$ "
        self.command_label.set_full_text(rendered)

    def set_meta(self, text: str) -> None:
        value = str(text or "").strip()
        self.meta_label.setText(value)
        severity = "error" if value.lower().startswith("error") else ""
        if self.meta_label.property("severity") != severity:
            self.meta_label.setProperty("severity", severity)
            style = self.meta_label.style()
            if style is not None:
                style.unpolish(self.meta_label)
                style.polish(self.meta_label)
        self.meta_label.setVisible(bool(value))

    def _is_near_bottom(self, threshold: int = 16) -> bool:
        scrollbar = self.output_view.verticalScrollBar()
        return (scrollbar.maximum() - scrollbar.value()) <= max(threshold, scrollbar.pageStep() // 8)

    def _on_scroll_value_changed(self, _value: int) -> None:
        if self._programmatic_scroll:
            return
        self._auto_follow_enabled = self._is_near_bottom()

    def _scroll_to_bottom(self) -> None:
        scrollbar = self.output_view.verticalScrollBar()
        self._programmatic_scroll = True
        scrollbar.setValue(scrollbar.maximum())
        self._programmatic_scroll = False
        self._auto_follow_enabled = True

    def _sync_output_height(self) -> None:
        _sync_plain_text_height(self.output_view, min_lines=6, max_lines=10, extra_padding=18)

    def _flush_pending_output(self) -> None:
        if not self._pending_output:
            return
        chunk = "".join(self._pending_output)
        self._pending_output.clear()
        scrollbar = self.output_view.verticalScrollBar()
        follow = self._auto_follow_enabled or self._is_near_bottom()
        previous_value = scrollbar.value()

        cursor = self.output_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(chunk)
        self.output_view.setTextCursor(cursor)
        self._sync_output_height()

        if follow:
            self._scroll_to_bottom()
            return

        self._programmatic_scroll = True
        scrollbar.setValue(previous_value)
        self._programmatic_scroll = False

    def append_output(self, text: str, stream: str = "stdout") -> None:
        chunk = _strip_ansi_for_display(text)
        if not chunk:
            return
        _ = stream
        self._pending_output.append(chunk)
        if not self._output_flush_timer.isActive():
            self._output_flush_timer.start()

    def ensure_final_output(self, final_text: str) -> None:
        self._output_flush_timer.stop()
        self._flush_pending_output()
        target = _strip_ansi_for_display(final_text)
        if not target:
            return
        current = self.output_view.toPlainText()
        if not current:
            self.output_view.setPlainText(target)
            self._sync_output_height()
            self._scroll_to_bottom()
            return
        if current == target:
            return
        if target.startswith(current):
            self._pending_output.append(target[len(current) :])
            self._flush_pending_output()
            return
        self.output_view.setPlainText(target)
        self._sync_output_height()
        self._scroll_to_bottom()


class _ToolActionLabel(ElidedLabel):
    """Size to the full caption while allowing the layout to elide it."""

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        hint.setWidth(self.fontMetrics().horizontalAdvance(self.full_text()))
        return hint

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())


class ToolCardWidget(QFrame):
    def __init__(self, payload: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ToolRow")
        self.setFrameShape(QFrame.NoFrame)
        self.tool_id = payload.get("tool_id", "")
        self.payload = payload
        self.output_section: CollapsibleSection | None = None
        self.output_view: QPlainTextEdit | None = None
        self.cli_exec_widget: CliExecWidget | None = None
        self._args_expanded = False
        self._is_cli_exec = self._is_cli_exec_name(payload.get("name", ""))
        self._is_mcp = str(payload.get("source_kind", "") or "").strip().lower() == "mcp"
        self._cli_expanded = True
        self._preview_token = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 1)
        layout.setSpacing(2)

        self.header_container = QWidget(self)
        header = QHBoxLayout(self.header_container)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        self.icon_label = QLabel(self.header_container)
        self.icon_label.setFixedSize(TOOL_ICON_SIZE, TOOL_ICON_SIZE)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setPixmap(
            _fa_icon(TOOL_ROLE_ICONS["tool"], color=TEXT_MUTED, size=TOOL_ICON_SIZE).pixmap(
                TOOL_ICON_SIZE,
                TOOL_ICON_SIZE,
            )
        )
        header.addWidget(self.icon_label, 0, Qt.AlignVCenter)

        self.tool_button = QPushButton(self.header_container)
        self.tool_button.setObjectName("ToolCallButton")
        self.tool_button.setCheckable(True)
        self.tool_button.setFlat(True)
        self.tool_button.setText("")
        self.tool_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.tool_button.setFixedSize(16, 18)
        self.tool_button.setCursor(Qt.PointingHandCursor)
        self.tool_button.setIcon(_fa_icon("fa5s.chevron-right", color=TEXT_MUTED, size=8))
        self.tool_button.setIconSize(QSize(8, 8))
        if not self._is_cli_exec:
            self.tool_button.toggled.connect(self._set_args_expanded)
        else:
            self.tool_button.toggled.connect(self._set_cli_expanded)

        self.action_label = _ToolActionLabel(self.header_container, elide_mode=Qt.ElideRight)
        self.action_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.action_label.setObjectName("ToolActionLabel")
        self.action_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.action_label.setMinimumWidth(0)
        self.action_label.setCursor(Qt.PointingHandCursor)
        self.action_label.mousePressEvent = self._handle_action_label_mouse_press  # type: ignore[method-assign]
        header.addWidget(self.action_label, 0, Qt.AlignVCenter)
        header.addWidget(self.tool_button, 0, Qt.AlignVCenter)

        self.phase_badge = QLabel("", self.header_container)
        self.phase_badge.setObjectName("ToolPhaseBadge")
        self.phase_badge.setVisible(False)
        header.addWidget(self.phase_badge, 0, Qt.AlignVCenter)

        header.addStretch(1)

        self.timing_label = QLabel("", self.header_container)
        self.timing_label.setObjectName("MetaText")
        self.timing_label.setVisible(False)
        header.addWidget(self.timing_label, 0, Qt.AlignVCenter | Qt.AlignRight)
        layout.addWidget(self.header_container)

        self.subtitle_label = ElidedLabel(self, elide_mode=Qt.ElideRight)
        self.subtitle_label.setObjectName("ToolSubtitle")
        self.subtitle_label.setVisible(False)
        self.subtitle_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.subtitle_label.setMinimumWidth(0)
        layout.addWidget(self.subtitle_label)

        self.args_view = CopySafePlainTextEdit(self)
        self.args_view.setObjectName("InlineCodeView")
        self.args_view.setReadOnly(True)
        self.args_view.setFont(_make_mono_font())
        self.args_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._set_inline_output(payload.get("content", ""), payload.get("summary", ""))

        self.args_container = QWidget(self)
        args_layout = QHBoxLayout(self.args_container)
        args_layout.setContentsMargins(15, 2, 0, 2)
        args_layout.setSpacing(0)
        args_layout.addWidget(self.args_view, 1)
        self.args_container.setVisible(False)
        layout.addWidget(self.args_container)

        self.diff_section: CollapsibleSection | None = None

        self._apply_payload_visual_state(self.payload)
        if self._is_cli_exec:
            self.args_container.setVisible(False)
            self._ensure_cli_exec_widget()
            self.tool_button.setChecked(True)
            self._set_cli_expanded(True)

    @staticmethod
    def _normalize_args(args: Any) -> dict[str, Any]:
        return dict(args) if isinstance(args, dict) else {}

    @staticmethod
    def _is_cli_exec_name(name: Any) -> bool:
        return str(name or "").strip().lower() == "cli_exec"

    @staticmethod
    def _tool_role(name: Any) -> str:
        normalized = str(name or "").strip()
        if normalized in {"write_file", "Write"}:
            return "write"
        if normalized in {"edit_file", "SearchReplace"}:
            return "edit"
        if normalized in {"read_file", "Read"}:
            return "read"
        if normalized in {"execute", "RunCommand", "cli_exec"}:
            return "command"
        if normalized in {"grep", "Grep", "glob", "Glob"}:
            return "search"
        if normalized in {"ls", "LS", "list_directory"}:
            return "list"
        if normalized == "batch_web_search":
            return "network"
        if normalized in {"fetch_url", "WebFetch", "fetch_content"}:
            return "fetch"
        if normalized == "crawl_site":
            return "crawl"
        if normalized == "download_file":
            return "download"
        if normalized in {"safe_delete_file", "safe_delete_directory"}:
            return "delete"
        if normalized == "run_background_process":
            return "start_process"
        if normalized == "stop_background_process":
            return "stop_process"
        if normalized == "find_process_by_port":
            return "find_process"
        if normalized == "request_user_input":
            return "input"
        return "tool"

    @classmethod
    def _tool_icon_name(cls, name: Any, source_kind: Any = "") -> str:
        if str(source_kind or "").strip().lower() == "mcp":
            return MCP_TOOL_ICON
        return TOOL_ROLE_ICONS[cls._tool_role(name)]

    @classmethod
    def _localized_title(cls, name: Any, payload: dict[str, Any]) -> str:
        normalized_name = str(name or "").strip()
        if str(payload.get("source_kind", "") or "") == "mcp":
            return str(payload.get("display", "") or normalized_name or "MCP")
        role = cls._tool_role(name)
        is_error = bool(payload.get("is_error", False))
        action_titles = {
            "write": "Writing",
            "edit": "Editing",
            "read": "Reading",
            "command": "Running",
            "search": "Searching",
            "list": "Listing",
            "network": "Searching",
            "fetch": "Fetching",
            "crawl": "Crawling",
            "download": "Downloading",
            "delete": "Deleting",
            "start_process": "Starting process",
            "stop_process": "Stopping process",
            "find_process": "Finding process",
            "input": "Requesting input",
        }
        completed_titles = {
            "write": "Wrote",
            "edit": "Edited",
            "read": "Read",
            "command": "Ran",
            "search": "Searched",
            "list": "Listed",
            "network": "Searched",
            "fetch": "Fetched",
            "crawl": "Crawled",
            "download": "Downloaded",
            "delete": "Deleted",
            "start_process": "Started process",
            "stop_process": "Stopped process",
            "find_process": "Found process",
            "input": "Requested input",
        }
        phase = str(payload.get("phase", "running") or "running")
        titles = action_titles if is_error or phase != "finished" else completed_titles
        title = titles.get(role)
        if title:
            return f"{title} failed" if is_error else title
        if is_error:
            return "Tool failed"
        return str(payload.get("display", "") or name or "Tool")

    def _set_visual_property(self, widget: QWidget, name: str, value: str) -> None:
        if widget.property(name) == value:
            return
        widget.setProperty(name, value)
        style = widget.style()
        if style is not None:
            style.unpolish(widget)
            style.polish(widget)

    @staticmethod
    def _command_from_args(args: dict[str, Any]) -> str:
        return str(args.get("command", "") or "")

    @staticmethod
    def _command_from_display(display: Any, tool_name: Any) -> str:
        text = str(display or "").strip()
        name = str(tool_name or "").strip() or "cli_exec"
        prefix = f"{name}("
        if text.startswith(prefix) and text.endswith(")"):
            inner = text[len(prefix):-1].strip()
            if len(inner) >= 2 and inner[:1] == inner[-1:] and inner[:1] in {'"', "'"}:
                inner = inner[1:-1]
            return inner
        return ""

    @staticmethod
    def _format_duration(duration: Any) -> str:
        try:
            value = float(duration)
        except (TypeError, ValueError):
            return ""
        if value < 0:
            value = 0.0
        if value < 1.0:
            milliseconds = int(round(value * 1000))
            if milliseconds <= 0 and value > 0:
                return "<1ms"
            return f"{milliseconds}ms"
        return f"{value:.1f}s"

    @staticmethod
    def _diff_stat_parts(diff_text: Any) -> tuple[str, str]:
        added = 0
        removed = 0
        for raw_line in str(diff_text or "").splitlines():
            if raw_line.startswith("+++") or raw_line.startswith("---"):
                continue
            if raw_line.startswith("+"):
                added += 1
            elif raw_line.startswith("-"):
                removed += 1
        if added or removed:
            return f"+{added}", f"-{removed}"
        return "", ""

    @classmethod
    def _diff_stats(cls, diff_text: Any) -> str:
        added, removed = cls._diff_stat_parts(diff_text)
        return f"{added} {removed}".strip()

    @staticmethod
    def _compact_single_line(value: Any) -> str:
        return " ".join(str(value or "").split())

    @staticmethod
    def _is_argument_placeholder(value: str) -> bool:
        normalized = str(value or "").strip().rstrip(".…").casefold()
        return normalized in {"waiting for arguments", "resolving arguments"}

    def _compact_argument_text(self, payload: dict[str, Any], normalized_args: dict[str, Any]) -> str:
        role = self._tool_role(payload.get("name", ""))
        if role == "command":
            command = self._command_from_args(normalized_args)
            if not command:
                command = self._command_from_display(payload.get("raw_display", ""), payload.get("name", "cli_exec"))
            return self._compact_single_line(command)
        subtitle = self._compact_single_line(payload.get("subtitle", ""))
        if self._is_argument_placeholder(subtitle):
            subtitle = ""
        if role in {"write", "edit"}:
            stats = self._diff_stats(payload.get("diff", ""))
            return f"{subtitle} {stats}".strip() if stats else subtitle
        if role in {"network", "fetch", "download"}:
            return subtitle
        if subtitle:
            return subtitle
        raw_display = self._compact_single_line(payload.get("raw_display", ""))
        tool_name = self._compact_single_line(payload.get("name", ""))
        if raw_display == tool_name:
            return ""
        display = self._compact_single_line(payload.get("display", ""))
        return "" if raw_display == display else raw_display

    def _compact_action_line(self, title: str, argument_text: str) -> str:
        title = self._compact_single_line(title)
        argument_text = self._compact_single_line(argument_text)
        if argument_text:
            return f"{title} {argument_text}"
        return title

    def _action_line_segments(
        self,
        title: str,
        payload: dict[str, Any],
        normalized_args: dict[str, Any],
    ) -> list[tuple[str, str]]:
        role = self._tool_role(payload.get("name", ""))
        if role not in {"write", "edit", "read", "list", "delete"}:
            return []
        target = self._compact_single_line(payload.get("subtitle", ""))
        if self._is_argument_placeholder(target):
            target = ""
        if not target:
            path_value = normalized_args.get("file_path") or normalized_args.get("path") or normalized_args.get("dir_path")
            target = self._compact_single_line(path_value)
        if not target:
            return []
        segments: list[tuple[str, str]] = [(f"{self._compact_single_line(title)} ", ""), (target, FILE_ACCENT_BLUE)]
        if role in {"write", "edit"}:
            added, removed = self._diff_stat_parts(payload.get("diff", ""))
            if added:
                segments.extend([(" ", ""), (added, SUCCESS_GREEN)])
            if removed:
                segments.extend([(" ", ""), (removed, ERROR_RED)])
        return segments

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized_args = self._normalize_args(payload.get("args", {}))
        normalized = dict(payload)
        normalized["args"] = normalized_args
        labels = build_tool_ui_labels(
            str(normalized.get("name", "") or ""),
            normalized_args,
            phase=str(normalized.get("phase", "running") or "running"),
            is_error=bool(normalized.get("is_error", False)),
        )
        normalized.setdefault("display", labels.get("title") or normalized.get("name", "tool"))
        normalized.setdefault("subtitle", labels.get("subtitle", ""))
        normalized.setdefault("raw_display", labels.get("raw_display") or format_tool_display(str(normalized.get("name", "")), normalized_args))
        normalized.setdefault("args_state", labels.get("args_state", "pending"))
        normalized.setdefault("source_kind", labels.get("source_kind", "tool"))
        normalized.setdefault(
            "display_state",
            "finished" if normalized.get("phase") == "finished" else ("resolved" if normalized.get("args_state") == "complete" else "preview"),
        )
        return normalized

    @staticmethod
    def _phase_badge_text(payload: dict[str, Any]) -> str:
        if payload.get("is_error"):
            return "Error"
        phase = str(payload.get("phase", "running") or "running")
        args_state = str(payload.get("args_state", "complete") or "complete")
        if phase == "finished":
            return "✓"
        if phase == "preparing" and args_state in {"pending", "partial"}:
            return "Preparing"
        return "Running"

    @staticmethod
    def _phase_variant(payload: dict[str, Any]) -> str:
        if payload.get("is_error"):
            return "error"
        phase = str(payload.get("phase", "running") or "running")
        args_state = str(payload.get("args_state", "complete") or "complete")
        if phase == "finished":
            return "success"
        if phase == "preparing" and args_state in {"pending", "partial"}:
            return "pending"
        return "active"

    def _set_tool_visible(self, visible: bool) -> None:
        if not isValid(self):
            return
        if self.isHidden() == (not visible):
            return
        self.setVisible(visible)

    def _cancel_preview_reveal(self) -> None:
        self._preview_token += 1

    def _queue_preview_reveal(self) -> None:
        token = self._preview_token + 1
        self._preview_token = token
        self._set_tool_visible(False)

    def _apply_payload_visual_state(self, payload: dict[str, Any], *, finished: bool = False) -> None:
        normalized = self._normalize_payload(payload)
        self.payload = normalized
        normalized_args = self._normalize_args(normalized.get("args", {}))

        tool_name = str(normalized.get("name", "") or "")
        tool_role = self._tool_role(tool_name)
        display = self._localized_title(tool_name, normalized).strip()
        subtitle = str(normalized.get("subtitle", "") or "").strip()
        raw_display = str(normalized.get("raw_display", "") or "").strip()
        phase_variant = self._phase_variant(normalized)
        argument_text = self._compact_argument_text(normalized, normalized_args)
        action_line = self._compact_action_line(display, argument_text)
        action_segments = self._action_line_segments(display, normalized, normalized_args)

        self._set_visual_property(self, "toolRole", tool_role)
        self._set_visual_property(self.tool_button, "toolRole", tool_role)
        self._set_visual_property(self.tool_button, "phase", phase_variant)
        self._set_visual_property(self.action_label, "toolRole", tool_role)
        self._set_visual_property(self.action_label, "phase", phase_variant)
        self.tool_button.setToolTip(raw_display if raw_display and raw_display != display else action_line)
        if action_segments:
            self.action_label.set_rich_segments(action_segments)
        else:
            self.action_label.set_full_text(action_line)
        self.action_label.updateGeometry()
        self.action_label.setToolTip(raw_display if raw_display and raw_display != action_line else action_line)

        if finished:
            subtitle = self._compact_argument_text(normalized, normalized_args)
        self.subtitle_label.set_full_text(subtitle)
        self.subtitle_label.setToolTip(raw_display if raw_display and raw_display != subtitle else subtitle)
        self.subtitle_label.setVisible(False)

        badge_text = self._phase_badge_text(normalized)
        self.phase_badge.setText(badge_text)
        self.phase_badge.setVisible(False)
        if self.phase_badge.property("variant") != phase_variant:
            self.phase_badge.setProperty("variant", phase_variant)
            style = self.phase_badge.style()
            if style is not None:
                style.unpolish(self.phase_badge)
                style.polish(self.phase_badge)

        icon_name = self._tool_icon_name(tool_name, normalized.get("source_kind", ""))
        is_mcp = str(normalized.get("source_kind", "") or "").strip().lower() == "mcp"
        self._is_mcp = is_mcp
        icon_color = TEXT_MUTED
        if phase_variant == "active":
            icon_color = "#A8A49E"
        elif phase_variant == "success":
            icon_color = SUCCESS_GREEN
        elif phase_variant == "error":
            icon_color = ERROR_RED
        self.icon_label.setPixmap(
            _fa_icon(icon_name, color=icon_color, size=TOOL_ICON_SIZE).pixmap(
                TOOL_ICON_SIZE,
                TOOL_ICON_SIZE,
            )
        )

        if normalized.get("display_state") == "preview" and not normalized.get("refresh") and not finished:
            self._queue_preview_reveal()
        else:
            self._cancel_preview_reveal()
            self._set_tool_visible(True)

        if is_mcp and not self._is_cli_exec:
            self._set_inline_output_mcp_height()

    def _ensure_cli_exec_widget(self) -> CliExecWidget:
        if self.cli_exec_widget is None:
            command = self._command_from_args(self._normalize_args(self.payload.get("args", {})))
            self.cli_exec_widget = CliExecWidget(command, parent=self)
            self.layout().addWidget(self.cli_exec_widget)
        return self.cli_exec_widget

    @staticmethod
    def _render_inline_output(content: Any, summary: Any = "") -> str:
        if isinstance(content, (dict, list)):
            rendered = json.dumps(content, ensure_ascii=False, indent=2)
        else:
            rendered = _strip_ansi_for_display(content)
        if rendered.strip():
            return rendered
        return _strip_ansi_for_display(summary)

    def _set_inline_output(self, content: Any, summary: Any = "") -> None:
        rendered = self._render_inline_output(content, summary)
        if self.args_view.toPlainText() != rendered:
            self.args_view.setPlainText(rendered)
        min_lines = 6 if self._is_mcp else 4
        _sync_plain_text_height(self.args_view, min_lines=min_lines, max_lines=10, extra_padding=14)

    def _set_inline_output_mcp_height(self) -> None:
        _sync_plain_text_height(self.args_view, min_lines=6, max_lines=10, extra_padding=14)

    def _uses_diff_only_output(self) -> bool:
        return self._tool_role(self.payload.get("name", "")) in {"write", "edit"} and bool(self.payload.get("diff", ""))

    def _diff_expanded_by_default(self) -> bool:
        return str(self.payload.get("name", "") or "").strip() in {"write_file", "Write"}

    def append_cli_output(self, text: str, stream: str = "stdout") -> None:
        if not text:
            return
        if not self._is_cli_exec:
            return
        self._cancel_preview_reveal()
        self._set_tool_visible(True)
        self.tool_button.setChecked(True)
        self._set_cli_expanded(True)
        self._ensure_cli_exec_widget().append_output(text, stream=stream)

    def update_started_payload(self, payload: dict[str, Any]) -> None:
        merged_payload = dict(self.payload)
        merged_payload.update(payload)
        self._apply_payload_visual_state(merged_payload)
        self._set_inline_output(self.payload.get("content", ""), self.payload.get("summary", ""))

        self._is_cli_exec = self._is_cli_exec or self._is_cli_exec_name(self.payload.get("name", ""))
        if self._is_cli_exec:
            cli_exec_widget = self._ensure_cli_exec_widget()
            command = self._command_from_args(self._normalize_args(self.payload.get("args", {})))
            if not command:
                command = self._command_from_display(self.payload.get("raw_display", ""), self.payload.get("name", "cli_exec"))
            if command:
                cli_exec_widget.set_command(command)
            self.tool_button.setChecked(True)
            self._set_cli_expanded(True)

    def finish(self, payload: dict[str, Any]) -> None:
        merged_payload = dict(self.payload)
        merged_payload.update(payload)
        merged_payload["phase"] = "finished"
        merged_payload["display_state"] = "finished"
        self._apply_payload_visual_state(merged_payload, finished=True)
        normalized_args = self._normalize_args(self.payload.get("args", {}))
        is_error = bool(self.payload.get("is_error", False))
        self._set_inline_output(self.payload.get("content", ""), self.payload.get("summary", ""))

        duration = self.payload.get("duration")
        duration_text = self._format_duration(duration)
        if duration_text:
            self.timing_label.setText(duration_text)
            self.timing_label.setVisible(True)

        self._is_cli_exec = self._is_cli_exec or self._is_cli_exec_name(self.payload.get("name", ""))
        if self._is_cli_exec:
            cli_exec_widget = self._ensure_cli_exec_widget()
            cli_exec_widget.set_command(self._command_from_args(normalized_args))
            status_text = ""
            if duration_text:
                status_text = duration_text
            if is_error:
                status_text = f"error · {status_text}" if status_text else "error"
            cli_exec_widget.set_meta(status_text)
            cli_exec_widget.ensure_final_output(str(self.payload.get("content", "") or ""))
            self.tool_button.setChecked(False)
            self._set_cli_expanded(False)
            return

        diff_text = self.payload.get("diff", "")
        if diff_text and self.diff_section is None:
            diff_expanded = self._diff_expanded_by_default()
            diff_widget = DiffBlockWidget(
                diff_text,
                source_path=str(normalized_args.get("path", "") or ""),
                parent=self,
            )
            self.diff_section = CollapsibleSection(
                "Edited file",
                diff_widget,
                expanded=diff_expanded,
                content_margins=(0, 2, 0, 8),
                parent=self,
            )
            self.diff_section.toggle_button.setVisible(False)
            self.diff_section.setProperty("variant", "diff")
            self.diff_section.toggle_button.setProperty("variant", "diff")
            self.diff_section.content_container.setProperty("variant", "diff")
            self.layout().addWidget(self.diff_section)
        elif diff_text and self.diff_section is not None:
            if isinstance(self.diff_section.content, DiffBlockWidget):
                self.diff_section.content.set_diff(diff_text, source_path=str(normalized_args.get("path", "") or ""))
            elif isinstance(self.diff_section.content, CodeBlockWidget):
                self.diff_section.content.set_code(diff_text, "diff")

        if not self._is_cli_exec:
            if self._uses_diff_only_output():
                diff_expanded = self._diff_expanded_by_default()
                self.tool_button.setChecked(diff_expanded)
                self._set_args_expanded(diff_expanded)
            else:
                self.tool_button.setChecked(False)
                self._set_args_expanded(False)

    def _set_args_expanded(self, expanded: bool) -> None:
        if self._is_cli_exec:
            return
        if self._uses_diff_only_output():
            self._args_expanded = expanded
            self.tool_button.setIcon(
                _fa_icon("fa5s.chevron-down" if expanded else "fa5s.chevron-right", color=TEXT_MUTED, size=8)
            )
            self.args_container.setVisible(False)
            if self.diff_section is not None:
                self.diff_section.set_expanded(expanded)
            return
        self._args_expanded = expanded
        self.tool_button.setIcon(
            _fa_icon("fa5s.chevron-down" if expanded else "fa5s.chevron-right", color=TEXT_MUTED, size=8)
        )
        self.args_container.setVisible(expanded)

    def _set_cli_expanded(self, expanded: bool) -> None:
        if not self._is_cli_exec:
            return
        self._cli_expanded = expanded
        self.tool_button.setIcon(
            _fa_icon("fa5s.chevron-down" if expanded else "fa5s.chevron-right", color=TEXT_MUTED, size=8)
        )
        if self.cli_exec_widget is not None:
            self.cli_exec_widget.setVisible(expanded)

    def _handle_action_label_mouse_press(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.tool_button.click()
            event.accept()
            return
        event.ignore()
