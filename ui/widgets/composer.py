from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QFileSystemWatcher, QPoint, QMimeData, Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent, QPainter, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

from core.multimodal import can_read_image_file
from .foundation import (
    COMPOSER_MENTION_EXCLUDED_DIRS,
    COMPOSER_MENTION_MAX_ITEMS,
    COMPOSER_MENTION_POPUP_MAX_WIDTH,
    COMPOSER_MENTION_POPUP_MIN_WIDTH,
    _fa_icon,
)
from ui.theme import ACCENT, TEXT_WARNING, TEXT_SUCCESS, TEXT_MUTED, FILE_PDF_ICON_TEXT


class ComposerTextEdit(QPlainTextEdit):
    submit_requested = Signal()
    image_pasted = Signal(object)
    image_files_pasted = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ComposerEdit")
        self._history_by_session: dict[str, list[str]] = {}
        self._history_session_id = "__default__"
        self._history_nav_index = 0
        self._file_index: list[dict[str, Any]] = []
        self._file_index_root = ""
        self._file_index_static = False
        self._file_index_last_scan_at = 0.0
        self._file_index_dirty = True
        self._file_index_watcher = QFileSystemWatcher(self)
        self._file_index_watcher.directoryChanged.connect(self._on_file_index_directory_changed)
        self._mention_popup: _ComposerMentionPopup | None = None
        self.textChanged.connect(self._refresh_mention_popup)
        self.cursorPositionChanged.connect(self._refresh_mention_popup)
        self.set_history_session("")
        QTimer.singleShot(0, self._warm_file_index)

    def set_history_session(self, session_id: str) -> None:
        self._history_session_id = session_id or "__default__"
        self._history_by_session.setdefault(self._history_session_id, [])
        self._reset_history_navigation()
        self._close_mention_popup()
        QTimer.singleShot(0, self._warm_file_index)

    def sync_session_history_from_transcript(self, session_id: str, transcript_payload: dict[str, Any] | None) -> None:
        key = session_id or "__default__"
        payload = transcript_payload if isinstance(transcript_payload, dict) else {}
        turns = payload.get("turns", []) or []
        history: list[str] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            user_text = str(turn.get("user_text", "") or "").strip()
            if not user_text:
                continue
            if not history or history[-1] != user_text:
                history.append(user_text)
        self._history_by_session[key] = history
        if key == self._history_session_id:
            self._reset_history_navigation()

    def append_submitted_message(self, text: str) -> None:
        value = str(text or "").strip()
        if not value:
            return
        history = self._history_for_session()
        if not history or history[-1] != value:
            history.append(value)
        self._reset_history_navigation()

    def clear_history_for_session(self, session_id: str | None = None) -> None:
        key = (session_id or self._history_session_id) or "__default__"
        self._history_by_session[key] = []
        if key == self._history_session_id:
            self._reset_history_navigation()

    def reset_history_navigation(self) -> None:
        self._reset_history_navigation()

    def format_file_reference(self, path_value: str | Path) -> str:
        path = Path(path_value)
        cwd = Path.cwd()
        try:
            normalized = str(path.relative_to(cwd).as_posix())
        except ValueError:
            normalized = str(path.as_posix())
        if " " in normalized:
            return f'"{normalized}"'
        return normalized

    def set_file_index_for_testing(self, rel_paths: list[str]) -> None:
        root = Path.cwd()
        self._file_index_root = str(root)
        self._file_index_static = True
        self._file_index_dirty = False
        self._clear_file_index_watcher()
        rows: dict[str, dict[str, Any]] = {}
        for value in rel_paths:
            relative_path = Path(value)
            self._add_index_row(rows, self._build_file_index_row(root, root / relative_path))
            for parent in relative_path.parents:
                if str(parent) in {"", "."}:
                    continue
                self._add_index_row(rows, self._build_file_index_row(root, root / parent, is_dir=True))
        self._file_index = list(rows.values())
        self._file_index.sort(key=lambda row: row["relative"])

    def keyPressEvent(self, event: QKeyEvent) -> None:
        popup = self._mention_popup
        if popup is not None and popup.isVisible():
            key = event.key()
            if key == Qt.Key_Up:
                event.accept()
                popup.move_selection(-1)
                return
            if key == Qt.Key_Down:
                event.accept()
                popup.move_selection(1)
                return
            if key in {Qt.Key_Return, Qt.Key_Enter} and not (event.modifiers() & Qt.ShiftModifier):
                event.accept()
                self._accept_current_mention()
                return
            if key == Qt.Key_Escape:
                event.accept()
                self._close_mention_popup()
                return

        history = self._history_for_session()
        is_browsing_history = 0 <= self._history_nav_index < len(history)
        if event.key() in {Qt.Key_Up, Qt.Key_Down} and (not self.toPlainText() or is_browsing_history):
            if self._navigate_history(-1 if event.key() == Qt.Key_Up else 1):
                event.accept()
                return

        if event.key() in {Qt.Key_Return, Qt.Key_Enter} and not (event.modifiers() & Qt.ShiftModifier):
            event.accept()
            self.submit_requested.emit()
            return

        if self._should_reset_history_nav(event):
            self._reset_history_navigation()

        super().keyPressEvent(event)
        self._refresh_mention_popup()

    def insertFromMimeData(self, source: QMimeData) -> None:
        if source.hasImage():
            image = source.imageData()
            if image is not None:
                self.image_pasted.emit(image)
                self._refresh_mention_popup()
                return
        if source.hasUrls():
            parts = []
            image_paths: list[str] = []
            handled_local_urls = False
            for url in source.urls():
                if url.isLocalFile():
                    handled_local_urls = True
                    local_path = url.toLocalFile()
                    if can_read_image_file(local_path):
                        image_paths.append(local_path)
                    else:
                        parts.append(self.format_file_reference(local_path))
            if image_paths:
                self.image_files_pasted.emit(image_paths)
            if parts:
                self.insertPlainText(" ".join(parts))
            if handled_local_urls:
                self._refresh_mention_popup()
                return
        if source.hasText():
            self.insertPlainText(self._normalize_pasted_text(source.text()))
        else:
            super().insertFromMimeData(source)
        self._refresh_mention_popup()

    @staticmethod
    def _normalize_pasted_text(raw_text: str) -> str:
        text = str(raw_text or "").replace("\u2028", "\n").replace("\u2029", "\n")
        if not text:
            return text
        trimmed = text.strip("\r\n")
        if trimmed and "\n" not in trimmed and "\r" not in trimmed:
            # Clipboard providers often append a trailing line break for single-line values.
            return trimmed
        return text
    
    def focusOutEvent(self, event) -> None:  # type: ignore[override]
        super().focusOutEvent(event)
        QTimer.singleShot(0, self._close_mention_popup_if_inactive)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        super().mouseReleaseEvent(event)
        self._refresh_mention_popup()

    def _history_for_session(self) -> list[str]:
        return self._history_by_session.setdefault(self._history_session_id, [])

    def _should_reset_history_nav(self, event: QKeyEvent) -> bool:
        if event.text():
            return True
        return event.key() in {
            Qt.Key_Backspace,
            Qt.Key_Delete,
        }

    def _reset_history_navigation(self) -> None:
        self._history_nav_index = len(self._history_for_session())

    def _set_plain_text_and_move_cursor_to_end(self, value: str) -> None:
        self.setPlainText(value)
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

    def _navigate_history(self, direction: int) -> bool:
        history = self._history_for_session()
        if not history:
            return False

        if direction < 0:
            if self._history_nav_index > 0:
                self._history_nav_index -= 1
            else:
                self._history_nav_index = 0
        else:
            if self._history_nav_index < len(history) - 1:
                self._history_nav_index += 1
            else:
                self._history_nav_index = len(history)
                self._set_plain_text_and_move_cursor_to_end("")
                return True

        if 0 <= self._history_nav_index < len(history):
            self._set_plain_text_and_move_cursor_to_end(history[self._history_nav_index])
            return True
        return False

    def _refresh_mention_popup(self) -> None:
        mention = self._current_mention_token()
        if mention is None:
            self._close_mention_popup()
            return

        query = mention["query"].lower()
        matches = self._filter_mention_candidates(query)
        if not matches:
            self._close_mention_popup()
            return

        popup = self._ensure_mention_popup()
        popup.ensure_host()
        popup.set_items(matches)
        self._position_mention_popup()
        if not popup.isVisible():
            popup.show()
        popup.raise_()

    def _current_mention_token(self) -> dict[str, int | str] | None:
        text = self.toPlainText()
        cursor = self.textCursor()
        pos = cursor.position()
        if pos < 0:
            return None

        before_cursor = text[:pos]
        match = re.search(r"(?:^|\s)@([^\s@]*)$", before_cursor)
        if not match:
            return None

        token_start = match.start(1) - 1
        if token_start < 0:
            return None

        return {
            "start": token_start,
            "end": pos,
            "query": match.group(1),
        }

    def _ensure_file_index(self, *, force_refresh: bool = False) -> None:
        if self._file_index_static:
            return
        root = Path.cwd()
        root_str = str(root)
        should_rescan = (
            force_refresh
            or self._file_index_dirty
            or self._file_index_root != root_str
            or not self._file_index
        )
        if not should_rescan:
            return

        rows: dict[str, dict[str, Any]] = {}
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in COMPOSER_MENTION_EXCLUDED_DIRS]
            for dirname in dirnames:
                full_path = Path(current_root) / dirname
                self._add_index_row(rows, self._build_file_index_row(root, full_path, is_dir=True))
            for filename in filenames:
                full_path = Path(current_root) / filename
                self._add_index_row(rows, self._build_file_index_row(root, full_path))
        self._file_index = sorted(rows.values(), key=lambda row: row["relative"])
        self._file_index_root = root_str
        self._file_index_last_scan_at = time.monotonic()
        self._file_index_dirty = False
        self._sync_file_index_watcher(root, self._file_index)

    @staticmethod
    def _add_index_row(rows: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
        relative = str(row.get("relative") or "")
        if relative:
            rows[relative] = row

    def _build_file_index_row(self, root: Path, full_path: Path, *, is_dir: bool = False) -> dict[str, Any]:
        try:
            relative = full_path.relative_to(root).as_posix()
        except ValueError:
            relative = full_path.as_posix()
        folder = Path(relative).parent.as_posix()
        if folder == ".":
            folder = ""
        display = f"{relative}/" if is_dir and not relative.endswith("/") else relative
        return {
            "name": full_path.name,
            "name_lower": full_path.name.lower(),
            "relative": relative,
            "relative_lower": relative.lower(),
            "folder": folder,
            "depth": relative.count("/"),
            "is_dir": is_dir,
            "display": display,
        }

    def _filter_mention_candidates(self, query_lower: str) -> list[dict[str, Any]]:
        force_refresh = not self._file_index_static and self._file_index and (
            self._file_index_dirty
            or (not query_lower and time.monotonic() - self._file_index_last_scan_at > 2.0)
        )
        self._ensure_file_index(force_refresh=force_refresh)
        if not self._file_index:
            return []

        matches = self._rank_mention_candidates(query_lower)
        if matches or self._file_index_static:
            return matches

        # Files can appear after startup; do one refresh on cache miss.
        self._ensure_file_index(force_refresh=True)
        return self._rank_mention_candidates(query_lower)

    def _rank_mention_candidates(self, query_lower: str) -> list[dict[str, Any]]:
        if not query_lower:
            return sorted(
                self._file_index,
                key=lambda row: (
                    int(row.get("depth", 0)),
                    1 if row.get("is_dir") else 0,
                    len(str(row["relative"])),
                    str(row["relative"]),
                ),
            )[:COMPOSER_MENTION_MAX_ITEMS]

        ranked: list[tuple[int, int, int, dict[str, str]]] = []
        for row in self._file_index:
            name_lower = row["name_lower"]
            relative_lower = row["relative_lower"]
            if query_lower not in name_lower and query_lower not in relative_lower:
                continue

            if name_lower.startswith(query_lower):
                rank = 0
            elif query_lower in name_lower:
                rank = 1
            elif relative_lower.startswith(query_lower):
                rank = 2
            else:
                rank = 3
            ranked.append(
                (
                    rank,
                    int(row.get("depth", 0)),
                    1 if row.get("is_dir") else 0,
                    len(row["relative"]),
                    row,
                )
            )

        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]["relative"]))
        return [item[4] for item in ranked[:COMPOSER_MENTION_MAX_ITEMS]]

    def _warm_file_index(self) -> None:
        if self._file_index_static:
            return
        if not self._file_index_dirty and time.monotonic() - self._file_index_last_scan_at < 3.0 and self._file_index:
            return
        self._ensure_file_index()

    def _on_file_index_directory_changed(self, _path: str) -> None:
        if self._file_index_static:
            return
        self._file_index_dirty = True
        if self._current_mention_token() is not None:
            QTimer.singleShot(0, self._refresh_mention_popup)

    def _clear_file_index_watcher(self) -> None:
        watched = list(self._file_index_watcher.directories())
        if watched:
            self._file_index_watcher.removePaths(watched)

    def _sync_file_index_watcher(self, root: Path, rows: list[dict[str, Any]]) -> None:
        self._clear_file_index_watcher()
        paths = {str(root)}
        for row in rows:
            if not row.get("is_dir"):
                continue
            relative = str(row.get("relative") or "").strip()
            if not relative:
                continue
            paths.add(str(root / Path(relative)))
        self._file_index_watcher.addPaths(sorted(paths))

    def _position_mention_popup(self) -> None:
        popup = self._mention_popup
        if popup is None:
            return
        anchor = self.cursorRect().bottomLeft() + QPoint(0, 6)
        popup_parent = popup.parentWidget() or self.window()
        local_pos = self.mapTo(popup_parent, anchor)
        popup_size = popup.sizeHint()
        bounds = popup_parent.rect()
        max_x = max(8, bounds.width() - popup_size.width() - 8)
        max_y = max(8, bounds.height() - popup_size.height() - 8)
        local_pos.setX(max(8, min(local_pos.x(), max_x)))
        local_pos.setY(max(8, min(local_pos.y(), max_y)))
        popup.move(local_pos)

    def _accept_current_mention(self) -> None:
        if self._mention_popup is None:
            self._close_mention_popup()
            return
        selected = self._mention_popup.current_relative_path()
        if not selected:
            self._close_mention_popup()
            return
        self._insert_selected_mention(selected)

    def _insert_selected_mention(self, relative_path: str) -> None:
        mention = self._current_mention_token()
        if mention is None:
            self._close_mention_popup()
            return

        replacement = self.format_file_reference(relative_path)
        cursor = self.textCursor()
        cursor.setPosition(int(mention["start"]))
        cursor.setPosition(int(mention["end"]), QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(replacement)
        self.setTextCursor(cursor)
        self.setFocus()
        self._close_mention_popup()

    def _close_mention_popup(self) -> None:
        if self._mention_popup is not None:
            self._mention_popup.hide()

    def _close_mention_popup_if_inactive(self) -> None:
        popup = self._mention_popup
        if popup is None or not popup.isVisible():
            return
        focus_widget = QApplication.focusWidget()
        if self.hasFocus():
            return
        if focus_widget is not None and popup.isAncestorOf(focus_widget):
            return
        if popup.underMouse():
            return
        self._close_mention_popup()

    def _popup_host_widget(self) -> QWidget:
        host = self.window()
        if host is None or host is self:
            return self.parentWidget() or self
        central_widget_getter = getattr(host, "centralWidget", None)
        if callable(central_widget_getter):
            central = central_widget_getter()
            if isinstance(central, QWidget):
                return central
        return host

    def _ensure_mention_popup(self) -> "_ComposerMentionPopup":
        if self._mention_popup is None:
            self._mention_popup = _ComposerMentionPopup(self, host=self._popup_host_widget())
            self._mention_popup.file_selected.connect(self._insert_selected_mention)
        return self._mention_popup

class _ComposerMentionPopup(QFrame):
    file_selected = Signal(str)
    DISPLAY_TEXT_ROLE = int(Qt.UserRole) + 1

    def __init__(self, owner: QWidget, *, host: QWidget) -> None:
        popup_parent = host
        super().__init__(popup_parent)
        self._owner = owner
        self.setObjectName("ComposerMentionPopup")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumWidth(COMPOSER_MENTION_POPUP_MIN_WIDTH)
        self.setMaximumWidth(COMPOSER_MENTION_POPUP_MAX_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.header_label = QLabel("Files", self)
        self.header_label.setObjectName("ComposerMentionHeader")
        layout.addWidget(self.header_label)

        self.list_widget = QListWidget(self)
        self.list_widget.setObjectName("ComposerMentionList")
        self.list_widget.setFrameShape(QFrame.NoFrame)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.list_widget.setUniformItemSizes(False)
        self.list_widget.setSpacing(1)
        self.list_widget.setFocusPolicy(Qt.NoFocus)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemActivated.connect(self._on_item_clicked)
        self.list_widget.currentRowChanged.connect(self._sync_item_states)
        layout.addWidget(self.list_widget)

    def ensure_host(self) -> None:
        host = self._owner.window()
        if host is None or host is self._owner:
            host = self._owner.parentWidget() or self._owner
        central_widget_getter = getattr(host, "centralWidget", None)
        if callable(central_widget_getter):
            central = central_widget_getter()
            if isinstance(central, QWidget):
                host = central
        if self.parentWidget() is host:
            return
        self.hide()
        self.setParent(host)

    def set_items(self, rows: list[dict[str, Any]]) -> None:
        self.list_widget.clear()
        max_text_width = self.header_label.fontMetrics().horizontalAdvance(self.header_label.text())
        metrics = self.list_widget.fontMetrics()
        for row in rows:
            relative = str(row["relative"])
            display_text = str(row.get("display") or relative)
            text = str(row.get("name") or display_text or relative)
            folder = str(row.get("folder") or "")
            item = QListWidgetItem()
            item.setData(Qt.DisplayRole, "")
            item.setData(self.DISPLAY_TEXT_ROLE, display_text)
            item.setData(Qt.UserRole, relative)
            widget = _ComposerMentionItemWidget(
                self._owner,
                text=text,
                folder=folder,
                relative=relative,
                is_dir=bool(row.get("is_dir")),
            )
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)
            folder_width = widget.folder_width_hint()
            max_text_width = max(max_text_width, metrics.horizontalAdvance(text) + folder_width + 64)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
        row_height = self.list_widget.sizeHintForRow(0) if self.list_widget.count() > 0 else 22
        visible_count = min(7, max(1, self.list_widget.count()))
        total_height = visible_count * max(24, row_height) + max(0, (visible_count - 1) * 2) + 8
        self.list_widget.setFixedHeight(total_height)
        self.setFixedHeight(total_height + self.header_label.sizeHint().height() + 24)
        target_width = max_text_width + 68
        popup_width = max(COMPOSER_MENTION_POPUP_MIN_WIDTH, min(COMPOSER_MENTION_POPUP_MAX_WIDTH, target_width))
        self.setFixedWidth(popup_width)
        self._sync_item_states(self.list_widget.currentRow())

    def move_selection(self, delta: int) -> None:
        count = self.list_widget.count()
        if count <= 0:
            return
        current = self.list_widget.currentRow()
        if current < 0:
            current = 0
        target = max(0, min(count - 1, current + delta))
        self.list_widget.setCurrentRow(target)

    def current_relative_path(self) -> str:
        item = self.list_widget.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.UserRole) or "")

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        relative = str(item.data(Qt.UserRole) or "")
        if relative:
            self.file_selected.emit(relative)

    def _sync_item_states(self, current_row: int) -> None:
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            widget = self.list_widget.itemWidget(item)
            if isinstance(widget, _ComposerMentionItemWidget):
                widget.set_selected(index == current_row)


class _ComposerMentionItemWidget(QWidget):
    def __init__(self, owner: QWidget, *, text: str, folder: str, relative: str, is_dir: bool) -> None:
        super().__init__(owner)
        self.setObjectName("ComposerMentionItem")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._selected = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        self.icon_label = QLabel(self)
        self.icon_label.setObjectName("ComposerMentionItemIcon")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFixedSize(18, 18)
        self.icon_label.setPixmap(self._icon_for_entry(owner, relative=relative, is_dir=is_dir).pixmap(14, 14))
        layout.addWidget(self.icon_label, 0, Qt.AlignVCenter)

        self.title_label = QLabel(text, self)
        self.title_label.setObjectName("ComposerMentionItemTitle")
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.title_label, 1, Qt.AlignVCenter)

        self.folder_label = QLabel(folder, self)
        self.folder_label.setObjectName("ComposerMentionItemMeta")
        self.folder_label.setVisible(bool(folder))
        layout.addWidget(self.folder_label, 0, Qt.AlignVCenter)

        self.set_selected(False)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        option = QStyleOption()
        option.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, option, painter, self)
        super().paintEvent(event)

    def set_selected(self, selected: bool) -> None:
        if self._selected == selected:
            return
        self._selected = selected
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)
        for child in (self.title_label, self.folder_label):
            child.style().unpolish(child)
            child.style().polish(child)
        self.update()

    def folder_width_hint(self) -> int:
        if not self.folder_label.isVisible():
            return 0
        return self.folder_label.fontMetrics().horizontalAdvance(self.folder_label.text()) + 8

    @staticmethod
    def _icon_for_entry(owner: QWidget, *, relative: str, is_dir: bool):
        if is_dir:
            return _fa_icon("fa5s.folder", color=ACCENT, size=13)

        name = Path(relative).name.lower()
        suffix = Path(relative).suffix.lower()
        if name in {".env", ".env.example"} or suffix in {".env", ".ini", ".cfg", ".conf", ".toml", ".yaml", ".yml"}:
            return _fa_icon("fa5s.sliders-h", color=TEXT_WARNING, size=13)
        if suffix in {".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".ps1", ".bat", ".cmd", ".json", ".xml"}:
            return _fa_icon("fa5s.file-code", color=TEXT_SUCCESS, size=13)
        if suffix in {".md", ".txt", ".rst", ".log"}:
            return _fa_icon("fa5s.file-alt", color=TEXT_MUTED, size=13)
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"}:
            return _fa_icon("fa5s.file-image", color=ACCENT, size=13)
        if suffix in {".zip", ".7z", ".tar", ".gz", ".rar"}:
            return _fa_icon("fa5s.file-archive", color=TEXT_WARNING, size=13)
        if suffix in {".pdf"}:
            return _fa_icon("fa5s.file-pdf", color=FILE_PDF_ICON_TEXT, size=13)
        if suffix in {".csv", ".tsv", ".xlsx"}:
            return _fa_icon("fa5s.file-excel", color=TEXT_SUCCESS, size=13)
        return _fa_icon("fa5s.file", color=TEXT_MUTED, size=13)


