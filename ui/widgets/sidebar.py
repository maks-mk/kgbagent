from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from PySide6.QtCore import QAbstractListModel, QModelIndex, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListView,
    QMenu,
    QStyle,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from ui.theme import SUCCESS_GREEN, SURFACE_ALT, TEXT_MUTED, TEXT_PRIMARY
from .foundation import _fa_icon


def _format_sidebar_time(value: str) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    local_dt = dt.astimezone()
    now = datetime.now(local_dt.tzinfo or timezone.utc)
    delta = now - local_dt
    if delta.total_seconds() < 0:
        return "now"
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "now"
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    if days < 7:
        return f"{days}d"
    weeks = days // 7
    if weeks < 5:
        return f"{weeks}w"
    return local_dt.strftime("%d %b")


def _sidebar_project_name(project_path: str) -> str:
    text = str(project_path or "").replace("\\", "/").rstrip("/")
    if not text:
        return "project"
    return text.split("/")[-1] or text


def _sidebar_dt(value: str) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _session_preview(session: dict[str, str]) -> str:
    project_path = str(session.get("project_path", "") or "").replace("\\", "/").strip()
    title = str(session.get("title", "") or "").strip()
    if project_path:
        project_name = _sidebar_project_name(project_path)
        if title and title != project_name:
            return project_path
        return f"Folder: {project_path}"
    return "Open this chat"


class SessionListModel(QAbstractListModel):
    KindRole = Qt.UserRole + 1
    SessionIdRole = Qt.UserRole + 2
    TitleRole = Qt.UserRole + 3
    UpdatedAtRole = Qt.UserRole + 4
    ProjectPathRole = Qt.UserRole + 5
    ProjectTitleRole = Qt.UserRole + 6
    PreviewRole = Qt.UserRole + 7
    ActiveProjectRole = Qt.UserRole + 8

    def __init__(self) -> None:
        super().__init__()
        self._items: list[dict[str, str]] = []
        self._source_sessions: list[dict[str, str]] = []
        self._expanded_projects: set[str] = set()
        self._project_visible_limit = 5
        self._active_project_path = ""

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        if parent.isValid():
            return 0
        return len(self._items)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:  # type: ignore[override]
        if not index.isValid():
            return Qt.NoItemFlags
        kind = str(self.data(index, self.KindRole) or "session")
        if kind == "group":
            return Qt.ItemIsEnabled
        if kind == "more":
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # type: ignore[override]
        if not index.isValid() or not (0 <= index.row() < len(self._items)):
            return None
        item = self._items[index.row()]
        if role == self.KindRole:
            return item.get("kind", "session")
        if role == Qt.DisplayRole:
            return item.get("title", "")
        if role == self.SessionIdRole:
            return item.get("session_id", "")
        if role == self.TitleRole:
            return item.get("title", "")
        if role == self.UpdatedAtRole:
            return item.get("updated_at", "")
        if role == self.ProjectPathRole:
            return item.get("project_path", "")
        if role == self.ProjectTitleRole:
            return item.get("project_title", "")
        if role == self.PreviewRole:
            return item.get("preview", "")
        if role == self.ActiveProjectRole:
            return bool(item.get("active_project", False))
        return None

    def set_sessions(self, sessions: list[dict[str, str]], active_session_id: str = "") -> None:
        self._source_sessions = [dict(row) for row in sessions]
        self._active_project_path = ""
        if active_session_id:
            for row in self._source_sessions:
                if str(row.get("session_id", "") or "") == active_session_id:
                    self._active_project_path = str(row.get("project_path", "") or "").strip()
                    break
        self._rebuild_items()

    def toggle_project_expansion(self, project_path: str) -> None:
        normalized = str(project_path or "")
        if normalized in self._expanded_projects:
            self._expanded_projects.remove(normalized)
        else:
            self._expanded_projects.add(normalized)

        old_group_row = next(
            (row for row, item in enumerate(self._items) if item.get("kind") == "group" and item.get("project_path") == normalized),
            -1,
        )
        new_items = self._build_items()
        new_group_row = next(
            (row for row, item in enumerate(new_items) if item.get("kind") == "group" and item.get("project_path") == normalized),
            -1,
        )
        if old_group_row < 0 or new_group_row < 0 or old_group_row != new_group_row:
            self.beginResetModel()
            self._items = new_items
            self.endResetModel()
            return

        old_end = next(
            (row for row in range(old_group_row + 1, len(self._items)) if self._items[row].get("kind") == "group"),
            len(self._items),
        )
        new_end = next(
            (row for row in range(new_group_row + 1, len(new_items)) if new_items[row].get("kind") == "group"),
            len(new_items),
        )
        old_children = self._items[old_group_row + 1 : old_end]
        new_children = new_items[new_group_row + 1 : new_end]
        common_prefix = 0
        for old_item, new_item in zip(old_children, new_children):
            if old_item != new_item:
                break
            common_prefix += 1

        first = old_group_row + 1 + common_prefix
        old_tail_count = len(old_children) - common_prefix
        new_tail = new_children[common_prefix:]
        if old_tail_count:
            last = first + old_tail_count - 1
            self.beginRemoveRows(QModelIndex(), first, last)
            del self._items[first : last + 1]
            self.endRemoveRows()
        if new_tail:
            self.beginInsertRows(QModelIndex(), first, first + len(new_tail) - 1)
            self._items[first:first] = new_tail
            self.endInsertRows()

    def _rebuild_items(self) -> None:
        self.beginResetModel()
        self._items = self._build_items()
        self.endResetModel()

    def _build_items(self) -> list[dict[str, str]]:
        sessions: list[dict[str, str]] = []
        for raw in self._source_sessions:
            row = dict(raw)
            row["preview"] = _session_preview(row)
            sessions.append(row)

        grouped: dict[str, list[dict[str, str]]] = {}
        for row in sessions:
            project_key = str(row.get("project_path", "")).strip()
            grouped.setdefault(project_key, []).append(row)

        project_rows: list[tuple[str, list[dict[str, str]]]] = sorted(
            grouped.items(),
            key=lambda pair: max(
                (_sidebar_dt(item.get("updated_at", "")) for item in pair[1]),
                default=datetime.fromtimestamp(0, tz=timezone.utc),
            ),
            reverse=True,
        )

        items: list[dict[str, str]] = []
        for project_path, rows in project_rows:
            items.append(
                {
                    "kind": "group",
                    "project_path": project_path,
                    "project_title": _sidebar_project_name(project_path),
                    "title": _sidebar_project_name(project_path),
                    "session_id": "",
                    "updated_at": "",
                    "preview": "",
                    "active_project": project_path == self._active_project_path,
                }
            )
            sorted_rows = sorted(rows, key=lambda item: _sidebar_dt(item.get("updated_at", "")), reverse=True)
            is_expanded = project_path in self._expanded_projects
            visible_rows = sorted_rows if is_expanded else sorted_rows[: self._project_visible_limit]
            for row in visible_rows:
                entry = dict(row)
                entry["kind"] = "session"
                entry["project_title"] = _sidebar_project_name(project_path)
                items.append(entry)
            hidden_count = len(sorted_rows) - len(visible_rows)
            if is_expanded and len(sorted_rows) > self._project_visible_limit:
                items.append(
                    {
                        "kind": "more",
                        "project_path": project_path,
                        "project_title": _sidebar_project_name(project_path),
                        "title": "Show less",
                        "session_id": "",
                        "updated_at": "",
                        "preview": "",
                    }
                )
            elif hidden_count > 0:
                items.append(
                    {
                        "kind": "more",
                        "project_path": project_path,
                        "project_title": _sidebar_project_name(project_path),
                        "title": "Show more",
                        "session_id": "",
                        "updated_at": "",
                        "preview": str(hidden_count),
                    }
                )

        return items

    def session_id_at(self, index: QModelIndex) -> str:
        if not index.isValid():
            return ""
        if str(self.data(index, self.KindRole) or "session") != "session":
            return ""
        return str(self.data(index, self.SessionIdRole) or "")

    def index_for_session(self, session_id: str) -> QModelIndex:
        if not session_id:
            return QModelIndex()
        for row, item in enumerate(self._items):
            if item.get("kind") == "session" and item.get("session_id") == session_id:
                return self.index(row, 0)
        return QModelIndex()

    def session_row_count(self) -> int:
        return sum(1 for item in self._items if item.get("kind") == "session")

    def title_for_session(self, session_id: str) -> str:
        if not session_id:
            return ""
        for item in self._items:
            if item.get("kind") == "session" and item.get("session_id") == session_id:
                return str(item.get("title", "") or "")
        return ""

    def project_title_for_path(self, project_path: str) -> str:
        return _sidebar_project_name(str(project_path or ""))

    def session_count_for_project(self, project_path: str) -> int:
        project_key = str(project_path or "").strip()
        if not project_key:
            return 0
        return sum(1 for row in self._source_sessions if str(row.get("project_path", "")).strip() == project_key)

    def has_source_sessions(self) -> bool:
        return bool(self._source_sessions)


class SessionItemDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:  # type: ignore[override]
        painter.save()
        item_kind = str(index.data(SessionListModel.KindRole) or "session")
        rect = option.rect.adjusted(6, 2, -6, -2)
        is_selected = bool(option.state & QStyle.State_Selected)
        is_hovered = bool(option.state & QStyle.State_MouseOver)

        if item_kind == "group":
            is_active_project = bool(index.data(SessionListModel.ActiveProjectRole))
            icon_color = TEXT_PRIMARY if is_active_project else TEXT_MUTED
            icon_rect = rect.adjusted(4, 7, 0, 0)
            painter.drawPixmap(icon_rect.left(), icon_rect.top(), _fa_icon("fa5.folder-open", color=icon_color, size=13).pixmap(13, 13))
            group_text = str(index.data(SessionListModel.ProjectTitleRole) or index.data(SessionListModel.TitleRole) or "")
            title_rect = rect.adjusted(26, 0, -8, 0)
            title_font = option.font
            title_font.setPointSize(10)
            title_font.setWeight(QFont.DemiBold if is_active_project else QFont.Medium)
            painter.setFont(title_font)
            painter.setPen(QColor(TEXT_PRIMARY if is_active_project else TEXT_MUTED))
            painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, group_text)
            painter.restore()
            return

        if item_kind == "more":
            text_rect = rect.adjusted(34, 0, -8, 0)
            font = option.font
            font.setPointSize(9)
            font.setWeight(QFont.Medium)
            painter.setFont(font)
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, str(index.data(SessionListModel.TitleRole) or "Show more"))
            painter.restore()
            return

        background = QColor(0, 0, 0, 0)
        if is_selected:
            background = QColor(SURFACE_ALT)
        elif is_hovered:
            background = QColor(SURFACE_ALT)
            background.setAlpha(110)
        if background.alpha() > 0:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(Qt.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(rect, 7, 7)

        title = str(index.data(SessionListModel.TitleRole) or "")
        updated_at = _format_sidebar_time(str(index.data(SessionListModel.UpdatedAtRole) or ""))

        title_font = option.font
        title_font.setPointSize(10)
        title_font.setWeight(QFont.Medium)
        title_metrics = QFontMetrics(title_font)

        meta_font = option.font
        meta_font.setPointSize(9)
        meta_font.setWeight(QFont.Medium)
        meta_metrics = QFontMetrics(meta_font)

        time_width = max(32, meta_metrics.horizontalAdvance(updated_at) + 6)
        content_rect = rect.adjusted(34, 0, -12, 0)
        title_rect = content_rect.adjusted(0, 0, -(time_width + 8), 0)
        time_rect = content_rect.adjusted(content_rect.width() - time_width, 0, 0, 0)

        painter.setFont(title_font)
        painter.setPen(QColor(TEXT_PRIMARY))
        painter.drawText(
            title_rect,
            Qt.AlignLeft | Qt.AlignVCenter,
            title_metrics.elidedText(title, Qt.ElideRight, max(10, title_rect.width())),
        )

        painter.setFont(meta_font)
        painter.setPen(QColor(TEXT_MUTED))
        painter.drawText(time_rect, Qt.AlignRight | Qt.AlignVCenter, updated_at)

        if is_selected:
            dot_diameter = 6
            dot_rect = QRect(
                rect.left() + 12,
                rect.center().y() - dot_diameter // 2,
                dot_diameter,
                dot_diameter,
            )
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(SUCCESS_GREEN))
            painter.drawEllipse(dot_rect)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # type: ignore[override]
        _ = option
        kind = str(index.data(SessionListModel.KindRole) or "session")
        if kind == "group":
            return QSize(260, 34)
        if kind == "more":
            return QSize(260, 30)
        return QSize(260, 32)


class SessionSidebarWidget(QWidget):
    session_activated = Signal(str)
    session_delete_requested = Signal(str)
    project_delete_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("SessionSidebar")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)

        title = QLabel("Projects")
        title.setObjectName("SidebarSectionTitle")
        header_row.addWidget(title, 1)

        root.addLayout(header_row)

        self.list_view = QListView()
        self.list_view.setObjectName("SessionListView")
        self.list_view.setMouseTracking(True)
        self.list_view.setUniformItemSizes(False)
        self.list_view.setEditTriggers(QListView.NoEditTriggers)
        self.list_view.setSelectionMode(QListView.SingleSelection)
        self.list_view.setSelectionBehavior(QListView.SelectRows)
        self.list_view.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_view.setAccessibleName("Chat list")
        self.list_view.setAccessibleDescription("Browse and open saved chats")

        self.model = SessionListModel()
        self.delegate = SessionItemDelegate()
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(self.delegate)
        self.list_view.clicked.connect(self._emit_clicked_session)
        self.list_view.customContextMenuRequested.connect(self._show_context_menu)
        self.list_view.selectionModel().currentChanged.connect(self._on_current_changed)
        root.addWidget(self.list_view, 1)

        self.empty_label = QLabel("No chats yet")
        self.empty_label.setObjectName("SidebarEmptyState")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.empty_label.setVisible(False)
        root.addWidget(self.empty_label)

    def set_sessions(self, sessions: list[dict[str, str]], active_session_id: str) -> None:
        self.model.set_sessions(sessions, active_session_id)
        self._refresh_empty_state()
        if not active_session_id:
            self.list_view.clearSelection()
            return
        index = self.model.index_for_session(active_session_id)
        if index.isValid():
            self.list_view.setCurrentIndex(index)
            self.list_view.scrollTo(index)
        else:
            self.list_view.clearSelection()

    def title_for_session(self, session_id: str) -> str:
        return self.model.title_for_session(session_id)

    def title_for_project(self, project_path: str) -> str:
        return self.model.project_title_for_path(project_path)

    def session_count_for_project(self, project_path: str) -> int:
        return self.model.session_count_for_project(project_path)

    def _refresh_empty_state(self) -> None:
        if self.model.session_row_count() > 0:
            self.empty_label.setVisible(False)
            self.list_view.setVisible(True)
            return
        self.list_view.setVisible(False)
        self.empty_label.setText("No chats yet" if not self.model.has_source_sessions() else "No chats available")
        self.empty_label.setVisible(True)

    def _emit_clicked_session(self, index: QModelIndex) -> None:
        if str(index.data(SessionListModel.KindRole) or "session") == "more":
            self.model.toggle_project_expansion(str(index.data(SessionListModel.ProjectPathRole) or ""))
            self._refresh_empty_state()
            return
        session_id = self.model.session_id_at(index)
        if session_id:
            self.session_activated.emit(session_id)

    def _selected_session_id(self) -> str:
        return self.model.session_id_at(self.list_view.currentIndex())

    def _on_current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        _ = current
        if str(current.data(SessionListModel.KindRole) or "session") == "more":
            self.list_view.clearSelection()

    def _show_context_menu(self, pos) -> None:
        index = self.list_view.indexAt(pos)
        if not index.isValid():
            return
        item_kind = str(index.data(SessionListModel.KindRole) or "session")
        if item_kind == "group":
            self._show_project_context_menu(index, pos)
        elif item_kind == "session":
            self._show_session_context_menu(index, pos)

    def _show_session_context_menu(self, index: QModelIndex, pos) -> None:
        session_id = self.model.session_id_at(index)
        if not session_id:
            return

        menu = QMenu(self.list_view)
        delete_action = menu.addAction("Delete chat")
        selected = menu.exec(self.list_view.viewport().mapToGlobal(pos))
        if selected is delete_action:
            self.session_delete_requested.emit(session_id)

    def _show_project_context_menu(self, index: QModelIndex, pos) -> None:
        project_path = str(index.data(SessionListModel.ProjectPathRole) or "").strip()
        if not project_path:
            return

        count = self.model.session_count_for_project(project_path)
        menu = QMenu(self.list_view)
        delete_action = menu.addAction(f"Delete project ({count} {'chat' if count == 1 else 'chats'})")
        selected = menu.exec(self.list_view.viewport().mapToGlobal(pos))
        if selected is delete_action:
            self.project_delete_requested.emit(project_path)
