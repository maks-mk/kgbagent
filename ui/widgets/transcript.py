from __future__ import annotations

import difflib
import re

from typing import Any

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from .foundation import TRANSCRIPT_MAX_WIDTH, _fa_icon
from .messages import AssistantMessageWidget, NoticeWidget, RunStatsWidget, StatusIndicatorWidget, UserMessageWidget
from .tool_group import ToolGroupWidget
from .tools import ToolCardWidget


class ConversationTurnWidget(QWidget):
    def __init__(
        self,
        user_text: str,
        attachments: list[dict[str, Any]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._assistant_markdown = ""
        self._force_new_assistant_block = False
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(5)
        self._timeline: list[tuple[str, QWidget]] = []
        self.assistant_segments: list[AssistantMessageWidget] = []
        self.tool_cards: dict[str, ToolCardWidget] = {}
        self.tool_group: ToolGroupWidget | None = None
        self.status_widget: StatusIndicatorWidget | None = None
        self._append_block("user", UserMessageWidget(user_text, attachments=list(attachments or []), parent=self))

    @staticmethod
    def _common_prefix_length(first: str, second: str) -> int:
        limit = min(len(first), len(second))
        index = 0
        while index < limit and first[index] == second[index]:
            index += 1
        return index

    @staticmethod
    def _normalized_text(text: str) -> str:
        return " ".join(str(text or "").split())

    @classmethod
    def _markdown_boundary_after_visible_prefix(cls, visible: str, incoming: str) -> int | None:
        visible_normalized = cls._normalized_text(visible)
        if not visible_normalized:
            return None

        best_index: int | None = None
        for match in re.finditer(r"\S+", incoming):
            prefix = incoming[: match.end()]
            prefix_normalized = cls._normalized_text(prefix)
            if not prefix_normalized:
                continue
            if visible_normalized.startswith(prefix_normalized):
                best_index = match.end()
                continue
            if prefix_normalized.startswith(visible_normalized):
                return best_index if best_index is not None else match.end()
            if len(prefix_normalized) >= min(len(visible_normalized), 80):
                similarity = difflib.SequenceMatcher(None, visible_normalized, prefix_normalized).quick_ratio()
                if similarity >= 0.94:
                    best_index = match.end()

        return best_index

    @classmethod
    def _assistant_resume_text(cls, visible: str, incoming: str) -> str:
        prefix_len = cls._common_prefix_length(visible, incoming)
        exact_prefix = incoming.startswith(visible)
        significant_prefix = bool(visible) and (
            prefix_len >= min(len(visible), 48)
            or prefix_len >= int(len(visible) * 0.8)
        )
        partial_word_prefix = (
            bool(visible)
            and prefix_len == len(visible)
            and prefix_len < len(incoming)
            and not visible[-1].isspace()
            and not incoming[prefix_len].isspace()
        )
        if prefix_len > 0 and (exact_prefix or significant_prefix) and not partial_word_prefix:
            return incoming[prefix_len:].lstrip()

        boundary = cls._markdown_boundary_after_visible_prefix(visible, incoming)
        if boundary is not None:
            replayed_prefix = incoming[:boundary]
            replayed_words = re.findall(r"\S+", replayed_prefix)
            replayed_normalized = cls._normalized_text(replayed_prefix)
            significant_replay = bool(replayed_normalized) and (
                len(replayed_normalized) >= min(len(cls._normalized_text(visible)), 48)
                or len(replayed_normalized) >= int(len(cls._normalized_text(visible)) * 0.8)
            )
            # A shared first word is ambiguous: it can be the beginning of a
            # new comment, especially when the stream split that word across
            # chunks. Only remove a replay after at least two complete words.
            if significant_replay and len(replayed_words) >= 2:
                return incoming[boundary:].lstrip()

        return incoming

    def _append_block(self, kind: str, widget: QWidget) -> QWidget:
        self._layout.addWidget(widget)
        self._timeline.append((kind, widget))
        if self.status_widget is not None and kind != "stats":
            self._layout.removeWidget(self.status_widget)
            self._layout.addWidget(self.status_widget)
        return widget

    def set_status(self, label: str, *, meta: str = "", phase: str = "working") -> None:
        if self.status_widget is None:
            self.status_widget = StatusIndicatorWidget(label, parent=self)
            self._layout.addWidget(self.status_widget)
        else:
            self._layout.removeWidget(self.status_widget)
            self._layout.addWidget(self.status_widget)
        self.status_widget.set_state(label, meta=meta, phase=phase)

    def clear_status(self) -> None:
        if self.status_widget is None:
            return
        self._layout.removeWidget(self.status_widget)
        self.status_widget.deleteLater()
        self.status_widget = None

    def has_status(self) -> bool:
        return self.status_widget is not None

    def _ensure_assistant_segment(self) -> AssistantMessageWidget:
        if self._timeline and self._timeline[-1][0] == "assistant" and not self._force_new_assistant_block:
            return self._timeline[-1][1]  # type: ignore[return-value]
        self._force_new_assistant_block = False
        segment = AssistantMessageWidget(parent=self)
        self.assistant_segments.append(segment)
        self._append_block("assistant", segment)
        return segment

    def _remove_trailing_empty_assistant_segment(self) -> bool:
        if not self._timeline or self._timeline[-1][0] != "assistant":
            return False
        widget = self._timeline[-1][1]
        if not isinstance(widget, AssistantMessageWidget) or AssistantMessageWidget.has_renderable_content(widget.markdown()):
            return False
        self._timeline.pop()
        self._layout.removeWidget(widget)
        if widget in self.assistant_segments:
            self.assistant_segments.remove(widget)
        widget.deleteLater()
        self._assistant_markdown = ""
        return True

    def set_assistant_markdown(self, markdown: str) -> None:
        if not AssistantMessageWidget.has_renderable_content(markdown):
            if self._remove_trailing_empty_assistant_segment():
                self._assistant_markdown = ""
            return
        if (
            markdown == self._assistant_markdown
            and self.assistant_segments
        ):
            return

        starts_new_assistant_block = not self._timeline or self._timeline[-1][0] != "assistant"
        resumes_after_tool_group = self.tool_group is not None and starts_new_assistant_block
        if resumes_after_tool_group and self._assistant_markdown:
            segment_text = self._assistant_resume_text(self._assistant_markdown, markdown)
            if not segment_text:
                self._assistant_markdown = markdown
                return
            self.tool_group.collapse()
            segment = self._ensure_assistant_segment()
            segment.set_content(segment_text)
            self._assistant_markdown = markdown
            return
        if resumes_after_tool_group:
            self.tool_group.collapse()

        segment = self._ensure_assistant_segment()

        if not self._assistant_markdown:
            segment.set_content(markdown)
        elif markdown.startswith(self._assistant_markdown):
            segment_text = markdown[len(self._assistant_markdown):]
            if segment_text:
                segment.set_content(segment.markdown() + segment_text)
        else:
            segment.set_content(markdown)

        self._assistant_markdown = markdown

    def begin_assistant_block(self) -> None:
        """Make the next streamed response a new timeline block."""
        self.set_assistant_streaming(False)
        self._assistant_markdown = ""
        self._force_new_assistant_block = True

    def set_assistant_streaming(self, active: bool) -> None:
        if not self.assistant_segments:
            return
        self.assistant_segments[-1].set_streaming(active)

    def add_notice(self, message: str, level: str = "info") -> None:
        self._append_block("notice", NoticeWidget(message, level=level, parent=self))

    def add_assistant_message(self, markdown: str) -> AssistantMessageWidget:
        if self.tool_group is not None and (not self._timeline or self._timeline[-1][0] != "assistant"):
            self.tool_group.collapse()
        segment = AssistantMessageWidget(parent=self)
        segment.set_content(markdown)
        self.assistant_segments.append(segment)
        self._append_block("assistant", segment)
        self._assistant_markdown = markdown
        return segment

    def start_tool(self, payload: dict[str, Any]) -> ToolCardWidget:
        self.set_assistant_streaming(False)
        self._remove_trailing_empty_assistant_segment()
        tool_id = payload.get("tool_id", "")
        card = self.tool_cards.get(tool_id)
        if card is None:
            if self.tool_group is None or not self._timeline or self._timeline[-1][0] != "tool_group":
                self.tool_group = ToolGroupWidget(parent=self)
                self._append_block("tool_group", self.tool_group)
            card = ToolCardWidget(payload, parent=self.tool_group.container)
            self.tool_cards[tool_id] = card
            self.tool_group.add_tool(card)
        card.update_started_payload(payload)
        return card

    @staticmethod
    def _tool_group_for_card(card: ToolCardWidget) -> ToolGroupWidget | None:
        parent = card.parentWidget()
        while parent is not None:
            if isinstance(parent, ToolGroupWidget):
                return parent
            parent = parent.parentWidget()
        return None

    def finish_tool(self, payload: dict[str, Any]) -> None:
        tool_id = payload.get("tool_id", "")
        card = self.tool_cards.get(tool_id)
        if card is None:
            card = self.start_tool(payload)
        card.finish(payload)
        group = self._tool_group_for_card(card)
        if group is not None:
            group.refresh_completion(auto_collapse=False)

    def append_tool_output(self, payload: dict[str, Any]) -> None:
        tool_id = str(payload.get("tool_id", "") or "").strip()
        if not tool_id:
            return
        card = self.tool_cards.get(tool_id)
        if card is None:
            card = self.start_tool(
                {
                    "tool_id": tool_id,
                    "name": "cli_exec",
                    "args": {},
                    "display": "cli_exec",
                }
            )
        card.append_cli_output(
            str(payload.get("data", "") or ""),
            stream=str(payload.get("stream", "stdout") or "stdout"),
        )

    def complete(self, stats: str) -> None:
        # Set to False to hide run statistics.
        show_run_stats = True
        if show_run_stats and stats:
            self._append_block("stats", RunStatsWidget(stats, parent=self))

    def restore_blocks(self, blocks: list[dict[str, Any]]) -> None:
        for block in blocks:
            block_type = block.get("type")
            if block_type == "assistant":
                markdown = str(block.get("markdown", "") or "").strip()
                if markdown:
                    self.add_assistant_message(markdown)
            elif block_type == "tool":
                payload = dict(block.get("payload") or {})
                if payload:
                    self.finish_tool(payload)
            elif block_type == "notice":
                message = str(block.get("message", "") or "").strip()
                level = str(block.get("level") or "info")
                if message and level == "error":
                    self.add_notice(message, level)
            elif block_type == "stats":
                stats = str(block.get("stats", "") or "").strip()
                if stats:
                    self.complete(stats)
        if self.tool_group is not None:
            self.tool_group.collapse()

    def block_kinds(self) -> list[str]:
        return [kind for kind, _widget in self._timeline]


class ChatTranscriptWidget(QWidget):
    HISTORY_BATCH_SIZE = 10

    def __init__(self, history_batch_size: int | None = None) -> None:
        super().__init__()
        try:
            requested = int(history_batch_size) if history_batch_size is not None else self.HISTORY_BATCH_SIZE
        except (TypeError, ValueError):
            requested = self.HISTORY_BATCH_SIZE
        # Same rule as the AgentConfig validator: non-positive or invalid
        # values fall back to the default, oversized values clamp to 200.
        if requested < 1:
            requested = self.HISTORY_BATCH_SIZE
        self.history_batch_size = min(200, requested)
        self._auto_follow_enabled = True
        self._pending_scroll = False
        self._pending_force_scroll = False
        self._programmatic_scroll = False
        self._range_follow_ticket = 0
        self._range_follow_force = False
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.timeout.connect(self._flush_pending_scroll)
        self._older_turns: list[dict[str, Any]] = []
        self._history_button = None
        self._loading_history = False
        self._history_anchor = None
        self._restoring_history_anchor = False
        self._history_anchor_timer = QTimer(self)
        self._history_anchor_timer.setSingleShot(True)
        self._history_anchor_timer.timeout.connect(self._restore_history_anchor)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.container = QWidget(self.scroll)
        self.container.setObjectName("TranscriptContainer")
        shell = QHBoxLayout(self.container)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        # Side spacers keep stretch 0 so the column takes all free space up to
        # its maximum width; any leftover is then split evenly between them.
        shell.addStretch(0)
        self.shell = shell

        self.column = QWidget(self.container)
        self.column.setObjectName("TranscriptColumn")
        self.column.setMaximumWidth(TRANSCRIPT_MAX_WIDTH)
        self.column.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.layout = QVBoxLayout(self.column)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        self.layout.addStretch(1)

        shell.addWidget(self.column, 1)
        shell.addStretch(0)
        self.scroll.setWidget(self.container)
        outer.addWidget(self.scroll)
        scrollbar = self.scroll.verticalScrollBar()
        scrollbar.valueChanged.connect(self._handle_scrollbar_value_changed)
        scrollbar.rangeChanged.connect(self._handle_scrollbar_range_changed)

        self.jump_to_latest_button = QPushButton(_fa_icon("fa5s.arrow-down", size=12), "Jump to latest", self)
        self.jump_to_latest_button.setObjectName("TranscriptJumpButton")
        self.jump_to_latest_button.setVisible(False)
        self.jump_to_latest_button.setAccessibleName("Jump to latest message")
        self.jump_to_latest_button.setAccessibleDescription("Scroll to the most recent transcript event")
        self.jump_to_latest_button.clicked.connect(self.scroll_to_bottom)

    def clear_transcript(self) -> None:
        self._scroll_timer.stop()
        self._history_anchor_timer.stop()
        self._older_turns = []
        self._history_button = None
        self._loading_history = False
        self._history_anchor = None
        while self.layout.count() > 1:
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        self._auto_follow_enabled = True
        self._pending_scroll = False
        self._pending_force_scroll = False
        self._range_follow_ticket += 1
        self._range_follow_force = False
        self.jump_to_latest_button.setVisible(False)

    def add_global_notice(self, message: str, level: str = "info") -> None:
        self.layout.insertWidget(self.layout.count() - 1, NoticeWidget(message, level=level, parent=self.column))
        self.notify_content_changed()

    def last_turn(self) -> ConversationTurnWidget | None:
        for index in range(self.layout.count() - 2, -1, -1):
            widget = self.layout.itemAt(index).widget()
            if isinstance(widget, ConversationTurnWidget):
                return widget
        return None

    def start_turn(self, user_text: str, attachments: list[dict[str, Any]] | None = None) -> ConversationTurnWidget:
        self._history_anchor = None
        turn = ConversationTurnWidget(user_text, attachments=attachments, parent=self.column)
        self.layout.insertWidget(self.layout.count() - 1, turn)
        self.notify_content_changed(force=True)
        return turn

    def load_transcript(self, payload: dict[str, Any] | None) -> None:
        self.setUpdatesEnabled(False)
        self.scroll.setUpdatesEnabled(False)
        self.container.setUpdatesEnabled(False)
        self.column.setUpdatesEnabled(False)
        try:
            self.clear_transcript()
            payload = payload or {}
            turns = list(payload.get("turns", []) or [])
            self._older_turns = turns[:-self.history_batch_size]
            if self._older_turns:
                self._history_button = QPushButton(self.column)
                self._history_button.setObjectName("TranscriptJumpButton")
                self._history_button.clicked.connect(self.load_older_turns)
                self.layout.insertWidget(0, self._history_button)
                self._update_history_button()
            for turn_data in turns[-self.history_batch_size:]:
                self._insert_restored_turn(turn_data, self.layout.count() - 1)
        finally:
            self.column.setUpdatesEnabled(True)
            self.container.setUpdatesEnabled(True)
            self.scroll.setUpdatesEnabled(True)
            self.setUpdatesEnabled(True)
        self.notify_content_changed(force=True)

    def _insert_restored_turn(self, data: dict[str, Any], index: int) -> None:
        turn = ConversationTurnWidget(
            str(data.get("user_text", "") or ""),
            attachments=list(data.get("attachments", []) or []), parent=self.column,
        )
        turn.hide()
        turn.restore_blocks(list(data.get("blocks", []) or []))
        self.layout.insertWidget(index, turn)
        turn.show()

    def _update_history_button(self) -> None:
        if self._history_button is not None:
            self._history_button.setText(
                "Loading earlier messages…" if self._loading_history
                else f"Load earlier messages ({len(self._older_turns)})"
            )
            self._history_button.setEnabled(not self._loading_history)
            self._history_button.setVisible(bool(self._older_turns))

    def load_older_turns(self) -> None:
        if self._loading_history or not self._older_turns:
            return
        # Anchor an existing widget rather than a scroll-range estimate: wrapped
        # Markdown has variable heights and the history button may disappear.
        anchor = self.layout.itemAt(1).widget()
        self._history_anchor = (
            anchor, anchor.mapTo(self.scroll.viewport(), anchor.rect().topLeft()).y()
        )
        self._scroll_timer.stop()
        self._pending_scroll = False
        self._range_follow_ticket += 1
        self._range_follow_force = False
        self._pending_force_scroll = False
        self._auto_follow_enabled = False
        self._loading_history = True
        self._update_history_button()
        # Paint the busy label before synchronous widget construction blocks
        # painting. Do not pump unrelated input/timer events here.
        self._history_button.repaint()
        self.setUpdatesEnabled(False)
        self._programmatic_scroll = True
        try:
            batch = self._older_turns[-self.history_batch_size:]
            del self._older_turns[-self.history_batch_size:]
            for index, data in enumerate(batch, 1):
                self._insert_restored_turn(data, index)
            self._update_history_button()
            self.layout.activate()
            self.container.layout().activate()
            self._restore_history_anchor()
        finally:
            self._programmatic_scroll = False
            self._loading_history = False
            self._update_history_button()
            self.setUpdatesEnabled(True)
        self._update_jump_button()

    def _restore_history_anchor(self) -> None:
        if self._history_anchor is None or self._restoring_history_anchor:
            return
        self._restoring_history_anchor = True
        anchor, viewport_y = self._history_anchor
        was_programmatic = self._programmatic_scroll
        self._programmatic_scroll = True
        try:
            delta = anchor.mapTo(self.scroll.viewport(), anchor.rect().topLeft()).y() - viewport_y
            scrollbar = self.scroll.verticalScrollBar()
            scrollbar.setValue(scrollbar.value() + delta)
        finally:
            self._programmatic_scroll = was_programmatic
            self._restoring_history_anchor = False

    @property
    def auto_follow_enabled(self) -> bool:
        return self._auto_follow_enabled

    def is_near_bottom(self, threshold: int = 28) -> bool:
        scrollbar = self.scroll.verticalScrollBar()
        return (scrollbar.maximum() - scrollbar.value()) <= max(threshold, scrollbar.pageStep() // 8)

    def _handle_scrollbar_value_changed(self, _value: int) -> None:
        if self._programmatic_scroll:
            return
        # QScrollBar also emits valueChanged when layout changes clamp its
        # range. That is not a user scroll and must not release the anchor.
        if self._history_anchor is not None and self._history_anchor_timer.isActive():
            return
        self._history_anchor = None
        self._auto_follow_enabled = self.is_near_bottom()
        if not self._auto_follow_enabled:
            self._range_follow_force = False
        self._update_jump_button()

    def _handle_scrollbar_range_changed(self, _minimum: int, _maximum: int) -> None:
        if self._history_anchor is not None:
            self._history_anchor_timer.start(0)
            self._update_jump_button()
            return
        if not self._range_follow_ticket:
            self._update_jump_button()
            return
        ticket = self._range_follow_ticket
        QTimer.singleShot(0, self, lambda: self._follow_to_bottom(ticket))

    def notify_content_changed(self, *, force: bool = False) -> None:
        self.queue_scroll_to_bottom(force=force)

    def queue_scroll_to_bottom(self, *, force: bool = False) -> None:
        if force:
            self._pending_force_scroll = True
        if self._pending_scroll:
            return
        self._pending_scroll = True
        self._scroll_timer.start(0)

    def _flush_pending_scroll(self) -> None:
        self._pending_scroll = False
        force = self._pending_force_scroll
        self._pending_force_scroll = False
        if self._history_anchor is not None or (not force and not self._auto_follow_enabled):
            self._range_follow_ticket += 1
            self._range_follow_force = False
            return

        self._range_follow_force = force
        self._scrollbar_to_bottom()
        self._auto_follow_enabled = True
        self._update_jump_button()
        self._schedule_follow_up_scroll(force=force)

    def _scrollbar_to_bottom(self) -> None:
        scrollbar = self.scroll.verticalScrollBar()
        self._programmatic_scroll = True
        scrollbar.setValue(scrollbar.maximum())
        self._programmatic_scroll = False

    def _schedule_follow_up_scroll(self, *, force: bool) -> None:
        self._range_follow_force = force
        self._range_follow_ticket += 1
        ticket = self._range_follow_ticket
        follow_delays = (0, 20, 80)
        for delay in follow_delays:
            QTimer.singleShot(delay, self, lambda current=ticket: self._follow_to_bottom(current))
        QTimer.singleShot(max(follow_delays) + 12, self, lambda current=ticket: self._finish_follow_up(current))

    def _follow_to_bottom(self, ticket: int) -> None:
        if ticket != self._range_follow_ticket or self._history_anchor is not None:
            return
        if not self._range_follow_force and not self._auto_follow_enabled:
            return
        self._scrollbar_to_bottom()
        self._auto_follow_enabled = True
        self._update_jump_button()

    def _finish_follow_up(self, ticket: int) -> None:
        if ticket != self._range_follow_ticket:
            return
        self._range_follow_force = False

    def scroll_to_bottom(self) -> None:
        self._scroll_timer.stop()
        self._history_anchor = None
        self._pending_scroll = False
        self._pending_force_scroll = False
        self._scrollbar_to_bottom()
        self._auto_follow_enabled = True
        self._update_jump_button()
        self._schedule_follow_up_scroll(force=True)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        button_size = self.jump_to_latest_button.sizeHint()
        x = max(12, self.width() - button_size.width() - 18)
        y = max(12, self.height() - button_size.height() - 18)
        self.jump_to_latest_button.move(x, y)

    def _update_jump_button(self) -> None:
        should_show = not self._auto_follow_enabled and not self.is_near_bottom()
        self.jump_to_latest_button.setVisible(should_show)
