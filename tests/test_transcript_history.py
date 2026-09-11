import unittest

from PySide6.QtWidgets import QApplication

from ui.widgets.transcript import ChatTranscriptWidget, ConversationTurnWidget


class TranscriptHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_history_is_restored_in_batches_without_rebuilding_existing_turns(self):
        widget = ChatTranscriptWidget()
        payload = {"turns": [
            {"user_text": str(i), "blocks": [{"type": "assistant", "markdown": f"Answer {i}"}]}
            for i in range(55)
        ]}
        try:
            widget.load_transcript(payload)
            self.assertEqual(len(widget._older_turns), 45)
            last = widget.last_turn()
            def turns():
                return [widget.layout.itemAt(i).widget() for i in range(widget.layout.count())
                        if isinstance(widget.layout.itemAt(i).widget(), ConversationTurnWidget)]
            self.assertEqual(len(turns()), 10)
            widget.load_older_turns()
            self.assertEqual(len(turns()), 20)
            self.assertIs(widget.last_turn(), last)
            for expected_count in (30, 40, 50, 55):
                widget.load_older_turns()
                self.assertEqual(len(turns()), expected_count)
                self.assertIs(widget.last_turn(), last)
            self.assertFalse(widget._older_turns)
            self.assertTrue(widget._history_button.isHidden())
            widget.load_transcript({"turns": []})
            self.assertIsNone(widget.last_turn())
            self.assertFalse(widget._older_turns)
        finally:
            widget.close()
            widget.deleteLater()

    def test_loading_older_history_preserves_viewport_after_layout_settles(self):
        from PySide6.QtTest import QTest

        widget = ChatTranscriptWidget()
        widget.resize(800, 600)
        widget.show()
        try:
            widget.load_transcript({"turns": [
                {"user_text": str(i), "blocks": [
                    {"type": "assistant", "markdown": "long wrapped text " * 100}
                ]} for i in range(55)
            ]})
            QTest.qWait(200)
            scrollbar = widget.scroll.verticalScrollBar()
            for _ in range(5):
                scrollbar.setValue(0)
                anchor = widget.layout.itemAt(1).widget()
                before = anchor.mapTo(widget.scroll.viewport(), anchor.rect().topLeft()).y()
                widget._history_button.click()
                QTest.qWait(250)
                after = anchor.mapTo(widget.scroll.viewport(), anchor.rect().topLeft()).y()
                self.assertLessEqual(abs(after - before), 1)
                self.assertFalse(widget.auto_follow_enabled)
            scrollbar.setValue(max(0, scrollbar.value() - 100))
            self.assertIsNone(widget._history_anchor)
            widget.scroll_to_bottom()
            self.assertTrue(widget.auto_follow_enabled)
        finally:
            widget.close()
            widget.deleteLater()

    def test_pending_follow_is_cancelled_when_loading_older_history(self):
        from PySide6.QtTest import QTest

        widget = ChatTranscriptWidget()
        widget.resize(800, 600)
        widget.show()
        try:
            widget.load_transcript({"turns": [
                {"user_text": str(i), "blocks": [
                    {"type": "assistant", "markdown": "wrapped text " * 100}
                ]} for i in range(45)
            ]})
            QTest.qWait(200)
            widget.scroll.verticalScrollBar().setValue(0)
            widget.queue_scroll_to_bottom(force=True)
            widget._schedule_follow_up_scroll(force=True)
            anchor = widget.layout.itemAt(1).widget()
            before = anchor.mapTo(widget.scroll.viewport(), anchor.rect().topLeft()).y()
            widget.load_older_turns()
            QTest.qWait(250)
            after = anchor.mapTo(widget.scroll.viewport(), anchor.rect().topLeft()).y()
            self.assertLessEqual(abs(after - before), 1)
            self.assertFalse(widget.auto_follow_enabled)
        finally:
            widget.close()
            widget.deleteLater()

    def test_deferred_follow_does_not_access_deleted_widget(self):
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtTest import QTest
        from shiboken6 import isValid

        widget = ChatTranscriptWidget()
        widget.queue_scroll_to_bottom(force=True)
        widget._schedule_follow_up_scroll(force=True)
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertFalse(isValid(widget))
        QTest.qWait(120)

    def test_layout_range_clamping_does_not_enable_follow_or_release_anchor(self):
        from PySide6.QtTest import QTest

        widget = ChatTranscriptWidget()
        widget.resize(800, 600)
        widget.show()
        try:
            widget.load_transcript({"turns": [
                {"user_text": str(i), "blocks": [
                    {"type": "assistant", "markdown": "wrapped text " * 100}
                ]} for i in range(25)
            ]})
            QTest.qWait(200)
            scrollbar = widget.scroll.verticalScrollBar()
            scrollbar.setValue(0)
            widget.load_older_turns()
            QTest.qWait(200)
            anchor = widget._history_anchor
            # A layout range shrink clamps the value, emitting valueChanged
            # without any wheel, keyboard or scrollbar interaction.
            scrollbar.setRange(0, 0)
            self.assertIs(widget._history_anchor, anchor)
            self.assertFalse(widget.auto_follow_enabled)
            widget.notify_content_changed(force=True)
            QTest.qWait(200)
            self.assertIs(widget._history_anchor, anchor)
            self.assertFalse(widget.auto_follow_enabled)
        finally:
            widget.close()
            widget.deleteLater()

    def test_history_button_paints_loading_state_before_inserting_turns(self):
        from unittest import mock
        from PySide6.QtCore import QEvent, QObject
        from PySide6.QtTest import QTest

        painted_labels = []

        class PaintObserver(QObject):
            def eventFilter(self, watched, event):
                if event.type() == QEvent.Type.Paint:
                    painted_labels.append(watched.text())
                return False

        widget = ChatTranscriptWidget()
        widget.resize(800, 600)
        widget.show()
        try:
            widget.load_transcript({"turns": [{"user_text": str(i)} for i in range(25)]})
            QTest.qWait(200)
            widget.scroll.verticalScrollBar().setValue(0)
            QTest.qWait(50)
            button = widget._history_button
            observer = PaintObserver(button)
            button.installEventFilter(observer)
            insert = widget._insert_restored_turn

            def insert_while_loading(data, index):
                self.assertEqual(button.text(), "Loading earlier messages…")
                self.assertIn(button.text(), painted_labels)
                self.assertFalse(button.isEnabled())
                # Repeated requests during the same batch are ignored.
                widget.load_older_turns()
                insert(data, index)

            with mock.patch.object(widget, "_insert_restored_turn", side_effect=insert_while_loading) as restored:
                button.click()
            self.assertEqual(restored.call_count, 10)
            self.assertEqual(button.text(), "Load earlier messages (5)")
            self.assertTrue(button.isEnabled())
            self.assertFalse(widget._loading_history)
            button.click()
            self.assertTrue(button.isHidden())
            self.assertFalse(widget._loading_history)
        finally:
            widget.close()
            widget.deleteLater()

    def test_history_button_leaves_loading_state_on_restore_error(self):
        from unittest import mock

        widget = ChatTranscriptWidget()
        try:
            widget.load_transcript({"turns": [{"user_text": str(i)} for i in range(35)]})
            with mock.patch.object(widget, "_insert_restored_turn", side_effect=RuntimeError("restore failed")):
                with self.assertRaisesRegex(RuntimeError, "restore failed"):
                    widget.load_older_turns()
            self.assertFalse(widget._loading_history)
            self.assertTrue(widget._history_button.isEnabled())
            self.assertTrue(widget._history_button.text().startswith("Load earlier messages ("))
            self.assertTrue(widget.updatesEnabled())
        finally:
            widget.close()
            widget.deleteLater()
