import os
import shutil
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QMimeData, QModelIndex, QObject, QPoint, Qt, QtMsgType, Signal
from PySide6.QtGui import QIcon, QImage, QKeyEvent, QTextCursor, QTextFormat
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QFrame, QMessageBox, QPushButton, QSizePolicy, QToolBar, QToolButton, QWidget

import qtawesome as qta

import main as agent_cli
from core.model_fetcher import ModelEntry
from core.text_utils import prepare_markdown_for_render, split_markdown_segments
from core.model_profiles import normalize_profiles_payload
from core.tool_policy import ToolMetadata
from ui.runtime import build_runtime_snapshot, summarize_approval_request
from ui.runtime_worker import AgentRunWorker, AgentRuntimeController
from ui.streaming import StreamEvent
from ui.theme import AMBER_WARNING, BORDER, ERROR_RED, SUCCESS_GREEN, SURFACE_BG, SURFACE_CARD, TEXT_MUTED, build_stylesheet
from ui.widgets.composer import _ComposerMentionItemWidget
from ui.widgets.foundation import AutoTextBrowser, CodeBlockWidget, CopySafePlainTextEdit, DiffBlockWidget, TRANSCRIPT_MAX_WIDTH
from ui.widgets.messages import AssistantMessageWidget, NoticeWidget
from ui.widgets.sidebar import SessionListModel
from ui.widgets.transcript import ConversationTurnWidget
from ui.widgets.tool_group import ToolGroupWidget
from ui.widgets.tools import ToolCardWidget


class FakeTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class FakeToolRegistry:
    def __init__(self):
        self.tools = [
            FakeTool("read_file", "Read a file"),
            FakeTool("edit_file", "Edit a file in place"),
            FakeTool("context7:resolve-library-id", "Resolve a Context7 library id"),
        ]
        self.tool_metadata = {
            "read_file": ToolMetadata(name="read_file", read_only=True),
            "edit_file": ToolMetadata(name="edit_file", mutating=True, requires_approval=True),
            "context7:resolve-library-id": ToolMetadata(
                name="context7:resolve-library-id",
                read_only=True,
                networked=True,
                source="mcp",
            ),
        }
        self.checkpoint_info = {
            "backend": "sqlite",
            "resolved_backend": "sqlite",
            "target": ".agent_state/checkpoints.sqlite",
            "warnings": [],
        }
        self.mcp_server_status = [
            {"server": "context7", "loaded_tools": ["resolve-library-id"], "error": ""},
        ]
        self.loader_status = []

    def get_runtime_status_lines(self):
        return [
            "Checkpoint: requested=sqlite active=sqlite target=.agent_state/checkpoints.sqlite",
            "MCP context7: loaded 1 tool(s)",
        ]


class FakeController(QObject):
    initialized = Signal(object)
    initialization_failed = Signal(str)
    event_emitted = Signal(object)
    approval_requested = Signal(object)
    user_choice_requested = Signal(object)
    session_changed = Signal(object)
    busy_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.start_calls: list[object] = []
        self.resume_calls: list[tuple[bool, bool]] = []
        self.resume_choice_calls: list[str] = []
        self.new_session_calls = 0
        self.switch_session_calls: list[str] = []
        self.delete_session_calls: list[str] = []
        self.delete_project_calls: list[str] = []
        self.set_active_profile_calls: list[str] = []
        self.save_profiles_calls: list[dict] = []
        self.reinitialize_calls: list[bool] = []
        self.set_tool_enabled_calls: list[tuple[str, bool]] = []
        self.set_mcp_server_enabled_calls: list[tuple[str, bool]] = []
        self.shutdown_calls = 0
        self.initialize_calls = 0

    def initialize(self):
        self.initialize_calls += 1

    def start_run(self, text: object):
        self.start_calls.append(text)

    def resume_approval(self, approved: bool, always: bool = False):
        self.resume_calls.append((approved, always))

    def resume_user_choice(self, chosen: str):
        self.resume_choice_calls.append(chosen)

    def new_session(self):
        self.new_session_calls += 1

    def switch_session(self, session_id: str):
        self.switch_session_calls.append(session_id)

    def delete_session(self, session_id: str):
        self.delete_session_calls.append(session_id)

    def delete_project(self, project_path: str):
        self.delete_project_calls.append(project_path)

    def set_active_profile(self, profile_id: str):
        self.set_active_profile_calls.append(profile_id)

    def save_profiles(self, config_payload: dict):
        self.save_profiles_calls.append(config_payload)

    def reinitialize(self, force_new_session: bool = False):
        self.reinitialize_calls.append(force_new_session)

    def set_tool_enabled(self, name: str, enabled: bool):
        self.set_tool_enabled_calls.append((name, enabled))

    def set_mcp_server_enabled(self, name: str, enabled: bool):
        self.set_mcp_server_enabled_calls.append((name, enabled))

    def shutdown(self):
        self.shutdown_calls += 1


class RuntimeControllerStopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_force_stop_watchdog_recovers_unresponsive_worker(self):
        controller = AgentRuntimeController()
        real_thread = controller._thread
        real_worker = controller._worker
        real_stop_worker_thread = controller._stop_worker_thread
        real_start_worker_thread = controller._start_worker_thread
        events = []
        busy_values = []
        controller.event_emitted.connect(events.append)
        controller.busy_changed.connect(busy_values.append)
        try:
            controller._force_stop_timer.stop()
            controller._worker_busy = True
            controller._disconnect_worker_thread()
            controller._thread = SimpleNamespace(isRunning=lambda: True)
            controller._worker = SimpleNamespace()
            controller._stop_worker_thread = mock.Mock()
            controller._start_worker_thread = mock.Mock()

            controller._force_stop_hung_worker()

            self.assertTrue(any(event.type == "run_failed" and event.payload.get("forced") for event in events))
            self.assertIn(False, busy_values)
            controller._stop_worker_thread.assert_called_once_with(force=True)
            controller._start_worker_thread.assert_called_once()
        finally:
            controller._thread = real_thread
            controller._worker = real_worker
            controller._stop_worker_thread = real_stop_worker_thread
            controller._start_worker_thread = real_start_worker_thread
            if real_thread is not None and real_thread.isRunning():
                real_thread.quit()
                real_thread.wait(1000)

    def test_worker_shutdown_finalizes_async_generators_before_closing_loop(self):
        worker = AgentRunWorker()
        client = mock.AsyncMock()

        async def _stream_resource():
            try:
                yield "ready"
            finally:
                await client.aclose()

        loop = worker._ensure_loop()
        stream_holder = {}

        async def _start_stream():
            stream_holder["stream"] = _stream_resource()
            return await anext(stream_holder["stream"])

        self.assertEqual(loop.run_until_complete(_start_stream()), "ready")

        worker.shutdown()

        client.aclose.assert_awaited_once()
        self.assertTrue(loop.is_closed())
        self.assertIsNone(worker._loop)

    def test_stop_signal_uses_queued_connection_for_worker_affinity(self):
        controller = AgentRuntimeController()
        worker = controller._worker
        request_stop = mock.Mock()
        worker.request_stop = request_stop
        try:
            controller.stop_run()
            request_stop.assert_called_once_with()
        finally:
            thread = controller._thread
            controller._force_stop_timer.stop()
            controller._disconnect_worker_thread()
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(1000)

    def test_controller_shutdown_requests_cancellation_before_stopping_thread(self):
        controller = AgentRuntimeController()
        real_thread = controller._thread
        real_worker = controller._worker
        calls = []
        try:
            controller.stop_run = lambda: calls.append("stop")
            controller._stop_worker_thread = lambda *, force: calls.append(("shutdown", force))

            controller.shutdown()

            self.assertEqual(calls, ["stop", ("shutdown", False)])
        finally:
            controller._thread = real_thread
            controller._worker = real_worker
            if real_thread is not None and real_thread.isRunning():
                real_thread.quit()
                real_thread.wait(1000)


class GuiUxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.controller = FakeController()
        self.window = agent_cli.MainWindow(controller=self.controller, auto_initialize=False)

    def tearDown(self):
        self.window.close()

    def _process_events(self):
        self.app.processEvents()

    def _wait_for_gui(self, ms: int):
        QTest.qWait(ms)
        self._process_events()

    def test_cli_exec_batches_output_until_timer_flush(self):
        from ui.widgets.tools import CliExecWidget

        widget = CliExecWidget("echo test")
        try:
            widget.append_output("a")
            widget.append_output("b")
            self.assertEqual(widget.output_view.toPlainText(), "")
            widget._flush_pending_output()
            self.assertEqual(widget.output_view.toPlainText(), "ab")
        finally:
            widget.deleteLater()

    def test_cli_exec_final_output_flushes_pending_chunks(self):
        from ui.widgets.tools import CliExecWidget

        widget = CliExecWidget("echo test")
        try:
            widget.append_output("partial")
            widget.ensure_final_output("partial\nfinal\n")
            self.assertEqual(widget.output_view.toPlainText(), "partial\nfinal\n")
        finally:
            widget.deleteLater()

    def test_tool_cards_use_semantic_qtawesome_icons(self):
        icon_cases = {
            "read_file": "fa5s.file-alt",
            "write_file": "fa5s.file-signature",
            "edit_file": "fa5s.edit",
            "list_directory": "fa5s.folder-open",
            "batch_web_search": "fa5s.globe",
            "fetch_content": "fa5s.link",
            "crawl_site": "fa5s.spider",
            "cli_exec": "fa5s.terminal",
            "safe_delete_file": "fa5s.trash-alt",
            "download_file": "fa5s.download",
            "run_background_process": "fa5s.play",
            "stop_background_process": "fa5s.stop",
            "find_process_by_port": "fa5s.network-wired",
            "request_user_input": "fa5s.comment-dots",
        }

        for name, expected_icon in icon_cases.items():
            with self.subTest(name=name):
                self.assertEqual(ToolCardWidget._tool_icon_name(name), expected_icon)

    def test_mcp_tools_share_one_qtawesome_icon(self):
        expected_icon = "fa5s.cubes"

        self.assertEqual(ToolCardWidget._tool_icon_name("resolve_library_id", "mcp"), expected_icon)
        self.assertEqual(ToolCardWidget._tool_icon_name("query_docs", "MCP"), expected_icon)

    def test_tool_icon_stays_semantic_after_completion(self):
        with mock.patch("ui.widgets.tools._fa_icon", return_value=QIcon()) as icon_factory:
            card = ToolCardWidget({"name": "read_file", "phase": "running"})
            try:
                card.finish({"name": "read_file"})
            finally:
                card.deleteLater()

        requested_names = [call.args[0] for call in icon_factory.call_args_list]
        self.assertIn("fa5s.file-alt", requested_names)
        self.assertNotIn("fa5s.check-circle", requested_names)

    def test_tool_titles_change_to_past_tense_after_successful_completion(self):
        title_cases = {
            "read_file": ("Reading", "Read"),
            "write_file": ("Writing", "Wrote"),
            "edit_file": ("Editing", "Edited"),
            "list_directory": ("Listing", "Listed"),
            "batch_web_search": ("Searching", "Searched"),
            "fetch_content": ("Fetching", "Fetched"),
            "crawl_site": ("Crawling", "Crawled"),
            "cli_exec": ("Running", "Ran"),
            "safe_delete_file": ("Deleting", "Deleted"),
            "download_file": ("Downloading", "Downloaded"),
            "run_background_process": ("Starting process", "Started process"),
            "stop_background_process": ("Stopping process", "Stopped process"),
            "find_process_by_port": ("Finding process", "Found process"),
            "request_user_input": ("Requesting input", "Requested input"),
        }

        for name, (active_title, completed_title) in title_cases.items():
            with self.subTest(name=name):
                card = ToolCardWidget({"name": name, "phase": "running"})
                self.assertEqual(card.action_label.full_text(), active_title)
                card.finish({"name": name})
                self.assertEqual(card.action_label.full_text(), completed_title)
                card.deleteLater()

    def test_safe_delete_cards_render_paths_like_other_filesystem_tools(self):
        cases = (
            ("safe_delete_file", "obsolete.txt"),
            ("safe_delete_directory", "old-cache"),
        )

        for name, path in cases:
            with self.subTest(name=name):
                card = ToolCardWidget({"name": name, "args": {"path": path}, "phase": "running"})
                self.assertEqual(card.action_label.full_text(), f"Deleting {path}")
                self.assertIn("color:#7CC7FF", card.action_label.text())

                card.finish({"name": name, "args": {"path": path}, "content": "Success"})

                self.assertEqual(card.action_label.full_text(), f"Deleted {path}")
                self.assertNotIn("safe_delete_", card.action_label.full_text())
                card.deleteLater()

    def test_crawl_site_card_renders_root_url(self):
        card = ToolCardWidget(
            {
                "name": "crawl_site",
                "args": {"url": "https://example.com/docs"},
                "phase": "running",
            }
        )
        self.assertEqual(card.action_label.full_text(), "Crawling https://example.com/docs")
        card.finish(
            {
                "name": "crawl_site",
                "args": {"url": "https://example.com/docs"},
                "content": "Crawl completed",
            }
        )
        self.assertEqual(card.action_label.full_text(), "Crawled https://example.com/docs")
        self.assertNotIn("crawl_site", card.action_label.full_text())
        card.deleteLater()

    def test_tool_title_keeps_action_form_for_errors(self):
        card = ToolCardWidget({"name": "read_file", "phase": "running", "is_error": True})
        self.assertEqual(card.action_label.full_text(), "Reading failed")
        card.finish({"name": "read_file", "is_error": True})
        self.assertEqual(card.action_label.full_text(), "Reading failed")
        card.deleteLater()

    def test_tool_group_titles_include_file_and_command_counts(self):
        cases = (
            (["write_file", "write_file"], "Wrote 2 files"),
            (["write_file", "edit_file"], "Edited 2 files"),
            (["read_file", "read_file"], "Read 2 files"),
            (["cli_exec", "cli_exec"], "Ran 2 commands"),
        )
        for names, expected_title in cases:
            with self.subTest(names=names):
                group = ToolGroupWidget(parent=self.window)
                for index, name in enumerate(names):
                    card = ToolCardWidget(
                        {"tool_id": f"group-{index}", "name": name, "phase": "finished"},
                        parent=group.container,
                    )
                    group.add_tool(card)
                group.refresh_completion()
                self.assertEqual(group.header_btn.text(), expected_title)
                group.deleteLater()

    def test_tool_group_error_title_includes_total_tools_and_errors(self):
        group = ToolGroupWidget(parent=self.window)
        for index, is_error in enumerate((False, True, True)):
            card = ToolCardWidget(
                {
                    "tool_id": f"error-group-{index}",
                    "name": "read_file",
                    "phase": "finished",
                    "is_error": is_error,
                },
                parent=group.container,
            )
            group.add_tool(card)

        group.refresh_completion()

        self.assertEqual(group.header_btn.text(), "Completed 3 tools with 2 errors")
        group.deleteLater()

    def test_transcript_does_not_duplicate_preface_after_tool_group_with_minor_text_drift(self):
        turn = ConversationTurnWidget("user", parent=self.window)
        preface = (
            "Поиск по коду не нашёл прямого использования `StateGraph` в `.py` файлах — только логи. "
            "Поищу шире: файлы графа, импорты `langgraph`, и параллельно запрошу документацию."
        )

        turn.set_assistant_markdown(preface)
        turn.start_tool({"tool_id": "search-graph", "name": "list_directory", "args": {"path": "core"}})
        turn.set_assistant_markdown(
            "Поиск по коду не нашёл прямого использования `StateGraph` в `.py` файлах — только логи. "
            "Поищу шире: файлы графа, импорты `langgraph`, и параллельно запрошу документацию. "
            "Нашёл — граф строится в `agent.py`, а узлы лежат в `nodes/`."
        )

        self.assertEqual(len(turn.assistant_segments), 2)
        self.assertEqual(turn.assistant_segments[0].markdown(), preface)
        self.assertEqual(
            turn.assistant_segments[1].markdown(),
            "Нашёл — граф строится в `agent.py`, а узлы лежат в `nodes/`.",
        )

    def test_transcript_ignores_same_preface_replay_after_tool_group(self):
        turn = ConversationTurnWidget("user", parent=self.window)
        preface = "Посмотрю конфигурацию лимитов и логику self-correction."

        turn.set_assistant_markdown(preface)
        turn.start_tool({"tool_id": "search-config", "name": "list_directory", "args": {"path": "core"}})
        turn.set_assistant_markdown(preface)

        self.assertEqual(turn.block_kinds(), ["user", "assistant", "tool_group"])
        self.assertEqual(len(turn.assistant_segments), 1)
        self.assertEqual(turn.assistant_segments[0].markdown(), preface)

    def test_transcript_does_not_create_assistant_block_for_invisible_markdown(self):
        turn = ConversationTurnWidget("user", parent=self.window)

        for invisible_markdown in ("```\n", "\u200b\ufeff"):
            with self.subTest(invisible_markdown=repr(invisible_markdown)):
                turn.set_assistant_markdown(invisible_markdown)

                self.assertEqual(turn.block_kinds(), ["user"])
                self.assertEqual(turn.assistant_segments, [])

    def test_invisible_assistant_fragment_does_not_split_tool_group(self):
        turn = ConversationTurnWidget("user", parent=self.window)
        turn.start_tool({"tool_id": "first", "name": "read_file", "args": {"path": "a.txt"}})
        turn.finish_tool({"tool_id": "first", "name": "read_file", "content": "a"})
        first_group = turn.tool_group

        turn.set_assistant_markdown("\u200b\ufeff")
        turn.start_tool({"tool_id": "second", "name": "read_file", "args": {"path": "b.txt"}})

        self.assertIs(turn.tool_group, first_group)
        self.assertEqual(turn.block_kinds(), ["user", "tool_group"])
        self.assertEqual(turn.assistant_segments, [])

    def test_transcript_keeps_new_comment_when_shared_prefix_is_partial_word(self):
        turn = ConversationTurnWidget("user", parent=self.window)
        previous = "Посмотрю конфигурацию лимитов и логику self-correction."
        incoming = "Посмотрю структуру проекта."

        turn.set_assistant_markdown(previous)
        turn.start_tool({"tool_id": "search-config", "name": "list_directory", "args": {"path": "core"}})
        turn.set_assistant_markdown("По")
        turn.set_assistant_markdown(incoming)

        self.assertEqual(turn.assistant_segments[-1].markdown(), incoming)
        self.assertEqual(
            ConversationTurnWidget._assistant_resume_text(
                previous,
                previous,
            ),
            "",
        )

    def test_status_spinner_uses_qtawesome_spin_in_chat_while_busy(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Проверь"}))
        self._process_events()

        status_widget = self.window.current_turn.status_widget
        self.assertIsNotNone(status_widget)
        status_widget.show()
        self._process_events()
        self.assertFalse(hasattr(self.window, "status_icon"))
        self.assertIsInstance(status_widget.spinner, QToolButton)
        self.assertIsInstance(status_widget._spinner_animation, qta.Spin)
        self.assertIs(status_widget._spinner_animation.parent_widget, status_widget.spinner)

        animation = status_widget._spinner_animation
        with mock.patch.object(animation, "stop", wraps=animation.stop) as stop_mock:
            status_widget.set_state("Ready", phase="success")
        stop_mock.assert_called_once_with()

        self.assertIs(status_widget._spinner_animation, animation)

    def test_user_cancel_does_not_render_canceled_notice_in_transcript(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Останови"}))

        self.window._handle_event(StreamEvent("run_failed", {"message": "Cancelled"}))
        self._process_events()

        self.assertIsNotNone(self.window.current_turn)
        self.assertEqual(self.window.current_turn.block_kinds(), ["user"])
        rendered_labels = [label.text() for label in self.window.current_turn.findChildren(QLabel)]
        self.assertNotIn("Canceled", rendered_labels)
        self.assertNotIn("Cancelled", rendered_labels)
        self.assertIn("stopped", self.window.statusBar().currentMessage().lower())

    def _press_composer_key(self, key: int, text: str = "", modifiers: Qt.KeyboardModifier = Qt.NoModifier):
        self.window.composer.setFocus()
        event = QKeyEvent(QEvent.KeyPress, key, modifiers, text)
        QApplication.sendEvent(self.window.composer, event)
        self._process_events()

    def _submit_text(self, text: str):
        self.window._set_input_enabled(True)
        self.window.composer.setPlainText(text)
        self.window._submit_request()

    def _make_test_image_file(self, name: str = "sample.png") -> str:
        temp_root = Path.cwd() / ".tmp_tests" / f"cli-ux-{time.time_ns()}"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: temp_root.exists() and shutil.rmtree(temp_root, ignore_errors=True))
        image_path = temp_root / name
        image = QImage(18, 12, QImage.Format_ARGB32)
        image.fill(0xFF3A7AFE)
        self.assertTrue(image.save(str(image_path), "PNG"))
        return str(image_path)

    def _make_test_file(self, name: str = "notes.txt", content: str = "demo") -> str:
        temp_root = Path.cwd() / ".tmp_tests" / f"cli-ux-file-{time.time_ns()}"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: temp_root.exists() and shutil.rmtree(temp_root, ignore_errors=True))
        file_path = temp_root / name
        file_path.write_text(content, encoding="utf-8")
        return str(file_path)

    def _snapshot_payload(self):
        config = type(
            "Config",
            (),
            {
                "provider": "openai",
                "openai_model": "gpt-4o",
                "gemini_model": "gemini-1.5-flash",
                "checkpoint_backend": "sqlite",
                "enable_approvals": True,
                "debug": False,
            },
        )()
        session = type(
            "Session",
            (),
            {
                "session_id": "session-1234567890abcdef",
                "thread_id": "thread-abcdef1234567890",
                "approval_mode": "prompt",
                "project_path": "D:/demo/workspace",
                "title": "Current chat",
            },
        )()
        snapshot = build_runtime_snapshot(config, FakeToolRegistry(), session)
        return {
            "snapshot": snapshot,
            "tools": snapshot["tools"],
            "help_markdown": "## Help\n- test",
            "model_capabilities": {"image_input_supported": True},
            "sessions": [
                {
                    "session_id": "session-1234567890abcdef",
                    "thread_id": "thread-abcdef1234567890",
                    "project_path": "D:/demo/workspace",
                    "title": "Current chat",
                    "created_at": "2026-03-31T10:00:00+00:00",
                    "updated_at": "2026-03-31T12:00:00+00:00",
                },
                {
                    "session_id": "session-older",
                    "thread_id": "thread-older",
                    "project_path": "D:/demo/other-project",
                    "title": "Older chat [demo/other-project]",
                    "created_at": "2026-03-30T10:00:00+00:00",
                    "updated_at": "2026-03-30T12:00:00+00:00",
                },
            ],
            "active_session_id": "session-1234567890abcdef",
            "model_profiles": {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "",
                        "supports_image_input": True,
                        "enabled": True,
                    },
                    {
                        "id": "gemini-1-5-flash",
                        "provider": "gemini",
                        "model": "gemini-1.5-flash",
                        "api_key": "gm-demo",
                        "base_url": "",
                        "supports_image_input": False,
                        "enabled": True,
                    },
                ],
            },
            "transcript": {"summary_notice": "", "turns": []},
        }

    def test_main_window_populates_runtime_panels_on_initialize(self):
        payload = self._snapshot_payload()
        self.window._handle_initialized(payload)

        self.assertEqual(self.window.overview_panel._labels["Provider"].text(), "OpenAI")
        self.assertEqual(self.window.overview_panel._labels["Model"].text(), "gpt-4o")
        self.assertEqual(self.window.overview_panel._labels["MCP"].text(), "context7")
        tool_cards = self.window.tools_panel.findChildren(QFrame, "ToolCard")
        self.assertEqual(len(tool_cards), 3)
        switches = self.window.tools_panel.findChildren(QCheckBox, "ToolAvailabilitySwitch")
        self.assertEqual(len(switches), 3)
        mcp_items = self.window.tools_panel.findChildren(QLabel, "MCPToolCardItem")
        self.assertEqual(len(mcp_items), 1)
        self.assertIn("context7:resolve-library-id", mcp_items[0].text())
        self.assertIn("Help", self.window.help_text.toPlainText())
        self.assertEqual(self.window.inspector_panel.tabs.tabText(0), "Run")
        self.assertEqual(self.window.inspector_panel.tabs.tabText(1), "Tools")
        self.assertEqual(self.window.inspector_panel.tabs.tabText(2), "Help")
        self.assertEqual(self.window.splitter.count(), 3)
        self.assertIn("Workdir:", self.window.runtime_meta_label.text())
        self.assertIn("Model: gpt-4o", self.window.runtime_meta_label.text())
        self.assertIn("Tools: 3", self.window.runtime_meta_label.text())
        self.assertEqual(self.window.sidebar.model.session_row_count(), 2)
        self.assertGreaterEqual(self.window.sidebar.model.rowCount(), 4)
        self.assertEqual(self.window.active_session_id, "session-1234567890abcdef")
        self.assertEqual(self.window.status_line_label.text(), "Ready")

    def test_failed_mcp_server_switch_is_unchecked_in_tools_panel(self):
        payload = self._snapshot_payload()
        server = next(item for item in payload["tools"] if item["kind"] == "server")
        server["enabled"] = False
        server["description"] = "MCP server - error: startup failed"
        payload["snapshot"]["tools"] = payload["tools"]

        self.window._handle_initialized(payload)

        switch = next(
            item
            for item in self.window.tools_panel.findChildren(QCheckBox, "ToolAvailabilitySwitch")
            if item.accessibleName() == "context7 enabled"
        )
        descriptions = [
            item.text()
            for item in self.window.tools_panel.findChildren(QLabel, "ToolCardDescription")
        ]
        self.assertFalse(switch.isChecked())
        self.assertTrue(switch.isEnabled())
        self.assertIn("MCP server - error: startup failed", descriptions)

    def test_tool_descriptions_expand_when_title_is_clicked(self):
        self.window._handle_initialized(self._snapshot_payload())
        title = next(
            item
            for item in self.window.tools_panel.findChildren(QToolButton, "ToolCardTitle")
            if item.text() == "read_file"
        )
        details = title.parentWidget().findChild(QWidget, "ToolCardDetails")

        self.assertIsNotNone(details)
        self.assertTrue(details.isHidden())
        self.assertEqual(title.arrowType(), Qt.RightArrow)

        title.click()

        self.assertFalse(details.isHidden())
        self.assertEqual(title.arrowType(), Qt.DownArrow)

    def test_local_tool_switch_stays_pending_until_runtime_is_initialized(self):
        self.window._handle_initialized(self._snapshot_payload())
        switch = next(
            item
            for item in self.window.tools_panel.findChildren(QCheckBox, "ToolAvailabilitySwitch")
            if item.accessibleName() == "edit_file enabled"
        )

        switch.setChecked(False)

        self.assertEqual(self.controller.set_tool_enabled_calls, [("edit_file", False)])
        self.assertEqual(self.controller.set_mcp_server_enabled_calls, [])
        self.assertTrue(switch.isChecked())
        self.assertFalse(switch.isEnabled())
        labels = self.window.tools_panel.findChildren(QLabel, "MCPServerLoadingLabel")
        self.assertEqual([label.text() for label in labels], ["Applying…"])
        self.assertIn("Tools: 3", self.window.runtime_meta_label.text())

        payload = self._snapshot_payload()
        tool = next(item for item in payload["tools"] if item["name"] == "edit_file")
        tool["enabled"] = False
        payload["snapshot"]["tools"] = payload["tools"]
        payload["snapshot"]["tools_count"] = 2
        self.window._handle_initialized(payload)
        self._process_events()

        updated_switch = next(
            item
            for item in reversed(self.window.tools_panel.findChildren(QCheckBox, "ToolAvailabilitySwitch"))
            if item.accessibleName() == "edit_file enabled"
        )
        self.assertFalse(updated_switch.isChecked())
        self.assertTrue(updated_switch.isEnabled())
        self.assertIn("Tools: 2", self.window.runtime_meta_label.text())

    def test_multiple_tool_switches_keep_each_pending_until_its_state_is_confirmed(self):
        self.window._handle_initialized(self._snapshot_payload())
        switches = {
            item.accessibleName(): item
            for item in self.window.tools_panel.findChildren(QCheckBox, "ToolAvailabilitySwitch")
        }

        switches["edit_file enabled"].setChecked(False)
        switches["read_file enabled"].setChecked(False)

        self.assertEqual(
            self.controller.set_tool_enabled_calls,
            [("edit_file", False), ("read_file", False)],
        )
        self.assertFalse(switches["edit_file enabled"].isEnabled())
        self.assertFalse(switches["read_file enabled"].isEnabled())
        self.assertEqual(
            sorted(label.text() for label in self.window.tools_panel.findChildren(QLabel, "MCPServerLoadingLabel")),
            ["Applying…", "Applying…"],
        )

        payload = self._snapshot_payload()
        next(item for item in payload["tools"] if item["name"] == "edit_file")["enabled"] = False
        payload["snapshot"]["tools"] = payload["tools"]
        payload["snapshot"]["tools_count"] = 2
        self.window._handle_initialized(payload)
        self._process_events()

        updated = {
            item.accessibleName(): item
            for item in self.window.tools_panel.findChildren(QCheckBox, "ToolAvailabilitySwitch")
        }
        self.assertTrue(updated["edit_file enabled"].isEnabled())
        self.assertFalse(updated["read_file enabled"].isEnabled())
        pending_labels = self.window.tools_panel.findChildren(QLabel, "MCPServerLoadingLabel")
        self.assertEqual([label.text() for label in pending_labels], ["Applying…"])

    def test_disabled_local_tool_switch_stays_pending_while_enabling(self):
        payload = self._snapshot_payload()
        tool = next(item for item in payload["tools"] if item["name"] == "edit_file")
        tool["enabled"] = False
        payload["snapshot"]["tools"] = payload["tools"]
        payload["snapshot"]["tools_count"] = 2
        self.window._handle_initialized(payload)
        switch = next(
            item
            for item in self.window.tools_panel.findChildren(QCheckBox, "ToolAvailabilitySwitch")
            if item.accessibleName() == "edit_file enabled"
        )

        switch.setChecked(True)

        self.assertEqual(self.controller.set_tool_enabled_calls, [("edit_file", True)])
        self.assertFalse(switch.isChecked())
        self.assertFalse(switch.isEnabled())
        labels = self.window.tools_panel.findChildren(QLabel, "MCPServerLoadingLabel")
        self.assertEqual([label.text() for label in labels], ["Applying…"])
        self.assertIn("Tools: 2", self.window.runtime_meta_label.text())

        confirmed = self._snapshot_payload()
        self.window._handle_initialized(confirmed)
        self._process_events()

        updated_switch = next(
            item
            for item in reversed(self.window.tools_panel.findChildren(QCheckBox, "ToolAvailabilitySwitch"))
            if item.accessibleName() == "edit_file enabled"
        )
        self.assertTrue(updated_switch.isChecked())
        self.assertTrue(updated_switch.isEnabled())
        self.assertIn("Tools: 3", self.window.runtime_meta_label.text())

    def test_local_tool_switch_error_restores_previous_state(self):
        self.window._handle_initialized(self._snapshot_payload())
        switch = next(
            item
            for item in self.window.tools_panel.findChildren(QCheckBox, "ToolAvailabilitySwitch")
            if item.accessibleName() == "edit_file enabled"
        )
        switch.setChecked(False)

        with mock.patch.object(QMessageBox, "critical"):
            self.window._handle_init_failed("Tool runtime did not start")

        self.assertTrue(switch.isChecked())
        self.assertTrue(switch.isEnabled())
        label = self.window.tools_panel.findChild(QLabel, "MCPServerLoadingLabel")
        self.assertIsNotNone(label)
        self.assertEqual(label.text(), "Failed to apply")
        self.assertIn("Tool runtime did not start", label.toolTip())

    def test_mcp_server_switch_stays_pending_until_runtime_is_initialized(self):
        self.window._handle_initialized(self._snapshot_payload())
        switch = next(
            item
            for item in self.window.tools_panel.findChildren(QCheckBox, "ToolAvailabilitySwitch")
            if item.accessibleName() == "context7 enabled"
        )

        switch.setChecked(False)

        self.assertEqual(self.controller.set_mcp_server_enabled_calls, [("context7", False)])
        self.assertTrue(switch.isChecked())
        self.assertFalse(switch.isEnabled())
        labels = self.window.tools_panel.findChildren(QLabel, "MCPServerLoadingLabel")
        self.assertEqual([label.text() for label in labels], ["Applying…"])

        payload = self._snapshot_payload()
        server = next(item for item in payload["tools"] if item["kind"] == "server")
        server["enabled"] = False
        payload["snapshot"]["tools"] = payload["tools"]
        self.window._handle_initialized(payload)
        self._process_events()

        updated_switch = next(
            item
            for item in reversed(self.window.tools_panel.findChildren(QCheckBox, "ToolAvailabilitySwitch"))
            if item.accessibleName() == "context7 enabled"
        )
        self.assertFalse(updated_switch.isChecked())
        self.assertTrue(updated_switch.isEnabled())

    def test_mcp_server_switch_error_restores_previous_state(self):
        self.window._handle_initialized(self._snapshot_payload())
        switch = next(
            item
            for item in self.window.tools_panel.findChildren(QCheckBox, "ToolAvailabilitySwitch")
            if item.accessibleName() == "context7 enabled"
        )
        switch.setChecked(False)

        with mock.patch.object(QMessageBox, "critical"):
            self.window._handle_init_failed("MCP process did not start")

        self.assertTrue(switch.isChecked())
        self.assertTrue(switch.isEnabled())
        label = self.window.tools_panel.findChild(QLabel, "MCPServerLoadingLabel")
        self.assertIsNotNone(label)
        self.assertEqual(label.text(), "Failed to apply")
        self.assertIn("MCP process did not start", label.toolTip())

    def test_tool_descriptions_have_transparent_background_style(self):
        stylesheet = build_stylesheet()
        for selector in ("QWidget#ToolCardDetails", "QLabel#ToolCardDescription", "QLabel#MCPToolCardItem"):
            self.assertIn(selector, stylesheet)
            start = stylesheet.index(selector)
            self.assertIn("background: transparent;", stylesheet[start:stylesheet.index("}", start)])

    def test_configure_qt_logging_adds_font_db_rule_once(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            agent_cli._configure_qt_logging()
            self.assertEqual(os.environ.get("QT_LOGGING_RULES"), "qt.text.font.db=false")
            agent_cli._configure_qt_logging()
            self.assertEqual(os.environ.get("QT_LOGGING_RULES"), "qt.text.font.db=false")

    def test_configure_qt_logging_preserves_existing_rules(self):
        with mock.patch.dict(os.environ, {"QT_LOGGING_RULES": "qt.network.ssl.warning=true"}, clear=False):
            agent_cli._configure_qt_logging()
            self.assertEqual(
                os.environ.get("QT_LOGGING_RULES"),
                "qt.network.ssl.warning=true;qt.text.font.db=false",
            )

    def test_qt_message_filter_suppresses_filesystem_watcher_warning(self):
        from ui.window_components.main_window import _qt_message_filter

        with mock.patch("builtins.print") as mock_print:
            _qt_message_filter(QtMsgType.QtWarningMsg, None, "QFileSystemWatcher: FindNextChangeNotification failed for /tmp")
            mock_print.assert_not_called()

    def test_qt_message_filter_passes_other_warnings(self):
        from ui.window_components.main_window import _qt_message_filter

        with mock.patch("builtins.print") as mock_print:
            _qt_message_filter(QtMsgType.QtWarningMsg, None, "Some other Qt warning")
            mock_print.assert_called_once()

    def test_submit_request_uses_controller_and_clears_editor(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.setPlainText("Собери summary")

        self.window._submit_request()

        self.assertEqual(
            self.controller.start_calls,
            [{"text": "Собери summary", "attachments": []}],
        )
        self.assertEqual(self.window.composer.toPlainText(), "")

    def test_submit_request_includes_pasted_text_and_manual_suffix(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.setPlainText("Вставка: ")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        mime = QMimeData()
        mime.setText("из буфера")
        self.window.composer.insertFromMimeData(mime)
        self.window.composer.insertPlainText(" + вручную")

        self.window._submit_request()

        self.assertEqual(self.controller.start_calls[0]["text"], "Вставка: из буфера + вручную")

    def test_submit_request_sanitizes_and_truncates_input_before_runtime(self):
        self.window._handle_initialized(self._snapshot_payload())
        raw_text = "  start\x00" + ("x" * 10_050)
        self.window.composer.setPlainText(raw_text)

        self.window._submit_request()

        self.assertEqual(len(self.controller.start_calls), 1)
        payload = self.controller.start_calls[0]
        self.assertEqual(len(payload["text"]), 10_000)
        self.assertNotIn("\x00", payload["text"])
        self.assertTrue(payload["text"].startswith("start"))
        self.assertFalse(self.window.composer_notice_label.isHidden())
        self.assertIn("Removed unsupported control characters", self.window.composer_notice_label.text())
        self.assertIn("truncated to 10000 characters", self.window.composer_notice_label.text())

    def test_send_button_keeps_visible_disabled_icon_until_input_exists(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._set_input_enabled(True)

        disabled_icon_key = self.window.send_button.icon().cacheKey()
        self.assertFalse(self.window.send_button.isEnabled())

        self.window.composer.setPlainText("go")
        self.window._refresh_submit_controls()
        self._process_events()

        enabled_icon_key = self.window.send_button.icon().cacheKey()
        self.assertTrue(self.window.send_button.isEnabled())
        self.assertNotEqual(disabled_icon_key, enabled_icon_key)

    def test_assistant_message_widget_renders_unclosed_fenced_block_as_code_widget(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_markdown("Вот код:\n```python\nprint('hi')\n")
        self._process_events()

        self.assertEqual(len(widget.parts_widgets), 2)
        self.assertIsInstance(widget.parts_widgets[0], AutoTextBrowser)
        self.assertIsInstance(widget.parts_widgets[1], CodeBlockWidget)
        self.assertEqual(widget.parts_widgets[1].editor.toPlainText(), "print('hi')")
        self.assertEqual(widget.parts_widgets[1].title_label.text(), "PYTHON")

    def test_assistant_message_widget_keeps_plain_text_unclosed_fence_as_markdown(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_markdown("```\nТеперь проверю, есть ли тесты для validation_missing_write_content через mock:\n")
        self._process_events()

        self.assertEqual(len(widget.parts_widgets), 1)
        self.assertIsInstance(widget.parts_widgets[0], AutoTextBrowser)
        self.assertNotIn("```", widget.parts_widgets[0].toPlainText())
        self.assertIn("Теперь проверю", widget.parts_widgets[0].toPlainText())

    def test_assistant_message_widget_does_not_flash_empty_bare_fence_as_code(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_content("```\n")
        self._process_events()

        self.assertEqual(len(widget.parts_widgets), 1)
        self.assertIsInstance(widget.parts_widgets[0], AutoTextBrowser)
        self.assertEqual(widget.parts_widgets[0].toPlainText(), "")

    def test_assistant_message_widget_replaces_part_widget_when_stream_opens_code_block(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_content("```python")
        self._process_events()
        widget.set_content("```python\nprint('hi')")
        self._process_events()

        self.assertEqual(len(widget.parts_widgets), 1)
        self.assertIsInstance(widget.parts_widgets[0], CodeBlockWidget)
        self.assertEqual(widget.parts_widgets[0].editor.toPlainText(), "print('hi')")

    def test_assistant_message_widget_updates_only_changed_stream_part(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_content("Intro\n\n```python\nprint('hi')\n```\n\nTail")
        self._process_events()
        markdown_widget = widget.parts_widgets[0]
        code_widget = widget.parts_widgets[1]
        markdown_widget.setMarkdown = mock.Mock(wraps=markdown_widget.setMarkdown)
        code_widget.set_code = mock.Mock(wraps=code_widget.set_code)

        widget.set_content("Intro\n\n```python\nprint('hi')\n```\n\nTail updated")
        self._process_events()

        markdown_widget.setMarkdown.assert_not_called()
        code_widget.set_code.assert_not_called()
        self.assertIs(widget.parts_widgets[0], markdown_widget)
        self.assertIs(widget.parts_widgets[1], code_widget)

    def test_assistant_message_widget_incrementally_splits_only_last_segment(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_content("Intro\n\n```python\nprint('hi')\n```\n\nTail")
        self._process_events()

        with mock.patch("ui.widgets.messages.split_markdown_segments", wraps=split_markdown_segments) as split_mock:
            widget.set_content("Intro\n\n```python\nprint('hi')\n```\n\nTail updated")
            self._process_events()

        split_mock.assert_called_once_with("\nTail updated")
        self.assertEqual(len(widget.parts_widgets), 3)
        self.assertIsInstance(widget.parts_widgets[0], AutoTextBrowser)
        self.assertIsInstance(widget.parts_widgets[1], CodeBlockWidget)
        self.assertIsInstance(widget.parts_widgets[2], AutoTextBrowser)

    def test_assistant_message_widget_renders_markdown_live_while_streaming(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_streaming(True)
        widget.set_content("**Live** text")
        self._process_events()
        markdown_widget = widget.parts_widgets[0]

        self.assertIn("Live", markdown_widget.toPlainText())
        self.assertNotIn("**", markdown_widget.toPlainText())
        self.assertNotIn(r"\**Live\**", markdown_widget.toMarkdown())

    def test_assistant_inline_code_keeps_body_font_size_when_closing_backtick_arrives(self):
        widget = AssistantMessageWidget()
        widget.setStyleSheet(build_stylesheet())
        self.addCleanup(widget.deleteLater)

        widget.set_content("- `tests/test_tooling_refactor.py`: `42 passed")
        self._process_events()
        markdown_widget = widget.parts_widgets[0]
        body_font_size = markdown_widget.font().pointSizeF()

        widget.set_content("- `tests/test_tooling_refactor.py`: `42 passed`")
        self._process_events()

        code_sizes = []
        block = markdown_widget.document().begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if fragment.isValid() and fragment.charFormat().font().fixedPitch():
                    code_sizes.append(fragment.charFormat().font().pointSizeF())
                iterator += 1
            block = block.next()

        self.assertTrue(code_sizes)
        self.assertTrue(all(size == body_font_size for size in code_sizes))

    def test_assistant_message_widget_renders_overescaped_markdown(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        markdown = prepare_markdown_for_render(r"""\# Header

\- \*\*bold\*\* item""")
        widget.set_content(markdown)
        self._process_events()

        self.assertEqual(len(widget.parts_widgets), 2)
        header_widget = widget.parts_widgets[0]
        list_widget = widget.parts_widgets[1]
        self.assertIn("Header", header_widget.toPlainText())
        self.assertIn("bold item", list_widget.toPlainText())
        self.assertIn("<ul", list_widget.toHtml().lower())
        self.assertNotIn("**", list_widget.toPlainText())

    def test_assistant_streaming_state_does_not_add_visual_indicator_or_change_geometry(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)
        widget.resize(720, 200)
        widget.set_content("Первая строка ответа.\n\nВторая строка ответа.")
        widget.set_streaming(True)
        widget.show()
        self._wait_for_gui(40)
        streaming_height = widget.sizeHint().height()
        body_font_size = widget.parts_widgets[-1].font().pointSizeF()
        self.assertEqual(widget.findChildren(QLabel, "AssistantStreamCursor"), [])

        widget.set_streaming(False)
        self._wait_for_gui(40)

        self.assertEqual(widget.sizeHint().height(), streaming_height)
        self.assertEqual(widget.parts_widgets[-1].font().pointSizeF(), body_font_size)

    def test_assistant_message_widget_freezes_completed_prose_paragraphs(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_content("First paragraph.\n\nSecond")
        self._process_events()
        first_widget = widget.parts_widgets[0]
        first_widget.setMarkdown = mock.Mock(wraps=first_widget.setMarkdown)

        widget.set_content("First paragraph.\n\nSecond paragraph grows")
        self._process_events()

        self.assertEqual(len(widget.parts_widgets), 2)
        self.assertIs(widget.parts_widgets[0], first_widget)
        first_widget.setMarkdown.assert_not_called()

    def test_assistant_message_widget_closes_compound_block_before_following_prose(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_content("- First item\n- Second item\n\nFollowing paragraph")
        self._process_events()

        self.assertEqual(len(widget.parts_widgets), 2)
        self.assertIsInstance(widget.parts_widgets[0], AutoTextBrowser)
        self.assertIsInstance(widget.parts_widgets[1], AutoTextBrowser)
        self.assertIn("Second item", widget.parts_widgets[0].toPlainText())
        self.assertEqual(widget.parts_widgets[1].toPlainText(), "Following paragraph")

    def test_assistant_message_widget_does_not_plain_text_draft_while_streaming(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_streaming(True)
        widget.set_content("**Live** text")
        self._process_events()
        markdown_widget = widget.parts_widgets[0]
        markdown_widget.setMarkdown = mock.Mock(wraps=markdown_widget.setMarkdown)
        markdown_widget.setPlainText = mock.Mock(wraps=markdown_widget.setPlainText)

        widget.set_content("**Live** text updated")
        self._process_events()

        markdown_widget.setMarkdown.assert_called_once_with("**Live** text updated")
        markdown_widget.setPlainText.assert_not_called()

    def test_assistant_message_widget_incremental_split_restarts_at_tilde_fence(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_content("Intro\n\n~~~python\nprint('hi')\n")
        self._process_events()

        with mock.patch("ui.widgets.messages.split_markdown_segments", wraps=split_markdown_segments) as split_mock:
            widget.set_content("Intro\n\n~~~python\nprint('hi')\n~~~\n\nTail")
            self._process_events()

        split_mock.assert_called_once_with("~~~python\nprint('hi')\n~~~\n\nTail")
        self.assertEqual(len(widget.parts_widgets), 3)
        self.assertIsInstance(widget.parts_widgets[0], AutoTextBrowser)
        self.assertIsInstance(widget.parts_widgets[1], CodeBlockWidget)
        self.assertIsInstance(widget.parts_widgets[2], AutoTextBrowser)

    def test_assistant_message_widget_has_no_thought_panel(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_content("Готово.")
        widget.show()
        self._process_events()

        self.assertFalse(hasattr(widget, "thought_section"))
        self.assertFalse(hasattr(widget, "thought_content"))
        self.assertEqual(widget.markdown(), "Готово.")
        self.assertIs(widget.content_layout.itemAt(0).widget(), widget.parts_widgets[0])

    def test_assistant_message_widget_does_not_render_thinking_placeholder_for_empty_content(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_content("")
        widget.show()
        self._process_events()

        self.assertEqual(widget.markdown(), "")
        self.assertEqual(widget.parts_widgets, [])

    def test_user_choice_card_renders_above_composer_and_resumes_selected_option(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Выбери режим"}))
        self.controller.user_choice_requested.emit(
            {
                "question": "Какой режим выбираем?",
                "recommended_key": "direct_api",
                "options": [
                    {
                        "key": "direct_api",
                        "label": "direct_api: убрать MCP и проверить только API",
                        "submit_text": "direct_api",
                        "recommended": True,
                    },
                    {
                        "key": "keep_mcp",
                        "label": "keep_mcp: оставить MCP и настроить сервер",
                        "submit_text": "keep_mcp",
                        "recommended": False,
                    },
                ],
            }
        )
        self._process_events()

        self.assertFalse(self.window.user_choice_card.isHidden())
        self.assertEqual(self.window.user_choice_card.title_label.text(), "Your input is required")
        self.assertEqual(self.window.user_choice_card.question_label.text(), "Какой режим выбираем?")
        option_buttons = self.window.user_choice_card.findChildren(QPushButton, "UserChoiceOptionButton")
        self.assertEqual(len(option_buttons), 2)
        self.assertTrue(option_buttons[0].property("recommended"))

        QTest.mouseClick(option_buttons[0], Qt.LeftButton)
        self._process_events()

        self.assertEqual(self.controller.start_calls, [])
        self.assertEqual(self.controller.resume_choice_calls, ["direct_api"])
        self.assertTrue(self.window.user_choice_card.isHidden())

    def test_user_choice_card_custom_option_arms_composer_and_resumes_on_submit(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.controller.user_choice_requested.emit(
            {
                "question": "Как продолжаем?",
                "recommended_key": "",
                "options": [
                    {
                        "key": "direct_api",
                        "label": "direct_api: тестируем только API",
                        "submit_text": "direct_api",
                        "recommended": False,
                    }
                ],
            }
        )
        self.window.composer.setPlainText("Мой вариант")
        self._process_events()

        QTest.mouseClick(self.window.user_choice_card.custom_button, Qt.LeftButton)
        self._process_events()

        self.assertFalse(self.window.user_choice_card.isHidden())
        self.assertEqual(self.window.composer.toPlainText(), "Мой вариант")
        self.assertEqual(self.window.composer.textCursor().selectedText(), "Мой вариант")

        self.window._submit_request()

        self.assertEqual(self.controller.start_calls, [])
        self.assertEqual(self.controller.resume_choice_calls, ["Мой вариант"])
        self.assertTrue(self.window.user_choice_card.isHidden())

    def test_composer_enter_submits_while_shift_enter_adds_newline(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.setPlainText("line1")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        self._press_composer_key(Qt.Key_Return, "\n", Qt.ShiftModifier)
        self.assertEqual(self.controller.start_calls, [])
        self.assertIn("\n", self.window.composer.toPlainText())

        self.window.composer.setPlainText("Сделай задачу")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
        self._press_composer_key(Qt.Key_Return, "\r")

        self.assertEqual(
            self.controller.start_calls,
            [{"text": "Сделай задачу", "attachments": []}],
        )
        self.assertEqual(self.window.composer.toPlainText(), "")

    def test_paste_image_creates_draft_attachment_chip(self):
        self.window._handle_initialized(self._snapshot_payload())
        mime = QMimeData()
        image = QImage(14, 10, QImage.Format_ARGB32)
        image.fill(0xFF44AA66)
        mime.setImageData(image)

        self.window.composer.insertFromMimeData(mime)
        self._process_events()

        self.assertEqual(len(self.window.draft_image_attachments), 1)
        self.assertFalse(self.window.composer_attachments_strip.isHidden())
        self.assertEqual(len(self.window.composer_attachments_strip._chips), 1)
        self.assertTrue(self.window.send_button.isEnabled())
        self.assertFalse(self.window.composer_notice_label.isVisible())

    def test_run_started_with_attachments_renders_user_preview_row(self):
        self.window._handle_initialized(self._snapshot_payload())
        attachment = {
            "id": "img-1",
            "path": self._make_test_image_file(),
            "mime_type": "image/png",
            "file_name": "sample.png",
            "width": 18,
            "height": 12,
            "size_bytes": 128,
        }

        self.window._handle_event(
            StreamEvent("run_started", {"text": "Опиши изображение", "attachments": [attachment]})
        )
        self._process_events()

        self.assertIsNotNone(self.window.current_turn)
        user_widget = self.window.current_turn._timeline[0][1]
        self.assertFalse(user_widget.attachments_strip.isHidden())
        self.assertEqual(len(user_widget.attachments_strip._chips), 1)

    def test_no_img_badge_is_visible_and_image_paste_shows_notice(self):
        payload = self._snapshot_payload()
        payload["model_capabilities"] = {"image_input_supported": False}
        payload["model_profiles"]["profiles"][0]["supports_image_input"] = False
        self.window._handle_initialized(payload)
        mime = QMimeData()
        image = QImage(14, 10, QImage.Format_ARGB32)
        image.fill(0xFFAA6644)
        mime.setImageData(image)

        self.window.composer.insertFromMimeData(mime)
        self._process_events()

        self.assertFalse(self.window.model_image_badge.isHidden())
        self.assertEqual(self.window.model_image_badge.text(), "")
        self.assertFalse(self.window.model_image_badge.pixmap().isNull())
        self.assertIn("Image input unavailable", self.window.model_image_badge.toolTip())
        self.assertEqual(self.window.draft_image_attachments, [])
        self.assertFalse(self.window.composer_notice_label.isHidden())
        self.assertIn("does not support image input", self.window.composer_notice_label.text())

    def test_profile_checkbox_can_override_runtime_no_img_badge(self):
        payload = self._snapshot_payload()
        payload["model_capabilities"] = {"image_input_supported": False}
        payload["model_profiles"]["profiles"][0]["supports_image_input"] = True

        self.window._handle_initialized(payload)
        self.window._set_input_enabled(True)

        self.assertTrue(self.window.model_image_badge.isHidden())
        self.assertTrue(self.window.add_image_action.isEnabled())

    def test_insert_file_paths_keeps_existing_text_reference_flow(self):
        self.window._handle_initialized(self._snapshot_payload())
        file_path = self._make_test_file()

        with mock.patch.object(agent_cli.QFileDialog, "getOpenFileNames", return_value=([file_path], "")):
            self.window._insert_file_paths()

        self.assertIn(self.window.composer.format_file_reference(file_path), self.window.composer.toPlainText())
        self.assertEqual(self.window.draft_image_attachments, [])

    def test_insert_file_paths_attaches_images_for_image_capable_model(self):
        self.window._handle_initialized(self._snapshot_payload())
        image_path = self._make_test_image_file("picked.png")

        with mock.patch.object(agent_cli.QFileDialog, "getOpenFileNames", return_value=([image_path], "")):
            self.window._insert_file_paths()

        self.assertEqual(self.window.composer.toPlainText(), "")
        self.assertEqual(len(self.window.draft_image_attachments), 1)
        self.assertEqual(self.window.draft_image_attachments[0]["file_name"], "picked.png")
        self.assertFalse(self.window.composer_attachments_strip.isHidden())

    def test_insert_file_paths_falls_back_to_text_when_image_input_is_disabled(self):
        payload = self._snapshot_payload()
        payload["model_capabilities"] = {"image_input_supported": False}
        payload["model_profiles"]["profiles"][0]["supports_image_input"] = False
        self.window._handle_initialized(payload)
        image_path = self._make_test_image_file("fallback.png")

        with mock.patch.object(agent_cli.QFileDialog, "getOpenFileNames", return_value=([image_path], "")):
            self.window._insert_file_paths()

        self.assertIn(self.window.composer.format_file_reference(image_path), self.window.composer.toPlainText())
        self.assertEqual(self.window.draft_image_attachments, [])
        self.assertFalse(self.window.composer_notice_label.isHidden())
        self.assertIn("does not support image input", self.window.composer_notice_label.text())

    def test_composer_paste_single_line_text_drops_trailing_newline(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.setPlainText("Открой ")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        mime = QMimeData()
        mime.setText("core/nodes.py\r\n")
        self.window.composer.insertFromMimeData(mime)

        self.assertEqual(self.window.composer.toPlainText(), "Открой core/nodes.py")

    def test_composer_paste_single_line_text_drops_leading_and_trailing_newlines(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.setPlainText("Открой ")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        mime = QMimeData()
        mime.setText("\r\ncore/nodes.py\r\n")
        self.window.composer.insertFromMimeData(mime)

        self.assertEqual(self.window.composer.toPlainText(), "Открой core/nodes.py")

    def test_composer_paste_multiline_text_keeps_newlines(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.setPlainText("Список:\n")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        mime = QMimeData()
        mime.setText("first.py\nsecond.py\n")
        self.window.composer.insertFromMimeData(mime)

        self.assertEqual(self.window.composer.toPlainText(), "Список:\nfirst.py\nsecond.py\n")

    def test_composer_drop_local_file_url_does_not_append_raw_file_uri_text(self):
        self.window._handle_initialized(self._snapshot_payload())
        file_path = self._make_test_file("dropped.txt")
        self.window.composer.setPlainText("Открой ")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        mime = QMimeData()
        from PySide6.QtCore import QUrl

        mime.setUrls([QUrl.fromLocalFile(file_path)])
        mime.setText(f"{file_path}file:///{file_path.replace(':', ':/')}")
        self.window.composer.insertFromMimeData(mime)

        self.assertEqual(
            self.window.composer.toPlainText(),
            f"Открой {self.window.composer.format_file_reference(file_path)}",
        )

    def test_copy_safe_plain_text_edit_copies_plain_text_without_hidden_paragraph_breaks(self):
        editor = CopySafePlainTextEdit()
        editor.setPlainText("alpha.py")
        editor.selectAll()
        editor.copy()

        self.assertEqual(QApplication.clipboard().text(), "alpha.py")

    def test_copy_safe_plain_text_edit_strips_spurious_edge_newlines_for_single_line_selection(self):
        editor = CopySafePlainTextEdit()
        editor.setPlainText("\nalpha.py\n")
        cursor = editor.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        editor.setTextCursor(cursor)

        mime = editor.createMimeDataFromSelection()

        self.assertEqual(mime.text(), "alpha.py")

    def test_auto_text_browser_copy_normalizes_qt_paragraph_separators(self):
        browser = AutoTextBrowser()
        browser.setMarkdown("first line\n\nsecond line")
        cursor = browser.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        browser.setTextCursor(cursor)

        browser.copy()

        self.assertEqual(QApplication.clipboard().text(), "first line\nsecond line")

    def test_auto_text_browser_copy_strips_spurious_edge_newlines_for_single_line_selection(self):
        browser = AutoTextBrowser()
        browser.setMarkdown("alpha.py")
        cursor = browser.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        browser.setTextCursor(cursor)

        mime = browser.createMimeDataFromSelection()

        self.assertEqual(mime.text(), "alpha.py")

    def test_auto_text_browser_highlights_filenames_in_blue(self):
        browser = AutoTextBrowser()
        browser.setMarkdown("Смотри файл main.py и путь src/ui/theme.py, а также Dockerfile.")

        highlighted = self._filename_highlighted_texts(browser)

        self.assertIn("main.py", highlighted)
        self.assertIn("src/ui/theme.py", highlighted)
        self.assertIn("Dockerfile", highlighted)

    def test_auto_text_browser_keeps_links_and_fenced_code_out_of_filename_highlight(self):
        browser = AutoTextBrowser()
        browser.setMarkdown(
            "Инлайн `theme.py`, [ссылка](main.py) и блок:\n\n```\nmain.py в коде\n```\n"
        )

        highlighted = self._filename_highlighted_texts(browser)

        self.assertEqual(highlighted, ["theme.py"])

    def test_auto_text_browser_filename_highlight_survives_streaming_updates(self):
        browser = AutoTextBrowser()
        for text in (
            "Отредактировал ",
            "Отредактировал main",
            "Отредактировал main.py",
            "Отредактировал main.py и core/utils.py",
        ):
            browser.setMarkdown(text)

        highlighted = self._filename_highlighted_texts(browser)

        self.assertIn("main.py", highlighted)
        self.assertIn("core/utils.py", highlighted)
        self.assertEqual(highlighted.count("main.py"), 1)

    @staticmethod
    def _filename_highlighted_texts(browser: AutoTextBrowser) -> list[str]:
        from ui.theme import FILENAME_BLUE

        highlighted: list[str] = []
        block = browser.document().firstBlock()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if fragment.isValid():
                    color = fragment.charFormat().foreground().color()
                    if color.name().upper() == FILENAME_BLUE.upper():
                        highlighted.append(fragment.text())
                iterator += 1
            block = block.next()
        return highlighted

    def test_composer_history_navigation_works_when_empty_and_dedupes_adjacent_entries(self):
        self.window._handle_initialized(self._snapshot_payload())
        self._submit_text("cmd alpha")
        self._submit_text("cmd alpha")
        self._submit_text("cmd beta")
        self.window._set_input_enabled(True)
        self.window.composer.clear()

        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "cmd beta")
        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "cmd alpha")
        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "cmd alpha")
        self._press_composer_key(Qt.Key_Down)
        self.assertEqual(self.window.composer.toPlainText(), "cmd beta")
        self._press_composer_key(Qt.Key_Down)
        self.assertEqual(self.window.composer.toPlainText(), "")

    def test_composer_up_down_do_not_override_text_when_editor_not_empty(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.append_submitted_message("history item")
        self.window.composer.setPlainText("typed")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        self._press_composer_key(Qt.Key_Up)

        self.assertEqual(self.window.composer.toPlainText(), "typed")

    def test_composer_history_is_restored_from_transcript_payload(self):
        payload = self._snapshot_payload()
        payload["transcript"] = {
            "summary_notice": "",
            "turns": [
                {"user_text": "first", "blocks": []},
                {"user_text": "first", "blocks": []},
                {"user_text": "second", "blocks": []},
            ],
        }
        self.window._handle_initialized(payload)
        self.window.composer.clear()

        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "second")
        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "first")

    def test_composer_history_is_session_scoped(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.set_history_session("session-a")
        self.window.composer.append_submitted_message("alpha")
        self.window.composer.set_history_session("session-b")
        self.window.composer.append_submitted_message("beta")

        self.window.composer.set_history_session("session-a")
        self.window.composer.clear()
        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "alpha")

        self.window.composer.set_history_session("session-b")
        self.window.composer.clear()
        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "beta")

    def test_mention_popup_selects_file_and_has_priority_over_submit(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.show()
        self._process_events()
        self.window.composer.set_file_index_for_testing(
            ["main.py", "manual/main_notes.md", "core/gui_widgets.py"]
        )
        self.window.composer.setPlainText("@ma")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        self._press_composer_key(Qt.Key_I, "i")
        self.window.composer._refresh_mention_popup()
        self.assertTrue(self.window.composer._mention_popup.isVisible())

        self.window.composer._mention_popup.move_selection(1)
        self.window.composer._accept_current_mention()

        self.assertEqual(self.window.composer.toPlainText(), "manual/main_notes.md")
        self.assertFalse(self.window.composer._mention_popup.isVisible())
        self.assertEqual(self.controller.start_calls, [])

    def test_mention_popup_opens_on_at_character_input(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.show()
        self._process_events()
        self.window.composer.set_file_index_for_testing(["main.py"])
        at_key = Qt.Key_At if hasattr(Qt, "Key_At") else Qt.Key_A

        self._press_composer_key(at_key, "@")

        self.assertTrue(self.window.composer._mention_popup.isVisible())

    def test_mention_popup_shows_root_files_first_and_is_wider(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.show()
        self._process_events()
        self.window.composer.set_file_index_for_testing(
            ["sub/main.py", "root.py", "nested/deep/file.txt"]
        )
        self.window.composer.setPlainText("@")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
        self.window.composer._refresh_mention_popup()

        self.assertTrue(self.window.composer._mention_popup.isVisible())
        self.assertEqual(self.window.composer._mention_popup.current_relative_path(), "root.py")
        self.assertGreaterEqual(self.window.composer._mention_popup.width(), 560)
        first_item = self.window.composer._mention_popup.list_widget.item(0)
        first_widget = self.window.composer._mention_popup.list_widget.itemWidget(first_item)
        self.assertEqual(first_item.toolTip(), "")
        self.assertIsNotNone(first_widget)
        self.assertTrue(bool(first_widget.property("selected")))
        self.assertIn(
            "/",
            str(
                self.window.composer._mention_popup.list_widget.item(1).data(
                    self.window.composer._mention_popup.DISPLAY_TEXT_ROLE
                )
                or ""
            ),
        )

    def test_mention_item_widget_children_are_not_created_as_top_level_windows(self):
        owner = QWidget()
        self.addCleanup(owner.deleteLater)
        widget = _ComposerMentionItemWidget(
            owner,
            text="theme.py",
            folder="ui",
            relative="ui/theme.py",
            is_dir=False,
        )
        self.addCleanup(widget.deleteLater)

        self.assertIs(widget.icon_label.parentWidget(), widget)
        self.assertIs(widget.title_label.parentWidget(), widget)
        self.assertIs(widget.folder_label.parentWidget(), widget)
        self.assertFalse(widget.icon_label.isWindow())
        self.assertFalse(widget.title_label.isWindow())
        self.assertFalse(widget.folder_label.isWindow())

    def test_mention_popup_includes_directories_from_indexed_files(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.show()
        self._process_events()
        self.window.composer.set_file_index_for_testing(
            ["docs/readme.md", "docs/nested/info.txt", "main.py"]
        )
        self.window.composer.setPlainText("@do")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
        self.window.composer._refresh_mention_popup()

        self.assertTrue(self.window.composer._mention_popup.isVisible())
        items = [
            str(
                self.window.composer._mention_popup.list_widget.item(index).data(
                    self.window.composer._mention_popup.DISPLAY_TEXT_ROLE
                )
                or ""
            )
            for index in range(self.window.composer._mention_popup.list_widget.count())
        ]
        self.assertIn("docs/", items)
        self.assertIn("docs/readme.md", items)

    def test_mention_popup_refresh_sees_files_created_after_startup(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.show()
        self._process_events()
        temp_root = Path.cwd() / ".tmp_tests" / f"composer-mention-{time.time_ns()}"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: temp_root.exists() and shutil.rmtree(temp_root, ignore_errors=True))

        with mock.patch("ui.widgets.composer.Path.cwd", return_value=temp_root):
            self.window.composer.setPlainText("@late")
            self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
            self.window.composer._refresh_mention_popup()
            self.assertTrue(
                self.window.composer._mention_popup is None
                or not self.window.composer._mention_popup.isVisible()
            )

            late_dir = temp_root / "late-dir"
            late_dir.mkdir()
            late_file = late_dir / "late-file.txt"
            late_file.write_text("demo", encoding="utf-8")

            self.window.composer._refresh_mention_popup()

        self.assertTrue(self.window.composer._mention_popup.isVisible())
        items = [
            str(
                self.window.composer._mention_popup.list_widget.item(index).data(
                    self.window.composer._mention_popup.DISPLAY_TEXT_ROLE
                )
                or ""
            )
            for index in range(self.window.composer._mention_popup.list_widget.count())
        ]
        self.assertIn("late-dir/", items)
        self.assertIn("late-dir/late-file.txt", items)

    def test_mention_popup_empty_query_refreshes_dirty_workspace_index(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.show()
        self._process_events()
        temp_root = Path.cwd() / ".tmp_tests" / f"composer-mention-dirty-{time.time_ns()}"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: temp_root.exists() and shutil.rmtree(temp_root, ignore_errors=True))

        with mock.patch("ui.widgets.composer.Path.cwd", return_value=temp_root):
            (temp_root / "first.py").write_text("print('first')", encoding="utf-8")
            self.window.composer._ensure_file_index(force_refresh=True)
            self.window.composer.setPlainText("@")
            self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
            self.window.composer._refresh_mention_popup()

            (temp_root / "late.py").write_text("print('late')", encoding="utf-8")
            self.window.composer._on_file_index_directory_changed(str(temp_root))
            self.window.composer._refresh_mention_popup()

        self.assertTrue(self.window.composer._mention_popup.isVisible())
        items = [
            str(
                self.window.composer._mention_popup.list_widget.item(index).data(
                    self.window.composer._mention_popup.DISPLAY_TEXT_ROLE
                )
                or ""
            )
            for index in range(self.window.composer._mention_popup.list_widget.count())
        ]
        self.assertIn("first.py", items)
        self.assertIn("late.py", items)

    def test_mention_popup_closes_on_escape_no_matches_and_cursor_change(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.show()
        self._process_events()
        self.window.composer.set_file_index_for_testing(["main.py"])
        self.window.composer.setPlainText("@ma")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
        self.window.composer._refresh_mention_popup()
        self.assertTrue(self.window.composer._mention_popup.isVisible())

        self._press_composer_key(Qt.Key_Escape)
        self.assertFalse(self.window.composer._mention_popup.isVisible())

        self.window.composer.setPlainText("@zz")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
        self.window.composer._refresh_mention_popup()
        self.assertFalse(self.window.composer._mention_popup.isVisible())

        self.window.composer.setPlainText("@ma")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
        self.window.composer._refresh_mention_popup()
        self.assertTrue(self.window.composer._mention_popup.isVisible())
        cursor = self.window.composer.textCursor()
        cursor.setPosition(0)
        self.window.composer.setTextCursor(cursor)
        self.window.composer._refresh_mention_popup()
        self.assertFalse(self.window.composer._mention_popup.isVisible())

    def test_realtime_elapsed_uses_minutes_above_sixty_seconds(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Долгий запрос"}))
        self.window.is_busy = True
        self.window._run_start_time = 1000.0

        with mock.patch("ui.main_window_state.time.time", return_value=1061.2):
            self.window._update_realtime_elapsed()

        self.assertEqual(self.window.current_turn.status_widget.meta_label.text(), "1m 1s")

    def test_stop_during_retries_does_not_leave_status_in_previous_turn(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_busy_changed(True)
        self.window._handle_event(StreamEvent("run_started", {"text": "продолжай"}))
        self.window._handle_event(
            StreamEvent(
                "status_changed",
                {"label": "Retrying provider request... 3/3", "node": "agent", "elapsed": 123.0},
            )
        )
        stopped_turn = self.window.current_turn
        self.assertIsNotNone(stopped_turn.status_widget)
        run_start_time = self.window._run_start_time or 0.0

        # Stop pressed: the worker cancels the run and reports itself idle.
        self.window._handle_busy_changed(False)
        self._process_events()
        self.assertIsNone(stopped_turn.status_widget)

        # The next run flips busy before run_started arrives, so the elapsed ticker
        # must not repaint the retry status into the already finished turn.
        self.window._handle_busy_changed(True)
        with mock.patch("ui.main_window_state.time.time", return_value=run_start_time + 200.0):
            self.window._update_realtime_elapsed()
        self.assertIsNone(stopped_turn.status_widget)

        self.window._handle_event(StreamEvent("run_started", {"text": "продолжай"}))
        self._process_events()

        self.assertIsNot(self.window.current_turn, stopped_turn)
        self.assertIsNone(stopped_turn.status_widget)
        self.assertIsNotNone(self.window.current_turn.status_widget)
        self.assertEqual(self.window.current_turn.status_widget.label.text(), "Working...")

    def test_new_turn_clears_status_row_left_in_previous_turn(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "первый запрос"}))
        previous_turn = self.window.current_turn
        previous_turn.set_status("Retrying provider request... 3/3", meta="2m 3s")
        self.assertTrue(previous_turn.has_status())

        self.window._handle_event(StreamEvent("run_started", {"text": "второй запрос"}))
        self._process_events()

        self.assertIsNot(self.window.current_turn, previous_turn)
        self.assertFalse(previous_turn.has_status())
        self.assertTrue(self.window.current_turn.has_status())

    def test_run_started_shows_inline_status_before_output(self):
        self.window._handle_initialized(self._snapshot_payload())

        self.window._handle_event(StreamEvent("run_started", {"text": "Сводка"}))
        self.window._handle_event(StreamEvent("status_changed", {"label": "Self-correcting", "node": "recovery"}))

        self.assertIsNotNone(self.window.current_turn.status_widget)
        self.assertEqual(self.window.current_turn.status_widget.label.text(), "Self-correcting")

    def test_run_started_requeues_autofollow_after_inserting_working_status(self):
        self.window._handle_initialized(self._snapshot_payload())
        with mock.patch.object(
            self.window.transcript,
            "notify_content_changed",
            wraps=self.window.transcript.notify_content_changed,
        ) as notify_mock:
            self.window._handle_event(StreamEvent("run_started", {"text": "Ещё запрос"}))

        self.assertIsNotNone(self.window.current_turn.status_widget)
        self.assertEqual(self.window.current_turn.status_widget.label.text(), "Working...")
        self.assertGreaterEqual(notify_mock.call_count, 2)
        self.assertEqual(notify_mock.call_args_list[-1].kwargs.get("force"), True)

    def test_status_is_rendered_below_existing_output_when_agent_keeps_thinking(self):
        self.window._handle_initialized(self._snapshot_payload())

        self.window._handle_event(StreamEvent("run_started", {"text": "Проверь"}))
        self.window._handle_event(
            StreamEvent(
                "assistant_delta",
                {
                    "text": "partial",
                    "full_text": "Ответ\n\n```python\nprint('hi')\n```",
                    "has_thought": False,
                },
            )
        )
        self.window._handle_event(StreamEvent("status_changed", {"label": "Self-correcting", "node": "recovery"}))

        self.assertIsNotNone(self.window.current_turn.status_widget)
        self.assertEqual(self.window.current_turn.status_widget.label.text(), "Self-correcting")

    def test_streaming_response_keeps_inline_status_visible_at_bottom_until_finish(self):
        self.window._handle_initialized(self._snapshot_payload())

        self.window._handle_event(StreamEvent("run_started", {"text": "Ответь подробно"}))
        self.window._handle_event(
            StreamEvent(
                "assistant_delta",
                {
                    "text": "Первый кусок",
                    "full_text": "Первый кусок ответа",
                    "has_thought": False,
                },
            )
        )

        self.assertIsNotNone(self.window.current_turn.status_widget)
        self.assertEqual(self.window.current_turn.block_kinds(), ["user", "assistant"])
        self.assertIs(
            self.window.current_turn.layout().itemAt(self.window.current_turn.layout().count() - 1).widget(),
            self.window.current_turn.status_widget,
        )

    def test_auto_summary_status_transitions_in_chat_without_duplicate_notice(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Большой контекст"}))

        self.window._handle_event(
            StreamEvent("status_changed", {"label": "Compressing context", "node": "summarize"})
        )
        self.assertEqual(self.window.current_turn.block_kinds(), ["user"])
        self.assertIsNotNone(self.window.current_turn.status_widget)
        self.assertEqual(self.window.current_turn.status_widget.label.text(), "Compressing context")
        self.assertEqual(self.window.current_turn.status_widget.property("phase"), "system")
        self.window._update_realtime_elapsed()
        self.assertEqual(self.window.current_turn.status_widget.label.text(), "Compressing context")
        self.assertNotEqual(self.window.current_turn.status_widget.meta_label.text(), "")
        self.assertEqual(self.window.statusBar().currentMessage(), "")

        self.window._handle_event(
            StreamEvent(
                "summary_notice",
                {
                    "kind": "auto_summary",
                    "count": 3,
                    "message": "Context compressed automatically (3 message(s) summarized).",
                },
            )
        )
        self._process_events()

        status_widget = self.window.current_turn.status_widget
        self.assertIsNotNone(status_widget)
        self.assertEqual(self.window.current_turn.block_kinds(), ["user"])
        self.assertEqual(status_widget.label.text(), "Context compressed automatically (3 message(s) summarized).")
        self.assertEqual(status_widget.meta_label.text(), "")
        self.assertEqual(status_widget.property("phase"), "success")
        self.assertNotIn("compressing context", status_widget.label.text().lower())
        self.assertEqual(len(self.window.current_turn.findChildren(NoticeWidget)), 0)
        self.assertEqual(self.window.statusBar().currentMessage(), "")

        self.window._update_realtime_elapsed()
        self.assertEqual(
            self.window.current_turn.status_widget.label.text(),
            "Context compressed automatically (3 message(s) summarized).",
        )
        self.assertEqual(self.window.current_turn.status_widget.meta_label.text(), "")

        self.window._handle_event(
            StreamEvent(
                "assistant_delta",
                {"text": "готово", "full_text": "Готово", "has_thought": False},
            )
        )
        self._process_events()

        self.assertEqual(self.window.current_turn.block_kinds(), ["user", "assistant"])
        self.assertIs(self.window.current_turn.status_widget, status_widget)
        self.assertEqual(len(self.window.current_turn.findChildren(NoticeWidget)), 0)

    def test_stream_events_render_transcript_and_compact_tool_sections(self):
        self.window._handle_initialized(self._snapshot_payload())

        self.window._handle_event(StreamEvent("run_started", {"text": "Покажи diff"}))
        self.window._handle_event(
            StreamEvent(
                "assistant_delta",
                {
                    "text": "partial",
                    "full_text": "Ответ\n\n```python\nprint('hi')\n```",
                    "has_thought": False,
                },
            )
        )
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-1",
                    "name": "edit_file",
                    "args": {"path": "demo.txt"},
                    "display": "edit_file(demo.txt)",
                },
            )
        )
        self.window._handle_event(
            StreamEvent(
                "assistant_delta",
                {
                    "text": "tail",
                    "full_text": "Ответ\n\n```python\nprint('hi')\n```\n\nГотово",
                    "has_thought": False,
                },
            )
        )
        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-1",
                    "name": "edit_file",
                    "content": "Success\n```diff\n-foo\n+bar\n```",
                    "summary": "File edited successfully",
                    "is_error": False,
                    "duration": 1.4,
                    "diff": "-foo\n+bar",
                },
            )
        )

        self.assertIsNotNone(self.window.current_turn)
        self.assertEqual(self.window.current_turn.block_kinds(), ["user", "assistant", "tool_group", "assistant"])
        self.assertEqual(len(self.window.current_turn.assistant_segments), 2)
        self.assertIn("Ответ", self.window.current_turn.assistant_segments[0].markdown())
        self.assertIn("Готово", self.window.current_turn.assistant_segments[1].markdown())
        self.assertEqual(self.window.current_turn.assistant_segments[0].frameShape(), QFrame.NoFrame)
        transcript_text_labels = [
            label.text()
            for label in self.window.current_turn.findChildren(QLabel)
            if label.text() in {"Agent", "You"}
        ]
        self.assertEqual(transcript_text_labels, [])
        self.assertIsInstance(self.window.current_turn.tool_group, ToolGroupWidget)
        self.assertEqual(self.window.current_turn.tool_group.header_btn.text(), "Edited 1 file")
        self.assertTrue(self.window.current_turn.tool_group.container.isHidden())
        tool_card = self.window.current_turn.tool_cards["call-1"]
        self.assertEqual(tool_card.frameShape(), QFrame.NoFrame)
        self.assertEqual(tool_card.tool_button.text(), "")
        self.assertEqual(tool_card.action_label.full_text(), "Edited demo.txt +1 -1")
        action_markup = tool_card.action_label.text()
        self.assertIn('color:#7CC7FF', action_markup)
        self.assertIn(f'color:{SUCCESS_GREEN}', action_markup)
        self.assertIn(f'color:{ERROR_RED}', action_markup)
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.args_container.isHidden())
        self.assertIsNone(tool_card.output_section)
        self.assertIsNotNone(tool_card.diff_section)
        self.assertEqual(tool_card.diff_section.toggle_button.text(), "Edited file")
        self.assertTrue(tool_card.diff_section.toggle_button.isHidden())
        self.assertFalse(tool_card.diff_section.toggle_button.isChecked())
        self.assertTrue(tool_card.diff_section.content_container.isHidden())
        self.assertEqual(tool_card.diff_section.content.path_label.text(), "demo.txt")
        self.assertEqual(tool_card.diff_section.content.added_label.text(), "+1")
        self.assertEqual(tool_card.diff_section.content.removed_label.text(), "-1")
        rendered_diff = tool_card.diff_section.content.editor.toPlainText()
        self.assertIn(" + bar", rendered_diff)
        self.assertNotIn("--- ", rendered_diff)
        self.assertNotIn("+++ ", rendered_diff)
        self.assertNotIn("@@ ", rendered_diff)
        full_width_selections = tool_card.diff_section.content.editor.extraSelections()
        self.assertEqual(len(full_width_selections), 2)
        self.assertTrue(all(bool(sel.format.property(QTextFormat.FullWidthSelection)) for sel in full_width_selections))
        selection_colors = {sel.format.background().color().name().lower() for sel in full_width_selections}
        self.assertEqual(selection_colors, {"#1e3425", "#472b2b"})
        self.window.current_turn.tool_group.header_btn.click()
        self._process_events()
        self.assertFalse(self.window.current_turn.tool_group.container.isHidden())
        tool_card.tool_button.click()
        self._process_events()
        self.assertTrue(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.args_container.isHidden())
        self.assertFalse(tool_card.diff_section.content_container.isHidden())

    def test_preview_tool_card_stays_hidden_until_resolved_refresh_arrives(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Сохрани файл"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-preview",
                    "name": "write_file",
                    "args": {},
                    "display": "Preparing file write",
                    "subtitle": "Waiting for arguments…",
                    "raw_display": "write_file",
                    "args_state": "pending",
                    "display_state": "preview",
                    "phase": "preparing",
                    "source_kind": "tool",
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-preview"]
        self.assertTrue(tool_card.isHidden())

        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-preview",
                    "name": "write_file",
                    "args": {"path": "notes.md"},
                    "display": "Writing file",
                    "subtitle": "notes.md",
                    "raw_display": "write_file(notes.md)",
                    "args_state": "complete",
                    "display_state": "resolved",
                    "phase": "running",
                    "source_kind": "tool",
                    "refresh": True,
                },
            )
        )
        self._process_events()

        self.assertFalse(tool_card.isHidden())
        self.assertEqual(tool_card.tool_button.text(), "")
        self.assertEqual(tool_card.action_label.full_text(), "Writing notes.md")
        self.assertTrue(tool_card.subtitle_label.isHidden())
        self.assertTrue(tool_card.phase_badge.isHidden())
        self.assertNotIn("write_file()", tool_card.action_label.full_text())

    def test_write_file_finished_label_says_file_created(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Создай файл"}))
        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-write",
                    "name": "write_file",
                    "args": {"path": "notes.md"},
                    "content": "Success",
                    "summary": "File written successfully",
                    "diff": "@@ -0,0 +1,1 @@\n+hello",
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-write"]
        self.assertEqual(tool_card.action_label.full_text(), "Wrote notes.md +1 -0")
        self.assertNotIn("Editing", tool_card.action_label.full_text())
        self.assertTrue(tool_card.tool_button.isChecked())
        self.assertFalse(tool_card.diff_section.content_container.isHidden())

    def test_cli_exec_renders_live_terminal_panel_and_streams_output(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Запусти команду"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-cli",
                    "name": "cli_exec",
                    "args": {"command": "echo hello"},
                    "display": 'cli_exec("echo hello")',
                },
            )
        )
        self.window._handle_event(
            StreamEvent("cli_output", {"tool_id": "call-cli", "data": "hello\n", "stream": "stdout"})
        )
        self.window._handle_event(
            StreamEvent("cli_output", {"tool_id": "call-cli", "data": "world\n", "stream": "stdout"})
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-cli"]
        self.assertFalse(tool_card.header_container.isHidden())
        self.assertTrue(tool_card.args_container.isHidden())
        self.assertTrue(tool_card.tool_button.isEnabled())
        self.assertTrue(tool_card.tool_button.isCheckable())
        self.assertTrue(tool_card.tool_button.isChecked())
        self.assertEqual(tool_card.tool_button.text(), "")
        self.assertEqual(tool_card.action_label.full_text(), "Running echo hello")
        self.assertIsNotNone(tool_card.cli_exec_widget)
        self.assertFalse(tool_card.cli_exec_widget.isHidden())
        self.assertEqual(tool_card.cli_exec_widget.command_label.text(), "$ echo hello")
        output_text = tool_card.cli_exec_widget.output_view.toPlainText()
        self.assertIn("hello", output_text)
        self.assertIn("world", output_text)
        self.assertGreaterEqual(tool_card.cli_exec_widget.output_view.height(), 96)

        tool_card.tool_button.click()
        self._process_events()
        self.assertTrue(tool_card.cli_exec_widget.isHidden())

    def test_cli_exec_header_command_is_single_line_for_multiline_command(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Запусти python"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-heredoc",
                    "name": "cli_exec",
                    "args": {"command": "python - <<'PY'\nimport sys\nprint(sys.version)\nPY"},
                    "display": 'cli_exec("python - <<\'PY\' import sys print(sys.version) PY")',
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-heredoc"]
        header_text = tool_card.cli_exec_widget.command_label.text()
        self.assertTrue(header_text.startswith("$ "))
        self.assertNotIn("\n", header_text)
        self.assertIn("python - <<'PY'", header_text)

    def test_cli_exec_header_uses_eliding_policy_and_does_not_force_wide_layout(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Проверь ширину"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-wide",
                    "name": "cli_exec",
                    "args": {"command": "python -c \"" + ("x" * 800) + "\""},
                    "display": 'cli_exec("python -c ...")',
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-wide"]
        action_label = tool_card.action_label
        self.assertEqual(action_label.sizePolicy().horizontalPolicy(), QSizePolicy.Ignored)
        action_label.setFixedWidth(180)
        self._process_events()
        self.assertNotIn("\n", action_label.text())
        self.assertTrue(bool(action_label.toolTip()))
        self.assertGreater(len(action_label.toolTip()), len(action_label.text()))

        label = tool_card.cli_exec_widget.command_label
        self.assertEqual(label.sizePolicy().horizontalPolicy(), QSizePolicy.Ignored)
        label.setFixedWidth(180)
        self._process_events()
        self.assertNotIn("\n", label.text())
        self.assertTrue(bool(label.toolTip()))
        self.assertGreater(len(label.toolTip()), len(label.text()))

    def test_cli_exec_card_created_by_output_is_refreshed_when_tool_started_arrives_later(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Проверь race"}))
        self.window._handle_event(
            StreamEvent("cli_output", {"tool_id": "call-race", "data": "line 1\n", "stream": "stdout"})
        )
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-race",
                    "name": "cli_exec",
                    "args": {"command": "echo race"},
                    "display": 'cli_exec("echo race")',
                    "refresh": True,
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-race"]
        self.assertEqual(tool_card.action_label.full_text(), "Running echo race")
        self.assertEqual(tool_card.cli_exec_widget.command_label.text(), "$ echo race")

    def test_cli_exec_output_autofollow_respects_manual_scroll(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Tail logs"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-tail",
                    "name": "cli_exec",
                    "args": {"command": "tail -f app.log"},
                    "display": 'cli_exec("tail -f app.log")',
                },
            )
        )
        for idx in range(90):
            self.window._handle_event(
                StreamEvent(
                    "cli_output",
                    {"tool_id": "call-tail", "data": f"line {idx}\n", "stream": "stdout"},
                )
            )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-tail"]
        output_view = tool_card.cli_exec_widget.output_view
        scrollbar = output_view.verticalScrollBar()
        self.assertEqual(scrollbar.value(), scrollbar.maximum())

        scrollbar.setValue(max(0, scrollbar.maximum() - 20))
        self._process_events()
        previous_value = scrollbar.value()
        self.window._handle_event(
            StreamEvent("cli_output", {"tool_id": "call-tail", "data": "manual-check\n", "stream": "stdout"})
        )
        self._process_events()
        self.assertEqual(scrollbar.value(), previous_value)

        scrollbar.setValue(scrollbar.maximum())
        self._process_events()
        self.assertTrue(self.window.transcript.auto_follow_enabled)
        self.window._handle_event(
            StreamEvent("cli_output", {"tool_id": "call-tail", "data": "follow-bottom\n", "stream": "stdout"})
        )
        self._process_events()
        self.assertEqual(scrollbar.value(), scrollbar.maximum())

    def test_cli_exec_collapses_after_finish_by_default(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Запусти команду"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-cli-finish",
                    "name": "cli_exec",
                    "args": {"command": "echo done"},
                    "display": 'cli_exec("echo done")',
                },
            )
        )
        self.window._handle_event(
            StreamEvent("tool_finished", {"tool_id": "call-cli-finish", "name": "cli_exec", "content": "done\n"})
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-cli-finish"]
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertIsNotNone(tool_card.cli_exec_widget)
        self.assertTrue(tool_card.cli_exec_widget.isHidden())
        self.assertIn("done", tool_card.cli_exec_widget.output_view.toPlainText())
        self._wait_for_gui(950)
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.cli_exec_widget.isHidden())

    def test_cli_exec_finish_collapses_even_if_user_kept_card_open(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Запусти команду"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-cli-finish-open",
                    "name": "cli_exec",
                    "args": {"command": "echo done"},
                    "display": 'cli_exec("echo done")',
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-cli-finish-open"]
        self.assertTrue(tool_card.tool_button.isChecked())
        self.assertIsNotNone(tool_card.cli_exec_widget)
        self.assertFalse(tool_card.cli_exec_widget.isHidden())

        # User can still toggle while running; finish keeps the compact row collapsed.
        tool_card.tool_button.click()
        self._process_events()
        self.assertTrue(tool_card.cli_exec_widget.isHidden())

        self.window._handle_event(
            StreamEvent("tool_finished", {"tool_id": "call-cli-finish-open", "name": "cli_exec", "content": "done\n"})
        )
        self._process_events()

        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.cli_exec_widget.isHidden())
        self._wait_for_gui(950)
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.cli_exec_widget.isHidden())

    def test_cli_exec_long_running_finishes_without_extra_delay(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Запусти команду"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-cli-long",
                    "name": "cli_exec",
                    "args": {"command": "echo done"},
                    "display": 'cli_exec("echo done")',
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-cli-long"]

        self.window._handle_event(
            StreamEvent("tool_finished", {"tool_id": "call-cli-long", "name": "cli_exec", "content": "done\n"})
        )
        self._process_events()

        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.cli_exec_widget.isHidden())

    def test_cli_exec_restored_from_transcript_is_collapsed(self):
        payload = self._snapshot_payload()
        payload["transcript"] = {
            "summary_notice": "",
            "turns": [
                {
                    "user_text": "run cli",
                    "blocks": [
                        {
                            "type": "tool",
                            "payload": {
                                "tool_id": "call-cli-restored",
                                "name": "cli_exec",
                                "args": {"command": "python --version"},
                                "display": 'cli_exec("python --version")',
                                "content": "Python 3.12.9\n",
                                "duration": 0.1,
                            },
                        }
                    ],
                }
            ],
        }
        self.window._handle_initialized(payload)
        self._process_events()

        restored_turn = self.window.transcript.layout.itemAt(0).widget()
        self.assertEqual(restored_turn.block_kinds(), ["user", "tool_group"])
        self.assertIsInstance(restored_turn.tool_group, ToolGroupWidget)
        self.assertTrue(restored_turn.tool_group.container.isHidden())
        self.assertEqual(restored_turn.tool_group.header_btn.text(), "Ran 1 command")
        tool_card = restored_turn.tool_cards["call-cli-restored"]
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertIsNotNone(tool_card.cli_exec_widget)
        self.assertTrue(tool_card.cli_exec_widget.isHidden())
        self.assertEqual(tool_card.action_label.full_text(), "Ran python --version")
        self.assertTrue(tool_card.phase_badge.isHidden())

    def test_restored_tool_widgets_are_parented_under_tool_group(self):
        payload = self._snapshot_payload()
        payload["transcript"] = {
            "summary_notice": "",
            "turns": [
                {
                    "user_text": "edit",
                    "blocks": [
                        {
                            "type": "tool",
                            "payload": {
                                "tool_id": "call-edit-restored",
                                "name": "edit_file",
                                "args": {"path": "index.html"},
                                "content": "Success",
                                "summary": "File edited successfully",
                                "duration": 0.2,
                                "diff": "@@ -1,1 +1,1 @@\n-old\n+new",
                            },
                        }
                    ],
                }
            ],
        }

        self.window._handle_initialized(payload)
        self._process_events()

        restored_turn = self.window.transcript.layout.itemAt(0).widget()
        tool_card = restored_turn.tool_cards["call-edit-restored"]
        self.assertIs(tool_card.parentWidget(), restored_turn.tool_group.container)
        self.assertFalse(tool_card.isWindow())
        self.assertIsNotNone(tool_card.diff_section)
        self.assertIs(tool_card.diff_section.parentWidget(), tool_card)
        self.assertFalse(tool_card.diff_section.isWindow())
        self.assertIsInstance(tool_card.diff_section.content, DiffBlockWidget)
        self.assertFalse(tool_card.diff_section.content.isWindow())
        self.assertIs(tool_card.diff_section.content.editor.parentWidget(), tool_card.diff_section.content)

    def test_finished_tool_restored_from_transcript_keeps_success_badge(self):
        payload = self._snapshot_payload()
        payload["transcript"] = {
            "summary_notice": "",
            "turns": [
                {
                    "user_text": "прочитай файл",
                    "blocks": [
                        {
                            "type": "tool",
                            "payload": {
                                "tool_id": "call-restored-finished",
                                "name": "read_file",
                                "args": {"path": "index.html"},
                                "display": "Reading file",
                                "subtitle": "index.html",
                                "raw_display": "read_file(index.html)",
                                "args_state": "complete",
                                "display_state": "finished",
                                "phase": "finished",
                                "source_kind": "tool",
                                "summary": "Read 78 lines (3574 chars)",
                                "content": "Read 78 lines.",
                                "is_error": False,
                            },
                        }
                    ],
                }
            ],
        }

        self.window._handle_initialized(payload)
        self._process_events()

        restored_turn = self.window.transcript.layout.itemAt(0).widget()
        tool_card = restored_turn.tool_cards["call-restored-finished"]
        self.assertEqual(tool_card.tool_button.text(), "")
        self.assertEqual(tool_card.action_label.full_text(), "Read index.html")
        self.assertEqual(tool_card.action_label.toolTip(), "read_file(index.html)")
        self.assertTrue(tool_card.subtitle_label.isHidden())
        self.assertTrue(tool_card.phase_badge.isHidden())

    def test_restored_turn_keeps_separate_tool_groups_between_assistant_blocks(self):
        payload = self._snapshot_payload()
        payload["transcript"] = {
            "summary_notice": "",
            "turns": [
                {
                    "user_text": "multi-step",
                    "blocks": [
                        {
                            "type": "tool",
                            "payload": {
                                "tool_id": "call-read",
                                "name": "read_file",
                                "args": {"path": "a.txt"},
                                "display": "Reading a.txt",
                                "content": "ok",
                            },
                        },
                        {"type": "assistant", "markdown": "Need another step."},
                        {
                            "type": "tool",
                            "payload": {
                                "tool_id": "call-edit",
                                "name": "edit_file",
                                "args": {"path": "a.txt"},
                                "display": "Editing a.txt",
                                "content": "ok",
                            },
                        },
                    ],
                }
            ],
        }

        self.window._handle_initialized(payload)
        self._process_events()

        restored_turn = self.window.transcript.layout.itemAt(0).widget()
        self.assertEqual(restored_turn.block_kinds(), ["user", "tool_group", "assistant", "tool_group"])
        groups = restored_turn.findChildren(ToolGroupWidget)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(group.container.isHidden() for group in groups))
        self.assertEqual([group.header_btn.text() for group in groups], ["Read 1 file", "Edited 1 file"])

    def test_tool_error_output_is_collapsed_by_default(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Почини"}))
        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-err",
                    "name": "edit_file",
                    "content": "error[access_denied]: blocked",
                    "summary": "Skipped",
                    "is_error": True,
                    "duration": 0.4,
                    "diff": "",
                },
            )
        )

        tool_card = self.window.current_turn.tool_cards["call-err"]
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.args_container.isHidden())
        tool_card.tool_button.click()
        self._process_events()
        self.assertIn("error[access_denied]", tool_card.args_view.toPlainText().lower())
        self.assertEqual(self.window.current_turn.tool_group.header_btn.text(), "Editing failed ·")
        self.assertFalse(self.window.current_turn.tool_group.error_icon_label.isHidden())
        self.assertEqual(self.window.current_turn.tool_group.error_count_label.text(), "1")

    def test_tool_validation_error_drops_waiting_for_arguments_placeholder(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Прочитай"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-missing-path",
                    "name": "read_file",
                    "args": {},
                    "subtitle": "Waiting for arguments…",
                    "args_state": "pending",
                    "display_state": "preview",
                    "phase": "preparing",
                },
            )
        )
        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-missing-path",
                    "name": "read_file",
                    "content": "ERROR[VALIDATION]: Missing required field(s): path.",
                    "summary": "Missing required field(s): path.",
                    "is_error": True,
                    "duration": 0.1,
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-missing-path"]
        self.assertEqual(tool_card.action_label.full_text(), "Reading failed")
        self.assertNotIn("Waiting for arguments", tool_card.action_label.full_text())
        self.assertNotIn("Waiting for arguments", tool_card.action_label.toolTip())

    def test_restored_error_tool_group_does_not_crash_and_shows_error_count(self):
        payload = self._snapshot_payload()
        payload["transcript"] = {
            "summary_notice": "",
            "turns": [
                {
                    "user_text": "почини файл",
                    "blocks": [
                        {
                            "type": "tool",
                            "payload": {
                                "tool_id": "call-restored-error",
                                "name": "edit_file",
                                "args": {"path": "broken.txt"},
                                "display": "Editing file",
                                "content": "error[access_denied]: blocked",
                                "summary": "Skipped",
                                "is_error": True,
                                "phase": "finished",
                                "display_state": "finished",
                            },
                        }
                    ],
                }
            ],
        }

        self.window._handle_initialized(payload)
        self._process_events()

        restored_turn = self.window.transcript.layout.itemAt(0).widget()
        self.assertEqual(restored_turn.tool_group.header_btn.text(), "Editing failed ·")
        self.assertFalse(restored_turn.tool_group.error_icon_label.isHidden())
        self.assertEqual(restored_turn.tool_group.error_count_label.text(), "1")
        self.assertTrue(restored_turn.tool_group.container.isHidden())

    def test_finish_overrides_stale_running_phase_and_collapses_non_cli_exec(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Прочитай"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-read",
                    "name": "read_file",
                    "args": {"path": "main.py"},
                    "display": "read_file(main.py)",
                    "phase": "running",
                },
            )
        )
        self._process_events()
        tool_card = self.window.current_turn.tool_cards["call-read"]
        self.assertEqual(tool_card.action_label.full_text(), "Reading main.py")
        self.assertTrue(tool_card.phase_badge.isHidden())

        tool_card.tool_button.click()
        self._process_events()
        self.assertTrue(tool_card.tool_button.isChecked())
        self.assertFalse(tool_card.args_container.isHidden())

        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-read",
                    "name": "read_file",
                    "content": "ok",
                    "phase": "running",
                },
            )
        )
        self._process_events()
        self.assertEqual(tool_card.action_label.full_text(), "Read main.py")
        self.assertTrue(tool_card.phase_badge.isHidden())
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.args_container.isHidden())
        self.assertFalse(self.window.current_turn.tool_group.container.isHidden())
        self.assertEqual(self.window.current_turn.tool_group.header_btn.text(), "Read 1 file")

    def test_finished_tool_group_stays_open_until_assistant_text_even_after_manual_expand(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Исправь файл"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-edit-open",
                    "name": "edit_file",
                    "args": {"path": "demo.txt"},
                    "display": "Editing file",
                    "phase": "running",
                },
            )
        )
        self._process_events()

        group = self.window.current_turn.tool_group
        tool_card = self.window.current_turn.tool_cards["call-edit-open"]
        self.assertIsNotNone(group)
        self.assertFalse(group.container.isHidden())

        tool_card.tool_button.click()
        self._process_events()
        self.assertFalse(tool_card.args_container.isHidden())

        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-edit-open",
                    "name": "edit_file",
                    "args": {"path": "demo.txt"},
                    "content": "Success",
                    "summary": "File edited successfully",
                    "phase": "running",
                },
            )
        )
        self._process_events()

        self.assertFalse(group.container.isHidden())
        self.assertEqual(group.header_btn.text(), "Edited 1 file")

        self.window._handle_event(
            StreamEvent(
                "assistant_delta",
                {
                    "text": "Комментирую результат",
                    "full_text": "Комментирую результат",
                    "has_thought": False,
                },
            )
        )
        self._process_events()

        self.assertTrue(group.container.isHidden())
        self.assertEqual(self.window.current_turn.block_kinds(), ["user", "tool_group", "assistant"])
        self.assertEqual(group.header_btn.text(), "Edited 1 file")

    def test_approval_notice_after_tool_finish_keeps_group_completed(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Исправь файл"}))
        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-approved-edit",
                    "name": "edit_file",
                    "args": {"path": "demo.txt"},
                    "content": "Success",
                    "summary": "File edited successfully",
                },
            )
        )
        self._process_events()

        group = self.window.current_turn.tool_group
        self.assertFalse(group.container.isHidden())
        self.assertEqual(group.header_btn.text(), "Edited 1 file")

        self.window._handle_event(
            StreamEvent("approval_resolved", {"approved": True, "always": False, "auto": False})
        )
        self._process_events()

        self.assertEqual(self.window.current_turn.block_kinds(), ["user", "tool_group"])
        self.assertIn("approved", self.window.statusBar().currentMessage().lower())
        self.assertFalse(group.container.isHidden())
        self.assertEqual(group.header_btn.text(), "Edited 1 file")

    def test_run_finished_keeps_completed_tool_group_open_without_assistant_text(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Используй инструмент"}))
        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-finish-open",
                    "name": "read_file",
                    "args": {"path": "main.py"},
                    "content": "ok",
                },
            )
        )
        self.window._handle_event(StreamEvent("run_finished", {"stats": "1.0s   In: 10   Out: 5"}))
        self._process_events()

        group = self.window.current_turn.tool_group
        self.assertIsNotNone(group)
        self.assertFalse(group.container.isHidden())
        self.assertEqual(self.window.current_turn.block_kinds(), ["user", "tool_group", "stats"])
        self.assertEqual(group.header_btn.text(), "Read 1 file")

    def test_assistant_comment_after_tool_group_does_not_repeat_previous_prefix(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Сделай анализ"}))
        self.window._handle_event(
            StreamEvent(
                "assistant_delta",
                {
                    "text": "Изучаю структуру проекта для проведения анализа.",
                    "full_text": "Изучаю структуру проекта для проведения анализа.",
                    "has_thought": False,
                },
            )
        )
        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-analysis",
                    "name": "read_file",
                    "args": {"path": "README.md"},
                    "content": "ok",
                },
            )
        )
        self.window._handle_event(
            StreamEvent(
                "assistant_delta",
                {
                    "text": "Проект представляет собой автономного AI-агента.",
                    "full_text": (
                        "Изучаю структуру проекта для проведения анализа.\n"
                        "Проект представляет собой автономного AI-агента."
                    ),
                    "has_thought": False,
                },
            )
        )
        self._process_events()

        self.assertEqual(self.window.current_turn.block_kinds(), ["user", "assistant", "tool_group", "assistant"])
        self.assertEqual(len(self.window.current_turn.assistant_segments), 2)
        self.assertEqual(
            self.window.current_turn.assistant_segments[0].markdown(),
            "Изучаю структуру проекта для проведения анализа.",
        )
        self.assertEqual(
            self.window.current_turn.assistant_segments[1].markdown(),
            "Проект представляет собой автономного AI-агента.",
        )

    def test_finish_overrides_stale_running_phase_for_cli_exec(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Запусти"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-cli-run",
                    "name": "cli_exec",
                    "args": {"command": "echo hi"},
                    "display": 'cli_exec("echo hi")',
                    "phase": "running",
                },
            )
        )
        self._process_events()
        tool_card = self.window.current_turn.tool_cards["call-cli-run"]
        self.assertEqual(tool_card.action_label.full_text(), "Running echo hi")
        self.assertTrue(tool_card.phase_badge.isHidden())
        self.assertTrue(tool_card.tool_button.isChecked())
        self.assertFalse(tool_card.cli_exec_widget.isHidden())

        tool_card.tool_button.click()
        self._process_events()
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.cli_exec_widget.isHidden())

        tool_card.tool_button.click()
        self._process_events()
        self.assertTrue(tool_card.tool_button.isChecked())
        self.assertFalse(tool_card.cli_exec_widget.isHidden())

        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-cli-run",
                    "name": "cli_exec",
                    "content": "hi\n",
                    "phase": "running",
                },
            )
        )
        self._process_events()
        self.assertEqual(tool_card.action_label.full_text(), "Ran echo hi")
        self.assertTrue(tool_card.phase_badge.isHidden())
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.cli_exec_widget.isHidden())

    def test_cli_exec_error_meta_is_marked_for_error_styling(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Почини"}))
        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-cli-err",
                    "name": "cli_exec",
                    "args": {"command": "python bad.py"},
                    "content": "boom",
                    "is_error": True,
                    "duration": 0.4,
                },
            )
        )

        tool_card = self.window.current_turn.tool_cards["call-cli-err"]
        self.assertIsNotNone(tool_card.cli_exec_widget)
        self.assertEqual(tool_card.cli_exec_widget.meta_label.property("severity"), "error")
        self.assertTrue(tool_card.cli_exec_widget.meta_label.text().lower().startswith("error"))

    def test_hidden_internal_notice_event_is_rendered_in_status_bar(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Проверь остановку"}))
        self.window._handle_event(
            StreamEvent(
                "summary_notice",
                {
                    "kind": "agent_internal_notice",
                    "message": "Автоматическое продолжение остановлено. Нужен новый запрос.",
                    "level": "warning",
                },
            )
        )

        self.assertEqual(self.window.current_turn.block_kinds(), ["user"])
        self.assertIn("продолжение", self.window.statusBar().currentMessage().lower())

    def test_tool_args_missing_diagnostic_is_not_shown_in_transcript(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Проверь list_directory"}))
        self.window._handle_event(
            StreamEvent(
                "tool_args_missing",
                {
                    "tool_id": "call-dir",
                    "name": "list_directory",
                    "message": "No canonical tool args were available when tool result arrived.",
                },
            )
        )

        self.assertEqual(self.window.current_turn.block_kinds(), ["user"])

    def test_cli_exec_display_strips_ansi_sequences(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Проверь ansi"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-cli-ansi",
                    "name": "cli_exec",
                    "args": {"command": "bad-command"},
                    "display": 'cli_exec("bad-command")',
                },
            )
        )
        self.window._handle_event(
            StreamEvent(
                "cli_output",
                {
                    "tool_id": "call-cli-ansi",
                    "stream": "stderr",
                    "data": "\u001b[31mboom\u001b[0m\n",
                },
            )
        )
        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-cli-ansi",
                    "name": "cli_exec",
                    "args": {"command": "bad-command"},
                    "content": "\u001b[31mboom\u001b[0m\n",
                    "is_error": True,
                    "duration": 0.3,
                },
            )
        )

        tool_card = self.window.current_turn.tool_cards["call-cli-ansi"]
        self.assertIsNotNone(tool_card.cli_exec_widget)
        rendered = tool_card.cli_exec_widget.output_view.toPlainText()
        self.assertEqual(rendered, "boom\n")
        self.assertNotIn("\u001b", rendered)

    def test_run_finished_renders_plain_stats_chip(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Сводка"}))
        self.window._handle_event(StreamEvent("status_changed", {"label": "Self-correcting", "node": "recovery"}))
        self.window._handle_event(StreamEvent("run_finished", {"stats": "3.1s   In: 5328   Out: 106"}))

        self.assertEqual(self.window.current_turn.block_kinds(), ["user", "stats"])
        self.assertEqual(self.window.status_meta.text(), "")
        stats_widget = self.window.current_turn.layout().itemAt(1).widget()
        labels = [label.text() for label in stats_widget.findChildren(QLabel)]
        self.assertTrue(any("3.1s" in text and "Out: 106" in text for text in labels))
        self.assertFalse(any("[dim]" in text for text in labels))

    def test_inline_approval_card_resumes_controller_with_selected_choice(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Исправь файл"}))
        payload = {
            "tools": [{"name": "edit_file", "args": {"path": "demo.txt"}, "policy": {"mutating": True}}],
            "summary": {"risk_level": "medium", "impacts": ["files"], "default_approve": True},
        }

        self.window._handle_approval_request(payload)
        self._process_events()

        self.assertTrue(self.window.awaiting_approval)
        self.assertIsNone(self.window.current_turn.status_widget)
        self.assertFalse(self.window.approval_card.isHidden())
        self.assertIn("agent is paused", self.window.approval_card.summary_label.text().lower())
        QTest.mouseClick(self.window.approval_card.always_button, Qt.LeftButton)
        self._process_events()
        self.assertEqual(self.controller.resume_calls, [(True, True)])
        self.assertTrue(self.window.approval_card.isHidden())

    def test_user_choice_request_clears_inline_status(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Выбери режим"}))

        self.window._handle_user_choice_request(
            {
                "question": "Какой режим выбираем?",
                "recommended_key": "direct_api",
                "options": [
                    {
                        "key": "direct_api",
                        "label": "direct_api: убрать MCP и проверить только API",
                        "submit_text": "direct_api",
                        "recommended": True,
                    }
                ],
            }
        )

        self.assertIsNone(self.window.current_turn.status_widget)
        self.assertFalse(self.window.user_choice_card.isHidden())

    def test_auto_approval_after_always_mode_does_not_repeat_notice(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Исправь файл"}))

        self.window._handle_event(
            StreamEvent("approval_resolved", {"approved": True, "always": True, "auto": False})
        )
        self.assertEqual(self.window.current_turn.block_kinds(), ["user"])
        self.assertIn("approved", self.window.statusBar().currentMessage().lower())

        self.window._handle_event(
            StreamEvent("approval_resolved", {"approved": True, "always": True, "auto": True})
        )
        self.assertEqual(self.window.current_turn.block_kinds(), ["user"])

    def test_restored_session_cache_hit_survives_chat_reset(self):
        payload = self._snapshot_payload()
        payload["snapshot"]["cache_hit_tokens"] = 15_360
        self.window._handle_initialized(payload)

        self.assertEqual(self.window.cache_hit_label.text(), "Cache Hit: 15.4K")
        self.window._handle_event(StreamEvent("chat_reset", {}))
        self.assertEqual(self.window.cache_hit_label.text(), "Cache Hit: 15.4K")

    def test_new_session_clears_transcript_and_calls_controller(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "hello"}))
        self.window._handle_event(StreamEvent("cache_hit", {"tokens": 15_360}))
        self.assertIsNotNone(self.window.current_turn)
        self.assertEqual(self.window.cache_hit_label.text(), "Cache Hit: 15.4K")
        initial_sessions = self.window.sidebar.model.session_row_count()

        self.window._new_session()

        self.assertEqual(self.controller.new_session_calls, 1)
        self.assertIsNone(self.window.current_turn)
        self.assertEqual(self.window.cache_hit_label.text(), "Cache Hit: 0")
        self.assertEqual(self.window.sidebar.model.session_row_count(), initial_sessions)

    def test_open_new_project_creates_fresh_session_without_hiding_history(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "history"}))
        self.assertIsNotNone(self.window.current_turn)
        initial_sessions = self.window.sidebar.model.session_row_count()

        with (
            mock.patch.object(agent_cli.QFileDialog, "getExistingDirectory", return_value="D:/demo/workspace"),
            mock.patch("os.chdir") as chdir_mock,
        ):
            self.window._open_new_project()

        chdir_mock.assert_called_once_with("D:/demo/workspace")
        self.assertEqual(self.controller.reinitialize_calls, [True])
        self.assertIsNone(self.window.current_turn)
        self.assertEqual(self.window.sidebar.model.session_row_count(), initial_sessions)

    def test_user_message_uses_chat_style_bubble(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Короткий запрос"}))

        user_widget = self.window.current_turn.layout().itemAt(0).widget()

        self.assertEqual(user_widget.bubble.styleSheet(), "")
        self.assertIn("QFrame#UserBubble", self.window.styleSheet())
        self.assertIn("border-radius: 0px", self.window.styleSheet())

    def test_long_user_message_is_collapsible(self):
        self.window._handle_initialized(self._snapshot_payload())
        long_text = "Очень длинный запрос " * 80
        self.window._handle_event(StreamEvent("run_started", {"text": long_text}))

        user_widget = self.window.current_turn.layout().itemAt(0).widget()

        self.assertFalse(user_widget.toggle_button.isHidden())
        self.assertEqual(user_widget.toggle_button.text(), "Show more")
        self.assertNotEqual(user_widget.body.text(), long_text)
        self.assertTrue(user_widget.body.text().endswith("…"))

        user_widget.toggle_button.click()
        self._process_events()
        self.assertEqual(user_widget.toggle_button.text(), "Show less")
        self.assertEqual(user_widget.body.text(), long_text)

    def test_approval_summary_matches_expected_defaults(self):
        destructive = summarize_approval_request(
            [{"name": "safe_delete_file", "policy": {"destructive": True, "mutating": True}}]
        )
        mixed = summarize_approval_request(
            [{"name": "download_file", "policy": {"mutating": True, "networked": True}}]
        )
        regular = summarize_approval_request([{"name": "edit_file", "policy": {"mutating": True}}])

        self.assertFalse(destructive.default_approve)
        self.assertEqual(destructive.risk_level, "high")
        self.assertFalse(mixed.default_approve)
        self.assertIn("network", mixed.impacts)
        self.assertTrue(regular.default_approve)
        self.assertEqual(regular.risk_level, "medium")

    def test_stylesheet_keeps_expected_accent_colors(self):
        stylesheet = self.window.styleSheet()
        self.assertIn(AMBER_WARNING, stylesheet)
        self.assertIn(ERROR_RED, stylesheet)
        self.assertIn(BORDER, stylesheet)
        self.assertIn(SURFACE_BG, stylesheet)
        self.assertIn(SURFACE_CARD, stylesheet)
        self.assertIn(TEXT_MUTED, stylesheet)

    def test_settings_window_stays_within_screen_and_does_not_resize_main_window(self):
        for width in (900, 1024, 1280, 1366, 1920):
            with self.subTest(width=width):
                self.window.resize(width, 700)
                self.window.show()
                self._process_events()
                self.window._open_settings_dialog()
                self._process_events()

                dialog = self.window._model_settings_window
                self.assertIsNotNone(dialog)
                # The Settings panel is a frameless top-level window: it must
                # never widen the main window and must stay inside the screen.
                self.assertEqual(self.window.width(), width)
                available = dialog.screen().availableGeometry()
                self.assertLessEqual(dialog.x(), available.right())
                self.assertLessEqual(dialog.y(), available.bottom())
                self.assertGreaterEqual(dialog.x() + dialog.width(), available.left())
                self.assertGreaterEqual(dialog.y() + dialog.height(), available.top())
                self.assertLessEqual(dialog.width(), available.width())
                self.assertLessEqual(dialog.height(), available.height())

                self.window._open_settings_dialog()
                self._process_events()

    def test_settings_window_is_frameless_and_centered_on_screen(self):
        self.window.show()
        self._process_events()
        self.window._open_settings_dialog()
        self._process_events()

        dialog = self.window._model_settings_window
        self.assertIsNotNone(dialog)
        self.assertTrue(dialog.windowFlags() & Qt.WindowType.FramelessWindowHint)

        available = dialog.screen().availableGeometry()
        dialog_center = dialog.frameGeometry().center()
        # Allow a small tolerance for window-manager frame adjustments.
        self.assertLessEqual(abs(dialog_center.x() - available.center().x()), 4)
        self.assertLessEqual(abs(dialog_center.y() - available.center().y()), 4)

    def test_settings_window_blocks_main_window_input(self):
        self.window.show()
        self._process_events()
        self.window._set_input_enabled(True)
        self._process_events()
        self.assertTrue(self.window.composer.isEnabled())
        self.assertTrue(self.window.sidebar.isEnabled())

        self.window._open_settings_dialog()
        self._process_events()
        # The panel is a separate top-level window that may overlap the sidebar,
        # so the main window input must be disabled while it is open.
        self.assertFalse(self.window.composer.isEnabled())
        self.assertFalse(self.window.sidebar.isEnabled())

        self.window._model_settings_window.hide()
        self._process_events()
        self.assertTrue(self.window.composer.isEnabled())
        self.assertTrue(self.window.sidebar.isEnabled())

    def test_inspector_toggle_collapses_and_restores_right_panel(self):
        self.window.show()
        self._process_events()
        self.assertTrue(self.window.sidebar_container.isVisible())
        self.assertFalse(self.window.inspector_container.isVisible())
        collapsed_sizes = self.window.splitter.sizes()
        self.assertEqual(collapsed_sizes[2], 0)
        self.assertTrue(self.window.inspector_collapsed)

        self.window._toggle_info_popup()
        self._process_events()
        expanded_sizes = self.window.splitter.sizes()
        self.assertFalse(self.window.inspector_collapsed)
        self.assertGreater(expanded_sizes[2], 0)

        self.window._toggle_info_popup()
        self._process_events()
        restored_sizes = self.window.splitter.sizes()
        self.assertTrue(self.window.inspector_collapsed)
        self.assertEqual(restored_sizes[2], 0)

    def test_sidebar_toggle_hides_and_restores_chat_list(self):
        self.window.show()
        self._process_events()

        self.assertTrue(self.window.sidebar_container.isVisible())
        self.window._toggle_sidebar()
        self._process_events()
        self.assertFalse(self.window.sidebar_container.isVisible())

        self.window._toggle_sidebar()
        self._process_events()
        self.assertTrue(self.window.sidebar_container.isVisible())

    def test_clicking_sidebar_session_switches_controller_session(self):
        payload = self._snapshot_payload()
        self.window._handle_initialized(payload)

        index = self.window.sidebar.model.index_for_session("session-older")
        self.window.sidebar._emit_clicked_session(index)

        self.assertEqual(self.controller.switch_session_calls, ["session-older"])

    def test_sidebar_marks_active_project_group(self):
        payload = self._snapshot_payload()
        self.window._handle_initialized(payload)

        active_group_titles = []
        inactive_group_titles = []
        for row in range(self.window.sidebar.model.rowCount()):
            index = self.window.sidebar.model.index(row, 0)
            if index.data(SessionListModel.KindRole) != "group":
                continue
            title = index.data(SessionListModel.ProjectTitleRole)
            if index.data(SessionListModel.ActiveProjectRole):
                active_group_titles.append(title)
            else:
                inactive_group_titles.append(title)

        self.assertEqual(active_group_titles, ["workspace"])
        self.assertIn("other-project", inactive_group_titles)

    def test_sidebar_expansion_uses_row_notifications_instead_of_reset(self):
        model = SessionListModel()
        sessions = [
            {
                "session_id": f"session-{index}",
                "project_path": "D:/demo/workspace",
                "title": f"Chat {index}",
                "updated_at": f"2026-03-31T11:{index:02d}:00+00:00",
            }
            for index in range(6)
        ]
        resets = []
        model.modelAboutToBeReset.connect(lambda: resets.append(True))
        model.set_sessions(sessions)
        resets.clear()

        model.toggle_project_expansion("D:/demo/workspace")

        self.assertEqual(resets, [])
        self.assertEqual(model.session_row_count(), 6)
        model.toggle_project_expansion("D:/demo/workspace")
        self.assertEqual(resets, [])
        self.assertEqual(model.session_row_count(), 5)

    def test_sidebar_has_no_search_and_expands_project_groups(self):
        payload = self._snapshot_payload()
        for index in range(6):
            payload["sessions"].append(
                {
                    "session_id": f"session-extra-{index}",
                    "thread_id": f"thread-extra-{index}",
                    "project_path": "D:/demo/workspace",
                    "title": f"Extra chat {index}",
                    "created_at": f"2026-03-31T09:0{index}:00+00:00",
                    "updated_at": f"2026-03-31T11:0{index}:00+00:00",
                }
            )
        self.window._handle_initialized(payload)

        self.assertFalse(hasattr(self.window.sidebar, "search_field"))
        self.assertLess(self.window.sidebar.model.session_row_count(), len(payload["sessions"]))
        self.assertFalse(self.window.sidebar.list_view.isHidden())

        more_index = QModelIndex()
        for row in range(self.window.sidebar.model.rowCount()):
            candidate = self.window.sidebar.model.index(row, 0)
            if candidate.data(SessionListModel.KindRole) == "more":
                more_index = candidate
                break
        self.assertTrue(more_index.isValid())
        self.assertEqual(more_index.data(SessionListModel.TitleRole), "Show more")

        self.window.sidebar._emit_clicked_session(more_index)
        self._process_events()

        self.assertEqual(self.window.sidebar.model.session_row_count(), len(payload["sessions"]))
        self.assertFalse(self.window.sidebar.list_view.isHidden())

        collapse_index = QModelIndex()
        for row in range(self.window.sidebar.model.rowCount()):
            candidate = self.window.sidebar.model.index(row, 0)
            if candidate.data(SessionListModel.KindRole) == "more" and candidate.data(SessionListModel.TitleRole) == "Show less":
                collapse_index = candidate
                break
        self.assertTrue(collapse_index.isValid())

        self.window.sidebar._emit_clicked_session(collapse_index)
        self._process_events()

        self.assertLess(self.window.sidebar.model.session_row_count(), len(payload["sessions"]))
        more_titles = [
            self.window.sidebar.model.index(row, 0).data(SessionListModel.TitleRole)
            for row in range(self.window.sidebar.model.rowCount())
            if self.window.sidebar.model.index(row, 0).data(SessionListModel.KindRole) == "more"
        ]
        self.assertIn("Show more", more_titles)

    def test_delete_session_requests_controller_after_confirmation(self):
        payload = self._snapshot_payload()
        self.window._handle_initialized(payload)

        with mock.patch.object(agent_cli.QMessageBox, "question", return_value=agent_cli.QMessageBox.Yes):
            self.window._request_delete_session("session-older")

        self.assertEqual(self.controller.delete_session_calls, ["session-older"])

    def test_delete_session_is_cancelled_without_confirmation(self):
        payload = self._snapshot_payload()
        self.window._handle_initialized(payload)

        with mock.patch.object(agent_cli.QMessageBox, "question", return_value=agent_cli.QMessageBox.No):
            self.window._request_delete_session("session-older")

        self.assertEqual(self.controller.delete_session_calls, [])

    def test_delete_project_requests_controller_after_confirmation(self):
        payload = self._snapshot_payload()
        payload["sessions"].append(
            {
                "session_id": "session-other-2",
                "thread_id": "thread-other-2",
                "project_path": "D:/demo/other-project",
                "title": "Second chat [demo/other-project]",
                "created_at": "2026-03-30T09:00:00+00:00",
                "updated_at": "2026-03-30T11:00:00+00:00",
            }
        )
        self.window._handle_initialized(payload)

        with mock.patch.object(
            agent_cli.QMessageBox, "question", return_value=agent_cli.QMessageBox.Yes
        ) as question_mock:
            self.window._request_delete_project("D:/demo/other-project")

        self.assertEqual(self.controller.delete_project_calls, ["D:/demo/other-project"])
        self.assertEqual(self.controller.delete_session_calls, [])
        self.assertIn("2 chats", question_mock.call_args.args[2])
        self.assertIn("other-project", question_mock.call_args.args[2])

    def test_delete_project_is_cancelled_without_confirmation(self):
        payload = self._snapshot_payload()
        self.window._handle_initialized(payload)

        with mock.patch.object(agent_cli.QMessageBox, "question", return_value=agent_cli.QMessageBox.No):
            self.window._request_delete_project("D:/demo/other-project")

        self.assertEqual(self.controller.delete_project_calls, [])

    def test_sidebar_routes_right_click_to_project_or_chat_menu(self):
        self.window._handle_initialized(self._snapshot_payload())
        sidebar = self.window.sidebar
        rows = {
            str(sidebar.model.index(row, 0).data(SessionListModel.KindRole)): row
            for row in range(sidebar.model.rowCount())
        }
        group_index = sidebar.model.index(rows["group"], 0)
        session_index = sidebar.model.index(rows["session"], 0)

        with (
            mock.patch.object(sidebar, "_show_project_context_menu") as project_menu_mock,
            mock.patch.object(sidebar, "_show_session_context_menu") as session_menu_mock,
            mock.patch.object(sidebar.list_view, "indexAt", return_value=group_index),
        ):
            sidebar._show_context_menu(QPoint(4, 4))

        session_menu_mock.assert_not_called()
        project_menu_mock.assert_called_once()
        self.assertIs(project_menu_mock.call_args.args[0], group_index)

        with (
            mock.patch.object(sidebar, "_show_project_context_menu") as project_menu_mock,
            mock.patch.object(sidebar, "_show_session_context_menu") as session_menu_mock,
            mock.patch.object(sidebar.list_view, "indexAt", return_value=session_index),
        ):
            sidebar._show_context_menu(QPoint(4, 4))

        project_menu_mock.assert_not_called()
        session_menu_mock.assert_called_once()
        self.assertIs(session_menu_mock.call_args.args[0], session_index)

    def test_sidebar_project_group_counts_chats_for_delete_prompt(self):
        payload = self._snapshot_payload()
        payload["sessions"].append(
            {
                "session_id": "session-other-2",
                "thread_id": "thread-other-2",
                "project_path": "D:/demo/other-project",
                "title": "Second chat [demo/other-project]",
                "created_at": "2026-03-30T09:00:00+00:00",
                "updated_at": "2026-03-30T11:00:00+00:00",
            }
        )
        self.window._handle_initialized(payload)
        sidebar = self.window.sidebar

        self.assertEqual(sidebar.session_count_for_project("D:/demo/other-project"), 2)
        self.assertEqual(sidebar.session_count_for_project("D:/demo/workspace"), 1)
        self.assertEqual(sidebar.session_count_for_project("D:/demo/missing"), 0)
        self.assertEqual(sidebar.session_count_for_project(""), 0)
        self.assertEqual(sidebar.title_for_project("D:/demo/other-project"), "other-project")

    def test_menu_bar_uses_corner_buttons_and_compact_composer(self):
        self.assertEqual(self.window.send_button.text(), "")
        self.assertLessEqual(self.window.composer.height(), 72)
        self.assertLessEqual(self.window.send_button.size().width(), 38)
        self.assertEqual(self.window.cache_hit_label.text(), "Cache Hit: 0")
        self.assertTrue(self.window.cache_hit_label.alignment() & Qt.AlignRight)
        self.assertIs(self.window.cache_hit_label.parentWidget(), self.window.composer_pill)
        self.assertIsNone(self.window.findChild(QToolBar))
        self.assertIsNotNone(self.window.menuWidget())
        embedded_menu = self.window.menuWidget().findChild(agent_cli.QMenuBar)
        self.assertIsNotNone(embedded_menu)
        self.assertEqual(embedded_menu.actions()[0].text(), "File")
        self.assertEqual(embedded_menu.actions()[1].text(), "View")
        self.assertEqual(self.window.new_session_button.iconSize().width(), 14)
        self.assertEqual(self.window.new_project_button.iconSize().width(), 14)
        self.assertEqual(self.window.info_button.iconSize().width(), 14)

    def test_approval_card_blocks_input_and_exposes_human_readable_fields(self):
        self.window._handle_initialized(self._snapshot_payload())
        payload = {
            "tools": [{"name": "edit_file", "display": "edit_file", "args": {"path": "demo.txt", "newText": "hello"}, "policy": {"mutating": True}}],
            "summary": {"risk_level": "medium", "impacts": ["files"], "default_approve": True},
        }

        self.window._handle_approval_request(payload)
        self._process_events()

        self.assertFalse(self.window.composer.isEnabled())
        self.assertEqual(self.window.approval_card.risk_badge.text(), "Medium")
        self.assertIn("Will affect: files", self.window.approval_card.impacts_label.text())
        self.assertEqual(self.window.approval_card.approve_button.text(), "Approve")
        self.assertEqual(self.window.approval_card.always_button.text(), "Always allow")
        self.assertEqual(self.window.approval_card.deny_button.text(), "Deny")
        detail_view = self.window.approval_card._tool_sections[0].content
        self.assertIsInstance(detail_view, CopySafePlainTextEdit)
        self.assertIn("Path: demo.txt", detail_view.toPlainText())
        self.assertIn("New Text: hello", detail_view.toPlainText())
        self.assertNotIn("{", detail_view.toPlainText())

    def test_transcript_sticky_autofollow_respects_manual_scroll(self):
        scrollbar = self.window.transcript.scroll.verticalScrollBar()
        scrollbar.setRange(0, 200)
        scrollbar.setValue(200)
        self.window.transcript.scroll_to_bottom()
        self.assertTrue(self.window.transcript.auto_follow_enabled)

        scrollbar.setValue(80)
        self._process_events()
        self.assertFalse(self.window.transcript.auto_follow_enabled)
        previous_value = scrollbar.value()

        self.window.transcript.notify_content_changed()
        self._process_events()
        self.assertEqual(scrollbar.value(), previous_value)

        scrollbar.setValue(scrollbar.maximum())
        self._process_events()
        self.assertTrue(self.window.transcript.auto_follow_enabled)

        scrollbar.setValue(120)
        self.window.transcript.notify_content_changed(force=True)
        self._process_events()
        self.assertEqual(scrollbar.value(), scrollbar.maximum())

    def test_transcript_autofollow_handles_range_growth_after_initial_scroll(self):
        scrollbar = self.window.transcript.scroll.verticalScrollBar()
        scrollbar.setRange(0, 200)
        scrollbar.setValue(200)
        self.window.transcript.scroll_to_bottom()
        self.assertTrue(self.window.transcript.auto_follow_enabled)

        self.window.transcript.notify_content_changed()
        self._process_events()
        scrollbar.setRange(0, 280)
        self._process_events()
        self.assertEqual(scrollbar.value(), scrollbar.maximum())

    def test_transcript_jump_button_appears_when_autofollow_is_off_and_scrolls_to_latest(self):
        self.window.show()
        self._process_events()
        scrollbar = self.window.transcript.scroll.verticalScrollBar()
        scrollbar.setRange(0, 240)
        with mock.patch.object(self.window.transcript, "is_near_bottom", return_value=False):
            self.window.transcript._auto_follow_enabled = False
            self.window.transcript._update_jump_button()
            self._process_events()

        self.assertFalse(self.window.transcript.auto_follow_enabled)
        self.assertFalse(self.window.transcript.jump_to_latest_button.isHidden())

        QTest.mouseClick(self.window.transcript.jump_to_latest_button, Qt.LeftButton)
        self._process_events()

        self.assertTrue(self.window.transcript.auto_follow_enabled)
        self.assertFalse(self.window.transcript.jump_to_latest_button.isVisible())
        self.assertEqual(scrollbar.value(), scrollbar.maximum())

    def test_transcript_uses_centered_column_instead_of_full_width_feed(self):
        self.assertEqual(self.window.transcript.column.maximumWidth(), TRANSCRIPT_MAX_WIDTH)
        self.assertEqual(self.window.transcript.column.objectName(), "TranscriptColumn")
        self.assertEqual(self.window.composer_container.maximumWidth(), TRANSCRIPT_MAX_WIDTH)
        self.assertEqual(self.window.composer_container.objectName(), "CenteredComposerRow")
        self.assertGreaterEqual(self.window.composer_shell.contentsMargins().top(), 16)

    def test_composer_buttons_have_correct_tooltips(self):
        self.assertEqual(self.window.attach_button.toolTip(), "Add images or insert file paths")
        self.assertEqual(self.window.send_button.toolTip(), "Send (Enter)")

    def test_model_selector_renders_active_profile_and_tooltip(self):
        self.window._handle_initialized(self._snapshot_payload())

        self.assertFalse(self.window.model_chip.isHidden())
        self.assertEqual(self.window.model_chip.text(), "gpt-4o")
        self.assertIn("Provider: openai", self.window.model_chip.toolTip())
        self.assertIn("Model: gpt-4o", self.window.model_chip.toolTip())
        self.assertTrue(self.window.no_models_label.isHidden())

    def test_model_selector_action_triggers_controller_switch(self):
        self.window._handle_initialized(self._snapshot_payload())
        actions = self.window.model_chip_menu.actions()
        target_action = next(action for action in actions if action.text() == "gemini-1-5-flash")

        target_action.trigger()

        self.assertEqual(self.controller.set_active_profile_calls, ["gemini-1-5-flash"])

    def test_reasoning_selector_uses_registry_options_and_saves_selection(self):
        payload = self._snapshot_payload()
        profile = payload["model_profiles"]["profiles"][0]
        profile.update({"model": "gpt-5.6", "base_url": "https://api.openai.com/v1"})
        self.window._handle_initialized(payload)

        self.assertFalse(self.window.reasoning_chip.isHidden())
        self.assertEqual(self.window.reasoning_chip.text(), "Reasoning")
        self.assertNotIn("Default", [action.text() for action in self.window.reasoning_chip_menu.actions()])
        action = next(action for action in self.window.reasoning_chip_menu.actions() if action.text() == "High")
        action.trigger()

        self.assertEqual(self.window.reasoning_chip.text(), "High")
        self.assertEqual(
            self.controller.save_profiles_calls[-1]["profiles"][0]["reasoning"],
            {"enabled": True, "effort": "high"},
        )

    def test_reasoning_selector_preserves_legacy_disabled_profile_state(self):
        payload = self._snapshot_payload()
        profile = payload["model_profiles"]["profiles"][0]
        profile.update(
            {
                "model": "gpt-5.6",
                "base_url": "https://api.openai.com/v1",
                "reasoning": {"enabled": False},
            }
        )
        self.window._handle_initialized(payload)

        self.assertEqual(self.window.reasoning_chip.text(), "Off")
        self.assertNotIn("Off", [action.text() for action in self.window.reasoning_chip_menu.actions()])
        self.assertEqual(
            [action.text() for action in self.window.reasoning_chip_menu.actions()],
            ["None", "Minimal", "Low", "Medium", "High", "X-High", "Max"],
        )

    def test_nvidia_gpt_oss_reasoning_selector_saves_effort_level(self):
        payload = self._snapshot_payload()
        profile = payload["model_profiles"]["profiles"][0]
        profile.update(
            {
                "model": "openai/gpt-oss-120b",
                "base_url": "https://integrate.api.nvidia.com/v1",
            }
        )
        self.window._handle_initialized(payload)

        self.assertEqual(
            [action.text() for action in self.window.reasoning_chip_menu.actions()],
            ["Low", "Medium", "High"],
        )
        action = next(action for action in self.window.reasoning_chip_menu.actions() if action.text() == "Medium")
        action.trigger()

        self.assertEqual(self.window.reasoning_chip.text(), "Medium")
        self.assertEqual(
            self.controller.save_profiles_calls[-1]["profiles"][0]["reasoning"],
            {"enabled": True, "effort": "medium"},
        )

    def test_agentrouter_glm5_reasoning_selector_exposes_supported_effort_levels(self):
        payload = self._snapshot_payload()
        profile = payload["model_profiles"]["profiles"][0]
        profile.update(
            {
                "model": "zai-org/glm-5.1",
                "base_url": "https://agentrouter.org/v1",
                "reasoning": {"enabled": True, "effort": "medium"},
            }
        )
        self.window._handle_initialized(payload)

        self.assertEqual(
            [action.text() for action in self.window.reasoning_chip_menu.actions()],
            ["Low", "High", "Max"],
        )
        self.assertEqual(self.window.reasoning_chip.text(), "Reasoning")

    def test_nvidia_thinking_reasoning_selector_saves_binary_mode(self):
        payload = self._snapshot_payload()
        profile = payload["model_profiles"]["profiles"][0]
        profile.update(
            {
                "model": "qwen/qwen3-235b-a22b",
                "base_url": "https://integrate.api.nvidia.com/v1",
            }
        )
        self.window._handle_initialized(payload)

        self.assertEqual([action.text() for action in self.window.reasoning_chip_menu.actions()], ["Off", "On"])
        self.assertEqual(self.window.reasoning_chip.text(), "On")
        action = next(action for action in self.window.reasoning_chip_menu.actions() if action.text() == "On")
        action.trigger()

        self.assertEqual(self.window.reasoning_chip.text(), "On")
        self.assertEqual(
            self.controller.save_profiles_calls[-1]["profiles"][0]["reasoning"],
            {"enabled": True},
        )

        action = next(action for action in self.window.reasoning_chip_menu.actions() if action.text() == "Off")
        action.trigger()

        self.assertEqual(self.window.reasoning_chip.text(), "Off")
        self.assertEqual(
            self.controller.save_profiles_calls[-1]["profiles"][0]["reasoning"],
            {"enabled": False},
        )

    def test_no_models_cta_is_visible_and_send_is_disabled(self):
        payload = self._snapshot_payload()
        payload["model_profiles"] = {"active_profile": None, "profiles": []}
        self.window._handle_initialized(payload)
        self.window._set_input_enabled(True)

        self.assertTrue(self.window.model_chip.isHidden())
        self.assertFalse(self.window.no_models_label.isHidden())
        self.assertFalse(self.window.open_settings_inline_button.isHidden())
        self.assertFalse(self.window.send_button.isEnabled())

    def test_open_settings_dialog_saves_profiles_via_controller(self):
        self.window._handle_initialized(self._snapshot_payload())
        payload = {
            "active_profile": "gemini-1-5-flash",
            "profiles": [
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-demo",
                    "base_url": "",
                }
            ],
        }

        class _FakeSignal:
            def __init__(self):
                self._callbacks = []

            def connect(self, callback):
                self._callbacks.append(callback)

            def emit(self, value):
                for callback in list(self._callbacks):
                    callback(value)

        class _FakeDialog:
            def __init__(self, emitted_payload):
                self.profiles_saved = _FakeSignal()
                self._payload = emitted_payload
                self._visible = False
                self.destroyed = _FakeSignal()

            def isVisible(self):
                return self._visible

            def show(self):
                self._visible = True
                self.profiles_saved.emit(self._payload)

            def raise_(self):
                return None

            def activateWindow(self):
                return None

        dialog_instance = _FakeDialog(payload)

        with mock.patch.object(agent_cli, "ModelSettingsDialog", return_value=dialog_instance):
            self.window._open_settings_dialog()

        self.assertEqual(
            self.controller.save_profiles_calls,
            [normalize_profiles_payload(payload)],
        )
        self.assertEqual(self.window.model_profiles_payload["active_profile"], "gemini-1-5-flash")

    def test_model_settings_dialog_preserves_reasoning_profile_setting(self):
        payload = {
            "active_profile": "gpt-5",
            "profiles": [
                {
                    "id": "gpt-5",
                    "provider": "openai",
                    "model": "gpt-5.6",
                    "api_key": "sk-demo",
                    "base_url": "https://api.openai.com/v1",
                    "reasoning": {"enabled": True, "effort": "high"},
                }
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.deleteLater)

        self.assertEqual(dialog._validated_payload()["profiles"][0]["reasoning"], {"enabled": True, "effort": "high"})

    def test_model_settings_panel_reopens_with_content_and_toggles_from_settings_action(self):
        self.window._handle_initialized(self._snapshot_payload())

        self.window.settings_action.trigger()
        self._process_events()

        dialog = self.window._model_settings_window
        profile_count = dialog.profile_list.count()
        self.assertFalse(dialog.isHidden())
        self.assertGreater(profile_count, 0)

        dialog.close_button.click()
        self._process_events()
        self.assertTrue(dialog.isHidden())

        self.window.settings_action.trigger()
        self._process_events()
        self.assertFalse(dialog.isHidden())
        self.assertIs(self.window._model_settings_window, dialog)
        self.assertEqual(dialog.profile_list.count(), profile_count)

        self.window.settings_action.trigger()
        self._process_events()
        self.assertTrue(dialog.isHidden())

    def test_model_settings_dialog_does_not_wipe_profile_on_initial_selection(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "openai/gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "https://api.openai.com/v1",
                    "supports_image_input": True,
                    "enabled": True,
                }
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertEqual(dialog._profiles[0]["model"], "openai/gpt-4o")
        self.assertEqual(dialog._profiles[0]["id"], "gpt-4o")
        self.assertEqual(dialog.model_edit.text(), "openai/gpt-4o")
        self.assertTrue(dialog.supports_images_checkbox.isChecked())

    def test_model_settings_dialog_save_keeps_window_open_and_emits_payload(self):
        dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
        self.addCleanup(dialog.close)
        saved_payloads = []
        dialog.profiles_saved.connect(saved_payloads.append)
        dialog.show()
        self._process_events()

        dialog._add_profile()
        self._process_events()
        dialog.model_edit.setText("openai/gpt-oss-120b")
        dialog.api_key_edit.setText("sk-demo")
        self._process_events()

        dialog._save_and_accept()
        self._process_events()

        self.assertTrue(dialog.isVisible())
        self.assertEqual(len(saved_payloads), 1)
        self.assertEqual(saved_payloads[0]["profiles"][0]["id"], "gpt-oss-120b")
        self.assertIn("keep this window open", dialog.save_state_label.text())
        self.assertEqual(dialog.save_button.text(), "Saved")
        self.assertTrue(bool(dialog.save_button.property("savedState")))
        self.assertEqual(dialog._save_button_reset_timer.interval(), 3000)
        self.assertTrue(dialog._save_button_reset_timer.isActive())

        dialog._save_button_reset_timer.start(1)
        self._wait_for_gui(10)
        self.assertEqual(dialog.save_button.text(), "Save")
        self.assertFalse(bool(dialog.save_button.property("savedState")))

    def test_model_settings_dialog_opens_with_active_profile_selected(self):
        payload = {
            "active_profile": "gemini-1-5-flash",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "",
                    "enabled": True,
                },
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-demo",
                    "base_url": "",
                    "enabled": True,
                },
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertEqual(dialog._current_row(), 1)
        self.assertIn("gemini-1-5-flash", dialog.form_hint.text())

    def test_model_settings_panel_reopens_with_currently_active_profile_selected(self):
        self.window._handle_initialized(self._snapshot_payload())

        self.window.settings_action.trigger()
        self._process_events()

        dialog = self.window._model_settings_window
        self.assertEqual(dialog._current_row(), 0)
        self.assertIn("gpt-4o", dialog.form_hint.text())

        # Switch the active model while the panel is hidden.
        dialog.close_button.click()
        self._process_events()
        self.assertTrue(dialog.isHidden())

        switched = normalize_profiles_payload(self.window.model_profiles_payload)
        switched["active_profile"] = "gemini-1-5-flash"
        self.window._apply_model_profiles_payload(switched)
        self._process_events()

        self.window.settings_action.trigger()
        self._process_events()

        self.assertFalse(dialog.isHidden())
        self.assertEqual(dialog._current_row(), 1)
        self.assertIn("gemini-1-5-flash", dialog.form_hint.text())

    def _active_badge_row(self, dialog) -> int:
        for row in range(dialog.profile_list.count()):
            item = dialog.profile_list.item(row)
            widget = dialog.profile_list.itemWidget(item) if item is not None else None
            if widget is None:
                continue
            for label in widget.findChildren(QLabel, "ModelProfileItemBadge"):
                if label.property("badgeVariant") == "active":
                    return row
        return -1

    def test_model_settings_dialog_shows_active_badge_on_active_profile(self):
        payload = {
            "active_profile": "gemini-1-5-flash",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "",
                    "enabled": True,
                },
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-demo",
                    "base_url": "",
                    "enabled": True,
                },
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertEqual(dialog._current_row(), 1)
        self.assertEqual(self._active_badge_row(dialog), 1)

    def test_model_settings_panel_reopens_with_active_badge_on_active_profile(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.settings_action.trigger()
        self._process_events()

        dialog = self.window._model_settings_window
        self.assertEqual(self._active_badge_row(dialog), 0)

        dialog.close_button.click()
        self._process_events()
        self.assertTrue(dialog.isHidden())

        switched = normalize_profiles_payload(self.window.model_profiles_payload)
        switched["active_profile"] = "gemini-1-5-flash"
        self.window._apply_model_profiles_payload(switched)
        self._process_events()

        self.window.settings_action.trigger()
        self._process_events()

        self.assertFalse(dialog.isHidden())
        self.assertEqual(dialog._current_row(), 1)
        self.assertEqual(self._active_badge_row(dialog), 1)

    def test_model_settings_dialog_autofills_name_from_api_url_and_model(self):
        dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
        self.addCleanup(dialog.close)
        dialog._add_profile()
        self._process_events()
        dialog.base_url_edit.setText("https://api.openai.com/v1")
        dialog.model_edit.setText("openai/gpt-oss-120b")
        self._process_events()

        self.assertEqual(dialog.name_edit.text(), "opn/gpt-oss-120b")
        dialog._save_and_accept()
        result = dialog.result_payload()
        self.assertEqual(result["profiles"][0]["id"], "opn/gpt-oss-120b")

    def test_model_settings_dialog_updates_auto_name_when_api_url_changes(self):
        dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
        self.addCleanup(dialog.close)
        dialog._add_profile()
        self._process_events()
        dialog.model_edit.setText("gpt-4o")
        dialog.base_url_edit.setText("https://api.openai.com/v1")
        self._process_events()

        self.assertEqual(dialog.name_edit.text(), "opn/gpt-4o")

    def test_model_settings_dialog_autofills_name_from_model_suffix_without_api_url(self):
        dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
        self.addCleanup(dialog.close)
        dialog._add_profile()
        self._process_events()
        dialog.model_edit.setText("openai/gpt-oss-120b")
        self._process_events()

        self.assertEqual(dialog.name_edit.text(), "gpt-oss-120b")
        dialog._save_and_accept()
        result = dialog.result_payload()
        self.assertEqual(result["profiles"][0]["id"], "gpt-oss-120b")

    def test_model_settings_dialog_updates_auto_name_when_model_changes(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "",
                    "enabled": True,
                }
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        dialog.model_edit.setText("openai/gpt-oss-120b")
        self._process_events()

        self.assertEqual(dialog.name_edit.text(), "gpt-oss-120b")

    def test_model_settings_dialog_keeps_manually_entered_model_after_save_and_reload(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "https://api.openai.com/v1",
                    "enabled": True,
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
            self.addCleanup(dialog.close)
            self._process_events()

            dialog._apply_loaded_models(
                [
                    ModelEntry("gpt-4o", "openai", True),
                    ModelEntry("gpt-4o-mini", "openai", True),
                ]
            )
            self._process_events()

            dialog.model_combo.setEditText("custom-manual-model")
            self._process_events()
            self.assertEqual(dialog._get_current_model_value(), "custom-manual-model")

            dialog._save_and_accept()
            self._process_events()
            result = dialog.result_payload()
            self.assertEqual(result["profiles"][0]["model"], "custom-manual-model")
            self.assertEqual(dialog._get_current_model_value(), "custom-manual-model")

            # A later model-list reload (e.g. after Save) must not wipe the manual entry.
            dialog._apply_loaded_models(
                [
                    ModelEntry("gpt-4o", "openai", True),
                    ModelEntry("gpt-4o-mini", "openai", True),
                ]
            )
            self._process_events()
            self.assertEqual(dialog._get_current_model_value(), "custom-manual-model")
            dialog._save_and_accept()
            self.assertEqual(dialog.result_payload()["profiles"][0]["model"], "custom-manual-model")

    def test_model_settings_dialog_updates_auto_name_when_only_model_is_replaced(self):
        payload = {
            "active_profile": "opn/gpt-4o",
            "profiles": [
                {
                    "id": "opn/gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "https://api.openai.com/v1",
                    "enabled": True,
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
            self.addCleanup(dialog.close)
            self._process_events()

            dialog._apply_loaded_models([ModelEntry("gpt-4.1", "openai", False)])
            self._process_events()

            self.assertEqual(dialog.model_combo.currentText(), "gpt-4.1")
            self.assertEqual(dialog.name_edit.text(), "opn/gpt-4-1")

            dialog._save_and_accept()
            result = dialog.result_payload()
            self.assertEqual(result["profiles"][0]["model"], "gpt-4.1")
            self.assertEqual(result["profiles"][0]["id"], "opn/gpt-4-1")

    def test_model_settings_dialog_preserves_manual_name_when_only_model_is_replaced(self):
        payload = {
            "active_profile": "custom-profile",
            "profiles": [
                {
                    "id": "custom-profile",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "https://api.openai.com/v1",
                    "enabled": True,
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
            self.addCleanup(dialog.close)
            self._process_events()

            dialog._apply_loaded_models([ModelEntry("gpt-4.1", "openai", False)])
            self._process_events()

            self.assertEqual(dialog.model_combo.currentText(), "gpt-4.1")
            self.assertEqual(dialog.name_edit.text(), "custom-profile")

    def test_model_settings_dialog_saves_manual_image_support_checkbox(self):
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
            self.addCleanup(dialog.close)
            dialog._add_profile()
            self._process_events()

            dialog.provider_combo.setCurrentText("gemini")
            dialog.model_edit.setText("gemini-2.5-pro")
            dialog.api_key_edit.setText("gm-demo")
            dialog.supports_images_checkbox.setChecked(True)
            self._process_events()

            dialog._save_and_accept()
            result = dialog.result_payload()

            self.assertTrue(result["profiles"][0]["supports_image_input"])

    def test_model_settings_dialog_does_not_expose_thinking_toggle_or_save_legacy_field(self):
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
            self.addCleanup(dialog.close)
            dialog._add_profile()
            self._process_events()

            dialog.provider_combo.setCurrentText("gemini")
            dialog.model_edit.setText("gemini-2.5-flash")
            dialog.api_key_edit.setText("gm-demo")
            self._process_events()

            dialog._save_and_accept()
            result = dialog.result_payload()

            self.assertFalse(hasattr(dialog, "show_model_thoughts_checkbox"))
            self.assertFalse(hasattr(dialog, "summary_thoughts"))
            self.assertNotIn("show_model_thoughts", result["profiles"][0])

    def test_model_settings_dialog_ignores_legacy_show_model_thoughts_from_profile(self):
        payload = {
            "active_profile": "gemini-2-5-flash",
            "profiles": [
                {
                    "id": "gemini-2-5-flash",
                    "provider": "gemini",
                    "model": "gemini-2.5-flash",
                    "api_key": "gm-demo",
                    "base_url": "",
                    "show_model_thoughts": True,
                    "enabled": True,
                }
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertFalse(hasattr(dialog, "show_model_thoughts_checkbox"))
        self.assertFalse(hasattr(dialog, "summary_thoughts"))
        dialog._save_and_accept()
        self.assertNotIn("show_model_thoughts", dialog.result_payload()["profiles"][0])

    def test_model_settings_dialog_rotation_editor_updates_profile_and_saves_with_main_save(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-primary",
                    "api_keys": ["sk-primary"],
                    "base_url": "https://api.openai.com/v1",
                    "enabled": True,
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
            self.addCleanup(dialog.close)
            dialog.show()
            self._process_events()

            dialog._edit_api_key_rotation()
            self.assertTrue(dialog.api_key_rotation_section.content_container.isVisible())
            dialog.api_key_rotation_editor.setPlainText("sk-primary\nsk-secondary")
            self._process_events()
            dialog._save_and_accept()
            self._process_events()

            saved_payload = dialog.result_payload()
            self.assertEqual(saved_payload["profiles"][0]["api_keys"], ["sk-primary", "sk-secondary"])
            self.assertEqual(saved_payload["profiles"][0]["api_key"], "sk-primary")
            self.assertIn("keep this window open", dialog.save_state_label.text())

    def test_model_settings_dialog_applies_rotation_pool_per_profile(self):
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
            self.addCleanup(dialog.close)
            dialog._add_profile()
            self._process_events()

            dialog.model_edit.setText("openai/gpt-4o")
            dialog.api_key_edit.setText("sk-primary")
            dialog._apply_api_key_rotation_to_profile(0, [" sk-primary ", "", "sk-secondary", "sk-primary"])
            dialog._save_and_accept()
            result = dialog.result_payload()

            self.assertEqual(result["profiles"][0]["api_keys"], ["sk-primary", "sk-secondary"])
            self.assertEqual(result["profiles"][0]["api_key"], "sk-primary")
            self.assertEqual(result["profiles"][0]["api_key_index"], 0)

    def test_model_settings_dialog_rotation_pool_stays_isolated_between_profiles(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-one",
                    "api_keys": ["sk-one", "sk-two"],
                    "base_url": "",
                    "enabled": True,
                },
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-one",
                    "api_keys": ["gm-one"],
                    "base_url": "",
                    "enabled": True,
                },
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
            self.addCleanup(dialog.close)
            self._process_events()

            dialog._apply_api_key_rotation_to_profile(1, ["gm-one", "gm-two"])
            dialog._save_and_accept()
            result = dialog.result_payload()

            self.assertEqual(result["profiles"][0]["api_keys"], ["sk-one", "sk-two"])
            self.assertEqual(result["profiles"][1]["api_keys"], ["gm-one", "gm-two"])

    def test_model_settings_dialog_save_preserves_rotation_metadata(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-two",
                    "api_keys": ["sk-one", "sk-two", "sk-three"],
                    "api_key_index": 1,
                    "invalid_api_keys": [],
                    "key_error_timestamps": {},
                    "base_url": "https://api.openai.com/v1",
                    "enabled": True,
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
            self.addCleanup(dialog.close)
            self._process_events()

            dialog.base_url_edit.setText("https://api.openai.com/v1")
            dialog._save_and_accept()
            result = dialog.result_payload()

            self.assertEqual(result["profiles"][0]["api_key"], "sk-two")
            self.assertEqual(result["profiles"][0]["api_keys"], ["sk-one", "sk-two", "sk-three"])
            self.assertEqual(result["profiles"][0]["api_key_index"], 1)
            self.assertEqual(result["profiles"][0]["invalid_api_keys"], [])
            self.assertEqual(result["profiles"][0]["key_error_timestamps"], {})

    def test_model_settings_dialog_rotation_pool_keeps_current_key_after_edit(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-two",
                    "api_keys": ["sk-one", "sk-two", "sk-three"],
                    "api_key_index": 1,
                    "invalid_api_keys": [],
                    "key_error_timestamps": {},
                    "base_url": "https://api.openai.com/v1",
                    "enabled": True,
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
            self.addCleanup(dialog.close)
            self._process_events()

            dialog._apply_api_key_rotation_to_profile(0, ["sk-one", "sk-three"])

            self.assertEqual(dialog._profiles[0]["api_key"], "sk-one")
            self.assertEqual(dialog._profiles[0]["api_key_index"], 0)
            self.assertEqual(dialog._profiles[0]["invalid_api_keys"], [])
            self.assertEqual(dialog._profiles[0]["key_error_timestamps"], {})

    def test_model_settings_dialog_toggle_disables_profile_without_deleting_it(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "",
                    "enabled": True,
                },
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-demo",
                    "base_url": "",
                    "enabled": True,
                },
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        dialog._toggle_profile_enabled(0, False)
        dialog._save_and_accept()
        result = dialog.result_payload()

        self.assertFalse(result["profiles"][0]["enabled"])
        self.assertEqual(result["active_profile"], "gemini-1-5-flash")

    def test_model_settings_dialog_switch_can_disable_and_reenable_profile(self):
        payload = {
            "active_profile": "mistral-medium-latest",
            "profiles": [
                {
                    "id": "gemini-3-1-flash-lite-preview",
                    "provider": "gemini",
                    "model": "gemini-3.1-flash-lite-preview",
                    "api_key": "gm-demo",
                    "base_url": "",
                    "enabled": True,
                },
                {
                    "id": "mistral-medium-latest",
                    "provider": "openai",
                    "model": "mistral-medium-latest",
                    "api_key": "sk-demo",
                    "base_url": "",
                    "enabled": True,
                },
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertGreaterEqual(dialog.profile_list.minimumWidth(), 300)

        first_item_widget = dialog.profile_list.itemWidget(dialog.profile_list.item(0))
        self.assertIsNotNone(first_item_widget)
        switch = first_item_widget.findChild(QCheckBox, "ModelProfileEnabledSwitch")
        self.assertIsNotNone(switch)
        self.assertGreaterEqual(switch.size().width(), 34)
        self.assertTrue(switch.isChecked())

        QTest.mouseClick(switch, Qt.LeftButton)
        self._process_events()

        updated_item_widget = dialog.profile_list.itemWidget(dialog.profile_list.item(0))
        self.assertIsNotNone(updated_item_widget)
        updated_switch = updated_item_widget.findChild(QCheckBox, "ModelProfileEnabledSwitch")
        self.assertIsNotNone(updated_switch)
        self.assertFalse(updated_switch.isChecked())

        QTest.mouseClick(updated_switch, Qt.LeftButton)
        self._process_events()

        reenabled_item_widget = dialog.profile_list.itemWidget(dialog.profile_list.item(0))
        self.assertIsNotNone(reenabled_item_widget)
        reenabled_switch = reenabled_item_widget.findChild(QCheckBox, "ModelProfileEnabledSwitch")
        self.assertIsNotNone(reenabled_switch)
        self.assertTrue(reenabled_switch.isChecked())

        dialog._save_and_accept()
        result = dialog.result_payload()
        self.assertTrue(result["profiles"][0]["enabled"])

    def test_model_settings_dialog_toggle_keeps_same_profile_selected(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "",
                    "enabled": True,
                },
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-demo",
                    "base_url": "",
                    "enabled": True,
                },
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        dialog.profile_list.setCurrentRow(1)
        self._process_events()

        second_item_widget = dialog.profile_list.itemWidget(dialog.profile_list.item(1))
        self.assertIsNotNone(second_item_widget)
        switch = second_item_widget.findChild(QCheckBox, "ModelProfileEnabledSwitch")
        self.assertIsNotNone(switch)

        QTest.mouseClick(switch, Qt.LeftButton)
        self._process_events()

        self.assertEqual(dialog._current_row(), 1)
        self.assertEqual(dialog.name_edit.text(), "gemini-1-5-flash")
        self.assertIn("gemini-1-5-flash", dialog.form_hint.text())

    def test_all_disabled_profiles_hide_model_picker_and_show_state(self):
        payload = self._snapshot_payload()
        for profile in payload["model_profiles"]["profiles"]:
            profile["enabled"] = False
        payload["model_profiles"]["active_profile"] = None

        self.window._handle_initialized(payload)
        self.window._set_input_enabled(True)

        self.assertTrue(self.window.model_chip.isHidden())
        self.assertFalse(self.window.no_models_label.isHidden())
        self.assertEqual(self.window.no_models_label.text(), "All models disabled")
        self.assertFalse(self.window.send_button.isEnabled())

    def test_model_settings_dialog_disables_base_url_for_gemini(self):
        payload = {
            "active_profile": "gemini-1-5-flash",
            "profiles": [
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-demo",
                    "base_url": "https://should-not-be-used.example",
                }
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertFalse(dialog.base_url_edit.isEnabled())
        self.assertIn("Not used for gemini", dialog.base_url_edit.placeholderText())

    def test_model_settings_dialog_uses_improved_layout_and_save_state(self):
        dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertEqual(dialog.objectName(), "ModelSettingsDialog")
        self.assertEqual(dialog.windowTitle(), "Settings")
        self.assertEqual(dialog.tabs.count(), 2)
        self.assertEqual(dialog.tabs.tabText(0), "Models")
        self.assertEqual(dialog.tabs.tabText(1), "Test")
        self.assertIs(dialog.tabs.widget(0), dialog.models_page)
        self.assertIs(dialog.tabs.widget(1), dialog.test_page)
        self.assertIsNotNone(dialog.test_page.layout())
        self.assertIsNotNone(dialog.session_size_slider)
        self.assertGreaterEqual(dialog.session_size_slider.minimum() * dialog.SESSION_SIZE_STEP, 10_000)
        self.assertLessEqual(dialog.session_size_slider.maximum() * dialog.SESSION_SIZE_STEP, 256_000)
        self.assertLessEqual(dialog.width(), QApplication.primaryScreen().availableGeometry().width())
        self.assertEqual(dialog.body_splitter.widget(0).minimumWidth(), 340)
        self.assertEqual(dialog.body_splitter.widget(1).minimumWidth(), 440)
        self.assertEqual(dialog.body_splitter.widget(0).sizePolicy().horizontalStretch(), 3)
        self.assertEqual(dialog.body_splitter.widget(1).sizePolicy().horizontalStretch(), 4)
        self.assertIsNotNone(dialog.save_button)
        self.assertFalse(dialog.save_button.isEnabled())
        self.assertIn("Add a profile", dialog.form_hint.text())
        self.assertFalse(dialog.api_key_rotation_section.content_container.isVisible())

        dialog._add_profile()
        self._process_events()
        self.assertTrue(dialog.save_button.isEnabled())
        selected_item = dialog.profile_list.itemWidget(dialog.profile_list.item(dialog._current_row()))
        self.assertIsNotNone(selected_item)
        self.assertTrue(bool(selected_item.property("selectedProfile")))
        self.assertEqual(dialog.selected_profile_title.text(), "(unnamed)")

    def test_model_settings_dialog_filters_model_popup_without_changing_selection(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "https://api.openai.com/v1",
                    "enabled": True,
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
            self.addCleanup(dialog.close)
            dialog.show()
            self._process_events()

            dialog._apply_loaded_models(
                [
                    ModelEntry("gpt-4o", "openai", True),
                    ModelEntry("gpt-4o-mini", "openai", True),
                    ModelEntry("claude-3-5-sonnet", "openai", False),
                ]
            )
            self._process_events()

            self.assertEqual(dialog.model_combo.currentText(), "gpt-4o")

            dialog._show_model_popup()
            self._process_events()
            self.assertIsNotNone(dialog._model_popup)
            self.assertIsNotNone(dialog._model_popup_search)
            self.assertIsNotNone(dialog._model_popup_list)

            dialog._model_popup_search.setText("mini")
            self._process_events()

            self.assertEqual(dialog.model_combo.currentText(), "gpt-4o")
            hidden_by_text = {
                dialog._model_popup_list.item(row).text(): dialog._model_popup_list.item(row).isHidden()
                for row in range(dialog._model_popup_list.count())
            }
            self.assertTrue(hidden_by_text["gpt-4o"])
            self.assertFalse(hidden_by_text["gpt-4o-mini"])
            self.assertTrue(hidden_by_text["claude-3-5-sonnet"])

    def test_model_settings_dialog_profile_list_shows_provider_and_model(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "",
                }
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertEqual(dialog.profile_list.count(), 1)
        item_widget = dialog.profile_list.itemWidget(dialog.profile_list.item(0))
        self.assertIsNotNone(item_widget)
        labels = item_widget.findChildren(QLabel)
        combined = "\n".join(label.text() for label in labels)
        self.assertIn("gpt-4o", combined)
        self.assertIn("openai", combined)
        # Long model names must wrap instead of being clipped by the narrow
        # Profiles column, so the full model identifier stays readable.
        meta_label = next((label for label in labels if label.objectName() == "ModelProfileItemMeta"), None)
        self.assertIsNotNone(meta_label)
        self.assertTrue(meta_label.wordWrap())

    def test_model_settings_dialog_long_model_name_wraps_in_profile_list(self):
        payload = {
            "active_profile": "long-model",
            "profiles": [
                {
                    "id": "long-model",
                    "provider": "openai",
                    "model": "openai/gpt-oss-120b-very-long-model-identifier-that-would-not-fit-in-a-narrow-column",
                    "api_key": "sk-demo",
                    "base_url": "",
                }
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        item_widget = dialog.profile_list.itemWidget(dialog.profile_list.item(0))
        self.assertIsNotNone(item_widget)
        meta_label = item_widget.findChild(QLabel, "ModelProfileItemMeta")
        self.assertIsNotNone(meta_label)
        self.assertTrue(meta_label.wordWrap())
        self.assertEqual(
            meta_label.text(),
            "openai/gpt-oss-120b-very-long-model-identifier-that-would-not-fit-in-a-narrow-column",
        )

    def test_model_settings_dialog_wrapped_model_name_grows_card_height(self):
        # Regression: a wrapped model name must grow the card height instead of
        # overlapping the profile title line inside a fixed-height card.
        long_name = "mdlscp/qwen3-8-27b-very-long-model-identifier-that-wraps-to-multiple-lines-in-a-narrow-profiles-column"
        payload = {
            "active_profile": "long-model",
            "profiles": [
                {"id": "long-model", "provider": "openai", "model": long_name, "api_key": "sk-demo", "base_url": ""},
                {"id": "short-model", "provider": "openai", "model": "gpt-4o", "api_key": "sk-demo", "base_url": ""},
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        dialog.show()
        self._process_events()

        long_item = dialog.profile_list.item(0)
        short_item = dialog.profile_list.item(1)
        self.assertIsNotNone(long_item)
        self.assertIsNotNone(short_item)

        long_meta = dialog.profile_list.itemWidget(long_item).findChild(QLabel, "ModelProfileItemMeta")
        short_meta = dialog.profile_list.itemWidget(short_item).findChild(QLabel, "ModelProfileItemMeta")
        long_wrapped_height = long_meta.heightForWidth(long_meta.width())
        short_wrapped_height = short_meta.heightForWidth(short_meta.width())
        self.assertGreater(long_wrapped_height, short_wrapped_height)

        # The wrapped card must be taller than the single-line card, and the
        # height must be stable across repeated relayouts (no accumulation).
        long_height = long_item.sizeHint().height()
        short_height = short_item.sizeHint().height()
        self.assertGreater(long_height, short_height)
        dialog.profile_list._fit_items_to_viewport()
        self._process_events()
        self.assertEqual(long_item.sizeHint().height(), long_height)

    def test_model_settings_dialog_enables_base_url_for_openai(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "https://api.openai.com/v1",
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
            self.addCleanup(dialog.close)
            self._process_events()

            self.assertTrue(dialog.base_url_edit.isEnabled())
            self.assertIn("api.openai.com", dialog.base_url_edit.placeholderText())

    def test_model_settings_dialog_clears_base_url_for_gemini_on_save(self):
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
            self.addCleanup(dialog.close)
            dialog._add_profile()
            self._process_events()
            dialog.provider_combo.setCurrentText("gemini")
            dialog.model_edit.setText("gemini-1.5-flash")
            dialog.api_key_edit.setText("gm-demo")
            dialog.base_url_edit.setText("https://ignored.example")
            self._process_events()

            dialog._save_and_accept()
            result = dialog.result_payload()
            self.assertEqual(result["profiles"][0]["provider"], "gemini")
            self.assertEqual(result["profiles"][0]["base_url"], "")

    def test_model_settings_dialog_openai_loaded_state_shows_popup_and_reload_buttons(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "https://api.openai.com/v1",
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        dialog.show()
        self._process_events()

        dialog._invalidate_pending_fetches()
        dialog._apply_loaded_models(
            [
                ModelEntry(id="gpt-4o", family="", supports_image_input=False),
                ModelEntry(id="gpt-4.1", family="", supports_image_input=False),
            ]
        )
        self._process_events()

        self.assertTrue(dialog.model_combo.isVisible())
        self.assertTrue(dialog.model_popup_button.isVisible())
        self.assertTrue(dialog.model_popup_button.isEnabled())
        self.assertTrue(dialog.model_reload_button.isVisible())
        self.assertTrue(dialog.model_reload_button.isEnabled())

    def test_model_settings_dialog_openai_popup_button_opens_model_list(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "https://api.openai.com/v1",
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        dialog.show()
        self._process_events()

        dialog._invalidate_pending_fetches()
        dialog._apply_loaded_models(
            [
                ModelEntry(id="gpt-4o", family="", supports_image_input=False),
                ModelEntry(id="gpt-4.1", family="", supports_image_input=False),
            ]
        )
        self._process_events()

        dialog._show_model_popup()
        self._process_events()

        self.assertIsNotNone(dialog._model_popup)
        self.assertTrue(dialog._model_popup.isVisible())
        self.assertIsNotNone(dialog._model_popup_search)
        self.assertIsNotNone(dialog._model_popup_list)
        self.assertEqual(dialog._model_popup_list.count(), 2)
        dialog._close_model_popup()

    def test_model_settings_dialog_gemini_loaded_state_shows_popup_and_reload_buttons(self):
        payload = {
            "active_profile": "gemini-2.5-pro",
            "profiles": [
                {
                    "id": "gemini-2.5-pro",
                    "provider": "gemini",
                    "model": "gemini-2.5-pro",
                    "api_key": "gm-demo",
                    "base_url": "",
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        dialog.show()
        self._process_events()

        dialog._invalidate_pending_fetches()
        dialog._apply_loaded_models(
            [
                ModelEntry(id="gemini-2.5-pro", family="gemini", supports_image_input=True),
                ModelEntry(id="gemini-2.5-flash", family="gemini", supports_image_input=True),
            ]
        )
        self._process_events()

        self.assertTrue(dialog.model_combo.isVisible())
        self.assertTrue(dialog.model_popup_button.isVisible())
        self.assertTrue(dialog.model_popup_button.isEnabled())
        self.assertTrue(dialog.model_reload_button.isVisible())
        self.assertTrue(dialog.model_reload_button.isEnabled())

    def test_model_settings_dialog_gemini_popup_button_opens_model_list(self):
        payload = {
            "active_profile": "gemini-2.5-pro",
            "profiles": [
                {
                    "id": "gemini-2.5-pro",
                    "provider": "gemini",
                    "model": "gemini-2.5-pro",
                    "api_key": "gm-demo",
                    "base_url": "",
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        dialog.show()
        self._process_events()

        dialog._invalidate_pending_fetches()
        dialog._apply_loaded_models(
            [
                ModelEntry(id="gemini-2.5-pro", family="gemini", supports_image_input=True),
                ModelEntry(id="gemini-2.5-flash", family="gemini", supports_image_input=True),
            ]
        )
        self._process_events()

        dialog._show_model_popup()
        self._process_events()

        self.assertIsNotNone(dialog._model_popup)
        self.assertTrue(dialog._model_popup.isVisible())
        self.assertIsNotNone(dialog._model_popup_search)
        self.assertIsNotNone(dialog._model_popup_list)
        self.assertEqual(dialog._model_popup_list.count(), 2)
        dialog._close_model_popup()

    def test_model_settings_dialog_enables_base_url_for_anthropic(self):
        payload = {
            "active_profile": "claude-sonnet",
            "profiles": [
                {
                    "id": "claude-sonnet",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5-20250929",
                    "api_key": "sk-ant-demo",
                    "base_url": "https://api.anthropic.com",
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
            self.addCleanup(dialog.close)
            self._process_events()

            self.assertTrue(dialog.base_url_edit.isEnabled())
            self.assertIn("anthropic", dialog.base_url_edit.placeholderText().lower())

    def test_model_settings_dialog_anthropic_fetch_inputs_use_anthropic_fetcher(self):
        from core.model_fetcher import AnthropicModelFetcher

        payload = {
            "active_profile": "claude-sonnet",
            "profiles": [
                {
                    "id": "claude-sonnet",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5-20250929",
                    "api_key": "sk-ant-demo",
                    "base_url": "https://api.anthropic.com",
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
            self.addCleanup(dialog.close)
            self._process_events()

            request = dialog._current_fetch_inputs()
            self.assertIsNotNone(request)
            provider, _api_key, base_url, fetcher, _cache_key = request
            self.assertEqual(provider, "anthropic")
            self.assertEqual(base_url, "https://api.anthropic.com")
            self.assertIsInstance(fetcher, AnthropicModelFetcher)

    def test_model_settings_dialog_anthropic_persists_base_url_on_save(self):
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
            self.addCleanup(dialog.close)
            dialog._add_profile()
            self._process_events()
            dialog.provider_combo.setCurrentText("anthropic")
            dialog.model_edit.setText("claude-sonnet-4-5-20250929")
            dialog.api_key_edit.setText("sk-ant-demo")
            dialog.base_url_edit.setText("https://api.anthropic.com")
            self._process_events()

            dialog._save_and_accept()
            result = dialog.result_payload()
            self.assertEqual(result["profiles"][0]["provider"], "anthropic")
            self.assertEqual(result["profiles"][0]["base_url"], "https://api.anthropic.com")

    def test_model_settings_dialog_anthropic_loaded_state_shows_popup_and_reload_buttons(self):
        payload = {
            "active_profile": "claude-sonnet",
            "profiles": [
                {
                    "id": "claude-sonnet",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5-20250929",
                    "api_key": "sk-ant-demo",
                    "base_url": "https://api.anthropic.com",
                }
            ],
        }
        with mock.patch.object(agent_cli.ModelSettingsDialog, "_schedule_fetch", autospec=True):
            dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        dialog.show()
        self._process_events()

        dialog._invalidate_pending_fetches()
        dialog._apply_loaded_models(
            [
                ModelEntry(id="claude-sonnet-4-5-20250929", family="", supports_image_input=True),
                ModelEntry(id="claude-opus-4-20250514", family="", supports_image_input=True),
            ]
        )
        self._process_events()

        self.assertTrue(dialog.model_combo.isVisible())
        self.assertTrue(dialog.model_popup_button.isVisible())
        self.assertTrue(dialog.model_popup_button.isEnabled())
        self.assertTrue(dialog.model_reload_button.isVisible())
        self.assertTrue(dialog.model_reload_button.isEnabled())

    def test_composer_expands_to_max_height_then_uses_internal_scroll(self):
        self.window._handle_initialized(self._snapshot_payload())
        long_text = "\n".join(f"line-{idx}" for idx in range(80))
        self.window.composer.setPlainText(long_text)
        self.window._update_composer_height()
        self._process_events()

        self.assertEqual(self.window.composer.height(), self.window.composer.maximumHeight())
        self.assertGreater(self.window.composer.verticalScrollBar().maximum(), 0)

    def test_composer_bottom_controls_match_new_layout(self):
        self.assertTrue(hasattr(self.window, "model_chip"))
        self.assertTrue(hasattr(self.window, "reasoning_chip"))
        self.assertFalse(hasattr(self.window, "voice_button"))
        self.assertEqual(self.window.stop_action_button.objectName(), "ComposerStopButton")
        self.assertEqual(self.window.model_chip.accessibleName(), "Model selector")
        self.assertEqual(self.window.reasoning_chip.accessibleName(), "Reasoning level selector")

    def test_primary_widgets_expose_accessible_names(self):
        self.assertFalse(hasattr(self.window.sidebar, "search_field"))
        self.assertEqual(self.window.sidebar.list_view.accessibleName(), "Chat list")
        self.assertEqual(self.window.transcript.accessibleName(), "Conversation transcript")
        self.assertEqual(self.window.composer.accessibleName(), "Composer")
        self.assertEqual(self.window.send_button.accessibleName(), "Send request")
        self.assertEqual(self.window.inspector_panel.tabs.accessibleName(), "Inspector tabs")


if __name__ == "__main__":
    unittest.main()
