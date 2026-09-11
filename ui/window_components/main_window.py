from __future__ import annotations

import logging
import os
import sys
from typing import Final

from PySide6.QtCore import QEvent, QMessageLogContext, Qt, QtMsgType, QSize, QTimer, qInstallMessageHandler
from PySide6.QtGui import QAction, QCloseEvent, QIcon
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMenuBar, QMessageBox, QSizePolicy, QVBoxLayout, QWidget

from core.constants import AGENT_VERSION
from core.input_sanitizer import build_user_input_notice, sanitize_user_text
from core.multimodal import DEFAULT_MODEL_CAPABILITIES, can_read_image_file, resolve_model_capabilities
from core.model_profiles import normalize_profiles_payload
from core.reasoning_controls import normalize_profile_reasoning, reasoning_options_for_profile
from core.text_utils import format_compact_tokens, prepare_markdown_for_render
from ui.main_window_state import ComposerStateController, RunStatusController, StreamEventRouter
from ui.runtime import AgentRuntimeController
from ui.theme import ACCENT_BLUE, build_stylesheet
from ui.widgets.window_chrome import install_dialog_chrome
from ui.widgets import ModelSettingsDialog, _fa_icon
from ui.window_components.inspector_controller import InspectorController
from ui.window_components.menu_builder import MenuBuilder
from ui.window_components.sidebar_controller import SidebarController
from ui.window_components.status_bar_manager import StatusBarManager
from ui.window_components.workspace_builder import WorkspaceBuilder

logger = logging.getLogger("agent")
APP_DISPLAY_NAME: Final[str] = "KGB|Agent"
WINDOWS_APP_USER_MODEL_ID: Final[str] = "simple_ai_agent.ai_agent"
ASSISTANT_DELTA_FLUSH_MS: Final[int] = 32


def _qt_message_filter(mode: QtMsgType, ctx: QMessageLogContext, message: str) -> None:
    """Suppress benign QFileSystemWatcher warnings (e.g. when a watched directory is deleted)."""
    if mode == QtMsgType.QtWarningMsg and "QFileSystemWatcher" in message:
        return
    # Fall back to default handler for everything else
    from PySide6.QtCore import qFormatLogMessage

    safe_ctx = ctx if isinstance(ctx, QMessageLogContext) else QMessageLogContext()
    formatted = qFormatLogMessage(mode, safe_ctx, message)
    if mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
        print(formatted, file=sys.stderr, flush=True)
    else:
        print(formatted, flush=True)


def _configure_qt_logging() -> None:
    rule = "qt.text.font.db=false"
    current = str(os.environ.get("QT_LOGGING_RULES") or "").strip()
    if rule not in {item.strip() for item in current.split(";") if item.strip()}:
        os.environ["QT_LOGGING_RULES"] = f"{current};{rule}" if current else rule
    qInstallMessageHandler(_qt_message_filter)


def _configure_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_USER_MODEL_ID)
    except Exception:
        logger.debug("Failed to set Windows AppUserModelID.", exc_info=True)


def _build_app_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pixmap = _fa_icon("fa5s.robot", color=ACCENT_BLUE, size=size).pixmap(size, size)
        if not pixmap.isNull():
            icon.addPixmap(pixmap)
    return icon if not icon.isNull() else _fa_icon("fa5s.robot", color=ACCENT_BLUE, size=32)


class MainWindow(QMainWindow):
    def __init__(self, controller: AgentRuntimeController | None = None, *, auto_initialize: bool = True) -> None:
        super().__init__()
        self.controller = controller or AgentRuntimeController(self)
        self.current_turn = None
        self.current_snapshot: dict | None = None
        self.active_session_id = ""
        self._model_settings_window = None
        self._settings_open = False
        self._input_enabled = False
        self.awaiting_approval = False
        self.awaiting_user_choice = False
        self.is_busy = False
        self.sidebar_collapsed = False
        self._sidebar_width = 300
        self.inspector_collapsed = False
        self._inspector_width = 400
        self._custom_choice_armed = False
        self._tools_hash: int = 0
        self._summarize_in_progress = False
        self.model_profiles_payload: dict = {"active_profile": None, "profiles": []}
        self.model_capabilities: dict = dict(DEFAULT_MODEL_CAPABILITIES)
        self.draft_image_attachments: list[dict] = []
        self._has_active_model = False
        self._composer_min_height = 56
        self._composer_max_height = 56
        self._composer_height_padding = 16
        self._composer_growth_lines = 5
        self._composer_height_sync_pending = False
        self._session_cache_hit_tokens = 0
        self._run_start_time: float | None = None
        self._current_status_label = ""
        self._current_status_phase = "working"
        self._last_rendered_elapsed_text = ""
        self._last_assistant_delta_sequence = 0
        self._last_assistant_delta_text = ""
        self._pending_assistant_delta_payload: dict | None = None
        self._event_router: StreamEventRouter | None = None
        self._assistant_delta_flush_timer = QTimer(self)
        self._assistant_delta_flush_timer.setSingleShot(True)
        self._assistant_delta_flush_timer.setInterval(ASSISTANT_DELTA_FLUSH_MS)
        self._assistant_delta_flush_timer.timeout.connect(self._flush_pending_assistant_delta)

        self._menu_builder = MenuBuilder(self)
        self._workspace_builder = WorkspaceBuilder(self)
        self._status_bar_manager = StatusBarManager(self)
        self._sidebar_controller = SidebarController(self)
        self._inspector_controller = InspectorController(self)
        self._realtime_timer = self._status_bar_manager.realtime_timer
        self._composer_state = ComposerStateController(self)
        self._status_controller = RunStatusController(self)

        self._build_ui()
        self._connect_signals()
        self._build_event_dispatch()
        # Track the center panel's final geometry to keep the chat centered.
        self.transcript.parentWidget().installEventFilter(self)
        if auto_initialize:
            self.controller.initialize()

    def _update_realtime_elapsed(self) -> None:
        self._status_controller.update_realtime_elapsed()

    def _build_ui(self) -> None:
        self.setWindowTitle(f"KGB|Agent {AGENT_VERSION}")
        self.resize(1300, 670)
        self.setMinimumSize(900, 600)
        if app := QApplication.instance():
            self.setWindowIcon(app.windowIcon())
        else:
            self.setWindowIcon(_build_app_icon())

        status_refs = self._status_bar_manager.build()
        self.runtime_meta_label = status_refs.runtime_meta_label
        self.setStatusBar(status_refs.status_bar)

        central = QWidget()
        central.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._build_menu_bar()
        self.workspace = self._build_workspace()
        root.addWidget(self.workspace, 1)

        self.setCentralWidget(central)
        install_dialog_chrome(QApplication.instance())
        self.setStyleSheet(build_stylesheet())
        self.inspector_container.hide()
        self.inspector_collapsed = True
        self._set_input_enabled(False)

    def _build_menu_bar(self) -> None:
        refs = self._menu_builder.build()
        self.toggle_sidebar_action = refs.toggle_sidebar_action
        self.new_session_action = refs.new_session_action
        self.settings_action = refs.settings_action
        self.info_action = refs.info_action
        self.quit_action = refs.quit_action
        self.status_text = refs.status_text
        self.status_line_label = refs.status_text
        self.status_meta = refs.status_meta
        self.top_status_chip = refs.top_status_chip
        self.new_project_button = refs.new_project_button
        self.sidebar_toggle_button = refs.sidebar_toggle_button
        self.new_session_button = refs.new_session_button
        self.settings_button = refs.settings_button
        self.info_button = refs.info_button
        self.setMenuWidget(refs.menu_widget)
        self._set_status_visual("Initializing runtime…", busy=True)

    def _build_workspace(self) -> QWidget:
        refs = self._workspace_builder.build()
        self.splitter = refs.splitter
        self.sidebar_container = refs.sidebar_container
        self.sidebar = refs.sidebar
        self.transcript = refs.transcript
        self.composer_shell = refs.composer_shell
        self.composer_container = refs.composer_container
        self.approval_card = refs.approval_card
        self.user_choice_card = refs.user_choice_card
        self.composer_pill = refs.composer_pill
        self.composer_attachments_strip = refs.composer_attachments_strip
        self.composer_notice_label = refs.composer_notice_label
        self.composer = refs.composer
        self.attach_button = refs.attach_button
        self.attach_menu = refs.attach_menu
        self.add_image_action = refs.add_image_action
        self.insert_file_path_action = refs.insert_file_path_action
        self.model_chip = refs.model_chip
        self.model_chip_menu = refs.model_chip_menu
        self.model_chip_group = refs.model_chip_group
        self.reasoning_chip = refs.reasoning_chip
        self.reasoning_chip_menu = refs.reasoning_chip_menu
        self.reasoning_chip_group = refs.reasoning_chip_group
        self.model_image_badge = refs.model_image_badge
        self.no_models_label = refs.no_models_label
        self.open_settings_inline_button = refs.open_settings_inline_button
        self.cache_hit_label = refs.cache_hit_label
        self.summary_progress_ring = refs.summary_progress_ring
        self.send_button = refs.send_button
        self.stop_action_button = refs.stop_action_button
        self.inspector_container = refs.inspector_container
        self.inspector_panel = refs.inspector_panel
        self.overview_panel = refs.overview_panel
        self.tools_panel = refs.tools_panel
        self.help_text = refs.help_text
        return refs.workspace

    def _connect_signals(self) -> None:
        self.toggle_sidebar_action.triggered.connect(lambda _checked=False: self._toggle_sidebar())
        self.new_session_action.triggered.connect(lambda _checked=False: self._new_session())
        self.settings_action.triggered.connect(lambda _checked=False: self._open_settings_dialog())
        self.info_action.triggered.connect(lambda _checked=False: self._toggle_info_popup())
        self.open_settings_inline_button.clicked.connect(lambda _checked=False: self._open_settings_dialog())
        self.quit_action.triggered.connect(lambda _checked=False: self.close())
        self.stop_action_button.clicked.connect(lambda _checked=False: self.controller.stop_run())
        self.new_project_button.clicked.connect(lambda _checked=False: self._open_new_project())
        self.sidebar_toggle_button.clicked.connect(lambda _checked=False: self.toggle_sidebar_action.trigger())
        self.new_session_button.clicked.connect(lambda _checked=False: self.new_session_action.trigger())
        self.settings_button.clicked.connect(lambda _checked=False: self.settings_action.trigger())
        self.info_button.clicked.connect(lambda _checked=False: self.info_action.trigger())
        self.splitter.splitterMoved.connect(lambda _pos, _index: self._update_centering())
        self.sidebar.session_activated.connect(self._switch_session)
        self.sidebar.session_delete_requested.connect(self._request_delete_session)
        self.sidebar.project_delete_requested.connect(self._request_delete_project)
        self.tools_panel.availability_changed.connect(self._handle_tool_availability_changed)

        self.send_button.clicked.connect(self._submit_request)
        self.add_image_action.triggered.connect(self._attach_images)
        self.insert_file_path_action.triggered.connect(self._insert_file_paths)
        self.model_chip_menu.triggered.connect(self._on_model_action_triggered)
        self.reasoning_chip_menu.triggered.connect(self._on_reasoning_action_triggered)
        self.composer.submit_requested.connect(self._submit_request)
        self.composer.image_pasted.connect(self._handle_pasted_image)
        self.composer.image_files_pasted.connect(self._handle_pasted_image_files)
        self.composer.textChanged.connect(self._queue_composer_height_sync)
        self.composer.textChanged.connect(self._refresh_submit_controls)
        self.composer.document().documentLayout().documentSizeChanged.connect(self._queue_composer_height_sync)
        self.composer_attachments_strip.attachment_remove_requested.connect(self._remove_draft_attachment)
        self.approval_card.decision_made.connect(self._handle_inline_approval_decision)
        self.user_choice_card.option_selected.connect(self._handle_user_choice_selected)
        self.user_choice_card.custom_option_requested.connect(self._handle_custom_choice_requested)
        self.controller.initialized.connect(self._handle_initialized)
        self.controller.initialization_failed.connect(self._handle_init_failed)
        self.controller.event_emitted.connect(self._handle_event)
        self.controller.approval_requested.connect(self._handle_approval_request)
        self.controller.user_choice_requested.connect(self._handle_user_choice_request)
        self.controller.session_changed.connect(self._handle_session_changed)
        self.controller.busy_changed.connect(self._handle_busy_changed)

    def _build_event_dispatch(self) -> None:
        self._event_router = StreamEventRouter(
            {
                "run_started": self._on_run_started,
                "status_changed": self._on_status_changed,
                "assistant_delta": self._on_assistant_delta,
                "assistant_commit": self._on_assistant_commit,
                "assistant_boundary": self._on_assistant_boundary,
                "tool_started": self._on_tool_started,
                "cli_output": self._on_cli_output,
                "tool_finished": self._on_tool_finished,
                "tool_args_missing": self._on_tool_args_missing,
                "summary_progress": self._on_summary_progress,
                "summary_notice": self._on_summary_notice,
                "cache_hit": self._on_cache_hit,
                "approval_resolved": self._on_approval_resolved,
                "run_finished": self._on_run_finished,
                "run_failed": self._on_run_failed,
                "chat_reset": self._on_chat_reset,
            }
        )

    def _queue_composer_height_sync(self, *_args) -> None:
        self._composer_state.queue_height_sync(*_args)

    def _flush_composer_height_sync(self) -> None:
        self._composer_state.flush_height_sync()

    def _update_composer_height(self, *_args) -> None:
        self._composer_state.update_height(*_args)

    def _active_model_supports_images(self) -> bool:
        capabilities = resolve_model_capabilities(self._active_model_profile(), self.model_capabilities)
        return bool(capabilities.get("image_input_supported"))

    def _composer_has_request_content(self) -> bool:
        return self._composer_state.composer_has_request_content()

    def _request_blocked_by_image_capability(self) -> bool:
        return self._composer_state.request_blocked_by_image_capability()

    def _show_composer_notice(self, message: str, *, level: str = "warning") -> None:
        self._composer_state.show_composer_notice(message, level=level)

    def _clear_composer_notice(self) -> None:
        self._composer_state.clear_composer_notice()

    def _clear_draft_image_attachments(self) -> None:
        self._composer_state.clear_draft_image_attachments()

    def _remove_draft_attachment(self, attachment_id: str) -> None:
        self._composer_state.remove_draft_attachment(attachment_id)

    def _import_image_files(self, file_paths: list[str]) -> None:
        self._composer_state.import_image_files(file_paths)

    def _handle_pasted_image(self, image: object) -> None:
        self._composer_state.handle_pasted_image(image)

    def _handle_pasted_image_files(self, file_paths: object) -> None:
        self._composer_state.handle_pasted_image_files(file_paths)

    def _refresh_submit_controls(self, *_args) -> None:
        self._set_input_enabled(True)

    def _update_send_button_visual(self, can_send: bool) -> None:
        icon_color = "#08090B" if can_send else "#8A857E"
        self.send_button.setIcon(_fa_icon("fa5s.arrow-up", color=icon_color, size=14))

    def _on_run_started(self, payload: dict) -> None:
        if timer := getattr(self, "_assistant_delta_flush_timer", None):
            timer.stop()
        self._pending_assistant_delta_payload = None
        self._last_assistant_delta_sequence = 0
        self._last_assistant_delta_text = ""
        self._status_controller.on_run_started(payload)

    def _on_status_changed(self, payload: dict) -> None:
        self._status_controller.on_status_changed(payload)

    def _on_assistant_delta(self, payload: dict) -> None:
        if self.current_turn is None:
            return
        full_text = str(payload.get("full_text", "") or "")
        sequence = payload.get("sequence")
        sequence_reset = False
        if isinstance(sequence, int):
            if sequence <= self._last_assistant_delta_sequence:
                if not self._assistant_delta_sequence_reset_is_new_content(full_text):
                    logger.debug(
                        "Stream ignored stale main_window_assistant_delta sequence=%s last_sequence=%s full_len=%s",
                        sequence,
                        self._last_assistant_delta_sequence,
                        len(full_text),
                    )
                    return
                sequence_reset = True
                logger.debug(
                    "Stream accepted reset main_window_assistant_delta sequence=%s previous_sequence=%s full_len=%s",
                    sequence,
                    self._last_assistant_delta_sequence,
                    len(full_text),
                )
            self._last_assistant_delta_sequence = sequence
        elif self._is_stale_assistant_delta_without_sequence(full_text):
            logger.debug(
                "Stream ignored stale main_window_assistant_delta without sequence full_len=%s last_len=%s",
                len(full_text),
                len(self._last_assistant_delta_text),
            )
            return

        self._last_assistant_delta_text = full_text
        self._queue_assistant_delta({"full_text": full_text}, force=sequence_reset)

    def _on_assistant_commit(self, payload: dict) -> None:
        """Render the terminal stream state before run_finished changes layout."""
        if self.current_turn is None:
            return
        full_text = str(payload.get("full_text", "") or "")
        if not full_text:
            return
        self._last_assistant_delta_text = full_text
        self._queue_assistant_delta({"full_text": full_text}, force=True)

    def _on_assistant_boundary(self, _payload: dict) -> None:
        self._flush_pending_assistant_delta()
        self._last_assistant_delta_text = ""
        if self.current_turn is not None:
            self.current_turn.begin_assistant_block()
            self.transcript.notify_content_changed()

    def _assistant_delta_sequence_reset_is_new_content(self, full_text: str) -> bool:
        previous = self._last_assistant_delta_text
        if previous and full_text.startswith(previous) and len(full_text) > len(previous):
            return True
        try:
            kinds = self.current_turn.block_kinds() if self.current_turn is not None else []
        except Exception:
            kinds = []
        return bool(full_text.strip()) and bool(kinds) and kinds[-1] != "assistant"

    def _queue_assistant_delta(self, payload: dict, *, force: bool = False) -> None:
        self._pending_assistant_delta_payload = dict(payload)
        if force or self._should_flush_assistant_delta_immediately():
            self._flush_pending_assistant_delta()
            return
        timer = getattr(self, "_assistant_delta_flush_timer", None)
        if timer is not None and not timer.isActive():
            timer.start()

    def _should_flush_assistant_delta_immediately(self) -> bool:
        if self.current_turn is None:
            return True
        kinds = self.current_turn.block_kinds()
        return not kinds or kinds[-1] != "assistant"

    def _flush_pending_assistant_delta(self) -> None:
        pending_payload = getattr(self, "_pending_assistant_delta_payload", None)
        if self.current_turn is None or not pending_payload:
            self._pending_assistant_delta_payload = None
            if timer := getattr(self, "_assistant_delta_flush_timer", None):
                timer.stop()
            return
        payload = pending_payload
        self._pending_assistant_delta_payload = None
        if timer := getattr(self, "_assistant_delta_flush_timer", None):
            timer.stop()
        full_text = str(payload.get("full_text", "") or "")
        rendered_markdown = prepare_markdown_for_render(full_text) if full_text else ""
        logger.debug(
            "Stream main_window_assistant_delta full_len=%s rendered_len=%s",
            len(full_text),
            len(rendered_markdown),
        )
        self.current_turn.set_assistant_markdown(
            rendered_markdown,
        )
        self.current_turn.set_assistant_streaming(True)
        self.transcript.notify_content_changed()

    def _on_tool_started(self, payload: dict) -> None:
        self._flush_pending_assistant_delta()
        if self.current_turn is None:
            return
        self.current_turn.set_assistant_streaming(False)
        self.current_turn.start_tool(payload)
        self.transcript.notify_content_changed()

    def _on_tool_finished(self, payload: dict) -> None:
        self._flush_pending_assistant_delta()
        if self.current_turn is None:
            return
        self.current_turn.set_assistant_streaming(False)
        self.current_turn.finish_tool(payload)
        self.transcript.notify_content_changed()

    def _on_cli_output(self, payload: dict) -> None:
        if self.current_turn is None:
            return
        self.current_turn.append_tool_output(payload)
        self.transcript.notify_content_changed()

    def _on_summary_notice(self, payload: dict) -> None:
        self._flush_pending_assistant_delta()
        kind = str(payload.get("kind", "") or "")
        level = str(payload.get("level", "info") or "info")
        message = str(payload.get("message", "") or "").strip()
        if kind == "auto_summary":
            self._summarize_in_progress = False
            if message:
                self._current_status_label = message
                self._current_status_phase = "success"
                self._last_rendered_elapsed_text = ""
                if self.current_turn is not None:
                    self.current_turn.set_status(message, phase="success")
                    self.transcript.notify_content_changed()
                else:
                    self.transcript.add_global_notice(message, level=level)
            return
        if level == "error":
            if self.current_turn is not None:
                self.current_turn.add_notice(message, level=level)
                self.transcript.notify_content_changed()
            else:
                self.transcript.add_global_notice(message, level=level)
            return
        if message:
            self._show_transient_status_message(message)

    def _on_tool_args_missing(self, payload: dict) -> None:
        _ = payload
        return

    def _on_summary_progress(self, payload: dict) -> None:
        self.summary_progress_ring.set_summary_progress(payload)

    def _set_session_cache_hit_tokens(self, tokens: int) -> None:
        self._session_cache_hit_tokens = max(0, int(tokens or 0))
        self.cache_hit_label.setText(f"Cache Hit: {format_compact_tokens(self._session_cache_hit_tokens)}")

    def _on_cache_hit(self, payload: dict) -> None:
        try:
            delta = max(0, int(payload.get("tokens", 0) or 0))
        except (TypeError, ValueError):
            return
        if delta:
            self._set_session_cache_hit_tokens(self._session_cache_hit_tokens + delta)

    def _on_approval_resolved(self, payload: dict) -> None:
        self._flush_pending_assistant_delta()
        auto_resolved = bool(payload.get("auto"))
        if payload.get("approved"):
            if auto_resolved:
                self.transcript.notify_content_changed()
                return
            message = "Protected action approved."
            if payload.get("always"):
                message = "Protected action approved for this and future actions in the current session."
            self._show_transient_status_message(message)
        else:
            self._show_transient_status_message("Protected action denied.")

    def _on_run_finished(self, payload: dict) -> None:
        self._flush_pending_assistant_delta()
        if self.current_turn is not None:
            self.current_turn.set_assistant_streaming(False)
        self._status_controller.on_run_finished(payload)

    def _on_run_failed(self, payload: dict) -> None:
        self._flush_pending_assistant_delta()
        if self.current_turn is not None:
            self.current_turn.set_assistant_streaming(False)
        self._status_controller.on_run_failed(payload)

    def _on_chat_reset(self, _payload: dict) -> None:
        if timer := getattr(self, "_assistant_delta_flush_timer", None):
            timer.stop()
        self._pending_assistant_delta_payload = None
        self._last_assistant_delta_sequence = 0
        self._last_assistant_delta_text = ""
        self._status_controller.on_chat_reset()

    def _is_stale_assistant_delta_without_sequence(self, full_text: str) -> bool:
        previous = self._last_assistant_delta_text
        if not previous or len(full_text) >= len(previous):
            return False
        if previous.startswith(full_text):
            return True
        prefix_len = self._common_prefix_len(previous, full_text)
        return prefix_len >= max(48, int(len(full_text) * 0.8))

    @staticmethod
    def _common_prefix_len(first: str, second: str) -> int:
        limit = min(len(first), len(second))
        index = 0
        while index < limit and first[index] == second[index]:
            index += 1
        return index

    def _show_transient_status_message(self, label: str, timeout_ms: int = 1800) -> None:
        self._status_bar_manager.show_transient_status_message(label, timeout_ms=timeout_ms)

    def _set_status_visual(self, label: str, *, busy: bool = False, success: bool = False, error: bool = False) -> None:
        self._status_bar_manager.set_status_visual(label, busy=busy, success=success, error=error)

    def _toggle_info_popup(self) -> None:
        self._inspector_controller.toggle_info_popup()
        self._update_centering()

    def _set_input_enabled(self, enabled: bool) -> None:
        # Remember the baseline so the input state can be recomputed when the
        # Settings panel opens or closes without the caller passing it again.
        self._input_enabled = bool(enabled)
        has_pending_interrupt = self.awaiting_approval or self.awaiting_user_choice
        # While the Settings panel is open the main window must not accept
        # input: the composer and the sidebar are disabled even though the
        # panel is a separate top-level window that may overlap them.
        settings_open = self._settings_open
        can_edit = (
            enabled
            and not settings_open
            and not self.awaiting_approval
            and not self.is_busy
            and (not self.awaiting_user_choice or self._custom_choice_armed)
        )
        can_send = (
            can_edit
            and self._has_active_model
            and self._composer_has_request_content()
            and not self._request_blocked_by_image_capability()
        )
        attach_enabled = enabled and not settings_open and not has_pending_interrupt and not self.is_busy
        self.composer.setEnabled(can_edit)
        self.send_button.setEnabled(can_send)
        self._update_send_button_visual(can_send)
        self.attach_button.setEnabled(attach_enabled)
        self.add_image_action.setEnabled(attach_enabled and self._has_active_model and self._active_model_supports_images())
        self.insert_file_path_action.setEnabled(attach_enabled)
        self.model_chip.setEnabled(can_edit and self._has_active_model)
        self.reasoning_chip.setEnabled(can_edit and self._has_active_model and not self.reasoning_chip.isHidden())
        model_settings_open = self._model_settings_window_is_visible()
        can_open_model_settings = enabled and not has_pending_interrupt and not self.is_busy
        self.open_settings_inline_button.setEnabled(can_open_model_settings)
        self.settings_button.setEnabled(can_open_model_settings or model_settings_open)
        self.info_button.setEnabled(enabled and not settings_open)
        self.sidebar.setEnabled(enabled and not settings_open and not has_pending_interrupt and not self.is_busy)
        self.approval_card.set_actions_enabled(self.awaiting_approval and not self.is_busy)
        self.user_choice_card.set_actions_enabled(
            enabled and self.awaiting_user_choice and not self.awaiting_approval and not self.is_busy
        )

    def _model_settings_window_is_visible(self) -> bool:
        window = self._model_settings_window
        if window is None:
            return False
        is_hidden = getattr(window, "isHidden", None)
        if callable(is_hidden):
            return not bool(is_hidden())
        is_visible = getattr(window, "isVisible", None)
        if callable(is_visible):
            return bool(is_visible())
        return True

    def _handle_initialized(self, payload: dict) -> None:
        self._apply_runtime_payload(payload, restore_transcript=True)
        self._set_status_visual("Ready", success=True)
        self.status_meta.setText("")
        self._set_input_enabled(True)
        self._handle_checkpoint_notice(payload)

    def _update_env_info(self, snapshot: dict) -> None:
        self._status_bar_manager.update_env_info(snapshot)

    def _active_model_profile(self) -> dict | None:
        payload = self.model_profiles_payload if isinstance(self.model_profiles_payload, dict) else {}
        active_id = str(payload.get("active_profile") or "").strip()
        for profile in payload.get("profiles", []) or []:
            if isinstance(profile, dict) and str(profile.get("id") or "").strip() == active_id:
                return profile
        return None

    def _refresh_model_selector(self) -> None:
        payload = self.model_profiles_payload if isinstance(self.model_profiles_payload, dict) else {}
        profiles = [item for item in payload.get("profiles", []) or [] if isinstance(item, dict)]
        enabled_profiles = [item for item in profiles if bool(item.get("enabled", True))]
        active_id = str(payload.get("active_profile") or "").strip()

        self.model_chip_menu.clear()
        for action in self.model_chip_group.actions():
            self.model_chip_group.removeAction(action)

        if not profiles:
            self.model_chip.setVisible(False)
            self.reasoning_chip.setVisible(False)
            self.no_models_label.setText("No models configured")
            self.no_models_label.setVisible(True)
            self.open_settings_inline_button.setVisible(True)
            self._has_active_model = False
            return

        if not enabled_profiles:
            self.model_chip.setVisible(False)
            self.reasoning_chip.setVisible(False)
            self.no_models_label.setText("All models disabled")
            self.no_models_label.setVisible(True)
            self.open_settings_inline_button.setVisible(True)
            self._has_active_model = False
            self.model_image_badge.setVisible(False)
            self._refresh_submit_controls()
            return

        self.model_chip.setVisible(True)
        self.no_models_label.setVisible(False)
        self.open_settings_inline_button.setVisible(False)

        for profile in enabled_profiles:
            profile_id = str(profile.get("id") or "").strip()
            if not profile_id:
                continue
            action = self.model_chip_menu.addAction(profile_id)
            action.setCheckable(True)
            action.setData(profile_id)
            provider = str(profile.get("provider") or "").strip()
            model = str(profile.get("model") or "").strip()
            action.setToolTip(f"Provider: {provider}\nModel: {model}")
            if profile_id == active_id:
                action.setChecked(True)
                self.model_chip.setText(profile_id)
                self.model_chip.setToolTip(f"Provider: {provider}\nModel: {model}")
            self.model_chip_group.addAction(action)

        self._has_active_model = any(str(item.get("id") or "").strip() == active_id for item in enabled_profiles)
        if not self._has_active_model:
            self.model_chip.setText("No models")
            self.model_chip.setToolTip("No enabled active model selected")
        self.model_image_badge.setVisible(self._has_active_model and not self._active_model_supports_images())
        self._refresh_reasoning_selector()
        self._refresh_submit_controls()

    def _refresh_reasoning_selector(self) -> None:
        profile = self._active_model_profile()
        options = reasoning_options_for_profile(profile)
        self.reasoning_chip_menu.clear()
        for action in self.reasoning_chip_group.actions():
            self.reasoning_chip_group.removeAction(action)
        if not options:
            self.reasoning_chip.setProperty("chipPosition", "hidden")
            self.reasoning_chip.setVisible(False)
            self.model_chip.setProperty("chipPosition", "single")
            self.model_chip.style().unpolish(self.model_chip)
            self.model_chip.style().polish(self.model_chip)
            return

        self.reasoning_chip.setProperty("chipPosition", "right")
        self.reasoning_chip.setVisible(True)
        self.model_chip.setProperty("chipPosition", "left")
        self.model_chip.style().unpolish(self.model_chip)
        self.model_chip.style().polish(self.model_chip)
        self.reasoning_chip.style().unpolish(self.reasoning_chip)
        self.reasoning_chip.style().polish(self.reasoning_chip)
        current = normalize_profile_reasoning((profile or {}).get("reasoning"))
        selected_value = "off" if current and not current.get("enabled", True) else ""
        provider = str((profile or {}).get("provider") or "").strip().lower()
        model = str((profile or {}).get("model") or "").strip().lower()
        if selected_value == "off" and provider == "gemini" and model.startswith(("gemini-3", "gemma-4")):
            selected_value = "minimal"
        if current.get("thinking_budget"):
            selected_value = f"budget:{current['thinking_budget']}"
        elif current.get("effort"):
            selected_value = str(current["effort"])
        elif provider == "anthropic" and current.get("mode"):
            selected_value = str(current["mode"])
        option_values = {str(option.get("value") or "") for option in options}
        if not selected_value and current.get("enabled", True) and "true" in option_values:
            selected_value = "true"
        elif (
            selected_value
            and selected_value not in option_values
            and "true" in option_values
            and current.get("enabled", True)
        ):
            selected_value = "true"

        selected_label = "Off" if selected_value == "off" else "Reasoning"
        for option in options:
            action = self.reasoning_chip_menu.addAction(str(option["label"]))
            action.setCheckable(True)
            action.setData(dict(option))
            action.setToolTip(f"Reasoning level: {option['label']}")
            is_selected = bool(selected_value and option["value"] == selected_value)
            action.setChecked(is_selected)
            self.reasoning_chip_group.addAction(action)
            if is_selected:
                selected_label = str(option["label"])
        self.reasoning_chip.setText(selected_label)
        self.reasoning_chip.setToolTip("Reasoning settings are saved with the active model profile")

    def _on_reasoning_action_triggered(self, action: QAction) -> None:
        option = action.data()
        profile = self._active_model_profile()
        if not isinstance(option, dict) or not isinstance(profile, dict):
            return
        profile_id = str(profile.get("id") or "").strip()
        reasoning = normalize_profile_reasoning(option.get("config"))
        if not profile_id or not reasoning:
            return

        payload = normalize_profiles_payload(self.model_profiles_payload)
        for candidate in payload.get("profiles", []):
            if str(candidate.get("id") or "").strip() == profile_id:
                candidate["reasoning"] = reasoning
                break
        else:
            return
        self._apply_model_profiles_payload(payload)
        self.controller.save_profiles(payload)

    def _apply_model_profiles_payload(self, payload: dict | None) -> None:
        self.model_profiles_payload = payload if isinstance(payload, dict) else {"active_profile": None, "profiles": []}
        self._refresh_model_selector()
        if self._request_blocked_by_image_capability():
            self._show_composer_notice(
                "Current model does not support image input. Remove the images or switch to an image-capable model.",
                level="warning",
            )
        else:
            self._clear_composer_notice()
        self._set_input_enabled(True)

    def _apply_model_capabilities_payload(self, payload: dict | None) -> None:
        capabilities = payload if isinstance(payload, dict) else {}
        self.model_capabilities = dict(DEFAULT_MODEL_CAPABILITIES)
        self.model_capabilities.update({"image_input_supported": bool(capabilities.get("image_input_supported"))})
        self.model_image_badge.setVisible(self._has_active_model and not self._active_model_supports_images())
        if self._request_blocked_by_image_capability():
            self._show_composer_notice(
                "Current model does not support image input. Remove the images or switch to an image-capable model.",
                level="warning",
            )
        else:
            self._clear_composer_notice()
        self._refresh_submit_controls()

    def _on_model_action_triggered(self, action: QAction) -> None:
        profile_id = str(action.data() or "").strip()
        if not profile_id:
            return
        self.controller.set_active_profile(profile_id)

    @staticmethod
    def _resolve_model_settings_dialog_class():
        main_module = sys.modules.get("main")
        if main_module is not None:
            patched_dialog = getattr(main_module, "ModelSettingsDialog", None)
            if patched_dialog is not None:
                return patched_dialog
        public_module = sys.modules.get("ui.main_window")
        if public_module is not None:
            patched_dialog = getattr(public_module, "ModelSettingsDialog", None)
            if patched_dialog is not None:
                return patched_dialog
        return ModelSettingsDialog

    def _open_settings_dialog(self) -> None:
        if self._model_settings_window is not None:
            if self.is_busy or self.awaiting_approval or self.awaiting_user_choice:
                self._model_settings_window.hide()
                return
            # Toggle: a second trigger hides the already-open panel.
            if not self._model_settings_window.isHidden():
                self._model_settings_window.hide()
                return
            self._model_settings_window.show()
            self._model_settings_window.raise_()
            self._model_settings_window.activateWindow()
            if hasattr(self._model_settings_window, "refresh_active_selection"):
                self._model_settings_window.refresh_active_selection(self.model_profiles_payload)
            return
        if self.is_busy or self.awaiting_approval or self.awaiting_user_choice:
            QMessageBox.information(self, "Busy", "Wait for the current run to finish before changing settings.")
            return
        dialog_class = self._resolve_model_settings_dialog_class()
        dialog = dialog_class(self.model_profiles_payload, self)
        dialog.profiles_saved.connect(self._save_model_profiles_from_dialog)
        visibility_changed = getattr(dialog, "visibility_changed", None)
        if visibility_changed is not None:
            visibility_changed.connect(self._handle_settings_visibility_changed)
        self._model_settings_window = dialog
        dialog.destroyed.connect(self._handle_settings_dialog_destroyed)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _handle_settings_dialog_destroyed(self, *_args) -> None:
        self._model_settings_window = None
        if self._settings_open:
            self._settings_open = False
            self._set_input_enabled(getattr(self, "_input_enabled", True))

    def _handle_settings_visibility_changed(self, visible: bool) -> None:
        """Block/unblock main-window input while the Settings panel is open."""
        self._settings_open = bool(visible)
        self._set_input_enabled(getattr(self, "_input_enabled", True))
        if not visible:
            self._update_centering()

    def _save_model_profiles_from_dialog(self, payload: dict | None) -> None:
        normalized = normalize_profiles_payload(payload or {})
        self._apply_model_profiles_payload(normalized)
        self.controller.save_profiles(normalized)

    def _handle_init_failed(self, message: str) -> None:
        self.tools_panel.fail_server_pending(message)
        self._set_status_visual("Initialization failed", error=True)
        QMessageBox.critical(self, "Initialization Failed", message)

    def _handle_session_changed(self, snapshot: dict) -> None:
        self._apply_runtime_payload(snapshot, restore_transcript="transcript" in snapshot)
        self._handle_checkpoint_notice(snapshot)

    def _handle_checkpoint_notice(self, payload: dict) -> None:
        notice = payload.get("checkpoint_notice") if isinstance(payload, dict) else None
        if isinstance(notice, dict):
            self._on_summary_notice(notice)

    def _apply_runtime_payload(self, payload: dict, *, restore_transcript: bool) -> None:
        previous_session_id = self.active_session_id
        self.current_snapshot = payload.get("snapshot", {})
        self.active_session_id = payload.get("active_session_id", "")
        cache_hit_tokens = self.current_snapshot.get("cache_hit_tokens", 0)
        self._set_session_cache_hit_tokens(cache_hit_tokens)

        self._apply_model_profiles_payload(payload.get("model_profiles"))
        self._apply_model_capabilities_payload(payload.get("model_capabilities"))
        if restore_transcript:
            self.composer.sync_session_history_from_transcript(
                self.active_session_id,
                payload.get("transcript"),
            )
        self.composer.set_history_session(self.active_session_id)
        self.overview_panel.set_snapshot(self.current_snapshot)
        self._update_env_info(self.current_snapshot)
        self.summary_progress_ring.set_summary_progress(payload.get("summary_progress"))
        tools = payload.get("tools", self.current_snapshot.get("tools", []))
        new_hash = hash(tuple(
            (
                t.get("kind", "tool"),
                t.get("name", ""),
                bool(t.get("enabled", True)),
                tuple((child.get("name", ""), child.get("description", "")) for child in t.get("tools", [])),
            )
            for t in tools
        ))

        if new_hash != self._tools_hash:
            self._tools_hash = new_hash
            self.tools_panel.set_tools(tools)
        if payload.get("help_markdown"):
            self.help_text.setMarkdown(payload.get("help_markdown", ""))
        self.sidebar.set_sessions(payload.get("sessions", []), self.active_session_id)
        if restore_transcript:
            same_session = bool(previous_session_id) and previous_session_id == self.active_session_id
            _ = same_session
            self.current_turn = None
            self.transcript.load_transcript(payload.get("transcript"))
            self._clear_draft_image_attachments()
            self._clear_composer_notice()
            self._clear_user_choice_request()
            self._clear_approval_request()
        pending_user_choice = payload.get("pending_user_choice") if isinstance(payload, dict) else None
        if isinstance(pending_user_choice, dict):
            self._handle_user_choice_request(pending_user_choice)

    def _handle_tool_availability_changed(self, kind: str, name: str, enabled: bool) -> None:
        setter_name = "set_mcp_server_enabled" if kind == "server" else "set_tool_enabled"
        setter = getattr(self.controller, setter_name, None)
        if callable(setter):
            setter(name, enabled)

    def _handle_busy_changed(self, busy: bool) -> None:
        self.tools_panel.setEnabled(not busy)
        self._status_controller.handle_busy_changed(busy)

    def _handle_event(self, event) -> None:
        if self._event_router is not None:
            self._event_router.dispatch(event)

    def _handle_approval_request(self, payload: dict) -> None:
        self.awaiting_approval = True
        self.awaiting_user_choice = False
        self._custom_choice_armed = False
        self._clear_user_choice_request()
        if self.current_turn is not None:
            self.current_turn.clear_status()
        self.approval_card.set_request(payload)
        self._set_input_enabled(False)
        self._set_status_visual("Waiting for approval", busy=False)
        self.approval_card.setFocus()
        self.transcript.notify_content_changed()

    def _handle_inline_approval_decision(self, approved: bool, always: bool) -> None:
        if not self.awaiting_approval:
            return
        self.awaiting_approval = False
        self._clear_approval_request()
        self.controller.resume_approval(approved, always)
        self._set_input_enabled(False)

    def _handle_user_choice_request(self, payload: dict) -> None:
        payload = payload if isinstance(payload, dict) else {}
        self.awaiting_user_choice = True
        self._custom_choice_armed = bool(payload.get("allow_custom_text"))
        if self.current_turn is not None:
            self.current_turn.clear_status()
        self._set_user_choice_request(payload)
        self._set_status_visual("Waiting for your choice", busy=False)
        self._set_input_enabled(True)
        if self._custom_choice_armed:
            self.composer.setFocus()

    def _sanitize_composer_text(self) -> tuple[str, str]:
        result = sanitize_user_text(self.composer.toPlainText())
        return result.text, build_user_input_notice(result)

    def _submit_request(self) -> None:
        text, sanitize_notice = self._sanitize_composer_text()
        attachments = list(self.draft_image_attachments)
        if not text and not attachments:
            if sanitize_notice:
                self._show_composer_notice(sanitize_notice, level="warning")
            return
        if self.awaiting_user_choice:
            if self.is_busy or self.awaiting_approval or not self._custom_choice_armed:
                return
            self._clear_user_choice_request()
            self.composer.clear()
            self._clear_draft_image_attachments()
            if sanitize_notice:
                self._show_composer_notice(sanitize_notice, level="warning")
            else:
                self._clear_composer_notice()
            self.controller.resume_user_choice(text)
            self._set_input_enabled(False)
            return
        if attachments and not self._active_model_supports_images():
            self._show_composer_notice(
                "Current model does not support image input. Switch models or remove the images.",
                level="warning",
            )
            return
        self._clear_user_choice_request()
        if text:
            self.composer.append_submitted_message(text)
        self.composer.clear()
        if sanitize_notice:
            self._show_composer_notice(sanitize_notice, level="warning")
        else:
            self._clear_composer_notice()
        request_payload = {
            "text": text,
            "attachments": attachments,
        }
        self._clear_draft_image_attachments()
        self.controller.start_run(request_payload)
        self._set_input_enabled(False)

    def _set_user_choice_request(self, payload: dict | None) -> None:
        if not isinstance(payload, dict) or not payload:
            self._clear_user_choice_request()
            return
        self.user_choice_card.set_request(payload)
        self.user_choice_card.set_actions_enabled(
            self.awaiting_user_choice and not self.awaiting_approval and not self.is_busy
        )

    def _clear_user_choice_request(self) -> None:
        self.awaiting_user_choice = False
        self._custom_choice_armed = False
        self.user_choice_card.clear_request()

    def _clear_approval_request(self) -> None:
        self.approval_card.clear_request()

    def _handle_user_choice_selected(self, submit_text: str) -> None:
        text = str(submit_text or "").strip()
        if not text or self.is_busy or self.awaiting_approval or not self.awaiting_user_choice:
            return
        self._clear_user_choice_request()
        self.controller.resume_user_choice(text)
        self._set_input_enabled(False)

    def _handle_custom_choice_requested(self) -> None:
        if self.is_busy or self.awaiting_approval or not self.awaiting_user_choice:
            return
        self._custom_choice_armed = True
        self._set_input_enabled(True)
        self.composer.setFocus()
        self.composer.selectAll()

    def _attach_images(self) -> None:
        if self.is_busy:
            return
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images to Attach",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif);;All files (*)",
        )
        if not file_paths:
            return
        self._import_image_files([str(path) for path in file_paths])

    def _insert_file_paths(self) -> None:
        if self.is_busy:
            return
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Files to Insert")
        if not file_paths:
            return

        image_paths: list[str] = []
        text_paths: list[str] = []
        for raw_path in file_paths:
            path = str(raw_path or "").strip()
            if not path:
                continue
            if can_read_image_file(path):
                image_paths.append(path)
            else:
                text_paths.append(path)

        if image_paths and self._active_model_supports_images():
            self._import_image_files(image_paths)
        elif image_paths:
            text_paths.extend(image_paths)
            self._show_composer_notice(
                "Current model does not support image input. Inserted image file paths as text references instead.",
                level="warning",
            )

        current_text = self.composer.toPlainText()
        if text_paths:
            if current_text and not current_text.endswith(" "):
                current_text += " "

            for path in text_paths:
                current_text += f"{self.composer.format_file_reference(path)} "

            self.composer.setPlainText(current_text)
            from PySide6.QtGui import QTextCursor

            cursor = self.composer.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.composer.setTextCursor(cursor)
        self.composer.setFocus()

    def _toggle_sidebar(self) -> None:
        self._sidebar_controller.toggle_sidebar()
        self._update_centering()

    def _update_centering(self) -> None:
        self._workspace_builder.apply_centering_compensation()

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        # Recompute chat centering whenever the center panel gets its final
        # geometry (window resize, sidebar/inspector toggle, dock open/close).
        # Move matters as much as Resize: a full-width Settings dock pushes the
        # central widget off-window without resizing it, and on close the panel
        # returns via a Move event only.
        if watched is self.transcript.parentWidget() and event.type() in (QEvent.Resize, QEvent.Move):
            self._workspace_builder.apply_centering_compensation()
        return super().eventFilter(watched, event)

    def _switch_session(self, session_id: str) -> None:
        self._sidebar_controller.switch_session(session_id)

    def _request_delete_session(self, session_id: str) -> None:
        self._sidebar_controller.request_delete_session(session_id)

    def _request_delete_project(self, project_path: str) -> None:
        self._sidebar_controller.request_delete_project(project_path)

    def _new_session(self) -> None:
        self._sidebar_controller.new_session()

    def _open_new_project(self) -> None:
        self._sidebar_controller.open_new_project()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._update_centering()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_centering()
        # The resize handler above still sees pre-layout geometry; recompute
        # centering once layouts have settled so the chat column stays centered.
        QTimer.singleShot(0, self._update_centering)

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            self.controller.shutdown()
        finally:
            super().closeEvent(event)


def main() -> int:
    _configure_qt_logging()
    _configure_windows_app_user_model_id()
    QApplication.setApplicationName(APP_DISPLAY_NAME)
    QApplication.setOrganizationName("Simple AI Agent")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app_icon = _build_app_icon()
    app.setWindowIcon(app_icon)
    window = MainWindow()
    window.setWindowIcon(app_icon)
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
