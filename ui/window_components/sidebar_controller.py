from __future__ import annotations

import os

from PySide6.QtWidgets import QFileDialog, QMessageBox


class SidebarController:
    def __init__(self, window) -> None:
        self.window = window

    def toggle_sidebar(self) -> None:
        self.set_sidebar_collapsed(not self.window.sidebar_collapsed)

    def set_sidebar_collapsed(self, collapsed: bool) -> None:
        if self.window.sidebar_collapsed == collapsed:
            return
        self.window.sidebar_collapsed = collapsed
        if collapsed:
            self.window._sidebar_width = max(250, self.window.sidebar_container.width())
            self.window.sidebar_container.hide()
            self.window.splitter.setSizes([0, 1000, 0 if self.window.inspector_collapsed else self.window._inspector_width])
            self.window.toggle_sidebar_action.setToolTip("Show chat history (Ctrl+B)")
            self.window.sidebar_toggle_button.setToolTip(self.window.toggle_sidebar_action.toolTip())
            return
        self.window.sidebar_container.show()
        self.window.splitter.setSizes([self.window._sidebar_width, 1000, 0 if self.window.inspector_collapsed else self.window._inspector_width])
        self.window.toggle_sidebar_action.setToolTip("Hide chat history (Ctrl+B)")
        self.window.sidebar_toggle_button.setToolTip(self.window.toggle_sidebar_action.toolTip())

    def switch_session(self, session_id: str) -> None:
        if (
            self.window.is_busy
            or self.window.awaiting_approval
            or self.window.awaiting_user_choice
            or not session_id
            or session_id == self.window.active_session_id
        ):
            return
        self.window.composer.set_history_session(session_id)
        self.window.current_turn = None
        self.window._clear_draft_image_attachments()
        self.window._clear_composer_notice()
        self.window._set_session_cache_hit_tokens(0)
        self.window._clear_user_choice_request()
        self.window._show_transient_status_message("Switching chat…")
        self.window.controller.switch_session(session_id)

    def request_delete_session(self, session_id: str) -> None:
        if not session_id or self.window.is_busy or self.window.awaiting_approval or self.window.awaiting_user_choice:
            return
        title = self.window.sidebar.title_for_session(session_id) or "this chat"
        message = f"Delete “{title}” from chat history?"
        answer = QMessageBox.question(
            self.window,
            "Delete chat",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.window._show_transient_status_message("Deleting chat from history…")
        self.window.controller.delete_session(session_id)

    def request_delete_project(self, project_path: str) -> None:
        if not project_path or self.window.is_busy or self.window.awaiting_approval or self.window.awaiting_user_choice:
            return
        title = self.window.sidebar.title_for_project(project_path) or "this project"
        count = self.window.sidebar.session_count_for_project(project_path)
        chats = "chat" if count == 1 else "chats"
        message = f"Delete all {count} {chats} of “{title}” from chat history?"
        answer = QMessageBox.question(
            self.window,
            "Delete project",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.window._show_transient_status_message("Deleting project from history…")
        self.window.controller.delete_project(project_path)

    def new_session(self) -> None:
        if self.window.is_busy or self.window.awaiting_user_choice:
            QMessageBox.information(self.window, "Busy", "Wait for the current run to finish before starting a new session.")
            return
        pending_key = "__pending_new_session__"
        self.window.composer.clear_history_for_session(pending_key)
        self.window.composer.set_history_session(pending_key)
        self.window.composer.reset_history_navigation()
        self.window.current_turn = None
        self.window.transcript.clear_transcript()
        self.window._clear_draft_image_attachments()
        self.window._clear_composer_notice()
        self.window._set_session_cache_hit_tokens(0)
        self.window._clear_user_choice_request()
        self.window._clear_approval_request()
        self.window.controller.new_session()
        self.window._show_transient_status_message("Created a new session")

    def open_new_project(self) -> None:
        if self.window.is_busy or self.window.awaiting_user_choice:
            QMessageBox.information(self.window, "Busy", "Wait for the current run to finish before changing the project directory.")
            return

        dir_path = QFileDialog.getExistingDirectory(
            self.window,
            "Select Project Directory",
            os.getcwd(),
        )
        if not dir_path:
            return

        self.window._show_transient_status_message(f"Switching project to: {dir_path}")
        os.chdir(dir_path)

        pending_key = "__pending_project_switch__"
        self.window.composer.clear_history_for_session(pending_key)
        self.window.composer.set_history_session(pending_key)
        self.window.composer.reset_history_navigation()
        self.window.current_turn = None
        self.window.transcript.clear_transcript()
        self.window._set_session_cache_hit_tokens(0)
        self.window._clear_user_choice_request()
        self.window._clear_approval_request()
        self.window._set_status_visual("Creating a new chat for the selected folder…", busy=True)
        self.window.controller.reinitialize(force_new_session=True)
