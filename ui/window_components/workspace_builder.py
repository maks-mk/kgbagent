from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.theme import TEXT_MUTED
from ui.widgets import (
    ApprovalRequestCardWidget,
    ChatTranscriptWidget,
    ComposerTextEdit,
    ImageAttachmentStripWidget,
    InspectorPanelWidget,
    SessionSidebarWidget,
    SummaryProgressRing,
    TRANSCRIPT_MAX_WIDTH,
    UserChoiceCardWidget,
    _fa_icon,
)

COMPOSER_ATTACH_TOOLTIP = "Add images or insert file paths"
COMPOSER_ADD_IMAGE_LABEL = "Add image…"
COMPOSER_INSERT_FILE_PATH_LABEL = "Add files…"


@dataclass(frozen=True)
class WorkspaceBuildResult:
    workspace: QWidget
    splitter: QSplitter
    sidebar_container: QFrame
    sidebar: SessionSidebarWidget
    transcript: ChatTranscriptWidget
    composer_shell: QHBoxLayout
    composer_container: QWidget
    approval_card: ApprovalRequestCardWidget
    user_choice_card: UserChoiceCardWidget
    composer_pill: QFrame
    composer_attachments_strip: ImageAttachmentStripWidget
    composer_notice_label: QLabel
    composer: ComposerTextEdit
    attach_button: QToolButton
    attach_menu: QMenu
    add_image_action: object
    insert_file_path_action: object
    model_chip: QToolButton
    model_chip_menu: QMenu
    model_chip_group: QActionGroup
    reasoning_chip: QToolButton
    reasoning_chip_menu: QMenu
    reasoning_chip_group: QActionGroup
    model_image_badge: QLabel
    no_models_label: QLabel
    open_settings_inline_button: QPushButton
    cache_hit_label: QLabel
    summary_progress_ring: SummaryProgressRing
    send_button: QPushButton
    stop_action_button: QPushButton
    inspector_container: QFrame
    inspector_panel: InspectorPanelWidget
    overview_panel: QWidget
    tools_panel: QWidget
    help_text: QWidget


class WorkspaceBuilder:
    def __init__(self, window) -> None:
        self.window = window

    def apply_centering_compensation(self) -> None:
        """Keep the chat column and composer centered on the whole window.

        The transcript/composer column is centered inside the splitter's center
        panel, so the open Projects sidebar shifts it off the window center.
        Compensate with asymmetric shell margins so the column stays visually
        centered on the window regardless of the sidebar state. The right-side
        inspector must NOT trigger this compensation: with it open the chat
        simply centers inside the center panel.
        """
        window = self.window
        center_panel = window.transcript.parentWidget()
        if center_panel is None or not center_panel.isVisible():
            return
        # Activate layouts first so the geometry reads below reflect the latest
        # splitter sizes instead of the pre-layout values.
        if window.layout() is not None:
            window.layout().activate()
        if center_panel.layout() is not None:
            center_panel.layout().activate()
        right_panel_open = window.inspector_container.isVisible()
        if right_panel_open:
            left = right = 0
        else:
            panel_left = center_panel.mapTo(window, center_panel.rect().topLeft()).x()
            panel_width = center_panel.width()
            column_width = min(TRANSCRIPT_MAX_WIDTH, panel_width)
            # Column start (in panel coordinates) that puts its center on the
            # window center; clamped to the panel so the sidebar is never overlapped.
            target_left = window.width() // 2 - column_width // 2 - panel_left
            left = max(0, min(target_left, panel_width - column_width))
            right = max(0, panel_width - column_width - left)
        shell = window.transcript.shell
        margins = shell.contentsMargins()
        shell.setContentsMargins(left, margins.top(), right, margins.bottom())
        composer_shell = window.composer_shell
        composer_margins = composer_shell.contentsMargins()
        composer_shell.setContentsMargins(left, composer_margins.top(), right, composer_margins.bottom())
        # Apply the new margins immediately instead of waiting for the next
        # event-loop pass, so the column never appears off-center.
        shell.activate()
        center_panel.layout().activate()

    def _history_batch_size(self) -> int | None:
        """Read HISTORY_BATCH_SIZE from the agent config without importing heavy modules."""
        try:
            from core.config import AgentConfig
        except Exception:
            return None
        try:
            return int(AgentConfig().history_batch_size)
        except Exception:
            return None

    def build(self) -> WorkspaceBuildResult:
        workspace = QWidget()
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        sidebar_container = QFrame()
        sidebar_container.setObjectName("SidebarCard")
        sidebar_container.setMinimumWidth(250)
        sidebar_container.setMaximumWidth(360)
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(0)

        sidebar = SessionSidebarWidget()
        sidebar_layout.addWidget(sidebar, 1)

        center_panel = QWidget()
        center_layout = QGridLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setHorizontalSpacing(8)
        center_layout.setVerticalSpacing(0)

        transcript = ChatTranscriptWidget(history_batch_size=self._history_batch_size())
        transcript.setAccessibleName("Conversation transcript")
        transcript.setAccessibleDescription("Shows user messages, assistant output, tools, and notices")
        center_layout.addWidget(transcript, 0, 0)
        center_layout.setRowStretch(0, 1)
        center_layout.setRowStretch(1, 0)
        center_layout.setColumnStretch(0, 1)
        center_layout.setColumnStretch(1, 0)

        composer_shell = QHBoxLayout()
        composer_shell.setContentsMargins(0, 18, 0, 12)
        composer_shell.setSpacing(0)
        composer_shell.addStretch(0)

        composer_container = QWidget()
        composer_container.setObjectName("CenteredComposerRow")
        composer_container.setMaximumWidth(TRANSCRIPT_MAX_WIDTH)
        composer_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        outer_layout = QVBoxLayout(composer_container)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(8)

        approval_card = ApprovalRequestCardWidget()
        outer_layout.addWidget(approval_card)

        user_choice_card = UserChoiceCardWidget()
        outer_layout.addWidget(user_choice_card)

        composer_pill = QFrame()
        composer_pill.setObjectName("ComposerPill")
        pill_layout = QVBoxLayout(composer_pill)
        pill_layout.setContentsMargins(10, 8, 8, 8)
        pill_layout.setSpacing(6)

        composer_attachments_strip = ImageAttachmentStripWidget(thumb_size=48, removable=True)
        pill_layout.addWidget(composer_attachments_strip)

        composer_notice_label = QLabel("")
        composer_notice_label.setObjectName("ComposerNoticeLabel")
        composer_notice_label.setWordWrap(True)
        composer_notice_label.setVisible(False)
        pill_layout.addWidget(composer_notice_label)

        composer = ComposerTextEdit()
        composer.setPlaceholderText("Describe your task, attach context, or type @ to add files…")
        composer.setAccessibleName("Composer")
        composer.setAccessibleDescription("Write a request for the agent")
        line_spacing = max(14, composer.fontMetrics().lineSpacing())
        self.window._composer_max_height = self.window._composer_min_height + (self.window._composer_growth_lines * line_spacing)
        composer.setMinimumHeight(self.window._composer_min_height)
        composer.setMaximumHeight(self.window._composer_max_height)
        composer.setFixedHeight(self.window._composer_min_height)
        composer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        composer.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        composer.set_history_session(self.window.active_session_id)
        pill_layout.addWidget(composer, 1)

        control_row = QHBoxLayout()
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(6)

        attach_button = QToolButton()
        attach_button.setIcon(_fa_icon("fa5s.plus", color=TEXT_MUTED, size=12))
        attach_button.setIconSize(QSize(12, 12))
        attach_button.setFixedSize(28, 28)
        attach_button.setObjectName("ComposerAttachButton")
        attach_button.setToolTip(COMPOSER_ATTACH_TOOLTIP)
        attach_button.setAccessibleName("Add attachments")
        attach_button.setAccessibleDescription("Add images or Add files")
        attach_menu = QMenu(attach_button)
        attach_menu.setObjectName("ComposerPopupMenu")
        add_image_action = attach_menu.addAction(COMPOSER_ADD_IMAGE_LABEL)
        insert_file_path_action = attach_menu.addAction(COMPOSER_INSERT_FILE_PATH_LABEL)
        attach_button.setMenu(attach_menu)
        attach_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        configure_composer_popup(attach_menu)
        control_row.addWidget(attach_button, 0, Qt.AlignVCenter)

        model_chip = QToolButton()
        model_chip.setObjectName("ComposerMetaChipButton")
        model_chip.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        model_chip.setToolButtonStyle(Qt.ToolButtonTextOnly)
        model_chip.setText("Model")
        model_chip.setCursor(Qt.PointingHandCursor)
        model_chip.setFocusPolicy(Qt.NoFocus)
        model_chip.setAccessibleName("Model selector")
        model_chip.setAccessibleDescription("Select the active model profile")
        model_chip_menu = QMenu(model_chip)
        model_chip_menu.setObjectName("ComposerPopupMenu")
        model_chip.setMenu(model_chip_menu)
        model_chip_group = QActionGroup(model_chip_menu)
        model_chip_group.setExclusive(True)
        configure_composer_popup(model_chip_menu)
        model_chip.setProperty("chipPosition", "single")
        meta_chip_row = QHBoxLayout()
        meta_chip_row.setContentsMargins(0, 0, 0, 0)
        meta_chip_row.setSpacing(0)
        meta_chip_row.addWidget(model_chip, 0, Qt.AlignVCenter)
        control_row.addLayout(meta_chip_row)

        reasoning_chip = QToolButton()
        reasoning_chip.setObjectName("ComposerMetaChipButton")
        reasoning_chip.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        reasoning_chip.setToolButtonStyle(Qt.ToolButtonTextOnly)
        reasoning_chip.setText("Reasoning")
        reasoning_chip.setCursor(Qt.PointingHandCursor)
        reasoning_chip.setFocusPolicy(Qt.NoFocus)
        reasoning_chip.setAccessibleName("Reasoning level selector")
        reasoning_chip.setAccessibleDescription("Select the reasoning level for the active model")
        reasoning_chip.setProperty("chipPosition", "right")
        meta_chip_row.addWidget(reasoning_chip, 0, Qt.AlignVCenter)
        reasoning_chip_menu = QMenu(reasoning_chip)
        reasoning_chip_menu.setObjectName("ComposerPopupMenu")
        reasoning_chip.setMenu(reasoning_chip_menu)
        reasoning_chip_group = QActionGroup(reasoning_chip_menu)
        reasoning_chip_group.setExclusive(True)
        configure_composer_popup(reasoning_chip_menu)

        model_image_badge = QLabel()
        model_image_badge.setObjectName("ComposerCapabilityBadge")
        model_image_badge.setPixmap(_fa_icon("mdi6.image-off-outline", color=TEXT_MUTED, size=14).pixmap(14, 14))
        model_image_badge.setFixedSize(28, 28)
        model_image_badge.setAlignment(Qt.AlignCenter)
        model_image_badge.setToolTip("Image input unavailable for this model")
        model_image_badge.setAccessibleName("Image input unavailable")
        model_image_badge.setAccessibleDescription("The active model profile does not accept image attachments")
        model_image_badge.setVisible(False)
        control_row.addWidget(model_image_badge, 0, Qt.AlignVCenter)

        no_models_label = QLabel("No models configured")
        no_models_label.setObjectName("ComposerNoModelText")
        no_models_label.setVisible(False)
        control_row.addWidget(no_models_label, 0, Qt.AlignVCenter)

        open_settings_inline_button = QPushButton("Open Settings")
        open_settings_inline_button.setObjectName("ComposerOpenSettingsButton")
        open_settings_inline_button.setVisible(False)
        open_settings_inline_button.setFocusPolicy(Qt.NoFocus)
        control_row.addWidget(open_settings_inline_button, 0, Qt.AlignVCenter)

        cache_hit_label = QLabel("Cache Hit: 0")
        cache_hit_label.setObjectName("ComposerCacheHitLabel")
        cache_hit_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        cache_hit_label.setAccessibleName("Session cache hit tokens")
        cache_hit_label.setToolTip("Prompt cache tokens reported by the provider for this session")
        control_row.addWidget(cache_hit_label, 0, Qt.AlignVCenter)

        control_row.addStretch(1)

        summary_progress_ring = SummaryProgressRing()
        control_row.addWidget(summary_progress_ring, 0, Qt.AlignVCenter)

        send_button = QPushButton(_fa_icon("fa5s.arrow-up", color="#08090B", size=14), "")
        send_button.setObjectName("ComposerSendButton")
        send_button.setToolTip("Send (Enter)")
        send_button.setFixedSize(32, 32)
        send_button.setAccessibleName("Send request")
        send_button.setAccessibleDescription("Submit the current request")
        control_row.addWidget(send_button, 0, Qt.AlignVCenter)

        stop_action_button = QPushButton(_fa_icon("fa5s.stop", color="#08090B", size=14), "")
        stop_action_button.setObjectName("ComposerStopButton")
        stop_action_button.setToolTip("Stop")
        stop_action_button.setFixedSize(32, 32)
        stop_action_button.setAccessibleName("Stop current run")
        stop_action_button.setVisible(False)
        control_row.addWidget(stop_action_button, 0, Qt.AlignVCenter)
        pill_layout.addLayout(control_row)

        outer_layout.addWidget(composer_pill)
        composer_shell.addWidget(composer_container, 1)
        composer_shell.addStretch(0)
        # Transcript and composer share column 0.
        center_layout.addLayout(composer_shell, 1, 0)

        inspector_container = QFrame()
        inspector_container.setObjectName("SidebarCard")
        inspector_container.setMinimumWidth(320)
        inspector_container.setMaximumWidth(420)
        inspector_layout = QVBoxLayout(inspector_container)
        inspector_layout.setContentsMargins(12, 12, 12, 12)
        inspector_layout.setSpacing(0)
        inspector_panel = InspectorPanelWidget()
        overview_panel = inspector_panel.overview_panel
        tools_panel = inspector_panel.tools_panel
        help_text = inspector_panel.help_text
        inspector_layout.addWidget(inspector_panel, 1)

        splitter.addWidget(sidebar_container)
        splitter.addWidget(center_panel)
        splitter.addWidget(inspector_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([self.window._sidebar_width, 1000, self.window._inspector_width])
        layout.addWidget(splitter, 1)

        return WorkspaceBuildResult(
            workspace=workspace,
            splitter=splitter,
            sidebar_container=sidebar_container,
            sidebar=sidebar,
            transcript=transcript,
            composer_shell=composer_shell,
            composer_container=composer_container,
            approval_card=approval_card,
            user_choice_card=user_choice_card,
            composer_pill=composer_pill,
            composer_attachments_strip=composer_attachments_strip,
            composer_notice_label=composer_notice_label,
            composer=composer,
            attach_button=attach_button,
            attach_menu=attach_menu,
            add_image_action=add_image_action,
            insert_file_path_action=insert_file_path_action,
            model_chip=model_chip,
            model_chip_menu=model_chip_menu,
            model_chip_group=model_chip_group,
            reasoning_chip=reasoning_chip,
            reasoning_chip_menu=reasoning_chip_menu,
            reasoning_chip_group=reasoning_chip_group,
            model_image_badge=model_image_badge,
            no_models_label=no_models_label,
            open_settings_inline_button=open_settings_inline_button,
            cache_hit_label=cache_hit_label,
            summary_progress_ring=summary_progress_ring,
            send_button=send_button,
            stop_action_button=stop_action_button,
            inspector_container=inspector_container,
            inspector_panel=inspector_panel,
            overview_panel=overview_panel,
            tools_panel=tools_panel,
            help_text=help_text,
        )


def configure_composer_popup(popup: QWidget) -> None:
    # Rounded stylesheet corners need a translucent native window. On Windows,
    # Qt also requires FramelessWindowHint; set both before the first show.
    popup.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
    popup.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    # A graphics shadow on the top-level menu expands dirty regions beyond
    # its backing store, causing UpdateLayeredWindowIndirect failures on Windows.
    # Keep the rounded alpha surface without an effect or a rectangular OS shadow.
    popup.setWindowFlags(popup.windowFlags() | Qt.WindowType.NoDropShadowWindowHint)
