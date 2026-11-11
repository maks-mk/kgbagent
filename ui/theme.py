from __future__ import annotations

from functools import lru_cache

# Monochrome interface palette. Change UI colors here, independently of content.
COLOR_BLACK = "#000000"
COLOR_WHITE = "#FFFFFF"

TEXT_PRIMARY = "#D0D0D0"
TEXT_CHAT = "#F5F5F5"
TEXT_SECONDARY = "#929292"
TEXT_MUTED = TEXT_SECONDARY
TEXT_DIM = "#6E6E6E"
TEXT_WARNING = TEXT_PRIMARY
TEXT_ERROR = TEXT_CHAT
TEXT_SUCCESS = TEXT_PRIMARY
ACCENT = "#BFBFBF"
CONTEXT_INDICATOR_GRAY = ACCENT
CONTEXT_INDICATOR_TRACK = "#5C5C5C"

SURFACE_BG = "#212121"
SURFACE_PROJECTS = "#1C1C1C"
SURFACE_DIALOG = "#242424"
SURFACE_SETTINGS = "#2A2A2A"
SURFACE_CARD = "#262626"
SURFACE_ALT = "#303030"
SURFACE_CONFIRMED = "#545454"
USER_MESSAGE_BG = "#2D2D2D"
AI_MESSAGE_BG = SURFACE_DIALOG
INPUT_BG = "#363636"
BORDER = "#3A3A3A"
ACCENT_SOFT = SURFACE_ALT

# Content highlighting remains colored: code, diff, links and filenames.
ACCENT_BLUE = "#8AB4F8"
AMBER_WARNING = "#D6AE72"
ERROR_RED = "#FF7B72"
SUCCESS_GREEN = "#63D882"
FILENAME_BLUE = "#6FC8FF"
CODE_BG = SURFACE_BG
CODE_TEXT = TEXT_PRIMARY
DIFF_ADDED_BG = "#1E3425"
DIFF_REMOVED_BG = "#472B2B"
DIFF_ADDED_TEXT = SUCCESS_GREEN
DIFF_REMOVED_TEXT = ERROR_RED

# Control colors shared by stylesheets and painted icons.
COMPOSER_BG = INPUT_BG
COMPOSER_BORDER = BORDER
SEND_BUTTON_BG = TEXT_PRIMARY
SEND_BUTTON_HOVER = "#DCDCDC"
SEND_BUTTON_DISABLED = BORDER
SAVE_BUTTON_BG = COLOR_WHITE
SAVE_BUTTON_TEXT = COLOR_BLACK
SAVE_BUTTON_HOVER = "#F0F0F0"
SAVED_BUTTON_BG = SURFACE_CONFIRMED
SAVED_BUTTON_TEXT = COLOR_WHITE
PROVIDER_ICON_TEXT = TEXT_MUTED
SWITCH_BORDER_TINT = TEXT_MUTED
SWITCH_THUMB_TINT = TEXT_PRIMARY
FILE_PDF_ICON_TEXT = TEXT_PRIMARY
# The active chat keeps a colored marker within the monochrome interface.
ACTIVE_CHAT_INDICATOR = "#91B99A"

APP_FONT_FAMILY = "Segoe UI"
MONO_FONT_FAMILY = "Cascadia Mono"

SOFT_RADIUS_XS = 2
SOFT_RADIUS_SM = 4
SOFT_RADIUS_MD = 6


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def blend_hex(start_hex: str, end_hex: str, factor: float) -> str:
    factor = max(0.0, min(1.0, factor))
    start = _hex_to_rgb(start_hex)
    end = _hex_to_rgb(end_hex)
    blended = tuple(
        round(start[index] + (end[index] - start[index]) * factor)
        for index in range(3)
    )
    return "#{:02X}{:02X}{:02X}".format(*blended)


# Shared interface surfaces and text.
SURFACE_HOVER = blend_hex(SURFACE_CARD, TEXT_PRIMARY, 0.05)
SURFACE_SUBTLE = blend_hex(SURFACE_CARD, TEXT_PRIMARY, 0.03)
SURFACE_RAISED = blend_hex(SURFACE_CARD, TEXT_PRIMARY, 0.04)
SURFACE_SOFT = blend_hex(SURFACE_ALT, TEXT_PRIMARY, 0.04)
SURFACE_BADGE = blend_hex(SURFACE_ALT, TEXT_PRIMARY, 0.05)
BORDER_SOFT = blend_hex(BORDER, TEXT_PRIMARY, 0.10)
TEXT_SOFT = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.18)
TEXT_META = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.26)
TEXT_SUMMARY = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.42)
STATUS_SUCCESS_BG = blend_hex(TEXT_SUCCESS, SURFACE_ALT, 0.82)
STATUS_SUCCESS_TEXT = blend_hex(TEXT_SUCCESS, TEXT_PRIMARY, 0.18)

# General controls, statuses, tools and content containers.
TOOL_PANEL_BG = SURFACE_RAISED
TOOL_TOGGLE_TEXT = TEXT_META
TOOL_PANEL_HOVER = blend_hex(TOOL_PANEL_BG, TEXT_PRIMARY, 0.05)
TOOL_CODE_BG = blend_hex(SURFACE_CARD, TEXT_PRIMARY, 0.07)
TOOL_TOGGLE_HOVER_TEXT = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.52)
TOOL_CALL_HOVER = TEXT_SUMMARY
DANGER_BUTTON_BG = blend_hex(TEXT_ERROR, SURFACE_ALT, 0.78)
SIDEBAR_DISABLED_TEXT = blend_hex(TEXT_MUTED, SURFACE_CARD, 0.45)
SIDEBAR_DISABLED_BG = blend_hex(SURFACE_CARD, SURFACE_BG, 0.4)
STATUS_BG = blend_hex(SURFACE_ALT, TEXT_PRIMARY, 0.06)
STATUS_BUSY_BG = blend_hex(ACCENT_SOFT, ACCENT, 0.24)
STATUS_ERROR_BG = blend_hex(TEXT_ERROR, SURFACE_ALT, 0.82)
STATUS_ERROR_TEXT = blend_hex(TEXT_ERROR, TEXT_PRIMARY, 0.18)
TEXT_LABEL = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.22)
ATTACHMENT_REMOVE_BG = blend_hex(SURFACE_BG, TEXT_PRIMARY, 0.08)
ATTACHMENT_REMOVE_HOVER = blend_hex(SURFACE_BG, TEXT_PRIMARY, 0.16)
SCROLLBAR_HANDLE = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.14)
SCROLLBAR_HANDLE_HOVER = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.24)
SAVED_BUTTON_HOVER = blend_hex(SAVED_BUTTON_BG, COLOR_BLACK, 0.10)
TEXT_DETAIL = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.28)
TEXT_SUBTITLE = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.30)
TEXT_QUESTION = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.44)
TEXT_HINT = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.34)
STATUS_WARNING_BG = blend_hex(TEXT_WARNING, SURFACE_ALT, 0.82)
STATUS_WARNING_TEXT = blend_hex(TEXT_WARNING, TEXT_PRIMARY, 0.14)
ACCENT_BUTTON_BG = blend_hex(ACCENT, SURFACE_ALT, 0.82)
ACCENT_BUTTON_BORDER = blend_hex(ACCENT, TEXT_PRIMARY, 0.10)
ACCENT_BUTTON_HOVER = blend_hex(ACCENT, SURFACE_ALT, 0.72)
DANGER_BUTTON_BORDER = blend_hex(TEXT_ERROR, TEXT_PRIMARY, 0.08)
TEXT_SELECTION_BG = blend_hex(ACCENT, SURFACE_ALT, 0.20)
STATUS_REVIEW_TEXT = blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.08)
STATUS_WAITING_TEXT = blend_hex(TEXT_WARNING, TEXT_PRIMARY, 0.24)
BUTTON_HOVER = blend_hex(SURFACE_ALT, ACCENT_SOFT, 0.55)
BUTTON_PRESSED = blend_hex(SURFACE_ALT, ACCENT_SOFT, 0.8)
HEADER_BUTTON_PRESSED = blend_hex(SURFACE_ALT, ACCENT_SOFT, 0.35)
BUTTON_BG = blend_hex(SURFACE_ALT, TEXT_PRIMARY, 0.08)
DISCLOSURE_HOVER_TEXT = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.45)
CODE_PANEL_BORDER = blend_hex(BORDER, TEXT_PRIMARY, 0.04)
CODE_HEADER_TEXT = blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.1)
BADGE_ACTIVE_BG = blend_hex(ACCENT_SOFT, ACCENT, 0.18)
BADGE_ACTIVE_TEXT = blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.04)
BADGE_ERROR_BG = blend_hex(TEXT_ERROR, SURFACE_ALT, 0.18)
DIFF_PANEL_BORDER = blend_hex(BORDER, TEXT_PRIMARY, 0.05)
DIFF_HEADER_TEXT = FILENAME_BLUE
DIFF_TOGGLE_TEXT = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.32)
DIFF_TOGGLE_CHECKED_TEXT = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.48)
TREE_BG = blend_hex(SURFACE_CARD, TEXT_PRIMARY, 0.01)
TREE_SELECTION_BG = blend_hex(ACCENT_SOFT, ACCENT, 0.25)
CHOICE_BUTTON_HOVER = blend_hex(SURFACE_ALT, TEXT_PRIMARY, 0.15)
CHOICE_BUTTON_HOVER_BORDER = blend_hex(BORDER, TEXT_PRIMARY, 0.20)
CHOICE_BUTTON_PRESSED = blend_hex(SURFACE_ALT, TEXT_PRIMARY, 0.19)
CHOICE_BUTTON_DISABLED_BG = blend_hex(SURFACE_ALT, SURFACE_CARD, 0.35)
TRANSCRIPT_JUMP_BG = blend_hex(SURFACE_CARD, TEXT_PRIMARY, 0.08)
TRANSCRIPT_ROLE_TEXT = blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.06)
TOOL_CALL_TEXT = blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.20)
TOOL_GROUP_ERROR_TEXT = blend_hex(TEXT_ERROR, TEXT_PRIMARY, 0.22)
TOOL_GROUP_HOVER_TEXT = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.58)
INSPECTOR_HELP_BG = blend_hex(SURFACE_CARD, TEXT_PRIMARY, 0.02)
TOOL_SWITCH_BORDER = blend_hex(BORDER, TEXT_PRIMARY, 0.16)
TOOL_SWITCH_HOVER = blend_hex(SURFACE_ALT, TEXT_PRIMARY, 0.16)
SWITCH_DISABLED_BG = blend_hex(TEXT_MUTED, SURFACE_BG, 0.35)
SWITCH_DISABLED_THUMB = blend_hex(TEXT_PRIMARY, SURFACE_BG, 0.5)
WARNING_CHIP_BORDER = blend_hex(TEXT_WARNING, SURFACE_CARD, 0.48)
ACCENT_CHIP_BORDER = blend_hex(ACCENT, SURFACE_CARD, 0.56)
STATUS_BAR_BG = blend_hex(SURFACE_BG, SURFACE_CARD, 0.28)
STATUS_BAR_META_TEXT = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.16)
STATUS_BAR_STATE_TEXT = blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.12)

# Settings controls.
MODEL_DIALOG_BG = SURFACE_SETTINGS
MODEL_DIALOG_CARD = SURFACE_SUBTLE
MODEL_DIALOG_CARD_ALT = blend_hex(SURFACE_CARD, SURFACE_ALT, 0.28)
MODEL_DIALOG_EDITOR = blend_hex(MODEL_DIALOG_CARD, TEXT_PRIMARY, 0.035)
MODEL_DIALOG_SOFT = SURFACE_SOFT
MODEL_DIALOG_BORDER = BORDER_SOFT
MODEL_DIALOG_TEXT = TEXT_PRIMARY
MODEL_DIALOG_MUTED = TEXT_SOFT
MODEL_DIALOG_SELECTED = blend_hex(SURFACE_ALT, ACCENT, 0.18)
MODEL_DIALOG_SELECTED_BORDER = blend_hex(ACCENT, TEXT_PRIMARY, 0.14)
MODEL_DIALOG_DISABLED = blend_hex(SURFACE_BG, SURFACE_ALT, 0.46)
MODEL_DIALOG_ACTIVE_BG = STATUS_SUCCESS_BG
MODEL_DIALOG_ACTIVE_TEXT = STATUS_SUCCESS_TEXT
MODEL_DIALOG_PROVIDER_BG = blend_hex(SURFACE_ALT, PROVIDER_ICON_TEXT, 0.20)
MODEL_DIALOG_PROVIDER_TEXT = blend_hex(PROVIDER_ICON_TEXT, TEXT_PRIMARY, 0.22)
MODEL_DIALOG_CHIP_BG = SURFACE_BADGE
MODEL_DIALOG_FIELD_BG = SURFACE_HOVER
MODEL_DIALOG_FIELD_BORDER = blend_hex(BORDER, TEXT_PRIMARY, 0.09)
MODEL_DIALOG_SWITCH_OFF = blend_hex(SURFACE_ALT, TEXT_PRIMARY, 0.10)
MODEL_DIALOG_SWITCH_THUMB = TEXT_PRIMARY
MODEL_DIALOG_SAVED_BG = SAVED_BUTTON_BG
MODEL_EDITOR_BORDER = blend_hex(MODEL_DIALOG_BORDER, TEXT_PRIMARY, 0.08)
SETTINGS_CLOSE_HOVER = blend_hex(TEXT_ERROR, SURFACE_ALT, 0.72)
SETTINGS_CLOSE_PRESSED = blend_hex(TEXT_ERROR, SURFACE_ALT, 0.55)
MODEL_HEADER_BG = blend_hex(MODEL_DIALOG_EDITOR, TEXT_PRIMARY, 0.035)
MODEL_HEADER_BORDER = blend_hex(MODEL_DIALOG_BORDER, TEXT_PRIMARY, 0.07)
MODEL_SELECTED_DISABLED_BG = blend_hex(MODEL_DIALOG_EDITOR, MODEL_DIALOG_DISABLED, 0.45)
MODEL_SELECTED_DISABLED_BORDER = blend_hex(MODEL_DIALOG_SELECTED_BORDER, MODEL_DIALOG_BORDER, 0.40)
MODEL_SELECTED_DISABLED_MARKER = blend_hex(MODEL_DIALOG_SELECTED_BORDER, MODEL_DIALOG_BORDER, 0.28)
MODEL_ACTIVE_BADGE_BORDER = blend_hex(MODEL_DIALOG_ACTIVE_BG, MODEL_DIALOG_ACTIVE_TEXT, 0.18)
MODEL_PROVIDER_BADGE_BORDER = blend_hex(MODEL_DIALOG_PROVIDER_BG, MODEL_DIALOG_PROVIDER_TEXT, 0.16)
MODEL_SWITCH_BORDER = blend_hex(MODEL_DIALOG_SWITCH_OFF, SWITCH_BORDER_TINT, 0.32)
MODEL_SWITCH_HOVER = blend_hex(MODEL_DIALOG_SWITCH_OFF, TEXT_PRIMARY, 0.14)
MODEL_SWITCH_THUMB_BORDER = blend_hex(MODEL_DIALOG_SWITCH_THUMB, SWITCH_THUMB_TINT, 0.24)
MODEL_CHECKBOX_BORDER = blend_hex(MODEL_DIALOG_FIELD_BORDER, TEXT_PRIMARY, 0.22)
MODEL_CHECKBOX_HOVER_BORDER = blend_hex(MODEL_DIALOG_SELECTED_BORDER, TEXT_PRIMARY, 0.18)
MODEL_FIELD_HOVER = blend_hex(MODEL_DIALOG_FIELD_BG, TEXT_PRIMARY, 0.08)
MODEL_CHECKBOX_DISABLED_BORDER = blend_hex(MODEL_DIALOG_FIELD_BORDER, MODEL_DIALOG_BG, 0.45)
MODEL_DISABLED_TEXT = blend_hex(MODEL_DIALOG_MUTED, MODEL_DIALOG_TEXT, 0.18)
MODEL_POPUP_BG = blend_hex(MODEL_DIALOG_CARD, MODEL_DIALOG_FIELD_BG, 0.20)
MODEL_POPUP_BORDER = blend_hex(MODEL_DIALOG_SELECTED_BORDER, MODEL_DIALOG_BORDER, 0.35)
MODEL_POPUP_SELECTION = blend_hex(MODEL_DIALOG_SELECTED, MODEL_DIALOG_FIELD_BG, 0.14)
MODEL_POPUP_SELECTED_BG = blend_hex(MODEL_DIALOG_SELECTED, MODEL_DIALOG_FIELD_BG, 0.18)
MODEL_POPUP_ITEM_HOVER = blend_hex(MODEL_DIALOG_SELECTED, MODEL_DIALOG_FIELD_BG, 0.10)
SETTINGS_ADD_HOVER = blend_hex(MODEL_DIALOG_SELECTED, TEXT_PRIMARY, 0.18)
SETTINGS_DELETE_TEXT = blend_hex(TEXT_ERROR, MODEL_DIALOG_TEXT, 0.2)
SETTINGS_DELETE_BORDER = blend_hex(TEXT_ERROR, MODEL_DIALOG_BORDER, 0.75)
MODEL_INLINE_HOVER = blend_hex(MODEL_DIALOG_SELECTED, MODEL_DIALOG_CARD, 0.14)
MODEL_INLINE_HOVER_BORDER = blend_hex(MODEL_DIALOG_SELECTED_BORDER, MODEL_DIALOG_FIELD_BORDER, 0.42)
MODEL_INLINE_PRESSED = blend_hex(MODEL_DIALOG_SELECTED, MODEL_DIALOG_CARD, 0.22)
MODEL_INLINE_DISABLED = blend_hex(MODEL_DIALOG_SOFT, MODEL_DIALOG_CARD, 0.35)
MODEL_INLINE_DISABLED_BORDER = blend_hex(MODEL_DIALOG_FIELD_BORDER, MODEL_DIALOG_CARD, 0.35)
SLIDER_TRACK = blend_hex(ACCENT, TEXT_PRIMARY, 0.06)
SLIDER_HANDLE = blend_hex(ACCENT, TEXT_PRIMARY, 0.55)
SLIDER_HANDLE_BORDER = blend_hex(ACCENT, TEXT_PRIMARY, 0.75)
SLIDER_HANDLE_HOVER = blend_hex(ACCENT, TEXT_PRIMARY, 0.7)

# Approval dialogs.
APPROVAL_DIALOG_BG = blend_hex(SURFACE_BG, TEXT_PRIMARY, 0.02)
APPROVAL_DIALOG_CARD = SURFACE_SUBTLE
APPROVAL_DIALOG_BORDER = BORDER_SOFT
APPROVAL_DIALOG_SOFT = SURFACE_BADGE
APPROVAL_DIALOG_CODE = blend_hex(CODE_BG, TEXT_PRIMARY, 0.02)

# Composer and popup controls.
POPUP_BG = blend_hex(SURFACE_BG, SURFACE_CARD, 0.66)
POPUP_BORDER = blend_hex(BORDER, TEXT_PRIMARY, 0.12)
POPUP_ITEM_SELECTED = blend_hex(SURFACE_ALT, ACCENT_SOFT, 0.28)
COMPOSER_HOVER_BORDER = blend_hex(COMPOSER_BORDER, TEXT_PRIMARY, 0.18)
COMPOSER_SCROLLBAR = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.06)
SEND_BUTTON_BORDER = blend_hex(SEND_BUTTON_BG, SURFACE_BG, 0.18)
SEND_BUTTON_HOVER_BORDER = blend_hex(SEND_BUTTON_HOVER, SURFACE_BG, 0.20)
SEND_BUTTON_PRESSED = blend_hex(TEXT_PRIMARY, TEXT_SECONDARY, 0.2)
SEND_BUTTON_PRESSED_BORDER = blend_hex(SEND_BUTTON_BG, SURFACE_BG, 0.22)
SEND_BUTTON_DISABLED_TEXT = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.2)
SEND_BUTTON_DISABLED_BORDER = blend_hex(SEND_BUTTON_DISABLED, TEXT_PRIMARY, 0.08)
COMPOSER_ATTACH_HOVER = blend_hex(COMPOSER_BG, SURFACE_ALT, 0.32)
COMPOSER_CHIP_BG = blend_hex(COMPOSER_BG, TEXT_PRIMARY, 0.04)
COMPOSER_CHIP_BORDER = blend_hex(BORDER, TEXT_PRIMARY, 0.08)
COMPOSER_CHIP_HOVER = blend_hex(COMPOSER_BG, SURFACE_ALT, 0.36)
COMPOSER_CHIP_HOVER_BORDER = blend_hex(BORDER, TEXT_PRIMARY, 0.14)
COMPOSER_SETTINGS_TEXT = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.15)
COMPOSER_SETTINGS_HOVER = blend_hex(COMPOSER_BG, SURFACE_ALT, 0.24)
MENTION_POPUP_BORDER = blend_hex(POPUP_BORDER, SURFACE_BG, 0.55)
MENTION_SELECTED_BG = blend_hex(SURFACE_ALT, TEXT_PRIMARY, 0.2)
MENTION_SELECTED_BORDER = blend_hex(TEXT_PRIMARY, SURFACE_ALT, 0.78)
MENTION_SELECTED_TEXT = TEXT_PRIMARY
MENTION_META_TEXT = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.12)
MENTION_SCROLLBAR = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.1)


@lru_cache(maxsize=1)
def build_stylesheet() -> str:
    """Build QSS from the named interface and content palettes."""
    return f"""
    QWidget {{
        background: {SURFACE_BG};
        color: {TEXT_PRIMARY};
        font-family: "{APP_FONT_FAMILY}";
        font-size: 10pt;
    }}

    QMainWindow {{
        background: {SURFACE_BG};
    }}

    QDialog[applicationDialog="true"] {{
        background: {SURFACE_CARD};
        border: 1px solid {BORDER};
    }}

    QDialog[applicationDialog="true"] QLabel,
    QWidget#DialogTitleBar {{
        background: transparent;
        border: none;
    }}

    QWidget#MainWindowHeader,
    QWidget#MainWindowTitleBar,
    QWidget#MainWindowTopBar,
    QWidget#MainWindowTopBarSection,
    QWidget#MainWindowCentral {{
        background: {SURFACE_DIALOG};
    }}

    QWidget#MainWindowTitleBar QToolButton {{
        background: transparent;
        border: none;
        border-radius: 0;
    }}

    QWidget#MainWindowTitleBar QToolButton:hover {{
        background: {SURFACE_ALT};
    }}

    QWidget#MainWindowTitleBar QToolButton#WindowCloseButton:hover {{
        background: {DANGER_BUTTON_BG};
    }}

    QLabel#DialogWindowTitle {{
        background: transparent;
        color: {TEXT_PRIMARY};
        font-weight: 600;
    }}

    QFrame#SidebarCard,
    QFrame#StatusCard,
    QFrame#NoticeCard {{
        background: {SURFACE_CARD};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QFrame#SidebarCard {{
        border: 1px solid {BORDER};
    }}

    QFrame#SidebarCard[panelRole="projects"] {{
        background: {SURFACE_PROJECTS};
    }}

    QWidget#SessionSidebar {{
        background: transparent;
    }}

    QPushButton#SidebarCompactButton {{
        background: {SURFACE_CARD};
        border: 1px solid {BORDER};
        border-radius: {SOFT_RADIUS_MD}px;
        color: {TEXT_MUTED};
        padding: 5px 10px;
    }}

    QPushButton#SidebarCompactButton:hover:enabled {{
        color: {TEXT_PRIMARY};
        background: {SURFACE_HOVER};
    }}

    QPushButton#SidebarCompactButton:disabled {{
        color: {SIDEBAR_DISABLED_TEXT};
        background: {SIDEBAR_DISABLED_BG};
    }}

    QFrame#TranscriptMetaChip {{
    background: transparent;
    border: none;
    }}
    
    QFrame#ApprovalCard,
    QFrame#ApprovalRequestCard {{
        background: {APPROVAL_DIALOG_CARD};
        border: 1px solid {APPROVAL_DIALOG_BORDER};
        border-radius: {SOFT_RADIUS_MD + 2}px;
    }}

    QFrame#ApprovalToolCard {{
        background: {APPROVAL_DIALOG_SOFT};
        border: 1px solid {APPROVAL_DIALOG_BORDER};
        border-radius: {SOFT_RADIUS_MD + 2}px;
    }}

    QLabel#TopStatusChip {{
        color: {TEXT_PRIMARY};
        background: {STATUS_BG};
        border: none;
        border-radius: {SOFT_RADIUS_MD + 6}px;
        padding: 4px 10px;
        font-size: 8.8pt;
        font-weight: 600;
    }}

    QLabel#TopStatusChip[statusState="busy"] {{
        background: {STATUS_BUSY_BG};
    }}

    QLabel#TopStatusChip[statusState="success"] {{
        background: {STATUS_SUCCESS_BG};
        color: {TEXT_CHAT};
    }}

    QLabel#TopStatusChip[statusState="error"] {{
        background: {STATUS_ERROR_BG};
        color: {STATUS_ERROR_TEXT};
    }}

    QLabel#TopStatusChip[statusState="idle"] {{
        color: {TEXT_LABEL};
    }}

    QFrame#UserBubble {{
        background: {USER_MESSAGE_BG};
        border: none;
        border-radius: {SOFT_RADIUS_MD + 4}px;
    }}

    QFrame#ImageAttachmentCard {{
        background: {SURFACE_HOVER};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QLabel#ImageAttachmentThumb {{
        background: {APPROVAL_DIALOG_CARD};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QToolButton#ImageAttachmentRemoveButton {{
        background: {ATTACHMENT_REMOVE_BG};
        border: none;
        border-radius: {SOFT_RADIUS_XS + 6}px;
        padding: 2px;
    }}

    QToolButton#ImageAttachmentRemoveButton:hover {{
        background: {ATTACHMENT_REMOVE_HOVER};
    }}

    QDialog#InfoPopup {{
        background: {SURFACE_CARD};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QDockWidget#ModelSettingsDock {{
        background: {MODEL_DIALOG_BG};
        border: none;
    }}

    QDockWidget#ModelSettingsDock QWidget {{
        background: transparent;
        color: {MODEL_DIALOG_TEXT};
    }}

    QDockWidget#ModelSettingsDock::title {{
        padding: 0px;
        margin: 0px;
        height: 0px;
        background: transparent;
        border: none;
    }}

    QDialog#ModelSettingsDialog {{
        background: {MODEL_DIALOG_BG};
        border: 1px solid {BORDER};
    }}

    QDialog#ModelSettingsDialog QWidget {{
        background: transparent;
        color: {MODEL_DIALOG_TEXT};
    }}

    QDialog#ModelSettingsDialog QLabel {{
        background: transparent;
    }}

    QFrame#ModelSettingsPane {{
        background: {MODEL_DIALOG_CARD_ALT};
        border: 1px solid {MODEL_DIALOG_BORDER};
        border-radius: {SOFT_RADIUS_MD + 4}px;
    }}

    QFrame#ModelSettingsPane[paneRole="editor"] {{
        background: {MODEL_DIALOG_EDITOR};
        border: 1px solid {MODEL_EDITOR_BORDER};
    }}

    QFrame#ModelSettingsHeroCard {{
        background: {MODEL_DIALOG_CARD};
        border: 1px solid {MODEL_DIALOG_BORDER};
        border-radius: {SOFT_RADIUS_MD + 8}px;
    }}

    QLabel#ModelSettingsTitle {{
        color: {MODEL_DIALOG_TEXT};
        font-weight: 760;
        font-size: 13pt;
        padding-left: 1px;
    }}

    QPushButton#SettingsCloseButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: {SOFT_RADIUS_MD + 2}px;
        padding: 0px;
    }}

    QPushButton#SettingsCloseButton:hover {{
        background: {SETTINGS_CLOSE_HOVER};
        border: 1px solid {STATUS_ERROR_TEXT};
    }}

    QPushButton#SettingsCloseButton:pressed {{
        background: {SETTINGS_CLOSE_PRESSED};
    }}

    QLabel#ModelSettingsSubtitle {{
        color: {MODEL_DIALOG_MUTED};
        font-size: 9pt;
        padding-left: 1px;
        padding-bottom: 1px;
    }}

    QLabel#ModelSettingsMeta {{
        color: {MODEL_DIALOG_MUTED};
        font-size: 8.5pt;
        padding-left: 1px;
        background: transparent;
    }}

    QLabel#ModelSettingsChip {{
        color: {MODEL_DIALOG_TEXT};
        background: {MODEL_DIALOG_CHIP_BG};
        border: 1px solid {MODEL_DIALOG_BORDER};
        border-radius: {SOFT_RADIUS_MD + 4}px;
        padding: 3px 8px;
        font-size: 8pt;
        font-weight: 700;
    }}

    QFrame#ModelSettingsFormCard,
    QFrame#ModelSettingsHelperCard {{
        background: {MODEL_DIALOG_CARD};
        border: 1px solid {MODEL_DIALOG_BORDER};
        border-radius: {SOFT_RADIUS_MD + 2}px;
    }}

    QFrame#ModelSettingsSelectedProfileHeader {{
        background: {MODEL_HEADER_BG};
        border: 1px solid {MODEL_HEADER_BORDER};
        border-radius: {SOFT_RADIUS_MD + 2}px;
    }}

    QLabel#ModelSettingsSelectedProfileTitle {{
        color: {MODEL_DIALOG_TEXT};
        background: transparent;
        font-size: 11pt;
        font-weight: 760;
        padding: 0px;
    }}

    QLabel#ModelSettingsSummaryLabel {{
        color: {MODEL_DIALOG_MUTED};
        background: transparent;
        font-size: 8.5pt;
        font-weight: 600;
    }}

    QLabel#ModelSettingsHelperTitle {{
        color: {MODEL_DIALOG_TEXT};
        background: transparent;
        font-size: 9pt;
        font-weight: 700;
    }}

    QScrollArea#ModelSettingsScrollArea,
    QWidget#ModelSettingsEditorContent {{
        background: transparent;
        border: none;
    }}

    QListWidget#ModelProfileList {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 2px;
        outline: none;
    }}

    QLineEdit#ModelSettingsSearchField {{
        background: {MODEL_DIALOG_FIELD_BG};
        border: 1px solid {MODEL_DIALOG_FIELD_BORDER};
        border-radius: {SOFT_RADIUS_MD + 2}px;
        min-height: 28px;
        padding: 3px 8px;
        color: {MODEL_DIALOG_TEXT};
        selection-background-color: {MODEL_DIALOG_SELECTED};
    }}

    QLineEdit#ModelSettingsSearchField:focus {{
        border: 1px solid {MODEL_DIALOG_SELECTED_BORDER};
        background: {MODEL_DIALOG_FIELD_BG};
    }}

    QListWidget#ModelProfileList::item {{
        border: none;
        background: transparent;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 5px 3px;
        margin: 2px 0px;
        color: {MODEL_DIALOG_TEXT};
    }}

    QListWidget#ModelProfileList::item:selected {{
        background: transparent;
        color: {MODEL_DIALOG_TEXT};
    }}

    QWidget#ModelProfileRowCard {{
        background: {MODEL_DIALOG_CARD};
        border: 1px solid {MODEL_DIALOG_BORDER};
        border-radius: {SOFT_RADIUS_MD + 4}px;
    }}

    QWidget#ModelProfileRowCard[disabledProfile="true"] {{
        background: {MODEL_DIALOG_DISABLED};
        border: 1px solid {MODEL_DIALOG_BORDER};
    }}

    QWidget#ModelProfileRowCard[selectedProfile="true"] {{
        background: {MODEL_DIALOG_EDITOR};
        border: 1px solid {MODEL_DIALOG_SELECTED_BORDER};
        border-right: 3px solid {MODEL_DIALOG_SELECTED_BORDER};
    }}

    QWidget#ModelProfileRowCard[selectedProfile="true"][disabledProfile="true"] {{
        background: {MODEL_SELECTED_DISABLED_BG};
        border: 1px solid {MODEL_SELECTED_DISABLED_BORDER};
        border-right: 3px solid {MODEL_SELECTED_DISABLED_MARKER};
    }}

    QFrame#ModelProfileTabRail {{
        background: transparent;
        border: none;
        border-radius: 1px;
        min-width: 3px;
        max-width: 3px;
    }}

    QFrame#ModelProfileTabRail[selectedProfile="true"] {{
        background: {MODEL_DIALOG_SELECTED_BORDER};
    }}

    QLabel#ModelProfileItemTitle {{
        color: {MODEL_DIALOG_TEXT};
        font-size: 9.5pt;
        font-weight: 700;
        background: transparent;
        padding: 0px;
    }}

    QLabel#ModelProfileItemMeta {{
        color: {MODEL_DIALOG_MUTED};
        font-size: 8.5pt;
        background: transparent;
        padding: 0px;
    }}

    QLabel#ModelProfileItemBadge {{
        background: {MODEL_DIALOG_SOFT};
        color: {MODEL_DIALOG_MUTED};
        border: 1px solid {MODEL_DIALOG_BORDER};
        border-radius: {SOFT_RADIUS_MD + 4}px;
        padding: 1px 6px;
        font-size: 7.8pt;
        font-weight: 700;
    }}

    QLabel#ModelProfileItemBadge[badgeVariant="active"] {{
        background: {MODEL_DIALOG_ACTIVE_BG};
        color: {MODEL_DIALOG_ACTIVE_TEXT};
        border: 1px solid {MODEL_ACTIVE_BADGE_BORDER};
    }}

    QLabel#ModelProfileItemBadge[badgeVariant="provider"] {{
        background: {MODEL_DIALOG_PROVIDER_BG};
        color: {MODEL_DIALOG_PROVIDER_TEXT};
        border: 1px solid {MODEL_PROVIDER_BADGE_BORDER};
    }}

    QCheckBox#ModelProfileEnabledSwitch {{
        background: {MODEL_DIALOG_SWITCH_OFF};
        border-radius: 10px;
        border: 1px solid {MODEL_SWITCH_BORDER};
        padding: 0px;
        spacing: 0px;
    }}

    QCheckBox#ModelProfileEnabledSwitch:hover {{
        background: {MODEL_SWITCH_HOVER};
    }}

    QCheckBox#ModelProfileEnabledSwitch:checked {{
        background: {ACCENT};
        border: 1px solid {ACCENT};
    }}

    QCheckBox#ModelProfileEnabledSwitch::indicator {{
        width: 14px;
        height: 14px;
        border-radius: 7px;
        background: {MODEL_DIALOG_SWITCH_THUMB};
        border: 1px solid {MODEL_SWITCH_THUMB_BORDER};
    }}

    QCheckBox#ModelProfileEnabledSwitch::indicator:unchecked {{
        subcontrol-origin: margin;
        subcontrol-position: left center;
        margin-left: 2px;
    }}

    QCheckBox#ModelProfileEnabledSwitch::indicator:checked {{
        subcontrol-origin: margin;
        subcontrol-position: right center;
        margin-right: 2px;
        background: {SURFACE_BG};
    }}
    
    QCheckBox#ModelSupportsImagesCheckbox,
    QCheckBox#LlmApiModeCheckbox {{
        background: transparent;
        border: none;
        padding: 0px;
        spacing: 6px;
        color: {MODEL_DIALOG_TEXT};
    }}

    QCheckBox#ModelSupportsImagesCheckbox::indicator,
    QCheckBox#LlmApiModeCheckbox::indicator {{
        width: 15px;
        height: 15px;
        border-radius: 4px;
        border: 1px solid {MODEL_CHECKBOX_BORDER};
        background: {SURFACE_HOVER};
    }}

    QCheckBox#ModelSupportsImagesCheckbox::indicator:hover,
    QCheckBox#LlmApiModeCheckbox::indicator:hover {{
        border: 1px solid {MODEL_CHECKBOX_HOVER_BORDER};
        background: {MODEL_FIELD_HOVER};
    }}

    QCheckBox#ModelSupportsImagesCheckbox::indicator:checked,
    QCheckBox#LlmApiModeCheckbox::indicator:checked {{
        border: 1px solid {MODEL_DIALOG_SELECTED_BORDER};
        background: {SURFACE_HOVER};
    }}

    QCheckBox#ModelSupportsImagesCheckbox::indicator:checked:hover,
    QCheckBox#LlmApiModeCheckbox::indicator:checked:hover {{
        border: 1px solid {MODEL_DIALOG_SELECTED_BORDER};
        background: {MODEL_FIELD_HOVER};
    }}

    QCheckBox#ModelSupportsImagesCheckbox::indicator:disabled,
    QCheckBox#LlmApiModeCheckbox::indicator:disabled {{
        border: 1px solid {MODEL_CHECKBOX_DISABLED_BORDER};
        background: {SURFACE_SOFT};
    }}

    QDialog#ModelSettingsDialog QLineEdit,
    QDialog#ModelSettingsDialog QComboBox {{
        background: {MODEL_DIALOG_FIELD_BG};
        border: 1px solid {MODEL_DIALOG_FIELD_BORDER};
        border-radius: {SOFT_RADIUS_MD + 2}px;
        min-height: 28px;
        padding: 3px 8px;
        color: {MODEL_DIALOG_TEXT};
        selection-background-color: {MODEL_DIALOG_SELECTED};
    }}

    QDialog#ModelSettingsDialog QLineEdit:focus,
    QDialog#ModelSettingsDialog QComboBox:focus {{
        border: 1px solid {MODEL_DIALOG_SELECTED_BORDER};
    }}

    QDialog#ModelSettingsDialog QLineEdit:disabled,
    QDialog#ModelSettingsDialog QComboBox:disabled {{
        background: {MODEL_DIALOG_SOFT};
        color: {MODEL_DISABLED_TEXT};
    }}

    QDialog#ModelSettingsDialog QComboBox::drop-down {{
        border: none;
        width: 18px;
        padding-right: 2px;
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo {{
        padding-right: 10px;
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo::drop-down {{
        border: none;
        width: 0px;
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo::down-arrow {{
        image: none;
        width: 0px;
        height: 0px;
    }}

    QFrame#ModelSettingsModelPopup {{
        background: {MODEL_POPUP_BG};
        border: 1px solid {MODEL_POPUP_BORDER};
        border-radius: {SOFT_RADIUS_MD + 4}px;
    }}

    QListWidget#ModelSettingsModelPopupList {{
        background: transparent;
        border: none;
        outline: 0px;
        color: {MODEL_DIALOG_TEXT};
        selection-background-color: {MODEL_POPUP_SELECTION};
        selection-color: {MODEL_DIALOG_TEXT};
    }}

    QListWidget#ModelSettingsModelPopupList::item {{
        min-height: 26px;
        padding: 4px 10px;
        margin: 1px 4px;
        border-radius: {SOFT_RADIUS_SM}px;
        background: transparent;
    }}

    QListWidget#ModelSettingsModelPopupList::item:selected {{
        background: {MODEL_POPUP_SELECTED_BG};
        color: {TEXT_PRIMARY};
    }}

    QListWidget#ModelSettingsModelPopupList::item:hover {{
        background: {MODEL_POPUP_ITEM_HOVER};
    }}

    QDialog#ModelSettingsDialog QComboBox QAbstractItemView {{
        background: {MODEL_DIALOG_CARD};
        border: 1px solid {MODEL_DIALOG_BORDER};
        selection-background-color: {MODEL_DIALOG_SELECTED};
        color: {MODEL_DIALOG_TEXT};
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView {{
        background: {MODEL_POPUP_BG};
        border: 1px solid {MODEL_POPUP_BORDER};
        border-radius: {SOFT_RADIUS_MD + 4}px;
        padding: 6px 4px;
        outline: 0px;
        selection-background-color: {MODEL_POPUP_SELECTION};
        selection-color: {MODEL_DIALOG_TEXT};
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView::item {{
        min-height: 26px;
        padding: 4px 10px;
        margin: 1px 4px;
        border-radius: {SOFT_RADIUS_SM}px;
        background: transparent;
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView::item:selected {{
        background: {MODEL_POPUP_SELECTED_BG};
        color: {TEXT_PRIMARY};
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView::item:hover {{
        background: {MODEL_POPUP_ITEM_HOVER};
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 4px 2px 4px 6px;
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView QScrollBar::handle:vertical {{
        background: {SCROLLBAR_HANDLE};
        border-radius: 4px;
        min-height: 28px;
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView QScrollBar::handle:vertical:hover {{
        background: {SCROLLBAR_HANDLE_HOVER};
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView QScrollBar::add-line:vertical,
    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView QScrollBar::sub-line:vertical,
    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView QScrollBar::add-page:vertical,
    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView QScrollBar::sub-page:vertical {{
        background: transparent;
        border: none;
        height: 0px;
    }}

    QLabel#ModelSettingsFieldLabel {{
        background: transparent;
        color: {MODEL_DIALOG_TEXT};
        border: none;
        min-height: 18px;
        padding: 0px;
        font-size: 9pt;
        font-weight: 700;
    }}

    QPushButton#SettingsAddButton,
    QPushButton#SettingsDeleteButton {{
        min-height: 30px;
        font-size: 9pt;
        font-weight: 700;
        border-radius: {SOFT_RADIUS_MD + 2}px;
        border: 1px solid {MODEL_DIALOG_FIELD_BORDER};
    }}

    QPushButton#SettingsAddButton {{
        background: {MODEL_DIALOG_SELECTED};
        color: {MODEL_DIALOG_TEXT};
    }}

    QPushButton#SettingsAddButton:hover {{
        background: {SETTINGS_ADD_HOVER};
    }}

    QPushButton#SettingsDeleteButton {{
        background: {MODEL_DIALOG_CARD};
        color: {SETTINGS_DELETE_TEXT};
        border: 1px solid {SETTINGS_DELETE_BORDER};
    }}

    QPushButton#ModelSettingsInlineButton,
    QToolButton#ModelSettingsInlineToolButton {{
        background: {MODEL_DIALOG_CARD};
        border: 1px solid {MODEL_DIALOG_FIELD_BORDER};
        border-radius: {SOFT_RADIUS_MD + 2}px;
        color: {MODEL_DIALOG_TEXT};
    }}

    QPushButton#ModelSettingsInlineButton {{
        min-height: 28px;
        padding: 3px 10px;
        font-size: 9pt;
        font-weight: 600;
    }}

    QToolButton#ModelSettingsInlineToolButton {{
        min-width: 26px;
        max-width: 26px;
        min-height: 26px;
        max-height: 26px;
        padding: 0px;
    }}

    QPushButton#ModelSettingsInlineButton:hover,
    QToolButton#ModelSettingsInlineToolButton:hover {{
        background: {MODEL_INLINE_HOVER};
        border: 1px solid {MODEL_INLINE_HOVER_BORDER};
    }}

    QPushButton#ModelSettingsInlineButton:pressed,
    QToolButton#ModelSettingsInlineToolButton:pressed {{
        background: {MODEL_INLINE_PRESSED};
    }}

    QToolButton#ModelSettingsInlineToolButton:disabled {{
        background: {MODEL_INLINE_DISABLED};
        border: 1px solid {MODEL_INLINE_DISABLED_BORDER};
    }}

    QLabel#ModelSettingsHintText {{
        color: {MODEL_DIALOG_MUTED};
        background: transparent;
        font-size: 8.2pt;
        padding-left: 2px;
    }}

    QSplitter::handle {{
        background: transparent;
    }}

    /* Round thumb over the accent track for the Test-tab SESSION_SIZE slider.
       Fusion paints the filled part of the groove with the highlight color;
       keep that blue line and add a circular handle on top of it. */
    QSlider#SessionSizeSlider {{
        min-height: 22px;
    }}

    QSlider#SessionSizeSlider::groove:horizontal {{
        height: 4px;
        border-radius: 2px;
        background: {MODEL_DIALOG_FIELD_BORDER};
    }}

    QSlider#SessionSizeSlider::sub-page:horizontal {{
        background: {SLIDER_TRACK};
        border-radius: 2px;
    }}

    QSlider#SessionSizeSlider::handle:horizontal {{
        background: {SLIDER_HANDLE};
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
        border: 1px solid {SLIDER_HANDLE_BORDER};
    }}

    QSlider#SessionSizeSlider::handle:horizontal:hover {{
        background: {SLIDER_HANDLE_HOVER};
    }}

    QDialog#ModelSettingsDialog QDialogButtonBox {{
        border-top: 1px solid {MODEL_DIALOG_BORDER};
        padding-top: 6px;
    }}

    QDialog#ModelSettingsDialog QPushButton#PrimaryButton {{
        background: {SAVE_BUTTON_BG};
        color: {SAVE_BUTTON_TEXT};
        border: 1px solid {SAVE_BUTTON_BG};
        border-radius: {SOFT_RADIUS_MD + 4}px;
        font-weight: 700;
        padding: 3px 10px;
    }}

    QDialog#ModelSettingsDialog QPushButton#PrimaryButton:hover {{
        background: {SAVE_BUTTON_HOVER};
        border-color: {SAVE_BUTTON_HOVER};
    }}

    QDialog#ModelSettingsDialog QPushButton#PrimaryButton[savedState="true"] {{
        background: {MODEL_DIALOG_SAVED_BG};
        color: {SAVED_BUTTON_TEXT};
        border-color: {MODEL_DIALOG_SAVED_BG};
    }}

    QDialog#ModelSettingsDialog QPushButton#PrimaryButton[savedState="true"]:hover {{
        background: {SAVED_BUTTON_HOVER};
        border-color: {SAVED_BUTTON_HOVER};
    }}

    QDialog#ModelSettingsDialog QPushButton {{
        color: {MODEL_DIALOG_TEXT};
    }}

    QDialog#ApprovalDialog {{
        background: {APPROVAL_DIALOG_BG};
        border: none;
    }}

    QDialog#ApprovalDialog QWidget,
    QDialog#ApprovalDialog QLabel {{
        background: transparent;
        color: {TEXT_PRIMARY};
    }}

    QDialog#ApprovalDialog QScrollArea {{
        background: transparent;
        border: none;
    }}

    QDialog#ApprovalDialog QDialogButtonBox {{
        border-top: 1px solid {APPROVAL_DIALOG_BORDER};
        padding-top: 10px;
    }}

    QFrame#ApprovalRequestCard QWidget,
    QFrame#ApprovalRequestCard QLabel {{
        background: transparent;
    }}

    QFrame#ApprovalRequestCard QScrollArea {{
        background: transparent;
        border: none;
    }}

    QWidget#ChatWorkspace,
    QWidget#CenteredComposerRow,
    QWidget#ChatTranscript,
    QScrollArea#TranscriptScrollArea,
    QWidget#TranscriptViewport,
    QWidget#TranscriptContainer,
    QWidget#TranscriptColumn,
    QWidget#ConversationTurn,
    QWidget#AssistantContent,
    QWidget#ToolGroupHeaderRow,
    QWidget#ToolGroupContainer,
    QWidget#ToolCardHeaderRow,
    QWidget#ToolArgsContainer,
    QWidget#RunStatsRow,
    QWidget#CodeBlock {{
        background: {SURFACE_DIALOG};
        border: none;
    }}

    QWidget#ToolGroupHeaderRow QLabel,
    QWidget#ToolCardHeaderRow QLabel,
    QWidget#RunStatsRow QLabel,
    QLabel#TranscriptMeta,
    QFrame#TranscriptMetaChip {{
        background: {SURFACE_DIALOG};
    }}

    QFrame#UserChoiceCard {{
        background: {SURFACE_RAISED};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QFrame#TranscriptRow,
    QFrame#ToolRow,
    QFrame#ToolGroupFrame,
    QFrame#FlatNoticeRow,
    QFrame#InlineStatusRow {{
        background: {SURFACE_DIALOG};
        border: none;
    }}

    QFrame#TranscriptRow[messageRole="assistant"] {{
        background: {AI_MESSAGE_BG};
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QLabel#SectionTitle {{
        color: {TEXT_PRIMARY};
        font-weight: 600;
        font-size: 12pt;
    }}

    QLabel#InspectorSectionTitle {{
        color: {TEXT_PRIMARY};
        font-weight: 600;
        font-size: 12pt;
        background: transparent;
        border: none;
    }}

    QLabel#ModelSettingsSectionTitle {{
        color: {TEXT_PRIMARY};
        font-weight: 600;
        font-size: 10pt;
    }}

    QLabel#SidebarSectionTitle {{
        background: transparent;
        border: none;
        color: {TEXT_LABEL};
        font-weight: 600;
        font-size: 11pt;
        padding-left: 2px;
    }}

    QLabel#SidebarEmptyState {{
        color: {TEXT_DETAIL};
        font-size: 9.2pt;
        background: transparent;
        border: none;
    }}

    QFrame#SidebarEmptyStateCard {{
        background: transparent;
        border: none;
    }}

    QLabel#TranscriptRole {{
        color: {TEXT_MUTED};
        font-weight: normal;
        font-size: 9.5pt;
    }}

    QLabel#TranscriptBody {{
        color: {TEXT_CHAT};
        font-size: 11pt;
        background: transparent;
    }}

    QLabel#ComposerCapabilityBadge {{
        color: {TEXT_SOFT};
        background: {SURFACE_SOFT};
        border: none;
        border-radius: {SOFT_RADIUS_MD + 8}px;
        padding: 0px;
        font-size: 8.6pt;
        font-weight: 600;
    }}

    QLabel#ComposerNoticeLabel {{
        color: {TEXT_MUTED};
        background: transparent;
        font-size: 8.8pt;
        padding-left: 2px;
    }}

    QLabel#ComposerNoticeLabel[severity="warning"] {{
        color: {TEXT_WARNING};
    }}

    QLabel#ComposerNoticeLabel[severity="error"] {{
        color: {TEXT_ERROR};
    }}

    QLabel#ComposerNoticeLabel[severity="success"] {{
        color: {TEXT_SUCCESS};
    }}

    QLabel#UserChoiceCardTitle {{
        color: {TEXT_SUBTITLE};
        font-size: 8.8pt;
        font-weight: 650;
        background: transparent;
    }}

    QLabel#UserChoiceCardQuestion {{
        color: {TEXT_QUESTION};
        font-size: 9pt;
        background: transparent;
    }}

    QLabel#UserChoiceCardHint {{
        color: {TEXT_HINT};
        font-size: 8.8pt;
        background: transparent;
    }}

    QLabel#ApprovalCardTitle {{
        color: {TEXT_PRIMARY};
        font-size: 10.8pt;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#ApprovalCardSummary {{
        color: {TEXT_SUMMARY};
        font-size: 9.7pt;
        background: transparent;
    }}

    QLabel#ApprovalCardImpacts {{
        color: {TEXT_DETAIL};
        font-size: 8.9pt;
        background: transparent;
    }}

    QLabel#ApprovalRiskBadge {{
        background: {STATUS_BG};
        color: {TEXT_HINT};
        border: none;
        border-radius: {SOFT_RADIUS_XS + 6}px;
        padding: 2px 8px;
        font-size: 8.1pt;
        font-weight: 700;
    }}

    QLabel#ApprovalRiskBadge[riskLevel="low"] {{
        background: {STATUS_SUCCESS_BG};
        color: {STATUS_SUCCESS_TEXT};
    }}

    QLabel#ApprovalRiskBadge[riskLevel="medium"] {{
        background: {STATUS_WARNING_BG};
        color: {STATUS_WARNING_TEXT};
    }}

    QLabel#ApprovalRiskBadge[riskLevel="high"],
    QLabel#ApprovalRiskBadge[riskLevel="critical"] {{
        background: {STATUS_ERROR_BG};
        color: {STATUS_ERROR_TEXT};
    }}

    QLabel#ApprovalToolTitle {{
        color: {TEXT_PRIMARY};
        font-size: 9.7pt;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#ApprovalToolSubtitle {{
        color: {TEXT_META};
        font-size: 8.9pt;
        background: transparent;
    }}

    QFrame#ApprovalRequestCard QPushButton#PrimaryButton,
    QDialog#ApprovalDialog QPushButton#PrimaryButton {{
        min-height: 28px;
        padding: 3px 11px;
        border-radius: {SOFT_RADIUS_MD + 2}px;
        background: {ACCENT_BUTTON_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {ACCENT_BUTTON_BORDER};
        font-weight: 700;
    }}

    QFrame#ApprovalRequestCard QPushButton#PrimaryButton:hover,
    QDialog#ApprovalDialog QPushButton#PrimaryButton:hover {{
        background: {ACCENT_BUTTON_HOVER};
    }}

    QFrame#ApprovalRequestCard QPushButton#SecondaryButton,
    QDialog#ApprovalDialog QPushButton#SecondaryButton {{
        min-height: 28px;
        padding: 3px 11px;
        border-radius: {SOFT_RADIUS_MD + 2}px;
        background: {APPROVAL_DIALOG_SOFT};
        color: {TEXT_PRIMARY};
        border: 1px solid {APPROVAL_DIALOG_BORDER};
        font-weight: 600;
    }}

    QFrame#ApprovalRequestCard QPushButton#DangerButton,
    QDialog#ApprovalDialog QPushButton#DangerButton {{
        min-height: 28px;
        padding: 3px 11px;
        border-radius: {SOFT_RADIUS_MD + 2}px;
        background: {DANGER_BUTTON_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {DANGER_BUTTON_BORDER};
        font-weight: 700;
    }}

    QPlainTextEdit#ApprovalDetailView {{
        background: {APPROVAL_DIALOG_CODE};
        color: {TEXT_PRIMARY};
        border: 1px solid {APPROVAL_DIALOG_BORDER};
        border-radius: {SOFT_RADIUS_MD + 2}px;
        padding: 10px;
        selection-background-color: {TEXT_SELECTION_BG};
    }}

    QPlainTextEdit#ApprovalArgsView,
    QPlainTextEdit#InlineCodeView {{
        background: {APPROVAL_DIALOG_CODE};
        color: {CODE_TEXT};
        border: 1px solid {APPROVAL_DIALOG_BORDER};
        border-radius: {SOFT_RADIUS_MD + 2}px;
        padding: 8px;
        selection-background-color: {TEXT_SELECTION_BG};
    }}

    QLabel#MutedText,
    QLabel#MetaText {{
        color: {TEXT_MUTED};
        background: transparent;
    }}

    QLabel#InspectorMetaText {{
        color: {TEXT_MUTED};
        background: transparent;
        border: none;
    }}

    QLabel#InspectorFieldText,
    QLabel#InspectorValueText {{
        background: transparent;
        border: none;
    }}

    QLabel#MetaText[severity="error"] {{
        color: {TEXT_ERROR};
    }}

    QLabel#WarningText {{
        color: {TEXT_WARNING};
    }}

    QLabel#StatusLabel {{
        color: {TEXT_PRIMARY};
        font-weight: 600;
    }}

    QFrame#InlineStatusRow {{
        border: none;
    }}

    QFrame#InlineStatusRow[phase="active"] QLabel#TranscriptMeta,
    QFrame#InlineStatusRow[phase="reviewing"] QLabel#TranscriptMeta {{
        color: {STATUS_REVIEW_TEXT};
    }}

    QFrame#InlineStatusRow[phase="success"] QLabel#TranscriptMeta {{
        color: {TEXT_SUCCESS};
    }}

    QFrame#InlineStatusRow[phase="waiting"] QLabel#TranscriptMeta {{
        color: {STATUS_WAITING_TEXT};
    }}

    QFrame#InlineStatusRow QLabel#TranscriptMeta {{
        color: {TEXT_MUTED};
    }}

    #InlineStatusSpinner {{
        background: {SURFACE_DIALOG};
        border: none;
        padding: 0px;
    }}

    QTabWidget::pane {{
        border: 1px solid {BORDER_SOFT};
        border-radius: {SOFT_RADIUS_MD + 4}px;
        background: transparent;
        top: -1px;
    }}

    QTabBar::tab {{
        background: transparent;
        border: 1px solid {BORDER_SOFT};
        border-bottom: 1px solid {BORDER_SOFT};
        padding: 3px 10px;
        margin-right: 4px;
        margin-bottom: 2px;
        color: {TEXT_MUTED};
        border-radius: {SOFT_RADIUS_SM}px;
        font-size: 8.5pt;
    }}

    QTabBar::tab:selected {{
        background: {SURFACE_ALT};
        color: {TEXT_PRIMARY};
        border: 1px solid {MODEL_DIALOG_SELECTED_BORDER};
        border-bottom-left-radius: {SOFT_RADIUS_SM}px;
        border-bottom-right-radius: {SOFT_RADIUS_SM}px;
    }}

    QToolBar {{
        background: {SURFACE_CARD};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        spacing: 3px;
        padding: 1px 3px;
    }}

    QToolButton,
    QPushButton {{
        background: {SURFACE_ALT};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 3px 6px;
        color: {TEXT_PRIMARY};
    }}

    QToolButton:hover,
    QPushButton:hover {{
        background: {BUTTON_HOVER};
        border: none;
    }}

    QToolButton:pressed,
    QPushButton:pressed {{
        background: {BUTTON_PRESSED};
    }}

    QToolButton#MainWindowToolButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QToolButton#MainWindowToolButton:hover {{
        background: {SURFACE_ALT};
    }}

    QToolButton#MainWindowToolButton:pressed {{
        background: {HEADER_BUTTON_PRESSED};
    }}

    QPushButton#PrimaryButton {{
        background: {BUTTON_BG};
        color: {TEXT_PRIMARY};
        border: none;
        font-weight: 600;
    }}

    QPushButton#SecondaryButton {{
        background: {SURFACE_SOFT};
        color: {TEXT_PRIMARY};
        border: none;
        font-weight: 600;
    }}

    QPushButton#DangerButton {{
        background: {DANGER_BUTTON_BG};
        color: {TEXT_PRIMARY};
        border: none;
        font-weight: 600;
    }}

    QToolButton#SidebarGhostButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 4px;
    }}

    QToolButton#SidebarGhostButton:hover {{
        background: {STATUS_BG};
    }}

    QToolButton#SidebarGhostButton:disabled {{
        background: transparent;
        border: 1px solid transparent;
    }}

    QPushButton:disabled,
    QToolButton:disabled {{
        color: {TEXT_DIM};
        background: {SURFACE_ALT};
        border: none;
    }}

    QToolButton#DisclosureButton,
    QPushButton#DisclosureButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 0px 2px;
        min-height: 18px;
        font-size: 9pt;
        color: {TEXT_MUTED};
        text-align: left;
    }}

    QToolButton#DisclosureButton:hover,
    QPushButton#DisclosureButton:hover {{
        background: transparent;
        color: {DISCLOSURE_HOVER_TEXT};
    }}

    QFrame#ToolExpandablePanel {{
        background: {SURFACE_RAISED};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 5px 8px;
    }}

    QFrame#ToolExpandablePanel[variant="thoughts"] {{
        background: {SURFACE_DIALOG};
        border: none;
        border-radius: 0px;
        padding: 0px;
    }}

    QFrame#ToolExpandablePanel[variant="diff"] {{
        background: {SURFACE_DIALOG};
        border: none;
        border-radius: 0px;
        padding: 2px 0px 0px 0px;
    }}

    QFrame#CliExecPanel {{
        background: {SURFACE_RAISED};
        border: 1px solid {CODE_PANEL_BORDER};
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QLabel#CliExecHeader {{
        background: transparent;
        color: {CODE_HEADER_TEXT};
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9.3pt;
        font-weight: 500;
    }}

    QLabel#ToolSubtitle {{
        background: transparent;
        color: {TEXT_SUBTITLE};
        font-size: 8.9pt;
        padding-left: 13px;
        padding-bottom: 1px;
    }}

    QLabel#ToolActionLabel {{
        background: transparent;
        color: {TEXT_SUMMARY};
        font-size: 10.15pt;
        font-weight: 500;
        padding: 1px 0px;
    }}

    QLabel#ToolActionLabel[toolRole="edit"] {{
        color: {TEXT_SUMMARY};
    }}

    QLabel#ToolActionLabel[phase="active"] {{
        color: {TEXT_SUMMARY};
    }}

    QLabel#ToolActionLabel[phase="success"] {{
        color: {TEXT_SUMMARY};
    }}

    QLabel#ToolActionLabel[phase="error"] {{
        color: {STATUS_ERROR_TEXT};
    }}

    QLabel#ToolPhaseBadge {{
        background: {SURFACE_SOFT};
        color: {TEXT_HINT};
        border: none;
        border-radius: {SOFT_RADIUS_XS + 5}px;
        padding: 2px 7px;
        font-size: 8.1pt;
        font-weight: 600;
    }}

    QLabel#ToolPhaseBadge[variant="pending"] {{
        background: {APPROVAL_DIALOG_SOFT};
        color: {TEXT_META};
    }}

    QLabel#ToolPhaseBadge[variant="active"] {{
        background: {BADGE_ACTIVE_BG};
        color: {BADGE_ACTIVE_TEXT};
    }}

    QLabel#ToolPhaseBadge[variant="success"] {{
        background: transparent;
        color: {TEXT_SUCCESS};
        padding: 0px;
        border-radius: 0px;
        font-size: 9.2pt;
        font-weight: 700;
    }}

    QLabel#ToolPhaseBadge[variant="error"] {{
        background: {BADGE_ERROR_BG};
        color: {TEXT_PRIMARY};
    }}

    QPlainTextEdit#CliExecOutput {{
        background: {SURFACE_RAISED};
        color: {CODE_TEXT};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 7px 10px 8px 10px;
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
    }}

    QFrame#DiffPanel {{
        background: {SURFACE_RAISED};
        border: 1px solid {DIFF_PANEL_BORDER};
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QWidget#DiffHeaderRow {{
        background: {SURFACE_RAISED};
        border-top-left-radius: {SOFT_RADIUS_SM}px;
        border-top-right-radius: {SOFT_RADIUS_SM}px;
    }}

    QLabel#DiffHeaderPath {{
        background: transparent;
        color: {DIFF_HEADER_TEXT};
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9.05pt;
        font-weight: 500;
    }}

    QLabel#DiffStatAdded {{
        background: transparent;
        color: {DIFF_ADDED_TEXT};
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
        font-weight: 600;
    }}

    QLabel#DiffStatRemoved {{
        background: transparent;
        color: {DIFF_REMOVED_TEXT};
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
        font-weight: 600;
    }}

    QPlainTextEdit#DiffCodeView {{
        background: {SURFACE_RAISED};
        color: {CODE_TEXT};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 6px 10px 9px 10px;
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
    }}

    QWidget#ToolExpandableContent {{
        background: {SURFACE_RAISED};
        border: none;
    }}

    QWidget#ToolExpandableContent[variant="thoughts"] {{
        background: {SURFACE_DIALOG};
        border: none;
    }}

    QWidget#ToolExpandableContent[variant="diff"] {{
        background: {SURFACE_DIALOG};
        border: none;
    }}

    QPushButton#ToolExpandableToggle {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 5px 8px;
        min-height: 20px;
        font-size: 9.2pt;
        font-weight: 400;
        color: {TEXT_META};
        text-align: left;
    }}

    QPushButton#ToolExpandableToggle:hover {{
        background: {TOOL_PANEL_HOVER};
        color: {TOOL_TOGGLE_HOVER_TEXT};
    }}

    QPushButton#ToolExpandableToggle:checked {{
        background: {TOOL_PANEL_HOVER};
        color: {TEXT_PRIMARY};
    }}

    QPushButton#ToolExpandableToggle[variant="thoughts"] {{
        background: transparent;
        border: none;
        border-radius: 0px;
        padding: 0px 0px 3px 0px;
        min-height: 16px;
        font-size: 8.8pt;
        font-weight: 400;
        color: {SCROLLBAR_HANDLE};
        text-align: left;
    }}

    QPushButton#ToolExpandableToggle[variant="thoughts"]:hover,
    QPushButton#ToolExpandableToggle[variant="thoughts"]:checked {{
        background: transparent;
        color: {SCROLLBAR_HANDLE_HOVER};
    }}

    QPushButton#ToolExpandableToggle[variant="diff"] {{
        background: transparent;
        border: none;
        border-radius: 0px;
        padding: 3px 0px 4px 0px;
        min-height: 18px;
        font-size: 9.2pt;
        font-weight: 500;
        color: {DIFF_TOGGLE_TEXT};
        text-align: left;
    }}

    QPushButton#ToolExpandableToggle[variant="diff"]:hover,
    QPushButton#ToolExpandableToggle[variant="diff"]:checked {{
        background: transparent;
        color: {DIFF_TOGGLE_CHECKED_TEXT};
    }}

    QToolButton#CodeCopyButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 2px 6px;
        color: {TEXT_META};
        font-size: 8.8pt;
    }}

    QToolButton#CodeCopyButton:hover {{
        background: {TOOL_PANEL_HOVER};
        color: {TEXT_PRIMARY};
    }}

    QPlainTextEdit,
    QTextBrowser,
    QListView,
    QListWidget,
    QTreeWidget {{
        background: {TREE_BG};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        selection-background-color: {TREE_SELECTION_BG};
        selection-color: {TEXT_PRIMARY};
    }}

    QListView#SessionListView {{
        background: transparent;
        border: none;
        outline: none;
        padding: 0px;
    }}

    /* ---- Composer pill ---- */
    QFrame#ComposerPill {{
        background: {COMPOSER_BG};
        border: 1px solid {COMPOSER_BORDER};
        border-radius: 18px;
    }}

    QFrame#ComposerPill:hover {{
        border: 1px solid {COMPOSER_HOVER_BORDER};
    }}

    QPushButton#UserChoiceOptionButton,
    QPushButton#UserChoiceCustomButton {{
        background: {BUTTON_BG};
        border: 1px solid {APPROVAL_DIALOG_BORDER};
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 7px 10px;
        color: {TEXT_PRIMARY};
        text-align: center;
        font-size: 9.1pt;
        font-weight: 650;
    }}

    QPushButton#UserChoiceOptionButton:hover,
    QPushButton#UserChoiceCustomButton:hover {{
        background: {CHOICE_BUTTON_HOVER};
        border: 1px solid {CHOICE_BUTTON_HOVER_BORDER};
    }}

    QPushButton#UserChoiceOptionButton:pressed,
    QPushButton#UserChoiceCustomButton:pressed {{
        background: {CHOICE_BUTTON_PRESSED};
    }}

    QPushButton#UserChoiceOptionButton[recommended="true"] {{
        background: {ACCENT_BUTTON_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {ACCENT_BUTTON_BORDER};
    }}

    QPushButton#UserChoiceOptionButton:disabled,
    QPushButton#UserChoiceCustomButton:disabled {{
        background: {CHOICE_BUTTON_DISABLED_BG};
        color: {TEXT_SOFT};
    }}

    QPlainTextEdit#ComposerEdit {{
        background: transparent;
        border: none;
        border-radius: 10px;
        padding: 4px 2px 2px 2px;
        font-size: 10.5pt;
    }}

    QPlainTextEdit#ComposerEdit:focus {{
        background: transparent;
        border: none;
    }}

    QPlainTextEdit#ComposerEdit QScrollBar:vertical {{
        background: transparent;
        border: none;
        width: 6px;
        margin: 2px 0 2px 0;
    }}

    QPlainTextEdit#ComposerEdit QScrollBar::handle:vertical {{
        background: {COMPOSER_SCROLLBAR};
        border-radius: {SOFT_RADIUS_XS}px;
        min-height: 16px;
    }}

    QPlainTextEdit#ComposerEdit QScrollBar::add-line:vertical,
    QPlainTextEdit#ComposerEdit QScrollBar::sub-line:vertical,
    QPlainTextEdit#ComposerEdit QScrollBar::add-page:vertical,
    QPlainTextEdit#ComposerEdit QScrollBar::sub-page:vertical {{
        background: transparent;
        border: none;
        height: 0px;
    }}

    QPushButton#ComposerSendButton {{
        background: {SEND_BUTTON_BG};
        color: {SURFACE_BG};
        border: 1px solid {SEND_BUTTON_BORDER};
        border-radius: 15px;
        min-width: 32px;
        max-width: 32px;
        min-height: 32px;
        max-height: 32px;
        padding: 0px;
        font-weight: 700;
    }}

    QPushButton#ComposerSendButton:hover {{
        background: {SEND_BUTTON_HOVER};
        border: 1px solid {SEND_BUTTON_HOVER_BORDER};
    }}

    QPushButton#ComposerSendButton:pressed {{
        background: {SEND_BUTTON_PRESSED};
        border: 1px solid {SEND_BUTTON_PRESSED_BORDER};
    }}

    QPushButton#ComposerSendButton:disabled {{
        background: {SEND_BUTTON_DISABLED};
        color: {SEND_BUTTON_DISABLED_TEXT};
        border: 1px solid {SEND_BUTTON_DISABLED_BORDER};
    }}

    QPushButton#ComposerStopButton {{
        background: {SEND_BUTTON_BG};
        color: {SURFACE_BG};
        border: none;
        border-radius: 15px;
        min-width: 32px;
        max-width: 32px;
        min-height: 32px;
        max-height: 32px;
        padding: 0px;
        font-weight: 700;
    }}

    QPushButton#ComposerStopButton:hover {{
        background: {SEND_BUTTON_HOVER};
    }}

    QPushButton#ComposerStopButton:pressed {{
        background: {SEND_BUTTON_PRESSED};
    }}

    QToolButton#ComposerAttachButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 0px;
    }}

    QToolButton#ComposerAttachButton:hover {{
        background: {COMPOSER_ATTACH_HOVER};
        border: none;
    }}

    QPushButton#ComposerMetaChip {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 2px 8px;
        color: {TEXT_LABEL};
        font-size: 8.8pt;
        font-weight: 500;
    }}

    QPushButton#ComposerMetaChip:disabled {{
        color: {TEXT_LABEL};
        background: transparent;
        border: 1px solid transparent;
    }}

    QToolButton#ComposerMetaChipButton {{
        background: {COMPOSER_CHIP_BG};
        border: 1px solid {COMPOSER_CHIP_BORDER};
        border-radius: {SOFT_RADIUS_MD + 6}px;
        padding: 4px 12px;
        color: {TEXT_PRIMARY};
        font-size: 9.4pt;
        font-weight: 500;
    }}

    QToolButton#ComposerMetaChipButton[chipPosition="left"] {{
        border-top-right-radius: 0px;
        border-bottom-right-radius: 0px;
    }}

    QToolButton#ComposerMetaChipButton[chipPosition="right"] {{
        border-left: none;
        border-top-left-radius: 0px;
        border-bottom-left-radius: 0px;
    }}

    QToolButton#ComposerMetaChipButton:hover {{
        background: {COMPOSER_CHIP_HOVER};
        border: 1px solid {COMPOSER_CHIP_HOVER_BORDER};
    }}

    QToolButton#ComposerMetaChipButton[chipPosition="right"]:hover {{
        border-left: none;
    }}

    QToolButton#ComposerMetaChipButton:disabled {{
        color: {TEXT_LABEL};
        background: transparent;
        border: 1px solid transparent;
    }}

    QLabel#ComposerNoModelText {{
        color: {DIFF_TOGGLE_TEXT};
        font-size: 8.8pt;
    }}

    QLabel#ComposerCacheHitLabel {{
        color: {TEXT_SOFT};
        background: transparent;
        padding: 0px 2px 0px 0px;
        font-size: 8.4pt;
    }}

    QLabel#ComposerChatOnlyLabel {{
        background: transparent;
    }}

    QLabel#ComposerChatOnlyLabel:disabled {{
        color: {TEXT_LABEL};
    }}

    QPushButton#ComposerOpenSettingsButton {{
        background: transparent;
        border: 1px solid {COMPOSER_SETTINGS_TEXT};
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 2px 8px;
        color: {DISCLOSURE_HOVER_TEXT};
        font-size: 8.8pt;
    }}

    QPushButton#ComposerOpenSettingsButton:hover {{
        background: {COMPOSER_SETTINGS_HOVER};
        border: 1px solid {TEXT_SUBTITLE};
    }}

    QToolButton#ComposerGhostButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 0px;
    }}

    QToolButton#ComposerGhostButton:disabled {{
        background: transparent;
        border: 1px solid transparent;
    }}

    QFrame#ComposerMentionPopup {{
        background: {POPUP_BG};
        border: 1px solid {MENTION_POPUP_BORDER};
        border-radius: {SOFT_RADIUS_MD + 10}px;
    }}

    QLabel#ComposerMentionHeader {{
        background: transparent;
        color: {TEXT_SUBTITLE};
        font-size: 8.8pt;
        font-weight: 600;
        padding: 1px 4px 2px 4px;
    }}

    QListWidget#ComposerMentionList {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_MD + 6}px;
        padding: 0px;
        outline: none;
        font-size: 10.2pt;
        show-decoration-selected: 1;
    }}

    QListWidget#ComposerMentionList::item {{
        background: transparent;
        border: none;
        padding: 0px;
        margin: 0px;
        color: transparent;
    }}

    QWidget#ComposerMentionItem {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_MD + 4}px;
    }}

    QWidget#ComposerMentionItem[selected="true"] {{
        background: {MENTION_SELECTED_BG};
        border: 1px solid {MENTION_SELECTED_BORDER};
    }}

    QWidget#ComposerMentionItem[selected="true"] QLabel#ComposerMentionItemTitle {{
        color: {MENTION_SELECTED_TEXT};
    }}

    QWidget#ComposerMentionItem[selected="true"] QLabel#ComposerMentionItemMeta {{
        color: {TEXT_SUMMARY};
    }}

    QLabel#ComposerMentionItemTitle {{
        background: transparent;
        color: {TEXT_PRIMARY};
        font-size: 10.2pt;
        font-weight: 600;
    }}

    QLabel#ComposerMentionItemMeta {{
        background: transparent;
        color: {MENTION_META_TEXT};
        font-size: 8.9pt;
        font-weight: 500;
        padding-left: 4px;
    }}

    QLabel#ComposerMentionItemIcon {{
        background: transparent;
        border: none;
    }}

    QListWidget#ComposerMentionList QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 4px 0px 4px 6px;
    }}

    QListWidget#ComposerMentionList QScrollBar::handle:vertical {{
        background: {MENTION_SCROLLBAR};
        border-radius: 4px;
        min-height: 28px;
    }}

    QListWidget#ComposerMentionList QScrollBar::handle:vertical:hover {{
        background: {SEND_BUTTON_DISABLED_TEXT};
    }}

    QListWidget#ComposerMentionList QScrollBar::add-line:vertical,
    QListWidget#ComposerMentionList QScrollBar::sub-line:vertical,
    QListWidget#ComposerMentionList QScrollBar::add-page:vertical,
    QListWidget#ComposerMentionList QScrollBar::sub-page:vertical {{
        background: transparent;
        border: none;
        height: 0px;
    }}

    QPushButton#TranscriptJumpButton {{
        background: {TRANSCRIPT_JUMP_BG};
        color: {TEXT_PRIMARY};
        border: none;
        border-radius: {SOFT_RADIUS_MD + 8}px;
        padding: 7px 12px;
        font-size: 8.9pt;
        font-weight: 600;
    }}

    QPushButton#TranscriptJumpButton:hover {{
        background: {BUTTON_BG};
    }}

    QTextBrowser#AssistantBody {{
        background: {AI_MESSAGE_BG};
        color: {TEXT_CHAT};
        border: none;
        border-radius: 0px;
        padding: 0px;
        font-family: "{APP_FONT_FAMILY}";
        font-size: 11.5pt;
    }}

    QLabel#TranscriptMeta {{
        color: {TEXT_MUTED};
        font-size: 10pt;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 0px;
    }}

    QLabel#TranscriptMeta,
    QLabel#TranscriptRole {{
        color: {TRANSCRIPT_ROLE_TEXT};
        font-size: 10.7pt;
        font-weight: 500;
    }}

    QPushButton#ToolCallButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_XS}px;
        padding: 0px 2px 0px 0px;
        color: {TOOL_CALL_TEXT};
        font-size: 10.35pt;
        font-weight: 500;
        text-align: left;
    }}

    QPushButton#ToolCallButton[toolRole="edit"] {{
        color: {CODE_HEADER_TEXT};
    }}

    QPushButton#ToolCallButton:hover {{
        background: transparent;
        color: {TEXT_SUMMARY};
        border: none;
    }}

    QPushButton#ToolCallButton:checked {{
        background: transparent;
        color: {STATUS_REVIEW_TEXT};
        border: none;
    }}

    QPushButton#ToolGroupHeaderButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 2px 2px 2px 0px;
        color: {TEXT_SUMMARY};
        font-size: 10pt;
        font-weight: 500;
        text-align: left;
    }}

    QPushButton#ToolGroupHeaderButton[state="active"] {{
        color: {TEXT_SUMMARY};
    }}

    QPushButton#ToolGroupHeaderButton[state="complete"] {{
        color: {TEXT_SUMMARY};
    }}

    QPushButton#ToolGroupHeaderButton[state="error"] {{
        color: {TOOL_GROUP_ERROR_TEXT};
    }}

    QPushButton#ToolGroupHeaderButton:hover {{
        background: transparent;
        color: {TOOL_TOGGLE_HOVER_TEXT};
        border: none;
    }}

    QPushButton#ToolGroupHeaderButton:checked {{
        background: transparent;
        color: {STATUS_REVIEW_TEXT};
        border: none;
    }}

    /* Folded sub-groups keep a muted, tool-row gray for their caption so that
       "Completed N tools" never brightens to the light gray the top-level
       group header uses when expanded. This keeps the two levels visually
       distinct even after a sub-group is opened. */
    QFrame#ToolGroupFrame[subgroup="true"] QPushButton#ToolGroupHeaderButton,
    QFrame#ToolGroupFrame[subgroup="true"] QPushButton#ToolGroupHeaderButton[state="active"],
    QFrame#ToolGroupFrame[subgroup="true"] QPushButton#ToolGroupHeaderButton[state="complete"],
    QFrame#ToolGroupFrame[subgroup="true"] QPushButton#ToolGroupHeaderButton:checked {{
        color: {TEXT_SUMMARY};
    }}

    QFrame#ToolGroupFrame[subgroup="true"] QPushButton#ToolGroupHeaderButton[state="error"] {{
        color: {TOOL_GROUP_ERROR_TEXT};
    }}

    QFrame#ToolGroupFrame[subgroup="true"] QPushButton#ToolGroupHeaderButton:hover {{
        color: {TOOL_GROUP_HOVER_TEXT};
    }}

    QPlainTextEdit#CodeView,
    QPlainTextEdit#InlineCodeView {{
        background: {TOOL_CODE_BG};
        color: {CODE_TEXT};
        border: 1px solid {CODE_PANEL_BORDER};
        border-radius: {SOFT_RADIUS_SM}px;
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
        padding: 8px 10px;
    }}

    QWidget#ToolsContainer,
    QWidget#InspectorPanel {{
        background: transparent;
    }}

    QTextBrowser#InspectorHelpText {{
        background: {INSPECTOR_HELP_BG};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 10px 12px;
    }}

    QFrame#ToolCard {{
        background: {SURFACE_RAISED};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
        min-height: 44px;
    }}

    QCheckBox#ToolAvailabilitySwitch {{
        background: {BUTTON_BG};
        border-radius: 10px;
        border: 1px solid {TOOL_SWITCH_BORDER};
        padding: 0px;
        spacing: 0px;
    }}

    QCheckBox#ToolAvailabilitySwitch:hover {{
        background: {TOOL_SWITCH_HOVER};
    }}

    QCheckBox#ToolAvailabilitySwitch:checked {{
        background: {ACCENT};
        border: 1px solid {ACCENT};
    }}

    QCheckBox#ToolAvailabilitySwitch:disabled {{
        background: {SURFACE_BG};
        border-color: {BORDER};
    }}

    QCheckBox#ToolAvailabilitySwitch:checked:disabled {{
        background: {SWITCH_DISABLED_BG};
        border: 1px solid {SWITCH_DISABLED_BG};
    }}

    QCheckBox#ToolAvailabilitySwitch::indicator {{
        width: 14px;
        height: 14px;
        border-radius: 7px;
        background: {TEXT_PRIMARY};
        border: none;
    }}

    QCheckBox#ToolAvailabilitySwitch::indicator:unchecked {{
        subcontrol-origin: margin;
        subcontrol-position: left center;
        margin-left: 2px;
    }}

    QCheckBox#ToolAvailabilitySwitch::indicator:checked {{
        subcontrol-origin: margin;
        subcontrol-position: right center;
        margin-right: 2px;
        background: {SURFACE_BG};
    }}

    QCheckBox#ToolAvailabilitySwitch::indicator:disabled {{
        background: {SWITCH_DISABLED_THUMB};
    }}

    QCheckBox#ToolAvailabilitySwitch::indicator:checked:disabled {{
        background: {SURFACE_BG};
    }}

    QLabel#ToolGroupHeader {{
        color: {TEXT_MUTED};
        font-size: 7.2pt;
        font-weight: 700;
        letter-spacing: 0.8px;
        padding: 8px 4px 3px 4px;
    }}

    QLabel#ToolGroupHeader[toolGroup="protected"] {{
        color: {TEXT_WARNING};
    }}

    QLabel#ToolGroupHeader[toolGroup="mcp"] {{
        color: {ACCENT};
    }}

    QToolButton#ToolCardTitle {{
        color: {TEXT_PRIMARY};
        font-weight: 600;
        font-size: 8.8pt;
        font-family: "{MONO_FONT_FAMILY}";
        background: transparent;
        border: none;
        padding: 0;
        text-align: left;
    }}

    QToolButton#ToolCardTitle:hover {{
        color: {ACCENT};
    }}

    QWidget#ToolCardDetails {{
        background: transparent;
    }}

    QLabel#ToolCardDescription {{
        color: {TEXT_MUTED};
        font-size: 8pt;
        background: transparent;
    }}

    QLabel#MCPToolCardItem {{
        color: {TEXT_MUTED};
        font-size: 8pt;
        background: transparent;
    }}

    QLabel#MCPServerLoadingLabel {{
        color: {ACCENT};
        font-size: 7.5pt;
        background: transparent;
    }}

    QLabel#ToolFlagChip {{
        color: {TEXT_MUTED};
        font-size: 7pt;
        border: 1px solid {SCROLLBAR_HANDLE};
        border-radius: 3px;
        padding: 0px 4px;
        background: transparent;
    }}

    QLabel#ToolFlagChip[flagVariant="warning"] {{
        color: {TEXT_WARNING};
        border: 1px solid {WARNING_CHIP_BORDER};
    }}

    QLabel#ToolFlagChip[flagVariant="accent"] {{
        color: {ACCENT};
        border: 1px solid {ACCENT_CHIP_BORDER};
    }}

    QFrame#ToolGroupSeparator {{
        background: {BORDER};
        margin: 4px 0px;
    }}

    QScrollArea {{
        border: none;
        background: transparent;
    }}

    QScrollBar:vertical {{
        background: transparent;
        width: 6px;
        margin: 4px 0 4px 0;
    }}

    QScrollBar:horizontal {{
        background: transparent;
        height: 6px;
        margin: 0 4px 0 4px;
    }}

    QScrollBar::handle:vertical {{
        background: {COMPOSER_SCROLLBAR};
        border-radius: {SOFT_RADIUS_XS}px;
        min-height: 16px;
    }}

    QScrollBar::handle:horizontal {{
        background: {COMPOSER_SCROLLBAR};
        border-radius: {SOFT_RADIUS_XS}px;
        min-width: 16px;
    }}

    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{
        background: transparent;
        border: none;
        height: 0px;
    }}

    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal,
    QScrollBar::sub-page:horizontal {{
        background: transparent;
        border: none;
        width: 0px;
    }}

    QMenuBar {{
        background: {SURFACE_DIALOG};
        color: {TEXT_SECONDARY};
    }}

    QMenuBar::item {{
        background: transparent;
        padding: 4px 8px;
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QMenu {{
        background: {SURFACE_CARD};
        color: {TEXT_PRIMARY};
        border: none;
        padding: 6px;
    }}

    QMenu#ComposerPopupMenu {{
        background: {POPUP_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {POPUP_BORDER};
        border-radius: {SOFT_RADIUS_MD + 8}px;
        padding: 6px;
    }}

    QMenu#ComposerPopupMenu::item {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_MD + 4}px;
        padding: 10px 14px;
        margin: 1px 0px;
    }}

    QMenu#ComposerPopupMenu::item:selected {{
        background: {POPUP_ITEM_SELECTED};
    }}

    QMenu#ComposerPopupMenu::separator {{
        height: 1px;
        background: {COMPOSER_CHIP_BORDER};
        margin: 6px 8px;
    }}

    QMenuBar::item:selected,
    QMenu::item:selected {{
        background: {ACCENT_SOFT};
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QStatusBar {{
        background: {STATUS_BAR_BG};
        color: {TEXT_MUTED};
        border-top: 1px solid {BORDER};
        min-height: 22px;
        padding-left: 6px;
    }}

    QStatusBar::item {{
        border: none;
    }}

    QLabel#StatusBarMeta {{
        color: {STATUS_BAR_META_TEXT};
        font-size: 9pt;
        padding-right: 4px;
    }}

    QLabel#StatusBarState {{
        color: {STATUS_BAR_STATE_TEXT};
        font-size: 9.6pt;
        font-weight: 600;
        padding-left: 2px;
    }}
    """
