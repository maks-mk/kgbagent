from __future__ import annotations

from functools import lru_cache

# --- Claude.ai Neutral-Warm Dark Theme ---
ACCENT_BLUE = "#A8A49E"          # Neutral warm gray accent
ACCENT_BLUE_SOFT = "#2A2825"     # Very subtle warm hover tint

TEXT_PRIMARY = "#ECEAE6"         # Near-white with a slight warm tint
TEXT_SECONDARY = "#9E9A94"       # Neutral warm gray
TEXT_MUTED = "#72706A"           # Muted, lightly warm
TEXT_DIM = "#524F4A"             # Dim neutral

# Base surfaces — neutral with a minimal warm undertone
SURFACE_BG = "#1E1D1B"          # Main background (nearly neutral, slightly warmer than black)
SURFACE_CARD = "#282623"        # Card surface, slightly warmer than the main background
SURFACE_ALT = "#38352F"         # Buttons and highlighted surfaces
BORDER = "#38352F"

AMBER_WARNING = "#D97706"
ERROR_RED = "#EF4444"
SUCCESS_GREEN = "#10B981"
FILENAME_BLUE = "#7CC7FF"         # Filename highlights in assistant markdown and tool cards
CODE_BG = "#161514"             # Code background
CODE_TEXT = "#ECEAE6"

_COMPOSER_BG = "#282623"
_COMPOSER_BORDER = "#38352F"
_SEND_BTN_BG = "#ECEAE6"
_SEND_BTN_HOVER = "#FFFFFF"
_SEND_BTN_DISABLED = "#38352F"

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


_POPUP_BG = blend_hex(SURFACE_BG, SURFACE_CARD, 0.66)
_POPUP_BORDER = blend_hex(BORDER, "#FFFFFF", 0.12)
_POPUP_ITEM_HOVER = blend_hex(SURFACE_ALT, "#FFFFFF", 0.08)
_POPUP_ITEM_SELECTED = blend_hex(SURFACE_ALT, ACCENT_BLUE_SOFT, 0.28)


def _build_theme_palette() -> dict[str, str]:
    transcript_panel_bg = blend_hex(SURFACE_CARD, SURFACE_ALT, 0.18)
    tool_panel_bg = blend_hex(SURFACE_CARD, "#FFFFFF", 0.04)
    tool_toggle_text = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.26)
    return {
        "transcript_panel_bg": transcript_panel_bg,
        "tool_panel_bg": tool_panel_bg,
        "tool_panel_hover": blend_hex(tool_panel_bg, "#FFFFFF", 0.05),
        "tool_code_bg": blend_hex(SURFACE_CARD, "#FFFFFF", 0.07),
        "tool_toggle_text": tool_toggle_text,
        "tool_toggle_hover_text": blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.52),
        "tool_call_hover": blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.42),
    }


@lru_cache(maxsize=1)
def build_stylesheet() -> str:
    palette = _build_theme_palette()
    transcript_panel_bg = palette["transcript_panel_bg"]
    tool_panel_bg = palette["tool_panel_bg"]
    tool_panel_hover = palette["tool_panel_hover"]
    tool_code_bg = palette["tool_code_bg"]
    tool_toggle_text = palette["tool_toggle_text"]
    tool_toggle_hover_text = palette["tool_toggle_hover_text"]
    tool_call_hover = palette["tool_call_hover"]
    model_dialog_bg = blend_hex(SURFACE_BG, "#FFFFFF", 0.03)
    model_dialog_card = blend_hex(SURFACE_CARD, "#FFFFFF", 0.03)
    model_dialog_card_alt = blend_hex(SURFACE_CARD, SURFACE_ALT, 0.28)
    model_dialog_editor = blend_hex(model_dialog_card, "#FFFFFF", 0.035)
    model_dialog_soft = blend_hex(SURFACE_ALT, "#FFFFFF", 0.04)
    model_dialog_border = blend_hex(BORDER, "#FFFFFF", 0.10)
    model_dialog_text = TEXT_PRIMARY
    model_dialog_muted = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.18)
    model_dialog_selected = blend_hex(SURFACE_ALT, "#5B8DEF", 0.18)
    model_dialog_selected_border = blend_hex("#5B8DEF", "#FFFFFF", 0.14)
    model_dialog_disabled = blend_hex(SURFACE_BG, SURFACE_ALT, 0.46)
    model_dialog_active_bg = blend_hex(SUCCESS_GREEN, SURFACE_ALT, 0.82)
    model_dialog_active_text = blend_hex(SUCCESS_GREEN, TEXT_PRIMARY, 0.18)
    model_dialog_provider_bg = blend_hex(SURFACE_ALT, "#92A4BC", 0.20)
    model_dialog_provider_text = blend_hex("#92A4BC", TEXT_PRIMARY, 0.22)
    model_dialog_chip_bg = blend_hex(SURFACE_ALT, "#FFFFFF", 0.05)
    model_dialog_field_bg = blend_hex(SURFACE_CARD, "#FFFFFF", 0.05)
    model_dialog_field_border = blend_hex(BORDER, "#FFFFFF", 0.09)
    model_dialog_switch_off = blend_hex(SURFACE_ALT, "#FFFFFF", 0.10)
    model_dialog_switch_thumb = "#F3F4F6"
    approval_dialog_bg = blend_hex(SURFACE_BG, "#FFFFFF", 0.02)
    approval_dialog_card = blend_hex(SURFACE_CARD, "#FFFFFF", 0.03)
    approval_dialog_border = blend_hex(BORDER, "#FFFFFF", 0.10)
    approval_dialog_soft = blend_hex(SURFACE_ALT, "#FFFFFF", 0.05)

    approval_dialog_code = blend_hex(CODE_BG, "#FFFFFF", 0.02)
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

    QFrame#SidebarCard,
    QFrame#StatusCard,
    QFrame#NoticeCard {{
        background: {SURFACE_CARD};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QFrame#TranscriptMetaChip {{
    background: transparent;
    border: none;
    }}
    
    QFrame#ApprovalCard,
    QFrame#ApprovalRequestCard {{
        background: {approval_dialog_card};
        border: 1px solid {approval_dialog_border};
        border-radius: {SOFT_RADIUS_MD + 2}px;
    }}

    QFrame#ApprovalToolCard {{
        background: {approval_dialog_soft};
        border: 1px solid {approval_dialog_border};
        border-radius: {SOFT_RADIUS_MD + 2}px;
    }}

    QLabel#TopStatusChip {{
        color: {TEXT_PRIMARY};
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.06)};
        border: none;
        border-radius: {SOFT_RADIUS_MD + 6}px;
        padding: 4px 10px;
        font-size: 8.8pt;
        font-weight: 600;
    }}

    QLabel#TopStatusChip[statusState="busy"] {{
        background: {blend_hex(ACCENT_BLUE_SOFT, ACCENT_BLUE, 0.24)};
    }}

    QLabel#TopStatusChip[statusState="success"] {{
        background: {blend_hex(SUCCESS_GREEN, SURFACE_ALT, 0.82)};
        color: {blend_hex(SUCCESS_GREEN, TEXT_PRIMARY, 0.18)};
    }}

    QLabel#TopStatusChip[statusState="error"] {{
        background: {blend_hex(ERROR_RED, SURFACE_ALT, 0.82)};
        color: {blend_hex(ERROR_RED, TEXT_PRIMARY, 0.18)};
    }}

    QLabel#TopStatusChip[statusState="idle"] {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.22)};
    }}

    QFrame#UserBubble {{
        background: {blend_hex(SURFACE_CARD, ACCENT_BLUE_SOFT, 0.32)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QFrame#ImageAttachmentCard {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.05)};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QLabel#ImageAttachmentThumb {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.03)};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QToolButton#ImageAttachmentRemoveButton {{
        background: {blend_hex(SURFACE_BG, "#FFFFFF", 0.08)};
        border: none;
        border-radius: {SOFT_RADIUS_XS + 6}px;
        padding: 2px;
    }}

    QToolButton#ImageAttachmentRemoveButton:hover {{
        background: {blend_hex(SURFACE_BG, "#FFFFFF", 0.16)};
    }}

    QDialog#InfoPopup {{
        background: {SURFACE_CARD};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QDockWidget#ModelSettingsDock {{
        background: {model_dialog_bg};
        border: none;
    }}

    QDockWidget#ModelSettingsDock QWidget {{
        background: transparent;
        color: {model_dialog_text};
    }}

    QDockWidget#ModelSettingsDock::title {{
        padding: 0px;
        margin: 0px;
        height: 0px;
        background: transparent;
        border: none;
    }}

    QDialog#ModelSettingsDialog {{
        background: {model_dialog_bg};
        border: none;
    }}

    QDialog#ModelSettingsDialog QWidget {{
        background: transparent;
        color: {model_dialog_text};
    }}

    QDialog#ModelSettingsDialog QLabel {{
        background: transparent;
    }}

    QFrame#ModelSettingsPane {{
        background: {model_dialog_card_alt};
        border: 1px solid {model_dialog_border};
        border-radius: {SOFT_RADIUS_MD + 6}px;
    }}

    QFrame#ModelSettingsPane[paneRole="editor"] {{
        background: {model_dialog_editor};
        border: 1px solid {blend_hex(model_dialog_border, "#FFFFFF", 0.08)};
    }}

    QFrame#ModelSettingsHeroCard {{
        background: {model_dialog_card};
        border: 1px solid {model_dialog_border};
        border-radius: {SOFT_RADIUS_MD + 8}px;
    }}

    QLabel#ModelSettingsTitle {{
        color: {model_dialog_text};
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
        background: {blend_hex(ERROR_RED, SURFACE_ALT, 0.72)};
        border: 1px solid {blend_hex(ERROR_RED, "#FFFFFF", 0.18)};
    }}

    QPushButton#SettingsCloseButton:pressed {{
        background: {blend_hex(ERROR_RED, SURFACE_ALT, 0.55)};
    }}

    QLabel#ModelSettingsSubtitle {{
        color: {model_dialog_muted};
        font-size: 9pt;
        padding-left: 1px;
        padding-bottom: 1px;
    }}

    QLabel#ModelSettingsMeta {{
        color: {model_dialog_muted};
        font-size: 8.5pt;
        padding-left: 1px;
        background: transparent;
    }}

    QLabel#ModelSettingsChip {{
        color: {model_dialog_text};
        background: {model_dialog_chip_bg};
        border: 1px solid {model_dialog_border};
        border-radius: {SOFT_RADIUS_MD + 4}px;
        padding: 3px 8px;
        font-size: 8pt;
        font-weight: 700;
    }}

    QFrame#ModelSettingsPane {{
        background: {model_dialog_card_alt};
        border: 1px solid {model_dialog_border};
        border-radius: {SOFT_RADIUS_MD + 4}px;
    }}

    QFrame#ModelSettingsPane[paneRole="editor"] {{
        background: {model_dialog_editor};
        border: 1px solid {blend_hex(model_dialog_border, "#FFFFFF", 0.08)};
    }}

    QFrame#ModelSettingsFormCard,
    QFrame#ModelSettingsHelperCard {{
        background: {model_dialog_card};
        border: 1px solid {model_dialog_border};
        border-radius: {SOFT_RADIUS_MD + 2}px;
    }}

    QFrame#ModelSettingsSelectedProfileHeader {{
        background: {blend_hex(model_dialog_editor, "#FFFFFF", 0.035)};
        border: 1px solid {blend_hex(model_dialog_border, "#FFFFFF", 0.07)};
        border-radius: {SOFT_RADIUS_MD + 2}px;
    }}

    QLabel#ModelSettingsSelectedProfileTitle {{
        color: {model_dialog_text};
        background: transparent;
        font-size: 11pt;
        font-weight: 760;
        padding: 0px;
    }}

    QLabel#ModelSettingsSummaryLabel {{
        color: {model_dialog_muted};
        background: transparent;
        font-size: 8.5pt;
        font-weight: 600;
    }}

    QLabel#ModelSettingsHelperTitle {{
        color: {model_dialog_text};
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
        background: {model_dialog_field_bg};
        border: 1px solid {model_dialog_field_border};
        border-radius: {SOFT_RADIUS_MD + 2}px;
        min-height: 28px;
        padding: 3px 8px;
        color: {model_dialog_text};
        selection-background-color: {model_dialog_selected};
    }}

    QLineEdit#ModelSettingsSearchField:focus {{
        border: 1px solid {model_dialog_selected_border};
        background: {model_dialog_field_bg};
    }}

    QListWidget#ModelProfileList::item {{
        border: none;
        background: transparent;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 5px 3px;
        margin: 2px 0px;
        color: {model_dialog_text};
    }}

    QListWidget#ModelProfileList::item:selected {{
        background: transparent;
        color: {model_dialog_text};
    }}

    QWidget#ModelProfileRowCard {{
        background: {model_dialog_card};
        border: 1px solid {model_dialog_border};
        border-radius: {SOFT_RADIUS_MD + 4}px;
    }}

    QWidget#ModelProfileRowCard[disabledProfile="true"] {{
        background: {model_dialog_disabled};
        border: 1px solid {model_dialog_border};
    }}

    QWidget#ModelProfileRowCard[selectedProfile="true"] {{
        background: {model_dialog_editor};
        border: 1px solid {model_dialog_selected_border};
        border-right: 3px solid {model_dialog_selected_border};
    }}

    QWidget#ModelProfileRowCard[selectedProfile="true"][disabledProfile="true"] {{
        background: {blend_hex(model_dialog_editor, model_dialog_disabled, 0.45)};
        border: 1px solid {blend_hex(model_dialog_selected_border, model_dialog_border, 0.40)};
        border-right: 3px solid {blend_hex(model_dialog_selected_border, model_dialog_border, 0.28)};
    }}

    QFrame#ModelProfileTabRail {{
        background: transparent;
        border: none;
        border-radius: 1px;
        min-width: 3px;
        max-width: 3px;
    }}

    QFrame#ModelProfileTabRail[selectedProfile="true"] {{
        background: {model_dialog_selected_border};
    }}

    QLabel#ModelProfileItemTitle {{
        color: {model_dialog_text};
        font-size: 9.5pt;
        font-weight: 700;
        background: transparent;
        padding: 0px;
    }}

    QLabel#ModelProfileItemMeta {{
        color: {model_dialog_muted};
        font-size: 8.5pt;
        background: transparent;
        padding: 0px;
    }}

    QLabel#ModelProfileItemBadge {{
        background: {model_dialog_soft};
        color: {model_dialog_muted};
        border: 1px solid {model_dialog_border};
        border-radius: {SOFT_RADIUS_MD + 4}px;
        padding: 1px 6px;
        font-size: 7.8pt;
        font-weight: 700;
    }}

    QLabel#ModelProfileItemBadge[badgeVariant="active"] {{
        background: {model_dialog_active_bg};
        color: {model_dialog_active_text};
        border: 1px solid {blend_hex(model_dialog_active_bg, model_dialog_active_text, 0.18)};
    }}

    QLabel#ModelProfileItemBadge[badgeVariant="provider"] {{
        background: {model_dialog_provider_bg};
        color: {model_dialog_provider_text};
        border: 1px solid {blend_hex(model_dialog_provider_bg, model_dialog_provider_text, 0.16)};
    }}

    QCheckBox#ModelProfileEnabledSwitch {{
        background: {model_dialog_switch_off};
        border-radius: 10px;
        border: 1px solid {blend_hex(model_dialog_switch_off, "#8A94A3", 0.32)};
        padding: 0px;
        spacing: 0px;
    }}

    QCheckBox#ModelProfileEnabledSwitch:hover {{
        background: {blend_hex(model_dialog_switch_off, "#FFFFFF", 0.14)};
    }}

    QCheckBox#ModelProfileEnabledSwitch:checked {{
        background: #FFFFFF;
        border: 1px solid #FFFFFF;
    }}

    QCheckBox#ModelProfileEnabledSwitch::indicator {{
        width: 14px;
        height: 14px;
        border-radius: 7px;
        background: {model_dialog_switch_thumb};
        border: 1px solid {blend_hex(model_dialog_switch_thumb, "#94A3B8", 0.24)};
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
    
    QCheckBox#ModelSupportsImagesCheckbox {{
        background: transparent;
        border: none;
        padding: 0px;
        spacing: 6px;
        color: {model_dialog_text};
    }}

    QCheckBox#ModelSupportsImagesCheckbox::indicator {{
        width: 15px;
        height: 15px;
        border-radius: 4px;
        border: 1px solid {blend_hex(model_dialog_field_border, "#FFFFFF", 0.22)};
        background: {model_dialog_field_bg};
    }}

    QCheckBox#ModelSupportsImagesCheckbox::indicator:hover {{
        border: 1px solid {blend_hex(model_dialog_selected_border, "#FFFFFF", 0.18)};
        background: {blend_hex(model_dialog_field_bg, "#FFFFFF", 0.08)};
    }}

    QCheckBox#ModelSupportsImagesCheckbox::indicator:checked {{
        border: 1px solid {model_dialog_selected_border};
        background: {model_dialog_selected};
    }}

    QCheckBox#ModelSupportsImagesCheckbox::indicator:checked:hover {{
        background: {blend_hex(model_dialog_selected, "#FFFFFF", 0.10)};
    }}

    QCheckBox#ModelSupportsImagesCheckbox::indicator:disabled {{
        border: 1px solid {blend_hex(model_dialog_field_border, model_dialog_bg, 0.45)};
        background: {model_dialog_soft};
    }}

    QDialog#ModelSettingsDialog QLineEdit,
    QDialog#ModelSettingsDialog QComboBox {{
        background: {model_dialog_field_bg};
        border: 1px solid {model_dialog_field_border};
        border-radius: {SOFT_RADIUS_MD + 2}px;
        min-height: 28px;
        padding: 3px 8px;
        color: {model_dialog_text};
        selection-background-color: {model_dialog_selected};
    }}

    QDialog#ModelSettingsDialog QLineEdit:focus,
    QDialog#ModelSettingsDialog QComboBox:focus {{
        border: 1px solid {model_dialog_selected_border};
    }}

    QDialog#ModelSettingsDialog QLineEdit:disabled,
    QDialog#ModelSettingsDialog QComboBox:disabled {{
        background: {model_dialog_soft};
        color: {blend_hex(model_dialog_muted, model_dialog_text, 0.18)};
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
        background: {blend_hex(model_dialog_card, model_dialog_field_bg, 0.20)};
        border: 1px solid {blend_hex(model_dialog_selected_border, model_dialog_border, 0.35)};
        border-radius: {SOFT_RADIUS_MD + 4}px;
    }}

    QListWidget#ModelSettingsModelPopupList {{
        background: transparent;
        border: none;
        outline: 0px;
        color: {model_dialog_text};
        selection-background-color: {blend_hex(model_dialog_selected, model_dialog_field_bg, 0.14)};
        selection-color: {model_dialog_text};
    }}

    QListWidget#ModelSettingsModelPopupList::item {{
        min-height: 26px;
        padding: 4px 10px;
        margin: 1px 4px;
        border-radius: {SOFT_RADIUS_SM}px;
        background: transparent;
    }}

    QListWidget#ModelSettingsModelPopupList::item:selected {{
        background: {blend_hex(model_dialog_selected, model_dialog_field_bg, 0.18)};
        color: {TEXT_PRIMARY};
    }}

    QListWidget#ModelSettingsModelPopupList::item:hover {{
        background: {blend_hex(model_dialog_selected, model_dialog_field_bg, 0.10)};
    }}

    QDialog#ModelSettingsDialog QComboBox QAbstractItemView {{
        background: {model_dialog_card};
        border: 1px solid {model_dialog_border};
        selection-background-color: {model_dialog_selected};
        color: {model_dialog_text};
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView {{
        background: {blend_hex(model_dialog_card, model_dialog_field_bg, 0.20)};
        border: 1px solid {blend_hex(model_dialog_selected_border, model_dialog_border, 0.35)};
        border-radius: {SOFT_RADIUS_MD + 4}px;
        padding: 6px 4px;
        outline: 0px;
        selection-background-color: {blend_hex(model_dialog_selected, model_dialog_field_bg, 0.14)};
        selection-color: {model_dialog_text};
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView::item {{
        min-height: 26px;
        padding: 4px 10px;
        margin: 1px 4px;
        border-radius: {SOFT_RADIUS_SM}px;
        background: transparent;
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView::item:selected {{
        background: {blend_hex(model_dialog_selected, model_dialog_field_bg, 0.18)};
        color: {TEXT_PRIMARY};
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView::item:hover {{
        background: {blend_hex(model_dialog_selected, model_dialog_field_bg, 0.10)};
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 4px 2px 4px 6px;
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView QScrollBar::handle:vertical {{
        background: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.14)};
        border-radius: 4px;
        min-height: 28px;
    }}

    QDialog#ModelSettingsDialog QComboBox#ModelSettingsModelCombo QAbstractItemView QScrollBar::handle:vertical:hover {{
        background: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.24)};
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
        color: {model_dialog_text};
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
        border: 1px solid {model_dialog_field_border};
    }}

    QPushButton#SettingsAddButton {{
        background: {model_dialog_selected};
        color: {model_dialog_text};
    }}

    QPushButton#SettingsAddButton:hover {{
        background: {blend_hex(model_dialog_selected, "#FFFFFF", 0.18)};
    }}

    QPushButton#SettingsDeleteButton {{
        background: {model_dialog_card};
        color: {blend_hex(ERROR_RED, model_dialog_text, 0.2)};
        border: 1px solid {blend_hex(ERROR_RED, model_dialog_border, 0.75)};
    }}

    QPushButton#ModelSettingsInlineButton,
    QToolButton#ModelSettingsInlineToolButton {{
        background: {model_dialog_card};
        border: 1px solid {model_dialog_field_border};
        border-radius: {SOFT_RADIUS_MD + 2}px;
        color: {model_dialog_text};
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
        background: {blend_hex(model_dialog_selected, model_dialog_card, 0.14)};
        border: 1px solid {blend_hex(model_dialog_selected_border, model_dialog_field_border, 0.42)};
    }}

    QPushButton#ModelSettingsInlineButton:pressed,
    QToolButton#ModelSettingsInlineToolButton:pressed {{
        background: {blend_hex(model_dialog_selected, model_dialog_card, 0.22)};
    }}

    QToolButton#ModelSettingsInlineToolButton:disabled {{
        background: {blend_hex(model_dialog_soft, model_dialog_card, 0.35)};
        border: 1px solid {blend_hex(model_dialog_field_border, model_dialog_card, 0.35)};
    }}

    QLabel#ModelSettingsHintText {{
        color: {model_dialog_muted};
        background: transparent;
        font-size: 8.2pt;
        padding-left: 2px;
    }}

    QSplitter::handle {{
        background: transparent;
    }}

    QDialog#ModelSettingsDialog QDialogButtonBox {{
        border-top: 1px solid {model_dialog_border};
        padding-top: 6px;
    }}

    QDialog#ModelSettingsDialog QPushButton#PrimaryButton {{
        background: {blend_hex("#5B8DEF", SURFACE_ALT, 0.12)};
        color: {TEXT_PRIMARY};
        border: 1px solid {blend_hex("#5B8DEF", "#FFFFFF", 0.10)};
        border-radius: {SOFT_RADIUS_MD + 4}px;
        font-weight: 700;
        padding: 3px 10px;
    }}

    QDialog#ModelSettingsDialog QPushButton#PrimaryButton:hover {{
        background: {blend_hex("#5B8DEF", SURFACE_ALT, 0.20)};
    }}

    QDialog#ModelSettingsDialog QPushButton#PrimaryButton[savedState="true"] {{
        background: {SUCCESS_GREEN};
        border-color: {blend_hex(SUCCESS_GREEN, "#FFFFFF", 0.18)};
    }}

    QDialog#ModelSettingsDialog QPushButton#PrimaryButton[savedState="true"]:hover {{
        background: {blend_hex(SUCCESS_GREEN, "#FFFFFF", 0.10)};
    }}

    QDialog#ModelSettingsDialog QPushButton {{
        color: {model_dialog_text};
    }}

    QDialog#ApprovalDialog {{
        background: {approval_dialog_bg};
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
        border-top: 1px solid {approval_dialog_border};
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

    QWidget#TranscriptContainer {{
        background: transparent;
    }}

    QFrame#UserChoiceCard {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.04)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QFrame#TranscriptRow,
    QFrame#ToolRow,
    QFrame#ToolGroupFrame,
    QFrame#FlatNoticeRow,
    QFrame#InlineStatusRow {{
        background: transparent;
        border: none;
    }}

    QWidget#ToolGroupContainer {{
        background: transparent;
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
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.22)};
        font-weight: 600;
        font-size: 11pt;
        padding-left: 2px;
    }}

    QLabel#SidebarEmptyState {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.28)};
        font-size: 9.2pt;
        padding: 18px 10px;
    }}

    QLabel#TranscriptRole {{
        color: {TEXT_MUTED};
        font-weight: normal;
        font-size: 9.5pt;
    }}

    QLabel#TranscriptBody {{
        color: {TEXT_PRIMARY};
        font-size: 11pt;
        background: transparent;
    }}

    QLabel#TranscriptMeta {{
        color: {TEXT_MUTED};
        font-size: 8.5pt;
        background: transparent;
    }}

    QLabel#ComposerCapabilityBadge {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.18)};
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.04)};
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
        color: {AMBER_WARNING};
    }}

    QLabel#ComposerNoticeLabel[severity="error"] {{
        color: {ERROR_RED};
    }}

    QLabel#ComposerNoticeLabel[severity="success"] {{
        color: {SUCCESS_GREEN};
    }}

    QLabel#UserChoiceCardTitle {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.30)};
        font-size: 8.8pt;
        font-weight: 650;
        background: transparent;
    }}

    QLabel#UserChoiceCardQuestion {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.44)};
        font-size: 9pt;
        background: transparent;
    }}

    QLabel#UserChoiceCardHint {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.34)};
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
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.42)};
        font-size: 9.7pt;
        background: transparent;
    }}

    QLabel#ApprovalCardImpacts {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.28)};
        font-size: 8.9pt;
        background: transparent;
    }}

    QLabel#ApprovalRiskBadge {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.06)};
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.34)};
        border: none;
        border-radius: {SOFT_RADIUS_XS + 6}px;
        padding: 2px 8px;
        font-size: 8.1pt;
        font-weight: 700;
    }}

    QLabel#ApprovalRiskBadge[riskLevel="low"] {{
        background: {blend_hex(SUCCESS_GREEN, SURFACE_ALT, 0.82)};
        color: {blend_hex(SUCCESS_GREEN, TEXT_PRIMARY, 0.18)};
    }}

    QLabel#ApprovalRiskBadge[riskLevel="medium"] {{
        background: {blend_hex(AMBER_WARNING, SURFACE_ALT, 0.82)};
        color: {blend_hex(AMBER_WARNING, TEXT_PRIMARY, 0.14)};
    }}

    QLabel#ApprovalRiskBadge[riskLevel="high"],
    QLabel#ApprovalRiskBadge[riskLevel="critical"] {{
        background: {blend_hex(ERROR_RED, SURFACE_ALT, 0.82)};
        color: {blend_hex(ERROR_RED, TEXT_PRIMARY, 0.18)};
    }}

    QFrame#ApprovalToolCard {{
        background: {approval_dialog_soft};
        border: 1px solid {approval_dialog_border};
        border-radius: {SOFT_RADIUS_MD + 2}px;
    }}

    QLabel#ApprovalToolTitle {{
        color: {TEXT_PRIMARY};
        font-size: 9.7pt;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#ApprovalToolSubtitle {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.26)};
        font-size: 8.9pt;
        background: transparent;
    }}

    QFrame#ApprovalRequestCard QPushButton#PrimaryButton,
    QDialog#ApprovalDialog QPushButton#PrimaryButton {{
        min-height: 28px;
        padding: 3px 11px;
        border-radius: {SOFT_RADIUS_MD + 2}px;
        background: {blend_hex("#5B8DEF", SURFACE_ALT, 0.14)};
        color: {TEXT_PRIMARY};
        border: 1px solid {blend_hex("#5B8DEF", "#FFFFFF", 0.10)};
        font-weight: 700;
    }}

    QFrame#ApprovalRequestCard QPushButton#PrimaryButton:hover,
    QDialog#ApprovalDialog QPushButton#PrimaryButton:hover {{
        background: {blend_hex("#5B8DEF", SURFACE_ALT, 0.22)};
    }}

    QFrame#ApprovalRequestCard QPushButton#SecondaryButton,
    QDialog#ApprovalDialog QPushButton#SecondaryButton {{
        min-height: 28px;
        padding: 3px 11px;
        border-radius: {SOFT_RADIUS_MD + 2}px;
        background: {approval_dialog_soft};
        color: {TEXT_PRIMARY};
        border: 1px solid {approval_dialog_border};
        font-weight: 600;
    }}

    QFrame#ApprovalRequestCard QPushButton#DangerButton,
    QDialog#ApprovalDialog QPushButton#DangerButton {{
        min-height: 28px;
        padding: 3px 11px;
        border-radius: {SOFT_RADIUS_MD + 2}px;
        background: {blend_hex(ERROR_RED, SURFACE_ALT, 0.78)};
        color: {TEXT_PRIMARY};
        border: 1px solid {blend_hex(ERROR_RED, "#FFFFFF", 0.08)};
        font-weight: 700;
    }}

    QPlainTextEdit#ApprovalDetailView {{
        background: {approval_dialog_code};
        color: {TEXT_PRIMARY};
        border: 1px solid {approval_dialog_border};
        border-radius: {SOFT_RADIUS_MD + 2}px;
        padding: 10px;
        selection-background-color: {blend_hex("#5B8DEF", SURFACE_ALT, 0.20)};
    }}

    QPlainTextEdit#ApprovalArgsView,
    QPlainTextEdit#InlineCodeView {{
        background: {approval_dialog_code};
        color: {CODE_TEXT};
        border: 1px solid {approval_dialog_border};
        border-radius: {SOFT_RADIUS_MD + 2}px;
        padding: 8px;
        selection-background-color: {blend_hex("#5B8DEF", SURFACE_ALT, 0.20)};
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

    QLabel#MetaText[severity="error"] {{
        color: {ERROR_RED};
    }}

    QLabel#WarningText {{
        color: {AMBER_WARNING};
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
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.08)};
    }}

    QFrame#InlineStatusRow[phase="success"] QLabel#TranscriptMeta {{
        color: {SUCCESS_GREEN};
    }}

    QFrame#InlineStatusRow[phase="waiting"] QLabel#TranscriptMeta {{
        color: {blend_hex(AMBER_WARNING, TEXT_PRIMARY, 0.24)};
    }}

    QFrame#InlineStatusRow QLabel#TranscriptMeta {{
        color: {TEXT_MUTED};
    }}

    QToolButton#InlineStatusSpinner {{
        background: transparent;
        border: none;
        padding: 0px;
    }}

    QTabWidget::pane {{
        border: 1px solid {model_dialog_border};
        border-radius: {SOFT_RADIUS_MD + 4}px;
        background: transparent;
        top: -1px;
    }}

    QTabBar::tab {{
        background: transparent;
        border: 1px solid {model_dialog_border};
        border-bottom: 1px solid {model_dialog_border};
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
        border: 1px solid {model_dialog_selected_border};
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
        background: {blend_hex(SURFACE_ALT, ACCENT_BLUE_SOFT, 0.55)};
        border: none;
    }}

    QToolButton:pressed,
    QPushButton:pressed {{
        background: {blend_hex(SURFACE_ALT, ACCENT_BLUE_SOFT, 0.8)};
    }}

    QPushButton#PrimaryButton {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.08)};
        color: white;
        border: none;
        font-weight: 600;
    }}

    QPushButton#SecondaryButton {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.04)};
        color: {TEXT_PRIMARY};
        border: none;
        font-weight: 600;
    }}

    QPushButton#DangerButton {{
        background: {ERROR_RED};
        color: white;
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
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.06)};
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
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.45)};
    }}

    QFrame#ToolExpandablePanel {{
        background: {tool_panel_bg};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 5px 8px;
    }}

    QFrame#ToolExpandablePanel[variant="thoughts"] {{
        background: transparent;
        border: none;
        border-radius: 0px;
        padding: 0px;
    }}

    QFrame#ToolExpandablePanel[variant="diff"] {{
        background: transparent;
        border: none;
        border-radius: 0px;
        padding: 2px 0px 0px 0px;
    }}

    QFrame#CliExecPanel {{
        background: {tool_panel_bg};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.04)};
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QLabel#CliExecHeader {{
        background: transparent;
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.1)};
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9.3pt;
        font-weight: 500;
    }}

    QLabel#ToolSubtitle {{
        background: transparent;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.30)};
        font-size: 8.9pt;
        padding-left: 13px;
        padding-bottom: 1px;
    }}

    QLabel#ToolActionLabel {{
        background: transparent;
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.18)};
        font-size: 10.15pt;
        font-weight: 500;
        padding: 1px 0px;
    }}

    QLabel#ToolActionLabel[toolRole="edit"] {{
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.08)};
    }}

    QLabel#ToolActionLabel[phase="active"] {{
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.12)};
    }}

    QLabel#ToolActionLabel[phase="success"] {{
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.22)};
    }}

    QLabel#ToolActionLabel[phase="error"] {{
        color: {blend_hex(ERROR_RED, TEXT_PRIMARY, 0.18)};
    }}

    QLabel#ToolPhaseBadge {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.04)};
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.34)};
        border: none;
        border-radius: {SOFT_RADIUS_XS + 5}px;
        padding: 2px 7px;
        font-size: 8.1pt;
        font-weight: 600;
    }}

    QLabel#ToolPhaseBadge[variant="pending"] {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.05)};
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.26)};
    }}

    QLabel#ToolPhaseBadge[variant="active"] {{
        background: {blend_hex(ACCENT_BLUE_SOFT, ACCENT_BLUE, 0.18)};
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.04)};
    }}

    QLabel#ToolPhaseBadge[variant="success"] {{
        background: transparent;
        color: {SUCCESS_GREEN};
        padding: 0px;
        border-radius: 0px;
        font-size: 9.2pt;
        font-weight: 700;
    }}

    QLabel#ToolPhaseBadge[variant="error"] {{
        background: {blend_hex(ERROR_RED, SURFACE_ALT, 0.18)};
        color: {TEXT_PRIMARY};
    }}

    QPlainTextEdit#CliExecOutput {{
        background: {tool_panel_bg};
        color: {CODE_TEXT};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 7px 10px 8px 10px;
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
    }}

    QFrame#DiffPanel {{
        background: {tool_panel_bg};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.05)};
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QWidget#DiffHeaderRow {{
        background: {tool_panel_bg};
        border-top-left-radius: {SOFT_RADIUS_SM}px;
        border-top-right-radius: {SOFT_RADIUS_SM}px;
    }}

    QLabel#DiffHeaderPath {{
        background: transparent;
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.14)};
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9.05pt;
        font-weight: 500;
    }}

    QLabel#DiffStatAdded {{
        background: transparent;
        color: #7EDB8B;
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
        font-weight: 600;
    }}

    QLabel#DiffStatRemoved {{
        background: transparent;
        color: #F08F8F;
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
        font-weight: 600;
    }}

    QPlainTextEdit#DiffCodeView {{
        background: {tool_panel_bg};
        color: {CODE_TEXT};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 6px 10px 9px 10px;
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
    }}

    QWidget#ToolExpandableContent {{
        background: transparent;
        border: none;
    }}

    QWidget#ToolExpandableContent[variant="thoughts"] {{
        background: transparent;
        border: none;
    }}

    QWidget#ToolExpandableContent[variant="diff"] {{
        background: transparent;
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
        color: {tool_toggle_text};
        text-align: left;
    }}

    QPushButton#ToolExpandableToggle:hover {{
        background: {tool_panel_hover};
        color: {tool_toggle_hover_text};
    }}

    QPushButton#ToolExpandableToggle:checked {{
        background: {tool_panel_hover};
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
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.14)};
        text-align: left;
    }}

    QPushButton#ToolExpandableToggle[variant="thoughts"]:hover,
    QPushButton#ToolExpandableToggle[variant="thoughts"]:checked {{
        background: transparent;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.24)};
    }}

    QPushButton#ToolExpandableToggle[variant="diff"] {{
        background: transparent;
        border: none;
        border-radius: 0px;
        padding: 3px 0px 4px 0px;
        min-height: 18px;
        font-size: 9.2pt;
        font-weight: 500;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.32)};
        text-align: left;
    }}

    QPushButton#ToolExpandableToggle[variant="diff"]:hover,
    QPushButton#ToolExpandableToggle[variant="diff"]:checked {{
        background: transparent;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.48)};
    }}

    QToolButton#CodeCopyButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 2px 6px;
        color: {tool_toggle_text};
        font-size: 8.8pt;
    }}

    QToolButton#CodeCopyButton:hover {{
        background: {tool_panel_hover};
        color: {TEXT_PRIMARY};
    }}

    QPlainTextEdit,
    QTextBrowser,
    QListView,
    QListWidget,
    QTreeWidget {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.01)};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        selection-background-color: {blend_hex(ACCENT_BLUE_SOFT, ACCENT_BLUE, 0.25)};
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
        background: {_COMPOSER_BG};
        border: none;
        border-radius: 18px;
    }}

    QPushButton#UserChoiceOptionButton,
    QPushButton#UserChoiceCustomButton {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.08)};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.10)};
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 7px 10px;
        color: {TEXT_PRIMARY};
        text-align: center;
        font-size: 9.1pt;
        font-weight: 650;
    }}

    QPushButton#UserChoiceOptionButton:hover,
    QPushButton#UserChoiceCustomButton:hover {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.15)};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.20)};
    }}

    QPushButton#UserChoiceOptionButton:pressed,
    QPushButton#UserChoiceCustomButton:pressed {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.19)};
    }}

    QPushButton#UserChoiceOptionButton[recommended="true"] {{
        background: {blend_hex(ACCENT_BLUE, SURFACE_ALT, 0.28)};
        color: {TEXT_PRIMARY};
        border: 1px solid {blend_hex(ACCENT_BLUE, "#FFFFFF", 0.10)};
    }}

    QPushButton#UserChoiceOptionButton:disabled,
    QPushButton#UserChoiceCustomButton:disabled {{
        background: {blend_hex(SURFACE_ALT, SURFACE_CARD, 0.35)};
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.18)};
    }}

    QPlainTextEdit#ComposerEdit {{
        background: transparent;
        border: none;
        border-radius: 10px;
        padding: 4px 2px 2px 2px;
        font-size: 10.5pt;
    }}

    QPlainTextEdit#ComposerEdit QScrollBar:vertical {{
        background: transparent;
        border: none;
        width: 6px;
        margin: 2px 0 2px 0;
    }}

    QPlainTextEdit#ComposerEdit QScrollBar::handle:vertical {{
        background: {blend_hex(TEXT_MUTED, "#FFFFFF", 0.06)};
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
        background: {_SEND_BTN_BG};
        color: #08090B;
        border: 1px solid {blend_hex(_SEND_BTN_BG, "#000000", 0.18)};
        border-radius: 15px;
        min-width: 32px;
        max-width: 32px;
        min-height: 32px;
        max-height: 32px;
        padding: 0px;
        font-weight: 700;
    }}

    QPushButton#ComposerSendButton:hover {{
        background: {_SEND_BTN_HOVER};
        border: 1px solid {blend_hex(_SEND_BTN_HOVER, "#000000", 0.20)};
    }}

    QPushButton#ComposerSendButton:pressed {{
        background: #C8CACF;
        border: 1px solid {blend_hex("#C8CACF", "#000000", 0.22)};
    }}

    QPushButton#ComposerSendButton:disabled {{
        background: {_SEND_BTN_DISABLED};
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.2)};
        border: 1px solid {blend_hex(_SEND_BTN_DISABLED, "#FFFFFF", 0.08)};
    }}

    QPushButton#ComposerStopButton {{
        background: {_SEND_BTN_BG};
        color: #08090B;
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
        background: {_SEND_BTN_HOVER};
    }}

    QPushButton#ComposerStopButton:pressed {{
        background: #C8CACF;
    }}

    QToolButton#ComposerAttachButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 0px;
    }}

    QToolButton#ComposerAttachButton:hover {{
        background: {blend_hex(_COMPOSER_BG, SURFACE_ALT, 0.32)};
        border: none;
    }}

    QPushButton#ComposerMetaChip {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 2px 8px;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.22)};
        font-size: 8.8pt;
        font-weight: 500;
    }}

    QPushButton#ComposerMetaChip:disabled {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.22)};
        background: transparent;
        border: 1px solid transparent;
    }}

    QToolButton#ComposerMetaChipButton {{
        background: {blend_hex(_COMPOSER_BG, "#FFFFFF", 0.04)};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.08)};
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
        background: {blend_hex(_COMPOSER_BG, SURFACE_ALT, 0.36)};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.14)};
    }}

    QToolButton#ComposerMetaChipButton[chipPosition="right"]:hover {{
        border-left: none;
    }}

    QToolButton#ComposerMetaChipButton:disabled {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.22)};
        background: transparent;
        border: 1px solid transparent;
    }}

    QLabel#ComposerNoModelText {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.32)};
        font-size: 8.8pt;
    }}

    QLabel#ComposerCacheHitLabel {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.18)};
        background: transparent;
        padding: 0px 2px 0px 0px;
        font-size: 8.4pt;
    }}

    QPushButton#ComposerOpenSettingsButton {{
        background: transparent;
        border: 1px solid {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.15)};
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 2px 8px;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.45)};
        font-size: 8.8pt;
    }}

    QPushButton#ComposerOpenSettingsButton:hover {{
        background: {blend_hex(_COMPOSER_BG, SURFACE_ALT, 0.24)};
        border: 1px solid {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.3)};
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
        background: {_POPUP_BG};
        border: 1px solid {blend_hex(_POPUP_BORDER, SURFACE_BG, 0.55)};
        border-radius: {SOFT_RADIUS_MD + 10}px;
    }}

    QLabel#ComposerMentionHeader {{
        background: transparent;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.3)};
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
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.2)};
        border: 1px solid {blend_hex(TEXT_PRIMARY, SURFACE_ALT, 0.78)};
    }}

    QWidget#ComposerMentionItem[selected="true"] QLabel#ComposerMentionItemTitle {{
        color: {blend_hex(TEXT_PRIMARY, "#FFFFFF", 0.16)};
    }}

    QWidget#ComposerMentionItem[selected="true"] QLabel#ComposerMentionItemMeta {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.42)};
    }}

    QLabel#ComposerMentionItemTitle {{
        background: transparent;
        color: {TEXT_PRIMARY};
        font-size: 10.2pt;
        font-weight: 600;
    }}

    QLabel#ComposerMentionItemMeta {{
        background: transparent;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.12)};
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
        background: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.1)};
        border-radius: 4px;
        min-height: 28px;
    }}

    QListWidget#ComposerMentionList QScrollBar::handle:vertical:hover {{
        background: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.2)};
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
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.08)};
        color: {TEXT_PRIMARY};
        border: none;
        border-radius: {SOFT_RADIUS_MD + 8}px;
        padding: 7px 12px;
        font-size: 8.9pt;
        font-weight: 600;
    }}

    QPushButton#TranscriptJumpButton:hover {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.08)};
    }}

    QTextBrowser#AssistantBody {{
        background: transparent;
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
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.06)};
        font-size: 10.7pt;
        font-weight: 500;
    }}

    QPushButton#ToolCallButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_XS}px;
        padding: 0px;
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.20)};
        font-size: 10.35pt;
        font-weight: 500;
        text-align: left;
    }}

    QPushButton#ToolCallButton[toolRole="edit"] {{
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.10)};
    }}

    QPushButton#ToolCallButton:hover {{
        background: transparent;
        color: {tool_call_hover};
        border: none;
    }}

    QPushButton#ToolCallButton:checked {{
        background: transparent;
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.08)};
        border: none;
    }}

    QPushButton#ToolGroupHeaderButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 1px 0px;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.28)};
        font-size: 10pt;
        font-weight: 500;
        text-align: left;
    }}

    QPushButton#ToolGroupHeaderButton[state="active"] {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.46)};
    }}

    QPushButton#ToolGroupHeaderButton[state="complete"] {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.38)};
    }}

    QPushButton#ToolGroupHeaderButton[state="error"] {{
        color: {blend_hex(ERROR_RED, TEXT_PRIMARY, 0.22)};
    }}

    QPushButton#ToolGroupHeaderButton:hover {{
        background: transparent;
        color: {tool_toggle_hover_text};
        border: none;
    }}

    QPushButton#ToolGroupHeaderButton:checked {{
        background: transparent;
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.08)};
        border: none;
    }}

    QPlainTextEdit#CodeView,
    QPlainTextEdit#InlineCodeView {{
        background: {tool_code_bg};
        color: {CODE_TEXT};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.04)};
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
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.02)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 10px 12px;
    }}

    QFrame#ToolCard {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.04)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QCheckBox#ToolAvailabilitySwitch {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.08)};
        border-radius: 10px;
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.16)};
        padding: 0px;
        spacing: 0px;
    }}

    QCheckBox#ToolAvailabilitySwitch:hover {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.16)};
    }}

    QCheckBox#ToolAvailabilitySwitch:checked {{
        background: #FFFFFF;
        border: 1px solid #FFFFFF;
    }}

    QCheckBox#ToolAvailabilitySwitch:disabled {{
        background: {SURFACE_BG};
        border-color: {BORDER};
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

    QLabel#ToolGroupHeader {{
        color: {TEXT_MUTED};
        font-size: 7.2pt;
        font-weight: 700;
        letter-spacing: 0.8px;
        padding: 8px 4px 3px 4px;
    }}

    QLabel#ToolGroupHeader[toolGroup="protected"] {{
        color: {AMBER_WARNING};
    }}

    QLabel#ToolGroupHeader[toolGroup="mcp"] {{
        color: {ACCENT_BLUE};
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
        color: {ACCENT_BLUE};
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
        color: {ACCENT_BLUE};
        font-size: 7.5pt;
        background: transparent;
    }}

    QLabel#ToolFlagChip {{
        color: {TEXT_MUTED};
        font-size: 7pt;
        border: 1px solid {blend_hex(TEXT_MUTED, "#FFFFFF", 0.14)};
        border-radius: 3px;
        padding: 0px 4px;
        background: transparent;
    }}

    QLabel#ToolFlagChip[flagVariant="warning"] {{
        color: {AMBER_WARNING};
        border: 1px solid {blend_hex(AMBER_WARNING, SURFACE_CARD, 0.48)};
    }}

    QLabel#ToolFlagChip[flagVariant="accent"] {{
        color: {ACCENT_BLUE};
        border: 1px solid {blend_hex(ACCENT_BLUE, SURFACE_CARD, 0.56)};
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
        background: {blend_hex(TEXT_MUTED, "#FFFFFF", 0.06)};
        border-radius: {SOFT_RADIUS_XS}px;
        min-height: 16px;
    }}

    QScrollBar::handle:horizontal {{
        background: {blend_hex(TEXT_MUTED, "#FFFFFF", 0.06)};
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
        background: {SURFACE_BG};
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
        background: {_POPUP_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {_POPUP_BORDER};
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
        background: {_POPUP_ITEM_SELECTED};
    }}

    QMenu#ComposerPopupMenu::separator {{
        height: 1px;
        background: {blend_hex(BORDER, "#FFFFFF", 0.08)};
        margin: 6px 8px;
    }}

    QMenuBar::item:selected,
    QMenu::item:selected {{
        background: {ACCENT_BLUE_SOFT};
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QStatusBar {{
        background: {blend_hex(SURFACE_BG, SURFACE_CARD, 0.28)};
        color: {TEXT_MUTED};
        border-top: 1px solid {BORDER};
        min-height: 22px;
        padding-left: 6px;
    }}

    QStatusBar::item {{
        border: none;
    }}

    QLabel#StatusBarMeta {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.16)};
        font-size: 9pt;
        padding-right: 4px;
    }}

    QLabel#StatusBarState {{
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.12)};
        font-size: 9.6pt;
        font-weight: 600;
        padding-left: 2px;
    }}
    """
