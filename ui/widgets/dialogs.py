from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRect, QSize, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen, QStandardItem
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QStyleOptionButton,
    QStyleOptionSlider,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.model_fetcher import (
    AnthropicModelFetcher,
    AuthError,
    EmptyResultError,
    FetchError,
    GeminiModelFetcher,
    InvalidResponseError,
    ModelEntry,
    ModelFetcher,
    NetworkError,
    OpenAICompatibleModelFetcher,
    RateLimitError,
    ServerError,
)
from core.model_profiles import (
    ALLOWED_PROVIDERS,
    ensure_unique_profile_id,
    generate_profile_id,
    normalize_api_key_list,
    normalize_profiles_payload,
    sanitize_profile_id,
)
from core.text_utils import build_tool_ui_labels
from ui.theme import TEXT_MUTED, TEXT_PRIMARY
from .foundation import (
    CollapsibleSection,
    CopySafePlainTextEdit,
    ElidedLabel,
    _fa_icon,
    _make_mono_font,
    _sync_plain_text_height,
    format_approval_detail_text,
)


class ModelLoadState(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    LOADED = "loaded"
    FALLBACK = "fallback"
    ERROR = "error"


class ImageSupportCheckBox(QCheckBox):
    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self.isChecked():
            return

        option = QStyleOptionButton()
        self.initStyleOption(option)
        indicator_rect = self.style().subElementRect(
            QStyle.SubElement.SE_CheckBoxIndicator,
            option,
            self,
        )

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(TEXT_PRIMARY if self.isEnabled() else TEXT_MUTED))
        pen.setWidthF(1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        x = indicator_rect.x()
        y = indicator_rect.y()
        width = indicator_rect.width()
        height = indicator_rect.height()
        first = QPointF(x + width * 0.24, y + height * 0.56)
        middle = QPointF(x + width * 0.44, y + height * 0.74)
        last = QPointF(x + width * 0.78, y + height * 0.32)
        painter.drawLine(first, middle)
        painter.drawLine(middle, last)


def _fetch_error_message(error: FetchError) -> str:
    if isinstance(error, AuthError):
        return "Invalid API key. Check the key and try again."
    if isinstance(error, RateLimitError):
        return "Rate limit exceeded. Please wait and try again."
    if isinstance(error, ServerError):
        return "Server error. Please try again later."
    if isinstance(error, NetworkError):
        return "No connection. Check your network."
    if isinstance(error, EmptyResultError):
        return "No models are available for this API key."
    if isinstance(error, InvalidResponseError):
        return "The provider returned a non-JSON response. Check the Base URL and /models support."
    return "Failed to load models."


class ModelFetchWorker(QThread):
    fetched = Signal(int, object)
    failed = Signal(int, str)

    def __init__(
        self,
        request_id: int,
        fetcher: ModelFetcher,
        api_key: str,
        base_url: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(None)
        self._request_id = int(request_id)
        self._fetcher = fetcher
        self._api_key = str(api_key or "").strip()
        self._base_url = str(base_url or "").strip()

    def run(self) -> None:
        try:
            result = asyncio.run(self._fetcher.fetch(self._api_key, self._base_url))
        except FetchError as error:
            self.failed.emit(self._request_id, _fetch_error_message(error))
            return
        except Exception:
            self.failed.emit(self._request_id, "Failed to load models.")
            return
        self.fetched.emit(self._request_id, result)


class SearchableModelComboBox(QComboBox):
    popup_requested = Signal()

    def showPopup(self) -> None:
        self.popup_requested.emit()


class ResponsiveProfileList(QListWidget):
    """Keep profile cards inside the list viewport when the pane is resized."""

    @staticmethod
    def _card_height(widget: QWidget | None, width: int, fallback: int) -> int:
        """Height for a profile card at *width*, honoring wrapped text.

        A word-wrapped QLabel does not grow its sizeHint(): the base card hint
        assumes a single text line. The extra height required by the wrapped
        lines comes from heightForWidth() and must be ADDED to the base hint —
        taking max() alone leaves the model-name line overlapping the title.
        """
        height = fallback
        if widget is None:
            return height
        height = max(height, widget.sizeHint().height(), widget.minimumSizeHint().height())
        extra = 0
        for label in widget.findChildren(QLabel):
            if not label.wordWrap() or not label.isVisibleTo(widget):
                continue
            # The label geometry is up to date: the caller activated the card
            # layout after assigning the viewport width. hasHeightForWidth()
            # is unreliable for QLabel (may report False even when
            # heightForWidth() returns the wrapped height), so query it directly.
            text_width = label.width() - label.margin() * 2
            if text_width <= 0:
                continue
            wrapped_height = label.heightForWidth(text_width) + label.margin() * 2
            single_line_height = label.fontMetrics().height()
            extra += max(0, wrapped_height - single_line_height)
        return height + extra

    # Right gutter kept free so profile cards never extend under the
    # overlay-style vertical scrollbar: its transparent track hides the
    # cards' right border and clips the selected card's accent border.
    SCROLLBAR_GUTTER = 10

    def _fit_items_to_viewport(self) -> None:
        viewport_width = self.viewport().width()
        if viewport_width <= 0:
            return
        card_width = viewport_width - self.SCROLLBAR_GUTTER
        for row in range(self.count()):
            item = self.item(row)
            if item is None:
                continue
            item_widget = self.itemWidget(item)
            if item_widget is None:
                continue
            if item_widget.width() != card_width:
                item_widget.setFixedWidth(card_width)
            # Activate the card layout at the new width so child label geometry
            # (and therefore the wrapped-text height below) is up to date.
            if item_widget.layout() is not None:
                item_widget.layout().activate()
            # Base the height on the widget's own hint (NOT the previous item
            # hint) so repeated calls cannot accumulate extra wrapped height.
            base = max(
                72,
                item_widget.sizeHint().height(),
                item_widget.minimumSizeHint().height(),
            )
            height = self._card_height(item_widget, card_width, base) + 6
            item.setSizeHint(QSize(viewport_width, height))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_items_to_viewport()


class _SliderScaleWidget(QWidget):
    """Scale labels aligned with a QSlider's handle positions.

    QSlider draws its built-in ticks across the full groove width, but the
    handle travels a reduced span (groove width minus handle width). This
    widget mirrors the style's handle-position formula so every label sits
    exactly under the handle position for its value.
    """

    def __init__(self, slider: QSlider, labels: tuple[str, ...], values: tuple[int, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._slider = slider
        self._labels = tuple(labels)
        self._values = tuple(values)
        self.setMinimumHeight(18)

    def _handle_center_x(self, value: int) -> float:
        """X coordinate of the slider handle center for the given value."""
        slider = self._slider
        opt = QStyleOptionSlider()
        slider.initStyleOption(opt)
        groove = slider.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, slider)
        handle = slider.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, slider)
        # Map the scale value into slider units (the slider works in steps).
        slider_min, slider_max = slider.minimum(), slider.maximum()
        value_min, value_max = min(self._values), max(self._values)
        fraction = (value - value_min) / (value_max - value_min) if value_max > value_min else 0.0
        slider_value = slider_min + fraction * (slider_max - slider_min)
        span = groove.width() - handle.width()
        return groove.left() + handle.width() / 2 + span * (slider_value - slider_min) / (slider_max - slider_min)

    def _slider_left_offset(self) -> int:
        """X offset of the slider's groove start relative to this widget."""
        slider = self._slider
        opt = QStyleOptionSlider()
        slider.initStyleOption(opt)
        groove = slider.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, slider)
        return groove.left()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setPen(QColor(TEXT_MUTED))
        font = self._slider.font()
        metrics = QFontMetrics(font)
        slider_offset = self._slider_left_offset()
        for label, value in zip(self._labels, self._values):
            x = slider_offset + self._handle_center_x(value)
            text_width = metrics.horizontalAdvance(label)
            # First label anchors left-aligned at its tick, last label
            # right-aligned, middle labels centered — keeps the scale inside
            # the widget bounds while pointing at the handle positions.
            if value == min(self._values):
                draw_x = x
            elif value == max(self._values):
                draw_x = x - text_width
            else:
                draw_x = x - text_width / 2
            painter.drawText(int(draw_x), int(self.height() * 0.75), label)
        painter.end()

    def sizeHint(self) -> QSize:
        return QSize(self._slider.sizeHint().width(), 18)


class ModelSettingsDialog(QDialog):
    profiles_saved = Signal(object)
    # Emitted when the panel is shown/hidden so the main window can block its
    # own input while the panel is open.
    visibility_changed = Signal(bool)

    # Upper bound for the window size. The actual size scales with the screen
    # (see WIDTH_RATIO/HEIGHT_RATIO) and is clamped to the available area so the
    # panel never overflows low-resolution displays (e.g. 1366x768).
    PREFERRED_SIZE = QSize(1080, 720)
    MINIMUM_SIZE = QSize(720, 480)
    # Fraction of the available screen area the window aims to occupy. This
    # keeps the panel proportional across resolutions instead of a fixed size.
    WIDTH_RATIO = 0.78
    HEIGHT_RATIO = 0.86
    # Gap kept to the screen edges when clamping the window to the screen.
    SCREEN_MARGIN = 24
    # Height of the top strip that drags the frameless window.
    TITLE_BAR_HEIGHT = 52

    def __init__(self, payload: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ModelSettingsDialog")
        self.setWindowTitle("Settings")
        # Application-modal: while the panel is open the main window (composer,
        # sidebar, transcript) must not accept input. The panel may overlap the
        # chat sidebar, which is expected.
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        # Keep the instance alive across close/reopen so the panel can be
        # re-shown with its state instead of being rebuilt every time.
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        # Frameless top-level window: the panel is shown centered on screen
        # instead of being docked to the main window.
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self._drag_offset: QPoint | None = None
        self._apply_screen_geometry()

        normalized = normalize_profiles_payload(payload or {})
        self._profiles: list[dict[str, Any]] = [dict(item) for item in normalized.get("profiles", [])]
        self._active_profile = str(normalized.get("active_profile") or "").strip()
        self._name_manual_flags: list[bool] = []
        self._model_manual_flags: list[bool] = []
        self._selected_row = -1
        self._loading_form = False
        self._filter_text = ""
        self._result_payload = normalized
        self._form_enabled = False
        self._model_state = ModelLoadState.IDLE
        self._model_cache: dict[tuple[str, ...], list[ModelEntry]] = {}
        self._model_entries_by_id: dict[str, ModelEntry] = {}
        self._model_workers: list[ModelFetchWorker] = []
        self._model_popup: QFrame | None = None
        self._model_popup_search: QLineEdit | None = None
        self._model_popup_list: QListWidget | None = None
        self._fetch_request_id = 0
        self._fetch_debounce = QTimer(self)
        self._fetch_debounce.setSingleShot(True)
        self._fetch_debounce.setInterval(600)
        self._fetch_debounce.timeout.connect(self._start_fetch)
        self._save_button_reset_timer = QTimer(self)
        self._save_button_reset_timer.setSingleShot(True)
        self._save_button_reset_timer.setInterval(3000)
        self._save_button_reset_timer.timeout.connect(self._restore_save_button_text)
        self._name_manual_flags = self._compute_initial_name_manual_flags()
        self._model_manual_flags = [False] * len(self._profiles)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        hero_card = QFrame()
        hero_card.setObjectName("ModelSettingsHeroCard")
        # The hero card doubles as the frameless window's title bar: dragging it
        # moves the whole panel.
        self._title_bar = hero_card
        hero_layout = QHBoxLayout(hero_card)
        hero_layout.setContentsMargins(10, 8, 10, 8)
        hero_layout.setSpacing(10)

        hero_copy = QVBoxLayout()
        hero_copy.setContentsMargins(0, 0, 0, 0)
        hero_copy.setSpacing(3)

        header_title = QLabel("Settings")
        header_title.setObjectName("ModelSettingsTitle")
        hero_copy.addWidget(header_title, 0, Qt.AlignLeft | Qt.AlignTop)

        active_name = str(self._active_profile or "").strip() or "none"
        self.active_profile_label = QLabel(f"Active: {active_name}")
        self.active_profile_label.setObjectName("ModelSettingsMeta")
        self.active_profile_label.setWordWrap(True)
        hero_copy.addWidget(self.active_profile_label)
        hero_layout.addLayout(hero_copy, 1)

        self.close_button = QPushButton()
        self.close_button.setObjectName("SettingsCloseButton")
        self.close_button.setIcon(_fa_icon("fa5s.times", color=TEXT_MUTED, size=12))
        self.close_button.setFixedSize(28, 28)
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_button.setToolTip("Close")
        self.close_button.setAccessibleName("Close settings")
        self.close_button.clicked.connect(self.reject)
        hero_layout.addWidget(self.close_button, 0, Qt.AlignTop | Qt.AlignRight)

        root.addWidget(hero_card)

        self.tabs = QTabWidget()
        self.tabs.setAccessibleName("Settings tabs")
        self.tabs.setAccessibleDescription("Switch between model settings and future settings")
        self.models_page = QWidget()
        self.models_page.setAccessibleName("Models settings")
        models_page_layout = QVBoxLayout(self.models_page)
        models_page_layout.setContentsMargins(0, 0, 0, 0)
        models_page_layout.setSpacing(0)
        self.test_page = QWidget()
        self.test_page.setAccessibleName("Test settings")
        self._build_test_page()
        self.tabs.addTab(self.models_page, _fa_icon("fa5s.cubes", color=TEXT_MUTED, size=14), "Models")
        self.tabs.addTab(self.test_page, _fa_icon("fa5s.flask", color=TEXT_MUTED, size=14), "Test")
        root.addWidget(self.tabs, 1)

        self.body_splitter = QSplitter(Qt.Horizontal)
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.setHandleWidth(8)

        left_container = QFrame()
        left_container.setObjectName("ModelSettingsPane")
        left_container.setProperty("paneRole", "library")
        left = QVBoxLayout(left_container)
        left.setContentsMargins(18, 12, 18, 12)
        left.setSpacing(8)

        left_header = QHBoxLayout()
        left_header.setContentsMargins(0, 0, 0, 0)
        left_header.setSpacing(4)
        left_label = QLabel("Profiles")
        left_label.setObjectName("ModelSettingsSectionTitle")
        left_header.addWidget(left_label, 0, Qt.AlignLeft | Qt.AlignVCenter)
        left_header.addStretch(1)
        left.addLayout(left_header)

        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("ModelSettingsSearchField")
        self.search_edit.setPlaceholderText("Search by name, provider, or model")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setAccessibleName("Profile search")
        self.search_edit.setAccessibleDescription("Filter profiles by name, provider, or model")
        self.search_edit.addAction(_fa_icon("fa5s.search", color=TEXT_MUTED, size=10), QLineEdit.LeadingPosition)
        left.addWidget(self.search_edit)

        self.profile_list = ResponsiveProfileList()
        self.profile_list.setObjectName("ModelProfileList")
        self.profile_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.profile_list.setMinimumWidth(300)
        self.profile_list.setAccessibleName("Profile list")
        self.profile_list.setAccessibleDescription("Select a model profile to edit")
        self.profile_list.currentRowChanged.connect(self._on_selection_changed)
        left.addWidget(self.profile_list, 1)

        left_buttons = QHBoxLayout()
        left_buttons.setSpacing(4)
        self.add_button = QPushButton("New Profile")
        self.add_button.setObjectName("SettingsAddButton")
        self.add_button.setIcon(_fa_icon("fa5s.plus", color=TEXT_PRIMARY, size=10))
        self.delete_button = QPushButton("Remove")
        self.delete_button.setObjectName("SettingsDeleteButton")
        self.delete_button.setIcon(_fa_icon("fa5s.trash", color="#F08F8F", size=10))
        left_buttons.addWidget(self.add_button)
        left_buttons.addWidget(self.delete_button)
        left.addLayout(left_buttons)

        right_container = QFrame()
        right_container.setObjectName("ModelSettingsPane")
        right_container.setProperty("paneRole", "editor")
        right = QVBoxLayout(right_container)
        right.setContentsMargins(18, 12, 18, 12)
        right.setSpacing(8)

        right_header = QHBoxLayout()
        right_header.setContentsMargins(0, 0, 0, 0)
        right_header.setSpacing(4)
        self.selected_profile_title = QLabel("No profile selected")
        self.selected_profile_title.setObjectName("ModelSettingsSectionTitle")
        self.selected_profile_title.setWordWrap(True)
        right_header.addWidget(self.selected_profile_title, 0, Qt.AlignLeft | Qt.AlignVCenter)
        right_header.addStretch(1)

        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.setObjectName("ModelSettingsInlineButton")
        self.duplicate_button.setIcon(_fa_icon("fa5s.copy", color=TEXT_PRIMARY, size=10))
        self.duplicate_button.setEnabled(False)
        right_header.addWidget(self.duplicate_button, 0, Qt.AlignRight | Qt.AlignVCenter)
        right.addLayout(right_header)

        self.form_hint = QLabel("Add a profile to start configuring models.")
        self.form_hint.setObjectName("ModelSettingsMeta")
        self.form_hint.setWordWrap(True)
        right.addWidget(self.form_hint)

        self.save_state_label = QLabel("")
        self.save_state_label.setObjectName("ModelSettingsMeta")
        self.save_state_label.setWordWrap(True)
        self.save_state_label.setVisible(False)
        right.addWidget(self.save_state_label)

        editor_scroll = QScrollArea()
        editor_scroll.setObjectName("ModelSettingsScrollArea")
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setFrameShape(QFrame.NoFrame)

        editor_content = QWidget()
        editor_content.setObjectName("ModelSettingsEditorContent")
        editor_layout = QVBoxLayout(editor_content)
        editor_layout.setContentsMargins(4, 4, 4, 4)
        editor_layout.setSpacing(8)

        form_frame = QFrame()
        form_frame.setObjectName("ModelSettingsFormCard")
        form_layout = QFormLayout(form_frame)
        form_layout.setContentsMargins(12, 10, 12, 10)
        form_layout.setHorizontalSpacing(10)
        form_layout.setVerticalSpacing(8)
        form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        form_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form_layout.setFormAlignment(Qt.AlignTop)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("profile-id")
        self.name_edit.setClearButtonEnabled(True)
        self.name_edit.setAccessibleName("Profile name")

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["openai", "gemini", "anthropic"])
        self.provider_combo.setAccessibleName("Provider")

        self.model_combo = SearchableModelComboBox()
        self.model_combo.setObjectName("ModelSettingsModelCombo")
        self.model_combo.setAccessibleName("Model")
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model_combo.setEditable(False)
        self.model_combo.setMaxVisibleItems(14)

        self.model_text_edit = QLineEdit()
        self.model_text_edit.setPlaceholderText("Enter a model name manually")
        self.model_text_edit.setClearButtonEnabled(True)
        self.model_text_edit.setAccessibleName("Model")
        self.model_edit = self.model_text_edit

        self.model_reload_button = QToolButton()
        self.model_reload_button.setObjectName("ModelSettingsInlineToolButton")
        self.model_reload_button.setIcon(_fa_icon("fa5s.redo-alt", color=TEXT_MUTED, size=10))
        self.model_reload_button.setToolTip("Retry loading models")
        self.model_reload_button.setAccessibleName("Retry model loading")
        self.model_reload_button.setCursor(Qt.PointingHandCursor)

        self.model_popup_button = QToolButton()
        self.model_popup_button.setObjectName("ModelSettingsInlineToolButton")
        self.model_popup_button.setIcon(_fa_icon("fa5s.caret-down", color=TEXT_MUTED, size=10))
        self.model_popup_button.setToolTip("Show model list")
        self.model_popup_button.setAccessibleName("Show model list")
        self.model_popup_button.setCursor(Qt.PointingHandCursor)

        self.model_loading_label = QLabel("")
        self.model_loading_label.setObjectName("ModelSettingsHintText")
        self.model_loading_label.setWordWrap(True)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("API key")
        self.api_key_edit.setAccessibleName("API key")

        self.api_key_reveal_button = QToolButton()
        self.api_key_reveal_button.setObjectName("ModelSettingsInlineToolButton")
        self.api_key_reveal_button.setIcon(_fa_icon("fa5s.eye", color=TEXT_MUTED, size=10))
        self.api_key_reveal_button.setToolTip("Show or hide API key")
        self.api_key_reveal_button.setAccessibleName("Toggle API key visibility")

        self.api_key_copy_button = QToolButton()
        self.api_key_copy_button.setObjectName("ModelSettingsInlineToolButton")
        self.api_key_copy_button.setIcon(_fa_icon("fa5s.copy", color=TEXT_MUTED, size=10))
        self.api_key_copy_button.setToolTip("Copy API key")
        self.api_key_copy_button.setAccessibleName("Copy API key")

        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("https://api.openai.com/v1")
        self.base_url_edit.setClearButtonEnabled(True)
        self.base_url_edit.setAccessibleName("Base URL")

        self.supports_images_checkbox = ImageSupportCheckBox("Image input support")
        self.supports_images_checkbox.setObjectName("ModelSupportsImagesCheckbox")
        self.supports_images_checkbox.setToolTip("Allow image attachments for this profile.")
        self.supports_images_checkbox.setAccessibleName("Image input support")

        api_key_row = QWidget()
        api_key_row.setObjectName("ModelSettingsFieldRow")
        api_key_layout = QHBoxLayout(api_key_row)
        api_key_layout.setContentsMargins(0, 0, 0, 0)
        api_key_layout.setSpacing(6)
        api_key_layout.addWidget(self.api_key_edit, 1)
        api_key_layout.addWidget(self.api_key_reveal_button, 0, Qt.AlignVCenter)
        api_key_layout.addWidget(self.api_key_copy_button, 0, Qt.AlignVCenter)

        model_row = QWidget()
        model_row.setObjectName("ModelSettingsFieldRow")
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        model_row_layout.setSpacing(6)
        model_row_layout.addWidget(self.model_combo, 1)
        model_row_layout.addWidget(self.model_text_edit, 1)
        model_row_layout.addWidget(self.model_popup_button, 0, Qt.AlignVCenter)
        model_row_layout.addWidget(self.model_reload_button, 0, Qt.AlignVCenter)

        model_field = QWidget()
        model_field_layout = QVBoxLayout(model_field)
        model_field_layout.setContentsMargins(0, 0, 0, 0)
        model_field_layout.setSpacing(4)
        model_field_layout.addWidget(model_row)
        model_field_layout.addWidget(self.model_loading_label)

        images_row = QWidget()
        images_layout = QVBoxLayout(images_row)
        images_layout.setContentsMargins(0, 2, 0, 2)
        images_layout.setSpacing(0)
        images_layout.addWidget(self.supports_images_checkbox)

        label_width = 68
        name_label = QLabel("&Name")
        provider_label = QLabel("&Provider")
        model_label = QLabel("&Model")
        api_key_label = QLabel("&API Key")
        base_url_label = QLabel("Base &URL")
        images_label = QLabel("I&mages")
        for label in (name_label, provider_label, model_label, api_key_label, base_url_label, images_label):
            label.setObjectName("ModelSettingsFieldLabel")
            label.setFixedWidth(label_width)

        name_label.setBuddy(self.name_edit)
        provider_label.setBuddy(self.provider_combo)
        model_label.setBuddy(self.model_text_edit)
        api_key_label.setBuddy(self.api_key_edit)
        base_url_label.setBuddy(self.base_url_edit)
        images_label.setBuddy(self.supports_images_checkbox)
        self.model_field_label = model_label

        form_layout.addRow(name_label, self.name_edit)
        form_layout.addRow(provider_label, self.provider_combo)
        form_layout.addRow(model_label, model_field)
        form_layout.addRow(api_key_label, api_key_row)
        editor_layout.addWidget(form_frame)

        advanced_content = QWidget()
        advanced_layout = QFormLayout(advanced_content)
        advanced_layout.setContentsMargins(6, 6, 6, 6)
        advanced_layout.setHorizontalSpacing(10)
        advanced_layout.setVerticalSpacing(8)
        advanced_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        advanced_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        advanced_layout.addRow(base_url_label, self.base_url_edit)
        advanced_layout.addRow(images_label, images_row)

        rotation_content = QWidget()
        rotation_content_layout = QVBoxLayout(rotation_content)
        rotation_content_layout.setContentsMargins(0, 0, 0, 0)
        rotation_content_layout.setSpacing(6)

        rotation_card = QFrame()
        rotation_card.setObjectName("ModelSettingsFormCard")
        rotation_card_layout = QVBoxLayout(rotation_card)
        rotation_card_layout.setContentsMargins(12, 10, 12, 10)
        rotation_card_layout.setSpacing(8)

        self.api_key_rotation_editor = CopySafePlainTextEdit()
        self.api_key_rotation_editor.setPlaceholderText("sk-key-1\nsk-key-2\ngm-key-3")
        self.api_key_rotation_editor.setObjectName("InlineCodeView")
        self.api_key_rotation_editor.setFont(_make_mono_font(10))
        self.api_key_rotation_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.api_key_rotation_editor.setMinimumHeight(120)
        rotation_card_layout.addWidget(self.api_key_rotation_editor, 1)

        self.api_key_rotation_status_label = QLabel("")
        self.api_key_rotation_status_label.setObjectName("ModelSettingsMeta")
        self.api_key_rotation_status_label.setWordWrap(True)
        rotation_card_layout.addWidget(self.api_key_rotation_status_label)

        rotation_content_layout.addWidget(rotation_card)
        self.api_key_rotation_section = CollapsibleSection(
            "API key pool",
            rotation_content,
            expanded=False,
            content_margins=(0, 0, 0, 0),
        )
        advanced_layout.addRow(self.api_key_rotation_section)
        self.advanced_section = CollapsibleSection(
            "Additional settings",
            advanced_content,
            expanded=True,
            content_margins=(0, 0, 0, 0),
        )
        editor_layout.addWidget(self.advanced_section)
        editor_layout.addStretch(1)

        editor_scroll.setWidget(editor_content)
        right.addWidget(editor_scroll, 1)

        left_container.setMinimumWidth(340)
        right_container.setMinimumWidth(440)
        left_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        right_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.body_splitter.addWidget(left_container)
        self.body_splitter.addWidget(right_container)
        self.body_splitter.setStretchFactor(0, 3)
        self.body_splitter.setStretchFactor(1, 4)
        self.body_splitter.setSizes([420, 620])
        models_page_layout.addWidget(self.body_splitter, 1)

        actions = QDialogButtonBox(QDialogButtonBox.Save)
        actions.setObjectName("ModelSettingsActions")
        self.save_button = actions.button(QDialogButtonBox.StandardButton.Save)
        if self.save_button is not None:
            self.save_button.setObjectName("PrimaryButton")
            self.save_button.setIcon(_fa_icon("fa5s.save", color="#FFFFFF", size=11))
            self.save_button.setMinimumHeight(24)
        root.addWidget(actions)

        self.add_button.clicked.connect(self._add_profile)
        self.delete_button.clicked.connect(self._delete_selected_profile)
        self.duplicate_button.clicked.connect(self._duplicate_selected_profile)
        self.search_edit.textChanged.connect(self._apply_profile_filter)
        if self.save_button is not None:
            self.save_button.clicked.connect(self._save_and_accept)

        self.name_edit.textEdited.connect(self._on_name_edited)
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.model_combo.popup_requested.connect(self._show_model_popup)
        self.model_text_edit.textChanged.connect(self._on_model_changed)
        self.api_key_edit.textChanged.connect(self._on_api_key_changed)
        self.api_key_reveal_button.clicked.connect(self._toggle_api_key_visibility)
        self.api_key_copy_button.clicked.connect(self._copy_api_key)
        self.api_key_rotation_editor.textChanged.connect(self._on_api_key_rotation_text_changed)
        self.base_url_edit.textChanged.connect(self._on_base_url_changed)
        self.model_popup_button.clicked.connect(self._show_model_popup)
        self.model_reload_button.clicked.connect(self._reload_models)
        self.supports_images_checkbox.checkStateChanged.connect(self._on_form_changed)

        self._set_model_state(ModelLoadState.IDLE)
        self._update_api_key_rotation_summary(api_keys=[])
        self._refresh_profile_list()
        if self.profile_list.count() > 0:
            self.profile_list.setCurrentRow(self._preferred_row_for_open())
        else:
            self._set_form_enabled(False)
        self._refresh_profile_counts()

    def result_payload(self) -> dict[str, Any]:
        return dict(self._result_payload)

    # --- Test tab: SESSION_SIZE slider ---

    SESSION_SIZE_MIN = 10_000
    SESSION_SIZE_MAX = 256_000
    # 2000 divides the 10k..256k span exactly, so every scale label (64k,
    # 128k, 192k, 256k) lands on a reachable slider position.
    SESSION_SIZE_STEP = 2_000

    def _build_test_page(self) -> None:
        layout = QVBoxLayout(self.test_page)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        title = QLabel("Session memory")
        title.setObjectName("ModelSettingsSectionTitle")
        layout.addWidget(title)

        description = QLabel(
            "SESSION_SIZE — estimated input context tokens before the agent summarizes "
            "older history. Higher values keep more live context; lower values compress sooner."
        )
        description.setObjectName("ModelSettingsMeta")
        description.setWordWrap(True)
        layout.addWidget(description)

        value_row = QHBoxLayout()
        value_row.setSpacing(8)
        self.session_size_value_label = QLabel()
        self.session_size_value_label.setObjectName("ModelSettingsMeta")
        value_row.addStretch(1)
        value_row.addWidget(self.session_size_value_label)
        layout.addLayout(value_row)

        self.session_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.session_size_slider.setObjectName("SessionSizeSlider")
        self.session_size_slider.setAccessibleName("Session size")
        self.session_size_slider.setAccessibleDescription(
            "Adjust SESSION_SIZE from 10k to 256k estimated context tokens"
        )
        self.session_size_slider.setRange(
            self.SESSION_SIZE_MIN // self.SESSION_SIZE_STEP,
            self.SESSION_SIZE_MAX // self.SESSION_SIZE_STEP,
        )
        self.session_size_slider.setSingleStep(1)
        self.session_size_slider.setPageStep(1)
        self.session_size_slider.valueChanged.connect(self._on_session_size_changed)
        layout.addWidget(self.session_size_slider)

        # Custom scale: QSlider's built-in ticks are drawn across the full
        # groove width while the handle travels a reduced span (groove minus
        # handle width), so built-in ticks and evenly-spread labels never line
        # up with the handle. This scale widget positions each label using the
        # same handle-position formula the style uses.
        self.session_size_scale = _SliderScaleWidget(
            self.session_size_slider,
            labels=("10k", "64k", "128k", "192k", "256k"),
            values=(10_000, 64_000, 128_000, 192_000, 256_000),
        )
        layout.addWidget(self.session_size_scale)

        layout.addStretch(1)

        self._session_size = self._initial_session_size()
        self._sync_session_size_slider()

    def _initial_session_size(self) -> int:
        """Resolve SESSION_SIZE: config.json first, then .env, then the model default."""
        raw = self._result_payload.get("session_size") if isinstance(self._result_payload, dict) else None
        if raw is not None:
            try:
                return self._clamp_session_size(int(float(raw)))
            except (TypeError, ValueError):
                pass
        from core.config import AgentConfig

        try:
            # AgentConfig resolves config.json first, then .env (see
            # settings_customise_sources), which is exactly the startup order.
            resolved = AgentConfig()
            return self._clamp_session_size(int(resolved.summary_threshold))
        except Exception:
            return self._clamp_session_size(int(AgentConfig.model_fields["summary_threshold"].default or 0))

    def _clamp_session_size(self, value: int) -> int:
        step = self.SESSION_SIZE_STEP
        clamped = max(self.SESSION_SIZE_MIN, min(self.SESSION_SIZE_MAX, value))
        return (clamped // step) * step

    def _sync_session_size_slider(self) -> None:
        self.session_size_slider.setValue(self._session_size // self.SESSION_SIZE_STEP)
        self._update_session_size_label()

    def _update_session_size_label(self) -> None:
        self.session_size_value_label.setText(f"{self._session_size // 1000}k tokens")

    def _on_session_size_changed(self, slider_value: int) -> None:
        self._session_size = slider_value * self.SESSION_SIZE_STEP
        self._update_session_size_label()

    def refresh_active_selection(self, payload: dict[str, Any]) -> None:
        """Select the currently active profile when the panel is (re)shown."""
        normalized = normalize_profiles_payload(payload or {})
        self._active_profile = str(normalized.get("active_profile") or "").strip()
        self._refresh_profile_counts()
        # Rebuild the list so the static "Active" badge follows the active profile
        # instead of staying pinned to the row that was active when first built.
        self._refresh_profile_list()

    def _target_screen(self):
        """Screen the panel should open on: the parent's screen, else primary."""
        screen = None
        parent = self.parentWidget()
        if parent is not None:
            window_handle = parent.windowHandle()
            if window_handle is not None:
                screen = window_handle.screen()
            if screen is None:
                screen = parent.screen()
        if screen is None:
            screen = self.screen()
        if screen is None:
            screen = QApplication.primaryScreen()
        return screen

    def _apply_screen_geometry(self) -> None:
        """Scale the window to the screen, clamp it and center it.

        The size is a fraction of the available screen area (WIDTH_RATIO /
        HEIGHT_RATIO), capped by PREFERRED_SIZE and floored by MINIMUM_SIZE.
        On low-resolution displays (e.g. 1366x768) the result is clamped to the
        available area minus a small margin, so the panel stays fully on-screen
        while still growing proportionally on larger displays.
        """
        screen = self._target_screen()
        if screen is None:
            self.resize(self.PREFERRED_SIZE)
            self.setMinimumSize(self.MINIMUM_SIZE)
            return
        available = screen.availableGeometry()
        max_width = max(320, available.width() - self.SCREEN_MARGIN * 2)
        max_height = max(240, available.height() - self.SCREEN_MARGIN * 2)
        width = int(available.width() * self.WIDTH_RATIO)
        height = int(available.height() * self.HEIGHT_RATIO)
        # Grow to the preferred size, but never below the minimum and never
        # past the available area (the clamp wins on very small displays).
        width = max(self.MINIMUM_SIZE.width(), min(width, self.PREFERRED_SIZE.width()))
        height = max(self.MINIMUM_SIZE.height(), min(height, self.PREFERRED_SIZE.height()))
        width = min(width, max_width)
        height = min(height, max_height)
        # The minimum must never exceed the clamped size, otherwise the window
        # would be forced past the screen edge on very small displays.
        self.setMinimumSize(min(self.MINIMUM_SIZE.width(), width), min(self.MINIMUM_SIZE.height(), height))
        self.resize(width, height)

    def _center_on_screen(self) -> None:
        screen = self._target_screen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = QRect(self.pos(), self.size())
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def _is_drag_zone(self, position: QPoint) -> bool:
        title_bar = getattr(self, "_title_bar", None)
        if title_bar is None:
            return False
        local = title_bar.mapFrom(self, position)
        return title_bar.rect().contains(local)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self._is_drag_zone(event.position().toPoint()):
            window_handle = self.windowHandle()
            if window_handle is not None and window_handle.startSystemMove():
                event.accept()
                return
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def showEvent(self, event) -> None:  # type: ignore[override]
        # Re-clamp to the current screen on every show: the panel may be moved
        # between displays or the resolution may change while it is hidden.
        self._apply_screen_geometry()
        super().showEvent(event)
        self._center_on_screen()
        self.visibility_changed.emit(True)

    def hideEvent(self, event) -> None:  # type: ignore[override]
        super().hideEvent(event)
        self.visibility_changed.emit(False)

    def closeEvent(self, event) -> None:
        self._fetch_debounce.stop()
        self._fetch_request_id += 1
        self._close_model_popup()
        for worker in list(self._model_workers):
            if worker.isRunning():
                worker.wait(3000)
            if worker in self._model_workers:
                self._model_workers.remove(worker)
            worker.deleteLater()
        super().closeEvent(event)

    def _current_row(self) -> int:
        return self.profile_list.currentRow()

    def _set_form_enabled(self, enabled: bool) -> None:
        self._form_enabled = bool(enabled)
        for widget in (
            self.name_edit,
            self.provider_combo,
            self.api_key_edit,
            self.api_key_rotation_editor,
            self.api_key_rotation_section.toggle_button,
            self.advanced_section.toggle_button,
            self.base_url_edit,
            self.supports_images_checkbox,
        ):
            widget.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)
        if enabled:
            self._update_base_url_field_state(self.provider_combo.currentText())
        self._set_model_state(self._model_state, message=self.model_loading_label.text())

    def _normalized_provider(self) -> str:
        provider = str(self.provider_combo.currentText() or "").strip().lower()
        return provider if provider in ALLOWED_PROVIDERS else "openai"

    def _invalidate_pending_fetches(self) -> None:
        self._fetch_debounce.stop()
        self._fetch_request_id += 1

    def _clear_model_options(self) -> None:
        self._model_entries_by_id = {}
        self._close_model_popup()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.blockSignals(False)

    def _set_combo_placeholder(self, text: str) -> None:
        placeholder = str(text or "").strip()
        self._close_model_popup()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if placeholder:
            self.model_combo.addItem(placeholder)
            self.model_combo.setCurrentIndex(0)
        self.model_combo.blockSignals(False)

    def _filter_model_popup_items(self, text: str = "") -> None:
        if self._model_popup_list is None:
            return
        query = str(text or "").strip().casefold()
        for row in range(self._model_popup_list.count()):
            item = self._model_popup_list.item(row)
            item_text = str(item.text() or "")
            selectable = bool(item.flags() & Qt.ItemFlag.ItemIsSelectable)
            item.setHidden(bool(query) and (not selectable or query not in item_text.casefold()))

    def _close_model_popup(self) -> None:
        if self._model_popup is None:
            return
        popup = self._model_popup
        self._model_popup = None
        self._model_popup_search = None
        self._model_popup_list = None
        popup.close()
        popup.deleteLater()

    def _set_model_state(self, state: ModelLoadState, *, message: str = "") -> None:
        self._model_state = state
        provider = self._normalized_provider()
        is_openai = provider == "openai"
        show_combo = state in {ModelLoadState.LOADING, ModelLoadState.LOADED, ModelLoadState.ERROR}
        show_text = not show_combo
        status_text = str(message or "").strip()
        has_fetch_inputs = self._current_fetch_inputs() is not None

        self.model_combo.setVisible(show_combo)
        self.model_text_edit.setVisible(show_text)
        self.model_combo.setEditable(is_openai and state in {ModelLoadState.LOADED, ModelLoadState.FALLBACK})
        self.model_combo.setEnabled(self._form_enabled and state in {ModelLoadState.LOADED, ModelLoadState.FALLBACK})
        self.model_text_edit.setEnabled(self._form_enabled and state == ModelLoadState.FALLBACK)
        if not (self._form_enabled and state == ModelLoadState.LOADED and self.model_combo.count() > 0):
            self._close_model_popup()
        self.model_popup_button.setVisible(self._form_enabled and show_combo)
        self.model_popup_button.setEnabled(self._form_enabled and state == ModelLoadState.LOADED and self.model_combo.count() > 0)
        self.model_reload_button.setVisible(self._form_enabled)
        self.model_reload_button.setEnabled(self._form_enabled and has_fetch_inputs and state != ModelLoadState.LOADING)
        self.model_loading_label.setText(status_text)
        self.model_loading_label.setVisible(bool(status_text))
        if hasattr(self, "model_field_label"):
            self.model_field_label.setBuddy(self.model_combo if show_combo else self.model_text_edit)

    def _set_current_model_widgets_text(self, value: str) -> None:
        text = str(value or "").strip()
        self.model_text_edit.blockSignals(True)
        self.model_text_edit.setText(text)
        self.model_text_edit.blockSignals(False)

        self.model_combo.blockSignals(True)
        if self.model_combo.count() == 0:
            self.model_combo.setCurrentIndex(-1)
        if self.model_combo.isEditable():
            self.model_combo.setEditText(text)
        else:
            index = self.model_combo.findText(text)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)

    def _show_model_popup(self) -> None:
        if not self.model_combo.isVisible() or not self.model_combo.isEnabled() or self.model_combo.count() <= 0:
            return
        self._close_model_popup()

        popup = QFrame(self, Qt.Popup | Qt.FramelessWindowHint)
        popup.setObjectName("ModelSettingsModelPopup")
        popup_layout = QVBoxLayout(popup)
        popup_layout.setContentsMargins(6, 6, 6, 6)
        popup_layout.setSpacing(6)

        search = QLineEdit(popup)
        search.setObjectName("ModelSettingsSearchField")
        search.setPlaceholderText("Quick model search")
        search.setClearButtonEnabled(True)
        search.setAccessibleName("Model list search")
        search.setAccessibleDescription("Filter available models returned by the current API URL")
        search.addAction(_fa_icon("fa5s.search", color=TEXT_MUTED, size=10), QLineEdit.LeadingPosition)
        popup_layout.addWidget(search)

        model_list = QListWidget(popup)
        model_list.setObjectName("ModelSettingsModelPopupList")
        model_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        model_list.setSelectionMode(QListWidget.SingleSelection)
        popup_layout.addWidget(model_list, 1)

        combo_model = self.model_combo.model()
        root_index = self.model_combo.rootModelIndex()
        model_column = self.model_combo.modelColumn()
        for row in range(self.model_combo.count()):
            item_text = str(self.model_combo.itemText(row) or "")
            combo_index = combo_model.index(row, model_column, root_index)
            flags = combo_model.flags(combo_index)
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, row)
            if not (flags & Qt.ItemFlag.ItemIsEnabled and flags & Qt.ItemFlag.ItemIsSelectable):
                item.setFlags(Qt.ItemFlag.NoItemFlags)
            model_list.addItem(item)
            if row == self.model_combo.currentIndex():
                model_list.setCurrentItem(item)

        def choose_model(item: QListWidgetItem) -> None:
            row = item.data(Qt.UserRole)
            if row is None or not (item.flags() & Qt.ItemFlag.ItemIsSelectable):
                return
            self.model_combo.setCurrentIndex(int(row))
            self._close_model_popup()

        search.textChanged.connect(self._filter_model_popup_items)
        model_list.itemActivated.connect(choose_model)
        model_list.itemClicked.connect(choose_model)

        self._model_popup = popup
        self._model_popup_search = search
        self._model_popup_list = model_list

        popup_width = max(self.model_combo.width(), 360)
        popup_height = min(max(240, self.model_combo.count() * 28 + 52), 460)
        popup.resize(popup_width, popup_height)
        popup.move(self.model_combo.mapToGlobal(QPoint(0, self.model_combo.height() + 2)))
        popup.show()
        search.setFocus(Qt.PopupFocusReason)
        self._filter_model_popup_items("")

    def _get_current_model_value(self) -> str:
        if self._model_state == ModelLoadState.LOADED:
            return str(self.model_combo.currentText() or "").strip()
        return str(self.model_text_edit.text() or "").strip()

    def _selected_profile_enabled(self) -> bool:
        row = self._current_row()
        if row < 0 or row >= len(self._profiles):
            return False
        return bool(self._profiles[row].get("enabled", True))

    def _current_fetch_inputs(self) -> tuple[str, str, str, ModelFetcher, tuple[str, ...]] | None:
        if not self._selected_profile_enabled():
            return None
        provider = self._normalized_provider()
        api_key = str(self.api_key_edit.text() or "").strip()
        if not api_key:
            return None
        if provider == "gemini":
            return provider, api_key, "", GeminiModelFetcher(), (provider, api_key)
        base_url = str(self.base_url_edit.text() or "").strip().rstrip("/")
        if provider == "anthropic":
            return provider, api_key, base_url, AnthropicModelFetcher(), (provider, api_key, base_url)
        if not base_url:
            return None
        return provider, api_key, base_url, OpenAICompatibleModelFetcher(), (provider, api_key, base_url)

    def _schedule_fetch(self, delay_ms: int = 600) -> None:
        if self._current_row() < 0:
            return
        if self._current_fetch_inputs() is None:
            self._set_model_state(ModelLoadState.IDLE)
            return
        self._fetch_debounce.setInterval(max(0, int(delay_ms)))
        self._fetch_debounce.start()

    def _cleanup_model_worker(self, worker: ModelFetchWorker) -> None:
        if worker in self._model_workers:
            self._model_workers.remove(worker)
        worker.deleteLater()

    def _sync_api_key_field_from_rotation_editor(self) -> tuple[list[str], int, str]:
        row = self._current_row()
        api_keys = self._current_rotation_api_keys()
        current_key = str(self.api_key_edit.text() or "").strip()
        preferred_index = self._profile_api_key_index(row)
        api_key_index, active_api_key = self._resolve_api_key_selection(
            api_keys,
            current_key=current_key,
            preferred_index=preferred_index,
        )
        self._update_api_key_rotation_summary(api_keys=api_keys, active_key=active_api_key)
        if self.api_key_edit.text() != active_api_key:
            self._loading_form = True
            self.api_key_edit.setText(active_api_key)
            self._loading_form = False
        return api_keys, api_key_index, active_api_key

    def _sync_rotation_editor_from_active_key(self) -> list[str]:
        row = self._current_row()
        api_keys = self._current_rotation_api_keys()
        current_key = str(self.api_key_edit.text() or "").strip()
        preferred_index = self._profile_api_key_index(row)
        if api_keys:
            preferred_index = max(0, min(preferred_index, len(api_keys) - 1))
        if current_key:
            if api_keys:
                api_keys[preferred_index] = current_key
            else:
                api_keys = [current_key]
        elif api_keys:
            api_keys.pop(preferred_index)
        normalized_keys = normalize_api_key_list(api_keys)
        self._set_api_key_rotation_editor_text(normalized_keys)
        return normalized_keys

    def _append_gemini_model_items(self, entries: list[ModelEntry]) -> list[str]:
        ordered_ids: list[str] = []
        gemini_ids = sorted((entry.id for entry in entries if entry.family == "gemini"), reverse=True)
        gemma_ids = sorted((entry.id for entry in entries if entry.family == "gemma"), reverse=True)
        for model_id in gemini_ids:
            self.model_combo.addItem(model_id)
            ordered_ids.append(model_id)
        if gemma_ids and gemini_ids:
            separator = QStandardItem("── Gemma ──")
            separator.setFlags(Qt.ItemFlag.NoItemFlags)
            self.model_combo.model().appendRow(separator)
        for model_id in gemma_ids:
            self.model_combo.addItem(model_id)
            ordered_ids.append(model_id)
        return ordered_ids

    def _apply_entry_image_support(self, model_id: str) -> None:
        entry = self._model_entries_by_id.get(str(model_id or "").strip())
        if entry is None:
            return
        self.supports_images_checkbox.setChecked(bool(entry.supports_image_input))

    def _sync_image_support_after_model_load(self, model_id: str) -> None:
        current_row = self._current_row()
        if 0 <= current_row < len(self._profiles):
            profile = self._profiles[current_row]
            profile_model = str(profile.get("model") or "").strip()
            if profile_model and profile_model == str(model_id or "").strip():
                self.supports_images_checkbox.setChecked(bool(profile.get("supports_image_input")))
                return
        self._apply_entry_image_support(model_id)

    def _apply_loaded_models(self, entries: list[ModelEntry]) -> None:
        provider = self._normalized_provider()
        current_value = self._get_current_model_value()
        current_row = self._current_row()
        manual_entry = 0 <= current_row < len(self._model_manual_flags) and self._model_manual_flags[current_row]
        new_entries = {entry.id: entry for entry in entries}
        if current_value and current_value not in new_entries and not manual_entry:
            current_value = ""
        self._model_entries_by_id = new_entries

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if provider == "gemini":
            ordered_ids = self._append_gemini_model_items(entries)
        else:
            ordered_ids = sorted(entry.id for entry in entries)
            for model_id in ordered_ids:
                self.model_combo.addItem(model_id)

        selected_model = current_value if current_value in self._model_entries_by_id else (ordered_ids[0] if ordered_ids else "")
        manual_model = bool(current_value) and selected_model != current_value
        if manual_model:
            self.model_combo.insertItem(0, current_value)
            selected_model = current_value
        if selected_model:
            selected_index = self.model_combo.findText(selected_model)
            if selected_index >= 0:
                self.model_combo.setCurrentIndex(selected_index)
            elif self.model_combo.isEditable():
                self.model_combo.setEditText(selected_model)
        self.model_combo.blockSignals(False)

        self._set_current_model_widgets_text(selected_model)
        if manual_model and 0 <= current_row < len(self._model_manual_flags):
            self._model_manual_flags[current_row] = True
        if selected_model and 0 <= current_row < len(self._profiles) and not self._name_manual_flags[current_row]:
            self._loading_form = True
            self.name_edit.setText(self._suggest_unique_id(selected_model, row=current_row))
            self._loading_form = False
        if not selected_model and provider == "openai":
            self._set_model_state(
                ModelLoadState.LOADED,
                message="The model list is empty. Enter a model name manually.",
            )
        else:
            self._set_model_state(ModelLoadState.LOADED)
        self._sync_image_support_after_model_load(selected_model)
        self._on_form_changed()

    def _start_fetch(self) -> None:
        self._fetch_debounce.setInterval(600)
        request = self._current_fetch_inputs()
        if request is None:
            self._set_model_state(ModelLoadState.IDLE)
            return

        provider, api_key, base_url, fetcher, cache_key = request
        cached_entries = self._model_cache.get(cache_key)
        if cached_entries is not None:
            self._apply_loaded_models(cached_entries)
            return

        self._fetch_request_id += 1
        request_id = self._fetch_request_id
        self._set_combo_placeholder("Loading models...")
        self._set_model_state(ModelLoadState.LOADING, message="Loading...")

        worker = ModelFetchWorker(request_id, fetcher, api_key, base_url)
        self._model_workers.append(worker)
        worker.fetched.connect(self._on_models_fetched)
        worker.failed.connect(self._on_models_failed)
        worker.finished.connect(lambda: self._cleanup_model_worker(worker))
        worker.start()

    def _on_models_fetched(self, request_id: int, payload: object) -> None:
        if request_id != self._fetch_request_id:
            return
        entries = [entry for entry in list(payload or []) if isinstance(entry, ModelEntry)]
        request = self._current_fetch_inputs()
        if request is None:
            return
        cache_key = request[-1]
        self._model_cache[cache_key] = entries
        self._apply_loaded_models(entries)

    def _on_models_failed(self, request_id: int, message: str) -> None:
        if request_id != self._fetch_request_id:
            return
        current_value = self._get_current_model_value()
        self._set_current_model_widgets_text(current_value)
        if self._normalized_provider() in {"openai", "anthropic"}:
            self._set_model_state(ModelLoadState.FALLBACK, message=message)
            return
        self._set_combo_placeholder(current_value or "Models unavailable")
        self._set_model_state(ModelLoadState.ERROR, message=message)

    def _reload_models(self) -> None:
        if self._loading_form:
            return
        request = self._current_fetch_inputs()
        if request is not None:
            self._model_cache.pop(request[-1], None)
        self._invalidate_pending_fetches()
        if request is None:
            self._set_model_state(ModelLoadState.IDLE)
            return
        self._schedule_fetch(0)

    def _on_api_key_changed(self, _text: str) -> None:
        if self._loading_form:
            return
        self._sync_rotation_editor_from_active_key()
        self._invalidate_pending_fetches()
        self._clear_model_options()
        self._set_model_state(ModelLoadState.IDLE)
        if self._current_fetch_inputs() is not None:
            self._schedule_fetch(600)
        self._on_form_changed()

    def _on_api_key_rotation_text_changed(self) -> None:
        if self._loading_form:
            return
        self._sync_api_key_field_from_rotation_editor()
        self._invalidate_pending_fetches()
        self._clear_model_options()
        self._set_model_state(ModelLoadState.IDLE)
        if self._current_fetch_inputs() is not None:
            self._schedule_fetch(600)
        self._on_form_changed()

    def _on_base_url_changed(self, _text: str) -> None:
        if self._loading_form:
            return
        row = self._current_row()
        if 0 <= row < len(self._profiles) and not self._name_manual_flags[row]:
            current_model = self._get_current_model_value()
            self._loading_form = True
            self.name_edit.setText(self._suggest_unique_id(current_model, row=row))
            self._loading_form = False
        if self._normalized_provider() in {"openai", "anthropic"}:
            self._invalidate_pending_fetches()
            self._clear_model_options()
            self._set_model_state(ModelLoadState.IDLE)
            if self._current_fetch_inputs() is not None:
                self._schedule_fetch(600)
        self._on_form_changed()

    def _profile_api_key_index(self, row: int) -> int:
        if row < 0 or row >= len(self._profiles):
            return 0
        api_keys = self._profile_api_keys(row)
        if not api_keys:
            return 0
        try:
            index = int(self._profiles[row].get("api_key_index") or 0)
        except (TypeError, ValueError):
            index = 0
        return max(0, min(index, len(api_keys) - 1))

    def _current_rotation_api_keys(self) -> list[str]:
        return normalize_api_key_list(self.api_key_rotation_editor.toPlainText())

    def _set_api_key_rotation_editor_text(self, api_keys: list[str]) -> None:
        normalized_keys = normalize_api_key_list(api_keys)
        text = "\n".join(normalized_keys)
        if self.api_key_rotation_editor.toPlainText() == text:
            self._update_api_key_rotation_summary(api_keys=normalized_keys)
            return
        self.api_key_rotation_editor.blockSignals(True)
        self.api_key_rotation_editor.setPlainText(text)
        self.api_key_rotation_editor.blockSignals(False)
        self._update_api_key_rotation_summary(api_keys=normalized_keys)

    def _update_api_key_rotation_summary(self, *, api_keys: list[str] | None = None, active_key: str | None = None) -> None:
        keys = normalize_api_key_list(api_keys if api_keys is not None else self.api_key_rotation_editor.toPlainText())
        active = str(self.api_key_edit.text() if active_key is None else active_key or "").strip()
        if not keys:
            self.api_key_rotation_status_label.setText("Uses the API key above.")
            return
        if active and active in keys:
            self.api_key_rotation_status_label.setText(f"Using key {keys.index(active) + 1} of {len(keys)}.")
            return
        self.api_key_rotation_status_label.setText("The first valid key will be used.")

    def _set_save_state(self, message: str) -> None:
        text = str(message or "").strip()
        self.save_state_label.setText(text)
        self.save_state_label.setVisible(bool(text))

    def _update_selected_profile_context(self, row: int) -> None:
        if row < 0 or row >= len(self._profiles):
            self.selected_profile_title.setText("No profile selected")
            hint = (
                "Add a profile to start configuring models."
                if not self._profiles
                else "Select a profile to edit its model settings."
            )
            self.form_hint.setText(hint)
            return
        profile = self._profiles[row]
        profile_id = str(profile.get("id") or "").strip() or "(unnamed)"
        status = "enabled" if bool(profile.get("enabled", True)) else "disabled"
        self.selected_profile_title.setText(profile_id)
        self.form_hint.setText(f"Editing profile: {profile_id} ({status})")

    def _profile_id_for_row(self, row: int) -> str:
        if row < 0 or row >= len(self._profiles):
            return ""
        return str(self._profiles[row].get("id") or "").strip()

    def _row_for_profile_id(self, profile_id: str) -> int:
        target_id = str(profile_id or "").strip()
        if not target_id:
            return -1
        for idx, profile in enumerate(self._profiles):
            if str(profile.get("id") or "").strip() == target_id:
                return idx
        return -1

    def _refresh_profile_counts(self) -> None:
        active_name = str(self._active_profile or "").strip() or "none"
        self.active_profile_label.setText(
            f"Active: <span style='color:#D5D9DF;font-weight:700'>{active_name}</span>"
        )

    def _apply_profile_filter(self, value: str) -> None:
        self._filter_text = str(value or "").strip().lower()
        visible_rows: list[int] = []
        for row in range(self.profile_list.count()):
            item = self.profile_list.item(row)
            if item is None:
                continue
            haystack = " ".join((str(item.data(Qt.UserRole) or ""), str(item.toolTip() or ""))).lower()
            should_hide = bool(self._filter_text) and self._filter_text not in haystack
            item.setHidden(should_hide)
            if not should_hide:
                visible_rows.append(row)

        current_row = self.profile_list.currentRow()
        if current_row < 0:
            if visible_rows:
                self.profile_list.setCurrentRow(visible_rows[0])
            return

        current_item = self.profile_list.item(current_row)
        if current_item is not None and current_item.isHidden():
            if visible_rows:
                self.profile_list.setCurrentRow(visible_rows[0])
            else:
                self.profile_list.clearSelection()
                self._sync_form_to_profile(-1)

    def _update_base_url_field_state(self, provider: str) -> None:
        provider_normalized = str(provider or "").strip().lower()
        enabled = provider_normalized in {"openai", "anthropic"}
        self.base_url_edit.setEnabled(enabled)
        if enabled:
            if provider_normalized == "anthropic":
                self.base_url_edit.setPlaceholderText("https://api.anthropic.com (optional, no /v1)")
                self.base_url_edit.setToolTip("Optional Anthropic-compatible Base URL. Do not include /v1; the SDK adds it automatically.")
            else:
                self.base_url_edit.setPlaceholderText("https://api.openai.com/v1")
                self.base_url_edit.setToolTip("")
        else:
            self.base_url_edit.setPlaceholderText(f"Not used for {provider_normalized or 'this provider'}")
            self.base_url_edit.setToolTip(f"Base URL is only used for openai or anthropic profiles.")

    def _display_name(self, profile: dict[str, str]) -> str:
        profile_id = str(profile.get("id") or "").strip()
        provider = str(profile.get("provider") or "").strip()
        model_name = str(profile.get("model") or "").strip()
        marker = ""
        if profile_id and profile_id == self._active_profile:
            marker = " • active"
        elif not bool(profile.get("enabled", True)):
            marker = " • disabled"
        title = profile_id if profile_id else "(unnamed)"
        if marker:
            title = f"{title}{marker}"
        details = " · ".join(part for part in (provider, model_name) if part)
        return f"{title}\n{details}" if details else title

    def _refresh_profile_item_states(self) -> None:
        current_row = self._current_row()
        for row in range(self.profile_list.count()):
            item = self.profile_list.item(row)
            widget = self.profile_list.itemWidget(item) if item is not None else None
            if widget is None:
                continue
            profile = self._profiles[row] if 0 <= row < len(self._profiles) else {}
            is_selected = row == current_row and not bool(item.isHidden()) if item is not None else False
            is_active = bool(str(profile.get("id") or "").strip() and str(profile.get("id") or "").strip() == self._active_profile)
            is_enabled = bool(profile.get("enabled", True))
            widget.setProperty("selectedProfile", is_selected)
            widget.setProperty("activeProfile", is_active)
            widget.setProperty("disabledProfile", not is_enabled)
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            rail = widget.findChild(QFrame, "ModelProfileTabRail")
            if rail is not None:
                rail.setProperty("selectedProfile", is_selected)
                rail.style().unpolish(rail)
                rail.style().polish(rail)

    def _toggle_api_key_visibility(self) -> None:
        reveal = self.api_key_edit.echoMode() != QLineEdit.Normal
        self.api_key_edit.setEchoMode(QLineEdit.Normal if reveal else QLineEdit.Password)
        icon_name = "fa5s.eye-slash" if reveal else "fa5s.eye"
        self.api_key_reveal_button.setIcon(_fa_icon(icon_name, color=TEXT_MUTED, size=12))

    def _copy_api_key(self) -> None:
        QApplication.clipboard().setText(self.api_key_edit.text())
        self._set_save_state("API key copied to clipboard.")

    def _profile_api_keys(self, row: int) -> list[str]:
        if row < 0 or row >= len(self._profiles):
            return []
        profile = self._profiles[row]
        return normalize_api_key_list(profile.get("api_keys"), fallback=profile.get("api_key"))

    def _resolve_api_key_selection(
        self,
        api_keys: list[str],
        *,
        current_key: str,
        preferred_index: int,
    ) -> tuple[int, str]:
        if not api_keys:
            return 0, ""
        clamped_index = max(0, min(int(preferred_index), len(api_keys) - 1))
        if current_key and current_key in api_keys:
            selected_index = api_keys.index(current_key)
        else:
            selected_index = clamped_index
        return selected_index, api_keys[selected_index]

    def _apply_api_key_rotation_to_profile(self, row: int, api_keys: list[str]) -> None:
        if row < 0 or row >= len(self._profiles):
            return
        cleaned_keys = normalize_api_key_list(api_keys)
        profile = dict(self._profiles[row])
        current_key = str(profile.get("api_key") or "").strip()
        api_key_index, active_api_key = self._resolve_api_key_selection(
            cleaned_keys,
            current_key=current_key,
            preferred_index=0,
        )
        profile["api_keys"] = cleaned_keys
        profile["api_key_index"] = api_key_index
        profile["invalid_api_keys"] = []
        profile["key_error_timestamps"] = {}
        profile["api_key"] = active_api_key
        self._profiles[row] = profile
        if row == self._current_row():
            self._loading_form = True
            self.api_key_edit.setText(active_api_key)
            self._set_api_key_rotation_editor_text(cleaned_keys)
            self._loading_form = False
            self._update_api_key_rotation_summary(api_keys=cleaned_keys, active_key=active_api_key)

    def _edit_api_key_rotation(self) -> None:
        """Reveal and focus the inline API-key rotation editor."""
        row = self._current_row()
        if row < 0 or row >= len(self._profiles):
            return
        self.advanced_section.set_expanded(True)
        self.api_key_rotation_section.set_expanded(True)
        self.api_key_rotation_editor.setFocus(Qt.OtherFocusReason)
        self.api_key_rotation_editor.ensureCursorVisible()
        self._set_save_state("Edit the rotation pool inline and press Save to persist.")

    def _build_profile_item_widget(self, profile: dict[str, Any], row: int) -> QWidget:
        container = QWidget()
        container.setObjectName("ModelProfileRowCard")
        container.setAttribute(Qt.WA_StyledBackground, True)
        is_enabled = bool(profile.get("enabled", True))
        container.setProperty("disabledProfile", not is_enabled)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(12, 10, 8, 10)
        layout.setSpacing(10)

        text_column = QVBoxLayout()
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.setSpacing(2)

        first_row = QHBoxLayout()
        first_row.setContentsMargins(0, 0, 0, 0)
        first_row.setSpacing(4)

        profile_id = str(profile.get("id") or "").strip() or "(unnamed)"
        is_active = bool(profile_id and profile_id != "(unnamed)" and profile_id == self._active_profile)
        # ElidedLabel keeps long profile names on one line and shows an ellipsis
        # instead of being clipped by the provider/Active badges next to it.
        # The full name stays available via the tooltip.
        title_label = ElidedLabel(elide_mode=Qt.ElideRight)
        title_label.setObjectName("ModelProfileItemTitle")
        title_label.setEnabled(is_enabled)
        title_label.setMargin(0)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_label.setMinimumWidth(0)
        title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        title_label.set_full_text(profile_id)
        # No alignment flags: they would stop the layout from stretching the
        # label, leaving no room to elide against the badges.
        first_row.addWidget(title_label, 1)

        provider = str(profile.get("provider") or "").strip()
        if provider:
            provider_label = QLabel(provider)
            provider_label.setObjectName("ModelProfileItemBadge")
            provider_label.setProperty("badgeVariant", "provider")
            first_row.addWidget(provider_label, 0, Qt.AlignLeft | Qt.AlignVCenter)

        if is_active:
            active_label = QLabel("Active")
            active_label.setObjectName("ModelProfileItemBadge")
            active_label.setProperty("badgeVariant", "active")
            first_row.addWidget(active_label, 0, Qt.AlignLeft | Qt.AlignVCenter)
        elif not is_enabled:
            disabled_label = QLabel("Disabled")
            disabled_label.setObjectName("ModelProfileItemBadge")
            disabled_label.setProperty("badgeVariant", "muted")
            first_row.addWidget(disabled_label, 0, Qt.AlignLeft | Qt.AlignVCenter)

        first_row.addStretch(1)
        text_column.addLayout(first_row)

        model_name = str(profile.get("model") or "").strip()
        details_label = QLabel(model_name)
        details_label.setObjectName("ModelProfileItemMeta")
        details_label.setEnabled(is_enabled)
        details_label.setMargin(0)
        details_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        # Long model names must wrap instead of being clipped by the narrow
        # Profiles column. The card height grows to fit the wrapped text.
        details_label.setWordWrap(True)
        details_label.setMinimumWidth(0)
        details_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # No alignment flags in addWidget(): an alignment constraint stops the
        # layout from stretching the label to the full column width, and a
        # word-wrapped QLabel then shrinks to its (underestimated) sizeHint and
        # wraps names that would easily fit on one line.
        text_column.addWidget(details_label)

        layout.addLayout(text_column, 1)

        enabled_switch = QCheckBox()
        enabled_switch.setObjectName("ModelProfileEnabledSwitch")
        enabled_switch.setChecked(is_enabled)
        enabled_switch.setCursor(Qt.PointingHandCursor)
        enabled_switch.setToolTip("Temporarily enable or disable this model")
        enabled_switch.setFocusPolicy(Qt.NoFocus)
        enabled_switch.setFixedSize(QSize(34, 20))
        enabled_switch.pressed.connect(lambda target_row=row: self.profile_list.setCurrentRow(target_row))
        enabled_switch.toggled.connect(lambda checked, target_row=row: self._toggle_profile_enabled(target_row, checked))
        layout.addWidget(enabled_switch, 0, Qt.AlignRight | Qt.AlignVCenter)

        tab_rail = QFrame()
        tab_rail.setObjectName("ModelProfileTabRail")
        tab_rail.setFixedWidth(3)
        tab_rail.setProperty("selectedProfile", False)
        layout.addWidget(tab_rail)
        container.ensurePolished()
        title_label.ensurePolished()
        details_label.ensurePolished()
        title_label.setMinimumHeight(title_label.fontMetrics().height() + 6)
        # Do NOT force a fixed minimum height on the model-name label: with
        # wordWrap enabled a long name must be able to grow the card height.
        container.adjustSize()
        return container

    def _refresh_profile_list(
        self,
        preferred_row: int | None = None,
        *,
        preferred_profile_id: str = "",
        restore_scroll_value: int | None = None,
    ) -> None:
        self.profile_list.blockSignals(True)
        self.profile_list.clear()
        for row, profile in enumerate(self._profiles):
            item_widget = self._build_profile_item_widget(profile, row)
            item = QListWidgetItem("")
            item.setData(Qt.UserRole, self._display_name(profile))
            item_widget.ensurePolished()
            widget_hint = item_widget.sizeHint()
            minimum_hint = item_widget.minimumSizeHint()
            item_height = max(72, widget_hint.height(), minimum_hint.height()) + 6
            item.setSizeHint(QSize(widget_hint.width(), item_height))
            provider = str(profile.get("provider") or "").strip()
            model_name = str(profile.get("model") or "").strip()
            enabled = "yes" if bool(profile.get("enabled", True)) else "no"
            item.setToolTip(f"Provider: {provider}\nModel: {model_name}\nEnabled: {enabled}".strip())
            self.profile_list.addItem(item)
            self.profile_list.setItemWidget(item, item_widget)
        self.profile_list.blockSignals(False)
        # _fit_items_to_viewport re-derives every card height at the real
        # viewport width, honoring wrapped model names (heightForWidth).
        self.profile_list._fit_items_to_viewport()
        if self.save_button is not None:
            self.save_button.setEnabled(bool(self._profiles))

        if not self._profiles:
            self._selected_row = -1
            self._set_form_enabled(False)
            self.duplicate_button.setEnabled(False)
            self._update_selected_profile_context(-1)
            return

        row = preferred_row
        if preferred_profile_id:
            resolved_row = self._row_for_profile_id(preferred_profile_id)
            if resolved_row >= 0:
                row = resolved_row
        if row is None:
            row = self._preferred_row_for_open()
        row = max(0, min(row, len(self._profiles) - 1))
        self.profile_list.setCurrentRow(row)
        self._apply_profile_filter(self.search_edit.text())
        self._refresh_profile_item_states()
        if restore_scroll_value is not None:
            self.profile_list.verticalScrollBar().setValue(restore_scroll_value)

    def _preferred_row_for_open(self) -> int:
        active_id = str(self._active_profile or "").strip()
        if active_id:
            for index, profile in enumerate(self._profiles):
                if str(profile.get("id") or "").strip() == active_id:
                    return index
        return self._current_row() if self._current_row() >= 0 else 0

    def _sync_form_to_profile(self, row: int) -> None:
        if row < 0 or row >= len(self._profiles):
            self._invalidate_pending_fetches()
            self._clear_model_options()
            self._set_current_model_widgets_text("")
            self._selected_row = -1
            self._set_form_enabled(False)
            self._set_model_state(ModelLoadState.IDLE)
            self.duplicate_button.setEnabled(False)
            self._update_selected_profile_context(-1)
            self._refresh_profile_item_states()
            return
        profile = self._profiles[row]
        self._loading_form = True
        self._invalidate_pending_fetches()
        self._clear_model_options()
        self._set_form_enabled(True)
        self.name_edit.setText(str(profile.get("id", "")))
        provider = str(profile.get("provider", "openai")).strip().lower()
        if provider not in ALLOWED_PROVIDERS:
            provider = "openai"
        self.provider_combo.setCurrentText(provider)
        self._set_current_model_widgets_text(str(profile.get("model", "")))
        self.api_key_edit.setText(str(profile.get("api_key", "")))
        self._set_api_key_rotation_editor_text(self._profile_api_keys(row))
        self.base_url_edit.setText(str(profile.get("base_url", "")))
        self.supports_images_checkbox.setChecked(bool(profile.get("supports_image_input")))
        self._update_base_url_field_state(provider)
        self._update_api_key_rotation_summary(
            api_keys=self._profile_api_keys(row),
            active_key=str(profile.get("api_key", "")),
        )
        self._set_model_state(ModelLoadState.IDLE)
        self._loading_form = False
        self._selected_row = row
        self.duplicate_button.setEnabled(True)
        self._update_selected_profile_context(row)
        self._refresh_profile_counts()
        self._refresh_profile_item_states()
        if self._current_fetch_inputs() is not None:
            self._schedule_fetch(100)

    def _sync_current_profile_from_form(self, row: int | None = None) -> None:
        if self._loading_form:
            return
        target_row = self._current_row() if row is None else row
        if target_row < 0 or target_row >= len(self._profiles):
            return
        provider = self._normalized_provider()
        base_url = str(self.base_url_edit.text() or "").strip() if provider in {"openai", "anthropic"} else ""
        existing_profile = dict(self._profiles[target_row])
        current_api_key = str(self.api_key_edit.text() or "").strip()
        api_keys = self._current_rotation_api_keys()
        existing_index = self._profile_api_key_index(target_row)
        if api_keys:
            existing_index = max(0, min(existing_index, len(api_keys) - 1))
        if current_api_key:
            if api_keys:
                api_keys[existing_index] = current_api_key
            else:
                api_keys = [current_api_key]
        elif api_keys:
            api_keys.pop(existing_index)
            existing_index = min(existing_index, max(0, len(api_keys) - 1))
        api_keys = normalize_api_key_list(api_keys)
        api_key_index, current_api_key = self._resolve_api_key_selection(
            api_keys,
            current_key=current_api_key,
            preferred_index=existing_index,
        )
        updated_profile = {
            "id": str(self.name_edit.text() or "").strip(),
            "provider": provider,
            "model": self._get_current_model_value(),
            "api_key": current_api_key,
            "api_keys": api_keys,
            "api_key_index": api_key_index,
            "invalid_api_keys": [],
            "key_error_timestamps": {},
            "base_url": base_url,
            "supports_image_input": self.supports_images_checkbox.isChecked(),
            "enabled": bool(existing_profile.get("enabled", True)),
        }
        if isinstance(existing_profile.get("reasoning"), dict):
            updated_profile["reasoning"] = dict(existing_profile["reasoning"])
        self._profiles[target_row] = updated_profile
        if target_row == self._selected_row:
            self._update_selected_profile_context(target_row)
        item = self.profile_list.item(target_row)
        if item is not None:
            item.setData(Qt.UserRole, self._display_name(self._profiles[target_row]))
            provider_text = str(self._profiles[target_row].get("provider") or "").strip()
            model_name = str(self._profiles[target_row].get("model") or "").strip()
            enabled = "yes" if bool(self._profiles[target_row].get("enabled", True)) else "no"
            item.setToolTip(f"Provider: {provider_text}\nModel: {model_name}\nEnabled: {enabled}".strip())
        self._refresh_profile_counts()

    def _reconcile_active_profile(self) -> None:
        enabled_ids = [
            str(profile.get("id") or "").strip()
            for profile in self._profiles
            if bool(profile.get("enabled", True)) and str(profile.get("id") or "").strip()
        ]
        active_id = str(self._active_profile or "").strip()
        if active_id in enabled_ids:
            return
        self._active_profile = enabled_ids[0] if enabled_ids else ""

    def _toggle_profile_enabled(self, row: int, state: bool | int) -> None:
        if row < 0 or row >= len(self._profiles):
            return
        self._sync_current_profile_from_form(self._selected_row)
        preferred_profile_id = self._profile_id_for_row(row)
        scroll_value = self.profile_list.verticalScrollBar().value()
        if isinstance(state, bool):
            is_enabled = bool(state)
        elif isinstance(state, int):
            is_enabled = state == Qt.Checked
        else:
            is_enabled = bool(state)
        self._profiles[row]["enabled"] = is_enabled
        self._reconcile_active_profile()
        self._set_save_state("")
        self._refresh_profile_list(
            preferred_row=row,
            preferred_profile_id=preferred_profile_id,
            restore_scroll_value=scroll_value,
        )
        self._refresh_profile_counts()

    def _suggest_unique_id(self, model_text: str, *, row: int, base_url: str | None = None) -> str:
        used = {
            str(profile.get("id") or "").strip()
            for idx, profile in enumerate(self._profiles)
            if idx != row and str(profile.get("id") or "").strip()
        }
        if base_url is None:
            base_url = str(self.base_url_edit.text() or "").strip()
        return generate_profile_id(model_text, used, base_url)

    def _compute_initial_name_manual_flags(self) -> list[bool]:
        flags: list[bool] = []
        for idx, profile in enumerate(self._profiles):
            profile_id = str(profile.get("id") or "").strip()
            if not profile_id:
                flags.append(False)
                continue
            expected_auto_id = self._suggest_unique_id(
                str(profile.get("model") or ""),
                row=idx,
                base_url=str(profile.get("base_url") or ""),
            )
            flags.append(profile_id != expected_auto_id)
        return flags

    def _on_selection_changed(self, row: int) -> None:
        previous_row = self._selected_row
        if previous_row != row:
            self._sync_current_profile_from_form(previous_row)
        self._sync_form_to_profile(row)

    def _on_name_edited(self, text: str) -> None:
        row = self._current_row()
        if 0 <= row < len(self._name_manual_flags):
            self._name_manual_flags[row] = bool(str(text or "").strip())
        self._sync_current_profile_from_form()

    def _on_model_changed(self, _text: str) -> None:
        if self._loading_form:
            return
        row = self._current_row()
        if row < 0 or row >= len(self._profiles):
            return
        current_model = self._get_current_model_value()
        if self._model_state == ModelLoadState.LOADED:
            self._apply_entry_image_support(current_model)
            if 0 <= row < len(self._model_manual_flags):
                self._model_manual_flags[row] = current_model not in self._model_entries_by_id
        self._loading_form = True
        self.name_edit.setText(self._suggest_unique_id(current_model, row=row))
        self._loading_form = False
        if 0 <= row < len(self._name_manual_flags):
            self._name_manual_flags[row] = False
        self._on_form_changed()

    def _on_form_changed(self) -> None:
        self._set_save_state("")
        self._sync_current_profile_from_form()

    def _on_provider_changed(self, provider: str) -> None:
        if self._loading_form:
            return
        self._update_base_url_field_state(provider)
        self._invalidate_pending_fetches()
        self._clear_model_options()
        row = self._current_row()
        if 0 <= row < len(self._model_manual_flags):
            self._model_manual_flags[row] = False
        self._loading_form = True
        self._set_current_model_widgets_text("")
        if str(provider or "").strip().lower() not in {"openai", "anthropic"}:
            self.base_url_edit.clear()
        self._loading_form = False
        self._set_model_state(ModelLoadState.IDLE)
        if self._current_fetch_inputs() is not None:
            self._schedule_fetch(600)
        self._on_form_changed()

    def _add_profile(self) -> None:
        self._sync_current_profile_from_form(self._selected_row)
        self._profiles.append(
            {
                "id": "",
                "provider": "openai",
                "model": "",
                "api_key": "",
                "api_keys": [],
                "api_key_index": 0,
                "invalid_api_keys": [],
                "key_error_timestamps": {},
                "base_url": "",
                "supports_image_input": False,
                "enabled": True,
            }
        )
        self._name_manual_flags.append(False)
        self._model_manual_flags.append(False)
        self._set_save_state("")
        self._refresh_profile_list(preferred_row=len(self._profiles) - 1)
        self.name_edit.setFocus()

    def _duplicate_selected_profile(self) -> None:
        row = self._current_row()
        if row < 0 or row >= len(self._profiles):
            return
        self._sync_current_profile_from_form(self._selected_row)
        source = dict(self._profiles[row])
        duplicated = dict(source)
        duplicated["id"] = ""
        duplicated["enabled"] = bool(source.get("enabled", True))
        self._profiles.insert(row + 1, duplicated)
        self._name_manual_flags.insert(row + 1, False)
        self._model_manual_flags.insert(row + 1, bool(self._model_manual_flags[row]) if row < len(self._model_manual_flags) else False)
        self._set_save_state("Duplicated profile. Rename it before saving if needed.")
        self._refresh_profile_list(preferred_row=row + 1)
        self.name_edit.setFocus()

    def _delete_selected_profile(self) -> None:
        row = self._current_row()
        if row < 0 or row >= len(self._profiles):
            return
        self._sync_current_profile_from_form(self._selected_row)
        removed_id = str(self._profiles[row].get("id") or "").strip()
        self._profiles.pop(row)
        self._name_manual_flags.pop(row)
        if row < len(self._model_manual_flags):
            self._model_manual_flags.pop(row)

        if removed_id and removed_id == self._active_profile:
            self._active_profile = ""
        self._reconcile_active_profile()

        self._set_save_state("")
        self._refresh_profile_list(preferred_row=row)
        self._refresh_profile_counts()

    def _validated_payload(self) -> dict[str, Any] | None:
        self._sync_current_profile_from_form(self._selected_row)
        profiles: list[dict[str, str]] = []
        used_ids: set[str] = set()

        for idx, profile in enumerate(self._profiles):
            provider = str(profile.get("provider") or "").strip().lower()
            model_name = str(profile.get("model") or "").strip()
            if provider not in ALLOWED_PROVIDERS:
                self.profile_list.setCurrentRow(idx)
                QMessageBox.warning(self, "Validation", "Provider must be openai, gemini, or anthropic.")
                return None
            if not model_name:
                self.profile_list.setCurrentRow(idx)
                QMessageBox.warning(self, "Validation", "Model cannot be empty.")
                return None

            requested_id = sanitize_profile_id(profile.get("id") or "")
            if not requested_id:
                requested_id = generate_profile_id(model_name, set(), profile.get("base_url"))
            profile_id = ensure_unique_profile_id(requested_id, used_ids)

            validated_profile = {
                "id": profile_id,
                "provider": provider,
                "model": model_name,
                "api_key": str(profile.get("api_key") or "").strip(),
                "api_keys": list(profile.get("api_keys") or []),
                "api_key_index": int(profile.get("api_key_index") or 0),
                "invalid_api_keys": list(profile.get("invalid_api_keys") or []),
                "key_error_timestamps": dict(profile.get("key_error_timestamps") or {}),
                "base_url": str(profile.get("base_url") or "").strip(),
                "supports_image_input": bool(profile.get("supports_image_input")),
                "enabled": bool(profile.get("enabled", True)),
            }
            if isinstance(profile.get("reasoning"), dict):
                validated_profile["reasoning"] = dict(profile["reasoning"])
            profiles.append(validated_profile)

        active = str(self._active_profile or "").strip()
        enabled_ids = [item["id"] for item in profiles if bool(item.get("enabled", True))]
        if active not in enabled_ids:
            active = enabled_ids[0] if enabled_ids else ""
        payload = {"active_profile": active or None, "profiles": profiles}
        # Carry the Test-tab SESSION_SIZE override through normalization so it
        # is persisted alongside the profiles in .agent_state/config.json.
        session_size = getattr(self, "_session_size", None)
        if session_size is not None:
            payload["session_size"] = int(session_size)
        return payload

    def _persist_profiles(self, message: str) -> bool:
        validated = self._validated_payload()
        if validated is None:
            return False
        self._result_payload = normalize_profiles_payload(validated)
        self._profiles = [dict(item) for item in self._result_payload.get("profiles", [])]
        self._active_profile = str(self._result_payload.get("active_profile") or "").strip()
        self._name_manual_flags = self._compute_initial_name_manual_flags()
        if len(self._model_manual_flags) != len(self._profiles):
            self._model_manual_flags = [False] * len(self._profiles)
        current_row = self._current_row()
        preferred_row = current_row if current_row >= 0 else self._preferred_row_for_open()
        self._refresh_profile_list(preferred_row=preferred_row)
        self._refresh_profile_counts()
        self._set_save_state(str(message or "").strip())
        self.profiles_saved.emit(dict(self._result_payload))
        return True

    def _restore_save_button_text(self) -> None:
        if self.save_button is not None:
            self.save_button.setText("Save")
            self.save_button.setProperty("savedState", False)
            self.save_button.style().unpolish(self.save_button)
            self.save_button.style().polish(self.save_button)

    def _show_saved_button_state(self) -> None:
        if self.save_button is None:
            return
        self.save_button.setText("Saved")
        self.save_button.setProperty("savedState", True)
        self.save_button.style().unpolish(self.save_button)
        self.save_button.style().polish(self.save_button)
        self._save_button_reset_timer.start()

    def _save_and_accept(self) -> None:
        if self._persist_profiles("Saved. You can keep this window open and continue editing."):
            self._show_saved_button_state()


class ApprovalDialog(QDialog):
    def __init__(self, payload: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.choice: tuple[bool, bool] = (False, False)
        self.setObjectName("ApprovalDialog")
        self.setWindowTitle("Approval required")
        self.setModal(True)
        self.resize(640, 420)
        self.setMinimumSize(520, 340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        risk_level = str(summary.get("risk_level", "unknown") or "unknown")
        impacts = [str(item).strip() for item in list(summary.get("impacts", []) or []) if str(item).strip()]
        tools = list(payload.get("tools", []) or [])

        hero_card = QFrame()
        hero_card.setObjectName("ApprovalRequestCard")
        hero_layout = QVBoxLayout(hero_card)
        hero_layout.setContentsMargins(16, 14, 16, 14)
        hero_layout.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)

        title = QLabel("Protected action review")
        title.setObjectName("ApprovalCardTitle")
        title_row.addWidget(title)

        self.dialog_risk_badge = QLabel(risk_level.title())
        self.dialog_risk_badge.setObjectName("ApprovalRiskBadge")
        self.dialog_risk_badge.setProperty("riskLevel", risk_level)
        style = self.dialog_risk_badge.style()
        if style is not None:
            style.unpolish(self.dialog_risk_badge)
            style.polish(self.dialog_risk_badge)
        title_row.addWidget(self.dialog_risk_badge, 0, Qt.AlignVCenter)
        title_row.addStretch(1)
        hero_layout.addLayout(title_row)

        noun = "action" if len(tools) == 1 else "actions"
        summary_label = QLabel(f"The agent is paused. Review {len(tools)} protected {noun}.")
        summary_label.setObjectName("ApprovalCardSummary")
        summary_label.setWordWrap(True)
        hero_layout.addWidget(summary_label)

        impacts_label = QLabel(f"Will affect: {', '.join(impacts)}")
        impacts_label.setObjectName("ApprovalCardImpacts")
        impacts_label.setWordWrap(True)
        impacts_label.setVisible(bool(impacts))
        hero_layout.addWidget(impacts_label)
        layout.addWidget(hero_card)

        scroll = QScrollArea()
        scroll.setObjectName("ApprovalDialogScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)
        for tool in tools:
            card = QFrame()
            card.setObjectName("ApprovalToolCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 12, 12, 12)
            card_layout.setSpacing(4)

            tool_name = str(tool.get("name") or tool.get("display") or "tool").strip() or "tool"
            tool_args = dict(tool.get("args") or {})
            labels = build_tool_ui_labels(tool_name, tool_args, phase="finished")

            name_label = QLabel(labels.get("title") or str(tool.get("display") or tool_name))
            name_label.setObjectName("ApprovalToolTitle")
            card_layout.addWidget(name_label)

            subtitle = str(labels.get("subtitle", "") or "").strip()
            if subtitle:
                subtitle_label = QLabel(subtitle)
                subtitle_label.setObjectName("ApprovalToolSubtitle")
                subtitle_label.setWordWrap(True)
                card_layout.addWidget(subtitle_label)

            args_view = CopySafePlainTextEdit()
            args_view.setObjectName("ApprovalDetailView")
            args_view.setReadOnly(True)
            args_view.setPlainText(format_approval_detail_text(tool_args))
            _sync_plain_text_height(args_view, min_lines=6, max_lines=12, extra_padding=18)
            card_layout.addWidget(CollapsibleSection("Details", args_view, expanded=len(tools) == 1))
            container_layout.addWidget(card)
        container_layout.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox()
        approve_button = QPushButton("Approve")
        approve_button.setObjectName("PrimaryButton")
        always_button = QPushButton("Always allow")
        always_button.setObjectName("SecondaryButton")
        deny_button = QPushButton("Deny")
        deny_button.setObjectName("DangerButton")
        buttons.addButton(approve_button, QDialogButtonBox.AcceptRole)
        buttons.addButton(always_button, QDialogButtonBox.ActionRole)
        buttons.addButton(deny_button, QDialogButtonBox.RejectRole)
        layout.addWidget(buttons)

        approve_button.setDefault(True)
        approve_button.clicked.connect(self._approve)
        always_button.clicked.connect(self._always)
        deny_button.clicked.connect(self._deny)

    def _approve(self) -> None:
        self.choice = (True, False)
        self.accept()

    def _always(self) -> None:
        self.choice = (True, True)
        self.accept()

    def _deny(self) -> None:
        self.choice = (False, False)
        self.reject()
