# План: панель Settings как безрамочное окно по центру

## Цель

Переделать панель Settings (`ModelSettingsDialog`) из правого `QDockWidget`
на весь экран в самостоятельное безрамочное (`FramelessWindowHint`) окно,
которое при открытии появляется по центру экрана.

## Текущее состояние

- `ui/widgets/dialogs.py`, класс `ModelSettingsDialog(QDialog)` (строка ~235):
  - `setObjectName("ModelSettingsDialog")`, `setModal(False)`,
    `setWindowModality(Qt.NonModal)`, `setAttribute(Qt.WA_DeleteOnClose, True)`.
  - Размер задаётся от `QApplication.primaryScreen().availableGeometry()`
    (824x680, минимум 660x450).
  - Есть `close_button` (QDialogButtonBox.Close), подключён к `self.reject`.
- `ui/window_components/main_window.py`, `_open_settings_dialog()` (строка ~822):
  - Создаёт диалог и **оборачивает его в `QDockWidget`** справа
    (`ModelSettingsDock`), растягивает на всю ширину окна.
  - Вспомогательные методы: `_size_settings_dock()` (~1177),
    `_on_settings_dock_visibility_changed()` (~1169), вызовы из
    `resizeEvent` (~1200).
  - `_model_settings_window` хранит либо `QDockWidget`, либо сам диалог.
  - `_model_settings_window_is_visible()` (~610) учитывает оба варианта.
- Тема `ui/theme.py`: стили `QDialog#ModelSettingsDialog ...` (строки ~232-780).
- Тесты `tests/test_cli_ux.py` завязаны на dock:
  - `test_settings_dock_does_not_expand_window_past_requested_width` (~3246);
  - `test_model_settings_panel_reopens_...` (~3839, ~3950, ~4017) —
    используют `dock = self.window._model_settings_window`,
    `dock.widget()`, `dock.isHidden()`.

## Подход

1. Диалог остаётся `QDialog`, но становится top-level безрамочным окном.
2. Кастомный заголовок (drag handle) вместо системного, т.к. рамка убрана.
3. Центрирование по `availableGeometry()` активного экрана.
4. `main_window` перестаёт оборачивать диалог в dock — просто показывает окно.

### Флаги окна

```python
self.setWindowFlags(
    Qt.WindowType.Window
    | Qt.WindowType.FramelessWindowHint
)
# опционально, для скруглённых углов и тени:
# self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
```

`Qt.Window` нужен, чтобы окно было самостоятельным top-level, а не
всплывающим popup. `FramelessWindowHint` убирает системную рамку и заголовок.

### Центрирование

Использовать экран родителя/виджета, а не всегда primary:

```python
def _center_on_screen(self) -> None:
    screen = self.screen() or QApplication.primaryScreen()
    if screen is None:
        return
    geo = screen.availableGeometry()
    frame = self.frameGeometry()
    frame.moveCenter(geo.center())
    self.move(frame.topLeft())
```

Вызывать в `showEvent` (после того как размеры применены) и/или один раз
после `resize`. Ориентир из документации Qt for Python (пример undock):

```python
geometry = self.screen().availableGeometry()
x = geometry.x() + (geometry.width() - self.width()) / 2
y = geometry.y() + (geometry.height() - self.height()) / 2
self.move(x, y)
```

### Перетаскивание безрамочного окна

Рекомендуемый способ (Qt 5.15+/6) — системный move, корректно работает с
Windows snap:

```python
def mousePressEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton and self._is_drag_zone(event.position()):
        handle = self.windowHandle()
        if handle is not None and handle.startSystemMove():
            event.accept()
            return
    super().mousePressEvent(event)
```

Альтернатива (ручной drag) — запоминать `event.globalPosition().toPoint()`
и смещение, двигать окно в `mouseMoveEvent`. Предпочтителен
`startSystemMove()`.

Зона перетаскивания: верхняя полоса/hero-карточка (`ModelSettingsHeroCard`).
Практично вынести заголовок в отдельный `QWidget` (`SettingsTitleBar`) с
`objectName` для стилей и повесить drag на него, чтобы клики по полям ввода
не двигали окно.

### Скруглённые углы и тень (опционально)

- `WA_TranslucentBackground` + внутренний контейнер `QFrame` с
  `border-radius` и `QGraphicsDropShadowEffect`.
- Риск: на Windows прозрачность иногда даёт артефакты и усложняет
  тесты/скриншоты. Если не критично — оставить непрозрачное окно с
  прямыми углами (минимальный риск).

## Изменения по файлам

### `ui/widgets/dialogs.py`

- В `ModelSettingsDialog.__init__`:
  - добавить `FramelessWindowHint` (+ `Qt.Window`);
  - добавить кастомный заголовок с кнопкой закрытия и зоной drag;
  - добавить `_center_on_screen()` и вызвать в `showEvent`;
  - реализовать `mousePressEvent`/`mouseMoveEvent` (или drag на заголовке)
    через `windowHandle().startSystemMove()`.
- `close_button` оставить подключённым к `reject` (dock-переподключение
  больше не нужно).
- Проверить, что `WA_DeleteOnClose=True` не конфликтует с повторным
  открытием (сейчас в dock-пути его сбрасывали в `False`).

### `ui/window_components/main_window.py`

- `_open_settings_dialog()`: убрать ветку с `QDockWidget`; создавать диалог,
  подключать `profiles_saved`, показывать:
  ```python
  dialog = dialog_class(self.model_profiles_payload, self)
  dialog.profiles_saved.connect(self._save_model_profiles_from_dialog)
  self._model_settings_window = dialog
  dialog.destroyed.connect(lambda *_a: setattr(self, "_model_settings_window", None))
  dialog.show()
  dialog.raise_()
  dialog.activateWindow()
  ```
- Удалить `_size_settings_dock()` и `_on_settings_dock_visibility_changed()`,
  убрать их вызовы из `resizeEvent`.
- `_model_settings_window_is_visible()` упростить (оставить проверку
  `isHidden`/`isVisible`).
- Убрать неиспользуемые импорты (`QDockWidget`, `QSizePolicy` при
  необходимости) — проверить `rg`.

### `ui/theme.py`

- Добавить стили кастомного заголовка (например
  `QWidget#SettingsTitleBar`, `QPushButton#SettingsCloseButton`), hover/close.
- Существующие `QDialog#ModelSettingsDialog ...` оставить; при
  `WA_TranslucentBackground` добавить фон/радиус контейнеру.

### `tests/test_cli_ux.py`

- Переписать тесты, завязанные на dock:
  - `test_settings_dock_does_not_expand_window_past_requested_width` —
    заменить на проверку, что окно Settings не меняет размер главного окна и
    центрируется в пределах `availableGeometry`.
  - Тесты reopen/close — вместо `dock.widget()`/`dock.isHidden()` использовать
    сам диалог (`self.window._model_settings_window`) и его `isHidden()`.
- Добавить тест: флаги окна содержат `FramelessWindowHint`; после `show()`
  центр окна близок к центру `availableGeometry` (допуск несколько px).

## Краевые случаи

- Несколько мониторов: центрировать на экране родителя (`self.screen()`),
  а не на primary.
- Окно больше доступной области: ограничить размер
  `min(desired, available)` (уже частично есть).
- Повторное открытие: `WA_DeleteOnClose` удаляет диалог — ссылку
  `_model_settings_window` обнулять через `destroyed`.
- Busy/approval: сохранить текущую логику блокировки открытия.
- Фокус/активация: `raise_()` + `activateWindow()` после `show()`.
- Закрытие по Esc: `QDialog` обрабатывает по умолчанию.

## Риски

- `WA_TranslucentBackground` на Windows: артефакты, проблемы с тенью.
- `startSystemMove()` возвращает `False` в некоторых средах — нужен
  fallback на ручной drag.
- Тесты, использующие dock API, сломаются — обязательны правки тестов.

## Чек-лист (реализовано)

- [x] Флаги: `Qt.Window | FramelessWindowHint`.
- [x] Кастомный заголовок (hero-карточка) + drag (`startSystemMove` с fallback).
- [x] Центрирование по `availableGeometry` в `showEvent`.
- [x] Адаптация размера к экрану: `_apply_screen_geometry()` клампит размер и
      минимум под `availableGeometry` минус `SCREEN_MARGIN` (24 px).
- [x] `main_window`: убрать dock, показывать диалог напрямую (с toggle).
- [x] Удалить `_size_settings_dock` / `_on_settings_dock_visibility_changed` /
      `_settings_dock_visible`.
- [x] Стили кнопки закрытия `SettingsCloseButton` в `theme.py`.
- [x] Обновить тесты dock -> dialog, добавить тест центрирования/флагов.
- [x] Прогнать `pytest tests` — 1029 passed.
- [x] Пропорциональный размер окна (`WIDTH_RATIO`/`HEIGHT_RATIO`, cap
      `PREFERRED_SIZE`, floor `MINIMUM_SIZE`, кламп по `availableGeometry`).
- [x] Elide длинных имён профилей (`ElidedLabel` + tooltip с полным именем).
- [x] Увеличены минимальные ширины колонок (профили 300, левая 340, правая 440).
- [x] Блокировка ввода главного окна при открытом Settings: сигнал
      `visibility_changed` -> `_handle_settings_visibility_changed` ->
      `_set_input_enabled` (composer, sidebar, info/attach отключаются).
- [x] Тест `test_settings_window_blocks_main_window_input`.

## Результаты проверки

- `pytest tests/test_cli_ux.py tests/test_main_window_facade.py` — 216 passed.
- `pytest tests` — 1030 passed, 97 subtests.
- Ручная проверка клампа (offscreen) на 1366x768, 1024x600, 800x600,
  1920x1080, 3840x2160, 640x480: окно всегда внутри экрана, центр совпадает
  с центром `availableGeometry`.

## Источники (Context7, Qt for Python 6)

- `QScreen.availableGeometry` — доступная геометрия экрана без панелей задач.
- Пример undock (`example_opengl_hellogl2`): центрирование окна через
  `availableGeometry` и `move(x, y)`.
- Пример Filesystem Explorer: безрамочное окно + кастомные элементы
  управления окном и перетаскивание.
