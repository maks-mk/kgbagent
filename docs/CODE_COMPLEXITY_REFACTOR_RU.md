# Выполненный рефакторинг Python desktop agent

Дата: 10 ноября 2026. Основание — [исходный анализ](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/docs/CODE_COMPLEXITY_REVIEW_RU.md). Этот документ описывает изменения рабочей копии; числа и ссылки на строки в исходном анализе относятся к состоянию до рефакторинга.

Основной результат — единые правила отображения сообщений, удаления повторного префикса ответа и классификации ошибок провайдера. Также устранено повторение обработки OpenAI chunks, повторный shell parsing внутри одной проверки approval и обратные импорты через runtime facade.

| Пункт анализа | Реализовано | Практический результат |
|---|---|---|
| 1. Видимый текст | `core.message_utils.visible_text_content`, общие regex и статический фильтр inline-thought | Live и восстановленная история скрывают одинаковые reasoning-блоки; refusal и вложенные payload обрабатываются общим обходом |
| 2. Replay | Чистые функции в `core/text_replay.py` | История, transcript widget и streaming используют общие правила сравнения; незавершённое слово `Проверя` не превращает новый ответ `Проверяю файлы` в `ю файлы` |
| 3. Ошибки провайдера | `classify_provider_error` в `core/api_key_rotation.py`; core и worker делегируют ему | HTTP status, provider code и вложенные причины учитываются одинаково; HTTP 500 распознаётся в UI как временная ошибка |
| 4. OpenAI streaming | `_OpenAIStreamState`, общие шаги Chat/Responses и обработка final completion | Преобразование чанков реализовано один раз для каждого API; sync/async сохраняют собственные циклы ввода-вывода |
| 5. Shell-policy | Проверка готового shell profile / ToolMetadata | Одна standalone preflight-проверка непустой CLI-команды разбирает её один раз; tool batch повторно использует уже вычисленные metadata |
| 6. Orchestrators | Явная передача ContextBuilder, RecoveryManager и ToolExecutor; общие функции чтения state | Уменьшено число зависимостей от owner; RecoveryTurnOrchestrator обращается к owner только за логированием, задачей и лимитами |
| 7. Runtime facade | `ui/runtime_setup.py`, прямые импорты builder и session services | Worker/session больше не импортируют facade для получения собственных зависимостей; публичный facade продолжает экспортировать runtime API |
| 8. Активный профиль | `select_active_profile_id` используется core и диалогом настроек | Сохранение доступного активного id и выбор первого enabled-профиля имеют одну реализацию |
| 9. Qt signals | Connection handles сохраняются при connect и используются при disconnect | Второй вручную поддерживаемый перечень сигналов и слотов удалён |
| 10. Clipboard | `CopySafeTextMixin` для двух текстовых виджетов | Нормализация selection, MIME и обработка Copy больше не дублируются |
| 11. State updates | `AgentTurnResult`, `message_state_update`, `recover_agent_update`; общий invoke/build шаг | Временные решения orchestrator отделены от обновлений графа; служебные поля больше не удаляются через `pop` |

## Поведение и совместимость

`visible_text_content` использует стандартные `content_blocks` для LangChain-сообщений и общий fallback для нестандартных payload, включая legacy `additional_kwargs.content_blocks`. Фильтрация отображения не меняет исходное сообщение и не удаляет provider reasoning из сохранённой истории. Stateful-фильтр streaming сохранён для `<think>`-тегов, разделённых между чанками. Refusal отображается в переписке; генерация заголовка явно исключает его через `include_refusal=False`.

У сравнения replay общие правила, но streaming сохраняет свой минимальный порог совпадения в 24 символа и собственные буферы. Порядок событий окна остаётся отдельной проверкой.

Ротация ключей, повторы LLM и восстановление прерванного stream сохраняют отдельные бюджеты. Исчерпанный rate-limit pool может использовать внешний LLM retry; `retry_exhausted` в diagnostics запрещает дальнейшее автоматическое восстановление stream. Закрытый клиент и постоянные HTTP 4xx не классифицируются как временная network error.

OpenAI adapter сохраняет reasoning text, encrypted reasoning, cache usage, response headers, callbacks и структурированный финальный ответ. Async-путь продолжает ожидать callbacks и использовать `_astream_with_chunk_timeout`. Зависимость адаптера от private API `langchain_openai` остаётся; существующие adapter tests и новые тесты проверяют совместимость с установленной версией.

В runtime сохранены публичные экспорты функций и классов. Внутренние тестовые mocks перенесены на модули, владеющие соответствующей зависимостью. Служебные экспорты `os`/`Path` из facade удалены. `MainWindow` принимает необязательную `settings_dialog_factory`, а выбор реализации диалога больше не зависит от поиска в `sys.modules`.

Существующие reducer для `messages` и `transcript_messages`, legacy recovery migration и схема checkpoint сохранены. `AgentTurnResult` используется внутри agent/orchestrator и не записывается в граф. Подключение `thread.finished → worker.deleteLater` и `QueuedConnection` для stop_run сохранены.

## Проверки

Новые regression tests:

- [test_display_contract.py](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/tests/test_display_contract.py): одинаковое отображение live/history, provider payload, legacy content blocks, различие refusal/title, разбитые inline-теги, partial words и replay.
- [test_provider_error_contract.py](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/tests/test_provider_error_contract.py): HTTP 408/429/5xx и постоянные 4xx, provider codes, циклическая цепочка исключений, закрытый клиент и exhausted budgets.
- [test_openai_stream_contract.py](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/tests/test_openai_stream_contract.py): одинаковые sync/async chunks и запросы через настоящий SDK с локальным HTTP transport; headers, пропущенные события, callbacks, reasoning, cache usage и final completion.
- Дополнение [test_policy_engine.py](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/tests/test_policy_engine.py): один разбор CLI-команды, переиспользование metadata и прежние решения approval для read-only, background, mutating и пустых команд.

Финальный полный запуск `venv\Scripts\python.exe -m pytest -q`: **1739 passed, 798 subtests passed, 1 warning**, 363.48 секунды. Предупреждение — deprecated `_UnionGenericAlias` в сторонней библиотеке `google/genai/types.py`.

Проверка синтаксиса `venv\Scripts\python.exe -m compileall -q core ui agent.py main.py` и `git diff --check` завершились без ошибок. Git сообщает о преобразовании LF → CRLF для `core/api_key_rotation.py` в соответствии с настройками рабочей копии.

Документация LangChain/LangGraph/Qt проверена через Context7; ссылки и версии приведены в исходном анализе. Дополнительно подтверждён Qt API хранения connection handles и их отключения через `QObject.disconnect`.

## Границы выполненной работы

Крупное выделение profile editing state и model discovery coordinator из `ModelSettingsDialog`, общий механизм перемещения окон и полная замена mixin-структуры `AgentNodes` не выполнялись. Для них нужно отдельно определить владение изменяемым состоянием; простое разнесение методов по файлам не устраняет связанность.

Изменения внесены поверх существующих правок пользователя. Сборка desktop-приложения и проверка реальными сетевыми запросами к провайдерам в этот этап не входят; SDK streaming проверяется локальным transport.
