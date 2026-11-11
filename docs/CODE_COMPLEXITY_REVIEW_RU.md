Анализ переусложнения и дублирования Python desktop agent

Дата исходного анализа: 10 ноября 2026. Анализ выполнен до рефакторинга, включая существовавшие незакоммиченные изменения; на этапе анализа код не изменялся. Последующие изменения описаны в [результатах рефакторинга](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/docs/CODE_COMPLEXITY_REFACTOR_RU.md). Числа и ссылки на строки ниже относятся к состоянию до этих изменений.

Проверка охватывала структуру 108 Python-файлов приложения в `core/`, `tools/`, `ui/`, `agent.py` и `main.py`: 42 255 строк и 1855 определений функций/методов, включая вложенные. Эти числа не включают тесты, venv, dist и данные пользователя. Для основных кандидатов выполнено чтение реализаций, анализ связей и локальные проверки поведения. Это не утверждение о построчном аудите всего приложения.

Основная проблема — несколько реализаций одних правил и незавершённое разделение ответственности. Наибольшую отдачу даст объединение обработки отображаемого текста, replay и классификации ошибок. Расхождения здесь уже воспроизводятся. Затем стоит упростить provider streaming и зависимости orchestrator/runtime. Наличие отдельного модуля или большой файл сами по себе не доказывают переусложнение.

Приоритеты ниже обозначают порядок рефакторинга. «Высокий» означает подтверждённое расхождение поведения либо большой риск дальнейшего рассогласования, «средний» — существенную стоимость сопровождения, «низкий» — локальную уборку.

| № | Кандидат | Приоритет | Основание |
|---|---|---|---|
| 1 | Единое извлечение видимого текста и фильтрация reasoning | Высокий | Разные списки скрытых блоков; подтверждённое различие live/history |
| 2 | Общая логика replay ответа | Высокий | Три похожих алгоритма с разными правилами границ |
| 3 | Единая классификация ошибок провайдера | Высокий | HTTP 500 получает разные решения о retry |
| 4 | Общая обработка sync/async OpenAI chunks | Средний | Два повторяющихся конвейера, зависимость от private API |
| 5 | Однократная классификация shell на одной границе проверки | Средний | Одна проверка approval трижды разбирает команду |
| 6 | Завершить разделение AgentNodes и orchestrators | Средний | 10 mixin; orchestrators обращаются к 12–22 членам owner |
| 7 | Убрать обратные зависимости через runtime facade | Средний | runtime → worker/session → runtime, поддержка patch через фасад |
| 8 | Переиспользовать выбор активного профиля и уменьшить ModelSettingsDialog | Средний | Повтор доменного правила в core и UI; 91 метод диалога |
| 9 | Один список подключений Qt signals | Низкий | Connect/disconnect вручную поддерживают одну таблицу |
| 10 | Общая логика clipboard и перемещения окна | Низкий | Точные дубли clipboard; похожие обработчики drag |
| 11 | Сборка обновлений состояния и технический результат agent | Низкий/средний | Повторяющиеся словари; служебные поля удаляются через pop |

1. **Обработка видимого текста: сначала устранить различия, затем объединить реализации.**

Места: [core/message_utils.py](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/message_utils.py:109), [runtime_payloads.py](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/runtime_payloads.py:558), [streaming.py](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/streaming.py:1389). Статическое удаление inline-thought также продублировано: [strip_inline_thought_blocks](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/runtime_payloads.py:551) и [_strip_inline_thought_content](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/streaming.py:102); регулярные выражения повторяются в начале обоих модулей.

Три рекурсивных обхода содержимого имеют разные списки скрытых типов и разные fallback. Например, `reasoning_delta` исключён в core и streaming, но отсутствует в списке `runtime_payloads._extract_visible_text`.

Локальная проверка для `AIMessage(content=[{"type": "reasoning_delta", "text": "INTERNAL-REASONING-SAMPLE"}])` дала:

| Путь | Результат |
|---|---|
| `visible_text_content` | Пустая строка |
| `StreamProcessor._extract_text_content` | Пустая строка |
| `runtime_payloads._extract_ai_text` | `INTERNAL-REASONING-SAMPLE` |

Это влияет на восстановление переписки: [build_transcript_payload](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/runtime_payloads.py:713) берёт результат `_extract_ai_text` и создаёт assistant-блок. Проверка использовала синтетическое сообщение; наличие такого блока в реальных сохранённых сессиях отдельно не исследовалось.

Рекомендация: расширить существующий `core/message_utils` одним контрактом извлечения отображаемого текста, общим набором типов reasoning и одной статической фильтрацией inline-тегов. Для типизированных LangChain-сообщений использовать стандартные `content_blocks`; для нестандартных aggregator payload оставить один compatibility fallback. Обрабатывать refusal как отдельный вид отображаемого содержимого.

Stateful `_InlineThoughtStreamFilter` нужен для тегов, разбитых между чанками: его нельзя заменить статической regex-функцией. Он должен пользоваться общими определениями тегов. `stringify_content` имеет другую задачу и не должен автоматически превращаться в фильтр отображаемого текста: результаты инструментов и исходные сообщения требуют сохранения содержимого.

Проверки после рефакторинга: одинаковый видимый текст live/restored для reasoning, analysis, thought, refusal, nested blocks и inline-тегов; отдельно — теги, разделённые между чанками.

2. **Объединить правила распознавания повторного префикса ответа.**

Места: [_assistant_replay_remainder](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/runtime_payloads.py:618), [_similar_prefix_boundary](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/streaming.py:1285), [_markdown_boundary_after_visible_prefix](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/widgets/transcript.py:52), [_assistant_resume_text](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/widgets/transcript.py:76).

Везде повторяются нормализация пробелов, проход по словам, сравнение startswith и `SequenceMatcher.quick_ratio() >= 0.94`. Дополнительные пороги 48/80 символов и 0.8 длины также повторяются. Однако widget проверяет границу незавершённого слова, history сразу режет по `incoming.startswith(previous)`, а streaming использует ещё один минимальный порог.

Локальная проверка:

| Вход | History helper | Transcript widget |
|---|---|---|
| previous = `Проверя`; incoming = `Проверяю файлы` | `ю файлы` | `Проверяю файлы` |

Для незавершённого слова helper истории убирает часть нового слова. Эти пути должны явно согласовать правило, когда префикс считается повтором. Фактический вызов history helper выполняется после tool-результата в [build_transcript_payload](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/runtime_payloads.py:719).

Рекомендация: одна чистая функция поиска replay-границы с явно заданными правилами законченных слов и достаточной длины совпадения. Пусть она возвращает границу/признак совпадения, а каждый потребитель решает, как обновить свою модель отображения. Live streaming и восстановление истории требуют разных буферов, но правил сравнения текста — общих.

Точные дубли `_normalized_visible_text`, `StreamProcessor._normalized_text`, `ConversationTurnWidget._normalized_text` можно убрать вместе с этим изменением. Общий поиск точного префикса также повторяется в [transcript.py](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/widgets/transcript.py:40) и [main_window.py](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/window_components/main_window.py:587). Sequence guard окна решает отдельную задачу порядка событий; объединять его целиком с replay не требуется.

Проверки: partial words, новый ответ с общим первым словом, replay с пунктуационными отличиями, повтор Markdown, границы tool-блоков, сброс sequence.

3. **Один классификатор ошибки, несколько явно различных retry-политик.**

Места: [LLMMixin._is_transient_llm_error](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/nodes/llm.py:326), [AgentRunWorker._classify_stream_error](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/runtime_worker.py:638), [describe_provider_error](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/api_key_rotation.py:115), [classify_api_key_error](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/api_key_rotation.py:168).

Core уже распознаёт HTTP 408/429/500/502/503/504 и обходит цепочку исключений. Worker заново классифицирует строку, не использует структурный status для распознавания 5xx и имеет другой набор маркеров.

Для `RuntimeError("HTTP 500 Internal Server Error")` со `status_code=500` локально получено:

| Проверка | Результат |
|---|---|
| `LLMMixin._is_transient_llm_error` | `True` |
| `AgentRunWorker._classify_stream_error(..., {"http_status": 500})` | `kind="unknown", retryable=False` |

Это доказывает различие классификаторов. Оно не доказывает, что весь runtime обязан повторять каждый HTTP 500: отдельный флаг `retry_exhausted` и бюджет повторов должны продолжать ограничивать retry.

Рекомендация: существующие безопасные diagnostics дополнить общей категорией ошибки на основе status, provider code и цепочки исключений. UI должен отображать готовую категорию, а не повторно выводить её из текста.

Ротация ключей, повтор вызова LLM и восстановление прерванного графа выполняют разные операции. Их бюджеты и действия следует сохранить отдельно. Унифицировать нужно классификацию, правила exhausted/permanent и общие расчёты там, где они действительно совпадают.

Проверки: 408, 429, 500/502/503/504, auth/billing, отсутствие status, вложенная network error, exhausted budget и закрытый клиент.

4. **OpenAI streaming: выделить общую обработку чанков из двух циклов.**

Места: [_stream](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/providers/openai_reasoning.py:518), [_astream](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/providers/openai_reasoning.py:652).

Методы занимают 133 и 150 строк соответственно. В обоих повторяются выбор Chat/Responses API, подготовка payload, headers, учёт индексов, обработка reasoning, conversion chunks, callbacks, API errors и final completion. Для Responses почти один и тот же блок находится на строках 550–579 и 697–728.

Динамическая фабрика subclass занимает 546 строк, но включает вложенный класс и его методы: это не один алгоритм с 546 строками. Основной долг — повтор конвейеров и копирование поведения, зависящего от private `langchain_openai.chat_models.base` API.

Рекомендация: вынести общий шаг обработки Responses chunk с состоянием индексов/reasoning и общий шаг Chat Completions chunk. В sync/async методах оставить получение данных, context manager и соответствующий вызов callback. Пользоваться уже выделенными `_restore_responses_reasoning_text` и `_attach_reasoning_content`.

Не переносить синхронное чтение потока внутрь async-пути: это изменит timeout и cancellation. Не удалять нестандартные поля агрегаторов, encrypted reasoning и thought signatures без проверки совместимости. Стандартные `content_blocks` уменьшают собственную нормализацию после получения сообщения, но не гарантируют автоматическую передачу всех нестандартных полей провайдера.

Проверки: sync/async parity для одинаковых сырых событий, reasoning delta/final item, cache usage, structured output, headers и timeout. Существующие provider tests — основа для этого изменения.

5. **Shell policy: одна команда разбирается трижды в одной проверке.**

Места: [_effective_tool_metadata](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/nodes/tool_preflight.py:231), [_tool_requires_approval](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/nodes/tool_preflight.py:256), [shell_command_requires_approval](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/policy_engine.py:636), [tool_requires_approval](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/policy_engine.py:652).

Цепочка `_tool_requires_approval` сначала вызывает `_effective_tool_metadata`: тот разбирает команду сам и снова через `shell_command_requires_approval`. Затем `tool_requires_approval` повторно вызывает `shell_command_requires_approval`.

Проверка с mock-счётчиками и командой `Get-Content README.md`: один вызов `_tool_requires_approval` дал 3 вызова `classify_shell_command`. Команда при этом не исполнялась.

[ToolBatchCoordinator](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/node_orchestrators.py:1100) перед этим уже получает metadata, затем отдельно вызывает approval predicate. Следовательно, повторное вычисление происходит и внутри обработки инструмента.

Рекомендация: отделить чистое решение по уже разобранному shell profile от функции, которая разбирает строку. Из одного profile получать effective metadata и approval decision. На каждой реальной границе approval/execution оценивать актуальные args, но внутри одной оценки использовать один результат. Глобальное кеширование команд не требуется.

Это прежде всего устранение повторного вычисления и зависимости правил в нескольких местах. Замер ускорения приложения не проводился.

6. **AgentNodes: разделение на файлы пока не стало разделением ответственности.**

Места: [AgentNodes](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/nodes/__init__.py:27), [_init_nodes](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/nodes/__init__.py:124), [ContextMixin](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/nodes/context.py:17), [AgentTurnOrchestrator](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/node_orchestrators.py:273), [ToolBatchCoordinator](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/node_orchestrators.py:703).

`AgentNodes` наследует 10 mixin, создаёт ContextBuilder/RecoveryManager/ToolExecutor и ещё три orchestrator с `self` в качестве owner. Общая реализация узлов занимает 2594 строки, orchestrators — ещё 1200.

AST-анализ обращений через локальную переменную `owner` показал:

| Orchestrator | Различных членов owner |
|---|---:|
| AgentTurnOrchestrator | 20 |
| RecoveryTurnOrchestrator | 12 |
| ToolBatchCoordinator | 22 |

Большая часть обращения идёт к приватным методам. Например, orchestrator вызывает owner._build_agent_context, который через mixin вызывает context_builder.build. Protocol описывает этот большой owner, но не уменьшает фактическую связанность.

Рекомендация: последовательно завершить уже начатое выделение сервисов. Первыми передавать существующие ContextBuilder/RecoveryManager/ToolExecutor непосредственно в места их использования, а оставшиеся входы сделать явными. Recovery — удобный начальный участок, поскольку логика планирования уже вынесена в RecoveryManager.

Убирать forwarding wrapper имеет смысл после переноса его внутренних вызовов и тестов. Публичный `AgentNodes` можно сохранить как точку сборки и API графа. Массовое добавление новых интерфейсов, универсального pipeline или service locator увеличит текущую проблему.

7. **Runtime facade используется как обратный источник зависимостей.**

Места: [ui/runtime.py](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/runtime.py:1), [_runtime_module в worker](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/runtime_worker.py:53), [_runtime_module в coordinator](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/runtime_session.py:26), [RuntimeSessionCoordinator](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/runtime_session.py:32), [resolver диалога](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/window_components/main_window.py:873).

`ui.runtime` импортирует worker и payloads. Worker и coordinator затем лениво импортируют `ui.runtime`, чтобы получить setup/build/repair/Path/os. Сам facade экспортирует `os` и `Path`. Тесты patch именно эти символы в публичном модуле. В окне похожая поддержка ищет заменённый ModelSettingsDialog через `sys.modules["main"]` и `sys.modules["ui.main_window"]`.

В worker также 16 методов обращаются к coordinator; многие являются короткими forwarding wrapper. Coordinator при этом получает весь worker как `Any` и изменяет его поля напрямую.

Рекомендация: facade оставить для стабильных импортов; реализации должны импортировать реальные зависимости в одном направлении. Для тестов patch делать в модуле использования. Если нужна подмена фабрики runtime или диалога — передавать конкретную factory через конструктор, без поиска в `sys.modules`.

Coordinator должен владеть своей session/profile-логикой и возвращать результат, который worker публикует через signals. Начать можно с импорта factory/repair и сокращения wrappers; полное изменение владения состоянием — отдельный этап.

Совместимость публичных импортов уже проверяется `tests/test_main_window_facade.py`; её следует сохранить или изменить явно. Удаление facade само по себе не решает связанность.

8. **Доменное правило активного профиля повторяется в UI.**

Места: [normalize_profiles_payload](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/model_profiles.py:316), [_reconcile_active_profile](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/widgets/dialogs.py:1955).

Оба места составляют список enabled ids, сохраняют активный id, если он доступен, иначе выбирают первый enabled профиль. Это один инвариант, которому полезно иметь одну реализацию.

Рекомендация: выделить небольшую функцию выбора активного id в `core/model_profiles` и использовать её в normalize и UI. Не вызывать полную нормализацию persisted payload на каждом изменении формы: она дополнительно отбрасывает невалидные профили, дедуплицирует их и назначает id, а черновик формы может быть временно неполным.

ModelSettingsDialog занимает 1892 строки и содержит 91 метод. Он одновременно строит UI, управляет формой и профилями, загрузкой моделей, статусами запросов и геометрией окна. После выделения общих правил разумно отделить profile editing state и model discovery coordination от виджетов. Простое разнесение 91 метода по нескольким файлам без изменения владения данными даст ограниченную пользу.

Проверки: выключение активного профиля, пустой список enabled, удаление профиля и сохранение незавершённой формы.

9. **Подключения Qt поддерживаются в двух вручную совпадающих списках.**

Места: [_start_worker_thread](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/runtime_worker.py:1425), [_disconnect_worker_thread](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/runtime_worker.py:1459).

Connect и disconnect повторяют пары сигнал/слот. При добавлении команды worker нужно синхронно изменять два места, при этом часть подключений имеет специальный connection type.

Рекомендация: при подключении сохранять конкретные пары/connection handles и отключать их по сохранённому списку. Оставить connection type явным, особенно `QueuedConnection` у stop_run.

Сам подход QObject → moveToThread → signals и связь `thread.finished → worker.deleteLater` соответствуют документации Qt. Их наличие не является лишним усложнением; убирать worker thread ради сокращения строк не следует.

10. **Точные локальные дубли в виджетах.**

Clipboard: [CopySafePlainTextEdit](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/widgets/foundation.py:143) и [AutoTextBrowser](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/widgets/foundation.py:556) содержат одинаковые `createMimeDataFromSelection`, `copy`, `keyPressEvent`.

Рекомендация: общий небольшой clipboard behavior либо helper для получения нормализованного selection и MIME. Один узкий mixin для поведения двух Qt-виджетов оправданнее, чем копирование трёх методов; если MRO становится неудобным, оставить маленькие overrides, делегирующие helper.

Перемещение окна: [ModelSettingsDialog](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/widgets/dialogs.py:1001) и [DialogTitleBar](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/ui/widgets/window_chrome.py:55) повторяют startSystemMove, fallback drag-offset и перемещение. Drag zone различается, поэтому переносить обработчики целиком без адаптации нельзя. Общими могут стать начало/обновление drag либо существующий title bar.

Существуют и однострочные дубли `_clean_text`, `_compact_text`, `_normalize_tool_name`. Их приоритет ниже: общий util для каждой пары строк легко добавляет больше связей, чем удаляет. Для нормализации replay уже есть содержательная причина создать общий helper.

11. **Повторяются словари state update и промежуточные вызовы LLM.**

Места: [AgentTurnOrchestrator.run](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/node_orchestrators.py:366), [второй early return](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/node_orchestrators.py:410), [_build_agent_result](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/nodes/agent.py:318), [tool batch payload](D:/py_projects/simple_ai_agent/agent+stategraph/v0.67.73.272b/core/node_orchestrators.py:884).

Повторяется набор `turn_id/current_task/turn_outcome/recovery_state/pending_approval/open_tool_issue/last_tool_error/last_tool_result`; запись новых сообщений в `messages` и `transcript_messages` тоже собирается вручную.

Дополнительно `_build_agent_result` возвращает `_retry_user_input_turn` и `_rejected_tool_calls` вместе со state update, а orchestrator вынимает их через `pop`. Вызов invoke → build result повторён до и после user-choice correction.

Рекомендация: небольшие helper для действительно одинаковых recover updates и пары message delta. Для state update и временных результатов возвращать отдельные значения либо небольшой dataclass. Общий invoke/build шаг можно вынести в локальную функцию orchestrator.

LangGraph принимает частичные обновления: не нужно каждый раз собирать полный AgentState. Helper не должен автоматически обнулять все необязательные поля. Сбрасывать `pending_approval` и stale issue следует только там, где это часть конкретного перехода.

Часть повторов обязательна по смыслу: `messages` и `transcript_messages` имеют разные reducer и назначение. Их объединение потеряет полную историю при compaction.

**Что уже хорошо переиспользуется и стоит сохранить.**

- `core/tool_args.tool_loop_budget` используется на двух guard-границах. Повтор проверок на разных границах допустим; правило уже общее.
- `core/tool_results` и `core/tool_issues` выделяют разбор результата и построение issue.
- Provider factories отделены от графа.
- Filesystem pathing/editing/manager уже разделены по содержательной ответственности.
- `messages` с `add_messages` и отдельный transcript reducer обеспечивают разные требования контекста и истории.
- Собственный ToolBatchCoordinator решает порядок результатов, sequential barriers, approvals и управление GraphBubbleUp. Его нельзя без проверки заменить стандартным ToolNode только ради краткости.
- Legacy recovery fields документированы как входы миграции checkpoint. Их удаление требует миграционной стратегии, а не только поиска текущих записей.

**Порядок работ.**

1. Согласовать один контракт видимого текста, устранить подтверждённое отличие reasoning_delta, вынести общие правила тегов.
2. Согласовать replay-границы и классификатор ошибок с отдельными retry-бюджетами.
3. Устранить повторный shell parsing внутри одной проверки.
4. Выделить обработку OpenAI chunks, сохранив sync/async I/O и cancellation.
5. Направить зависимости runtime в одну сторону, затем постепенно сузить owner orchestrators.
6. Убрать повтор выбора профиля, signal lists и clipboard; после этого оценить, осталось ли оправданным выделение частей крупных UI-классов.

Измерять результат лучше числом мест, где реализуется одно правило, размером зависимостей owner и совпадением live/history, чем только сокращением строк. Процент потенциального уменьшения кода не рассчитан.

**Проверка и Context7.**

Полный regression run: `venv\\Scripts\\python.exe -m pytest -q`.

Результат: **1693 passed, 792 subtests passed, 1 warning**, 487.47 секунды. Предупреждение относится к deprecated `_UnionGenericAlias` в `google/genai/types.py`. Прохождение тестов не покрывает автоматически найденные расхождения: они проверены отдельными локальными вызовами существующих функций. Новые тесты и изменения реализации в рамках анализа не добавлялись.

Локальные версии: LangGraph 1.2.16, langchain-core 1.6.12, langchain-openai 1.7.2, PySide6 6.12.0, pytest 9.1.1. Версии проверены в проектном venv; численные данные — для состояния файлов на момент анализа.

Через Context7 получены официальные документы:

- [LangChain messages](https://docs.langchain.com/oss/python/langchain/messages), library ID `/websites/langchain_oss_python_langchain`: стандартные `content_blocks`, включая нормализацию reasoning. Основание для общего message adapter; наличие repo-specific fallback всё ещё необходимо проверить.
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) и [Memory / RemoveMessage](https://docs.langchain.com/oss/python/langgraph/add-memory), library ID `/websites/langchain_oss_python_langgraph`: partial state updates, reducers, обновление сообщений по id и удаление через RemoveMessage.
- [Qt QThread](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html) и [QObject](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QObject.html), library ID `/websites/doc_qt_io_qtforpython-6`: worker ownership, queued connections и deleteLater.

Документация использована для проверки API и библиотечных ограничений. Выводы о дублировании и расхождениях основаны на локальном коде и проверках поведения.
