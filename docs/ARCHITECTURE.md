# Архитектура

## Runtime Flow

```text
START
  → summarize        # сжать контекст если сессия стала большой
  → update_step
  → agent            # LLM решает: ответить / вызвать tool / recovery
     → approval      # пауза перед мутирующим действием
        → tools
     → tools         # исполнить tool calls (read-only — параллельно, остальные — последовательно)
        → recovery   # если tool вернул ошибку
        → summarize → update_step
     → recovery      # если агент вернул protocol error или loop
        → summarize → update_step
        → END
     → END
```

- `MAX_LOOPS` и per-tool loop guards предотвращают бесконечные циклы.
- Recovery использует stateful error tracking: `attempts_by_strategy`, `progress_markers`, `llm_replan_attempted_for` — адаптивные повторы с учётом уникальных fingerprints ошибок.
- При смене проблемы (новый fingerprint) retry-бюджет сбрасывается; для одной и той же проблемы разрешены несколько `llm_replan` попыток в рамках `SELF_CORRECTION_RETRY_LIMIT`.
- Stream-interruption recovery: при обрыве потока провайдера история автоматически чинится, ошибка классифицируется (`rate_limit` / `timeout` / `server_error` / `network`), и запуск продолжается после backoff с джиттером. Для обычных ошибок используется экспоненциальная задержка (`RETRY_DELAY * 2^attempt + random jitter`), для rate-limit — `RETRY_DELAY * 1.5`. Лимит попыток авто-продолжения — `min(MAX_RETRIES, 2)` (не отдельная env-переменная).
- После `tools` без открытой ошибки выполнение возвращается в `summarize`, затем в `update_step` и `agent`. После `recovery` исходы `recover_agent` и `continue_agent` проходят тем же путём; остальные исходы завершают выполнение. Таким образом, порог контекста проверяется и внутри одного запроса, после результатов инструментов, а не только в начале хода. Ниже порога узел суммаризации не изменяет историю.
- `agent.py` сначала загружает `ToolRegistry` и MCP, создаёт checkpoint runtime и run logger, затем создаёт provider adapter через `core/providers/factory.py`. Инструменты привязываются к LLM только после нормализации схем; при ошибке binding tool calling отключается для текущего runtime.
- `ToolRegistry` объединяет встроенные tools и MCP tools, применяет фиче-флаги, сохранённые overrides из `mcp.json` и metadata риска. `read_only` tools могут выполняться mixed-mode batch параллельно, остальные tools идут последовательно.
- Для OpenAI-compatible профилей reasoning kwargs выбираются через `provider_registry.json` по hostname `base_url` и, при необходимости, по имени модели. Нативные адаптеры Gemini и Anthropic используют собственные provider-specific настройки.

## Параллельное выполнение инструментов

- `ToolBatchCoordinator` выполняет соседние parallel-safe вызовы ограниченным пулом asyncio-задач. `MAX_PARALLEL_TOOL_CALLS` по умолчанию равен `4`; `1` отключает параллельность. Лимит действует на batch, а не на все сессии или внутренние сетевые подзапросы инструментов.
- Зарегистрированные read-only инструменты, включая MCP с metadata из `ToolRegistry`, допускаются без отдельного списка имён. Для встроенных инструментов сохранён fallback-список; неизвестные инструменты без явных metadata не распараллеливаются. `request_user_input` всегда последовательный.
- Последовательный вызов — барьер: дожидается предыдущей параллельной группы и завершается до запуска следующей. `cli_exec` сохраняет существующее исключение и может выполняться параллельно даже с mutating metadata; независимость команд обязан обеспечить вызывающий агент. Approval-проверки не обходятся.
- Результаты поступают в stream сразу после завершения каждого вызова, а в `messages` собираются по исходным позициям. Каждый parallel-safe вызов получает отдельный task/context, в том числе после освобождения слота.
- Обычная ошибка одного параллельного вызова превращается в `ToolMessage`, не отменяя соседей. Отмена batch, отмена дочерней задачи и `GraphBubbleUp` (включая `GraphInterrupt`) останавливают запуск очереди; незавершённые задачи отменяются и ожидаются перед выходом. Управляющее исключение передаётся графу без упаковки в ошибку инструмента.
- Отмена asyncio кооперативна: она не откатывает уже совершённые действия, не гарантирует остановку удалённого MCP-запроса или синхронного кода в executor. Возобновление после interrupt может повторить вызовы из узла; это не гарантия exactly-once.

При анализе использованы официальные документы через Context7: [asyncio tasks](https://docs.python.org/3/library/asyncio-task.html) и [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts). Поведение проверено на локальных Python 3.14.6, LangGraph 1.2.11 и langchain-core 1.6.0.

## Runtime Lifecycle

```text
main.py
  → MainWindow / AgentRuntimeController
  → AgentRunWorker
  → build_agent_app()
     → AgentConfig (.env)
     → ToolRegistry (built-in + MCP)
     → checkpoint runtime (SQLite или memory)
     → provider adapter и model profile
     → LangGraph application
  → stream/status events
  → session snapshot, checkpoints и JSONL run log
```

`AgentRuntimeController` отделяет Qt UI-поток от асинхронного выполнения графа. Interrupts для approval и user choice возобновляются через runtime session coordination; stream errors классифицируются до retry/continue flow. Текущая версия приложения задаётся единственным источником `core/constants.py` (`AGENT_VERSION`) и отображается в заголовке окна и runtime payload.

---

## Prompt Layers

Промпт собирается из нескольких слоёв при каждом вызове агента:

| Слой | Файл / модуль | Содержимое |
|---|---|---|
| Базовый | `prompt.txt` | Системный промпт агента |
| Runtime | `core/runtime_prompt_policy.py` | OS, shell, workspace, дата, tool policy |
| Safety | `core/context_builder.py` | Workspace boundary, shell warning |
| Recovery | `core/recovery_manager.py` | Инструкции при активной ошибке |
| Memory | state: `summary` | Автосуммаризованный контекст прошлых ходов |

---

## Сессии и Checkpoints

- Graph checkpoints: `sqlite` (по умолчанию) или `memory`
- `.agent_state/checkpoints.sqlite` — durable checkpoint store
- `.agent_state/session.json` — активная сессия
- `.agent_state/session_index.json` — индекс всех сессий
- `logs/runs/` — JSONL-логи каждого запуска

В состоянии графа хранятся два списка сообщений: `messages` (контекст LLM, сжимается автосуммаризацией) и `transcript_messages` (полная append-only история переписки для UI). Compaction удаляет сообщения только из `messages`; новые assistant/tool/error-сообщения пишутся в оба списка с дедупликацией по ID. Legacy-сессии без `transcript_messages` получают bootstrap из текущих `messages` при первом сжатии.


При открытии сессии runtime формирует transcript payload из сохранённой истории. `ChatTranscriptWidget` группирует его в turns и создаёт виджеты последних `HISTORY_BATCH_SIZE` ходов, оставляя ранние данные для кнопки подгрузки. Это оптимизация рендеринга, не pagination SQLite. `SidebarController` откладывает замену старого transcript до получения истории выбранной сессии.
