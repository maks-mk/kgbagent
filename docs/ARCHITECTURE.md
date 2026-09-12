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
