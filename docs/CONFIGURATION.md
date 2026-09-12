# Конфигурация

Все настройки читаются из `.env` через `core/config.py`. Скопируй `env_example.txt` в `.env` и заполни нужные поля.

---

## Провайдер и модели

| Переменная | По умолчанию | Описание |
|---|---|---|
| `PROVIDER` | `gemini` | `gemini`, `openai` или `anthropic` |
| `GEMINI_API_KEY` | — | Обязателен для Gemini |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Имя модели Gemini |
| `OPENAI_API_KEY` | — | Обязателен для OpenAI (если нет `OPENAI_BASE_URL`) |
| `OPENAI_MODEL` | `gpt-4o` | Имя модели OpenAI |
| `OPENAI_BASE_URL` | — | Для OpenAI-compatible бэкендов (Ollama и др.) |
| `ANTHROPIC_API_KEY` | — | API-ключ Anthropic |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5-20250929` | Имя модели Anthropic |
| `ANTHROPIC_BASE_URL` | — | Необязательный URL Anthropic-compatible API (без `/v1` — SDK добавляет его сам) |
| `ANTHROPIC_MAX_TOKENS` | `8192` | Максимум выходных токенов Anthropic |
| `ANTHROPIC_THINKING_BUDGET` | `4096` | Fixed-budget thinking для Claude Haiku/Sonnet/Opus 4.5; для Opus 4.5 может использоваться вместе с effort `low`, `medium` или `high` |
| `ANTHROPIC_REASONING` | — | Управление Anthropic reasoning: `off`/`none`, `adaptive` или effort. Для Opus 4.5 доступны `low`, `medium`, `high`; для Claude 4.6+ и 5 набор зависит от модели, включая `max`, а `xhigh` — только для поддерживаемых моделей 4.7+/5. Для effort runtime передаёт adaptive `thinking` и `output_config.effort`. При включённом thinking sampling-параметр `temperature` не передаётся. |
| `LLM_API_MODE` | `chat` | Режим API для OpenAI-провайдера: `chat` (`/v1/chat/completions`, работает со всеми OpenAI-compatible) или `responses` (`/v1/responses`, для gpt-5/o-series reasoning). Если не указан, LangChain автоопределяет по модели и payload |
| `ENABLE_MODEL_REASONING` | `true` | Включает provider-side reasoning/thinking для поддерживаемых моделей |
| `MODEL_REASONING_EFFORT` | `medium` | Усилие reasoning для OpenAI/OpenAI-compatible моделей (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`) |
| `GEMINI_THINKING_BUDGET` | `4096` | Thinking budget для `gemini-2.5*` / `gemini-3*`; старые Gemini-модели получают запрос без этого параметра |
| `PROVIDER_REGISTRY_PATH` | `provider_registry.json` | Реестр OpenAI-compatible агрегаторов и их reasoning-параметров |
| `ACTIVE_MODEL_PROFILE_ID` | — | ID активного профиля модели (для ротации ключей) |
| `SHOW_MODEL_THOUGHTS` | `false` | Legacy-флаг отображения reasoning (runtime выставляет false) |

### Добавление OpenAI-compatible агрегаторов

Подробная инструкция по `provider_registry.json`, полям схемы, matching rules и добавлению новых агрегаторов находится в [`provider_registry_guide.md`](./provider_registry_guide.md).

---

## Управление runtime

| Переменная | Описание |
|---|---|
| `TEMPERATURE` | Температура модели |
| `MAX_LOOPS` | Максимум шагов на один запрос (default: 50) |
| `TOOL_LOOP_WINDOW` | Окно истории для детекции дублей tool calls |
| `TOOL_LOOP_LIMIT_MUTATING` | Лимит повторов для мутирующих инструментов |
| `TOOL_LOOP_LIMIT_READONLY` | Лимит повторов для read-only инструментов |
| `SELF_CORRECTION_RETRY_LIMIT` | Потолок попыток self-correction |

---

## Фиче-флаги

| Переменная | Описание |
|---|---|
| `MODEL_SUPPORTS_TOOLS` | Включить tool calling |
| `ENABLE_TEXT_TOOL_CALL_RECOVERY` | Диагностический fallback для провайдеров, которые пишут `call:...<tool_call|>` текстом вместо structured `tool_calls`; по умолчанию выключен |
| `ENABLE_FILESYSTEM_TOOLS` | Инструменты для работы с файлами |
| `ENABLE_SHELL_TOOL` | Shell-выполнение команд |
| `ENABLE_SEARCH_TOOLS` | Web search через Tavily |
| `ENABLE_PROCESS_TOOLS` | Управление процессами |
| `ENABLE_APPROVALS` | Approval-паузы перед рискованными действиями |
| `ALLOW_EXTERNAL_PROCESS_CONTROL` | Разрешить управление внешними процессами |
| `TAVILY_API_KEY` | Ключ Tavily для web search и извлечения содержимого |

### Tavily search tools

При `ENABLE_SEARCH_TOOLS=true` реестр подключает два read-only сетевых инструмента: `batch_web_search` и `fetch_content`. Для обоих требуется установленный пакет `tavily-python` и непустой `TAVILY_API_KEY`; при отсутствии ключа или SDK инструмент возвращает конфигурационную ошибку.

- `batch_web_search` принимает до 5 уникальных запросов за вызов и выполняет их параллельно. Допустимые `search_depth`: `basic`, `advanced`, `fast`, `ultra-fast`; допустимые `topic`: `general`, `news`, `finance`.
- `fetch_content` принимает 1–20 HTTP(S)-адресов и передаёт их в Tavily одним batch-запросом. Допустимые `content_format`: `markdown` и `text`; параметр `advanced=true` включает углублённое извлечение. `query` включает выбор релевантных фрагментов, а `chunks_per_source` задаёт от 1 до 5 фрагментов на страницу.
- `MAX_SEARCH_CHARS` ограничивает один поисковый подзапрос, а `MAX_RAW_TOOL_OUTPUT` — суммарный результат до сжатия и финального `MAX_TOOL_OUTPUT`; успешные результаты кешируются в памяти процесса.

---

## Лимиты

| Переменная | Описание |
|---|---|
| `MAX_FILE_SIZE` | Максимальный размер файла в байтах; строки могут содержать единицы `KB`, `MB`, `GB`, а также двоичные `KiB`, `MiB`, `GiB` (например, `300MiB`) |
| `MAX_READ_LINES` | Лимит строк при чтении файла |
| `MAX_TOOL_OUTPUT` | Финальный лимит символов в выводе инструмента для контекста модели |
| `MAX_RAW_TOOL_OUTPUT` | Лимит символов, собираемых до семантического сжатия (по умолчанию `100000`) |
| `MAX_SEARCH_CHARS` | Лимит символов одного поискового подзапроса |
| `MAX_BACKGROUND_PROCESSES` | Лимит фоновых процессов |
| `STREAM_TEXT_MAX_CHARS` | Лимит символов streaming-текста |
| `STREAM_EVENTS_MAX` | Лимит streaming-событий |
| `STREAM_TOOL_BUFFER_MAX` | Буфер streaming tool output |

---

## Суммаризация и retry

| Переменная | Описание |
|---|---|
| `SESSION_SIZE` | Порог оценки контекста (токены) для запуска суммаризации; `0` — автосуммаризация выключена. Индикатор в GUI показывает процент, оставшийся именно до этого порога |
| `SUMMARY_RESERVED_TOKENS` | Запас на системные инструкции, tool schemas и provider overhead |
| `SUMMARY_KEEP_LAST` | Сколько последних сообщений не трогать при суммаризации |
| `SUMMARY_MAX_TOKENS` | Лимит токенов сжатой памяти; `0` — четверть от `SESSION_SIZE` |
| `HISTORY_BATCH_SIZE` | Сколько сообщений (turns) GUI подгружает за один раз при прокрутке длинной истории; по умолчанию 10, диапазон 1–200 |
| `MAX_RETRIES` | Число попыток при ошибке LLM |
| `RETRY_DELAY` | Базовая задержка между попытками (секунды); также используется как base delay для stream-repair backoff |

### Как работает порог

Оценка контекста считается как токены истории сообщений (tiktoken `cl100k_base`, при недоступности — эвристика ~3 символа/токен) плюс `SUMMARY_RESERVED_TOKENS` и токены сжатой памяти. Когда оценка превышает `SESSION_SIZE`, выполняется автосуммаризация. Внутренний soft-margin (до +35% при уже существующей памяти) может дополнительно задержать сжатие, если сохранять слишком мало сообщений — но пользовательский индикатор прогресса всегда считается от базового `SESSION_SIZE`. Полная история переписки при этом сохраняется и после сжатия.

---

## HTTP-заголовки для provider-запросов

Для эмуляции совместимого клиента или добавления пользовательских заголовков в запросы к OpenAI-compatible и Anthropic LLM используется файл `headers.json` в корне проекта.

**Поведение:**
- Файл отсутствует → provider SDK отправляет свои стандартные заголовки.
- Файл присутствует → его строковые значения переопределяют/добавляют заголовки SDK. Битый или не-объектный JSON игнорируется (возвращается пустой словарь).

**Пример `headers.json` (эмуляция OpenAI-compatible клиента):**
```json
{
  "User-Agent": "QwenCode/0.12.6 (win32; x64)",
  "x-stainless-lang": "js",
  "x-stainless-package-version": "5.11.0",
  "x-stainless-os": "Windows",
  "x-stainless-arch": "x64",
  "x-stainless-runtime": "node",
  "x-stainless-runtime-version": "v24.3.0",
  "accept-language": "*",
  "sec-fetch-mode": "cors"
}
```

**Применение:**
- Заголовки используются в `core/model_fetcher.py` для OpenAI-compatible model discovery, а также в `core/providers/openai_reasoning.py` и `core/providers/anthropic.py` для LLM-запросов.
- Для Anthropic model discovery используются обязательные API-заголовки, но пользовательские значения из `headers.json` не добавляются.
- Для Gemini-запросов заголовки не применяются.

**Модуль:** `core/http_headers.py` — загрузчик `load_provider_headers()`; `load_openai_headers()` сохранён как обратно совместимый alias.

---

## Персистентность

| Переменная | По умолчанию | Описание |
|---|---|---|
| `CHECKPOINT_BACKEND` | `sqlite` | `sqlite` или `memory` |
| `CHECKPOINT_SQLITE_PATH` | `.agent_state/checkpoints.sqlite` | Путь к БД |
| `SESSION_STATE_PATH` | `.agent_state/session.json` | Активная сессия |
| `MODEL_PROFILE_CONFIG_PATH` | `.agent_state/config.json` | Файл профилей моделей и активного профиля |
| `RUN_LOG_DIR` | `logs/runs` | Директория JSONL-логов |
| `LOG_FILE` | `logs/agent.log` | Файл лога |
| `PROMPT_PATH` | `prompt.txt` | Путь к системному промпту |
| `MCP_CONFIG_PATH` | `mcp.json` | Путь к конфигу MCP |

---

## Диагностика

| Переменная | Описание |
|---|---|
| `DEBUG` | Включить debug-режим |
| `LOG_LEVEL` | Уровень логирования (`INFO`, `DEBUG`, `WARNING`) |
| `DEBUG_REASONING_STREAM` | Отдельный подробный лог reasoning/thinking stream для диагностики провайдеров |
| `STRICT_MODE` | Строгий режим: без догадок, точное выполнение |
