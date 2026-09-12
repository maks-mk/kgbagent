# KGB|Agent

**Knowledge & Goal-Based Agent**

[English README](./README_EN.md)

> *"Created by a SysAdmin for developers. Focus on safety, portability, and zero-nonsense execution. No Docker, no heavy environments, just one binary."*

Десктопный AI-агент с runtime на `LangGraph` и графическим интерфейсом на `PySide6`.
Работает с файлами, shell-командами, процессами, MCP-серверами и веб-поиском.

Запуск из исходников: `python main.py`  
Сборка в portable `.exe` для Windows: `build.bat`

---

## Цель проекта

Это не AI IDE. Цель — предоставить переносимого автономного помощника, которого можно скопировать на другой компьютер и сразу использовать с минимальной настройкой.

Основные приоритеты:

- переносимость;
- безопасность;
- локальные инструменты;
- автоматизация;
- работа с файлами и локальными процессами;
- поиск информации в интернете;
- надёжность.

Проект не стремится конкурировать с AI IDE по количеству функций и не пытается заменить специализированные инструменты для разработки. Фокус — практическое выполнение повседневных задач: работа с файлами, shell-командами, локальными процессами, веб-поиском, документацией, скриптами и автоматизацией в одном переносимом приложении без сложной инфраструктуры и дополнительных сервисов.

---

## Возможности

- Графовый runtime на `LangGraph` с bounded recovery и self-correction
- Mixed-mode parallel tool batch: зарегистрированные read-only инструменты (включая MCP) и `cli_exec` выполняются ограниченным пулом (`MAX_PARALLEL_TOOL_CALLS=4`); остальные вызовы служат последовательными барьерами, результаты сохраняют исходный порядок
- GUI: безрамочные окна, проекты и чаты в левой панели, пакетное отображение длинной истории, streaming transcript, tool cards, approvals, user-choice карточки, вложения
- Настройка порога контекста Session size в GUI; проверка автосуммаризации также после выполнения инструментов
- Fuzzy replay suppression: повторный префейс модели после tool-вызова подавляется даже при минимальных расхождениях текста (опечатки, пунктуация)
- Live CLI output streaming: вывод shell-команд отображается в карточке инструмента в реальном времени, а не только после завершения
- Exit-code-neutral команды: `grep`, `rg`, `vulture`, `pytest`, `diff` и др. с ненулевым exit code не помечаются как ошибка — вывод возвращается с префиксом `Exit Code: N`
- Stream-interruption recovery с классификацией ошибок (`rate_limit` / `timeout` / `server_error` / `network`) и экспоненциальным backoff с джиттером перед авто-продолжением
- Инструменты: filesystem (включая `download_file`), shell, Tavily web search/fetch, process management, MCP
- Approval-паузы перед мутирующими и деструктивными действиями
- Автосуммаризация контекста при длинных сессиях
- Настраиваемые HTTP-заголовки для OpenAI-compatible и Anthropic LLM через `headers.json` (эмуляция совместимых клиентов и прокси)
- Несколько профилей моделей с переключением прямо в GUI
- Durable checkpoints — сессии сохраняются между запусками
- Опциональный image input, если модель его поддерживает

---

## Быстрый старт

Требования: **Python 3.10+**, API-ключ Gemini, OpenAI или Anthropic. Web search/fetch — опционально: для них нужны пакет `tavily-python` из `requirements.txt` и `TAVILY_API_KEY` в `.env`.

```powershell
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt
Copy-Item env_example.txt .env
# Открой .env и укажи API-ключ выбранного LLM-провайдера
# Для Tavily-поиска также укажи TAVILY_API_KEY
venv\Scripts\python.exe main.py
```

Для Claude 4.6+ используйте adaptive thinking с effort (`ANTHROPIC_REASONING=low|medium|high|max`, а `xhigh` только для поддерживаемых моделей 4.7+/5). Для Claude 4.5 используется `ANTHROPIC_THINKING_BUDGET`; Opus 4.5 дополнительно поддерживает effort `low|medium|high`. При включённом thinking `temperature` не отправляется согласно ограничениям Anthropic. Полный список параметров находится в [документации конфигурации](./docs/CONFIGURATION.md).

---

## Portable сборка

```powershell
.\build.bat
```

Использует локальный `venv` и `PyInstaller` в режиме `--onefile --windowed`; результат — `dist/kgb.exe`. Python на целевой машине не требуется, но конфигурация и данные остаются внешними: рядом с `.exe` разместите `prompt.txt`, свою `.env`, а при использовании — `mcp.json`, `provider_registry.json` и `headers.json`. Каталог должен быть доступен для записи состояния и логов. Для локальных MCP-серверов отдельно нужны их runtime и пакеты (например, Node.js/npx или uv/uvx); они не включаются в `.exe`.

---

## Архитектура

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

Подробнее: [Runtime Flow, Prompt Layers, Sessions & Checkpoints](./docs/ARCHITECTURE.md)

---

## Структура проекта

```text
.
├── main.py                # Точка входа GUI
├── agent.py               # Сборка графа LangGraph: routing, tool binding, checkpointing
├── prompt.txt             # Основной системный промпт
├── mcp.json               # Конфигурация MCP-серверов
├── env_example.txt        # Шаблон .env
├── provider_registry.json # Reasoning kwargs для OpenAI-compatible провайдеров
├── build.bat              # Сборка portable .exe
├── requirements.txt
├── core/                  # Ядро агента: config, state, policies, recovery, provider registry
│   ├── nodes/             # Узлы LangGraph: context, llm, agent, tools, approval, recovery
│   └── providers/         # Provider-адаптеры (Anthropic, Gemini, OpenAI-compatible)
├── tools/                 # Filesystem/download, shell, search, process, user input, MCP registry
├── ui/                    # PySide6 GUI, runtime worker, streaming/status handling
├── docs/                  # Документация
├── tests/                 # Runtime, UI, tools, provider registry, logging, policies
├── .agent_state/          # Локальное состояние, профили, checkpoints
└── logs/                  # JSONL/runtime/debug logs
```

Полная карта модулей: [`docs/PROJECT_STRUCTURE.md`](./docs/PROJECT_STRUCTURE.md)

---

## Тесты

Запустите полный regression-набор:

```powershell
venv\Scripts\python.exe -m pytest
```

---

## Зависимости

### ripgrep (`rg`) — рекомендуется

Для более эффективного поиска по файлам, логам, конфигурациям и кодовым базам рекомендуется установить [`ripgrep`](https://github.com/BurntSushi/ripgrep). Готовые сборки для Windows — в [Releases](https://github.com/BurntSushi/ripgrep/releases) (архив `x86_64-pc-windows-msvc.zip`).

Для portable-сборки скопируйте `rg.exe` рядом с исполняемым файлом агента. Если `rg` отсутствует, агент продолжит работать в обычном режиме, используя стандартные инструменты файловой системы.

### Python-пакеты

| Пакет | Назначение |
|---|---|
| `langgraph` | Граф агента и state management |
| `langchain` | LLM abstraction, tool calling |
| `langchain-google-genai` | Gemini provider |
| `langchain-openai` | OpenAI / compatible provider |
| `langchain-mcp-adapters` | MCP интеграция |
| `PySide6` | GUI |
| `pydantic-settings` | Конфигурация через `.env` |
| `tiktoken` | Подсчёт токенов для суммаризации |
| `tavily-python` | Web search |
| `psutil` | Управление процессами |
| `httpx` | HTTP для загрузки файлов, model discovery, MCP и web fetch |
| `aiofiles` | Async файловые операции |
| `aiosqlite` | Async SQLite для checkpointing |
| `mcp` | Model Context Protocol |
| `requests` | HTTP-клиент (Google API, Tavily) |
| `QtAwesome` | Иконки для GUI |
| `sqlite-vec` | Vector-расширение для SQLite checkpoints |

---

## Документация

| Документ | Содержание |
|---|---|
| [Архитектура](./docs/ARCHITECTURE.md) | Runtime Flow, Prompt Layers, Sessions & Checkpoints |
| [Конфигурация](./docs/CONFIGURATION.md) | Все переменные `.env` (провайдеры, runtime, фиче-флаги, лимиты, retry, персистентность, диагностика), HTTP-заголовки `headers.json` для provider-запросов |
| [GUI](./docs/GUI_GUIDE.md) | Transcript, CLI output widget, Composer, горячие клавиши |
| [Безопасность](./docs/SECURITY.md) | Approvals, workspace boundary, `request_user_input` |
| [Профили моделей](./docs/MODEL_PROFILES.md) | Управление профилями, автозагрузка моделей, ротация API-ключей |
| [MCP](./docs/MCP.md) | Конфигурация MCP-серверов, policy, пример |
| [Структура проекта](./docs/PROJECT_STRUCTURE.md) | Полная карта модулей |
| [Provider Registry](./docs/provider_registry_guide.md) | Добавление OpenAI-compatible агрегаторов |
