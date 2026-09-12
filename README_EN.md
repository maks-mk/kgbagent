# KGB|Agent

**Knowledge & Goal-Based Agent**

[Russian README](./README.md)

> *"Created by a SysAdmin for developers. Focus on safety, portability, and zero-nonsense execution. No Docker, no heavy environments, just one binary."*

A desktop AI agent with a `LangGraph` runtime and a `PySide6` GUI.  
It works with files, shell commands, process management, MCP servers, and web search.

Run from source: `python main.py`.  
Build a portable Windows `.exe`: `build.bat`.

---

## Project Goal

This is not an AI IDE. The goal is to provide a portable autonomous assistant that can be copied to another computer and used immediately with minimal setup.

Main priorities:

- portability;
- safety;
- local tools;
- automation;
- working with files and local processes;
- web search;
- reliability.

The project does not try to compete with AI IDEs by feature count and does not try to replace specialized development tools. Its focus is practical execution of everyday tasks: working with files, shell commands, local processes, web search, documentation, scripts, and automation in one portable application without complex infrastructure or additional services.

---

## Features

- Graph runtime on `LangGraph` with bounded recovery and self-correction
- Mixed-mode parallel tool batch: registered read-only tools (including MCP) and `cli_exec` use a bounded pool (`MAX_PARALLEL_TOOL_CALLS=4`); other calls act as sequential barriers, and results retain their original order
- GUI: frameless windows, projects and chats in the sidebar, batched rendering of long history, streaming transcript, tool cards, approvals, user-choice cards, attachments
- GUI Session size setting; automatic context compaction is also checked after tool execution
- Fuzzy replay suppression: the model's repeated preface after a tool call is suppressed even with minor text drift (typos, punctuation)
- Live CLI output streaming: shell command output is shown in the tool card in real time, not only after completion
- Exit-code-neutral commands: `grep`, `rg`, `vulture`, `pytest`, `diff`, etc. with a non-zero exit code are not marked as errors — the output is returned with an `Exit Code: N` prefix
- Stream-interruption recovery with error classification (`rate_limit` / `timeout` / `server_error` / `network`) and exponential backoff with jitter before auto-continue
- Tools: filesystem (including `download_file`), shell, Tavily web search/fetch, process management, MCP
- Approval pauses before mutating and destructive actions
- Automatic context summarization for long sessions
- Customizable HTTP headers for OpenAI-compatible and Anthropic LLM requests via `headers.json` (client/proxy emulation)
- Multiple model profiles with switching directly in the GUI
- Durable checkpoints: sessions persist between launches
- Optional image input when the selected model supports vision

---

## Quick Start

Requirements: **Python 3.10+**, plus a Gemini, OpenAI, or Anthropic API key. Web search/fetch is optional and requires the `tavily-python` package from `requirements.txt` plus `TAVILY_API_KEY` in `.env`.

```powershell
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt
Copy-Item env_example.txt .env
# Open .env and add the selected LLM provider key
# Add TAVILY_API_KEY as well when Tavily search/fetch is needed
venv\Scripts\python.exe main.py
```

---

## Portable Build

```powershell
.\build.bat
```

Uses the local `venv` and `PyInstaller` in `--onefile --windowed` mode, producing `dist/kgb.exe`. Python is not required on the target machine, but configuration remains external: place `prompt.txt`, your `.env`, and, when used, `mcp.json`, `provider_registry.json` and `headers.json` next to the executable. The directory must be writable for state and logs. Local MCP servers still require their own runtimes and packages (such as Node.js/npx or uv/uvx); these are not bundled.

---

## Architecture

```text
START
  -> summarize        # compact context if the session has grown large
  -> update_step
  -> agent            # LLM decides: answer / call tool / recover
     -> approval      # pause before a mutating action
        -> tools
     -> tools         # execute tool calls (read-only in parallel, the rest sequentially)
        -> recovery   # if a tool returned an error
        -> summarize -> update_step
     -> recovery      # if the agent returned a protocol error or loop
        -> summarize -> update_step
        -> END
     -> END
```

Details: [Runtime Flow, Prompt Layers, Sessions & Checkpoints](./docs/ARCHITECTURE.md)

---

## Project Structure

```text
.
|-- main.py                # GUI entry point
|-- agent.py               # LangGraph graph assembly: routing, tool binding, checkpointing
|-- prompt.txt             # Main system prompt
|-- mcp.json               # MCP server configuration
|-- env_example.txt        # .env template
|-- provider_registry.json # Reasoning kwargs for OpenAI-compatible providers
|-- build.bat              # Portable .exe build
|-- requirements.txt
|-- core/                  # Agent core: config, state, policies, recovery, provider registry
|   |-- nodes/             # LangGraph nodes: context, llm, agent, tools, approval, recovery
|   `-- providers/         # Provider adapters (Anthropic, Gemini, OpenAI-compatible)
|-- tools/                 # Filesystem/download, shell, search, process, user input, MCP registry
|-- ui/                    # PySide6 GUI, runtime worker, streaming/status handling
|-- docs/                  # Documentation
|-- tests/                 # Runtime, UI, tools, provider registry, logging, policies
|-- .agent_state/          # Local state, profiles, checkpoints
`-- logs/                  # JSONL/runtime/debug logs
```

Full module map: [`docs/PROJECT_STRUCTURE.md`](./docs/PROJECT_STRUCTURE.md)

---

## Tests

Run the full regression suite:

```powershell
venv\Scripts\python.exe -m pytest
```

---

## Dependencies

### ripgrep (`rg`) — recommended

For more efficient search through files, logs, configuration, and codebases, installing [`ripgrep`](https://github.com/BurntSushi/ripgrep) is recommended. Prebuilt Windows binaries are available in [Releases](https://github.com/BurntSushi/ripgrep/releases) (the `x86_64-pc-windows-msvc.zip` archive).

For a portable build, copy `rg.exe` next to the agent executable. If `rg` is not available, the agent continues to work normally with the standard filesystem tools.

### Python Packages

| Package | Purpose |
|---|---|
| `langgraph` | Agent graph and state management |
| `langchain` | LLM abstraction, tool calling |
| `langchain-google-genai` | Gemini provider |
| `langchain-openai` | OpenAI / compatible provider |
| `langchain-mcp-adapters` | MCP integration |
| `PySide6` | GUI |
| `pydantic-settings` | Configuration through `.env` |
| `tiktoken` | Token counting for summarization |
| `tavily-python` | Web search |
| `psutil` | Process management |
| `httpx` | HTTP for file downloads, model discovery, MCP, and web fetch |
| `aiofiles` | Async file operations |
| `aiosqlite` | Async SQLite for checkpointing |
| `mcp` | Model Context Protocol |
| `requests` | HTTP client for Google API and Tavily |
| `QtAwesome` | Icons for the GUI |
| `sqlite-vec` | Vector extension for SQLite checkpoints |

---

## Documentation

| Document | Contents |
|---|---|
| [Architecture](./docs/ARCHITECTURE.md) | Runtime Flow, Prompt Layers, Sessions & Checkpoints |
| [Configuration](./docs/CONFIGURATION.md) | All `.env` variables (providers, runtime, feature flags, limits, retry, persistence, diagnostics), provider request headers via `headers.json` |
| [GUI](./docs/GUI_GUIDE.md) | Transcript, CLI output widget, Composer, hotkeys |
| [Security](./docs/SECURITY.md) | Approvals, workspace boundary, `request_user_input` |
| [Model Profiles](./docs/MODEL_PROFILES.md) | Profile management, auto-loading models, API key rotation |
| [MCP](./docs/MCP.md) | MCP server configuration, policy, example |
| [Project Structure](./docs/PROJECT_STRUCTURE.md) | Full module map |
| [Provider Registry](./docs/provider_registry_guide.md) | Adding OpenAI-compatible aggregators |
