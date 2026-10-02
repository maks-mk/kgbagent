# Руководство по установке MCP-серверов в Windows

Этот документ описывает подключение внешних MCP-серверов к проекту через `mcp.json`.

Важно:

- MCP-серверы являются сторонними процессами. Их код и набор возможностей проверяйте по официальному репозиторию или странице пакета перед установкой.
- Примеры ниже являются фрагментами корневого объекта `mcp.json`. Не заменяйте ими весь файл: в существующем конфиге могут находиться другие MCP-серверы и секция `_builtin_tools`.
- Сервер загружается только при `"enabled": true`. Если поле отсутствует, загрузчик считает сервер включённым.
- В конфигурацию MCP передаются только поля `command`, `args`, `env`, `cwd`, `encoding`, `encoding_error_handler`, `url`, `headers`, `timeout`, `sse_read_timeout`, `auth`, `terminate_on_close`, `httpx_client_factory`, `transport`, `session_kwargs`. `enabled`, `policy` и `policy.tools` обрабатываются самим приложением и не передаются серверу.

## Предварительные требования

### Node.js

Node.js LTS нужен для серверов, запускаемых через `npx`, и для npm-пакетов.

Установка: [nodejs.org](https://nodejs.org/)

Проверка:

```powershell
node --version
npm --version
```

### uv

`uv` нужен для Python-серверов, запускаемых через `uvx` или устанавливаемых командой `uv tool install`.

Официальная установка для Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Проверка:

```powershell
uv --version
```

После установки перезапустите PowerShell, если команда `uv` ещё не появилась в `PATH`.

## Настройка `mcp.json`

Путь к конфигу задаётся переменной `MCP_CONFIG_PATH`; по умолчанию используется `mcp.json` в корне проекта. Сначала откройте существующий файл и добавьте новый объект рядом с уже имеющимися объектами серверов.

**Важные особенности путей и сессий:**

- **Разрешение путей**: Относительный путь в `command` (например, `./bin/server.exe`) и относительный `cwd` разрешаются относительно каталога, в котором находится `mcp.json`. Это гарантирует работу серверов даже при смене рабочего каталога агентом. Команды из `PATH` (`python`, `uv`, `npx`) остаются без изменений.
- **Постоянные сессии**: Для каждого stdio-сервера создаётся один долгоживущий процесс и MCP-сеанс на всё время работы runtime. Состояние сервера (например, открытые SSH-сессии) сохраняется между вызовами инструментов.
- **Очистка**: При завершении работы runtime или пересоздании реестра инструментов все MCP-процессы корректно завершаются.

Подробности см. в [docs/MCP.md](./MCP.md).

Текущий шаблон `mcp.json` выглядит так:

```json
{
  "context7": {
    "type": "remote",
    "url": "https://mcp.context7.com/mcp",
    "transport": "http",
    "enabled": true,
    "policy": {
      "read_only": true
    }
  },
  "sequential-thinking": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
    "transport": "stdio",
    "enabled": false,
    "policy": {
      "read_only": true
    }
  },
  "_builtin_tools": {
    "download_file": false,
    "list_directory": false,
    "batch_web_search": false,
    "fetch_content": false
  }
}
```

После добавления сервера корневой объект должен сохранить существующие записи и их policy:

```json
{
  "context7": {
    "type": "remote",
    "url": "https://mcp.context7.com/mcp",
    "transport": "http",
    "enabled": true,
    "policy": {
      "read_only": true
    }
  },
  "sequential-thinking": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
    "transport": "stdio",
    "enabled": false,
    "policy": {
      "read_only": true
    }
  },
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
    "transport": "stdio",
    "enabled": false
  },
  "_builtin_tools": {
    "download_file": false,
    "list_directory": false,
    "batch_web_search": false,
    "fetch_content": false
  }
}
```

Значение `.` в аргументах filesystem-сервера задаёт рабочую директорию, доступную серверу. Используйте вместо него минимально необходимый абсолютный путь, если сервер должен работать не из каталога проекта.

## Запуск через `npx` или `uvx`

`npx -y` и `uvx` могут скачать пакет при первом запуске. Это удобно для локальной проверки, но требует доступа к сети и доверия к указанной версии пакета. Для воспроизводимой настройки фиксируйте версию пакета, например:

```json
{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem@<VERSION>", "."],
    "transport": "stdio",
    "enabled": false
  }
}
```

Не включайте сервер, пока не проверили пакет и его аргументы отдельно. Для проверки npm-пакета используйте его официальную документацию и, при необходимости:

```powershell
npm view @modelcontextprotocol/server-filesystem name version bin engines
```

Известные npm-пакеты из примеров проекта имеют разные имена исполняемых файлов:

- `@modelcontextprotocol/server-filesystem` предоставляет команду `mcp-server-filesystem`.
- `@simonb97/server-win-cli` предоставляет команду `server-win-cli`.
- `mcp-cli-exec` предоставляет команду `mcp-cli-exec`.

Это не означает, что любой пакет из старой или сторонней инструкции установлен или безопасен. Пакет `mcp-server-fetch` в этой инструкции не используется: его имя и способ запуска не подтверждены контрактом проекта.

Пример подключения проверенного npm-пакета через `npx`:

```json
{
  "win-cli": {
    "command": "npx",
    "args": ["-y", "@simonb97/server-win-cli@<VERSION>", "--config", "win-cli-config.json"],
    "transport": "stdio",
    "enabled": false
  }
}
```

Для Python-пакета сначала убедитесь, что он действительно существует и поддерживает MCP stdio-сервер. Только после этого добавляйте его команду `uvx` в конфиг:

```json
{
  "python-server": {
    "command": "uvx",
    "args": ["<PACKAGE_NAME>==<VERSION>"],
    "transport": "stdio",
    "enabled": false
  }
}
```

## Глобальная установка

Глобальная установка не обязательна. Она может уменьшить задержку запуска, но требует самостоятельного контроля версий и доступности команд в `PATH`.

Для npm-пакетов:

```powershell
npm install --global @modelcontextprotocol/server-filesystem@<VERSION>
npm install --global @simonb97/server-win-cli@<VERSION>
npm install --global mcp-cli-exec@<VERSION>
```

Проверить расположение глобальных npm-команд можно так:

```powershell
npm bin --global
where.exe mcp-server-filesystem
where.exe server-win-cli
where.exe mcp-cli-exec
```

В Windows глобальные npm-команды обычно доступны как `.cmd`. Если Python `subprocess` не находит команду, укажите полный путь к соответствующему `.cmd`-файлу в `command`, например:

```json
{
  "filesystem": {
    "command": "C:\\Users\\<USER>\\AppData\\Roaming\\npm\\mcp-server-filesystem.cmd",
    "args": ["."],
    "transport": "stdio",
    "enabled": false
  }
}
```

Для Python-инструментов не используйте `uv tool install` вслепую. Сначала проверьте официальное имя пакета и команду запуска. Затем установите зафиксированную версию:

```powershell
uv tool install <PACKAGE_NAME>==<VERSION>
```

После установки проверьте команду:

```powershell
where.exe <COMMAND_NAME>
<COMMAND_NAME> --help
```

## Переменные окружения и секреты

Загрузчик проекта рекурсивно раскрывает переменные окружения во всех строковых значениях конфигурации, включая `env`, `args`, `headers`, `url` и `cwd`. В Windows можно использовать синтаксис `%NAME%`:

```json
{
  "remote-server": {
    "url": "%MCP_SERVER_URL%",
    "headers": {
      "Authorization": "Bearer %MCP_SERVER_TOKEN%"
    },
    "transport": "http",
    "enabled": false
  }
}
```

Перед запуском задайте переменные в текущем сеансе PowerShell:

```powershell
$env:MCP_SERVER_URL = "https://example.invalid/mcp"
$env:MCP_SERVER_TOKEN = "replace-me"
```

Не записывайте реальные токены, API-ключи и пароли в `mcp.json`, документацию или коммиты. Переменные окружения защищают секрет от попадания в сам JSON, но значение всё равно может быть доступно процессу MCP и появиться в логах стороннего сервера. Проверьте `.gitignore` и права доступа к окружению.

## Политики approval приложения

Секция `policy` не является sandbox и не ограничивает возможности внешнего MCP-сервера. Она используется проектом после загрузки инструментов для локальной классификации риска и определения необходимости approval.

`read_only: true` допустимо у сервера только тогда, когда все его инструменты действительно не изменяют состояние, не удаляют данные и не выполняют команды. Для сервера, который может выполнять shell-команды, изменять файлы или вызывать внешние действия, не указывайте `read_only: true`.

Пример для действительно read-only сервера:

```json
{
  "read-only-server": {
    "command": "<COMMAND>",
    "args": ["<ARGUMENT>"],
    "transport": "stdio",
    "enabled": false,
    "policy": {
      "read_only": true
    }
  }
}
```

Для потенциально изменяющего состояние сервера оставьте `policy` без `read_only` и включите approval в настройках приложения. Для точечной классификации поддерживается `policy.tools` с именами инструментов, возвращаемыми сервером:

```json
{
  "server": {
    "command": "<COMMAND>",
    "transport": "stdio",
    "enabled": false,
    "policy": {
      "tools": {
        "tool_name": {
          "read_only": true
        }
      }
    }
  }
}
```

Такая запись меняет только metadata и approval-логику проекта; она не запрещает инструменту выполнить операцию. Для реального ограничения доступа используйте настройки и sandbox самого MCP-сервера, операционной системы и отдельной учётной записи.

## Проверка и диагностика

1. Проверьте JSON до запуска:

   ```powershell
   Get-Content .\mcp.json -Raw | ConvertFrom-Json | Out-Null
   ```

2. Проверьте команды серверов отдельно, используя `--help`, если это поддерживается.
3. Убедитесь, что `command` находится в `PATH`, либо укажите полный путь к исполняемому файлу.
4. Для stdio-сервера убедитесь, что команда не пишет служебный текст в stdout: stdout используется MCP-протоколом.
5. Посмотрите статус загрузки MCP-сервера и логи приложения. Ошибка подключения одного сервера не подтверждает корректность другого.

Ошибка `spawn ENOENT` обычно означает, что команда не найдена, неверно указан путь или процесс запускается с другим `PATH`. Исправьте окружение и перезапустите приложение.

Если сервер не нужен, оставьте `"enabled": false` или удалите только его объект, сохранив остальные записи `mcp.json`.
