# Названия инструментов в UI

Название действия меняется после успешного завершения:

- во время выполнения: `Reading`, `Writing`, `Editing`, `Running`;
- после успешного завершения: `Read`, `Wrote`, `Edited`, `Ran`;
- при ошибке: `Reading failed`, `Writing failed`, `Editing failed`, `Running failed`.

Такой переход показывает состояние без потери смысла действия. Для групп инструментов с известным объектом используется количество:

- `Editing 2 files` → `Edited 2 files`;
- `Running 2 commands` → `Ran 2 commands`;
- `Reading 1 file` → `Read 1 file`.

Считаются объекты, а не вызовы: один вызов `read_images` с двумя изображениями даёт `Reading 2 images` → `Read 2 images`. Для одного изображения используется `Reading 1 image` → `Read 1 image`.

Для групп смешанных или неизвестных инструментов используется нейтральное `Running N tools` → `Completed N tools`.

## Основные инструменты

- `read_file` → `Reading` → `Read`
  - пакетное чтение (`requests`) подписывает файлы: `Reading src/a.py, src/b.py` → `Read src/a.py, src/b.py`;
  - в группе считаются файлы, а не вызовы: два файла в одном пакете дают `Reading 2 files` → `Read 2 files`, повторные диапазоны одного файла считаются один раз;
  - если видимая строка обрезается, полный список путей остаётся в подсказке карточки.
- `read_images` → `Reading images` → `Read images` (при ошибке `Reading images failed`, до получения аргументов `Preparing image read`)
- `write_file` → `Writing` → `Wrote`
- `edit_file` → `Editing` → `Edited`
- `list_directory` → `Listing` → `Listed`
- `safe_delete_file` → `Deleting` → `Deleted`
- `safe_delete_directory` → `Deleting` → `Deleted`
- `download_file` → `Downloading` → `Downloaded`
- `batch_web_search` → `Searching` → `Searched`
- `fetch_content` → `Fetching` → `Fetched`
- `cli_exec` → `Running` → `Ran`
- `request_user_input` → `Requesting input` → `Requested input`
- `read_skills` → `Reading Skills` → `Read Skills` (при ошибке `Reading Skills failed`)

## Дополнительная информация

Рядом с коротким названием UI может показывать объект действия:

- `Reading config.py` → `Read config.py`
- `Writing report.md +20 -0` → `Wrote report.md +20 -0`
- `Editing app.py +3 -1` → `Edited app.py +3 -1`
- `Running python -m pytest` → `Ran python -m pytest`
- `Searching Python asyncio` → `Searched Python asyncio`
- `Reading Skills pdf, design` → `Read Skills pdf, design`
- `Reading images img/one.png, img/two.png` → `Read images img/one.png, img/two.png`
- `Reading src/a.py, src/b.py` → `Read src/a.py, src/b.py` (пакетное чтение `read_file` с `requests`)

Если инструмент принимает несколько путей, показываются все запрошенные пути через запятую, а не только первый.

Путь, команда, запрос, URL и статистика изменений не являются частью названия инструмента.

## Совместимые псевдонимы

UI распознаёт альтернативные имена инструментов из других провайдеров:

- `Read` → `Reading` → `Read`
- `Write` → `Writing` → `Wrote`
- `SearchReplace` → `Editing` → `Edited`
- `ls`, `LS` → `Listing` → `Listed`
- `grep`, `Grep`, `glob`, `Glob` → `Searching` → `Searched`
- `execute`, `RunCommand` → `Running` → `Ran`
- `fetch_url`, `WebFetch` → `Fetching` → `Fetched`

## MCP-инструменты

Для MCP-инструментов сохраняется динамическое название, поскольку оно помогает отличать сервер и конкретное действие:

- выполняется: `<Server>: <Tool>`
- завершён: `<Server>: <Tool> completed`
- ошибка: `<Server>: <Tool> failed`

Имена сервера и инструмента преобразуются в читаемый вид: `:`, `-` и `_` заменяются пробелами, каждое слово начинается с заглавной буквы. Если сервер неизвестен, используется `MCP`.

## Неизвестные инструменты

Для инструмента без специального правила внутреннее имя преобразуется в заголовок: `_` и `:` заменяются пробелами, первые буквы слов переводятся в верхний регистр.

Например, `custom_tool` отображается как `Custom Tool`, при ошибке — `Tool failed`. Для неизвестного инструмента отдельная завершённая форма не создаётся, поскольку UI не может надёжно определить смысл действия.
