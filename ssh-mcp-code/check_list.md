Проведи полный аудит проекта `ssh-mcp` и сразу исправь найденные проблемы. Не ограничивайся анализом: если проблема подтверждается — внеси исправление, не ломая существующую функциональность.

Проверь и при необходимости исправь:

1. SECURITY

* Проверь `extra_ssh_args` и все способы обхода его текущей blocklist/валидации.
* Особое внимание: `-F`, `-oKnownHostsCommand`, `-oPKCS11Provider`, `-oSecurityKeyProvider` и любые другие SSH options, которые могут выполнить локальную команду, загрузить локальный provider/library, изменить SSH config, proxy или forwarding.
* Не используй хрупкий blacklist там, где безопаснее сделать allowlist.
* Проверь все пользовательские параметры, которые потенциально могут быть сформированы LLM и привести к выполнению команд на локальной машине вместо удалённой.
* Проверь path traversal, symlink issues, shell injection, command injection и небезопасное quoting во всех remote file tools.
* Проверь `ssh_create`, `ssh_edit`, `ssh_view`, `ssh_grep`, `ssh_glob`, `scp`, `rsync`, forwarding и persistent sessions.
* Для файлов проверь race conditions между проверкой существования и записью, включая symlink/dangling symlink cases.
* Проверь, что ограничения output/buffer действительно защищают от чрезмерного потребления памяти.

2. PACKAGING / BUILD

* Проверь `pyproject.toml`, package layout, setuptools configuration, package discovery, README и все metadata.
* Убедись, что проект корректно устанавливается через `pip install .` и собирается через `python -m build`.
* Проверь wheel/sdist: после установки пакет должен реально импортироваться и запускаться.
* Не меняй структуру проекта без необходимости.

3. MCP PROTOCOL

* Проверь поддерживаемые MCP protocol versions и negotiation.
* Убедись, что сервер корректно обрабатывает неизвестную/несовместимую версию, а не выбирает её молча.
* Проверь совместимость с актуальным MCP protocol и сохрани backward compatibility там, где это возможно без нарушения спецификации.
* Проверь `initialize`, `initialized`, `tools/list`, `tools/call`, errors и lifecycle.

4. SSH SESSION / PTY

* Проверь state machine persistent sessions.
* Проверь timeout, cancellation, disconnect, reconnect, process termination, process groups, PTY cleanup и зависшие процессы.
* Проверь ограничение unread buffer и поведение при очень большом/бесконечном выводе.
* Проверь Windows-specific code и корректность ConPTY/process handling.

5. REMOTE FILE TOOLS

* Проверь корректность quoting путей с пробелами, кавычками, спецсимволами, Unicode и shell metacharacters.
* Проверь atomicity и целостность записи.
* Проверь поведение при отсутствующем файле, директории, symlink, dangling symlink и permission denied.
* Проверь, чтобы `grep/glob` не загружали потенциально огромный результат целиком, если можно ограничить его раньше.

6. SCP / RSYNC / FORWARDING

* Проверь argument validation, escaping и корректность формирования команд.
* Проверь, что forwarding не позволяет непреднамеренно превратить удалённый запрос в локальный network access.
* Проверь ошибки, timeout и cleanup.

7. TESTING

* Запусти существующие проверки.
* Добавь минимальный набор regression tests для найденных security bugs и packaging/protocol issues.
* Обязательно протестируй положительные и отрицательные сценарии.
* После изменений запусти compile/test/build/install проверки.

8. CODE QUALITY

* Проверь duplicate logic, dead code, inconsistent error handling, unsafe defaults и места, где исключения могут скрывать реальные ошибки.
* Не делай косметический рефакторинг без необходимости.
* Сохрани API и поведение инструментов, если изменение не требуется для исправления проблемы.

В конце:

* Покажи список реально найденных проблем.
* Для каждой укажи, что именно исправлено.
* Укажи, какие проверки/тесты были выполнены и их результат.
* Не утверждай, что проблема исправлена, если ты её фактически не проверил.
* Если потенциальная проблема не подтверждается — так и укажи, ничего не ломая ради гипотезы.
