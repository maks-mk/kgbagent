"""Repair Windows environment variables that the MCP client dropped.

MCP clients that launch a stdio server do not forward the full environment of
the launching process.  The reference Python client, for example, forwards only
``mcp.client.stdio.DEFAULT_INHERITED_ENV_VARS``, a short allow-list that omits
machine-wide settings such as ``ProgramData``.

Windows OpenSSH is sensitive to that omission: when ``ProgramData`` is not
defined, ``ssh.exe`` (and its siblings ``scp.exe``/``rsync.exe``) terminates
immediately with exit code 255 and produces no output at all, which surfaces
through this server as an unexplained failed connection.

Because every child process spawned by this package inherits ``os.environ``,
restoring the missing variables once, as early as possible, fixes all of the
spawning code paths (plain subprocesses, interactive PTYs and port forwards)
without passing a custom environment around.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

# Machine-wide environment variables live in the registry; this is the
# authoritative source when the process environment is incomplete.
_MACHINE_ENVIRONMENT_KEY = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
_USER_ENVIRONMENT_KEY = "Environment"


def _registry_environment() -> dict[str, str]:
    """Return the machine and user environment variables from the registry.

    User values are applied after machine values so that they win on name
    collisions.  Any failure (non-Windows host, locked-down registry, missing
    key) degrades to an empty result rather than raising.
    """
    try:
        import winreg
    except ImportError:  # pragma: no cover - non-Windows host
        return {}

    values: dict[str, str] = {}
    for hive, subkey in (
        (winreg.HKEY_LOCAL_MACHINE, _MACHINE_ENVIRONMENT_KEY),
        (winreg.HKEY_CURRENT_USER, _USER_ENVIRONMENT_KEY),
    ):
        try:
            with winreg.OpenKey(hive, subkey) as key:
                index = 0
                while True:
                    try:
                        name, value, _kind = winreg.EnumValue(key, index)
                    except OSError:
                        break
                    index += 1
                    # REG_EXPAND_SZ values are stored unexpanded.
                    if isinstance(value, str):
                        values[name] = os.path.expandvars(value)
        except OSError:
            continue
    return values


def ensure_complete_environment(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Define any environment variable present in the registry but missing here.

    Returns the names that were added, which makes the behaviour observable in
    tests.  ``os.environ`` is used by default, so the restored variables are
    inherited by every child process spawned afterwards.
    """
    if os.name != "nt":
        return []

    env: MutableMapping[str, str] = os.environ if environ is None else environ
    # os.environ is case-insensitive on Windows, but a caller-supplied mapping
    # may not be; compare uppercased names so an existing "ProgramData" is
    # never shadowed by a differently-cased duplicate.
    known = {name.upper() for name in env}

    restored: list[str] = []
    for name, value in _registry_environment().items():
        if value and name.upper() not in known:
            env[name] = value
            known.add(name.upper())
            restored.append(name)

    # ProgramData is the variable Windows OpenSSH cannot start without, so fall
    # back to its documented default if the registry lookup was unavailable.
    if "PROGRAMDATA" not in known:
        drive = env.get("SystemDrive") or os.environ.get("SystemDrive") or "C:"
        env["ProgramData"] = drive.rstrip("\\/") + r"\ProgramData"
        restored.append("ProgramData")

    return restored
