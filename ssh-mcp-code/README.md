# ssh-mcp

A stdio [Model Context Protocol](https://modelcontextprotocol.io) server that
exposes Bash-like SSH execution, persistent interactive sessions, remote file
read/edit/search, SCP/rsync transfers, and port forwarding to MCP clients.

It wraps the local OpenSSH client (`ssh`, `scp`) and `rsync`, so it reuses your
existing SSH configuration, keys, and agents. Works on POSIX and Windows
(interactive sessions use ConPTY via `pywinpty` on Windows).

## Installation

```bash
pip install .
```

Windows interactive sessions additionally require `pywinpty` (installed
automatically from the `platform_system == 'Windows'` dependency marker).

## Running

The server speaks JSON-RPC 2.0 over stdio:

```bash
python -m ssh_mcp
# or, after install, via the console script:
ssh-mcp
```

Point your MCP client at that command with the `stdio` transport.

## Portable Build (Windows)

To build a standalone executable that doesn't require a Python installation on the target machine:

```powershell
.\build.bat
```

This uses `PyInstaller` to produce `dist/ssh-mcp.exe`. The build includes `winpty` and other necessary dependencies for interactive PTY sessions on Windows.

## Tools

- `ssh_exec` — run a one-off remote command.
- `ssh_start_session` / `ssh_ensure_session` / `ssh_read_session` /
  `ssh_write_session` / `ssh_stop_session` / `ssh_list_sessions` — persistent
  interactive PTY-backed sessions.
- `ssh_view` / `ssh_create` / `ssh_edit` / `ssh_grep` / `ssh_glob` — remote
  file read/create/edit/search.
- `ssh_scp` / `ssh_sync` — copy and incrementally sync files.
- `ssh_forward` / `ssh_list_forwards` / `ssh_stop_forward` — SSH port
  forwarding.

## Statefulness & Persistence

The server is designed to be run as a long-lived process. Interactive sessions
and port forwards are maintained in memory for the duration of the server's
lifecycle. When used with an MCP client that supports persistent stdio sessions,
SSH connections remain open between tool calls, avoiding repeated handshakes.

## Security notes

`extra_ssh_args` is validated to reject SSH options and flags that would run
commands or load libraries on the local machine, or redirect the transport:

- `-o` options: `ProxyCommand`, `LocalCommand`, `PermitLocalCommand`,
  `LocalForward`, `RemoteForward`, `DynamicForward`, `KnownHostsCommand`,
  `PKCS11Provider`, `SecurityKeyProvider`.
- Short flags: `-L`, `-R`, `-D`, `-W` (use the `ssh_forward` tool instead) and
  `-F` (loading an arbitrary ssh config file). Both separated (`-L spec`) and
  attached (`-Lspec`) forms are rejected.

Use the dedicated `port`, `identity_file`, `known_hosts_file`, and
`strict_host_key_checking` parameters instead of smuggling them through
`extra_ssh_args`.

## Protocol versions

Supported MCP protocol versions: `2025-06-18` (default) and `2024-11-05`. If a
client requests an unsupported version, the server negotiates down to a version
it supports rather than echoing the unknown version.

## License

MIT
