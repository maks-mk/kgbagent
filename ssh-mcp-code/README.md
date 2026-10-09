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

`ssh_sync` shells out to a local `rsync` client, which Windows does not ship.
Install one (MSYS2, Git for Windows, WSL) and put it on `PATH`, or use `ssh_scp`
for one-off transfers.

## Running

The server speaks JSON-RPC 2.0 over stdio:

```bash
python -m ssh_mcp
# or, after install, via the console script:
ssh-mcp
```

Point your MCP client at that command with the `stdio` transport.

## Configuration

The server can be configured using environment variables:

- `SSH_MCP_MAX_CONCURRENCY`: Number of concurrent tool call workers (default: `4`). Set to `1` for strictly sequential execution.
- `SSH_MCP_LOCAL_ROOT`: (See [Local paths](#local-paths))

## Portable Build (Windows)

To build a standalone executable that doesn't require a Python installation on the target machine:

```powershell
.\build.bat
```

This uses `PyInstaller` to produce `dist/ssh-mcp.exe`. The build includes `winpty` and other necessary dependencies for interactive PTY sessions on Windows.

## Local paths

`ssh_scp` and `ssh_sync` move files between the remote host and the *local*
machine, so they must know which directory a relative local path means. Every
relative local path (including `.`) is resolved against the **local root**:

1. `SSH_MCP_LOCAL_ROOT`, when that environment variable is set and points to an
   existing directory, otherwise
2. the server process's working directory, which MCP clients normally set to the
   directory that holds `mcp.json` — usually *not* the project you are working
   in.

The tool result always reports `local_root` and `resolved_local_paths`, so it is
visible which local paths were actually used. Pass absolute paths when in doubt.
Both the `env` and `cwd` keys of the stdio server config can be used to pin the
root:

```json
{
  "mcpServers": {
    "ssh": {
      "command": "./ssh-mcp/ssh-mcp.exe",
      "cwd": "D:/work/my-project",
      "env": { "SSH_MCP_LOCAL_ROOT": "D:/work/my-project" }
    }
  }
}
```

As a safety net, a relative local *source* that resolves to the local root itself
(or to one of its parents) is rejected with a validation error instead of
silently copying the whole directory the agent was launched from; pass an
absolute path if that directory is really what you mean. Relative
`identity_file` and `known_hosts_file` paths follow the same rule, so `cwd`
decides where they are looked up when `SSH_MCP_LOCAL_ROOT` is not set.

## Tools

- `ssh_exec` — run a one-off remote command.
- `ssh_start_session` / `ssh_ensure_session` / `ssh_read_session` /
  `ssh_write_session` / `ssh_stop_session` / `ssh_list_sessions` — persistent
  interactive PTY-backed sessions.
- `ssh_view` / `ssh_create` / `ssh_edit` / `ssh_grep` / `ssh_glob` — remote
  file read/create/edit/search.
- `ssh_scp` / `ssh_sync` — copy and incrementally sync files (`ssh_sync` needs a
  local `rsync` on `PATH`).
- `ssh_forward` / `ssh_list_forwards` / `ssh_stop_forward` — SSH port
  forwarding.

## Statefulness & Persistence

The server is designed to be run as a long-lived process. Interactive sessions
and port forwards are maintained in memory for the duration of the server's
lifecycle. When used with an MCP client that supports persistent stdio sessions,
SSH connections remain open between tool calls, avoiding repeated handshakes.

Tool calls are executed concurrently in a thread pool, so a long-running command
will not block other requests. Responses are returned as they complete (order
may vary). To prevent permanent hangs, tool calls have default timeouts: 60
seconds for `ssh_exec` and 120 seconds for remote scripts.

## Security notes

`extra_ssh_args` and `target` are validated to reject SSH options and flags that would run
commands or load libraries on the local machine, or redirect the transport:

- `target` (host/destination) must not start with a hyphen (`-`) to prevent option injection.
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
