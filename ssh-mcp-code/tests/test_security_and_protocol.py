"""Regression tests for the issues found during the check_list.md audit.

These intentionally use only the standard library (unittest) so they can run
without any extra dependencies:

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssh_mcp import __version__
from ssh_mcp import ssh as ssh_module
from ssh_mcp.server import DEFAULT_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS, McpServer
from ssh_mcp.ssh import (
    LOCAL_ROOT_ENV,
    ConnectionSettings,
    SshToolService,
    ValidationError,
    _build_scp_remote_path,
    _extra_ssh_args_argument,
    _float_argument,
    _normalize_local_destination,
    _normalize_local_sources,
)


def _check(args):
    return _extra_ssh_args_argument({"extra_ssh_args": args})


class ExtraSshArgsSecurityTests(unittest.TestCase):
    """Covers the extra_ssh_args blocklist and its former bypasses."""

    def test_blocks_proxycommand_separated_and_attached(self):
        for args in (["-o", "ProxyCommand=nc evil 1"], ["-oProxyCommand=nc evil 1"]):
            with self.assertRaises(ValidationError):
                _check(args)

    def test_blocks_newly_added_local_code_execution_options(self):
        # KnownHostsCommand executes a shell command; PKCS11Provider and
        # SecurityKeyProvider dlopen() a local library -> code execution.
        for option in ("KnownHostsCommand=/tmp/x", "PKCS11Provider=/tmp/x.so",
                       "SecurityKeyProvider=/tmp/x.so"):
            with self.assertRaises(ValidationError, msg=option):
                _check(["-o", option])
            with self.assertRaises(ValidationError, msg=option):
                _check([f"-o{option}"])

    def test_blocks_config_file_flag(self):
        # -F can load a config file that itself sets ProxyCommand, etc.
        for args in (["-F", "/tmp/evil_ssh_config"], ["-F/tmp/evil_ssh_config"]):
            with self.assertRaises(ValidationError):
                _check(args)

    def test_blocks_forwarding_flags_separated(self):
        for flag in ("-L", "-R", "-D", "-W"):
            with self.assertRaises(ValidationError, msg=flag):
                _check([flag, "8080:localhost:80"])

    def test_blocks_forwarding_flags_attached(self):
        # Attached form was the bypass: "-L8080:..." must be rejected too.
        for arg in ("-L8080:localhost:80", "-R9000:localhost:90", "-D1080",
                    "-Wlocalhost:22"):
            with self.assertRaises(ValidationError, msg=arg):
                _check([arg])

    def test_case_insensitive_option_matching(self):
        with self.assertRaises(ValidationError):
            _check(["-o", "proxycommand=nc evil 1"])

    def test_allows_benign_args(self):
        # These must NOT raise (regression guard against over-blocking).
        self.assertEqual(_check(["-J", "jumphost"]), ["-J", "jumphost"])
        self.assertEqual(_check(["-o", "Compression=yes"]), ["-o", "Compression=yes"])
        self.assertEqual(_check(["-C"]), ["-C"])
        self.assertEqual(_check(["-i", "/home/u/.ssh/id_ed25519"]),
                         ["-i", "/home/u/.ssh/id_ed25519"])
    def test_rejects_option_like_target(self):
        for target in ("-oProxyCommand=echo pwned", "-Funsafe-config", "--help"):
            with self.assertRaises(ValidationError, msg=target):
                ConnectionSettings.from_arguments({"target": target})


class NumericArgumentValidationTests(unittest.TestCase):
    def test_rejects_non_finite_timeout(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValidationError, msg=repr(value)):
                _float_argument({"timeout": value}, "timeout", allow_zero=False)


class McpProtocolTests(unittest.TestCase):
    def _server(self):
        # No stdio needed; we call _dispatch directly.
        return McpServer(stdin=io.BytesIO(b""), stdout=io.BytesIO())

    def test_initialize_echoes_supported_version(self):
        server = self._server()
        for version in SUPPORTED_PROTOCOL_VERSIONS:
            result = server._dispatch("initialize", {"protocolVersion": version})
            self.assertEqual(result["protocolVersion"], version)
            self.assertEqual(result["serverInfo"]["version"], __version__)

    def test_initialize_negotiates_down_for_unknown_version(self):
        server = self._server()
        result = server._dispatch("initialize", {"protocolVersion": "1999-01-01"})
        # Must not silently echo the unsupported version back.
        self.assertNotEqual(result["protocolVersion"], "1999-01-01")
        self.assertEqual(result["protocolVersion"], DEFAULT_PROTOCOL_VERSION)
        self.assertIn(result["protocolVersion"], SUPPORTED_PROTOCOL_VERSIONS)

    def test_tools_list_requires_initialize(self):
        from ssh_mcp.server import JsonRpcRequestError, JSONRPC_SERVER_NOT_INITIALIZED
        server = self._server()
        with self.assertRaises(JsonRpcRequestError) as ctx:
            server._dispatch("tools/list", {})
        self.assertEqual(ctx.exception.code, JSONRPC_SERVER_NOT_INITIALIZED)

    def test_tools_list_after_initialize(self):
        server = self._server()
        server._dispatch("initialize", {})
        result = server._dispatch("tools/list", {})
        names = {tool["name"] for tool in result["tools"]}
        self.assertIn("ssh_exec", names)
        self.assertIn("ssh_forward", names)

    def test_tool_calls_do_not_serialize_behind_each_other(self):
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "slow_tool", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "fast_tool", "arguments": {}}},
        ]
        stdin = io.BytesIO(
            b"".join(json.dumps(request).encode("utf-8") + b"\n" for request in requests)
        )
        stdout = io.BytesIO()
        server = McpServer(stdin=stdin, stdout=stdout)

        def slow_tool(_arguments):
            time.sleep(0.5)
            return {"stdout": "slow"}

        server._tool_handlers = {
            "slow_tool": slow_tool,
            "fast_tool": lambda _arguments: {"stdout": "fast"},
        }
        server.serve()

        responses = [json.loads(line) for line in stdout.getvalue().decode("utf-8").splitlines()]
        # The fast call was submitted after the slow one but must not wait for it.
        self.assertEqual([response["id"] for response in responses], [1, 3, 2])

    def test_max_concurrency_env_override(self):
        from ssh_mcp.server import _configured_max_concurrent_tool_calls

        with mock.patch.dict(os.environ, {"SSH_MCP_MAX_CONCURRENCY": "8"}):
            self.assertEqual(_configured_max_concurrent_tool_calls(), 8)
        with mock.patch.dict(os.environ, {"SSH_MCP_MAX_CONCURRENCY": "0"}):
            self.assertEqual(_configured_max_concurrent_tool_calls(), 1)
        with mock.patch.dict(os.environ, {"SSH_MCP_MAX_CONCURRENCY": "junk"}):
            self.assertEqual(_configured_max_concurrent_tool_calls(), 4)


class LocalPathResolutionTests(unittest.TestCase):
    """Relative local paths must be anchored to the ssh-mcp local root.

    Regression test: relative local paths used to be handed to scp/rsync
    verbatim, so they resolved against the MCP server process's working
    directory -- i.e. the directory the agent was launched from -- and a
    relative '.' silently copied that whole directory.
    """

    def setUp(self):
        tmp = Path(tempfile.mkdtemp(prefix="ssh-mcp-local-root-"))
        self.addCleanup(shutil.rmtree, str(tmp), ignore_errors=True)
        self.root = tmp.resolve()
        (self.root / "payload.txt").write_text("payload", encoding="utf-8")
        previous = os.environ.get(LOCAL_ROOT_ENV)
        os.environ[LOCAL_ROOT_ENV] = str(self.root)
        self.addCleanup(self._restore_local_root, previous)

    @staticmethod
    def _restore_local_root(previous):
        if previous is None:
            os.environ.pop(LOCAL_ROOT_ENV, None)
        else:
            os.environ[LOCAL_ROOT_ENV] = previous

    def test_relative_source_is_anchored_to_the_local_root(self):
        self.assertEqual(
            _normalize_local_sources(["payload.txt"], recursive=False),
            [str(self.root / "payload.txt")],
        )

    def test_relative_destination_is_anchored_to_the_local_root(self):
        self.assertEqual(
            _normalize_local_destination("payload.txt"), str(self.root / "payload.txt")
        )
        # An explicit '.' stays allowed for downloads, but must mean the local
        # root, not "whatever directory the agent happened to start in".
        self.assertEqual(_normalize_local_destination("."), str(self.root))

    def test_absolute_paths_are_left_alone(self):
        absolute = str(self.root / "payload.txt")
        self.assertEqual(_normalize_local_sources([absolute], recursive=False), [absolute])

    def test_dot_source_is_rejected_instead_of_copying_the_local_root(self):
        with self.assertRaises(ValidationError) as ctx:
            _normalize_local_sources(["."], recursive=True)
        self.assertIn(LOCAL_ROOT_ENV, str(ctx.exception))

    def test_missing_relative_source_error_names_the_local_root(self):
        with self.assertRaises(ValidationError) as ctx:
            _normalize_local_sources(["nope.txt"], recursive=False)
        self.assertIn(str(self.root), str(ctx.exception))

    def test_directory_source_still_requires_recursive(self):
        (self.root / "nested").mkdir()
        with self.assertRaises(ValidationError):
            _normalize_local_sources(["nested"], recursive=False)


class ScpRemotePathTests(unittest.TestCase):
    """scp >= 9 speaks SFTP: the remote path is not parsed by a remote shell."""

    def setUp(self):
        self.connection = ConnectionSettings(target="host")
        patcher = mock.patch.object(ssh_module, "_openssh_major_version", return_value=9)
        self.addCleanup(patcher.stop)
        self.major_version = patcher.start()

    def test_modern_client_receives_the_path_verbatim(self):
        self.assertEqual(
            _build_scp_remote_path(self.connection, "/tmp/a b", ssh_binary="ssh"),
            "host:/tmp/a b",
        )

    def test_legacy_client_still_gets_a_quoted_path(self):
        self.major_version.return_value = 8
        self.assertEqual(
            _build_scp_remote_path(self.connection, "/tmp/a b", ssh_binary="ssh"),
            "host:'/tmp/a b'",
        )

    def test_dash_o_forces_the_legacy_protocol(self):
        connection = ConnectionSettings(target="host", extra_ssh_args=["-O"])
        self.assertEqual(
            _build_scp_remote_path(connection, "/tmp/a b", ssh_binary="ssh"),
            "host:'/tmp/a b'",
        )

    def test_unknown_version_is_treated_as_a_modern_client(self):
        self.major_version.return_value = None
        self.assertEqual(
            _build_scp_remote_path(self.connection, "/tmp/x", ssh_binary="ssh"), "host:/tmp/x"
        )


class ScpVersionProbeTests(unittest.TestCase):
    """Version probing must never turn a working scp into an error."""

    def tearDown(self):
        ssh_module._openssh_major_version.cache_clear()

    def test_version_probe_degrades_when_ssh_is_missing(self):
        with mock.patch.object(
            ssh_module, "_resolve_ssh_binary", side_effect=ValidationError("no ssh")
        ):
            ssh_module._openssh_major_version.cache_clear()
            self.assertIsNone(ssh_module._openssh_major_version("missing-ssh-binary"))
            # And an unresolvable ssh must still yield a usable scp path.
            self.assertEqual(
                _build_scp_remote_path(
                    ConnectionSettings(target="host"), "/tmp/a b", ssh_binary="missing-ssh-binary"
                ),
                "host:/tmp/a b",
            )


class RsyncAvailabilityTests(unittest.TestCase):
    def test_missing_rsync_error_points_at_an_alternative(self):
        with self.assertRaises(ValidationError) as ctx:
            ssh_module._resolve_rsync_binary("rsync-not-installed-for-tests")
        message = str(ctx.exception)
        self.assertIn("rsync client", message)
        self.assertIn("ssh_scp", message)


class ScpArgvTests(unittest.TestCase):
    """End-to-end check of the argv handed to scp (process launch is stubbed)."""

    def setUp(self):
        tmp = Path(tempfile.mkdtemp(prefix="ssh-mcp-scp-"))
        self.addCleanup(shutil.rmtree, str(tmp), ignore_errors=True)
        self.root = tmp.resolve()
        (self.root / "payload.txt").write_text("payload", encoding="utf-8")
        scp_stub = self.root / "scp-stub.exe"
        scp_stub.write_text("", encoding="utf-8")
        previous = os.environ.get(LOCAL_ROOT_ENV)
        os.environ[LOCAL_ROOT_ENV] = str(self.root)
        self.addCleanup(self._restore_local_root, previous)
        self.service = SshToolService(scp_binary=str(scp_stub), state_dir=self.root / "state")
        self.addCleanup(self.service.close)
        self.argv: list[str] = []

    @staticmethod
    def _restore_local_root(previous):
        if previous is None:
            os.environ.pop(LOCAL_ROOT_ENV, None)
        else:
            os.environ[LOCAL_ROOT_ENV] = previous

    def _fake_run(self, argv, *, timeout):
        self.argv = list(argv)
        return {"exit_code": 0, "timed_out": False, "stdout": "", "stderr": ""}

    def _run_scp(self, arguments):
        with mock.patch.object(ssh_module, "_run_without_pty", self._fake_run):
            with mock.patch.object(ssh_module, "_openssh_major_version", return_value=9):
                return self.service.ssh_scp(arguments)

    def test_upload_passes_an_absolute_local_source(self):
        result = self._run_scp(
            {
                "target": "host",
                "direction": "upload",
                "sources": ["payload.txt"],
                "destination": "/tmp/dir",
            }
        )
        self.assertEqual(self.argv[-2], str(self.root / "payload.txt"))
        self.assertEqual(self.argv[-1], "host:/tmp/dir")
        self.assertEqual(result["resolved_local_paths"], [str(self.root / "payload.txt")])
        self.assertEqual(result["local_root"], str(self.root))

    def test_upload_does_not_shell_quote_the_remote_path(self):
        self._run_scp(
            {
                "target": "host",
                "direction": "upload",
                "sources": ["payload.txt"],
                "destination": "/tmp/a b",
            }
        )
        self.assertEqual(self.argv[-1], "host:/tmp/a b")

    def test_download_anchors_a_relative_destination(self):
        result = self._run_scp(
            {
                "target": "host",
                "direction": "download",
                "sources": ["/tmp/a b"],
                "destination": ".",
            }
        )
        self.assertEqual(self.argv[-2], "host:/tmp/a b")
        self.assertEqual(self.argv[-1], str(self.root))
        self.assertEqual(result["resolved_local_paths"], [str(self.root)])

    def test_misconfigured_local_root_does_not_fail_absolute_paths(self):
        os.environ[LOCAL_ROOT_ENV] = str(self.root / "does-not-exist")
        result = self._run_scp(
            {
                "target": "host",
                "direction": "download",
                "sources": ["/tmp/x"],
                "destination": str(self.root / "payload.txt"),
            }
        )
        self.assertTrue(result["ok"])
        self.assertIn("invalid", result["local_root"])


class PackagingTests(unittest.TestCase):
    def test_entry_point_importable(self):
        from ssh_mcp.__main__ import main
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
