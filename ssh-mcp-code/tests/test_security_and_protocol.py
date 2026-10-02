"""Regression tests for the issues found during the check_list.md audit.

These intentionally use only the standard library (unittest) so they can run
without any extra dependencies:

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssh_mcp import __version__
from ssh_mcp.server import DEFAULT_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS, McpServer
from ssh_mcp.ssh import ValidationError, _extra_ssh_args_argument


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


class PackagingTests(unittest.TestCase):
    def test_entry_point_importable(self):
        from ssh_mcp.__main__ import main
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
