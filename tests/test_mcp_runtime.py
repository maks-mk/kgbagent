import asyncio
from contextlib import asynccontextmanager, chdir
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
from uuid import uuid4

import anyio
import mcp.client.stdio as sdk_stdio

from core.config import AgentConfig
from tools.mcp_session import PersistentMCPClient
from tools.tool_registry import ToolRegistry


ROOT = Path(__file__).resolve().parents[1]
# Minimal local-only JSON-RPC fixture. Its counter belongs to one process, so a
# stateless adapter cannot pass the repeated/concurrent-call regression tests.
STATEFUL_SERVER = '''
import json
import os
import sys

count = 0
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method = msg["method"]
    if method == "initialize":
        result = {
            "protocolVersion": msg["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "local-regression", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [{
            "name": "state",
            "description": "Read or advance process-local state",
            "inputSchema": {
                "type": "object",
                "properties": {"op": {"type": "string", "default": "advance"}},
            },
        }]}
    elif method == "tools/call":
        op = msg["params"].get("arguments", {}).get("op", "advance")
        if op == "advance":
            count += 1
        result = {"content": [{"type": "text", "text": json.dumps({
            "count": count, "pid": os.getpid(), "cwd": os.getcwd(),
        })}], "isError": op == "fail"}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
'''


class MCPRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = ROOT / ".tmp_tests" / uuid4().hex
        self.config_dir = self.temp / "config with spaces"
        self.other_dir = self.temp / "other project"
        self.config_dir.mkdir(parents=True)
        self.other_dir.mkdir()
        self.addCleanup(lambda: shutil.rmtree(self.temp, ignore_errors=True))
        self.config_path = self.config_dir / "mcp.json"
        self.config_path.write_text("{}", encoding="utf-8")

    def make_config(self):
        return AgentConfig(
            _env_file=None,
            PROVIDER="openai",
            OPENAI_API_KEY="test-key",
            PROMPT_PATH=ROOT / "prompt.txt",
            MCP_CONFIG_PATH=self.config_path,
            ENABLE_SEARCH_TOOLS=False,
            ENABLE_PROCESS_TOOLS=False,
            ENABLE_SHELL_TOOL=False,
        )

    def make_registry(self, servers=None):
        if servers is not None:
            self.config_path.write_text(json.dumps(servers), encoding="utf-8")
        registry = ToolRegistry(self.make_config())
        self.addAsyncCleanup(registry.cleanup)
        return registry

    def stateful_config(self):
        (self.config_dir / "server.py").write_text(STATEFUL_SERVER, encoding="utf-8")
        return {
            "transport": "stdio",
            "command": os.path.relpath(sys.executable, self.config_dir),
            "args": ["server.py"],
            "cwd": None,
            "enabled": True,
        }

    def track_processes(self):
        processes = []
        create_process = sdk_stdio._create_platform_compatible_process

        async def spawn(*args, **kwargs):
            process = await create_process(*args, **kwargs)
            processes.append(process)
            return process

        patcher = mock.patch.object(sdk_stdio, "_create_platform_compatible_process", side_effect=spawn)
        patcher.start()
        self.addCleanup(patcher.stop)
        return processes

    @staticmethod
    def result_data(result):
        return json.loads(next(block["text"] for block in result if block["type"] == "text"))

    def test_stdio_paths_are_relative_to_config_not_process_cwd(self):
        registry = self.make_registry()
        for cwd in (None, ".", "server data", str(self.other_dir)):
            with self.subTest(cwd=cwd), chdir(self.other_dir):
                cfg = {
                    "transport": "stdio", "command": "./bin/server.exe", "cwd": cwd,
                    "args": ["relative.py", "--url=https://example.invalid/path"],
                    "enabled": True, "policy": {"read_only": True},
                }
                result = registry._prepare_mcp_server_config(cfg, {"transport", "command", "cwd", "args"})
                self.assertEqual(result["command"], str(self.config_dir / "bin" / "server.exe"))
                expected_cwd = (self.config_dir / (cwd or ".")).resolve()
                self.assertEqual(result["cwd"], str(expected_cwd))
                self.assertEqual(result["args"], cfg["args"])
                self.assertNotIn("enabled", result)
                self.assertNotIn("policy", result)
                self.assertEqual(cfg["command"], "./bin/server.exe")
                self.assertEqual(cfg["cwd"], cwd)

    def test_path_commands_and_environment_expansion(self):
        registry = self.make_registry()
        for command in ("python", "uv", "npx", "server.exe"):
            with self.subTest(command=command):
                cfg = {"transport": "stdio", "command": command}
                result = registry._prepare_mcp_server_config(cfg, set(cfg))
                self.assertEqual(result["command"], command)
                self.assertEqual(result["cwd"], str(self.config_dir))
        with mock.patch.dict(os.environ, {"MCP_TEST_BIN": "./bin", "MCP_TEST_CWD": "./data"}):
            cfg = registry._expand_env_vars({
                "transport": "stdio", "command": "${MCP_TEST_BIN}/server.exe", "cwd": "${MCP_TEST_CWD}",
            })
            result = registry._prepare_mcp_server_config(cfg, set(cfg))
        self.assertEqual(result["command"], str(self.config_dir / "bin" / "server.exe"))
        self.assertEqual(result["cwd"], str(self.config_dir / "data"))
        absolute = {"transport": "stdio", "command": sys.executable}
        self.assertEqual(
            registry._prepare_mcp_server_config(absolute, set(absolute))["command"],
            str(Path(sys.executable).resolve()),
        )
        if os.name == "nt":
            cfg = {"transport": "stdio", "command": r".\bin\server.exe"}
            self.assertEqual(
                registry._prepare_mcp_server_config(cfg, set(cfg))["command"],
                str(self.config_dir / "bin" / "server.exe"),
            )

    async def test_remote_transports_keep_stateless_adapter_and_config(self):
        for transport in ("http", "sse", "streamable_http"):
            with self.subTest(transport=transport):
                cfg = {"transport": transport, "url": "https://example.invalid/mcp"}
                registry = self.make_registry({"remote": cfg})
                client = SimpleNamespace(get_tools=mock.AsyncMock(return_value=[]), aclose=mock.AsyncMock())
                with mock.patch("langchain_mcp_adapters.client.MultiServerMCPClient", return_value=client) as constructor:
                    await registry._load_mcp_tools()
                constructor.assert_called_once_with({"remote": cfg})
                client.get_tools.assert_awaited_once()
                self.assertEqual(registry.mcp_clients, [client])
                await registry.cleanup()
                client.aclose.assert_awaited_once()

    async def test_disabled_stdio_server_is_not_started(self):
        registry = self.make_registry({"off": {**self.stateful_config(), "enabled": False}})
        with mock.patch("langchain_mcp_adapters.client.MultiServerMCPClient") as constructor:
            await registry._load_mcp_tools()
        constructor.assert_not_called()
        self.assertEqual(registry.mcp_clients, [])
        self.assertIn("off", registry.disabled_mcp_servers)

    async def test_stdio_state_survives_cwd_change_and_closes_before_reload(self):
        registry = self.make_registry({"local": self.stateful_config()})
        original_config = self.config_path.read_bytes()
        processes = self.track_processes()
        # Loading, calls and cleanup use distinct asyncio tasks, like the UI.
        with chdir(self.other_dir):
            await asyncio.wait_for(asyncio.create_task(registry._load_mcp_tools()), 15)
        self.assertEqual(registry.mcp_server_status[0]["error"], "")
        self.assertEqual(len(processes), 1)
        tool = registry.tools[0]
        first = self.result_data(await asyncio.create_task(tool.ainvoke({})))
        with chdir(self.other_dir):
            second = self.result_data(await asyncio.create_task(tool.ainvoke({})))
            results = await asyncio.gather(*(tool.ainvoke({}) for _ in range(3)))
        self.assertEqual([first["count"], second["count"]], [1, 2])
        self.assertEqual(first["pid"], second["pid"])
        self.assertEqual(Path(first["cwd"]), self.config_dir)
        self.assertEqual(sorted(self.result_data(result)["count"] for result in results), [3, 4, 5])
        failed = await tool.ainvoke({"type": "tool_call", "name": tool.name, "id": "failed", "args": {"op": "fail"}})
        self.assertEqual(failed.status, "error")
        self.assertEqual(self.result_data(await tool.ainvoke({}))["count"], 6)
        self.assertEqual(len(processes), 1)
        owner = registry.mcp_clients[0]
        await asyncio.wait_for(asyncio.create_task(registry.cleanup()), 10)
        self.assertTrue(owner._task.done())
        self.assertIsNotNone(processes[0].returncode)
        self.assertEqual(registry.mcp_clients, [])
        await registry.cleanup()  # idempotent shutdown
        with self.assertRaises(RuntimeError):
            await owner.get_tools()
        self.assertEqual(self.config_path.read_bytes(), original_config)

        replacement = self.make_registry()
        with chdir(self.other_dir):
            await asyncio.wait_for(replacement._load_mcp_tools(), 15)
            fresh = self.result_data(await replacement.tools[0].ainvoke({}))
        self.assertEqual(fresh["count"], 1)
        self.assertEqual(len(processes), 2)
        await replacement.cleanup()
        self.assertTrue(all(process.returncode is not None for process in processes))

    async def test_missing_executable_does_not_break_other_servers(self):
        registry = self.make_registry({
            "missing": {"transport": "stdio", "command": "./does-not-exist.exe"},
            "local": self.stateful_config(),
        })
        processes = self.track_processes()
        with self.assertLogs("tools.tool_registry", level="ERROR"):
            await asyncio.wait_for(registry._load_mcp_tools(), 15)
        statuses = {status["server"]: status for status in registry.mcp_server_status}
        self.assertTrue(statuses["missing"]["error"])
        self.assertEqual(statuses["local"]["error"], "")
        self.assertEqual(len(registry.mcp_clients), 1)
        self.assertEqual(self.result_data(await registry.tools[0].ainvoke({}))["count"], 1)
        await registry.cleanup()
        self.assertTrue(all(process.returncode is not None for process in processes))

    async def test_cancelled_batch_closes_servers_already_loaded(self):
        registry = self.make_registry({"ready": {"enabled": True}, "waiting": {"enabled": True}})
        client = SimpleNamespace(aclose=mock.AsyncMock())
        waiting = asyncio.Event()

        async def load(name, *args):
            if name == "ready":
                return name, client, [], None
            waiting.set()
            await asyncio.Event().wait()

        with mock.patch.object(ToolRegistry, "_load_single_mcp_server", side_effect=load):
            task = asyncio.create_task(registry._load_mcp_tools())
            await asyncio.wait_for(waiting.wait(), 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        client.aclose.assert_awaited_once()
        self.assertEqual(registry.mcp_clients, [])

    def fake_owner(self):
        events = []
        session = object()

        @asynccontextmanager
        async def open_session(name):
            # Real AnyIO scope catches cross-task __aexit__ regressions.
            with anyio.CancelScope():
                events.append(("enter", asyncio.current_task()))
                try:
                    yield session
                finally:
                    events.append(("exit", asyncio.current_task()))

        client = SimpleNamespace(
            session=open_session, callbacks=None, tool_interceptors=[],
            tool_name_prefix=False, handle_tool_errors=True,
        )
        owner = PersistentMCPClient(client, "local")
        self.addAsyncCleanup(owner.aclose)
        return owner, session, events

    async def test_session_context_is_owned_and_closed_by_same_task(self):
        owner, session, events = self.fake_owner()
        with mock.patch("langchain_mcp_adapters.tools.load_mcp_tools", new=mock.AsyncMock(return_value=[])) as load:
            self.assertEqual(await asyncio.create_task(owner.get_tools()), [])
            self.assertEqual(len(events), 1)
            await asyncio.create_task(owner.aclose())
        self.assertIs(load.call_args.args[0], session)
        self.assertEqual([event[0] for event in events], ["enter", "exit"])
        self.assertIs(events[0][1], events[1][1])
        self.assertTrue(owner._task.done())

    async def test_list_tools_failure_closes_session(self):
        owner, _, events = self.fake_owner()
        with mock.patch("langchain_mcp_adapters.tools.load_mcp_tools", new=mock.AsyncMock(side_effect=ValueError("invalid tools"))):
            with self.assertRaisesRegex(ValueError, "invalid tools"):
                await asyncio.wait_for(owner.get_tools(), 5)
        self.assertEqual([event[0] for event in events], ["enter", "exit"])
        self.assertTrue(owner._task.done())

    async def test_cancelled_startup_closes_session_and_propagates_cancellation(self):
        owner, _, events = self.fake_owner()
        loading = asyncio.Event()

        async def wait_for_tools(*args, **kwargs):
            loading.set()
            await asyncio.Event().wait()

        with mock.patch("langchain_mcp_adapters.tools.load_mcp_tools", side_effect=wait_for_tools):
            task = asyncio.create_task(owner.get_tools())
            await asyncio.wait_for(loading.wait(), 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        self.assertEqual([event[0] for event in events], ["enter", "exit"])
        self.assertIs(events[0][1], events[1][1])
        self.assertTrue(owner._task.done())

    async def test_failed_or_cancelled_agent_build_cleans_registry(self):
        import agent

        for failure in (RuntimeError("checkpoint failed"), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                registry = self.make_registry()
                client = SimpleNamespace(aclose=mock.AsyncMock())
                registry.mcp_clients.append(client)
                with (
                    mock.patch.object(agent, "setup_logging"),
                    mock.patch.object(agent, "ToolRegistry", return_value=registry),
                    mock.patch.object(ToolRegistry, "load_all", new=mock.AsyncMock()),
                    mock.patch.object(agent, "create_checkpoint_runtime", new=mock.AsyncMock(side_effect=failure)),
                ):
                    with self.assertRaises(type(failure)):
                        await agent.build_agent_app(registry.config)
                client.aclose.assert_awaited_once()

    async def test_failed_graph_build_closes_checkpoint_and_mcp(self):
        import agent

        registry = self.make_registry()
        client = SimpleNamespace(aclose=mock.AsyncMock())
        registry.mcp_clients.append(client)
        checkpoint = SimpleNamespace(aclose=mock.AsyncMock(), to_dict=lambda: {})
        with (
            mock.patch.object(agent, "setup_logging"),
            mock.patch.object(agent, "ToolRegistry", return_value=registry),
            mock.patch.object(ToolRegistry, "load_all", new=mock.AsyncMock()),
            mock.patch.object(agent, "create_checkpoint_runtime", new=mock.AsyncMock(return_value=checkpoint)),
            mock.patch.object(agent, "JsonlRunLogger"),
            mock.patch.object(agent, "build_compiled_agent", side_effect=RuntimeError("graph failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "graph failed"):
                await agent.build_agent_app(registry.config)
        client.aclose.assert_awaited_once()
        checkpoint.aclose.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
