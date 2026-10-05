import importlib
import json
import re
import shutil
import sys
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path
from uuid import uuid4

from core.config import AgentConfig
from core.multimodal import DEFAULT_MODEL_CAPABILITIES
from core.nodes.tools import ToolsMixin
from core.safety_policy import SafetyPolicy
from core.validation import validate_tool_result
from langchain_core.tools import StructuredTool
from tools import process_tools
from tools.process_tools import run_background_process
from tools.search_tools import _RUNTIME, _format_tavily_error, _parse_urls_input, crawl_site, fetch_content
from tools.tool_registry import ToolRegistry
from ui.runtime_payloads import build_tools_snapshot
from tools.user_input_tool import request_user_input


class ToolingRefactorTests(unittest.IsolatedAsyncioTestCase):
    def _make_config(self, **overrides):
        defaults = {
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "PROMPT_PATH": Path(__file__).resolve().parents[1] / "prompt.txt",
            "MCP_CONFIG_PATH": Path(__file__).resolve().parents[1] / "tests" / "missing_mcp.json",
            "ENABLE_SEARCH_TOOLS": False,
            "ENABLE_PROCESS_TOOLS": False,
            "ENABLE_SHELL_TOOL": False,
        }
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def _workspace_tempdir(self) -> Path:
        path = Path.cwd() / ".tmp_tests" / uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    async def test_execute_tool_serializes_structured_results_as_json(self):
        owner = ToolsMixin()
        owner.tools_map = {
            "mcp-tool": mock.Mock(ainvoke=mock.AsyncMock(return_value=[{"type": "text", "text": "данные"}]))
        }
        owner._log_run_event = mock.Mock()

        result = await owner._execute_tool("mcp-tool", {})

        self.assertEqual([{"type": "text", "text": "данные"}], json.loads(result))

    async def test_execute_tool_preserves_string_results(self):
        owner = ToolsMixin()
        owner.tools_map = {"text-tool": mock.Mock(ainvoke=mock.AsyncMock(return_value="plain text"))}
        owner._log_run_event = mock.Mock()

        result = await owner._execute_tool("text-tool", {})

        self.assertEqual("plain text", result)

    async def test_tool_registry_preserves_filesystem_delete_tools(self):
        registry = ToolRegistry(self._make_config())
        await registry.load_all()
        names = {tool.name for tool in registry.tools}
        self.assertIn("safe_delete_file", names)
        self.assertIn("safe_delete_directory", names)
        self.assertIn("download_file", names)

    async def test_tool_registry_filters_disabled_local_tool(self):
        tmp = self._workspace_tempdir()
        mcp_config_path = tmp / "mcp.json"
        mcp_config_path.write_text("{}", encoding="utf-8")
        registry = ToolRegistry(self._make_config(MCP_CONFIG_PATH=mcp_config_path))
        await registry.load_all()

        registry.set_tool_enabled("read_file", False)

        self.assertNotIn("read_file", {tool.name for tool in registry.active_tools()})
        self.assertIn("read_file", {tool.name for tool in registry.tools})
        persisted = json.loads(mcp_config_path.read_text(encoding="utf-8"))
        self.assertFalse(persisted["_builtin_tools"]["read_file"])

    async def test_tool_registry_catalogs_globally_disabled_builtin_tools(self):
        registry = ToolRegistry(self._make_config(ENABLE_FILESYSTEM_TOOLS=False))

        await registry.load_all()

        catalog_names = {tool.name for tool in registry.builtin_tools}
        self.assertIn("read_file", catalog_names)
        self.assertIn("batch_web_search", catalog_names)
        self.assertIn("fetch_content", catalog_names)
        self.assertIn("crawl_site", catalog_names)
        self.assertNotIn("web_search", catalog_names)
        self.assertIn("cli_exec", catalog_names)
        self.assertNotIn("read_file", {tool.name for tool in registry.active_tools()})

    async def test_tool_registry_persists_builtin_state_without_replacing_mcp_servers(self):
        tmp = self._workspace_tempdir()
        mcp_config_path = tmp / "mcp.json"
        mcp_config_path.write_text(
            json.dumps({"context7": {"url": "https://mcp.context7.com/mcp", "enabled": False}}),
            encoding="utf-8",
        )
        registry = ToolRegistry(self._make_config(MCP_CONFIG_PATH=mcp_config_path))
        await registry.load_all()

        registry.set_tool_enabled("read_file", False)

        persisted = json.loads(mcp_config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            persisted["context7"],
            {"url": "https://mcp.context7.com/mcp", "enabled": False},
        )
        self.assertEqual(persisted["_builtin_tools"], {"read_file": False})

        restored_registry = ToolRegistry(self._make_config(MCP_CONFIG_PATH=mcp_config_path))
        await restored_registry.load_all()
        restored_rows = build_tools_snapshot(restored_registry)
        restored_read_file = next(row for row in restored_rows if row.get("name") == "read_file")
        self.assertFalse(restored_read_file["enabled"])
        self.assertNotIn("read_file", {tool.name for tool in restored_registry.active_tools()})

    async def test_tool_snapshot_matches_active_tools_for_all_builtin_overrides(self):
        tmp = self._workspace_tempdir()
        mcp_config_path = tmp / "mcp.json"
        mcp_config_path.write_text("{}", encoding="utf-8")
        registry = ToolRegistry(self._make_config(MCP_CONFIG_PATH=mcp_config_path))
        await registry.load_all()

        for tool in registry.builtin_tools:
            registry.set_tool_enabled(tool.name, False)
            rows = build_tools_snapshot(registry)
            row = next(item for item in rows if item.get("name") == tool.name)
            self.assertFalse(row["enabled"], tool.name)
            self.assertNotIn(tool.name, {item.name for item in registry.active_tools()}, tool.name)

            registry.set_tool_enabled(tool.name, True)
            rows = build_tools_snapshot(registry)
            row = next(item for item in rows if item.get("name") == tool.name)
            self.assertEqual(
                row["enabled"],
                tool.name in {item.name for item in registry.active_tools()},
                tool.name,
            )

    def _patch_mcp_tools(self, server: str, specs):
        """Stand in for a real MCP server so registry tests stay process-free."""

        def inject(registry_self):
            injected = []
            for name, description in specs:
                tool = StructuredTool.from_function(
                    func=lambda **kwargs: "ok",
                    name=name,
                    description=description,
                )
                registry_self.mcp_tool_servers[id(tool)] = server
                registry_self.tools.append(tool)
                registry_self.tool_metadata[name] = registry_self._infer_mcp_metadata(
                    tool,
                    server_policy={"read_only": True},
                )
                injected.append(tool)
            registry_self.mcp_server_status.append(
                {
                    "server": server,
                    "loaded_tools": [tool.name for tool in injected],
                    "error": "",
                    "enabled": True,
                }
            )

        async def loader(registry_self):
            inject(registry_self)

        return mock.patch.object(ToolRegistry, "_load_mcp_tools", new=loader)

    def _mcp_config_path(self, payload) -> Path:
        tmp = self._workspace_tempdir()
        mcp_config_path = tmp / "mcp.json"
        mcp_config_path.write_text(json.dumps(payload), encoding="utf-8")
        return mcp_config_path

    async def test_mcp_tool_stays_available_when_same_named_builtin_is_disabled(self):
        mcp_config_path = self._mcp_config_path(
            {
                "ddg-search": {"command": "uvx", "args": ["duckduckgo-mcp-server"], "enabled": True},
                "_builtin_tools": {"fetch_content": False},
            }
        )
        config = self._make_config(MCP_CONFIG_PATH=mcp_config_path, ENABLE_SEARCH_TOOLS=True)

        with self._patch_mcp_tools(
            "ddg-search",
            [
                ("search", "Search the web."),
                ("fetch_content", "Fetch and extract the main text content from a webpage."),
                ("expand_link", "Expand a shortened ref link."),
            ],
        ):
            registry = ToolRegistry(config)
            await registry.load_all()

        active = registry.active_tools()
        active_fetch = [tool for tool in active if tool.name == "fetch_content"]
        self.assertEqual(len(active_fetch), 1)
        self.assertEqual(registry.mcp_server_for_tool(active_fetch[0]), "ddg-search")
        self.assertTrue(registry.is_builtin_tool(registry.builtin_tools[0]))
        self.assertEqual(
            registry.mcp_tool_groups([tool.name for tool in active]),
            [("ddg-search", ["search", "fetch_content", "expand_link"])],
        )

        rows = build_tools_snapshot(registry)
        server_row = next(row for row in rows if row.get("name") == "ddg-search")
        self.assertEqual(server_row["description"], "MCP server - 3 tool(s)")
        self.assertEqual(
            [tool["name"] for tool in server_row["tools"]],
            ["expand_link", "fetch_content", "search"],
        )
        builtin_row = next(row for row in rows if row["kind"] == "tool" and row["name"] == "fetch_content")
        self.assertFalse(builtin_row["enabled"])

    async def test_mcp_tool_wins_over_same_named_enabled_builtin(self):
        mcp_config_path = self._mcp_config_path(
            {"ddg-search": {"command": "uvx", "args": ["duckduckgo-mcp-server"], "enabled": True}}
        )
        config = self._make_config(MCP_CONFIG_PATH=mcp_config_path, ENABLE_SEARCH_TOOLS=True)

        with self._patch_mcp_tools("ddg-search", [("fetch_content", "Fetch a page.")]):
            registry = ToolRegistry(config)
            await registry.load_all()

        self.assertNotIn("fetch_content", registry.disabled_local_tools)
        active_fetch = [tool for tool in registry.active_tools() if tool.name == "fetch_content"]
        self.assertEqual(len(active_fetch), 1)
        self.assertEqual(registry.mcp_server_for_tool(active_fetch[0]), "ddg-search")

    async def test_disabling_mcp_server_keeps_same_named_builtin_tool(self):
        mcp_config_path = self._mcp_config_path(
            {"ddg-search": {"command": "uvx", "args": ["duckduckgo-mcp-server"], "enabled": True}}
        )
        config = self._make_config(MCP_CONFIG_PATH=mcp_config_path, ENABLE_SEARCH_TOOLS=True)

        with self._patch_mcp_tools("ddg-search", [("fetch_content", "Fetch a page.")]):
            registry = ToolRegistry(config)
            await registry.load_all()

        registry.set_mcp_server_enabled("ddg-search", False)

        active_names = [tool.name for tool in registry.active_tools()]
        self.assertEqual(active_names.count("fetch_content"), 1)
        active_fetch = next(tool for tool in registry.active_tools() if tool.name == "fetch_content")
        self.assertTrue(registry.is_builtin_tool(active_fetch))

    async def test_tool_registry_does_not_keep_selector_catalog_state(self):
        registry = ToolRegistry(self._make_config())
        await registry.load_all()
        self.assertFalse(hasattr(registry, "selector_catalog"))

    async def test_tool_registry_fallback_delete_tools_without_filesystem(self):
        registry = ToolRegistry(self._make_config(ENABLE_FILESYSTEM_TOOLS=False))
        await registry.load_all()
        names = [tool.name for tool in registry.tools]
        self.assertEqual(names, ["safe_delete_file", "safe_delete_directory", "read_skills", "request_user_input"])

    async def test_tool_registry_compacts_descriptions_and_schema_docs(self):
        registry = ToolRegistry(self._make_config(ENABLE_SHELL_TOOL=True))
        await registry.load_all()

        tools_by_name = {tool.name: tool for tool in registry.tools}
        cli_tool = tools_by_name["cli_exec"]
        self.assertLess(len(cli_tool.description), 260)
        self.assertNotIn("\n", cli_tool.description)
        schema = cli_tool.args_schema.model_json_schema()
        self.assertNotIn("description", schema)
        timeout_schema = schema["properties"]["timeout"]
        self.assertEqual(timeout_schema["default"], 120)
        self.assertEqual(timeout_schema["exclusiveMinimum"], 0)
        self.assertNotIn("timeout", schema["required"])

    def test_tool_registry_initializes_model_capabilities_slot(self):
        registry = ToolRegistry(self._make_config())
        self.assertEqual(registry.model_capabilities, DEFAULT_MODEL_CAPABILITIES)

    async def test_mcp_clients_are_registered_for_cleanup(self):
        tmp = self._workspace_tempdir()
        mcp_config_path = tmp / "mcp.json"
        mcp_config_path.write_text("{}", encoding="utf-8")
        registry = ToolRegistry(self._make_config(MCP_CONFIG_PATH=mcp_config_path))
        fake_client = mock.AsyncMock()
        fake_tool = SimpleNamespace(
            name="context7:resolve-library-id",
            description="Resolve a Context7 library id",
            metadata={"readOnlyHint": True},
        )

        with (
            mock.patch.object(ToolRegistry, "_read_mcp_config", return_value={"context7": {"enabled": True}}),
            mock.patch.object(
                ToolRegistry,
                "_load_single_mcp_server",
                new=mock.AsyncMock(return_value=("context7", fake_client, [fake_tool], None)),
            ),
        ):
            await registry.load_all()

        registry.set_tool_enabled("context7:resolve-library-id", False)

        self.assertEqual(registry.mcp_clients, [fake_client])
        self.assertIn("context7:resolve-library-id", {tool.name for tool in registry.active_tools()})
        await registry.cleanup()
        fake_client.aclose.assert_awaited_once()

    async def test_failed_mcp_server_is_disabled_in_tools_snapshot(self):
        tmp = self._workspace_tempdir()
        mcp_config_path = tmp / "mcp.json"
        mcp_config_path.write_text(
            json.dumps({"broken": {"command": "missing-mcp", "enabled": True}}),
            encoding="utf-8",
        )
        registry = ToolRegistry(self._make_config(MCP_CONFIG_PATH=mcp_config_path))

        with mock.patch.object(
            ToolRegistry,
            "_load_single_mcp_server",
            new=mock.AsyncMock(return_value=("broken", None, None, RuntimeError("startup failed"))),
        ):
            await registry.load_all()

        self.assertEqual(
            registry.mcp_server_status,
            [{"server": "broken", "loaded_tools": [], "error": "startup failed", "enabled": True}],
        )
        self.assertTrue(registry.mcp_config["broken"]["enabled"])
        server_row = next(row for row in build_tools_snapshot(registry) if row.get("name") == "broken")
        self.assertFalse(server_row["enabled"])
        self.assertEqual(server_row["description"], "MCP server - error: startup failed")

    async def test_tool_registry_cleanup_awaits_registered_async_callbacks(self):
        registry = ToolRegistry(self._make_config())
        callback = mock.AsyncMock()
        registry.register_cleanup_callback(callback)

        await registry.cleanup()

        callback.assert_awaited_once()

    async def test_tool_registry_cleanup_awaits_sync_close_returning_coroutine(self):
        registry = ToolRegistry(self._make_config())
        closed = False

        async def close_impl():
            nonlocal closed
            closed = True

        client = SimpleNamespace(close=mock.Mock(side_effect=close_impl))
        registry.mcp_clients.append(client)

        await registry.cleanup()

        client.close.assert_called_once()
        self.assertTrue(closed)

    def test_mcp_metadata_keeps_safe_tools_read_only(self):
        metadata = ToolRegistry._infer_mcp_metadata(
            "context7:resolve-library-id",
            server_policy={"read_only": True},
        )

        self.assertTrue(metadata.read_only)
        self.assertFalse(metadata.mutating)
        self.assertFalse(metadata.destructive)
        self.assertFalse(metadata.requires_approval)
        self.assertTrue(metadata.networked)

    def test_mcp_metadata_requires_approval_without_explicit_policy(self):
        metadata = ToolRegistry._infer_mcp_metadata("filesystem:write_file")

        self.assertFalse(metadata.read_only)
        self.assertTrue(metadata.mutating)
        self.assertFalse(metadata.destructive)
        self.assertTrue(metadata.requires_approval)
        self.assertTrue(metadata.networked)

    def test_mcp_metadata_without_policy_uses_approval_even_for_execution_hint(self):
        tool = SimpleNamespace(
            name="terminal:run_command",
            description="Run a command",
            metadata={"executionHint": True},
        )

        metadata = ToolRegistry._infer_mcp_metadata(tool)

        self.assertFalse(metadata.read_only)
        self.assertTrue(metadata.mutating)
        self.assertFalse(metadata.destructive)
        self.assertTrue(metadata.requires_approval)
        self.assertTrue(metadata.networked)

    def test_mcp_metadata_without_policy_keeps_readonly_hint_without_approval(self):
        tool = SimpleNamespace(
            name="acme_search_docs",
            description="Search documentation pages",
            metadata={"readOnlyHint": True, "openWorldHint": True},
        )

        metadata = ToolRegistry._infer_mcp_metadata(tool)

        self.assertTrue(metadata.read_only)
        self.assertFalse(metadata.mutating)
        self.assertFalse(metadata.destructive)
        self.assertFalse(metadata.requires_approval)

    def test_mcp_metadata_applies_server_read_only_policy(self):
        metadata = ToolRegistry._infer_mcp_metadata(
            "acme:write_file",
            server_policy={"read_only": True},
        )

        self.assertTrue(metadata.read_only)
        self.assertFalse(metadata.mutating)
        self.assertFalse(metadata.destructive)
        self.assertFalse(metadata.requires_approval)
        self.assertTrue(metadata.networked)

    def test_mcp_metadata_tool_override_can_disable_read_only(self):
        tool = SimpleNamespace(
            name="acme:docs_search",
            description="Search docs",
            metadata={"readOnlyHint": True},
        )

        metadata = ToolRegistry._infer_mcp_metadata(
            tool,
            server_policy={"read_only": True},
            tool_policy={"read_only": False},
        )

        self.assertFalse(metadata.read_only)
        self.assertTrue(metadata.mutating)
        self.assertFalse(metadata.destructive)
        self.assertTrue(metadata.requires_approval)

    def test_mcp_metadata_without_policy_uses_destructive_hint_for_shape(self):
        tool = SimpleNamespace(
            name="acme_workspace",
            description="Manage workspace entries",
            metadata={"destructiveHint": True},
        )

        metadata = ToolRegistry._infer_mcp_metadata(tool)

        self.assertFalse(metadata.read_only)
        self.assertTrue(metadata.mutating)
        self.assertTrue(metadata.destructive)
        self.assertTrue(metadata.requires_approval)

    async def test_mcp_loader_applies_policy_overrides_from_config(self):
        tmp = self._workspace_tempdir()
        mcp_config_path = tmp / "mcp.json"
        mcp_config_path.write_text(
            json.dumps(
                {
                    "acme": {
                        "enabled": True,
                        "policy": {
                            "read_only": True,
                            "tools": {
                                "terminal:run_command": {
                                    "read_only": False,
                                }
                            },
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        registry = ToolRegistry(self._make_config(MCP_CONFIG_PATH=mcp_config_path))
        fake_client = mock.AsyncMock()
        fake_tool = SimpleNamespace(
            name="terminal:run_command",
            description="Run a command",
            metadata={},
        )

        with mock.patch.object(
            ToolRegistry,
            "_load_single_mcp_server",
            new=mock.AsyncMock(return_value=("acme", fake_client, [fake_tool], None)),
        ):
            await registry.load_all()

        metadata = registry.tool_metadata["terminal:run_command"]
        self.assertFalse(metadata.read_only)
        self.assertTrue(metadata.mutating)
        self.assertTrue(metadata.requires_approval)
        self.assertTrue(metadata.networked)

    def test_validation_supports_delete_argument_aliases(self):
        tmp = self._workspace_tempdir()
        file_path = tmp / "data.txt"
        file_path.write_text("demo", encoding="utf-8")
        error = validate_tool_result("safe_delete_file", {"file_path": str(file_path)}, "Success")
        self.assertIsNotNone(error)
        self.assertIn("still exists", error)

        dir_path = tmp / "folder"
        dir_path.mkdir()
        error = validate_tool_result("safe_delete_directory", {"dir_path": str(dir_path)}, "Success")
        self.assertIsNotNone(error)
        self.assertIn("still exists", error)

    def test_validation_does_not_require_local_file_for_successful_edit_or_write(self):
        self.assertIsNone(
            validate_tool_result(
                "edit_file",
                {"path": "virtual.txt", "old_string": "a", "new_string": "b"},
                "Success: File edited.",
            )
        )
        self.assertIsNone(
            validate_tool_result(
                "write_file",
                {"path": "virtual.txt", "content": "hello"},
                "Success: File written.",
            )
        )

    def test_request_user_input_schema_normalizes_payload(self):
        schema = request_user_input.get_input_schema()
        validated = schema.model_validate(
            {
                "question": "  Какой режим выбрать?  ",
                "options": [" direct_api ", "keep_mcp", "direct_api", " "],
                "recommended": " direct_api ",
            }
        )

        self.assertEqual(validated.question, "Какой режим выбрать?")
        self.assertEqual(validated.options, ["direct_api", "keep_mcp"])
        self.assertEqual(validated.recommended, "direct_api")

    def test_request_user_input_schema_rejects_invalid_payload(self):
        schema = request_user_input.get_input_schema()
        with self.assertRaises(Exception):
            schema.model_validate(
                {
                    "question": "Выбери вариант",
                    "options": ["only_one"],
                    "recommended": "missing",
                }
            )

    def test_fetch_content_schema_uses_array_urls_for_gemini_compatibility(self):
        schema = fetch_content.get_input_schema().model_json_schema()
        urls_schema = schema["properties"]["urls"]

        self.assertEqual(urls_schema["type"], "array")
        self.assertEqual(urls_schema["items"]["type"], "string")
        self.assertNotIn("anyOf", urls_schema)

    def test_fetch_content_schema_normalizes_single_string_url_into_list(self):
        schema = fetch_content.get_input_schema()
        validated = schema.model_validate({"urls": "https://example.com"})

        self.assertEqual(validated.urls, ["https://example.com"])

    def test_fetch_content_schema_declares_tavily_batch_limits(self):
        schema = fetch_content.get_input_schema().model_json_schema()
        urls_schema = schema["properties"]["urls"]
        chunks_schema = schema["properties"]["chunks_per_source"]

        self.assertEqual(urls_schema["minItems"], 1)
        self.assertEqual(urls_schema["maxItems"], 20)
        self.assertEqual(chunks_schema["minimum"], 1)
        self.assertEqual(chunks_schema["maximum"], 5)
        self.assertIn("batch", urls_schema["description"].lower())

    async def test_fetch_content_sends_all_urls_in_one_tavily_batch_call(self):
        client = SimpleNamespace(extract=mock.AsyncMock())
        client.extract.return_value = {
            "results": [
                {"url": "https://example.com/one", "raw_content": "One"},
                {"url": "https://example.com/two", "raw_content": "Two"},
            ],
            "failed_results": [],
        }

        with mock.patch.object(_RUNTIME, "get_client", return_value=client):
            result = await fetch_content.ainvoke(
                {
                    "urls": [
                        "https://example.com/one",
                        "https://example.com/two",
                    ],
                    "query": "main points",
                    "chunks_per_source": 5,
                }
            )

        self.assertIn("https://example.com/one", result)
        self.assertIn("https://example.com/two", result)
        client.extract.assert_awaited_once_with(
            urls=["https://example.com/one", "https://example.com/two"],
            extract_depth="basic",
            format="markdown",
            query="main points",
            chunks_per_source=5,
        )

    async def test_fetch_content_raw_limit_preserves_all_sources(self):
        client = SimpleNamespace(extract=mock.AsyncMock())
        client.extract.return_value = {
            "results": [
                {"url": f"https://example.com/{index}", "raw_content": str(index) * 2000}
                for index in range(3)
            ],
            "failed_results": [],
        }
        previous_policy = _RUNTIME.safety_policy
        self.addCleanup(lambda: setattr(_RUNTIME, "safety_policy", previous_policy))
        _RUNTIME.safety_policy = SafetyPolicy(max_tool_output=500, max_raw_tool_output=1200)

        with mock.patch.object(_RUNTIME, "get_client", return_value=client):
            result = await fetch_content.ainvoke(
                {"urls": [f"https://example.com/{index}" for index in range(3)]}
            )

        self.assertLessEqual(len(result), 1200)
        self.assertGreater(len(result), 500)
        for index in range(3):
            self.assertIn(f"=== SOURCE: https://example.com/{index} ===", result)

    async def test_crawl_site_sends_supported_tavily_options_and_formats_pages(self):
        client = SimpleNamespace(crawl=mock.AsyncMock())
        client.crawl.return_value = {
            "results": [
                {
                    "url": "https://example.com/docs",
                    "raw_content": "Documentation content",
                    "images": ["https://example.com/image.png"],
                }
            ],
            "usage": {"credits": 2},
        }

        with mock.patch.object(_RUNTIME, "get_client", return_value=client):
            result = await crawl_site.ainvoke(
                {
                    "url": " https://example.com ",
                    "max_depth": 99,
                    "max_breadth": 999,
                    "limit": 9999,
                    "instructions": " Find API docs ",
                    "select_paths": ["/docs/.*"],
                    "select_domains": ["example\\.com"],
                    "exclude_paths": ["/blog/.*"],
                    "exclude_domains": ["ads\\.example\\.com"],
                    "allow_external": True,
                    "include_images": True,
                    "advanced": True,
                    "content_format": "md",
                    "timeout": 150,
                    "chunks_per_source": 5,
                }
            )

        self.assertIn("https://example.com/docs", result)
        self.assertIn("Documentation content", result)
        self.assertIn("https://example.com/image.png", result)
        self.assertIn("Usage:", result)
        client.crawl.assert_awaited_once_with(
            url="https://example.com",
            max_depth=99,
            max_breadth=999,
            limit=9999,
            instructions="Find API docs",
            select_paths=["/docs/.*"],
            select_domains=["example\\.com"],
            exclude_paths=["/blog/.*"],
            exclude_domains=["ads\\.example\\.com"],
            allow_external=True,
            include_images=True,
            extract_depth="advanced",
            format="markdown",
            timeout=150.0,
            include_usage=True,
            chunks_per_source=5,
        )

    async def test_crawl_site_omits_chunks_without_instructions(self):
        client = SimpleNamespace(crawl=mock.AsyncMock())
        client.crawl.return_value = {"results": [{"url": "https://example.com", "raw_content": "Home"}]}

        with mock.patch.object(_RUNTIME, "get_client", return_value=client):
            await crawl_site.ainvoke({"url": "https://example.com", "chunks_per_source": 5})

        self.assertNotIn("chunks_per_source", client.crawl.await_args.kwargs)

    def test_fetch_content_parser_accepts_url_lists_and_discards_invalid_entries(self):
        raw_urls = [
            "https://openai.com/index/introducing-gpt-5-5/,",
            "not-a-url",
            "https://openai.com/index/introducing-gpt-5-5/",
            "https://example.com/docs",
        ]

        self.assertEqual(
            _parse_urls_input(raw_urls),
            [
                "https://openai.com/index/introducing-gpt-5-5/",
                "https://example.com/docs",
            ],
        )

    def test_fetch_content_parser_accepts_stringified_url_list(self):
        raw_urls = "['https://openai.com/index/introducing-gpt-5-5/', 'https://example.com/docs']"

        self.assertEqual(
            _parse_urls_input(raw_urls),
            [
                "https://openai.com/index/introducing-gpt-5-5/",
                "https://example.com/docs",
            ],
        )

    def test_tavily_error_formatter_logs_a_warning(self):
        from tavily import errors as tavily_errors

        forbidden = tavily_errors.ForbiddenError("403 Forbidden for https://example.com")

        with self.assertLogs("tools.search_tools", level="WARNING") as captured:
            formatted = _format_tavily_error(forbidden)

        self.assertIn("ACCESS_DENIED", formatted)
        self.assertEqual(len(captured.records), 1)
        self.assertIn("ForbiddenError", captured.output[0])
        self.assertIn("403 Forbidden for https://example.com", captured.output[0])

    def test_run_background_process_schema_uses_array_command_for_gemini_compatibility(self):
        schema = run_background_process.get_input_schema().model_json_schema()
        command_schema = schema["properties"]["command"]

        self.assertEqual(command_schema["type"], "array")
        self.assertEqual(command_schema["items"]["type"], "string")
        self.assertNotIn("anyOf", command_schema)

    def test_run_background_process_schema_normalizes_single_command_string_into_list(self):
        schema = run_background_process.get_input_schema()
        validated = schema.model_validate({"command": "python -V"})

        self.assertEqual(validated.command, ["python -V"])

    def test_max_file_size_numeric_value_is_bytes(self):
        config = self._make_config(MAX_FILE_SIZE="4096")
        self.assertEqual(config.max_file_size, 4096)

    def test_max_file_size_supports_explicit_units(self):
        self.assertEqual(self._make_config(MAX_FILE_SIZE="4MB").max_file_size, 4_000_000)
        self.assertEqual(self._make_config(MAX_FILE_SIZE="300MiB").max_file_size, 300 * 1024 * 1024)

    def test_max_file_size_rejects_invalid_strings(self):
        with self.assertRaises(Exception):
            self._make_config(MAX_FILE_SIZE="300MBps")

    def test_unknown_sampling_env_keys_are_ignored(self):
        config = self._make_config(TOP_P="none", TOP_K="")

        self.assertFalse(hasattr(config, "top_p"))
        self.assertFalse(hasattr(config, "top_k"))

    def test_logging_env_keys_are_loaded_via_agent_config(self):
        config = self._make_config(
            LOG_LEVEL="debug",
            LOG_FILE="logs/custom-agent.log",
            DEBUG_REASONING_STREAM=True,
            ENABLE_TEXT_TOOL_CALL_RECOVERY=True,
        )
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.log_file.name, "custom-agent.log")
        self.assertTrue(config.debug_reasoning_stream)
        self.assertTrue(config.enable_text_tool_call_recovery)

    def test_provider_registry_path_is_resolved_from_project_root(self):
        config = self._make_config(PROVIDER_REGISTRY_PATH="provider_registry.json")

        self.assertTrue(config.provider_registry_path.is_absolute())
        self.assertEqual(config.provider_registry_path.name, "provider_registry.json")
        self.assertTrue(config.provider_registry_path.exists())

    def test_text_utils_import_does_not_require_prompt_toolkit(self):
        import core.text_utils as text_utils

        with mock.patch.dict(sys.modules, {"prompt_toolkit": None, "prompt_toolkit.key_binding": None}):
            reloaded = importlib.reload(text_utils)
            self.assertTrue(callable(reloaded.prepare_markdown_for_render))

    def test_run_background_process_rejects_shell_operators(self):
        result = process_tools.run_background_process.invoke({"command": "python -c \"print(1)\" && whoami"})
        self.assertIn("ERROR[VALIDATION]", result)
        self.assertIn("Shell operators are not allowed", result)

    def test_run_background_process_rejects_cwd_outside_workspace(self):
        tmp = self._workspace_tempdir()
        process_tools.set_working_directory(str(tmp))
        result = process_tools.run_background_process.invoke(
            {"command": [sys.executable, "-c", "print('ok')"], "cwd": ".."}
        )
        self.assertIn("ERROR[VALIDATION]", result)
        self.assertIn("ACCESS DENIED", result)

    def test_run_background_process_accepts_argument_list(self):
        tmp = self._workspace_tempdir()
        process_tools.set_working_directory(str(tmp))
        result = process_tools.run_background_process.invoke(
            {"command": [sys.executable, "-c", "import time; time.sleep(30)"], "cwd": "."}
        )
        self.assertIn("Success: Process started with PID", result)
        match = re.search(r"PID (\d+)", result)
        self.assertIsNotNone(match)
        stop_result = process_tools.stop_background_process.invoke({"pid": int(match.group(1))})
        self.assertIn("Success:", stop_result)

    def test_find_process_by_port_uses_net_connections(self):
        class FakePsutil:
            class AccessDenied(Exception):
                pass

            class NoSuchProcess(Exception):
                pass

            class ZombieProcess(Exception):
                pass

            @staticmethod
            def net_connections(kind="inet"):
                return [SimpleNamespace(laddr=SimpleNamespace(port=8000), pid=4321)]

            @staticmethod
            def Process(pid):
                return SimpleNamespace(name=lambda: "python")

        with mock.patch.object(process_tools, "psutil", FakePsutil):
            result = process_tools.find_process_by_port.invoke({"port": 8000})
        self.assertIn("Found process 'python' (PID: 4321) on port 8000.", result)

    def test_find_process_by_port_returns_non_error_when_port_is_free(self):
        class FakePsutil:
            class AccessDenied(Exception):
                pass

            class NoSuchProcess(Exception):
                pass

            class ZombieProcess(Exception):
                pass

            @staticmethod
            def net_connections(kind="inet"):
                return []

        with mock.patch.object(process_tools, "psutil", FakePsutil):
            result = process_tools.find_process_by_port.invoke({"port": 8000})
        self.assertEqual(result, "No process found listening on port 8000.")
        self.assertNotIn("ERROR[", result)


if __name__ == "__main__":
    unittest.main()
