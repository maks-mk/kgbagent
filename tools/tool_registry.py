import asyncio
import importlib
import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Union

from langchain_core.tools import BaseTool

from core.config import AgentConfig
from core.multimodal import DEFAULT_MODEL_CAPABILITIES
from core.tool_policy import ToolMetadata, default_tool_metadata

logger = logging.getLogger(__name__)
_MCP_POLICY_FIELDS = (
    "read_only",
)
_BUILTIN_TOOLS_CONFIG_KEY = "_builtin_tools"


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").split())


@dataclass(frozen=True)
class ToolLoaderSpec:
    name: str
    enabled: Callable[[AgentConfig], bool]
    module_name: str
    tool_names: Sequence[str]
    configure: Callable[[Any, AgentConfig], None] | None = None
    metadata: Dict[str, ToolMetadata] | None = None


class ToolRegistry:
    __slots__ = (
        "config",
        "tools",
        "tool_metadata",
        "model_capabilities",
        "mcp_clients",
        "loader_status",
        "mcp_server_status",
        "checkpoint_info",
        "checkpoint_runtime",
        "_cleanup_callbacks",
        "_cleanup_callback_keys",
        "disabled_mcp_servers",
        "disabled_local_tools",
        "builtin_tools",
        "builtin_specs",
        "mcp_config",
    )

    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools: List[BaseTool] = []
        self.tool_metadata: Dict[str, ToolMetadata] = {}
        self.model_capabilities: Dict[str, Any] = dict(DEFAULT_MODEL_CAPABILITIES)
        self.mcp_clients = []
        self.loader_status: List[Dict[str, Any]] = []
        self.mcp_server_status: List[Dict[str, Any]] = []
        self.checkpoint_info: Dict[str, Any] = {}
        self.checkpoint_runtime: Any = None
        self._cleanup_callbacks: List[Callable[[], Any]] = []
        self._cleanup_callback_keys: set[tuple[int, int]] = set()
        self.disabled_mcp_servers: set[str] = set()
        self.disabled_local_tools: set[str] = set()
        self.builtin_tools: List[BaseTool] = []
        self.builtin_specs: Dict[str, ToolLoaderSpec] = {}
        self.mcp_config: Dict[str, Any] = {}

    async def load_all(self):
        if self.config.mcp_config_path.exists():
            self.mcp_config = self._read_mcp_config()
        builtin_states = self.mcp_config.get(_BUILTIN_TOOLS_CONFIG_KEY, {})
        if not isinstance(builtin_states, dict):
            builtin_states = {}

        builtin_defaults: dict[str, bool] = {}
        builtin_by_name: dict[str, BaseTool] = {}
        builtin_specs: dict[str, ToolLoaderSpec] = {}
        for spec in self._loader_specs():
            spec_enabled = bool(spec.enabled(self.config))
            loaded_tools = self._load_local_spec(spec, add_to_runtime=spec_enabled)
            for tool in loaded_tools:
                builtin_by_name.setdefault(tool.name, tool)
                builtin_specs.setdefault(tool.name, spec)
                builtin_defaults[tool.name] = builtin_defaults.get(tool.name, False) or spec_enabled

        self.builtin_tools = list(builtin_by_name.values())
        self.builtin_specs = dict(builtin_specs)
        for name, default_enabled in builtin_defaults.items():
            override = builtin_states.get(name)
            desired = default_enabled if not isinstance(override, bool) else override
            if desired:
                self._activate_builtin_tool(builtin_by_name[name])
                continue
            if default_enabled:
                self.disabled_local_tools.add(name)

        if self.config.mcp_config_path.exists():
            await self._load_mcp_tools()

    def set_mcp_server_enabled(self, server_name: str, enabled: bool) -> None:
        name = str(server_name or "").strip()
        if not name or not isinstance(self.mcp_config.get(name), dict):
            return
        updated_config = dict(self.mcp_config)
        updated_server_config = dict(self.mcp_config[name])
        updated_server_config["enabled"] = bool(enabled)
        updated_config[name] = updated_server_config
        try:
            self.config.mcp_config_path.write_text(
                json.dumps(updated_config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            logger.exception("Failed to persist MCP server state to %s", self.config.mcp_config_path)
            raise
        self.mcp_config = updated_config
        self.disabled_mcp_servers.discard(name) if enabled else self.disabled_mcp_servers.add(name)

    def set_tool_enabled(self, tool_name: str, enabled: bool) -> bool:
        """Apply a builtin tool override; returns True when the runtime catalog needs a rebuild."""
        name = str(tool_name or "").strip()
        mcp_tool_names = {
            tool_name
            for status in self.mcp_server_status
            for tool_name in status.get("loaded_tools", [])
        }
        tool = next((tool for tool in self.builtin_tools if tool.name == name), None)
        metadata = self.tool_metadata.get(name)
        is_mcp = bool(
            name in mcp_tool_names
            or (metadata and metadata.source == "mcp")
            or (tool is not None and hasattr(tool, "_is_mcp"))
            or ":" in name
        )
        if not name or tool is None or is_mcp:
            return False
        catalog_changed = False
        if enabled:
            self.disabled_local_tools.discard(name)
            catalog_changed = self._activate_builtin_tool(tool)
        else:
            self.disabled_local_tools.add(name)
        self._persist_builtin_tool_state(name, bool(enabled))
        return catalog_changed

    def _activate_builtin_tool(self, tool: BaseTool) -> bool:
        """Add a catalog-only builtin tool to the runtime catalog.

        The `_builtin_tools` override wins over the feature flag in both directions, so a tool whose
        loader flag is off in `.env` must still become usable when it is explicitly enabled.
        """
        if any(existing.name == tool.name for existing in self.tools):
            return False
        spec = self.builtin_specs.get(tool.name)
        if spec is not None and not spec.enabled(self.config):
            self._configure_spec_module(spec)
        self.tools.append(tool)
        return True

    def _configure_spec_module(self, spec: ToolLoaderSpec) -> None:
        if not spec.configure:
            return
        try:
            module = importlib.import_module(spec.module_name)
            spec.configure(module, self.config)
        except Exception:
            logger.debug("Failed to configure %s tools on activation.", spec.name, exc_info=True)

    def _persist_builtin_tool_state(self, name: str, enabled: bool) -> None:
        config = dict(self.mcp_config)
        states = dict(config.get(_BUILTIN_TOOLS_CONFIG_KEY, {}))
        states[name] = bool(enabled)
        config[_BUILTIN_TOOLS_CONFIG_KEY] = states
        try:
            self.config.mcp_config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            logger.exception("Failed to persist built-in tool state to %s", self.config.mcp_config_path)
            raise
        self.mcp_config = config

    def available_tools(self) -> List[BaseTool]:
        """Return the loaded runtime tool catalog, including locally disabled tools."""
        return list(self.tools)

    def active_tools(self) -> List[BaseTool]:
        server_by_tool = {
            tool_name: status.get("server", "")
            for status in self.mcp_server_status
            for tool_name in status.get("loaded_tools", [])
        }
        return [
            tool for tool in self.available_tools()
            if server_by_tool.get(tool.name, "") not in self.disabled_mcp_servers
            and tool.name not in self.disabled_local_tools
        ]

    def sync_working_directory(self, cwd: str | Path | None = None) -> None:
        """Propagate runtime cwd to local tool modules that cache workspace roots."""
        target_cwd = str(Path(cwd or Path.cwd()).resolve())
        for module_name in ("tools.filesystem", "tools.local_shell", "tools.process_tools"):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            setter = getattr(module, "set_working_directory", None)
            if callable(setter):
                try:
                    setter(target_cwd)
                except Exception:
                    logger.debug("Failed to sync cwd for %s", module_name, exc_info=True)

    def reconfigure(self, config: AgentConfig) -> None:
        """Refresh module-level runtime config without reloading tools or MCP clients."""
        self.config = config
        for spec in self._loader_specs():
            if not spec.enabled(self.config) or not spec.configure:
                continue
            try:
                module = importlib.import_module(spec.module_name)
            except Exception:
                logger.debug("Failed to import %s during tool registry reconfigure.", spec.module_name, exc_info=True)
                continue
            try:
                spec.configure(module, self.config)
            except Exception:
                logger.debug("Failed to reconfigure %s tools.", spec.name, exc_info=True)

    def _loader_specs(self) -> List[ToolLoaderSpec]:
        return [
            ToolLoaderSpec(
                name="filesystem",
                enabled=lambda config: config.enable_filesystem_tools,
                module_name="tools.filesystem",
                tool_names=(
                    "read_file_tool",
                    "write_file_tool",
                    "edit_file_tool",
                    "list_directory_tool",
                    "safe_delete_file",
                    "safe_delete_directory",
                    "download_file",
                ),
                configure=self._configure_safety,
                metadata={
                    "read_file_tool": ToolMetadata(name="read_file", read_only=True),
                    "write_file_tool": ToolMetadata(
                        name="write_file", mutating=True, requires_approval=True
                    ),
                    "edit_file_tool": ToolMetadata(
                        name="edit_file", mutating=True, requires_approval=True
                    ),
                    "list_directory_tool": ToolMetadata(name="list_directory", read_only=True),
                    "safe_delete_file": ToolMetadata(
                        name="safe_delete_file",
                        mutating=True,
                        destructive=True,
                        requires_approval=True,
                    ),
                    "safe_delete_directory": ToolMetadata(
                        name="safe_delete_directory",
                        mutating=True,
                        destructive=True,
                        requires_approval=True,
                    ),
                    "download_file": ToolMetadata(
                        name="download_file",
                        mutating=True,
                        networked=True,
                        requires_approval=True,
                    ),
                },
            ),
            ToolLoaderSpec(
                name="filesystem_delete",
                enabled=lambda config: not config.enable_filesystem_tools,
                module_name="tools.filesystem",
                tool_names=("safe_delete_file", "safe_delete_directory"),
                configure=self._configure_safety,
                metadata={
                    "safe_delete_file": ToolMetadata(
                        name="safe_delete_file",
                        mutating=True,
                        destructive=True,
                        requires_approval=True,
                    ),
                    "safe_delete_directory": ToolMetadata(
                        name="safe_delete_directory",
                        mutating=True,
                        destructive=True,
                        requires_approval=True,
                    ),
                },
            ),
            ToolLoaderSpec(
                name="search",
                enabled=lambda config: config.enable_search_tools,
                module_name="tools.search_tools",
                tool_names=("batch_web_search", "fetch_content", "crawl_site"),
                configure=self._configure_search,
                metadata={
                    "batch_web_search": ToolMetadata(
                        name="batch_web_search", read_only=True, networked=True
                    ),
                    "fetch_content": ToolMetadata(
                        name="fetch_content", read_only=True, networked=True
                    ),
                    "crawl_site": ToolMetadata(
                        name="crawl_site", read_only=True, networked=True
                    ),
                },
            ),
            ToolLoaderSpec(
                name="process",
                enabled=lambda config: config.enable_process_tools,
                module_name="tools.process_tools",
                tool_names=("run_background_process", "stop_background_process", "find_process_by_port"),
                configure=self._configure_safety,
                metadata={
                    "run_background_process": ToolMetadata(
                        name="run_background_process",
                        mutating=True,
                        requires_approval=True,
                    ),
                    "stop_background_process": ToolMetadata(
                        name="stop_background_process",
                        mutating=True,
                        destructive=True,
                        requires_approval=True,
                    ),
                    "find_process_by_port": ToolMetadata(
                        name="find_process_by_port", read_only=True
                    ),
                },
            ),
            ToolLoaderSpec(
                name="shell",
                enabled=lambda config: getattr(config, "enable_shell_tool", False),
                module_name="tools.local_shell",
                tool_names=("cli_exec",),
                configure=self._configure_shell,
                metadata={
                    "cli_exec": ToolMetadata(
                        name="cli_exec",
                        mutating=True,
                    )
                },
            ),
            ToolLoaderSpec(
                name="user_input",
                enabled=lambda config: True,
                module_name="tools.user_input_tool",
                tool_names=("request_user_input",),
                metadata={
                    "request_user_input": ToolMetadata(
                        name="request_user_input",
                        read_only=True,
                        requires_approval=False,
                    )
                },
            ),
        ]

    @staticmethod
    def _metadata_for_spec_attr(spec: ToolLoaderSpec, attr_name: str, tool_name: str) -> ToolMetadata:
        metadata = (spec.metadata or {}).get(attr_name)
        if metadata:
            return metadata
        return default_tool_metadata(tool_name)

    @staticmethod
    def _optimize_tool_description(tool: BaseTool) -> None:
        """Minimize tool metadata sent to the model without changing call shape."""
        compact_description = _compact_text(getattr(tool, "description", ""))
        if compact_description:
            try:
                tool.description = compact_description
            except Exception:
                logger.debug("Failed to compact description for tool %s", getattr(tool, "name", ""), exc_info=True)

        args_schema = getattr(tool, "args_schema", None)
        schema_doc = _compact_text(getattr(args_schema, "__doc__", ""))
        if args_schema is not None and compact_description and schema_doc == compact_description:
            try:
                args_schema.__doc__ = None
            except Exception:
                logger.debug("Failed to strip duplicate args schema doc for %s", getattr(tool, "name", ""), exc_info=True)

    def _record_loader_status(
        self,
        *,
        spec: ToolLoaderSpec,
        loaded_tools: List[BaseTool],
        error: str = "",
    ) -> None:
        self.loader_status.append(
            {
                "loader": spec.name,
                "module": spec.module_name,
                "loaded_tools": [tool.name for tool in loaded_tools],
                "error": error,
            }
        )

    def _load_local_spec(self, spec: ToolLoaderSpec, *, add_to_runtime: bool) -> List[BaseTool]:
        try:
            module = importlib.import_module(spec.module_name)
            if spec.configure and spec.enabled(self.config):
                spec.configure(module, self.config)

            catalog_tools: List[BaseTool] = []
            loaded_tools: List[BaseTool] = []
            existing_names = {tool.name for tool in self.tools}
            for attr_name in spec.tool_names:
                tool = getattr(module, attr_name)
                self._optimize_tool_description(tool)
                catalog_tools.append(tool)
                if add_to_runtime and tool.name not in existing_names:
                    loaded_tools.append(tool)
                    existing_names.add(tool.name)
                metadata = self._metadata_for_spec_attr(spec, attr_name, tool.name)
                self.tool_metadata[tool.name] = metadata
            self.tools.extend(loaded_tools)
            self._record_loader_status(spec=spec, loaded_tools=loaded_tools)
            return catalog_tools
        except Exception as e:
            self._record_loader_status(spec=spec, loaded_tools=[], error=str(e))
            logger.exception("Failed to load %s tools: %s", spec.name, e)
            return []

    @staticmethod
    def _configure_safety(module: Any, config: AgentConfig) -> None:
        if hasattr(module, "set_safety_policy"):
            module.set_safety_policy(config.safety)
        if hasattr(module, "set_working_directory"):
            module.set_working_directory(str(Path.cwd()))

    @staticmethod
    def _configure_search(module: Any, config: AgentConfig) -> None:
        if hasattr(module, "set_safety_policy"):
            module.set_safety_policy(config.safety)
        if hasattr(module, "set_runtime_config"):
            module.set_runtime_config(config)

    @staticmethod
    def _configure_shell(module: Any, config: AgentConfig) -> None:
        ToolRegistry._configure_safety(module, config)

    @staticmethod
    def _metadata_flag(raw_metadata: Dict[str, Any], *keys: str) -> bool | None:
        for key in keys:
            if key in raw_metadata:
                return bool(raw_metadata.get(key))
        return None

    @classmethod
    def _extract_mcp_metadata_hints(cls, raw_metadata: Dict[str, Any]) -> Dict[str, bool]:
        if not isinstance(raw_metadata, dict):
            return {}

        flags: Dict[str, bool] = {}
        read_only_hint = cls._metadata_flag(raw_metadata, "readOnlyHint", "read_only", "readOnly")
        destructive_hint = cls._metadata_flag(raw_metadata, "destructiveHint", "destructive", "is_destructive")
        mutating_hint = cls._metadata_flag(raw_metadata, "mutatingHint", "mutating", "writes")
        execution_hint = cls._metadata_flag(raw_metadata, "executionHint", "executes", "runsCommands")
        requires_approval_hint = cls._metadata_flag(raw_metadata, "requiresApproval", "approvalRequired")
        networked_hint = cls._metadata_flag(raw_metadata, "networkHint", "networked", "usesNetwork")

        if read_only_hint is not None:
            flags["read_only"] = bool(read_only_hint)
        if destructive_hint is not None:
            flags["destructive"] = bool(destructive_hint)
        if mutating_hint is not None:
            flags["mutating"] = bool(mutating_hint)
        if execution_hint:
            flags["mutating"] = True
        if requires_approval_hint is not None:
            flags["requires_approval"] = bool(requires_approval_hint)
        if networked_hint is not None:
            flags["networked"] = bool(networked_hint)
        return flags

    @staticmethod
    def _sanitize_mcp_policy_flags(policy: Dict[str, Any] | None) -> Dict[str, bool]:
        if not isinstance(policy, dict):
            return {}
        return {
            field: bool(policy[field])
            for field in _MCP_POLICY_FIELDS
            if field in policy
        }

    @classmethod
    def _split_mcp_policy_config(cls, cfg: Dict[str, Any] | None) -> tuple[Dict[str, bool], Dict[str, Dict[str, bool]]]:
        if not isinstance(cfg, dict):
            return {}, {}

        raw_policy = cfg.get("policy")
        if not isinstance(raw_policy, dict):
            return {}, {}

        server_policy = cls._sanitize_mcp_policy_flags(raw_policy)
        tool_policies: Dict[str, Dict[str, bool]] = {}
        raw_tools = raw_policy.get("tools")
        if isinstance(raw_tools, dict):
            for tool_name, tool_policy in raw_tools.items():
                normalized_name = str(tool_name or "").strip()
                if not normalized_name:
                    continue
                normalized_policy = cls._sanitize_mcp_policy_flags(tool_policy)
                if normalized_policy:
                    tool_policies[normalized_name] = normalized_policy
        return server_policy, tool_policies

    @staticmethod
    def _apply_mcp_policy_flags(
        state: Dict[str, bool],
        flags: Dict[str, bool],
        *,
        approval_explicit: bool,
    ) -> bool:
        if not flags:
            return approval_explicit

        if "read_only" in flags:
            state["read_only"] = bool(flags["read_only"])
            if state["read_only"]:
                state["mutating"] = False
                state["destructive"] = False
            else:
                state["mutating"] = True
        state["requires_approval"] = not bool(state["read_only"])
        approval_explicit = True
        return approval_explicit

    @classmethod
    def _infer_mcp_metadata(
        cls,
        tool: BaseTool | str,
        *,
        server_policy: Dict[str, Any] | None = None,
        tool_policy: Dict[str, Any] | None = None,
    ) -> ToolMetadata:
        if isinstance(tool, str):
            tool_name = tool
            raw_metadata = {}
        else:
            tool_name = str(getattr(tool, "name", "") or "")
            candidate_metadata = getattr(tool, "metadata", None)
            raw_metadata = candidate_metadata if isinstance(candidate_metadata, dict) else {}
        base_metadata = default_tool_metadata(tool_name, source="mcp")
        state = {
            "read_only": bool(base_metadata.read_only),
            "mutating": bool(base_metadata.mutating),
            "destructive": bool(base_metadata.destructive),
            "requires_approval": bool(base_metadata.requires_approval),
            "networked": bool(base_metadata.networked),
        }
        approval_explicit = False
        server_flags = cls._sanitize_mcp_policy_flags(server_policy)
        hint_flags = cls._extract_mcp_metadata_hints(raw_metadata)
        tool_flags = cls._sanitize_mcp_policy_flags(tool_policy)

        for flags in (server_flags, tool_flags):
            approval_explicit = cls._apply_mcp_policy_flags(
                state,
                flags,
                approval_explicit=approval_explicit,
            )

        if "networked" in hint_flags:
            state["networked"] = bool(hint_flags["networked"])

        if not approval_explicit:
            if hint_flags.get("destructive"):
                state["destructive"] = True
                state["mutating"] = True
                state["read_only"] = False
            elif hint_flags.get("mutating"):
                state["mutating"] = True
                state["read_only"] = False
            elif "read_only" in hint_flags:
                state["read_only"] = bool(hint_flags["read_only"])
                if state["read_only"]:
                    state["mutating"] = False
                    state["destructive"] = False
                else:
                    state["mutating"] = True
            state["requires_approval"] = not bool(state["read_only"])

        return ToolMetadata(
            name=tool_name,
            read_only=bool(state["read_only"]),
            mutating=bool(state["mutating"]),
            destructive=bool(state["destructive"]),
            requires_approval=bool(state["requires_approval"]),
            networked=bool(state["networked"]),
            source="mcp",
        )

    async def _load_single_mcp_server(
        self,
        name: str,
        cfg: Dict[str, Any],
        valid_keys: set[str],
        semaphore: asyncio.Semaphore,
    ):
        async with semaphore:
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient

                server_config = {key: value for key, value in cfg.items() if key in valid_keys}
                client = MultiServerMCPClient({name: server_config})
                return name, client, await client.get_tools(), None
            except BaseException as e:
                return name, None, None, e

    @staticmethod
    def _format_mcp_error(err: BaseException) -> str:
        parts = [str(err)]
        seen = set()
        stack = [err]
        while stack:
            exc = stack.pop()
            if id(exc) in seen:
                continue
            seen.add(id(exc))
            for sub in getattr(exc, "exceptions", []) or []:
                cause = f"{type(sub).__name__}: {sub}"
                if cause not in parts:
                    parts.append(cause)
                stack.append(sub)
        return " | ".join(parts)

    async def _load_mcp_tools(self):
        try:
            raw_cfg = self._read_mcp_config()
            self.mcp_config = raw_cfg
            runtime_cfg = self._expand_env_vars(raw_cfg)
            enabled_servers = []
            for name, cfg in raw_cfg.items():
                if name == _BUILTIN_TOOLS_CONFIG_KEY:
                    continue
                if not isinstance(cfg, dict):
                    logger.warning(
                        "⚠ Skipping invalid config entry '%s': Expected dict, got %s",
                        name,
                        type(cfg).__name__,
                    )
                    continue
                enabled = bool(cfg.get("enabled", True))
                if enabled:
                    enabled_servers.append((name, runtime_cfg[name]))
                else:
                    self.disabled_mcp_servers.add(name)
                    self.mcp_server_status.append(
                        {"server": name, "loaded_tools": [], "error": "", "enabled": False}
                    )

            if not enabled_servers:
                logger.debug("No enabled MCP servers in config.")
                return

            valid_keys = {
                "command",
                "args",
                "env",
                "cwd",
                "encoding",
                "encoding_error_handler",
                "url",
                "headers",
                "timeout",
                "sse_read_timeout",
                "auth",
                "terminate_on_close",
                "httpx_client_factory",
                "transport",
                "session_kwargs",
            }

            semaphore = asyncio.Semaphore(4)
            results = await asyncio.gather(
                *(
                    self._load_single_mcp_server(name, cfg, valid_keys, semaphore)
                    for name, cfg in enabled_servers
                )
            )
            for name, client, mcp_tools, err in results:
                if err is not None:
                    error_text = self._format_mcp_error(err)
                    self.mcp_server_status.append({"server": name, "loaded_tools": [], "error": error_text, "enabled": True})
                    logger.error("❌ MCP Server '%s' Error: %s", name, error_text)
                    continue

                server_cfg = raw_cfg.get(name) if isinstance(raw_cfg.get(name), dict) else {}
                server_policy, tool_policies = self._split_mcp_policy_config(server_cfg)

                if client is not None:
                    self.mcp_clients.append(client)
                if mcp_tools:
                    self.tools.extend(mcp_tools)
                    for tool in mcp_tools:
                        self._optimize_tool_description(tool)
                        metadata = self._infer_mcp_metadata(
                            tool,
                            server_policy=server_policy,
                            tool_policy=tool_policies.get(tool.name),
                        )
                        self.tool_metadata[tool.name] = metadata
                    self.mcp_server_status.append(
                        {
                            "server": name,
                            "loaded_tools": [tool.name for tool in mcp_tools],
                            "error": "",
                            "enabled": True,
                        }
                    )
                    logger.info("✔ MCP Server '%s': Loaded %s tools", name, len(mcp_tools))
                else:
                    self.mcp_server_status.append({"server": name, "loaded_tools": [], "error": "No tools found", "enabled": True})
                    logger.warning("⚠ MCP Server '%s': No tools found", name)
        except Exception as e:
            logger.exception(f"Failed to load MCP tools: {e}")

    def register_cleanup_callback(self, callback: Callable[[], Any]) -> None:
        target = getattr(callback, "__self__", None)
        function = getattr(callback, "__func__", callback)
        key = (id(target), id(function))
        if key in self._cleanup_callback_keys:
            return
        self._cleanup_callback_keys.add(key)
        self._cleanup_callbacks.append(callback)

    def get_runtime_status_lines(self) -> List[str]:
        lines: List[str] = []
        checkpoint = self.checkpoint_info or {}
        if checkpoint:
            lines.append(
                f"Checkpoint: requested={checkpoint.get('backend')} active={checkpoint.get('resolved_backend')} target={checkpoint.get('target')}"
            )
            for warning in checkpoint.get("warnings", []):
                lines.append(f"Checkpoint warning: {warning}")
        for status in self.loader_status:
            if status["error"]:
                lines.append(f"Loader {status['loader']}: ERROR {status['error']}")
        for status in self.mcp_server_status:
            if status["error"]:
                lines.append(f"MCP {status['server']}: ERROR {status['error']}")
            else:
                lines.append(
                    f"MCP {status['server']}: loaded {len(status['loaded_tools'])} tool(s)"
                )
        return lines

    def _read_mcp_config(self) -> Dict[str, Any]:
        try:
            raw_cfg = json.loads(self.config.mcp_config_path.read_text("utf-8"))
        except json.JSONDecodeError:
            logger.error(f"❌ Invalid JSON in {self.config.mcp_config_path}")
            return {}

        if not isinstance(raw_cfg, dict):
            logger.error(f"❌ MCP Config must be a dictionary, got {type(raw_cfg).__name__}")
            return {}
        return raw_cfg

    def _expand_env_vars(self, data: Union[Dict[str, Any], List[Any], str]) -> Any:
        if isinstance(data, dict):
            return {k: self._expand_env_vars(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        if isinstance(data, str):
            return os.path.expandvars(data)
        return data

    async def cleanup(self):
        for client in self.mcp_clients:
            try:
                close_method = getattr(client, "aclose", None) or getattr(client, "close", None)
                if callable(close_method):
                    result = close_method()
                    if inspect.isawaitable(result):
                        await result
                elif hasattr(client, "__aexit__"):
                    try:
                        await client.__aexit__(None, None, None)
                    except Exception as e:
                        if "MultiServerMCPClient cannot be used as a context manager" not in str(e):
                            raise
            except Exception as e:
                logger.error("Error closing MCP client: %s", e)

        callbacks = self._cleanup_callbacks
        self._cleanup_callbacks = []
        self._cleanup_callback_keys.clear()
        for callback in callbacks:
            try:
                result = callback()
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                logger.error("Error during runtime cleanup: %s", e)

        self.mcp_clients.clear()
        logger.info("ToolRegistry cleanup completed.")
