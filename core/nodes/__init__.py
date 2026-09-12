from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from core.config import AgentConfig
from core.run_logger import JsonlRunLogger
from core.state import AgentState
from core.tool_policy import ToolMetadata

from core.nodes.base import BaseMixin
from core.nodes.llm import LLMMixin
from core.nodes.context import ContextMixin
from core.nodes.tool_preflight import ToolPreflightMixin
from core.nodes.protocol import ProtocolMixin
from core.nodes.summarize import SummarizeMixin
from core.nodes.agent import AgentMixin
from core.nodes.approval import ApprovalMixin
from core.nodes.tools import ToolsMixin
from core.nodes.recovery import RecoveryMixin

__all__ = ["AgentNodes"]


class AgentNodes(
    BaseMixin,
    LLMMixin,
    ContextMixin,
    ToolPreflightMixin,
    ProtocolMixin,
    SummarizeMixin,
    AgentMixin,
    ApprovalMixin,
    ToolsMixin,
    RecoveryMixin,
):
    __slots__ = (
        "config",
        "llm",
        "tools",
        "llm_with_tools",
        "tools_map",
        "_all_tool_names",
        "tool_metadata",
        "model_capabilities",
        "run_logger",
        "_cached_base_prompt",
        "message_context",
        "context_builder",
        "recovery_manager",
        "tool_executor",
        "agent_turn",
        "recovery_turn",
        "tool_batch",
        "_required_fields_cache",
        "active_tools_provider",
    )

    # Built-in fallback when explicit metadata is unavailable. Registered tools
    # with read-only metadata (including MCP) may also run in parallel.
    # cli_exec remains an explicit exception; its independence is the caller's
    # responsibility. Unknown tools and request_user_input stay sequential.
    PARALLEL_SAFE_TOOL_NAMES = frozenset(
        {
            "read_file",
            "list_directory",
            "fetch_content",
            "crawl_site",
            "batch_web_search",
            "find_process_by_port",
            "cli_exec",
        }
    )
    # Read-only tools can be called repeatedly while an agent verifies edits/results.
    READ_ONLY_LOOP_TOLERANT_TOOL_NAMES = frozenset(
        {
            "read_file",
            "list_directory",
            "fetch_content",
            "crawl_site",
            "batch_web_search",
        }
    )
    PROVIDER_SAFE_TOOL_CALL_ID_RE = __import__("re").compile(r"^[A-Za-z0-9]{9}$")

    def __init__(
        self,
        config: AgentConfig,
        llm: BaseChatModel,
        tools: List[BaseTool],
        llm_with_tools: Optional[BaseChatModel] = None,
        tool_metadata: Optional[Dict[str, ToolMetadata]] = None,
        model_capabilities: Optional[Dict[str, Any]] = None,
        run_logger: Optional[JsonlRunLogger] = None,
        active_tools_provider: Callable[[], List[BaseTool]] | None = None,
    ):
        self.config = config
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = llm_with_tools or llm
        self.active_tools_provider = active_tools_provider

        # Optimization: O(1) tool lookup instead of O(N) list traversal
        self.tools_map = {t.name: t for t in tools}
        self._all_tool_names = tuple(self.tools_map.keys())
        self.tool_metadata = tool_metadata or {}
        self.model_capabilities = dict(model_capabilities or {})
        self.run_logger = run_logger

        # Cache: tool_name -> tuple of required field names.
        # Tool schemas are immutable after registration, so this is safe for
        # the lifetime of this AgentNodes instance.
        self._required_fields_cache: Dict[str, tuple[str, ...]] = {}

        # Lazy initialization of helpers (mixins may define these)
        self._init_nodes()

    def _init_nodes(self) -> None:
        from core.message_context import MessageContextHelper
        from core.context_builder import ContextBuilder
        from core.recovery_manager import RecoveryManager
        from core.tool_executor import ToolExecutor
        from core.node_orchestrators import AgentTurnOrchestrator, RecoveryTurnOrchestrator, ToolBatchCoordinator

        self.message_context = MessageContextHelper()
        self.recovery_manager = RecoveryManager()
        self._cached_base_prompt: Optional[str] = None
        self.context_builder = ContextBuilder(
            config=self.config,
            model_capabilities=self.model_capabilities,
            prompt_loader=self._get_base_prompt,
            is_internal_retry=self._is_internal_retry_message,
            log_run_event=self._log_run_event,
            recovery_message_builder=self.recovery_manager.build_recovery_system_message,
            provider_safe_tool_call_id_re=self.PROVIDER_SAFE_TOOL_CALL_ID_RE,
        )
        self.tool_executor = ToolExecutor(
            config=self.config,
            metadata_for_tool=self._metadata_for_tool,
            log_run_event=self._log_run_event,
            workspace_boundary_violated=self._workspace_boundary_violated,
        )
        self.agent_turn = AgentTurnOrchestrator(self)
        self.recovery_turn = RecoveryTurnOrchestrator(self)
        self.tool_batch = ToolBatchCoordinator(self)
