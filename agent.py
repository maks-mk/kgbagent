import asyncio
import logging
from typing import Any, Optional, Tuple

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from core.checkpointing import create_checkpoint_runtime
from core.config import AgentConfig
from core.logging_config import setup_logging
from core.model_profiles import ModelProfileStore, find_active_profile, find_profile_by_id
from core.multimodal import extract_model_capabilities, resolve_model_capabilities
from core.nodes import AgentNodes
from core.providers import (
    create_llm,
    create_runtime_llm,
    prepare_llm_with_tools,
)
# Back-compat re-exports — tests reference these via ``agent_module._...``.
from core.providers.gemini import (
    gemini_model_supports_thinking_budget as _gemini_model_supports_thinking_budget,
    patch_langchain_google_genai_retry_kwargs as _patch_langchain_google_genai_retry_kwargs,
)
from core.providers.openai_reasoning import (
    extract_openai_reasoning_delta as _extract_openai_reasoning_delta,
)
from core.run_logger import JsonlRunLogger
from core.state import AgentState
from core.turn_outcomes import (
    TURN_OUTCOME_CONTINUE_AGENT,
    TURN_OUTCOME_FINISH_TURN,
    TURN_OUTCOME_RECOVER_AGENT,
    TURN_OUTCOME_RUN_TOOLS,
    normalize_turn_outcome,
)
from tools.tool_registry import ToolRegistry

logger = logging.getLogger("agent")


def _register_llm_cleanup_callback(tool_registry: ToolRegistry, llm: Any) -> bool:
    close_method = getattr(llm, "aclose", None) or getattr(llm, "close", None)
    if callable(close_method):
        tool_registry.register_cleanup_callback(close_method)
        return True
    for target in (
        getattr(llm, "root_async_client", None),
        getattr(llm, "async_client", None),
        getattr(llm, "root_client", None),
        getattr(llm, "client", None),
    ):
        if target is None:
            continue
        target_close = getattr(target, "aclose", None) or getattr(target, "close", None)
        if callable(target_close):
            tool_registry.register_cleanup_callback(target_close)
            return True
    return False


def _resolve_effective_model_capabilities(config: AgentConfig, runtime_capabilities: dict[str, Any]) -> dict[str, Any]:
    profiles_payload = ModelProfileStore(config.model_profile_config_path).load()
    selected_profile_id = str(config.active_model_profile_id or "").strip()
    active_profile = (
        find_profile_by_id(profiles_payload, selected_profile_id)
        if selected_profile_id
        else find_active_profile(profiles_payload)
    )
    return resolve_model_capabilities(active_profile, runtime_capabilities)


# --- Builder ---


def create_agent_workflow(
    nodes: AgentNodes,
    config: AgentConfig,
    tools_enabled: Optional[bool] = None,
) -> StateGraph:
    """Builds the LangGraph workflow with tool-call based routing and bounded recovery."""
    tools_enabled = bool(nodes.tools) and config.model_supports_tools if tools_enabled is None else tools_enabled
    approval_enabled = bool(tools_enabled and config.enable_approvals)

    workflow = StateGraph(AgentState)

    workflow.add_node("summarize", nodes.summarize_node)
    workflow.add_node("agent", nodes.agent_node)
    workflow.add_node("recovery", nodes.recovery_node)
    workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})

    workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "update_step")
    workflow.add_edge("update_step", "agent")

    if tools_enabled:
        workflow.add_node("tools", nodes.tools_node)
        if approval_enabled:
            workflow.add_node("approval", nodes.approval_node)

    def route_after_agent(state: AgentState):
        steps = state.get("steps", 0)
        messages = state.get("messages") or []

        if not messages:
            logger.warning("Agent node returned no messages; ending turn safely.")
            return END

        turn_outcome = normalize_turn_outcome(state.get("turn_outcome"))
        has_open_tool_issue = bool(state.get("open_tool_issue"))
        if tools_enabled and turn_outcome == TURN_OUTCOME_RUN_TOOLS:
            if steps >= config.max_loops:
                logger.warning(
                    "Loop guard reached at step %s/%s with pending tool calls. Routing to recovery.",
                    steps,
                    config.max_loops,
                )
                return "recovery"
            pending_ai_with_tools = nodes._get_last_pending_ai_with_tool_calls(messages)
            if isinstance(pending_ai_with_tools, AIMessage) and pending_ai_with_tools.tool_calls:
                if (
                    approval_enabled
                    and nodes.tool_calls_require_approval(pending_ai_with_tools.tool_calls)
                ):
                    return "approval"
                return "tools"
            logger.warning(
                "Agent reported run_tools outcome without a valid tool call payload. "
                "Routing to recovery."
            )
            return "recovery"

        if turn_outcome == TURN_OUTCOME_RECOVER_AGENT or has_open_tool_issue:
            return "recovery"

        return END

    def route_after_tools(state: AgentState):
        if state.get("open_tool_issue"):
            return "recovery"
        # Mid-run compaction: re-check the context threshold after tool results
        # land, before the model writes its next comment. The summarize node is
        # a no-op below the SESSION_SIZE threshold.
        return "summarize"

    def route_after_recovery(state: AgentState):
        normalized = normalize_turn_outcome(state.get("turn_outcome"))
        if normalized in (TURN_OUTCOME_RECOVER_AGENT, TURN_OUTCOME_CONTINUE_AGENT):
            return "summarize"
        return END

    if tools_enabled:
        agent_routes = ["tools", "recovery", END]
        if approval_enabled:
            agent_routes.insert(0, "approval")
            workflow.add_edge("approval", "tools")
        workflow.add_conditional_edges(
            "agent",
            route_after_agent,
            agent_routes,
        )
        workflow.add_conditional_edges("tools", route_after_tools, ["recovery", "summarize"])
    else:
        workflow.add_conditional_edges(
            "agent",
            route_after_agent,
            ["recovery", END],
        )

    workflow.add_conditional_edges(
        "recovery",
        route_after_recovery,
        ["summarize", END],
    )

    return workflow


def build_compiled_agent(
    config: AgentConfig,
    tool_registry: ToolRegistry,
    checkpoint_runtime: Any,
    *,
    run_logger: Optional[JsonlRunLogger] = None,
) -> Tuple[Any, ToolRegistry]:
    """Compile an agent app using already-loaded tools and an existing checkpointer."""
    llm = create_runtime_llm(config)
    tool_registry.config = config
    tool_registry.model_capabilities = extract_model_capabilities(llm)
    effective_model_capabilities = _resolve_effective_model_capabilities(
        config,
        tool_registry.model_capabilities,
    )
    tool_registry.checkpoint_info = checkpoint_runtime.to_dict()
    tool_registry.checkpoint_runtime = checkpoint_runtime

    all_tools = (
        tool_registry.available_tools()
        if hasattr(tool_registry, "available_tools")
        else tool_registry.tools
    )
    tool_calling_enabled = bool(all_tools) and config.model_supports_tools
    llm_with_tools = llm
    if tool_calling_enabled:
        llm_with_tools, tool_calling_enabled, bind_error = prepare_llm_with_tools(llm, all_tools)
        if tool_calling_enabled:
            logger.info("🛠️ Tools bound to LLM successfully.")
        else:
            tool_registry.loader_status.append(
                {
                    "loader": "llm_tool_binding",
                    "module": config.provider,
                    "loaded_tools": [],
                    "error": bind_error,
                }
            )
            logger.error("Tool calling disabled for this runtime because tool binding failed: %s", bind_error)
    elif not config.model_supports_tools:
        logger.debug("⚠️ Tools disabled: Model does not support tool calling.")

    registered_cleanup_ids: set[int] = set()
    for cleanup_target in (llm, llm_with_tools):
        marker = id(cleanup_target)
        if marker in registered_cleanup_ids:
            continue
        if _register_llm_cleanup_callback(tool_registry, cleanup_target):
            registered_cleanup_ids.add(marker)

    active_tool_metadata = tool_registry.tool_metadata if tool_calling_enabled else {}
    nodes = AgentNodes(
        config=config,
        llm=llm,
        tools=all_tools,
        llm_with_tools=llm_with_tools,
        tool_metadata=active_tool_metadata,
        model_capabilities=effective_model_capabilities,
        run_logger=run_logger,
        active_tools_provider=tool_registry.active_tools,
    )
    workflow = create_agent_workflow(nodes, config, tools_enabled=tool_calling_enabled)
    return workflow.compile(checkpointer=checkpoint_runtime.checkpointer), tool_registry


async def build_agent_app(
    config: Optional[AgentConfig] = None,
    *,
    disabled_local_tools: set[str] | None = None,
) -> Tuple[Any, ToolRegistry]:
    """
    Builds the LangGraph application and returns it along with the tool registry.
    """
    # Pydantic AgentConfig автоматически загружает .env.
    config = config or AgentConfig()
    setup_logging(
        level=config.log_level,
        log_file=config.log_file,
        reasoning_debug_enabled=config.debug_reasoning_stream,
    )

    logger.info(f"Initializing agent: [bold cyan]{config.provider}[/]", extra={"markup": True})

    # 1. Initialize Resources
    tool_registry = ToolRegistry(config)
    await tool_registry.load_all()
    tool_registry.disabled_local_tools.update(disabled_local_tools or set())
    checkpoint_runtime = await create_checkpoint_runtime(config)
    run_logger = JsonlRunLogger(config.run_log_dir)
    tool_registry.checkpoint_info = checkpoint_runtime.to_dict()
    tool_registry.checkpoint_runtime = checkpoint_runtime
    tool_registry.register_cleanup_callback(checkpoint_runtime.aclose)
    return build_compiled_agent(
        config,
        tool_registry,
        checkpoint_runtime,
        run_logger=run_logger,
    )


if __name__ == "__main__":

    async def main():
        app, registry = await build_agent_app()
        print(f"✔ Agent Ready. Tools: {len(registry.tools)}")
        await registry.cleanup()

    asyncio.run(main())
