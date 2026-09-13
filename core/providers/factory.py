"""Provider-agnostic LLM factory.

This module is the single entry point for creating chat models. It dispatches
to provider-specific factories in :mod:`core.providers.gemini` and
:mod:`core.providers.openai_reasoning` and keeps the orchestration-level
concerns (provider selection, API-key rotation, tool binding) separate from
provider-specific private-method overrides.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.utils.function_calling import convert_to_openai_tool

from core.api_key_rotation import RotatingChatModel
from core.config import AgentConfig
from core.providers.anthropic import create_anthropic_chat_model
from core.providers.gemini import create_gemini_chat_model
from core.providers.openai_reasoning import create_openai_chat_model

logger = logging.getLogger("agent")


def create_llm(config: AgentConfig, *, api_key_override: str | None = None) -> BaseChatModel:
    """Initialize an LLM based on the configured provider.

    Dispatches to the appropriate provider factory. Raises ``ValueError`` for
    unknown providers.
    """
    if config.provider == "gemini":
        return create_gemini_chat_model(config, api_key_override=api_key_override)
    if config.provider == "openai":
        return create_openai_chat_model(config, api_key_override=api_key_override)
    if config.provider == "anthropic":
        return create_anthropic_chat_model(config, api_key_override=api_key_override)
    raise ValueError(f"Unknown provider: {config.provider}")


def create_runtime_llm(config: AgentConfig) -> BaseChatModel | RotatingChatModel:
    """Create the runtime LLM, optionally wrapped in API-key rotation."""
    profile_id = str(config.active_model_profile_id or "").strip()
    if not profile_id:
        return create_llm(config)
    return RotatingChatModel(
        config=config,
        profile_id=profile_id,
        profile_store_path=config.model_profile_config_path,
        llm_factory=create_llm,
    )


def _ensure_required_arrays(schema: Any) -> None:
    """Make object schemas explicit for OpenAI-compatible validators."""
    if isinstance(schema, dict):
        if schema.get("type") == "object" or "properties" in schema:
            schema.setdefault("required", [])
        for value in schema.values():
            _ensure_required_arrays(value)
    elif isinstance(schema, list):
        for value in schema:
            _ensure_required_arrays(value)


def _normalize_tool_for_binding(tool: Any) -> Any:
    """Return a portable tool schema, preserving provider-native tools as-is."""
    try:
        normalized = deepcopy(convert_to_openai_tool(tool))
    except Exception:
        return tool
    function = normalized.get("function")
    if isinstance(function, dict):
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            # MCP/JSON schemas can repeat the entire tool description here.
            # Keep distinct parameter guidance and every validation constraint.
            description = function.get("description")
            if description and parameters.get("description") == description:
                parameters.pop("description")
        _ensure_required_arrays(parameters)
    return normalized


def prepare_llm_with_tools(
    llm: BaseChatModel,
    tools: list[Any],
) -> tuple[BaseChatModel, bool, str]:
    """Bind normalized tools and report whether structured tool calling is available."""
    if not tools:
        return llm, False, ""

    binder = getattr(llm, "bind_tools", None)
    if not callable(binder):
        return llm, False, "LLM backend does not implement bind_tools()."

    bind_tools = [_normalize_tool_for_binding(tool) for tool in tools]
    try:
        return binder(bind_tools), True, ""
    except Exception as exc:
        return llm, False, str(exc)
