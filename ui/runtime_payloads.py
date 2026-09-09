from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from core.config import AgentConfig
from core.constants import AGENT_VERSION
from core.message_utils import is_tool_message_error, stringify_content
from core.model_profiles import find_active_profile, normalize_profiles_payload
from core.multimodal import extract_user_turn_data, resolve_model_capabilities
from core.session_store import (
    DEFAULT_CHAT_TITLE,
    SessionListEntry,
    SessionSnapshot,
    SessionStore,
    normalize_project_path,
)
from core.summarize_policy import (
    estimate_context_tokens,
    estimate_summary_tokens,
    should_summarize,
    summary_remaining_ratio,
    summary_trigger_tokens,
)
from core.text_utils import build_mcp_tool_ui_labels, build_tool_ui_labels, format_tool_output, prepare_markdown_for_render
from core.tool_args import canonicalize_tool_args
from core.tool_policy import ToolMetadata
from ui.tool_message_utils import extract_tool_args
from ui.visibility import get_internal_ui_notice, is_hidden_internal_message

APPROVAL_MODE_PROMPT = "prompt"
APPROVAL_MODE_ALWAYS = "always"
_INLINE_THOUGHT_BLOCK_RE = re.compile(r"<(think|thought)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_INLINE_THOUGHT_CLOSE_PREFIX_RE = re.compile(r"^.*?</(think|thought)>\s*", re.IGNORECASE | re.DOTALL)
_INLINE_THOUGHT_UNCLOSED_RE = re.compile(r"<(think|thought)\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
CHAT_TITLE_MAX_LENGTH = 50
CHAT_TITLE_FALLBACK = DEFAULT_CHAT_TITLE
TITLE_PREFIX_RE = re.compile(
    r"^(?:(?:пожалуйста|плиз|please)\s+)?"
    r"(?:(?:помоги(?:те)?|можешь(?:\s+ли)?|сделай(?:те)?|подскажи(?:те)?|нужно|надо|хочу|help|can you|could you|please)\s+)+",
    re.IGNORECASE,
)
TITLE_STRIP_RE = re.compile(r"^[\s\-\.,:;!?\"'`~()\[\]{}<>/\\]+|[\s\-\.,:;!?\"'`~()\[\]{}<>/\\]+$")
DIFF_BLOCK_RE = re.compile(r"```diff\r?\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class ApprovalSummary:
    destructive_count: int
    mutating_count: int
    networked_count: int
    default_approve: bool
    risk_level: str
    impacts: tuple[str, ...]


def _provider_model(config: AgentConfig) -> tuple[str, str]:
    if config.provider == "gemini":
        return "Gemini", config.gemini_model
    if config.provider == "anthropic":
        return "Anthropic", config.anthropic_model
    return "OpenAI", config.openai_model


def _short_id(value: str, length: int = 16) -> str:
    if len(value) <= length:
        return value
    return f"{value[:length]}…"


def _tool_is_mcp(tool, metadata: ToolMetadata | None) -> bool:
    return bool((metadata and metadata.source == "mcp") or hasattr(tool, "_is_mcp") or ":" in tool.name)


def _tool_group(tool, metadata: ToolMetadata | None) -> str:
    if _tool_is_mcp(tool, metadata):
        return "MCP"
    if metadata and (metadata.mutating or metadata.destructive or metadata.requires_approval):
        return "Protected"
    return "Read-only"


def _enabled_mcp_servers(tool_registry) -> list[str]:
    names = []
    for status in getattr(tool_registry, "mcp_server_status", []):
        server = status.get("server", "unknown")
        names.append(f"{server} (error)" if status.get("error") else server)
    return names


def _runtime_issue_count(tool_registry) -> int:
    lines = tool_registry.get_runtime_status_lines()
    return sum(
        1
        for line in lines
        if any(keyword in line.lower() for keyword in ("error", "warning", "failed", "unavailable"))
    )


def build_tools_snapshot(tool_registry) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata_map = getattr(tool_registry, "tool_metadata", {})
    tools = getattr(tool_registry, "tools", [])
    builtin_tools = getattr(tool_registry, "builtin_tools", tools)
    statuses = {
        str(status.get("server", "")): status
        for status in getattr(tool_registry, "mcp_server_status", [])
        if status.get("server")
    }
    mcp_config = {
        name: cfg
        for name, cfg in getattr(tool_registry, "mcp_config", {}).items()
        if name != "_builtin_tools"
    }
    active_tool_names = {
        tool.name
        for tool in (
            tool_registry.active_tools()
            if hasattr(tool_registry, "active_tools")
            else tools
        )
    }
    mcp_tool_names = {
        str(tool_name)
        for status in statuses.values()
        for tool_name in status.get("loaded_tools", [])
    }

    server_names = list(mcp_config)
    server_names.extend(name for name in statuses if name not in mcp_config)
    for server_name in server_names:
        cfg = mcp_config.get(server_name, {})
        if not isinstance(cfg, dict):
            continue
        status = statuses.get(str(server_name), {})
        error = str(status.get("error", "") or "")
        enabled = bool(cfg.get("enabled", status.get("enabled", True))) and not error
        loaded_names = {str(name) for name in status.get("loaded_tools", [])}
        server_tools = []
        for tool in tools:
            metadata = metadata_map.get(tool.name)
            belongs_to_server = (
                tool.name in loaded_names
                or tool.name.startswith(f"{server_name}:")
                or (_tool_is_mcp(tool, metadata) and tool.name.rsplit(":", 1)[-1] in loaded_names)
            )
            if belongs_to_server:
                server_tools.append(
                    {
                        "kind": "mcp_tool",
                        "name": tool.name,
                        "description": tool.description or "No description",
                    }
                )
        server_tools.sort(key=lambda item: item["name"])
        if error:
            description = f"MCP server - error: {error}"
        elif not enabled:
            description = "MCP server - disabled"
        else:
            description = f"MCP server - {len(server_tools)} tool(s)"
        rows.append(
            {
                "group": "MCP",
                "kind": "server",
                "name": str(server_name),
                "description": description,
                "enabled": enabled,
                "tools": server_tools,
            }
        )

    for group_name in ("Read-only", "Protected"):
        items = []
        for tool in builtin_tools:
            metadata = metadata_map.get(tool.name)
            is_mcp = _tool_is_mcp(tool, metadata) or tool.name in mcp_tool_names
            if not is_mcp and _tool_group(tool, metadata) == group_name:
                items.append((tool, metadata))
        for tool, metadata in sorted(items, key=lambda item: item[0].name):
            rows.append(
                {
                    "group": group_name,
                    "kind": "tool",
                    "name": tool.name,
                    "description": tool.description or "No description",
                    "enabled": tool.name in active_tool_names,
                }
            )
    return rows


def build_runtime_snapshot(config: AgentConfig, tool_registry, snapshot: SessionSnapshot) -> dict[str, Any]:
    provider_label, model_name = _provider_model(config)
    checkpoint_info = getattr(tool_registry, "checkpoint_info", {}) or {}
    backend = checkpoint_info.get("resolved_backend", config.checkpoint_backend)
    approvals = "off"
    if config.enable_approvals:
        approvals = "on"
        if getattr(snapshot, "approval_mode", APPROVAL_MODE_PROMPT) == APPROVAL_MODE_ALWAYS:
            approvals = "on (always for this session)"

    mcp_servers = _enabled_mcp_servers(tool_registry)
    issue_count = _runtime_issue_count(tool_registry)
    status = "ready" if issue_count == 0 else f"degraded ({issue_count} issue{'s' if issue_count != 1 else ''})"
    tools = build_tools_snapshot(tool_registry)
    return {
        "version": AGENT_VERSION,
        "provider": provider_label,
        "model": model_name,
        "backend": backend,
        "tools_count": (
            len(tool_registry.active_tools())
            if hasattr(tool_registry, "active_tools")
            else len(getattr(tool_registry, "tools", []))
        ),
        "session_id": snapshot.session_id,
        "session_short": _short_id(snapshot.session_id),
        "session_title": snapshot.title,
        "cache_hit_tokens": max(0, int(getattr(snapshot, "cache_hit_tokens", 0) or 0)),
        "thread_id": snapshot.thread_id,
        "thread_short": _short_id(snapshot.thread_id),
        "project_path": snapshot.project_path,
        "approvals": approvals,
        "mcp_servers": mcp_servers,
        "mcp_text": ", ".join(mcp_servers) if mcp_servers else "none",
        "status": status,
        "config_mode": "debug" if config.debug else "standard",
        "runtime_lines": tool_registry.get_runtime_status_lines(),
        "tools": tools,
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _provider_input_tokens(token_usage: Any) -> int:
    usage = token_usage if isinstance(token_usage, dict) else {}
    for key in ("input_tokens", "prompt_tokens", "prompt_token_count", "input_token_count"):
        if key in usage:
            return max(0, _safe_int(usage.get(key), 0))
    output_tokens = 0
    for key in ("output_tokens", "completion_tokens", "completion_token_count", "output_token_count"):
        if key in usage:
            output_tokens = max(0, _safe_int(usage.get(key), 0))
            break
    total_tokens = max(0, _safe_int(usage.get("total_tokens"), 0))
    return max(0, total_tokens - output_tokens) if total_tokens else 0


def build_summary_progress_payload(config: AgentConfig, state_values: dict[str, Any] | None) -> dict[str, Any]:
    values = state_values if isinstance(state_values, dict) else {}
    messages = list(values.get("messages", []) or [])
    threshold = max(0, _safe_int(getattr(config, "summary_threshold", 0), 0))
    reserved_tokens = max(0, _safe_int(getattr(config, "summary_reserved_tokens", 0), 0))
    summary_text = str(values.get("summary") or "").strip()
    summary_tokens = estimate_summary_tokens(summary_text)
    effective_reserved_tokens = reserved_tokens + summary_tokens
    estimated_tokens = estimate_context_tokens(messages, reserved_tokens=effective_reserved_tokens)
    has_summary = bool(summary_text)
    trigger_tokens = summary_trigger_tokens(threshold, has_summary=has_summary)
    progress = summary_remaining_ratio(
        estimated_tokens,
        threshold=threshold,
    )
    keep_last = _safe_int(getattr(config, "summary_keep_last", 0), 0)
    will_summarize = should_summarize(
        messages,
        threshold=threshold,
        keep_last=keep_last,
        has_summary=has_summary,
        reserved_tokens=effective_reserved_tokens,
    ) if messages and threshold > 0 else False
    return {
        "estimated_tokens": estimated_tokens,
        "threshold": threshold,
        "trigger_tokens": trigger_tokens,
        "remaining_tokens": max(0, trigger_tokens - estimated_tokens),
        "reserved_tokens": reserved_tokens if messages else 0,
        "summary_tokens": summary_tokens,
        "provider_input_tokens": _provider_input_tokens(values.get("token_usage")),
        "progress": progress,
        "message_count": len(messages),
        "has_summary": has_summary,
        "will_summarize": will_summarize,
    }


def build_help_markdown() -> str:
    return (
        "## Workflow\n"
        "- Type a request and press **Enter**.\n"
        "- Use **Shift+Enter** for a new line.\n"
        "- Use **Up/Down** in an empty composer to browse earlier prompts from the current chat.\n"
        "- Type **@** to mention files from the current workspace.\n"
        "- Use the **+** button to attach images or Add files.\n"
        "- You can also paste images or files directly into the composer.\n"
        "- Use **Ctrl+N** for **New Session**.\n"
        "- Use **Ctrl+B** to show or hide the chat history sidebar.\n"
        "- Use **Ctrl+I** to open the runtime information popup.\n"
        "- Use **Open project folder** to switch the working directory and start a fresh chat for that folder.\n"
        "- Open **Settings** to add models, switch the active profile, or enable image support for a profile.\n"
        "- Open **Tools** to inspect read-only, protected, and MCP capabilities.\n"
        "- Open **Session** to review provider, backend, session, and MCP runtime state.\n"
        "- Right-click a chat in the sidebar to delete it from history.\n"
        "- Right-click a project in the sidebar to delete all of its chats from history.\n"
        "- Use **New Session** to reset the active session and clear session-scoped approvals.\n"
        "- Approval dialogs support **Approve**, **Deny**, and **Always for this session**.\n"
    )


def _plain_summary_text(text: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", text or "").strip()


def generate_chat_title(user_text: str) -> str:
    text = " ".join(str(user_text or "").replace("\r", " ").replace("\n", " ").split()).strip()
    text = TITLE_PREFIX_RE.sub("", text)
    text = TITLE_STRIP_RE.sub("", text).strip()
    if not text:
        return CHAT_TITLE_FALLBACK
    if len(text) > CHAT_TITLE_MAX_LENGTH:
        text = text[: CHAT_TITLE_MAX_LENGTH - 1].rstrip() + "…"
    return text[:1].upper() + text[1:] if text else CHAT_TITLE_FALLBACK


def short_project_label(project_path: str | Path | None) -> str:
    normalized = normalize_project_path(project_path)
    path = Path(normalized)
    parts = [part for part in path.parts if part and part not in {path.anchor, "/", "\\"}]
    if not parts:
        return path.drive or normalized
    if len(parts) == 1:
        return parts[0]
    return "/".join(parts[-2:])


def append_project_label(title: str, project_path: str | Path | None) -> str:
    base_title = str(title or CHAT_TITLE_FALLBACK).strip() or CHAT_TITLE_FALLBACK
    label = short_project_label(project_path)
    if not label:
        return base_title
    suffix = f" [{label}]"
    if base_title.endswith(suffix):
        return base_title
    return f"{base_title}{suffix}"


def serialize_session_entries(entries: list[SessionListEntry]) -> list[dict[str, str]]:
    return [
        {
            "session_id": entry.session_id,
            "thread_id": entry.thread_id,
            "project_path": entry.project_path,
            "title": entry.title,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        }
        for entry in entries
    ]


def _extract_ai_text(message: AIMessage | AIMessageChunk) -> str:
    def _strip_inline_thought_content(text: str) -> str:
        cleaned = _INLINE_THOUGHT_BLOCK_RE.sub("", text)
        cleaned = _INLINE_THOUGHT_CLOSE_PREFIX_RE.sub("", cleaned)
        return _INLINE_THOUGHT_UNCLOSED_RE.sub("", cleaned)

    def _extract_visible_text(content: Any) -> str:
        if isinstance(content, str):
            return _strip_inline_thought_content(content)
        if content is None:
            return ""
        if isinstance(content, list):
            return "".join(_extract_visible_text(item) for item in content)
        if isinstance(content, dict):
            item_type = str(content.get("type") or "").strip().lower()
            if bool(content.get("thought")) or item_type in {
                "thinking",
                "thought",
                "reasoning",
                "reasoning_content",
                "reasoning_summary",
                "analysis",
                "analysis_content",
                "summary_text",
            } or item_type.startswith(("reasoning.", "thinking.", "thought.", "analysis.")):
                return ""
            for key in ("text", "output_text", "content", "answer", "response", "final", "final_text"):
                if key in content:
                    text = _extract_visible_text(content.get(key))
                    if text:
                        return text
            return "".join(
                _extract_visible_text(content.get(key))
                for key in ("parts", "items", "content_blocks", "message", "messages", "data")
                if key in content
            )
        for key in ("content", "content_blocks", "text"):
            try:
                value = getattr(content, key)
            except Exception:
                continue
            if callable(value) or value is None:
                continue
            text = _extract_visible_text(value)
            if text:
                return text
        try:
            additional_kwargs = getattr(content, "additional_kwargs")
        except Exception:
            additional_kwargs = None
        if isinstance(additional_kwargs, dict):
            return _extract_visible_text(additional_kwargs.get("content_blocks"))
        return stringify_content(content)

    return _extract_visible_text(message.content).strip()


def _normalized_visible_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _assistant_replay_remainder(previous: str, incoming: str) -> str | None:
    previous_normalized = _normalized_visible_text(previous)
    incoming_normalized = _normalized_visible_text(incoming)
    if not previous_normalized or not incoming_normalized:
        return None
    if incoming.startswith(previous):
        return incoming[len(previous) :].lstrip()
    if incoming_normalized == previous_normalized:
        return ""

    best_index: int | None = None
    for match in re.finditer(r"\S+", incoming):
        prefix = incoming[: match.end()]
        prefix_normalized = _normalized_visible_text(prefix)
        if not prefix_normalized:
            continue
        if previous_normalized.startswith(prefix_normalized):
            best_index = match.end()
            continue
        if prefix_normalized.startswith(previous_normalized):
            return incoming[best_index if best_index is not None else match.end() :].lstrip()
        if len(prefix_normalized) >= min(len(previous_normalized), 80):
            similarity = difflib.SequenceMatcher(None, previous_normalized, prefix_normalized).quick_ratio()
            if similarity >= 0.94:
                best_index = match.end()

    if best_index is not None:
        replayed_prefix = _normalized_visible_text(incoming[:best_index])
        significant_replay = (
            len(replayed_prefix) >= min(len(previous_normalized), 48)
            or len(replayed_prefix) >= int(len(previous_normalized) * 0.8)
        )
        if significant_replay:
            return incoming[best_index:].lstrip()
    return None


def _diff_from_tool_content(content: str) -> str:
    match = DIFF_BLOCK_RE.search(content or "")
    return match.group(1).strip() if match else ""


def build_transcript_payload(
    state_values: dict[str, Any] | None,
    *,
    last_run_stats: str = "",
    tool_sources: dict[str, str] | None = None,
    mcp_tool_servers: dict[str, str] | None = None,
) -> dict[str, Any]:
    values = state_values or {}
    summary_text = str(values.get("summary") or "").strip()
    normalized_last_run_stats = str(last_run_stats or "").strip()
    # ``messages`` is the compact model context and may contain only the tail
    # after auto-summarization. Prefer the durable transcript when it is present;
    # retain the old fallback for checkpoints created before this channel existed.
    transcript_messages = values.get("transcript_messages")
    has_transcript_user_turn = isinstance(transcript_messages, list) and any(
        isinstance(message, HumanMessage) for message in transcript_messages
    )
    source_messages = transcript_messages if has_transcript_user_turn else values.get("messages", [])
    turns: list[dict[str, Any]] = []
    pending_tool_calls: dict[str, dict[str, Any]] = {}
    current_turn: dict[str, Any] | None = None
    for message in source_messages or []:
        if isinstance(message, HumanMessage):
            text, attachments = extract_user_turn_data(message.content)
            text = text.strip()
            if not text and not attachments:
                continue
            current_turn = {"user_text": text, "attachments": attachments, "blocks": []}
            turns.append(current_turn)
            continue

        if current_turn is None:
            continue

        if isinstance(message, (AIMessage, AIMessageChunk)):
            for tool_call in getattr(message, "tool_calls", []) or []:
                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    pending_tool_calls[tool_call_id] = {
                        "name": tool_call.get("name", "tool"),
                        "args": canonicalize_tool_args(tool_call.get("args")),
                    }
            if is_hidden_internal_message(message):
                notice = get_internal_ui_notice(message)
                if notice:
                    current_turn["blocks"].append(
                        {
                            "type": "notice",
                            "message": notice,
                            "level": "warning",
                        }
                    )
                continue
            text = _extract_ai_text(message)
            if text.strip():
                text = text.strip()
                if getattr(message, "tool_calls", None):
                    current_turn["_last_assistant_before_tool"] = text
                elif current_turn.get("_after_tool") and current_turn.get("_last_assistant_before_tool"):
                    remainder = _assistant_replay_remainder(
                        str(current_turn.get("_last_assistant_before_tool") or ""),
                        text,
                    )
                    if remainder is not None:
                        text = remainder
                markdown = prepare_markdown_for_render(text.strip())
                if markdown:
                    current_turn["blocks"].append(
                        {
                            "type": "assistant",
                            "markdown": markdown,
                        }
                    )
            continue

        if isinstance(message, ToolMessage):
            if is_hidden_internal_message(message):
                notice = get_internal_ui_notice(message)
                if notice:
                    current_turn["blocks"].append(
                        {
                            "type": "notice",
                            "message": notice,
                            "level": "warning",
                        }
                    )
                continue
            tool_meta = pending_tool_calls.get(message.tool_call_id, {})
            tool_name = tool_meta.get("name") or message.name or "tool"
            tool_args = canonicalize_tool_args(tool_meta.get("args"))
            if not tool_args:
                tool_args = extract_tool_args(message)
            content = stringify_content(message.content)
            is_error = is_tool_message_error(message)
            if (tool_sources or {}).get(tool_name) == "mcp":
                labels = build_mcp_tool_ui_labels(
                    tool_name,
                    tool_args,
                    phase="finished",
                    is_error=is_error,
                    server_name=(mcp_tool_servers or {}).get(tool_name, ""),
                )
            else:
                labels = build_tool_ui_labels(tool_name, tool_args, phase="finished", is_error=is_error)
            current_turn["_after_tool"] = True
            current_turn["blocks"].append(
                {
                    "type": "tool",
                    "payload": {
                        "tool_id": message.tool_call_id,
                        "name": tool_name,
                        "args": tool_args,
                        "display": labels.get("title") or tool_name,
                        "subtitle": labels.get("subtitle", ""),
                        "raw_display": labels.get("raw_display", tool_name),
                        "args_state": labels.get("args_state", "complete"),
                        "display_state": "finished",
                        "phase": "finished",
                        "source_kind": labels.get("source_kind", "tool"),
                        "summary": _plain_summary_text(format_tool_output(tool_name, content, is_error)),
                        "content": content,
                        "is_error": is_error,
                        "duration": None,
                        "diff": _diff_from_tool_content(content),
                    },
                }
            )

    for turn in turns:
        turn.pop("_after_tool", None)
        turn.pop("_last_assistant_before_tool", None)

    if normalized_last_run_stats and turns:
        last_turn = turns[-1]
        blocks = last_turn.setdefault("blocks", [])
        if blocks and not any(isinstance(block, dict) and block.get("type") == "stats" for block in blocks):
            blocks.append({"type": "stats", "stats": normalized_last_run_stats})

    return {
        "summary_notice": (
            "Model context was compressed, but the full chat history is preserved below."
            if summary_text and has_transcript_user_turn
            else (
                "Early messages were compressed automatically; the restored chat may be incomplete."
                if summary_text
                else ""
            )
        ),
        "turns": turns,
    }


async def load_state_snapshot(agent_app, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    async_get_state = getattr(agent_app, "aget_state", None)
    if callable(async_get_state):
        return await async_get_state(config)
    return agent_app.get_state(config)


def find_state_snapshot_interrupt_payload(state_snapshot, *, choice_types: set[str] | None = None) -> dict[str, Any] | None:
    tasks = getattr(state_snapshot, "tasks", ()) or ()
    for task in tasks:
        interrupts = getattr(task, "interrupts", ()) or ()
        for entry in interrupts:
            value = getattr(entry, "value", entry)
            if isinstance(value, dict):
                choice_type = str(value.get("choice_type") or "")
                if choice_types is None or choice_type in choice_types:
                    return value
    interrupts = getattr(state_snapshot, "interrupts", ()) or ()
    for entry in interrupts:
        value = getattr(entry, "value", entry)
        if isinstance(value, dict):
            choice_type = str(value.get("choice_type") or "")
            if choice_types is None or choice_type in choice_types:
                return value
    return None


async def load_state_values(agent_app, thread_id: str) -> dict[str, Any]:
    state = await load_state_snapshot(agent_app, thread_id)
    values = getattr(state, "values", {}) if state is not None else {}
    return values if isinstance(values, dict) else {}


async def load_transcript_payload(agent_app, thread_id: str, snapshot: SessionSnapshot) -> dict[str, Any]:
    values = await load_state_values(agent_app, thread_id)
    return build_transcript_payload(values, last_run_stats=getattr(snapshot, "last_run_stats", ""))


def summarize_approval_request(req_tools: list[dict]) -> ApprovalSummary:
    destructive_count = 0
    mutating_count = 0
    networked_count = 0
    impacts: set[str] = set()

    for tool in req_tools:
        policy = tool.get("policy") or {}
        name = (tool.get("name") or "").lower()
        destructive = bool(policy.get("destructive"))
        mutating = bool(policy.get("mutating"))
        networked = bool(policy.get("networked"))

        destructive_count += int(destructive)
        mutating_count += int(mutating)
        networked_count += int(networked)

        if destructive or mutating:
            if any(token in name for token in ("process", "pid", "port", "shell", "exec", "command")):
                impacts.add("processes")
            elif any(token in name for token in ("file", "directory", "path", "write", "edit", "delete", "download")):
                impacts.add("files")
            else:
                impacts.add("local state")
        if networked:
            impacts.add("network")

    risk_kinds = sum(int(count > 0) for count in (destructive_count, mutating_count, networked_count))
    mixed_risk = risk_kinds > 1
    default_approve = destructive_count == 0 and not mixed_risk
    risk_level = "high" if destructive_count or mixed_risk else "medium" if (mutating_count or networked_count) else "low"

    return ApprovalSummary(
        destructive_count=destructive_count,
        mutating_count=mutating_count,
        networked_count=networked_count,
        default_approve=default_approve,
        risk_level=risk_level,
        impacts=tuple(sorted(impacts)),
    )


def normalize_approval_mode(value: str | None) -> str:
    if value == APPROVAL_MODE_ALWAYS:
        return APPROVAL_MODE_ALWAYS
    return APPROVAL_MODE_PROMPT


def build_user_choice_payload(interrupt_payload: dict) -> dict[str, Any]:
    options_payload: list[dict[str, Any]] = []
    raw_options = interrupt_payload.get("options", []) if isinstance(interrupt_payload, dict) else []
    recommended = str((interrupt_payload or {}).get("recommended", "") or "").strip()
    normalized_recommended = recommended.casefold()
    matched_recommended_label = ""

    for index, raw_option in enumerate(raw_options, start=1):
        if isinstance(raw_option, dict):
            label = str(raw_option.get("label") or raw_option.get("value") or "").strip()
            submit_text = str(raw_option.get("submit_text") or raw_option.get("value") or label).strip()
            key = str(raw_option.get("key") or f"option_{index}").strip()
        else:
            label = str(raw_option or "").strip()
            submit_text = label
            key = f"option_{index}"
        if not label or not submit_text:
            continue
        candidates = {key.casefold(), submit_text.casefold(), label.casefold()}
        is_recommended = bool(normalized_recommended and normalized_recommended in candidates)
        if is_recommended and not matched_recommended_label:
            matched_recommended_label = label
        options_payload.append(
            {
                "key": key,
                "label": label,
                "submit_text": submit_text,
                "recommended": is_recommended,
            }
        )

    return {
        "kind": "user_choice",
        "choice_type": str((interrupt_payload or {}).get("choice_type", "clarification") or "clarification"),
        "question": interrupt_payload.get("question", "How would you like to proceed?"),
        "options": options_payload,
        "recommended_key": matched_recommended_label,
        "allow_custom_text": bool((interrupt_payload or {}).get("allow_custom_text")),
        "custom_label": str((interrupt_payload or {}).get("custom_label", "") or "").strip(),
        "reason": str((interrupt_payload or {}).get("reason", "") or "").strip(),
    }


def build_approval_payload(interrupt_payload: dict, current_session: SessionSnapshot) -> dict[str, Any]:
    req_tools = interrupt_payload.get("tools", []) if isinstance(interrupt_payload, dict) else []
    summary = summarize_approval_request(req_tools)
    return {
        "kind": interrupt_payload.get("kind", ""),
        "tools": req_tools,
        "summary": {
            "destructive_count": summary.destructive_count,
            "mutating_count": summary.mutating_count,
            "networked_count": summary.networked_count,
            "default_approve": summary.default_approve,
            "risk_level": summary.risk_level,
            "impacts": list(summary.impacts),
        },
        "approval_mode": normalize_approval_mode(getattr(current_session, "approval_mode", APPROVAL_MODE_PROMPT)),
    }


async def build_ui_payload(
    config: AgentConfig,
    tool_registry,
    store: SessionStore,
    snapshot: SessionSnapshot,
    model_profiles: dict[str, Any] | None = None,
    model_capabilities: dict[str, Any] | None = None,
    *,
    agent_app=None,
    include_transcript: bool = False,
) -> dict[str, Any]:
    runtime_snapshot = build_runtime_snapshot(config, tool_registry, snapshot)
    normalized_profiles = normalize_profiles_payload(model_profiles or {})
    effective_capabilities = resolve_model_capabilities(
        find_active_profile(normalized_profiles),
        model_capabilities if model_capabilities is not None else getattr(tool_registry, "model_capabilities", None),
    )
    state_values: dict[str, Any] = {}
    state_snapshot = None
    if agent_app is not None:
        try:
            state_snapshot = await load_state_snapshot(agent_app, snapshot.thread_id)
            values = getattr(state_snapshot, "values", {}) if state_snapshot is not None else {}
            state_values = values if isinstance(values, dict) else {}
        except Exception:
            state_values = {}
    payload = {
        "snapshot": runtime_snapshot,
        "summary_progress": build_summary_progress_payload(config, state_values),
        "tools": runtime_snapshot["tools"],
        "help_markdown": build_help_markdown(),
        "sessions": serialize_session_entries(store.list_sessions()),
        "active_session_id": snapshot.session_id,
        "model_profiles": normalized_profiles,
        "model_capabilities": effective_capabilities,
    }
    if include_transcript and agent_app is not None:
        tool_sources = {
            name: str(getattr(metadata, "source", "local") or "local")
            for name, metadata in getattr(tool_registry, "tool_metadata", {}).items()
        }
        mcp_tool_servers = {
            tool_name: str(status.get("server", "") or "")
            for status in getattr(tool_registry, "mcp_server_status", [])
            for tool_name in status.get("loaded_tools", [])
        }
        payload["transcript"] = build_transcript_payload(
            state_values,
            last_run_stats=getattr(snapshot, "last_run_stats", ""),
            tool_sources=tool_sources,
            mcp_tool_servers=mcp_tool_servers,
        )
    return payload


async def close_runtime_resources(tool_registry) -> None:
    if tool_registry:
        await tool_registry.cleanup()
    try:
        from tools.filesystem import close_download_client

        await close_download_client()
    except ImportError:
        pass
