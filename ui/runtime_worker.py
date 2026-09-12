from __future__ import annotations

import asyncio
import logging
import os
import random
import sys
import threading
import uuid
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from core.config import AgentConfig
from core.logging_config import setup_logging
from core.model_profiles import ModelProfileStore, find_active_profile, normalize_profiles_payload
from core.multimodal import (
    DEFAULT_MODEL_CAPABILITIES,
    build_user_message_content,
    normalize_request_payload,
    request_has_content,
    request_task_text,
    request_user_text,
)
from core.run_logger import JsonlRunLogger
from core.session_store import SessionSnapshot, SessionStore
from core.summarize_policy import estimate_tokens, summary_remaining_ratio, summary_trigger_tokens
from ui.runtime_payloads import (
    APPROVAL_MODE_ALWAYS,
    APPROVAL_MODE_PROMPT,
    build_summary_progress_payload,
    build_approval_payload,
    build_ui_payload,
    build_user_choice_payload,
    close_runtime_resources,
    load_state_values,
    normalize_approval_mode,
    generate_chat_title,
    generate_chat_title_with_llm,
    append_project_label,
)

from ui.runtime_session import RuntimeSessionCoordinator
from ui.streaming import StreamEvent, StreamProcessor

logger = logging.getLogger("agent")


def _runtime_module():
    from ui import runtime as runtime_module

    return runtime_module


def setup_runtime() -> AgentConfig:
    if getattr(sys, "frozen", False):
        os.chdir(os.getcwd())
    config = AgentConfig()
    setup_logging(
        level=config.log_level,
        log_file=config.log_file,
        reasoning_debug_enabled=config.debug_reasoning_stream,
    )
    return config


def build_initial_state(user_input: Any, session_id: str, safety_mode: str = "default") -> dict:
    request_payload = normalize_request_payload(user_input)
    user_text = request_payload["text"]
    attachments = request_payload["attachments"]
    current_task = request_task_text(request_payload)
    user_message = HumanMessage(content=build_user_message_content(user_text, attachments))
    return {
        "messages": [user_message],
        "transcript_messages": [user_message],
        "steps": 0,
        "token_usage": {},
        "current_task": current_task,
        "requires_evidence": False,
        "session_id": session_id,
        "run_id": uuid.uuid4().hex,
        "turn_id": 1,
        "pending_approval": None,
        "open_tool_issue": None,
        "recovery_state": {
            "turn_id": 1,
            "retry_count": 0,
            "retry_fingerprint_history": [],
            "last_reason": "",
            "active_issue": None,
            "active_strategy": None,
            "strategy_queue": [],
            "attempts_by_strategy": {},
            "progress_markers": [],
            "last_successful_evidence": "",
            "external_blocker": None,
            "llm_replan_attempted_for": [],
        },
        "last_tool_error": "",
        "last_tool_result": "",
        "safety_mode": safety_mode,
    }


def build_graph_config(thread_id: str, max_loops: int) -> dict:
    graph_supersteps_per_logical_step = 6
    graph_superstep_overhead = 8
    recursion_limit = max(
        16,
        int(max_loops or 0) * graph_supersteps_per_logical_step + graph_superstep_overhead,
    )
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


class AgentRunWorker(QObject):
    initialized = Signal(object)
    initialization_failed = Signal(str)
    event_emitted = Signal(object)
    approval_requested = Signal(object)
    user_choice_requested = Signal(object)
    session_changed = Signal(object)
    busy_changed = Signal(bool)
    shutdown_complete = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._current_task: asyncio.Task | None = None
        self._task_state_lock = threading.Lock()
        self.config: AgentConfig | None = None
        self.base_config: AgentConfig | None = None
        self.store: SessionStore | None = None
        self.profile_store: ModelProfileStore | None = None
        self.model_profiles: dict[str, Any] = {"active_profile": None, "profiles": []}
        self.runtime_model_capabilities: dict[str, Any] = dict(DEFAULT_MODEL_CAPABILITIES)
        self.model_capabilities: dict[str, Any] = dict(DEFAULT_MODEL_CAPABILITIES)
        self._runtime_profile_id: str = ""
        self.agent_app = None
        self.tool_registry = None
        self._pending_mcp_server_name = ""
        self.checkpoint_runtime = None
        self.current_session: SessionSnapshot | None = None
        self.ui_run_logger: JsonlRunLogger | None = None
        self._is_busy = False
        self._awaiting_approval = False
        self._awaiting_interrupt_kind = ""
        self._pending_user_choice_type = ""
        self._active_run_elapsed_seconds = 0.0
        self._active_request_has_images = False
        self._active_summary_estimated_tokens = 0
        self._active_summary_message_count = 0
        self._active_summary_has_summary = False
        self._active_summary_reserved_tokens = 0
        self._active_summary_memory_tokens = 0
        self._active_provider_input_tokens = 0
        self._coordinator = RuntimeSessionCoordinator(self)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._task_state_lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop

    def _run(self, coro):
        loop = self._ensure_loop()

        async def _wrapper():
            with self._task_state_lock:
                self._current_task = asyncio.current_task()
            try:
                return await coro
            finally:
                with self._task_state_lock:
                    self._current_task = None

        try:
            return loop.run_until_complete(_wrapper())
        except asyncio.CancelledError:
            self.event_emitted.emit(StreamEvent("run_failed", {"message": "Stopped by user."}))
            self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        if self._is_busy == busy:
            return
        self._is_busy = busy
        self.busy_changed.emit(busy)

    def _current_project_path(self) -> str:
        return self._coordinator.current_project_path()

    @staticmethod
    def _project_path_is_valid(project_path) -> bool:
        return RuntimeSessionCoordinator.project_path_is_valid(project_path)

    def _create_new_session_for_project(self, project_path: str, *, with_project_label: bool = False):
        return self._coordinator.create_new_session_for_project(project_path, with_project_label=with_project_label)

    def _sync_tool_registry_workdir(self, target_project_path: str) -> None:
        self._coordinator.sync_tool_registry_workdir(target_project_path)

    def _ensure_current_session_persisted(self) -> bool:
        return self._coordinator.ensure_current_session_persisted()

    def _try_change_workdir(self, target_project_path: str) -> tuple[bool, str]:
        return self._coordinator.try_change_workdir(target_project_path)

    def _activate_session_with_workdir_or_fallback(self, target, **kwargs):
        return self._coordinator.activate_session_with_workdir_or_fallback(target, **kwargs)

    def _select_session_for_project(self, *, force_new_session: bool = False):
        return self._coordinator.select_session_for_project(force_new_session=force_new_session)

    def _maybe_set_session_title(self, user_text: str) -> bool:
        return self._coordinator.maybe_set_session_title(user_text)

    async def _emit_session_payload(self, *, include_transcript: bool) -> dict[str, Any]:
        return await self._coordinator.emit_session_payload(include_transcript=include_transcript)

    def _sync_pending_user_choice_from_payload(self, payload: dict[str, Any]) -> None:
        pending_choice = payload.get("pending_user_choice") if isinstance(payload, dict) else None
        if not isinstance(pending_choice, dict):
            return
        choice_type = str(pending_choice.get("choice_type") or "").strip()
        if not choice_type:
            return
        self._awaiting_approval = True
        self._awaiting_interrupt_kind = "user_choice"
        self._pending_user_choice_type = choice_type

    async def _repair_current_session_if_needed(self) -> list[str]:
        return await self._coordinator.repair_current_session_if_needed()

    @staticmethod
    def _profile_config_path():
        return RuntimeSessionCoordinator.profile_config_path()

    @staticmethod
    def _profile_bootstrap_env_from_config(config: AgentConfig) -> dict[str, str]:
        return RuntimeSessionCoordinator.profile_bootstrap_env_from_config(config)

    def _build_config_for_active_profile(self, model_profiles: dict[str, Any]):
        return self._coordinator.build_config_for_active_profile(model_profiles)

    @staticmethod
    def _selected_profile_id(model_profiles: dict[str, Any]) -> str:
        return RuntimeSessionCoordinator.selected_profile_id(model_profiles)

    def _set_effective_model_capabilities(self, runtime_capabilities: Any, *, model_profiles: dict[str, Any] | None = None) -> None:
        self._coordinator.set_effective_model_capabilities(runtime_capabilities, model_profiles=model_profiles)

    async def _rebuild_runtime_for_active_profile(self, model_profiles: dict[str, Any]) -> None:
        await self._coordinator.rebuild_runtime_for_active_profile(model_profiles)

    async def _apply_model_profiles(self, candidate_payload: dict[str, Any], **kwargs) -> bool:
        return await self._coordinator.apply_model_profiles(candidate_payload, **kwargs)

    async def _ensure_runtime_matches_selected_profile(self) -> bool:
        return await self._coordinator.ensure_runtime_matches_selected_profile()

    def _refresh_model_profiles_from_store(self) -> None:
        if self.profile_store is None:
            return
        self.model_profiles = self.profile_store.load()
        self._set_effective_model_capabilities(
            self.runtime_model_capabilities,
            model_profiles=self.model_profiles,
        )

    def _normalize_approval_mode(self, value: str | None) -> str:
        return normalize_approval_mode(value)

    def _log_ui_run_event(self, event_type: str, **payload: Any) -> None:
        if not self.ui_run_logger or not self.current_session:
            return
        normalized_payload = dict(payload)
        normalized_payload.pop("session_id", None)
        try:
            self.ui_run_logger.log_event(self.current_session.session_id, event_type, **normalized_payload)
        except Exception:
            logger.debug("Failed to write UI run event '%s'", event_type, exc_info=True)

    def _emit_cli_output_event(self, payload: dict[str, Any]) -> None:
        tool_id = str((payload or {}).get("tool_id", "")).strip()
        data = str((payload or {}).get("data", ""))
        if not tool_id or not data:
            return
        stream = str((payload or {}).get("stream", "stdout") or "stdout")
        self._emit_stream_event(StreamEvent("cli_output", {"tool_id": tool_id, "data": data, "stream": stream}))

    def _configure_cli_output_bridge(self) -> None:
        try:
            from tools import local_shell

            local_shell.set_cli_output_emitter(self._emit_cli_output_event)
        except Exception:
            logger.debug("Failed to set cli_output bridge.", exc_info=True)

    @staticmethod
    def _clear_cli_output_bridge() -> None:
        try:
            from tools import local_shell

            local_shell.set_cli_output_emitter(None)
        except Exception:
            logger.debug("Failed to clear cli_output bridge.", exc_info=True)

    def _emit_stream_event(self, event: StreamEvent) -> None:
        if event.type == "cache_hit":
            payload = event.payload if isinstance(event.payload, dict) else {}
            try:
                delta = max(0, int(payload.get("tokens", 0) or 0))
            except (TypeError, ValueError):
                delta = 0
            if delta and self.current_session is not None and self.store is not None:
                self.current_session.cache_hit_tokens = max(
                    0,
                    int(getattr(self.current_session, "cache_hit_tokens", 0) or 0) + delta,
                )
                try:
                    self.store.save_active_session(self.current_session, touch=False, set_active=True)
                except Exception:
                    logger.debug("Failed to persist session Cache Hit tokens.", exc_info=True)
        self.event_emitted.emit(event)
        self._maybe_reset_summary_progress_after_compaction(event)
        self._maybe_emit_live_summary_progress(event)
        if event.type in {"tool_args_missing"}:
            self._log_ui_run_event(
                event.type,
                thread_id=getattr(self.current_session, "thread_id", ""),
                **dict(event.payload or {}),
            )

    def _maybe_reset_summary_progress_after_compaction(self, event: StreamEvent) -> None:
        if event.type != "summary_notice" or self.config is None:
            return
        payload = dict(event.payload or {})
        if str(payload.get("kind", "") or "") != "auto_summary":
            return

        if self.current_session is not None:
            self.current_session.last_run_stats = ""
            try:
                self.store.save_active_session(self.current_session, touch=False, set_active=True)
            except Exception:
                logger.debug("Failed to persist cleared run stats after auto-summary.", exc_info=True)
        self._refresh_live_summary_progress_from_state_soon()

    def _refresh_live_summary_progress_from_state_soon(self) -> None:
        if self.config is None or self.agent_app is None or self.current_session is None:
            return

        async def _refresh() -> None:
            await self._reset_live_summary_progress_from_state(emit=True)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(lambda: asyncio.create_task(_refresh()))
                return
            self._run(_refresh())
            return
        loop.create_task(_refresh())

    def _maybe_emit_live_summary_progress(self, event: StreamEvent) -> None:
        if event.type != "tool_finished" or self.config is None:
            return
        payload = dict(event.payload or {})
        content = str(payload.get("content", "") or "")
        if not content:
            return
        tool_name = str(payload.get("name", "") or "tool").strip() or "tool"
        tool_tokens = estimate_tokens(
            [ToolMessage(content=content, tool_call_id="live-summary-progress", name=tool_name)]
        )
        self._active_summary_estimated_tokens = max(0, self._active_summary_estimated_tokens + tool_tokens)
        self._active_summary_message_count = max(0, self._active_summary_message_count + 1)
        self.event_emitted.emit(StreamEvent("summary_progress", self._build_live_summary_progress_payload()))

    def _build_live_summary_progress_payload(self) -> dict[str, Any]:
        if self.config is None:
            return {}
        threshold = max(0, int(getattr(self.config, "summary_threshold", 0) or 0))
        estimated = max(0, int(self._active_summary_estimated_tokens or 0))
        reserved = max(0, int(self._active_summary_reserved_tokens or 0))
        memory_tokens = max(0, int(self._active_summary_memory_tokens or 0))
        has_summary = bool(self._active_summary_has_summary)
        trigger = summary_trigger_tokens(threshold, has_summary=has_summary)
        return {
            "estimated_tokens": estimated,
            "threshold": threshold,
            "trigger_tokens": trigger,
            "remaining_tokens": max(0, trigger - estimated),
            "reserved_tokens": reserved,
            "summary_tokens": memory_tokens,
            "provider_input_tokens": max(0, int(self._active_provider_input_tokens or 0)),
            "progress": summary_remaining_ratio(
                estimated,
                threshold=threshold,
            ),
            "message_count": max(0, int(self._active_summary_message_count or 0)),
            "has_summary": has_summary,
            # Live updates are approximate; the full checkpoint payload decides readiness.
            "will_summarize": False,
            "live": True,
        }

    async def _reset_live_summary_progress_from_state(self, *, emit: bool = False) -> None:
        self._active_summary_estimated_tokens = 0
        self._active_summary_message_count = 0
        self._active_summary_has_summary = False
        self._active_summary_reserved_tokens = 0
        self._active_summary_memory_tokens = 0
        self._active_provider_input_tokens = 0
        if self.config is None or self.agent_app is None or self.current_session is None:
            return
        try:
            from ui.runtime_payloads import load_state_values

            values = await load_state_values(self.agent_app, self.current_session.thread_id)
        except Exception:
            logger.debug("Failed to refresh live summary progress baseline.", exc_info=True)
            return
        progress = build_summary_progress_payload(self.config, values)
        self._active_summary_estimated_tokens = max(0, int(progress.get("estimated_tokens", 0) or 0))
        self._active_summary_message_count = max(0, int(progress.get("message_count", 0) or 0))
        self._active_summary_has_summary = bool(progress.get("has_summary"))
        self._active_summary_reserved_tokens = max(0, int(progress.get("reserved_tokens", 0) or 0))
        self._active_summary_memory_tokens = max(0, int(progress.get("summary_tokens", 0) or 0))
        self._active_provider_input_tokens = max(0, int(progress.get("provider_input_tokens", 0) or 0))
        if emit:
            self.event_emitted.emit(StreamEvent("summary_progress", self._build_live_summary_progress_payload()))

    @Slot()
    def initialize(self) -> None:
        try:
            self._run(self._initialize_async())
        except Exception as exc:
            self.initialization_failed.emit(str(exc))

    @Slot(bool)
    def reinitialize(self, force_new_session: bool = False) -> None:
        if self._loop is None:
            try:
                self._run(self._initialize_async(force_new_session=force_new_session))
            except Exception as exc:
                self.initialization_failed.emit(str(exc))
            return

        try:
            self._run(self._shutdown_async())
            self._run(self._initialize_async(force_new_session=force_new_session))
            self.event_emitted.emit(StreamEvent("chat_reset", {}))
        except Exception as exc:
            logger.exception("Reinitialization failed:")
            self.initialization_failed.emit(str(exc))

    async def _initialize_async(self, force_new_session: bool = False) -> None:
        self.base_config = _runtime_module().setup_runtime()
        self.config = self.base_config
        self.profile_store = ModelProfileStore(self._profile_config_path())
        self.model_profiles = self.profile_store.load_or_initialize(
            self._profile_bootstrap_env_from_config(self.base_config)
        )
        try:
            self.config = self._build_config_for_active_profile(self.model_profiles)
        except Exception:
            sanitized = dict(self.model_profiles)
            sanitized["active_profile"] = None
            self.model_profiles = self.profile_store.save(sanitized)
            self.config = self.base_config
        self._runtime_profile_id = self._selected_profile_id(self.model_profiles)
        self.ui_run_logger = JsonlRunLogger(self.config.run_log_dir)
        self.store = SessionStore(self.config.session_state_path)
        self.agent_app, self.tool_registry = await _runtime_module().build_agent_app(
            self.config
        )
        self.checkpoint_runtime = getattr(self.tool_registry, "checkpoint_runtime", None)
        pending_mcp_server = self._pending_mcp_server_name
        if pending_mcp_server:
            status = next(
                (item for item in self.tool_registry.mcp_server_status if item.get("server") == pending_mcp_server),
                {},
            )
            if status.get("error"):
                raise RuntimeError(str(status["error"]))
        self._set_effective_model_capabilities(getattr(self.tool_registry, "model_capabilities", None))
        self._configure_cli_output_bridge()
        selected_session = self._select_session_for_project(force_new_session=force_new_session)
        self.current_session = self._activate_session_with_workdir_or_fallback(
            selected_session,
            fallback_event_type="session_restore_fallback",
            notice_message=(
                "Could not open the restored chat workspace. "
                "Created a new chat in the current project."
            ),
        )

        await self._repair_current_session_if_needed()

        payload = await build_ui_payload(
            self.config,
            self.tool_registry,
            self.store,
            self.current_session,
            model_profiles=self.model_profiles,
            model_capabilities=self.model_capabilities,
            agent_app=self.agent_app,
            include_transcript=True,
        )
        self._sync_pending_user_choice_from_payload(payload)
        self.initialized.emit(payload)
        self.session_changed.emit(payload)

    def _apply_tool_availability_changes(self, changes: list[dict[str, Any]]) -> None:
        if self._is_busy or self._awaiting_approval or self.tool_registry is None:
            return

        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in changes:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip()
            name = str(item.get("name") or "").strip()
            if kind not in {"server", "tool"} or not name or (kind, name) in seen:
                continue
            seen.add((kind, name))
            normalized.append({"kind": kind, "name": name, "enabled": bool(item.get("enabled"))})
        if not normalized:
            return

        registry = self.tool_registry
        previous: list[dict[str, Any]] = []
        requires_reinitialize = False
        try:
            for item in normalized:
                if item["kind"] == "server":
                    current = registry.mcp_config.get(item["name"])
                    previous_enabled = bool(current.get("enabled", True)) if isinstance(current, dict) else None
                    if previous_enabled is None:
                        continue
                    previous.append({**item, "enabled": previous_enabled})
                    registry.set_mcp_server_enabled(item["name"], item["enabled"])
                    requires_reinitialize = True
                else:
                    previous_enabled = item["name"] not in registry.disabled_local_tools
                    previous.append({**item, "enabled": previous_enabled})
                    registry.set_tool_enabled(item["name"], item["enabled"])

            if requires_reinitialize:
                enabled_servers = [
                    item["name"]
                    for item in normalized
                    if item["kind"] == "server" and item["enabled"]
                ]
                self._pending_mcp_server_name = enabled_servers[-1] if enabled_servers else ""
                self._run(self._shutdown_async())
                self._run(self._initialize_async())
            else:
                self._run(self._coordinator.emit_session_payload(include_transcript=False))
        except Exception as exc:
            logger.exception("Failed to apply tool availability changes:")
            for item in reversed(previous):
                try:
                    if item["kind"] == "server":
                        registry.set_mcp_server_enabled(item["name"], item["enabled"])
                    else:
                        registry.set_tool_enabled(item["name"], item["enabled"])
                except Exception:
                    logger.exception("Failed to restore tool state after apply failure:")
            self.initialization_failed.emit(f"Failed to apply tool change: {exc}")
        finally:
            self._pending_mcp_server_name = ""

    @Slot(object)
    def apply_tool_availability_changes(self, changes: object) -> None:
        self._apply_tool_availability_changes(changes if isinstance(changes, list) else [])

    @Slot(object)
    def set_tool_enabled(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        self._apply_tool_availability_changes([{"kind": "tool", **data}])

    @Slot(object)
    def set_mcp_server_enabled(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        self._apply_tool_availability_changes([{"kind": "server", **data}])

    @Slot(object)
    def start_run(self, user_text: object) -> None:
        request_payload = normalize_request_payload(user_text)
        if not request_has_content(request_payload) or self._is_busy or self._awaiting_approval or not self.current_session:
            return
        if find_active_profile(self.model_profiles) is None:
            profiles_exist = bool((self.model_profiles or {}).get("profiles"))
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": (
                            "No enabled models available. Open Settings and enable a profile."
                            if profiles_exist
                            else "No models configured. Open Settings and add a profile."
                        ),
                        "kind": "model_missing",
                    },
                )
            )
            return
        if request_payload["attachments"] and not bool(self.model_capabilities.get("image_input_supported")):
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": "Current model does not accept image input. Switch to an image-capable model or remove the images.",
                        "kind": "image_input_unsupported",
                        "level": "warning",
                    },
                )
            )
            return
        self._set_busy(True)
        try:
            if not self._run(self._ensure_runtime_matches_selected_profile()):
                self._set_busy(False)
                return
            persisted_now = self._ensure_current_session_persisted()
            if persisted_now:
                self._run(self._emit_session_payload(include_transcript=False))
            self._run(self._start_run_async(request_payload))
        except Exception as exc:
            self.event_emitted.emit(StreamEvent("run_failed", {"message": str(exc)}))
            self._set_busy(False)

    def _apply_fallback_chat_title(self, user_text: str) -> None:
        if self.current_session is None:
            return
        self._log_ui_run_event("chat_title_generation_fallback")
        try:
            title = generate_chat_title(user_text)
            if str(self.current_session.title or "").strip().startswith("New Chat ["):
                title = append_project_label(title, self.current_session.project_path)
            if not title or title == self.current_session.title:
                return
            self.current_session.title = title
            self.store.save_active_session(self.current_session, touch=False, set_active=True)
            self._run(self._emit_session_payload(include_transcript=False))
        except Exception:
            logger.exception("chat_title_fallback_apply_failed")

    async def _start_run_async(self, user_text: object) -> None:
        request_payload = normalize_request_payload(user_text)
        if self.current_session is not None:
            current_title = str(self.current_session.title or "").strip()
            is_default_title = current_title in {"New Chat", ""} or current_title.startswith("New Chat [")
            if is_default_title:
                try:
                    from core.providers.factory import create_runtime_llm
                    title_llm = create_runtime_llm(self.config)
                    generated = await generate_chat_title_with_llm(title_llm, request_payload["text"], logger)
                    if generated:
                        self.current_session.title = generated
                        self.store.save_active_session(self.current_session, touch=False, set_active=True)
                        await self._emit_session_payload(include_transcript=False)
                    else:
                        self._apply_fallback_chat_title(request_payload["text"])
                except Exception:
                    logger.exception("chat_title_generation_error")
                    self._apply_fallback_chat_title(request_payload["text"])
        self._active_run_elapsed_seconds = 0.0
        self._active_request_has_images = bool(request_payload["attachments"])
        await self._reset_live_summary_progress_from_state()
        if self.current_session is not None:
            self.current_session.last_run_stats = ""
        repair_notices = await self._repair_current_session_if_needed()
        if repair_notices:
            self._log_ui_run_event("pre_run_session_repair", repaired_count=len(repair_notices))
        initial_state = build_initial_state(request_payload, session_id=self.current_session.session_id)
        self.event_emitted.emit(
            StreamEvent(
                "run_started",
                {
                    "text": request_payload["text"],
                    "attachments": request_payload["attachments"],
                },
            )
        )
        await self._run_graph_payload(initial_state)

    # --- Stream-interruption recovery helpers ---

    @staticmethod
    def _classify_stream_error(error_message: str, error_details: dict[str, Any] | None = None) -> dict[str, Any]:
        """Classify a provider stream error to drive retry strategy.

        Returns a dict with:
          * ``kind``: "rate_limit" | "timeout" | "server_error" | "network" | "unknown"
          * ``retryable``: bool — whether a retry makes sense
          * ``label``: human-readable label for UI notices
        """
        details = error_details or {}
        status = details.get("http_status")
        if details.get("retry_exhausted"):
            return {"kind": "retry_exhausted", "retryable": False, "label": "Retry budget exhausted"}
        if status is not None and 400 <= int(status) < 500 and int(status) not in {408, 429}:
            return {"kind": "client_error", "retryable": False, "label": f"HTTP {status}"}

        text = " ".join(str(error_message or "").lower().split())

        if any(m in text for m in ("429", "rate limit", "rate_limit", "too many requests")):
            return {"kind": "rate_limit", "retryable": True, "label": "Rate limit"}
        if any(m in text for m in ("timeout", "timed out", "deadline exceeded")):
            return {"kind": "timeout", "retryable": True, "label": "Timeout"}
        if any(m in text for m in ("502", "503", "504", "bad gateway", "service unavailable", "gateway timeout")):
            return {"kind": "server_error", "retryable": True, "label": "Server error"}
        if any(m in text for m in ("connection", "network", "reset by peer", "broken pipe", "eof", "stream ended", "disconnected")):
            return {"kind": "network", "retryable": True, "label": "Network error"}
        return {"kind": "unknown", "retryable": False, "label": "Stream error"}

    @staticmethod
    def _stream_retry_backoff(attempt: int, base_delay: float, error_kind: str) -> float:
        """Compute exponential backoff with jitter for stream-repair retries.

        Uses the same pattern as ``_invoke_llm_with_retry``: delay = base * 2^attempt + jitter.
        Rate-limit errors get a longer base to avoid hammering the provider.
        """
        effective_base = base_delay * 1.5 if error_kind == "rate_limit" else base_delay
        return effective_base * (2 ** attempt) + random.uniform(0, effective_base)

    async def _run_graph_payload(self, payload: dict | Command | None) -> None:
        try:
            stream_repair_resume_attempts = 0
            max_stream_repair_resumes = max(0, min(int(getattr(self.config, "max_retries", 1) or 1), 2))
            while True:
                stream = self.agent_app.astream(
                    payload,
                    config=build_graph_config(self.current_session.thread_id, self.config.max_loops),
                    stream_mode=["messages", "updates", "custom"],
                    version="v2",
                )
                processor = StreamProcessor(
                    self._emit_stream_event,
                    text_max_chars=self.config.stream_text_max_chars,
                    events_max=self.config.stream_events_max,
                    tool_buffer_max=self.config.stream_tool_buffer_max,
                    base_elapsed_seconds=self._active_run_elapsed_seconds,
                    tool_sources={
                        name: str(getattr(metadata, "source", "local") or "local")
                        for name, metadata in getattr(self.tool_registry, "tool_metadata", {}).items()
                    },
                    mcp_tool_servers={
                        tool_name: str(status.get("server", "") or "")
                        for status in getattr(self.tool_registry, "mcp_server_status", [])
                        for tool_name in status.get("loaded_tools", [])
                    },
                )
                result = await processor.process_stream(stream)
                self._active_run_elapsed_seconds = result.elapsed_seconds

                if result.cancelled:
                    self._log_ui_run_event("run_cancelled", interrupted_tool_count=len(result.cancelled_tools))
                    await self._repair_current_session_if_needed()
                    self._refresh_model_profiles_from_store()
                    self._active_run_elapsed_seconds = 0.0
                    self._active_request_has_images = False
                    self._set_busy(False)
                    return

                if result.failed:
                    self._log_ui_run_event(
                        "run_failed",
                        **{"message": result.error_message, **result.error_details},
                    )
                    error_info = self._classify_stream_error(result.error_message, result.error_details)
                    repair_notices = await self._repair_current_session_if_needed()
                    if (
                        error_info["retryable"]
                        and repair_notices
                        and stream_repair_resume_attempts < max_stream_repair_resumes
                    ):
                        stream_repair_resume_attempts += 1
                        backoff = self._stream_retry_backoff(
                            stream_repair_resume_attempts - 1,
                            float(getattr(self.config, "retry_delay", 2) or 2),
                            error_info["kind"],
                        )
                        self._log_ui_run_event(
                            "run_auto_continued_after_stream_repair",
                            attempt=stream_repair_resume_attempts,
                            max_attempts=max_stream_repair_resumes,
                            message=result.error_message,
                            error_kind=error_info["kind"],
                            backoff_seconds=round(backoff, 2),
                        )
                        self.event_emitted.emit(
                            StreamEvent(
                                "summary_notice",
                                {
                                    "message": (
                                        f"Provider stream interrupted ({error_info['label']}). "
                                        "History was repaired automatically; "
                                        f"continuing the run in {backoff:.1f}s…"
                                    ),
                                    "kind": "stream_repair_auto_continue",
                                    "level": "warning",
                                },
                            )
                        )
                        await asyncio.sleep(backoff)
                        payload = None
                        continue
                    self._refresh_model_profiles_from_store()
                    self._active_run_elapsed_seconds = 0.0
                    self._active_request_has_images = False
                    self._set_busy(False)
                    return

                if result.interrupt is None:
                    repair_notices = await self._repair_current_session_if_needed()
                    if repair_notices and stream_repair_resume_attempts < max_stream_repair_resumes:
                        stream_repair_resume_attempts += 1
                        backoff = self._stream_retry_backoff(
                            stream_repair_resume_attempts - 1,
                            float(getattr(self.config, "retry_delay", 2) or 2),
                            "network",
                        )
                        self._log_ui_run_event(
                            "run_auto_continued_after_stream_repair",
                            attempt=stream_repair_resume_attempts,
                            max_attempts=max_stream_repair_resumes,
                            phase="post_success",
                            repaired_count=len(repair_notices),
                            backoff_seconds=round(backoff, 2),
                        )
                        self.event_emitted.emit(
                            StreamEvent(
                                "summary_notice",
                                {
                                    "message": (
                                        "Provider stream ended with an incomplete tool call. "
                                        "History was repaired automatically; "
                                        f"continuing the run in {backoff:.1f}s..."
                                    ),
                                    "kind": "stream_repair_auto_continue",
                                    "level": "warning",
                                },
                            )
                        )
                        await asyncio.sleep(backoff)
                        payload = None
                        continue

                    self._refresh_model_profiles_from_store()
                    if self.current_session is not None:
                        self.current_session.last_run_stats = str(result.stats or "")
                    self.store.save_active_session(self.current_session, touch=True, set_active=True)
                    # Stream events already rendered this run. Reloading the persisted
                    # transcript here rebuilds widgets and resets the reading position.
                    await self._emit_session_payload(include_transcript=False)
                    self._active_run_elapsed_seconds = 0.0
                    self._active_request_has_images = False
                    self._set_busy(False)
                    return

                interrupt_kind = str((result.interrupt or {}).get("kind", "") or "")
                if interrupt_kind == "user_choice":
                    self._awaiting_approval = True
                    self._awaiting_interrupt_kind = "user_choice"
                    self._pending_user_choice_type = str(
                        (result.interrupt or {}).get("choice_type", "clarification") or "clarification"
                    )
                    self.user_choice_requested.emit(build_user_choice_payload(result.interrupt))
                    self._set_busy(False)
                    return

                approval_payload = build_approval_payload(result.interrupt, self.current_session)
                if self.current_session.approval_mode == APPROVAL_MODE_ALWAYS:
                    self.event_emitted.emit(
                        StreamEvent("approval_resolved", {"approved": True, "always": True, "auto": True})
                    )
                    payload = Command(resume={"approved": True})
                    continue

                self._awaiting_approval = True
                self._awaiting_interrupt_kind = interrupt_kind or "tool_approval"
                self.approval_requested.emit(approval_payload)
                self._set_busy(False)
                return
        except Exception as exc:
            self._refresh_model_profiles_from_store()
            self._active_run_elapsed_seconds = 0.0
            if self._active_request_has_images:
                self.event_emitted.emit(
                    StreamEvent(
                        "summary_notice",
                        {
                            "message": (
                                "The model rejected the attached image. "
                                "Check the profile image-input setting or send the request without an image."
                            ),
                            "kind": "image_input_failed",
                            "level": "warning",
                        },
                    )
                )
            self._active_request_has_images = False
            self.event_emitted.emit(StreamEvent("run_failed", {"message": str(exc)}))
            self._set_busy(False)

    @Slot(bool, bool)
    def resume_approval(self, approved: bool, always: bool = False) -> None:
        if not self.current_session or not self._awaiting_approval or self._awaiting_interrupt_kind == "user_choice":
            return
        self._set_busy(True)
        try:
            self._run(self._resume_approval_async(approved, always))
        except Exception as exc:
            self.event_emitted.emit(StreamEvent("run_failed", {"message": str(exc)}))
            self._set_busy(False)

    async def _resume_approval_async(self, approved: bool, always: bool) -> None:
        self._awaiting_approval = False
        self._awaiting_interrupt_kind = ""
        self._pending_user_choice_type = ""

        if always:
            self.current_session.approval_mode = APPROVAL_MODE_ALWAYS
            self.store.save_active_session(self.current_session, touch=False, set_active=True)
            await self._emit_session_payload(include_transcript=False)

        self.event_emitted.emit(
            StreamEvent("approval_resolved", {"approved": approved, "always": always, "auto": False})
        )
        await self._run_graph_payload(Command(resume={"approved": approved}))

    @Slot(str)
    def resume_user_choice(self, chosen: str) -> None:
        if not self.current_session or not self._awaiting_approval or self._awaiting_interrupt_kind != "user_choice":
            return
        chosen_text = str(chosen or "").strip()
        if not chosen_text:
            return
        self._set_busy(True)
        try:
            self._run(self._resume_user_choice_async(chosen_text))
        except Exception as exc:
            self.event_emitted.emit(StreamEvent("run_failed", {"message": str(exc)}))
            self._set_busy(False)

    async def _resume_user_choice_async(self, chosen: str) -> None:
        choice_type = str(getattr(self, "_pending_user_choice_type", "") or "clarification")
        self._awaiting_approval = False
        self._awaiting_interrupt_kind = ""
        self._pending_user_choice_type = ""
        self.event_emitted.emit(StreamEvent("user_choice_resolved", {"chosen": chosen, "choice_type": choice_type}))
        await self._run_graph_payload(Command(resume=chosen))

    def _request_task_cancellation(self) -> None:
        with self._task_state_lock:
            loop = self._loop
            task = self._current_task
        if loop is None or task is None:
            return

        def _cancel_task() -> None:
            if not task.done():
                task.cancel()

        try:
            loop.call_soon_threadsafe(_cancel_task)
        except RuntimeError:
            return

    @Slot()
    def stop_run(self) -> None:
        self._request_task_cancellation()

    def request_stop(self) -> None:
        """Request cancellation without requiring the worker event loop to dispatch a Qt slot."""
        self._request_task_cancellation()

    @Slot()
    def new_session(self) -> None:
        if not self.current_session or self._is_busy or self._awaiting_approval:
            return
        checkpoint_info = self.tool_registry.checkpoint_info
        self.current_session = self.store.new_session(
            checkpoint_backend=checkpoint_info.get("resolved_backend", self.config.checkpoint_backend),
            checkpoint_target=checkpoint_info.get("target", "unknown"),
            project_path=self._current_project_path(),
            persisted=False,
        )
        self.current_session.approval_mode = APPROVAL_MODE_PROMPT
        self.store.save_active_session(self.current_session, touch=False, set_active=True)
        self._sync_tool_registry_workdir(self.current_session.project_path)
        self.event_emitted.emit(StreamEvent("chat_reset", {}))
        self._run(self._emit_session_payload(include_transcript=True))

    @Slot(str)
    def switch_session(self, session_id: str) -> None:
        if not session_id or self._is_busy or self._awaiting_approval:
            return
        target = self.store.get_session(session_id)
        if target is None or target.session_id == getattr(self.current_session, "session_id", ""):
            return
        self._run(self._switch_session_async(target))

    async def _switch_session_async(self, target: SessionSnapshot) -> None:
        self.current_session = self._activate_session_with_workdir_or_fallback(
            target,
            fallback_event_type="session_switch_fallback",
            notice_message=(
                "Could not open the selected chat workspace. "
                "Created a new chat in the current project."
            ),
        )
        await self._repair_current_session_if_needed()
        self.event_emitted.emit(StreamEvent("chat_reset", {}))
        await self._emit_session_payload(include_transcript=True)

    @Slot(str)
    def delete_session(self, session_id: str) -> None:
        if not session_id or self._is_busy or self._awaiting_approval:
            return
        self._run(self._delete_session_async(session_id))

    async def _delete_session_async(self, session_id: str) -> None:
        if not self.store or not self.current_session:
            return

        deleted = self.store.delete_session(session_id)
        if not deleted:
            self.event_emitted.emit(
                StreamEvent("summary_notice", {"message": "Chat not found in history.", "kind": "session_delete"})
            )
            return

        active_deleted = self.current_session.session_id == session_id
        self.event_emitted.emit(
            StreamEvent("summary_notice", {"message": "Chat deleted from history.", "kind": "session_delete"})
        )

        if not active_deleted:
            self.store.save_active_session(self.current_session, touch=False, set_active=True)
            await self._emit_session_payload(include_transcript=False)
            return

        replacement = self.store.get_last_active_session()
        if replacement is None or not self._project_path_is_valid(replacement.project_path):
            fallback_project = self._current_project_path()
            replacement = self._create_new_session_for_project(fallback_project, with_project_label=True)

        self.current_session = self._activate_session_with_workdir_or_fallback(
            replacement,
            fallback_event_type="session_delete_fallback",
            notice_message=(
                "Could not open the next chat workspace. "
                "Created a new chat in the current project."
            ),
        )

        await self._repair_current_session_if_needed()
        self.event_emitted.emit(StreamEvent("chat_reset", {}))
        await self._emit_session_payload(include_transcript=True)

    @Slot(str)
    def delete_project(self, project_path: str) -> None:
        if not project_path or self._is_busy or self._awaiting_approval:
            return
        self._run(self._delete_project_async(project_path))

    async def _delete_project_async(self, project_path: str) -> None:
        if not self.store or not self.current_session:
            return

        deleted_ids = self.store.delete_project(project_path)
        if not deleted_ids:
            self.event_emitted.emit(
                StreamEvent("summary_notice", {"message": "Project not found in chat history.", "kind": "project_delete"})
            )
            return

        active_deleted = self.current_session.session_id in set(deleted_ids)
        chats = "chat" if len(deleted_ids) == 1 else "chats"
        self.event_emitted.emit(
            StreamEvent(
                "summary_notice",
                {"message": f"Deleted {len(deleted_ids)} {chats} of the project from history.", "kind": "project_delete"},
            )
        )

        if not active_deleted:
            self.store.save_active_session(self.current_session, touch=False, set_active=True)
            await self._emit_session_payload(include_transcript=False)
            return

        replacement = self.store.get_last_active_session()
        if replacement is None or not self._project_path_is_valid(replacement.project_path):
            fallback_project = self._current_project_path()
            replacement = self._create_new_session_for_project(fallback_project, with_project_label=True)

        self.current_session = self._activate_session_with_workdir_or_fallback(
            replacement,
            fallback_event_type="project_delete_fallback",
            notice_message=(
                "Could not open the next chat workspace. "
                "Created a new chat in the current project."
            ),
        )

        await self._repair_current_session_if_needed()
        self.event_emitted.emit(StreamEvent("chat_reset", {}))
        await self._emit_session_payload(include_transcript=True)

    @Slot(str)
    def set_active_profile(self, profile_id: str) -> None:
        if self._is_busy or self._awaiting_approval:
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": "Cannot switch models while a run is active.",
                        "kind": "model_switch_blocked",
                    },
                )
            )
            return
        if self.profile_store is None:
            return
        try:
            self._run(self._set_active_profile_async(profile_id))
        except Exception as exc:
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": f"Failed to switch models: {exc}",
                        "kind": "model_switch_failed",
                    },
                )
            )

    async def _set_active_profile_async(self, profile_id: str) -> None:
        target_id = str(profile_id or "").strip()
        current = normalize_profiles_payload(self.model_profiles)
        known_ids = {
            str(profile.get("id") or "").strip()
            for profile in current.get("profiles", []) or []
            if isinstance(profile, dict)
        }
        if target_id and target_id not in known_ids:
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": "The selected profile was not found.",
                        "kind": "model_switch_failed",
                    },
                )
            )
            return
        if target_id == str(current.get("active_profile") or ""):
            return
        candidate = dict(current)
        candidate["active_profile"] = target_id or None
        target_profile = find_active_profile(candidate)
        target_model = str((target_profile or {}).get("model") or "").strip()
        success_notice_message = (
            f"Model switched to {target_model}."
            if target_model
            else "Model switched."
        )
        await self._apply_model_profiles(
            candidate,
            success_notice_kind="model_switched",
            success_notice_message=success_notice_message,
            sync_runtime=True,
            runtime_failure_kind="model_switch_failed",
            runtime_failure_message_prefix="Failed to apply the selected model",
        )

    @Slot(object)
    def save_profiles(self, config_payload: object) -> None:
        if self._is_busy or self._awaiting_approval:
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": "Cannot save profiles while a run is active.",
                        "kind": "profiles_save_blocked",
                    },
                )
            )
            return
        if self.profile_store is None:
            return
        try:
            payload = config_payload if isinstance(config_payload, dict) else {}
            self._run(self._save_profiles_async(payload))
        except Exception as exc:
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": f"Failed to save profiles: {exc}",
                        "kind": "profiles_save_failed",
                    },
                )
            )

    async def _save_profiles_async(self, config_payload: dict[str, Any]) -> None:
        await self._apply_model_profiles(
            config_payload,
            success_notice_kind="profiles_saved",
            success_notice_message="Model profiles saved.",
            sync_runtime=True,
            runtime_failure_kind="profiles_apply_failed",
            runtime_failure_message_prefix="Profiles were saved, but the active model could not be applied",
        )

    @Slot()
    def shutdown(self) -> None:
        try:
            self._run(self._shutdown_async())
        finally:
            self._close_event_loop()
            self.shutdown_complete.emit()

    def _close_event_loop(self) -> None:
        """Finalize pending async work before releasing the worker event loop."""
        with self._task_state_lock:
            loop = self._loop
            self._loop = None
        if loop is None:
            return
        try:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            # Async HTTP transports use async generators during streaming. They
            # must be finalized while the loop is still alive; otherwise their
            # finalizers can create AsyncClient.aclose() after loop.close().
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            logger.debug("Failed to fully finalize the runtime event loop.", exc_info=True)
        finally:
            loop.close()

    async def _shutdown_async(self) -> None:
        self._clear_cli_output_bridge()
        tool_registry = self.tool_registry
        self.tool_registry = None
        self.agent_app = None
        self.checkpoint_runtime = None
        self.ui_run_logger = None
        await close_runtime_resources(tool_registry)


class AgentRuntimeController(QObject):
    initialized = Signal(object)
    initialization_failed = Signal(str)
    event_emitted = Signal(object)
    approval_requested = Signal(object)
    user_choice_requested = Signal(object)
    session_changed = Signal(object)
    busy_changed = Signal(bool)

    _initialize_requested = Signal()
    _start_run_requested = Signal(object)
    _stop_run_requested = Signal()
    _resume_requested = Signal(bool, bool)
    _resume_user_choice_requested = Signal(str)
    _new_session_requested = Signal()
    _switch_session_requested = Signal(str)
    _delete_session_requested = Signal(str)
    _delete_project_requested = Signal(str)
    _set_active_profile_requested = Signal(str)
    _save_profiles_requested = Signal(object)
    _apply_tool_availability_changes_requested = Signal(object)
    _reinitialize_requested = Signal(bool)
    _shutdown_requested = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: AgentRunWorker | None = None
        self._worker_busy = False
        self._force_stop_timeout_ms = 5000
        self._force_stop_timer = QTimer(self)
        self._force_stop_timer.setSingleShot(True)
        self._force_stop_timer.timeout.connect(self._force_stop_hung_worker)
        self._pending_tool_availability: dict[tuple[str, str], dict[str, Any]] = {}
        self._tool_availability_timer = QTimer(self)
        self._tool_availability_timer.setSingleShot(True)
        self._tool_availability_timer.timeout.connect(self._flush_tool_availability_changes)
        self._start_worker_thread()

    def _start_worker_thread(self) -> None:
        self._thread = QThread(self)
        self._worker = AgentRunWorker()
        self._worker.moveToThread(self._thread)

        self._initialize_requested.connect(self._worker.initialize)
        self._reinitialize_requested.connect(self._worker.reinitialize)
        self._start_run_requested.connect(self._worker.start_run)
        self._stop_run_requested.connect(self._worker.stop_run, Qt.QueuedConnection)
        self._resume_requested.connect(self._worker.resume_approval)
        self._resume_user_choice_requested.connect(self._worker.resume_user_choice)
        self._new_session_requested.connect(self._worker.new_session)
        self._switch_session_requested.connect(self._worker.switch_session)
        self._delete_session_requested.connect(self._worker.delete_session)
        self._delete_project_requested.connect(self._worker.delete_project)
        self._set_active_profile_requested.connect(self._worker.set_active_profile)
        self._save_profiles_requested.connect(self._worker.save_profiles)
        self._apply_tool_availability_changes_requested.connect(self._worker.apply_tool_availability_changes)
        self._shutdown_requested.connect(self._worker.shutdown)

        self._worker.initialized.connect(self.initialized)
        self._worker.initialization_failed.connect(self.initialization_failed)
        self._worker.event_emitted.connect(self.event_emitted)
        self._worker.approval_requested.connect(self.approval_requested)
        self._worker.user_choice_requested.connect(self.user_choice_requested)
        self._worker.session_changed.connect(self.session_changed)
        self._worker.busy_changed.connect(self._on_worker_busy_changed)
        self._worker.shutdown_complete.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)

        self._thread.start()

    def _disconnect_worker_thread(self) -> None:
        worker = self._worker
        thread = self._thread
        if worker is None:
            return
        for signal, slot in (
            (self._initialize_requested, worker.initialize),
            (self._reinitialize_requested, worker.reinitialize),
            (self._start_run_requested, worker.start_run),
            (self._stop_run_requested, worker.stop_run),
            (self._resume_requested, worker.resume_approval),
            (self._resume_user_choice_requested, worker.resume_user_choice),
            (self._new_session_requested, worker.new_session),
            (self._switch_session_requested, worker.switch_session),
            (self._delete_session_requested, worker.delete_session),
            (self._delete_project_requested, worker.delete_project),
            (self._set_active_profile_requested, worker.set_active_profile),
            (self._save_profiles_requested, worker.save_profiles),
            (self._apply_tool_availability_changes_requested, worker.apply_tool_availability_changes),



            (self._shutdown_requested, worker.shutdown),

            (worker.initialized, self.initialized),
            (worker.initialization_failed, self.initialization_failed),
            (worker.event_emitted, self.event_emitted),
            (worker.approval_requested, self.approval_requested),
            (worker.user_choice_requested, self.user_choice_requested),
            (worker.session_changed, self.session_changed),
            (worker.busy_changed, self._on_worker_busy_changed),
        ):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        if thread is not None:
            try:
                worker.shutdown_complete.disconnect(thread.quit)
            except (RuntimeError, TypeError):
                pass
            try:
                thread.finished.disconnect(worker.deleteLater)
            except (RuntimeError, TypeError):
                pass

    def _stop_worker_thread(self, *, force: bool) -> None:
        thread = self._thread
        worker = self._worker
        if thread is None:
            self._worker = None
            return
        if force and thread.isRunning():
            self._disconnect_worker_thread()
            logger.warning("Force-stopping unresponsive runtime worker thread after stop request.")
            thread.terminate()
            thread.wait(3000)
        elif thread.isRunning():
            self._shutdown_requested.emit()
            if not thread.wait(3000):
                self._disconnect_worker_thread()
                logger.warning("Runtime worker shutdown timed out; terminating thread.")
                thread.terminate()
                thread.wait(3000)
            else:
                self._disconnect_worker_thread()
        else:
            self._disconnect_worker_thread()
        if worker is not None:
            try:
                worker.deleteLater()
            except RuntimeError:
                pass
        self._thread = None
        self._worker = None

    @Slot(bool)
    def _on_worker_busy_changed(self, busy: bool) -> None:
        self._worker_busy = bool(busy)
        if not busy and self._force_stop_timer.isActive():
            self._force_stop_timer.stop()
        self.busy_changed.emit(busy)

    @Slot()
    def _force_stop_hung_worker(self) -> None:
        thread = self._thread
        if thread is None or not thread.isRunning():
            return
        self.event_emitted.emit(
            StreamEvent(
                "run_failed",
                {
                    "message": (
                        "Stopped by user. The provider did not respond to cancellation, "
                        "so the runtime worker was restarted."
                    ),
                    "forced": True,
                },
            )
        )
        self._worker_busy = False
        self.busy_changed.emit(False)
        self._stop_worker_thread(force=True)
        self._start_worker_thread()
        self._initialize_requested.emit()

    def initialize(self) -> None:
        self._initialize_requested.emit()

    def reinitialize(self, force_new_session: bool = False) -> None:
        self._reinitialize_requested.emit(force_new_session)

    def start_run(self, user_text: object) -> None:
        self._start_run_requested.emit(user_text)

    def stop_run(self) -> None:
        worker = self._worker
        if worker is not None:
            worker.request_stop()
        self._stop_run_requested.emit()
        if self._worker_busy:
            self._force_stop_timer.start(self._force_stop_timeout_ms)

    def resume_approval(self, approved: bool, always: bool = False) -> None:
        self._resume_requested.emit(approved, always)

    def resume_user_choice(self, chosen: str) -> None:
        self._resume_user_choice_requested.emit(chosen)

    def new_session(self) -> None:
        self._new_session_requested.emit()

    def switch_session(self, session_id: str) -> None:
        self._switch_session_requested.emit(session_id)

    def delete_session(self, session_id: str) -> None:
        self._delete_session_requested.emit(session_id)

    def delete_project(self, project_path: str) -> None:
        self._delete_project_requested.emit(project_path)

    def set_active_profile(self, profile_id: str) -> None:
        self._set_active_profile_requested.emit(profile_id)

    def save_profiles(self, config_payload: dict[str, Any]) -> None:
        self._save_profiles_requested.emit(config_payload)

    def _flush_tool_availability_changes(self) -> None:
        changes = list(self._pending_tool_availability.values())
        self._pending_tool_availability.clear()
        if changes:
            self._apply_tool_availability_changes_requested.emit(changes)

    def _queue_tool_availability_change(self, kind: str, name: str, enabled: bool) -> None:
        key = (kind, name)
        self._pending_tool_availability[key] = {
            "kind": kind,
            "name": name,
            "enabled": bool(enabled),
        }
        if not self._tool_availability_timer.isActive():
            self._tool_availability_timer.start(0)

    def set_tool_enabled(self, name: str, enabled: bool) -> None:
        self._queue_tool_availability_change("tool", name, enabled)

    def set_mcp_server_enabled(self, name: str, enabled: bool) -> None:
        self._queue_tool_availability_change("server", name, enabled)

    def shutdown(self) -> None:
        # The worker can be blocked inside run_until_complete while a provider is
        # streaming. stop_run uses call_soon_threadsafe, allowing that slot to
        # return so the queued shutdown request can be processed normally.
        self.stop_run()
        if self._tool_availability_timer.isActive():
            self._tool_availability_timer.stop()
        self._pending_tool_availability.clear()
        if self._force_stop_timer.isActive():
            self._force_stop_timer.stop()
        self._stop_worker_thread(force=False)
