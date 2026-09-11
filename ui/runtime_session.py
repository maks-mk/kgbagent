from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from core.constants import BASE_DIR
from core.model_profiles import ModelProfileStore, find_active_profile
from core.reasoning_controls import profile_reasoning_overrides
from core.multimodal import normalize_model_capabilities, resolve_model_capabilities
from core.run_logger import JsonlRunLogger
from core.session_store import SessionSnapshot, normalize_project_path
from ui.runtime_payloads import (
    APPROVAL_MODE_PROMPT,
    CHAT_TITLE_FALLBACK,
    append_project_label,
    build_ui_payload,
    close_runtime_resources,
    generate_chat_title,
)
from ui.streaming import StreamEvent


def _runtime_module():
    from ui import runtime as runtime_module

    return runtime_module


class RuntimeSessionCoordinator:
    def __init__(self, worker: Any) -> None:
        self.worker = worker

    def current_project_path(self) -> str:
        return normalize_project_path(_runtime_module().Path.cwd())

    @staticmethod
    def project_path_is_valid(project_path: str | Path | None) -> bool:
        if not project_path:
            return False
        path = Path(str(project_path))
        try:
            resolved = path.resolve()
        except Exception:
            return False
        return resolved.exists() and resolved.is_dir()

    def create_new_session_for_project(self, project_path: str, *, with_project_label: bool = False) -> SessionSnapshot:
        checkpoint_info = self.worker.tool_registry.checkpoint_info
        title = append_project_label(CHAT_TITLE_FALLBACK, project_path) if with_project_label else CHAT_TITLE_FALLBACK
        return self.worker.store.new_session(
            checkpoint_backend=checkpoint_info.get("resolved_backend", self.worker.config.checkpoint_backend),
            checkpoint_target=checkpoint_info.get("target", "unknown"),
            project_path=project_path,
            title=title,
            persisted=False,
        )

    def sync_tool_registry_workdir(self, target_project_path: str) -> None:
        sync_fn = getattr(self.worker.tool_registry, "sync_working_directory", None)
        if callable(sync_fn):
            sync_fn(target_project_path)

    def set_current_session_active(self, session: SessionSnapshot, *, touch: bool = False) -> None:
        self.worker.current_session = session
        self.worker.current_session.approval_mode = self.worker._normalize_approval_mode(
            self.worker.current_session.approval_mode
        )
        self.worker.store.save_active_session(self.worker.current_session, touch=touch, set_active=True)

    def ensure_current_session_persisted(self) -> bool:
        if not self.worker.current_session or self.worker.current_session.is_persisted:
            return False
        self.worker.current_session.is_persisted = True
        self.worker.store.save_active_session(self.worker.current_session, touch=False, set_active=True)
        return True

    def try_change_workdir(self, target_project_path: str) -> tuple[bool, str]:
        normalized_target = normalize_project_path(target_project_path)
        if normalize_project_path(_runtime_module().Path.cwd()) == normalized_target:
            return True, ""
        try:
            _runtime_module().os.chdir(normalized_target)
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return True, ""

    def fallback_to_current_project_session(
        self,
        *,
        event_type: str,
        reason: str,
        notice_message: str,
        target_session_id: str = "",
        target_project_path: str = "",
        error: str = "",
    ) -> SessionSnapshot:
        fallback_project = self.current_project_path()
        fallback_session = self.create_new_session_for_project(fallback_project, with_project_label=True)
        self.set_current_session_active(fallback_session, touch=False)
        self.sync_tool_registry_workdir(fallback_project)

        payload: dict[str, Any] = {
            "reason": reason,
            "fallback_project_path": fallback_project,
        }
        if target_session_id:
            payload["target_session_id"] = target_session_id
        if target_project_path:
            payload["target_project_path"] = target_project_path
        if error:
            payload["error"] = error
            payload["error_type"] = error.split(":", 1)[0].strip() or "RuntimeError"
        self.worker._log_ui_run_event(event_type, **payload)
        self.worker.event_emitted.emit(
            StreamEvent("summary_notice", {"message": notice_message, "kind": "session_fallback"})
        )
        return fallback_session

    def activate_session_with_workdir_or_fallback(
        self,
        target: SessionSnapshot,
        *,
        fallback_event_type: str,
        notice_message: str,
    ) -> SessionSnapshot:
        target_project_path = normalize_project_path(target.project_path)
        if not self.project_path_is_valid(target_project_path):
            return self.fallback_to_current_project_session(
                event_type=fallback_event_type,
                reason="invalid_project_path",
                notice_message=notice_message,
                target_session_id=target.session_id,
                target_project_path=target_project_path,
            )

        changed, error = self.try_change_workdir(target_project_path)
        if not changed:
            return self.fallback_to_current_project_session(
                event_type=fallback_event_type,
                reason="chdir_failed",
                notice_message=notice_message,
                target_session_id=target.session_id,
                target_project_path=target_project_path,
                error=error,
            )

        self.sync_tool_registry_workdir(target_project_path)
        self.set_current_session_active(target, touch=False)
        return self.worker.current_session

    def select_session_for_project(self, *, force_new_session: bool = False) -> SessionSnapshot:
        project_path = self.current_project_path()
        if force_new_session:
            return self.create_new_session_for_project(project_path, with_project_label=True)

        seen_ids: set[str] = set()
        candidates: list[SessionSnapshot] = []

        last_active = self.worker.store.get_last_active_session()
        if last_active is not None and last_active.session_id not in seen_ids:
            candidates.append(last_active)
            seen_ids.add(last_active.session_id)

        project_active = self.worker.store.get_active_session_for_project(project_path)
        if project_active is not None and project_active.session_id not in seen_ids:
            candidates.append(project_active)
            seen_ids.add(project_active.session_id)

        all_sessions = self.worker.store.list_sessions()
        if all_sessions:
            newest = self.worker.store.get_session(all_sessions[0].session_id)
            if newest is not None and newest.session_id not in seen_ids:
                candidates.append(newest)
                seen_ids.add(newest.session_id)

        for candidate in candidates:
            if self.project_path_is_valid(candidate.project_path):
                return candidate

        return self.create_new_session_for_project(project_path, with_project_label=True)

    def maybe_set_session_title(self, user_text: str) -> bool:
        if not self.worker.current_session:
            return False
        current_title = str(self.worker.current_session.title or "").strip()
        project_default_title = append_project_label(CHAT_TITLE_FALLBACK, self.worker.current_session.project_path)
        if current_title and current_title not in {CHAT_TITLE_FALLBACK, project_default_title}:
            return False
        title = generate_chat_title(user_text)
        if current_title == project_default_title:
            title = append_project_label(title, self.worker.current_session.project_path)
        if title == self.worker.current_session.title:
            return False
        self.worker.current_session.title = title
        self.worker.store.save_active_session(self.worker.current_session, touch=False, set_active=True)
        return True

    async def emit_session_payload(self, *, include_transcript: bool) -> dict[str, Any]:
        payload = await build_ui_payload(
            self.worker.config,
            self.worker.tool_registry,
            self.worker.store,
            self.worker.current_session,
            model_profiles=self.worker.model_profiles,
            model_capabilities=self.worker.model_capabilities,
            agent_app=self.worker.agent_app,
            include_transcript=include_transcript,
        )
        self.worker._sync_pending_user_choice_from_payload(payload)
        self.worker.session_changed.emit(payload)
        return payload

    async def repair_current_session_if_needed(self) -> list[str]:
        return await _runtime_module().repair_session_if_needed(
            self.worker.agent_app,
            self.worker.current_session.thread_id,
            notifier=lambda message: self.worker.event_emitted.emit(
                StreamEvent("summary_notice", {"message": message, "kind": "session_repair"})
            ),
            event_logger=lambda event_type, payload: self.worker._log_ui_run_event(event_type, **payload),
        )

    @staticmethod
    def profile_config_path() -> Path:
        return BASE_DIR / ".agent_state" / "config.json"

    @staticmethod
    def profile_bootstrap_env_from_config(config) -> dict[str, str]:
        provider = str(config.provider or "").strip().lower()
        openai_key = config.openai_api_key.get_secret_value() if config.openai_api_key else ""
        gemini_key = config.gemini_api_key.get_secret_value() if config.gemini_api_key else ""
        anthropic_key = config.anthropic_api_key.get_secret_value() if config.anthropic_api_key else ""
        if provider == "openai":
            model, api_key = config.openai_model, openai_key
            base_url = str(config.openai_base_url or "")
        elif provider == "anthropic":
            model, api_key = config.anthropic_model, anthropic_key
            base_url = str(config.anthropic_base_url or "")
        else:
            model, api_key = config.gemini_model, gemini_key
            base_url = ""

        return {
            "PROVIDER": provider,
            "MODEL": str(model or ""),
            "API_KEY": str(api_key or ""),
            "BASE_URL": base_url,
            "OPENAI_MODEL": str(config.openai_model or ""),
            "OPENAI_API_KEY": str(openai_key or ""),
            "OPENAI_BASE_URL": str(config.openai_base_url or ""),
            "GEMINI_MODEL": str(config.gemini_model or ""),
            "GEMINI_API_KEY": str(gemini_key or ""),
            "ANTHROPIC_MODEL": str(config.anthropic_model or ""),
            "ANTHROPIC_API_KEY": str(anthropic_key or ""),
            "ANTHROPIC_BASE_URL": str(config.anthropic_base_url or ""),
            "SHOW_MODEL_THOUGHTS": "false",
        }

    @staticmethod
    def config_overrides_for_profile(profile: dict[str, Any]) -> dict[str, Any]:
        provider = str(profile.get("provider") or "").strip().lower()
        model_name = str(profile.get("model") or "").strip()
        api_key = str(profile.get("api_key") or "").strip()
        base_url = str(profile.get("base_url") or "").strip()
        profile_id = str(profile.get("id") or "").strip()
        overrides: dict[str, Any] = {
            "provider": provider,
            "active_model_profile_id": profile_id or None,
            "show_model_thoughts": False,
        }
        if provider == "openai":
            overrides["openai_model"] = model_name
            overrides["openai_api_key"] = SecretStr(api_key) if api_key else None
            overrides["openai_base_url"] = base_url or None
        elif provider == "anthropic":
            overrides["anthropic_model"] = model_name
            overrides["anthropic_api_key"] = SecretStr(api_key) if api_key else None
            overrides["anthropic_base_url"] = base_url or None
        else:
            overrides["gemini_model"] = model_name
            overrides["gemini_api_key"] = SecretStr(api_key) if api_key else None
        overrides.update(profile_reasoning_overrides(profile))
        return overrides

    def resolve_base_config(self):
        base = getattr(self.worker, "base_config", None) or getattr(self.worker, "config", None)
        if base is None:
            base = _runtime_module().setup_runtime()
            self.worker.base_config = base
            if getattr(self.worker, "config", None) is None:
                self.worker.config = base
        return base

    def build_config_for_active_profile(self, model_profiles: dict[str, Any]):
        base = self.resolve_base_config()
        base_values = base.model_dump(mode="python")
        active_profile = find_active_profile(model_profiles)
        if active_profile is None:
            return base
        overrides = self.config_overrides_for_profile(active_profile)
        if getattr(self.worker, "profile_store", None) is not None:
            overrides["model_profile_config_path"] = self.worker.profile_store.path
        # Apply the UI-saved SESSION_SIZE override from config.json.
        session_size = model_profiles.get("session_size") if isinstance(model_profiles, dict) else None
        if session_size is not None:
            try:
                overrides["summary_threshold"] = int(float(session_size))
            except (TypeError, ValueError):
                pass
        base_values.update(overrides)
        return base.__class__(**base_values)

    @staticmethod
    def selected_profile_id(model_profiles: dict[str, Any]) -> str:
        active = find_active_profile(model_profiles)
        return str(active.get("id") or "").strip() if active else ""

    def set_effective_model_capabilities(
        self,
        runtime_capabilities: Any,
        *,
        model_profiles: dict[str, Any] | None = None,
    ) -> None:
        self.worker.runtime_model_capabilities = normalize_model_capabilities(runtime_capabilities)
        active_profile = find_active_profile(model_profiles or self.worker.model_profiles or {})
        self.worker.model_capabilities = resolve_model_capabilities(
            active_profile, self.worker.runtime_model_capabilities
        )

    async def rebuild_runtime_for_active_profile(self, model_profiles: dict[str, Any]) -> None:
        active_profile = find_active_profile(model_profiles)
        if active_profile is None:
            return

        new_config = self.build_config_for_active_profile(model_profiles)
        old_tool_registry = self.worker.tool_registry
        checkpoint_runtime = getattr(old_tool_registry, "checkpoint_runtime", None)
        if old_tool_registry is not None and checkpoint_runtime is not None:
            reconfigure = getattr(old_tool_registry, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(new_config)
            new_agent_app, new_tool_registry = _runtime_module().build_compiled_agent(
                new_config,
                old_tool_registry,
                checkpoint_runtime,
                run_logger=JsonlRunLogger(new_config.run_log_dir),
            )
        else:
            new_agent_app, new_tool_registry = await _runtime_module().build_agent_app(new_config)
            checkpoint_runtime = getattr(new_tool_registry, "checkpoint_runtime", None)

        self.worker._clear_cli_output_bridge()
        self.worker.config = new_config
        self.worker.agent_app = new_agent_app
        self.worker.tool_registry = new_tool_registry
        self.worker.checkpoint_runtime = checkpoint_runtime
        self.worker._set_effective_model_capabilities(getattr(new_tool_registry, "model_capabilities", None))
        self.worker._configure_cli_output_bridge()

        if self.worker.current_session is not None:
            checkpoint_info = getattr(self.worker.tool_registry, "checkpoint_info", {}) or {}
            self.worker.current_session.checkpoint_backend = checkpoint_info.get(
                "resolved_backend",
                self.worker.current_session.checkpoint_backend,
            )
            self.worker.current_session.checkpoint_target = checkpoint_info.get(
                "target",
                self.worker.current_session.checkpoint_target,
            )
            self.worker.store.save_active_session(self.worker.current_session, touch=False, set_active=True)
            self.sync_tool_registry_workdir(self.worker.current_session.project_path)

        if old_tool_registry is not None and old_tool_registry is not new_tool_registry:
            await _runtime_module().close_runtime_resources(old_tool_registry)

    async def apply_model_profiles(
        self,
        candidate_payload: dict[str, Any],
        *,
        success_notice_kind: str,
        success_notice_message: str,
        sync_runtime: bool = False,
        runtime_failure_kind: str = "model_switch_failed",
        runtime_failure_message_prefix: str = "Failed to apply the selected model",
    ) -> bool:
        if self.worker.profile_store is None:
            raise RuntimeError("Model profile store is not initialized.")

        normalized_target = self.worker.profile_store.save(candidate_payload)
        self.worker.model_profiles = normalized_target
        self.worker._set_effective_model_capabilities(
            self.worker.runtime_model_capabilities,
            model_profiles=normalized_target,
        )
        if sync_runtime and self.selected_profile_id(normalized_target):
            try:
                await self.worker._rebuild_runtime_for_active_profile(normalized_target)
            except Exception as exc:
                self.worker.event_emitted.emit(
                    StreamEvent(
                        "summary_notice",
                        {
                            "message": f"{runtime_failure_message_prefix}: {exc}",
                            "kind": runtime_failure_kind,
                        },
                    )
                )
                await self.worker._emit_session_payload(include_transcript=False)
                return False
            self.worker._runtime_profile_id = self.selected_profile_id(normalized_target)
        self.worker.event_emitted.emit(
            StreamEvent(
                "summary_notice",
                {
                    "message": success_notice_message,
                    "kind": success_notice_kind,
                },
            )
        )
        await self.worker._emit_session_payload(include_transcript=False)
        return True

    async def ensure_runtime_matches_selected_profile(self) -> bool:
        target_profile_id = self.selected_profile_id(self.worker.model_profiles)
        if not target_profile_id:
            return False
        if target_profile_id == self.worker._runtime_profile_id:
            return True

        try:
            await self.worker._rebuild_runtime_for_active_profile(self.worker.model_profiles)
        except Exception as exc:
            self.worker.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": f"Failed to apply the selected model: {exc}",
                        "kind": "model_switch_failed",
                        "level": "error",
                    },
                )
            )
            return False

        self.worker._runtime_profile_id = target_profile_id
        await self.worker._emit_session_payload(include_transcript=False)
        return True
