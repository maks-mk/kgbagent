import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

from core.config import AgentConfig
from core.session_store import SessionSnapshot, SessionStore
from ui.runtime import AgentRunWorker


class RuntimeSessionCoordinationTests(unittest.IsolatedAsyncioTestCase):
    def _session_snapshot(self, project_path: Path) -> SessionSnapshot:
        return SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(project_path),
        )

    async def test_current_project_path_uses_runtime_facade_path(self):
        worker = AgentRunWorker()
        project_path = Path("D:/tmp/project-a")

        with mock.patch("ui.runtime.Path.cwd", return_value=project_path):
            self.assertEqual(worker._current_project_path(), str(project_path))

    async def test_try_change_workdir_uses_runtime_facade_os(self):
        worker = AgentRunWorker()
        project_path = Path("D:/tmp/project-b")

        with mock.patch("ui.runtime.Path.cwd", return_value=Path("D:/tmp/other")), mock.patch(
            "ui.runtime.os.chdir"
        ) as chdir_mock:
            changed, error = worker._try_change_workdir(str(project_path))

        self.assertTrue(changed)
        self.assertEqual(error, "")
        chdir_mock.assert_called_once_with(str(project_path))

    async def test_emit_session_payload_keeps_public_shape(self):
        tmp_dir = Path.cwd() / ".tmp_tests" / uuid4().hex
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp_dir, ignore_errors=True))

        worker = AgentRunWorker()
        session_path = tmp_dir / "session.json"
        worker.store = SessionStore(session_path)
        project_path = tmp_dir
        worker.current_session = worker.store.new_session(
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            project_path=str(project_path),
        )
        worker.config = mock.Mock(
            provider="openai",
            openai_model="gpt-4o",
            checkpoint_backend="sqlite",
            enable_approvals=True,
            debug=False,
        )
        worker.model_profiles = {"active_profile": None, "profiles": []}
        worker.model_capabilities = {"image_input_supported": False}
        worker.tool_registry = mock.Mock(
            tools=[],
            builtin_tools=[],
            tool_metadata={},
            mcp_config={},
            mcp_server_status=[],
            disabled_local_tools=set(),
            checkpoint_info={"resolved_backend": "sqlite"},
            active_tools=mock.Mock(return_value=[]),
            get_runtime_status_lines=mock.Mock(return_value=[]),
        )
        worker.agent_app = None

        payload = await worker._emit_session_payload(include_transcript=False)

        self.assertIn("snapshot", payload)
        self.assertIn("sessions", payload)
        self.assertEqual(payload["active_session_id"], worker.current_session.session_id)

    async def test_emit_session_payload_restores_pending_user_choice_flags(self):
        worker = AgentRunWorker()
        worker.config = mock.Mock()
        worker.tool_registry = mock.Mock()
        worker.store = mock.Mock()
        worker.current_session = self._session_snapshot(Path.cwd())
        worker.model_profiles = {"active_profile": None, "profiles": []}
        worker.model_capabilities = {"image_input_supported": False}
        worker.agent_app = mock.Mock()
        payload = {
            "snapshot": {},
            "sessions": [],
            "active_session_id": "session-1",
            "pending_user_choice": {"choice_type": "clarification"},
        }

        with mock.patch("ui.runtime_session.build_ui_payload", new=mock.AsyncMock(return_value=payload)):
            result = await worker._emit_session_payload(include_transcript=False)

        self.assertIs(result, payload)
        self.assertTrue(worker._awaiting_approval)
        self.assertEqual(worker._awaiting_interrupt_kind, "user_choice")
        self.assertEqual(worker._pending_user_choice_type, "clarification")

    async def test_build_config_for_active_profile_bootstraps_base_config_when_worker_config_missing(self):
        worker = AgentRunWorker()
        worker.base_config = None
        worker.config = None
        worker.profile_store = None
        bootstrapped = AgentConfig(PROVIDER="openai", OPENAI_API_KEY="test-key", OPENAI_MODEL="gpt-4o")
        model_profiles = {
            "active_profile": "claude-via-openai-compatible",
            "profiles": [
                {
                    "id": "claude-via-openai-compatible",
                    "provider": "openai",
                    "model": "claude-3-7-sonnet",
                    "api_key": "sk-claude-proxy",
                    "base_url": "http://127.0.0.1:3002/v1",
                    "show_model_thoughts": True,
                }
            ],
        }

        with mock.patch("ui.runtime.setup_runtime", return_value=bootstrapped) as setup_mock:
            result = worker._build_config_for_active_profile(model_profiles)

        setup_mock.assert_called_once()
        self.assertIs(worker.base_config, bootstrapped)
        self.assertIs(worker.config, bootstrapped)
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.openai_model, "claude-3-7-sonnet")
        self.assertEqual(result.openai_base_url, "http://127.0.0.1:3002/v1")
        self.assertFalse(result.show_model_thoughts)

    async def test_build_config_for_active_profile_applies_profile_reasoning(self):
        worker = AgentRunWorker()
        worker.base_config = AgentConfig(PROVIDER="openai", OPENAI_API_KEY="test-key", OPENAI_MODEL="gpt-4o")
        worker.config = worker.base_config
        worker.profile_store = None
        model_profiles = {
            "active_profile": "gpt-5",
            "profiles": [
                {
                    "id": "gpt-5",
                    "provider": "openai",
                    "model": "gpt-5.6",
                    "api_key": "sk-profile",
                    "base_url": "https://api.openai.com/v1",
                    "reasoning": {"enabled": True, "effort": "high"},
                }
            ],
        }

        result = worker._build_config_for_active_profile(model_profiles)

        self.assertTrue(result.enable_model_reasoning)
        self.assertEqual(result.model_reasoning_effort, "high")

    async def test_build_config_for_active_profile_applies_llm_api_mode_override(self):
        worker = AgentRunWorker()
        worker.base_config = AgentConfig(
            PROVIDER="openai",
            OPENAI_API_KEY="test-key",
            OPENAI_MODEL="gpt-4o",
            LLM_API_MODE="chat",
        )
        worker.config = worker.base_config
        worker.profile_store = None
        model_profiles = {
            "active_profile": "gpt-5",
            "llm_api_mode": "responses",
            "profiles": [
                {
                    "id": "gpt-5",
                    "provider": "openai",
                    "model": "gpt-5.6",
                    "api_key": "sk-profile",
                    "base_url": "https://api.openai.com/v1",
                }
            ],
        }

        result = worker._build_config_for_active_profile(model_profiles)

        self.assertEqual(result.llm_api_mode, "responses")
