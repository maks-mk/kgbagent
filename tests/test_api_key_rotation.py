import asyncio
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.api_key_rotation import (
    ApiKeyRotationExhaustedError,
    RotatingChatModel,
)
from core.config import AgentConfig
from core.model_profiles import ModelProfileStore


class _FakeStatusResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeProviderError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, response_status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = _FakeStatusResponse(response_status_code) if response_status_code is not None else None


class _FakeAuthenticationError(_FakeProviderError):
    pass


class _FakeResourceExhaustedError(_FakeProviderError):
    def __init__(self, message: str):
        super().__init__(message)
        self.code = lambda: "RESOURCE_EXHAUSTED"


class _FakePermissionDeniedError(_FakeProviderError):
    def __init__(self, message: str):
        super().__init__(message)
        self.code = lambda: "PERMISSION_DENIED"


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = []
        self.invalid_tool_calls = []


class _FakeModel:
    def __init__(self, api_key: str, outcomes: dict[str, list[object]], calls: list[str]):
        self.api_key = api_key
        self._outcomes = outcomes
        self._calls = calls
        self.profile = {}

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _input, **_kwargs):
        self._calls.append(self.api_key)
        queue = self._outcomes.setdefault(self.api_key, [])
        if not queue:
            return _FakeResponse(f"ok:{self.api_key}")
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(str(outcome))


class ApiKeyRotationTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = Path.cwd() / ".tmp_tests" / f"api_key_rotation_{id(self)}"
        self._tmpdir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self._tmpdir, ignore_errors=True))

    def _config(self, profile_path: Path) -> AgentConfig:
        return AgentConfig(
            provider="openai",
            openai_model="gpt-4o",
            openai_api_key="sk-seed",
            active_model_profile_id="gpt-4o",
            model_profile_config_path=profile_path,
        )

    def _store(self, profile_path: Path) -> ModelProfileStore:
        store = ModelProfileStore(profile_path)
        store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_keys": ["sk-1", "sk-2", "sk-3"],
                        "api_key_index": 0,
                        "api_key": "sk-1",
                        "base_url": "",
                    }
                ],
            }
        )
        return store

    def test_rotating_model_retries_next_key_on_rate_limit(self):
        profile_path = self._tmpdir / "config.json"
        store = self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("429 Too Many Requests", status_code=429)],
            "sk-2": ["success-from-sk-2"],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        response = asyncio.run(model.ainvoke("hello"))

        self.assertEqual(response.content, "success-from-sk-2")
        self.assertEqual(calls, ["sk-1", "sk-2"])
        self.assertEqual(store.load()["profiles"][0]["api_key_index"], 1)

    def test_call_overrides_survive_key_rotation_without_affecting_next_call(self):
        profile_path = self._tmpdir / "config.json"
        self._store(profile_path)
        first = mock.Mock()
        first.ainvoke = mock.AsyncMock(side_effect=_FakeProviderError("429 Too Many Requests", status_code=429))
        second = mock.Mock()
        second.ainvoke = mock.AsyncMock(return_value=_FakeResponse("ok"))
        models = {"sk-seed": first, "sk-1": first, "sk-2": second}
        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=lambda config, *, api_key_override=None: models[api_key_override],
        )

        asyncio.run(model.ainvoke("summary", reasoning_effort="low"))
        first.ainvoke.assert_awaited_once_with("summary", reasoning_effort="low")
        second.ainvoke.assert_awaited_once_with("summary", reasoning_effort="low")
        asyncio.run(model.ainvoke("regular request"))
        second.ainvoke.assert_awaited_with("regular request")

    def test_rotating_model_rotates_on_auth_error_without_marking_invalid(self):
        profile_path = self._tmpdir / "config.json"
        store = self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("Unauthorized", status_code=401)],
            "sk-2": ["success-from-sk-2"],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        response = asyncio.run(model.ainvoke("hello"))

        self.assertEqual(response.content, "success-from-sk-2")
        self.assertEqual(calls, ["sk-1", "sk-2"])
        saved_profile = store.load()["profiles"][0]
        self.assertEqual(saved_profile["api_key_index"], 1)
        self.assertEqual(saved_profile["invalid_api_keys"], [])

    def test_rotating_model_rotates_on_billing_error(self):
        profile_path = self._tmpdir / "config.json"
        store = self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("Payment Required", status_code=402)],
            "sk-2": ["success-from-sk-2"],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        response = asyncio.run(model.ainvoke("hello"))

        self.assertEqual(response.content, "success-from-sk-2")
        self.assertEqual(calls, ["sk-1", "sk-2"])
        self.assertEqual(store.load()["profiles"][0]["api_key_index"], 1)

    def test_rotating_model_rotates_on_class_name_based_auth_error(self):
        profile_path = self._tmpdir / "config.json"
        store = self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeAuthenticationError("bad key")],
            "sk-2": ["success-from-sk-2"],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        response = asyncio.run(model.ainvoke("hello"))

        self.assertEqual(response.content, "success-from-sk-2")
        self.assertEqual(calls, ["sk-1", "sk-2"])
        self.assertEqual(store.load()["profiles"][0]["api_key_index"], 1)

    def test_rotating_model_rotates_on_provider_code_markers(self):
        profile_path = self._tmpdir / "config.json"
        store = self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeResourceExhaustedError("quota bucket exhausted")],
            "sk-2": ["success-from-sk-2"],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        response = asyncio.run(model.ainvoke("hello"))

        self.assertEqual(response.content, "success-from-sk-2")
        self.assertEqual(calls, ["sk-1", "sk-2"])
        self.assertEqual(store.load()["profiles"][0]["api_key_index"], 1)

    def test_rotating_model_rotates_on_response_status_code(self):
        profile_path = self._tmpdir / "config.json"
        store = self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("Forbidden", response_status_code=403)],
            "sk-2": ["success-from-sk-2"],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        response = asyncio.run(model.ainvoke("hello"))

        self.assertEqual(response.content, "success-from-sk-2")
        self.assertEqual(calls, ["sk-1", "sk-2"])
        self.assertEqual(store.load()["profiles"][0]["api_key_index"], 1)

    def test_rotating_model_stops_after_pool_is_exhausted(self):
        profile_path = self._tmpdir / "config.json"
        store = ModelProfileStore(profile_path)
        store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_keys": ["sk-1", "sk-2"],
                        "api_key_index": 0,
                        "api_key": "sk-1",
                        "base_url": "",
                    }
                ],
            }
        )
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("429 Too Many Requests", status_code=429)],
            "sk-2": [_FakeProviderError("Quota exceeded", status_code=429)],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        with self.assertRaises(ApiKeyRotationExhaustedError) as captured:
            asyncio.run(model.ainvoke("hello"))

        self.assertEqual(calls, ["sk-1", "sk-2"])
        self.assertEqual(captured.exception.error_kind, "rate_limit")
        self.assertEqual(captured.exception.keys_tried, 2)
        self.assertEqual(captured.exception.pool_size, 2)
        self.assertTrue(captured.exception.llm_retry_exhausted)

    def test_rotating_model_raises_exhausted_error_when_all_keys_fail_auth(self):
        profile_path = self._tmpdir / "config.json"
        store = ModelProfileStore(profile_path)
        store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_keys": ["sk-1", "sk-2"],
                        "api_key_index": 0,
                        "api_key": "sk-1",
                        "base_url": "",
                    }
                ],
            }
        )
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("Unauthorized", status_code=401)],
            "sk-2": [_FakeProviderError("Forbidden", status_code=403)],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        with self.assertRaises(ApiKeyRotationExhaustedError) as ctx:
            asyncio.run(model.ainvoke("hello"))

        message = str(ctx.exception)
        self.assertIn("All API keys", message)
        self.assertIn("Last error", message)
        self.assertIn("_FakeProviderError", message)
        self.assertIn("HTTP 403", message)
        self.assertIn("Forbidden", message)
        self.assertEqual(calls, ["sk-1", "sk-2"])

    def test_rotating_model_exhausts_pool_on_provider_permission_denied_errors(self):
        profile_path = self._tmpdir / "config.json"
        store = ModelProfileStore(profile_path)
        store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_keys": ["sk-1", "sk-2"],
                        "api_key_index": 0,
                        "api_key": "sk-1",
                        "base_url": "",
                    }
                ],
            }
        )
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakePermissionDeniedError("permission denied")],
            "sk-2": [_FakePermissionDeniedError("permission denied")],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        with self.assertRaises(ApiKeyRotationExhaustedError):
            asyncio.run(model.ainvoke("hello"))

        self.assertEqual(calls, ["sk-1", "sk-2"])

    def _rotation_model(self, profile_path: Path, outcomes: dict[str, list[object]], calls: list[str]):
        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        return RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

    def test_rotating_model_emits_rotation_event_only_before_next_attempt(self):
        profile_path = self._tmpdir / "config.json"
        self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("429 Too Many Requests", status_code=429)],
            "sk-2": ["success-from-sk-2"],
        }
        model = self._rotation_model(profile_path, outcomes, calls)

        with mock.patch("core.api_key_rotation.get_stream_writer") as writer_factory:
            asyncio.run(model.ainvoke("hello"))

        self.assertEqual(calls, ["sk-1", "sk-2"])
        writer = writer_factory.return_value
        self.assertEqual(writer.call_count, 1)
        event = writer.call_args.args[0]
        self.assertEqual(event["type"], "api_key_rotated")
        self.assertEqual(event["error_kind"], "rate_limit")
        self.assertEqual(event["from_index"], 0)
        self.assertEqual(event["to_index"], 1)

    def test_rotating_model_skips_rotation_event_for_single_key_pool(self):
        profile_path = self._tmpdir / "config.json"
        store = ModelProfileStore(profile_path)
        store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_keys": ["sk-1"],
                        "api_key_index": 0,
                        "api_key": "sk-1",
                        "base_url": "",
                    }
                ],
            }
        )
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("429 Too Many Requests", status_code=429)],
        }
        model = self._rotation_model(profile_path, outcomes, calls)

        with mock.patch("core.api_key_rotation.get_stream_writer") as writer:
            with self.assertRaises(ApiKeyRotationExhaustedError):
                asyncio.run(model.ainvoke("hello"))

        self.assertEqual(calls, ["sk-1"])
        writer.assert_not_called()

    def test_rotating_model_skips_rotation_event_on_last_key_of_pool(self):
        profile_path = self._tmpdir / "config.json"
        store = ModelProfileStore(profile_path)
        store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_keys": ["sk-1", "sk-2"],
                        "api_key_index": 0,
                        "api_key": "sk-1",
                        "base_url": "",
                    }
                ],
            }
        )
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("429 Too Many Requests", status_code=429)],
            "sk-2": [_FakeProviderError("Quota exceeded", status_code=429)],
        }
        model = self._rotation_model(profile_path, outcomes, calls)

        with mock.patch("core.api_key_rotation.get_stream_writer") as writer_factory:
            with self.assertRaises(ApiKeyRotationExhaustedError):
                asyncio.run(model.ainvoke("hello"))

        self.assertEqual(calls, ["sk-1", "sk-2"])
        # Only the sk-1 -> sk-2 switch announces a retry; the final sk-2
        # failure goes straight to run_failed via the exhausted error.
        writer = writer_factory.return_value
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(writer.call_args.args[0]["from_index"], 0)
        self.assertEqual(writer.call_args.args[0]["to_index"], 1)

    def test_rotating_model_survives_missing_stream_writer_context(self):
        profile_path = self._tmpdir / "config.json"
        self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("429 Too Many Requests", status_code=429)],
            "sk-2": ["success-from-sk-2"],
        }
        model = self._rotation_model(profile_path, outcomes, calls)

        # Direct unit invocations run outside any LangGraph streaming context.
        response = asyncio.run(model.ainvoke("hello"))

        self.assertEqual(response.content, "success-from-sk-2")
        self.assertEqual(calls, ["sk-1", "sk-2"])

    def test_bind_tools_descendants_share_model_cache_per_key_and_tools(self):
        profile_path = self._tmpdir / "config.json"
        self._store(profile_path)
        calls: list[str] = []
        outcomes: dict[str, list[object]] = {}
        created: list[_FakeModel] = []

        def factory(config, *, api_key_override=None):
            _ = config
            model = _FakeModel(str(api_key_override or ""), outcomes, calls)
            created.append(model)
            return model

        root = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )
        tool_a = SimpleNamespace(name="read_file")
        tool_b = SimpleNamespace(name="write_file")
        bound_a = root.bind_tools([tool_a])
        bound_ab = root.bind_tools([tool_a, tool_b])

        asyncio.run(bound_a.ainvoke("hello"))
        asyncio.run(bound_ab.ainvoke("hello"))
        asyncio.run(bound_a.ainvoke("hello"))

        # One provider model per (api_key, tools) pair, shared between the
        # root and its bind_tools() descendants.
        sk1_models = [m for m in created if m.api_key == "sk-1"]
        self.assertEqual(len(sk1_models), 2)
        self.assertIs(root._model_cache[bound_a._model_cache_key("sk-1")], sk1_models[0])
        self.assertIs(root._model_cache[bound_ab._model_cache_key("sk-1")], sk1_models[1])
        self.assertIs(bound_a._model_cache, root._model_cache)
        self.assertIs(bound_ab._model_cache, root._model_cache)

    def test_bind_tools_descendant_cache_key_includes_tool_names(self):
        profile_path = self._tmpdir / "config.json"
        self._store(profile_path)
        calls: list[str] = []
        outcomes: dict[str, list[object]] = {}

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        root = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )
        tool_a = SimpleNamespace(name="read_file")
        openai_tool = {"type": "function", "function": {"name": "search_web"}}

        bound_a = root.bind_tools([tool_a])
        bound_openai = root.bind_tools([openai_tool])

        self.assertNotEqual(
            bound_a._model_cache_key("sk-1"),
            bound_openai._model_cache_key("sk-1"),
        )
        self.assertIn("read_file", bound_a._model_cache_key("sk-1"))
        self.assertIn("search_web", bound_openai._model_cache_key("sk-1"))
        self.assertEqual(root._model_cache_key("sk-1"), "sk-1")


if __name__ == "__main__":
    unittest.main()
