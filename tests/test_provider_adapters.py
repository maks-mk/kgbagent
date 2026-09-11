"""Provider adapter regression tests.

These tests verify that the provider-specific subclasses and monkey-patches
moved to ``core/providers/`` preserve their structural contract — i.e. that
the private-method overrides and module-level patches are still in place
after the refactor.  They do **not** make real API calls.

If a LangChain/OpenAI/Google SDK update renames or removes a private method,
these tests will fail early instead of silently breaking streaming or
thought-signature round-tripping at runtime.
"""

import asyncio
import unittest
import warnings
from unittest import mock

from pydantic import BaseModel, ConfigDict

from core.providers import (
    chat_model_accepts_kwarg,
    create_llm,
    extract_openai_reasoning_delta,
    gemini_model_supports_thinking_budget,
    gemini_model_supports_thinking_level,
    normalized_gemini_model_name,
    normalized_gemini_thinking_level,
    normalized_model_name,
    normalized_reasoning_effort,
    patch_langchain_google_genai_retry_kwargs,
    prepare_llm_with_tools,
)
from core.providers.base import (
    chat_model_accepts_kwarg as base_chat_model_accepts_kwarg,
)
from core.providers.gemini import (
    _build_gemini_thought_signature_adapter,
    _encode_gemini_thought_signature,
    _decode_gemini_thought_signature,
    create_gemini_chat_model,
)
from core.reasoning_debug import preview_value
from core.providers.openai_reasoning import (
    _build_reasoning_debug_chat_openai,
    _normalize_responses_text_delta,
    _safe_openai_model_dump,
    create_openai_chat_model,
)
from core.text_utils import extract_cache_hit_tokens


class ProviderPackageExportsTests(unittest.TestCase):
    """Verify that all public symbols are re-exported from ``core.providers``."""

    def test_public_api_callable(self):
        for fn in (
            create_llm,
            prepare_llm_with_tools,
            extract_openai_reasoning_delta,
            patch_langchain_google_genai_retry_kwargs,
            gemini_model_supports_thinking_budget,
            gemini_model_supports_thinking_level,
            normalized_model_name,
            normalized_gemini_model_name,
            normalized_reasoning_effort,
            normalized_gemini_thinking_level,
            chat_model_accepts_kwarg,
        ):
            self.assertTrue(callable(fn), f"{fn!r} is not callable")

    def test_base_helpers_are_same_object(self):
        """Re-exported helpers must be identical to the base-module originals."""
        self.assertIs(chat_model_accepts_kwarg, base_chat_model_accepts_kwarg)


class GeminiThoughtSignatureHelperTests(unittest.TestCase):
    """Unit tests for thought-signature encode/decode round-trip."""

    def test_encode_decode_bytes_roundtrip(self):
        original = b"\x00\x01\xff\xfe"
        encoded = _encode_gemini_thought_signature(original)
        decoded = _decode_gemini_thought_signature(encoded)
        self.assertEqual(decoded, original)

    def test_encode_str_passthrough(self):
        self.assertEqual(_encode_gemini_thought_signature("abc"), "abc")

    def test_decode_empty_returns_empty_bytes(self):
        self.assertEqual(_decode_gemini_thought_signature(""), b"")
        self.assertEqual(_decode_gemini_thought_signature(None), b"")

    def test_decode_invalid_base64_falls_back_to_utf8(self):
        # Non-base64 string should fall back to utf-8 encoding, not raise.
        result = _decode_gemini_thought_signature("not!base64?")
        self.assertEqual(result, "not!base64?".encode("utf-8"))

    def test_encode_memoryview(self):
        mv = memoryview(b"hello")
        encoded = _encode_gemini_thought_signature(mv)
        self.assertEqual(_decode_gemini_thought_signature(encoded), b"hello")

    def test_encode_bytearray(self):
        ba = bytearray(b"world")
        encoded = _encode_gemini_thought_signature(ba)
        self.assertEqual(_decode_gemini_thought_signature(encoded), b"world")


class GeminiModelDetectionTests(unittest.TestCase):
    def test_thinking_budget_for_2_5_models(self):
        self.assertTrue(gemini_model_supports_thinking_budget("gemini-2.5-flash"))
        self.assertTrue(gemini_model_supports_thinking_budget("gemini-2.5-pro"))

    def test_thinking_budget_for_latest_aliases(self):
        self.assertTrue(gemini_model_supports_thinking_budget("gemini-flash-latest"))
        self.assertTrue(gemini_model_supports_thinking_budget("gemini-pro-latest"))

    def test_no_thinking_budget_for_older_models(self):
        self.assertFalse(gemini_model_supports_thinking_budget("gemini-1.5-pro"))
        self.assertFalse(gemini_model_supports_thinking_budget("gemini-1.0-pro"))

    def test_thinking_level_for_gemini3(self):
        self.assertTrue(gemini_model_supports_thinking_level("gemini-3-pro"))
        self.assertFalse(gemini_model_supports_thinking_level("gemini-2.5-flash"))

    def test_normalized_gemini_model_name_strips_prefix(self):
        self.assertEqual(normalized_gemini_model_name("models/gemini-2.5-flash"), "gemini-2.5-flash")
        self.assertEqual(normalized_gemini_model_name("Gemini-2.5-Flash"), "gemini-2.5-flash")


class ReasoningEffortNormalizationTests(unittest.TestCase):
    def test_valid_values(self):
        self.assertEqual(normalized_reasoning_effort("high"), "high")
        self.assertEqual(normalized_reasoning_effort("none"), "none")
        self.assertEqual(normalized_reasoning_effort("xhigh"), "xhigh")

    def test_invalid_defaults_to_medium(self):
        self.assertEqual(normalized_reasoning_effort("ultra"), "medium")
        self.assertEqual(normalized_reasoning_effort(None), "medium")
        self.assertEqual(normalized_reasoning_effort(""), "medium")

    def test_normalized_reasoning_effort_preserves_registry_boolean_level(self):
        self.assertEqual(normalized_reasoning_effort("true"), "true")

    def test_gemini_thinking_level_mapping(self):
        self.assertEqual(normalized_gemini_thinking_level("none"), "minimal")
        self.assertEqual(normalized_gemini_thinking_level("minimal"), "minimal")
        self.assertEqual(normalized_gemini_thinking_level("xhigh"), "high")
        self.assertEqual(normalized_gemini_thinking_level("medium"), "medium")


class OpenAIReasoningDeltaTests(unittest.TestCase):
    """Tests for ``extract_openai_reasoning_delta`` — non-standard reasoning keys."""

    def test_standard_reasoning_content(self):
        chunk = {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}
        self.assertEqual(extract_openai_reasoning_delta(chunk), "thinking...")

    def test_non_standard_thinking_key(self):
        chunk = {"choices": [{"delta": {"thinking": "analyzing..."}}]}
        self.assertEqual(extract_openai_reasoning_delta(chunk), "analyzing...")

    def test_analysis_key(self):
        chunk = {"choices": [{"delta": {"analysis": "evaluating..."}}]}
        self.assertEqual(extract_openai_reasoning_delta(chunk), "evaluating...")

    def test_no_reasoning_returns_none(self):
        chunk = {"choices": [{"delta": {"content": "hello"}}]}
        self.assertIsNone(extract_openai_reasoning_delta(chunk))

    def test_empty_delta_returns_none(self):
        chunk = {"choices": [{"delta": {}}]}
        self.assertIsNone(extract_openai_reasoning_delta(chunk))

    def test_no_choices_returns_none(self):
        self.assertIsNone(extract_openai_reasoning_delta({}))

    def test_object_with_model_dump(self):
        class FakeChunk:
            def model_dump(self):
                return {"choices": [{"delta": {"reasoning": "from_obj"}}]}

        self.assertEqual(extract_openai_reasoning_delta(FakeChunk()), "from_obj")

    def test_non_dict_non_object_returns_none(self):
        self.assertIsNone(extract_openai_reasoning_delta(42))
        self.assertIsNone(extract_openai_reasoning_delta("string"))


class ChatModelAcceptsKwargTests(unittest.TestCase):
    def test_pydantic_model_with_field(self):
        class FakeModel:
            model_fields = {"temperature": ..., "model": ...}

        self.assertTrue(chat_model_accepts_kwarg(FakeModel, "temperature"))
        self.assertTrue(chat_model_accepts_kwarg(FakeModel, "model"))

    def test_pydantic_model_accepts_field_alias(self):
        class FakeField:
            alias = "thinking_level"

        class FakeModel:
            model_fields = {"reasoning_effort": FakeField()}

        self.assertTrue(chat_model_accepts_kwarg(FakeModel, "thinking_level"))

    def test_pydantic_model_without_field(self):
        class FakeModel:
            model_fields = {"temperature": ...}

        self.assertFalse(chat_model_accepts_kwarg(FakeModel, "top_p"))

    def test_non_pydantic_model_defaults_true(self):
        class PlainModel:
            pass

        self.assertTrue(chat_model_accepts_kwarg(PlainModel, "anything"))


class GeminiAdapterStructureTests(unittest.TestCase):
    """Verify that the Gemini thought-signature adapter overrides the right methods."""

    def test_adapter_returns_base_when_module_missing_helpers(self):
        """If chat_models_module lacks required functions, base class is returned."""
        class FakeBase:
            pass

        fake_module = mock.MagicMock()
        # Remove required callables
        del fake_module._chat_with_retry
        result = _build_gemini_thought_signature_adapter(FakeBase, fake_module)
        self.assertIs(result, FakeBase)

    def test_adapter_subclass_has_overrides(self):
        """The adapter subclass must define _prepare_request, _generate, _agenerate."""
        class FakeBase:
            def _prepare_request(self, messages, *args, **kwargs):
                return mock.MagicMock()

        fake_module = mock.MagicMock()
        # All required callables present
        fake_module._chat_with_retry = mock.MagicMock()
        fake_module._achat_with_retry = mock.MagicMock()
        fake_module._response_to_result = mock.MagicMock()

        adapter_cls = _build_gemini_thought_signature_adapter(FakeBase, fake_module)
        self.assertIsNot(adapter_cls, FakeBase)
        self.assertTrue(hasattr(adapter_cls, "_prepare_request"))
        self.assertTrue(hasattr(adapter_cls, "_generate"))
        self.assertTrue(hasattr(adapter_cls, "_agenerate"))


class OpenAIAdapterStructureTests(unittest.TestCase):
    """Verify that the ReasoningDebugChatOpenAI subclass overrides the right methods."""

    def test_subclass_has_stream_overrides(self):
        class FakeBase:
            pass

        cls = _build_reasoning_debug_chat_openai(FakeBase)
        self.assertTrue(hasattr(cls, "_get_request_payload"))
        self.assertTrue(hasattr(cls, "_generate"))
        self.assertTrue(hasattr(cls, "_stream"))
        self.assertTrue(hasattr(cls, "_astream"))
        self.assertTrue(hasattr(cls, "_log_raw_provider_chunk"))
        self.assertTrue(hasattr(cls, "_attach_raw_reasoning_delta"))
        self.assertTrue(hasattr(cls, "_attach_reasoning_content"))

    @staticmethod
    def _responses_text_event(delta):
        from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

        return ResponseTextDeltaEvent.model_construct(
            content_index=0,
            delta=delta,
            item_id="msg-1",
            output_index=0,
            sequence_number=0,
            type="response.output_text.delta",
        )

    def test_responses_block_delta_is_split_into_visible_and_reasoning_text(self):
        event = self._responses_text_event(
            [
                {"type": "thinking", "thinking": [{"type": "text", "text": "Consider "}]},
                {"type": "reasoning", "reasoning": "carefully."},
                {"type": "text", "text": "Visible answer"},
            ]
        )

        normalized, reasoning = _normalize_responses_text_delta(event)

        self.assertEqual(normalized.delta, "Visible answer")
        self.assertEqual(reasoning, "Consider carefully.")

    def test_responses_thinking_only_delta_becomes_empty_visible_text(self):
        event = self._responses_text_event(
            [{"type": "thinking", "thinking": [{"type": "text", "text": "Hidden"}]}]
        )

        normalized, reasoning = _normalize_responses_text_delta(event)

        self.assertEqual(normalized.delta, "")
        self.assertEqual(reasoning, "Hidden")

    def test_safe_openai_model_dump_suppresses_mismatched_delta_warning(self):
        event = self._responses_text_event(
            [{"type": "thinking", "thinking": [{"type": "text", "text": "Let"}]}]
        )

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            dumped = _safe_openai_model_dump(event)

        self.assertEqual(captured, [])
        self.assertIsInstance(dumped["delta"], list)

    def test_safe_openai_model_dump_excludes_structured_parsed_without_warning(self):
        class FakeParsedPayload(BaseModel):
            summary: str

        class FakeOpenAIMessage(BaseModel):
            model_config = ConfigDict(extra="allow")
            parsed: None = None

        class FakeOpenAIChoice(BaseModel):
            message: FakeOpenAIMessage

        class FakeOpenAICompletion(BaseModel):
            choices: list[FakeOpenAIChoice]

        draft = FakeParsedPayload(summary="Implement a fix")
        message = FakeOpenAIMessage.model_construct(parsed=draft)
        completion = FakeOpenAICompletion(choices=[FakeOpenAIChoice(message=message)])

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            dumped = _safe_openai_model_dump(completion)

        self.assertEqual(captured, [])
        self.assertNotIn("parsed", dumped["choices"][0]["message"])

    def test_reasoning_debug_preview_excludes_structured_parsed_without_warning(self):
        class FakeParsedPayload(BaseModel):
            summary: str

        class FakeOpenAIMessage(BaseModel):
            model_config = ConfigDict(extra="allow")
            parsed: None = None

        class FakeOpenAIChoice(BaseModel):
            message: FakeOpenAIMessage

        class FakeOpenAICompletion(BaseModel):
            choices: list[FakeOpenAIChoice]

        draft = FakeParsedPayload(summary="Implement a fix")
        message = FakeOpenAIMessage.model_construct(parsed=draft)
        completion = FakeOpenAICompletion(choices=[FakeOpenAIChoice(message=message)])

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            rendered = preview_value(completion)

        self.assertEqual(captured, [])
        self.assertNotIn("parsed", rendered)

    def test_reasoning_debug_preview_excludes_top_level_parsed_without_warning(self):
        class FakeParsedPayload(BaseModel):
            summary: str

        class FakeParsedCompletion(BaseModel):
            model_config = ConfigDict(extra="allow")
            parsed: None = None

        draft = FakeParsedPayload(summary="Implement a fix")
        completion = FakeParsedCompletion.model_construct(parsed=draft)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            rendered = preview_value(completion)

        self.assertEqual(captured, [])
        self.assertNotIn("parsed", rendered)


class RetryPatchIdempotencyTests(unittest.TestCase):
    """The retry-kwargs monkey-patch must be idempotent."""

    def test_patch_is_idempotent(self):
        with mock.patch("importlib.import_module") as mock_import:
            fake_module = mock.MagicMock()
            fake_module._chat_with_retry = mock.MagicMock()
            fake_module._achat_with_retry = mock.MagicMock()
            fake_module._agent_retry_patch_applied = False
            mock_import.return_value = fake_module

            patch_langchain_google_genai_retry_kwargs()
            first_chat = fake_module._chat_with_retry
            first_achat = fake_module._achat_with_retry

            # Second call should be a no-op
            patch_langchain_google_genai_retry_kwargs()
            self.assertIs(fake_module._chat_with_retry, first_chat)
            self.assertIs(fake_module._achat_with_retry, first_achat)
            self.assertTrue(fake_module._agent_retry_patch_applied)


class PrepareLLMWithToolsTests(unittest.TestCase):
    def test_no_tools_returns_false(self):
        llm = mock.MagicMock()
        result_llm, enabled, error = prepare_llm_with_tools(llm, [])
        self.assertIs(result_llm, llm)
        self.assertFalse(enabled)
        self.assertEqual(error, "")

    def test_no_bind_tools_method(self):
        llm = mock.MagicMock()
        del llm.bind_tools
        result_llm, enabled, error = prepare_llm_with_tools(llm, [mock.MagicMock()])
        self.assertIs(result_llm, llm)
        self.assertFalse(enabled)
        self.assertIn("bind_tools", error)

    def test_successful_binding(self):
        llm = mock.MagicMock()
        bound = mock.MagicMock()
        llm.bind_tools.return_value = bound
        result_llm, enabled, error = prepare_llm_with_tools(llm, [mock.MagicMock()])
        self.assertIs(result_llm, bound)
        self.assertTrue(enabled)
        self.assertEqual(error, "")

    def test_binding_normalizes_required_arrays_recursively(self):
        llm = mock.MagicMock()
        tool = {
            "type": "function",
            "function": {
                "name": "optional_tool",
                "description": "Tool with optional arguments",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "options": {
                            "type": "object",
                            "properties": {"limit": {"type": "integer"}},
                        }
                    },
                },
            },
        }

        prepare_llm_with_tools(llm, [tool])

        bound_tool = llm.bind_tools.call_args.args[0][0]
        parameters = bound_tool["function"]["parameters"]
        self.assertEqual(parameters["required"], [])
        self.assertEqual(parameters["properties"]["options"]["required"], [])
        self.assertNotIn("required", tool["function"]["parameters"])

    def test_binding_preserves_existing_required_fields(self):
        llm = mock.MagicMock()
        tool = {
            "type": "function",
            "function": {
                "name": "required_tool",
                "description": "Tool with a required argument",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }

        prepare_llm_with_tools(llm, [tool])

        bound_tool = llm.bind_tools.call_args.args[0][0]
        self.assertEqual(bound_tool["function"]["parameters"]["required"], ["path"])

    def test_binding_exception(self):
        llm = mock.MagicMock()
        llm.bind_tools.side_effect = ValueError("unsupported tool schema")
        result_llm, enabled, error = prepare_llm_with_tools(llm, [mock.MagicMock()])
        self.assertIs(result_llm, llm)
        self.assertFalse(enabled)
        self.assertIn("unsupported tool schema", error)


class FactoryUnknownProviderTests(unittest.TestCase):
    def test_unknown_provider_raises(self):
        from core.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        config.provider = "unknown_provider"
        with self.assertRaises(ValueError) as ctx:
            create_llm(config)
        self.assertIn("unknown_provider", str(ctx.exception))


class LlmApiModeTests(unittest.TestCase):
    """Tests for LLM_API_MODE config field and its effect on the OpenAI factory."""

    def _make_config(self, api_mode: str | None) -> "AgentConfig":
        import os
        from core.config import AgentConfig

        env = {
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "gpt-5",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "MODEL_REASONING_EFFORT": "medium",
        }
        if api_mode is not None:
            env["LLM_API_MODE"] = api_mode
        with mock.patch.dict(os.environ, env, clear=False):
            # Remove LLM_API_MODE if explicitly None to test default
            if api_mode is None:
                os.environ.pop("LLM_API_MODE", None)
            # _env_file=None keeps the test hermetic: without it the local
            # developer .env (e.g. LLM_API_MODE=responses) leaks in and
            # breaks the "default is chat" expectation.
            return AgentConfig(_env_file=None)

    def test_default_is_chat(self):
        cfg = self._make_config(None)
        self.assertEqual(cfg.llm_api_mode, "chat")

    def test_responses_mode(self):
        cfg = self._make_config("responses")
        self.assertEqual(cfg.llm_api_mode, "responses")

    def test_invalid_falls_back_to_chat(self):
        cfg = self._make_config("bogus")
        self.assertEqual(cfg.llm_api_mode, "chat")

    def test_case_insensitive(self):
        cfg = self._make_config("RESPONSES")
        self.assertEqual(cfg.llm_api_mode, "responses")

    def test_separate_models_do_not_share_http_transports(self):
        cfg = self._make_config("chat")
        first = create_openai_chat_model(cfg)
        second = create_openai_chat_model(cfg)
        first_sync_transport = first.root_client._client
        first_async_transport = first.root_async_client._client
        second_sync_transport = second.root_client._client
        second_async_transport = second.root_async_client._client

        self.assertIsNot(first_sync_transport, second_sync_transport)
        self.assertIsNot(first_async_transport, second_async_transport)

        first.root_client.close()
        asyncio.run(first.root_async_client.close())

        self.assertTrue(first_sync_transport.is_closed)
        self.assertTrue(first_async_transport.is_closed)
        self.assertFalse(second_sync_transport.is_closed)
        self.assertFalse(second_async_transport.is_closed)
        second.root_client.close()
        asyncio.run(second.root_async_client.close())

    def test_responses_mode_sets_use_responses_api(self):
        cfg = self._make_config("responses")
        model = create_openai_chat_model(cfg)
        self.assertTrue(model.use_responses_api)

    def test_chat_mode_does_not_set_use_responses_api(self):
        cfg = self._make_config("chat")
        model = create_openai_chat_model(cfg)
        self.assertFalse(model.use_responses_api)

    def test_chat_mode_flattens_reasoning_dict_to_effort_string(self):
        """In chat mode, the registry's reasoning.effort path must not create a
        top-level 'reasoning' dict that would auto-trigger the Responses API."""
        cfg = self._make_config("chat")
        model = create_openai_chat_model(cfg)
        # reasoning_effort should be set as a plain string
        self.assertEqual(model.reasoning_effort, "medium")
        # reasoning dict should NOT be set (it would auto-switch to Responses API)
        self.assertIsNone(model.reasoning)

    def test_responses_mode_keeps_reasoning_dict(self):
        """In responses mode, the registry's reasoning.effort path should produce
        a 'reasoning' dict with effort and summary fields."""
        cfg = self._make_config("responses")
        model = create_openai_chat_model(cfg)
        self.assertIsInstance(model.reasoning, dict)
        self.assertEqual(model.reasoning.get("effort"), "medium")
        self.assertEqual(model.reasoning.get("summary"), "auto")

    def test_chat_mode_use_responses_api_returns_false_with_reasoning(self):
        """Even with reasoning enabled, chat mode must not route to Responses API."""
        cfg = self._make_config("chat")
        model = create_openai_chat_model(cfg)
        payload = model._get_request_payload([{"role": "user", "content": "hi"}])
        self.assertFalse(model._use_responses_api(payload))

    def test_responses_mode_use_responses_api_returns_true(self):
        """In responses mode, _use_responses_api must return True."""
        cfg = self._make_config("responses")
        model = create_openai_chat_model(cfg)
        payload = model._get_request_payload([{"role": "user", "content": "hi"}])
        self.assertTrue(model._use_responses_api(payload))


class OpenAIStreamCacheHitTests(unittest.TestCase):
    """Streamed chunks must expose provider cache-hit tokens as ``cache_read``."""

    def setUp(self):
        from langchain_openai import ChatOpenAI as BaseChatOpenAI

        adapter_cls = _build_reasoning_debug_chat_openai(BaseChatOpenAI)
        self.model = adapter_cls(model="deepseek-chat", api_key="sk-test", base_url="https://api.deepseek.com/v1")

    def _convert_usage_chunk(self, usage: dict) -> dict:
        from langchain_core.messages import AIMessageChunk

        generation_chunk = self.model._convert_chunk_to_generation_chunk(
            {"id": "chatcmpl-1", "model": "deepseek-chat", "choices": [], "usage": usage},
            AIMessageChunk,
            None,
        )
        return dict(generation_chunk.message.usage_metadata or {})

    def test_deepseek_cache_hit_tokens_are_reported_as_cache_read(self):
        usage_metadata = self._convert_usage_chunk(
            {
                "prompt_tokens": 5000,
                "completion_tokens": 7,
                "total_tokens": 5007,
                "prompt_cache_hit_tokens": 4608,
                "prompt_cache_miss_tokens": 392,
            }
        )

        self.assertEqual(usage_metadata["input_token_details"], {"cache_read": 4608})
        self.assertEqual(extract_cache_hit_tokens(usage_metadata), 4608)
        self.assertEqual(usage_metadata["input_tokens"], 5000)

    def test_openai_cached_tokens_are_reported_once(self):
        usage_metadata = self._convert_usage_chunk(
            {
                "prompt_tokens": 9000,
                "completion_tokens": 10,
                "total_tokens": 9010,
                "prompt_tokens_details": {"cached_tokens": 8192},
            }
        )

        self.assertEqual(usage_metadata["input_token_details"], {"cache_read": 8192})
        self.assertEqual(extract_cache_hit_tokens(usage_metadata), 8192)

    def test_usage_without_cache_fields_reports_no_cache_read(self):
        usage_metadata = self._convert_usage_chunk(
            {"prompt_tokens": 900, "completion_tokens": 10, "total_tokens": 910}
        )

        self.assertEqual(usage_metadata["input_token_details"], {})
        self.assertIsNone(extract_cache_hit_tokens(usage_metadata))

    def test_content_chunk_without_usage_is_untouched(self):
        from langchain_core.messages import AIMessageChunk

        generation_chunk = self.model._convert_chunk_to_generation_chunk(
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}],
            },
            AIMessageChunk,
            None,
        )

        self.assertEqual(generation_chunk.message.content, "Hi")
        self.assertIsNone(generation_chunk.message.usage_metadata)


class AnthropicCompatibleStreamTests(unittest.TestCase):
    def test_message_delta_dict_metadata_is_compatible_with_chat_anthropic(self):
        from types import SimpleNamespace

        from core.providers.anthropic import _build_anthropic_compatible_stream_adapter

        class FakeBase:
            def _make_message_chunk_from_anthropic_event(self, event, *args, **kwargs):
                return (
                    event.usage.cache_creation.model_dump(),
                    event.context_management.model_dump(),
                    event.delta.container.model_dump(mode="json"),
                )

        event = SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(cache_creation={"ephemeral_5m_input_tokens": 0}),
            context_management={"edits": []},
            delta=SimpleNamespace(container={"id": "container-1"}),
        )
        adapter = _build_anthropic_compatible_stream_adapter(FakeBase)()

        cache_creation, context_management, container = adapter._make_message_chunk_from_anthropic_event(event)

        self.assertEqual(cache_creation, {"ephemeral_5m_input_tokens": 0})
        self.assertEqual(context_management, {"edits": []})
        self.assertEqual(container, {"id": "container-1"})

    def test_non_message_delta_event_is_not_changed(self):
        from types import SimpleNamespace

        from core.providers.anthropic import _normalize_anthropic_message_delta_event

        event = SimpleNamespace(type="content_block_delta", delta={"text": "hello"})
        self.assertIs(_normalize_anthropic_message_delta_event(event), event)


class AnthropicReasoningTests(unittest.TestCase):
    """Tests for Anthropic effort-based reasoning configuration (section 17.10)."""

    def test_non_claude_model_through_anthropic_endpoint_omits_claude_parameters(self):
        """Compatible aggregators may expose other models through Messages API."""
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(
            ANTHROPIC_MODEL="deepseek-v3.2",
            ANTHROPIC_BASE_URL="https://proxy.example/v1",
            ANTHROPIC_REASONING="high",
        )
        model = create_anthropic_chat_model(cfg)
        payload = model._get_request_payload([{"role": "user", "content": "hi"}])

        self.assertEqual(model.model, "deepseek-v3.2")
        self.assertIsNone(model.thinking)
        self.assertIsNone(model.reasoning_effort)
        self.assertEqual(payload["model"], "deepseek-v3.2")
        self.assertNotIn("thinking", payload)
        self.assertNotIn("output_config", payload)
        # With anthropic>=1 the SDK no longer accepts sampling params as named
        # arguments, so langchain-anthropic relocates `temperature` into
        # `extra_body`; the value still reaches the API.
        self.assertEqual(payload.get("temperature", payload.get("extra_body", {}).get("temperature")), cfg.temperature)

    def test_non_claude_model_does_not_reject_anthropic_reasoning_setting(self):
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(
            ANTHROPIC_MODEL="zai-org/glm-5",
            ANTHROPIC_BASE_URL="https://proxy.example",
            ANTHROPIC_REASONING="max",
        )

        model = create_anthropic_chat_model(cfg)

        self.assertIsNone(model.thinking)
        self.assertIsNone(model.reasoning_effort)

    def _make_config(self, **overrides) -> "AgentConfig":
        import os
        from core.config import AgentConfig

        env = {
            "PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "ANTHROPIC_MODEL": "claude-sonnet-4-5-20250929",
            "ANTHROPIC_MAX_TOKENS": "8192",
        }
        env.update(overrides)
        with mock.patch.dict(os.environ, env, clear=False):
            return AgentConfig()

    def test_reasoning_effort_uses_adaptive_thinking_and_output_config(self):
        """Supported effort labels are forwarded as output_config.effort."""
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(ANTHROPIC_MODEL="claude-opus-4-6", ANTHROPIC_REASONING="low")
        model = create_anthropic_chat_model(cfg)
        payload = model._get_request_payload([{"role": "user", "content": "hi"}])
        self.assertEqual(model.thinking, {"type": "adaptive"})
        self.assertEqual(payload["output_config"], {"effort": "low"})

    def test_reasoning_adaptive_sets_adaptive_thinking(self):
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(ANTHROPIC_MODEL="claude-sonnet-4-6", ANTHROPIC_REASONING="adaptive")
        model = create_anthropic_chat_model(cfg)
        self.assertEqual(model.thinking, {"type": "adaptive"})

    def test_reasoning_off_disables_thinking(self):
        """ANTHROPIC_REASONING=off → thinking.type=disabled."""
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(ANTHROPIC_REASONING="off")
        model = create_anthropic_chat_model(cfg)
        self.assertEqual(model.thinking, {"type": "disabled"})

    def test_reasoning_empty_uses_budget_mode(self):
        """ANTHROPIC_REASONING empty → budget-based thinking.type=enabled."""
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(ANTHROPIC_REASONING="")
        model = create_anthropic_chat_model(cfg)
        self.assertEqual(model.thinking.get("type"), "enabled")
        self.assertIn("budget_tokens", model.thinking)

    def test_adaptive_only_models_omit_sampling_and_manual_budget(self):
        """New Claude models use adaptive thinking without temperature or a budget."""
        from core.providers.anthropic import create_anthropic_chat_model

        for model_name in (
            "claude-sonnet-5",
            "claude-sonnet-5-20260101",
            "claude-opus-4-7",
            "claude-opus-4-8",
        ):
            with self.subTest(model_name=model_name):
                cfg = self._make_config(ANTHROPIC_MODEL=model_name, ANTHROPIC_REASONING="")
                model = create_anthropic_chat_model(cfg)
                payload = model._get_request_payload([{"role": "user", "content": "hi"}])
                self.assertEqual(payload["thinking"], {"type": "adaptive"})
                self.assertNotIn("budget_tokens", payload["thinking"])
                self.assertNotIn("temperature", payload)

    def test_adaptive_only_model_passes_documented_effort(self):
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(ANTHROPIC_MODEL="claude-sonnet-5", ANTHROPIC_REASONING="high")
        model = create_anthropic_chat_model(cfg)
        payload = model._get_request_payload([{"role": "user", "content": "hi"}])
        self.assertEqual(payload["thinking"], {"type": "adaptive"})
        self.assertEqual(payload["output_config"], {"effort": "high"})
        self.assertNotIn("temperature", payload)

    def test_claude_48_opus_alias_accepts_medium_effort(self):
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(ANTHROPIC_MODEL="claude-4.8-opus", ANTHROPIC_REASONING="medium")
        model = create_anthropic_chat_model(cfg)
        payload = model._get_request_payload([{"role": "user", "content": "hi"}])

        self.assertEqual(model.thinking, {"type": "adaptive"})
        self.assertEqual(payload["output_config"], {"effort": "medium"})
        self.assertNotIn("temperature", payload)

    def test_reasoning_off_disables_adaptive_only_models(self):
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(ANTHROPIC_MODEL="claude-opus-4-8", ANTHROPIC_REASONING="off")
        model = create_anthropic_chat_model(cfg)
        self.assertEqual(model.thinking, {"type": "disabled"})

    def test_always_on_models_reject_disabling_thinking(self):
        from core.providers.anthropic import create_anthropic_chat_model

        for model_name in ("claude-fable-5", "claude-mythos-5", "claude-mythos-preview"):
            with self.subTest(model_name=model_name):
                cfg = self._make_config(ANTHROPIC_MODEL=model_name, ANTHROPIC_REASONING="off")
                with self.assertRaisesRegex(ValueError, "does not support disabling thinking"):
                    create_anthropic_chat_model(cfg)

    def test_sonnet_4_5_omits_sampling_with_manual_thinking(self):
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(ANTHROPIC_REASONING="")
        model = create_anthropic_chat_model(cfg)
        payload = model._get_request_payload([{"role": "user", "content": "hi"}])
        self.assertEqual(payload["thinking"]["type"], "enabled")
        self.assertIn("budget_tokens", payload["thinking"])
        self.assertNotIn("temperature", payload)

    def test_opus_4_6_empty_reasoning_uses_adaptive_thinking(self):
        """Opus 4.6 must not receive deprecated budget-based thinking."""
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(
            ANTHROPIC_MODEL="claude-opus-4-6",
            ANTHROPIC_REASONING="",
            ANTHROPIC_THINKING_BUDGET="4096",
        )
        model = create_anthropic_chat_model(cfg)
        self.assertEqual(model.thinking, {"type": "adaptive"})

    def test_adaptive_mode_takes_priority_over_budget(self):
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(
            ANTHROPIC_MODEL="claude-opus-4-6",
            ANTHROPIC_REASONING="high",
            ANTHROPIC_THINKING_BUDGET="8192",
        )
        model = create_anthropic_chat_model(cfg)
        payload = model._get_request_payload([{"role": "user", "content": "hi"}])
        self.assertEqual(model.thinking, {"type": "adaptive"})
        self.assertEqual(payload["output_config"], {"effort": "high"})

    def test_opus_4_5_combines_manual_budget_and_effort(self):
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(
            ANTHROPIC_MODEL="claude-opus-4-5-20251101",
            ANTHROPIC_REASONING="high",
            ANTHROPIC_THINKING_BUDGET="4096",
        )
        model = create_anthropic_chat_model(cfg)
        payload = model._get_request_payload([{"role": "user", "content": "hi"}])
        self.assertEqual(payload["thinking"], {"type": "enabled", "budget_tokens": 4096})
        self.assertEqual(payload["output_config"], {"effort": "high"})
        self.assertNotIn("temperature", payload)

    def test_opus_4_5_rejects_max_effort(self):
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(ANTHROPIC_MODEL="claude-opus-4-5-20251101", ANTHROPIC_REASONING="max")
        with self.assertRaisesRegex(ValueError, "does not support reasoning effort"):
            create_anthropic_chat_model(cfg)

    def test_sonnet_4_5_rejects_effort(self):
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(ANTHROPIC_REASONING="high")
        with self.assertRaisesRegex(ValueError, "does not support reasoning effort"):
            create_anthropic_chat_model(cfg)

    def test_claude_4_6_rejects_xhigh(self):
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(ANTHROPIC_MODEL="claude-sonnet-4-6", ANTHROPIC_REASONING="xhigh")
        with self.assertRaisesRegex(ValueError, "does not support reasoning effort"):
            create_anthropic_chat_model(cfg)

    def test_reasoning_budget_clamped_to_max_tokens(self):
        """budget_tokens must be >= 1024 and < max_tokens."""
        from core.providers.anthropic import create_anthropic_chat_model

        cfg = self._make_config(ANTHROPIC_REASONING="", ANTHROPIC_THINKING_BUDGET="99999", ANTHROPIC_MAX_TOKENS="8192")
        model = create_anthropic_chat_model(cfg)
        budget = model.thinking.get("budget_tokens")
        self.assertGreaterEqual(budget, 1024)
        self.assertLess(budget, 8192)

    def test_reasoning_invalid_raises(self):
        """Invalid ANTHROPIC_REASONING value must raise ValueError."""
        from core.config import AgentConfig

        with self.assertRaises(Exception):
            AgentConfig(
                PROVIDER="anthropic",
                ANTHROPIC_API_KEY="sk-ant-test",
                ANTHROPIC_MODEL="claude-sonnet-4-5-20250929",
                ANTHROPIC_REASONING="invalid_level",
            )


class AnthropicHeadersTests(unittest.TestCase):
    """Tests for Anthropic header overrides from headers.json (like OpenAI)."""

    def test_anthropic_factory_uses_native_chat_anthropic_parameter_names(self):
        """Do not rely on OpenAI-style aliases accepted by a LangChain version."""
        import os
        from core.config import AgentConfig
        from core.providers.anthropic import create_anthropic_chat_model

        received_kwargs = {}

        class FakeChatAnthropic:
            def __init__(self, **kwargs):
                received_kwargs.update(kwargs)

        env = {
            "PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "ANTHROPIC_MODEL": "claude-sonnet-4-5-20250929",
            "ANTHROPIC_BASE_URL": "https://proxy.example/v1",
            "ANTHROPIC_REASONING": "",
        }

        with mock.patch.dict(os.environ, env, clear=False):
            cfg = AgentConfig()
        with mock.patch("langchain_anthropic.ChatAnthropic", FakeChatAnthropic):
            create_anthropic_chat_model(cfg)

        kwargs = received_kwargs
        self.assertEqual(kwargs["anthropic_api_key"], "sk-ant-test")
        self.assertEqual(kwargs["anthropic_api_url"], "https://proxy.example")
        self.assertNotIn("api_key", kwargs)
        self.assertNotIn("base_url", kwargs)

    def test_load_provider_headers_alias_matches_openai(self):
        from core.http_headers import load_openai_headers, load_provider_headers

        # load_openai_headers is a backward-compatible wrapper that delegates
        # to load_provider_headers — both must return identical results.
        import json
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "headers.json"
            path.write_text(json.dumps({"User-Agent": "X/1", "x-y": "z"}), encoding="utf-8")
            self.assertEqual(load_openai_headers(path), load_provider_headers(path))

    def test_anthropic_factory_passes_default_headers(self):
        """The Anthropic factory must pass default_headers from headers.json,
        mirroring the OpenAI provider behavior."""
        import os
        from core.providers.anthropic import create_anthropic_chat_model
        from core.config import AgentConfig

        env = {
            "PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "ANTHROPIC_MODEL": "claude-sonnet-4-5-20250929",
            "ANTHROPIC_MAX_TOKENS": "8192",
            "ANTHROPIC_REASONING": "",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = AgentConfig()
        with mock.patch(
            "core.providers.anthropic.load_provider_headers",
            return_value={"User-Agent": "CustomAgent/2.0", "x-custom": "yes"},
        ):
            model = create_anthropic_chat_model(cfg)
        self.assertEqual(model.default_headers.get("User-Agent"), "CustomAgent/2.0")
        self.assertEqual(model.default_headers.get("x-custom"), "yes")

    def test_anthropic_factory_strips_v1_from_base_url(self):
        """The Anthropic SDK appends /v1/messages itself, so a base_url ending
        with /v1 must be stripped to avoid .../v1/v1/messages."""
        import os
        from core.providers.anthropic import create_anthropic_chat_model
        from core.config import AgentConfig

        env = {
            "PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "ANTHROPIC_MODEL": "claude-sonnet-4-5-20250929",
            "ANTHROPIC_MAX_TOKENS": "8192",
            "ANTHROPIC_BASE_URL": "https://proxy.example/v1",
            "ANTHROPIC_REASONING": "",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = AgentConfig()
        model = create_anthropic_chat_model(cfg)
        self.assertEqual(str(model.anthropic_api_url), "https://proxy.example")

    def test_anthropic_factory_keeps_base_url_without_v1(self):
        """A base_url without /v1 must be passed through unchanged."""
        import os
        from core.providers.anthropic import create_anthropic_chat_model
        from core.config import AgentConfig

        env = {
            "PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "ANTHROPIC_MODEL": "claude-sonnet-4-5-20250929",
            "ANTHROPIC_MAX_TOKENS": "8192",
            "ANTHROPIC_BASE_URL": "https://proxy.example",
            "ANTHROPIC_REASONING": "",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = AgentConfig()
        model = create_anthropic_chat_model(cfg)
        self.assertEqual(str(model.anthropic_api_url), "https://proxy.example")


if __name__ == "__main__":
    unittest.main()
