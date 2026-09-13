"""Lossless context reductions and provider prompt-cache regression tests (no API calls)."""

import json
import os
import re
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import SecretStr

from core.config import AgentConfig
from core.context_builder import ContextBuilder
from core.message_utils import is_internal_retry_message
from core.nodes.llm import LLMMixin
from core.nodes.tools import ToolsMixin
from core.providers.anthropic import anthropic_prompt_cache_kwargs, create_anthropic_chat_model
from core.providers.factory import prepare_llm_with_tools


def make_config(**overrides):
    # Do not load local credentials, saved profiles or .env during these tests.
    defaults = {
        "provider": "openai",
        "anthropic_model": "claude-sonnet-4-5-20250929",
        "anthropic_base_url": "https://api.anthropic.com",
        "anthropic_api_key": SecretStr("test-key"),
        "anthropic_prompt_caching": "auto",
        "anthropic_reasoning": "off",
        "strict_mode": False,
        "max_retries": 1,
    }
    defaults.update(overrides)
    return AgentConfig.model_construct(**defaults)


def sample_tool():
    description = "Read structured records without modifying the source."
    return {
        "type": "function",
        "function": {
            "name": "records",
            "description": description,
            "parameters": {
                "type": "object",
                "description": description,
                "properties": {
                    "path": {"type": "string", "description": "Source path"},
                    "limit": {"type": "integer", "minimum": 1, "default": 10},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


class ContextTokenSavingsTests(unittest.TestCase):
    def setUp(self):
        self.builder = ContextBuilder(
            config=make_config(),
            prompt_loader=lambda: "Base instructions",
            is_internal_retry=is_internal_retry_message,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=re.compile(r"^[A-Za-z0-9]{9}$"),
        )

    def build(self, messages, task, **kwargs):
        return self.builder.build(
            messages, None, summary=kwargs.get("summary", ""), current_task=task,
            tools_available=True, active_tool_names=["read_file"],
            open_tool_issue=None, recovery_state=None,
        )

    def test_duplicate_task_removed_without_mutating_history_or_file_content(self):
        task = "Проверь файл и сохрани точные отступы. " * 30
        messages = [
            HumanMessage(content=task, id="user-1"),
            AIMessage(content="", tool_calls=[{
                "name": "read_file", "args": {"path": "demo.py"}, "id": "call-1",
            }]),
            ToolMessage(content="def f():\n    return '  exact  '\n", name="read_file", tool_call_id="call-1"),
        ]
        original = deepcopy(messages)
        context = self.build(messages, task, summary="Existing memory")
        system_text = "\n".join(str(m.content) for m in context if isinstance(m, SystemMessage))
        self.assertNotIn("Current task:", system_text)
        self.assertIn("Existing memory", system_text)
        self.assertIn("SAFETY POLICY:", system_text)
        self.assertEqual(context[-3:], self.builder.sanitize_messages(original))
        self.assertEqual(context[-1].content, original[-1].content)
        self.assertEqual(messages, original)
        self.assertIsNone(self.builder.detect_tool_history_mismatch(context))

    def test_continuation_and_missing_history_keep_task_hint(self):
        task = "Исправь ротацию ключей"
        for messages in ([], [HumanMessage(content="продолжай")], [
            HumanMessage(content=task), AIMessage(content="Работаю"), HumanMessage(content="продолжай"),
        ]):
            with self.subTest(messages=messages):
                context = self.build(messages, task)
                self.assertTrue(any(f"Current task: {task}" in str(m.content) for m in context))

    def test_internal_retry_does_not_change_duplicate_detection(self):
        task = "Проверь файл"
        messages = [HumanMessage(content=task), HumanMessage(
            content="Retry", additional_kwargs={"agent_internal": {"kind": "retry_instruction"}},
        )]
        context = self.build(messages, task)
        self.assertFalse(any("Current task:" in str(m.content) for m in context))
        self.assertEqual(messages[0].content, task)

    def test_multimodal_user_message_keeps_image_and_text(self):
        self.builder.config.provider = "anthropic"
        message = HumanMessage(content=[
            {"type": "text", "text": "Describe image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        ])
        original = deepcopy(message)
        context = self.build([message], "Describe image")
        self.assertFalse(any("Current task:" in str(m.content) for m in context))
        self.assertEqual(message, original)
        self.assertEqual(len(context[-1].content), 2)

    def test_system_prefix_stays_identical_across_normal_user_turns(self):
        first = self.build([HumanMessage(content="First task")], "First task")
        second = self.build([
            HumanMessage(content="First task"), AIMessage(content="Done"),
            HumanMessage(content="Second task"),
        ], "Second task")
        self.assertEqual(
            [m.content for m in first if isinstance(m, SystemMessage)],
            [m.content for m in second if isinstance(m, SystemMessage)],
        )


class ToolSchemaTokenSavingsTests(unittest.TestCase):
    def test_only_exact_duplicate_root_description_is_removed(self):
        tool = sample_tool()
        original = deepcopy(tool)
        llm = mock.Mock()
        prepare_llm_with_tools(llm, [tool])
        bound = llm.bind_tools.call_args.args[0][0]
        expected = deepcopy(original)
        expected["function"]["parameters"].pop("description")
        self.assertEqual(bound, expected)
        self.assertEqual(tool, original)

    def test_distinct_descriptions_and_provider_native_tools_are_preserved(self):
        tool = sample_tool()
        tool["function"]["parameters"]["description"] = "Path must stay inside the workspace."
        native = {"type": "web_search_preview"}
        llm = mock.Mock()
        prepare_llm_with_tools(llm, [tool, native])
        self.assertEqual(llm.bind_tools.call_args.args[0], [tool, native])

    def test_active_tool_subsets_use_same_normalization(self):
        owner = LLMMixin()
        owner.llm = mock.Mock()
        owner.llm_with_tools = mock.Mock()
        owner._all_tool_names = ["records", "other"]
        tool = sample_tool()
        result = owner._select_llm_for_active_tools([tool], ["records"])
        self.assertIs(result, owner.llm.bind_tools.return_value)
        parameters = owner.llm.bind_tools.call_args.args[0][0]["function"]["parameters"]
        self.assertNotIn("description", parameters)
        self.assertEqual(parameters["required"], ["path"])


class StructuredToolOutputTests(unittest.IsolatedAsyncioTestCase):
    async def execute(self, result):
        owner = ToolsMixin()
        owner.tools_map = {"read_file": SimpleNamespace(ainvoke=mock.AsyncMock(return_value=result))}
        owner._log_run_event = mock.Mock()
        return await owner._execute_tool("read_file", {})

    async def test_json_output_is_lossless_and_compact(self):
        payload = {"rows": [{"name": "Русский текст", "text": "  exact\n\tcode  ", "n": 1.25}],
                   "ok": True, "empty": None}
        for value in (payload, [payload], {}, []):
            with self.subTest(value=value):
                result = await self.execute(value)
                self.assertEqual(json.loads(result), value)
                self.assertEqual(result, json.dumps(value, ensure_ascii=False, separators=(",", ":")))

    async def test_string_results_are_not_minified_or_reformatted(self):
        for text in ('{\n  "a": 1,\n  "b": 2\n}', "def f():\n    return '  exact  '\n"):
            self.assertEqual(await self.execute(text), text)

    async def test_non_json_object_still_uses_existing_fallback(self):
        payload = {"values": {1, 2}}
        self.assertEqual(await self.execute(payload), str(payload))


class AnthropicPromptCacheTests(unittest.IsolatedAsyncioTestCase):
    def test_auto_requires_direct_https_endpoint_and_claude(self):
        for url in ("https://api.anthropic.com", "https://api.anthropic.com/v1/", "https://api.anthropic.com:443"):
            self.assertEqual(anthropic_prompt_cache_kwargs(make_config(provider="anthropic", anthropic_base_url=url)),
                             {"cache_control": {"type": "ephemeral"}})
        for url in ("https://proxy.example", "http://api.anthropic.com", "https://api.anthropic.com.example",
                    "https://api.anthropic.com/proxy", "https://api.anthropic.com:8080", "https://[invalid"):
            self.assertEqual(anthropic_prompt_cache_kwargs(make_config(provider="anthropic", anthropic_base_url=url)), {})
        for overrides in ({"provider": "openai"}, {"provider": "gemini"},
                          {"provider": "anthropic", "anthropic_model": "deepseek-v3.2"},
                          {"provider": "anthropic", "anthropic_prompt_caching": "off"}):
            self.assertEqual(anthropic_prompt_cache_kwargs(make_config(**overrides)), {})

    def test_proxy_can_opt_in_and_sdk_environment_override_is_respected(self):
        config = make_config(provider="anthropic", anthropic_base_url=None)
        with mock.patch.dict(os.environ, {"ANTHROPIC_BASE_URL": "https://proxy.example"}):
            self.assertEqual(anthropic_prompt_cache_kwargs(config), {})
            config.anthropic_prompt_caching = "on"
            self.assertEqual(anthropic_prompt_cache_kwargs(config), {"cache_control": {"type": "ephemeral"}})
        config.anthropic_prompt_caching = "auto"
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(anthropic_prompt_cache_kwargs(config), {"cache_control": {"type": "ephemeral"}})

    async def test_agent_invocation_passes_cache_control_only_when_enabled(self):
        for provider in ("anthropic", "openai", "gemini"):
            owner = LLMMixin()
            owner.config = make_config(provider=provider)
            owner._log_run_event = mock.Mock()
            owner._normalize_system_prefix_for_provider = lambda messages: messages
            llm = SimpleNamespace(ainvoke=mock.AsyncMock(return_value=AIMessage(content="Done")))
            messages = [HumanMessage(content="Task")]
            await owner._invoke_llm_with_retry(llm, messages)
            self.assertEqual(llm.ainvoke.call_args.args[0], messages)
            self.assertEqual(llm.ainvoke.call_args.kwargs, anthropic_prompt_cache_kwargs(owner.config))

    async def test_retry_keeps_cache_parameter_and_original_context(self):
        owner = LLMMixin()
        owner.config = make_config(provider="anthropic")
        owner._log_run_event = mock.Mock()
        owner._emit_provider_retry_status = mock.Mock()
        owner._normalize_system_prefix_for_provider = lambda messages: messages
        llm = SimpleNamespace(ainvoke=mock.AsyncMock(side_effect=[
            TimeoutError("temporary timeout"), AIMessage(content="Done"),
        ]))
        messages = [SystemMessage(content="Instructions"), HumanMessage(content="Task")]
        original = deepcopy(messages)
        with mock.patch("core.nodes.llm.asyncio.sleep", new_callable=mock.AsyncMock):
            response = await owner._invoke_llm_with_retry(llm, messages)
        self.assertEqual(response.content, "Done")
        self.assertEqual(llm.ainvoke.await_count, 2)
        for call in llm.ainvoke.call_args_list:
            self.assertEqual(call.args[0], original)
            self.assertEqual(call.kwargs, {"cache_control": {"type": "ephemeral"}})
        self.assertEqual(messages, original)

    async def test_installed_sdk_payload_preserves_tool_exchange_and_thinking(self):
        config = make_config(provider="anthropic")
        with mock.patch("core.providers.anthropic.load_provider_headers", return_value={}):
            model = create_anthropic_chat_model(config)
        try:
            messages = [SystemMessage(content="Instructions"), HumanMessage(content="Task"), AIMessage(
                content=[{"type": "thinking", "thinking": "Inspect", "signature": "test-signature"}],
                tool_calls=[{"name": "records", "args": {"path": "data"}, "id": "call-1"}],
            ), ToolMessage(content="Exact tool data", name="records", tool_call_id="call-1")]
            original = deepcopy(messages)
            plain = model._get_request_payload(messages)
            cached = model._get_request_payload(messages, **anthropic_prompt_cache_kwargs(config))
            self.assertEqual(cached.pop("cache_control"), {"type": "ephemeral"})
            self.assertEqual(cached, plain)
            self.assertNotIn("cache_control", model.model_kwargs)  # one-shot summaries stay uncached
            self.assertEqual(messages, original)
        finally:
            model._client.close()
            await model._async_client.close()
