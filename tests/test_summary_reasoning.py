"""Summary reasoning overrides are local to each call; no provider requests."""

import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from core.config import AgentConfig
from core.nodes import AgentNodes
from core.provider_registry import ProviderRegistry, RegistryValidationError
from core.providers.factory import create_llm, summary_reasoning_kwargs
from core.providers.openai_reasoning import openai_reasoning_kwargs


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "provider_registry.json"


def make_config(provider="openai", model="gpt-6-astra", **overrides):
    values = {
        "provider": provider,
        f"{provider}_model": model,
        f"{provider}_api_key": "test-key",
        "openai_base_url": "https://api.openai.com/v1",
        "provider_registry_path": REGISTRY_PATH,
        "enable_model_reasoning": True,
        "model_reasoning_effort": "high",
        "anthropic_reasoning": "",
        "llm_api_mode": "chat",
        "model_supports_tools": False,
    }
    values.update(overrides)
    return AgentConfig(_env_file=None, **values)


class SummaryReasoningOptionsTests(unittest.TestCase):
    def test_minimum_supported_levels_and_provider_specific_parameters(self):
        cases = [
            ("openai", "gpt-6-astra", {}, {"reasoning_effort": "low"}),
            ("openai", "gpt-5", {}, {"reasoning_effort": "none"}),
            ("openai", "gpt-5", {"llm_api_mode": "responses"},
             {"reasoning": {"effort": "none", "summary": "auto"}}),
            ("openai", "gpt-6-astra", {"llm_api_mode": "responses"},
             {"reasoning": {"effort": "low"}}),
            ("openai", "deepseek-v4-pro", {"openai_base_url": "https://api.deepseek.com/v1"},
             {"reasoning_effort": "low", "extra_body": {"thinking": {"type": "enabled"}}}),
            ("openai", "qwen3.8-max", {"openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
             {"reasoning_effort": "low", "extra_body": {"enable_thinking": True}}),
            ("openai", "glm-5.2", {"openai_base_url": "https://api.z.ai/api/paas/v4"},
             {"reasoning_effort": "none", "extra_body": {"thinking": {"type": "enabled"}}}),
            ("openai", "openai/gpt-5", {"openai_base_url": "https://openrouter.ai/api/v1"},
             {"extra_body": {"reasoning": {"effort": "none"}}}),
            ("gemini", "gemini-3-flash-preview", {}, {"thinking_level": "minimal"}),
            ("gemini", "models/gemini-3.1-pro-preview", {}, {"thinking_level": "low"}),
            ("gemini", "gemma-4-31b-it", {}, {"thinking_level": "minimal"}),
            ("anthropic", "claude-opus-4-5", {}, {"effort": "low"}),
            ("anthropic", "claude-sonnet-4-6", {}, {"effort": "low"}),
            ("anthropic", "claude-opus-4-8", {"anthropic_reasoning": "max"}, {"effort": "low"}),
            ("anthropic", "claude-fable-5", {}, {"effort": "low"}),
        ]
        for provider, model, overrides, expected in cases:
            with self.subTest(provider=provider, model=model, overrides=overrides):
                config = make_config(provider, model, **overrides)
                original = config.model_dump()
                self.assertEqual(summary_reasoning_kwargs(config), expected)
                self.assertEqual(config.model_dump(), original)

    def test_unsupported_models_toggles_and_budgets_are_unchanged(self):
        cases = [
            ("openai", "gpt-4o", {}),
            ("openai", "gpt-6-astra", {"openai_base_url": "https://unknown.invalid/v1"}),
            ("openai", "kimi-k2.5", {"openai_base_url": "https://api.moonshot.ai/v1"}),
            ("openai", "qwen3-32b", {"openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"}),
            ("gemini", "gemini-2.5-pro", {}),
            ("gemini", "gemini-2.0-flash", {}),
            ("anthropic", "claude-sonnet-4-5", {}),
            ("anthropic", "some-compatible-model", {}),
        ]
        for provider, model, overrides in cases:
            with self.subTest(provider=provider, model=model):
                self.assertEqual(summary_reasoning_kwargs(make_config(provider, model, **overrides)), {})

    def test_disabled_reasoning_is_not_enabled(self):
        for provider, model in (("openai", "gpt-6-astra"), ("gemini", "gemini-3-flash-preview"),
                                ("anthropic", "claude-opus-4-6")):
            with self.subTest(provider=provider):
                config = make_config(provider, model, enable_model_reasoning=False)
                self.assertEqual(summary_reasoning_kwargs(config), {})
        for mode in ("off", "none"):
            config = make_config("anthropic", "claude-opus-4-6", anthropic_reasoning=mode)
            self.assertEqual(summary_reasoning_kwargs(config), {})

    def test_registry_order_and_input_aliases_do_not_determine_minimum(self):
        # The lowest provider level is reached by an input alias named "medium".
        rule = {"param": "reasoning_effort", "values": {"low": "high", "medium": "low", "high": "max"}}
        registry = Mock()
        registry.match.return_value = rule
        with patch.object(ProviderRegistry, "from_path", return_value=registry):
            self.assertEqual(summary_reasoning_kwargs(make_config()), {"reasoning_effort": "low"})
            for values in ({"low": True, "high": True}, {"low": 1024, "high": 8192}, {"high": "unknown"}):
                rule["values"] = values
                self.assertEqual(summary_reasoning_kwargs(make_config()), {})

    def test_unreadable_or_invalid_registry_preserves_settings(self):
        with patch.object(ProviderRegistry, "from_path", side_effect=RegistryValidationError("invalid registry")):
            self.assertEqual(summary_reasoning_kwargs(make_config()), {})

    def test_reasoning_builder_keeps_existing_nested_parameters(self):
        kwargs = {"extra_body": {"keep": True}}
        rule = {"param": "extra_body.reasoning.effort", "values": {"low": "low"}}
        result = openai_reasoning_kwargs(kwargs, rule, "low")
        self.assertIs(result, kwargs)
        self.assertEqual(result, {"extra_body": {"keep": True, "reasoning": {"effort": "low"}}})


class SummaryReasoningPayloadTests(unittest.TestCase):
    def test_openai_payload_override_does_not_change_normal_requests_or_api_mode(self):
        for mode in ("chat", "responses"):
            for model_name, minimum in (("gpt-5", "none"), ("gpt-6-astra", "low")):
                with self.subTest(mode=mode, model=model_name):
                    config = make_config(model=model_name, llm_api_mode=mode)
                    model = create_llm(config)
                    try:
                        original = model._get_request_payload("regular request")
                        summary = model._get_request_payload("summary", **summary_reasoning_kwargs(config))
                        after = model._get_request_payload("regular request")
                        self.assertEqual(original, after)
                        self.assertEqual(model._use_responses_api(summary), mode == "responses")
                        if mode == "responses":
                            self.assertEqual(original["reasoning"]["effort"], "high")
                            self.assertEqual(summary["reasoning"]["effort"], minimum)
                            self.assertFalse(summary["store"])
                            self.assertIn("reasoning.encrypted_content", summary["include"])
                        else:
                            self.assertEqual(original["reasoning_effort"], "high")
                            self.assertEqual(summary["reasoning_effort"], minimum)
                            self.assertNotIn("reasoning", summary)
                    finally:
                        model.root_client.close()
                        asyncio.run(model.root_async_client.close())

    def test_openai_nested_reasoning_override_is_local(self):
        config = make_config(model="openai/gpt-5", openai_base_url="https://openrouter.ai/api/v1")
        model = create_llm(config)
        try:
            summary = model._get_request_payload("summary", **summary_reasoning_kwargs(config))
            self.assertEqual(summary["extra_body"]["reasoning"]["effort"], "none")
            regular = model._get_request_payload("regular")
            self.assertEqual(regular["extra_body"]["reasoning"]["effort"], "high")
        finally:
            model.root_client.close()
            asyncio.run(model.root_async_client.close())

    def test_anthropic_override_preserves_thinking_and_normal_effort(self):
        for model_name in ("claude-opus-4-5", "claude-opus-4-6"):
            with self.subTest(model=model_name):
                config = make_config("anthropic", model_name, anthropic_reasoning="high")
                model = create_llm(config)
                messages = [HumanMessage(content="summary")]
                original = model._get_request_payload(messages)
                summary = model._get_request_payload(messages, **summary_reasoning_kwargs(config))
                self.assertEqual(summary["output_config"]["effort"], "low")
                self.assertEqual(summary["thinking"], original["thinking"])
                self.assertNotIn("effort", summary)
                self.assertEqual(original["output_config"]["effort"], "high")
                self.assertEqual(model._get_request_payload(messages), original)

    def test_gemini_override_preserves_normal_thinking_level(self):
        config = make_config("gemini", "gemini-3-flash-preview")
        model = create_llm(config)
        original = model._build_thinking_config()
        summary = model._build_thinking_config(**summary_reasoning_kwargs(config))
        self.assertEqual(summary.thinking_level.value.lower(), "minimal")
        self.assertEqual(original.thinking_level.value.lower(), "high")
        self.assertEqual(summary.include_thoughts, original.include_thoughts)
        self.assertEqual(model._build_thinking_config(), original)


class SummaryReasoningInvocationTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_and_fold_use_overrides_without_changing_shared_model(self):
        for model_name, expected in (("gpt-6-astra", {"reasoning_effort": "low"}), ("gpt-4o", {})):
            with self.subTest(model=model_name):
                config = make_config(
                    model=model_name, summary_threshold=1, summary_keep_last=1,
                    summary_max_tokens=120, summary_reserved_tokens=2000,
                )
                original_config = config.model_dump()
                memory = "- Keep the build failure and user restrictions."
                llm = Mock()
                llm.ainvoke = AsyncMock(side_effect=[
                    AIMessage(content=memory + "\n- Extra detail\n" * 300),
                    AIMessage(content=memory),
                    AIMessage(content="Continue the main task."),
                ])
                nodes = AgentNodes(config=config, llm=llm, llm_with_tools=llm, tools=[])
                state = {
                    "messages": [HumanMessage(id="h1", content="old request"),
                                 AIMessage(id="a1", content="old result"),
                                 HumanMessage(id="h2", content="continue")],
                    "current_task": "Continue the investigation", "steps": 0,
                }
                result = await nodes.summarize_node(state)
                self.assertEqual(result["summary"], memory)
                self.assertEqual(llm.ainvoke.await_count, 2)
                self.assertEqual([call.kwargs for call in llm.ainvoke.await_args_list], [expected, expected])
                self.assertIs(nodes.llm, llm)
                self.assertIs(nodes.llm_with_tools, llm)
                self.assertEqual(config.model_dump(), original_config)
                await nodes.agent_node({**state, "summary": memory})
                self.assertEqual(llm.ainvoke.await_count, 3)
                self.assertNotIn("reasoning_effort", llm.ainvoke.await_args.kwargs)


if __name__ == "__main__":
    unittest.main()
