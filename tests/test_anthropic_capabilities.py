import unittest

from core import anthropic_capabilities as capabilities


class AnthropicCapabilitiesTests(unittest.TestCase):
    def test_provider_prefix_preserves_capabilities(self):
        checks = (
            capabilities.is_claude_model,
            capabilities.anthropic_model_uses_manual_thinking,
            capabilities.anthropic_model_uses_adaptive_thinking,
            capabilities.anthropic_model_requires_thinking,
            capabilities.anthropic_model_reasoning_efforts,
            capabilities.anthropic_model_disallows_sampling,
        )
        models = (
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-6",
            "claude-opus-4-5",
            "claude-opus-4-7",
            "claude-4.8-opus",
            "claude-opus-5.5",
            "claude-fable-5",
            "claude-mythos-preview",
            "claude-unknown",
        )
        for model in models:
            for prefix in ("abc/", "router/anthropic/"):
                for check in checks:
                    with self.subTest(model=model, prefix=prefix, check=check.__name__):
                        self.assertEqual(check(f" {prefix}{model.upper()} "), check(model))

    def test_non_claude_names_are_not_recognized(self):
        for model in (None, "", "abc/", "abc/gpt-4o", "claude-opus-4-7/gpt-4o", "abc/not-claude-opus-4-7"):
            with self.subTest(model=model):
                self.assertFalse(capabilities.is_claude_model(model))
                self.assertFalse(capabilities.anthropic_model_uses_adaptive_thinking(model))
                self.assertEqual(capabilities.anthropic_model_reasoning_efforts(model), ())
