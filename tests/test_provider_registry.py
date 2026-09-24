import unittest
from pathlib import Path

from core.provider_registry import (
    PathConflictError,
    ProviderRegistry,
    RegistryValidationError,
    build_reasoning_kwargs,
    set_nested,
)


def _registry(*providers):
    return {"schema_version": 2, "data_version": 1, "providers": list(providers)}


def _provider(**overrides):
    payload = {
        "id": "openrouter",
        "hosts": ["openrouter.ai"],
        "match_type": "suffix",
        "rules": [
            {
                "param": "extra_body.reasoning.effort",
                "values": {
                    "minimal": "low",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": "high",
                },
            }
        ],
    }
    payload.update(overrides)
    return payload


class ProviderRegistryTests(unittest.TestCase):
    def test_match_uses_hostname_only_and_accepts_missing_scheme(self):
        registry = ProviderRegistry(
            _registry(
                _provider(
                    id="openai",
                    hosts=["api.openai.com"],
                    match_type="exact",
                    rules=[{"param": "reasoning.effort", "values": {"low": "low"}}],
                )
            )
        )

        self.assertEqual(registry.match("api.openai.com/v1?foo=bar")["id"], "openai")

    def test_suffix_match_requires_label_boundary(self):
        registry = ProviderRegistry(_registry(_provider()))

        self.assertEqual(registry.match("https://api.openrouter.ai/api/v1")["id"], "openrouter")
        self.assertIsNone(registry.match("https://api.evil-openrouter.ai/v1"))

    def test_disabled_provider_is_skipped_and_first_host_match_wins(self):
        registry = ProviderRegistry(
            _registry(
                _provider(id="disabled", enabled=False, hosts=["openrouter.ai"]),
                _provider(id="active", hosts=["openrouter.ai"]),
            )
        )

        self.assertEqual(registry.match("https://api.openrouter.ai/v1")["id"], "active")

    def test_match_uses_model_name_to_select_rule(self):
        registry = ProviderRegistry(
            _registry(
                _provider(
                    id="multi",
                    hosts=["openrouter.ai"],
                    match_type="suffix",
                    rules=[
                        {"models": {"contains": ["gpt-oss"]}, "param": "reasoning_effort", "values": {"high": "oss"}},
                        {"models": {"contains": ["qwen3"]}, "param": "reasoning_effort", "values": {"high": "qwen"}},
                    ],
                )
            )
        )

        self.assertEqual(registry.match("https://openrouter.ai/v1", "openai/gpt-oss-120b")["values"]["high"], "oss")
        self.assertEqual(registry.match("https://openrouter.ai/v1", "qwen/qwen3-235b")["values"]["high"], "qwen")
        self.assertIsNone(registry.match("https://openrouter.ai/v1", "meta/llama-3.3"))

    def test_unknown_host_returns_none_and_payload_is_unchanged(self):
        payload = {"model": "x"}

        self.assertIsNone(ProviderRegistry(_registry(_provider())).match("https://unknown.example/v1"))
        self.assertIs(build_reasoning_kwargs(payload, None, "high"), payload)
        self.assertEqual(payload, {"model": "x"})

    def test_provider_without_rules_resolves_to_none(self):
        config = ProviderRegistry(
            _registry(_provider(id="gateway", hosts=["gw.example"], match_type="exact", rules=[]))
        ).match("https://gw.example/v1", "any-model")

        self.assertIsNone(config)

    def test_rule_without_models_matches_any_model(self):
        config = ProviderRegistry(_registry(_provider())).match("https://openrouter.ai/api/v1", "any-new-model")

        self.assertIsNotNone(config)
        self.assertEqual(config["id"], "openrouter")

    def test_rule_models_limit_matching(self):
        registry = ProviderRegistry(
            _registry(
                _provider(
                    id="openai",
                    hosts=["api.openai.com"],
                    match_type="exact",
                    rules=[
                        {
                            "models": {"prefix": ["gpt-5", "o1", "o3", "o4"]},
                            "param": "reasoning.effort",
                            "values": {"medium": "medium"},
                        }
                    ],
                )
            )
        )

        self.assertIsNotNone(registry.match("https://api.openai.com/v1", "gpt-5-mini"))
        self.assertIsNone(registry.match("https://api.openai.com/v1", "gpt-4o"))

    def test_registry_validation_applies_match_type_and_mode_defaults(self):
        config = ProviderRegistry(
            _registry(
                {
                    "id": "defaults",
                    "hosts": ["api.openai.com"],
                    "rules": [{"param": "reasoning.effort", "values": {"low": "low"}}],
                }
            )
        ).match("https://api.openai.com/v1", "gpt-5")

        self.assertIsNotNone(config)
        self.assertEqual(config["mode"], "effort")

    def test_build_reasoning_kwargs_maps_value_and_sets_nested_path(self):
        config = ProviderRegistry(_registry(_provider())).match("https://openrouter.ai/api/v1", "x")
        payload = {"model": "x"}

        build_reasoning_kwargs(payload, config, "xhigh")

        self.assertEqual(payload, {"model": "x", "extra_body": {"reasoning": {"effort": "high"}}})

    def test_agentrouter_glm5_registry_uses_documented_reasoning_effort(self):
        registry = ProviderRegistry.from_path(Path(__file__).parents[1] / "provider_registry.json")

        config = registry.match("https://agentrouter.org/v1", "zai-org/glm-5.1")
        self.assertEqual(config["id"], "agentrouter_org")
        self.assertEqual(config["param"], "reasoning_effort")

        for effort, expected in (("low", "low"), ("high", "high"), ("max", "max"), ("medium", "high")):
            payload = {"model": "zai-org/glm-5.1"}
            build_reasoning_kwargs(payload, config, effort)
            self.assertEqual(payload["reasoning_effort"], expected)

    def test_nvidia_deepseek_v4_registry_uses_documented_reasoning_effort(self):
        registry = ProviderRegistry.from_path(Path(__file__).parents[1] / "provider_registry.json")

        config = registry.match(
            "https://integrate.api.nvidia.com/v1",
            "deepseek-ai/deepseek-v4-flash-0731",
        )
        self.assertEqual(config["id"], "nvidia_nim")
        self.assertEqual(config["param"], "reasoning_effort")

        for effort in ("none", "high", "max"):
            payload = {"model": "deepseek-ai/deepseek-v4-flash-0731"}
            build_reasoning_kwargs(payload, config, effort)
            self.assertEqual(payload["reasoning_effort"], effort)
        self.assertNotIn("extra_body", payload)

    def test_deepseek_registry_enables_thinking_via_extra(self):
        registry = ProviderRegistry.from_path(Path(__file__).parents[1] / "provider_registry.json")

        config = registry.match("https://api.deepseek.com/v1", "deepseek-v4-flash")
        payload = {"model": "deepseek-v4-flash"}
        build_reasoning_kwargs(payload, config, "medium")

        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["extra_body"]["thinking"]["type"], "enabled")

    def test_build_reasoning_kwargs_toggle_preserves_typed_values(self):
        config = ProviderRegistry(
            _registry(
                _provider(
                    rules=[
                        {
                            "mode": "toggle",
                            "param": "extra_body.chat_template_kwargs.enable_thinking",
                            "toggle": {"on": True, "off": False},
                        }
                    ]
                )
            )
        ).match("https://openrouter.ai/v1", "x")
        enabled_payload = {}
        disabled_payload = {}

        build_reasoning_kwargs(enabled_payload, config, "high")
        build_reasoning_kwargs(disabled_payload, config, "high", enabled=False)

        self.assertEqual(enabled_payload, {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}})
        self.assertEqual(disabled_payload, {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})

    def test_build_reasoning_kwargs_supports_extra_fields(self):
        config = ProviderRegistry(
            _registry(
                _provider(
                    rules=[
                        {
                            "param": "reasoning.effort",
                            "values": {"low": "low", "medium": "medium", "high": "high", "xhigh": "high"},
                            "extra": {"reasoning.summary": "auto"},
                        }
                    ]
                )
            )
        ).match("https://openrouter.ai/api/v1", "x")
        payload = {}

        build_reasoning_kwargs(payload, config, "medium")

        self.assertEqual(payload, {"reasoning": {"effort": "medium", "summary": "auto"}})

    def test_effort_not_in_values_skips_payload(self):
        config = ProviderRegistry(_registry(_provider())).match("https://openrouter.ai/v1", "x")
        payload = {"model": "x"}

        build_reasoning_kwargs(payload, config, "extreme")

        self.assertEqual(payload, {"model": "x"})

    def test_effort_none_without_key_skips_payload(self):
        config = ProviderRegistry(_registry(_provider())).match("https://openrouter.ai/v1", "x")
        payload = {"model": "x"}

        build_reasoning_kwargs(payload, config, "none")

        self.assertEqual(payload, {"model": "x"})

    def test_effort_disabled_skips_for_effort_mode(self):
        config = ProviderRegistry(_registry(_provider())).match("https://openrouter.ai/v1", "x")
        payload = {}

        build_reasoning_kwargs(payload, config, "high", enabled=False)

        self.assertEqual(payload, {})

    def test_set_nested_extends_existing_objects(self):
        payload = {"reasoning": {"tokens": 1000}}

        set_nested(payload, "reasoning.effort", "high")

        self.assertEqual(payload, {"reasoning": {"tokens": 1000, "effort": "high"}})

    def test_set_nested_raises_on_path_conflict(self):
        with self.assertRaises(PathConflictError):
            set_nested({"reasoning": "bad"}, "reasoning.effort", "high")

    def test_registry_validation_requires_versions_and_providers(self):
        with self.assertRaises(RegistryValidationError):
            ProviderRegistry({"providers": []})

    def test_registry_validation_rejects_duplicate_ids(self):
        with self.assertRaises(RegistryValidationError):
            ProviderRegistry(_registry(_provider(id="same"), _provider(id="same")))

    def test_registry_validation_rejects_invalid_provider_shapes(self):
        invalid_cases = [
            _provider(hosts=[]),
            _provider(match_type="contains"),
            _provider(rules="not-a-list"),
            _provider(rules=[{"param": ".bad", "values": {"low": "low"}}]),
            _provider(rules=[{"param": "reasoning_effort"}]),
            _provider(rules=[{"param": "reasoning_effort", "values": {}}]),
            _provider(rules=[{"mode": "bad", "param": "reasoning_effort", "values": {"low": "low"}}]),
            _provider(rules=[{"mode": "toggle", "param": "x", "toggle": {"on": True}}]),
            _provider(rules=[{"param": "x", "values": {"low": "low"}, "models": {}}]),
            _provider(rules=[{"param": "x", "values": {"low": "low"}, "models": {"regex": ["bad"]}}]),
            _provider(rules=[{"param": "x", "values": {"low": "low"}, "models": {"prefix": []}}]),
        ]

        for provider in invalid_cases:
            with self.subTest(provider=provider):
                with self.assertRaises(RegistryValidationError):
                    ProviderRegistry(_registry(provider))

    def test_shipped_registry_loads_and_resolves(self):
        registry = ProviderRegistry.from_path(Path(__file__).parents[1] / "provider_registry.json")

        self.assertIsNotNone(registry.match("https://api.openai.com/v1", "gpt-6-astra"))
        self.assertIsNone(registry.match("https://not-in-registry.example/v1", "gpt-6-astra"))


if __name__ == "__main__":
    unittest.main()
