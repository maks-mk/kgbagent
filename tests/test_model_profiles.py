import shutil
import unittest
from pathlib import Path

from core.model_profiles import (
    ModelProfileStore,
    bootstrap_profiles_from_env,
    find_active_profile,
    generate_profile_id,
    merge_profiles_with_env,
    normalize_profiles_payload,
)
from core.reasoning_controls import profile_reasoning_overrides, reasoning_options_for_profile


class ModelProfilesTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = Path.cwd() / ".tmp_tests" / f"model_profiles_{id(self)}"
        self._tmpdir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self._tmpdir, ignore_errors=True))

    def test_generate_profile_id_uses_api_domain_consonants(self):
        self.assertEqual(generate_profile_id("gpt-4o", set(), "https://api.openai.com/v1"), "opn/gpt-4o")
        self.assertEqual(generate_profile_id("model_name", set(), "openai.com"), "opn/model_name")

    def test_generate_profile_id_uses_domain_without_api_or_www_labels(self):
        self.assertEqual(
            generate_profile_id("model_name", set(), "https://api.anthropic.com/v1"),
            "anthrpc/model_name",
        )
        self.assertEqual(
            generate_profile_id("model_name", set(), "https://www.example-provider.io/v1"),
            "exmplprvdr/model_name",
        )

    def test_normalization_preserves_profile_id_namespace(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "opn/model_name",
                "profiles": [
                    {
                        "id": "opn/model_name",
                        "provider": "openai",
                        "model": "model_name",
                        "api_key": "",
                        "base_url": "https://api.openai.com/v1",
                    }
                ],
            }
        )
        self.assertEqual(payload["profiles"][0]["id"], "opn/model_name")
        self.assertEqual(payload["active_profile"], "opn/model_name")

    def test_bootstrap_uses_generic_env_first(self):
        payload = bootstrap_profiles_from_env(
            {
                "PROVIDER": "openai",
                "MODEL": "meta-llama/llama-3-70b",
                "API_KEY": "generic-key",
                "BASE_URL": "http://localhost:8000/v1",
                "OPENAI_MODEL": "gpt-4o",
            }
        )
        self.assertEqual(payload["active_profile"], "llama-3-70b")
        self.assertEqual(len(payload["profiles"]), 1)
        self.assertEqual(payload["profiles"][0]["provider"], "openai")
        self.assertEqual(payload["profiles"][0]["model"], "meta-llama/llama-3-70b")
        self.assertEqual(payload["profiles"][0]["api_key"], "generic-key")
        self.assertEqual(payload["profiles"][0]["base_url"], "http://localhost:8000/v1")

    def test_bootstrap_legacy_openai(self):
        payload = bootstrap_profiles_from_env(
            {
                "PROVIDER": "openai",
                "OPENAI_MODEL": "gpt-4o",
                "OPENAI_API_KEY": "sk-123",
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
            }
        )
        self.assertEqual(payload["active_profile"], "gpt-4o")
        self.assertEqual(payload["profiles"][0]["provider"], "openai")
        self.assertEqual(payload["profiles"][0]["model"], "gpt-4o")

    def test_normalization_skips_invalid_and_uniquifies_ids(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "custom",
                "profiles": [
                    {"id": "custom", "provider": "openai", "model": "gpt-4o", "api_key": "", "base_url": ""},
                    {"id": "custom", "provider": "openai", "model": "gpt-4o-mini", "api_key": "", "base_url": ""},
                    {"id": "", "provider": "gemini", "model": "gemini-1.5-flash", "api_key": "", "base_url": ""},
                    {"id": "x", "provider": "invalid", "model": "x", "api_key": "", "base_url": ""},
                    {"id": "z", "provider": "openai", "model": "", "api_key": "", "base_url": ""},
                ],
            }
        )
        self.assertEqual(len(payload["profiles"]), 3)
        ids = [item["id"] for item in payload["profiles"]]
        self.assertEqual(ids[0], "custom")
        self.assertEqual(ids[1], "custom-1")
        self.assertEqual(ids[2], "gemini-1-5-flash")
        self.assertEqual(payload["active_profile"], "custom")

    def test_normalization_deduplicates_identical_profiles(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "https://api.openai.com/v1",
                    },
                    {
                        "id": "gpt-4o-copy",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "https://api.openai.com/v1",
                    },
                ],
            }
        )
        self.assertEqual(len(payload["profiles"]), 1)
        self.assertEqual(payload["profiles"][0]["id"], "gpt-4o")

    def test_normalization_preserves_manual_image_support_flag(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gemini-1-5-flash",
                "profiles": [
                    {
                        "id": "gemini-1-5-flash",
                        "provider": "gemini",
                        "model": "gemini-1.5-flash",
                        "api_key": "gm-demo",
                        "base_url": "",
                        "supports_image_input": True,
                    }
                ],
            }
        )

        self.assertTrue(payload["profiles"][0]["supports_image_input"])
        active = find_active_profile(payload)
        self.assertIsNotNone(active)
        self.assertTrue(active["supports_image_input"])
        self.assertTrue(active["enabled"])

    def test_normalization_ignores_legacy_show_model_thoughts_flag(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gemini-2-5-flash",
                "profiles": [
                    {
                        "id": "gemini-2-5-flash",
                        "provider": "gemini",
                        "model": "gemini-2.5-flash",
                        "api_key": "gm-demo",
                        "base_url": "",
                        "show_model_thoughts": False,
                    }
                ],
            }
        )

        self.assertNotIn("show_model_thoughts", payload["profiles"][0])
        active = find_active_profile(payload)
        self.assertIsNotNone(active)
        self.assertNotIn("show_model_thoughts", active)

    def test_normalization_preserves_reasoning_profile_setting(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gpt-5",
                "profiles": [
                    {
                        "id": "gpt-5",
                        "provider": "openai",
                        "model": "gpt-5.6",
                        "api_key": "sk-demo",
                        "base_url": "https://api.openai.com/v1",
                        "reasoning": {"enabled": True, "effort": "high"},
                    }
                ],
            }
        )

        self.assertEqual(payload["profiles"][0]["reasoning"], {"enabled": True, "effort": "high"})

    def test_reasoning_options_use_documented_gemini_and_anthropic_parameters(self):
        gemini_level_options = reasoning_options_for_profile({"provider": "gemini", "model": "gemini-3.6-flash"})
        gemini_budget_options = reasoning_options_for_profile({"provider": "gemini", "model": "gemini-2.5-flash"})
        anthropic_options = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-sonnet-5"})

        self.assertIn({"value": "high", "label": "High", "config": {"enabled": True, "effort": "high"}}, gemini_level_options)
        self.assertIn(
            {"value": "budget:4096", "label": "Thinking: 4,096", "config": {"enabled": True, "thinking_budget": 4096}},
            gemini_budget_options,
        )
        self.assertEqual(
            [option["value"] for option in anthropic_options],
            ["off", "low", "medium", "high", "max", "xhigh"],
        )

    def test_nvidia_gpt_oss_reasoning_options_use_documented_effort_levels(self):
        options = reasoning_options_for_profile(
            {
                "provider": "openai",
                "model": "openai/gpt-oss-120b",
                "base_url": "https://integrate.api.nvidia.com/v1",
            }
        )

        self.assertEqual([option["value"] for option in options], ["low", "medium", "high"])
        self.assertEqual(options[-1]["config"], {"enabled": True, "effort": "high"})

    def test_agentrouter_glm5_reasoning_options_use_documented_effort_levels(self):
        profile = {
            "provider": "openai",
            "model": "zai-org/glm-5.1",
            "base_url": "https://agentrouter.org/v1",
        }

        options = reasoning_options_for_profile(profile)

        self.assertEqual([option["value"] for option in options], ["low", "high", "max"])
        self.assertEqual(
            profile_reasoning_overrides({**profile, "reasoning": {"enabled": True, "effort": "medium"}}),
            {"enable_model_reasoning": True, "model_reasoning_effort": "medium"},
        )

    def test_nvidia_deepseek_v4_reasoning_options_use_documented_effort_levels(self):
        profile = {
            "provider": "openai",
            "model": "deepseek-ai/deepseek-v4-flash-0731",
            "base_url": "https://integrate.api.nvidia.com/v1",
        }

        options = reasoning_options_for_profile(profile)

        self.assertEqual([option["value"] for option in options], ["none", "high", "max"])
        self.assertEqual(
            options[0],
            {"value": "none", "label": "None", "config": {"enabled": True, "effort": "none"}},
        )
        self.assertEqual(
            profile_reasoning_overrides({**profile, "reasoning": {"enabled": True, "effort": "max"}}),
            {"enable_model_reasoning": True, "model_reasoning_effort": "max"},
        )

    def test_nvidia_thinking_reasoning_options_are_binary(self):
        profile = {
            "provider": "openai",
            "model": "qwen/qwen3-235b-a22b",
            "base_url": "https://integrate.api.nvidia.com/v1",
        }

        options = reasoning_options_for_profile(profile)

        self.assertEqual(
            options,
            [
                {"value": "off", "label": "Off", "config": {"enabled": False}},
                {"value": "true", "label": "On", "config": {"enabled": True}},
            ],
        )
        self.assertEqual(
            profile_reasoning_overrides({**profile, "reasoning": {"enabled": True}}),
            {"enable_model_reasoning": True},
        )
        self.assertEqual(
            profile_reasoning_overrides({**profile, "reasoning": {"enabled": False}}),
            {"enable_model_reasoning": False},
        )

    def test_reasoning_off_option_is_only_exposed_for_providers_that_support_disabling(self):
        openai_options = reasoning_options_for_profile(
            {"provider": "openai", "model": "gpt-5.6", "base_url": "https://api.openai.com/v1"}
        )
        compatible_options = reasoning_options_for_profile(
            {"provider": "openai", "model": "openai/gpt-oss-120b", "base_url": "https://openrouter.ai/api/v1"}
        )
        gemini_options = reasoning_options_for_profile({"provider": "gemini", "model": "gemini-2.5-flash"})
        anthropic_options = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-sonnet-5"})

        off_option = {"value": "off", "label": "Off", "config": {"enabled": False}}

        self.assertNotIn("off", [option["value"] for option in openai_options])
        self.assertNotIn("off", [option["value"] for option in compatible_options])
        self.assertEqual(gemini_options[0], off_option)
        self.assertEqual(anthropic_options[0], off_option)

    def test_anthropic_reasoning_options_follow_model_capabilities(self):
        sonnet_45 = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"})
        opus_45 = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-opus-4-5-20251101"})
        sonnet_46 = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-sonnet-4-6"})
        opus_47 = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-opus-4-7"})
        opus_48_alias = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-4.8-opus"})
        mythos_preview = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-mythos-preview"})
        always_on = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-fable-5"})
        unknown = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-unknown"})

        self.assertEqual([option["value"] for option in sonnet_45], ["off", "budget:1024", "budget:4096", "budget:8192"])
        self.assertEqual([option["value"] for option in opus_45], ["off", "low", "medium", "high"])
        self.assertEqual([option["value"] for option in sonnet_46], ["off", "low", "medium", "high", "max"])
        self.assertEqual([option["value"] for option in opus_47], ["off", "low", "medium", "high", "max", "xhigh"])
        self.assertEqual([option["value"] for option in opus_48_alias], ["off", "low", "medium", "high", "max", "xhigh"])
        self.assertEqual([option["value"] for option in mythos_preview], ["low", "medium", "high", "max"])
        self.assertEqual([option["value"] for option in always_on], ["low", "medium", "high", "max", "xhigh"])
        self.assertEqual(unknown, [])

    def test_anthropic_reasoning_options_accept_provider_prefix(self):
        prefixed_sonnet = reasoning_options_for_profile(
            {"provider": "anthropic", "model": "abc/claude-sonnet-4-6"}
        )
        prefixed_opus = reasoning_options_for_profile(
            {"provider": "anthropic", "model": "router/claude-opus-4-7"}
        )

        self.assertEqual(
            [option["value"] for option in prefixed_sonnet],
            ["off", "low", "medium", "high", "max"],
        )
        self.assertEqual(
            [option["value"] for option in prefixed_opus],
            ["off", "low", "medium", "high", "max", "xhigh"],
        )

    def test_anthropic_reasoning_options_accept_dot_and_dash_separators(self):
        # Anthropic aliases the same model with both "claude-opus-5.5" and
        # "claude-opus-5-5". Both spellings must resolve to the same family
        # without listing each alias in the capability tables.
        opus_5_dot = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-opus-5.5"})
        opus_5_dash = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-opus-5-5"})
        opus_48_dot = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-opus-4.8"})
        # "claude-fable-5" forces thinking on, so it intentionally has no "off".
        fable_5_dot = reasoning_options_for_profile({"provider": "anthropic", "model": "claude-fable-5.0"})

        expected_full = ["off", "low", "medium", "high", "max", "xhigh"]
        expected_always_on = ["low", "medium", "high", "max", "xhigh"]
        self.assertEqual([option["value"] for option in opus_5_dot], expected_full)
        self.assertEqual([option["value"] for option in opus_5_dash], expected_full)
        self.assertEqual([option["value"] for option in opus_48_dot], expected_full)
        self.assertEqual([option["value"] for option in fable_5_dot], expected_always_on)

    def test_profile_reasoning_overrides_use_provider_specific_fields(self):
        self.assertEqual(
            profile_reasoning_overrides({"provider": "openai", "reasoning": {"enabled": False}}),
            {"enable_model_reasoning": False},
        )
        self.assertEqual(
            profile_reasoning_overrides({"provider": "gemini", "reasoning": {"enabled": True, "thinking_budget": 4096}}),
            {"enable_model_reasoning": True, "gemini_thinking_budget": 4096},
        )
        self.assertEqual(
            profile_reasoning_overrides({"provider": "anthropic", "reasoning": {"enabled": True, "mode": "adaptive"}}),
            {"enable_model_reasoning": True, "anthropic_reasoning": "adaptive"},
        )
        self.assertEqual(
            profile_reasoning_overrides(
                {
                    "provider": "anthropic",
                    "reasoning": {"enabled": True, "effort": "high", "thinking_budget": 4096},
                }
            ),
            {
                "enable_model_reasoning": True,
                "anthropic_reasoning": "high",
                "anthropic_thinking_budget": 4096,
            },
        )


    def test_normalization_migrates_legacy_api_key_to_rotation_pool(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "",
                    }
                ],
            }
        )

        profile = payload["profiles"][0]
        self.assertEqual(profile["api_key"], "sk-demo")
        self.assertEqual(profile["api_keys"], ["sk-demo"])
        self.assertEqual(profile["api_key_index"], 0)
        self.assertEqual(profile["invalid_api_keys"], [])

    def test_normalization_trims_and_deduplicates_rotation_keys(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": " sk-third ",
                        "api_keys": [" sk-first ", "", "sk-second", "sk-first", "sk-third "],
                        "api_key_index": 2,
                        "base_url": "",
                    }
                ],
            }
        )

        profile = payload["profiles"][0]
        self.assertEqual(profile["api_keys"], ["sk-first", "sk-second", "sk-third"])
        self.assertEqual(profile["api_key"], "sk-third")
        self.assertEqual(profile["api_key_index"], 2)

    def test_normalization_resets_rotation_error_state_on_load(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-1",
                        "api_keys": ["sk-1", "sk-2"],
                        "api_key_index": 0,
                        "invalid_api_keys": ["sk-1", "missing"],
                        "key_error_timestamps": {"sk-1": "123.5", "sk-2": 456, "missing": 789, "bad": "oops"},
                        "base_url": "",
                    }
                ],
            }
        )

        profile = payload["profiles"][0]
        self.assertEqual(profile["invalid_api_keys"], [])
        self.assertEqual(profile["key_error_timestamps"], {})
        self.assertEqual(profile["api_key"], "sk-1")
        self.assertEqual(profile["api_key_index"], 0)

    def test_normalization_keeps_profiles_with_different_image_support_flags(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "",
                        "supports_image_input": False,
                    },
                    {
                        "id": "gpt-4o-img",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "",
                        "supports_image_input": True,
                    },
                ],
            }
        )

        self.assertEqual(len(payload["profiles"]), 2)

    def test_normalization_drops_disabled_profile_from_active_selection(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "",
                        "enabled": False,
                    },
                    {
                        "id": "gemini-2-5-pro",
                        "provider": "gemini",
                        "model": "gemini-2.5-pro",
                        "api_key": "gm-demo",
                        "base_url": "",
                        "enabled": True,
                    },
                ],
            }
        )

        self.assertEqual(payload["active_profile"], "gemini-2-5-pro")

    def test_normalization_allows_all_profiles_to_be_temporarily_disabled(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "",
                        "enabled": False,
                    }
                ],
            }
        )

        self.assertIsNone(payload["active_profile"])

    def test_store_load_existing_merges_new_env_model(self):
        store = ModelProfileStore(self._tmpdir / "config.json")
        first = store.load_or_initialize(
            {
                "PROVIDER": "openai",
                "MODEL": "gpt-4o",
                "API_KEY": "first",
            }
        )
        second = store.load_or_initialize(
            {
                "PROVIDER": "gemini",
                "MODEL": "gemini-1.5-flash",
                "API_KEY": "second",
            }
        )
        self.assertEqual(len(first["profiles"]), 1)
        self.assertEqual(len(second["profiles"]), 2)
        self.assertEqual(find_active_profile(second)["provider"], "openai")
        self.assertTrue(any(item.get("provider") == "gemini" for item in second["profiles"]))

    def test_store_existing_unconfigured_uses_env_fallback(self):
        store = ModelProfileStore(self._tmpdir / "config.json")
        store.save({"active_profile": None, "profiles": []})
        loaded = store.load_or_initialize(
            {
                "PROVIDER": "openai",
                "OPENAI_MODEL": "gpt-4o",
                "OPENAI_API_KEY": "sk-env",
            }
        )
        self.assertEqual(loaded["active_profile"], "gpt-4o")
        self.assertEqual(len(loaded["profiles"]), 1)

    def test_store_first_launch_bootstrap_from_env_does_not_duplicate_on_restart(self):
        store = ModelProfileStore(self._tmpdir / "config.json")
        env_payload = {
            "PROVIDER": "openai",
            "OPENAI_MODEL": "gpt-4o",
            "OPENAI_API_KEY": "sk-env",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        }
        first = store.load_or_initialize(env_payload)
        second = store.load_or_initialize(env_payload)

        self.assertEqual(first["active_profile"], "gpt-4o")
        self.assertEqual(len(first["profiles"]), 1)
        self.assertEqual(first, second)

    def test_merge_profiles_with_env_appends_missing_env_model_once(self):
        existing = {
            "active_profile": "gemini-1-5-flash",
            "profiles": [
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-existing",
                    "base_url": "",
                }
            ],
        }
        env_payload = {
            "active_profile": "gpt-oss-120b",
            "profiles": [
                {
                    "id": "gpt-oss-120b",
                    "provider": "openai",
                    "model": "openai/gpt-oss-120b",
                    "api_key": "sk-env",
                    "base_url": "https://api.openai.com/v1",
                }
            ],
        }
        merged_once = merge_profiles_with_env(existing, env_payload)
        merged_twice = merge_profiles_with_env(merged_once, env_payload)

        self.assertEqual(merged_once["active_profile"], "gemini-1-5-flash")
        self.assertEqual(len(merged_once["profiles"]), 2)
        self.assertEqual(len(merged_twice["profiles"]), 2)
        self.assertTrue(
            any(
                item.get("provider") == "openai" and item.get("model") == "openai/gpt-oss-120b"
                for item in merged_twice["profiles"]
            )
        )

    def test_store_load_existing_merges_new_env_model_without_duplication(self):
        store = ModelProfileStore(self._tmpdir / "config.json")
        store.save(
            {
                "active_profile": "gemini-1-5-flash",
                "profiles": [
                    {
                        "id": "gemini-1-5-flash",
                        "provider": "gemini",
                        "model": "gemini-1.5-flash",
                        "api_key": "gm-existing",
                        "base_url": "",
                    }
                ],
            }
        )
        env_payload = {
            "PROVIDER": "openai",
            "MODEL": "openai/gpt-oss-120b",
            "API_KEY": "sk-env",
            "BASE_URL": "https://api.openai.com/v1",
        }
        first = store.load_or_initialize(env_payload)
        second = store.load_or_initialize(env_payload)

        self.assertEqual(first["active_profile"], "gemini-1-5-flash")
        self.assertEqual(len(first["profiles"]), 2)
        self.assertEqual(len(second["profiles"]), 2)

    def test_store_save_normalizes_active_profile(self):
        store = ModelProfileStore(self._tmpdir / "config.json")
        saved = store.save(
            {
                "active_profile": "missing-id",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "",
                        "base_url": "",
                    }
                ],
            }
        )
        self.assertEqual(saved["active_profile"], "gpt-4o")

    def test_store_rotate_api_key_moves_once_for_same_failed_key(self):
        store = ModelProfileStore(self._tmpdir / "config.json")
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

        first_rotation = store.rotate_api_key("gpt-4o", "sk-1")
        second_rotation = store.rotate_api_key("gpt-4o", "sk-1")

        self.assertEqual(first_rotation["current_key"], "sk-2")
        self.assertEqual(second_rotation["current_key"], "sk-2")
        self.assertEqual(store.load()["profiles"][0]["api_key_index"], 1)

    def test_store_rotate_api_key_marks_invalid_key_and_skips_it(self):
        store = ModelProfileStore(self._tmpdir / "config.json")
        store.save(
            {
                "active_profile": "gemini-1-5-flash",
                "profiles": [
                    {
                        "id": "gemini-1-5-flash",
                        "provider": "gemini",
                        "model": "gemini-1.5-flash",
                        "api_keys": ["gm-1", "gm-2"],
                        "api_key_index": 0,
                        "api_key": "gm-1",
                        "base_url": "",
                    }
                ],
            }
        )

        rotated = store.rotate_api_key("gemini-1-5-flash", "gm-1")

        self.assertEqual(rotated["current_key"], "gm-2")
        self.assertEqual(rotated["invalid_api_keys"], [])
        self.assertEqual(rotated["key_error_timestamps"], {})
        saved_profile = store.load()["profiles"][0]
        self.assertEqual(saved_profile["invalid_api_keys"], [])
        self.assertEqual(saved_profile["key_error_timestamps"], {})


if __name__ == "__main__":
    unittest.main()
