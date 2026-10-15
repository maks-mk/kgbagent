import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import core.http_headers as http_headers
from core.http_headers import load_openai_headers, load_provider_headers


class OpenAIHeadersTests(unittest.TestCase):
    def test_missing_file_returns_empty_for_standard_sdk_headers(self):
        with TemporaryDirectory() as temp_dir:
            result = load_openai_headers(Path(temp_dir) / "missing.json")

        self.assertEqual(result, {})

    def test_json_values_override_and_add_headers(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "headers.json"
            path.write_text(
                json.dumps({"User-Agent": "CustomAgent/1.0", "x-custom": "enabled", "ignored": 42}),
                encoding="utf-8",
            )

            result = load_openai_headers(path)

        self.assertEqual(result, {"User-Agent": "CustomAgent/1.0", "x-custom": "enabled"})

    def test_invalid_json_returns_empty(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "headers.json"
            path.write_text("{invalid", encoding="utf-8")

            result = load_openai_headers(path)

        self.assertEqual(result, {})

    def test_non_object_json_returns_empty(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "headers.json"
            path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

            result = load_openai_headers(path)

        self.assertEqual(result, {})


class HeadersFileEnvTests(unittest.TestCase):
    @staticmethod
    def _patched_config(headers_file: str):
        config = mock.MagicMock()
        config.headers_file = headers_file
        return mock.patch("core.config.AgentConfig", return_value=config)

    def test_default_path_uses_base_dir_codex_headers_json(self):
        with mock.patch.object(http_headers, "BASE_DIR", Path("/base")):
            with mock.patch.dict(os.environ, {}, clear=True):
                with self._patched_config("codex_headers.json"):
                    self.assertEqual(
                        http_headers._default_headers_path(), Path("/base") / "codex_headers.json"
                    )

    def test_config_value_is_used_when_env_var_is_absent(self):
        """``HEADERS_FILE`` from .env / config.json must be honoured."""
        with mock.patch.object(http_headers, "BASE_DIR", Path("/base")):
            with mock.patch.dict(os.environ, {}, clear=True):
                with self._patched_config("conf/from_dotenv.json"):
                    self.assertEqual(
                        http_headers._default_headers_path(),
                        Path("/base") / "conf/from_dotenv.json",
                    )

    def test_env_var_wins_over_config_value(self):
        with mock.patch.object(http_headers, "BASE_DIR", Path("/base")):
            with mock.patch.dict(os.environ, {"HEADERS_FILE": "conf/from_process.json"}, clear=True):
                with self._patched_config("conf/from_dotenv.json"):
                    self.assertEqual(
                        http_headers._default_headers_path(),
                        Path("/base") / "conf/from_process.json",
                    )

    def test_blank_config_value_falls_back_to_default(self):
        with mock.patch.object(http_headers, "BASE_DIR", Path("/base")):
            with mock.patch.dict(os.environ, {"HEADERS_FILE": "   "}, clear=True):
                with self._patched_config("  "):
                    self.assertEqual(
                        http_headers._default_headers_path(), Path("/base") / "codex_headers.json"
                    )

    def test_config_absolute_path_is_honored(self):
        abs_path = Path("/abs/from_config.json").resolve()
        with mock.patch.object(http_headers, "BASE_DIR", Path("/base")):
            with mock.patch.dict(os.environ, {}, clear=True):
                with self._patched_config(str(abs_path)):
                    self.assertEqual(http_headers._default_headers_path(), abs_path)

    def test_env_relative_name_resolved_against_base_dir(self):
        with mock.patch.object(http_headers, "BASE_DIR", Path("/base")):
            with mock.patch.dict(os.environ, {"HEADERS_FILE": "conf/custom.json"}, clear=True):
                self.assertEqual(
                    http_headers._default_headers_path(),
                    Path("/base") / "conf/custom.json",
                )

    def test_env_absolute_path_is_honored(self):
        with mock.patch.object(http_headers, "BASE_DIR", Path("/base")):
            abs_path = Path("/abs/custom.json").resolve()
            with mock.patch.dict(os.environ, {"HEADERS_FILE": str(abs_path)}, clear=True):
                self.assertEqual(http_headers._default_headers_path(), abs_path)

    def test_env_override_is_loaded_end_to_end(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "qwen_headers.json"
            path.write_text(json.dumps({"User-Agent": "QwenCode/1.0"}), encoding="utf-8")

            with mock.patch.object(http_headers, "BASE_DIR", Path(temp_dir)):
                with mock.patch.dict(os.environ, {"HEADERS_FILE": "qwen_headers.json"}, clear=True):
                    self.assertEqual(load_provider_headers(), {"User-Agent": "QwenCode/1.0"})


class SectionedHeadersTests(unittest.TestCase):
    def test_only_default_headers_section_is_applied(self):
        payload = {
            "_note": "example document",
            "default_headers": {
                "User-Agent": "codex_cli_rs/0.160.1",
                "originator": "codex_cli_rs",
                "X-OpenAI-Internal-Codex-Residency": "us",
            },
            "chatgpt_backend_auth_headers": {"Authorization": "Bearer example", "Cookie": "session=x"},
            "responses_http_headers": {"session-id": "fixed-session", "Content-Type": "application/json"},
        }
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "codex_headers.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            result = load_provider_headers(path)

        self.assertEqual(result, payload["default_headers"])

    def test_non_object_default_headers_section_returns_empty(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "codex_headers.json"
            path.write_text(json.dumps({"default_headers": ["User-Agent"]}), encoding="utf-8")

            with self.assertLogs("agent", level="WARNING"):
                self.assertEqual(load_provider_headers(path), {})

    def test_shipped_codex_headers_never_override_auth_or_cookies(self):
        result = load_provider_headers(Path(http_headers.BASE_DIR) / http_headers.DEFAULT_HEADERS_FILE)

        self.assertEqual(result.get("originator"), "codex_cli_rs")
        self.assertTrue(result.get("User-Agent", "").startswith("codex_cli_rs/"))
        for forbidden in ("Authorization", "Cookie", "Host", "Content-Length", "ChatGPT-Account-ID"):
            self.assertNotIn(forbidden, result)


if __name__ == "__main__":
    unittest.main()
