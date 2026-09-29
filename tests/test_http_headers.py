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
    def test_default_path_uses_base_dir_headers_json(self):
        with mock.patch.object(http_headers, "BASE_DIR", Path("/base")):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(http_headers._default_headers_path(), Path("/base") / "headers.json")

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


if __name__ == "__main__":
    unittest.main()
