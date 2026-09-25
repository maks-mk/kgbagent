import asyncio
import contextlib
import os
import shutil
import unittest
from pathlib import Path
from uuid import uuid4
from unittest import mock

from core.safety_policy import SafetyPolicy
from core.tool_results import parse_tool_execution_result
from tools import local_shell
from tools.shell_output import (
    DEFAULT_MAX_PERSISTED_CHARS,
    ShellOutputCapture,
    build_persisted_output_envelope,
    format_character_size,
    format_persisted_output_pointer,
    is_persisted_output,
    prune_artifacts,
    resolve_artifact_directory,
    resolve_max_persisted_chars,
)
from tools.shell_semantics import analyze_command, interpret_non_error_exit


class _FakeReader:
    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)

    async def read(self, _size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _FakeProcess:
    def __init__(
        self,
        stdout_chunks: list[bytes] | None = None,
        stderr_chunks: list[bytes] | None = None,
        returncode: int = 0,
        pid: int | None = None,
    ):
        self.stdout = _FakeReader(stdout_chunks or [])
        self.stderr = _FakeReader(stderr_chunks or [])
        self.returncode = returncode
        self.killed = False
        self.pid = pid

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True


class _WorkspaceTestCase(unittest.TestCase):
    def _workspace_tempdir(self) -> Path:
        path = Path.cwd() / ".tmp_tests" / uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def _bind_workspace(self, workspace: Path) -> None:
        previous = local_shell._WORKING_DIRECTORY
        self.addCleanup(lambda: local_shell.set_working_directory(previous))
        local_shell.set_working_directory(str(workspace))

    def _set_raw_limit(self, raw_limit: int) -> None:
        previous = local_shell._SAFETY_POLICY
        self.addCleanup(lambda: local_shell.set_safety_policy(previous))
        local_shell.set_safety_policy(
            SafetyPolicy(allow_shell=True, max_tool_output=500, max_raw_tool_output=raw_limit)
        )

    def _run_cli_exec(self, process: _FakeProcess, payload: dict, wait_for=None) -> str:
        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess
                )
            )
            stack.enter_context(
                mock.patch.object(
                    local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess
                )
            )
            if wait_for is not None:
                stack.enter_context(
                    mock.patch.object(local_shell.asyncio, "wait_for", side_effect=wait_for)
                )
            return asyncio.run(local_shell.cli_exec.ainvoke(payload))


class ShellSemanticsTests(unittest.TestCase):
    def test_protocol_exits_are_recognized_by_executable_segment(self):
        cases = (
            ("rg 'nonexistent' .", "No matches found"),
            ("grep -rn TODO src", "No matches found"),
            ("Get-ChildItem . | Select-String 'foo'", "No matches found"),
            ("git diff --stat", "Files differ"),
            ("git grep -n TODO", "No matches found"),
            ("vulture core/ tools/", "Dead code found"),
            ("pytest -q tests", "Tests failed"),
            ("findstr /n foo bar.txt", "No matches found"),
        )
        for command, expected in cases:
            with self.subTest(command=command):
                self.assertEqual(interpret_non_error_exit(command, 1), expected)

    def test_command_names_inside_arguments_are_not_executables(self):
        for command in (
            "echo grep failed",
            "python -c \"print('pytest')\"",
            "echo 'diff a b'",
            "type vulture",
        ):
            with self.subTest(command=command):
                self.assertIsNone(interpret_non_error_exit(command, 1))

    def test_last_pipeline_and_chained_groups_are_checked(self):
        self.assertEqual(
            interpret_non_error_exit("rg foo . && echo found", 1), "No matches found"
        )
        self.assertEqual(
            interpret_non_error_exit("echo start; rg foo . | Select-Object -First 3", 1),
            "No matches found",
        )
        self.assertEqual(
            interpret_non_error_exit("pytest -q && echo done", 1), "Tests failed"
        )
        # ``;`` reports only the last command, so a real failure after a neutral
        # command must stay an error.
        self.assertIsNone(interpret_non_error_exit("pytest -q; python build.py", 1))
        self.assertIsNone(interpret_non_error_exit("python broken.py && echo done", 1))

    def test_pipelines_without_neutral_leaf_stay_errors(self):
        self.assertIsNone(interpret_non_error_exit("python a.py | python b.py", 1))

    def test_wrappers_and_assignments_are_skipped(self):
        self.assertEqual(interpret_non_error_exit("sudo time rg foo .", 1), "No matches found")
        self.assertEqual(interpret_non_error_exit("CI=1 pytest -q", 1), "Tests failed")

    def test_other_exit_codes_stay_execution_errors(self):
        for exit_code in (2, 127, 130):
            with self.subTest(exit_code=exit_code):
                self.assertIsNone(interpret_non_error_exit("rg foo .", exit_code))

    def test_unparsable_command_uses_token_boundaries(self):
        self.assertEqual(interpret_non_error_exit("rg \"unterminated .", 1), "No matches found")
        # The fallback keeps the previous boundary behaviour: a word glued to
        # its neighbours (``pytestfoo``) still never matches.
        self.assertIsNone(interpret_non_error_exit("echo \"unterminated pytestfoo", 1))

    def test_analyze_command_splits_segments_and_keeps_quoted_pipes(self):
        segments = analyze_command("rg 'a|b' . | Select-String x")
        self.assertIsNotNone(segments)
        self.assertEqual([segment.name for segment in segments], ["rg", "select-string"])
        self.assertEqual(segments[0].argv[1], "a|b")
        self.assertEqual(segments[1].operator_before, "|")

    def test_analyze_command_reports_unbalanced_quotes(self):
        self.assertIsNone(analyze_command("rg 'a b ."))


class ShellTimeoutPolicyTests(unittest.TestCase):
    def test_default_limit_matches_reference_bash_tool(self):
        self.assertEqual(local_shell._max_timeout_seconds(), 600)
        self.assertEqual(local_shell._resolve_timeout(local_shell.DEFAULT_TIMEOUT), (120, False))

    def test_requested_timeout_below_limit_is_untouched(self):
        self.assertEqual(local_shell._resolve_timeout(37), (37, False))

    def test_requested_timeout_above_limit_is_clamped(self):
        self.assertEqual(local_shell._resolve_timeout(99999), (600, True))

    def test_env_override_changes_the_limit(self):
        with mock.patch.dict(os.environ, {local_shell.MAX_TIMEOUT_ENV: "30000"}):
            self.assertEqual(local_shell._max_timeout_seconds(), 30)
            self.assertEqual(local_shell._resolve_timeout(60), (30, True))

    def test_invalid_env_value_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {local_shell.MAX_TIMEOUT_ENV: "not-a-number"}):
            self.assertEqual(local_shell._max_timeout_seconds(), 600)


class ShellOutputCaptureTests(_WorkspaceTestCase):
    def _capture(self, workspace: Path, *, inline_limit: int, **kwargs) -> ShellOutputCapture:
        capture = ShellOutputCapture(
            inline_limit=inline_limit,
            artifact_directory=workspace / "artifacts",
            **kwargs,
        )
        self.addCleanup(capture.close)
        return capture

    def test_small_output_creates_no_artifact(self):
        workspace = self._workspace_tempdir()
        capture = self._capture(workspace, inline_limit=100)
        capture.append("stdout", "hello\n")
        capture.append("stderr", "warn\n")
        capture.close()

        self.assertIsNone(capture.persisted_path)
        self.assertFalse(capture.artifact_truncated)
        self.assertFalse((workspace / "artifacts").exists())
        self.assertEqual(capture.stdout, "hello\n")
        self.assertEqual(capture.stderr, "warn\n")
        self.assertEqual(capture.total_chars, 11)

    def test_artifact_keeps_full_stream_after_inline_overflow(self):
        workspace = self._workspace_tempdir()
        capture = self._capture(workspace, inline_limit=20)
        capture.append("stdout", "head-")
        capture.append("stdout", "x" * 200)
        capture.append("stderr", "err\n")
        capture.close()

        self.assertIsNotNone(capture.persisted_path)
        artifact = Path(capture.persisted_path)
        self.assertTrue(artifact.is_file())
        self.assertEqual(artifact.read_text(encoding="utf-8"), "head-" + "x" * 200 + "err\n")
        self.assertFalse(capture.artifact_truncated)
        self.assertIn("[TRUNCATED:", capture.stdout)

    def test_artifact_is_capped_and_marked(self):
        workspace = self._workspace_tempdir()
        capture = self._capture(workspace, inline_limit=10, max_persisted_chars=50)
        capture.append("stdout", "a" * 40)
        capture.append("stdout", "b" * 40)
        capture.close()

        self.assertIsNotNone(capture.persisted_path)
        content = Path(capture.persisted_path).read_text(encoding="utf-8")
        self.assertTrue(content.startswith("a" * 40))
        self.assertIn("[ARTIFACT TRUNCATED: output exceeded 50 characters]", content)
        self.assertTrue(capture.artifact_truncated)

    def test_unusable_artifact_directory_disables_persistence(self):
        workspace = self._workspace_tempdir()
        blocker = workspace / "not-a-directory"
        blocker.write_text("x", encoding="utf-8")
        capture = ShellOutputCapture(inline_limit=5, artifact_directory=blocker)
        self.addCleanup(capture.close)
        capture.append("stdout", "y" * 50)
        capture.close()

        self.assertIsNone(capture.persisted_path)
        self.assertEqual(capture.total_chars, 50)
        self.assertIn("40 characters omitted", capture.stdout)

    def test_resolve_artifact_directory_defaults_to_workspace_state(self):
        workspace = self._workspace_tempdir()
        resolved = resolve_artifact_directory(str(workspace))
        self.assertEqual(resolved, (workspace / ".agent_state" / "cli_exec").resolve())

    def test_resolve_artifact_directory_honours_env_override(self):
        workspace = self._workspace_tempdir()
        with mock.patch.dict(os.environ, {"CLI_EXEC_RESULTS_DIR": "shell-logs"}):
            self.assertEqual(
                resolve_artifact_directory(str(workspace)),
                (workspace / "shell-logs").resolve(),
            )
        with mock.patch.dict(os.environ, {"CLI_EXEC_RESULTS_DIR": str(workspace / "custom")}):
            self.assertEqual(
                resolve_artifact_directory(str(workspace)),
                (workspace / "custom").resolve(),
            )

    def test_resolve_artifact_directory_ignores_paths_outside_workspace(self):
        workspace = self._workspace_tempdir()
        outside = self._workspace_tempdir()
        with mock.patch.dict(os.environ, {"CLI_EXEC_RESULTS_DIR": str(outside)}):
            self.assertEqual(
                resolve_artifact_directory(str(workspace)),
                (workspace / ".agent_state" / "cli_exec").resolve(),
            )

    def test_prune_keeps_newest_artifacts_only(self):
        workspace = self._workspace_tempdir()
        directory = workspace / "artifacts"
        directory.mkdir()
        for index in range(25):
            (directory / f"cli-exec-20260101-0000{index:02d}-{'0' * 7}{index % 10}.log").write_text(
                "data", encoding="utf-8"
            )
            os.utime(
                directory / f"cli-exec-20260101-0000{index:02d}-{'0' * 7}{index % 10}.log",
                (1_700_000_000 + index, 1_700_000_000 + index),
            )
        (directory / "unrelated.log").write_text("keep me", encoding="utf-8")

        prune_artifacts(directory, 20)

        remaining = sorted(entry.name for entry in directory.iterdir())
        self.assertEqual(len(remaining), 21)
        self.assertIn("unrelated.log", remaining)
        self.assertIn("cli-exec-20260101-000024-00000004.log", remaining)
        self.assertNotIn("cli-exec-20260101-000000-00000000.log", remaining)

    def test_max_persisted_chars_env_override(self):
        with mock.patch.dict(os.environ, {"CLI_EXEC_MAX_PERSISTED_CHARS": "1234"}):
            self.assertEqual(resolve_max_persisted_chars(), 1234)
        with mock.patch.dict(os.environ, {"CLI_EXEC_MAX_PERSISTED_CHARS": "bogus"}):
            self.assertEqual(resolve_max_persisted_chars(), DEFAULT_MAX_PERSISTED_CHARS)


class ShellOutputFormattingTests(unittest.TestCase):
    def test_character_size_uses_decimal_units(self):
        self.assertEqual(format_character_size(0), "0 chars")
        self.assertEqual(format_character_size(999), "999 chars")
        self.assertEqual(format_character_size(1500), "1.5K chars")
        self.assertEqual(format_character_size(2_500_000), "2.5M chars")

    def test_envelope_points_at_the_artifact(self):
        envelope = build_persisted_output_envelope(
            body="preview",
            original_chars=2500,
            persisted_path="C:/ws/.agent_state/cli_exec/cli-exec-1.log",
        )
        self.assertTrue(is_persisted_output(envelope))
        self.assertIn("<persisted-output>", envelope)
        self.assertIn("</persisted-output>", envelope)
        self.assertIn("Output too large (2.5K chars)", envelope)
        self.assertIn("C:/ws/.agent_state/cli_exec/cli-exec-1.log", envelope)
        self.assertIn("preview", envelope)
        self.assertNotIn("capped as well", envelope)

    def test_envelope_mentions_capped_artifact(self):
        envelope = build_persisted_output_envelope(
            body="preview",
            original_chars=5000,
            persisted_path="artifact.log",
            artifact_truncated=True,
        )
        self.assertIn("capped as well", envelope)

    def test_pointer_keeps_error_prefix_readable(self):
        pointer = format_persisted_output_pointer(
            original_chars=3000,
            persisted_path="artifact.log",
        )
        parsed = parse_tool_execution_result(f"ERROR[EXECUTION]: boom\n\n{pointer}")
        self.assertFalse(parsed.ok)
        self.assertEqual(parsed.error_type, "EXECUTION")
        self.assertIn("Full output saved to: artifact.log", parsed.message)

    def test_is_persisted_output_rejects_plain_text(self):
        self.assertFalse(is_persisted_output("plain output"))


class CliExecPersistenceTests(_WorkspaceTestCase):
    def test_small_output_has_no_envelope_or_artifact(self):
        workspace = self._workspace_tempdir()
        self._bind_workspace(workspace)
        self._set_raw_limit(1500)
        process = _FakeProcess(stdout_chunks=[b"ok\n"])

        result = self._run_cli_exec(process, {"command": "demo"})

        self.assertIn("ok", result)
        self.assertNotIn("<persisted-output>", result)
        self.assertFalse((workspace / ".agent_state").exists())

    def test_oversized_output_is_persisted_and_referenced(self):
        workspace = self._workspace_tempdir()
        self._bind_workspace(workspace)
        self._set_raw_limit(1500)
        process = _FakeProcess(stdout_chunks=[b"y" * 4000])

        result = self._run_cli_exec(process, {"command": "demo"})

        self.assertIn("<persisted-output>", result)
        self.assertIn("Full output saved to:", result)
        self.assertIn("Output too large (4.0K chars)", result)
        self.assertRegex(result, r"\[TRUNCATED from \d+ chars \| source=shell-raw\]")
        self.assertFalse(result.startswith("ERROR"))
        artifacts = list((workspace / ".agent_state" / "cli_exec").iterdir())
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].read_text(encoding="utf-8"), "y" * 4000)

    def test_error_result_keeps_prefix_and_points_to_artifact(self):
        workspace = self._workspace_tempdir()
        self._bind_workspace(workspace)
        self._set_raw_limit(1500)
        process = _FakeProcess(stdout_chunks=[b"z" * 4000], stderr_chunks=[b"boom\n"], returncode=2)

        result = self._run_cli_exec(process, {"command": "demo --fail"})

        self.assertTrue(result.startswith("ERROR[EXECUTION]"))
        self.assertIn("Full output saved to:", result)
        self.assertNotIn("<persisted-output>", result)
        parsed = parse_tool_execution_result(result)
        self.assertFalse(parsed.ok)
        self.assertEqual(parsed.error_type, "EXECUTION")
        artifacts = list((workspace / ".agent_state" / "cli_exec").iterdir())
        self.assertEqual(len(artifacts), 1)
        content = artifacts[0].read_text(encoding="utf-8")
        self.assertIn("z" * 4000, content)
        self.assertIn("boom", content)

    def test_neutral_exit_result_is_wrapped_and_reports_meaning(self):
        workspace = self._workspace_tempdir()
        self._bind_workspace(workspace)
        self._set_raw_limit(1500)
        process = _FakeProcess(stdout_chunks=[b"n" * 4000], returncode=1)

        result = self._run_cli_exec(process, {"command": "rg 'x' ."})

        self.assertIn("Exit Code: 1 (No matches found)", result)
        self.assertIn("Full output saved to:", result)
        self.assertNotIn("ERROR[", result)

    def test_execution_failure_without_artifact_stays_plain(self):
        workspace = self._workspace_tempdir()
        self._bind_workspace(workspace)
        self._set_raw_limit(1500)
        process = _FakeProcess(stderr_chunks=[b"oops\n"], returncode=2)

        result = self._run_cli_exec(process, {"command": "demo --fail"})

        self.assertTrue(result.startswith("ERROR[EXECUTION]"))
        self.assertNotIn("Full output saved to:", result)

    def test_timeout_limit_is_clamped_and_reported(self):
        workspace = self._workspace_tempdir()
        self._bind_workspace(workspace)
        captured_timeouts: list[int | float] = []

        async def _fake_wait_for(awaitable, timeout):
            captured_timeouts.append(timeout)
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise asyncio.TimeoutError

        process = _FakeProcess(stdout_chunks=[b"partial\n"])
        with mock.patch.dict(os.environ, {local_shell.MAX_TIMEOUT_ENV: "30000"}):
            result = self._run_cli_exec(
                process, {"command": "demo", "timeout": 3600}, wait_for=_fake_wait_for
            )

        self.assertIn("ERROR[TIMEOUT]", result)
        self.assertIn("timed out after 30 seconds", result)
        self.assertIn("clamped by CLI_EXEC_MAX_TIMEOUT_MS", result)
        self.assertEqual(captured_timeouts[0], 30)


if __name__ == "__main__":
    unittest.main()
