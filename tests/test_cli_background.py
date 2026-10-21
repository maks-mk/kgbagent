"""Background mode of cli_exec: detached start, shared registry, no hangs."""

import asyncio
import contextlib
import io
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import shutil
import sys
import time
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

from core.policy_engine import tool_requires_approval
from core.safety_policy import SafetyPolicy
from tools import background_processes, local_shell


class _BlockingReader:
    """Stream reader that never reaches EOF, like a pipe held by a grandchild."""

    async def read(self, _size: int) -> bytes:
        await asyncio.Event().wait()
        return b""


class _HangingProcess:
    """Fake subprocess that never exits and keeps both pipes open."""

    def __init__(self) -> None:
        self.stdout = _BlockingReader()
        self.stderr = _BlockingReader()
        self.returncode = None
        self.pid = None
        self.killed = False

    async def wait(self) -> int:
        await asyncio.Event().wait()
        return 0

    def kill(self) -> None:
        self.killed = True


class _ExitedProcess(_HangingProcess):
    """The command exited, but a child it spawned still owns stdout/stderr."""

    def __init__(self) -> None:
        super().__init__()
        self.returncode = 0

    async def wait(self) -> int:
        return 0


class CliExecBackgroundTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_policy = local_shell._SAFETY_POLICY
        local_shell.set_safety_policy(
            SafetyPolicy(allow_shell=True, max_background_processes=5)
        )
        self._workspace = Path.cwd() / ".tmp_tests" / uuid4().hex
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._previous_directory = local_shell._WORKING_DIRECTORY
        local_shell.set_working_directory(str(self._workspace))

    def tearDown(self) -> None:
        background_processes.shutdown_all()
        local_shell.set_working_directory(self._previous_directory)
        local_shell.set_safety_policy(self._previous_policy)
        shutil.rmtree(self._workspace, ignore_errors=True)

    def _run(self, payload: dict) -> str:
        return asyncio.run(local_shell.cli_exec.ainvoke(payload))

    def _python_command(self, body: str) -> str:
        prefix = "& " if os.name == "nt" else ""
        return f'{prefix}"{sys.executable}" -c "{body}"'

    def _sleep_command(self, seconds: int = 30) -> str:
        return self._python_command(
            "import pathlib,time; pathlib.Path('ready').write_text('ready'); "
            f"time.sleep({seconds})"
        )

    def _wait_until(self, predicate, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("Background process did not reach the expected state.")
            time.sleep(0.02)

    def _log_path(self, result: str) -> Path:
        return self._workspace / result.split("Log: ", 1)[1].splitlines()[0].strip()

    def _started_pid(self, result: str) -> int:
        self.assertFalse(result.startswith("ERROR"), result)
        return int(result.split("PID ", 1)[1].split(".", 1)[0])

    def _run_fake_process(self, process, payload: dict, *, force_timeout: bool = True) -> str:
        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        async def _fake_wait_for(awaitable, timeout):
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise asyncio.TimeoutError

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
            if force_timeout:
                stack.enter_context(
                    mock.patch.object(local_shell.asyncio, "wait_for", side_effect=_fake_wait_for)
                )
            stack.enter_context(mock.patch.object(local_shell, "STREAM_DRAIN_TIMEOUT", 0.2))
            return asyncio.run(local_shell.cli_exec.ainvoke(payload))

    def test_background_stop_hint_uses_platform_command(self):
        for platform, command in (("nt", "taskkill /PID 123 /T /F"), ("posix", "kill -TERM -123")):
            with self.subTest(platform=platform), mock.patch.object(local_shell.os, "name", platform):
                self.assertEqual(local_shell._background_stop_hint(123), f'cli_exec(command="{command}")')

    def test_background_argument_is_part_of_the_tool_schema(self):
        schema = local_shell.cli_exec.get_input_schema().model_json_schema()

        self.assertIn("background", schema["properties"])

    def test_background_start_returns_immediately_with_pid_and_log(self):
        started = time.monotonic()

        result = self._run({"command": self._sleep_command(), "background": True})

        elapsed = time.monotonic() - started
        pid = self._started_pid(result)
        self.addCleanup(background_processes.terminate, pid)
        self.assertLess(elapsed, 15.0, result)
        self.assertIn(pid, background_processes.registry())
        self._wait_until(lambda: (self._workspace / "ready").is_file())
        self.assertIsNone(background_processes.get(pid).poll(), result)

        log_path = self._log_path(result)
        self.assertTrue(log_path.is_file(), log_path)
        self.assertIn("cli_exec background", log_path.read_text(encoding="utf-8"))
        self.assertIn(local_shell._background_stop_hint(pid), result)

    def test_background_process_is_stoppable_through_registry(self):
        result = self._run({"command": self._sleep_command(), "background": True})
        pid = self._started_pid(result)
        process = background_processes.get(pid)

        self.assertTrue(background_processes.terminate(pid))
        self.assertIsNotNone(process.poll())
        self.assertNotIn(pid, background_processes.registry())

    def test_background_mode_enforces_process_limit(self):
        local_shell.set_safety_policy(
            SafetyPolicy(allow_shell=True, max_background_processes=1)
        )
        first = self._run({"command": self._sleep_command(), "background": True})
        pid = self._started_pid(first)
        self.addCleanup(background_processes.terminate, pid)

        second = self._run({"command": self._sleep_command(), "background": True})

        self.assertTrue(second.startswith("ERROR[LIMIT_EXCEEDED]"), second)
        self.assertIn("Maximum background processes (1)", second)

    def test_background_mode_bypasses_foreground_service_guard(self):
        with mock.patch.object(
            local_shell, "_start_background_command", return_value="Success: stub"
        ) as starter:
            result = self._run({"command": "python -m http.server 8765", "background": True})

        self.assertEqual(result, "Success: stub")
        starter.assert_called_once()

    def test_foreground_service_command_points_to_background_mode(self):
        result = self._run({"command": "python -m http.server 8765"})

        self.assertTrue(result.startswith("ERROR[VALIDATION]"), result)
        self.assertIn("background=true", result)
        self.assertNotIn("run_background_process", result)

    def test_foreground_service_guard_covers_interpreter_paths(self):
        commands = (
            r".\venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1",
            "./venv/bin/python -m http.server 8765",
            "py -m http.server 8765",
            r".\venv\Scripts\uvicorn.exe app.main:app",
            r".\venv\Scripts\flask.exe run",
        )
        for command in commands:
            with self.subTest(command=command):
                with mock.patch.object(
                    local_shell.asyncio,
                    "create_subprocess_exec",
                    side_effect=AssertionError("foreground service must not be launched"),
                ), mock.patch.object(
                    local_shell.asyncio,
                    "create_subprocess_shell",
                    side_effect=AssertionError("foreground service must not be launched"),
                ):
                    result = self._run({"command": command})

                self.assertTrue(result.startswith("ERROR[VALIDATION]"), result)
                self.assertIn("background=true", result)

    def test_timeout_does_not_hang_when_pipes_stay_open(self):
        started = time.monotonic()

        result = self._run_fake_process(_HangingProcess(), {"command": "demo", "timeout": 5})

        elapsed = time.monotonic() - started
        self.assertTrue(result.startswith("ERROR[TIMEOUT]"), result)
        self.assertIn("background=true", result)
        self.assertLess(elapsed, 10.0, result)

    def test_exited_command_with_open_pipes_returns_note_instead_of_hanging(self):
        started = time.monotonic()

        result = self._run_fake_process(_ExitedProcess(), {"command": "demo"}, force_timeout=False)

        elapsed = time.monotonic() - started
        self.assertFalse(result.startswith("ERROR"), result)
        self.assertIn("kept the command's stdout/stderr open", result)
        self.assertLess(elapsed, 10.0, result)

    def _finished_background(self, command: str) -> tuple[str, Path]:
        result = self._run({"command": command, "background": True})
        pid = self._started_pid(result)
        process = background_processes.get(pid)
        self.assertEqual(process.wait(timeout=15), 0, result)
        process._cli_output_reader.join(timeout=5)
        self.assertFalse(process._cli_output_reader.is_alive())
        return result, self._log_path(result)

    def test_background_log_normalizes_legacy_output_and_nul_to_utf8(self):
        command = self._python_command(
            "import sys; sys.stdout.buffer.write(bytes([207,240,232,226,229,242,0,10])); "
            "sys.stderr.buffer.write(b'stderr-ready')"
        )
        _, log_path = self._finished_background(command)
        text = log_path.read_text(encoding="utf-8")
        self.assertIn("stderr-ready", text)
        self.assertNotIn("\x00", text)
        self.assertIn("\ufffd", text)
        from tools.filesystem_impl.manager import FilesystemManager
        result = FilesystemManager(self._workspace).read_file(str(log_path))
        self.assertFalse(result.startswith("ERROR"), result)
        self.assertIn("stderr-ready", result)

    def test_background_log_keeps_unicode_output(self):
        command = self._python_command(
            "import sys; sys.stdout.buffer.write((chr(1046)+chr(128512)).encode('utf-8'))"
        )
        _, log_path = self._finished_background(command)
        self.assertIn(chr(1046) + chr(128512), log_path.read_text(encoding="utf-8"))

    def test_background_log_cap_keeps_draining_large_output(self):
        command = self._python_command("import sys; sys.stdout.buffer.write(b'x'*1048576)")
        with mock.patch.object(local_shell, "resolve_max_persisted_chars", return_value=32):
            _, log_path = self._finished_background(command)
        text = log_path.read_text(encoding="utf-8")
        marker = "\n[ARTIFACT TRUNCATED: output exceeded 32 characters]\n"
        self.assertEqual(len(text), 32 + len(marker))
        self.assertEqual(text.count(marker), 1)
        self.assertNotIn("x" * 4096, text)

    def test_background_log_reading_continues_after_write_error(self):
        from tools.shell_output import PersistedOutputWriter

        for close_fails in (False, True):
            with self.subTest(close_fails=close_fails):
                stream = io.BytesIO(b"x" * 200000)
                process = mock.Mock(stdout=mock.Mock(wraps=stream), pid=900001)
                handle = mock.Mock()
                handle.write.side_effect = OSError("disk full")
                if close_fails:
                    handle.close.side_effect = OSError("disk full on close")
                writer = PersistedOutputWriter(handle, 50)
                with self.assertLogs(local_shell.__name__, level="WARNING"):
                    local_shell._capture_background_output(process, handle, writer)
                self.assertEqual(process.stdout.read1.call_count, 5)  # All chunks and EOF.
                self.assertTrue(stream.closed)
                handle.close.assert_called_once()
                handle.write.assert_called_once()

    def test_concurrent_background_starts_share_the_same_limit(self):
        local_shell.set_safety_policy(SafetyPolicy(allow_shell=True, max_background_processes=1))
        entered = threading.Event()
        release = threading.Event()
        process = mock.Mock(spec=subprocess.Popen, pid=900001, stdout=io.BytesIO())
        process.poll.return_value = None

        def slow_popen(*_args, **_kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test did not release startup")
            return process

        def start_cli():
            return self._run({"command": "echo demo", "background": True})

        try:
            with mock.patch.object(local_shell.subprocess, "Popen", side_effect=slow_popen) as popen:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(start_cli)
                    try:
                        self.assertTrue(entered.wait(timeout=5))
                        denied = start_cli()
                        self.assertTrue(denied.startswith("ERROR[LIMIT_EXCEEDED]"), denied)
                    finally:
                        release.set()
                    self.assertIn("Success", future.result(timeout=5))
                self.assertEqual(popen.call_count, 1)
        finally:
            background_processes.unregister(900001)
            reader = getattr(process, "_cli_output_reader", None)
            if reader is not None:
                reader.join(timeout=5)

    def test_registered_process_consumes_reservation_without_double_counting(self):
        process = mock.Mock(spec=subprocess.Popen, pid=900005)
        process.poll.return_value = None
        try:
            with background_processes.reserve_slot(2) as register:
                self.assertEqual(register(process), process.pid)
                self.assertEqual(background_processes._STARTING, 0)
                with background_processes.reserve_slot(2):
                    self.assertEqual(background_processes._STARTING, 1)
                    with self.assertRaises(background_processes.BackgroundProcessLimitError):
                        with background_processes.reserve_slot(2):
                            self.fail("Limit must reject a third process")
        finally:
            background_processes.unregister(process.pid)
        self.assertEqual(background_processes._STARTING, 0)

    def test_failed_start_releases_shared_slot(self):
        local_shell.set_safety_policy(SafetyPolicy(allow_shell=True, max_background_processes=1))
        with mock.patch.object(local_shell.subprocess, "Popen", side_effect=OSError("startup failed")):
            for _ in range(2):
                result = self._run({"command": "echo demo", "background": True})
                self.assertTrue(result.startswith("ERROR[EXECUTION]"), result)
        with background_processes.reserve_slot(1):
            self.assertEqual(background_processes._STARTING, 1)
        self.assertEqual(background_processes._STARTING, 0)

    def test_failed_termination_retains_process_for_retry(self):
        process = mock.Mock(spec=subprocess.Popen, pid=900002)
        process.poll.return_value = None
        background_processes.register(process)
        try:
            with mock.patch.object(background_processes, "_terminate_tree", side_effect=OSError("denied")):
                with self.assertRaisesRegex(OSError, "denied"):
                    background_processes.terminate(process.pid)
            self.assertIs(background_processes.get(process.pid), process)
            process.poll.return_value = 0
            self.assertTrue(background_processes.terminate(process.pid))
            self.assertIsNone(background_processes.get(process.pid))
        finally:
            background_processes.unregister(process.pid)

    def test_surviving_kill_does_not_report_success_or_forget_process(self):
        process = mock.Mock(spec=subprocess.Popen, pid=900003)
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("fake", 0.01)
        background_processes.register(process)
        try:
            with mock.patch.object(background_processes, "_terminate_tree", side_effect=lambda p, **kw: background_processes._kill_directly(p, **kw)):
                with mock.patch.object(background_processes.logger, "warning"):
                    self.assertFalse(background_processes.terminate(process.pid))
            self.assertIs(background_processes.get(process.pid), process)
            process.kill.assert_called_once()
        finally:
            background_processes.unregister(process.pid)

    def test_slow_start_does_not_block_event_loop(self):
        entered = threading.Event()
        release = threading.Event()

        def slow_start(*_args, **_kwargs):
            entered.set()
            release.wait(timeout=0.5)
            return "Success: stub"

        async def check():
            starter = asyncio.create_task(local_shell.cli_exec.ainvoke({"command": "echo demo", "background": True}))
            start = time.monotonic()
            try:
                while not entered.is_set():
                    await asyncio.sleep(0.005)
                await asyncio.sleep(0.03)
                self.assertLess(time.monotonic() - start, 0.3)
            finally:
                release.set()
                self.assertEqual(await starter, "Success: stub")

        with mock.patch.object(local_shell, "_start_background_command", side_effect=slow_start):
            asyncio.run(check())

    def test_cancelled_start_waits_for_registration_and_stops_process(self):
        entered = threading.Event()
        release = threading.Event()
        created = []
        real_popen = subprocess.Popen

        def slow_popen(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test did not release startup")
            process = real_popen(*args, **kwargs)
            if kwargs.get("stdout") == subprocess.PIPE:
                created.append(process)
            return process

        async def check():
            starter = asyncio.create_task(local_shell.cli_exec.ainvoke({"command": self._sleep_command(), "background": True}))
            try:
                deadline = time.monotonic() + 5
                while not entered.is_set() and time.monotonic() < deadline:
                    await asyncio.sleep(0.005)
                self.assertTrue(entered.is_set())
                starter.cancel()
                await asyncio.sleep(0.02)
                starter.cancel()  # Repeated cancellation must not interrupt cleanup.
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await starter
            self.assertEqual(len(created), 1)
            self.assertIsNotNone(created[0].poll())
            self.assertIsNone(background_processes.get(created[0].pid))
            self.assertEqual(background_processes._STARTING, 0)

        with mock.patch.object(local_shell.subprocess, "Popen", side_effect=slow_popen):
            asyncio.run(check())

    def test_background_tree_stop_terminates_ready_child(self):
        import psutil
        command = self._python_command(
            "import sys,subprocess,pathlib,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "pathlib.Path('child.pid').write_text(str(p.pid)); time.sleep(30)"
        )
        result = self._run({"command": command, "background": True})
        pid = self._started_pid(result)
        self._wait_until(lambda: (self._workspace / "child.pid").is_file())
        child_pid = int((self._workspace / "child.pid").read_text())
        child = psutil.Process(child_pid)
        try:
            self.assertTrue(background_processes.terminate(pid))
            self._wait_until(lambda: not child.is_running() or child.status() == psutil.STATUS_ZOMBIE)
            self.assertIsNone(background_processes.get(pid))
        finally:
            if child.is_running():
                try:
                    child.kill()
                    child.wait(timeout=5)
                except psutil.NoSuchProcess:
                    pass

    def test_reader_start_failure_stops_process_and_releases_slot(self):
        real_start = threading.Thread.start

        def start_thread(thread):
            if thread.name.startswith("cli-exec-output-"):
                raise RuntimeError("no reader")
            return real_start(thread)

        with mock.patch.object(local_shell.threading.Thread, "start", new=start_thread):
            result = self._run({"command": self._sleep_command(), "background": True})
        self.assertTrue(result.startswith("ERROR[EXECUTION]"), result)
        self.assertIn("no reader", result)
        self.assertEqual(background_processes.count(), 0)
        self.assertEqual(background_processes._STARTING, 0)

    def test_reader_start_failure_retains_surviving_process_and_original_error(self):
        process = mock.Mock(spec=subprocess.Popen, pid=900006, stdout=io.BytesIO())
        process.poll.return_value = None
        try:
            with mock.patch.object(local_shell.subprocess, "Popen", return_value=process):
                with mock.patch.object(local_shell.threading.Thread, "start", side_effect=RuntimeError("no reader")):
                    with mock.patch.object(background_processes, "_terminate_tree", side_effect=OSError("denied")):
                        with self.assertLogs(local_shell.__name__, level="WARNING"):
                            result = local_shell._start_background_command("echo demo", os.environ.copy())
            self.assertTrue(result.startswith("ERROR[EXECUTION]"), result)
            self.assertIn("no reader", result)
            self.assertIn(local_shell._background_stop_hint(process.pid), result)
            self.assertIs(background_processes.get(process.pid), process)
            self.assertTrue(process.stdout.closed)
            self.assertEqual(background_processes._STARTING, 0)
        finally:
            background_processes.unregister(process.pid)

    def test_cancelled_start_preserves_cancellation_when_cleanup_fails(self):
        process = mock.Mock(spec=subprocess.Popen, pid=900007)
        process.poll.return_value = None
        entered = threading.Event()
        release = threading.Event()

        def start(_command, _env, *, started_processes):
            background_processes.register(process)
            started_processes.append(process)
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test did not release startup")
            return "Success: stub"

        async def check():
            task = asyncio.create_task(local_shell._start_background_async("echo demo", {}))
            try:
                deadline = time.monotonic() + 5
                while not entered.is_set() and time.monotonic() < deadline:
                    await asyncio.sleep(0.005)
                self.assertTrue(entered.is_set())
                task.cancel()
                await asyncio.sleep(0.01)
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertIs(background_processes.get(process.pid), process)

        try:
            with mock.patch.object(local_shell, "_start_background_command", side_effect=start):
                with mock.patch.object(background_processes, "_terminate_tree", side_effect=OSError("denied")):
                    with self.assertLogs(local_shell.__name__, level="WARNING"):
                        asyncio.run(check())
        finally:
            background_processes.unregister(process.pid)

    def test_decoder_keeps_only_incomplete_utf8_suffix(self):
        decoder = local_shell._CliOutputDecoder()
        self.assertEqual(decoder.decode(b"a" * 65535 + b"\xf0"), "a" * 65535)
        self.assertEqual(len(decoder._pending), 1)
        self.assertEqual(decoder.decode(b"\x9f\x98\x80"), chr(128512))
        self.assertEqual(len(decoder._pending), 0)

    def test_background_mode_always_requires_approval(self):
        self.assertFalse(tool_requires_approval("cli_exec", {"command": "dir"}))
        self.assertTrue(
            tool_requires_approval("cli_exec", {"command": "dir", "background": True})
        )


if __name__ == "__main__":
    unittest.main()
