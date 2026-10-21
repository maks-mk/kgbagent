"""Shared registry for processes the agent starts in the background.

The background mode of ``cli_exec`` (``tools.local_shell``) starts processes
that outlive a single tool call. They live in one registry so that the
``SafetyPolicy`` limit, zombie cleanup and shutdown-on-exit all operate on
the same set of processes.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)

_PROCESSES: dict[int, subprocess.Popen] = {}
_LOCK = threading.Lock()
_STARTING = 0


class BackgroundProcessLimitError(RuntimeError):
    """No shared background-process slot is available."""


@contextmanager
def reserve_slot(limit: int) -> Iterator[Callable[[subprocess.Popen], int]]:
    """Reserve capacity until startup and registration finish, even on failure."""

    global _STARTING
    with _LOCK:
        finished = [pid for pid, process in _PROCESSES.items() if process.poll() is not None]
        for pid in finished:
            _PROCESSES.pop(pid, None)
        if len(_PROCESSES) + _STARTING >= limit:
            raise BackgroundProcessLimitError(f"Maximum background processes ({limit}) reached.")
        _STARTING += 1
    pending = True

    def register_reserved(process: subprocess.Popen) -> int:
        nonlocal pending
        global _STARTING
        pid = int(process.pid)
        with _LOCK:
            if not pending:
                raise RuntimeError("Background slot already consumed.")
            _PROCESSES[pid] = process
            _STARTING -= 1
            pending = False
        return pid

    try:
        yield register_reserved
    finally:
        with _LOCK:
            if pending:
                _STARTING -= 1


def registry() -> dict[int, subprocess.Popen]:
    """Live ``pid -> Popen`` mapping shared by every background-capable tool."""

    return _PROCESSES


def count() -> int:
    """Number of tracked processes that are still running."""

    cleanup_finished()
    with _LOCK:
        return len(_PROCESSES)


def register(process: subprocess.Popen) -> int:
    """Track a freshly started process and return its PID."""

    pid = int(process.pid)
    with _LOCK:
        _PROCESSES[pid] = process
    return pid


def get(pid: int) -> Optional[subprocess.Popen]:
    with _LOCK:
        return _PROCESSES.get(int(pid))


def unregister(pid: int) -> Optional[subprocess.Popen]:
    with _LOCK:
        return _PROCESSES.pop(int(pid), None)


def cleanup_finished() -> list[int]:
    """Forget processes that already exited; returns the removed PIDs."""

    with _LOCK:
        finished = [pid for pid, process in _PROCESSES.items() if process.poll() is not None]
        for pid in finished:
            _PROCESSES.pop(pid, None)
    return finished


def terminate(pid: int, *, timeout: float = 3.0) -> bool:
    """Stop a tracked process together with its children.

    Returns ``False`` when ``pid`` is not tracked or could not be stopped.
    A process that survives termination stays registered for another attempt.
    """

    process = get(pid)
    if process is None:
        return False
    if process.poll() is None:
        _terminate_tree(process, timeout=timeout)
    if process.poll() is None:
        return False
    reader = getattr(process, "_cli_output_reader", None)
    if reader is not None:
        reader.join(timeout=timeout)
    with _LOCK:
        if _PROCESSES.get(pid) is process:
            _PROCESSES.pop(pid)
    return True


def _terminate_tree(process: subprocess.Popen, *, timeout: float) -> None:
    if os.name == "nt":
        _terminate_tree_windows(process, timeout=timeout)
        return
    _terminate_tree_posix(process, timeout=timeout)


def _terminate_tree_windows(process: subprocess.Popen, *, timeout: float) -> None:
    try:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout + 2,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("taskkill failed for background process %s.", process.pid, exc_info=True)
    else:
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            logger.debug("Background process %s survived taskkill; killing it directly.", process.pid)
    _kill_directly(process, timeout=timeout)


def _terminate_tree_posix(process: subprocess.Popen, *, timeout: float) -> None:
    # Background processes are started in their own session, so the PID doubles
    # as the process-group id; verify it before signalling the whole group.
    pgid: Optional[int] = None
    try:
        candidate = os.getpgid(process.pid)
    except OSError:
        candidate = None
    if candidate == process.pid:
        pgid = candidate

    def _signal(sig: int) -> None:
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except OSError:
            pass

    _signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        logger.debug("Background process %s survived SIGTERM; sending SIGKILL.", process.pid)
    _signal(signal.SIGKILL)
    _kill_directly(process, timeout=timeout)


def _kill_directly(process: subprocess.Popen, *, timeout: float) -> None:
    try:
        process.kill()
    except OSError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Background process %s did not exit after termination.", process.pid)


def shutdown_all() -> None:
    """Terminate every tracked process; registered as an interpreter exit hook."""

    with _LOCK:
        pids = list(_PROCESSES)
    for pid in pids:
        try:
            terminate(pid, timeout=1.0)
        except Exception:
            logger.debug("Failed to stop background process %s on shutdown.", pid, exc_info=True)


atexit.register(shutdown_all)
