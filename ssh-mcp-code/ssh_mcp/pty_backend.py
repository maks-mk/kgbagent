from __future__ import annotations

import errno
import os
import select
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any


class InteractivePtyError(RuntimeError):
    """Raised when a local interactive PTY cannot be created."""


@dataclass(slots=True)
class InteractiveProcess:
    """Cross-platform handle for a local process attached to a PTY."""

    process: Any
    master_fd: int | None = None
    windows_pty: bool = False
    _closed: bool = False

    @classmethod
    def spawn(cls, argv: list[str]) -> "InteractiveProcess":
        if os.name == "nt":
            return cls._spawn_windows(argv)
        return cls._spawn_posix(argv)

    @classmethod
    def _spawn_posix(cls, argv: list[str]) -> "InteractiveProcess":
        master_fd, slave_fd = os.openpty()
        try:
            process = subprocess.Popen(
                argv,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                start_new_session=True,
                close_fds=True,
            )
        except Exception:
            os.close(master_fd)
            os.close(slave_fd)
            raise
        try:
            os.close(slave_fd)
        except OSError:
            pass
        return cls(process=process, master_fd=master_fd, windows_pty=False)

    @classmethod
    def _spawn_windows(cls, argv: list[str]) -> "InteractiveProcess":
        try:
            from winpty import PtyProcess  # type: ignore
        except ImportError as exc:
            raise InteractivePtyError(
                "Windows interactive PTY support requires pywinpty. "
                "Install it with: python -m pip install pywinpty>=3.0.5"
            ) from exc

        env = dict(os.environ)
        env.setdefault("TERM", "xterm-256color")
        try:
            process = PtyProcess.spawn(
                list(argv),
                env=env,
                dimensions=(40, 120),
            )
        except Exception as exc:
            raise InteractivePtyError(f"Failed to create Windows ConPTY: {exc}") from exc
        return cls(process=process, windows_pty=True)

    def poll(self) -> int | None:
        if self.windows_pty:
            process = self.process
            try:
                if process.isalive():
                    return None
            except Exception:
                return None
            try:
                return int(process.exitstatus)
            except (AttributeError, TypeError, ValueError):
                return 1
        return self.process.poll()

    @property
    def returncode(self) -> int | None:
        return self.poll()

    def read_available(self, timeout: float = 0.1, max_bytes: int = 4096) -> bytes:
        if self._closed:
            return b""
        if self.windows_pty:
            fileobj = getattr(self.process, "fileobj", None)
            if fileobj is None:
                return b""
            try:
                ready, _, _ = select.select([fileobj], [], [], max(0.0, timeout))
            except (OSError, ValueError):
                return b""
            if not ready:
                return b""
            try:
                data = fileobj.recv(max_bytes)
            except (OSError, ValueError):
                return b""
            if not data:
                return b""
            # pywinpty internally uses this marker while idle.
            if data == b"0011Ignore":
                return b""
            return data

        assert self.master_fd is not None
        try:
            ready, _, _ = select.select([self.master_fd], [], [], max(0.0, timeout))
        except (OSError, ValueError):
            return b""
        if not ready:
            return b""
        try:
            return os.read(self.master_fd, max_bytes)
        except OSError as exc:
            if exc.errno in {errno.EIO, errno.EBADF}:
                return b""
            raise

    def write(self, data: bytes) -> int:
        if self._closed:
            raise OSError(errno.EBADF, "PTY is closed")
        if not data:
            return 0
        if self.windows_pty:
            try:
                # pywinpty's public API accepts text, not bytes.
                written = self.process.write(data.decode("utf-8", errors="replace"))
            except Exception as exc:
                raise OSError(errno.EIO, str(exc)) from exc
            if isinstance(written, int):
                return written
            return len(data)

        assert self.master_fd is not None
        return os.write(self.master_fd, data)

    def terminate(self, *, force: bool = False) -> bool:
        if self.poll() is not None:
            return False
        if self.windows_pty:
            try:
                result = self.process.terminate(force=force)
                if result:
                    return False
            except Exception:
                pass
            if force and self.poll() is None:
                try:
                    self.process.close(force=True)
                except Exception:
                    pass
                return True
            return self.poll() is None

        process = self.process
        try:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True

    def close_io(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.windows_pty:
            try:
                self.process.close(force=False)
            except Exception:
                try:
                    self.process.close(force=True)
                except Exception:
                    pass
            return
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass

    def terminate_and_close(self, *, force: bool = False) -> None:
        if self.poll() is None:
            self.terminate(force=force)
            if self.poll() is None and not force:
                self.terminate(force=True)
        self.close_io()


def popen_process_group_kwargs() -> dict[str, Any]:
    """Return Popen kwargs that isolate a child process on the current OS."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def terminate_process_tree(process: subprocess.Popen[Any], *, grace_period: float = 0.5) -> bool:
    """Terminate a regular subprocess cross-platform; return whether force-kill was required."""
    if process.poll() is not None:
        return False
    if os.name == "nt":
        try:
            process.terminate()
        except ProcessLookupError:
            return False
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
    deadline = time.monotonic() + max(0.0, grace_period)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        time.sleep(0.05)
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        return True
    return False
