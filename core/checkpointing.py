import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from langgraph.checkpoint.memory import MemorySaver

from core.config import AgentConfig

logger = logging.getLogger("agent")


def _ensure_connection_is_alive_compat(conn: Any) -> Any:
    if hasattr(conn, "is_alive"):
        return conn

    def is_alive() -> bool:
        thread = getattr(conn, "_thread", None)
        if thread is not None:
            return bool(thread.is_alive())
        return bool(getattr(conn, "_connection", None) is not None)

    setattr(conn, "is_alive", is_alive)
    return conn


async def _close_after_failed_init(close_callback: Callable[[], Any]) -> None:
    try:
        result = close_callback()
        if inspect.isawaitable(result):
            await result
    except BaseException:
        # Cleanup must not replace the initialization error, including cancellation.
        logger.warning("Failed to close SQLite checkpointer after initialization failure.", exc_info=True)


async def _create_async_sqlite_checkpointer(db_path: Path) -> tuple[Any, Callable[[], Any]]:
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    conn = await aiosqlite.connect(str(db_path))
    try:
        checkpointer = AsyncSqliteSaver(_ensure_connection_is_alive_compat(conn))
        return checkpointer, conn.close
    except BaseException:
        await _close_after_failed_init(conn.close)
        raise


@dataclass
class CheckpointRuntime:
    backend: str
    resolved_backend: str
    target: str
    checkpointer: Any
    warnings: list[str] = field(default_factory=list)
    close_callback: Optional[Callable[[], Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "resolved_backend": self.resolved_backend,
            "target": self.target,
            "warnings": list(self.warnings),
        }

    async def aclose(self) -> None:
        if self.close_callback is None:
            return
        result = self.close_callback()
        if inspect.isawaitable(result):
            await result


def format_bytes(size: int) -> str:
    """Human-readable byte size for checkpoint database messages."""
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


async def compact_checkpoint_database(db_path: Path | str) -> tuple[int, int] | None:
    """Run VACUUM on the checkpoint database from a dedicated connection.

    VACUUM is a write operation that fails when another connection holds a
    conflicting lock, so it runs with a busy timeout instead of blocking the
    caller indefinitely. Returns ``(size_before, size_after)`` in bytes, or
    ``None`` when the database file does not exist.
    """
    import aiosqlite

    path = Path(db_path)
    if not path.exists():
        return None

    size_before = path.stat().st_size
    conn = await aiosqlite.connect(str(path))
    try:
        cursor = await conn.execute("PRAGMA busy_timeout = 5000")
        await cursor.close()
        cursor = await conn.execute("VACUUM")
        await cursor.close()
    finally:
        await conn.close()

    size_after = path.stat().st_size
    return size_before, size_after


async def _maybe_setup(checkpointer: Any) -> None:
    setup = getattr(checkpointer, "setup", None)
    if not callable(setup):
        return
    result = setup()
    if inspect.isawaitable(result):
        await result


async def create_checkpoint_runtime(config: AgentConfig) -> CheckpointRuntime:
    backend = config.checkpoint_backend

    if backend == "memory":
        return CheckpointRuntime(
            backend=backend,
            resolved_backend="memory",
            target="in-memory",
            checkpointer=MemorySaver(),
        )

    if backend == "sqlite":
        try:
            import aiosqlite  # noqa: F401
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: F401
        except ImportError:
            warning = (
                "SQLite checkpointer is unavailable because 'langgraph-checkpoint-sqlite' is not installed. "
                "Falling back to MemorySaver."
            )
            logger.warning(warning)
            return CheckpointRuntime(
                backend=backend,
                resolved_backend="memory",
                target="in-memory",
                checkpointer=MemorySaver(),
                warnings=[warning],
            )

        db_path = Path(config.checkpoint_sqlite_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        checkpointer, close_callback = await _create_async_sqlite_checkpointer(db_path)
        try:
            await _maybe_setup(checkpointer)
            return CheckpointRuntime(
                backend=backend,
                resolved_backend="sqlite",
                target=str(db_path),
                checkpointer=checkpointer,
                close_callback=close_callback,
            )
        except BaseException:
            await _close_after_failed_init(close_callback)
            raise

    raise ValueError(f"Unsupported checkpoint backend: {backend}")
