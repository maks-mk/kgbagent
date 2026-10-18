import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import aiosqlite

from core.checkpointing import compact_checkpoint_database, create_checkpoint_runtime
from core.config import AgentConfig


class CompactCheckpointDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        tempdir = tempfile.TemporaryDirectory(
            prefix=".checkpoint-compact-tests-", dir=Path(__file__).resolve().parents[1]
        )
        self.addCleanup(tempdir.cleanup)
        self.db_path = Path(tempdir.name) / "checkpoints.sqlite"

    async def _create_bloated_db(self) -> None:
        def _setup() -> None:
            import sqlite3

            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute("CREATE TABLE data (id INTEGER PRIMARY KEY, blob TEXT)")
                payload = "x" * 4096
                conn.executemany("INSERT INTO data (blob) VALUES (?)", [(payload,)] * 400)
                conn.commit()
                conn.execute("DELETE FROM data")
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(_setup)

    async def test_compact_missing_file_returns_none(self):
        result = await compact_checkpoint_database(self.db_path)
        self.assertIsNone(result)

    async def test_compact_returns_sizes_and_shrinks_database(self):
        await self._create_bloated_db()
        size_before_expected = self.db_path.stat().st_size
        result = await compact_checkpoint_database(self.db_path)
        self.assertIsNotNone(result)
        size_before, size_after = result
        self.assertGreater(size_before, 0)
        self.assertLess(size_after, size_before)
        # Database stays valid after VACUUM.
        conn = await aiosqlite.connect(str(self.db_path))
        try:
            async with conn.execute("PRAGMA integrity_check") as cursor:
                row = await cursor.fetchone()
            self.assertEqual(row[0], "ok")
            async with conn.execute("SELECT COUNT(*) FROM data") as cursor:
                count = await cursor.fetchone()
            self.assertEqual(count[0], 0)
        finally:
            await conn.close()
        self.assertEqual(size_after, self.db_path.stat().st_size)


class CheckpointRuntimeCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        tempdir = tempfile.TemporaryDirectory(
            prefix=".checkpoint-tests-", dir=Path(__file__).resolve().parents[1]
        )
        self.addCleanup(tempdir.cleanup)
        self.db_path = Path(tempdir.name) / "checkpoints.sqlite"
        with mock.patch.object(
            AgentConfig, "settings_customise_sources",
            side_effect=lambda *args, **kwargs: (kwargs["init_settings"],),
        ):
            self.config = AgentConfig(
                PROVIDER="openai", OPENAI_BASE_URL="http://localhost",
                CHECKPOINT_BACKEND="sqlite", CHECKPOINT_SQLITE_PATH=self.db_path,
            )
        self.conn = await aiosqlite.connect(str(self.db_path))
        self.addAsyncCleanup(self.conn.close)
        self.close = mock.AsyncMock(wraps=self.conn.close)
        close_patch = mock.patch.object(self.conn, "close", self.close)
        close_patch.start()
        self.addCleanup(close_patch.stop)
        connect_patch = mock.patch("aiosqlite.connect", mock.AsyncMock(return_value=self.conn))
        connect_patch.start()
        self.addCleanup(connect_patch.stop)

    async def _assert_closed(self):
        self.close.assert_awaited_once_with()
        with self.assertRaises(ValueError):
            await self.conn.execute("SELECT 1")

    async def test_setup_failure_closes_connection_and_preserves_error(self):
        failure = RuntimeError("setup failed")
        with mock.patch("langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver.setup", side_effect=failure):
            with self.assertRaises(RuntimeError) as caught:
                await create_checkpoint_runtime(self.config)
        self.assertIs(caught.exception, failure)
        await self._assert_closed()

    async def test_cancellation_during_setup_closes_connection(self):
        entered = asyncio.Event()

        async def blocked_setup():
            entered.set()
            await asyncio.Future()

        with mock.patch("langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver.setup", side_effect=blocked_setup):
            task = asyncio.create_task(create_checkpoint_runtime(self.config))
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await self._assert_closed()
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_constructor_failure_closes_connection_and_preserves_error(self):
        failure = RuntimeError("constructor failed")
        with mock.patch("langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver", side_effect=failure):
            with self.assertRaises(RuntimeError) as caught:
                await create_checkpoint_runtime(self.config)
        self.assertIs(caught.exception, failure)
        await self._assert_closed()

    async def test_cleanup_failure_does_not_replace_initialization_error(self):
        for stage in ("AsyncSqliteSaver", "AsyncSqliteSaver.setup"):
            for failure in (RuntimeError("initialization failed"), asyncio.CancelledError()):
                with self.subTest(stage=stage, error=type(failure).__name__):
                    self.close.reset_mock()
                    self.close.side_effect = RuntimeError("close failed")
                    with (
                        mock.patch(f"langgraph.checkpoint.sqlite.aio.{stage}", side_effect=failure),
                        self.assertLogs("agent", level="WARNING") as logs,
                    ):
                        with self.assertRaises(type(failure)) as caught:
                            await create_checkpoint_runtime(self.config)
                    self.assertIs(caught.exception, failure)
                    self.close.assert_awaited_once_with()
                    self.assertIn("close failed", "\n".join(logs.output))

    async def test_success_transfers_connection_ownership_to_runtime(self):
        runtime = await create_checkpoint_runtime(self.config)
        self.close.assert_not_awaited()
        self.assertEqual(runtime.resolved_backend, "sqlite")
        async with self.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'") as cursor:
            tables = {row[0] for row in await cursor.fetchall()}
        self.assertIn("checkpoints", tables)
        await runtime.aclose()
        await self._assert_closed()
