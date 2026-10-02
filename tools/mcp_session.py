"""Persistent MCP sessions with task-local ownership of AnyIO context managers."""

import asyncio
import logging
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class PersistentMCPClient:
    """Keep a stdio server alive until registry cleanup, even between agent runs.

    MCP/AnyIO contexts must be entered and exited in the same task. Loading,
    invoking tools and cleaning up the registry can all happen in different tasks.
    """

    def __init__(self, client: Any, server_name: str):
        self._client = client
        self._server_name = server_name
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._tools: list[BaseTool] | None = None
        self._error: Exception | None = None

    async def _run(self) -> None:
        try:
            from langchain_mcp_adapters.tools import load_mcp_tools

            async with self._client.session(self._server_name) as session:
                self._tools = await load_mcp_tools(
                    session,
                    server_name=self._server_name,
                    callbacks=self._client.callbacks,
                    tool_interceptors=self._client.tool_interceptors,
                    tool_name_prefix=self._client.tool_name_prefix,
                    handle_tool_errors=self._client.handle_tool_errors,
                )
                self._ready.set()
                await self._stop.wait()
        except Exception as exc:
            self._error = exc
            if self._ready.is_set():
                logger.exception("MCP session '%s' failed", self._server_name)
        finally:
            # Also wake the loader if startup failed, was cancelled, or an SDK
            # context manager suppressed an exception before publishing tools.
            self._ready.set()

    async def get_tools(self) -> list[BaseTool]:
        if self._stop.is_set():
            raise RuntimeError(f"MCP session '{self._server_name}' is closed")
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"mcp-session:{self._server_name}")
        try:
            await self._ready.wait()
            if self._error is not None:
                raise self._error
            if self._tools is None or self._task.done():
                raise RuntimeError(f"MCP session '{self._server_name}' ended during startup")
            return self._tools
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        self._stop.set()
        if self._task is not None:
            if not self._ready.is_set():
                # Initialization may be waiting for a server that never replies.
                self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
