"""Tests for mixed-mode parallel/sequential tool batch execution (Level 1).

Verifies that ``ToolBatchCoordinator`` correctly partitions tool calls into
parallel-safe and sequential groups, runs them concurrently vs. one-by-one,
and re-assembles ``ToolMessage`` results in the original ``tool_calls`` order.
Shell calls are explicitly parallel-safe even though their metadata is mutating.
"""

import asyncio
import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from pydantic import ValidationError

from core.config import AgentConfig
from core.node_orchestrators import (
    AgentTurnOrchestrator,
    AgentTurnOwner,
    RecoveryTurnOrchestrator,
    RecoveryTurnOwner,
    ToolBatchCoordinator,
    ToolBatchOwner,
)
from core.nodes import AgentNodes
from core.tool_policy import ToolMetadata


def _tc(name: str, tc_id: str, args: dict | None = None) -> dict[str, Any]:
    return {"name": name, "id": tc_id, "args": args or {}}


def _make_tool_metadata(name: str, *, read_only: bool = True) -> ToolMetadata:
    return ToolMetadata(
        name=name,
        read_only=read_only,
        mutating=not read_only,
        destructive=False,
        requires_approval=False,
        networked=False,
        source="local",
    )


class _FakeOwner:
    """Minimal stand-in for ``AgentNodes`` with just the methods the coordinator calls."""

    PARALLEL_SAFE_TOOL_NAMES = AgentNodes.PARALLEL_SAFE_TOOL_NAMES
    READ_ONLY_LOOP_TOLERANT_TOOL_NAMES = AgentNodes.READ_ONLY_LOOP_TOLERANT_TOOL_NAMES

    def __init__(self, tool_metadata: dict[str, ToolMetadata]) -> None:
        self._metadata = tool_metadata
        self.tool_metadata = tool_metadata
        self._all_tool_names = tuple(tool_metadata.keys())
        self.config = SimpleNamespace(
            max_parallel_tool_calls=4,
            model_supports_tools=True,
            effective_tool_loop_window=0,
            effective_tool_loop_limit_readonly=99,
            effective_tool_loop_limit_mutating=99,
        )
        self.recovery_manager = SimpleNamespace(
            reset_after_success=lambda state, current_turn_id, successful_evidence: state,
        )
        self.tool_executor = SimpleNamespace(
            handle_result=self._fake_handle_result,
        )
        self._process_delay = 0.05
        self._call_log: list[str] = []

    # --- Methods called by ToolBatchCoordinator.run ---

    @staticmethod
    def _fake_handle_result(*, tool_name, tool_call_id, content, had_error, **kwargs):
        from core.tool_executor import ToolExecutionOutcome
        from core.tool_results import parse_tool_execution_result

        parsed = parse_tool_execution_result(content)
        return ToolExecutionOutcome(
            tool_message=ToolMessage(content=content, tool_call_id=tool_call_id),
            parsed_result=parsed,
            had_error=had_error,
            issue=None,
            content=content,
        )

    def _log_node_start(self, state, node, **payload):
        return 0.0

    def _log_node_end(self, state, node, started_at, **payload):
        pass

    def _log_node_error(self, state, node, started_at, error, **payload):
        pass

    def _log_run_event(self, state, event, **payload):
        pass

    def _check_invariants(self, state):
        pass

    def _get_last_pending_ai_with_tool_calls(self, messages):
        return messages[-1] if messages else None

    def _current_turn_id(self, state, messages):
        return 1

    def _active_tools_for_turn(self, state, messages):
        return list(self._metadata.keys()), list(self._metadata.keys())

    def _tool_is_read_only(self, tool_name: str) -> bool:
        meta = self._metadata.get(tool_name)
        return bool(meta and meta.read_only and not meta.mutating and not meta.destructive)

    def _tool_call_is_parallel_safe(self, tool_call: dict[str, Any]) -> bool:
        return AgentNodes._tool_call_is_parallel_safe(self, tool_call)

    def _partition_tool_calls(self, tool_calls):
        parallel = []
        sequential = []
        for tc in tool_calls:
            if self._tool_call_is_parallel_safe(tc):
                parallel.append(tc)
            else:
                sequential.append(tc)
        return parallel, sequential

    def _metadata_for_tool(self, tool_name: str) -> ToolMetadata:
        return self._metadata.get(tool_name, _make_tool_metadata(tool_name))

    def _effective_tool_metadata(self, tool_name, tool_args=None):
        return self._metadata_for_tool(tool_name)

    def _tool_is_allowed_for_turn(self, tool_name, allowed_tool_names=None):
        return True

    def _tool_requires_approval(self, tool_name, tool_args):
        return False

    def _tool_call_is_approved(self, tool_call_id, approval_state):
        return True

    def _missing_required_tool_fields(self, tool_name, tool_args):
        return None

    def _merge_open_tool_issues(self, tool_issues, current_turn_id):
        return None

    async def _process_tool_call(self, tool_call, recent_calls, state, approval_state, current_turn_id, allowed_tool_names):
        name = tool_call.get("name")
        self._call_log.append(name)
        await asyncio.sleep(self._process_delay)
        tool_msg = ToolMessage(
            content=f"result:{name}",
            tool_call_id=tool_call.get("id", ""),
        )
        return tool_msg, False, None


class OwnerProtocolContractTests(unittest.TestCase):
    def test_orchestrator_constructors_use_explicit_owner_protocols(self):
        self.assertEqual(
            AgentTurnOrchestrator.__init__.__annotations__["owner"],
            "AgentTurnOwner",
        )
        self.assertEqual(
            RecoveryTurnOrchestrator.__init__.__annotations__["owner"],
            "RecoveryTurnOwner",
        )
        self.assertEqual(
            ToolBatchCoordinator.__init__.__annotations__["owner"],
            "ToolBatchOwner",
        )

    def test_owner_protocols_match_all_orchestrator_owner_accesses(self):
        pairs = (
            (AgentTurnOrchestrator, AgentTurnOwner),
            (RecoveryTurnOrchestrator, RecoveryTurnOwner),
            (ToolBatchCoordinator, ToolBatchOwner),
        )
        for orchestrator, owner_protocol in pairs:
            with self.subTest(orchestrator=orchestrator.__name__):
                tree = ast.parse(textwrap.dedent(inspect.getsource(orchestrator)))
                owner_accesses = {
                    node.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "owner"
                }
                self.assertEqual(owner_accesses, owner_protocol.__protocol_attrs__)


class ParallelToolConfigTests(unittest.TestCase):
    def _config(self, **overrides):
        # Isolate this test from local profiles, environment files and credentials.
        with mock.patch.object(AgentConfig, "settings_customise_sources", side_effect=lambda *args, **kwargs: (kwargs["init_settings"],)):
            return AgentConfig(PROVIDER="openai", OPENAI_BASE_URL="http://localhost", **overrides)

    def test_default_concurrency_limit(self):
        self.assertEqual(self._config().max_parallel_tool_calls, 4)

    def test_concurrency_limit_accepts_positive_integers(self):
        for value in (1, 8, "16"):
            with self.subTest(value=value):
                self.assertEqual(self._config(MAX_PARALLEL_TOOL_CALLS=value).max_parallel_tool_calls, int(value))

    def test_concurrency_limit_rejects_invalid_values(self):
        for value in (0, -1, "invalid", 1.5):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                self._config(MAX_PARALLEL_TOOL_CALLS=value)


class MixedParallelToolsTests(unittest.IsolatedAsyncioTestCase):
    def _make_owner(self, names_read_only: dict[str, bool]) -> _FakeOwner:
        metadata = {
            name: _make_tool_metadata(name, read_only=ro)
            for name, ro in names_read_only.items()
        }
        return _FakeOwner(metadata)

    def _make_state(self, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content="", tool_calls=tool_calls)],
            "run_id": "test-run",
        }

    async def test_all_parallel_safe_uses_gather(self):
        """When every call is parallel-safe, all run concurrently."""
        owner = self._make_owner({"read_file": True, "list_directory": True})
        coordinator = ToolBatchCoordinator(owner)

        tool_calls = [
            _tc("read_file", "tc1"),
            _tc("list_directory", "tc2"),
        ]
        state = self._make_state(tool_calls)

        with mock.patch.object(
            ToolBatchCoordinator, "_parallel_mode_label", return_value="all"
        ) as mock_label:
            result = await coordinator.run(state)

        # Both calls should have been started within the same concurrency window.
        # If sequential, total time would be ~2*delay; with gather it's ~1*delay.
        mock_label.assert_called_once()
        self.assertEqual(len(result["messages"]), 2)
        # Order preserved
        self.assertEqual(result["messages"][0].tool_call_id, "tc1")
        self.assertEqual(result["messages"][1].tool_call_id, "tc2")

    async def test_parallel_results_stream_as_each_call_finishes(self):
        owner = self._make_owner({"read_file": True, "list_directory": True})
        coordinator = ToolBatchCoordinator(owner)
        original_process = owner._process_tool_call

        async def process_with_different_delays(tool_call, *args, **kwargs):
            if tool_call.get("id") == "tc1":
                await asyncio.sleep(0.08)
            return await original_process(tool_call, *args, **kwargs)

        owner._process_tool_call = process_with_different_delays
        streamed = []
        state = self._make_state([
            _tc("read_file", "tc1"),
            _tc("list_directory", "tc2"),
        ])

        with mock.patch("core.node_orchestrators.get_stream_writer", return_value=streamed.append):
            result = await coordinator.run(state)

        self.assertEqual(streamed[0]["type"], "tool_batch_started")
        self.assertEqual(
            [tool_call["id"] for tool_call in streamed[0]["tool_calls"]],
            ["tc1", "tc2"],
        )
        result_events = [event for event in streamed if event["type"] == "tool_result"]
        self.assertEqual(
            [event["message"]["tool_call_id"] for event in result_events],
            ["tc2", "tc1"],
        )
        self.assertEqual(
            [message.tool_call_id for message in result["messages"]],
            ["tc1", "tc2"],
        )

    async def test_cli_exec_calls_run_concurrently_despite_mutating_metadata(self):
        owner = self._make_owner({"cli_exec": False})
        owner._process_delay = 0.15
        coordinator = ToolBatchCoordinator(owner)
        state = self._make_state([
            _tc("cli_exec", "tc1", {"command": "echo one"}),
            _tc("cli_exec", "tc2", {"command": "echo two"}),
        ])

        import time

        start = time.perf_counter()
        result = await coordinator.run(state)
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.27, f"Expected concurrent cli_exec calls, took {elapsed:.3f}s")
        self.assertEqual(
            [message.tool_call_id for message in result["messages"]],
            ["tc1", "tc2"],
        )

    async def test_cli_exec_does_not_make_other_mutating_tools_parallel(self):
        owner = self._make_owner({"cli_exec": False, "write_file": False})
        coordinator = ToolBatchCoordinator(owner)
        state = self._make_state([
            _tc("cli_exec", "tc1", {"command": "echo one"}),
            _tc("write_file", "tc2"),
            _tc("cli_exec", "tc3", {"command": "echo two"}),
        ])

        result = await coordinator.run(state)

        self.assertEqual(owner._call_log, ["cli_exec", "write_file", "cli_exec"])
        self.assertEqual(
            [message.tool_call_id for message in result["messages"]],
            ["tc1", "tc2", "tc3"],
        )

    async def test_all_sequential(self):
        """When no call is parallel-safe, all run sequentially."""
        owner = self._make_owner({"write_file": False, "safe_delete_file": False})
        coordinator = ToolBatchCoordinator(owner)

        tool_calls = [
            _tc("write_file", "tc1"),
            _tc("safe_delete_file", "tc2"),
        ]
        state = self._make_state(tool_calls)

        result = await coordinator.run(state)

        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0].tool_call_id, "tc1")
        self.assertEqual(result["messages"][1].tool_call_id, "tc2")

    async def test_mixed_mode_preserves_order(self):
        """Mixed batch: parallel-safe calls run concurrently, mutating calls sequentially,
        and the final ToolMessage order matches the original tool_calls order."""
        # read_file and list_directory are parallel-safe; write_file is not.
        owner = self._make_owner({
            "read_file": True,
            "write_file": False,
            "list_directory": True,
        })
        coordinator = ToolBatchCoordinator(owner)

        tool_calls = [
            _tc("read_file", "tc1"),
            _tc("write_file", "tc2"),
            _tc("list_directory", "tc3"),
        ]
        state = self._make_state(tool_calls)

        result = await coordinator.run(state)

        self.assertEqual(len(result["messages"]), 3)
        # Original order must be preserved regardless of execution mode.
        self.assertEqual(result["messages"][0].tool_call_id, "tc1")
        self.assertEqual(result["messages"][1].tool_call_id, "tc2")
        self.assertEqual(result["messages"][2].tool_call_id, "tc3")
        # Content matches the tool name.
        self.assertEqual(result["messages"][0].content, "result:read_file")
        self.assertEqual(result["messages"][1].content, "result:write_file")
        self.assertEqual(result["messages"][2].content, "result:list_directory")
        self.assertEqual(owner._call_log, ["read_file", "write_file", "list_directory"])

    async def test_mutating_call_is_a_barrier_for_following_read(self):
        owner = self._make_owner({"write_file": False, "read_file": True})
        coordinator = ToolBatchCoordinator(owner)
        state = self._make_state([
            _tc("write_file", "tc1"),
            _tc("read_file", "tc2"),
        ])

        result = await coordinator.run(state)

        self.assertEqual(owner._call_log, ["write_file", "read_file"])
        self.assertEqual(
            [message.tool_call_id for message in result["messages"]],
            ["tc1", "tc2"],
        )

    async def test_mixed_mode_parallel_runs_concurrently(self):
        """In mixed mode, the parallel-safe calls should overlap in time
        (total time < sum of individual delays)."""
        owner = self._make_owner({
            "read_file": True,
            "list_directory": True,
            "write_file": False,
        })
        owner._process_delay = 0.15
        coordinator = ToolBatchCoordinator(owner)

        tool_calls = [
            _tc("read_file", "tc1"),
            _tc("list_directory", "tc2"),
            _tc("write_file", "tc3"),
        ]
        state = self._make_state(tool_calls)

        import time

        start = time.perf_counter()
        result = await coordinator.run(state)
        elapsed = time.perf_counter() - start

        # 2 parallel (0.15s concurrent) + 1 sequential (0.15s) = ~0.30s
        # If all were sequential: ~0.45s
        self.assertLess(elapsed, 0.42, f"Expected concurrent execution, took {elapsed:.3f}s")
        self.assertEqual(len(result["messages"]), 3)

    async def test_single_tool_call(self):
        """A single tool call should work regardless of its parallel-safety."""
        owner = self._make_owner({"write_file": False})
        coordinator = ToolBatchCoordinator(owner)

        tool_calls = [_tc("write_file", "tc1")]
        state = self._make_state(tool_calls)

        result = await coordinator.run(state)

        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].tool_call_id, "tc1")

    async def test_parallel_exception_does_not_crash_batch(self):
        """If one parallel call raises, the others should still complete."""
        owner = self._make_owner({"read_file": True, "list_directory": True})
        coordinator = ToolBatchCoordinator(owner)

        original_process = owner._process_tool_call

        async def flaky_process(tool_call, *args, **kwargs):
            if tool_call.get("id") == "tc1":
                raise RuntimeError("boom")
            return await original_process(tool_call, *args, **kwargs)

        owner._process_tool_call = flaky_process

        tool_calls = [
            _tc("read_file", "tc1"),
            _tc("list_directory", "tc2"),
        ]
        state = self._make_state(tool_calls)

        result = await coordinator.run(state)

        self.assertEqual(len(result["messages"]), 2)
        # tc2 should have a valid result.
        self.assertEqual(result["messages"][1].content, "result:list_directory")

    async def test_parallel_pool_limits_in_flight_calls_and_refills_slots(self):
        owner = self._make_owner({"read_file": True})
        owner.config.max_parallel_tool_calls = 2
        gates = [asyncio.Event() for _ in range(6)]
        started = [asyncio.Event() for _ in gates]
        active = 0
        peak = 0

        async def process(tool_call, *args):
            nonlocal active, peak
            index = int(tool_call["id"])
            active += 1
            peak = max(peak, active)
            started[index].set()
            try:
                await gates[index].wait()
                return ToolMessage(content=f"result:{index}", tool_call_id=str(index)), False, None
            finally:
                active -= 1

        owner._process_tool_call = process
        state = self._make_state([_tc("read_file", str(i)) for i in range(len(gates))])
        task = asyncio.create_task(ToolBatchCoordinator(owner).run(state))
        try:
            await asyncio.wait_for(started[1].wait(), 2)
            self.assertEqual([event.is_set() for event in started], [True, True, False, False, False, False])
            gates[1].set()
            # A free slot must be reused without waiting for the slow first call.
            await asyncio.wait_for(started[2].wait(), 2)
            self.assertFalse(gates[0].is_set())
            for gate in gates:
                gate.set()
            result = await asyncio.wait_for(task, 2)
            self.assertEqual(peak, 2)
            self.assertEqual([message.tool_call_id for message in result["messages"]], [str(i) for i in range(6)])
        finally:
            task.cancel()
            for gate in gates:
                gate.set()
            await asyncio.gather(task, return_exceptions=True)

    async def test_limit_one_serializes_parallel_safe_calls(self):
        owner = self._make_owner({"read_file": True})
        owner.config.max_parallel_tool_calls = 1
        events = []

        async def process(tool_call, *args):
            events.append(("start", tool_call["id"]))
            await asyncio.sleep(0)
            events.append(("end", tool_call["id"]))
            return ToolMessage(content="ok", tool_call_id=tool_call["id"]), False, None

        owner._process_tool_call = process
        await ToolBatchCoordinator(owner).run(self._make_state([_tc("read_file", "a"), _tc("read_file", "b")]))
        self.assertEqual(events, [("start", "a"), ("end", "a"), ("start", "b"), ("end", "b")])

    async def test_write_barrier_waits_for_reads_and_blocks_following_reads(self):
        owner = self._make_owner({"read_file": True, "write_file": False})
        events = []

        async def process(tool_call, *args):
            events.append(("start", tool_call["id"]))
            await asyncio.sleep(0)
            events.append(("end", tool_call["id"]))
            return ToolMessage(content="ok", tool_call_id=tool_call["id"]), False, None

        owner._process_tool_call = process
        await ToolBatchCoordinator(owner).run(self._make_state([
            _tc("read_file", "a"), _tc("read_file", "b"),
            _tc("write_file", "w"), _tc("read_file", "c"),
        ]))
        self.assertLess(events.index(("end", "a")), events.index(("start", "w")))
        self.assertLess(events.index(("end", "b")), events.index(("start", "w")))
        self.assertLess(events.index(("end", "w")), events.index(("start", "c")))

    async def test_graph_control_and_child_cancellation_stop_and_drain_siblings(self):
        for exception_type in (GraphInterrupt, GraphBubbleUp, asyncio.CancelledError):
            with self.subTest(exception_type=exception_type.__name__):
                owner = self._make_owner({"read_file": True, "write_file": False})
                owner.config.max_parallel_tool_calls = 2
                sibling_started = asyncio.Event()
                sibling_cleaned = asyncio.Event()
                release = asyncio.Event()
                calls = []
                exception = exception_type()

                async def process(tool_call, *args):
                    calls.append(tool_call["id"])
                    if tool_call["id"] == "fail":
                        await sibling_started.wait()
                        raise exception
                    if tool_call["id"] == "slow":
                        sibling_started.set()
                        try:
                            await release.wait()
                        finally:
                            await asyncio.sleep(0)
                            sibling_cleaned.set()
                    return ToolMessage(content="ok", tool_call_id=tool_call["id"]), False, None

                owner._process_tool_call = process
                coordinator = ToolBatchCoordinator(owner)
                state = self._make_state([
                    _tc("read_file", "fail"), _tc("read_file", "slow"),
                    _tc("read_file", "queued"), _tc("write_file", "write"),
                ])
                try:
                    with self.assertRaises(exception_type) as raised:
                        await asyncio.wait_for(coordinator.run(state), 2)
                    self.assertIs(raised.exception, exception)
                    self.assertTrue(sibling_cleaned.is_set(), "Sibling task must finish cleanup before propagation")
                    self.assertEqual(calls, ["fail", "slow"])
                finally:
                    release.set()
                    await asyncio.sleep(0)

    async def test_parent_cancellation_drains_running_tools_without_starting_queued_calls(self):
        owner = self._make_owner({"read_file": True})
        owner.config.max_parallel_tool_calls = 2
        started = asyncio.Event()
        calls = []
        cleaned = []

        async def process(tool_call, *args):
            calls.append(tool_call["id"])
            if len(calls) == 2:
                started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.append(tool_call["id"])

        owner._process_tool_call = process
        task = asyncio.create_task(ToolBatchCoordinator(owner).run(self._make_state([
            _tc("read_file", str(i)) for i in range(5)
        ])))
        try:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(calls, ["0", "1"])
            self.assertCountEqual(cleaned, calls)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_registered_read_only_mcp_calls_overlap(self):
        from tools.tool_registry import ToolRegistry

        owner = _FakeOwner({
            name: ToolRegistry._infer_mcp_metadata(name, server_policy={"read_only": True})
            for name in ("mcp_lookup", "mcp_search")
        })
        both_started = asyncio.Event()
        calls = []

        async def process(tool_call, *args):
            calls.append(tool_call["id"])
            if len(calls) == 2:
                both_started.set()
            await both_started.wait()
            return ToolMessage(content="ok", tool_call_id=tool_call["id"]), False, None

        owner._process_tool_call = process
        result = await asyncio.wait_for(ToolBatchCoordinator(owner).run(self._make_state([
            _tc("mcp_lookup", "a"), _tc("mcp_search", "b"),
        ])), 2)
        self.assertEqual([message.tool_call_id for message in result["messages"]], ["a", "b"])

    def test_unregistered_tools_and_user_input_stay_sequential(self):
        owner = self._make_owner({"request_user_input": True, "custom_read": True})
        self.assertTrue(owner._tool_call_is_parallel_safe(_tc("custom_read", "a")))
        self.assertFalse(owner._tool_call_is_parallel_safe(_tc("request_user_input", "b")))
        self.assertFalse(owner._tool_call_is_parallel_safe(_tc("unknown", "c")))
        owner.tool_metadata.clear()
        self.assertFalse(owner._tool_call_is_parallel_safe(_tc("custom_read", "d")))

    def test_conflicting_metadata_keeps_tools_sequential(self):
        for flag in ("mutating", "destructive"):
            with self.subTest(flag=flag):
                owner = _FakeOwner({"custom_read": ToolMetadata(name="custom_read", read_only=True, **{flag: True})})
                self.assertFalse(owner._tool_call_is_parallel_safe(_tc("custom_read", "a")))

    async def test_execute_tool_propagates_graph_control_exceptions(self):
        from core.nodes.tools import ToolsMixin

        for exception_type in (GraphInterrupt, GraphBubbleUp):
            with self.subTest(exception_type=exception_type.__name__):
                exception = exception_type()
                owner = ToolsMixin()
                owner.tools_map = {"graph_tool": SimpleNamespace(ainvoke=mock.AsyncMock(side_effect=exception))}
                owner._log_run_event = mock.Mock()
                with self.assertRaises(exception_type) as raised:
                    await owner._execute_tool("graph_tool", {})
                self.assertIs(raised.exception, exception)

    async def test_parallel_error_does_not_skip_queued_calls_or_duplicate_results(self):
        owner = self._make_owner({"read_file": True})
        owner.config.max_parallel_tool_calls = 2
        streamed = []

        async def process(tool_call, *args):
            if tool_call["id"] == "bad":
                raise RuntimeError("boom")
            await asyncio.sleep(0)
            return ToolMessage(content="ok", tool_call_id=tool_call["id"]), False, None

        owner._process_tool_call = process
        state = self._make_state([_tc("read_file", tc_id) for tc_id in ("a", "bad", "b", "c")])
        with mock.patch("core.node_orchestrators.get_stream_writer", return_value=streamed.append):
            result = await ToolBatchCoordinator(owner).run(state)
        self.assertEqual([message.tool_call_id for message in result["messages"]], ["a", "bad", "b", "c"])
        self.assertIn("boom", result["messages"][1].content)
        self.assertEqual(len([event for event in streamed if event["type"] == "tool_result"]), 4)

    async def test_result_positions_do_not_depend_on_tool_call_object_identity(self):
        owner = self._make_owner({"read_file": True})
        shared_call = _tc("read_file", "shared")
        # Bypass AIMessage's copying of tool-call dictionaries to exercise aliasing.
        state = {"messages": [SimpleNamespace(tool_calls=[shared_call, shared_call])], "run_id": "test"}
        result = await ToolBatchCoordinator(owner).run(state)
        self.assertEqual([message.tool_call_id for message in result["messages"]], ["shared", "shared"])

    async def test_tool_tasks_inherit_context_without_leaking_changes_to_next_calls(self):
        from contextvars import ContextVar

        context = ContextVar("tool_test_context", default="outside")
        owner = self._make_owner({"read_file": True})
        owner.config.max_parallel_tool_calls = 1
        seen = []

        async def process(tool_call, *args):
            seen.append(context.get())
            context.set(tool_call["id"])
            await asyncio.sleep(0)
            self.assertEqual(context.get(), tool_call["id"])
            return ToolMessage(content="ok", tool_call_id=tool_call["id"]), False, None

        owner._process_tool_call = process
        token = context.set("batch")
        try:
            await ToolBatchCoordinator(owner).run(self._make_state([_tc("read_file", "a"), _tc("read_file", "b")]))
            self.assertEqual(seen, ["batch", "batch"])
            self.assertEqual(context.get(), "batch")
        finally:
            context.reset(token)

    # --- _partition_tool_calls unit tests ---

    def test_partition_all_parallel(self):
        owner = self._make_owner({"read_file": True, "list_directory": True})
        tool_calls = [_tc("read_file", "tc1"), _tc("list_directory", "tc2")]
        parallel, sequential = owner._partition_tool_calls(tool_calls)  # type: ignore[attr-defined]
        self.assertEqual(len(parallel), 2)
        self.assertEqual(len(sequential), 0)

    def test_partition_all_sequential(self):
        owner = self._make_owner({"write_file": False, "safe_delete_file": False})
        tool_calls = [_tc("write_file", "tc1"), _tc("safe_delete_file", "tc2")]
        parallel, sequential = owner._partition_tool_calls(tool_calls)  # type: ignore[attr-defined]
        self.assertEqual(len(parallel), 0)
        self.assertEqual(len(sequential), 2)

    def test_partition_mixed(self):
        owner = self._make_owner({"read_file": True, "write_file": False, "list_directory": True})
        tool_calls = [
            _tc("read_file", "tc1"),
            _tc("write_file", "tc2"),
            _tc("list_directory", "tc3"),
        ]
        parallel, sequential = owner._partition_tool_calls(tool_calls)  # type: ignore[attr-defined]
        self.assertEqual(len(parallel), 2)
        self.assertEqual(len(sequential), 1)
        self.assertEqual(parallel[0]["name"], "read_file")
        self.assertEqual(parallel[1]["name"], "list_directory")
        self.assertEqual(sequential[0]["name"], "write_file")

    # --- _parallel_mode_label unit tests ---

    def test_parallel_mode_label_all(self):
        self.assertEqual(
            ToolBatchCoordinator._parallel_mode_label([{"name": "a"}], []),
            "all",
        )

    def test_parallel_mode_label_mixed(self):
        self.assertEqual(
            ToolBatchCoordinator._parallel_mode_label([{"name": "a"}], [{"name": "b"}]),
            "mixed",
        )

    def test_parallel_mode_label_sequential(self):
        self.assertEqual(
            ToolBatchCoordinator._parallel_mode_label([], [{"name": "b"}]),
            "sequential",
        )


if __name__ == "__main__":
    unittest.main()
