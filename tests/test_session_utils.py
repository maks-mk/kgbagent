import types
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.session_utils import repair_session_if_needed
from core.config import AgentConfig
from core.nodes import AgentNodes


class _FakeAgentApp:
    def __init__(self, messages):
        self.values = {"messages": list(messages)}
        self.update_calls = []

    async def aget_state(self, _config):
        return types.SimpleNamespace(values=self.values)

    async def aupdate_state(self, _config, update, as_node=None):
        self.update_calls.append({"update": update, "as_node": as_node})
        self.values.setdefault("messages", []).extend(update.get("messages", []))


class SessionRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_repair_inserts_interrupted_tool_message_for_missing_tool_output(self):
        app = _FakeAgentApp(
            [
                HumanMessage(content="Сделай шаг"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc-1", "name": "cli_exec", "args": {"command": "echo 1"}}],
                ),
            ]
        )

        notices = await repair_session_if_needed(app, "thread-a")

        self.assertEqual(len(app.update_calls), 1)
        self.assertGreaterEqual(len(notices), 2)
        tool_messages = [message for message in app.values["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].tool_call_id, "tc-1")
        self.assertIn("ERROR[NETWORK]", str(tool_messages[0].content))
        self.assertIn("not a user stop", str(tool_messages[0].content))
        self.assertEqual(tool_messages[0].additional_kwargs["tool_args"], {"command": "echo 1"})
        self.assertIs(tool_messages[0].additional_kwargs["agent_internal"]["visible_in_ui"], False)
        self.assertIn("History was repaired", tool_messages[0].additional_kwargs["agent_internal"]["ui_notice"])
        self.assertEqual(tool_messages[0].status, "error")

    async def test_repair_logs_structured_event_for_inserted_tool_message(self):
        app = _FakeAgentApp(
            [
                HumanMessage(content="Сделай шаг"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc-log", "name": "cli_exec", "args": {"command": "echo 1"}}],
                ),
            ]
        )
        logged_events = []

        notices = await repair_session_if_needed(
            app,
            "thread-log",
            event_logger=lambda event_type, payload: logged_events.append((event_type, payload)),
        )

        self.assertGreaterEqual(len(notices), 2)
        self.assertEqual(len(logged_events), 1)
        self.assertEqual(logged_events[0][0], "tool_repair_inserted")
        self.assertEqual(logged_events[0][1]["tool_call_id"], "tc-log")
        self.assertEqual(logged_events[0][1]["tool_args"], {"command": "echo 1"})
        self.assertEqual(logged_events[0][1]["reason"], "missing_tool_output_after_stream_interruption")

    async def test_repair_skips_when_loop_budget_handoff_exists_after_pending_tool_call(self):
        app = _FakeAgentApp(
            [
                HumanMessage(content="Сделай шаг"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc-loop", "name": "cli_exec", "args": {"command": "echo 1"}}],
                    id="ai-loop",
                ),
                AIMessage(
                    content="Остановился по лимиту шагов.",
                    additional_kwargs={"agent_internal": {"kind": "loop_budget_handoff", "turn_id": 1}},
                ),
            ]
        )

        notices = await repair_session_if_needed(app, "thread-b")

        self.assertEqual(notices, [])
        self.assertEqual(app.update_calls, [])
        self.assertFalse(any(isinstance(message, ToolMessage) for message in app.values["messages"]))

    async def test_repair_inserts_all_missing_tool_outputs_in_unresolved_tail(self):
        app = _FakeAgentApp(
            [
                HumanMessage(content="Сделай шаг"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc-1", "name": "read_file", "args": {"path": "a.txt"}}],
                ),
                ToolMessage(content="ok", tool_call_id="tc-1", name="read_file"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc-2", "name": "read_file", "args": {"path": "b.txt"}}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc-3", "name": "read_file", "args": {"path": "c.txt"}}],
                ),
            ]
        )

        notices = await repair_session_if_needed(app, "thread-multi")

        self.assertGreaterEqual(len(notices), 2)
        tool_messages = [message for message in app.values["messages"] if isinstance(message, ToolMessage)]
        repaired_ids = [message.tool_call_id for message in tool_messages if "ERROR[NETWORK]" in str(message.content)]
        self.assertEqual(repaired_ids, ["tc-2", "tc-3"])

    async def test_repair_preserves_tool_call_identity_for_next_provider_sanitization(self):
        long_tool_id = "chatcmpl-tool-9d8dc0bbe38d6202"
        app = _FakeAgentApp(
            [
                HumanMessage(content="Сделай шаг"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": long_tool_id, "name": "read_file", "args": {"path": "README.md"}}],
                ),
            ]
        )

        await repair_session_if_needed(app, "thread-remap")

        config = AgentConfig(
            PROVIDER="openai",
            OPENAI_API_KEY="test-key",
            PROMPT_PATH=Path(__file__).resolve().parents[1] / "prompt.txt",
            # Hermetic: chat mode remaps non-compliant tool call IDs; a local
            # .env selecting LLM_API_MODE=responses must not change that.
            LLM_API_MODE="chat",
        )
        nodes = AgentNodes(config=config, llm=types.SimpleNamespace(), tools=[], llm_with_tools=types.SimpleNamespace())
        sanitized = nodes._sanitize_messages_for_model(app.values["messages"])

        ai_message = next(message for message in sanitized if isinstance(message, AIMessage))
        tool_message = next(message for message in sanitized if isinstance(message, ToolMessage))
        self.assertEqual(ai_message.tool_calls[0]["id"], tool_message.tool_call_id)
        self.assertRegex(tool_message.tool_call_id, r"^[A-Za-z0-9]{9}$")


if __name__ == "__main__":
    unittest.main()
