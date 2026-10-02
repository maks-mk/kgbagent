"""Local summary budgeting regressions; no provider calls."""
import unittest
from unittest.mock import AsyncMock, Mock, patch

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from core.config import AgentConfig
from core.nodes import AgentNodes
from core.state import AgentState
from core import summarize_policy as policy
from ui.runtime_payloads import build_summary_progress_payload
from ui.streaming import StreamProcessor


class TokenEstimateTests(unittest.TestCase):
    def setUp(self):
        policy._MESSAGE_TOKEN_CACHE.clear()

    def test_special_token_spellings_are_counted_as_text(self):
        text = "log: <|endoftext|> " * 100
        encoder = tiktoken.get_encoding("cl100k_base")
        self.assertEqual(policy.estimate_text_tokens(text), len(encoder.encode_ordinary(text)))
        self.assertGreater(policy.estimate_text_tokens(text), 100)

    def test_encoder_failure_and_unavailable_library_preserve_nonzero_estimate(self):
        broken = Mock()
        broken.encode_ordinary.side_effect = ValueError("bad text")
        for encoder in (broken, None):
            with self.subTest(encoder=encoder), patch.object(policy, "_get_encoder", return_value=encoder):
                self.assertEqual(policy.estimate_text_tokens("1234567"), 3)
                self.assertEqual(policy.estimate_tokens([HumanMessage(content="1234567")]), 7)

    def test_model_switch_does_not_reuse_other_encoding_cache(self):
        message = HumanMessage(content="Привет, как устроена конфигурация? " * 20, id="same")
        estimates = []
        for model in ("gpt-4", "gpt-4o", "gpt-4"):
            expected = len(tiktoken.encoding_for_model(model).encode_ordinary(message.content)) + 4
            result = policy.estimate_tokens([message], model_name=model)
            self.assertEqual(result, expected)
            estimates.append(result)
        self.assertNotEqual(estimates[0], estimates[1])
        message.content = "changed"
        self.assertLess(policy.estimate_tokens([message], model_name="gpt-4"), estimates[0])

    def test_unknown_model_uses_fallback_encoding(self):
        self.assertEqual(policy._get_encoder("unknown-local-model").name, "cl100k_base")

    def test_model_encoding_reaches_trigger_and_boundary(self):
        messages = [HumanMessage(content="старое" * 100), AIMessage(content="done"), HumanMessage(content="новое" * 100)]
        with patch.object(policy, "estimate_context_tokens", return_value=2000) as estimate:
            self.assertTrue(policy.should_summarize(messages, threshold=100, keep_last=1, model_name="gpt-4o"))
        self.assertGreaterEqual(estimate.call_count, 2)
        self.assertTrue(all(call.kwargs["model_name"] == "gpt-4o" for call in estimate.call_args_list))

    def test_memory_truncation_uses_selected_model(self):
        result = policy.truncate_summary_to_token_budget("- важный факт " * 300, 100, model_name="gpt-4o")
        self.assertLessEqual(policy.estimate_summary_tokens(result, model_name="gpt-4o"), 100)


class RuntimeOverheadTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_and_active_schemas_change_budget_and_ui(self):
        config = AgentConfig(
            _env_file=None, provider="openai", openai_api_key="test-key", openai_model="gpt-4o",
            summary_threshold=1_000_000, summary_reserved_tokens=25,
        )
        llm = Mock()
        active = []
        nodes = AgentNodes(config=config, llm=llm, tools=[], active_tools_provider=lambda: active)
        state = {"messages": [HumanMessage(content="hello")], "summary": "remember me", "steps": 0}
        first = await nodes.summarize_node(state)
        overhead = first["summary_context_overhead_tokens"]
        self.assertGreater(overhead, 0)

        def inspect_path(path: str) -> str:
            return path

        active.append(StructuredTool.from_function(inspect_path, description="Detailed inspection instructions. " * 300))
        with_tool = await nodes.summarize_node(state)
        self.assertGreater(with_tool["summary_context_overhead_tokens"], overhead + 500)
        # Descriptions are counted afresh, even when the tool name is unchanged.
        active[0].description += "More configuration details. " * 300
        expanded = await nodes.summarize_node(state)
        self.assertGreater(expanded["summary_context_overhead_tokens"], with_tool["summary_context_overhead_tokens"])
        config.model_supports_tools = False
        disabled = await nodes.summarize_node(state)
        self.assertEqual(disabled["summary_context_overhead_tokens"], overhead)

        payload = build_summary_progress_payload(config, {**state, **first})
        expected = nodes._effective_reserved_tokens(state["summary"], overhead)
        expected += policy.estimate_tokens(state["messages"], model_name="gpt-4o")
        self.assertEqual(payload["estimated_tokens"], expected)
        self.assertEqual(payload["reserved_tokens"], overhead + 25)
        llm.ainvoke.assert_not_called()

    async def test_larger_system_prompt_can_trigger_compaction(self):
        config = AgentConfig(
            _env_file=None, provider="openai", openai_api_key="test-key", openai_model="gpt-4o",
            summary_threshold=1_000_000, summary_reserved_tokens=0, summary_keep_last=1,
            model_supports_tools=False,
        )
        nodes = AgentNodes(config=config, llm=Mock(), tools=[])
        state = {"messages": [HumanMessage(content="old"), AIMessage(content="answer"), HumanMessage(content="new")], "steps": 0}
        first = await nodes.summarize_node(state)
        with patch.object(nodes.context_builder, "_build_base_system_message", return_value=SystemMessage(content="large instruction " * 10000)):
            larger = await nodes.summarize_node(state)
        self.assertGreater(larger["summary_context_overhead_tokens"], first["summary_context_overhead_tokens"])
        threshold = first["summary_context_overhead_tokens"] + 1000
        config.summary_threshold = threshold
        self.assertFalse(build_summary_progress_payload(config, {**state, **first})["will_summarize"])
        self.assertTrue(build_summary_progress_payload(config, {**state, **larger})["will_summarize"])


class SummaryProgressTests(unittest.IsolatedAsyncioTestCase):
    def _make_nodes(self, **kwargs):
        config = AgentConfig(
            _env_file=None, provider="openai", openai_api_key="test-key", openai_model="gpt-4o",
            summary_reserved_tokens=25, summary_keep_last=1, summary_max_tokens=1000,
            model_supports_tools=False, **kwargs,
        )
        llm = Mock(ainvoke=AsyncMock(return_value=AIMessage(content="- Compressed memory.")))
        return AgentNodes(config=config, llm=llm, tools=[])

    async def test_progress_tracks_user_model_and_prompt_changes(self):
        nodes = self._make_nodes(summary_threshold=1_000_000)
        state = {"messages": [HumanMessage(content="hello", id="user-1")], "steps": 0}
        events = []
        with patch("core.nodes.summarize.get_stream_writer", return_value=events.append):
            for message in (None, HumanMessage(content="details " * 500, id="user-2"),
                            AIMessage(content="answer " * 500, id="assistant-1")):
                if message is not None:
                    state["messages"].append(message)
                update = await nodes.summarize_node(state)
                expected = build_summary_progress_payload(nodes.config, {**state, **update})
                expected.pop("provider_input_tokens")
                self.assertEqual(events[-1], {"type": "summary_progress", **expected})
            with patch.object(nodes.context_builder, "_build_base_system_message",
                              return_value=SystemMessage(content="large instructions " * 10000)):
                await nodes.summarize_node(state)
        self.assertEqual(len(events), 4)
        for previous, current in zip(events, events[1:]):
            self.assertGreater(current["estimated_tokens"], previous["estimated_tokens"])
            self.assertLess(current["progress"], previous["progress"])
        nodes.llm.ainvoke.assert_not_called()

    async def test_graph_stream_fills_then_resets_progress_to_committed_state(self):
        nodes = self._make_nodes(summary_threshold=10000)
        workflow = StateGraph(AgentState)
        workflow.add_node("summarize", nodes.summarize_node)
        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", END)
        app = workflow.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "summary-progress"}}
        state = {
            "messages": [HumanMessage(content="old detail " * 18000, id="old-user"),
                         AIMessage(content="old response", id="old-assistant"),
                         HumanMessage(content="continue", id="current-user")],
            "steps": 0,
        }
        events = []
        processor = StreamProcessor(events.append)
        result = await processor.process_stream(
            app.astream(state, config=config, stream_mode=["custom", "updates"], version="v2")
        )
        self.assertFalse(result.failed)
        progress = [event.payload for event in events if event.type == "summary_progress"]
        self.assertEqual(len(progress), 2)
        self.assertTrue(progress[0]["will_summarize"])
        self.assertEqual(progress[0]["progress"], 0.0)
        self.assertFalse(progress[1]["will_summarize"])
        self.assertGreater(progress[1]["progress"], 0.9)
        self.assertEqual(progress[1]["message_count"], 1)
        self.assertTrue(progress[1]["has_summary"])
        snapshot = await app.aget_state(config)
        expected = build_summary_progress_payload(nodes.config, snapshot.values)
        expected.pop("provider_input_tokens")
        self.assertEqual(progress[1], expected)
        relevant = [event.type for event in events if event.type in {"summary_progress", "summary_notice"}]
        self.assertEqual(relevant, ["summary_progress", "summary_progress", "summary_notice"])

    async def test_failed_or_empty_summary_does_not_reset_progress(self):
        for response in (RuntimeError("summarizer failed"), AIMessage(content="")):
            with self.subTest(response=response):
                nodes = self._make_nodes(summary_threshold=10000)
                events = []

                async def summarize(*args, **kwargs):
                    self.assertEqual(len(events), 1)
                    self.assertTrue(events[0]["will_summarize"])
                    self.assertEqual(events[0]["progress"], 0.0)
                    if isinstance(response, Exception):
                        raise response
                    return response

                nodes.llm.ainvoke.side_effect = summarize
                state = {"messages": [HumanMessage(content="old detail " * 18000, id="old"),
                                      HumanMessage(content="continue", id="new")], "steps": 0}
                with patch("core.nodes.summarize.get_stream_writer", return_value=events.append):
                    update = await nodes.summarize_node(state)
                nodes.llm.ainvoke.assert_awaited_once()
                self.assertEqual(len(events), 1)
                self.assertNotIn("messages", update)
                self.assertNotIn("summary", update)


if __name__ == "__main__":
    unittest.main()
