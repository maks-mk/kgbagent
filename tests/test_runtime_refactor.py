import json
import asyncio
import os
import shutil
import sqlite3
import sys
import unittest
import base64
import logging
from types import ModuleType, SimpleNamespace
from unittest import mock
from pathlib import Path
from uuid import uuid4

import agent as agent_module
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, RootModel

from agent import _register_llm_cleanup_callback, create_agent_workflow, create_llm, prepare_llm_with_tools
from core.api_key_rotation import ApiKeyRotationExhaustedError, RotatingChatModel, describe_provider_error
from core.checkpointing import create_checkpoint_runtime
from core.config import AgentConfig
from core.constants import REFLECTION_PROMPT
from core.multimodal import (
    extract_model_capabilities,
    materialize_user_message_content_for_model,
    normalize_model_capabilities,
    resolve_model_capabilities,
)
from core.model_profiles import ModelProfileStore
from core.nodes import AgentNodes
from core.run_logger import JsonlRunLogger
from core.session_store import SessionSnapshot, SessionStore
from core.state import AgentState, append_transcript_messages
from core.summarize_policy import (
    choose_summary_boundary,
    estimate_summary_tokens,
    format_history_for_summary,
    should_summarize,
    summary_progress_ratio,
    summary_trigger_tokens,
    truncate_summary_to_token_budget,
)
from core.tool_policy import ToolMetadata
from core.turn_outcomes import (
    TURN_OUTCOME_CONTINUE_AGENT,
    TURN_OUTCOME_FINISH_TURN,
    TURN_OUTCOME_RECOVER_AGENT,
    TURN_OUTCOME_RUN_TOOLS,
    normalize_turn_outcome,
)
from tools import process_tools
from tools.filesystem import write_file_tool
from tools.tool_registry import ToolRegistry
import ui.runtime as gui_runtime
from ui.runtime import (
    append_project_label,
    build_help_markdown,
    build_initial_state,
    build_transcript_payload,
    build_user_choice_payload,
    generate_chat_title,
    short_project_label,
)
from ui.streaming import StreamEvent, StreamProcessResult, StreamProcessor


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    async def ainvoke(self, context):
        self.invocations.append(context)
        if not self.responses:
            return AIMessage(content="STATUS: FINISHED\nREASON: fallback\nNEXT_STEP: NONE\nCONTROL: FINISH_TURN")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ProviderSafeFakeLLM(FakeLLM):
    async def ainvoke(self, context):
        last_visible = next((message for message in reversed(context) if message.type != "system"), None)
        if isinstance(last_visible, AIMessage):
            raise AssertionError("provider-unsafe assistant-last context")
        return await super().ainvoke(context)


class OpenAIContentSafeFakeLLM(FakeLLM):
    async def ainvoke(self, context):
        for message in context:
            if isinstance(message, AIMessage) and not isinstance(message.content, str):
                raise AssertionError(f"assistant content must be string for OpenAI chat history, got {type(message.content).__name__}")
        return await super().ainvoke(context)


class FakeBindableLLM(FakeLLM):
    def __init__(self, responses):
        super().__init__(responses)
        self.bound_tool_name_batches = []

    def bind_tools(self, tools):
        self.bound_tool_name_batches.append([tool.name for tool in tools])
        return self


class FailingBindableLLM(FakeLLM):
    def __init__(self, responses, message="bind failed"):
        super().__init__(responses)
        self.bound_tool_name_batches = []
        self.message = message

    def bind_tools(self, tools):
        self.bound_tool_name_batches.append([tool.name for tool in tools])
        raise RuntimeError(self.message)


class FakeTool:
    def __init__(self, name, result):
        self.name = name
        self.description = f"Fake tool {name}"
        self.result = result
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        if callable(self.result):
            return self.result(args)
        return self.result


class FakeSchemaTool(FakeTool):
    def __init__(self, name, result, schema):
        super().__init__(name, result)
        self._schema = schema

    def get_input_schema(self):
        return self._schema


class FakeProfileLLM:
    def __init__(self, profile):
        self.profile = profile


class RuntimeRefactorTests(unittest.IsolatedAsyncioTestCase):
    def _workspace_tempdir(self) -> Path:
        path = Path.cwd() / ".tmp_tests" / uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def _require_sqlite_filesystem(self) -> None:
        probe_dir = self._workspace_tempdir()
        probe_path = probe_dir / "probe.sqlite"
        try:
            conn = sqlite3.connect(probe_path)
            conn.execute("create table t(x int)")
            conn.commit()
            conn.close()
        except sqlite3.OperationalError as exc:
            self.skipTest(f"SQLite file backend is unavailable in this environment: {exc}")

    def _make_config(self, **overrides):
        defaults = {
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "PROMPT_PATH": Path(__file__).resolve().parents[1] / "prompt.txt",
            "MCP_CONFIG_PATH": Path(__file__).resolve().parents[1] / "tests" / "missing_mcp.json",
            "ENABLE_SEARCH_TOOLS": False,
            "ENABLE_PROCESS_TOOLS": False,
            "ENABLE_SHELL_TOOL": False,
        }
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def _initial_state(self, task="Проверь задачу", session_id="session-test", run_id="run-test"):
        return {
            "messages": [HumanMessage(content=task)],
            "steps": 0,
            "token_usage": {},
            "current_task": task,
            "turn_outcome": "",
            "recovery_state": {
                "turn_id": 1,
                "retry_count": 0,
                "retry_fingerprint_history": [],
                "last_reason": "",
                "active_issue": None,
                "active_strategy": None,
                "last_successful_evidence": "",
                "external_blocker": None,
                "llm_replan_attempted_for": [],
            },
            "session_id": session_id,
            "run_id": run_id,
            "turn_id": 1,
            "pending_approval": None,
            "open_tool_issue": None,
            "last_tool_error": "",
            "last_tool_result": "",
        }

    def test_model_capabilities_normalize_profile_variants(self):
        self.assertEqual(normalize_model_capabilities(None), {"image_input_supported": False})
        self.assertEqual(normalize_model_capabilities({"image_input": True}), {"image_input_supported": True})
        self.assertEqual(normalize_model_capabilities({"imageInputs": True}), {"image_input_supported": True})
        self.assertEqual(normalize_model_capabilities({"image_inputs": True}), {"image_input_supported": True})
        self.assertEqual(
            normalize_model_capabilities({"image_input_supported": True}),
            {"image_input_supported": True},
        )
        self.assertEqual(
            extract_model_capabilities(FakeProfileLLM({"imageInputs": True})),
            {"image_input_supported": True},
        )

    def test_resolve_model_capabilities_prefers_profile_override(self):
        self.assertEqual(
            resolve_model_capabilities(
                {"supports_image_input": True},
                {"image_input_supported": False},
            ),
            {"image_input_supported": True},
        )

    def test_root_model_tool_schema_does_not_require_literal_root_field(self):
        class SequentialThinkingArgs(RootModel[dict]):
            pass

        tool = FakeSchemaTool("sequentialthinking", "ok", SequentialThinkingArgs)
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM([]),
        )

        missing = nodes._missing_required_tool_fields(
            "sequentialthinking",
            {"thought": "test", "thoughtNumber": 1, "totalThoughts": 1},
        )

        self.assertEqual(missing, [])

    def test_root_model_ref_schema_uses_inner_required_fields(self):
        class SequentialThinkingPayload(BaseModel):
            thought: str
            nextThoughtNeeded: bool
            thoughtNumber: int
            totalThoughts: int

        class SequentialThinkingArgs(RootModel[SequentialThinkingPayload]):
            pass

        tool = FakeSchemaTool("sequentialthinking", "ok", SequentialThinkingArgs)
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM([]),
        )

        missing = nodes._missing_required_tool_fields(
            "sequentialthinking",
            {"thought": "test", "thoughtNumber": 1, "totalThoughts": 1},
        )

        self.assertEqual(missing, ["nextThoughtNeeded"])

    def test_missing_required_tool_fields_respects_write_file_aliases(self):
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[write_file_tool],
            llm_with_tools=FakeLLM([]),
        )

        missing = nodes._missing_required_tool_fields(
            "write_file",
            {"path": "notes.md", "body": "# Notes\n\nReady"},
        )

        self.assertEqual(missing, [])

    def test_build_agent_result_does_not_invent_gemini_thought_signature(self):
        tool = FakeTool("batch_web_search", "ok")
        nodes = AgentNodes(
            config=self._make_config(
                PROVIDER="gemini",
                GEMINI_API_KEY="test-key",
                GEMINI_MODEL="gemini-robotics-er-1.6-preview",
            ),
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM([]),
        )
        response = AIMessage(content="").model_copy(
            update={
                "tool_calls": [
                    {"id": "tc-1", "name": "batch_web_search", "args": {"queries": ["latest Claude Opus 4.7"]}},
                ]
            }
        )

        result = nodes._build_agent_result(
            response=response,
            current_task="Найди последнюю информацию",
            tools_available=True,
            turn_id=1,
            messages=[HumanMessage(content="Найди последнюю информацию")],
            allowed_tool_names=["batch_web_search"],
        )

        message = next(item for item in result["messages"] if isinstance(item, AIMessage))
        self.assertNotIn("__gemini_function_call_thought_signatures__", message.additional_kwargs)
        self.assertEqual(result["turn_outcome"], "run_tools")

    def test_build_agent_result_rejects_textual_tool_call_by_default(self):
        tool = FakeTool("read_file", "ok")
        nodes = AgentNodes(
            config=self._make_config(PROVIDER="openai"),
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM([]),
        )
        response = AIMessage(
            content='Читаю файл.call:read_file{path:<|"|>index.html<|"|>}<tool_call|>'
        )

        result = nodes._build_agent_result(
            response=response,
            current_task="прочитай файл",
            tools_available=True,
            turn_id=1,
            messages=[HumanMessage(content="прочитай файл")],
            allowed_tool_names=["read_file"],
        )

        message = next(item for item in result["messages"] if isinstance(item, AIMessage))
        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_RECOVER_AGENT)
        self.assertEqual(result["open_tool_issue"]["kind"], "protocol_error")
        self.assertEqual(result["open_tool_issue"]["details"]["protocol_reason"], "textual_tool_call_disabled")
        self.assertEqual(result["open_tool_issue"]["tool_names"], ["read_file"])
        self.assertEqual(result["open_tool_issue"]["tool_args"], {"path": "index.html"})
        self.assertFalse(message.tool_calls)
        self.assertFalse(message.additional_kwargs["agent_internal"]["visible_in_ui"])
        self.assertNotIn("call:read_file", str(message.content))
        self.assertNotIn("<tool_call|>", str(message.content))

    def test_build_agent_result_recovers_textual_tool_call_when_flag_enabled(self):
        tool = FakeTool("read_file", "ok")
        nodes = AgentNodes(
            config=self._make_config(PROVIDER="openai", ENABLE_TEXT_TOOL_CALL_RECOVERY=True),
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM([]),
        )
        response = AIMessage(
            content='Читаю файл.call:read_file{path:<|"|>index.html<|"|>}<tool_call|>'
        )

        result = nodes._build_agent_result(
            response=response,
            current_task="прочитай файл",
            tools_available=True,
            turn_id=1,
            messages=[HumanMessage(content="прочитай файл")],
            allowed_tool_names=["read_file"],
        )

        message = next(item for item in result["messages"] if isinstance(item, AIMessage))
        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_RUN_TOOLS)
        self.assertEqual(message.tool_calls[0]["name"], "read_file")
        self.assertEqual(message.tool_calls[0]["args"], {"path": "index.html"})
        self.assertNotIn("call:read_file", str(message.content))
        self.assertNotIn("<tool_call|>", str(message.content))

    def test_normalize_turn_outcome_defaults_unknown_values_to_finish_turn(self):
        self.assertEqual(normalize_turn_outcome("run_tools"), TURN_OUTCOME_RUN_TOOLS)
        self.assertEqual(normalize_turn_outcome(" recover_agent "), TURN_OUTCOME_RECOVER_AGENT)
        self.assertEqual(normalize_turn_outcome(""), TURN_OUTCOME_FINISH_TURN)
        self.assertEqual(normalize_turn_outcome("unexpected"), TURN_OUTCOME_FINISH_TURN)

    def test_build_agent_result_preserves_existing_gemini_thought_signature(self):
        tool = FakeTool("batch_web_search", "ok")
        nodes = AgentNodes(
            config=self._make_config(
                PROVIDER="gemini",
                GEMINI_API_KEY="test-key",
                GEMINI_MODEL="gemini-robotics-er-1.6-preview",
            ),
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM([]),
        )
        response = AIMessage(
            content="",
            additional_kwargs={
                "__gemini_function_call_thought_signatures__": {
                    "tc-1": "existing-signature",
                }
            },
        ).model_copy(
            update={
                "tool_calls": [
                    {"id": "tc-1", "name": "batch_web_search", "args": {"queries": ["latest Claude Opus 4.7"]}},
                ]
            }
        )

        result = nodes._build_agent_result(
            response=response,
            current_task="Найди последнюю информацию",
            tools_available=True,
            turn_id=1,
            messages=[HumanMessage(content="Найди последнюю информацию")],
            allowed_tool_names=["batch_web_search"],
        )

        message = next(item for item in result["messages"] if isinstance(item, AIMessage))
        signature_map = message.additional_kwargs["__gemini_function_call_thought_signatures__"]
        self.assertEqual(signature_map["tc-1"], "existing-signature")

    def test_sanitize_messages_for_model_remaps_gemini_thought_signature_keys_with_tool_ids(self):
        tool = FakeTool("batch_web_search", "ok")
        nodes = AgentNodes(
            config=self._make_config(
                PROVIDER="gemini",
                GEMINI_API_KEY="test-key",
                GEMINI_MODEL="gemini-robotics-er-1.6-preview",
            ),
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM([]),
        )
        long_tool_id = "chatcmpl-tool-9d8dc0bbe38d6202"
        messages = [
            HumanMessage(content="Найди последнюю информацию"),
            AIMessage(
                content="",
                additional_kwargs={
                    "__gemini_function_call_thought_signatures__": {
                        long_tool_id: "existing-signature",
                    }
                },
                tool_calls=[
                    {"id": long_tool_id, "name": "batch_web_search", "args": {"queries": ["latest Claude Opus 4.7"]}},
                ],
            ),
            ToolMessage(content="ok", tool_call_id=long_tool_id, name="batch_web_search"),
        ]

        sanitized = nodes._sanitize_messages_for_model(messages)

        ai_message = next(message for message in sanitized if isinstance(message, AIMessage))
        tool_message = next(message for message in sanitized if isinstance(message, ToolMessage))
        remapped_id = ai_message.tool_calls[0]["id"]
        signature_map = ai_message.additional_kwargs["__gemini_function_call_thought_signatures__"]

        self.assertRegex(remapped_id, r"^[A-Za-z0-9]{9}$")
        self.assertEqual(tool_message.tool_call_id, remapped_id)
        self.assertEqual(signature_map[remapped_id], "existing-signature")
        self.assertNotIn(long_tool_id, signature_map)

    async def test_process_tool_call_allows_repeated_distinct_args_without_name_based_budget(self):
        tool = FakeTool("analysis_helper", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM([]),
        )
        state = self._initial_state(task="Проведи анализ")
        recent_calls = [
            {"name": "analysis_helper", "args": {"step": 1}},
            {"name": "analysis_helper", "args": {"step": 2}},
            {"name": "analysis_helper", "args": {"step": 3}},
            {"name": "analysis_helper", "args": {"step": 4}},
        ]

        tool_message, had_error, issue = await nodes._process_tool_call(
            {"id": "call_123", "name": "analysis_helper", "args": {"step": 5}},
            recent_calls,
            state,
            approval_state={},
            current_turn_id=1,
        )

        self.assertFalse(had_error)
        self.assertIsNone(issue)
        self.assertEqual(tool.calls, [{"step": 5}])
        self.assertEqual(tool_message.status, "success")

    async def test_process_tool_call_parses_json_string_args_before_preflight_validation(self):
        class SaveDocumentArgs(BaseModel):
            path: str
            content: str

        tool = FakeSchemaTool("save_document", "ok", SaveDocumentArgs)
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM([]),
        )
        state = self._initial_state(task="Сохрани отчёт")

        tool_message, had_error, issue = await nodes._process_tool_call(
            {
                "id": "call_123",
                "name": "save_document",
                "args": json.dumps({"path": "report.md", "content": "# Отчёт\n\nГотово"}, ensure_ascii=False),
            },
            recent_calls=[],
            state=state,
            approval_state={},
            current_turn_id=1,
        )

        self.assertFalse(had_error)
        self.assertIsNone(issue)
        self.assertEqual(tool.calls, [{"path": "report.md", "content": "# Отчёт\n\nГотово"}])
        self.assertEqual(tool_message.status, "success")

    async def test_agent_node_skips_llm_when_recovery_strategy_repeats_exhausted_tool_call(self):
        config = self._make_config(TOOL_LOOP_LIMIT_READONLY=2)
        llm = FakeLLM([AIMessage(content="Этого ответа не должно быть.")])
        tool = FakeTool("read_file", "Success: file contents")
        nodes = AgentNodes(
            config=config,
            llm=llm,
            tools=[tool],
            llm_with_tools=llm,
        )
        state = self._initial_state("Проверь файл")
        state["messages"] = [
            HumanMessage(content="Проверь файл"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "demo.txt"}, "id": "tc-1"}]),
            ToolMessage(tool_call_id="tc-1", name="read_file", content="ERROR[LOOP_DETECTED]: loop"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "demo.txt"}, "id": "tc-2"}]),
            ToolMessage(tool_call_id="tc-2", name="read_file", content="ERROR[LOOP_DETECTED]: loop"),
        ]
        state["open_tool_issue"] = {
            "turn_id": 1,
            "kind": "tool_error",
            "summary": "Loop detected for read_file.",
            "tool_names": ["read_file"],
            "tool_args": {"path": "demo.txt"},
            "source": "tools",
            "error_type": "LOOP_DETECTED",
            "fingerprint": "fp-loop-read",
            "progress_fingerprint": "fp-loop-read",
            "details": {"loop_detected": True},
        }
        state["recovery_state"] = {
            "turn_id": 1,
            "active_issue": dict(state["open_tool_issue"]),
            "active_strategy": {
                "id": "strategy-read-file",
                "strategy": "retry_same_call",
                "tool_name": "read_file",
                "suggested_tool_name": "read_file",
                "patched_args": {"path": "demo.txt"},
            },
            "last_successful_evidence": "",
            "external_blocker": None,
            "llm_replan_attempted_for": [],
        }

        result = await nodes.agent_node(state)

        self.assertEqual(result["turn_outcome"], "recover_agent")
        self.assertTrue(result["open_tool_issue"]["details"]["preflight_blocked"])
        self.assertEqual(llm.invocations, [])

    async def test_process_tool_call_logs_json_string_args_canonicalization(self):
        tool = FakeTool("save_document", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM([]),
        )
        state = self._initial_state(task="Сохрани отчёт")
        events: list[tuple[str, dict[str, Any]]] = []
        with mock.patch.object(
            AgentNodes,
            "_log_run_event",
            autospec=True,
            side_effect=lambda _self, _state, event_type, **payload: events.append((event_type, payload)),
        ):
            await nodes._process_tool_call(
                {
                    "id": "call_123",
                    "name": "save_document",
                    "args": json.dumps({"path": "report.md", "content": "body"}, ensure_ascii=False),
                },
                recent_calls=[],
                state=state,
                approval_state={},
                current_turn_id=1,
            )

        canonicalized = [payload for event_type, payload in events if event_type == "tool_call_args_canonicalized"]
        self.assertEqual(len(canonicalized), 1)
        self.assertEqual(canonicalized[0]["tool_name"], "save_document")
        self.assertEqual(canonicalized[0]["source_kind"], "json_string")
        self.assertEqual(canonicalized[0]["arg_keys"], ["content", "path"])

    def test_prepare_llm_with_tools_disables_tool_calling_on_bind_failure(self):
        llm = FailingBindableLLM([], message="provider bind error")
        tool = FakeTool("read_file", "ok")

        bound_llm, enabled, error = prepare_llm_with_tools(llm, [tool])

        self.assertIs(bound_llm, llm)
        self.assertFalse(enabled)
        self.assertEqual(error, "provider bind error")

    def test_create_llm_for_gemini_does_not_force_system_to_human_conversion(self):
        captured = {}

        class FakeChatGoogleGenerativeAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_google_genai": mock.Mock(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="gemini",
                    GEMINI_API_KEY="gm-test",
                    GEMINI_MODEL="gemini-2.5-flash",
                )
            )

        self.assertEqual(captured["model"], "gemini-2.5-flash")
        self.assertEqual(captured["google_api_key"], "gm-test")
        self.assertEqual(captured["max_retries"], 0)
        self.assertNotIn("convert_system_message_to_human", captured)
        self.assertNotIn("default_headers", captured)

    def test_create_llm_for_gemini_does_not_send_top_p_or_top_k(self):
        captured = {}

        class FakeChatGoogleGenerativeAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_google_genai": mock.Mock(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="gemini",
                    GEMINI_API_KEY="gm-test",
                    GEMINI_MODEL="gemini-2.5-flash",
                    TEMPERATURE=0.4,
                    TOP_P=0.9,
                    TOP_K=32,
                )
            )

        self.assertEqual(captured["temperature"], 0.4)
        self.assertNotIn("top_p", captured)
        self.assertNotIn("top_k", captured)

    def test_create_llm_for_fixed_sampling_gemini_does_not_send_temperature(self):
        captured = {}

        class FakeChatGoogleGenerativeAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_google_genai": mock.Mock(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="gemini",
                    GEMINI_API_KEY="gm-test",
                    GEMINI_MODEL="models/gemini-3.6-flash-001",
                    TEMPERATURE=0.4,
                )
            )

        self.assertNotIn("temperature", captured)

    def test_create_llm_for_gemini_enables_thinking_budget_by_default_for_thinking_models(self):
        captured = {}

        class FakeChatGoogleGenerativeAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_google_genai": mock.Mock(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="gemini",
                    GEMINI_API_KEY="gm-test",
                    GEMINI_MODEL="gemini-2.5-flash",
                )
            )

        self.assertEqual(captured["thinking_budget"], 4096)
        self.assertIs(captured["include_thoughts"], True)

    def test_create_llm_for_gemini3_uses_thinking_level_when_supported(self):
        captured = {}

        class FakeChatGoogleGenerativeAI:
            model_fields = {"thinking_level": object(), "thinking_budget": object(), "include_thoughts": object()}

            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_google_genai": mock.Mock(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="gemini",
                    GEMINI_API_KEY="gm-test",
                    GEMINI_MODEL="gemini-3-flash",
                    MODEL_REASONING_EFFORT="low",
                )
            )

        self.assertEqual(captured["thinking_level"], "low")
        self.assertIs(captured["include_thoughts"], True)
        self.assertNotIn("thinking_budget", captured)

    def test_create_llm_for_gemini3_uses_thinking_level_alias_from_installed_api_shape(self):
        captured = {}

        class FakeField:
            def __init__(self, alias=None):
                self.alias = alias

        class FakeChatGoogleGenerativeAI:
            model_fields = {
                "reasoning_effort": FakeField(alias="thinking_level"),
                "include_thoughts": FakeField(),
            }

            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_google_genai": mock.Mock(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="gemini",
                    GEMINI_API_KEY="gm-test",
                    GEMINI_MODEL="gemini-3.6-flash",
                    MODEL_REASONING_EFFORT="high",
                )
            )

        self.assertEqual(captured["thinking_level"], "high")
        self.assertIs(captured["include_thoughts"], True)
        self.assertNotIn("thinking_budget", captured)

    def test_create_llm_for_gemma4_uses_documented_minimal_or_high_thinking_level(self):
        captured = {}

        class FakeChatGoogleGenerativeAI:
            model_fields = {"thinking_level": object(), "include_thoughts": object()}

            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_google_genai": mock.Mock(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="gemini",
                    GEMINI_API_KEY="gm-test",
                    GEMINI_MODEL="gemma-4-26b-a4b-it",
                    MODEL_REASONING_EFFORT="high",
                )
            )

        self.assertEqual(captured["thinking_level"], "high")
        self.assertIs(captured["include_thoughts"], True)


    def test_create_llm_for_gemini3_does_not_send_legacy_thinking_budget(self):
        captured = {}

        class FakeChatGoogleGenerativeAI:
            model_fields = {"thinking_budget": object()}

            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_google_genai": mock.Mock(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="gemini",
                    GEMINI_API_KEY="gm-test",
                    GEMINI_MODEL="gemini-3-flash",
                )
            )

        self.assertNotIn("thinking_level", captured)
        self.assertNotIn("thinking_budget", captured)
        self.assertNotIn("include_thoughts", captured)

    def test_create_llm_for_gemini_skips_thinking_budget_for_older_models(self):
        captured = {}

        class FakeChatGoogleGenerativeAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_google_genai": mock.Mock(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="gemini",
                    GEMINI_API_KEY="gm-test",
                    GEMINI_MODEL="gemini-1.5-flash",
                )
            )

        self.assertNotIn("thinking_budget", captured)
        self.assertNotIn("include_thoughts", captured)

    def test_gemini_adapter_preserves_function_call_thought_signatures_from_response(self):
        from google.genai.types import Candidate, Content, FunctionCall, GenerateContentResponse, Part

        llm = create_llm(
            self._make_config(
                PROVIDER="gemini",
                GEMINI_API_KEY="gm-test",
                GEMINI_MODEL="gemini-2.5-flash",
            )
        )
        response = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        role="model",
                        parts=[
                            Part(
                                functionCall=FunctionCall(name="read_file", args={"path": "index.html"}),
                                thoughtSignature=b"real-signature",
                            )
                        ],
                    )
                )
            ],
            modelVersion="gemini-2.5-flash",
        )
        llm.client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_kwargs: response))

        result = llm._generate([HumanMessage(content="Read the file")])

        message = result.generations[0].message
        tool_call_id = message.tool_calls[0]["id"]
        signature_map = message.additional_kwargs["__gemini_function_call_thought_signatures__"]
        self.assertEqual(base64.b64decode(signature_map[tool_call_id]), b"real-signature")

    def test_gemini_adapter_adds_thought_signatures_to_outbound_function_call_parts(self):
        llm = create_llm(
            self._make_config(
                PROVIDER="gemini",
                GEMINI_API_KEY="gm-test",
                GEMINI_MODEL="gemini-2.5-flash",
            )
        )
        ai_message = AIMessage(
            content="",
            additional_kwargs={
                "__gemini_function_call_thought_signatures__": {
                    "tc-1": base64.b64encode(b"real-signature").decode("ascii"),
                }
            },
            tool_calls=[{"id": "tc-1", "name": "read_file", "args": {"path": "index.html"}}],
        )

        request = llm._prepare_request(
            [
                HumanMessage(content="Read the file"),
                ai_message,
                ToolMessage(content="ok", tool_call_id="tc-1", name="read_file"),
            ]
        )

        model_content = next(content for content in request["contents"] if content.role == "model")
        self.assertEqual(model_content.parts[0].thought_signature, b"real-signature")

    async def test_gemini_retry_patch_strips_retry_kwargs_before_async_generation_call(self):
        forwarded_kwargs = {}

        async def fake_generation_method(**kwargs):
            forwarded_kwargs.update(kwargs)
            return "ok"

        def fake_chat_with_retry(*, generation_method, **kwargs):
            return generation_method(**kwargs)

        async def fake_achat_with_retry(*, generation_method, **kwargs):
            return await generation_method(**kwargs)

        fake_chat_models = ModuleType("langchain_google_genai.chat_models")
        fake_chat_models._chat_with_retry = fake_chat_with_retry
        fake_chat_models._achat_with_retry = fake_achat_with_retry

        fake_package = ModuleType("langchain_google_genai")
        fake_package.ChatGoogleGenerativeAI = object

        with mock.patch.dict(
            sys.modules,
            {
                "langchain_google_genai": fake_package,
                "langchain_google_genai.chat_models": fake_chat_models,
            },
        ):
            agent_module._patch_langchain_google_genai_retry_kwargs()
            await fake_chat_models._achat_with_retry(
                generation_method=fake_generation_method,
                request=SimpleNamespace(model="gemini-2.5-flash"),
                timeout=30,
                metadata=[("x-test", "1")],
                max_retries=6,
                wait_exponential_multiplier=2.0,
                wait_exponential_min=1.0,
                wait_exponential_max=60.0,
            )

        self.assertEqual(forwarded_kwargs["timeout"], 30)
        self.assertEqual(forwarded_kwargs["metadata"], [("x-test", "1")])
        self.assertEqual(forwarded_kwargs["request"].model, "gemini-2.5-flash")
        self.assertNotIn("max_retries", forwarded_kwargs)
        self.assertNotIn("wait_exponential_multiplier", forwarded_kwargs)
        self.assertNotIn("wait_exponential_min", forwarded_kwargs)
        self.assertNotIn("wait_exponential_max", forwarded_kwargs)

    def test_create_llm_for_openai_disables_sdk_retries(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="gpt-4o",
                    OPENAI_BASE_URL="https://api.openai.com/v1",
                )
            )

        self.assertEqual(captured["model"], "gpt-4o")
        self.assertEqual(captured["api_key"], "sk-test")
        self.assertEqual(captured["max_retries"], 0)
        self.assertTrue(captured["stream_usage"])
        self.assertNotIn("reasoning", captured)
        self.assertNotIn("reasoning_effort", captured)

    def test_create_llm_for_openai_uses_configured_headers(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        headers = {"User-Agent": "CustomAgent/1.0", "x-provider-header": "enabled"}
        with (
            mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}),
            mock.patch("core.providers.openai_reasoning.load_openai_headers", return_value=headers),
        ):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="gpt-4o",
                    OPENAI_BASE_URL="https://api.openai.com/v1",
                )
            )

        self.assertEqual(captured["default_headers"], headers)

    def test_create_llm_for_openai_does_not_send_sampling_controls(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="gpt-4o",
                    OPENAI_BASE_URL="https://api.openai.com/v1",
                    TOP_P=0.9,
                    TOP_K=32,
                )
            )

        self.assertNotIn("top_p", captured)
        self.assertNotIn("top_k", captured)

    def test_create_llm_for_gpt_6_astra_does_not_send_temperature(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="gpt-6-astra",
                    OPENAI_BASE_URL="https://api.openai.com/v1",
                    TEMPERATURE=0.4,
                )
            )

        self.assertNotIn("temperature", captured)
        self.assertEqual(captured["model"], "gpt-6-astra")

    def test_create_llm_for_regular_openai_model_still_sends_temperature(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="gpt-4o",
                    OPENAI_BASE_URL="https://api.openai.com/v1",
                    TEMPERATURE=0.4,
                )
            )

        self.assertEqual(captured["temperature"], 0.4)

    async def test_register_llm_cleanup_callback_closes_async_client(self):
        registry = ToolRegistry(self._make_config())
        fake_client = mock.AsyncMock()
        fake_llm = SimpleNamespace(async_client=fake_client)

        self.assertTrue(_register_llm_cleanup_callback(registry, fake_llm))

        await registry.cleanup()
        fake_client.aclose.assert_awaited_once()

    async def test_register_llm_cleanup_callback_closes_root_async_client(self):
        registry = ToolRegistry(self._make_config())
        fake_client = mock.AsyncMock()
        fake_llm = SimpleNamespace(root_async_client=fake_client)

        self.assertTrue(_register_llm_cleanup_callback(registry, fake_llm))

        await registry.cleanup()
        fake_client.aclose.assert_awaited_once()

    async def test_registry_cleanup_deduplicates_shared_llm_client_and_is_idempotent(self):
        registry = ToolRegistry(self._make_config())
        shared_client = mock.AsyncMock()
        llm = SimpleNamespace(root_async_client=shared_client)
        bound_llm = SimpleNamespace(root_async_client=shared_client)

        self.assertTrue(_register_llm_cleanup_callback(registry, llm))
        self.assertTrue(_register_llm_cleanup_callback(registry, bound_llm))

        await registry.cleanup()
        await registry.cleanup()

        shared_client.aclose.assert_awaited_once()

    async def test_rotating_chat_model_reuses_cached_model_and_closes_clients(self):
        tmp = Path.cwd() / ".tmp_tests" / uuid4().hex
        tmp.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))

        created_models = []

        class FakeProviderModel:
            def __init__(self):
                self.async_client = mock.AsyncMock()

            async def ainvoke(self, _context, **_kwargs):
                return AIMessage(content="ok")

        def _factory(_config, *, api_key_override=None):
            self.assertEqual(api_key_override, "gm-test")
            model = FakeProviderModel()
            created_models.append(model)
            return model

        model = RotatingChatModel(
            config=self._make_config(PROVIDER="gemini", GEMINI_API_KEY="gm-test", GEMINI_MODEL="gemini-2.5-flash"),
            profile_id="gemini-profile",
            profile_store_path=tmp / "profiles.json",
            llm_factory=_factory,
        )

        await model.ainvoke([HumanMessage(content="one")])
        await model.ainvoke([HumanMessage(content="two")])

        self.assertEqual(len(created_models), 1)

        await model.aclose()
        created_models[0].async_client.aclose.assert_awaited_once()

    async def test_rotating_chat_model_closes_root_clients_inside_bound_model(self):
        tmp = Path.cwd() / ".tmp_tests" / uuid4().hex
        tmp.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        root_async_client = mock.AsyncMock()
        root_client = SimpleNamespace(close=mock.Mock())
        provider_model = SimpleNamespace(
            root_async_client=root_async_client,
            root_client=root_client,
        )
        bound_model = SimpleNamespace(
            bound=provider_model,
            ainvoke=mock.AsyncMock(return_value=AIMessage(content="ok")),
        )

        model = RotatingChatModel(
            config=self._make_config(PROVIDER="openai", OPENAI_API_KEY="sk-test", OPENAI_MODEL="gpt-4o"),
            profile_id="openai-profile",
            profile_store_path=tmp / "profiles.json",
            llm_factory=lambda *_args, **_kwargs: bound_model,
        )

        await model.ainvoke([HumanMessage(content="one")])
        await model.aclose()

        root_async_client.aclose.assert_awaited_once()
        root_client.close.assert_called_once_with()

    def test_create_llm_for_openai_reasoning_models_requests_reasoning_summary_by_default(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="gpt-5-mini",
                    OPENAI_BASE_URL="https://api.openai.com/v1",
                    LLM_API_MODE="responses",
                )
            )

        self.assertEqual(captured["reasoning"], {"effort": "medium", "summary": "auto"})

    def test_create_llm_for_openai_reasoning_models_keeps_reasoning_with_legacy_flag_off(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="gpt-5-mini",
                    OPENAI_BASE_URL="https://api.openai.com/v1",
                    SHOW_MODEL_THOUGHTS=False,
                    LLM_API_MODE="responses",
                )
            )

        self.assertEqual(captured["reasoning"], {"effort": "medium", "summary": "auto"})

    def test_create_llm_for_openai_compatible_reasoning_model_uses_reasoning_effort(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="openai/gpt-oss-120b",
                    OPENAI_BASE_URL="https://openrouter.ai/api/v1",
                    MODEL_REASONING_EFFORT="high",
                )
            )

        self.assertNotIn("reasoning", captured)
        self.assertEqual(captured["extra_body"], {"reasoning": {"effort": "high"}})

    def test_create_llm_for_openai_compatible_provider_does_not_require_model_name_gate(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="new-provider-reasoning-model",
                    OPENAI_BASE_URL="https://api.b.ai/v1",
                    MODEL_REASONING_EFFORT="medium",
                )
            )

        self.assertNotIn("reasoning", captured)
        self.assertNotIn("extra_body", captured)
        self.assertEqual(captured["reasoning_effort"], "medium")

    def test_openai_raw_reasoning_delta_is_extracted_from_compatible_stream_chunk(self):
        chunk = {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "reasoning": "checking",
                        "reasoning_content": "checking",
                    }
                }
            ]
        }

        self.assertEqual(agent_module._extract_openai_reasoning_delta(chunk), "checking")

    def test_gemini_latest_flash_alias_uses_thinking_budget(self):
        self.assertTrue(agent_module._gemini_model_supports_thinking_budget("gemini-flash-latest"))

    def test_create_llm_for_openai_compatible_reasoning_model_does_not_force_responses_api(self):
        from langchain_core.messages import HumanMessage

        llm = create_llm(
            self._make_config(
                PROVIDER="openai",
                OPENAI_API_KEY="sk-test",
                OPENAI_MODEL="openai/gpt-oss-120b",
                OPENAI_BASE_URL="https://openrouter.ai/api/v1",
                MODEL_REASONING_EFFORT="xhigh",
            )
        )

        payload = llm._get_request_payload([HumanMessage(content="hi")])

        self.assertFalse(llm._use_responses_api(payload))
        self.assertEqual(payload["extra_body"], {"reasoning": {"effort": "xhigh"}})

    def test_create_llm_for_ollama_cloud_reasoning_model_uses_reasoning_effort(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="gpt-oss:120b",
                    OPENAI_BASE_URL="https://ollama.com/v1",
                    MODEL_REASONING_EFFORT="xhigh",
                )
            )

        self.assertEqual(captured["reasoning_effort"], "high")
        self.assertNotIn("extra_body", captured)

    def test_create_llm_for_ollama_cloud_gemma_reasoning_model_uses_reasoning_effort(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="gemma4:31b",
                    OPENAI_BASE_URL="https://ollama.com/v1",
                    MODEL_REASONING_EFFORT="medium",
                )
            )

        self.assertEqual(captured["reasoning_effort"], "medium")
        self.assertNotIn("extra_body", captured)

    def test_create_llm_for_nvidia_gpt_oss_uses_documented_reasoning_effort(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="openai/gpt-oss-120b",
                    OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1",
                    MODEL_REASONING_EFFORT="high",
                )
            )

        self.assertNotIn("reasoning", captured)
        self.assertNotIn("extra_body", captured)
        self.assertEqual(captured["reasoning_effort"], "high")

    def test_create_llm_for_nvidia_deepseek_v4_uses_documented_reasoning_effort(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="deepseek-ai/deepseek-v4-flash-0731",
                    OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1",
                    MODEL_REASONING_EFFORT="max",
                )
            )

        self.assertNotIn("reasoning", captured)
        self.assertNotIn("extra_body", captured)
        self.assertEqual(captured["reasoning_effort"], "max")

        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="qwen/qwen3-235b-a22b",
                    OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1",
                    MODEL_REASONING_EFFORT="high",
                )
            )

        self.assertNotIn("reasoning", captured)
        self.assertNotIn("reasoning_effort", captured)
        self.assertEqual(
            captured["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": True}},
        )

    def test_create_llm_for_nvidia_qwen_explicitly_disables_default_thinking(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="qwen/qwen3-235b-a22b",
                    OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1",
                    ENABLE_MODEL_REASONING=False,
                )
            )

        self.assertEqual(
            captured["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_create_llm_for_fireworks_reasoning_model_uses_registry_reasoning_effort(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="accounts/fireworks/models/deepseek-r1",
                    OPENAI_BASE_URL="https://api.fireworks.ai/inference/v1",
                    MODEL_REASONING_EFFORT="high",
                )
            )

        self.assertEqual(captured["reasoning_effort"], "high")

    def test_create_llm_for_mistral_reasoning_model_uses_registry_reasoning_effort(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="mistral-medium-3-5",
                    OPENAI_BASE_URL="https://api.mistral.ai/v1",
                    MODEL_REASONING_EFFORT="xhigh",
                )
            )

        self.assertEqual(captured["reasoning_effort"], "high")

    def test_create_llm_for_aihubmix_glm_reasoning_model_uses_registry_reasoning_effort(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="glm-5",
                    OPENAI_BASE_URL="https://aihubmix.com/v1",
                    MODEL_REASONING_EFFORT="medium",
                )
            )

        self.assertNotIn("reasoning", captured)
        self.assertNotIn("extra_body", captured)
        self.assertEqual(captured["reasoning_effort"], "medium")

    def test_create_llm_for_baai_minimax_reasoning_model_uses_registry_reasoning_effort(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="minimax-m3",
                    OPENAI_BASE_URL="https://api.b.ai/v1",
                    MODEL_REASONING_EFFORT="xhigh",
                )
            )

        self.assertNotIn("reasoning", captured)
        self.assertNotIn("extra_body", captured)
        self.assertEqual(captured["reasoning_effort"], "high")

    def test_create_llm_for_freetheai_gemini_uses_provider_level_reasoning(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="bbl/gemini-3.5-flash",
                    OPENAI_BASE_URL="https://api.freetheai.xyz/v1",
                    MODEL_REASONING_EFFORT="xhigh",
                    LLM_API_MODE="responses",
                )
            )

        self.assertEqual(captured["reasoning"], {"effort": "high", "summary": "auto"})
        self.assertNotIn("extra_body", captured)
        self.assertNotIn("reasoning_effort", captured)

    def test_create_llm_for_freetheai_kimi_uses_same_provider_level_reasoning(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="wsf/kimi-k2.6",
                    OPENAI_BASE_URL="https://api.freetheai.xyz/v1",
                    MODEL_REASONING_EFFORT="high",
                    LLM_API_MODE="responses",
                )
            )

        self.assertEqual(captured["reasoning"], {"effort": "high", "summary": "auto"})
        self.assertNotIn("extra_body", captured)
        self.assertNotIn("reasoning_effort", captured)

    def test_create_llm_for_modelscope_reasoning_model_uses_registry_reasoning_effort(self):
        for base_url in (
            "https://api-inference.modelscope.ai/v1",
            "https://api-inference.modelscope.cn/v1",
        ):
            with self.subTest(base_url=base_url):
                captured = {}

                class FakeChatOpenAI:
                    def __init__(self, **kwargs):
                        captured.update(kwargs)

                with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
                    create_llm(
                        self._make_config(
                            PROVIDER="openai",
                            OPENAI_API_KEY="sk-test",
                            OPENAI_MODEL="openai/gpt-oss-120b",
                            OPENAI_BASE_URL=base_url,
                            MODEL_REASONING_EFFORT="xhigh",
                        )
                    )

                self.assertNotIn("reasoning", captured)
                self.assertNotIn("extra_body", captured)
                self.assertEqual(captured["reasoning_effort"], "high")

    def test_create_llm_for_base_url_without_registry_entry_skips_reasoning_kwargs(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="openai/gpt-oss-120b",
                    OPENAI_BASE_URL="http://localhost:3002/v1",
                    MODEL_REASONING_EFFORT="high",
                )
            )

        self.assertNotIn("reasoning", captured)
        self.assertNotIn("extra_body", captured)
        self.assertNotIn("reasoning_effort", captured)

    def test_create_llm_for_reasoning_effort_none_skips_reasoning_kwargs(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="gpt-5-mini",
                    OPENAI_BASE_URL="https://api.openai.com/v1",
                    MODEL_REASONING_EFFORT="none",
                )
            )

        self.assertNotIn("reasoning", captured)
        self.assertNotIn("extra_body", captured)
        self.assertEqual(captured["reasoning_effort"], "none")

    def test_create_llm_for_openai_allows_disabling_reasoning_controls(self):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"langchain_openai": mock.Mock(ChatOpenAI=FakeChatOpenAI)}):
            create_llm(
                self._make_config(
                    PROVIDER="openai",
                    OPENAI_API_KEY="sk-test",
                    OPENAI_MODEL="gpt-5-mini",
                    OPENAI_BASE_URL="https://api.openai.com/v1",
                    ENABLE_MODEL_REASONING=False,
                )
            )

        self.assertNotIn("reasoning", captured)

    def test_build_initial_state_supports_text_and_image_attachments(self):
        state = build_initial_state(
            {
                "text": "Что на изображении?",
                "attachments": [
                    {
                        "id": "img-1",
                        "path": "D:/demo/sample.png",
                        "mime_type": "image/png",
                        "file_name": "sample.png",
                        "width": 32,
                        "height": 24,
                        "size_bytes": 2048,
                    }
                ],
            },
            session_id="session-1",
        )

        self.assertEqual(state["current_task"], "Что на изображении?")
        message = state["messages"][0]
        self.assertIsInstance(message, HumanMessage)
        self.assertIsInstance(message.content, list)
        self.assertEqual(message.content[0]["type"], "text")
        self.assertEqual(message.content[1]["type"], "image")
        self.assertEqual(message.content[1]["path"], "D:/demo/sample.png")

    def test_transcript_reducer_ignores_compaction_removals_and_deduplicates_snapshots(self):
        first_user = HumanMessage(content="Первый запрос", id="user-1")
        first_answer = AIMessage(content="Первый ответ", id="ai-1")
        repeated_answer = AIMessage(content="Исправленный ответ", id="ai-1")
        result = append_transcript_messages(
            [first_user, first_answer],
            [
                RemoveMessage(id="user-1"),
                repeated_answer,
                AIMessage(content="Новый ответ", id="ai-2"),
            ],
        )

        self.assertEqual([message.id for message in result], ["user-1", "ai-1", "ai-2"])
        self.assertEqual(result[1].content, "Исправленный ответ")
        self.assertEqual(result[0].content, "Первый запрос")

    def test_build_initial_state_accepts_image_only_request(self):
        state = build_initial_state(
            {
                "text": "",
                "attachments": [
                    {
                        "id": "img-1",
                        "path": "D:/demo/sample.png",
                        "mime_type": "image/png",
                    }
                ],
            },
            session_id="session-1",
        )

        self.assertEqual(state["current_task"], "Analyze the attached images.")
        message = state["messages"][0]
        self.assertIsInstance(message.content, list)
        self.assertEqual(len(message.content), 1)
        self.assertEqual(message.content[0]["type"], "image")

    def test_build_initial_state_keys_are_declared_in_agent_state(self):
        state = build_initial_state("Проверь задачу", session_id="session-1")
        declared_keys = set(AgentState.__required_keys__) | set(AgentState.__optional_keys__)

        self.assertTrue(set(state).issubset(declared_keys))
        self.assertNotIn("summary", AgentState.__required_keys__)
        self.assertNotIn("self_correction_retry_count", state)
        self.assertNotIn("self_correction_retry_turn_id", state)
        self.assertNotIn("self_correction_fingerprint_history", state)
        self.assertEqual(state["recovery_state"]["retry_count"], 0)
        self.assertEqual(state["recovery_state"]["retry_fingerprint_history"], [])
        self.assertEqual(state["recovery_state"]["last_reason"], "")
        self.assertEqual(state["transcript_messages"], state["messages"])

    def test_build_transcript_payload_parses_json_string_tool_args(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Сохрани файл"),
                    AIMessage(content="").model_copy(
                        update={
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "name": "write_file",
                                    "args": json.dumps(
                                        {"path": "notes.md", "content": "line 1\nline 2"},
                                        ensure_ascii=False,
                                    ),
                                }
                            ]
                        }
                    ),
                    ToolMessage(
                        tool_call_id="call_1",
                        name="write_file",
                        content="Success: File 'notes.md' saved (13 chars).",
                    ),
                ]
            }
        )

        tool_block = payload["turns"][0]["blocks"][0]["payload"]
        self.assertEqual(tool_block["name"], "write_file")
        self.assertEqual(tool_block["args"], {"path": "notes.md", "content": "line 1\nline 2"})
        self.assertEqual(tool_block["display"], "Writing file")
        self.assertEqual(tool_block["subtitle"], "notes.md")
        self.assertEqual(tool_block["raw_display"], "write_file(notes.md)")
        self.assertEqual(tool_block["args_state"], "complete")
        self.assertEqual(tool_block["display_state"], "finished")
        self.assertEqual(tool_block["phase"], "finished")
        self.assertEqual(tool_block["source_kind"], "tool")

    def test_build_transcript_payload_restores_user_image_attachments(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "Посмотри на картинку"},
                            {
                                "type": "image",
                                "path": "D:/demo/sample.png",
                                "mime_type": "image/png",
                                "file_name": "sample.png",
                                "attachment_id": "img-1",
                                "width": 64,
                                "height": 48,
                                "size_bytes": 4096,
                            },
                        ]
                    ),
                    AIMessage(content="Готово"),
                ]
            }
        )

        self.assertEqual(len(payload["turns"]), 1)
        turn = payload["turns"][0]
        self.assertEqual(turn["user_text"], "Посмотри на картинку")
        self.assertEqual(len(turn["attachments"]), 1)
        self.assertEqual(turn["attachments"][0]["id"], "img-1")
        self.assertEqual(turn["attachments"][0]["path"], "D:/demo/sample.png")

    def test_materialize_user_message_content_for_openai_uses_image_url_data_uri(self):
        temp_dir = self._workspace_tempdir()
        image_path = temp_dir / "sample.png"
        image_path.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WnSUs8AAAAASUVORK5CYII="
            )
        )

        content = [
            {"type": "text", "text": "Опиши изображение"},
            {"type": "image", "path": str(image_path), "mime_type": "image/png"},
        ]

        materialized = materialize_user_message_content_for_model(content, provider="openai")

        self.assertEqual(materialized[0], {"type": "text", "text": "Опиши изображение"})
        self.assertEqual(materialized[1]["type"], "image_url")
        self.assertTrue(materialized[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    async def test_openai_context_materializes_human_image_content_as_image_url_blocks(self):
        temp_dir = self._workspace_tempdir()
        image_path = temp_dir / "sample.png"
        image_path.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WnSUs8AAAAASUVORK5CYII="
            )
        )

        config = self._make_config()
        llm = OpenAIContentSafeFakeLLM([AIMessage(content="ok")])
        nodes = AgentNodes(
            config=config,
            llm=llm,
            tools=[],
            llm_with_tools=llm,
        )
        state = self._initial_state("Опиши изображение")
        context = nodes.context_builder.build(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Опиши изображение"},
                        {"type": "image", "path": str(image_path), "mime_type": "image/png"},
                    ]
                ),
            ],
            state,
            summary="",
            current_task="Опиши изображение",
            tools_available=False,
            active_tool_names=[],
            open_tool_issue=None,
            recovery_state=None,
        )

        response = await nodes._invoke_llm_with_retry(llm, context, state=state, node_name="agent")

        self.assertEqual(str(response.content), "ok")
        human_messages = [message for message in llm.invocations[0] if isinstance(message, HumanMessage)]
        self.assertTrue(human_messages)
        image_blocks = [block for block in human_messages[-1].content if isinstance(block, dict) and block.get("type") == "image_url"]
        self.assertEqual(len(image_blocks), 1)
        self.assertTrue(image_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_openai_normalize_system_prefix_merges_multiple_system_messages(self):
        config = self._make_config()
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[],
            llm_with_tools=FakeLLM([]),
        )

        normalized = nodes.context_builder.normalize_system_prefix(
            [
                HumanMessage(content="user-first"),
                SystemMessage(content="system-one"),
                HumanMessage(content="user-second"),
                SystemMessage(content="system-two"),
            ]
        )

        self.assertEqual(len([message for message in normalized if isinstance(message, SystemMessage)]), 1)
        self.assertIsInstance(normalized[0], SystemMessage)
        self.assertIn("system-one", normalized[0].content)
        self.assertIn("system-two", normalized[0].content)
        self.assertIsInstance(normalized[1], HumanMessage)
        self.assertEqual(normalized[1].content, "user-first")

    async def test_create_checkpoint_runtime_uses_sqlite_backend_when_available(self):
        self._require_sqlite_filesystem()
        tmp = self._workspace_tempdir()
        db_path = tmp / "checkpoints.sqlite"
        runtime = await create_checkpoint_runtime(
            self._make_config(CHECKPOINT_BACKEND="sqlite", CHECKPOINT_SQLITE_PATH=db_path)
        )
        try:
            self.assertEqual(runtime.resolved_backend, "sqlite")
            self.assertEqual(Path(runtime.target), db_path.resolve())
            self.assertTrue(db_path.exists())
        finally:
            await runtime.aclose()

    async def test_sqlite_checkpointer_persists_state_across_recompiled_app(self):
        self._require_sqlite_filesystem()
        tmp = self._workspace_tempdir()
        db_path = tmp / "persist.sqlite"
        config = self._make_config(CHECKPOINT_BACKEND="sqlite", CHECKPOINT_SQLITE_PATH=db_path)
        thread_config = {"configurable": {"thread_id": "persist-thread"}, "recursion_limit": 24}

        runtime1 = await create_checkpoint_runtime(config)
        app1 = create_agent_workflow(
            AgentNodes(
                config=config,
                llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: ok\nNEXT_STEP: NONE")]),
                tools=[],
                llm_with_tools=FakeLLM([AIMessage(content="Первый ответ.")]),
            ),
            config,
            tools_enabled=False,
        ).compile(checkpointer=runtime1.checkpointer)
        await app1.ainvoke(self._initial_state("Первая задача"), config=thread_config)
        await runtime1.aclose()

        runtime2 = await create_checkpoint_runtime(config)
        try:
            app2 = create_agent_workflow(
                AgentNodes(
                    config=config,
                    llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: ok\nNEXT_STEP: NONE")]),
                    tools=[],
                    llm_with_tools=FakeLLM([AIMessage(content="Второй ответ.")]),
                ),
                config,
                tools_enabled=False,
            ).compile(checkpointer=runtime2.checkpointer)
            saved_state = await app2.aget_state({"configurable": {"thread_id": "persist-thread"}})
            saved_messages = saved_state.values["messages"]
            self.assertTrue(any(isinstance(msg, HumanMessage) and msg.content == "Первая задача" for msg in saved_messages))
            self.assertTrue(
                any(
                    isinstance(msg, AIMessage)
                    and "STATUS: FINISHED" in str(msg.content)
                    and "REASON: ok" in str(msg.content)
                    for msg in saved_messages
                )
            )
        finally:
            await runtime2.aclose()

    async def test_approval_interrupt_requires_resume_before_mutating_tool_runs(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: done\nNEXT_STEP: NONE")]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-1"}]),
                    AIMessage(content="Готово."),
                ]
            ),
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-thread"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(tool.calls, [])

        resumed = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)
        self.assertEqual(tool.calls, [{"action": "apply"}])
        self.assertEqual(resumed["messages"][-1].content, "Готово.")

    async def test_destructive_cli_exec_interrupts_before_execution(self):
        config = self._make_config(ENABLE_APPROVALS=True, ENABLE_SHELL_TOOL=True)
        tool = FakeTool("cli_exec", "Success: command completed.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: done\nNEXT_STEP: NONE")]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "cli_exec",
                                "args": {"command": "Remove-Item -LiteralPath demo.txt -Force"},
                                "id": "tc-cli-delete",
                            }
                        ],
                    ),
                    AIMessage(content="Команда подтверждена и выполнена."),
                ]
            ),
            tool_metadata={
                "cli_exec": ToolMetadata(
                    name="cli_exec",
                    mutating=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-cli-thread"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Удалить файл через shell"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(tool.calls, [])

        interrupt_entries = interrupted["__interrupt__"]
        interrupt_value = getattr(interrupt_entries[0], "value", interrupt_entries[0])
        self.assertEqual(interrupt_value["tools"][0]["name"], "cli_exec")
        self.assertTrue(interrupt_value["tools"][0]["policy"]["destructive"])
        self.assertTrue(interrupt_value["tools"][0]["policy"]["requires_approval"])

        resumed = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)
        self.assertEqual(tool.calls, [{"command": "Remove-Item -LiteralPath demo.txt -Force"}])
        self.assertEqual(resumed["messages"][-1].content, "Команда подтверждена и выполнена.")

    async def test_safe_delete_file_interrupts_before_execution(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("safe_delete_file", "Success: file deleted.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: done\nNEXT_STEP: NONE")]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "safe_delete_file",
                                "args": {"path": "demo.txt"},
                                "id": "tc-delete-file",
                            }
                        ],
                    ),
                    AIMessage(content="Файл удален после подтверждения."),
                ]
            ),
            tool_metadata={
                "safe_delete_file": ToolMetadata(
                    name="safe_delete_file",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-safe-delete"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Удали файл"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(tool.calls, [])

        resumed = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)
        self.assertEqual(tool.calls, [{"path": "demo.txt"}])
        self.assertEqual(resumed["messages"][-1].content, "Файл удален после подтверждения.")

    async def test_mcp_execution_tool_interrupts_before_execution(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("terminal:run_command", "Success: command completed.")
        tool.metadata = {"executionHint": True}
        mcp_metadata = ToolRegistry._infer_mcp_metadata(tool)
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: done\nNEXT_STEP: NONE")]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "terminal:run_command",
                                "args": {"command": "git status"},
                                "id": "tc-mcp-run",
                            }
                        ],
                    ),
                    AIMessage(content="MCP команда выполнена после подтверждения."),
                ]
            ),
            tool_metadata={"terminal:run_command": mcp_metadata},
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-mcp-exec"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Запусти MCP команду"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(tool.calls, [])

        interrupt_entries = interrupted["__interrupt__"]
        interrupt_value = getattr(interrupt_entries[0], "value", interrupt_entries[0])
        self.assertEqual(interrupt_value["tools"][0]["name"], "terminal:run_command")
        self.assertTrue(interrupt_value["tools"][0]["policy"]["requires_approval"])

        resumed = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)
        self.assertEqual(tool.calls, [{"command": "git status"}])
        self.assertEqual(resumed["messages"][-1].content, "MCP команда выполнена после подтверждения.")

    async def test_regular_edit_requires_resume_before_execution(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        temp_dir = self._workspace_tempdir()

        def _edit_result(args):
            (temp_dir / args["path"]).write_text("b", encoding="utf-8")
            return "Success: File edited."

        with mock.patch("tools.filesystem._WORKING_DIRECTORY", temp_dir):
            tool = FakeTool("edit_file", _edit_result)
            nodes = AgentNodes(
                config=config,
                llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: done\nNEXT_STEP: NONE")]),
                tools=[tool],
                llm_with_tools=FakeLLM(
                    [
                        AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"path": "demo.txt", "old_string": "a", "new_string": "b"}, "id": "tc-edit-plain"}]),
                        AIMessage(content="Готово без дополнительного approval."),
                    ]
                ),
                tool_metadata={
                    "edit_file": ToolMetadata(
                        name="edit_file",
                        mutating=True,
                        requires_approval=False,
                    )
                },
            )
            app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())

            thread_config = {"configurable": {"thread_id": "plain-edit-approval"}, "recursion_limit": 36}

            interrupted = await app.ainvoke(
                self._initial_state("Исправь файл"),
                config=thread_config,
            )

            self.assertIn("__interrupt__", interrupted)
            self.assertEqual(tool.calls, [])

            resumed = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)

        self.assertEqual(tool.calls, [{"path": "demo.txt", "old_string": "a", "new_string": "b"}])
        self.assertEqual(resumed["turn_outcome"], "finish_turn")
        self.assertIn("без дополнительного approval", str(resumed["messages"][-1].content).lower())

    async def test_mutating_non_destructive_tool_error_enters_recovery_after_approval(self):
        config = self._make_config(ENABLE_APPROVALS=True)

        def _edit_result(args):
            if args.get("old_string") == "bad":
                return "ERROR[VALIDATION]: Could not find a match for 'old_string'."
            return "Success: File edited."

        edit_tool = FakeTool("edit_file", _edit_result)
        read_tool = FakeTool("read_file", "import sys\nimport logging")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: done\nNEXT_STEP: NONE")]),
            tools=[edit_tool, read_tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"path": "demo.txt", "old_string": "bad", "new_string": "good"}, "id": "tc-edit-1"}]),
                    AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "demo.txt"}, "id": "tc-read-1"}]),
                    AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"path": "demo.txt", "old_string": "import sys", "new_string": "import sys\nimport json"}, "id": "tc-edit-2"}]),
                    AIMessage(content="Исправление завершено после recovery."),
                ]
            ),
            tool_metadata={
                "edit_file": ToolMetadata(name="edit_file", mutating=True, destructive=False, requires_approval=False),
                "read_file": ToolMetadata(name="read_file", read_only=True),
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())

        thread_config = {"configurable": {"thread_id": "mutating-recovery-approval"}, "recursion_limit": 64}

        interrupted = await app.ainvoke(
            self._initial_state("Исправь файл"),
            config=thread_config,
        )

        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(edit_tool.calls, [])
        self.assertEqual(read_tool.calls, [])

        second_interrupt = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)

        self.assertIn("__interrupt__", second_interrupt)
        self.assertEqual(len(edit_tool.calls), 1)
        self.assertEqual(len(read_tool.calls), 1)

        result = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)

        self.assertEqual(len(edit_tool.calls), 2)
        self.assertEqual(len(read_tool.calls), 1)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])

    async def test_invoke_llm_with_retry_normalizes_context_once_per_attempt(self):
        config = self._make_config(MAX_RETRIES=2, RETRY_DELAY=0)
        llm = FakeLLM([RuntimeError("temporary"), AIMessage(content="ok")])
        nodes = AgentNodes(
            config=config,
            llm=llm,
            tools=[],
            llm_with_tools=llm,
        )
        context = [HumanMessage(content="Проверь задачу")]

        with mock.patch.object(
            AgentNodes,
            "_normalize_system_prefix_for_provider",
            autospec=True,
            side_effect=lambda _self, payload: list(payload),
        ) as normalize_mock:
            response = await nodes._invoke_llm_with_retry(llm, context, state=self._initial_state(), node_name="agent")

        self.assertEqual(str(response.content), "ok")
        self.assertEqual(normalize_mock.call_count, 2)

    async def test_invoke_llm_with_retry_retries_transient_errors_with_fixed_backoff_and_status(self):
        config = self._make_config(MAX_RETRIES=1, RETRY_DELAY=0)
        llm = FakeLLM(
            [
                RuntimeError("429 Too Many Requests"),
                RuntimeError("503 Service Unavailable"),
                RuntimeError("connection reset by peer"),
                AIMessage(content="ok"),
            ]
        )
        nodes = AgentNodes(config=config, llm=llm, tools=[], llm_with_tools=llm)
        writer = mock.Mock()

        with (
            mock.patch("core.nodes.llm.asyncio.sleep", new=mock.AsyncMock()) as sleep_mock,
            mock.patch("core.nodes.llm.get_stream_writer", return_value=writer),
        ):
            response = await nodes._invoke_llm_with_retry(
                llm,
                [HumanMessage(content="Проверь задачу")],
                state=self._initial_state(),
                node_name="agent",
            )

        self.assertEqual(str(response.content), "ok")
        self.assertEqual([call.args[0] for call in sleep_mock.await_args_list], [2, 4, 8])
        self.assertEqual(
            [call.args[0]["label"] for call in writer.call_args_list],
            ["Retrying provider request... 1/3", "Retrying provider request... 2/3", "Retrying provider request... 3/3"],
        )

    async def test_invoke_llm_with_retry_logs_safe_provider_error_details(self):
        config = self._make_config(MAX_RETRIES=1, RETRY_DELAY=0)
        request = SimpleNamespace(url="https://sk-secret@api.example.test/v1/chat?api_key=secret")
        response = SimpleNamespace(
            status_code=503,
            request=request,
            headers={"x-request-id": "req-safe-123"},
        )
        error = RuntimeError(
            "Service Unavailable at https://sk-message@api.example.test/v1/chat?api_key=message-secret"
        )
        error.response = response
        llm = FakeLLM([error, AIMessage(content="ok")])
        nodes = AgentNodes(config=config, llm=llm, tools=[], llm_with_tools=llm)

        with (
            mock.patch("core.nodes.llm.asyncio.sleep", new=mock.AsyncMock()),
            mock.patch("core.nodes.llm.get_stream_writer", return_value=mock.Mock()),
            self.assertLogs("agent", level=logging.WARNING) as captured,
        ):
            await nodes._invoke_llm_with_retry(
                llm,
                [HumanMessage(content="Проверь задачу")],
                state=self._initial_state(),
                node_name="agent",
            )

        log_text = "\n".join(captured.output)
        self.assertIn("type=RuntimeError", log_text)
        self.assertIn("http_status=503", log_text)
        self.assertIn("url_host=api.example.test", log_text)
        self.assertIn("request_error_id=req-safe-123", log_text)
        self.assertNotIn("sk-secret", log_text)
        self.assertNotIn("api_key=secret", log_text)
        self.assertNotIn("sk-message", log_text)
        self.assertNotIn("message-secret", log_text)
        self.assertNotIn("/v1/chat", log_text)

    def test_describe_provider_error_extracts_error_id_from_message(self):
        details = describe_provider_error(RuntimeError("capacity unavailable. Error id: olm-443aa38422e5"))

        self.assertEqual(details["error_type"], "RuntimeError")
        self.assertEqual(details["request_error_id"], "olm-443aa38422e5")
        self.assertIsNone(details["http_status"])
        self.assertEqual(details["url_host"], "")

    def test_describe_provider_error_extracts_http_status_from_message(self):
        details = describe_provider_error(RuntimeError("503 Service Unavailable"))

        self.assertEqual(details["http_status"], 503)

    async def test_invoke_llm_with_retry_does_not_retry_permanent_http_client_errors(self):
        config = self._make_config(MAX_RETRIES=3, RETRY_DELAY=0)
        for status in (400, 401, 403, 404):
            error = RuntimeError(f"HTTP {status} client error")
            error.status_code = status
            llm = FakeLLM([error, AIMessage(content="must not be used")])
            nodes = AgentNodes(config=config, llm=llm, tools=[], llm_with_tools=llm)

            with mock.patch("core.nodes.llm.asyncio.sleep", new=mock.AsyncMock()) as sleep_mock:
                with self.assertRaises(RuntimeError):
                    await nodes._invoke_llm_with_retry(
                        llm,
                        [HumanMessage(content="Проверь задачу")],
                        state=self._initial_state(),
                        node_name="agent",
                    )

            self.assertEqual(len(llm.invocations), 1)
            sleep_mock.assert_not_awaited()

    async def test_invoke_llm_with_retry_does_not_restart_exhausted_key_pool(self):
        config = self._make_config(MAX_RETRIES=3, RETRY_DELAY=0)
        cause = RuntimeError("429 Too Many Requests")
        cause.status_code = 429
        exhausted = ApiKeyRotationExhaustedError(
            "Все API-ключи исчерпаны. Последняя ошибка: HTTP 429",
            error_kind="rate_limit",
            keys_tried=3,
            pool_size=3,
        )
        exhausted.__cause__ = cause
        llm = FakeLLM([exhausted, AIMessage(content="must not be used")])
        nodes = AgentNodes(config=config, llm=llm, tools=[], llm_with_tools=llm)

        with mock.patch("core.nodes.llm.asyncio.sleep", new=mock.AsyncMock()) as sleep_mock:
            with self.assertRaises(ApiKeyRotationExhaustedError):
                await nodes._invoke_llm_with_retry(
                    llm,
                    [HumanMessage(content="Проверь задачу")],
                    state=self._initial_state(),
                    node_name="agent",
                )

        self.assertEqual(len(llm.invocations), 1)
        sleep_mock.assert_not_awaited()

    async def test_invoke_llm_with_retry_does_not_retry_closed_client_cause(self):
        config = self._make_config(MAX_RETRIES=3, RETRY_DELAY=0)
        error = RuntimeError("Connection error.")
        error.__cause__ = RuntimeError("Cannot send a request, as the client has been closed.")
        llm = FakeLLM([error, AIMessage(content="must not be used")])
        nodes = AgentNodes(config=config, llm=llm, tools=[], llm_with_tools=llm)

        with (
            mock.patch("core.nodes.llm.asyncio.sleep", new=mock.AsyncMock()) as sleep_mock,
            mock.patch("core.nodes.llm.get_stream_writer", return_value=mock.Mock()) as writer_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "Connection error"):
                await nodes._invoke_llm_with_retry(
                    llm,
                    [HumanMessage(content="Проверь задачу")],
                    state=self._initial_state(),
                    node_name="agent",
                )

        self.assertEqual(len(llm.invocations), 1)
        sleep_mock.assert_not_awaited()
        writer_mock.return_value.assert_not_called()

    async def test_invoke_llm_with_retry_marks_exhausted_transient_error(self):
        config = self._make_config(MAX_RETRIES=3, RETRY_DELAY=0)
        errors = [RuntimeError("connection reset by peer") for _ in range(4)]
        llm = FakeLLM(errors)
        nodes = AgentNodes(config=config, llm=llm, tools=[], llm_with_tools=llm)

        with mock.patch("core.nodes.llm.asyncio.sleep", new=mock.AsyncMock()):
            with self.assertRaises(RuntimeError) as captured:
                await nodes._invoke_llm_with_retry(
                    llm,
                    [HumanMessage(content="Проверь задачу")],
                    state=self._initial_state(),
                    node_name="agent",
                )

        self.assertTrue(getattr(captured.exception, "llm_retry_exhausted", False))
        self.assertEqual(len(llm.invocations), 4)

    async def test_invoke_llm_with_retry_does_not_use_transient_backoff_for_non_transient_error(self):
        config = self._make_config(MAX_RETRIES=1, RETRY_DELAY=0)
        llm = FakeLLM([RuntimeError("invalid request payload")])
        nodes = AgentNodes(config=config, llm=llm, tools=[], llm_with_tools=llm)
        writer = mock.Mock()

        with (
            mock.patch("core.nodes.llm.asyncio.sleep", new=mock.AsyncMock()) as sleep_mock,
            mock.patch("core.nodes.llm.get_stream_writer", return_value=writer),
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid request payload"):
                await nodes._invoke_llm_with_retry(
                    llm,
                    [HumanMessage(content="Проверь задачу")],
                    state=self._initial_state(),
                    node_name="agent",
                )

        sleep_mock.assert_not_awaited()
        writer.assert_not_called()

    async def test_invoke_llm_with_retry_applies_auto_tool_choice_fallback_warning_once(self):
        config = self._make_config(MAX_RETRIES=3, RETRY_DELAY=0)
        fallback_llm = FakeLLM([RuntimeError("temporary"), AIMessage(content="ok")])
        tools_llm = FakeLLM([RuntimeError("auto tool choice requires explicit support")])
        nodes = AgentNodes(
            config=config,
            llm=fallback_llm,
            tools=[],
            llm_with_tools=tools_llm,
        )
        context = [SystemMessage(content="Base system prompt"), HumanMessage(content="Проверь задачу")]

        response = await nodes._invoke_llm_with_retry(tools_llm, context, state=self._initial_state(), node_name="agent")

        self.assertEqual(str(response.content), "ok")
        self.assertEqual(len(fallback_llm.invocations), 2)
        for invocation in fallback_llm.invocations:
            self.assertEqual(str(invocation[0].content).count("WARNING: Tools are disabled due to server configuration error."), 1)

    async def test_approval_rejection_returns_access_denied_without_tool_execution(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: blocker reported\nNEXT_STEP: NONE")]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-2"}]),
                    AIMessage(content="Не удалось выполнить действие без подтверждения."),
                ]
            ),
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-thread-2"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        resumed = await app.ainvoke(Command(resume={"approved": False}), config=thread_config)

        self.assertEqual(tool.calls, [])
        tool_messages = [msg for msg in resumed["messages"] if isinstance(msg, ToolMessage)]
        self.assertTrue(tool_messages)
        self.assertIn("ACCESS_DENIED", str(tool_messages[-1].content))
        self.assertIsNone(resumed["open_tool_issue"])

    def test_tool_calls_require_approval_skips_invalid_write_file_without_content(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[write_file_tool],
            llm_with_tools=FakeLLM([]),
            tool_metadata={
                "write_file": ToolMetadata(
                    name="write_file",
                    mutating=True,
                    requires_approval=True,
                )
            },
        )

        required = nodes.tool_calls_require_approval(
            [{"name": "write_file", "args": {"path": "REFACTORING_SUGGESTIONS.md"}, "id": "tc-write"}]
        )

        self.assertFalse(required)

    async def test_approval_node_skips_invalid_write_file_without_content(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[write_file_tool],
            llm_with_tools=FakeLLM([]),
            tool_metadata={
                "write_file": ToolMetadata(
                    name="write_file",
                    mutating=True,
                    requires_approval=True,
                )
            },
        )

        result = await nodes.approval_node(
            {
                **self._initial_state("Создай файл"),
                "messages": [
                    AIMessage(
                        content="Запишу файл.",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"path": "REFACTORING_SUGGESTIONS.md"},
                                "id": "tc-write",
                            }
                        ],
                    )
                ],
            }
        )

        self.assertEqual(result, {"pending_approval": None})

    async def test_approval_rejection_blocks_followup_tool_calls_in_same_turn(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        denied_tool = FakeTool("danger_tool", "Изменение применено.")
        fallback_tool = FakeTool("write_file", "Файл записан.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: blocker reported\nNEXT_STEP: NONE")]),
            tools=[denied_tool, fallback_tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-3"}]),
                    AIMessage(
                        content="Сохраню результат в другой файл.",
                        tool_calls=[{"name": "write_file", "args": {"path": "alt.md", "content": "x"}, "id": "tc-4"}],
                    ),
                ]
            ),
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                ),
                "write_file": ToolMetadata(
                    name="write_file",
                    mutating=True,
                    requires_approval=True,
                ),
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-thread-3"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        resumed = await app.ainvoke(Command(resume={"approved": False}), config=thread_config)

        self.assertEqual(denied_tool.calls, [])
        self.assertEqual(fallback_tool.calls, [])
        self.assertIsInstance(resumed["messages"][-1], AIMessage)
        self.assertIn("declined", str(resumed["messages"][-1].content).lower())
        self.assertIsNone(resumed["open_tool_issue"])

    async def test_approval_rejection_resume_keeps_provider_safe_order(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM(
                [
                    AIMessage(
                        content=(
                            "STATUS: INCOMPLETE\n"
                            "REASON: The assistant is waiting for the user after denial.\n"
                            "NEXT_STEP: wait for the next instruction\n"
                            "CONTROL: FINISH_TURN"
                        )
                    )
                ]
            ),
            tools=[tool],
            llm_with_tools=ProviderSafeFakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-safe"}]),
                    AIMessage(content="Действие не выполнено: вы выбрали Нет для необратимой операции."),
                ]
            ),
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-thread-safe"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        resumed = await app.ainvoke(Command(resume={"approved": False}), config=thread_config)

        self.assertEqual(tool.calls, [])
        self.assertEqual(resumed["turn_outcome"], "finish_turn")
        self.assertIn("declined", str(resumed["messages"][-1].content).lower())

    async def test_approval_rejection_finishes_turn_without_secondary_verifier(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        agent_llm = ProviderSafeFakeLLM(
            [
                AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-mal"}]),
                AIMessage(content="Не сделал, потому что вы выбрали Нет. Жду следующую инструкцию."),
            ]
        )
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-thread-malformed"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        resumed = await app.ainvoke(Command(resume={"approved": False}), config=thread_config)

        self.assertEqual(tool.calls, [])
        self.assertEqual(resumed["turn_outcome"], "finish_turn")
        self.assertIsNone(resumed["open_tool_issue"])
        self.assertEqual(len(agent_llm.invocations), 1)
        final_text = str(resumed["messages"][-1].content).lower()
        self.assertIn("declined", final_text)

    def test_sanitize_messages_for_model_remaps_non_compliant_tool_call_ids(self):
        config = self._make_config()
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[],
            llm_with_tools=FakeLLM([]),
        )
        source_id = "chatcmpl-tool-9d8dc0bbe38d6202"
        sanitized = nodes._sanitize_messages_for_model(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": source_id,
                            "name": "read_file",
                            "args": {"path": "README.md"},
                        }
                    ],
                ),
                ToolMessage(
                    tool_call_id=source_id,
                    name="read_file",
                    content="ok",
                ),
            ]
        )

        self.assertEqual(len(sanitized), 2)
        self.assertIsInstance(sanitized[0], AIMessage)
        self.assertIsInstance(sanitized[1], ToolMessage)
        remapped_id = sanitized[0].tool_calls[0]["id"]
        self.assertRegex(remapped_id, r"^[A-Za-z0-9]{9}$")
        self.assertEqual(sanitized[1].tool_call_id, remapped_id)

    def test_anthropic_context_preserves_provider_tool_use_ids_for_tool_results(self):
        config = self._make_config(
            PROVIDER="anthropic",
            ANTHROPIC_API_KEY="test-key",
            ANTHROPIC_MODEL="claude-haiku-4-5-20251001",
        )
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[],
            llm_with_tools=FakeLLM([]),
        )
        source_id = "toolu_016HR4mVCZZjkeCPZvWq5Htu"
        sanitized = nodes._sanitize_messages_for_model(
            [
                AIMessage(
                    content=[
                        {
                            "type": "tool_use",
                            "id": source_id,
                            "name": "list_directory",
                            "input": {"path": "."},
                        }
                    ],
                    tool_calls=[{"id": source_id, "name": "list_directory", "args": {"path": "."}}],
                ),
                ToolMessage(tool_call_id=source_id, name="list_directory", content="index.html found"),
            ]
        )

        ai_message, tool_message = sanitized
        self.assertEqual(ai_message.tool_calls[0]["id"], source_id)
        self.assertEqual(ai_message.content[0]["id"], source_id)
        self.assertEqual(tool_message.tool_call_id, source_id)

    async def test_openai_context_stringifies_assistant_content_lists_before_invoke(self):
        config = self._make_config()
        llm = OpenAIContentSafeFakeLLM([AIMessage(content="ok")])
        nodes = AgentNodes(
            config=config,
            llm=llm,
            tools=[],
            llm_with_tools=llm,
        )
        state = self._initial_state("Продолжай анализ")
        context = nodes.context_builder.build(
            [
                HumanMessage(content="Проверь файл"),
                AIMessage(content=["Промежуточный ", "ответ."]),
                HumanMessage(content="Продолжай анализ"),
            ],
            state,
            summary="",
            current_task="Продолжай анализ",
            tools_available=False,
            active_tool_names=[],
            open_tool_issue=None,
            recovery_state=None,
        )

        response = await nodes._invoke_llm_with_retry(llm, context, state=state, node_name="agent")

        self.assertEqual(str(response.content), "ok")
        self.assertTrue(any(isinstance(message, AIMessage) and message.content == "Промежуточный ответ." for message in llm.invocations[0]))

    async def test_tools_node_marks_approval_denied_as_open_issue_before_agent_ack(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM([]),
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )

        result = await nodes.tools_node(
            {
                **self._initial_state("Сделай изменение"),
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-issue"}],
                    )
                ],
                "pending_approval": {
                    "approved": False,
                    "decision": {"approved": False},
                    "tool_call_ids": ["tc-issue"],
                    "tool_names": ["danger_tool"],
                },
            }
        )

        self.assertEqual(result["open_tool_issue"]["kind"], "approval_denied")

    async def test_tools_node_preserves_parallel_results_when_one_tool_processing_crashes(self):
        config = self._make_config()
        read_tool = FakeTool("read_file", "ok")
        list_tool = FakeTool("list_directory", "ok")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[read_tool, list_tool],
            llm_with_tools=FakeLLM([]),
            tool_metadata={
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "list_directory": ToolMetadata(name="list_directory", read_only=True),
            },
        )

        async def fake_process(_self, tool_call, recent_calls, state, approval_state, current_turn_id, allowed_tool_names=None):
            if tool_call["id"] == "tc-bad":
                raise RuntimeError("boom")
            return (
                ToolMessage(tool_call_id=tool_call["id"], name=tool_call["name"], content="Success: ok"),
                False,
                None,
            )

        state = {
            **self._initial_state("Проверь файлы"),
            "messages": [
                HumanMessage(content="Проверь файлы"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read_file", "args": {"path": "README.md"}, "id": "tc-good"},
                        {"name": "list_directory", "args": {"path": "."}, "id": "tc-bad"},
                    ],
                ),
            ],
        }

        with mock.patch.object(AgentNodes, "_process_tool_call", new=fake_process):
            result = await nodes.tools_node(state)

        self.assertEqual(result["turn_outcome"], "run_tools")
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual({message.tool_call_id for message in result["messages"]}, {"tc-good", "tc-bad"})
        failed = next(message for message in result["messages"] if message.tool_call_id == "tc-bad")
        self.assertIn("Unhandled exception while executing 'list_directory': boom", str(failed.content))
        self.assertTrue(result["last_tool_error"])
        self.assertIsNone(result["open_tool_issue"])

    async def test_tools_node_resets_retry_state_after_successful_result(self):
        tool = FakeTool("read_file", "Success: file contents")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM([]),
        )

        state = self._initial_state("Проверь файл")
        state["recovery_state"]["retry_count"] = 2
        state["recovery_state"]["retry_fingerprint_history"] = ["fp-2"]
        state["recovery_state"]["last_reason"] = "recover_llm_replan"
        state["recovery_state"]["active_issue"] = {"summary": "old issue"}
        state["recovery_state"]["active_strategy"] = {"strategy": "llm_replan"}
        state["recovery_state"]["llm_replan_attempted_for"] = ["fp-2"]
        state["messages"] = [
            HumanMessage(content="Проверь файл"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"path": "demo.txt"}, "id": "tc-ok"}],
            ),
        ]

        result = await nodes.tools_node(state)

        self.assertNotIn("self_correction_retry_count", result)
        self.assertNotIn("self_correction_retry_turn_id", result)
        self.assertNotIn("self_correction_fingerprint_history", result)
        self.assertEqual(result["recovery_state"]["retry_count"], 0)
        self.assertEqual(result["recovery_state"]["retry_fingerprint_history"], [])
        self.assertEqual(result["recovery_state"]["last_reason"], "")
        self.assertIsNone(result["open_tool_issue"])
        self.assertIsNone(result["recovery_state"]["active_strategy"])
        self.assertEqual(result["recovery_state"]["llm_replan_attempted_for"], [])
        self.assertEqual(result["recovery_state"]["last_successful_evidence"], "Success: file contents")

    async def test_recovery_node_does_not_stop_early_at_two_retries_when_budget_allows_more(self):
        agent_llm = FakeLLM([])
        read_tool = FakeTool("read_file", "ignored")
        nodes = AgentNodes(
            config=self._make_config(SELF_CORRECTION_RETRY_LIMIT=8),
            llm=agent_llm,
            tools=[read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={"read_file": ToolMetadata(name="read_file", read_only=True)},
        )

        state = self._initial_state("Прочитай README.md")
        state["open_tool_issue"] = {
            "turn_id": 1,
            "kind": "protocol_error",
            "summary": "Malformed tool payload.",
            "tool_names": ["read_file"],
            "tool_args": {"path": "README.md"},
            "source": "agent",
            "error_type": "VALIDATION",
            "fingerprint": "fp-read",
            "progress_fingerprint": "fp-read",
            "details": {"protocol_reason": "tool_protocol_error"},
        }
        state["recovery_state"] = {
            "turn_id": 1,
            "retry_count": 2,
            "retry_fingerprint_history": ["fp-read"],
            "last_reason": "recover_llm_replan",
            "active_issue": dict(state["open_tool_issue"]),
            "active_strategy": {"strategy": "llm_replan", "progress_fingerprint": "fp-read"},
            "strategy_queue": [],
            "attempts_by_strategy": {"fp-read::llm_replan": 1},
            "progress_markers": ["fp-read"],
            "last_successful_evidence": "",
            "external_blocker": None,
            "llm_replan_attempted_for": ["fp-read"],
        }

        result = await nodes.recovery_node(state)

        self.assertEqual(result["turn_outcome"], "recover_agent")
        self.assertNotIn("self_correction_retry_count", result)
        self.assertEqual(result["recovery_state"]["retry_count"], 3)
        self.assertIsNotNone(result["open_tool_issue"])
        self.assertEqual(result["recovery_state"]["active_strategy"]["strategy"], "llm_replan")

    async def test_recovery_node_propagates_continue_agent_when_tool_result_ready(self):
        """Recovery with no open issue and a trailing ToolMessage must yield continue_agent
        so route_after_recovery sends the turn back to the agent instead of END."""
        agent_llm = FakeLLM([])
        read_tool = FakeTool("read_file", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[read_tool],
            llm_with_tools=agent_llm,
        )

        state = self._initial_state("Прочитай README.md")
        state["messages"] = [
            HumanMessage(content="Прочитай README.md"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "tc-ok"}],
            ),
            ToolMessage(tool_call_id="tc-ok", name="read_file", content="Success: file contents"),
        ]

        result = await nodes.recovery_node(state)

        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_CONTINUE_AGENT)
        self.assertIsNone(result["open_tool_issue"])

    async def test_run_logger_writes_structured_tool_failure_event(self):
        tmp = self._workspace_tempdir()
        logger = JsonlRunLogger(tmp)
        failing_tool = FakeTool("danger_tool", "ERROR[EXECUTION]: boom")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[failing_tool],
            llm_with_tools=FakeLLM([]),
            run_logger=logger,
        )

        await nodes.tools_node(
            {
                **self._initial_state("Почини ошибку", session_id="session-log", run_id="run-log"),
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "danger_tool", "args": {"action": "x"}, "id": "tc-log"}],
                    )
                ],
            }
        )

        log_path = logger.file_path_for("session-log")
        records = [json.loads(line) for line in log_path.read_text("utf-8").splitlines()]
        end_records = [record for record in records if record["event"] == "tool_call_end"]
        self.assertTrue(end_records)
        self.assertEqual(end_records[-1]["result"]["ok"], False)
        self.assertEqual(end_records[-1]["result"]["error_type"], "EXECUTION")
        self.assertIn("summary", end_records[-1]["result"])
        self.assertNotIn("raw", end_records[-1]["result"])

    def test_should_summarize_skips_small_overflow_when_summary_already_exists(self):
        messages = [
            HumanMessage(content="A" * 8000),
            AIMessage(content="B" * 8200),
            HumanMessage(content="коротко"),
            AIMessage(content="ok"),
            HumanMessage(content="ещё"),
            AIMessage(content="ok"),
        ]

        self.assertFalse(
            should_summarize(
                messages,
                threshold=8000,
                keep_last=4,
                has_summary=True,
            )
        )

    def test_truncate_summary_to_token_budget_keeps_leading_memory_items(self):
        summary = "\n".join(f"- fact {index}: keep the audit going with detail {index}" for index in range(40))

        truncated = truncate_summary_to_token_budget(summary, 120)

        self.assertLessEqual(estimate_summary_tokens(truncated), 120)
        self.assertTrue(truncated.startswith("- fact 0:"))
        self.assertTrue(truncated.endswith("... [memory truncated]"))
        self.assertNotIn("- fact 39:", truncated)

    def test_truncate_summary_to_token_budget_keeps_memory_within_budget(self):
        summary = "- single oversized memory item: " + "деталь конфигурации " * 400

        truncated = truncate_summary_to_token_budget(summary, 120)

        self.assertLessEqual(estimate_summary_tokens(truncated), 120)
        self.assertIn("single oversized memory item", truncated)
        self.assertTrue(truncated.endswith("... [memory truncated]"))

    def test_truncate_summary_to_token_budget_keeps_memory_within_budget_unchanged(self):
        summary = "- short memory"

        self.assertEqual(truncate_summary_to_token_budget(summary, 120), summary)
        self.assertEqual(truncate_summary_to_token_budget(summary, 0), summary)

    def test_summary_progress_ratio_excludes_fixed_overhead(self):
        trigger = summary_trigger_tokens(8000, has_summary=True)
        baseline = 5000

        just_compacted = summary_progress_ratio(
            baseline + 400,
            threshold=8000,
            baseline_tokens=baseline,
            has_summary=True,
        )
        at_trigger = summary_progress_ratio(
            trigger,
            threshold=8000,
            baseline_tokens=baseline,
            has_summary=True,
        )

        self.assertGreater(trigger, 8000)
        self.assertLess(just_compacted, 0.2)
        self.assertEqual(at_trigger, 1.0)
        self.assertEqual(summary_progress_ratio(1000, threshold=0), 0.0)
        self.assertEqual(
            summary_progress_ratio(1000, threshold=100, baseline_tokens=10_000),
            1.0,
        )

    def test_choose_summary_boundary_never_splits_tool_call_and_result(self):
        messages = [
            HumanMessage(content="Проверь конфиг"),
            AIMessage(content="Читаю файл", tool_calls=[{"id": "tc-1", "name": "read_file", "args": {}}]),
            ToolMessage(tool_call_id="tc-1", name="read_file", content="ok"),
            AIMessage(content="Правлю файл", tool_calls=[{"id": "tc-2", "name": "edit_file", "args": {}}]),
            ToolMessage(tool_call_id="tc-2", name="edit_file", content="ok"),
            AIMessage(content="Готово"),
        ]

        self.assertEqual(choose_summary_boundary(messages, keep_last=4), 0)

    def test_choose_summary_boundary_mid_run_keeps_active_user_turn(self):
        messages = [
            HumanMessage(content="Проверь конфиг"),
            AIMessage(content="Читаю файл", tool_calls=[{"id": "tc-1", "name": "read_file", "args": {}}]),
            ToolMessage(tool_call_id="tc-1", name="read_file", content="ok"),
            AIMessage(content="Правлю файл", tool_calls=[{"id": "tc-2", "name": "edit_file", "args": {}}]),
            ToolMessage(tool_call_id="tc-2", name="edit_file", content="ok"),
            AIMessage(content="Готово"),
        ]

        # The current user message anchors the visible transcript turn. The
        # user boundary at index 0 cannot be selected, while tool boundaries
        # inside this active turn must not remove its user-message anchor.
        boundary = choose_summary_boundary(
            messages,
            keep_last=4,
            allow_tool_round_boundaries=True,
        )
        self.assertEqual(boundary, 0)

        # Without the flag, only user turns are valid cut points.
        self.assertEqual(choose_summary_boundary(messages, keep_last=4), 0)

    def test_choose_summary_boundary_mid_run_cuts_before_active_user_turn(self):
        messages = [
            HumanMessage(content="Старая задача"),
            AIMessage(content="Старый ответ"),
            HumanMessage(content="Проверь конфиг"),
            AIMessage(content="Читаю файл", tool_calls=[{"id": "tc-1", "name": "read_file", "args": {}}]),
            ToolMessage(tool_call_id="tc-1", name="read_file", content="ok"),
        ]

        boundary = choose_summary_boundary(
            messages,
            keep_last=1,
            allow_tool_round_boundaries=True,
        )
        self.assertEqual(boundary, 2)
        self.assertEqual(messages[boundary].content, "Проверь конфиг")

    def test_choose_summary_boundary_mid_run_skips_unfinished_tool_round(self):
        messages = [
            HumanMessage(content="Проверь конфиг"),
            AIMessage(content="Читаю файл", tool_calls=[{"id": "tc-1", "name": "read_file", "args": {}}]),
            ToolMessage(tool_call_id="tc-1", name="read_file", content="ok"),
            # Round tc-2 started but its ToolMessage is missing -> must not be a cut point.
            AIMessage(content="Правлю файл", tool_calls=[{"id": "tc-2", "name": "edit_file", "args": {}}]),
        ]

        boundary = choose_summary_boundary(
            messages,
            keep_last=4,
            allow_tool_round_boundaries=True,
        )
        # There is no older user turn to remove, and the active user message
        # must remain as the anchor for the final visible transcript.
        self.assertEqual(boundary, 0)

    def test_choose_summary_boundary_keeps_at_least_keep_last_messages(self):
        messages = [
            HumanMessage(content="Первая задача"),
            AIMessage(content="Ответ по первой задаче"),
            HumanMessage(content="Вторая задача"),
            AIMessage(content="Шаг 1"),
            AIMessage(content="Шаг 2"),
            AIMessage(content="Шаг 3"),
            HumanMessage(content="продолжай"),
        ]

        boundary = choose_summary_boundary(messages, keep_last=4)

        self.assertEqual(boundary, 2)
        self.assertEqual(messages[boundary].content, "Вторая задача")

    def test_choose_summary_boundary_escalates_when_retained_history_exceeds_threshold(self):
        messages = [
            HumanMessage(content="Первая задача"),
            AIMessage(content="Ответ по первой задаче"),
            HumanMessage(content="Вторая задача"),
            AIMessage(content="деталь " * 4000),
            AIMessage(content="ещё деталь " * 4000),
            HumanMessage(content="продолжай"),
        ]

        self.assertEqual(choose_summary_boundary(messages, keep_last=4), 2)
        self.assertEqual(choose_summary_boundary(messages, keep_last=4, threshold=200), 5)

    def test_choose_summary_boundary_ignores_internal_retry_hint(self):
        messages = [
            HumanMessage(content="Проверь конфиг"),
            AIMessage(content="Читаю файл", tool_calls=[{"id": "tc-1", "name": "read_file", "args": {}}]),
            ToolMessage(tool_call_id="tc-1", name="read_file", content="ERROR[EXECUTION]: fail"),
            HumanMessage(content=REFLECTION_PROMPT),
            AIMessage(content="Повторяю"),
        ]

        self.assertEqual(choose_summary_boundary(messages, keep_last=2), 0)

    def test_format_history_for_summary_keeps_tool_call_names_and_args(self):
        rendered = format_history_for_summary(
            [
                HumanMessage(content="Проверь конфиг"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc-1",
                            "name": "read_file",
                            "args": {"path": "config/settings.json"},
                        }
                    ],
                ),
                ToolMessage(
                    tool_call_id="tc-1",
                    name="read_file",
                    content='{"mode":"safe"}',
                ),
            ],
            is_internal_retry=lambda _message: False,
        )

        self.assertIn("read_file", rendered)
        self.assertIn("config/settings.json", rendered)
        self.assertIn('tool(read_file)', rendered)

    def test_session_store_round_trip(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        snapshot = store.new_session(checkpoint_backend="sqlite", checkpoint_target="demo.sqlite")
        snapshot.approval_mode = "always"
        snapshot.cache_hit_tokens = 15_360
        store.save_active_session(snapshot)
        loaded = store.load_active_session()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.session_id, snapshot.session_id)
        self.assertEqual(loaded.thread_id, snapshot.thread_id)
        self.assertEqual(loaded.approval_mode, "always")
        self.assertEqual(loaded.cache_hit_tokens, 15_360)

    def test_session_store_defaults_missing_approval_mode_to_prompt(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        session_path.write_text(
            json.dumps(
                {
                    "session_id": "session-old",
                    "thread_id": "thread-old",
                    "checkpoint_backend": "sqlite",
                    "checkpoint_target": "demo.sqlite",
                    "created_at": "2026-03-25T10:00:00+00:00",
                    "updated_at": "2026-03-25T10:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

        loaded = SessionStore(session_path).load_active_session()

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.approval_mode, "prompt")

    def test_session_store_keeps_project_scoped_index(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        first = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-a", title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-b", title="Second")
        store.save_active_session(first, touch=False)
        store.save_active_session(second, touch=False)

        project_a = store.list_sessions(tmp / "project-a")
        project_b = store.list_sessions(tmp / "project-b")

        self.assertEqual([entry.title for entry in project_a], ["First"])
        self.assertEqual([entry.title for entry in project_b], ["Second"])
        self.assertEqual(store.get_active_session_for_project(tmp / "project-a").session_id, first.session_id)
        self.assertEqual(store.get_active_session_for_project(tmp / "project-b").session_id, second.session_id)
        self.assertCountEqual([entry.title for entry in store.list_sessions()], ["First", "Second"])

    def test_session_store_tracks_global_last_active_session(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        first = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-a", title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-b", title="Second")
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        last_active = store.get_last_active_session()

        self.assertIsNotNone(last_active)
        self.assertEqual(last_active.session_id, second.session_id)

    def test_session_store_delete_session_soft_removes_from_index_and_updates_last_active(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        first = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-a", title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-b", title="Second")
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        deleted = store.delete_session(second.session_id)

        self.assertTrue(deleted)
        self.assertIsNone(store.get_session(second.session_id))
        self.assertEqual(store.get_last_active_session().session_id, first.session_id)
        self.assertEqual(SessionStore(tmp / "session.json").load_active_session().session_id, first.session_id)

    def test_session_store_delete_project_removes_every_chat_of_that_project(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        kept = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-a", title="Kept")
        first = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-b", title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-b", title="Second")
        store.save_active_session(kept, touch=False, set_active=True)
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        deleted = store.delete_project(tmp / "project-b")

        self.assertCountEqual(deleted, [first.session_id, second.session_id])
        self.assertEqual([entry.session_id for entry in store.list_sessions()], [kept.session_id])
        self.assertEqual(store.list_sessions(tmp / "project-b"), [])
        self.assertIsNone(store.get_active_session_for_project(tmp / "project-b"))
        self.assertEqual(store.get_last_active_session().session_id, kept.session_id)
        self.assertEqual(SessionStore(tmp / "session.json").load_active_session().session_id, kept.session_id)

    def test_session_store_delete_project_ignores_unknown_and_empty_project(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        kept = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-a", title="Kept")
        store.save_active_session(kept, touch=False, set_active=True)

        self.assertEqual(store.delete_project(tmp / "project-missing"), [])
        self.assertEqual(store.delete_project(""), [])
        self.assertEqual(store.delete_project(None), [])
        self.assertEqual([entry.session_id for entry in store.list_sessions()], [kept.session_id])

    def test_session_store_migrates_legacy_session_into_index(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        session_path.write_text(
            json.dumps(
                {
                    "session_id": "legacy-session",
                    "thread_id": "legacy-thread",
                    "checkpoint_backend": "sqlite",
                    "checkpoint_target": "demo.sqlite",
                    "created_at": "2026-03-25T10:00:00+00:00",
                    "updated_at": "2026-03-25T10:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

        store = SessionStore(session_path)
        loaded = store.load_active_session()
        entries = store.list_sessions()

        self.assertIsNotNone(loaded)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].session_id, "legacy-session")

    def test_session_store_draft_session_stays_out_of_history_until_persisted(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        draft = store.new_session(
            "sqlite",
            "demo.sqlite",
            project_path=tmp / "project-a",
            title="New Chat [project-a]",
            persisted=False,
        )

        store.save_active_session(draft, touch=False, set_active=True)

        loaded = store.load_active_session()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.session_id, draft.session_id)
        self.assertEqual(store.list_sessions(), [])
        self.assertIsNone(store.get_last_active_session())

        draft.is_persisted = True
        store.save_active_session(draft, touch=False, set_active=True)

        entries = store.list_sessions()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].session_id, draft.session_id)

    def test_generate_chat_title_strips_common_prefixes_and_limits_length(self):
        self.assertEqual(
            generate_chat_title("Помоги скачать и настроить Apache на Windows"),
            "Скачать и настроить Apache на Windows",
        )
        self.assertEqual(generate_chat_title("   \n   "), "New Chat")
        self.assertTrue(generate_chat_title("сделай " + ("очень длинный запрос " * 10)).endswith("…"))

    def test_project_title_helpers_append_short_folder_label(self):
        project_path = Path("D:/work/client/demo-app")
        self.assertEqual(short_project_label(project_path), "client/demo-app")
        self.assertEqual(append_project_label("New Chat", project_path), "New Chat [client/demo-app]")

    def test_help_markdown_lists_shortcuts_and_history_commands(self):
        help_markdown = build_help_markdown()

        self.assertIn("**Ctrl+N**", help_markdown)
        self.assertIn("**Ctrl+B**", help_markdown)
        self.assertIn("**Ctrl+I**", help_markdown)
        self.assertIn("**Up/Down**", help_markdown)
        self.assertIn("Type **@**", help_markdown)
        self.assertIn("Right-click a chat", help_markdown)

    def test_worker_preserves_project_label_when_first_title_is_generated(self):
        tmp = self._workspace_tempdir()
        worker = gui_runtime.AgentRunWorker()
        worker.store = SessionStore(tmp / "session.json")
        worker.current_session = worker.store.new_session(
            "sqlite",
            "demo.sqlite",
            project_path=tmp / "client" / "demo-app",
            title=append_project_label("New Chat", tmp / "client" / "demo-app"),
        )

        updated = worker._maybe_set_session_title("Сделай сводку по ошибкам")

        self.assertTrue(updated)
        self.assertEqual(worker.current_session.title, "Сводку по ошибкам [client/demo-app]")

    def test_worker_persists_cache_hit_delta_for_current_session(self):
        tmp = self._workspace_tempdir()
        worker = gui_runtime.AgentRunWorker()
        worker.store = SessionStore(tmp / "session.json")
        worker.current_session = worker.store.new_session("sqlite", "demo.sqlite", project_path=tmp)

        worker._emit_stream_event(StreamEvent("cache_hit", {"tokens": 8192}))
        worker._emit_stream_event(StreamEvent("cache_hit", {"tokens": 4096}))

        self.assertEqual(worker.current_session.cache_hit_tokens, 12_288)
        restored = SessionStore(tmp / "session.json").load_active_session()
        self.assertIsNotNone(restored)
        self.assertEqual(restored.cache_hit_tokens, 12_288)

    def test_worker_emits_live_summary_progress_after_tool_finished(self):
        worker = gui_runtime.AgentRunWorker()
        worker.config = SimpleNamespace(summary_threshold=100, summary_keep_last=1)
        worker._active_summary_estimated_tokens = 10
        worker._active_summary_message_count = 2
        worker._active_summary_has_summary = False
        emitted = []
        worker.event_emitted.connect(emitted.append)

        worker._emit_stream_event(
            StreamEvent(
                "tool_finished",
                {"tool_id": "call-read", "name": "read_file", "content": "alpha beta gamma " * 12},
            )
        )

        self.assertEqual(emitted[0].type, "tool_finished")
        progress_events = [event for event in emitted if event.type == "summary_progress"]
        self.assertEqual(len(progress_events), 1)
        payload = progress_events[0].payload
        self.assertGreater(payload["estimated_tokens"], 10)
        self.assertEqual(payload["threshold"], 100)
        self.assertTrue(payload["live"])

    async def test_worker_resets_summary_progress_from_checkpoint_after_auto_summary(self):
        checkpoint_values = {
            "summary": "compressed earlier messages",
            "messages": [HumanMessage(content="recent user message"), AIMessage(content="recent answer")],
        }
        worker = gui_runtime.AgentRunWorker()
        worker.config = SimpleNamespace(
            summary_threshold=40000,
            summary_keep_last=4,
            summary_reserved_tokens=3000,
        )
        worker.agent_app = SimpleNamespace(
            get_state=lambda _config: SimpleNamespace(values=checkpoint_values),
        )
        worker.current_session = SimpleNamespace(
            thread_id="thread-1",
            last_run_stats="31.8s  ↓ 15921  ↑ 574",
        )
        worker.store = SimpleNamespace(save_active_session=lambda *_args, **_kwargs: None)
        worker._active_summary_estimated_tokens = 57731
        worker._active_summary_message_count = 40
        worker._active_summary_has_summary = False
        worker._active_summary_reserved_tokens = 3000
        emitted = []
        worker.event_emitted.connect(emitted.append)

        worker._emit_stream_event(StreamEvent("summary_notice", {"kind": "auto_summary", "count": 12}))
        await asyncio.sleep(0)

        progress_events = [event for event in emitted if event.type == "summary_progress"]
        self.assertEqual(len(progress_events), 1)
        payload = progress_events[0].payload
        self.assertGreater(payload["estimated_tokens"], 3000)
        self.assertNotEqual(payload["estimated_tokens"], payload["reserved_tokens"])
        self.assertLess(payload["estimated_tokens"], 37000)
        self.assertEqual(
            payload["remaining_tokens"],
            payload["trigger_tokens"] - payload["estimated_tokens"],
        )
        self.assertEqual(payload["message_count"], 2)
        self.assertTrue(payload["has_summary"])
        self.assertTrue(payload["live"])
        self.assertEqual(worker.current_session.last_run_stats, "")

    def test_build_transcript_payload_restores_turns_and_summary_notice(self):
        payload = build_transcript_payload(
            {
                "summary": "compressed earlier messages",
                "messages": [
                    HumanMessage(content="Покажи diff"),
                    AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"path": "demo.txt"}, "id": "tc-1"}]),
                    ToolMessage(tool_call_id="tc-1", content="Done\n```diff\n-old\n+new\n```", name="edit_file"),
                    AIMessage(content="Готово."),
                ],
            }
        )

        self.assertTrue(payload["summary_notice"])
        self.assertEqual(len(payload["turns"]), 1)
        self.assertEqual(payload["turns"][0]["user_text"], "Покажи diff")
        block_types = [block["type"] for block in payload["turns"][0]["blocks"]]
        self.assertEqual(block_types, ["tool", "assistant"])
        self.assertEqual(payload["turns"][0]["blocks"][0]["payload"]["name"], "edit_file")
        self.assertIn("Готово", payload["turns"][0]["blocks"][1]["markdown"])

    def test_build_transcript_payload_does_not_restore_assistant_thought_markdown(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Сделай вывод"),
                    AIMessage(content="<thought>Сначала проверю ограничения.</thought>Итог готов."),
                ],
            }
        )

        self.assertEqual(len(payload["turns"]), 1)
        blocks = payload["turns"][0]["blocks"]
        self.assertEqual([block["type"] for block in blocks], ["assistant"])
        self.assertEqual(blocks[0]["markdown"], "Итог готов.")
        self.assertNotIn("thought_markdown", blocks[0])

    def test_build_transcript_payload_restores_internal_handoff_notice_without_assistant_text(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Проверь завершение"),
                    AIMessage(
                        content="internal handoff",
                        additional_kwargs={
                            "agent_internal": {
                                "kind": "tool_issue_handoff",
                                "visible_in_ui": False,
                                "ui_notice": "Автопродолжение остановлено. Нужен новый запрос.",
                            }
                        },
                    ),
                ],
            }
        )

        self.assertEqual(len(payload["turns"]), 1)
        self.assertEqual(payload["turns"][0]["user_text"], "Проверь завершение")
        self.assertEqual(
            payload["turns"][0]["blocks"],
            [
                {
                    "type": "notice",
                    "message": "Автопродолжение остановлено. Нужен новый запрос.",
                    "level": "warning",
                }
            ],
        )

    def test_build_transcript_payload_hides_repaired_interruption_tool_message(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Сделай анализ проекта"),
                    AIMessage(
                        content="",
                        tool_calls=[{"id": "tc-repaired", "name": "read_file", "args": {"path": "script.js"}}],
                    ),
                    ToolMessage(
                        content=(
                            "ERROR[NETWORK]: The provider or aggregator stream ended before this tool returned a result. "
                            "This is usually a transient upstream disconnect, not a user stop. Please retry or continue."
                        ),
                        tool_call_id="tc-repaired",
                        name="read_file",
                        additional_kwargs={
                            "tool_args": {"path": "script.js"},
                            "agent_internal": {
                                "kind": "repaired_interrupted_tool_call",
                                "visible_in_ui": False,
                                "ui_notice": (
                                    "Provider stream interrupted while a tool was running. "
                                    "History was repaired automatically."
                                ),
                            },
                        },
                        status="error",
                    ),
                ]
            }
        )

        blocks = payload["turns"][0]["blocks"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "notice")
        self.assertIn("History was repaired automatically", blocks[0]["message"])

    def test_build_transcript_payload_keeps_assistant_text_without_pending_choice_state(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Нужно выбрать режим"),
                    AIMessage(
                        content=(
                            "Сначала нужно решить, какой путь подтверждаем.\n\n"
                            "Сделаю паузу и запрошу выбор через tool.\n"
                        )
                    ),
                ],
            }
        )

        self.assertEqual(len(payload["turns"]), 1)
        blocks = payload["turns"][0]["blocks"]
        self.assertEqual([block["type"] for block in blocks], ["assistant"])
        self.assertIn("Сначала нужно решить", blocks[0]["markdown"])
        legacy_choice_key = "_".join(("pending", "user", "choice"))
        self.assertNotIn(legacy_choice_key, payload)

    def test_build_transcript_payload_restores_assistant_text_from_list_content(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Подведи итог"),
                    AIMessage(
                        content=[
                            "Я завершил основную часть задачи. ",
                            {"type": "text", "text": "Изменения уже внесены."},
                            {"content": [" Проверка пройдена."]},
                        ]
                    ),
                ],
            }
        )

        self.assertEqual(len(payload["turns"]), 1)
        blocks = payload["turns"][0]["blocks"]
        self.assertEqual([block["type"] for block in blocks], ["assistant"])
        self.assertIn("Я завершил основную часть задачи.", blocks[0]["markdown"])
        self.assertIn("Изменения уже внесены.", blocks[0]["markdown"])
        self.assertIn("Проверка пройдена.", blocks[0]["markdown"])

    def test_build_user_choice_payload_adapts_interrupt_options_for_card(self):
        payload = build_user_choice_payload(
            {
                "kind": "user_choice",
                "question": "Введите ключ API или выберите другой вариант:",
                "options": [
                    "Ввести ключ API",
                    "Пропустить проверку и вернуть скрипт",
                    "Завершить проверку",
                ],
                "recommended": "Ввести ключ API",
            }
        )

        self.assertEqual(payload["kind"], "user_choice")
        self.assertEqual(payload["question"], "Введите ключ API или выберите другой вариант:")
        self.assertEqual(payload["recommended_key"], "Ввести ключ API")
        self.assertEqual(
            [option["submit_text"] for option in payload["options"]],
            [
                "Ввести ключ API",
                "Пропустить проверку и вернуть скрипт",
                "Завершить проверку",
            ],
        )
        recommended = [option for option in payload["options"] if option["recommended"]]
        self.assertEqual(len(recommended), 1)
        self.assertEqual(recommended[0]["submit_text"], "Ввести ключ API")

    def test_build_user_choice_payload_matches_recommended_by_synthesized_option_key(self):
        payload = build_user_choice_payload(
            {
                "kind": "user_choice",
                "question": "Как продолжаем?",
                "options": [
                    "Вариант 1: direct_api",
                    "Вариант 2: keep_mcp",
                ],
                "recommended": "option_1",
            }
        )

        recommended = [option for option in payload["options"] if option["recommended"]]
        self.assertEqual(len(recommended), 1)
        self.assertEqual(recommended[0]["key"], "option_1")
        self.assertEqual(payload["recommended_key"], "Вариант 1: direct_api")

    async def test_worker_initialize_can_force_new_session_on_reinitialize(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        old_session = store.new_session(checkpoint_backend="sqlite", checkpoint_target="demo.sqlite")
        store.save_active_session(old_session)
        project_path = tmp / "project-folder"
        project_path.mkdir()

        worker = gui_runtime.AgentRunWorker()
        config = self._make_config(SESSION_STATE_PATH=session_path)
        fake_app = type("FakeApp", (), {"get_state": lambda self, _config: type("State", (), {"values": {}})()})()
        tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
            },
        )()

        with (
            mock.patch.object(gui_runtime, "setup_runtime", return_value=config),
            mock.patch.object(gui_runtime, "build_agent_app", new=mock.AsyncMock(return_value=(fake_app, tool_registry))),
            mock.patch.object(gui_runtime, "repair_session_if_needed", new=mock.AsyncMock()),
            mock.patch.object(gui_runtime.Path, "cwd", return_value=project_path),
        ):
            await worker._initialize_async(force_new_session=True)

        self.assertNotEqual(worker.current_session.session_id, old_session.session_id)
        self.assertNotEqual(worker.current_session.thread_id, old_session.thread_id)
        self.assertEqual(worker.current_session.title, append_project_label("New Chat", project_path))
        persisted = SessionStore(session_path).load_active_session()
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.session_id, worker.current_session.session_id)

    async def test_worker_switch_session_updates_active_snapshot(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        project_a = tmp / "project-a"
        project_b = tmp / "project-b"
        project_a.mkdir()
        project_b.mkdir()
        first = store.new_session("sqlite", "demo.sqlite", project_path=project_a, title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=project_b, title="Second")
        store.save_active_session(first, touch=False)
        store.save_active_session(second, touch=False)

        worker = gui_runtime.AgentRunWorker()
        worker.store = store
        worker.config = self._make_config(SESSION_STATE_PATH=session_path)
        worker.tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
            },
        )()
        worker.agent_app = type(
            "FakeApp",
            (),
            {"get_state": lambda self, _config: type("State", (), {"values": {}})()},
        )()
        worker.current_session = first

        with (
            mock.patch.object(gui_runtime, "repair_session_if_needed", new=mock.AsyncMock()),
            mock.patch.object(gui_runtime.os, "chdir") as chdir_mock,
            mock.patch.object(gui_runtime.Path, "cwd", return_value=(tmp / "project-a")),
        ):
            await worker._switch_session_async(second)

        self.assertEqual(worker.current_session.session_id, second.session_id)
        self.assertEqual(SessionStore(session_path).load_active_session().session_id, second.session_id)
        chdir_mock.assert_called_once_with(str((tmp / "project-b").resolve()))

    async def test_worker_initialize_fallbacks_when_chdir_fails(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        blocked_project = tmp / "blocked-project"
        fallback_project = tmp / "fallback-project"
        blocked_project.mkdir()
        fallback_project.mkdir()
        blocked = store.new_session("sqlite", "demo.sqlite", project_path=blocked_project, title="Blocked")
        store.save_active_session(blocked, touch=False, set_active=True)

        worker = gui_runtime.AgentRunWorker()
        events = []
        worker.event_emitted.connect(events.append)
        config = self._make_config(SESSION_STATE_PATH=session_path)
        fake_app = type("FakeApp", (), {"get_state": lambda self, _config: type("State", (), {"values": {}})()})()
        tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
                "sync_working_directory": lambda self, _path: None,
            },
        )()

        with (
            mock.patch.object(gui_runtime, "setup_runtime", return_value=config),
            mock.patch.object(gui_runtime, "build_agent_app", new=mock.AsyncMock(return_value=(fake_app, tool_registry))),
            mock.patch.object(gui_runtime, "repair_session_if_needed", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(gui_runtime.Path, "cwd", return_value=fallback_project),
            mock.patch.object(gui_runtime.os, "chdir", side_effect=PermissionError("denied")),
            mock.patch.object(worker, "_log_ui_run_event") as log_mock,
        ):
            await worker._initialize_async()

        self.assertNotEqual(worker.current_session.session_id, blocked.session_id)
        self.assertEqual(worker.current_session.project_path, str(fallback_project.resolve()))
        self.assertEqual(SessionStore(session_path).load_active_session().session_id, worker.current_session.session_id)
        fallback_notices = [
            event
            for event in events
            if getattr(event, "type", "") == "summary_notice" and (event.payload or {}).get("kind") == "session_fallback"
        ]
        self.assertTrue(fallback_notices)
        self.assertTrue(any(call.args and call.args[0] == "session_restore_fallback" for call in log_mock.call_args_list))
        self.assertTrue(
            any(call.kwargs.get("reason") == "chdir_failed" and "PermissionError" in call.kwargs.get("error", "") for call in log_mock.call_args_list)
        )

    async def test_worker_switch_session_fallbacks_when_chdir_raises(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        project_a = tmp / "project-a"
        project_b = tmp / "project-b"
        project_a.mkdir()
        project_b.mkdir()
        first = store.new_session("sqlite", "demo.sqlite", project_path=project_a, title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=project_b, title="Second")
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        worker = gui_runtime.AgentRunWorker()
        events = []
        worker.event_emitted.connect(events.append)
        worker.store = store
        worker.config = self._make_config(SESSION_STATE_PATH=session_path)
        worker.tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
                "sync_working_directory": lambda self, _path: None,
            },
        )()
        worker.agent_app = type(
            "FakeApp",
            (),
            {"get_state": lambda self, _config: type("State", (), {"values": {}})()},
        )()
        worker.current_session = first

        with (
            mock.patch.object(gui_runtime, "repair_session_if_needed", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(gui_runtime.Path, "cwd", return_value=project_a),
            mock.patch.object(gui_runtime.os, "chdir", side_effect=OSError("network share denied")),
            mock.patch.object(worker, "_log_ui_run_event") as log_mock,
        ):
            await worker._switch_session_async(second)

        self.assertNotEqual(worker.current_session.session_id, second.session_id)
        self.assertEqual(worker.current_session.project_path, str(project_a.resolve()))
        self.assertEqual(SessionStore(session_path).load_active_session().session_id, worker.current_session.session_id)
        fallback_notices = [
            event
            for event in events
            if getattr(event, "type", "") == "summary_notice" and (event.payload or {}).get("kind") == "session_fallback"
        ]
        self.assertTrue(fallback_notices)
        self.assertTrue(any(call.args and call.args[0] == "session_switch_fallback" for call in log_mock.call_args_list))
        self.assertTrue(
            any(call.kwargs.get("reason") == "chdir_failed" and "OSError" in call.kwargs.get("error", "") for call in log_mock.call_args_list)
        )

    def test_worker_selects_global_last_active_session_on_initialize(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        project_a = tmp / "project-a"
        project_b = tmp / "project-b"
        project_a.mkdir()
        project_b.mkdir()

        first = store.new_session("sqlite", "demo.sqlite", project_path=project_a, title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=project_b, title="Second")
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        worker = gui_runtime.AgentRunWorker()
        worker.store = store
        worker.config = self._make_config(SESSION_STATE_PATH=session_path)
        worker.tool_registry = type(
            "ToolRegistryStub",
            (),
            {"checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"}},
        )()

        with mock.patch.object(gui_runtime.Path, "cwd", return_value=project_a):
            selected = worker._select_session_for_project()

        self.assertEqual(selected.session_id, second.session_id)

    async def test_new_user_turn_ignores_old_open_tool_issue(self):
        agent_llm = FakeLLM([AIMessage(content="Короткая сводка на экране.")])
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[],
            llm_with_tools=agent_llm,
        )
        state = {
            **self._initial_state("Покажи коротко эту инфу на экран"),
            "messages": [
                HumanMessage(content="Сохрани в файл"),
                AIMessage(content="Не сделал, так как вы выбрали Нет. Ожидаю дальнейших инструкций."),
                HumanMessage(content="Покажи коротко эту инфу на экран"),
            ],
            "turn_id": 1,
            "open_tool_issue": {
                "turn_id": 1,
                "kind": "approval_denied",
                "summary": "Execution of 'write_file' was cancelled by approval policy.",
                "tool_names": ["write_file"],
                "source": "approval",
            },
            "current_task": "Покажи коротко эту инфу на экран",
        }

        result = await nodes.agent_node(state)

        self.assertEqual(result["turn_id"], 2)
        self.assertIsNone(result["open_tool_issue"])
        unresolved_messages = [
            str(message.content)
            for message in agent_llm.invocations[0]
            if "UNRESOLVED TOOL FAILURE" in str(message.content) or "TOOL EXECUTION DENIED BY USER" in str(message.content)
        ]
        self.assertFalse(unresolved_messages)

    async def test_latest_user_message_overrides_stale_current_task_in_state(self):
        agent_llm = FakeLLM([AIMessage(content="Проверяю только указанный файл.")])
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[],
            llm_with_tools=agent_llm,
        )
        state = {
            **self._initial_state("проверь list_mistral_models.py"),
            "messages": [
                HumanMessage(content="восстанови проверку mistral api"),
                AIMessage(content="Промежуточный результат."),
                HumanMessage(content="проверь list_mistral_models.py"),
            ],
            "current_task": "восстанови проверку mistral api",
            "turn_id": 1,
        }

        result = await nodes.agent_node(state)

        self.assertEqual(result["current_task"], "проверь list_mistral_models.py")
        visible_humans = [
            str(message.content)
            for message in agent_llm.invocations[0]
            if isinstance(message, HumanMessage)
        ]
        self.assertIn("проверь list_mistral_models.py", visible_humans)

    async def test_continue_message_keeps_specific_current_task_in_context(self):
        agent_llm = FakeLLM([AIMessage(content="Продолжаю работу по исходной задаче.")])
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[],
            llm_with_tools=agent_llm,
        )
        state = {
            **self._initial_state("исправь ротацию ключей"),
            "messages": [
                HumanMessage(content="исправь ротацию ключей"),
                AIMessage(content="Понял, проверяю реализацию."),
                HumanMessage(content="продолжай"),
            ],
            "current_task": "исправь ротацию ключей",
            "turn_id": 2,
        }

        result = await nodes.agent_node(state)

        self.assertEqual(result["current_task"], "исправь ротацию ключей")
        system_texts = [
            str(message.content)
            for message in agent_llm.invocations[0]
            if isinstance(message, SystemMessage)
        ]
        self.assertTrue(any("Current task: исправь ротацию ключей" in text for text in system_texts))

    async def test_agent_node_allows_prose_only_response_without_forced_recovery(self):
        agent_llm = FakeLLM([AIMessage(content="Сейчас исправлю файл и внесу правки.")])
        edit_tool = FakeTool("edit_file", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[edit_tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "edit_file": ToolMetadata(name="edit_file", mutating=True, requires_approval=False),
            },
        )

        result = await nodes.agent_node(self._initial_state("исправь файл demo.txt"))

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        self.assertEqual(str(result["messages"][-1].content), "Сейчас исправлю файл и внесу правки.")
        self.assertEqual(len(agent_llm.invocations), 1)

    async def test_agent_node_clears_open_tool_issue_after_final_prose_response(self):
        agent_llm = FakeLLM([AIMessage(content="Не нашёл docs, поэтому даю вывод по доступным данным.")])
        read_tool = FakeTool("batch_web_search", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={"batch_web_search": ToolMetadata(name="batch_web_search", read_only=True)},
        )
        state = {
            **self._initial_state("проверь провайдера"),
            "messages": [
                HumanMessage(content="проверь провайдера"),
                AIMessage(
                    content="Ищу документацию.",
                    tool_calls=[
                        {
                            "id": "call-search",
                            "name": "batch_web_search",
                            "args": {"queries": ["provider docs"]},
                        }
                    ],
                ),
                ToolMessage(
                    content="ERROR[NOT_FOUND]: No results found.",
                    tool_call_id="call-search",
                    name="batch_web_search",
                ),
            ],
            "open_tool_issue": {
                "turn_id": 1,
                "kind": "tool_error",
                "summary": "No results found.",
                "tool_names": ["batch_web_search"],
                "tool_args": {"queries": ["provider docs"]},
                "source": "tools",
                "error_type": "NOT_FOUND",
                "fingerprint": "fp-search",
                "progress_fingerprint": "fp-search",
            },
            "recovery_state": {
                "turn_id": 1,
                "active_issue": {"summary": "No results found."},
                "active_strategy": {"strategy": "llm_replan"},
                "strategy_queue": [],
                "attempts_by_strategy": {"fp-search::llm_replan": 1},
                "progress_markers": ["fp-search"],
                "last_successful_evidence": "",
                "external_blocker": None,
                "llm_replan_attempted_for": ["fp-search"],
            },
        }

        result = await nodes.agent_node(state)

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        self.assertIsNone(result["recovery_state"]["active_issue"])
        self.assertIsNone(result["recovery_state"]["active_strategy"])
        self.assertEqual(result["recovery_state"]["llm_replan_attempted_for"], [])
        self.assertEqual(len(agent_llm.invocations), 1)

    async def test_agent_node_marks_malformed_tool_payload_as_protocol_error(self):
        agent_llm = FakeLLM(
            [
                AIMessage(
                    content="Открываю файл.",
                    invalid_tool_calls=[
                        {
                            "id": "tc-invalid-1",
                            "name": "read_file",
                            "args": "{path: README.md}",
                            "error": "malformed arguments",
                        }
                    ],
                )
            ]
        )
        read_tool = FakeTool("read_file", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={"read_file": ToolMetadata(name="read_file", read_only=True)},
        )

        result = await nodes.agent_node(self._initial_state("прочитай README.md"))

        self.assertEqual(result["turn_outcome"], "recover_agent")
        self.assertEqual(result["open_tool_issue"]["kind"], "protocol_error")
        self.assertEqual(result["open_tool_issue"]["details"]["protocol_reason"], "tool_protocol_error")
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertFalse(response.additional_kwargs["agent_internal"]["visible_in_ui"])

    def test_select_llm_for_active_tools_falls_back_to_prebound_model_when_subset_bind_fails(self):
        llm = FailingBindableLLM([], message="subset bind failed")
        fallback_llm = FakeLLM([AIMessage(content="Готово без сбоя.")])
        user_input_tool = FakeTool("request_user_input", "ignored")
        read_tool = FakeTool("read_file", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=llm,
            tools=[user_input_tool, read_tool],
            llm_with_tools=fallback_llm,
            tool_metadata={"read_file": ToolMetadata(name="read_file", read_only=True)},
        )
        result = nodes._select_llm_for_active_tools([read_tool], ["read_file"])

        self.assertIs(result, fallback_llm)
        self.assertEqual(llm.bound_tool_name_batches, [["read_file"]])

    async def test_agent_node_treats_plaintext_tool_markup_as_regular_text(self):
        agent_llm = FakeLLM(
            [
                AIMessage(
                    content=(
                        '<tool_code>{"name":"read_file","args":{"path":"README.md"}}</tool_code>'
                    )
                )
            ]
        )
        read_tool = FakeTool("read_file", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={"read_file": ToolMetadata(name="read_file", read_only=True)},
        )

        result = await nodes.agent_node(self._initial_state("прочитай README.md"))

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(
            str(response.content),
            '<tool_code>{"name":"read_file","args":{"path":"README.md"}}</tool_code>',
        )

    async def test_agent_node_allows_plaintext_tool_markup_in_regular_chat_explanation(self):
        agent_llm = FakeLLM(
            [
                AIMessage(
                    content=(
                        'Пример синтаксиса: <tool_call name="read_file">{\"path\":\"README.md\"}</tool_call>'
                    )
                )
            ]
        )
        read_tool = FakeTool("read_file", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={"read_file": ToolMetadata(name="read_file", read_only=True)},
        )
        result = await nodes.agent_node(self._initial_state("объясни синтаксис tool_call"))

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(
            str(response.content),
            'Пример синтаксиса: <tool_call name="read_file">{"path":"README.md"}</tool_call>',
        )

    async def test_agent_node_detects_history_tool_mismatch_before_llm_invoke(self):
        agent_llm = FakeLLM([AIMessage(content="Это сообщение не должно быть вызвано.")])
        read_tool = FakeTool("read_file", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={"read_file": ToolMetadata(name="read_file", read_only=True)},
        )
        state = self._initial_state("продолжай")
        state["messages"] = [
            HumanMessage(content="прочитай README.md"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "tc-history-1"}],
            ),
            AIMessage(content="Продолжаю без результата инструмента."),
            HumanMessage(content="продолжай"),
        ]

        result = await nodes.agent_node(state)

        self.assertEqual(result["turn_outcome"], "recover_agent")
        self.assertEqual(result["open_tool_issue"]["kind"], "protocol_error")
        self.assertEqual(result["open_tool_issue"]["details"]["protocol_reason"], "history_tool_mismatch")
        self.assertEqual(agent_llm.invocations, [])

    async def test_agent_node_drops_extra_request_user_input_calls_in_one_response(self):
        agent_llm = FakeLLM(
            [
                AIMessage(
                    content="Запрашиваю выбор пользователя.",
                    tool_calls=[
                        {
                            "name": "request_user_input",
                            "args": {"question": "Первый выбор", "options": ["A", "B"]},
                            "id": "tc-user-1",
                        },
                        {
                            "name": "request_user_input",
                            "args": {"question": "Второй выбор", "options": ["C", "D"]},
                            "id": "tc-user-2",
                        },
                    ],
                )
            ]
        )
        user_input_tool = FakeTool("request_user_input", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool],
            llm_with_tools=agent_llm,
        )

        result = await nodes.agent_node(self._initial_state("Проведи тест user input"))

        self.assertEqual(result["turn_outcome"], "run_tools")
        self.assertIsNone(result["open_tool_issue"])
        self.assertIn("INTERNAL TOOL PROTOCOL ERROR", result["last_tool_error"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0]["id"], "tc-user-1")
        self.assertEqual(str(response.content), "Запрашиваю выбор пользователя.")

    async def test_agent_node_blocks_second_request_user_input_in_same_turn(self):
        agent_llm = FakeLLM(
            [
                AIMessage(
                    content="Запрашиваю еще один выбор.",
                    tool_calls=[
                        {
                            "name": "request_user_input",
                            "args": {"question": "Еще вопрос", "options": ["Да", "Нет"]},
                            "id": "tc-user-repeat",
                        }
                    ],
                ),
                AIMessage(content="Продолжаю работу с уже выбранным вариантом."),
            ]
        )
        user_input_tool = FakeTool("request_user_input", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool],
            llm_with_tools=agent_llm,
        )
        state = self._initial_state("Проведи тест user input")
        state["messages"] = [
            HumanMessage(content="Проведи тест user input"),
            AIMessage(
                content="Запрашиваю выбор.",
                tool_calls=[
                    {
                        "name": "request_user_input",
                        "args": {"question": "Первый вопрос", "options": ["A", "B"]},
                        "id": "tc-user-initial",
                    }
                ],
            ),
            ToolMessage(
                content="A",
                tool_call_id="tc-user-initial",
                name="request_user_input",
            ),
        ]

        result = await nodes.agent_node(state)

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertFalse(response.tool_calls)
        self.assertEqual(str(response.content), "Продолжаю работу с уже выбранным вариантом.")
        self.assertEqual(len(agent_llm.invocations), 2)

    async def test_agent_node_excludes_request_user_input_from_bound_tools_after_choice(self):
        agent_llm = FakeBindableLLM([AIMessage(content="Продолжаю без нового вопроса.")])
        user_input_tool = FakeTool("request_user_input", "ignored")
        read_tool = FakeTool("read_file", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool, read_tool],
            llm_with_tools=agent_llm,
        )
        state = self._initial_state("Проведи тест user input")
        state["messages"] = [
            HumanMessage(content="Проведи тест user input"),
            AIMessage(
                content="Запрашиваю выбор.",
                tool_calls=[
                    {
                        "name": "request_user_input",
                        "args": {"question": "Первый вопрос", "options": ["A", "B"]},
                        "id": "tc-user-initial",
                    }
                ],
            ),
            ToolMessage(
                content="A",
                tool_call_id="tc-user-initial",
                name="request_user_input",
            ),
        ]

        await nodes.agent_node(state)

        self.assertTrue(agent_llm.bound_tool_name_batches)
        self.assertNotIn("request_user_input", agent_llm.bound_tool_name_batches[-1])
        self.assertIn("read_file", agent_llm.bound_tool_name_batches[-1])

    async def test_agent_node_binds_all_tools_for_regular_turn_without_intent_scoping(self):
        base_llm = FakeLLM([AIMessage(content="Этот экземпляр не должен вызываться.")])
        tools_llm = FakeBindableLLM([AIMessage(content="Сначала посмотрю проект через инструменты.")])
        read_tool = FakeTool("read_file", "ok")
        edit_tool = FakeTool("edit_file", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=base_llm,
            tools=[read_tool, edit_tool],
            llm_with_tools=tools_llm,
        )
        await nodes.agent_node(self._initial_state("Проанализируй проект в папке"))

        self.assertEqual(base_llm.invocations, [])
        self.assertEqual(len(tools_llm.invocations), 1)

    async def test_agent_node_binds_tools_without_keyword_intent_scoping(self):
        base_llm = FakeLLM([AIMessage(content="Понял, не буду выполнять установку.")])
        tools_llm = FakeBindableLLM([AIMessage(content="", tool_calls=[{"name": "batch_web_search", "args": {"queries": ["npcap"]}, "id": "tc-web"}])])
        search_tool = FakeTool("batch_web_search", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=base_llm,
            tools=[search_tool],
            llm_with_tools=tools_llm,
            tool_metadata={"batch_web_search": ToolMetadata(name="batch_web_search", read_only=True)},
        )
        state = self._initial_state("Не пытайся установить, он требует pcap")

        result = await nodes.agent_node(state)

        self.assertEqual(result["turn_outcome"], "run_tools")
        self.assertEqual(base_llm.invocations, [])
        self.assertEqual(len(tools_llm.invocations), 1)
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(response.tool_calls[0]["name"], "batch_web_search")

    async def test_agent_node_handles_repeated_empty_llm_response_without_ui_error(self):
        llm = FakeLLM([AIMessage(content="")])
        nodes = AgentNodes(
            config=self._make_config(MAX_RETRIES=1),
            llm=llm,
            tools=[],
            llm_with_tools=llm,
        )
        result = await nodes.agent_node(self._initial_state("Ответь коротко"))

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(result["last_tool_error"], "Empty response from LLM")
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertIn("The model returned an empty response", str(response.content))

    async def test_worker_delete_project_removes_all_project_chats_and_switches_session(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        project_a = tmp / "project-a"
        project_b = tmp / "project-b"
        project_a.mkdir()
        project_b.mkdir()
        kept = store.new_session("sqlite", "demo.sqlite", project_path=project_a, title="Kept")
        first = store.new_session("sqlite", "demo.sqlite", project_path=project_b, title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=project_b, title="Second")
        store.save_active_session(kept, touch=False, set_active=True)
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        worker = gui_runtime.AgentRunWorker()
        events = []
        worker.event_emitted.connect(events.append)
        worker.store = store
        worker.config = self._make_config(SESSION_STATE_PATH=session_path)
        worker.current_session = second
        worker.tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
                "sync_working_directory": lambda self, _path: None,
            },
        )()
        worker.agent_app = type(
            "FakeApp",
            (),
            {"get_state": lambda self, _config: type("State", (), {"values": {}})()},
        )()

        with (
            mock.patch.object(gui_runtime, "repair_session_if_needed", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(gui_runtime.Path, "cwd", return_value=project_b),
            mock.patch.object(gui_runtime.os, "chdir"),
        ):
            await worker._delete_project_async(str(project_b))

        reopened = SessionStore(session_path)
        self.assertEqual([entry.session_id for entry in reopened.list_sessions()], [kept.session_id])
        self.assertEqual(worker.current_session.session_id, kept.session_id)
        notices = [
            (event.payload or {}).get("message")
            for event in events
            if getattr(event, "type", "") == "summary_notice" and (event.payload or {}).get("kind") == "project_delete"
        ]
        self.assertEqual(notices, ["Deleted 2 chats of the project from history."])

    async def test_worker_delete_project_keeps_active_session_from_other_project(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        project_a = tmp / "project-a"
        project_b = tmp / "project-b"
        project_a.mkdir()
        project_b.mkdir()
        active = store.new_session("sqlite", "demo.sqlite", project_path=project_a, title="Active")
        other = store.new_session("sqlite", "demo.sqlite", project_path=project_b, title="Other")
        store.save_active_session(other, touch=False, set_active=True)
        store.save_active_session(active, touch=False, set_active=True)

        worker = gui_runtime.AgentRunWorker()
        events = []
        worker.event_emitted.connect(events.append)
        worker.store = store
        worker.config = self._make_config(SESSION_STATE_PATH=session_path)
        worker.current_session = active
        worker.tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
                "sync_working_directory": lambda self, _path: None,
            },
        )()
        worker.agent_app = type(
            "FakeApp",
            (),
            {"get_state": lambda self, _config: type("State", (), {"values": {}})()},
        )()

        with mock.patch.object(gui_runtime.os, "chdir") as chdir_mock:
            await worker._delete_project_async(str(project_b))

        self.assertEqual(worker.current_session.session_id, active.session_id)
        chdir_mock.assert_not_called()
        self.assertIsNone(SessionStore(session_path).get_session(other.session_id))
        self.assertFalse(any(getattr(event, "type", "") == "chat_reset" for event in events))

    async def test_worker_delete_active_session_switches_to_fallback_session(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        project_a = tmp / "project-a"
        project_b = tmp / "project-b"
        project_a.mkdir()
        project_b.mkdir()
        first = store.new_session("sqlite", "demo.sqlite", project_path=project_a, title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=project_b, title="Second")
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        worker = gui_runtime.AgentRunWorker()
        worker.store = store
        worker.config = self._make_config(SESSION_STATE_PATH=session_path)
        worker.current_session = second
        worker.tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
                "sync_working_directory": lambda self, _path: None,
            },
        )()
        worker.agent_app = type(
            "FakeApp",
            (),
            {"get_state": lambda self, _config: type("State", (), {"values": {}})()},
        )()

        with (
            mock.patch.object(gui_runtime, "repair_session_if_needed", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(gui_runtime.Path, "cwd", return_value=(tmp / "project-a")),
            mock.patch.object(gui_runtime.os, "chdir"),
        ):
            await worker._delete_session_async(second.session_id)

        self.assertEqual(worker.current_session.session_id, first.session_id)
        self.assertIsNone(SessionStore(session_path).get_session(second.session_id))

    async def test_worker_delete_active_session_fallbacks_when_replacement_chdir_fails(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        project_a = tmp / "project-a"
        project_b = tmp / "project-b"
        project_a.mkdir()
        project_b.mkdir()
        first = store.new_session("sqlite", "demo.sqlite", project_path=project_a, title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=project_b, title="Second")
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        worker = gui_runtime.AgentRunWorker()
        events = []
        worker.event_emitted.connect(events.append)
        worker.store = store
        worker.config = self._make_config(SESSION_STATE_PATH=session_path)
        worker.current_session = second
        worker.tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
                "sync_working_directory": lambda self, _path: None,
            },
        )()
        worker.agent_app = type(
            "FakeApp",
            (),
            {"get_state": lambda self, _config: type("State", (), {"values": {}})()},
        )()

        with (
            mock.patch.object(gui_runtime, "repair_session_if_needed", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(gui_runtime.Path, "cwd", return_value=project_b),
            mock.patch.object(gui_runtime.os, "chdir", side_effect=OSError("access denied")),
            mock.patch.object(worker, "_log_ui_run_event") as log_mock,
        ):
            await worker._delete_session_async(second.session_id)

        self.assertIsNone(SessionStore(session_path).get_session(second.session_id))
        self.assertEqual(worker.current_session.project_path, str(project_b.resolve()))
        self.assertNotEqual(worker.current_session.session_id, first.session_id)
        self.assertNotEqual(worker.current_session.session_id, second.session_id)
        self.assertEqual(SessionStore(session_path).load_active_session().session_id, worker.current_session.session_id)
        fallback_notices = [
            event
            for event in events
            if getattr(event, "type", "") == "summary_notice" and (event.payload or {}).get("kind") == "session_fallback"
        ]
        self.assertTrue(fallback_notices)
        self.assertTrue(any(call.args and call.args[0] == "session_delete_fallback" for call in log_mock.call_args_list))
        self.assertTrue(
            any(call.kwargs.get("reason") == "chdir_failed" and "OSError" in call.kwargs.get("error", "") for call in log_mock.call_args_list)
        )

    async def test_model_profiles_apply_can_skip_runtime_reload_by_default(self):
        tmp = self._workspace_tempdir()
        profile_path = tmp / "config.json"
        worker = gui_runtime.AgentRunWorker()
        worker.profile_store = ModelProfileStore(profile_path)
        worker.model_profiles = worker.profile_store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-old",
                        "base_url": "",
                    }
                ],
            }
        )
        worker._emit_session_payload = mock.AsyncMock(return_value={})

        with mock.patch.object(
            worker,
            "_rebuild_runtime_for_active_profile",
            new=mock.AsyncMock(return_value=None),
        ) as rebuild_mock:
            result = await worker._apply_model_profiles(
                {
                    "active_profile": "gemini-1-5-flash",
                    "profiles": [
                        {
                            "id": "gemini-1-5-flash",
                            "provider": "gemini",
                            "model": "gemini-1.5-flash",
                            "api_key": "gm-new",
                            "base_url": "",
                        }
                    ],
                },
                success_notice_kind="model_switched",
                success_notice_message="ok",
            )

        self.assertTrue(result)
        rebuild_mock.assert_not_called()
        restored = worker.profile_store.load_or_initialize()
        self.assertEqual(restored["active_profile"], "gemini-1-5-flash")

    async def test_model_profiles_apply_rebuilds_runtime_when_requested(self):
        tmp = self._workspace_tempdir()
        profile_path = tmp / "config.json"
        worker = gui_runtime.AgentRunWorker()
        worker.profile_store = ModelProfileStore(profile_path)
        worker.model_profiles = worker.profile_store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-old",
                        "base_url": "",
                    },
                    {
                        "id": "gemini-1-5-flash",
                        "provider": "gemini",
                        "model": "gemini-1.5-flash",
                        "api_key": "gm-new",
                        "base_url": "",
                    },
                ],
            }
        )
        worker._emit_session_payload = mock.AsyncMock(return_value={})

        with mock.patch.object(
            worker,
            "_rebuild_runtime_for_active_profile",
            new=mock.AsyncMock(return_value=None),
        ) as rebuild_mock:
            result = await worker._apply_model_profiles(
                {
                    "active_profile": "gemini-1-5-flash",
                    "profiles": [
                        {
                            "id": "gemini-1-5-flash",
                            "provider": "gemini",
                            "model": "gemini-1.5-flash",
                            "api_key": "gm-new",
                            "base_url": "",
                        }
                    ],
                },
                success_notice_kind="model_switched",
                success_notice_message="ok",
                sync_runtime=True,
            )

        self.assertTrue(result)
        rebuild_mock.assert_awaited_once()
        self.assertEqual(worker._runtime_profile_id, "gemini-1-5-flash")

    async def test_rebuild_runtime_for_active_profile_reuses_existing_registry_and_checkpoint_runtime(self):
        worker = gui_runtime.AgentRunWorker()
        worker.base_config = self._make_config()
        worker.config = worker.base_config
        worker.model_profiles = {
            "active_profile": "gemini-1-5-flash",
            "profiles": [
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-key",
                    "base_url": "",
                }
            ],
        }
        worker.current_session = SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
        )
        worker.store = mock.Mock()
        tool_registry = ToolRegistry(worker.config)
        tool_registry.checkpoint_runtime = type(
            "CheckpointRuntimeStub",
            (),
            {
                "checkpointer": MemorySaver(),
                "to_dict": lambda self: {
                    "backend": "sqlite",
                    "resolved_backend": "sqlite",
                    "target": "demo.sqlite",
                    "warnings": [],
                },
            },
        )()
        tool_registry.checkpoint_info = tool_registry.checkpoint_runtime.to_dict()
        worker.tool_registry = tool_registry

        fake_app = object()
        with (
            mock.patch.object(ToolRegistry, "reconfigure", autospec=True) as reconfigure_mock,
            mock.patch.object(gui_runtime, "build_agent_app", new=mock.AsyncMock(side_effect=AssertionError("full rebuild not expected"))),
            mock.patch.object(gui_runtime, "build_compiled_agent", return_value=(fake_app, tool_registry)) as compile_mock,
            mock.patch.object(worker, "_configure_cli_output_bridge"),
            mock.patch.object(worker, "_clear_cli_output_bridge"),
            mock.patch.object(worker, "_set_effective_model_capabilities"),
            mock.patch.object(gui_runtime, "close_runtime_resources", new=mock.AsyncMock()) as close_mock,
        ):
            await worker._rebuild_runtime_for_active_profile(worker.model_profiles)

        compile_mock.assert_called_once()
        reconfigure_mock.assert_called_once_with(tool_registry, mock.ANY)
        close_mock.assert_not_awaited()
        self.assertIs(worker.agent_app, fake_app)
        self.assertIs(worker.tool_registry, tool_registry)
        self.assertIs(worker.checkpoint_runtime, tool_registry.checkpoint_runtime)

    async def test_model_profiles_apply_success_updates_state(self):
        tmp = self._workspace_tempdir()
        profile_path = tmp / "config.json"
        worker = gui_runtime.AgentRunWorker()
        worker.profile_store = ModelProfileStore(profile_path)
        worker.model_profiles = worker.profile_store.save({"active_profile": None, "profiles": []})
        worker._emit_session_payload = mock.AsyncMock(return_value={})

        result = await worker._apply_model_profiles(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-new",
                        "base_url": "",
                    }
                ],
            },
            success_notice_kind="profiles_saved",
            success_notice_message="saved",
        )

        self.assertTrue(result)
        self.assertEqual(worker.model_profiles["active_profile"], "gpt-4o")
        self.assertEqual(worker.profile_store.load_or_initialize()["active_profile"], "gpt-4o")

    async def test_set_active_profile_notice_mentions_target_model(self):
        tmp = self._workspace_tempdir()
        profile_path = tmp / "config.json"
        worker = gui_runtime.AgentRunWorker()
        worker.profile_store = ModelProfileStore(profile_path)
        worker.model_profiles = worker.profile_store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-old",
                        "base_url": "",
                    },
                    {
                        "id": "gemini-1-5-flash",
                        "provider": "gemini",
                        "model": "gemini-1.5-flash",
                        "api_key": "gm-new",
                        "base_url": "",
                    },
                ],
            }
        )
        emitted_events = []
        worker.event_emitted.connect(emitted_events.append)
        worker._emit_session_payload = mock.AsyncMock(return_value={})

        with mock.patch.object(
            worker,
            "_rebuild_runtime_for_active_profile",
            new=mock.AsyncMock(return_value=None),
        ) as rebuild_mock:
            await worker._set_active_profile_async("gemini-1-5-flash")

        notice_events = [
            event for event in emitted_events
            if getattr(event, "type", "") == "summary_notice" and (event.payload or {}).get("kind") == "model_switched"
        ]
        self.assertEqual(len(notice_events), 1)
        self.assertEqual(
            notice_events[0].payload.get("message"),
            "Model switched to gemini-1.5-flash.",
        )
        rebuild_mock.assert_awaited_once()
        self.assertEqual(worker._runtime_profile_id, "gemini-1-5-flash")

    async def test_runtime_switch_happens_lazy_on_next_request_path(self):
        worker = gui_runtime.AgentRunWorker()
        worker.model_profiles = {
            "active_profile": "gemini-1-5-flash",
            "profiles": [
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-key",
                    "base_url": "",
                }
            ],
        }
        worker._runtime_profile_id = "gpt-4o"
        worker._emit_session_payload = mock.AsyncMock(return_value={})

        with mock.patch.object(
            worker,
            "_rebuild_runtime_for_active_profile",
            new=mock.AsyncMock(return_value=None),
        ):
            result = await worker._ensure_runtime_matches_selected_profile()

        self.assertTrue(result)
        self.assertEqual(worker._runtime_profile_id, "gemini-1-5-flash")

    def test_profile_bootstrap_env_comes_from_loaded_config(self):
        config = self._make_config(
            PROVIDER="openai",
            OPENAI_MODEL="openai/gpt-oss-120b",
            OPENAI_API_KEY="sk-config",
            OPENAI_BASE_URL="https://openrouter.ai/api/v1",
            GEMINI_MODEL="gemini-1.5-flash",
            GEMINI_API_KEY="gm-config",
        )
        env_map = gui_runtime.AgentRunWorker._profile_bootstrap_env_from_config(config)
        self.assertEqual(env_map["PROVIDER"], "openai")
        self.assertEqual(env_map["MODEL"], "openai/gpt-oss-120b")
        self.assertEqual(env_map["API_KEY"], "sk-config")
        self.assertEqual(env_map["BASE_URL"], "https://openrouter.ai/api/v1")
        self.assertEqual(env_map["OPENAI_MODEL"], "openai/gpt-oss-120b")

    def test_profile_bootstrap_env_uses_anthropic_base_url_for_anthropic(self):
        config = self._make_config(
            PROVIDER="anthropic",
            ANTHROPIC_MODEL="claude-haiku-4-5-20251001",
            ANTHROPIC_API_KEY="sk-ant-config",
            ANTHROPIC_BASE_URL="https://cc.freemodel.dev",
            OPENAI_MODEL="gpt-4o",
            OPENAI_API_KEY="sk-openai-config",
            OPENAI_BASE_URL="https://api.openai.com/v1",
        )
        env_map = gui_runtime.AgentRunWorker._profile_bootstrap_env_from_config(config)
        self.assertEqual(env_map["PROVIDER"], "anthropic")
        self.assertEqual(env_map["MODEL"], "claude-haiku-4-5-20251001")
        self.assertEqual(env_map["API_KEY"], "sk-ant-config")
        self.assertEqual(env_map["BASE_URL"], "https://cc.freemodel.dev")
        self.assertEqual(env_map["ANTHROPIC_BASE_URL"], "https://cc.freemodel.dev")
        self.assertEqual(env_map["OPENAI_BASE_URL"], "https://api.openai.com/v1")

    def test_worker_start_run_with_missing_model_profile_emits_notice(self):
        worker = gui_runtime.AgentRunWorker()
        worker.model_profiles = {"active_profile": None, "profiles": []}
        worker.current_session = SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
        )
        events = []
        worker.event_emitted.connect(events.append)

        worker.start_run("hello")

        self.assertTrue(any(event.type == "summary_notice" and event.payload.get("kind") == "model_missing" for event in events))

    def test_worker_start_run_with_profile_apply_failure_emits_error_and_clears_busy(self):
        worker = gui_runtime.AgentRunWorker()
        worker.model_profiles = {
            "active_profile": "anthropic-opus",
            "profiles": [
                {
                    "id": "anthropic-opus",
                    "provider": "anthropic",
                    "model": "claude-4.8-opus",
                    "api_key": "sk-demo",
                    "base_url": "https://proxy.example",
                }
            ],
        }
        worker.model_capabilities = {"image_input_supported": False}
        worker.current_session = SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
        )
        events = []
        busy_values = []
        worker.event_emitted.connect(events.append)
        worker.busy_changed.connect(busy_values.append)

        with mock.patch.object(
            worker,
            "_rebuild_runtime_for_active_profile",
            new=mock.AsyncMock(side_effect=ValueError('does not support reasoning effort "medium". Allowed: none')),
        ):
            worker.start_run("hello")

        matching = [
            event for event in events
            if event.type == "summary_notice" and event.payload.get("kind") == "model_switch_failed"
        ]
        self.assertTrue(matching)
        self.assertEqual(matching[0].payload.get("level"), "error")
        self.assertIn("does not support reasoning effort", matching[0].payload.get("message", ""))
        self.assertEqual(busy_values, [True, False])

    def test_worker_start_run_with_all_profiles_disabled_emits_enable_notice(self):
        worker = gui_runtime.AgentRunWorker()
        worker.model_profiles = {
            "active_profile": None,
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "",
                    "enabled": False,
                }
            ],
        }
        worker.current_session = SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
        )
        events = []
        worker.event_emitted.connect(events.append)

        worker.start_run("hello")

        matching = [
            event for event in events
            if event.type == "summary_notice" and event.payload.get("kind") == "model_missing"
        ]
        self.assertTrue(matching)
        self.assertIn("No enabled models available", matching[0].payload.get("message", ""))

    def test_worker_start_run_with_unsupported_image_input_emits_notice(self):
        worker = gui_runtime.AgentRunWorker()
        worker.model_profiles = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "",
                }
            ],
        }
        worker.model_capabilities = {"image_input_supported": False}
        worker.current_session = SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
        )
        events = []
        worker.event_emitted.connect(events.append)

        worker.start_run(
            {
                "text": "Посмотри на изображение",
                "attachments": [{"id": "img-1", "path": "D:/demo/sample.png", "mime_type": "image/png"}],
            }
        )

        self.assertTrue(
            any(event.type == "summary_notice" and event.payload.get("kind") == "image_input_unsupported" for event in events)
        )

    def test_worker_start_run_uses_manual_profile_image_support_override(self):
        worker = gui_runtime.AgentRunWorker()
        worker.model_profiles = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "",
                    "supports_image_input": True,
                }
            ],
        }
        worker.runtime_model_capabilities = {"image_input_supported": False}
        worker.model_capabilities = {"image_input_supported": True}
        worker.current_session = SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
        )
        events = []
        worker.event_emitted.connect(events.append)

        with (
            mock.patch.object(worker, "_ensure_runtime_matches_selected_profile", return_value=True),
            mock.patch.object(worker, "_maybe_set_session_title", return_value=False),
            mock.patch.object(worker, "_run", side_effect=lambda coro: coro.close()) as run_mock,
        ):
            worker.start_run(
                {
                    "text": "Посмотри на изображение",
                    "attachments": [{"id": "img-1", "path": "D:/demo/sample.png", "mime_type": "image/png"}],
                }
            )

        self.assertFalse(
            any(event.type == "summary_notice" and event.payload.get("kind") == "image_input_unsupported" for event in events)
        )
        run_mock.assert_called()

    async def test_run_graph_payload_with_image_failure_emits_friendly_notice(self):
        worker = gui_runtime.AgentRunWorker()

        worker.agent_app = type(
            "FailingApp",
            (),
            {
                "astream": lambda self, *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("image input not supported"))
            },
        )()
        worker.store = mock.Mock()
        worker.current_session = SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
        )
        worker.config = self._make_config()
        worker._active_request_has_images = True
        worker._set_busy(True)
        events = []
        worker.event_emitted.connect(events.append)

        await worker._run_graph_payload({"messages": []})

        self.assertTrue(
            any(event.type == "summary_notice" and event.payload.get("kind") == "image_input_failed" for event in events)
        )
        self.assertTrue(any(event.type == "run_failed" for event in events))
        self.assertFalse(worker._active_request_has_images)

    async def test_run_graph_payload_failed_stream_skips_success_finalization(self):
        worker = gui_runtime.AgentRunWorker()
        worker.agent_app = type(
            "DummyApp",
            (),
            {"astream": lambda self, *_args, **_kwargs: object()},
        )()
        worker.store = mock.Mock()
        worker.current_session = SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
        )
        worker.config = self._make_config()
        worker._set_busy(True)

        with (
            mock.patch.object(
                StreamProcessor,
                "process_stream",
                new=mock.AsyncMock(
                    return_value=StreamProcessResult(
                        stats=None,
                        failed=True,
                        error_message="graph exploded",
                        elapsed_seconds=0.5,
                    )
                ),
            ),
            mock.patch.object(worker, "_repair_current_session_if_needed", new=mock.AsyncMock(return_value=[])) as repair_mock,
            mock.patch.object(worker, "_emit_session_payload", new=mock.AsyncMock()) as emit_session_mock,
        ):
            await worker._run_graph_payload({"messages": []})

        worker.store.save_active_session.assert_not_called()
        emit_session_mock.assert_not_awaited()
        repair_mock.assert_awaited_once()
        self.assertFalse(worker._is_busy)

    async def test_run_graph_payload_auto_continues_once_after_stream_repair(self):
        worker = gui_runtime.AgentRunWorker()

        class DummyApp:
            def __init__(self):
                self.inputs = []

            def astream(self, payload, *_args, **_kwargs):
                self.inputs.append(payload)
                return object()

        dummy_app = DummyApp()
        worker.agent_app = dummy_app
        worker.store = mock.Mock()
        worker.current_session = SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
        )
        worker.config = self._make_config(MAX_RETRIES=3, RETRY_DELAY=0)
        worker._set_busy(True)
        events = []
        worker.event_emitted.connect(events.append)

        with (
            mock.patch.object(
                StreamProcessor,
                "process_stream",
                new=mock.AsyncMock(
                    side_effect=[
                        StreamProcessResult(
                            stats=None,
                            failed=True,
                            error_message="upstream disconnected",
                            elapsed_seconds=0.5,
                        ),
                        StreamProcessResult(
                            stats="1.0s   In: 10   Out: 2",
                            elapsed_seconds=1.0,
                        ),
                    ]
                ),
            ),
            mock.patch.object(
                worker,
                "_repair_current_session_if_needed",
                new=mock.AsyncMock(side_effect=[["repaired missing tool output"], []]),
            ) as repair_mock,
            mock.patch.object(worker, "_emit_session_payload", new=mock.AsyncMock()) as emit_session_mock,
        ):
            await worker._run_graph_payload({"messages": []})

        self.assertEqual(len(dummy_app.inputs), 2)
        self.assertEqual(dummy_app.inputs[0], {"messages": []})
        self.assertIsNone(dummy_app.inputs[1])
        self.assertEqual(repair_mock.await_count, 2)
        worker.store.save_active_session.assert_called_once()
        emit_session_mock.assert_awaited_once()
        self.assertTrue(
            any(
                event.type == "summary_notice" and event.payload.get("kind") == "stream_repair_auto_continue"
                for event in events
            )
        )
        self.assertFalse(worker._is_busy)

    async def test_run_graph_payload_auto_continues_after_successful_stream_repair(self):
        worker = gui_runtime.AgentRunWorker()

        class DummyApp:
            def __init__(self):
                self.inputs = []

            def astream(self, payload, *_args, **_kwargs):
                self.inputs.append(payload)
                return object()

        dummy_app = DummyApp()
        worker.agent_app = dummy_app
        worker.store = mock.Mock()
        worker.current_session = SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
        )
        worker.config = self._make_config(MAX_RETRIES=3, RETRY_DELAY=0)
        worker._set_busy(True)
        events = []
        worker.event_emitted.connect(events.append)

        with (
            mock.patch.object(
                StreamProcessor,
                "process_stream",
                new=mock.AsyncMock(
                    side_effect=[
                        StreamProcessResult(
                            stats="0.5s   In: 10   Out: 2",
                            elapsed_seconds=0.5,
                        ),
                        StreamProcessResult(
                            stats="1.0s   In: 12   Out: 4",
                            elapsed_seconds=1.0,
                        ),
                    ]
                ),
            ),
            mock.patch.object(
                worker,
                "_repair_current_session_if_needed",
                new=mock.AsyncMock(side_effect=[["repaired missing tool output"], []]),
            ) as repair_mock,
            mock.patch.object(worker, "_emit_session_payload", new=mock.AsyncMock()) as emit_session_mock,
        ):
            await worker._run_graph_payload({"messages": []})

        self.assertEqual(len(dummy_app.inputs), 2)
        self.assertEqual(dummy_app.inputs[0], {"messages": []})
        self.assertIsNone(dummy_app.inputs[1])
        self.assertEqual(repair_mock.await_count, 2)
        worker.store.save_active_session.assert_called_once()
        emit_session_mock.assert_awaited_once()
        self.assertTrue(
            any(
                event.type == "summary_notice" and event.payload.get("kind") == "stream_repair_auto_continue"
                for event in events
            )
        )
        self.assertFalse(worker._is_busy)

    async def test_run_graph_payload_success_refreshes_transcript_payload(self):
        worker = gui_runtime.AgentRunWorker()
        worker.agent_app = type(
            "DummyApp",
            (),
            {"astream": lambda self, *_args, **_kwargs: object()},
        )()
        worker.store = mock.Mock()
        worker.current_session = SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
        )
        worker.config = self._make_config()
        worker._set_busy(True)

        with (
            mock.patch.object(
                StreamProcessor,
                "process_stream",
                new=mock.AsyncMock(
                    return_value=StreamProcessResult(
                        stats="0.5s   In: 10   Out: 2",
                        elapsed_seconds=0.5,
                    )
                ),
            ),
            mock.patch.object(worker, "_repair_current_session_if_needed", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(worker, "_emit_session_payload", new=mock.AsyncMock(return_value={})) as emit_session_mock,
        ):
            await worker._run_graph_payload({"messages": []})

        worker.store.save_active_session.assert_called_once_with(worker.current_session, touch=True, set_active=True)
        emit_session_mock.assert_awaited_once_with(include_transcript=True)
        self.assertFalse(worker._is_busy)

    def test_stop_background_process_denies_external_pid_by_default(self):
        result = process_tools.stop_background_process.invoke({"pid": os.getpid()})
        self.assertIn("ACCESS_DENIED", result)

    def test_classify_stream_error_rate_limit(self):
        info = gui_runtime.AgentRunWorker._classify_stream_error("Error: 429 Too Many Requests")
        self.assertEqual(info["kind"], "rate_limit")
        self.assertTrue(info["retryable"])

    def test_classify_stream_error_timeout(self):
        info = gui_runtime.AgentRunWorker._classify_stream_error("Request timed out after 30s")
        self.assertEqual(info["kind"], "timeout")
        self.assertTrue(info["retryable"])

    def test_classify_stream_error_server_error(self):
        info = gui_runtime.AgentRunWorker._classify_stream_error("503 Service Unavailable")
        self.assertEqual(info["kind"], "server_error")
        self.assertTrue(info["retryable"])

    def test_classify_stream_error_network(self):
        info = gui_runtime.AgentRunWorker._classify_stream_error("Connection reset by peer")
        self.assertEqual(info["kind"], "network")
        self.assertTrue(info["retryable"])

    def test_classify_stream_error_unknown(self):
        info = gui_runtime.AgentRunWorker._classify_stream_error("Something unexpected happened")
        self.assertEqual(info["kind"], "unknown")
        self.assertFalse(info["retryable"])

    def test_classify_stream_error_does_not_retry_exhausted_llm_error(self):
        info = gui_runtime.AgentRunWorker._classify_stream_error(
            "Connection error.",
            {"retry_exhausted": True, "http_status": None},
        )
        self.assertEqual(info["kind"], "retry_exhausted")
        self.assertFalse(info["retryable"])

    def test_classify_stream_error_does_not_retry_permanent_http_error(self):
        info = gui_runtime.AgentRunWorker._classify_stream_error(
            "Function not found",
            {"retry_exhausted": False, "http_status": 404},
        )
        self.assertEqual(info["kind"], "client_error")
        self.assertFalse(info["retryable"])

    def test_stream_retry_backoff_increases_with_attempts(self):
        """Backoff should increase exponentially across attempts."""
        delay_0 = gui_runtime.AgentRunWorker._stream_retry_backoff(0, 2.0, "network")
        delay_1 = gui_runtime.AgentRunWorker._stream_retry_backoff(1, 2.0, "network")
        delay_2 = gui_runtime.AgentRunWorker._stream_retry_backoff(2, 2.0, "network")
        # Base component (without jitter) should be 2, 4, 8
        self.assertGreater(delay_0, 0)
        self.assertGreater(delay_1, delay_0 - 2)  # account for jitter range
        self.assertGreater(delay_2, delay_1 - 4)

    def test_stream_retry_backoff_rate_limit_uses_longer_base(self):
        """Rate-limit errors should produce longer backoff than network errors."""
        network_delay = gui_runtime.AgentRunWorker._stream_retry_backoff(0, 2.0, "network")
        rate_limit_delay = gui_runtime.AgentRunWorker._stream_retry_backoff(0, 2.0, "rate_limit")
        # rate_limit base = 2.0 * 1.5 = 3.0, network base = 2.0
        # With jitter [0, base], rate_limit minimum = 3.0, network maximum = 4.0
        # So we check the base component: rate_limit should be at least 1.5x on average
        # More robust: check multiple samples
        network_samples = [gui_runtime.AgentRunWorker._stream_retry_backoff(0, 2.0, "network") for _ in range(100)]
        rate_limit_samples = [gui_runtime.AgentRunWorker._stream_retry_backoff(0, 2.0, "rate_limit") for _ in range(100)]
        self.assertGreater(sum(rate_limit_samples) / len(rate_limit_samples),
                           sum(network_samples) / len(network_samples))

    async def test_run_graph_payload_includes_backoff_and_error_kind_in_log(self):
        """Verify that backoff_seconds and error_kind are logged during stream repair."""
        worker = gui_runtime.AgentRunWorker()

        class DummyApp:
            def __init__(self):
                self.inputs = []

            def astream(self, payload, *_args, **_kwargs):
                self.inputs.append(payload)
                return object()

        dummy_app = DummyApp()
        worker.agent_app = dummy_app
        worker.store = mock.Mock()
        worker.current_session = SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
        )
        worker.config = self._make_config(MAX_RETRIES=3, RETRY_DELAY=0)
        worker._set_busy(True)
        logged_events = []
        worker._log_ui_run_event = lambda event_type, **kwargs: logged_events.append((event_type, kwargs))

        with (
            mock.patch.object(
                StreamProcessor,
                "process_stream",
                new=mock.AsyncMock(
                    side_effect=[
                        StreamProcessResult(
                            stats=None,
                            failed=True,
                            error_message="429 Too Many Requests",
                            elapsed_seconds=0.5,
                        ),
                        StreamProcessResult(
                            stats="1.0s   In: 10   Out: 2",
                            elapsed_seconds=1.0,
                        ),
                    ]
                ),
            ),
            mock.patch.object(
                worker,
                "_repair_current_session_if_needed",
                new=mock.AsyncMock(side_effect=[["repaired missing tool output"], []]),
            ),
            mock.patch.object(worker, "_emit_session_payload", new=mock.AsyncMock()),
        ):
            await worker._run_graph_payload({"messages": []})

        repair_logs = [kwargs for et, kwargs in logged_events if et == "run_auto_continued_after_stream_repair"]
        self.assertTrue(len(repair_logs) >= 1)
        self.assertEqual(repair_logs[0]["error_kind"], "rate_limit")
        self.assertIn("backoff_seconds", repair_logs[0])

    def test_stream_processor_ignores_tool_call_without_id(self):
        processor = StreamProcessor()
        processor._remember_tool_call({"name": "broken_tool", "args": {"x": 1}})
        self.assertEqual(processor.tool_buffer, {})

    def test_stream_processor_emits_notice_for_hidden_internal_message(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(
                content="internal handoff",
                additional_kwargs={
                    "agent_internal": {
                        "kind": "tool_issue_handoff",
                        "visible_in_ui": False,
                        "ui_notice": "Нужен новый запрос.",
                    }
                },
            )
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "summary_notice")
        self.assertEqual(events[0].payload["kind"], "agent_internal_notice")
        self.assertEqual(events[0].payload["message"], "Нужен новый запрос.")
        self.assertEqual(processor.full_text, "")
        self.assertEqual(processor.clean_full, "")

if __name__ == "__main__":
    unittest.main()
