import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent import create_agent_workflow
from core.config import AgentConfig
from core.nodes import AgentNodes
from core.summarize_policy import estimate_summary_tokens
from core.tool_policy import ToolMetadata
from tools.user_input_tool import request_user_input
from ui.runtime import build_graph_config
from ui.runtime_payloads import build_transcript_payload


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    async def ainvoke(self, context):
        self.invocations.append(context)
        if not self.responses:
            return AIMessage(content="Готово.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ProviderSafeFakeLLM(FakeLLM):
    async def ainvoke(self, context):
        last_visible = next(
            (message for message in reversed(context) if not isinstance(message, SystemMessage)),
            None,
        )
        if isinstance(last_visible, AIMessage):
            raise AssertionError("provider-unsafe assistant-last context")
        return await super().ainvoke(context)


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


class StabilityGraphTests(unittest.IsolatedAsyncioTestCase):
    def _make_config(
        self,
        *,
        model_supports_tools=True,
        max_loops=8,
        max_retries=3,
        retry_delay=0,
        enable_approvals=False,
        summary_threshold=None,
        summary_keep_last=None,
        summary_reserved_tokens=None,
        summary_max_tokens=None,
    ):
        config_kwargs = {
            "provider": "openai",
            "openai_api_key": "test-key",
            "model_supports_tools": model_supports_tools,
            "max_loops": max_loops,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "enable_approvals": enable_approvals,
            "prompt_path": Path(__file__).resolve().parents[1] / "prompt.txt",
        }
        if summary_threshold is not None:
            config_kwargs["summary_threshold"] = summary_threshold
        if summary_keep_last is not None:
            config_kwargs["summary_keep_last"] = summary_keep_last
        if summary_reserved_tokens is not None:
            config_kwargs["summary_reserved_tokens"] = summary_reserved_tokens
        if summary_max_tokens is not None:
            config_kwargs["summary_max_tokens"] = summary_max_tokens
        return AgentConfig(**config_kwargs)

    def _build_app(
        self,
        *,
        agent_responses,
        tools=None,
        model_supports_tools=True,
        enable_approvals=False,
        agent_llm_cls=FakeLLM,
        tool_metadata=None,
        max_loops=8,
        summary_threshold=None,
        summary_keep_last=None,
        summary_reserved_tokens=None,
        summary_max_tokens=None,
    ):
        config = self._make_config(
            model_supports_tools=model_supports_tools,
            enable_approvals=enable_approvals,
            max_loops=max_loops,
            summary_threshold=summary_threshold,
            summary_keep_last=summary_keep_last,
            summary_reserved_tokens=summary_reserved_tokens,
            summary_max_tokens=summary_max_tokens,
        )
        agent_llm = agent_llm_cls(agent_responses)
        nodes = AgentNodes(
            config=config,
            llm=agent_llm,
            tools=tools or [],
            llm_with_tools=agent_llm,
            tool_metadata=tool_metadata or {},
        )
        workflow = create_agent_workflow(
            nodes,
            config,
            tools_enabled=bool(tools) and model_supports_tools,
        )
        app = workflow.compile(checkpointer=MemorySaver())
        return app, agent_llm

    def _initial_state(self, task="Проверь задачу"):
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
            "turn_id": 1,
            "open_tool_issue": None,
            "pending_approval": None,
            "last_tool_error": "",
            "last_tool_result": "",
        }

    async def test_chat_only_turn_finishes_without_retry(self):
        app, agent_llm = self._build_app(
            agent_responses=[AIMessage(content="Задача выполнена.")],
            tools=[],
            model_supports_tools=False,
        )

        result = await app.ainvoke(
            self._initial_state("Скажи готово"),
            config={"configurable": {"thread_id": "chat-only"}, "recursion_limit": 24},
        )

        self.assertEqual(result["messages"][-1].content, "Задача выполнена.")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertIsNone(result["open_tool_issue"])

    async def test_auto_summary_keeps_active_user_turn_for_final_transcript(self):
        tool = FakeTool("read_file", "important result")
        app, _agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="Проверяю файл",
                    tool_calls=[{"id": "tc-transcript", "name": "read_file", "args": {}}],
                ),
                AIMessage(content="Файл проверен, результат готов."),
            ],
            tools=[tool],
            summary_threshold=1,
            summary_keep_last=1,
        )

        result = await app.ainvoke(
            self._initial_state("Проверь файл"),
            config={"configurable": {"thread_id": "summary-final-transcript"}, "recursion_limit": 32},
        )

        transcript = build_transcript_payload(result)
        self.assertEqual(len(transcript["turns"]), 1)
        self.assertEqual(transcript["turns"][0]["user_text"], "Проверь файл")
        self.assertTrue(
            any(
                block.get("type") == "assistant"
                and "результат готов" in block.get("markdown", "")
                for block in transcript["turns"][0]["blocks"]
            )
        )

    async def test_auto_summary_is_included_in_final_model_payload_as_memory(self):
        summary_text = "- Completed earlier investigation: relevant file is core/context_builder.py."
        app, agent_llm = self._build_app(
            agent_responses=[AIMessage(content=summary_text), AIMessage(content="Continue from memory.")],
            tools=[],
            model_supports_tools=False,
            summary_threshold=1,
            summary_keep_last=1,
        )
        state = self._initial_state("Continue the investigation")
        state["messages"] = [
            HumanMessage(content="old request " + "x" * 2000),
            AIMessage(content="old response " + "x" * 2000),
            HumanMessage(content="old follow-up " + "x" * 2000),
            AIMessage(content="old result " + "x" * 2000),
            HumanMessage(content="old decision " + "x" * 2000),
            HumanMessage(content="latest request"),
        ]

        result = await app.ainvoke(
            state,
            config={"configurable": {"thread_id": "summary-memory-payload"}, "recursion_limit": 24},
        )

        self.assertEqual(result["summary"], summary_text)
        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertEqual(
            [message.content for message in result["transcript_messages"] if isinstance(message, HumanMessage)],
            [
                "old request " + "x" * 2000,
                "old follow-up " + "x" * 2000,
                "old decision " + "x" * 2000,
                "latest request",
            ],
        )
        transcript = build_transcript_payload(result)
        self.assertEqual(
            [turn["user_text"] for turn in transcript["turns"]],
            [
                "old request " + "x" * 2000,
                "old follow-up " + "x" * 2000,
                "old decision " + "x" * 2000,
                "latest request",
            ],
        )
        final_payload = agent_llm.invocations[1]
        memory_text = "\n".join(
            str(message.content)
            for message in final_payload
            if isinstance(message, SystemMessage)
        )
        self.assertIn(f"<memory>\n{summary_text}\n</memory>", memory_text)

    async def test_auto_summary_retains_recent_tool_exchange_for_the_model(self):
        summary_text = "- Older turn compacted: the goal is to finish the config audit."
        app, agent_llm = self._build_app(
            agent_responses=[AIMessage(content=summary_text), AIMessage(content="Continuing the audit.")],
            tools=[],
            model_supports_tools=False,
            summary_threshold=600,
            summary_keep_last=4,
            summary_reserved_tokens=0,
        )
        state = self._initial_state("Доведи аудит конфига до конца")
        state["messages"] = [
            HumanMessage(content="старая задача: " + "детали конфигурации " * 400),
            AIMessage(content="старый ответ: " + "детали конфигурации " * 400),
            HumanMessage(content="Доведи аудит конфига до конца"),
            AIMessage(
                content="Читаю конфиг",
                tool_calls=[{"id": "tc-audit-1", "name": "read_file", "args": {"path": "core/config.py"}}],
            ),
            ToolMessage(tool_call_id="tc-audit-1", name="read_file", content="summary_threshold = 8000"),
            AIMessage(content="Нашёл текущее значение порога."),
            HumanMessage(content="продолжай"),
        ]

        result = await app.ainvoke(
            state,
            config={"configurable": {"thread_id": "summary-retains-tool-exchange"}, "recursion_limit": 24},
        )

        self.assertEqual(result["summary"], summary_text)
        retained = result["messages"]
        self.assertIsInstance(retained[0], HumanMessage)
        self.assertEqual(retained[0].content, "Доведи аудит конфига до конца")

        retained_tool_call_ids = {
            str(tool_call.get("id"))
            for message in retained
            if isinstance(message, AIMessage)
            for tool_call in (message.tool_calls or [])
        }
        retained_tool_result_ids = {
            str(message.tool_call_id) for message in retained if isinstance(message, ToolMessage)
        }
        self.assertEqual(retained_tool_result_ids, {"tc-audit-1"})
        self.assertEqual(retained_tool_result_ids, retained_tool_call_ids)

        agent_payload = agent_llm.invocations[1]
        self.assertTrue(
            any(
                isinstance(message, ToolMessage) and "summary_threshold = 8000" in str(message.content)
                for message in agent_payload
            )
        )
        self.assertTrue(
            any(
                isinstance(message, SystemMessage) and f"<memory>\n{summary_text}\n</memory>" in str(message.content)
                for message in agent_payload
            )
        )

    async def test_auto_summary_folds_memory_that_exceeds_its_budget(self):
        oversized_memory = "\n".join(
            f"- fact {index}: keep the audit going with detail {index}" for index in range(40)
        )
        folded_memory = "\n".join(f"- fact {index}: audit detail kept" for index in range(4))
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(content=oversized_memory),
                AIMessage(content=folded_memory),
                AIMessage(content="Continue from memory."),
            ],
            tools=[],
            model_supports_tools=False,
            summary_threshold=1,
            summary_keep_last=1,
            summary_max_tokens=120,
        )
        state = self._initial_state("Continue the investigation")
        state["messages"] = [
            HumanMessage(content="old request " + "x" * 2000),
            AIMessage(content="old response " + "x" * 2000),
            HumanMessage(content="old follow-up " + "x" * 2000),
            AIMessage(content="old result " + "x" * 2000),
            HumanMessage(content="old decision " + "x" * 2000),
            HumanMessage(content="latest request"),
        ]

        result = await app.ainvoke(
            state,
            config={"configurable": {"thread_id": "summary-memory-folded"}, "recursion_limit": 24},
        )

        self.assertEqual(result["summary"], folded_memory)
        self.assertLessEqual(estimate_summary_tokens(result["summary"]), 120)
        self.assertEqual(len(agent_llm.invocations), 3)
        self.assertIn("Memory compaction mode", str(agent_llm.invocations[1]))
        self.assertIn(oversized_memory, str(agent_llm.invocations[1]))

    async def test_auto_summary_truncates_memory_when_the_fold_stays_oversized(self):
        oversized_memory = "\n".join(
            f"- fact {index}: keep the audit going with detail {index}" for index in range(40)
        )
        still_oversized_memory = "\n".join(
            f"- fact {index}: still too long for the memory budget" for index in range(30)
        )
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(content=oversized_memory),
                AIMessage(content=still_oversized_memory),
                AIMessage(content="Continue from memory."),
            ],
            tools=[],
            model_supports_tools=False,
            summary_threshold=1,
            summary_keep_last=1,
            summary_max_tokens=120,
        )
        state = self._initial_state("Continue the investigation")
        state["messages"] = [
            HumanMessage(content="old request " + "x" * 2000),
            AIMessage(content="old response " + "x" * 2000),
            HumanMessage(content="old follow-up " + "x" * 2000),
            AIMessage(content="old result " + "x" * 2000),
            HumanMessage(content="old decision " + "x" * 2000),
            HumanMessage(content="latest request"),
        ]

        result = await app.ainvoke(
            state,
            config={"configurable": {"thread_id": "summary-memory-truncated"}, "recursion_limit": 24},
        )

        memory = result["summary"]
        self.assertLessEqual(estimate_summary_tokens(memory), 120)
        self.assertTrue(memory.startswith("- fact 0: still too long"))
        self.assertTrue(memory.endswith("... [memory truncated]"))
        self.assertNotIn("- fact 29:", memory)

    async def test_auto_summary_keeps_memory_when_the_fold_call_fails(self):
        oversized_memory = "\n".join(
            f"- fact {index}: keep the audit going with detail {index}" for index in range(40)
        )
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(content=oversized_memory),
                RuntimeError("provider unavailable"),
                AIMessage(content="Continue from memory."),
            ],
            tools=[],
            model_supports_tools=False,
            summary_threshold=1,
            summary_keep_last=1,
            summary_max_tokens=120,
        )
        state = self._initial_state("Continue the investigation")
        state["messages"] = [
            HumanMessage(content="old request " + "x" * 2000),
            AIMessage(content="old response " + "x" * 2000),
            HumanMessage(content="old follow-up " + "x" * 2000),
            AIMessage(content="old result " + "x" * 2000),
            HumanMessage(content="old decision " + "x" * 2000),
            HumanMessage(content="latest request"),
        ]

        result = await app.ainvoke(
            state,
            config={"configurable": {"thread_id": "summary-memory-fold-failed"}, "recursion_limit": 24},
        )

        memory = result["summary"]
        self.assertLessEqual(estimate_summary_tokens(memory), 120)
        self.assertTrue(memory.startswith("- fact 0: keep the audit going"))
        self.assertTrue(memory.endswith("... [memory truncated]"))

    async def test_auto_summary_keeps_history_when_memory_comes_back_empty(self):
        app, agent_llm = self._build_app(
            agent_responses=[AIMessage(content="   "), AIMessage(content="Continuing with full history.")],
            tools=[],
            model_supports_tools=False,
            summary_threshold=1,
            summary_keep_last=1,
        )
        state = self._initial_state("Continue the investigation")
        original_messages = [
            HumanMessage(content="old request " + "x" * 2000),
            AIMessage(content="old response " + "x" * 2000),
            HumanMessage(content="old follow-up " + "x" * 2000),
            AIMessage(content="old result " + "x" * 2000),
            HumanMessage(content="old decision " + "x" * 2000),
            HumanMessage(content="latest request"),
        ]
        state["messages"] = list(original_messages)

        result = await app.ainvoke(
            state,
            config={"configurable": {"thread_id": "summary-empty-memory"}, "recursion_limit": 24},
        )

        self.assertEqual(result.get("summary", ""), "")
        self.assertEqual(
            [str(message.content) for message in result["messages"][:-1]],
            [str(message.content) for message in original_messages],
        )
        self.assertEqual(result["messages"][-1].content, "Continuing with full history.")

    async def test_request_user_input_interrupts_and_resumes_with_selected_option(self):
        tool = FakeTool("edit_file", "Success: File edited.")
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "request_user_input",
                            "args": {
                                "question": "Какой вариант выбираем?",
                                "options": ["direct_api", "keep_mcp"],
                                "recommended": "direct_api",
                            },
                            "id": "tc-choice-1",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "args": {"path": "demo.txt", "old_string": "a", "new_string": "b"},
                            "id": "tc-edit-1",
                        }
                    ]
                ),
                AIMessage(content="Режим выбран, правка готова."),
            ],
            tools=[request_user_input, tool],
            tool_metadata={
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
                "edit_file": ToolMetadata(name="edit_file", mutating=True),
            },
        )

        thread_config = {"configurable": {"thread_id": "await-user-input"}, "recursion_limit": 24}
        interrupted = await app.ainvoke(
            self._initial_state("Выбери стратегию и внеси правку"),
            config=thread_config,
        )

        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(tool.calls, [])
        self.assertEqual(len(agent_llm.invocations), 1)

        resumed = await app.ainvoke(Command(resume="direct_api"), config=thread_config)

        self.assertEqual(tool.calls, [{"path": "demo.txt", "old_string": "a", "new_string": "b"}])
        self.assertEqual(len(agent_llm.invocations), 3)
        self.assertEqual(resumed["turn_outcome"], "finish_turn")
        self.assertIn("правка готова", str(resumed["messages"][-1].content).lower())

    async def test_plain_text_without_await_control_does_not_pause_tool_execution(self):
        tool = FakeTool("edit_file", "Success: File edited.")
        app, _agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="Нужно уточнение, но сначала быстро подготовлю правку.",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "args": {"path": "demo.txt", "old_string": "a", "new_string": "b"},
                            "id": "tc-no-await-1",
                        }
                    ],
                ),
                AIMessage(content="Готово."),
            ],
            tools=[tool],
        )

        result = await app.ainvoke(
            self._initial_state("Подготовь правку"),
            config={"configurable": {"thread_id": "no-await-user-input"}, "recursion_limit": 24},
        )

        self.assertEqual(tool.calls, [{"path": "demo.txt", "old_string": "a", "new_string": "b"}])
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIn("готово", str(result["messages"][-1].content).lower())

    async def test_protocol_issue_routes_to_recovery_without_boolean_flag(self):
        read_tool = FakeTool("read_file", "unused")
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": ""}],
                ),
                AIMessage(content="Recovered without executing the malformed call."),
            ],
            tools=[read_tool],
            tool_metadata={"read_file": ToolMetadata(name="read_file", read_only=True)},
        )

        result = await app.ainvoke(
            self._initial_state("Прочитай README.md"),
            config={"configurable": {"thread_id": "protocol-issue-routing"}, "recursion_limit": 24},
        )

        self.assertEqual(read_tool.calls, [])
        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        self.assertNotIn("has_protocol_error", result)
        self.assertIn("recovered", str(result["messages"][-1].content).lower())

    async def test_read_only_inspection_code_dump_finishes_without_forced_edit(self):
        read_tool = FakeTool(
            "read_file",
            "document.addEventListener('keydown', event => { if (event.keyCode === 37) playerMove(-1); });",
        )
        edit_tool = FakeTool("edit_file", "Success: File edited.")
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"path": "script.js"},
                            "id": "tc-read-keys",
                        }
                    ],
                ),
                AIMessage(
                    content=(
                        "```javascript\n"
                        "document.addEventListener('keydown', event => {\n"
                        "  if (event.key === 'ArrowLeft') playerMove(-1);\n"
                        "});\n"
                        "```"
                    )
                ),
            ],
            tools=[read_tool, edit_tool],
            tool_metadata={
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "edit_file": ToolMetadata(name="edit_file", mutating=True),
            },
        )

        result = await app.ainvoke(
            self._initial_state("Обнови управление в script.js"),
            config={"configurable": {"thread_id": "recover-from-code-dump"}, "recursion_limit": 48},
        )

        self.assertEqual(read_tool.calls, [{"path": "script.js"}])
        self.assertEqual(edit_tool.calls, [])
        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        self.assertIn("arrowleft", str(result["messages"][-1].content).lower())

    async def test_analysis_after_read_only_inspection_finishes_without_recovery(self):
        read_tool = FakeTool(
            "read_file",
            "function saveScore(score) { localStorage.setItem('score', score); }",
        )
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"path": "script.js"},
                            "id": "tc-read-analysis",
                        }
                    ],
                ),
                AIMessage(
                    content=(
                        "В проекте используется JavaScript с сохранением данных в localStorage.\n\n"
                        "1. Архитектура простая: логика сосредоточена в одном файле.\n"
                        "2. Сильная сторона: код читается быстро и без лишних зависимостей.\n"
                        "3. Ограничение: хранение рекордов только в памяти браузера.\n\n"
                        "Итог: это законченный небольшой проект, который можно улучшить, "
                        "не меняя базовую структуру."
                    )
                ),
            ],
            tools=[read_tool],
            tool_metadata={
                "read_file": ToolMetadata(name="read_file", read_only=True),
            },
        )

        result = await app.ainvoke(
            self._initial_state("Сделай анализ кода в папке"),
            config={"configurable": {"thread_id": "analysis-no-recovery"}, "recursion_limit": 36},
        )

        self.assertEqual(read_tool.calls, [{"path": "script.js"}])
        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        self.assertIn("итог", str(result["messages"][-1].content).lower())

    async def test_retryable_cli_exec_issue_gets_auto_retry_then_finishes_after_success(self):
        cli_tool = FakeTool(
            "cli_exec",
            "ERROR[VALIDATION]: Foreground service/server commands are not supported. Use run_background_process.",
        )
        bg_tool = FakeTool("run_background_process", "Success: process started")
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "cli_exec", "args": {"command": "python -m http.server"}, "id": "tc-1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "run_background_process",
                            "args": {"command": ["python", "-m", "http.server"]},
                            "id": "tc-2",
                        }
                    ],
                ),
                AIMessage(content="Готово."),
            ],
            tools=[cli_tool, bg_tool],
        )

        result = await app.ainvoke(
            self._initial_state("Сделай задачу"),
            config={"configurable": {"thread_id": "single-retry-success"}, "recursion_limit": 48},
        )

        self.assertEqual(len(cli_tool.calls), 1)
        self.assertEqual(len(bg_tool.calls), 1)
        self.assertGreaterEqual(len(agent_llm.invocations), 3)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])

    async def test_retry_counter_resets_after_success_before_later_retryable_error(self):
        cli_tool = FakeTool(
            "cli_exec",
            "ERROR[VALIDATION]: Foreground service/server commands are not supported. Use run_background_process.",
        )
        bg_tool = FakeTool("run_background_process", "Success: process started")
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "cli_exec", "args": {"command": "python -m http.server"}, "id": "tc-1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "run_background_process",
                            "args": {"command": ["python", "-m", "http.server"]},
                            "id": "tc-2",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "cli_exec", "args": {"command": "python -m http.server"}, "id": "tc-3"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "run_background_process",
                            "args": {"command": ["python", "-m", "http.server"]},
                            "id": "tc-4",
                        }
                    ],
                ),
                AIMessage(content="Готово."),
            ],
            tools=[cli_tool, bg_tool],
        )

        result = await app.ainvoke(
            self._initial_state("Сделай задачу"),
            config={"configurable": {"thread_id": "retry-reset-after-success"}, "recursion_limit": 64},
        )

        self.assertEqual(len(cli_tool.calls), 2)
        self.assertEqual(len(bg_tool.calls), 2)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        self.assertEqual(str(result["messages"][-1].content), "Готово.")

    async def test_non_retryable_mutating_execution_issue_handoffs_without_auto_retry(self):
        tool = FakeTool("demo_tool", "ERROR[EXECUTION]: boom")
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"action": "a"}, "id": "tc-a"}]),
            ],
            tools=[tool],
            tool_metadata={
                "demo_tool": ToolMetadata(
                    name="demo_tool",
                    read_only=False,
                    mutating=True,
                )
            },
        )

        result = await app.ainvoke(
            self._initial_state("Обнови проект"),
            config={"configurable": {"thread_id": "retry-budget-exhausted"}, "recursion_limit": 48},
        )

        self.assertEqual(len(tool.calls), 1)
        self.assertGreaterEqual(len(agent_llm.invocations), 1)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        self.assertGreaterEqual(result["steps"], 2)
        self.assertIn("boom", str(result["messages"][-1].content).lower())

    async def test_read_only_cli_probe_failure_returns_to_agent_instead_of_handoff(self):
        cli_tool = FakeTool(
            "cli_exec",
            "ERROR[EXECUTION]: Command failed with Exit Code 1. Output: 000",
        )
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "cli_exec",
                            "args": {"command": "curl -s -o NUL -w \"%{http_code}\" http://localhost:8000"},
                            "id": "tc-probe",
                        }
                    ],
                ),
                AIMessage(content="Сервер больше не отвечает, значит остановка подтверждена."),
            ],
            tools=[cli_tool],
        )

        result = await app.ainvoke(
            self._initial_state("Останови сервер и проверь, что он больше не отвечает"),
            config={"configurable": {"thread_id": "inspect-cli-failure"}, "recursion_limit": 48},
        )

        self.assertEqual(len(cli_tool.calls), 1)
        self.assertGreaterEqual(len(agent_llm.invocations), 2)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        self.assertIn("остановка подтверждена", str(result["messages"][-1].content).lower())

    async def test_validation_missing_path_handoffs_without_auto_retry(self):
        tool = FakeTool("edit_file", "ERROR[VALIDATION]: Missing required field: path.")
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "edit_file", "args": {"old_string": "x", "new_string": "y"}, "id": "tc-v1"}],
                ),
            ],
            tools=[tool],
            tool_metadata={
                "edit_file": ToolMetadata(
                    name="edit_file",
                    read_only=False,
                    mutating=True,
                )
            },
        )

        result = await app.ainvoke(
            self._initial_state("Исправь файл"),
            config={"configurable": {"thread_id": "validation-path"}, "recursion_limit": 36},
        )

        self.assertEqual(len(tool.calls), 1)
        self.assertGreaterEqual(len(agent_llm.invocations), 1)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        self.assertGreaterEqual(result["steps"], 2)
        self.assertIn("missing required field", str(result["messages"][-1].content).lower())

    async def test_edit_file_match_failure_returns_to_agent_and_allows_alternate_tool_path(self):
        def _edit_result(args):
            if args.get("old_string") == "bad-snippet":
                return (
                    "ERROR[VALIDATION]: Could not find a match for 'old_string'.\n"
                    "Make sure you are replacing existing lines and DID NOT include line numbers in old_string."
                )
            return "Success: File edited.\n\nDiff:\n```diff\n-old\n+new\n```"

        edit_tool = FakeTool("edit_file", _edit_result)
        read_tool = FakeTool("read_file", "1: import sys\n2: import logging\n3: def main(): pass")
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "args": {
                                "path": "weather7.py",
                                "old_string": "bad-snippet",
                                "new_string": "patched-snippet",
                            },
                            "id": "tc-edit-1",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"path": "weather7.py"},
                            "id": "tc-read-1",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "args": {
                                "path": "weather7.py",
                                "old_string": "import sys\nimport logging",
                                "new_string": "import sys\nimport logging\nimport json",
                            },
                            "id": "tc-edit-2",
                        }
                    ],
                ),
                AIMessage(content="Исправление применено после дополнительного чтения файла."),
            ],
            tools=[edit_tool, read_tool],
            tool_metadata={
                "edit_file": ToolMetadata(name="edit_file", read_only=False, mutating=True),
                "read_file": ToolMetadata(name="read_file", read_only=True, mutating=False),
            },
        )

        result = await app.ainvoke(
            self._initial_state("Исправь файл"),
            config={"configurable": {"thread_id": "edit-file-retry"}, "recursion_limit": 64},
        )

        self.assertEqual(len(edit_tool.calls), 2)
        self.assertEqual(len(read_tool.calls), 1)
        self.assertGreaterEqual(len(agent_llm.invocations), 4)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        self.assertTrue(any(msg.tool_call_id == "tc-edit-1" and msg.status == "error" for msg in tool_messages))
        self.assertIn("дополнительного чтения", str(result["messages"][-1].content).lower())

    async def test_powershell_parse_error_can_recover_via_read_edit_and_rerun(self):
        def _cli_result(args):
            command = str(args.get("command") or "")
            if "Get-Weather7DaysFree.ps1" in command and "fixed" not in command:
                return "ERROR[EXECUTION]: Unexpected token '{' in expression or statement."
            return "Success: script completed"

        cli_tool = FakeTool("cli_exec", _cli_result)
        read_tool = FakeTool("read_file", "$data = \"JSON\" {\nWrite-Host 'broken'\n}")
        edit_tool = FakeTool("edit_file", "Success: File edited.\n\nDiff:\n```diff\n-old\n+fixed\n```")
        app, _agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "cli_exec", "args": {"command": "powershell -ExecutionPolicy Bypass -File Get-Weather7DaysFree.ps1"}, "id": "tc-cli-1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "read_file", "args": {"path": "Get-Weather7DaysFree.ps1"}, "id": "tc-read-1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "edit_file", "args": {"path": "Get-Weather7DaysFree.ps1", "old_string": "\"JSON\" {", "new_string": "\"JSON\"; # fixed"}, "id": "tc-edit-1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "cli_exec", "args": {"command": "powershell -ExecutionPolicy Bypass -File Get-Weather7DaysFree.ps1 # fixed"}, "id": "tc-cli-2"}],
                ),
                AIMessage(content="Скрипт исправлен и повторный запуск завершился успешно."),
            ],
            tools=[cli_tool, read_tool, edit_tool],
            tool_metadata={
                "cli_exec": ToolMetadata(name="cli_exec", mutating=True),
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "edit_file": ToolMetadata(name="edit_file", mutating=True),
            },
        )

        result = await app.ainvoke(
            self._initial_state("Исправь скрипт и добейся успешного запуска"),
            config={"configurable": {"thread_id": "powershell-parse-recovery"}, "recursion_limit": 72},
        )

        self.assertEqual(len(cli_tool.calls), 2)
        self.assertEqual(len(read_tool.calls), 1)
        self.assertEqual(len(edit_tool.calls), 1)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        self.assertIn("повторный запуск", str(result["messages"][-1].content).lower())

    async def test_read_file_not_found_can_fallback_to_direct_verification(self):
        bg_tool = FakeTool("run_background_process", "Success: Process started with PID 4321.")
        read_tool = FakeTool("read_file", "ERROR[NOT_FOUND]: File 'demo.output' not found.")
        cli_tool = FakeTool("cli_exec", "stdout: service healthy")
        app, _agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "run_background_process", "args": {"command": ["demo.exe"], "cwd": "."}, "id": "tc-bg-1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "read_file", "args": {"path": "demo.output"}, "id": "tc-read-1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "cli_exec", "args": {"command": "Get-Process | Select-Object -First 1"}, "id": "tc-cli-verify"}],
                ),
                AIMessage(content="Файл вывода не появился, поэтому состояние проверено напрямую через процесс."),
            ],
            tools=[bg_tool, read_tool, cli_tool],
            tool_metadata={
                "run_background_process": ToolMetadata(name="run_background_process", mutating=True),
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "cli_exec": ToolMetadata(name="cli_exec", mutating=True),
            },
        )

        result = await app.ainvoke(
            self._initial_state("Запусти процесс и проверь результат даже если output-файл не появится"),
            config={"configurable": {"thread_id": "read-fallback"}, "recursion_limit": 64},
        )

        self.assertEqual(len(bg_tool.calls), 1)
        self.assertEqual(len(read_tool.calls), 1)
        self.assertEqual(len(cli_tool.calls), 1)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        self.assertIn("проверено напрямую", str(result["messages"][-1].content).lower())

    async def test_approval_denied_finishes_without_retry_and_without_tool_execution(self):
        tool = FakeTool("danger_tool", "Изменение применено.")
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-d1"}]),
                AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "again"}, "id": "tc-d2"}]),
            ],
            tools=[tool],
            enable_approvals=True,
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )

        thread_config = {"configurable": {"thread_id": "approval-thread"}, "recursion_limit": 36}
        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)

        resumed = await app.ainvoke(Command(resume={"approved": False}), config=thread_config)

        self.assertEqual(tool.calls, [])
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(resumed["turn_outcome"], "finish_turn")
        self.assertIsNone(resumed["open_tool_issue"])
        self.assertIn("declined", str(resumed["messages"][-1].content).lower())

    async def test_mutating_tool_interrupts_and_executes_only_after_approval(self):
        tool = FakeTool("edit_file", "Success: updated")
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"path": "demo.txt"}, "id": "tc-e1"}]),
                AIMessage(content="Готово."),
            ],
            tools=[tool],
            enable_approvals=True,
            tool_metadata={
                "edit_file": ToolMetadata(
                    name="edit_file",
                    mutating=True,
                )
            },
        )

        thread_config = {"configurable": {"thread_id": "approval-approve-thread"}, "recursion_limit": 36}
        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)

        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(tool.calls, [])

        resumed = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)

        self.assertEqual(tool.calls, [{"path": "demo.txt"}])
        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertEqual(resumed["turn_outcome"], "finish_turn")
        self.assertIsNone(resumed["open_tool_issue"])
        self.assertIn("готово", str(resumed["messages"][-1].content).lower())

    async def test_provider_safe_order_is_kept_during_internal_retry(self):
        cli_tool = FakeTool(
            "cli_exec",
            "ERROR[VALIDATION]: Foreground service/server commands are not supported. Use run_background_process.",
        )
        bg_tool = FakeTool("run_background_process", "Success: process started")
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "cli_exec", "args": {"command": "python -m http.server"}, "id": "tc-1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "run_background_process",
                            "args": {"command": ["python", "-m", "http.server"]},
                            "id": "tc-2",
                        }
                    ],
                ),
                AIMessage(content="Готово после повтора."),
            ],
            tools=[cli_tool, bg_tool],
            agent_llm_cls=ProviderSafeFakeLLM,
        )

        result = await app.ainvoke(
            self._initial_state("Сделай шаги"),
            config={"configurable": {"thread_id": "provider-safe"}, "recursion_limit": 48},
        )

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(cli_tool.calls), 1)
        self.assertEqual(len(bg_tool.calls), 1)
        self.assertGreaterEqual(len(agent_llm.invocations), 3)

    async def test_pending_tool_call_executes_even_when_last_message_is_not_ai(self):
        tool = FakeTool("demo_tool", "Success: done")
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    id="dup-ai-id",
                    tool_calls=[{"name": "demo_tool", "args": {"step": "new"}, "id": "tc-new"}],
                ),
                AIMessage(content="Готово."),
            ],
            tools=[tool],
        )

        state = self._initial_state("Сделай снова")
        state["messages"] = [
            HumanMessage(content="Старый запрос"),
            AIMessage(
                content="",
                id="dup-ai-id",
                tool_calls=[{"name": "demo_tool", "args": {"step": "old"}, "id": "tc-old"}],
            ),
            ToolMessage(content="Success: old", tool_call_id="tc-old", name="demo_tool"),
            HumanMessage(content="Сделай снова"),
        ]

        result = await app.ainvoke(
            state,
            config={"configurable": {"thread_id": "pending-tool-not-last-ai"}, "recursion_limit": 36},
        )

        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(tool.calls[0], {"step": "new"})
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertTrue(any(isinstance(msg, ToolMessage) and msg.tool_call_id == "tc-new" for msg in result["messages"]))

    async def test_loop_budget_with_pending_tool_call_finishes_with_handoff_without_dangling_call(self):
        tool = FakeTool("demo_tool", "Success: done")
        app, agent_llm = self._build_app(
            agent_responses=[
                AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"x": 1}, "id": "tc-loop"}]),
            ],
            tools=[tool],
            max_loops=1,
        )

        result = await app.ainvoke(
            self._initial_state("Сделай задачу"),
            config={"configurable": {"thread_id": "loop-budget-pending-call"}, "recursion_limit": 24},
        )

        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(tool.calls, [])
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        self.assertEqual(result["steps"], 1)

        messages = result.get("messages", [])
        self.assertTrue(messages)
        self.assertIn("limit", str(messages[-1].content).lower())
        self.assertFalse(any(isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None) for msg in messages))

    def test_build_graph_config_keeps_recursion_limit_as_technical_overhead(self):
        config = build_graph_config("thread-1", 50)
        self.assertEqual(config["configurable"]["thread_id"], "thread-1")
        self.assertGreater(config["recursion_limit"], 50)
        self.assertEqual(config["recursion_limit"], 308)

    def test_workflow_routes_update_step_to_agent_without_selector(self):
        config = self._make_config()
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[FakeTool("demo_tool", "Success: done")],
            llm_with_tools=FakeLLM([]),
        )

        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())

        self.assertIn("agent", app.nodes)
        self.assertIn("tools", app.nodes)

    def test_workflow_omits_approval_node_when_approvals_disabled(self):
        config = self._make_config(enable_approvals=False)
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[FakeTool("demo_tool", "Success: done")],
            llm_with_tools=FakeLLM([]),
        )

        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())

        self.assertNotIn("approval", app.nodes)
        self.assertIn("tools", app.nodes)


if __name__ == "__main__":
    unittest.main()
