import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.summarize_policy import estimate_tokens, should_summarize
from ui.runtime_payloads import (
    append_project_label,
    build_summary_progress_payload,
    build_transcript_payload,
    build_ui_payload,
    build_user_choice_payload,
    generate_chat_title,
)


class RuntimePayloadTests(unittest.TestCase):
    def test_build_summary_progress_payload_reports_context_budget(self):
        config = type("Config", (), {"summary_threshold": 100, "summary_keep_last": 4})()

        payload = build_summary_progress_payload(
            config,
            {"messages": [HumanMessage(content="hello world")], "summary": ""},
        )

        self.assertGreater(payload["estimated_tokens"], 0)
        self.assertEqual(payload["threshold"], 100)
        self.assertGreater(payload["trigger_tokens"], payload["threshold"])
        self.assertEqual(
            payload["remaining_tokens"],
            max(0, payload["trigger_tokens"] - payload["estimated_tokens"]),
        )
        self.assertAlmostEqual(
            payload["progress"],
            max(0.0, 1.0 - (payload["estimated_tokens"] / payload["threshold"])),
        )
        self.assertGreaterEqual(payload["progress"], 0.0)
        self.assertLessEqual(payload["progress"], 1.0)
        self.assertFalse(payload["will_summarize"])

    def test_build_summary_progress_payload_includes_reserved_context_tokens(self):
        config = type(
            "Config",
            (),
            {
                "summary_threshold": 100,
                "summary_keep_last": 4,
                "summary_reserved_tokens": 25,
            },
        )()
        messages = [HumanMessage(content="hello world")]

        payload = build_summary_progress_payload(config, {"messages": messages, "summary": ""})

        self.assertEqual(payload["reserved_tokens"], 25)
        self.assertEqual(payload["estimated_tokens"], estimate_tokens(messages) + 25)
        self.assertEqual(
            payload["remaining_tokens"],
            max(0, payload["trigger_tokens"] - payload["estimated_tokens"]),
        )
        self.assertAlmostEqual(
            payload["progress"],
            max(0.0, 1.0 - (payload["estimated_tokens"] / payload["threshold"])),
        )

    def test_build_summary_progress_payload_reports_summary_and_provider_input_separately(self):
        config = type(
            "Config",
            (),
            {"summary_threshold": 10000, "summary_keep_last": 4, "summary_reserved_tokens": 25},
        )()
        messages = [HumanMessage(content="recent question")]

        payload = build_summary_progress_payload(
            config,
            {
                "messages": messages,
                "summary": "compressed memory " * 20,
                "token_usage": {"input_tokens": 229094, "output_tokens": 2426},
            },
        )

        self.assertGreater(payload["summary_tokens"], 0)
        self.assertEqual(
            payload["estimated_tokens"],
            estimate_tokens(messages) + payload["summary_tokens"] + 25,
        )
        self.assertEqual(payload["provider_input_tokens"], 229094)

    def test_should_summarize_uses_reserved_context_tokens(self):
        messages = [
            HumanMessage(content="old question"),
            AIMessage(content="old answer"),
            HumanMessage(content="new question"),
        ]

        self.assertTrue(
            should_summarize(
                messages,
                threshold=50,
                keep_last=1,
                reserved_tokens=1000,
            )
        )

    def test_build_summary_progress_payload_marks_ready_to_summarize(self):
        config = type("Config", (), {"summary_threshold": 10, "summary_keep_last": 1})()
        messages = [
            HumanMessage(content="token " * 1200),
            AIMessage(content="old answer " * 1200),
            HumanMessage(content="new question"),
        ]

        payload = build_summary_progress_payload(config, {"messages": messages})

        self.assertEqual(payload["threshold"], 10)
        self.assertEqual(payload["remaining_tokens"], 0)
        self.assertEqual(payload["progress"], 0.0)
        self.assertTrue(payload["will_summarize"])

    def test_generate_chat_title_strips_common_prefixes_and_limits_length(self):
        self.assertEqual(
            generate_chat_title("Помоги скачать и настроить Apache на Windows"),
            "Скачать и настроить Apache на Windows",
        )
        self.assertEqual(generate_chat_title("   \n   "), "New Chat")
        self.assertTrue(generate_chat_title("сделай " + ("очень длинный запрос " * 10)).endswith("…"))

    def test_append_project_label_uses_last_two_segments(self):
        project_path = Path("D:/work/client/demo-app")
        self.assertEqual(append_project_label("New Chat", project_path), "New Chat [client/demo-app]")

    def test_build_user_choice_payload_marks_recommended_entry(self):
        payload = build_user_choice_payload(
            {
                "question": "Continue?",
                "recommended": "option_2",
                "options": [
                    {"label": "Skip", "submit_text": "skip"},
                    {"label": "Retry", "submit_text": "retry"},
                ],
            }
        )

        self.assertEqual(payload["question"], "Continue?")
        self.assertEqual(payload["recommended_key"], "Retry")
        self.assertEqual([item["recommended"] for item in payload["options"]], [False, True])

    def test_build_transcript_payload_prefers_full_transcript_after_compaction(self):
        payload = build_transcript_payload(
            {
                "summary": "compressed",
                "messages": [HumanMessage(content="latest request"), AIMessage(content="latest answer")],
                "transcript_messages": [
                    HumanMessage(content="old request"),
                    AIMessage(content="old answer"),
                    HumanMessage(content="latest request"),
                    AIMessage(content="latest answer"),
                ],
            }
        )

        self.assertEqual([turn["user_text"] for turn in payload["turns"]], ["old request", "latest request"])
        self.assertEqual(payload["summary_notice"], "Model context was compressed, but the full chat history is preserved below.")

    def test_build_transcript_payload_restores_turns_attachments_and_tool_args(self):
        payload = build_transcript_payload(
            {
                "summary": "compressed",
                "messages": [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "Покажи diff"},
                            {
                                "type": "image",
                                "path": "C:/tmp/screenshot.png",
                                "mime_type": "image/png",
                                "file_name": "screenshot.png",
                                "attachment_id": "img-1",
                            },
                        ]
                    ),
                    AIMessage(
                        content="Смотрю изменения",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "name": "edit_file",
                                "args": {"path": "app.py", "oldText": "a", "newText": "b"},
                            }
                        ],
                    ),
                    ToolMessage(
                        content="```diff\n-a\n+b\n```",
                        tool_call_id="call-1",
                        name="edit_file",
                    ),
                ],
            }
        )

        self.assertIn("compressed automatically", payload["summary_notice"])
        self.assertEqual(len(payload["turns"]), 1)
        turn = payload["turns"][0]
        self.assertEqual(turn["user_text"], "Покажи diff")
        self.assertEqual(turn["attachments"][0]["id"], "img-1")
        tool_block = turn["blocks"][-1]["payload"]
        self.assertEqual(tool_block["args"]["path"], "app.py")
        self.assertEqual(tool_block["diff"], "-a\n+b")

    def test_build_transcript_payload_skips_replayed_pretool_commentary_after_tool(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Проверь файл"),
                    AIMessage(
                        content="Проверю файл перед изменением.",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "name": "read_file",
                                "args": {"path": "index.html"},
                            }
                        ],
                    ),
                    ToolMessage(content="ok", tool_call_id="call-1", name="read_file"),
                    AIMessage(content="Проверю файл перед изменением.\n\nТеперь внесу правку."),
                ]
            }
        )

        turn = payload["turns"][0]
        assistant_blocks = [block["markdown"] for block in turn["blocks"] if block["type"] == "assistant"]
        self.assertEqual(assistant_blocks, ["Проверю файл перед изменением.", "Теперь внесу правку."])

    def test_build_transcript_payload_does_not_parse_assistant_thought_markdown(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Сделай вывод"),
                    AIMessage(content="<think>Проверяю ограничения.</think>Готово"),
                ],
            }
        )

        self.assertEqual(len(payload["turns"]), 1)
        assistant_block = payload["turns"][0]["blocks"][0]
        self.assertEqual(assistant_block["type"], "assistant")
        self.assertEqual(assistant_block["markdown"], "Готово")
        self.assertNotIn("thought_markdown", assistant_block)

    def test_build_transcript_payload_ignores_structured_assistant_reasoning(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Проверь"),
                    AIMessage(
                        content=[
                            {"type": "reasoning", "text": "Сначала сверю ограничения."},
                            {"type": "text", "text": "Итог готов."},
                        ]
                    ),
                ]
            }
        )

        assistant_block = payload["turns"][0]["blocks"][0]
        self.assertEqual(assistant_block["type"], "assistant")
        self.assertEqual(assistant_block["markdown"], "Итог готов.")
        self.assertNotIn("thought_markdown", assistant_block)

    def test_build_transcript_payload_ignores_reasoning_details_text(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Проверь"),
                    AIMessage(
                        content=[
                            {"type": "reasoning.text", "text": "Скрытое рассуждение."},
                            {"type": "text", "text": "Итог готов."},
                        ]
                    ),
                ]
            }
        )

        assistant_block = payload["turns"][0]["blocks"][0]
        self.assertEqual(assistant_block["markdown"], "Итог готов.")
        self.assertNotIn("Скрытое", assistant_block["markdown"])

    def test_build_transcript_payload_restores_hidden_internal_notice_as_notice_block(self):
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
                                "ui_notice": "Нужен новый запрос.",
                            }
                        },
                    ),
                ]
            }
        )

        turn = payload["turns"][0]
        self.assertEqual(turn["user_text"], "Проверь завершение")
        self.assertEqual(
            turn["blocks"],
            [{"type": "notice", "message": "Нужен новый запрос.", "level": "warning"}],
        )

    def test_build_transcript_payload_appends_last_run_stats_to_final_turn(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Подведи итог"),
                    AIMessage(content="Готово."),
                ],
            },
            last_run_stats="3.1s  ↓ 5328  ↑ 106",
        )

        turn = payload["turns"][0]
        self.assertEqual([block["type"] for block in turn["blocks"]], ["assistant", "stats"])
        self.assertEqual(turn["blocks"][-1]["stats"], "3.1s  ↓ 5328  ↑ 106")

    def test_build_transcript_payload_does_not_attach_stats_to_empty_trailing_turn(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Первый запрос"),
                    AIMessage(content="Первый ответ"),
                    HumanMessage(content="Второй запрос"),
                ],
            },
            last_run_stats="2.0s  ↓ 100  ↑ 20",
        )

        self.assertEqual([block["type"] for block in payload["turns"][0]["blocks"]], ["assistant"])
        self.assertEqual(payload["turns"][1]["blocks"], [])

    def test_build_user_choice_payload_includes_default_selection(self):
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
        self.assertEqual(payload["recommended_key"], "Ввести ключ API")
        recommended = [option for option in payload["options"] if option["recommended"]]
        self.assertEqual(len(recommended), 1)
        self.assertEqual(recommended[0]["submit_text"], "Ввести ключ API")


class RuntimePayloadAsyncTests(unittest.IsolatedAsyncioTestCase):
    def _config(self):
        return SimpleNamespace(
            provider="openai",
            openai_model="gpt-4o",
            gemini_model="gemini-pro",
            summary_threshold=100,
            summary_keep_last=4,
            summary_reserved_tokens=0,
            checkpoint_backend="memory",
            enable_approvals=True,
            debug=False,
        )

    def _snapshot(self):
        return SimpleNamespace(
            session_id="session-1",
            thread_id="thread-1",
            title="Chat",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
            checkpoint_backend="memory",
            checkpoint_target="memory",
            last_run_stats="",
            approval_mode="prompt",
        )

    async def _build_payload(self, *, tasks):
        class FakeApp:
            async def aget_state(self, _config):
                return SimpleNamespace(values={}, tasks=tasks)

        store = SimpleNamespace(list_sessions=lambda: [])
        registry = SimpleNamespace(
            tools=[],
            metadata={},
            model_capabilities={},
            checkpoint_info={},
            get_runtime_status_lines=lambda: [],
        )
        return await build_ui_payload(
            self._config(),
            registry,
            store,
            self._snapshot(),
            agent_app=FakeApp(),
        )

    async def test_build_ui_payload_omits_pending_user_choice_without_runtime_interrupt(self):
        payload = await self._build_payload(tasks=[])

        self.assertNotIn("pending_user_choice", payload)
        self.assertNotIn("checkpoint_notice", payload)
