import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.summarize_policy import estimate_tokens, should_summarize
from ui.runtime_payloads import (
    append_project_label,
    build_summary_progress_payload,
    build_transcript_payload,
    build_ui_payload,
    build_user_choice_payload,
    generate_chat_title,
    generate_chat_title_with_llm,
    validate_chat_title,
)


class RuntimePayloadTests(unittest.IsolatedAsyncioTestCase):
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
        # Progress excludes the fixed reserve: only history fills the compactable span.
        self.assertAlmostEqual(
            payload["progress"],
            1.0 - (estimate_tokens(messages) / (payload["threshold"] - 25)),
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

    def test_summary_progress_uses_mid_run_tool_boundaries(self):
        config = SimpleNamespace(summary_threshold=1000, summary_keep_last=1)
        messages = [
            HumanMessage(content="active task"),
            AIMessage(content="", tool_calls=[{"id": "tc-1", "name": "read_file", "args": {}}]),
            ToolMessage(content="large output " * 2000, tool_call_id="tc-1"),
            AIMessage(content="", tool_calls=[{"id": "tc-2", "name": "read_file", "args": {}}]),
            ToolMessage(content="recent output", tool_call_id="tc-2"),
        ]
        entry = build_summary_progress_payload(config, {"messages": messages, "steps": 0})
        active = build_summary_progress_payload(config, {"messages": messages, "steps": 2})
        self.assertFalse(entry["will_summarize"])
        self.assertTrue(active["will_summarize"])
        self.assertEqual(active["progress"], 0.0)

    def test_generate_chat_title_strips_common_prefixes_and_limits_length(self):
        self.assertEqual(
            generate_chat_title("Помоги скачать и настроить Apache на Windows"),
            "Скачать и настроить Apache на Windows",
        )
        self.assertEqual(generate_chat_title("   \n   "), "New Chat")
        self.assertTrue(generate_chat_title("сделай " + ("очень длинный запрос " * 10)).endswith("…"))

    def test_validate_chat_title_accepts_question_style_title(self):
        self.assertEqual(validate_chat_title("Что на изображении?"), "Что на изображении")
        self.assertEqual(validate_chat_title("Title: Настройка сети."), "Настройка сети")
        self.assertEqual(validate_chat_title('"Анализ данных"'), "Анализ данных")
        self.assertEqual(validate_chat_title("Оптимизация запросов"), "Оптимизация запросов")

    def test_validate_chat_title_trims_long_titles_to_word_budget(self):
        self.assertEqual(
            validate_chat_title("Настройка Apache веб сервера на Windows"),
            "Настройка Apache веб сервера",
        )
        self.assertEqual(
            validate_chat_title("Анализ логов и поиск ошибок в приложении"),
            "Анализ логов и поиск",
        )

    def test_validate_chat_title_accepts_single_word_term(self):
        self.assertEqual(validate_chat_title("Солверы"), "Солверы")
        self.assertEqual(validate_chat_title("Docker"), "Docker")
        self.assertEqual(validate_chat_title("Title:\nНастройка"), "Настройка")
        self.assertIsNone(validate_chat_title("Ок"))
        self.assertIsNone(validate_chat_title(""))

    def test_validate_chat_title_strips_deepseek_think_blocks(self):
        self.assertEqual(
            validate_chat_title("<think>Нужно дать короткий заголовок из 2-4 слов</think>\nЧто такое солверы"),
            "Что такое солверы",
        )
        self.assertEqual(
            validate_chat_title("<think>reasoning here</think>Настройка Apache на Windows"),
            "Настройка Apache на Windows",
        )
        self.assertIsNone(validate_chat_title("<think>Только рассуждение без заголовка</think>"))

    def test_validate_chat_title_rejects_invalid_titles(self):
        self.assertIsNone(validate_chat_title("Ок"))
        self.assertIsNone(validate_chat_title("the"))
        self.assertIsNone(validate_chat_title(""))
        self.assertIsNone(validate_chat_title("This is a test title"))

    def test_validate_chat_title_rejects_leaked_reasoning_prefixes(self):
        # DeepSeek-compatible gateways can return reasoning as plain text, so the
        # first words of the leaked plan look like a title.
        self.assertIsNone(validate_chat_title("We need answer only with a short title"))
        self.assertIsNone(validate_chat_title("We need answer title for the chat"))
        self.assertIsNone(validate_chat_title("We should keep it short"))
        self.assertIsNone(validate_chat_title("I need to produce a title"))
        self.assertIsNone(validate_chat_title("Let me think about the title"))
        self.assertIsNone(validate_chat_title("Only answer with the title"))
        self.assertIsNone(validate_chat_title("Нужно придумать короткий заголовок"))
        self.assertIsNone(validate_chat_title("Пользователь просит заголовок чата"))
        # Topical titles must survive the filter.
        self.assertEqual(validate_chat_title("Настройка Apache на Windows"), "Настройка Apache на Windows")
        self.assertEqual(validate_chat_title("OnlyOffice документация сервера"), "OnlyOffice документация сервера")
        self.assertEqual(validate_chat_title("Weneedness анализ термина"), "Weneedness анализ термина")

    def test_validate_chat_title_uses_visible_text_blocks(self):
        hidden_blocks = [
            {"type": "reasoning", "text": "Внутренний план"},
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Внутренний план"}]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Внутренний план"}]},
            {"type": "reasoning.text", "text": "Внутренний план"},
            {"type": "reasoning_text", "text": "Внутренний план"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "thinking", "thinking": "Внутренний план", "signature": "opaque"},
            {"type": "analysis", "text": "Внутренний план"},
            {"type": "text", "text": "Внутренний план", "thought": True},
        ]
        for hidden in hidden_blocks:
            with self.subTest(block_type=hidden["type"], fields=list(hidden)):
                self.assertEqual(
                    validate_chat_title([
                        hidden,
                        {"type": "text", "text": "Возможности "},
                        {"type": "output_text", "text": "ассистента"},
                    ]),
                    "Возможности ассистента",
                )
                self.assertIsNone(validate_chat_title([hidden]))

    def test_validate_chat_title_strips_think_tags_across_text_blocks(self):
        content = [
            {"type": "text", "text": "<think>Внутренний план. "},
            {"type": "text", "text": "Продолжение плана."},
            {"type": "text", "text": "</think>Возможности ассистента"},
        ]
        self.assertEqual(validate_chat_title(content), "Возможности ассистента")
        self.assertIsNone(validate_chat_title(content[:2]))
        payload = build_transcript_payload({"messages": [
            HumanMessage(content="Что ты умеешь?"), AIMessage(content=content),
        ]})
        self.assertEqual(payload["turns"][0]["blocks"][0]["markdown"], "Возможности ассистента")

    def test_validate_chat_title_rejects_non_answer_blocks(self):
        for content in (
            [{"type": "refusal", "refusal": "Невозможно выполнить запрос"}],
            [{"type": "reasoning", "encrypted_content": "opaque", "summary": []}],
            [{"type": "image", "url": "https://example.test/image.png"}],
        ):
            with self.subTest(content=content):
                self.assertIsNone(validate_chat_title(content))

    def test_validate_chat_title_handles_nested_and_tuple_content(self):
        self.assertEqual(
            validate_chat_title({"content": [
                {"type": "reasoning", "text": "Внутренний план"},
                {"type": "text", "text": "Настройка Apache"},
            ]}),
            "Настройка Apache",
        )
        self.assertEqual(
            validate_chat_title(({"type": "text", "text": "Настройка Apache"},)),
            "Настройка Apache",
        )

    async def test_generate_chat_title_with_llm_ignores_structured_reasoning(self):
        llm = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content=[
            {"type": "reasoning", "content": [
                {"type": "reasoning_text", "text": "We need answer only title."},
            ]},
            {"type": "text", "text": "Возможности ассистента"},
        ])))
        title = await generate_chat_title_with_llm(llm, "Какие у тебя инструменты?")
        self.assertEqual(title, "Возможности ассистента")
        llm.ainvoke.assert_awaited_once()

    async def test_generate_chat_title_with_llm_retries_with_format_feedback(self):
        llm = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
            AIMessage(content="We need answer only title. " + "Internal plan. " * 20),
            AIMessage(content="Возможности ассистента"),
        ]))
        logger = Mock()
        with patch("ui.runtime_payloads.asyncio.sleep", new_callable=AsyncMock):
            title = await generate_chat_title_with_llm(llm, "Какие у тебя инструменты?", logger)
        self.assertEqual(title, "Возможности ассистента")
        prompts = [call.args[0] for call in llm.ainvoke.await_args_list]
        self.assertNotEqual(prompts[0], prompts[1])
        self.assertIn("Какие у тебя инструменты?", prompts[1])
        self.assertNotIn("Internal plan", prompts[1])
        warning = logger.warning.call_args
        diagnostic = warning.args[0] % warning.args[1:]
        self.assertIn("content_type=str", diagnostic)
        self.assertIn("preview_truncated=True", diagnostic)

    async def test_generate_chat_title_with_llm_logs_empty_visible_response(self):
        llm = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content=[
            {"type": "reasoning", "text": "Внутренний план", "encrypted_content": "opaque"},
        ])))
        logger = Mock()
        with patch("ui.runtime_payloads.asyncio.sleep", new_callable=AsyncMock):
            title = await generate_chat_title_with_llm(llm, "Какие у тебя инструменты?", logger)
        self.assertIsNone(title)
        self.assertEqual(llm.ainvoke.await_count, 3)
        warning = logger.warning.call_args
        diagnostic = warning.args[0] % warning.args[1:]
        self.assertIn("block_types=['reasoning']", diagnostic)
        self.assertIn("visible_chars=0", diagnostic)
        self.assertNotIn("Внутренний план", diagnostic)
        self.assertNotIn("opaque", diagnostic)

    async def test_generate_chat_title_with_llm_passes_extra_kwargs(self):
        captured = {}

        class KwargsLLM:
            async def ainvoke(self, _prompt, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(content="Настройка Apache")

        title = await generate_chat_title_with_llm(
            KwargsLLM(),
            "Помоги настроить Apache",
            logger=None,
            extra_kwargs={"reasoning_effort": "low"},
        )
        self.assertEqual(title, "Настройка Apache")
        self.assertEqual(captured, {"reasoning_effort": "low"})

    async def test_generate_chat_title_with_llm_retries_until_valid_title(self):
        class FlakyLLM:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, _prompt):
                self.calls += 1
                if self.calls == 1:
                    raise ConnectionError("transient network error")
                if self.calls == 2:
                    return SimpleNamespace(content="This is a test title")
                return SimpleNamespace(content="Настройка Apache")

        llm = FlakyLLM()
        title = await generate_chat_title_with_llm(llm, "Помоги настроить Apache", logger=None)
        self.assertEqual(title, "Настройка Apache")
        self.assertEqual(llm.calls, 3)

    async def test_generate_chat_title_with_llm_returns_none_after_exhausted_retries(self):
        class BrokenLLM:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, _prompt):
                self.calls += 1
                raise TimeoutError("provider timeout")

        llm = BrokenLLM()
        title = await generate_chat_title_with_llm(llm, "Помоги настроить Apache", logger=None)
        self.assertIsNone(title)
        self.assertEqual(llm.calls, 3)

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
