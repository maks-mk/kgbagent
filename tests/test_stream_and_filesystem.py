import asyncio
import base64
import shutil
import unittest
from pathlib import Path
from uuid import uuid4
from unittest import mock

import httpx
from langchain_core.messages import AIMessage, AIMessageChunk, RemoveMessage, ToolMessage

from core.safety_policy import SafetyPolicy
from core.text_utils import (
    extract_cache_hit_tokens,
    format_compact_tokens,
    format_elapsed_seconds,
    format_tool_output,
    prepare_markdown_for_render,
    split_markdown_segments,
)
from ui.streaming import StreamProcessor
from ui.window_components.main_window import MainWindow
from tools import filesystem, local_shell
from tools.filesystem import FilesystemManager, _DOWNLOAD_HEADERS, _format_download_http_error


class StreamAndFilesystemTests(unittest.TestCase):
    class _FakeReader:
        def __init__(self, chunks: list[bytes]):
            self._chunks = list(chunks)

        async def read(self, _size: int) -> bytes:
            if self._chunks:
                return self._chunks.pop(0)
            return b""

    class _FakeProcess:
        def __init__(
            self,
            stdout_chunks: list[bytes],
            stderr_chunks: list[bytes],
            returncode: int = 0,
            wait_exception: Exception | None = None,
            pid: int | None = None,
        ):
            self.stdout = StreamAndFilesystemTests._FakeReader(stdout_chunks)
            self.stderr = StreamAndFilesystemTests._FakeReader(stderr_chunks)
            self.returncode = returncode
            self.wait_exception = wait_exception
            self.killed = False
            self.pid = pid

        async def wait(self) -> int:
            if self.wait_exception is not None:
                raise self.wait_exception
            return self.returncode

        def kill(self) -> None:
            self.killed = True

    def _workspace_tempdir(self) -> Path:
        path = Path.cwd() / ".tmp_tests" / uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_elapsed_seconds_uses_minutes_only_above_sixty_seconds(self):
        self.assertEqual(format_elapsed_seconds(60), "60s")
        self.assertEqual(format_elapsed_seconds(61), "1m 1s")
        self.assertEqual(format_elapsed_seconds(120), "2m 0s")

    def test_stream_status_elapsed_text_uses_minutes_after_rounding(self):
        processor = StreamProcessor()
        with mock.patch.object(StreamProcessor, "_elapsed_seconds", return_value=60.4):
            self.assertEqual(processor._status_elapsed_text(), "60s")
        with mock.patch.object(StreamProcessor, "_elapsed_seconds", return_value=60.6):
            self.assertEqual(processor._status_elapsed_text(), "1m 1s")

    def test_final_stats_use_minutes_above_sixty_seconds(self):
        processor = StreamProcessor()

        self.assertEqual(processor.tracker.render(60), "60.0s  ↓ 0  ↑ 0")
        self.assertEqual(processor.tracker.render(60.1), "1m 0s  ↓ 0  ↑ 0")
        self.assertEqual(processor.tracker.render(61), "1m 1s  ↓ 0  ↑ 0")
        self.assertEqual(processor.tracker.render(120), "2m 0s  ↓ 0  ↑ 0")

    def test_cache_hit_extractor_prefers_exact_paths_and_ignores_disallowed_usage(self):
        usage = {
            "usage": {
                "prompt_tokens_details": {"cached_tokens": 8192},
                "cache_read_input_tokens": 4096,
                "cache_creation_input_tokens": 12000,
                "prompt_eval_count": 16000,
            },
            "nested": {"cached_tokens": 2048},
        }

        self.assertEqual(extract_cache_hit_tokens(usage), 8192)
        self.assertEqual(
            extract_cache_hit_tokens(
                {
                    "usage": {"prompt_tokens_details": {"cached_tokens": 0}},
                    "nested": {"cached_tokens": 2048},
                }
            ),
            0,
        )
        self.assertIsNone(
            extract_cache_hit_tokens(
                {
                    "cache_write_tokens": 100,
                    "cache_creation_input_tokens": 200,
                    "prompt_cache_miss_tokens": 300,
                    "prompt_tokens": 400,
                    "input_tokens": 500,
                    "prompt_eval_count": 600,
                }
            )
        )

    def test_cache_hit_extractor_supports_known_provider_and_langchain_fields(self):
        cases = (
            ({"usage": {"input_tokens_details": {"cached_tokens": 101}}}, 101),
            ({"usage": {"cached_tokens": 102}}, 102),
            ({"usage": {"prompt_cache_hit_tokens": 103}}, 103),
            ({"usage": {"cache_read_input_tokens": 104}}, 104),
            ({"usageMetadata": {"cachedContentTokenCount": 105}}, 105),
            ({"usage": {"cached_prompt_text_tokens": 106}}, 106),
            ({"input_token_details": {"cache_read": 107}}, 107),
            (
                {
                    "input_tokens": 9000,
                    "input_token_details": {"priority_cache_read": 109, "priority": 8891},
                },
                109,
            ),
            ({"input_token_details": {"flex_cache_read": 110}}, 110),
            ({"provider": {"details": {"cached_tokens": 108}}}, 108),
        )

        for payload, expected in cases:
            with self.subTest(payload=payload):
                self.assertEqual(extract_cache_hit_tokens(payload), expected)

    def test_compact_token_format_uses_standard_suffixes(self):
        self.assertEqual(format_compact_tokens(0), "0")
        self.assertEqual(format_compact_tokens(999), "999")
        self.assertEqual(format_compact_tokens(15_360), "15.4K")
        self.assertEqual(format_compact_tokens(100_000), "100K")
        self.assertEqual(format_compact_tokens(1_250_000), "1.2M")

    def test_prepare_markdown_does_not_guess_code_blocks_from_plain_text(self):
        source = 'Пример (файл main.go):\npackage main\nimport "fmt"\nfunc main() {\n    fmt.Println("hi")\n}'
        rendered = prepare_markdown_for_render(source)
        self.assertNotIn("```", rendered)
        self.assertIn("package main", rendered)
        self.assertIn("fmt.Println", rendered)

    def test_format_tool_output_cli_summary_has_no_rich_markup(self):
        summary = format_tool_output(
            "cli_exec",
            "curl -I google.com\nHTTP/1.1 301 Moved Permanently\nlocation: https://www.google.com/",
            False,
        )
        self.assertIn("(+2 lines)", summary)
        self.assertNotIn("[dim]", summary)
        self.assertNotIn("[/]", summary)

    def test_format_tool_output_error_summary_has_no_rich_markup(self):
        summary = format_tool_output(
            "read_file",
            "ERROR[EXECUTION]: Unauthorized 401",
            True,
        )
        self.assertIn("ERROR[EXECUTION]: Unauthorized 401", summary)
        self.assertIn("Hint: Check your API keys in .env", summary)
        self.assertNotIn("[red]", summary)
        self.assertNotIn("[/]", summary)

    def test_stream_processor_formats_registered_mcp_tool_for_people(self):
        processor = StreamProcessor(
            tool_sources={"resolve_library_id": "mcp"},
            mcp_tool_servers={"resolve_library_id": "context7"},
        )

        payload = processor._build_tool_event_payload(
            "call-mcp",
            "resolve_library_id",
            {"libraryName": "PySide6"},
            phase="preparing",
        )

        self.assertEqual(payload["source_kind"], "mcp")
        self.assertEqual(payload["display"], "Context7: Resolve Library ID")
        self.assertEqual(payload["subtitle"], "PySide6")
        self.assertNotIn("libraryName=", payload["display"])

    def test_prepare_markdown_normalizes_simple_latex_symbols(self):
        source = (
            'При загрузке страницы звук не инициализируется $\\rightarrow$ браузер не ругается.\n'
            'При нажатии на «СТАРТ» $\\Rightarrow$ звук активируется.'
        )

        rendered = prepare_markdown_for_render(source)

        self.assertIn("→", rendered)
        self.assertIn("⇒", rendered)
        self.assertNotIn("$\\rightarrow$", rendered)
        self.assertNotIn("$\\Rightarrow$", rendered)

    def test_prepare_markdown_keeps_latex_symbols_literal_inside_code(self):
        source = 'Текст `$\\\\rightarrow$` и блок:\n```text\n$\\\\Rightarrow$\n```'

        rendered = prepare_markdown_for_render(source)

        self.assertIn("`$\\\\rightarrow$`", rendered)
        self.assertIn("```text\n$\\\\Rightarrow$\n```", rendered)

    def test_prepare_markdown_unescapes_common_markdown_markers_outside_code(self):
        source = r"""\# Title

\- \*\*bold\*\* item

`\*literal\*`"""

        rendered = prepare_markdown_for_render(source)

        self.assertIn("# Title", rendered)
        self.assertIn("- **bold** item", rendered)
        self.assertIn(r"`\*literal\*`", rendered)

    def test_prepare_markdown_keeps_open_fenced_code_literal_while_streaming(self):
        source = "```text\n$\\\\Rightarrow$\n"

        rendered = prepare_markdown_for_render(source)

        self.assertEqual(rendered.rstrip("\n"), source.rstrip("\n"))

    def test_split_markdown_segments_treats_unclosed_fence_as_code(self):
        segments = split_markdown_segments("До кода\n```python\nprint('hi')\n")

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].kind, "markdown")
        self.assertEqual(segments[0].text, "До кода\n")
        self.assertEqual(segments[1].kind, "code")
        self.assertEqual(segments[1].language, "python")
        self.assertEqual(segments[1].text.rstrip("\n"), "print('hi')")
        self.assertFalse(segments[1].closed)

    def test_split_markdown_segments_supports_tilde_fences(self):
        segments = split_markdown_segments("Intro\n~~~python\nprint('hi')\n~~~\nTail")

        self.assertEqual([segment.kind for segment in segments], ["markdown", "code", "markdown"])
        self.assertEqual(segments[1].language, "python")
        self.assertEqual(segments[1].text, "print('hi')\n")
        self.assertTrue(segments[1].closed)

    def test_split_markdown_segments_matches_closing_fence_marker_and_length(self):
        segments = split_markdown_segments("````python\n```\nprint('hi')\n````\n")

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].kind, "code")
        self.assertEqual(segments[0].text, "```\nprint('hi')\n")
        self.assertTrue(segments[0].closed)

    def test_split_markdown_segments_ignores_nested_looking_other_fence_type(self):
        segments = split_markdown_segments("~~~text\n```\nvalue\n~~~\n")

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].kind, "code")
        self.assertEqual(segments[0].text, "```\nvalue\n")
        self.assertTrue(segments[0].closed)

    def test_download_headers_request_binary_content(self):
        self.assertEqual(_DOWNLOAD_HEADERS["Accept"], "*/*")

    def test_download_http_errors_are_specific(self):
        forbidden = httpx.HTTPStatusError(
            "forbidden",
            request=httpx.Request("GET", "https://example.com/file.mp4"),
            response=httpx.Response(403, request=httpx.Request("GET", "https://example.com/file.mp4")),
        )
        not_found = httpx.HTTPStatusError(
            "not found",
            request=httpx.Request("GET", "https://example.com/file.mp4"),
            response=httpx.Response(404, request=httpx.Request("GET", "https://example.com/file.mp4")),
        )
        self.assertIn("ACCESS_DENIED", _format_download_http_error(forbidden))
        self.assertIn("browser-only access", _format_download_http_error(forbidden))
        self.assertIn("NOT_FOUND", _format_download_http_error(not_found))
        self.assertIn("direct file", _format_download_http_error(not_found))

    def test_stream_processor_forwards_custom_status_event(self):
        events = []
        processor = StreamProcessor(events.append)

        async def _stream():
            yield {
                "type": "custom",
                "data": {
                    "type": "status_changed",
                    "label": "Reconnecting... 1/3",
                    "node": "agent",
                },
            }

        result = asyncio.run(processor.process_stream(_stream()))

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertFalse(result.failed)
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["label"], "Reconnecting... 1/3")
        self.assertEqual(statuses[-1]["node"], "agent")

    def test_stream_processor_restarts_assistant_section_on_api_key_rotation(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages(
            (AIMessage(content="Частичный ответ упавшего ключа."), {"langgraph_node": "agent"})
        )
        self.assertEqual(processor.full_text, "Частичный ответ упавшего ключа.")

        processor._handle_custom(
            {
                "type": "api_key_rotated",
                "error_kind": "rate_limit",
                "from_index": 0,
                "to_index": 1,
            }
        )

        notices = [event.payload for event in events if event.type == "summary_notice"]
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0]["kind"], "api_key_rotated")
        self.assertEqual(notices[0]["level"], "warning")
        self.assertIn("rate_limit", notices[0]["message"])

        boundaries = [event for event in events if event.type == "assistant_boundary"]
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(processor.full_text, "")
        self.assertEqual(processor._previous_assistant_section_text, "")

        # The next key regenerates the answer from scratch; its opening may
        # repeat the aborted partial text and must not be stripped as a replay.
        processor._handle_messages(
            (AIMessage(content="Частичный ответ упавшего ключа. Полный ответ нового ключа."), {"langgraph_node": "agent"})
        )
        self.assertEqual(
            processor.full_text,
            "Частичный ответ упавшего ключа. Полный ответ нового ключа.",
        )

    def test_stream_processor_ignores_malformed_api_key_rotation_payload(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages(
            (AIMessage(content="Частичный ответ."), {"langgraph_node": "agent"})
        )

        processor._handle_custom({"type": "api_key_rotated"})

        notices = [event.payload for event in events if event.type == "summary_notice"]
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0]["kind"], "api_key_rotated")
        self.assertEqual(processor.full_text, "")
        self.assertEqual(processor._previous_assistant_section_text, "")

    def test_stream_processor_tool_preview_does_not_claim_execution_started(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._remember_tool_call(
            {"id": "call-preview", "name": "read_file", "args": {"path": "a.txt"}}
        )
        processor._emit_tool_started(
            {"id": "call-preview", "name": "read_file", "args": {"path": "a.txt"}}
        )

        self.assertEqual([event for event in events if event.type == "status_changed"], [])
        self.assertEqual(len([event for event in events if event.type == "tool_started"]), 1)
        self.assertEqual(processor.active_node, "agent")

    def test_stream_processor_starts_tool_batch_before_results(self):
        events = []
        processor = StreamProcessor(events.append)
        tool_calls = [
            {"id": "call-a", "name": "read_file", "args": {"path": "a.txt"}},
            {"id": "call-b", "name": "read_file", "args": {"path": "b.txt"}},
        ]

        processor._handle_custom({"type": "tool_batch_started", "tool_calls": tool_calls})

        event_types = [event.type for event in events]
        statuses = [event.payload for event in events if event.type == "status_changed"]
        started = [event.payload for event in events if event.type == "tool_started"]
        self.assertEqual(event_types[0], "status_changed")
        self.assertEqual(statuses[-1]["node"], "tools")
        self.assertEqual(statuses[-1]["label"], "Running tools")
        self.assertEqual([payload["tool_id"] for payload in started], ["call-a", "call-b"])

    def test_stream_processor_keeps_running_tools_until_parallel_batch_finishes(self):
        events = []
        processor = StreamProcessor(events.append)
        tool_calls = [
            {"id": "call-a", "name": "read_file", "args": {"path": "a.txt"}},
            {"id": "call-b", "name": "read_file", "args": {"path": "b.txt"}},
        ]
        processor._handle_custom({"type": "tool_batch_started", "tool_calls": tool_calls})

        processor._handle_tool_result(
            ToolMessage(content="a", tool_call_id="call-a", name="read_file")
        )
        statuses_after_first = [
            event.payload for event in events if event.type == "status_changed"
        ]
        self.assertEqual(statuses_after_first[-1]["label"], "Running tools")

        processor._handle_tool_result(
            ToolMessage(content="b", tool_call_id="call-b", name="read_file")
        )
        statuses_after_second = [
            event.payload for event in events if event.type == "status_changed"
        ]
        self.assertEqual(statuses_after_second[-1]["label"], "Working...")

    def test_stream_processor_does_not_reemit_running_tools_for_late_tool_message(self):
        events = []
        processor = StreamProcessor(events.append)
        tool_call = {"id": "call-read", "name": "read_file", "args": {"path": "a.txt"}}
        result = ToolMessage(
            content="a",
            tool_call_id="call-read",
            name="read_file",
            additional_kwargs={"tool_args": {"path": "a.txt"}},
        )
        processor._handle_custom({"type": "tool_batch_started", "tool_calls": [tool_call]})
        processor._handle_custom(
            {
                "type": "tool_result",
                "message": {
                    "content": result.content,
                    "tool_call_id": result.tool_call_id,
                    "name": result.name,
                    "additional_kwargs": result.additional_kwargs,
                    "status": result.status,
                },
            }
        )

        processor._handle_messages((result, {"langgraph_node": "tools"}))

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertEqual([status["label"] for status in statuses], ["Running tools", "Working..."])
        self.assertEqual(processor.active_node, "agent")

    def test_stream_processor_forwards_custom_tool_result_once(self):
        events = []
        processor = StreamProcessor(events.append)
        processor.tool_buffer["call-live"] = {"name": "read_file", "args": {"path": "demo.txt"}}
        message_payload = {
            "content": "live result",
            "tool_call_id": "call-live",
            "name": "read_file",
            "additional_kwargs": {"tool_args": {"path": "demo.txt"}},
            "status": "success",
        }

        processor._handle_custom({"type": "tool_result", "message": message_payload})
        processor._handle_tool_result(ToolMessage(**message_payload))

        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["tool_id"], "call-live")
        self.assertEqual(finished[0]["content"], "live result")

    def test_stream_processor_emits_tool_error_and_diff_events(self):
        events = []
        processor = StreamProcessor(events.append)
        processor.tool_buffer["call-1"] = {"name": "edit_file", "args": {"path": "demo.txt"}}

        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-1",
                name="edit_file",
                content="Success: File edited.\n\nDiff:\n```diff\n-foo\n+bar\n```",
            )
        )
        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-2",
                name="read_file",
                content="ERROR[EXECUTION]: boom",
            )
        )

        event_types = [event.type for event in events]
        self.assertIn("tool_finished", event_types)
        self.assertIn("tool_diff", event_types)
        tool_finished_payloads = [event.payload for event in events if event.type == "tool_finished"]
        self.assertTrue(any(payload["name"] == "edit_file" and payload["diff"] for payload in tool_finished_payloads))
        self.assertTrue(any(payload["is_error"] and "boom" in payload["content"] for payload in tool_finished_payloads))

    def test_stream_processor_treats_free_port_result_as_success(self):
        events = []
        processor = StreamProcessor(events.append)
        processor.tool_buffer["call-port-check"] = {"name": "find_process_by_port", "args": {"port": 8000}}

        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-port-check",
                name="find_process_by_port",
                content="No process found listening on port 8000.",
            )
        )

        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertFalse(finished[0]["is_error"])
        self.assertIn("No process found listening on port 8000.", finished[0]["content"])

    def test_stream_processor_treats_successful_read_file_error_like_contents_as_success(self):
        events = []
        processor = StreamProcessor(events.append)
        processor.tool_buffer["call-read-log"] = {"name": "read_file", "args": {"path": "server.log"}}

        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-read-log",
                name="read_file",
                content="Error: connection reset by peer\nTraceback follows below as part of the file",
                status="success",
            )
        )

        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertFalse(finished[0]["is_error"])
        self.assertEqual(finished[0]["display"], "Reading file")

    def test_stream_processor_treats_structured_error_without_status_as_error(self):
        events = []
        processor = StreamProcessor(events.append)
        processor.tool_buffer["call-legacy-error"] = {"name": "edit_file", "args": {"path": "demo.txt"}}

        legacy_message = ToolMessage(
            tool_call_id="call-legacy-error",
            name="edit_file",
            content="ERROR[VALIDATION]: Could not find a match for old_string.",
        )
        legacy_message = legacy_message.model_copy(update={"status": ""})

        processor._handle_tool_result(legacy_message)

        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertTrue(finished[0]["is_error"])

    def test_stream_processor_does_not_inject_preface_before_tool_when_model_text_is_empty(self):
        events = []
        processor = StreamProcessor(events.append)
        processor._remember_tool_call({"id": "call-preface", "name": "read_file", "args": {"path": "demo.txt"}})

        processor._emit_tool_started({"id": "call-preface", "name": "read_file", "args": {"path": "demo.txt"}})

        event_types = [event.type for event in events]
        self.assertIn("tool_started", event_types)
        self.assertNotIn("assistant_delta", event_types)

    def test_stream_processor_emits_summarization_notice(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_updates(
            {
                "summarize": {
                    "summary": "compressed summary",
                    "messages": [RemoveMessage(id="1"), RemoveMessage(id="2")],
                }
            }
        )

        notice_events = [event for event in events if event.type == "summary_notice"]
        self.assertEqual(len(notice_events), 1)
        self.assertIn("Context compressed automatically", notice_events[0].payload["message"])
        self.assertEqual(notice_events[0].payload["count"], 2)
        self.assertEqual(notice_events[0].payload["kind"], "auto_summary")

    def test_stream_processor_stats_use_numeric_input_fallback_when_usage_missing(self):
        events = []
        processor = StreamProcessor(events.append)

        async def _stream():
            yield {
                "type": "messages",
                "data": (AIMessage(content="Готово"), {"langgraph_node": "agent"}),
            }

        result = asyncio.run(processor.process_stream(_stream()))
        self.assertIsNotNone(result.stats)
        assert result.stats is not None
        self.assertIn("↓ 0", result.stats)
        self.assertNotIn("↓ ?", result.stats)

    def test_stream_processor_accepts_tuple_mode_message_chunks(self):
        events = []
        processor = StreamProcessor(events.append)

        async def _stream():
            yield (
                "messages",
                (AIMessage(content="Готово"), {"langgraph_node": "agent"}),
            )

        result = asyncio.run(processor.process_stream(_stream()))

        self.assertFalse(result.failed)
        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "Готово")

    def test_stream_processor_accepts_tuple_mode_update_chunks(self):
        events = []
        processor = StreamProcessor(events.append)

        async def _stream():
            yield (
                "updates",
                {
                    "__interrupt__": [
                        {
                            "kind": "user_choice",
                            "question": "Выберите режим:",
                            "options": ["A", "B"],
                        }
                    ]
                },
            )

        result = asyncio.run(processor.process_stream(_stream()))

        self.assertIsNotNone(result.interrupt)
        assert result.interrupt is not None
        self.assertEqual(result.interrupt["kind"], "user_choice")
        self.assertEqual(result.interrupt["question"], "Выберите режим:")

    def test_stream_processor_marks_stream_exception_as_failed(self):
        events = []
        processor = StreamProcessor(events.append)

        async def _stream():
            raise RuntimeError("graph exploded")
            yield

        result = asyncio.run(processor.process_stream(_stream()))

        self.assertTrue(result.failed)
        self.assertFalse(result.cancelled)
        self.assertIsNone(result.interrupt)
        self.assertEqual(result.error_message, "graph exploded")
        failed_events = [event for event in events if event.type == "run_failed"]
        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0].payload["message"], "graph exploded")

    def test_stream_processor_marks_active_tool_as_stream_interrupted_on_provider_error(self):
        events = []
        processor = StreamProcessor(events.append)

        async def _stream():
            yield {
                "type": "updates",
                "data": {
                    "agent": {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[{"id": "tc-stream", "name": "read_file", "args": {"path": "a.py"}}],
                            )
                        ]
                    }
                },
            }
            raise RuntimeError("upstream disconnected")

        result = asyncio.run(processor.process_stream(_stream()))

        self.assertTrue(result.failed)
        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertTrue(finished[0]["interrupted"])
        self.assertEqual(finished[0]["interruption_reason"], "stream_error")
        self.assertIn("ERROR[NETWORK]", finished[0]["content"])
        self.assertIn("not a user stop", finished[0]["content"])

    def test_stream_processor_reads_token_usage_from_update_payload(self):
        processor = StreamProcessor()
        processor._handle_updates({"agent": {"token_usage": {"prompt_tokens": 321, "completion_tokens": 8}}})
        stats = processor.tracker.render(0.1)
        self.assertIn("↓ 321", stats)
        self.assertIn("↑ 8", stats)

    def test_stream_processor_sums_input_tokens_across_model_steps(self):
        processor = StreamProcessor()
        processor._handle_updates({"agent": {"token_usage": {"input_tokens": 16, "output_tokens": 32}}})
        processor._handle_updates({"agent": {"token_usage": {"input_tokens": 59, "output_tokens": 71}}})
        processor._handle_updates({"agent": {"token_usage": {"input_tokens": 117, "output_tokens": 609}}})

        stats = processor.tracker.render(0.1)
        self.assertIn("↓ 192", stats)
        self.assertIn("↑ 712", stats)

    def test_stream_processor_deduplicates_usage_between_messages_and_updates(self):
        processor = StreamProcessor()
        message = AIMessage(
            content="",
            usage_metadata={"input_tokens": 117, "output_tokens": 609, "total_tokens": 726},
        )
        processor._handle_messages((message, {"langgraph_node": "agent"}))
        processor._handle_updates({"agent": {"token_usage": {"input_tokens": 117, "output_tokens": 609}}})

        stats = processor.tracker.render(0.1)
        self.assertIn("↓ 117", stats)
        self.assertIn("↑ 609", stats)

    def test_stream_processor_emits_only_new_cache_hit_tokens(self):
        events = []
        processor = StreamProcessor(events.append)
        message = AIMessage(
            content="",
            id="cache-message",
            usage_metadata={
                "input_tokens": 9000,
                "output_tokens": 10,
                "total_tokens": 9010,
                "input_token_details": {"cache_read": 8192},
            },
        )

        processor._handle_messages((message, {"langgraph_node": "agent"}))
        processor._handle_updates(
            {
                "agent": {
                    "messages": [message],
                    "token_usage": {
                        "input_tokens": 9000,
                        "output_tokens": 10,
                        "input_token_details": {"cache_read": 8192},
                    },
                }
            }
        )
        processor._handle_updates(
            {
                "agent": {
                    "messages": [AIMessage(content="", id="cache-message-2")],
                    "token_usage": {"prompt_tokens": 4500, "cache_read_input_tokens": 4096},
                }
            }
        )
        processor._handle_updates(
            {"agent": {"token_usage": {"prompt_eval_count": 32000, "input_tokens": 32000}}}
        )

        self.assertEqual([event.payload["tokens"] for event in events if event.type == "cache_hit"], [8192, 4096])
        self.assertEqual(processor.tracker.total_cache_hit, 12_288)

    def test_stream_processor_does_not_double_count_usage_with_removal_messages(self):
        events = []
        processor = StreamProcessor(events.append)
        usage = {
            "input_tokens": 9000,
            "output_tokens": 10,
            "total_tokens": 9010,
            "input_token_details": {"cache_read": 8192},
        }
        message = AIMessage(content="done", id="cache-message-removals", usage_metadata=usage)

        processor._handle_messages((message, {"langgraph_node": "agent"}))
        processor._handle_updates(
            {
                "agent": {
                    "messages": [RemoveMessage(id="internal-retry-prompt"), message],
                    "token_usage": usage,
                }
            }
        )

        self.assertEqual([event.payload["tokens"] for event in events if event.type == "cache_hit"], [8192])
        self.assertEqual(processor.tracker.total_cache_hit, 8192)
        self.assertEqual(processor.tracker.total_input, 9000)
        self.assertEqual(processor.tracker.total_output, 10)

    def test_stream_processor_sums_usage_reported_as_streamed_deltas(self):
        events = []
        processor = StreamProcessor(events.append)
        # langchain-google-genai converts Gemini's already-accumulated counts into
        # per-chunk deltas, so streamed usage has to be added up, not maximised.
        deltas = [
            {
                "input_tokens": 1000,
                "output_tokens": 10,
                "total_tokens": 1010,
                "input_token_details": {"cache_read": 800},
            },
            {
                "input_tokens": 0,
                "output_tokens": 30,
                "total_tokens": 30,
                "input_token_details": {"cache_read": 0},
            },
            {
                "input_tokens": 0,
                "output_tokens": 50,
                "total_tokens": 50,
                "input_token_details": {"cache_read": 0},
            },
        ]
        chunks = [
            AIMessageChunk(content=f"part{index} ", id="delta-usage-message", usage_metadata=usage)
            for index, usage in enumerate(deltas)
        ]
        aggregated = chunks[0] + chunks[1] + chunks[2]

        for chunk in chunks:
            processor._handle_messages((chunk, {"langgraph_node": "summarize"}))

        self.assertEqual(processor.tracker.total_input, aggregated.usage_metadata["input_tokens"])
        self.assertEqual(processor.tracker.total_output, aggregated.usage_metadata["output_tokens"])
        self.assertEqual(processor.tracker.total_cache_hit, 800)
        self.assertEqual([event.payload["tokens"] for event in events if event.type == "cache_hit"], [800])

    def test_stream_processor_does_not_double_count_delta_usage_with_node_update(self):
        events = []
        processor = StreamProcessor(events.append)
        cumulative = {
            "input_tokens": 1000,
            "output_tokens": 90,
            "total_tokens": 1090,
            "input_token_details": {"cache_read": 800},
        }
        deltas = [
            {
                "input_tokens": 1000,
                "output_tokens": 40,
                "total_tokens": 1040,
                "input_token_details": {"cache_read": 800},
            },
            {
                "input_tokens": 0,
                "output_tokens": 50,
                "total_tokens": 50,
                "input_token_details": {"cache_read": 0},
            },
        ]
        for index, usage in enumerate(deltas):
            processor._handle_messages(
                (
                    AIMessageChunk(content=f"part{index} ", id="delta-update-message", usage_metadata=usage),
                    {"langgraph_node": "agent"},
                )
            )
        processor._handle_updates(
            {
                "agent": {
                    "messages": [AIMessage(content="part0 part1 ", id="delta-update-message")],
                    "token_usage": cumulative,
                }
            }
        )

        self.assertEqual(processor.tracker.total_input, 1000)
        self.assertEqual(processor.tracker.total_output, 90)
        self.assertEqual(processor.tracker.total_cache_hit, 800)
        self.assertEqual([event.payload["tokens"] for event in events if event.type == "cache_hit"], [800])

    def test_stream_processor_applies_negative_input_token_delta_compensation(self):
        processor = StreamProcessor()
        # Gemini 2.0 lowers the cumulative prompt count in the final chunk, which
        # langchain-google-genai forwards as a negative input-token delta.
        deltas = [
            {"input_tokens": 1200, "output_tokens": 10, "total_tokens": 1210},
            {"input_tokens": -200, "output_tokens": 80, "total_tokens": 80},
        ]
        for index, usage in enumerate(deltas):
            processor._handle_messages(
                (
                    AIMessageChunk(content=f"part{index} ", id="compensated-message", usage_metadata=usage),
                    {"langgraph_node": "agent"},
                )
            )

        self.assertEqual(processor.tracker.total_input, 1000)
        self.assertEqual(processor.tracker.total_output, 90)

    def test_stream_processor_uses_total_tokens_when_output_is_zero(self):
        processor = StreamProcessor()
        processor._handle_updates({"agent": {"token_usage": {"total_tokens": 321, "output_tokens": 0}}})

        stats = processor.tracker.render(0.1)
        self.assertIn("↓ 321", stats)
        self.assertIn("↑ 0", stats)

    def test_stream_processor_does_not_estimate_output_tokens_from_text_length(self):
        processor = StreamProcessor()
        processor._handle_messages((AIMessage(content="x" * 120), {"langgraph_node": "agent"}))

        stats = processor.tracker.render(0.1)
        self.assertIn("↓ 0", stats)
        self.assertIn("↑ 0", stats)

    def test_stream_processor_accumulates_total_elapsed_from_previous_segments(self):
        events = []
        perf_values = iter([100.0, 100.0])

        def _fake_perf_counter():
            try:
                return next(perf_values)
            except StopIteration:
                return 102.7

        processor = None
        with mock.patch("ui.streaming.time.perf_counter", side_effect=_fake_perf_counter):
            processor = StreamProcessor(events.append, base_elapsed_seconds=87.0)

            async def _stream():
                yield {
                    "type": "messages",
                    "data": (AIMessage(content="Готово"), {"langgraph_node": "agent"}),
                }

            result = asyncio.run(processor.process_stream(_stream()))

        self.assertIsNotNone(result.stats)
        self.assertAlmostEqual(result.elapsed_seconds, 89.7, places=1)
        assert result.stats is not None
        self.assertIn("1m 30s", result.stats)

    def test_stream_processor_renders_recovery_handoff_messages(self):
        events = []
        processor = StreamProcessor(events.append)
        processor._handle_updates(
            {
                "recovery": {
                    "messages": [AIMessage(content="Автовыполнение остановлено. Уточните следующий шаг.")],
                }
            }
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertIn("Автовыполнение остановлено", deltas[0]["full_text"])

    def test_stream_processor_hides_internal_handoff_messages_from_ui_events(self):
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

    def test_stream_processor_marks_recovery_as_reviewing(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages((AIMessage(content=""), {"langgraph_node": "recovery"}))

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["node"], "recovery")
        self.assertEqual(statuses[-1]["label"], "Reviewing results")

    def test_stream_processor_reports_agent_status_as_working_for_plain_agent_output(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages((AIMessage(content="Готовлю ответ."), {"langgraph_node": "agent"}))

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["node"], "agent")
        self.assertEqual(statuses[-1]["label"], "Working...")

    def test_stream_processor_reports_agent_status_as_thinking_for_reasoning(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages((AIMessage(content=[{"type": "reasoning", "text": "Планирую."}]), {"langgraph_node": "agent"}))

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["node"], "agent")
        self.assertEqual(statuses[-1]["label"], "Thinking...")

    def test_stream_processor_keeps_gemini_thinking_status_for_following_text_chunks(self):
        events = []
        processor = StreamProcessor(events.append)
        metadata = {"langgraph_node": "agent"}

        processor._handle_messages(
            (AIMessageChunk(content=[{"type": "thinking", "thinking": "Планирую."}]), metadata)
        )
        processor._handle_messages((AIMessageChunk(content="Готово."), metadata))

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["label"], "Thinking...")
        self.assertNotIn("Working...", [status["label"] for status in statuses])

    def test_stream_processor_reports_update_agent_status_as_thinking_for_reasoning(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_updates(
            {"agent": {"messages": [AIMessage(content=[{"type": "reasoning", "text": "Планирую."}])]}}
        )

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["node"], "agent")
        self.assertEqual(statuses[-1]["label"], "Thinking...")

    def test_stream_processor_reports_agent_status_as_thinking_for_reasoning_content_kwarg(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages(
            (
                AIMessage(content="", additional_kwargs={"reasoning_content": "Сверяю варианты."}),
                {"langgraph_node": "agent"},
            )
        )

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["node"], "agent")
        self.assertEqual(statuses[-1]["label"], "Thinking...")

    def test_stream_processor_reports_openai_reasoning_delta_as_thinking(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages(
            (
                AIMessageChunk(content="", additional_kwargs={"reasoning_delta": "Checking constraints."}),
                {"langgraph_node": "agent"},
            )
        )

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertEqual(statuses[-1]["label"], "Thinking...")

    def test_stream_processor_suppresses_reasoning_only_content_duplicate(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages(
            (
                AIMessageChunk(content="Now", additional_kwargs={"reasoning_content": "Now"}),
                {"langgraph_node": "agent"},
            )
        )

        statuses = [event.payload for event in events if event.type == "status_changed"]
        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["node"], "agent")
        self.assertEqual(statuses[-1]["label"], "Thinking...")
        self.assertEqual(deltas, [])

    def test_stream_processor_reports_agent_status_as_thinking_for_reasoning_details(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages(
            (
                AIMessage(
                    content="",
                    additional_kwargs={
                        "reasoning_details": [
                            {"type": "reasoning.text", "text": "Сверяю варианты."},
                        ]
                    },
                ),
                {"langgraph_node": "agent"},
            )
        )

        statuses = [event.payload for event in events if event.type == "status_changed"]
        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["node"], "agent")
        self.assertEqual(statuses[-1]["label"], "Thinking...")
        self.assertEqual(deltas, [])

    def test_stream_processor_reports_agent_status_as_thinking_for_analysis_field(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages(
            (
                AIMessage(content=[{"analysis": "internal reasoning"}]),
                {"langgraph_node": "agent"},
            )
        )

        statuses = [event.payload for event in events if event.type == "status_changed"]
        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["label"], "Thinking...")
        self.assertEqual(deltas, [])

    def test_stream_processor_reports_agent_status_as_thinking_for_partial_inline_think_tag(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages((AIMessage(content="<think>Сверяю"), {"langgraph_node": "agent"}))

        statuses = [event.payload for event in events if event.type == "status_changed"]
        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["node"], "agent")
        self.assertEqual(statuses[-1]["label"], "Thinking...")
        self.assertEqual(deltas, [])

    def test_stream_processor_reports_agent_status_as_thinking_for_anthropic_thinking_block(self):
        """Anthropic extended thinking emits content blocks with type=thinking
        and a thinking/signature pair. The UI status must switch to Thinking."""
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages(
            (
                AIMessage(
                    content=[
                        {"type": "thinking", "thinking": "Размышляю над задачей.", "signature": "sig_abc"},
                        {"type": "text", "text": "Ответ"},
                    ]
                ),
                {"langgraph_node": "agent"},
            )
        )

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["node"], "agent")
        self.assertEqual(statuses[-1]["label"], "Thinking...")

    def test_stream_processor_reports_agent_status_as_thinking_for_anthropic_redacted_thinking_block(self):
        """Anthropic redacted_thinking blocks (display=omitted) must also trigger
        the Thinking status in the UI."""
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages(
            (
                AIMessage(
                    content=[
                        {"type": "redacted_thinking", "data": "encrypted_blob"},
                        {"type": "text", "text": "Ответ"},
                    ]
                ),
                {"langgraph_node": "agent"},
            )
        )

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["node"], "agent")
        self.assertEqual(statuses[-1]["label"], "Thinking...")

    def test_stream_processor_suppresses_anthropic_redacted_thinking_as_visible_text(self):
        """redacted_thinking blocks must not leak as visible assistant text —
        only the text block content should appear."""
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages(
            (
                AIMessage(
                    content=[
                        {"type": "redacted_thinking", "data": "encrypted_blob"},
                        {"type": "text", "text": "Видимый ответ"},
                    ]
                ),
                {"langgraph_node": "agent"},
            )
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        visible_text = "".join(delta.get("text", "") for delta in deltas)
        self.assertNotIn("encrypted_blob", visible_text)
        self.assertIn("Видимый ответ", visible_text)

    def test_stream_processor_skips_unknown_node_status_instead_of_fallback_thinking(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages((AIMessage(content=""), {"langgraph_node": "unknown_node"}))

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertEqual(statuses, [])

    def test_stream_processor_does_not_parse_choice_requests_from_plain_text(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(
                content=(
                    "Нужно выбрать один из вариантов.\n\n"
                    "Как продолжаем?\n"
                    "- direct_api: тестируем только API\n"
                    "- keep_mcp: оставляем MCP\n"
                )
            )
        )

        choice_events = [event for event in events if event.type == "user_choice_requested"]
        self.assertEqual(len(choice_events), 0)
        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertIn("Как продолжаем?", deltas[0]["full_text"])

    def test_stream_processor_does_not_parse_inline_think_tags(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(content="<think>Сначала проверю входные данные.</think>Готово")
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "Готово")
        self.assertNotIn("thought_markdown", deltas[0])
        self.assertNotIn("has_thought", deltas[0])

    def test_stream_processor_ignores_structured_reasoning_from_provider_content(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(
                content=[
                    {"type": "reasoning", "text": "Сначала сверю формат входа."},
                    {"type": "text", "text": "Готово"},
                ]
            )
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "Готово")
        self.assertNotIn("thought_markdown", deltas[0])
        self.assertNotIn("has_thought", deltas[0])

    def test_stream_processor_does_not_parse_escaped_think_tags(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(content="&lt;think&gt;Проверяю ограничения.&lt;/think&gt;Итог готов")
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertIn("&lt;think&gt;Проверяю ограничения.&lt;/think&gt;Итог готов", deltas[0]["full_text"])
        self.assertNotIn("thought_markdown", deltas[0])
        self.assertNotIn("has_thought", deltas[0])

    def test_stream_processor_suppresses_chunked_textual_tool_call_marker(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessageChunk(content="Читаю файл.call:read_file{"),
            source="messages",
        )
        processor._handle_agent_message(
            AIMessageChunk(content='path:<|"|>index.html<|"|>}<tool_call|>'),
            source="messages",
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "Читаю файл.")
        self.assertNotIn("call:read_file", processor.full_text)
        self.assertNotIn("<tool_call|>", processor.full_text)

    def test_stream_processor_hides_request_user_input_tool_events(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(
                content="",
                tool_calls=[{"id": "choice-1", "name": "request_user_input", "args": {"question": "Continue?"}}],
            ),
            source="updates_agent",
        )
        processor._handle_tool_result(
            ToolMessage(content="implement", name="request_user_input", tool_call_id="choice-1")
        )

        self.assertEqual([event.type for event in events if event.type.startswith("tool_")], [])

    def test_stream_processor_emits_cumulative_full_text_for_chunked_string_output(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(AIMessage(content="Понял"), source="messages")
        processor._handle_agent_message(AIMessage(content=", продолжаю"), source="messages")

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 2)
        self.assertEqual(deltas[0]["full_text"], "Понял")
        self.assertEqual(deltas[1]["full_text"], "Понял, продолжаю")

    def test_stream_processor_replaces_final_agent_update_without_duplication(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(AIMessageChunk(content="Понял"), source="messages")
        processor._handle_agent_message(AIMessageChunk(content=", продолжаю"), source="messages")
        processor._handle_updates(
            {
                "agent": {
                    "messages": [
                        AIMessage(content="Понял, продолжаю")
                    ]
                }
            }
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 2)
        self.assertEqual(deltas[-1]["full_text"], "Понял, продолжаю")
        self.assertEqual(processor.full_text, "Понял, продолжаю")

    def test_stream_processor_suppresses_near_duplicate_final_agent_update_after_message_stream(self):
        events = []
        processor = StreamProcessor(events.append)
        streamed = (
            "## Анализ проекта\n\n"
            "Проект в текущей директории — не полноценое приложение, а небольшой registry-модуль "
            "для OpenAI-compatible провайдеров.\n\n"
            "Найдено 3 файла: freemodel_home.html, provider_registry.json, provider_registry_guide.md.\n"
        )
        final_update = streamed.replace("полноценое", "полноценное")

        processor._handle_agent_message(AIMessageChunk(content=streamed), source="messages")
        processor._handle_updates({"agent": {"messages": [AIMessage(content=final_update)]}})

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], streamed.rstrip("\n"))
        self.assertEqual(processor.full_text, streamed)

    def test_stream_processor_defers_post_tool_text_until_tool_finishes(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(AIMessageChunk(content="Читаю файл."), source="messages")
        processor._emit_tool_started({"id": "call-read", "name": "read_file", "args": {"path": "index.html"}})
        processor._handle_agent_message(AIMessageChunk(content="Готово, вот анализ."), source="messages")

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "Читаю файл.")
        self.assertEqual(processor.full_text, "Читаю файл.Готово, вот анализ.")

        processor._handle_tool_result(
            ToolMessage(
                content="ok",
                name="read_file",
                tool_call_id="call-read",
            )
        )

        event_types = [event.type for event in events]
        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 2)
        self.assertEqual(deltas[-1]["full_text"], "Готово, вот анализ.")
        self.assertIn("assistant_boundary", event_types)
        self.assertLess(event_types.index("tool_finished"), event_types.index("assistant_boundary"))
        self.assertLess(event_types.index("assistant_boundary"), event_types.index("assistant_delta", 2))

    def test_stream_processor_keeps_tool_call_message_text_visible_before_result(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(
                content="Проверю файл.",
                tool_calls=[{"id": "call-read", "name": "read_file", "args": {"path": "index.html"}}],
            ),
            source="updates_agent",
        )

        event_types = [event.type for event in events]
        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertIn("tool_started", event_types)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "Проверю файл.")

    def test_stream_processor_suppresses_duplicate_preface_replay_while_tool_is_active(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(AIMessageChunk(content="Проверю файл."), source="messages")
        processor._handle_agent_message(
            AIMessage(
                content="Проверю файл.\n\nПроверю файл.",
                tool_calls=[{"id": "call-read", "name": "read_file", "args": {"path": "index.html"}}],
            ),
            source="updates_agent",
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "Проверю файл.")
        self.assertEqual(processor.full_text, "Проверю файл.")
        self.assertIn("call-read", processor.tool_start_times)

    def test_stream_processor_suppresses_updates_agent_tool_preface_after_message_stream(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(AIMessageChunk(content="Открою официальные страницы."), source="messages")
        processor._handle_agent_message(
            AIMessage(
                content="\nОткрою официальные страницы.\n",
                tool_calls=[{"id": "call-fetch", "name": "fetch_content", "args": {"urls": ["https://example.com"]}}],
            ),
            source="updates_agent",
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "Открою официальные страницы.")
        self.assertEqual(processor.full_text, "Открою официальные страницы.")
        self.assertIn("call-fetch", processor.tool_start_times)

    def test_stream_processor_suppresses_messages_tool_preface_after_message_stream(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(AIMessageChunk(content="Открою официальные страницы."), source="messages")
        processor._handle_agent_message(
            AIMessage(
                content="\nОткрою официальные страницы.\n",
                tool_calls=[{"id": "call-fetch", "name": "fetch_content", "args": {"urls": ["https://example.com"]}}],
            ),
            source="messages",
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "Открою официальные страницы.")
        self.assertEqual(processor.full_text, "Открою официальные страницы.")
        self.assertIn("call-fetch", processor.tool_start_times)

    def test_stream_processor_suppresses_last_paragraph_replay_while_tool_is_active(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(AIMessageChunk(content="Проверю информацию."), source="messages")
        processor._handle_tool_result(
            ToolMessage(content="ok", name="batch_web_search", tool_call_id="call-search")
        )
        processor._handle_agent_message(
            AIMessageChunk(content="Открою официальные страницы Zhipu AI."),
            source="messages",
        )
        processor._emit_tool_started(
            {"id": "call-fetch", "name": "fetch_content", "args": {"urls": ["https://example.com"]}}
        )
        processor._handle_agent_message(
            AIMessage(
                content="Открою официальные страницы Zhipu AI.",
                tool_calls=[{"id": "call-fetch", "name": "fetch_content", "args": {"urls": ["https://example.com"]}}],
            ),
            source="updates_agent",
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(deltas[-1]["full_text"].count("Открою официальные страницы Zhipu AI."), 1)
        self.assertEqual(processor.full_text.count("Открою официальные страницы Zhipu AI."), 1)

    def test_stream_processor_suppresses_messages_tool_preface_after_updates_agent_preface(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(
                content="Проверю файл.",
                tool_calls=[{"id": "call-read", "name": "read_file", "args": {"path": "index.html"}}],
            ),
            source="updates_agent",
        )
        processor._handle_agent_message(AIMessageChunk(content="Проверю "), source="messages")
        processor._handle_agent_message(AIMessageChunk(content="файл."), source="messages")

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "Проверю файл.")
        self.assertEqual(processor.full_text, "Проверю файл.")

    def test_stream_processor_suppresses_updates_agent_replay_when_messages_text_was_deferred(self):
        """Text streamed via messages while tools are active is deferred (not emitted,
        so _visible_full_text is stale). When updates_agent arrives with the same text
        plus tool_calls, it must NOT emit a duplicate assistant_delta."""
        events = []
        processor = StreamProcessor(events.append)

        # Simulate: tool already active (e.g. from a prior tool_call in the same turn)
        processor._emit_tool_started(
            {"id": "call-search", "name": "batch_web_search", "args": {"queries": ["test"]}}
        )
        # Text streamed via messages while tool is active → deferred, _visible_full_text stays ""
        processor._handle_agent_message(
            AIMessageChunk(content="Проверю информацию в источниках."),
            source="messages",
        )
        # The deferred text was NOT emitted yet (tool still active)
        deferred_deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deferred_deltas), 0)
        # Now updates_agent arrives with the SAME text + tool_calls
        processor._handle_agent_message(
            AIMessage(
                content="Проверю информацию в источниках.",
                tool_calls=[{"id": "call-fetch", "name": "fetch_content", "args": {"urls": ["https://example.com"]}}],
            ),
            source="updates_agent",
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        # Should emit exactly ONE delta (the text is new to the visible UI, not a replay)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "Проверю информацию в источниках.")
        self.assertEqual(processor.full_text, "Проверю информацию в источниках.")

    def test_stream_processor_merges_post_tool_answer_without_repeating_last_preface(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(AIMessageChunk(content="Проверю источники."), source="messages")
        processor._emit_tool_started({"id": "call-search", "name": "batch_web_search", "args": {"queries": ["gpt-5.4"]}})
        processor._handle_tool_result(
            ToolMessage(content="ok", name="batch_web_search", tool_call_id="call-search")
        )
        processor._handle_agent_message(AIMessageChunk(content="Открою официальные страницы."), source="messages")
        processor._emit_tool_started(
            {"id": "call-fetch", "name": "fetch_content", "args": {"urls": ["https://openai.com"]}}
        )
        processor._handle_tool_result(
            ToolMessage(content="ok", name="fetch_content", tool_call_id="call-fetch")
        )
        processor._handle_agent_message(
            AIMessage(content="Открою официальные страницы.\n\nВот полная информация."),
            source="updates_agent",
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(deltas[-1]["full_text"], "Вот полная информация.")
        self.assertNotIn("Открою официальные страницы.", deltas[-1]["full_text"])

    def test_stream_processor_suppresses_similar_replayed_preface_before_more_text(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessageChunk(content="Посмотрю конфигурацию лимитов и логику self-corection."),
            source="messages",
        )
        processor._emit_tool_started(
            {"id": "call-search", "name": "list_directory", "args": {"path": "core"}}
        )
        processor._handle_tool_result(
            ToolMessage(content="ok", name="list_directory", tool_call_id="call-search")
        )
        processor._handle_agent_message(
            AIMessageChunk(
                content=(
                    "Посмотрю конфигурацию лимитов и логику self-correction. "
                    "Нашёл настройки в `core/config.py`."
                )
            ),
            source="messages",
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(deltas[-1]["full_text"], "Нашёл настройки в `core/config.py`.")
        self.assertNotIn("Посмотрю конфигурацию лимитов", deltas[-1]["full_text"])

    def test_stream_processor_ignores_reasoning_only_update_without_text_chunk(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(content=[{"type": "reasoning", "text": "Планирую следующие шаги."}])
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 0)

    def test_stream_processor_ignores_gemini_thought_flagged_text_part(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(
                content=[
                    {"type": "text", "text": "Сначала проверю структуру проекта.", "thought": True},
                    {"type": "text", "text": "Готово."},
                ]
            )
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(statuses[-1]["label"], "Thinking...")
        self.assertEqual(deltas[0]["full_text"], "Готово.")
        self.assertNotIn("thought_markdown", deltas[0])
        self.assertNotIn("has_thought", deltas[0])

    def test_stream_processor_reports_anthropic_text_only_as_working(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages(
            (AIMessage(content=[{"type": "text", "text": "Ответ без thinking-блока."}]), {"langgraph_node": "agent"})
        )

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertEqual(statuses[-1]["label"], "Working...")

    def test_stream_processor_ignores_openai_responses_reasoning_summary(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(
                content=[
                    {
                        "type": "reasoning",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "Сначала оценю ограничения задачи.",
                            }
                        ],
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Готово.",
                            }
                        ],
                    },
                ]
            )
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "Готово.")
        self.assertNotIn("thought_markdown", deltas[0])
        self.assertNotIn("has_thought", deltas[0])

    def test_stream_processor_ignores_reasoning_details_text(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(
                content=[
                    {"type": "reasoning.text", "text": "Скрытое рассуждение."},
                    {"type": "text", "text": "Готово."},
                ]
            )
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "Готово.")
        self.assertNotIn("Скрытое", deltas[0]["full_text"])

    def test_stream_processor_ignores_langchain_content_blocks_reasoning(self):
        events = []
        processor = StreamProcessor(events.append)
        message = AIMessage(
            content="",
            additional_kwargs={
                "content_blocks": [
                    {"type": "reasoning", "reasoning": "Сверяю план действий."},
                    {"type": "text", "text": "План готов."},
                ]
            },
        )

        processor._handle_agent_message(message)

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], "План готов.")
        self.assertNotIn("thought_markdown", deltas[0])
        self.assertNotIn("has_thought", deltas[0])

    def test_stream_processor_returns_user_choice_interrupt_from_updates(self):
        events = []
        processor = StreamProcessor(events.append)

        async def _stream():
            yield {
                "type": "updates",
                "data": {
                    "__interrupt__": [
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
                    ]
                },
            }

        result = asyncio.run(processor.process_stream(_stream()))
        self.assertIsNotNone(result.interrupt)
        assert result.interrupt is not None
        self.assertEqual(result.interrupt["kind"], "user_choice")
        self.assertEqual(result.interrupt["question"], "Введите ключ API или выберите другой вариант:")
        self.assertEqual(
            result.interrupt["options"],
            [
                "Ввести ключ API",
                "Пропустить проверку и вернуть скрипт",
                "Завершить проверку",
            ],
        )

    def test_stream_processor_uses_bounded_memory_caps(self):
        processor = StreamProcessor(
            emit_event=None,
            text_max_chars=24,
            events_max=3,
            tool_buffer_max=2,
        )

        processor._handle_agent_message(AIMessage(content="0123456789" * 6))
        self.assertLessEqual(len(processor.full_text), 24)
        self.assertLessEqual(len(processor.clean_full), 24)

        for idx in range(6):
            processor._emit(f"evt_{idx}", {"i": idx})
        self.assertEqual(len(processor.events), 3)
        self.assertEqual(processor.events[-1].type, "evt_5")

        for idx in range(4):
            processor._remember_tool_call(
                {"id": f"tool-{idx}", "name": "read_file", "args": {"path": f"{idx}.txt"}}
            )
        self.assertLessEqual(len(processor.tool_buffer), 2)
        self.assertLessEqual(len(processor.tool_start_times), 2)

    def test_stream_processor_emit_trims_oldest_events_directly(self):
        processor = StreamProcessor(events_max=2)
        processor._emit("evt_1", {"i": 1})
        processor._emit("evt_2", {"i": 2})
        processor._emit("evt_3", {"i": 3})

        self.assertEqual([event.type for event in processor.events], ["evt_2", "evt_3"])

    def test_stream_processor_merges_tool_args_and_finishes_with_canonical_payload(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._remember_tool_call({"id": "call-merge", "name": "edit_file", "args": {}})
        processor._emit_tool_started({"id": "call-merge", "name": "edit_file", "args": {}})
        processor._remember_tool_call(
            {
                "id": "call-merge",
                "name": "edit_file",
                "args": {"path": "demo.txt", "old_string": "old", "new_string": "new"},
            }
        )
        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-merge",
                name="edit_file",
                content="Success: File edited.",
            )
        )

        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertEqual(
            finished[0]["args"],
            {"path": "demo.txt", "old_string": "old", "new_string": "new"},
        )
        self.assertEqual(finished[0]["display"], "Editing file")
        self.assertIn("demo.txt", finished[0]["subtitle"])
        self.assertIn("demo.txt", finished[0]["raw_display"])

    def test_stream_processor_safe_delete_payloads_use_filesystem_labels(self):
        processor = StreamProcessor()
        cases = (
            ("safe_delete_file", "obsolete.txt", "Preparing file deletion"),
            ("safe_delete_directory", "old-cache", "Preparing directory deletion"),
        )

        for name, path, expected_display in cases:
            with self.subTest(name=name):
                payload = processor._build_tool_event_payload(
                    f"call-{name}",
                    name,
                    {"path": path},
                    phase="preparing",
                )

                self.assertEqual(payload["display"], expected_display)
                self.assertEqual(payload["subtitle"], path)
                self.assertEqual(payload["raw_display"], f"{name}({path})")
                self.assertEqual(payload["args_state"], "complete")

    def test_stream_processor_emits_tool_started_refresh_when_args_arrive_late(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._remember_tool_call({"id": "call-refresh", "name": "edit_file", "args": {}})
        processor._emit_tool_started({"id": "call-refresh", "name": "edit_file", "args": {}})
        processor._remember_tool_call(
            {
                "id": "call-refresh",
                "name": "edit_file",
                "args": {"path": "late.txt", "old_string": "a", "new_string": "b"},
            }
        )

        started = [event.payload for event in events if event.type == "tool_started"]
        self.assertGreaterEqual(len(started), 2)
        self.assertEqual(started[-1]["args"]["path"], "late.txt")
        self.assertTrue(started[-1].get("refresh"))

    def test_stream_processor_preview_payload_avoids_empty_signature_flash(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._remember_tool_call({"id": "call-write", "name": "write_file", "args": {}})
        processor._emit_tool_started({"id": "call-write", "name": "write_file", "args": {}})

        started = [event.payload for event in events if event.type == "tool_started"]
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0]["display_state"], "preview")
        self.assertEqual(started[0]["args_state"], "pending")
        self.assertEqual(started[0]["display"], "Preparing file write")
        self.assertEqual(started[0]["subtitle"], "Waiting for arguments…")
        self.assertEqual(started[0]["raw_display"], "write_file")

    def test_stream_processor_tool_display_flattens_multiline_command(self):
        events = []
        processor = StreamProcessor(events.append)
        multiline_command = "python - <<'PY'\nimport sys\nprint(sys.version)\nPY"
        processor._remember_tool_call(
            {
                "id": "call-cmd",
                "name": "cli_exec",
                "args": {"command": multiline_command},
            }
        )

        processor._emit_tool_started(
            {
                "id": "call-cmd",
                "name": "cli_exec",
                "args": {"command": multiline_command},
            }
        )

        started = [event.payload for event in events if event.type == "tool_started"]
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0]["display"], "Preparing command")
        self.assertNotIn("\n", started[0]["subtitle"])
        self.assertIn("cli_exec", started[0]["raw_display"])

    def test_stream_processor_finish_before_start_keeps_args_when_buffer_has_tool_call(self):
        events = []
        processor = StreamProcessor(events.append)
        processor.tool_buffer["call-late"] = {
            "name": "read_file",
            "args": {"path": "service.log", "offset": 0, "limit": 20},
        }

        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-late",
                name="read_file",
                content="Last 20 line(s)...",
            )
        )

        started = [event.payload for event in events if event.type == "tool_started"]
        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0]["args"], {"path": "service.log", "offset": 0, "limit": 20})
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["args"], {"path": "service.log", "offset": 0, "limit": 20})

    def test_stream_processor_recovers_args_from_tool_message_metadata(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-meta",
                name="edit_file",
                content="Success: File edited.",
                additional_kwargs={
                    "tool_args": {
                        "path": "parse_yandex_forecast_fixed.py",
                        "old_string": "foo",
                        "new_string": "bar",
                    }
                },
            )
        )

        missing = [event for event in events if event.type == "tool_args_missing"]
        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(missing, [])
        self.assertEqual(len(finished), 1)
        self.assertEqual(
            finished[0]["args"],
            {
                "path": "parse_yandex_forecast_fixed.py",
                "old_string": "foo",
                "new_string": "bar",
            },
        )
        self.assertEqual(finished[0]["display"], "Editing file")
        self.assertIn("parse_yandex_forecast_fixed.py", finished[0]["subtitle"])
        self.assertIn("parse_yandex_forecast_fixed.py", finished[0]["raw_display"])

    def test_stream_processor_prefers_tool_execution_duration_from_metadata(self):
        events = []
        processor = StreamProcessor(events.append)
        processor.tool_buffer["call-late-duration"] = {
            "name": "cli_exec",
            "args": {"command": "ping -n 4 google.com"},
        }

        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-late-duration",
                name="cli_exec",
                content="done",
                additional_kwargs={
                    "tool_args": {"command": "ping -n 4 google.com"},
                    "tool_duration_seconds": 1.75,
                },
            )
        )

        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["duration"], 1.75)

    def test_stream_processor_deduplicates_duplicate_tool_results(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._remember_tool_call({"id": "call-dir", "name": "list_directory", "args": {"path": "."}})
        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-dir",
                name="list_directory",
                content="Directory '.':\n[FILE] demo.py",
            )
        )
        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-dir",
                name="list_directory",
                content="Directory '.':\n[FILE] demo.py",
            )
        )

        finished = [event.payload for event in events if event.type == "tool_finished"]
        missing = [event for event in events if event.type == "tool_args_missing"]
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["args"], {"path": "."})
        self.assertEqual(missing, [])

    def test_stream_processor_accumulates_streamed_tool_call_chunks(self):
        events = []
        processor = StreamProcessor(
            events.append,
        )

        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "read_file", "args": '{"pa', "id": "call-stream", "index": 0}
                ],
            ),
            source="messages",
        )
        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": None, "args": 'th": "demo.txt"}', "id": None, "index": 0}
                ],
                chunk_position="last",
            ),
            source="messages",
        )

        started = [event.payload for event in events if event.type == "tool_started"]
        self.assertEqual(len(started), 2)
        self.assertEqual({payload["tool_id"] for payload in started}, {"call-stream"})
        self.assertEqual(started[0]["args"], {})
        self.assertFalse(started[0].get("refresh", False))
        self.assertEqual(started[-1]["args"], {"path": "demo.txt"})
        self.assertTrue(started[-1].get("refresh", False))

    def test_stream_processor_starts_tool_before_streamed_args_are_parseable(self):
        events = []
        processor = StreamProcessor(
            events.append,
        )

        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "read_file", "args": "", "id": "call-delayed", "index": 0}
                ],
            ),
            source="messages",
        )
        started = [event.payload for event in events if event.type == "tool_started"]
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0]["tool_id"], "call-delayed")
        self.assertEqual(started[0]["name"], "read_file")
        self.assertEqual(started[0]["args"], {})

        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": None, "args": '{"path": "demo.txt"}', "id": None, "index": 0}
                ],
                chunk_position="last",
            ),
            source="messages",
        )

        started = [event.payload for event in events if event.type == "tool_started"]
        self.assertEqual(len(started), 2)
        self.assertEqual(started[-1]["tool_id"], "call-delayed")
        self.assertEqual(started[-1]["name"], "read_file")
        self.assertEqual(started[-1]["args"], {"path": "demo.txt"})
        self.assertTrue(started[-1].get("refresh", False))

    def test_stream_processor_starts_tool_from_result_when_streamed_args_never_arrive(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "status_tool", "args": "", "id": "call-no-args", "index": 0}
                ],
                chunk_position="last",
            ),
            source="messages",
        )
        self.assertEqual([event for event in events if event.type == "tool_started"], [])

        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-no-args",
                name="status_tool",
                content="ok",
            )
        )

        event_types = [event.type for event in events]
        self.assertIn("tool_started", event_types)
        self.assertIn("tool_finished", event_types)
        self.assertLess(event_types.index("tool_started"), event_types.index("tool_finished"))
        started = [event.payload for event in events if event.type == "tool_started"]
        self.assertEqual(started[0]["tool_id"], "call-no-args")
        self.assertEqual(started[0]["name"], "status_tool")
        self.assertEqual(started[0]["args"], {})

    def test_stream_processor_aliases_late_tool_call_chunk_id_to_preview_card(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "read_file", "args": '{"pa', "id": "call-preview", "index": 0}
                ],
            ),
            source="messages",
        )
        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": None, "args": 'th": "demo.txt"}', "id": "call-final", "index": 0}
                ],
                chunk_position="last",
            ),
            source="messages",
        )
        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-final",
                name="read_file",
                content="demo contents",
                additional_kwargs={"tool_args": {"path": "demo.txt"}},
            )
        )

        started = [event.payload for event in events if event.type == "tool_started"]
        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual({payload["tool_id"] for payload in started}, {"call-preview"})
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["tool_id"], "call-preview")
        self.assertEqual(finished[0]["args"], {"path": "demo.txt"})

    def test_stream_processor_matches_idless_stream_preview_to_tool_result(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "read_file", "args": '{"path": "demo.txt"}', "id": None, "index": 0}
                ],
                chunk_position="last",
            ),
            source="messages",
        )
        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-real",
                name="read_file",
                content="demo contents",
                additional_kwargs={"tool_args": {"path": "demo.txt"}},
            )
        )

        started = [event.payload for event in events if event.type == "tool_started"]
        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len({payload["tool_id"] for payload in started}), 1)
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["tool_id"], started[0]["tool_id"])
        self.assertEqual(finished[0]["args"], {"path": "demo.txt"})

    def test_stream_processor_keeps_parallel_chunk_indexes_isolated_when_some_calls_are_not_yet_parseable(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "read_file", "args": " ", "id": "call-b", "index": 1}
                ],
            ),
            source="messages",
        )
        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "list_directory", "args": '{"path": "."}', "id": "call-a", "index": 0}
                ],
            ),
            source="messages",
        )
        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "read_file", "args": '{"path": "b.txt"}', "id": "call-b", "index": 1}
                ],
                chunk_position="last",
            ),
            source="messages",
        )

        started = [event.payload for event in events if event.type == "tool_started"]
        self.assertEqual(len(started), 2)
        started_by_id = {payload["tool_id"]: payload for payload in started}
        self.assertEqual(set(started_by_id), {"call-a", "call-b"})
        self.assertEqual(started_by_id["call-a"]["name"], "list_directory")
        self.assertEqual(started_by_id["call-a"]["args"], {"path": "."})
        self.assertEqual(started_by_id["call-b"]["name"], "read_file")
        self.assertEqual(started_by_id["call-b"]["args"], {"path": "b.txt"})

    def test_stream_processor_clears_completed_index_accumulator_before_index_reuse(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "list_directory", "args": '{"path": "."}', "id": "call-info", "index": 0}
                ],
            ),
            source="messages",
        )
        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-info",
                name="list_directory",
                content="ok",
                additional_kwargs={"tool_args": {"path": "."}},
            )
        )
        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "read_file", "args": '{"pa', "id": "call-read", "index": 0}
                ],
            ),
            source="messages",
        )
        processor._handle_agent_message(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": None, "args": 'th": "a.py"}', "id": None, "index": 0}
                ],
            ),
            source="messages",
        )

        started = [event.payload for event in events if event.type == "tool_started"]
        read_starts = [payload for payload in started if payload["tool_id"] == "call-read"]
        self.assertGreaterEqual(len(read_starts), 1)
        self.assertEqual({payload["name"] for payload in read_starts}, {"read_file"})
        self.assertEqual(read_starts[-1]["args"], {"path": "a.py"})

    def test_filesystem_delete_uses_virtual_mode_path_guard(self):
        tmp = self._workspace_tempdir()
        manager = FilesystemManager(root_dir=tmp, virtual_mode=True)
        result = manager.delete_file("..\\outside.txt")
        self.assertIn("ERROR[EXECUTION]", result)
        self.assertIn("ACCESS DENIED", result)

    def test_filesystem_delete_directory_requires_recursive_for_non_empty(self):
        tmp = self._workspace_tempdir()
        manager = FilesystemManager(root_dir=tmp, virtual_mode=True)
        folder = tmp / "folder"
        folder.mkdir()
        (folder / "child.txt").write_text("data", encoding="utf-8")
        result = manager.delete_directory("folder")
        self.assertIn("recursive=True", result)

    def test_read_file_repairs_trailing_comma_in_existing_path(self):
        tmp = self._workspace_tempdir()
        manager = FilesystemManager(root_dir=tmp, virtual_mode=True)
        file_path = tmp / "model_info.md"
        file_path.write_text("hello", encoding="utf-8")

        result = manager.read_file("model_info.md, ", show_line_numbers=False)

        self.assertEqual(result, "hello")

    def test_read_file_tool_defaults_to_plain_content_without_line_numbers(self):
        tmp = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))
        filesystem.set_working_directory(str(tmp))

        target = tmp / "demo.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")

        result = filesystem.read_file_tool.invoke({"path": "demo.txt"})

        self.assertTrue(result.startswith("alpha\nbeta"))
        self.assertNotIn("     1  alpha", result)

    def test_module_filesystem_switches_workspace_after_directory_change(self):
        first = self._workspace_tempdir()
        second = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))

        (first / "from_first.txt").write_text("first", encoding="utf-8")
        (second / "from_second.txt").write_text("second", encoding="utf-8")

        filesystem.set_working_directory(str(first))
        first_result = filesystem.list_directory_tool.invoke({"path": "."})
        self.assertIn("from_first.txt", first_result)
        self.assertNotIn("from_second.txt", first_result)

        filesystem.set_working_directory(str(second))
        second_result = filesystem.list_directory_tool.invoke({"path": "."})
        self.assertIn("from_second.txt", second_result)
        self.assertNotIn("from_first.txt", second_result)

    def test_edit_file_accepts_legacy_aliases_for_old_and_new(self):
        tmp = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))
        filesystem.set_working_directory(str(tmp))

        target = tmp / "demo.txt"
        target.write_text("hello old world", encoding="utf-8")

        result = filesystem.edit_file_tool.invoke(
            {
                "path": "demo.txt",
                "old_text": "old",
                "new_text": "new",
            }
        )

        self.assertIn("Success: File edited.", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "hello new world")

    def test_edit_file_accepts_file_path_alias(self):
        tmp = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))
        filesystem.set_working_directory(str(tmp))

        target = tmp / "alias.txt"
        target.write_text("before", encoding="utf-8")

        result = filesystem.edit_file_tool.invoke(
            {
                "file_path": "alias.txt",
                "old_string": "before",
                "new_string": "after",
            }
        )

        self.assertIn("Success: File edited.", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "after")

    def test_edit_file_strips_read_file_line_numbers_before_matching_code(self):
        tmp = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))
        filesystem.set_working_directory(str(tmp))

        target = tmp / "demo.py"
        target.write_text(
            "def main():\n"
            "    if True:\n"
            "        print('old')\n",
            encoding="utf-8",
        )

        result = filesystem.edit_file_tool.invoke(
            {
                "path": "demo.py",
                "old_string": (
                    "     1  def main():\n"
                    "     2      if True:\n"
                    "     3          print('old')"
                ),
                "new_string": (
                    "     1  def main():\n"
                    "     2      if True:\n"
                    "     3          print('new')"
                ),
            }
        )

        self.assertIn("Success: File edited.", result)
        self.assertIn("line-number prefixes were removed", result)
        self.assertEqual(
            target.read_text(encoding="utf-8"),
            "def main():\n"
            "    if True:\n"
            "        print('new')\n",
        )
        self.assertNotIn("     1  ", target.read_text(encoding="utf-8"))

    def test_write_file_accepts_file_path_alias(self):
        tmp = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))
        filesystem.set_working_directory(str(tmp))

        result = filesystem.write_file_tool.invoke(
            {
                "file_path": "nested/out.txt",
                "content": "hello",
            }
        )

        self.assertIn("Success: File 'nested/out.txt' saved", result)
        self.assertEqual((tmp / "nested" / "out.txt").read_text(encoding="utf-8"), "hello")

    def test_safe_delete_tools_accept_canonical_path_aliases(self):
        tmp = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))
        filesystem.set_working_directory(str(tmp))

        file_target = tmp / "delete-me.txt"
        file_target.write_text("bye", encoding="utf-8")
        dir_target = tmp / "delete-dir"
        dir_target.mkdir()

        file_result = asyncio.run(filesystem.safe_delete_file.ainvoke({"path": "delete-me.txt"}))
        dir_result = asyncio.run(filesystem.safe_delete_directory.ainvoke({"path": "delete-dir"}))

        self.assertIn("Success", file_result)
        self.assertIn("Success", dir_result)
        self.assertFalse(file_target.exists())
        self.assertFalse(dir_target.exists())

    def test_edit_file_missing_new_string_returns_friendly_validation_error(self):
        tmp = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))
        filesystem.set_working_directory(str(tmp))

        target = tmp / "demo.txt"
        target.write_text("hello old world", encoding="utf-8")

        result = filesystem.edit_file_tool.invoke(
            {
                "path": "demo.txt",
                "old_text": "old",
            }
        )

        self.assertIn("ERROR[VALIDATION]", result)
        self.assertIn("new_string", result)

    def test_write_file_missing_content_returns_validation_error(self):
        tmp = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))
        filesystem.set_working_directory(str(tmp))

        result = filesystem.write_file_tool.invoke({"path": "empty.txt"})

        self.assertIn("ERROR[VALIDATION]", result)
        self.assertIn("content", result)
        self.assertFalse((tmp / "empty.txt").exists())

    def test_write_file_whitespace_only_content_returns_validation_error(self):
        tmp = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))
        filesystem.set_working_directory(str(tmp))

        result = filesystem.write_file_tool.invoke({"path": "ws.txt", "content": "   \n  "})

        self.assertIn("ERROR[VALIDATION]", result)
        self.assertIn("content", result)

    def test_edit_file_path_sanitizer_strips_browser_user_agent_tail(self):
        tmp = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))
        filesystem.set_working_directory(str(tmp))

        target = tmp / "parse_yandex_forecast_fixed.py"
        target.write_text("x = 1\n", encoding="utf-8")
        noisy_path = "parse_yandex_forecast_fixed.py Mozilla/5.0 AppleWebKit/537.36 Safari/537.36"

        result = filesystem.edit_file_tool.invoke(
            {
                "path": noisy_path,
                "old_string": "x = 1",
                "new_string": "x = 2",
            }
        )

        self.assertIn("Success: File edited.", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "x = 2\n")

    def test_cli_exec_stream_emits_live_chunks_with_tool_id(self):
        process = self._FakeProcess(
            stdout_chunks=[b"line-1\n", b"line-2\n"],
            stderr_chunks=[b"warn-1\n"],
            returncode=0,
        )
        live_events: list[dict[str, str]] = []
        self.addCleanup(lambda: local_shell.set_cli_output_emitter(None))

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            local_shell.set_cli_output_emitter(live_events.append)
            with local_shell.cli_output_context("call-cli-1"):
                result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "demo"}))

        self.assertIn("line-1", result)
        self.assertIn("line-2", result)
        self.assertIn("[stderr]", result)
        self.assertTrue(live_events)
        self.assertTrue(all(item.get("tool_id") == "call-cli-1" for item in live_events))
        self.assertTrue(any(item.get("stream") == "stdout" for item in live_events))
        self.assertTrue(any(item.get("stream") == "stderr" for item in live_events))

    def test_cli_exec_uses_raw_limit_before_tool_executor(self):
        process = self._FakeProcess(
            stdout_chunks=[b"x" * 2000],
            stderr_chunks=[],
            returncode=0,
        )
        previous_policy = local_shell._SAFETY_POLICY
        self.addCleanup(lambda: local_shell.set_safety_policy(previous_policy))
        local_shell.set_safety_policy(
            SafetyPolicy(allow_shell=True, max_tool_output=500, max_raw_tool_output=1500)
        )

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "demo"}))

        self.assertGreater(len(result), 500)
        self.assertIn("[TRUNCATED from 2000 chars | source=shell-raw]", result)

    def test_cli_exec_uses_non_interactive_stdin_and_npm_env(self):
        process = self._FakeProcess(
            stdout_chunks=[b"ok\n"],
            stderr_chunks=[],
            returncode=0,
        )
        captured_kwargs: dict[str, object] = {}

        async def _fake_create_subprocess(*_args, **kwargs):
            captured_kwargs.update(kwargs)
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "npm --version"}))

        self.assertIn("ok", result)
        self.assertEqual(captured_kwargs.get("stdin"), asyncio.subprocess.DEVNULL)
        env = dict(captured_kwargs.get("env") or {})
        self.assertEqual(env.get("CI"), "1")
        self.assertEqual(env.get("npm_config_yes"), "true")

    def test_cli_exec_uses_default_timeout(self):
        process = self._FakeProcess(
            stdout_chunks=[b"ok\n"],
            stderr_chunks=[],
            returncode=0,
        )
        captured_timeouts: list[int | float | None] = []
        original_wait_for = asyncio.wait_for

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        async def _record_wait_for(awaitable, timeout):
            captured_timeouts.append(timeout)
            return await original_wait_for(awaitable, timeout)

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "wait_for", side_effect=_record_wait_for),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "demo"}))

        self.assertIn("ok", result)
        self.assertEqual(captured_timeouts, [local_shell.DEFAULT_TIMEOUT])

    def test_cli_exec_uses_requested_timeout_in_timeout_error(self):
        process = self._FakeProcess(
            stdout_chunks=[b"partial\n"],
            stderr_chunks=[],
            returncode=0,
        )
        captured_timeouts: list[int | float | None] = []

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        async def _fake_wait_for(awaitable, timeout):
            captured_timeouts.append(timeout)
            if timeout == 37:
                awaitable.close()
                raise asyncio.TimeoutError
            return await awaitable

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "wait_for", side_effect=_fake_wait_for),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "demo", "timeout": 37}))

        self.assertIn("ERROR[TIMEOUT]", result)
        self.assertIn("timed out after 37 seconds", result)
        self.assertIn("partial", result)
        self.assertEqual(captured_timeouts, [37, 3])
        self.assertTrue(process.killed)

    def test_cli_exec_detects_interactive_prompt_and_aborts(self):
        process = self._FakeProcess(
            stdout_chunks=[b"npx@10.2.2\n", b"OK TO PROCEED? (Y)\n"],
            stderr_chunks=[],
            returncode=0,
        )

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "npm install demo-package"}))

        self.assertIn("Interactive prompt detected", result)
        self.assertIn("OK TO PROCEED? (Y)", result)
        self.assertIn("npm/npx", result)
        self.assertTrue(process.killed)

    def test_cli_exec_stream_preserves_error_result_on_non_zero_exit(self):
        process = self._FakeProcess(
            stdout_chunks=[b"partial-out\n"],
            stderr_chunks=[b"fatal-err\n"],
            returncode=2,
        )
        live_events: list[dict[str, str]] = []
        self.addCleanup(lambda: local_shell.set_cli_output_emitter(None))

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            local_shell.set_cli_output_emitter(live_events.append)
            with local_shell.cli_output_context("call-cli-2"):
                result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "demo --fail"}))

        self.assertIn("Exit Code 2", result)
        self.assertIn("partial-out", result)
        self.assertIn("fatal-err", result)
        self.assertGreaterEqual(len(live_events), 2)

    def test_cli_exec_exit_code_neutral_command_not_marked_as_error(self):
        """Commands like vulture/grep/rg use exit code 1 as part of their
        normal protocol — the result should NOT be wrapped in ERROR[...]."""
        process = self._FakeProcess(
            stdout_chunks=[b"dead_code.py:10: unused variable 'x'\n"],
            stderr_chunks=[],
            returncode=1,
        )

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "vulture core/ tools/"}))

        self.assertIn("Exit Code: 1", result)
        self.assertIn("unused variable", result)
        self.assertNotIn("ERROR[", result)

    def test_cli_exec_exit_code_neutral_command_rg_no_matches(self):
        """rg returns exit code 1 when no matches found — not an error."""
        process = self._FakeProcess(
            stdout_chunks=[b""],
            stderr_chunks=[b""],
            returncode=1,
        )

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "rg 'nonexistent_pattern' ."}))

        self.assertIn("Exit Code: 1", result)
        self.assertNotIn("ERROR[", result)

    def test_cli_exec_non_neutral_command_still_marked_as_error(self):
        """Commands not in the neutral list should still be marked as error
        on non-zero exit code."""
        process = self._FakeProcess(
            stdout_chunks=[b"some output\n"],
            stderr_chunks=[b"oops\n"],
            returncode=1,
        )

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "python broken_script.py"}))

        self.assertIn("ERROR[EXECUTION]", result)
        self.assertIn("Exit Code 1", result)

    def test_cli_exec_rejects_foreground_service_commands(self):
        result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "npm exec -- npx http-server -p 8080"}))

        self.assertIn("Foreground service/server commands are not supported", result)
        self.assertIn("run_background_process", result)

    def test_cli_exec_decodes_utf8_and_legacy_windows_output(self):
        cases = (
            ("utf-8", "Привет мир"),
            ("cp1251", "Привет мир"),
        )
        for encoding, expected in cases:
            with self.subTest(encoding=encoding):
                with mock.patch.object(local_shell.os, "name", "nt"), mock.patch.object(
                    local_shell.locale, "getpreferredencoding", return_value="cp1251"
                ):
                    decoder = local_shell._CliOutputDecoder()
                    encoded = expected.encode(encoding)
                    split = max(1, len(encoded) // 2)
                    actual = decoder.decode(encoded[:split]) + decoder.decode(encoded[split:], final=True)
                self.assertEqual(actual, expected)

    def test_cli_exec_sets_utf8_environment_for_windows_commands(self):
        with mock.patch.object(local_shell.os, "name", "nt"):
            env = local_shell._prepare_shell_env("python -c \"print('Привет')\"")
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(env["PYTHONUTF8"], "1")

    def test_cli_exec_wraps_windows_powershell_with_utf8_settings(self):
        with mock.patch.object(local_shell.os, "name", "nt"):
            wrapped = local_shell._windows_powershell_command("Write-Output 'Привет'")
        self.assertIn("$OutputEncoding", wrapped)
        self.assertIn("[Console]::OutputEncoding", wrapped)
        self.assertIn("Write-Output 'Привет'", wrapped)

        process = self._FakeProcess(
            stdout_chunks=[b"ok\n"],
            stderr_chunks=[],
            returncode=0,
        )
        captured_argv: list[tuple[str, ...]] = []

        async def _fake_create_subprocess_exec(*args, **_kwargs):
            if args:
                captured_argv.append(tuple(str(part) for part in args))
            return process

        heredoc_command = "python - <<'PY'\nprint('hello')\nPY"
        with (
            mock.patch.object(local_shell.os, "name", "nt"),
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess_exec),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": heredoc_command}))

        self.assertIn("ok", result)
        self.assertTrue(captured_argv)
        self.assertEqual(captured_argv[0][1:3], ("-NoProfile", "-Command"))
        powershell_command = captured_argv[0][3]
        self.assertIn("FromBase64String", powershell_command)
        self.assertIn("| python -", powershell_command)

    def test_cli_exec_normalizes_common_python_heredoc_variants_on_windows(self):
        cases = (
            ("python - <<PY\nprint('$x')\nPY", "python -"),
            ("python3 - <<\"PY\"\r\nprint('ok')\r\nPY\r\n", "python3 -"),
            ("py -u - <<'PY'\nprint('ok')\nPY", "py -u -"),
            ("python - <<PY\n    print('ok')\n    PY", "python -"),
        )
        with mock.patch.object(local_shell.os, "name", "nt"):
            for command, expected_suffix in cases:
                with self.subTest(command=command):
                    normalized = local_shell._normalize_windows_python_heredoc(command)
                    self.assertIn("FromBase64String", normalized)
                    self.assertTrue(normalized.endswith(f"| {expected_suffix}"))
                    encoded = normalized.split("FromBase64String('")[1].split("')", 1)[0]
                    decoded = base64.b64decode(encoded).decode("utf-8")
                    self.assertEqual(decoded.strip(), "print('$x')" if "'$x'" in command else "print('ok')")

    def test_cli_exec_unwraps_nested_powershell_wrapper_on_windows(self):
        process = self._FakeProcess(
            stdout_chunks=[b"ok\n"],
            stderr_chunks=[],
            returncode=0,
        )
        captured_argv: list[tuple[str, ...]] = []

        async def _fake_create_subprocess_exec(*args, **_kwargs):
            if args:
                captured_argv.append(tuple(str(part) for part in args))
            return process

        nested = 'powershell -Command "try { $r = 1; Write-Output $r } catch { Write-Output $_ }"'
        with (
            mock.patch.object(local_shell.os, "name", "nt"),
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess_exec),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": nested}))

        self.assertIn("ok", result)
        self.assertTrue(captured_argv)
        self.assertEqual(captured_argv[0][1:3], ("-NoProfile", "-Command"))
        self.assertIn("try { $r = 1; Write-Output $r } catch { Write-Output $_ }", captured_argv[0][3])
        self.assertIn("[Console]::OutputEncoding", captured_argv[0][3])


    def test_cli_exec_rewrites_posix_null_device_on_windows(self):
        process = self._FakeProcess(
            stdout_chunks=[b"200\n"],
            stderr_chunks=[],
            returncode=0,
        )
        captured_argv: list[tuple[str, ...]] = []

        async def _fake_create_subprocess_exec(*args, **_kwargs):
            if args:
                captured_argv.append(tuple(str(part) for part in args))
            return process

        command = 'curl -s -o /dev/null -w "%{http_code}" http://localhost:8000'
        with (
            mock.patch.object(local_shell.os, "name", "nt"),
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess_exec),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": command}))

        self.assertIn("200", result)
        self.assertTrue(captured_argv)
        normalized_command = captured_argv[0][3]
        self.assertIn("-o NUL", normalized_command)
        self.assertNotIn("/dev/null", normalized_command)

    def test_cli_exec_hides_windows_console_for_powershell_process(self):
        process = self._FakeProcess(
            stdout_chunks=[b"ok\n"],
            stderr_chunks=[],
            returncode=0,
        )
        captured_kwargs: dict[str, object] = {}

        async def _fake_create_subprocess_exec(*_args, **kwargs):
            captured_kwargs.update(kwargs)
            return process

        with (
            mock.patch.object(local_shell.os, "name", "nt"),
            mock.patch.object(local_shell.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True),
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess_exec),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "python --version"}))

        self.assertIn("ok", result)
        self.assertEqual(captured_kwargs.get("creationflags"), 0x08000000)

    def test_terminate_process_tree_hides_windows_console_for_taskkill(self):
        process = self._FakeProcess(
            stdout_chunks=[],
            stderr_chunks=[],
            returncode=0,
            pid=4321,
        )
        killer = self._FakeProcess(
            stdout_chunks=[],
            stderr_chunks=[],
            returncode=0,
        )
        captured_argv: list[tuple[str, ...]] = []
        captured_kwargs: dict[str, object] = {}

        async def _fake_create_subprocess_exec(*args, **kwargs):
            captured_argv.append(tuple(str(part) for part in args))
            captured_kwargs.update(kwargs)
            return killer

        with (
            mock.patch.object(local_shell.os, "name", "nt"),
            mock.patch.object(local_shell.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True),
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess_exec),
        ):
            asyncio.run(local_shell._terminate_process_tree(process))

        self.assertEqual(captured_argv[0], ("taskkill", "/PID", "4321", "/T", "/F"))
        self.assertEqual(captured_kwargs.get("creationflags"), 0x08000000)

    def test_cli_exec_cancellation_terminates_running_process(self):
        process = self._FakeProcess(
            stdout_chunks=[],
            stderr_chunks=[],
            wait_exception=asyncio.CancelledError(),
        )

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(local_shell.cli_exec.ainvoke({"command": "Get-Process | Select-Object Id"}))

        self.assertTrue(process.killed)

    def test_stream_processor_emits_interrupted_tool_finish_on_cancel(self):
        events = []
        processor = StreamProcessor(events.append)

        async def _stream():
            yield {
                "type": "updates",
                "data": {
                    "agent": {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[{"id": "tc-cancel", "name": "cli_exec", "args": {"command": "echo 1"}}],
                            )
                        ]
                    }
                },
            }
            raise asyncio.CancelledError()

        result = asyncio.run(processor.process_stream(_stream()))

        self.assertTrue(result.cancelled)
        self.assertEqual(len(result.cancelled_tools), 1)
        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertTrue(finished[0]["interrupted"])
        self.assertIn("ERROR[CANCELLED]", finished[0]["content"])

    def test_stream_processor_preserves_structured_error_and_root_cause(self):
        processor = StreamProcessor()

        async def _stream():
            if False:
                yield None
            try:
                raise OSError("DNS lookup failed")
            except OSError as cause:
                error = RuntimeError("Connection error.")
                error.status_code = 503
                error.request_id = "req-123"
                error.llm_retry_exhausted = True
                raise error from cause

        result = asyncio.run(processor.process_stream(_stream()))

        self.assertTrue(result.failed)
        self.assertEqual(result.error_details["error_type"], "RuntimeError")
        self.assertEqual(result.error_details["http_status"], 503)
        self.assertEqual(result.error_details["request_error_id"], "req-123")
        self.assertEqual(result.error_details["root_cause_type"], "OSError")
        self.assertEqual(result.error_details["root_cause_message"], "DNS lookup failed")
        self.assertTrue(result.error_details["retry_exhausted"])

    def test_stream_processor_hides_repaired_stream_interruption_tool_message(self):
        events = []
        processor = StreamProcessor(events.append)

        message = ToolMessage(
            content=(
                "ERROR[NETWORK]: The provider or aggregator stream ended before this tool returned a result. "
                "This is usually a transient upstream disconnect, not a user stop. Please retry or continue."
            ),
            tool_call_id="tc-repaired",
            name="read_file",
            additional_kwargs={
                "tool_args": {"path": "a.py"},
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
        )

        processor._handle_tool_result(message)

        finished = [event.payload for event in events if event.type == "tool_finished"]
        notices = [event.payload for event in events if event.type == "summary_notice"]
        self.assertEqual(finished, [])
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0]["kind"], "agent_internal_notice")
        self.assertIn("History was repaired automatically", notices[0]["message"])

    def test_stream_processor_assistant_delta_sequence_increases_monotonically(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(AIMessageChunk(content="Первый "))
        processor._handle_agent_message(AIMessageChunk(content="ответ"))

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual([payload["sequence"] for payload in deltas], [1, 2])
        self.assertEqual(deltas[-1]["full_text"], "Первый ответ")

    def test_stream_processor_commits_visible_text_before_run_finished(self):
        events = []
        processor = StreamProcessor(events.append)

        async def _stream():
            yield ("messages", (AIMessageChunk(content="**Готово**"), {"langgraph_node": "agent"}))

        asyncio.run(processor.process_stream(_stream()))

        event_types = [event.type for event in events]
        self.assertIn("assistant_commit", event_types)
        self.assertLess(event_types.index("assistant_commit"), event_types.index("run_finished"))
        commit = next(event.payload for event in events if event.type == "assistant_commit")
        self.assertEqual(commit["full_text"], "**Готово**")

    def test_main_window_stale_delta_fallback_detects_shorter_prefix_replay(self):
        window = MainWindow.__new__(MainWindow)
        window._last_assistant_delta_text = "Проверяю файлы.\n\n## Анализ"

        self.assertTrue(window._is_stale_assistant_delta_without_sequence("Проверяю файлы."))
        self.assertFalse(window._is_stale_assistant_delta_without_sequence("Другая корректировка"))

    def test_stream_processor_suppresses_updates_agent_replay_after_messages_segment(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(AIMessageChunk(content="Проверяю файлы."), source="messages")
        processor._emit_tool_started({"id": "call-read", "name": "read_file", "args": {"path": "main.py"}})
        processor._handle_tool_result(ToolMessage(content="ok", tool_call_id="call-read", name="read_file"))
        processor._handle_agent_message(AIMessage(content="Проверяю файлы."), source="updates_agent")

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual([payload["full_text"] for payload in deltas], ["Проверяю файлы."])

    def test_stream_processor_appends_distinct_updates_agent_text_after_tool(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(AIMessageChunk(content="Проверяю файлы."), source="messages")
        processor._emit_tool_started({"id": "call-read", "name": "read_file", "args": {"path": "main.py"}})
        processor._handle_tool_result(ToolMessage(content="ok", tool_call_id="call-read", name="read_file"))
        processor._handle_agent_message(AIMessage(content="Готово."), source="updates_agent")

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(deltas[-1]["full_text"], "Готово.")

    def test_stream_processor_holds_replayed_post_tool_tokens_out_of_live_ui(self):
        events = []
        processor = StreamProcessor(events.append)
        previous = "Рабочая директория почти пустая. Создам автономный шаблон лендинга."
        processor._handle_agent_message(AIMessageChunk(content=previous), source="messages")
        processor._emit_tool_started({"id": "call-read", "name": "read_file", "args": {"path": "index.html"}})
        processor._handle_tool_result(ToolMessage(content="ok", tool_call_id="call-read", name="read_file"))

        processor._handle_agent_message(AIMessageChunk(content="Рабочая директория почти "), source="messages")
        processor._handle_agent_message(AIMessageChunk(content="пустая. Создам автономный "), source="messages")
        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual([payload["full_text"] for payload in deltas], [previous])

        processor._handle_agent_message(
            AIMessageChunk(content="шаблон лендинга.\n\nТеперь запишу index.html."),
            source="messages",
        )
        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 2)
        self.assertEqual(deltas[-1]["full_text"], "Теперь запишу index.html.")
        self.assertNotIn(previous, deltas[-1]["full_text"])

    def test_stream_processor_suppresses_exact_post_tool_replay_before_next_tool(self):
        events = []
        processor = StreamProcessor(events.append)
        previous = "Проверю файл перед изменением."
        processor._handle_agent_message(AIMessageChunk(content=previous), source="messages")
        processor._emit_tool_started({"id": "call-read", "name": "read_file", "args": {"path": "index.html"}})
        processor._handle_tool_result(ToolMessage(content="ok", tool_call_id="call-read", name="read_file"))
        processor._handle_agent_message(
            AIMessage(
                content=previous,
                tool_calls=[{"id": "call-write", "name": "write_file", "args": {"path": "index.html"}}],
            ),
            source="messages",
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual([payload["full_text"] for payload in deltas], [previous])
        self.assertIn("call-write", processor.tool_start_times)

    def test_stream_processor_ignores_invisible_chunk_between_tool_calls(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(AIMessageChunk(content="Проверю файлы."), source="messages")
        processor._emit_tool_started({"id": "call-first", "name": "read_file", "args": {"path": "a.txt"}})
        processor._handle_tool_result(ToolMessage(content="ok", tool_call_id="call-first", name="read_file"))
        processor._handle_agent_message(AIMessageChunk(content="\u200b\ufeff"), source="messages")
        processor._emit_tool_started({"id": "call-second", "name": "read_file", "args": {"path": "b.txt"}})

        event_types = [event.type for event in events]
        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual([payload["full_text"] for payload in deltas], ["Проверю файлы."])
        self.assertEqual(event_types.count("assistant_boundary"), 0)
        self.assertEqual(event_types.count("tool_started"), 2)

    def test_stream_processor_starts_new_assistant_section_after_tool_result(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(AIMessageChunk(content="Проверю файл."), source="messages")
        processor._emit_tool_started({"id": "call-read", "name": "read_file", "args": {"path": "index.html"}})
        processor._handle_tool_result(ToolMessage(content="ok", tool_call_id="call-read", name="read_file"))
        processor._handle_agent_message(AIMessageChunk(content="Файл пустой."), source="messages")

        event_types = [event.type for event in events]
        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual([payload["full_text"] for payload in deltas], ["Проверю файл.", "Файл пустой."])
        self.assertIn("assistant_boundary", event_types)
        self.assertLess(event_types.index("tool_finished"), event_types.index("assistant_boundary"))
        self.assertLess(event_types.index("assistant_boundary"), event_types.index("assistant_delta", 2))

    def test_stream_processor_removes_previous_step_replay_after_boundary(self):
        events = []
        processor = StreamProcessor(events.append)
        previous = "Рабочая директория почти пустая. Создам автономный шаблон лендинга."
        processor._handle_agent_message(AIMessageChunk(content=previous), source="messages")

        processor._begin_assistant_stream_section()
        processor._handle_agent_message(
            AIMessage(content=f"{previous}\n\nФайл index.html создам с готовым шаблоном."),
            source="updates_agent",
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(deltas[-1]["full_text"], "Файл index.html создам с готовым шаблоном.")
        self.assertEqual(processor.full_text, "Файл index.html создам с готовым шаблоном.")

    def test_stream_processor_suppresses_fuzzy_substring_replay_with_typo(self):
        """updates_agent replays a portion of the visible messages text with a
        minor typo fix (e.g. «патерн» → «паттерн»).  The overall quick_ratio is
        too low because visible is much longer, but the containment ratio is
        high — the text must be suppressed to avoid a temporary duplicate."""
        events = []
        processor = StreamProcessor(events.append)

        visible_text = (
            "Получу документацию по LangGraph и asyncio для коректной реализации, "
            "параллельно перечитывая полный ToolBatchCoordinator.run."
            "Документация подтверждает: asyncio.gather с return_exceptions=True "
            "— правильный патерн для параллельных tool-вызовов в кастомном узле. "
            "Теперь изучу импорты и начало файла, чтобы спланировать чистую реализацию."
        )
        processor._handle_agent_message(AIMessageChunk(content=visible_text), source="messages")
        processor._emit_tool_started({"id": "call-read", "name": "read_file", "args": {"path": "tools.py"}})
        processor._handle_tool_result(ToolMessage(content="ok", tool_call_id="call-read", name="read_file"))

        replay_text = (
            "Документация подтверждает: asyncio.gather с return_exceptions=True "
            "— правильный паттерн для параллельных tool-вызовов в кастомном узле. "
            "Теперь изучу импорты и начало файла, чтобы спланировать чистую реализацию."
        )
        processor._handle_agent_message(AIMessage(content=replay_text), source="updates_agent")

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        # Only the messages-stream delta should be emitted; the fuzzy replay
        # must be suppressed.
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], visible_text)
        self.assertEqual(processor.full_text, visible_text)

    def test_stream_processor_suppresses_short_fuzzy_replay_with_added_word(self):
        """updates_agent replays a short visible text with a minor addition
        (e.g. «...синтаксис импорт.» → «...синтаксис и импорт.»).  The texts are
        short (< 120 chars) but quick_ratio is high — the replay must be
        suppressed."""
        events = []
        processor = StreamProcessor(events.append)

        visible_text = "Теперь проверю синтаксис импорт."
        processor._handle_agent_message(AIMessageChunk(content=visible_text), source="messages")
        processor._emit_tool_started({"id": "call-read", "name": "read_file", "args": {"path": "tools.py"}})
        processor._handle_tool_result(ToolMessage(content="ok", tool_call_id="call-read", name="read_file"))

        replay_text = "Теперь проверю синтаксис и импорт."
        processor._handle_agent_message(AIMessage(content=replay_text), source="updates_agent")

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["full_text"], visible_text)
        self.assertEqual(processor.full_text, visible_text)

    class _FakeAssistantDeltaTimer:
        def __init__(self):
            self.active = False

        def isActive(self):
            return self.active

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

    class _FakeTurn:
        def __init__(self):
            self.markdown = ""
            self.markdown_updates = []
            self.streaming = False
            self.tool_started = False
            self._kinds = ["user"]

        def block_kinds(self):
            return list(self._kinds)

        def set_assistant_markdown(self, markdown):
            self.markdown = markdown
            self.markdown_updates.append(markdown)
            if self._kinds[-1] != "assistant":
                self._kinds.append("assistant")

        def set_assistant_streaming(self, active):
            self.streaming = bool(active)

        def start_tool(self, _payload):
            self.tool_started = True
            self._kinds.append("tool_group")

    class _FakeTranscript:
        def __init__(self):
            self.notify_count = 0

        def notify_content_changed(self):
            self.notify_count += 1

    class _FakeStatusController:
        def __init__(self):
            self.run_finished_payload = None

        def on_run_finished(self, payload):
            self.run_finished_payload = payload

    def _make_lightweight_main_window(self):
        window = MainWindow.__new__(MainWindow)
        window.current_turn = self._FakeTurn()
        window.transcript = self._FakeTranscript()
        window._last_assistant_delta_sequence = 0
        window._last_assistant_delta_text = ""
        window._pending_assistant_delta_payload = None
        window._assistant_delta_flush_timer = self._FakeAssistantDeltaTimer()
        window._status_controller = self._FakeStatusController()
        return window

    def test_main_window_coalesces_fast_assistant_delta_updates(self):
        window = self._make_lightweight_main_window()

        MainWindow._on_assistant_delta(window, {"full_text": "A", "sequence": 1})
        self.assertEqual(window.current_turn.markdown, "A")
        self.assertTrue(window.current_turn.streaming)
        self.assertFalse(window._assistant_delta_flush_timer.isActive())

        MainWindow._on_assistant_delta(window, {"full_text": "AB", "sequence": 2})
        self.assertEqual(window.current_turn.markdown, "A")
        self.assertTrue(window._assistant_delta_flush_timer.isActive())

        MainWindow._on_assistant_delta(window, {"full_text": "ABC", "sequence": 3})
        self.assertEqual(window.current_turn.markdown, "A")
        self.assertTrue(window._assistant_delta_flush_timer.isActive())

        MainWindow._flush_pending_assistant_delta(window)
        self.assertEqual(window.current_turn.markdown, "ABC")
        self.assertTrue(window.current_turn.streaming)
        self.assertFalse(window._assistant_delta_flush_timer.isActive())
        self.assertEqual(window.transcript.notify_count, 2)

    def test_main_window_run_finished_flushes_last_coalesced_delta(self):
        window = self._make_lightweight_main_window()

        MainWindow._on_assistant_delta(window, {"full_text": "A", "sequence": 1})
        MainWindow._on_assistant_delta(window, {"full_text": "AB", "sequence": 2})
        MainWindow._on_run_finished(window, {"stats": "done"})

        self.assertEqual(window.current_turn.markdown, "AB")
        self.assertFalse(window.current_turn.streaming)
        self.assertIsNone(window._pending_assistant_delta_payload)
        self.assertFalse(window._assistant_delta_flush_timer.isActive())
        self.assertEqual(window.transcript.notify_count, 2)
        self.assertEqual(window._status_controller.run_finished_payload, {"stats": "done"})

    def test_main_window_terminal_commit_flushes_before_run_finished_without_second_render(self):
        window = self._make_lightweight_main_window()

        MainWindow._on_assistant_delta(window, {"full_text": "**Готов", "sequence": 1})
        MainWindow._on_assistant_delta(window, {"full_text": "**Готово**", "sequence": 2})
        MainWindow._on_assistant_commit(window, {"full_text": "**Готово**", "sequence": 2})

        updates_before_finish = list(window.current_turn.markdown_updates)
        MainWindow._on_run_finished(window, {"stats": "done"})

        self.assertEqual(window.current_turn.markdown, "**Готово**")
        self.assertEqual(window.current_turn.markdown_updates, updates_before_finish)
        self.assertIsNone(window._pending_assistant_delta_payload)

    def test_main_window_normalizes_markdown_only_when_coalesced_delta_is_rendered(self):
        window = self._make_lightweight_main_window()

        with mock.patch(
            "ui.window_components.main_window.prepare_markdown_for_render",
            wraps=prepare_markdown_for_render,
        ) as prepare_mock:
            MainWindow._on_assistant_delta(window, {"full_text": "A", "sequence": 1})
            MainWindow._on_assistant_delta(window, {"full_text": r"\# Header", "sequence": 2})

            self.assertEqual(prepare_mock.call_count, 1)
            MainWindow._flush_pending_assistant_delta(window)

        self.assertEqual(prepare_mock.call_count, 2)
        self.assertEqual(window.current_turn.markdown, "# Header")

    def test_main_window_tool_start_flushes_pending_delta_and_stops_streaming(self):
        window = self._make_lightweight_main_window()

        MainWindow._on_assistant_delta(window, {"full_text": "П", "sequence": 1})
        MainWindow._on_assistant_delta(window, {"full_text": "Проверю.", "sequence": 2})
        MainWindow._on_tool_started(window, {"tool_id": "call-read", "name": "read_file", "args": {"path": "main.py"}})

        self.assertEqual(window.current_turn.markdown, "Проверю.")
        self.assertFalse(window.current_turn.streaming)
        self.assertTrue(window.current_turn.tool_started)

    def test_main_window_assistant_delta_ignores_older_sequence(self):
        window = self._make_lightweight_main_window()

        MainWindow._on_assistant_delta(window, {"full_text": "Новый текст", "sequence": 2})
        MainWindow._on_assistant_delta(window, {"full_text": "Старый текст", "sequence": 1})

        self.assertEqual(window.current_turn.markdown, "Новый текст")
        self.assertEqual(window.transcript.notify_count, 1)

    def test_main_window_accepts_sequence_reset_when_text_extends_previous(self):
        window = self._make_lightweight_main_window()

        MainWindow._on_assistant_delta(window, {"full_text": "Проверю файл.", "sequence": 168})
        MainWindow._on_assistant_delta(window, {"full_text": "Проверю файл. Нашёл проблему.", "sequence": 1})

        self.assertEqual(window.current_turn.markdown, "Проверю файл. Нашёл проблему.")
        self.assertEqual(window._last_assistant_delta_sequence, 1)
        self.assertEqual(window.transcript.notify_count, 2)

    def test_main_window_accepts_sequence_reset_after_tool_group(self):
        window = self._make_lightweight_main_window()

        MainWindow._on_assistant_delta(window, {"full_text": "Проверю файл.", "sequence": 168})
        MainWindow._on_tool_started(window, {"tool_id": "call-read", "name": "read_file", "args": {"path": "main.py"}})
        MainWindow._on_assistant_delta(window, {"full_text": "Добавлю стиль.", "sequence": 1})

        self.assertEqual(window.current_turn.markdown, "Добавлю стиль.")
        self.assertEqual(window._last_assistant_delta_sequence, 1)
        self.assertEqual(window.transcript.notify_count, 3)


if __name__ == "__main__":
    unittest.main()
