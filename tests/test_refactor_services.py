import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core import constants
from core.config import AgentConfig
from core.context_builder import ContextBuilder
from core.recovery_manager import RecoveryManager
from core.runtime_prompt_policy import RuntimePromptContext, RuntimePromptPolicyBuilder
from core.self_correction_engine import RepairPlan
from core.tool_executor import ToolExecutor
from core.tool_output_compressor import ToolOutputCompressor
from core.tool_issues import build_tool_issue
from core.tool_policy import ToolMetadata
from core.tool_results import parse_tool_execution_result


class RefactorServicesTests(unittest.TestCase):
    def _make_config(self, **overrides) -> AgentConfig:
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

    def _run_recovery_plan(
        self,
        manager,
        *,
        issue=None,
        recovery_state=None,
        state_overrides=None,
        messages=None,
        last_ai=None,
        last_message=None,
        step_count=0,
        max_loops=50,
        hard_loop_ceiling=8,
        max_auto_repairs=8,
    ):
        active_messages = messages or [HumanMessage(content="Исправь задачу")]
        state = {
            "messages": active_messages,
            "steps": step_count,
            "token_usage": {},
        }
        state.update(state_overrides or {})
        return manager.plan_recovery(
            state=state,
            messages=active_messages,
            current_task="Исправь задачу",
            current_turn_id=1,
            open_tool_issue=issue,
            recovery_state=recovery_state or manager.empty_state(turn_id=1),
            last_ai=last_ai,
            last_message=last_message,
            step_count=step_count,
            max_loops=max_loops,
            hard_loop_ceiling=hard_loop_ceiling,
            max_auto_repairs=max_auto_repairs,
            successful_tool_stagnation_limit=3,
        )

    def test_summary_prompt_explicitly_sets_history_summarization_mode(self):
        self.assertIn("Conversation-history summarization mode", constants.SUMMARY_PROMPT_TEMPLATE)
        self.assertIn("update memory for the main model", constants.SUMMARY_PROMPT_TEMPLATE)
        self.assertIn("do not continue or answer the task", constants.SUMMARY_PROMPT_TEMPLATE)

    def test_context_builder_uses_compact_tool_notice_for_large_catalog(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Проверь задачу",
            tools_available=True,
            active_tool_names=[f"tool_{i}" for i in range(8)],
            open_tool_issue=None,
            recovery_state=None,
        )

        system_text = "\n".join(
            str(message.content) for message in context if isinstance(message, SystemMessage)
        )
        self.assertIn(
            "Multiple tools are available in this runtime. Do not invent unavailable tools.", system_text
        )
        self.assertIn("prefer the read-only inspection tool over a mutating one", system_text)
        self.assertNotIn("tool_0, tool_1", system_text)

    def test_context_builder_injects_runtime_contract_from_code(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Editable prompt only",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Проверь задачу",
            tools_available=True,
            active_tool_names=["read_file"],
            open_tool_issue=None,
            recovery_state=None,
        )

        system_texts = [str(message.content) for message in context if isinstance(message, SystemMessage)]
        joined = "\n".join(system_texts)
        self.assertIn("RUNTIME CONTRACT:", joined)
        self.assertNotIn("Always respond in Russian.", joined)
        self.assertNotIn("Before using any tool or tool batch", joined)
        self.assertNotIn("After any system change", joined)
        self.assertIn("Current task: Проверь задачу", joined)
        self.assertIn("TOOLS:", joined)
        self.assertIn("TOOL INTENT REQUIREMENT:", joined)
        self.assertIn("comment in content", joined)
        self.assertIn("Execution environment: os=windows;", joined)
        self.assertIn("paths=windows.", joined)
        self.assertIn("Workspace:", joined)
        self.assertNotIn("Working directory:", joined)
        self.assertIn("Local time:", joined)
        self.assertIn("date=", joined)

    def test_prompt_file_controls_default_response_language(self):
        prompt_text = (Path(__file__).resolve().parents[1] / "prompt.txt").read_text(encoding="utf-8")
        self.assertIn("Respond in Russian unless the task explicitly requires another language.", prompt_text)

    def test_context_builder_keeps_only_workspace_safety_overlay_for_tools(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Editable prompt only",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Проверь задачу",
            tools_available=True,
            active_tool_names=["read_file"],
            open_tool_issue=None,
            recovery_state=None,
        )

        system_texts = [str(message.content) for message in context if isinstance(message, SystemMessage)]
        joined = "\n".join(system_texts)
        self.assertIn("SAFETY POLICY: Any write, delete, move, or process-launch working directory must stay inside the active workspace.", joined)
        self.assertNotIn("Before every tool call", joined)

    def test_context_builder_requests_reasoning_summary_by_default(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Editable prompt only",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Проверь задачу",
            tools_available=False,
            active_tool_names=[],
            open_tool_issue=None,
            recovery_state=None,
        )

        joined = "\n".join(str(message.content) for message in context if isinstance(message, SystemMessage))
        self.assertNotIn("THOUGHT VISIBILITY POLICY:", joined)
        self.assertNotIn("<think>", joined)
        self.assertNotIn("舞台上边...dr", joined)

    def test_context_builder_requests_reasoning_summary_even_when_legacy_toggle_is_false(self):
        builder = ContextBuilder(
            config=self._make_config(SHOW_MODEL_THOUGHTS=False),
            prompt_loader=lambda: "Editable prompt only",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Проверь задачу",
            tools_available=False,
            active_tool_names=[],
            open_tool_issue=None,
            recovery_state=None,
        )

        joined = "\n".join(str(message.content) for message in context if isinstance(message, SystemMessage))
        self.assertNotIn("THOUGHT VISIBILITY POLICY:", joined)
        self.assertNotIn("<think>", joined)
        self.assertNotIn("舞台上边...dr", joined)

    def test_runtime_prompt_policy_detects_environment_once_per_builder(self):
        with mock.patch("core.runtime_prompt_policy.platform.system", return_value="Windows") as detect_os:
            builder = RuntimePromptPolicyBuilder(config=self._make_config())
            context = RuntimePromptContext(
                current_task="Проверь задачу",
                tools_available=False,
                active_tool_names=(),
            )

            builder.build_messages(context)
            builder.build_messages(context)

        self.assertEqual(detect_os.call_count, 1)

    def test_runtime_prompt_policy_updates_workspace_after_directory_change(self):
        first = Path("C:/projects/first")
        second = Path("C:/projects/second")
        with mock.patch("core.runtime_prompt_policy.Path.cwd", side_effect=[first, first, second]):
            builder = RuntimePromptPolicyBuilder(config=self._make_config())
            context = RuntimePromptContext(
                current_task="Проверь задачу",
                tools_available=False,
                active_tool_names=(),
            )

            first_contract = str(builder.build_messages(context)[0].content)
            second_contract = str(builder.build_messages(context)[0].content)

        self.assertIn(str(first.resolve()), first_contract)
        self.assertIn(str(second.resolve()), second_contract)
        self.assertNotIn(str(first.resolve()), second_contract)

    def test_runtime_prompt_policy_maps_supported_operating_systems(self):
        builder = RuntimePromptPolicyBuilder(config=self._make_config())

        with (
            mock.patch("core.runtime_prompt_policy.platform.system", return_value="Windows"),
            mock.patch.dict(
                "core.runtime_prompt_policy.os.environ",
                {"PSModulePath": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\Modules"},
                clear=True,
            ),
        ):
            self.assertEqual(builder._detect_os_family(), "windows")
            self.assertEqual(
                builder._build_execution_environment_line(),
                "Execution environment: os=windows; shell=powershell; paths=windows.",
            )

        with (
            mock.patch("core.runtime_prompt_policy.platform.system", return_value="Linux"),
            mock.patch.dict(
                "core.runtime_prompt_policy.os.environ",
                {"SHELL": "/bin/bash"},
                clear=True,
            ),
        ):
            self.assertEqual(builder._detect_os_family(), "linux")
            self.assertEqual(
                builder._build_execution_environment_line(),
                "Execution environment: os=linux; shell=bash; paths=unix.",
            )

        with (
            mock.patch("core.runtime_prompt_policy.platform.system", return_value="Darwin"),
            mock.patch.dict(
                "core.runtime_prompt_policy.os.environ",
                {"SHELL": "/bin/zsh"},
                clear=True,
            ),
        ):
            self.assertEqual(builder._detect_os_family(), "mac")
            self.assertEqual(
                builder._build_execution_environment_line(),
                "Execution environment: os=mac; shell=zsh; paths=unix.",
            )

    def test_runtime_prompt_policy_falls_back_to_reasonable_shell_defaults(self):
        builder = RuntimePromptPolicyBuilder(config=self._make_config())

        with (
            mock.patch("core.runtime_prompt_policy.platform.system", return_value="Linux"),
            mock.patch.dict("core.runtime_prompt_policy.os.environ", {}, clear=True),
        ):
            self.assertEqual(builder._detect_shell_family(os_family="linux"), "sh")

        with (
            mock.patch("core.runtime_prompt_policy.platform.system", return_value="Windows"),
            mock.patch.dict("core.runtime_prompt_policy.os.environ", {}, clear=True),
        ):
            self.assertEqual(builder._detect_shell_family(os_family="windows"), "unknown")

    def test_runtime_prompt_policy_detects_workspace_and_timezone_metadata(self):
        builder = RuntimePromptPolicyBuilder(config=self._make_config())
        fake_now = datetime(2026, 4, 6, 12, 0, tzinfo=timezone(timedelta(hours=3)))

        with (
            mock.patch("core.runtime_prompt_policy.platform.system", return_value="Linux"),
            mock.patch.dict("core.runtime_prompt_policy.os.environ", {"SHELL": "/bin/bash"}, clear=True),
            mock.patch("core.runtime_prompt_policy.Path.cwd", return_value=Path("/tmp/project")),
            mock.patch("core.runtime_prompt_policy.datetime") as datetime_mock,
        ):
            datetime_mock.now.return_value = fake_now
            environment = builder._detect_execution_environment()

        self.assertEqual(environment.workspace_root, str(Path("/tmp/project").resolve()))
        self.assertEqual(environment.current_working_directory, str(Path("/tmp/project").resolve()))
        self.assertEqual(environment.timezone_name, "UTC+03:00")
        self.assertEqual(environment.utc_offset, "UTC+03:00")

    def test_context_builder_injects_request_user_input_policy_from_code(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Editable prompt only",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Нужен выбор пользователя",
            tools_available=True,
            active_tool_names=["read_file", "request_user_input"],
            open_tool_issue=None,
            recovery_state=None,
        )

        system_texts = [str(message.content) for message in context if isinstance(message, SystemMessage)]
        joined = "\n".join(system_texts)
        self.assertIn("REQUEST_USER_INPUT POLICY:", joined)
        self.assertIn("Never batch multiple request_user_input calls.", joined)
        self.assertIn("Do not use request_user_input for approvals of risky actions", joined)
        self.assertIn("Provide 2 to 5 short mutually exclusive options.", joined)
        self.assertIn("Make the request_user_input tool call by itself.", joined)

    def test_context_builder_does_not_inject_request_user_input_demo_policy(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Editable prompt only",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Сделай тест request_user_input для примера",
            tools_available=True,
            active_tool_names=["request_user_input"],
            open_tool_issue=None,
            recovery_state=None,
        )

        system_texts = [str(message.content) for message in context if isinstance(message, SystemMessage)]
        joined = "\n".join(system_texts)
        self.assertNotIn("REQUEST_USER_INPUT TEST POLICY:", joined)

        regular_context = builder.build(
            [],
            None,
            summary="",
            current_task="Нужно спросить пользователя, в какую папку сохранить файл",
            tools_available=True,
            active_tool_names=["request_user_input"],
            open_tool_issue=None,
            recovery_state=None,
        )
        regular_texts = [str(message.content) for message in regular_context if isinstance(message, SystemMessage)]
        self.assertFalse(any("REQUEST_USER_INPUT TEST POLICY:" in text for text in regular_texts))

    def test_context_builder_preserves_tool_then_user_sequence_without_bridge_messages(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        sanitized = builder.sanitize_messages(
            [
                ToolMessage(content="ok", tool_call_id="tool-1", name="read_file"),
                HumanMessage(content="Продолжай"),
            ]
        )

        self.assertEqual(len(sanitized), 2)
        self.assertIsInstance(sanitized[0], ToolMessage)
        self.assertIsInstance(sanitized[1], HumanMessage)

    def test_context_builder_does_not_repeat_current_task_after_tool_result(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [ToolMessage(content="ok", tool_call_id="tool-1", name="read_file")],
            None,
            summary="",
            current_task="Проверь list_mistral_models.py",
            tools_available=True,
            active_tool_names=["read_file"],
            open_tool_issue=None,
            recovery_state=None,
        )

        self.assertIsInstance(context[-1], ToolMessage)
        self.assertEqual(str(context[-1].content), "ok")

    def test_context_builder_locks_user_choice_after_choice_was_collected(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [ToolMessage(content="Опция C", tool_call_id="tool-1", name="request_user_input")],
            None,
            summary="",
            current_task="Покажи результат теста",
            tools_available=True,
            active_tool_names=["read_file"],
            open_tool_issue=None,
            recovery_state=None,
        )

        system_texts = [str(message.content) for message in context if isinstance(message, SystemMessage)]
        self.assertTrue(any("Do not call request_user_input again" in text for text in system_texts))
        self.assertTrue(any("latest request_user_input ToolMessage" in text for text in system_texts))

    def test_context_builder_strips_historical_images_for_text_only_model_and_keeps_text(self):
        builder = ContextBuilder(
            config=self._make_config(),
            model_capabilities={"image_input_supported": False},
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        sanitized = builder.sanitize_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Опиши предыдущее изображение"},
                        {"type": "image", "path": "C:/tmp/demo.png", "mime_type": "image/png"},
                    ]
                )
            ]
        )

        self.assertEqual(len(sanitized), 1)
        self.assertIsInstance(sanitized[0], HumanMessage)
        self.assertEqual(sanitized[0].content, "Опиши предыдущее изображение")

    def test_context_builder_replaces_image_only_history_for_text_only_model_with_placeholder(self):
        builder = ContextBuilder(
            config=self._make_config(),
            model_capabilities={"image_input_supported": False},
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        sanitized = builder.sanitize_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "image", "path": "C:/tmp/demo.png", "mime_type": "image/png"},
                    ]
                )
            ]
        )

        self.assertEqual(len(sanitized), 1)
        self.assertIsInstance(sanitized[0], HumanMessage)
        self.assertIn("Previous image input omitted", sanitized[0].content)

    def test_context_builder_stringifies_openai_assistant_content_lists(self):
        builder = ContextBuilder(
            # Hermetic: the chat-mode content flattening below must not depend on
            # a local .env that selects LLM_API_MODE=responses.
            config=self._make_config(LLM_API_MODE="chat"),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        sanitized = builder.sanitize_messages(
            [
                HumanMessage(content="Проверь историю"),
                AIMessage(content=["Первый фрагмент. ", "Второй фрагмент."]),
                ToolMessage(content=[{"type": "text", "text": "ok"}], tool_call_id="tool-1", name="read_file"),
            ]
        )

        self.assertEqual(sanitized[1].content, "Первый фрагмент. Второй фрагмент.")
        self.assertEqual(sanitized[2].content, "ok")

    def test_tool_executor_readonly_error_stays_visible_to_agent_without_issue(self):
        executor = ToolExecutor(
            config=self._make_config(),
            metadata_for_tool=lambda name: ToolMetadata(name=name, read_only=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )

        outcome = executor.handle_result(
            state={"run_id": "run"},
            current_turn_id=1,
            tool_name="read_file",
            tool_args={"path": "README.md"},
            tool_call_id="call-1",
            content="ERROR[EXECUTION]: boom",
            apply_validation=False,
            had_error=True,
        )

        self.assertTrue(outcome.had_error)
        self.assertIsNone(outcome.issue)
        self.assertEqual(outcome.tool_message.status, "error")

    def test_tool_executor_treats_plain_error_like_file_contents_as_success(self):
        executor = ToolExecutor(
            config=self._make_config(),
            metadata_for_tool=lambda name: ToolMetadata(name=name, read_only=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )

        outcome = executor.handle_result(
            state={"run_id": "run"},
            current_turn_id=1,
            tool_name="read_file",
            tool_args={"path": "app.log"},
            tool_call_id="call-plain-error-text",
            content="Error: connection reset by peer\nTraceback follows below as part of the log",
            apply_validation=False,
        )

        self.assertFalse(outcome.had_error)
        self.assertIsNone(outcome.issue)
        self.assertEqual(outcome.tool_message.status, "success")

    def test_tool_executor_compresses_oversized_noisy_output_when_enabled(self):
        executor = ToolExecutor(
            config=self._make_config(ENABLE_HEADROOM_COMPRESSION=True, MAX_TOOL_OUTPUT=1000),
            metadata_for_tool=lambda name: ToolMetadata(name=name, read_only=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )
        big_output = "x" * 5000

        with mock.patch.object(
            executor._output_compressor,
            "compress",
            wraps=executor._output_compressor.compress,
        ) as compress_spy:
            outcome = executor.handle_result(
                state={"run_id": "run"},
                current_turn_id=1,
                tool_name="cli_exec",
                tool_args={"command": "pytest"},
                tool_call_id="call-compress",
                content=big_output,
                apply_validation=False,
            )

        self.assertTrue(compress_spy.called)
        # headroom 0.38 folds dense machine-generated text itself (head/tail kept)
        self.assertIn("[COMPRESSED by headroom", outcome.content)
        self.assertLessEqual(len(outcome.content), 1000)

    def test_tool_executor_compresses_real_build_log_through_headroom(self):
        executor = ToolExecutor(
            config=self._make_config(ENABLE_HEADROOM_COMPRESSION=True, MAX_TOOL_OUTPUT=4000),
            metadata_for_tool=lambda name: ToolMetadata(name=name, read_only=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )
        lines = []
        for index in range(600):
            lines.append(f"[{index:04d}] INFO  compiling module_{index % 40}.py ... ok in {index % 7}ms")
            if index % 250 == 0:
                lines.append(f"[{index:04d}] WARNING deprecated call in module_{index % 40}.py:12")
        lines.append("ERROR: build failed: module_37.py:88 SyntaxError: invalid syntax")
        build_log = "\n".join(lines)

        outcome = executor.handle_result(
            state={"run_id": "run", "messages": [HumanMessage(content="why does the build fail")]},
            current_turn_id=1,
            tool_name="cli_exec",
            tool_args={"command": "make build"},
            tool_call_id="call-build",
            content=build_log,
            apply_validation=False,
        )

        self.assertIn("[COMPRESSED by headroom", outcome.content)
        self.assertIn("build failed: module_37.py:88", outcome.content)
        self.assertLess(len(outcome.content), len(build_log))
        self.assertLessEqual(len(outcome.content), 4000)

    def test_tool_executor_marks_mcp_tools_for_compression(self):
        executor = ToolExecutor(
            config=self._make_config(ENABLE_HEADROOM_COMPRESSION=True, MAX_TOOL_OUTPUT=1000),
            metadata_for_tool=lambda name: ToolMetadata(name=name, source="mcp"),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )

        with mock.patch.object(
            executor._output_compressor,
            "compress",
            wraps=executor._output_compressor.compress,
        ) as compress_spy:
            outcome = executor.handle_result(
                state={"run_id": "run"},
                current_turn_id=1,
                tool_name="resolve-library-id",
                tool_args={"query": "headroom"},
                tool_call_id="call-mcp",
                content="MCP-HEAD\n" + ("z" * 5000) + "\nMCP-TAIL",
                apply_validation=False,
            )

        self.assertTrue(compress_spy.called)
        self.assertTrue(compress_spy.call_args.kwargs["is_mcp"])
        self.assertLessEqual(len(outcome.content), 1000)
        self.assertIn("MCP-HEAD", outcome.content)
        self.assertIn("MCP-TAIL", outcome.content)
        # headroom 0.38 compresses the noisy MCP body while keeping its head/tail
        self.assertIn("[COMPRESSED by headroom", outcome.content)

    def test_tool_executor_limits_mcp_output_after_compression(self):
        executor = ToolExecutor(
            config=self._make_config(ENABLE_HEADROOM_COMPRESSION=True, MAX_TOOL_OUTPUT=1000),
            metadata_for_tool=lambda name: ToolMetadata(name=name, source="mcp"),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )
        compressed = "COMPRESSED-HEAD\n" + ("z" * 2000) + "\nCOMPRESSED-TAIL"

        with mock.patch.object(executor._output_compressor, "compress", return_value=compressed):
            outcome = executor.handle_result(
                state={"run_id": "run"},
                current_turn_id=1,
                tool_name="query-docs",
                tool_args={"query": "headroom"},
                tool_call_id="call-mcp",
                content="ORIGINAL-HEAD\n" + ("x" * 5000) + "\nORIGINAL-TAIL",
                apply_validation=False,
            )

        self.assertLessEqual(len(outcome.content), 1000)
        self.assertIn("COMPRESSED-HEAD", outcome.content)
        self.assertIn("COMPRESSED-TAIL", outcome.content)
        self.assertNotIn("ORIGINAL-HEAD", outcome.content)

    def test_tool_executor_never_compresses_file_reads(self):
        executor = ToolExecutor(
            config=self._make_config(ENABLE_HEADROOM_COMPRESSION=True, MAX_TOOL_OUTPUT=1000),
            metadata_for_tool=lambda name: ToolMetadata(name=name, read_only=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )

        with mock.patch.object(
            executor._output_compressor,
            "compress",
            wraps=executor._output_compressor.compress,
        ) as compress_spy:
            outcome = executor.handle_result(
                state={"run_id": "run"},
                current_turn_id=1,
                tool_name="read_file",
                tool_args={"path": "big.py"},
                tool_call_id="call-read",
                content="y" * 5000,
                apply_validation=False,
            )

        # compress() is consulted but declines file reads: exact truncate fallback stays
        self.assertTrue(compress_spy.called)
        self.assertIn("[TRUNCATED from 5000 chars", outcome.content)

    def test_tool_output_compressor_returns_none_when_disabled(self):
        compressor = ToolOutputCompressor(enabled=False)
        self.assertIsNone(
            compressor.compress(content="z" * 5000, tool_name="cli_exec", tool_args=None, limit=1000)
        )

    def test_tool_output_compressor_skips_small_and_incompressible_content(self):
        compressor = ToolOutputCompressor(enabled=True)
        # below min size
        self.assertIsNone(
            compressor.compress(content="z" * 1500, tool_name="cli_exec", tool_args=None, limit=1000)
        )
        # non-compressible tool
        self.assertIsNone(
            compressor.compress(content="z" * 5000, tool_name="read_file", tool_args=None, limit=1000)
        )

    def test_tool_output_compressor_uses_routed_content_when_smaller(self):
        compressor = ToolOutputCompressor(enabled=True)

        fake_result = mock.Mock()
        fake_result.compressed = "compressed body\n" * 30
        fake_result.strategy_used = mock.Mock(value="log")
        fake_router = mock.Mock()
        fake_router.compress.return_value = fake_result

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            result = compressor.compress(
                content="z" * 5000,
                tool_name="cli_exec",
                tool_args={"command": "ls -la"},
                limit=1000,
            )

        self.assertIsNotNone(result)
        self.assertIn("compressed body", result)
        self.assertIn("[COMPRESSED by headroom", result)
        self.assertIn("strategy=log", result)
        # marker must be a footer so a leading ERROR[...] envelope stays detectable
        self.assertTrue(result.startswith("compressed body"))
        fake_router.compress.assert_called_once()
        self.assertIn("ls -la", fake_router.compress.call_args.kwargs["context"])

    def test_tool_output_compressor_compresses_directory_listings(self):
        compressor = ToolOutputCompressor(enabled=True)
        fake_router = mock.Mock()
        fake_router.compress.return_value = mock.Mock(
            compressed="listing body\n" * 40, strategy_used=mock.Mock(value="text")
        )

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            result = compressor.compress(
                content="[FILE] name\n" * 500,
                tool_name="list_directory",
                tool_args={"path": "core"},
                limit=1000,
            )

        self.assertIsNotNone(result)
        fake_router.compress.assert_called_once()

    def test_tool_output_compressor_forwards_user_query_as_question(self):
        compressor = ToolOutputCompressor(enabled=True)
        fake_router = mock.Mock()
        fake_router.compress.return_value = mock.Mock(
            compressed="body\n" * 100, strategy_used=mock.Mock(value="log")
        )

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            compressor.compress(
                content="z" * 5000,
                tool_name="cli_exec",
                tool_args={"command": "pytest"},
                limit=1000,
                user_query="  why do the tests fail  ",
            )

        self.assertEqual(
            "why do the tests fail", fake_router.compress.call_args.kwargs["question"]
        )

    def test_tool_output_compressor_omits_empty_question(self):
        compressor = ToolOutputCompressor(enabled=True)
        fake_router = mock.Mock()
        fake_router.compress.return_value = mock.Mock(
            compressed="body\n" * 100, strategy_used=mock.Mock(value="log")
        )

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            compressor.compress(
                content="z" * 5000,
                tool_name="cli_exec",
                tool_args=None,
                limit=1000,
                user_query="   ",
            )

        self.assertIsNone(fake_router.compress.call_args.kwargs["question"])

    def test_tool_output_compressor_routes_mcp_tools_through_same_router(self):
        compressor = ToolOutputCompressor(enabled=True)

        fake_router = mock.Mock()
        fake_router.compress.return_value = mock.Mock(
            compressed='[{"id": 1}]\n' * 40, strategy_used=mock.Mock(value="smart_crusher")
        )

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            result = compressor.compress(
                content="z" * 5000,
                tool_name="resolve-library-id",
                tool_args={"query": "headroom mcp"},
                limit=1000,
                is_mcp=True,
            )

        self.assertIsNotNone(result)
        self.assertIn('[{"id": 1}]', result)
        self.assertIn("tool=resolve-library-id", result)
        fake_router.compress.assert_called_once()
        self.assertIn("headroom mcp", fake_router.compress.call_args.kwargs["context"])

    def test_tool_output_compressor_passthrough_returns_none(self):
        compressor = ToolOutputCompressor(enabled=True)

        fake_router = mock.Mock()
        fake_router.compress.return_value = mock.Mock(
            compressed="z" * 5000, strategy_used=mock.Mock(value="passthrough")
        )

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            result = compressor.compress(
                content="z" * 5000,
                tool_name="some-mcp-tool",
                tool_args=None,
                limit=1000,
                is_mcp=True,
            )

        self.assertIsNone(result)

    def test_tool_output_compressor_rejects_unresolved_retrieval_markers(self):
        compressor = ToolOutputCompressor(enabled=True)
        marker_bodies = (
            '[{"text": "<<ccr:abc123,string,10KB>>"}]' + ("\npadding" * 200),
            "[1000 items compressed to 20. Retrieve more: hash=abc123.]" + ("\npadding" * 200),
            "[items compressed. Retrieve original: hash=abc123.]" + ("\npadding" * 200),
        )

        for body in marker_bodies:
            for is_mcp in (True, False):
                fake_router = mock.Mock()
                fake_router.compress.return_value = mock.Mock(
                    compressed=body, strategy_used=mock.Mock(value="smart_crusher")
                )
                with mock.patch.object(compressor, "_get_router", return_value=fake_router):
                    result = compressor.compress(
                        content="z" * 5000,
                        tool_name="cli_exec" if not is_mcp else "some-mcp-tool",
                        tool_args=None,
                        limit=1000,
                        is_mcp=is_mcp,
                    )
                self.assertIsNone(result, msg=f"marker leaked (is_mcp={is_mcp}): {body[:40]}")

    def test_tool_output_compressor_rejects_results_that_delete_content(self):
        compressor = ToolOutputCompressor(enabled=True)
        content = "\n".join(f"[{index:04d}] INFO compiling module_{index}.py ... ok" for index in range(400))
        fake_router = mock.Mock()
        fake_router.compress.return_value = mock.Mock(
            compressed="[400 lines omitted: 400 INFO]", strategy_used=mock.Mock(value="log")
        )

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            result = compressor.compress(
                content=content,
                tool_name="cli_exec",
                tool_args=None,
                limit=4000,
            )

        self.assertIsNone(result)

    def test_tool_output_compressor_accepts_pure_deduplication(self):
        compressor = ToolOutputCompressor(enabled=True)
        content = "ERROR[EXECUTION]: boom\nOutput:\n" + ('  File "x.py", line 1\n' * 400)
        deduped = 'ERROR[EXECUTION]: boom\nOutput:\n  File "x.py", line 1\n... (repeated 400 times)'
        fake_router = mock.Mock()
        fake_router.compress.return_value = mock.Mock(
            compressed=deduped, strategy_used=mock.Mock(value="text")
        )

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            result = compressor.compress(
                content=content,
                tool_name="cli_exec",
                tool_args=None,
                limit=4000,
            )

        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("ERROR[EXECUTION]: boom"))

    def test_tool_output_compressor_rejects_results_that_drop_diagnostics(self):
        compressor = ToolOutputCompressor(enabled=True)
        content = "\n".join(
            f"ERROR: module_{index}.py:{index} undefined reference to symbol_{index}"
            if index % 10 == 0
            else f"[{index:04d}] INFO compiling module_{index}.py ... ok"
            for index in range(400)
        )
        # headroom's log compressor caps how many errors it keeps, so most of them vanish
        kept_errors = "\n".join(
            f"ERROR: module_{index}.py:{index} undefined reference to symbol_{index}"
            for index in range(0, 100, 10)
        )
        fake_router = mock.Mock()
        fake_router.compress.return_value = mock.Mock(
            compressed=kept_errors, strategy_used=mock.Mock(value="log")
        )

        # Even the fallback must be rejected if it loses the same diagnostics.
        fake_log = mock.Mock()
        fake_log.compress.return_value = mock.Mock(
            compressed=kept_errors, format_detected=mock.Mock(value="generic")
        )
        with (
            mock.patch.object(compressor, "_get_router", return_value=fake_router),
            mock.patch.object(compressor, "_get_log_compressor", return_value=fake_log),
        ):
            result = compressor.compress(
                content=content,
                tool_name="cli_exec",
                tool_args={"command": "make build"},
                limit=15000,
            )

        self.assertIsNone(result)

    def test_tool_output_compressor_accepts_the_candidate_that_keeps_more_diagnostics(self):
        compressor = ToolOutputCompressor(enabled=True)
        content = "\n".join(
            f"ERROR: shard-{index:03d} unreachable after {index} retries"
            if index % 2
            else f"[{index:04d}] INFO  visiting shard-{index:03d}"
            for index in range(400)
        )
        # more failures than the budget can carry: keeping 70 of them beats the
        # deterministic reducer, which spends most of the budget on head and tail
        kept_errors = "\n".join(
            f"ERROR: shard-{index:03d} unreachable after {index} retries"
            for index in range(1, 140, 2)
        )
        fake_router = mock.Mock()
        fake_router.compress.return_value = mock.Mock(
            compressed=kept_errors, strategy_used=mock.Mock(value="log")
        )

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            result = compressor.compress(
                content=content,
                tool_name="cli_exec",
                tool_args={"command": "make build"},
                limit=4000,
            )

        self.assertIsNotNone(result)
        self.assertIn("shard-139 unreachable", result)

    def test_tool_output_compressor_scales_log_caps_to_budget(self):
        from headroom.transforms import LogCompressorConfig

        defaults = LogCompressorConfig()
        small = ToolOutputCompressor._log_compressor_config(4000, dedupe_warnings=False)
        large = ToolOutputCompressor._log_compressor_config(60000, dedupe_warnings=False)

        # headroom's defaults are the floor, so a small budget is never made worse
        self.assertEqual(defaults.max_total_lines, small.max_total_lines)
        self.assertGreater(large.max_total_lines, small.max_total_lines)
        for config in (small, large):
            self.assertGreaterEqual(config.max_errors, defaults.max_errors)
            self.assertGreaterEqual(config.max_warnings, defaults.max_warnings)
            self.assertGreaterEqual(config.max_stack_traces, defaults.max_stack_traces)
            # no retrieval tool is exposed, so markers must stay off
            self.assertFalse(config.enable_ccr)

    def test_tool_output_compressor_condenses_a_log_the_router_only_folded(self):
        compressor = ToolOutputCompressor(enabled=True)
        lines = []
        for index in range(400):
            lines.append(
                f"2026-09-03 10:{index // 60:02d}:{index % 60:02d} INFO  [build] compiling "
                f"pkg/module_{index:03d}.py ok size={1000 + index}"
            )
            if index % 20 == 7:
                lines.append(
                    f"2026-09-03 10:{index // 60:02d}:{index % 60:02d} ERROR [build] "
                    f"pkg/module_{index:03d}.py: undefined reference to symbol_{index}"
                )
        content = "\n".join(lines)

        result = compressor.compress(
            content=content,
            tool_name="cli_exec",
            tool_args={"command": "make build"},
            limit=15000,
        )

        # the router stops at its lossless fold, which stays above the budget; the
        # log pass has to condense it instead of leaving the cut to the reducer
        self.assertIsNotNone(result)
        self.assertIn("strategy=log", result)
        self.assertLessEqual(len(result), 15000)
        for index in range(7, 400, 20):
            self.assertIn(f"undefined reference to symbol_{index}", result)

    def test_tool_output_compressor_keeps_warnings_that_differ_only_in_an_identifier(self):
        compressor = ToolOutputCompressor(enabled=True)
        lines = []
        for index in range(400):
            stamp = f"2026-09-03 11:{index // 60:02d}:{index % 60:02d}"
            lines.append(f"{stamp} INFO  [build] compiled pkg/module_{index:03d}.py in {index}ms")
            if index and index % 100 == 0:
                lines.append(f"{stamp} WARNING [deps] version drift detected for package-{index}")
        lines.append("2026-09-03 11:07:00 ERROR [build] BUILD FAILED: 1 error, 3 warnings")
        content = "\n".join(lines)

        result = compressor.compress(
            content=content,
            tool_name="cli_exec",
            tool_args={"command": "make build"},
            limit=15000,
        )

        # headroom's warning dedupe normalises digits, so these three warnings look
        # identical to it; collapsing them would hide which packages drifted
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 15000)
        for index in (100, 200, 300):
            self.assertIn(f"package-{index}", result)

    def test_tool_output_compressor_folds_warnings_that_only_repeat(self):
        compressor = ToolOutputCompressor(enabled=True)
        lines = [
            f"2026-09-03 13:{index // 60:02d}:{index % 60:02d} WARNING [cache] "
            f"miss for key=session-token"
            for index in range(400)
        ]
        lines.append("2026-09-03 13:07:00 ERROR [cache] eviction storm: 400 misses in 7 minutes")
        content = "\n".join(lines)

        result = compressor.compress(
            content=content,
            tool_name="cli_exec",
            tool_args={"command": "cat cache.log"},
            limit=15000,
        )

        # the same warning 400 times carries no more information than once, so it must
        # not be spread over the budget just because identifiers are preserved elsewhere
        self.assertIsNotNone(result)
        self.assertLess(len(result), 15000 // 4)
        self.assertIn("session-token", result)
        self.assertIn("eviction storm", result)

    def test_tool_output_compressor_rejects_content_deleting_log_fold(self):
        compressor = ToolOutputCompressor(enabled=True)
        content = "\n".join(
            f"2026-09-03 12:00:{index % 60:02d} INFO  [sync] processed record {index:04d} status=ok"
            for index in range(420)
        )

        result = compressor.compress(
            content=content,
            tool_name="cli_exec",
            tool_args={"command": "cat sync.log"},
            limit=15000,
        )

        # headroom 0.38 folds a diagnostics-free log to a bare line-count marker
        # that carries no content, so compression is rejected and the deterministic
        # reducer keeps verbatim head/tail instead of a useless "lines omitted" note
        self.assertIsNone(result)
        reduced = compressor.reduce_to_limit(
            content=content, tool_name="cli_exec", limit=15000
        )
        self.assertLessEqual(len(reduced), 15000)
        self.assertIn("processed record 0000", reduced)
        self.assertNotIn("lines omitted", reduced)

    def test_tool_output_compressor_leaves_prose_to_the_deterministic_reducer(self):
        compressor = ToolOutputCompressor(enabled=True)
        nouns = ("scheduler", "cache", "queue", "router", "worker", "index")
        verbs = ("retries", "drains", "rebuilds", "validates", "compacts")
        lines = [
            f"Section {index}: the {nouns[index % len(nouns)]} {verbs[index % len(verbs)]} batch "
            f"{index} whenever the operator adjusts threshold {index} in the configuration file, "
            f"which keeps throughput stable during the maintenance window."
            for index in range(400)
        ]
        lines.insert(137, "Section 137b: a warning is emitted when retention drops below one day.")
        content = "\n".join(lines)

        for tool_name, is_mcp in (("fetch_content", False), ("cli_exec", False), ("query-docs", True)):
            result = compressor.compress(
                content=content,
                tool_name=tool_name,
                tool_args={"url": "https://example.test/doc"},
                limit=15000,
                is_mcp=is_mcp,
            )
            # dropping log lines out of prose would delete most of it
            self.assertIsNone(result, msg=f"{tool_name}: prose was log-compressed")

    def test_tool_output_compressor_accepts_reserialised_diagnostics(self):
        compressor = ToolOutputCompressor(enabled=True)
        rows = [
            f'{{"id": "REC-{index:04d}", "status": "ok", "message": "processed record {index:04d}"}}'
            for index in range(1, 200)
        ]
        rows.append('{"id": "REC-0200", "status": "error", "message": "timeout on shard-05"}')
        content = "\n".join(rows)
        # tabular compressors re-serialise records: same data, different shape
        compressed = "id,status,message\nREC-0200,error,timeout on shard-05\n[199 ok rows]"
        fake_router = mock.Mock()
        fake_router.compress.return_value = mock.Mock(
            compressed=compressed, strategy_used=mock.Mock(value="tabular")
        )

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            result = compressor.compress(
                content=content,
                tool_name="query-docs",
                tool_args={"query": "failed records"},
                limit=4000,
                is_mcp=True,
            )

        self.assertIsNotNone(result)
        self.assertIn("REC-0200,error,timeout on shard-05", result)

    def test_tool_output_compressor_skips_mcp_content_within_limit(self):
        compressor = ToolOutputCompressor(enabled=True)
        fake_router = mock.Mock()

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            result = compressor.compress(
                content="z" * 1000,
                tool_name="some-mcp-tool",
                tool_args=None,
                limit=1000,
                is_mcp=True,
            )

        self.assertIsNone(result)
        fake_router.compress.assert_not_called()

    def test_tool_output_compressor_includes_list_arguments_in_context(self):
        compressor = ToolOutputCompressor(enabled=True)
        fake_router = mock.Mock()
        fake_router.compress.return_value = mock.Mock(
            compressed="short", strategy_used=mock.Mock(value="text")
        )

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            compressor.compress(
                content="z" * 5000,
                tool_name="batch_web_search",
                tool_args={"queries": ["first query", "second query"]},
                limit=1000,
            )

        context = fake_router.compress.call_args.kwargs["context"]
        self.assertIn("first query", context)
        self.assertIn("second query", context)

    def test_tool_output_reducer_preserves_all_search_sections(self):
        compressor = ToolOutputCompressor(enabled=False)
        content = "\n".join(
            f"Query: query-{index}\n" + (str(index) * 1200)
            for index in range(5)
        )

        result = compressor.reduce_to_limit(
            content=content,
            tool_name="batch_web_search",
            limit=1000,
        )

        self.assertLessEqual(len(result), 1000)
        for index in range(5):
            self.assertIn(f"Query: query-{index}", result)

    def test_tool_output_reducer_preserves_shell_head_diagnostics_and_tail(self):
        compressor = ToolOutputCompressor(enabled=False)
        content = "HEAD\n" + ("noise\n" * 1000) + "ERROR important failure\n" + ("more\n" * 1000) + "TAIL"

        result = compressor.reduce_to_limit(content=content, tool_name="cli_exec", limit=1000)

        self.assertLessEqual(len(result), 1000)
        self.assertIn("HEAD", result)
        self.assertIn("ERROR important failure", result)
        self.assertIn("TAIL", result)

    def test_tool_output_reducer_preserves_mcp_head_and_tail(self):
        compressor = ToolOutputCompressor(enabled=False)
        content = "MCP-HEAD\n" + ("middle\n" * 1000) + "MCP-TAIL"

        result = compressor.reduce_to_limit(
            content=content,
            tool_name="query-docs",
            limit=1000,
            is_mcp=True,
        )

        self.assertLessEqual(len(result), 1000)
        self.assertIn("MCP-HEAD", result)
        self.assertIn("MCP-TAIL", result)
        omitted = int(result.split("[OMITTED ", 1)[1].split(" chars]", 1)[0])
        marker_length = len(f"\n... [OMITTED {omitted} chars] ...\n")
        self.assertEqual(len(content) - (1000 - marker_length), omitted)

    def test_tool_output_reducer_keeps_error_envelope_detectable(self):
        compressor = ToolOutputCompressor(enabled=False)
        error_head = "ERROR[EXECUTION]: Command failed with Exit Code 1.\nOutput:\n"
        payloads = {
            "cli_exec": error_head + ("stderr noise line\n" * 900),
            "batch_web_search": error_head
            + "".join(f"Query: q{index}\n" + ("y" * 300) + "\n" for index in range(10)),
            "read_file": error_head + ("payload line\n" * 900),
        }

        for tool_name, content in payloads.items():
            reduced = compressor.reduce_to_limit(content=content, tool_name=tool_name, limit=4000)
            self.assertFalse(
                parse_tool_execution_result(reduced).ok,
                msg=f"{tool_name}: reduction hid the ERROR envelope",
            )

    def test_tool_executor_keeps_error_status_for_oversized_shell_failure(self):
        executor = ToolExecutor(
            config=self._make_config(MAX_TOOL_OUTPUT=1000),
            metadata_for_tool=lambda name: ToolMetadata(name=name, mutating=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )

        outcome = executor.handle_result(
            state={"run_id": "run"},
            current_turn_id=1,
            tool_name="cli_exec",
            tool_args={"command": "npm install"},
            tool_call_id="call-shell-error",
            content="ERROR[EXECUTION]: Command failed with Exit Code 1.\nOutput:\n"
            + ("stderr noise line\n" * 400),
            apply_validation=False,
        )

        self.assertTrue(outcome.had_error)
        self.assertFalse(outcome.parsed_result.ok)
        self.assertEqual("EXECUTION", outcome.parsed_result.error_type)
        self.assertEqual("error", outcome.tool_message.status)
        self.assertIsNotNone(outcome.issue)

    def test_tool_executor_forwards_latest_user_query_to_compressor(self):
        executor = ToolExecutor(
            config=self._make_config(ENABLE_HEADROOM_COMPRESSION=True, MAX_TOOL_OUTPUT=1000),
            metadata_for_tool=lambda name: ToolMetadata(name=name, read_only=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )

        with mock.patch.object(
            executor._output_compressor, "compress", return_value=None
        ) as compress_spy:
            executor.handle_result(
                state={
                    "run_id": "run",
                    "messages": [
                        HumanMessage(content="first question"),
                        AIMessage(content="answer"),
                        HumanMessage(content="why does the build fail"),
                    ],
                },
                current_turn_id=1,
                tool_name="cli_exec",
                tool_args={"command": "make"},
                tool_call_id="call-query",
                content="x" * 5000,
                apply_validation=False,
            )

        self.assertEqual(
            "why does the build fail", compress_spy.call_args.kwargs["user_query"]
        )

    def test_tool_output_compressor_falls_back_when_compression_grows_content(self):
        compressor = ToolOutputCompressor(enabled=True)

        fake_result = mock.Mock()
        fake_result.compressed = "z" * 6000  # larger than original
        fake_result.strategy_used = mock.Mock(value="text")
        fake_router = mock.Mock()
        fake_router.compress.return_value = fake_result

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            result = compressor.compress(
                content="z" * 5000,
                tool_name="cli_exec",
                tool_args=None,
                limit=1000,
            )

        self.assertIsNone(result)

    def test_tool_output_compressor_survives_compression_exception(self):
        compressor = ToolOutputCompressor(enabled=True)

        fake_router = mock.Mock()
        fake_router.compress.side_effect = RuntimeError("boom")

        with mock.patch.object(compressor, "_get_router", return_value=fake_router):
            result = compressor.compress(
                content="z" * 5000,
                tool_name="cli_exec",
                tool_args=None,
                limit=1000,
            )

        self.assertIsNone(result)

    def test_tool_executor_promotes_validation_failure_to_error_status(self):
        executor = ToolExecutor(
            config=self._make_config(),
            metadata_for_tool=lambda name: ToolMetadata(name=name, mutating=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )

        with mock.patch("core.tool_executor.validate", return_value="ERROR[VALIDATION]: file was not updated"):
            outcome = executor.handle_result(
                state={"run_id": "run"},
                current_turn_id=1,
                tool_name="edit_file",
                tool_args={"path": "demo.txt"},
                tool_call_id="call-validation",
                content="Success: File edited.",
                apply_validation=True,
            )

        self.assertTrue(outcome.had_error)
        self.assertEqual(outcome.tool_message.status, "error")
        self.assertFalse(outcome.parsed_result.ok)
        self.assertIn("Tool output:", outcome.content)

    def test_tool_executor_isolates_nested_tool_args_from_validation_and_message_state(self):
        executor = ToolExecutor(
            config=self._make_config(),
            metadata_for_tool=lambda name: ToolMetadata(name=name, mutating=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )
        tool_args = {
            "path": "demo.txt",
            "options": {"mode": "append"},
            "lines": ["first"],
        }

        def mutate_validation_payload(_content, context):
            context["args"]["options"]["mode"] = "overwrite"
            context["args"]["lines"].append("second")
            return None

        with mock.patch("core.tool_executor.validate", side_effect=mutate_validation_payload):
            outcome = executor.handle_result(
                state={"run_id": "run"},
                current_turn_id=1,
                tool_name="edit_file",
                tool_args=tool_args,
                tool_call_id="call-nested-copy",
                content="Success: File edited.",
                apply_validation=True,
            )

        self.assertEqual(tool_args["options"]["mode"], "append")
        self.assertEqual(tool_args["lines"], ["first"])
        self.assertEqual(
            outcome.tool_message.additional_kwargs["tool_args"],
            {"path": "demo.txt", "options": {"mode": "append"}, "lines": ["first"]},
        )

    def test_tool_executor_merges_multiple_issues(self):
        executor = ToolExecutor(
            config=self._make_config(),
            metadata_for_tool=lambda name: ToolMetadata(name=name, mutating=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )

        merged = executor.merge_issues(
            [
                build_tool_issue(
                    current_turn_id=2,
                    kind="tool_error",
                    summary="Missing path",
                    tool_names=["edit_file"],
                    tool_args={"path": "a.txt"},
                    source="tools",
                    error_type="VALIDATION",
                    fingerprint="fp-1",
                    details={"missing_required_fields": ["path"]},
                    progress_fingerprint="fp-1",
                ),
                build_tool_issue(
                    current_turn_id=2,
                    kind="tool_error",
                    summary="Loop detected",
                    tool_names=["edit_file"],
                    tool_args={"path": "a.txt"},
                    source="tools",
                    error_type="LOOP_DETECTED",
                    fingerprint="fp-2",
                    details={"loop_detected": True},
                    progress_fingerprint="fp-2",
                ),
            ],
            current_turn_id=2,
        )

        self.assertIsNotNone(merged)
        self.assertIn("Missing path", merged["summary"])
        self.assertIn("edit_file", merged["tool_names"])
        self.assertTrue(merged["details"]["loop_detected"])

    def test_build_tool_issue_copies_nested_payloads(self):
        tool_args = {"path": "demo.txt", "options": {"mode": "append"}}
        details = {"missing_required_fields": ["path"], "nested": {"retry": ["read_file"]}}

        issue = build_tool_issue(
            current_turn_id=2,
            kind="tool_error",
            summary="Missing path",
            tool_names=["edit_file"],
            tool_args=tool_args,
            source="tools",
            error_type="VALIDATION",
            fingerprint="fp-1",
            details=details,
            progress_fingerprint="fp-1",
        )

        tool_args["options"]["mode"] = "overwrite"
        details["nested"]["retry"].append("list_directory")

        self.assertEqual(issue["tool_args"]["options"]["mode"], "append")
        self.assertEqual(issue["details"]["nested"]["retry"], ["read_file"])

    def test_recovery_manager_builds_compact_recovery_message(self):
        manager = RecoveryManager()
        message = manager.build_recovery_system_message(
            {
                "active_issue": {"summary": "Port must be integer"},
                "active_strategy": {
                    "strategy": "normalize_args",
                    "strategy_kind": "fix_args",
                    "llm_guidance": "Retry with normalized arguments.",
                    "suggested_tool_name": "find_process_by_port",
                    "patched_args": {"port": 8080, "extra": "x" * 200},
                    "notes": "Normalize the port before retry.",
                },
            }
        )

        self.assertIsNotNone(message)
        text = str(message.content)
        self.assertIn("RECOVERY MODE:", text)
        self.assertIn("Recovery strategy: fix_args", text)
        self.assertIn("Prepared arguments:", text)
        self.assertNotIn("Structured issue details:", text)
        self.assertNotIn("Do not repeat the exact same failing call unchanged.", text)

    def test_recovery_manager_build_recovery_strategy_copies_nested_payloads(self):
        manager = RecoveryManager()
        repair_plan = RepairPlan(
            strategy="normalize_args",
            reason="validation",
            fingerprint="fp-1",
            tool_name="edit_file",
            suggested_tool_name="edit_file",
            original_args={"path": "demo.txt"},
            patched_args={"path": "demo.txt", "edits": [{"old": "a", "new": "b"}]},
            notes="Retry with normalized arguments.",
        )
        open_tool_issue = {
            "summary": "Need exact old text",
            "details": {"candidates": [{"path": "demo.txt"}]},
        }

        strategy = manager.build_recovery_strategy(
            repair_plan=repair_plan,
            open_tool_issue=open_tool_issue,
            current_task="Обнови файл",
            strategy_id="strategy-1",
        )

        repair_plan.patched_args["edits"][0]["new"] = "c"
        open_tool_issue["details"]["candidates"][0]["path"] = "changed.txt"

        self.assertEqual(strategy["patched_args"]["edits"][0]["new"], "b")
        self.assertEqual(strategy["issue_details"]["candidates"][0]["path"], "demo.txt")

    def test_recovery_manager_handoff_text_hides_internal_recovery_hints(self):
        manager = RecoveryManager()
        text = manager.build_tool_issue_handoff_text(
            {
                "kind": "tool_error",
                "summary": "Command failed with Exit Code 1",
                "tool_names": ["cli_exec"],
                "details": {},
            },
            repair_plan=RepairPlan(
                strategy="llm_replan",
                reason="recovery_stagnated",
                fingerprint="fp-1",
                tool_name="cli_exec",
                suggested_tool_name="cli_exec",
                original_args={"command": "rm bad.txt"},
                patched_args={"command": "rm bad.txt"},
                notes="No deterministic auto-repair available.",
            ),
        )

        self.assertIn("Unable to complete the task", text)
        self.assertIn("stagnation", text.lower())
        self.assertNotIn("Prepared arguments:", text)
        self.assertNotIn("Suggested next tool:", text)
        self.assertNotIn("Hint:", text)

    def test_recovery_manager_builds_soft_internal_ui_notices_by_reason(self):
        manager = RecoveryManager()

        loop_notice = manager.build_internal_ui_notice("loop_budget_exhausted_pending_tool_call")
        stagnation_notice = manager.build_internal_ui_notice("successful_tool_stagnation")
        fallback_notice = manager.build_internal_ui_notice("recovery_stagnated")

        self.assertIn("internal retry limit", loop_notice.lower())
        self.assertIn("loop", stagnation_notice.lower())
        self.assertIn("paused", fallback_notice.lower())

    def test_recovery_manager_finishes_when_self_correction_is_disabled(self):
        manager = RecoveryManager()
        issue = build_tool_issue(
            current_turn_id=1,
            kind="tool_error",
            summary="Missing required field: path",
            tool_names=["edit_file"],
            tool_args={"old_string": "a", "new_string": "b"},
            source="tools",
            error_type="VALIDATION",
            fingerprint="fp-disabled",
            progress_fingerprint="fp-disabled",
            details={"missing_required_fields": ["path"]},
        )

        result = self._run_recovery_plan(
            manager,
            issue=issue,
            hard_loop_ceiling=0,
            max_auto_repairs=0,
        )

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(result["completion_reason"], "self_correction_disabled")
        self.assertIsNone(result["open_tool_issue"])
        self.assertEqual(result["recovery_state"]["active_issue"], issue)
        self.assertEqual(result["recovery_state"]["external_blocker"]["reason"], "self_correction_disabled")
        self.assertTrue(result["handoff_message"])

    def test_recovery_manager_finishes_when_recovery_stagnates(self):
        manager = RecoveryManager()
        issue = build_tool_issue(
            current_turn_id=1,
            kind="protocol_error",
            summary="Malformed tool payload",
            tool_names=["read_file"],
            tool_args={"path": "README.md"},
            source="agent",
            error_type="PROTOCOL",
            fingerprint="fp-stagnated",
            progress_fingerprint="fp-stagnated",
            details={"protocol_reason": "tool_protocol_error"},
        )
        recovery_state = manager.get_recovery_state(
            {
                "turn_id": 1,
                "retry_count": 3,
                "progress_markers": ["fp-stagnated"],
            },
            current_turn_id=1,
        )

        result = self._run_recovery_plan(
            manager,
            issue=issue,
            recovery_state=recovery_state,
            hard_loop_ceiling=3,
        )

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(result["completion_reason"], "recovery_stagnated")
        self.assertEqual(result["recovery_state"]["retry_count"], 3)
        self.assertEqual(result["recovery_state"]["last_reason"], "recovery_stagnated")
        self.assertIsNone(result["open_tool_issue"])
        self.assertEqual(result["recovery_state"]["external_blocker"]["reason"], "recovery_stagnated")
        self.assertTrue(result["handoff_message"])

    def test_recovery_manager_finishes_on_loop_budget_without_pending_tool_call(self):
        manager = RecoveryManager()

        result = self._run_recovery_plan(
            manager,
            step_count=5,
            max_loops=5,
        )

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(result["completion_reason"], "loop_budget_exhausted")
        self.assertTrue(result["loop_budget_reached"])
        self.assertFalse(result["had_pending_tool_calls"])
        self.assertFalse(result["drop_trailing_tool_call"])
        self.assertIsNone(result["open_tool_issue"])
        self.assertEqual(result["recovery_state"]["external_blocker"]["reason"], "loop_budget_exhausted")
        self.assertEqual(result["handoff_message"], "")

    def test_recovery_manager_finishes_without_open_issue_or_trailing_tool_message(self):
        manager = RecoveryManager()
        recovery_state = manager.get_recovery_state(
            {
                "turn_id": 1,
                "active_issue": {"summary": "stale issue"},
                "active_strategy": {"strategy": "llm_replan"},
                "retry_count": 2,
                "retry_fingerprint_history": ["fp-stale"],
                "external_blocker": {"reason": "stale blocker"},
                "llm_replan_attempted_for": ["fp-stale"],
            },
            current_turn_id=1,
        )

        result = self._run_recovery_plan(
            manager,
            recovery_state=recovery_state,
        )

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(result["completion_reason"], "no_open_tool_issue")
        self.assertEqual(result["recovery_state"]["retry_count"], 0)
        self.assertEqual(result["recovery_state"]["retry_fingerprint_history"], [])
        self.assertEqual(result["recovery_state"]["last_reason"], "no_open_tool_issue")
        self.assertIsNone(result["recovery_state"]["active_issue"])
        self.assertIsNone(result["recovery_state"]["active_strategy"])
        self.assertIsNone(result["recovery_state"]["external_blocker"])
        self.assertEqual(result["recovery_state"]["llm_replan_attempted_for"], [])
        self.assertEqual(result["handoff_message"], "")

    def test_recovery_manager_resets_retry_budget_when_issue_fingerprint_changes(self):
        manager = RecoveryManager()
        recovery_state = manager.get_recovery_state(
            {
                "turn_id": 1,
                "retry_count": 5,
                "retry_fingerprint_history": ["fp-old"],
                "progress_markers": ["fp-old"],
                "attempts_by_strategy": {"fp-old::llm_replan": 2},
                "llm_replan_attempted_for": ["fp-old"],
            },
            current_turn_id=1,
        )
        issue = {
            "turn_id": 1,
            "kind": "tool_error",
            "summary": "Missing required field(s): path.",
            "tool_names": ["edit_file"],
            "tool_args": {"old_string": "a", "new_string": "b"},
            "source": "tools",
            "error_type": "VALIDATION",
            "fingerprint": "fp-new",
            "progress_fingerprint": "fp-new",
            "details": {"missing_required_fields": ["path"]},
        }

        result = manager.plan_recovery(
            state={},
            messages=[HumanMessage(content="Исправь файл")],
            current_task="Исправь файл",
            current_turn_id=1,
            open_tool_issue=issue,
            recovery_state=recovery_state,
            last_ai=None,
            last_message=None,
            step_count=0,
            max_loops=50,
            hard_loop_ceiling=8,
            max_auto_repairs=8,
            successful_tool_stagnation_limit=3,
        )

        self.assertEqual(result["turn_outcome"], "recover_agent")
        self.assertEqual(result["completion_reason"], "recover_refresh_context")
        self.assertEqual(result["recovery_state"]["retry_count"], 1)
        self.assertEqual(result["recovery_state"]["retry_fingerprint_history"], ["fp-old", "fp-new"])
        self.assertEqual(result["recovery_state"]["progress_markers"][-1], "fp-new")
        self.assertEqual(result["recovery_state"]["attempts_by_strategy"]["fp-new::refresh_context"], 1)

    def test_recovery_manager_allows_multiple_llm_replans_for_same_issue(self):
        manager = RecoveryManager()
        recovery_state = manager.get_recovery_state(
            {
                "turn_id": 1,
                "retry_count": 3,
                "retry_fingerprint_history": ["fp-protocol"],
                "progress_markers": ["fp-protocol"],
                "attempts_by_strategy": {"fp-protocol::llm_replan": 2},
                "llm_replan_attempted_for": ["fp-protocol"],
            },
            current_turn_id=1,
        )
        issue = {
            "turn_id": 1,
            "kind": "protocol_error",
            "summary": "Malformed tool payload.",
            "tool_names": ["read_file"],
            "tool_args": {"path": "README.md"},
            "source": "agent",
            "error_type": "VALIDATION",
            "fingerprint": "fp-protocol",
            "progress_fingerprint": "fp-protocol",
            "details": {"protocol_reason": "tool_protocol_error"},
        }

        result = manager.plan_recovery(
            state={},
            messages=[HumanMessage(content="Прочитай README.md")],
            current_task="Прочитай README.md",
            current_turn_id=1,
            open_tool_issue=issue,
            recovery_state=recovery_state,
            last_ai=None,
            last_message=None,
            step_count=0,
            max_loops=50,
            hard_loop_ceiling=8,
            max_auto_repairs=8,
            successful_tool_stagnation_limit=3,
        )

        self.assertEqual(result["turn_outcome"], "recover_agent")
        self.assertEqual(result["completion_reason"], "recover_llm_replan")
        self.assertEqual(result["recovery_state"]["retry_count"], 4)
        self.assertEqual(result["recovery_state"]["active_strategy"]["strategy"], "llm_replan")
        self.assertEqual(result["recovery_state"]["attempts_by_strategy"]["fp-protocol::llm_replan"], 3)

    def test_recovery_manager_migrates_legacy_retry_checkpoint_fields(self):
        manager = RecoveryManager()

        migrated = manager.get_recovery_state(
            {"turn_id": 1, "active_strategy": {"strategy": "llm_replan"}},
            current_turn_id=1,
            legacy_state={
                "self_correction_retry_count": 4,
                "self_correction_retry_turn_id": 1,
                "self_correction_fingerprint_history": ["fp-legacy"],
                "self_correction_last_reason": "recover_llm_replan",
            },
        )

        self.assertEqual(migrated["retry_count"], 4)
        self.assertEqual(migrated["retry_fingerprint_history"], ["fp-legacy"])
        self.assertEqual(migrated["last_reason"], "recover_llm_replan")
        self.assertEqual(migrated["active_strategy"]["strategy"], "llm_replan")

    def test_recovery_manager_prefers_nested_retry_state_over_legacy_channels(self):
        manager = RecoveryManager()

        migrated = manager.get_recovery_state(
            {
                "turn_id": 1,
                "retry_count": 2,
                "retry_fingerprint_history": ["fp-current"],
                "last_reason": "recover_refresh_context",
            },
            current_turn_id=1,
            legacy_state={
                "self_correction_retry_count": 7,
                "self_correction_retry_turn_id": 1,
                "self_correction_fingerprint_history": ["fp-legacy"],
                "self_correction_last_reason": "legacy_reason",
            },
        )

        self.assertEqual(migrated["retry_count"], 2)
        self.assertEqual(migrated["retry_fingerprint_history"], ["fp-current"])
        self.assertEqual(migrated["last_reason"], "recover_refresh_context")

    def test_recovery_manager_does_not_migrate_legacy_retry_state_to_new_turn(self):
        manager = RecoveryManager()

        migrated = manager.get_recovery_state(
            {"turn_id": 1, "retry_count": 4},
            current_turn_id=2,
            legacy_state={
                "self_correction_retry_count": 4,
                "self_correction_retry_turn_id": 1,
                "self_correction_fingerprint_history": ["fp-old-turn"],
                "self_correction_last_reason": "recover_llm_replan",
            },
        )

        self.assertEqual(migrated["turn_id"], 2)
        self.assertEqual(migrated["retry_count"], 0)
        self.assertEqual(migrated["retry_fingerprint_history"], [])
        self.assertEqual(migrated["last_reason"], "")

    def test_sanitize_strips_openai_reasoning_content_blocks_for_gemini(self):
        """OpenAI Responses API reasoning blocks (type=reasoning, summary=[...])
        must be stripped so they don't cause KeyError in langchain-google-genai."""
        builder = ContextBuilder(
            config=self._make_config(PROVIDER="gemini"),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        ai_msg = AIMessage(
            content=[
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Thinking..."}], "id": "rs_001"},
                {"type": "text", "text": "Ответ модели"},
            ],
            tool_calls=[],
        )

        sanitized = builder.sanitize_messages([HumanMessage(content="Вопрос"), ai_msg])

        self.assertEqual(len(sanitized), 2)
        self.assertIsInstance(sanitized[1], AIMessage)
        # reasoning block removed, text block preserved
        self.assertEqual(len(sanitized[1].content), 1)
        self.assertEqual(sanitized[1].content[0]["type"], "text")
        self.assertEqual(sanitized[1].content[0]["text"], "Ответ модели")

    def test_sanitize_strips_openai_reasoning_from_additional_kwargs_for_gemini(self):
        """When output_version='v0', langchain-openai moves reasoning into
        additional_kwargs['reasoning'] as a dict with 'summary'. This must be
        stripped for non-OpenAI providers."""
        builder = ContextBuilder(
            config=self._make_config(PROVIDER="gemini"),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        ai_msg = AIMessage(
            content=[{"type": "text", "text": "Ответ"}],
            additional_kwargs={
                "reasoning": {"summary": [{"type": "summary_text", "text": "Hidden thought"}]},
            },
            tool_calls=[],
        )

        sanitized = builder.sanitize_messages([ai_msg])

        self.assertEqual(len(sanitized), 1)
        self.assertNotIn("reasoning", sanitized[0].additional_kwargs)

    def test_sanitize_preserves_gemini_native_reasoning_blocks(self):
        """Gemini's own reasoning blocks have a 'reasoning' string key and must
        NOT be stripped — they are needed for multi-turn tool-calling."""
        builder = ContextBuilder(
            config=self._make_config(PROVIDER="gemini"),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        ai_msg = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "My thought process", "extras": {"signature": "abc123"}},
                {"type": "text", "text": "Ответ"},
            ],
            tool_calls=[],
        )

        sanitized = builder.sanitize_messages([ai_msg])

        self.assertEqual(len(sanitized), 1)
        # Both blocks preserved — Gemini reasoning has 'reasoning' key
        self.assertEqual(len(sanitized[0].content), 2)
        self.assertEqual(sanitized[0].content[0]["type"], "reasoning")
        self.assertIn("reasoning", sanitized[0].content[0])

    def test_sanitize_strips_reasoning_blocks_for_openai_too(self):
        """Reasoning is ephemeral — strip from history even for OpenAI to keep
        cross-provider replay clean and avoid stale reasoning accumulation."""
        builder = ContextBuilder(
            # Hermetic: reasoning is stripped in chat mode; in responses mode it
            # is preserved for tool-call replay, so pin the mode explicitly.
            config=self._make_config(PROVIDER="openai", LLM_API_MODE="chat"),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        ai_msg = AIMessage(
            content=[
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Thinking..."}]},
                {"type": "text", "text": "Ответ"},
            ],
            additional_kwargs={"reasoning": {"summary": [{"text": "thought"}]}},
            tool_calls=[],
        )

        sanitized = builder.sanitize_messages([ai_msg])

        self.assertEqual(len(sanitized), 1)
        # OpenAI provider stringifies content lists — reasoning block stripped,
        # text block becomes a plain string.
        self.assertEqual(sanitized[0].content, "Ответ")
        self.assertNotIn("reasoning", sanitized[0].additional_kwargs)

    def test_sanitize_preserves_anthropic_thinking_blocks_for_anthropic(self):
        """Anthropic thinking blocks (type=thinking, thinking+signature) must be
        preserved for multi-turn continuity — they carry the signature needed
        for extended thinking round-trip."""
        builder = ContextBuilder(
            config=self._make_config(PROVIDER="anthropic", ANTHROPIC_API_KEY="test-key"),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        ai_msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "My reasoning", "signature": "sig_abc"},
                {"type": "text", "text": "Ответ"},
            ],
            tool_calls=[],
        )

        sanitized = builder.sanitize_messages([ai_msg])

        self.assertEqual(len(sanitized), 1)
        # Both blocks preserved — thinking block has signature for round-trip
        self.assertEqual(len(sanitized[0].content), 2)
        self.assertEqual(sanitized[0].content[0]["type"], "thinking")
        self.assertIn("signature", sanitized[0].content[0])
        self.assertEqual(sanitized[0].content[1]["type"], "text")

    def test_sanitize_preserves_anthropic_redacted_thinking_blocks(self):
        """Anthropic redacted_thinking blocks (display=omitted) contain ``data``
        and must be preserved verbatim for multi-turn continuity."""
        builder = ContextBuilder(
            config=self._make_config(PROVIDER="anthropic", ANTHROPIC_API_KEY="test-key"),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        ai_msg = AIMessage(
            content=[
                {"type": "redacted_thinking", "data": "encrypted_blob"},
                {"type": "text", "text": "Ответ"},
            ],
            tool_calls=[],
        )

        sanitized = builder.sanitize_messages([ai_msg])

        self.assertEqual(len(sanitized), 1)
        self.assertEqual(len(sanitized[0].content), 2)
        self.assertEqual(sanitized[0].content[0]["type"], "redacted_thinking")
        self.assertIn("data", sanitized[0].content[0])

    def test_sanitize_strips_openai_reasoning_blocks_for_anthropic(self):
        """When switching from OpenAI to Anthropic, OpenAI reasoning blocks
        (type=reasoning, summary=[...]) must be stripped — they are incompatible
        with Anthropic and are ephemeral."""
        builder = ContextBuilder(
            config=self._make_config(PROVIDER="anthropic", ANTHROPIC_API_KEY="test-key"),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        ai_msg = AIMessage(
            content=[
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Thinking..."}], "id": "rs_001"},
                {"type": "text", "text": "Ответ модели"},
            ],
            tool_calls=[],
        )

        sanitized = builder.sanitize_messages([HumanMessage(content="Вопрос"), ai_msg])

        self.assertEqual(len(sanitized), 2)
        self.assertIsInstance(sanitized[1], AIMessage)
        # OpenAI reasoning block removed, text block preserved
        self.assertEqual(len(sanitized[1].content), 1)
        self.assertEqual(sanitized[1].content[0]["type"], "text")

    def test_normalize_system_prefix_merges_multiple_system_messages_for_anthropic(self):
        """Anthropic (like OpenAI) benefits from merging multiple SystemMessage
        into a single top-level system prompt."""
        builder = ContextBuilder(
            config=self._make_config(PROVIDER="anthropic", ANTHROPIC_API_KEY="test-key"),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = [
            SystemMessage(content="System rule A"),
            SystemMessage(content="System rule B"),
            HumanMessage(content="Вопрос"),
        ]

        normalized = builder.normalize_system_prefix(context)

        self.assertEqual(len(normalized), 2)
        self.assertIsInstance(normalized[0], SystemMessage)
        self.assertIn("System rule A", normalized[0].content)
        self.assertIn("System rule B", normalized[0].content)
        self.assertIsInstance(normalized[1], HumanMessage)


if __name__ == "__main__":
    unittest.main()
