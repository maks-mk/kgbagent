import unittest

from core.self_correction_engine import _inject_yes_flag, build_repair_plan, normalize_tool_args, repair_fingerprint


class SelfCorrectionEngineTests(unittest.TestCase):
    def test_normalize_run_background_process_command_string(self):
        args, changes = normalize_tool_args(
            "run_background_process",
            {"command": "python -m http.server 8000", "cwd": ""},
        )
        self.assertIn("command:str->list", changes)
        self.assertIn("cwd:empty_removed", changes)
        self.assertEqual(args["command"][:3], ["python", "-m", "http.server"])

    def test_normalize_edit_file_aliases(self):
        args, changes = normalize_tool_args(
            "edit_file",
            {"path": "demo.txt", "old_text": "a", "new_text": "b"},
        )
        self.assertIn("old_text->old_string", changes)
        self.assertIn("new_text->new_string", changes)
        self.assertEqual(args["old_string"], "a")
        self.assertEqual(args["new_string"], "b")

    def test_normalize_filesystem_path_aliases_to_canonical_path(self):
        args, changes = normalize_tool_args(
            "write_file",
            {"file_path": "demo.txt", "content": "hello"},
        )
        self.assertIn("file_path->path", changes)
        self.assertEqual(args["path"], "demo.txt")

        delete_args, delete_changes = normalize_tool_args(
            "safe_delete_directory",
            {"dir_path": "build"},
        )
        self.assertIn("dir_path->path", delete_changes)
        self.assertEqual(delete_args["path"], "build")

    def test_build_repair_plan_for_interactive_cli_exec(self):
        issue = {
            "tool_names": ["cli_exec"],
            "tool_args": {"command": "npm install demo-package"},
            "summary": "Interactive prompt detected in cli_exec output ('Ok to proceed? (y)').",
            "error_type": "EXECUTION",
        }
        plan = build_repair_plan(issue, current_task="Установи пакет", max_auto_repairs=2)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan.retryable)
        self.assertEqual(plan.reason, "cli_exec_noninteractive_retry")
        self.assertIn("--yes", str(plan.patched_args.get("command", "")))

    def test_inject_yes_flag_supports_pnpm_and_yarn(self):
        self.assertEqual(_inject_yes_flag("pnpm add demo-package"), "pnpm add demo-package --yes")
        self.assertEqual(_inject_yes_flag("pnpm install"), "pnpm install --yes")
        self.assertEqual(_inject_yes_flag("yarn add demo-package"), "yarn add demo-package --non-interactive")

    def test_inject_yes_flag_does_not_duplicate_existing_flags(self):
        self.assertEqual(_inject_yes_flag("npx --yes create-demo"), "npx --yes create-demo")
        self.assertEqual(_inject_yes_flag("yarn add demo-package --non-interactive"), "yarn add demo-package --non-interactive")

    def test_build_repair_plan_missing_path_prefers_list_directory_retry(self):
        issue = {
            "tool_names": ["edit_file"],
            "tool_args": {"old_string": "x", "new_string": "y"},
            "summary": "Missing required field(s): path.",
            "error_type": "VALIDATION",
            "details": {"missing_required_fields": ["path"]},
        }
        plan = build_repair_plan(issue, current_task="Исправь файл", max_auto_repairs=2)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan.retryable)
        self.assertEqual(plan.reason, "validation_missing_fields")
        self.assertEqual(plan.suggested_tool_name, "list_directory")

    def test_build_repair_plan_write_file_missing_content_replans_write(self):
        issue = {
            "tool_names": ["write_file"],
            "tool_args": {"path": "REFACTORING_SUGGESTIONS.md"},
            "summary": "Missing required field(s): content.",
            "error_type": "VALIDATION",
            "details": {"missing_required_fields": ["content"]},
        }
        plan = build_repair_plan(issue, current_task="Создай файл с рекомендациями", max_auto_repairs=2)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan.retryable)
        self.assertEqual(plan.reason, "validation_missing_write_content")
        self.assertEqual(plan.suggested_tool_name, "write_file")
        self.assertIn("full file body", plan.llm_guidance.lower())

    def test_build_repair_plan_edit_file_match_failure_is_retryable(self):
        issue = {
            "tool_names": ["edit_file"],
            "tool_args": {"path": "weather7.py", "old_string": "bad", "new_string": "good"},
            "summary": "Could not find a match for 'old_string'. Make sure you are replacing existing lines.",
            "error_type": "VALIDATION",
        }
        plan = build_repair_plan(issue, current_task="Исправь файл", max_auto_repairs=2)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan.retryable)
        self.assertEqual(plan.reason, "validation_recoverable")
        self.assertEqual(plan.suggested_tool_name, "read_file")
        self.assertIn("retry edit_file", plan.notes.lower())

    def test_build_repair_plan_workspace_boundary_violation_is_not_retryable(self):
        issue = {
            "kind": "tool_error",
            "tool_names": ["write_file"],
            "tool_args": {"path": "..\\outside.txt", "content": "x"},
            "summary": "Access denied.",
            "error_type": "ACCESS_DENIED",
            "details": {"safety_violation": True},
        }
        plan = build_repair_plan(issue, current_task="Запиши файл", max_auto_repairs=2)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertFalse(plan.retryable)
        self.assertTrue(plan.safety_violation)
        self.assertEqual(plan.reason, "workspace_boundary_violation")

    def test_build_repair_plan_non_retryable_mutating_tool_error_replans_with_llm(self):
        issue = {
            "kind": "tool_error",
            "tool_names": ["ssh_list_sessions"],
            "tool_args": {"host": "demo"},
            "summary": "ERROR[EXECUTION]: [WinError 2] The system cannot find the file specified",
            "error_type": "EXECUTION",
            "details": {
                "retryable": False,
                "tool_read_only": False,
                "tool_mutating": True,
                "tool_requires_approval": True,
            },
        }
        plan = build_repair_plan(issue, current_task="Покажи активные сессии", max_auto_repairs=2)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan.retryable)
        self.assertEqual(plan.strategy, "llm_replan")
        self.assertEqual(plan.reason, "non_retryable_mutating_tool_error")
        self.assertEqual(plan.terminal_reason, "")
        self.assertFalse(plan.needs_external_input)

    def test_repair_fingerprint_is_stable(self):
        first = repair_fingerprint("edit_file", {"path": "a.txt", "old_string": "x"}, "VALIDATION")
        second = repair_fingerprint("edit_file", {"path": "a.txt", "old_string": "x"}, "VALIDATION")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
