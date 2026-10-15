import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.message_utils import recent_tool_calls
from core.tool_args import normalize_tool_args_for_repeat_detection, tool_loop_budget


class NormalizeToolArgsForRepeatDetectionTests(unittest.TestCase):
    def test_cosmetic_whitespace_differences_count_as_the_same_call(self):
        base = normalize_tool_args_for_repeat_detection({"path": "a.txt"})
        for variant in (
            {"path": "a.txt "},
            {"path": "  a.txt"},
            {"path": " a.txt  "},
        ):
            with self.subTest(variant=variant):
                self.assertEqual(
                    base,
                    normalize_tool_args_for_repeat_detection(variant),
                    variant,
                )

    def test_inner_whitespace_is_collapsed(self):
        base = normalize_tool_args_for_repeat_detection({"path": "a b.txt"})
        for variant in (
            {"path": "a  b.txt"},
            {"path": "a\tb.txt"},
            {"path": "a \t b.txt"},
        ):
            with self.subTest(variant=variant):
                self.assertEqual(
                    base,
                    normalize_tool_args_for_repeat_detection(variant),
                    variant,
                )

    def test_command_whitespace_and_case_differences_count_as_the_same_call(self):
        base = normalize_tool_args_for_repeat_detection({"command": "Get-Content a.txt"})
        for variant in (
            {"command": "Get-Content  a.txt"},
            {"command": "get-content a.txt"},
            {"command": "GET-CONTENT a.txt"},
            {"command": "Get-Content a.txt "},
        ):
            with self.subTest(variant=variant):
                self.assertEqual(
                    base,
                    normalize_tool_args_for_repeat_detection(variant),
                    variant,
                )

    def test_windows_path_case_and_separator_differences_count_as_the_same_call(self):
        base = normalize_tool_args_for_repeat_detection({"path": "a\\b.txt"})
        for variant in (
            {"path": "a/b.txt"},
            {"path": "A\\B.TXT"},
            {"path": " a\\b.txt "},
        ):
            with self.subTest(variant=variant):
                self.assertEqual(
                    base,
                    normalize_tool_args_for_repeat_detection(variant),
                    variant,
                )

    def test_argument_name_case_and_key_order_do_not_defeat_loop_detection(self):
        base = normalize_tool_args_for_repeat_detection(
            {"path": "a.txt", "content": "x"}
        )
        self.assertEqual(
            base,
            normalize_tool_args_for_repeat_detection({"content": "x", "path": "a.txt"}),
        )
        self.assertEqual(
            base,
            normalize_tool_args_for_repeat_detection({"Path": "a.txt", "CONTENT": "x"}),
        )

    def test_nested_structures_are_normalized_recursively(self):
        base = normalize_tool_args_for_repeat_detection(
            {"args": ["a", "b"], "options": {"Flag": True}}
        )
        self.assertEqual(
            base,
            normalize_tool_args_for_repeat_detection(
                {"options": {"flag": True}, "args": ["a ", "b "]}
            ),
        )

    def test_different_targets_still_count_as_different_calls(self):
        base = normalize_tool_args_for_repeat_detection({"path": "a.txt"})
        for variant in (
            {"path": "b.txt"},
            {"path": "c.txt"},
            {"content": "a.txt"},
        ):
            with self.subTest(variant=variant):
                self.assertNotEqual(
                    base,
                    normalize_tool_args_for_repeat_detection(variant),
                    variant,
                )

    def test_different_commands_still_count_as_different_calls(self):
        base = normalize_tool_args_for_repeat_detection({"command": "rm a.txt"})
        for variant in (
            {"command": "rm b.txt"},
            {"command": "del a.txt"},
            {"path": "a.txt"},
        ):
            with self.subTest(variant=variant):
                self.assertNotEqual(
                    base,
                    normalize_tool_args_for_repeat_detection(variant),
                    variant,
                )

    def test_missing_and_empty_args_normalize_consistently(self):
        self.assertEqual(
            normalize_tool_args_for_repeat_detection({}),
            normalize_tool_args_for_repeat_detection(None),
        )
        self.assertEqual(
            normalize_tool_args_for_repeat_detection('{"path": "a.txt"}'),
            normalize_tool_args_for_repeat_detection({"path": "a.txt"}),
        )


class ToolLoopBudgetTests(unittest.TestCase):
    def test_budget_normalizes_calls_and_uses_readonly_allowlist(self):
        calls = [
            {"name": "read_file", "args": {"path": "A\\B.TXT"}},
            {"name": " read_file ", "args": '{"path": "a/b.txt "}'},
            {"name": "read_file", "args": {"path": "other.txt"}},
            {"name": "write_file", "args": {"path": "a/b.txt"}},
        ]
        for name, expected in (("read_file", (2, 3)), ("write_file", (1, 2)), ("custom_read", (0, 2))):
            with self.subTest(name=name):
                self.assertEqual(
                    tool_loop_budget(
                        calls,
                        tool_name=name,
                        tool_args={"path": "a/b.txt"},
                        readonly_tool_names={"read_file"},
                        readonly_limit=3,
                        mutating_limit=2,
                    ),
                    expected,
                )

    def test_recent_calls_window_counts_messages_not_tool_calls(self):
        def call(call_id):
            return {"name": "read_file", "id": call_id, "args": {}}

        messages = [
            HumanMessage(content="Read files"),
            AIMessage(content="", tool_calls=[call("a"), call("b")]),
            ToolMessage(content="ok", tool_call_id="a"),
            AIMessage(content="", tool_calls=[call("c")]),
            ToolMessage(content="ok", tool_call_id="c"),
        ]
        for window, expected in (
            (0, ["c", "a", "b"]), (-1, ["c", "a", "b"]),
            (1, []), (2, ["c"]), (3, ["c"]), (4, ["c", "a", "b"]),
            (99, ["c", "a", "b"]),
        ):
            with self.subTest(window=window):
                self.assertEqual([call["id"] for call in recent_tool_calls(messages, window)], expected)
                pending = AIMessage(content="", tool_calls=[call("pending")])
                self.assertEqual(
                    [call["id"] for call in recent_tool_calls(messages + [pending], window, exclude_latest=True)],
                    expected,
                )
        self.assertEqual(recent_tool_calls([], 10), [])
        self.assertEqual(recent_tool_calls([], 10, exclude_latest=True), [])


if __name__ == "__main__":
    unittest.main()
