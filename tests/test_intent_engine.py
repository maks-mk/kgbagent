import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.message_context import MessageContextHelper


class MessageContextHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.helper = MessageContextHelper()
        self.no_internal_retry = lambda _msg: False

    def test_recent_tool_context_uses_previous_turn_activity(self):
        messages = [
            HumanMessage(content="Запусти сервер"),
            AIMessage(content="", tool_calls=[{"id": "tc-1", "name": "cli_exec", "args": {}}]),
            ToolMessage(name="cli_exec", tool_call_id="tc-1", content="ok"),
            HumanMessage(content="Повтори ещё раз"),
        ]
        recent_names = self.helper.recent_tool_context_names(
            messages=messages,
            is_internal_retry=self.no_internal_retry,
        )
        self.assertEqual(recent_names, ["cli_exec"])

    def test_current_turn_has_tool_evidence_detects_tool_result(self):
        messages = [
            HumanMessage(content="Проверь лог"),
            ToolMessage(name="read_file", tool_call_id="tc-1", content="ok"),
        ]
        self.assertTrue(
            self.helper.current_turn_has_tool_evidence(
                messages,
                is_internal_retry=self.no_internal_retry,
            )
        )

    def test_had_tool_activity_in_previous_turn_uses_history_not_keywords(self):
        messages = [
            HumanMessage(content="Сделай проверку"),
            AIMessage(content="", tool_calls=[{"id": "tc-1", "name": "read_file", "args": {}}]),
            ToolMessage(name="read_file", tool_call_id="tc-1", content="ok"),
            HumanMessage(content="Продолжай"),
        ]
        self.assertTrue(
            self.helper.had_tool_activity_in_previous_turn(
                messages=messages,
                current_turn_id=2,
                is_internal_retry=self.no_internal_retry,
            )
        )

if __name__ == "__main__":
    unittest.main()
