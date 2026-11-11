import pytest
from langchain_core.messages import AIMessage, HumanMessage

from core.message_utils import visible_text_content
from ui.runtime_payloads import _assistant_replay_remainder, build_transcript_payload, validate_chat_title
from ui.streaming import StreamProcessor, _InlineThoughtStreamFilter
from ui.widgets.transcript import ConversationTurnWidget


@pytest.mark.parametrize("kind", [
    "reasoning_delta", "reasoning", "analysis_content", "thinking_content", "redacted_thinking",
])
def test_live_and_restored_text_hide_the_same_reasoning_blocks(kind):
    message = AIMessage(content=[{"type": kind, "text": "private"}, {"type": "text", "text": "answer"}])
    original = message.model_dump()
    payload = build_transcript_payload({"messages": [HumanMessage(content="question"), message]})
    assert visible_text_content(message) == "answer"
    assert StreamProcessor._extract_text_content(message.content) == "answer"
    assert payload["turns"][0]["blocks"][0]["markdown"] == "answer"
    assert message.model_dump() == original


@pytest.mark.parametrize("content,expected", [
    ({"type": "refusal", "refusal": "cannot comply"}, "cannot comply"),
    ({"parts": [{"thought": True, "text": "private"}, {"text": "answer"}]}, "answer"),
    (({"text": "a"}, {"text": "b"}), "ab"),
    ([{"text": "<thi"}, {"text": "nk>private</think>answer"}], "answer"),
])
def test_shared_visible_content_supports_provider_payloads(content, expected):
    assert visible_text_content(content, strip_inline=True) == expected
    assert StreamProcessor._extract_text_content(content) == expected


def test_incremental_thought_filter_preserves_split_tags():
    filter_ = _InlineThoughtStreamFilter()
    assert "".join(filter_.feed(part) for part in ["<thi", "nk>private", "</th", "ink>answer"]) == "answer"


@pytest.mark.parametrize("content,expected", [("", "fallback answer"), ("primary answer", "primary answer")])
def test_legacy_additional_content_blocks_match_live_and_restored_display(content, expected):
    message = AIMessage(content=content, additional_kwargs={"content_blocks": [
        {"type": "reasoning", "reasoning": "private"},
        {"type": "text", "text": "fallback answer"},
    ]})
    assert visible_text_content(message) == expected
    events = []
    StreamProcessor(events.append)._handle_agent_message(message)
    assert [event.payload["full_text"] for event in events if event.type == "assistant_delta"] == [expected]
    payload = build_transcript_payload({"messages": [HumanMessage(content="question"), message]})
    assert payload["turns"][0]["blocks"][0]["markdown"] == expected


@pytest.mark.parametrize("content", [
    {"type": "refusal", "refusal": "Cannot complete request"},
    {"content": [{"type": "refusal", "text": "Cannot complete request"}]},
    ({"refusal": "Cannot complete request"},),
])
def test_refusal_is_visible_but_cannot_be_a_chat_title(content):
    assert visible_text_content(content) == "Cannot complete request"
    assert validate_chat_title(content) is None


@pytest.mark.parametrize("previous,incoming,expected", [
    ("Проверя", "Проверяю файлы", "Проверяю файлы"),
    ("Проверю файлы", "Проверю файлы\n\nНашёл ошибку", "Нашёл ошибку"),
    ("Проверю файлы", "Проверю другой проект", "Проверю другой проект"),
    ("Проверю файлы", "Проверю   файлы", ""),
])
def test_restored_and_widget_replay_agree(previous, incoming, expected):
    remainder = _assistant_replay_remainder(previous, incoming)
    assert (incoming if remainder is None else remainder) == expected
    assert ConversationTurnWidget._assistant_resume_text(previous, incoming) == expected
