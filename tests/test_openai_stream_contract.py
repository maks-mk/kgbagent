"""Exercise both stream loops through the real SDK using a local transport."""

import asyncio
import json
from unittest import mock

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from core.providers.openai_reasoning import _build_reasoning_debug_chat_openai


class Answer(BaseModel):
    answer: str


@pytest.mark.parametrize("structured", [False, True])
def test_sync_async_chat_streams_preserve_reasoning_headers_callbacks_and_final_completion(structured):
    requests = []
    text = '{"answer":"OK"}' if structured else "Visible answer"

    def handle(request):
        assert request.url.path.endswith("/chat/completions")
        requests.append(json.loads(request.content))
        deltas = [
            {"role": "assistant", "content": "", "reasoning_content": "Plan first. "},
            {"content": text, "reasoning_content": "Then answer."},
            {},
        ]
        if not structured:
            # Plain Chat streaming accepts an event that has no delta. The SDK's
            # structured stream instead emits content.delta events we must skip.
            deltas.insert(0, None)
        events = []
        for index, delta in enumerate(deltas):
            events.append({
                "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
                "model": "deepseek-v4-flash",
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": "stop" if index == len(deltas) - 1 else None,
                             "logprobs": {"content": []} if delta and delta.get("content") == text else None}],
            })
        if not structured:
            events.append({
                "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
                "model": "deepseek-v4-flash", "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16,
                          "prompt_cache_hit_tokens": 5},
            })
        data = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, content=data.encode(), headers={
            "content-type": "text/event-stream", "x-request-id": "stream-contract",
        })

    sync_client = httpx.Client(transport=httpx.MockTransport(handle))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    model = _build_reasoning_debug_chat_openai(ChatOpenAI)(
        model="deepseek-v4-flash", api_key="test", base_url="https://gateway.example/v1",
        use_responses_api=False, include_response_headers=not structured,
        http_client=sync_client, http_async_client=async_client,
    )
    sync_manager = mock.Mock()
    async_manager = mock.Mock(on_llm_new_token=mock.AsyncMock())
    kwargs = {"response_format": Answer} if structured else {"stream_usage": True}
    messages = [HumanMessage(content="Answer briefly")]

    async def run_async():
        try:
            sync_chunks = list(model._stream(messages, run_manager=sync_manager, **kwargs))
            async_chunks = [chunk async for chunk in model._astream(messages, run_manager=async_manager, **kwargs)]
            return sync_chunks, async_chunks
        finally:
            await async_client.aclose()

    try:
        sync_chunks, async_chunks = asyncio.run(run_async())
    finally:
        sync_client.close()

    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert [chunk.model_dump() for chunk in sync_chunks] == [chunk.model_dump() for chunk in async_chunks]
    # The structured SDK also produces empty completion/logprobs events.
    assert len(sync_chunks) >= 4 if structured else len(sync_chunks) == 4
    assert sync_manager.on_llm_new_token.call_count == len(sync_chunks)
    assert async_manager.on_llm_new_token.await_count == len(async_chunks)
    for chunks, calls in (
        (sync_chunks, sync_manager.on_llm_new_token.call_args_list),
        (async_chunks, async_manager.on_llm_new_token.await_args_list),
    ):
        assert [call.args[0] for call in calls] == [chunk.text for chunk in chunks]
        assert all(call.kwargs["chunk"] is chunk for call, chunk in zip(calls, chunks))
        assert calls[1].kwargs["logprobs"]["content"] == []
        merged = sum(chunks[1:], chunks[0]).message
        assert merged.content == text
        assert merged.additional_kwargs["reasoning_content"] == "Plan first. Then answer."
        if structured:
            assert merged.additional_kwargs["parsed"] == Answer(answer="OK")
            assert "reasoning_content" not in chunks[-1].message.additional_kwargs
        else:
            assert chunks[0].generation_info["headers"]["x-request-id"] == "stream-contract"
            assert all("headers" not in (chunk.generation_info or {}) for chunk in chunks[1:])
            assert merged.usage_metadata["input_tokens"] == 12
            assert merged.usage_metadata["input_token_details"]["cache_read"] == 5
