"""OpenAI-compatible provider adapter.

Provides a :class:`ReasoningDebugChatOpenAI` subclass that overrides private
streaming methods (``_stream``, ``_astream``) to:

* log raw provider chunks for reasoning-debug diagnostics;
* attach ``reasoning_content`` from non-standard delta fields
  (``reasoning``, ``thinking``, ``analysis``, …) so that downstream code can
  surface thinking tokens from OpenAI-compatible aggregators that don't follow
  the official ``reasoning_content`` key.

The factory :func:`create_openai_chat_model` wires up sampling controls,
disables SDK-level retries (the agent has its own retry/recovery layer), and
applies reasoning kwargs from the provider registry.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from core.config import AgentConfig
from core.http_headers import load_openai_headers
from core.providers.base import normalized_reasoning_effort, normalized_model_name
from core.provider_registry import (
    ProviderRegistry,
    build_reasoning_kwargs,
    provider_supports_reasoning_for_model,
)
from core.reasoning_debug import (
    STRUCTURED_OUTPUT_PARSED_EXCLUDE,
    debug_event,
    elapsed_since,
    log_unknown_fields,
    now,
    preview_value,
)
from core.text_utils import extract_cache_hit_tokens

logger = logging.getLogger("agent")
reasoning_logger = logging.getLogger("agent.reasoning_debug")


def _safe_openai_model_dump(value: Any) -> dict[str, Any]:
    """Dump OpenAI SDK models without serializing structured-output ``parsed``.

    Some OpenAI/LangChain typed response objects carry the parsed Pydantic
    structured output in ``choices[].message.parsed``.  Recent Pydantic versions
    warn when that runtime value is serialized through a field schema that
    expects ``None``.  The parsed object is not needed for stream chunk
    conversion/debug field discovery; LangChain copies it from the typed final
    response separately when required.
    """
    if not hasattr(value, "model_dump"):
        return value
    try:
        return value.model_dump(
            exclude=STRUCTURED_OUTPUT_PARSED_EXCLUDE,
            warnings="none",
        )
    except TypeError:
        return value.model_dump()


def _flatten_provider_text(value: Any) -> str:
    """Extract text from nested provider blocks without including metadata."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(_flatten_provider_text(item) for item in value)
    if not isinstance(value, dict):
        return ""
    return "".join(
        _flatten_provider_text(value.get(key))
        for key in ("text", "content", "delta", "thinking", "reasoning")
        if value.get(key) not in (None, "")
    )


def _normalize_responses_text_delta(chunk: Any) -> tuple[Any, str]:
    """Normalize non-standard block-list Responses API text deltas.

    Some OpenAI-compatible providers emit ``response.output_text.delta`` with
    a list of ``thinking``/``text`` blocks even though the OpenAI SDK declares
    that field as ``str``. Keep visible text in ``delta`` and return reasoning
    separately so it can remain available to downstream diagnostics/UI.
    """
    if getattr(chunk, "type", None) != "response.output_text.delta":
        return chunk, ""
    delta = getattr(chunk, "delta", None)
    if isinstance(delta, str) or not isinstance(delta, (list, tuple)):
        return chunk, ""

    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    for block in delta:
        if isinstance(block, str):
            visible_parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "").strip().lower()
        if block_type in {"thinking", "reasoning", "analysis", "thought"}:
            reasoning_parts.append(_flatten_provider_text(block))
            continue
        for key in ("thinking", "reasoning", "analysis"):
            if block.get(key) not in (None, ""):
                reasoning_parts.append(_flatten_provider_text(block.get(key)))
        visible_parts.append(
            "".join(
                _flatten_provider_text(block.get(key))
                for key in ("text", "content", "delta")
                if block.get(key) not in (None, "")
            )
        )

    visible_text = "".join(visible_parts)
    reasoning_text = "".join(reasoning_parts)
    if hasattr(chunk, "model_copy"):
        return chunk.model_copy(update={"delta": visible_text}), reasoning_text
    try:
        chunk.delta = visible_text
    except Exception:
        pass
    return chunk, reasoning_text


# ---------------------------------------------------------------------------
# Reasoning-delta extraction
# ---------------------------------------------------------------------------


def extract_openai_reasoning_delta(chunk: Any) -> Any:
    """Extract a reasoning/thinking value from an OpenAI-compatible stream chunk.

    Checks multiple non-standard delta keys used by different aggregators.
    Returns ``None`` if no reasoning content is present.
    """
    if not isinstance(chunk, dict) and hasattr(chunk, "model_dump"):
        try:
            chunk = _safe_openai_model_dump(chunk)
        except Exception:
            return None
    if not isinstance(chunk, dict):
        return None

    candidates: list[Any] = []
    for choice in chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        candidates.extend(
            delta.get(key)
            for key in ("reasoning", "reasoning_content", "thinking", "thinking_content", "analysis", "analysis_content")
            if delta.get(key) not in (None, "")
        )
    for value in candidates:
        if value not in (None, ""):
            return value
    return None


# ---------------------------------------------------------------------------
# Cache-hit normalization
# ---------------------------------------------------------------------------


def extract_openai_stream_cache_read_tokens(chunk: Any) -> int | None:
    """Extract cache-hit tokens from a raw chat-completions stream chunk.

    ``langchain_openai`` maps only ``usage.prompt_tokens_details.cached_tokens``
    into ``usage_metadata``, so OpenAI-compatible providers that report cache
    hits through other usage fields (for example DeepSeek's
    ``usage.prompt_cache_hit_tokens``) would lose the count while streaming.
    """
    if not isinstance(chunk, dict):
        return None
    usage = chunk.get("usage")
    if not isinstance(usage, dict) or not usage:
        return None
    return extract_cache_hit_tokens(usage)


RESPONSES_REASONING_TEXT_EVENTS = frozenset(
    {"response.reasoning_text.delta", "response.reasoning_text.done"}
)


def responses_reasoning_text_blocks(item: Any) -> list[dict[str, Any]]:
    """Extract ``reasoning_text`` content blocks from a Responses reasoning item.

    Aggregators such as AgentRouter stream the full chain of thought as
    ``content: [{"type": "reasoning_text", "text": ...}]`` instead of OpenAI's
    ``summary``. ``langchain_openai`` builds streaming reasoning blocks from
    ``summary`` only, so that text never reaches the outbound payload and strict
    thinking-mode backends reject the follow-up turn with
    ``The `reasoning_text` in the thinking mode must be passed back to the API``.
    """
    content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
    if not isinstance(content, (list, tuple)):
        return []
    blocks: list[dict[str, Any]] = []
    for entry in content:
        if isinstance(entry, dict):
            entry_type, text = entry.get("type"), entry.get("text")
        else:
            entry_type, text = getattr(entry, "type", None), getattr(entry, "text", None)
        if str(entry_type or "") != "reasoning_text" or not isinstance(text, str) or not text:
            continue
        blocks.append({"type": "reasoning_text", "text": text})
    return blocks


def _is_deepseek_chat_model(model: Any) -> bool:
    """Recognize DeepSeek model IDs, including OpenAI-compatible gateway prefixes."""
    name = normalized_model_name(getattr(model, "model_name", None)).rsplit("/", 1)[-1]
    return name.startswith("deepseek-")


# ---------------------------------------------------------------------------
# Reasoning-debug chat model
# ---------------------------------------------------------------------------


def _build_reasoning_debug_chat_openai(base_cls: type) -> type:
    """Return a subclass of ``ChatOpenAI`` with reasoning-debug instrumentation."""

    class ReasoningDebugChatOpenAI(base_cls):
        def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict:
            start = now()
            messages = self._convert_input(input_).to_messages() if _is_deepseek_chat_model(self) else None
            payload = super()._get_request_payload(
                messages if messages is not None else input_, stop=stop, **kwargs
            )
            if messages is not None and "messages" in payload:
                # Keep LangChain's tool/content serialization; restore only the
                # provider field it drops. Preserve all assistant turns, not just
                # tool calls. Sending stored reasoning also covers default thinking
                # mode and invocation-level overrides (without tools it is ignored).
                for message, serialized in zip(messages, payload["messages"]):
                    if not isinstance(message, AIMessage):
                        continue
                    reasoning = message.additional_kwargs.get("reasoning_content")
                    if isinstance(reasoning, str):
                        serialized["reasoning_content"] = reasoning
                    # DeepSeek V4 requires non-null content for tool-call messages.
                    if serialized.get("tool_calls") and serialized.get("content") is None:
                        serialized["content"] = ""
            debug_event(
                "final_payload",
                provider="openai",
                model=getattr(self, "model_name", None) or getattr(self, "model", None),
                base_url=str(getattr(self, "openai_api_base", "") or getattr(self, "base_url", "") or ""),
                payload=payload,
                elapsed=elapsed_since(start),
            )
            log_unknown_fields("openai_final_payload", payload)
            return payload

        def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
            start = now()
            result = super()._generate(*args, **kwargs)
            debug_event(
                "final_response_object",
                provider="openai",
                model=getattr(self, "model_name", None) or getattr(self, "model", None),
                response_preview=preview_value(result),
                elapsed=elapsed_since(start),
            )
            log_unknown_fields("openai_final_response", result)
            return result

        def _create_chat_result(self, response: Any, generation_info: dict | None = None) -> ChatResult:
            result = super()._create_chat_result(response, generation_info)
            if _is_deepseek_chat_model(self):
                response_dict = response if isinstance(response, dict) else _safe_openai_model_dump(response)
                # Match each generation to its own choice, as the parent does.
                for generation, choice in zip(result.generations, response_dict["choices"]):
                    reasoning = choice["message"].get("reasoning_content")
                    if isinstance(generation.message, AIMessage) and isinstance(reasoning, str):
                        generation.message.additional_kwargs["reasoning_content"] = reasoning
            return result

        def _get_generation_chunk_from_completion(self, completion: Any):
            chunk = super()._get_generation_chunk_from_completion(completion)
            if _is_deepseek_chat_model(self):
                # The structured-output final completion repeats reasoning already
                # emitted as deltas. LangChain concatenates string kwargs on merge.
                chunk.message.additional_kwargs.pop("reasoning_content", None)
            return chunk

        def _log_raw_provider_chunk(self, source: str, chunk: Any) -> None:
            fields_source = chunk
            if not isinstance(fields_source, dict) and hasattr(fields_source, "model_dump"):
                try:
                    fields_source = _safe_openai_model_dump(fields_source)
                except Exception:
                    fields_source = chunk
            debug_event(
                "raw_stream_chunk",
                source=source,
                provider="openai",
                model=getattr(self, "model_name", None) or getattr(self, "model", None),
                chunk_preview=preview_value(chunk),
            )
            log_unknown_fields(source, fields_source)

        def _attach_raw_reasoning_delta(self, generation_chunk: Any, chunk: Any) -> None:
            if _is_deepseek_chat_model(self) and isinstance(chunk, dict):
                choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
                delta = (choices[0].get("delta") or {}) if choices else {}
                reasoning = delta.get("reasoning_content")
                message = getattr(generation_chunk, "message", None)
                if isinstance(reasoning, str) and isinstance(message, AIMessage):
                    # Preserve the exact provider string, including empty deltas,
                    # rather than a UI summary or an alternative reasoning field.
                    message.additional_kwargs["reasoning_content"] = reasoning
                    return
            reasoning_delta = extract_openai_reasoning_delta(chunk)
            self._attach_reasoning_content(generation_chunk, reasoning_delta)

        def _convert_chunk_to_generation_chunk(self, chunk: Any, *args: Any, **kwargs: Any):
            generation_chunk = super()._convert_chunk_to_generation_chunk(chunk, *args, **kwargs)
            self._attach_provider_cache_read_tokens(generation_chunk, chunk)
            self._attach_raw_reasoning_delta(generation_chunk, chunk)
            return generation_chunk

        @staticmethod
        def _attach_provider_cache_read_tokens(generation_chunk: Any, chunk: Any) -> None:
            """Report non-OpenAI cache-hit usage fields as standard ``cache_read``."""
            message = getattr(generation_chunk, "message", None)
            usage_metadata = getattr(message, "usage_metadata", None)
            if not isinstance(usage_metadata, dict) or not usage_metadata:
                return
            details = usage_metadata.get("input_token_details")
            details = dict(details) if isinstance(details, dict) else {}
            # ``cache_read`` is prefixed with the service tier for priority/flex requests.
            if any(str(key).endswith("cache_read") for key in details):
                return
            cache_read = extract_openai_stream_cache_read_tokens(chunk)
            if cache_read is None:
                return
            details["cache_read"] = cache_read
            message.usage_metadata = {**usage_metadata, "input_token_details": details}

        @staticmethod
        def _attach_reasoning_content(generation_chunk: Any, reasoning_delta: Any) -> None:
            if reasoning_delta in (None, ""):
                return
            message = getattr(generation_chunk, "message", None)
            if not isinstance(message, AIMessage):
                return
            existing = dict(getattr(message, "additional_kwargs", {}) or {})
            # Store as string for debug/chat-mode compatibility
            existing["reasoning_content"] = reasoning_delta
            # Store as dict for langchain-openai Responses API converter
            existing["reasoning"] = {"type": "reasoning", "encrypted_content": reasoning_delta}
            message.additional_kwargs = existing

        @staticmethod
        def _responses_reasoning_state() -> dict[str, Any]:
            # item id -> accumulated ``reasoning_text``, item id -> block content index
            return {"text": {}, "index": {}}

        @staticmethod
        def _responses_reasoning_blocks(generation_chunk: Any) -> list[dict[str, Any]]:
            content = getattr(getattr(generation_chunk, "message", None), "content", None)
            if not isinstance(content, list):
                return []
            return [block for block in content
                    if isinstance(block, dict) and block.get("type") == "reasoning"]

        @staticmethod
        def _select_responses_reasoning_block(
            blocks: list[dict[str, Any]],
            item_id: str,
            index: Any,
        ) -> dict[str, Any] | None:
            """Pick the block a ``reasoning_text`` payload belongs to.

            Responses may carry several reasoning items, so prefer the block with the
            matching item id, then the recorded block index, and only fall back to a
            lone block when it cannot be ambiguous.
            """
            for block in blocks:
                if item_id and str(block.get("id") or "") == item_id:
                    return block
            if index is not None:
                for block in blocks:
                    if block.get("index") == index:
                        return block
            return blocks[0] if len(blocks) == 1 else None

        @staticmethod
        def _remember_responses_reasoning_text(chunk: Any, state: dict[str, Any]) -> None:
            """Accumulate ``reasoning_text`` events that ``langchain_openai`` ignores."""
            item_id = str(getattr(chunk, "item_id", "") or "")
            if not item_id:
                return
            if str(getattr(chunk, "type", "") or "") == "response.reasoning_text.done":
                text = getattr(chunk, "text", None)
                if isinstance(text, str) and text:
                    state["text"][item_id] = text
                return
            delta = getattr(chunk, "delta", None)
            if isinstance(delta, str) and delta:
                state["text"][item_id] = state["text"].get(item_id, "") + delta

        def _restore_responses_reasoning_text(
            self,
            chunk: Any,
            generation_chunk: Any,
            state: dict[str, Any],
        ) -> Any:
            """Preserve provider ``reasoning_text`` so the next turn can pass it back.

            ``langchain_openai`` collects reasoning blocks from ``summary`` and has no
            branch for ``response.reasoning_text.*`` events, so a provider that streams
            its thinking as ``reasoning_text`` content blocks ends up with
            ``content: []`` in the outbound payload. The recovered text is written back
            onto the emitted block, as thinking-mode providers require.
            """
            chunk_type = str(getattr(chunk, "type", "") or "")
            if chunk_type in RESPONSES_REASONING_TEXT_EVENTS:
                self._remember_responses_reasoning_text(chunk, state)
                return generation_chunk

            if chunk_type != "response.output_item.done":
                for block in self._responses_reasoning_blocks(generation_chunk):
                    if block.get("id") and block.get("index") is not None:
                        state["index"][str(block["id"])] = block["index"]
                return generation_chunk

            item = getattr(chunk, "item", None)
            if getattr(item, "type", None) != "reasoning":
                return generation_chunk
            item_id = str(getattr(item, "id", "") or "")
            text_blocks = responses_reasoning_text_blocks(item)
            if not text_blocks:
                remembered = state["text"].get(item_id, "")
                text_blocks = [{"type": "reasoning_text", "text": remembered}] if remembered else []
            if not text_blocks:
                return generation_chunk
            state["text"].pop(item_id, None)

            index = state["index"].get(item_id)
            if index is None:
                index = getattr(chunk, "output_index", -1)
            block = self._select_responses_reasoning_block(
                self._responses_reasoning_blocks(generation_chunk), item_id, index
            )
            if block is not None:
                # Keep the already streamed block (and its ``index``) so the text merges
                # into it instead of producing a second reasoning item.
                resolved = {**block, "content": text_blocks}
                content = [resolved if entry is block else entry
                           for entry in generation_chunk.message.content]
                return ChatGenerationChunk(
                    message=generation_chunk.message.model_copy(update={"content": content})
                )

            # No block was emitted for the closing event (providers that omit
            # ``encrypted_content``), so surface one bound to the same block index.
            # ``status`` is intentionally absent: langchain-core concatenates string
            # fields while merging, which would corrupt an already streamed status.
            resolved = {
                "type": "reasoning",
                "id": item_id,
                "summary": [],
                "content": text_blocks,
                "index": index,
            }
            encrypted_content = getattr(item, "encrypted_content", None)
            if encrypted_content:
                resolved["encrypted_content"] = encrypted_content
            return ChatGenerationChunk(message=AIMessageChunk(content=[resolved]))

        def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any):
            from langchain_core.messages import AIMessageChunk, BaseMessageChunk
            from langchain_openai.chat_models.base import (
                _convert_responses_chunk_to_generation_chunk,
                _handle_openai_api_error,
                _handle_openai_bad_request,
            )
            import openai
            import warnings

            self._ensure_sync_client_available()
            kwargs["stream"] = True

            if self._use_responses_api({**kwargs, **self.model_kwargs}):
                payload = self._get_request_payload(messages, stop=stop, **kwargs)
                try:
                    if self.include_response_headers:
                        raw_context_manager = self.root_client.with_raw_response.responses.create(**payload)
                        context_manager = raw_context_manager.parse()
                        headers = {"headers": dict(raw_context_manager.headers)}
                    else:
                        context_manager = self.root_client.responses.create(**payload)
                        headers = {}
                    original_schema_obj = kwargs.get("response_format")

                    with context_manager as response:
                        is_first_chunk = True
                        current_index = -1
                        current_output_index = -1
                        current_sub_index = -1
                        has_reasoning = False
                        reasoning_state = self._responses_reasoning_state()
                        for chunk in response:
                            chunk, reasoning_delta = _normalize_responses_text_delta(chunk)
                            self._log_raw_provider_chunk("openai_responses_stream", chunk)
                            metadata = headers if is_first_chunk else {}
                            (
                                current_index,
                                current_output_index,
                                current_sub_index,
                                generation_chunk,
                            ) = _convert_responses_chunk_to_generation_chunk(
                                chunk,
                                current_index,
                                current_output_index,
                                current_sub_index,
                                schema=original_schema_obj,
                                metadata=metadata,
                                has_reasoning=has_reasoning,
                                output_version=self.output_version,
                            )
                            generation_chunk = self._restore_responses_reasoning_text(
                                chunk, generation_chunk, reasoning_state
                            )
                            if generation_chunk:
                                self._attach_reasoning_content(generation_chunk, reasoning_delta)
                                if run_manager:
                                    run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                                is_first_chunk = False
                                if "reasoning" in generation_chunk.message.additional_kwargs:
                                    has_reasoning = True
                                yield generation_chunk
                except openai.BadRequestError as exc:
                    _handle_openai_bad_request(exc)
                except openai.APIError as exc:
                    _handle_openai_api_error(exc)
                return

            from langchain_openai.chat_models.base import _handle_openai_api_error, _handle_openai_bad_request

            stream_usage = self._should_stream_usage(kwargs.pop("stream_usage", None), **kwargs)
            if stream_usage:
                kwargs["stream_options"] = {"include_usage": stream_usage}
            payload = self._get_request_payload(messages, stop=stop, **kwargs)
            default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
            base_generation_info = {}
            response = None

            try:
                if "response_format" in payload:
                    if self.include_response_headers:
                        warnings.warn(
                            "Cannot currently include response headers when response_format is specified."
                        )
                    payload.pop("stream")
                    context_manager = self.root_client.beta.chat.completions.stream(**payload)
                else:
                    if self.include_response_headers:
                        raw_response = self.client.with_raw_response.create(**payload)
                        response = raw_response.parse()
                        base_generation_info = {"headers": dict(raw_response.headers)}
                    else:
                        response = self.client.create(**payload)
                    context_manager = response
                with context_manager as response:
                    is_first_chunk = True
                    for chunk in response:
                        self._log_raw_provider_chunk("openai_chat_completions_stream", chunk)
                        if not isinstance(chunk, dict):
                            chunk = _safe_openai_model_dump(chunk)
                        generation_chunk = self._convert_chunk_to_generation_chunk(
                            chunk,
                            default_chunk_class,
                            base_generation_info if is_first_chunk else {},
                        )
                        if generation_chunk is None:
                            continue
                        default_chunk_class = generation_chunk.message.__class__
                        logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                        if run_manager:
                            run_manager.on_llm_new_token(
                                generation_chunk.text,
                                chunk=generation_chunk,
                                logprobs=logprobs,
                            )
                        is_first_chunk = False
                        yield generation_chunk
            except openai.BadRequestError as exc:
                _handle_openai_bad_request(exc)
            except openai.APIError as exc:
                _handle_openai_api_error(exc)
            if response is not None and hasattr(response, "get_final_completion") and "response_format" in payload:
                final_completion = response.get_final_completion()
                debug_event(
                    "final_response_object",
                    provider="openai",
                    source="openai_chat_completions_stream_final_completion",
                    response_preview=preview_value(final_completion),
                )
                generation_chunk = self._get_generation_chunk_from_completion(final_completion)
                if run_manager:
                    run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                yield generation_chunk

        async def _astream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            from langchain_core.messages import AIMessageChunk, BaseMessageChunk
            from langchain_openai.chat_models.base import (
                _astream_with_chunk_timeout,
                _convert_responses_chunk_to_generation_chunk,
                _handle_openai_api_error,
                _handle_openai_bad_request,
            )
            import openai
            import warnings

            kwargs["stream"] = True

            if self._use_responses_api({**kwargs, **self.model_kwargs}):
                payload = self._get_request_payload(messages, stop=stop, **kwargs)
                try:
                    if self.include_response_headers:
                        raw_context_manager = await self.root_async_client.with_raw_response.responses.create(
                            **payload
                        )
                        context_manager = raw_context_manager.parse()
                        headers = {"headers": dict(raw_context_manager.headers)}
                    else:
                        context_manager = await self.root_async_client.responses.create(**payload)
                        headers = {}
                    original_schema_obj = kwargs.get("response_format")

                    async with context_manager as response:
                        is_first_chunk = True
                        current_index = -1
                        current_output_index = -1
                        current_sub_index = -1
                        has_reasoning = False
                        reasoning_state = self._responses_reasoning_state()
                        async for chunk in _astream_with_chunk_timeout(
                            response,
                            self.stream_chunk_timeout,
                            model_name=self.model_name,
                        ):
                            chunk, reasoning_delta = _normalize_responses_text_delta(chunk)
                            self._log_raw_provider_chunk("openai_responses_stream", chunk)
                            metadata = headers if is_first_chunk else {}
                            (
                                current_index,
                                current_output_index,
                                current_sub_index,
                                generation_chunk,
                            ) = _convert_responses_chunk_to_generation_chunk(
                                chunk,
                                current_index,
                                current_output_index,
                                current_sub_index,
                                schema=original_schema_obj,
                                metadata=metadata,
                                has_reasoning=has_reasoning,
                                output_version=self.output_version,
                            )
                            generation_chunk = self._restore_responses_reasoning_text(
                                chunk, generation_chunk, reasoning_state
                            )
                            if generation_chunk:
                                self._attach_reasoning_content(generation_chunk, reasoning_delta)
                                if run_manager:
                                    await run_manager.on_llm_new_token(
                                        generation_chunk.text,
                                        chunk=generation_chunk,
                                    )
                                is_first_chunk = False
                                if "reasoning" in generation_chunk.message.additional_kwargs:
                                    has_reasoning = True
                                yield generation_chunk
                except openai.BadRequestError as exc:
                    _handle_openai_bad_request(exc)
                except openai.APIError as exc:
                    _handle_openai_api_error(exc)
                return

            stream_usage = self._should_stream_usage(kwargs.pop("stream_usage", None), **kwargs)
            if stream_usage:
                kwargs["stream_options"] = {"include_usage": stream_usage}
            payload = self._get_request_payload(messages, stop=stop, **kwargs)
            default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
            base_generation_info = {}
            response = None

            try:
                if "response_format" in payload:
                    if self.include_response_headers:
                        warnings.warn(
                            "Cannot currently include response headers when response_format is specified."
                        )
                    payload.pop("stream")
                    context_manager = self.root_async_client.beta.chat.completions.stream(**payload)
                else:
                    if self.include_response_headers:
                        raw_response = await self.async_client.with_raw_response.create(**payload)
                        response = raw_response.parse()
                        base_generation_info = {"headers": dict(raw_response.headers)}
                    else:
                        response = await self.async_client.create(**payload)
                    context_manager = response
                async with context_manager as response:
                    is_first_chunk = True
                    async for chunk in _astream_with_chunk_timeout(
                        response,
                        self.stream_chunk_timeout,
                        model_name=self.model_name,
                    ):
                        self._log_raw_provider_chunk("openai_chat_completions_stream", chunk)
                        if not isinstance(chunk, dict):
                            chunk = _safe_openai_model_dump(chunk)
                        generation_chunk = self._convert_chunk_to_generation_chunk(
                            chunk,
                            default_chunk_class,
                            base_generation_info if is_first_chunk else {},
                        )
                        if generation_chunk is None:
                            continue
                        default_chunk_class = generation_chunk.message.__class__
                        logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                        if run_manager:
                            await run_manager.on_llm_new_token(
                                generation_chunk.text,
                                chunk=generation_chunk,
                                logprobs=logprobs,
                            )
                        is_first_chunk = False
                        yield generation_chunk
            except openai.BadRequestError as exc:
                _handle_openai_bad_request(exc)
            except openai.APIError as exc:
                _handle_openai_api_error(exc)
            if response is not None and hasattr(response, "get_final_completion") and "response_format" in payload:
                final_completion = await response.get_final_completion()
                debug_event(
                    "final_response_object",
                    provider="openai",
                    source="openai_chat_completions_stream_final_completion",
                    response_preview=preview_value(final_completion),
                )
                generation_chunk = self._get_generation_chunk_from_completion(final_completion)
                if run_manager:
                    await run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                yield generation_chunk

    return ReasoningDebugChatOpenAI


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


# Models that reject the ``temperature`` parameter: reasoning-only models with
# fixed sampling. Sending it causes a 400 from the API.
_FIXED_SAMPLING_OPENAI_MODELS = frozenset({"gpt-6-astra"})


def _openai_uses_fixed_sampling(model_name: str | None) -> bool:
    """Match OpenAI-compatible models that reject custom sampling controls."""
    normalized = normalized_model_name(model_name).rsplit("/", 1)[-1]
    if len(normalized) > 4 and normalized[-4] == "-" and normalized[-3:].isdigit():
        normalized = normalized[:-4]
    return normalized in _FIXED_SAMPLING_OPENAI_MODELS


def create_openai_chat_model(config: AgentConfig, *, api_key_override: str | None = None) -> BaseChatModel:
    """Build an OpenAI-compatible chat model with reasoning-debug instrumentation."""
    # Lazy import to avoid loading both providers on startup.
    from langchain_openai import ChatOpenAI as BaseChatOpenAI
    from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

    ReasoningDebugChatOpenAI = _build_reasoning_debug_chat_openai(BaseChatOpenAI)
    ChatOpenAI = ReasoningDebugChatOpenAI

    if api_key_override is None:
        api_key = config.openai_api_key.get_secret_value() if config.openai_api_key else None
    else:
        api_key = str(api_key_override or "")
    openai_kwargs: dict[str, Any] = {
        "model": config.openai_model,
        "api_key": api_key,
        "base_url": config.openai_base_url,
        "default_headers": load_openai_headers(),
        "http_client": DefaultHttpxClient(),
        "http_async_client": DefaultAsyncHttpxClient(),
        "max_retries": 0,
        "stream_usage": True,
    }
    # gpt-6-astra and other fixed-sampling models reject ``temperature``.
    if not _openai_uses_fixed_sampling(config.openai_model):
        openai_kwargs["temperature"] = config.temperature

    # Explicit API mode selection: "responses" forces /v1/responses, "chat" forces
    # /v1/chat/completions. When omitted, LangChain auto-detects based on payload.
    api_mode = str(getattr(config, "llm_api_mode", "chat") or "chat").strip().lower()
    if api_mode == "responses":
        openai_kwargs["use_responses_api"] = True
    registry = ProviderRegistry.from_path(config.provider_registry_path)
    provider_config = registry.match(config.openai_base_url, config.openai_model)
    reasoning_enabled = bool(getattr(config, "enable_model_reasoning", True))
    provider_model_supports_reasoning = provider_supports_reasoning_for_model(provider_config, config.openai_model)
    if provider_model_supports_reasoning and (
        reasoning_enabled
        or (
            isinstance(provider_config, dict)
            and isinstance(provider_config.get("reasoning"), dict)
            and "disabled_value" in provider_config["reasoning"]
        )
    ):
        debug_event(
            "reasoning_request",
            provider=provider_config.get("id") if isinstance(provider_config, dict) else None,
            base_url=config.openai_base_url,
            model=config.openai_model,
            reasoning_enabled=bool(getattr(config, "enable_model_reasoning", True)),
            reasoning_effort=normalized_reasoning_effort(getattr(config, "model_reasoning_effort", "medium")),
        )
        reasoning_logger.debug(
            "openai reasoning registry match model=%s base_url=%s provider_id=%s supports_reasoning=%s model_supported=%s validation=%s path=%s effort=%s enabled=%s",
            config.openai_model,
            config.openai_base_url,
            provider_config.get("id") if isinstance(provider_config, dict) else None,
            provider_config.get("supports_reasoning") if isinstance(provider_config, dict) else None,
            provider_model_supports_reasoning,
            provider_config.get("validation") if isinstance(provider_config, dict) else None,
            (provider_config.get("reasoning") or {}).get("path") if isinstance(provider_config, dict) else None,
            normalized_reasoning_effort(getattr(config, "model_reasoning_effort", "medium")),
            reasoning_enabled,
        )
        build_reasoning_kwargs(
            openai_kwargs,
            provider_config,
            normalized_reasoning_effort(getattr(config, "model_reasoning_effort", "medium")),
            enabled=reasoning_enabled,
        )

        # In chat mode, a top-level "reasoning" dict (set by providers whose registry
        # path is "reasoning.effort") would cause LangChain to auto-switch to the
        # Responses API. Flatten it to the chat-completions "reasoning_effort" string
        # so the explicit API mode is respected.
        if api_mode != "responses" and isinstance(openai_kwargs.get("reasoning"), dict):
            reasoning_dict = openai_kwargs.pop("reasoning")
            effort = reasoning_dict.get("effort")
            if effort and "reasoning_effort" not in openai_kwargs:
                openai_kwargs["reasoning_effort"] = str(effort)
            # reasoning.summary is Responses-API-only; drop it in chat mode.
        reasoning_logger.debug(
            "openai reasoning kwargs applied model=%s provider_id=%s reasoning_path=%s has_reasoning_key=%s has_extra_body=%s reasoning_effort=%s extra_body_keys=%s applied_keys=%s",
            config.openai_model,
            provider_config.get("id") if isinstance(provider_config, dict) else None,
            (provider_config.get("reasoning") or {}).get("path") if isinstance(provider_config, dict) else None,
            "reasoning" in openai_kwargs,
            "extra_body" in openai_kwargs,
            openai_kwargs.get("reasoning_effort"),
            sorted((openai_kwargs.get("extra_body") or {}).keys())
            if isinstance(openai_kwargs.get("extra_body"), dict)
            else [],
            sorted(k for k in openai_kwargs if k not in ("model", "temperature", "api_key", "base_url", "max_retries", "stream_usage")),
        )
    else:
        reasoning_logger.debug(
            "openai reasoning skipped model=%s base_url=%s provider_id=%s reasoning_enabled=%s provider_supports_reasoning=%s model_supported=%s",
            config.openai_model,
            config.openai_base_url,
            provider_config.get("id") if isinstance(provider_config, dict) else None,
            reasoning_enabled,
            provider_config.get("supports_reasoning") if isinstance(provider_config, dict) else None,
            provider_model_supports_reasoning,
        )
    return ChatOpenAI(**openai_kwargs)
