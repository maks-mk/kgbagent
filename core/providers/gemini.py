"""Gemini provider adapter.

Encapsulates Gemini-specific concerns that cannot be expressed through the
public LangChain API:

* **Thought signatures** — Gemini attaches opaque ``thought_signature`` blobs to
  function-call parts.  These must be round-tripped through LangChain messages
  so that multi-turn tool-calling preserves the model's internal reasoning
  state.  We subclass :class:`ChatGoogleGenerativeAI` and override the private
  ``_prepare_request`` / ``_generate`` / ``_agenerate`` methods to inject and
  extract these signatures.

* **Retry-kwargs stripping** — older ``langchain-google-genai`` versions leak
  internal retry-control kwargs (``max_retries``, ``wait_exponential_*``) into
  the underlying SDK call, causing ``TypeError``.  We monkey-patch the module
  functions ``_chat_with_retry`` / ``_achat_with_retry`` to filter them out.

* **Thinking configuration** — Gemini 2.5 models use ``thinking_budget`` while
  Gemini 3 models use ``thinking_level``.  The factory selects the right one
  based on the model name and pydantic-field availability.
"""

from __future__ import annotations

import base64
import importlib
import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

from core.config import AgentConfig
from core.providers.base import (
    chat_model_accepts_kwarg,
    gemini_model_supports_thinking_budget,
    gemini_model_supports_thinking_level,
    normalized_gemini_thinking_level,
)
from core.reasoning_debug import debug_event, log_unknown_fields, preview_value

logger = logging.getLogger("agent")
reasoning_logger = logging.getLogger("agent.reasoning_debug")

_GOOGLE_RETRY_CONTROL_KWARGS = frozenset(
    {
        "max_retries",
        "wait_exponential_multiplier",
        "wait_exponential_min",
        "wait_exponential_max",
    }
)
_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY = "__gemini_function_call_thought_signatures__"


# ---------------------------------------------------------------------------
# Thought-signature encoding / decoding helpers
# ---------------------------------------------------------------------------


def _encode_gemini_thought_signature(signature: Any) -> str:
    if isinstance(signature, str):
        return signature
    if isinstance(signature, memoryview):
        signature = signature.tobytes()
    if isinstance(signature, bytearray):
        signature = bytes(signature)
    if isinstance(signature, bytes):
        return base64.b64encode(signature).decode("ascii")
    return ""


def _decode_gemini_thought_signature(signature: Any) -> bytes:
    if isinstance(signature, bytes):
        return signature
    if isinstance(signature, bytearray):
        return bytes(signature)
    if isinstance(signature, memoryview):
        return signature.tobytes()
    if not isinstance(signature, str) or not signature:
        return b""
    try:
        return base64.b64decode(signature, validate=True)
    except Exception:
        return signature.encode("utf-8")


def _attach_gemini_thought_signatures_to_result(response: Any, result: ChatResult) -> ChatResult:
    if not result.generations:
        return result
    message = result.generations[0].message
    if not isinstance(message, AIMessage) or not message.tool_calls:
        return result

    candidates = list(getattr(response, "candidates", []) or [])
    if not candidates:
        return result
    content = getattr(candidates[0], "content", None)
    raw_parts = list(getattr(content, "parts", []) or [])
    raw_signatures: list[str] = []
    for part in raw_parts:
        function_call = getattr(part, "function_call", None)
        if not str(getattr(function_call, "name", "") or "").strip():
            continue
        encoded_signature = _encode_gemini_thought_signature(getattr(part, "thought_signature", b""))
        raw_signatures.append(encoded_signature)

    signature_map = {
        str(tool_call.get("id") or "").strip(): signature
        for tool_call, signature in zip(message.tool_calls, raw_signatures)
        if str(tool_call.get("id") or "").strip() and signature
    }
    if not signature_map:
        return result

    metadata = dict(getattr(message, "additional_kwargs", {}) or {})
    existing = metadata.get(_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY)
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged.update(signature_map)
    metadata[_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY] = merged
    result.generations[0].message = message.model_copy(update={"additional_kwargs": metadata})
    return result


def _inject_gemini_thought_signatures_into_request(request: Any, messages: list[BaseMessage]) -> None:
    signature_bytes: list[bytes] = []
    for message in messages:
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        metadata = dict(getattr(message, "additional_kwargs", {}) or {})
        signature_map = metadata.get(_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY)
        if not isinstance(signature_map, dict):
            continue
        for tool_call in message.tool_calls:
            tool_call_id = str(tool_call.get("id") or "").strip()
            signature = _decode_gemini_thought_signature(signature_map.get(tool_call_id))
            if signature:
                signature_bytes.append(signature)

    if not signature_bytes:
        return

    next_signature = iter(signature_bytes)
    for content in list(getattr(request, "contents", []) or []):
        if str(getattr(content, "role", "") or "") != "model":
            continue
        for part in list(getattr(content, "parts", []) or []):
            function_call = getattr(part, "function_call", None)
            if not str(getattr(function_call, "name", "") or "").strip():
                continue
            try:
                part.thought_signature = next(next_signature)
            except StopIteration:
                return


def _gemini_finish_reason(response: Any) -> str:
    candidates = list(getattr(response, "candidates", []) or [])
    if not candidates:
        return ""
    return str(getattr(candidates[0], "finish_reason", "") or "")


# ---------------------------------------------------------------------------
# Thought-signature chat-model adapter
# ---------------------------------------------------------------------------


def _build_gemini_thought_signature_adapter(base_cls: type, chat_models_module: Any) -> type:
    """Return a subclass of *base_cls* that round-trips Gemini thought signatures.

    If *chat_models_module* is missing the required private helpers, the original
    class is returned unchanged.
    """
    if not all(
        callable(getattr(chat_models_module, name, None))
        for name in ("_chat_with_retry", "_achat_with_retry", "_response_to_result")
    ):
        return base_cls

    class GeminiThoughtSignatureChatModel(base_cls):
        def _prepare_request(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
            request = super()._prepare_request(messages, *args, **kwargs)
            _inject_gemini_thought_signatures_into_request(request, messages)
            debug_event(
                "final_payload",
                provider="gemini",
                model=getattr(self, "model", None),
                payload_preview=preview_value(request),
            )
            log_unknown_fields("gemini_final_payload", request)
            return request

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager=None,
            *,
            tools=None,
            functions=None,
            safety_settings=None,
            tool_config=None,
            generation_config=None,
            cached_content=None,
            tool_choice=None,
            **kwargs: Any,
        ) -> ChatResult:
            request = self._prepare_request(
                messages,
                stop=stop,
                tools=tools,
                functions=functions,
                safety_settings=safety_settings,
                tool_config=tool_config,
                generation_config=generation_config,
                cached_content=cached_content or self.cached_content,
                tool_choice=tool_choice,
                **kwargs,
            )
            if self.timeout is not None and "timeout" not in kwargs:
                kwargs["timeout"] = self.timeout
            if "max_retries" not in kwargs:
                kwargs["max_retries"] = self.max_retries
            response = chat_models_module._chat_with_retry(
                request=request,
                **kwargs,
                generation_method=self.client.generate_content,
                metadata=self.default_metadata,
            )
            debug_event(
                "final_response_object",
                provider="gemini",
                model=getattr(self, "model", None),
                response_preview=preview_value(response),
                finish_reason=_gemini_finish_reason(response),
            )
            log_unknown_fields("gemini_final_response", response)
            result = chat_models_module._response_to_result(response)
            return _attach_gemini_thought_signatures_to_result(response, result)

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager=None,
            *,
            tools=None,
            functions=None,
            safety_settings=None,
            tool_config=None,
            generation_config=None,
            cached_content=None,
            tool_choice=None,
            **kwargs: Any,
        ) -> ChatResult:
            if not self.async_client:
                updated_kwargs = {
                    **kwargs,
                    "tools": tools,
                    "functions": functions,
                    "safety_settings": safety_settings,
                    "tool_config": tool_config,
                    "generation_config": generation_config,
                }
                return await super()._agenerate(messages, stop, run_manager, **updated_kwargs)

            request = self._prepare_request(
                messages,
                stop=stop,
                tools=tools,
                functions=functions,
                safety_settings=safety_settings,
                tool_config=tool_config,
                generation_config=generation_config,
                cached_content=cached_content or self.cached_content,
                tool_choice=tool_choice,
                **kwargs,
            )
            if self.timeout is not None and "timeout" not in kwargs:
                kwargs["timeout"] = self.timeout
            if "max_retries" not in kwargs:
                kwargs["max_retries"] = self.max_retries
            response = await chat_models_module._achat_with_retry(
                request=request,
                **kwargs,
                generation_method=self.async_client.generate_content,
                metadata=self.default_metadata,
            )
            debug_event(
                "final_response_object",
                provider="gemini",
                model=getattr(self, "model", None),
                response_preview=preview_value(response),
                finish_reason=_gemini_finish_reason(response),
            )
            log_unknown_fields("gemini_final_response", response)
            result = chat_models_module._response_to_result(response)
            return _attach_gemini_thought_signatures_to_result(response, result)

    return GeminiThoughtSignatureChatModel


# ---------------------------------------------------------------------------
# Retry-kwargs monkey-patch
# ---------------------------------------------------------------------------


def patch_langchain_google_genai_retry_kwargs() -> None:
    """Strip internal retry-control kwargs before they reach the Google SDK.

    Idempotent: sets a module-level flag so repeated calls are no-ops.
    """
    try:
        chat_models = importlib.import_module("langchain_google_genai.chat_models")
    except ImportError:
        return

    if bool(getattr(chat_models, "_agent_retry_patch_applied", False)):
        return

    original_chat_with_retry = getattr(chat_models, "_chat_with_retry", None)
    original_achat_with_retry = getattr(chat_models, "_achat_with_retry", None)
    if not callable(original_chat_with_retry) or not callable(original_achat_with_retry):
        return

    def _strip_retry_kwargs(call_kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in dict(call_kwargs).items()
            if key not in _GOOGLE_RETRY_CONTROL_KWARGS
        }

    def _patched_chat_with_retry(generation_method, **kwargs):
        def _wrapped_generation_method(**call_kwargs):
            return generation_method(**_strip_retry_kwargs(call_kwargs))

        return original_chat_with_retry(generation_method=_wrapped_generation_method, **kwargs)

    async def _patched_achat_with_retry(generation_method, **kwargs):
        async def _wrapped_generation_method(**call_kwargs):
            return await generation_method(**_strip_retry_kwargs(call_kwargs))

        return await original_achat_with_retry(generation_method=_wrapped_generation_method, **kwargs)

    chat_models._chat_with_retry = _patched_chat_with_retry
    chat_models._achat_with_retry = _patched_achat_with_retry
    chat_models._agent_retry_patch_applied = True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_gemini_chat_model(config: AgentConfig, *, api_key_override: str | None = None) -> BaseChatModel:
    """Build a Gemini chat model with thought-signature support and thinking config."""
    patch_langchain_google_genai_retry_kwargs()
    # Lazy import to avoid loading both providers on startup.
    from langchain_google_genai import ChatGoogleGenerativeAI

    try:
        chat_models_module = importlib.import_module("langchain_google_genai.chat_models")
    except ImportError:
        chat_models_module = None
    GeminiChatModel = _build_gemini_thought_signature_adapter(ChatGoogleGenerativeAI, chat_models_module)

    # Безопасное извлечение ключа (защита от краша, если ключ None)
    if api_key_override is None:
        api_key = config.gemini_api_key.get_secret_value() if config.gemini_api_key else None
    else:
        api_key = str(api_key_override or "")
    # Sampling is left to the provider default: the agent never sends
    # ``temperature`` so each model uses its own default sampling.
    gemini_kwargs: dict[str, Any] = {
        "model": config.gemini_model,
        "google_api_key": api_key,
        # Keep retry policy in LLMMixin; SDK retries would multiply requests
        # across application retries, key rotation, and stream repair.
        "max_retries": 0,
    }
    gemini_thinking_enabled = False
    gemini_reasoning_mode = "disabled"
    if bool(getattr(config, "enable_model_reasoning", True)):
        if gemini_model_supports_thinking_budget(config.gemini_model) and chat_model_accepts_kwarg(
            GeminiChatModel,
            "thinking_budget",
        ):
            gemini_kwargs["thinking_budget"] = int(getattr(config, "gemini_thinking_budget", 4096))
            gemini_thinking_enabled = True
            gemini_reasoning_mode = "thinking_budget"
        elif gemini_model_supports_thinking_level(config.gemini_model) and chat_model_accepts_kwarg(
            GeminiChatModel,
            "thinking_level",
        ):
            gemini_kwargs["thinking_level"] = normalized_gemini_thinking_level(
                getattr(config, "model_reasoning_effort", "medium")
            )
            gemini_thinking_enabled = True
            gemini_reasoning_mode = "thinking_level"
        if gemini_thinking_enabled and chat_model_accepts_kwarg(GeminiChatModel, "include_thoughts"):
            gemini_kwargs["include_thoughts"] = True
    reasoning_logger.debug(
        "gemini reasoning config model=%s reasoning_enabled=%s mode=%s include_thoughts=%s thinking_budget=%s thinking_level=%s",
        config.gemini_model,
        bool(getattr(config, "enable_model_reasoning", True)),
        gemini_reasoning_mode,
        bool(gemini_kwargs.get("include_thoughts")),
        gemini_kwargs.get("thinking_budget"),
        gemini_kwargs.get("thinking_level"),
    )
    debug_event(
        "reasoning_request",
        provider="gemini",
        base_url=None,
        model=config.gemini_model,
        reasoning_enabled=bool(getattr(config, "enable_model_reasoning", True)),
        reasoning_effort=getattr(config, "model_reasoning_effort", "medium"),
        mode=gemini_reasoning_mode,
    )
    return GeminiChatModel(
        **gemini_kwargs,
    )
