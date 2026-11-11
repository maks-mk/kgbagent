from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, List

from langgraph.config import get_stream_writer

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from core.api_key_rotation import (
    ApiKeyRotationExhaustedError,
    classify_provider_error,
    describe_provider_error,
)
from core.state import AgentState
from core.node_errors import EmptyLLMResponseError
from core.message_utils import visible_text_content
from core.text_utils import has_visible_text
from core.providers.anthropic import anthropic_prompt_cache_kwargs
from core.providers.factory import prepare_llm_with_tools

logger = logging.getLogger("agent")

_TRANSIENT_RETRY_DELAYS = (2, 4, 8)
class LLMMixin:
    """LLM selection, invocation with retry, and fatal-error classification."""

    def _select_llm_for_active_tools(
        self,
        active_tools: List[Any],
        active_tool_names: List[str],
    ) -> BaseChatModel:
        if not active_tool_names:
            return self.llm

        if active_tool_names == list(self._all_tool_names):
            return self.llm_with_tools

        bound_llm, enabled, error = prepare_llm_with_tools(self.llm, active_tools)
        if enabled:
            return bound_llm
        logger.warning(
            "Failed to bind active tool subset; falling back to pre-bound tool model: %s",
            error,
        )
        return self.llm_with_tools

    async def _invoke_llm_with_retry(
        self,
        llm,
        context: List[Any],
        state: AgentState | None = None,
        node_name: str = "",
    ):
        current_llm = llm
        context = list(context)
        # One-shot summarization calls bypass this loop: only agent requests
        # build a reusable conversation prefix worth caching.
        invocation_kwargs = anthropic_prompt_cache_kwargs(self.config)
        configured_max_attempts = max(1, self.config.max_retries)
        retry_delay = max(0, self.config.retry_delay)
        transient_retry_count = 0
        configured_retry_count = 0
        invocation_attempt = 0
        auto_tool_choice_fallback_used = False
        auto_tool_choice_warning = "WARNING: Tools are disabled due to server configuration error."
        self._log_run_event(
            state,
            "llm_invoke_start",
            run_id=None if state is None else state.get("run_id", ""),
            node=node_name,
            max_attempts=configured_max_attempts,
            transient_max_retries=len(_TRANSIENT_RETRY_DELAYS),
            context_messages=len(context),
        )

        while True:
            invocation_attempt += 1
            try:
                normalized_context = self._normalize_system_prefix_for_provider(context)
                response = await current_llm.ainvoke(normalized_context, **invocation_kwargs)
                invalid_calls = getattr(response, "invalid_tool_calls", None)
                # Thinking/redacted blocks are non-empty content, but not a visible answer.
                has_content = has_visible_text(visible_text_content(response.content))
                if not has_content and not response.tool_calls and not invalid_calls:
                    metadata = getattr(response, "response_metadata", {}) or {}
                    content_types = (
                        [
                            block.get("type", "unknown") if isinstance(block, dict) else "text"
                            for block in response.content
                        ]
                        if isinstance(response.content, list)
                        else ["text"]
                    )
                    stop_reason = metadata.get("stop_reason") or metadata.get("finish_reason")
                    model_name = metadata.get("model_name") or metadata.get("model")
                    logger.warning(
                        "Empty LLM response: no visible text or tool calls "
                        "(model=%s stop_reason=%s content_blocks=%s)",
                        model_name or "-",
                        stop_reason or "-",
                        ",".join(content_types) or "-",
                    )
                    self._log_run_event(
                        state,
                        "llm_invoke_empty_response",
                        run_id=None if state is None else state.get("run_id", ""),
                        node=node_name,
                        attempt=invocation_attempt,
                        stop_reason=stop_reason,
                        model=model_name,
                        content_types=content_types,
                    )
                    raise EmptyLLMResponseError("Empty response from LLM: no visible text or tool calls")
                self._log_run_event(
                    state,
                    "llm_invoke_success",
                    run_id=None if state is None else state.get("run_id", ""),
                    node=node_name,
                    attempt=invocation_attempt,
                    has_content=has_content,
                    tool_calls=len(getattr(response, "tool_calls", []) or []),
                )
                return response
            except Exception as e:
                err_str = str(e)
                if (
                    "auto" in err_str
                    and "tool choice" in err_str
                    and "requires" in err_str
                    and not auto_tool_choice_fallback_used
                ):
                    logger.warning(
                        "⚠ Server does not support 'auto' tool choice. Falling back to chat-only mode."
                    )
                    auto_tool_choice_fallback_used = True
                    current_llm = self.llm
                    context = list(context)
                    if context and isinstance(context[0], SystemMessage):
                        system_content = str(context[0].content)
                        if auto_tool_choice_warning not in system_content:
                            system_content = f"{system_content}\n\n{auto_tool_choice_warning}"
                        context[0] = SystemMessage(content=system_content)
                    continue

                is_transient = self._is_transient_llm_error(e)
                is_fatal = self._is_fatal_llm_error(e) and not is_transient
                error_details = describe_provider_error(e)
                logger.warning(
                    "LLM Error (Attempt %s): type=%s http_status=%s url_host=%s "
                    "request_error_id=%s message=%s",
                    invocation_attempt,
                    error_details["error_type"],
                    error_details["http_status"] or "-",
                    error_details["url_host"] or "-",
                    error_details["request_error_id"] or "-",
                    error_details["message"],
                )

                if is_transient and transient_retry_count < len(_TRANSIENT_RETRY_DELAYS):
                    transient_retry_count += 1
                    backoff_delay = _TRANSIENT_RETRY_DELAYS[transient_retry_count - 1]
                    self._log_run_event(
                        state,
                        "llm_retry",
                        node=node_name,
                        attempt=transient_retry_count,
                        max_attempts=len(_TRANSIENT_RETRY_DELAYS),
                        invocation_attempt=invocation_attempt,
                        retry_kind="transient",
                        backoff_seconds=backoff_delay,
                        fatal=False,
                        error_type=error_details["error_type"],
                        http_status=error_details["http_status"],
                        url_host=error_details["url_host"],
                        request_error_id=error_details["request_error_id"],
                        error=error_details["message"],
                    )
                    self._emit_provider_retry_status(node_name, transient_retry_count)
                    await asyncio.sleep(backoff_delay)
                    continue

                if is_fatal:
                    logger.error("Fatal LLM error detected. Aborting request: %s", e)
                    self._log_run_event(
                        state,
                        "llm_invoke_fatal",
                        run_id=None if state is None else state.get("run_id", ""),
                        node=node_name,
                        attempt=invocation_attempt,
                        retry_exhausted=error_details["retry_exhausted"],
                        rotation_error_kind=error_details["rotation_error_kind"],
                        rotation_keys_tried=error_details["rotation_keys_tried"],
                        rotation_pool_size=error_details["rotation_pool_size"],
                        http_status=error_details["http_status"],
                        error_type=error_details["error_type"],
                        error=error_details["message"],
                    )
                    raise

                is_permanent_client_error = (
                    self._is_permanent_client_error(e) or self._is_closed_client_error(e)
                )
                if is_transient or is_permanent_client_error or configured_retry_count >= configured_max_attempts - 1:
                    if is_transient:
                        try:
                            setattr(e, "llm_retry_exhausted", True)
                        except Exception:
                            pass
                    self._log_run_event(
                        state,
                        "llm_invoke_exhausted",
                        run_id=None if state is None else state.get("run_id", ""),
                        node=node_name,
                        attempt=invocation_attempt,
                        retryable=is_transient,
                        retry_exhausted=is_transient,
                        http_status=error_details["http_status"],
                        error_type=error_details["error_type"],
                        error=error_details["message"],
                    )
                    raise

                backoff_delay = retry_delay * (2 ** configured_retry_count) + random.uniform(0, retry_delay)
                configured_retry_count += 1
                self._log_run_event(
                    state,
                    "llm_retry",
                    node=node_name,
                    attempt=configured_retry_count,
                    max_attempts=max(0, configured_max_attempts - 1),
                    invocation_attempt=invocation_attempt,
                    retry_kind="configured",
                    backoff_seconds=backoff_delay,
                    fatal=False,
                    error=str(e),
                )
                await asyncio.sleep(backoff_delay)

    @staticmethod
    def _emit_provider_retry_status(node_name: str, retry_number: int) -> None:
        try:
            writer = get_stream_writer()
            # A transient failure aborts the current attempt after partial
            # content — including tool_use deltas that already rendered a
            # "preparing" tool card — has streamed. The retry regenerates the
            # whole turn from scratch with fresh tool-call ids, so signal the UI
            # to finalize the aborted attempt's orphan cards before the next
            # attempt streams, otherwise those cards hang as if still running.
            writer(
                {
                    "type": "provider_retry",
                    "node": node_name or "agent",
                    "retry_number": retry_number,
                    "max_attempts": len(_TRANSIENT_RETRY_DELAYS),
                }
            )
            writer(
                {
                    "type": "status_changed",
                    "label": f"Retrying provider request... {retry_number}/{len(_TRANSIENT_RETRY_DELAYS)}",
                    "node": node_name or "agent",
                }
            )
        except RuntimeError:
            # Direct unit invocations do not have a LangGraph streaming context.
            return

    @staticmethod
    def _is_closed_client_error(error: Exception) -> bool:
        return classify_provider_error(error)["kind"] == "closed_client"

    @staticmethod
    def _is_transient_llm_error(error: Exception) -> bool:
        return classify_provider_error(error)["retryable"]

    @staticmethod
    def _is_permanent_client_error(error: Exception) -> bool:
        return classify_provider_error(error)["kind"] in {"auth", "billing", "client_error"}

    def _is_fatal_llm_error(self, error: Exception) -> bool:
        if isinstance(error, ApiKeyRotationExhaustedError):
            return error.error_kind != "rate_limit"
        return classify_provider_error(error)["kind"] in {"auth", "billing"}
