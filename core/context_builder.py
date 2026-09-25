from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from core import constants
from core.config import AgentConfig
from core.message_utils import stringify_content
from core.multimodal import (
    IMAGE_INPUT_PROFILE_KEYS,
    human_message_has_image_content,
    materialize_user_message_content_for_model,
    normalize_model_capabilities,
    strip_image_content_from_message_content,
)
from core.tool_args import canonicalize_tool_args
from core.runtime_prompt_policy import RuntimePromptContext, RuntimePromptPolicyBuilder
from core.state import AgentState, OpenToolIssue, RecoveryState

logger = logging.getLogger("agent")

_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY = "__gemini_function_call_thought_signatures__"

IsInternalRetry = Callable[[BaseMessage], bool]
PromptLoader = Callable[[], str]
RunLogger = Callable[..., None]
RecoveryMessageBuilder = Callable[[RecoveryState | None], SystemMessage | None]


class ContextBuilder:
    def __init__(
        self,
        *,
        config: AgentConfig,
        model_capabilities: Any = None,
        prompt_loader: PromptLoader,
        is_internal_retry: IsInternalRetry,
        log_run_event: RunLogger,
        recovery_message_builder: RecoveryMessageBuilder,
        provider_safe_tool_call_id_re: re.Pattern[str],
    ) -> None:
        self.config = config
        self._model_capabilities = normalize_model_capabilities(model_capabilities)
        capability_payload = model_capabilities if isinstance(model_capabilities, dict) else {}
        self._image_capability_known = any(
            key in capability_payload
            for key in ("image_input_supported", "supports_image_input", *IMAGE_INPUT_PROFILE_KEYS)
        )
        self._prompt_loader = prompt_loader
        self._is_internal_retry = is_internal_retry
        self._log_run_event = log_run_event
        self._recovery_message_builder = recovery_message_builder
        self._provider_safe_tool_call_id_re = provider_safe_tool_call_id_re
        self._runtime_policy_builder = RuntimePromptPolicyBuilder(config=config)

    def build(
        self,
        messages: List[BaseMessage],
        state: AgentState | None,
        *,
        summary: str,
        current_task: str,
        tools_available: bool,
        active_tool_names: List[str],
        open_tool_issue: OpenToolIssue | None,
        recovery_state: RecoveryState | None,
        user_choice_locked: bool = False,
    ) -> List[BaseMessage]:
        sanitized_messages = self.sanitize_messages(messages, state=state)
        full_context: List[BaseMessage] = [
            self._build_base_system_message()
        ]
        # Memory goes early: it's context ballast, must not override operational rules
        if summary:
            full_context.append(SystemMessage(content=f"<memory>\n{summary}\n</memory>"))
        # The latest user message already carries the full task. Keep the hint only
        # for continuation/recovery contexts where it adds information.
        task_hint = current_task
        for message in reversed(sanitized_messages):
            if isinstance(message, HumanMessage) and not self._is_internal_retry(message):
                if stringify_content(message.content).strip() == str(current_task or "").strip():
                    task_hint = ""
                break
        inferred_user_choice_locked = user_choice_locked or self._has_request_user_input_tool_result(sanitized_messages)
        full_context.extend(
            self._runtime_policy_builder.build_messages(
                RuntimePromptContext(
                    current_task=task_hint,
                    tools_available=tools_available,
                    active_tool_names=tuple(active_tool_names),
                    user_choice_locked=inferred_user_choice_locked,
                )
            )
        )
        safety_overlay = self._build_safety_overlay(tools_available=tools_available)
        if safety_overlay:
            full_context.append(SystemMessage(content=safety_overlay))
        issue_message = self._build_tool_issue_system_message(open_tool_issue)
        if issue_message:
            full_context.append(issue_message)
        recovery_message = self._recovery_message_builder(recovery_state)
        if recovery_message:
            full_context.append(recovery_message)
        full_context.extend(sanitized_messages)
        return self.normalize_system_prefix(full_context)

    def _has_request_user_input_tool_result(self, messages: List[BaseMessage]) -> bool:
        for message in messages:
            if isinstance(message, ToolMessage) and str(message.name or "").strip().lower() == "request_user_input":
                return True
        return False

    def _uses_openai_responses_api(self) -> bool:
        return self.config.provider == "openai" and getattr(self.config, "llm_api_mode", "chat") == "responses"

    @staticmethod
    def _iter_message_provider_ids(message: BaseMessage):
        for tool_call in getattr(message, "tool_calls", None) or []:
            if isinstance(tool_call, dict):
                yield str(tool_call.get("id") or "")
        content = getattr(message, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    yield str(block.get("id") or "")

    def _message_is_responses_native(self, message: BaseMessage) -> bool:
        """True when an assistant message originated from an OpenAI Responses call.

        Only such messages carry item/call IDs the Responses API accepts when the
        history is replayed. Messages produced by other providers (e.g.
        Anthropic/Bedrock ``toolu_...`` IDs) must be flattened and their IDs
        remapped; otherwise the Responses endpoint rejects them with
        ``Invalid 'input[N].id'`` after the active model is switched mid-session.
        """
        metadata = getattr(message, "response_metadata", None)
        provider = ""
        if isinstance(metadata, dict):
            provider = str(metadata.get("model_provider") or "").strip().lower()
        if provider:
            return provider == "openai"
        # No provider tag: infer origin from content-block / tool-call ID shapes.
        for candidate in self._iter_message_provider_ids(message):
            if candidate.startswith("toolu_"):
                return False
        return True

    def sanitize_messages(
        self,
        messages: List[BaseMessage],
        *,
        state: AgentState | None = None,
    ) -> List[BaseMessage]:
        # Provider-specific normalization is applied only to the outbound model context.
        # The persisted graph state keeps original message identities for execution/UI flows.
        sanitized: List[BaseMessage] = []
        tool_call_id_map: Dict[str, str] = {}
        used_tool_call_ids: set[str] = set()
        remapped_count = 0
        normalized_content_count = 0
        filtered_image_message_count = 0
        filtered_image_block_count = 0
        stripped_reasoning_block_count = 0
        stripped_reasoning_kwarg_count = 0
        reordered_responses_tool_turn_count = 0
        image_input_supported = bool(self._model_capabilities.get("image_input_supported"))

        for message in messages:
            normalized_message: BaseMessage = message

            if isinstance(message, (AIMessage, AIMessageChunk)):
                responses_verbatim = self._uses_openai_responses_api() and self._message_is_responses_native(message)
                raw_tool_calls = list(getattr(message, "tool_calls", []) or [])
                if raw_tool_calls:
                    normalized_tool_calls: List[Dict[str, Any]] = []
                    tool_calls_changed = False
                    for tool_call in raw_tool_calls:
                        if not isinstance(tool_call, dict):
                            normalized_tool_calls.append(tool_call)
                            continue
                        cloned_call = dict(tool_call)
                        raw_id = str(cloned_call.get("id") or "").strip()
                        if raw_id:
                            mapped_id = tool_call_id_map.get(raw_id)
                            if not mapped_id:
                                mapped_id = self._normalize_tool_call_id_for_provider(
                                    raw_id,
                                    used_ids=used_tool_call_ids,
                                    responses_verbatim=responses_verbatim,
                                )
                                tool_call_id_map[raw_id] = mapped_id
                            if mapped_id != raw_id:
                                cloned_call["id"] = mapped_id
                                tool_calls_changed = True
                                remapped_count += 1
                        normalized_tool_calls.append(cloned_call)
                    if tool_calls_changed:
                        normalized_message = message.model_copy(update={"tool_calls": normalized_tool_calls})

                if isinstance(normalized_message, (AIMessage, AIMessageChunk)):
                    metadata = dict(getattr(normalized_message, "additional_kwargs", {}) or {})
                    signature_map = metadata.get(_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY)
                    if isinstance(signature_map, dict) and signature_map:
                        remapped_signature_map: Dict[str, Any] = {}
                        signature_map_changed = False
                        for raw_id, signature in signature_map.items():
                            normalized_raw_id = str(raw_id or "").strip()
                            if not normalized_raw_id:
                                continue
                            mapped_id = tool_call_id_map.get(normalized_raw_id)
                            if not mapped_id:
                                mapped_id = self._normalize_tool_call_id_for_provider(
                                    normalized_raw_id,
                                    used_ids=used_tool_call_ids,
                                    responses_verbatim=responses_verbatim,
                                )
                                tool_call_id_map[normalized_raw_id] = mapped_id
                            remapped_signature_map[mapped_id] = signature
                            if mapped_id != normalized_raw_id:
                                signature_map_changed = True
                        if signature_map_changed:
                            metadata[_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY] = remapped_signature_map
                            normalized_message = normalized_message.model_copy(update={"additional_kwargs": metadata})

                    normalized_message, block_count, kwarg_count = self._strip_cross_provider_reasoning(
                        normalized_message
                    )
                    stripped_reasoning_block_count += block_count
                    stripped_reasoning_kwarg_count += kwarg_count

                    normalized_message, reordered = self._reorder_trailing_responses_function_calls(
                        normalized_message
                    )
                    if reordered:
                        reordered_responses_tool_turn_count += 1

            if isinstance(normalized_message, ToolMessage):
                raw_tool_id = str(normalized_message.tool_call_id or "").strip()
                if raw_tool_id:
                    mapped_tool_id = tool_call_id_map.get(raw_tool_id)
                    if not mapped_tool_id:
                        mapped_tool_id = self._normalize_tool_call_id_for_provider(
                            raw_tool_id,
                            used_ids=used_tool_call_ids,
                        )
                        tool_call_id_map[raw_tool_id] = mapped_tool_id
                    if mapped_tool_id != raw_tool_id:
                        normalized_message = normalized_message.model_copy(update={"tool_call_id": mapped_tool_id})
                        remapped_count += 1

            if isinstance(normalized_message, HumanMessage) and human_message_has_image_content(normalized_message.content):
                if self._image_capability_known and not image_input_supported:
                    sanitized_content, removed_blocks = strip_image_content_from_message_content(
                        normalized_message.content
                    )
                    if removed_blocks:
                        normalized_message = normalized_message.model_copy(update={"content": sanitized_content})
                        filtered_image_message_count += 1
                        filtered_image_block_count += removed_blocks
                if human_message_has_image_content(normalized_message.content):
                    materialized_content = materialize_user_message_content_for_model(
                        normalized_message.content,
                        provider=self.config.provider,
                    )
                    if materialized_content != normalized_message.content:
                        normalized_message = normalized_message.model_copy(update={"content": materialized_content})

            # Responses assistant blocks carry reasoning, item IDs and tool calls;
            # flattening them loses the protocol state required by the next request.
            preserve_responses_content = (
                self._uses_openai_responses_api()
                and isinstance(normalized_message, AIMessage)
                and self._message_is_responses_native(normalized_message)
            )
            if self.config.provider == "openai" and not preserve_responses_content and not (
                isinstance(normalized_message, HumanMessage) and human_message_has_image_content(normalized_message.content)
            ):
                raw_content = getattr(normalized_message, "content", None)
                normalized_content = stringify_content(raw_content)
                if normalized_content != raw_content:
                    normalized_message = normalized_message.model_copy(update={"content": normalized_content})
                    normalized_content_count += 1

            if isinstance(normalized_message, HumanMessage):
                content = stringify_content(normalized_message.content).strip()
                if content == constants.REFLECTION_PROMPT:
                    continue
            sanitized.append(normalized_message)

        if remapped_count:
            self._log_run_event(
                state,
                "provider_tool_call_id_remap",
                run_id=None if state is None else state.get("run_id", ""),
                remapped_count=remapped_count,
                distinct_ids=len(tool_call_id_map),
            )
        if normalized_content_count:
            self._log_run_event(
                state,
                "provider_content_normalized",
                run_id=None if state is None else state.get("run_id", ""),
                provider=self.config.provider,
                normalized_count=normalized_content_count,
            )
        if filtered_image_block_count:
            self._log_run_event(
                state,
                "provider_unsupported_image_history_filtered",
                run_id=None if state is None else state.get("run_id", ""),
                provider=self.config.provider,
                filtered_message_count=filtered_image_message_count,
                filtered_image_block_count=filtered_image_block_count,
            )
        if stripped_reasoning_block_count or stripped_reasoning_kwarg_count:
            self._log_run_event(
                state,
                "provider_cross_provider_reasoning_stripped",
                run_id=None if state is None else state.get("run_id", ""),
                provider=self.config.provider,
                stripped_content_block_count=stripped_reasoning_block_count,
                stripped_additional_kwargs_count=stripped_reasoning_kwarg_count,
            )
        if reordered_responses_tool_turn_count:
            self._log_run_event(
                state,
                "provider_responses_function_call_reordered",
                run_id=None if state is None else state.get("run_id", ""),
                provider=self.config.provider,
                reordered_tool_turn_count=reordered_responses_tool_turn_count,
            )
        return sanitized

    def _reorder_trailing_responses_function_calls(
        self,
        message: AIMessage | AIMessageChunk,
    ) -> tuple[AIMessage | AIMessageChunk, bool]:
        """Move ``function_call`` content blocks to the end of an assistant turn.

        OpenAI Responses replay requires every ``function_call`` item to be
        immediately followed by its matching ``function_call_output``. When an
        assistant turn carries text/message blocks *after* a ``function_call``
        (e.g. a provider that emits a final answer together with the tool call),
        the replayed ``input`` interposes an assistant ``message`` item between
        the ``function_call`` and the ``function_call_output`` that follows in the
        next ``ToolMessage``. Strict backends (e.g. AgentRouter/DeepSeek) reject
        this with ``No tool output found for tool call ...``; lenient ones accept
        it, so the malformed shape survives cross-provider replay.

        Stable-partition the content so all non-``function_call`` blocks keep
        their order first, followed by the ``function_call`` blocks in their
        original order. Only responses-native assistant messages replayed to a
        Responses endpoint are affected, and only when a block actually trails a
        ``function_call``.
        """
        if not (self._uses_openai_responses_api() and self._message_is_responses_native(message)):
            return message, False

        content = getattr(message, "content", None)
        if not isinstance(content, list) or len(content) < 2:
            return message, False

        seen_function_call = False
        needs_reorder = False
        for block in content:
            is_function_call = isinstance(block, dict) and block.get("type") == "function_call"
            if is_function_call:
                seen_function_call = True
            elif seen_function_call:
                # A non-function_call block trails a function_call block.
                needs_reorder = True
                break

        if not needs_reorder:
            return message, False

        non_function_calls = [
            block
            for block in content
            if not (isinstance(block, dict) and block.get("type") == "function_call")
        ]
        function_calls = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "function_call"
        ]
        reordered_content = [*non_function_calls, *function_calls]
        return message.model_copy(update={"content": reordered_content}), True

    def _strip_cross_provider_reasoning(
        self,
        message: AIMessage | AIMessageChunk,
    ) -> tuple[AIMessage | AIMessageChunk, int, int]:
        """Remove reasoning content blocks that are incompatible with the target provider.

        OpenAI Responses API (``output_version="responses/v1"``) emits reasoning
        blocks with ``type: "reasoning"`` and a ``summary`` list — **not** the
        ``reasoning`` string key that langchain-google-genai expects.  When the
        agent switches from OpenAI to Gemini mid-session, these blocks remain in
        the persisted message history and cause a ``KeyError`` inside
        ``langchain_google_genai.chat_models`` (``part["reasoning"]`` /
        ``content_block["reasoning"]``).

        Responses reasoning (including opaque ``encrypted_content``) is required
        when replaying tool turns to a Responses endpoint. Preserve both modern
        content blocks and the legacy ``additional_kwargs['reasoning']`` form in
        that mode; remove them for other target APIs.

        Returns ``(message, stripped_block_count, stripped_kwarg_count)``.
        """
        if self._uses_openai_responses_api():
            return message, 0, 0

        block_count = 0
        kwarg_count = 0

        content = getattr(message, "content", None)
        if isinstance(content, list):
            new_content: list = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "reasoning":
                        # OpenAI Responses API reasoning blocks have ``summary`` (list)
                        # but no ``reasoning`` string key.  Gemini's native reasoning
                        # blocks have ``reasoning`` (string) and ``extras.signature``
                        # — those must be preserved for multi-turn tool-calling.
                        if "reasoning" not in block:
                            block_count += 1
                            continue
                    elif block_type == "thinking":
                        # Anthropic thinking blocks carry ``thinking`` (text) and
                        # ``signature`` for multi-turn continuity.  Preserve them
                        # so extended thinking round-trips across turns.  When the
                        # API returns ``display="omitted"`` the block arrives as
                        # ``redacted_thinking`` with ``data`` instead — also keep.
                        continue
                    elif block_type == "redacted_thinking":
                        # Anthropic redacted thinking blocks contain ``data`` and
                        # must be preserved verbatim for multi-turn continuity.
                        continue
                new_content.append(block)
            if block_count:
                message = message.model_copy(update={"content": new_content})

        additional_kwargs = getattr(message, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict) and "reasoning" in additional_kwargs:
            additional_kwargs = dict(additional_kwargs)
            del additional_kwargs["reasoning"]
            kwarg_count = 1
            message = message.model_copy(update={"additional_kwargs": additional_kwargs})

        return message, block_count, kwarg_count

    def detect_tool_history_mismatch(
        self,
        messages: List[BaseMessage],
    ) -> Dict[str, Any] | None:
        pending_calls: Dict[str, Dict[str, Any]] = {}
        pending_order: List[str] = []
        duplicate_ids: List[str] = []
        orphan_tool_results: List[str] = []
        pending_interleaving: List[str] = []

        for message in messages:
            if isinstance(message, HumanMessage):
                if self._is_internal_retry(message):
                    continue
                if pending_calls:
                    pending_interleaving.append("human_before_tool_result")
                continue

            if isinstance(message, ToolMessage):
                tool_call_id = str(message.tool_call_id or "").strip()
                if not tool_call_id:
                    orphan_tool_results.append("<missing>")
                    continue
                if tool_call_id in pending_calls:
                    pending_calls.pop(tool_call_id, None)
                    pending_order = [item for item in pending_order if item != tool_call_id]
                else:
                    orphan_tool_results.append(tool_call_id)
                continue

            if not isinstance(message, (AIMessage, AIMessageChunk)):
                continue

            tool_calls = list(getattr(message, "tool_calls", []) or [])
            if not tool_calls:
                if pending_calls:
                    pending_interleaving.append("assistant_before_tool_result")
                continue

            if pending_calls:
                pending_interleaving.append("tool_call_before_previous_tool_result")

            for tool_call in tool_calls:
                tool_call_id = str(tool_call.get("id") or "").strip()
                tool_name = str(tool_call.get("name") or "").strip()
                tool_args = canonicalize_tool_args(tool_call.get("args"))
                if not tool_call_id or not tool_name:
                    continue
                if tool_call_id in pending_calls:
                    duplicate_ids.append(tool_call_id)
                pending_calls[tool_call_id] = {
                    "id": tool_call_id,
                    "name": tool_name,
                    "args": tool_args,
                }
                if tool_call_id not in pending_order:
                    pending_order.append(tool_call_id)

        unresolved_calls = [pending_calls[tool_call_id] for tool_call_id in pending_order if tool_call_id in pending_calls]
        if not unresolved_calls and not duplicate_ids and not orphan_tool_results and not pending_interleaving:
            return None

        return {
            "pending_tool_calls": unresolved_calls,
            "duplicate_tool_call_ids": duplicate_ids,
            "orphan_tool_results": orphan_tool_results,
            "interleaving_markers": pending_interleaving,
        }

    def normalize_system_prefix(self, context: List[BaseMessage]) -> List[BaseMessage]:
        system_messages: List[BaseMessage] = []
        non_system_messages: List[BaseMessage] = []
        for message in context:
            if self.message_is_provider_system(message):
                system_messages.append(message)
            else:
                non_system_messages.append(message)
        if self.config.provider in ("openai", "anthropic") and len(system_messages) > 1:
            merged_system_content = "\n\n".join(
                stringify_content(message.content).strip()
                for message in system_messages
                if stringify_content(message.content).strip()
            ).strip()
            if merged_system_content:
                return [SystemMessage(content=merged_system_content), *non_system_messages]
            return non_system_messages
        return [*system_messages, *non_system_messages]

    def assert_provider_safe_context(
        self,
        context: List[BaseMessage],
        *,
        state: AgentState | None = None,
    ) -> None:
        seen_non_system = False
        system_after_non_system = False
        for message in context:
            if self.message_is_provider_system(message):
                if seen_non_system:
                    system_after_non_system = True
                    break
                continue
            seen_non_system = True

        last_visible = self.get_last_model_visible_message(context)
        valid = isinstance(last_visible, (AIMessage, HumanMessage, ToolMessage)) and not system_after_non_system
        if valid:
            return

        self._log_run_event(
            state,
            "provider_context_invalid",
            run_id=None if state is None else state.get("run_id", ""),
            valid=False,
            last_visible_type=type(last_visible).__name__ if last_visible else "",
            system_after_non_system=system_after_non_system,
        )

        raise RuntimeError(
            "Provider-unsafe agent context: system messages must form a prefix and the last model-visible message must be AIMessage, HumanMessage, or ToolMessage."
        )

    def get_last_model_visible_message(self, context: List[BaseMessage]) -> BaseMessage | None:
        for message in reversed(context):
            if self.message_is_provider_system(message):
                continue
            return message
        return None

    def message_is_provider_system(self, message: BaseMessage) -> bool:
        return self._message_role_for_provider(message) in {"system", "developer"}

    def _message_role_for_provider(self, message: BaseMessage) -> str:
        role = ""
        if isinstance(message, SystemMessage):
            return "system"
        raw_role = getattr(message, "role", "")
        if isinstance(raw_role, str):
            role = raw_role.strip().lower()
        if not role:
            raw_type = getattr(message, "type", "")
            if isinstance(raw_type, str):
                role = raw_type.strip().lower()
        return role

    def _normalize_tool_call_id_for_provider(
        self,
        raw_id: str,
        *,
        used_ids: set[str],
        responses_verbatim: bool = False,
    ) -> str:
        normalized = str(raw_id or "").strip()
        # Responses content blocks retain the provider's call_id. Remapping only
        # tool_calls / ToolMessage would create duplicate or unmatched calls.
        # Only preserve IDs from messages that actually originated from a
        # Responses call; foreign IDs (e.g. Anthropic/Bedrock ``toolu_...``) left
        # verbatim are rejected by the Responses API as ``Invalid 'input[N].id'``.
        if responses_verbatim and normalized and normalized not in used_ids:
            used_ids.add(normalized)
            return normalized
        # Anthropic tool_use IDs are provider-issued opaque values (``toolu_…``).
        # They must remain identical to the ID in the assistant content block so
        # the following ToolMessage serializes as its matching tool_result.
        if self.config.provider == "anthropic" and normalized.startswith("toolu_") and normalized not in used_ids:
            used_ids.add(normalized)
            return normalized
        if self._provider_safe_tool_call_id_re.match(normalized) and normalized not in used_ids:
            used_ids.add(normalized)
            return normalized

        seed = normalized or "tool_call"
        suffix = 0
        while True:
            hash_source = seed if suffix == 0 else f"{seed}:{suffix}"
            candidate = hashlib.sha1(hash_source.encode("utf-8")).hexdigest()[:9]
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate
            suffix += 1

    def _build_safety_overlay(self, *, tools_available: bool) -> str:
        if not tools_available:
            return ""
        overlay_lines: List[str] = []
        overlay_lines.append(
            "SAFETY POLICY: Any write, delete, move, or process-launch working directory must stay inside the active workspace."
        )
        if self.config.enable_shell_tool:
            overlay_lines.append(
                "SAFETY POLICY: Shell execution is high risk. Prefer safer project-local tools whenever possible."
            )
        return "\n".join(overlay_lines).strip()

    def _build_tool_issue_system_message(self, open_tool_issue: OpenToolIssue | None) -> SystemMessage | None:
        if not open_tool_issue:
            return None

        issue_summary = str(open_tool_issue.get("summary", "")).strip()
        if open_tool_issue.get("kind") == "approval_denied":
            return SystemMessage(
                content=(
                    "TOOL EXECUTION DENIED BY USER:\n"
                    f"{issue_summary}\n\n"
                    "The user explicitly rejected this tool call. "
                    "Do not simulate the denied tool or describe imaginary results. "
                    "Do not make any more tool calls in this turn. "
                    "Reply briefly: say that you did not do it because the user chose No, then wait for the next instruction."
                )
            )

        return SystemMessage(
            content=constants.UNRESOLVED_TOOL_ERROR_PROMPT_TEMPLATE.format(
                error_summary=issue_summary
            )
        )

    def _build_base_system_message(self) -> SystemMessage:
        raw_prompt = self._prompt_loader()

        prompt = raw_prompt.replace("{{current_date}}", datetime.now().strftime("%Y-%m-%d"))
        prompt = prompt.replace("{{cwd}}", str(Path.cwd()))

        return SystemMessage(content=prompt)
