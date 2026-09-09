from __future__ import annotations

import logging
from typing import List

from langchain_core.messages import RemoveMessage

from core.state import AgentState, OpenToolIssue, RecoveryState, transcript_message_delta
from core.summarize_policy import (
    choose_summary_boundary,
    estimate_context_tokens,
    estimate_summary_tokens,
    format_history_for_summary,
    should_summarize,
    truncate_summary_to_token_budget,
)
from core import constants
from core.message_utils import stringify_content
from core.text_utils import format_exception_friendly

logger = logging.getLogger("agent")


class SummarizeMixin:
    """Summarize node: compacts message history when it grows beyond the token threshold."""

    def _is_mid_run_compaction(self, state: AgentState) -> bool:
        """True when the summarize node runs inside an active turn (after tools/recovery),
        not on the turn entry edge. Mid-run compaction may cut at completed tool-round
        boundaries in addition to user-turn boundaries."""
        try:
            return int(state.get("steps") or 0) > 0
        except (TypeError, ValueError):
            return False

    def _effective_reserved_tokens(self, summary: str) -> int:
        """Memory is part of every outbound prompt, so it counts against the context budget."""
        try:
            base_reserved = max(0, int(getattr(self.config, "summary_reserved_tokens", 0) or 0))
        except (TypeError, ValueError):
            base_reserved = 0
        return base_reserved + estimate_summary_tokens(summary)

    def _memory_token_budget(self) -> int:
        """Token cap for compressed memory; 0 means the cap is disabled."""
        try:
            return max(0, int(getattr(self.config, "effective_summary_max_tokens", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _memory_word_budget(self) -> int:
        """Word target handed to the summarizer: mixed ru/en memory averages ~2 tokens per word."""
        budget = self._memory_token_budget()
        if budget <= 0:
            try:
                budget = max(0, int(getattr(self.config, "summary_threshold", 0) or 0)) // 4
            except (TypeError, ValueError):
                budget = 0
        return max(80, budget // 2)

    async def _fit_memory_to_budget(self, state: AgentState, summary: str) -> str:
        """Keep compressed memory inside its budget so it cannot squeeze out live history.

        Without a cap, memory grows with every compaction, inflates the reserved part of the
        context estimate, and eventually triggers compaction on every turn for a couple of
        messages. One model-driven fold handles the common case; the deterministic truncation
        keeps the guarantee when the model overshoots or the call fails.
        """
        budget = self._memory_token_budget()
        if budget <= 0:
            return summary
        before_tokens = estimate_summary_tokens(summary)
        if before_tokens <= budget:
            return summary

        folded = ""
        try:
            res = await self.llm.ainvoke(
                constants.SUMMARY_FOLD_PROMPT_TEMPLATE.format(
                    summary=summary,
                    max_words=self._memory_word_budget(),
                )
            )
            folded = stringify_content(getattr(res, "content", res)).strip()
        except Exception as exc:
            logger.warning(
                "🧹 Memory fold failed, truncating memory instead: %s", format_exception_friendly(exc)
            )

        candidate = folded if folded and estimate_summary_tokens(folded) < before_tokens else summary
        result = truncate_summary_to_token_budget(candidate, budget)
        after_tokens = estimate_summary_tokens(result)
        logger.info(
            "🧹 Memory folded: ~%s -> ~%s tokens (budget ~%s).", before_tokens, after_tokens, budget
        )
        self._log_run_event(
            state,
            "summary_memory_folded",
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            budget_tokens=budget,
            folded_by_model=candidate is not summary,
            truncated=result != candidate,
        )
        return result

    async def summarize_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")
        existing_transcript = state.get("transcript_messages")
        transcript_bootstrap = (
            {"transcript_messages": transcript_message_delta(messages)}
            if not isinstance(existing_transcript, list) or not existing_transcript
            else {}
        )
        current_turn_id = self._current_turn_id(state, messages)
        current_task = self._resolve_current_task(state, messages)
        open_tool_issue = self._get_active_open_tool_issue(state, messages, current_turn_id=current_turn_id)
        recovery_state = self._get_recovery_state(state, current_turn_id=current_turn_id)

        reserved_tokens = self._effective_reserved_tokens(summary)
        estimated_tokens = estimate_context_tokens(messages, reserved_tokens=reserved_tokens)
        node_timer = self._log_node_start(
            state,
            "summarize",
            message_count=len(messages),
            estimated_tokens=estimated_tokens,
            threshold=self.config.summary_threshold,
            keep_last=self.config.summary_keep_last,
            has_summary=bool(summary),
        )

        if not should_summarize(
            messages,
            threshold=self.config.summary_threshold,
            keep_last=self.config.summary_keep_last,
            has_summary=bool(summary),
            reserved_tokens=reserved_tokens,
            allow_tool_round_boundaries=self._is_mid_run_compaction(state),
        ):
            self._log_node_end(
                state,
                "summarize",
                node_timer,
                outcome="skipped",
                reason="below_threshold",
            )
            return transcript_bootstrap

        logger.debug(f"📊 Context size: ~{estimated_tokens} tokens. Summarizing...")

        # Determine cut-off point
        idx = choose_summary_boundary(
            messages,
            keep_last=self.config.summary_keep_last,
            threshold=self.config.summary_threshold,
            reserved_tokens=reserved_tokens,
            allow_tool_round_boundaries=self._is_mid_run_compaction(state),
        )

        to_summarize = messages[:idx]

        # SAFEGUARD: If the last N messages alone exceed the limit,
        # we cannot compress anything without losing recent context.
        if not to_summarize:
            logger.warning(
                f"⚠ Context (~{estimated_tokens} tokens) exceeds threshold, "
                "but cannot summarize further without deleting the most recent active messages. "
                "Expanding context dynamically for this turn."
            )
            self._log_node_end(
                state,
                "summarize",
                node_timer,
                outcome="skipped",
                reason="no_summarizable_messages",
            )
            return transcript_bootstrap

        history_text = self._format_history_for_summary(to_summarize)
        state_snapshot = self._build_summary_state_snapshot(
            current_task=current_task,
            open_tool_issue=open_tool_issue,
            recovery_state=recovery_state,
        )

        prompt = constants.SUMMARY_PROMPT_TEMPLATE.format(
            summary=summary,
            state_snapshot=state_snapshot,
            history_text=history_text,
            max_words=self._memory_word_budget(),
        )

        try:
            res = await self.llm.ainvoke(prompt)

            updated_summary = stringify_content(getattr(res, "content", res)).strip()
            if not updated_summary:
                logger.warning(
                    "🧹 Summarization returned empty memory. Keeping full history to avoid context loss."
                )
                self._log_node_end(
                    state,
                    "summarize",
                    node_timer,
                    outcome="skipped",
                    reason="empty_summary",
                    estimated_tokens=estimated_tokens,
                )
                return transcript_bootstrap

            delete_msgs = [RemoveMessage(id=m.id) for m in to_summarize if m.id]
            updated_summary = await self._fit_memory_to_budget(state, updated_summary)
            logger.info(f"🧹 Summary: Removed {len(delete_msgs)} messages. Generated new summary.")
            self._log_run_event(
                state,
                "summary_compacted",
                estimated_tokens=estimated_tokens,
                removed_messages=len(delete_msgs),
                summarized_messages=len(to_summarize),
                retained_messages=len(messages) - len(to_summarize),
                memory_tokens=estimate_summary_tokens(updated_summary),
            )
            self._log_node_end(
                state,
                "summarize",
                node_timer,
                outcome="compacted",
                removed_messages=len(delete_msgs),
                summarized_messages=len(to_summarize),
                retained_messages=len(messages) - len(to_summarize),
                memory_tokens=estimate_summary_tokens(updated_summary),
            )

            return {
                **transcript_bootstrap,
                "summary": updated_summary,
                "messages": delete_msgs,
            }
        except Exception as e:
            err_str = str(e)
            if "content_filter" in err_str or "Moderation Block" in err_str:
                logger.warning(
                    "🧹 Summarization skipped due to Content Filter (False Positive). Continuing with full history."
                )
            else:
                logger.error(f"Summarization Error: {format_exception_friendly(e)}")
            self._log_node_error(
                state,
                "summarize",
                node_timer,
                e,
                outcome="failed",
                estimated_tokens=estimated_tokens,
            )
            return transcript_bootstrap

    def _format_history_for_summary(self, messages: List) -> str:
        return format_history_for_summary(messages, is_internal_retry=self._is_internal_retry_message)

    def _build_summary_state_snapshot(
        self,
        *,
        current_task: str,
        open_tool_issue: OpenToolIssue | None,
        recovery_state: RecoveryState | None,
    ) -> str:
        parts: List[str] = []
        if str(current_task or "").strip():
            parts.append(f"current_task: {str(current_task).strip()}")
        if isinstance(open_tool_issue, dict):
            issue_summary = str(open_tool_issue.get("summary") or "").strip()
            if issue_summary:
                parts.append(f"open_tool_issue: {issue_summary}")
        if isinstance(recovery_state, dict):
            strategy = recovery_state.get("active_strategy")
            if isinstance(strategy, dict):
                strategy_kind = str(strategy.get("strategy_kind") or strategy.get("strategy") or "").strip()
                if strategy_kind:
                    parts.append(f"recovery_strategy: {strategy_kind}")
            blocker = recovery_state.get("external_blocker")
            if isinstance(blocker, dict):
                blocker_reason = str(blocker.get("reason") or blocker.get("issue_summary") or "").strip()
                if blocker_reason:
                    parts.append(f"external_blocker: {blocker_reason}")
        return "\n".join(parts) if parts else "none"
