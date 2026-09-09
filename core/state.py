from typing import TypedDict, Annotated, List, NotRequired, Dict, Any
from langchain_core.messages import BaseMessage, RemoveMessage
from langgraph.graph.message import add_messages
from core.turn_outcomes import TurnOutcome


def append_transcript_messages(
    left: List[BaseMessage] | None,
    right: List[BaseMessage] | None,
) -> List[BaseMessage]:
    """Merge the durable UI transcript without applying context compaction.

    ``messages`` is intentionally allowed to contain ``RemoveMessage`` updates for
    the model context.  The transcript channel is different: removals are ignored
    and message IDs are used only to make repeated checkpoint/UI updates idempotent.
    This follows the same identity semantics as LangGraph's ``add_messages`` while
    keeping the full user-visible history append-only.
    """
    current = [message for message in (left or []) if not isinstance(message, RemoveMessage)]
    incoming = [message for message in (right or []) if not isinstance(message, RemoveMessage)]
    if not incoming:
        return list(current)
    return add_messages(current, incoming)


def transcript_message_delta(messages: List[BaseMessage] | None) -> List[BaseMessage]:
    """Return only real messages suitable for the durable transcript channel."""
    return [message for message in (messages or []) if not isinstance(message, RemoveMessage)]


class OpenToolIssue(TypedDict, total=False):
    turn_id: int
    kind: str
    summary: str
    tool_names: List[str]
    tool_args: Dict[str, Any]
    source: str
    error_type: str
    fingerprint: str
    progress_fingerprint: str
    details: Dict[str, Any]


class RecoveryStrategy(TypedDict, total=False):
    id: str
    strategy: str
    strategy_kind: str
    reason: str
    tool_name: str
    suggested_tool_name: str
    patched_args: Dict[str, Any]
    notes: str
    llm_guidance: str
    current_task: str
    issue_summary: str
    issue_details: Dict[str, Any]
    progress_fingerprint: str


class RecoveryBlocker(TypedDict, total=False):
    reason: str
    issue_summary: str


class RecoveryState(TypedDict, total=False):
    turn_id: int
    retry_count: int
    retry_fingerprint_history: List[str]
    last_reason: str
    active_issue: OpenToolIssue | None
    active_strategy: RecoveryStrategy | None
    strategy_queue: List[RecoveryStrategy]
    attempts_by_strategy: Dict[str, int]
    progress_markers: List[str]
    last_successful_evidence: str
    external_blocker: RecoveryBlocker | None
    llm_replan_attempted_for: List[str]


class RecoveryPlanResult(TypedDict):
    turn_id: int
    turn_outcome: TurnOutcome
    current_task: str
    recovery_state: RecoveryState
    open_tool_issue: OpenToolIssue | None
    handoff_message: str
    completion_reason: str
    drop_trailing_tool_call: bool
    had_pending_tool_calls: bool
    loop_budget_reached: bool
    successful_tool_repeat_count: int
    successful_tool_name: str


class AgentState(TypedDict):
    """State split into model context and durable user-visible transcript."""

    # Model-facing message history. Summarization may remove entries from here.
    messages: Annotated[List[BaseMessage], add_messages]

    # Full append-only history for UI/session restoration. Never compacted.
    transcript_messages: NotRequired[Annotated[List[BaseMessage], append_transcript_messages]]

    # Compressed memory
    summary: NotRequired[str]

    # Step counter
    steps: int

    # Token usage tracking (Last step usage)
    token_usage: Dict[str, Any]

    # Original user task for the current request
    current_task: NotRequired[str]
    requires_evidence: NotRequired[bool]
    safety_mode: NotRequired[str]

    # Internal workflow state
    turn_outcome: NotRequired[TurnOutcome]
    recovery_state: NotRequired[RecoveryState]

    # Deprecated checkpoint compatibility channels. Runtime nodes only read
    # these when migrating a checkpoint that predates nested recovery fields;
    # all new updates are written exclusively to recovery_state.
    self_correction_retry_count: NotRequired[int]
    self_correction_retry_turn_id: NotRequired[int]
    self_correction_fingerprint_history: NotRequired[List[str]]
    self_correction_last_reason: NotRequired[str]

    # Durable runtime/session info
    session_id: NotRequired[str]
    run_id: NotRequired[str]
    turn_id: NotRequired[int]
    pending_approval: NotRequired[Dict[str, Any] | None]
    open_tool_issue: NotRequired[OpenToolIssue | None]
    last_tool_error: NotRequired[str]
    last_tool_result: NotRequired[str]
