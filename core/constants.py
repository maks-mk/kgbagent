import sys
from pathlib import Path

# Agent version (single source of truth)
AGENT_VERSION = "v0.67.73.247b"

# Determine the project root directory
if getattr(sys, 'frozen', False):
    # When running as an executable, use the executable directory for config files,
    # but keep the current working directory (cwd) as the process launch location.
    BASE_DIR = Path(sys.executable).parent
else:
    # core/constants.py -> core/ -> root/
    BASE_DIR = Path(__file__).resolve().parent.parent

# --- PROMPTS ---

SUMMARY_PROMPT_TEMPLATE = (
    "Current memory:\n<previous_context>\n{summary}\n</previous_context>\n\n"
    "Operational state:\n{state_snapshot}\n\n"
    "New events (deleted from history after this update; anything you omit is lost):\n{history_text}\n\n"
    "Conversation-history summarization mode: update memory for the main model; do not continue or answer the task. Rules:\n"
    "- Merge new information into the existing memory without duplication.\n"
    "- Prefer newer evidence when facts conflict; preserve uncertainty when unresolved.\n"
    "- Preserve the active task, progress, blockers, pending decisions, next steps, and tool/recovery state.\n"
    "- Keep exact paths, commands, identifiers, outcomes, errors, decisions, and task status when relevant.\n"
    "- Remove stale or completed details unless needed for continuity or recovery.\n"
    "- Omit greetings, filler, reasoning, and raw tool output already captured by its outcome.\n"
    "- Do not infer or invent facts.\n"
    "- Keep the whole memory under {max_words} words; merge or drop the least valuable items to stay within it.\n"
    "- Write concise, self-contained bullet points, ordered from most to least important for continuing the task.\n"
    "- Return only the updated memory."
)

SUMMARY_FOLD_PROMPT_TEMPLATE = (
    "Current memory:\n<previous_context>\n{summary}\n</previous_context>\n\n"
    "Memory compaction mode: rewrite the memory above to fit under {max_words} words; "
    "do not continue or answer the task. Rules:\n"
    "- Keep the active task, progress, blockers, pending decisions, next steps, and tool/recovery state.\n"
    "- Keep exact paths, commands, identifiers, and errors that recovery still depends on.\n"
    "- Merge duplicates; drop stale, completed, or low-value details first.\n"
    "- Do not infer or invent facts.\n"
    "- Write concise, self-contained bullet points, ordered from most to least important for continuing the task.\n"
    "- Return only the compacted memory."
)

REFLECTION_PROMPT = (
    "SYSTEM HINT: The previous tool execution failed. "
    "Fix arguments or choose a different tool, then continue immediately. "
    "Do not claim success until a tool result confirms it."
)

UNRESOLVED_TOOL_ERROR_PROMPT_TEMPLATE = (
    "UNRESOLVED TOOL FAILURE:\n"
    "{error_summary}\n\n"
    "Do not claim success, completion, or verified characteristics unless you have actually resolved this "
    "with later successful tool results. Either retry with corrected arguments, use another tool, "
    "or clearly explain the blocker to the user."
)

TOOL_ISSUE_NOT_FOUND_TEXT = "Unable to continue: the active tool issue could not be found."

TOOL_ISSUE_APPROVAL_DENIED_TEXT = (
    "Action not completed: you declined an irreversible operation.\n"
    "The next attempt requires a new request or explicit confirmation."
)

TOOL_ISSUE_WORKSPACE_BOUNDARY_TEMPLATE = (
    "Unable to continue: the request goes outside the workspace boundary{tool_hint}.{summary_line}\n"
    "This cannot be fixed automatically without changing the target path."
)

TOOL_ISSUE_MISSING_FIELDS_TEMPLATE = (
    "External data is missing to continue{tool_hint}.{summary_line}\n"
    "Please clarify: {fields_label}."
)

TOOL_ISSUE_STAGNATION_TEMPLATE = (
    "Unable to complete the task after automatic recovery attempts{tool_hint}.{summary_line}\n"
    "Auto-recovery stopped due to stagnation: without new external data or changed conditions, no safe next steps remain."
)

LOOP_BUDGET_HANDOFF_TEMPLATE = (
    "Unable to continue recovery for task {task_hint}{tool_hint}: the internal step limit has been reached.\n"
    "In the current context, automatic strategies have been exhausted."
)

DEFAULT_INTERNAL_UI_NOTICE = "Paused at this step. You can continue with a new message."

LOOP_BUDGET_UI_NOTICE = (
    "Paused at this step: the internal retry limit for this request has been reached. "
    "You can continue with a new message."
)

TOOL_ISSUE_UI_NOTICE = (
    "Paused at this step. To move forward without unnecessary repetition, send a new request or a short clarification."
)

SUCCESSFUL_TOOL_STAGNATION_UI_NOTICE = (
    "Paused at this step: the result has already been confirmed several times in a row, and continuing may just loop. "
    "You can continue with a short message."
)
