import logging
import re
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import Field, PrivateAttr, SecretStr, model_validator, field_validator
from pydantic_settings import BaseSettings, JsonConfigSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict

from core.constants import BASE_DIR

# --- Defaults ---
DEFAULT_MAX_FILE_SIZE = 300 * 1024 * 1024  # 300 MB
DEFAULT_READ_LIMIT = 2000
_SIZE_WITH_UNIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)\s*$", re.IGNORECASE)
_SIZE_MULTIPLIERS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000 ** 2,
    "gb": 1000 ** 3,
    "tb": 1000 ** 4,
    "kib": 1024,
    "mib": 1024 ** 2,
    "gib": 1024 ** 3,
    "tib": 1024 ** 4,
}


def _coerce_env_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _candidate_runtime_dirs() -> list[Path]:
    if getattr(sys, "frozen", False):
        return [BASE_DIR]

    dirs: list[Path] = [BASE_DIR]
    cwd = Path.cwd()
    if cwd not in dirs:
        dirs.append(cwd)
    return dirs


def _existing_path_or_default(filename: str, default_dir: Path = BASE_DIR) -> Path:
    for directory in _candidate_runtime_dirs():
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return default_dir / filename


def _env_file_candidates() -> tuple[Path, ...]:
    seen: list[Path] = []
    for directory in _candidate_runtime_dirs():
        candidate = directory / ".env"
        if candidate not in seen:
            seen.append(candidate)
    return tuple(seen)


def _resolve_runtime_path(value: Union[str, Path], *, base_dir: Path = BASE_DIR) -> Path:
    path = value if isinstance(value, Path) else Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


class AgentConfig(BaseSettings):
    """
    Agent configuration loaded from environment variables and the .env file.
    """

    model_config = SettingsConfigDict(
        env_file=_env_file_candidates(),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority: explicit init kwargs and real env vars first, then
        # .agent_state/config.json (UI-saved overrides such as SESSION_SIZE),
        # then the .env file. The JSON file wins over .env but never over
        # process environment variables. profile_store takes care to preserve
        # the session_size key when it rewrites config.json.
        return (
            init_settings,
            env_settings,
            JsonConfigSettingsSource(settings_cls, json_file=BASE_DIR / ".agent_state" / "config.json"),
            dotenv_settings,
            file_secret_settings,
        )

    # Paths
    prompt_path: Path = Field(default_factory=lambda: _existing_path_or_default("prompt.txt"), alias="PROMPT_PATH")
    mcp_config_path: Path = Field(default_factory=lambda: _existing_path_or_default("mcp.json"), alias="MCP_CONFIG_PATH")
    checkpoint_backend: Literal["sqlite", "memory"] = Field(
        default="sqlite",
        alias="CHECKPOINT_BACKEND",
    )
    checkpoint_sqlite_path: Path = Field(
        default=BASE_DIR / ".agent_state" / "checkpoints.sqlite",
        alias="CHECKPOINT_SQLITE_PATH",
    )
    model_profile_config_path: Path = Field(
        default=BASE_DIR / ".agent_state" / "config.json",
        alias="MODEL_PROFILE_CONFIG_PATH",
    )
    provider_registry_path: Path = Field(
        default_factory=lambda: _existing_path_or_default("provider_registry.json"),
        alias="PROVIDER_REGISTRY_PATH",
    )
    session_state_path: Path = Field(
        default=BASE_DIR / ".agent_state" / "session.json",
        alias="SESSION_STATE_PATH",
    )
    run_log_dir: Path = Field(default=BASE_DIR / "logs" / "runs", alias="RUN_LOG_DIR")
    log_file: Path = Field(default=BASE_DIR / "logs" / "agent.log", alias="LOG_FILE")

    # Provider Settings
    provider: Literal["gemini", "openai", "anthropic"] = Field(default="gemini", alias="PROVIDER")
    active_model_profile_id: Optional[str] = Field(default=None, alias="ACTIVE_MODEL_PROFILE_ID")

    # Tavily Search
    tavily_api_key: Optional[SecretStr] = Field(default=None, alias="TAVILY_API_KEY")

    # Gemini
    gemini_api_key: Optional[SecretStr] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")

    # OpenAI / Compatible
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")

    # Anthropic
    anthropic_api_key: Optional[SecretStr] = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-5-20250929", alias="ANTHROPIC_MODEL")
    anthropic_base_url: Optional[str] = Field(default=None, alias="ANTHROPIC_BASE_URL")
    anthropic_max_tokens: int = Field(default=8192, alias="ANTHROPIC_MAX_TOKENS")
    anthropic_thinking_budget: int = Field(default=4096, alias="ANTHROPIC_THINKING_BUDGET")
    anthropic_reasoning: str = Field(default="", alias="ANTHROPIC_REASONING")
    enable_model_reasoning: bool = Field(default=True, alias="ENABLE_MODEL_REASONING")
    debug_reasoning_stream: bool = Field(default=False, alias="DEBUG_REASONING_STREAM")
    model_reasoning_effort: str = Field(default="medium", alias="MODEL_REASONING_EFFORT")
    gemini_thinking_budget: int = Field(default=4096, alias="GEMINI_THINKING_BUDGET")
    show_model_thoughts: bool = Field(default=False, alias="SHOW_MODEL_THOUGHTS")
    llm_api_mode: Literal["chat", "responses"] = Field(default="chat", alias="LLM_API_MODE")

    # Common Logic
    temperature: float = Field(default=0.2, alias="TEMPERATURE")
    max_loops: int = Field(default=50, alias="MAX_LOOPS", description="Limit steps per request")
    tool_loop_window: Optional[int] = Field(default=None, alias="TOOL_LOOP_WINDOW")
    tool_loop_limit_mutating: Optional[int] = Field(default=None, alias="TOOL_LOOP_LIMIT_MUTATING")
    tool_loop_limit_readonly: Optional[int] = Field(default=None, alias="TOOL_LOOP_LIMIT_READONLY")

    # Features Toggle
    enable_search_tools: bool = Field(default=True, alias="ENABLE_SEARCH_TOOLS")
    model_supports_tools: bool = Field(default=True, alias="MODEL_SUPPORTS_TOOLS")
    enable_filesystem_tools: bool = Field(default=True, alias="ENABLE_FILESYSTEM_TOOLS")
    enable_process_tools: bool = Field(default=False, alias="ENABLE_PROCESS_TOOLS")
    enable_shell_tool: bool = Field(default=False, alias="ENABLE_SHELL_TOOL")
    enable_approvals: bool = Field(default=True, alias="ENABLE_APPROVALS")
    enable_text_tool_call_recovery: bool = Field(default=False, alias="ENABLE_TEXT_TOOL_CALL_RECOVERY")
    allow_external_process_control: bool = Field(default=False, alias="ALLOW_EXTERNAL_PROCESS_CONTROL")

    # Tools Limits
    max_tool_output_length: int = Field(default=4000, alias="MAX_TOOL_OUTPUT")
    max_raw_tool_output_length: int = Field(
        default=100000,
        alias="MAX_RAW_TOOL_OUTPUT",
        description="Maximum characters captured before tool output compression",
    )
    enable_headroom_compression: bool = Field(
        default=False,
        alias="ENABLE_HEADROOM_COMPRESSION",
        description="Compress oversized noisy tool outputs (shell/search/listings) with headroom instead of hard truncation",
    )
    stream_text_max_chars: int = Field(default=120000, alias="STREAM_TEXT_MAX_CHARS")
    stream_events_max: int = Field(default=400, alias="STREAM_EVENTS_MAX")
    stream_tool_buffer_max: int = Field(default=128, alias="STREAM_TOOL_BUFFER_MAX")
    self_correction_retry_limit: int = Field(default=8, alias="SELF_CORRECTION_RETRY_LIMIT")
    max_file_size: int = Field(
        default=DEFAULT_MAX_FILE_SIZE,
        alias="MAX_FILE_SIZE",
        description="Max file size in bytes",
    )
    max_background_processes: int = Field(default=5, alias="MAX_BACKGROUND_PROCESSES")
    max_search_chars: int = Field(default=15000, alias="MAX_SEARCH_CHARS")
    max_read_lines: int = Field(default=DEFAULT_READ_LIMIT, alias="MAX_READ_LINES")

    # Deterministic Mode
    strict_mode: bool = Field(default=False, alias="STRICT_MODE")

    # Summarization
    summary_threshold: int = Field(
        default=8000,
        alias="SESSION_SIZE",
        description="Estimated input context tokens before summarizing (~chars/2)",
    )
    summary_reserved_tokens: int = Field(
        default=3000,
        alias="SUMMARY_RESERVED_TOKENS",
        description="Estimated fixed context overhead for prompts, tool schemas, and provider wrappers",
    )
    summary_keep_last: int = Field(default=4, alias="SUMMARY_KEEP_LAST")
    summary_max_tokens: int = Field(
        default=0,
        alias="SUMMARY_MAX_TOKENS",
        description="Maximum estimated tokens for compressed memory; 0 derives a quarter of the summarization threshold",
    )

    # Network / Retry
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_delay: int = Field(default=2, alias="RETRY_DELAY")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("max_file_size", mode="before")
    @classmethod
    def parse_max_file_size(cls, v: Union[int, float, str]) -> int:
        """
        Parse byte limits strictly.
        Plain numeric values are treated as bytes.
        String values may optionally include explicit units, e.g. 4MB or 300MiB.
        """
        if isinstance(v, (int, float)):
            value = int(v)
        elif isinstance(v, str):
            raw = v.strip()
            if not raw:
                raise ValueError("MAX_FILE_SIZE cannot be empty.")
            if raw.isdigit():
                value = int(raw)
            else:
                match = _SIZE_WITH_UNIT_RE.match(raw)
                if not match:
                    raise ValueError(
                        "Invalid MAX_FILE_SIZE format. Use bytes (e.g. 4096) or explicit units like 4MB / 300MiB."
                    )
                amount = float(match.group(1))
                unit = match.group(2).lower()
                value = int(amount * _SIZE_MULTIPLIERS[unit])
        else:
            raise ValueError("Invalid MAX_FILE_SIZE value.")

        if value < 1:
            raise ValueError("MAX_FILE_SIZE must be greater than 0.")
        return value

    @field_validator(
        "prompt_path",
        "mcp_config_path",
        "checkpoint_sqlite_path",
        "model_profile_config_path",
        "provider_registry_path",
        "session_state_path",
        "run_log_dir",
        "log_file",
        mode="before",
    )
    @classmethod
    def resolve_path_fields(cls, v: Union[str, Path]) -> Path:
        return _resolve_runtime_path(v)

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: Any) -> str:
        value = str(v or "INFO").strip().upper()
        return value if value in logging.getLevelNamesMapping() else "INFO"

    @field_validator("max_loops", mode="before")
    @classmethod
    def validate_max_loops(cls, v: Union[int, float, str]) -> int:
        """
        Ensure MAX_LOOPS is a positive integer.
        Prevents invalid recursion_limit values and ambiguous loop-guard behavior.
        """
        try:
            value = int(float(v))
        except (TypeError, ValueError):
            return 50

        if value < 1:
            return 1
        if value > 10000:
            return 10000
        return value

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_self_correction_settings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw = dict(data)
        if raw.get("SELF_CORRECTION_RETRY_LIMIT") not in (None, "") or raw.get("self_correction_retry_limit") not in (None, ""):
            return raw

        legacy_enabled = raw.get("SELF_CORRECTION_ENABLE_AUTO_REPAIR", raw.get("self_correction_enable_auto_repair"))
        if _coerce_env_bool(legacy_enabled) is False:
            raw["SELF_CORRECTION_RETRY_LIMIT"] = 0
            return raw

        legacy_ceiling = raw.get("SELF_CORRECTION_HARD_CEILING", raw.get("self_correction_hard_ceiling"))
        if legacy_ceiling not in (None, ""):
            raw["SELF_CORRECTION_RETRY_LIMIT"] = legacy_ceiling
            return raw

        legacy_repairs = raw.get("SELF_CORRECTION_MAX_AUTO_REPAIRS", raw.get("self_correction_max_auto_repairs"))
        if legacy_repairs not in (None, ""):
            raw["SELF_CORRECTION_RETRY_LIMIT"] = legacy_repairs
        return raw

    @field_validator("checkpoint_backend", mode="before")
    @classmethod
    def normalize_checkpoint_backend(cls, v: str) -> str:
        value = str(v or "sqlite").strip().lower()
        if value not in {"sqlite", "memory"}:
            return "sqlite"
        return value

    @field_validator("llm_api_mode", mode="before")
    @classmethod
    def normalize_llm_api_mode(cls, v: Any) -> str:
        value = str(v or "chat").strip().lower()
        if value not in {"chat", "responses"}:
            return "chat"
        return value

    @field_validator(
        "tool_loop_window",
        "tool_loop_limit_mutating",
        "tool_loop_limit_readonly",
        mode="before",
    )
    @classmethod
    def parse_optional_loop_guard_value(cls, v: Union[int, float, str, None]) -> Optional[int]:
        """Parse optional loop-guard value from env and clamp to safe bounds."""
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        try:
            value = int(float(v))
        except (TypeError, ValueError):
            return None
        if value < 1:
            return 1
        if value > 10000:
            return 10000
        return value

    @field_validator(
        "stream_text_max_chars",
        "stream_events_max",
        "stream_tool_buffer_max",
        mode="before",
    )
    @classmethod
    def validate_positive_runtime_limits(cls, v: Union[int, float, str]) -> int:
        try:
            value = int(float(v))
        except (TypeError, ValueError):
            return 1
        if value < 1:
            return 1
        if value > 1_000_000:
            return 1_000_000
        return value

    @field_validator("self_correction_retry_limit", mode="before")
    @classmethod
    def validate_self_correction_retry_limit(cls, v: Union[int, float, str]) -> int:
        try:
            value = int(float(v))
        except (TypeError, ValueError):
            return 0
        if value < 0:
            return 0
        if value > 1_000_000:
            return 1_000_000
        return value

    # Private cache attributes — PrivateAttr is the correct Pydantic v2 way to cache
    # computed values on BaseSettings instances (functools.cached_property doesn't work
    # because Pydantic v2 models don't expose a writable __dict__ slot for it).
    _cache_tool_loop_window: Optional[int] = PrivateAttr(default=None)
    _cache_tool_loop_limit_mutating: Optional[int] = PrivateAttr(default=None)
    _cache_tool_loop_limit_readonly: Optional[int] = PrivateAttr(default=None)
    _cache_safety: Optional[Any] = PrivateAttr(default=None)

    @property
    def effective_tool_loop_window(self) -> int:
        """History window for tool duplicate detection.
        Default is synchronized with MAX_LOOPS, unless overridden in env."""
        if self._cache_tool_loop_window is None:
            value = self.tool_loop_window if self.tool_loop_window is not None else max(10, min(60, self.max_loops))
            self._cache_tool_loop_window = value
        return self._cache_tool_loop_window

    @property
    def effective_tool_loop_limit_mutating(self) -> int:
        """Mutating tools duplicate limit.
        Formula default: max(6, min(24, MAX_LOOPS))."""
        if self._cache_tool_loop_limit_mutating is None:
            value = self.tool_loop_limit_mutating if self.tool_loop_limit_mutating is not None else max(6, min(24, self.max_loops))
            self._cache_tool_loop_limit_mutating = value
        return self._cache_tool_loop_limit_mutating

    @property
    def effective_tool_loop_limit_readonly(self) -> int:
        """Read-only tools duplicate limit.
        Formula default: max(12, min(40, MAX_LOOPS * 2))."""
        if self._cache_tool_loop_limit_readonly is None:
            value = self.tool_loop_limit_readonly if self.tool_loop_limit_readonly is not None else max(12, min(40, self.max_loops * 2))
            self._cache_tool_loop_limit_readonly = value
        return self._cache_tool_loop_limit_readonly

    @property
    def effective_summary_max_tokens(self) -> int:
        """Token budget for compressed memory.
        Default is a quarter of SESSION_SIZE, so memory can never squeeze out live history."""
        if self.summary_max_tokens > 0:
            return self.summary_max_tokens
        return max(0, self.summary_threshold) // 4

    @property
    def safety(self):
        """Returns SafetyPolicy object. Cached via PrivateAttr to prevent multiple instantiations."""
        if self._cache_safety is None:
            from core.safety_policy import SafetyPolicy
            self._cache_safety = SafetyPolicy(
                max_tool_output=self.max_tool_output_length,
                max_raw_tool_output=max(self.max_tool_output_length, self.max_raw_tool_output_length),
                max_file_size=self.max_file_size,
                max_background_processes=self.max_background_processes,
                max_search_chars=self.max_search_chars,
                max_read_lines=self.max_read_lines,
                allow_shell=self.enable_shell_tool,
                allow_external_process_control=self.allow_external_process_control,
            )
        return self._cache_safety

    @model_validator(mode="after")
    def validate_provider_keys(self) -> "AgentConfig":
        if self.provider == "anthropic":
            reasoning = str(self.anthropic_reasoning or "").strip().lower()
            allowed = {"", "adaptive", "off", "none", "low", "medium", "high", "xhigh", "max"}
            if reasoning not in allowed:
                raise ValueError(
                    f"ANTHROPIC_REASONING='{self.anthropic_reasoning}' is invalid. "
                    "Allowed: adaptive, off, none, low, medium, high, xhigh, max, or empty."
                )
            self.anthropic_reasoning = reasoning
            if not self.anthropic_api_key and not self.anthropic_base_url:
                raise ValueError("ANTHROPIC_API_KEY required for anthropic provider.")

        if self.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for gemini provider.")

        if self.provider == "openai" and not self.openai_api_key:
            # Bypass API key check if a base_url is provided (common for local models like Ollama/vLLM)
            if not self.openai_base_url:
                raise ValueError("OPENAI_API_KEY required for openai provider.")

        return self
