from __future__ import annotations

import inspect
import logging
import re
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit

from langgraph.config import get_stream_writer

from core.logging_config import SensitiveDataFilter
from core.model_profiles import ModelProfileStore


class ApiKeyRotationError(RuntimeError):
    """Base error for model-profile API key rotation failures."""


class ApiKeyRotationExhaustedError(ApiKeyRotationError):
    """Raised when every key in the pool has been tried in one full circle without success."""

    def __init__(
        self,
        message: str,
        *,
        error_kind: str | None = None,
        keys_tried: int = 0,
        pool_size: int = 0,
    ) -> None:
        super().__init__(message)
        self.error_kind = error_kind
        self.keys_tried = max(0, int(keys_tried))
        self.pool_size = max(0, int(pool_size))
        # One complete key-pool pass consumes the application retry budget.
        self.llm_retry_exhausted = True


def _normalized_error_text(error: Exception) -> str:
    return " ".join(str(error).lower().split())


def _extract_status_code(error: Exception) -> int | None:
    for candidate in (
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    text = str(error).strip()
    match = re.search(r"^(?:HTTP\s*)?([45]\d{2})(?!\d)", text, re.IGNORECASE)
    if match is None:
        match = re.search(r"\b(?:HTTP(?:\s+status)?|status(?:\s+code)?|error\s+code)\s*[:=]?\s*([45]\d{2})(?!\d)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_request_host(error: Exception) -> str:
    request = getattr(error, "request", None) or getattr(getattr(error, "response", None), "request", None)
    url = getattr(request, "url", None)
    if url is None:
        return ""
    host = getattr(url, "host", None)
    if host:
        return str(host).strip().lower()
    try:
        return str(urlsplit(str(url)).hostname or "").strip().lower()
    except (TypeError, ValueError):
        return ""


def _extract_request_error_id(error: Exception) -> str:
    for candidate in (
        getattr(error, "request_id", None),
        getattr(error, "error_id", None),
    ):
        if candidate:
            return str(candidate).strip()

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        for name in ("x-request-id", "request-id", "x-error-id", "error-id"):
            try:
                candidate = headers.get(name)
            except (AttributeError, TypeError):
                candidate = None
            if candidate:
                return str(candidate).strip()

    match = re.search(r"\b(?:request|error)[ _-]?id\s*[:=]\s*([A-Za-z0-9._:-]+)", str(error), re.IGNORECASE)
    return match.group(1) if match else ""


def _safe_provider_error_message(error: Exception) -> str:
    text = " ".join(str(error or "").split())

    def replace_url(match: re.Match[str]) -> str:
        try:
            parsed = urlsplit(match.group(0))
        except ValueError:
            return "<redacted-url>"
        if not parsed.hostname:
            return "<redacted-url>"
        scheme = parsed.scheme.lower() or "https"
        return f"{scheme}://{parsed.hostname.lower()}"

    text = re.sub(r"https?://[^\s'\"<>]+", replace_url, text, flags=re.IGNORECASE)
    return SensitiveDataFilter._sanitize_string(text)


def describe_provider_error(error: Exception) -> dict[str, Any]:
    """Return safe structured diagnostics without URL paths, queries, or API keys."""
    root_cause = error
    seen: set[int] = set()
    while True:
        seen.add(id(root_cause))
        next_cause = getattr(root_cause, "__cause__", None) or getattr(root_cause, "__context__", None)
        if next_cause is None or id(next_cause) in seen:
            break
        root_cause = next_cause

    return {
        **classify_provider_error(error),
        "error_type": type(error).__name__,
        "http_status": _extract_status_code(error),
        "url_host": _extract_request_host(error),
        "request_error_id": _extract_request_error_id(error),
        "message": _safe_provider_error_message(error),
        "root_cause_type": type(root_cause).__name__,
        "root_cause_message": _safe_provider_error_message(root_cause),
        "retry_exhausted": bool(getattr(error, "llm_retry_exhausted", False)),
        "rotation_error_kind": getattr(error, "error_kind", None),
        "rotation_keys_tried": int(getattr(error, "keys_tried", 0) or 0),
        "rotation_pool_size": int(getattr(error, "pool_size", 0) or 0),
    }


def _extract_provider_code_text(error: Exception) -> str:
    code = getattr(error, "code", None)
    if callable(code):
        try:
            code = code()
        except TypeError:
            code = None
    status = getattr(error, "status", None)
    parts = [type(error).__name__, type(error).__module__, code, status]
    return " ".join(str(part or "").strip().lower() for part in parts if str(part or "").strip())


def _format_error_for_user(error: Exception | None, error_kind: str | None = None) -> str:
    if error is None:
        return "unknown error"
    parts = [type(error).__name__]
    status_code = _extract_status_code(error)
    if error_kind:
        parts.append(f"type: {error_kind}")
    if status_code is not None:
        parts.append(f"HTTP {status_code}")
    message = " ".join(str(error or "").split())
    if message:
        parts.append(message)
    return " | ".join(parts)


def _classify_key_error(error: Exception) -> str | None:
    status_code = _extract_status_code(error)
    if status_code in {401, 403}:
        return "auth"
    if status_code == 402:
        return "billing"
    if status_code == 429:
        return "rate_limit"

    text = _normalized_error_text(error)
    provider_code_text = _extract_provider_code_text(error)
    combined = f"{provider_code_text} {text}".strip()
    auth_markers = (
        "authenticationerror",
        "permissiondenied",
        "unauthenticated",
        "forbidden",
        "invalid_api_key",
        "incorrect api key",
        "authentication failed",
        "unauthorized",
        "permission denied",
        "permission_denied",
        "invalid api key",
        "error code: 401",
        "error code: 403",
    )
    rate_limit_markers = (
        "ratelimiterror",
        "toomanyrequests",
        "resourceexhausted",
        "resource_exhausted",
        "429",
        "too many requests",
        "resource exhausted",
        "rate limit",
        "rate_limit",
        "quota exceeded",
        "insufficient_quota",
    )
    billing_markers = (
        "402",
        "payment required",
        "insufficient_balance",
        "insufficient balance",
        "insufficient account balance",
        "billing",
        "credit balance",
    )
    if any(marker in combined for marker in auth_markers):
        return "auth"
    if any(marker in combined for marker in billing_markers):
        return "billing"
    if any(marker in combined for marker in rate_limit_markers):
        return "rate_limit"
    return None


def _provider_exception_chain(error: Exception):
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(cause for cause in (
            getattr(current, "__context__", None), getattr(current, "__cause__", None)
        ) if cause is not None)


def classify_provider_error(error: Exception | str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Share error categories while keeping each caller's retry budget separate."""
    details = details or {}
    causes = list(_provider_exception_chain(error)) if isinstance(error, Exception) else []
    try:
        status = int(details["http_status"]) if details.get("http_status") is not None else None
    except (TypeError, ValueError):
        status = None
    if status is None:
        status = next((value for cause in causes if (value := _extract_status_code(cause)) is not None), None)
    if status is None and not causes:
        status = _extract_status_code(RuntimeError(str(error)))
    text = " ".join(" ".join([
        *(f"{_extract_provider_code_text(cause)} {cause}" for cause in causes),
        str(error), str(details.get("root_cause_message") or "")
    ]).lower().split())
    key_kind = details.get("rotation_error_kind") or next((
        kind for cause in causes if (kind := getattr(cause, "error_kind", None) or _classify_key_error(cause))
    ), None)
    if not key_kind and not causes:
        key_kind = _classify_key_error(RuntimeError(str(error)))
    # A rate-limited key pool can be retried by the outer LLM policy. Only the
    # exhausted budget explicitly supplied in diagnostics prevents stream retry.
    if details.get("retry_exhausted"):
        kind = "retry_exhausted"
    elif "client has been closed" in text or "client is closed" in text:
        kind = "closed_client"
    elif status in {401, 403} or key_kind == "auth":
        kind = "auth"
    elif status == 402 or key_kind == "billing":
        kind = "billing"
    elif status is not None and 400 <= status < 500 and status not in {408, 429}:
        kind = "client_error"
    elif status == 429 or key_kind == "rate_limit":
        kind = "rate_limit"
    elif status == 408 or any(marker in text for marker in ("timeout", "timed out", "deadline exceeded", "deadline_exceeded")):
        kind = "timeout"
    elif (status is not None and 500 <= status < 600) or any(marker in text for marker in (
        "bad gateway", "service unavailable", "temporarily unavailable", "internal server error", "temporary failure"
    )):
        kind = "server_error"
    elif any(marker in text for marker in ("connection", "network", "reset by peer", "broken pipe", "eof", "stream ended", "disconnected")):
        kind = "network"
    else:
        # Preserve code-only categories already captured in safe diagnostics.
        kind = details.get("kind") if details.get("kind") in {"timeout", "server_error", "network"} else "unknown"
    labels = {
        "retry_exhausted": "Retry budget exhausted", "closed_client": "Client is closed",
        "auth": "Authentication error", "billing": "Billing error", "client_error": f"HTTP {status}",
        "rate_limit": "Rate limit", "timeout": "Timeout", "server_error": "Server error",
        "network": "Network error", "unknown": "Stream error",
    }
    return {"kind": kind, "retryable": kind in {"rate_limit", "timeout", "server_error", "network"}, "label": labels[kind]}


def classify_api_key_error(error: Exception) -> str | None:
    kind = classify_provider_error(error)["kind"]
    return kind if kind in {"auth", "billing", "rate_limit"} else None


def _setup_rotation_logger(config: Any) -> logging.Logger:
    """Ensure the api_key_rotation logger writes to a dedicated log file."""
    logger = logging.getLogger("agent.api_key_rotation")
    # Avoid adding duplicate handlers on repeated instantiations
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return logger

    log_file = getattr(config, "log_file", None)
    if log_file is None:
        return logger

    rotation_log = Path(log_file).parent / "api_key_rotation.log"
    try:
        rotation_log.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(rotation_log), encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)
    except Exception:
        pass
    return logger


def _mask_key(key: str, tail: int = 5) -> str:
    """Return key with only the first chars and last *tail* visible, middle replaced with ***.
    Short keys (<= tail+2) are returned as-is with *** appended."""
    if len(key) <= tail + 2:
        return key[:2] + "***"
    return key[: -tail].rstrip() + "***" + key[-tail:]


async def _close_model_clients(model: Any) -> None:
    if model is None:
        return
    pending_models = [model]
    seen_models: set[int] = set()
    closed_clients: set[int] = set()
    while pending_models:
        current = pending_models.pop()
        marker = id(current)
        if marker in seen_models:
            continue
        seen_models.add(marker)

        # bind_tools() commonly returns a RunnableBinding. The provider model,
        # including its root HTTP clients, remains reachable through ``bound``.
        bound = getattr(current, "bound", None)
        if bound is not None and bound is not current:
            pending_models.append(bound)

        for target in (
            getattr(current, "root_async_client", None),
            getattr(current, "async_client", None),
            getattr(current, "root_client", None),
            getattr(current, "client", None),
        ):
            if target is None or id(target) in closed_clients:
                continue
            close_method = getattr(target, "aclose", None) or getattr(target, "close", None)
            if not callable(close_method):
                continue
            closed_clients.add(id(target))
            result = close_method()
            if inspect.isawaitable(result):
                await result


class RotatingChatModel:
    def __init__(
        self,
        *,
        config: Any,
        profile_id: str,
        profile_store_path: str | Path,
        llm_factory: Callable[..., Any],
        bound_tools: Sequence[Any] | None = None,
        model_cache: dict[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._profile_id = str(profile_id or "").strip()
        self._profile_store = ModelProfileStore(profile_store_path)
        self._llm_factory = llm_factory
        self._bound_tools = list(bound_tools or [])
        self._logger = _setup_rotation_logger(config)
        # Share the model cache across bind_tools() descendants so per-turn
        # tool-subset bindings reuse the same provider clients instead of
        # leaking a fresh client pool per agent turn.
        self._model_cache: dict[str, Any] = model_cache if model_cache is not None else {}
        self._prototype_model = self._build_model(self._initial_api_key())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._prototype_model, name)

    def bind_tools(self, tools: Sequence[Any]) -> "RotatingChatModel":
        return self.__class__(
            config=self._config,
            profile_id=self._profile_id,
            profile_store_path=self._profile_store.path,
            llm_factory=self._llm_factory,
            bound_tools=list(tools),
            model_cache=self._model_cache,
        )

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        state = self._profile_store.get_api_key_state(self._profile_id)
        api_keys = list(state.get("api_keys") or [])

        # Пул ключей не задан — работаем с дефолтным ключом из конфига.
        if not api_keys:
            return await self._build_model(self._initial_api_key()).ainvoke(input, **kwargs)

        start_index = int(state.get("api_key_index") or 0)
        total = len(api_keys)
        last_error: Exception | None = None

        for offset in range(total):
            current_index = (start_index + offset) % total
            active_key = api_keys[current_index]

            self._logger.debug(
                f"Attempt {offset + 1}/{total}: key #{current_index} "
                f"for profile '{self._profile_id}'"
            )

            try:
                result = await self._build_model(active_key).ainvoke(input, **kwargs)
                # Успех — индекс уже был сдвинут при предыдущих ошибках,
                # следующий вызов начнёт с текущего рабочего ключа.
                return result

            except Exception as exc:
                error_kind = classify_api_key_error(exc)

                if error_kind is None:
                    # Non-key error (network, timeout, model error, etc.) —
                    # propagate immediately; rotation will not help.
                    details = describe_provider_error(exc)
                    self._logger.warning(
                        "Non-key error for profile '%s': type=%s http_status=%s "
                        "url_host=%s request_error_id=%s message=%s",
                        self._profile_id,
                        details["error_type"],
                        details["http_status"] or "-",
                        details["url_host"] or "-",
                        details["request_error_id"] or "-",
                        details["message"],
                    )
                    raise

                last_error = exc
                next_index = (current_index + 1) % total
                next_key_masked = _mask_key(api_keys[next_index]) if total > 1 else "(none)"
                self._logger.info(
                    f"Key #{current_index} ({_mask_key(active_key)}) returned "
                    f"{error_kind} error for profile '{self._profile_id}' — "
                    f"switching to key #{next_index} ({next_key_masked})"
                )

                # Advance the index in the store without marking the key as invalid.
                self._profile_store.rotate_api_key(self._profile_id, active_key)
                if offset + 1 < total:
                    self._emit_api_key_rotated_event(
                        error_kind=error_kind,
                        from_index=current_index,
                        to_index=next_index,
                    )

        # Full circle completed — no key succeeded.
        model_label = self._model_label()
        tried_summary = ", ".join(
            f"#{i}({_mask_key(k)})" for i, k in enumerate(api_keys)
        )
        last_error_text = _format_error_for_user(
            last_error,
            classify_api_key_error(last_error) if last_error is not None else None,
        )
        self._logger.error(
            f"All {total} keys exhausted for profile '{self._profile_id}' "
            f"[{tried_summary}]. Last error: {last_error_text}"
        )
        exhausted_error_kind = classify_api_key_error(last_error) if last_error is not None else None
        raise ApiKeyRotationExhaustedError(
            f"All API keys for '{model_label}' were exhausted in one full rotation. "
            f"Last error: {last_error_text}. "
            "Check the keys and their limits, or try again later.",
            error_kind=exhausted_error_kind,
            keys_tried=total,
            pool_size=total,
        ) from last_error

    def _build_model(self, api_key: str) -> Any:
        cache_key = self._model_cache_key(api_key)
        cached_model = self._model_cache.get(cache_key)
        if cached_model is not None:
            return cached_model
        model = self._llm_factory(self._config, api_key_override=api_key)
        if self._bound_tools:
            model = model.bind_tools(self._bound_tools)
        self._model_cache[cache_key] = model
        return model

    def _model_cache_key(self, api_key: str) -> str:
        # Bound tools change the model's behavior, so cache entries must be
        # per (key, tools) pair even when the model cache is shared between
        # bind_tools() descendants.
        if not self._bound_tools:
            return str(api_key or "")
        tool_names = tuple(self._tool_cache_name(tool) for tool in self._bound_tools)
        return f"{api_key or ''}::tools::{len(tool_names)}::{'|'.join(tool_names)}"

    @staticmethod
    def _tool_cache_name(tool: Any) -> str:
        name = getattr(tool, "name", None)
        if name is None and isinstance(tool, dict):
            function = tool.get("function")
            name = tool.get("name") or (function.get("name") if isinstance(function, dict) else None)
        return str(name if name is not None else tool)

    def _emit_api_key_rotated_event(
        self,
        *,
        error_kind: str,
        from_index: int,
        to_index: int,
    ) -> None:
        """Notify the live stream that the assistant section restarts on a new key.

        Tokens already streamed by the failed key belong to an aborted response;
        the retry on the next key streams a fresh answer from scratch. Without
        this signal the UI would concatenate the partial text with the full
        replay from the next key.
        """
        try:
            writer = get_stream_writer()
        except RuntimeError:
            # Direct unit invocations have no LangGraph streaming context.
            return
        writer(
            {
                "type": "api_key_rotated",
                "error_kind": str(error_kind or ""),
                "from_index": int(from_index),
                "to_index": int(to_index),
            }
        )

    async def aclose(self) -> None:
        seen: set[int] = set()
        for model in list(self._model_cache.values()):
            marker = id(model)
            if marker in seen:
                continue
            seen.add(marker)
            await _close_model_clients(model)
        self._model_cache.clear()

    def _initial_api_key(self) -> str:
        provider = getattr(self._config, "provider", "")
        if provider == "gemini":
            secret = getattr(self._config, "gemini_api_key", None)
        elif provider == "anthropic":
            secret = getattr(self._config, "anthropic_api_key", None)
        else:
            secret = getattr(self._config, "openai_api_key", None)
        return secret.get_secret_value() if secret is not None else ""

    def _model_label(self) -> str:
        provider = getattr(self._config, "provider", "")
        if provider == "gemini":
            return str(getattr(self._config, "gemini_model", "") or self._profile_id or "model")
        if provider == "anthropic":
            return str(getattr(self._config, "anthropic_model", "") or self._profile_id or "model")
        return str(getattr(self._config, "openai_model", "") or self._profile_id or "model")
