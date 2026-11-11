"""Core retry and desktop stream repair must agree on provider error categories."""

import pytest

from core.api_key_rotation import (
    ApiKeyRotationExhaustedError,
    classify_api_key_error,
    classify_provider_error,
    describe_provider_error,
)
from core.nodes.llm import LLMMixin
from ui.runtime_worker import AgentRunWorker


@pytest.mark.parametrize("status,kind,retryable", [
    (400, "client_error", False),
    (401, "auth", False),
    (402, "billing", False),
    (403, "auth", False),
    (404, "client_error", False),
    (408, "timeout", True),
    (429, "rate_limit", True),
    (500, "server_error", True),
    (502, "server_error", True),
    (503, "server_error", True),
    (504, "server_error", True),
])
def test_http_status_agrees_across_exception_and_stream_diagnostics(status, kind, retryable):
    error = RuntimeError("Provider request failed")
    error.status_code = status
    classification = classify_provider_error(error)
    assert classification["kind"] == kind
    assert classification["retryable"] is retryable
    assert LLMMixin._is_transient_llm_error(error) is retryable
    assert AgentRunWorker._classify_stream_error(str(error), {"http_status": status}) == classification
    assert AgentRunWorker._classify_stream_error(str(error), describe_provider_error(error)) == classification
    assert classify_api_key_error(error) == (kind if kind in {"auth", "billing", "rate_limit"} else None)


@pytest.mark.parametrize("error,kind", [
    (RuntimeError("resource exhausted"), "rate_limit"),
    (RuntimeError("RESOURCE_EXHAUSTED"), "rate_limit"),
    (RuntimeError("HTTP 503"), "server_error"),
    (RuntimeError("connection reset by peer"), "network"),
    (TimeoutError(), "timeout"),
    (RuntimeError("client has been closed: connection unavailable"), "closed_client"),
    (RuntimeError("Unrecognized provider failure"), "unknown"),
])
def test_text_only_and_typed_errors_survive_safe_diagnostics(error, kind):
    classification = classify_provider_error(error)
    assert classification["kind"] == kind
    assert AgentRunWorker._classify_stream_error(str(error), describe_provider_error(error)) == classification


def test_nested_provider_code_and_cyclic_exception_context():
    cause = RuntimeError("Provider request failed")
    cause.code = lambda: "DEADLINE_EXCEEDED"
    error = RuntimeError("Invocation failed")
    error.__cause__ = cause
    cause.__context__ = error
    classification = classify_provider_error(error)
    assert classification["kind"] == "timeout"
    assert LLMMixin._is_transient_llm_error(error)
    assert AgentRunWorker._classify_stream_error(str(error), describe_provider_error(error)) == classification


def test_stream_repair_stops_after_outer_llm_retry_budget_is_exhausted():
    error = RuntimeError("HTTP 500")
    error.llm_retry_exhausted = True
    classification = AgentRunWorker._classify_stream_error(str(error), describe_provider_error(error))
    assert classification["kind"] == "retry_exhausted"
    assert classification["retryable"] is False


def test_rate_limited_key_pool_retains_outer_llm_retry_policy():
    error = ApiKeyRotationExhaustedError("All keys tried", error_kind="rate_limit", keys_tried=2, pool_size=2)
    assert LLMMixin._is_transient_llm_error(error)
    assert classify_api_key_error(error) == "rate_limit"
    assert AgentRunWorker._classify_stream_error(str(error), describe_provider_error(error))["retryable"] is False


def test_nested_closed_client_is_never_a_network_retry():
    error = RuntimeError("Network invocation failed")
    error.__cause__ = RuntimeError("Cannot send a request, as the client has been closed")
    assert LLMMixin._is_closed_client_error(error)
    assert not LLMMixin._is_transient_llm_error(error)


def test_nested_permanent_http_error_does_not_retry():
    cause = RuntimeError("Invalid request")
    cause.status_code = 400
    error = RuntimeError("Provider invocation failed")
    error.__cause__ = cause
    assert LLMMixin._is_permanent_client_error(error)
    assert not LLMMixin._is_transient_llm_error(error)
