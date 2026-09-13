"""
Semantic compression and bounded reduction for verbose tool outputs.

File-oriented tools are intentionally excluded because their content must stay
exact for subsequent edit operations. Headroom is optional; deterministic
reducers remain available when it is disabled, unavailable, or returns a
passthrough result.

Compression is delegated to headroom's ``ContentRouter``, which detects the
content type (build log, search results, JSON array, tabular data, plain text)
and applies the matching compressor. The router has no notion of this runtime's
output budget and may stop after a small lossless fold, so an oversized or rejected
routed result gets a log-only fallback whose caps are sized to the budget instead
of headroom's small defaults. Candidates are validated lazily against one source
diagnostic profile; an explicit lossless chain permits compact repetition folds.
CCR retrieval markers are disabled:
this runtime exposes no ``headroom_retrieve`` tool, so a marker would leave the
model with an unresolvable pointer instead of data. MCP tool outputs use the same
router and are identified by their runtime metadata (``source == "mcp"``)
instead of a name allowlist, so no extra per-server option in ``mcp.json`` is
required.

Reduction markers are appended as a footer so that a leading ``ERROR[TYPE]:``
envelope stays at the start of the content and remains detectable by
``parse_tool_execution_result``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterator, Optional

from core.tool_results import parse_tool_execution_result
from core.utils import truncate_output

logger = logging.getLogger(__name__)

COMPRESSIBLE_TOOL_NAMES = frozenset(
    {
        "cli_exec",
        "batch_web_search",
        "fetch_content",
        "crawl_site",
        "list_directory",
    }
)

_MIN_COMPRESSION_CHARS = 2000
# Unless Headroom explicitly reports a lossless chain, content without diagnostics
# must still fill at least this fraction (1/N) of the output budget. Otherwise a
# compressor that deleted content instead of condensing it would win over the
# deterministic reducer, which keeps verbatim head/diagnostics/tail.
_MIN_BUDGET_FRACTION = 4
# Log lines stay well below this width, so the output budget divided by it is a line
# count whose rendering still fits the budget. Used to size headroom's log-compressor
# caps, which otherwise stay at defaults tuned for a small fixed budget.
_LOG_LINE_CHARS = 80
# Line-scale caps rise with the budget; stack traces are bulkier than single lines,
# so they get this fraction (1/N) of the line budget.
_STACK_TRACE_BUDGET_FRACTION = 20
# headroom's names for the log strategy and for "no known build/test log format".
_LOG_STRATEGY = "log"
_GENERIC_LOG_FORMAT = "generic"
# Mirrors headroom.parser.CCR_RETRIEVAL_MARKER_RE: every form that asks the model
# to fetch offloaded content through headroom's retrieval tool.
_CCR_MARKER_RE = re.compile(r"<<ccr:[^>]+>>|Retrieve more: hash=|Retrieve original: hash=")
_DIAGNOSTIC_LINE_RE = re.compile(
    r"(?i)\b(errors?|failed|failures?|fatal|warnings?|warn|traceback|exceptions?|timeout|denied)\b"
)
# Identifiers, paths and numbers without the punctuation that structured
# compressors rewrite (quotes, colons, commas), so a re-serialised record still
# matches the diagnostic line it came from.
_SIGNIFICANT_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\\-]{3,}")
_SECTION_PREFIXES = {
    "batch_web_search": "Query: ",
    "fetch_content": "=== SOURCE: ",
    "crawl_site": "=== PAGE: ",
}


class ToolOutputCompressor:
    """Lazily-initialised headroom wrapper with deterministic fallbacks."""

    def __init__(self, enabled: bool, min_chars: int = _MIN_COMPRESSION_CHARS) -> None:
        self._enabled = enabled
        self._min_chars = min_chars
        self._routers: Dict[int, Any] = {}
        self._log_compressors: Dict[tuple[int, bool], Any] = {}
        self._headroom_unavailable = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_router(self, limit: int) -> Any:
        router = self._routers.get(limit)
        if router is not None:
            return router
        if self._headroom_unavailable:
            return None
        try:
            from headroom.transforms import ContentRouter, ContentRouterConfig

            router = ContentRouter(
                ContentRouterConfig(
                    ccr_enabled=False,
                    ccr_inject_marker=False,
                    log_compressor=self._log_compressor_config(limit, dedupe_warnings=False),
                )
            )
        except Exception:
            self._headroom_unavailable = True
            logger.debug("headroom unavailable, tool output compression disabled", exc_info=True)
            return None
        self._routers[limit] = router
        return router

    def _get_log_compressor(self, limit: int, *, dedupe_warnings: bool) -> Any:
        compressor = self._log_compressors.get((limit, dedupe_warnings))
        if compressor is not None:
            return compressor
        if self._headroom_unavailable:
            return None
        try:
            from headroom.transforms import LogCompressor

            compressor = LogCompressor(
                self._log_compressor_config(limit, dedupe_warnings=dedupe_warnings)
            )
        except Exception:
            self._headroom_unavailable = True
            logger.debug("headroom log compressor unavailable", exc_info=True)
            return None
        self._log_compressors[(limit, dedupe_warnings)] = compressor
        return compressor

    @staticmethod
    def _log_compressor_config(limit: int, *, dedupe_warnings: bool) -> Any:
        """Log-compressor caps sized to the output budget instead of headroom's defaults.

        The defaults keep at most 10 errors, 5 warnings and 100 lines, which is far
        less than a large budget can carry: every failure past the cap is dropped
        before this class can judge the result, and a build that failed in 40 places
        arrives as 10. Caps below headroom's own defaults would make small budgets
        worse, so those defaults are the floor. Retrieval markers stay off for the
        same reason as in the router: no retrieval tool is exposed here.
        """
        from headroom.transforms import LogCompressorConfig

        defaults = LogCompressorConfig()
        lines = max(defaults.max_total_lines, limit // _LOG_LINE_CHARS)
        return LogCompressorConfig(
            max_errors=lines,
            max_warnings=lines,
            max_stack_traces=max(defaults.max_stack_traces, lines // _STACK_TRACE_BUDGET_FRACTION),
            dedupe_warnings=dedupe_warnings,
            max_total_lines=lines,
            enable_ccr=False,
        )

    def compress(
        self,
        *,
        content: str,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]],
        limit: int,
        is_mcp: bool = False,
        user_query: str = "",
    ) -> Optional[str]:
        """Return compressed content, or None when compression is not applicable."""
        if not self._enabled:
            return None
        if limit <= 0 or not content or len(content) <= max(limit, self._min_chars):
            return None
        if not is_mcp and tool_name not in COMPRESSIBLE_TOOL_NAMES:
            return None

        router = self._get_router(limit)
        if router is None:
            return None

        routed, strategy, lossless = self._route(
            router,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args,
            user_query=user_query,
        )
        # Prepare the source-side comparison only once, and only if a candidate
        # exists. Re-tokenizing every diagnostic and reducing the source for each
        # candidate used to cost several full passes over large outputs.
        diagnostic_profile: Optional[tuple[list[frozenset[str]], int]] = None

        def validate(
            candidate: str, candidate_strategy: str, *, lossless: bool = False
        ) -> Optional[str]:
            nonlocal diagnostic_profile
            result = candidate + (
                f"\n[COMPRESSED by headroom | {len(content)} -> {len(candidate)} chars | "
                f"tool={tool_name} | strategy={candidate_strategy}]"
            )
            # Metadata is part of the cost, not useful content or proof of savings.
            if len(result) >= len(content):
                return None
            if diagnostic_profile is None:
                diagnostics = self._diagnostic_tokens(content)
                baseline = ""
                if diagnostics:
                    baseline = self.reduce_to_limit(
                        content=content, tool_name=tool_name, limit=limit, is_mcp=is_mcp
                    )
                diagnostic_profile = diagnostics, self._diagnostics_kept(diagnostics, baseline)
            diagnostics, baseline_kept = diagnostic_profile
            if self._preserves_diagnostics(
                diagnostics=diagnostics,
                baseline_kept=baseline_kept,
                compressed=result,
                body_length=len(candidate),
                tool_name=tool_name,
                limit=limit,
                is_mcp=is_mcp,
                lossless=lossless,
            ):
                return result
            logger.debug(
                "headroom %s result dropped too much content for tool %s",
                candidate_strategy,
                tool_name,
            )
            return None

        routed_result = validate(routed, strategy, lossless=lossless) if routed is not None else None
        if routed_result is not None and len(routed_result) <= limit:
            return routed_result

        # The router can stop after a tiny lossless fold, or return a short but
        # unusable result. Try log condensation in both cases. Warning dedupe can
        # still normalize values after ':'/'=' in headroom 0.37, so retry without
        # it only when the first candidate fails validation, not unconditionally.
        for candidate in self._log_candidates(
            content=content, tool_name=tool_name, limit=limit, routed_strategy=strategy
        ):
            log_result = validate(candidate, _LOG_STRATEGY)
            if log_result is not None:
                return min(
                    (result for result in (routed_result, log_result) if result is not None),
                    key=len,
                )
        return routed_result

    def _route(
        self,
        router: Any,
        *,
        content: str,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]],
        user_query: str,
    ) -> tuple[Optional[str], str, bool]:
        """Return routed content, strategy and explicit lossless provenance."""
        question = str(user_query or "").strip()
        try:
            result = router.compress(
                content,
                context=self._build_query(tool_name, tool_args),
                question=question or None,
            )
        except Exception:
            logger.debug("headroom compression failed for tool %s", tool_name, exc_info=True)
            return None, "", False

        strategy = getattr(getattr(result, "strategy_used", None), "value", "") or "unknown"
        compressed = getattr(result, "compressed", None)
        if not self._is_usable(compressed, content=content, tool_name=tool_name):
            return None, strategy, False
        # Only an explicit, entirely lossless chain proves a tiny result is safe.
        # A strategy name alone ("text", "log", ...) says nothing about data loss.
        chain = getattr(result, "strategy_chain", None)
        lossless = isinstance(chain, (list, tuple)) and bool(chain) and all(
            isinstance(step, str) and step.startswith("lossless_") for step in chain
        )
        return compressed, strategy, lossless

    def _log_candidates(
        self,
        *,
        content: str,
        tool_name: str,
        limit: int,
        routed_strategy: str,
    ) -> Iterator[str]:
        """Yield deduped then verbatim-warning candidates, lazily and only for logs."""
        for dedupe_warnings in (True, False):
            compressor = self._get_log_compressor(limit, dedupe_warnings=dedupe_warnings)
            if compressor is None:
                return
            try:
                result = compressor.compress(content)
            except Exception:
                logger.debug("headroom log compression failed for tool %s", tool_name, exc_info=True)
                continue

            # Detect once: a generic non-log never benefits from a second pass.
            # Missing format metadata is not evidence that prose is a build log.
            log_format = getattr(getattr(result, "format_detected", None), "value", "")
            if routed_strategy != _LOG_STRATEGY and log_format in ("", _GENERIC_LOG_FORMAT):
                return
            compressed = getattr(result, "compressed", None)
            if self._is_usable(compressed, content=content, tool_name=tool_name):
                yield compressed

    @staticmethod
    def _is_usable(compressed: Any, *, content: str, tool_name: str) -> bool:
        """True when a compressor returned content that can replace the original."""
        if not isinstance(compressed, str) or not compressed.strip():
            return False
        if len(compressed) >= len(content):
            return False
        # Retrieval markers need headroom's retrieval tool, which this runtime does not expose.
        if _CCR_MARKER_RE.search(compressed):
            logger.debug("headroom returned an unresolved retrieval marker for tool %s", tool_name)
            return False
        # Diagnostics anywhere in the body are not enough: the executor parses
        # ERROR[TYPE] only at the start. Never turn a failed tool into a success
        # (or change its retryability) by moving or rewriting that envelope.
        if (
            parse_tool_execution_result(content).error_type
            != parse_tool_execution_result(compressed).error_type
        ):
            return False
        return True

    def _preserves_diagnostics(
        self,
        *,
        diagnostics: list[frozenset[str]],
        baseline_kept: int,
        compressed: str,
        body_length: int,
        tool_name: str,
        limit: int,
        is_mcp: bool,
        lossless: bool,
    ) -> bool:
        """Compare diagnostics on the delivered text, not an oversized candidate.

        Without diagnostics, only explicit lossless provenance may bypass the
        minimum useful-content budget. A footer cannot make an omission-only
        summary useful. Structured records are compared by significant tokens
        so JSON-to-CSV reserialization remains acceptable.
        """
        if not diagnostics:
            return lossless or body_length * _MIN_BUDGET_FRACTION >= limit
        delivered = self.reduce_to_limit(
            content=compressed, tool_name=tool_name, limit=limit, is_mcp=is_mcp
        )
        return self._diagnostics_kept(diagnostics, delivered) >= baseline_kept

    @staticmethod
    def _diagnostic_tokens(content: str) -> list[frozenset[str]]:
        lines = {
            line.strip()
            for line in content.splitlines()
            if line.strip() and _DIAGNOSTIC_LINE_RE.search(line)
        }
        return [frozenset(_SIGNIFICANT_TOKEN_RE.findall(line)) for line in lines]

    @staticmethod
    def _diagnostics_kept(diagnostics: list[frozenset[str]], text: str) -> int:
        tokens = set(_SIGNIFICANT_TOKEN_RE.findall(text))
        return sum(1 for line_tokens in diagnostics if line_tokens <= tokens)

    def reduce_to_limit(
        self,
        *,
        content: str,
        tool_name: str,
        limit: int,
        is_mcp: bool = False,
    ) -> str:
        """Fit output while preserving coverage of structured sections when possible."""
        if not content or len(content) <= limit:
            return content
        if limit <= 0:
            return truncate_output(content, limit, source=tool_name)

        prefix = _SECTION_PREFIXES.get(tool_name)
        if prefix:
            reduced = self._reduce_sections(content, prefix=prefix, limit=limit, tool_name=tool_name)
            if reduced is not None:
                return reduced
        if tool_name == "cli_exec":
            return self._reduce_cli(content, limit=limit)
        if is_mcp:
            return self._middle_fit(content, limit)
        return truncate_output(content, limit, source=tool_name)

    @staticmethod
    def _middle_fit(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        if limit <= 0:
            return ""
        omitted = len(text) - limit
        while True:
            marker = f"\n... [OMITTED {omitted} chars] ...\n"
            if len(marker) >= limit:
                return text[:limit]
            kept = limit - len(marker)
            actual_omitted = len(text) - kept
            if actual_omitted == omitted:
                break
            omitted = actual_omitted
        head = (kept + 1) // 2
        tail = kept - head
        return text[:head] + marker + (text[-tail:] if tail else "")

    @classmethod
    def _reduce_sections(cls, content: str, *, prefix: str, limit: int, tool_name: str) -> Optional[str]:
        starts = [match.start() for match in re.finditer(rf"(?m)^{re.escape(prefix)}", content)]
        if not starts:
            return None

        preamble = content[: starts[0]].strip()
        sections = [
            content[start : starts[index + 1] if index + 1 < len(starts) else len(content)].strip()
            for index, start in enumerate(starts)
        ]
        marker = (
            f"[REDUCED preserving {len(sections)} sections | source={tool_name}]\n"
            f"... [TRUNCATED from {len(content)} chars | source={tool_name}]"
        )
        separator = "\n\n"
        preamble_budget = min(len(preamble), max(0, limit // 12)) if preamble else 0
        rendered_count = 1 + len(sections) + (1 if preamble_budget else 0)
        fixed = len(marker) + len(separator) * max(0, rendered_count - 1)
        if fixed >= limit:
            return truncate_output(content, limit, source=tool_name)
        available = limit - fixed
        if preamble_budget:
            preamble_budget = min(preamble_budget, available)
            available -= preamble_budget
        per_section = available // max(1, len(sections))
        remainder = available % max(1, len(sections))

        rendered = []
        if preamble_budget:
            rendered.append(cls._middle_fit(preamble, preamble_budget))
        rendered.extend(
            cls._middle_fit(section, per_section + (1 if index < remainder else 0))
            for index, section in enumerate(sections)
        )
        rendered.append(marker)
        return separator.join(rendered)

    @classmethod
    def _reduce_cli(cls, content: str, *, limit: int) -> str:
        lines = content.splitlines()
        diagnostics = []
        seen = set()
        for line in lines:
            stripped = line.strip()
            if stripped and _DIAGNOSTIC_LINE_RE.search(stripped) and stripped not in seen:
                seen.add(stripped)
                diagnostics.append(stripped)

        marker = (
            f"[REDUCED shell output | {len(content)} chars | head+diagnostics+tail]\n"
            f"... [TRUNCATED from {len(content)} chars | source=cli_exec]"
        )
        diagnostics_text = "\n".join(diagnostics)
        diagnostics_label = "[diagnostics]\n" if diagnostics_text else ""
        tail_label = "[tail]\n"
        separator_count = 3 if diagnostics_text else 2
        fixed = len(marker) + len(diagnostics_label) + len(tail_label) + separator_count
        if fixed >= limit:
            return truncate_output(content, limit, source="cli_exec")
        available = limit - fixed
        diagnostics_budget = min(len(diagnostics_text), max(0, available // 5))
        remaining = max(0, available - diagnostics_budget)
        head_budget = remaining // 2
        tail_budget = remaining - head_budget

        parts = [content[:head_budget]]
        if diagnostics_budget:
            parts.append(diagnostics_label + cls._middle_fit(diagnostics_text, diagnostics_budget))
        if tail_budget:
            parts.append(tail_label + content[-tail_budget:])
        parts.append(marker)
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _build_query(tool_name: str, tool_args: Optional[Dict[str, Any]]) -> str:
        if not tool_args:
            return tool_name
        parts: list[str] = []
        for key in (
            "query",
            "queries",
            "command",
            "path",
            "url",
            "urls",
            "pattern",
            "instructions",
            "select_paths",
            "select_domains",
        ):
            value = tool_args.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
            elif isinstance(value, (list, tuple)):
                rendered = " ".join(str(item).strip() for item in value if str(item).strip())
                if rendered:
                    parts.append(rendered)
        query = " ".join(parts) if parts else tool_name
        return query[:4000]
