"""Regression coverage for budget-aware Headroom candidate selection."""

import unittest
from types import SimpleNamespace
from unittest import mock

from core.tool_output_compressor import ToolOutputCompressor
from core.tool_results import parse_tool_execution_result


def routed_result(body, strategy="log", chain=()):
    return SimpleNamespace(
        compressed=body,
        strategy_used=SimpleNamespace(value=strategy),
        strategy_chain=list(chain),
    )


def log_result(body, log_format="generic"):
    return SimpleNamespace(compressed=body, format_detected=SimpleNamespace(value=log_format))


class ToolOutputCompressorTests(unittest.TestCase):
    def setUp(self):
        self.compressor = ToolOutputCompressor(enabled=True)
        self.content = "INFO compiling module\n" * 400 + "ERROR: shard-alpha unavailable\n"

    def compress(self, content=None, limit=1000):
        return self.compressor.compress(
            content=self.content if content is None else content,
            tool_name="cli_exec",
            tool_args=None,
            limit=limit,
        )

    def test_accepted_log_candidate_does_not_run_second_compressor(self):
        router = mock.Mock()
        router.compress.return_value = routed_result(self.content[:-1])
        log = mock.Mock()
        log.compress.return_value = log_result("ERROR: shard-alpha unavailable")
        with (
            mock.patch.object(self.compressor, "_get_router", return_value=router),
            mock.patch.object(self.compressor, "_get_log_compressor", return_value=log) as get_log,
        ):
            result = self.compress()
        self.assertIn("shard-alpha unavailable", result)
        get_log.assert_called_once_with(1000, dedupe_warnings=True)
        log.compress.assert_called_once()

    def test_short_rejected_route_can_be_rescued_by_log_compression(self):
        router = mock.Mock()
        router.compress.return_value = routed_result("[400 lines omitted]")
        log = mock.Mock()
        log.compress.return_value = log_result("ERROR: shard-alpha unavailable")
        with (
            mock.patch.object(self.compressor, "_get_router", return_value=router),
            mock.patch.object(self.compressor, "_get_log_compressor", return_value=log),
        ):
            result = self.compress()
        self.assertIn("ERROR: shard-alpha unavailable", result)
        self.assertLessEqual(len(result), 1000)

    def test_lost_warning_identifiers_retry_without_dedupe(self):
        content = self.content + "WARNING: package-100 drift\nWARNING: package-200 drift\n"
        router = mock.Mock()
        router.compress.return_value = routed_result(content)
        deduped, verbatim = mock.Mock(), mock.Mock()
        deduped.compress.return_value = log_result("ERROR: shard-alpha unavailable\nWARNING: package-100 drift")
        verbatim.compress.return_value = log_result(
            "ERROR: shard-alpha unavailable\nWARNING: package-100 drift\nWARNING: package-200 drift"
        )
        with (
            mock.patch.object(self.compressor, "_get_router", return_value=router),
            mock.patch.object(self.compressor, "_get_log_compressor", side_effect=[deduped, verbatim]) as get_log,
            mock.patch.object(self.compressor, "_diagnostic_tokens", wraps=self.compressor._diagnostic_tokens) as profile,
            mock.patch.object(self.compressor, "reduce_to_limit", wraps=self.compressor.reduce_to_limit) as reduce,
        ):
            result = self.compress(content)
        self.assertIn("package-200 drift", result)
        self.assertEqual(get_log.call_args_list, [
            mock.call(1000, dedupe_warnings=True), mock.call(1000, dedupe_warnings=False)
        ])
        profile.assert_called_once_with(content)
        source_reductions = [call for call in reduce.call_args_list if call.kwargs["content"] == content]
        self.assertEqual(len(source_reductions), 1)

    def test_non_log_probe_does_not_run_twice(self):
        router = mock.Mock()
        router.compress.return_value = routed_result("x" * 5000, strategy="text")
        log = mock.Mock()
        log.compress.return_value = log_result("[400 lines omitted]")
        with (
            mock.patch.object(self.compressor, "_get_router", return_value=router),
            mock.patch.object(self.compressor, "_get_log_compressor", return_value=log) as get_log,
        ):
            self.assertIsNone(self.compress("x" * 5000))
        get_log.assert_called_once()
        log.compress.assert_called_once()

    def test_missing_log_format_is_not_evidence_of_build_output(self):
        router = mock.Mock()
        router.compress.return_value = routed_result(self.content, strategy="text")
        log = mock.Mock()
        log.compress.return_value = SimpleNamespace(compressed="ERROR: shard-alpha unavailable")
        with (
            mock.patch.object(self.compressor, "_get_router", return_value=router),
            mock.patch.object(self.compressor, "_get_log_compressor", return_value=log),
        ):
            self.assertIsNone(self.compress())
        log.compress.assert_called_once()

    def test_known_build_format_can_rescue_non_log_route(self):
        router = mock.Mock()
        router.compress.return_value = routed_result(self.content, strategy="text")
        log = mock.Mock()
        log.compress.return_value = log_result("ERROR: shard-alpha unavailable", log_format="make")
        with (
            mock.patch.object(self.compressor, "_get_router", return_value=router),
            mock.patch.object(self.compressor, "_get_log_compressor", return_value=log),
        ):
            self.assertIn("shard-alpha unavailable", self.compress())

    def test_explicit_lossless_fold_can_be_smaller_than_budget_floor(self):
        router = mock.Mock()
        router.compress.return_value = routed_result(
            "INFO worker completed task [repeated 400 times]", chain=["lossless_log"]
        )
        with (
            mock.patch.object(self.compressor, "_get_router", return_value=router),
            mock.patch.object(self.compressor, "_get_log_compressor") as get_log,
        ):
            result = self.compress("INFO worker completed task\n" * 400, limit=4000)
        self.assertIn("repeated 400 times", result)
        self.assertLess(len(result), 4000 // 4)
        get_log.assert_not_called()

    def test_lossless_then_lossy_chain_cannot_bypass_budget_floor(self):
        router = mock.Mock()
        router.compress.return_value = routed_result(
            "[400 lines omitted]", strategy="text", chain=["lossless_text", "kompress"]
        )
        with (
            mock.patch.object(self.compressor, "_get_router", return_value=router),
            mock.patch.object(self.compressor, "_get_log_compressor", return_value=None),
        ):
            self.assertIsNone(self.compress("x" * 5000))

    def test_footer_is_not_counted_as_useful_content(self):
        router = mock.Mock()
        router.compress.return_value = routed_result("x" * 190, strategy="text")
        with (
            mock.patch.object(self.compressor, "_get_router", return_value=router),
            mock.patch.object(self.compressor, "_get_log_compressor", return_value=None),
        ):
            self.assertIsNone(self.compress("x" * 5000))

    def test_footer_must_not_erase_net_savings(self):
        router = mock.Mock()
        router.compress.return_value = routed_result("x" * 4999, strategy="text")
        with (
            mock.patch.object(self.compressor, "_get_router", return_value=router),
            mock.patch.object(self.compressor, "_get_log_compressor", return_value=None),
        ):
            self.assertIsNone(self.compress("x" * 5000))

    def test_footer_over_budget_triggers_log_pass(self):
        router = mock.Mock()
        router.compress.return_value = routed_result("INFO padding\n" * 70 + "ERROR: shard-alpha unavailable")
        self.assertLess(len(router.compress.return_value.compressed), 1000)
        log = mock.Mock()
        log.compress.return_value = log_result("ERROR: shard-alpha unavailable")
        with (
            mock.patch.object(self.compressor, "_get_router", return_value=router),
            mock.patch.object(self.compressor, "_get_log_compressor", return_value=log),
        ):
            result = self.compress()
        log.compress.assert_called_once()
        self.assertLessEqual(len(result), 1000)
        self.assertNotIn("padding", result)

    def test_error_envelope_cannot_move_or_change_type(self):
        content = "ERROR[NETWORK]: unavailable\n" + "detail\n" * 500
        for candidate in (
            "summary\nERROR[NETWORK]: unavailable",
            "ERROR[TIMEOUT]: unavailable",
            "NETWORK error: unavailable",
        ):
            with self.subTest(candidate=candidate):
                self.assertFalse(self.compressor._is_usable(candidate, content=content, tool_name="cli_exec"))
        candidate = "ERROR[NETWORK]: unavailable\n[repeated detail]"
        self.assertTrue(self.compressor._is_usable(candidate, content=content, tool_name="cli_exec"))
        self.assertTrue(parse_tool_execution_result(candidate).retryable)

    def test_compression_cannot_introduce_an_error_envelope(self):
        self.assertFalse(self.compressor._is_usable(
            "ERROR[EXECUTION]: extracted example", content="example text\n" * 500, tool_name="cli_exec"
        ))

    def test_nonpositive_budget_skips_headroom(self):
        with mock.patch.object(self.compressor, "_get_router") as get_router:
            self.assertIsNone(self.compress(limit=0))
            self.assertIsNone(self.compress(limit=-1))
        get_router.assert_not_called()

    def test_real_headroom_folds_repetition_without_diagnostics(self):
        result = self.compress("INFO worker completed task successfully\n" * 4000, limit=15000)
        self.assertIsNotNone(result)
        self.assertIn("worker completed task successfully", result)
        self.assertLess(len(result), 15000 // 4)
        self.assertNotIn("<<ccr:", result)

    def test_real_log_pass_recovers_all_errors_after_lossy_route(self):
        content = "\n".join(
            f"ERROR: module_{index}.py:{index} undefined reference to symbol_{index}"
            if index % 10 == 0
            else f"[{index:04d}] INFO compiling module_{index}.py ... ok"
            for index in range(400)
        )
        router = mock.Mock()
        router.compress.return_value = routed_result("\n".join(content.splitlines()[:10]))
        with mock.patch.object(self.compressor, "_get_router", return_value=router):
            result = self.compress(content, limit=15000)
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 15000)
        for index in range(0, 400, 10):
            self.assertIn(f"undefined reference to symbol_{index}", result)

    def test_log_exception_retries_without_dedupe(self):
        router = mock.Mock()
        router.compress.return_value = routed_result(self.content)
        deduped, verbatim = mock.Mock(), mock.Mock()
        deduped.compress.side_effect = RuntimeError("compression unavailable")
        verbatim.compress.return_value = log_result("ERROR: shard-alpha unavailable")
        with (
            mock.patch.object(self.compressor, "_get_router", return_value=router),
            mock.patch.object(self.compressor, "_get_log_compressor", side_effect=[deduped, verbatim]),
        ):
            result = self.compress()
        self.assertIn("shard-alpha unavailable", result)
        deduped.compress.assert_called_once()
        verbatim.compress.assert_called_once()

    def test_router_and_log_compressors_remain_cached_and_marker_free(self):
        router = self.compressor._get_router(1000)
        self.assertIs(router, self.compressor._get_router(1000))
        self.assertFalse(router.config.ccr_enabled)
        self.assertFalse(router.config.ccr_inject_marker)
        for dedupe in (True, False):
            compressor = self.compressor._get_log_compressor(1000, dedupe_warnings=dedupe)
            self.assertIs(compressor, self.compressor._get_log_compressor(1000, dedupe_warnings=dedupe))
            self.assertFalse(compressor.config.enable_ccr)
