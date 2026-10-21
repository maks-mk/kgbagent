import json
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from pydantic import ValidationError

from core.nodes.tool_preflight import ToolPreflightMixin
from core.safety_policy import SafetyPolicy
from core.tool_policy import ToolMetadata
from core.tool_results import parse_tool_execution_result
from core.utils import truncate_output
from tools import filesystem
from tools.filesystem_impl import editing
from tools.filesystem_impl.manager import FilesystemManager
from tools.filesystem_impl.pathing import atomic_write_text, count_file_lines, is_binary_path


@pytest.fixture
def workspace():
    root = Path(__file__).resolve().parents[1] / ".tmp_tests"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="filesystem-regression-", dir=root) as name:
        yield Path(name)


@pytest.fixture
def manager(workspace):
    backend = FilesystemManager(workspace)
    backend.set_policy(SafetyPolicy(max_tool_output=15000, max_read_lines=2000))
    return backend


@pytest.fixture
def tools(manager):
    with mock.patch.object(filesystem, "_get_synced_backend", return_value=manager):
        yield filesystem


@pytest.mark.parametrize("mode", ["exact", "indentation"])
def test_edit_does_not_match_different_internal_whitespace(workspace, mode):
    target = workspace / "sample.py"
    original = b'value = "hello world"\n'
    target.write_bytes(original)
    result = editing.edit_text_file(target, "sample.py", 'value="helloworld"', 'value="changed"', match_mode=mode)
    assert result.startswith("ERROR[VALIDATION]")
    assert target.read_bytes() == original


def test_edit_rejects_invalid_utf8_without_writing(workspace):
    target = workspace / "sample.txt"
    original = b'\xff\nvalue=1\n'
    target.write_bytes(original)
    with mock.patch.object(editing, "atomic_write_text") as write:
        result = editing.edit_text_file(target, "sample.txt", "value=1", "value=2")
    assert "not valid UTF-8" in result
    write.assert_not_called()
    assert target.read_bytes() == original


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_explicit_indentation_preserves_nested_indentation(workspace, newline):
    target = workspace / "sample.py"
    target.write_bytes(newline.join(["def run():", "    if True:", "        old()", "    end()", ""]).encode())
    result = editing.edit_text_file(target, "sample.py", "if True:\n    old()\n", "if True:\n    new()\n", match_mode="indentation")
    assert result.startswith("Success:")
    assert "Explicit indentation" in result
    assert target.read_bytes() == newline.join(["def run():", "    if True:", "        new()", "    end()", ""]).encode()


@pytest.mark.parametrize("old", ["if True:\n  old()", "if True: \n    old()", "if  True:\n    old()"])
def test_indentation_mode_does_not_ignore_relative_or_trailing_whitespace(workspace, old):
    target = workspace / "sample.py"
    original = b"def run():\n    if True:\n        old()\n"
    target.write_bytes(original)
    result = editing.edit_text_file(target, "sample.py", old, "new()", match_mode="indentation")
    assert result.startswith("ERROR[VALIDATION]")
    assert target.read_bytes() == original


@pytest.mark.parametrize("original,old", [
    (b"first()\nsecond()\n", "    first()\nsecond()"),
    (b"\tfirst()\nsecond()\n", "    first()\nsecond()"),
    (b"    first()\n   second()\n", "\tfirst()\n\tsecond()"),
])
def test_indentation_rejects_unrepresentable_relative_dedent(workspace, original, old):
    target = workspace / "sample.py"
    target.write_bytes(original)
    with mock.patch.object(editing, "atomic_write_text") as write:
        result = editing.edit_text_file(target, "sample.py", old, "changed()", match_mode="indentation")
    assert result.startswith("ERROR[VALIDATION]")
    write.assert_not_called()
    assert target.read_bytes() == original


def test_indentation_preserves_tab_nested_and_blank_lines(workspace):
    target = workspace / "sample.py"
    target.write_bytes(b"\tif True:\n\t\told()  \n\t\n")
    result = editing.edit_text_file(
        target, "sample.py", "if True:\n\told()  ", "if True:\n\tnew()  ",
        match_mode="indentation",
    )
    assert result.startswith("Success:")
    assert target.read_bytes() == b"\tif True:\n\t\tnew()  \n\t\n"


def test_indentation_skips_unsafe_candidate_before_valid_block(workspace):
    target = workspace / "sample.py"
    target.write_bytes(b"first()\nsecond()\n\n        first()\n    second()\n")
    result = editing.edit_text_file(
        target, "sample.py", "    first()\nsecond()", "    changed()\nend()",
        match_mode="indentation",
    )
    assert result.startswith("Success:")
    assert target.read_bytes() == b"first()\nsecond()\n\n        changed()\n    end()\n"


def test_indentation_rejects_unrepresentable_replacement(workspace):
    target = workspace / "sample.py"
    original = b"  first()\n  second()\n"
    target.write_bytes(original)
    with mock.patch.object(editing, "atomic_write_text") as write:
        result = editing.edit_text_file(
            target, "sample.py", "first()\nsecond()", "    changed()\nend()",
            match_mode="indentation",
        )
    assert result.startswith("ERROR[VALIDATION]")
    write.assert_not_called()
    assert target.read_bytes() == original


def test_batch_edits_are_ordered_and_write_once(workspace):
    target = workspace / "sample.txt"
    original = b"first\nsecond\n"
    target.write_bytes(original)
    operations = [
        {"old_string": "first", "new_string": "intermediate"},
        {"old_string": "intermediate", "new_string": "final"},
        {"old_string": "second\n", "new_string": ""},
    ]
    with mock.patch.object(editing, "atomic_write_text", wraps=atomic_write_text) as write:
        result = editing.edit_text_file(target, "sample.txt", edits=operations)
    assert "3 replacement(s)" in result
    write.assert_called_once_with(target, "final\n", expected_content=original)
    assert target.read_bytes() == b"final\n"


@pytest.mark.parametrize("late_edit", [
    {"old_string": "missing", "new_string": "x"},
    {"old_string": "second", "new_string": None},
    {"old_string": "", "new_string": "x"},
    {"old_string": "second", "new_string": "x", "path": "other.txt"},
])
def test_late_invalid_edit_writes_nothing(workspace, late_edit):
    target = workspace / "sample.txt"
    original = b"first\nsecond\n"
    target.write_bytes(original)
    with mock.patch.object(editing, "atomic_write_text") as write:
        result = editing.edit_text_file(target, "sample.txt", edits=[{"old_string": "first", "new_string": "changed"}, late_edit])
    assert result.startswith("ERROR[VALIDATION]")
    write.assert_not_called()
    assert target.read_bytes() == original


def test_batch_checks_only_final_json(workspace):
    target = workspace / "sample.json"
    target.write_bytes(b'{"a": 1}')
    result = editing.edit_text_file(target, "sample.json", edits=[
        {"old_string": "1", "new_string": "TEMP"},
        {"old_string": "TEMP", "new_string": "2"},
    ])
    assert result.startswith("Success:")
    assert json.loads(target.read_bytes()) == {"a": 2}
    before = target.read_bytes()
    result = editing.edit_text_file(target, "sample.json", "2", "INVALID")
    assert "invalid JSON" in result
    assert target.read_bytes() == before


def test_concurrent_change_is_not_overwritten(workspace):
    target = workspace / "sample.txt"
    target.write_bytes(b"original")

    def concurrent_write(path, content, **kwargs):
        path.write_bytes(b"external change")
        atomic_write_text(path, content, **kwargs)

    with mock.patch.object(editing, "atomic_write_text", side_effect=concurrent_write):
        result = editing.edit_text_file(target, "sample.txt", "original", "ours")
    assert "File changed" in result
    assert target.read_bytes() == b"external change"
    assert sorted(path.name for path in workspace.iterdir()) == ["sample.txt"]


def test_edit_preserves_bom_and_untouched_mixed_newlines(workspace):
    target = workspace / "sample.txt"
    original = b"\xef\xbb\xbfhead\r\nvalue=1\ntail\r\n"
    target.write_bytes(original)
    result = editing.edit_text_file(target, "sample.txt", "value=1", "value=2")
    assert result.startswith("Success:")
    assert target.read_bytes() == original.replace(b"value=1", b"value=2")


def test_repeated_matches_need_replace_all(workspace):
    target = workspace / "sample.txt"
    target.write_bytes(b"same same")
    result = editing.edit_text_file(target, "sample.txt", "same", "new")
    assert result.startswith("ERROR[VALIDATION]")
    assert target.read_bytes() == b"same same"
    result = editing.edit_text_file(target, "sample.txt", "same", "new", replace_all=True)
    assert "2 replacement(s)" in result
    assert target.read_bytes() == b"new new"


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
@pytest.mark.parametrize("replace_all", [False, True])
def test_exact_match_starting_with_newline(workspace, newline, replace_all):
    target = workspace / "sample.txt"
    original = b"head" + newline + b"old"
    if replace_all:
        original += newline + b"old"
    target.write_bytes(original)
    result = editing.edit_text_file(
        target, "sample.txt", "\nold", "\nnew", replace_all=replace_all,
    )
    assert result.startswith("Success:")
    assert target.read_bytes() == original.replace(b"old", b"new")


@pytest.mark.parametrize("replace_all", [False, True])
def test_overlapping_exact_matches_are_rejected(workspace, replace_all):
    target = workspace / "sample.txt"
    target.write_bytes(b"aaa")
    result = editing.edit_text_file(target, "sample.txt", "aa", "b", replace_all=replace_all)
    assert result.startswith("ERROR[VALIDATION]")
    assert target.read_bytes() == b"aaa"


def test_numbered_read_can_be_edited(tools, workspace):
    target = workspace / "sample.py"
    target.write_bytes(b"def run():\n    old()\n")
    result = tools.edit_file_tool.invoke({"path": "sample.py", "edits": [{
        "old_string": "     1  def run():\n     2      old()",
        "new_string": "     1  def run():\n     2      new()",
    }]})
    assert "line-number prefixes were removed" in result
    assert target.read_bytes() == b"def run():\n    new()\n"


@pytest.mark.parametrize("original,replacement", [
    ("1  old\n", "2  new\n"),
    ("     1  old()\n", "     2  new()\n"),
    ("42  old\nplain\n", "43  new\nplain\n"),
])
def test_literal_numbered_text_is_matched_before_prefix_stripping(workspace, original, replacement):
    target = workspace / "sample.txt"
    target.write_bytes(original.encode())
    result = editing.edit_text_file(target, "sample.txt", original, replacement)
    assert result.startswith("Success:")
    assert "line-number prefixes" not in result
    assert target.read_bytes() == replacement.encode()


@pytest.mark.parametrize("replacement", ["123  value\nplain\n", "     1  literal\n", "1  first\n2  second\n"])
def test_plain_edit_preserves_numeric_replacement(workspace, replacement):
    target = workspace / "sample.txt"
    target.write_bytes(b"old\n")
    result = editing.edit_text_file(target, "sample.txt", "old\n", replacement)
    assert result.startswith("Success:")
    assert "line-number prefixes" not in result
    assert target.read_bytes() == replacement.encode()


def test_literal_numbered_match_is_not_confused_with_unnumbered_occurrence(workspace):
    target = workspace / "sample.txt"
    target.write_bytes(b"     1  old()\nold()\n")
    result = editing.edit_text_file(target, "sample.txt", "     1  old()", "changed()")
    assert result.startswith("Success:")
    assert target.read_bytes() == b"changed()\nold()\n"


@pytest.mark.parametrize("old", [
    "1  old()\n2  next()\n",
    "     1  old()\n     3  next()\n",
    "     1  old()\nnext()\n",
    "     1\t\told()\n     2\t\tnext()\n",
])
def test_incomplete_or_nonstandard_numbering_is_not_stripped(workspace, old):
    target = workspace / "sample.txt"
    original = b"old()\nnext()\n"
    target.write_bytes(original)
    with mock.patch.object(editing, "atomic_write_text") as write:
        result = editing.edit_text_file(target, "sample.txt", old, "changed()\n")
    assert result.startswith("ERROR[VALIDATION]")
    write.assert_not_called()
    assert target.read_bytes() == original


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_numbered_read_preserves_blank_lines_and_plain_replacement(workspace, newline):
    target = workspace / "sample.txt"
    target.write_bytes(newline.join(["head", "old()", "", "next()", ""]).encode())
    result = editing.edit_text_file(
        target, "sample.txt", newline.join(["     2  old()", "     3  ", "     4  next()", ""]),
        "123  literal\n\nchanged()\n",
    )
    assert result.startswith("Success:")
    assert "line-number prefixes were removed" in result
    assert target.read_bytes() == newline.join(["head", "123  literal", "", "changed()", ""]).encode()


@pytest.mark.parametrize("mode", ["exact", "indentation"])
@pytest.mark.parametrize("actual,requested", [
    ('value = "hello world"', 'value="helloworld"'),
    ("\told()  ", "\told()"),
])
def test_match_failure_reports_whitespace_difference_without_writing(workspace, mode, actual, requested):
    target = workspace / "sample.py"
    original = ("head\r\n" + actual + "\r\n").encode()
    target.write_bytes(original)
    with mock.patch.object(editing, "atomic_write_text") as write:
        result = editing.edit_text_file(target, "sample.py", requested + "\n", "changed()\n", match_mode=mode)
    assert result.startswith("ERROR[VALIDATION]")
    assert "Possible target starts at line 2" in result
    assert repr(requested) in result
    assert repr(actual) in result
    assert "show_line_numbers=false" in result
    write.assert_not_called()
    assert target.read_bytes() == original


def test_match_failure_reports_leading_indentation_without_enabling_fallback(workspace):
    target = workspace / "sample.py"
    original = b"def run():\n\tif True:\n\t\told()\n"
    target.write_bytes(original)
    result = editing.edit_text_file(target, "sample.py", "if True:\n\told()\n", "changed()\n")
    assert result.startswith("ERROR[VALIDATION]")
    assert "Possible target starts at line 2" in result
    assert repr("\tif True:") in result
    assert target.read_bytes() == original


def test_match_failure_explains_missing_final_newline(workspace):
    target = workspace / "sample.txt"
    target.write_bytes(b"old()")
    result = editing.edit_text_file(target, "sample.txt", "old()\n", "new()\n")
    assert result.startswith("ERROR[VALIDATION]")
    assert "file=<EOF>" in result
    assert target.read_bytes() == b"old()"


def test_match_failure_does_not_choose_between_similar_blocks(workspace):
    target = workspace / "sample.py"
    original = b"    old()\n    old()\n"
    target.write_bytes(original)
    result = editing.edit_text_file(target, "sample.py", "old()\n\n", "new()\n")
    assert result.startswith("ERROR[VALIDATION]")
    assert "Possible target blocks start at lines 1, 2" in result
    assert target.read_bytes() == original


@pytest.mark.parametrize("replace_all", [False, True])
def test_numbered_read_still_requires_unique_match_or_replace_all(workspace, replace_all):
    target = workspace / "sample.txt"
    original = b"old()\nold()\n"
    target.write_bytes(original)
    result = editing.edit_text_file(
        target, "sample.txt", "     1  old()", "     1  new()", replace_all=replace_all,
    )
    if replace_all:
        assert "2 replacement(s)" in result
        assert target.read_bytes() == b"new()\nnew()\n"
    else:
        assert result.startswith("ERROR[VALIDATION]")
        assert "2 identical occurrences" in result
        assert target.read_bytes() == original


def test_stale_block_hint_shows_changed_code_without_applying_it(workspace):
    target = workspace / "sample.py"
    original = b"# header\ndef calculate_total():\n    return 2\n"
    target.write_bytes(original)
    result = editing.edit_text_file(
        target, "sample.py", "def calculate_total():\n    return 1\n", "def calculate_total():\n    return 3\n",
    )
    assert result.startswith("ERROR[VALIDATION]")
    assert "Possible target starts at line 2" in result
    assert "First differing line 3" in result
    assert "old_string='    return 1'; file='    return 2'" in result
    assert target.read_bytes() == original


def test_missing_anchor_requests_reread_without_inventing_a_location(workspace):
    target = workspace / "sample.txt"
    target.write_bytes(b"unrelated text\n")
    result = editing.edit_text_file(target, "sample.txt", "missing text\n", "new text\n")
    assert result.startswith("ERROR[VALIDATION]")
    assert "show_line_numbers=false" in result
    assert "Possible target" not in result
    assert target.read_bytes() == b"unrelated text\n"


def test_match_failure_diagnostics_are_bounded(workspace):
    target = workspace / "sample.txt"
    original = ("head\n" + "x" * 10000 + "  \n").encode()
    target.write_bytes(original)
    result = editing.edit_text_file(target, "sample.txt", "x" * 10000 + "\n", "new\n")
    assert result.startswith("ERROR[VALIDATION]")
    assert "Possible target starts at line 2" in result
    assert len(result) < 1500
    assert target.read_bytes() == original


def test_late_match_failure_diagnostics_describe_unwritten_batch_state(workspace):
    target = workspace / "sample.py"
    original = b"first()\n    old()\n"
    target.write_bytes(original)
    result = editing.edit_text_file(target, "sample.py", edits=[
        {"old_string": "first()", "new_string": "first()\ninserted()"},
        {"old_string": "old()\n\n", "new_string": "changed()\n"},
    ])
    assert result.startswith("ERROR[VALIDATION]: Edit 2:")
    assert "Possible target starts at line 3" in result
    assert "after preceding edits in memory" in result
    assert "No changes were written" in result
    assert target.read_bytes() == original


@pytest.mark.parametrize("alias", ["old_text", "search_text", "from", "from_"])
def test_nested_edit_aliases(tools, workspace, alias):
    target = workspace / "sample.txt"
    target.write_bytes(b"old")
    result = tools.edit_file_tool.invoke({"file_path": "sample.txt", "edits": [{alias: "old", "new_text": "new"}]})
    assert result.startswith("Success:")
    assert target.read_bytes() == b"new"


def test_single_and_batch_read_aliases(tools, workspace):
    (workspace / "sample.txt").write_bytes(b"first\nsecond\nthird\n")
    assert tools.read_file_tool.invoke({"file_path": "sample.txt"}).startswith("first\nsecond\nthird")
    result = tools.read_file_tool.invoke({"requests": [{"filepath": "sample.txt", "offset": 1, "limit": 1, "show_line_numbers": True}]})
    assert "     2  second" in result
    assert "lines=2-2 of 3" in result
    assert 'offset=2' in result
    assert "status=TRUNCATED" in result


@pytest.mark.parametrize("separator", [b"\n", b"\r", b"\r\n"])
@pytest.mark.parametrize("trailing", [False, True])
def test_read_newlines_and_continuation(manager, workspace, separator, trailing):
    target = workspace / "sample.txt"
    target.write_bytes(separator.join([b"first", b"second", b"third"]) + (separator if trailing else b""))
    assert count_file_lines(target) == 3
    result = manager.read_file("sample.txt", limit=1, show_line_numbers=True)
    assert "     1  first" in result
    assert "of 3" in result and "offset=1" in result
    result = manager.read_file("sample.txt", offset=1, show_line_numbers=True)
    assert result.startswith("     2  second\n     3  third")
    assert "[EOF] Lines 2-3 of 3" in result
    result = manager.read_files([{"path": "sample.txt", "offset": 1, "limit": 1}])
    assert "lines=2-2 of 3" in result and "offset=2" in result


@pytest.mark.parametrize("prefix_size", [65534, 65535, 65536])
def test_line_count_normalizes_crlf_across_chunk_boundary(workspace, prefix_size):
    target = workspace / "sample.txt"
    target.write_bytes(b"x" * prefix_size + b"\r\nsecond\rthird\n")
    assert count_file_lines(target) == 3


@pytest.mark.parametrize("separator", [b"\n", b"\r", b"\r\n"])
def test_read_blank_lines_preserves_offsets(manager, workspace, separator):
    target = workspace / "sample.txt"
    target.write_bytes(separator * 2 + b"last")
    assert count_file_lines(target) == 3
    result = manager.read_file("sample.txt", limit=2, show_line_numbers=True)
    assert result.startswith("     1  \n     2  ")
    assert "offset=2" in result
    assert manager.read_file("sample.txt", offset=2).startswith("last")


def test_read_multibyte_utf8_across_chunk_boundary(manager, workspace):
    text = "x" * 65535 + "Привет\nlast"
    target = workspace / "sample.txt"
    target.write_bytes(text.encode("utf-8"))
    assert count_file_lines(target) == 2
    assert manager.read_file("sample.txt", offset=1).startswith("last")


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("prefix_size", [9000, 70000])
def test_read_rejects_late_binary_content(manager, workspace, batch, prefix_size):
    (workspace / "sample.txt").write_bytes(b"first\n" + b"x" * prefix_size + b"\x00")
    result = manager.read_files([{"path": "sample.txt", "limit": 1}]) if batch else manager.read_file("sample.txt", limit=1)
    assert result.startswith("ERROR[VALIDATION]")
    assert "NUL bytes" in result


@pytest.mark.parametrize("batch", [False, True])
def test_read_svg_as_text(manager, workspace, batch):
    text = '<svg xmlns="http://www.w3.org/2000/svg"><text>hello</text></svg>'
    (workspace / "sample.svg").write_bytes(text.encode())
    result = manager.read_files([{"path": "sample.svg"}]) if batch else manager.read_file("sample.svg")
    assert text in result
    assert not result.startswith("ERROR")


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("data", [b"\xff\ntext", b"text\n" + b"a" * 9000 + b"\xff", b"text\n" + b"a" * 70000 + b"\xff", "Привет".encode("cp1251")], ids=["invalid-first-line", "invalid-later-line", "invalid-later-chunk", "cp1251"])
def test_read_invalid_utf8_is_explicit_error(manager, workspace, batch, data):
    (workspace / "sample.txt").write_bytes(data)
    result = manager.read_files([{"path": "sample.txt", "limit": 1}]) if batch else manager.read_file("sample.txt", limit=1)
    assert result.startswith("ERROR[VALIDATION]")
    assert "not valid UTF-8" in result
    assert "\ufffd" not in result


def test_binary_detection_refreshes_after_rewrite(manager, workspace):
    target = workspace / "sample.unknown"
    target.write_bytes(b"\x00binary")
    assert is_binary_path(str(target))
    target.write_bytes(b"text")
    assert manager.read_file("sample.unknown") == "text"
    target.write_bytes(b"\x00binary")
    assert manager.read_file("sample.unknown").startswith("ERROR[VALIDATION]")


def test_binary_content_with_text_extension_is_rejected(manager, workspace):
    (workspace / "sample.txt").write_bytes(b"\x00binary")
    assert manager.read_file("sample.txt").startswith("ERROR[VALIDATION]")


@pytest.mark.parametrize("budget", [100, 500, 5000])
@pytest.mark.parametrize("numbered", [False, True])
def test_single_read_keeps_metadata_within_budget(manager, workspace, budget, numbered):
    manager.set_policy(SafetyPolicy(max_tool_output=budget))
    (workspace / "sample.txt").write_bytes(b"x" * 20000 + b"\nnext\n")
    result = manager.read_file("sample.txt", show_line_numbers=numbered)
    assert len(result) <= budget
    if budget == 100:
        assert result.startswith("ERROR[LIMIT_EXCEEDED]")
    else:
        assert "[TRUNCATED LINE]" in result
        assert "To continue:" in result and "offset=1" in result
        assert "[EOF]" not in result


def test_single_read_truncates_at_complete_line_boundary(manager, workspace):
    manager.set_policy(SafetyPolicy(max_tool_output=500))
    (workspace / "sample.txt").write_bytes(("a" * 100 + "\n").encode() * 10)
    result = manager.read_file("sample.txt")
    content = result.split("\n\n[TRUNCATED]", 1)[0]
    assert all(line == "a" * 100 for line in content.split("\n"))
    assert f"offset={len(content.splitlines())}" in result
    assert "[TRUNCATED LINE]" not in result
    assert len(result) <= 500


def test_last_truncated_line_does_not_report_eof(manager, workspace):
    manager.set_policy(SafetyPolicy(max_tool_output=500))
    (workspace / "sample.txt").write_bytes(b"x" * 20000)
    result = manager.read_file("sample.txt", show_line_numbers=True)
    assert "[TRUNCATED LINE]" in result
    assert "[EOF]" not in result
    assert len(result) <= 500


def test_small_complete_file_needs_no_metadata_reserve(manager, workspace):
    manager.set_policy(SafetyPolicy(max_tool_output=5))
    (workspace / "sample.txt").write_bytes(b"hello")
    assert manager.read_file("sample.txt") == "hello"


def test_single_read_errors_fit_budget(manager, workspace):
    manager.set_policy(SafetyPolicy(max_tool_output=100))
    result = manager.read_file("x" * 150 + ".txt")
    assert result.startswith("ERROR[NOT_FOUND]")
    assert len(result) <= 100


def test_line_count_io_error_is_not_an_empty_file(manager, workspace):
    (workspace / "sample.txt").write_bytes(b"text")
    with mock.patch.object(FilesystemManager, "_count_lines", side_effect=OSError("cannot read")):
        result = manager.read_file("sample.txt")
    assert result.startswith("ERROR[EXECUTION]")


@pytest.mark.parametrize("budget", [-1, 0, 5, 100])
def test_truncation_marker_is_included_in_budget(budget):
    result = truncate_output("x" * 200, budget, source="read_file")
    assert len(result) <= max(0, budget)
    if budget == 100:
        assert "[TRUNCATED from 200 chars | source=read_file]" in result


def test_tiny_truncation_preserves_error_prefix():
    text = "ERROR[LIMIT_EXCEEDED]: " + "x" * 200
    result = truncate_output(text, 32, source="read_file")
    assert result == text[:32]


@pytest.mark.parametrize("budget", [0, 5, 32, 100])
@pytest.mark.parametrize("requests,limit", [
    ([{"path": "sample.txt"}], 2000),
    ([], 2000),
    ([{"path": "sample.txt"}], 0),
    ([{}], 2000),
    ([{"path": "sample.txt", "unexpected": True}], 2000),
    ([None], 2000),
])
def test_batch_errors_fit_small_budget(manager, workspace, budget, requests, limit):
    manager.set_policy(SafetyPolicy(max_tool_output=budget))
    (workspace / "sample.txt").write_bytes(b"text")
    result = manager.read_files(requests, limit=limit)
    assert len(result) <= budget
    if budget >= 32:
        assert result.startswith("ERROR[")


def test_batch_top_level_limit_is_shared(tools, workspace):
    (workspace / "sample.txt").write_bytes(b"a\nb\nc\nd\ne\nf\n")
    result = tools.read_file_tool.invoke({"limit": 4, "requests": [{"path": "sample.txt"}] * 2})
    assert not result.startswith("ERROR")
    ranges = re.findall(r"lines=(\d+)-(\d+) of", result)
    assert len(ranges) == 2
    assert sum(int(end) - int(start) + 1 for start, end in ranges) == 4
    assert result.count("To continue:") == 2


@pytest.mark.parametrize("tool_name,payload", [
    ("read_file_tool", {"path": "sample.txt", "requests": [{"path": "sample.txt"}]}),
    ("read_file_tool", {"offset": 1, "requests": [{"path": "sample.txt"}]}),
    ("edit_file_tool", {"path": "sample.txt", "old_string": "old", "edits": [{"old_string": "old", "new_string": "new"}]}),
    ("edit_file_tool", {"path": "sample.txt", "match_mode": "indentation", "edits": [{"old_string": "old", "new_string": "new"}]}),
])
def test_conflicting_batch_arguments_are_rejected(tools, tool_name, payload):
    result = getattr(tools, tool_name).invoke(payload)
    assert result.startswith("ERROR[VALIDATION]")


@pytest.mark.parametrize("tool_name,payload", [
    ("read_file_tool", {"requests": []}),
    ("read_file_tool", {"requests": [{"path": "a.txt"}] * 9}),
    ("read_file_tool", {"requests": [{"offset": 0}]}),
    ("read_file_tool", {"requests": [{"path": "a.txt", "offset": -1}]}),
    ("edit_file_tool", {"path": "a.txt", "edits": []}),
    ("edit_file_tool", {"path": "a.txt", "edits": [{"old_string": "a", "new_string": "b"}] * 21}),
    ("edit_file_tool", {"path": "a.txt", "edits": [{"old_string": "a"}]}),
    ("edit_file_tool", {"path": "a.txt", "edits": [{"old_string": "a", "new_string": "b", "path": "other.txt"}]}),
])
def test_nested_schema_validation(tools, tool_name, payload):
    with pytest.raises(ValidationError):
        getattr(tools, tool_name).invoke(payload)


def test_eight_read_requests_have_independent_statuses(manager, workspace):
    (workspace / "sample.txt").write_bytes(b"one\ntwo\n")
    result = manager.read_files([{"path": "sample.txt", "limit": 1}] * 8)
    assert result.startswith("Batch read: TRUNCATED")
    assert result.count("status=TRUNCATED") == 8
    assert result.count("To continue:") == 8
    assert len(result) <= manager._tool_limit()


def test_read_batch_shares_line_budget(manager, workspace):
    manager.set_policy(SafetyPolicy(max_tool_output=15000, max_read_lines=6))
    (workspace / "sample.txt").write_bytes(b"a\nb\nc\nd\ne\nf\ng\n")
    result = manager.read_files([{"path": "sample.txt"}] * 3)
    ranges = re.findall(r"lines=(\d+)-(\d+) of", result)
    assert len(ranges) == 3
    assert sum(int(end) - int(start) + 1 for start, end in ranges) == 6
    assert result.count("To continue:") == 3


@pytest.mark.parametrize("budget", [5000, 15000])
def test_batch_output_including_metadata_fits_character_budget(manager, workspace, budget):
    manager.set_policy(SafetyPolicy(max_tool_output=budget))
    (workspace / "sample.txt").write_bytes(("x" * 20000 + "\nnext\n").encode())
    result = manager.read_files([{"path": "sample.txt", "show_line_numbers": True}] * 8)
    assert not result.startswith("ERROR")
    assert len(result) <= budget
    assert result.count("status=TRUNCATED") == 8
    assert result.count("[TRUNCATED LINE]") == 8
    assert result.count("--- End request") == 8


def test_batch_partial_failure_reaches_result_parser(manager, workspace):
    (workspace / "sample.txt").write_bytes(b"readable")
    result = manager.read_files([{"path": "missing.txt"}, {"path": "sample.txt"}])
    parsed = parse_tool_execution_result(result)
    assert not parsed.ok
    assert parsed.error_type == "NOT_FOUND"
    assert "status=ERROR" in result
    assert "status=OK" in result
    assert "readable" in result


@pytest.mark.parametrize("offset", [0, 7])
def test_batch_empty_file_and_exhausted_line_budget(manager, workspace, offset):
    manager.set_policy(SafetyPolicy(max_tool_output=15000, max_read_lines=1))
    (workspace / "sample.txt").write_bytes(b"text")
    result = manager.read_files([{"path": "sample.txt"}] * 2)
    assert "status=SKIPPED" in result
    assert "status=OK" in result
    (workspace / "empty.txt").write_bytes(b"")
    result = manager.read_files([{"path": "empty.txt", "offset": offset}])
    assert "status=OK; lines=0-0 of 0" in result
    assert "empty contents" in result


@pytest.mark.parametrize("range_args", [
    {"path": "sample.txt", "limit": 0},
    {"path": "sample.txt", "limit": "bad"},
    {"path": "sample.txt", "limit": True},
    {"path": "sample.txt", "offset": -1},
])
def test_backend_batch_invalid_ranges_return_errors(manager, workspace, range_args):
    (workspace / "sample.txt").write_bytes(b"text")
    result = manager.read_files([range_args, {"path": "sample.txt"}])
    assert result.startswith("ERROR[VALIDATION]")
    assert "status=OK" in result


def test_batch_paths_stay_within_allowed_roots(manager, workspace):
    result = manager.read_files([{"path": "../outside.txt"}])
    assert "ACCESS DENIED" in result
    assert not parse_tool_execution_result(result).ok


def test_preflight_enumerates_nested_paths_and_missing_fields():
    preflight = ToolPreflightMixin()
    preflight.tools_map = {"read_file": filesystem.read_file_tool, "edit_file": filesystem.edit_file_tool}
    preflight._required_fields_cache = {}
    assert preflight._iter_path_like_targets({"requests": [{"path": "one.txt"}, {"filepath": "../two.txt"}]}) == [
        ("requests[0].path", "one.txt"), ("requests[1].filepath", "../two.txt"),
    ]
    assert preflight._missing_required_tool_fields("read_file", {}) == ["path"]
    assert preflight._missing_required_tool_fields("read_file", {"requests": [{"path": "one.txt"}]}) == []
    assert preflight._missing_required_tool_fields("read_file", {"requests": [{}]}) == ["requests.0.path"]
    assert preflight._missing_required_tool_fields("edit_file", {"path": "one.txt", "edits": [{"old_string": "a"}]}) == ["edits.0.new_string"]
    assert preflight._missing_required_tool_fields("edit_file", {"path": "one.txt", "edits": [{"old_string": "a", "new_string": ""}]}) == []


@pytest.mark.parametrize("tool_name", ["read_file", "edit_file"])
def test_preflight_preserves_tools_without_batch_schemas(tool_name):
    from pydantic import BaseModel

    class SinglePathInput(BaseModel):
        path: str

    preflight = ToolPreflightMixin()
    preflight.tools_map = {
        tool_name: SimpleNamespace(get_input_schema=lambda: SinglePathInput),
    }
    preflight._required_fields_cache = {}
    assert preflight._missing_required_tool_fields(tool_name, {"path": "one.txt"}) == []
    assert preflight._missing_required_tool_fields(tool_name, {}) == ["path"]
    preflight.tools_map = {}
    preflight._required_fields_cache = {}
    assert preflight._missing_required_tool_fields(tool_name, {}) == []


def test_batch_edit_does_not_bypass_approval():
    preflight = ToolPreflightMixin()
    preflight.config = SimpleNamespace(enable_approvals=True)
    preflight.tool_metadata = {
        "edit_file": ToolMetadata(name="edit_file", mutating=True, requires_approval=True),
        "read_file": ToolMetadata(name="read_file", read_only=True),
    }
    assert preflight._tool_requires_approval("edit_file", {"path": "one.txt", "edits": [{"old_string": "a", "new_string": "b"}]})
    assert not preflight._tool_requires_approval("read_file", {"requests": [{"path": "one.txt"}]})


@pytest.mark.parametrize("budget,count", [(800, 1), (1500, 2), (5000, 8)])
def test_long_errors_keep_every_status_within_budget(manager, budget, count):
    manager.set_policy(SafetyPolicy(max_tool_output=budget))
    with mock.patch.object(FilesystemManager, "_read_file_page", return_value={"error": "ERROR[EXECUTION]: " + "e" * 10000}):
        result = manager.read_files([{"path": "a.txt"}] * count)
    assert len(result) <= budget
    assert result.startswith("ERROR[EXECUTION]")
    assert result.count("status=ERROR") == count
    assert result.count("--- End request") == count


def test_metadata_budget_rejects_before_reading(manager):
    manager.set_policy(SafetyPolicy(max_tool_output=500))
    with mock.patch.object(FilesystemManager, "_read_file_page") as read:
        result = manager.read_files([{"path": "a.txt"}] * 8)
    assert result.startswith("ERROR[LIMIT_EXCEEDED]")
    assert len(result) <= 500
    read.assert_not_called()


def test_invalid_late_range_is_not_hidden_by_exhausted_budget(manager, workspace):
    manager.set_policy(SafetyPolicy(max_tool_output=15000, max_read_lines=1))
    (workspace / "sample.txt").write_bytes(b"text")
    result = manager.read_files([{"path": "sample.txt"}, {"path": "sample.txt", "limit": "bad"}])
    assert result.startswith("ERROR[VALIDATION]")
    assert "status=ERROR" in result


def test_single_edit_aliases_allow_deletion(tools, workspace):
    target = workspace / "sample.txt"
    target.write_bytes(b"old remaining")
    result = tools.edit_file_tool.invoke({"filepath": "sample.txt", "search_text": "old ", "new_text": ""})
    assert result.startswith("Success:")
    assert target.read_bytes() == b"remaining"
