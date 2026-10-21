import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.text_utils import build_tool_ui_labels, format_tool_display
from ui.widgets.tool_group import ToolGroupWidget
from ui.widgets.tools import FILE_ACCENT_BLUE, ToolCardWidget


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("name", ["read_file", "Read"])
@pytest.mark.parametrize(
    "requests, expected",
    [
        ([{"path": "src/one.py"}, {"path": "src/two.py"}], "src/one.py, src/two.py"),
        ([{"path": "one.py", "offset": 0}, {"path": "one.py", "offset": 80}], "one.py"),
        ([{"file_path": "one.py"}, {"filepath": "two.py"}], "one.py, two.py"),
        ([None, {}, {"path": None}, {"path": " "}, {"path": "one.py"}], "one.py"),
    ],
)
def test_batch_read_labels_show_unique_paths(name, requests, expected):
    labels = build_tool_ui_labels(name, {"requests": requests})
    assert labels["subtitle"] == expected
    assert labels["raw_display"] == f"{name}({expected})"
    assert labels["args_state"] == "complete"
    assert format_tool_display(name, {"requests": requests}) == f"{name}({expected})"


@pytest.mark.parametrize("requests", [None, [], [{}], [None], [{"path": None}], {"path": "one.py"}])
def test_batch_read_without_paths_stays_partial(requests):
    labels = build_tool_ui_labels("read_file", {"requests": requests})
    assert labels["args_state"] == "partial"
    assert labels["subtitle"] == "Resolving arguments…"
    assert labels["raw_display"] == "read_file"


def test_batch_read_card_keeps_paths_when_finished(app):
    card = ToolCardWidget({
        "name": "read_file",
        "args": {"requests": [{"path": "src/one.py"}, {"path": "src/two.py"}]},
        "phase": "running",
    })
    try:
        assert card.action_label.full_text() == "Reading src/one.py, src/two.py"
        card.finish({"content": "Batch read: OK; 2 request(s)."})
        assert card.action_label.full_text() == "Read src/one.py, src/two.py"
        assert card.action_label.toolTip() == "read_file(src/one.py, src/two.py)"
        assert card._action_line_segments("Read", card.payload, card.payload["args"]) == [
            ("Read ", ""), ("src/one.py, src/two.py", FILE_ACCENT_BLUE),
        ]
    finally:
        card.deleteLater()


@pytest.mark.parametrize("subtitle", ["", "Resolving arguments…"])
def test_batch_read_card_refreshes_legacy_labels(app, subtitle):
    card = ToolCardWidget({
        "name": "read_file",
        "args": {"requests": [{"path": "one.py"}, {"path": "two.py"}]},
        "phase": "finished",
        "subtitle": subtitle,
        "raw_display": "read_file",
        "args_state": "partial",
        "display_state": "finished",
    })
    try:
        assert card.action_label.full_text() == "Read one.py, two.py"
        assert card.action_label.toolTip() == "read_file(one.py, two.py)"
        assert card.payload["args_state"] == "complete"
    finally:
        card.deleteLater()


def test_batch_read_card_refreshes_streamed_arguments(app):
    card = ToolCardWidget({"name": "read_file", "args": {}, "phase": "preparing"})
    try:
        card.update_started_payload({"args": {"requests": [{"path": "one.py"}]}, "phase": "running"})
        assert card.action_label.full_text() == "Reading one.py"
        card.update_started_payload({"args": {"requests": [{"path": "one.py"}, {"path": "two.py"}]}})
        assert card.action_label.full_text() == "Reading one.py, two.py"
        card.finish({})
        assert card.action_label.full_text() == "Read one.py, two.py"
    finally:
        card.deleteLater()


def test_batch_read_tooltip_keeps_all_long_paths(app):
    paths = ["long-directory/" * 8 + f"file-{index}.py" for index in range(8)]
    card = ToolCardWidget({
        "name": "read_file", "phase": "finished",
        "args": {"requests": [{"path": path} for path in paths]},
    })
    try:
        for width in (120, 800, 4000):
            card.action_label.setFixedWidth(width)
            for path in paths:
                assert path in card.action_label.toolTip()
        assert "requests=" not in card.action_label.full_text()
    finally:
        card.deleteLater()


def test_batch_read_group_counts_files_not_ranges(app):
    group = ToolGroupWidget()
    try:
        card = ToolCardWidget({
            "name": "read_file", "phase": "running",
            "args": {"requests": [{"path": "one.py"}, {"path": "two.py"}, {"path": "one.py", "offset": 80}]},
        }, parent=group.container)
        group.add_tool(card)
        assert group.header_btn.text() == "Reading 2 files"
        card.finish({})
        group.refresh_completion()
        assert group.header_btn.text() == "Read 2 files"
    finally:
        group.deleteLater()


def test_single_file_caption_is_unchanged(app):
    card = ToolCardWidget({"name": "read_file", "args": {"path": "one.py"}, "phase": "running"})
    try:
        assert card.action_label.full_text() == "Reading one.py"
        card.finish({})
        assert card.action_label.full_text() == "Read one.py"
    finally:
        card.deleteLater()
