"""
Unit tests for annotate_video's report loading, which has to accept two
different report shapes this project writes -- detect_all_actions.py's
*.all_actions.json ("actions") and main.py's *.report.json ("events") --
since both are plausible files to pass to `annotate_video.py --report`.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotate_video import _filter_only_composite, _load_actions_from_report
from src.action_analyzer import DetectedAction
from src.video_annotator import VideoAnnotationError


def test_loads_all_actions_report(tmp_path: Path):
    # detect_all_actions.py's format: top-level "actions", each keyed by "name".
    report_path = tmp_path / "video.all_actions.json"
    report_path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "name": "person using phone",
                        "source": "derived",
                        "start_seconds": 3.0,
                        "end_seconds": 6.0,
                        "duration_seconds": 3.0,
                        "confidence": 0.8,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    actions = _load_actions_from_report(report_path)

    assert len(actions) == 1
    assert actions[0].name == "person using phone"
    assert actions[0].source == "derived"
    assert actions[0].start_seconds == pytest.approx(3.0)
    assert actions[0].end_seconds == pytest.approx(6.0)
    assert actions[0].confidence == pytest.approx(0.8)


def test_loads_single_action_report(tmp_path: Path):
    # main.py's format: top-level "events", each keyed by "matched_term"
    # rather than "name" -- this is what `main.py --action "using phone"`
    # writes, and a natural (if differently-shaped) file to pass to
    # `annotate_video.py --report`.
    report_path = tmp_path / "video.report.json"
    report_path.write_text(
        json.dumps(
            {
                "action": "using phone",
                "events": [
                    {
                        "source": "derived",
                        "matched_term": "person using phone",
                        "start_seconds": 0.08,
                        "end_seconds": 18.84,
                        "duration_seconds": 18.76,
                        "confidence": 0.757,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    actions = _load_actions_from_report(report_path)

    assert len(actions) == 1
    assert actions[0].name == "person using phone"
    assert actions[0].source == "derived"
    assert actions[0].start_seconds == pytest.approx(0.08)
    assert actions[0].end_seconds == pytest.approx(18.84)
    assert actions[0].confidence == pytest.approx(0.757)


def test_rejects_unrecognized_report_shape(tmp_path: Path):
    report_path = tmp_path / "not_a_report.json"
    report_path.write_text(json.dumps({"something_else": []}), encoding="utf-8")

    with pytest.raises(VideoAnnotationError):
        _load_actions_from_report(report_path)


def _sample_mixed_actions() -> list[DetectedAction]:
    return [
        DetectedAction(
            name="person",
            source="labels",
            start_seconds=0.0,
            end_seconds=10.0,
            confidence=1.0,
        ),
        DetectedAction(
            name="cell phone",
            source="objects",
            start_seconds=3.0,
            end_seconds=6.0,
            confidence=0.8,
        ),
        DetectedAction(
            name="person using phone",
            source="derived",
            start_seconds=3.0,
            end_seconds=6.0,
            confidence=0.8,
            evidence="person (labels) + cell phone (objects)",
        ),
        DetectedAction(
            name="person handling cash",
            source="derived",
            start_seconds=8.0,
            end_seconds=9.0,
            confidence=0.9,
            evidence="person (labels) + WE TRUST (ocr)",
        ),
    ]


def test_filter_only_composite_keeps_only_derived_actions():
    actions = _sample_mixed_actions()

    filtered = _filter_only_composite(actions, only_composite=True)

    assert {a.source for a in filtered} == {"derived"}
    assert {a.name for a in filtered} == {
        "person using phone",
        "person handling cash",
    }


def test_filter_only_composite_disabled_returns_everything_unchanged():
    actions = _sample_mixed_actions()

    filtered = _filter_only_composite(actions, only_composite=False)

    assert filtered == actions


def test_filter_only_composite_empty_when_no_derived_actions():
    actions = [a for a in _sample_mixed_actions() if a.source != "derived"]

    filtered = _filter_only_composite(actions, only_composite=True)

    assert filtered == []
