"""
Unit tests for action_analyzer.analyze_all / report's all-actions renderers,
using the same realistic sample Video Indexer insights payload as
test_action_analyzer.py, so this can be verified without a live Azure
account.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.action_analyzer import analyze_all
from src.report import to_console_text_all, to_dict_all, write_html_all, write_json_all

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_insights_jumping.json"


@pytest.fixture()
def sample_index() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_analyze_all_finds_every_labeled_and_keyword_occurrence(
    sample_index: dict[str, Any],
):
    report = analyze_all(sample_index)

    # Fixture has: labels "person" (1), "jumping" (3), "outdoor" (1);
    # keywords "trampoline" (1) -> 6 total timestamped occurrences.
    assert report.occurrence_count == 6
    assert report.distinct_action_count == 4
    assert report.video_duration_seconds == pytest.approx(42.5)
    assert report.distinct_people_tracked == 2

    names = {a.name for a in report.actions}
    assert names == {"person", "jumping", "outdoor", "trampoline"}

    # Actions must be sorted chronologically.
    starts = [a.start_seconds for a in report.actions]
    assert starts == sorted(starts)


def test_analyze_all_includes_both_label_and_keyword_sources(
    sample_index: dict[str, Any],
):
    report = analyze_all(sample_index)
    sources = {a.source for a in report.actions}
    assert sources == {"labels", "keywords"}


def test_analyze_all_summaries_aggregate_per_action(sample_index: dict[str, Any]):
    report = analyze_all(sample_index)
    summaries = {s.name: s for s in report.summaries}

    jumping = summaries["jumping"]
    assert jumping.occurrence_count == 3
    assert jumping.first_occurrence_seconds == pytest.approx(3.2)
    assert jumping.last_occurrence_seconds == pytest.approx(28.4)
    assert jumping.max_confidence == pytest.approx(0.87)

    person = summaries["person"]
    assert person.occurrence_count == 1
    assert person.total_duration_seconds == pytest.approx(42.5)


def test_analyze_all_min_confidence_filters_low_confidence_occurrences(
    sample_index: dict[str, Any],
):
    # "trampoline" (keyword) has no confidence and must always be kept;
    # "jumping" instances have confidences 0.87 / 0.81 / 0.76.
    report = analyze_all(sample_index, min_confidence=0.8)
    names = [a.name for a in report.actions]

    assert "trampoline" in names  # no confidence -> always kept
    jumping_count = names.count("jumping")
    assert jumping_count == 2  # only the 0.87 and 0.81 instances survive


def test_analyze_all_handles_empty_insights(sample_index: dict[str, Any] | None):
    empty_index: dict[str, Any] = {
        "videos": [{"insights": {"labels": [], "keywords": []}}]
    }
    report = analyze_all(empty_index)
    assert report.occurrence_count == 0
    assert report.distinct_action_count == 0
    assert report.summaries == []


def test_analyze_all_includes_detected_objects(sample_index: dict[str, Any] | None):
    # detectedObjects items use "displayName"/"type" instead of labels' and
    # keywords' "name", and are a distinct insight bucket Video Indexer
    # returns when object detection is enabled -- verify they're picked up
    # under source "objects" rather than silently dropped.
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [],
                    "keywords": [],
                    "detectedObjects": [
                        {
                            "type": "Cup",
                            "displayName": "cup",
                            "instances": [
                                {
                                    "confidence": 0.82,
                                    "start": "0:00:01.0",
                                    "end": "0:00:02.0",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }
    report = analyze_all(index)

    assert report.occurrence_count == 1
    assert report.actions[0].name == "cup"
    assert report.actions[0].source == "objects"
    assert report.actions[0].confidence == pytest.approx(0.82)


def test_analyze_all_includes_ocr_text(sample_index: dict[str, Any] | None):
    # ocr items are keyed by "text" rather than "name"/"displayName" -- this
    # is the bucket that lets something like currency show up at all, since
    # its printed text ("100", "WE TRUST") gets read regardless of whether
    # "money" exists as an object/label class (it doesn't).
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [],
                    "keywords": [],
                    "ocr": [
                        {
                            "text": "WE TRUST",
                            "confidence": 0.998,
                            "instances": [{"start": "0:00:00.08", "end": "0:00:00.6"}],
                        }
                    ],
                }
            }
        ]
    }
    report = analyze_all(index)

    assert report.occurrence_count == 1
    assert report.actions[0].name == "WE TRUST"
    assert report.actions[0].source == "ocr"
    assert report.actions[0].confidence == pytest.approx(0.998)


def test_analyze_all_derives_person_using_phone_from_overlap(
    sample_index: dict[str, Any] | None,
):
    # No single label/object means "using phone" -- it's derived from a
    # "person" label and a "cell phone" object overlapping in time.
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        }
                    ],
                    "keywords": [],
                    "detectedObjects": [
                        {
                            "displayName": "cell phone",
                            "instances": [
                                {
                                    "confidence": 0.8,
                                    "start": "0:00:03",
                                    "end": "0:00:06",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }
    report = analyze_all(index)
    derived = [a for a in report.actions if a.source == "derived"]

    assert len(derived) == 1
    assert derived[0].name == "person using phone"
    assert derived[0].start_seconds == pytest.approx(3.0)
    assert derived[0].end_seconds == pytest.approx(6.0)
    assert derived[0].confidence == pytest.approx(0.8)  # min(1.0, 0.8)
    # Evidence names the two contributing items, so a reviewer can see why
    # this fired without re-running raw insights.
    assert derived[0].evidence == "person (labels) + cell phone (objects)"


def test_analyze_all_does_not_derive_when_no_overlap(
    sample_index: dict[str, Any] | None,
):
    # "person" and "cell phone" both present, but at non-overlapping times
    # -- there's no basis to claim the person was using the phone.
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:02",
                                }
                            ],
                        }
                    ],
                    "keywords": [],
                    "detectedObjects": [
                        {
                            "displayName": "cell phone",
                            "instances": [
                                {
                                    "confidence": 0.8,
                                    "start": "0:00:05",
                                    "end": "0:00:06",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }
    report = analyze_all(index)
    assert not [a for a in report.actions if a.source == "derived"]


def _person_and_cell_phone_index(phone_start: str, phone_end: str) -> dict[str, Any]:
    """Shared fixture builder for the min_overlap_seconds tests below: a
    person label spanning 0-10s, and a cell phone object at the given
    (start, end) timestamps."""
    return {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        }
                    ],
                    "keywords": [],
                    "detectedObjects": [
                        {
                            "displayName": "cell phone",
                            "instances": [
                                {
                                    "confidence": 0.8,
                                    "start": phone_start,
                                    "end": phone_end,
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }


def test_analyze_all_min_overlap_seconds_drops_short_overlaps() -> None:
    # A 0.05s overlap is the kind of single-frame detector blip
    # min_overlap_seconds exists to filter -- e.g. the real 0.04s
    # "person using phone" occurrence seen in this project's own captured
    # data (see README), not a meaningful co-occurrence.
    index = _person_and_cell_phone_index("0:00:03", "0:00:03.05")

    unfiltered = analyze_all(index)
    assert len([a for a in unfiltered.actions if a.source == "derived"]) == 1

    filtered = analyze_all(index, min_overlap_seconds=0.15)
    assert not [a for a in filtered.actions if a.source == "derived"]


def test_analyze_all_min_overlap_seconds_keeps_long_enough_overlaps() -> None:
    index = _person_and_cell_phone_index("0:00:03", "0:00:06")

    report = analyze_all(index, min_overlap_seconds=0.15)
    derived = [a for a in report.actions if a.source == "derived"]

    assert len(derived) == 1
    assert derived[0].name == "person using phone"


def test_analyze_all_derives_person_handling_cash_from_ocr_overlap() -> None:
    # "person handling cash" has no exact-name match target on its right
    # side -- it's derived via CompositeSide.synonyms, matching the same
    # cash-related phrases as DEFAULT_SYNONYMS["cash"] against OCR text,
    # since printed currency text varies per banknote.
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 0.95,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        }
                    ],
                    "keywords": [],
                    "ocr": [
                        {
                            "text": "WE TRUST",
                            "confidence": 0.99,
                            "instances": [{"start": "0:00:02", "end": "0:00:04"}],
                        }
                    ],
                }
            }
        ]
    }
    report = analyze_all(index)
    derived = [a for a in report.actions if a.source == "derived"]

    assert len(derived) == 1
    assert derived[0].name == "person handling cash"
    assert derived[0].start_seconds == pytest.approx(2.0)
    assert derived[0].end_seconds == pytest.approx(4.0)
    assert derived[0].confidence == pytest.approx(0.95)  # min(0.95, 0.99)
    assert derived[0].evidence == "person (labels) + WE TRUST (ocr)"


def test_analyze_all_merge_gap_seconds_merges_nearby_occurrences(
    sample_index: dict[str, Any],
):
    # Fixture's "jumping" instances are at 3.2-4.0, 15.6-16.4, 28.0-28.4
    # (approx, per test_analyze_all_summaries_aggregate_per_action's
    # first/last timestamps) -- far enough apart that no merge_gap_seconds
    # used here should combine them; this instead uses a small synthetic
    # index with two occurrences close enough to merge.
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "jumping",
                            "instances": [
                                {
                                    "confidence": 0.7,
                                    "start": "0:00:01",
                                    "end": "0:00:02",
                                },
                                {
                                    "confidence": 0.9,
                                    "start": "0:00:02.3",
                                    "end": "0:00:03",
                                },
                            ],
                        }
                    ],
                    "keywords": [],
                }
            }
        ]
    }

    unmerged = analyze_all(index)
    assert len(unmerged.actions) == 2

    merged = analyze_all(index, merge_gap_seconds=0.5)
    assert len(merged.actions) == 1
    assert merged.actions[0].start_seconds == pytest.approx(1.0)
    assert merged.actions[0].end_seconds == pytest.approx(3.0)
    assert merged.actions[0].confidence == pytest.approx(0.9)  # max of the two

    # A gap tighter than the actual 0.3s gap between them must not merge.
    not_merged = analyze_all(index, merge_gap_seconds=0.1)
    assert len(not_merged.actions) == 2


def test_analyze_all_min_temporal_iou_and_require_single_person() -> None:
    # Same person+phone shape used in test_action_analyzer.py's analyze()
    # tests, exercised here through analyze_all() for parity: both new
    # opt-in filters must reach the "all actions" path too, not just the
    # single-action one.
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        }
                    ],
                    "keywords": [],
                    "detectedObjects": [
                        {
                            "displayName": "cell phone",
                            "instances": [
                                {
                                    "confidence": 0.8,
                                    "start": "0:00:03",
                                    "end": "0:00:06",
                                }
                            ],
                        }
                    ],
                    "observedPeople": [
                        {"instances": [{"start": "0:00:00", "end": "0:00:10"}]},
                        {"instances": [{"start": "0:00:02", "end": "0:00:08"}]},
                    ],
                }
            }
        ]
    }

    baseline = analyze_all(index)
    derived_names = {a.name for a in baseline.actions if a.source == "derived"}
    assert "person using phone" in derived_names
    derived = next(a for a in baseline.actions if a.source == "derived")
    assert "2 person(s) in frame" in (derived.evidence or "")

    low_iou = analyze_all(index, min_temporal_iou=0.5)
    assert not any(a.source == "derived" for a in low_iou.actions)

    single_person = analyze_all(index, require_single_person=True)
    assert not any(a.source == "derived" for a in single_person.actions)


def test_all_actions_report_rendering_round_trip(
    sample_index: dict[str, Any], tmp_path: Path
):
    report = analyze_all(sample_index)

    text = to_console_text_all(report, "people_jumping.mp4")
    assert "jumping" in text
    assert "trampoline" in text

    as_dict = to_dict_all(report, "people_jumping.mp4")
    assert as_dict["occurrence_count"] == 6
    assert as_dict["distinct_action_count"] == 4
    assert len(as_dict["actions"]) == 6
    assert len(as_dict["action_summaries"]) == 4

    json_path = tmp_path / "all_actions.json"
    html_path = tmp_path / "all_actions.html"
    write_json_all(report, "people_jumping.mp4", json_path)
    write_html_all(report, "people_jumping.mp4", html_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["occurrence_count"] == 6

    html = html_path.read_text(encoding="utf-8")
    assert "trampoline" in html
    assert "bar-fill" in html
