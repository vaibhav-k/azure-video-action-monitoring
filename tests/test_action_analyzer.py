"""
Unit tests for action_analyzer / report using a realistic sample Video
Indexer insights payload, so the extraction and reporting logic can be
verified end-to-end without needing a live Azure Video Indexer account.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.action_analyzer import analyze
from src.report import to_console_text, to_dict, write_html, write_json

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_insights_jumping.json"


@pytest.fixture()
def sample_index() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_analyze_finds_all_jumping_label_instances(sample_index: dict[str, Any]):
    report = analyze(sample_index, action="jumping")

    assert report.occurrence_count == 3
    assert report.video_duration_seconds == pytest.approx(42.5)
    assert report.distinct_people_tracked == 2
    assert report.first_occurrence_seconds == pytest.approx(3.2)
    assert report.last_occurrence_seconds == pytest.approx(28.4)

    # Events must be sorted chronologically.
    starts = [e.start_seconds for e in report.events]
    assert starts == sorted(starts)

    # Confidence values should be carried through from the fixture.
    assert all(e.confidence is not None for e in report.events)


def test_analyze_ignores_unrelated_labels(sample_index: dict[str, Any]):
    report = analyze(sample_index, action="jumping")
    matched_names = {e.matched_term for e in report.events}
    assert "person" not in matched_names
    assert "outdoor" not in matched_names


def test_analyze_supports_extra_synonyms(sample_index: dict[str, Any]):
    # "trampoline" only appears in keywords and wouldn't match "jump" by
    # default; verify a custom synonym picks it up.
    report = analyze(sample_index, action="jumping", extra_synonyms=["trampoline"])
    sources = {e.source for e in report.events}
    assert "keywords" in sources


def test_analyze_missing_action_returns_empty_report():
    empty_index: dict[str, Any] = {
        "videos": [{"insights": {"labels": [], "keywords": []}}]
    }
    report = analyze(empty_index, action="jumping")
    assert report.occurrence_count == 0
    assert report.total_action_seconds == 0.0
    assert report.first_occurrence_seconds is None


def test_analyze_cash_matches_currency_ocr_text():
    # "cash" has no visual-concept vocabulary to match (no label/keyword/
    # object class for money) -- it only matches via `ocr`, against phrases
    # that are literally printed on US currency. A bare denomination number
    # ("100") deliberately isn't a match target (too ambiguous alone); a
    # motto fragment like "WE TRUST" is.
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
                        },
                        {
                            "text": "100",
                            "confidence": 0.996,
                            "instances": [{"start": "0:00:00.08", "end": "0:00:03"}],
                        },
                    ],
                }
            }
        ]
    }
    report = analyze(index, action="cash")

    matched_names = {e.matched_term for e in report.events}
    assert matched_names == {"WE TRUST"}
    assert report.events[0].source == "ocr"


def test_analyze_phone_matches_derived_person_using_phone_overlap():
    # "phone" matches the "cell phone" object directly (substring match,
    # same as any other action search) *and* the composite "person using
    # phone" derived from that same object overlapping a person label --
    # both are legitimate, independent hits for "--action phone".
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
    report = analyze(index, action="phone")

    assert report.occurrence_count == 2
    matched = {(e.matched_term, e.source) for e in report.events}
    assert matched == {
        ("cell phone", "objects"),
        ("person using phone", "derived"),
    }


def test_analyze_min_overlap_seconds_filters_short_composite_overlap():
    # Same shape as test_analyze_phone_matches_derived_person_using_phone_overlap,
    # but the overlap is a 0.02s sliver -- min_overlap_seconds should drop
    # the derived match while leaving the direct "cell phone" object match
    # untouched (that one isn't a composite, so the threshold doesn't apply
    # to it).
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
                                    "end": "0:00:03.02",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }
    report = analyze(index, action="phone", min_overlap_seconds=0.15)

    matched = {(e.matched_term, e.source) for e in report.events}
    assert matched == {("cell phone", "objects")}


def test_analyze_register_matches_derived_person_at_register_overlap():
    # "person at register" is the weakest-evidence composite (any
    # keyboard/monitor-like object near a person), added for a checkout/
    # cashier monitoring use case -- verify it derives the same way as the
    # phone composite.
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 0.9,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        }
                    ],
                    "keywords": [],
                    "detectedObjects": [
                        {
                            "displayName": "keyboard",
                            "instances": [
                                {
                                    "confidence": 0.6,
                                    "start": "0:00:02",
                                    "end": "0:00:05",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }
    report = analyze(index, action="register")

    matched_names = {e.matched_term for e in report.events}
    assert matched_names == {"person at register"}
    assert report.events[0].confidence == pytest.approx(0.6)  # min(0.9, 0.6)


def test_report_rendering_round_trip(sample_index: dict[str, Any], tmp_path: Path):
    report = analyze(sample_index, action="jumping")

    text = to_console_text(report, "people_jumping.mp4")
    assert "jumping" in text
    assert "3" in text  # occurrence count appears somewhere in the summary

    as_dict = to_dict(report, "people_jumping.mp4")
    assert as_dict["occurrence_count"] == 3
    assert len(as_dict["events"]) == 3

    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"
    write_json(report, "people_jumping.mp4", json_path)
    write_html(report, "people_jumping.mp4", html_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["occurrence_count"] == 3

    html = html_path.read_text(encoding="utf-8")
    assert "jumping" in html
    assert "bar-fill" in html  # timeline rendered since duration is known
