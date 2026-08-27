"""
Unit tests for action_analyzer / report using a realistic sample Video
Indexer insights payload, so the extraction and reporting logic can be
verified end-to-end without needing a live Azure Video Indexer account.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.action_analyzer import analyze
from src.report import to_console_text, to_dict, write_html, write_json

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_insights_jumping.json"


@pytest.fixture()
def sample_index():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_analyze_finds_all_jumping_label_instances(sample_index):
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


def test_analyze_ignores_unrelated_labels(sample_index):
    report = analyze(sample_index, action="jumping")
    matched_names = {e.matched_term for e in report.events}
    assert "person" not in matched_names
    assert "outdoor" not in matched_names


def test_analyze_supports_extra_synonyms(sample_index):
    # "trampoline" only appears in keywords and wouldn't match "jump" by
    # default; verify a custom synonym picks it up.
    report = analyze(sample_index, action="jumping", extra_synonyms=["trampoline"])
    sources = {e.source for e in report.events}
    assert "keywords" in sources


def test_analyze_missing_action_returns_empty_report():
    empty_index = {"videos": [{"insights": {"labels": [], "keywords": []}}]}
    report = analyze(empty_index, action="jumping")
    assert report.occurrence_count == 0
    assert report.total_action_seconds == 0.0
    assert report.first_occurrence_seconds is None


def test_report_rendering_round_trip(sample_index, tmp_path):
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
