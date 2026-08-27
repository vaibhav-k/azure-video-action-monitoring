"""
Unit tests for action_analyzer.analyze_all / report's all-actions renderers,
using the same realistic sample Video Indexer insights payload as
test_action_analyzer.py, so this can be verified without a live Azure
account.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.action_analyzer import analyze_all
from src.report import to_console_text_all, to_dict_all, write_html_all, write_json_all

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_insights_jumping.json"


@pytest.fixture()
def sample_index():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_analyze_all_finds_every_labeled_and_keyword_occurrence(sample_index):
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


def test_analyze_all_includes_both_label_and_keyword_sources(sample_index):
    report = analyze_all(sample_index)
    sources = {a.source for a in report.actions}
    assert sources == {"labels", "keywords"}


def test_analyze_all_summaries_aggregate_per_action(sample_index):
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


def test_analyze_all_min_confidence_filters_low_confidence_occurrences(sample_index):
    # "trampoline" (keyword) has no confidence and must always be kept;
    # "jumping" instances have confidences 0.87 / 0.81 / 0.76.
    report = analyze_all(sample_index, min_confidence=0.8)
    names = [a.name for a in report.actions]

    assert "trampoline" in names  # no confidence -> always kept
    jumping_count = names.count("jumping")
    assert jumping_count == 2  # only the 0.87 and 0.81 instances survive


def test_analyze_all_handles_empty_insights():
    empty_index = {"videos": [{"insights": {"labels": [], "keywords": []}}]}
    report = analyze_all(empty_index)
    assert report.occurrence_count == 0
    assert report.distinct_action_count == 0
    assert report.summaries == []


def test_all_actions_report_rendering_round_trip(sample_index, tmp_path):
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
