"""
Unit tests for analyze_video.py's parsing/normalization/association logic --
built entirely on a synthetic raw Get-Video-Index payload shaped like Video
Indexer's real response (videos[0].insights.{labels,observedPeople,faces,
detectedObjects}), so none of this needs a live Azure account or network
access. A hand-rolled FakeVideoIndexerClient (this project's convention;
no unittest.mock) stands in for the real client to test the batch/skip/
--force orchestration in process_video() the same way.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_video as av


def _label(name: str, *instances: tuple[str, str, float | None]) -> dict[str, Any]:
    return {
        "name": name,
        "instances": [
            {"start": start, "end": end, "confidence": confidence}
            for start, end, confidence in instances
        ],
    }


def _observed_person(
    person_id: int,
    *instances: tuple[str, str],
    matching_face_id: int | None = None,
    matching_face_confidence: float | None = None,
) -> dict[str, Any]:
    person: dict[str, Any] = {
        "id": person_id,
        "instances": [{"start": s, "end": e} for s, e in instances],
    }
    if matching_face_id is not None:
        person["matchingFace"] = {
            "id": matching_face_id,
            "confidence": matching_face_confidence,
        }
    return person


def _raw_payload(
    *,
    labels: list[dict[str, Any]] | None = None,
    observed_people: list[dict[str, Any]] | None = None,
    faces: list[dict[str, Any]] | None = None,
    detected_objects: list[dict[str, Any]] | None = None,
    duration_seconds: float = 60.0,
) -> dict[str, Any]:
    return {
        "state": "Processed",
        "durationInSeconds": duration_seconds,
        "videos": [
            {
                "insights": {
                    "duration": "0:01:00",
                    "labels": labels or [],
                    "observedPeople": observed_people or [],
                    "faces": faces or [],
                    "detectedObjects": detected_objects or [],
                }
            }
        ],
    }


# ----------------------------------------------------------------------
# Timestamp helpers
# ----------------------------------------------------------------------


def test_parse_vi_timestamp_hours_minutes_seconds():
    assert av._parse_vi_timestamp("0:00:09.84") == 9.84
    assert av._parse_vi_timestamp("1:02:03.5") == 3723.5


def test_fmt_seconds_minutes_and_hours():
    assert av._fmt_seconds(18.0) == "00:18"
    assert av._fmt_seconds(65.0) == "01:05"
    assert av._fmt_seconds(3661.0) == "1:01:01"
    assert av._fmt_seconds(None) == "?"


# ----------------------------------------------------------------------
# _looks_like_activity: the activity/label display heuristic
# ----------------------------------------------------------------------


def test_looks_like_activity_accepts_ing_words_and_curated_hints():
    assert av._looks_like_activity("Running")
    assert av._looks_like_activity("skateboarding")  # -ing catch-all
    assert av._looks_like_activity("swim")  # curated hint word


def test_looks_like_activity_rejects_boilerplate_and_object_labels():
    assert not av._looks_like_activity("person")
    assert not av._looks_like_activity("necktie")
    assert not av._looks_like_activity("outdoor")
    assert not av._looks_like_activity("building")


def test_looks_like_activity_rejects_compound_labels_matching_by_substring_only():
    # Real-world regression (seen live: a "swimming" video's labels also
    # included "swimming pool", which the old check promoted to an
    # "activity" too, via `"swim" in "swimming pool"` as a raw substring
    # test -- even though it's a place, not an action). Curated out
    # explicitly since "swimming" is a genuine whole word there, not a
    # substring artifact.
    assert not av._looks_like_activity("swimming pool")
    # "sandbox" is the substring-artifact case instead: "box" only
    # appears as part of a different word, never as its own word.
    assert not av._looks_like_activity("sandbox")


def test_looks_like_activity_accepts_genuine_multi_word_activity_labels():
    assert av._looks_like_activity("swimming")  # still a real activity word
    assert av._looks_like_activity("long jumping")  # "-ing" word not last


# ----------------------------------------------------------------------
# analyze_insights: end-to-end parsing + activity/person association
# ----------------------------------------------------------------------


def test_analyze_insights_extracts_duration_people_labels_objects() -> None:
    raw = _raw_payload(
        labels=[_label("outdoor", ("0:00:00", "0:01:00", 0.99))],
        observed_people=[_observed_person(1, ("0:00:00", "0:00:30"))],
        detected_objects=[
            {
                "displayName": "backpack",
                "instances": [
                    {"start": "0:00:05", "end": "0:00:10", "confidence": 0.7}
                ],
            }
        ],
        duration_seconds=60.0,
    )

    analysis = av.analyze_insights(raw, video_filename="demo.mp4")

    assert analysis.video_filename == "demo.mp4"
    assert analysis.duration_seconds == 60.0
    assert len(analysis.people) == 1
    assert len(analysis.labels) == 1
    assert len(analysis.objects) == 1
    assert analysis.objects[0].name == "backpack"


def test_activity_associated_with_single_overlapping_person() -> None:
    raw = _raw_payload(
        labels=[_label("running", ("0:00:18", "0:00:27", 0.87))],
        observed_people=[_observed_person(2, ("0:00:15", "0:00:30"))],
    )

    analysis = av.analyze_insights(raw, video_filename="demo.mp4")

    assert len(analysis.activities) == 1
    activity = analysis.activities[0]
    assert activity.name == "running"
    assert activity.person_id == 2
    assert activity.person_label == "Person 2"
    assert "inferred" in activity.association_note


def test_activity_left_unassigned_when_no_person_overlaps() -> None:
    raw = _raw_payload(
        labels=[_label("walking", ("0:00:04", "0:00:12", 0.91))],
        observed_people=[_observed_person(1, ("0:00:40", "0:00:50"))],  # no overlap
    )

    analysis = av.analyze_insights(raw, video_filename="demo.mp4")

    assert len(analysis.activities) == 1
    activity = analysis.activities[0]
    assert activity.person_id is None
    assert activity.person_label is None
    assert "no observed-people appearance overlapped" in activity.association_note


def test_activity_left_unassigned_when_multiple_people_overlap() -> None:
    raw = _raw_payload(
        labels=[_label("talking", ("0:00:32", "0:00:47", 0.79))],
        observed_people=[
            _observed_person(1, ("0:00:30", "0:00:50")),
            _observed_person(2, ("0:00:35", "0:00:45")),
        ],
    )

    analysis = av.analyze_insights(raw, video_filename="demo.mp4")

    assert len(analysis.activities) == 1
    activity = analysis.activities[0]
    assert activity.person_id is None
    assert "ambiguous" in activity.association_note


def test_non_activity_labels_excluded_from_activities_but_kept_in_labels() -> None:
    raw = _raw_payload(
        labels=[
            _label("outdoor", ("0:00:00", "0:01:00", 0.99)),
            _label("running", ("0:00:05", "0:00:10", 0.9)),
        ]
    )

    analysis = av.analyze_insights(raw, video_filename="demo.mp4")

    assert len(analysis.labels) == 2  # full unfiltered list
    assert len(analysis.activities) == 1  # only "running" promoted
    assert analysis.activities[0].name == "running"


def test_matched_face_name_used_when_video_indexer_links_them() -> None:
    raw = _raw_payload(
        labels=[_label("dancing", ("0:00:00", "0:00:05", 0.8))],
        observed_people=[
            _observed_person(
                1,
                ("0:00:00", "0:00:05"),
                matching_face_id=7,
                matching_face_confidence=0.93,
            )
        ],
        faces=[{"id": 7, "name": "Jane Doe"}],
    )

    analysis = av.analyze_insights(raw, video_filename="demo.mp4")

    assert analysis.people[0].label == "Jane Doe"
    assert analysis.people[0].matched_face_confidence == 0.93
    assert analysis.activities[0].person_label == "Jane Doe"


# ----------------------------------------------------------------------
# Normalized JSON shape
# ----------------------------------------------------------------------


def test_to_activities_dict_matches_required_shape_and_has_no_bounding_boxes() -> None:
    raw = _raw_payload(
        labels=[_label("running", ("0:00:18", "0:00:27", 0.87))],
        observed_people=[_observed_person(2, ("0:00:15", "0:00:30"))],
    )
    analysis = av.analyze_insights(raw, video_filename="demo.mp4")

    data = av.to_activities_dict(analysis)
    json.dumps(data)  # must be JSON-serializable

    assert data["video"] == "demo.mp4"
    assert set(data.keys()) >= {"video", "people", "activities", "objects", "labels"}
    activity = data["activities"][0]
    assert activity["person_label"] == "Person 2"
    assert activity["bounding_box"] is None
    assert activity["association"]["provided_by_video_indexer"] is False


# ----------------------------------------------------------------------
# Human-readable report
# ----------------------------------------------------------------------


def test_render_report_matches_example_line_format() -> None:
    raw = _raw_payload(
        labels=[
            _label("walking", ("0:00:04", "0:00:12", 0.91)),
            _label("running", ("0:00:18", "0:00:27", 0.87)),
        ],
        observed_people=[_observed_person(2, ("0:00:15", "0:00:30"))],
    )
    analysis = av.analyze_insights(raw, video_filename="customer_demo.mp4")

    report = av.render_report(analysis)

    assert "Video: customer_demo.mp4" in report
    assert "People detected: 1" in report
    assert "Activities detected: 2" in report
    assert "00:04 - 00:12   walking   confidence: 0.91" in report
    assert "00:18 - 00:27   Person 2: running   confidence: 0.87" in report
    assert "ACTIVITY SUMMARY" in report
    assert "walking: 1 occurrence" in report
    assert "NOTES" in report
    assert "not output from a dedicated human-action-recognition model" in report


def test_render_report_handles_no_activities_gracefully() -> None:
    raw = _raw_payload(labels=[_label("outdoor", ("0:00:00", "0:00:05", 0.9))])
    analysis = av.analyze_insights(raw, video_filename="quiet.mp4")

    report = av.render_report(analysis)

    assert "Activities detected: 0" in report
    assert "No activity-shaped labels were detected" in report


def test_render_report_includes_summary_section() -> None:
    raw = _raw_payload(
        labels=[
            _label("walking", ("0:00:04", "0:00:12", 0.91)),
            _label("running", ("0:00:18", "0:00:27", 0.87)),
        ],
        observed_people=[_observed_person(2, ("0:00:15", "0:00:30"))],
    )
    analysis = av.analyze_insights(raw, video_filename="customer_demo.mp4")

    report = av.render_report(analysis)

    assert "SUMMARY" in report
    assert report.index("SUMMARY") < report.index("PEOPLE")  # summary comes first


# ----------------------------------------------------------------------
# build_summary: plain-English summary paragraph
# ----------------------------------------------------------------------


def test_build_summary_handles_no_people_or_activities() -> None:
    analysis = av.analyze_insights(
        _raw_payload(duration_seconds=10.0), video_filename="quiet.mp4"
    )

    summary = av.build_summary(analysis)

    assert "doesn't have any tracked people" in summary
    assert "No activity-shaped labels came back for it" in summary


def test_build_summary_reports_dominant_activity_and_full_association() -> None:
    raw = _raw_payload(
        labels=[_label("running", ("0:00:18", "0:00:27", 0.87))],
        observed_people=[_observed_person(2, ("0:00:15", "0:00:30"))],
    )
    analysis = av.analyze_insights(raw, video_filename="demo.mp4")

    summary = av.build_summary(analysis)

    assert "tracks one person" in summary
    assert "flag one activity here: running" in summary
    assert "That lines up cleanly with the one tracked person" in summary


def test_build_summary_reports_partial_association() -> None:
    raw = _raw_payload(
        labels=[
            _label("running", ("0:00:18", "0:00:27", 0.87)),  # 1 overlap -> assigned
            _label("talking", ("0:00:32", "0:00:47", 0.79)),  # 2 overlaps -> unassigned
        ],
        observed_people=[
            _observed_person(1, ("0:00:15", "0:00:30")),
            _observed_person(2, ("0:00:30", "0:00:50")),
            _observed_person(3, ("0:00:35", "0:00:45")),
        ],
    )
    analysis = av.analyze_insights(raw, video_filename="demo.mp4")

    summary = av.build_summary(analysis)

    assert "Of those, one lines up with a specific tracked person" in summary
    assert "the other one doesn't" in summary


def test_build_summary_reads_as_natural_prose_not_a_template() -> None:
    # Guards against regressing back to the old robotic phrasing this was
    # explicitly rewritten away from (e.g. "1 tracked person",
    # "activity-shaped label occurrences across N distinct activities").
    raw = _raw_payload(
        labels=[_label("running", ("0:00:18", "0:00:27", 0.87))],
        observed_people=[_observed_person(2, ("0:00:15", "0:00:30"))],
    )
    analysis = av.analyze_insights(raw, video_filename="demo.mp4")

    summary = av.build_summary(analysis)

    assert "1 tracked person" not in summary
    assert "occurrences across" not in summary


# ----------------------------------------------------------------------
# Batch / skip / --force orchestration
# ----------------------------------------------------------------------


class FakeVideoIndexerClient:
    """Stands in for VideoIndexerClient.index_video -- returns a canned raw
    payload and records how many times it was actually called, so tests
    can assert on skip-vs-reprocess behavior without any network access."""

    def __init__(self, raw: dict[str, Any]):
        self._raw = raw
        self.index_video_calls: list[str] = []

    def index_video(self, video_path: Path, name: str) -> dict[str, Any]:
        self.index_video_calls.append(name)
        return self._raw


def _simple_raw_payload() -> dict[str, Any]:
    return _raw_payload(labels=[_label("running", ("0:00:00", "0:00:05", 0.9))])


def test_process_video_writes_all_three_artifacts(tmp_path: Path) -> None:
    client = FakeVideoIndexerClient(_simple_raw_payload())
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake video bytes")
    out_dir = tmp_path / "output"

    ran = av.process_video(client, video_path, out_dir, force=False)

    assert ran is True
    assert (out_dir / "demo.insights.json").is_file()
    assert (out_dir / "demo.activities.json").is_file()
    assert (out_dir / "demo.report.txt").is_file()
    assert client.index_video_calls == ["demo"]


def test_process_video_skips_when_already_processed(tmp_path: Path) -> None:
    client = FakeVideoIndexerClient(_simple_raw_payload())
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake video bytes")
    out_dir = tmp_path / "output"

    av.process_video(client, video_path, out_dir, force=False)
    ran_again = av.process_video(client, video_path, out_dir, force=False)

    assert ran_again is False
    assert client.index_video_calls == ["demo"]  # not called a second time


def test_process_video_force_reprocesses(tmp_path: Path) -> None:
    client = FakeVideoIndexerClient(_simple_raw_payload())
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake video bytes")
    out_dir = tmp_path / "output"

    av.process_video(client, video_path, out_dir, force=False)
    ran_again = av.process_video(client, video_path, out_dir, force=True)

    assert ran_again is True
    assert client.index_video_calls == ["demo", "demo"]


class FakeIdVideoIndexerClient:
    """Stands in for VideoIndexerClient.wait_for_processing -- used to test
    process_video_by_id's fetch-only, no-upload path without any network
    access. Records how many times wait_for_processing was actually
    called, so tests can assert on skip-vs-reprocess behavior."""

    def __init__(self, raw: dict[str, Any]):
        self._raw = raw
        self.wait_for_processing_calls: list[str] = []

    def wait_for_processing(self, video_id: str) -> dict[str, Any]:
        self.wait_for_processing_calls.append(video_id)
        return self._raw


def test_process_video_by_id_writes_all_three_artifacts_using_returned_name(
    tmp_path: Path,
) -> None:
    raw = _simple_raw_payload()
    raw["name"] = "already-indexed-clip"
    client = FakeIdVideoIndexerClient(raw)
    out_dir = tmp_path / "output"

    ran = av.process_video_by_id(client, "abc123", out_dir, force=False)

    assert ran is True
    assert (out_dir / "already-indexed-clip.insights.json").is_file()
    assert (out_dir / "already-indexed-clip.activities.json").is_file()
    assert (out_dir / "already-indexed-clip.report.txt").is_file()
    assert client.wait_for_processing_calls == ["abc123"]


def test_process_video_by_id_falls_back_to_video_id_when_name_missing(
    tmp_path: Path,
) -> None:
    client = FakeIdVideoIndexerClient(_simple_raw_payload())  # no "name" key
    out_dir = tmp_path / "output"

    av.process_video_by_id(client, "abc123", out_dir, force=False)

    assert (out_dir / "abc123.insights.json").is_file()


def test_process_video_by_id_skips_when_already_processed(tmp_path: Path) -> None:
    raw = _simple_raw_payload()
    raw["name"] = "already-indexed-clip"
    client = FakeIdVideoIndexerClient(raw)
    out_dir = tmp_path / "output"

    av.process_video_by_id(client, "abc123", out_dir, force=False)
    ran_again = av.process_video_by_id(client, "abc123", out_dir, force=False)

    assert ran_again is False
    # wait_for_processing is still called both times (the stem/name isn't
    # known until after the fetch), but no extra files are written.
    assert client.wait_for_processing_calls == ["abc123", "abc123"]


def test_process_video_by_id_force_reprocesses(tmp_path: Path) -> None:
    raw = _simple_raw_payload()
    raw["name"] = "already-indexed-clip"
    client = FakeIdVideoIndexerClient(raw)
    out_dir = tmp_path / "output"

    av.process_video_by_id(client, "abc123", out_dir, force=False)
    ran_again = av.process_video_by_id(client, "abc123", out_dir, force=True)

    assert ran_again is True


def test_print_activity_associations_logs_person_activity_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = _raw_payload(
        labels=[_label("running", ("0:00:18", "0:00:27", 0.87))],
        observed_people=[_observed_person(2, ("0:00:15", "0:00:30"))],
    )
    analysis = av.analyze_insights(raw, video_filename="demo.mp4")

    with caplog.at_level("INFO", logger="analyze_video"):
        av._print_activity_associations(analysis)

    assert "00:18 - 00:27   Person 2: running   confidence: 0.87" in caplog.text


def test_print_activity_associations_handles_no_activities(
    caplog: pytest.LogCaptureFixture,
) -> None:
    analysis = av.analyze_insights(_raw_payload(), video_filename="quiet.mp4")

    with caplog.at_level("INFO", logger="analyze_video"):
        av._print_activity_associations(analysis)

    assert "no activity-shaped labels detected" in caplog.text


def test_build_arg_parser_accepts_video_id() -> None:
    args = av.build_arg_parser().parse_args(["--video-id", "abc123"])
    assert args.video_id == "abc123"
    assert args.video is None


def test_main_rejects_video_and_video_id_together(tmp_path: Path) -> None:
    exit_code = av.main(
        [
            "--video",
            "does-not-matter.mp4",
            "--video-id",
            "abc123",
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert exit_code == 2


def test_discover_batch_videos_finds_only_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"x")
    (tmp_path / "clip.mov").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / "clip.MP4").write_bytes(b"x")  # case-insensitive match

    found = av._discover_batch_videos(tmp_path)

    names = sorted(p.name for p in found)
    assert names == ["clip.MP4", "clip.mov", "clip.mp4"]


def test_discover_batch_videos_empty_when_dir_missing(tmp_path: Path) -> None:
    assert av._discover_batch_videos(tmp_path / "does_not_exist") == []


# ----------------------------------------------------------------------
# Settings / config validation (no Azure account needed)
# ----------------------------------------------------------------------


def test_settings_from_env_raises_config_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "AVI_SUBSCRIPTION_ID",
        "AVI_RESOURCE_GROUP",
        "AVI_ACCOUNT_NAME",
        "AVI_ACCOUNT_ID",
        "AVI_LOCATION",
    ):
        monkeypatch.delenv(name, raising=False)

    try:
        av.Settings.from_env()
        assert False, "expected ConfigError"
    except av.ConfigError as exc:
        assert "AVI_SUBSCRIPTION_ID" in str(exc)


def test_settings_from_env_reads_all_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVI_SUBSCRIPTION_ID", "sub-id")
    monkeypatch.setenv("AVI_RESOURCE_GROUP", "rg")
    monkeypatch.setenv("AVI_ACCOUNT_NAME", "account")
    monkeypatch.setenv("AVI_ACCOUNT_ID", "account-id")
    monkeypatch.setenv("AVI_LOCATION", "eastus")

    settings = av.Settings.from_env()

    assert settings.subscription_id == "sub-id"
    assert settings.indexing_preset == "Advanced"  # default
    assert settings.resource_group == "rg"
    assert settings.account_name == "account"
    assert settings.account_id == "account-id"
    assert settings.location == "eastus"
