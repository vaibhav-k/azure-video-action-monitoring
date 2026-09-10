"""
Unit tests for describe_artifacts.py -- built entirely on synthetic
artifact files shaped like Azure Video Indexer's real "download
artifacts" export (labels.computervision.json, detectedobjects.json,
ocr.json, contentmoderation.json), so none of this needs a real export
or network access.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import describe_artifacts as da


def _write_json(folder: Path, filename: str, data: dict) -> None:
    (folder / filename).write_text(json.dumps(data), encoding="utf-8")


def _labels_payload(*frames: list[tuple[str, float]]) -> dict:
    return {
        "Fps": 30.0,
        "Results": [
            {
                "FrameIndex": i * 30,
                "Label": [
                    {
                        "name": name,
                        "confidence": conf,
                        "hint": None,
                        "highlyConfident": conf > 0.8,
                    }
                    for name, conf in frame
                ],
            }
            for i, frame in enumerate(frames)
        ],
    }


def _objects_payload(*tracks: tuple[str, str, str]) -> dict:
    """Each track is (type, start, end); one instance at each end."""
    return {
        "algoVersion": "test",
        "schemaVersion": "0.0.1",
        "results": [
            {
                "id": i + 1,
                "type": object_type,
                "instances": [
                    {
                        "start": start,
                        "end": start,
                        "confidence": 0.8,
                        "detection_class": object_type.lower(),
                    },
                    {
                        "start": end,
                        "end": end,
                        "confidence": 0.8,
                        "detection_class": object_type.lower(),
                    },
                ],
            }
            for i, (object_type, start, end) in enumerate(tracks)
        ],
        "low_confidence_results": [],
    }


def _ocr_payload(*contents: str) -> dict:
    return {
        "Fps": 30.0,
        "Results": [
            {"FrameIndex": i * 30, "Ocr": {"content": content}}
            for i, content in enumerate(contents)
        ],
    }


def _moderation_payload(*scores: tuple[float, float, float, bool]) -> dict:
    return {
        "Fps": 30.0,
        "Results": [
            {
                "FrameIndex": i * 30,
                "Adult": {
                    "isAdultContent": flagged,
                    "isRacyContent": False,
                    "isGoryContent": False,
                    "adultScore": adult,
                    "racyScore": racy,
                    "goreScore": gore,
                },
            }
            for i, (adult, racy, gore, flagged) in enumerate(scores)
        ],
    }


# ----------------------------------------------------------------------
# Label parsing / categorization
# ----------------------------------------------------------------------


def test_categorizes_labels_by_frame_coverage(tmp_path: Path):
    _write_json(
        tmp_path,
        da.LABELS_FILENAME,
        _labels_payload(
            [("outdoor", 0.99), ("bicycle", 0.9)],
            [("outdoor", 0.98)],
            [("outdoor", 0.97), ("car", 0.8)],
        ),
    )

    observations, sampled = da._parse_label_observations(tmp_path)
    persistent, occasional = da._categorize_labels(observations, sampled)

    assert sampled == 3
    assert [o.name for o in persistent] == ["outdoor"]  # 3/3 frames
    assert {o.name for o in occasional} == {
        "bicycle",
        "car",
    }  # 1/3 frames each, below the 0.5 threshold


def test_meta_label_text_is_excluded_from_observations(tmp_path: Path):
    _write_json(
        tmp_path, da.LABELS_FILENAME, _labels_payload([("text", 0.9), ("outdoor", 0.9)])
    )

    observations, _ = da._parse_label_observations(tmp_path)

    assert {o.name for o in observations} == {"outdoor"}


def test_missing_labels_file_returns_empty(tmp_path: Path):
    observations, sampled = da._parse_label_observations(tmp_path)
    assert observations == []
    assert sampled == 0


# ----------------------------------------------------------------------
# Object tracking
# ----------------------------------------------------------------------


def test_objects_grouped_by_type_with_span(tmp_path: Path):
    _write_json(
        tmp_path,
        da.OBJECTS_FILENAME,
        _objects_payload(
            ("Bicycle", "0:00:00", "0:00:05"),
            ("Bicycle", "0:00:02", "0:00:09.7"),
            ("Car", "0:00:03", "0:00:04"),
        ),
    )

    groups = da._parse_objects(tmp_path)

    by_type = {g.object_type: g for g in groups}
    assert by_type["Bicycle"].count == 2
    assert by_type["Bicycle"].first_seconds == 0.0
    assert by_type["Bicycle"].last_seconds == 9.7
    assert by_type["Car"].count == 1
    # Most-tracked type sorts first.
    assert groups[0].object_type == "Bicycle"


# ----------------------------------------------------------------------
# OCR snippet extraction
# ----------------------------------------------------------------------


def test_ocr_requires_recurrence_and_filters_short_noise(tmp_path: Path):
    _write_json(
        tmp_path,
        da.OCR_FILENAME,
        _ocr_payload(
            "CHETAN\nJO0",
            "CHETAN\n--",
            "CHETAN",
        ),
    )

    snippets = da._parse_ocr_snippets(tmp_path)

    texts = {s.text for s in snippets}
    assert texts == {"CHETAN"}  # "JO0" and "--" each only appear once / are noise


def test_ocr_snippets_capped_at_max_shown(tmp_path: Path):
    contents = []
    for i in range(6):
        line = f"SHOPNAME{i}"
        contents.append(f"{line}\n{line}")  # each appears twice across 2 frames
    _write_json(tmp_path, da.OCR_FILENAME, _ocr_payload(*contents))

    snippets = da._parse_ocr_snippets(tmp_path)

    assert len(snippets) <= da.MAX_OCR_SNIPPETS_SHOWN


# ----------------------------------------------------------------------
# Content moderation
# ----------------------------------------------------------------------


def test_moderation_reports_max_scores_and_flags(tmp_path: Path):
    _write_json(
        tmp_path,
        da.MODERATION_FILENAME,
        _moderation_payload((0.01, 0.02, 0.01, False), (0.03, 0.9, 0.01, True)),
    )

    moderation = da._parse_moderation(tmp_path)

    assert moderation is not None
    assert moderation.any_flagged is True
    assert moderation.max_racy_score == 0.9


def test_missing_moderation_file_returns_none(tmp_path: Path):
    assert da._parse_moderation(tmp_path) is None


# ----------------------------------------------------------------------
# Natural-language description
# ----------------------------------------------------------------------


def test_setting_sentence_excludes_technical_and_redundant_labels():
    observations = [
        da.LabelObservation("land vehicle", 10, 0.9),
        da.LabelObservation("wheel", 10, 0.9),
        da.LabelObservation("person", 10, 0.9),
        da.LabelObservation("street", 10, 0.9),
    ]

    sentence = da._setting_sentence(observations, video_span_seconds=20.0)

    assert "land vehicle" not in sentence
    assert "wheel" not in sentence
    assert "person" not in sentence  # "people" preferred when both present
    assert "street" in sentence
    assert "20-second" in sentence


def test_setting_sentence_handles_no_persistent_labels():
    assert "isn't very clear" in da._setting_sentence([], video_span_seconds=None)


def test_setting_sentence_hedges_clip_length_as_approximate():
    sentence = da._setting_sentence(
        [da.LabelObservation("street", 10, 0.9)], video_span_seconds=20.0
    )
    assert "roughly" in sentence


def test_setting_sentence_falls_back_when_span_unknown():
    sentence = da._setting_sentence(
        [da.LabelObservation("street", 10, 0.9)], video_span_seconds=None
    )
    assert sentence.startswith("This clip looks like")


# ----------------------------------------------------------------------
# Pluralization
# ----------------------------------------------------------------------


def test_pluralize_regular_and_sibilant_endings():
    assert da._pluralize("car", 1) == "car"
    assert da._pluralize("car", 2) == "cars"
    assert da._pluralize("bus", 1) == "bus"
    assert da._pluralize("bus", 2) == "buses"  # not the naive "buss"


def test_objects_sentence_pluralizes_bus_correctly():
    objects = [
        da.TrackedObjectGroup(
            object_type="Bus", count=2, first_seconds=0.0, last_seconds=5.0
        )
    ]
    sentence = da._objects_sentence(objects)
    assert sentence is not None
    assert "buses" in sentence
    assert "buss" not in sentence


def test_describe_scene_skips_sections_with_nothing_to_say(tmp_path: Path):
    _write_json(tmp_path, da.LABELS_FILENAME, _labels_payload([("outdoor", 0.99)]))
    # No objects, OCR, or moderation files at all.

    summary = da.build_artifacts_summary(tmp_path)
    description = da.describe_scene(summary)

    assert "outdoor" in description
    assert "Object tracking" not in description
    assert "Text recognition" not in description
    assert "content screening" not in description
    assert da._CAVEAT_SENTENCE in description  # always present


def test_describe_scene_full_pipeline_reads_as_prose(tmp_path: Path):
    _write_json(
        tmp_path,
        da.LABELS_FILENAME,
        _labels_payload(
            [("outdoor", 0.99), ("street", 0.95), ("rickshaw", 0.8)],
            [("outdoor", 0.98), ("street", 0.9)],
        ),
    )
    _write_json(
        tmp_path,
        da.OBJECTS_FILENAME,
        _objects_payload(("Bicycle", "0:00:00", "0:00:05")),
    )
    _write_json(
        tmp_path, da.OCR_FILENAME, _ocr_payload("SHOP NAME\nSHOP NAME", "SHOP NAME")
    )
    _write_json(
        tmp_path, da.MODERATION_FILENAME, _moderation_payload((0.01, 0.01, 0.01, False))
    )

    summary = da.build_artifacts_summary(tmp_path)
    description = da.describe_scene(summary)

    assert "outdoor" in description
    assert "one bicycle" in description
    assert "SHOP NAME" in description
    assert "didn't flag anything" in description
    # No robotic template artifacts.
    assert "1 of" not in description
    assert "occurrences across" not in description


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def test_main_writes_description_file(tmp_path: Path):
    _write_json(tmp_path, da.LABELS_FILENAME, _labels_payload([("outdoor", 0.99)]))

    exit_code = da.main(["--folder", str(tmp_path)])

    assert exit_code == 0
    output_path = tmp_path.with_name(tmp_path.name + ".description.txt")
    assert output_path.is_file()
    assert "outdoor" in output_path.read_text(encoding="utf-8")


def test_main_reports_error_for_missing_folder(tmp_path: Path):
    exit_code = da.main(["--folder", str(tmp_path / "does_not_exist")])
    assert exit_code == 2
