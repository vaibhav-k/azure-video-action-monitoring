"""
Unit tests for src/video_annotator.py's pure overlay-selection logic, plus an
integration smoke test of render_annotated_video against a tiny synthetic
clip. The smoke test is skipped (not failed) if opencv/numpy aren't
installed, since annotate_video.py is the one script in this project with a
real third-party dependency beyond requests/azure-identity.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.action_analyzer import DetectedAction
from src.video_annotator import (
    DEFAULT_MIN_CONFIDENCE,
    active_actions_at,
    filter_by_confidence,
    format_overlay_lines,
)


def _action(name: str, start: float, end: float, confidence: float | None):
    return DetectedAction(
        name=name,
        source="labels",
        start_seconds=start,
        end_seconds=end,
        confidence=confidence,
    )


ACTIONS = [
    _action("jumping", 1.0, 2.0, 0.9),
    _action("boxing", 1.5, 3.0, 0.6),
    _action("person", 0.0, 5.0, 0.99),
    _action("outdoor", 0.0, 5.0, None),
    _action("trampoline", 1.0, 4.0, 0.3),
]


def test_filter_by_confidence_default_drops_unscored_and_low_confidence():
    kept = filter_by_confidence(ACTIONS, min_confidence=0.5)
    names = {a.name for a in kept}
    assert names == {
        "jumping",
        "boxing",
        "person",
    }  # trampoline (0.3) and outdoor (None) dropped


def test_filter_by_confidence_can_keep_unscored():
    kept = filter_by_confidence(ACTIONS, min_confidence=0.5, include_unscored=True)
    names = {a.name for a in kept}
    assert "outdoor" in names
    assert "trampoline" not in names  # still below threshold


def test_default_min_confidence_constant():
    assert DEFAULT_MIN_CONFIDENCE == 0.5


def test_active_actions_at_returns_overlapping_actions_most_confident_first():
    active = active_actions_at(ACTIONS, timestamp_seconds=1.7)
    names = [a.name for a in active]
    # person(0.99)/outdoor(None) cover 0-5, jumping(0.9) 1-2, boxing(0.6) 1.5-3,
    # trampoline(0.3) 1-4 all cover 1.7. active_actions_at itself does no
    # confidence filtering (that's filter_by_confidence's job, applied
    # upstream in the real pipeline) -- unscored actions just sort last.
    assert names == ["person", "jumping", "boxing", "trampoline", "outdoor"]


def test_active_actions_at_excludes_actions_outside_their_interval():
    active = active_actions_at(ACTIONS, timestamp_seconds=4.5)
    names = {a.name for a in active}
    assert names == {"person", "outdoor"}


def test_format_overlay_lines_includes_confidence_and_truncates():
    active = active_actions_at(ACTIONS, timestamp_seconds=1.7)
    lines = format_overlay_lines(active, max_lines=2)
    assert lines[0].startswith("person (0.99)")
    assert lines[1].startswith("jumping (0.90)")
    assert lines[2] == "+3 more"


def test_format_overlay_lines_empty_when_nothing_active():
    assert format_overlay_lines([]) == []


def test_render_annotated_video_smoke(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")

    from src.video_annotator import render_annotated_video

    src_path: Path = tmp_path / "src.mp4"
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(src_path), fourcc, 10, (64, 48))
    for _ in range(20):
        frame = numpy.zeros((48, 64, 3), dtype=numpy.uint8)
        writer.write(frame)
    writer.release()

    out_path: Path = tmp_path / "out.mp4"
    actions = [_action("test-action", 0.0, 2.0, 0.9)]
    written = render_annotated_video(
        video_path=src_path,
        actions=actions,
        output_path=out_path,
        min_confidence=0.5,
        keep_audio=False,
    )

    assert written == out_path
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_render_annotated_video_missing_file_raises(tmp_path: Path):
    pytest.importorskip("cv2")
    from src.video_annotator import VideoAnnotationError, render_annotated_video

    with pytest.raises(VideoAnnotationError):
        render_annotated_video(
            video_path=tmp_path / "does_not_exist.mp4",
            actions=[],
            output_path=tmp_path / "out.mp4",
            keep_audio=False,
        )
