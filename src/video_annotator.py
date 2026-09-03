"""
Burns "action detected" overlays into a video's frames, based on the
timestamped actions produced by src/action_analyzer.py (analyze_all).

Honest caveat: Video Indexer's labels/keywords are frame-level tags, not
spatial bounding boxes, so what gets drawn is a text overlay (an on-screen
caption box listing whatever is active at that moment) rather than boxes
drawn around the person/object doing it -- that's what the underlying
insights actually contain.

The frame-selection logic here is deliberately independent of OpenCV so
it's unit-testable without a real video file: `filter_by_confidence` and
`active_actions_at` are pure functions; `render_annotated_video` is the
only piece that touches cv2/ffmpeg, and imports cv2 lazily so the rest of
the project doesn't need it installed just to run the other scripts/tests.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .action_analyzer import DetectedAction
from .constants import DEFAULT_MIN_CONFIDENCE, MAX_OVERLAY_LINES

logger = logging.getLogger(__name__)


class VideoAnnotationError(RuntimeError):
    """Raised when annotating the video fails for a reason worth a clear message."""


def filter_by_confidence(
    actions: Iterable[DetectedAction],
    min_confidence: float,
    include_unscored: bool = False,
) -> list[DetectedAction]:
    """Keep only actions whose confidence is known and >= min_confidence.

    Actions with no confidence value (some Video Indexer insight buckets
    don't score every item) are dropped by default, since "above a
    threshold" implies a comparable score -- pass include_unscored=True to
    draw them regardless of the threshold.
    """
    kept: list[DetectedAction] = []
    for a in actions:
        if a.confidence is None:
            if include_unscored:
                kept.append(a)
            continue
        if a.confidence >= min_confidence:
            kept.append(a)
    return kept


def active_actions_at(
    actions: Sequence[DetectedAction], timestamp_seconds: float
) -> list[DetectedAction]:
    """Every action whose [start, end] interval covers `timestamp_seconds`,
    most-confident first (unscored actions sort last)."""
    active = [
        a for a in actions if a.start_seconds <= timestamp_seconds <= a.end_seconds
    ]
    active.sort(
        key=lambda a: a.confidence if a.confidence is not None else -1.0, reverse=True
    )
    return active


def format_overlay_lines(
    active: Sequence[DetectedAction], max_lines: int = MAX_OVERLAY_LINES
) -> list[str]:
    """Turn currently-active actions into the text lines to draw, truncating
    with a "+N more" line rather than overflowing the frame."""
    if not active:
        return []
    lines: list[str] = []
    for a in active[:max_lines]:
        conf = f"{a.confidence:.2f}" if a.confidence is not None else "n/a"
        lines.append(f"{a.name} ({conf})")
    remaining = len(active) - max_lines
    if remaining > 0:
        lines.append(f"+{remaining} more")
    return lines


def _fmt_timecode(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{int(minutes):02d}:{secs:05.2f}"


def _open_capture(cv2: Any, video_path: Path) -> Any:
    """Open `video_path` for reading, raising a clear error if it can't be."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoAnnotationError(f"Could not open video for reading: {video_path}")
    return cap


def _video_properties(
    cv2: Any, cap: Any, video_path: Path
) -> tuple[float, int, int, int]:
    """Read (fps, width, height, frame_count) from an opened capture,
    raising a clear error if the dimensions can't be read."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise VideoAnnotationError(
            f"Could not read frame dimensions from: {video_path}"
        )
    return fps, width, height, frame_count


def _write_annotated_frames(
    cv2: Any,
    cap: Any,
    writer: Any,
    actions: Sequence[DetectedAction],
    fps: float,
    max_lines: int,
    frame_count: int,
) -> int:
    """Read every frame from `cap`, draw the active-actions overlay on it,
    and write it to `writer`. Returns the number of frames written."""
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_idx / fps
            active = active_actions_at(actions, timestamp)
            _draw_overlay(
                cv2,
                frame,
                format_overlay_lines(active, max_lines),
                _fmt_timecode(timestamp),
            )
            writer.write(frame)
            frame_idx += 1
            if frame_count > 0 and frame_idx % 200 == 0:
                logger.debug("Annotated %d/%d frames", frame_idx, frame_count)
    finally:
        cap.release()
        writer.release()
    return frame_idx


def _write_output(
    video_path: Path, silent_path: Path, output_path: Path, keep_audio: bool
) -> None:
    """Produce `output_path` from the silent render, muxing the original
    audio back in via ffmpeg when it's available and requested."""
    if keep_audio and shutil.which("ffmpeg"):
        _mux_audio(video_path, silent_path, output_path)
        return
    if keep_audio:
        logger.warning(
            "ffmpeg not found on PATH; writing video-only output with no "
            "audio track. Install ffmpeg and re-run to keep the original audio."
        )
    shutil.copyfile(silent_path, output_path)


def render_annotated_video(
    video_path: Path,
    actions: Sequence[DetectedAction],
    output_path: Path,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    include_unscored: bool = False,
    max_lines: int = MAX_OVERLAY_LINES,
    keep_audio: bool = True,
) -> Path:
    """Render `video_path` with a burned-in overlay of every action active
    at each frame (confidence >= min_confidence), writing the result to
    `output_path`. Returns the actual output path written.

    Requires `opencv-python` / `opencv-python-headless` (imported lazily so
    the rest of the project doesn't need it installed). Audio is preserved
    by shelling out to `ffmpeg` if it's on PATH; otherwise the output is
    silent and a warning is logged.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise VideoAnnotationError(
            "opencv-python (or opencv-python-headless) is required to render "
            "annotated video. Install it with: pip install opencv-python-headless"
        ) from exc

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept_actions = filter_by_confidence(
        actions, min_confidence, include_unscored=include_unscored
    )
    logger.info(
        "Annotating with %d/%d actions at confidence >= %.2f%s",
        len(kept_actions),
        len(actions),
        min_confidence,
        " (unscored kept)" if include_unscored else "",
    )
    if not kept_actions:
        logger.warning(
            "No actions meet the confidence threshold -- output will have no "
            "overlays (just the timecode watermark). Lower --min-confidence "
            "or pass --include-unscored if that's not what you expected."
        )

    cap = _open_capture(cv2, video_path)
    fps, width, height, frame_count = _video_properties(cv2, cap, video_path)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    with tempfile.TemporaryDirectory() as tmpdir:
        silent_path = Path(tmpdir) / "silent.mp4"
        writer = cv2.VideoWriter(str(silent_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise VideoAnnotationError(
                "Could not open a video writer (unsupported codec/container)."
            )

        frames_written = _write_annotated_frames(
            cv2, cap, writer, kept_actions, fps, max_lines, frame_count
        )
        logger.info("Annotated %d frames", frames_written)

        _write_output(video_path, silent_path, output_path, keep_audio)

    return output_path


def _draw_overlay(cv2: Any, frame: Any, lines: list[str], timecode: str) -> None:
    """Mutate `frame` in place: a translucent caption box (top-left) listing
    active actions, plus a small timecode watermark (bottom-right)."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(0.7, w / 1600))
    thickness = 1
    line_height = int(24 * font_scale / 0.5)

    if lines:
        box_w = int(w * 0.42)
        box_h = 10 + line_height * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, text in enumerate(lines):
            y = 10 + line_height * (i + 1) - 6
            cv2.putText(
                frame,
                text,
                (18, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    (tw, th), _ = cv2.getTextSize(timecode, font, font_scale, thickness)
    cv2.rectangle(frame, (w - tw - 20, h - th - 18), (w - 4, h - 4), (0, 0, 0), -1)
    cv2.putText(
        frame,
        timecode,
        (w - tw - 12, h - 10),
        font,
        font_scale,
        (0, 255, 120),
        thickness,
        cv2.LINE_AA,
    )


def _mux_audio(original_video: Path, silent_video: Path, output_path: Path) -> None:
    """Copy the audio stream from `original_video` onto `silent_video`'s
    picture, writing the result to `output_path`. Falls back to the
    video-only file if the mux fails (e.g. the source has no audio stream)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(original_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.warning(
            "ffmpeg audio mux failed (exit %s); falling back to video-only output. stderr: %s",
            result.returncode,
            result.stderr[-2000:],
        )
        shutil.copyfile(silent_video, output_path)
