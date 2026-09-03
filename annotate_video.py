#!/usr/bin/env python3
"""
CLI: render an annotated copy of a video, with a burned-in overlay showing
every action Azure AI Video Indexer detected above a configurable
confidence threshold, at the moment(s) it was detected.

Only one of --video / --video-id is required, not both:
    - --video alone           uploads and indexes it from scratch, then
                               annotates that same local file.
    - --video-id alone        reuses an already-indexed video's insights,
                               and downloads its source file from Video
                               Indexer to get frames to annotate (no local
                               copy needed on hand).
    - --video AND --video-id  reuses the insights (skips re-uploading) but
                               reads frames from the local file (skips the
                               download).
    - --report + --video      reuses a saved report for the actions (no
                               Azure calls at all).
    - --report + --video-id   reuses a saved report for the actions, but
                               still downloads the source file for frames.

--report accepts either report file this project writes: detect_all_actions.py's
*.all_actions.json (every detected action) or main.py's *.report.json (just
the occurrences of one --action of interest).

--only-composite restricts the overlay to composite/derived actions only
(e.g. "person using phone", "person handling cash") -- see README
'Composite (derived) actions'. Combine it with --report/--video-id to
annotate every composite the video has, or with a single-action --report
from main.py to isolate just one.

Examples:
    python annotate_video.py --video-id tuxagrberr --min-confidence 0.6
    python annotate_video.py --video tuxagrberr.mp4 --report output/tuxagrberr.all_actions.json
    python .\annotate_video.py --video-id f9m21irr9l --report ./output/f9m21irr9l.report.json --min-confidence 0.7
    python annotate_video.py --video tuxagrberr.mp4 --report output/tuxagrberr.all_actions.json --only-composite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.action_analyzer import DetectedAction, analyze_all
from src.config import ConfigError, Settings
from src.video_annotator import (
    DEFAULT_MIN_CONFIDENCE,
    VideoAnnotationError,
    render_annotated_video,
)
from src.video_indexer_client import VideoIndexerClient, VideoIndexerError

logger = logging.getLogger("azure_action_monitoring.annotate_video")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a video with a burned-in overlay of every action "
        "detected above a confidence threshold."
    )
    parser.add_argument(
        "--video",
        default=None,
        type=Path,
        help="Local video file to annotate. Optional if --video-id is given: "
        "its source file will be downloaded from Video Indexer instead.",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="Reuse an already-indexed video's insights instead of re-uploading "
        "--video. Also required (instead of --video) to have frames downloaded "
        "from Video Indexer for you.",
    )
    parser.add_argument(
        "--report",
        default=None,
        type=Path,
        help="Reuse a saved report for the actions instead of calling Video "
        "Indexer's insights API: either detect_all_actions.py's "
        "*.all_actions.json (every action) or main.py's *.report.json "
        "(occurrences of one --action of interest).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name to register the upload under, if uploading. Default: the video file's stem.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Only annotate actions with confidence >= this (0.0-1.0). Default: {DEFAULT_MIN_CONFIDENCE}.",
    )
    parser.add_argument(
        "--include-unscored",
        action="store_true",
        help="Also annotate actions with no confidence value (excluded by default, "
        "since a threshold can't be applied to them).",
    )
    parser.add_argument(
        "--only-composite",
        action="store_true",
        help='Only draw composite/derived actions (source "derived", e.g. '
        "'person using phone', 'person handling cash') -- drops every raw "
        "label/keyword/object/ocr detection so the overlay shows just the "
        "synthesized actions. See README 'Composite (derived) actions'.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=10,
        help="Max simultaneous action lines drawn per frame before collapsing to '+N more'. Default: 10.",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=Path,
        help="Output video path. Default: output/<name>.annotated.mp4",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("output"),
        type=Path,
        help="Directory for default outputs (annotated video, and a downloaded "
        "source file when --video is omitted). Default: ./output",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio muxing even if ffmpeg is available (faster; silent output).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser


def _load_actions_from_report(report_path: Path) -> list[DetectedAction]:
    """Load actions to draw from a saved report JSON file.

    Accepts either report format this project produces:
      - detect_all_actions.py's *.all_actions.json: a top-level "actions"
        list, each item keyed by "name" -- every action detected.
      - main.py's *.report.json: a top-level "events" list, each item keyed
        by "matched_term" -- occurrences of the one action searched for via
        --action. (Both use the same "source"/"start_seconds"/"end_seconds"/
        "confidence" fields otherwise.)
    """
    data = json.loads(report_path.read_text(encoding="utf-8"))

    actions = data.get("actions")
    if actions is not None:
        return [
            DetectedAction(
                name=a["name"],
                source=a["source"],
                start_seconds=a["start_seconds"],
                end_seconds=a["end_seconds"],
                confidence=a.get("confidence"),
            )
            for a in actions
        ]

    events = data.get("events")
    if events is not None:
        return [
            DetectedAction(
                name=e["matched_term"],
                source=e["source"],
                start_seconds=e["start_seconds"],
                end_seconds=e["end_seconds"],
                confidence=e.get("confidence"),
            )
            for e in events
        ]

    raise VideoAnnotationError(
        f"{report_path} doesn't look like a report from this project (no "
        "top-level 'actions' list as written by detect_all_actions.py, or "
        "'events' list as written by main.py)."
    )


def _filter_only_composite(
    actions: list[DetectedAction], only_composite: bool
) -> list[DetectedAction]:
    """Apply --only-composite: keep just source=="derived" actions (composite
    actions synthesized by _derive_composite_actions) when set, otherwise
    return `actions` unchanged."""
    if not only_composite:
        return actions
    return [a for a in actions if a.source == "derived"]


def _validate_args(args: argparse.Namespace) -> str | None:
    """Return a human-readable error message if `args` fail validation
    (beyond what argparse itself enforces), or None if they're all fine.
    Kept separate from `main` so each check is one flat, easily-testable
    statement rather than nested inside a long function body. Checking
    `--report`'s existence here too (rather than only once its branch
    actually runs) means a bad path fails fast, before any Azure/client
    setup that might not even be needed."""
    if not args.video and not args.video_id:
        return (
            "Either --video (to upload/read frames from) or --video-id (to "
            "reuse an already-indexed video, downloading its source file for "
            "frames) is required."
        )
    if args.video and not args.video.is_file():
        return f"Video file not found: {args.video}"
    if not (0.0 <= args.min_confidence <= 1.0):
        return "--min-confidence must be between 0.0 and 1.0."
    if args.report and not args.report.is_file():
        return f"Report file not found: {args.report}"
    return None


def _acquire_actions(
    client: VideoIndexerClient | None, args: argparse.Namespace
) -> list[DetectedAction]:
    """Get the actions to draw: from a saved --report (no Azure calls
    needed at all), by reusing --video-id's insights, or by uploading
    --video and waiting for processing. Raises
    VideoIndexerError/VideoAnnotationError on failure; left for `main` to
    turn into a clean exit code rather than a traceback."""
    if args.report:
        logger.info(
            "Loading actions from %s (no Video Indexer insights call needed)",
            args.report,
        )
        return _load_actions_from_report(args.report)

    assert client is not None  # need_client is True whenever --report is absent
    if args.video_id:
        logger.info("Reusing existing video id=%s", args.video_id)
        index = client.wait_for_processing(args.video_id)
    else:
        video_id = client.upload_video(args.video, name=args.name)
        logger.info(
            "Video uploaded (id=%s). Save this ID to re-run without "
            "re-uploading via --video-id.",
            video_id,
        )
        index = client.wait_for_processing(video_id)
    return analyze_all(index).actions


def _resolve_frame_source(
    client: VideoIndexerClient | None, args: argparse.Namespace
) -> Path:
    """The video file to read frames from: the local --video if given,
    otherwise --video-id's source file downloaded from Video Indexer.
    Raises VideoIndexerError on a failed download."""
    if args.video:
        return args.video
    assert client is not None  # need_client is True whenever --video is omitted
    downloaded_path = args.out_dir / f"{args.video_id}.source.mp4"
    return client.download_video(args.video_id, downloaded_path)


def _build_client_if_needed(args: argparse.Namespace) -> VideoIndexerClient | None:
    """Build a VideoIndexerClient when Azure has to do something: fetch/
    build the insights (unless --report covers that) or download the
    source file (unless a local --video covers that). Returns None when
    nothing needs one (both covered), so `main` never pays for/validates
    Azure config it won't use. Raises ConfigError if credentials aren't
    configured; left for `main` to turn into a clean exit code."""
    need_client = (not args.report) or (not args.video)
    if not need_client:
        return None
    settings = Settings.from_env()
    return VideoIndexerClient(settings)


def _apply_only_composite_filter(
    actions: list[DetectedAction], only_composite: bool
) -> list[DetectedAction]:
    """Apply --only-composite (keep just source=="derived" actions) and log
    the outcome, including a warning if that leaves nothing to draw."""
    if not only_composite:
        return actions
    filtered = _filter_only_composite(actions, only_composite=True)
    logger.info(
        "--only-composite: %d composite/derived action(s) to draw.", len(filtered)
    )
    if not filtered:
        logger.warning(
            "No composite actions found -- the overlay will be empty. "
            "Composite actions only exist where a rule in COMPOSITE_ACTIONS "
            "(src/constants.py) actually overlapped two detections; see "
            "README 'Composite (derived) actions'."
        )
    return filtered


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    error = _validate_args(args)
    if error:
        logger.error(error)
        return 2

    try:
        client = _build_client_if_needed(args)
    except ConfigError as exc:
        logger.error(str(exc))
        return 2

    try:
        actions = _acquire_actions(client, args)
    except (VideoIndexerError, VideoAnnotationError) as exc:
        logger.error("Could not obtain actions: %s", exc)
        return 1

    actions = _apply_only_composite_filter(actions, args.only_composite)

    # Base name for default output filenames, and the frame source itself.
    base_name = args.video.stem if args.video else (args.name or args.video_id)
    if args.only_composite:
        base_name = f"{base_name}.composite_actions"

    try:
        frame_source = _resolve_frame_source(client, args)
    except VideoIndexerError as exc:
        logger.error("Could not download source video for frames: %s", exc)
        return 1

    out_path = args.out or (args.out_dir / f"{base_name}.annotated.mp4")

    try:
        written = render_annotated_video(
            video_path=frame_source,
            actions=actions,
            output_path=out_path,
            min_confidence=args.min_confidence,
            include_unscored=args.include_unscored,
            max_lines=args.max_lines,
            keep_audio=not args.no_audio,
        )
    except VideoAnnotationError as exc:
        logger.error(str(exc))
        return 1

    logger.info("Wrote annotated video to %s", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
