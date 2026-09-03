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


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if not args.video and not args.video_id:
        logger.error(
            "Either --video (to upload/read frames from) or --video-id (to "
            "reuse an already-indexed video, downloading its source file for "
            "frames) is required."
        )
        return 2

    if args.video and not args.video.is_file():
        logger.error("Video file not found: %s", args.video)
        return 2

    if not (0.0 <= args.min_confidence <= 1.0):
        logger.error("--min-confidence must be between 0.0 and 1.0.")
        return 2

    # A client is needed whenever Azure has to do something: fetch/build the
    # insights (unless --report covers that) or download the source file
    # (unless a local --video covers that).
    need_client = (not args.report) or (not args.video)
    client: VideoIndexerClient | None = None
    if need_client:
        try:
            settings = Settings.from_env()
        except ConfigError as exc:
            logger.error(str(exc))
            return 2
        client = VideoIndexerClient(settings)

    try:
        if args.report:
            if not args.report.is_file():
                logger.error("Report file not found: %s", args.report)
                return 2
            logger.info(
                "Loading actions from %s (no Video Indexer insights call needed)",
                args.report,
            )
            actions = _load_actions_from_report(args.report)
        elif args.video_id:
            assert client is not None  # need_client is True on this branch
            logger.info("Reusing existing video id=%s", args.video_id)
            index = client.wait_for_processing(args.video_id)
            actions = analyze_all(index).actions
        else:
            assert client is not None  # need_client is True on this branch
            video_id = client.upload_video(args.video, name=args.name)
            logger.info(
                "Video uploaded (id=%s). Save this ID to re-run without "
                "re-uploading via --video-id.",
                video_id,
            )
            index = client.wait_for_processing(video_id)
            actions = analyze_all(index).actions
    except (VideoIndexerError, VideoAnnotationError) as exc:
        logger.error("Could not obtain actions: %s", exc)
        return 1

    if args.only_composite:
        actions = _filter_only_composite(actions, only_composite=True)
        logger.info(
            "--only-composite: %d composite/derived action(s) to draw.", len(actions)
        )
        if not actions:
            logger.warning(
                "No composite actions found -- the overlay will be empty. "
                "Composite actions only exist where a rule in COMPOSITE_ACTIONS "
                "(src/constants.py) actually overlapped two detections; see "
                "README 'Composite (derived) actions'."
            )

    # Base name for default output filenames, and the frame source itself.
    base_name = args.video.stem if args.video else (args.name or args.video_id)
    if args.only_composite:
        base_name = f"{base_name}.composite_actions"

    if args.video:
        frame_source = args.video
    else:
        assert client is not None  # need_client is True whenever --video is omitted
        downloaded_path = args.out_dir / f"{args.video_id}.source.mp4"
        try:
            frame_source = client.download_video(args.video_id, downloaded_path)
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
