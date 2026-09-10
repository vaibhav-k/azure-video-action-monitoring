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
    - --report + --video      reuses a saved detect_all_actions.py report
                               for the actions (no Azure calls at all).
    - --report + --video-id   reuses a saved report for the actions, but
                               still downloads the source file for frames.

Examples:
    python annotate_video.py --video-id tuxagrberr --min-confidence 0.6
    python annotate_video.py --video tuxagrberr.mp4 --report output/tuxagrberr.all_actions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Protocol

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
        help="Reuse a *.all_actions.json from detect_all_actions.py for the "
        "actions instead of calling Video Indexer's insights API.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name to register the upload under, if uploading. Default: the video file's stem.",
    )
    parser.add_argument(
        "--only-composite",
        action="store_true",
        help="Only annotate composite ('derived') actions, e.g. 'person using "
        "phone' -- hides the raw labels/keywords/objects/ocr rows they were "
        "built from. Useful for isolating what the composite layer actually "
        "found.",
    )
    parser.add_argument(
        "--min-overlap-seconds",
        type=float,
        default=None,
        help="Drop composite action overlaps shorter than this many seconds "
        "(detector jitter). Ignored when --report is used (its actions are "
        "already materialized). Unfiltered by default.",
    )
    parser.add_argument(
        "--merge-gap-seconds",
        type=float,
        default=None,
        help="Collapse occurrences of the same action within this many "
        "seconds of each other (or overlapping) into one continuous "
        "occurrence. Ignored when --report is used.",
    )
    parser.add_argument(
        "--min-temporal-iou",
        type=float,
        default=None,
        help="Drop composite overlaps whose temporal Intersection-over-Union "
        "is below this (0.0-1.0). Ignored when --report is used.",
    )
    parser.add_argument(
        "--require-single-person",
        action="store_true",
        help="Only keep composite occurrences where observedPeople (Advanced "
        "preset only) tracked exactly one person in frame. Ignored when "
        "--report is used.",
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
        "--max-lines",
        type=int,
        default=6,
        help="Max simultaneous action lines drawn per frame before collapsing to '+N more'. Default: 6.",
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
    """Load actions from either report shape this project writes: a
    detect_all_actions.py *.all_actions.json (top-level "actions", each keyed
    by "name") or a main.py *.report.json (top-level "events", each keyed by
    "matched_term" instead) -- both are plausible files to pass to --report,
    since main.py's single-action report is a perfectly good source of
    actions to annotate too."""
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
                evidence=a.get("evidence"),
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
                evidence=e.get("evidence"),
            )
            for e in events
        ]

    raise VideoAnnotationError(
        f"{report_path} doesn't look like a detect_all_actions.py or main.py "
        "report (no top-level 'actions' or 'events' list)."
    )


def _filter_only_composite(
    actions: list[DetectedAction], only_composite: bool
) -> list[DetectedAction]:
    """When `only_composite` is set, keep only composite ('derived') actions
    -- e.g. 'person using phone' -- hiding the raw labels/keywords/objects/
    ocr rows they were built from. Returns `actions` unchanged otherwise."""
    if not only_composite:
        return actions
    return [a for a in actions if a.source == "derived"]


def _validate_args(args: argparse.Namespace) -> str | None:
    """Validate parsed args, returning a human-readable error message, or
    None if they're valid. Split out from main() so this (the least
    exciting but most bug-prone part of the CLI) is directly unit-testable
    without spinning up Settings/VideoIndexerClient."""
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
    if args.min_overlap_seconds is not None and args.min_overlap_seconds < 0:
        return "--min-overlap-seconds must be >= 0."
    if args.min_temporal_iou is not None and not (0.0 <= args.min_temporal_iou <= 1.0):
        return "--min-temporal-iou must be between 0.0 and 1.0."
    if args.merge_gap_seconds is not None and args.merge_gap_seconds < 0:
        return "--merge-gap-seconds must be >= 0."
    return None


def _build_client_if_needed(args: argparse.Namespace) -> VideoIndexerClient | None:
    """A client is needed whenever Azure has to do something: fetch/build
    the insights (unless --report covers that) or download the source file
    (unless a local --video covers that). Returns None only when --report
    and --video are both given -- the one combination that needs no Azure
    config or client at all."""
    if args.report and args.video:
        return None
    settings = Settings.from_env()
    return VideoIndexerClient(settings)


class _DownloadsVideo(Protocol):
    """The one VideoIndexerClient method this function actually calls --
    accepting this instead of the concrete `VideoIndexerClient` type lets
    tests pass a hand-rolled FakeVideoIndexerClient double without it
    needing to subclass or fully replicate VideoIndexerClient's real
    surface, while `main()`'s real `client: VideoIndexerClient | None` is
    still fully type-checked at every other call site."""

    def download_video(self, video_id: str, dest_path: Path) -> Path: ...


def _resolve_frame_source(
    client: _DownloadsVideo | None, args: argparse.Namespace
) -> Path:
    """The video file to read frames from: the local --video if given,
    otherwise --video-id's source file downloaded from Video Indexer."""
    if args.video:
        return args.video
    # No local --video means _build_client_if_needed always returned a real
    # client (it only returns None when --report and --video are both
    # given), so a download is always possible here.
    assert client is not None, "a client is required to download by --video-id alone"
    downloaded_path = args.out_dir / f"{args.video_id}.source.mp4"
    return client.download_video(args.video_id, downloaded_path)


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
        if args.report:
            logger.info(
                "Loading actions from %s (no Video Indexer insights call needed)",
                args.report,
            )
            actions = _load_actions_from_report(args.report)
        elif args.video_id:
            # _build_client_if_needed only returns None when --report and
            # --video are both given, which this branch (no --report) can't
            # be, so a real client is always available here.
            assert client is not None, "a client is required to reuse --video-id"
            logger.info("Reusing existing video id=%s", args.video_id)
            index = client.wait_for_processing(args.video_id)
            actions = analyze_all(
                index,
                min_overlap_seconds=args.min_overlap_seconds,
                merge_gap_seconds=args.merge_gap_seconds,
                min_temporal_iou=args.min_temporal_iou,
                require_single_person=args.require_single_person,
            ).actions
        else:
            assert client is not None, "a client is required to upload --video"
            video_id = client.upload_video(args.video, name=args.name)
            logger.info(
                "Video uploaded (id=%s). Save this ID to re-run without "
                "re-uploading via --video-id.",
                video_id,
            )
            index = client.wait_for_processing(video_id)
            actions = analyze_all(
                index,
                min_overlap_seconds=args.min_overlap_seconds,
                merge_gap_seconds=args.merge_gap_seconds,
                min_temporal_iou=args.min_temporal_iou,
                require_single_person=args.require_single_person,
            ).actions
    except (VideoIndexerError, VideoAnnotationError) as exc:
        logger.error("Could not obtain actions: %s", exc)
        return 1

    actions = _filter_only_composite(actions, args.only_composite)
    if args.only_composite and not actions:
        logger.warning(
            "--only-composite was set but no composite ('derived') "
            "actions were found -- output will have no overlays."
        )

    # Base name for default output filenames, and the frame source itself.
    base_name = args.video.stem if args.video else (args.name or args.video_id)

    try:
        frame_source = _resolve_frame_source(client, args)
    except VideoIndexerError as exc:
        logger.error("Could not download source video for frames: %s", exc)
        return 1

    suffix = "composite_actions.annotated" if args.only_composite else "annotated"
    out_path = args.out or (args.out_dir / f"{base_name}.{suffix}.mp4")

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
