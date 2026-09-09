#!/usr/bin/env python3
"""
CLI: detect ALL actions in a video using Azure AI Video Indexer.

Unlike main.py (which searches for one action of interest, e.g. "jumping"),
this script uploads/indexes the video and reports *every* timestamped
label/keyword Video Indexer detected -- a full "what happened, and when"
timeline, grouped by distinct action with per-action counts and durations.

Example:
    python detect_all_actions.py --video ./people_jumping.mp4

On first run against a given video, this uploads it to your Azure AI Video
Indexer account and waits for processing (a few minutes, depending on
length). To skip re-uploading on subsequent runs against the same video,
pass --video-id with the ID printed on the first run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.action_analyzer import analyze_all
from src.config import ConfigError, Settings
from src.report import to_console_text_all, write_html_all, write_json_all
from src.video_indexer_client import VideoIndexerClient, VideoIndexerError

logger = logging.getLogger("azure_action_monitoring.detect_all_actions")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and report every action Azure AI Video Indexer finds in a video."
    )
    parser.add_argument(
        "--video",
        required=False,
        type=Path,
        help="Path to a local video file. Not required with --check-auth.",
    )
    parser.add_argument(
        "--check-auth",
        action="store_true",
        help="Just verify Azure auth + account access (no upload) and exit. Use this to debug config before uploading a large file.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name to register the upload under in Video Indexer. Default: the video file's stem. Override this if a prior upload of the same filename hit VIDEO_ALREADY_FAILED.",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="Skip upload and reuse an already-indexed video ID.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("output"),
        type=Path,
        help="Directory to write report files into. Default: ./output",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Drop occurrences whose confidence score is below this (0.0-1.0). "
        "Occurrences with no confidence value are always kept.",
    )
    parser.add_argument(
        "--indexing-preset",
        default="Default",
        choices=["Default", "Advanced"],
        help="Video Indexer indexing preset to use when uploading a new video "
        "(ignored when reusing --video-id). 'Advanced' additionally returns "
        "observedPeople, which --require-single-person needs -- without it, "
        "--require-single-person silently keeps every composite occurrence "
        "instead of filtering. Default: Default.",
    )
    parser.add_argument(
        "--min-overlap-seconds",
        type=float,
        default=None,
        help="Drop composite ('derived') action overlaps shorter than this "
        "many seconds (a 1-2 frame overlap is usually detector jitter, not a "
        "real co-occurrence). Unfiltered by default.",
    )
    parser.add_argument(
        "--merge-gap-seconds",
        type=float,
        default=None,
        help="Collapse occurrences of the same action within this many "
        "seconds of each other (or overlapping) into one continuous "
        "occurrence, instead of several fragmented rows.",
    )
    parser.add_argument(
        "--min-temporal-iou",
        type=float,
        default=None,
        help="Drop composite overlaps whose temporal Intersection-over-Union "
        "is below this (0.0-1.0) -- a stricter proxy than --min-overlap-"
        "seconds alone for 'these two detections describe the same event'.",
    )
    parser.add_argument(
        "--require-single-person",
        action="store_true",
        help="Only keep composite occurrences where observedPeople (Advanced "
        "preset only -- see --indexing-preset) tracked exactly one person in "
        "frame during the overlap. Has no effect if that data isn't available.",
    )
    parser.add_argument(
        "--save-raw-insights",
        action="store_true",
        help="Also save the full raw Video Indexer insights JSON.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser


def _validate_args(args: argparse.Namespace) -> str | None:
    """Validate parsed args, returning a human-readable error message, or
    None if they're valid. Split out from main() so this (the least
    exciting but most bug-prone part of the CLI) is directly unit-testable
    without spinning up Settings/VideoIndexerClient."""
    if not args.check_auth and not args.video and not args.video_id:
        return (
            "Either --video (to upload) or --video-id (to reuse an already-"
            "indexed video) is required, unless using --check-auth."
        )
    if args.video and not args.video.is_file():
        return f"Video file not found: {args.video}"
    if args.min_confidence is not None and not (0.0 <= args.min_confidence <= 1.0):
        return "--min-confidence must be between 0.0 and 1.0."
    if args.min_overlap_seconds is not None and args.min_overlap_seconds < 0:
        return "--min-overlap-seconds must be >= 0."
    if args.min_temporal_iou is not None and not (0.0 <= args.min_temporal_iou <= 1.0):
        return "--min-temporal-iou must be between 0.0 and 1.0."
    if args.merge_gap_seconds is not None and args.merge_gap_seconds < 0:
        return "--merge-gap-seconds must be >= 0."
    return None


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
        settings = Settings.from_env()
    except ConfigError as exc:
        logger.error(str(exc))
        return 2

    client = VideoIndexerClient(settings)

    if args.check_auth:
        logger.info(
            "Checking auth against subscription=%s resource_group=%s account_name=%s location=%s ...",
            settings.subscription_id,
            settings.resource_group,
            settings.account_name,
            settings.location,
        )
        try:
            client.get_access_token()
        except VideoIndexerError as exc:
            logger.error(str(exc))
            return 1
        logger.info(
            "Success: obtained a Video Indexer access token. Config and permissions look good."
        )
        return 0

    # Base name used for both the Video Indexer upload label and the output
    # filenames. Prefer the local video's filename; fall back to --name or
    # the video ID when running against an already-indexed video with no
    # local file on hand (e.g. `--video-id` alone).
    display_name = args.video.name if args.video else (args.name or args.video_id)
    stem = args.video.stem if args.video else (args.name or args.video_id)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.video_id:
            logger.info("Reusing existing video id=%s", args.video_id)
            index = client.wait_for_processing(args.video_id)
        else:
            video_id = client.upload_video(
                args.video, name=args.name, indexing_preset=args.indexing_preset
            )
            logger.info(
                "Video uploaded (id=%s, preset=%s). Save this ID to re-run "
                "analysis without re-uploading via --video-id.",
                video_id,
                args.indexing_preset,
            )
            index = client.wait_for_processing(video_id)
    except VideoIndexerError as exc:
        logger.error("Video Indexer processing failed: %s", exc)
        return 1

    if args.save_raw_insights:
        raw_path = args.out_dir / f"{stem}.raw_insights.json"
        raw_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        logger.info("Saved raw insights to %s", raw_path)

    report = analyze_all(
        index,
        min_confidence=args.min_confidence,
        min_overlap_seconds=args.min_overlap_seconds,
        merge_gap_seconds=args.merge_gap_seconds,
        min_temporal_iou=args.min_temporal_iou,
        require_single_person=args.require_single_person,
    )

    print()
    print(to_console_text_all(report, display_name))
    print()

    json_path = args.out_dir / f"{stem}.all_actions.json"
    html_path = args.out_dir / f"{stem}.all_actions.html"
    write_json_all(report, display_name, json_path)
    write_html_all(report, display_name, html_path)
    logger.info("Wrote %s and %s", json_path, html_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
