#!/usr/bin/env python3
"""
CLI entry point for the Azure AI Video Indexer action-monitoring prototype.

Example:
    python main.py --video ./people_jumping.mp4 --action jumping

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

from src.action_analyzer import analyze
from src.config import ConfigError, Settings
from src.report import to_console_text, write_html, write_json
from src.video_indexer_client import VideoIndexerClient, VideoIndexerError

logger = logging.getLogger("azure_action_monitoring")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and report an action of interest in a video using Azure AI Video Indexer."
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
        "--action",
        default="jumping",
        help="Action of interest to look for (e.g. 'jumping'). Default: jumping.",
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
        "--synonym",
        action="append",
        default=[],
        help="Extra term to also match (repeatable).",
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


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

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

    if not args.video and not args.video_id:
        logger.error(
            "Either --video (to upload) or --video-id (to reuse an already-"
            "indexed video) is required, unless using --check-auth."
        )
        return 2

    if args.video and not args.video.is_file():
        logger.error("Video file not found: %s", args.video)
        return 2

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
            video_id = client.upload_video(args.video, name=args.name)
            logger.info(
                "Video uploaded (id=%s). Save this ID to re-run analysis "
                "without re-uploading via --video-id.",
                video_id,
            )
            index = client.wait_for_processing(video_id)
    except VideoIndexerError as exc:
        logger.error("Video Indexer processing failed: %s", exc)
        return 1

    if args.save_raw_insights:
        raw_path = args.out_dir / f"{stem}.raw_insights.json"
        raw_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        logger.info("Saved raw insights to %s", raw_path)

    report = analyze(index, action=args.action, extra_synonyms=args.synonym)

    print()
    print(to_console_text(report, display_name))
    print()

    json_path = args.out_dir / f"{stem}.report.json"
    html_path = args.out_dir / f"{stem}.report.html"
    write_json(report, display_name, json_path)
    write_html(report, display_name, html_path)
    logger.info("Wrote %s and %s", json_path, html_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
