#!/usr/bin/env python3
"""
CLI: showcase every Azure AI Video Indexer insight category this project's
action-focused scripts (main.py, detect_all_actions.py, annotate_video.py)
don't already cover -- faces, transcript/speakers/language, topics, named
entities (brands/locations/people), sentiments, emotions, audio effects,
shots, and content moderation.

Most of these are Advanced-preset-only or richer under Advanced (see
--indexing-preset below), so this script defaults to Advanced -- unlike the
other three scripts here, which default to Default. See README's "What to
test it with" section for what kind of video content actually exercises
each of these (a silent security-camera clip, for instance, will come back
with every one of these buckets empty except maybe shots).

Example:
    python explore_insights.py --video ./interview_clip.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from src.capabilities_report import (
    to_console_text_capabilities,
    write_html_capabilities,
    write_json_capabilities,
)
from src.config import ConfigError, Settings
from src.insights_explorer import CapabilitiesReport, FaceInsight, analyze_capabilities
from src.video_indexer_client import VideoIndexerClient, VideoIndexerError

logger = logging.getLogger("azure_action_monitoring.explore_insights")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Showcase every Azure AI Video Indexer insight category "
        "besides actions/objects (faces, transcript, topics, named "
        "entities, sentiments, emotions, audio effects, shots, content "
        "moderation)."
    )
    parser.add_argument(
        "--video",
        required=False,
        type=Path,
        help="Path to a local video file. Not required with --check-auth or --video-id.",
    )
    parser.add_argument(
        "--check-auth",
        action="store_true",
        help="Just verify Azure auth + account access (no upload) and exit.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name to register the upload under in Video Indexer. Default: the video file's stem.",
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
        "--indexing-preset",
        default="Advanced",
        choices=["Default", "Advanced"],
        help="Video Indexer indexing preset to use when uploading a new "
        "video (ignored when reusing --video-id, since the video keeps "
        "whatever preset it was originally indexed with). Defaults to "
        "Advanced here (unlike this project's other scripts) because "
        "almost everything this script reports -- topics, named entities, "
        "sentiments, emotions, audio effects, observedPeople -- is "
        "Advanced-only or richer under Advanced; faces and transcript are "
        "the exceptions and work under Default too.",
    )
    parser.add_argument(
        "--save-raw-insights",
        action="store_true",
        help="Also save the full raw Video Indexer insights JSON.",
    )
    parser.add_argument(
        "--save-face-thumbnails",
        action="store_true",
        help="Download and save each detected face's thumbnail image (one "
        "extra Video Indexer API call per face) to "
        "<out-dir>/<name>.faces/. Useful for visually checking whether two "
        "'Unknown #N' faces are actually the same person -- Video Indexer "
        "has no facial-recognition re-identification for unregistered "
        "faces, so its short-term tracking can split one real person into "
        "multiple Unknown entries if tracking drops partway through.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser


def _safe_face_filename(name: str) -> str:
    """Turn a face name (e.g. 'Unknown #1', 'Jane Doe') into a filesystem-
    safe filename stem -- strips anything that isn't alphanumeric/space/
    hyphen/underscore, then replaces spaces with underscores, so this can
    never escape the target directory or collide with shell-unsafe
    characters regardless of what Video Indexer names a recognized face."""
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    return re.sub(r"\s+", "_", cleaned) or "face"


def _save_face_thumbnails(
    client: VideoIndexerClient,
    video_id: str,
    faces: list[FaceInsight],
    out_dir: Path,
) -> int:
    """Download every face's thumbnail image to `out_dir`, named after the
    face (recognized faces get their real name; unrecognized ones keep
    their 'Unknown #N' label) plus a numeric suffix to keep filenames
    unique when two faces would otherwise collide. Returns how many were
    saved; a face with no thumbnail_id (shouldn't normally happen, but
    nothing in the schema guarantees it) is skipped rather than raising."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for index, face in enumerate(faces):
        if not face.thumbnail_id:
            continue
        try:
            image_bytes = client.get_video_thumbnail(video_id, face.thumbnail_id)
        except VideoIndexerError as exc:
            logger.warning(
                "Could not download thumbnail for %s (%s): %s",
                face.name,
                face.thumbnail_id,
                exc,
            )
            continue
        filename = f"{_safe_face_filename(face.name)}_{index}.jpg"
        (out_dir / filename).write_bytes(image_bytes)
        saved += 1
    return saved


def _validate_args(args: argparse.Namespace) -> str | None:
    """Validate parsed args, returning a human-readable error message, or
    None if they're valid."""
    if not args.check_auth and not args.video and not args.video_id:
        return (
            "Either --video (to upload) or --video-id (to reuse an already-"
            "indexed video) is required, unless using --check-auth."
        )
    if args.video and not args.video.is_file():
        return f"Video file not found: {args.video}"
    return None


def _run_check_auth(client: VideoIndexerClient, settings: Settings) -> int:
    """--check-auth: verify Azure auth + account access with no upload,
    reporting success/failure via the process's exit code."""
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


def _resolve_video_id_and_index(
    client: VideoIndexerClient, args: argparse.Namespace
) -> tuple[str, dict[str, Any]]:
    """Either reuse an already-indexed --video-id, or upload --video fresh
    -- either way, wait for and return its finished insights index. Raises
    VideoIndexerError on failure, same as the calls it wraps."""
    if args.video_id:
        video_id = args.video_id
        logger.info("Reusing existing video id=%s", video_id)
        return video_id, client.wait_for_processing(video_id)

    video_id = client.upload_video(
        args.video, name=args.name, indexing_preset=args.indexing_preset
    )
    logger.info(
        "Video uploaded (id=%s, preset=%s). Save this ID to re-run "
        "analysis without re-uploading via --video-id.",
        video_id,
        args.indexing_preset,
    )
    return video_id, client.wait_for_processing(video_id)


def _maybe_save_raw_insights(
    args: argparse.Namespace, stem: str, index: dict[str, Any]
) -> None:
    """Write the full raw Video Indexer insights payload to
    <out-dir>/<stem>.raw_insights.json when --save-raw-insights was passed;
    a no-op otherwise."""
    if not args.save_raw_insights:
        return
    raw_path = args.out_dir / f"{stem}.raw_insights.json"
    raw_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    logger.info("Saved raw insights to %s", raw_path)


def _maybe_save_face_thumbnails(
    args: argparse.Namespace,
    client: VideoIndexerClient,
    video_id: str,
    report: CapabilitiesReport,
    stem: str,
) -> None:
    """Download every detected face's thumbnail to <out-dir>/<stem>.faces/
    when --save-face-thumbnails was passed; a no-op otherwise (logging a
    heads-up if the flag was set but no faces were actually detected)."""
    if not args.save_face_thumbnails:
        return
    if not report.faces:
        logger.info("--save-face-thumbnails was set but no faces were detected.")
        return
    faces_dir = args.out_dir / f"{stem}.faces"
    saved = _save_face_thumbnails(client, video_id, report.faces, faces_dir)
    logger.info(
        "Saved %d/%d face thumbnail(s) (%d recognized) to %s",
        saved,
        len(report.faces),
        len(report.recognized_faces),
        faces_dir,
    )


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
        return _run_check_auth(client, settings)

    # Base name used for both the Video Indexer upload label and the output
    # filenames. Prefer the local video's filename; fall back to --name or
    # the video ID when running against an already-indexed video with no
    # local file on hand (e.g. `--video-id` alone).
    display_name = args.video.name if args.video else (args.name or args.video_id)
    stem = args.video.stem if args.video else (args.name or args.video_id)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_id, index = _resolve_video_id_and_index(client, args)
    except VideoIndexerError as exc:
        logger.error("Video Indexer processing failed: %s", exc)
        return 1

    _maybe_save_raw_insights(args, stem, index)

    report = analyze_capabilities(index)

    print()
    print(to_console_text_capabilities(report, display_name))
    print()

    json_path = args.out_dir / f"{stem}.capabilities.json"
    html_path = args.out_dir / f"{stem}.capabilities.html"
    write_json_capabilities(report, display_name, json_path)
    write_html_capabilities(report, display_name, html_path)
    logger.info("Wrote %s and %s", json_path, html_path)

    _maybe_save_face_thumbnails(args, client, video_id, report, stem)

    return 0


if __name__ == "__main__":
    sys.exit(main())
