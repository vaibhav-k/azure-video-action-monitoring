#!/usr/bin/env python3
"""
CLI: write a single, hedged, plain-language paragraph describing what a
video's Azure AI Video Indexer metadata *suggests* might be happening in
it -- built entirely from already-computed Video Indexer insights (the
same label/keyword/object timeline detect_all_actions.py reports), with
NO frame inspection, NO video file needed, and NO LLM/vision-model call of
any kind.

This is a fundamentally different, much weaker guarantee than actually
watching the video: every sentence this produces can only repeat what
Video Indexer's own classifiers flagged (filtered by --min-confidence),
phrased as a hedge ("was flagged", "the metadata suggests", "may be
present"), and the paragraph always ends by saying so explicitly. See
src/scene_summary.py's module docstring for the heuristics behind how
that paragraph gets built.

Two ways to supply the metadata:
    --metadata <path>   a raw Video Indexer insights export -- either the
                         live API response, or a *.raw_insights.json file
                         saved by explore_insights.py/detect_all_actions.py
                         --save-raw-insights. No Azure account needed.
    --video-id <id>     reuse an already-indexed Video Indexer video's own
                         insights: a cached <out-dir>/<id>.raw_insights.json
                         is used if one already exists there, otherwise
                         insights are fetched fresh from Video Indexer
                         (needs an Azure AI Video Indexer account).

Example:
    python analyze_video.py --metadata output/c34jzsrlcg.raw_insights.json
    python analyze_video.py --video-id c34jzsrlcg --min-confidence 0.7

Only the final line block printed to stdout -- "Scene understanding:"
followed by a blank line and the paragraph -- is meant to be consumed
programmatically. Everything else (progress logging) goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.action_analyzer import analyze_all
from src.config import ConfigError, Settings
from src.constants import DEFAULT_MIN_CONFIDENCE
from src.scene_summary import summarize_insights
from src.video_indexer_client import VideoIndexerClient, VideoIndexerError

logger = logging.getLogger("azure_action_monitoring.analyze_video")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a hedged, plain-language paragraph describing "
        "what a video's Video Indexer metadata suggests, with no frame "
        "inspection and no LLM call."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="A raw Video Indexer insights export (the live API response, "
        "or a *.raw_insights.json file saved by explore_insights.py/"
        "detect_all_actions.py --save-raw-insights). No Azure account "
        "needed when this is given. Required unless --video-id is used.",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="Reuse an already-indexed Video Indexer video's own insights: "
        "a cached <out-dir>/<video-id>.raw_insights.json is used if one "
        "already exists, otherwise insights are fetched fresh from Video "
        "Indexer. Required unless --metadata is used.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Drop occurrences whose confidence score is below this "
        "(0.0-1.0). Occurrences with no confidence value are always kept. "
        f"Default: {DEFAULT_MIN_CONFIDENCE}.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("output"),
        type=Path,
        help="Directory --video-id looks in for a cached "
        "<video-id>.raw_insights.json, and where a freshly-fetched one is "
        "saved for next time. Default: ./output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to also save the paragraph as a plain text "
        "file. Not required -- it's always printed to stdout regardless.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser


def _validate_args(args: argparse.Namespace) -> str | None:
    """Validate parsed args, returning a human-readable error message, or
    None if they're valid."""
    if not args.metadata and not args.video_id:
        return (
            "Either --metadata (a raw Video Indexer insights JSON file) or "
            "--video-id (to reuse an already-indexed video's insights) is "
            "required."
        )
    if args.metadata is not None and not args.metadata.is_file():
        return f"Metadata file not found: {args.metadata}"
    if not (0.0 <= args.min_confidence <= 1.0):
        return "--min-confidence must be between 0.0 and 1.0."
    return None


def _cached_raw_insights_path(args: argparse.Namespace) -> Path:
    """Where explore_insights.py/detect_all_actions.py --save-raw-insights
    would have written this video id's raw insights, when run without an
    explicit --name (the common case for a bare --video-id re-run):
    <out-dir>/<video-id>.raw_insights.json."""
    return args.out_dir / f"{args.video_id}.raw_insights.json"


def _resolve_index_payload(args: argparse.Namespace) -> dict[str, Any]:
    """The raw Video Indexer insights payload to analyze: an explicit
    --metadata file always wins. Otherwise, for --video-id, a cached
    <out-dir>/<video-id>.raw_insights.json is reused if present (avoiding a
    redundant Video Indexer call); if not, insights are fetched fresh and
    saved to that same path for next time."""
    if args.metadata is not None:
        raw_text = args.metadata.read_text(encoding="utf-8")
        return json.loads(raw_text)

    cached_path = _cached_raw_insights_path(args)
    if cached_path.is_file():
        logger.info("Using cached raw insights from %s", cached_path)
        return json.loads(cached_path.read_text(encoding="utf-8"))

    settings = Settings.from_env()
    client = VideoIndexerClient(settings)
    logger.info("Fetching insights for video id=%s ...", args.video_id)
    index = client.wait_for_processing(args.video_id)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cached_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    logger.info("Saved raw insights to %s for next time", cached_path)
    return index


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    error = _validate_args(args)
    if error:
        logger.error(error)
        return 2

    try:
        index_payload = _resolve_index_payload(args)
    except OSError as exc:
        logger.error("Could not read metadata file: %s", exc)
        return 2
    except json.JSONDecodeError as exc:
        logger.error("Metadata file is not valid JSON: %s", exc)
        return 2
    except ConfigError as exc:
        logger.error(str(exc))
        return 2
    except VideoIndexerError as exc:
        logger.error("Could not fetch insights for --video-id: %s", exc)
        return 1

    report = analyze_all(index_payload, min_confidence=args.min_confidence)
    paragraph = summarize_insights(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(paragraph + "\n", encoding="utf-8")

    print(f"Scene understanding:\n\n{paragraph}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
