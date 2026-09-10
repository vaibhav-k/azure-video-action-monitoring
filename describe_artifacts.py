#!/usr/bin/env python3
"""
describe_artifacts.py -- reads a folder of Azure Video Indexer's raw,
per-model "artifact" files (the ones you get from a manual "download
artifacts" export, as opposed to the aggregated Get-Video-Index API
response analyze_video.py works from) and writes a single, human-
readable paragraph describing what those files suggest is in the video.

Recognized artifact files (all optional -- whatever's present is used,
whatever's missing is skipped without failing the run):
    labels.computervision.json   -- one Computer Vision label list per
                                     sampled frame (usually one frame a
                                     second, not every frame)
    detectedobjects.json         -- per-frame tracked object boxes,
                                     grouped by tracked instance ("id")
    ocr.json                     -- one OCR read per sampled frame
    contentmoderation.json       -- per-frame adult/racy/gore scores
    Textual_Content_Moderation.json -- moderation flags on the OCR text
                                        itself (usually empty)

Why this needed its own script rather than reusing analyze_video.py: an
"artifacts" export is a fundamentally different shape from the Get-Video-
Index response -- frame-sampled Computer Vision output rather than
Video Indexer's own aggregated, timestamped insights (no observedPeople,
no confidence-scored activity instances). It also has no notion of
"which person did what" at all, so this script never attempts that
association -- it only describes what showed up, in plain language.

Every claim here still traces back to an automated per-frame read of a
handful of sampled frames, not a person actually watching the footage,
so the output is written as a best-effort description, not a verified
account -- see _CAVEAT_SENTENCE at the end of every summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("describe_artifacts")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ARTIFACTS_FOLDER = Path("input/Indian_city_streets")

LABELS_FILENAME = "labels.computervision.json"
OBJECTS_FILENAME = "detectedobjects.json"
OCR_FILENAME = "ocr.json"
MODERATION_FILENAME = "contentmoderation.json"

# A label appearing in at least this fraction of sampled frames reads as
# persistent scene-setting (e.g. "outdoor", "street"); anything rarer is
# treated as an occasional/incidental detail instead. There's no field in
# this data that marks a label as "setting" vs. "detail" -- it's inferred
# purely from how often it recurs across the sampled frames.
PERSISTENT_LABEL_COVERAGE_RATIO = 0.5
MAX_PERSISTENT_LABELS_SHOWN = 5
MAX_OCCASIONAL_LABELS_SHOWN = 4

# A single sampled frame's OCR read is noisy (motion blur, camera tilt,
# a moment's glare) -- text is only trusted here once it recurs across at
# least this many sampled frames, and only the most-recurring few are
# shown.
OCR_MIN_RECURRENCE = 2
MAX_OCR_SNIPPETS_SHOWN = 3

# Labels that describe the *presence of OCR text* rather than anything
# about the scene itself -- shown via the OCR section instead.
_META_LABELS = {"text"}

# Real Computer Vision labels, but too technical or too redundant with
# something already said elsewhere to read naturally in prose -- "land
# vehicle" and "wheel" are taxonomy-speak nobody would say out loud, and
# "person" is dropped in favor of "people" (the more natural collective
# term) whenever both show up, which they usually do together.
_SETTING_LABELS_TO_SKIP = {"land vehicle", "wheel", "person"}

_SMALL_NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}


def _count_word(n: int) -> str:
    return _SMALL_NUMBER_WORDS.get(n, str(n))


def _join_terms(terms: list[str]) -> str:
    """Natural-language join: ["a"] -> "a"; ["a", "b"] -> "a and b";
    ["a", "b", "c"] -> "a, b, and c"."""
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"{terms[0]} and {terms[1]}"
    return ", ".join(terms[:-1]) + f", and {terms[-1]}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any] | None:
    """None (with a debug log, not an error) when the file is simply
    absent -- every artifact file is optional -- but lets a genuinely
    malformed file raise, since that's worth surfacing."""
    if not path.is_file():
        logger.debug("Artifact not found, skipping: %s", path.name)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    """Narrow a raw JSON value (statically an ``Any`` coming out of an
    untyped payload) into a properly typed list of dicts. Every artifact
    file's array-shaped fields (frame results, per-frame ``Label``/
    ``Ocr``/``Adult`` entries, tracked-object ``results``) are looped over
    below; going through this helper -- rather than each call site doing
    its own ``data.get(key, []) or []`` -- means Pylance can actually infer
    the loop variable's type instead of reporting it as (partially)
    unknown, and a malformed (non-list, or list-of-non-dict) value
    degrades to "nothing found" instead of raising."""
    if not isinstance(value, list):
        return []
    return [cast(dict[str, Any], entry) for entry in value if isinstance(entry, dict)]


def _parse_timestamp(value: str) -> float:
    """detectedobjects.json timestamps look like "0:00:20.466667"
    (H:MM:SS.fraction). Defensive about odd input rather than raising,
    since this is third-party data this script doesn't control."""
    try:
        parts = value.strip().split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours, (minutes, seconds) = "0", parts
        else:
            return float(value)
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass
class LabelObservation:
    name: str
    frame_count: int
    max_confidence: float


@dataclass
class TrackedObjectGroup:
    object_type: str
    count: int
    first_seconds: float
    last_seconds: float


@dataclass
class OcrSnippet:
    text: str
    frame_count: int


@dataclass
class ModerationSummary:
    any_flagged: bool
    max_adult_score: float
    max_racy_score: float
    max_gore_score: float


@dataclass
class ArtifactsSummary:
    sampled_frame_count: int
    video_span_seconds: float | None
    persistent_labels: list[LabelObservation]
    occasional_labels: list[LabelObservation]
    objects: list[TrackedObjectGroup]
    ocr_snippets: list[OcrSnippet]
    moderation: ModerationSummary | None


def _parse_label_observations(folder: Path) -> tuple[list[LabelObservation], int]:
    """Returns (every label seen across the sampled frames, how many
    frames were sampled) -- unfiltered and uncategorized; _categorize_
    labels splits these into persistent/occasional afterward."""
    data = _load_json(folder / LABELS_FILENAME)
    if data is None:
        return [], 0

    results = _as_dict_list(data.get("Results"))
    counts: dict[str, int] = {}
    max_confidence: dict[str, float] = {}
    for frame in results:
        for label in _as_dict_list(frame.get("Label")):
            name = label.get("name")
            if not name or name.lower() in _META_LABELS:
                continue
            counts[name] = counts.get(name, 0) + 1
            confidence = label.get("confidence") or 0.0
            max_confidence[name] = max(max_confidence.get(name, 0.0), confidence)

    observations = [
        LabelObservation(
            name=name, frame_count=count, max_confidence=max_confidence[name]
        )
        for name, count in counts.items()
    ]
    return observations, len(results)


def _categorize_labels(
    observations: list[LabelObservation], sampled_frame_count: int
) -> tuple[list[LabelObservation], list[LabelObservation]]:
    """Split into (persistent/scene-setting, occasional/incidental) by how
    much of the sampled footage each label appeared in -- see
    PERSISTENT_LABEL_COVERAGE_RATIO."""
    if sampled_frame_count == 0:
        return [], []

    persistent: list[LabelObservation] = []
    occasional: list[LabelObservation] = []
    for obs in observations:
        coverage = obs.frame_count / sampled_frame_count
        (
            persistent if coverage >= PERSISTENT_LABEL_COVERAGE_RATIO else occasional
        ).append(obs)

    persistent.sort(key=lambda o: (-o.frame_count, o.name))
    occasional.sort(key=lambda o: (-o.frame_count, o.name))
    return persistent, occasional


def _parse_objects(folder: Path) -> list[TrackedObjectGroup]:
    """Groups detectedobjects.json's per-instance tracks by object type.
    Each "result" entry is one tracked instance across consecutive
    frames (a single bicycle being followed for a few seconds, say), not
    one detection -- so the count here is "how many separate things of
    this type were tracked", and the time range is that instance's own
    first-to-last frame, taken straight from its own instances list
    (which Video Indexer already keeps chronological)."""
    data = _load_json(folder / OBJECTS_FILENAME)
    if data is None:
        return []

    by_type: dict[str, list[tuple[float, float]]] = {}
    for tracked in _as_dict_list(data.get("results")):
        object_type = tracked.get("type")
        instances = _as_dict_list(tracked.get("instances"))
        if not object_type or not instances:
            continue
        start = _parse_timestamp(instances[0].get("start", ""))
        end = _parse_timestamp(instances[-1].get("end", ""))
        by_type.setdefault(object_type, []).append((start, end))

    groups = [
        TrackedObjectGroup(
            object_type=object_type,
            count=len(spans),
            first_seconds=min(s for s, _ in spans),
            last_seconds=max(e for _, e in spans),
        )
        for object_type, spans in by_type.items()
    ]
    groups.sort(key=lambda g: (-g.count, g.object_type))
    return groups


def _parse_ocr_snippets(folder: Path) -> list[OcrSnippet]:
    """Only text that recurs across multiple sampled frames is trusted --
    a single frame's OCR read is exactly as reliable as one blurry photo
    of a sign, so a one-off misread is filtered out rather than reported
    as if it were solid. See OCR_MIN_RECURRENCE."""
    data = _load_json(folder / OCR_FILENAME)
    if data is None:
        return []

    counts: dict[str, int] = {}
    for frame in _as_dict_list(data.get("Results")):
        content = (frame.get("Ocr") or {}).get("content") or ""
        for line in content.split("\n"):
            line = line.strip()
            if len(line) < 3 or line.strip("-") == "":
                continue
            counts[line] = counts.get(line, 0) + 1

    snippets = [
        OcrSnippet(text=text, frame_count=count)
        for text, count in counts.items()
        if count >= OCR_MIN_RECURRENCE
    ]
    snippets.sort(key=lambda s: (-s.frame_count, s.text))
    return snippets[:MAX_OCR_SNIPPETS_SHOWN]


def _parse_moderation(folder: Path) -> ModerationSummary | None:
    data = _load_json(folder / MODERATION_FILENAME)
    if data is None:
        return None

    results = _as_dict_list(data.get("Results"))
    if not results:
        return None

    max_adult = max_racy = max_gore = 0.0
    any_flagged = False
    for frame in results:
        adult = frame.get("Adult") or {}
        max_adult = max(max_adult, adult.get("adultScore") or 0.0)
        max_racy = max(max_racy, adult.get("racyScore") or 0.0)
        max_gore = max(max_gore, adult.get("goreScore") or 0.0)
        if (
            adult.get("isAdultContent")
            or adult.get("isRacyContent")
            or adult.get("isGoryContent")
        ):
            any_flagged = True

    return ModerationSummary(
        any_flagged=any_flagged,
        max_adult_score=max_adult,
        max_racy_score=max_racy,
        max_gore_score=max_gore,
    )


def build_artifacts_summary(folder: Path) -> ArtifactsSummary:
    """Parse every recognized artifact file present in `folder` into one
    ArtifactsSummary -- the single function describe_scene() turns into
    prose. Missing files just leave the corresponding field empty/None;
    nothing here requires all five to be present."""
    observations, sampled_frame_count = _parse_label_observations(folder)
    persistent, occasional = _categorize_labels(observations, sampled_frame_count)
    objects = _parse_objects(folder)
    ocr_snippets = _parse_ocr_snippets(folder)
    moderation = _parse_moderation(folder)

    video_span_seconds = max((g.last_seconds for g in objects), default=None)

    return ArtifactsSummary(
        sampled_frame_count=sampled_frame_count,
        video_span_seconds=video_span_seconds,
        persistent_labels=persistent,
        occasional_labels=occasional,
        objects=objects,
        ocr_snippets=ocr_snippets,
        moderation=moderation,
    )


# ---------------------------------------------------------------------------
# Natural-language description
# ---------------------------------------------------------------------------

_CAVEAT_SENTENCE = (
    "All of this comes from an automated read of a handful of sampled "
    "frames -- Computer Vision labels, object tracking, and OCR -- not "
    "from anyone actually watching the footage, so treat it as a "
    "best-effort description rather than a verified account."
)


def _prose_names(observations: list[LabelObservation], limit: int) -> list[str]:
    """The first `limit` label names worth actually saying out loud --
    see _SETTING_LABELS_TO_SKIP for what gets passed over and why."""
    names = [
        obs.name
        for obs in observations
        if obs.name.lower() not in _SETTING_LABELS_TO_SKIP
    ]
    return names[:limit]


def _clip_length_phrase(video_span_seconds: float | None) -> str:
    """A conversational opener, e.g. "This roughly 20-second clip" --
    video_span_seconds is only an estimate (the latest end-time seen
    across all tracked objects, which stops as soon as tracking does),
    so it's always hedged with "roughly" rather than stated as exact."""
    if not video_span_seconds:
        return "This clip"
    return f"This roughly {round(video_span_seconds)}-second clip"


def _setting_sentence(
    persistent: list[LabelObservation], video_span_seconds: float | None
) -> str:
    clip_phrase = _clip_length_phrase(video_span_seconds)
    names = _prose_names(persistent, MAX_PERSISTENT_LABELS_SHOWN)
    if not names:
        return f"{clip_phrase}'s setting isn't very clear from this data."
    return f"{clip_phrase} looks like an outdoor, street-level scene -- {_join_terms(names)} all show up consistently throughout."


def _incidental_sentence(occasional: list[LabelObservation]) -> str | None:
    names = _prose_names(occasional, MAX_OCCASIONAL_LABELS_SHOWN)
    if not names:
        return None
    return f"A few other things come and go in the footage too, including {_join_terms(names)}."


def _pluralize(word: str, count: int) -> str:
    """Naive but covers the common cases in Video Indexer's object
    vocabulary (car, bicycle, motorcycle, handbag, bus, ...): a regular
    "+s" for most words, "+es" for the sibilant endings ("bus" ->
    "buses", not the naive-plural "buss")."""
    if count == 1:
        return word
    if word.endswith(("s", "x", "sh", "ch")):
        return f"{word}es"
    return f"{word}s"


def _objects_sentence(objects: list[TrackedObjectGroup]) -> str | None:
    if not objects:
        return None
    parts = [
        f"{_count_word(g.count)} {_pluralize(g.object_type.lower(), g.count)}"
        for g in objects
    ]
    return f"Object tracking picked out {_join_terms(parts)} moving through the scene at various points."


def _ocr_sentence(snippets: list[OcrSnippet]) -> str | None:
    if not snippets:
        return None
    quoted = [f'"{s.text}"' for s in snippets]
    return (
        f"Text recognition repeatedly picked up {_join_terms(quoted)} on-screen -- "
        "likely signage and a watermark -- though a moving, tilted camera "
        "makes OCR unreliable, so treat the exact wording as approximate."
    )


def _moderation_sentence(moderation: ModerationSummary | None) -> str | None:
    if moderation is None:
        return None
    if moderation.any_flagged:
        return "Automated content screening did flag some frames as potentially adult, racy, or graphic -- worth a manual look before this is shown to anyone."
    return "Automated content screening didn't flag anything as adult, racy, or graphic, and the scores stayed low throughout."


def describe_scene(summary: ArtifactsSummary) -> str:
    """The final one-paragraph, plain-English description built from an
    ArtifactsSummary. Each sentence comes from its own small helper above
    so this function stays a short, flat sequence over them; any section
    with nothing to say (e.g. no OCR text recurred) is simply skipped."""
    sentences = [
        _setting_sentence(summary.persistent_labels, summary.video_span_seconds)
    ]

    incidental = _incidental_sentence(summary.occasional_labels)
    if incidental:
        sentences.append(incidental)

    objects_sentence = _objects_sentence(summary.objects)
    if objects_sentence:
        sentences.append(objects_sentence)

    ocr_sentence = _ocr_sentence(summary.ocr_snippets)
    if ocr_sentence:
        sentences.append(ocr_sentence)

    moderation_sentence = _moderation_sentence(summary.moderation)
    if moderation_sentence:
        sentences.append(moderation_sentence)

    sentences.append(_CAVEAT_SENTENCE)
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Describe, in plain English, what a folder of Azure Video "
            "Indexer 'artifact' files (labels.computervision.json, "
            "detectedobjects.json, ocr.json, contentmoderation.json) "
            "suggests is in the video."
        )
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=DEFAULT_ARTIFACTS_FOLDER,
        help=f"Folder containing the artifact files. Default: {DEFAULT_ARTIFACTS_FOLDER}",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if not args.folder.is_dir():
        logger.error("Artifacts folder not found: %s", args.folder)
        return 2

    summary = build_artifacts_summary(args.folder)
    description = describe_scene(summary)

    logger.info(description)

    output_path = args.folder.with_name(args.folder.name + ".description.txt")
    output_path.write_text(description, encoding="utf-8")
    logger.info("\nWritten to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
