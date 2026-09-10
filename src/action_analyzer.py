"""
Turns a raw Azure AI Video Indexer insights payload into a focused list of
"action events" for a specific action of interest (e.g. "jumping"), plus a
summary suitable for an oversight report.

Important, honest caveat (see README "Limitations"): Video Indexer's default
model tags general visual concepts and objects (its `labels` and `keywords`
insights) rather than running a dedicated human-action classifier. A label
like "jumping" will only appear if the underlying vision model recognized
that concept in the frame. For guaranteed, high-recall detection of a
specific motion like a jump, the recommended production path is a
custom-trained model (Custom Vision / Azure ML with a pose or action
recognition model) -- see README for that upgrade path. This module is
written so that swapping in a different insights source later only requires
changing where `insights` comes from, not the extraction logic below.

This also reads the `detectedObjects` insight bucket (source "objects"),
Video Indexer's object detector -- a fixed ~80-class vocabulary (vehicles,
furniture, food, sports equipment, etc., the same class list Microsoft's
generic vision object detector uses everywhere). It does NOT include
domain-specific objects like money/cash/currency, so no combination of
`labels`, `keywords`, or `detectedObjects` will ever surface a concept
that isn't in one of those vocabularies to begin with -- see README
"Limitations" for what to do when the action/object you need genuinely
isn't one Video Indexer's default models know about.

It also reads the `ocr` bucket (source "ocr") -- printed/on-screen text
Video Indexer recognized, regardless of what object it's printed on. This
is how something like currency becomes detectable at all despite not
existing in any vision model's object vocabulary: a banknote's printed
text ("100", "WE TRUST", a serial number) gets read by OCR even though
detectedObjects has no "money" class. `DEFAULT_SYNONYMS["cash"]` matches
against phrases known to appear on US currency for this reason. Treat this
as a manual-review aid more than a reliable auto-detector, though: OCR on
video frames is noisy (motion blur, rotation, partial frames -- expect
fragmented/duplicate reads of the same text) and only catches what's both
in frame and legible, not "cash is present" in general (e.g. currency
sitting face-down, or a foreign banknote with different printed text,
won't match).

Finally, `_derive_composite_actions` synthesizes named "composite" actions
(`COMPOSITE_ACTIONS` in constants.py, source "derived") from temporal
overlaps between two independently-detected concepts -- e.g. "person using
phone" isn't a label/object Video Indexer has, but a person-like label and
a "cell phone" object both present at once is a reasonable, inspectable
proxy for it. Same honest-caveat pattern as everywhere else in this
module: an overlap is circumstantial evidence, not a verified claim that
the person is actually using the phone (they could be nearby without
touching it) -- treat it the same way as the rest of this module's output,
as a first pass to review rather than a guaranteed-accurate label. Each
derived `DetectedAction` records its `evidence` (the two contributing item
names/sources) so a reviewer can see why it was flagged without re-running
raw insights.

Two more honest limitations worth stating plainly, since they bound what
this module can ever claim: (1) `min_overlap_seconds` (opt-in) drops
overlaps shorter than a threshold, since a 1-2 frame overlap is usually
detector jitter, not a real co-occurrence -- but this is a blunt filter,
not a confidence measure. (2) None of `labels`/`detectedObjects`/
`observedPeople` include *spatial* (bounding-box) coordinates in this
project's Video Indexer responses (confirmed against Microsoft's own
insight schemas as well as this project's own raw insights captures) --
so nothing here can distinguish *which* person is which (e.g. "the
cashier" vs. a customer walking past) when more than one is in frame. A
derived "person X" action means *some* detected person overlapped the
other evidence, not a specific, identified one.

`ocr` is the one exception with *any* spatial field at all: each OCR item
carries a single static `left`/`top`/`width`/`height` box (not tracked
per-instance -- if the same text is read again later at a different
on-screen position, the box isn't updated). That's nowhere near enough to
gate person-vs-object proximity (the person side still has no box to
compare against), but it is enough to catch one concrete false-positive
source: a burned-in overlay -- a camera timestamp, a watermark -- sitting
in a fixed corner/edge region, small relative to the frame.
`_looks_like_ocr_overlay` applies that heuristic (thresholds in
`constants.py`) to exclude such OCR items from `person handling cash`
specifically, using the video's width/height when the payload has them
(`_video_dimensions`); it never removes or hides them from the raw
`ocr`-source rows elsewhere in a report. This is still just a pixel
heuristic on one item's static box, not a guarantee.

Two more opt-in filters simulate a *little* of what real spatial gating
would give, without pretending to have coordinates that don't exist --
both were checked exhaustively (every instance-level field across this
project's own raw captures, not just the top-level items) and confirmed
absent before building either of these:

1. `min_temporal_iou` is a temporal analog of spatial IoU (Intersection-
   over-Union, the standard measure of how well two bounding boxes align)
   applied to *time* instead of space: `overlap_duration / union_duration`
   for the two contributing items. A person label spanning the whole video
   and an object appearing for one brief instant can still pass
   `min_overlap_seconds`, but their durations barely coincide -- a low
   temporal IoU. Requiring a higher IoU favors composites where both
   detections' durations line up closely, which is a better (still
   circumstantial) proxy for "these describe the same real event" than
   "any overlap of at least N seconds." This is explicitly a *time*-domain
   proxy, not spatial evidence.
2. `require_single_person` uses `observedPeople` (Advanced-preset only) to
   count how many distinct people were tracked as present during a
   composite's overlap window (`_people_in_frame_count`). This never says
   *which* person -- there's still no position data -- but when exactly
   one person was present, there's no "which person" ambiguity for that
   specific occurrence either, so it's meaningful to keep. Every derived
   action's `evidence` records this count (e.g. "1 person in frame") when
   `observedPeople` data is available at all; `require_single_person`
   additionally drops any occurrence where the count is 0, 2, or more --
   i.e., keeps only occurrences that happen not to be ambiguous, rather
   than resolving the ambiguous ones.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import product
from typing import Any, cast

from src.constants import (
    COMPOSITE_ACTIONS,
    DEFAULT_OCR_OVERLAY_EDGE_MARGIN_FRACTION,
    DEFAULT_OCR_OVERLAY_MAX_AREA_FRACTION,
    DEFAULT_SYNONYMS,
    CompositeAction,
    CompositeSide,
)


@dataclass
class ActionEvent:
    """A single timestamped occurrence of the action of interest."""

    source: str  # which insight bucket this came from, e.g. "labels" or "keywords"
    matched_term: str  # the label/keyword text that matched
    start_seconds: float
    end_seconds: float
    confidence: float | None
    evidence: str | None = None  # for source="derived": the two items that overlapped

    @property
    def duration_seconds(self) -> float:
        """How long this occurrence lasted, in seconds (never negative)."""
        return max(0.0, self.end_seconds - self.start_seconds)


@dataclass
class ActionReport:
    """Aggregated result of scanning a video's insights for one action of interest."""

    action: str
    video_duration_seconds: float | None
    events: list[ActionEvent] = field(default_factory=list[ActionEvent])
    distinct_people_tracked: int = 0
    matched_terms_found: set[str] = field(default_factory=set[str])

    @property
    def occurrence_count(self) -> int:
        """Number of distinct timestamped occurrences found."""
        return len(self.events)

    @property
    def total_action_seconds(self) -> float:
        """Sum of the duration of every occurrence, in seconds."""
        return sum(e.duration_seconds for e in self.events)

    @property
    def first_occurrence_seconds(self) -> float | None:
        """Start time of the earliest occurrence, or None if none were found."""
        return min((e.start_seconds for e in self.events), default=None)

    @property
    def last_occurrence_seconds(self) -> float | None:
        """End time of the latest occurrence, or None if none were found."""
        return max((e.end_seconds for e in self.events), default=None)


def _timestamp_to_seconds(value: Any) -> float:
    """Video Indexer timestamps are usually 'HH:MM:SS.fff' strings, but be
    defensive and also accept a plain number of seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"Unsupported timestamp value: {value!r}")

    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unexpected timestamp format: {value!r}")
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _extract_insights(index_payload: dict[str, Any]) -> dict[str, Any]:
    """Handle both the common 'videos[0].insights' shape and a flatter
    'insights' shape defensively, since payload shape can vary slightly by
    API version / indexing preset."""
    videos = index_payload.get("videos")
    if isinstance(videos, list) and videos:
        first_video = cast("dict[str, Any]", videos[0])
        insights = first_video.get("insights")
        if isinstance(insights, dict):
            return cast("dict[str, Any]", insights)
    insights = index_payload.get("insights")
    if isinstance(insights, dict):
        return cast("dict[str, Any]", insights)
    return {}


def _video_duration_seconds(
    index_payload: dict[str, Any], insights: dict[str, Any]
) -> float | None:
    """
    Best-effort extraction of the video's total duration, in seconds.

    `insights.duration` is tried first, with `durationInSeconds` (a plain
    top-level number) as the fallback when it's absent or unparseable.
    Real Video Indexer responses have consistently returned
    insights.duration as a plain "H:MM:SS.ff" string (confirmed against
    this project's own raw insights captures, e.g.
    output/dollar_calendar_phone.raw_insights.json's "0:00:12.64") -- the
    previous version of this function only handled a
    `{"seconds": ...}`/`{"time": ...}` dict or a bare number, so it
    silently returned None for that actual common case, which in turn
    disabled every HTML report's timeline (write_html/write_html_all treat
    an unknown duration as "nothing to position bars against").
    """
    duration = insights.get("duration")
    if isinstance(duration, dict):
        # Some responses nest as {"time": "HH:MM:SS.fff", "seconds": ...}
        duration_dict = cast("dict[str, Any]", duration)
        if "seconds" in duration_dict:
            return float(duration_dict["seconds"])
        if "time" in duration_dict:
            return _timestamp_to_seconds(duration_dict["time"])
    elif isinstance(duration, (int, float)):
        return float(duration)
    elif isinstance(duration, str) and duration:
        try:
            return _timestamp_to_seconds(duration)
        except (TypeError, ValueError):
            pass

    fallback = index_payload.get("durationInSeconds")
    return float(fallback) if isinstance(fallback, (int, float)) else None


def _matches(term: str, needles: Iterable[str]) -> bool:
    term_lower = term.lower()
    return any(needle in term_lower for needle in needles)


def _item_name(item: dict[str, Any]) -> str:
    """Resolve the display name of one labels/keywords/detectedObjects item.

    `labels` and `keywords` items use "name" (or occasionally "text");
    `detectedObjects` items instead use "displayName" (lowercase, e.g.
    "car") or "type" (capitalized, e.g. "Car") -- checked in this order so
    the lowercase, human-friendly form is preferred when both are present.
    """
    return (
        item.get("name")
        or item.get("displayName")
        or item.get("text")
        or item.get("type")
        or ""
    )


def _item_ocr_box(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Resolve an `ocr` item's static (left, top, width, height) pixel box,
    if all four are present and numeric. This is one box per OCR item, not
    per-instance -- see module docstring. Returns None for any other bucket
    (labels/keywords/detectedObjects items never have these keys)."""
    left, top, width, height = (
        item.get("left"),
        item.get("top"),
        item.get("width"),
        item.get("height"),
    )
    # Checked as individual isinstance calls (rather than
    # `all(isinstance(v, ...) for v in (...))`) so mypy can actually narrow
    # each variable's type through the `and` chain -- a loop over a tuple
    # doesn't narrow anything, which previously left the `float(...)` calls
    # below type-checking against `Any | None` despite a (mismatched)
    # `# type: ignore` silencing the symptom rather than the cause.
    if (
        isinstance(left, (int, float))
        and isinstance(top, (int, float))
        and isinstance(width, (int, float))
        and isinstance(height, (int, float))
    ):
        return float(left), float(top), float(width), float(height)
    return None


def _video_dimensions(index_payload: dict[str, Any]) -> tuple[float, float] | None:
    """Best-effort extraction of the source video's frame width/height in
    pixels (Video Indexer reports these on the video object itself, e.g.
    `videos[0].width`/`height`, not inside `insights` -- confirmed against
    this project's own raw insights captures). Used only to interpret
    `ocr`'s static per-item box in `_looks_like_ocr_overlay`; returns None
    (which disables that heuristic entirely) if either dimension is
    missing, non-numeric, or non-positive."""
    videos = index_payload.get("videos")
    source: dict[str, Any] = index_payload
    if isinstance(videos, list) and videos:
        source = cast("dict[str, Any]", videos[0])
    width, height = source.get("width"), source.get("height")
    if (
        isinstance(width, (int, float))
        and isinstance(height, (int, float))
        and width > 0
        and height > 0
    ):
        return float(width), float(height)
    return None


def _looks_like_ocr_overlay(
    ocr_box: tuple[float, float, float, float] | None,
    frame_width: float | None,
    frame_height: float | None,
    edge_margin_fraction: float = DEFAULT_OCR_OVERLAY_EDGE_MARGIN_FRACTION,
    max_area_fraction: float = DEFAULT_OCR_OVERLAY_MAX_AREA_FRACTION,
) -> bool:
    """Heuristic: does this OCR item's static box look like a burned-in
    overlay (a camera timestamp, a watermark) rather than real in-scene
    text? True only when both the box and the frame's dimensions are known,
    the box is small relative to the frame, AND it sits within
    `edge_margin_fraction` of at least one edge -- e.g. a security-camera
    clock reading "17 14 07" near a top corner, as literally captured in
    this project's own cashier.raw_insights.json. A pixel heuristic on one
    item's static box, not a guarantee; used only to gate the
    `person handling cash` composite, never to hide raw OCR rows."""
    if ocr_box is None or not frame_width or not frame_height:
        return False
    left, top, width, height = ocr_box
    if (width * height) / (frame_width * frame_height) > max_area_fraction:
        return False
    near_left = left <= frame_width * edge_margin_fraction
    near_right = (left + width) >= frame_width * (1 - edge_margin_fraction)
    near_top = top <= frame_height * edge_margin_fraction
    near_bottom = (top + height) >= frame_height * (1 - edge_margin_fraction)
    return near_left or near_right or near_top or near_bottom


def _iter_valid_instances(
    item: dict[str, Any],
) -> Iterator[tuple[float, float, float | None]]:
    """Yield (start_seconds, end_seconds, confidence) for every instance of
    one labels/keywords item that has usable start/end timestamps. Shared by
    `_events_from_bucket` and `_all_events_from_bucket` so the instance-level
    parsing (and its error handling) exists in exactly one place."""
    raw_instances = cast("list[Any]", item.get("instances") or [])
    for raw_instance in raw_instances:
        instance = cast("dict[str, Any]", raw_instance)
        start = instance.get("start")
        end = instance.get("end")
        if start is None or end is None:
            continue
        try:
            start_s = _timestamp_to_seconds(start)
            end_s = _timestamp_to_seconds(end)
        except ValueError:
            continue
        confidence = instance.get("confidence", item.get("confidence"))
        yield start_s, end_s, float(confidence) if confidence is not None else None


def _events_from_bucket(
    bucket: Any, source: str, needles: list[str]
) -> list[ActionEvent]:
    if not isinstance(bucket, list):
        return []

    events: list[ActionEvent] = []
    for raw_item in cast("list[Any]", bucket):
        item = cast("dict[str, Any]", raw_item)
        name = _item_name(item)
        if not name or not _matches(name, needles):
            continue
        events.extend(
            ActionEvent(
                source=source,
                matched_term=name,
                start_seconds=start_s,
                end_seconds=end_s,
                confidence=confidence,
            )
            for start_s, end_s, confidence in _iter_valid_instances(item)
        )
    return events


@dataclass
class DetectedAction:
    """A single timestamped occurrence of *any* label/keyword Video Indexer
    tagged, i.e. one entry in the full "everything detected" timeline."""

    name: str  # the label/keyword text as returned by Video Indexer
    source: str  # which insight bucket this came from, e.g. "labels" or "keywords"
    start_seconds: float
    end_seconds: float
    confidence: float | None
    evidence: str | None = None  # for source="derived": the two items that overlapped
    # for source="ocr": that item's static (left, top, width, height) pixel
    # box, if present -- used only by `_looks_like_ocr_overlay` to spot a
    # likely burned-in overlay; never surfaced in reports.
    ocr_box: tuple[float, float, float, float] | None = None

    @property
    def duration_seconds(self) -> float:
        """How long this occurrence lasted, in seconds (never negative)."""
        return max(0.0, self.end_seconds - self.start_seconds)


@dataclass
class ActionSummary:
    """Aggregated stats for one distinct action/label name across the video."""

    name: str
    sources: set[str]
    occurrence_count: int
    total_duration_seconds: float
    first_occurrence_seconds: float
    last_occurrence_seconds: float
    max_confidence: float | None


@dataclass
class AllActionsReport:
    """Every action/label/keyword Video Indexer detected anywhere in the
    video, timestamped and grouped -- the "what happened, and when" view,
    as opposed to `ActionReport`'s search for one action of interest."""

    video_duration_seconds: float | None
    actions: list[DetectedAction] = field(default_factory=list[DetectedAction])
    distinct_people_tracked: int = 0
    min_confidence: float | None = None

    @property
    def occurrence_count(self) -> int:
        """Number of distinct timestamped occurrences found, across all actions."""
        return len(self.actions)

    @property
    def distinct_action_count(self) -> int:
        """Number of distinct action/label names detected."""
        return len({a.name for a in self.actions})

    @property
    def summaries(self) -> list[ActionSummary]:
        """One aggregated row per distinct action name, ordered by first
        occurrence (earliest first)."""
        by_name: dict[str, list[DetectedAction]] = {}
        for a in self.actions:
            by_name.setdefault(a.name, []).append(a)

        result: list[ActionSummary] = []
        for name, items in by_name.items():
            confidences = [i.confidence for i in items if i.confidence is not None]
            result.append(
                ActionSummary(
                    name=name,
                    sources={i.source for i in items},
                    occurrence_count=len(items),
                    total_duration_seconds=sum(i.duration_seconds for i in items),
                    first_occurrence_seconds=min(i.start_seconds for i in items),
                    last_occurrence_seconds=max(i.end_seconds for i in items),
                    max_confidence=max(confidences) if confidences else None,
                )
            )
        result.sort(key=lambda s: s.first_occurrence_seconds)
        return result


def _all_events_from_bucket(bucket: Any, source: str) -> list[DetectedAction]:
    """Like `_events_from_bucket`, but keeps every item regardless of name
    (no keyword/synonym filtering) -- used to build the full timeline."""
    if not isinstance(bucket, list):
        return []

    events: list[DetectedAction] = []
    for raw_item in cast("list[Any]", bucket):
        item = cast("dict[str, Any]", raw_item)
        name = _item_name(item)
        if not name:
            continue
        ocr_box = _item_ocr_box(item) if source == "ocr" else None
        events.extend(
            DetectedAction(
                name=name,
                source=source,
                start_seconds=start_s,
                end_seconds=end_s,
                confidence=confidence,
                ocr_box=ocr_box,
            )
            for start_s, end_s, confidence in _iter_valid_instances(item)
        )
    return events


def _overlap_seconds(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> tuple[float, float] | None:
    """The [start, end] interval where two occurrences overlap, or None if
    they don't overlap at all."""
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end <= start:
        return None
    return start, end


def _temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Temporal Intersection-over-Union: overlap duration divided by the
    combined (union) duration the two occurrences span together. 1.0 means
    they cover exactly the same interval; a value near 0 means one is much
    longer than their shared overlap (e.g. a person label spanning the
    whole video against a one-second object blip). This is IoU applied to
    *time*, not space -- see module docstring's `min_temporal_iou`
    paragraph for why that's a useful proxy despite not being spatial."""
    overlap = _overlap_seconds(a_start, a_end, b_start, b_end)
    if overlap is None:
        return 0.0
    union_start, union_end = min(a_start, b_start), max(a_end, b_end)
    union = union_end - union_start
    if union <= 0:
        return 0.0
    return (overlap[1] - overlap[0]) / union


def _observed_people_intervals(
    insights: dict[str, Any],
) -> list[list[tuple[float, float]]] | None:
    """Per-distinct-person lists of (start, end) intervals from the
    `observedPeople` insight (Advanced-preset only -- see README), or None
    if that bucket is absent/empty, meaning the data needed for
    `_people_in_frame_count` simply isn't there (distinct from "zero people
    tracked", a real but different outcome)."""
    raw_people = insights.get("observedPeople") or insights.get("people")
    if not isinstance(raw_people, list) or not raw_people:
        return None
    intervals: list[list[tuple[float, float]]] = []
    for raw_person in cast("list[Any]", raw_people):
        person = cast("dict[str, Any]", raw_person)
        intervals.append(
            [(start_s, end_s) for start_s, end_s, _ in _iter_valid_instances(person)]
        )
    return intervals


def _people_in_frame_count(
    people_intervals: list[list[tuple[float, float]]] | None,
    start: float,
    end: float,
) -> int | None:
    """How many distinct tracked people (`_observed_people_intervals`) had
    at least one instance overlapping [start, end], or None if no
    `observedPeople` data was available at all to answer the question."""
    if people_intervals is None:
        return None
    return sum(
        1
        for person_spans in people_intervals
        if any(_overlap_seconds(start, end, s, e) is not None for s, e in person_spans)
    )


def _matches_composite_side(
    actions: list[DetectedAction],
    side: CompositeSide,
    frame_width: float | None,
    frame_height: float | None,
) -> list[DetectedAction]:
    """Every item in `actions` that matches one side of a composite
    (`CompositeSide.matches`), excluding an OCR item that looks like a
    burned-in overlay (`_looks_like_ocr_overlay`) -- overlays are the only
    items with a spatial box to judge at all, so this excludes nothing for
    any non-OCR item."""
    return [
        a
        for a in actions
        if side.matches(a.name)
        and not _looks_like_ocr_overlay(a.ocr_box, frame_width, frame_height)
    ]


def _composite_overlap_window(
    left: DetectedAction,
    right: DetectedAction,
    min_overlap_seconds: float | None,
    min_temporal_iou: float | None,
) -> tuple[float, float] | None:
    """The [start, end] window `left` and `right` overlap in, or None if
    they don't overlap at all, or the overlap fails the opt-in
    `min_overlap_seconds`/`min_temporal_iou` thresholds -- see
    `_derive_composite_actions`'s docstring for what each means."""
    overlap = _overlap_seconds(
        left.start_seconds, left.end_seconds, right.start_seconds, right.end_seconds
    )
    if overlap is None:
        return None
    if (
        min_overlap_seconds is not None
        and (overlap[1] - overlap[0]) < min_overlap_seconds
    ):
        return None
    if min_temporal_iou is not None:
        iou = _temporal_iou(
            left.start_seconds, left.end_seconds, right.start_seconds, right.end_seconds
        )
        if iou < min_temporal_iou:
            return None
    return overlap


def _composite_evidence(
    left: DetectedAction, right: DetectedAction, people_in_frame: int | None
) -> str:
    """The human-readable `evidence` string for one derived composite
    occurrence: the two contributing items, plus a people-in-frame count
    when `observedPeople` data was available at all (see
    `_people_in_frame_count`)."""
    evidence = f"{left.name} ({left.source}) + {right.name} ({right.source})"
    if people_in_frame is not None:
        evidence += f" -- {people_in_frame} person(s) in frame"
    return evidence


def _rejected_by_single_person_filter(
    people_in_frame: int | None, require_single_person: bool
) -> bool:
    """True if `require_single_person` is on and the people-in-frame count
    is known but isn't exactly 1 -- see `_derive_composite_actions`'s
    docstring for why "unknown" (None) is never rejected here."""
    if not require_single_person or people_in_frame is None:
        return False
    return people_in_frame != 1


def _best_composite_occurrences(
    composite: CompositeAction,
    actions: list[DetectedAction],
    min_overlap_seconds: float | None,
    frame_width: float | None,
    frame_height: float | None,
    min_temporal_iou: float | None,
    people_intervals: list[list[tuple[float, float]]] | None,
    require_single_person: bool,
) -> dict[tuple[str, float, float], tuple[float, str]]:
    """The best (highest-confidence) occurrence per (name, start, end) key
    for one composite action, across every left/right item pair that
    overlaps and survives every opt-in filter -- one composite's
    contribution to `_derive_composite_actions`'s `best_by_key`."""
    left_items = _matches_composite_side(
        actions, composite.left, frame_width, frame_height
    )
    right_items = _matches_composite_side(
        actions, composite.right, frame_width, frame_height
    )

    best: dict[tuple[str, float, float], tuple[float, str]] = {}
    for left, right in product(left_items, right_items):
        overlap = _composite_overlap_window(
            left, right, min_overlap_seconds, min_temporal_iou
        )
        if overlap is None or left.confidence is None or right.confidence is None:
            continue
        people_in_frame = _people_in_frame_count(
            people_intervals, overlap[0], overlap[1]
        )
        if _rejected_by_single_person_filter(people_in_frame, require_single_person):
            continue
        confidence = min(left.confidence, right.confidence)
        key = (composite.name, *overlap)
        if key not in best or confidence > best[key][0]:
            best[key] = (confidence, _composite_evidence(left, right, people_in_frame))
    return best


def _derive_composite_actions(
    actions: list[DetectedAction],
    min_overlap_seconds: float | None = None,
    frame_width: float | None = None,
    frame_height: float | None = None,
    min_temporal_iou: float | None = None,
    people_intervals: list[list[tuple[float, float]]] | None = None,
    require_single_person: bool = False,
) -> list[DetectedAction]:
    """Synthesize named composite actions (`COMPOSITE_ACTIONS`, source
    "derived") from temporal overlaps between two independently-detected
    concepts already in `actions` -- e.g. a person-like label and a "cell
    phone" object both present at once, standing in for "person using
    phone" since Video Indexer has no such label/object of its own.

    Confidence is the min of the two overlapping items' confidences (a
    composite claim is only as strong as its weaker piece of evidence);
    an overlap where either side has no confidence value is dropped, since
    there's nothing meaningful to threshold. Identical (name, start, end)
    overlaps -- e.g. both a "person" and a "man" label spanning the same
    interval, each pairing with the same object -- are deduplicated,
    keeping the highest confidence, so one real overlap doesn't get
    reported as several near-duplicate rows.

    `min_overlap_seconds`, if given, drops overlaps shorter than it (a 1-2
    frame overlap between two independent detections is usually detector
    jitter, not a real co-occurrence) -- opt-in, so existing callers see
    every overlap unless they ask for filtering.

    Each returned action's `evidence` names the two items that produced it
    (e.g. "person (labels) + cell phone (objects)"), so a reviewer can see
    why it was flagged without re-running raw insights.

    `frame_width`/`frame_height` (the source video's pixel dimensions, from
    `_video_dimensions`), if both given, let `_looks_like_ocr_overlay`
    exclude an OCR item that looks like a burned-in overlay (a camera
    timestamp, a watermark) from matching as composite evidence -- this
    only ever affects OCR-sourced items (everything else has no spatial
    box to judge), and currently only `person handling cash`'s right side
    is OCR-based.

    `min_temporal_iou`, if given, drops an overlap whose temporal
    Intersection-over-Union (`_temporal_iou`) is below it -- a *time*-only
    proxy for "these two detections' durations line up," not a spatial
    measure (see module docstring).

    `people_intervals` (from `_observed_people_intervals`), if given,
    attaches a people-in-frame count to `evidence` for every derived
    action, and `require_single_person`, if true, additionally drops any
    overlap where that count isn't exactly 1 (see module docstring).

    The per-composite work (matching each side, filtering, and picking the
    best occurrence per key) lives in `_best_composite_occurrences`; this
    function just runs that once per entry in `COMPOSITE_ACTIONS` and merges
    the results, keeping this top-level function's own control flow (and
    Cognitive Complexity) minimal.
    """
    best_by_key: dict[tuple[str, float, float], tuple[float, str]] = {}
    for composite in COMPOSITE_ACTIONS:
        best_by_key.update(
            _best_composite_occurrences(
                composite,
                actions,
                min_overlap_seconds,
                frame_width,
                frame_height,
                min_temporal_iou,
                people_intervals,
                require_single_person,
            )
        )

    return [
        DetectedAction(
            name=name,
            source="derived",
            start_seconds=start_s,
            end_seconds=end_s,
            confidence=confidence,
            evidence=evidence,
        )
        for (name, start_s, end_s), (confidence, evidence) in best_by_key.items()
    ]


def _merge_nearby_occurrences(
    actions: list[DetectedAction], gap_seconds: float
) -> list[DetectedAction]:
    """Merge occurrences of the same (name, source) that overlap or are
    within `gap_seconds` of each other into a single occurrence spanning
    their combined interval, keeping the higher confidence and either's
    evidence. Reduces fragmented near-duplicate detections -- e.g. Video
    Indexer (or `_derive_composite_actions`) reporting what's really one
    continuous event as several adjacent slivers -- to a smaller number of
    human-meaningful spans. Opt-in: only called when a caller passes
    `merge_gap_seconds` to `analyze_all`.
    """
    by_key: dict[tuple[str, str], list[DetectedAction]] = {}
    for a in actions:
        by_key.setdefault((a.name, a.source), []).append(a)

    merged: list[DetectedAction] = []
    for items in by_key.values():
        items.sort(key=lambda a: a.start_seconds)
        current = items[0]
        for nxt in items[1:]:
            if nxt.start_seconds <= current.end_seconds + gap_seconds:
                confidences = [
                    c for c in (current.confidence, nxt.confidence) if c is not None
                ]
                current = DetectedAction(
                    name=current.name,
                    source=current.source,
                    start_seconds=current.start_seconds,
                    end_seconds=max(current.end_seconds, nxt.end_seconds),
                    confidence=max(confidences) if confidences else None,
                    evidence=current.evidence or nxt.evidence,
                )
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
    return merged


def _merge_events(events: list[ActionEvent], gap_seconds: float) -> list[ActionEvent]:
    """`_merge_nearby_occurrences` for `ActionEvent` instead of
    `DetectedAction` -- same merge, just round-tripped through
    `DetectedAction` (whose `name` field lines up with `ActionEvent`'s
    `matched_term`) so the interval-merge logic exists in exactly one place.
    Used by `analyze()`'s opt-in `merge_gap_seconds`.
    """
    as_actions = [
        DetectedAction(
            name=e.matched_term,
            source=e.source,
            start_seconds=e.start_seconds,
            end_seconds=e.end_seconds,
            confidence=e.confidence,
            evidence=e.evidence,
        )
        for e in events
    ]
    merged = _merge_nearby_occurrences(as_actions, gap_seconds=gap_seconds)
    return [
        ActionEvent(
            source=a.source,
            matched_term=a.name,
            start_seconds=a.start_seconds,
            end_seconds=a.end_seconds,
            confidence=a.confidence,
            evidence=a.evidence,
        )
        for a in merged
    ]


def analyze_all(
    index_payload: dict[str, Any],
    min_confidence: float | None = None,
    min_overlap_seconds: float | None = None,
    merge_gap_seconds: float | None = None,
    min_temporal_iou: float | None = None,
    require_single_person: bool = False,
) -> AllActionsReport:
    """Extract *every* timestamped label/keyword Video Indexer detected in
    the video -- a full "all actions" timeline, with no filtering to a
    single action of interest.

    `min_confidence` (0.0-1.0), if given, drops occurrences whose confidence
    is known and below the threshold; occurrences with no confidence value
    are always kept, since Video Indexer doesn't score every insight bucket.

    `min_overlap_seconds`, if given, is passed to `_derive_composite_actions`
    to drop composite overlaps shorter than it (detector jitter).

    `merge_gap_seconds`, if given, is passed to `_merge_nearby_occurrences`
    to collapse occurrences of the same action within that many seconds of
    each other (or overlapping) into fewer, more meaningful spans. Applied
    after composite derivation, so it also merges fragmented composites.

    `min_temporal_iou`/`require_single_person`, if given, are passed to
    `_derive_composite_actions` -- see its docstring and the module
    docstring's paragraph on both (neither is real spatial gating).
    """
    insights = _extract_insights(index_payload)
    duration = _video_duration_seconds(index_payload, insights)
    frame_dims = _video_dimensions(index_payload)
    people_intervals = _observed_people_intervals(insights)

    actions: list[DetectedAction] = []
    actions.extend(_all_events_from_bucket(insights.get("labels"), "labels"))
    actions.extend(_all_events_from_bucket(insights.get("keywords"), "keywords"))
    actions.extend(_all_events_from_bucket(insights.get("detectedObjects"), "objects"))
    actions.extend(_all_events_from_bucket(insights.get("ocr"), "ocr"))
    actions.extend(
        _derive_composite_actions(
            actions,
            min_overlap_seconds=min_overlap_seconds,
            frame_width=frame_dims[0] if frame_dims else None,
            frame_height=frame_dims[1] if frame_dims else None,
            min_temporal_iou=min_temporal_iou,
            people_intervals=people_intervals,
            require_single_person=require_single_person,
        )
    )

    if min_confidence is not None:
        actions = [
            a for a in actions if a.confidence is None or a.confidence >= min_confidence
        ]

    if merge_gap_seconds is not None:
        actions = _merge_nearby_occurrences(actions, gap_seconds=merge_gap_seconds)

    actions.sort(key=lambda a: (a.start_seconds, a.name))

    raw_people: Any = insights.get("observedPeople") or insights.get("people") or []
    distinct_people = (
        len(cast("list[Any]", raw_people)) if isinstance(raw_people, list) else 0
    )

    return AllActionsReport(
        video_duration_seconds=duration,
        actions=actions,
        distinct_people_tracked=distinct_people,
        min_confidence=min_confidence,
    )


def analyze(
    index_payload: dict[str, Any],
    action: str,
    extra_synonyms: list[str] | None = None,
    min_overlap_seconds: float | None = None,
    merge_gap_seconds: float | None = None,
    min_temporal_iou: float | None = None,
    require_single_person: bool = False,
) -> ActionReport:
    """Extract action events for `action` (e.g. "jumping") from a Video
    Indexer index/insights payload.

    `merge_gap_seconds`, if given, collapses occurrences of the same matched
    term (or overlapping ones) within that many seconds of each other into a
    single occurrence -- the same opt-in fragmentation cleanup `analyze_all`
    offers, applied here too so a single-action report (e.g. `main.py
    --action "using phone"`) isn't stuck with duplicate/overlapping rows for
    what's really one continuous event. See `_merge_nearby_occurrences`.

    `min_temporal_iou`/`require_single_person`, if given, are passed to
    `_derive_composite_actions` -- see its docstring and the module
    docstring's paragraph on both (neither is real spatial gating).
    """
    action_key = action.lower().strip()
    needles = set(DEFAULT_SYNONYMS.get(action_key, [action_key]))
    if extra_synonyms:
        needles.update(s.lower() for s in extra_synonyms)
    needles.add(action_key)

    insights = _extract_insights(index_payload)
    duration = _video_duration_seconds(index_payload, insights)

    events: list[ActionEvent] = []
    events.extend(_events_from_bucket(insights.get("labels"), "labels", list(needles)))
    events.extend(
        _events_from_bucket(insights.get("keywords"), "keywords", list(needles))
    )
    events.extend(
        _events_from_bucket(insights.get("detectedObjects"), "objects", list(needles))
    )
    events.extend(_events_from_bucket(insights.get("ocr"), "ocr", list(needles)))

    # Composite ("derived") actions -- e.g. "person using phone" -- aren't
    # any single label/keyword/object Video Indexer returns, so they can't
    # be found by filtering one bucket; build every base action first
    # (unfiltered), derive overlaps from that, then keep only ones whose
    # composite name matches this search (substring match against e.g.
    # "phone" or "using phone", same as any other action).
    all_base_actions = (
        _all_events_from_bucket(insights.get("labels"), "labels")
        + _all_events_from_bucket(insights.get("keywords"), "keywords")
        + _all_events_from_bucket(insights.get("detectedObjects"), "objects")
        + _all_events_from_bucket(insights.get("ocr"), "ocr")
    )
    frame_dims = _video_dimensions(index_payload)
    people_intervals = _observed_people_intervals(insights)
    derived_actions = _derive_composite_actions(
        all_base_actions,
        min_overlap_seconds=min_overlap_seconds,
        frame_width=frame_dims[0] if frame_dims else None,
        frame_height=frame_dims[1] if frame_dims else None,
        min_temporal_iou=min_temporal_iou,
        people_intervals=people_intervals,
        require_single_person=require_single_person,
    )
    for derived in derived_actions:
        if _matches(derived.name, needles):
            events.append(
                ActionEvent(
                    source=derived.source,
                    matched_term=derived.name,
                    start_seconds=derived.start_seconds,
                    end_seconds=derived.end_seconds,
                    confidence=derived.confidence,
                    evidence=derived.evidence,
                )
            )

    if merge_gap_seconds is not None:
        events = _merge_events(events, gap_seconds=merge_gap_seconds)

    events.sort(key=lambda e: e.start_seconds)

    raw_people: Any = insights.get("observedPeople") or insights.get("people") or []
    distinct_people = (
        len(cast("list[Any]", raw_people)) if isinstance(raw_people, list) else 0
    )

    report = ActionReport(
        action=action,
        video_duration_seconds=duration,
        events=events,
        distinct_people_tracked=distinct_people,
        matched_terms_found={e.matched_term for e in events},
    )
    return report
