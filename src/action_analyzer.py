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
as a first pass to review rather than a guaranteed-accurate label.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, cast

from src.constants import COMPOSITE_ACTIONS, DEFAULT_SYNONYMS


@dataclass
class ActionEvent:
    """A single timestamped occurrence of the action of interest."""

    source: str  # which insight bucket this came from, e.g. "labels" or "keywords"
    matched_term: str  # the label/keyword text that matched
    start_seconds: float
    end_seconds: float
    confidence: float | None

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
    """Best-effort extraction of the video's total duration, in seconds."""
    duration = insights.get("duration") or index_payload.get("durationInSeconds")
    if isinstance(duration, dict):
        # Some responses nest as {"time": "HH:MM:SS.fff", "seconds": ...}
        duration_dict = cast("dict[str, Any]", duration)
        if "seconds" in duration_dict:
            return float(duration_dict["seconds"])
        if "time" in duration_dict:
            return _timestamp_to_seconds(duration_dict["time"])
    if isinstance(duration, (int, float)):
        return float(duration)
    return None


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
        events.extend(
            DetectedAction(
                name=name,
                source=source,
                start_seconds=start_s,
                end_seconds=end_s,
                confidence=confidence,
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


def _derive_composite_actions(actions: list[DetectedAction]) -> list[DetectedAction]:
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
    """
    best_by_key: dict[tuple[str, float, float], float | None] = {}
    for name, left_names, right_names in COMPOSITE_ACTIONS:
        left_items = [a for a in actions if a.name.lower() in left_names]
        right_items = [a for a in actions if a.name.lower() in right_names]
        for left in left_items:
            for right in right_items:
                overlap = _overlap_seconds(
                    left.start_seconds,
                    left.end_seconds,
                    right.start_seconds,
                    right.end_seconds,
                )
                if overlap is None:
                    continue
                if left.confidence is None or right.confidence is None:
                    continue
                confidence = min(left.confidence, right.confidence)
                key = (name, *overlap)
                if key not in best_by_key or confidence > cast(
                    "float", best_by_key[key]
                ):
                    best_by_key[key] = confidence

    return [
        DetectedAction(
            name=name,
            source="derived",
            start_seconds=start_s,
            end_seconds=end_s,
            confidence=confidence,
        )
        for (name, start_s, end_s), confidence in best_by_key.items()
    ]


def analyze_all(
    index_payload: dict[str, Any],
    min_confidence: float | None = None,
) -> AllActionsReport:
    """Extract *every* timestamped label/keyword Video Indexer detected in
    the video -- a full "all actions" timeline, with no filtering to a
    single action of interest.

    `min_confidence` (0.0-1.0), if given, drops occurrences whose confidence
    is known and below the threshold; occurrences with no confidence value
    are always kept, since Video Indexer doesn't score every insight bucket.
    """
    insights = _extract_insights(index_payload)
    duration = _video_duration_seconds(index_payload, insights)

    actions: list[DetectedAction] = []
    actions.extend(_all_events_from_bucket(insights.get("labels"), "labels"))
    actions.extend(_all_events_from_bucket(insights.get("keywords"), "keywords"))
    actions.extend(_all_events_from_bucket(insights.get("detectedObjects"), "objects"))
    actions.extend(_all_events_from_bucket(insights.get("ocr"), "ocr"))
    actions.extend(_derive_composite_actions(actions))

    if min_confidence is not None:
        actions = [
            a for a in actions if a.confidence is None or a.confidence >= min_confidence
        ]

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
) -> ActionReport:
    """Extract action events for `action` (e.g. "jumping") from a Video
    Indexer index/insights payload."""
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
    for derived in _derive_composite_actions(all_base_actions):
        if _matches(derived.name, needles):
            events.append(
                ActionEvent(
                    source=derived.source,
                    matched_term=derived.name,
                    start_seconds=derived.start_seconds,
                    end_seconds=derived.end_seconds,
                    confidence=derived.confidence,
                )
            )

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
