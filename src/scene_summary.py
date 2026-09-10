"""
Synthesizes a single, human-readable paragraph describing what a video's
Video Indexer metadata *suggests* might be happening in it -- built
entirely from action_analyzer.py's `analyze_all()` (the same
label/keyword/object timeline detect_all_actions.py reports), with no
frame inspection, no video file, and no LLM call of any kind.

This is a fundamentally weaker guarantee than actually looking at the
video: it can only ever repeat what Video Indexer's own classifiers
flagged, never anything actually seen in the footage. The paragraph is
written to read like an ordinary piece of descriptive writing about the
scene -- it never names "Video Indexer", "Azure", "metadata", "labels", a
confidence threshold, or anything else about how the guess was produced --
but it still hedges every claim in plain English ("appears to", "seems
to", "it's unclear"), since none of it has actually been confirmed by
watching the footage.

The one heuristic worth calling out: rather than hardcoding which label
names count as "scene setting" vs. "a notable moment" (Video Indexer's
label vocabulary is effectively open-ended, so any such list would be
incomplete), this classifies purely by how much of the video each label
spans: a label covering at least half the video's duration is treated as
persistent/scene-setting context (e.g. "outdoor", "building"); anything
briefer is treated as a punctual, timestamped moment worth calling out
(e.g. "kiss" at one specific instant). detectedObjects-sourced items are
always kept in their own "objects" bucket regardless of duration, since a
detected object is inherently punctual/incidental, not scene-setting.
"""

from __future__ import annotations

from src.action_analyzer import ActionSummary, AllActionsReport
from src.constants import (
    SCENE_SUMMARY_BOILERPLATE_NAMES as _BOILERPLATE_NAMES,
)
from src.constants import (
    SCENE_SUMMARY_LOCATION_LABELS as _LOCATION_LABELS,
)
from src.constants import (
    SCENE_SUMMARY_MAX_NOTABLE_MOMENTS as _MAX_NOTABLE_MOMENTS,
)
from src.constants import (
    SCENE_SUMMARY_MAX_OBJECT_TERMS as _MAX_OBJECT_TERMS,
)
from src.constants import (
    SCENE_SUMMARY_MAX_SETTING_TERMS as _MAX_SETTING_TERMS,
)
from src.constants import (
    SCENE_SUMMARY_NO_ARTICLE_NAMES as _NO_ARTICLE_NAMES,
)
from src.constants import (
    SCENE_SUMMARY_PERSISTENT_COVERAGE_RATIO as _PERSISTENT_COVERAGE_RATIO,
)
from src.constants import (
    SCENE_SUMMARY_SCATTERED_MAX_COVERAGE_RATIO as _SCATTERED_MAX_COVERAGE_RATIO,
)
from src.constants import (
    SCENE_SUMMARY_SCATTERED_MIN_SPAN_SECONDS as _SCATTERED_MIN_SPAN_SECONDS,
)
from src.constants import (
    SCENE_SUMMARY_SMALL_NUMBER_WORDS as _SMALL_NUMBER_WORDS,
)


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


def _with_article(name: str) -> str:
    """ "kiss" -> "a kiss"; "outerwear" -> "outerwear" (see
    _NO_ARTICLE_NAMES); "outfit" -> "an outfit"."""
    if name.lower() in _NO_ARTICLE_NAMES:
        return name
    article = "an" if name[:1].lower() in "aeiou" else "a"
    return f"{article} {name}"


def _count_word(n: int) -> str:
    """Spell out small counts ("Three people") rather than starting a
    sentence with a bare numeral, which reads oddly in prose; falls back
    to the numeral itself for anything larger."""
    return _SMALL_NUMBER_WORDS.get(n, str(n))


def _time_phrase(seconds: float) -> str:
    """A rounded, conversational timestamp -- "around the 10-second mark"
    reads like something a person would actually write, unlike a decimal
    timestamp."""
    return f"around the {seconds:.0f}-second mark"


def _fmt_moment(summary: ActionSummary) -> str:
    """Phrase one notable-moment clause for `summary` as an ordinary
    descriptive sentence fragment -- choosing between a single occurrence
    ("a kiss seems to happen around the 10-second mark") and a scattered/
    recurring one ("a coat is glimpsed early on, then again near the end")
    based on how much of its first-to-last window it actually covers --
    see the note above `_SCATTERED_MIN_SPAN_SECONDS`."""
    span = summary.last_occurrence_seconds - summary.first_occurrence_seconds
    coverage = summary.total_duration_seconds / span if span > 0 else 1.0
    subject = _with_article(summary.name)
    if span > _SCATTERED_MIN_SPAN_SECONDS and coverage < _SCATTERED_MAX_COVERAGE_RATIO:
        return f"{subject} is glimpsed early on, then again later in the clip"
    return f"{subject} seems to appear {_time_phrase(summary.first_occurrence_seconds)}"


def _categorize(
    report: AllActionsReport,
) -> tuple[list[ActionSummary], list[ActionSummary], list[ActionSummary]]:
    """Split `report.summaries` (already confidence-filtered by
    `analyze_all`) into (persistent/setting, punctual/notable, objects) --
    see the module docstring for the coverage-ratio heuristic."""
    duration = report.video_duration_seconds
    persistent: list[ActionSummary] = []
    punctual: list[ActionSummary] = []
    objects: list[ActionSummary] = []

    for summary in report.summaries:
        if summary.name.lower() in _BOILERPLATE_NAMES:
            continue
        if summary.sources == {"objects"}:
            objects.append(summary)
            continue
        coverage = (
            summary.total_duration_seconds / duration
            if duration and duration > 0
            else 0.0
        )
        if coverage >= _PERSISTENT_COVERAGE_RATIO:
            persistent.append(summary)
        else:
            punctual.append(summary)

    persistent.sort(key=lambda s: s.total_duration_seconds, reverse=True)
    punctual.sort(key=lambda s: s.first_occurrence_seconds)
    objects.sort(key=lambda s: s.first_occurrence_seconds)
    return persistent, punctual, objects


def _has_person_signal(report: AllActionsReport) -> bool:
    """Whether any surviving (confidence-filtered) action at all suggests a
    person is present -- used only as a fallback when
    `distinct_people_tracked` is zero/unavailable (e.g. the Default
    indexing preset, which doesn't return observedPeople)."""
    return any(a.name.lower() in {"person", "human face"} for a in report.actions)


def summarize_insights(report: AllActionsReport) -> str:
    """Build the final one-paragraph, plain-language description from an
    `AllActionsReport` (see action_analyzer.analyze_all) -- written as
    ordinary descriptive prose, with no reference to Video Indexer, Azure,
    labels, metadata, or confidence scores anywhere in the text."""
    persistent, punctual, objects = _categorize(report)

    sentences: list[str] = []

    duration = report.video_duration_seconds
    duration_phrase = (
        f"This roughly {duration:.0f}-second clip"
        if duration and duration > 0
        else "This clip"
    )
    location_term: str | None = None
    other_setting_terms: list[str] = []
    for s in persistent:
        key = s.name.lower()
        if key in _LOCATION_LABELS and location_term is None:
            location_term = _LOCATION_LABELS[key]
        else:
            other_setting_terms.append(s.name)
    other_setting_terms = other_setting_terms[:_MAX_SETTING_TERMS]

    if location_term and other_setting_terms:
        names = _join_terms([_with_article(n) for n in other_setting_terms])
        sentences.append(
            f"{duration_phrase} appears to take place {location_term}, with "
            f"{names} visible."
        )
    elif location_term:
        sentences.append(f"{duration_phrase} appears to take place {location_term}.")
    elif other_setting_terms:
        names = _join_terms([_with_article(n) for n in other_setting_terms])
        sentences.append(f"{duration_phrase} appears to feature {names}.")
    else:
        sentences.append(f"{duration_phrase}'s setting isn't very clear.")

    if report.distinct_people_tracked:
        noun = "person" if report.distinct_people_tracked == 1 else "people"
        verb = "appears" if report.distinct_people_tracked == 1 else "appear"
        sentences.append(
            f"{_count_word(report.distinct_people_tracked)} {noun} {verb} to "
            "be present across the footage."
        )
    elif _has_person_signal(report):
        sentences.append(
            "At least one person seems to appear, though it's unclear how many."
        )

    if punctual:
        # Truncation picks the shortest-duration (most punctual/eventful)
        # occurrences first -- e.g. a one-second "kiss" over a ten-second
        # "jacket" -- since picking purely by chronological order would
        # arbitrarily favor whatever happens earliest in the video
        # regardless of how brief/notable it is. The selected subset is
        # then re-sorted chronologically so the sentence still reads in
        # the order things happened.
        shown = sorted(punctual, key=lambda s: s.total_duration_seconds)[
            :_MAX_NOTABLE_MOMENTS
        ]
        shown.sort(key=lambda s: s.first_occurrence_seconds)
        clauses = [_fmt_moment(s) for s in shown]
        if len(clauses) == 1:
            joined = clauses[0]
        else:
            joined = "; ".join(clauses[:-1]) + f"; and {clauses[-1]}"
        sentences.append(f"Along the way, {joined}.")

    if objects:
        names = _join_terms(
            [_with_article(s.name) for s in objects[:_MAX_OBJECT_TERMS]]
        )
        sentences.append(f"There also seem to be {names} somewhere in frame.")

    sentences.append(
        "None of this has actually been confirmed by watching the footage, "
        "so it should be treated as an uncertain guess rather than a "
        "reliable account of what happens."
    )

    return " ".join(sentences)
