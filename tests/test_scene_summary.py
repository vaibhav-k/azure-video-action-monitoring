"""
Unit tests for src/scene_summary.py's summarize_insights(): the persistent-
vs-punctual-vs-object categorization, the scattered-vs-continuous-span
phrasing, the notable-moment truncation/ordering, the always-present
hedging closer, and that the output never leaks vendor/pipeline vocabulary
("Video Indexer", "Azure", "metadata", "confidence", "labels"). Built on
plain dicts shaped like Video Indexer's raw "videos[0].insights" export --
exactly what action_analyzer.analyze_all() already parses -- so no
fixtures or network calls are needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.action_analyzer import analyze_all
from src.scene_summary import summarize_insights

_BANNED_WORDS = ("video indexer", "azure", "metadata", "confidence", "label")


def _assert_reads_naturally(paragraph: str) -> None:
    lowered = paragraph.lower()
    for banned in _BANNED_WORDS:
        assert banned not in lowered, f"{banned!r} leaked into: {paragraph!r}"
    assert "\n" not in paragraph


def _label(name: str, *instances: tuple[str, str, float]) -> dict:
    """One `labels`-bucket item with the given (start, end, confidence)
    instances, in the raw Video Indexer shape analyze_all() expects."""
    return {
        "name": name,
        "instances": [
            {"start": start, "end": end, "confidence": confidence}
            for start, end, confidence in instances
        ],
    }


def _payload(labels: list[dict], duration_seconds: float = 30.0) -> dict:
    return {
        "durationInSeconds": duration_seconds,
        "videos": [{"insights": {"duration": str(duration_seconds), "labels": labels}}],
    }


def test_persistent_location_label_becomes_setting_sentence():
    payload = _payload(
        [_label("outdoor", ("0:00:00", "0:00:30", 0.99))],
        duration_seconds=30.0,
    )
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert "take place outdoors" in paragraph
    _assert_reads_naturally(paragraph)


def test_persistent_non_location_label_becomes_feature_sentence():
    payload = _payload([_label("building", ("0:00:00", "0:00:30", 0.99))])
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert "appears to feature a building" in paragraph


def test_persistent_location_and_noun_combine_naturally():
    payload = _payload(
        [
            _label("outdoor", ("0:00:00", "0:00:30", 0.99)),
            _label("building", ("0:00:00", "0:00:30", 0.97)),
        ]
    )
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert "take place outdoors, with a building visible" in paragraph


def test_punctual_label_becomes_notable_moment_with_time_phrase():
    payload = _payload(
        [
            _label("outdoor", ("0:00:00", "0:00:30", 0.99)),
            _label("kiss", ("0:00:09.8", "0:00:10.6", 0.97)),
        ]
    )
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert "a kiss seems to appear around the 10-second mark" in paragraph
    _assert_reads_naturally(paragraph)


def test_scattered_label_gets_recurring_phrasing_not_a_misleading_span():
    """Two brief, far-apart instances of the same label must not read as
    one continuous span (e.g. "worn from 10s to 27s") -- see
    _SCATTERED_MIN_SPAN_SECONDS in scene_summary.py."""
    payload = _payload(
        [
            _label("outdoor", ("0:00:00", "0:00:30", 0.99)),
            _label(
                "coat",
                ("0:00:10.3", "0:00:10.36", 0.96),
                ("0:00:26.9", "0:00:26.92", 0.96),
            ),
        ]
    )
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert "a coat is glimpsed early on, then again later in the clip" in paragraph


def test_object_bucket_kept_separate_from_notable_moments():
    payload = _payload([_label("outdoor", ("0:00:00", "0:00:30", 0.99))])
    payload["videos"][0]["insights"]["detectedObjects"] = [
        {
            "displayName": "handbag",
            "instances": [{"start": "0:00:02", "end": "0:00:06", "confidence": 0.6}],
        }
    ]
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert "There also seem to be a handbag somewhere in frame" in paragraph


def test_boilerplate_names_excluded_from_setting_and_notable_sentences():
    payload = _payload(
        [
            _label("person", ("0:00:00", "0:00:30", 0.99)),
            _label("clothing", ("0:00:00", "0:00:30", 0.99)),
            _label("human face", ("0:00:00", "0:00:30", 0.99)),
        ]
    )
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert "setting isn't very clear" in paragraph
    # The only "person" mention allowed is the dedicated fallback sentence.
    assert "feature a person" not in paragraph


def test_distinct_people_tracked_mentioned_when_available():
    payload = _payload([_label("outdoor", ("0:00:00", "0:00:30", 0.99))])
    payload["videos"][0]["insights"]["observedPeople"] = [
        {"id": 1, "instances": [{"start": "0:00:00", "end": "0:00:10"}]},
        {"id": 2, "instances": [{"start": "0:00:05", "end": "0:00:20"}]},
    ]
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert "Two people appear to be present across the footage" in paragraph


def test_single_person_uses_singular_grammar():
    payload = _payload([_label("outdoor", ("0:00:00", "0:00:30", 0.99))])
    payload["videos"][0]["insights"]["observedPeople"] = [
        {"id": 1, "instances": [{"start": "0:00:00", "end": "0:00:10"}]},
    ]
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert "One person appears to be present across the footage" in paragraph


def test_person_signal_fallback_when_no_observed_people_data():
    """No observedPeople bucket at all (e.g. Default indexing preset) --
    falls back to noting at least one person seems to appear, rather than
    saying nothing about people."""
    payload = _payload(
        [
            _label("outdoor", ("0:00:00", "0:00:30", 0.99)),
            _label("person", ("0:00:00", "0:00:30", 0.99)),
        ]
    )
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert "At least one person seems to appear" in paragraph


def test_notable_moments_truncated_by_shortest_duration_not_chronologically():
    """With more punctual labels than the display cap, the shortest-lived
    (most punctual) ones are kept -- a brief, highly notable "kiss" must
    not be pushed out by a label that merely happens to start earlier but
    lasts much longer."""
    payload = _payload(
        [
            _label("outdoor", ("0:00:00", "0:00:30", 0.99)),
            _label("tree", ("0:00:00", "0:00:20", 0.96)),  # long-lived, starts first
            _label("jacket", ("0:00:01", "0:00:22", 0.97)),  # long-lived
            _label("man", ("0:00:03", "0:00:24", 0.97)),  # long-lived
            _label("smile", ("0:00:04", "0:00:25", 0.97)),  # long-lived
            _label("hat", ("0:00:05", "0:00:26", 0.95)),  # long-lived
            _label("kiss", ("0:00:09.8", "0:00:10.6", 0.97)),  # brief, starts later
        ]
    )
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert "a kiss seems to appear" in paragraph


def test_paragraph_always_ends_with_hedging_closer():
    payload = _payload([_label("outdoor", ("0:00:00", "0:00:30", 0.99))])
    report = analyze_all(payload, min_confidence=0.75)

    paragraph = summarize_insights(report)

    assert paragraph.endswith(
        "None of this has actually been confirmed by watching the footage, "
        "so it should be treated as an uncertain guess rather than a "
        "reliable account of what happens."
    )
    _assert_reads_naturally(paragraph)


def test_unknown_duration_does_not_crash_and_is_worded_accordingly():
    """With no known video duration, coverage-ratio categorization can't
    tell persistent context from a punctual moment, so everything falls
    into the notable-moments bucket instead -- this just needs to not
    crash and to open with the duration-unknown phrasing."""
    payload = {
        "videos": [
            {"insights": {"labels": [_label("outdoor", ("0:00:00", "0:00:05", 0.99))]}}
        ]
    }
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert paragraph.startswith("This clip's setting isn't very clear.")
    _assert_reads_naturally(paragraph)


def test_paragraph_is_a_single_line_no_newlines():
    """The whole point is one paragraph -- no list-like line breaks."""
    payload = _payload(
        [
            _label("outdoor", ("0:00:00", "0:00:30", 0.99)),
            _label("kiss", ("0:00:09.8", "0:00:10.6", 0.97)),
        ]
    )
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert "\n" not in paragraph


def test_paragraph_multiple_notable_moments_joined_with_and():
    payload = _payload(
        [
            _label("outdoor", ("0:00:00", "0:00:30", 0.99)),
            _label("kiss", ("0:00:09.8", "0:00:10.6", 0.97)),
            _label("wave", ("0:00:15", "0:00:15.5", 0.9)),
        ]
    )
    report = analyze_all(payload)

    paragraph = summarize_insights(report)

    assert "; and a wave seems to appear" in paragraph
