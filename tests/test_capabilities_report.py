"""
Unit tests for src/capabilities_report.py: to_console_text_capabilities,
to_dict_capabilities/write_json_capabilities, and write_html_capabilities.

Built on synthetic CapabilitiesReport instances (no raw Video Indexer
payload needed) covering every section renderer individually, plus a
report with nothing at all so the "nothing found" fallback in both the
console and HTML renderers gets exercised too. This is also regression
coverage for the section-per-capability refactor of
to_console_text_capabilities/write_html_capabilities: each is now a flat
dispatch over one small helper per capability instead of one long
function, and these tests pin the rendered text/HTML so a future edit to
any of those helpers gets caught here rather than only via manual
smoke-testing against real data.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capabilities_report import (
    to_console_text_capabilities,
    to_dict_capabilities,
    write_html_capabilities,
    write_json_capabilities,
)
from src.insights_explorer import (
    Appearance,
    AudioEffectInsight,
    CapabilitiesReport,
    ClothingItem,
    ContentModerationInsight,
    EmotionInsight,
    FaceInsight,
    InteractionSignal,
    NamedEntity,
    ObservedPersonInsight,
    SentimentInsight,
    ShotInsight,
    SpeakerInsight,
    TopicInsight,
    TranscriptLine,
)

EMPTY_REPORT = CapabilitiesReport(video_duration_seconds=30.0, source_language="en-US")

FULL_REPORT = CapabilitiesReport(
    video_duration_seconds=29.6,
    source_language="en-US",
    faces=[
        FaceInsight(
            name="Unknown #1",
            confidence=0.0,
            thumbnail_id="thumb-1",
            appearances=[Appearance(start_seconds=0.0, end_seconds=26.84)],
        ),
        FaceInsight(
            name="Jane Doe",
            confidence=0.91,
            thumbnail_id="thumb-2",
            appearances=[Appearance(start_seconds=1.0, end_seconds=2.0)],
        ),
    ],
    transcript=[
        TranscriptLine(
            text="Hello there.",
            speaker="Speaker #1",
            language="en-US",
            confidence=0.95,
            start_seconds=0.0,
            end_seconds=1.5,
        ),
        TranscriptLine(
            text="No speaker on this line.",
            speaker=None,
            language="en-US",
            confidence=0.8,
            start_seconds=2.0,
            end_seconds=3.0,
        ),
    ],
    speakers=[SpeakerInsight(name="Speaker #1", appearances=[Appearance(0.0, 1.5)])],
    topics=[
        TopicInsight(
            name="Fashion", category="Lifestyle", confidence=0.7, appearances=[]
        ),
        TopicInsight(
            name="Untagged Topic", category=None, confidence=None, appearances=[]
        ),
    ],
    named_entities=[
        NamedEntity(name="Acme Corp", kind="brand", confidence=0.8, appearances=[]),
    ],
    sentiments=[
        SentimentInsight(
            sentiment_type="Positive", score=0.6, appearances=[Appearance(0.0, 5.0)]
        ),
    ],
    emotions=[
        EmotionInsight(emotion_type="Joy", appearances=[Appearance(0.0, 2.0)]),
    ],
    audio_effects=[
        AudioEffectInsight(
            name="Music", confidence=0.99, appearances=[Appearance(0.0, 10.0)]
        ),
    ],
    shots=[
        ShotInsight(index=0, start_seconds=0.0, end_seconds=29.6, keyframe_count=3),
    ],
    content_moderation=ContentModerationInsight(
        visual_adult_score=0.01,
        visual_racy_score=None,
        textual_banned_words_count=0,
        textual_banned_words_ratio=0.0,
    ),
    observed_people=[
        ObservedPersonInsight(
            person_id=1,
            thumbnail_id="p1",
            clothing=[ClothingItem(type="sleeve", length="long")],
            matched_face_name="Unknown #1",
            matched_face_confidence=1.0,
            appearances=[Appearance(0.0, 25.88)],
        ),
        ObservedPersonInsight(
            person_id=2,
            thumbnail_id="p2",
            clothing=[],
            matched_face_name=None,
            matched_face_confidence=None,
            appearances=[Appearance(0.0, 2.2)],
        ),
    ],
    interaction_signals=[
        InteractionSignal(
            name="kiss",
            confidence=0.97,
            appearances=[Appearance(9.84, 9.88), Appearance(10.56, 10.6)],
        ),
    ],
)


# ----------------------------------------------------------------------
# to_console_text_capabilities
# ----------------------------------------------------------------------


def test_console_empty_report_shows_coverage_zeros_and_fallback_notice():
    text = to_console_text_capabilities(EMPTY_REPORT, "empty_video")

    assert "Capabilities report for 'empty_video'" in text
    assert "Faces                                      0" in text
    assert "Nothing found in any of these categories." in text
    # None of the per-capability section headers should appear at all.
    assert "Faces (" not in text
    assert "Transcript (" not in text


def test_console_faces_section_lists_each_face_with_confidence_and_timing():
    text = to_console_text_capabilities(FULL_REPORT, "video")

    assert "Faces (2, 1 recognized):" in text
    assert "Unknown #1  confidence=0.00  appearances=1 (00:00.00-00:26.84)" in text
    assert (
        "Jane Doe [recognized]  confidence=0.91  appearances=1 (00:01.00-00:02.00)"
        in text
    )


def test_console_observed_people_shows_matched_face_only_when_present():
    """Regression test for the nested-conditional-expression fix: person #1
    has a matched face (with a confidence suffix), person #2 has none --
    the trailing "matched face: ..." text must only appear for #1."""
    text = to_console_text_capabilities(FULL_REPORT, "video")

    assert (
        "Person #1  seen=00:25.88  clothing: sleeve (long)  matched face: Unknown #1 (confidence=1.00)"
        in text
    )
    assert "Person #2  seen=00:02.20  clothing: no clothing data" in text
    person_2_line = next(
        line for line in text.splitlines() if line.startswith("  Person #2")
    )
    assert "matched face" not in person_2_line


def test_console_interaction_signals_include_caveat_and_timing():
    text = to_console_text_capabilities(FULL_REPORT, "video")

    assert "Interaction signals (1):" in text
    assert "not a dedicated interaction detector" in text
    assert "kiss  confidence=0.97  (00:09.84-00:09.88, 00:10.56-00:10.60)" in text


def test_console_transcript_shows_speaker_when_present_and_omits_when_absent():
    text = to_console_text_capabilities(FULL_REPORT, "video")

    assert "[00:00.00-00:01.50] Speaker #1: Hello there." in text
    assert "[00:02.00-00:03.00] No speaker on this line." in text


def test_console_topics_named_entities_sentiments_emotions_audio_shots():
    text = to_console_text_capabilities(FULL_REPORT, "video")

    assert "Fashion [Lifestyle]  confidence=0.70" in text
    assert "Untagged Topic  confidence=n/a" in text
    assert "[brand] Acme Corp  confidence=0.80  appearances=0" in text
    assert "Positive  score=0.60  appearances=1" in text
    assert "Joy  appearances=1" in text
    assert "Music  confidence=0.99  appearances=1" in text
    assert "#0  [00:00.00-00:29.60]  keyframes=3" in text


def test_console_content_moderation_only_shows_signals_that_are_set():
    text = to_console_text_capabilities(FULL_REPORT, "video")

    assert "visual adult score:  0.010" in text
    assert "visual racy score" not in text  # None -- not shown
    assert "banned words count:  0" in text
    assert "banned words ratio:  0.000" in text


def test_console_no_content_moderation_signal_omits_section():
    report = CapabilitiesReport(
        video_duration_seconds=10.0,
        source_language="en-US",
        content_moderation=ContentModerationInsight(),  # has_any_signal is False
    )

    text = to_console_text_capabilities(report, "video")

    assert "Content moderation:" not in text


# ----------------------------------------------------------------------
# to_dict_capabilities / write_json_capabilities
# ----------------------------------------------------------------------


def test_to_dict_capabilities_round_trips_through_json():
    data = to_dict_capabilities(FULL_REPORT, "video")

    # JSON-serializable, and shaped the way explore_insights.py's consumers
    # expect: one top-level list per capability, plain dicts inside.
    encoded = json.dumps(data)
    decoded = json.loads(encoded)
    assert decoded["video_name"] == "video"
    assert len(decoded["faces"]) == 2
    assert decoded["faces"][0]["name"] == "Unknown #1"
    assert decoded["observed_people"][0]["matched_face_name"] == "Unknown #1"
    assert decoded["observed_people"][1]["matched_face_name"] is None
    assert decoded["content_moderation"]["visual_adult_score"] == 0.01


def test_to_dict_capabilities_content_moderation_none_when_absent():
    data = to_dict_capabilities(EMPTY_REPORT, "video")
    assert data["content_moderation"] is None
    assert data["faces"] == []


def test_write_json_capabilities_writes_pretty_printed_file(tmp_path: Path):
    out_path = tmp_path / "report.json"

    write_json_capabilities(FULL_REPORT, "video", out_path)

    assert out_path.is_file()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["video_name"] == "video"
    assert "\n" in out_path.read_text(encoding="utf-8")  # pretty-printed, not one line


# ----------------------------------------------------------------------
# write_html_capabilities
# ----------------------------------------------------------------------


def test_write_html_capabilities_empty_report_shows_empty_notice(tmp_path: Path):
    out_path = tmp_path / "empty.html"

    write_html_capabilities(EMPTY_REPORT, "empty_video", out_path)

    html = out_path.read_text(encoding="utf-8")
    assert "Capabilities detected in empty_video" in html
    assert "Nothing found in any of these categories." in html
    assert "<h2>Faces" not in html


def test_write_html_capabilities_renders_every_section(tmp_path: Path):
    out_path = tmp_path / "full.html"

    write_html_capabilities(FULL_REPORT, "video", out_path)

    html = out_path.read_text(encoding="utf-8")
    assert "<h2>Faces (2, 1 recognized)</h2>" in html
    assert "<h2>Observed people / body tracking (2)</h2>" in html
    assert "<h2>Interaction signals (1)</h2>" in html
    assert "<h2>Transcript (2 lines)</h2>" in html
    assert "<h2>Topics (2)</h2>" in html
    assert "<h2>Named entities (1)</h2>" in html
    assert "<h2>Sentiments (1)</h2>" in html
    assert "<h2>Emotions (1)</h2>" in html
    assert "<h2>Audio effects (1)</h2>" in html
    assert "<h2>Shots (1)</h2>" in html
    assert "<h2>Content moderation</h2>" in html
    assert "Nothing found in any of these categories." not in html


def test_write_html_capabilities_observed_people_matched_face_dash_when_absent(
    tmp_path: Path,
):
    out_path = tmp_path / "full.html"

    write_html_capabilities(FULL_REPORT, "video", out_path)

    html = out_path.read_text(encoding="utf-8")
    assert "<td>Unknown #1</td>" in html  # person #1's matched face name
    # person #2 has neither clothing data nor a matched face -- both cells
    # render as a bare dash. Anchor on the section's own <h2>, not the
    # identically-worded coverage-stat label earlier in the page.
    section_start = html.index("<h2>Observed people")
    section_end = html.index("<h2>Interaction signals")
    rows_html = html[section_start:section_end]
    assert rows_html.count("<td>-</td>") == 2


def test_write_html_capabilities_escapes_untrusted_text(tmp_path: Path):
    """Names/text drawn from Video Indexer output are untrusted strings --
    confirm they're HTML-escaped, not interpolated raw."""
    report = CapabilitiesReport(
        video_duration_seconds=5.0,
        source_language="en-US",
        faces=[
            FaceInsight(
                name="<script>alert(1)</script>",
                confidence=0.5,
                thumbnail_id=None,
                appearances=[],
            )
        ],
    )
    out_path = tmp_path / "escaped.html"

    write_html_capabilities(report, "video", out_path)

    html = out_path.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
