"""
Unit tests for src/insights_explorer.py: extraction of every non-action
insight bucket (faces, transcript/speakers, topics, named entities,
sentiments, emotions, audio effects, shots, content moderation) from a
synthetic Video Indexer payload, including the "appearances" fallback shape
for named entities (see module docstring on why that's the less-confirmed
of the two shapes handled).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.insights_explorer import analyze_capabilities

SAMPLE_INDEX = {
    "videos": [
        {
            "insights": {
                "duration": "0:01:30.0",
                "sourceLanguage": "en-US",
                "faces": [
                    {
                        "name": "Jane Doe",
                        "confidence": 0.91,
                        "thumbnailId": "thumb-1",
                        "instances": [{"start": "0:00:01.0", "end": "0:00:05.0"}],
                    },
                    {
                        "name": "Unknown #1",
                        "instances": [{"start": "0:00:10.0", "end": "0:00:12.0"}],
                    },
                ],
                "speakers": [
                    {
                        "id": 1,
                        "name": "Speaker #1",
                        "instances": [{"start": "0:00:00.0", "end": "0:00:20.0"}],
                    }
                ],
                "transcript": [
                    {
                        "text": "Welcome to the demo.",
                        "confidence": 0.88,
                        "speakerId": 1,
                        "language": "en-US",
                        "instances": [{"start": "0:00:00.5", "end": "0:00:03.0"}],
                    },
                    {
                        "text": "No speaker id here.",
                        "instances": [{"start": "0:00:04.0", "end": "0:00:06.0"}],
                    },
                ],
                "topics": [
                    {
                        "name": "Technology",
                        "iptcName": "technology and computing",
                        "confidence": 0.75,
                        "instances": [{"start": "0:00:00.0", "end": "0:01:30.0"}],
                    }
                ],
                "brands": [
                    {
                        "name": "Contoso",
                        "confidence": 0.8,
                        "instances": [{"start": "0:00:02.0", "end": "0:00:03.0"}],
                    }
                ],
                "namedLocations": [
                    {
                        "name": "Seattle",
                        "confidence": 0.7,
                        "instances": [{"start": "0:00:05.0", "end": "0:00:06.0"}],
                    }
                ],
                "namedPeople": [
                    {
                        "name": "Satya Nadella",
                        "confidence": 0.65,
                        "appearances": [
                            {"startSeconds": 10.0, "endSeconds": 12.5},
                        ],
                    }
                ],
                "sentiments": [
                    {
                        "sentimentType": "Positive",
                        "averageScore": 0.6,
                        "instances": [{"start": "0:00:00.0", "end": "0:00:20.0"}],
                    }
                ],
                "emotions": [
                    {
                        "type": "Joy",
                        "instances": [{"start": "0:00:01.0", "end": "0:00:02.0"}],
                    }
                ],
                "audioEffects": [
                    {
                        "type": "Music",
                        "confidence": 0.55,
                        "instances": [{"start": "0:00:00.0", "end": "0:00:10.0"}],
                    }
                ],
                "shots": [
                    {
                        "keyFrames": [{"id": 1}, {"id": 2}],
                        "instances": [{"start": "0:00:00.0", "end": "0:00:15.0"}],
                    },
                    {
                        "keyFrames": [{"id": 3}],
                        "instances": [{"start": "0:00:15.0", "end": "0:00:30.0"}],
                    },
                ],
                "visualContentModeration": [{"adultScore": 0.01, "racyScore": 0.02}],
                "textualContentModeration": [
                    {"bannedWordsCount": 0, "bannedWordsRatio": 0.0}
                ],
            }
        }
    ]
}


def test_video_duration_and_language():
    report = analyze_capabilities(SAMPLE_INDEX)
    assert report.video_duration_seconds == 90.0
    assert report.source_language == "en-US"


def test_faces_extracted_with_confidence_and_appearances():
    report = analyze_capabilities(SAMPLE_INDEX)
    assert len(report.faces) == 2
    jane = next(f for f in report.faces if f.name == "Jane Doe")
    assert jane.confidence == 0.91
    assert jane.thumbnail_id == "thumb-1"
    assert len(jane.appearances) == 1
    assert jane.appearances[0].start_seconds == 1.0
    assert jane.appearances[0].end_seconds == 5.0


def test_transcript_resolves_speaker_name_and_handles_missing_speaker():
    report = analyze_capabilities(SAMPLE_INDEX)
    assert len(report.transcript) == 2
    first, second = report.transcript
    assert first.text == "Welcome to the demo."
    assert first.speaker == "Speaker #1"
    assert first.language == "en-US"
    assert second.speaker is None


def test_topics_include_category_from_iptc_name():
    report = analyze_capabilities(SAMPLE_INDEX)
    assert len(report.topics) == 1
    assert report.topics[0].name == "Technology"
    assert report.topics[0].category == "technology and computing"


def test_named_entities_merge_all_three_buckets_with_kind():
    report = analyze_capabilities(SAMPLE_INDEX)
    by_kind = {e.kind: e for e in report.named_entities}
    assert by_kind["brand"].name == "Contoso"
    assert by_kind["location"].name == "Seattle"
    assert by_kind["person"].name == "Satya Nadella"


def test_named_entities_fall_back_to_appearances_shape():
    report = analyze_capabilities(SAMPLE_INDEX)
    person = next(e for e in report.named_entities if e.kind == "person")
    assert len(person.appearances) == 1
    assert person.appearances[0].start_seconds == 10.0
    assert person.appearances[0].end_seconds == 12.5


def test_sentiments_use_average_score_as_confidence_like_value():
    report = analyze_capabilities(SAMPLE_INDEX)
    assert len(report.sentiments) == 1
    assert report.sentiments[0].sentiment_type == "Positive"
    assert report.sentiments[0].score == 0.6


def test_emotions_extracted():
    report = analyze_capabilities(SAMPLE_INDEX)
    assert len(report.emotions) == 1
    assert report.emotions[0].emotion_type == "Joy"


def test_audio_effects_extracted():
    report = analyze_capabilities(SAMPLE_INDEX)
    assert len(report.audio_effects) == 1
    assert report.audio_effects[0].name == "Music"
    assert report.audio_effects[0].confidence == 0.55


def test_shots_include_keyframe_counts():
    report = analyze_capabilities(SAMPLE_INDEX)
    assert len(report.shots) == 2
    assert report.shots[0].keyframe_count == 2
    assert report.shots[1].keyframe_count == 1
    assert report.shots[0].start_seconds == 0.0
    assert report.shots[0].end_seconds == 15.0


def test_content_moderation_extracted():
    report = analyze_capabilities(SAMPLE_INDEX)
    assert report.content_moderation is not None
    assert report.content_moderation.visual_adult_score == 0.01
    assert report.content_moderation.textual_banned_words_count == 0
    assert report.content_moderation.has_any_signal


def test_coverage_counts_match_each_bucket():
    report = analyze_capabilities(SAMPLE_INDEX)
    coverage = report.coverage
    assert coverage["faces"] == 2
    assert coverage["transcript_lines"] == 2
    assert coverage["speakers"] == 1
    assert coverage["topics"] == 1
    assert coverage["named_entities"] == 3
    assert coverage["sentiments"] == 1
    assert coverage["emotions"] == 1
    assert coverage["audio_effects"] == 1
    assert coverage["shots"] == 2
    assert coverage["content_moderation_signals"] == 1


def test_empty_payload_returns_empty_report_not_an_error():
    report = analyze_capabilities({"videos": [{"insights": {}}]})
    assert report.faces == []
    assert report.transcript == []
    assert report.named_entities == []
    assert report.content_moderation is None
    assert all(count == 0 for count in report.coverage.values())
