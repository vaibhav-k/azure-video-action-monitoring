"""
Unit tests for action_analyzer / report using a realistic sample Video
Indexer insights payload, so the extraction and reporting logic can be
verified end-to-end without needing a live Azure Video Indexer account.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.action_analyzer import (
    _looks_like_ocr_overlay,
    _observed_people_intervals,
    _people_in_frame_count,
    _temporal_iou,
    _video_dimensions,
    analyze,
    analyze_all,
)
from src.report import to_console_text, to_dict, write_html, write_json

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_insights_jumping.json"


@pytest.fixture()
def sample_index() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_analyze_finds_all_jumping_label_instances(sample_index: dict[str, Any]):
    report = analyze(sample_index, action="jumping")

    assert report.occurrence_count == 3
    assert report.video_duration_seconds == pytest.approx(42.5)
    assert report.distinct_people_tracked == 2
    assert report.first_occurrence_seconds == pytest.approx(3.2)
    assert report.last_occurrence_seconds == pytest.approx(28.4)

    # Events must be sorted chronologically.
    starts = [e.start_seconds for e in report.events]
    assert starts == sorted(starts)

    # Confidence values should be carried through from the fixture.
    assert all(e.confidence is not None for e in report.events)


def test_analyze_ignores_unrelated_labels(sample_index: dict[str, Any]):
    report = analyze(sample_index, action="jumping")
    matched_names = {e.matched_term for e in report.events}
    assert "person" not in matched_names
    assert "outdoor" not in matched_names


def test_analyze_supports_extra_synonyms(sample_index: dict[str, Any]):
    # "trampoline" only appears in keywords and wouldn't match "jump" by
    # default; verify a custom synonym picks it up.
    report = analyze(sample_index, action="jumping", extra_synonyms=["trampoline"])
    sources = {e.source for e in report.events}
    assert "keywords" in sources


def test_analyze_missing_action_returns_empty_report():
    empty_index: dict[str, Any] = {
        "videos": [{"insights": {"labels": [], "keywords": []}}]
    }
    report = analyze(empty_index, action="jumping")
    assert report.occurrence_count == 0
    assert report.total_action_seconds == 0.0
    assert report.first_occurrence_seconds is None


def test_analyze_cash_matches_currency_ocr_text():
    # "cash" has no visual-concept vocabulary to match (no label/keyword/
    # object class for money) -- it only matches via `ocr`, against phrases
    # that are literally printed on US currency. A bare denomination number
    # ("100") deliberately isn't a match target (too ambiguous alone); a
    # motto fragment like "WE TRUST" is.
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [],
                    "keywords": [],
                    "ocr": [
                        {
                            "text": "WE TRUST",
                            "confidence": 0.998,
                            "instances": [{"start": "0:00:00.08", "end": "0:00:00.6"}],
                        },
                        {
                            "text": "100",
                            "confidence": 0.996,
                            "instances": [{"start": "0:00:00.08", "end": "0:00:03"}],
                        },
                    ],
                }
            }
        ]
    }
    report = analyze(index, action="cash")

    matched_names = {e.matched_term for e in report.events}
    assert matched_names == {"WE TRUST"}
    assert report.events[0].source == "ocr"


def test_analyze_phone_matches_derived_person_using_phone_overlap():
    # "phone" matches the "cell phone" object directly (substring match,
    # same as any other action search) *and* the composite "person using
    # phone" derived from that same object overlapping a person label --
    # both are legitimate, independent hits for "--action phone".
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        }
                    ],
                    "keywords": [],
                    "detectedObjects": [
                        {
                            "displayName": "cell phone",
                            "instances": [
                                {
                                    "confidence": 0.8,
                                    "start": "0:00:03",
                                    "end": "0:00:06",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }
    report = analyze(index, action="phone")

    assert report.occurrence_count == 2
    matched = {(e.matched_term, e.source) for e in report.events}
    assert matched == {
        ("cell phone", "objects"),
        ("person using phone", "derived"),
    }


def test_analyze_min_overlap_seconds_filters_short_composite_overlap():
    # Same shape as test_analyze_phone_matches_derived_person_using_phone_overlap,
    # but the overlap is a 0.02s sliver -- min_overlap_seconds should drop
    # the derived match while leaving the direct "cell phone" object match
    # untouched (that one isn't a composite, so the threshold doesn't apply
    # to it).
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        }
                    ],
                    "keywords": [],
                    "detectedObjects": [
                        {
                            "displayName": "cell phone",
                            "instances": [
                                {
                                    "confidence": 0.8,
                                    "start": "0:00:03",
                                    "end": "0:00:03.02",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }
    report = analyze(index, action="phone", min_overlap_seconds=0.15)

    matched = {(e.matched_term, e.source) for e in report.events}
    assert matched == {("cell phone", "objects")}


def test_analyze_register_matches_derived_person_at_register_overlap():
    # "person at register" is the weakest-evidence composite (any
    # keyboard/monitor-like object near a person), added for a checkout/
    # cashier monitoring use case -- verify it derives the same way as the
    # phone composite.
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 0.9,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        }
                    ],
                    "keywords": [],
                    "detectedObjects": [
                        {
                            "displayName": "keyboard",
                            "instances": [
                                {
                                    "confidence": 0.6,
                                    "start": "0:00:02",
                                    "end": "0:00:05",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }
    report = analyze(index, action="register")

    matched_names = {e.matched_term for e in report.events}
    assert matched_names == {"person at register"}
    assert report.events[0].confidence == pytest.approx(0.6)  # min(0.9, 0.6)


def test_analyze_merge_gap_seconds_merges_fragmented_matches():
    # Two "jumping" label instances 0.3s apart -- close enough to collapse
    # into one occurrence with --merge-gap-seconds, same fragmentation
    # cleanup analyze_all() already had (this is the parity fix for
    # analyze()/main.py, which previously had no way to do this at all).
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "jumping",
                            "instances": [
                                {
                                    "confidence": 0.7,
                                    "start": "0:00:01",
                                    "end": "0:00:02",
                                },
                                {
                                    "confidence": 0.9,
                                    "start": "0:00:02.3",
                                    "end": "0:00:03",
                                },
                            ],
                        }
                    ],
                    "keywords": [],
                }
            }
        ]
    }

    unmerged = analyze(index, action="jumping")
    assert unmerged.occurrence_count == 2

    merged = analyze(index, action="jumping", merge_gap_seconds=0.5)
    assert merged.occurrence_count == 1
    assert merged.events[0].start_seconds == pytest.approx(1.0)
    assert merged.events[0].end_seconds == pytest.approx(3.0)
    assert merged.events[0].confidence == pytest.approx(0.9)  # max of the two

    # A gap tighter than the actual 0.3s gap between them must not merge.
    not_merged = analyze(index, action="jumping", merge_gap_seconds=0.1)
    assert not_merged.occurrence_count == 2


def test_playing_sports_matches_real_detected_object_class():
    # "sports ball" is a real detectedObjects class (Video Indexer's fixed
    # 80-class vocabulary) -- exact-name match, no guessing involved.
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        }
                    ],
                    "keywords": [],
                    "detectedObjects": [
                        {
                            "displayName": "sports ball",
                            "instances": [
                                {
                                    "confidence": 0.8,
                                    "start": "0:00:02",
                                    "end": "0:00:05",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }
    report = analyze(index, action="playing sports")
    assert {e.matched_term for e in report.events} == {"person playing sports"}


def test_playing_sports_matches_via_synonym_fragment_not_exact_name():
    # "baseball bat" isn't a real detectedObjects class and isn't in this
    # composite's exact `names` set either -- it only matches because "bat"
    # is a substring `synonym`, which is the whole point of moving guessed
    # word-fragments out of exact-name matching.
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        },
                        {
                            "name": "baseball bat",
                            "instances": [
                                {
                                    "confidence": 0.6,
                                    "start": "0:00:02",
                                    "end": "0:00:05",
                                }
                            ],
                        },
                    ],
                    "keywords": [],
                }
            }
        ]
    }
    report = analyze(index, action="playing sports")
    assert {e.matched_term for e in report.events} == {"person playing sports"}


def test_driving_car_no_longer_matches_bare_vehicle_label():
    # Regression lock-in: "vehicle"/"outdoor vehicle" used to be (wrongly)
    # exact-matched here despite not being real detectedObjects classes or
    # confirmed labels -- they were dropped rather than guessed back in, so
    # a bare "vehicle" label must NOT derive "person driving car" anymore.
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        },
                        {
                            "name": "vehicle",
                            "instances": [
                                {
                                    "confidence": 0.6,
                                    "start": "0:00:02",
                                    "end": "0:00:05",
                                }
                            ],
                        },
                    ],
                    "keywords": [],
                }
            }
        ]
    }
    report = analyze(index, action="driving car")
    assert report.events == []


def test_register_matches_real_computer_mouse_class_and_synonym_fragment():
    # "computer mouse" is a real detectedObjects class (exact match);
    # "television" isn't in the fixed vocabulary at all, so it only matches
    # via the "tv"/"television" substring synonyms.
    index: dict[str, Any] = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        },
                        {
                            "name": "television",
                            "instances": [
                                {
                                    "confidence": 0.5,
                                    "start": "0:00:06",
                                    "end": "0:00:08",
                                }
                            ],
                        },
                    ],
                    "keywords": [],
                    "detectedObjects": [
                        {
                            "displayName": "computer mouse",
                            "instances": [
                                {
                                    "confidence": 0.7,
                                    "start": "0:00:01",
                                    "end": "0:00:03",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }
    report = analyze(index, action="register")
    matched_spans = {(e.start_seconds, e.end_seconds) for e in report.events}
    assert matched_spans == {(1.0, 3.0), (6.0, 8.0)}
    assert all(e.matched_term == "person at register" for e in report.events)


def _index_with_person_and_object(displayName: str) -> dict[str, Any]:
    return {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        }
                    ],
                    "keywords": [],
                    "detectedObjects": [
                        {
                            "displayName": displayName,
                            "instances": [
                                {
                                    "confidence": 0.7,
                                    "start": "0:00:02",
                                    "end": "0:00:05",
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }


@pytest.mark.parametrize("bag_class", ["backpack", "handbag", "suitcase"])
def test_carrying_bag_matches_real_bag_classes(bag_class: str):
    index = _index_with_person_and_object(bag_class)
    report = analyze(index, action="carrying bag")
    assert {e.matched_term for e in report.events} == {"person carrying bag"}


def test_reading_matches_real_book_class():
    index = _index_with_person_and_object("book")
    report = analyze(index, action="reading")
    assert {e.matched_term for e in report.events} == {"person reading"}


def test_near_knife_matches_real_class():
    # "knife" is both the raw object's own name and a substring of the
    # composite's name, so --action "knife" legitimately matches both
    # (same pattern as --action "phone" matching "cell phone" directly
    # *and* the derived "person using phone").
    index = _index_with_person_and_object("knife")
    report = analyze(index, action="knife")
    matched = {(e.matched_term, e.source) for e in report.events}
    assert matched == {
        ("knife", "objects"),
        ("person near knife or scissors", "derived"),
    }


def test_near_scissors_matches_real_class():
    index = _index_with_person_and_object("scissors")
    report = analyze(index, action="near knife or scissors")
    assert {e.matched_term for e in report.events} == {"person near knife or scissors"}


def test_carrying_bag_derives_against_multiple_person_and_bag_instances():
    """Regression test for a real detect_all_actions.py run against multiple
    person spans and multiple handbag instances (people_carrying_bags.mp4):
    "person carrying bag" must derive the same way the structurally
    identical "person driving car" composite does, rather than silently
    producing zero rows just because there's more than one instance on
    either side."""
    index = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00.07",
                                    "end": "0:00:00.10",
                                },
                                {
                                    "confidence": 0.99,
                                    "start": "0:00:03.07",
                                    "end": "0:00:07.10",
                                },
                            ],
                        }
                    ],
                    "keywords": [],
                    "detectedObjects": [
                        {
                            "displayName": "handbag",
                            "instances": [
                                {
                                    "confidence": 0.60,
                                    "start": "0:00:02.43",
                                    "end": "0:00:02.80",
                                },
                                {
                                    "confidence": 0.47,
                                    "start": "0:00:02.43",
                                    "end": "0:00:02.80",
                                },
                                {
                                    "confidence": 0.78,
                                    "start": "0:00:04.53",
                                    "end": "0:00:06.07",
                                },
                                {
                                    "confidence": 0.34,
                                    "start": "0:00:06.87",
                                    "end": "0:00:07.73",
                                },
                                {
                                    "confidence": 0.47,
                                    "start": "0:00:07.53",
                                    "end": "0:00:08.67",
                                },
                            ],
                        },
                        {
                            "displayName": "car",
                            "instances": [
                                {
                                    "confidence": 0.72,
                                    "start": "0:00:04.37",
                                    "end": "0:00:04.90",
                                }
                            ],
                        },
                    ],
                }
            }
        ]
    }

    report = analyze_all(index, min_overlap_seconds=0.15)
    derived_names = {a.name for a in report.actions if a.source == "derived"}
    assert "person driving car" in derived_names
    assert "person carrying bag" in derived_names


def test_video_dimensions_reads_from_video_object_not_insights():
    # Video Indexer reports width/height on the video object itself (a
    # sibling of "insights"), not inside insights -- confirmed against this
    # project's own raw insights captures.
    index = {"videos": [{"width": 320, "height": 240, "insights": {}}]}
    assert _video_dimensions(index) == (320.0, 240.0)


def test_video_dimensions_none_when_missing_or_invalid():
    assert _video_dimensions({"videos": [{"insights": {}}]}) is None
    assert _video_dimensions({"videos": [{"width": 0, "height": 240}]}) is None
    assert _video_dimensions({}) is None


def test_looks_like_ocr_overlay_true_for_small_corner_box():
    # Same box shape as the "17 14 07" camera-clock overlay literally
    # captured in this project's cashier.raw_insights.json (320x240 frame,
    # a small box near the top-right corner).
    assert _looks_like_ocr_overlay(
        ocr_box=(233.0, 39.0, 28.0, 7.0), frame_width=320.0, frame_height=240.0
    )


def test_looks_like_ocr_overlay_false_for_centered_box():
    assert not _looks_like_ocr_overlay(
        ocr_box=(140.0, 110.0, 40.0, 20.0), frame_width=320.0, frame_height=240.0
    )


def test_looks_like_ocr_overlay_false_without_frame_dimensions():
    # Can't judge relative position without knowing the frame size -- the
    # heuristic must disable itself rather than guess.
    assert not _looks_like_ocr_overlay(
        ocr_box=(233.0, 39.0, 28.0, 7.0), frame_width=None, frame_height=None
    )


def test_person_handling_cash_suppressed_for_overlay_positioned_ocr_text():
    # Same OCR box as the real camera-clock overlay in
    # cashier.raw_insights.json, but with cash-matching text -- this is
    # exactly the false-positive source the overlay heuristic exists to
    # catch: text that reads like currency but sits where a burned-in
    # timestamp would.
    index: dict[str, Any] = {
        "videos": [
            {
                "width": 320,
                "height": 240,
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        }
                    ],
                    "keywords": [],
                    "ocr": [
                        {
                            "text": "WE TRUST",
                            "confidence": 0.99,
                            "left": 233,
                            "top": 39,
                            "width": 28,
                            "height": 7,
                            "instances": [{"start": "0:00:02", "end": "0:00:04"}],
                        }
                    ],
                },
            }
        ]
    }
    report = analyze(index, action="cash")
    matched = {e.matched_term for e in report.events}
    # The raw OCR hit is still reported -- only the derived composite is
    # suppressed.
    assert "WE TRUST" in matched
    assert "person handling cash" not in matched


def test_person_handling_cash_still_matches_centered_ocr_text():
    # Same shape, but the OCR box is centered rather than corner-positioned
    # -- must NOT be treated as an overlay, so the composite still fires.
    index: dict[str, Any] = {
        "videos": [
            {
                "width": 320,
                "height": 240,
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:10",
                                }
                            ],
                        }
                    ],
                    "keywords": [],
                    "ocr": [
                        {
                            "text": "WE TRUST",
                            "confidence": 0.99,
                            "left": 140,
                            "top": 110,
                            "width": 40,
                            "height": 20,
                            "instances": [{"start": "0:00:02", "end": "0:00:04"}],
                        }
                    ],
                },
            }
        ]
    }
    report = analyze(index, action="cash")
    matched = {e.matched_term for e in report.events}
    assert "person handling cash" in matched


def test_temporal_iou_full_overlap_is_one():
    assert _temporal_iou(0.0, 10.0, 0.0, 10.0) == pytest.approx(1.0)


def test_temporal_iou_low_for_long_span_against_brief_blip():
    # A person label spanning the whole video against a one-second object
    # blip -- high raw overlap duration is possible, but the two durations
    # barely coincide, which is exactly what a low temporal IoU should
    # reflect (a time-domain proxy for "these detections do NOT describe
    # the same event," not a spatial judgment).
    iou = _temporal_iou(0.0, 100.0, 10.0, 11.0)
    assert iou == pytest.approx(1.0 / 100.0)


def test_temporal_iou_zero_when_no_overlap():
    assert _temporal_iou(0.0, 5.0, 10.0, 15.0) == 0.0


def test_observed_people_intervals_none_when_bucket_absent():
    assert _observed_people_intervals({}) is None
    assert _observed_people_intervals({"observedPeople": []}) is None


def test_observed_people_intervals_one_list_per_distinct_person():
    insights = {
        "observedPeople": [
            {"instances": [{"start": "0:00:00", "end": "0:00:05"}]},
            {
                "instances": [
                    {"start": "0:00:03", "end": "0:00:04"},
                    {"start": "0:00:06", "end": "0:00:08"},
                ]
            },
        ]
    }
    intervals = _observed_people_intervals(insights)
    assert intervals == [[(0.0, 5.0)], [(3.0, 4.0), (6.0, 8.0)]]


def test_people_in_frame_count_none_without_data():
    assert _people_in_frame_count(None, 0.0, 5.0) is None


def test_people_in_frame_count_counts_distinct_overlapping_people():
    intervals = [[(0.0, 5.0)], [(3.0, 4.0), (6.0, 8.0)], [(20.0, 21.0)]]
    # Window 3.5-3.8 overlaps person 0 (0-5) and person 1's first span
    # (3-4), but not person 2 (20-21) -- 2 distinct people in frame.
    assert _people_in_frame_count(intervals, 3.5, 3.8) == 2
    # Window 6.5-7 only overlaps person 1's second span.
    assert _people_in_frame_count(intervals, 6.5, 7.0) == 1
    # Window 100-101 overlaps nobody.
    assert _people_in_frame_count(intervals, 100.0, 101.0) == 0


def _index_with_person_and_phone(
    *, observed_people: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    insights: dict[str, Any] = {
        "labels": [
            {
                "name": "person",
                "instances": [
                    {"confidence": 1.0, "start": "0:00:00", "end": "0:00:10"}
                ],
            }
        ],
        "keywords": [],
        "detectedObjects": [
            {
                "displayName": "cell phone",
                "instances": [
                    {"confidence": 0.8, "start": "0:00:03", "end": "0:00:06"}
                ],
            }
        ],
    }
    if observed_people is not None:
        insights["observedPeople"] = observed_people
    return {"videos": [{"insights": insights}]}


def test_min_temporal_iou_drops_low_iou_composite():
    # Person spans 0-10s, phone spans 3-6s -- overlap is the full phone
    # span (3s) but the union is the full person span (10s), so IoU is
    # 3/10 = 0.3. A threshold above that must drop it; below (or equal)
    # must keep it.
    index = _index_with_person_and_phone()

    kept = analyze(index, action="phone", min_temporal_iou=0.3)
    assert {(e.matched_term, e.source) for e in kept.events} == {
        ("cell phone", "objects"),
        ("person using phone", "derived"),
    }

    dropped = analyze(index, action="phone", min_temporal_iou=0.5)
    matched = {(e.matched_term, e.source) for e in dropped.events}
    assert ("person using phone", "derived") not in matched
    assert ("cell phone", "objects") in matched  # non-composite match unaffected


def test_composite_evidence_includes_people_in_frame_count_when_available():
    index = _index_with_person_and_phone(
        observed_people=[{"instances": [{"start": "0:00:00", "end": "0:00:10"}]}]
    )
    report = analyze(index, action="phone")
    derived = next(e for e in report.events if e.source == "derived")
    assert derived.evidence is not None
    assert "1 person(s) in frame" in derived.evidence


def test_composite_evidence_omits_people_in_frame_when_no_observed_people_data():
    index = _index_with_person_and_phone()  # no observedPeople bucket at all
    report = analyze(index, action="phone")
    derived = next(e for e in report.events if e.source == "derived")
    assert derived.evidence is not None
    assert "person(s) in frame" not in derived.evidence


def test_require_single_person_drops_composite_when_two_people_tracked():
    index = _index_with_person_and_phone(
        observed_people=[
            {"instances": [{"start": "0:00:00", "end": "0:00:10"}]},
            {"instances": [{"start": "0:00:02", "end": "0:00:08"}]},
        ]
    )
    report = analyze(index, action="phone", require_single_person=True)
    matched = {(e.matched_term, e.source) for e in report.events}
    assert ("person using phone", "derived") not in matched
    assert ("cell phone", "objects") in matched  # non-composite match unaffected


def test_require_single_person_keeps_composite_when_one_person_tracked():
    index = _index_with_person_and_phone(
        observed_people=[{"instances": [{"start": "0:00:00", "end": "0:00:10"}]}]
    )
    report = analyze(index, action="phone", require_single_person=True)
    matched = {(e.matched_term, e.source) for e in report.events}
    assert ("person using phone", "derived") in matched


def test_require_single_person_no_effect_without_observed_people_data():
    # Silently a no-op when observedPeople data was never available (e.g. a
    # Default-preset video) -- can't require something it has no way to
    # check, so this must not accidentally drop everything.
    index = _index_with_person_and_phone()
    report = analyze(index, action="phone", require_single_person=True)
    matched = {(e.matched_term, e.source) for e in report.events}
    assert ("person using phone", "derived") in matched


def test_report_rendering_round_trip(sample_index: dict[str, Any], tmp_path: Path):
    report = analyze(sample_index, action="jumping")

    text = to_console_text(report, "people_jumping.mp4")
    assert "jumping" in text
    assert "3" in text  # occurrence count appears somewhere in the summary

    as_dict = to_dict(report, "people_jumping.mp4")
    assert as_dict["occurrence_count"] == 3
    assert len(as_dict["events"]) == 3

    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"
    write_json(report, "people_jumping.mp4", json_path)
    write_html(report, "people_jumping.mp4", html_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["occurrence_count"] == 3

    html = html_path.read_text(encoding="utf-8")
    assert "jumping" in html
    assert "bar-fill" in html  # timeline rendered since duration is known
