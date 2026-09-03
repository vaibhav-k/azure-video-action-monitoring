"""
This module contains constants used throughout the Azure Video Action Monitoring project.
These constants include URLs, API versions, and terminal states for the Video Indexer processing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

# Keyword synonyms so a single "action of interest" like "jump" also matches
# related concepts Video Indexer's vision models are more likely to tag.

# "cash"/"money" are different in kind from the others: Video Indexer's
# labels/keywords/detectedObjects vocabularies have no currency concept at
# all (detectedObjects' ~80 classes are generic COCO-style objects), so
# these synonyms only have anything to match against once the `ocr` bucket
# is included too (see action_analyzer.py) -- they're phrases that
# literally appear printed on US currency ("WE TRUST" from "IN GOD WE
# TRUST", "FEDERAL RESERVE NOTE", etc.), not visual concepts a label/object
# model would recognize. This is inherently narrower than the others: a
# bare denomination number ("100") isn't included since it's too ambiguous
# out of context (room numbers, percentages, ...) to match automatically --
# inspect `--save-raw-insights`' `ocr` bucket manually for those.
DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "jump": ["jump", "jumping", "jumped", "leap", "leaping", "hop", "hopping"],
    "fall": ["fall", "falling", "fell", "collapse"],
    "run": ["run", "running", "sprint", "sprinting"],
    "fight": ["fight", "fighting", "punch", "kick", "brawl"],
    "cash": [
        "cash",
        "dollar",
        "dollars",
        "currency",
        "banknote",
        "legal tender",
        "federal reserve",
        "united states of america",
        "we trust",
    ],
    "money": [
        "cash",
        "dollar",
        "dollars",
        "currency",
        "banknote",
        "legal tender",
        "federal reserve",
        "united states of america",
        "we trust",
    ],
}


@dataclass(frozen=True)
class CompositeSide:
    """One side of a composite-action match (see `CompositeAction` below).

    `names`, when given, are exact detected-item names, matched
    case-insensitively -- e.g. a specific object class like "cell phone".
    `synonyms`, when given, are substring needles, matched case-insensitively
    the same way `DEFAULT_SYNONYMS` are -- needed for a side like currency,
    where the underlying OCR text varies per banknote ("100", "WE TRUST", a
    serial number) and no fixed set of exact names could ever cover it.
    A name matches this side if it matches either `names` or `synonyms`
    (both may be given at once).
    """

    names: frozenset[str] = frozenset()
    synonyms: tuple[str, ...] = ()

    def matches(self, name: str) -> bool:
        name_lower = name.lower()
        if name_lower in self.names:
            return True
        return any(needle in name_lower for needle in self.synonyms)


@dataclass(frozen=True)
class CompositeAction:
    """A named action that isn't any single label/keyword/object Video
    Indexer returns on its own, but a temporal overlap between two things it
    DOES detect. E.g. there's no "using phone" label or object class, but if
    a person-like label and a "cell phone" object are both present at the
    same moment, that overlap is a reasonable, inspectable proxy for "a
    person is using a phone" -- see action_analyzer.py's
    `_derive_composite_actions`.
    """

    name: str
    left: CompositeSide
    right: CompositeSide


_PERSON_LIKE = CompositeSide(names=frozenset({"person", "man", "woman", "people"}))

# Add more entries here for other person+object (or object+object) overlaps
# without touching the derivation logic in action_analyzer.py at all.
COMPOSITE_ACTIONS: list[CompositeAction] = [
    CompositeAction(
        name="person using phone",
        left=_PERSON_LIKE,
        right=CompositeSide(names=frozenset({"cell phone"})),
    ),
    CompositeAction(
        name="person driving car",
        left=_PERSON_LIKE,
        right=CompositeSide(names=frozenset({"car", "outdoor vehicle", "vehicle"})),
    ),
    CompositeAction(
        name="person playing sports",
        left=_PERSON_LIKE,
        right=CompositeSide(
            names=frozenset(
                {"sports equipment", "athletic game", "ball", "bat", "racket"}
            )
        ),
    ),
    CompositeAction(
        name="person handling cash",
        left=_PERSON_LIKE,
        # Reuses DEFAULT_SYNONYMS["cash"] rather than a fixed name set --
        # there's no "money" object/label class at all, so this only ever
        # matches via OCR text literally printed on currency, which varies
        # per banknote and can't be enumerated as exact names.
        right=CompositeSide(synonyms=tuple(DEFAULT_SYNONYMS["cash"])),
    ),
    CompositeAction(
        name="person at register",
        left=_PERSON_LIKE,
        # Loose proxy for operating a point-of-sale terminal/keypad: Video
        # Indexer's object vocabulary has no "cash register" or "POS
        # terminal" class, so this leans on the closest generic classes
        # actually available. Weakest-evidence composite in this list --
        # "a person near any keyboard/screen" is a much looser claim than
        # "a person near a cell phone".
        right=CompositeSide(
            names=frozenset({"keyboard", "laptop", "computer", "monitor", "tv"})
        ),
    ),
    CompositeAction(
        name="person dining",
        left=_PERSON_LIKE,
        # Strong proxy for an active meal, cafe setting, or dining room insight.
        # Combines tableware and common food categories to isolate eating/serving.
        right=CompositeSide(
            names=frozenset(
                {
                    "fork",
                    "knife",
                    "spoon",
                    "bowl",
                    "cup",
                    "wine glass",
                    "bottle",
                    "sandwich",
                    "pizza",
                    "donut",
                    "cake",
                    "orange",
                    "apple",
                    "banana",
                }
            )
        ),
    ),
    CompositeAction(
        name="person traveling",
        left=_PERSON_LIKE,
        # Isolates typical travel, airport, or hotel check-in behaviors.
        right=CompositeSide(names=frozenset({"suitcase", "backpack", "handbag"})),
    ),
    CompositeAction(
        name="person interacting with pet",
        left=_PERSON_LIKE,
        # Flags interactions with domestic animals typically found indoors or on walks.
        right=CompositeSide(names=frozenset({"cat", "dog"})),
    ),
]

# An overlap shorter than this is treated as detector jitter rather than a
# real co-occurrence. Opt-in: `analyze()`/`analyze_all()` only apply this
# when a caller explicitly passes `min_overlap_seconds` (e.g. the CLIs
# default to this constant; direct library/test callers see every overlap
# unless they ask for filtering).
DEFAULT_MIN_COMPOSITE_OVERLAP_SECONDS = 0.15

# Video annotation and overlay settings.
DEFAULT_MIN_CONFIDENCE = 0.5
MAX_OVERLAY_LINES = 6  # avoid the caption box swallowing the frame

# URL and API version constants for Azure Resource Manager (ARM) and Video Indexer data plane.
ARM_BASE_URL = "https://management.azure.com"
ARM_API_VERSION = "2025-04-01"
DATA_PLANE_BASE_URL = "https://api.videoindexer.ai"

# Terminal states reported by the Video Indexer processing pipeline.
STATE_PROCESSED = "Processed"
STATE_FAILED = "Failed"
