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

# `names` below are drawn from Video Indexer's actual fixed 80-class
# `detectedObjects` vocabulary (confirmed against Microsoft's own docs --
# https://learn.microsoft.com/en-us/azure/azure-video-indexer/object-detection-insight
# -- and cross-checked against this project's own raw insight captures in
# output/*.raw_insights.json), matched *exactly* (case-insensitively) since
# that's the literal `displayName` string Video Indexer returns. `synonyms`
# are substring guesses against the separate, much larger and *unfixed*
# labels/keywords tagging vocabulary -- unverified against real captures,
# so kept as substrings rather than exact names (partly so a fragment like
# "ball" or "racket" still catches a real class name that contains it, e.g.
# "sports ball" / "tennis racket", without having to hardcode every one).
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
        # "car" is the only real detectedObjects class that fits "driving" --
        # the vocabulary's other vehicle classes (bicycle, motorcycle, bus,
        # boat, train, airplane) aren't what this composite's name claims,
        # so they're deliberately left out rather than guessed in.
        right=CompositeSide(names=frozenset({"car"})),
    ),
    CompositeAction(
        name="person playing sports",
        left=_PERSON_LIKE,
        right=CompositeSide(
            names=frozenset(
                {
                    "sports ball",
                    "tennis racket",
                    "baseball glove",
                    "skis",
                    "snowboard",
                    "skateboard",
                    "surfboard",
                    "frisbee",
                    "kite",
                }
            ),
            synonyms=("sports equipment", "athletic game", "ball", "bat", "racket"),
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
        # "a person near a cell phone", and it will fire in plenty of
        # non-checkout scenes (an office desk, a living room TV) too.
        right=CompositeSide(
            names=frozenset({"keyboard", "laptop", "computer mouse"}),
            synonyms=("computer", "monitor", "television", "tv"),
        ),
    ),
    CompositeAction(
        name="person carrying bag",
        left=_PERSON_LIKE,
        # backpack/handbag/suitcase are all real detectedObjects classes.
        # Useful for loss-prevention framing (leaving with an unscanned
        # item) or just customer-flow tracking -- but this can't tell
        # whether the bag came from the store or was already theirs, so
        # treat a hit as "worth a look," not a shoplifting claim.
        right=CompositeSide(names=frozenset({"backpack", "handbag", "suitcase"})),
    ),
    CompositeAction(
        name="person reading",
        left=_PERSON_LIKE,
        # "book" is a real detectedObjects class. Pairs thematically with
        # `person using phone` as an "employee distraction" signal, but a
        # hit is just as likely to be a customer reading a product label
        # or a book they're buying.
        right=CompositeSide(names=frozenset({"book"})),
    ),
    CompositeAction(
        name="person near knife or scissors",
        left=_PERSON_LIKE,
        # knife/scissors are real detectedObjects classes. Deliberately
        # named neutrally rather than as a "weapon" or safety alert: a hit
        # here is at least as likely to be inventory being scanned
        # (kitchenware for sale, a box cutter opening a delivery) as
        # anything actually held or brandished, and this project has no
        # way to tell the difference. Heaviest false-positive risk of any
        # composite in this list -- review before acting on it, never
        # auto-escalate.
        right=CompositeSide(names=frozenset({"knife", "scissors"})),
    ),
]

# An overlap shorter than this is treated as detector jitter rather than a
# real co-occurrence. Opt-in: `analyze()`/`analyze_all()` only apply this
# when a caller explicitly passes `min_overlap_seconds` (e.g. the CLIs
# default to this constant; direct library/test callers see every overlap
# unless they ask for filtering).
DEFAULT_MIN_COMPOSITE_OVERLAP_SECONDS = 0.15

# Heuristic thresholds for recognizing a burned-in overlay (a camera
# timestamp, watermark, station bug) in `ocr`'s one spatial field -- a
# static left/top/width/height box per OCR item (not tracked per-instance).
# Used only to gate the `person handling cash` composite in
# `_derive_composite_actions` (never to hide raw OCR rows): an OCR item is
# treated as a likely overlay, and excluded as composite evidence, when its
# box is both small relative to the frame and sitting within a margin of an
# edge -- e.g. a security-camera clock reading "17 14 07" near a top
# corner, as literally captured in this project's own
# `cashier.raw_insights.json`. This can only run when the video's
# width/height are present in the insights payload (see
# `_video_dimensions`); it's a pixel heuristic, not a guarantee, and always
# on -- not currently CLI-configurable.
DEFAULT_OCR_OVERLAY_EDGE_MARGIN_FRACTION = 0.2
DEFAULT_OCR_OVERLAY_MAX_AREA_FRACTION = 0.05

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
