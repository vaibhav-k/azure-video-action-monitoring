"""
This module contains constants used throughout the Azure Video Action Monitoring project.
These constants include URLs, API versions, and terminal states for the Video Indexer processing pipeline.
"""

# Keyword synonyms so a single "action of interest" like "jump" also matches
# related concepts Video Indexer's vision models are more likely to tag.
#
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

# Composite ("derived") actions: a named action that isn't any single
# label/keyword/object Video Indexer returns on its own, but a temporal
# overlap between two things it DOES detect. E.g. there's no "using phone"
# label or object class, but if a person-like label and a "cell phone"
# object are both present at the same moment, that overlap is a reasonable,
# inspectable proxy for "a person is using a phone" -- see
# action_analyzer.py's `_derive_composite_actions`. Each entry is
# (derived name, {left-side names}, {right-side names}); both name sets are
# matched case-insensitively against `labels`/`keywords`/`detectedObjects`/
# `ocr` item names. Add more rows here for other person+object overlaps
# without touching the derivation logic itself.
COMPOSITE_ACTIONS: list[tuple[str, set[str], set[str]]] = [
    ("person using phone", {"person", "man", "woman", "people"}, {"cell phone"}),
]

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
