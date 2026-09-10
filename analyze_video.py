#!/usr/bin/env python3
"""
analyze_video.py -- a small, honest proof-of-concept CLI that uses Azure AI
Video Indexer's current REST API to analyze pre-recorded videos and surface,
as concretely as the API actually allows:

    * people Video Indexer observed/tracked in the footage
    * "activities" -- Video Indexer's own visual *labels*, since that is the
      one insight category whose documentation explicitly includes actions
      (Microsoft's own example: "swimming") alongside objects/scenes
    * where the timing of an activity and an observed person's appearance
      overlap closely enough that this script can, on its own, *infer* a
      likely association between the two

That last point is the most important thing to be upfront about: Video
Indexer does not provide a "this person did this activity" relationship
anywhere in its API. It gives two independent, purely temporal insight
buckets (labels; observedPeople) with no cross-references between them.
Every person<->activity line in this tool's output is this script's own
inference from overlapping timestamps -- never a fact Video Indexer itself
asserts. See the "WHAT THIS ACTUALLY DEMONSTRATES" section in README.md,
and the NOTES section this script appends to every report, for the full
honesty statement this design leans on throughout.

Verified against Microsoft's current public documentation while building
this (fetched 2026-09-10):
    - Auth:   Generate Access Token (ARM control plane), api-version
              2025-04-01 -- https://learn.microsoft.com/en-us/rest/api/
              videoindexer/stable/generate/access-token
    - Labels: https://learn.microsoft.com/en-us/azure/azure-video-indexer/
              labels-identification-insight (explicitly documents that
              labels cover "visual objects", "actions" (e.g. "swimming"),
              and "general entities" -- there is no separate, dedicated
              action/activity-recognition insight in the current API)
    - Observed people: https://learn.microsoft.com/en-us/azure/
              azure-video-indexer/observed-matched-people-insight
    - Detected objects: https://learn.microsoft.com/en-us/azure/
              azure-video-indexer/object-detection-insight

One specific, deliberately-flagged gap in that verification: none of the
three insight schemas above documents a bounding-box/spatial field (no
left/top/width/height, no x/y) on labels, detectedObjects, or
observedPeople instances -- only start/end timestamps. This script
therefore always reports bounding boxes as unavailable (null) rather than
fabricating spatial coordinates the API does not return. If a future API
version adds them, `_person_dict`/`_activity_dict`/`_object_dict` below are
exactly where to start plumbing them through.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import requests
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is in requirements.txt
    load_dotenv = None  # type: ignore[assignment]

logger = logging.getLogger("analyze_video")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARM_BASE_URL = "https://management.azure.com"
ARM_API_VERSION = "2025-04-01"
DATA_PLANE_BASE_URL = "https://api.videoindexer.ai"

# generateAccessToken's request body -- always the same for this script,
# since it only ever needs one Video Indexer account's worth of access.
ARM_TOKEN_PERMISSION_TYPE = "Contributor"
ARM_TOKEN_SCOPE = "Account"

# Every video this script uploads is indexed the same way: never shared
# publicly, indexed under the account's configured preset.
UPLOAD_PRIVACY = "Private"

# Per-request timeouts (seconds), one per Video Indexer HTTP call this
# client makes -- kept as named constants rather than inline numbers so
# all three are visible and adjustable in one place.
ARM_TOKEN_REQUEST_TIMEOUT_SECONDS = 30
UPLOAD_REQUEST_TIMEOUT_SECONDS = 600
GET_INDEX_REQUEST_TIMEOUT_SECONDS = 60

STATE_PROCESSED = "Processed"
STATE_FAILED = "Failed"

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
}

# Output artifact filename suffixes -- the three files process_video and
# process_video_by_id write for every video, named "{stem}{suffix}".
INSIGHTS_SUFFIX = ".insights.json"
ACTIVITIES_SUFFIX = ".activities.json"
REPORT_SUFFIX = ".report.txt"

# render_report's section headers, named here so the human-readable report
# and this module's own docstrings/tests reference the same literal text
# instead of retyping it.
REPORT_TITLE = "VIDEO ACTIVITY REPORT"
REPORT_SUMMARY_HEADER = "SUMMARY"
REPORT_PEOPLE_HEADER = "PEOPLE"
REPORT_ACTIVITY_SUMMARY_HEADER = "ACTIVITY SUMMARY"
REPORT_NOTES_HEADER = "NOTES"

# build_summary spells out small counts ("one person", "three activities")
# rather than starting a clause with a bare numeral -- reads like something
# a person would actually write. Falls back to the numeral for anything
# larger than this covers.
_SMALL_NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}

# Video Indexer's `labels` vocabulary is open-ended and mixes objects/scenes
# ("building", "necktie") in with actions ("running", "swimming") -- there is
# no field that tells you which is which. These names describe *presence*,
# not behavior, and would misleadingly pad an "activities" list if left in,
# so they're excluded from the activity view (they still appear in the full,
# unfiltered `labels` output). This is a display heuristic this script
# applies on top of real Video Indexer output -- not a Video Indexer field,
# and not a claim that this list is exhaustive.
_NON_ACTIVITY_LABELS = {
    "person",
    "people",
    "human face",
    "man",
    "woman",
    "child",
    "adult",
    "clothing",
    "outerwear",
    "footwear",
    "jewelry",
    "makeup",
    "street fashion",
    "necktie",
    "handbag",
    "backpack",
    "suitcase",
    "hat",
    "glasses",
    "sunglasses",
    "indoor",
    "outdoor",
    "building",
    "wall",
    "floor",
    "ceiling",
    "room",
    "furniture",
    "table",
    "chair",
    "window",
    "door",
    "sky",
    "tree",
    "grass",
    "road",
    "street",
    "vehicle",
    "car",
    # Compound scene/place labels that happen to contain a genuine "-ing"
    # activity word as one of their own words (e.g. "swimming pool"
    # contains "swimming"), so the general word-level heuristic below
    # can't tell them apart from a real activity label on its own --
    # curated out explicitly, the same way the plain object/scene words
    # above are.
    "swimming pool",
}

# A label is treated as describing an activity when its name either ends in
# "ing" (catches the open-ended gerund-shaped tags Video Indexer's
# vocabulary can return, e.g. "skateboarding") or appears in this small,
# explicitly-curated list of common non-"-ing" action words. Both this list
# and _NON_ACTIVITY_LABELS are a reporting/display choice this script makes
# on top of Video Indexer's real labels -- never a Video Indexer field, and
# never a claim of exhaustive or certain action detection. See this
# module's docstring and README.md's "Known limitations".
_ACTIVITY_WORD_HINTS = {
    "walk",
    "run",
    "jump",
    "sit",
    "stand",
    "talk",
    "dance",
    "fight",
    "drive",
    "eat",
    "drink",
    "wave",
    "hug",
    "kiss",
    "clap",
    "cook",
    "play",
    "exercise",
    "swim",
    "cycle",
    "box",
    "work",
    "read",
    "write",
    "shop",
    "carry",
    "fall",
    "climb",
    "throw",
    "catch",
    "push",
    "pull",
    "point",
}


def _looks_like_activity(label_name: str) -> bool:
    """Whole-word matching, deliberately: a naive substring check would
    flag compound object/scene labels that merely contain an action word
    as a substring of a *different* word -- e.g. a sandbox-shaped toy
    label would contain "box" (an _ACTIVITY_WORD_HINTS entry) purely as
    a substring of "sandbox". Splitting into words first (and checking
    each word's own "-ing" suffix, not just the whole label's) avoids
    that whole class of false positive while still catching genuine
    multi-word activity labels, e.g. "long jumping".

    That word-level check alone isn't enough for a label like "swimming
    pool", though: "swimming" is a real, whole "-ing" word there, not a
    substring artifact -- it's just that the label as a whole names a
    place, not an action. Cases like that are curated directly into
    _NON_ACTIVITY_LABELS instead (checked first, below), since no
    general rule can reliably tell "swimming" (the activity) from
    "swimming pool" (the place) apart."""
    name = label_name.lower().strip()
    if name in _NON_ACTIVITY_LABELS:
        return False
    words = name.split()
    if any(word.endswith("ing") for word in words):
        return True
    return any(word in _ACTIVITY_WORD_HINTS for word in words)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigError(RuntimeError):
    """Raised when required Azure configuration is missing or invalid."""


@dataclass
class Settings:
    subscription_id: str
    resource_group: str
    account_name: str
    account_id: str
    location: str
    indexing_preset: str = "Advanced"
    processing_timeout_seconds: float = 1800.0
    poll_interval_seconds: float = 10.0

    @staticmethod
    def from_env() -> Settings:
        if load_dotenv is not None:
            load_dotenv()

        required = {
            "AVI_SUBSCRIPTION_ID": "subscription_id",
            "AVI_RESOURCE_GROUP": "resource_group",
            "AVI_ACCOUNT_NAME": "account_name",
            "AVI_ACCOUNT_ID": "account_id",
            "AVI_LOCATION": "location",
        }
        values: dict[str, str] = {}
        missing: list[str] = []
        for env_name in required:
            value = os.environ.get(env_name, "").strip()
            if not value:
                missing.append(env_name)
            values[env_name] = value

        if missing:
            raise ConfigError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill in your Video "
                "Indexer account's details (Azure Portal -> your Video "
                "Indexer resource -> Overview / Properties)."
            )

        preset = os.environ.get("AVI_INDEXING_PRESET", "Advanced").strip() or "Advanced"
        if preset not in ("Default", "Advanced"):
            raise ConfigError(
                f"AVI_INDEXING_PRESET must be 'Default' or 'Advanced', got {preset!r}."
            )

        return Settings(
            subscription_id=values["AVI_SUBSCRIPTION_ID"],
            resource_group=values["AVI_RESOURCE_GROUP"],
            account_name=values["AVI_ACCOUNT_NAME"],
            account_id=values["AVI_ACCOUNT_ID"],
            location=values["AVI_LOCATION"],
            indexing_preset=preset,
            processing_timeout_seconds=float(
                os.environ.get("AVI_PROCESSING_TIMEOUT_SECONDS", "1800")
            ),
            poll_interval_seconds=float(
                os.environ.get("AVI_POLL_INTERVAL_SECONDS", "10")
            ),
        )


# ---------------------------------------------------------------------------
# Video Indexer client
# ---------------------------------------------------------------------------


class VideoIndexerError(RuntimeError):
    """Base class for all Video Indexer client errors."""


class AuthenticationFailedError(VideoIndexerError):
    """Raised when an Azure AD or Video Indexer token could not be obtained."""


class UploadFailedError(VideoIndexerError):
    """Raised when the video upload request itself fails."""


class ProcessingFailedError(VideoIndexerError):
    """Raised when Video Indexer finishes processing with a Failed state."""


class ProcessingTimeoutError(VideoIndexerError):
    """Raised when processing does not complete within the configured timeout."""


class VideoIndexerClient:
    """Client for Azure AI Video Indexer using the current ARM-based
    authentication flow: Azure AD -> ARM `generateAccessToken` -> the
    api.videoindexer.ai data-plane API. This is the supported path for
    accounts created after the classic API-key trial model was retired for
    new accounts; it needs no Video Indexer API key at all, only an Azure
    identity (`az login`, or AZURE_CLIENT_ID/AZURE_TENANT_ID/
    AZURE_CLIENT_SECRET for a service principal) with access to the
    account's resource group.
    """

    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self._settings = settings
        self._session = session or requests.Session()
        self._credential = DefaultAzureCredential()
        self._access_token: str | None = None

    def _get_arm_token(self) -> str:
        try:
            token = self._credential.get_token("https://management.azure.com/.default")
        except ClientAuthenticationError as exc:
            raise AuthenticationFailedError(
                "Could not obtain an Azure AD token. Make sure you're logged "
                "in (`az login`), or that AZURE_CLIENT_ID / AZURE_TENANT_ID / "
                "AZURE_CLIENT_SECRET are set for a service principal with "
                "access to the Video Indexer account's resource group."
            ) from exc
        return token.token

    def get_access_token(self) -> str:
        """Mint a Video Indexer data-plane access token via the ARM control
        plane's `generateAccessToken` operation. Never logs or prints the
        token itself -- only that a token was (or wasn't) obtained."""
        arm_token = self._get_arm_token()
        url = (
            f"{ARM_BASE_URL}/subscriptions/{self._settings.subscription_id}"
            f"/resourceGroups/{self._settings.resource_group}"
            f"/providers/Microsoft.VideoIndexer/accounts/{self._settings.account_name}"
            f"/generateAccessToken"
        )
        response = self._session.post(
            url,
            params={"api-version": ARM_API_VERSION},
            json={
                "permissionType": ARM_TOKEN_PERMISSION_TYPE,
                "scope": ARM_TOKEN_SCOPE,
            },
            headers={"Authorization": f"Bearer {arm_token}"},
            timeout=ARM_TOKEN_REQUEST_TIMEOUT_SECONDS,
        )
        if not response.ok:
            raise AuthenticationFailedError(
                f"generateAccessToken failed ({response.status_code}). "
                "Check that AVI_SUBSCRIPTION_ID/AVI_RESOURCE_GROUP/"
                "AVI_ACCOUNT_NAME in .env match the Video Indexer resource "
                "in the Azure Portal, and that your identity has at least "
                "'Contributor' on it."
            )
        token = response.json().get("accessToken")
        if not token:
            raise AuthenticationFailedError(
                "generateAccessToken response did not contain an accessToken."
            )
        self._access_token = token
        return token

    def _ensure_token(self) -> str:
        return self._access_token or self.get_access_token()

    def upload_video(self, video_path: Path, name: str) -> str:
        """Upload a local video file and return the Video Indexer video ID."""
        token = self._ensure_token()
        url = (
            f"{DATA_PLANE_BASE_URL}/{self._settings.location}"
            f"/Accounts/{self._settings.account_id}/Videos"
        )
        params = {
            "accessToken": token,
            "name": name,
            "privacy": UPLOAD_PRIVACY,
            "indexingPreset": self._settings.indexing_preset,
        }
        with video_path.open("rb") as fh:
            files = {"file": (video_path.name, fh, "application/octet-stream")}
            response = self._session.post(
                url, params=params, files=files, timeout=UPLOAD_REQUEST_TIMEOUT_SECONDS
            )

        if not response.ok:
            raise UploadFailedError(
                f"Upload failed ({response.status_code}): {response.text[:500]}"
            )
        video_id = response.json().get("id")
        if not video_id:
            raise UploadFailedError("Upload response did not contain a video id.")
        return video_id

    def get_video_index(self, video_id: str) -> dict[str, Any]:
        """Fetch the video's current index/insights payload, in whatever
        state it's currently in. `includeSummarizedInsights=false` asks
        Video Indexer to omit its separate, simplified `summarizedInsights`
        section -- this script only ever reads the detailed `videos[0]
        .insights` section regardless, so this is a best-effort request to
        avoid an unused, larger response, not something this script's
        correctness depends on."""
        token = self._ensure_token()
        url = (
            f"{DATA_PLANE_BASE_URL}/{self._settings.location}/Accounts/"
            f"{self._settings.account_id}/Videos/{video_id}/Index"
        )
        response = self._session.get(
            url,
            params={"accessToken": token, "includeSummarizedInsights": "false"},
            timeout=GET_INDEX_REQUEST_TIMEOUT_SECONDS,
        )
        if not response.ok:
            raise VideoIndexerError(
                f"Get index failed ({response.status_code}): {response.text[:500]}"
            )
        return response.json()

    def wait_for_processing(self, video_id: str) -> dict[str, Any]:
        """Poll the video index until it reaches a terminal state, logging
        each state transition (never the access token) as progress."""
        deadline = time.monotonic() + self._settings.processing_timeout_seconds
        last_state = None
        while True:
            index = self.get_video_index(video_id)
            state = index.get("state")
            if state != last_state:
                logger.info("  video state: %s", state)
                last_state = state

            if state == STATE_PROCESSED:
                return index
            if state == STATE_FAILED:
                raise ProcessingFailedError(
                    f"Video Indexer failed to process this video: "
                    f"{index.get('processingProgress', 'no details provided')}"
                )
            if time.monotonic() >= deadline:
                raise ProcessingTimeoutError(
                    f"Timed out after {self._settings.processing_timeout_seconds:.0f}s "
                    f"waiting for processing (last state: {state})."
                )
            time.sleep(self._settings.poll_interval_seconds)

    def index_video(self, video_path: Path, name: str) -> dict[str, Any]:
        """Upload a video and block until its full insights index is ready."""
        video_id = self.upload_video(video_path, name=name)
        logger.info("  uploaded, video id=%s", video_id)
        return self.wait_for_processing(video_id)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _parse_vi_timestamp(value: str) -> float:
    """
    Video Indexer timestamps look like "0:00:09.84" (H:MM:SS.fraction) --
    parse into seconds. Defensive about missing/odd parts rather than
    raising, since this is third-party data this script doesn't control.
    """
    try:
        parts = value.strip().split(":")
        parts = list(parts)
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = "0"
            minutes, seconds = parts
        else:
            return float(value)
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (ValueError, TypeError):
        return 0.0


def _fmt_seconds(seconds: float | None) -> str:
    """Render seconds as "MM:SS" (or "H:MM:SS" past an hour), matching the
    style of this project's example report."""
    if seconds is None:
        return "?"
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Insight extraction / normalization
# ---------------------------------------------------------------------------


@dataclass
class Instance:
    start_seconds: float
    end_seconds: float
    confidence: float | None = None


@dataclass
class Person:
    person_id: int
    label: str  # "Person 2", or the matched/recognized name when confident
    matched_face_confidence: float | None
    appearances: list[Instance] = field(default_factory=list[Instance])


@dataclass
class Activity:
    name: str
    instance: Instance
    person_id: int | None = None
    person_label: str | None = None
    association_note: str = ""


@dataclass
class ObjectDetection:
    name: str
    instances: list[Instance] = field(default_factory=list[Instance])


@dataclass
class Label:
    name: str
    instances: list[Instance] = field(default_factory=list[Instance])


@dataclass
class VideoAnalysis:
    video_filename: str
    duration_seconds: float | None
    people: list[Person]
    activities: list[Activity]
    objects: list[ObjectDetection]
    labels: list[Label]


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    """Narrow a raw JSON value (statically an ``Any`` coming out of an
    untyped payload) into a properly typed list of dicts. Video Indexer's
    array-shaped insight fields (``labels``, ``faces``, ``observedPeople``,
    ``detectedObjects``, an item's own ``instances``) are looped over a lot
    below; going through this helper -- rather than each call site doing
    its own ``insights.get(key, []) or []`` -- means Pylance can actually
    infer the loop variable's type instead of reporting it as partially
    unknown, and a payload with a malformed (non-list, or list-of-non-dict)
    value degrades to "nothing found" instead of raising."""
    if not isinstance(value, list):
        return []
    return [cast(dict[str, Any], entry) for entry in value if isinstance(entry, dict)]


def _extract_insights(raw: dict[str, Any]) -> dict[str, Any]:
    """Video Indexer's Get-Video-Index response nests the detailed,
    non-summarized insights under videos[0].insights. Fall back to a
    top-level "insights" key defensively, in case a payload from a
    different retrieval path is ever passed in (e.g. a previously-saved
    file), rather than raising."""
    videos = raw.get("videos")
    if isinstance(videos, list) and videos:
        insights = videos[0].get("insights")
        if isinstance(insights, dict):
            return insights
    insights = raw.get("insights")
    if isinstance(insights, dict):
        return insights
    return {}


def _instances_of(item: dict[str, Any]) -> list[Instance]:
    result = []
    for inst in _as_dict_list(item.get("instances")):
        start = inst.get("adjustedStart") or inst.get("start")
        end = inst.get("adjustedEnd") or inst.get("end")
        if not start or not end:
            continue
        result.append(
            Instance(
                start_seconds=_parse_vi_timestamp(start),
                end_seconds=_parse_vi_timestamp(end),
                confidence=inst.get("confidence"),
            )
        )
    return result


def _face_names_by_id(insights: dict[str, Any]) -> dict[int, str]:
    names: dict[int, str] = {}
    for face in _as_dict_list(insights.get("faces")):
        face_id = face.get("id")
        name = face.get("name")
        if face_id is not None and name:
            names[face_id] = name
    return names


def _extract_people(insights: dict[str, Any]) -> list[Person]:
    """observedPeople is Video Indexer's body-tracking insight -- a
    separate model from `faces`, so a person here is only given a real
    name when Video Indexer itself linked the two via `matchingFace`
    (never guessed by this script). Otherwise this reports a generic
    "Person N" label. Per Video Indexer's documented schema, appearances
    carry only start/end timestamps -- no bounding box/spatial field is
    documented for this insight, so none is reported here."""
    face_names = _face_names_by_id(insights)
    people: list[Person] = []
    for item in _as_dict_list(insights.get("observedPeople")):
        person_id = item.get("id")
        if person_id is None:
            continue
        matching_face = item.get("matchingFace") or {}
        face_id = matching_face.get("id")
        confidence = matching_face.get("confidence")
        label = (
            face_names.get(face_id, f"Person {person_id}")
            if face_id is not None
            else f"Person {person_id}"
        )
        people.append(
            Person(
                person_id=person_id,
                label=label,
                matched_face_confidence=confidence,
                appearances=_instances_of(item),
            )
        )
    return people


def _extract_labels(insights: dict[str, Any]) -> list[Label]:
    """The complete, unfiltered set of labels Video Indexer returned --
    Video Indexer's documentation states this category covers "visual
    objects", "actions" (its own example: "swimming"), and "general
    entities" all together, with no field distinguishing which is which."""
    labels: list[Label] = []
    for item in _as_dict_list(insights.get("labels")):
        name = item.get("name")
        if not name:
            continue
        labels.append(Label(name=name, instances=_instances_of(item)))
    return labels


def _extract_objects(insights: dict[str, Any]) -> list[ObjectDetection]:
    """detectedObjects: Video Indexer's fixed-vocabulary object detector.
    Per its documented schema, instances carry only confidence + start/end
    -- no bounding box field is documented, so none is reported here."""
    objects: list[ObjectDetection] = []
    for item in _as_dict_list(insights.get("detectedObjects")):
        name = item.get("displayName") or item.get("type")
        if not name:
            continue
        objects.append(ObjectDetection(name=name, instances=_instances_of(item)))
    return objects


def _overlaps(a: Instance, b: Instance) -> bool:
    return a.start_seconds < b.end_seconds and b.start_seconds < a.end_seconds


def _build_activities(labels: list[Label], people: list[Person]) -> list[Activity]:
    """Promote the labels that look like activities (see
    _looks_like_activity) into Activity records, then -- separately, and
    only as an inference this script itself performs -- try to associate
    each occurrence with an observed person whose appearance overlaps it
    in time.

    This association is deliberately conservative: it's only made when
    *exactly one* observed person's appearance overlaps the activity's
    time range. Zero overlapping people means there's nothing to associate
    with; two or more overlapping people means this script has no way to
    tell which of them the label actually describes -- in both cases the
    activity is still reported, just without a person attached, along with
    a plain-English note on why. Video Indexer itself never asserts this
    link; it is not present anywhere in the raw insights.
    """
    activities: list[Activity] = []
    for label in labels:
        if not _looks_like_activity(label.name):
            continue
        for instance in label.instances:
            overlapping = [
                p for p in people if any(_overlaps(instance, a) for a in p.appearances)
            ]
            if len(overlapping) == 1:
                person = overlapping[0]
                activities.append(
                    Activity(
                        name=label.name,
                        instance=instance,
                        person_id=person.person_id,
                        person_label=person.label,
                        association_note=(
                            "inferred: this person's tracked appearance "
                            "overlaps this activity's time range"
                        ),
                    )
                )
            elif len(overlapping) == 0:
                activities.append(
                    Activity(
                        name=label.name,
                        instance=instance,
                        association_note=(
                            "no observed-people appearance overlapped this time range"
                        ),
                    )
                )
            else:
                ids = ", ".join(str(p.person_id) for p in overlapping)
                activities.append(
                    Activity(
                        name=label.name,
                        instance=instance,
                        association_note=(
                            f"{len(overlapping)} people (ids: {ids}) overlapped "
                            "this time range -- ambiguous, not assigned"
                        ),
                    )
                )
    activities.sort(key=lambda a: a.instance.start_seconds)
    return activities


def analyze_insights(raw: dict[str, Any], video_filename: str) -> VideoAnalysis:
    """Turn a raw Get-Video-Index response into this script's normalized
    VideoAnalysis -- the single function tests exercise without any
    network access. See this module's docstring for what "activities"
    does and doesn't mean here."""
    insights = _extract_insights(raw)
    duration = raw.get("durationInSeconds")
    if duration is None:
        duration_str = insights.get("duration")
        duration = _parse_vi_timestamp(duration_str) if duration_str else None

    people = _extract_people(insights)
    labels = _extract_labels(insights)
    objects = _extract_objects(insights)
    activities = _build_activities(labels, people)

    return VideoAnalysis(
        video_filename=video_filename,
        duration_seconds=duration,
        people=people,
        activities=activities,
        objects=objects,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Output: normalized JSON
# ---------------------------------------------------------------------------


def _instance_dict(instance: Instance) -> dict[str, Any]:
    return {
        "start_seconds": instance.start_seconds,
        "end_seconds": instance.end_seconds,
        "start": _fmt_seconds(instance.start_seconds),
        "end": _fmt_seconds(instance.end_seconds),
        "confidence": instance.confidence,
        # Not documented for this insight in the current Video Indexer API
        # (temporal localization only) -- see this module's docstring.
        "bounding_box": None,
    }


def _person_dict(person: Person) -> dict[str, Any]:
    return {
        "id": person.person_id,
        "label": person.label,
        "matched_face_confidence": person.matched_face_confidence,
        "appearances": [_instance_dict(a) for a in person.appearances],
    }


def _activity_dict(activity: Activity) -> dict[str, Any]:
    return {
        "name": activity.name,
        **_instance_dict(activity.instance),
        "person_id": activity.person_id,
        "person_label": activity.person_label,
        "association": {
            "basis": (
                "inferred_temporal_overlap" if activity.person_id is not None else None
            ),
            "note": activity.association_note,
            "provided_by_video_indexer": False,
        },
        "source": "labels",
    }


def _object_dict(obj: ObjectDetection) -> dict[str, Any]:
    return {
        "name": obj.name,
        "instances": [_instance_dict(i) for i in obj.instances],
    }


def _label_dict(label: Label) -> dict[str, Any]:
    return {
        "name": label.name,
        "instances": [_instance_dict(i) for i in label.instances],
    }


def to_activities_dict(analysis: VideoAnalysis) -> dict[str, Any]:
    return {
        "video": analysis.video_filename,
        "duration_seconds": analysis.duration_seconds,
        "people": [_person_dict(p) for p in analysis.people],
        "activities": [_activity_dict(a) for a in analysis.activities],
        "objects": [_object_dict(o) for o in analysis.objects],
        "labels": [_label_dict(label) for label in analysis.labels],
    }


# ---------------------------------------------------------------------------
# Output: human-readable report
# ---------------------------------------------------------------------------


def _report_header(analysis: VideoAnalysis) -> list[str]:
    lines = [
        REPORT_TITLE,
        "=" * 22,  # matches the exact underline length in README.md's example
        "",
        f"Video: {analysis.video_filename}",
    ]
    if analysis.duration_seconds:
        lines.append(f"Duration: {_fmt_seconds(analysis.duration_seconds)}")
    lines.append("")
    lines.append(f"People detected: {len(analysis.people)}")
    lines.append(f"Activities detected: {len(analysis.activities)}")
    lines.append("")
    return lines


def _join_terms(terms: list[str]) -> str:
    """Natural-language join: ["a"] -> "a"; ["a", "b"] -> "a and b";
    ["a", "b", "c"] -> "a, b, and c" -- the way a person actually writes a
    list in a sentence, rather than a robotic comma-separated blob."""
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"{terms[0]} and {terms[1]}"
    return ", ".join(terms[:-1]) + f", and {terms[-1]}"


def _count_word(n: int) -> str:
    return _SMALL_NUMBER_WORDS.get(n, str(n))


def _duration_phrase(seconds: float | None) -> str:
    """A conversational opener -- "This 5-second clip"/"This 2-minute
    clip" -- rather than a raw MM:SS timestamp, which reads like a log
    line, not a sentence."""
    if not seconds:
        return "This clip"
    total = round(seconds)
    if total < 60:
        return f"This {total}-second clip"
    minutes, secs = divmod(total, 60)
    if secs == 0:
        return f"This {minutes}-minute clip"
    return f"This {minutes}-minute, {secs}-second clip"


def _people_sentence(analysis: VideoAnalysis, duration_phrase: str) -> str:
    people_count = len(analysis.people)
    if people_count == 0:
        return f"{duration_phrase} doesn't have any tracked people in it."
    if people_count == 1:
        return f"{duration_phrase} tracks one person."
    return f"{duration_phrase} tracks {_count_word(people_count)} people."


def _no_activities_sentence(people_sentence: str) -> str:
    return (
        f"{people_sentence} No activity-shaped labels came back for it, "
        "so there's nothing to say here about what anyone was doing -- "
        "see NOTES for what Video Indexer's labels insight can and "
        "can't catch."
    )


def _rank_activity_counts(activities: list[Activity]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for activity in activities:
        counts[activity.name] = counts.get(activity.name, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _activity_sentence(ranked: list[tuple[str, int]]) -> str:
    names = [name for name, _ in ranked]
    if len(ranked) == 1:
        return f"Video Indexer's labels flag one activity here: {names[0]}."

    sentence = (
        f"Video Indexer's labels flag {_count_word(len(ranked))} activities "
        f"here: {_join_terms(names)}."
    )
    top_name, top_count = ranked[0]
    if top_count > ranked[1][1]:
        times = "time" if top_count == 1 else "times"
        sentence += f' "{top_name}" comes up the most, {top_count} {times}.'
    return sentence


def _full_association_sentence(total: int) -> str:
    if total == 1:
        return (
            "That lines up cleanly with the one tracked person, by timestamp overlap."
        )
    return "Every one of those lines up cleanly with a specific tracked person, by timestamp overlap."


def _partial_association_sentence(assigned: int, unassigned: int) -> str:
    verb_line_up = "lines up" if assigned == 1 else "line up"
    verb_dont = "doesn't" if unassigned == 1 else "don't"
    return (
        f"Of those, {_count_word(assigned)} {verb_line_up} with a "
        f"specific tracked person by timestamp overlap; the other "
        f"{_count_word(unassigned)} {verb_dont}, because either no one "
        "or more than one person was in frame at that point."
    )


def _association_sentence(activities: list[Activity]) -> str:
    total = len(activities)
    assigned = sum(1 for a in activities if a.person_id is not None)
    if assigned == total:
        return _full_association_sentence(total)
    if assigned == 0:
        return (
            "None of that could be pinned to a specific person, though -- "
            "at each moment, either no one or more than one person was in frame."
        )
    return _partial_association_sentence(assigned, total - assigned)


_SUMMARY_CAVEAT_SENTENCE = (
    "Worth keeping in mind: these are Video Indexer's own visual "
    "labels, not output from a dedicated action-recognition model, "
    "and the person-to-activity links above are this script's own "
    "inference from overlapping timestamps -- never something Video "
    "Indexer states outright. See NOTES below for the full picture."
)


def build_summary(analysis: VideoAnalysis) -> str:
    """A single, plain-English paragraph summarizing this analysis --
    written to read like a person's own description, not a report
    generator's template -- meant to be the one thing a reader (or a
    console-watcher) sees first, ahead of the itemized PEOPLE/activity
    lines. It never claims anything beyond what render_report's own NOTES
    section already discloses: activity names are still Video Indexer's
    own visual labels (not a dedicated action-recognition model), and
    every person<->activity pairing mentioned here is still this script's
    own inference from overlapping timestamps, never a Video Indexer
    fact -- that caveat is still spelled out here, just as one more
    sentence of prose rather than a bolted-on disclaimer. Each clause is
    built by its own small, single-purpose helper above so this function
    itself stays a short, flat dispatch over them."""
    duration_phrase = _duration_phrase(analysis.duration_seconds)
    people_sentence = _people_sentence(analysis, duration_phrase)

    if not analysis.activities:
        return _no_activities_sentence(people_sentence)

    ranked = _rank_activity_counts(analysis.activities)
    activity_sentence = _activity_sentence(ranked)
    association_sentence = _association_sentence(analysis.activities)

    return (
        f"{people_sentence} "
        f"{activity_sentence} "
        f"{association_sentence} "
        f"{_SUMMARY_CAVEAT_SENTENCE}"
    )


def _report_summary_section(analysis: VideoAnalysis) -> list[str]:
    return [
        REPORT_SUMMARY_HEADER,
        "-" * len(REPORT_SUMMARY_HEADER),
        build_summary(analysis),
        "",
    ]


def _fmt_person_line(person: Person) -> str:
    if person.appearances:
        span = (
            f"{_fmt_seconds(min(a.start_seconds for a in person.appearances))} - "
            f"{_fmt_seconds(max(a.end_seconds for a in person.appearances))}"
        )
    else:
        span = "no timing data"
    recognized = (
        f" (matched, confidence {person.matched_face_confidence:.2f})"
        if person.matched_face_confidence is not None
        else " (tracked only, not recognized)"
    )
    return f"  {person.label:<12} seen {span}{recognized}"


def _report_people_section(analysis: VideoAnalysis) -> list[str]:
    if not analysis.people:
        return []
    lines = [REPORT_PEOPLE_HEADER, "-" * len(REPORT_PEOPLE_HEADER)]
    for person in sorted(analysis.people, key=lambda p: p.person_id):
        lines.append(_fmt_person_line(person))
    lines.append("")
    return lines


def _fmt_activity_line(activity: Activity) -> str:
    start = _fmt_seconds(activity.instance.start_seconds)
    end = _fmt_seconds(activity.instance.end_seconds)
    conf = (
        f"{activity.instance.confidence:.2f}"
        if activity.instance.confidence is not None
        else "n/a"
    )
    who = f"{activity.person_label}: " if activity.person_label else ""
    return f"{start} - {end}   {who}{activity.name}   confidence: {conf}"


def _report_activity_summary(activities: list[Activity]) -> list[str]:
    lines = [REPORT_ACTIVITY_SUMMARY_HEADER, "-" * len(REPORT_ACTIVITY_SUMMARY_HEADER)]
    counts: dict[str, int] = {}
    for activity in activities:
        counts[activity.name] = counts.get(activity.name, 0) + 1
    for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        occurrence = "occurrence" if count == 1 else "occurrences"
        lines.append(f"{name}: {count} {occurrence}")
    lines.append("")
    return lines


def _report_activities_section(analysis: VideoAnalysis) -> list[str]:
    if not analysis.activities:
        return [
            "No activity-shaped labels were detected in this video (see NOTES).",
            "",
        ]
    lines = [_fmt_activity_line(a) for a in analysis.activities]
    lines.append("")
    lines.extend(_report_activity_summary(analysis.activities))
    return lines


def _report_notes() -> list[str]:
    return [
        REPORT_NOTES_HEADER,
        "-" * len(REPORT_NOTES_HEADER),
        (
            '- "Activities" are Video Indexer\'s own visual labels, filtered to '
            'those that read as actions (e.g. "running", "-ing"-ending '
            "tags) -- not output from a dedicated human-action-recognition "
            "model. Video Indexer's current API has no such model; only "
            "actions its general label vocabulary happens to name can appear "
            "here."
        ),
        (
            "- Person <-> activity pairings shown above are this script's own "
            "inference from overlapping timestamps, made only when exactly one "
            "observed person's appearance overlaps the activity -- never a "
            "relationship Video Indexer itself provides. See each activity's "
            '"association" field in the .activities.json file for the exact '
            "reasoning, including cases left unassigned."
        ),
        (
            "- Video Indexer's current API does not document a bounding-box/"
            "spatial field for labels, objects, or observed people -- only "
            "start/end timestamps. Bounding boxes are therefore always "
            "reported as unavailable (null) rather than invented."
        ),
        "",
    ]


def render_report(analysis: VideoAnalysis) -> str:
    lines: list[str] = []
    lines.extend(_report_header(analysis))
    lines.extend(_report_summary_section(analysis))
    lines.extend(_report_people_section(analysis))
    lines.extend(_report_activities_section(analysis))
    lines.extend(_report_notes())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch / single-video orchestration
# ---------------------------------------------------------------------------


def _output_paths(out_dir: Path, stem: str) -> tuple[Path, Path, Path]:
    return (
        out_dir / f"{stem}{INSIGHTS_SUFFIX}",
        out_dir / f"{stem}{ACTIVITIES_SUFFIX}",
        out_dir / f"{stem}{REPORT_SUFFIX}",
    )


def _already_processed(out_dir: Path, stem: str) -> bool:
    return all(p.is_file() for p in _output_paths(out_dir, stem))


def _discover_batch_videos(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


class _IndexesVideo(Protocol):
    """The one VideoIndexerClient method process_video actually calls --
    accepting this instead of the concrete VideoIndexerClient type lets
    tests pass a hand-rolled FakeVideoIndexerClient double without it
    needing to replicate VideoIndexerClient's full real surface, while
    main()'s real client is still fully type-checked at its own call site."""

    def index_video(self, video_path: Path, name: str) -> dict[str, Any]: ...


class _FetchesVideoById(Protocol):
    """The one VideoIndexerClient method process_video_by_id actually calls.
    A video reached by --video-id is already indexed, so this path never
    uploads anything -- it only waits out any remaining processing and
    fetches the finished insights, exactly like VideoIndexerClient.
    wait_for_processing already does for the upload path internally."""

    def wait_for_processing(self, video_id: str) -> dict[str, Any]: ...


def _print_summary(analysis: VideoAnalysis) -> None:
    """Print the same plain-English paragraph build_summary produces for
    the report file, so someone watching the console gets the headline
    takeaway without having to open the .report.txt file themselves."""
    logger.info("  summary: %s", build_summary(analysis))


def _print_activity_associations(analysis: VideoAnalysis) -> None:
    """Print, to the console, which activities were recognized and which
    person (if any) each was associated with -- the same per-activity
    lines render_report writes into the .report.txt file (via
    _fmt_activity_line), so the terminal output and the written report
    never disagree. Printed via logger.info so it shows up at default
    verbosity, right after a video finishes processing."""
    if not analysis.activities:
        logger.info("  no activity-shaped labels detected")
        return
    logger.info("  activities recognized:")
    for activity in analysis.activities:
        logger.info("    %s", _fmt_activity_line(activity))


def _write_outputs(
    analysis: VideoAnalysis, raw: dict[str, Any], out_dir: Path, stem: str
) -> tuple[Path, Path, Path]:
    """Write the three output artifacts (raw insights, normalized
    activities JSON, human-readable report) for one already-analyzed
    video. Shared by both process_video (local file, uploaded) and
    process_video_by_id (already-indexed, fetched by ID) so the output
    shape never drifts between the two entry points."""
    out_dir.mkdir(parents=True, exist_ok=True)
    insights_path, activities_path, report_path = _output_paths(out_dir, stem)
    insights_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    activities_path.write_text(
        json.dumps(to_activities_dict(analysis), indent=2), encoding="utf-8"
    )
    report_path.write_text(render_report(analysis), encoding="utf-8")
    return insights_path, activities_path, report_path


def process_video(
    client: _IndexesVideo, video_path: Path, out_dir: Path, force: bool
) -> bool:
    """Process one video end-to-end: upload, wait, extract, write all three
    output artifacts. Returns True if it ran, False if skipped (already
    processed, not --force)."""
    stem = video_path.stem

    if not force and _already_processed(out_dir, stem):
        logger.info(
            "Skipping %s (already processed; use --force to redo)", video_path.name
        )
        return False

    logger.info("Processing %s ...", video_path.name)
    raw = client.index_video(video_path, name=stem)

    analysis = analyze_insights(raw, video_filename=video_path.name)
    insights_path, activities_path, report_path = _write_outputs(
        analysis, raw, out_dir, stem
    )
    _print_summary(analysis)
    _print_activity_associations(analysis)

    logger.info(
        "  done: %d people, %d activities -> %s / %s / %s",
        len(analysis.people),
        len(analysis.activities),
        insights_path.name,
        activities_path.name,
        report_path.name,
    )
    return True


def process_video_by_id(
    client: _FetchesVideoById, video_id: str, out_dir: Path, force: bool
) -> bool:
    """Process an already-indexed Video Indexer video by ID: no local file
    and no upload -- just wait out any remaining processing, fetch its
    insights, and write the same three output artifacts as process_video.

    The output stem is the video's own registered `name` field from the
    fetched index, falling back to the video ID itself when Video Indexer
    didn't return one. Because that name is only known after the fetch,
    the already-processed skip check runs after fetching rather than
    before it (unlike process_video, which can check before uploading) --
    an acceptable trade-off for a single by-ID lookup, since fetching an
    already-processed video's index is a cheap read, not a re-upload."""
    logger.info("Fetching video id=%s ...", video_id)
    raw = client.wait_for_processing(video_id)
    stem = raw.get("name") or video_id

    if not force and _already_processed(out_dir, stem):
        logger.info("Skipping %s (already processed; use --force to redo)", stem)
        return False

    analysis = analyze_insights(raw, video_filename=stem)
    insights_path, activities_path, report_path = _write_outputs(
        analysis, raw, out_dir, stem
    )
    _print_summary(analysis)
    _print_activity_associations(analysis)

    logger.info(
        "  done: %d people, %d activities -> %s / %s / %s",
        len(analysis.people),
        len(analysis.activities),
        insights_path.name,
        activities_path.name,
        report_path.name,
    )
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze pre-recorded videos with Azure AI Video Indexer: upload, "
            "wait for processing, and write detailed insights, a normalized "
            "activities/people summary, and a human-readable report."
        )
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help=(
            "Path to a single video file to analyze (inside input/ or "
            "anywhere else on disk). If omitted, every supported video in "
            "input/ is processed (batch mode)."
        ),
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help=(
            "Analyze an already-indexed Video Indexer video by its video ID "
            "instead of a local file -- no upload, just wait out any "
            "remaining processing, fetch its insights, and write the same "
            "output artifacts. Single-video mode only: not combined with "
            "--video or batch mode."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if this video's output artifacts already exist.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Batch mode source directory. Default: ./input",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Where output artifacts are written. Default: ./output",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser


def _resolve_candidates(args: argparse.Namespace) -> list[Path]:
    """Video paths to consider: the single --video path, or every file
    _discover_batch_videos finds in --input-dir. Logs the "found N" / "none
    found" message either way -- an empty batch-mode result isn't an
    error, just nothing to do."""
    if args.video is not None:
        return [args.video]
    candidates = _discover_batch_videos(args.input_dir)
    if candidates:
        logger.info("Found %d video(s) in %s/", len(candidates), args.input_dir)
    else:
        logger.info(
            "No supported video files found in %s/. Supported: %s",
            args.input_dir,
            ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS)),
        )
    return candidates


def _invalid_video_reason(video_path: Path) -> str | None:
    """None when video_path is usable; otherwise a human-readable reason
    it isn't. Checked with no Azure config/credentials needed, so a
    typo'd path or wrong extension can be reported without requiring
    either."""
    if not video_path.is_file():
        return f"Video file not found: {video_path}"
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        return (
            f"Unsupported file type {video_path.name} ({video_path.suffix}). "
            f"Supported: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        )
    return None


def _filter_valid_videos(candidates: list[Path]) -> tuple[list[Path], int]:
    """Split candidates into (usable videos, count rejected) -- logging
    each rejection's reason as it's found."""
    videos: list[Path] = []
    failed = 0
    for video_path in candidates:
        reason = _invalid_video_reason(video_path)
        if reason:
            logger.error(reason)
            failed += 1
        else:
            videos.append(video_path)
    return videos, failed


def _process_all(
    client: VideoIndexerClient, videos: list[Path], out_dir: Path, force: bool
) -> tuple[int, int, int]:
    """Run process_video over every video, returning (processed, skipped,
    failed) counts. A per-video VideoIndexerError is logged and counted as
    a failure rather than aborting the rest of the batch."""
    processed = skipped = failed = 0
    for video_path in videos:
        try:
            if process_video(client, video_path, out_dir, force=force):
                processed += 1
            else:
                skipped += 1
        except VideoIndexerError:
            logger.exception("  failed to analyze %s", video_path.name)
            failed += 1
    return processed, skipped, failed


def _main_video_id(args: argparse.Namespace) -> int:
    """Handle --video-id mode: no local file, no upload, no batch -- fetch
    and process exactly one already-indexed video by ID."""
    try:
        settings = Settings.from_env()
    except ConfigError:
        logger.exception("Invalid Azure configuration")
        return 2

    client = VideoIndexerClient(settings)
    try:
        processed = process_video_by_id(
            client, args.video_id, args.output_dir, args.force
        )
    except VideoIndexerError:
        logger.exception("  failed to analyze video id=%s", args.video_id)
        return 1

    logger.info(
        "Done. %d processed, %d skipped, 0 failed.",
        1 if processed else 0,
        0 if processed else 1,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    args.input_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.video_id is not None:
        if args.video is not None:
            logger.error("Use either --video or --video-id, not both.")
            return 2
        return _main_video_id(args)

    candidates = _resolve_candidates(args)
    if not candidates:
        return 0

    # Local, no-credentials-needed validation first: a bad path or an
    # unsupported extension shouldn't require Azure config just to report.
    videos, failed = _filter_valid_videos(candidates)
    if not videos:
        return 2 if failed else 0

    try:
        settings = Settings.from_env()
    except ConfigError:
        logger.exception("Invalid Azure configuration")
        return 2

    client = VideoIndexerClient(settings)
    processed, skipped, process_failed = _process_all(
        client, videos, args.output_dir, args.force
    )
    failed += process_failed

    logger.info(
        "Done. %d processed, %d skipped, %d failed.", processed, skipped, failed
    )
    return 1 if failed and not processed and not skipped else 0


if __name__ == "__main__":
    sys.exit(main())
