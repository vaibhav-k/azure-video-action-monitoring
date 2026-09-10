"""
Surfaces every Azure AI Video Indexer insight category this project doesn't
already cover in action_analyzer.py -- faces, transcript/speakers/language,
topics, named entities (brands/locations/people), sentiments, emotions,
audio effects, shots, content moderation, observed-people body tracking
(with detected clothing, when the account has it), and a curated set of
interaction-shaped labels (e.g. "kiss") -- as one "capabilities" report.
Where action_analyzer.py answers "what actions/objects happened, and when",
this module answers "what does Video Indexer know about this video besides
that" -- the two are deliberately separate concerns, read by
explore_insights.py rather than the action-focused CLIs.

Faces vs. observed people, worth stating plainly since it trips people up:
these are two independent Video Indexer models that don't always agree on
headcount. `faces` tracks distinct faces and can lose lock on the same
physical person (a head turn, a profile view) and pick it back up as a new
"Unknown #N" entry; `observedPeople` tracks bodies and can stay locked onto
that same person the whole time. Cross-referencing the two by overlapping
time ranges (see ObservedPersonInsight's docstring) is how to notice one
real person got split into multiple face entries -- confirmed against a
real capture from this project's own account (2026-09-09), where two
"Unknown #N" faces with overlapping timestamps turned out to be one
continuously-tracked observed-person body.

Honest caveat up front: several of these buckets' exact field names are
confirmed against Microsoft's own schema docs (video-indexer-output-json-v2,
named-entities, insights-overview -- fetched 2026-09-09) but a few are not
independently verified against a real raw payload the way this project's
existing OCR/detectedObjects handling was (see action_analyzer.py's module
docstring for that verification). Marked below wherever a field name is a
best-effort guess rather than a confirmed one; every extractor is written
defensively (try the documented key, fall back to plausible alternates,
return an empty result rather than raising) precisely because of that
uncertainty -- the same "don't crash on an unexpected/absent bucket" stance
action_analyzer.py already takes for labels/keywords/ocr.

Every one of these insight categories except faces/transcript/speakers is
Advanced-preset-only or richer under Advanced (see README and
--indexing-preset's help text) -- a video indexed with the Default preset
will come back with most of these buckets simply absent, which every
extractor here treats as "nothing found", not an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from src.action_analyzer import (
    _extract_insights,
    _item_name,
    _iter_valid_instances,
    _timestamp_to_seconds,
    _video_duration_seconds,
)


@dataclass
class Appearance:
    """One timestamped occurrence of a face/topic/entity/etc. Deliberately
    the same shape as action_analyzer.DetectedAction's start/end/confidence,
    but kept as its own tiny type here rather than importing that one, since
    these aren't "actions" and forcing them into that dataclass would imply
    a closer relationship than actually exists."""

    start_seconds: float
    end_seconds: float
    confidence: float | None = None


@dataclass
class FaceInsight:
    name: str  # "Unknown #1" for an unrecognized face Video Indexer still tracked
    confidence: float | None
    thumbnail_id: str | None
    appearances: list[Appearance] = field(default_factory=list)

    @property
    def is_recognized(self) -> bool:
        """True when Video Indexer matched this face to a known identity (a
        celebrity, or someone registered in this account's Person Model)
        rather than merely detecting and tracking an unidentified face.
        There's no separate boolean field for this in the raw payload --
        every unmatched face is named "Unknown #N" and always reports
        confidence 0, confirmed against this project's own captures -- so a
        name that doesn't start with "Unknown" is the reliable signal.
        `confidence` alone isn't: a low-but-nonzero match confidence would
        still mean *some* identity was matched, just not confidently, so
        name is checked rather than thresholding confidence."""
        return not self.name.startswith("Unknown")


@dataclass
class TranscriptLine:
    text: str
    speaker: (
        str | None
    )  # e.g. "Speaker #1" -- resolved via `speakers`, see _speaker_names
    language: str | None
    confidence: float | None
    start_seconds: float
    end_seconds: float


@dataclass
class SpeakerInsight:
    name: str
    appearances: list[Appearance] = field(default_factory=list)


@dataclass
class TopicInsight:
    name: str
    category: str | None  # IPTC media-topic taxonomy name, when present
    confidence: float | None
    appearances: list[Appearance] = field(default_factory=list)


@dataclass
class NamedEntity:
    name: str
    kind: str  # "brand" | "location" | "person"
    confidence: float | None
    appearances: list[Appearance] = field(default_factory=list)


@dataclass
class SentimentInsight:
    sentiment_type: str  # "Positive" | "Neutral" | "Negative"
    score: float | None
    appearances: list[Appearance] = field(default_factory=list)


@dataclass
class EmotionInsight:
    emotion_type: str  # e.g. "Joy", "Sadness", "Anger", "Fear", "Surprise"
    appearances: list[Appearance] = field(default_factory=list)


@dataclass
class AudioEffectInsight:
    name: str  # e.g. "Silence", "Speech", "Music", "Crowd", "ApplauseAndCheering"
    confidence: float | None
    appearances: list[Appearance] = field(default_factory=list)


@dataclass
class ShotInsight:
    index: int
    start_seconds: float
    end_seconds: float
    keyframe_count: int


@dataclass
class ContentModerationInsight:
    visual_adult_score: float | None = None
    visual_racy_score: float | None = None
    textual_banned_words_count: int | None = None
    textual_banned_words_ratio: float | None = None

    @property
    def has_any_signal(self) -> bool:
        return any(
            v is not None
            for v in (
                self.visual_adult_score,
                self.visual_racy_score,
                self.textual_banned_words_count,
                self.textual_banned_words_ratio,
            )
        )


@dataclass
class ClothingItem:
    """One garment Video Indexer's detected-clothing feature identified on
    an observed person -- confirmed against a real payload from this
    project's own account (2026-09-09). Microsoft's docs describe detected
    clothing as needing a separate Face Recognition approval; whether this
    particular account already had that approval, or the requirement has
    since loosened, isn't something this project can determine from here --
    either way, treat its absence on a different account as "not available
    there" rather than a bug in this extractor. Coarse by design: a garment
    category (e.g. "sleeve", "pants", "skirtAndDress") plus an optional
    length property (e.g. "short"/"long") -- no color or style, confirmed
    against Microsoft's own detected-clothing doc."""

    type: str
    length: str | None = None


@dataclass
class ObservedPersonInsight:
    """One tracked human body, from Video Indexer's `observedPeople`
    insight -- a genuinely separate model from `faces` (FaceInsight above),
    not just another view of the same data. The two trackers don't always
    agree on headcount: a body can stay in continuous track through a head
    turn or profile view that breaks face tracking, so one real person can
    correspond to a single ObservedPersonInsight but multiple FaceInsight
    entries (e.g. two "Unknown #N" faces) -- cross-referencing the two by
    overlapping time ranges is how to notice this, since Video Indexer only
    links them explicitly when `matched_face_confidence` is set."""

    person_id: int
    thumbnail_id: str | None
    clothing: list[ClothingItem] = field(default_factory=list)
    # Set only when Video Indexer itself linked this body to a specific
    # tracked face (the `matchingFace` field) -- None means no link was
    # made, not "no face present"; see class docstring on why body/face
    # tracking can disagree on headcount.
    matched_face_name: str | None = None
    matched_face_confidence: float | None = None
    appearances: list[Appearance] = field(default_factory=list)

    @property
    def total_seen_seconds(self) -> float:
        return sum(a.end_seconds - a.start_seconds for a in self.appearances)


@dataclass
class InteractionSignal:
    """A generic label/keyword Video Indexer tagged that plausibly names a
    physical interaction between two or more people (e.g. "kiss", "hug",
    "handshake") -- NOT a dedicated interaction-detection insight, since
    Video Indexer has none. `_INTERACTION_LABEL_NAMES` is a hand-curated
    subset of its generic labels/keywords vocabulary seeded from what this
    project has actually observed in a real capture ("kiss"), not from an
    exhaustive Microsoft-documented list of every interaction-shaped label
    that could appear -- extend it as more are observed. Same honest-
    caveat strength as the rest of this project's label-based signals: a
    lead worth reviewing, not a confirmed claim about what happened."""

    name: str
    confidence: float | None
    appearances: list[Appearance] = field(default_factory=list)


_INTERACTION_LABEL_NAMES = frozenset(
    {"kiss", "hug", "embrace", "handshake", "fight", "dance"}
)


@dataclass
class CapabilitiesReport:
    """Every non-action insight Video Indexer returned for one video."""

    video_duration_seconds: float | None
    source_language: str | None
    faces: list[FaceInsight] = field(default_factory=list)
    transcript: list[TranscriptLine] = field(default_factory=list)
    speakers: list[SpeakerInsight] = field(default_factory=list)
    topics: list[TopicInsight] = field(default_factory=list)
    named_entities: list[NamedEntity] = field(default_factory=list)
    sentiments: list[SentimentInsight] = field(default_factory=list)
    emotions: list[EmotionInsight] = field(default_factory=list)
    audio_effects: list[AudioEffectInsight] = field(default_factory=list)
    shots: list[ShotInsight] = field(default_factory=list)
    content_moderation: ContentModerationInsight | None = None
    observed_people: list[ObservedPersonInsight] = field(default_factory=list)
    interaction_signals: list[InteractionSignal] = field(default_factory=list)

    @property
    def recognized_faces(self) -> list[FaceInsight]:
        """Faces Video Indexer matched to a known identity, rather than
        just tracking as "Unknown #N" -- see FaceInsight.is_recognized."""
        return [f for f in self.faces if f.is_recognized]

    @property
    def coverage(self) -> dict[str, int]:
        """How many items were found per capability -- 0 means either the
        preset didn't return that bucket (see module docstring) or nothing
        of that kind was actually present in the video. This is the "which
        capabilities did this video actually exercise" summary
        explore_insights.py leads its report with."""
        return {
            "faces": len(self.faces),
            "transcript_lines": len(self.transcript),
            "speakers": len(self.speakers),
            "topics": len(self.topics),
            "named_entities": len(self.named_entities),
            "sentiments": len(self.sentiments),
            "emotions": len(self.emotions),
            "audio_effects": len(self.audio_effects),
            "shots": len(self.shots),
            "content_moderation_signals": (
                1
                if (self.content_moderation and self.content_moderation.has_any_signal)
                else 0
            ),
            "observed_people": len(self.observed_people),
            "interaction_signals": len(self.interaction_signals),
        }


def _appearances(item: dict[str, Any]) -> list[Appearance]:
    """`instances` is every other bucket's shape (see action_analyzer.py);
    `appearances` is a documented alternate for namedPeople/namedLocations
    that this project hasn't independently verified against a real payload
    (see module docstring) -- checked second, and its differently-named
    start/end fields (startSeconds/endSeconds, falling back to
    startTime/endTime as HH:MM:SS timestamps) are handled here rather than
    in `_iter_valid_instances`, which only knows the `instances` shape."""
    raw_instances = item.get("instances")
    if isinstance(raw_instances, list):
        return [
            Appearance(start_seconds=s, end_seconds=e, confidence=c)
            for s, e, c in _iter_valid_instances(item)
        ]

    raw_appearances = item.get("appearances")
    if not isinstance(raw_appearances, list):
        return []

    result: list[Appearance] = []
    for raw in cast("list[Any]", raw_appearances):
        appearance = cast("dict[str, Any]", raw)
        start = appearance.get("startSeconds")
        end = appearance.get("endSeconds")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            result.append(
                Appearance(start_seconds=float(start), end_seconds=float(end))
            )
            continue
        start_time, end_time = appearance.get("startTime"), appearance.get("endTime")
        if start_time is not None and end_time is not None:
            try:
                result.append(
                    Appearance(
                        start_seconds=_timestamp_to_seconds(start_time),
                        end_seconds=_timestamp_to_seconds(end_time),
                    )
                )
            except (TypeError, ValueError):
                continue
    return result


def _faces(insights: dict[str, Any]) -> list[FaceInsight]:
    bucket = insights.get("faces")
    if not isinstance(bucket, list):
        return []
    faces: list[FaceInsight] = []
    for raw in cast("list[Any]", bucket):
        item = cast("dict[str, Any]", raw)
        name = item.get("name") or item.get("title") or "Unknown"
        confidence = item.get("confidence")
        faces.append(
            FaceInsight(
                name=name,
                confidence=float(confidence) if confidence is not None else None,
                thumbnail_id=item.get("thumbnailId"),
                appearances=_appearances(item),
            )
        )
    return faces


def _speaker_names(insights: dict[str, Any]) -> dict[Any, str]:
    """Map a transcript item's `speakerId` to a human-readable name from the
    `speakers` bucket, when present -- falls back to "Speaker #<id>" in
    `_transcript` when this bucket is absent (some API versions only expose
    the inline speakerId with no separate `speakers` bucket -- see module
    docstring)."""
    bucket = insights.get("speakers")
    if not isinstance(bucket, list):
        return {}
    names: dict[Any, str] = {}
    for raw in cast("list[Any]", bucket):
        item = cast("dict[str, Any]", raw)
        speaker_id = item.get("id")
        name = item.get("name")
        if speaker_id is not None and name:
            names[speaker_id] = name
    return names


def _speakers(insights: dict[str, Any]) -> list[SpeakerInsight]:
    bucket = insights.get("speakers")
    if not isinstance(bucket, list):
        return []
    speakers: list[SpeakerInsight] = []
    for raw in cast("list[Any]", bucket):
        item = cast("dict[str, Any]", raw)
        speaker_id = item.get("id")
        name = item.get("name") or (
            f"Speaker #{speaker_id}" if speaker_id is not None else "Speaker"
        )
        speakers.append(SpeakerInsight(name=name, appearances=_appearances(item)))
    return speakers


def _transcript(insights: dict[str, Any]) -> list[TranscriptLine]:
    bucket = insights.get("transcript")
    if not isinstance(bucket, list):
        return []
    speaker_names = _speaker_names(insights)
    lines: list[TranscriptLine] = []
    for raw in cast("list[Any]", bucket):
        item = cast("dict[str, Any]", raw)
        text = item.get("text")
        if not text:
            continue
        speaker_id = item.get("speakerId")
        speaker = (
            speaker_names.get(speaker_id, f"Speaker #{speaker_id}")
            if speaker_id is not None
            else None
        )
        confidence = item.get("confidence")
        for start_s, end_s, _ in _iter_valid_instances(item):
            lines.append(
                TranscriptLine(
                    text=text,
                    speaker=speaker,
                    language=item.get("language"),
                    confidence=float(confidence) if confidence is not None else None,
                    start_seconds=start_s,
                    end_seconds=end_s,
                )
            )
    lines.sort(key=lambda t: t.start_seconds)
    return lines


def _topics(insights: dict[str, Any]) -> list[TopicInsight]:
    bucket = insights.get("topics")
    if not isinstance(bucket, list):
        return []
    topics: list[TopicInsight] = []
    for raw in cast("list[Any]", bucket):
        item = cast("dict[str, Any]", raw)
        name = item.get("name")
        if not name:
            continue
        confidence = item.get("confidence")
        topics.append(
            TopicInsight(
                name=name,
                category=item.get("iptcName") or item.get("category"),
                confidence=float(confidence) if confidence is not None else None,
                appearances=_appearances(item),
            )
        )
    return topics


def _named_entities(insights: dict[str, Any]) -> list[NamedEntity]:
    """`brands`/`namedLocations`/`namedPeople` are three separate top-level
    buckets (confirmed against Microsoft's named-entities doc), merged here
    into one list tagged with `kind` -- a video's OCR/brand mentions and its
    spoken references to people/places are conceptually the same "named
    entity" capability, just extracted from different source material
    (transcript+OCR for all three, per Video Indexer's own NLP pipeline)."""
    entities: list[NamedEntity] = []
    for bucket_name, kind in (
        ("brands", "brand"),
        ("namedLocations", "location"),
        ("namedPeople", "person"),
    ):
        bucket = insights.get(bucket_name)
        if not isinstance(bucket, list):
            continue
        for raw in cast("list[Any]", bucket):
            item = cast("dict[str, Any]", raw)
            name = item.get("name")
            if not name:
                continue
            confidence = item.get("confidence")
            entities.append(
                NamedEntity(
                    name=name,
                    kind=kind,
                    confidence=float(confidence) if confidence is not None else None,
                    appearances=_appearances(item),
                )
            )
    return entities


def _sentiments(insights: dict[str, Any]) -> list[SentimentInsight]:
    bucket = insights.get("sentiments")
    if not isinstance(bucket, list):
        return []
    sentiments: list[SentimentInsight] = []
    for raw in cast("list[Any]", bucket):
        item = cast("dict[str, Any]", raw)
        sentiment_type = item.get("sentimentType") or item.get("sentimentKey")
        if not sentiment_type:
            continue
        # `averageScore` is sentiments' own confidence-like field (no plain
        # "confidence" key confirmed for this bucket) -- fall back to
        # "confidence" defensively in case a different API version uses it.
        score = item.get("averageScore", item.get("confidence"))
        sentiments.append(
            SentimentInsight(
                sentiment_type=sentiment_type,
                score=float(score) if score is not None else None,
                appearances=_appearances(item),
            )
        )
    return sentiments


def _emotions(insights: dict[str, Any]) -> list[EmotionInsight]:
    bucket = insights.get("emotions")
    if not isinstance(bucket, list):
        return []
    emotions: list[EmotionInsight] = []
    for raw in cast("list[Any]", bucket):
        item = cast("dict[str, Any]", raw)
        emotion_type = item.get("type") or item.get("name")
        if not emotion_type:
            continue
        emotions.append(
            EmotionInsight(emotion_type=emotion_type, appearances=_appearances(item))
        )
    return emotions


def _audio_effects(insights: dict[str, Any]) -> list[AudioEffectInsight]:
    bucket = insights.get("audioEffects")
    if not isinstance(bucket, list):
        return []
    effects: list[AudioEffectInsight] = []
    for raw in cast("list[Any]", bucket):
        item = cast("dict[str, Any]", raw)
        name = item.get("type") or item.get("name")
        if not name:
            continue
        confidence = item.get("confidence")
        effects.append(
            AudioEffectInsight(
                name=name,
                confidence=float(confidence) if confidence is not None else None,
                appearances=_appearances(item),
            )
        )
    return effects


def _shots(insights: dict[str, Any]) -> list[ShotInsight]:
    """`scenes` as a distinct top-level bucket wasn't confirmed against
    Microsoft's docs (it may not exist as a separate key in this API
    version) -- `shots` was confirmed, so that's the one this project reads;
    per-shot keyframe counts come from each shot's nested `keyFrames`."""
    bucket = insights.get("shots")
    if not isinstance(bucket, list):
        return []
    shots: list[ShotInsight] = []
    for index, raw in enumerate(cast("list[Any]", bucket)):
        item = cast("dict[str, Any]", raw)
        spans = list(_iter_valid_instances(item))
        if not spans:
            continue
        start_s = min(s for s, _, _ in spans)
        end_s = max(e for _, e, _ in spans)
        keyframes = item.get("keyFrames")
        keyframe_count = (
            len(cast("list[Any]", keyframes)) if isinstance(keyframes, list) else 0
        )
        shots.append(
            ShotInsight(
                index=index,
                start_seconds=start_s,
                end_seconds=end_s,
                keyframe_count=keyframe_count,
            )
        )
    return shots


def _first_moderation_item(bucket: Any) -> dict[str, Any] | None:
    """`visualContentModeration`/`textualContentModeration` were assumed to
    always be a list of items (`[0]` taken as "the" result) -- a real
    payload from this project's own account (2026-09-09) showed
    `textualContentModeration` returned as a single dict instead, which the
    old list-only check silently treated as "bucket absent" rather than
    parsing it (no crash, just quietly dropped real data -- exactly the
    kind of gap this module's defensive-parsing stance exists to catch).
    Handles both shapes now."""
    if isinstance(bucket, dict):
        return cast("dict[str, Any]", bucket)
    if isinstance(bucket, list) and bucket:
        return cast("dict[str, Any]", bucket[0])
    return None


def _content_moderation(insights: dict[str, Any]) -> ContentModerationInsight | None:
    visual = _first_moderation_item(insights.get("visualContentModeration"))
    textual = _first_moderation_item(insights.get("textualContentModeration"))
    if visual is None and textual is None:
        return None

    result = ContentModerationInsight()
    if visual is not None:
        adult, racy = visual.get("adultScore"), visual.get("racyScore")
        result.visual_adult_score = (
            float(adult) if isinstance(adult, (int, float)) else None
        )
        result.visual_racy_score = (
            float(racy) if isinstance(racy, (int, float)) else None
        )
    if textual is not None:
        count, ratio = textual.get("bannedWordsCount"), textual.get("bannedWordsRatio")
        result.textual_banned_words_count = (
            int(count) if isinstance(count, (int, float)) else None
        )
        result.textual_banned_words_ratio = (
            float(ratio) if isinstance(ratio, (int, float)) else None
        )
    return result


def _face_id_to_name(insights: dict[str, Any]) -> dict[Any, str]:
    """Map a `faces` item's raw `id` to its `name`, for resolving
    `observedPeople[].matchingFace.id` back to a human-readable face name
    in `_observed_people`. Kept separate from FaceInsight, which doesn't
    carry the raw numeric id -- this mapping is only ever used internally,
    right after extraction, while both buckets' raw items are on hand."""
    bucket = insights.get("faces")
    if not isinstance(bucket, list):
        return {}
    mapping: dict[Any, str] = {}
    for raw in cast("list[Any]", bucket):
        item = cast("dict[str, Any]", raw)
        face_id = item.get("id")
        name = item.get("name") or item.get("title")
        if face_id is not None and name:
            mapping[face_id] = name
    return mapping


def _clothing_items(raw_clothing: Any) -> list[ClothingItem]:
    if not isinstance(raw_clothing, list):
        return []
    items: list[ClothingItem] = []
    for raw in cast("list[Any]", raw_clothing):
        item = cast("dict[str, Any]", raw)
        clothing_type = item.get("type")
        if not clothing_type:
            continue
        properties = item.get("properties")
        length = (
            cast("dict[str, Any]", properties).get("length")
            if isinstance(properties, dict)
            else None
        )
        items.append(ClothingItem(type=clothing_type, length=length))
    return items


def _observed_people(insights: dict[str, Any]) -> list[ObservedPersonInsight]:
    """`observedPeople` -- Video Indexer's body/person tracker, a genuinely
    separate model from `faces` (see ObservedPersonInsight docstring).
    `clothing`/`matchingFace` confirmed against a real payload from this
    project's own account (2026-09-09) -- see ClothingItem's docstring on
    why that account had detected clothing enabled at all. Absence on
    another account just means "not available there", not a parsing bug;
    every field is read defensively regardless."""
    bucket = insights.get("observedPeople")
    if not isinstance(bucket, list):
        return []
    face_names = _face_id_to_name(insights)

    people: list[ObservedPersonInsight] = []
    for raw in cast("list[Any]", bucket):
        item = cast("dict[str, Any]", raw)
        person_id = item.get("id")
        if person_id is None:
            continue
        matched_face_name: str | None = None
        matched_face_confidence: float | None = None
        matching_face = item.get("matchingFace")
        if isinstance(matching_face, dict):
            matching_face_dict = cast("dict[str, Any]", matching_face)
            matched_face_name = face_names.get(matching_face_dict.get("id"))
            confidence = matching_face_dict.get("confidence")
            matched_face_confidence = (
                float(confidence) if isinstance(confidence, (int, float)) else None
            )
        people.append(
            ObservedPersonInsight(
                person_id=int(person_id),
                thumbnail_id=item.get("thumbnailId"),
                clothing=_clothing_items(item.get("clothing")),
                matched_face_name=matched_face_name,
                matched_face_confidence=matched_face_confidence,
                appearances=_appearances(item),
            )
        )
    return people


def _interaction_signals(insights: dict[str, Any]) -> list[InteractionSignal]:
    """A curated subset of generic `labels`/`keywords` that name a physical
    interaction between people -- see InteractionSignal's docstring on why
    this isn't a dedicated Video Indexer insight. These items' `confidence`
    lives per-instance, not at the top level (confirmed against a real
    "kiss" label capture), so the reported confidence is the max across
    that label's instances when no top-level value is present."""
    signals: list[InteractionSignal] = []
    for bucket_name in ("labels", "keywords"):
        bucket = insights.get(bucket_name)
        if not isinstance(bucket, list):
            continue
        for raw in cast("list[Any]", bucket):
            item = cast("dict[str, Any]", raw)
            name = _item_name(item)
            if name.lower() not in _INTERACTION_LABEL_NAMES:
                continue
            appearances = _appearances(item)
            confidence = item.get("confidence")
            if confidence is None:
                scored = [a.confidence for a in appearances if a.confidence is not None]
                confidence = max(scored) if scored else None
            signals.append(
                InteractionSignal(
                    name=name,
                    confidence=float(confidence) if confidence is not None else None,
                    appearances=appearances,
                )
            )
    signals.sort(key=lambda s: s.appearances[0].start_seconds if s.appearances else 0.0)
    return signals


def analyze_capabilities(index_payload: dict[str, Any]) -> CapabilitiesReport:
    """Extract every non-action insight Video Indexer returned for one
    video. Every bucket this reads is Advanced-preset-only or richer under
    Advanced except faces/transcript/speakers -- see --indexing-preset's
    help text on explore_insights.py and this module's docstring. Absent or
    empty buckets simply produce empty lists/None fields; nothing here
    raises for a Default-preset payload missing most of these."""
    insights = _extract_insights(index_payload)
    duration = _video_duration_seconds(index_payload, insights)

    source_language = insights.get("sourceLanguage") or insights.get("language")

    return CapabilitiesReport(
        video_duration_seconds=duration,
        source_language=source_language if isinstance(source_language, str) else None,
        faces=_faces(insights),
        transcript=_transcript(insights),
        speakers=_speakers(insights),
        topics=_topics(insights),
        named_entities=_named_entities(insights),
        sentiments=_sentiments(insights),
        emotions=_emotions(insights),
        audio_effects=_audio_effects(insights),
        shots=_shots(insights),
        content_moderation=_content_moderation(insights),
        observed_people=_observed_people(insights),
        interaction_signals=_interaction_signals(insights),
    )
