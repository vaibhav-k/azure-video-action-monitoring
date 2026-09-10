"""
Renders a CapabilitiesReport (src/insights_explorer.py) as: a console
summary, a machine-readable JSON file, and a small self-contained HTML page
-- the same three-output pattern report.py already uses for action reports,
kept in its own module since a capabilities report has a genuinely
different shape (named/typed insight categories, not one timestamped
occurrence list) rather than forcing it into report.py's existing
functions.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from .insights_explorer import CapabilitiesReport
from .report import _fmt_seconds


def _fmt_score(value: float | None) -> str:
    """Render an optional confidence/score as "0.xx", or 'n/a' when unknown."""
    return f"{value:.2f}" if value is not None else "n/a"


_CAPABILITY_LABELS: dict[str, str] = {
    "faces": "Faces",
    "transcript_lines": "Transcript lines",
    "speakers": "Speakers",
    "topics": "Topics",
    "named_entities": "Named entities (brands/locations/people)",
    "sentiments": "Sentiments",
    "emotions": "Emotions",
    "audio_effects": "Audio effects",
    "shots": "Shots",
    "content_moderation_signals": "Content moderation signals",
    "observed_people": "Observed people (body tracking)",
    "interaction_signals": "Interaction signals (e.g. kiss, hug)",
}


def _fmt_matched_face(name: str | None, confidence: float | None) -> str:
    """ "  matched face: Jane (confidence=0.91)" when `name` is set (the
    observed-people table's per-row face-match suffix), else ""."""
    if not name:
        return ""
    confidence_suffix = (
        f" (confidence={confidence:.2f})" if confidence is not None else ""
    )
    return f"  matched face: {name}{confidence_suffix}"


def _console_coverage(report: CapabilitiesReport) -> list[str]:
    lines = [
        (
            "Coverage (0 = not present in this payload -- see README on "
            "--indexing-preset for which of these need Advanced):"
        ),
    ]
    for key, count in report.coverage.items():
        lines.append(f"  {_CAPABILITY_LABELS[key]:<42} {count}")
    lines.append("")
    return lines


def _console_faces(report: CapabilitiesReport) -> list[str]:
    if not report.faces:
        return []
    lines = [f"Faces ({len(report.faces)}, {len(report.recognized_faces)} recognized):"]
    for f in report.faces:
        conf = _fmt_score(f.confidence)
        tag = " [recognized]" if f.is_recognized else ""
        spans = (
            ", ".join(
                f"{_fmt_seconds(a.start_seconds)}-{_fmt_seconds(a.end_seconds)}"
                for a in f.appearances
            )
            or "no timing data"
        )
        lines.append(
            f"  {f.name}{tag}  confidence={conf}  appearances={len(f.appearances)} ({spans})"
        )
    lines.append("")
    return lines


def _console_observed_people(report: CapabilitiesReport) -> list[str]:
    if not report.observed_people:
        return []
    lines = [f"Observed people / body tracking ({len(report.observed_people)}):"]
    for p in report.observed_people:
        face = _fmt_matched_face(p.matched_face_name, p.matched_face_confidence)
        clothing = (
            ", ".join(
                f"{c.type}" + (f" ({c.length})" if c.length else "") for c in p.clothing
            )
            or "no clothing data"
        )
        lines.append(
            f"  Person #{p.person_id}  seen={_fmt_seconds(p.total_seen_seconds)}"
            f"  clothing: {clothing}{face}"
        )
    lines.append("")
    return lines


def _console_interaction_signals(report: CapabilitiesReport) -> list[str]:
    if not report.interaction_signals:
        return []
    lines = [
        f"Interaction signals ({len(report.interaction_signals)}):",
        (
            "  (a curated subset of generic labels that name a physical "
            "interaction, e.g. 'kiss' -- not a dedicated interaction "
            "detector; treat as a lead worth reviewing, not confirmation)"
        ),
    ]
    for s in report.interaction_signals:
        conf = _fmt_score(s.confidence)
        spans = (
            ", ".join(
                f"{_fmt_seconds(a.start_seconds)}-{_fmt_seconds(a.end_seconds)}"
                for a in s.appearances
            )
            or "no timing data"
        )
        lines.append(f"  {s.name}  confidence={conf}  ({spans})")
    lines.append("")
    return lines


def _console_transcript(report: CapabilitiesReport) -> list[str]:
    if not report.transcript:
        return []
    lines = [f"Transcript ({len(report.transcript)} lines):"]
    for t in report.transcript:
        speaker = f"{t.speaker}: " if t.speaker else ""
        lines.append(
            f"  [{_fmt_seconds(t.start_seconds)}-{_fmt_seconds(t.end_seconds)}] {speaker}{t.text}"
        )
    lines.append("")
    return lines


def _console_topics(report: CapabilitiesReport) -> list[str]:
    if not report.topics:
        return []
    lines = [f"Topics ({len(report.topics)}):"]
    for t in report.topics:
        conf = _fmt_score(t.confidence)
        category = f" [{t.category}]" if t.category else ""
        lines.append(f"  {t.name}{category}  confidence={conf}")
    lines.append("")
    return lines


def _console_named_entities(report: CapabilitiesReport) -> list[str]:
    if not report.named_entities:
        return []
    lines = [f"Named entities ({len(report.named_entities)}):"]
    for e in report.named_entities:
        conf = _fmt_score(e.confidence)
        lines.append(
            f"  [{e.kind}] {e.name}  confidence={conf}  appearances={len(e.appearances)}"
        )
    lines.append("")
    return lines


def _console_sentiments(report: CapabilitiesReport) -> list[str]:
    if not report.sentiments:
        return []
    lines = [f"Sentiments ({len(report.sentiments)}):"]
    for s in report.sentiments:
        score = _fmt_score(s.score)
        lines.append(
            f"  {s.sentiment_type}  score={score}  appearances={len(s.appearances)}"
        )
    lines.append("")
    return lines


def _console_emotions(report: CapabilitiesReport) -> list[str]:
    if not report.emotions:
        return []
    lines = [f"Emotions ({len(report.emotions)}):"]
    for e in report.emotions:
        lines.append(f"  {e.emotion_type}  appearances={len(e.appearances)}")
    lines.append("")
    return lines


def _console_audio_effects(report: CapabilitiesReport) -> list[str]:
    if not report.audio_effects:
        return []
    lines = [f"Audio effects ({len(report.audio_effects)}):"]
    for a in report.audio_effects:
        conf = _fmt_score(a.confidence)
        lines.append(f"  {a.name}  confidence={conf}  appearances={len(a.appearances)}")
    lines.append("")
    return lines


def _console_shots(report: CapabilitiesReport) -> list[str]:
    if not report.shots:
        return []
    lines = [f"Shots ({len(report.shots)}):"]
    for s in report.shots:
        lines.append(
            f"  #{s.index}  [{_fmt_seconds(s.start_seconds)}-{_fmt_seconds(s.end_seconds)}]  "
            f"keyframes={s.keyframe_count}"
        )
    lines.append("")
    return lines


def _console_content_moderation(report: CapabilitiesReport) -> list[str]:
    cm = report.content_moderation
    if not cm or not cm.has_any_signal:
        return []
    lines = ["Content moderation:"]
    if cm.visual_adult_score is not None:
        lines.append(f"  visual adult score:  {cm.visual_adult_score:.3f}")
    if cm.visual_racy_score is not None:
        lines.append(f"  visual racy score:   {cm.visual_racy_score:.3f}")
    if cm.textual_banned_words_count is not None:
        lines.append(f"  banned words count:  {cm.textual_banned_words_count}")
    if cm.textual_banned_words_ratio is not None:
        lines.append(f"  banned words ratio:  {cm.textual_banned_words_ratio:.3f}")
    lines.append("")
    return lines


def _console_empty_notice(report: CapabilitiesReport) -> list[str]:
    if any(report.coverage.values()):
        return []
    return [
        (
            "Nothing found in any of these categories. If this video was "
            "indexed with --indexing-preset Default, that's expected -- "
            "most of these are Advanced-only. Re-run with "
            "--indexing-preset Advanced (a fresh upload; an already-"
            "indexed --video-id keeps whatever preset it was first "
            "uploaded with)."
        ),
    ]


# One function per report section, in the order they're rendered. Each
# returns the lines for its section (including the trailing blank line
# separator), or [] when that section has nothing to show -- keeping
# to_console_text_capabilities itself a flat, linear dispatch list instead
# of one long function with a branch per capability.
_CONSOLE_SECTIONS = (
    _console_faces,
    _console_observed_people,
    _console_interaction_signals,
    _console_transcript,
    _console_topics,
    _console_named_entities,
    _console_sentiments,
    _console_emotions,
    _console_audio_effects,
    _console_shots,
    _console_content_moderation,
    _console_empty_notice,
)


def to_console_text_capabilities(report: CapabilitiesReport, video_name: str) -> str:
    """Human-readable summary: a coverage line per capability (0 means
    either the preset didn't return that bucket or nothing of that kind was
    in the video -- see insights_explorer.py's module docstring), followed
    by a section per capability that found anything."""
    lines = [
        f"Capabilities report for '{video_name}'",
        f"Video duration: {_fmt_seconds(report.video_duration_seconds)}",
        f"Source language: {report.source_language or 'unknown'}",
        "",
    ]
    lines.extend(_console_coverage(report))
    for section in _CONSOLE_SECTIONS:
        lines.extend(section(report))
    return "\n".join(lines)


def to_dict_capabilities(report: CapabilitiesReport, video_name: str) -> dict[str, Any]:
    """Render the capabilities report as a plain, JSON-serializable dict."""
    cm = report.content_moderation
    return {
        "video_name": video_name,
        "video_duration_seconds": report.video_duration_seconds,
        "source_language": report.source_language,
        "coverage": report.coverage,
        "faces": [
            {
                "name": f.name,
                "confidence": f.confidence,
                "thumbnail_id": f.thumbnail_id,
                "is_recognized": f.is_recognized,
                "appearances": [
                    {"start_seconds": a.start_seconds, "end_seconds": a.end_seconds}
                    for a in f.appearances
                ],
            }
            for f in report.faces
        ],
        "transcript": [
            {
                "text": t.text,
                "speaker": t.speaker,
                "language": t.language,
                "confidence": t.confidence,
                "start_seconds": t.start_seconds,
                "end_seconds": t.end_seconds,
            }
            for t in report.transcript
        ],
        "speakers": [
            {
                "name": s.name,
                "appearances": [
                    {"start_seconds": a.start_seconds, "end_seconds": a.end_seconds}
                    for a in s.appearances
                ],
            }
            for s in report.speakers
        ],
        "topics": [
            {
                "name": t.name,
                "category": t.category,
                "confidence": t.confidence,
                "appearances": [
                    {"start_seconds": a.start_seconds, "end_seconds": a.end_seconds}
                    for a in t.appearances
                ],
            }
            for t in report.topics
        ],
        "named_entities": [
            {
                "name": e.name,
                "kind": e.kind,
                "confidence": e.confidence,
                "appearances": [
                    {"start_seconds": a.start_seconds, "end_seconds": a.end_seconds}
                    for a in e.appearances
                ],
            }
            for e in report.named_entities
        ],
        "sentiments": [
            {
                "sentiment_type": s.sentiment_type,
                "score": s.score,
                "appearances": [
                    {"start_seconds": a.start_seconds, "end_seconds": a.end_seconds}
                    for a in s.appearances
                ],
            }
            for s in report.sentiments
        ],
        "emotions": [
            {
                "emotion_type": e.emotion_type,
                "appearances": [
                    {"start_seconds": a.start_seconds, "end_seconds": a.end_seconds}
                    for a in e.appearances
                ],
            }
            for e in report.emotions
        ],
        "audio_effects": [
            {
                "name": a.name,
                "confidence": a.confidence,
                "appearances": [
                    {"start_seconds": ap.start_seconds, "end_seconds": ap.end_seconds}
                    for ap in a.appearances
                ],
            }
            for a in report.audio_effects
        ],
        "shots": [
            {
                "index": s.index,
                "start_seconds": s.start_seconds,
                "end_seconds": s.end_seconds,
                "keyframe_count": s.keyframe_count,
            }
            for s in report.shots
        ],
        "content_moderation": (
            {
                "visual_adult_score": cm.visual_adult_score,
                "visual_racy_score": cm.visual_racy_score,
                "textual_banned_words_count": cm.textual_banned_words_count,
                "textual_banned_words_ratio": cm.textual_banned_words_ratio,
            }
            if cm
            else None
        ),
        "observed_people": [
            {
                "person_id": p.person_id,
                "thumbnail_id": p.thumbnail_id,
                "clothing": [{"type": c.type, "length": c.length} for c in p.clothing],
                "matched_face_name": p.matched_face_name,
                "matched_face_confidence": p.matched_face_confidence,
                "total_seen_seconds": p.total_seen_seconds,
                "appearances": [
                    {"start_seconds": a.start_seconds, "end_seconds": a.end_seconds}
                    for a in p.appearances
                ],
            }
            for p in report.observed_people
        ],
        "interaction_signals": [
            {
                "name": s.name,
                "confidence": s.confidence,
                "appearances": [
                    {"start_seconds": a.start_seconds, "end_seconds": a.end_seconds}
                    for a in s.appearances
                ],
            }
            for s in report.interaction_signals
        ],
    }


def write_json_capabilities(
    report: CapabilitiesReport, video_name: str, path: Path
) -> None:
    """Write the capabilities report as pretty-printed JSON to `path`."""
    path.write_text(
        json.dumps(to_dict_capabilities(report, video_name), indent=2), encoding="utf-8"
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Capabilities Report</title>
<style>
  body {{
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
    margin: 2rem; color: #1a1a2e; background: #fafafa;
  }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
  .summary {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0 2rem; }}
  .stat {{
    background: #fff; border: 1px solid #e2e2e8; border-radius: 8px;
    padding: 0.75rem 1rem; min-width: 150px;
  }}
  .stat .label {{
    font-size: 0.75rem; color: #666;
    text-transform: uppercase; letter-spacing: 0.03em;
  }}
  .stat .value {{ font-size: 1.3rem; font-weight: 600; }}
  .stat.zero .value {{ color: #999; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; }}
  th, td {{
    border-bottom: 1px solid #eee; padding: 0.5rem 0.75rem;
    text-align: left; font-size: 0.9rem;
  }}
  th {{ background: #f2f2f7; }}
  .empty {{ padding: 1rem; background: #fff3cd; border-radius: 8px; }}
  .pill {{
    display: inline-block; background: #eef; border-radius: 999px;
    padding: 0.15rem 0.6rem; margin: 0.15rem; font-size: 0.85rem;
  }}
  .transcript-line {{ margin: 0.3rem 0; font-size: 0.9rem; }}
  .transcript-time {{ color: #888; font-variant-numeric: tabular-nums; }}
  .transcript-speaker {{ font-weight: 600; }}
</style>
</head>
<body>
  <h1>Capabilities detected in {video_name}</h1>
  <div class="summary">
    <div class="stat"><div class="label">Duration</div><div class="value">{duration}</div></div>
    <div class="stat"><div class="label">Language</div><div class="value">{language}</div></div>
    {coverage_stats}
  </div>
  {sections}
</body>
</html>
"""


def _html_coverage_stats(report: CapabilitiesReport) -> str:
    return "".join(
        f'<div class="stat{" zero" if count == 0 else ""}">'
        f'<div class="label">{escape(_CAPABILITY_LABELS[key])}</div>'
        f'<div class="value">{count}</div></div>'
        for key, count in report.coverage.items()
    )


def _html_faces(report: CapabilitiesReport) -> str:
    if not report.faces:
        return ""
    rows = "".join(
        f"<tr><td>{escape(f.name)}</td>"
        f"<td>{'✓' if f.is_recognized else '—'}</td>"
        f"<td>{_fmt_score(f.confidence)}</td>"
        f"<td>{len(f.appearances)}</td>"
        f"<td>{escape(', '.join(f'{_fmt_seconds(a.start_seconds)}-{_fmt_seconds(a.end_seconds)}' for a in f.appearances) or '-')}</td>"
        "</tr>"
        for f in report.faces
    )
    return (
        f"<h2>Faces ({len(report.faces)}, {len(report.recognized_faces)} recognized)</h2>"
        "<table><thead><tr><th>Name</th><th>Recognized</th><th>Confidence</th>"
        f"<th>Appearances</th><th>Timing</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _html_observed_people(report: CapabilitiesReport) -> str:
    if not report.observed_people:
        return ""
    rows = "".join(
        f"<tr><td>#{p.person_id}</td>"
        f"<td>{_fmt_seconds(p.total_seen_seconds)}</td>"
        f"<td>{escape(', '.join(c.type + (f' ({c.length})' if c.length else '') for c in p.clothing) or '-')}</td>"
        f"<td>{escape(p.matched_face_name) if p.matched_face_name else '-'}</td>"
        "</tr>"
        for p in report.observed_people
    )
    return (
        f"<h2>Observed people / body tracking ({len(report.observed_people)})</h2>"
        '<p style="color:#666;font-size:0.85rem">A separate body-tracking '
        "model from the face table above -- it can stay locked onto one "
        "person through a head turn or profile view that breaks face "
        "tracking, so cross-check against the Faces table by overlapping "
        "timing if the counts don't match.</p>"
        "<table><thead><tr><th>Person</th><th>Total seen</th><th>Clothing</th>"
        f"<th>Matched face</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _html_interaction_signals(report: CapabilitiesReport) -> str:
    if not report.interaction_signals:
        return ""
    rows = "".join(
        f"<tr><td>{escape(s.name)}</td>"
        f"<td>{_fmt_score(s.confidence)}</td>"
        f"<td>{escape(', '.join(f'{_fmt_seconds(a.start_seconds)}-{_fmt_seconds(a.end_seconds)}' for a in s.appearances) or '-')}</td>"
        "</tr>"
        for s in report.interaction_signals
    )
    return (
        f"<h2>Interaction signals ({len(report.interaction_signals)})</h2>"
        '<p style="color:#666;font-size:0.85rem">A curated subset of '
        'generic labels that name a physical interaction (e.g. "kiss") '
        "-- not a dedicated interaction detector; a lead worth reviewing, "
        "not confirmation.</p>"
        "<table><thead><tr><th>Signal</th><th>Confidence</th>"
        f"<th>Timing</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _html_transcript(report: CapabilitiesReport) -> str:
    if not report.transcript:
        return ""
    lines_html = "".join(
        '<div class="transcript-line">'
        f'<span class="transcript-time">[{_fmt_seconds(t.start_seconds)}-{_fmt_seconds(t.end_seconds)}]</span> '
        + (
            f'<span class="transcript-speaker">{escape(t.speaker)}:</span> '
            if t.speaker
            else ""
        )
        + f"{escape(t.text)}</div>"
        for t in report.transcript
    )
    return f"<h2>Transcript ({len(report.transcript)} lines)</h2>{lines_html}"


def _html_topics(report: CapabilitiesReport) -> str:
    if not report.topics:
        return ""
    pills = "".join(
        f'<span class="pill">{escape(t.name)}'
        + (f" ({escape(t.category)})" if t.category else "")
        + "</span>"
        for t in report.topics
    )
    return f"<h2>Topics ({len(report.topics)})</h2><div>{pills}</div>"


def _html_named_entities(report: CapabilitiesReport) -> str:
    if not report.named_entities:
        return ""
    pills = "".join(
        f'<span class="pill">[{escape(e.kind)}] {escape(e.name)}</span>'
        for e in report.named_entities
    )
    return f"<h2>Named entities ({len(report.named_entities)})</h2><div>{pills}</div>"


def _html_sentiments(report: CapabilitiesReport) -> str:
    if not report.sentiments:
        return ""
    rows = "".join(
        f"<tr><td>{escape(s.sentiment_type)}</td>"
        f"<td>{_fmt_score(s.score)}</td>"
        f"<td>{len(s.appearances)}</td></tr>"
        for s in report.sentiments
    )
    return (
        f"<h2>Sentiments ({len(report.sentiments)})</h2>"
        "<table><thead><tr><th>Type</th><th>Score</th>"
        f"<th>Appearances</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _html_emotions(report: CapabilitiesReport) -> str:
    if not report.emotions:
        return ""
    pills = "".join(
        f'<span class="pill">{escape(e.emotion_type)} ({len(e.appearances)})</span>'
        for e in report.emotions
    )
    return f"<h2>Emotions ({len(report.emotions)})</h2><div>{pills}</div>"


def _html_audio_effects(report: CapabilitiesReport) -> str:
    if not report.audio_effects:
        return ""
    pills = "".join(
        f'<span class="pill">{escape(a.name)} ({len(a.appearances)})</span>'
        for a in report.audio_effects
    )
    return f"<h2>Audio effects ({len(report.audio_effects)})</h2><div>{pills}</div>"


def _html_shots(report: CapabilitiesReport) -> str:
    if not report.shots:
        return ""
    rows = "".join(
        f"<tr><td>{s.index}</td><td>{_fmt_seconds(s.start_seconds)}</td>"
        f"<td>{_fmt_seconds(s.end_seconds)}</td><td>{s.keyframe_count}</td></tr>"
        for s in report.shots
    )
    return (
        f"<h2>Shots ({len(report.shots)})</h2>"
        "<table><thead><tr><th>#</th><th>Start</th><th>End</th>"
        f"<th>Keyframes</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _html_content_moderation(report: CapabilitiesReport) -> str:
    cm = report.content_moderation
    if not cm or not cm.has_any_signal:
        return ""
    rows = "".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>"
        for label, value in (
            ("Visual adult score", cm.visual_adult_score),
            ("Visual racy score", cm.visual_racy_score),
            ("Banned words count", cm.textual_banned_words_count),
            ("Banned words ratio", cm.textual_banned_words_ratio),
        )
        if value is not None
    )
    return f"<h2>Content moderation</h2><table><tbody>{rows}</tbody></table>"


def _html_empty_notice(report: CapabilitiesReport) -> str:
    if any(report.coverage.values()):
        return ""
    return (
        '<div class="empty">Nothing found in any of these categories. '
        "If this video was indexed with --indexing-preset Default, "
        "that's expected -- most of these are Advanced-only. Re-run "
        "with --indexing-preset Advanced (a fresh upload).</div>"
    )


# One function per report section, in the order they're rendered. Each
# returns that section's HTML, or "" when it has nothing to show -- keeps
# write_html_capabilities itself a flat, linear dispatch list instead of one
# long function with a branch per capability (mirrors _CONSOLE_SECTIONS
# above).
_HTML_SECTIONS = (
    _html_faces,
    _html_observed_people,
    _html_interaction_signals,
    _html_transcript,
    _html_topics,
    _html_named_entities,
    _html_sentiments,
    _html_emotions,
    _html_audio_effects,
    _html_shots,
    _html_content_moderation,
    _html_empty_notice,
)


def write_html_capabilities(
    report: CapabilitiesReport, video_name: str, path: Path
) -> None:
    """Write a self-contained HTML capabilities showcase (a coverage
    scorecard plus a section per capability that found anything) to `path`."""
    sections = [section(report) for section in _HTML_SECTIONS]

    html = _HTML_TEMPLATE.format(
        video_name=escape(video_name),
        duration=_fmt_seconds(report.video_duration_seconds),
        language=escape(report.source_language or "unknown"),
        coverage_stats=_html_coverage_stats(report),
        sections="".join(sections),
    )
    path.write_text(html, encoding="utf-8")
