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
}


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
        (
            "Coverage (0 = not present in this payload -- see README on "
            "--indexing-preset for which of these need Advanced):"
        ),
    ]
    for key, count in report.coverage.items():
        lines.append(f"  {_CAPABILITY_LABELS[key]:<42} {count}")
    lines.append("")

    if report.faces:
        lines.append(f"Faces ({len(report.faces)}):")
        for f in report.faces:
            conf = f"{f.confidence:.2f}" if f.confidence is not None else "n/a"
            lines.append(
                f"  {f.name:<24} confidence={conf}  appearances={len(f.appearances)}"
            )
        lines.append("")

    if report.transcript:
        lines.append(f"Transcript ({len(report.transcript)} lines):")
        for t in report.transcript:
            speaker = f"{t.speaker}: " if t.speaker else ""
            lines.append(
                f"  [{_fmt_seconds(t.start_seconds)}-{_fmt_seconds(t.end_seconds)}] {speaker}{t.text}"
            )
        lines.append("")

    if report.topics:
        lines.append(f"Topics ({len(report.topics)}):")
        for t in report.topics:
            conf = f"{t.confidence:.2f}" if t.confidence is not None else "n/a"
            category = f" [{t.category}]" if t.category else ""
            lines.append(f"  {t.name}{category}  confidence={conf}")
        lines.append("")

    if report.named_entities:
        lines.append(f"Named entities ({len(report.named_entities)}):")
        for e in report.named_entities:
            conf = f"{e.confidence:.2f}" if e.confidence is not None else "n/a"
            lines.append(
                f"  [{e.kind}] {e.name}  confidence={conf}  appearances={len(e.appearances)}"
            )
        lines.append("")

    if report.sentiments:
        lines.append(f"Sentiments ({len(report.sentiments)}):")
        for s in report.sentiments:
            score = f"{s.score:.2f}" if s.score is not None else "n/a"
            lines.append(
                f"  {s.sentiment_type}  score={score}  appearances={len(s.appearances)}"
            )
        lines.append("")

    if report.emotions:
        lines.append(f"Emotions ({len(report.emotions)}):")
        for e in report.emotions:
            lines.append(f"  {e.emotion_type}  appearances={len(e.appearances)}")
        lines.append("")

    if report.audio_effects:
        lines.append(f"Audio effects ({len(report.audio_effects)}):")
        for a in report.audio_effects:
            conf = f"{a.confidence:.2f}" if a.confidence is not None else "n/a"
            lines.append(
                f"  {a.name}  confidence={conf}  appearances={len(a.appearances)}"
            )
        lines.append("")

    if report.shots:
        lines.append(f"Shots ({len(report.shots)}):")
        for s in report.shots:
            lines.append(
                f"  #{s.index}  [{_fmt_seconds(s.start_seconds)}-{_fmt_seconds(s.end_seconds)}]  "
                f"keyframes={s.keyframe_count}"
            )
        lines.append("")

    if report.content_moderation and report.content_moderation.has_any_signal:
        cm = report.content_moderation
        lines.append("Content moderation:")
        if cm.visual_adult_score is not None:
            lines.append(f"  visual adult score:  {cm.visual_adult_score:.3f}")
        if cm.visual_racy_score is not None:
            lines.append(f"  visual racy score:   {cm.visual_racy_score:.3f}")
        if cm.textual_banned_words_count is not None:
            lines.append(f"  banned words count:  {cm.textual_banned_words_count}")
        if cm.textual_banned_words_ratio is not None:
            lines.append(f"  banned words ratio:  {cm.textual_banned_words_ratio:.3f}")
        lines.append("")

    if not any(report.coverage.values()):
        lines.append(
            "Nothing found in any of these categories. If this video was "
            "indexed with --indexing-preset Default, that's expected -- "
            "most of these are Advanced-only. Re-run with "
            "--indexing-preset Advanced (a fresh upload; an already-"
            "indexed --video-id keeps whatever preset it was first "
            "uploaded with)."
        )

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


def write_html_capabilities(
    report: CapabilitiesReport, video_name: str, path: Path
) -> None:
    """Write a self-contained HTML capabilities showcase (a coverage
    scorecard plus a section per capability that found anything) to `path`."""
    coverage_stats = "".join(
        f'<div class="stat{" zero" if count == 0 else ""}">'
        f'<div class="label">{escape(_CAPABILITY_LABELS[key])}</div>'
        f'<div class="value">{count}</div></div>'
        for key, count in report.coverage.items()
    )

    sections: list[str] = []

    if report.faces:
        rows = "".join(
            f"<tr><td>{escape(f.name)}</td>"
            f"<td>{_fmt_score(f.confidence)}</td>"
            f"<td>{len(f.appearances)}</td></tr>"
            for f in report.faces
        )
        sections.append(
            f"<h2>Faces ({len(report.faces)})</h2>"
            "<table><thead><tr><th>Name</th><th>Confidence</th>"
            f"<th>Appearances</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    if report.transcript:
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
        sections.append(
            f"<h2>Transcript ({len(report.transcript)} lines)</h2>{lines_html}"
        )

    if report.topics:
        pills = "".join(
            f'<span class="pill">{escape(t.name)}'
            + (f" ({escape(t.category)})" if t.category else "")
            + "</span>"
            for t in report.topics
        )
        sections.append(f"<h2>Topics ({len(report.topics)})</h2><div>{pills}</div>")

    if report.named_entities:
        pills = "".join(
            f'<span class="pill">[{escape(e.kind)}] {escape(e.name)}</span>'
            for e in report.named_entities
        )
        sections.append(
            f"<h2>Named entities ({len(report.named_entities)})</h2><div>{pills}</div>"
        )

    if report.sentiments:
        rows = "".join(
            f"<tr><td>{escape(s.sentiment_type)}</td>"
            f"<td>{_fmt_score(s.score)}</td>"
            f"<td>{len(s.appearances)}</td></tr>"
            for s in report.sentiments
        )
        sections.append(
            f"<h2>Sentiments ({len(report.sentiments)})</h2>"
            "<table><thead><tr><th>Type</th><th>Score</th>"
            f"<th>Appearances</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    if report.emotions:
        pills = "".join(
            f'<span class="pill">{escape(e.emotion_type)} ({len(e.appearances)})</span>'
            for e in report.emotions
        )
        sections.append(f"<h2>Emotions ({len(report.emotions)})</h2><div>{pills}</div>")

    if report.audio_effects:
        pills = "".join(
            f'<span class="pill">{escape(a.name)} ({len(a.appearances)})</span>'
            for a in report.audio_effects
        )
        sections.append(
            f"<h2>Audio effects ({len(report.audio_effects)})</h2><div>{pills}</div>"
        )

    if report.shots:
        rows = "".join(
            f"<tr><td>{s.index}</td><td>{_fmt_seconds(s.start_seconds)}</td>"
            f"<td>{_fmt_seconds(s.end_seconds)}</td><td>{s.keyframe_count}</td></tr>"
            for s in report.shots
        )
        sections.append(
            f"<h2>Shots ({len(report.shots)})</h2>"
            "<table><thead><tr><th>#</th><th>Start</th><th>End</th>"
            f"<th>Keyframes</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    if report.content_moderation and report.content_moderation.has_any_signal:
        cm = report.content_moderation
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
        sections.append(
            f"<h2>Content moderation</h2><table><tbody>{rows}</tbody></table>"
        )

    if not any(report.coverage.values()):
        sections.append(
            '<div class="empty">Nothing found in any of these categories. '
            "If this video was indexed with --indexing-preset Default, "
            "that's expected -- most of these are Advanced-only. Re-run "
            "with --indexing-preset Advanced (a fresh upload).</div>"
        )

    html = _HTML_TEMPLATE.format(
        video_name=escape(video_name),
        duration=_fmt_seconds(report.video_duration_seconds),
        language=escape(report.source_language or "unknown"),
        coverage_stats=coverage_stats,
        sections="".join(sections),
    )
    path.write_text(html, encoding="utf-8")
