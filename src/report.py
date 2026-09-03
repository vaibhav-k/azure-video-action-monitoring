"""
Renders an ActionReport as: a console summary, a machine-readable JSON file,
and a small self-contained HTML "oversight" timeline (no external assets, so
it opens directly in a browser with no server needed).
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from .action_analyzer import (
    ActionEvent,
    ActionReport,
    ActionSummary,
    AllActionsReport,
    DetectedAction,
)

TBODY_TABLE = "</tbody></table>"


def _fmt_seconds(value: float | None) -> str:
    """Render seconds as MM:SS.ss, or 'n/a' when the value is unknown."""
    if value is None:
        return "n/a"
    minutes, seconds = divmod(value, 60)
    return f"{int(minutes):02d}:{seconds:05.2f}"


def to_console_text(report: ActionReport, video_name: str) -> str:
    """Render a human-readable plain-text summary of the report."""
    lines = [
        f"Action monitoring report for '{video_name}'",
        f"Action of interest: {report.action}",
        f"Video duration: {_fmt_seconds(report.video_duration_seconds)}",
        f"Occurrences detected: {report.occurrence_count}",
        f"Total time flagged as '{report.action}': {_fmt_seconds(report.total_action_seconds)}",
        f"Distinct people tracked in video: {report.distinct_people_tracked}",
        "",
    ]
    if not report.events:
        lines.append(
            f"No '{report.action}'-related labels/keywords were found. "
            "See README 'Limitations' for why a generic Video Indexer model "
            "may miss a specific motion, and how to upgrade detection."
        )
    else:
        lines.append(
            f"{'Start':>8}  {'End':>8}  {'Source':<8}  {'Matched term':<20}  "
            f"{'Confidence':<10}  Evidence"
        )
        for e in report.events:
            conf = f"{e.confidence:.2f}" if e.confidence is not None else "n/a"
            lines.append(
                f"{_fmt_seconds(e.start_seconds):>8}  {_fmt_seconds(e.end_seconds):>8}  "
                f"{e.source:<8}  {e.matched_term:<20}  {conf:<10}  {e.evidence or '-'}"
            )
    return "\n".join(lines)


def to_dict(report: ActionReport, video_name: str) -> dict[str, Any]:
    """Render the report as a plain, JSON-serializable dict."""
    return {
        "video_name": video_name,
        "action": report.action,
        "video_duration_seconds": report.video_duration_seconds,
        "occurrence_count": report.occurrence_count,
        "total_action_seconds": report.total_action_seconds,
        "distinct_people_tracked": report.distinct_people_tracked,
        "matched_terms_found": sorted(report.matched_terms_found),
        "events": [
            {
                "source": e.source,
                "matched_term": e.matched_term,
                "start_seconds": e.start_seconds,
                "end_seconds": e.end_seconds,
                "duration_seconds": e.duration_seconds,
                "confidence": e.confidence,
                "evidence": e.evidence,
            }
            for e in report.events
        ],
    }


def write_json(report: ActionReport, video_name: str, path: Path) -> None:
    """Write the report as pretty-printed JSON to `path`."""
    path.write_text(json.dumps(to_dict(report, video_name), indent=2), encoding="utf-8")


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Action Monitoring Report</title>
<style>
  body {{
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
    margin: 2rem; color: #1a1a2e; background: #fafafa;
  }}
  h1 {{ font-size: 1.4rem; }}
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
  table {{ border-collapse: collapse; width: 100%; background: #fff; }}
  th, td {{
    border-bottom: 1px solid #eee; padding: 0.5rem 0.75rem;
    text-align: left; font-size: 0.9rem;
  }}
  th {{ background: #f2f2f7; }}
  .empty {{ padding: 1rem; background: #fff3cd; border-radius: 8px; }}
  .bar-track {{
    position: relative; height: 22px; background: #eef;
    border-radius: 4px; overflow: hidden;
  }}
  .bar-fill {{ position: absolute; top: 0; bottom: 0; background: #5b6cff; }}
</style>
</head>
<body>
  <h1>Action monitoring report: "{action}" in {video_name}</h1>
  <div class="summary">
    <div class="stat"><div class="label">Video duration</div><div class="value">{duration}</div></div>
    <div class="stat"><div class="label">Occurrences</div><div class="value">{count}</div></div>
    <div class="stat"><div class="label">Total flagged time</div><div class="value">{total_time}</div></div>
    <div class="stat"><div class="label">People tracked</div><div class="value">{people}</div></div>
  </div>
  {timeline}
  {table}
</body>
</html>
"""


def _render_timeline_bar(
    start: float, duration: float, total: float, title: str
) -> str:
    """One `.bar-fill` div, positioned as a percentage of `total` (the
    video's overall duration) -- shared by the single-action and
    all-actions HTML timelines."""
    left_pct = max(0.0, min(100.0, (start / total) * 100))
    width_pct = max(0.5, min(100.0 - left_pct, (duration / total) * 100))
    return (
        f'<div class="bar-fill" '
        f'style="left:{left_pct:.2f}%;width:{width_pct:.2f}%" '
        f'title="{title}"></div>'
    )


def _render_event_row(e: ActionEvent) -> str:
    """One `<tr>` for a single occurrence in the events table."""
    conf = f"{e.confidence:.2f}" if e.confidence is not None else "n/a"
    return (
        "<tr>"
        f"<td>{_fmt_seconds(e.start_seconds)}</td>"
        f"<td>{_fmt_seconds(e.end_seconds)}</td>"
        f"<td>{escape(e.source)}</td>"
        f"<td>{escape(e.matched_term)}</td>"
        f"<td>{conf}</td>"
        f"<td>{escape(e.evidence) if e.evidence else '-'}</td>"
        "</tr>"
    )


def _render_events_table(events: list[ActionEvent]) -> str:
    rows = "".join(_render_event_row(e) for e in events)
    return (
        "<table><thead><tr><th>Start</th><th>End</th><th>Source</th>"
        "<th>Matched term</th><th>Confidence</th><th>Evidence</th></tr></thead><tbody>"
        + rows
        + TBODY_TABLE
    )


def _render_events_timeline(events: list[ActionEvent], duration: float) -> str:
    """The single `.bar-track` div for `write_html`'s timeline, or "" when
    the video's duration is unknown (there's nothing to position bars
    against)."""
    if duration <= 0:
        return ""
    bars = "".join(
        _render_timeline_bar(
            e.start_seconds,
            e.duration_seconds,
            duration,
            f"{escape(e.matched_term)} "
            f"{_fmt_seconds(e.start_seconds)}-{_fmt_seconds(e.end_seconds)}",
        )
        for e in events
    )
    return f'<div class="bar-track">{bars}</div><p></p>'


def write_html(report: ActionReport, video_name: str, path: Path) -> None:
    """Write a self-contained HTML oversight report (table + timeline) to `path`."""
    duration = report.video_duration_seconds or 0.0

    if not report.events:
        table_html = (
            f'<div class="empty">No "{escape(report.action)}"-related labels/keywords '
            "were found in this video's insights. See the project README's "
            "Limitations section.</div>"
        )
        timeline_html = ""
    else:
        table_html = _render_events_table(report.events)
        timeline_html = _render_events_timeline(report.events, duration)

    html = _HTML_TEMPLATE.format(
        action=escape(report.action),
        video_name=escape(video_name),
        duration=_fmt_seconds(report.video_duration_seconds),
        count=report.occurrence_count,
        total_time=_fmt_seconds(report.total_action_seconds),
        people=report.distinct_people_tracked,
        timeline=timeline_html,
        table=table_html,
    )
    path.write_text(html, encoding="utf-8")


# ----------------------------------------------------------------------
# "All actions" report -- every label/keyword detected, not just one
# action of interest. See action_analyzer.analyze_all / AllActionsReport.
# ----------------------------------------------------------------------


def to_console_text_all(report: AllActionsReport, video_name: str) -> str:
    """Render a human-readable plain-text summary of every detected action."""
    lines = [
        f"All-actions report for '{video_name}'",
        f"Video duration: {_fmt_seconds(report.video_duration_seconds)}",
        f"Distinct actions detected: {report.distinct_action_count}",
        f"Total timestamped occurrences: {report.occurrence_count}",
        f"Distinct people tracked in video: {report.distinct_people_tracked}",
    ]
    if report.min_confidence is not None:
        lines.append(f"Confidence filter: >= {report.min_confidence:.2f}")
    lines.append("")

    if not report.actions:
        lines.append(
            "No labels or keywords were found in this video's insights. "
            "See README 'Limitations' for why a generic Video Indexer model "
            "may not tag every action."
        )
        return "\n".join(lines)

    lines.append(
        f"{'Action':<24}  {'Count':>5}  {'First':>8}  {'Last':>8}  {'Total time':>10}  Max conf."
    )
    for s in report.summaries:
        conf = f"{s.max_confidence:.2f}" if s.max_confidence is not None else "n/a"
        lines.append(
            f"{s.name:<24}  {s.occurrence_count:>5}  "
            f"{_fmt_seconds(s.first_occurrence_seconds):>8}  "
            f"{_fmt_seconds(s.last_occurrence_seconds):>8}  "
            f"{_fmt_seconds(s.total_duration_seconds):>10}  {conf}"
        )

    lines.append("")
    lines.append(
        f"{'Start':>8}  {'End':>8}  {'Source':<8}  {'Action':<24}  "
        f"{'Confidence':<10}  Evidence"
    )
    for a in report.actions:
        conf = f"{a.confidence:.2f}" if a.confidence is not None else "n/a"
        lines.append(
            f"{_fmt_seconds(a.start_seconds):>8}  {_fmt_seconds(a.end_seconds):>8}  "
            f"{a.source:<8}  {a.name:<24}  {conf:<10}  {a.evidence or '-'}"
        )
    return "\n".join(lines)


def to_dict_all(report: AllActionsReport, video_name: str) -> dict[str, Any]:
    """Render the all-actions report as a plain, JSON-serializable dict."""
    return {
        "video_name": video_name,
        "video_duration_seconds": report.video_duration_seconds,
        "distinct_action_count": report.distinct_action_count,
        "occurrence_count": report.occurrence_count,
        "distinct_people_tracked": report.distinct_people_tracked,
        "min_confidence": report.min_confidence,
        "action_summaries": [
            {
                "name": s.name,
                "sources": sorted(s.sources),
                "occurrence_count": s.occurrence_count,
                "total_duration_seconds": s.total_duration_seconds,
                "first_occurrence_seconds": s.first_occurrence_seconds,
                "last_occurrence_seconds": s.last_occurrence_seconds,
                "max_confidence": s.max_confidence,
            }
            for s in report.summaries
        ],
        "actions": [
            {
                "name": a.name,
                "source": a.source,
                "start_seconds": a.start_seconds,
                "end_seconds": a.end_seconds,
                "duration_seconds": a.duration_seconds,
                "confidence": a.confidence,
                "evidence": a.evidence,
            }
            for a in report.actions
        ],
    }


def write_json_all(report: AllActionsReport, video_name: str, path: Path) -> None:
    """Write the all-actions report as pretty-printed JSON to `path`."""
    path.write_text(
        json.dumps(to_dict_all(report, video_name), indent=2), encoding="utf-8"
    )


_ALL_ACTIONS_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>All Actions Report</title>
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
  table {{ border-collapse: collapse; width: 100%; background: #fff; }}
  th, td {{
    border-bottom: 1px solid #eee; padding: 0.5rem 0.75rem;
    text-align: left; font-size: 0.9rem;
  }}
  th {{ background: #f2f2f7; }}
  .empty {{ padding: 1rem; background: #fff3cd; border-radius: 8px; }}
  .lane-label {{ font-size: 0.8rem; color: #444; margin: 0.5rem 0 0.15rem; }}
  .bar-track {{
    position: relative; height: 18px; background: #eef;
    border-radius: 4px; overflow: hidden; margin-bottom: 0.4rem;
  }}
  .bar-fill {{ position: absolute; top: 0; bottom: 0; background: #5b6cff; }}
</style>
</head>
<body>
  <h1>All actions detected in {video_name}</h1>
  <div class="summary">
    <div class="stat"><div class="label">Video duration</div><div class="value">{duration}</div></div>
    <div class="stat"><div class="label">Distinct actions</div><div class="value">{distinct_count}</div></div>
    <div class="stat"><div class="label">Occurrences</div><div class="value">{count}</div></div>
    <div class="stat"><div class="label">People tracked</div><div class="value">{people}</div></div>
  </div>
  <h2>Timeline by action</h2>
  {timeline}
  <h2>Summary by action</h2>
  {summary_table}
  <h2>Full occurrence log</h2>
  {table}
</body>
</html>
"""


def _group_actions_by_name(
    actions: list[DetectedAction],
) -> dict[str, list[DetectedAction]]:
    """`actions` grouped by name, preserving each group's original relative
    order -- used to build one timeline lane per distinct action without
    re-scanning the full action list once per lane (an O(actions x
    summaries) nested scan for a video with many distinct actions)."""
    by_name: dict[str, list[DetectedAction]] = {}
    for a in actions:
        by_name.setdefault(a.name, []).append(a)
    return by_name


def _render_action_lane(
    name: str,
    occurrence_count: int,
    actions: list[DetectedAction],
    duration: float,
) -> str:
    """One timeline lane (label + bar track) for a single distinct action
    name -- one row of write_html_all's "Timeline by action" section."""
    bars = "".join(
        _render_timeline_bar(
            a.start_seconds,
            a.duration_seconds,
            duration,
            f"{escape(a.name)} {_fmt_seconds(a.start_seconds)}-{_fmt_seconds(a.end_seconds)}",
        )
        for a in actions
    )
    return (
        f'<div class="lane-label">{escape(name)} ({occurrence_count})</div>'
        f'<div class="bar-track">{bars}</div>'
    )


def _render_all_actions_timeline(
    summaries: list[ActionSummary], actions: list[DetectedAction], duration: float
) -> str:
    """One timeline lane per distinct action, in `summaries`' (first-
    occurrence) order, or an explanatory line when the video's duration is
    unknown (there's nothing to position lanes against)."""
    if duration <= 0:
        return "<p>Video duration unknown; timeline omitted.</p>"
    by_name = _group_actions_by_name(actions)
    return "".join(
        _render_action_lane(
            s.name, s.occurrence_count, by_name.get(s.name, []), duration
        )
        for s in summaries
    )


def _render_summary_row(s: ActionSummary) -> str:
    conf = f"{s.max_confidence:.2f}" if s.max_confidence is not None else "n/a"
    return (
        "<tr>"
        f"<td>{escape(s.name)}</td>"
        f"<td>{escape(', '.join(sorted(s.sources)))}</td>"
        f"<td>{s.occurrence_count}</td>"
        f"<td>{_fmt_seconds(s.first_occurrence_seconds)}</td>"
        f"<td>{_fmt_seconds(s.last_occurrence_seconds)}</td>"
        f"<td>{_fmt_seconds(s.total_duration_seconds)}</td>"
        f"<td>{conf}</td>"
        "</tr>"
    )


def _render_summary_table(summaries: list[ActionSummary]) -> str:
    rows = "".join(_render_summary_row(s) for s in summaries)
    return (
        "<table><thead><tr><th>Action</th><th>Source</th><th>Count</th>"
        "<th>First</th><th>Last</th><th>Total time</th><th>Max conf.</th></tr></thead><tbody>"
        + rows
        + TBODY_TABLE
    )


def _render_action_log_row(a: DetectedAction) -> str:
    conf = f"{a.confidence:.2f}" if a.confidence is not None else "n/a"
    return (
        "<tr>"
        f"<td>{_fmt_seconds(a.start_seconds)}</td>"
        f"<td>{_fmt_seconds(a.end_seconds)}</td>"
        f"<td>{escape(a.source)}</td>"
        f"<td>{escape(a.name)}</td>"
        f"<td>{conf}</td>"
        f"<td>{escape(a.evidence) if a.evidence else '-'}</td>"
        "</tr>"
    )


def _render_action_log_table(actions: list[DetectedAction]) -> str:
    rows = "".join(_render_action_log_row(a) for a in actions)
    return (
        "<table><thead><tr><th>Start</th><th>End</th><th>Source</th>"
        "<th>Action</th><th>Confidence</th><th>Evidence</th></tr></thead><tbody>"
        + rows
        + TBODY_TABLE
    )


def write_html_all(report: AllActionsReport, video_name: str, path: Path) -> None:
    """Write a self-contained HTML report (per-action timeline lanes, a
    per-action summary table, and the full occurrence log) to `path`."""
    duration = report.video_duration_seconds or 0.0

    if not report.actions:
        empty = (
            '<div class="empty">No labels or keywords were found in this '
            "video's insights. See the project README's Limitations section.</div>"
        )
        html = _ALL_ACTIONS_HTML_TEMPLATE.format(
            video_name=escape(video_name),
            duration=_fmt_seconds(report.video_duration_seconds),
            distinct_count=0,
            count=0,
            people=report.distinct_people_tracked,
            timeline=empty,
            summary_table=empty,
            table=empty,
        )
        path.write_text(html, encoding="utf-8")
        return

    summaries = report.summaries
    html = _ALL_ACTIONS_HTML_TEMPLATE.format(
        video_name=escape(video_name),
        duration=_fmt_seconds(report.video_duration_seconds),
        distinct_count=report.distinct_action_count,
        count=report.occurrence_count,
        people=report.distinct_people_tracked,
        timeline=_render_all_actions_timeline(summaries, report.actions, duration),
        summary_table=_render_summary_table(summaries),
        table=_render_action_log_table(report.actions),
    )
    path.write_text(html, encoding="utf-8")
