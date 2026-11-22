#!/usr/bin/env python3
"""Analytics learning log — owner-entered performance data.

Simple JSON store for YouTube analytics that can propose future experiments.
"""

import json
import time
from pathlib import Path

# stdlib-only, so this cannot cycle into pipeline.*
from json_util import dump_json_atomic


def load_analytics_log(root):
    """Load analytics_learning_log.json. Returns list of entries."""
    p = Path(root) / "analytics_learning_log.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_analytics_log(log, root):
    """Save analytics log."""
    p = Path(root) / "analytics_learning_log.json"
    # Atomic: this file is the only consumer of the "learning" history. A
    # truncate-then-dump that dies mid-write left a partial file, and the next
    # load_analytics_log returned [] silently — so the following append wrote a
    # 1-row file that overwrote the entire history with no warning anywhere.
    dump_json_atomic(p, log[-200:], indent=2, ensure_ascii=False)
    return p


def add_entry(root, chapter, impressions=None, ctr=None, avg_view_duration=None,
              retention_30s=None, completion_rate=None, subscribers_gained=None,
              comments_themes=None, manual_lesson=None,
              quality_score=None, episode_seconds=None, scene_count=None,
              verdict=None, novel=None):
    """Add an analytics entry for a chapter.

    The owner-entered YouTube metrics (ctr, completion_rate, ...) are
    optional. The pipeline-known facts (quality_score, episode_seconds,
    scene_count, verdict) should always be passed — a row with nothing but
    `chapter` is what this log used to append on every render, which made
    the "learning log" a list of empty rows and poisoned the last-5 window
    that propose_experiments() reads.
    """
    log = load_analytics_log(root)
    entry = {
        "chapter": chapter,
        "novel": novel or "sinners_world",
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if quality_score is not None:
        entry["quality_score"] = float(quality_score)
    if episode_seconds is not None:
        entry["episode_seconds"] = float(episode_seconds)
    if scene_count is not None:
        entry["scene_count"] = int(scene_count)
    if verdict:
        entry["verdict"] = str(verdict)
    if impressions is not None:
        entry["impressions"] = int(impressions)
    if ctr is not None:
        entry["ctr"] = float(ctr)
    if avg_view_duration is not None:
        entry["avg_view_duration"] = float(avg_view_duration)
    if retention_30s is not None:
        entry["retention_30s"] = float(retention_30s)
    if completion_rate is not None:
        entry["completion_rate"] = float(completion_rate)
    if subscribers_gained is not None:
        entry["subscribers_gained"] = int(subscribers_gained)
    if comments_themes:
        entry["comments_themes"] = comments_themes
    if manual_lesson:
        entry["manual_lesson"] = manual_lesson

    log.append(entry)
    save_analytics_log(log, root)
    return entry


def summarize(log, window=10):
    """Summarize recent analytics. Returns dict of averages and observations."""
    if not log:
        return {"entries": 0}
    recent = log[-window:]
    n = len(recent)
    summary = {"entries": n}
    for key in ("impressions", "ctr", "avg_view_duration", "retention_30s",
                "completion_rate", "subscribers_gained", "quality_score",
                "episode_seconds"):
        vals = [e[key] for e in recent if key in e and isinstance(e[key], (int, float))]
        if vals:
            summary[f"avg_{key}"] = round(sum(vals) / len(vals), 2)
            summary[f"min_{key}"] = round(min(vals), 2)
            summary[f"max_{key}"] = round(max(vals), 2)
    lessons = [e.get("manual_lesson", "") for e in recent if e.get("manual_lesson")]
    if lessons:
        summary["recent_lessons"] = lessons[-5:]
    return summary


def propose_experiments(log):
    """Analyze trends and propose simple experiments.

    Returns list of proposal strings. Does NOT auto-apply anything.
    """
    # Only rows that actually carry a metric can support a trend claim.
    measured = [e for e in log if any(k in e for k in
                 ("ctr", "completion_rate", "avg_view_duration", "retention_30s"))]
    if len(measured) < 3:
        return ["Need at least 3 entries with YouTube metrics before proposing experiments"]

    proposals = []
    recent = measured[-5:]

    ctrs = [e["ctr"] for e in recent if "ctr" in e]
    if ctrs:
        avg_ctr = sum(ctrs) / len(ctrs)
        if avg_ctr < 0.03:
            proposals.append(
                f"CTR is low ({avg_ctr:.1%}). Test different thumbnail concepts "
                "or title phrasing for the next episode."
            )
        elif avg_ctr > 0.08:
            proposals.append(
                f"CTR is strong ({avg_ctr:.1%}). Current thumbnail/title "
                "strategy is working — keep the same formula."
            )

    completions = [e["completion_rate"] for e in recent if "completion_rate" in e]
    if completions:
        avg_completion = sum(completions) / len(completions)
        if avg_completion < 0.3:
            proposals.append(
                f"Completion rate is low ({avg_completion:.0%}). "
                "Consider shorter episodes or stronger cliffhangers."
            )

    durations = [e["avg_view_duration"] for e in recent if "avg_view_duration" in e]
    if durations:
        avg_dur = sum(durations) / len(durations)
        if avg_dur < 30:
            proposals.append(
                "Average view duration under 30s. "
                "The hook may not be landing — test a more dramatic cold open."
            )

    return proposals or ["No significant trends detected yet"]
