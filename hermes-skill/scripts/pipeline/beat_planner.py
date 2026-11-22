"""Beat Planner — chapter → beats before scene-by-scene work.

Imported from DramaClaw's "chapter segmentation, beat planning" feature.
Assigns every scene a beat role (HOOK → RISE → TURN → CLIMAX → LAND) by
position so an episode gets an emotional arc instead of a flat wall of
scenes. Deterministic (no LLM) — this is a guide layer over scenes that
already exist; the shot_plan_pass in episode_producer uses the `purpose`
field, and we reuse that when present.
"""
from __future__ import annotations

from dataclasses import dataclass, field


BEATS = ["HOOK", "RISE", "TURN", "CLIMAX", "LAND"]


@dataclass
class BeatReport:
    assignment: dict[int, str] = field(default_factory=dict)
    mood_per_beat: dict[str, dict[str, int]] = field(default_factory=dict)
    gaps: list[int] = field(default_factory=list)
    source: str = "heuristic"
    status: str = "unplanned"

    def to_dict(self) -> dict:
        return {
            "assignment": self.assignment,
            "mood_per_beat": self.mood_per_beat,
            "gaps": self.gaps,
            "source": self.source,
            "status": self.status,
        }


def _beat_for_index(i: int, n: int) -> str:
    """Positional beat assignment. Purpose preset beats survive."""
    if n == 1:
        return "HOOK"
    pos = i / max(n - 1, 1)
    if pos < 0.15:
        return "HOOK"
    if pos < 0.55:
        return "RISE"
    if pos < 0.80:
        return "TURN"
    if pos < 0.93:
        return "CLIMAX"
    return "LAND"


def plan_beats(scenes: list[dict]) -> BeatReport:
    """Annotate scenes with a `beat` field (in place) and return the report."""
    rep = BeatReport()
    if not isinstance(scenes, list) or any(not isinstance(s, dict) for s in scenes):
        rep.status = "invalid"
        return rep
    n = len(scenes)
    purposes = [str(s.get("purpose", "")).strip().upper() for s in scenes]
    purposes = ["LAND" if p == "RESOLUTION" else p for p in purposes]
    planned = sum(p in BEATS for p in purposes)
    rep.source = "purpose" if n and planned == n else "mixed" if planned else "heuristic"
    for i, s in enumerate(scenes):
        existing = purposes[i]
        beat = existing if existing in BEATS else _beat_for_index(i, n)
        s["source"] = "purpose" if existing in BEATS else "heuristic"
        s["beat"] = beat
        s["beat_index"] = i  # order within the chapter
        rep.assignment[i] = beat

        mood = str(s.get("mood", "")).lower().strip()
        if mood:
            rep.mood_per_beat.setdefault(beat, {}).setdefault(mood, 0)
            rep.mood_per_beat[beat][mood] += 1

    if any(purposes):
        gaps = {i for i, p in enumerate(purposes) if p not in BEATS}
        if purposes[0] != "HOOK":
            gaps.add(0)
        if purposes[-1] != "LAND":
            gaps.add(n - 1)
        highest = -1
        for i, purpose in enumerate(purposes):
            if purpose in BEATS:
                rank = BEATS.index(purpose)
                if rank < highest:
                    gaps.add(i)
                highest = max(highest, rank)
        rep.gaps = sorted(gaps)
    if n and planned == n:
        rep.status = "gaps" if rep.gaps else "planned"
    return rep


def beats_summary(scenes: list[dict]) -> dict:
    """Compact summary for API/UI consumption (no mutation)."""
    if not isinstance(scenes, list) or any(not isinstance(s, dict) for s in scenes):
        return BeatReport(status="invalid").to_dict()
    rep = plan_beats([dict(s) for s in scenes])
    return rep.to_dict()