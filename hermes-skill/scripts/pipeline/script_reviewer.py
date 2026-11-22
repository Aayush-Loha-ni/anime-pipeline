"""Script Reviewer — review-and-repair loops for generated scripts.

Inspired by DramaClaw's adaptive/literal/staged script modes with
review-and-repair loops. Validates scene JSON structure, shot diversity,
character continuity, and emotional arc.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pipeline.viral_narration import score_narration

# stdlib-only, so this cannot cycle into pipeline.*
from json_util import dump_json_atomic

logger = logging.getLogger(__name__)


@dataclass
class ReviewIssue:
    severity: str  # "error", "warning", "info"
    scene: int
    field: str
    message: str
    suggestion: str = ""

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "scene": self.scene,
            "field": self.field,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class ReviewResult:
    valid: bool
    issues: list[ReviewIssue]
    score: int  # 0-100
    auto_fixes: list[dict]  # {scene, field, old, new}

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "issues": [i.to_dict() for i in self.issues],
            "score": self.score,
            "auto_fixes": self.auto_fixes,
        }


REQUIRED_FIELDS = ["scene_number", "type", "narration"]
SHOT_TYPES = {"extreme close-up", "close-up", "medium close-up", "medium",
              "medium wide", "wide", "extreme wide", "establishing", "over-the-shoulder",
              "pov", "low angle", "high angle", "bird's eye"}
ANGLES = {"eye-level", "low", "high", "dutch", "overhead", "worm's eye", "birds eye", "birds-eye"}
MOODS = {"tense", "somber", "warm", "joyful", "melancholic", "eerie", "romantic",
         "action", "calm", "mysterious", "dramatic", "humorous", "neutral"}


ROOT = Path(__file__).resolve().parent.parent.parent.parent


def review_script(scenes: list[dict], chapter_num: int = 0,
                  known_characters: set[str] | None = None,
                  persist: bool = False,
                  scenes_dir: str | Path | None = None) -> ReviewResult:
    """Review a scene array for quality issues. Returns issues + auto-fixes."""
    issues: list[ReviewIssue] = []
    auto_fixes: list[dict] = []

    if not isinstance(scenes, list) or any(not isinstance(s, dict) for s in scenes):
        return ReviewResult(valid=False, issues=[
            ReviewIssue("error", 0, "root", "Expected a list of scene dictionaries")
        ], score=0, auto_fixes=[])

    if not scenes:
        return ReviewResult(valid=False, issues=[
            ReviewIssue("error", 0, "root", "No scenes provided")
        ], score=0, auto_fixes=[])

    # --- Per-scene checks ---
    shot_counts: dict[str, int] = {}
    mood_counts: dict[str, int] = {}
    all_characters: set[str] = set()

    for i, s in enumerate(scenes):
        number = s.get("scene_number")
        num = number if type(number) is int and number > 0 else i + 1

        # Required fields
        for field in REQUIRED_FIELDS:
            if not s.get(field):
                issues.append(ReviewIssue("warning", num, field,
                                          f"Missing '{field}'",
                                          f"Add a {field} to scene {num}"))

        # Shot diversity tracking
        shot = s.get("shot_size", "").lower().strip()
        if shot:
            shot_counts[shot] = shot_counts.get(shot, 0) + 1

        # Angle validation
        angle = s.get("angle", "").lower().strip()
        if angle and angle not in ANGLES:
            issues.append(ReviewIssue("warning", num, "angle",
                                      f"Unusual angle: '{angle}'",
                                      f"Consider: {', '.join(sorted(ANGLES)[:5])}"))

        # Mood tracking
        mood = s.get("mood", "").lower().strip()
        if mood:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1

        # Character continuity
        chars = s.get("characters", [])
        if isinstance(chars, list):
            all_characters.update(chars)
        elif isinstance(chars, str):
            # Auto-fix: split comma-separated string
            chars_list = [c.strip() for c in chars.split(",") if c.strip()]
            auto_fixes.append({"scene": num, "field": "characters",
                               "old": chars, "new": chars_list})
            scenes[i]["characters"] = chars_list
            all_characters.update(chars_list)

        # Narration/dialogue presence
        narration = s.get("narration", "")
        dialogue = s.get("dialogue", "")
        if not narration and not dialogue:
            issues.append(ReviewIssue("info", num, "narration",
                                      "No narration or dialogue",
                                      "Consider adding voice content"))

        # Empty scene number check
        if type(number) is not int or number <= 0:
            auto_fixes.append({"scene": num, "field": "scene_number",
                               "old": number, "new": num})
            scenes[i]["scene_number"] = num

    # --- Global checks ---

    # Shot diversity: if one shot type is >60% of scenes, warn
    total = len(scenes)
    for shot, count in shot_counts.items():
        if count / total > 0.6 and total > 3:
            issues.append(ReviewIssue("warning", 0, "shot_diversity",
                                      f"Shot '{shot}' used {count}/{total} times ({count/total:.0%})",
                                      "Vary shot sizes for visual interest"))

    # Mood diversity: if all scenes same mood
    if len(mood_counts) == 1 and total > 2:
        mood = list(mood_counts.keys())[0]
        issues.append(ReviewIssue("warning", 0, "mood_diversity",
                                  f"All scenes have mood '{mood}'",
                                  "Vary emotional tone across scenes"))

    # Scene numbering continuity
    nums = [s.get("scene_number", i + 1) for i, s in enumerate(scenes)]
    expected = list(range(1, total + 1))
    if nums != expected:
        issues.append(ReviewIssue("warning", 0, "numbering",
                                  f"Scene numbers not sequential: {nums}",
                                  f"Expected: {expected}"))

    # Check against known characters
    if known_characters:
        unknown = all_characters - known_characters
        if unknown:
            issues.append(ReviewIssue("info", 0, "characters",
                                      f"Characters not in bible: {', '.join(sorted(unknown))}",
                                      "Add to characters.json or check spelling"))

    if persist and auto_fixes:
        base = Path(scenes_dir) if scenes_dir else (
            ROOT / "novels" / "Example Novel" / "scenes" if
            (ROOT / "novels" / "Example Novel" / "scenes").is_dir()
            else ROOT / "scenes")
        target = base / f"ch{chapter_num:03d}.json"
        # mkdir first. NamedTemporaryFile(dir=target.parent) raises
        # FileNotFoundError when scenes/ does not exist, and the old try had a
        # finally but no except — so a fresh project (or a deleted scenes/)
        # turned a dashboard "review" click into a 500 instead of a review.
        # A genuine write failure still propagates: rules/00_overview.md says
        # fail loud, and swallowing it would claim the auto-fixes were saved.
        target.parent.mkdir(parents=True, exist_ok=True)
        dump_json_atomic(target, scenes, indent=2, ensure_ascii=False)

    # --- Viral retention pass (deterministic, no LLM) ---
    # ponytail: warning/info only — this drags the score, never flips validity.
    # Never fail OPEN: a bare `except: pass` made a raised score_narration
    # yield zero findings and a clean-looking script. A degraded check must
    # say so.
    try:
        for f in score_narration(scenes).get("findings", []):
            if f.get("severity") in ("warning", "info"):
                issues.append(ReviewIssue(f.get("severity"), f.get("scene", 0),
                                          f.get("check", "viral"),
                                          f.get("message", ""), ""))
    except Exception as e:
        issues.append(ReviewIssue("warning", 0, "viral",
                                  f"retention pass failed to run ({e}) — "
                                  "hook/pacing NOT checked", ""))

    # Calculate score
    error_count = sum(1 for i in issues if i.severity == "error")
    warn_count = sum(1 for i in issues if i.severity == "warning")
    info_count = sum(1 for i in issues if i.severity == "info")
    score = max(0, 100 - error_count * 20 - warn_count * 5 - info_count * 1)

    return ReviewResult(
        valid=error_count == 0,
        issues=issues,
        score=score,
        auto_fixes=auto_fixes,
    )


def auto_repair(scenes: list[dict], chapter_num: int = 0) -> list[dict]:
    """Run review + auto-fix pass. Returns repaired scenes."""
    review_script(scenes, chapter_num)
    # Auto-fixes are already applied in-place by review_script
    return scenes
