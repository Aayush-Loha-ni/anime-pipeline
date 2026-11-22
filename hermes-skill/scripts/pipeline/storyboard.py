"""Storyboard Generator — auto-generate shot plans from scene data.

Grid splitting, shot planning, and visual storyboard layout
inspired by DramaClaw's storyboard pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

# stdlib-only, so this cannot cycle into pipeline.*
from json_util import dump_json_atomic


@dataclass
class ShotPlan:
    scene_number: int
    shot_type: str  # "single", "split_2", "split_3", "split_4"
    panels: list[dict]  # [{shot_size, angle, duration, description}]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Storyboard:
    chapter: int
    shots: list[ShotPlan]
    total_duration_s: float = 0.0
    total_panels: int = 0

    def to_dict(self) -> dict:
        return {
            "chapter": self.chapter,
            "shots": [s.to_dict() for s in self.shots],
            "total_duration_s": self.total_duration_s,
            "total_panels": self.total_panels,
        }


# Default shot templates by scene type
SHOT_TEMPLATES = {
    "establishing": [
        {"shot_size": "extreme wide", "angle": "eye-level", "duration": 4.0,
         "description": "Wide establishing shot of location"},
    ],
    "action": [
        {"shot_size": "wide", "angle": "low", "duration": 3.0,
         "description": "Wide action shot"},
        {"shot_size": "close-up", "angle": "eye-level", "duration": 2.0,
         "description": "Close-up reaction"},
    ],
    "dialogue": [
        {"shot_size": "medium", "angle": "eye-level", "duration": 3.0,
         "description": "Two-shot dialogue"},
        {"shot_size": "close-up", "angle": "eye-level", "duration": 2.5,
         "description": "Speaker close-up"},
    ],
    "emotional": [
        {"shot_size": "close-up", "angle": "eye-level", "duration": 3.5,
         "description": "Emotional close-up"},
        {"shot_size": "medium", "angle": "eye-level", "duration": 2.0,
         "description": "Context medium shot"},
    ],
    "transition": [
        {"shot_size": "wide", "angle": "high", "duration": 3.0,
         "description": "Overhead transition"},
    ],
    "default": [
        {"shot_size": "medium", "angle": "eye-level", "duration": 3.0,
         "description": "Standard medium shot"},
        {"shot_size": "close-up", "angle": "eye-level", "duration": 2.0,
         "description": "Detail shot"},
    ],
}


def infer_scene_type(scene: dict) -> str:
    """Infer scene type from dialogue/narration content."""
    text = (scene.get("narration", "") + " " + scene.get("dialogue", "")).lower()

    if any(w in text for w in ["fight", "battle", "chase", "run", "attack"]):
        return "action"
    if any(w in text for w in ["said", "asked", "replied", "whispered", "shouted"]):
        return "dialogue"
    if any(w in text for w in ["remembered", "felt", "tears", "crying", "smiled"]):
        return "emotional"
    if any(w in text for w in ["arrived", "entered", "stepped into", "landscape"]):
        return "establishing"
    if any(w in text for w in ["then", "meanwhile", "later", "next"]):
        return "transition"

    return "default"


def generate_storyboard(chapter: int, scenes: list[dict],
                        default_duration: float = 5.0) -> Storyboard:
    """Generate a storyboard shot plan from scene data."""
    shots: list[ShotPlan] = []
    total_duration = 0.0
    total_panels = 0

    for i, s in enumerate(scenes):
        num = s.get("scene_number", i + 1)
        scene_type = infer_scene_type(s)

        # Use scene's own shot_size if specified, else template
        existing_shot = s.get("shot_size", "")
        existing_angle = s.get("angle", "")

        template = SHOT_TEMPLATES.get(scene_type, SHOT_TEMPLATES["default"])

        duration = s.get("duration", default_duration)
        panels = []
        for t in template:
            panel = {
                "shot_size": existing_shot or t["shot_size"],
                "angle": existing_angle or t["angle"],
                "duration": duration / len(template),
                "description": t["description"],
            }
            panels.append(panel)

        shot = ShotPlan(
            scene_number=num,
            shot_type="single" if len(panels) == 1 else f"split_{len(panels)}",
            panels=panels,
        )
        shots.append(shot)
        total_duration += sum(p["duration"] for p in panels)
        total_panels += len(panels)

    return Storyboard(
        chapter=chapter,
        shots=shots,
        total_duration_s=total_duration,
        total_panels=total_panels,
    )


def save_storyboard(storyboard: Storyboard, output_dir: Path):
    """Save storyboard JSON to disk."""
    # Path() first: every sibling save_* coerces its dir, and a str arg raised
    # AttributeError on mkdir instead of writing the file.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"ch{storyboard.chapter:03d}_storyboard.json"
    dump_json_atomic(out_file, storyboard.to_dict(), indent=2)
    return out_file


def load_storyboard(chapter: int, storyboard_dir: Path) -> Storyboard | None:
    """Load storyboard from disk."""
    f = storyboard_dir / f"ch{chapter:03d}_storyboard.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text("utf-8"))
        shots_data = data.get("shots", [])
        shots = []
        for s in shots_data:
            valid_keys = {f.name for f in ShotPlan.__dataclass_fields__.values()}
            filtered = {k: v for k, v in s.items() if k in valid_keys}
            shots.append(ShotPlan(**filtered))
        return Storyboard(
            chapter=data.get("chapter", chapter),
            shots=shots,
            total_duration_s=data.get("total_duration_s", 0),
            total_panels=data.get("total_panels", 0),
        )
    except Exception:
        return None
