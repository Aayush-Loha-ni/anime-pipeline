#!/usr/bin/env python3
"""Audio planner — per-scene audio design.

Creates an audio plan with voice priority, ambience, music, SFX, and intensity
for each scene. Ties every asset to the rights manifest.
"""

import json
from pathlib import Path

# stdlib-only, so this cannot cycle into pipeline.*
from json_util import dump_json_atomic


def generate_audio_plan(scenes, chapter_num, music_dir=None, sfx_dir=None):
    """Generate an audio plan for all scenes.

    Returns a list of per-scene audio plan dicts.
    """
    plan = []
    music_tracks = []
    if music_dir and Path(music_dir).exists():
        music_tracks = sorted([p.name for p in Path(music_dir).glob("*.mp3")]
                            + [p.name for p in Path(music_dir).glob("*.wav")])
    sfx_files = []
    if sfx_dir and Path(sfx_dir).exists():
        sfx_files = sorted([p.name for p in Path(sfx_dir).glob("*.mp3")]
                         + [p.name for p in Path(sfx_dir).glob("*.wav")])

    _mood_intensity = {
        "tense": 0.8, "menacing": 0.9, "somber": 0.4,
        "melancholic": 0.5, "hopeful": 0.6,
    }
    # Mood -> filename keywords (valence/arousal style mapping).
    # Tracks are matched by substring, so "Dark Fog.mp3" matches tense/mystery.
    _MOOD_KEYWORDS = {
        "tense": ["dark", "fog", "tension", "drone", "shadow"],
        "menacing": ["dark", "shadow", "menace", "aggressive", "drone"],
        "somber": ["quiet", "somber", "melancholic", "pad", "sad"],
        "melancholic": ["melancholic", "sad", "quiet", "pad", "grief"],
        "hopeful": ["resolve", "resolution", "hope", "light", "dawn"],
        "mystery": ["shadow", "mystery", "fog", "dark"],
        "grief": ["melancholic", "grief", "sad", "quiet"],
        "resolve": ["resolve", "resolution", "hope"],
        "action": ["action", "tension", "drive"],
        "transition": ["transition", "sting"],
    }

    def _pick_music(mood, tracks, scene_num=None):
        names = [(t, t.lower()) for t in tracks]
        keywords = _MOOD_KEYWORDS.get(mood, [])
        best, best_score = None, -1
        for orig, low in names:
            score = sum(1 for kw in keywords if kw in low)
            if score > best_score:
                best, best_score = orig, score
        if best_score > 0:
            return best
        # No keyword hit: diversify by mood. ponytail: hash(str) is salted
        # per process (PYTHONHASHSEED), so the same script + same music dir
        # produced a DIFFERENT track on every run. md5 is stable — the
        # sibling implementation in episode_producer.pick_music_track
        # already does this.
        import hashlib
        if not tracks:
            return "none"
        idx = int(hashlib.md5(str(mood).encode("utf-8")).hexdigest(), 16) % len(tracks)
        return sorted(tracks)[idx]

    for i, scene in enumerate(scenes):
        num = scene.get("scene_number", i + 1)
        # `or` not `.get(k, default)`: that default covers a MISSING key, not a null
        # one, and scenes/*.json is hand-edited — `mood: None` raised
        # AttributeError on `.lower()`. Sibling modules (beat_planner,
        # script_reviewer) already used the tolerant form.
        mood = (scene.get("mood") or "neutral").lower()
        shot = scene.get("shot_size", "")
        has_characters = bool(scene.get("characters"))
        has_narration = bool(scene.get("narration") or scene.get("dialogue"))
        scene_type = scene.get("type", "")

        voice_priority = "high" if has_narration else "none"
        intended_silence = not has_narration and scene_type == "establishing"

        ambience_cue = "silence"
        if scene_type == "establishing" or not has_characters:
            ambience_cue = "environmental"
        elif mood in ("tense", "menacing"):
            ambience_cue = "tense_drone"
        elif mood in ("somber", "melancholic"):
            ambience_cue = "quiet_room"

        music_cue = "none"
        if music_tracks:
            music_cue = _pick_music(mood, music_tracks, num)

        sfx_cue = "none"
        if sfx_files:
            if mood in ("tense", "menacing") and len(sfx_files) > 0:
                sfx_cue = sfx_files[0]
            elif mood == "hopeful" and len(sfx_files) > 1:
                sfx_cue = sfx_files[1]

        emotional_intensity = _mood_intensity.get(mood, 0.5)
        if shot in ("close-up", "extreme-close-up"):
            emotional_intensity = min(1.0, emotional_intensity + 0.15)
        elif shot in ("wide", "extreme-wide"):
            emotional_intensity = max(0.1, emotional_intensity - 0.15)

        asset_id = f"audio_ch{chapter_num:03d}_s{num:02d}"

        entry = {
            "scene_number": num,
            "voice_priority": voice_priority,
            "intended_silence": intended_silence,
            "ambience_cue": ambience_cue,
            "music_cue": music_cue,
            "sfx_cue": sfx_cue,
            "emotional_intensity": round(emotional_intensity, 2),
            "asset_id": asset_id,
            "notes": "",
        }
        plan.append(entry)

    return plan


def save_audio_plan(plan, chapter_num, scenes_dir):
    """Save audio plan as JSON."""
    scenes_dir = Path(scenes_dir)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    path = scenes_dir / f"ch{chapter_num:03d}_audio_plan.json"
    dump_json_atomic(path, plan, indent=2, ensure_ascii=False)
    return path


def load_audio_plan(chapter_num, scenes_dir):
    """Load audio plan for a chapter. Returns list or None."""
    path = Path(scenes_dir) / f"ch{chapter_num:03d}_audio_plan.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def audio_plan_to_rights(plan, rights_manifest):
    """Ensure every audio asset in the plan has a rights entry.

    Returns list of missing asset IDs.
    """
    existing_ids = set()
    for category in ("audio_assets", "music", "sfx"):
        for e in rights_manifest.get(category, []):
            existing_ids.add(e.get("asset_id", ""))

    missing = []
    for entry in plan:
        aid = entry.get("asset_id", "")
        if aid and aid not in existing_ids:
            missing.append(aid)
    return missing
