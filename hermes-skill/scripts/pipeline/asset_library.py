"""Asset Library — unified metadata layer over chapters/, characters/, scenes/, images/, voice/, videos/.

Provides structured access, cross-references, and identity tracking
inspired by DramaClaw's XiaTang asset library.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent.parent.parent


def is_scene_script(value) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(scene, dict) and type(scene.get("scene_number")) is int
        and scene["scene_number"] > 0 for scene in value
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Character:
    key: str
    name: str
    role: str = ""
    prompt_tags: str = ""
    voice_key: str = ""
    page_path: Optional[str] = None
    voice_path: Optional[str] = None
    appearances: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scene:
    chapter: int
    scene_number: int
    scene_type: str = ""
    shot_size: str = ""
    angle: str = ""
    camera_move: str = ""
    mood: str = ""
    characters: list[str] = field(default_factory=list)
    narration: str = ""
    dialogue: str = ""
    text_overlay: str = ""
    stills: list[str] = field(default_factory=list)
    clips: dict[str, str] = field(default_factory=dict)
    has_voice: bool = False
    echofail: bool = False
    voice_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chapter:
    number: int
    title: str = ""
    source_file: str = ""
    has_script: bool = False
    scene_count: int = 0
    has_video: bool = False
    video_path: Optional[str] = None
    total_stills: int = 0
    total_clips: int = 0
    total_voices: int = 0
    characters: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

class AssetLibrary:
    """Unified view over the project filesystem."""

    def __init__(self, root: Path | None = None):
        self.root = root or ROOT
        self._characters_cache: dict[str, Character] | None = None
        self._chapters_cache: dict[int, Chapter] | None = None

    # -- Characters ---------------------------------------------------------

    def load_characters(self) -> dict[str, Character]:
        if self._characters_cache is not None:
            return self._characters_cache

        bible_path = self.root / "characters" / "characters.json"
        pages_dir = self.root / "characters" / "pages"
    
        bible = {}
        if bible_path.exists():
            try:
                raw = json.loads(bible_path.read_text("utf-8"))
                bible = raw.get("characters", {})
            except Exception:
                pass

        # Load voice mappings from config
        voice_map = {}
        config_path = self.root / "hermes-skill" / "scripts" / "config.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text("utf-8"))
                voice_map = cfg.get("character_voices", {})
            except Exception:
                pass

        chars: dict[str, Character] = {}
        for key, data in bible.items():
            page = pages_dir / f"{key}.html"
            voice = voice_map.get(key, "")

            voice_key = ""
            voice_path = None
            if voice:
                voice_key = Path(voice).stem
                try:
                    voice_abs = Path(voice) if Path(voice).is_absolute() else (self.root / voice)
                    voice_path = str(voice_abs.resolve().relative_to(self.root.resolve())) if voice_abs.exists() else None
                except ValueError:
                    voice_path = None

            chars[key] = Character(
                key=key,
                name=data.get("full_name", key),
                role=data.get("role", ""),
                prompt_tags=data.get("prompt_tags", ""),
                voice_key=voice_key,
                page_path=page.relative_to(self.root).as_posix() if page.exists() else None,
                voice_path=voice_path,
            )

        # Scan for appearances across scene JSONs
        for scenes_file in (self.root / "scenes").glob("ch*.json"):
            m = re.fullmatch(r"ch(\d{3})\.json", scenes_file.name)
            if not m:
                continue
            ch_num = int(m.group(1))
            try:
                scenes = json.loads(scenes_file.read_text("utf-8"))
                if not is_scene_script(scenes):
                    continue
                for s in scenes:
                    for c in s.get("characters", []):
                        if c in chars and ch_num not in chars[c].appearances:
                            chars[c].appearances.append(ch_num)
            except Exception:
                pass

        for c in chars.values():
            c.appearances.sort()

        self._characters_cache = chars
        return chars

    def get_character(self, key: str) -> Character | None:
        return self.load_characters().get(key)

    # -- Chapters -----------------------------------------------------------

    def load_chapters(self) -> dict[int, Chapter]:
        if self._chapters_cache is not None:
            return self._chapters_cache

        chapters: dict[int, Chapter] = {}

        # Scan chapter text files
        for f in (self.root / "chapters").glob("*.txt"):
            m = re.search(r"(\d+)", f.stem)
            if not m:
                continue
            num = int(m.group(1))
            ch = Chapter(number=num, source_file=f.name)
            chapters[num] = ch

        # Enrich with scene data
        for scenes_file in (self.root / "scenes").glob("ch*.json"):
            m = re.fullmatch(r"ch(\d{3})\.json", scenes_file.name)
            if not m:
                continue
            num = int(m.group(1))
            try:
                scenes = json.loads(scenes_file.read_text("utf-8"))
                if not is_scene_script(scenes):
                    continue
                if num not in chapters:
                    chapters[num] = Chapter(number=num)
                ch = chapters[num]
                ch.has_script = True
                ch.scene_count = len(scenes)
                seen_chars = set()
                for s in scenes:
                    for c in s.get("characters", []):
                        seen_chars.add(c)
                ch.characters = sorted(seen_chars)
            except Exception:
                pass

        # Enrich with video output
        for v in (self.root / "videos").glob("ch*.mp4"):
            m = re.search(r"ch(\d+)", v.stem)
            if not m:
                continue
            # Skip rotated backups. rotate_versions() writes ch001_youtube.v1.mp4
            # and .v2.mp4 beside the real file, so a plain glob maps both to
            # chapter 1 and mtime-ordering can hand the dashboard a version
            # file as "the video".
            if re.search(r"\.v\d+$", v.stem):
                continue
            num = int(m.group(1))
            if num not in chapters:
                chapters[num] = Chapter(number=num)
            ch = chapters[num]
            ch.has_video = True
            try:
                existing_mtime = (self.root / ch.video_path).stat().st_mtime if ch.video_path else 0
            except (OSError, ValueError):
                existing_mtime = 0
            try:
                v_mtime = v.stat().st_mtime
            except OSError:
                # A file removed by a render mid-scan must not abort the load.
                continue
            if v_mtime > existing_mtime:
                # .as_posix(): this string is served to the browser as JSON, and
                # a Windows backslash is a JS escape character, so
                # "videos\ch001_youtube.mp4" is not a path any client can split.
                ch.video_path = v.relative_to(self.root).as_posix()

        # Enrich with stills/clips/voices counts
        tag_counts: dict[int, dict[str, int]] = {}
        for p in (self.root / "images").glob("ch*_s*.png"):
            m = re.search(r"ch(\d+)", p.stem)
            # ponytail: exclude the beat-2 ("sNNb_") and companion
            # ("sNNc_") stills. They are derivatives of one scene, not extra
            # scenes, so counting them inflated ch001 to 38 stills for 25
            # scenes (+52%) and made this disagree with quality_evolve, which
            # already excluded them for exactly this reason.
            if m and not re.search(r"_s\d+[bc]_", p.name):
                n = int(m.group(1))
                tag_counts.setdefault(n, {"stills": 0, "clips": 0})["stills"] += 1
        for p in (self.root / "images").glob("ch*.mp4"):
            m = re.search(r"ch(\d+)", p.stem)
            if m:
                n = int(m.group(1))
                tag_counts.setdefault(n, {"stills": 0, "clips": 0})["clips"] += 1
        for p in (self.root / "voice").glob("ch*_s*.wav"):
            m = re.search(r"ch(\d+)", p.stem)
            if m:
                n = int(m.group(1))
                d = tag_counts.setdefault(n, {"stills": 0, "clips": 0, "voices": 0})
                d["voices"] = d.get("voices", 0) + 1

        for num, counts in tag_counts.items():
            if num in chapters:
                chapters[num].total_stills = counts.get("stills", 0)
                chapters[num].total_clips = counts.get("clips", 0)
                chapters[num].total_voices = counts.get("voices", 0)

        self._chapters_cache = chapters
        return chapters

    def get_chapter(self, number: int) -> Chapter | None:
        return self.load_chapters().get(number)

    # -- Scenes -------------------------------------------------------------

    def load_scenes(self, chapter: int) -> list[Scene]:
        scenes_file = self.root / "scenes" / f"ch{chapter:03d}.json"
        if not scenes_file.exists():
            return []

        try:
            raw = json.loads(scenes_file.read_text("utf-8"))
        except Exception:
            return []

        if not is_scene_script(raw):
            return []

        tag = f"ch{chapter:03d}"

        # Glob once for efficiency
        stills_map: dict[int, list[str]] = {}
        for p in (self.root / "images").glob(f"{tag}_s*.png"):
            sm = re.search(r"_s(\d+)", p.name)
            if sm:
                stills_map.setdefault(int(sm.group(1)), []).append(p.name)

        clips_map: dict[int, dict[str, str]] = {}
        for p in (self.root / "images").glob(f"{tag}_*.mp4"):
            cm = re.search(r"_(h3|echo|wan|ltx)(\d+)\.mp4$", p.name)
            if cm:
                clips_map.setdefault(int(cm.group(2)), {})[cm.group(1)] = p.name

        echofail_set = {int(m.group(1)) for p in (self.root / "images").glob(f"{tag}_echofail*.flag")
                        if (m := re.search(r"echofail(\d+)", p.name))}

        voice_set = {int(m.group(1)) for p in (self.root / "voice").glob(f"{tag}_s*.wav")
                     if (m := re.search(r"_s(\d+)", p.name))}

        scenes = []
        for i, s in enumerate(raw):
            num = s.get("scene_number", i + 1)
            scenes.append(Scene(
                chapter=chapter,
                scene_number=num,
                scene_type=s.get("type", ""),
                shot_size=s.get("shot_size", ""),
                angle=s.get("angle", ""),
                camera_move=s.get("camera_move", ""),
                mood=s.get("mood", ""),
                characters=s.get("characters", []),
                narration=s.get("narration", ""),
                dialogue=s.get("dialogue", ""),
                text_overlay=s.get("text_overlay", ""),
                stills=sorted(stills_map.get(num, [])),
                clips=clips_map.get(num, {}),
                has_voice=num in voice_set,
                echofail=num in echofail_set,
                voice_path=f"voice/{tag}_s{num:02d}.wav" if num in voice_set else None,
            ))

        return scenes

    # -- Summary / stats ----------------------------------------------------

    def summary(self) -> dict:
        chars = self.load_characters()
        chapters = self.load_chapters()
        total_stills = sum(ch.total_stills for ch in chapters.values())
        total_clips = sum(ch.total_clips for ch in chapters.values())
        total_voices = sum(ch.total_voices for ch in chapters.values())
        rendered = sum(1 for ch in chapters.values() if ch.has_video)
        scripted = sum(1 for ch in chapters.values() if ch.has_script)

        return {
            "total_chapters": len(chapters),
            "chapters_scripted": scripted,
            "chapters_rendered": rendered,
            "total_characters": len(chars),
            "total_stills": total_stills,
            "total_clips": total_clips,
            "total_voices": total_voices,
            "characters": [c.to_dict() for c in chars.values()],
        }

    def invalidate(self):
        self._characters_cache = None
        self._chapters_cache = None
