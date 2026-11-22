#!/usr/bin/env python3
"""YouTube release package — complete deliverable set per approved episode.

Produces local release_package/ with video, subtitles, metadata,
disclosure checklist, thumbnail concepts, and credit note.
"""

import hashlib
import json
from pathlib import Path


def _file_sha256(path):
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return ""


def music_license_state(rights_manifest):
    """Derive the music-licence claim from the manifest. Never assert it.

    Returns one of not_used | verified | unverified, using exactly the rule
    disclosure.json already used. Both callers go through here so the copy a
    viewer reads and the compliance document cannot disagree.
    """
    music = (rights_manifest or {}).get("music") or []
    if not isinstance(music, list) or not music:
        return "not_used"
    licenses = {str(m.get("commercial_use", "unknown")).lower()
                for m in music if isinstance(m, dict)}
    # ponytail: an empty set is a subset of ANY set, so a manifest whose rows
    # are all non-dicts filtered `licenses` down to set() and
    # `set() <= {"yes","original"}` returned True — auto-signing "verified"
    # from a manifest that verified nothing, which is the exact failure this
    # function exists to prevent. Verified: music=["oops"] -> verified.
    if not licenses:
        return "unverified"
    return "verified" if licenses <= {"yes", "original"} else "unverified"


def _hashtag(name):
    """#Example Novel -> #SinnersWorld (YouTube hashtags are alphanumeric)."""
    return "#" + "".join(ch for ch in str(name or "") if ch.isalnum())


def generate_metadata(chapter_num, scenes, director_package=None,
                      rights_manifest=None, series_name=None):
    """Generate title options, description, chapters, tags from episode content.

    rights_manifest is used for exactly one thing: the music credit line. The
    previous version hardcoded "Music: Licensed/Fesliyan Studios (see credits
    below)" for every episode — a licence claim nothing verified, pointing at a
    credits section the function never emitted, and in direct contradiction of
    rules/assembly.md ("Do not automatically assert Fesliyan licensing") while
    rights_manifest.json still carries "Needs owner verification".
    """
    scenes = scenes or []
    series_name = (series_name or (director_package or {}).get("series_name")
                   or "Example Novel")

    moods = [s.get("mood") or "neutral" for s in scenes if isinstance(s, dict)]
    # max() on an empty iterable used to raise ValueError, which the caller's
    # bare `except Exception` turned into "Release package failed" — a
    # half-written package directory and no metadata at all.
    avg_mood = (max(sorted(set(moods)), key=moods.count) if moods else "neutral")

    title_options = [
        f"Chapter {chapter_num} — {series_name}",
        f"The Truth Revealed | {series_name} Ep {chapter_num}",
        f"What He Found Changed Everything | {series_name} Ch {chapter_num}",
    ]

    if director_package:
        promise = director_package.get("episode_promise", "")
        if promise:
            title_options.insert(0, promise[:80])

    description_parts = [
        f"{series_name} — Chapter {chapter_num}",
        "",
    ]
    if director_package:
        description_parts.append(director_package.get("episode_promise", ""))
        description_parts.append("")

    description_parts.extend([
        "An original AI-assisted animated adaptation.",
        "",
        "This episode was produced using:",
        "- Novel: Original work by the channel owner",
        "- Animation: AI-generated with Qwen-Image-2.1 + MiniMax H3",
        "- Voice: Owner's own voice clone (Qwen3-TTS)",
    ])

    # Credits: derived, and it always emits the section it refers to.
    state = music_license_state(rights_manifest)
    music_rows = [m for m in ((rights_manifest or {}).get("music") or [])
                  if isinstance(m, dict)]
    if state == "verified":
        description_parts.append(
            "- Music: Licensed / cleared for commercial use (see Credits)")
    elif state == "unverified":
        description_parts.append(
            "- Music: USED — licence NOT YET VERIFIED. Do not publish until "
            "confirmed; see disclosure.json")
    else:
        description_parts.append("- Music: none used in this episode")

    description_parts += ["", f"#Anime #OriginalAnimation {_hashtag(series_name)}", "", "Credits"]
    if state == "verified" and music_rows:
        for m in music_rows:
            who = m.get("source") or m.get("asset_id") or "music bed"
            note = m.get("license_note") or m.get("license") or ""
            description_parts.append(f"- Music: {who}" + (f" ({note})" if note else ""))
    else:
        description_parts.append(
            "- Music: see disclosure.json in the release package for the "
            "licence status of every audio asset used.")

    chapters = []
    cumulative = 0.0   # no hook: the timeline starts at the first scene
    for s in scenes:
        dur = float(s.get("_cap_dur", s.get("_dur", 5)))
        num = s.get("scene_number", 0)
        narration = (s.get("narration") or s.get("dialogue") or "")[:40]
        if narration:
            chapters.append(f"{_fmt_time(cumulative)} Scene {num}: {narration}")
        cumulative += dur

    tags = [
        "anime", "original animation", series_name.lower(), "dark fantasy",
        "gothic anime", "ai animation", "animated series",
    ]
    if avg_mood:
        tags.append(avg_mood)

    return {
        "title_options": title_options,
        "description": "\n".join(description_parts),
        "chapters": chapters,
        "tags": tags,
        "pinned_comment_draft": (
            "Thanks for watching! This is an original AI-assisted animated "
            "series. Every frame, voice, and note is created locally. "
            "Subscribe for the next chapter."
        ),
    }


def _fmt_time(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def generate_disclosure_checklist(has_voice_clone=False, has_ai_video=False,
                                   has_ai_images=False, has_music=False,
                                   music_license=""):
    """Generate YouTube disclosure checklist.

    music_license is DERIVED from has_music when not supplied: reporting
    "unverified" for an episode with no music is noise, and reporting
    "verified" for music nobody checked is worse.
    """
    if not music_license:
        music_license = "unverified" if has_music else "not_used"
    return {
        "synthetic_content": has_ai_video or has_ai_images,
        "altered_content": has_voice_clone,
        "voice_consent": has_voice_clone,
        "music_license": music_license,
        "age_suitable": True,
        "sponsorship": False,
        "ai_disclosure_recommended": has_ai_video or has_ai_images,
        "notes": "Owner retains final creative, legal, and YouTube Studio decisions",
    }


def generate_thumbnail_concepts(chapter_num, scenes, director_package=None):
    """Generate 3 thumbnail concepts per episode."""
    if not scenes:
        return []
    concepts = []
    focus_scene = next((s for s in scenes if s.get("shot_size") in ("close-up", "medium")), scenes[0])

    concepts.append({
        "concept_number": 1,
        "description": f"Close-up of {', '.join(focus_scene.get('characters', ['character']))} "
                      f"with intense expression, dark background",
        "text_overlay": "THE TRUTH",
        "focal_point": "face",
        "contrast": "high — face lit, background dark",
    })

    establishing = next((s for s in scenes if s.get("type") == "establishing"
                         or s.get("intended_shot_type") == "extreme-wide"
                         or s.get("_director_purpose") == "HOOK"), None)
    if establishing:
        concepts.append({
            "concept_number": 2,
            "description": "Wide establishing shot of the white-sky city, "
                          "small figure in the distance",
            "text_overlay": "Example Novel",
            "focal_point": "environment",
            "contrast": "atmospheric — fog, muted tones",
        })

    concepts.append({
        "concept_number": 3,
        "description": "Split composition: half character, half key object/scene element",
        "text_overlay": "WHAT CHANGED",
        "focal_point": "duality",
        "contrast": "medium — balanced light",
    })

    return concepts


def _screen_prompts(scenes):
    """Run the prompt safety screen over every scene. Returns (ran, errors).

    `ran` is False when the screen could not execute at all, which is NOT
    the same as "clean" — the credit note distinguishes the two.
    """
    try:
        from pipeline.rights_manifest import check_safety
    except Exception:
        return False, ["safety screen unavailable"]
    errors = []
    ran = False
    for i, s in enumerate(scenes or [], 1):
        for field in ("image_prompt", "narration", "dialogue"):
            text = s.get(field)
            if not text or not isinstance(text, str):
                continue
            ran = True
            try:
                ok, reason = check_safety(text, "prompt")
            except Exception as e:
                return True, [f"scene {i} {field}: screen raised {e}"]
            if not ok:
                errors.append(f"scene {i} {field}: {reason}")
    return ran, errors


def build_release_package(chapter_num, output_path, scenes,
                          director_package=None, release_report=None,
                          rights_manifest=None, srt_path=None, vtt_path=None,
                          series_name=None):
    """Assemble the complete release package directory.

    Returns the package dict.
    """
    output_path = Path(output_path)
    pkg_dir = output_path.parent / "release_package" / f"ch{chapter_num:03d}"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    metadata = generate_metadata(chapter_num, scenes, director_package,
                             rights_manifest=rights_manifest,
                             series_name=series_name)
    # Disclosure is derived, not asserted. The old call hardcoded
    # has_music=False and left music_license "unverified" no matter what,
    # which auto-signed the compliance document with statements that could
    # be false. A compliance artifact must report what it can prove and
    # say "unverified" where it cannot.
    _voice_ok = bool((rights_manifest or {}).get("voice", {}).get("reference_file_hash")
                     or (rights_manifest or {}).get("voice", {}).get("owner_confirmation"))
    _music = (rights_manifest or {}).get("music") or []
    _music_present = bool(_music) if isinstance(_music, list) else False
    # Same function the description text uses — one rule, two call sites, so
    # the viewer-facing copy and the compliance file can never disagree.
    _music_license = music_license_state(rights_manifest)
    disclosure = generate_disclosure_checklist(
        has_voice_clone=_voice_ok,
        # This pipeline always renders AI video and AI stills; the only
        # honest question is the music and the voice clone.
        has_ai_video=True, has_ai_images=True,
        has_music=_music_present, music_license=_music_license
    )
    thumbnails = generate_thumbnail_concepts(chapter_num, scenes, director_package)

    video_sha = _file_sha256(output_path) if output_path.exists() else ""

    series_name = (series_name or (director_package or {}).get("series_name")
                   or "Example Novel")
    # ponytail: screen the prompts, or don't claim the result. The old note
    # asserted "no living persons, studios, or existing franchises were
    # imitated" unconditionally while check_safety / check_prompt_rights had
    # zero production callers. Now the claim is only made when a screen
    # actually ran and passed; otherwise it says so.
    _screen_ran, _screen_errors = _screen_prompts(scenes)
    if _screen_errors:
        _imitation_line = ("NOT SCREENED: prompt safety screening reported "
                           + "; ".join(_screen_errors[:3])
                           + ". Review before publishing.")
    elif not _screen_ran:  # `ran` is a bool — `is None` let False through
        _imitation_line = ("NOT SCREENED: prompt safety screening is unavailable "
                           "in this build. Review before publishing.")
    else:
        _imitation_line = ("No living persons, studios, or existing franchises "
                           "were imitated (all scene prompts cleared the "
                           "automated safety screen).")
    credit_note = (
        f"{series_name} — Chapter {chapter_num}\n"
        "An original AI-assisted animated adaptation.\n\n"
        "Produced with:\n"
        "- Original novel by the channel owner\n"
        "- AI Animation: Qwen-Image-2.1 (stills) + MiniMax H3 (motion)\n"
        "- Voice: Owner's own voice (zero-shot clone)\n"
        "- All characters, story, and visual design are original\n\n"
        + _imitation_line
    )

    # Save metadata
    (pkg_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (pkg_dir / "disclosure.json").write_text(
        json.dumps(disclosure, indent=2, ensure_ascii=False), encoding="utf-8")
    (pkg_dir / "thumbnail_concepts.json").write_text(
        json.dumps(thumbnails, indent=2, ensure_ascii=False), encoding="utf-8")
    (pkg_dir / "credit_note.txt").write_text(credit_note, encoding="utf-8")

    import shutil
    # The package must contain the episode. It used to record a video_path
    # pointing outside the directory and hash a file it never copied, so
    # zipping release_package/chNNN/ produced a "video" that can't play.
    if output_path.exists():
        shutil.copy2(output_path, pkg_dir / output_path.name)
    if srt_path and Path(srt_path).exists():
        shutil.copy2(srt_path, pkg_dir / Path(srt_path).name)
    if vtt_path and Path(vtt_path).exists():
        shutil.copy2(vtt_path, pkg_dir / Path(vtt_path).name)

    if release_report:
        (pkg_dir / "release_report.json").write_text(
            json.dumps(release_report, indent=2, ensure_ascii=False), encoding="utf-8")
    if rights_manifest:
        (pkg_dir / "rights_manifest.json").write_text(
            json.dumps(rights_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    pkg = {
        "chapter": chapter_num,
        # Point at the COPY that ships with the package. The old value was the
        # absolute source path in videos/, so a consumer resolving it relative
        # to the package directory got a file that was not there. Keep the
        # original location too — it is genuinely useful for provenance.
        "video_path": str(pkg_dir / output_path.name) if output_path.exists()
                       else str(output_path),
        "source_video": str(output_path),
        "video_sha256": video_sha,
        "subtitles": {
            "srt": str(pkg_dir / Path(srt_path).name) if srt_path else None,
            "vtt": str(pkg_dir / Path(vtt_path).name) if vtt_path else None,
        },
        "metadata": metadata,
        "disclosure_checklist": disclosure,
        "thumbnail_concepts": thumbnails,
        "release_report": release_report,
        "rights_manifest": rights_manifest,
        "credit_note": credit_note,
    }

    (pkg_dir / "package.json").write_text(
        json.dumps(pkg, indent=2, ensure_ascii=False), encoding="utf-8")
    return pkg
