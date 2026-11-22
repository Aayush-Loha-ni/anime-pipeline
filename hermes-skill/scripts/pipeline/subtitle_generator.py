#!/usr/bin/env python3
"""Subtitle generator — clean sidecar SRT + VTT from final script/audio.

Generates subtitle files with speaker labels for dialogue.
Preserves existing ASS generation, adds SRT/VTT export.
"""

from pathlib import Path


def _srt_ts(total_ms):
    """Format milliseconds as HH:MM:SS,mmm for SRT."""
    total_ms = max(0, int(total_ms))
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_ts(total_ms):
    """Format milliseconds as HH:MM:SS.mmm for VTT."""
    total_ms = max(0, int(total_ms))
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _parse_ass_ts(ts):
    """Parse ASS timestamp (H:MM:SS.cc) to seconds."""
    try:
        h, m, s = str(ts).split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0.0


def _scene_timing(scene, cursor, default_dur=5.0):
    """Resolve (start_ms, end_ms, next_cursor) without overlaps.

    Uses _cap_start/_cap_dur when present, else sequential estimation
    from text length (~15 chars/sec, clamped 1-7s).
    """
    text = ((scene.get("narration") or "") + " " + (scene.get("dialogue") or "")).strip()
    try:
        start_s = float(scene["_cap_start"]) if scene.get("_cap_start") is not None else cursor
    except (ValueError, TypeError):
        start_s = cursor
    start_s = max(start_s, cursor)
    dur = None
    for key in ("_cap_dur", "_dur"):
        try:
            if scene.get(key) is not None:
                dur = float(scene.get(key))
                if dur > 0:
                    break
        except (ValueError, TypeError):
            dur = None
    if not dur or dur <= 0:
        dur = min(7.0, max(1.0, len(text) / 15.0 if text else default_dur))
    end_s = start_s + max(0.4, dur)
    return int(start_s * 1000), int(end_s * 1000), end_s + 0.2


def _build_cues(scenes, ts_fn, header=()):
    """Shared SRT/VTT body. One implementation — the two formats differ
    only in the header lines and the timestamp separator, and the pair was
    34 duplicated lines that could (and did) drift apart.

    Speaker label: only applied when the caption IS the dialogue. The old
    `text = narration or dialogue` + `if dialogue: speaker=` combination
    attributed the narrator's words to a character whenever a scene had
    both.
    """
    lines = list(header)
    idx = 1
    cursor = 0.0

    for i, scene in enumerate(scenes):
        start_ms, end_ms, cursor = _scene_timing(scene, cursor)

        narration = (scene.get("narration") or "").strip()
        dialogue = (scene.get("dialogue") or "").strip()
        characters = scene.get("characters") or []

        if not narration and not dialogue:
            continue

        # dialogue wins: it is the line a character speaks, and the
        # assembly mixes that audio in this slot.
        is_dialogue = bool(dialogue)
        text = dialogue or narration
        # Only label the speaker when the scene has exactly ONE character. With three
        # on screen, `characters[0]` is a guess: a scene where only the second
        # character speaks was captioned `[Kael] "I never said that."` —
        # attributed to the wrong person, in the shipped subtitle track. There
        # is no speaker field in the scene schema, so an ambiguous scene gets
        # no label rather than a confident wrong one.
        speaker = (f"[{characters[0]}] "
                   if (is_dialogue and len(characters) == 1) else "")

        text = text.strip('"').strip("'")
        if not text:
            continue

        lines.append(str(idx))
        lines.append(f"{ts_fn(start_ms)} --> {ts_fn(end_ms)}")
        lines.append(f"{speaker}{text}")
        lines.append("")
        idx += 1

    return lines


def _write(lines, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return Path(output_path)


def generate_srt(scenes, output_path, dialogue_audios=None):
    """Generate SRT subtitle file from scenes.

    Speaker label for dialogue scenes. `dialogue_audios` is accepted for
    API compatibility and unused — scene text carries the caption content.
    """
    return _write(_build_cues(scenes, _srt_ts), output_path)


def generate_vtt(scenes, output_path, dialogue_audios=None):
    """Generate WebVTT subtitle file from scenes."""
    return _write(_build_cues(scenes, _vtt_ts, ("WEBVTT", "")), output_path)


def generate_subtitles(scenes, chapter_num, output_dir, dialogue_audios=None):
    """Generate both SRT and VTT files. Returns (srt_path, vtt_path)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    srt_path = output_dir / f"ch{chapter_num:03d}.srt"
    vtt_path = output_dir / f"ch{chapter_num:03d}.vtt"
    generate_srt(scenes, srt_path, dialogue_audios)
    generate_vtt(scenes, vtt_path, dialogue_audios)
    return srt_path, vtt_path
