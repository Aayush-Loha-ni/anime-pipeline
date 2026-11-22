#!/usr/bin/env python3
"""Release report — structured QA verdict per episode.

Groups results into BLOCKER, REVIEW, WARNING, and PASS with
representative timestamps for efficient owner review.
"""

import json
import time
from pathlib import Path

# stdlib-only, so this cannot cycle into pipeline.*
from json_util import dump_json_atomic


def create_release_report(chapter_num, scenes, assembly_receipt,
                          rights_manifest=None, audio_plan=None,
                          quality_log_entry=None,
                          expect_fps=24.0, expect_dims=(1920, 1080)):
    """Create a structured release report for an episode.

    expect_fps / expect_dims default to the YouTube landscape target. The
    caller MUST pass the platform's real values — they used to be hardcoded
    here, so switching `platform` to a vertical preset made every render
    permanently NOT_RELEASE_READY on a file the pipeline was told to make.
    Returns the report dict.
    """
    issues = []
    blockers = []
    reviews = []
    warnings = []
    passes = []

    def _add(severity, category, message, scene=None, timestamp=None):
        entry = {
            "severity": severity,
            "category": category,
            "message": message,
        }
        if scene is not None:
            entry["scene"] = scene
        if timestamp is not None:
            entry["timestamp"] = timestamp
        issues.append(entry)
        if severity == "BLOCKER":
            blockers.append(entry)
        elif severity == "REVIEW":
            reviews.append(entry)
        elif severity == "WARNING":
            warnings.append(entry)
        else:
            passes.append(entry)

    # Check assembly receipt
    if not assembly_receipt:
        _add("BLOCKER", "assembly", "No assembly receipt — cannot verify output")
        return {
            "chapter": chapter_num,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "verdict": "NOT_RELEASE_READY",
            "summary": {
                "blockers": len(blockers), "reviews": len(reviews),
                "warnings": len(warnings), "passes": len(passes),
            },
            "issues": issues, "scores": {}, "notes": "",
        }
    elif not assembly_receipt.get("promoted"):
        _add("BLOCKER", "assembly",
             f"Assembly failed: {assembly_receipt.get('error', 'unknown')}")

    # Check A/V sync
    av_sync = assembly_receipt.get("qa_av_sync", {})
    if av_sync and not av_sync.get("passed", True):
        violations = av_sync.get("violations", [])
        if violations:
            _add("BLOCKER", "sync",
                 f"A/V sync gate failed: {len(violations)} violations",
                 scene=violations[0].get("scene") if violations else None)

    # Check true peak
    tp = assembly_receipt.get("true_peak", {})
    if tp and not tp.get("passed", True):
        _add("BLOCKER", "audio",
             f"True peak gate failed: {tp.get('peak_dbtp', '?')} dBTP")

    # Check voice presence
    voiceless = [s.get("scene_number", i + 1)
                 for i, s in enumerate(scenes)
                 if not (s.get("narration") or s.get("dialogue"))]
    if voiceless:
        _add("REVIEW", "voice",
             f"Scenes without voice: {voiceless}")

    # Check face flags
    face_flagged = [(s.get("scene_number", i + 1), s.get("_face_flag", ""))
                    for i, s in enumerate(scenes) if s.get("_face_flag")]
    if face_flagged:
        for num, flag in face_flagged:
            _add("REVIEW", "character",
                 f"Face flagged: {flag}", scene=num)

    # Check reframes
    reframed = [(s.get("scene_number", i + 1))
                for i, s in enumerate(scenes) if s.get("_reframed")]
    if reframed:
        _add("WARNING", "character",
             f"Reframes used in scenes: {reframed}")

    # Check clip rate — a receipt slot only counts if it has a real clip.
    receipt_scenes = assembly_receipt.get("scenes", [])
    if not isinstance(receipt_scenes, list):
        receipt_scenes = []
    clips_ok = 0
    for i, s in enumerate(scenes):
        if i >= len(receipt_scenes):
            continue
        slot = receipt_scenes[i]
        if isinstance(slot, dict):
            if slot.get("clip") or slot.get("path") or slot.get("file") or slot.get("ok"):
                clips_ok += 1
        elif slot:  # string path or truthy non-dict
            clips_ok += 1
    total_scenes = len(scenes)
    if total_scenes > 0:
        clip_rate = clips_ok / total_scenes
        if clip_rate < 0.8:
            _add("REVIEW", "clips",
                 f"Low clip coverage: {clip_rate:.0%}")

    # Check music bed
    music_bed = assembly_receipt.get("music_bed")
    if not music_bed:
        _add("WARNING", "audio", "No music bed in assembly")

    # Check rights
    if rights_manifest:
        from pipeline.rights_manifest import preflight_rights_gate
        # for_publish=True: this is the RELEASE gate. A non-commercial entry
        # (a fan work) may render but must never come out release-ready.
        ok, rights_blockers, rights_warnings = preflight_rights_gate(
            rights_manifest, for_publish=True)
        if not ok:
            for b in rights_blockers:
                _add("BLOCKER", "rights", b)
        for w in rights_warnings:
            _add("WARNING", "rights", w)

    # Check media validation. Fail CLOSED: `if streams:` used to skip the
    # entire gate whenever the receipt had no streams, so a missing
    # receipt silently produced a clean report. The media gate is the one
    # that must never go missing.
    streams = assembly_receipt.get("streams") or {}
    if not streams:
        _add("BLOCKER", "media",
             "No media probe in assembly receipt — stream properties unverified")
    else:
        fps = streams.get("fps", 0)
        if abs(float(fps) - float(expect_fps)) > 0.01:
            _add("BLOCKER", "media",
                 f"Wrong frame rate: {fps} fps (expected {expect_fps:g})")
        w, h = streams.get("width", 0), streams.get("height", 0)
        if (w, h) != tuple(expect_dims):
            _add("BLOCKER", "media",
                 f"Wrong dimensions: {w}x{h} (expected {expect_dims[0]}x{expect_dims[1]})")
        channels = streams.get("channels", 0)
        if channels != 2:
            _add("BLOCKER", "media",
                 f"Audio not stereo: {channels} channels")
        _add("PASS", "media",
             f"streams verified: {w}x{h} @ {fps}fps, {channels}ch")

    # Continuity / quality facts recorded by the clip phase. These used to
    # be unreachable in this report: they exist in the assembly receipt but
    # nothing read them, so a still-pan downgrade or a fast-tier episode was
    # invisible outside the log.
    grade = assembly_receipt.get("colour_grade") or {}
    if grade:
        if grade.get("applied"):
            _add("PASS", "colour",
                 f"{grade['applied']} clip(s) colour-graded to the episode "
                 f"reference (deflicker "
                 f"{'on' if grade.get('deflicker') else 'off'})")
        if grade.get("rejected"):
            _add("REVIEW", "colour",
                 f"{grade['rejected']} clip(s) exceeded the grade range or "
                 "failed to encode — treated as ungraded")
    lra = assembly_receipt.get("lra")
    if isinstance(lra, (int, float)):
        _add("PASS" if lra <= 14.0 else "WARNING", "audio",
             f"Loudness range {lra:.1f} LU")
    tiers = assembly_receipt.get("h3_tiers") or []
    if len(tiers) > 1 or (tiers and tiers != [assembly_receipt.get("quality_requested")]):
        _add("REVIEW", "quality",
             f"H3 quality tier fell back during this episode: {tiers}")
    # ponytail: the producer RECORDS cinematic_fallbacks and the degraded-scene
    # budget, but nothing here read them -- so an episode with a sixth of its
    # scenes shipping as Ken Burns pans over stills still reported PASS with no
    # mention. That is why re-wording the producer's warning changed nothing an
    # auditor ever sees. Surface the actual counts.
    fallbacks = assembly_receipt.get("cinematic_fallbacks")
    if isinstance(fallbacks, int) and fallbacks > 0:
        _add("REVIEW", "clips",
             f"{fallbacks} scene(s) degraded to a still-pan instead of a "
             f"generated clip (producer budget: max(1, scenes // 4))")
    vids = assembly_receipt.get("video_clips")
    par = assembly_receipt.get("cinematic_clips")
    if isinstance(vids, int) and isinstance(par, int):
        _add("INFO", "clips",
             f"{vids} generated clip(s), {par} cinematic shot(s)")
    relay = assembly_receipt.get("relay_used") or {}
    if relay.get("enabled") and not relay.get("clips_with_relay"):
        _add("WARNING", "continuity",
             "Continuity relay enabled but never applied - no adjacent pair of "
             "generated clips, or the previous clip had no readable last frame")

    dupes = assembly_receipt.get("duplicate_stills") or []
    if dupes:
        _add("REVIEW", "stills",
             f"{len(dupes)} near-identical still pair(s) survived the "
             "perceptual duplicate gate")

    # Quality scores
    scores = {}
    if quality_log_entry:
        scores = {
            "overall": quality_log_entry.get("score", 0),
            "image": quality_log_entry.get("avg_image", 0),
            "audio": quality_log_entry.get("avg_audio", 0),
        }
        if quality_log_entry.get("vlm_grades"):
            scores["vlm_grades"] = quality_log_entry["vlm_grades"]

    # Determine verdict
    if blockers:
        verdict = "NOT_RELEASE_READY"
    elif reviews:
        verdict = "NEEDS_REVIEW"
    elif warnings:
        verdict = "PASS_WITH_WARNINGS"
    else:
        verdict = "PASS"

    report = {
        "chapter": chapter_num,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": verdict,
        "summary": {
            "blockers": len(blockers),
            "reviews": len(reviews),
            "warnings": len(warnings),
            "passes": len(passes),
        },
        "issues": issues,
        "scores": scores,
        "notes": "",
    }
    return report


def save_release_report(report, output_dir, chapter_num):
    """Save release report as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"ch{chapter_num:03d}_release_report.json"
    dump_json_atomic(path, report, indent=2, ensure_ascii=False)
    return path


def load_release_report(output_dir, chapter_num):
    """Load release report for a chapter. Returns dict or None."""
    path = Path(output_dir) / f"ch{chapter_num:03d}_release_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
