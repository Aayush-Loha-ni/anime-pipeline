"""Quality rating + the closed-loop auto-tuner.

Split out of episode_producer.py (2026-09-29). Nothing here is on the
critical render path: it runs once, after a finished episode, to score the
take and to decide whether any generation parameter should change for the
next one. Splitting it out keeps a ~500-line analytics tail out of the
7k-line producer and gives the tuner its own importable home.

Runtime collaborators (CONFIG, ROOT, log, the VLM helpers) are resolved
through the `episode_producer` module object at CALL time, not imported by
value, so tests that patch `ep.vlm_available` / `ep.CONFIG` still take
effect here.

The `episode_producer` import itself is DEFERRED into `_ep_get()`.
episode_producer imports this module from its own tail, so a module-scope
import made `import pipeline.quality_evolve` FIRST fail with a circular
ImportError — it only worked by the accident that every consumer happened to
import episode_producer first. An IDE indexer or a new entry point opening
this file would have hit it.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

# json_util is stdlib-only, so a top-level import cannot cycle back into
# pipeline.* — unlike importing episode_producer, which the note above this
# import block is about.
from json_util import dump_json_atomic

_ep = None

# Stand-in values so a bare module import never explodes. _sync() overwrites
# every one of these before any public entry point does real work.
CONFIG = {}
CONFIG_FILE = Path("config.json")
ROOT = Path(".")
AUTO_ALLOWLIST = ()


def _ep_get():
    """Import episode_producer lazily and cache it.

    Deferred because episode_producer imports this module from its tail;
    importing it at module scope made the cycle a hard ImportError.
    """
    global _ep
    if _ep is None:
        _ep = importlib.import_module("episode_producer")
    return _ep


def _sync():
    """Refresh the module-level aliases from episode_producer at call time."""
    global _ep, CONFIG, CONFIG_FILE, ROOT, AUTO_ALLOWLIST
    _ep = _ep_get()
    CONFIG = _ep.CONFIG
    CONFIG_FILE = _ep.CONFIG_FILE
    ROOT = _ep.ROOT
    AUTO_ALLOWLIST = _ep.AUTO_ALLOWLIST


def _log(msg, indent=0):
    return _ep_get().log(msg, indent)


def _novel_id():
    """Active novel id, or the legacy default. Quality history stays in one
    shared file (the tuner is model-wide), tagged per novel."""
    ep = _ep_get()
    n = ep.active_novel()
    return getattr(n, "id", None) or getattr(ep, "DEFAULT_NOVEL_ID", "sinners_world")


def _newest(paths):
    """Newest file in `paths`, or None.

    Delegated to episode_producer rather than redefined. This was a live
    NameError at split time: the two call sites below referenced a bare
    `_newest` that the module no longer had, so every `rate_episode()` call
    died on scene 0 — and the blanket `except` around rating in
    _produce_inner turned that into a one-line WARN, leaving quality_log.json
    frozen and the tuner running on stale history. pyflakes catches it in
    under a second; `test_quality_evolve_covers` now pins it.
    """
    return _ep_get()._newest(paths)


# --- Quality Rater + Evolve ----------------------------------------
def _image_metrics(img_path):
    """Brightness / contrast / sharpness of a reference image (0..100-ish)."""
    try:
        from PIL import Image, ImageFilter, ImageStat
        im = Image.open(img_path).convert("L").resize((256, 256))
        st = ImageStat.Stat(im)
        brightness = st.mean[0] / 2.55
        contrast = min(100.0, st.stddev[0] / 1.2)
        edges = im.filter(ImageFilter.FIND_EDGES)
        sharpness = min(100.0, ImageStat.Stat(edges).stddev[0] / 0.9)
        return {"brightness": round(brightness, 1),
                "contrast": round(contrast, 1),
                "sharpness": round(sharpness, 1)}
    except Exception:
        return {"brightness": 0.0, "contrast": 0.0, "sharpness": 0.0}


def _audio_metrics(wav_path):
    """Duration + peak level of a narration wav.

    Reads ffmpeg via `_ep_get().CONFIG` rather than the module-level CONFIG
    alias: this is the one helper that can be reached without an entry
    point having called _sync() first, and the placeholder CONFIG is {},
    which would raise KeyError and — via the blanket except below — silently
    return peak_db -99.0, quietly zeroing the audio sub-score.
    """
    dur = _ep_get().audio_duration(wav_path)
    peak = -99.0
    try:
        ffmpeg = _ep_get().CONFIG.get("ffmpeg", "ffmpeg")
        out = subprocess.run(
            [ffmpeg, "-i", str(wav_path), "-af", "volumedetect",
             "-f", "null", "NUL" if os.name == "nt" else "/dev/null"],
            capture_output=True, text=True, timeout=60)
        m = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", out.stderr)
        if m:
            peak = float(m.group(1))
    except Exception:
        pass
    return {"duration": round(dur, 2), "peak_db": peak}


def text_prompt_alignment(scene):
    """P3 QA: cheap text-image proxy (no CLIP weights) — word overlap of the
    spoken line vs image_prompt. Trend, not gate. Real CLIP score when staged."""
    try:
        line = f"{scene.get('narration','')} {scene.get('dialogue','')}".lower().split()
        prompt = str(scene.get("image_prompt", "")).lower()
        stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "it", "he", "she", "they"}
        # m2: >2 keeps sky/red/fox/box visible (was >3, dropped all 3-letter words).
        words = [w.strip(".,!?\"'") for w in line if len(w) > 2 and w not in stop]
        if not words:
            return 100.0
        hit = sum(1 for w in words if w in prompt)
        return round(100.0 * hit / len(words), 1)
    except Exception:
        return 50.0


def run_clip_still_checks(queued, scenes, chapter_num):
    """Run the queued clip-vs-still vision checks now the GPU is free.

    The check needs two 1024-token images through a 32B-class VLM. Run during
    clip generation it competed with ComfyUI for VRAM and timed out on 2 of 4
    scenes in a real render, so it is queued in the clip phase and executed
    here, after the VLM is up and ComfyUI has released the card.

    Returns a list of mismatch records (empty when everything matches or the
    VLM is unavailable). Never raises — this is observability, not a gate.
    """
    _sync()
    flags = []
    if not queued:
        return flags
    if not _ep_get().vlm_available():
        _log(f"  [WARN] {len(queued)} clip/still check(s) skipped — VLM down",
            1)
        return flags
    for item in queued:
        frame, still = item.get("frame"), item.get("still")
        if not (frame and still and Path(frame).exists()
                and Path(still).exists()):
            continue
        n = item.get("scene")
        scene = next((s for s in scenes
                      if s.get("scene_number") == n), {"image_prompt": ""})
        ok, note = _ep_get()._vlm_clip_matches_still(frame, still, scene)
        if ok is False:
            # warn only — the VLM is the only check that can see a subject
            # swap, but its false-positive rate is unmeasured, so it must not
            # block a render.
            _log(f"  Scene {n}: [VLM] clip no longer matches the still "
                f"(drift {item.get('drift')}) {note}", 1)
            flags.append({"scene": n, "drift": item.get("drift"), "note": note})
    if not flags:
        _log(f"  [VLM] all {len(queued)} clip(s) still match their reference",
            1)
    return flags


def rate_episode(chapter_num, scenes, scene_clips, dialogue_audios, novel=None):
    """Score the finished episode 0..100 and persist a quality log."""
    _sync()
    images_dir = Path(CONFIG["images_dir"])
    per_scene, img_scores, aud_scores = [], [], []
    for i, scene in enumerate(scenes):
        num = scene.get("scene_number", i + 1)
        # P3 ops: exclude b_ AND c_ companions — _newest() must be the
        # scene A-still, never a close-up companion.
        refs = [p for p in images_dir.glob(f"ch{chapter_num:03d}_s{num:02d}_*.png")
                if f"s{num:02d}b_" not in p.name and f"s{num:02d}c_" not in p.name]
        im = _image_metrics(_newest(refs)) if refs else \
            {"brightness": 0.0, "contrast": 0.0, "sharpness": 0.0}
        aud = dialogue_audios[i] if i < len(dialogue_audios) else None
        au = _audio_metrics(aud) if aud and Path(aud).exists() else \
            {"duration": 0.0, "peak_db": -99.0}
        # Image sub-score: penalize dark / flat / blurry (+10% text alignment trend)
        b_ok = 100 - min(100, abs(im["brightness"] - 50) * 1.5)
        c_ok = min(100, im["contrast"] * 1.4)
        s_ok = min(100, im["sharpness"] * 1.2)
        align = text_prompt_alignment(scene)
        img_score = round(0.32 * b_ok + 0.27 * c_ok + 0.31 * s_ok + 0.10 * align, 1)
        aud_score = 100.0 if (au["duration"] > 0 and au["peak_db"] > -35) else \
            (50.0 if au["duration"] > 0 else 0.0)
        img_scores.append(img_score)
        aud_scores.append(aud_score)
        per_scene.append({"scene": num, "image": im, "audio": au,
                          "image_score": img_score, "audio_score": aud_score,
                          "align": align,
                          # q1: face-gate observability lands in quality_log.
                          "face_flag": scene.get("_face_flag", ""),
                          "reframed": bool(scene.get("_reframed", False)),
                          "has_clip": scene_clips[i] is not None})
    clip_rate = (sum(1 for c in scene_clips if c is not None) / max(1, len(scenes))) * 100
    # VLM spot-check: 3 weakest PIL scenes (bounded cost)
    vlm_grades = []
    if _ep_get().vlm_available():
        order = sorted(range(len(per_scene)),
                       key=lambda i: per_scene[i]["image_score"])[:3]
        for i in order:
            num = per_scene[i]["scene"]
            # M6: same A-still-only predicate as main refs (was missing c_).
            refs = [p for p in images_dir.glob(f"ch{chapter_num:03d}_s{num:02d}_*.png")
                    if f"s{num:02d}b_" not in p.name and f"s{num:02d}c_" not in p.name]
            if refs:
                g = _ep_get().vlm_grade_image(_newest(refs), "anime scene still")
                if g:
                    vlm_grades.append({"scene": num, **g})
                    _log(f"  VLM scene {num}: {g.get('grade')} "
                        f"({'; '.join(g.get('issues', [])[:2])})")
    vlm_score = None
    if vlm_grades:
        # ponytail: issues cap the grade — a non-empty issues[] can never
        # stay S (scene 15 shipped S + "head out of frame"). Cap S->A when
        # any issue is reported so the grader stays honest by construction.
        pts = {"S": 100, "A": 85, "B": 70, "C": 50, "D": 30}
        _vals = []
        for g in vlm_grades:
            _v = pts.get(g.get("grade", "C"), 50)
            if g.get("issues") and _v >= 100:
                _v = 85
            _vals.append(_v)
        vlm_score = round(sum(_vals) / len(_vals), 1)
    img_avg = sum(img_scores) / max(1, len(img_scores))
    if vlm_score is not None:
        img_avg = round(img_avg * 0.7 + vlm_score * 0.3, 1)
    overall = round(
        img_avg * 0.55 +
        (sum(aud_scores) / max(1, len(aud_scores))) * 0.30 +
        clip_rate * 0.15, 1)
    log_entry = {
        "chapter": chapter_num,
        "novel": novel or _novel_id(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "score": overall,
        "avg_image": round(sum(img_scores) / max(1, len(img_scores)), 1),
        "avg_audio": round(sum(aud_scores) / max(1, len(aud_scores)), 1),
        "clip_rate": round(clip_rate, 1),
        "vlm_score": vlm_score,
        "vlm_grades": vlm_grades,
        "avg_sharpness": round(sum(s["image"]["sharpness"] for s in per_scene) /
                               max(1, len(per_scene)), 1),
        "avg_brightness": round(sum(s["image"]["brightness"] for s in per_scene) /
                                max(1, len(per_scene)), 1),
        "config_snapshot": {"image_steps": int(CONFIG.get("auto_tune", {}).get("image_steps", 30)), "cfg": float(CONFIG.get("image_cfg", 6.0)),
                            "platform": CONFIG.get("platform"),
                            "video_engine": CONFIG.get("video_engine")},
        "scenes": per_scene,
    }
    log_path = ROOT / "quality_log.json"
    history = []
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(log_entry)
    history = history[-200:]  # qa.md: logs cap at 200 rows, never unbounded
    # Atomic: evolve_quality() reads this file to decide what to tune. A torn
    # write froze the tuner on stale history for days with no signal.
    dump_json_atomic(log_path, history, indent=2)
    _log(f"  Quality score: {overall}/100 "
        f"(image {log_entry['avg_image']}, audio {log_entry['avg_audio']})")
    return log_entry


def _num(x, default):
    """Coerce to float or default — quality logs must never crash the loop."""
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else default


def _evolve_signals(history, window=5):
    """Aggregate learnable signals over the recent window.

    Backward-compatible: old entries lack align/face_flag/reframed —
    defaults keep them neutral so history never breaks the loop.
    Malformed rows (non-dict scenes, None metrics) are skipped, never fatal.
    """
    if not history:
        return {"avg_sharp": 50.0, "avg_bright": 50.0, "avg_align": None,
                "reframe_rate": 0.0, "reframes_used": 0, "face_rate": 0.0,
                "clip_rate": 100.0, "score_now": None, "score_prev": None,
                "face_coverage": 0.0}
    recent = [e for e in history[-window:] if isinstance(e, dict)]
    if not recent:
        return {"avg_sharp": 50.0, "avg_bright": 50.0, "avg_align": None,
                "reframe_rate": 0.0, "reframes_used": 0, "face_rate": 0.0,
                "clip_rate": 100.0, "score_now": None, "score_prev": None,
                "face_coverage": 0.0}
    n = max(1, len(recent))
    avg_sharp = sum(_num(e.get("avg_sharpness", 50), 50) for e in recent) / n
    avg_bright = sum(_num(e.get("avg_brightness", 50), 50) for e in recent) / n
    alg, refr, flag, clip, tot, covered = [], 0, 0, [], 0, 0
    for e in recent:
        if not isinstance(e, dict):
            continue
        sc = e.get("scenes", []) or []
        for s in sc:
            if not isinstance(s, dict):
                continue
            tot += 1
            if "reframed" in s or "face_flag" in s or "align" in s:
                covered += 1
            a = s.get("align", None)
            if isinstance(a, (int, float)) and not isinstance(a, bool):
                alg.append(float(a))
            if s.get("reframed", False) is True:
                refr += 1
            if s.get("face_flag", ""):
                flag += 1
        clip.append(_num(e.get("clip_rate", 100), 100))
    return {
        "avg_sharp": avg_sharp,
        "avg_bright": avg_bright,
        "avg_align": sum(alg) / len(alg) if alg else None,
        "reframe_rate": refr / max(1, tot),
        "reframes_used": refr,
        "face_rate": flag / max(1, tot),
        "clip_rate": sum(clip) / len(clip) if clip else 100.0,
        "score_now": _num(recent[-1].get("score", None), None) if isinstance(recent[-1], dict) else None,
        "score_prev": _num(recent[-2].get("score", None), None) if len(recent) > 1 and isinstance(recent[-2], dict) else None,
        "face_coverage": covered / max(1, tot),
    }


def _ledger_load():
    p = ROOT / "tuning_ledger.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    except Exception:
        return []
    if not isinstance(data, list):
        try:
            p.rename(p.with_name(f"tuning_ledger.corrupt.{time.strftime('%Y%m%dT%H%M%S')}.json"))
        except Exception:
            pass
        return []
    rows = [e for e in data if isinstance(e, dict)]
    if len(rows) != len(data):
        # Corrupt rows are dropped, not fatal — but back the file up first.
        try:
            p.rename(p.with_name(f"tuning_ledger.corrupt.{time.strftime('%Y%m%dT%H%M%S')}.json"))
        except Exception:
            pass
    return rows


def _ledger_save(ledger):
    try:
        dump_json_atomic(ROOT / "tuning_ledger.json", ledger[-200:], indent=2)
    except Exception as e:
        _log(f"  [WARN] ledger persist failed: {e}")


def _propose(key, detail):
    """Human-review proposals: never auto-applied. Deduped by open key."""
    p = ROOT / "tuning_proposals.json"
    try:
        props = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    except Exception:
        props = []
    if any(x.get("key") == key and x.get("status") == "pending" for x in props):
        return
    props.append({"key": key, "detail": detail,
                  "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "pending"})
    try:
        dump_json_atomic(p, props[-200:], indent=2)
    except Exception as e:
        _log(f"  [WARN] proposal persist failed: {e}")
        return
    _log(f"  Evolve proposal [{key}]: {detail} (pending — human flips status)", 1)


def evolve_quality():
    """Closed-loop self-improvement: analyze recent renders, safe-apply
    allowlisted params within bounds + cooldown, revert on measured harm,
    and file human proposals for everything else.

    Allowlist (auto): image_steps 25..50 (downward only).
    Face budget is propose-only (never auto-applied).
    Anything else -> tuning_proposals.json.
    """
    _sync()
    log_path = ROOT / "quality_log.json"
    if not log_path.exists():
        return
    try:
        with open(log_path, encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        return
    if not isinstance(history, list) or not history:
        return
    ledger = [e for e in _ledger_load() if isinstance(e, dict)]
    # render_idx must survive a trimmed quality_log (else cooldowns freeze).
    # render_idx must survive a trimmed quality_log (else cooldowns freeze).
    render_idx = max(len(history),
                     max((e.get("render_idx", 0) for e in ledger), default=0) + 1)
    sig = _evolve_signals(history)
    tune = CONFIG.setdefault("auto_tune", {})
    changed = []

    def cooled(param, gap=2):
        """True if param untouched for >=gap renders (reverts count too —
        otherwise change->hurt->revert->change flaps forever)."""
        for e in reversed(ledger):
            if e.get("param") == param:
                return (render_idx - e.get("render_idx", 0)) >= gap
        return True

    def apply(param, old, new, target, rationale):
        # ponytail: NOT an assert. `python -O` strips asserts, which would
        # silently disable the allowlist and let any param auto-apply.
        if param not in AUTO_ALLOWLIST:
            raise ValueError(f"refusing to auto-apply non-allowlisted param {param!r}")
        tune[param] = new
        ledger.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "render_idx": render_idx, "param": param,
                       "old": old, "new": new, "target": target,
                       "target_before": sig[target] if target in sig else None,
                       "outcome": "pending", "rationale": rationale})
        changed.append(f"{param}->{new} ({rationale})")

    # 0. Impact review: judge every pending change by its target metric
    # (cheap arithmetic — no reason to lag one render per change).
    # "proposed" entries (face budget) are human-applied, never judged here.
    for e in reversed(ledger):
        if e.get("outcome") == "proposed":
            continue
        if e.get("outcome") != "pending" or e.get("target_before") is None:
            continue
        if not isinstance(e.get("param"), str) or not isinstance(e.get("target"), str):
            continue
        now = sig.get(e["target"])
        if now is None:
            continue
        try:
            delta = float(now) - float(e["target_before"])
        except Exception:
            continue
        if delta < -5:  # measurably hurt -> revert once, never re-apply blindly
            if e["param"] in AUTO_ALLOWLIST:
                tune[e["param"]] = e["old"]
                e["outcome"] = "reverted"
                changed.append(f"{e['param']} reverted to {e['old']} "
                               f"({e['target']} {e['target_before']:.0f}->{now:.0f})")
            else:
                e["outcome"] = "neutral"
        elif delta > 2:
            e["outcome"] = "helped"
        else:
            e["outcome"] = "neutral"

    # Retire pre-fix bright_boost rows so they can never write the dead key
    # back into config.json, and drop the key itself from the live config.
    if "bright_boost" in tune:
        tune.pop("bright_boost", None)
        changed.append("bright_boost retired (measured-inert prompt string)")
    for e in ledger:
        if e.get("param") == "bright_boost" and e.get("outcome") in ("pending", "helped", "neutral"):
            e["outcome"] = "retired"

    # 1. Sampler steps. Upward auto-tuning is REMOVED on purpose.
    # It used to raise steps whenever avg_sharp < 45, which is how a real
    # ch900 render ended up at 40 steps chasing "soft frames" when the
    # actual fault was the karras schedule — 30% more GPU time, zero
    # improvement, and it buried the real bug in the auto_tune history.
    # Measured: 25 vs 40 steps on the correct schedule changes brightness
    # (21.4 -> 31.6 mean luma) but not sharpness. Official Qwen-Image-2.1
    # baseline is 25. Only the downward path stays automatic, because
    # "images are already sharp, save the time" is a real win.
    try:
        steps = min(50, max(25, int(tune.get("image_steps", 25))))
    except Exception:
        steps = 25
    if sig["avg_sharp"] > 75 and steps > 25 and cooled("image_steps"):
        apply("image_steps", steps, max(25, steps - 5),
              "avg_sharp", "sharp, saving time")
    # 2. bright_boost arm REMOVED. It was a prompt prefix ("bright vivid
    # lighting, ") and a Qwen-Image still generator ignored it: quality_log
    # 37->38->39 shows brightness 17.9->15.2->17.9 (A-B-A) with the ledger
    # recording set/revert/set, because the controller measured its own noise
    # and read it as harm. An inert knob in a closed loop burns a full episode
    # render per cycle. Brightness is now a human decision, not an auto-tune.
    # 3. Face-gate budget follows MEASURED reframe pressure (2..6).
    # Coverage gate: old-schema history has no reframed keys — missing data
    # must never read as "measured zero" (once proposed a false trim).
    try:
        _fb = min(6, max(2, int(CONFIG.get("face_check_max_reframes", 4))))
    except Exception:
        _fb = 4
    _used = sig["reframes_used"]
    _cov = sig.get("face_coverage", 0)
    _last_fb = next((e.get("new") for e in reversed(ledger) if e.get("param") == "face_budget"), None)
    if _cov < 0.5:
        pass  # no measured data yet — stay silent
    elif sig["reframe_rate"] > 0.15 and _used >= _fb and _fb < 6 and cooled("face_budget") and _last_fb != _fb + 1:
        ledger.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "render_idx": render_idx, "param": "face_budget",
                       "old": _fb, "new": _fb + 1, "target": "face_rate",
                       "target_before": sig["face_rate"], "outcome": "proposed",
                       "rationale": f"budget exhausted ({_used}/{_fb} used)"})
        changed.append(f"face_budget->({_fb + 1} proposed in ledger; set face_check_max_reframes to apply)")
    elif sig["reframe_rate"] == 0 and _fb > 2 and cooled("face_budget") and _last_fb != _fb - 1:
        ledger.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "render_idx": render_idx, "param": "face_budget",
                       "old": _fb, "new": _fb - 1, "target": "face_rate",
                       "target_before": sig["face_rate"], "outcome": "proposed",
                       "rationale": "budget unused, trimming"})
        changed.append(f"face_budget->({_fb - 1} proposed in ledger; set face_check_max_reframes to apply)")
    # NOTE: face budget only PROPOSES (needs a config edit to take effect) —
    # auto-touching GPU-spend knobs without a human in the loop is how
    # pipelines surprise their owners with 2x render times.

    # 4. Human proposals (never auto-applied; skipped with no measured data).
    if sig["avg_align"] is not None and sig["avg_align"] < 50:
        _propose("align", f"avg text-image alignment {sig['avg_align']:.0f} — "
                 "review image_prompt wording vs spoken lines")
    if _cov >= 0.5 and sig["face_rate"] > 0.25:
        _propose("faces", f"{sig['face_rate'] * 100:.0f}% scenes face-flagged — "
                 "review sheet hair/fringe, pose presets, or headroom weight")
    if sig["clip_rate"] < 100:
        _propose("clips", f"clip_rate {sig['clip_rate']:.0f}% — inspect failed scenes before next render")
    if (sig["score_now"] is not None and sig["score_prev"] is not None
            and sig["score_now"] < sig["score_prev"] - 10):
        _propose("regression", f"score {sig['score_prev']:.0f}->{sig['score_now']:.0f} — "
                 "compare takes before locking params")

    if changed:
        _hist = tune.setdefault("history", [])
        _hist.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "changes": changed})
        del _hist[:-50]  # QoL: ledger capped at 200; history was unbounded
        # ponytail: the ledger is written only AFTER config.json is replaced.
        # Saving it first meant a failed persist left a "pending" row
        # recording a change that never reached config.json — and the
        # cooldown then suppressed the retry for the next 2 renders.
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                disk = json.load(f)
            disk["auto_tune"] = tune
            # Shared atomic writer. with_suffix(".tmp") replaced the LAST
            # suffix, so any config named config.prod.json became
            # config.prod.tmp — and, more importantly, this hand-rolled pair
            # had no fsync and no raise, so a failed swap left the owner
            # believing a new config was on disk.
            dump_json_atomic(CONFIG_FILE, disk, indent=2)
        except Exception as e:
            _log(f"  [WARN] Could not persist auto-tune: {e}")
            return
        # config.json now holds the change, so recording it is truthful.
        _ledger_save(ledger)
        for c in changed:
            _log(f"  Evolve: {c}")
    else:
        _ledger_save(ledger)
        _log("  Evolve: params optimal, no changes")
