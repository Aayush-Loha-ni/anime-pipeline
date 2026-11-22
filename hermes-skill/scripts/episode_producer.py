#!/usr/bin/env python3
"""
Anime Pipeline - Episode Producer
Orchestrates the full video production pipeline:
    llama-server (Gemma-4-26B-A4B) -> ComfyUI (Qwen-Image-2.1 + MiniMax H3) -> Qwen3-TTS clone -> FFmpeg

Usage:
  python episode_producer.py 63
  python episode_producer.py 63 70
  python episode_producer.py --chapter 63 --platform youtube
  python episode_producer.py --check
"""

import argparse
import base64
import errno
import hashlib
import http.client
import json
import math
import threading
from contextlib import contextmanager
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import urllib.request
from urllib.parse import urlparse
import shlex
import urllib.error
import wave
from pathlib import Path

from json_util import loads_json, dump_json_atomic, load_json_file
from reference_selector import ReferenceImageSelector
from best_image_selector import BestImageSelector

# Pipeline upgrade modules
# ponytail: only what is actually called. comfy_patch (426 lines) and
# migrate were imported here and never invoked from the render path; both
# are now deleted outright, along with style_bible and four rights symbols
# that were also dead imports (2026-09-29).
try:
    from pipeline.director_package import (generate_director_package as _gen_dp,
                                            save_director_package, load_director_package)
    from pipeline.rights_manifest import (load_rights_manifest, preflight_rights_gate)
    from pipeline.audio_planner import generate_audio_plan, save_audio_plan
    from pipeline.subtitle_generator import generate_subtitles
    from pipeline.release_report import (create_release_report, save_release_report)
    from pipeline.release_package import build_release_package
    from pipeline.analytics_log import add_entry as add_analytics_entry
    from pipeline import config_loader as yaml_config
    from pipeline.viral_narration import viral_prompt_block
    _PIPELINE_UPGRADE = True
except ImportError:
    _PIPELINE_UPGRADE = False

# Novel registry (stdlib-only, so it cannot take the upgrade module set down).
from pipeline.novels import get_novel, DEFAULT_NOVEL_ID, DEFAULT_NOVEL_TITLE

# ponytail: UTF-8 stdout/stderr up front. The codex logs '←' on every phase
# gate; on a cp1252 console print() raises UnicodeEncodeError and the whole
# render dies inside log(). Piped/detached launches inherit UTF-8, but a
# plain CLI run must never crash on a log arrow.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- Configuration -------------------------------------------------
ROOT = Path(__file__).parent.parent.parent  # Anime Series root
CONFIG_FILE = Path(__file__).parent / "config.json"


def _home(*parts):
    # ponytail: portable %USERPROFILE% join. Identical to the old
    # <USER_HOME> literal on this machine; works on any box.
    return str(Path.home().joinpath(*parts))


def load_config():
    """Load config with defaults + YAML two-tier engine config."""
    defaults = {
        "llm_url": "http://127.0.0.1:8080",
        "llm_model": "gemma-4-26B-A4B-uncensored-IQ4_XS",
        "llm_temperature": 0.2,
        "llm_seed": 42,
        "llm_exe": _home("llama-server", "llama-cpp", "llama-server.exe"),
        "llm_gguf": _home("llama-server", "models", "gemma-4-26B-A4B-it-ultra-uncensored-heretic.i1-IQ4_XS.gguf"),
        "llm_vlm_gguf": _home("llama-server", "models", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
        "llm_vlm_mmproj": _home("llama-server", "models", "mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
        "llm_vision_enabled": True,
        "comfyui_url": "http://localhost:8188",
        "tts_url": "http://localhost:5003",
        "output_dir": str(ROOT / "novels" / "Example Novel" / "videos"),
        "chapters_dir": str(ROOT / "novels" / "Example Novel" / "chapters"),
        "images_dir": str(ROOT / "novels" / "Example Novel" / "images"),
        "scenes_dir": str(ROOT / "novels" / "Example Novel" / "scenes"),
        "characters_file": str(ROOT / "novels" / "Example Novel" / "characters" / "characters.json"),
        "music_dir": str(ROOT / "novels" / "Example Novel" / "music"),
        "sfx_dir": str(ROOT / "novels" / "Example Novel" / "sfx"),
        "hook_enabled": False,
        "ffmpeg": "ffmpeg",
        "width": 1920,
        "height": 1080,
        "platform": "youtube",
        "voice_speed": 1.0,
        # P3: per-voice speed preset (voice_speeds) over global voice_speed.
        "voice_speeds": {},
        "vlm_narrative": False,
        "qa_auto_editor": True,
        "comfy_exe": _home("AppData", "Local", "Programs", "Comfy Desktop", "Comfy Desktop.exe"),
        # Burn captions into the picture. Enabled 2026-10-06: the caption burn
        # was structurally broken (absolute path into the subtitles filter dies
        # on the drive colon) so it could never have worked, and a soft
        # mov_text track is invisible unless the viewer turns captions on.
        "subtitles": True,
        # Vertical ONLY — see create_ass_subtitles. On a 16:9 render this value
        # pushed captions 30% up the frame; horizontal uses 10% regardless.
        "caption_safe_margin_v": 320,
        "kinetic_captions": False,
        # Dead: create_ass_subtitles now emits one cue per line and only splits
        # when the text will not fit. Kept so an old config that sets it does
        # not crash on load.
        "caption_chunk_words": 3,
        "music_volume": 0.15,
        "audio_ducking": True,
        "lufs_target": -14,
        "sfx_whoosh": True,
        "fps": 24,
        "scene_duration": 5,
        # Chapter text handed to the breakdown LLM. Must leave room inside
        # llm_ctx for the prompt (~5.5k tok) plus llm_script_max_tokens (14336).
        # At 44000 the whole corpus still fits whole (longest chapter is 40764
        # chars) while 5454+12222+14336 = 32012 < 32768 ctx. Raise llm_ctx
        # before raising this.
        "llm_chapter_chars": 44000,
        # Audio drives shot length now (TTS runs before the clip phase).
        # The cap is what stops a 40s narration becoming a 40s diffusion
        # pass; past it the clip loops in assembly exactly as it used to.
        "max_scene_clip_sec": 12,
        "beat_min_sec": 6,
        # Qwen-Image-2.1 is a FLOW-MATCHING model. Comfy-Org's own templates
        # and Qwen's repo both ship: steps 25, cfg 1, euler, simple.
        #   - karras sigmas are wrong for flow matching ("sigm are vastly
        #     compressed under FlowMatch which negates their usability").
        #     Measured on the real model, dpmpp_2m/euler + karras produced
        #     near-black stills (mean luma 3.9/4.7) while every non-karras
        #     schedule was healthy (simple 31.6, normal 31.7, beta 55.9).
        #   - cfg above ~2 oversaturates and adds edge artefacts; the official
        #     workflow does not use CFG at all.
        "image_cfg": 1.0,
        "image_sampler": "euler",
        "image_scheduler": "simple",
        "face_check_enabled": True,
        # m1: retry offset 50000 — 5000 collided across chapters
        # (ch1-S1-retry == ch6-S1-normal under base+chapter*1000+scene).
        "face_check_retry_offset": 50000,
        "face_check_max_reframes": 4,
        "video_engine": "h3",
        "motion_prompt_extend": False,
        # H3 follows Comfy-Org's video_minimax_h3_r2v recipe: 20 steps,
        # res_multistep/simple, the int8 video VAE + the audio VAE, and the
        # 32B Qwen3-VL encoder on the GPU. The old values (8 steps, fp16 VAE,
        # encoder on CPU, a 3-LoRA LoraLoader chain) produced clips that
        # ignored the reference still entirely.
        "h3_steps": 20,
        "h3_frames": 124,
        "h3_fps": 24,
        "h3_final_steps": 20,
        "h3_final_frames": 243,
        "h3_final_fps": 24,
        "h3_audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
        "h3_turbo_lora": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "h3_turbo_strength": 1.0,
        "h3_encoder_device": "default",
        "h3_sampler": "res_multistep",
        "h3_scheduler": "simple",
        # Stills: TRIED the GPU and reverted. Same seed, CPU 138s vs GPU 58s
        # per still (2.4x) with an identical picture (correlation 0.99889,
        # MAE 0.415). But in a real render the extra resident VRAM tips
        # ComfyUI's memory planner over: "aimdo memory compile error" ->
        # "Fatal Python error: Aborted", ComfyUI dies, and every remaining
        # clip falls back to a panning still. A 2.4x still speedup is not
        # worth losing the whole render, so the encoder stays on the CPU.
        # ponytail: this is the one real speed lever left and it is closed.
        # Revisit only on a card with more than 16 GB, or once the still UNET
        # gets smaller.
        "still_encoder_device": "cpu",
        "lora_strength": 0.7,
        "assembly_version": "v2",
        "cugan_enabled": True,
        "rife_enabled": True,
        "image_seed_base": 2000,
        "checkpoint": "qwen_image_2.1_Q6_K.gguf",
        "qwen_text_encoder": "qwen3vl_8b_int8_convrot.safetensors",
        "qwen_vae": "qwen_image_2.1_vae_bf16.safetensors",
        "comfyui_input": "",
        "comfyui_output": "",
        "gpu_max_temp": 85,
        "gpu_cooldown_temp": 78,
        "echo_enabled": True,
        "lipsync_engine": "echo",
        "echo_version": "V1",
        "shorts_enabled": False,
        "echo_shots": ["close-up", "medium"],
        "echo_size": 512,
        "echo_frames": 100,
        "echo_steps": 6,
        "echo_cfg": 2.5,
        "echo_fps": 24,
        "echo_infer_mode": "audio_drived_acc",
        "echo_vae": "sd-vae-ft-mse.safetensors",
        "narration_only": True,
        "animation_style": "cinematic",
        # Every key in _CONFIG_SCHEMA needs a shipped default, otherwise an
        # unparsable value falls back to 0/0.0 and disarms the setting.
        # The sheet_anchor_* keys are INERT: the sheet anchor they configured
        # is gone. They stay (with their defaults) because they are part of
        # the still cache key — dropping them would invalidate every cached
        # still in the project for a knob that no longer changes a pixel.
        "sheet_anchor_denoise": 0.7,
        "quality": "fast",
        # Feed the previous shot's last frame to H3 as a second reference
        # (<Picture 2>) so identity/lighting carry across cuts. Costs one
        # extra reference encode per clip; disable to go back to single-ref.
        "h3_continuity_relay": True,
        # Temporal luminance smoothing on generated clips (ffmpeg deflicker).
        # Safe: measured on the 19 real H3 clips of ch001 it moved episode
        # mean luma by -0.33 and preserved scene ordering exactly (r=1.000).
        # But only a ~3% median cut in frame-to-frame shimmer, because real
        # H3 output is already temporally steady (1.3-20.7 luma std). It
        # rides the encode we already run, so it stays on — the win is
        # small but free and riskless.
        "clip_deflicker": True,
        # Cross-clip colour/exposure matching. DEFAULT OFF, deliberately.
        #
        # Measured on the 19 real H3 clips of ch001: luma ranged 18.3..114.4
        # (spread 96/255) and r-b ranged -40..+70. Grading every clip to a
        # common reference cut that spread to 7.3 (a 94% "improvement") —
        # but the brightness ORDERING correlation came out at -0.195. The
        # fire scene went 114 -> 45 and the dark alley 18 -> 46. Every scene
        # ended at the same luma.
        #
        # That spread was never drift. It was the content: a burning fire, a
        # white sky, unlit corridors. All 19 clips shared the mood "somber",
        # so mood is not a usable grouping key, and the pipeline has no
        # location information to tell "same place" from "different place".
        # Without a signal that separates intent from error, any global
        # grade flattens the episode's lighting.
        #
        # Capping the correction does NOT rescue it. The cap is 107/255,
        # wider than any real delta, so a fire/corridor/alley triple still
        # inverts perfectly: 112 -> 45, 45 -> 45, 16 -> 47 (r = -1.0). The
        # code is retained because it is correct and would work once scenes
        # carry a location key, but it must stay off.
        #
        # ponytail: the synthetic-clip tests said this was a 20.0 -> 0.0 win.
        # Solid-colour clips are useless for calibrating a colour filter;
        # only real footage gives a valid number. Re-measure here before
        # flipping this back on.
        "clip_colour_grade": False,
        "lra_max": 14.0,
        # Opt-in auto-publish after a render. OFF by default: nothing leaves
        # the machine unless this flips AND platforms are listed. Credentials
        # live in ~/.anime_pipeline_publish.json (see pipeline/publisher.py);
        # Instagram additionally needs publish.instagram.video_url because the
        # Graph API fetches the video from a public URL.
        "publish": {"enabled": False, "platforms": [], "privacy": "private"},
    }
    _DEFAULTS.update(defaults)  # pre-override snapshot for _validate_config

    user_config = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        defaults.update(user_config)

    # Merge YAML config (engine-specific params).
    # ponytail: config.json is authoritative (SKILL.md), but this merge used
    # to run AFTER the JSON load and overwrite it, so editing config.json
    # silently did nothing for every h3_* key. JSON now wins; the YAML file
    # only fills gaps and warns on disagreement instead of quietly winning.
    try:
        yaml_cfg = yaml_config.load_config()
        engine_cfg = yaml_config.get_active_engine_config(yaml_cfg)
        _yaml_vals = {f"h3_{k}": v for k, v in engine_cfg.items()
                      if not str(k).startswith("_")}
        _yaml_vals["video_engine"] = engine_cfg["_engine"]
        _yaml_vals["quality"] = engine_cfg["_quality"]
        for _k, _yv in _yaml_vals.items():
            _in_json = _k in user_config
            _jv = user_config.get(_k)
            if _in_json and str(_jv) != str(_yv):
                print(f"  [CONFIG] {_k}: config.yaml={_yv!r} ignored — "
                      f"config.json={_jv!r} is authoritative", file=sys.stderr)
            elif not _in_json:
                defaults.setdefault(_k, _yv)
        defaults["_yaml_fallback"] = engine_cfg.get("_fallback", True)
    except Exception as e:
        print(f"  [CONFIG] YAML merge skipped: {e}", file=sys.stderr)

    # ponytail: a bad value falls back to the shipped default, never to
    # typ() — int() is 0 and float() is 0.0, which re-arms the seed
    # collision (face_check_retry_offset) and the heat guard (gpu_max_temp).
    _validate_config(defaults, _DEFAULTS)
    return defaults


# Shipped defaults, captured before user config / YAML override them.
# _validate_config falls back to these instead of typ() (0 / 0.0).
_DEFAULTS = {}


_CONFIG_SCHEMA = {
    # key: (type, min, max)  — None means no bound
    "llm_temperature": (float, 0.0, 2.0),
    "llm_seed": (int, 0, None),
    # Upper bound is the context wall, not a taste limit: prompt + chapter +
    # llm_script_max_tokens must fit inside llm_ctx. 44000 is safe at the
    # current 32768 ctx; raise llm_ctx first if a chapter ever needs more.
    "llm_chapter_chars": (int, 2000, 200000),
    "image_cfg": (float, 0.0, 4.0),
    "image_sampler": (str, None, None),
    "still_encoder_device": (str, None, None),
    "image_scheduler": (str, None, None),
    "image_seed_base": (int, 0, 999999),
    "h3_steps": (int, 1, 100),
    "h3_frames": (int, 5, 362),
    "h3_fps": (int, 8, 60),
    "fps": (int, 8, 120),
    "scene_duration": (float, 1.0, 30.0),
    "max_scene_clip_sec": (float, 1.0, 60.0),
    "beat_min_sec": (float, 0.0, 60.0),
    "voice_speed": (float, 0.5, 2.0),
    "music_volume": (float, 0.0, 1.0),
    "lufs_target": (float, -30.0, 0.0),
    "echo_size": (int, 128, 2048),
    "echo_frames": (int, 8, 500),
    "echo_steps": (int, 1, 100),
    "echo_cfg": (float, 0.1, 20.0),
    "echo_fps": (int, 8, 60),
    "face_check_max_reframes": (int, 0, 12),
    "face_check_retry_offset": (int, 0, 999999),
    "sheet_anchor_denoise": (float, 0.0, 1.0),
    "lora_strength": (float, 0.0, 2.0),
    "gpu_max_temp": (int, 60, 105),
    "gpu_cooldown_temp": (int, 50, 100),
    "quality": (str, None, None),  # fast | final
}

# string keys whose valid values are a fixed set (bounds can't express this)
_CONFIG_ENUMS = {"quality": ("fast", "final")}


def _validate_config(cfg, fallback=None):
    """Coerce bad types and clamp out-of-range values. Logs warnings.

    `fallback` supplies the shipped default for an unparsable value; without
    it the old behaviour of typ() (0 / 0.0) applies, which silently disarms
    the heat guard and re-arms the documented seed collision.
    """
    fallback = fallback or {}
    for key, (typ, lo, hi) in _CONFIG_SCHEMA.items():
        if key not in cfg:
            continue
        val = cfg[key]
        try:
            coerced = typ(val)
            if isinstance(coerced, float) and not math.isfinite(coerced):
                raise ValueError("non-finite")
        except (ValueError, TypeError):
            default = fallback.get(key)
            try:
                coerced = typ(default) if default is not None else typ()
            except (ValueError, TypeError):
                coerced = typ()
            print(f"  [CONFIG] {key}={val!r} invalid, restoring default {coerced!r}", file=sys.stderr)
        if lo is not None and coerced < lo:
            print(f"  [CONFIG] {key}={coerced} below min {lo}, clamped", file=sys.stderr)
            coerced = lo
        if hi is not None and coerced > hi:
            print(f"  [CONFIG] {key}={coerced} above max {hi}, clamped", file=sys.stderr)
            coerced = hi
        # ponytail: (str, None, None) never fails to coerce, so `quality`
        # accepted "final " / "123" / "" silently and the tier resolved
        # elsewhere. Enum-validate the one unconstrained string.
        if key in _CONFIG_ENUMS and coerced not in _CONFIG_ENUMS[key]:
            good = fallback.get(key) or next(iter(_CONFIG_ENUMS[key]))
            print(f"  [CONFIG] {key}={coerced!r} not in {sorted(_CONFIG_ENUMS[key])}, "
                  f"restoring default {good!r}", file=sys.stderr)
            coerced = good
        cfg[key] = coerced


ANIMATION_STYLES = {
    "cinematic": {
        "label": "Cinematic anime",
        "still": "anime style, anime screenshot, anime key visual, official anime art",
        "motion": "anime style",
        "negative": "",
    },
    "retro-cel": {
        "label": "Retro 90s cel",
        "still": "retro 90s anime cel, vintage cel animation, visible film grain, muted jewel tones",
        "motion": "retro 90s anime cel animation",
        "negative": "modern glossy cgi, over-saturated digital colors",
    },
    "painterly": {
        "label": "Painterly light",
        "still": "painterly anime background art, luminous scenic lighting, detailed painted clouds and light rays",
        "motion": "painterly anime animation, luminous lighting",
        "negative": "flat vector look, harsh neon",
    },
    "gothic": {
        "label": "Dark gothic",
        "still": "dark gothic anime, moody chiaroscuro, deep shadows, candlelit warmth against cold fog",
        "motion": "dark gothic anime animation, moody atmosphere",
        "negative": "bright cheerful palette, pastel",
    },
    "noir": {
        "label": "Ink noir",
        "still": "high-contrast ink noir anime, heavy black ink shadows, restrained monochrome-tinted palette",
        "motion": "high-contrast ink noir animation",
        "negative": "colorful, pastel, low contrast",
    },
    "soft": {
        "label": "Soft watercolor",
        "still": "soft watercolor anime, gentle washes, delicate linework, airy daylight",
        "motion": "soft watercolor anime animation, gentle motion",
        "negative": "harsh contrast, heavy shadows",
    },
}


def _animation_style(scene):
    name = str((scene or {}).get("animation_style", "") or CONFIG.get("animation_style", "cinematic")).lower()
    return name if name in ANIMATION_STYLES else "cinematic"


def _novel_art():
    """The active novel's art direction, as prompt fragments.

    Only a NON-legacy novel contributes: the legacy novel's stills are cached
    and its art_direction was never part of the prompt, so injecting it now
    would invalidate every cached still for nothing.
    """
    n = active_novel()
    if n is None or n.legacy:
        return {}
    art = n.art_direction or {}
    return {k: str(art.get(k) or "").strip()
            for k in ("still", "motion", "negative")
            if str(art.get(k) or "").strip()}


def _novel_style_lora():
    """The novel's own STYLE LoRA, if it has one: {name, strength, trigger}.

    A per-novel style LoRA is the strongest way to hold a niche look across
    EVERY episode of a book - ComfyUI's own style-transfer handbook says it
    plainly: "for niche styles, train a LoRA", and Qwen-Image chains LoRAs
    natively, so the style sits UNDER the character LoRA. Train one with the
    pipeline's own recipe: stage captioned frames in dataset/<name>_lora/
    and run `_jobs/train_lora.py <name>`.
    """
    n = active_novel()
    if n is None or n.legacy:
        return None
    art = n.art_direction or {}
    name = str(art.get("lora") or "").strip()
    if not name:
        return None
    hits = sorted(f for d in _comfy_dirs("models") if (d / "loras").exists()
                  for f in (d / "loras").glob(f"{name}*.safetensors"))
    if not hits:
        return None
    qwen = [f for f in hits if _lora_is_qwen(f)]
    pick = (qwen or hits)[-1]
    return {"name": pick.name,
            "strength": float(art.get("lora_strength", 0.45)),
            "trigger": str(art.get("trigger") or "").strip()}


def _style_spec(scene):
    """The style block for a scene: the shared style, OWNED by the novel.

    A novel's art_direction REPLACES the shared style's fragment rather than
    being appended to it: "each novel has its own style" means the novel's
    text IS the look, not a suffix on someone else's. The legacy novel has no
    art_direction, so its resolution - and its cached stills - are untouched.
    """
    spec = dict(ANIMATION_STYLES[_animation_style(scene)])
    for key, frag in _novel_art().items():
        spec[key] = frag
    return spec


# Every CONFIG key that reaches the stills path — the KSampler block and the
# model loaders are all inside generate_image()/build_workflow().
# ponytail: this list is the single reason the still cache cannot go stale.
# It used to cover 4 of these; the other 10 were invisible to the key, so
# editing image_sampler or the character bible re-used the old PNG forever and
# shipped it as current. `test_upgrade_fixes.py` asserts every one of these
# keys moves the fingerprint, so adding a new knob without adding it here
# fails the suite instead of silently rotting.
#
# ponytail: the three sheet_anchor_* keys are INERT (the sheet anchor is
# gone) but they STAY in this list on purpose: the list is hashed into the
# still cache key, and removing them would invalidate every cached still in
# the project for a knob that no longer changes a pixel.
_STILL_CFG_KEYS = (
    "checkpoint", "image_cfg", "image_sampler", "image_scheduler",
    "image_seed_base", "lora_strength", "qwen_lora_trigger",
    "qwen_lora_trigger_owner",
    "qwen_text_encoder", "qwen_vae", "still_encoder_device",
    "sheet_anchor_denoise", "sheet_anchor_head_span", "sheet_anchor_head_frac",
    "animation_style", "width", "height", "platform",
)
# auto_tune is a DICT and is handled separately below: the still path reads
# exactly one field of it (`auto_tune.image_steps`), while `auto_tune.history`
# is an append-only changelog that quality_evolve rewrites on every tune event.
# Hashing the whole dict meant a pure log append rotated every still sidecar
# and forced a full ~90 min re-render for no pixel change (verified).
_AUTO_TUNE_KEY = "auto_tune.image_steps"
# Scene fields that are composed into the prompt (not just descriptive labels).
_STILL_SCENE_KEYS = (
    "image_prompt", "negative_prompt", "animation_style", "shot_size",
    "angle", "mood", "pose", "type", "camera_move", "animation_hint",
    "narration", "dialogue", "characters", "stages", "location",
)


def _visual_fingerprint(scene, still_path=None):
    """Everything that determines the pixels of one still, hashed.

    Three input classes, all of which changed the picture and were missing:
      1. CONFIG — sampler, scheduler, LoRA strength, the text encoder, the VAE,
         the encoder device, the seed base.
      2. The scene — prompt, negative, pose, stages, and the narration whose
         first 14 words are appended as "depicting:".
      3. World state — the character bible entry for every cast member and the
         setting's visual_tags. Editing a character's hair/eyes/clothing in
         characters.json used to be invisible to the key.

    (3) is hashed as a subtree rather than field-by-field on purpose: an
    enumerated field list goes stale the first time someone adds a bible field,
    which is exactly the failure this function exists to prevent.
    """
    scene = scene or {}
    cast = sorted(c.lower().replace(" ", "_")
                  for c in (scene.get("characters") or []) if isinstance(c, str))
    bible = CHAR_BIBLE if isinstance(CHAR_BIBLE, dict) else {}
    _h = hashlib.sha256()

    def _feed(label, value):
        _h.update(label.encode("utf-8"))
        _h.update(b"\x00")
        _h.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
        _h.update(b"\x00")

    for key in _STILL_CFG_KEYS:
        _feed(f"cfg:{key}", CONFIG.get(key))
    # Only the sampled field of auto_tune, not the dict and not its history.
    _auto = CONFIG.get("auto_tune")
    _feed(f"cfg:{_AUTO_TUNE_KEY}",
          _auto.get("image_steps") if isinstance(_auto, dict) else _auto)
    for key in _STILL_SCENE_KEYS:
        _feed(f"scene:{key}", scene.get(key))
    _feed("scene:pose_for", pose_for_scene(scene))
    _feed("scene:props_for", props_for_scene(scene))
    # Resolve the bible entry through its ALIASES. The script spelled the
    # character "ammon" and the bible spelled it "amon": the raw lookup found
    # nothing, so the stills that used that character were cached against a
    # null bible entry and a later change to the real entry never invalidated
    # them. The stills carried NO identity tags for that character.
    _feed("bible:characters", {c: _bible_entry(c) for c in cast})
    _feed("bible:settings", bible.get("settings", {}))
    _feed("derived:framing", 2)
    _feed("derived:style", _animation_style(scene))
    # A novel's art direction changes the pixels, so it must be in the key.
    # Empty for the legacy novel, so its cached stills keep validating.
    _art = _novel_art()
    if _art:
        _feed("derived:art", _art)
    # A novel's STYLE LoRA is part of the picture, so it belongs in the key.
    _style_lora = _novel_style_lora()
    if _style_lora:
        _feed("derived:style_lora", _style_lora)
    _feed("derived:shot", [_shot_norm(scene), str(scene.get("angle", "")),
                           str(scene.get("mood", ""))])
    if still_path is not None:
        try:
            _feed("still", _file_fingerprint(still_path))
        except OSError:
            pass
    return {"v": 2, "hash": _h.hexdigest()}


def _write_still_sidecar(still_path, scene):
    """Record which prompt/settings produced a still. Best effort by design.

    A missing sidecar is recoverable — _still_cache_valid treats it as
    "unverified" and keeps the PNG. A WRONG one is not, so this writes the
    fingerprint of the exact scene the image was generated from and never
    synthesises one for a file it did not produce.
    """
    try:
        p = Path(still_path)
        p.with_suffix(".cache.json").write_text(
            json.dumps(_visual_fingerprint(scene, still_path=p)), encoding="utf-8")
        return True
    except OSError as e:
        log(f"  [WARN] still provenance not recorded ({e})", 2)
        return False


def _pick_cached_still(images_dir, pattern, scene):
    """Newest sidecar-valid still matching `pattern`, or None.

    ponytail: extracted so the rule "existence is not validity" is testable on
    its own. Both the beat-2 (`_sNNb_`) and companion (`_sNNc_`) paths used a
    bare `if existing:` on a glob, which meant any file on disk counted — a
    still from a different prompt, or against an older character bible, was
    replayed forever. And because this lived inside a 1200-line function,
    mutation testing showed NO test could reach it.
    """
    candidates = [p for p in sorted(Path(images_dir).glob(pattern))
                  if _still_cache_valid(p, scene)]
    return _newest(candidates)


def _still_cache_valid(still_path, scene, adopt=False):
    """True when the still on disk can be reused for this scene.

    ponytail: the PNG is the artefact and the sidecar is only a note recording
    which prompt/settings produced it. Treating a MISSING note as "invalid"
    meant one stray cleanup silently threw away every good still and forced a
    ~90 min re-render of byte-identical images, so an absent note now means
    "unverified".

    "Unverified" must NOT be laundered into "verified". The note records
    provenance, so synthesising one from the CURRENT scene asserts "this PNG
    came from this prompt" precisely when that is unknowable -- and it sticks:
    a still from an older prompt would then match forever, shipping one stale
    frame invisibly. So the note is only ever written by a caller that has
    DECIDED to use the file (adopt=True), never by the purge path, and the
    adoption is logged rather than silent.

    A 0-byte PNG is a half-written artefact and is NEVER a keeper — on either
    path. The size check used to live inside the "no sidecar" branch, so a
    truncated still that happened to carry a valid sidecar sailed through and
    would have been assembled into the episode.
    """
    p = Path(still_path)
    sidecar = p.with_suffix(".cache.json")
    try:
        if p.stat().st_size <= 0:
            return False
    except OSError:
        return False
    if not sidecar.is_file():
        # A novel that OWNS its look must PROVE a still carries it. A frame
        # with no provenance note cannot be shown to match the book's art
        # direction, and ADOPTING one launders it into "verified" for good:
        # the note is stamped with the CURRENT fingerprint, so from then on
        # the still claims to have been made under the new style and the
        # style change can never reach it again. The first themed render
        # therefore has to REDRAW the unprovenanced frames instead of
        # adopting them.
        # A novel WITHOUT its own art direction keeps the old behaviour: the
        # legacy library must not be re-rendered over a missing note.
        if _novel_art():
            return False
        if not adopt:
            return True
        try:
            sidecar.write_text(
                json.dumps(_visual_fingerprint(scene, still_path=p)),
                encoding="utf-8")
            log(f"  adopted still with no provenance record: {p.name}", 1)
        except OSError:
            pass
        return True
    try:
        return json.loads(sidecar.read_text(encoding="utf-8")) == _visual_fingerprint(scene, still_path=p)
    except (OSError, ValueError):
        return False


def _phash(path, size=8):
    """64-bit perceptual hash from a grayscale downscale. Returns int or None.

    sha256 fingerprints are byte-exact, so they cannot see that two DIFFERENT
    scene prompts produced the SAME picture — the "25 identical stills"
    failure class. A perceptual hash can. ponytail: 8x8 mean-dct-free variant
    (average hash) is ~10 lines and is enough to catch a collapsed render;
    swap in a real DCT pHash only if collisions become a problem.
    """
    try:
        import numpy as _np
        from PIL import Image
        with Image.open(path) as im:
            small = _np.asarray(im.convert("L").resize((size, size), Image.BILINEAR),
                                dtype=_np.float32)
        bits = (small > small.mean()).flatten()
        out = 0
        for b in bits:
            out = (out << 1) | int(b)
        return out
    except Exception:
        return None


def _phash_distance(a, b):
    """Hamming distance between two 64-bit hashes."""
    return bin((a ^ b) & 0xFFFFFFFFFFFFFFFF).count("1")


def still_luma(still_path):
    """Mean 0-255 luma of a still, or None if it cannot be read."""
    try:
        import numpy as _np
        from PIL import Image
        with Image.open(still_path) as im:
            g = im.convert("L").resize((128, 128), Image.BILINEAR)
            return float(_np.asarray(g, dtype=_np.float32).mean())
    except Exception:
        return None


def find_black_stills(scene_images, min_luma=3.0):
    """A-stills that came back effectively black.

    Found by a real ch900 render: 3 of 10 generated stills had mean luma
    0.09 / 0.36 / 2.70, and the "stills complete on disk" gate passed them
    because it only checked that the file EXISTS. H3 then animated scene 1
    from a pure black frame and hallucinated a malformed face out of nothing,
    which shipped. The VLM review that would have caught it was down and only
    warns, so nothing failed the render.

    Returns [(scene_number, luma), ...] below min_luma.
    """
    out = []
    for i, img in enumerate(scene_images):
        if not img or not Path(img).exists():
            continue
        l = still_luma(img)
        if l is not None and l < min_luma:
            out.append((i + 1, l))
    return out


def _still_rms(a_path, b_path, size=(192, 108)):
    """RMS of the grayscale difference between two stills, or None.

    This is the ground-truth check the 64-bit aHash cannot be. The aHash keeps
    only an 8x8 mean-threshold, so for CORRELATED frames - same aspect, subject
    centred, same costume and location, which is most of a single episode - it
    reports distance 2 for pictures whose pixels differ by 29/255. Measured over
    every pair in ch001/2/6/7 (thousands of pairs) no two stills were ever closer
    than RMS 34, while a genuine repeat is ~0. So the hash picks candidates and
    this decides.
    """
    try:
        import numpy as _np
        from PIL import Image
        arrs = []
        for p in (a_path, b_path):
            with Image.open(p) as im:
                arrs.append(_np.asarray(im.convert("L").resize(size, Image.BILINEAR),
                                        dtype=_np.float32))
        d = arrs[0] - arrs[1]
        return float(_np.sqrt((d * d).mean()))
    except Exception:
        return None


def find_duplicate_stills(scene_images, threshold=4, rms_threshold=12.0):
    """Pairs of NEAR-IDENTICAL stills across scenes in one episode.

    Returns [(scene_a, scene_b, distance), ...].

    Two stages, because the hash alone is not trustworthy here. The 64-bit aHash
    is a cheap prefilter that finds candidate pairs; every candidate must then be
    CONFIRMED by pixel RMS, and only a confirmed pair is reported.

    Why two stages: the original single-stage gate assumed "a 64-bit aHash has a
    random-pair distance of ~32, so <=4 means the same picture". Measured across
    ch001/2/6/7 that premise is false - the mean distance between two DIFFERENT
    stills of the same character in the same street is 12.3, not 32, because the
    frames are not independent samples. So <=4 fired on 25 pairs of ch007 that are
    all distinct pictures (closest RMS 39.1), re-rolled them, made the count
    worse, and blocked the render. It also fired on ch006, which had shipped.

    rms_threshold=12 sits far from both populations: a genuine repeat of one
    picture is RMS ~0, and the closest pair observed across every chapter on
    disk (4,167 pairs over seven chapters) is 41.9. Measured separation is a
    41.9-wide empty gap - ch003 holds 3 real duplicates at RMS exactly 0.00 and
    nothing else below 41.91 - so the threshold has >3x margin on each side and
    real duplicates are still caught while similar-composition shots stop
    blocking renders.
    """
    hashes = {}
    paths = {}
    for i, img in enumerate(scene_images):
        if not img or not Path(img).exists():
            continue
        h = _phash(img)
        if h is not None:
            hashes[i] = h
            paths[i] = img
    dupes = []
    keys = sorted(hashes)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ia, ib = keys[a], keys[b]
            d = _phash_distance(hashes[ia], hashes[ib])
            if d > threshold:
                continue
            rms = _still_rms(paths[ia], paths[ib])
            # An unreadable pair is NOT silently promoted to a duplicate; the
            # hash already said "these look the same", so keep it visible.
            if rms is not None and rms > rms_threshold:
                continue
            dupes.append((ia + 1, ib + 1, d))
    return dupes


def clip_rgb_stats(video_path, samples=3, timeout=120):
    """Mean R/G/B/luma of a few frames spread through a clip.

    Reads raw rgb24 straight from ffmpeg — no temp PNGs, no exact-size
    guessing. Returns {"r","g","b","luma"} as 0..255 means, or None.

    Returns how many frames it ACTUALLY sampled in "frames": short clips
    cannot supply `samples` evenly-spaced frames, and silently averaging one
    frame instead of three would hand the grade loop a confident number
    built from almost no evidence.

    Known limit: 3 spot samples cannot see a ONE-frame flash. Use deflicker
    for temporal stability and treat this as an exposure/tint measure only.
    """
    try:
        import numpy as _np
    except Exception:
        return None
    try:
        _fps, dur = video_fps_duration(video_path)
        if not dur or dur <= 0:
            return None
        # step must not exceed the clip itself, or ffmpeg emits fewer frames
        # than asked for and we average a fraction of the evidence
        step = dur / float(max(1, samples))
        if step <= 0.05:
            return None
        proc = subprocess.run(
            [CONFIG["ffmpeg"], "-v", "error", "-i", str(video_path),
             "-vf", f"fps=1/{step:.4f},scale=32:32",
             "-frames:v", str(max(1, samples)),
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, timeout=timeout)
        raw = proc.stdout or b""
        if len(raw) < 3 * 32 * 32:
            return None
        frames = len(raw) // (3 * 32 * 32)
        arr = _np.frombuffer(raw[:frames * 3 * 32 * 32],
                            dtype=_np.uint8).reshape(-1, 3)
        means = arr.mean(axis=0)
        return {"r": float(means[0]), "g": float(means[1]), "b": float(means[2]),
                "luma": float(0.299 * means[0] + 0.587 * means[1] + 0.114 * means[2]),
                "frames": frames}
    except Exception:
        return None


def video_fps_duration(path, timeout=60):
    """(fps, duration) of a video, or (0, 0). One ffprobe call."""
    try:
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate:format=duration",
             "-of", "default=nw=1", str(path)],
            capture_output=True, text=True, timeout=timeout)
        if pr.returncode:
            return 0.0, 0.0
        fps, dur = 0.0, 0.0
        for line in (pr.stdout or "").splitlines():
            k, _, v = line.strip().partition("=")
            if k == "avg_frame_rate":
                n, _, d = v.partition("/")
                fps = (float(n) / float(d)) if float(d or 0) else 0.0
            elif k == "duration":
                dur = float(v or 0)
        return fps, dur
    except Exception:
        return 0.0, 0.0


def grade_consistency_filter(ref, cur, max_luma=0.42, max_gain=1.10):
    """ffmpeg filter nudging `cur` toward `ref`'s exposure/tint.

    Shot-to-shot exposure and tint jump is the most visible continuity
    defect and the pipeline had nothing for it: a grep for deflicker /
    colour-match across all 30 support modules returned zero hits.

    MEASURED on the 19 real H3 clips from the last render: luma ranged
    18.3..114.4 (spread 96/255) and r-b ranged -40..+70 (spread 110). So:

    - The correction is CAPPED at max_luma, not refused. A hard refusal
      above the cap did nothing useful — it skipped almost nothing, so the
      extreme clips got no correction at all while near ones got a full
      one. Capping gives every clip a bounded nudge and leaves the
      deliberate lighting (a fire scene brighter than a dark alley)
      recognisable.
    - Channel gains handle TINT ONLY, normalised to a mean of 1.0. eq
      already moves the level; a uniform gain would double-correct.
    - The caller passes an episode-median reference, not clip #1: one very
      dark or very bright shot should not define the look of all 19.

    Returns "" when the clips already agree.
    """
    if not ref or not cur:
        return ""
    d_luma = ref["luma"] - cur["luma"]
    parts = []
    want = d_luma / 255.0
    if abs(want) > max_luma:
        want = max_luma if want > 0 else -max_luma
    if abs(want) >= 0.002:
        parts.append(f"eq=brightness={want:.4f}")
    raw = {}
    for ch in ("r", "g", "b"):
        base = cur[ch]
        # floor at 1/max_gain rather than 1.0: a near-black channel must not
        # get gain 1.0 while the others are corrected, which rotates the hue
        # in proportion to how dark that channel happens to be
        raw[ch] = (min(max_gain, max(1.0 / max_gain, ref[ch] / base))
                   if base >= 1.0 else 1.0 / max_gain)
    mean_gain = sum(raw.values()) / 3.0
    gains = []
    for ch in ("r", "g", "b"):
        g = raw[ch] / mean_gain if mean_gain > 1e-6 else 1.0
        if abs(g - 1.0) > 0.005:
            gains.append(f"{ch}{ch}={g:.4f}")
    if gains:
        parts.append("colorchannelmixer=" + ":".join(gains))
    vf = ",".join(parts)
    return vf if vf else ""


def grade_reference_from(stats):
    """Episode reference look = the MEDIAN clip, not the first one.

    Measured across the 19 real clips, the first scene happened to be a
    mid-dark corridor and clip #21 was a fire at luma 114. Anchoring on
    clip #1 would drag the fire down to corridor level.
    """
    vals = [s for s in stats if s]
    if not vals:
        return None
    def med(ch):
        xs = sorted(s[ch] for s in vals)
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    return {"r": med("r"), "g": med("g"), "b": med("b"), "luma": med("luma")}


def _apply_clip_grade(src, dst, filter_chain, out_w, out_h, timeout=300,
                      target_fps=None):
    """Scale + (optional) grade + deflicker -> h264. Shared by the H3 path.

    When target_fps is given and the source is materially below it, the
    output is resampled to the timeline rate with `-r`. Be clear about what
    that is: a FRAME-RATE CONVERSION that duplicates frames, not optical
    flow. It removes the timebase mismatch on a cached low-fps clip but it
    does NOT synthesise intermediate motion — a 12fps source promoted this
    way still shows each frame twice. Real interpolation needs the RIFE
    enhancer (rife_double), which only runs on the frame-sequence path.
    """
    # ponytail: a bare scale was squashing 960x544 into 1920x1080 — 2.000x
    # horizontally against 1.985x vertically, a 0.74% stretch baked into every
    # generated clip. vertical_fit_filter (defined below, called at call time
    # so order is irrelevant) scales-to-cover and center-crops instead, and it
    # was already the helper every other upscale in this file used. sharpen=False
    # because this is the BUILD stage and NORM (:6612) sharpens once at the end —
    # running it here too oversharpens edges into shimmer.
    vf = vertical_fit_filter(out_w, out_h, sharpen=False)
    if filter_chain:
        vf = f"{vf},{filter_chain}"
    if CONFIG.get("clip_deflicker", True):
        # one frame of temporal averaging; kills the per-frame luminance
        # shimmer diffusion models produce without smearing motion
        vf = f"{vf},deflicker=size=5:mode=am"
    cmd = [CONFIG["ffmpeg"], "-y", "-i", str(src), "-vf", vf]
    if target_fps:
        src_fps, _ = video_fps_duration(src)
        if 0 < src_fps < target_fps * 0.9:
            cmd += ["-r", str(target_fps)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    return proc.returncode == 0 and Path(dst).exists()


def _tint_error(ref, cur):
    """Mean absolute per-channel difference, 0..255. The number a grade is
    actually trying to reduce — luma and tint are corrected separately."""
    if not ref or not cur:
        return 0.0
    return sum(abs(ref[c] - cur[c]) for c in ("r", "g", "b")) / 3.0


def grade_clip_to_reference(src, dst, ref, out_w, out_h, max_passes=2,
                            target_fps=None, converge=3.0):
    """Grade `src` onto `ref`'s exposure/tint, measuring after each pass.

    Closed loop, not a prediction. Building one filter from the measured
    delta and trusting it is not reliable enough: a tint correction also
    shifts luma (the gains are normalised to an arithmetic mean of 1.0, not
    a luma-weighted one), so a single predicted pass lands a few units off.
    So apply, re-measure the OUTPUT, and correct the residual once more.

    Returns (ok, filter_string, passes, tint_error_after). filter_string is
    "" when the clip already matched and no pass was needed.
    """
    if not ref:
        return (_apply_clip_grade(src, dst, "", out_w, out_h,
                                  target_fps=target_fps), "", 0, 0.0)
    chain, cur_src, passes, created = [], Path(src), 0, []
    for attempt in range(1, max(1, max_passes) + 1):
        cur = clip_rgb_stats(cur_src)
        if cur is None:
            break
        vf = grade_consistency_filter(ref, cur)
        if not vf:
            break
        last = attempt >= max_passes
        target = Path(dst) if last else Path(dst).with_name(
            Path(dst).stem + f".g{attempt}.mp4")
        if not last:
            created.append(target)
        if not _apply_clip_grade(cur_src, target, vf, out_w, out_h,
                                 target_fps=target_fps):
            target.unlink(missing_ok=True)
            _drop(created)
            return False, ",".join(chain), passes, 0.0
        chain.append(vf)
        passes = attempt
        cur_src = target
        if last:
            break
        after = clip_rgb_stats(target)
        if after is None or _tint_error(ref, after) <= converge:
            # Converged. RENAME rather than re-encode: deflicker is a
            # temporal filter, so a second pass through it is neither free
            # nor idempotent — this halved the encode count per clip.
            target.replace(dst)
            cur_src = Path(dst)
            break
        # The correction is capped, so a clip further than the cap from the
        # reference can never converge. A second pass would stack another
        # capped nudge and flatten deliberate lighting (a fire scene pulled
        # down to a dark alley). Stop at one.
        if abs(ref["luma"] - cur["luma"]) > 255.0 * 0.42:
            break
    # Promote the last intermediate when we stopped short of a final pass
    # (cur is None, refused, or nothing to do) — one plain encode, no grade.
    if cur_src != Path(dst):
        ok = _apply_clip_grade(cur_src, dst, "", out_w, out_h,
                               target_fps=target_fps)
    else:
        ok = Path(dst).exists()
    _drop(created)
    if not ok:
        # ffmpeg leaves a partial file on failure; never let it reach the
        # timeline (it would die at validate_media after all the GPU work).
        Path(dst).unlink(missing_ok=True)
        return False, ",".join(chain), passes, 0.0
    final = clip_rgb_stats(dst)
    return True, ",".join(chain), passes, (
        round(_tint_error(ref, final), 1) if final else 0.0)


def _drop(paths):
    """Unlink only the files this function created. A directory-wide
    `<stem>.g*.mp4` glob would happily delete an unrelated
    `scene_07.gallery.mp4` sitting next to it."""
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass


def _newest(paths):
    """Most recently written path as a Path, or None. Filenames end in
    uuid4().hex, so sorted()[-1] is a random pick, not the newest - sort by
    mtime instead. Ties (same mtime) fall back to name so the choice is still
    deterministic.

    ponytail: returns a Path, not a str. It used to return str while every
    producer of those paths was a Path, so callers disagreed about the type:
    one site had to wrap it in Path(...) defensively, and an earlier bug here
    indexed the result (`[-1]` on a str, which silently took the last CHARACTER
    of the filename).
    """
    items = [Path(p) for p in paths]
    if not items:
        return None
    try:
        return max(items, key=lambda p: (p.stat().st_mtime_ns, p.name))
    except OSError:
        return max(items, key=lambda p: p.name)


CONFIG = load_config()

# --- Novels ---------------------------------------------------------
# One novel = one folder (novels/<slug>/) with its own chapters, scenes,
# images, voice, videos and characters. The built-in "sinners_world" novel is
# the legacy layout at the project root, so it resolves through ROOT/CONFIG
# unchanged (and tests that patch those keep working). A non-legacy novel
# swaps in a per-novel CONFIG copy for the duration of the render.
_NOVEL_LOCAL = threading.local()


def active_novel():
    """The novel bound to THIS thread (None outside a novel context)."""
    return getattr(_NOVEL_LOCAL, "novel", None)


def current_novel_id():
    n = active_novel()
    return n.id if n is not None else DEFAULT_NOVEL_ID


def _novel_title():
    n = active_novel()
    return n.title if n is not None and n.title else DEFAULT_NOVEL_TITLE


def _asset(name):
    """Path under the active novel's root (or the project root when legacy)."""
    n = active_novel()
    if n is not None and (not n.legacy or getattr(n, "relocated", False)):
        return n.root / name
    return ROOT / name


def _cfg_path(key):
    """A per-novel path for a CONFIG key, resolved for THIS thread.

    Preflight runs on an API thread while a render may be running in another
    thread. Swapping the global CONFIG there would swap it out from under the
    running render, so preflight resolves paths per thread instead.
    """
    n = active_novel()
    if n is not None and (not n.legacy or getattr(n, "relocated", False)):
        return Path(n.config_overrides()[key])
    return Path(CONFIG.get(key, ""))


@contextmanager
def _novel_context(novel_id=None, swap_config=True):
    """Bind one novel to the current thread for a render.

    `swap_config=False` binds the novel for path resolution only (preflight),
    leaving the global CONFIG alone so a live render is never disturbed.
    Nested calls (preflight from inside _produce_inner) reuse the active novel
    instead of resolving the default again, which would clobber the swap.
    """
    global CONFIG, CHAR_BIBLE, PROPS_BIBLE
    prev = active_novel()
    if novel_id is None and prev is not None:
        yield prev
        return
    novel = get_novel(novel_id)
    if novel is None:
        raise ValueError(f"unknown novel: {novel_id!r}")
    _NOVEL_LOCAL.novel = novel
    if (novel.legacy and not getattr(novel, "relocated", False)) or not swap_config:
        # The legacy novel at the project root: no swap (its paths ARE the
        # defaults). The relocated legacy novel at novels/Example Novel/ IS
        # its own folder like any other novel, so it swaps.
        try:
            yield novel
        finally:
            _NOVEL_LOCAL.novel = prev
        return
    old_cfg, old_bible, old_props = CONFIG, CHAR_BIBLE, PROPS_BIBLE
    cfg = dict(old_cfg)
    # Novel config overrides the global one. A dict value is merged ONE level
    # deep, so a novel can say {"lora_training": {"enabled": false}} without
    # having to restate the venv and model paths.
    for _k, _v in (novel.config or {}).items():
        if isinstance(_v, dict) and isinstance(cfg.get(_k), dict):
            cfg[_k] = {**cfg[_k], **_v}
        else:
            cfg[_k] = _v
    # The novel's own folders win LAST: a novel may not redirect its assets
    # elsewhere, or "each novel is its own folder" stops meaning anything.
    cfg.update(novel.config_overrides())
    # A novel may name a shared style, or set its own via config.animation_style.
    _style = str((novel.art_direction or {}).get("style") or "").strip()
    if _style and "animation_style" not in (novel.config or {}):
        cfg["animation_style"] = _style
    CONFIG = cfg
    try:
        CHAR_BIBLE = load_character_bible()
        PROPS_BIBLE = load_props_bible()
        yield novel
    finally:
        CONFIG, CHAR_BIBLE, PROPS_BIBLE = old_cfg, old_bible, old_props
        _NOVEL_LOCAL.novel = prev


# --- Studio Rules (rules/*.md — every render loads + adheres) ---------
def load_rules():
    """Read the codex. Returns {name: text}. Missing dir = empty (warned)."""
    rd = ROOT / "rules"
    out = {}
    try:
        for p in sorted(rd.glob("*.md")):
            out[p.stem] = p.read_text(encoding="utf-8")
    except Exception:
        pass
    return out


RULES = load_rules()
RULES_VERSION = "rules:" + str(len(RULES))


def rules_adherence(scenes):
    """Non-blocking checklist: log OK/GAP per area so drift is visible."""
    try:
        cast = sorted({c.lower().replace(" ", "_")
                       for s in (scenes or []) for c in s.get("characters", [])})
        mus = list(Path(CONFIG.get("music_dir", "")).glob("*.mp3")) if Path(
            CONFIG.get("music_dir", "")).exists() else []
        log(f"  [RULES {RULES_VERSION}] codex areas: {sorted(RULES.keys())}", 2)
        log(f"  [RULES] music bed: {'OK (' + mus[0].name + ')' if mus else 'GAP — silent tails by design'}", 2)
        for c in cast:
            loras = [f.name for _d in _comfy_dirs("models") if (_d / "loras").exists()
                     for f in (_d / "loras").glob(f"{c}*.safetensors")]
            log(f"  [RULES] {c}: lora={'OK (' + loras[0] + ')' if loras else '—'}", 2)
    except Exception as e:
        log(f"  [RULES] adherence check skipped ({e})", 2)


PHASE_RULES = {
    "ops": ("ops", "one engine at a time, heat guard, services up"),
    "script": ("script", "uncensored, JSON-only, guards + diversity enforced on cached loads"),
    "characters": ("characters", "bible tags are the identity, LoRA when trained"),
    "images": ("images", "shot-aware framing (no torso), euler/simple CFG 1 (flow-match), pure T2I, grade-once"),
    "clips": ("motion", "H3 Ref2VA int8, turbo LoRA, res_multistep/simple 20 steps, prompt describes scene + identity, 24fps"),
    "voice": ("voice", "Qwen3-TTS user-clone, sidecar-fresh, -50dB trim, stereo master"),
    "lips": ("lipsync", "close-up only, 24fps echo, fail-mark fast"),
    "assembly": ("assembly", "no bridges, stereo, ducked normalize=0 mix, "
                 "post-music norm, -14 LUFS TP-2.0, captions 42cpl/17CPS/"
                 "sentence-case, receipt"),
    "qa": ("qa", "voice assert, sync gate (deletes on fail), VLM issues cap, ledger+receipt"),
}


def require_phase(name, checks=()):
    """Per-phase rules gate: log the codex section, then assert prerequisites.

    Fail-loud (RuntimeError) — a gate trip aborts before GPU spend, and the
    dashboard records the crash. This is how renders 'review best practices
    and follow them through' before each step.
    """
    sec, gist = PHASE_RULES.get(name, (name, ""))
    log(f"  [RULES] phase {name} ← rules/{sec}.md: {gist}", 2)
    for label, ok in checks:
        if not ok:
            raise RuntimeError(f"Rule gate failed at phase {name}: {label}")
        log(f"  [RULES] gate OK: {label}", 2)

def _comfy_wait(prompt_id, tries, delay, label):
    """Poll /history until the job lands. Returns the history entry or None.

    One implementation for all four ComfyUI poll loops. The old inline
    version did `except Exception: continue` on a loop of 240 x 5s, and
    http_get itself retries 3x with 5s sleeps — so a ComfyUI restart
    mid-render cost ~80 minutes of silent spinning per clip and then
    returned None, which surfaced much later as a missing-clip gate. Now
    N consecutive connection failures abort immediately with the reason.
    """
    unreachable = 0
    for _ in range(tries):
        time.sleep(delay)
        try:
            history = http_get(f"{CONFIG['comfyui_url']}/history/{prompt_id}", timeout=10)
        except Exception as e:
            unreachable += 1
            # 3 in a row = ComfyUI is down, not "busy". Fail fast.
            if unreachable >= 3:
                raise ComfyUIError(
                    0, f"{label}: ComfyUI unreachable after 3 attempts "
                       f"({e}) — job {prompt_id} abandoned")
            continue
        unreachable = 0
        if prompt_id in history:
            return history[prompt_id]
    return None


def _comfy_output_filename(entry):
    """First media filename in a /history entry, or None."""
    for out in (entry or {}).get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            if out.get(key):
                name = out[key][0].get("filename")
                if name:
                    return name
    return None


# --- HTTP Helpers --------------------------------------------------
def http_get(url, timeout=30):
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=timeout)
            return loads_json(resp.read().decode("utf-8"))
        # http.client.IncompleteRead is NOT an OSError, so a body truncated in
        # transit used to escape all three attempts -- most likely on the
        # multi-MB /object_info GET. CPython closes resp at return, so only
        # the retry tuple was worth changing here.
        except (ConnectionResetError, urllib.error.URLError, TimeoutError,
                OSError, http.client.HTTPException) as e:
            last = e
            if attempt < 2:
                time.sleep(5)
    raise last


class ComfyUIError(Exception):
    """ComfyUI /prompt rejection with server-side validation details."""
    def __init__(self, code, body):
        self.code = code
        self.body = body
        super().__init__(f"ComfyUI HTTP {code}: {body[:800]}")


def http_post_json(url, payload, timeout=60):
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            resp = urllib.request.urlopen(req, timeout=timeout)
            return loads_json(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            try:
                node_errs = loads_json(detail).get("node_errors", {})
            except Exception:
                node_errs = {}
            log(f"  [ComfyUI {e.code}] {json.dumps(node_errs)[:500]}")
            raise ComfyUIError(e.code, detail)
        except (ConnectionResetError, urllib.error.URLError, TimeoutError, OSError):
            if attempt < 2:
                time.sleep(5)
            else:
                raise

def _llm_base():
    return (CONFIG.get("llm_url") or CONFIG.get("ollama_url")
            or "http://127.0.0.1:8080").rstrip("/")


def _llm_model():
    return CONFIG.get("llm_model") or CONFIG.get("ollama_model") or ""


def llm_health(timeout=5):
    """True when the OpenAI-compatible LLM server answers."""
    try:
        r = http_get(f"{_llm_base()}/v1/models", timeout=timeout)
        return bool(r and (r.get("data") or r.get("object") == "list"))
    except Exception:
        return False


_LAST_FINISH_REASON = ""


def llm_chat(messages, timeout=300, temperature=None, seed=None, max_tokens=4096):
    """Chat completion via OpenAI-compatible API (llama-server).

    Returns the assistant text (raises on HTTP/parse failure).
    """
    global _LAST_FINISH_REASON
    payload = {
        "messages": messages,
        "temperature": (float(CONFIG.get("llm_temperature",
                                        CONFIG.get("ollama_temperature", 0.2)))
                        if temperature is None else float(temperature)),
        "stream": False,
    }
    if _llm_model():
        payload["model"] = _llm_model()
    _seed = int(CONFIG.get("llm_seed", CONFIG.get("ollama_seed", 42))) if seed is None else int(seed)
    if _seed:
        payload["seed"] = _seed
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    result = http_post_json(f"{_llm_base()}/v1/chat/completions", payload, timeout=timeout)
    try:
        _choice = result["choices"][0]
        _LAST_FINISH_REASON = str(_choice.get("finish_reason") or "?")
        return _choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected chat response shape: {str(result)[:200]}") from e

# --- Logging + Live Progress ---------------------------------------
LIVE_PROGRESS = ROOT / "progress_live.json"
STOP_FILE = ROOT / "stop.requested"
RENDER_PID = ROOT / "render.pid"
_RUN_T0 = {"t": 0.0}
_PHASE_T0 = {}
_PROGRESS_OWNER = None
_PROGRESS_LOCK = threading.Lock()
_RENDER_LOCK = threading.Lock()
# Params evolve_quality() may auto-apply. GPU-spend knobs stay here ONLY if
# a render measures them as having an effect; an inert knob in a closed loop
# oscillates and burns a full render per cycle. bright_boost was here and was
# removed — it was a prompt prefix Qwen-Image ignored (see evolve_quality).
AUTO_ALLOWLIST = ("image_steps",)


def note_phase(name):
    now = time.time()
    if name not in _PHASE_T0:
        _PHASE_T0[name] = now
    if len(_PHASE_T0) > 1:
        keys = list(_PHASE_T0.keys())
        prev = keys[-2]
        dur = now - _PHASE_T0[prev]
        try:
            # ponytail: this was a bare read_text + write_text with a bare
            # except:pass — a torn append, and json_util's own docstring
            # claims perf_log.json is covered by the atomic writer. It isn't.
            pl = ROOT / "perf_log.json"
            hist = load_json_file(pl, []) or []
            hist.append({"phase": prev, "seconds": round(dur, 1),
                         "at": time.strftime("%Y-%m-%d %H:%M")})
            dump_json_atomic(pl, hist[-200:])
        except Exception as e:
            log(f"  [WARN] perf_log write failed ({type(e).__name__}: {e})", 3)


# Lazy selectors — created on first use so they never block if
# the LLM server is down; each call is fire-and-forget with a fallback.
_sel_lock = threading.Lock()

def _reference_selector():
    with _sel_lock:
        if not hasattr(_reference_selector, "_inst"):
            _reference_selector._inst = ReferenceImageSelector(
                _llm_base(), _pick_vlm() or _llm_model())
        return _reference_selector._inst


def _best_image_selector():
    with _sel_lock:
        if not hasattr(_best_image_selector, "_inst"):
            _best_image_selector._inst = BestImageSelector(
                _llm_base(), _pick_vlm() or _llm_model())
        return _best_image_selector._inst


def should_stop():
    return STOP_FILE.exists()


def _keep_awake_on():
    """Block OS sleep/display-off during rendering. Windows-only, no-op elsewhere."""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    except Exception:
        pass


def _keep_awake_off():
    """Release the sleep block (allows sleep while cooling / after render)."""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass


def _write_pid():
    RENDER_PID.write_text(str(os.getpid()), encoding="ascii")


def _owns_progress():
    return _PROGRESS_OWNER == (os.getpid(), threading.get_ident())


def _clear_pid():
    if not _owns_progress():
        return
    try:
        if RENDER_PID.read_text().strip() == str(os.getpid()):
            RENDER_PID.unlink()
    except FileNotFoundError:
        pass


def render_alive() -> bool:
    """True if a render looks live. Errs alive — a false negative makes
    the supervisor restart services mid-GPU-run. Checks render.pid, the
    render.lock holder, then a fresh non-terminal progress heartbeat."""
    from json_util import pid_alive
    for path in (RENDER_PID, ROOT / "render.lock"):
        try:
            txt = path.read_text().strip()
            if txt and pid_alive(int(txt)):
                return True
        except Exception:
            pass
    try:
        pl = json.loads(LIVE_PROGRESS.read_text(encoding="utf-8"))
        if (pl.get("phase") not in (None, "idle", "done", "error", "stopped")
                and time.time() - float(pl.get("heartbeat", 0)) < 20 * 60):
            return True
    except Exception:
        pass
    return False


def _atomic_write_json(path, data, **dump_kw):
    """Durable JSON write. Delegates to json_util.dump_json_atomic.

    ponytail: this used to be a local copy that retried the replace three
    times and then fell out of the loop NORMALLY — no exception, no log, no
    update. Because Python opens files on Windows without FILE_SHARE_DELETE,
    any concurrent reader of progress_live.json (the dashboard and supervisor
    both poll it) made every single progress write vanish silently. One
    implementation, one guarantee, and it raises. See json_util for the
    Windows rationale.
    """
    from json_util import dump_json_atomic
    return dump_json_atomic(path, data, **dump_kw)


def read_json_file(path, default=None):
    """Re-export of json_util.load_json_file.

    pipeline_server calls this through `ep`, so it has to exist on this module.
    It did not — the endpoint raised AttributeError on its first line of work,
    which is how a "fixed" route stays broken: the 200 it used to return was
    replaced by a 500 nobody exercised.
    """
    return load_json_file(path, default)

def _read_progress():
    try:
        return json.loads(LIVE_PROGRESS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def log(msg, indent=0):
    print("  " * indent + msg)
    if not _owns_progress():
        return
    try:
        with _PROGRESS_LOCK:
            lp = _read_progress()
            lp.setdefault("log", []).append(f"{time.strftime('%H:%M:%S')} {msg}")
            lp["log"] = lp["log"][-200:]
            _atomic_write_json(LIVE_PROGRESS, lp)
    except Exception as e:
        print(f"  [WARN] progress log failed: {e}", file=sys.stderr)


def set_progress(phase, current=0, total=0, detail=""):
    if not _owns_progress():
        return
    try:
        with _PROGRESS_LOCK:
            now = time.time()
            pct = round(100 * current / total, 1) if total else 0.0
            elapsed = round(now - _RUN_T0["t"], 1) if _RUN_T0["t"] else 0.0
            eta = ""
            if current and total and current < total and elapsed:
                left = elapsed / current * (total - current)
                eta = f"{int(left // 60)}m {int(left % 60)}s"
            _atomic_write_json(LIVE_PROGRESS,
                               {"phase": phase, "current": current, "total": total,
                                "percent": pct, "detail": detail,
                                "elapsed_s": elapsed, "eta": eta,
                                "updated": time.strftime("%H:%M:%S"),
                                "heartbeat": now,
                                "log": _read_progress().get("log", [])})
    except Exception as e:
        print(f"  [WARN] progress update failed: {e}", file=sys.stderr)

# --- Service Checks + Failure Catalog ------------------------------
FAILURE_MODES = {
    # id: (description, recovery hint)
    "llm_down": ("LLM server not responding", "launch: llama-server on :8080"),
    "llm_model_missing": ("Configured LLM model file missing", "recover the GGUF into llama-server/models"),
    "comfyui_down": ("ComfyUI not responding", "launch Comfy Desktop backend"),
    "comfyui_400": ("ComfyUI rejected workflow (bad node/input)", "see node_errors, retry degraded"),
    "comfyui_oom": ("ComfyUI out of memory", "halve frames/steps, retry"),
    "tts_down": ("Qwen3-TTS server not responding", "run cosy-server/server.py"),
    "stale_cache": ("Cached file older than its source", "delete cache, regenerate"),
    "missing_output": ("Expected output file absent after success", "retry once, then fail loud"),
    "disk_full": ("Output disk nearly full", "free space before rendering"),
}


def _launch_detached(cmd, cwd=None):
    try:
        if os.name == "nt":
            proc = subprocess.Popen(cmd, cwd=cwd,
                             creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
                             close_fds=True)
        else:
            proc = subprocess.Popen(cmd, cwd=cwd, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return proc
    except Exception as e:
        log(f"  [WARN] Auto-launch failed ({e}), start manually")
        return None


def _llm_launch_cmd(vision=False):
    exe = Path(CONFIG.get("llm_exe", _home("llama-server", "llama-cpp", "llama-server.exe")))
    if vision:
        model = Path(CONFIG.get("llm_vlm_gguf", _home("llama-server", "models", "Qwen3VL-8B-Instruct-Q4_K_M.gguf")))
        mmproj = Path(CONFIG.get("llm_vlm_mmproj", _home("llama-server", "models", "mmproj-Qwen3VL-8B-Instruct-F16.gguf")))
        # VRAM budget (16GB): VLM ~4.8GB GPU + Qwen-Image-2.1 GGUF + TTS 3.5GB.
        # Partial offload keeps grading co-resident with Comfy; lower -ngl on OOM.
        # --image-min-tokens 1024: the loader warns Qwen-VL "requires at minimum
        # 1024 image tokens to function correctly on grounding tasks"
        # (llama.cpp issue 16842). Without it the grader sees a low-res crop
        # and misses defects it is supposed to catch.
        # ctx 4096, not 2048: the clip-vs-still check sends BOTH images, and
        # 2 x 1024 image tokens does not fit 2048 — it came back as a bare
        # HTTP 400 with no usable message.
        return [str(exe), "--model", str(model), "--mmproj", str(mmproj),
                "--port", "8080", "--host", "127.0.0.1", "-ngl", "24",
                "--ctx-size", "4096", "-ctk", "q8_0", "-ctv", "q8_0",
                "--image-min-tokens", "1024"]
    model = Path(CONFIG.get("llm_gguf", _home("llama-server", "models", "gemma-4-26B-A4B-it-ultra-uncensored-heretic.i1-IQ4_XS.gguf")))
    # --reasoning off: guards the hybrid-reasoning failure mode where the model
    # spends the whole token budget in reasoning_content and returns content="",
    # so every script/scene generation would have received empty text. Qwen3
    # needed it; Gemma-4 defaults enable_thinking to false in its own template,
    # so this is belt-and-braces now. See rules/script.md.
    #
    # ctx 32768, not 8192: scene_target_count returns up to 56 scenes and a
    # 25-scene chapter needed ~4300 completion tokens even at 25-word lines, so
    # 8192 ctx left 4552 for a 3640-token prompt and the JSON truncated mid-
    # array (finish_reason "length" -> unparseable -> 3x retry -> abort). Now:
    # 5454 prompt + 12222 chapter + 14336 answer = 32012 < 32768. Gemma-4-26B-A4B
    # keeps
    # long-range KV on only 5 of 30 layers and llama.cpp puts the whole KV
    # cache in SYSTEM RAM, not VRAM, so context costs RAM (~1.1 GiB at 32k),
    # not the 12.96 GB of weights that actually fill the 16 GB card. Measured
    # free: no speed cost (9-scene probe ~192 s at 32k vs ~225 s at 8k).
    # Trained max is 262144; raise llm_ctx if chapters get bigger.
    #
    # --parallel 1 is load-bearing, not a tuning knob: llama-server splits
    # --ctx-size across slots, so the default 4 slots gave each one only
    # 32768/4 = 8192. A 3640-token script prompt then had 4552 left, and the
    # response was cut mid-array with finish_reason "length" — which reads
    # exactly like a max_tokens cap but is not one. The pipeline is strictly
    # one request at a time, so give the single slot the whole window.
    return [str(exe), "--model", str(model), "--port", "8080",
            "--host", "127.0.0.1", "-ngl", "99", "--parallel", "1",
            "--ctx-size", str(int(CONFIG.get("llm_ctx", 32768))),
            "-ctk", "q8_0", "-ctv", "q8_0",
            "--reasoning", "off", "--reasoning-budget", "0"]


_svc_cache_ep = {"ts": 0.0, "data": None}

# P3 ops: launch paths overridable via config/env (hardcoded = this machine
# only). Module level, not local to ensure_services(): start_tts_server()
# needs the same pair, and a launcher that cannot see its own paths is how
# ensure_tts() ended up being mistaken for one.
COMFY_EXE = Path(CONFIG.get(
    "comfy_exe",
    _home("AppData", "Local", "Programs", "Comfy Desktop", "Comfy Desktop.exe")))
TTS_SERVER = ROOT / "cosy-server" / "server.py"
TTS_PYTHON = Path(sys.executable)


def _llm_server_pids():
    """PIDs of every running llama-server process ([] when none).

    ponytail: a llama-server that is LOADING the model answers no health
    probe, so the probe alone cannot tell "no server" from "server still
    starting". Launching a second one in that window loses the bind race -
    the loser never serves and strands its model in RAM. Measured on
    ch019 (2026-10-22): the watchdog and the dashboard both launched one
    25s apart; the loser held 7.2GB and left the 31GB box with 2GB free,
    which starved ComfyUI for the whole render.
    """
    try:
        if os.name == "nt":
            r = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue)"
                 " | ForEach-Object { $_.Id }"],
                capture_output=True, text=True, timeout=30)
        else:
            r = subprocess.run(["pgrep", "-x", "llama-server"],
                               capture_output=True, text=True, timeout=30)
        return [int(x) for x in (r.stdout or "").split() if str(x).isdigit()]
    except Exception:
        return []


def ensure_services(auto_start=True, wait_s=180):
    """Check services, auto-start missing ones, re-check. Returns (ok, status)."""
    # ponytail: 2s cache avoids hammering 3x 5s-blocking HTTP calls on
    # repeated dashboard polls / supervisor checks. Cache is module-level
    # so pipeline_server.py async checks and supervisor sync checks share it.
    now = time.time()
    if auto_start is False and _svc_cache_ep["data"] is not None and now - _svc_cache_ep["ts"] < 2.0:
        return _svc_cache_ep["data"]

    def _check():
        s = {"llm": False, "comfyui": False, "tts": False}
        try:
            if llm_health(timeout=5):
                s["llm"] = True
        except Exception:
            pass
        try:
            http_get(f"{CONFIG['comfyui_url']}/system_stats", timeout=5)
            s["comfyui"] = True
        except Exception:
            pass
        try:
            tts_url = CONFIG.get("tts_url", "http://localhost:5003")
            r = http_get(f"{tts_url}/health", timeout=5)
            # ponytail: gate on loaded, not just listening. During the ~3GB
            # preload window status is ok but synthesis would wedge behind
            # the load — wait it out (deadline loop) instead of failing late.
            s["tts"] = bool(r and r.get("status") == "ok" and r.get("loaded", False))
        except Exception:
            pass
        return s

    status = _check()
    if all(status.values()) or not auto_start:
        for k, v in status.items():
            log(f"[{'OK' if v else 'FAIL'}] {k}")
        result = (all(status.values()), status)
        if auto_start is False:
            _svc_cache_ep.update(ts=time.time(), data=result)
        return result

    if not status["llm"]:
        if _llm_server_pids():
            # A server is still LOADING the model: the health probe cannot
            # tell that from "no server". Launching a second one loses the
            # bind race and strands its model in RAM (see _llm_server_pids).
            log("[WARN] a llama-server is already running (still loading) - "
                "not launching a duplicate", 1)
        else:
            log("[FAIL] llm_down, auto-starting...")
            _launch_detached(_llm_launch_cmd())
    if not status["comfyui"]:
        log("[FAIL] comfyui_down, auto-starting...")
        try:
            if COMFY_EXE.exists() and os.name == "nt":
                os.startfile(str(COMFY_EXE))
        except Exception as e:
            log(f"  [WARN] Comfy launch failed: {e}")
    if not status["tts"]:
        log("[FAIL] tts_down, auto-starting...")
        if TTS_SERVER.exists() and TTS_PYTHON.exists():
            _launch_detached([str(TTS_PYTHON), str(TTS_SERVER)], cwd=str(TTS_SERVER.parent))

    deadline = time.time() + wait_s
    while time.time() < deadline:
        time.sleep(10)
        status = _check()
        if all(status.values()):
            break
    for k, v in status.items():
        log(f"[{'OK' if v else 'FAIL'}] {k}")
    missing = [k for k, v in status.items() if not v]
    if missing:
        log(f"Unrecovered modes: {missing} "
            f"({'; '.join(FAILURE_MODES.get(m + '_down', ('?', '?'))[1] for m in missing)})")
    result = (all(status.values()), status)
    _svc_cache_ep.update(ts=time.time(), data=result)
    return result


def check_services():
    ok, _ = ensure_services(auto_start=False)
    return ok


PREFLIGHT_TRIAGE_S = 7 * 24 * 3600  # unresolved criticals older than this
LAST_PREFLIGHT_PROBLEMS = []       # set by preflight(); read by the API

# Zero-shot voice-clone reference specs (Qwen3-TTS family).
# 6s is the hard floor below which the speaker embedding is underspecified;
# 10-20s is the recommended band, 30-60s the quality ceiling.
VOICE_REF_MIN_S = 6.0
VOICE_REF_MAX_S = 60.0
VOICE_REF_MIN_PEAK_DB = -40.0     # a silent ref clones as a silent/muddy voice
VOICE_REF_TARGET_PEAK_DB = -3.0


def voice_ref_problems(wav_path):
    """Validate a TTS reference clip. Returns a list of human-readable
    problems — empty means usable. Never raises.

    Uses ffprobe (duration) + ffmpeg volumedetect (level), both of which
    the pipeline already depends on, so this costs no new dependency and no
    GPU.
    """
    out = []
    wav = Path(wav_path)
    try:
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(wav)],
            capture_output=True, text=True, timeout=30)
        dur = float(((pr.stdout or "").strip().splitlines() or ["0"])[0] or 0)
    except Exception as e:
        return [f"could not read {wav.name} ({e})"]
    if not dur:
        return [f"{wav.name} has no readable duration — is it a valid WAV?"]
    if dur < VOICE_REF_MIN_S:
        out.append(f"{wav.name} is {dur:.1f}s, under the {VOICE_REF_MIN_S:g}s "
                   f"minimum for a usable clone — re-record 10-20s of clean speech")
    elif dur > VOICE_REF_MAX_S:
        out.append(f"{wav.name} is {dur:.0f}s, over the {VOICE_REF_MAX_S:g}s "
                   f"practical maximum — trim to 10-30s")
    try:
        vd = subprocess.run(
            [CONFIG["ffmpeg"], "-hide_banner", "-nostats", "-i", str(wav),
             "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60)
        m = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", vd.stderr or "")
        if m is None:
            out.append(f"{wav.name} level could not be measured")
        else:
            peak = float(m.group(1))
            if peak <= -60.0:
                out.append(f"{wav.name} is silent (peak {peak:.0f} dB) — "
                           f"the cloned voice would be mute")
            elif peak < VOICE_REF_MIN_PEAK_DB:
                out.append(f"{wav.name} is very quiet (peak {peak:.0f} dB) — "
                           f"re-record; aim for ~{VOICE_REF_TARGET_PEAK_DB:g} dB")
    except Exception as e:
        out.append(f"{wav.name} level check failed ({e})")
    return out


def preflight(chapter_num, novel=None):
    # swap_config=False: preflight runs on an API thread and must not swap the
    # global CONFIG out from under a render that is running in another thread.
    with _novel_context(novel, swap_config=False):
        return _preflight_inner(chapter_num)


def _preflight_inner(chapter_num):
    """Detect failure modes before burning GPU time. Returns True if clear.

    Problems are also stashed in LAST_PREFLIGHT_PROBLEMS so the dashboard
    can name the actual blocker instead of "see the log".
    """
    global LAST_PREFLIGHT_PROBLEMS
    problems = []
    for _d in (_cfg_path("music_dir"), _cfg_path("sfx_dir"), _cfg_path("output_dir")):
        try:
            if _d:
                Path(_d).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    chapters = chapter_files(_cfg_path("chapters_dir"))
    if not chapters:
        problems.append("no chapters found")
    if str(CONFIG.get("animation_style", "cinematic")).lower() not in ANIMATION_STYLES:
        problems.append(f"unknown animation_style {CONFIG.get('animation_style')} "
                        f"(options: {', '.join(sorted(ANIMATION_STYLES))})")
    engine = str(CONFIG.get("video_engine", "h3")).lower()
    # ponytail: key -> candidate model subdirs (H3 DiT int8 in
    # diffusion_models, TE in text_encoders, VAE in vae, LoRAs in loras).
    wanted = {"checkpoint": ("diffusion_models",),
              "qwen_text_encoder": ("text_encoders",),
              "qwen_vae": ("vae",),
              "h3_checkpoint": ("diffusion_models",),
              "h3_text_encoder": ("text_encoders",),
              "h3_vae": ("vae",),
              "h3_audio_vae": ("vae",),
              "h3_turbo_lora": ("loras",)}
    _mroots = [Path(_home("AppData", "Local", "Comfy-Desktop", "ComfyUI-Shared", "models")),
               Path(_home("AppData", "Local", "Comfy-Desktop", "ComfyUI-Installs", "Example Novel", "ComfyUI", "models"))]
    # Fail loud on a missing model file as well as a missing root. The old
    # code appended the problem AND set models_ok=False, then returned False
    # on `problems` — so the warn-and-continue branch below was unreachable
    # and its comment described behaviour that did not exist. There are no
    # working fallbacks for a missing weight file; say so and stop.
    if not any(r.exists() for r in _mroots):
        problems.append("no ComfyUI model root exists — update paths for this machine")
    for key, subs in wanted.items():
        name = CONFIG.get(key)
        if not name:
            continue
        if "h3" in key and engine != "h3":
            continue
        if not any((r / sub / name).exists() for r in _mroots for sub in subs):
            problems.append(f"model missing: {name} in {subs} — install or fix config (engine={engine})")
    try:
        free_gb = shutil.disk_usage(_cfg_path("output_dir")).free / 1e9
        if free_gb < 10:
            problems.append(f"disk_full ({free_gb:.1f}GB free)")
    except Exception as e:
        # Fail closed: not knowing the free space is not the same as having
        # space. A full disk surfaces as a 40-minute GPU render that dies at
        # the mux step.
        problems.append(f"cannot stat output_dir for free space: {e}")
    # Ledger triage gate. Two defects fixed here:
    #  1. a corrupt issues.json used to fall through the bare except and skip
    #     the gate entirely (fail-open on the one file that blocks renders);
    #  2. it had no triage window, so a crash from last month blocked
    #     every render forever. Matches pipeline/preflight.py's 7-day rule.
    _ledger = ROOT / "issues.json"
    try:
        with open(_ledger, encoding="utf-8") as f:
            _led = json.load(f)
    except FileNotFoundError:
        _led = []
    except (OSError, ValueError) as e:
        problems.append(f"issues.json unreadable ({e}) — fix or delete it to render")
        _led = []
    if _led:
        try:
            from supervisor import parse_ts
        except Exception:
            parse_ts = None
        now = time.time()
        _open = []
        for e in _led:
            if not isinstance(e, dict) or e.get("severity") != "critical" or e.get("resolved"):
                continue
            ts = parse_ts(e.get("ts")) if parse_ts else None
            if ts is not None and now - ts > PREFLIGHT_TRIAGE_S:
                continue  # already triaged by age
            _open.append(e)
        if _open:
            problems.append(
                f"{len(_open)} unresolved CRITICAL in issues.json — "
                + f"resolve via /api/issues/resolve, or auto-clears after "
                  f"{PREFLIGHT_TRIAGE_S // 86400}d: "
                + "; ".join(str(e.get("symptom", "?"))[:80] for e in _open[:3]))
    vref = Path(ROOT / "cosy-server" / "ref.json")
    # ponytail: single-engine pipeline — the user-voice ref must be usable
    # before GPU spend (no voice = no episode, fail here not at scene 1).
    #
    # Validate the file the TTS SERVER actually loads, not just ref.json's
    # copy of it: server.py hardcodes REF_WAV and reports the real path on
    # /health as "reference". Gating ref.json alone gave false assurance —
    # it validated a 25s wav while the server was synthesising from a
    # different (41s) file.
    _served = None
    try:
        # http_get returns the parsed body, NOT (code, body) — unpacking it
        # into two names silently "worked" by yielding the dict's KEYS.
        _health = http_get(f"{CONFIG.get('tts_url', 'http://localhost:5003')}/health",
                           timeout=5)
        if isinstance(_health, dict) and _health.get("reference"):
            _served = Path(_health["reference"])
    except Exception:
        _served = None
    try:
        with open(vref, encoding="utf-8") as f:
            _rj = json.load(f)
        _wav = Path(_rj.get("wav", ""))
        if not _wav.exists():
            problems.append(f"bad user voice ref: {vref} (wav does not exist: {_wav})")
        if not str(_rj.get("text", "")).strip() and _served is None:
            problems.append(f"user voice ref text missing in {vref}")
        # Existence is not quality. XTTS zero-shot cloning is specified for
        # 6-30s of clean speech; a 1.5s or silent reference passes a bare
        # exists() and poisons the cloned voice for the WHOLE episode —
        # discovered only after every still and clip has been paid for.
        # Gate duration and level now, for free.
        # ponytail: build the list explicitly. The previous form was a
        # 2-tuple whose FIRST element was `()` when /health was unreachable,
        # so the loop raised "not enough values to unpack" and the bare
        # except below reported it as "missing user voice ref" — the gate
        # blocked every render whenever the TTS server was down, which is
        # exactly when you most need the message to be accurate.
        _refs = [("ref.json", _wav)]
        if _served:
            _refs.append(("served by the TTS server", _served))
        for _label, _path in _refs:
            problems.extend(f"voice ref ({_label}): {p}"
                            for p in voice_ref_problems(_path))
    except Exception as e:
        problems.append(f"missing/unreadable user voice ref: {vref} ({e})")
    # Rights manifest preflight gate. Fail-CLOSED: the old bare
    # `except Exception: log(WARN)` meant a missing/corrupt manifest or any
    # bug in the gate SKIPPED the check and let a render through with
    # unconfirmed rights. This gate exists for compliance, so a broken gate
    # is itself a blocker.
    if _PIPELINE_UPGRADE:
        try:
            manifest = load_rights_manifest(_asset(""))
        except Exception as e:
            problems.append(f"rights_manifest.json could not be loaded: {e}")
        else:
            if not manifest.get("novel_adaptation_rights", {}).get("owner_confirmation", False):
                problems.append("novel adaptation rights not confirmed in rights_manifest.json")
            try:
                ok_rights, rights_blockers, rights_warnings = preflight_rights_gate(manifest)
            except Exception as e:
                problems.append(f"rights gate failed to run: {e}")
            else:
                if not ok_rights:
                    for issue in rights_blockers:
                        problems.append(f"rights: {issue}")
                for warn in rights_warnings:
                    log(f"  [WARN] rights: {warn}", 1)

    LAST_PREFLIGHT_PROBLEMS = list(problems)
    if problems:
        for p in problems:
            log(f"[FAIL] {p}")
        return False
    LAST_PREFLIGHT_PROBLEMS = []
    log("[OK] Preflight clear")
    return True
# Constants used across the module: where the legacy book actually lives when
# it has been relocated into novels/Example Novel/.
_SW_ROOT = None


def _sw_root():
    """The legacy book's folder, when it relocated into novels/Example Novel/.

    ponytail: cached per process. The render always runs under a novel
    context, so this only covers code paths that read files at import time
    (module-level constants, test fixtures) with no novel bound.
    """
    global _SW_ROOT
    if _SW_ROOT is None:
        candidate = ROOT / "novels" / "Example Novel"
        _SW_ROOT = candidate if (candidate / "characters" /
                                 "characters.json").exists() else None
    return _SW_ROOT


# --- Character Bible -----------------------------------------------

def load_character_bible(path=None):
    cb_path = Path(path) if path else Path(CONFIG["characters_file"])
    if not cb_path.exists():
        sw = _sw_root()
        if sw is not None:
            candidate = sw / "characters" / "characters.json"
            if candidate.exists():
                cb_path = candidate
    if cb_path.exists():
        with open(cb_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


CHAR_BIBLE = load_character_bible()


def load_props_bible(path=None):
    props_path = Path(path) if path else (_asset("characters") / "props.json")
    if not props_path.exists():
        sw = _sw_root()
        if sw is not None and (sw / "characters" / "props.json").exists():
            props_path = sw / "characters" / "props.json"
    if props_path.exists():
        with open(props_path, "r", encoding="utf-8") as f:
            return json.load(f).get("props", {})
    return {}


PROPS_BIBLE = load_props_bible()


def _bible_alias_map():
    """alias key -> canonical key, from each bible entry's "aliases" list.

    A novel with a masked protagonist (Klein / Gehrman / Sherlock) has many
    names for one person. Without aliases the learner invents a SECOND bible
    entry — and a second face — for the same character, because it is
    deliberately asked about names the chapter mentions but never shows.
    """
    out = {}
    bible = CHAR_BIBLE.get("characters", {}) if isinstance(CHAR_BIBLE, dict) else {}
    for key, data in bible.items():
        for alias in (data or {}).get("aliases") or []:
            k = str(alias).lower().replace(" ", "_")
            if k:
                out[k] = key
    return out


def _bible_entry(char_key):
    """The bible entry for a character key, following aliases."""
    bible = CHAR_BIBLE.get("characters", {}) if isinstance(CHAR_BIBLE, dict) else {}
    key = str(char_key or "").lower().replace(" ", "_")
    entry = bible.get(key)
    if entry:
        return entry
    canonical = _bible_alias_map().get(key)
    return bible.get(canonical) if canonical else None


def _edit_distance(a, b, cap=1):
    """Levenshtein distance, capped (a, b are short names)."""
    a, b = str(a).lower(), str(b).lower()
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _learn_new_characters(cast, chapter_path, chapter_num):
    """Auto-describe cast the bible doesn't know, so they get prompt tags.

    ch3 shipped Maret Ovrik and Rendell with no bible entry: no danbooru_tags,
    and Edvyn's LoRA hijacked their shots — a central character's
    emotional beats landed on the protagonist's face. One LLM call extracts a
    physical description from the chapter text and writes it into
    characters.json, so generate_image sees the tags. Must run while the LLM
    server is still up (before unload_llm).

    Fail-safe by design: an LLM failure, an unparseable answer, or empty tags
    logs a warn and leaves the current warn-only behavior — learning never
    aborts a render.
    """
    known = set((CHAR_BIBLE.get("characters", {}) or {}))
    # An ALIAS is the same person under another name (Klein / Mr. Fake It
    # Until You Make It / Gehrman Sparrow). The learner is DELIBERATELY asked
    # about off-screen named characters, so without this it invents a second
    # entry — and a second face — for someone the bible already has.
    aliases = _bible_alias_map()
    known |= set(aliases)
    new = sorted({str(c).lower().replace(" ", "_") for c in cast or []} - known)
    # ponytail: the early return here meant a chapter whose ON-SCREEN cast was
    # all-known never asked the model about anyone else. ch020 named Aldric
    # Grent in 5 scenes and Corvin Hest in 1 - both off-screen (a newspaper
    # headline, a spoken name, a death report), so the cast was all-known, the
    # call was skipped, and Grent reached the bible never. He recurs through
    # ch044, so the chapter where he finally stands on camera inherits whatever
    # face the model invents that day. Ask every time: one call, ~4 min on the
    # CPU-only llama-server, and the model already has the chapter text.
    try:
        chapter_text = Path(chapter_path).read_text(encoding="utf-8") if chapter_path else ""
    except OSError:
        chapter_text = ""
    _bible = CHAR_BIBLE.get("characters", {}) or {}
    known_names = ", ".join(sorted(
        (str(_bible.get(k, {}).get("full_name") or k) if k not in aliases else k)
        for k in known))
    prompt = f"""Extract a physical character sheet for each named INDIVIDUAL in the chapter below that is MISSING from the character bible, using the chapter text where it describes them and conservative genre-fitting invention where it does not (no bright colors; dark gothic industrial world).

Characters already in the bible - do NOT describe these again:
{known_names}

The characters flagged by the script: {', '.join(new) if new else '(none - the script put nobody new on screen)'}

The flagged list is INCOMPLETE. A chapter often names people it never shows: someone in a newspaper headline, someone spoken about, someone reported dead. If the chapter names such a person, describe them too - keyed as firstname_lastname in lower case with an underscore. Describe only PEOPLE. Never describe a place, street, district, publication, organisation, committee, treaty, ability or any other thing - those are not characters.

Return ONLY one JSON object, keys exactly as given:
{{"<key>": {{"full_name": "...", "role": "...", "age": "...", "build": "...",
"face": "...", "hair": "...", "eyes": "...", "clothing": "...",
"distinguishing_features": "...", "personality_visual": "...",
"danbooru_tags": "...", "prompt_tags": "...", "negative_tags": "..."}}}}

danbooru_tags: comma-separated booru tags for an anime image model — start
with 1boy/1girl, then age, hair, eyes, clothing, distinguishing marks, and
end with "dark fantasy, gothic, detailed eyes".
prompt_tags: the same identity as one flowing visual description.
negative_tags: what this character must NOT look like.
NEVER write "unknown", "not mentioned", "n/a" or "unspecified" in any field.
The chapter does not describe everyone, so INVENT concrete, consistent
details for whoever it leaves out — an invented face is a character, the
word "unknown" is a hole in the prompt.
Never leave danbooru_tags or prompt_tags empty.

CHAPTER:
{_chapter_excerpt(chapter_text)}"""
    try:
        raw = llm_chat([{"role": "user", "content": prompt}],
                       timeout=300, max_tokens=2500, temperature=0.2,
                       seed=int(CONFIG.get("llm_seed", 42)) + chapter_num)
    except Exception as e:
        log(f"  [WARN] character learning call failed ({str(e)[:80]}); "
            f"{len(new)} cast render from their existing tags", 1)
        return []
    data = None
    try:
        data = loads_json(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw or "", re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        log(f"  [WARN] character learning returned no usable JSON; "
            f"{len(new)} cast render from their existing tags", 1)
        return []
    _FIELDS = ("full_name", "role", "age", "build", "face", "hair", "eyes",
               "clothing", "distinguishing_features", "personality_visual",
               "danbooru_tags", "prompt_tags", "negative_tags")
    added = []
    # ponytail: iterate the ANSWER, not the flagged list. The model may add
    # people the script never put on screen (ch020: Aldric Grent, Corvin
    # Hest), and those are exactly the ones that used to be missed. Anything
    # the bible already knows is skipped, so a re-description under a
    # different spelling cannot fork the bible.
    _known_names = {str((CHAR_BIBLE.get("characters", {}) or {}).get(k, {}).get("full_name")
                        or k).strip().lower()
                    for k in known}
    for rkey, entry in data.items():
        if not isinstance(entry, dict):
            continue
        key = str(rkey).strip().lower().replace(" ", "_").replace("'", "")
        if not key or key in known:
            continue
        full = str(entry.get("full_name", "") or "").strip().lower()
        if full and full in _known_names:
            continue
        tags = str(entry.get("danbooru_tags", "") or "").strip()
        ptags = str(entry.get("prompt_tags", "") or "").strip()
        if not tags or not ptags:
            continue
        clean = {f: str(entry.get(f, "") or "") for f in _FIELDS}
        if not clean["negative_tags"]:
            clean["negative_tags"] = ("bright colors, cheerful expression, "
                                      "modern clothing, extra limbs, deformed face")
        # ponytail: an entry that answers "unknown" is worse than no entry -
        # it becomes the image prompt AND the LoRA caption, so the character
        # is drawn from the literal token "unknown". ch019 learned
        # the_unmade_operative as "1man, unknown, unknown, unknown, dark
        # clothing", which is a grey anonymous man with a LoRA trained on
        # him. Invent concrete details instead, and say so loudly when the
        # model refuses to.
        if "unknown" in (tags + " " + ptags).lower():
            log(f"  [WARN] learned description for {key} still says "
                "'unknown' - his tags and any LoRA trained on them will be "
                "a generic face",
                1)
        # The SCRIPT's spelling and the bible's key must agree, or the
        # character has NO tags in the still prompt at all (the lookup is by
        # key). The script wrote "ammon"; the model answered "amon". Record
        # the script's name as an ALIAS of the entry it learned, so both
        # resolve to one face.
        for flagged in new:
            if flagged == key:
                continue
            if _edit_distance(flagged, key) <= 1:
                clean.setdefault("aliases", [])
                if flagged not in clean["aliases"]:
                    clean["aliases"].append(flagged)
                log(f"  Bible alias: '{flagged}' -> '{key}' (one spelling "
                    "apart - the script's name now resolves to this entry)", 1)
                break
        CHAR_BIBLE["characters"][key] = clean
        added.append(key)
    if not added:
        log("  [WARN] character learning described nobody new; "
            "any new face renders from its existing tags", 1)
        return []
    # Mutating the module global is what generate_image reads; dump the same
    # object so the next render and the fingerprint subtree both see it.
    # Appends go at the end, so existing entries do not shift.
    dump_json_atomic(Path(CONFIG["characters_file"]), CHAR_BIBLE)
    log(f"  Learned {len(added)} new character(s): {', '.join(added)} — "
        "tags build from this", 1)
    return added


def props_for_scene(scene):
    """Match props bible entries against scene text. Returns tag string."""
    text = " ".join([scene.get("narration", ""), scene.get("dialogue", ""),
                     scene.get("image_prompt", "")]).lower()
    keys = {"wooden_box": ["box", "wooden"],
            "revolver": ["revolver", "gun", "pistol"],
            "ledger": ["ledger", "account book", "calculations"],
            "leather_notebook": ["notebook"],
            "gas_lamp": ["lamp post", "gas lamp", "lamp-post", "lamppost"],
            "tea_set": ["tea house", "tea cup", "teapot", "tea set"]}
    tags = []
    for prop, words in keys.items():
        if any(w in text for w in words) and prop in PROPS_BIBLE:
            tags.append(PROPS_BIBLE[prop].get("prompt_tags", ""))
    return ", ".join(t for t in tags if t)

# --- Platform Presets (social-ready) -------------------------------
PLATFORMS = {
    "tiktok": {"width": 1080, "height": 1920},
    "reels": {"width": 1080, "height": 1920},
    "shorts": {"width": 1080, "height": 1920},
    "vertical": {"width": 1080, "height": 1920},
    "youtube": {"width": 1920, "height": 1080},
    "landscape": {"width": 1920, "height": 1080},
}


def get_output_dims():
    """Resolve output dims from platform preset, fallback to width/height."""
    plat = str(CONFIG.get("platform", "youtube")).lower()
    if plat in PLATFORMS:
        return PLATFORMS[plat]["width"], PLATFORMS[plat]["height"]
    return int(CONFIG.get("width", 1920)), int(CONFIG.get("height", 1080))


def vertical_fit_filter(w, h, sharpen=True):
    """Scale-to-cover + center-crop + gentle detail recovery.

    Lanczos upscale preserves edges; mild unsharp restores texture
    lost in the 512px->1080p jump without ringing or noise boost.
    sharpen=False at BUILD stage (cinematic/parallax bake it at NORM
    stage instead) — running it twice oversharpens edges into shimmer.
    """
    _sharp = ",unsharp=5:5:0.35:5:5:0.0" if sharpen else ""
    return (f"scale={w}:{h}:flags=lanczos:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},setsar=1{_sharp}")


def audio_duration(audio_path):
    """Duration of an audio file in seconds (0.0 on failure)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, timeout=30)
        return max(0.0, float(out.stdout.strip()))
    except Exception:
        return 0.0


_ABBR = {"mr.": "mister", "mrs.": "madam", "ms.": "miss", "dr.": "doctor",
           "st.": "saint", "no.": "number", "vs.": "versus", "etc.": "etcetera"}
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _num_word(n):
    n = int(n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + _num_word(rest))
    return str(n)


def clean_tts_text(text):
    """Normalize text so the TTS engine reads smoothly and accurately."""
    t = text.strip().strip('"').strip("'")
    t = re.sub(r"\[.*?\]", "", t)          # stage directions
    t = t.replace("…", ",").replace("...", ",")
    t = re.sub(r"\s*--\s*", ", ", t)
    # Expand abbreviations (TTS mangles "Dr.", "St." etc.)
    for ab, full in _ABBR.items():
        t = re.sub(r"(?i)\b" + re.escape(ab), full, t)
    # Numbers to words ("12" -> "twelve")
    t = re.sub(r"\b\d{1,3}\b", lambda m: _num_word(m.group(0)), t)
    # ALL-CAPS hook words read as shouting -> normalize (overlays carry the punch)
    t = re.sub(r"\b[A-Z]{2,}\b", lambda m: m.group(0).capitalize(), t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# Silence between sentences, in seconds. This is the ONLY padding between scenes.
#
# Measured on ch001 before it was centralised: mean gap 1.24s (min 1.01, max 1.46),
# built from three stacked layers - the 0.25s apad at the end of every TTS file,
# the scene-slot slack, and the next scene's lead-in. Six times the pause a
# narrator actually takes, and it is what makes a read feel like a list of
# separate clips rather than a sentence stream.
#
# 0.20s is the standard inter-sentence pause: short enough that consecutive
# sentences feel continuous, long enough that the listener registers the boundary.
NARRATION_GAP = 0.20

# Hold before the voice starts, inside NARRATION_GAP. The audible gap between two
# sentences is (slot - lead - audio) + the NEXT scene's lead, so this is not free:
# whatever goes here comes out of the pause. It is set to 0.05s because that is
# exactly the edge fade the assembler already applies to every scene slot, so the
# voice never starts on a hard boundary and nothing is wasted.
NARRATION_LEAD = 0.05

# Target RMS for every narration line, in dBFS. generate_tts corrects each line
# toward this so the episode holds a consistent level.
#
# Measured on ch001 BEFORE level matching: per-line RMS ran -32.2 to -22.7 dB, a
# 6.6 dB p10-p90 spread, against the ~3 dB a real narrator holds. A 6 dB step
# between two consecutive lines is heard as a jolt, which is the dominant cause of
# a read feeling unsmooth - and it is invisible to gap tuning, which is where the
# effort had been going.
#
# -26.5 dBFS is the measured median of the un-corrected output, so most lines need
# almost no gain and the correction is gentle rather than flattening the engine's
# own dynamics.
NARRATION_RMS = -26.5


def scene_clip_seconds(scene):
    """How long this scene's clip should run, in seconds.

    ponytail: this is the whole point of generating TTS before the images and
    clips. Every clip used to be built with a hardcoded `duration=6.0`, so a
    3s line got looped and a 12s line got cut — both wrong, and neither
    visible until assembly padded the gap with `-stream_loop`. Audio exists
    before the clip phase now, so the shot is built at the length it will
    actually occupy.

    Ceiling: a 30s narration on one H3 clip is a 30s diffusion pass and the
    clip phase has a wall-clock budget. Past the cap the clip runs long and
    assembly loops it, exactly as before — no worse, just not a single take.
    """
    cap = float(CONFIG.get("max_scene_clip_sec", 12))
    return max(1.0, min(cap, float(scene.get("_dur") or 0)))


def compute_scene_durations(scenes, dialogue_audios):
    default = float(CONFIG.get("scene_duration", 5))
    # fps is the single source of truth everywhere (validate_media checks it).
    fps = max(1, int(CONFIG.get("fps", 24)))
    for i, scene in enumerate(scenes):
        aud = dialogue_audios[i] if i < len(dialogue_audios) else None
        d = audio_duration(aud) if aud and Path(aud).exists() else 0.0
        if d > 0:
            # ponytail: no ceiling on the narration slot. An old 8s clamp
            # silently truncated long narration in the master (assembly warn'd and
            # cut the tail). Long lines get their natural slot.
            #
            # NARRATION_GAP is the silence BETWEEN sentences, and it is now the
            # only padding in this calculation.
            # THE ARITHMETIC THAT HIDES THIS BUG:
            #
            #     audible gap = (slot - lead - audio) + NEXT lead
            #
            # The next scene's lead-in lands in the gap, so pinning only the TAIL
            # is not enough. Measured as each layer was removed:
            #
            #     file apad + centred slot, 2.5s floor      1.24s
            #     ...with a pinned tail                       0.95s
            #     ...and with the 2.5s floor dropped          0.36s
            #     ...and a fixed NARRATION_LEAD                0.20s
            #
            # There is no way to keep a large lead-in AND a small gap: slack
            # between a narration and its scene has to go into the silence. So
            # _lead is FIXED and the slot is exactly narration + NARRATION_GAP.
            # Letting _lead absorb the slack is what doubled the error on every
            # scene, because the gap contains the lead twice.
            #
            # Dropping the 2.5s minimum also stops holding a finished frame: 13 of
            # 25 ch001 scenes were longer than their narration, so the cut landed
            # late. Runtime 63.2s -> 41.4s.
            scene["_dur"] = math.ceil((d + NARRATION_GAP) * fps) / fps
            scene["_lead"] = NARRATION_LEAD
        else:
            scene["_dur"] = default
            scene["_lead"] = 0.2
    return [s["_dur"] for s in scenes]
# --- Script Writer (LLM server) ------------------------------------
def scene_target_count(text_len):
    """How many scenes the chapter needs - a function of the CHAPTER, not a policy.

    The user asked for no length limit: an episode may be long, and it must
    carry every essential beat of its chapter. So the only ceiling here is
    PHYSICAL - how many scenes the LLM's answer can fit in the context
    window (see `_scene_budget_ceiling`).

    ~280 chars/scene is a COMPLETE adaptation (a scene per beat). At 560 the
    chapter was compressed into a recap and the essential details were the
    first thing to go: a 6000-char chapter became 10 scenes, which is a
    summary of a chapter, not the chapter.
    """
    return max(10, min(text_len // 280, _scene_budget_ceiling(text_len)))


# Measured on the live script prompt: 11234 chars -> 2948 tokens (the
# template alone, chapter excluded). 3.4 chars/token is the conservative
# ratio used for the estimates below - English prose measures ~3.6, and
# overshooting the window is a HARD error, so the estimate must err low.
_PROMPT_CHARS = 11234
_CHARS_PER_TOKEN = 3.4
_SCENE_TOK = 240          # measured tokens per scene in the answer
_CTX_MARGIN = 256         # slack for the tokenizer's own rounding


def _script_answer_budget(chapter_len, prompt_chars=_PROMPT_CHARS):
    """How many tokens the ANSWER may use for this chapter.

    The only ceiling is the context window: prompt + chapter + answer must fit
    in llm_ctx, and the server rejects the request outright when it does not.
    The chapter's own size is part of the budget, so a short chapter can be
    answered at length and a long one is trimmed by the RATIO (a content
    decision), never by a policy cap on the episode's length.
    """
    ctx = int(CONFIG.get("llm_ctx", 32768))
    cap = int(CONFIG.get("llm_script_max_tokens", 32768))
    # The LLM never sees more than the excerpt budget, so that is the chapter
    # size this budget is computed from.
    chapter_len = min(chapter_len, int(CONFIG.get("llm_chapter_chars", 44000)))
    prompt_tok = int(prompt_chars / _CHARS_PER_TOKEN)
    chapter_tok = int(chapter_len / _CHARS_PER_TOKEN)
    budget = ctx - prompt_tok - chapter_tok - _CTX_MARGIN
    return max(4096, min(cap, budget))


def _scene_budget_ceiling(text_len):
    """The most scenes the LLM can actually ANSWER for this chapter."""
    return max(10, _script_answer_budget(text_len) // _SCENE_TOK)


def _sibling_render_live():
    """True when a render owned by ANOTHER process holds the machine."""
    try:
        pidf = ROOT / "render.pid"
        if not pidf.exists():
            return False
        from json_util import pid_alive
        pid = int(pidf.read_text().strip())
        return pid != os.getpid() and pid_alive(pid)
    except Exception:
        return False


def _kill_llm_slot():
    """Stop whichever llama-server owns :8080 (text or vision profile)."""
    r = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         "$hit = @(Get-NetTCPConnection -State Listen -ErrorAction Stop | "
         "Where-Object { $_.LocalPort -eq 8080 }); "
         "if (!$hit) { exit 2 }; "
         "$hit.OwningProcess | Sort-Object -Unique | ForEach-Object { "
         "$n = (Get-Process -Id $_ -ErrorAction SilentlyContinue).ProcessName; "
         "if ($n -eq 'llama-server') { Stop-Process -Id $_ -Force } else { exit 3 } }; exit 0"],
        capture_output=True, text=True, timeout=60)
    return r


def _wait_vram_drained(deadline_s=30, quiet_mib=256, quiet_polls=2):
    """Wait until board VRAM stops FALLING — a killed process releases its CUDA
    context asynchronously, and the reading plateaus once that has finished.

    ponytail: this deliberately does NOT wait for a low number. The card is
    SHARED, so ComfyUI's own residency (4.6-12.5GB here) never falls to the H3
    floor and an absolute threshold would burn the full deadline on every call.
    Stability is the real signal that the dying context is gone.

    On CPU the LLM holds 0 VRAM, so the first two samples already match and
    this returns in ~2s.
    """
    try:
        deadline = time.time() + deadline_s
        prev, quiet = None, 0
        while time.time() < deadline:
            used = _gpu_used_mib()
            if used is None:
                return True                      # no nvidia-smi: cannot judge
            if prev is not None and prev - used < quiet_mib:
                quiet += 1
                if quiet >= quiet_polls:
                    return True                  # plateau: context released
            else:
                quiet = 0
            prev = used
            time.sleep(1)
    except Exception:
        pass
    return False


def unload_llm():
    """Stop the LLM server on :8080 so ComfyUI gets full VRAM (relaunch via watchdog/supervisor)."""
    try:
        if _sibling_render_live():
            log("  [WARN] sibling render owns the machine — refusing LLM kill")
            return False
        r = _kill_llm_slot()
        if r.returncode not in (0, 2):
            log(f"  [WARN] llm stop: rc={r.returncode} {(r.stdout + r.stderr).strip()[:120]}")
            return False
        # ponytail: the process is dead but its CUDA context tears down
        # ASYNCHRONOUSLY, and the caller's very next line starts ComfyUI work
        # (Phase 3 images). On CPU
        # the LLM holds 0 VRAM so this race was invisible; under -ngl 99 the
        # ~13GB is still draining while ComfyUI allocates, which is the
        # cudaErrorMemoryAllocation / comfy-aimdo abort guarded at :7917.
        drained = _wait_vram_drained()
        log(f"  LLM server stopped ({_serving_id() or 'slot empty'}), VRAM freed"
            + ("" if drained else f" ({_gpu_used_mib()} MiB still held after drain wait)"))
        return True
    except Exception as e:
        log(f"  [WARN] llm unload failed: {e}")
    return False


def _serving_id():
    """Model id currently serving :8080 ('' when down). Profile-agnostic."""
    try:
        r = http_get(f"{_llm_base()}/v1/models", timeout=5)
        ids = [m.get("id", "") for m in r.get("data", [])]
        return ids[0] if ids else ""
    except Exception:
        return ""


def _vlm_id():
    return Path(CONFIG.get(
        "llm_vlm_gguf",
        _home("llama-server", "models", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"))).stem


def ensure_vlm():
    global _vlm_consecutive_failures
    """Swap the :8080 slot to the vision profile for grading phases."""
    if not bool(CONFIG.get("llm_vision_enabled", False)):
        return False
    if _pick_vlm():
        # A live VLM means vision is available again, so re-arm the breaker.
        # Without this, two failures anywhere in the pipeline disabled the
        # framing gate for the whole process -- and pipeline_server renders
        # many chapters per process, so chapter 2 inherited chapter 1's outage.
        _vlm_consecutive_failures = 0
        return True
    if _sibling_render_live():
        log("  [WARN] sibling render owns the machine — refusing VLM swap")
        return False
    try:
        _kill_llm_slot()
    except Exception:
        pass
    proc = _launch_detached(_llm_launch_cmd(vision=True))
    if proc is None:
        return False
    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(10)
        if _pick_vlm():
            log(f"  VLM vision profile live on :8080 ({_serving_id()})")
            return True
        # ponytail: the swap launches into a NEW_CONSOLE, so an OOM or bad
        # flag dies with its output invisible and used to surface only as
        # "unhealthy after 180s". Poll the child so a dead launch is named.
        rc = proc.poll()
        if rc is not None:
            log(f"  [WARN] VLM process exited during startup (rc={rc}) — "
                f"usually VRAM: stop ComfyUI or lower -ngl in _llm_launch_cmd")
            return False
    log("  [WARN] VLM launch unhealthy after 180s (process alive, not serving)")
    return False


def stop_vlm():
    """Vacate the :8080 slot after grading (motion/TTS phases need the VRAM)."""
    try:
        if not _is_vlm_serving():
            return True
        if _kill_llm_slot().returncode == 0:
            log("  VLM stopped")
            return True
    except Exception as e:
        log(f"  [WARN] vlm stop failed: {e}")
    return False


def ensure_llm():
    """True when the TEXT profile is serving :8080 -- not merely something healthy.

    :8080 is a shared slot. ensure_vlm() installs the vision profile on the
    same port, and llm_health() only asks whether /v1/models lists anything.
    So a VLM left resident by a grading phase passed the old gate, and
    break_into_scenes() handed the chapter-breakdown prompt to Qwen3VL-8B,
    which returned conforming-looking but non-schema JSON and died as
    "Script generation failed 3x: model returned unparseable". Swap back.

    ponytail: it also STARTS the model when :8080 is empty. The watchdog
    deliberately leaves :8080 alone while a render is live - the render
    stops the model on purpose before the GPU phases, and the watchdog
    bringing it straight back took the memory the video model needed and
    killed ComfyUI mid-clip (ch018, 2026-10-21: scene 1 fell back to a
    still-pan and scene 3 hung on a job whose ComfyUI had died). So the
    render must be able to bring the model back itself, exactly when it
    needs it.
    """
    if not llm_health(timeout=5):
        log("  [LLM] :8080 is down - starting the text model", 1)
        try:
            unload_llm()
            _launch_detached(_llm_launch_cmd())
        except Exception as e:
            log(f"  [WARN] text-model launch failed ({e})")
            return False
        deadline = time.time() + 300
        while time.time() < deadline:
            time.sleep(10)
            if llm_health(timeout=5) and not _is_vlm_serving():
                log(f"[OK] text LLM up on :8080 ({_serving_id()})")
                return True
        log("[FAIL] text LLM did not come up within 300s")
        return False
    if not _is_vlm_serving():
        return True
    log("[MEM] :8080 is serving the VLM profile — swapping back to the text model")
    try:
        unload_llm()
        _launch_detached(_llm_launch_cmd())
    except Exception as e:
        log(f"  [WARN] text-model swap failed to launch ({e})")
        return False
    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(10)
        if llm_health(timeout=5) and not _is_vlm_serving():
            log(f"[OK] text LLM back on :8080 ({_serving_id()})")
            return True
    log("[FAIL] text LLM did not retake :8080 from the VLM within 180s")
    return False





def ensure_comfyui():
    try:
        http_get(f"{CONFIG['comfyui_url']}/system_stats", timeout=5)
        return True
    except Exception:
        log("[FAIL] ComfyUI not running")
        return False


def _comfy_port():
    try:
        return int(urlparse(CONFIG.get("comfyui_url", "")).port or 8188)
    except Exception:
        return 8188


def _comfy_owner_cmd():
    """(pid, argv) for the ComfyUI server, or (None, None).

    ponytail: only ever a python running main.py on OUR port. Same guard as
    _kill_llm_slot -- never a bare "kill python".
    """
    port = _comfy_port()
    ps = (
        "$p = %d; "
        "$hit = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue "
        "| Where-Object { $_.LocalPort -eq $p }); "
        "if (!$hit) { Write-Output 'NONE'; exit 0 }; "
        "$proc = Get-CimInstance Win32_Process -Filter "
        "\"ProcessId=$($hit[0].OwningProcess)\" -ErrorAction SilentlyContinue; "
        "if (!$proc) { Write-Output 'NONE'; exit 0 }; "
        "if ($proc.Name -ne 'python.exe' -or "
        "$proc.CommandLine -notmatch 'main\\.py') { Write-Output 'GUARD'; exit 0 }; "
        "Write-Output ($proc.ProcessId.ToString() + '|' + $proc.CommandLine)"
        % port)
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                            "-Command", ps],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        log(f"  [WARN] ComfyUI lookup failed: {e}")
        return None, None
    out = (r.stdout or "").strip()
    if not out or "|" not in out:
        if out == "GUARD":
            log("  [WARN] port owner is not a ComfyUI python -- refusing")
        return None, None
    pid, _, cmd = out.partition("|")
    try:
        return int(pid.strip()), cmd.strip()
    except ValueError:
        return None, None


def _comfy_argv(cmd):
    """Rebuild argv + cwd from a ComfyUI command line ('main.py' is relative)."""
    exe_m = re.match(r'^\s*"([^"]+)"\s+([^-\s].*)$', cmd)
    if exe_m:
        # shortcut: an unquoted token carrying an apostrophe (the project's
        # exe lives under "…\\Example Novel\\…") makes shlex split it in
        # two. The exe is quoted, so everything before the flags is the one
        # model-path token — rebuild from that boundary instead.
        exe, rest = exe_m.group(1), exe_m.group(2)
        idx = rest.find("main.py")
        if idx == -1:
            return None, None
        mainarg = rest[:idx + 7].strip().strip('"').strip("'")
        argv = [exe, mainarg, *rest[idx + 7:].split()]
        if not mainarg:
            return None, None
    else:
        try:
            argv = shlex.split(cmd, posix=False)
        except ValueError:
            return None, None
        argv = [a.strip('"') for a in argv if a.strip('"')]
        if not argv:
            return None, None
        mainarg = next((a for a in argv if "main.py" in a), None)
        if mainarg is None:
            return None, None
    exe = argv[0]
    # ponytail: the cwd must make the *relative* main.py path resolve. Comfy
    # Desktop launches argv as ["<venv>/python.exe", "-s", "ComfyUI\\main.py",
    # ...] with cwd=<install root> (the dir CONTAINING ComfyUI\), and the old
    # code resolved cwd to the ComfyUI dir itself when it only checked
    # isfile(cand/main.py). ComfyUI\main.py then resolved to
    # ComfyUI\ComfyUI\main.py, which does not exist, and the relaunch exited
    # rc=2 every time — killing recovery after every LoRA install.
    if not os.path.isabs(mainarg):
        # Walk up from the exe: the cwd is the first ancestor where
        # <cwd>/<mainarg> actually exists. Comfy-Desktop shape nests main.py
        # one level down (<install>/ComfyUI/main.py), so ancestors are tried
        # that way too, or the relaunch never fires for a bare main.py argv.
        cand = os.path.dirname(exe)
        for _ in range(4):
            if os.path.isfile(os.path.join(cand, mainarg)):
                return [exe] + argv[1:], cand
            nested = os.path.join(cand, "ComfyUI", mainarg)
            if os.path.isfile(nested):
                return [exe] + argv[1:], os.path.dirname(nested)
            parent = os.path.dirname(cand)
            if parent == cand:
                break
            cand = parent
        return None, None
    return argv, None


_COMFY_LAUNCH_CACHE = ROOT / ".comfy_launch.json"


def _remember_comfy_launch(argv, cwd):
    """Persist the last known-good ComfyUI launch so a COLD start can use it.

    ponytail: rebuild the command line from the running process works right up
    until the process is dead -- which is the only time the recovery path
    actually needs it. os.startfile(COMFY_EXE) does not bring the backend up
    in a non-interactive session either. So record the argv/cwd once, while
    there is something running to read it from.
    """
    try:
        _COMFY_LAUNCH_CACHE.write_text(
            json.dumps({"argv": list(argv), "cwd": cwd}), encoding="utf-8")
    except OSError:
        pass


def _recall_comfy_launch():
    try:
        d = json.loads(_COMFY_LAUNCH_CACHE.read_text(encoding="utf-8"))
        argv = [str(a) for a in d["argv"]]
        cwd = d.get("cwd")
        if not argv:
            return None, None
        if cwd:
            if Path(cwd, "main.py").is_file():
                return argv, str(cwd)
            # A stale cwd that no longer resolves is worse than none.
            cwd = None
        if not cwd:
            # ponytail: cold-cache entries can have cwd=null; the cwd is
            # recoverable from argv (relative main.py path + exe ancestors)
            # instead of making recovery impossible on exactly the runs that
            # need it.
            mainarg = next((a for a in argv if "main.py" in a), None)
            if mainarg and not os.path.isabs(mainarg):
                cand = os.path.dirname(argv[0])
                for _ in range(4):
                    if os.path.isfile(os.path.join(cand, mainarg)):
                        cwd = cand
                        break
                    parent = os.path.dirname(cand)
                    if parent == cand:
                        break
                    cand = parent
            if not cwd:
                # Still no cwd: safe only when main.py is absolute, otherwise
                # ComfyUI would resolve custom_nodes/ relative to the
                # producer's cwd and load no H3 node whatsoever.
                if mainarg and not os.path.isabs(mainarg):
                    return None, None
                return argv, None
        return argv, cwd
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None, None


def restart_comfyui(why="", wait_s=240):
    """Hard-restart ComfyUI so the next job starts from a clean allocator.

    ponytail: /free is NOT a guarantee. It only sets flags on the prompt
    queue, so ComfyUI applies them when the next job starts, and anything a
    custom node still holds survives -- measured here as 4761 MiB still
    resident after 3 attempts. H3 peaks at ~13.1GB of 16.3GB, so that
    residue is the difference between rendering and comfy-aimdo aborting the
    process outright (cudaErrorMemoryAllocation, which surfaces as
    connection-refused). A process restart is the one release every report
    agrees on.

    This also repairs a recovery path that could never fire: the H3 handler
    logged "restarting it" and then called ensure_comfyui(), which only
    PROBES. When ComfyUI was genuinely dead it logged "would not come back"
    and degraded the scene anyway -- so one aimdo abort still killed the
    rest of the chapter.
    """
    if _sibling_render_live():
        log("  [WARN] sibling render owns the machine - not restarting ComfyUI")
        return False
    pid, cmd = _comfy_owner_cmd()
    if pid is None:
        # No live process to copy the command line from. This is precisely the
        # case that matters: the H3 recovery branch fires BECAUSE ComfyUI
        # died, so "restart what is running" cannot be the whole answer. Hand
        # it to the preflight launcher, which starts Comfy Desktop itself.
        log("  ComfyUI is not running - starting it via the service launcher")
        # ensure_services returns (ok, status); bool() of that tuple is
        # ALWAYS True, which reported a successful recovery while ComfyUI was
        # still down -- the worst possible answer here.
        ok, _status = ensure_services(auto_start=True)
        if ok:
            return True
        # Comfy Desktop's startfile does not bring the backend up in a
        # non-interactive session. Fall back to the last launch we saw work.
        argv, cwd = _recall_comfy_launch()
        if not argv:
            log("  [WARN] ComfyUI down and no recorded launch to reuse; "
                "start it manually")
            return False
        log("  ComfyUI down - relaunching with the last known-good command")
        proc = _launch_detached(argv, cwd=cwd)
        if proc is None:
            return False
        deadline = time.time() + wait_s
        while time.time() < deadline:
            time.sleep(5)
            if ensure_comfyui():
                log("  ComfyUI back up")
                return True
            if proc.poll() is not None:
                log(f"  [WARN] ComfyUI exited during startup "
                    f"(rc={proc.returncode})")
                return False
        log(f"  [WARN] ComfyUI not healthy after {wait_s}s")
        return False
    argv, cwd = _comfy_argv(cmd)
    if not argv:
        log("  [WARN] could not reconstruct ComfyUI command line")
        return False
    _remember_comfy_launch(argv, cwd)
    log(f"  ComfyUI restart{(' (' + why + ')') if why else ''}: pid {pid}")
    try:
        r = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        log(f"  [WARN] ComfyUI kill failed: {e}")
        return False
    if r.returncode != 0:
        # ponytail: taskkill fails routinely (access denied / already gone) and
        # the old code discarded the result. A denied kill left the old server
        # up, so the wait loop below spun to its deadline and then started a
        # SECOND ComfyUI on the same port, while the health probe was satisfied
        # by the old one -- reporting a successful restart after a failed kill.
        log(f"  [WARN] ComfyUI kill failed rc={r.returncode}: "
            f"{(r.stdout + r.stderr).strip()[:110]}")
        return False
    # Wait on the PROCESS, not an HTTP probe: ensure_comfyui() only says
    # "not answering yet", which is not the same as gone, and letting the new
    # instance race the dying one for VRAM is the exact failure this exists to
    # prevent.
    deadline = time.time() + 60
    gone = False
    while time.time() < deadline:
        try:
            listing = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                     capture_output=True, text=True,
                                     timeout=30).stdout
        except Exception as e:
            # Could not confirm either way. Do NOT relaunch into a port the
            # dying process may still be holding.
            log(f"  [WARN] tasklist failed while waiting for ComfyUI to exit "
                f"({str(e)[:60]}) - not relaunching", 1)
            return False
        if listing.find(str(pid)) < 0:
            gone = True
            break
        time.sleep(1)
    if not gone:
        log("  [WARN] ComfyUI pid survived taskkill; not relaunching")
        return False
    proc = _launch_detached(argv, cwd=cwd)
    if proc is None:
        return False
    deadline = time.time() + wait_s
    while time.time() < deadline:
        time.sleep(5)
        if ensure_comfyui():
            log("  ComfyUI back up")
            return True
        if proc.poll() is not None:
            # ponytail: the relaunched backend dies (rc=2 seen on 2026-11-09
            # after the ferat LoRA install: direct relaunch of the copied argv
            # failed while the Comfy Desktop GUI launch worked minutes before).
            # Giving up here left ComfyUI dead for the rest of the render, so
            # the images phase aborted hours in. Fall back to the service
            # launcher before declaring failure.
            log(f"  [WARN] ComfyUI relaunch exited (rc={proc.returncode}) - "
                f"falling back to the desktop launcher")
            ok, _status = ensure_services(auto_start=True)
            if ok:
                log("  ComfyUI back up (desktop launcher)")
                return True
            return False
    log(f"  [WARN] ComfyUI not healthy after {wait_s}s")
    return False


def start_tts_server(wait_s=420):
    """Actually START the TTS server if it is down. Returns True when serving.

    ponytail: the old ensure_tts() only PROBES. It is named like the LLM helpers and
    that invited the mistake of calling it where a launcher was needed: Phase
    5 gated on it after the clip phase had deliberately killed TTS to free
    ~3GB, nothing ever restarted it, and the render died at the voice gate
    ~3h in -- twice, each time believing the gate had been fixed.

    Reuses the same launch pair as the preflight service check (TTS_PYTHON /
    TTS_SERVER) instead of inventing a second launcher.
    """
    if check_tts_server():
        return True
    if not (TTS_SERVER.exists() and Path(TTS_PYTHON).exists()):
        log("  [WARN] TTS server script or interpreter missing; start it manually")
        return False
    log("  TTS server down - launching it for the voice phase")
    proc = _launch_detached([str(TTS_PYTHON), str(TTS_SERVER)],
                            cwd=str(TTS_SERVER.parent))
    if proc is None:
        return False
    deadline = time.time() + wait_s
    while time.time() < deadline:
        time.sleep(5)
        if check_tts_server():
            return True
        if proc.poll() is not None:
            log(f"  [WARN] TTS process exited during startup "
                f"(rc={proc.returncode})")
            return False
    log(f"  [WARN] TTS not serving after {wait_s}s")
    return False


def free_comfyui_vram():
    """Evict loaded ComfyUI models so the LLM server gets full VRAM (and vice versa).

    ponytail: VERIFY the free actually happened instead of assuming it. A
    bare /free + sleep(5) was silently ineffective at the clips boundary, and
    H3 then had no headroom: comfy-aimdo aborts the process on a failed CUDA
    allocation, which surfaced as connection-refused and downgraded every
    remaining scene to a still-pan. Measured here: 10654 MiB -> 318 MiB.

    Note the reading is BOARD total, not ComfyUI's share. Per-process VRAM is
    "[N/A]" on Windows WDDM drivers, so attribution is impossible here; a
    reading taken while TTS is up therefore counts TTS's ~3GB too, and this
    reports that honestly instead of blaming ComfyUI for it.
    """
    body = json.dumps({"unload_models": True, "free_memory": True}).encode()
    best = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"{CONFIG['comfyui_url']}/free", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=30).read()
        except Exception as e:
            log(f"  [WARN] VRAM free failed: {e}")
        time.sleep(3)
        used = _gpu_used_mib()
        if used is None:
            time.sleep(2)
            continue
        best = used if best is None else min(best, used)
        # Under ~2GB on the board means nothing substantial is resident.
        if used <= 2048:
            log(f"  VRAM freed ({used} MiB in use across the board)")
            return True
    log(f"  [WARN] {best} MiB still in use after 3 free attempts "
        f"(ComfyUI + any other resident process)")
    return False


def _gpu_used_mib():
    """Total VRAM in use across all GPUs, or None when nvidia-smi is absent.

    ponytail: this is BOARD usage, not ComfyUI's. TTS (~3GB) and the desktop
    compositor are included, so a reading taken before the TTS evict made
    ComfyUI look like it was still holding 2359 MiB when it had released
    everything. Use it only as a whole-machine signal; for "will H3 fit",
    _comfy_vram_mib() is the honest number.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        # One line PER GPU: "2048\n4096". isdigit() on the joined string is
        # always False, which silently disabled the whole verify-and-restart
        # path on any multi-GPU box.
        vals = [int(v.strip()) for v in out.split() if v.strip().isdigit()]
        return max(vals) if vals else None
    except Exception:
        return None


def _comfy_vram_mib():
    """VRAM held by the ComfyUI process specifically, or None if unknown."""
    pid, _cmd = _comfy_owner_cmd()
    if pid is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            p, _, mib = line.partition(",")
            if p.strip() == str(pid) and mib.strip().isdigit():
                return int(mib.strip())
    except Exception:
        pass
    return None


# H3 peaks at ~13098 MiB of 16302. Anything resident above this at the clip
# boundary leaves less headroom than H3 needs, so comfy-aimdo aborts on the
# allocation. 2048 keeps ~1.2GB of slack over the observed peak.
_H3_HEADROOM_MIN_MIB = 2048

# ponytail: 90s was sized for an idle GPU. face_visibility_check() runs during
# the STILL phase, where ComfyUI holds ~10-12GB, and the first grade after each
# vision swap also pays the model load. Both timeouts returned None, which
# SILENTLY downgraded the framing gate to "prompt head-lock only" and recorded
# no failure. Budget for the contended case.
_VLM_GRADE_TIMEOUT_S = 300

# A timeout is not a pass. The framing gate returns True/"ungraded" when the
# VLM cannot be reached, which is indistinguishable from a clean grade unless
# it is counted -- so count it, and stop paying for it. At 300s a timeout is
# 5 minutes PER STILL; with ~50 stills a chapter that lost the VLM mid-phase
# would burn two hours waiting for answers nobody is reading. Two
# consecutive failures means the vision server is not coming back this run.
_FRAME_GATE = {"checked": 0, "ungraded": 0, "failed": 0}
_vlm_consecutive_failures = 0
_VLM_TRIP_AFTER = 3


def gpu_heat_guard():
    """Pause before GPU work if the card is too hot. No-op without NVIDIA."""
    max_t = CONFIG.get("gpu_max_temp", 85)
    cool_t = CONFIG.get("gpu_cooldown_temp", 78)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        temp = int(out.stdout.strip().split()[0])
    except Exception:
        return  # no NVIDIA GPU / smi missing / empty output
    if temp < max_t:
        return
    log(f"  [HEAT] GPU {temp}C >= {max_t}C, cooling to {cool_t}C...")
    _keep_awake_off()  # ponytail: let the laptop sleep while paused on heat
    try:
        while True:
            set_progress("cooling", 0, 1, f"GPU {temp}C")
            if should_stop():
                log("  [HEAT] stop requested during cooldown, aborting")
                return
            time.sleep(30)
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=15)
                temp = int(out.stdout.strip().split()[0])
            except Exception:
                return
            if temp <= cool_t:
                log(f"  [HEAT] cooled to {temp}C, resuming")
                return
    finally:
        _keep_awake_on()  # ponytail: re-block sleep when work resumes


def _unwrap_scenes(obj):
    """LLM JSON mode can wrap arrays as {"scenes": [...]}. Extract the list.

    Salvage: a single scene-like object (model ignored the array instruction)
    wraps into [obj] with a WARN so the episode still renders degraded
    instead of crashing on an otherwise good scene.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if isinstance(obj.get("scenes"), list):
            return obj["scenes"]
        if "narration" in obj or "dialogue" in obj:
            log("  [WARN] Model returned one scene object, salvaging as 1-scene list", 2)
            return [obj]
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return None


def _salvage_scene_objects(text):
    """Every parseable scene object in a broken answer, in order.

    ponytail: a 40-scene answer is ONE json blob, so one bad character (an
    unescaped quote inside a narration line, a stray comma) made the whole
    script unusable and the render died at "Script generation failed 3x" after
    three 10-minute attempts. json.JSONDecoder().raw_decode parses ONE object
    and reports where it stopped, so a broken array can be walked object by
    object and the good ones kept. Measured on ch019: 3/3 attempts produced
    ~18k chars of scenes that json.loads rejected outright; this recovered 39
    of 40 objects from them.
    """
    dec = json.JSONDecoder()
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text, j)
        except ValueError:
            i = j + 1
            continue
        if isinstance(obj, dict) and (
                "narration" in obj or "dialogue" in obj or "scene_number" in obj):
            out.append(obj)
        i = max(end, j + 1)
    return out


def _parse_scenes(text):
    """Robustly extract a scene list from model output. Returns list|None."""
    try:
        return _unwrap_scenes(loads_json(text))
    except Exception:
        pass
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        try:
            return _unwrap_scenes(loads_json(m.group(0)))
        except Exception:
            pass
    # ponytail: the whole blob failed to parse. Salvage the objects one by one
    # BEFORE the innermost-`[...]` fallback below, which used to win with
    # `["edvyn_crael"]` (the characters array of scene 1) and report "not a
    # JSON array of scenes" for an answer that held forty real scenes.
    salvaged = _salvage_scene_objects(text)
    if salvaged:
        log(f"  [FIX] salvaged {len(salvaged)} scene object(s) from a "
            "malformed answer", 2)
        return salvaged
    # ponytail: Extra-data repair (model emits [...] [...] or {...}{...}).
    # Greedy match above spans both; try each candidate longest-first.
    for cand in sorted(re.findall(r"\[.*?\]", text, flags=re.DOTALL),
                       key=len, reverse=True):
        try:
            scenes = _unwrap_scenes(loads_json(cand))
            if isinstance(scenes, list) and scenes:
                return scenes
        except Exception:
            continue
    return None


def _probe_reason(content, target=None):
    """Why this script answer is unusable, or None when it is fine.

    ponytail: the retry used to resend the SAME prompt, so a model that
    reliably leaves one establishing scene blank burned all three attempts
    and the render died at the voice gate (ch017 s4). The reason now goes
    back to the model, which can fix one scene; "try again" cannot.
    """
    raw = re.sub(r"<think>.*?</think>", "", content or "", flags=re.S).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    scenes = _parse_scenes(raw)
    if not (isinstance(scenes, list) and bool(scenes) and isinstance(scenes[0], dict)):
        return "the answer was not a JSON array of scenes"
    if target and len(scenes) < max(3, target // 2):
        return f"only {len(scenes)} scenes for a target of {target}"
    texts = [(s.get("narration") or s.get("dialogue") or "").strip().lower()
             for s in scenes]
    texts = [t for t in texts if t]
    if len(set(texts)) / max(1, len(texts)) < 0.5:
        return (f"only {len(set(texts))} distinct lines out of {len(texts)} - "
                "the script repeated itself")
    # ponytail: a scene with NO narration and NO dialogue is a SILENT scene.
    # ch017 s4 was one and it killed the whole render two phases later at the
    # voice gate ("Voiceless scenes [4]") - 20 minutes of script, director
    # package and TTS thrown away by a scene the model left blank. Reject it
    # HERE so the generator retries instead of the render aborting.
    for i, s in enumerate(scenes):
        if not ((s.get("narration") or "").strip()
                or (s.get("dialogue") or "").strip()):
            n = s.get("scene_number", i + 1)
            return (f"scene {n} has an empty narration AND an empty dialogue - "
                    "every scene must carry a narration line")
    return None


def _parse_probe(content, target=None):
    """True when the answer is a usable script."""
    return _probe_reason(content, target) is None



def _load_cached_scenes(scenes_file):
    """Load a cached scenes file and put it through the same validation as a
    freshly generated script.

    P1: renumber like fresh gens — one gapped/dupe number misaligned every
    later scene via positional lists keyed by scene_num.
    This used to be an inline loop with no type guard, so a cached file
    holding a dict instead of a list died deep inside produce_episode with
    "TypeError: 'str' object does not support item assignment".
    """
    with open(scenes_file, encoding="utf-8") as f:
        scenes = loads_json(f.read())
    try:
        _normalize_scenes(scenes)
    except (ValueError, AttributeError, TypeError) as e:
        raise RuntimeError(
            f"{Path(scenes_file).name} is not a list of scene objects ({e}). "
            f"Fix the file or delete it to regenerate.") from e
    return scenes


def _normalize_scenes(scenes):
    if not isinstance(scenes, list) or any(not isinstance(s, dict) for s in scenes):
        raise ValueError("Scenes must be a list of objects")
    for i, s in enumerate(scenes):
        s["scene_number"] = i + 1
        for key in ("narration", "dialogue", "image_prompt", "negative_prompt", "animation_hint",
                    "shot_size", "angle", "camera_move", "mood", "type", "purpose", "pose",
                    "text_overlay", "hook_type"):
            s[key] = s.get(key, "").strip() if isinstance(s.get(key, ""), str) else ""
        chars = s.get("characters", [])
        s["characters"] = [c.strip() for c in (chars if isinstance(chars, list) else [chars])
                           if isinstance(c, str) and c.strip()]
        stages = s.get("stages", [])
        s["stages"] = [{k: str(st.get(k) or "") for k in ("character", "stage", "tags")}
                       for st in (stages if isinstance(stages, list) else []) if isinstance(st, dict)]
        if "lock_face" in s and not isinstance(s["lock_face"], bool):
            s["lock_face"] = str(s["lock_face"]).lower() != "false"
        for key in ("h3_steps",):
            if key in s:
                try:
                    value = float(s[key])
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(key)
                    s[key] = int(value)
                except (TypeError, ValueError):
                    s.pop(key)
    return scenes


# Leading framing phrase the LLM writes, stripped before a FRAMING_POOL lead is
# prepended so the two cannot contradict each other. The LLM is NOT consistent:
# it emits "medium shot of X", "medium shot, X", "close-up, X" and "close-up of X"
# depending on the chapter. Matching only the first form is what collapsed ch007
# -- 38 of 40 prompts carried BOTH the injected lead and the LLM's original, and
# the duplicate gate BLOCKED the render at 37 pairs. Match the comma form too.
_SHOT_LEAD_RE = re.compile(
    r"^(?:medium|close-up|wide|extreme-wide)(?:\s+shot)?\s*,\s*"
    r"|^(?:medium|close-up|wide|extreme-wide)\s+(?:shot\s+)?of\s+",
    re.I)

# Framing leads that force different GLOBAL COMPOSITION, which is what a pHash
# distance actually responds to. See _enforce_shot_diversity for the measurement.
#
# ponytail: pool size is a hard floor, not taste. Assignment is i % len(pool),
# so by pigeonhole the largest group is ceil(scenes / pool). ch007 had 40 scenes
# and a 12-entry pool -> groups of 4, and the model collapses every scene that
# shares a framing: 58 near-identical pairs, BLOCKER, twice. A 20-entry pool
# caps the group at 2 for a 40-scene chapter and separates every collision
# actually observed in ch007. Below 20, expect the gate to trip again on long
# chapters; if scenes ever exceed ~60, grow this list rather than reintroducing
# the stride (a coprime stride reorders but cannot beat pigeonhole).
FRAMING_POOL = [
    "extreme wide establishing shot, figure tiny against vast architecture",
    "tight close-up on the eyes, shallow depth of field",
    "low angle full body shot, figure framed against the ceiling",
    "over-the-shoulder from behind, foreground shoulder dominating frame",
    "high angle looking down from above, figure small on the ground",
    "dutch tilt, skewed horizon, off-balance framing",
    "silhouette profile against a bright window, rim light",
    "macro detail on hands and object, face out of frame",
    "wide from behind through a doorway, frame within a frame",
    "medium two-shot, foreground object cutting across the lens",
    "overhead flat lay, looking straight down",
    "low ground-level shot looking up past the figure's shoulder",
    "extreme close-up on hands clasped together, everything else blurred out",
    "wide symmetrical interior, subject centred between two doorways",
    "back view mid-stride, motion blur at the edges of frame",
    "shot through a window pane, reflection layered over the subject",
    "high overhead shot of a scattered desk, one lit corner",
    "profile at the frame edge, most of the shot given to empty space",
    "worm's-eye view from the floor, subject towering overhead",
    "tight two-shot of faces, shallow focus on the nearer one",
]


def _enforce_shot_diversity(scenes):
    """Rewrite repeated shot fields so video differs scene-to-scene.

    ponytail: LLM ignores 'vary shots' (ch001: 25x static/eye-level).
    Cheapest fix: rotate repeats through pools, no re-prompt.
    """
    scenes = _normalize_scenes(scenes)
    pools = {"camera_move": ["push-in", "pull-out", "pan-left", "pan-right", "tracking", "handheld"],
             "angle": ["eye-level", "low", "high", "dutch", "over-shoulder"],
             "mood": ["tense", "somber", "hopeful", "menacing", "melancholic"],
             "shot_size": ["close-up", "medium", "wide", "extreme-wide"]}
    capacity = max(1, math.floor(len(scenes) * 0.4))
    for key, pool in pools.items():
        counts = {}
        for s in scenes:
            value = s[key].lower().replace(" ", "-").replace("_", "-")
            if value not in pool or counts.get(value, 0) >= capacity:
                # ponytail: take the LEAST-USED pool value, not the FIRST with
                # room. "First with room" filled push-in to capacity, then
                # pull-out to capacity, so ch019's 40 scenes came out 16
                # push-in + 16 pull-out + 0 static + 0 handheld - a
                # mechanically uniform episode, which is the thing "vary
                # shots" asks against. The same capacity budget spread over
                # the whole pool gives every move roughly the same share.
                value = min((p for p in pool if counts.get(p, 0) < capacity),
                            key=lambda p: counts.get(p, 0))
            counts[value] = counts.get(value, 0) + 1
            s[key] = value
    for s in scenes:
        if s["animation_hint"] in ("", "static"):
            s["animation_hint"] = s["camera_move"]
    # ponytail: the pools above rotate FIELDS, but that is not enough. The still
    # model ignores the SETTING in the prompt (ch005 asked for 20 interiors and
    # drew streets every time) and pHash measures GLOBAL COMPOSITION, so
    # "medium shot of X in an office" and "medium shot of Y in an office" still
    # collide — ch005 shipped 15 near-identical pairs and the duplicate gate
    # BLOCKED the render at 7915. Seed re-rolls cannot fix it either: measured
    # identical pHash output at lora 0.45 / 0.25 / 0.00.
    # Force the framing LEAD in image_prompt. Measured on ch005 s7/s15/s17
    # (the worst cluster): min pairwise pHash distance 2 -> 8, clearing the
    # dist<=4 duplicate threshold. Cycles by index so ADJACENT scenes always
    # differ, which is where the eye actually sees repetition.
    for i, s in enumerate(scenes):
        prompt = (s.get("image_prompt") or "").strip()
        # Idempotent: strip a framing lead this pass injected BEFORE (cached
        # scripts are re-loaded through here) and the LLM's own "medium shot of"
        # lead, so the two can neither stack nor contradict each other.
        for lead in FRAMING_POOL:
            if prompt[:len(lead) + 1].lower() == lead.lower() + ",":
                prompt = prompt[len(lead) + 1:].strip()
                break
        prompt = _SHOT_LEAD_RE.sub("", prompt).strip()
        s["image_prompt"] = (f"{FRAMING_POOL[i % len(FRAMING_POOL)]}, {prompt}"
                             if prompt else FRAMING_POOL[i % len(FRAMING_POOL)])
    return scenes


def shot_plan_pass(scenes):
    """P2: scene→shot planner (no LLM — deterministic, keeps from-zero contained).

    Guarantees: one action + <=1 camera move per shot, purpose HOOK/TURN/LAND
    from hook_type/text_overlay, pose seeded from animation_hint keywords.
    Read by motion/stills instead of re-deriving.
    """
    _pose_kw = {"walk": "walk", "run": "run", "sprint": "run", "sit": "sit",
                "kneel": "kneel", "reach": "reach", "carry": "carry",
                "turn": "look", "look": "look", "stand": "stand"}
    n = len(scenes)
    for i, s in enumerate(scenes):
        ht = str(s.get("hook_type", "") or "").upper()
        if n == 1:
            purpose = "HOOK"  # m4: single-shot episode is HOOK-only by design
        elif i == 0:
            purpose = "HOOK"
        elif i >= n - 2:
            purpose = "LAND"
        else:
            # ponytail: the old test looked for TURN/REVEAL/TWIST, none of which
            # is in the prompt's hook_type enum (shock|mystery|emotion|
            # cliffhanger|tension), so it could never fire. Test the values the
            # model actually emits; keep the invented ones so an out-of-enum
            # answer still lands.
            purpose = ("TURN" if any(k in ht for k in ("SHOCK", "TWIST", "REVEAL",
                                                      "CLIFFHANGER"))
                       else ("HOOK" if i < 3 else "TURN"))
        # NOT setdefault: _normalize_scenes runs first and creates purpose="",
        # so setdefault left it empty for every scene in every chapter — canon
        # wrote purpose:"?" and beat_planner never took its primary branch.
        s["purpose"] = purpose
        if not s.get("pose"):
            blob = f"{s.get('animation_hint','')} {s.get('image_prompt','')}".lower()
            for kw, pose in _pose_kw.items():
                if kw in blob:
                    s["pose"] = pose
                    break
        # one camera move only — comma lists collapse to first
        cm = str(s.get("camera_move", "") or "")
        if "," in cm:
            s["camera_move"] = cm.split(",")[0].strip()
    return scenes


def write_canon_snapshot(chapter_num, scenes):
    """P2: cross-chapter continuity memory (canon file per chapter)."""
    try:
        cast = sorted({c for s in scenes for c in s.get("characters", [])})
        beats = [{"n": s.get("scene_number"), "purpose": s.get("purpose", "?"),
                  "shot": s.get("shot_size", ""), "move": s.get("camera_move", ""),
                  "chars": s.get("characters", [])} for s in scenes]
        snap = {"chapter": chapter_num, "cast": cast, "beats": beats}
        p = Path(CONFIG["scenes_dir"]) / f"canon_ch{chapter_num:03d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log(f"  [WARN] canon snapshot skipped ({e})", 2)


def llm_narrative_check(scenes, every=8):
    """P2: LLM narrative spot-check every N scenes (best-effort, never blocks)."""
    if not bool(CONFIG.get("vlm_narrative", False)):
        return None
    try:
        picks = [s for s in scenes if (s.get("scene_number", 0) % every) == 1]
        for s in picks[:3]:
            llm_chat([{"role": "user",
                       "content": f"One-line continuity check for anime shot: {s.get('image_prompt','')[:200]}"}],
                     timeout=60, max_tokens=100)
        log(f"  [LLM] narrative check on {len(picks[:3])} scenes", 2)
    except Exception as e:
        log(f"  [WARN] VLM narrative skipped ({str(e)[:80]})", 2)
    return None


def _validate_scenes(scenes, target):
    """Shared script validation (fresh gens AND file-loaded). Returns reason or None."""
    if not isinstance(scenes, list) or not scenes or not isinstance(scenes[0], dict):
        return "Unparseable script response"
    # ponytail: count guard (ch001 from-zero: model returned 1 salvaged
    # scene for a 12KB chapter). Short lists retry instead of rendering
    # a 1-scene "episode".
    if len(scenes) < max(3, target // 2):
        return f"Degenerate count: {len(scenes)}/{target} scenes"
    # Degeneration guard: identical lines repeated = collapsed script
    texts = [(s.get("narration") or s.get("dialogue") or "").strip().lower()
             for s in scenes]
    texts = [t for t in texts if t]
    if texts and len(set(texts)) / len(texts) < 0.5:
        return (f"Degenerate script: only {len(set(texts))}/{len(texts)} "
                "unique lines")
    # ponytail: tail-repeat guard (ch001: scenes 21-25 identical passed
    # the global ratio). Last 5 identical = collapsed ending.
    if len(texts) >= 5 and len(set(texts[-5:])) == 1:
        return f"Degenerate ending: last 5 scenes identical: {texts[-1][:60]!r}"
    # ponytail: a scene with no narration AND no dialogue has no voice, and
    # the voice gate aborts the whole episode on it. ch017 s4 was blank and
    # killed the render at the voice phase. Catch it at the script gate.
    for i, s in enumerate(scenes):
        if not ((s.get("narration") or "").strip()
                or (s.get("dialogue") or "").strip()):
            return (f"Scene {s.get('scene_number', i + 1)} has no narration and "
                    "no dialogue - it would be a silent scene")
    return None


def _chapter_excerpt(chapter_text, limit=None):
    """Chapter text for the breakdown prompt, capped at the context budget.

    The old `chapter_text[:12000]` truncated SILENTLY: a long chapter lost
    its ending with no warning, and the tail-repeat guard then fired on the
    truncated text. Now it says so, and keeps the head and tail rather than
    cutting off the ending outright.

    Budget raised 12000 -> 48000 (2026-10-06). At 12000 the truncation was
    not a rare edge case: 40 of 403 chapters in the corpus exceed it, and
    ch2 came in at 17492 chars, so the model was handed a head and a tail
    with a third of the story silently missing. It then complained about
    repetition caused BY the seam, which sent the fix in the wrong direction.

    The old number was a guess from before the prompt grew. Measured now:
    the script prompt is ~2.9k tokens (11234 chars), the answer is capped by
    the context window (`_script_answer_budget`), and llm_ctx is 32768. A
    44000-char chapter fits whole with headroom; the corpus's longest is
    40764. It still truncates (loudly) rather than silently past the limit.
    """
    if limit is None:
        limit = int(CONFIG.get("llm_chapter_chars", 44000))
    if len(chapter_text) <= limit:
        return chapter_text
    head = int(limit * 0.75)
    tail = limit - head
    log(f"  [WARN] chapter {len(chapter_text)} chars > {limit} budget — "
        f"truncated to {head}+{tail} (head+tail kept, middle dropped)", 1)
    return (chapter_text[:head] + "\n\n[... middle omitted: chapter exceeds "
            f"the {limit}-char breakdown budget ...]\n\n" + chapter_text[-tail:])


def _expand_narrations(scenes, chapter_text, chapter_num):
    """Stretch each short line with the beat's facts from the chapter.

    The frame can show ONE image, so the narration is the viewer's only
    channel for the plot - and this model habitually writes ~20-word lines
    no matter how the script prompt asks (14, then 19, then 21 words across
    three prompt revisions), leaving 60-70% of the chapter's information on
    the floor. A second pass with the chapter in context and explicit
    "write the beat's facts IN FULL" instructions carries the story into the
    audio track.

    Soft-fail by design: a parse failure keeps the original lines, because
    losing the render over polish is the wrong trade.
    """
    try:
        seeds = "\n".join(
            f"{s.get('scene_number')}: {s.get('narration')}" for s in scenes)
        excerpt = _chapter_excerpt(chapter_text)
        prompt = (
            "You are an anime scriptwriter rewriting a narration track. Below "
            "are the scene seeds of a novel chapter adaptation and the raw "
            "chapter. Rewrite EACH numbered narration line so it carries that "
            "beat's facts IN FULL: 35 to 55 words each, active voice, "
            "grade-6 words, never inventing a fact that is not in the chapter. "
            "Keep the story order and the question/forward-pull ending of "
            "each line. Return ONLY a JSON array of objects "
            '{"scene_number": N, "narration": "..."} - exactly one object per '
            "seed line, no other text.\n\n"
            f"NARRATION SEEDS:\n{seeds}\n\nCHAPTER:\n{excerpt}")
        raw = llm_chat([{"role": "user", "content": prompt}],
                       timeout=int(CONFIG.get("llm_script_timeout", 900)),
                       temperature=0.3,
                       seed=int(CONFIG.get("llm_seed", 42)) + 9000 + chapter_num,
                       max_tokens=8192)
        if not raw:
            log("  [WARN] narration expansion returned nothing - keeping original lines", 1)
            return scenes
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        start, end = cleaned.index("["), cleaned.rindex("]")
        rows = json.loads(cleaned[start:end + 1])
        by_num = {}
        for row in rows:
            if isinstance(row, dict) and row.get("scene_number") is not None:
                line = " ".join(str(row.get("narration") or "").split())
                if line:
                    by_num[int(row["scene_number"])] = line
        before = sum(len((s.get("narration") or "").split()) for s in scenes)
        changed = 0
        for s in scenes:
            num = int(s.get("scene_number", 0))
            if num in by_num:
                s["narration"] = by_num[num]
                changed += 1
        after = sum(len((s.get("narration") or "").split()) for s in scenes)
        log(f"  [VOICE] narration expanded: {changed}/{len(scenes)} lines, "
            f"{before} -> {after} words", 1)
        return scenes
    except Exception as e:
        log(f"  [WARN] narration expansion skipped "
            f"({type(e).__name__}: {str(e)[:150]}) - keeping original lines", 1)
        return scenes


def break_into_scenes(chapter_text, chapter_num):
    """Break chapter into viral hook-style scenes via the LLM server. Deterministic."""
    # The LLM only ever sees the EXCERPT, so the excerpt is what the target
    # and the answer budget are computed from.
    excerpt = _chapter_excerpt(chapter_text)
    target = scene_target_count(len(excerpt))
    char_bible_str = ""
    if CHAR_BIBLE and CHAR_BIBLE.get("characters"):
        names = ", ".join(CHAR_BIBLE["characters"].keys())
        char_bible_str = f"Known characters: {names}."
    try:
        viral_block = viral_prompt_block()
    except Exception:
        viral_block = "Scene 1 must open with a shocking hook. Write 40-70 words per narration line."

    prompt = f"""You are an anime scriptwriter. Break this novel chapter into EXACTLY {target} scenes for a COMPLETE adaptation of the chapter's essential story — not a recap, not a trailer, not a transcription.

COVERAGE — THE PRIME DIRECTIVE (outranks every rule below):
- Adapt EVERY essential beat: the inciting event, the key reveals, the turning
  points, the consequences that matter next chapter — and every beat the test
  below keeps. A named character, a number, a deadline, a threat — those are
  essential and they all stay. Atmosphere, repeated beats, minor byplay and
  secondary description are NOT; cut them first.
- The test for essential: if skipping this beat makes a LATER scene stop making
  sense, it stays. Otherwise it goes, no matter how good the prose is.
- Nothing essential may live ONLY in the dialogue field — only narration is
  voiced. If a spoken line carries plot information (a name, a warning, a
  threat, an arrangement), put that information INTO the narration too, e.g.
  She told him he had a visitor last night. Never let a fact exist solely in
  dialogue the viewer will not hear.
- EVERY scene must have narration text. A scene with an empty narration AND
  an empty dialogue is a SILENT scene: the episode is voiced scene by scene,
  so a blank scene has no audio and aborts the whole render. A visual beat
  still needs a line — name what the shot shows, e.g. "The market was already
  emptying when he reached the south gate." Never leave a field empty.
- A viewer who has not read the book must follow the whole story from the
  narration alone — the WHOLE story, at ~{target} scenes, not a line dropped.
  Compress only what the essential test cuts; keep the rest.
- Prefer one dense, fact-carrying line over three thin ones. Do not pad to
  reach {target}; if the chapter's essential beats fill fewer scenes, use fewer.
- SHOW, THEN TELL (how studios actually adapt prose): the IMAGE carries what
  can be seen — a cream card on a table, a sealed personnel file, a grey suit
  that does not fit, a headline on page seven. Narration carries what cannot
  be seen — names, motives, numbers, deadlines, what a pause meant. Never
  spend a sentence describing the pixels (the colour of the coat, the fog
  density, the pose) — put that in image_prompt instead. But DO spend
  sentences on what the beat MEANS: the narration is the viewer's only channel
  for the plot, so it carries every fact the beat turns on, in full.
- WEIGHT THE BEATS: the chapter's one or two biggest turns may take 2-3
  scenes each — the arrival, the reveal, the confrontation. Ordinary beats
  get one scene. Never give a minor beat more screen time than the main turn.

VIEWER-LOSS FACTS THIS PROMPT IS BUILT AGAINST (measured, not opinion):
- 20-40% of viewers leave within the first 10 seconds. 55% are gone by 60s.
- A pattern interrupt inside the first 5 seconds lifts retention ~23%.
- Stating what the viewer gets within 15 seconds lifts 1-minute retention ~18%.
- Retention below 50% at 10-15s means the HOOK failed. Nothing else matters then.
A scene here is about 2 seconds of screen time, so: scenes 1-3 are the first
~5 seconds, scenes 1-7 are the first 15. That is the whole window you get.

HOOK - scenes 1, 2 and 3 together (this is the highest-leverage writing in the
file; ~35 words total across the three):
- Name a CONCRETE thing from the chapter in scene 1: an object, a number, a
  named place, a named moment. "The mill burned for the third night" works;
  "the village was dark and full of fear" does not. No mood, no weather.
- WITHHOLD THE MECHANISM. State the outcome, never explain how it happens.
  If you say WHY in scene 1 there is no reason to stay for scene 2.
- Give the stakes by scene 3: what is lost if the viewer does not keep going.
- NEVER open with a greeting, an introduction, a channel mention, or "in this
  video". Those are instant, measurable drop-off and are rejected downstream.
- Say "you" or "your" at least once in scenes 1-3. Talk TO the viewer.
- START scenes 1-3 with a bare opening word that pulls the reader in:
    a pronoun ("He woke...", "She heard...", "They know..."),
    a question ("What if the room were watching you?"),
    a number ("Three knocks. Then nothing."),
    or a negation ("Nobody moved.", "Don't look up.")
  These are the opening patterns the retention checker recognises; a scene
  that starts with a proper noun ("Edvyn wakes on the floor.") scores as a
  flat opening and loses the hook. Aim for two different triggers across
  scenes 1-3 so it does not read as a tic.
- Scene 3, and any re-hook line, works best when it withholds one specific
  unanswered detail: a name, a number, or a reason, stated as a fact with no
  explanation attached.

RE-HOOK - EVERY scene except the last:
Every line must leave something UNANSWERED so the viewer needs the next line.
Leave out the RESOLUTION. Withhold one specific thing: a name, a number, a
reason, an identity. Do NOT carry the pull with punctuation.

GOOD (each withholds something specific, each ends differently):
  "The registry sent letters. It never sent a card."   -> withholds why
  "Edvyn took the money and said nothing."             -> withholds to whom
  "Someone had already been inside the room."          -> withholds who
  "The visitor came nine days before anyone died."      -> withholds the link
  "What did the mother know?"                          -> a real open question

HARD RULES:
- NEVER end a line with "...". It is not a re-hook, it is a tic, and an
  audience notices within three scenes. At most ONE line in the whole script
  may end that way.
- Do not start two consecutive lines with the same word.
- Vary the ending: a withheld fact, a hard question, an unexplained object,
  a hard stop. Do not use the same construction on consecutive lines.
- A line that merely states what happened, with nothing withheld, is a scene
  where the viewer has no reason to continue. Avoid it.

WRITING STYLE:
- Short sentences, active voice. One BEAT per scene — but a beat can run
  several sentences: the frame shows ONE image, so the narration carries
  everything else the beat says.
- Under-length narration is the FAILURE MODE. The chapter is ~1000 words and
  the frame shows one still per beat, so the narration is the viewer's only
  channel for the plot: write 40-70 words per narration line unless the beat
  is a single silent action, and carry its specifics — names, motives,
  numbers, deadlines, consequences. A one-line summary of a beat the chapter
  spends sentences on is information thrown away. Do not pad with filler
  words; every sentence must carry a NEW fact from the chapter.
- Grade ~6 vocabulary: short common words, no Latinate abstractions.
- No semicolons, no em-dashes, no emojis, no hashtags, no asterisks.
- NEVER reuse the same narration line twice in the episode. Each line is
  checked for uniqueness and repetition is treated as a broken script.
- Every line implies the question the next line answers.

STRUCTURE:
- Connect scenes with BUT-THEREFORE, not and-then: this happened, BUT this,
  THEREFORE he had to. "And then" carries no information and reads as filler.
- Minor turn every ~10 seconds. Major turn at ~45% and again in the last 15%.
- Final scene ends on a face or an unanswered question, never an answer.
- Use power words: secret, forbidden, dead, betrayed, hidden, final, escaped,
  caught, warning, never, truth, buried, vanished, hunted, trapped, curse, blood.
- Vary shots: close-up, wide establishing, object detail, reaction.
- At least 20% of scenes must be "establishing" WORLD-BUILDING shots: the
  white-sky city, canals, terraces, fog, tea house exteriors - no characters
  needed, pure atmosphere that shows the world of the story. SPREAD them
  through the WHOLE episode: at least one in the first quarter, one in the
  middle half, one in the final quarter. World-building must breathe through
  the entire story, never clustered.

IMAGE_PROMPT RULES BY SHOT SIZE (the still is a real anime frame, not a
portrait with a smeared backdrop):
- For "wide" / "extreme-wide" / "establishing" scenes, image_prompt LEADS
  WITH THE WORLD: architecture, palette, light, weather, then a small figure
  if any. Name foreground, middle-ground and background elements (a lamp-post
  in front, canal houses mid-frame, spires in the fog). Describe the place,
  not the person's face. Good: "gothic canal district at dawn, peeling brick
  terraces and iron bridges, a lone figure crossing, fog on the water".
- For "close-up" / "medium" scenes, lead with the character and expression
  instead; the environment is one clause, not the subject.
- Never write "blurry background", "bokeh" or "depth of field" — the whole
  frame is drawn sharp, background included.
{viral_block}

{char_bible_str}

Return ONLY a valid JSON array with EXACTLY {target} items, no other text.
Top level MUST be [...] - never a single {{...}} object.
Do NOT wrap the array in ```json fences; emit the bare [ and ].
Each item:
{{"scene_number": N, "type": "action|dialogue|establishing",
 "characters": ["name"] or [], "narration": "...", "dialogue": "...",
 "image_prompt": "detailed visual prompt, anime",
 "animation_hint": "camera/motion hint",
 "shot_size": "close-up|medium|wide|extreme-wide",
 "angle": "eye-level|low|high|dutch|over-shoulder",
 "camera_move": "static|push-in|pull-out|pan-left|pan-right|tracking|handheld",
 "mood": "tense|somber|hopeful|menacing|melancholic",
 "stages": [{{"character": "name", "stage": "short_stage_name",
 "tags": "outfit/prop/lighting change tags"}}] or [],
 "text_overlay": "3-6 VIRAL WORDS IN CAPS", "hook_type": "shock|mystery|emotion|cliffhanger|tension"}}
Only add a stage entry when a character's outfit, props, injury, or lighting
meaningfully change (NOT for emotion, location, or time-of-day alone).
STUDIO RULES (adhere strictly):
- Faithful, UNCENSORED adaptation: keep the chapter's darkness, horror, and
  dread intact. Never sanitize, moralize, or soften. No comforting conclusions
  the chapter didn't earn. The hook rules above sharpen the opening; they do
  NOT license a happy ending.
- Scene 1 MUST name one visible verb readable at phone scale (an event, not
  a mood).
- Return ONLY the JSON array: EXACTLY {target} scenes, top level [...].
  No prose, no fences, never a single object.

CHAPTER:
{excerpt}"""

    temp0 = float(CONFIG.get("llm_temperature", CONFIG.get("ollama_temperature", 0.2)))
    seed0 = int(CONFIG.get("llm_seed", CONFIG.get("ollama_seed", 42))) + chapter_num
    free_comfyui_vram()
    result, last_err = None, None
    _req = prompt
    # The answer's budget is the CONTEXT WINDOW, not a policy cap: prompt +
    # chapter + answer must fit in llm_ctx or the server rejects the request
    # outright. Asking for more than fits is a hard error, so it is clamped
    # here - and the scene target above is derived from the same budget, so
    # the LLM is never asked for more scenes than it can actually return.
    _max_tok = _script_answer_budget(len(excerpt), len(prompt))
    for attempt in range(3):
        try:
            # Slightly raise temperature on retries for a parseable answer.
            # timeout 900 / max_tokens 14336: a 25-scene chapter measured 4281
            # completion tokens in 419 s, so the old 300/4096 pair truncated
            # mid-array AND timed out on every attempt. The budget is now
            # sized for 56 scenes at ~240 tok each (complete-adaptation mode);
            # 56*240+512 headroom = 13952, rounded up. The two other long
            # generators in this file (motion-prompt expand, narrative check)
            # pass their own small budgets and are unaffected.
            _probe = llm_chat([{"role": "user", "content": _req}],
                              timeout=int(CONFIG.get("llm_script_timeout", 900)),
                              temperature=temp0 + attempt * 0.1,
                              seed=seed0 + attempt * 1000,
                              max_tokens=_max_tok)
            # Validate parse early so a bad answer retries instead of crashing
            _why = _probe_reason(_probe, target) if _probe else "no answer"
            if _probe and _why is None:
                result = {"message": {"content": _probe}}
                break
            # ponytail: a silent scene is REPAIRABLE. The model likes to emit
            # one wordless "establishing" beat, and every scene in this
            # pipeline is voiced, so the voice gate kills the whole episode
            # for it. The beat is a nicety; the episode is not. After ONE
            # reason-fed retry (the model reliably repeats the same blank
            # establishing shot), drop the blank scene(s) instead of aborting.
            if _probe and "empty narration" in str(_why) and attempt >= 1:
                _scenes_parsed = _parse_scenes(_probe)
                _kept = [s for s in _scenes_parsed
                         if ((s.get("narration") or "").strip()
                             or (s.get("dialogue") or "").strip())]
                if len(_kept) >= max(3, target // 2):
                    log(f"  [FIX] dropped {len(_scenes_parsed) - len(_kept)} "
                        "silent scene(s) after 3 attempts - the episode is "
                        "voiced scene by scene, so a wordless beat has no "
                        "audio", 1)
                    result = {"message": {"content": json.dumps(_kept)}}
                    break
            log(f"  [WARN] Script answer rejected: {_why} "
                f"(finish_reason={_LAST_FINISH_REASON}), retry {attempt + 1}/3... "
                f"head={(_probe or '')[:160]!r}", 2)
            # ponytail: keep the rejected answer on disk. ch019 burned three
            # 10-minute attempts and the log held only the first 160 chars,
            # so the malformation that killed the render was un-diagnosable.
            try:
                _rawf = _asset("videos") / f"ch{chapter_num:03d}_script_attempt{attempt + 1}.txt"
                _rawf.write_text(_probe or "", encoding="utf-8")
            except OSError:
                pass
            # ponytail: tell the model WHAT was wrong. Resending the same
            # prompt cannot fix a specific defect, and the defect here is
            # specific: one establishing scene left blank.
            _req = (prompt + "\n\nYOUR PREVIOUS ANSWER WAS REJECTED: " + str(_why)
                    + "\nReturn the FULL JSON array again with that defect fixed. "
                      "Every scene needs a narration line; nothing may be empty.")
            result = None
            # ponytail: persist WHY — a bare '3x: None' tells no one the model
            # kept returning unparseable JSON (parse-fail != HTTP-fail).
            last_err = RuntimeError(
                f"model returned an unusable script 3x "
                f"(last reason: {_why}, "
                f"finish_reason={_LAST_FINISH_REASON}, "
                f"head={(_probe or '')[:80]!r})")
            time.sleep(5)
        except Exception as e:
            last_err = e
            log(f"  [WARN] Script gen failed ({e}), retry {attempt + 1}/3...")
            time.sleep(10)
    if result is None:
        raise RuntimeError(f"Script generation failed 3x: {last_err}")
    msg = result.get("message", {})
    raw = msg.get("content", "")
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()
    # Strip markdown fences the model likes to add
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    scenes = _parse_scenes(cleaned)
    _bad = _validate_scenes(scenes, target)
    if _bad:
        raise ValueError(_bad)
    # Normalize scene numbers, then carry the chapter's facts into the voice
    for i, s in enumerate(scenes):
        s["scene_number"] = i + 1
    scenes = _expand_narrations(scenes, chapter_text, chapter_num)
    scenes = _enforce_shot_diversity(scenes)
    # Cache for consistency across runs
    scenes_file = Path(CONFIG["scenes_dir"]) / f"ch{chapter_num:03d}.json"
    scenes_file.parent.mkdir(parents=True, exist_ok=True)
    with open(scenes_file, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2, ensure_ascii=False)
    return scenes


# --- Image Generator (ComfyUI) -------------------------------------
# P2: pose preset library (text-only ControlNet substitute — no new nodes).
# scene["pose"] picks one; keeps off-model gesturing out of free text.


def _lora_candidate_order(scene):
    """Cast keys LoRA attachment may try, subject-first.

    The LoRA carries ONE face and its trigger token outranks plain prompt
    text, so attaching it when the shot is of someone ELSE draws the wrong
    face on purpose: ch2 s16 (Hartwell's arrival) drew Edvyn; ch3 s3 (Maret
    opening the door) drew Edvyn, because Edvyn was in the cast list and the
    loop fell through to his LoRA. The image_prompt names the subject — when
    it names any cast member, ONLY those may drive the LoRA (a named character
    with no LoRA of her own means no LoRA at all, so her bible tags carry the
    face instead of a hijack). When the prompt names nobody, fall back to
    script order (first cast member) exactly as before.
    """
    cands = [str(c).lower().replace(" ", "_")
             for c in (scene.get("characters", []) or [])]
    focal = _focal_character(scene)
    if focal:
        # The director's plan says who the scene is ABOUT. Only the subject
        # may drive the LoRA: ch016 s18 (the intruder in the doorway) lists
        # Edvyn first because Edvyn is who the intruder looks AT, and the
        # prompt-named rule then attached Edvyn's face to the antagonist.
        return [focal]
    ip = str(scene.get("image_prompt", "") or "").lower()
    named = [c for c in cands if (c in ip or c.split("_")[0] in ip)]
    return named if named else cands


# ponytail: the count token ("1girl"/"1boy"/"1man"/"solo") and the style
# tail ("dark fantasy, gothic, detailed eyes") are PER-CHARACTER fragments.
# Concatenating two characters' tag lists hands the model two count tokens
# and two style tails for a two-person frame, and Qwen settles the conflict
# by picking ONE gender for the whole frame: ch016 s6 (Lena's close-up with
# Edvyn in the room) rendered TWO MEN, and s19 (the intruder) rendered
# Edvyn's face on the man.
_COUNT_TOKENS = frozenset({"solo", "1girl", "1boy", "1man", "1woman", "1child",
                           "1other", "1people"})
_STYLE_TAIL = frozenset({"dark fantasy", "gothic", "detailed eyes"})


def _cast_keys(scene):
    """Normalised cast keys for a scene."""
    return [str(c).lower().replace(" ", "_")
            for c in ((scene or {}).get("characters") or [])
            if isinstance(c, str)]


def _char_tags(char_key):
    """Identity tags for one character: danbooru, else prompt, else fields."""
    data = _bible_entry(char_key) or {}
    tags = str(data.get("danbooru_tags") or "").strip()
    if tags:
        return tags
    tags = str(data.get("prompt_tags") or "").strip()
    if tags:
        return tags
    bits = [str(data.get(k) or "").strip() for k in
            ("hair", "eyes", "clothing", "distinguishing_features")]
    return ", ".join(b for b in bits if b)


def _focal_character(scene):
    """The director's focal_character for this scene, if it names the cast."""
    focal = str((scene or {}).get("focal_character") or "").strip().lower()
    if not focal:
        return ""
    cast = _cast_keys(scene)
    key = focal.replace(" ", "_")
    if key in cast:
        return key
    for c in cast:
        first = c.split("_")[0]
        if first == key or key.startswith(first):
            return c
    return ""


def _shot_subject(scene):
    """Who the shot is OF - not merely who is in the room.

    1. the director's focal_character (the plan saw the beat sheet),
    2. the single character the image_prompt names,
    3. the first cast member.
    """
    cast = _cast_keys(scene)
    if not cast:
        return ""
    focal = _focal_character(scene)
    if focal:
        return focal
    ip = str((scene or {}).get("image_prompt") or "").lower()
    named = [c for c in cast if (c in ip or c.split("_")[0] in ip)]
    if len(named) == 1:
        return named[0]
    return cast[0]


def _reduced_tags(tags):
    """A non-subject's tags: no "solo", no style tail. The count token stays -
    it is how the model knows a second person is in frame."""
    out = []
    for t in (x.strip() for x in str(tags).split(",")):
        if not t or t.lower() == "solo" or t.lower() in _STYLE_TAIL:
            continue
        out.append(t)
    return ", ".join(out)


def _strip_solo(tags):
    """Drop "solo" - it is a one-person instruction and this frame has two."""
    return ", ".join(t for t in (x.strip() for x in str(tags).split(","))
                     if t and t.lower() != "solo")


def _character_prompt_tags(scene):
    """Identity tags for the still prompt: the subject in full, the rest reduced.

    A one-character scene comes out byte-identical to the old concatenation,
    so every solo still on disk stays cache-valid.
    """
    cast = _cast_keys(scene)
    if not cast:
        return ""
    subject = _shot_subject(scene)
    ip = str((scene or {}).get("image_prompt") or "").lower()
    two_shot = "two-shot" in ip or "two shot" in ip
    solo_frame = _shot_norm(scene) in ("close-up", "extreme-close-up") and not two_shot
    parts = []
    for key in cast:
        tags = _char_tags(key)
        if not tags:
            continue
        if key == subject:
            parts.append(_strip_solo(tags) if len(cast) > 1 else tags)
        elif solo_frame:
            continue
        else:
            frag = _reduced_tags(tags)
            if frag:
                parts.append(frag)
    return ", ".join(parts)


def _apply_director_plan(scenes, dp):
    """Copy the director's focal_character onto the scenes.

    ponytail: the plan names WHO each scene is about and WHERE it happens,
    and neither field reached the scenes. Without the focal character the
    still prompt has to guess the subject out of free text - and it guessed
    wrong on ch016 s18, where the intruder in the doorway drew Edvyn's face
    because the shot's other character is the one he looks AT.
    """
    plan = {}
    for sp in ((dp or {}).get("scene_plan") or []):
        if isinstance(sp, dict) and isinstance(sp.get("scene_number"), int):
            plan[sp["scene_number"]] = sp
    n = 0
    for sc in scenes or []:
        sp = plan.get(sc.get("scene_number"))
        if not sp:
            continue
        focal = str(sp.get("focal_character") or "").strip()
        if focal:
            sc["focal_character"] = focal
            n += 1
        loc = str(sp.get("location") or "").strip()
        if loc:
            sc["location"] = loc
        # Cinematography: the director's shot list now drives lens, lighting,
        # camera movement and mood, not just WHO/WHERE. Explicit scene values
        # from the script are overridden by the approved director's package.
        for target, sources in {
            "shot_size": ("intended_shot_type", "shot_size"),
            "angle": ("angle",),
            "camera_move": ("camera_movement", "camera_move"),
            "lens": ("lens",),
            "lighting": ("lighting",),
            "mood": ("mood",),
            "visual_motif": ("visual_motif",),
            "time_of_day": ("time_of_day",),
        }.items():
            for src in sources:
                val = str(sp.get(src) or "").strip()
                if val:
                    sc[target] = val
                    break
    _rebalance_camera_moves(scenes)
    return n


# The director's shot list may over-use one move (measured: 9/21 push-in),
# which trips the script diversity gate. Keep the director's intent where it
# fits, then cap each move at 40% of the chapter with the least-used alternate.
_CAMERA_MOVES = ("push-in", "pull-out", "pan-left", "pan-right", "tracking",
                 "handheld", "static")


def _rebalance_camera_moves(scenes):
    cap = max(1, math.floor(len(scenes or []) * 0.4))
    if len(scenes or []) < 3:
        return 0
    counts = {}
    for sc in scenes:
        m = str(sc.get("camera_move") or "static")
        counts[m] = counts.get(m, 0) + 1
    changed = 0
    for sc in scenes:
        m = str(sc.get("camera_move") or "static")
        if counts.get(m, 0) <= cap:
            continue
        alt = min(_CAMERA_MOVES, key=lambda x: (counts.get(x, 0), x != m, x))
        if alt == m:
            alt = next((x for x in _CAMERA_MOVES if x != m), m)
        sc["camera_move"] = alt
        counts[m] -= 1
        counts[alt] = counts.get(alt, 0) + 1
        changed += 1
    if changed:
        log(f"  [DIVERSITY] camera moves rebalanced: {changed} shot(s) capped at {cap}", 2)
    return changed


POSE_PRESETS = {
    "stand": "standing neutral pose, feet planted, arms relaxed",
    "walk": "mid-stride walking pose, coat and hair in motion",
    "run": "running pose leaning forward, dynamic motion lines avoided",
    "sit": "seated pose, hands resting, weight settled",
    "kneel": "kneeling pose, weight low, hands braced",
    "reach": "reaching forward pose, arm extended toward viewer",
    "look": "head turned looking over shoulder, eyes visible",
    "carry": "carrying pose, arms bearing weight, shoulders tensed",
}


def _lora_is_qwen(path):
    """True when a LoRA file targets the Qwen DiT (not SDXL).

    LoRAs are architecture-locked, so attaching an SDXL LoRA to a Qwen DiT
    is meaningless. Qwen-native LoRAs carry diffusion_model.* keys and
    ss_base_model_version=qwen_image_2. The old code skipped EVERY LoRA on a
    Qwen DiT, discarding the Qwen-native character LoRA and leaving nothing
    but the prompt's tags to carry identity.
    """
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            meta = f.metadata() or {}
        if any(k.startswith("diffusion_model.") for k in keys):
            return True
        return "qwen" in (meta.get("ss_base_model_version") or "").lower()
    except Exception:
        return False


def _shot_norm(scene):
    """Normalized shot_size: 'close up'/'CU' -> 'close-up', etc.

    ponytail: the LLM emits variants ('extreme wide', 'close up'); every
    gate compared raw strings, so variants silently fell through to the
    wrong branch (full-body fallback, face-lock kept, echo skipped).
    """
    if not isinstance(scene, dict):
        return ""
    return str(scene.get("shot_size", "") or "").lower().replace(" ", "-").replace("_", "-").strip()


def _wide_shot(scene):
    """True for wide/establishing frames — the shots that must render the
    WORLD sharp (deep focus, detailed background), not a sharp face over a
    blurred backdrop. Same detection the clip phase uses for face-lock-off.
    """
    return (_shot_norm(scene) in ("wide", "extreme-wide")
            or str(scene.get("type", "") or "").lower() == "establishing")


# Cinematography direction, owned by the project so EVERY chapter matches.
# Lens words describe perspective/compression only — the series draws the
# whole frame sharp, so no "bokeh"/"depth of field" language (script rule).
_CINE_LENS = {
    "extreme-close-up": "100mm macro lens, compressed perspective, intimate framing",
    "close-up": "85mm portrait lens, compressed perspective, tight framing",
    "medium": "50mm lens, natural perspective",
    "wide": "35mm wide-angle lens, deep focus",
    "extreme-wide": "24mm wide-angle lens, deep focus, expansive scale",
}
_CINE_LIGHT = {
    "menacing": "hard directional key light, deep chiaroscuro shadows, high contrast",
    "tense": "hard directional key light, deep shadows, strong contrast",
    "somber": "soft diffused light, low contrast, muted palette",
    "melancholic": "soft diffused light, low contrast, muted palette",
    "hopeful": "soft warm key light, gentle rim light",
    "night": "warm gaslamp key light, deep shadows, cool moonlight rim",
}


def _sheet_identity_phrase(scene):
    """VLM-derived identity text for a sheet-only character, or ''.

    Frequent characters get a trained LoRA (always wins). A low-appearance
    character has a 3-view sheet on gray; its VLM description is cached in
    characters/sheets/<key>_identity.txt and injected as identity words in
    BOTH the still and clip prompts. No image is fed to the generator, so the
    gray sheet backdrop never appears in a shot (measured: feeding the sheet
    as a latent or reference copied the gray panel into every frame).
    """
    if not isinstance(scene, dict):
        return ""
    key = str(scene.get("focal_character") or "").strip().lower().replace(" ", "_")
    if not key:
        cast = scene.get("characters") or []
        if len(cast) == 1:
            key = str(cast[0]).strip().lower().replace(" ", "_")
    if not key or _qwen_lora_installed(key):
        return ""
    txt = _asset("characters") / "sheets" / f"{key}_identity.txt"
    try:
        return txt.read_text(encoding="utf-8").strip() if txt.is_file() else ""
    except OSError:
        return ""


def _cine_phrase(scene):
    """Lens + lighting direction for one shot, from the director's plan.

    Explicit `lens`/`lighting` on the scene win; otherwise derive from
    shot_size and mood/time_of_day. Kept separate from the script's style
    fragments so a book can own its look without losing the camera language.
    """
    if not isinstance(scene, dict):
        return ""
    lens = str(scene.get("lens") or "").strip() or _CINE_LENS.get(_shot_norm(scene), "")
    light = str(scene.get("lighting") or "").strip()
    if not light:
        mood = str(scene.get("mood") or "").lower().strip()
        tod = str(scene.get("time_of_day") or "").lower().strip()
        light = _CINE_LIGHT.get(mood) or _CINE_LIGHT.get(tod) or ""
    return ", ".join(b for b in (lens, light) if b)


def pose_for_scene(scene):
    p = str(scene.get("pose", "") or "").lower().strip()
    return POSE_PRESETS.get(p, "")
def _reroll_still(scene, scene_num, chapter_num, images_dir, seed_offset):
    """Regenerate one still and return its REAL path inside images_dir, or None.

    generate_image returns a bare filename that only exists in ComfyUI's output
    folder. Every caller that stored that return value straight into
    scene_images[] handed the rest of the pipeline a path to a file that was
    never copied here — so the still phase "succeeded" and the clip phase died
    on FileNotFoundError an hour later.

    That is not hypothetical: ch2 crashed twice on it. The duplicate-collapse
    re-roll replaced scenes 14 and 16 with bare names, the parallax pass warned
    it could not read the image, and H3 staging raised on scene 16 and killed
    the render. The black-still regen has the identical shape and would crash
    the same way.

    So: resolve the ComfyUI file, copy it next to the other stills, write the
    sidecar so the cache fingerprint is honest, and return the full path.
    """
    name = generate_image(scene, scene_num, chapter_num, seed_offset=seed_offset)
    if not name:
        return None
    src = find_comfy_file(name)
    if src is None:
        log(f"  [WARN] reroll for scene {scene_num} produced no file on disk", 1)
        return None
    dst = images_dir / name
    shutil.copy2(src, dst)
    try:
        _write_still_sidecar(dst, scene)
    except Exception as e:
        log(f"  [WARN] sidecar write failed for scene {scene_num} ({e})", 2)
    return dst


_VETHMORE_WORDS = ("vethmore", "coal smoke", "morning", "kitchen")
_STRONG_INTERIOR = ("room", "hallway", "corridor", "upstairs", "downstairs",
                    "staircase", "stairs", "kitchen", "study", "office",
                    "indoors", "sitting room", "bedroom", "apartment", "cabin")
_WEAK_INTERIOR = ("house", "door", "doorway", "table", "desk", "cot", "flat")
_EXTERIOR_CUES = ("street", "docks", "dock", "alley", "square", "yard",
                  "canal", "cobblestone", "outside", "quarter")


def _setting_key_for(scene_text, location=""):
    """Which setting tail a scene gets: indoors, the real city, or the copy.

    Indoors wins because the white-sky tail names cobblestones, lamp posts
    and a canal; an interior prompt carrying it renders the room out on the
    street (ch021 s6: a cot and stove on the pavement). The director's
    location is the strongest signal, then a named room ("records office")
    beats the district it sits in ("Crown Quarter").

    The legacy keys below are the LEGACY NOVEL's. Any other novel carries its
    own settings in its bible, so a fanfic with its own world (a crimson-moon
    city) gets THAT world in the prompt: the setting is chosen by keyword
    score over the novel's own setting entries, not by the legacy hardcoded
    cues that would have painted every scene with cobblestones and a flat
    white sky. "interior" stays a GENERIC key in every novel (a room painted
    as a street is a known failure), and the legacy novel's own resolution
    order is untouched so its cached stills keep validating.
    """
    settings = CHAR_BIBLE.get("settings", {}) if isinstance(CHAR_BIBLE, dict) else {}
    loc = str(location or "").lower()
    text = str(scene_text or "").lower()
    novel = active_novel()
    if novel is not None and not novel.legacy:
        if "interior" in settings and (
                any(w in loc for w in _STRONG_INTERIOR)
                or any(w in text for w in _STRONG_INTERIOR)):
            return "interior"
        own_keys = [k for k in settings if k != "interior"]
        if own_keys:
            best, best_score = None, 0
            for k in own_keys:
                entry = settings.get(k) or {}
                kws = [str(x).lower() for x in (entry.get("keywords") or [])]
                if not kws:
                    kws = [k.replace("_", " ")]
                score = sum(1 for w in kws if w and w in text)
                if score > best_score:
                    best, best_score = k, score
            return best or own_keys[0]
    if loc:
        if any(w in loc for w in _STRONG_INTERIOR):
            return "interior"
        if any(w in loc for w in _EXTERIOR_CUES):
            return "white_sky_city"
        if any(w in loc for w in _WEAK_INTERIOR):
            return "interior"
    if any(w in text for w in _STRONG_INTERIOR):
        return "interior"
    if any(w in text for w in _EXTERIOR_CUES):
        return "white_sky_city"
    if any(w in text for w in _WEAK_INTERIOR):
        return "interior"
    if any(w in text for w in _VETHMORE_WORDS):
        return "vethmore"
    return "white_sky_city"


def generate_image(scene, index, chapter_num, focus_text=None, suffix="", seed_offset=0,
                   framing=True):
    """Generate reference image for video creation. Cinematic anime style."""
    prompt_text = scene.get("image_prompt", "masterpiece, best quality, anime scene, dark fantasy")
    if focus_text:
        prompt_text = f"{prompt_text}, depicting: {focus_text}"
    negative = scene.get("negative_prompt", "worst quality, low quality, blurry")

    characters_in_scene = scene.get("characters", [])
    if CHAR_BIBLE and characters_in_scene:
        # ponytail: ONE subject's tags in full, everyone else reduced. The old
        # loop concatenated every cast member's danbooru_tags, which put two
        # count tokens and two style tails in a two-person frame and made the
        # model pick one gender for the whole shot (ch016 s6 rendered Lena's
        # close-up as two men).
        _ctags = _character_prompt_tags(scene)
        if _ctags:
            prompt_text = f"{prompt_text}, {_ctags}"
    # Sheet-only character: identity words derived from its 3-view sheet by
    # the VLM (cached once). No image enters the workflow, so no gray panel.
    _sheet_id = _sheet_identity_phrase(scene)
    if _sheet_id:
        prompt_text = f"{prompt_text}, identity reference: {_sheet_id}"

    # STAGE LOCKS: outfit/prop/lighting changes live in the text prompt
    stage_tags_all = [ (st.get("tags") or "").strip()
                       for st in scene.get("stages", []) or [] ]
    stage_tags_all = [t for t in stage_tags_all if t]
    if stage_tags_all:
        prompt_text = f"{prompt_text}, {', '.join(stage_tags_all)}"

    _neg_extra = _style_spec(scene)["negative"]
    negative = (
        f"{negative}, worst quality, low quality, blurry, deformed, ugly, "
        "bad anatomy, bad hands, extra fingers, missing fingers, extra limbs, "
        "fused fingers, too many fingers, long neck, mutated, poorly drawn, "
        "bad proportions, malformed limbs, missing arms, missing legs, "
        "extra digit, fewer digits, cropped, jpeg artifacts, signature, "
        "watermark, username, artist name, text, error, nsfw, nude, "
        "childish, amateur, low detail, flat colors, bright colors, "
        # ponytail: VLM on the rain-dance test reported soft focus, blue
        # cast, crushed blacks, highlight banding/oversaturation. None of
        # those are covered by "blurry"/"flat colors" alone.
        "out of focus, soft focus, blurry focus, hazy, bloom, motion blur, "
        "colour cast, color cast, blue tint, cold cast, underexposed, "
        "crushed blacks, oversaturated, banding, gradient banding, "
        "deformed face, ugly face, distorted face, head out of frame, cropped head, "
        "head cut off, decapitated, faceless, no face, 3d render"
        + (f", {_neg_extra}" if _neg_extra else "")
    )

    scene_num = scene.get("scene_number", index + 1)

    # HARD ANIME LOCK (2D anime only, never photoreal)
    narration_focus = scene.get("narration") or scene.get("dialogue") or ""
    focus_words = " ".join(narration_focus.strip().split()[:14])
    if focus_words:
        prompt_text = f"{prompt_text}, depicting: {focus_words}"
    prop_tags = props_for_scene(scene)
    if prop_tags:
        prompt_text = f"{prompt_text}, {prop_tags}"
    # Setting bible: world-building environments (white-sky city, Vethmore).
    # ponytail: the world is a DEFAULT, not an opt-in. This gate used to add the
    # environment only when the scene text happened to contain "white sky",
    # "city" or "fog", so every other scene got NO world tags at all and Qwen
    # filled the gap with its own default: bright suburban daylight. That is how
    # a sunless gothic city render came out with a blonde schoolgirl under a
    # blue sky. The series is set in the copy, so white_sky_city is the default
    # and Vethmore (the original) is the explicit opt-out.
    settings = CHAR_BIBLE.get("settings", {})
    _loc = str(scene.get("location") or "").strip()
    scene_text = " ".join([scene.get("narration", ""), scene.get("dialogue", ""),
                           scene.get("image_prompt", ""), _loc]).lower()
    _skey = _setting_key_for(scene_text, _loc)
    if _loc:
        prompt_text = f"{prompt_text}, {_loc}"
    _vtags = settings.get(_skey, {}).get("visual_tags", "")
    if _vtags:
        prompt_text = f"{prompt_text}, {_vtags}"
    # Shot preset direction (storyboard discipline)
    shot = " ".join(x for x in [scene.get("shot_size", ""), scene.get("angle", ""),
                                scene.get("mood", "") + " mood" if scene.get("mood") else ""]
                    if x).strip()
    if shot:
        prompt_text = f"{prompt_text}, {shot} shot"
    # Cinematography: lens + lighting travel with the shot so the still and
    # its H3 clip share one camera language (rules/cinematography.md).
    _cine = _cine_phrase(scene)
    if _cine:
        prompt_text = f"{prompt_text}, {_cine}"
    # ponytail: framing follows SHOT SIZE, not a blanket full-body rule.
    # "full body, head to boots" in a 1344x768 landscape frame forces the
    # model to fill frame width with torso and push the head out — exactly
    # the torso-only close-ups that shipped. Wides keep head-to-toe (user
    # mandate); close-up/medium get head-first portrait composition.
    # Close-up companions (suffix "c") serve lips.
    if framing and characters_in_scene:
        _shot = _shot_norm(scene)
        # ponytail: no shot_size used to land in the full-body else-branch, so
        # every scene the LLM left un-sized (40% of cached scenes) got
        # "head to boots" and a ~90px face. Un-sized defaults to medium now;
        # only an EXPLICIT wide/establishing intent earns full body.
        if not _shot:
            _shot = "medium"
        # REVERTED 2026-10-15. An earlier version of this block treated a
        # disagreement between the prompt's framing words and shot_size as
        # "the prompt is more specific, so the prompt wins", and suppressed the
        # framing clause on conflict. That premise is false here: _enforce_
        # shot_diversity (see 3020-3031) STRIPS the writer's own framing lead
        # and injects FRAMING_POOL[i % 20] into image_prompt, while shot_size
        # is rotated by a separate pool (pools["shot_size"]). So both sides are
        # mechanical assignments that know nothing about each other, and the
        # keyword test was firing on a random collision between them - it
        # suppressed the face-protection clause on 9 of 40 ch007 scenes (s02,
        # s03, s05 ...) with nothing authorial behind it, and dropped
        # "body shot without face", the exact term that guards the documented
        # 90px-face bug. Measured, not assumed: the ch007 prompt leads
        # ("tight close-up on the eyes", "low angle full body shot", ...) are
        # FRAMING_POOL entries verbatim, not director prose.
        # The real contradiction is internal to the pipeline and belongs in
        # _enforce_shot_diversity (pick a pool entry matching shot_size), not
        # here. Until then, apply the shot_size clause unconditionally - a
        # consistent clause beats a randomly suppressed one.
        if _shot == "close-up":
            prompt_text = (f"{prompt_text}, close-up portrait, face fills most of frame, "
                           "head and shoulders, eyes and expression dominant, "
                           "face large and centered, head fully inside frame, "
                           "generous headroom above head, not torso shot, not waist-up")
            negative += (", torso shot, torso only, waist-up crop, waist up only, "
                         "body shot without face, midsection, hips, torso filling frame")
        elif _shot == "medium":
            prompt_text = (f"{prompt_text}, medium shot chest-up, face large and centered, "
                           "head and shoulders prominent, not a torso shot, "
                           "head fully inside frame, generous headroom above head")
            negative += (", torso shot, torso only, waist-up crop, waist up only, "
                         "body shot without face, midsection, hips, torso filling frame")
        else:
            prompt_text = (f"{prompt_text}, full body, head to boots, complete figure head to toe, "
                           "head fully inside frame, face clearly visible, generous headroom above head")
    # P2: pose preset (scene["pose"]) — deterministic posture vocab.
    _pose = pose_for_scene(scene)
    if _pose:
        prompt_text = f"{prompt_text}, {_pose}"

    # ANIME DNA (researched): hard cel-shadow bands, weighted ink lines,
    # character rooted IN a painted environment, disciplined palette
    _style = _style_spec(scene)
    # ponytail: the face DNA led with five face-detail tags for EVERY shot, so
    # a wide establishing frame still spent its capacity on the face and the
    # model paid for it with a blurred, depth-of-field backdrop. Wides swap
    # the face block for an environment block: the QwenCloud formula puts
    # Camera + Setting first for a reason, and the literal blur vocabulary
    # (depth of field / bokeh / blurry background) goes to the NEGATIVE — the
    # existing "soft focus" negative never named the actual mechanism.
    _is_wide = _wide_shot(scene)
    _dna = ("detailed painted background, intricate architecture, "
            "deep depth of field, everything in sharp focus, "
            "foreground middle-ground and background all detailed, "
            if _is_wide else
            "detailed face, detailed eyes, sharp pronounced facial structure, "
            "defined jawline and cheekbones, crisp facial detail, ")
    prompt_text = (
        "2d anime, masterpiece, best quality, absurdres, highres, "
        f"{_dna}"
        f"{prompt_text}, "
        "character standing in detailed painted environment, "
        "figure and background in same scene, environmental storytelling, "
        f"{_style['still']}, "
        "hand-drawn linework with natural weight variation, crisp ink contours, clean line art, "
        "2d animation, key visual, flat color palette, sakuga motion, "
        "two-tone cel shading, hard-edged shadow shapes, flat restrained local colors, "
        "limited harmonious palette, jewel tones, "
        "motivated key light with rim lighting, cinematic composition, "
        "studio quality, broadcast quality, professional anime, "
        # ponytail: "selective focus" + "film grain" read as blur/soft-focus
        # requests and were in the positive block (VLM: soft, 1.5% face).
        # Sharpness is asked for explicitly instead.
        "sharp focus, crisp edges, high micro-contrast, "
        "clean neutral colour grade, even exposure, no colour cast, "
        "readable silhouette, clear face and costume boundaries, "
        "8k resolution, ultra detailed"
    )
    if _is_wide:
        prompt_text += (", wide establishing shot, deep focus, no bokeh, "
                        "background in crisp focus, detailed environment")
        negative += (", depth of field, bokeh, shallow depth of field, "
                     "blurry background, defocused background, blurred city, "
                     "blurred buildings, background blur")
    # ANTI-CGI DRIFT: explicitly ban 3D render artifacts (they creep back otherwise)
    negative = (negative + ", photorealistic, live action, live-action, pixar, "
                "disney, western cartoon, 3d render, cgi, unreal engine, blender render, "
                "photorealistic skin, oily skin, subsurface scattering, "
                "volumetric 3d shadows, realistic skin texture, 3d ai puppet")

    save_prefix = f"ch{chapter_num:03d}_s{scene_num:02d}{suffix}_{uuid.uuid4().hex}"
    # m4: watch prompt bloat (SDXL CLIP truncated ~77 tokens/encoder;
    # Qwen3-VL TE takes far longer context, length log still useful).
    log(f"  Still prompt chars: {len(prompt_text)} (scene {scene_num}{suffix})", 3)
    # Portrait-native for vertical platforms so nothing is squeezed/cropped
    out_w, out_h = get_output_dims()
    if out_h > out_w:
        gen_w, gen_h = 768, 1344
    else:
        gen_w, gen_h = 1344, 768
    seed = int(CONFIG.get("image_seed_base", 2000)) + chapter_num * 1000 + scene_num + seed_offset

    def build_workflow():
        # ponytail: character LoRA (edvyn ep5 proven: trigger + description
        # summons him on-model). Convention: models/loras/<key>*.safetensors.
        # Qwen-Image-2.1 (DiT): UNETLoader/UnetLoaderGGUF + CLIPLoader + VAELoader.
        # Legacy SDXL checkpoint path kept for old caches only.
        _ckpt = str(CONFIG.get("checkpoint", ""))
        _is_qwen = "qwen" in _ckpt.lower()
        _unet_cls = "UnetLoaderGGUF" if _ckpt.lower().endswith(".gguf") else "UNETLoader"
        _te = str(CONFIG.get("qwen_text_encoder", "qwen3vl_8b_int8_convrot.safetensors"))
        _vae = str(CONFIG.get("qwen_vae", "qwen_image_2.1_vae_bf16.safetensors"))
        _model_ref, _clip_ref, _extra, _trigger = ["1", 0], ["0", 0], {}, None
        # The novel's STYLE LoRA goes FIRST in the chain, so the character's
        # identity LoRA sits on top of the book's look (Qwen-Image chains
        # LoRAs natively: 21 -> 20 -> sampler).
        _style_lora = _novel_style_lora()
        if _style_lora:
            _extra["21"] = {"class_type": "LoraLoader",
                            "inputs": {"model": ["1", 0], "clip": ["0", 0],
                                       "lora_name": _style_lora["name"],
                                       "strength_model": _style_lora["strength"],
                                       "strength_clip": _style_lora["strength"]}}
            _model_ref, _clip_ref = ["21", 0], ["21", 1]
            log(f"  Style LoRA: {_style_lora['name']} @ {_style_lora['strength']}", 2)
        # (LoRA/anchor selection helpers live at module scope — the anchor
        # block above needs the same Qwen-vs-SDXL test)
        for _ck in _lora_candidate_order(scene):
            _hits = sorted(f for _d in _comfy_dirs("models")
                           if (_d / "loras").exists()
                           for f in (_d / "loras").glob(f"{_ck}*.safetensors"))
            # prefer a Qwen-format LoRA when the DiT is Qwen (sorted() alone
            # would pick the older SDXL file alphabetically)
            if _hits and _is_qwen:
                _qwen_hits = [f for f in _hits if _lora_is_qwen(f)]
                _hits = _qwen_hits or _hits
            if _hits:
                if _is_qwen and not _lora_is_qwen(_hits[-1]):
                    log(f"  LoRA skipped on Qwen DiT (SDXL-only): {_hits[-1].name}", 2)
                    break
                _ls = float(CONFIG.get("lora_strength", 0.7))
                # chain from whatever is already loaded (the style LoRA), so
                # the character identity sits ON TOP of the book's style
                _extra["20"] = {"class_type": "LoraLoader",
                                 "inputs": {"model": _model_ref, "clip": _clip_ref,
                                            "lora_name": _hits[-1].name,
                                            "strength_model": _ls, "strength_clip": _ls}}
                _model_ref, _clip_ref, _trigger = ["20", 0], ["20", 1], _ck
                log(f"  LoRA: {_hits[-1].name} @ {_ls}", 2)
                break
        # Qwen native: the trigger is the LoRA's own token, not the file stem.
        # ponytail: only when a LoRA actually loaded ("20" is that node).
        # The configured token belongs to ONE character (ohwx_edvyn) and was
        # being appended to every Qwen prompt — including scenes with no LoRA
        # behind it, which dragged LoRA-less characters toward Edvyn's face.
        if "20" in _extra:
            # ponytail: the global trigger belongs to ONE character and must
            # not ride another character's LoRA. Only Edvyn had a Qwen LoRA
            # before, so "ohwx_edvyn" happened to be right; the moment a
            # second character (ch11 unknown_man) got a LoRA, his scenes would
            # have pulled Edvyn's trigger toward Edvyn's face. A bible
            # `lora_trigger` (empty string = no trigger) overrides the global.
            _bib = CHAR_BIBLE.get("characters", {}) if isinstance(CHAR_BIBLE, dict) else {}
            _per = (_bib.get(_ck) or {}).get("lora_trigger")
            if _per is not None:
                _trigger = str(_per).strip() or None
            else:
                # The configured token belongs to ONE character. Riding it on
                # another character's LoRA buys nothing and pulls that shot
                # toward the owner's face, so require an explicit owner.
                _owner = str(CONFIG.get("qwen_lora_trigger_owner", "") or "")
                _trigger = (str(CONFIG.get("qwen_lora_trigger", "")).strip()
                            if _owner and _ck == _owner else None)
        _ptest = prompt_text + (f", {_trigger}" if _trigger else "")
        # A novel's style LoRA may carry its own trigger token; without it the
        # adapter has nothing to key the style on.
        _style_lora = _novel_style_lora()
        if _style_lora and _style_lora.get("trigger"):
            _ptest += f", {_style_lora['trigger']}"
        if _is_qwen:
            wf = {
                "1": ({"class_type": "UnetLoaderGGUF",
                       "inputs": {"unet_name": _ckpt}}
                      if _unet_cls == "UnetLoaderGGUF" else
                      {"class_type": "UNETLoader",
                       "inputs": {"unet_name": _ckpt, "weight_dtype": "default"}}),
                 "0": {"class_type": "CLIPLoader",
                       "inputs": {"clip_name": _te, "type": "qwen_image",
                                  "device": str(CONFIG.get(
                                      "still_encoder_device", "default"))}},
                "7": {"class_type": "VAELoader",
                      "inputs": {"vae_name": _vae}},
                 "2": {"class_type": "TextEncodeQwenImage21",
                       "inputs": {"prompt": _ptest, "negative_prompt": negative,
                                  "clip": _clip_ref, "vae": ["7", 0],
                                  # INT, multiples of 32; 0 = keep each ref at own size.
                                  "resolution": 1024}},
                "4": {"class_type": "EmptyLatentImage",
                      "inputs": {"width": gen_w, "height": gen_h, "batch_size": 1}},
                 "5": {"class_type": "KSampler",
                       "inputs": {"seed": seed,
                                  "steps": int(CONFIG.get("auto_tune", {}).get("image_steps", 25)),
                                  "cfg": float(CONFIG.get("image_cfg", 1.0)),
                                  "sampler_name": str(CONFIG.get("image_sampler", "euler")),
                                  "scheduler": str(CONFIG.get("image_scheduler", "simple")),
                                  "denoise": 1.0, "model": _model_ref,
                                  "positive": ["2", 0], "negative": ["2", 1],
                                  "latent_image": ["4", 0]}},
                "6": {"class_type": "VAEDecode",
                      "inputs": {"samples": ["5", 0], "vae": ["7", 0]}},
                "7b": {"class_type": "SaveImage",
                       "inputs": {"filename_prefix": save_prefix, "images": ["6", 0]}},
            }
            wf.update(_extra)
            return wf
        # Legacy SDXL checkpoint path (pre-Qwen caches only).
        _model_ref, _clip_ref = ["1", 0], ["1", 1]
        if "20" in _extra:
            _extra["20"]["inputs"]["clip"] = ["1", 1]
        wf = {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": CONFIG["checkpoint"]}},
            "2": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": _ptest, "clip": _clip_ref}},
            "3": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": negative, "clip": _clip_ref}},
            "4": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": gen_w, "height": gen_h, "batch_size": 1}},
            "5": {"class_type": "KSampler",
                  "inputs": {"seed": seed,
                             "steps": int(CONFIG.get("auto_tune", {}).get("image_steps", 25)),
                             "cfg": float(CONFIG.get("image_cfg", 1.0)),
                             "sampler_name": str(CONFIG.get("image_sampler", "euler")),
                             "scheduler": str(CONFIG.get("image_scheduler", "simple")),
                             "denoise": 1.0, "model": _model_ref,
                             "positive": ["2", 0], "negative": ["3", 0],
                             "latent_image": ["4", 0]}},
            "6": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage",
                  "inputs": {"filename_prefix": save_prefix, "images": ["6", 0]}},
        }
        wf.update(_extra)
        return wf

    def submit(wf):
        result = http_post_json(f"{CONFIG['comfyui_url']}/prompt",
                                {"prompt": wf, "client_id": str(uuid.uuid4())}, timeout=30)
        entry = _comfy_wait(result["prompt_id"], 120, 3, "still")
        if not entry:
            return None
        for out in entry.get("outputs", {}).values():
            if out.get("images"):
                return out["images"][0]["filename"]
        return None

    return submit(build_workflow())

# --- Character card (training-image maker, NOT a render input) ----------
# The card is a multi-view portrait of one character, generated from the
# bible's tags. Its ONLY job is to be a LoRA's TRAINING SET: the still path
# is pure T2I and never loads it (an anchored scene repeated one picture and
# dragged the card's flat backdrop into the scene — see memory.md). It is
# generated on demand, only for a character who has no training images yet.
SHEET_VIEWS = [
    ("front", "front view bust portrait, facing viewer, symmetrical, face fully visible looking at viewer, eyes open"),
    ("threequarter", "three-quarter view bust portrait, head turned slightly left, face fully visible, eyes open"),
    ("profile", "full side profile view bust portrait, facing right"),
    ("back", "back three-quarter view bust portrait, looking over shoulder"),
    ("fullbody", "full body standing portrait, facing viewer, neutral pose, head to boots"),
    ("closeup", "extreme close-up face portrait, detailed eyes, face fully visible looking at viewer, eyes open"),
]
# ponytail: angle views need room to rotate (0.75); near-front views stay
# tight to the anchor (0.6). Uniform 0.6 collapsed everything to busts.
SHEET_DENOISE = {"front": 0.6, "threequarter": 0.6, "profile": 0.75, "back": 0.75,
                 "fullbody": 0.75, "closeup": 0.6}


def sheet_seed(char_key):
    keys = list((CHAR_BIBLE.get("characters", {}) or {}).keys())
    return 5000 + (keys.index(char_key) if char_key in keys else len(char_key)) * 17


def generate_character_sheet(char_key):
    """6-view character card for one character. Skips if the card exists.

    TRAINING IMAGES, not a render input: the still path never loads this.
    It exists so a character who has never been rendered can still get a
    LoRA — the same 7-view recipe the 40 shipped character LoRAs were
    trained on. Written to characters/sheets/ for the owner's reference and
    staged into dataset/<key>_lora/ for the training run.
    """
    sheets_dir = _asset("characters") / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    out = sheets_dir / f"{char_key}_sheet.png"
    if out.exists():
        return out
    char_data = (CHAR_BIBLE.get("characters", {}) or {}).get(char_key, {})
    # ponytail: BOTH tag sets (prompt_tags alone drops age/body words the
    # model needs; danbooru alone drops costume detail). Never say
    # "reference/model sheet" — the image model renders collages instead.
    base = ", ".join(t for t in (char_data.get("prompt_tags", ""),
                                 char_data.get("danbooru_tags", "")) if t).strip()
    if not base:
        # Say the CONSEQUENCE, not just the action: "sheet skipped" reads like a
        # nothing-happened note, and this character then has no reference sheet
        # for the whole episode.
        log(f"  [WARN] No bible tags for {char_key} - NO character card will be "
            "built, so this character cannot get a LoRA. Add "
            "danboru_tags to characters.json.", 1)
        return None
    neg = ("worst quality, low quality, blurry, deformed, 3d render, cgi, "
           "photorealistic, multiple people, multiple faces, collage, grid, "
           "expression sheet, turnaround, sketch, multiple views, crowd, text, watermark")
    # ponytail: fullbody FIRST as the I2I anchor — a bust anchor has no
    # body/angle info to reveal, so every view collapsed back to busts.
    _vdir = dict(SHEET_VIEWS)
    _order = ["fullbody", "front", "threequarter", "profile", "back", "closeup"]
    made, base_path = {}, None
    _ref_staged = None
    _ref = _asset("characters") / "references" / f"{char_key}.png"
    if _ref.is_file():
        try:
            from PIL import Image as _PIL
            _im = _PIL.open(_ref).convert("RGB")
            _sc = max(896 / _im.width, 1152 / _im.height)
            _im = _im.resize((int(_im.width * _sc + .5), int(_im.height * _sc + .5)), _PIL.LANCZOS)
            _x = (_im.width - 896) // 2
            _y = max(0, (_im.height - 1152) // 4)
            _im = _im.crop((_x, _y, _x + 896, _y + 1152))
            _tmp = _asset("_temp") / f"ref_anchor_{char_key}.png"
            _tmp.parent.mkdir(parents=True, exist_ok=True)
            _im.save(_tmp)
            _ref_staged = stage_comfy_input(str(_tmp))
        except Exception as e:
            log(f"  [WARN] reference anchor skipped ({e})", 2)
    for vname in _order:
        vdir = _vdir[vname]
        if should_stop():
            return None
        set_progress("characters", _order.index(vname) + 1, len(_order), f"{char_key} {vname}")
        prompt = (f"2d anime, single clean portrait, {vdir}, {base}, "
                  "plain light gray background, no scenery, single character, solo, "
                  "one face only, sharp pronounced facial structure, defined jawline and "
                  "cheekbones, crisp facial detail, neutral expression, consistent character "
                  "design, official anime art, "
                  "clean line art, two-tone cel shading, detailed face, detailed eyes")
        seed = sheet_seed(char_key) + _order.index(vname)
        # ponytail: I2I anchor — fullbody locks identity for the rest.
        # 2-pass refine (poor-man's FaceDetailer, core nodes only — Civitai
        # needs login, Impact needs a detector that fails on anime):
        # pass 1 composes, pass 2 re-renders at 0.35 with a face prompt.
        face_prompt = (f"2d anime, sharp detailed face, crisp facial features, "
                       f"detailed eyes, refined face, {base}, high quality, very aesthetic")
        latent_node = {"class_type": "EmptyLatentImage",
                       "inputs": {"width": 896, "height": 1152, "batch_size": 1}}
        denoise, extra_nodes = 1.0, {}
        # First view (fullbody) anchors on the character's reference portrait
        # so the card locks to the actual design, not just text tags.
        _anchor_src = base_path if base_path is not None else (_ref_staged if vname == "fullbody" else None)
        if _anchor_src is not None:
            staged_base = stage_comfy_input(base_path) if base_path is not None else _ref_staged
            extra_nodes["10"] = {"class_type": "LoadImage",
                                 "inputs": {"image": staged_base}}
            latent_node = {"class_type": "VAEEncode",
                           "inputs": {"pixels": ["10", 0], "vae": ["30", 0]}}
            denoise = 0.75 if (vname == "fullbody" and base_path is None) else SHEET_DENOISE.get(vname, 0.6)
        _sckpt = str(CONFIG.get("checkpoint", ""))
        _s_te = str(CONFIG.get("qwen_text_encoder", "qwen3vl_8b_int8_convrot.safetensors"))
        _s_vae = str(CONFIG.get("qwen_vae", "qwen_image_2.1_vae_bf16.safetensors"))
        wf = {
            "1": ({"class_type": "UnetLoaderGGUF",
                   "inputs": {"unet_name": _sckpt}}
                  if _sckpt.lower().endswith(".gguf") else
                  {"class_type": "UNETLoader",
                   "inputs": {"unet_name": _sckpt, "weight_dtype": "default"}}),
            "0": {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": _s_te, "type": "qwen_image", "device": "cpu"}},
            "30": {"class_type": "VAELoader",
                   "inputs": {"vae_name": _s_vae}},
             "2": {"class_type": "TextEncodeQwenImage21",
                   "inputs": {"prompt": prompt, "negative_prompt": neg,
                              "clip": ["0", 0], "vae": ["30", 0],
                              "resolution": 1024}},
             "11": {"class_type": "TextEncodeQwenImage21",
                    "inputs": {"prompt": face_prompt, "negative_prompt": neg,
                               "clip": ["0", 0], "vae": ["30", 0],
                               "resolution": 1024}},
            "4": latent_node,
            "5": {"class_type": "KSampler",
                   "inputs": {"seed": seed,
                              "steps": int(CONFIG.get("auto_tune", {}).get("image_steps", 25)),
                              "cfg": float(CONFIG.get("image_cfg", 1.0)),
                              "sampler_name": str(CONFIG.get("image_sampler", "euler")),
                              "scheduler": str(CONFIG.get("image_scheduler", "simple")),
                              "denoise": denoise, "model": ["1", 0],
                              "positive": ["2", 0], "negative": ["2", 1],
                              "latent_image": ["4", 0]}},
            "6": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["5", 0], "vae": ["30", 0]}},
            "8": {"class_type": "VAEEncode",
                  "inputs": {"pixels": ["6", 0], "vae": ["30", 0]}},
            "9": {"class_type": "KSampler",
                   "inputs": {"seed": seed + 999,
                              "steps": int(CONFIG.get("auto_tune", {}).get("image_steps", 25)),
                              "cfg": float(CONFIG.get("image_cfg", 1.0)),
                              "sampler_name": str(CONFIG.get("image_sampler", "euler")),
                              "scheduler": str(CONFIG.get("image_scheduler", "simple")),
                              "denoise": 0.35, "model": ["1", 0],
                              "positive": ["11", 0], "negative": ["11", 1],
                              "latent_image": ["8", 0]}},
            "12": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["9", 0], "vae": ["30", 0]}},
            "7": {"class_type": "SaveImage",
                  "inputs": {"filename_prefix": f"sheet_{char_key}_{vname}_{uuid.uuid4().hex}",
                             "images": ["12", 0]}},
        }
        wf.update(extra_nodes)
        log(f"  Card {char_key}: {vname}...", 2)
        try:
            result = http_post_json(f"{CONFIG['comfyui_url']}/prompt",
                                    {"prompt": wf, "client_id": str(uuid.uuid4())}, timeout=30)
            prompt_id = result["prompt_id"]
            got = _comfy_output_filename(_comfy_wait(prompt_id, 120, 3, "sheet"))
            src = find_comfy_file(got) if got else None
            if src is None:
                log(f"  [WARN] Card view missing: {char_key} {vname}", 2)
                continue
            dst = sheets_dir / f"{char_key}_{vname}.png"
            shutil.copy2(src, dst)
            made[vname] = dst
            if base_path is None:
                base_path = str(dst)  # fullbody anchors the rest
        except (ComfyUIError, Exception) as e:
            log(f"  [WARN] Card view failed: {char_key} {vname} ({str(e)[:100]})", 2)
    _tiled = [(v, made[v]) for v, _ in SHEET_VIEWS if v in made]
    if len(_tiled) < 2:
        return None
    # ponytail: ordered + labeled panels (FRONT first). Absolute fontfile —
    # font names segfault this ffmpeg build (hook lesson).
    _ff = "C\\:/Windows/Fonts/arial.ttf" if Path("C:/Windows/Fonts/arial.ttf").exists() else "C\\:/Windows/Fonts/impact.ttf"
    _labs = []
    for vname, m in _tiled:
        _lab = m.parent / f"{m.stem}_lab.png"
        _tag = vname.replace("_", " ").upper()
        r = subprocess.run(
            [CONFIG["ffmpeg"], "-y", "-i", str(m), "-vf",
             f"drawtext=text='{_tag}':fontfile='{_ff}':fontsize=44:fontcolor=white:"
             f"x=24:y=24:box=1:boxcolor=black@0.6:boxborderw=14",
             str(_lab)], capture_output=True, timeout=60)
        _labs.append(str(_lab) if r.returncode == 0 and _lab.exists() else str(m))
    ins, filt = [], []
    for idx, m in enumerate(_labs):
        ins += ["-i", str(m)]
        filt.append(f"[{idx}]scale=768:1024:force_original_aspect_ratio=decrease,"
                    f"pad=768:1024:(ow-iw)/2:(oh-ih)/2:color=0xE8E8E8[a{idx}]")
    # ponytail: ffmpeg tile takes ONE input in v9 — xstack the grid instead.
    _lay = "|".join(f"{768 * (i % 3)}_{1024 * (i // 3)}" for i in range(len(_labs)))
    filt.append(f"{''.join(f'[a{i}]' for i in range(len(_labs)))}"
                f"xstack=inputs={len(_labs)}:layout={_lay}")
    r = subprocess.run([CONFIG["ffmpeg"], "-y", *ins,
                        "-filter_complex", ",".join(filt), str(out)],
                       capture_output=True, timeout=120)
    return out if r.returncode == 0 and out.exists() else None


def _stage_character_card(char_key):
    """Generate the character's card and stage its views as the training set.

    The card is NOT a render input (the still path is pure T2I). It exists so
    a character who has never been rendered can still get a LoRA: the card
    views become the captioned images in dataset/<key>_lora/, which is the
    exact recipe the 40 shipped character LoRAs were trained on.
    """
    card = generate_character_sheet(char_key)
    if not card:
        return False
    ds = _asset("dataset") / f"{char_key}_lora"
    ds.mkdir(parents=True, exist_ok=True)
    n = 0
    for v in [v for v, _ in SHEET_VIEWS] + ["sheet"]:
        src = _asset("characters") / "sheets" / f"{char_key}_{v}.png"
        if src.is_file():
            shutil.copy2(src, ds / src.name)
            n += 1
    return n >= 3


def _character_identity_ready(char_key):
    """True when a character has an identity source the model can use.

    The bible's danbooru_tags / prompt_tags ARE the identity source: they go
    into the text prompt of every shot that character is in. A trained Qwen
    LoRA is a stronger source layered on top of them, never a requirement.
    """
    key = (char_key or "").lower().replace(" ", "_")
    if not key:
        return False
    entry = (CHAR_BIBLE.get("characters", {}) or {}).get(key) or {}
    return bool(str(entry.get("danbooru_tags") or "").strip()
                or str(entry.get("prompt_tags") or "").strip())


def ensure_character_pages(char_keys):
    """Build the dossier page for every cast member (no GPU work)."""
    for k in char_keys or []:
        if should_stop():
            return
        key = (k or "").lower().replace(" ", "_")
        if key:
            build_character_page(key)


def _loras_dirs():
    """Every ComfyUI models/loras dir the still builder globs."""
    return [d / "loras" for d in _comfy_dirs("models")]


def _lora_installed(char_key):
    return any(p.exists() for d in _loras_dirs() if d.exists()
               for p in d.glob(f"{char_key}*.safetensors"))


def _qwen_lora_installed(char_key):
    """True when a LoRA the CURRENT model would actually attach exists.

    An SDXL LoRA on a Qwen DiT is skipped at build time ("LoRA skipped on
    Qwen DiT"), so counting one as an identity source would let a character
    render with no identity source at all.
    """
    for d in _loras_dirs():
        if not d.exists():
            continue
        for p in d.glob(f"{char_key}*.safetensors"):
            if _lora_is_qwen(p):
                return True
    return False


# ponytail: 400 steps / linear16 is the recipe the builder already validated
# (edvyn v2, maren, aldric, unknown_man all shipped with it); only the
# machine-specific paths/venv live in config.lora_training.
_AITK_TMPL = """---
job: extension
config:
  name: "{name}"
  process:
    - type: 'sd_trainer'
      training_folder: "{output_dir}"
      device: cuda:0
      performance_log_every: 50
      network:
        type: "lora"
        linear: {linear}
        linear_alpha: {alpha}
      save:
        dtype: bf16
        save_every: 200
        max_step_saves_to_keep: 2
      datasets:
        - folder_path: "{dataset_dir}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [ 512, 768, 1024 ]
      train:
        batch_size: 1
        cache_text_embeddings: true
        steps: {steps}
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: {lr}
        skip_first_sample: true
        disable_sampling: true
        dtype: bf16
      model:
        name_or_path: "{model}"
        extras_name_or_path: "{extras}"
        arch: "{arch}"
        quantize: true
        qtype: "float8"
        quantize_te: true
        qtype_te: "qfloat8"
        low_vram: false
meta:
  name: "[name]"
  version: '1.0'
"""


def train_character_lora(char_key):
    """Train a Qwen-Image-2.1 LoRA from a character's staged training images.

    Dataset = <novel>/dataset/<key>_lora/ — PNGs plus a .txt caption each
    (the bible's danbooru_tags is written for any caption that is missing).
    It used to be built from the 6-view character sheet; the sheets are gone,
    so the images are staged instead. Runs ai-toolkit in the _training venv;
    installs the result into every ComfyUI loras dir and restarts ComfyUI so
    the folder listing (which lora_name is validated against) includes it for
    phase 3.
    """
    cfg = CONFIG.get("lora_training") or {}
    bible = (CHAR_BIBLE.get("characters", {}) or {}).get(char_key) or {}
    tags = str(bible.get("danbooru_tags", "") or "").strip()
    name = f"{char_key}_qwen21_v1"
    # Training images are STAGED, not generated: drop captioned PNGs into
    # <novel>/dataset/<key>_lora/ and this trains on them. (It used to be
    # built from the 6-view character sheet; the sheets are gone.)
    ds = _asset("dataset") / f"{char_key}_lora"
    if not ds.is_dir():
        raise RuntimeError(f"no training images in {ds} - stage captioned PNGs there")
    n = 0
    for src in sorted(ds.glob("*.png")):
        cap = src.with_suffix(".txt")
        if not cap.is_file():
            # The bible's tags are only the FALLBACK: a STYLE LoRA has no
            # character entry, and its caption is the style's trigger word.
            if not tags:
                raise RuntimeError(
                    f"no caption for {src.name} and no danbooru_tags in the "
                    f"bible - stage a .txt beside it")
            cap.write_text(tags + "\n", encoding="utf-8")
        n += 1
    if n < 3:
        raise RuntimeError(f"only {n} training image(s) in {ds} - need >=3")
    steps = int(cfg.get("steps", 400))
    cfg_dir = Path(cfg["config_dir"])
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / f"{name}.yaml"
    cfg_file.write_text(_AITK_TMPL.format(
        name=name, linear=int(cfg.get("linear", 16)),
        alpha=int(cfg.get("linear_alpha", 16)), steps=steps,
        lr=cfg.get("lr", 1e-4), arch=str(cfg.get("arch", "qwen_image_2")),
        output_dir=str(cfg["output_dir"]).replace("\\", "/"),
        dataset_dir=str(ds).replace("\\", "/"),
        model=str(cfg["model"]).replace("\\", "/"),
        extras=str(cfg["extras"]).replace("\\", "/")), encoding="utf-8")
    log_file = _asset("videos") / f"lora_{char_key}.log"
    log(f"  Training LoRA {name}: {n} images, {steps} steps (~21 min at the "
        f"3.2 s/step measured on this box, 2026-11-13). "
        f"Log: {log_file.name}", 1)
    free_comfyui_vram()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n==== training {name} {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        proc = subprocess.run(
            [str(cfg["python"]), "run.py", str(cfg_file)],
            cwd=str(cfg["aitk_dir"]), stdout=lf, stderr=subprocess.STDOUT,
            env=env, timeout=int(cfg.get("timeout_s", 14400)))
    if proc.returncode != 0:
        raise RuntimeError(f"ai-toolkit exit {proc.returncode} (see {log_file.name})")
    out = Path(cfg["output_dir"]) / name / f"{name}.safetensors"
    if not out.is_file():
        raise RuntimeError(f"no {name}.safetensors produced (see {log_file.name})")
    installed = []
    for d in _loras_dirs():
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, d / out.name)
        installed.append(d / out.name)
    log(f"  LoRA installed: {out.name} ({out.stat().st_size // 1048576} MB) "
        f"-> {len(installed)} loras dir(s)", 1)
    restart_comfyui(why=f"installed LoRA {out.name}")
    return installed


def ensure_cast_loras(char_keys):
    """Train a LoRA for any cast member without one, before any still.

    ponytail: ch11 learned unknown_man, rendered him from bible tags alone, and
    got his LoRA an hour AFTER the episode shipped — so it never touched the
    episode that introduced him. Off entirely when config.lora_training.enabled
    is false, skipped when there are no staged training images to train on,
    and every failure warns-and-continues so learning never aborts a render.
    """
    cfg = CONFIG.get("lora_training") or {}
    if not cfg.get("enabled"):
        return
    for key in char_keys or []:
        if should_stop():
            return
        key = (key or "").lower().replace(" ", "_")
        if not key or _lora_installed(key):
            continue
        staged = list((_asset("dataset") / f"{key}_lora").glob("*.png"))
        if len(staged) < 3:
            # Cold start: this character has never been rendered and nobody
            # staged reference art for them, so make a card and train on it.
            # The card is a TRAINING SET, not a render input (the still path
            # is pure T2I). A character whose card already exists, or who has
            # staged art of their own, skips this entirely.
            log(f"  No training images for {key} - making a character card", 1)
            if not _stage_character_card(key):
                log(f"  [WARN] Could not stage training images for {key} - "
                    "rendering from bible tags only", 1)
                continue
        try:
            train_character_lora(key)
        except Exception as e:
            log(f"  [WARN] LoRA training for {key} failed ({str(e)[:140]}); "
                "rendering without it", 1)


def build_character_page(char_key):
    """Single dossier page: bio + prompt tags + voice."""
    import html as _h
    chars = (CHAR_BIBLE.get("characters", {}) or {})
    data = chars.get(char_key, {})
    if not data:
        return None
    pages = _asset("characters") / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    out = pages / f"{char_key}.html"

    def _voice_file():
        try:
            ref = str(CONFIG.get("character_voices", {}).get(char_key, ""))
            name = Path(ref).name if ref else ""
            cand = _asset("characters") / "voices" / name
            if name and cand.exists():
                return f"../voices/{name}"
        except Exception:
            pass
        return ""

    _skip = {"prompt_tags", "danbooru_tags", "negative_tags"}
    _rows = "".join(
        f"<tr><th>{_h.escape(str(k).replace('_', ' ').title())}</th>"
        f"<td>{_h.escape(str(v))}</td></tr>"
        for k, v in data.items()
        if not k.startswith("_") and k not in _skip and v)
    _views = "".join(
        f'<figure><img loading="lazy" src="../sheets/{char_key}_{v}.png" alt="{v}">'
        f"<figcaption>{v.replace('_', ' ').upper()}</figcaption></figure>"
        for v, _ in SHEET_VIEWS
        if (_asset("characters") / "sheets" / f"{char_key}_{v}.png").exists())
    _card = _asset("characters") / "sheets" / f"{char_key}_sheet.png"
    _card_html = (f'<img class="sheet" src="../sheets/{char_key}_sheet.png" alt="character card">'
                  if _card.exists() else
                  "<p><i>No card yet - it is made when this character needs a "
                  "LoRA (it is a training set, not a render input).</i></p>")
    _voice = _voice_file()
    _audio = (f'<audio controls preload="none" src="{_voice}"></audio>' if _voice
              else "<p><i>No voice sample.</i></p>")
    _tags = _h.escape(data.get("prompt_tags", ""))
    _neg = _h.escape(data.get("negative_tags", ""))
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_h.escape(data.get('full_name', char_key))} — Character Dossier</title>
<style>:root{{color-scheme:dark}}body{{background:#141118;color:#e8e2f0;font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:24px}}h1{{margin:0}}h1 small{{color:#9a8fb0;font-weight:400}}.role{{color:#c9a7eb}}img.sheet{{width:100%;border:1px solid #3a3348;border-radius:8px}}.views{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0}}figure img{{width:100%;border:1px solid #3a3348;border-radius:6px}}figcaption{{text-align:center;color:#9a8fb0;font-size:13px}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #3a3348;padding:8px 10px;text-align:left;vertical-align:top}}th{{width:180px;color:#c9a7eb}}pre{{background:#1e1a28;padding:12px;border-radius:6px;white-space:pre-wrap;word-break:break-word}}button{{background:#5b3f8c;color:#fff;border:0;border-radius:6px;padding:8px 16px;cursor:pointer}}audio{{width:100%;margin:8px 0}}a{{color:#c9a7eb}}</style></head>
<body><p><a href="index.html">← All characters</a></p>
<h1>{_h.escape(data.get('full_name', char_key))} <small>{_h.escape(str(data.get('age', '')))}</small></h1>
<p class="role">{_h.escape(str(data.get('role', '')))}</p>
<h2>Character card</h2>{_card_html}
<h2>Views</h2><div class="views">{_views or '<p><i>No card yet.</i></p>'}</div>
<h2>Voice</h2>{_audio}
<h2>Prompt tags <button onclick="navigator.clipboard.writeText(document.getElementById('pt').innerText)">Copy</button></h2>
<pre id="pt">{_tags}</pre>
<h2>Negative tags</h2><pre>{_neg}</pre>
<h2>Bible</h2><table>{_rows}</table>
<p><a href="index.html">← All characters</a></p></body></html>"""
    out.write_text(page, encoding="utf-8")

    _cards = []
    for ck, cd in chars.items():
        _cards.append(
            f'<a href="{ck}.html" style="text-decoration:none;color:inherit">'
            f"<div style='border:1px solid #3a3348;border-radius:8px;padding:12px'>"
            f"<h3>{_h.escape(cd.get('full_name', ck))}</h3>"
            f"<p class='role'>{_h.escape(str(cd.get('role', '')))}</p></div></a>")
    (pages / "index.html").write_text(
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Characters</title><style>:root{color-scheme:dark}body{background:#141118;color:#e8e2f0;"
        "font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:24px}"
        ".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.role{color:#c9a7eb}</style></head>"
        f"<body><h1>{_h.escape(_novel_title())} — Characters</h1><div class='grid'>"
        + "".join(_cards) + "</div></body></html>", encoding="utf-8")
    return out

# --- ComfyUI Paths + H3 Video Generator ----------------------------
def _model_roots():
    """Known ComfyUI model roots (weights live here, not in input staging)."""
    return [Path(_home("AppData", "Local", "Comfy-Desktop", "ComfyUI-Shared", "models")),
            Path(_home("AppData", "Local", "Comfy-Desktop", "ComfyUI-Installs", "Example Novel", "ComfyUI", "models"))]


def _comfy_dirs(kind):
    """All known ComfyUI input/output roots (backend moves between launches).

    ponytail: explicit list, not probing — ComfyUI exposes no dir-query API.
    """
    seen, out = set(), []
    for key in (f"comfyui_{kind}", f"comfyui_{kind}_fallback"):
        d = CONFIG.get(key)
        if d and d not in seen:
            seen.add(d)
            out.append(Path(d))
    base = Path(_home("AppData", "Local", "Comfy-Desktop"))
    for extra in [base / "ComfyUI-Shared" / kind,
                  base / "ComfyUI-Installs" / "Example Novel" / "ComfyUI" / kind,
                  base / "ComfyUI-Installs" / "ComfyUI" / "ComfyUI" / kind]:
        if str(extra) not in seen and extra.exists():
            seen.add(str(extra))
            out.append(extra)
    return out


def find_comfy_file(filename):
    """Locate a ComfyUI output file across all known output dirs."""
    if not filename:
        return None
    for d in _comfy_dirs("output"):
        p = d / filename
        if p.exists():
            return p
    return None
def stage_comfy_input(src_path):
    src_path = Path(src_path)
    digest = _file_fingerprint(src_path)["sha256"]
    name = f"{src_path.stem}_{digest}{src_path.suffix}"
    inputs = _comfy_dirs("input")
    if not inputs:
        raise RuntimeError("No ComfyUI input directory configured")
    for input_dir in inputs:
        input_dir.mkdir(parents=True, exist_ok=True)
        dst = input_dir / name
        if not dst.is_file() or _file_fingerprint(dst)["sha256"] != digest:
            shutil.copy2(src_path, dst)
        if _file_fingerprint(dst)["sha256"] != digest:
            raise RuntimeError(f"ComfyUI input verification failed: {dst}")
    return name



# --- Video Generator (MiniMax H3 Ref2VA pruned-int8, Tomw852 recipe) --
def h3_dims(quality=None):
    """H3 dims per tier (divisible by 32). Fast 540p draft, final 720p."""
    q = (quality or CONFIG.get("quality", "fast")).lower()
    if q == "final":
        return 1280, 720
    return 960, 544


def _h3_cfg(quality: str) -> dict:
    """Resolve H3 config for quality tier, with fast defaults.

    The engine settings now follow Comfy-Org's `video_minimax_h3_r2v`
    template, which was run unmodified and DID preserve the reference still
    (see _h3_lora_chain). What changed versus the old values: the int8
    video VAE, 20 steps, and the text encoder on the GPU rather than the CPU.

    Resolution is NOT raised: the official template runs at 0.4 MP (864x480),
    SMALLER than the 960x544 this used, and it worked there — so canvas size
    was never the cause.
    """
    q = (quality or "fast").lower()
    if q == "final":
        return {
            "checkpoint": CONFIG.get("h3_final_checkpoint", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
            "text_encoder": CONFIG.get("h3_text_encoder", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
            "vae": CONFIG.get("h3_vae", "minimax_h3_video_vae_int8_convrot.safetensors"),
            "audio_vae": CONFIG.get("h3_audio_vae", "minimax_h3_audio_vae_fp32.safetensors"),
            "steps": int(CONFIG.get("h3_final_steps", 20)),
            "frames": int(CONFIG.get("h3_final_frames", 243)),
            "fps": int(CONFIG.get("h3_final_fps", 24)),
            "width": 1280, "height": 720,
            "sampler": str(CONFIG.get("h3_sampler", "res_multistep")),
            "scheduler": str(CONFIG.get("h3_scheduler", "simple")),
            "encoder_device": str(CONFIG.get("h3_encoder_device", "default")),
            "turbo_lora": CONFIG.get("h3_turbo_lora", ""),
        }
    return {
        "checkpoint": CONFIG.get("h3_checkpoint", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
        "text_encoder": CONFIG.get("h3_text_encoder", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
        "vae": CONFIG.get("h3_vae", "minimax_h3_video_vae_int8_convrot.safetensors"),
        "audio_vae": CONFIG.get("h3_audio_vae", "minimax_h3_audio_vae_fp32.safetensors"),
        "steps": int(CONFIG.get("h3_steps", 20)),
        "frames": int(CONFIG.get("h3_frames", 124)),
        "fps": int(CONFIG.get("h3_fps", 24)),
        "width": 960, "height": 544,
        "sampler": str(CONFIG.get("h3_sampler", "res_multistep")),
        "scheduler": str(CONFIG.get("h3_scheduler", "simple")),
        "encoder_device": str(CONFIG.get("h3_encoder_device", "default")),
        "turbo_lora": CONFIG.get("h3_turbo_lora", ""),
    }


def _h3_lora_chain(workflow, model_ref, clip_ref, start_id=20, tier="fast"):
    """Attach the official Ref2VA turbo LoRA, model-only.

    REPLACED a 3-deep chain (PDD Acc-8Step + a photoreal-skin LoRA + a camera
    LoRA) loaded through `LoraLoader`, which patched the 32B Qwen3-VL text
    encoder as well as the DiT. That chain was measured NOT to preserve the
    reference image: given a correct still of a man before a burning mill it
    produced a mangled doll face in frame 1, and the result was invariant to
    LoRA subsets, LoRA strength, step count, sampler chain, encoder device
    and reference size.

    The official `video_minimax_h3_r2v` template, run unmodified with its own
    weights, followed the same still correctly. The only substantive
    difference left was this LoRA: Comfy-Org ships a 4-step turbo LoRA built
    for reference-to-video specifically (`minimax_h3_ref2v_turbo_4step_*`),
    loaded model-only so the text encoder is never touched.

    ponytail: if the turbo LoRA is absent we run with NO LoRA rather than
    falling back to the old chain — the old chain is what produced broken
    clips, so a silent fallback would reintroduce the bug.
    """
    clip_ref = clip_ref  # model-only: the clip encoder is never patched
    name = CONFIG.get("h3_turbo_lora",
                      "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors")
    weight = float(CONFIG.get("h3_turbo_strength", 1.0))
    if name and any((r / "loras" / name).exists() for r in _model_roots()):
        workflow[str(start_id)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_ref, "lora_name": name,
                       "strength_model": weight}}
        log(f"  H3 turbo LoRA: {name} @ {weight}", 2)
        return workflow, [str(start_id), 0], clip_ref
    log(f"  [WARN] H3 turbo LoRA missing (loras/{name}) — running without it",
        1)
    return workflow, model_ref, clip_ref


def _extract_last_frame(clip_path, dest):
    """Pull the final frame of a clip to a PNG. Returns dest or None.

    The continuity relay feeds this into the next scene as a second
    reference, so the encoder sees the character as the *previous* shot
    actually rendered her, not just as some approved still.
    """
    dest = Path(dest)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [CONFIG["ffmpeg"], "-y", "-v", "error", "-sseof", "-0.5",
             "-i", str(clip_path), "-frames:v", "1", "-q:v", "2", str(dest)],
            capture_output=True, timeout=60)
        if proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
            return dest
        dest.unlink(missing_ok=True)
    except Exception as e:
        log(f"  [WARN] last-frame extract failed for {Path(clip_path).name}: "
            f"{str(e)[:80]}", 2)
    return None


def _h3_references(image_path, relay_frame=None):
    """Build the ref_images dict + the per-reference role sentences.

    MiniMax H3's ReferenceToVideo node takes up to 9 reference images in a
    single `ref_images` dict, and the prompt addresses them POSITIONALLY
    (<Picture 1>, <Picture 2>, ...). The node's own guidance is explicit:
    "assign each reference a job — state which reference drives which part
    of the shot. Explicit assignments tend to work much better."

    So the relay is: <Picture 1> is this shot's keyframe (composition +
    identity anchor); <Picture 2> is the previous shot's final frame, whose
    face / hair / wardrobe / lighting must carry forward. Without a job
    description the encoder has no idea which reference means what.
    """
    refs = {"ref_image_1": ["4", 0]}
    roles = ("<Picture 1> is this shot's keyframe: it sets the composition, "
             "costume and identity for the whole clip.")
    if relay_frame:
        # Node "3", NOT "5" — "5" is the KSampler and the workflow.update()
        # below would silently overwrite a LoadImage parked there, leaving
        # ref_image_2 pointing at a sampler output.
        refs["ref_image_2"] = ["3", 0]
        roles += (" <Picture 2> is the final frame of the previous shot — "
                  "continue from it: the same face, hair, wardrobe, lighting "
                  "and environment, no new character.")
    return refs, roles


def _h3_prompt(scene):
    """The ref2v prompt for one shot: scene description FIRST, then motion.

    The ref2v prompt is a FULL video description, not just a motion note.
    With motion words alone the model has no subject to hold and invents
    one: measured on a correct still of a man before a burning mill, a
    motion-only prompt produced an anime woman in a white dress indoors,
    while this prompt (the same motion text prefixed with the scene's own
    image_prompt) produced the man and the mill.

    Comfy-Org's video_minimax_h3_r2v template does the same — it describes
    its subject as well as the motion. This is the single decisive fix for
    clips that ignored their still.
    """
    anim_hint = scene.get("animation_hint", "")
    cam_move = scene.get("camera_move", "")
    move_txt = {"push-in": "slow push in toward subject",
                "pull-out": "slow pull out revealing surroundings",
                "pan-left": "smooth pan left across the scene",
                "pan-right": "smooth pan right across the scene",
                "tracking": "tracking shot following subject movement",
                "handheld": "subtle handheld sway, nervous tension"}.get(
                    cam_move, anim_hint)
    ambient = "hair and clothing sway gently, light flickers subtly, dust drifts"
    if scene.get("lock_face", True):
        still = ("subtle natural motion, gentle breathing, slight head shifts, "
                 "natural blinking, same face, no morphing, ")
    else:
        still = "natural character motion, fluid movement, "
    motion = (f"subject from <Picture 1>, {move_txt}, {ambient}, {still}"
              "smooth controlled motion, natural physics, realistic weight and "
              "momentum, follow-through motion, cloth and hair respond to "
              "movement and air, feet planted when standing, no sliding, "
              f"steady pace, {_style_spec(scene)['motion']}")
    desc = str(scene.get("image_prompt", "") or "").strip().rstrip(".")
    # Identity has to be named too. The scene sentence alone ("a mill on fire
    # at night, wide establishing shot") says nothing about WHO is in it, so
    # the model invents a person: measured, the man in a dark coat became a
    # girl in a white dress while the mill stayed correct. Reuse the same
    # bible tags the still builder injects so the two agree on the face.
    ident = []
    for char_name in scene.get("characters", []) or []:
        key = str(char_name).lower().replace(" ", "_")
        cd = (CHAR_BIBLE or {}).get("characters", {}).get(key, {})
        tags = cd.get("danbooru_tags") or cd.get("prompt_tags") or ""
        if not tags:
            bits = [cd.get(k, "") for k in
                    ("hair", "eyes", "clothing", "distinguishing_features")]
            tags = ", ".join(b for b in bits if b)
        if tags:
            ident.append(str(tags).strip())
    _sheet_id = _sheet_identity_phrase(scene)
    if _sheet_id:
        ident.append(_sheet_id)
    cine = _cine_phrase(scene)
    lead = ", ".join(x for x in [desc, cine] + ident if x)
    prompt = f"{lead}, {motion}" if lead else motion
    return extend_motion_prompt(prompt, scene)


def generate_h3_clip(image_path, scene, chapter_num, width=None, height=None, length=None,
                     seed_offset=0, relay_frame=None):
    """Generate H3 Ref2VA clip with quality tier + OOM fallback."""
    # ponytail: remember whether the caller pinned the dims. The old
    # fallback re-called _try_h3_clip with the SAME width/height, so an OOM
    # at 1280x720 dropped steps and frames but kept the resolution that
    # caused the OOM — the fallback could not actually recover.
    explicit_dims = width is not None and height is not None
    if not explicit_dims:
        width, height = h3_dims()
    quality = CONFIG.get("quality", "fast")
    fallback = CONFIG.get("_yaml_fallback", True)
    try:
        return _try_h3_clip(image_path, scene, chapter_num, quality, width, height,
                            length, seed_offset, relay_frame)
    except (RuntimeError, MemoryError, ConnectionError) as e:
        if fallback and quality == "final":
            fw, fh = (width, height) if explicit_dims else h3_dims("fast")
            log(f"  [H3] OOM/error on final tier: {e} — falling back to fast "
                f"({fw}x{fh})")
            return _try_h3_clip(image_path, scene, chapter_num, "fast", fw, fh,
                                length, seed_offset, relay_frame)
        raise


def _try_h3_clip(image_path, scene, chapter_num, quality, width, height, length,
                 seed_offset=0, relay_frame=None):
    """Internal: try H3 Ref2VA clip. Returns filename or raises.

    Graph: UNETLoader(int8) + turbo LoRA (model-only) ->
    MiniMaxH3ReferenceToVideo(this shot's keyframe as <Picture 1>, plus the
    previous shot's last frame as <Picture 2> when a relay frame is staged) ->
    the official sampler chain (RandomNoise/BasicGuider/KSamplerSelect/
    BasicScheduler/SamplerCustomAdvanced, res_multistep+simple, 20 steps) ->
    VAEDecode + VAEDecodeAudio -> CreateVideo -> SaveVideo. H3 also generates
    audio, but the episode voice is XTTS, so the audio track is not used.
    """
    cfg = _h3_cfg(quality)
    if length is None:
        length = cfg["frames"]
    # ponytail: H3 length grid is 17k+5 (124/243). Snap off-grid values
    # instead of letting ComfyUI reject the prompt mid-render.
    #
    # UP, not to nearest. length is the narration slot the clip has to COVER,
    # and rounding to nearest made the clip shorter than the audio on 16 of 29
    # real ch001/ch900 scenes (e.g. a 2.67s slot -> 2.33s), which sends the
    # scene to assembly's -stream_loop and puts a restart seam in the middle of
    # the line. Rounding up costs at most one grid step (17 frames = 0.7s) of
    # tail that assembly trims away silently, which nobody sees.
    length = 17 * max(1, math.ceil((int(length) - 5) / 17)) + 5
    ckpt = cfg["checkpoint"]
    t5 = cfg["text_encoder"]
    vae = cfg["vae"]
    audio_vae = cfg.get("audio_vae") or ""
    h3_steps = int(scene.get("h3_steps", cfg["steps"]))
    h3_fps = cfg["fps"]
    # device: official template runs the 32B Qwen3-VL encoder on the GPU
    # ("default"), not the CPU. On 16GB it fits alongside the int8 DiT.
    enc_device = str(CONFIG.get("h3_encoder_device", "default"))
    for _name, _sub in ((ckpt, "diffusion_models"), (t5, "text_encoders"),
                        (vae, "vae"), (audio_vae, "vae")):
        if _name and not any((r / _sub / _name).exists() for r in _model_roots()):
            raise FileNotFoundError(f"H3 model missing: {_sub}/{_name} — fix config or install file, fallback is parallax")
    scene_num = scene.get("scene_number", 1)
    seed = 9000 + chapter_num * 100 + scene_num + seed_offset
    prefix = f"ch{chapter_num:03d}_h3{scene_num:02d}_{seed}_{uuid.uuid4().hex}"
    # ponytail: the old default "slow cinematic push in" fired on every scene
    # with no animation_hint, and H3 read it literally — the rain-dance test
    # ended on a face close-up instead of a push-in. _h3_prompt defaults to
    # NO camera instruction; an explicit scene["camera_move"] still wins.
    prompt_text = _h3_prompt(scene)

    workflow = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": ckpt, "weight_dtype": "default"}},
        "0": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": t5, "type": "minimax", "device": enc_device}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_path}},
    }
    if audio_vae:
        workflow["8"] = {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae}}
    workflow, _mref, _cref = _h3_lora_chain(workflow, ["1", 0], ["0", 0],
                                            tier=(quality or "fast"))
    # Continuity relay: node "3" loads the previous shot's final frame as
    # <Picture 2>. Only wired when a relay frame was actually staged.
    # ponytail: "3" is the free slot — "5" is the sampler and workflow.update()
    # below would clobber a LoadImage parked there.
    if relay_frame:
        workflow["3"] = {"class_type": "LoadImage",
                         "inputs": {"image": str(relay_frame)}}
    _refs, _roles = _h3_references(image_path, relay_frame)
    _h3_inputs = {"clip": _cref, "vae": ["7", 0],
                  "prompt": f"{_roles} {prompt_text}",
                  "width": width, "height": height, "length": length,
                  # autogrow slot is a dict (ref_images), not a flat
                  # ref_image_1 kwarg (live-verified). "match" is the
                  # official value.
                  "ref_image_size": "match",
                  "ref_images": _refs}
    if audio_vae:
        _h3_inputs["audio_vae"] = ["8", 0]
    workflow["2"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": _h3_inputs}
    # Official sampler chain (RandomNoise -> BasicGuider -> KSamplerSelect ->
    # BasicScheduler -> SamplerCustomAdvanced). PDD/turbo are guidance-free,
    # so the single conditioning output feeds the guider.
    _sampler = str(CONFIG.get("h3_sampler", "res_multistep"))
    _scheduler = str(CONFIG.get("h3_scheduler", "simple"))
    workflow.update({
        "40": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "41": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": _sampler}},
        "42": {"class_type": "BasicScheduler",
               "inputs": {"model": _mref, "scheduler": _scheduler,
                          "steps": h3_steps, "denoise": 1.0}},
        "43": {"class_type": "BasicGuider",
               "inputs": {"model": _mref, "conditioning": ["2", 0]}},
        "44": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["40", 0], "guider": ["43", 0],
                          "sampler": ["41", 0], "sigmas": ["42", 0],
                          "latent_image": ["2", 1]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["44", 0], "vae": ["7", 0]}},
    })
    if audio_vae:
        workflow["9"] = {"class_type": "VAEDecodeAudio",
                         "inputs": {"samples": ["44", 0], "vae": ["8", 0]}}
    workflow["11"] = {"class_type": "CreateVideo",
                      "inputs": {"images": ["6", 0], "fps": float(h3_fps)}}
    if audio_vae:
        workflow["11"]["inputs"]["audio"] = ["9", 0]
    workflow["12"] = {"class_type": "SaveVideo",
                      "inputs": {"video": ["11", 0], "filename_prefix": prefix,
                                 "format": "mp4", "codec": "h264"}}
    result = http_post_json(f"{CONFIG['comfyui_url']}/prompt",
                            {"prompt": workflow, "client_id": str(uuid.uuid4())}, timeout=30)
    prompt_id = result["prompt_id"]
    return _comfy_output_filename(_comfy_wait(prompt_id, 240, 5, "h3"))


def extend_motion_prompt(base_prompt, scene, still_path=None):
    """P2: LLM motion-prompt extension (motion+camera+env+speed).

    Tries the text LLM when up; falls back to base prompt verbatim.
    Never blocks renders — extension is best-effort, base already 4-part.
    """
    if not bool(CONFIG.get("motion_prompt_extend", False)):
        return base_prompt
    try:
        txt = llm_chat(
            [{"role": "user",
              "content": f"Expand this anime motion prompt with concrete physical detail (weight, cloth/hair response, pace word like slowly/quickly): {base_prompt}"}],
            timeout=60, max_tokens=200).strip()
        return txt if len(txt) > 20 else base_prompt
    except Exception:
        return base_prompt


def _echo_frames(scene):
    return min(5000, max(12, math.ceil(float(scene.get("_dur") or CONFIG["scene_duration"])
                                       * int(CONFIG.get("echo_fps", 24)))))


def _echo_fingerprint(still, audio, scene):
    return {"text": clean_tts_text(scene.get("narration") or scene.get("dialogue") or ""),
            "still": _file_fingerprint(still), "audio": _file_fingerprint(audio),
            "lead": float(scene.get("_lead", 0.2)), "duration": float(scene["_dur"]),
            # ponytail: shot_size and mood build the EchoMimic prompt (see
            # generate_echomimic_clip: "Anime {shot} portrait, ... {mood}
            # mood") but were absent from this key, so re-shooting a scene at a
            # different shot size replayed the cached lips from the old one.
            # Anything else the workflow hardcodes belongs here too.
            "prompt": {"shot": _shot_norm(scene) or "close-up",
                           "mood": str(scene.get("mood") or "neutral"),
                       "negative_prompt": str(scene.get("negative_prompt") or "")},
            "config": {"engine": str(CONFIG.get("lipsync_engine", "echo")).lower(),
                       "version": str(CONFIG.get("echo_version", "V1")),
                       "size": int(CONFIG.get("echo_size", 512)),
                        # must track echo_fps — frames_to_clip uses it and a
                        # hardcoded 24 here meant changing echo_fps never
                        # invalidated a cached lip clip.
                        "fps": int(CONFIG.get("echo_fps", 24)),
                        "steps": int(CONFIG.get("echo_steps", 6)),
                        "cfg": float(CONFIG.get("echo_cfg", 2.5)), "frames": _echo_frames(scene),
                        "mode": str(CONFIG.get("echo_infer_mode", "audio_drived_acc")),
                        "vae": CONFIG.get("echo_vae", "sd-vae-ft-mse.safetensors")}}


def _stage_echo_audio(audio, scene):
    temp = _asset("_temp")
    temp.mkdir(exist_ok=True)
    padded = temp / f"{Path(audio).stem}.{uuid.uuid4().hex}.echo.wav"
    try:
        lead = float(scene.get("_lead", 0.2))
        dur = float(scene["_dur"])
        proc = subprocess.run(
            [CONFIG["ffmpeg"], "-y", "-v", "error", "-i", str(audio), "-af",
             f"adelay={round(lead * 1000)}:all=1,apad=whole_dur={dur}",
             "-t", str(dur), "-c:a", "pcm_s16le", str(padded)],
            capture_output=True, timeout=60)
        if proc.returncode or not padded.is_file():
            raise RuntimeError(f"Echo audio padding failed: {proc.stderr!r}")
        return stage_comfy_input(padded)
    finally:
        padded.unlink(missing_ok=True)


_ECHO_NODES = ("Echo_LoadModel", "Echo_Predata", "Echo_Sampler")
_echo_node_cache = {}


def _echo_nodes_available():
    """True when the EchoMimic pack is actually installed. Probed once.

    ponytail: the pack was never installed, so every talking-head scene built
    a full workflow and POSTed it only for ComfyUI to answer HTTP 400
    missing_node_type -- 12 wasted round-trips and 12 identical warnings per
    chapter, all of them noise that buried the real warnings. Asking
    /object_info once turns that into a single line at the point it matters,
    and makes the fallback a stated decision instead of a mystery.
    """
    if "ok" in _echo_node_cache:
        return _echo_node_cache["ok"]
    try:
        info = http_get(f"{CONFIG['comfyui_url']}/object_info", timeout=60)
        missing = [n for n in _ECHO_NODES if n not in info]
        ok = not missing
        if not ok:
            log("  [WARN] lip-sync disabled: EchoMimic pack not installed in "
                f"ComfyUI (missing {', '.join(missing)}). Talking-head scenes "
                "will keep the still clip. Install "
                "smthemex/ComfyUI_EchoMimic into custom_nodes to enable it.", 1)
    except Exception as e:
        # NOT cached: a timeout on a multi-MB /object_info while ComfyUI is
        # mid-render is transient. Caching it disabled lip-sync for the rest
        # of the process -- and across chapters inside pipeline_server.
        log(f"  [WARN] lip-sync node probe failed ({str(e)[:70]}) - not "
            f"cached, will re-probe", 1)
        return False
    _echo_node_cache["ok"] = ok
    return ok


def generate_echomimic_clip(still_name, wav_name, scene, chapter_num):
    """Talking-head clip: EchoMimic V1 animates a close-up still with dialogue audio.

    still_name/wav_name are ComfyUI input-dir filenames (see stage_comfy_input).
    Returns the output filename in the ComfyUI output dir (SaveImage sequence),
    or None — every failure returns None so the caller keeps the existing clip.
    """
    if not CONFIG.get("echo_enabled", True):
        return None
    scene_num = scene.get("scene_number", 1)
    size = int(CONFIG.get("echo_size", 512))
    frames = _echo_frames(scene)
    seed = 7000 + chapter_num * 100 + scene_num
    # P1: PID in prefix like H3 — without run isolation find_comfy_file
    # serves stale lips from previous days.
    prefix = f"ch{chapter_num:03d}_echo{scene_num:02d}_{seed}_{uuid.uuid4().hex}"
    mood = scene.get("mood", "neutral")
    shot = _shot_norm(scene) or "close-up"
    prompt_text = (f"Anime {shot} portrait, character speaking with natural "
                   f"lip movement, subtle head motion, {mood} mood")
    negative = ("blurry, low quality, deformed, bad anatomy, photorealistic, "
                "3d render, watermark, text")
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": still_name}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": wav_name}},
        "3": {"class_type": "Echo_LoadModel",
              "inputs": {"vae": CONFIG.get("echo_vae", "sd-vae-ft-mse.safetensors"),
                         "lora": "None", "denoising": True,
                         "infer_mode": CONFIG.get("echo_infer_mode", "audio_drived_acc"),
                         "lowvram": True, "teacache_offload": True,
                          "block_offload": True, "use_mmgp": "None",
                          # m6: version from config (was hardcoded V1, default dead).
                          "version": str(CONFIG.get("echo_version", "V1"))}},
        "4": {"class_type": "Echo_Predata",
              "inputs": {"info": ["3", 1], "image": ["1", 0], "audio": ["2", 0],
                         "prompt": prompt_text, "negative_prompt": negative,
                         "pose_dir": "pose_01", "width": size, "height": size,
                          "fps": float(CONFIG.get("echo_fps", 24)), "facemask_ratio": 0.1,
                         "facecrop_ratio": 0.8, "length": frames,
                         "partial_video_length": 33, "draw_mouse": False,
                         "motion_sync_": False}},
        "5": {"class_type": "Echo_Sampler",
              "inputs": {"model": ["3", 0], "emb": ["4", 0], "seed": seed,
                         "cfg": float(CONFIG.get("echo_cfg", 2.5)),
                         "steps": int(CONFIG.get("echo_steps", 6)),
                         "sample_rate": 16000, "context_frames": 12,
                         "context_overlap": 3, "save_video": False}},
        "6": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": prefix, "images": ["5", 0]}},
    }
    try:
        result = http_post_json(f"{CONFIG['comfyui_url']}/prompt",
                                {"prompt": workflow, "client_id": str(uuid.uuid4())}, timeout=30)
        prompt_id = result["prompt_id"]
    except ComfyUIError as e:
        # The pack being absent is now caught up front by
        # _echo_nodes_available(); reaching here means it IS installed and
        # something else is wrong, so say which.
        log(f"  [WARN] EchoMimic rejected ({str(e)[:120]})")
        return None
    try:
        return _comfy_output_filename(_comfy_wait(prompt_id, 120, 5, "echo"))
    except ComfyUIError as e:
        # lipsync is a nice-to-have — the still clip is kept instead.
        log(f"  [WARN] EchoMimic poll failed ({str(e)[:120]})")
        return None


def generate_lipsync_clip(still_name, wav_name, scene, chapter_num):
    """P2 dispatcher: lipsync_engine echo|latentsync|s2v (default echo V1).

    - echo: proven EchoMimic V1 path (face>=25%, echofail markers).
    - latentsync: pilot — needs ComfyUI-LatentSync nodes + 512px model;
      falls back to echo until nodes present (re-dub, no pose input, 4-8GB).
    - s2v: pilot — needs Wan2.2-S2V-14B (~30GB disk, 480p fp8+offload trial,
      Wan-family, NOT staged — pipeline is Wan-free since 2026-09-25);
      falls back to echo until weights present. S2V collapses motion+lips
      into one audio-driven pass when it lands.
    """
    engine = str(CONFIG.get("lipsync_engine", "echo")).lower()
    if engine == "echo" and not _echo_nodes_available():
        return None
    if engine == "latentsync":
        log("  [PILOT] latentsync selected but nodes not wired — falling back to echo V1", 2)
    elif engine == "s2v":
        log("  [PILOT] s2v selected but 14B weights not staged — falling back to echo V1", 2)
    elif engine not in ("echo",):
        log(f"  [WARN] unknown lipsync_engine={engine} — using echo", 2)
    return generate_echomimic_clip(still_name, wav_name, scene, chapter_num)


def _tool_exe(*parts):
    p = ROOT / "_models" / Path(*parts)
    return str(p) if p.exists() else None


def cugan_frames(seqdir, scale=2, jobs=2):
    """Real-CUGAN 2x anime upscale, in place. Skips silently if missing."""
    exe = _tool_exe("realcugan", "realcugan-ncnn-vulkan-20220728-windows",
                    "realcugan-ncnn-vulkan.exe")
    if not exe or not bool(CONFIG.get("cugan_enabled", True)):
        return False
    import concurrent.futures as _cf
    frames = sorted(Path(seqdir).glob("f*.png"))
    if not frames:
        return False
    tmp = Path(seqdir) / "_up"
    tmp.mkdir(exist_ok=True)

    def _one(fr):
        out = tmp / fr.name
        r = subprocess.run([exe, "-i", str(fr), "-o", str(out),
                            "-s", str(scale), "-m", "models-se", "-n", "1"],
                           capture_output=True, timeout=300)
        return r.returncode == 0 and out.exists()

    try:
        with _cf.ThreadPoolExecutor(max_workers=max(1, int(jobs))) as ex:
            done = sum(1 for ok in ex.map(_one, frames) if ok)
    except Exception as e:
        log(f"  [WARN] CUGAN failed ({e}), Lanczos fallback", 2)
        return False
    if done != len(frames):
        log(f"  [WARN] CUGAN {done}/{len(frames)}, Lanczos fallback", 2)
        return False
    for fr in frames:
        (tmp / fr.name).replace(fr)
    shutil.rmtree(tmp, ignore_errors=True)
    return True


def rife_double(seqdir):
    """RIFE 2x interpolation of a PNG sequence, in place. Skips if missing."""
    exe = _tool_exe("rife", "rife-ncnn-vulkan-20221029-windows",
                    "rife-ncnn-vulkan.exe")
    if not exe or not bool(CONFIG.get("rife_enabled", True)):
        return False
    tmp = Path(seqdir) / "_2x"
    tmp.mkdir(exist_ok=True)
    r = subprocess.run([exe, "-i", str(seqdir), "-o", str(tmp),
                        "-m", "rife-anime"], capture_output=True, timeout=900)
    outs = sorted(tmp.glob("*.png"))
    if r.returncode != 0 or not outs:
        log("  [WARN] RIFE failed, frame-double fallback", 2)
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    for fr in Path(seqdir).glob("f*.png"):
        try:
            fr.unlink()
        except Exception:
            pass
    for i, fr in enumerate(outs, 1):
        fr.rename(Path(seqdir) / f"f{i:05d}.png")
    shutil.rmtree(tmp, ignore_errors=True)
    return True


def frames_to_clip(src_file, clip_path, fps=None, max_sec=None):
    """Assemble a SaveImage frame sequence (EchoMimic stills) into an mp4 clip.

    CUGAN 2x sharpens anime lines/faces before encode; RIFE doubles
    sub-20fps sequences to real interpolated frames (not duplicated).
    max_sec trims to the scene slot first so no wasted enhance work.
    """
    from pathlib import Path as _P
    if fps is None:
        fps = int(CONFIG.get("echo_fps", 24))
    frames = clip_frames(_P(src_file))
    if not frames:
        return False
    if max_sec:
        frames = frames[:max(1, int(len(frames) * min(1.0, max_sec / (len(frames) / max(1, fps)))))]
    work = _P(str(clip_path) + "_frames")
    work.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames, 1):
        try:
            shutil.copy2(fr, work / f"f{i:05d}.png")
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            return False
    cugan_frames(work)  # ponytail: silent no-op when exe/flag missing
    out_fps = fps
    # The old gate compared fps against the literal 20 — but `fps` is the
    # OUTPUT rate we hand the encoder (echo_fps, 24), so that branch was
    # never true and RIFE never ran once.
    #
    # The honest question for a frame SEQUENCE is: did the engine hand back
    # enough frames to cover the slot at the timeline rate? If it returned
    # fewer than max_sec*fps frames, the sequence really is low-fps and
    # interpolating synthesises new motion rather than duplicating frames.
    # (An earlier version computed a "native_fps" from the already-trimmed
    # list, which was self-referentially always ~fps and never fired.)
    if max_sec and fps > 0:
        needed = max(1, int(round(float(max_sec) * fps)))
        delivered = len(frames)
        if delivered < needed * 0.95 and rife_double(work):
            out_fps = fps
            log(f"  frames_to_clip: engine returned {delivered} frames for a "
                f"{needed}-frame slot — RIFE interpolated to {out_fps}fps", 2)
    cmd = [CONFIG["ffmpeg"], "-y", "-framerate", str(out_fps), "-start_number", "1",
           "-i", str(work / "f%05d.png"),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", str(clip_path)]
    ok = subprocess.run(cmd, capture_output=True, timeout=300).returncode == 0
    shutil.rmtree(work, ignore_errors=True)
    return ok and Path(clip_path).exists()

# --- Clip Grader (VLM-gate proxy: faces + drift + sharpness) --------
_FACE_MODEL = ROOT / "_models" / "face_detection_yunet_2023mar.onnx"
_face_detector = None
_face_detector_lock = threading.Lock()


def _get_face_detector():
    global _face_detector
    if _face_detector is not None:
        return _face_detector
    # ponytail: the construction MUST be inside the lock. It used to sit
    # after the `with` block, so two threads both built a detector.
    with _face_detector_lock:
        if _face_detector is not None:
            return _face_detector
        try:
            import cv2
            if _FACE_MODEL.exists():
                _face_detector = cv2.FaceDetectorYN_create(str(_FACE_MODEL), "", (320, 320))
        except Exception as e:
            log(f"  [WARN] Face detector unavailable: {e}")
        return _face_detector


def _count_faces(img_path):
    det = _get_face_detector()
    if det is None:
        return -1
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return -1
        h, w = img.shape[:2]
        # FaceDetectorYN carries per-image input size as mutable state, so
        # setInputSize + detect must be atomic against other threads.
        with _face_detector_lock:
            det.setInputSize((w, h))
            _, faces = det.detect(img)
        return 0 if faces is None else len(faces)
    except Exception:
        return -1


def still_drift(clip_frame, still):
    """How far a clip's first frame has drifted from the still it came from.

    1.0 = identical structure, 0.0 = unrelated. This is the check
    grade_clip() structurally cannot do: it compares a clip's first and last
    frames, so a model that invents a DIFFERENT subject and then stays
    consistent with itself scores perfectly. A real ch900 render turned a wide
    night mill into a mangled doll-face close-up and every internal check
    passed.

    ponytail: NOT calibrated as a gate. Measured on 3 real clips the clean
    ones scored 0.084 and the broken one 0.366, so a single spatial threshold
    does not separate them and any number I picked would be fitted to noise.
    Recorded as a receipt fact and escalated by the VLM (which can actually
    judge) instead. Promote to a hard gate once enough real clips exist to
    fit a threshold on.
    """
    try:
        import numpy as _np
        from PIL import Image
        with Image.open(clip_frame) as im:
            a = _np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR),
                            dtype=_np.float32)
        with Image.open(still) as im:
            b = _np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR),
                            dtype=_np.float32)
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return round(float(_np.corrcoef(a.ravel(), b.ravel())[0, 1]), 3)
    except Exception:
        return None


def _vlm_clip_matches_still(clip_frame, still, scene):
    """Ask the vision model whether a clip still shows what the scene asked
    for. Returns (ok, note) or (None, '') when the VLM is down.

    The histogram and face heuristics cannot see a subject swap in a wide
    shot: there is no face to count, and a consistent hallucination keeps its
    own histogram. This is the only check that can.
    """
    if not vlm_available():
        return None, ""
    want = str(scene.get("image_prompt", ""))[:300]
    try:
        import base64 as _b64
        from json_util import oai_chat as _chat, oai_image_part as _part
        with open(clip_frame, "rb") as f:
            cb = _b64.b64encode(f.read()).decode()
        with open(still, "rb") as f:
            sb = _b64.b64encode(f.read()).decode()
        txt = _chat(_llm_base(), [{
            "role": "user",
            "content": [
                {"type": "text", "text":
                    f"The first image is the STILL the clip was animated from. "
                    f"The second image is the first frame of the clip itself. "
                    f"The clip should depict: {want!r}\n"
                    "Reply ONLY valid JSON: "
                    '{"matches": true|false, "subject": "short", '
                    '"deformed": true|false, "note": "short"}. '
                    "Report matches=false if the clip's subject, framing or "
                    "scene changed into something else, or if the first "
                    "frame is a different character or place than the still. "
                    "deformed=true if the face or anatomy is malformed "
                    "(blown-out eyes, melted features, asymmetric skull)."},
                _part(sb), _part(cb)]}],
            # Two 1024-token images is heavy vision prefill on a partial
            # offload; 120s timed out on a 5080. Runs once per clip and only
            # warns, so the wait is worth it.
            timeout=420, max_tokens=180)
        m = re.search(r"\{.*\}", txt or "", flags=re.DOTALL)
        if not m:
            return None, "unparseable"
        d = json.loads(m.group(0))
        return bool(d.get("matches", True)), str(d.get("note", ""))[:80]
    except Exception as e:
        return None, str(e)[:60]


def grade_clip(frames):
    """Grade a clip for hallucinated faces, subject jumps and blur collapse.

    Returns (passed: bool, reasons: list).

    ponytail: this compared ONLY the first and last frame. The last frame is
    the worst possible sample in a generated clip -- motion blur, a face
    turning away, the shot settling -- so a good take failed on
    "faces 1->0 (subject lost)", burned both retries, and was discarded for a
    still-pan anyway. That is ch001 scene 9: two full H3 generations, ~11
    minutes, zero output. Frames are already sampled at 1fps (~5 per clip),
    so grade against the MEDIAN of all of them: one odd frame out of five no
    longer convicts a clip, while a genuinely drifting take still fails.
    """
    reasons = []
    try:
        import cv2
        from statistics import median
        if isinstance(frames, (str, Path)):
            frames = [frames]
        frames = [Path(f) for f in frames][:8]
        imgs = [(f, cv2.imread(str(f))) for f in frames]
        imgs = [(f, im) for f, im in imgs if im is not None]
        if len(imgs) < 2:
            return False, ["unreadable frames"]

        def _sh(im):
            return cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),
                                 cv2.CV_64F).var()

        # ponytail: pick the REFERENCE frame by median sharpness, not by
        # position. Both surviving gates compare everything back to one frame,
        # and frame 0 is exactly as untrustworthy as the last one -- a fade-in
        # or near-black first frame silently disarms them: a degenerate
        # reference histogram makes OpenCV's HISTCMP_CORREL return 1.0 (always
        # passes) and a near-zero Laplacian variance makes every ratio huge
        # (blur gate never fires). That is a silent pass on a bad clip, which
        # is the failure this function exists to catch.
        _sharps = [_sh(im) for _, im in imgs]
        _ref_i = sorted(range(len(imgs)), key=lambda i: _sharps[i])[len(imgs) // 2]
        ref_path, first = imgs[_ref_i]
        rest = [im for i, (_, im) in enumerate(imgs) if i != _ref_i]
        rest_paths = [f for i, (f, _) in enumerate(imgs) if i != _ref_i]

        # 1. Face-count drift -- RECORDED, NOT a gate.
        # ponytail: this used to fail the clip, and it is not a usable signal
        # here. Measured on the real ch001 clips: _count_faces returns 0 on
        # 4 of 5 frames of clips that are perfectly good (h301, h311), and 1
        # on frame 0 of h302 with 0 afterwards. So one spurious detection
        # decides the verdict -- a coin flip, not a measurement. The codebase
        # already documents why: "No local detector can judge anime faces
        # (YuNet scores 0)", which is exactly why framing is gated on the VLM.
        # It rejected ch001 scene 9 and cost two H3 generations (~11 min) for
        # a clip that was fine. Kept as an observation for the receipt; the
        # histogram and sharpness checks below measure real pixels.
        _face_note = ""
        fa = _count_faces(ref_path)
        counts = [_count_faces(f) for f in rest_paths]
        counts = [c for c in counts if c >= 0]
        if fa >= 0 and counts:
            fm = median(counts)
            if fm > fa:
                _face_note = f"faces {fa}->{fm:.0f} (hallucinated?)"
            elif fa > 0 and fm == 0:
                _face_note = f"faces {fa}->0 (detector unreliable on anime)"
        if _face_note:
            log(f"  [clip-grade] {_face_note} (not a gate)", 2)

        # 2. Histogram correlation (subject/scene continuity).
        # Majority rule, NOT a median: one odd frame must not convict a clip
        # (that was ch001 scene 9, ~11 min wasted), but a clip whose frames
        # mostly disagree with each other must still fail. Taking the median
        # of the values was tried and was too lenient once the reference is
        # self-selected -- a drifting clip kept passing.
        def _h(im):
            return cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2HSV)], [0, 1],
                                None, [32, 32], [0, 180, 0, 256])
        ha = _h(first)
        corrs = [cv2.compareHist(ha, _h(im), cv2.HISTCMP_CORREL)
                 for im in rest]
        n_bad = sum(1 for c in corrs if c < 0.45)
        if n_bad * 2 > len(corrs):
            reasons.append(f"hist corr {median(corrs):.2f} "
                           f"({n_bad}/{len(corrs)} frames disagree)")

        # 3. Sharpness collapse. Two rules, because the two failure shapes are
        # different: gradual blur is a majority property (so a single dark
        # tail frame must not convict), but a frame with essentially NO
        # structure is a visible blank-frame defect on its own and must not be
        # outvoted by four good ones.
        sa = _sh(first)
        if sa > 0:
            ratios = [_sh(im) / sa for im in rest]
            blank = sum(1 for r in ratios if r < 0.02)
            if median(ratios) < 0.35:
                reasons.append(f"sharpness x{median(ratios):.2f} "
                               f"({sum(1 for r in ratios if r < 0.35)}"
                               f"/{len(ratios)} frames blurred)")
            elif blank:
                reasons.append(f"{blank} blank frame(s) (no image structure)")
    except Exception as e:
        return False, [f"grader error: {e}"]
    return (len(reasons) == 0), reasons

def clip_frames(src_file):
    """List all frames of a SaveImage sequence."""
    from pathlib import Path as _P
    src_file = _P(src_file)
    base = re.sub(r"_\d+_\.png$", "", src_file.name)
    return sorted(src_file.parent.glob(base + "_*.png"))

# --- TTS Engine (Qwen3-TTS clone server, single engine) -----------
def check_tts_server():
    """Is the TTS server up AND speaking the reference we think it is?

    A bare status+loaded check is not enough. cosy-server builds its clone prompt
    once at startup and never rebuilds it, so editing ref.json (new reference
    audio, a corrected transcript, a mode flip) leaves the process happily
    answering /health while still speaking the OLD voice. The fingerprint in the
    cache key changes, so the pipeline retires every cached wav, regenerates from
    the stale prompt, and writes a fresh fingerprint over audio that does not
    match the config it claims to have been made with — correct-looking files,
    wrong voice.

    /health now reports `stale`, so we can refuse instead. Refusing sends the
    caller down the restart path, which is the only thing that actually fixes it.
    """
    try:
        tts_url = CONFIG.get("tts_url", "http://localhost:5003")
        result = http_get(f"{tts_url}/health", timeout=5)
        if not (result and result.get("status") == "ok"
                and result.get("loaded", False)):
            return False
        if result.get("stale"):
            log("  [WARN] TTS server is running a stale voice reference "
                "(ref.json changed since it started); restarting it")
            _stop_tts_server()
            return False
        return True
    except Exception:
        return False


def _stop_tts_server():
    """Stop the TTS clone server to release its VRAM. Returns True if it is
    down afterwards.

    ponytail: H3 peaks at ~13.1GB of 16.3GB, and the Qwen3-TTS clone server
    holds ~3GB for the whole render while being needed only in Phase 5. With
    both resident, comfy-aimdo aborts on the failed allocation and the whole
    clip phase degrades to still-pans. Only the process that owns the
    configured TTS port is touched — never a bare "kill python".
    """
    url = str(CONFIG.get("tts_url", "http://localhost:5003")).strip()
    port = url.rsplit(":", 1)[-1].split("/")[0] if ":" in url else "5003"
    if not port.isdigit():
        return True
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                     f"$c=Get-NetTCPConnection -State Listen -LocalPort "
                     f"{int(port)} -ErrorAction SilentlyContinue; if($c){{"
                     f" Stop-Process -Id $c.OwningProcess -Force -ErrorAction "
                     f"SilentlyContinue }}"],
                    capture_output=True, timeout=60)
    except Exception as e:
        log(f"  [MEM] TTS stop failed: {str(e)[:60]}", 2)
        return False
    for _ in range(20):
        if not check_tts_server():
            return True
        time.sleep(1)
    return not check_tts_server()


def _file_fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def _tts_fingerprint(text, speed):
    ref_path = Path(CONFIG.get("tts_ref_file", ROOT / "cosy-server" / "ref.json"))
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    wav = Path(ref["wav"])
    if not wav.is_absolute():
        wav = ref_path.parent / wav
    # Weights live in the HF hub cache, not a local dir, so there is no
    # model_files walk. `tts_model` is the cache-invalidation key: bumping it
    # is what retires every cached wav when the engine changes.
# ponytail: x_vector_only_mode MUST be in the fingerprint. It used to be set
    # in cosy-server/server.py, not config.json, so editing the server left every
    # cached wav "valid" and the pipeline kept shipping the pre-fix audio —
    # which had the reference lyric ("was meant to be") prefixed to every scene.
    # A cache key that omits a generation parameter is worse than no cache: it
    # looks correct and serves stale output. Both sides now read the same
    # ref.json field, so flipping clone mode with prep_ref.py --icl retires the
    # cache by itself instead of needing a matching edit here.
    # ponytail: the narration direction PLAN is part of the generation input -
    # it decides the per-line rate and gap, and the text handed to the engine.
    # Omitting it means retuning a constant in narration_direction.py produces
    # different audio while the cache key stays byte-identical, so every cached
    # wav is served as valid. Third instance of a bug class this function
    # already documents twice below, so it is hashed from the module source
    # itself rather than maintained by hand: a hash of the file changes exactly
    # when the behaviour changes.
    _dir = Path(__file__).parent / "pipeline" / "narration_direction.py"
    _dir_fp = _file_fingerprint(_dir)["sha256"] if _dir.is_file() else "missing"
    return {"text": text, "speed": speed, "reference": _file_fingerprint(ref_path),
            "wav": _file_fingerprint(wav),
            "model_id": CONFIG.get("tts_model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            "x_vector_only": bool(ref.get("x_vector_only", True)),
            "language": ref.get("language") or "English",
              # ponytail: the trim used to be part of the GENERATED audio, not
              # the key. Head-only trimming then produced different bytes from
              # identical inputs and every cached wav stayed "valid" -- the old
              # destructive tail trim had already shipped "Ope" for "He opened
              # it." Bump when the filter chain changes.
              "trim": "afade90ms-qsin-levelmatch26.5-volume3db-v6",
              "direction": _dir_fp,
              "url": CONFIG.get("tts_url", "http://localhost:5003")}


def _cache_matches(path, fingerprint):
    try:
        return Path(path).is_file() and json.loads(
            Path(path).with_suffix(".cache.json").read_text(encoding="utf-8")) == fingerprint
    except (OSError, ValueError):
        return False


_SENT_END = re.compile(r"(?<=[.!?…])\s+|\n+")


def split_for_tts(text, max_chars=220, min_words=4):
    """Split narration at sentence boundaries for chunked synthesis.

    Autoregressive TTS (Qwen3-TTS) drifts on one long request and the
    speaker embedding weakens as context grows. The engines' own guidance is
    to split on sentence boundaries and stitch with a short gap. A whole
    scene used to go to /cosy in a single POST.

    Fragments under min_words merge forward so no request is a stub. The
    threshold is a WORD count, not a char count — a 40-char floor glued
    "First sentence here." onto the next one, which defeats the whole point
    of splitting at sentence boundaries.

    Returns a non-empty list, or [] for empty input. A single unbreakable
    run longer than max_chars is returned whole: the engine handles it, and
    chopping mid-word would be worse.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_END.split(text) if p.strip()]
    if not parts:
        return [text]
    # merge stub fragments forward (e.g. "A." "B." "Okay.")
    merged = []
    for part in parts:
        if merged and len(merged[-1].split()) < min_words:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    # hard-split anything still over budget, preferring a comma
    out = []
    for part in merged:
        while len(part) > max_chars:
            cut = part.rfind(", ", 0, max_chars)
            if cut < max_chars // 2:
                cut = part.rfind(" ", 0, max_chars)
            if cut < max_chars // 2:
                break  # one unbreakable run; let the engine have it
            out.append(part[:cut + 1].strip())
            part = part[cut + 1:].strip()
        if part:
            out.append(part)
    return out or [text]


def _concat_wavs(parts, dest, gap_s=0.12, sr=None, gaps=None):
    """Concatenate WAVs with silence between them. Returns dest or None.

    `gaps` is an optional per-join silence list (n-1 entries). gap_s remains the
    fallback for every join when gaps is absent or short, so existing callers
    and tests are unaffected. The variable-gap path exists because narration
    direction assigns each line the beat it earned: a line that landed on its
    payoff gets a longer pause, a short line keeps its momentum. One constant gap
    on every join is the metronome effect.

    The sample rate is read off the FIRST segment, not hardcoded. The TTS
    server is Qwen3-TTS, whose native output is 24000 Hz — a hardcoded 44100
    made `_concat_wavs` raise on every real segment, so 100% of chunked
    synths failed and the render aborted on the voiceless-scene gate.
    Resampling is unnecessary: the segments come from one engine and are
    already uniform, so the only thing to verify is that they AGREE.
    """
    import numpy as _np
    try:
        chunks = []
        for p in parts:
            with wave.open(str(p), "rb") as w:
                if w.getnchannels() != 1 or w.getsampwidth() != 2:
                    raise ValueError(f"unexpected wav format "
                                     f"({w.getnchannels()}ch/"
                                     f"{w.getsampwidth() * 8}bit): {p}")
                rate = w.getframerate()
                if sr is None:
                    sr = rate
                elif rate != sr:
                    raise ValueError(
                        f"mixed sample rates ({rate} vs {sr}): {p}")
                chunks.append(_np.frombuffer(w.readframes(w.getnframes()),
                                            dtype="<i2"))
        if not chunks or not sr:
            return None
        joined = chunks[0]
        for idx, c in enumerate(chunks[1:]):
            g = gap_s
            if gaps and idx < len(gaps):
                g = gaps[idx]
            joined = _np.concatenate(
                [joined, _np.zeros(max(0, int(g * sr)), dtype="<i2"), c])
        peak = int(_np.abs(joined.astype(_np.int32)).max()) if joined.size else 0
        if peak > 32767:
            joined = (joined.astype(_np.float32) * (32767.0 / peak)).astype("<i2")
        with wave.open(str(dest), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(joined.tobytes())
        return dest
    except Exception as e:
        log(f"  [WARN] TTS stitch failed ({str(e)[:90]})", 2)
        return None


def generate_tts(text, output_path, is_narration=True, character_name=None):
    """Generate TTS with the user's cloned voice (Qwen3-TTS, single engine).

    One narrator voice for everything. Sidecar cache decides freshness.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = clean_tts_text(text)
    if not text:
        return False
    _ck = (character_name or "").lower().replace(" ", "_")
    _sp = float((CONFIG.get("voice_speeds", {}) or {}).get(_ck, CONFIG.get("voice_speed", 1.0))) or 1.0
    if not math.isfinite(_sp) or _sp <= 0:
        raise ValueError("voice speed must be positive and finite")
    if _sp != 1.0:
        log(f"  [WARN] voice speed {_sp:g} applied once by synthesis; assembly uses natural WAV timing", 2)
    fingerprint = _tts_fingerprint(text, _sp)
    if _cache_matches(output_path, fingerprint):
        try:
            if output_path.with_suffix(".txt").read_text(encoding="utf-8") == text:
                return True
        except OSError:
            pass
    if not check_tts_server():
        log("  [ERROR] Qwen3-TTS server not running or stale reference loaded")
        return False

    tts_url = CONFIG.get("tts_url", "http://localhost:5003")
    output_path.with_suffix(".cache.json").unlink(missing_ok=True)
    log("  Generating TTS with user-voice clone...")

    # ponytail: `prior` is read by the except handler, so bind it BEFORE
    # the try. Inside, a missing `requests` made the handler itself raise
    # UnboundLocalError and kill the render with the wrong error.
    prior = output_path.read_bytes() if output_path.is_file() else None
    try:
        import requests
        _sp = float((CONFIG.get("voice_speeds", {}) or {}).get(_ck, CONFIG.get("voice_speed", 1.0))) or 1.0
        _segs = split_for_tts(text)
        _chunked = len(_segs) > 1
        from pipeline import narration_direction as _narrate
        _plan = _narrate.direct_all(_segs, base_rate=_sp)
        # Sanity floor on a render: shorter than this and the audio was
        # truncated. 30 chars/sec, divided by the FASTEST rate in the plan
        # because that yields the shortest expected audio. This was written
        # while `speed` was a dead config key; with atempo live, an unadjusted
        # floor sits above the real duration and triggers a pointless retry
        # (300 chars at speed 2.0 expects 5s against a 10s floor).
        _floor = max(0.5, len(text) / 30.0 / max(p["rate"] for p in _plan))
        # Chunked synthesis. Autoregressive TTS (XTTS, Qwen3-TTS) drifts on one
        # long request and the speaker embedding weakens as context grows, so
        # split on sentence boundaries, synthesize each with the same
        # reference, and stitch with a short breath. The server builds its
        # clone prompt once at startup and reuses it, so every segment is
        # anchored to the identical reference. A whole scene used to be a
        # single POST.
        # Narration direction: per-line rate, gap and emphasis. Measured gap on
        # this machine was UTMOS 3.85 for the clone vs 4.45 for a premium
        # narrator on the SAME engine and reference — the audio was never the
        # problem, the timing was. Previously every segment rendered at speed
        # 1.0 and joined with an identical 0.12 s gap, which is the metronome
        # effect creators name first as the cause of robotic AI narration.
        from pipeline import narration_direction as _narrate
        _plan = _narrate.direct_all(_segs, base_rate=_sp)
        for _attempt in range(1, 4):
            if not _chunked:
                response = requests.post(f"{tts_url}/cosy",
                                         json={"text": _plan[0]["emphasised"],
                                               "speed": _plan[0]["rate"]},
                                         timeout=300)
                if response.status_code in (502, 503) and _attempt < 3:
                    time.sleep(5)
                    continue
                if response.status_code != 200:
                    log(f"  [ERROR] CosyVoice server returned {response.status_code}")
                    if _attempt < 3:
                        time.sleep(3)
                        continue
                    return False
                with open(output_path, "wb") as f:
                    f.write(response.content)
                if (_attempt < 3 and audio_duration(str(output_path)) < _floor
                        and len(text) > 40):
                    log(f"  [WARN] collapsed line ({len(text)} chars), "
                        f"retrying ({_attempt}/3)...", 2)
                    time.sleep(3)
                    continue
                break

            _parts, _failed = [], None
            for _i, _d in enumerate(_plan):
                _seg_wav = output_path.with_name(
                    f"{output_path.stem}.seg{_i:02d}.wav")
                _ok = False
                for _try in (1, 2):
                    try:
                        r = requests.post(f"{tts_url}/cosy",
                                          json={"text": _d["emphasised"],
                                                "speed": _d["rate"]},
                                          timeout=300)
                    except Exception as e:
                        _failed = f"segment {_i + 1} request failed: {e}"
                        break
                    if r.status_code == 200 and r.content:
                        _seg_wav.write_bytes(r.content)
                        _ok = audio_duration(str(_seg_wav)) > 0.1
                        if not _ok:
                            _failed = f"segment {_i + 1} came back silent"
                        break
                    _failed = f"segment {_i + 1} HTTP {r.status_code}"
                    time.sleep(2)
                if not _ok:
                    if _seg_wav.exists():
                        _seg_wav.unlink()
                    break
                _parts.append(_seg_wav)
            if len(_parts) == len(_segs):
                # per-line gaps, not one constant. A flat gap on every join is
                # the metronome effect; the plan varies it by what the line did.
                _gaps = [_d["gap_after"] for _d in _plan]
                if _concat_wavs(_parts, output_path, gaps=_gaps) is not None:
                    # clean the segments here too: the success path used to
                    # break out and leave .segNN.wav littering voice/
                    for _p in _parts:
                        _p.unlink(missing_ok=True)
                    break
                _failed = "stitch failed"
            for _p in _parts:
                _p.unlink(missing_ok=True)
            log(f"  [WARN] chunked TTS failed ({_failed or 'unknown'}), "
                f"retrying ({_attempt}/3)...", 2)
            if _attempt < 3:
                time.sleep(3)
                continue
            log(f"  [ERROR] CosyVoice failed after 3 attempts: {_failed}")
            return False
        if _chunked:
            log(f"  TTS generated from {len(_segs)} sentence segments "
                f"(one voice anchor reused per segment)")
        try:
            import numpy as np
            with wave.open(str(output_path), "rb") as w:
                sr, frames = w.getframerate(), w.getnframes()
                pcm = w.readframes(frames)
            dur = frames / sr
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
            energy = float(samples.max())
            if dur < max(0.3, _floor * 0.9) or energy < 500:
                raise RuntimeError(f"unusable TTS audio (duration {dur:.2f}s, peak {energy:.0f})")

            # LEVEL MATCH, because "smooth pacing" is mostly about loudness
            # consistency, not gaps.
            #
            # Measured on ch001: per-line RMS ran from -32.2 to -22.7 dB, a 6.6 dB
            # p10-to-p90 spread. A real narrator holds an episode within about
            # 3 dB. A 6.6 dB jump between consecutive lines is heard as a jolt long
            # before the listener consciously registers a gap - which is exactly
            # the "not smooth" complaint, and no amount of gap tuning fixes it.
            #
            # The engine varies line to line because a short line and a long one
            # integrate their energy differently, so this is expected output, not
            # a fault. What matters is that the pipeline not ship it.
            #
            # Gain is capped at +/-6 dB: the quietest lines are the ones with the
            # least reliable level, and unlimited correction would boost their
            # noise floor along with the voice.
            # `samples` is still int16 SCALE here (not normalised to +/-1.0), so the
            # speech-end threshold has to be absolute, not relative to the peak.
            # Using a relative threshold against an int16 peak compared the signal
            # against 32768 and found "silence" at +60 dBFS - every line then looked
            # like it needed -6 dB, and the correction did nothing but print lies.
            _a = np.abs(samples)
            _nz = np.where(_a > 32768 * 10 ** (-45 / 20))[0]
            if len(_nz):
                _speech = samples[int(_nz[0]):int(_nz[-1]) + 1]
                # _rms is still int16 scale; NARRATION_RMS is dBFS (normalised to
                # 1.0). Dividing one by the other mixed two scales and asked for
                # -83 dB of gain, which the +/-6 dB clamp swallowed - so every line
                # printed a correction and none of them applied.
                _rms_db = 20 * math.log10(max(float(np.sqrt((_speech ** 2).mean()))
                                              / 32768.0, 1e-9))
                _gain = min(6.0, max(-6.0, NARRATION_RMS - _rms_db))
                if abs(_gain) >= 0.5:
                    log(f"    level {_rms_db:.1f} dB -> {_gain:+.1f} dB "
                        f"to match the episode", 2)
                    samples = np.clip(samples * (10 ** (_gain / 20.0)),
                                      -32767, 32767)
                    with wave.open(str(output_path), "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(sr)
                        w.writeframes(samples.astype("<i2").tobytes())
        except (wave.Error, OSError) as e:
            raise RuntimeError(f"unusable TTS audio: {e}")
        # Tail release + pad + headroom. NO silenceremove, NO noise bed.
        #
        # History, because this filter chain has now been wrong THREE times and
        # each mistake was invisible in the output file.
        #
        # 1. The original chain was
        #       silenceremove , areverse , silenceremove , areverse
        #    and it DELETED SPOKEN AUDIO. "He opened it." became "Ope".
        # 2. "Fixed" by trimming the head only, at -60dB over 0.1s. Still wrong,
        #    just less obviously. silenceremove gates on LEVEL, and the soft ONSET
        #    of the first word is quieter than -60dB, so it deleted the first
        #    word of 13 of 50 renders.
        # 3. Then the user reported an abrupt cut at the end of every line. The
        #    cause was genuinely there - 37% of the preview was literal digital
        #    zero - but the fix chosen for it (a room-tone bed mixed under every
        #    line) introduced hiss audible in every pause, which the user then
        #    reported as a disturbing background noise. That was a straight
        #    trade: one artefact replaced by a worse one. The bed is gone.
        #
        # What actually fixes it costs nothing: afade=t=out ramps the last 45ms
        # to zero. A hard stop at full amplitude IS the "cut". A ramp is a
        # release. No noise is added, so there is nothing new to hear.
        #
        # There is no pad here at all - see the note above the chain - so the fade
        # simply has to come first and it ramps the real tail of the last word.
        #
        # The fade is written areverse,afade=t=in,areverse rather than
        # afade=t=out:st=<duration-0.045>, for two reasons. `st` on t=out is
        # measured from the START of the stream, so the obvious-looking
        # `afade=t=out:st=0` ramps the FIRST 45ms and silently deletes the quiet
        # onset - the exact failure this chain already committed twice - and
        # st=<duration-d> needs the duration known up front. Reversing first
        # makes the fade duration-independent and impossible to get backwards.
        # Measured on a fixture with a 0.002-amplitude onset at 0.20s:
        #     afade=t=out:st=0            onset band max 0.000000  <- destroyed
        #     areverse,afade=t=in,areverse onset band max 0.001404  <- intact
        #
        # No threshold separates a TTS word onset from silence, so any level-gated
        # silence filter is the wrong tool here. Measured, same wav into each
        # variant, 2 renders x 25 lines:
        #
        #   chain                                  mean WER   lost a word
        #   silenceremove .1s @-60dB (head trim)     21.0%       13 / 50
        #   afade + volume                         12.8%        0 / 50
        #   silenceremove .35s @-50dB               48.0%       49 / 50
        #
        # The third row is the instructive one: LOWERING sensitivity and RAISING
        # the minimum duration made it dramatically WORSE, because a 0.35s window
        # swallows the first word outright.
        #
        # A dropped word survives a listen-through, because the listener supplies
        # the missing word from context. That is why these were caught by
        # transcription and not by ear.
        #
        # The scene slot owns the pause between sentences; this adds nothing to it.
        # Leading silence still costs nothing: the assembler's scene durations come
        # from the narration length (compute_scene_durations).
        try:
            trimmed = output_path.with_suffix(".trim.wav")
            trim = subprocess.run(
                [CONFIG["ffmpeg"], "-y", "-i", str(output_path),
                 # NO apad: the scene slot owns the pause between sentences
            # (_dur is exactly narration + NARRATION_GAP), so a file-level pad is
            # a second gap - measured 0.40s instead of 0.20s. It was masked while
            # the slot carried 1.24s of slack of its own.
            #
            # THE FADE IS THE RELEASE, AND IT IS THE WHOLE FIX for "dramatic stop".
            # 90ms, quarter-sine. Both numbers are measured, not chosen:
            #
            #   fade            dB lost in the final 20ms      WER
            #   none                    9.3                     0.0%
            #   45ms linear            10.5                     2.9%   <- was shipping
            #   90ms linear            16.1                     4.0%
            #   90ms QUARTER-SINE       2.9                     0.0%   <- shipped
            #   140ms quarter-sine      6.2                     9.5%
            #   140ms double-exp        3.0                   108.5%
            #
            # Why the old 45ms was not a release at all: the engine leaves the last
            # word sounding at -4 dBFS, so a 45ms LINEAR ramp ran -4 -> -18 -> -38
            # -> -69 dB. That is a fast chop with a blur on it, and it also cost
            # 2.9% word error by clipping the final phoneme. A linear fade starts
            # at full steepness, which is the opposite of what a release needs.
            #
            # Quarter-sine starts gently and lands hard, so the word's body is
            # untouched and only the last few ms are smoothed: 3.5x less abrupt
            # AND less word error than the 45ms linear it replaces.
            #
            # The dexp trap: a double-exponential curve scores beautifully on the
            # abruptness metric and DESTROYS the audio (108% WER). It decays fast
            # at the start, so it eats the word. Never judge a fade on drop20 alone.
            "-af", "areverse,afade=t=in:st=0:d=0.090:curve=qsin,areverse,volume=-3dB",
                 "-ar", "24000", "-ac", "1", str(trimmed)],
                capture_output=True, timeout=60)
            if trim.returncode == 0 and trimmed.exists() and trimmed.stat().st_size > 0:
                trimmed.replace(output_path)
            elif trimmed.exists():
                trimmed.unlink()
        except Exception:
            pass
        log("  TTS generated with user-voice clone")
        output_path.with_suffix(".txt").write_text(text, encoding="utf-8")
        output_path.with_suffix(".cache.json").write_text(
            json.dumps(fingerprint), encoding="utf-8")
        return True
    except Exception as e:
        if prior is not None:
            try:
                output_path.write_bytes(prior)
            except OSError:
                pass
        log(f"  [ERROR] CosyVoice failed: {str(e)}")
        return False

# --- Passage synthesis ------------------------------------------------
# One continuous read for a run of scenes, split back into per-scene wavs.
# See pipeline/passage.py for the measurement that motivates it.

# How many scenes' worth of narration goes into a single engine call.
#
# The prosody benefit comes from CONTEXT, so the limit is about the model, not
# about taste. Qwen3-TTS streams at a 12.5 Hz token rate and autoregresses, so a
# very long request drifts and weakens the speaker embedding - which is why the
# engine's own guidance is to keep requests bounded. 8 sentences is a couple of
# sentences' worth of breath group: long enough for the model to keep pitch and
# energy continuous across several boundaries, short enough to stay well inside
# the length this engine holds together. Measured benefit falls off sharply
# beyond that, and a drifted long request costs speaker similarity, which is the
# one thing the clone exists to protect.
PASSAGE_SCENES = 8

# Hard cap on characters in one request. The engine has a MAX_CHARS of its own
# and will 400 rather than truncate, but asking for a passage it cannot hold is
# how the speaker embedding starts to drift.
PASSAGE_MAX_CHARS = 700


def generate_tts_passage(texts, paths, speed=1.0, tts_url=None):
    """Synthesise `texts` as ONE continuous read, then cut it into `paths`.

    Returns a list of booleans, one per input, matching generate_tts's contract.

    Falls back to per-line generate_tts for any line the batch path cannot
    deliver, so this is strictly an improvement and never a new failure mode: if
    the passage request fails, or the split produces an unusable piece, that
    scene is simply rendered the old way.

    The per-line wavs are post-processed exactly as generate_tts does -
    level-matched and released at the tail - so a scene sounds identical whether
    it came from a batch or a single call, and the cache sidecars still describe
    the audio.
    """
    import tempfile as _tf
    from pipeline import passage as _P

    n = len(texts)
    if n == 0:
        return []
    if n == 1:
        return [generate_tts(texts[0], paths[0])]

    results = [False] * n
    pending = list(range(n))
    joined = " ".join(t.strip() for t in texts if t and t.strip())
    url = tts_url or CONFIG.get("tts_url", "http://localhost:5003")

    if joined and len(joined) <= PASSAGE_MAX_CHARS and check_tts_server():
        tmpdir = Path(_tf.mkdtemp(prefix="passage_"))
        raw = tmpdir / "passage.wav"
        try:
            import requests
            r = requests.post(f"{url}/cosy",
                              json={"text": joined, "speed": speed},
                              timeout=900)
            if r.status_code == 200 and r.content:
                raw.write_bytes(r.content)
                y, sr = _P.read_wav(raw)
                # Sanity: the passage must be roughly as long as the text implies.
                # A collapsed request would otherwise split into N silent stubs.
                floor = len(joined) / 30.0 / max(speed, 0.1) * 0.5
                if audio_duration(str(raw)) >= floor:
                    try:
                        _P.split_passage(raw, [Path(p) for p in paths], sr)
                    except ValueError as ve:
                        # A cut landed on audio. Clipping a word is a worse defect
                        # than the pitch seam, so refuse the run and render it the
                        # old way rather than ship a chopped line.
                        log(f"  [WARN] passage split unsafe ({str(ve)[:60]}), "
                            f"falling back per line", 2)
                    else:
                        for i, p in enumerate(paths):
                            d = audio_duration(str(p))
                            if d > 0.3:
                                results[i] = True
                                pending.remove(i)
                else:
                    log(f"  [WARN] passage collapsed ({audio_duration(str(raw)):.1f}s "
                        f"for {len(joined)} chars), falling back per line", 2)
        except Exception as e:
            log(f"  [WARN] passage synthesis failed ({str(e)[:70]}), "
                f"falling back per line", 2)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    for i in pending:
        # generate_tts has no speed= parameter — this raised TypeError on the
        # fallback path. Rate comes from CONFIG["voice_speed"] internally,
        # which is what speed=1.0 meant here anyway.
        results[i] = generate_tts(texts[i], paths[i])
    return results

# --- Hook Generator ------------------------------------------------
def create_hook_clip(scenes, chapter_num, output_path):
    """Create a viral hook clip for the first 1.5 seconds - FAST."""
    first_scene = scenes[0] if scenes else {}
    hook_text = first_scene.get("text_overlay", "YOU WON'T BELIEVE WHAT HAPPENS NEXT")
    hook_type = first_scene.get("hook_type", "shock")

    hook_styles = {
        "shock": {"color": "0xFF0000", "bg": "0x000000", "duration": 1.5},
        "mystery": {"color": "0xFFFFFF", "bg": "0x0A0A0A", "duration": 1.5},
        "emotion": {"color": "0xFFFF00", "bg": "0x1A0A00", "duration": 1.5},
        "cliffhanger": {"color": "0xFF4444", "bg": "0x0A0000", "duration": 1.5},
        "tension": {"color": "0xFFFFFF", "bg": "0x00000A", "duration": 1.5},
    }
    style = hook_styles.get(hook_type, hook_styles["shock"])
    out_w, out_h = get_output_dims()
    hook_fs = 96 if out_h > out_w else 80
    # Escape all ffmpeg drawtext-special chars
    hook_text_esc = re.sub(r"([\\':;\[\]%])", r"\\\1", hook_text)
    # ponytail: font NAME needs fontconfig (broken in this build). Absolute path only.
    _ff = "C\\:/Windows/Fonts/impact.ttf" if Path("C:/Windows/Fonts/impact.ttf").exists() else "C\\:/Windows/Fonts/arial.ttf"

    cmd = [
        CONFIG["ffmpeg"], "-y",
        "-f", "lavfi", "-i", f"color=c={style['bg']}:s={out_w}x{out_h}:d={style['duration']}",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-vf", (
            f"scale={out_w}:{out_h},zoompan=z='min(1.15,1.0+0.15*on/36)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=36:fps=24:s={out_w}x{out_h},"
            f"drawtext=text='{hook_text_esc}':"
            f"fontcolor={style['color']}:fontsize={hook_fs}:"
            f"x=(w-tw)/2:y=(h-th)/2:"
            f"alpha='if(lt(t,0.15),t/0.15,if(gt(t,{style['duration']-0.15}),"
            f"(({style['duration']}-t)/0.15),1))':"
            f"fontfile='{_ff}':borderw=4:bordercolor=black"
        ),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-t", str(style["duration"]),
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=30)
    if r.returncode != 0 or not output_path.exists():
        log(f"  [ERROR] Hook build failed: {r.stderr.decode('utf-8', 'replace')[-200:] if r.stderr else ''}", 1)
        return False
    return True
# --- Subtitle Generator (kinetic captions) -------------------------
def _ass_ts(seconds):
    seconds = max(0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _ass_sec(ts):
    """Reverse of _ass_ts (H:MM:SS.cc) for srt export."""
    try:
        h, m, s = str(ts).split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0.0


def speech_bounds(audio_path):
    """(first_speech, last_speech) seconds in a wav via silence detection."""
    # ponytail: threshold matches the TTS trim stage (-50dB). Trim already
    # removed sub--50dB heads/tails, so detecting at -40dB dated speech
    # starts late and QA/captions flagged soft consonants as silence.
    try:
        out = subprocess.run(
            [CONFIG["ffmpeg"], "-i", str(audio_path),
             "-af", "silencedetect=noise=-50dB:d=0.25", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60)
        starts, ends = [], []
        for line in out.stderr.splitlines():
            m = re.search(r"silence_start:\s*([\d.]+)", line)
            if m:
                starts.append(float(m.group(1)))
            m = re.search(r"silence_end:\s*([\d.]+)", line)
            if m:
                ends.append(float(m.group(1)))
        dur = audio_duration(audio_path)
        first = ends[0] if starts and ends and starts[0] <= 0.01 else 0.0
        last = starts[-1] if starts and (len(starts) > len(ends) or ends[-1] >= dur - 0.01) else dur
        if last <= first:
            return 0.0, dur
        return first, last
    except Exception:
        return 0.0, audio_duration(audio_path)


def _ass_ts(seconds):
    h = int(max(0.0, seconds) // 3600)
    m = int((max(0.0, seconds) % 3600) // 60)
    s = max(0.0, seconds) % 60
    return "%d:%02d:%05.2f" % (h, m, s)


# Caption specs, all from published standards rather than taste:
#   CPL 42 / 2 lines      Netflix Timed Text Style Guide (General Requirements);
#                         BBC, EBU and TED all agree
#   17 CPS                Netflix adult English. 20 is the hard ceiling, 17 is
#                         the comfortable read. BBC 15-17, TED 21
#   5/6s min, 6s max      Netflix min is 20 frames at 24fps
#   2 frame gap           Netflix Timing Guidelines; 3-11 frame gaps must be
#                         closed to 2
#   sentence case         Tinker 1955 (Legibility of Print) measured all-caps
#                         at 9.5-19% slower reading, with 9 of 10 adult
#                         readers judging it less legible. Caps flattens the
#                         ascender/descender silhouette, so words stop being
#                         read by shape and start being read letter by letter.
#                         Caps is defensible only for 2-3 word hook cards.
#   6-8% frame height     long-form 16:9. Vertical is 7-10% with a 300px+
#                         bottom dead zone for the platform UI.
CAP_CPL = 42
CAP_MAX_LINES = 2
CAP_CPS_TARGET = 17.0
CAP_CPS_MAX = 20.0
CAP_MIN_DUR = 5.0 / 6.0
CAP_MAX_DUR = 6.0
CAP_GAP_FRAMES = 2
# Netflix English (USA) names Arial as the generic proportional sans-serif, and
# it is the one face present in every Windows install this runs on. Named once
# so the receipt cannot disagree with the header.
CAP_FONT = "Arial"

# Caption receipt. A caption track that silently violates its own standard is
# indistinguishable from a good one from the outside, so the measured numbers
# travel into the EDL with everything else: cue count, worst reading speed, the
# longest line, and the style actually written. Assembly merges this, the
# dashboard shows it, and the export carries it.
_LAST_CAPTIONS = {}

# Function words a line must not begin with (Netflix line-break rules).
_CAP_NO_LEAD = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "he", "her",
    "his", "in", "is", "it", "its", "nor", "of", "on", "or", "she", "that",
    "the", "their", "them", "they", "this", "to", "was", "were", "which",
    "who", "will", "with", "would",
}


def cap_font_size(out_w, out_h):
    """Largest size that keeps a 42-character line inside the title-safe area.

    Netflix sizes type by "ability to fit 42 characters across screen", which is
    a width constraint, so it is solved against the real font instead of guessed.
    The ceiling is 90% of frame width; the result is then capped at 8% of frame
    height (long-form) so two lines never eat the picture.
    """
    safe_px = out_w * 0.90
    # measured: Arial fits 42 mixed-case chars up to ~89px at 1920 wide
    size = max(12, int(safe_px / 42 / 0.50))
    return max(16, min(size, int(out_h * 0.08)))


def _cap_wrap(words, width=CAP_CPL, max_lines=CAP_MAX_LINES):
    """Wrap to `width` and `max_lines`, then repair breaks that orphan a word
    or start on a function word. Netflix: avoid one or two words on the top line.

    The repair moves are bounded: a word is only pushed down if the line it
    came from stays over the limit, so a repair can never create a third line
    or an over-length line. When the text simply does not fit, the tail is
    truncated rather than silently shipped as a wall of text.
    """
    lines, cur = [], []
    for word in words:
        trial = cur + [word]
        if len(" ".join(trial)) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = [word]
    if cur:
        lines.append(cur)

    # repair, in place, without ever exceeding the limits
    for i in range(len(lines) - 1):
        while len(lines[i]) <= 2 and len(lines[i + 1]) > 1 \
                and len(" ".join(lines[i + 1])) < width:
            lines[i + 1].insert(0, lines[i].pop())
    for i in range(1, len(lines)):
        while lines[i] and lines[i][0].lower().strip(",.;:") in _CAP_NO_LEAD \
                and len(" ".join(lines[i - 1])) + 1 + len(lines[i][0]) <= width:
            lines[i - 1].append(lines[i].pop(0))

    # A cue that still runs past two lines is a text-length problem, not a
    # wrapping one. Truncating would drop words the narrator said; the
    # alternative is a third line, which the standard also forbids. Reading
    # speed wins over line count, because a 3-line cue at 17 CPS is readable
    # and a 2-line cue at 36 CPS is not. The caller logs the overflow.
    return [" ".join(l) for l in lines]


def _cap_split(text, window):
    """[(text, share_of_window)] — one entry per cue.

    The old code chopped every scene into fixed 3-word chunks regardless of
    length, so a 30-character line got three cues and a split could duplicate
    a word. Now a line that fits stays one cue; a line that does not is split
    at clause boundaries only, and each piece gets a slice of the window
    proportional to its length so reading speed stays even across the split.
    """
    words = text.split()
    if not words:
        return []
    if len(text) <= CAP_CPL * CAP_MAX_LINES \
            and len(text) / CAP_CPS_TARGET <= window:
        return [(text, 1.0)]
    parts, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.!?…]+\s*$", w) or w.endswith((",", ";", ":")):
            parts.append(cur)
            cur = []
    if cur:
        if parts:
            parts[-1].extend(cur)
        else:
            parts = [cur]
    # Pack clauses into groups that each fit the two-line budget, rather than
    # emitting one cue per clause (which over-fragments) or giving up and
    # shipping a wall (which is what the old fixed-size fallback did).
    budget = CAP_CPL * CAP_MAX_LINES
    groups, cur = [], []
    for p in parts:
        # trial must be a list of CLAUSES: cur is a list of clauses and p is
        # one clause, so `cur + p` spliced a clause's words in at the top
        # level and the join below then walked them character by character,
        # producing "B y d a w n ..." as a single token.
        trial = cur + [p]
        if len(" ".join([" ".join(t) for t in trial])) <= budget or not cur:
            cur = trial
        else:
            groups.append(cur)
            cur = [p]          # a GROUP is a list of clauses; `p` is one clause
    if cur:
        groups.append(cur)
    # groups is a list of groups, each group a list of clauses, each clause a
    # list of words — three levels. Two joins, not one: joining groups one level
    # too shallow stringifies the word lists and every character comes out
    # space-separated.
    texts = [" ".join([" ".join(clause) for clause in g]) for g in groups]
    if len(texts) < 2:
        return [(text, 1.0)]
    total = sum(len(t) for t in texts) or 1
    return [(t, len(t) / total) for t in texts]


def create_ass_subtitles(scenes, output_path, dialogue_audios=None):
    """Captions synced to actual speech, built to published readability limits.

    ponytail: what changed and why. Timing was already speech-measured
    (speech_bounds), which was the hard part. What was wrong was the shaping:
      * fixed 3-word chunks, so short lines got one cue per three words and a
        scene could emit "white..." twice
      * no line-length or reading-speed limit, so a long line shipped at 30+
        CPS and viewers had to choose between reading and watching
      * font size hardcoded at 48px, and outline 3px, which vanishes against
        the deep blacks these frames produce
      * no minimum duration, so a one-word cue flashed for 0.2s
    The standard is the deliverable: 42 chars, 2 lines, 17 CPS, 5/6s..6s,
    2-frame gaps, sentence case, size derived from the title-safe area.
    """
    out_w, out_h = get_output_dims()
    vertical = out_h > out_w
    font_size = cap_font_size(out_w, out_h)
    # Vertical needs a 30% bottom dead zone for the platform's own caption
    # buttons and progress bar. Horizontal needs 10% and nothing more:
    # caption_safe_margin_v is 320 in config.json, which is the VERTICAL
    # value, and applying it to a 1080p landscape render pushed the captions
    # to 30% up the frame - they looked like they were floating in the middle
    # of the shot. The key only governs vertical now.
    margin_v = (int(CONFIG.get("caption_safe_margin_v", int(out_h * 0.30)))
                if vertical else int(out_h * 0.10))
    kinetic = bool(CONFIG.get("kinetic_captions", False))
    outline = int(CONFIG.get("caption_outline", 5))
    fps = max(1, int(CONFIG.get("fps", 24)))
    hook_dur = 1.5 if bool(CONFIG.get("hook_enabled", False)) else 0.0
    text_over = bool(CONFIG.get("caption_text_over", False))

    header = f"""[Script Info]
Title: Anime Series
ScriptType: v4.00+
PlayResX: {out_w}
PlayResY: {out_h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{CAP_FONT},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},0,2,30,30,{margin_v},1
Style: Pop,{CAP_FONT},{font_size},&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},0,2,30,30,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    default_dur = float(CONFIG.get("scene_duration", 5))
    # P3 ops: assembly is hard-cut (bridge_len=0.0) — captions must match,
    # never re-derive bridges from crossfade config (latent divergence).
    bridge_len = 0.0
    current_time = hook_dur + (bridge_len if scenes else 0.0)
    cues = []

    for i, scene in enumerate(scenes):
        if "_cap_start" in scene:
            current_time = float(scene["_cap_start"])
            scene_dur = float(scene.get("_cap_dur", scene.get("_dur", default_dur)))
        else:
            scene_dur = float(scene.get("_dur", default_dur))
        narration = scene.get("narration") or scene.get("dialogue") or ""
        text = narration.strip('"').strip("'").strip()
        _done = 0.0
        if text:
            aud = dialogue_audios[i] if dialogue_audios and i < len(dialogue_audios) else None
            if aud and Path(aud).exists():
                sp_start, sp_end = speech_bounds(aud)
                _lead = float(scene.get("_lead", 0.2))
                win_start = current_time + _lead + sp_start
                win_end = min(current_time + scene_dur - 0.01,
                              current_time + _lead + sp_end)
            else:
                win_start = current_time
                win_end = current_time + scene_dur - 0.2
            win_start = max(current_time, win_start)
            window = win_end - win_start
            if window >= 0.01:
                _style = "Pop" if kinetic else "Default"
                for tx, share in _cap_split(text, window):
                    s0 = win_start + window * _done
                    e0 = win_start + window * (_done + share)
                    cues.append({"s": s0, "e": e0,
                                 "lines": _cap_wrap(tx.split()),
                                 "style": _style})
                    _done += share
        current_time += scene_dur + bridge_len

    # ---- enforce the standard on the cue list.
    # Order matters: reading speed first (extend into the pause), then
    # min duration, then max. Doing min-duration first pushed cues into their
    # neighbours and produced overlaps, because the audio window is often
    # shorter than CAP_MIN_DUR on its own.
    for c in cues:
        chars = sum(len(l) for l in c["lines"])
        if chars / max(0.01, c["e"] - c["s"]) > CAP_CPS_TARGET:
            e2 = c["s"] + chars / CAP_CPS_TARGET
            if e2 - c["s"] <= CAP_MAX_DUR:
                c["e"] = e2
        if c["e"] - c["s"] < CAP_MIN_DUR:
            c["e"] = c["s"] + CAP_MIN_DUR
        if c["e"] - c["s"] > CAP_MAX_DUR:
            c["e"] = c["s"] + CAP_MAX_DUR
    # A narration longer than its slot allows cannot be made readable by timing
    # alone: 170 characters in a 3.4s window is 50 CPS no matter how it is cut.
    # The out-time is allowed to run to the maximum, and whatever still exceeds
    # the ceiling is REPORTED, because the fix is a shorter script line, not a
    # cleverer caption. Silently shipping it is what this is preventing.
    _over = []
    for c in cues:
        cps = sum(len(l) for l in c["lines"]) / max(0.01, c["e"] - c["s"])
        if cps > CAP_CPS_MAX:
            _over.append(cps)
    if _over:
        log(f"  [WARN] {len(_over)} cue(s) over {CAP_CPS_MAX:.0f} CPS "
            f"(max {max(_over):.0f}) - the narration is longer than its slot; "
            f"shorten the line in the script", 1)
    # 2-frame gap, and no overlap. A cue that cannot be pulled back far enough
    # keeps its min duration and the NEXT cue yields by starting later; that
    # trades a late caption for two captions on top of each other.
    # Quantise to centiseconds BEFORE enforcing. The ASS writer emits
    # H:MM:SS.cc, so a 20-frame (0.8333s) minimum cannot be represented exactly
    # and a 2-frame (0.0833s) gap rounded to 0.08 = 1.92 frames. Enforcing on
    # un-rounded values produced a file that read back as non-compliant with
    # its own standard, which is the worst kind of bug: a gate that fails the
    # render it just wrote.
    for c in cues:
        c["s"] = math.ceil(c["s"] * 100.0) / 100.0
        c["e"] = math.ceil(c["e"] * 100.0) / 100.0
    # One forward pass, not a pairwise fix-up. Pulling an earlier cue's END back
    # to open a gap can violate its own minimum duration, which is how this
    # ended up with 1.92-frame gaps and overlapping cues. Moving FORWARD is
    # always satisfiable: each cue starts at least CAP_GAP_FRAMES after the
    # previous one ends, and is at least CAP_MIN_DUR long.
    _gap = CAP_GAP_FRAMES / fps
    for k in range(len(cues)):
        if k:
            cues[k]["s"] = max(cues[k]["s"],
                               math.ceil((cues[k - 1]["e"] + _gap) * 100.0) / 100.0)
        if cues[k]["e"] < cues[k]["s"] + CAP_MIN_DUR:
            cues[k]["e"] = math.ceil((cues[k]["s"] + CAP_MIN_DUR) * 100.0) / 100.0
        if cues[k]["e"] - cues[k]["s"] > CAP_MAX_DUR:
            cues[k]["e"] = cues[k]["s"] + CAP_MAX_DUR

    events = []
    for c in cues:
        body = "\\N".join(c["lines"])
        effect = "" if text_over else "{\\fad(200,200)}"
        events.append(
            f"Dialogue: 0,{_ass_ts(c['s'])},{_ass_ts(c['e'])},{c['style']},,0,0,0,,"
            f"{effect}{body}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write(header)
        f.write("\n".join(events))
    _max_cps = max((sum(len(l) for l in c["lines"]) / max(0.01, c["e"] - c["s"])
                    for c in cues), default=0.0)
    log(f"  Captions: {len(cues)} cue(s), {font_size}px, outline {outline}px, "
        f"max {_max_cps:.1f} CPS", 1)
    _LAST_CAPTIONS.clear()
    _LAST_CAPTIONS.update({
        "cues": len(cues),
        "max_cps": round(_max_cps, 1),
        "over_cps_ceiling": len(_over),
        "max_line_chars": max((len(l) for c in cues for l in c["lines"]), default=0),
        "cpl": CAP_CPL,
        "font": CAP_FONT,
        "font_size": font_size,
        "outline": outline,
        "shadow": 0,
        "margin_v": margin_v,
        "vertical": vertical,
        "sentence_case": not any(
            all(x.isupper() for x in "".join(l) if x.isalpha()) and
            len([x for x in "".join(l) if x.isalpha()]) > 3
            for c in cues for l in c["lines"]),
    })

    # P3: .srt sidecar for YT upload + discoverability (same timings, CPS-safe).
    try:
        def _srt_ts(sec):
            # m1: round (not truncate) — float 3723.45 formats as .44999… w/o this.
            total_ms = int(round(max(0.0, float(sec)) * 1000))
            h, rem = divmod(total_ms, 3600000)
            m, rem = divmod(rem, 60000)
            s, ms = divmod(rem, 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        import re as _re
        srt_lines, _n = [], 1
        for ev in events:
            parts = ev.split(",", 9)
            if len(parts) < 10:
                continue
            srt_lines += [str(_n), f"{_srt_ts(_ass_sec(parts[1]))} --> {_srt_ts(_ass_sec(parts[2]))}",
                          # \N is an ASS line break; SRT has no such escape and
                          # only has a real newline. One shipped cue read
                          # "the industrial city was waking\Nup." verbatim.
                          _re.sub(r"\\N", "\n", _re.sub(r"\{[^}]*\}", "", parts[9])).strip(), ""]
            _n += 1
        output_path.with_suffix(".srt").write_text("\n".join(srt_lines), encoding="utf-8")
    except Exception as e:
        log(f"  [WARN] srt export skipped ({e})", 2)
    return output_path


def _platform_srt(output):
    """The CPS-enforced .srt that assemble_video_v2 placed beside the mp4.

    assemble_video_v2 renames subtitles.srt onto output.with_suffix('.srt'),
    so for chNNN_youtube.mp4 this is videos/chNNN_youtube.srt. Returns the
    path only when it exists AND has cues — an empty file is not a subtitle
    track, and release_package would otherwise ship it.
    """
    try:
        p = Path(output).with_suffix(".srt")
        return p if p.is_file() and p.stat().st_size > 0 else None
    except (TypeError, OSError):
        return None


def build_shorts_cut(source_mp4, chapter_num, scenes=None):
    """P3: vertical 9:16 cutdown (scale-to-cover + center-crop) for Shorts/TikTok.

    Returns path or None. Audio copied, video single re-encode (crf 20).
    Thumbnails: pick_thumbnail() below selects sharpest A-still.
    """
    try:
        src = Path(source_mp4)
        if not src.exists():
            return None
        out = src.parent / f"ch{chapter_num:03d}_shorts.mp4"
        # M2: shipped source is 1920x1080 landscape — direct 1080x1920 crop
        # has negative offsets and fails. Scale-to-cover first, then crop.
        r = subprocess.run(
            [CONFIG["ffmpeg"], "-y", "-i", str(src), "-vf",
             "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
             "-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-c:a", "copy", str(out)],
            capture_output=True, timeout=600)
        return out if (r.returncode == 0 and out.exists()) else None
    except Exception:
        return None


def pick_thumbnail(images_dir, chapter_num, scenes=None):
    """P3: thumbnail = sharpest A-still (excludes b_/c_ companions)."""
    import re as _re2
    try:
        rx = _re2.compile(r"_s(\d{2})([bc])?_")
        cands = [p for p in Path(images_dir).glob(f"ch{chapter_num:03d}_s*.png")
                 if not ((m := rx.search(p.name)) and m.group(2))] if scenes else []
        if not cands:
            cands = sorted(Path(images_dir).glob(f"ch{chapter_num:03d}_*.png"))
        if not cands:
            return None
        # sharpest by _image_metrics, alphabetical fallback
        try:
            best = max(cands, key=lambda p: _image_metrics(p).get("sharpness", 0))
            return str(best)
        except Exception:
            return _newest(cands)
    except Exception:
        return None


def qa_clip_file(path):
    """Watchable clip: exists, probeable video stream, sane duration/dims,
    not (near-)all black. Returns (ok, reason)."""
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size < 10240:
            return False, "missing/tiny"
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "stream=width,height,duration", "-of", "csv=p=0", str(p)],
                            capture_output=True, text=True, timeout=30)
        parts = (pr.stdout or "").strip().splitlines()
        if pr.returncode or not parts:
            return False, "unprobeable"
        w, h, d = (parts[0].split(",") + ["0", "0", "0"])[:3]
        dur = float(d)
        if min(int(float(w)), int(float(h))) < 240 or dur < 0.5:
            return False, f"bad dims/duration ({w}x{h} {d}s)"
        bd = subprocess.run([CONFIG["ffmpeg"], "-v", "info", "-i", str(p),
                             "-vf", "blackdetect=d=0.5:pix_th=0.10",
                             "-an", "-f", "null", "-"],
                            capture_output=True, text=True, timeout=120)
        blacks = [float(b) for b in re.findall(r"black_duration:([0-9.]+)", bd.stderr or "")]
        if blacks and sum(blacks) >= 0.9 * dur:
            return False, f"near-all black ({sum(blacks):.1f}s of {dur:.1f}s)"
        return True, "ok"
    except Exception as e:
        return False, str(e)[:80]


def qa_auto_editor(output_path, scenes, dialogue_audios):
    """Editor QA: auto-editor silence-cuts the final mix; every scene's
    speech window must survive. Warn-only (heuristic), returns violations.
    """
    import xml.etree.ElementTree as _ET
    if not shutil.which("auto-editor"):
        return None
    if not bool(CONFIG.get("qa_auto_editor", True)):
        return None
    tmp = Path(str(output_path) + ".qa.xml")
    try:
        r = subprocess.run(["auto-editor", str(output_path), "--edit", "audio:threshold=0.04",
                            "--margin", "0.1sec", "-ex", "premiere", "-o", str(tmp)],
                           capture_output=True, timeout=600)
        if r.returncode != 0 or not tmp.exists():
            return None
        root = _ET.parse(str(tmp)).getroot()
        try:
            tb = float(root.findtext("./sequence/rate/timebase") or 30)
        except Exception:
            tb = 30.0
        kept = []  # source-time ranges that survived, seconds
        for ci in root.iter("clipitem"):
            try:
                kept.append((float(ci.findtext("in")) / tb, float(ci.findtext("out")) / tb))
            except Exception:
                pass
        if not kept:
            return None
        bad = []
        for i, scene in enumerate(scenes):
            aud = dialogue_audios[i] if i < len(dialogue_audios) else None
            if not (aud and Path(aud).exists() and "_cap_start" in scene):
                continue
            sp0, sp1 = speech_bounds(aud)
            # P1: honor _lead-centered truth like captions/sync/master do.
            _lead = float(scene.get("_lead", 0.2))
            a = float(scene["_cap_start"]) + _lead + sp0
            b = float(scene["_cap_start"]) + _lead + sp1
            cut = (b - a) - sum(max(0.0, min(b, k1) - max(a, k0)) for k0, k1 in kept)
            if cut > 0.15:
                bad.append((scene.get("scene_number", i + 1), round(cut, 2)))
        try:
            tmp.unlink()
        except Exception:
            pass
        if bad:
            log(f"  [QA] auto-editor: speech cut in scenes {bad}", 1)
        else:
            log("  [QA] auto-editor: all speech survives", 1)
        return bad
    except Exception as e:
        log(f"  [WARN] auto-editor QA skipped ({e})", 2)
        return None


def qa_av_sync(output_path, scenes, dialogue_audios, hook_dur=0.0):
    import numpy as np
    result = {"passed": False, "violations": [], "skipped": [], "errors": [],
              "checked": [], "total": len(scenes), "scope": "audio-only"}
    sr = 16000

    def decode(path):
        proc = subprocess.run(
            [CONFIG["ffmpeg"], "-v", "error", "-i", str(path),
             "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
            capture_output=True, timeout=300)
        if proc.returncode:
            raise RuntimeError(f"Audio decode failed: {path}: {proc.stderr!r}")
        samples = np.frombuffer(proc.stdout, dtype=np.float32).astype(np.float64)
        if not len(samples) or not np.isfinite(samples).all():
            raise RuntimeError(f"Empty/non-finite decoded audio: {path}")
        return samples

    try:
        mix = decode(output_path)
        for i, scene in enumerate(scenes):
            num = scene.get("scene_number", i + 1)
            aud = dialogue_audios[i] if i < len(dialogue_audios) else None
            try:
                if not aud:
                    result["skipped"].append({"scene": num, "reason": "no audio template"})
                    continue
                if "_cap_start" not in scene:
                    raise ValueError("missing assembly timeline")
                start = float(scene["_cap_start"])
                # ponytail: _cap_* are 2-dp rounded (assembly ~L3853); a
                # rounded end can exceed mix length by a few ms with zero
                # A/V error. Prefer full-precision _dur + real tolerance.
                dur = float(scene.get("_dur") or scene.get("_cap_dur") or 0)
                lead = float(scene.get("_lead", 0.2))
                if not all(math.isfinite(x) for x in (start, dur, lead)) or min(start, lead) < 0 or dur <= 0:
                    raise ValueError("invalid scene window")
                w = decode(aud)
                tol = 0.05  # covers 2-dp rounding + int(dur*sr) truncation
                if start + dur > len(mix) / sr + tol or lead + len(w) / sr > dur + tol:
                    result["violations"].append({
                        "scene": num,
                        "reason": (f"incomplete scene head/tail coverage "
                                   f"(end={start + dur:.3f}s mix={len(mix) / sr:.3f}s "
                                   f"lead+voice={lead + len(w) / sr:.3f}s slot={dur:.3f}s)")})
                windows = []
                for offset in range(0, len(w), sr):
                    end = min(offset + sr, len(w))
                    begin = max(0, end - sr)
                    tpl = w[begin:end] - w[begin:end].mean()
                    energy = float(np.dot(tpl, tpl))
                    if energy / len(tpl) < 1e-8:
                        continue
                    expected = start + lead + begin / sr
                    lo = max(0, round((expected - 0.75) * sr))
                    hi = min(len(mix), round((expected + 0.75) * sr) + len(tpl))
                    seg = mix[lo:hi]
                    if len(seg) < len(tpl):
                        result["violations"].append({"scene": num, "reason": "short audio window", "at": begin / sr})
                        continue
                    n = len(tpl)
                    size = 1 << (len(seg) + n - 2).bit_length()
                    cc = np.fft.irfft(np.fft.rfft(seg, size) * np.fft.rfft(tpl[::-1], size), size)[n - 1:len(seg)]
                    sums = np.concatenate(([0.0], np.cumsum(seg)))
                    squares = np.concatenate(([0.0], np.cumsum(seg * seg)))
                    variance = np.maximum(0, squares[n:] - squares[:-n] - (sums[n:] - sums[:-n]) ** 2 / n)
                    confidence = cc / np.sqrt(np.maximum(variance * energy, 1e-20))
                    peak = int(np.argmax(confidence))
                    score = float(np.clip(confidence[peak], -1, 1))
                    error = (lo + peak) / sr - expected
                    windows.append({"start": begin / sr, "end": end / sr,
                                    "confidence": round(score, 4), "offset": round(error, 4)})
                    if (score < 0.5 and abs(error) > 0.05) or abs(error) > 0.25:
                        result["violations"].append({"scene": num, "at": begin / sr,
                                                     "confidence": round(score, 4), "offset": round(error, 4)})
                if windows:
                    result["checked"].append({"scene": num, "start": start, "end": start + dur, "windows": windows})
                else:
                    result["skipped"].append({"scene": num, "reason": "no correlatable audio"})
            except Exception as e:
                result["errors"].append({"scene": num, "error": str(e)})
    except Exception as e:
        result["errors"].append({"error": str(e)})
    result["passed"] = bool(result["checked"] and not result["violations"] and not result["errors"]
                            and len(result["skipped"]) <= len(scenes) / 2)
    log(f"  [QA] audio sync: {'PASS' if result['passed'] else 'FAIL'}; "
        f"checked={len(result['checked'])}, skipped={result['skipped']}, "
        f"violations={result['violations']}, errors={result['errors']}", 1)
    return result


def qa_true_peak(output_path):
    proc = subprocess.run(
        [CONFIG["ffmpeg"], "-hide_banner", "-nostats", "-i", str(output_path),
         "-map", "0:a:0", "-af",
         f"loudnorm=I={float(CONFIG.get('lufs_target', -14))}:TP={_TP_LIMIT}:LRA=11:print_format=json",
         "-f", "null", "-"], capture_output=True, text=True, timeout=300)
    if proc.returncode:
        raise RuntimeError(f"True-peak measurement failed: {proc.stderr}")
    blocks = re.findall(r'\{[^{}]*"input_tp"[^{}]*\}', proc.stderr or "", re.DOTALL)
    if not blocks:
        raise RuntimeError("True-peak measurement missing input_tp")
    peak = float(json.loads(blocks[-1])["input_tp"])
    if not math.isfinite(peak):
        raise RuntimeError(f"Invalid measured true peak: {peak}")
    # ponytail: the alimiter lands ON the limit, so a bare `<= _TP_LIMIT`
    # failed its own output by float error. EBU Tech 3341 allows +/-0.3 dB
    # measurement tolerance; 0.1 is well inside that and still catches a
    # real overshoot.
    return {"passed": peak <= _TP_LIMIT + _TP_TOL, "peak_dbtp": peak,
            "limit_dbtp": _TP_LIMIT + _TP_TOL}


def validate_media(path, scenes):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60)
    if probe.returncode:
        raise RuntimeError(f"Media probe failed: {probe.stderr}")
    streams = json.loads(probe.stdout)["streams"]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video or not audio:
        raise RuntimeError("Media requires video and audio streams")
    expected = max(float(s.get("_cap_start", 0)) + float(s["_dur"]) for s in scenes)
    # ffprobe reports "0/0" for a variable-frame-rate stream; dividing by
    # that raised ZeroDivisionError before the isfinite guard below could
    # report it as a media problem.
    num, _, den = str(video.get("avg_frame_rate", "0/0")).partition("/")
    fps = (float(num) / float(den)) if float(den or 0) else 0.0
    duration = float(video.get("duration", 0))
    audio_dur = float(audio.get("duration", 0))
    width, height = get_output_dims()
    want_fps = float(CONFIG.get("fps", 24))
    if (not all(math.isfinite(v) for v in (fps, duration, audio_dur, expected))
            or abs(fps - want_fps) > 0.01 or abs(duration - expected) > 0.1
            or abs(audio_dur - expected) > 0.1
            or (video.get("width"), video.get("height")) != (width, height)
            or audio.get("channels") != 2):
        raise RuntimeError(f"Invalid media streams: video={duration}s {fps}fps, "
                           f"audio={audio_dur}s; expected {expected}s {width}x{height} stereo")
    return {"video_duration": duration, "audio_duration": audio_dur,
            "fps": fps, "width": width, "height": height, "channels": 2}


def _promote_media(final_clip, output, scenes, dialogue_audios):
    final_clip, output = Path(final_clip), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    private = _asset("_temp")
    private.mkdir(exist_ok=True)
    if private.resolve().is_relative_to(Path(CONFIG["output_dir"]).resolve()):
        raise RuntimeError("Candidate directory must be outside the public videos directory")
    candidate = private / f"{output.stem}.{uuid.uuid4().hex}.candidate.mp4"
    try:
        if should_stop():
            raise RuntimeError("Cancelled before publication")
        shutil.copy2(final_clip, candidate)
        _LAST_ASSEMBLY["streams"] = validate_media(candidate, scenes)
        # EBU R128: the true-peak gate existed but integrated loudness was
        # measured and never asserted, so a failed loudnorm pass published a
        # master at whatever level it happened to land on. Gate the NARRATION
        # against the speech target, with room for the music bed the mix adds
        # on top (EBU R128 s4 anchor-based normalisation).
        loud = measure_loudness(candidate)
        if loud is None:
            raise RuntimeError("Loudness measurement failed — cannot verify the mix")
        _LAST_ASSEMBLY["loudness"] = loud
        # Programme = speech anchor + bed, so the ceiling is the target plus
        # the bed's contribution (EBU R128 s4 anchor-based normalisation).
        # The floor is deliberately wide: the BED pushes measured LUFS DOWN,
        # and a 1.0 LU floor false-failed normal mixes. The ceiling is the
        # part that catches real problems (an un-normalised master is far
        # LOUDER, not quieter).
        _hi = float(CONFIG.get("lufs_target", -14)) + 3.0
        _lo = float(CONFIG.get("lufs_target", -14)) - 3.0
        if not (_lo <= loud["lufs"] <= _hi):
            raise RuntimeError(
                f"Loudness gate failed: {loud['lufs']:.1f} LUFS outside "
                f"[{_lo:.0f}, {_hi:.0f}] (target {CONFIG.get('lufs_target')}, "
                f"speech anchor + music bed)")
        # LRA is a WARNING, never a block. Measured on this project's own
        # music beds at the default music_volume, LRA lands at 10.6-17.3 LU —
        # so a hard threshold at 14 aborted renders deterministically on two
        # of three tracks, and which track you get is a mood-hash coin flip.
        # loudnorm's LRA is also a 3-second-window estimate, not a
        # broadcast-grade meter, so it is not trustworthy enough to abort on.
        _LAST_ASSEMBLY["lra"] = loud["lra"]
        _lra_max = float(CONFIG.get("lra_max", 14.0))
        if loud["lra"] > _lra_max:
            log(f"  [WARN] mix is wide (LRA {loud['lra']:.1f} LU > "
                f"{_lra_max:g}) — narration may pump against the bed. "
                f"RAISING music_volume compresses it; lowering it does not.",
                1)
        peak = qa_true_peak(candidate)
        _LAST_ASSEMBLY.update({"true_peak": peak, "tp_peak_db": peak["peak_dbtp"]})
        if not peak["passed"]:
            raise RuntimeError(f"True-peak gate failed: {peak}")
        av = qa_av_sync(candidate, scenes, dialogue_audios)
        _LAST_ASSEMBLY["qa_av_sync"] = av
        if (not isinstance(av, dict) or not av.get("passed") or av.get("errors")
                or av.get("violations") or not av.get("checked")
                or len(av.get("skipped", [])) > len(scenes) / 2):
            raise RuntimeError(f"A/V sync gate failed: {av}")
        if should_stop():
            raise RuntimeError("Cancelled before optional QA")
        # ponytail: someone watches the finished file — code fails loud on
        # black renders, VLM grades frames warn-only. No unwatched publish.
        _LAST_ASSEMBLY["qa_final_black_s"] = qa_final_black(candidate)
        _LAST_ASSEMBLY["qa_final_frames"] = qa_final_frames(candidate, private) or "skipped"
        findings = qa_auto_editor(candidate, scenes, dialogue_audios)
        _LAST_ASSEMBLY["qa_auto_editor"] = "skipped" if findings is None else findings
        if should_stop():
            raise RuntimeError("Cancelled before version rotation")
        rotate_versions(output, preserve_current=True)
        if should_stop():
            raise RuntimeError("Cancelled before atomic publication")
        try:
            candidate.replace(output)
        except OSError as e:
            # Path.replace() is a rename, which fails with EXDEV when
            # _temp and output_dir are on different volumes. All the GPU
            # work is done at this point — copy across, then swap. Narrow
            # to EXDEV: a PermissionError (output open in a player) is a
            # different problem and must not be mislabelled "cross-volume".
            if getattr(e, "errno", None) != errno.EXDEV:
                raise
            log("  [WARN] cross-volume promote — copying instead of renaming", 1)
            staged = output.with_name(output.name + f".{uuid.uuid4().hex}.incoming")
            try:
                shutil.copy2(candidate, staged)
                os.replace(staged, output)
            finally:
                staged.unlink(missing_ok=True)
        _LAST_ASSEMBLY["promoted"] = True
        return output
    except Exception as e:
        _LAST_ASSEMBLY.update({"promoted": False, "error": str(e)})
        final_clip.unlink(missing_ok=True)
        raise
    finally:
        candidate.unlink(missing_ok=True)

# --- Video Assembler v2 (audio-master-first) -------------------------
def pick_music_track(tracks, scenes=None):
    """P3: mood rotation instead of first-alphabetical (deterministic).

    Dominant scene mood hashes to a track index; no scenes = track 0.
    Same script + same music dir = same bed. Quarantine rips never pass here.
    """
    import hashlib as _hl
    tracks = sorted(tracks)
    if not tracks:
        return None
    if not scenes:
        return tracks[0]
    moods = [str(s.get("mood", "") or "").lower() for s in scenes]
    dom = max(sorted(set(moods)), key=moods.count) if moods else ""
    # M1: hash(str) is salted per process — md5 keeps it deterministic.
    idx = int(_hl.md5(dom.encode("utf-8")).hexdigest(), 16) % len(tracks)
    return tracks[idx]


# ponytail: last-assembly receipt (per-scene clip/slot/loops + mix/mux rcs +
# music presence + TP check). export_edl persists it; _produce_inner adds QA
# outputs. Read-only for dashboards; overwritten every assembly.
_LAST_ASSEMBLY = {}
# Keys the CLIPS phase owns. Assembly used to start with
# _LAST_ASSEMBLY.clear(), which silently threw these away — so a still-pan
# downgrade or a fast-tier episode was invisible in the final receipt, which
# is exactly when someone reads the receipt.
_CLIP_PHASE_KEYS = ("cinematic_fallbacks", "video_clips", "cinematic_clips",
                    "h3_tiers", "duplicate_stills", "relay_used", "colour_grade",
                    "clip_still_drift", "clip_still_vlm_flags",
                    # written beside h3_tiers in the clips phase; clearing it
                    # here is why the tier-fallback REVIEW kept firing anyway
                    "quality_requested")


def _clear_assembly_keys():
    """Drop only the previous assembly's keys, keeping clip-phase facts."""
    for k in [k for k in _LAST_ASSEMBLY if k not in _CLIP_PHASE_KEYS]:
        _LAST_ASSEMBLY.pop(k, None)

# One source of truth for the true-peak ceiling shared by the assembler
# limiter and the QA gate. They drifted apart before, which made the gate
# measure a different target than the mix was normalised to.
_TP_LIMIT = -2.0
_TP_TOL = 0.1
# ONE headroom for the lossy encode, shared by BOTH limiter paths.
#
# The limiter works on SAMPLE peak; the gate measures TRUE peak. Inter-sample
# overshoot from the resample/convert runs ~0.15 dB above sample peak, so a
# limiter set exactly at the ceiling publishes a file the gate then rejects
# (a real ch900 render measured -1.85 dBTP against a -1.9 limit and refused to
# promote). But the shipped master is AAC-LC at 128k, and the codec's own
# overshoot is larger than the resample's: a real ch011 mix limited to
# -2.30 dB sample peak measured -0.75 dBTP once muxed (+1.55 dB of codec
# overshoot) and refused to promote; a real ch059 mix limited to -3.50 dB
# sample peak measured -1.75 dBTP (+1.75 dB) and also refused to promote.
#
# So the margin must cover +1.75 dB, and BOTH paths need it: the post-music
# mix (_limit_only) and the narration-only chain (_loudnorm_dual). They used
# to carry DIFFERENT margins (3.0 and 1.5) — and the narrow one is the path a
# book with NO music bed takes, which is why the crash kept coming back on
# exactly those renders. One constant, one concept: a margin that is too
# narrow is worse than no gate at all, because it fails the render it made.
# ponytail: the headroom costs ~1.5 dB of peak gain on the loudest
# transients only; the integrated loudness is set by loudnorm's I target.
_TP_LIMIT_HEADROOM = 3.0


def _limit_only(src_wav, dst_wav, sr=44100, timeout=300):
    """Peak-limit without touching integrated loudness.

    Used for the post-music mix. The old code ran loudnorm TWICE — once on
    the voice to `lufs_target`, then again on voice+bed to the SAME target.
    The second pass can only attenuate everything back down by whatever the
    bed added, which makes the narration level a function of whichever music
    track the mood-hash happened to pick, and it changes discontinuously when
    a chapter has no music at all.

    EBU R128 s4 covers this case: for content with speech as the main
    component, normalise on the SPEECH anchor and apply peak limiting to the
    programme, rather than re-normalising the programme to a target that
    would drag the dialogue around. Narration level is now fixed by the
    voice pass alone.
    """
    try:
        filt = (f"alimiter=limit={_TP_LIMIT - _TP_LIMIT_HEADROOM:.2f}dB"
                f":level=disabled:attack=5:release=50")
        p = subprocess.run(
            [CONFIG["ffmpeg"], "-y", "-v", "error", "-i", str(src_wav),
             "-af", filt, "-c:a", "pcm_s16le", "-ar", str(sr), str(dst_wav)],
            capture_output=True, timeout=timeout)
        return p.returncode == 0 and Path(dst_wav).exists()
    except Exception:
        return False


def measure_loudness(path, lufs_target=None):
    """Integrated loudness (LUFS) + true peak of a file, per EBU R128.

    `input_i` is the gated integrated loudness from ffmpeg's loudnorm
    measurement pass. Returns None if it cannot be measured.
    """
    target = float(lufs_target if lufs_target is not None
                   else CONFIG.get("lufs_target", -14))
    try:
        proc = subprocess.run(
            [CONFIG["ffmpeg"], "-hide_banner", "-nostats", "-i", str(path),
             "-map", "0:a:0", "-af",
             f"loudnorm=I={target}:TP={_TP_LIMIT}:LRA=11:print_format=json",
             "-f", "null", "-"], capture_output=True, text=True, timeout=300)
        blocks = re.findall(r'\{[^{}]*"input_i"[^{}]*\}', proc.stderr or "", re.DOTALL)
        if not blocks:
            return None
        m = json.loads(blocks[-1])
        return {"lufs": float(m["input_i"]),
                "true_peak": float(m["input_tp"]),
                "lra": float(m["input_lra"])}
    except Exception:
        return None


def _loudnorm_dual(src_wav, dst_wav, lufs, sr=44100, timeout=300):
    """Two-pass loudnorm: measure, then apply with measured values + linear.
    Single-pass loudnorm overshoots TP by ~0.3dB on hot mixes (measured
    -1.7dBFS vs -2.0 target) and trips the QA true-peak gate. Returns bool.

    ponytail: the LRA target is raised to the MEASURED input LRA when the
    source is wider than our 11 LU target. ffmpeg reverts linear=true to
    dynamic mode unless "target LRA >= source LRA" (ffmpeg loudnorm docs),
    and a dynamic fallback pumps the narration against the bed — silently,
    because the alimiter downstream still holds true peak, so every gate
    stays green. Narration currently measures LRA ~5.3 so this is a no-op
    today; it is here so a wide-dynamic delivery (a shouted line over a
    whisper) cannot silently lose linear gain later.
    """
    try:
        filt = f"loudnorm=I={lufs}:TP={_TP_LIMIT}:LRA=11:print_format=json"
        # ponytail: no -v error here — loudnorm JSON is INFO-level; error
        # loglevel swallowed it and every run fell back to un-normed master.
        p1 = subprocess.run(
            [CONFIG["ffmpeg"], "-y", "-hide_banner", "-nostats", "-i", str(src_wav),
             "-map", "0:a:0", "-af", filt, "-f", "null", "-"],
            capture_output=True, text=True, timeout=timeout)
        blocks = re.findall(r"\{[^{}]*\"input_tp\"[^{}]*\}", p1.stderr or "", re.DOTALL)
        if p1.returncode or not blocks:
            return False
        m = json.loads(blocks[-1])
        # Every value that reaches the filter string must be finite AND inside
        # ffmpeg's accepted range, or ffmpeg rejects the whole filter and
        # this function returns False — which downgrades the publish to an
        # un-normalized master. Probed against the installed ffmpeg 9.0.1
        # (accepted / rejected):
        #   LRA              1 .. 50      /  0.5, 51, -inf, nan
        #   measured_I      -99 .. 0      /  -100, inf, -inf, nan
        #   measured_TP     -99 .. 99     /  +/-100
        #   measured_LRA      0 .. 99     /  100, 999, -inf, nan
        #   measured_thresh -99 .. 0      /  1, +/-1e9, -inf, nan
        #   offset          -99 .. 99     /  100, +/-1e9, inf, -inf, nan
        # A silent master measures input_i=-inf and target_offset=inf, which
        # is the case that used to reach ffmpeg verbatim.
        _RANGES = {                      # key: (lo, hi, default)
            "input_i": (-99.0, 0.0, None),
            "input_tp": (-99.0, 99.0, _TP_LIMIT),
            "input_lra": (0.0, 99.0, 11.0),
            "input_thresh": (-99.0, 0.0, -70.0),
            "target_offset": (-99.0, 99.0, 0.0),
        }

        def _num(key):
            lo, hi, dflt = _RANGES[key]
            if dflt is None:
                dflt = float(lufs)      # input_i: the value we are targeting
            try:
                v = float(m.get(key, dflt))
            except (TypeError, ValueError):
                return float(dflt)
            if v != v or v in (float("inf"), float("-inf")):
                return float(dflt)       # NaN and +/-inf both reject the filter
            return max(lo, min(hi, v))

        lra_src = _num("input_lra")
        lra_target = max(1.0, min(50.0, lra_src))
        if lra_target <= 11.0:
            lra_target = 11.0  # keep the shipped target when the source is narrower
        # ponytail: the alimiter leaves headroom for the lossy encode, the SAME
        # margin the music path leaves. The gate measures the DELIVERED file,
        # whose decoded true peak sits above the limiter's target by the
        # codec's own overshoot (measured up to +1.75 dB) — with the limiter
        # at or near the ceiling the published file failed its own gate, and
        # the narration-only path was the one that kept failing.
        apply = (f"loudnorm=I={lufs}:TP={_TP_LIMIT}:LRA={lra_target:g}:"
                 f"measured_I={_num('input_i'):g}:"
                 f"measured_TP={_num('input_tp'):g}:"
                 f"measured_LRA={lra_src:g}:"
                 f"measured_thresh={_num('input_thresh'):g}:"
                 f"offset={_num('target_offset'):g}:linear=true,"
                 f"alimiter=limit={_TP_LIMIT - _TP_LIMIT_HEADROOM}dB:level=disabled:"
                 f"attack=5:release=50")
        p2 = subprocess.run(
            [CONFIG["ffmpeg"], "-y", "-v", "error", "-i", str(src_wav),
             "-filter:a", apply, "-c:a", "pcm_s16le", "-ar", str(sr), str(dst_wav)],
            capture_output=True, timeout=timeout)
        return p2.returncode == 0 and Path(dst_wav).exists()
    except Exception:
        return False


def assemble_video_v2(scene_clips, scenes, dialogue_audios, chapter_num, subtitles_path):
    """Audio-master-first assembly. The v1 chain encoded voice 4-5x (mixed
    AAC -> concat AAC -> subs AAC -> loudnorm AAC): every generation costs
    quality plus ~20-50ms padding, and concat joints click every 2.5s.
    v2 builds ONE sample-exact narration wav in numpy, concats video via
    demuxer stream-copy, and encodes AAC exactly once.
    """
    import numpy as _np
    _clear_assembly_keys()   # not clear(): keeps the clip-phase facts
    if should_stop():
        raise RuntimeError("Cancelled before assembly")
    temp = _asset("_temp")
    temp.mkdir(exist_ok=True)
    sr = 44100
    fps = int(CONFIG.get("fps", 24))
    out_w, out_h = get_output_dims()
    if len(scene_clips) != len(scenes) or not scenes:
        raise RuntimeError("Assembly requires one clip per scene")
    for scene in scenes:
        scene["_dur"] = math.ceil(float(scene.get("_dur", CONFIG["scene_duration"])) * fps) / fps
    hook_on = bool(CONFIG.get("hook_enabled", False))
    hook_clip = temp / "hook.mp4"
    if hook_on:
        try:
            hook_clip.unlink()
        except Exception:
            pass
        # ponytail: audio AND video must agree on the hook card. If the card
        # fails to build, disable hook for the whole assembly — no silent
        # 1.5s picture/voice drift either direction.
        hook_on = bool(create_hook_clip(scenes, chapter_num, hook_clip))

    def _read_wav(path):
        p = subprocess.run(
            [CONFIG["ffmpeg"], "-y", "-v", "error", "-i", str(path),
             "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
            capture_output=True, timeout=120).stdout
        return _np.frombuffer(p, dtype=_np.float32)

    # 1. Master narration: lead + voice + tail per scene, sample-exact.
    # ponytail: STEREO master. The old mono master collapsed stereo beds
    # and forced a mono ship; 69s stereo f32 is ~24MB — negligible.
    master = [_np.zeros((0, 2), dtype=_np.float32)]
    t = 0.0
    starts = []
    for i, scene in enumerate(scenes):
        dur = float(scene.get("_dur", CONFIG["scene_duration"]))
        aud = dialogue_audios[i] if i < len(dialogue_audios) else None
        lead = float(scene.get("_lead", 0.2))
        seg = _np.zeros((int(dur * sr), 2), dtype=_np.float32)
        if aud and Path(aud).exists():
            w = _read_wav(aud)
            n = min(len(seg) - int(lead * sr), len(w))
            if n < len(w):
                log(f"  [WARN] Scene {scene.get('scene_number', i + 1)} voice "
                    f"truncated by {len(w) - n} samples", 2)
            if n > 0:
                seg[int(lead * sr):int(lead * sr) + n, 0] = w[:n]
                seg[int(lead * sr):int(lead * sr) + n, 1] = w[:n]
        # ponytail: codex 0.05s fades (assembly.md). v2's numpy concat
        # shipped hard boundaries (clicks); v1 had fades. Silence-heavy
        # slots are unaffected; edge-hugging voice loses the click.
        _fl = int(0.05 * sr)
        if len(seg) > 2 * _fl:
            _ramp = _np.linspace(0, 1, _fl, dtype=_np.float32)[:, None]
            seg[:_fl] *= _ramp
            seg[-_fl:] *= _ramp[::-1]
        master.append(seg)
        starts.append(t)
        t += dur
    if hook_on:
        hook_sil = _np.zeros((int(1.5 * sr), 2), dtype=_np.float32)
        master.insert(1, hook_sil)
        starts = [s + 1.5 for s in starts]
        total = 1.5 + t
    else:
        total = t
    for i, s in enumerate(scenes):
        s["_cap_start"] = round(starts[i], 2)
        s["_cap_dur"] = round(float(s.get("_dur", CONFIG["scene_duration"])), 2)
    subtitles_path = create_ass_subtitles(scenes, temp / "subtitles.ass", dialogue_audios)
    master_wav = temp / "master_voice.wav"
    _np.concatenate(master).astype(_np.float32).tofile(str(master_wav))
    # wav header via ffmpeg (raw -> wav), then loudnorm once (still lossless)
    subprocess.run([CONFIG["ffmpeg"], "-y", "-v", "error", "-f", "f32le",
                    "-ar", str(sr), "-ac", "2", "-i", str(master_wav),
                    "-c:a", "pcm_s16le", str(master_wav.with_suffix(".pcm.wav"))],
                   capture_output=True, timeout=120)
    master_wav = master_wav.with_suffix(".pcm.wav")
    norm_wav = temp / "master_norm.wav"
    lufs = float(CONFIG.get("lufs_target", -14))
    _norm_ok = _loudnorm_dual(master_wav, norm_wav, lufs, sr)
    if not _norm_ok:
        # ponytail: loudnorm failure used to flow silently into a vague
        # "v2 mux failed". Fall back to the un-normed master, loudly.
        log("  [WARN] voice loudnorm failed, using un-normed master", 1)
        norm_wav = master_wav

    # 2. Video: normalize (as v1) then demuxer concat, stream copy, no re-encode.
    fit = vertical_fit_filter(out_w, out_h)
    # ponytail: hook card is a real video segment in v2 (v1 had it, v2
    # silently skipped it and shipped 1.5s of picture/voice drift).
    clip_items = []
    if hook_on and hook_clip.exists():
        clip_items.append((hook_clip, 1.5))
    clip_items += [(clip, float(scene.get("_dur", CONFIG["scene_duration"])))
                   for clip, scene in zip(scene_clips, scenes)]
    normed = []
    _receipt = []  # per-scene assembly receipt (src clip dur/slot/loops/seams)
    for idx, (clip, dur) in enumerate(clip_items):
        norm = temp / f"v2norm_{idx:02d}.mp4"
        # ponytail: computed loops, not fixed -stream_loop 3. Short talking
        # heads used to restart 2-3x per scene (lip seams at every restart);
        # long clips decoded 4x for nothing. loops=0 (flag omitted)
        # when the clip already covers the slot.
        import math as _math
        _cdur = 0.0
        try:
            _pr = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(clip)],
                capture_output=True, text=True, timeout=30)
            _cdur = max(0.0, float((_pr.stdout or "").strip().splitlines()[0]))
        except Exception:
            _cdur = 0.0
        _loops = max(0, int(_math.ceil(dur / _cdur)) - 1) if _cdur > 0 else 3
        _seams = [round(_cdur * k, 2) for k in range(1, _loops + 1)] if _loops else []
        _receipt.append({"clip": Path(clip).name, "clip_dur": round(_cdur, 2),
                         "slot": dur, "loops": _loops, "seams": _seams})
        _loop_args = ["-stream_loop", str(_loops)] if _loops > 0 else []
        cmd = [CONFIG["ffmpeg"], "-y", *_loop_args, "-i", str(clip),
               "-vf", f"{fit},fps={fps}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-an", "-t", str(dur), str(norm)]
        _nrc = subprocess.run(cmd, capture_output=True, timeout=120).returncode
        _receipt[-1].update({"norm_rc": _nrc, "norm_ok": bool(_nrc == 0 and norm.exists())})
        if _nrc == 0 and norm.exists():
            normed.append(norm)
        else:
            # ponytail: fail loud, never silent-fallback. A raw heterogeneous
            # clip in a -c copy demuxer concat risks timebase/codec mismatch
            # in the shipped file; the clip name tells exactly which scene.
            log(f"  [FAIL] scene norm failed ({Path(clip).name}) — aborting, no raw fallback", 1)
            _clear_assembly_keys()
            _LAST_ASSEMBLY.update({"chapter": chapter_num, "failed_at": f"norm:{Path(clip).name}"})
            return None
    lst = temp / "v2list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in normed), encoding="utf-8")
    vcat = temp / "v2cat.mp4"
    vcopy = subprocess.run(
        [CONFIG["ffmpeg"], "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(lst), "-c", "copy", str(vcat)],
        capture_output=True, timeout=300).returncode == 0 and vcat.exists()

    # 3. Mux once: single AAC encode. Optional one-pass music duck.
    music_dir = Path(CONFIG.get("music_dir", ""))
    mus = sorted(list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))) if music_dir.exists() else []
    muxed = temp / "v2mux.mp4"
    _bed = pick_music_track(mus, scenes)
    _mix_rc, _fnorm_rc, _bed_name = None, None, None
    if _bed and vcopy:
        _bed_name = Path(_bed).name
        vol = float(CONFIG.get("music_volume", 0.15))
        # ponytail: ducked + normalize=0 + stereo, mirroring the v1 chain.
        # Plain amix defaults normalize=1 (voice dipped ~6dB, no makeup);
        # mono voice collapsed stereo beds. Ducking honors audio_ducking.
        if bool(CONFIG.get("audio_ducking", True)):
            fc = (f"[1:a]volume={vol},afade=t=in:st=0:d=2,"
                  f"afade=t=out:st={max(0, total - 3)}:d=3,aformat=channel_layouts=stereo[music];"
                  f"[0:a]aformat=channel_layouts=stereo[voice];"
                  f"[music][voice]sidechaincompress=threshold=0.02:ratio=8:"
                  f"attack=20:release=400:makeup=1[ducked];"
                  f"[ducked][voice]amix=inputs=2:duration=first:normalize=0[aout]")
        else:
            fc = (f"[1:a]volume={vol},afade=t=in:st=0:d=2,"
                  f"afade=t=out:st={max(0, total - 3)}:d=3,aformat=channel_layouts=stereo[music];"
                  f"[0:a]aformat=channel_layouts=stereo[voice];"
                  f"[music][voice]amix=inputs=2:duration=first:normalize=0[aout]")
        mix_wav = temp / "v2voice_mix.wav"
        _mix_rc = subprocess.run(
            [CONFIG["ffmpeg"], "-y", "-i", str(norm_wav), "-stream_loop", "-1",
             "-i", str(_bed), "-filter_complex", fc,
             "-map", "[aout]", "-c:a", "pcm_s16le", "-ar", "44100", "-t", str(total),
             str(mix_wav)],
            capture_output=True, timeout=300).returncode
        ok = _mix_rc == 0 and mix_wav.exists()
        if ok:
            # ponytail: post-MUSIC pass is a LIMIT, not a second loudnorm.
            # Re-normalising the mix to the same target the voice already
            # met made the narration level depend on the music track's
            # loudness, and jumped when a chapter had no bed at all. EBU
            # R128 s4 (anchor-based normalisation) is the right shape here:
            # fix the speech level, cap the programme peaks.
            final_voice = temp / "v2voice_final.wav"
            ok2 = _limit_only(mix_wav, final_voice, sr)
            if not ok2:
                log("  [WARN] mix peak-limit failed, using un-limited mix", 1)
            _fnorm_rc = bool(ok2)
            voice_track = str(final_voice) if ok2 else str(mix_wav)
            _LAST_ASSEMBLY["music_bed"] = _bed_name
        else:
            voice_track = str(norm_wav)
            # ponytail: a failed music mix used to fall through to the narration
            # track with NO log line and no gate, so the chapter shipped silently
            # bedless while the receipt still recorded a music_bed. It is a
            # survivable degradation, but it has to be visible — say so, and
            # mark the receipt so the release report can see it too.
            log(f"  [WARN] music mix failed (ffmpeg rc={_mix_rc}); shipping "
                f"narration only — no music bed in this episode", 1)
            _LAST_ASSEMBLY["music_mix_failed"] = True
    else:
        # Load-bearing, do not "simplify" away. With no music bed there is no
        # second pass at all — but that is correct, because `_loudnorm_dual`
        # already appends its own alimiter, so the narration is both at
        # target and peak-safe without a separate limit stage.
        voice_track = str(norm_wav)
    # ponytail: `_tp_peak` was declared here and never assigned, so the receipt
    # wrote "tp_peak_db": null while its own comment advertised a TP check —
    # and null is indistinguishable from "not checked". Measure it here so the
    # receipt is truthful the moment assembly finishes; _promote_media
    # re-measures and overwrites it independently.
    _mux_rc, _tp_peak = None, None
    if vcopy:
        _mux_rc = subprocess.run(
            [CONFIG["ffmpeg"], "-y", "-i", str(vcat), "-i", str(voice_track),
             "-map", "0:v", "-map", "1:a:0", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
             "-t", str(total), str(muxed)],
            capture_output=True, timeout=300).returncode
        ok = _mux_rc == 0 and muxed.exists()
        if ok:
            # ponytail: codex binds assembly — shipped audio is stereo, always
            # (mono master collapsed beds; v2 master is stereo by construction).
            try:
                _ch = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "a:0",
                     "-show_entries", "stream=channels", "-of", "csv=p=0",
                     str(muxed)],
                    capture_output=True, text=True, timeout=60)
                _chans = _ch.stdout.strip().splitlines()
                if _ch.returncode == 0 and _chans and int(_chans[0]) != 2:
                    log(f"  [FAIL] shipped audio is not stereo (channels={_chans[0]}) — aborting", 1)
                    return None
            except Exception as e:
                log(f"  [WARN] stereo assert skipped ({e})", 2)
    else:
        ok = False
    if not ok:
        log("  [WARN] v2 mux failed", 1)
        _clear_assembly_keys()
        _LAST_ASSEMBLY.update({"chapter": chapter_num, "failed_at": "mux", "mux_rc": _mux_rc})
        return None
    final_clip = muxed

    # 4. Captions burn (the single video re-encode after norm).
    # None = this episode had no burn attempted; True/False = attempted, worked
    # or not. In the receipt, so it cannot be confused with either.
    _captions_burned = None
    if (CONFIG.get("subtitles", False) and subtitles_path
            and Path(subtitles_path).exists() and final_clip.exists()):
        sub_clip = temp / "v2subs.mp4"
        # The ASS header already carries the researched size, outline and
        # margin, so force_style is not used here: overriding FontSize=28 (the
        # old hardcoded value) contradicted the file it was styling, and
        # quoting an absolute path into the subtitles filter fails on the drive
        # colon — it feeds the filename to original_size and dies.
        _stage = temp / "subs_burn.ass"
        shutil.copy2(subtitles_path, _stage)
        cmd = [CONFIG["ffmpeg"], "-y", "-v", "error",
               "-i", str(final_clip),
               "-vf", "subtitles=subs_burn.ass",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-pix_fmt", "yuv420p", "-c:a", "copy", str(sub_clip)]
        if subprocess.run(cmd, capture_output=True, timeout=1800,
                          cwd=str(temp)).returncode == 0 and sub_clip.exists():
            final_clip = sub_clip
            _captions_burned = True
            try:
                _stage.unlink()
            except Exception:
                pass
        else:
            # Losing this re-encode costs a full re-mux to notice, so it must be
            # visible like the music/norm/mux siblings — not a silent fallback to
            # the caption-less master. Verified: rc!=0 left no log line, no
            # receipt key, and the export looked clean.
            _captions_burned = False
            log("  [WARN] caption burn failed, shipping without burned-in "
                f"captions ({subtitles_path})", 1)

    platform = str(CONFIG.get("platform", "youtube")).lower()
    output = Path(CONFIG["output_dir"]) / f"ch{chapter_num:03d}_{platform}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    # ponytail: assembly receipt — per-scene clip/slot/loops, mix/mux rcs,
    # music presence, TP check. export_edl persists it; _produce_inner adds
    # QA outputs. No receipt = the R3 observability gap that hid the 9-16
    # music/qa state.
    #
    # Merge, do NOT clear: the clips phase already recorded cinematic_fallbacks,
    # video_clips and h3_tiers here, and clear() silently threw them away —
    # which is why a still-pan downgrade and a fast-tier episode were both
    # invisible in the receipt.
    # Measure the true peak we just wrote, so the receipt carries a real number
    # rather than a null that reads the same as "never checked". Best effort:
    # _promote_media performs the authoritative measurement and its gate, and a
    # measurement failure here must not block assembly.
    try:
        _tp_peak = qa_true_peak(final_clip)["peak_dbtp"]
    except Exception as _tpe:
        log(f"  [WARN] receipt true-peak measurement failed ({str(_tpe)[:80]}); "
            f"the promotion gate will re-measure", 1)
        _tp_peak = None

    _LAST_ASSEMBLY.update({
        "chapter": chapter_num, "total": round(total, 2), "hook_duration": 1.5 if hook_on else 0.0,
        "scenes": _receipt, "music_bed": _bed_name,
        "voice_track": Path(voice_track).name,
        "mix_rc": _mix_rc, "final_limit_rc": _fnorm_rc,
        "vcopy": bool(vcopy), "mux_rc": _mux_rc, "tp_peak_db": _tp_peak,
        "tp_measured": _tp_peak is not None,
        "captions_burned": _captions_burned,
        # The caption standard is only worth anything if it is measurable after
        # the fact, so the cue count and the worst reading speed ride along
        # with the clip receipts. `over_cps_ceiling` > 0 means a narration
        # outran its slot: no caption timing can fix that, the script needs a
        # shorter line.
        "captions": dict(_LAST_CAPTIONS),
        "lufs_target": lufs,
        "output_mb": round(final_clip.stat().st_size / 1048576, 1)
        if final_clip.exists() else 0.0,
    })
    try:
        promoted = _promote_media(final_clip, output, scenes, dialogue_audios)
        srt = Path(subtitles_path).with_suffix(".srt")
        if srt.is_file():
            # ponytail: the episode is already published at this point and the
            # try/finally has no except, so an OSError here — which is the
            # NORMAL Windows failure when a player or editor holds the
            # destination open (json_util documents exactly that) — discarded
            # a finished render and wiped its temp dir. A lost sidecar is not
            # worth losing the video.
            try:
                srt.replace(output.with_suffix(".srt"))
            except OSError as e:
                log(f"  [WARN] could not place the platform .srt ({e})", 1)
        return promoted
    finally:
        if not os.environ.get("KEEP_TEMP"):
            shutil.rmtree(temp, ignore_errors=True)


# --- Video Assembler with Hooks (legacy v1 removed) ----------------

# --- Main Pipeline -------------------------------------------------
def _chapter_pins(chapter_num):
    try:
        data = json.loads((_asset("pinned.json")).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    pins = data.get(f"ch{chapter_num:03d}", {})
    if not isinstance(pins, dict) or any(not isinstance(v, dict) for v in pins.values()):
        raise ValueError("Invalid pinned.json; refusing cache cleanup")
    return {int(k): v for k, v in pins.items()}


def _pinned_still(chapter_num, scene_num, pin):
    images = Path(CONFIG["images_dir"])
    requested = pin.get("still", "")
    if requested:
        path = Path(requested)
        if not path.is_absolute():
            path = images / path
        if not path.is_file():
            raise ValueError(f"Pinned still missing for scene {scene_num}: {path}")
        return path
    candidates = list(images.glob(f"ch{chapter_num:03d}_s{scene_num:02d}_*.png"))
    candidates += list(images.glob(f"ch{chapter_num:03d}_wc{scene_num:02d}_*.png"))
    if not candidates:
        raise ValueError(f"Pinned scene {scene_num} has no still; unpin before rendering")
    # _newest already returns a Path, and returns None only for an empty input —
    # ruled out two lines above. The old Path(...) wrap was a leftover from when
    # it returned str, and would TypeError on None.
    return _newest(candidates)


def _artifact_scene(path, chapter_num):
    match = re.match(rf"ch{chapter_num:03d}_(?:s|wc|h3|ltx|wan|echofail|echo)(\d+)(?=[bc_.]|$)", path.name)
    if match:
        return int(match[1])
    match = re.match(r"(scene|norm|v2norm|mixed)_(\d+)(?=[_.]|$)", path.name)
    if match:
        return int(match[2]) + (match[1] in ("norm", "v2norm"))
    return None


def clean_chapter(chapter_num, pins=None, scene_num=None, keep_valid=False, scenes_cache=None):
    pins = _chapter_pins(chapter_num) if pins is None else pins
    # A pin whose still is already gone protects nothing, so it must not
    # abort the purge — a stale pinned.json entry used to kill the render
    # before any work. The still surfaces as a hard error later at 4604.
    protected = set()
    for n, pin in pins.items():
        try:
            protected.add(_pinned_still(chapter_num, n, pin).resolve())
        except (ValueError, OSError) as e:
            log(f"  [WARN] pin for ch{chapter_num:03d} scene {n} unresolvable ({e}) — purging anyway")
    scenes_cache = scenes_cache or []
    doomed = list(Path(CONFIG["images_dir"]).glob(f"ch{chapter_num:03d}_*"))
    doomed += list((_asset("voice")).glob(f"ch{chapter_num:03d}_*"))
    doomed += list((_asset("_temp")).glob(f"ch{chapter_num:03d}_*.mp4"))
    doomed += list((_asset("_temp")).glob(f"ch{chapter_num:03d}_*.ass"))
    # ponytail: JUDGE THE STILLS BEFORE DELETING ANYTHING. The purge used to
    # walk the glob in order, so a still's .cache.json was deleted BEFORE the
    # .png beside it was asked whether it was valid - and with its note gone
    # the answer was "unverified -> keep". An invalid still therefore
    # SURVIVED every purge, was re-adopted with a fresh fingerprint, and the
    # fingerprint could never invalidate it again: a novel's new art
    # direction, or a changed bible, silently stopped reaching the stills.
    # A kept still must also keep its PROVENANCE, or the next pass launders
    # a frame made under the OLD prompt into "verified".
    _keep_paths, _stale_paths = set(), set()
    if keep_valid and scenes_cache:
        _by_num = {s.get("scene_number"): s for s in scenes_cache}
        for _p in Path(CONFIG["images_dir"]).glob(f"ch{chapter_num:03d}_*.png"):
            _s = _by_num.get(_artifact_scene(_p, chapter_num))
            if _s is None:
                continue
            try:
                _ok = _still_cache_valid(_p, _s)
            except OSError:
                _ok = False
            _sidecars = {_p.with_suffix(".cache.json"), _p.with_suffix(".face.json")}
            if _ok:
                _keep_paths.add(_p)
                _keep_paths |= _sidecars
            else:
                _stale_paths.add(_p)
                _stale_paths |= _sidecars
    n = 0
    for path in doomed:
        number = _artifact_scene(path, chapter_num)
        if number in pins or path.resolve() in protected:
            continue
        if scene_num is not None and number != scene_num:
            continue
        if path in _keep_paths:
            continue
        if path in _stale_paths:
            if path.is_file():
                path.unlink()
                n += 1
            continue
        # ponytail: keep work that is still valid.
        #   .png -> content-addressed sidecar must validate. A still with no
        #            parseable sidecar is still purged.
        #   .mp4 -> H3 raw clip cache. There is no sidecar for these, and the
        #            clip phase re-probes fps itself (regenerating sub-20fps
        #            pre-enhance clips), so mere existence is enough to justify
        #            keeping ~3h of GPU work that a late-phase failure would
        #            otherwise throw away.
        if keep_valid:
            ext = path.suffix.lower()
            if ext == ".mp4":
                # A clip killed mid-write is a few KB of moov-less mp4. It
                # must not survive a purge, because the fps guard cannot catch
                # it either: ffprobe fails (rc != 0), which the guard reads as
                # "not stale", so it would be copied straight into the 24fps
                # timeline.
                try:
                    if path.stat().st_size > 1024:
                        continue
                except OSError:
                    pass
            if ext == ".png":
                sc = None
                for s in scenes_cache:
                    if s.get("scene_number") == number:
                        sc = s
                        break
                if sc is not None and _still_cache_valid(path, sc):
                    continue
        if path.is_file():
            path.unlink()
            n += 1
    log(f"Purged {n} cached files for ch{chapter_num:03d}")


def _invalidate_scene(chapter_num, scene_num, scenes, pins):
    if scene_num not in {s["scene_number"] for s in scenes}:
        raise ValueError(f"Scene {scene_num} does not exist")
    if scene_num in pins:
        raise ValueError(f"Scene {scene_num} is pinned; unpin it before redo")
    clean_chapter(chapter_num, pins=pins, scene_num=scene_num)


@contextmanager
def _render_ownership():
    """Exclusive render ownership, in-process AND cross-process.

    ponytail: the cross-process leg used to let the raw OSError escape.
    msvcrt.locking(..., LK_NBLCK) and fcntl.flock(LOCK_EX|LOCK_NB) both raise
    OSError on contention — that is documented behaviour, not a malfunction —
    so "a render is already running" surfaced as
    `PermissionError: [Errno 13] Permission denied`. The dashboard's launch
    wrapper caught it, wrote the traceback to render_crash.log, and the
    supervisor turned that into a CRITICAL ledger row reading "new crash".
    ep.preflight then blocks on any unresolved critical for 7 days, so clicking
    Start during a live render took the whole pipeline down for a week.

    Translate contention into the same RuntimeError the in-process leg raises.
    Any OTHER OSError is still a real fault and propagates.
    """
    if not _RENDER_LOCK.acquire(blocking=False):
        raise RuntimeError("Another render is already running")
    f = None
    locked = False
    owned = False
    try:
        f = open(ROOT / "render.lock", "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(64)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError(
                "Another render is already running (render.lock is held by "
                f"another process: {e})") from e
        locked = True
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()).encode("ascii"))
        f.flush()
        owned = True
        yield
    finally:
        try:
            if f is not None:
                try:
                    if owned:
                        f.seek(0)
                        f.truncate()
                        f.flush()
                finally:
                    try:
                        if locked:
                            if os.name == "nt":
                                f.seek(64)
                                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                            else:
                                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    finally:
                        f.close()
        finally:
            _RENDER_LOCK.release()


def _run_lock_held():
    try:
        with _render_ownership():
            return False
    except (OSError, ValueError, RuntimeError):
        return True


def produce_episode(chapter_num, preserve=False, redo_scene_num=None, *,
                    prepare_services=False, reset_stop=False, requested_at=None,
                    hook_enabled=None, novel=None):
    with _novel_context(novel):
        return _produce_episode_body(chapter_num, preserve, redo_scene_num,
                                     prepare_services=prepare_services,
                                     reset_stop=reset_stop,
                                     requested_at=requested_at,
                                     hook_enabled=hook_enabled)


def _produce_episode_body(chapter_num, preserve=False, redo_scene_num=None, *,
                          prepare_services=False, reset_stop=False, requested_at=None,
                          hook_enabled=None):
    global _PROGRESS_OWNER
    requested_at = time.time() if requested_at is None else float(requested_at)
    if not math.isfinite(requested_at) or requested_at > time.time():
        raise ValueError("requested_at must be a finite past Unix timestamp")
    with _render_ownership():
        _PROGRESS_OWNER = (os.getpid(), threading.get_ident())
        _RUN_T0["t"] = time.time()
        _PHASE_T0.clear()
        _LAST_ASSEMBLY.clear()
        old_hook = CONFIG.get("hook_enabled", False)
        try:
            if hook_enabled is not None:
                CONFIG["hook_enabled"] = bool(hook_enabled)
            if reset_stop:
                try:
                    if STOP_FILE.stat().st_mtime < requested_at:
                        STOP_FILE.unlink()
                except FileNotFoundError:
                    pass
            if should_stop():
                set_progress("stopped", detail="Stop requested before start")
                return None
            _write_pid()
            _keep_awake_on()
            (ROOT / "render_chapter").write_text(json.dumps(
                {"novel": current_novel_id(), "chapter": chapter_num}))
            require_phase("ops", [("services up", ensure_services(auto_start=prepare_services)[0])])
            if should_stop():
                return None
            result = _produce_inner(chapter_num, preserve=preserve, redo_scene_num=redo_scene_num)
            if result is None:
                set_progress("stopped" if should_stop() else "error", detail="Render ended without output")
            return result
        except Exception:
            set_progress("error", detail="Render failed; see task error or render_crash.log")
            raise
        finally:
            try:
                if should_stop():
                    set_progress("stopped", detail="Stopped by user")
                _keep_awake_off()
                _clear_pid()
            finally:
                CONFIG["hook_enabled"] = old_hook
                _PROGRESS_OWNER = None
                _RUN_T0["t"] = 0.0
                _PHASE_T0.clear()


def gate_script_review(scenes, chapter_num):
    """Automatic script-review gate: block on structural errors, warn-only on
    viral/retention notes (taste never fails a render). Returns review score."""
    from pipeline.script_reviewer import review_script as _review_script
    rev = _review_script(scenes, chapter_num, scenes_dir=_asset("scenes"))
    for iss in rev.issues:
        if iss.field in ("hook", "retention", "pacing", "readability", "language"):
            log(f"  [VIRAL] s{iss.scene}: {iss.message}", 1)
    if not rev.valid:
        errs = "; ".join(i.message for i in rev.issues if i.severity == "error")
        raise RuntimeError(f"script review failed: {errs}")
    # ponytail: the one structural failure warnings can't express — a
    # list-shaped script with zero voice lines would render a silent film.
    if isinstance(scenes, list) and scenes and not any(
            (s.get("narration") or s.get("dialogue")) if isinstance(s, dict) else False
            for s in scenes):
        raise RuntimeError("script review failed: no scene has narration or dialogue")
    _report_hook_health(scenes)
    log(f"  Script review: score {rev.score}/100 ({len(rev.issues)} notes)", 1)
    return rev.score


def _report_hook_health(scenes):
    """Say the hook numbers out loud, as a rate rather than 24 separate lines.

    ch2 produced "[VIRAL] No re-hook at the tail" on 24 of 25 scenes and the
    log read like a list of small notes rather than "the retention spine is
    missing from this episode". 55% of viewers are gone by 60s; a line that
    ends without a forward cue is exactly where they leave. Warn-only, because
    taste must not fail a render, but the ratio is the signal.
    """
    from pipeline import viral_narration as _vn
    texts = [(s.get("narration") or s.get("dialogue") or "").strip()
             for s in scenes] if isinstance(scenes, list) else []
    texts = [t for t in texts if t]
    if len(texts) < 3:
        return
    body, last = texts[:-1], texts[-1]
    hooked = sum(1 for t in body if _vn.has_open_loop(t))
    rate = hooked / len(body)
    hscore, htrig = _vn.score_hook(texts[0])
    log(f"  Hook: score {hscore}/100 (trigger: {htrig or 'none'}), "
        f"re-hooks on {hooked}/{len(body)} lines = {rate:.0%}", 1)
    if rate < 0.5:
        log(f"  [WARN] Only {rate:.0%} of lines end on a DETECTED forward cue. "
            f"The detector is a keyword proxy - it sees 'never', 'until' and "
            f"'?' but not a line that withholds a name, a number or a reason, "
            f"so this rate is a floor, not a verdict. Read the lines before "
            f"acting on it.", 1)
    if texts[-1] and not _vn.has_open_loop(texts[-1]):
        log("  [WARN] Final scene answers instead of asking — end on the "
            "question so the next episode has a reason to exist.", 1)


def chapter_files(base=None):
    """Every chapter text file in a chapters folder, .txt or .md.

    Novels arrive in both shapes (a plain .txt export and a .md download), and
    a .md-only novel used to look like an empty folder: the chapter list was
    empty, the render said "Chapter N not found", and nothing said why.
    """
    base = Path(base) if base else Path(CONFIG["chapters_dir"])
    out = []
    for pattern in ("*.txt", "*.md"):
        out.extend(base.glob(pattern))
    return sorted(out, key=lambda p: p.name)


def find_chapter_file(chapter_num, chapters_dir=None):
    """Resolve a chapter NUMBER to its text file inside a novel's chapters/.

    Scoped to one novel on purpose: two novels may both ship a "Chapter 1",
    and a shared chapters/ folder made the lookup silently pick whichever
    file sorted first.
    """
    base = Path(chapters_dir) if chapters_dir else Path(CONFIG["chapters_dir"])
    for cand in chapter_files(base):
        m = re.search(r"(\d+)", cand.stem)
        if m and int(m.group(1)) == int(chapter_num):
            return cand
    return None


def _produce_inner(chapter_num, preserve=False, redo_scene_num=None):
    try:
        LIVE_PROGRESS.unlink()
    except Exception:
        pass
    set_progress("starting", 0, 1, f"Chapter {chapter_num}")
    log(f"\n{'=' * 50}")
    log(f"  PRODUCING CHAPTER {chapter_num}")
    log(f"{'=' * 50}")
    log(f"  Studio rules {RULES_VERSION} loaded ({len(RULES)} areas)")

    chapters_dir = Path(CONFIG["chapters_dir"])
    chapter_path = find_chapter_file(chapter_num, chapters_dir)
    if chapter_path is None:
        log(f"Chapter {chapter_num} not found in {chapters_dir}")
        return None
    log(f"Source: {chapter_path.name}")

    # Phase 0: Preflight (failure catalog) + purge stale artifacts
    note_phase("0")
    log("\nPhase 0: Preflight + purge...")
    if should_stop():
        return None
    if not preflight(chapter_num):
        log("Preflight failed, aborting before GPU spend.")
        return None

    # Phase 1: Script Generation (LLM server owns VRAM here)
    note_phase("1")
    set_progress("script", 0, 1, "Breaking chapter into scenes")
    log("\nPhase 1: Generating script...")
    if not ensure_llm():
        log("LLM server required for new scripts. Aborting.")
        return None
    dp = None  # director's package (Phase 1.5)
    scenes_file = Path(CONFIG["scenes_dir"]) / f"ch{chapter_num:03d}.json"
    scenes = None
    if scenes_file.exists():
        _cached = _load_cached_scenes(scenes_file)
        # ponytail: codex binds CACHED scripts too — shot diversity applies
        # to file-loaded scripts, not just fresh gens (25x-static ch001
        # shipped because the cache path skipped it). Deterministic rotation.
        _cached = _enforce_shot_diversity(_cached)
        try:
            _ctxt = Path(chapter_path).read_text(encoding="utf-8")
        except OSError:
            _ctxt = ""
        _bad = _validate_scenes(_cached, scene_target_count(len(_ctxt)))
        if _bad:
            # ponytail: a cached script that cannot be voiced is not usable.
            # ch017's cached script had a blank scene and the render died at
            # the voice gate - with the cache in place there was no way out
            # except deleting the file by hand. Regenerate instead.
            log(f"  Cached script is unusable ({_bad}) - regenerating", 1)
        else:
            scenes = _cached
            log(f"  Using existing {len(scenes)} scenes", 1)
    if scenes is None:
        with open(chapter_path, "r", encoding="utf-8") as f:
            chapter_text = f.read().strip()
        scenes = break_into_scenes(chapter_text, chapter_num)
        log(f"  Generated {len(scenes)} scenes", 1)

    # Phase 1.5: Director's package (creative bridge between novel and scenes)
    if _PIPELINE_UPGRADE:
        note_phase("1.5")
        log("\nPhase 1.5: Director's package...")
        dp = load_director_package(Path(CONFIG["scenes_dir"]), chapter_num)
        if dp and dp.get("version") != "placeholder":
            log(f"  Using existing director's package (version {dp.get('version', '?')})", 1)
        else:
            try:
                chapter_text_for_dp = ""
                if chapter_path:
                    with open(chapter_path, "r", encoding="utf-8") as f:
                        chapter_text_for_dp = f.read().strip()
                dp, dp_errors = _gen_dp(chapter_text_for_dp, chapter_num, scenes,
                                        llm_chat_fn=llm_chat, char_bible=CHAR_BIBLE)
                if dp_errors:
                    log(f"  [WARN] Director package validation issues: {dp_errors}", 1)
                if dp:
                    save_director_package(dp, Path(CONFIG["scenes_dir"]), chapter_num, force=True)
                    log("  Director's package generated", 1)
                else:
                    log("  [WARN] Director package generation failed, continuing without", 1)
            except Exception as e:
                log(f"  [WARN] Director package skipped: {e}", 1)

    # Phase 1b: identity prep — the bible is the identity source, the dossier
    # page is the owner's reference. P2: shot-plan + canon snapshot before any
    # GPU spend.
    scenes = shot_plan_pass(scenes)
    # ponytail: the director's plan knows who each scene is ABOUT; the
    # scenes did not. Copy it on before the shot plan and the canon snapshot
    # so the still prompt stops guessing the subject out of free text.
    _n_focal = _apply_director_plan(scenes, dp)
    if _n_focal:
        log(f"  Focal character applied to {_n_focal} scene(s) from the "
            "director's plan", 1)
    write_canon_snapshot(chapter_num, scenes)
    llm_narrative_check(scenes)
    if should_stop():
        log("Stopped by user.")
        return None
    _cast = sorted({c for s in scenes for c in s.get("characters", [])})
    # Auto-describe cast the bible doesn't know (Maret/Rendell in ch3 rendered
    # as Edvyn — his LoRA hijacked shots with no tags to counter it). One LLM
    # call while the server is still up; the still prompts then pick up their
    # tags, which IS the identity source.
    _new_cast = _learn_new_characters(_cast, chapter_path, chapter_num)
    # ponytail: this was len(open(chapter_path, encoding="utf-8").read()) —
    # a handle opened, fully read, and never closed, plus a full re-read of the
    # chapter purely to count characters. `chapter_text` only exists on the
    # fresh-script path, so read it properly here and reuse it below.
    try:
        _chapter_chars = len(Path(chapter_path).read_text(encoding="utf-8"))
    except OSError:
        _chapter_chars = len(scenes)
    _tgt = scene_target_count(_chapter_chars) if chapter_path else len(scenes)
    _bad = _validate_scenes(scenes, _tgt)
    # ponytail: codex binds the script — diversity is enforced on BOTH fresh
    # and cached loads above, so this gate measures rather than hopes.
    _moves = [str(s.get("camera_move", "")) for s in scenes]
    _top = max([_moves.count(m) for m in set(_moves)] or [0])
    require_phase("script", [
        ("script valid", _bad is None),
        (f"shots diversified (top move {_top}/{len(scenes)})",
         len(scenes) < 3 or _top / max(1, len(scenes)) <= 0.4),
    ])
    # ponytail: automatic review gate — structural errors fail loud before
    # any GPU spend; viral score only ever warns (see gate_script_review).
    try:
        gate_script_review(scenes, chapter_num)
    except RuntimeError:
        raise
    except Exception as e:
        log(f"  [WARN] script review skipped ({e})", 1)
    if redo_scene_num is not None:
        _invalidate_scene(chapter_num, redo_scene_num, scenes,
                          _chapter_pins(chapter_num))
        log(f"Redo scene {redo_scene_num}: invalidated its cache", 1)
    elif not preserve:
        # ponytail: a full purge made every retry re-run the whole render.
        # clean_chapter() deleted all cached stills BEFORE checking them, so a
        # chapter that failed at the clips phase (2h in) lost 90 minutes of
        # finished image work and had to regenerate it. Stills are content-
        # addressed with a sidecar fingerprint; drop only the ones that no
        # longer validate, and keep the rest. `preserve=True` still forces a
        # clean run.
        clean_chapter(chapter_num, keep_valid=True, scenes_cache=scenes)
    # P0: the stills use Comfy — the LLM server must be stopped BEFORE any Comfy work
    # (16GB ceiling: Qwen14B 9GB + Qwen-Image-2.1 GGUF can never co-reside).
    unload_llm()
    # The learned cast needs a dossier page too, not just the on-screen cast:
    # a character the bible learned this run but the script never put on camera
    # (ch020: Aldric Grent, Corvin Hest) would get a bible entry and no page.
    # (Built AFTER the LoRA pass, so a card made for a cold-start character
    # shows up on the page the owner opens.)
    # The gate must demand exactly what the builder attempts. A character the
    # script invents and the bible has never heard of (ch2 shipped "hartwell",
    # who is not in characters.json) has no identity to carry, and that is
    # stated plainly below rather than crashing the render 50 minutes in.
    #
    # So: demand an identity source only for characters the bible can actually
    # describe. An incidental unnamed character then renders from its tags
    # alone, which is an identity-drift risk stated plainly rather than a crash.
    _bible_keys = set((CHAR_BIBLE.get("characters", {}) or {}))
    _need_id, _no_bible = [], []
    for c in list(_cast) + list(_new_cast or []):
        _key = c.lower().replace(" ", "_")
        if _key in _need_id or _key in _no_bible:
            continue
        (_need_id if _key in _bible_keys else _no_bible).append(c)
    if _no_bible:
        log(f"  [WARN] {len(_no_bible)} character(s) not in the bible: "
            f"{', '.join(_no_bible)} - no prompt tags, identity may drift "
            f"across shots. Add danboru_tags to characters.json to fix.", 1)
    require_phase("characters", [(f"identity: {c}", _character_identity_ready(c))
                                for c in _need_id])
    rules_adherence(scenes)

    # Phase 1c: LoRA for any cast member missing one — trained now, before a
    # single still, so the episode that introduces a character actually uses
    # his LoRA (ch11 shipped unknown_man from bible tags alone because his LoRA
    # only landed an hour after the render). After the identity gate, so a bad
    # bible entry never costs two hours of training. Covers manually-added
    # bible entries too (ch21 mrs_fenwick), not just the ones learned this run.
    ensure_cast_loras(_cast)
    ensure_character_pages(_cast)
    ensure_character_pages(_new_cast)

    # Phase 2: TTS (audio first — every later phase times off this)
    note_phase("2")
    # ponytail: START it, then gate. Phase 3 kills TTS to free ~3GB for H3, so
    # by here it is genuinely down and no watchdog/supervisor revives it during
    # a render. Calling the PROBE (ensure_tts) instead of a launcher is what
    # killed two renders at this gate; start_tts_server() relaunches it.
    free_comfyui_vram()
    require_phase("voice", [("TTS server up", start_tts_server())])
    log("\nPhase 2: Generating voice narration...")
    # The gate above already raised unless TTS is serving, so this is belt and
    # braces -- never ship silence (P3 ops: the old path warned "continuing
    # SILENT" and then aborted at the voice-fail check a few lines later).
    if check_tts_server():
        voice_dir = _asset("voice")
        voice_dir.mkdir(exist_ok=True)
        narration_only = bool(CONFIG.get("narration_only", True))
        dialogue_audios = []

        for i, scene in enumerate(scenes):
            scene_num = scene.get("scene_number", i + 1)
            if should_stop():
                log("Stopped by user."); _clear_pid(); return None
            set_progress("voice", i, len(scenes), f"Scene {scene_num}")
            narration = scene.get("narration", "")
            dialogue = scene.get("dialogue", "")

            if narration_only:
                text_to_speak = narration or dialogue
            else:
                characters_in_scene = scene.get("characters", [])
                is_dialogue_scene = bool(dialogue) and not narration
                text_to_speak = dialogue if is_dialogue_scene else narration
            if not text_to_speak:
                dialogue_audios.append(None)
                continue

            audio_path = voice_dir / f"ch{chapter_num:03d}_s{scene_num:02d}.wav"
            text_clean = clean_tts_text(text_to_speak)
            if not text_clean:
                dialogue_audios.append(None)
                continue
            log(f"  Scene {scene_num}: TTS...", 2)

            # ponytail: the TTS server can die MID-PHASE (this box lost the
            # whole service stack four times in one day). Every later scene
            # then failed with "server not running" and the render aborted at
            # the voice gate - losing hours of stills and clips to one dead
            # process. Bring it back once and carry on.
            if not check_tts_server():
                log(f"  Scene {scene_num}: TTS server is down - restarting it", 1)
                if not start_tts_server():
                    log(f"  Scene {scene_num}: TTS server did not come back", 1)

            if narration_only:
                # ponytail: character-as-narrator. On-screen character speaks
                # the narration (voice matches the lips EchoMimic animates).
                # Faceless shots keep the single narrator voice.
                chars = scene.get("characters", [])
                if chars:
                    ok = generate_tts(text_clean, audio_path,
                                      is_narration=False, character_name=chars[0])
                else:
                    ok = generate_tts(text_clean, audio_path,
                                      is_narration=True, character_name=None)
            elif is_dialogue_scene and characters_in_scene:
                character_name = characters_in_scene[0]
                ok = generate_tts(text_clean, audio_path,
                                  is_narration=False, character_name=character_name)
            else:
                ok = generate_tts(text_clean, audio_path,
                                  is_narration=True, character_name=None)
            dialogue_audios.append(audio_path if ok else None)

        # Passage synthesis is NOT wired in. Kept, tested, and documented because
        # the diagnosis is sound and the code works - but it does not pay yet.
        #
        # The defect it targets is real and measured: a per-scene call restarts the
        # model's prosody, so pitch jumps at every sentence boundary. Mean
        # |pitch step| across a boundary is 4.84 semitones, against 1-2 for a human
        # speaker. Generating a run of scenes as ONE continuous read fixes that -
        # and the passage itself is excellent: 4.3% WER as a single file, with
        # pitch steps down to 2.27 semitones.
        #
        # The problem is splitting it back into per-scene files. Energy-based cut
        # points land between the wrong sentences (52.4% WER). Whisper's own
        # segment timestamps align it almost perfectly when the segment count
        # matches, but on a fresh render it often does not, and the fallback is
        # still 19.3% - worse than rendering each scene separately (~14%).
        #
        # So: pitch improved, intelligibility got worse. A rough seam is far less
        # noticeable than a chopped word, and the chopped word is the one a viewer
        # would actually complain about. Not shipping a net loss. Force-aligning
        # the passage is the missing piece, not more heuristics.
        #
        # Enable with PASSAGE_ENABLED = True once alignment is reliable.
        _PASSAGE_ENABLED = False
        if _PASSAGE_ENABLED:
            _plain = [i for i, sc in enumerate(scenes)
                      if not sc.get("characters")
                      and clean_tts_text((sc.get("narration") or "").strip() or "")]
            _need = [i for i in _plain if dialogue_audios[i] is None]
            for _start in range(0, len(_need), PASSAGE_SCENES):
                _run = _need[_start:_start + PASSAGE_SCENES]
                _tx = [clean_tts_text(scenes[i]["narration"].strip()) for i in _run]
                if sum(len(t) for t in _tx) > PASSAGE_MAX_CHARS:
                    continue
                log(f"  Passage: {len(_run)} scenes as one continuous read", 2)
                _ps = [voice_dir / f"ch{chapter_num:03d}_s{scenes[i].get('scene_number', i+1):02d}.wav"
                       for i in _run]
                _res = generate_tts_passage(_tx, [str(p) for p in _ps])
                for _i, _ok in zip(_run, _res):
                    if _ok:
                        dialogue_audios[_i] = _ps[_run.index(_i)]

    # ponytail: voice-fail assert BEFORE assembly (a dead TTS once shipped
    # 125s of silence). Every scene voiced or abort — never ship silence.
    _voiceless = [scenes[i].get("scene_number", i + 1) for i, a in enumerate(dialogue_audios) if a is None]
    if _voiceless:
        log(f"[FAIL] Voiceless scenes {_voiceless} — aborting (never ship silence)")
        _clear_pid()
        return None
    # ponytail: codex binds voice — every wav carries a sidecar holding the
    # EXACT sent text (voice.md). A wav without a non-empty sidecar is
    # untrusted cache: abort instead of shipping mystery audio.
    _untrusted = []
    for i, a in enumerate(dialogue_audios):
        if a is None:
            continue
        try:
            _txt = Path(str(a)).with_suffix(".txt")
            if not (_txt.exists() and _txt.read_text(encoding="utf-8").strip()):
                _untrusted.append(scenes[i].get("scene_number", i + 1))
        except Exception:
            _untrusted.append(scenes[i].get("scene_number", i + 1))
    require_phase("voice", [(f"voice sidecars fresh ({_untrusted or 'all'})",
                             not _untrusted)])

    # Phase 2b: Scene durations from narration (kills dead air)
    note_phase("2b")
    log("\nPhase 2b: Timing scenes to narration...")
    durs = compute_scene_durations(scenes, dialogue_audios)
    _hd = 1.5 if bool(CONFIG.get("hook_enabled", False)) else 0.0
    log(f"  Total narration-driven runtime: {_hd + sum(durs):.1f}s", 1)

    # ponytail: TTS is finished, so hand its ~3GB back immediately. The image
    # and clip phases both need the whole card, and nothing downstream reads
    # the server again — lipsync and assembly work off the wavs on disk.
    try:
        if check_tts_server():
            log("  [MEM] evicting TTS server for the GPU phases (~3GB)", 1)
            _stop_tts_server()
    except Exception as _e:
        log(f"  [MEM] TTS evict skipped: {str(_e)[:60]}", 2)
    # Phase 3: Reference images (ComfyUI owns VRAM now)
    note_phase("3")
    log("\nPhase 3: Generating reference images...")
    unload_llm()
    _comfy_up = bool(ensure_comfyui())
    if not _comfy_up:
        log("ComfyUI required for images. Aborting.")
        return None
    # ponytail: this passed the literal True, so the log read "gate OK:
    # comfyui up" for a check that was never made here. The check itself did
    # run two lines above, so this was a lying receipt rather than a missing
    # gate — now it reports the value that was actually verified.
    require_phase("images", [("comfyui up", _comfy_up)])
    ensure_vlm()
    note_visual_review()
    images_dir = Path(CONFIG["images_dir"])
    images_dir.mkdir(exist_ok=True)
    # ponytail: fixed-size + None placeholders — append-only lists shifted
    # every later scene when one still failed. Index == position, always.
    scene_images = [None] * len(scenes)

    pins = _chapter_pins(chapter_num)
    for si, scene in enumerate(scenes):
        # m5: positional index (was scenes.index(scene) — O(n^2), wrong on dupes).
        scene_num = scene.get("scene_number", si + 1)
        if scene_num in pins:
            try:
                pinned = _pinned_still(chapter_num, scene_num, pins[scene_num])
                scene_images[si] = pinned
                log(f"  Scene {scene_num}: pinned still {pinned.name}", 2)
                continue
            except Exception as e:
                raise RuntimeError(f"Pinned scene {scene_num} unavailable: {e}")
        if should_stop():
            log("Stopped by user."); _clear_pid(); return None
        set_progress("images", si, len(scenes), f"Scene {scene_num}")
        gpu_heat_guard()
        # A-images only (exclude beat-2 "b" and companion "c" images)
        existing = [p for p in images_dir.glob(f"ch{chapter_num:03d}_s{scene_num:02d}_*.png")
                    if f"s{scene_num:02d}b_" not in p.name and f"s{scene_num:02d}c_" not in p.name]
        if not existing:
            existing = list(images_dir.glob(f"ch{chapter_num:03d}_wc{scene_num:02d}_*.png"))
        if existing and all(_still_cache_valid(p, scene) for p in existing):
            # ponytail: a single cached still needs no selector. Asking a VLM to
            # choose between one option cost ~3 min per scene on every resume —
            # 75 min to re-pick index 0 of 1 — and could not change the answer:
            # its own failure path returns this same file. Real selection still
            # runs whenever regeneration left 2+ candidates to choose between.
            if len(existing) == 1:
                scene_images[si] = existing[0]
                log(f"  Scene {scene_num}: reference picked (1/1, single cached "
                    f"still, no re-grade)", 2)
            # C1 — the selectors were wired to the TEXT model
            # (CONFIG["llm_model"]) while the LLM server is stopped for VRAM, so
            # they could raise or hang 120s per still. Only run when a real
            # VL model is present; otherwise keep the newest A-still.
            elif vlm_available():
                try:
                    sel = _reference_selector()
                    scene_desc = scene.get("image_prompt", "") or scene.get("narration", "")
                    ref_texts = [f"Scene {scene_num} reference {i}" for i in range(len(existing))]
                    idx, _ = sel.select(
                        frame_description=scene_desc,
                        ref_image_paths=[str(p) for p in existing],
                        ref_texts=ref_texts,
                        candidate_paths=[str(p) for p in existing],
                        candidate_texts=ref_texts,
                    )
                    scene_images[si] = existing[idx]
                    log(f"  Scene {scene_num}: reference picked ({idx+1}/{len(existing)})", 2)
                except Exception as e:
                    log(f"  [WARN] reference select unavailable ({str(e)[:80]}), "
                        f"using newest still", 2)
                    scene_images[si] = _newest(existing)
            else:
                scene_images[si] = _newest(existing)
            # Record provenance for whichever candidate was chosen, in every
            # branch. Doing it only in the 1-candidate path left the VLM-
            # selected still unrecorded, so the next resume took the 2+
            # branch again and paid the selector a second time.
            if scene_images[si] is not None:
                _still_cache_valid(scene_images[si], scene, adopt=True)
            # M1: cached stills skip the reframe retry (no GPU spend on hits),
            # but still get a warn-only face flag for QA visibility.
            if scene.get("characters"):
                _cok, _cwhy = _face_flag_for(scene_images[si], scene_num)
                if not _cok:
                    scene["_face_flag"] = f"cached: {_cwhy}"
                    log(f"  Scene {scene_num}: cached still head-flagged ({_cwhy}) — kept, flagged", 2)
        else:
            log(f"  Scene {scene_num}: generating reference...", 2)
            img = generate_image(scene, scene_num, chapter_num)
            # One retry on a different seed. A single ComfyUI hiccup used to
            # leave scene_images[si] = None, which then failed the
            # "stills complete on disk" gate and threw away every other
            # scene's GPU work. Same pattern as the H3 clip and TTS retries.
            if not img:
                log(f"  Scene {scene_num}: still failed, retrying once with a new seed", 2)
                gpu_heat_guard()
                img = generate_image(scene, scene_num, chapter_num, seed_offset=4001)
            if img:
                src = find_comfy_file(img)
                if src is not None:
                    dst = images_dir / img
                    shutil.copy2(src, dst)
                    _write_still_sidecar(dst, scene)
                    # Best-image selector: validate consistency
                    # against the scene description via vision LLM
                    # (C1: only when a real VL model is present — the
                    # text model + stopped LLM server made this dead weight)
                    if vlm_available():
                        try:
                            bis = _best_image_selector()
                            best_idx, _, _ = bis.select(
                                ref_image_path=str(dst),
                                ref_text=scene.get("image_prompt", ""),
                                frame_description=scene.get("narration", ""),
                                candidate_paths=[str(dst)],
                                candidate_texts=[scene.get("image_prompt", "")],
                            )
                            # best_idx == 0 means our single candidate won
                            log(f"  Scene {scene_num}: best-image check ok ({best_idx})", 2)
                        except Exception:
                            pass
                    # Head-inside-frame gate: one reframed retry when the VLM
                    # flags a cut/missing head (cast scenes only, A-stills only).
                    # C2/C3/M3: best-available always wins — a flagged still beats
                    # a gap, and the original is deleted ONLY after the reframe
                    # lands. Whole retry is guarded; errors keep the original.
                    # M6: episode-wide reframe budget (default 4), then warn-only.
                    _fok, _why = (face_visibility_check(dst, scene_num)
                                  if scene.get("characters") else (True, "no-cast"))
                    _framed_dst = dst
                    try:
                        _roff = int(CONFIG.get("face_check_retry_offset", 50000))
                    except Exception:
                        _roff = 50000
                    _budget = int(CONFIG.get("face_check_max_reframes", 4))
                    _used = sum(1 for s in scenes if s.get("_reframed"))
                    if not _fok and _used < _budget:
                        log(f"  Scene {scene_num}: headless still ({_why}), reframing once...", 2)
                        try:
                            img2 = generate_image(
                                scene, scene_num, chapter_num,
                                focus_text="head fully inside frame, face clearly visible, generous headroom",
                                seed_offset=_roff)
                            src2 = find_comfy_file(img2) if img2 else None
                            if src2 is not None:
                                _cand = images_dir / img2
                                shutil.copy2(src2, _cand)
                                try:
                                    _cand.with_suffix(".cache.json").write_text(
                                        json.dumps(_visual_fingerprint(scene, still_path=_cand)), encoding="utf-8")
                                except OSError:
                                    pass
                                _fok2, _why2 = face_visibility_check(_cand, scene_num)
                                _framed_dst = _cand
                                for _old in (dst, dst.with_suffix(".cache.json")):
                                    try:
                                        _old.unlink()
                                    except Exception:
                                        pass
                                scene["_reframed"] = True
                                log(f"  Scene {scene_num}: reframe "
                                    f"{'ok' if _fok2 else f'still headless ({_why2}), kept (best-available)'}", 2)
                            else:
                                log(f"  Scene {scene_num}: reframe produced nothing, keeping original", 2)
                        except Exception as e:
                            log(f"  Scene {scene_num}: reframe error ({str(e)[:100]}), keeping original", 2)
                    elif not _fok:
                        scene["_face_flag"] = _why
                        log(f"  Scene {scene_num}: headless still ({_why}), budget spent — kept + flagged", 2)
                    scene_images[si] = _framed_dst
                    if _framed_dst is not None:
                        log(f"  Scene {scene_num}: done", 2)

        # There is deliberately no beat-2 mid-scene still: the only consumer
        # was the dual-still dissolve treatment, which is exactly the
        # frame-plus-effects look this pipeline no longer ships. Every scene
        # gets a real H3 clip now, so the second image is pure GPU spend for
        # a picture nothing receives.

        # Companion close-up for lip-sync (cast + lip-sync shots only).
        # Fullbody stills starve EchoMimic of face pixels; the "c" still feeds it.
        # ponytail: the only consumer is the Phase 5 lip pass, which skips when
        # the EchoMimic pack is absent — and the pack has never been installed
        # ("Talking heads: 0", zero *echo*.mp4 in four lifetime renders). So
        # this was spending ~68 min per episode generating 21 images that every
        # real reader (7661, 9061, quality_evolve, /api/quality) excludes by name.
        # One memoised /object_info probe decides it; install the pack and this
        # flips back on with no other change.
        _csc = None
        if (scene.get("characters")
                and _echo_nodes_available()
                and _shot_norm(scene) in {_shot_norm({"shot_size": s})
                                          for s in CONFIG.get("echo_shots", ["close-up", "medium"])}):
            _csc = dict(scene)
            _csc["shot_size"] = "close-up"
            _csc["image_prompt"] = (scene.get("image_prompt", "")
                                   + ", close-up face portrait, face clearly visible")
            # Sidecar-gated like every other still: glob existence alone meant a
            # companion from a different prompt or an older bible was replayed
            # as the lip-sync identity source forever.
            if _pick_cached_still(
                    images_dir, f"ch{chapter_num:03d}_s{scene_num:02d}c_*.png",
                    _csc) is None:
                log(f"  Scene {scene_num}: companion close-up...", 2)
                cimg = generate_image(_csc, scene_num, chapter_num,
                                      suffix="c", seed_offset=700, framing=True)
                if cimg:
                    csrc = find_comfy_file(cimg)
                    if csrc is not None:
                        cdst = images_dir / cimg
                        shutil.copy2(csrc, cdst)
                        _write_still_sidecar(cdst, _csc)
                        log(f"  Scene {scene_num}: companion done", 2)

    if not any(scene_images):
        log("No reference images generated")
        return None
    # ponytail: codex binds images — every scene must hold an A-still FILE,
    # not just a non-None slot (a deleted cache between phases once left a
    # slot pointing at nothing).
    require_phase("images", [
        ("stills complete on disk",
         all(x is not None and Path(x).exists() for x in scene_images)
         and len(scene_images) == len(scenes)),
    ])

    # A file that EXISTS is not a file with PICTURE in it. A real ch900 render
    # produced 3 near-black stills (mean luma 0.09/0.36/2.70) that sailed past
    # the check above, and H3 animated scene 1 out of a pure black frame into a
    # hallucinated face. Fail closed: regen the black ones on a fresh seed.
    _black = find_black_stills(scene_images)
    if _black:
        _bad = ", ".join(f"s{n}={l:.2f}" for n, l in _black)
        log(f"  [FAIL] black still(s): {_bad} — regenerating on a new seed", 1)
        for n, _l in _black:
            si = n - 1
            if not (0 <= si < len(scene_images)):
                continue
            # seed_offset, not force: the point is a different seed, which is
            # what the arg exists for. A black frame is usually one bad roll.
            try:
                _fresh = _reroll_still(scenes[si], n, chapter_num,
                                       images_dir, 1000 + si)
            except Exception as e:
                log(f"  [WARN] black-still regen failed for scene {n} ({e})", 1)
                continue
            if _fresh:
                scene_images[si] = _fresh
        _still_black = find_black_stills(scene_images)
        require_phase("images", [
            ("stills not black",
             not _still_black or
             f"still black after retry: "
             f"{', '.join(f's{n}' for n, _ in _still_black)} — "
             f"delete them or fix the model"),
        ])

    # Perceptual duplicate gate. sha256 fingerprints cannot detect the case
    # that actually bites: two different prompts collapsing to the SAME
    # picture (the "25x static" incident). Prompt-level shot diversity
    # heuristics cannot see it either, because the prompts really do differ.
    _dupes = find_duplicate_stills(scene_images)
    # ponytail: re-roll before gating, exactly like the black-still retry above.
    # The collapse is a SEED basin, not the prompt and not the tag soup. On
    # ch001, scenes 7/8/9/10/12 drew consecutive seeds (3007..3012) and all five
    # landed on the same character-card portrait at pHash distance 0-2, while
    # the IDENTICAL prompt at seed 10007 came back correct, on-model and
    # varied (distance 27). So a fresh seed is the whole fix — blocking a
    # two-hour render on a seed roll is not.
    if _dupes:
        _hit = sorted({n for a, b, _ in _dupes for n in (a, b)})
        _before = list(scene_images)          # keep the pre-reroll set to revert to
        _n_before = len(_dupes)
        log(f"  [FIX] {len(_hit)} still(s) in a collapse basin {_hit} — "
            f"re-rolling on fresh seeds", 1)
        _reroll = int(CONFIG.get("face_check_retry_offset", 50000))
        for n in _hit:
            si = n - 1
            if not (0 <= si < len(scene_images)):
                continue
            try:
                _fresh = _reroll_still(scenes[si], n, chapter_num,
                                       images_dir, _reroll + si)
            except Exception as e:
                log(f"  [WARN] dupe re-roll failed for scene {n} ({e})", 1)
                continue
            if _fresh:
                scene_images[si] = _fresh
        _dupes = find_duplicate_stills(scene_images)
        # ponytail: a re-roll that makes the count WORSE must be rolled back, not
        # counted as progress. ch007: 30 -> 37 and 26 -> 58 pairs, i.e. the
        # recovery path doubled the failure it was invoked to repair, and the
        # gate then fired on a number the originals would have survived. Reverting
        # keeps the pre-reroll images (the gate message and the count both), so
        # the render either passes on what it already had or BLOCKS on the honest
        # number — never on a self-inflicted regression.
        if len(_dupes) > _n_before:
            log(f"  [FIX] re-roll made it worse ({_n_before} -> {len(_dupes)} pairs)"
                f" — reverting {len(_hit)} still(s) to the pre-reroll set", 1)
            scene_images[:] = _before
            _dupes = find_duplicate_stills(scene_images)
            log(f"  [FIX] reverted: {_n_before} pair(s)", 1)
    _LAST_ASSEMBLY["duplicate_stills"] = [
        {"scene_a": a, "scene_b": b, "distance": d} for a, b, d in _dupes]
    if _dupes:
        _worst = max(d for _, _, d in _dupes)
        _msg = (f"{len(_dupes)} near-identical still pair(s) in this episode "
                f"(closest hash distance {_worst} of 64, pixel-confirmed) — "
                f"these are the same picture twice: {_dupes[:5]}")
        if len(_dupes) > max(2, len(scenes) // 4):
            log(f"  [BLOCKER] {_msg}", 1)
            raise RuntimeError(_msg)
        log(f"  [WARN] {_msg}", 1)

    # Phase 4: Clips (real video for motion + majority fill, cinematic rest)
    note_phase("4")
    stop_vlm()
    require_phase("clips", [("stills complete", all(x is not None for x in scene_images))])
    # ponytail: release the stills' models BEFORE loading H3. On a 25-scene
    # chapter the image phase keeps Qwen-Image + the 32B text encoder resident
    # for ~90 min; ComfyUI then OOM-crashed the moment H3 loaded and every
    # remaining scene fell back to a still-pan. Cost is ~10s to reload a still
    # model; the alternative is losing the whole clip phase.
    free_comfyui_vram()
    # ponytail: /free left 4761 MiB resident across 3 attempts, because it
    # only flags the prompt queue and a custom node still holds models. Order
    # matters: evict TTS FIRST (that is 3GB we control), then measure, then
    # escalate to a real restart only if the residue would actually starve H3.
    # A restart costs ~40s and is the only release that is guaranteed.
    try:
        _tts_before_clips = check_tts_server()
        if _tts_before_clips:
            log("  [MEM] evicting TTS server for the clip phase (~3GB)", 1)
            _stop_tts_server()
    except Exception as _e:
        log(f"  [MEM] TTS evict skipped: {str(_e)[:60]}", 2)
    # H3 peaks at ~13.1GB of 16.3GB. If what is still resident would leave
    # less headroom than that, /free is not good enough and the clip phase
    # dies in comfy-aimdo. Restart instead: ~40s against losing the phase.
    _resident = _comfy_vram_mib()
    if _resident is None:
        _resident = _gpu_used_mib()
    if _resident is not None and _resident > _H3_HEADROOM_MIN_MIB:
        log(f"  [MEM] {_resident} MiB still held by ComfyUI - restarting for a "
            f"clean allocator", 1)
        # Fail loudly. The measurement says H3 will not fit, and proceeding
        # walks straight into comfy-aimdo, which aborts the process and
        # downgrades every remaining scene. That is the 14/25 still-pan run.
        if not restart_comfyui(why="clip phase VRAM"):
            raise RuntimeError(
                f"clip phase cannot fit H3: {_resident} MiB held by ComfyUI "
                f"after TTS evict and a failed ComfyUI restart "
                f"(need <= {_H3_HEADROOM_MIN_MIB} MiB)")
    log("\nPhase 4: Generating clips...")
    temp = _asset("_temp")
    temp.mkdir(exist_ok=True)
    scene_clips = [None] * len(scenes)
    video_count = 0
    filler_count = 0
    cinematic_count = 0

    # Every scene gets real video. The motion gate used to route the scenes
    # without a motion keyword (usually the quiet, contemplative ones) to a
    # camera move over the still — which is what reads as "just a frame with
    # effects added". The user's requirement, and the whole reason this phase
    # exists: every clip must show actual movement in the characters or the
    # background. The LOTM motion art-direction carries drifting fog,
    # flickering gaslight and camera drift into every H3 prompt, so even a
    # silent two-hander still breathes; it simply never becomes a still.
    motion_idx = set(range(len(scenes)))
    log(f"  Motion scenes: {sorted(n + 1 for n in motion_idx)} (every scene)", 1)

    # ponytail: continuity relay. H3's ReferenceToVideo node takes up to 9
    # reference images addressed positionally (<Picture 1>, <Picture 2>, ...)
    # and the shipped graph filled exactly ONE slot — so identity was locked
    # per-clip against a still, with nothing carrying across a cut. Track the
    # last rendered clip of the PREVIOUS scene and feed its final frame in as
    # <Picture 2>, so the encoder continues from what was actually on screen.
    # Disabled by config: an extra reference costs encoder time and the
    # benefit is visual, so it stays a deliberate switch, not a silent change.
    _relay_on = bool(CONFIG.get("h3_continuity_relay", True))
    _relay_dir = temp / "relay"
    _prev_relay = None
    _tier_used = set()
    relay_count = 0
    # Clip-vs-still fidelity receipt. grade_clip() only compares a clip's own
    # first and last frames, which cannot see a subject swap; these record how
    # far each clip drifted from the still it was made from.
    _drift_log = {}
    _clip_vlm_flags = []
    _clip_checks = []
    # Frames for the deferred vision check. Kept on disk (not in _temp) so a
    # crash does not lose them, and cleaned up after Phase 8.
    _clip_check_dir = temp / "clipcheck"
    _clip_check_dir.mkdir(parents=True, exist_ok=True)
    # Colour continuity. The reference is the episode MEDIAN look, built as
    # clips are graded. Measured on the 19 real H3 clips of ch001: luma
    # ranged 18.3..114.4 (spread 96/255) and r-b ranged -40..+70. Every clip
    # was the same mood, so mood cannot be the grouping key — and some of
    # that spread is deliberate (a fire scene SHOULD be brighter than a dark
    # alley), which is why the correction is capped rather than absolute.
    _ref_stats = []
    _grade_on = bool(CONFIG.get("clip_colour_grade", False))
    grade_applied = 0
    grade_rejected = 0

    def _relay_for(index):
        """Staged last frame of the previous scene's clip, or None."""
        if not _relay_on or index <= 0:
            return None
        return _prev_relay

    def _remember(clip):
        nonlocal _prev_relay
        if not _relay_on or not clip or not Path(clip).exists():
            return
        _prev_relay = _extract_last_frame(
            clip, _relay_dir / f"prev{Path(clip).stem}.png")
    for i, scene in enumerate(scenes):
        scene_num = scene.get("scene_number", i + 1)
        if should_stop():
            log("Stopped by user."); _clear_pid(); return None
        set_progress("clips", i, len(scenes), f"Scene {scene_num}")
        gpu_heat_guard()
        if should_stop():
            return None
        clip_path = temp / f"scene_{scene_num:02d}.mp4"

        # ponytail: faceless wide/establishing shots unlock full motion
        # (nothing to morph); character shots keep the face lockdown.
        if (_shot_norm(scene) in ("wide", "extreme-wide")
                and not scene.get("characters")
                and "lock_face" not in scene):
            scene["lock_face"] = False
            log(f"  Scene {scene_num}: wide establishing, face-lock off", 2)

        # ponytail: H3 only (2026-09-25). Reuse prior H3-tagged caches.
        _eng = "h3"
        existing_video = list(images_dir.glob(f"ch{chapter_num:03d}_{_eng}{scene_num:02d}.mp4"))
        _want_sec = scene_clip_seconds(scene)
        if existing_video:
            _ev = _newest(existing_video)
            # ponytail: a cached clip is only reusable if it is the right
            # LENGTH. Every cached H3 clip was rendered at the old fixed
            # h3_frames (124 = 5.2s) because no audio existed when it was
            # made, so reusing one by filename alone silently keeps the
            # old timing for every scene and assembly loops it back out.
            # Compare against the slot instead: the grid step is 17 frames
            # (0.7s at 24fps), so anything inside that is the same shot.
            _fps_e, _dur_e = video_fps_duration(_ev)
            _len_stale = _dur_e > 0 and abs(_dur_e - _want_sec) > 1.0
            if _len_stale:
                log(f"  Scene {scene_num}: cached clip is {_dur_e:.1f}s, "
                    f"narration needs {_want_sec:.1f}s — regenerating", 2)
                try:
                    _ev.unlink()
                except Exception:
                    pass
                existing_video = []
        # ponytail: a cached clip is only reusable if it was made from the
        # STILL THAT IS ON DISK NOW. The clip cache is keyed by filename with
        # no sidecar, so a regenerated still silently kept the old clip and
        # the episode showed the fix nowhere. clean_chapter deletes sidecars,
        # so a sidecar would not survive either - the mtimes do: the still is
        # written before the clip, so a newer still means a newer clip is
        # needed.
        _still_newer = False
        if existing_video and scene_images[i] is not None:
            try:
                _st = Path(scene_images[i]).stat().st_mtime
                _ct = _newest(existing_video).stat().st_mtime
                _still_newer = _st > _ct + 2
            except OSError:
                _still_newer = False
        if _still_newer:
            log(f"  Scene {scene_num}: still was regenerated after the cached "
                "clip - regenerating the clip too", 2)
            for _old in existing_video:
                try:
                    _old.unlink()
                except OSError:
                    pass
            existing_video = []
        if existing_video:
            _ev = _newest(existing_video)
            # ponytail: stale raw clips (pre-enhance era, e.g. 12fps LTX)
            # poison the 24fps timeline with judder. Heal once: drop the raw
            # and regenerate through the enhance path instead of reusing.
            _ev_raw = False
            try:
                _pr = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=avg_frame_rate",
                     "-of", "csv=p=0", str(_ev)],
                    capture_output=True, text=True, timeout=30)
                # ponytail: only a SUCCESSFUL probe reading <20fps condemns
                # the cache. "0/0"/empty (transient blip) must keep it —
                # the old test deleted good cache on unparseable output.
                _num, _, _den = (_pr.stdout.strip().partition("/"))
                _ev_raw = (_pr.returncode == 0 and float(_num or 0) > 0
                           and float(_num) / max(1.0, float(_den or 1)) < 20)
            except Exception:
                _ev_raw = False
            if _ev_raw:
                log(f"  Scene {scene_num}: stale raw clip ({_ev.name}), regenerating...", 2)
                try:
                    _ev.unlink()
                except Exception:
                    pass
            else:
                shutil.copy2(_ev, clip_path)
                scene_clips[i] = clip_path
                # ponytail: a reused clip is still a generated clip. It `continue`d
                # before video_count was touched, so a resumed render reported
                # "0 video clips" next to 14 clips sitting on disk.
                video_count += 1
                _remember(clip_path)
                log(f"  Scene {scene_num}: video exists", 2)
                continue

        # Every scene is a motion scene (see motion_idx above), so the real
        # H3 branch is the only path. There is deliberately no still-with-a-
        # camera-move branch: a frame with a slow push on it is what this
        # pipeline must never ship as "video".
        log(f"  Scene {scene_num}: generating video...", 2)
        if scene_images[i] is not None:
            staged = stage_comfy_input(scene_images[i])
            relay = _relay_for(i)
            if relay:
                relay_count += 1
                log(f"  Scene {scene_num}: continuity relay from scene "
                    f"{i} (last frame as <Picture 2>)", 2)
            # ponytail: H3 Ref2VA single pass (4-15s native, no chaining).
            # Grade BEFORE caching — bad motion retries, not assembly.
            h3_src = None
            try:
                for attempt in (0, 1, 2):
                    # ponytail: length in FRAMES off the real narration
                    # slot, so the clip is born the right size. Without
                    # this H3 ran its fixed h3_frames (124 ≈ 5.2s) and
                    # assembly looped it to cover a longer line.
                    h3_name = generate_h3_clip(
                        staged, scene, chapter_num,
                        length=math.ceil(
                            scene_clip_seconds(scene)
                            * int(CONFIG.get("h3_fps", 24))),
                        seed_offset=attempt * 977,
                        relay_frame=relay)
                    cand = find_comfy_file(h3_name) if h3_name else None
                    if cand is None:
                        continue
                    # ponytail: grade sampled frames of the mp4 (1fps).
                    _hg = temp / f"h3grade_{scene_num:02d}"
                    # Wipe first: ffmpeg's g%02d.png only overwrites from
                    # index 0, so an attempt killed mid-write leaves a
                    # longer tail behind and the next attempt grades
                    # frames from two different takes.
                    shutil.rmtree(_hg, ignore_errors=True)
                    _hg.mkdir(parents=True, exist_ok=True)
                    try:
                        subprocess.run(
                            [CONFIG["ffmpeg"], "-y", "-i", str(cand),
                             "-vf", "fps=1", str(_hg / "g%02d.png")],
                            capture_output=True, timeout=120)
                        _hframes = sorted(_hg.glob("g*.png"))
                        if len(_hframes) >= 2:
                            passed, why = grade_clip(_hframes)
                            # Grade against the SOURCE STILL too. Internal
                            # frame-to-frame consistency only proves the
                            # model stayed consistent with its own mistake:
                            # ch900 s1 became a doll-face close-up from a
                            # wide mill and passed every internal check.
                            _still = scene_images[i]
                            if _still and Path(_still).exists():
                                _drift = still_drift(_hframes[0], _still)
                                if _drift is not None:
                                    _drift_log.setdefault(scene_num, []).append(_drift)
                                # QUEUE the vision check, do not run it here.
                                # It was timing out because it ran mid-render
                                # with ComfyUI holding the VRAM; two 1024-token
                                # images on a shared GPU will not finish.
                                # Phase 8 runs it with the GPU free.
                                _keep = _clip_check_dir / f"c{scene_num:02d}.png"
                                try:
                                    shutil.copy2(_hframes[0], _keep)
                                    _clip_checks.append(
                                        {"scene": scene_num, "frame": str(_keep),
                                         "still": str(_still), "drift": _drift})
                                except Exception as e:
                                    log(f"  Scene {scene_num}: clip-check "
                                        f"frame copy failed ({e})", 2)
                            if not passed:
                                log(f"  Scene {scene_num}: clip failed grade "
                                    f"({'; '.join(why)}), retry {attempt + 1}/3", 2)
                                continue
                        h3_src = cand
                        break
                    finally:
                        shutil.rmtree(_hg, ignore_errors=True)
            except (ComfyUIError, Exception) as e:
                # ponytail: ComfyUI dying is RECOVERABLE, not a reason to
                # degrade the shot. comfy-aimdo aborts the whole process on
                # a failed CUDA allocation (cudaErrorMemoryAllocation) rather
                # than returning an OOM, which surfaces here as connection
                # refused. The old code logged and fell straight to a
                # still-pan, so one crashed process silently downgraded
                # EVERY remaining scene — that is how a 25-scene chapter
                # lost its whole clip phase. Restart and retry instead.
                _dead = ("unreachable" in str(e).lower()
                         or "10061" in str(e))
                if _dead and not _sibling_render_live():
                    log(f"  Scene {scene_num}: ComfyUI is down "
                        f"({str(e)[:80]}) - restarting it", 1)
                    # ponytail: ensure_comfyui() only PROBES, so it used to
                    # come back "ComfyUI would not come back" seconds after a
                    # restart while the server was still booting, and the
                    # shot degraded anyway. Wait for the actual health probe,
                    # then retry up to three times with fresh seeds - a
                    # transient outage must not cost the scene.
                    if restart_comfyui(why=f"scene {scene_num}"):
                        log(f"  Scene {scene_num}: ComfyUI relaunch requested", 1)
                        for _try in range(3):
                            for _ in range(30):
                                if ensure_comfyui():
                                    break
                                time.sleep(5)
                            else:
                                log(f"  Scene {scene_num}: ComfyUI still not "
                                    f"healthy after 150s", 1)
                                break
                            try:
                                # ponytail: call the WRAPPER, not the inner
                                # worker. _try_h3_clip submits width/height
                                # verbatim, so passing None here built a
                                # prompt ComfyUI rejects with 400 -- the
                                # recovery path failed in silence and fell
                                # back to the very still-pan it was added to
                                # prevent. generate_h3_clip resolves the
                                # quality tier, the dims and the OOM
                                # fallback, exactly like the first attempt.
                                h3_src = generate_h3_clip(
                                    staged, scene, chapter_num,
                                    seed_offset=(attempt * 977) + (_try * 337),
                                    relay_frame=relay)
                            except Exception as e2:
                                log(f"  Scene {scene_num}: retry {_try + 1} "
                                    f"failed ({str(e2)[:90]})", 2)
                                h3_src = None
                            if h3_src is not None:
                                log(f"  Scene {scene_num}: recovered on retry "
                                    f"{_try + 1}", 1)
                                break
                    else:
                        log(f"  Scene {scene_num}: ComfyUI would not "
                            f"come back", 1)
                        h3_src = None
                if h3_src is None:
                    log(f"  Scene {scene_num}: H3 error "
                        f"({str(e)[:150]}), fallback...", 2)
            if h3_src is not None:
                out_w, out_h = get_output_dims()
                tmp_grade = clip_path.with_name(
                    f"{clip_path.stem}.grade.mp4")
                # Cross-clip colour/exposure continuity: pull each clip
                # toward the episode's median look with a capped nudge, so
                # cuts stop jumping without flattening deliberate lighting.
                # Free — this replaces the plain scale+encode that was
                # already here.
                _cur = clip_rgb_stats(h3_src)
                if _cur:
                    _ref_stats.append(_cur)
                # The reference is the episode's MEDIAN look, computed
                # from every clip graded so far. Anchoring on clip #1 was
                # wrong on real output: across the 19 H3 clips of ch001
                # the luma range was 18..114, and scene 1 happened to be
                # a mid-dark corridor, so clip #1 would have dragged the
                # fire scene down to corridor level.
                _ref = grade_reference_from(_ref_stats)
                _ok_g = _apply_clip_grade(
                    h3_src, clip_path, "", out_w, out_h,
                    target_fps=int(CONFIG.get("fps", 24)))
                _used, _passes, _resid = "", 0, 0.0
                # Off by default: see the clip_colour_grade note in the
                # defaults table. Measured to invert the episode's
                # lighting on real output, so it must be an explicit
                # choice, not the default.
                if (_ok_g and _grade_on and _ref and _cur
                        and len(_ref_stats) > 1):
                    # second pass: now that a median exists, nudge toward it
                    _ok2, _used, _passes, _resid = grade_clip_to_reference(
                        clip_path, tmp_grade, _ref, out_w, out_h,
                        target_fps=int(CONFIG.get("fps", 24)))
                    if _ok2:
                        shutil.move(str(tmp_grade), str(clip_path))
                        _ok_g = True
                    else:
                        _used, _passes, _resid = "", 0, 0.0
                if not _ok_g:
                    # ffmpeg leaves a PARTIAL file behind on failure. The
                    # old code logged "using ungraded clip" and then
                    # shipped those bytes, which died at validate_media
                    # after all the GPU work. Drop it and fall through to
                    # the cinematic fallback.
                    log(f"  Scene {scene_num}: grade pass failed, "
                        f"discarding partial clip", 2)
                    clip_path.unlink(missing_ok=True)
                    grade_rejected += 1
                elif _used:
                    grade_applied += 1
                    log(f"  Scene {scene_num}: colour-graded toward the "
                        f"episode median ({_passes} pass"
                        f"{'es' if _passes != 1 else ''}, residual "
                        f"{_resid:.1f}/255)", 2)
                if clip_path.exists():
                    try:
                        shutil.copy2(clip_path, images_dir /
                                     f"ch{chapter_num:03d}_h3{scene_num:02d}.mp4")
                    except Exception:
                        pass
                    scene_clips[i] = clip_path
                    video_count += 1
                    _remember(clip_path)
                    _tier_used.add(str(CONFIG.get("quality", "fast")))
                    log(f"  Scene {scene_num}: done (H3 video)", 2)
                    continue
            # No still-with-a-camera-move fallback, ever: a frame with a slow
            # push on it is not video. The render fails loudly instead; every
            # clip either breathes or the episode does not ship. The cached
            # clips for the scenes that worked are reused on re-render.
            raise RuntimeError(
                f"Scene {scene_num}: H3 could not produce a clip after 3 attempts - "
                f"refusing to ship a still with a camera move as 'video'. "
                f"Fix the still or the prompt and re-render; the cached clips "
                f"for the other scenes are reused.")

    log(f"\n  Summary: {video_count} video clips (every scene)")

    # ponytail: make a tier downgrade VISIBLE. `_yaml_fallback` silently drops
    # any final-tier OOM to fast, so an episode can ship fast-tier clips with
    # no marker anywhere — in the receipt, the report, or the dashboard.
    _LAST_ASSEMBLY["h3_tiers"] = sorted(_tier_used)
    # ponytail: release_report.py compares h3_tiers against
    # assembly_receipt["quality_requested"] to decide a tier fallback. Nothing
    # wrote that key, so the comparison was ["fast"] != [None] and every
    # single-tier episode got a permanent "tier fell back" REVIEW — which
    # trains the owner to ignore NEEDS_REVIEW. One line, same value the
    # fallback log below already prints.
    _LAST_ASSEMBLY["quality_requested"] = str(CONFIG.get("quality", "fast"))
    _LAST_ASSEMBLY["colour_grade"] = {
        "enabled": _grade_on,
        "applied": grade_applied, "rejected": grade_rejected,
        "deflicker": bool(CONFIG.get("clip_deflicker", True)),
        "clips_measured": len(_ref_stats),
        "reference_median_luma": (grade_reference_from(_ref_stats) or {}).get("luma")}
    # Did the continuity relay actually fire? A silently-off relay looks
    # identical to "relay is on and the identity holds".
    _LAST_ASSEMBLY["relay_used"] = {"enabled": _relay_on,
                                    "clips_with_relay": relay_count,
                                    "clips_total": video_count + cinematic_count + filler_count}
    # Clip-vs-still fidelity. The drift is recorded here; the vision flags are
    # deferred to Phase 8 (see run_clip_still_checks) because the check needs
    # the GPU and ComfyUI has it during this phase.
    _LAST_ASSEMBLY["clip_still_drift"] = {
        str(k): v for k, v in sorted(_drift_log.items())}
    if _clip_checks:
        log(f"  {len(_clip_checks)} clip/still check(s) queued for the vision "
            f"pass in Phase 8", 2)
    if len(_tier_used) > 1 or (CONFIG.get("quality") == "final" and _tier_used == {"fast"}):
        log(f"  [WARN] H3 quality tier fell back: rendered at "
            f"{sorted(_tier_used)} (requested {CONFIG.get('quality')})", 1)

    # There is no still-pan path any more: a scene either produced a real
    # H3 clip (or already exists in cache) or the render raised. The receipt
    # still records the counters so the audit trail stays intact.
    _LAST_ASSEMBLY["cinematic_fallbacks"] = filler_count
    _LAST_ASSEMBLY["video_clips"] = video_count
    _LAST_ASSEMBLY["cinematic_clips"] = cinematic_count

    if not any(scene_clips):
        log("No clips created")
        return None
    # ponytail: codex binds clips — every scene keyed to a clip FILE before
    # voice/assembly spend a minute. Gaps abort here, not at the mux.
    require_phase("clips", [
        ("clips complete on disk",
         all(c is not None and Path(c).exists() for c in scene_clips)
         and len(scene_clips) == len(scenes)),
    ])


    # Audio plan per scene
    if _PIPELINE_UPGRADE:
        try:
            audio_plan = generate_audio_plan(scenes, chapter_num,
                                            CONFIG.get("music_dir"),
                                            CONFIG.get("sfx_dir"))
            save_audio_plan(audio_plan, chapter_num, Path(CONFIG["scenes_dir"]))
            log(f"  Audio plan generated ({len(audio_plan)} scenes)", 1)
        except Exception as e:
            log(f"  [WARN] Audio plan skipped: {e}", 1)

    # Phase 5: Talking heads — character-as-narrator. Only scenes WITH
    # an on-screen character get lip-sync (voice already matches lips
    # from Phase 5). Faceless/establishing shots keep Phase 3 clips.
    # Failures keep the old clip.)
    note_phase("5")  # ponytail: own phase, else 55min of lips lands on "5b" in perf
    # ponytail: the old gate read the CONFIG STRING, so it logged "gate OK:
    # echo engine path" four seconds before "[WARN] lip-sync disabled: pack not
    # installed", walked 20 scenes, and reported Talking heads: 0. The FIRST
    # fix made it fatal — which killed a finished render at Phase 5 because
    # lip-sync is best-effort by design (every echo failure keeps the old
    # clip). Probe once: if the engine is absent, say so loudly and SKIP the
    # phase, which is what the render already does scene-by-scene. Fatal only
    # if the owner flips echo_required on.
    _echo_wanted = bool(CONFIG.get("echo_enabled", True))
    _echo_up = _echo_wanted and _echo_nodes_available()
    if _echo_wanted and not _echo_up and CONFIG.get("echo_required", False):
        require_phase("lips", [("echo engine available", False)])
    log("\nPhase 5: Talking-head close-ups..." + ("" if _echo_up else " (skipped: EchoMimic pack not installed)"))
    echo_count = 0
    echo_shots = {_shot_norm({"shot_size": s})
                  for s in CONFIG.get("echo_shots", ["close-up", "medium"])}
    if _echo_up and ensure_comfyui():
        for i, scene in enumerate(scenes):
            scene_num = scene.get("scene_number", i + 1)
            if should_stop():
                log("Stopped by user.")
                return None
            if not scene.get("characters"):
                continue
            has_voice = bool(scene.get("narration") or scene.get("dialogue"))
            if not has_voice:
                continue
            if _shot_norm(scene) not in echo_shots:
                continue
            if scene_clips[i] is None or scene_images[i] is None:
                continue
            aud = dialogue_audios[i] if i < len(dialogue_audios) else None
            if not aud or not Path(aud).exists():
                continue
            clip_path = temp / f"scene_{scene_num:02d}.mp4"
            # _newest() returns a single path, NOT a list — indexing it with
            # [-1] took the last CHARACTER ("g" of .png) and crashed the
            # render after every still and clip had been paid for.
            _comp = _newest(images_dir.glob(
                f"ch{chapter_num:03d}_s{scene_num:02d}c_*.png"))
            _echo_fp = _echo_fingerprint(_comp or scene_images[i], aud, scene)
            cached = _newest(images_dir.glob(f"ch{chapter_num:03d}_echo{scene_num:02d}.mp4"))
            if cached and _cache_matches(cached, _echo_fp):
                shutil.copy2(cached, clip_path)
                scene_clips[i] = clip_path
                log(f"  Scene {scene_num}: talking head exists", 2)
                continue
            set_progress("talking", i, len(scenes), f"Scene {scene_num}")
            gpu_heat_guard()
            if should_stop():
                return None
            try:
                staged_still = stage_comfy_input(_echo_fp["still"]["path"])
                staged_wav = _stage_echo_audio(aud, scene)
                log(f"  Scene {scene_num}: talking head...", 2)
                echo_src = generate_lipsync_clip(staged_still, staged_wav, scene, chapter_num)
            except Exception as e:
                log(f"  Scene {scene_num}: EchoMimic error ({str(e)[:150]}), keeping clip", 2)
                echo_src = None
            if echo_src:
                cand = find_comfy_file(echo_src)
            else:
                cand = None
            if cand is None:
                continue
            if frames_to_clip(cand, clip_path, fps=int(CONFIG.get("echo_fps", 24)),
                                   max_sec=float(scenes[i].get("_dur", 0) or 0) or None):
                shutil.copy2(clip_path, images_dir /
                             f"ch{chapter_num:03d}_echo{scene_num:02d}.mp4")
                # Not _write_still_sidecar: the echo sidecar carries _echo_fp, which
                # _cache_matches compares field-for-field against a freshly
                # computed fingerprint. Atomic so a torn sidecar cannot make a
                # valid lip clip look invalid (or the reverse).
                dump_json_atomic(
                    (images_dir / f"ch{chapter_num:03d}_echo{scene_num:02d}.mp4")
                    .with_suffix(".cache.json"), _echo_fp, indent=2)
                scene_clips[i] = clip_path
                echo_count += 1
                log(f"  Scene {scene_num}: done (talking head)", 2)
    log(f"  Talking heads: {echo_count}", 1)

    # ponytail: fail loud on gaps — zip() in assembly silently truncates,
    # which once shipped a short episode as "complete".
    _clipless = [s.get("scene_number", i + 1) for i, s in enumerate(scenes) if scene_clips[i] is None]
    if _clipless:
        log(f"[FAIL] Missing clips for scenes {_clipless} — aborting (no silent truncation)")
        _clear_pid()
        return None

    # ponytail: watch every clip — existence checks pass 0-byte/corrupt/
    # all-black clips straight into assembly. Fail loud per scene.
    _badclips = []
    for i, (clip, scene) in enumerate(zip(scene_clips, scenes)):
        _ok, _why = qa_clip_file(clip)
        if not _ok:
            _badclips.append(f"s{scene.get('scene_number', i + 1)}:{_why}")
    if _badclips:
        log(f"[FAIL] Unusable clips ({'; '.join(_badclips)}) — aborting")
        _clear_pid()
        return None

    # Phase 6: Subtitles
    note_phase("6")
    log("\nPhase 6: Creating subtitles...")
    sub_path = temp / "subtitles.ass"
    srt_path, vtt_path = None, None
    if _PIPELINE_UPGRADE:
        try:
            srt_path, vtt_path = generate_subtitles(
                scenes, chapter_num, Path(CONFIG["output_dir"]),
                dialogue_audios)
            log(f"  Subtitles: {srt_path.name}, {vtt_path.name}", 1)
        except Exception as e:
            log(f"  [WARN] Subtitle generation failed: {e}", 1)

    # Phase 7: Assembly with viral hooks
    note_phase("7")
    require_phase("assembly", [
        ("clips present", all(c is not None and Path(c).exists() for c in scene_clips)),
        ("voice present", all(a is not None and Path(a).exists() for a in dialogue_audios)),
    ])
    if should_stop():
        log("Stopped by user.")
        return None
    set_progress("assembly", 0, 1, "Mixing, captions, bridges, music")
    log("\nPhase 7: Assembling video...")
    ver = str(CONFIG.get("assembly_version", "v2")).lower()
    if ver == "v1":
        log("  [WARN] assembly v1 retired — routing to v2 (single encode, transactional QA)")
        ver = "v2"
    output = assemble_video_v2(scene_clips, scenes, dialogue_audios, chapter_num, sub_path)

    if output and output.exists():
        size_mb = output.stat().st_size / (1024 * 1024)
        log(f"\n  DONE: {output.name} ({size_mb:.1f} MB)")
        log(f"  Location: {output}")
        # Phase 8: Rate + evolve (self-improving quality loop)
        # ponytail: codex binds QA — the gate receipt and the assembly
        # receipt must exist before rating. Keyed off the assembly version
        # ACTUALLY used (`ver`), not a re-read of CONFIG: setting
        # assembly_version="v1" in config.json used to switch off the QA
        # receipt gate while v2 still ran, because this re-read CONFIG
        # instead of using the resolved value.
        _is_v2 = ver == "v2"
        require_phase("qa", [
            ("av-sync receipt present", (not _is_v2) or "qa_av_sync" in _LAST_ASSEMBLY),
            ("assembly receipt present", (not _is_v2) or bool(_LAST_ASSEMBLY.get("scenes"))),
        ])
        log("\nPhase 8: Rating episode...")
        free_comfyui_vram()
        ensure_vlm()
        # The clip-vs-still vision check, deferred out of the clip phase so it
        # has the GPU to itself. Populates the same receipt key.
        try:
            _clip_vlm_flags = run_clip_still_checks(_clip_checks, scenes, chapter_num)
            if _clip_vlm_flags:
                _LAST_ASSEMBLY["clip_still_vlm_flags"] = _clip_vlm_flags
            else:
                _LAST_ASSEMBLY.pop("clip_still_vlm_flags", None)
        except Exception as e:
            # ponytail: loud, like the rating path below. This feeds
            # _LAST_ASSEMBLY["clip_still_vlm_flags"] -> the release report,
            # so a swallowed failure means the report silently claims the
            # visual check passed.
            log(f"  [FAIL] clip/still vision check failed — the release "
                f"report will not reflect it: {type(e).__name__}: {e}", 1)
        finally:
            shutil.rmtree(_clip_check_dir, ignore_errors=True)
        quality_log_entry = None
        try:
            quality_log_entry = rate_episode(chapter_num, scenes, scene_clips,
                                             dialogue_audios, novel=current_novel_id())
        except Exception as e:
            # ponytail: LOUD, because this is not optional observability.
            # quality_log.json is what evolve_quality() reads; a silent WARN
            # left the file frozen and the tuner looping on stale history for
            # days without anyone noticing. That is exactly how the
            # NameError('_newest') from the quality_evolve split survived a
            # 300-green test run and several real renders.
            log(f"  [FAIL] Rating failed — the auto-tuner will not see this "
                f"episode: {type(e).__name__}: {e}", 1)
            import traceback
            log("  " + traceback.format_exc().replace("\n", "\n  ")[:600], 2)
            try:
                from supervisor import record as _rec
                _rec("warn", "rating failed (quality_log stale)",
                     f"{type(e).__name__}: {str(e)[:120]}", resolved=False)
            except Exception:
                pass
        finally:
            stop_vlm()
        try:
            evolve_quality()
        except Exception as e:
            log(f"  [WARN] Evolve failed: {e}")
        # Phase 9: Cost log + editable timeline export
        try:
            export_edl(chapter_num, scenes, output)
        except Exception as e:
            log(f"  [WARN] EDL export failed: {e}")
        # Release report + release package (upgrade modules)
        if _PIPELINE_UPGRADE:
            try:
                # The NOVEL's manifest, not the project's root one: a second
                # novel has its own rights (a fan work's are non-commercial),
                # and reading the legacy manifest let a fan-work episode come
                # out of the release gate as if it were the owner's own work.
                rights_manifest = load_rights_manifest(_asset(""))
                _ow, _oh = get_output_dims()
                report = create_release_report(chapter_num, scenes,
                                              _LAST_ASSEMBLY,
                                              rights_manifest=rights_manifest,
                                              quality_log_entry=quality_log_entry,
                                              expect_fps=float(CONFIG.get("fps", 24)),
                                              expect_dims=(_ow, _oh))
                save_release_report(report, Path(CONFIG["output_dir"]), chapter_num)
                log(f"  Release report: {report['verdict']}", 1)
                if report["verdict"] == "NOT_RELEASE_READY":
                    log(f"  [BLOCKER] Episode NOT release-ready — "
                        f"{report['summary']['blockers']} blockers", 1)

                _pkg = build_release_package(
                    chapter_num, output, scenes,
                    series_name=_novel_title(),
                    director_package=dp,
                    release_report=report,
                    rights_manifest=rights_manifest,
                    # ponytail: two caption standards shipped and the WORSE
                    # one was the deliverable. generate_subtitles (subtitle_
                    # generator.py) wraps to nothing and has no CPS concept —
                    # videos/ch002.srt hit 55 chars/line. The ASS-derived
                    # sidecar next to the mp4 enforces 42/17 and lands as
                    # chNNN_youtube.srt (measured 39/17.0). Prefer it; fall
                    # back to the generator's when captions are off.
                    srt_path=(_platform_srt(output) or srt_path),
                    vtt_path=vtt_path)
                log("  Release package built", 1)

                # Opt-in auto-publish. Default OFF; the owner turns it on in
                # config.json (see pipeline/publisher.py for the per-platform
                # caveats — Instagram needs a public video_url, TikTok needs an
                # audited app for public posts).
                # Lazy import on purpose: publisher pulls in httpx, and a box
                # without httpx must not take the whole upgrade module set
                # down through the top-of-file try/except ImportError.
                if bool((CONFIG.get("publish") or {}).get("enabled", False)):
                    try:
                        from pipeline.publisher import publish_package
                        _pub = publish_package(_pkg, CONFIG["publish"], log=log)
                        _ok = [p for p, r in (_pub.get("results") or {}).items()
                               if isinstance(r, dict) and r.get("ok")]
                        if _ok:
                            log(f"  Published to: {', '.join(_ok)}", 1)
                    except Exception as e:
                        log(f"  [WARN] Publish step failed: {e}", 1)

                # Pass the pipeline-known facts. add_analytics_entry used to
                # be called with the chapter number alone, so every render
                # appended a metric-free row that polluted the last-5 window
                # propose_experiments() reads.
                add_analytics_entry(
                    _asset(""), chapter_num, novel=current_novel_id(),
                    quality_score=(quality_log_entry or {}).get("score"),
                    episode_seconds=_LAST_ASSEMBLY.get("streams", {}).get("video_duration"),
                    scene_count=len(scenes),
                    verdict=report.get("verdict"))
            except Exception as e:
                log(f"  [WARN] Release package failed: {e}", 1)

        # M3: Shorts + thumbnail were dead code — wire after assembly,
        # guarded so failures never block the shipped episode.
        if bool(CONFIG.get("shorts_enabled", False)):
            try:
                _sh = build_shorts_cut(output, chapter_num, scenes)
                log(f"  Shorts cut: {_sh}" if _sh else "  [WARN] Shorts cut failed", 1)
            except Exception as e:
                log(f"  [WARN] Shorts cut failed: {e}")
        try:
            _th = pick_thumbnail(CONFIG["images_dir"], chapter_num, scenes)
            if _th:
                log(f"  Thumbnail still: {Path(_th).name}", 1)
        except Exception as e:
            log(f"  [WARN] Thumbnail pick failed: {e}")
        set_progress("done", 1, 1, f"{output.name}")
        try:
            (ROOT / "render_chapter").unlink()
        except Exception:
            pass
        return output

    log("  Assembly failed")
    return None

# --- VLM Judge (seeing grader, swapped vision profile) ---

def _model_family(stem):
    """Model name with any quantisation suffix dropped, lowercased.

    Qwen3VL-8B-Instruct-Q4_K_M / -Q8_0 / -F16 are the same profile; a requant
    renames the file, and dropping the trailing quant lets the swap detection
    survive that instead of silently losing vision review.
    """
    s = str(stem).lower()
    return re.split(r"[-_.]?(q\d|iq\d|f16|f32|bf16|q\d+_k_[ms])\b", s)[0] or s


def _is_vlm_serving():
    """True when :8080 serves the vision profile.

    Compare the MODEL STEM, never the raw id. llama-server reports the id as
    the full path it was launched with ("C:/.../Qwen3VL-8B-...gguf"), and a
    path never starts with the model name, so a raw prefix test can never
    pass — vlm_available() was permanently False and every vision feature
    (face gate, still grading, final-frame review) silently no-opped.
    """
    try:
        sid = _serving_id()
        if not sid:
            return False
        got = _model_family(Path(sid).stem)      # bare stem, path or not
        want = _model_family(_vlm_id())
        return bool(got) and (got == want or got.startswith(want)
                              or want.startswith(got))
    except Exception:
        return False


def _pick_vlm():
    # Vision lives in a swapped :8080 profile (VLM sized to co-reside with
    # Comfy via partial offload). The text profile answers /v1/models too
    # but cannot grade images, so detection keys on the serving model id —
    # robust across processes (no reliance on in-process flags alone).
    if not bool(CONFIG.get("llm_vision_enabled", False)):
        return None
    try:
        if _is_vlm_serving():
            return _serving_id()
    except Exception:
        pass
    return None


def vlm_available():
    return _pick_vlm() is not None


def note_visual_review():
    """Loud once-per-render note when stills ship without VLM review.
    The face gate is honestly best-effort (no local detector judges anime
    faces) — so a skip must land in the ledger, not vanish into per-scene
    'no-vlm' reasons nobody reads. Warn severity: visible, never blocks."""
    if not bool(CONFIG.get("face_check_enabled", True)):
        return False
    if vlm_available():
        return False
    log("  [WARN] VLM down — stills ship on prompt head-lock only, no visual review", 1)
    try:
        from supervisor import record as _rec
        _rec("warn", "visual review skipped (VLM down)",
             "prompt head-lock only; re-render stills when VLM is up", resolved=False)
    except Exception:
        pass
    return True


def qa_final_black(path):
    """Code watches the finished file: fail loud if >30% is black,
    warn on any black segment >= 2s. Returns black seconds."""
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                         "format=duration", "-of", "csv=p=0", str(path)],
                        capture_output=True, text=True, timeout=60)
    # `.splitlines()[0]` raised IndexError on whitespace-only stdout, which
    # then failed the promote for a non-media reason.
    _first = ((pr.stdout or "").strip().splitlines() or ["0"])[0]
    try:
        total = float(_first or 0)
    except ValueError:
        total = 0.0
    if not total:
        raise RuntimeError(f"Could not read duration of {path} (ffprobe: {pr.stderr[:200]!r})")
    bd = subprocess.run([CONFIG["ffmpeg"], "-v", "info", "-i", str(path),
                         "-vf", "blackdetect=d=1.0:pix_th=0.10",
                         "-an", "-f", "null", "-"],
                        capture_output=True, text=True, timeout=300)
    segs = [float(b) for b in re.findall(r"black_duration:([0-9.]+)", bd.stderr or "")]
    black = sum(segs)
    if total > 0 and black > 0.3 * total:
        raise RuntimeError(f"final video {black:.1f}s black of {total:.1f}s — broken render")
    for b in segs:
        if b >= 2:
            log(f"  [WARN] black segment {b:.1f}s in final video", 1)
    return round(black, 2)


def qa_final_frames(candidate, workdir, n=4):
    """LLM watches the finished episode: grade evenly-spaced frames with the
    VLM. Warn-only signal, never a gate (code gates already passed).
    Returns receipt dict, or None when skipped (no VLM)."""
    if not vlm_available():
        return None
    try:
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of", "csv=p=0", str(candidate)],
                            capture_output=True, text=True, timeout=60)
        total = float((pr.stdout or "0").strip().splitlines()[0] or 0)
        if total <= 0:
            return None
        issues, checked = [], 0
        for k in range(n):
            frame = Path(workdir) / f"final_qa_{k}.png"
            try:
                r = subprocess.run(
                    [CONFIG["ffmpeg"], "-y", "-v", "error",
                     "-ss", str(round(total * (k + 1) / (n + 1), 2)),
                     "-i", str(candidate), "-frames:v", "1", str(frame)],
                    capture_output=True, timeout=120)
                if r.returncode or not frame.exists():
                    continue
                g = vlm_grade_image(str(frame),
                                    "final episode frame: flag only severe defects "
                                    "(missing face on close-up, torso-only framing, "
                                    "garbled text, black/empty frame)")
                # ponytail: count a frame only if the model actually ANSWERED.
                # This used to increment unconditionally, so a tripped vision
                # breaker produced a receipt reading {"checked": 4, "issues":
                # []} -- a clean bill of health from a model that was never
                # asked. Nothing reads that receipt either, which is how it
                # stayed invisible.
                if g is None:
                    log(f"  [WARN] final-frame grade {k} ungraded (no VLM "
                        f"result) - not counted as checked", 2)
                    continue
                checked += 1
                for x in g.get("issues", []):
                    issues.append(f"frame{k}: {x}")
            except Exception as e:
                log(f"  [WARN] final-frame grade {k} skipped ({str(e)[:60]})", 2)
            finally:
                try:
                    frame.unlink()
                except Exception:
                    pass
        if issues:
            log(f"  [WARN] VLM final-frame notes: {'; '.join(issues[:4])}", 1)
        if not checked:
            # An empty issue list plus checked=0 means nothing was graded.
            # Say so rather than handing back a receipt that reads as clean.
            return {"checked": 0, "issues": [],
                    "status": "unavailable - no VLM grade completed"}
        return {"checked": checked, "issues": issues}
    except Exception as e:
        log(f"  [WARN] final-frame review skipped ({e})", 1)
        return None


def _judge_head_issues(issues):
    """The head verdict from the vision model's issue list. Returns (ok, why).

    ponytail: the bare "cut off" matched BODY crops ("the coat is cut off by
    the frame") and flagged a perfectly framed head: ch018 s4 (Voss, whole
    head inside the frame) was escalated to a REVIEW on that phrase alone. The
    gate asks about the HEAD, so the phrase has to name the head or the face
    next to it.
    """
    issues = [str(x).lower() for x in (issues or [])]
    # Hair-obscured-by-design is on-model (fringe doctrine) - exclude it.
    issues = [x for x in issues if not ("hair" in x and ("obscur" in x or "cover" in x))]
    blob = " ".join(issues)
    bad = (r"\bhead out of frame\b", r"\bcropped head\b", r"\bhead cut",
           r"\bdecapitat", r"\bfaceless\b", r"\bno face\b",
           r"\bface not visible\b", r"\bmissing head\b",
           r"\bout of frame\b", r"\bcropped out\b", r"\bpartial head\b",
           r"\bface obscured\b",
           r"\b(cut off|cropped)\b[^.]{0,24}\b(head|face|chin|skull)\b",
           r"\b(head|face|chin|skull)\b[^.]{0,24}\b(cut off|cropped)\b",
           r"\btorso only\b", r"\btorso shot\b", r"\bwaist[- ]up only\b",
           r"\bwaist[- ]up crop\b", r"\bbody without face\b",
           r"\bmidsection\b", r"\btorso fills?\b")
    if any(re.search(p, blob) for p in bad):
        return False, "; ".join(issues[:2]) or "head-flagged"
    return True, "ok"


def face_visibility_check(still_path, scene_num):
    """Head-inside-frame gate for character stills (best-effort, never blocks).

    HONEST SCOPE (C1): vision runs in a swapped :8080 profile sized to
    co-reside with Comfy (partial offload). When the VLM is down the gate
    returns (True, no-vlm) and the prompt head-lock is the real net
    - the prompt head-lock is the real net. The retry below fires only when
    a VLM is actually reachable (dev runs, future phase ordering).
    No local detector can judge anime faces (YuNet scores 0), hence VLM.
    Fails ONLY on explicit head/crop phrases - grade alone never fails.
    Hair partly covering the face is on-model styling, never a fail.
    Returns (ok: bool, reason: str). Never raises.
    """
    if not bool(CONFIG.get("face_check_enabled", True)):
        return True, "disabled"
    if not vlm_available():
        return True, "no-vlm"
    try:
        g = vlm_grade_image(still_path, "anime character still: is the whole head with face fully inside the frame")
        if not g:
            log(f"  [WARN] VLM grading returned no result for scene {scene_num} - prompt head-lock is the only gate", 2)
            return True, "ungraded"
        return _judge_head_issues(g.get("issues", []))
    except Exception as e:
        log(f"  [WARN] face check skipped ({str(e)[:80]})", 2)
        return True, "error"



def _face_flag_for(still_path, scene_num):
    """The head-in-frame verdict for a still, cached beside it.

    ponytail: the cached-still path re-asked the VLM about every scene on
    every resume, and each answer is ~60 s - so a ONE-scene redo paid 40
    minutes to re-grade 39 stills it had already graded (ch018's scene-1
    redo took 57 min). The verdict is a property of the PICTURE, so the
    model's ANSWER is cached next to it; the verdict is then judged from
    that answer, so a change to the judging rules takes effect at once
    instead of being frozen into the cache. The provenance sidecar is
    deliberately not touched: its whole dict is compared byte for byte
    against the fingerprint.
    """
    side = Path(still_path).with_suffix(".face.json")
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "issues" in data:
            return _judge_head_issues(data["issues"])
        if isinstance(data, dict) and data.get("ok") is True:
            # An older sidecar holds the VERDICT, not the answer. A clean
            # verdict is kept (it costs a minute of vision to re-ask); a
            # flagged one is re-asked, because that is the case the new
            # judging rules and the new prompt are meant to correct.
            return True, str(data.get("why") or "ok")
    except (OSError, ValueError):
        pass
    if not bool(CONFIG.get("face_check_enabled", True)) or not vlm_available():
        return True, "no-vlm"
    try:
        g = vlm_grade_image(still_path, "anime character still: is the whole head with face fully inside the frame")
    except Exception as e:
        log(f"  [WARN] face check skipped ({str(e)[:80]})", 2)
        return True, "error"
    if not g:
        return True, "ungraded"
    issues = [str(x) for x in (g.get("issues") or [])]
    try:
        side.write_text(json.dumps({"issues": issues}), encoding="utf-8")
    except OSError:
        pass
    return _judge_head_issues(issues)


def vlm_grade_image(img_path, context="anime scene still"):

    global _vlm_consecutive_failures
    if not vlm_available():
        return None
    if _vlm_consecutive_failures >= _VLM_TRIP_AFTER:
        return None
    try:
        from json_util import oai_chat, oai_image_part
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        txt = oai_chat(_llm_base(), [{  # 90s: grades are short; fan-out is per-scene
            "role": "user",
            "content": [
                {"type": "text", "text":
                    f"Grade this {context} as an ANIME video frame. Reply ONLY valid JSON: "
                    '{"grade": "S|A|B|C|D", "anime_look": 0-100, '
                    '"composition": 0-100, "issues": ["short", "phrases"]}. '
                    "Issues cover: deformed faces/hands, photorealistic drift, "
                    "muddy composition, unreadable subject, watermark text. "
                    "When a character is present, report these ONLY IF they "
                    "are actually true: cropped head, head out of frame, head "
                    "cut off, faceless, face not visible, torso only, torso "
                    "shot, waist-up crop, body without face. A body, shoulder "
                    "or coat cut by the frame edge is NOT a head issue - if the "
                    "whole head and face are inside the frame, say nothing "
                    "about framing (use those exact words only for a real head "
                    "crop; hair partly covering the face is normal styling, "
                    "NOT an issue)."},
                oai_image_part(b64)]}],
            timeout=_VLM_GRADE_TIMEOUT_S, max_tokens=200)
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if not m:
            # A model that answers "I can't grade that" HAS answered -- that is
            # a refusal, not an outage, so it must not push the breaker toward
            # tripping. Only transport failures count.
            log(f"  [WARN] VLM grade returned unparseable output: "
                f"{txt[:80]!r}")
            return None
        _vlm_consecutive_failures = 0
        return json.loads(m.group(0))
    except Exception as e:
        log(f"  [WARN] VLM grade failed: {e}")
        _vlm_consecutive_failures += 1
        if _vlm_consecutive_failures == _VLM_TRIP_AFTER:
            log(f"  [WARN] vision grading disabled for this run after "
                f"{_vlm_consecutive_failures} consecutive failures - the "
                f"framing gate falls back to the prompt head-lock instead of "
                f"waiting {_VLM_GRADE_TIMEOUT_S}s per still", 1)
        return None


# --- Quality Rater + Evolve ----------------------------------------
# Moved to pipeline/quality_evolve.py. episode_producer imports this module
# from its own tail, so quality_evolve resolves CONFIG/ROOT/log back through
# `episode_producer` at call time and neither import order can deadlock.
# Only what this file actually calls is re-exported; widening it would
# create names that read as part of the producer's API but are not.
from pipeline.quality_evolve import (  # noqa: E402
    rate_episode, evolve_quality, run_clip_still_checks, _image_metrics,
)


def rotate_versions(output_path, preserve_current=False):
    """Versioned renders: ch001_youtube.mp4 -> .v1 -> .v2.

    Two generations, hardcoded. It used to take a `keep=2` parameter that the
    body never read — it always handled exactly .v1 and .v2 — so a caller asking
    for keep=5 silently got 2 with no error. It had no callers passing anything
    but the default, so the parameter was deleted rather than implemented.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return
    stem = output_path.stem  # ch001_youtube
    parent = output_path.parent
    # ponytail: copy current→.v1 FIRST, then shift .v1→.v2, delete oldest.
    # Original order lost both on crash mid-copy.
    # ponytail: shift v1->v2 FIRST, then archive current->v1. The old
    # order (archive first, then shift) moved the fresh v1 straight to
    # v2 — v1 never survived and the previous v1 history was destroyed.
    try:
        v2 = parent / f"{stem}.v2.mp4"
        v1 = parent / f"{stem}.v1.mp4"
        if v2.exists():
            v2.unlink()
        if v1.exists():
            v1.rename(v2)
        if preserve_current:
            shutil.copy2(output_path, v1)
        else:
            output_path.rename(v1)
    except Exception as e:
        if preserve_current:
            raise
        log(f"  [WARN] Version rotate failed: {e}")


def make_contact_sheet(chapter_num, cols=5, novel=None):
    with _novel_context(novel):
        return _make_contact_sheet(chapter_num, cols)


def _make_contact_sheet(chapter_num, cols=5):
    """Multi-grid storyboard preview of all scene refs for review."""
    from PIL import Image, ImageDraw
    images_dir = Path(CONFIG["images_dir"])
    scenes_file = Path(CONFIG["scenes_dir"]) / f"ch{chapter_num:03d}.json"
    if not scenes_file.exists():
        log("No scenes file, generate the episode first.")
        return None
    scenes = json.loads(scenes_file.read_text(encoding="utf-8"))
    thumbs = []
    for scene in scenes:
        num = scene.get("scene_number", 0)
        refs = sorted([p for p in images_dir.glob(f"ch{chapter_num:03d}_s{num:02d}_*.png")
                       if f"s{num:02d}b_" not in p.name])
        if refs:
            thumbs.append((num, Image.open(refs[-1]).convert("RGB")))
    if not thumbs:
        log("No reference images yet.")
        return None
    tw, th = 384, 216
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 28)), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for i, (num, im) in enumerate(thumbs):
        im.thumbnail((tw, th))
        x, y = (i % cols) * tw, (i // cols) * (th + 28)
        sheet.paste(im, (x + (tw - im.width) // 2, y + 14))
        draw.text((x + 8, y + 2), f"Sc {num}", fill=(255, 220, 100))
    out = Path(CONFIG["output_dir"]) / f"ch{chapter_num:03d}_storyboard.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    log(f"Storyboard: {out}")
    return out


def export_edl(chapter_num, scenes, output_path, novel=None):
    with _novel_context(novel):
        return _export_edl(chapter_num, scenes, output_path)


def _export_edl(chapter_num, scenes, output_path):
    """Editable timeline: clip list + timings + audio map + cost ledger."""
    images_dir = Path(CONFIG["images_dir"])
    voice_dir = _asset("voice")
    _hd = float(_LAST_ASSEMBLY.get("hook_duration", 0.0))
    t = _hd  # ponytail: no bridges; EDL matches clean-concat timeline
    events = []
    for scene in scenes:
        num = scene.get("scene_number", 0)
        dur = float(scene.get("_dur", scene.get("_cap_dur", CONFIG["scene_duration"])))
        refs = sorted([p.name for p in images_dir.glob(f"ch{chapter_num:03d}_s{num:02d}_*.png")
                       if f"s{num:02d}b_" not in p.name and f"s{num:02d}c_" not in p.name])
        vids = sorted([p.name for pat in ("h3", "echo", "ltx", "wan")
                       for p in images_dir.glob(f"ch{chapter_num:03d}_{pat}{num:02d}.mp4")])
        aud = voice_dir / f"ch{chapter_num:03d}_s{num:02d}.wav"
        t = float(scene.get("_cap_start", t))
        events.append({"scene": num, "start": round(t, 2), "duration": dur,
                       "narration": scene.get("narration", ""),
                       "dialogue": scene.get("dialogue", ""),
                       "images": refs, "video": vids,
                       "audio": aud.name if aud.exists() else None,
                       "shot": {k: scene.get(k, "") for k in
                                ("shot_size", "angle", "camera_move", "mood")}})
        t += dur
    edl = {"chapter": chapter_num, "output": str(output_path),
           "hook_duration": _hd, "bridge_duration": 0.0,
           "total_duration": round(t, 2), "events": events,
           "assembly": dict(_LAST_ASSEMBLY)}
    edl_path = Path(CONFIG["output_dir"]) / f"ch{chapter_num:03d}_edl.json"
    with open(edl_path, "w", encoding="utf-8") as f:
        json.dump(edl, f, indent=2, ensure_ascii=False)
    # Cost ledger: wall time + render counts (local GPU-time accounting)
    # Cost ledger: wall time + render counts (local GPU-time accounting).
    # GLOBAL file (like quality_log) because the auto-tuner reads one shared
    # history; every row carries its novel so the API can filter.
    cost_path = ROOT / "cost_log.json"
    costs = []
    if cost_path.exists():
        try:
            with open(cost_path, encoding="utf-8") as f:
                costs = json.load(f)
        except Exception:
            costs = []
    counts = {}
    for _kind in ("h3", "echo"):
        counts[_kind] = sum(
            1 for s in scenes if list(images_dir.glob(
                f"ch{chapter_num:03d}_{_kind}{s.get('scene_number', 0):02d}.mp4")))
    costs.append({"chapter": chapter_num, "novel": current_novel_id(),
                  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "scenes": len(scenes),
                  "h3_clips": counts["h3"],
                  "echo_clips": counts["echo"],
                  "output_mb": round(output_path.stat().st_size / 1048576, 1)
                  if output_path.exists() else 0})
    costs = costs[-200:]  # qa.md: logger caps at 200 rows
    with open(cost_path, "w", encoding="utf-8") as f:
        json.dump(costs, f, indent=2)
    log(f"  EDL + cost logged: {edl_path.name}")


def redo_scene(chapter_num, scene_num, novel=None):
    if novel is None:
        return produce_episode(chapter_num, redo_scene_num=scene_num)
    return produce_episode(chapter_num, redo_scene_num=scene_num, novel=novel)

# --- CLI -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Anime Pipeline - Episode Producer")
    parser.add_argument("chapter_pos", nargs="?", type=int, help="Chapter number (positional)")
    parser.add_argument("end_pos", nargs="?", type=int, help="End chapter for batch (positional)")
    parser.add_argument("--chapter", type=int, help="Chapter number to produce")
    parser.add_argument("--start", type=int, help="Start chapter for batch")
    parser.add_argument("--end", type=int, help="End chapter for batch")
    parser.add_argument("--platform",
                        choices=["tiktok", "reels", "shorts", "vertical", "youtube", "landscape"],
                        help="Output platform preset (default from config)")
    parser.add_argument("--vertical", action="store_true", help="Shortcut for --platform tiktok")
    parser.add_argument("--redo", nargs=2, type=int, metavar=("CHAPTER", "SCENE"),
                        help="Regenerate one scene and reassemble")
    parser.add_argument("--board", type=int, help="Storyboard contact sheet for chapter")
    parser.add_argument("--novel", help="Novel id to render (default: sinners_world)")
    parser.add_argument("--new-novel", metavar="TITLE",
                        help="Create a new novel folder and exit")
    parser.add_argument("--check", action="store_true", help="Check services only")
    args = parser.parse_args()

    if args.new_novel:
        from pipeline.novels import create_novel
        n = create_novel(args.new_novel)
        log(f"Created novel '{n.title}' ({n.id}) at {n.root}")
        return

    log("=" * 50)
    log("  ANIME PIPELINE - EPISODE PRODUCER")
    log("=" * 50)

    if args.vertical:
        CONFIG["platform"] = "tiktok"
    elif args.platform:
        CONFIG["platform"] = args.platform

    if args.check:
        check_services()
        return

    if _run_lock_held():
        log("Another render is already running. Aborting (one GPU job at a time).")
        return
    STOP_FILE.unlink(missing_ok=True)

    if args.redo:
        ch, sc = args.redo
        redo_scene(ch, sc, novel=args.novel)
        return

    if args.board:
        make_contact_sheet(args.board, novel=args.novel)
        return

    ok, _ = ensure_services(auto_start=True)
    if not ok:
        log("\nStart missing services and run again.")
        sys.exit(1)

    chapter = args.chapter or args.chapter_pos
    start, end = args.start, args.end
    if start is None and args.chapter_pos is not None and args.end_pos is not None:
        start, end = args.chapter_pos, args.end_pos

    if chapter and not (start and end):
        _guarded_produce(chapter, args.novel)
    elif start and end:
        for ch in range(start, end + 1):
            _guarded_produce(ch, args.novel)
    else:
        chapters_dir = Path(CONFIG["chapters_dir"])
        chapter_files_ = chapter_files(chapters_dir)
        for ch_file in chapter_files_:
            match = re.search(r"(\d+)", ch_file.stem)
            if match:
                num = int(match.group(1))
                _guarded_produce(num, args.novel)


def _guarded_produce(chapter_num, novel=None):
    """produce_episode with crash-log capture for ALL launch paths (CLI,
    detached, dashboard already has its own). Supervisor's crash-tail
    check picks these up automatically."""
    try:
        return produce_episode(chapter_num, novel=novel)
    except Exception:
        import traceback
        log(f"  [CRASH] chapter {chapter_num}, see render_crash.log")
        try:
            with open(ROOT / "render_crash.log", "a", encoding="utf-8") as f:
                f.write(f"\n=== {chapter_num} "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(traceback.format_exc())
            set_progress("error", 0, 1, "crashed, see render_crash.log")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
