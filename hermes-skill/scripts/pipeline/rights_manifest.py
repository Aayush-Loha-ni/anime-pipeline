#!/usr/bin/env python3
"""Rights manifest and preflight gate — v2.

Tracks provenance and commercial-use status for every non-trivial asset.
Blocks final assembly when any required item is unknown, non-commercial, or disallowed.

Each record includes: stable ID, used/unused state, exact version, local path,
SHA-256 when locally present, source URL, license URL or locally stored license text,
commercial status, reviewer, reviewed timestamp, and notes.
"""

import hashlib
import json
import re
import time
from pathlib import Path

from pipeline.schemas import RIGHTS_ENTRY

# stdlib-only, so this cannot cycle into pipeline.*
from json_util import dump_json_atomic


def _file_hash(path):
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return ""


def load_rights_manifest(root):
    """Load rights_manifest.json from project root. Returns dict.

    Fail CLOSED at the load boundary: a payload that is not a dict (a stray
    `[`, a torn write, a hand-edit gone wrong) used to be returned verbatim and
    every caller then died on `.get(...)` — turning one typo into a 500 from
    /api/readiness and a crash from the preflight rights gate. Callers already
    handle an empty manifest, so hand them one.
    """
    p = Path(root) / "rights_manifest.json"
    if not p.exists():
        return _empty_manifest()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_manifest()
    return data if isinstance(data, dict) else _empty_manifest()


def save_rights_manifest(manifest, root):
    """Save rights_manifest.json atomically."""
    p = Path(root) / "rights_manifest.json"
    # Shared writer: fsync + bounded replace retry + a RAISE if the swap never
    # lands. The hand-rolled version here used a fixed-name temp and no fsync,
    # and `load_rights_manifest` returns the parsed payload unchecked — so a
    # torn write took the rights gate down with it.
    dump_json_atomic(p, manifest, indent=2, ensure_ascii=False)
    return p


def _empty_manifest():
    return {
        "version": "2.0",
        "novel_adaptation_rights": {
            "asset_id": "novel_adaptation",
            "asset_type": "adaptation_rights",
            "source": "owner",
            "license": "original_work",
            "commercial_use": "unknown",
            "local_path_or_hash": "",
            "owner_confirmation": False,
            "reviewer": "",
            "reviewed_at": "",
            "notes": "Owner wrote the novel; adaptation rights need explicit confirmation"
        },
        "voice": {
            "asset_id": "voice_narrator",
            "asset_type": "voice_clone",
            "source": "owner",
            "license": "original_voice",
            "commercial_use": "unknown",
            "local_path_or_hash": "",
            "model_version": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "reference_file_hash": "",
            "owner_confirmation": False,
            "reviewer": "",
            "reviewed_at": "",
            "notes": "Owner's own voice, zero-shot clone via Qwen3-TTS "
                     "(Apache-2.0 weights, commercial-safe. Previous engine "
                     "XTTS v2 is CPML, non-commercial only — do not restore it)"
        },
        "image_models": [],
        "video_models": [],
        "llm_models": [],
        "vlm_models": [],
        "loras": [],
        "custom_nodes": [],
        "upscalers": [],
        "interpolators": [],
        "music": [],
        "sfx": [],
        "fonts": [],
        "stock_assets": [],
        "character_references": [],
        "entries": [],
        "last_verified": "",
        "notes": ""
    }


def add_entry(manifest, entry):
    """Add or update a rights entry. Returns errors list."""
    from pipeline.schemas import validate_dict
    real_errors = validate_dict(entry, RIGHTS_ENTRY, "rights_entry")
    if real_errors:
        return real_errors
    entries = manifest.setdefault("entries", [])
    existing = next((i for i, e in enumerate(entries)
                     if e.get("asset_id") == entry.get("asset_id")), None)
    if existing is not None:
        entries[existing] = entry
    else:
        entries.append(entry)
    manifest["last_verified"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return []


def add_model_license(manifest, model_name, license_type, commercial_use,
                      source="civitai/huggingface", model_type="image"):
    """Add a model license entry to the appropriate category."""
    entry = {
        "asset_id": f"model_{model_name}",
        "asset_type": "model",
        "source": source,
        "license": license_type,
        "commercial_use": commercial_use,
        "local_path_or_hash": model_name,
        "version": "",
        "reviewer": "",
        "reviewed_at": "",
        "notes": "",
    }
    category = {
        "image": "image_models",
        "video": "video_models",
        "llm": "llm_models",
        "vlm": "vlm_models",
    }.get(model_type, "image_models")
    lic = manifest.setdefault(category, [])
    existing = next((i for i, e in enumerate(lic) if e.get("asset_id") == entry["asset_id"]), None)
    if existing is not None:
        lic[existing] = entry
    else:
        lic.append(entry)


def add_audio_asset(manifest, asset_id, asset_type, source, license_type,
                    commercial_use, local_path="", notes=""):
    """Add an audio asset (voice, music, SFX) entry."""
    entry = {
        "asset_id": asset_id,
        "asset_type": asset_type,
        "source": source,
        "license": license_type,
        "commercial_use": commercial_use,
        "local_path_or_hash": local_path,
        "version": "",
        "reviewer": "",
        "reviewed_at": "",
        "notes": notes,
    }
    # Route to music or sfx category
    if asset_type == "music":
        cat = "music"
    elif asset_type == "sfx":
        cat = "sfx"
    else:
        cat = "entries"
    audio = manifest.setdefault(cat, [])
    existing = next((i for i, e in enumerate(audio) if e.get("asset_id") == asset_id), None)
    if existing is not None:
        audio[existing] = entry
    else:
        audio.append(entry)


def add_character_reference(manifest, character_id, source, license_type,
                            commercial_use, local_path="", notes=""):
    """Add a character reference asset entry."""
    entry = {
        "asset_id": f"char_ref_{character_id}",
        "asset_type": "character_reference",
        "source": source,
        "license": license_type,
        "commercial_use": commercial_use,
        "local_path_or_hash": local_path,
        "version": "",
        "reviewer": "",
        "reviewed_at": "",
        "notes": notes,
    }
    refs = manifest.setdefault("character_references", [])
    existing = next((i for i, e in enumerate(refs) if e.get("asset_id") == entry["asset_id"]), None)
    if existing is not None:
        refs[existing] = entry
    else:
        refs.append(entry)


def add_lora(manifest, lora_name, trigger, commercial_use, source="local",
             license_type="unknown", local_path="", notes=""):
    """Add a LoRA entry."""
    entry = {
        "asset_id": f"lora_{lora_name}",
        "asset_type": "lora",
        "source": source,
        "license": license_type,
        "commercial_use": commercial_use,
        "local_path_or_hash": local_path,
        "version": "",
        "trigger": trigger,
        "reviewer": "",
        "reviewed_at": "",
        "notes": notes,
    }
    loras = manifest.setdefault("loras", [])
    existing = next((i for i, e in enumerate(loras) if e.get("asset_id") == entry["asset_id"]), None)
    if existing is not None:
        loras[existing] = entry
    else:
        loras.append(entry)


def _commercial_status(value):
    """Normalize commercial_use to (allowed: bool, status: str). Fail-closed."""
    cu = str(value or "").strip().lower()
    if cu in ("yes", "y", "true", "allowed", "cleared", "approved"):
        return True, cu
    if cu.startswith("original"):
        return True, cu
    return False, cu or "missing"


# A fan work of someone else's novel: the owner may make it for personal use,
# but it is NOT cleared for publication or monetisation. It is a first-class
# state rather than a lie: recording it as "yes" would assert rights the
# owner does not hold, and recording it as "unknown" would block the render
# the owner is entitled to make.
NON_COMMERCIAL_VALUES = ("non-commercial", "noncommercial", "non_commercial",
                          "fan", "fan-work", "fan work", "fanfic",
                          "personal", "personal-use", "personal use", "non-profit")


def _is_non_commercial(value):
    cu = str(value or "").strip().lower()
    return cu in NON_COMMERCIAL_VALUES or cu.startswith("non-commercial")


def preflight_rights_gate(manifest, for_publish=False):
    """Check all rights entries. Returns (ok, blockers, warnings).

    Fail-closed: unknown / pending / missing commercial_use blocks release.

    `for_publish=True` is the RELEASE gate: nothing non-commercial passes it,
    so a fan work can be rendered but its release report stays blocked and
    the publisher refuses it. `for_publish=False` is the RENDER gate, where a
    confirmed non-commercial entry is allowed with a loud warning.
    """
    blockers = []
    warnings = []

    def _commercial_block(label, value):
        """One place: 'not commercially cleared' means different things for a
        render and for a publish."""
        allowed, status = _commercial_status(value)
        if allowed:
            return
        if _is_non_commercial(value):
            if for_publish:
                blockers.append(f"BLOCKER: {label} is recorded as non-commercial "
                                f"('{status}') - it cannot be published or monetised")
            else:
                warnings.append(f"WARNING: {label} is recorded as non-commercial "
                                f"('{status}') - personal use only, never publish")
            return
        blockers.append(f"BLOCKER: {label} commercial use '{status}' is not cleared")

    # Check novel rights
    novel = manifest.get("novel_adaptation_rights", {}) if isinstance(manifest, dict) else {}
    if not novel.get("owner_confirmation", False):
        blockers.append("BLOCKER: novel adaptation rights not confirmed by owner")
    _commercial_block("novel", novel.get("commercial_use", ""))

    # Check voice rights
    voice = manifest.get("voice", {}) if isinstance(manifest, dict) else {}
    if voice:
        if not voice.get("owner_confirmation", False):
            blockers.append("BLOCKER: narrator voice consent not confirmed by owner")
        _commercial_block("voice", voice.get("commercial_use", ""))

    # ponytail: one shape guard for all six category loops. The old code guarded
    # the CONTAINER with `manifest.get(cat, []) if isinstance(manifest, dict)`
    # and then did `row.get(...)` on each element — so a hand-edited manifest
    # with one string in `music` raised AttributeError and turned the release
    # gate into a 500 from /api/readiness. Verified. The conditional also read
    # as though it guarded the loop variable, which it never did.
    def _rows(category):
        rows = manifest.get(category, []) if isinstance(manifest, dict) else []
        if not isinstance(rows, list):
            rows = []
        return [r for r in rows if isinstance(r, dict)]

    # Check model categories
    for category in ("image_models", "video_models", "llm_models", "vlm_models"):
        for model in _rows(category):
            _commercial_block(f"{category} '{model.get('asset_id', '?')}'",
                              model.get("commercial_use", ""))

    # Check LoRAs
    for lora in _rows("loras"):
        _commercial_block(f"LoRA '{lora.get('asset_id', '?')}'",
                          lora.get("commercial_use", ""))

    # Check audio
    for category in ("music", "sfx"):
        for audio in _rows(category):
            _commercial_block(f"{category} '{audio.get('asset_id', '?')}'",
                              audio.get("commercial_use", ""))

    # Check character references
    for ref in _rows("character_references"):
        _commercial_block(f"character ref '{ref.get('asset_id', '?')}'",
                          ref.get("commercial_use", ""))

    # Check general entries
    for entry in _rows("entries"):
        _commercial_block(f"asset '{entry.get('asset_id', '?')}'",
                          entry.get("commercial_use", ""))
        if not entry.get("local_path_or_hash"):
            warnings.append(f"WARNING: asset '{entry.get('asset_id', '?')}' "
                            f"missing hash/path")

    ok = len(blockers) == 0
    return ok, blockers, warnings


def check_prompt_rights(prompt_text, manifest=None):
    """Check if a prompt contains imitation of named studios/artists/franchises.

    Returns (ok: bool, reason: str).
    manifest parameter is accepted for API compatibility but not currently used.
    """
    forbidden = [
        r"(?:in\s+the\s+style\s+of|imitat|copy)\s+(?:studio\s+)?ghibli",
        r"(?:in\s+the\s+style\s+of|imitat|copy)\s+makoto\s+shinkai",
        r"(?:in\s+the\s+style\s+of|imitat|copy)\s+hayao\s+miyazaki",
        r"(?:in\s+the\s+style\s+of|imitat|copy)\s+takehiko\s+inoue",
        r"studio\s+ghibli\s+(?:style|look|aesthetic)",
        r"(?:dragon\s+ball|naruto|one\s+piece|attack\s+on\s+titan)\s+(?:style|fan)",
        r"(?:in\s+the\s+style\s+of|imitat|copy)\s+[\w\s]+(?:anime|manga)\s+(?:studio|artist)",
        r"(?:real\s+person|celebrity|politician)\s+(?:likeness|portrait|photo|image)",
        r"(?:portrait|photo|image)\s+of\s+(?:real\s+person|celebrity|politician)",
        r"(?:likeness|portrait)\s+of\s+[\w\s]+(?:actor|actress|singer|athlete)",
    ]
    text = prompt_text.lower()
    for pat in forbidden:
        if re.search(pat, text):
            return False, f"Prompt imitates named studio/artist/franchise: {pat}"
    return True, ""


def check_safety(text, text_type="prompt"):
    """Basic safety check on text content. Returns (ok, reason).

    Checks for: real person names, protected franchises, unsafe content.
    """
    if not text or not isinstance(text, str):
        return True, ""

    lower = text.lower()

    # Real person / studio imitation
    ok, reason = check_prompt_rights(text, {})
    if not ok:
        return False, reason

    # Protected character names outside approved list
    protected = [
        r"\b(?:naruto|sasuke|goku|vegeta|luffy|ichigo)\b",
        r"\b(?:pikachu|ash\s+ketchum|mickey\s+mouse)\b",
        r"(?:real\s+person|celebrity)\s+(?:portrait|photo|likeness)",
        r"(?:portrait|photo)\s+of\s+(?:real\s+person|celebrity)",
    ]
    for pat in protected:
        if re.search(pat, lower):
            return False, f"Text references protected character: {pat}"

    return True, ""


def populate_initial_manifest(root, config):
    """Create or update rights_manifest.json with known pipeline assets.

    Non-destructive: existing entries are preserved.
    """
    manifest = load_rights_manifest(root)

    # Novel rights
    novel = manifest.get("novel_adaptation_rights", {})
    if not novel.get("local_path_or_hash"):
        chapters_dir = Path(config.get("chapters_dir", ""))
        if chapters_dir.exists():
            first_chapter = next(chapters_dir.glob("*.txt"), None)
            if first_chapter:
                novel["local_path_or_hash"] = _file_hash(first_chapter)
                manifest["novel_adaptation_rights"] = novel

    # Voice
    voice = manifest.get("voice", {})
    voice_ref = Path(root) / "cosy-server" / "ref.json"
    if voice_ref.exists():
        try:
            ref = json.loads(voice_ref.read_text(encoding="utf-8"))
            wav = ref.get("wav", "")
            if wav:
                wav_path = Path(root) / "cosy-server" / wav
                if wav_path.exists():
                    voice["reference_file_hash"] = _file_hash(wav_path)
                    voice["local_path_or_hash"] = wav
            manifest["voice"] = voice
        except Exception:
            pass

    # Image models (Qwen-Image-2.1, Apache-2.0 family)
    checkpoint = config.get("checkpoint", "")
    if checkpoint:
        add_model_license(manifest, checkpoint,
                         "Qwen-Image-2.1 (Apache-2.0)",
                         "yes", "local", "image")

    # Video models (MiniMax H3, H3 Community License)
    h3 = config.get("h3_checkpoint", "")
    if h3:
        add_model_license(manifest, h3,
                         "MiniMax-H3-Ref2VA-int8 (H3 Community License)",
                         "yes", "local", "video")

    # Music — default to unknown until verified
    music_dir = Path(config.get("music_dir", ""))
    if music_dir.exists():
        for mp3 in music_dir.glob("*.mp3"):
            add_audio_asset(manifest, f"music_{mp3.stem}", "music",
                          "local", "unknown",
                          "unknown", str(mp3),
                          "Music asset; license needs verification")

    # Character references: there is no per-character image folder any more
    # (the model sheets are gone), so a character is recorded by hand with
    # add_character_reference() when its design is first used.

    # LoRAs
    loras_dir = Path(root) / "loras"
    if loras_dir.exists():
        for lora in loras_dir.glob("*.safetensors"):
            # Same reasoning: a license file sitting next to a LoRA is not
            # proof of its commercial terms. "unknown" until reviewed.
            add_lora(manifest, lora.stem, "", "unknown", "local",
                    "unknown", str(lora),
                    f"LoRA: {lora.stem} — confirm license")

    save_rights_manifest(manifest, root)
    return manifest


def derive_manifest_from_production(root, edl, sidecars):
    """Derive rights manifest from actual production assets used.

    This is the authoritative method: the final manifest reflects only what
    was actually used in the final render, not a hard-coded starter list.
    """
    manifest = load_rights_manifest(root)

    # Collect all asset IDs referenced in EDL and sidecars
    used_assets = set()
    used_models = set()
    used_loras = set()

    if isinstance(edl, dict):
        for scene in edl.get("scenes", []):
            if isinstance(scene, dict):
                for key in ("clip_path", "image_path", "audio_path", "music_path"):
                    val = scene.get(key, "")
                    if val:
                        used_assets.add(val)
                # Track model/LoRA from sidecars
                for sidecar_key in ("sidecar", "manifest"):
                    sc = scene.get(sidecar_key, {})
                    if isinstance(sc, dict):
                        used_models.add(sc.get("checkpoint", ""))
                        for lora in sc.get("loras", []):
                            if isinstance(lora, dict):
                                used_loras.add(lora.get("name", ""))
                            elif isinstance(lora, str):
                                used_loras.add(lora)

    if isinstance(sidecars, dict):
        for path, sc in sidecars.items():
            if isinstance(sc, dict):
                used_models.add(sc.get("checkpoint", ""))
                for lora in sc.get("loras", []):
                    if isinstance(lora, dict):
                        used_loras.add(lora.get("name", ""))
                    elif isinstance(lora, str):
                        used_loras.add(lora)

    # Mark unused entries
    for category in ("image_models", "video_models", "llm_models", "vlm_models"):
        for entry in manifest.get(category, []):
            aid = entry.get("asset_id", "")
            entry["used"] = any(m in aid for m in used_models if m)

    for lora in manifest.get("loras", []):
        aid = lora.get("asset_id", "")
        lora["used"] = any(l in aid for l in used_loras if l)

    return manifest
