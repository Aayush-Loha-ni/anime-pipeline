"""Local image captioning — ported from blndev/AI-Anime-Maker src/AI.py.

BLIP-base auto-describes reference/candidate stills so the VLM
reference selector gets text even when scene JSON has empty prompts.
Lazy imports: torch/transformers load only on first caption call,
never at server startup. SKIP_GENAI=1 (or [GenAI] skip=true) → "".
"""
from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_PIPELINE = None
# Two concurrent /api/caption requests both saw _PIPELINE is None and both
# loaded a full BLIP model into VRAM — an OOM on a 16GB card that is
# already hosting ComfyUI. One lock, one model.
_LOAD_LOCK = threading.Lock()


def available() -> bool:
    """True unless captioning is explicitly skipped."""
    from pipeline import config as _cfg
    return not _cfg.skip_genai()


def _load():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    with _LOAD_LOCK:
        if _PIPELINE is not None:  # another thread won the race
            return _PIPELINE
        from transformers import pipeline  # lazy: heavy
        from pipeline import config as _cfg
        # ponytail: no device= meant transformers defaulted to CPU, so every
        # caption was a CPU forward pass on a GPU box. Use CUDA when present;
        # BLIP-base is ~1GB and trivially fits alongside ComfyUI.
        kwargs = {"model": _cfg.caption_model()}
        try:
            import torch
            if torch.cuda.is_available():
                kwargs["device"] = 0
        except Exception:
            pass
        _PIPELINE = pipeline("image-to-text", **kwargs)
        return _PIPELINE


def _unload() -> None:
    global _PIPELINE
    try:
        with _LOAD_LOCK:
            if _PIPELINE is not None:
                _PIPELINE = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception as e:
        logger.error("Captioner unload failed: %s", e)


def describe_image(image) -> str:
    """Caption a PIL image (or path). Returns "" on any failure — never raises."""
    if not available():
        return ""
    try:
        if isinstance(image, (str, Path)):
            from PIL import Image
            image = Image.open(image).convert("RGB")
        pipe = _load()
        out = pipe(image)
        return (out[0].get("generated_text", "") or "").strip()
    except Exception as e:
        logger.warning("Captioning failed: %s", e)
        _unload()
        return ""


def describe_many(paths: list, max_items: int = 8) -> dict:
    """Caption up to max_items image paths → {path: caption}. Failures → ""."""
    return {p: describe_image(p) for p in list(paths)[:max(0, max_items)]}
