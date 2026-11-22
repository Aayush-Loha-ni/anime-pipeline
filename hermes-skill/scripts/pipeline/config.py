"""Cascading INI config — ported from blndev/AI-Anime-Maker src/config.py.

Resolution order (later files override earlier ones):
  <project-root>/app.config → <project-root>/local.config → <project-root>/dev.config
  + same three names inside hermes-skill/scripts/ (per-machine overrides).

Works with zero files present — every getter has a default.
Env overrides: SKIP_GENAI=1 disables captioner/model helpers.
"""
from __future__ import annotations

import logging
import os
from configparser import ConfigParser
from pathlib import Path

logger = logging.getLogger(__name__)

current_config: ConfigParser | None = None


def skip_genai() -> bool:
    """True when GenAI helpers (the BLIP captioner) must stay unloaded.

    ponytail: this was a module-level global, `SKIP_GENAI = os.getenv(...) == "1"`,
    evaluated ONCE at import and then latched with `SKIP_GENAI = SKIP_GENAI or ...`.
    Both halves were wrong:
      * import-time evaluation means the value depends on which module happened
        to import pipeline.config first — so SKIP_GENAI=1 set by a test module
        loaded after the server had no effect, and the captioner tried to pull a
        ~1GB model into VRAM beside ComfyUI;
      * the `or` latch means the flag can never be turned back off, so a test
        could not clear it.
    A function has neither problem, and it is the only reader in the codebase.
    """
    if os.getenv("SKIP_GENAI") == "1":
        return True
    return get_bool("GenAI", "skip", False)


def debug_enabled() -> bool:
    """[General] debug. Was the `DEBUG` global, which had ZERO readers."""
    return get_bool("General", "debug", False)


def _config_files() -> list[str]:
    # Path(...) and .parent cannot raise, so this used to sit in a try/except
    # whose fallback returned CWD-relative names — a WORSE outcome than the
    # primary path, since it would silently read a different app.config.
    scripts = Path(__file__).parent.parent
    root = scripts.parent.parent  # <project-root>
    out = []
    for base in (root, scripts):
        for name in ("app.config", "local.config", "dev.config"):
            out.append(str(base / name))
    return out


def read_configuration() -> ConfigParser | None:
    """(Re)read all config files. Later files override earlier ones.

    Returns None on failure. Note the flags are NOT latched here any more —
    skip_genai()/debug_enabled() recompute on every call.
    """
    global current_config
    try:
        current_config = ConfigParser()
        found = current_config.read(_config_files())
        logger.debug("Config files loaded: %s", found)
        return current_config
    except Exception as e:
        logger.error("Failed to read configuration: %s", e)
        return None


def _cfg() -> ConfigParser:
    if current_config is None:
        read_configuration()
    return current_config


def get(section: str, option: str, default: str = "") -> str:
    try:
        cfg = _cfg()
        if cfg is not None and cfg.has_option(section, option):
            return cfg.get(section, option)
    except Exception:
        pass
    return default


def get_bool(section: str, option: str, default: bool = False) -> bool:
    try:
        cfg = _cfg()
        if cfg is not None and cfg.has_option(section, option):
            return cfg.getboolean(section, option)
    except Exception:
        pass
    return default


def get_float(section: str, option: str, default: float = 0.0) -> float:
    try:
        cfg = _cfg()
        if cfg is not None and cfg.has_option(section, option):
            return cfg.getfloat(section, option)
    except Exception:
        pass
    return default


def get_int(section: str, option: str, default: int = 0) -> int:
    try:
        cfg = _cfg()
        if cfg is not None and cfg.has_option(section, option):
            return cfg.getint(section, option)
    except Exception:
        pass
    return default


# --- GenAI section -------------------------------------------------------
def genai_max_size(default: int = 1024) -> int:
    return get_int("GenAI", "max_size", default)


def genai_default_strength(default: float = 0.5) -> float:
    v = get_float("GenAI", "default_strength", default)
    return v if 0 < v < 1 else default


def genai_default_steps(default: int = 50) -> int:
    v = get_int("GenAI", "default_steps", default)
    return v if 10 <= v <= 100 else default


def genai_model_folder(default: str = "./models/") -> str:
    return get("GenAI", "model_folder", default)


def caption_model(default: str = "Salesforce/blip-image-captioning-base") -> str:
    return get("GenAI", "caption_model", default)


# --- Token section -------------------------------------------------------
def tokens_enabled(default: bool = False) -> bool:
    return get_bool("Token", "enabled", default)


def token_cost_render(default: int = 1) -> int:
    return get_int("Token", "render_cost", default)


def token_cost_redo(default: int = 1) -> int:
    return get_int("Token", "redo_cost", default)


def token_grant_new_image(default: int = 3) -> int:
    return get_int("Token", "new_image", default)
