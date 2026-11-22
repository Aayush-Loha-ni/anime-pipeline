"""YAML config loader with cascading files + env overrides — ported from vedgen/config.py.

Resolution order (later overrides earlier):
  <project-root>/config.yaml → <project-root>/local.yaml → <project-root>/dev.yaml
  + same three names inside hermes-skill/scripts/ (per-machine overrides).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a, **_k):  # type: ignore
        return False


def _find_root(start: Optional[Path] = None) -> Path:
    """Locate the project root.

    ponytail: try the module's own location FIRST. Walking up from cwd only
    meant that running `python <abs path>/episode_producer.py` from an
    unrelated directory found no config.yaml, returned cwd, and
    load_config raised FileNotFoundError — which episode_producer swallows
    into one stderr line. The whole H3 engine config (checkpoint, VAE,
    steps, frames, quality tier) then silently fell back to hardcoded
    defaults and the episode rendered at the fast tier with no visible
    error. Module-relative is always right; cwd is a fallback.
    """
    here = Path(__file__).resolve()
    mod_root = here.parents[3] if len(here.parents) > 3 else here.parent
    if (mod_root / "hermes-skill" / "scripts" / "pipeline" / "config.yaml").exists():
        return mod_root
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "hermes-skill" / "scripts" / "pipeline" / "config.yaml").exists():
            return p
        if (p / "config.yaml").exists():
            return p
    return mod_root


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    root = _find_root()
    load_dotenv(root / ".env")
    path = Path(config_path) if config_path else root / "hermes-skill" / "scripts" / "pipeline" / "config.yaml"
    if not path.is_absolute():
        path = (root / path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(root)
    # env overrides
    if os.environ.get("FFMPEG_PATH"):
        cfg.setdefault("paths", {})["ffmpeg"] = os.environ["FFMPEG_PATH"]
    if os.environ.get("SKIP_GENAI"):
        cfg.setdefault("captioner", {})["skip_genai"] = os.environ["SKIP_GENAI"] == "1"
    if os.environ.get("VIDEO_ENGINE"):
        cfg.setdefault("generation", {})["video_engine"] = os.environ["VIDEO_ENGINE"]
    if os.environ.get("QUALITY"):
        cfg.setdefault("generation", {})["quality"] = os.environ["QUALITY"]
    return cfg


def project_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg.get("_root") or _find_root())


def get_engine_config(cfg: Dict[str, Any], engine: str = "h3", quality: str = "fast") -> Dict[str, Any]:
    """Return merged model config for engine + quality tier with fast fallback."""
    if engine not in ("h3",):
        engine = "h3"
    if quality not in ("fast", "final"):
        quality = "fast"
    models = cfg.get("models", {})
    engine_cfg = models.get(engine, {})
    tier_cfg = engine_cfg.get(quality, {})
    # fast fallback flag
    fallback = engine_cfg.get("oom_fallback_to_fast", True)
    # merge base engine config (shared vae/text_encoder) with tier-specific
    base = {k: v for k, v in engine_cfg.items() if k not in ("fast", "final", "oom_fallback_to_fast")}
    out = {**base, **tier_cfg}
    out["_engine"] = engine
    out["_quality"] = quality
    out["_fallback"] = fallback
    return out


def get_active_engine_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience: engine + quality from generation section."""
    gen = cfg.get("generation", {})
    return get_engine_config(cfg, gen.get("video_engine", "h3"), gen.get("quality", "fast"))