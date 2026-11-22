"""File/image helpers — ported from blndev/AI-Anime-Maker src/utils.py.

SHA1-content filenames (= dedup for free), timestamped saves,
local model discovery, lazy downloads.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from hashlib import sha1
from pathlib import Path

logger = logging.getLogger(__name__)


def get_all_local_models(model_folder: str, extension: str = ".safetensors") -> list[str]:
    """List local model files under model_folder (recursive)."""
    out: list[str] = []
    try:
        for root, _, files in os.walk(model_folder, followlinks=True):
            for f in files:
                if f.endswith(extension):
                    out.append("./" + os.path.relpath(os.path.join(root, f)))
    except Exception as e:
        logger.error("Error listing model files: %s", e)
    return out


def download_file_if_not_existing(url: str, local_path: str) -> None:
    """Download url → local_path unless the file already exists."""
    if os.path.exists(local_path):
        return
    import requests  # lazy: pipeline servers often lack it until needed

    logger.info("Downloading %s ...", local_path)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(r.content)


def image_sha1(image) -> str:
    """Content hash of a PIL image (used for dedup + filenames)."""
    return sha1(image.tobytes()).hexdigest()


def save_image_as_file(image, dir: str | Path) -> str | None:
    """Save PIL image as JPEG named by SHA1 hash. Returns path or None."""
    try:
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        fp = d / (image_sha1(image) + ".jpg")
        if not fp.exists():
            image.save(fp, format="JPEG")
        return str(fp)
    except Exception as e:
        logger.error("Save image failed: %s", e)
        return None


def save_image_with_timestamp(image, folder_path: str | Path,
                              reference: str = "") -> str | None:
    """Save PIL image as {reference-}{timestamp}.png. Returns path or None.

    ponytail: `reference` was interpolated straight into a filename. A "/"
    made the save raise FileNotFoundError and return None — the function's own
    documented failure channel — for a perfectly good image, and ":" (illegal on
    Windows) produced a file nothing downstream could open. Sanitise, and add a
    content hash so two saves in the same second no longer overwrite silently.
    """
    try:
        d = Path(folder_path)
        d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{_safe_name(reference)}-{stamp}.png" if reference else f"{stamp}.png"
        fp = d / name
        image.save(fp)
        return str(fp)
    except Exception as e:
        logger.error("Save image failed: %s", e)
        return None


def _safe_name(value: str, limit: int = 40) -> str:
    """A filename-safe fragment: word chars, dot and dash only."""
    import re as _re
    cleaned = _re.sub(r"[^\w.-]", "_", str(value or "")).strip("._")
    return (cleaned or "image")[:limit]


def file_hash(path: str | Path) -> str:
    """SHA-256 hex digest of a file. Returns empty string on error."""
    import hashlib
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return ""
