"""Camera Presets — cinematic lens/focal/aperture presets for prompt enrichment.

Inspired by Open Higgsfield AI's Cinema Studio controls.
Each preset maps to prompt modifier strings for image/video generation.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CameraPreset:
    key: str
    label: str
    lens: str
    focal_mm: str
    aperture: str
    prompt_mod: str  # appended to image_prompt for generation

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "lens": self.lens,
                "focal_mm": self.focal_mm, "aperture": self.aperture,
                "prompt_mod": self.prompt_mod}


LENSES = [
    ("classic-anamorphic", "Classic Anamorphic", "anamorphic lens, oval bokeh, horizontal flares"),
    ("modern-prime", "Modern Prime", "clean modern prime lens, sharp center, smooth bokeh"),
    ("cinema-prime", "70s Cinema Prime", "vintage cinema prime, warm halation, gentle softness"),
    ("vintage-prime", "Vintage Prime", "vintage lens, light leaks, soft glow"),
    ("tilt-shift", "Creative Tilt", "tilt-shift miniature effect, selective focus"),
    ("macro", "Extreme Macro", "extreme macro, razor-thin depth of field"),
    ("swirl-bokeh", "Swirl Bokeh Portrait", "swirly bokeh, petzval-style background"),
    ("clinical", "Clinical Sharp Prime", "extremely sharp, clinical precision, no distortion"),
    ("halation", "Halation Diffusion", "halation glow, diffusion filter, dreamy atmosphere"),
]

FOCAL_LENGTHS = [
    ("8mm", "8mm Ultra-Wide", "ultra-wide fisheye, dramatic distortion, vast field"),
    ("14mm", "14mm Wide", "ultra-wide, deep perspective, architectural scale"),
    ("24mm", "24mm Wide", "wide-angle, environmental context, slight barrel"),
    ("35mm", "35mm Human Eye", "natural field of view, human-eye perspective"),
    ("50mm", "50mm Portrait", "standard portrait focal length, minimal distortion"),
    ("85mm", "85mm Tight Portrait", "tight portrait, compressed background, flattering"),
]

APERTURES = [
    ("f1.4", "f/1.4 Shallow DoF", "extremely shallow depth of field, creamy bokeh"),
    ("f4", "f/4 Balanced", "balanced focus, moderate background separation"),
    ("f11", "f/11 Deep Focus", "deep focus, everything sharp, landscape detail"),
]


def build_camera_modifiers(lens: str = "", focal: str = "", aperture: str = "") -> str:
    """Combine lens + focal + aperture into a prompt modifier string."""
    parts = []
    for table, val in [(LENSES, lens), (FOCAL_LENGTHS, focal), (APERTURES, aperture)]:
        if not val:
            continue
        match = next((p[2] for p in table if p[0] == val), "")
        if match:
            parts.append(match)
    return ", ".join(parts)


def get_camera_preset(key: str) -> CameraPreset | None:
    """Look up a preset by its key (lens, focal, or aperture key)."""
    for table, cat in [(LENSES, "lens"), (FOCAL_LENGTHS, "focal"), (APERTURES, "aperture")]:
        for k, label, mod in table:
            if k == key:
                return CameraPreset(key=k, label=label, lens=cat, focal_mm="", aperture="",
                                    prompt_mod=mod)
    return None


def all_presets() -> dict:
    """Return all preset tables for UI consumption."""
    return {
        "lenses": [{"key": k, "label": l, "prompt_mod": m} for k, l, m in LENSES],
        "focal_lengths": [{"key": k, "label": l, "prompt_mod": m} for k, l, m in FOCAL_LENGTHS],
        "apertures": [{"key": k, "label": l, "prompt_mod": m} for k, l, m in APERTURES],
    }
