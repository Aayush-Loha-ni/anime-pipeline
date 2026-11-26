#!/usr/bin/env python3
"""Read-only preflight: reports what a fresh clone still needs before rendering."""
import json
import shutil
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "hermes-skill" / "scripts" / "config.json"


def ok(label, condition, detail=""):
    print(f"  [{'OK' if condition else 'MISSING'}] {label}" + (f" - {detail}" if detail else ""))
    return bool(condition)


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    cfg = json.loads(CFG.read_text(encoding="utf-8")) if CFG.exists() else {}
    print("Python / tools")
    ok("Python 3.10+", True)
    ok("ffmpeg on PATH", shutil.which("ffmpeg") is not None)

    print("\nConfiguration")
    ok("config.json exists", bool(cfg))
    placeholders = [v for v in cfg.values() if isinstance(v, str) and "<" in v and ">" in v]
    ok("no unfilled config placeholders", not placeholders, ", ".join(placeholders[:3]))

    print("\nModel / server paths")
    for key in ("llm_exe", "llm_gguf", "llm_vlm_gguf", "llm_vlm_mmproj"):
        v = cfg.get(key, "")
        ok(key, bool(v) and Path(v).is_file(), v)

    comfy = Path(cfg.get("comfyui_output", ""))
    shared_models = comfy.parent / "models" if comfy else Path("none")
    print(f"\nComfyUI models (looked in {shared_models})")
    for sub in ("diffusion_models", "text_encoders", "vae", "loras"):
        p = shared_models / sub
        files = list(p.glob("*")) if p.is_dir() else []
        ok(f"models/{sub}", bool(files), f"{len(files)} files")

    print("\nVoice reference")
    ref = ROOT / "cosy-server" / "ref.json"
    ref_ok = ref.is_file()
    text, wav = "", None
    if ref_ok:
        d = json.loads(ref.read_text(encoding="utf-8"))
        text = str(d.get("text", ""))
        wav = ROOT / d.get("wav", "")
    ok("ref.json exists", ref_ok)
    ok("reference transcript present", bool(text.strip()))
    ok("reference wav exists", bool(wav and Path(wav).is_file()), str(wav))

    print("\nExample novel")
    for rel in ("novels/example/chapters/chapter1.txt",
                "novels/example/characters/characters.json", "novels/index.json"):
        ok(rel, (ROOT / rel).is_file())

    print("\nServices (start them first if these are missing)")
    for port, name in ((8000, "dashboard"), (8080, "script LLM"), (8188, "ComfyUI"), (5003, "voice")):
        ok(f"{name} :{port}", port_open(port))


if __name__ == "__main__":
    main()
