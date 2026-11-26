#!/usr/bin/env python3
"""One-time setup for a fresh clone. No personal data is written or uploaded.

Run:  python setup.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "hermes-skill" / "scripts" / "config.json"


def ask(prompt, default=""):
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def main():
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ is required (3.12 recommended).")

    cfg = json.loads(CFG.read_text(encoding="utf-8"))

    print("Paths to your own local installs (nothing is downloaded by this script):")
    llama_dir = ask("Folder containing llama-server.exe and models/", "C:\\llama.cpp")
    cfg["llm_exe"] = ask("Path to llama-server.exe", str(Path(llama_dir) / "llama-cpp" / "llama-server.exe"))
    cfg["llm_gguf"] = ask("Path to the script-model GGUF", str(Path(llama_dir) / "models" / "script-model.gguf"))
    cfg["llm_vlm_gguf"] = ask("Path to the vision-model GGUF", str(Path(llama_dir) / "models" / "vision-model.gguf"))
    cfg["llm_vlm_mmproj"] = ask("Path to the vision mmproj GGUF", str(Path(llama_dir) / "models" / "mmproj.gguf"))

    comfy_dir = ask("ComfyUI Desktop folder (holds ComfyUI-Shared and ComfyUI-Installs)", "C:\\Users\\<you>\\AppData\\Local\\Comfy-Desktop")
    cfg["comfyui_input"] = str(Path(comfy_dir) / "ComfyUI-Shared" / "input")
    cfg["comfyui_output"] = str(Path(comfy_dir) / "ComfyUI-Shared" / "output")

    model_dir = ask("Folder with Qwen-Image weights (leave blank to skip LoRA training)", "")
    if model_dir:
        cfg["lora_training"]["model"] = str(Path(model_dir) / "your-image-model.safetensors")
        cfg["lora_training"]["extras"] = model_dir
        cfg["lora_training"]["aitk_dir"] = ask("Path to ai-toolkit folder", str(Path(model_dir) / "ai-toolkit"))
    else:
        cfg["lora_training"] = {"enabled": False}

    CFG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for key in ("output_dir", "chapters_dir", "images_dir", "scenes_dir", "music_dir", "sfx_dir"):
        p = Path(cfg[key])
        (ROOT / p if not p.is_absolute() else p).mkdir(parents=True, exist_ok=True)
    Path(cfg["characters_file"]).parent.mkdir(parents=True, exist_ok=True)

    ref = ROOT / "cosy-server" / "ref.json"
    if not ref.exists():
        example = json.loads((ROOT / "cosy-server" / "ref.example.json").read_text(encoding="utf-8"))
        example["wav"] = str(ROOT / "cosy-server" / "voice_reference.wav")
        example["text"] = "Replace this with a transcript of your own short recording."
        ref.write_text(json.dumps(example, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Created cosy-server/ref.json from the template.")

    print("\nSetup written. Next:")
    print("  1. Place the models listed in MODELS.md in your ComfyUI models folders.")
    print("  2. Record a short clean voice sample to cosy-server/voice_reference.wav")
    print("     and put its exact transcript in cosy-server/ref.json (empty text stops the server).")
    print("  3. Run: python verify_setup.py")
    print("  4. Start the dashboard with Open Dashboard.bat (or watchdog.bat).")


if __name__ == "__main__":
    main()
