# Local AI Anime Pipeline

Turn a text chapter into a short 1080p animated episode — locally, with no paid APIs.

The engine takes a chapter, writes a scene script, generates still images, turns them into short video clips, adds a spoken narration, and assembles the result into an MP4 with subtitles and a QA report.

> **This is a code-only release.** Model weights, voice samples, and example media are **not included** — you supply your own. See [What is not here](#what-is-not-here).

## What is not here

The following were deliberately left out so nothing private or copyrighted is published:

- **Your voice** — no `.wav`/`.m4a` recordings and no `ref.json`. You must record your own narration sample.
- **Any book or chapter** except a short, original example in `novels/example/`.
- **AI model weights** (around 60 GB) — download them yourself; see below.
- **Generated stills, clips, videos, and caches** — regenerate them by rendering.
- **Local machine state** — paths, tokens, logs, quality/rights ledgers, and the original author's notes.

## Requirements

- Windows 10/11, Python 3.12, NVIDIA GPU with CUDA-capable PyTorch (the original setup used an RTX 5080 Laptop / `sm_120` and cu128 builds).
- [ComfyUI Desktop](https://www.comfy.org/) with the Qwen-Image 2.1 still model, MiniMax H3 Ref2VA int8 motion model, and the EchoMimic lip-sync nodes.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` for script/scene generation.
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) for narration.
- `ffmpeg` on your PATH.

Check each model's own license before commercial use. This repo's MIT license covers the code only.

## Quick start

1. **Install Python dependencies** from the repo root: `pip install -r requirements.txt`. On Blackwell NVIDIA cards, install a `cu128` PyTorch build first (see the note in that file).
2. **Install the services** listed under Requirements and place the models from [MODELS.md](MODELS.md) in your ComfyUI and llama.cpp folders. This is the only large download — roughly 60 GB.
3. **Run `python setup.py`** once. It asks for your local paths, fills in `hermes-skill/scripts/config.json`, creates the example novel's folders, and writes a `cosy-server/ref.json` template. It never sends anything anywhere.
4. **Provide your own voice.** Record a short clean sample to `cosy-server/voice_reference.wav` and put its word-for-word transcript in `cosy-server/ref.json`. An empty transcript stops the voice server on purpose.
5. **Check readiness:** `python verify_setup.py`. It lists anything still missing — no guessing required.
6. **Start the services.** Double-click `Open Dashboard.bat`, or run `watchdog.bat` to keep the dashboard, ComfyUI, and voice server alive automatically.
7. **Open the dashboard** at `http://localhost:8000/`, pick the `example` novel, and start chapter 1. The finished MP4 lands in `novels/example/videos/`.

To use your own story, add a folder under `novels/` with `novel.json`, `chapters/`, and `characters/characters.json`, then register it in `novels/index.json`.

## Running the tests

From the `hermes-skill` folder:

```
python -m unittest discover -s scripts -p "test*.py"
```

## Rights and safety

Only render and publish material you own or have permission to adapt. The pipeline includes a rights check that marks non-commercial or unknown assets as **NOT_RELEASE_READY**, but it cannot make that decision for you. Never clone a third party's voice without documented consent.

## Layout

- `hermes-skill/scripts/` — the engine, dashboard, tests, and config.
- `rules/` — production guidelines the render consults.
- `cosy-server/` — voice server and reference-management tools.
- `novels/` — one folder per book; each owns its chapters, cast, images, voice, and videos.
- `watchdog.bat`, `comfy_launch.ps1`, `run_hidden.vbs`, `app.config` — startup and recovery helpers.

## License

MIT — see [LICENSE](LICENSE). Model weights, LoRAs, and third-party tools keep their own licenses.
