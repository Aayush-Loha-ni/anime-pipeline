# Anime Pipeline Skill

Novel chapters → anime videos, controlled by `scripts/config.json`.

## Local stack

  llama server :8080: configured script model (currently `gemma-4-26B-A4B` uncensored/heretic IQ4_XS).
  ComfyUI :8188: Qwen Image 2.1 stills (GGUF DiT, text encoder on CPU), MiniMax H3 Ref2VA pruned int8 motion (4-step turbo LoRA, 20 steps, 960x544 fast / 1280x720 final) and EchoMimic lip sync.
  Qwen3-TTS :5003: single TTS engine, user-voice zero-shot clone (`cosy-server/ref.json`).
  FFmpeg assembly v2: audio master first stereo assembly, music ducking, loudness normalization and QA gated promotion.
  Dashboard: `scripts/pipeline_server.py` on :8000; root `watchdog.bat` handles guarded recovery and appends startup output to `%TEMP%\opencode\*_watchdog.log`.

## Quick start

1. From the project root, run `watchdog.bat`, then `Open Dashboard.bat`.
2. Or from `hermes-skill/`, run `python scripts/episode_producer.py 63`.
3. Current defaults: `videos/ch001_youtube.mp4`, 1920×1080 at 24 fps; hook cards off; captions ON and burned in (`subtitles: true`), 42 chars / 17 CPS / sentence case. Duration follows narration, not a fixed episode length.

## Novels

One book = one folder. `novels/<slug>/` holds that book's `chapters/ scenes/ images/ voice/
videos/ characters/ dataset/` plus a `novel.json` (title, art direction, config overrides).
The engine (models, ComfyUI, the voice clone, `rules/`) is shared; one episode renders at a
time. The built-in `sinners_world` novel keeps the legacy layout at the project root.

```bash
python scripts/episode_producer.py --new-novel "My Novel"   # create the folder + register it
python scripts/episode_producer.py 1 --novel my_novel        # render one chapter of it
python scripts/episode_producer.py --board 1 --novel my_novel
```

Dashboard: pick the novel in the dropdown (or create one with "New novel"); every API route
accepts `novel=<id>` and defaults to `sinners_world`.

Run regression checks from `hermes-skill/`: `python -m unittest discover -s scripts -p "test*.py"`.

See `SKILL.md` for workflow and ownership notes.
