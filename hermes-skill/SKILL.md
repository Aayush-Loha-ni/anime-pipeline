---
name: anime pipeline skill
description: Produce anime videos from novel chapters using llama.cpp, ComfyUI, Qwen3-TTS and FFmpeg.
version: 2.0.0
author: [hermes-agent]
license: MIT
platforms: [windows]
compatibility: "Requires llama.cpp :8080 (Gemma-4-26B-A4B GGUF), ComfyUI (Qwen-Image-2.1, MiniMax H3 Ref2VA int8, EchoMimic), Qwen3-TTS :5003 and FFmpeg."
prerequisites:
  commands: ["python", "ffmpeg"]
metadata:
  hermes:
    tags: [anime, video-generation, comfyui, llm, tts, ffmpeg]
    category: creative
---

# Anime Pipeline

Use for chapter-to-video production, episode rebuilds and scene redo requests.

## Stack

- llama-server :8080: script/scene generation via OpenAI-compatible `/v1/chat/completions`; GGUF + flags in `scripts/config.json` (`llm_*`) and `<LLAMA_DIR>\models\` (currently gemma-4-26B-A4B uncensored/heretic IQ4_XS, 8k ctx, q8 KV). `scripts/launch_llm.py` starts it from that same config — use it rather than hand-writing a llama-server command, which is how the watchdog drifted onto a stale model path before.
- ComfyUI :8188: Qwen-Image-2.1 stills; MiniMax H3 Ref2VA int8 motion, EchoMimic lip-sync.
- Qwen3-TTS :5003 (`cosy-server/server.py`): the single TTS engine, user-voice zero-shot clone configured by `cosy-server/ref.json`. Needs a reference TRANSCRIPT in that file — an empty `text` aborts startup by design.
- FFmpeg assembly v2: audio-master-first assembly, stereo audio, music ducking and loudness normalization; QA gates media promotion.
- Dashboard: `scripts/pipeline_server.py` on :8000. Root `watchdog.bat` checks services; hidden launches append stdout/stderr and watchdog actions to `%TEMP%\opencode\*_watchdog.log`.

## Commands

From `hermes-skill/`:

```bash
python scripts/episode_producer.py 63
python scripts/episode_producer.py 63 70
python scripts/episode_producer.py 63 --platform youtube
python scripts/episode_producer.py --new-novel "My Novel"
python scripts/episode_producer.py 1 --novel my_novel
```

From the project root, run `watchdog.bat` for service recovery or `Open Dashboard.bat` to open the dashboard. Recovery retries health checks, skips live renders and never kills unverified listeners. `watchdog.bat` cycles every `WD_INTERVAL` seconds (default 60, minimum 5): a bare run starts a minimised `--loop` copy and returns immediately, so the dashboard can be opened straight after. Only one loop ever runs. The claim is an **atomic compare-and-swap**: a fresh lock is a single failing `New-Item`, and a stale lock is reclaimed by an atomic `Rename-Item` (NTFS guarantees exactly one racer wins) — verified with 8 concurrent claimers. The heartbeat beats *before* and after the cycle sleep, and `WD_STALE_S` defaults to `WD_INTERVAL*10 + 300` so a slow recovery cycle cannot outrun it. Stop with `taskkill /FI "WINDOWTITLE eq anime-watchdog" /F`.

## Configuration and ownership

`scripts/config.json` is authoritative — and the only file the render reads. `config.yaml` is **warning-only** for every key that already has a value (built-in or JSON): it prints `[CONFIG] key: config.yaml=X ignored — config.json=Y is authoritative` on any disagreement and contributes nothing else, because every `h3_*` key it declares now exists in `config.json`. T2I/image params live in `config.json` only; do not re-add a `t2i:` block to `pipeline/config.yaml`. Its `token:`, `captioner:`, `paths:`, `comfyui:`, `debug` and `analytics_enabled` blocks have **no reader** — those live in the root `app.config` (INI) and in `config.json`. `POST /api/config/engine` writes `config.json` and updates the live `CONFIG`, and returns `applied: true|false` so a no-op cannot be reported as success. Outputs live in `videos/`.

Character tags live in `characters/characters.json`. Chapter/scene seeds support repeatable stills. Render ownership, stop flow and QA remain in `episode_producer.py` and `pipeline_server.py`; do not bypass their gates. Post-render rating and the auto-tuner moved to `pipeline/quality_evolve.py` (re-exported from `episode_producer`, so `ep.evolve_quality()` still works).

## Novels (more than one book)

A **novel** owns its chapters, scripts, stills, voice, videos, characters, dataset and pins;
the engine (models, ComfyUI, the voice clone, `rules/`) is shared and one episode renders at a
time. The registry is `novels/index.json`; each novel is `novels/<slug>/` with a `novel.json`
(title, art_direction, config overrides). `pipeline/novels.py` is the only module that knows
the layout.

- `sinners_world` relocated into `novels/Example Novel/` (its assets and `series_bible.json` /
  `pinned.json` / `rights_manifest.json` live there; the project root holds only the shared engine).
  The pipeline resolves it there when the folder exists (`relocated`), i.e. with its own paths; the
  tests' fake legacy novels stay on `ROOT` because they never create that folder. It is always
  listed first.
- `episode_producer._novel_context(novel_id)` binds one novel to a render: it swaps `CONFIG`
  (chapters/scenes/images/voice/videos/characters/music/sfx) and the character bible for the
  duration, and restores them afterwards. **Any new code that writes an asset must use
  `_asset("<dir>")` or `CONFIG[...]`, never a bare `ROOT / "..."`** — a bare root path writes
  into the wrong book's folder.
- Chapter lookup is scoped: `find_chapter_file(n)` searches the ACTIVE novel's `chapters/`,
  so two books may both ship a "Chapter 1".
- Shared, one copy for every novel: models, ComfyUI, `rules/`, `cosy-server/` (the voice
  clone), `_models/`, `_training/`, `issues.json`, `tokens.json`, render state
  (`render.lock`/`render.pid`/`stop.requested`), and the tuner's `quality_log.json` /
  `cost_log.json` — those two carry a `novel` field and are filtered per novel in the API.
- Per-novel analytics live in `<novel>/analytics_learning_log.json`.
- The dashboard picks the novel; every API route takes `novel=<id>` and defaults to
  `sinners_world`. Media for a non-legacy novel is served from `/novels/<id>/<kind>/<file>`.
- **Chapter files may be `.txt` or `.md`.** `chapter_files()` (producer) and
  `Novel.chapter_files()` / `Novel.chapter_file(n)` glob both, and the chapter number is the
  FIRST number in the filename. A novel downloaded as markdown used to look like an empty
  folder (0 chapters, "Chapter N not found", no reason given).
- **A novel's SETTINGS come from its own bible.** `_setting_key_for` picks the setting tail by
  keyword score over the active novel's `settings` entries (each may carry `keywords`); the
  legacy novel keeps its hardcoded `interior` / `white_sky_city` / `vethmore` cues. Without
  this, every scene of any other novel was painted with the legacy novel's cobblestones and
  flat white sky.
- **A novel OWNS its style.** `novel.json` → `art_direction`:
  - `style` — the shared base (one of `ANIMATION_STYLES`), e.g. `gothic`.
  - `still` / `motion` / `negative` — the novel's OWN look. They **REPLACE** the base style's fragment (they are not a suffix on it) and reach BOTH the still prompt and the H3 clip prompt, so every episode of the book shares one look.
  - `lora` / `lora_strength` / `trigger` — an optional per-novel **style LoRA**. It is attached UNDER the character LoRA (Qwen-Image chains LoRAs natively: style → character → sampler), so the character's face sits on top of the book's look.
    - Train one from the book's OWN stills: `python _jobs/stage_style_dataset.py <key> <novel>` (stages the novel's stills into `novels/<novel>/dataset/<key>_lora/` with the trigger word as the caption) then `python _jobs/train_style_lora.py <key> <novel>` (~2 h). Then set `"lora": "<key>"` and `"trigger": "<key>"`.
    - The caption is the TRIGGER WORD plus the scene's content, never the style text: the trigger is what carries the look, so the adapter can paint ANY subject in the book's style. A missing `.safetensors` is not an error — the slot simply resolves to nothing until the training finishes.
  - The whole art direction is in the still cache key, so changing a novel's look invalidates that novel's stills and nothing else.
  - A style fragment must not name a LOCATION (the per-scene `settings` do that) — "gaslit cobblestone street" in a room renders a room on the street.
- **The voice is the viewer's channel for the beat's facts.** The script LLM cannot be bullied into writing long lines (25 → 45 → 90 words over three revisions; it landed at ~21 words per line, ~40% of the chapter's 1021 words reaching the viewer). So `break_into_scenes()` runs `_expand_narrations()` after the parse validates: it re-sends the raw chapter plus the scene seeds with "rewrite each line in full, 35-55 words" and merges by `scene_number`. Measured live: 472 → ~830 words (~80% of the chapter) in the same run. Soft-fail: a parse failure keeps the original lines; expansion never aborts a render.
- **An episode's LENGTH is a function of its chapter, not a policy.** `scene_target_count()` asks for
  ~280 chars/scene (a complete adaptation — a scene per beat), and the only ceiling is PHYSICAL:
  `_script_answer_budget()` = what the LLM's answer can fit in `llm_ctx` beside the prompt and the
  chapter. The request's `max_tokens` is CLAMPED to that budget (asking for more than fits is a hard
  server error). `llm_script_max_tokens` is a CAP, not the ask.
  - The prompt keeps the essential-beat test ("if skipping this beat makes a LATER scene stop making
    sense, it stays") but must not tell the model to over-cut: the old "lean"/"compress hard" wording
    is what turned a chapter into a recap.
  - To lift the remaining ctx-bound ceiling, raise `llm_ctx` — the KV cache lives in system RAM
    (~1.1 GiB per 32k tokens), so check free RAM first.
  - A cached script is reused when it has at least half the target scenes, so a chapter that must grow
    needs its cached `scenes/chNNN.json` deleted.
- **A bible entry may carry `aliases`.** One person, many names (Klein / Gehrman Sparrow /
  Sherlock Moriarty / The Fool). `_bible_alias_map()` maps alias→canonical, `_char_tags()` follows it,
  and the character learner skips an alias instead of inventing a SECOND entry — and a second face —
  for someone the bible already has. The learner is deliberately asked about named characters the
  chapter mentions but never shows, so a novel without aliases grows duplicate faces.
- **The character learner rewrites the whole bible file.** It dumps `characters.json` from its
  in-memory copy, so an edit made while a render runs is silently overwritten. Read the bible
  before a render; do not edit it mid-render.
- **A fan work may RENDER but never PUBLISH.** `rights_manifest.commercial_use:
  "non-commercial"` passes the RENDER gate (`preflight_rights_gate(manifest)`) with a warning
  and is BLOCKED by the RELEASE gate (`for_publish=True`, used by `create_release_report`).
  The dashboard's "Approve rights" never flips a non-commercial entry to "yes" — approving a
  render must not claim publication rights the owner does not hold.
- **Identity = bible tags + the character's LoRA. The character card is a TRAINING SET, not a
  render input.** The still path is pure T2I and never loads the card — the sheet anchor was
  never used on the Qwen DiT, and it made five consecutive scenes repeat one picture. The
  bible's `danbooru_tags`/`prompt_tags` go into the text prompt of every shot; a trained Qwen
  LoRA is a stronger identity source layered on top, never a requirement. The render gate
  (`_character_identity_ready`) therefore demands TAGS, and the dossier page keeps the bio +
  tags + voice.
- **The card generator is restored (2026-11-11) as the cold-start path.** `ensure_cast_loras`
  makes a 6-view card for a character who has NO training images (never rendered, no
  reference art staged) and stages it into `dataset/<key>_lora/` — the exact recipe the 40
  shipped character LoRAs were trained on. A character who already has a card, a LoRA, or
  staged art pays nothing. The old 1.2 GB of cards stays out of the project (backup:
  `<SHEETS_BACKUP_DIR>\`); a card is regenerated per character
  on demand.
- **LoRA training images are STAGED, not generated.** `dataset/<key>_lora/` holds the
  PNGs plus a `.txt` caption each (the bible's `danbooru_tags` is written for any missing
  caption); `_jobs/stage_dataset_from_stills.py <key>` builds that folder from the
  character's own rendered stills. A character with fewer than 3 staged images and no card
  trains nothing and renders from its bible tags.
- **LoRAs are global and keyed by character key.** `<key>_qwen21_v1.safetensors` lives in the
  SHARED ComfyUI `loras/` dir, so two novels must not reuse a character key for different
  people — the second book would silently wear the first book's face. The trigger token is
  per character (`lora_trigger` in the bible), so it is per novel.
- **LoRA training blocks the render.** `ensure_cast_loras` trains inline (400 steps, ~2 h per
  character) BEFORE the first still, so a new novel's first chapter pays ~2 h for every new
  character. Opt that novel out with `"config": {"lora_training": {"enabled": false}}` in its
  `novel.json` — novel config overrides the global config, and dict values merge one level
  deep, so the venv and model paths survive. The episode then renders from the bible tags
  alone.

## Caching rules (read before touching a cache key)

Every artefact here is content-addressed by a sidecar fingerprint, and **the fingerprint must cover every input that changes the artefact**. `_STILL_CFG_KEYS` / `_STILL_SCENE_KEYS` in `episode_producer.py` enumerate the still-generation inputs and are asserted one-by-one by `test_producer_state.py`; adding a knob to `generate_image` without adding it there fails the suite. The same rule applies to `_echo_fingerprint` (lip-sync: shot size and mood build the prompt, so both are keyed). A key that omits an input is worse than no cache — it looks correct and serves stale output.

Beat-2 (`_sNNb_`) and companion (`_sNNc_`) stills use the same sidecar as A-stills; they are derivatives of one scene, not extra scenes, so `asset_library` excludes them from still counts.

**A still is only reusable if the pipeline can PROVE it matches.** The sidecar (`.cache.json`) records the fingerprint of the exact scene/config that produced the PNG; the PNG is the artefact and the sidecar is the note.
- The purge must JUDGE every still BEFORE deleting anything, and a kept still must KEEP its sidecar. Deleting the note first makes an invalid still look "unverified", and an adopted still gets re-stamped with the CURRENT fingerprint — which launders a frame made under the OLD prompt into "verified" for good, so the key can never invalidate it again.
- A still with NO note is unverified. For a novel that owns its look (`art_direction`), unverified means REGENERATE — the book cannot show a frame carries its style it cannot prove. For a novel without one (the legacy library), unverified means KEEP, so the library is never re-rendered over a missing note.
- Consequence: the first render after a novel gains an `art_direction` costs a full re-draw of its stills (and of the clips derived from them). That is the price of the guarantee, paid once.

All durable JSON goes through `json_util.dump_json_atomic` (fsync + bounded replace retry + **raises** on failure) and cross-process locks use `json_util.FileLock`. Do not hand-roll `open(path,"w") + json.dump` for state — Windows `os.replace` fails while any reader holds the file open, and a silent retry-loop exit loses the write with no error.
## Verification

```bash
python -m unittest discover -s scripts -p "test*.py"
```

481 tests, ~60s. Two rules that the suite now actually enforces:

- **Mutation-tested.** Every fix in this repo is verified by reverting it and
  confirming a test fails. If you change a load-bearing behaviour, add a test
  that fails without your change — a test that passes either way is worse than
  none, because it reads as coverage. Two examples of how that goes wrong, both
  fixed here: looping `for key in ep._STILL_CFG_KEYS` (deleting a key deletes its
  own check), and changing two inputs in one assertion so a second, unrelated
  feed satisfies it.
- **Order-independent.** Verified across shuffled module orders. Tests must not
  depend on import order, on a live service, or on wall-clock timing. The
  contention tests assert the *mechanism* (the lock is taken; the same path maps
  to the same lock) rather than trying to provoke a race on a timer.

`test_upgrade_fixes.py` contains ~23% source-greps — static guards for
invariants that are awkward to prove behaviourally ("no KSampler hardcodes
karras"). Its docstring explains when to prefer behaviour instead. Prefer
behaviour; reach for a grep only when the property is genuinely static.

Reading logs: `print` with indentation to the console, plus
`render_crash.log` and `%TEMP%\opencode\*_watchdog.log`. There is no structured
logging — grep the console scrollback when a render looks wrong.

## Known gaps (stated, not hidden)

- `pid_alive`'s `pid <= 0` guard is in the POSIX branch only; on Windows
  `OpenProcess` already rejects it, so no Windows test can observe the
  difference. The *outcome* is asserted on both platforms.
- `_produce_inner` is a ~1270-line orchestration function. Its **rules** are
  extracted and tested (`_pick_cached_still`, `_visual_fingerprint`,
  `_echo_fingerprint`, `_write_still_sidecar`), but the loop that calls them
  needs ComfyUI and is only covered indirectly. Keep new logic out of it.
- `fsync` ordering is asserted as a *call*, not as its crash-window effect —
  that is not observable from Python.
- The fingerprint format is versioned (`{"v": 2, ...}`), so every existing
  `.cache.json` invalidates once and all stills regenerate. That is the safe
  direction, but budget ~90 min for the first render after any key change.