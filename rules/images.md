# RULE: Stills — Qwen-Image-2.1 Reference Images

## Engine (verified)

- `qwen_image_2.1_Q4_K_M.gguf` via `UnetLoaderGGUF` (safetensors alt via
  `UNETLoader`) + `qwen3vl_8b_int8_convrot` text encoder on CPU
  (`CLIPLoader`, `device: cpu`) + `qwen_image_vae` (`VAELoader`,
  SaveImage id `7b`). Landscape 1344×768, portrait 768×1344
  (platform-native, never squeezed).
- **Sampler: `euler` + `simple`, 25 steps, CFG 1.0.** This is not a style
  choice, it is a correctness requirement, and the file used to get it
  wrong. Qwen-Image-2.1 is a **flow-matching** model, and Karras sigmas are
  meaningless under flow matching — sigmas are compressed and shifted so
  differently that the ramp does not resolve (diffusers: "the sigmas /
  timesteps are vastly compressed under FlowMatch which negates their
  usability"; karras/exponential on flow models "results in full blur").
  Comfy-Org's own Qwen-Image-2.1 templates ship `euler` + `simple` at
  25 steps / CFG 1, and Qwen's repo describes "Flow Matching with Euler
  discrete scheduling and dynamic shifting".
- **MEASURED, do not re-learn.** A real ch900 render ran `dpmpp_2m` +
  `karras` and produced near-black stills that the pipeline shipped. Same
  seed, same prompt, only the schedule varied:

  | sampler | scheduler | CFG | steps | mean luma |
  |---|---|---|---|---|
  | dpmpp_2m | **karras** | 6.0 | 40 | **3.87** (black) |
  | euler | **karras** | 6.0 | 40 | **4.74** (black) |
  | dpmpp_2m | simple | 6.0 | 40 | 31.61 |
  | dpmpp_2m | normal | 6.0 | 40 | 31.70 |
  | dpmpp_2m | beta | 6.0 | 40 | 55.93 |
  | dpmpp_2m | simple | 4.0 | 40 | 28.74 |
  | dpmpp_2m | simple | 6.0 | 25 | 21.44 |

  Karras broke it under BOTH samplers, and nothing else mattered — steps,
  CFG, LoRA on/off, text-encoder device and the `resolution` hint all left
  the result dark. **Karras is the variable, always.**
- CFG 1.0 is the official baseline. Pushing CFG to 5–7 on this model
  oversaturates and rings edges rather than improving prompt adherence; the
  official template does not use CFG at all. If you need a negative prompt
  to bite, raise CFG to ~2, not 6.
- A black still is caught by the `stills not black` gate (mean luma < 3),
  which regenerates on a fresh seed and then fails the render. It exists
  because this happened: 3–4 of 10 stills came back at luma 0.09/0.36/2.70
  and H3 animated scene 1 out of a pure black frame into a hallucinated
  face. Do not weaken that gate.
- Do NOT "fix" soft frames by raising `image_steps`. The auto-tuner did
  exactly that (30→35→40) while chasing a symptom of the broken scheduler.
  More steps on a bad schedule buys nothing; it costs 30% more GPU time.
- No ClipSkip node on the DiT path — ClipSkip is an SDXL-only concept, N/A.
- Seeds: `image_seed_base + chapter*1000 + scene` — deterministic takes.
  New take = bump `image_seed_base` (1000 → 2000 was take 2).
- SDXL LoRAs (e.g. Edvyn) are architecture-locked and NEVER attach to the
  Qwen DiT — the builder logs a skip. A Qwen-native LoRA DOES attach (it
  carries `diffusion_model.*` keys). Character identity comes from the
  bible tags plus the character's LoRA. `pose` per scene picks from
  POSE_PRESETS (stand/walk/run/sit/kneel/reach/look/carry).

## Anime DNA (baked into every still prompt)

`2d anime, masterpiece, best quality, absurdres, highres, detailed face,
detailed eyes, sharp pronounced facial structure, defined jawline and
cheekbones, crisp facial detail` + bible `prompt_tags`/`danbooru_tags` +
shot preset + setting tags + hard cel-shadow/ink-line block. Anti-CGI
block bans photoreal/3d/pixar/disney explicitly (drift creeps back).

## Consistency stack (in order — each layer compounds)
1. **Bible tags** (`characters/characters.json`): danbooru + prompt tags
   injected per cast member. Identity fallback composes from bible fields.
2. **Shot-aware framing** (BINDING, post-9-17 + torso ban): close-up =
   face-fills-frame portrait (head/shoulders, not torso); medium =
   chest-up with face large and centered; wide = full-body head-to-toe.
   Negatives ban torso-only / waist-up / body-without-face on close-up
   and medium. The old blanket full-body rule drew torso-only close-ups
   in 1344×768 (the model fills frame width and pushes the head out).
   The images gate requires every A-still FILE on disk before clips.
3. **Pure T2I, no anchor**: the still path never loads a reference image.
   The character sheets (and the I2I anchor they fed) were removed — an
   anchored scene repeated one picture (pHash 0-2 across five scenes) and
   dragged the sheet's flat backdrop into the scene. Identity is the bible
   tags plus the character's LoRA, nothing else.
4. **Character LoRA:** convention `models/loras/<key>*.safetensors`,
   strength `lora_strength` (0.45), trigger word auto-appended per
   character (bible `lora_trigger`; the global `qwen_lora_trigger` is the
   fallback for `qwen_lora_trigger_owner` only). An SDXL LoRA is skipped
   with a log line on the Qwen DiT. Mains only (Edvyn ✓); incidentals live
   on their bible tags.
4b. **A NOVEL OWNS ITS LOOK (2026-11-13).** `novel.json` → `art_direction`
   REPLACES the shared style's still/motion/negative fragment (it is not a
   suffix on it) and reaches BOTH the still prompt and the clip prompt, so
   every episode of a book shares one look. An optional per-novel **style
   LoRA** (`art_direction.lora`/`lora_strength`/`trigger`) is attached UNDER
   the character LoRA — Qwen-Image chains LoRAs natively (style → character
   → sampler), so the face sits on top of the book's look. Train one from the
   book's own stills (`_jobs/stage_style_dataset.py` then
   `_jobs/train_style_lora.py`; ~21 min for 400 steps). The whole art
   direction is in the still cache key, so changing a book's look invalidates
   that book's stills and nothing else. A style fragment must not name a
   LOCATION — the per-scene `settings` own that.
5. **Head-inside-frame gate** (fresh A-stills, cast only): prompt locks
   head + headroom, negatives ban cropped/decapitated heads; VLM
   `face_visibility_check` (regex on explicit head/crop phrases, hair
   exclusion) then ONE reframed retry (`face_check_retry_offset`,
   default 50000). Episode budget `face_check_max_reframes` (default 4),
   then keep + flag. Best-available always wins — a flagged still beats a
   gap, never aborts. Cached stills get warn-only flags (no GPU spend).
   HONEST LIMIT: the LLM server is stopped before Phase 2 and no vision
   adapter is staged, so in normal renders the VLM is unreachable and
   the gate passes — the prompt lock is the real net.
   `face_check_enabled: false` to disable.

## Limits (do not re-learn)

- Fringe hides faces BY DESIGN (messy parted-left hair). The bible tags
  lock outfit/hair/eye-color; frontal face detail has to come from the
  scene prompt.
- Civitai LoRAs need login (blocked). Impact-Pack detectors score 0 on
  anime — no detector-based refiner will fire.
- LoRA training images are STAGED, not generated: captioned PNGs in
  `dataset/<key>_lora/` (see characters.md). The character sheets that used
  to supply them are gone.
- Grade once: `unsharp` lives in the assembly norm stage ONLY. Build-stage
  filters (cinematic/parallax) run fit-only — double sharpening shimmers
  edges into fake motion.
