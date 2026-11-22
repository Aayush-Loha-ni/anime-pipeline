# RULE: Motion — MiniMax H3 Ref2VA int8 Only

## Engine: Comfy-Org's `video_minimax_h3_r2v`, verified live

This is the recipe, and it is not optional. H3 ignored its reference still
completely before it: given a correct still of a man before a burning mill it
produced a mangled doll face in **frame 1**, and the result was invariant to
LoRA subset, LoRA strength, step count, sampler chain, encoder device,
reference size, slot naming and canvas size. Comfy-Org's template, downloaded
and run unmodified with its own weights, produced the correct scene. The
builder now matches it.

- **DiT** `minimax_h3_ref2va_pruned_int8_convrot` (int8 is what makes H3 fit
  16GB), **text encoder** `qwen3vl_32b_minimax_h3_nvfp4_awq` with
  `type: minimax` and **`device: default` (GPU, not CPU)**, **video VAE**
  `minimax_h3_video_vae_int8_convrot`, **audio VAE**
  `minimax_h3_audio_vae_fp32` (Ref2VA is video AND audio — wire it and
  `VAEDecodeAudio`, even though the episode voice is XTTS).
- **LoRA: exactly one**, `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16` at
  1.0, loaded through **`LoraLoaderModelOnly`**. It is built for
  reference-to-video. The old chain — an 8-step accel LoRA + a photoreal-skin
  LoRA + a camera LoRA, stacked three deep through `LoraLoader` — patched the
  32B Qwen3-VL encoder as well as the DiT and is what shipped broken clips.
  If the turbo LoRA is missing, run with **no** LoRA; never fall back to the
  old chain.
- **Sampler: the official chain**, not the plain `KSampler`:
  `RandomNoise` → `BasicGuider` → `KSamplerSelect(res_multistep)` →
  `BasicScheduler(simple, 20 steps, denoise 1.0)` → `SamplerCustomAdvanced`.
- **The prompt must DESCRIBE THE SCENE.** This is the decisive part. The
  ref2v prompt is a full video description, not a motion note. A
  motion-words-only prompt ("subject from `<Picture 1>`, slow push in, hair
  sways…") left the model free to invent the subject — measured, it turned
  the man-and-mill still into an anime woman in a white dress indoors.
  Prefixing the same motion text with the scene's own `image_prompt` produced
  the man and the mill. `_h3_prompt` owns this; see the test that asserts the
  description leads.
- LTX (all variants) and Wan (all variants) are REMOVED (2026-09-25); old
  `chNNN_ltxNN.mp4` / `chNNN_wanNN.mp4` caches are ignored, never deleted.
- `video_engine` is `h3`-only (API + schema reject anything else).
- Frame counts must satisfy `(F-5) % 17 == 0` (124/243). Off-grid values are
  snapped rather than rejected mid-render.

## Gate (2026-11-13: EVERY scene gets H3)

- **H3: every scene.** The old gate sent the scenes without a motion keyword
  (usually the quiet ones) to a camera-move-over-the-still, which reads as
  "a frame with effects on it". The user's rule is absolute: a clip must show
  real movement — characters or background — or it does not ship. If H3 cannot
  produce a clip in 3 attempts, the render raises instead of falling back to a
  still-pan. The `type: action` exclusion and `is_motion_scene` are gone with
  the gate (there is nothing to gate).
- P2: per-scene `h3_steps` override for quality A/B;
  `motion_prompt_extend: true` tries text-LLM 4-part expansion, falls
  back verbatim when the LLM server is down.

## Motion prompt doctrine

- `_h3_prompt` = scene description FIRST, then motion. See the engine
  section: omitting the description is what broke reference-following.
- Locked faces get SUBTLE motion (breathing, blinks, head shifts, same
  face, no morph) — never frozen, never free. Open wides get full motion.
- Physics block on all builders: weight, momentum, follow-through,
  cloth/hair response, planted feet, no sliding.
- Negatives always include: warping, morphing, melting, extra limbs,
  sliding feet, jitter, flicker, popping.
- Cache: `images/chNNN_h3NN.mp4`, engine-matching file preferred on
  resume.

## Cadence law (BINDING, post-9-17 — the wavy-video postmortem)

- **One delivery cadence: 24fps.** Echo generates at 24fps (`echo_fps`).
  `fps=` frame-drops (25→24) and frame-dups (12→24) both read as wave —
  never mix cadences and fix it in norm.
- **Computed loops, never fixed.** Norm uses `loops=ceil(slot/clip)-1`
  (0 = no loop flag when the clip covers the slot). Fixed `-stream_loop 3`
  restarted 0.76s talking heads 3× per scene (lip seams) and decoded long
  clips 4× for nothing. Per-scene clip/slot/loops land in the receipt.
- **Cache provenance is checked.** A cached motion file under 20fps is a
  stale raw (pre-H3 era, e.g. 12fps LTX): delete once, regenerate
  through the H3 path. Never serve judder to protect GPU time.

## Speed: what is actually the bottleneck (measured, do not re-learn)

Wall clock for a 4-scene episode, RTX 5080 Laptop 16 GB, 10 stills +
4 clips: **~3500s**. Stills 1773s (49%), clips 1387s (38%), everything else
~445s (13%). Clips run ~335s each.

Two things that LOOK like free speed, and are not:

- **Sampler steps are free.** The official template pairs the 4-step turbo
  LoRA with 4 steps and reserves 20 for the no-LoRA path, so cutting
  20 → 4 reads as a 5x win. Measured on the real model: 20 steps 148s,
  4 steps 157s — **no difference**, because sampling is not what costs
  anything. The cost is the 32B Qwen3-VL encoder and the VAE. Both step
  counts produced an equally good clip, so 20 stays (marginally closer to
  the still). Do not "optimise" this without re-measuring.
- **The still text encoder on the GPU crashes ComfyUI.** Same seed, the
  picture is identical (correlation 0.99889, MAE 0.415) and it is 2.4x
  faster: CPU 138s vs GPU 58s per still. But in a real render the extra
  resident VRAM tips ComfyUI's memory planner —
  `RuntimeError: aimdo memory compile error` → `Fatal Python error:
  Aborted` — the process dies, and every remaining clip falls back to a
  panning still. `still_encoder_device` therefore ships as `cpu`. Revisit
  only on a card with more than 16 GB, or once the still UNET is smaller.
  This matches the general rule: on a single GPU with tight VRAM, leaving a
  large text encoder on the CPU is correct and forcing it onto the card
  trades a slow render for a dead one.

**The vision check is deferred, not dropped.** `run_clip_still_checks` needs
two 1024-token images through a 32B-class VLM; run during the clip phase it
competed with ComfyUI for VRAM and timed out on 2 of 4 scenes. It is queued
in the clip phase and executed in Phase 8, after `free_comfyui_vram()` and
`ensure_vlm()`. A real render then reports
`[VLM] all N clip(s) still match their reference`.

## A clip is graded against its STILL, never against itself (BINDING)

`grade_clip` compares a clip's first and last frames. That proves the model
stayed **consistent with itself**, which is not the same as staying
**consistent with the still**. A ch900 render turned a wide night mill into
a mangled doll-face close-up — blown-out eyes, melted features — and every
internal check passed: the hallucination was internally consistent, its
histogram matched itself, and there was no face to count because the source
was a wide shot with no face in it.

So the internal checks have three blind spots, and all three fire on wide
and establishing shots:

1. **No face to count.** A landscape or wide shot has no face, so
   `faces 0->0` is silent no matter what the clip becomes.
2. **Consistency is not fidelity.** A wrong subject invented in frame 1 and
   held for 5s scores a perfect hist corr against itself.
3. **A histogram is colour, not content.** A pinkish face and a firelit
   night scene can share an HSV histogram closely enough to pass 0.45.

Therefore: **every clip is also measured against its source still**
(`still_drift`, recorded in the receipt as `clip_still_drift`) and offered
to the vision model (`_vlm_clip_matches_still`, recorded as
`clip_still_vlm_flags`). A VLM mismatch warns loudly.

`ponytail:` the drift number is NOT a gate. Measured on three real clips the
clean ones scored 0.084 and the broken one 0.366, so spatial correlation
does not separate them and any threshold would be fitted to noise. The VLM
is the check that can actually judge, and it currently warns rather than
blocks because its false-positive rate is unmeasured. Promote drift to a
hard gate only once enough real clips exist to fit a threshold on.

**Do not write a "wide shots are bad" rule.** The data refutes it: in the
same render the wide scene 1 scored 0.366 and the medium scene 4 scored
0.084, yet scene 4 was the clean one. Shot size does not predict clip
fidelity. If wide shots look worse, find out why with the vision model —
do not encode the guess as a rule.

## Still generation feeds this — get it right upstream

Clip quality is capped by still quality. The single biggest historical cause
of bad clips here was a broken *still* sampler, not a motion problem:
`dpmpp_2m` + `karras` on the flow-matching Qwen-Image-2.1 produced
near-black stills (luma 3.87 / 4.74) which H3 then animated out of nothing.
Black stills went 3–4 per 10 → 0 per 10 on `euler` + `simple` + cfg 1.
See `rules/images.md` for the schedule and the measurement table.
