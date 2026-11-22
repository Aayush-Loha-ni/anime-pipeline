# Studio Rules — Codex Overview

Every render loads this codex (`rules/*.md`) at startup, logs its version,
and adheres to it. Rules are written from VERIFIED pipeline behavior —
each rule earned its place through a real render, a real crash, or a real
eye-test. No speculation, no cargo-cult.

## Files

| File | Owns |
|------|------|
| `script.md` | Novel chapter → viral scenes (LLM directives + guards) |
| `images.md` | Stills: Qwen-Image-2.1, DNA, pure T2I, seeds |
| `motion.md` | Motion: H3 Ref2VA int8 only (fast/final tiers), gate, grading |
| `voice.md` | TTS: Qwen3-TTS direction, reference transcript, sidecar cache |
| `lipsync.md` | Talking heads: EchoMimic, grader, fail-markers |
| `assembly.md` | Assembly v2, music bed, loudness, captions, EDL |
| `qa.md` | Quality gates: thresholds, what blocks shipment |
| `ops.md` | Machine safety: VRAM order, heat, services, power |
| `characters.md` | Sheets, dossiers, LoRA, trigger discipline |

## Doctrine (applies everywhere)

1. **Determinism is a feature.** Same script + same seeds = same video.
   A new "take" means a new `image_seed_base`, never random edits.
2. **Fail loud, never silent.** Missing clip = abort. Missing voice =
   abort. Silence ships over no dead body — the voice-fail assert exists
   because 125 seconds of silence shipped once.
3. **Adapt faithfully — never sanitize.** The novel is dark (fire, death,
   dread, marked flesh). The script model is abliterated precisely so
   horror survives adaptation. No moralizing rewrites, no softened
   endings, no safety-blanket narration. Creative and intelligent means
   honoring the source, dread included.
4. **Cheapest rung first.** Parallax before H3, warn
   before retry, retry before fallback, fallback before abort.
5. **One engine owns the GPU at a time.** LLM → TTS → ComfyUI, with
   explicit unload between phases. 16GB VRAM is a hard ceiling. TTS runs
   BEFORE images and clips (2026-10-06) because every shot length is
   measured off the narration it carries, so audio has to exist first.
6. **Caches are sacred — within a run.** `produce_episode` purges chapter
   artifacts at Phase 0 BY DESIGN (stale + fresh never mix); `rebuild`
   scripts reuse good caches. Sidecars (`.txt` next to `.wav`) decide
   voice freshness, not timestamps. Never delete a good cache to "try
   again" outside a purposeful purge.
7. **The codex binds through gates, not goodwill.** Every phase opens
   with `require_phase` checks that MEASURE the rule (files on disk,
   sidecars fresh, diversity ratio, stereo ship, receipts present) and
   abort fail-loud before GPU spend. A rule without a gate is a wish.
8. **Every render leaves receipts.** Assembly persists per-scene
   clip/slot/loops + mix/mux return codes + music presence + TP check +
   caption standard (cues, worst CPS, longest line, style actually written)
   + QA outputs into the EDL (`"assembly"` block). Unreceipted work is
   unverifiable work — the 9-16 music/QA state proved it.
