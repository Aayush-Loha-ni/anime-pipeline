# RULE: Script — Novel Chapter → Viral Scenes

Source of truth for `break_into_scenes` (llama-server `gemma-4-26B-A4B`
uncensored/heretic, IQ4_XS, temperature 0.2, seed `llm_seed + chapter`,
OAI chat completions). Replaced `Qwen3-14B-abliterated` 2026-09-28.

- **No `/no_think` suffix.** That was a Qwen3 control token; Gemma-4 uses
  `<|think|>` and its own template defaults `enable_thinking` to false.
  The server still passes `--reasoning off --reasoning-budget 0` as
  belt-and-braces. Dropping the token measured 3/3 valid JSON arrays.
- Gemma-4 is 25.2B total / 4B active MoE, 262k trained ctx, run at
  `llm_ctx` = 32768. Its long-range KV lives on only 5 of 30 layers and
  llama.cpp puts the WHOLE KV cache in system RAM, not VRAM, so context
  costs ~1.1 GiB of RAM rather than fighting the 12.96 GB of weights for
  the 16 GB card. The weights are what fill the card; ComfyUI must still
  be free during script gen (`free_comfyui_vram()` does that). Do not
  co-reside ComfyUI with the LLM.
- **`--parallel 1` is load-bearing.** llama-server splits `--ctx-size`
  across slots, so the default 4 slots gave each only `ctx/4`. A
  3640-token prompt then had ~4552 left and the JSON was cut mid-array
  with `finish_reason: "length"` — indistinguishable from a max_tokens
  cap in the logs, but not one. One request at a time, so one slot.
- Budget is `llm_script_max_tokens` 8192 / `llm_script_timeout` 900.
  Both long-JSON generators need it: `break_into_scenes` AND
  `director_package._llm_generate` (the latter takes them as parameters,
  it has no CONFIG import). The small callers (motion-prompt expand,
  narrative check, supervisor triage) pass their own budgets.
- Verified after the change, real chapters, `_parse_probe` as the judge:
  `sinners_world_chapter1` 25/25 scenes, 4377 completion tokens, 522 s;
  `Chapter_63` 30/30 scenes, 3908 tokens, 485 s. Both were
  `finish_reason: "length"` and unparseable before.
- llama.cpp logs `detected an outdated gemma4 chat template` for this
  community quant and applies compatibility workarounds. Harmless so far,
  but it is why we do not also use it as the VLM.
- A/B at 9 scenes / 6.1k prompt, judged by `_parse_probe`:
  Qwen3-14B 2/3 parse, median 337 s. Gemma-4-26B-A4B 3/3, median 146 s.

## LLM Directives (injected into every script prompt)

- Scene 1 opens with a SHOCKING hook (curiosity, danger, mystery). Never slow.
- Each scene: 1 narration OR 1 dialogue. A narration should run 40-70 words
  when the beat supplies it — the frame shows ONE image, so the narration is
  the viewer's channel for the whole beat. Measured gap this closed
  (2026-11-13): at ~14 words/line a 1021-word chapter reached the viewer at
  ~30%; 70% of its information was dropped by the word caps. Carry the facts,
  in full — a one-line summary of a beat the chapter spends sentences on is
  information thrown away. SIMPLE everyday language.
- Bedtime-storyteller energy, viral-optimized: hook in first 5 seconds,
  short plain sentences, one BEAT per scene, talk TO the viewer sometimes,
  concrete picturable things, re-hook every scene.
- ≥20% establishing WORLD shots (white-sky city, canals, fog, exteriors),
  SPREAD through first/middle/final quarter — never clustered.
- Vary shots: close-up, wide establishing, object detail, reaction.
- Power words: secret, forbidden, dead, betrayed, hidden, final, escaped, caught.
- Return ONLY a valid JSON array, EXACTLY the requested count, top-level `[...]`.
- Schema per scene: scene_number, type, characters, narration, dialogue,
  image_prompt, animation_hint, shot_size, angle, camera_move, mood,
  stages[], text_overlay (3–6 VIRAL CAPS WORDS), hook_type.

## Beat structure (retention engineering, 2026-09-16 audit)

- Scene 1 MUST name one visible verb readable at phone scale (no mood-only
  openers — "you have been here before" over a static close-up is banned).
- Beat map: minor turn ~every 10s, major turn ~45% and in the last 15%;
  final scene ends on a face or an unanswered question, never a flat wide.
- One action + ≤1 camera move per shot; dialogue ≤8 words/scene.
- Purpose per shot (HOOK early, TURN mid, LAND late) lives in hook_type +
  text_overlay — read them when boarding, don't re-derive.

## Uncensored Adaptation Policy (mandatory)

- The novel's darkness is LOAD-BEARING. Fire, corpses, marked flesh,
  dread, moral rot — all of it stays. Never soften, summarize away, or
  moralize grim content into lessons.
- Narration may thrill and chill; it must never preach, warn the viewer,
  or append comforting conclusions the chapter didn't earn.
- If a scene disturbs, good — that means it was adapted, not sanitized.
- banter, warmth, and hope stay ONLY where the chapter put them.

## Verified Guards (all in code — do not weaken)

0. **THE EPISODE'S LENGTH FOLLOWS THE CHAPTER (2026-11-13).** `scene_target_count`
   asks for ~280 chars/scene — a complete adaptation, a scene per beat — and
   the ONLY ceiling is physical: `_script_answer_budget()` = what the LLM's
   answer fits in `llm_ctx` beside the prompt and the chapter, with the
   request's `max_tokens` CLAMPED to it (asking for more than fits is a hard
   server error, not a short answer). `llm_script_max_tokens` is a CAP, not
   the ask. A hard scene cap (the old 40) and a "lean / compress hard"
   prompt were what turned a chapter into a RECAP — a 6000-char LOTM chapter
   became 10 scenes. The essential-beat test stays: if skipping a beat makes
   a LATER scene stop making sense, it stays. To lift the ctx-bound ceiling,
   raise `llm_ctx` (the KV cache is system RAM, ~1.1 GiB per 32k tokens).
   A cached script is reused when it has at least half the target scenes, so
   a chapter that must GROW needs its `scenes/chNNN.json` deleted.
1. **Count guard:** fewer than max(3, target/2) scenes → retry with new
   seed (2026-09-11: Qwen3 collapses from-zero gens to a single `{...}`).
2. **Degenerate guard:** <50% unique lines → reject.
3. **Tail guard:** last 5 identical → reject (ch001 s21–25 passed the
   global ratio and shipped a repeated ending once).
4. **Timeout 300s, 3 attempts**, temperature +0.1 and seed +1000 per retry.
5. **From-zero generation is UNRELIABLE** on current Qwen3 — curated
   `scenes/*.json` is the source of truth. "New video" = new seed take.
6. **Shot-diversity post-pass** (`_enforce_shot_diversity`): the LLM
   ignores "vary shots" (ch001: 25× static/eye-level/somber), so repeats
   rotate through pools deterministically — no re-prompt. BINDING: runs
   on cached `scenes/*.json` loads too, never fresh-gens only (the static
   ch001 file shipped for weeks because the cache path skipped it). The
   script gate measures top-move share (≤40%) and aborts otherwise.
   Shot-size strings are normalized (`_shot_norm`: "close up"→"close-up")
   because every downstream gate compares them.
7. **Framing follows shot size** (enforced in stills, stated here because
   the script chooses it): close-up = face-fills-frame; medium = chest-up
   face large (never torso-only); wide = full-body head-to-toe. A blanket
   full-body rule in a landscape frame draws torso-only close-ups.
8. **P2 shot-plan pass** (`shot_plan_pass`, no LLM): purpose HOOK/TURN/
   LAND per shot, ≤1 camera move, pose seeded from animation_hint.
   `canon_chNNN.json` snapshot per chapter; `llm_narrative_check`
   optional text spot-check every 8 scenes (`vlm_narrative: true`,
   best-effort, never blocks).
