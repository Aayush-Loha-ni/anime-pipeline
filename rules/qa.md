# RULE: QA — Gates That Block Shipment

## Hard gates (fail = no ship)

1. **Voice-fail assert before assembly.** A dead TTS once shipped 125s
   of silence. `assert all(auds)` — every scene voiced or abort. The
   voice gate additionally requires every wav to carry a non-empty
   sidecar (untrusted cache aborts).
2. **A/V cross-correlation sync gate** (`qa_av_sync`) — exact on all
   scenes. Trust the correlator over everything below. Violations DELETE
   the output and abort (no stale "done" video, no silent truncation).
3. **Clip grade** (`grade_clip`) — face drift / corr < 0.45 / blur
   collapse → new seed → fallback. Never assemble a failed clip.
4. **Shipped-audio asserts** (post-9-17): stereo channels + true peak
   ≤ −2.0 dBTP measured on the muxed bytes, fail-loud. Norm failure
   aborts (no raw fallback into stream-copy concat).
5. **VLM issues cap the grade** — non-empty `issues[]` can never stay S
   (S + "head out of frame" shipped once). The cap lives in the scorer,
   not in reviewer judgment.

## Soft signals (inform, don't block)

- **auto-editor speech-survival gate**: flags ~0.3–0.5s on soft line
  endings (ch1 s9 "her"). Known false-positive pattern — correlator wins.
- **VLM spot-checks** (qwen3-vl): S/A grades tracked in quality_log.json.
  Naive brightness/contrast/sharpness scores are TRENDS, not verdicts.
- **Tail-repeat script guard**: last-5-identical narration = collapsed
  ending. Regenerate script, never hand-patch around it (s21–25 lesson:
  patch once, then fix the guard that allowed it).
- **Text alignment trend** (`text_prompt_alignment`, 10% of image score):
  spoken-line vs image_prompt word overlap. Cheap CLIP stand-in, trend
  not gate. QA still excludes `c_` companions (A-still only).

## Ledger discipline

- Every render appends: quality_log, cost_log (cap 200 rows), EDL.
  Logs are history — never delete to "clean up". (perf_log was
  write-only and has been removed.)
- Every EDL carries an `"assembly"` receipt block: per-scene clip/slot/
  loops/seams + mix/mux return codes + music presence + TP check +
  `qa_av_sync`/`qa_auto_editor` outputs. Unreceipted work is
  unverifiable work.
- `issues.json` via supervisor: warn/critical with cause. Open CRITICALs
  BLOCK `produce_episode` at preflight (assert, 2026-09-16) — the human
  clears them by triaging the ledger. No silent renders past a red flag.

## Feedback loop (every render gets better)

- `evolve_quality()` runs after rating: aggregates sharpness, brightness,
  alignment, reframe/face-flag rates, clip rate over the last 5 renders.
- AUTO allowlist only: `image_steps` 25..50, downward only, with a
  2-render cooldown (cooldowns count reverts and proposals too — no
  change→hurt→revert→change flapping). Last change is judged next
  render by its target metric — measurably worse (>5pts) REVERTS
  automatically.
- `bright_boost` was REMOVED from the allowlist (2026-09-29). It was a
  prompt prefix (`"bright vivid lighting, "`) that Qwen-Image ignored:
  `quality_log` 37→38→39 read brightness 17.9→15.2→17.9, an A-B-A flap
  where the controller measured its own noise and read it as harm,
  costing a full episode render per cycle. Do not re-add a knob here
  until a render proves it moves its target metric.
- Face budget (`face_check_max_reframes` 2..6) is PROPOSED in
  `tuning_ledger.json`, never auto-applied (GPU-spend knobs need a human).
- Everything else lands in `tuning_proposals.json` as pending
  (alignment slumps, face-flag storms, clip gaps, score regressions) —
  the human flips status, the loop never touches them.
