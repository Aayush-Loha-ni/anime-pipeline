# RULE: Voice — The User's Own Clone, One Engine

## Direction (mandated — Qwen3-TTS since 2026-09-28)

- **Single TTS engine: Qwen3-TTS-12Hz-1.7B-Base on :5003.**
  XTTS v2, CosyVoice 3, Chatterbox, F5 and Kokoro presets,
  per-character clone refs, DeepFilterNet and VoiceFixer are DELETED —
  not disabled, gone.
  One narrator voice (the user's) for every scene. No consent risk
  (own voice), deterministic.
- **Why:** XTTS v2 is Coqui Public Model License 1.0.0 — non-commercial
  use of the model AND its outputs only. We ship to YouTube. Qwen3-TTS
  weights are Apache-2.0. This is a licensing fix, not a taste call.
- Zero-shot, no training. Handles imperfect references gracefully.
- **Canonical ref: `cosy-server/ref34.wav`** (24 kHz, cut from <your-recording>,
  1772.7-1785.8 s). Chosen by RERANKING, not by hand: 8 candidate clips from the
  take were scored on 3 sentences and the winner was then validated on 9
  sentences it had never seen — **+0.0239 speaker similarity, 8/9 sentences won,
  t=+2.99**. That is the largest single win available on this machine.
  `ref32.wav` and `ref33.wav` are the previous two, kept for comparison.
- **Why reranking beats training here, measured.** Per-candidate speaker
  similarity spread across reference clips is **0.049** — about **28x** the 0.0017
  spread between trained and untrained model configurations. Reference choice
  dominates; model choice is noise at this scale. See SFT_FINDINGS.md.
- **VoiceFixer restoration was tried and rejected** (`restore_ref.py`, kept so it
  is not retried blindly). It restores a 16 kHz phone recording to full band:
  bandwidth +1.42 pp, but UTMOS **-0.255** and similarity to the ORIGINAL
  **-0.053**. It is a generative vocoder — it invents plausible high frequencies
  rather than recovering yours, so the spectrum widens while the voice drifts.
  `compare_refs.py` is the gate that caught it.
- **Recording quality, not volume, is the ceiling.** <your-recording> is 34 min /
  26.7 min voiced at 16 kHz 37 kbps AAC with Whisper small.en transcripts.
  Training on 178 s and on 1188 s both failed to beat zero-shot ICL
  (+0.0031 and -0.0007). A studio mic would help; more minutes on a phone mic
  will not. NOTE the previous reference below, kept for history.
  `<YOUR_RECORDINGS_DIR>\Recording (32).m4a` — 4:17 of the user
  READING THE SERIES SCRIPT), with a machine transcript in `cosy-server/ref.json`.
  Rebuild it with `python cosy-server/build_sft.py "<recording>"`, which rewrites
  the wav, `ref.json`, and every training row's `ref_audio` in one pass so they
  cannot drift.
- **The reference must be SPEECH, not singing** (2026-10-03). The previous ref
  was an 8.63 s cut of "Little Things" — the entire 41.77 s source recording was
  that pop song. Cloning a singer transfers sustained vowels, vibrato and
  musical dynamics instead of conversational prosody, and with ICL conditioning
  the model *continued the lyric*: four words of it were prefixed to every
  generated line. `x_vector_only_mode=True` had been set to hide that, which is
  why the mode was pinned there in the first place — the mode was a symptom
  workaround for a bad reference, not a preference.
- **Mode: ICL (`x_vector_only: false`), measured.** Paired per-sentence scoring
  on 12 held-out sentences, WavLM-base-plus-sv cosine vs the reference:
  ICL beat the old x-vector-only config by **+0.0152 SIM (t=+4.07)**; UTMOS
  naturalness drops 0.27 (t=−4.88). It is a trade — more like you, slightly less
  "natural" by MOS prediction — and likeness is what was asked for.
- **A 4-minute fine-tune did NOT beat zero-shot ICL** (+0.0137 vs +0.0152, and
  upstream's `custom_voice` SFT was indistinguishable from the baseline at
  +0.0031). So no checkpoint is referenced in `ref.json`. The trainer
  (`train_voice.bat`) is kept and working for when there is 15-30 min of audio.
  Upstream SFT also *relabels the checkpoint `custom_voice` and drops the
  speaker encoder*, which disables ICL entirely — see QwenLM/Qwen3-TTS#72.
- **Qwen3-TTS needs a reference transcript** (`ref_text`, ICL mode).
  Changing the ref means changing both `ref.json` keys together —
  an empty `text` aborts server startup by design, not silently.
- **The server loads its prompt ONCE at startup.** `/health` now returns
  `loaded_ref` (what the in-memory prompt was built from), `config_ref` (what
  `ref.json` says now) and `stale`. `episode_producer.check_tts_server` refuses
  a stale server and restarts it. Without this, editing `ref.json` retired the
  cache and then regenerated every scene from the OLD voice while writing a new
  fingerprint over it — correct-looking files, wrong voice.
- Verify the shipped voice any time with `python cosy-server/verify_voice.py`:
  it starts the real server, checks the health gate, synthesises a line,
  transcribes it, and probes for reference-text leakage.
- `cosy-server/server.py`: Qwen3-TTS model + preload-at-startup lifespan,
  SYNC endpoint, single-level infer lock, raw-wav `/cosy`, MAX_CHARS 600,
  fail-loud 400/502. Same API shape as before, pipeline untouched.
  Install: `pip install --no-deps qwen-tts==0.1.1` + `accelerate` +
  `transformers==4.57.3` + `sox`. No torch pin — chatterbox-tts pins
  `torch==2.6.0`, which has no sm_120 kernels and cannot run on the 5080.
- Output is 24000 Hz, same as XTTS v2, so the trim chain and the 44.1 kHz
  assembly resample are unchanged.
- `speed` is accepted and IGNORED. Qwen3-TTS has no speed control.
  `voice_speed` is pinned to 1.0; wire an atempo pass if it ever moves.
- Throughput: ~5x real-time on the 5080 (≈29 s for 5.8 s of audio) with
  `attn_implementation="sdpa"`. flash-attn has no Windows wheel.
  Cost is a few extra minutes per episode; quality/licence is the point.

## Mastering

- Target: integrated −14 LUFS, true peak ≤ −2.0 dBTP, LRA 11 (YouTube).
- STEREO dual-mono master (mono voice in both channels); post-music
  loudnorm + TP assert on shipped bytes.
- Trim chain (engine-agnostic, keep): silenceremove 0.2s/−50dB both
  ends (reverse), `apad=0.25`, `volume=-3dB`. QA detects at the same
  −50dB so soft consonants aren't flagged as silence.

## Sidecar cache (sacred, gated)

- Every `.wav` has a `.txt` sidecar holding the EXACT sent text.
  Match = cache hit (no server call). Mismatch = regen.
- BINDING: the voice gate requires every wav to carry a NON-EMPTY
  sidecar. Untrusted cache aborts the render — never ship mystery audio.
- `compute_scene_durations` kills dead air from real wav lengths;
  `voice_speed` 1.0; `voice_speeds: {<char_key>: <speed>}` overrides per
  voice. NOT honoured by Qwen3-TTS — see Direction.
- Voice-swap (new reference, same text) still needs manual wav delete —
  documented gap, rare path.
