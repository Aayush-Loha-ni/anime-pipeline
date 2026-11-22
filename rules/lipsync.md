# RULE: Lip-Sync — Talking Heads That Survive Anime Faces

## Current engine: EchoMimic V1 (ComfyUI nodes)

- Talking heads ONLY on proven close-up face clips. Faceless stills get
  `chNNN_echofailNN.flag` markers and skip in seconds — never retry blind.
- Echo generates at DELIVERY cadence (24fps, `echo_fps`) — a 25fps head
  normalized to 24fps drops a frame every second (judder reads as wave).
  Clips cover their slots; assembly loops only the computed shortfall.
- Proven dead ends (do not retry without new information):
  - EchoMimic on faceless stills (blank-oval faces have no landmarks).
  - EchoMimicV2 (pose-bleed duplication; 15GB weights + env DELETED).
  - Hallo (needs 50–70% frontal face; our stills don't qualify).
  - YuNet face gate (scores 0 on anime).
- V3_flash currently disabled in our ComfyUI_EchoMimic build (module
  naming fault) — re-check before Phase 3, else LatentSync/MuseTalk
  re-dub path (syncs existing clips, no pose input, 4–8GB).
- P2 dispatcher LANDED: `lipsync_engine` echo|latentsync|s2v (default
  echo). latentsync/s2v log PILOT + fall back to echo V1 until nodes /
  Wan2.2-S2V-14B weights are staged. S2V is the end-state (one
  audio-driven pass replaces TI2V + lip-sync for talking heads).

## Discipline

- No talking-head attempt without a close-up asset (face ≥25% of frame,
  eyes + nose visible). Faceless wides echofail IMMEDIATELY — zero retries,
  zero Comfy calls. Companion `sNNc_` stills exist precisely for this.
- Character-as-narrator: the voice EchoMimic animates must be the voice
  on the track for that scene (match by construction, Phase 5c).
- Grade every talking head; fail-markers are data, not shame.
- Close-up stills are a prerequisite asset — faceless wides can never
  become talking heads no matter the engine.
