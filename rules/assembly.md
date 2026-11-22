# RULE: Assembly — v2 Audio-Master-First (v1 FROZEN, touch only to delete)

## Pipeline (verified, do not regress to v1)

- Clean concat, no bridges (`bridge_len = 0.0` — bridges duplicated
  tail/head audio into stutter). No `adelay`, 0.05s fades, voice centered
  via `scene["_lead"]` (front-loaded voice read as "cut").
- ONE AAC encode, stream-copy video. Clips keyed by scene_num, never
  positional (one gap misaligned everything after it once).
- STEREO end to end (post-9-17): stereo master, stereo mix, stereo ship.
  The mono master collapsed beds — asserted on shipped bytes, fail-loud.
- Music mix: ducked sidechain, `normalize=0` (plain amix dips voice ~6dB),
  then POST-MUSIC loudnorm + TP assert on the shipped file. Loudnorming
  the pre-mix voice shipped −1.7dBFS once — never again.
- Norm uses computed loops (`ceil(slot/clip)-1`, 0 = no loop) at 24fps
  single cadence; norm failure aborts (no raw fallback into `-c copy`).
- Hook audio AND video agree: card build failure disables hook for both
  tracks, never a 1.5s drift in one direction.
- EDL/caption timelines derive from ACTUAL hook presence, never
  hardcoded durations.
- Hook cards OFF (`hook_enabled: false`); font= names segfault this
  ffmpeg build (fontconfig) — absolute `fontfile=` only, everywhere.

## Music bed (licensed only)

- `music/` holds Fesliyan beds (Dark Fog/Shadows/It Is Coming) + 
- `ATTRIBUTION.txt` credit lines — paste into EVERY video description
  that ships the bed, or risk strikes. `music_quarantine/` rips NEVER
  auto-ship (copyright-risky).
- Mix: mood-rotated bed via `pick_music_track` (deterministic hash of
  dominant scene mood; same script = same bed), volume 0.15, ducking
  under voice, 2s in / 3s out fades. Dead-silence tails are BY DESIGN
  when no bed.
- `sfx/` empty = `sfx_whoosh` stays effectively off (also gated on hooks).

## Mastering

- Loudnorm to LUFS −14, true peak ≤ −2.0 dBTP, LRA 11 (YouTube).
  1920×1080/24 YouTube, 1080×1920 vertical presets. EDL + cost logged
  per chapter (`chNNN_edl.json`, cost_log).
- Captions: `subtitles: true` BURNS them into the picture (2026-10-06).
  A soft `mov_text` track was tried first and was effectively invisible — off
  by default in most players, ignored in several — so burn is the default and
  the `.srt` sidecar is the upload artefact, not the delivery.
- Captions are built to a published standard, not to taste: 42 chars/line,
  2 lines max, 17 CPS target with 20 a hard ceiling, 0.83s–6s per cue,
  2-frame minimum gap, sentence case. Netflix Timed Text Style Guide
  (General Requirements + English USA + Timing), cross-checked against BBC,
  EBU and AVTpro. Tinker 1955 measured all-caps at 9.5–19% slower reading,
  so caps is for 2–3 word hook cards only, never narration.
- Timing comes from `speech_bounds` on the actual wav, NOT from estimating
  text length. The estimate drifted ~⅓ s per line, which is visible.
- Font size is DERIVED from the title-safe area, not hardcoded: 42 characters
  must fit 90% of frame width. Horizontal 62–86px by resolution; vertical
  gets 30% bottom dead zone for the platform UI, horizontal 10%.
- The burn passes the ASS by BARE FILENAME with `cwd` set to the clip's
  directory. An absolute path into the `subtitles` filter dies on the drive
  colon (`Unable to parse original_size`) — that bug meant `subtitles: true`
  had never once produced captions.
- A narration longer than its slot cannot meet 17 CPS by any timing change
  (170 chars in 3.6s is a 47 CPS floor). That is reported as a WARN, because
  the fix is a shorter script line, not a cleverer caption.
- P3 LANDED: `.srt` sidecar auto-exports with every `.ass`; `bridge_len`
  is constant 0.0 (hard-cut); `build_shorts_cut` (9:16 center-crop) +
  `pick_thumbnail` (sharpest A-still) power the Shorts pipeline.
