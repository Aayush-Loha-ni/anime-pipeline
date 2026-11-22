"""Synthesise a run of narration lines as ONE continuous passage, then split it.

Why this exists
---------------
Every sentence was previously a separate engine call. The model restarts its
prosody from scratch each time, so pitch does not flow across a sentence
boundary - it jumps. Measured on ch001:

    mean |pitch step| across a sentence boundary
        one call per sentence      4.84 semitones   (2 of 4 seams > 3 st)
        one call for the run       1.42 semitones   (0 of 4 seams)

A human speaker moves 1-2 semitones across a boundary inside one breath group.
That discontinuity is a large part of what reads as "abrupt" between sentences,
and no amount of gap or level tuning can fix it: the seam is in the pitch, not
in the silence.

Why the research agrees
-----------------------
Makarov et al., Interspeech 2022 ("Simple and Effective Multi-sentence TTS with
Expressive and Coherent Prosody"): long context disambiguates prosody and
improves modelling of inter-sentence pauses, and synthesis on multi-sentence
input using a model not trained with long context "does not work well".

He et al., CoNLL 2025: TTS systems depend on explicit cues to place intonational
boundaries, and otherwise fall back to the statistically most likely pattern -
i.e. generic. Keeping real sentence punctuation in the request is what lets the
model place the boundaries itself.

Why the API stays per-line
--------------------------
Scenes are separate timeline slots with separate durations and separate lip-sync.
This module is therefore a BATCHER: it produces exactly the same one-wav-per-scene
artefacts, so nothing downstream changes. The only difference is that the audio
was generated as a single continuous read.

What it deliberately does NOT do
--------------------------------
* No pitch correction. Smoothing the seam by shifting each segment's pitch would
  be cheaper, but it changes timbre and costs speaker similarity - the one thing
  the whole clone setup exists to preserve.
* No synthetic silence. The pause at a boundary is the engine's own, placed
  where the sentence actually ends. Inserting a fixed gap on top is what produced
  the earlier 1.24s gaps.
"""

import numpy as np

SR = 24000
FRAME = 1024
HOP = 240


def _rms_envelope(y, sr=SR):
    n = len(y)
    w = FRAME
    out = np.empty(max(1, n // HOP))
    for i in range(len(out)):
        a = i * HOP
        b = min(n, a + w)
        out[i] = float(np.sqrt((y[a:b] ** 2).mean())) if b > a else 0.0
    return out


def whisper_boundaries(wav_path, n_parts, model=None, device="cuda:0"):
    """Cut points from Whisper's own segment timestamps.

    Energy-based guessing does not work on TTS narration. Measured on 8 real
    ch001 scenes: the passage synthesises beautifully (4.3% WER as one file) but
    splitting it on energy gave 52.4% WER, because the cuts landed between the
    wrong sentences. Whisper already knows where the sentences are.

        segment 0.00 -> 1.56  "The sky was a flat, dead white."
        segment 2.02 -> 3.40  "Edwin walked the streets like a ghost."
        segment 3.58 -> 4.52  "But the streets felt wrong."

    One segment per sentence, boundaries exactly where the sentences end. Returns
    None if the segment count does not match `n_parts` - the caller then falls
    back rather than splitting at the wrong places.

    temperature=0 makes this deterministic, so the same passage always splits the
    same way and the cache stays meaningful.
    """
    import whisper
    m = model if model is not None else whisper.load_model("small.en", device=device)
    res = m.transcribe(str(wav_path), language="en", temperature=0.0,
                       condition_on_previous_text=False)
    segs = [s for s in (res.get("segments") or []) if s.get("end", 0) > s.get("start", 0)]
    if len(segs) != n_parts:
        return None
    cuts = [int(round((segs[k]["start"] + segs[k + 1]["end"]) / 2 * SR))
            for k in range(n_parts - 1)]
    return cuts


def find_boundaries(y, n_parts, sr=SR, min_gap_s=0.06):
    """Split `y` into `n_parts` at its quietest points, closest to even spacing.

    Returns sample indices, one per cut, so `n_parts` pieces come out. Silences
    shorter than `min_gap_s` are ignored: a plosive between words is a few
    milliseconds and cutting there would clip a consonant.

    Falls back to even proportional cuts when the audio has no usable silence,
    which is better than failing: a proportional cut is roughly right, because a
    TTS line's duration is close to proportional to its length.
    """
    n = len(y)
    if n_parts <= 1:
        return []
    env = _rms_envelope(y, sr)
    if len(env) < n_parts * 3:
        return [int(n * k / n_parts) for k in range(1, n_parts)]
    peak = float(env.max()) or 1.0
    quiet = env < peak * 0.04
    # runs of quiet frames
    runs, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * HOP / sr >= min_gap_s:
                runs.append(((i + j) / 2.0) * HOP)
            i = j
        else:
            i += 1
    if not runs:
        return [int(n * k / n_parts) for k in range(1, n_parts)]

    # Greedy: for each cut, take the unused silence nearest the ideal position.
    cuts, used = [], set()
    for k in range(1, n_parts):
        ideal = n * k / n_parts
        best, best_cost = None, None
        for idx, pos in enumerate(runs):
            if idx in used:
                continue
            # never cut inside 150ms of an already-made cut
            if any(abs(pos - c) < 0.15 * sr for c in cuts):
                continue
            cost = abs(pos - ideal)
            if best_cost is None or cost < best_cost:
                best, best_cost = idx, cost
        if best is None:
            cuts.append(int(ideal))
        else:
            used.add(best)
            cuts.append(int(runs[best]))
    cuts.sort()
    # guarantee strictly increasing and inside the file
    clean, last = [], 0
    for c in cuts:
        c = max(last + 1, min(int(c), n - 1))
        clean.append(c)
        last = c
    return clean


def read_wav(path):
    import wave
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError("expected 16-bit mono wav: {}".format(path))
        return (np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
                .astype(np.float32) / 32768.0), w.getframerate()


def write_wav(path, y, sr=SR):
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.clip(y * 32767, -32768, 32767).astype("<i2").tobytes())


def validate_cuts(y, cuts, sr=SR, peak_frac=0.06, window_s=0.02):
    """Are these cut points actually in silence?

    A cut placed on audio clips a phoneme, and a clipped word is worse than the
    pitch seam that passage synthesis was introduced to remove. When the engine
    does not put a usable pause at a sentence boundary, the honest move is to
    reject the whole run and let the caller fall back to per-line synthesis.

    Returns (ok, first_bad_index).
    """
    if not cuts:
        return True, None
    w = int(window_s * sr)
    peak = float(np.abs(y).max()) or 1.0
    for i, c in enumerate(cuts):
        a, b = max(0, c - w), min(len(y), c + w)
        if float(np.abs(y[a:b]).max()) > peak * peak_frac:
            return False, i
    return True, None


def split_passage(wav_path, out_paths, sr=SR, cuts=None):
    """Cut one continuous wav into `len(out_paths)` files. Returns the cut points.

    `cuts` should come from whisper_boundaries(). Without it the energy heuristic
    is used, which is measurably wrong on narration - see that function.

    Raises ValueError if a cut point is not in silence - the caller then falls
    back to per-line synthesis, because a word clipped at a join is a worse
    defect than the seam this whole mechanism exists to remove.

    Each piece keeps the tail silence on its left so a word's onset is not
    shaved; the caller re-imposes its own gap when it lays the timeline out.
    """
    y, file_sr = read_wav(wav_path)
    sr = file_sr or sr
    if cuts is None:
        cuts = whisper_boundaries(wav_path, len(out_paths))
    if cuts is None:
        cuts = find_boundaries(y, len(out_paths), sr)
    cuts = [int(c) for c in cuts][:len(out_paths) - 1]
    ok, bad = validate_cuts(y, cuts, sr)
    if not ok:
        raise ValueError(
            "cut {} of {} landed on audio ({:.3f}s); refusing to split a word".format(
                bad + 1, len(cuts), cuts[bad] / sr))
    edges = [0] + cuts + [len(y)]
    for i, p in enumerate(out_paths):
        a, b = edges[i], edges[i + 1]
        piece = y[a:b]
        if i > 0:
            piece = piece[np.abs(piece) > 1e-5][:max(1, len(piece) - int(0.02 * sr))]
        if len(piece) < int(0.05 * sr):
            piece = y[a:b]
        write_wav(p, piece, sr)
    return cuts
