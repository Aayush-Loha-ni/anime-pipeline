"""Build a VERIFIED voice-clone reference from raw recordings.

Why this exists: the clone is only as good as its reference, and nothing in the
pipeline checked the reference was *speech*. The previous one was an 8.63s cut
of a pop song ("Little Things") — all 41.77s of the source file was sung — and
because it was sung, full-ref conditioning echoed the lyric into every
generation. That is why the server runs in x_vector_only_mode, which the Qwen3-TTS
docs describe as the lower-quality option. This tool is the way out of that
loop: give it spoken audio and it hands back a reference plus a transcript
that has been checked against that audio.

It transcribes the clip it chose and writes that transcript into ref.json, so
ref_text and ref_audio cannot drift apart again. ICL mode (x_vector_only=False)
is only sound when they agree, and that is the whole ballgame:

  ICL mode      - uses the reference's audio tokens too. Higher cloning
                  fidelity, better voice-characteristic preservation, more
                  natural prosody. Requires an accurate ref_text.
  x_vector_only - speaker embedding only. No ref_text needed, but the model
                  card says cloning quality may be reduced.

Run:
    python prep_ref.py "path/to/recording.m4a" [more recordings...]
    python prep_ref.py --seconds 25 --dry-run "path/to/recording.m4a"

Then restart cosy-server/server.py and delete voice/*.wav, or bump
tts_clone_mode in ref.json so the pipeline's cache key retires them.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
REF_JSON = HERE / "ref.json"
SR = 24000                      # Qwen3-TTS tokenizer rate; do not resample past it
FRAME = 480                     # 20 ms at 24 kHz
TARGET_PEAK = -3.0              # dBFS. LOUD_REF_TARGET_PEAK_DB in the pipeline.


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------
def load_24k_mono(path):
    """ffmpeg does the codec work — m4a/aac/mp3/flac/wav all land here."""
    out = Path(tempfile.gettempdir()) / f"_prep_ref_{abs(hash(str(path)))}.wav"
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(path),
         "-ac", "1", "-ar", str(SR), str(out)],
        capture_output=True, text=True)
    if r.returncode or not out.is_file():
        raise RuntimeError(f"ffmpeg could not decode {path}: {r.stderr[-300:]}")
    a, sr = sf.read(str(out), dtype="float32")
    out.unlink(missing_ok=True)
    return np.ascontiguousarray(a), sr


# --------------------------------------------------------------------------
# voiced-region detection
# --------------------------------------------------------------------------
def regions(x, sr=SR):
    """Energy-gated speech regions, merged across short gaps."""
    hop = FRAME
    n = len(x) // hop * hop
    if n == 0:
        return []
    env = np.array([np.sqrt((x[i:i + hop] ** 2).mean() + 1e-12)
                    for i in range(0, n, hop)])
    # Speech sits far above the noise floor; a floor-relative gate survives both
    # a quiet studio take and a noisy phone take without a magic absolute level.
    floor = np.percentile(env, 10)
    thr = max(floor * 4.0, env.max() * 0.06)
    on = env > thr
    if not on.any():
        return []
    edges = np.where(np.diff(on.astype(np.int8)) != 0)[0] + 1
    bounds = [0, *edges.tolist(), len(on)]
    raw = [(bounds[i] * hop / sr, bounds[i + 1] * hop / sr)
           for i in range(len(bounds) - 1) if on[bounds[i]]]
    merged = []
    for s, e in raw:
        if merged and s - merged[-1][1] < 0.35:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s > 0.5]


def snr_db(x, sr=SR):
    env = np.array([np.sqrt((x[i:i + FRAME] ** 2).mean() + 1e-12)
                    for i in range(0, len(x) - FRAME, FRAME)])
    floor = np.percentile(env, 10) + 1e-12
    return float(20 * np.log10(env.max() / floor))


# --------------------------------------------------------------------------
# pick, trim, level
# --------------------------------------------------------------------------
def _longest(runs):
    return max(runs, key=lambda c: c[-1][1] - c[0][0]) if runs else None


def pick_window(regions_, max_s):
    """Longest run of regions whose total span fits max_s.

    Greedy: extend while the next region still fits, then close. Never splits a
    region, so the clip starts and ends on a breath rather than mid-word.
    """
    runs, cur = [], []
    for r in regions_:
        if not cur:
            cur = [r]
        elif cur[-1][1] - cur[0][0] + (r[1] - r[0]) <= max_s:
            cur.append(r)
        else:
            runs.append(cur)
            cur = [r]
    if cur:
        runs.append(cur)
    best = _longest(runs)
    return (best[0][0], best[-1][1]) if best else None


def normalise(x, target_db=TARGET_PEAK):
    peak = float(np.abs(x).max())
    if peak < 1e-9:
        return x
    return x * (10 ** (target_db / 20) / peak)


def transcribe(path):
    """Whisper is already a pipeline dependency for QA; reuse it rather than
    adding another dependency. Returns "" if it cannot load."""
    try:
        import whisper
        import torch
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        m = whisper.load_model("small.en", device=dev)
        r = m.transcribe(str(path), language="en", temperature=0.0,
                         condition_on_previous_text=False)
        return (r.get("text") or "").strip()
    except Exception as e:
        print(f"  ! whisper failed ({type(e).__name__}: {e})", file=sys.stderr)
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--seconds", type=float, default=25.0,
                    help="max reference length; Qwen3-TTS docs say 3s+ helps, "
                         "longer usually better. 25-30s of clean speech is the "
                         "sweet spot for ICL mode.")
    ap.add_argument("--out", default=str(HERE.parent / "voice_user" / "ref_24k.wav"))
    ap.add_argument("--dry-run", action="store_true",
                    help="analyse, write the candidate wav so you can listen "
                         "to it, and report — but do not touch ref.json")
    ap.add_argument("--icl", action="store_true",
                    help="write x_vector_only=false (ICL mode). Only correct "
                         "once the transcript below matches the audio.")
    args = ap.parse_args()

    chunks = []
    for p in args.inputs:
        x, sr = load_24k_mono(p)
        rs = regions(x, sr)
        voiced = sum(e - s for s, e in rs)
        print(f"\n{p}")
        print(f"  {len(x)/sr:6.2f}s   {len(rs)} voiced regions   "
              f"{voiced:5.2f}s voiced ({100*voiced/(len(x)/sr):.0f}%)")
        print(f"  SNR {snr_db(x, sr):5.1f} dB")
        if voiced < args.seconds * 0.6:
            print(f"  WARNING: only {voiced:.1f}s voiced, wanted "
                  f"~{args.seconds:.0f}s. The reference will be shorter than ideal.")
        chunks.append((x, rs))

    x = np.concatenate([c[0] for c in chunks]) if len(chunks) > 1 else chunks[0][0]
    rs = []
    off = 0.0
    for c in chunks:
        rs += [(s + off, e + off) for s, e in c[1]]
        off += len(c[0]) / SR
    win = pick_window(rs, args.seconds)
    if not win:
        print("\nNo speech found — nothing to do.")
        return 1
    clip = x[int(win[0] * SR):int(win[1] * SR)]
    clip = normalise(clip)
    print("  NOTE: the reference MUST be speech. Cloned from song, the model "
          "learns\n  sustained vowels, vibrato and musical dynamics instead of "
          "conversational\n  prosody, and tends to echo the lyric into every "
          "generation.")
    print(f"\nchosen window {win[0]:.2f}-{win[1]:.2f}s  ({len(clip)/SR:.2f}s)  "
          f"peak {20*np.log10(np.abs(clip).max()):.1f} dBFS  SNR {snr_db(clip):.1f} dB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), clip, SR, subtype="PCM_16")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB, {SR} Hz mono PCM16)")

    text = transcribe(out)
    if not text:
        print("  ! no transcript — leave ref.json 'text' by hand; ICL mode "
              "cannot be enabled until it matches the audio exactly.")
        return 0
    print(f"\n  transcript: {text}")
    if args.dry_run:
        print("  (dry run — ref.json not modified)")
        return 0

    cfg = {}
    if REF_JSON.is_file():
        try:
            cfg = json.loads(REF_JSON.read_text(encoding="utf-8"))
        except ValueError:
            cfg = {}
    cfg["wav"] = str(out.resolve())
    cfg["text"] = text
    # single source of truth for BOTH server.py and _tts_fingerprint()
    cfg["x_vector_only"] = not args.icl
    cfg.setdefault("language", "English")
    REF_JSON.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {REF_JSON}")
    print(f"  x_vector_only = {cfg['x_vector_only']}"
          + ("  (ICL — higher fidelity, correct only because the transcript "
             "above was taken from this exact audio)" if args.icl else
             "  (speaker-embedding only; re-run with --icl once you are "
             "happy with the clip and the transcript)"))
    print("\nnext: restart cosy-server/server.py, then delete voice/*.wav "
          "(or let the fingerprint retire them) and re-run the voice phase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())