"""Diagnose why a reference wins or loses: the "why" companion to ab_refs.py.

ab_refs.py answers WHETHER a reference is better. This answers WHY, because
"the recording was quieter and it still lost" is a finding worth keeping and
"the new take was in a different key" is a finding you can act on.

Everything measured here is a property of the SOURCE READINGS, because
Qwen3-TTS clones by in-context learning: it copies the reference's TIMBRE
and its PERFORMANCE. The repo's own notes record the consequence - a
13.5-semitone reference produced flat speech at UTMOS 3.41, and a
20.1-semitone one produced UTMOS 4.32. So the performance of the reading is
part of the reference.

MEASURED HERE

  session acoustics   SNR, noise floor, spectral tilt per recording
  clipping            a flat-topped waveform is a damaged signal; peak
                      0.0 dBFS on a long take is a warning, not a certificate
  local SNR           per clip against its OWN noise floor, so a clip sitting
                      in a noisier part of the take is visible instead of
                      being hidden by the session average
  transcript          how recoverable the reference's own text is. ICL
                      conditions generation on ref_text, and the repo records
                      the failure mode: a wrong ref_text teaches wrong
                      pronunciations
  pitch               median F0 and the semitone spread. A reference read in
                      a different register produces output in that register
  rate                phrases per second - a fast read is a poor reference
  gate signature      a -80 dBFS floor with a -55 dBFS pause level is a
                      NOISE GATE, not a quiet room: the silence between
                      phrases has been pushed below the speech, and the
                      step into it is audible as a delivery artefact

Run:
    python voice_diag.py                       # source recordings + clips
    python voice_diag.py --generated ab_refs   # add the generated speech
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep_ref as P  # noqa: E402

REC = Path(r"<YOUR_RECORDINGS_DIR>")
SR = P.SR
FRAME = P.FRAME


# --------------------------------------------------------------------------
# acoustic measures
# --------------------------------------------------------------------------
def noise_floor(x, sr=SR):
    n = len(x) // FRAME * FRAME
    env = np.array([np.sqrt((x[i:i + FRAME] ** 2).mean() + 1e-12)
                    for i in range(0, n, FRAME)])
    return float(np.percentile(env, 10))


def clipped_fraction(x, thresh=0.999):
    return float((np.abs(x) >= thresh).mean())


def local_snr(x, sr=SR):
    """SNR against the clip's OWN quietest frames."""
    n = len(x) // FRAME * FRAME
    env = np.array([np.sqrt((x[i:i + FRAME] ** 2).mean() + 1e-12)
                    for i in range(0, n, FRAME)])
    floor = np.percentile(env, 5)
    return float(20 * np.log10(max(env.max(), 1e-12) / max(floor, 1e-12)))


def spectral_tilt(x, sr=SR):
    """0-1 kHz over 6-12 kHz. A rumbling room or a music bed tilts it down."""
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    p = X ** 2
    lo = p[f < 1000].sum()
    hi = p[(f >= 6000) & (f < 12000)].sum()
    return float(10 * np.log10(max(lo, 1e-12) / max(hi, 1e-12)))


def f0_track(x, sr=SR, fmin=60.0, fmax=400.0):
    """Autocorrelation pitch track over the voiced frames."""
    win = int(0.04 * sr)
    hop = int(0.01 * sr)
    f0 = []
    for i in range(0, len(x) - win, hop):
        fr = x[i:i + win]
        if np.sqrt((fr ** 2).mean()) < 0.005:
            continue
        fr = fr - fr.mean()
        c = np.correlate(fr, fr, "full")[len(fr) - 1:]
        lo, hi = int(sr / fmax), int(sr / fmin)
        if hi >= len(c):
            continue
        seg = c[lo:hi]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg)) + lo
        if c[k] > 0:
            f0.append(sr / k)
    return np.array(f0) if f0 else np.array([])


def semitone_range(f0):
    if len(f0) < 10:
        return 0.0, 0.0
    med = np.median(f0)
    if med <= 0:
        return 0.0, 0.0
    st = 12 * np.log2(f0 / med)
    return float(np.percentile(st, 95) - np.percentile(st, 5)), float(med)


def phrases_per_s(x, sr=SR):
    env = np.array([np.sqrt((x[i:i + FRAME] ** 2).mean() + 1e-12)
                    for i in range(0, len(x) - FRAME, FRAME)])
    floor = np.percentile(env, 10) + 1e-12
    thr = max(floor * 4.0, env.max() * 0.06)
    on = env > thr
    if len(on) < 2:
        return 0.0
    edges = np.where(np.diff(on.astype(np.int8)) != 0)[0] + 1
    bounds = [0, *edges.tolist(), len(on)]
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if on[bounds[i]]]
    merged = []
    for s, e in spans:
        if merged and (s - merged[-1][1]) * FRAME / sr < 0.35:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return len(merged) / (len(x) / sr)


def gate_signature(x, sr=SR):
    """The level in the gaps, and how hard the step into them is. A gated
    recording has a low, hard-edged silence; a room has a soft, noisy one."""
    n = len(x) // FRAME * FRAME
    env = np.array([np.sqrt((x[i:i + FRAME] ** 2).mean() + 1e-12)
                    for i in range(0, n, FRAME)])
    floor = np.percentile(env, 10) + 1e-12
    thr = max(floor * 4.0, env.max() * 0.06)
    on = env > thr
    edges = np.where(np.diff(on.astype(np.int8)) != 0)[0]
    sil = env[~on]
    if len(sil) < 10:
        return None
    sil_db = 20 * np.log10(sil)
    steps = [20 * np.log10(max(env[i], 1e-12) / max(env[i + 1], 1e-12))
             for i in edges
             if i + 1 < len(env) and on[i] and not on[i + 1]]
    return {"silence_db": float(np.median(sil_db)),
            "silence_spread_db": float(np.percentile(sil_db, 90) - np.percentile(sil_db, 10)),
            "step_db": float(np.mean(steps)) if steps else 0.0}


# --------------------------------------------------------------------------
def source_row(label, path):
    x, sr = P.load_24k_mono(path)
    rng, med = semitone_range(f0_track(x))
    g = gate_signature(x) or {}
    print("  {:<20} SNR {:5.1f}  floor {:6.1f}  tilt {:5.1f}  clipped {:6.4f}%  "
          "F0 {:5.1f}Hz  st {:4.1f}  pps {:.2f}  silence {:6.1f}  step {:4.1f}".format(
              label, P.snr_db(x, sr), 20 * np.log10(max(noise_floor(x), 1e-12)),
              spectral_tilt(x), 100 * clipped_fraction(x), med, rng,
              phrases_per_s(x), g.get("silence_db", 0.0), g.get("step_db", 0.0)))
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated", default="",
                    help="also measure the GENERATED speech in this ab_refs "
                         "output dir")
    ap.add_argument("--clips", nargs="*", default=["rec33_sent", "rec35_sent"],
                    help="clip dirs to report per-clip detail for")
    args = ap.parse_args()

    print("SOURCE RECORDINGS")
    for n in (4, 30, 32, 33, 35):
        p = REC / ("Recording (%d).m4a" % n)
        if p.is_file():
            source_row("Recording (%d)" % n, p)

    print("\nCLIP DETAIL (local SNR, so a noisy part of a take shows up)")
    for d in args.clips:
        dd = HERE / d
        if not dd.is_dir():
            continue
        meta_p = dd / "meta.json"
        meta = {}
        if meta_p.is_file():
            meta = {m["file"]: m for m in
                    json.loads(meta_p.read_text(encoding="utf-8"))}
        for wav in sorted(dd.glob("*.wav")):
            x, sr = P.load_24k_mono(wav)
            extra = ""
            if wav.name in meta:
                extra = "  agree {:3.0%}".format(meta[wav.name]["agreement"])
            print("  {:<14} {:<10} local SNR {:5.1f} dB  clipped {:6.4f}%{}".format(
                d, wav.name, local_snr(x), 100 * clipped_fraction(x), extra))
        ags = [m["agreement"] for m in meta.values()]
        if ags:
            print("  {:<14} transcript recoverability: min {:.0%}  mean {:.0%}".format(
                d, min(ags), sum(ags) / len(ags)))

    if args.generated:
        root = HERE / args.generated
        print("\nGENERATED SPEECH (what actually ships)")
        groups = {}
        for w in root.rglob("*.wav"):
            if w.is_file():
                groups.setdefault(w.parent.parent.name + "/" + w.parent.name,
                                  []).append(w)
        for arm in sorted(groups):
            meds, rngs, durs = [], [], []
            for w in sorted(groups[arm]):
                x, sr = sf.read(str(w), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                r, m = semitone_range(f0_track(x, sr))
                if m > 0:
                    meds.append(m)
                    rngs.append(r)
                durs.append(len(x) / sr)
            if meds:
                print("  {:<28} F0 {:6.1f}Hz  st {:4.1f}  dur {:5.2f}s  "
                      "n={}".format(arm, np.mean(meds), np.mean(rngs),
                                    np.mean(durs), len(meds)))


if __name__ == "__main__":
    main()
