"""Is the restored reference actually better, or did it just add noise?

Measures on three axes, and only credits the restoration on all three:

  1. bandwidth   did high-frequency energy come back?
  2. UTMOS       does a MOS predictor think it sounds better?
  3. similarity  does it still sound like the SAME person? This is the one
                 restoration can quietly destroy, because a generative vocoder
                 replaces the speaker's own spectral detail with a plausible
                 average. Bandwidth and MOS can both rise while the voice drifts
                 toward the model's notion of generic speech.

A reference that wins on 1 and 2 while losing on 3 is NOT an improvement, and
this script is what stops that from shipping.
"""

import pathlib
import sys

import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_voice as SV  # noqa: E402


def band(path):
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    p = X ** 2
    hi = p[f >= 8000].sum() / p.sum()
    return hi * 100


def snr(path):
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = 480
    env = np.array([np.sqrt((x[i:i + hop] ** 2).mean() + 1e-12)
                    for i in range(0, max(1, len(x) - hop), hop)])
    p10 = float(np.percentile(env, 10)) + 1e-12
    return 20 * np.log10(float(env.max()) / p10)


def main():
    a = HERE / (sys.argv[1] if len(sys.argv) > 1 else "ref33.wav")
    b = HERE / (sys.argv[2] if len(sys.argv) > 2 else "ref33_restored.wav")

    print("  {:<22} {:>10} {:>8} {:>8} {:>8}".format(
        "reference", "hi>8k %", "SNR dB", "UTMOS", "SIM vs orig"))
    sim_a = sim_b = ut_a = ut_b = None
    for label, p in (("original", a), ("restored", b)):
        if not p.is_file():
            print("  {:<22} MISSING".format(label))
            return 1
        u = SV.utmos(p)
        s = SV.speaker_similarity(a, p)
        print("  {:<22} {:10.2f} {:8.1f} {:8.3f} {:>8}".format(
            p.name, band(p), snr(p), u if u else -1,
            "{:.4f}".format(s) if s else "-"))
        if label == "original":
            ut_a, sim_a = u, s
        else:
            ut_b, sim_b = u, s

    print()
    ok = True
    d_hi = band(b) - band(a)
    d_ut = (ut_b - ut_a) if (ut_a and ut_b) else None
    d_sim = (sim_b - sim_a) if (sim_a is not None and sim_b is not None) else None
    print("  bandwidth  {:+.2f} pp  {}".format(
        d_hi, "PASS" if d_hi > 0 else "no change"))
    ok &= d_hi > 0
    if d_ut is not None:
        print("  UTMOS      {:+.3f}     {}".format(
            d_ut, "PASS" if d_ut > 0 else "worse or flat"))
        ok &= d_ut > 0
    if d_sim is not None:
        verdict = "PASS" if d_sim > -0.05 else ("DRIFT" if d_sim < -0.05 else "flat")
        print("  similarity {:+.4f}    {}   (below -0.05 means the voice moved)".format(
            d_sim, verdict))
        ok &= d_sim > -0.05

    print()
    print("  {}".format("VERDICT: the restored reference is an upgrade"
                        if ok else
                        "VERDICT: NOT a clear upgrade - do not ship it blind"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())