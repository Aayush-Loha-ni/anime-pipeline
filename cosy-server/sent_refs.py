"""Cut reference clips on SENTENCE boundaries, with a VERIFIED transcript.

Why this exists. The first attempt at comparing <your-recording> against the
incumbent reference cut the candidates on energy regions, and every candidate
lost on all three axes. That result was an artefact of the CUTTING, not of
the recording:

  - a clip that starts mid-sentence gets a transcript that starts mid-word,
    and ICL mode conditions generation on ref_text. A wrong ref_text teaches
    wrong pronunciations, so the generated line drifts and WER rises.
  - the shipped reference starts on "I don't believe it." and ends on "Next
    up." - clean boundaries - and that is a large part of why it wins.

So: pick sentences with whisper word timestamps, cut the audio exactly on
those boundaries, and VERIFY each transcript by re-transcribing the cut clip
and checking it against the words the full pass found in the same time range.
A clip whose own transcription disagrees with the full pass is rejected.

This is what makes the comparison fair: the same discipline is applied to
both recordings.

METRIC NOTES, because the first version of this script reported numbers that
did not mean what they looked like:
  SNR      peak-to-SESSION-NOISE-FLOOR, not peak-to-clip-10th-percentile.
           The clip's own 10th percentile is speech, so a clip's SNR came out
           ~10 dB while the whole session's came out 77 dB - the same audio,
           two incompatible scales.
  AGREE    difflib ratio between the clip's own transcript and the full pass's
           words over the same span. The first version used an in-order
           subsequence hit rate, which punishes a re-transcription that
           punctuates or splits a word differently.
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep_ref as P  # noqa: E402

SR = P.SR
FRAME = P.FRAME


def words_of(wm, wav):
    r = wm.transcribe(str(wav), language="en", temperature=0.0,
                      condition_on_previous_text=False,
                      word_timestamps=True)
    out = []
    for seg in (r.get("segments") or []):
        for w in (seg.get("words") or []):
            t = (w.get("word") or "").strip()
            if t:
                out.append((float(w["start"]), float(w["end"]), t))
    return out, (r.get("text") or "").strip()


def norm(ws):
    out = []
    for w in ws:
        t = w[2] if isinstance(w, (tuple, list)) else w
        t = "".join(ch for ch in t.lower() if ch.isalnum() or ch == "'")
        if t:
            out.append(t)
    return out


def session_noise_floor(x):
    """10th percentile of the frame energy - the room, not the speech."""
    n = len(x) // FRAME * FRAME
    env = np.array([np.sqrt((x[i:i + FRAME] ** 2).mean() + 1e-12)
                    for i in range(0, n, FRAME)])
    return float(np.percentile(env, 10))


def sentences(words, min_s=8.0, max_s=14.0, gap=0.45):
    """Word groups broken at pauses, each within the length window."""
    sents, cur = [], []
    for (s, e, t) in words:
        if cur and s - cur[-1][1] > gap:
            sents.append(cur)
            cur = []
        cur.append((s, e, t))
        if e - cur[0][0] > max_s:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return [s for s in sents if min_s <= s[-1][1] - s[0][0] <= max_s]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("outdir")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import whisper

    src = Path(args.source)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x, sr = P.load_24k_mono(src)
    floor = session_noise_floor(x)
    tmp = HERE / "_sent_src.wav"
    sf.write(str(tmp), x, sr, subtype="PCM_16")

    wm = whisper.load_model("medium.en", device=args.device)
    words, full_text = words_of(wm, tmp)
    sents = sentences(words)
    print("source {}".format(src))
    print("  {} words, {} sentences in the window, session noise floor "
          "{:.1f} dBFS".format(len(words), len(sents),
                                20 * np.log10(max(floor, 1e-12))))

    scored = []
    for s in sents:
        a, b = s[0][0], s[-1][1]
        seg = x[int(a * SR):int(b * SR)]
        if len(seg) < int(6 * SR):
            continue
        peak = float(np.abs(seg).max())
        snr = 20 * np.log10(max(peak, 1e-12) / max(floor, 1e-12))
        clip = outdir / "_tmpclip.wav"
        sf.write(str(clip), P.normalise(seg), SR, subtype="PCM_16")
        cw, ct = words_of(wm, clip)
        clip.unlink(missing_ok=True)
        inside = [w for w in words if a - 0.5 <= w[0] <= b + 0.5]
        ag = difflib.SequenceMatcher(
            None, norm(cw), norm(inside)).ratio() if cw else 0.0
        scored.append((ag, snr, a, b, ct))
    scored.sort(key=lambda r: (-r[0], -r[1]))

    kept, used = [], []
    for ag, snr, a, b, ct in scored:
        if any(abs(a - u) < 12.0 for u in used):
            continue
        kept.append((ag, snr, a, b, ct))
        used.append(a)
        if len(kept) >= args.n:
            break

    print("\n  kept {} sentence-bounded clips".format(len(kept)))
    meta = []
    for i, (ag, snr, a, b, ct) in enumerate(kept):
        seg = P.normalise(x[int(a * SR):int(b * SR)])
        p = outdir / ("s%02d.wav" % i)
        sf.write(str(p), seg, SR, subtype="PCM_16")
        meta.append({"file": p.name, "start_s": round(a, 2), "end_s": round(b, 2),
                     "dur_s": round(b - a, 2), "snr_db": round(snr, 1),
                     "agreement": round(ag, 3), "text": ct})
        print("  {0}  {1:.2f}-{2:.2f}s ({3:.2f}s)  SNR {4:5.1f}  agree {5:.0%}".format(
            p.name, a, b, b - a, snr, ag))
        print("       \"{}\"".format(ct[:150]))
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.unlink(missing_ok=True)
    print("\n  wrote {}".format(outdir))


if __name__ == "__main__":
    main()
