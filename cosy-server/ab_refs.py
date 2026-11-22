"""A/B a candidate reference against the shipped one, with the confounds
controlled.

The question this answers is always the same: "the user recorded a new take.
Is it a better TTS reference than the one we ship?" Two things make that
question easy to answer wrongly, and both are handled here.

WHAT MAKES IT EASY TO ANSWER WRONGLY

  1. TRUTH-SESSION BIAS. Speaker similarity is scored against reference audio
     of the same person. If the truth comes from one session and the
     incumbent reference comes from another, the truth and the incumbent
     share a room, a mic position and a noise floor, and speaker embeddings
     carry some of that. MEASURED on this project: a Recording-32 clip
     scores 0.9677 against a Recording-33 clip but only 0.9531 against a
     Recording-35 clip, so an incumbent drawn from Recording 33 starts
     ~0.015 ahead for reasons that have nothing to do with reference quality.
     FIX: truth clips come from several sessions (30, 32, 4) that are
     neither arm's source, and the paired delta is ALSO reported against
     each truth session separately. A real quality difference shows up
     against every truth; a confound shows up against one.

  2. PROBE LUCK. pick_ref.py measured a 0.0323 SIM spread between two runs
     of the same probe, so a single challenger chosen on a few sentences is
     a coin flip - and it can pick a clip that happens to be a bad
     conditioning clip.
     FIX: EVERY candidate of every arm is swept on the holdout. This is
     deliberately optimistic for the challenger - it gets its best clip,
     selected on the holdout - so a loss cannot be blamed on the choice.

THE VERDICT IS PAIRED. Every arm speaks the SAME sentences, and the
comparison is the per-sentence paired delta with a t-statistic. |t| > 2.2 is
the bar the repo already uses in pick_ref.py and direct_ab.py.

Scored on three axes, because they fail independently:
  SIM   speaker similarity to the neutral truth - the likeness metric
  UTMOS predicted naturalness
  WER   word error rate of whisper's transcript - intelligibility

Also reported: each reference's own similarity to the truth (the CEILING for
that arm, since a clone cannot be closer to the truth than its own reference
is) and the duration of each arm's output, because score_voice.py documents
that UTMOS correlates with duration and a length confound has to be visible.

Usage:
    python ab_refs.py --arms rec35_sent=rec35_sent --incumbent voice_reference.wav
    python ab_refs.py --arms a=dirA b=dirB --incumbent voice_reference.wav --n 12
"""
import argparse
import json
import pathlib
import statistics as st
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_voice as SV  # noqa: E402

REC = Path(r"<YOUR_RECORDINGS_DIR>")
# Truth sessions: neither the incumbent's source nor any challenger's.
TRUTH_SESSIONS = [30, 32, 4]


# --------------------------------------------------------------------------
# sentences and truth
# --------------------------------------------------------------------------
def heldout_sentences(n):
    """Production narration lines from the most recently rendered chapters.

    Real pipeline text: the model is never conditioned on these, and they are
    the shape of text the engine actually speaks.
    """
    out = []
    # Root scenes/ was relocated into novels/<slug>/scenes/; holdout text must
    # still be found or the A/B silently has zero sentences to validate on.
    dirs = [HERE.parent / "scenes"] + sorted((HERE.parent / "novels").glob("*/scenes"))
    for ch in (60, 59, 58, 57, 56):
        p = next((d / f"ch{ch:03d}.json" for d in dirs
                  if (d / f"ch{ch:03d}.json").is_file()), None)
        if p is None:
            continue
        scenes = json.loads(p.read_text(encoding="utf-8"))
        scenes = scenes.get("scenes", scenes) if isinstance(scenes, dict) else scenes
        for sc in scenes:
            t = (sc.get("narration") or "").strip()
            if 25 <= len(t) <= 160:
                out.append(t)
    out.sort(key=lambda s: (len(s), s))
    if len(out) <= n:
        chosen = out
    else:
        step = len(out) / n
        chosen = [out[int(i * step)] for i in range(n)]
    return {"s{:02d}".format(i): t for i, t in enumerate(chosen)}


def truth_clips(per_session=2):
    """Long stretches of the user's own voice, from sessions that are neither
    arm's source."""
    import prep_ref as P
    out = HERE / "ab_truth"
    out.mkdir(exist_ok=True)
    paths = []
    for n in TRUTH_SESSIONS:
        src = REC / ("Recording (%d).m4a" % n)
        if not src.is_file():
            continue
        x, sr = P.load_24k_mono(src)
        rs = P.regions(x, sr)
        longs = sorted([r for r in rs if r[1] - r[0] >= 9.0],
                       key=lambda r: r[1] - r[0], reverse=True)
        for i, (s, e) in enumerate(longs[:per_session]):
            p = out / ("t%02d_%d.wav" % (n, i))
            sf.write(str(p), P.normalise(x[int(s * sr):int(e * sr)]), sr,
                     subtype="PCM_16")
            paths.append(p)
    return paths


def sim_to_truths(truths, wav):
    vals = [SV.speaker_similarity(str(t), str(wav)) for t in truths]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else 0.0


def tstat(deltas):
    if len(deltas) < 2:
        return 0.0
    sd = st.stdev(deltas)
    return 0.0 if sd == 0 else st.mean(deltas) / (sd / len(deltas) ** 0.5)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="name=dir pairs; each dir holds reference wavs")
    ap.add_argument("--incumbent", default="voice_reference.wav")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--probe", type=int, default=4)
    ap.add_argument("--out", default="ab_refs")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    pool = []
    for spec in args.arms:
        name, _, d = spec.partition("=")
        cands = sorted((HERE / d).glob("*.wav"))
        if not cands:
            print("no candidates in {}".format(d))
            return 1
        pool += [(name + "/" + c.name, c) for c in cands]
    inc = HERE / args.incumbent
    if not inc.is_file():
        print("no incumbent {}".format(inc))
        return 1
    inc_label = "incumbent/" + inc.name
    pool.append((inc_label, inc))

    texts = heldout_sentences(args.n)
    keys = ["s{:02d}".format(i) for i in range(len(texts))]
    probe_keys, hold_keys = keys[:args.probe], keys[args.probe:]
    print("  probe   {} sentences (choose on)".format(len(probe_keys)))
    print("  holdout {} sentences (validate on)".format(len(hold_keys)))

    truths = truth_clips()
    by_session = {}
    for t in truths:
        by_session.setdefault(t.name.split("_")[0], []).append(t)
    print("  truth: {} clips from sessions {}".format(len(truths), TRUTH_SESSIONS))

    from qwen_tts import Qwen3TTSModel
    import whisper

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map=args.device,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    wm = whisper.load_model("medium.en", device=args.device)
    tcache, prompts = {}, {}

    def transcript(p):
        if p not in tcache:
            tcache[p] = (wm.transcribe(str(p), language="en", temperature=0.0,
                                      condition_on_previous_text=False)
                         .get("text") or "").strip()
        return tcache[p]

    def prompt_for(path):
        """The clone prompt is expensive and is per-REFERENCE, not
        per-sentence, so it is built once and reused - as the server does."""
        if str(path) not in prompts:
            pr = model.create_voice_clone_prompt(
                ref_audio=str(path), ref_text=transcript(path),
                x_vector_only_mode=False)
            prompts[str(path)] = pr if isinstance(pr, list) else [pr]
        return prompts[str(path)]

    out_root = HERE / args.out
    tmp = HERE / "_abrefs_tmp.wav"

    def gen(path, key, arm):
        wavs, sr = model.generate_voice_clone(
            text=texts[key], language="English",
            voice_clone_prompt=prompt_for(path))
        sf.write(str(tmp), wavs[0], sr)
        dst = out_root / arm / (key + ".wav")
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), wavs[0], sr)
        got = (wm.transcribe(str(tmp), language="en", temperature=0.0,
                              condition_on_previous_text=False)
               .get("text") or "").strip()
        return {"sim": sim_to_truths(truths, tmp),
                "per_session": {s: sim_to_truths(v, tmp)
                                for s, v in by_session.items()},
                "utmos": SV.utmos(tmp) or 0.0,
                "wer": SV.wer(got, texts[key]) or 0.0,
                "dur": SV.wav_duration(tmp)}

    # ---- reference ceiling ------------------------------------------------
    print("\n  REFERENCE CEILING (each reference vs the truth set; a clone "
          "cannot beat this)")
    ceilings = {label: sim_to_truths(truths, path) for label, path in pool}
    for label in sorted(ceilings, key=lambda k: -ceilings[k]):
        print("     {:<28} {:.4f}".format(label, ceilings[label]))

    # ---- stage 1: probe ---------------------------------------------------
    print("\n  STAGE 1  probing {} refs on {} sentences".format(
        len(pool), len(probe_keys)))
    table = {}
    for label, path in pool:
        rows = [gen(path, k, "probe/" + label) for k in probe_keys]
        table[label] = rows
        print("     {:<28} SIM {:.4f}  UTMOS {:.3f}  WER {:.3f}".format(
            label, np.mean([r["sim"] for r in rows]),
            np.mean([r["utmos"] for r in rows]),
            np.mean([r["wer"] for r in rows])))

    # ---- stage 2: paired holdout, EVERY candidate ------------------------
    print("\n  STAGE 2  paired holdout on {} unseen sentences, every "
          "candidate".format(len(hold_keys)))
    results = {label: [gen(dict(pool)[label], k, "hold/" + label)
                       for k in hold_keys] for label, _ in pool}
    inc_rows = results[inc_label]

    print()
    print("  " + "=" * 92)
    print("  {:<28} {:>9} {:>9} {:>9} {:>8} {:>9}".format(
        "arm", "SIM", "UTMOS", "WER", "dur", "ceiling"))
    for label, _ in pool:
        rows = results[label]
        print("  {:<28} {:9.4f} {:9.3f} {:9.4f} {:8.2f} {:9.4f}".format(
            label, np.mean([r["sim"] for r in rows]),
            np.mean([r["utmos"] for r in rows]),
            np.mean([r["wer"] for r in rows]),
            np.mean([r["dur"] for r in rows]), ceilings[label]))
    print("  " + "-" * 92)

    print("  every candidate vs the incumbent, per truth session")
    for label, _ in pool:
        if label == inc_label:
            continue
        ch = results[label]
        d = [a["sim"] - b["sim"] for a, b in zip(ch, inc_rows)]
        cells = []
        for s in sorted(by_session):
            per = [a["per_session"][s] - b["per_session"][s]
                   for a, b in zip(ch, inc_rows)]
            cells.append("{} {:+.4f} (t{:+.1f})".format(
                s, st.mean(per), tstat(per)))
        print("     {:<28} dSIM {:+.4f} t{:+.2f}   {}".format(
            label, st.mean(d), tstat(d), "  ".join(cells)))

    # ---- best case per arm ------------------------------------------------
    print("\n  BEST CASE per arm (its own best holdout clip vs the incumbent)")
    for name in [a.partition("=")[0] for a in args.arms]:
        labels = [l for l in results if l.startswith(name + "/")]
        best = max(labels, key=lambda l: np.mean([r["sim"] for r in results[l]]))
        d = [a["sim"] - b["sim"] for a, b in zip(results[best], inc_rows)]
        print("     {:<14} {}  dSIM {:+.4f} t{:+.2f} wins {}/{}  "
              "dUTMOS {:+.3f}  dWER {:+.4f}  dDur {:+.2f}s".format(
                  name, best, st.mean(d), tstat(d),
                  sum(1 for x in d if x > 0), len(d),
                  np.mean([a["utmos"] for a in results[best]])
                  - np.mean([b["utmos"] for b in inc_rows]),
                  np.mean([a["wer"] for a in results[best]])
                  - np.mean([b["wer"] for b in inc_rows]),
                  np.mean([a["dur"] for a in results[best]])
                  - np.mean([b["dur"] for b in inc_rows])))

    # ---- arm vs arm, no truth-session bias between the arms --------------
    names = [a.partition("=")[0] for a in args.arms]
    if len(names) == 2:
        print("\n  ARM vs ARM, each at its own best holdout clip (no "
              "truth-session bias between the arms)")
        bests = {n: max([l for l in results if l.startswith(n + "/")],
                        key=lambda l: np.mean([r["sim"] for r in results[l]]))
                 for n in names}
        d = [a["sim"] - b["sim"] for a, b in
             zip(results[bests[names[0]]], results[bests[names[1]]])]
        print("     {} vs {}: dSIM {:+.4f} t{:+.2f} wins {}/{}".format(
            bests[names[0]], bests[names[1]], st.mean(d), tstat(d),
            sum(1 for x in d if x > 0), len(d)))

    tmp.unlink(missing_ok=True)
    print("\n  outputs: {}".format(out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
