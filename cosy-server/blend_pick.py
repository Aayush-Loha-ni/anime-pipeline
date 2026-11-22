"""Pick a blend by PAIRED test, and prove it on sentences it was not chosen on.

blend_sweep.py found a promising knee, but it compares means, and means at this
scale move between runs (measured: the SAME w=1.0 config scored SIM 0.7524 in one
sweep and 0.8167 in the next, because the narrator reference had been
re-synthesised between them). So the knee cannot be chosen from means.

This does it properly:

  1. for each candidate weight, generate every sentence
  2. compare each weight to w=1.0 **paired per sentence**, same run, same
     narrator clip
  3. pick the knee on the paired ratio
  4. re-test the chosen weight on a DISJOINT set of sentences

Step 4 is the part that catches a knee fitted to noise. With n=6 the sweep showed
-0.0135 SIM for +0.245 UTMOS; at n=12 the same axis showed -0.079 SIM for +0.177.
The 6-sentence knee was partly luck, and only the disjoint retest distinguishes
those.

Also fixes the narrator to a CLIP ON DISK, chosen once, so weights are compared
against the same donor. Re-synthesising it per run is what made the two sweeps
disagree.

Usage:
    python blend_pick.py --n 6 --holdout 6
"""

import argparse
import json
import pathlib
import statistics as st
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_voice as SV  # noqa: E402

NARRATOR_TEXT = ("The city that wrote his fate had no logic for a canon, "
                 "and no reason for one.")
FIXED_NARRATOR = HERE / "narrator_ref.wav"


def blend(a, b, w):
    return (a.float() * w + b.float() * (1.0 - w)).to(a.dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6, help="sentences to SELECT on")
    ap.add_argument("--holdout", type=int, default=6,
                    help="sentences to VALIDATE on (disjoint)")
    ap.add_argument("--sentences", default=str(HERE / "heldout33.json"))
    ap.add_argument("--weights", default="1.0,0.9,0.8,0.7")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--keep-narrator", default=str(FIXED_NARRATOR))
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
    import soundfile as sf
    import whisper

    texts = json.loads(pathlib.Path(args.sentences).read_text(encoding="utf-8"))
    keys = sorted(texts)
    sel = keys[:args.n]
    hold = keys[args.n:args.n + args.holdout]
    weights = [float(w) for w in args.weights.split(",")]
    print("  select on  : {}".format(sel))
    print("  validate on: {}".format(hold))

    ref_cfg = json.loads((HERE / "ref.json").read_text(encoding="utf-8"))
    user_wav = pathlib.Path(ref_cfg["wav"])
    wm = whisper.load_model("small.en", device=args.device)

    def tx(p):
        return (wm.transcribe(str(p), language="en", temperature=0.0,
                              condition_on_previous_text=False)
                .get("text") or "").strip()

    base = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map=args.device,
        dtype=torch.bfloat16, attn_implementation="sdpa")

    # ONE narrator, pinned to disk. Every weight is compared against this same
    # donor, which is what makes the sweep internally consistent.
    narr = pathlib.Path(args.keep_narrator)
    if not narr.is_file():
        cv = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map=args.device,
            dtype=torch.bfloat16, attn_implementation="sdpa")
        wavs, sr = cv.generate_custom_voice(
            text=NARRATOR_TEXT, language="English", speaker="Ryan",
            instruct="Measured, cinematic narration. Restrained, never announcer.")
        sf.write(str(narr), wavs[0], sr)
        del cv
        torch.cuda.empty_cache()
        print("  narrator ref synthesised once -> {} ({:.1f}s)".format(
            narr.name, len(wavs[0]) / sr))
    else:
        print("  narrator ref pinned: {} (reused, so weights are comparable)".format(
            narr.name))

    user_item = base.create_voice_clone_prompt(
        ref_audio=str(user_wav), ref_text=ref_cfg["text"], x_vector_only_mode=False)
    if isinstance(user_item, list):
        user_item = user_item[0]
    narr_item = base.create_voice_clone_prompt(
        ref_audio=str(narr), ref_text=tx(narr), x_vector_only_mode=False)
    if isinstance(narr_item, list):
        narr_item = narr_item[0]

    truth = HERE / "rerank_truth.wav"
    tmp = HERE / "_bp_out.wav"

    def prompt_for(w):
        return VoiceClonePromptItem(
            ref_code=narr_item.ref_code,
            ref_spk_embedding=blend(user_item.ref_spk_embedding,
                                   narr_item.ref_spk_embedding, w),
            x_vector_only_mode=False, icl_mode=True, ref_text=NARRATOR_TEXT)

    def run(w, kset):
        out = {}
        for k in kset:
            wavs, sr = base.generate_voice_clone(
                text=texts[k], language="English",
                voice_clone_prompt=[prompt_for(w)])
            sf.write(str(tmp), wavs[0], sr)
            out[k] = (SV.speaker_similarity(truth, tmp) or 0.0,
                      SV.utmos(tmp) or 0.0,
                      SV.wer(tx(tmp), texts[k]) or 0.0)
        return out

    # ---- select, paired -------------------------------------------------
    print()
    print("  SELECTION (paired vs w=1.0, same run, same narrator clip)")
    print("  {:<6} {:>10} {:>10} {:>10} {:>9} {:>9}".format(
        "w", "dSIM", "dUTMOS", "dWER", "t_SIM", "t_UTMOS"))
    print("  (|t| > 2.2 is about p<0.05 at this n)")
    res = {}
    base_r = run(1.0, sel)
    res[1.0] = base_r
    means = {1.0: _mean(base_r)}
    for w in weights:
        if w == 1.0:
            continue
        r = run(w, sel)
        res[w] = r
        means[w] = _mean(r)
        ds = [r[k][0] - base_r[k][0] for k in sel]
        du = [r[k][1] - base_r[k][1] for k in sel]
        print("  {:<6} {:>+10.4f} {:>+10.3f} {:>+10.3f} {:>9} {:>9}".format(
            w, st.mean(ds), st.mean(du),
            st.mean([r[k][2] - base_r[k][2] for k in sel]),
            _t(ds), _t(du)))

    best = None
    for w in weights:
        if w == 1.0:
            continue
        ds = st.mean([res[w][k][0] - base_r[k][0] for k in sel])
        du = st.mean([res[w][k][1] - base_r[k][1] for k in sel])
        if du > 0.15 and ds > -0.03:          # meaningful gain, bounded cost
            ratio = du / -ds if ds < 0 else float("inf")
            if best is None or ratio > best[0]:
                best = (ratio, w)
    print()
    if best is None:
        print("  no weight passed the bar (needs UTMOS > +0.15 and SIM > -0.03)")
        tmp.unlink(missing_ok=True)
        return 1
    chosen = best[1]
    print("  selected w={} (best gain-per-likeness on the selection set)".format(chosen))

    # ---- validate, disjoint ---------------------------------------------
    print()
    print("  VALIDATION on {} sentences never used for selection".format(len(hold)))
    vb = run(1.0, hold)
    vw = run(chosen, hold)
    ds = [vw[k][0] - vb[k][0] for k in hold]
    du = [vw[k][1] - vb[k][1] for k in hold]
    print("     w=1.00   SIM {:.4f}  UTMOS {:.3f}  WER {:.3f}".format(*_mean(vb)))
    print("     w={:<6} SIM {:.4f}  UTMOS {:.3f}  WER {:.3f}".format(
        chosen, _mean(vw)[0], _mean(vw)[1], _mean(vw)[2]))
    print()
    print("     paired dSIM   {:+.4f}  t={:+.2f}   {}".format(
        st.mean(ds), _t(ds), "significant" if abs(_t(ds)) > 2.2 else "NOT significant"))
    print("     paired dUTMOS {:+.3f}   t={:+.2f}   {}".format(
        st.mean(du), _t(du), "significant" if abs(_t(du)) > 2.2 else "NOT significant"))
    print("     per-sentence dSIM: {}".format(["{:+.3f}".format(d) for d in ds]))
    tmp.unlink(missing_ok=True)
    print()
    real = abs(_t(du)) > 2.2
    print("     {}".format(
        "VALIDATED on held-out sentences - this blend is real" if real else
        "did NOT hold up on held-out sentences; the selection was fitting noise"))
    print()
    print("  to ship:  python build_hybrid.py --strength custom --weight {}".format(chosen))
    return 0 if real else 1


def _mean(d):
    n = len(d)
    return (sum(v[0] for v in d.values()) / n,
            sum(v[1] for v in d.values()) / n,
            sum(v[2] for v in d.values()) / n)


def _t(deltas):
    if len(deltas) < 2:
        return 0.0
    s = st.stdev(deltas)
    if s == 0:
        return 0.0
    return st.mean(deltas) / (s / len(deltas) ** 0.5)


if __name__ == "__main__":
    sys.exit(main())