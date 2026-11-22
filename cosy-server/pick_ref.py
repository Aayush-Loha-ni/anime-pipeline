"""Pick ONE better reference, validated on sentences not used to pick it.

rerank.py showed the per-candidate SIM spread is ~0.049, roughly 28x the spread
between model configurations (0.0017). That means the choice of reference clip
matters far more than any training decision - so the cheap win is not per-sentence
reranking at inference time, it is picking a better single reference ONCE.

This script does the honest version of that:
  1. score every candidate reference across the PROBE sentences
  2. pick the winner
  3. validate that winner against the shipped reference on the HELD-OUT
     sentences it never saw during selection

Without step 3 this is just fitting noise, because a spread that large is
partly per-sentence luck.
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
import rerank as RR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-refs", type=int, default=8)
    ap.add_argument("--probe", type=int, default=3,
                    help="sentences used to CHOOSE the reference")
    ap.add_argument("--sentences", default=str(HERE / "heldout33.json"))
    ap.add_argument("--shipped", default=str(HERE / "ref33.wav"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    texts = json.loads(pathlib.Path(args.sentences).read_text(encoding="utf-8"))
    keys = sorted(texts)
    probe_keys, holdout_keys = keys[:args.probe], keys[args.probe:]
    print("  probe (choose on): {}".format(probe_keys))
    print("  holdout (validate on): {}".format(holdout_keys))

    cands = RR.candidate_refs(args.n_refs)
    shipped = pathlib.Path(args.shipped)
    print("  {} candidates + the shipped ref ({})".format(len(cands), shipped.name))

    from qwen_tts import Qwen3TTSModel
    import soundfile as sf
    import whisper

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map=args.device,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    wm = whisper.load_model("small.en", device=args.device)

    truth = HERE / "rerank_truth.wav"
    transcripts = {}

    def transcript(p):
        if p not in transcripts:
            transcripts[p] = (wm.transcribe(str(p), language="en", temperature=0.0,
                                           condition_on_previous_text=False)
                              .get("text") or "").strip()
        return transcripts[p]

    def sim_for(ref_path, key):
        prompt = model.create_voice_clone_prompt(
            ref_audio=str(ref_path), ref_text=transcript(ref_path),
            x_vector_only_mode=False)
        wavs, sr = model.generate_voice_clone(
            text=texts[key], language="English",
            voice_clone_prompt=prompt if isinstance(prompt, list) else [prompt])
        tmp = HERE / "_rr_out.wav"
        sf.write(str(tmp), wavs[0], sr)
        s = SV.speaker_similarity(truth, tmp)
        tmp.unlink(missing_ok=True)
        return s or 0.0

    # ---- step 1: choose on the probe sentences --------------------------
    print()
    print("  STEP 1  scoring {} candidates on {} probe sentences".format(
        len(cands) + 1, len(probe_keys)))
    # the shipped reference competes in the selection, so "is it already best?"
    # is answerable rather than assumed
    pool = [(c[0].name, c[0], shipped) for c in cands]
    table = {}
    for name, path, _ in pool:
        sims = [sim_for(path, k) for k in probe_keys]
        table[name] = sims
        print("     {:<20} mean SIM {:.4f}   {}".format(
            name, sum(sims) / len(sims), ["{:.3f}".format(s) for s in sims]))

    winner_name = max(table, key=lambda k: sum(table[k]) / len(table[k]))
    winner_path = dict((n, pth) for n, pth, _ in pool)[winner_name]
    print()
    print("  winner on the probe: {}".format(winner_name))
    if winner_name == shipped.name:
        print("  the SHIPPED reference already wins - nothing to change")

    # ---- step 2: validate on holdout -------------------------------------
    if winner_name != shipped.name:
        print()
        print("  STEP 2  validating on {} sentences neither saw during selection".format(
            len(holdout_keys)))
        w = [sim_for(winner_path, k) for k in holdout_keys]
        b = [sim_for(shipped, k) for k in holdout_keys]
        print("     {:<20} mean SIM {:.4f}".format(winner_name, sum(w) / len(w)))
        print("     {:<20} mean SIM {:.4f}".format(shipped.name, sum(b) / len(b)))
        deltas = [x - y for x, y in zip(w, b)]
        m, s = st.mean(deltas), st.stdev(deltas)
        t = m / (s / len(deltas) ** 0.5)
        print()
        print("     paired delta {:+.4f}  sd {:.4f}  t={:+.2f}  (|t|>2.2 ~ p<0.05)".format(
            m, s, t))
        print("     per-sentence: {}".format(["{:+.3f}".format(d) for d in deltas]))
        print("     wins {}/{}".format(sum(1 for d in deltas if d > 0), len(deltas)))
        print()
        if m > 0 and abs(t) > 2.2:
            print("     VERDICT: {} is a real improvement. Ship it:".format(winner_name))
            print("       copy {} over ref33.wav, then rebuild ref.json's text".format(
                winner_path))
            print("       from THAT clip, and restart the server.")
        elif m > 0:
            print("     VERDICT: positive but inside noise (t={:+.2f}).".format(t))
            print("     Not proven. The spread is mostly per-sentence luck, so")
            print("     picking a global winner from it is not justified yet.")
        else:
            print("     VERDICT: no better than the shipped reference.")
    print()
    print("  NOTE: generation is stochastic - rerank.py measured spread 0.0323 on")
    print("  one run and 0.0233 on a repeat of the same probe. Any single-number")
    print("  comparison at this scale needs the paired test above, not the means.")
    return 0


if __name__ == "__main__":
    sys.exit(main())