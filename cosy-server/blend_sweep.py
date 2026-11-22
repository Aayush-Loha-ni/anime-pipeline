"""Fill the gaps without giving away the voice: sweep both blend axes.

WHY BOTH AXES. The first blend kept the user's speaker embedding and swapped in
narrator CODES, and similarity still fell 0.125 (SIM 0.9735 -> 0.8336). That is
the measurement that matters: if the timbre vector had been carrying the
identity, keeping it would have preserved likeness. It did not. So the CODES -
the narrated delivery itself - carry most of the identity, and blending only the
embedding swaps rather than fills.

That gives two genuinely different things to try:

  A  narrator CODES + user's embedding, weight-blended toward the user
     -> borrows the narrator's PERFORMANCE. Big delivery gain, big likeness cost.
     This is the earlier "speaker" blend.

  B  user's CODES + embedding blended toward the narrator
     -> borrows only TIMBRE-adjacent polish. The user's own delivery is intact,
        so this is the axis that can fill gaps instead of trading them.

Axis B is the one that could actually deliver what was asked for. Both are swept,
because guessing wrong costs a training run to find out.

THE KNEE, NOT THE MAXIMUM. Every point is scored on likeness (SIM), naturalness
(UTMOS) and intelligibility (WER), all paired per-sentence against the shipped
baseline. The output is the frontier and the recommended weight, chosen as the
most naturalness gained per unit of likeness lost - not the point that maximises
either column, because both extremes are worse answers than the middle.

Run:
    python blend_sweep.py --n 6
    python blend_sweep.py --axis B --n 6

RESULT, 2026-10-04. Two axes swept (user weight 1.0 -> 0.5, 6 sentences, means):

  axis A  narrator codes + user timbre   dSIM -0.013  dUTMOS +0.208  looks great
          w=0.8                          dSIM -0.014  dUTMOS +0.245  looks better
  axis B  user codes + blended timbre   dSIM -0.008  dUTMOS -0.274  no gain
          w=0.5                          dSIM -0.020  dUTMOS +0.875  big gain, real cost

THEN IT FAILED THE PROPER TEST. Re-run at n=12 the SAME axis A gave
dSIM -0.079 / dUTMOS +0.177 - nothing like the n=6 numbers. Root cause: the
narrator reference is SYNTHESISED, and CustomVoice sampling is stochastic, so each
sweep blended against a different donor and compared its deltas to a baseline
measured against a different one. The apparent knee was an artefact.

With the narrator pinned to one clip and deltas computed paired per sentence
within a run (blend_pick.py), NO weight passes:

  w=0.9   dSIM -0.069  dUTMOS +0.038   neither significant
  w=0.8   dSIM -0.060  dUTMOS +0.068   neither significant
  w=0.7   dSIM -0.079  dUTMOS +0.110   neither significant

So the honest conclusion is that there is no free lunch on this axis: a blend that
buys meaningful naturalness costs meaningful likeness, and the earlier
"good trade" verdicts were measuring the wrong thing.

WHY AXIS B EXISTS. The first blend kept the user's speaker embedding and swapped
in narrator CODES, and similarity still fell 0.125. That is the evidence that the
codes - the narrated delivery - carry most of the identity, not the timbre vector.
Axis B (keep the user's codes, blend only timbre) was the attempt to borrow
delivery-adjacent polish without touching delivery; it also failed, though less
expensively.

Kept because the negative result is the useful part: it took a paired test on
pinned inputs to see it, and blend_pick.py will do that test for any future
narrator the user nominates.
"""

import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_voice as SV  # noqa: E402

NARRATOR_TEXT = ("The city that wrote his fate had no logic for a canon, "
                 "and no reason for one.")


def blend(a, b, w):
    """Convex mix. a is the anchor (weight w), b the donor. Both 1-D."""
    return (a.float() * w + b.float() * (1.0 - w)).to(a.dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", default="both", choices=("A", "B", "both"))
    ap.add_argument("--n", type=int, default=6,
                    help="sentences to evaluate; more is better but this is "
                         "GPU time (one generation per sentence per weight)")
    ap.add_argument("--sentences", default=str(HERE / "heldout33.json"))
    ap.add_argument("--weights", default="1.0,0.9,0.8,0.65,0.5",
                    help="user weight; 1.0 = pure user voice")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
    import soundfile as sf
    import whisper

    texts = json.loads(pathlib.Path(args.sentences).read_text(encoding="utf-8"))
    keys = sorted(texts)[:args.n]
    weights = [float(w) for w in args.weights.split(",")]
    print("  axis={}  {} sentences  weights={}".format(args.axis, len(keys), weights))

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

    print("  user reference : {} ({})".format(user_wav.name, tx(user_wav)[:52]))
    user_item = base.create_voice_clone_prompt(
        ref_audio=str(user_wav), ref_text=ref_cfg["text"], x_vector_only_mode=False)
    if isinstance(user_item, list):
        user_item = user_item[0]

    # narrator reference: the model's own CustomVoice, since no specific
    # narrator was nominated. Its TIMBRE mostly does not survive - what is
    # borrowed is its delivery, which is the point.
    narr_wav = HERE / "_sweep_narrator.wav"
    if not narr_wav.is_file():
        cv = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map=args.device,
            dtype=torch.bfloat16, attn_implementation="sdpa")
        wavs, sr = cv.generate_custom_voice(
            text=NARRATOR_TEXT, language="English", speaker="Ryan",
            instruct="Measured, cinematic narration. Restrained, never announcer.")
        sf.write(str(narr_wav), wavs[0], sr)
        del cv
        torch.cuda.empty_cache()
        print("  narrator ref   : synthesised {} ({:.1f}s)".format(
            narr_wav.name, len(wavs[0]) / sr))
    else:
        print("  narrator ref   : reusing {}".format(narr_wav.name))
    narr_item = base.create_voice_clone_prompt(
        ref_audio=str(narr_wav), ref_text=tx(narr_wav), x_vector_only_mode=False)
    if isinstance(narr_item, list):
        narr_item = narr_item[0]

    # ground truth for likeness: a long stretch of the user's own voice
    truth = HERE / "rerank_truth.wav"

    axes = []
    if args.axis in ("A", "both"):
        axes.append(("A  narrator CODES  + user timbre", "codes"))
    if args.axis in ("B", "both"):
        axes.append(("B  user CODES      + blended timbre", "timb"))

    tmp = HERE / "_sweep_out.wav"

    def score(prompt, key):
        wavs, sr = base.generate_voice_clone(
            text=texts[key], language="English", voice_clone_prompt=[prompt])
        sf.write(str(tmp), wavs[0], sr)
        sim = SV.speaker_similarity(truth, tmp) or 0.0
        ut = SV.utmos(tmp) or 0.0
        got = tx(tmp)
        return sim, ut, (SV.wer(got, texts[key]) or 0.0)

    results = {}
    for label, kind in axes:
        print()
        print("  === axis {} ===".format(label))
        rows = {}
        for w in weights:
            if kind == "codes":
                item = VoiceClonePromptItem(
                    ref_code=narr_item.ref_code,
                    ref_spk_embedding=blend(user_item.ref_spk_embedding,
                                           narr_item.ref_spk_embedding, w),
                    x_vector_only_mode=False, icl_mode=True,
                    ref_text=NARRATOR_TEXT)
            else:
                item = VoiceClonePromptItem(
                    ref_code=user_item.ref_code,
                    ref_spk_embedding=blend(user_item.ref_spk_embedding,
                                           narr_item.ref_spk_embedding, w),
                    x_vector_only_mode=False, icl_mode=True,
                    ref_text=ref_cfg["text"])
            sc = [score(item, k) for k in keys]
            rows[w] = sc
            sim = sum(s for s, _u, _e in sc) / len(sc)
            ut = sum(u for _s, u, _e in sc) / len(sc)
            wer = sum(e for _s, _u, e in sc) / len(sc)
            print("     w={:<5} SIM {:.4f}  UTMOS {:.3f}  WER {:.3f}".format(
                w, sim, ut, wer))
            if w == 1.0:
                # Anchor every delta to this run's own baseline. Comparing
                # against a baseline from a PREVIOUS run is what produced a
                # contradiction here: w=1.0 read SIM 0.7524 in one sweep and
                # 0.8167 in the next on the same config, because the narrator
                # reference had been re-synthesised between them and generation
                # is stochastic. Within-run only.
                print("       (baseline anchor for this run: SIM {:.4f} UTMOS {:.3f})".format(
                    sim, ut))
        results[kind] = rows

    tmp.unlink(missing_ok=True)

    # ---- the frontier, and a recommendation -----------------------------
    print()
    print("  " + "=" * 74)
    print("  {} sentences per point. Baseline = w=1.0, which IS the shipped voice.".format(len(keys)))
    print("  {:<4} {:>10} {:>10} {:>10} {:>14} {:>12}".format(
        "axis", "dSIM", "dUTMOS", "dWER", "gain per loss", "verdict"))
    print("  " + "-" * 74)
    for label, kind in axes:
        rows = results[kind]
        b = rows[1.0]
        b_sim = sum(s for s, _u, _e in b) / len(b)
        b_ut = sum(u for _s, u, _e in b) / len(b)
        b_wer = sum(e for _s, _u, e in b) / len(b)
        for w in weights:
            sc = rows[w]
            sim = sum(s for s, _u, _e in sc) / len(sc)
            ut = sum(u for _s, u, _e in sc) / len(sc)
            wer = sum(e for _s, _u, e in sc) / len(sc)
            ds, du = sim - b_sim, ut - b_ut
            ratio = (du / -ds) if ds < -0.002 else float("inf")
            # a blend is only worth taking if likeness barely moves and the
            # naturalness gain is bigger than the noise floor (~0.1 UTMOS)
            if abs(ds) < 0.004 and du > 0.10:
                verdict = "FREE GAIN"
            elif du > 0.10 and ds > -0.02:
                verdict = "good trade"
            elif du > 0.10:
                verdict = "costs too much"
            else:
                verdict = "not worth it"
            print("  {:<4} {:>+10.4f} {:>+10.3f} {:>+10.3f} {:>14} {:>12}".format(
                label[0], ds, du, wer - b_wer,
                ("{:+.2f}".format(ratio) if ratio != float("inf") else "n/a"),
                verdict))

    print()
    print("  noise floor from the earlier rounds: UTMOS varies ~0.10-0.30 "
          "between\n  repeats of the same config, and SIM ~0.012. Treat smaller "
          "gains as nothing.")


if __name__ == "__main__":
    sys.exit(main())