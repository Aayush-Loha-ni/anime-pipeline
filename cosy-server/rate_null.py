"""Is the rate change measurable at all, given how noisy generation is?

The first A/B compared two INDEPENDENT generations of the same text, one per arm.
Before concluding anything about the treatment, the null must be measured: the
same configuration, generated twice. If flat-vs-flat already swings UTMOS by half
a point, then no treatment effect smaller than that is knowable, and the honest
answer is "not measurable", not "slightly negative".

So three arms, all on the same sentences:

  flatA  flat, run 1
  flatB  flat, run 2      -> NULL: same config, different sample
  fast   rate applied     -> TREATMENT

flatA vs flatB is the noise floor. The treatment only counts if it clears it.

Also fixes the eval set. heldout33's h01 expects "Blurr, here you..." - that is
Whisper mis-transcribing a shout in the RECORDING, not a sentence. Scoring a
model against a broken label measures the label, so such sentences are excluded
here and reported separately rather than silently inflating WER.

Run:
    python rate_null.py --n 6
"""

import argparse
import json
import pathlib
import re
import statistics as st
import sys

import numpy as np
import soundfile as sf
import torch

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent / "hermes-skill" / "scripts"
sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(HERE))

import server as S  # noqa: E402
import score_voice as SV  # noqa: E402



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--rate", type=float, default=1.06)
    ap.add_argument("--sentences", default=str(HERE / "heldout33.json"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    texts = json.loads(pathlib.Path(args.sentences).read_text(encoding="utf-8"))

    def is_broken(t):
        # a label the transcriber could not parse: mid-word punctuation, or a
        # stray ellipsis standing in for shouting
        return bool(re.search(r"[a-z]\.\.\.|\b[A-Z][a-z]{5,}[,.]\s*$", t))

    keys = [k for k in sorted(texts) if not is_broken(texts[k])][:args.n]
    dropped = [k for k in sorted(texts) if is_broken(texts[k])]
    print("  {} usable sentences; dropped {} as untranscribable labels: {}".format(
        len(keys), len(dropped), dropped))
    for k in keys:
        print("     {}: {}".format(k, texts[k][:66]))

    ref_cfg = json.loads((HERE / "ref.json").read_text(encoding="utf-8"))
    from qwen_tts import Qwen3TTSModel
    import whisper
    wm = whisper.load_model("small.en", device=args.device)
    m = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map=args.device,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    rt = (wm.transcribe(ref_cfg["wav"], language="en", temperature=0.0,
                        condition_on_previous_text=False).get("text") or "").strip()
    prompt = m.create_voice_clone_prompt(
        ref_audio=ref_cfg["wav"], ref_text=rt, x_vector_only_mode=False)
    if not isinstance(prompt, list):
        prompt = [prompt]

    truth = HERE / "rerank_truth.wav"
    tmp = HERE / "_rn.wav"
    rows = {k: {} for k in keys}

    for key in keys:
        arms = {}
        for tag, rate in (("flatA", 1.0), ("flatB", 1.0), ("fast", args.rate)):
            wavs, sr = m.generate_voice_clone(
                text=texts[key], language="English", voice_clone_prompt=prompt)
            w = np.asarray(wavs[0], dtype=np.float32)
            if abs(rate - 1.0) > 1e-3:
                w = S._atempo(w, sr, rate)
            sf.write(str(tmp), w, sr)
            got = (wm.transcribe(str(tmp), language="en", temperature=0.0,
                                 condition_on_previous_text=False)
                   .get("text") or "").strip()
            arms[tag] = {"sim": SV.speaker_similarity(truth, tmp) or 0.0,
                         "utmos": SV.utmos(tmp) or 0.0,
                         "wer": SV.wer(got, texts[key]) or 0.0,
                         "dur": len(w) / sr}
        rows[key] = arms
        print("  [{}] flatA {:.3f} | flatB {:.3f} | fast{:.2f} {:.3f}   "
              "wer {:.2f}/{:.2f}/{:.2f}".format(
                  key, arms["flatA"]["utmos"], arms["flatB"]["utmos"],
                  args.rate, arms["fast"]["utmos"],
                  arms["flatA"]["wer"], arms["flatB"]["wer"],
                  arms["fast"]["wer"]))
    tmp.unlink(missing_ok=True)

    def deltas(x, y, metric):
        return [rows[k][x][metric] - rows[k][y][metric] for k in keys]

    def tstat(d):
        if len(d) < 2:
            return 0.0
        sd = st.stdev(d)
        return 0.0 if sd == 0 else st.mean(d) / (sd / len(d) ** 0.5)

    null_u = deltas("flatB", "flatA", "utmos")
    null_s = deltas("flatB", "flatA", "sim")
    null_w = deltas("flatB", "flatA", "wer")
    treat_u = deltas("fast", "flatA", "utmos")
    treat_s = deltas("fast", "flatA", "sim")
    treat_w = deltas("fast", "flatA", "wer")

    print()
    print("  NULL   same config, two samples   (this is pure noise)")
    print("     dUTMOS {:+.3f}   dSIM {:+.4f}   dWER {:+.4f}".format(
        st.mean(null_u), st.mean(null_s), st.mean(null_w)))
    print()
    print("  TREATMENT rate={:.2f} vs flat".format(args.rate))
    print("     dUTMOS {:+.3f}   dSIM {:+.4f}   dWER {:+.4f}".format(
        st.mean(treat_u), st.mean(treat_s), st.mean(treat_w)))
    print()
    noise_u = abs(st.mean(null_u))
    treat_abs = abs(st.mean(treat_u))
    print("  null |dUTMOS| = {:.3f}".format(noise_u))
    print("  treat |dUTMOS| = {:.3f}".format(treat_abs))
    # Guard against reading a verdict off a handful of sentences. The effect
    # being hunted is ~0.07 UTMOS against a null of the same size, so n must be
    # large enough for the null to be estimated at all -- otherwise a lucky pair
    # reports a "win" that is pure sampling. Measured: at n=2 the null came out
    # 0.007 and the treatment 0.099, which reads as a large effect and means
    # nothing.
    _MIN_N = 6
    if len(keys) < _MIN_N:
        print("  VERDICT: only {} sentences - too few to estimate a {} null. "
              "Re-run with --n >= {}.".format(len(keys), 0.07, _MIN_N))
        return 0
    if treat_abs < noise_u:
        print("  VERDICT: the rate change is NOT distinguishable from re-rolling")
        print("  the same config. Do not ship it on the strength of this test.")
    elif treat_abs < noise_u * 1.5:
        print("  VERDICT: treatment ({:.3f}) sits inside 1.5x the null ({:.3f}).".format(
            treat_abs, noise_u))
        print("  Treat as UNPROVEN - not a win, and not a demonstrated harm.")
    else:
        print("  VERDICT: treatment clears the noise floor ({:.3f} > {:.3f})".format(
            treat_abs, noise_u))
    if st.mean(treat_w) > 0.03:
        print("  WARNING: intelligibility regressed by {:+.3f} WER".format(
            st.mean(treat_w)))
    return 0


if __name__ == "__main__":
    sys.exit(main())