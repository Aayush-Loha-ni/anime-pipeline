"""A/B the narration direction: does varying pace and beat actually help?

Paired per sentence, same reference, same model, same text. The ONLY difference
between the two arms is what narration_direction.py asks for:

  flat  every segment at speed 1.0, every join the same 0.12 s — the old
        behaviour, and the metronome
  directed  per-line rate from the text, per-join gap from what the line did,
        one payoff word capitalised

Scored on likeness (must not drop), naturalness (the target), intelligibility
(must not drop) and DURATION SPREAD, because "sounds less mechanical" has a
measurable correlate: if every line is the same length, the rhythm is flat. A
direction system that raises UTMOS while leaving per-scene duration variance at
zero has not actually fixed the metronome.

Run:
    python direct_ab.py --n 8
    python direct_ab.py --n 12 --validate
"""

import argparse
import json
import pathlib
import statistics as st
import sys

import numpy as np
import soundfile as sf
import torch

HERE = pathlib.Path(__file__).resolve().parent
# cosy-server and hermes-skill are siblings under "Anime Series", so the pipeline
# package is HERE.parent / "hermes-skill" / "scripts". Going up two levels from
# cosy-server lands in the user's home directory instead.
PIPE = HERE.parent / "hermes-skill" / "scripts"
sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(HERE))

# narration_direction lives in hermes-skill/scripts/pipeline/, which is NOT this
# package. Add that path before importing it or the module is simply absent.
sys.path.insert(0, str(PIPE / "pipeline"))
import narration_direction as ND  # noqa: E402
import score_voice as SV  # noqa: E402


def cat(parts, gaps, dest):
    joined = parts[0]
    sr = 24000
    for i, c in enumerate(parts[1:]):
        g = gaps[i] if i < len(gaps) else 0.12
        joined = np.concatenate([joined, np.zeros(int(g * sr), dtype=np.float32), c])
    sf.write(str(dest), joined, sr, subtype="PCM_16")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--validate", action="store_true",
                    help="run the A/B on sentences NOT used to pick the settings")
    ap.add_argument("--sentences", default=str(HERE / "heldout33.json"))
    ap.add_argument("--out", default=str(HERE / "variants33"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    sys.path.insert(0, str(PIPE))
    import episode_producer as ep

    texts = json.loads(pathlib.Path(args.sentences).read_text(encoding="utf-8"))
    keys = sorted(texts)
    if args.validate:
        keys = keys[args.n:]
    else:
        keys = keys[:args.n]
    print("  {} sentences{}".format(len(keys), " (validation)" if args.validate else ""))

    ref_cfg = json.loads((HERE / "ref.json").read_text(encoding="utf-8"))
    from qwen_tts import Qwen3TTSModel
    import whisper

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map=args.device,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    wm = whisper.load_model("small.en", device=args.device)
    ref_txt = (wm.transcribe(ref_cfg["wav"], language="en", temperature=0.0,
                            condition_on_previous_text=False)
               .get("text") or "").strip()
    prompt = model.create_voice_clone_prompt(
        ref_audio=ref_cfg["wav"], ref_text=ref_txt, x_vector_only_mode=False)
    if not isinstance(prompt, list):
        prompt = [prompt]

    truth = HERE / "rerank_truth.wav"
    out_root = pathlib.Path(args.out) / ("direct_val" if args.validate else "direct")
    tmp = HERE / "_ab_seg.wav"

    rows = {}
    for key in keys:
        segs = ep.split_for_tts(ep.clean_tts_text(texts[key]))
        if not segs:
            continue
        plan = ND.direct_all(segs, base_rate=1.0)

        arms = {}
        for arm in ("flat", "directed"):
            parts, gaps = [], []
            for i, seg in enumerate(segs):
                if arm == "flat":
                    txt, rate = seg, 1.0
                    gap = 0.12
                else:
                    txt, rate = plan[i]["emphasised"], plan[i]["rate"]
                    gap = plan[i]["gap_after"]
                # NOTE: no speed= kwarg. qwen_tts has no speed support - it
                # rides **kwargs into model.generate() and is silently dropped
                # (measured: speed=1.06 vs no kwarg gives a BIT-IDENTICAL wav).
                # The retime is done locally below, the same way the server does.
                wavs, sr = model.generate_voice_clone(
                    text=txt, language="English", voice_clone_prompt=prompt)
                sf.write(str(tmp), wavs[0], sr)
                if arm == "directed":
                    # server applies atempo; replicate locally so the arms are
                    # identical in every respect except the direction
                    import server as S
                    wav = S._atempo(np.asarray(wavs[0], dtype=np.float32), sr, rate)
                else:
                    wav = np.asarray(wavs[0], dtype=np.float32)
                parts.append(wav)
                gaps.append(gap)
            dst = out_root / arm
            dst.mkdir(parents=True, exist_ok=True)
            final = dst / (key + ".wav")
            cat(parts, gaps, final)
            arms[arm] = {
                "sim": SV.speaker_similarity(truth, final) or 0.0,
                "utmos": SV.utmos(final) or 0.0,
                "wer": SV.wer((wm.transcribe(str(final), language="en",
                                              temperature=0.0,
                                              condition_on_previous_text=False)
                                .get("text") or "").strip(), texts[key]) or 0.0,
                "dur": len(parts[0]) / 24000.0 if parts else 0.0,
                "seg_durs": [len(p) / 24000.0 for p in parts],
            }
        rows[key] = arms
        print("  [{}] flat  SIM {:.4f} UTMOS {:.3f} dur {:.2f}s".format(
            key, arms["flat"]["sim"], arms["flat"]["utmos"], arms["flat"]["dur"]))
        print("        {}  SIM {:.4f} UTMOS {:.3f} dur {:.2f}s".format(
            "dir ", arms["directed"]["sim"], arms["directed"]["utmos"],
            arms["directed"]["dur"]))
    tmp.unlink(missing_ok=True)

    def arm_stats(arm):
        s = [rows[k][arm]["sim"] for k in rows]
        u = [rows[k][arm]["utmos"] for k in rows]
        w = [rows[k][arm]["wer"] for k in rows]
        # rhythm variance: how much segment lengths differ inside one scene
        var = []
        for k in rows:
            d = rows[k][arm]["seg_durs"]
            if len(d) > 1:
                var.append(st.stdev(d))
        return s, u, w, var

    fs, fu, fw, fv = arm_stats("flat")
    ds_, du, dw, dv = arm_stats("directed")

    def tstat(deltas):
        if len(deltas) < 2:
            return 0.0
        sd = st.stdev(deltas)
        return 0.0 if sd == 0 else st.mean(deltas) / (sd / len(deltas) ** 0.5)

    print()
    print("  " + "=" * 72)
    print("  n={}".format(len(rows)))
    print("  {:<10} {:>9} {:>9} {:>9} {:>9}".format(
        "arm", "SIM", "UTMOS", "WER", "seg-len sd"))
    print("  {:<10} {:9.4f} {:9.3f} {:9.4f} {:9.4f}".format(
        "flat", sum(fs) / len(fs), sum(fu) / len(fu), sum(fw) / len(fw),
        sum(fv) / max(len(fv), 1)))
    print("  {:<10} {:9.4f} {:9.3f} {:9.4f} {:9.4f}".format(
        "directed", sum(ds_) / len(ds_), sum(du) / len(du), sum(dw) / len(dw),
        sum(dv) / max(len(dv), 1)))
    print("  " + "-" * 72)
    d_sim = [a - b for a, b in zip(ds_, fs)]
    d_ut = [a - b for a, b in zip(du, fu)]
    d_w = [a - b for a, b in zip(dw, fw)]
    print("  paired dSIM   {:+.4f}  t={:+.2f}  {}".format(
        st.mean(d_sim), tstat(d_sim),
        "OK" if st.mean(d_sim) > -0.005 else "REGRESSION"))
    print("  paired dUTMOS {:+.3f}   t={:+.2f}  {}".format(
        st.mean(d_ut), tstat(d_ut),
        "significant" if abs(tstat(d_ut)) > 2.2 else "not significant"))
    print("  paired dWER    {:+.4f}  t={:+.2f}  {}".format(
        st.mean(d_w), tstat(d_w),
        "OK" if st.mean(d_w) < 0.02 else "worse"))
    print("  per-sentence dUTMOS: {}".format(["{:+.2f}".format(x) for x in d_ut]))
    print()
    print("  outputs: {}".format(out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())