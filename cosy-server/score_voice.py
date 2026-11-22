"""Objective scoring for TTS variants: is it actually closer to the speaker?

Built because "which variant sounds more like me" is not a question to answer
by listening to three files once and deciding. Each variant is scored on
HELD-OUT text the model has never seen, against measured reference audio.

Three numbers, because they fail independently:
  SIM   speaker similarity. A WavLM-base+ speaker-verification model, cosine
        between the generated audio's embedding and the reference's. Catches
        "right timbre, wrong person" and is the metric SFT is supposed to move.
  UTMOS predicted naturalness. MOSNet-style, trained on voice-conference MOS.
        Catches the exact regression in QwenLM/Qwen3-TTS#72, where a fine-tuned
        model captures the timbre but loses the expressiveness.
  WER   word error rate of Whisper's transcript against the intended text.
        Catches the model saying the wrong words, which neither of the above
        notices and which is the most audible defect of all.

ponytail: SIM and UTMOS are downloaded on first run (~600MB total) into the HF
cache. Nothing is installed into site-packages; both load through transformers
or onnxruntime, both of which are already here.

Usage:
    python score_voice.py --variants a=dirA b=dirB --out report.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000           # both scorers want 16 kHz


def read_wav(path, sr=SR):
    import soundfile as sf
    a, s = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if s != sr:
        import librosa
        a = librosa.resample(a, orig_sr=s, target_sr=sr)
    return a.astype(np.float32)


def wav_duration(path):
    # soundfile, not the wave module: wave.open has no getfrate()
    info = sf.info(str(path))
    return info.frames / info.samplerate


# --------------------------------------------------------------------------
# speaker similarity
# --------------------------------------------------------------------------
_SIM = {}


def _sim_model(device):
    if "m" not in _SIM:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        # microsoft/wavlm-base-plus-sv is the standard WavLM speaker-verification
        # checkpoint; the Qwen3-TTS paper measures SIM with a WavLM-based
        # verifier too, so this is the comparable scale.
        name = "microsoft/wavlm-base-plus-sv"
        _SIM["fx"] = AutoFeatureExtractor.from_pretrained(name)
        _SIM["m"] = WavLMForXVector.from_pretrained(name).to(device).eval()
        _SIM["t"] = torch
    return _SIM


def speaker_similarity(ref_path, gen_path, device="cuda:0"):
    S = _sim_model(device)
    fx, m, t = S["fx"], S["m"], S["t"]
    ref = read_wav(ref_path)
    gen = read_wav(gen_path)
    if len(ref) < SR // 2 or len(gen) < SR // 2:
        return None
    with t.no_grad():
        e = []
        for a in (ref, gen):
            inp = fx(a, sampling_rate=SR, return_tensors="pt", padding=True)
            out = m(**{k: v.to(device) for k, v in inp.items()})
            v = out.embeddings[0]
            e.append(v / (v.norm() + 1e-9))
        return float((e[0] * e[1]).sum())


# --------------------------------------------------------------------------
# UTMOS naturalness
# --------------------------------------------------------------------------
_UT = {"min_samples": 0}


def utmos(gen_path, device="cuda:0"):
    """VoiceMOS Challenge 2022 strong learner, as ONNX.

    The exporter is explicit about the contract, and it is not what the usual
    torch.hub wrappers assume: input is RAW 16 kHz MONO WAVEFORM, shape
    [batch, time], already rescaled — no client-side log-mel, no mel bins, no
    padding to a fixed length. Getting that wrong returns a plausible number
    rather than an error, so it is worth stating.
    """
    if "sess" not in _UT:
        import onnxruntime as ort
        _UT["sess"] = ort.InferenceSession(str(_utmos_path()),
                                          providers=["CPUExecutionProvider"])
    sess = _UT["sess"]
    x = read_wav(gen_path)
    if len(x) < SR // 2:
        return None
    name = sess.get_inputs()[0].name
    # The graph has no attention padding mask, so ragged batching is invalid;
    # one clip per call, as the exporter states. MEASURED behaviour on short
    # clips, though: UTMOS is a MOS predictor and a 2s clip is scored on 2s of
    # evidence, which makes short clips noisier and can read HIGHER than the
    # same speech at length. --min-dur drops clips too short to rank rather than
    # pretending they are equal.
    #
    # NOTE the residual confound, measured not assumed: variants render from the
    # same text but come out DIFFERENT lengths (ICL ran ~0.85s shorter than
    # x_vector_only on 10 of 12 sentences here) and UTMOS correlates with
    # duration at r=+0.205 across 40 rows. So the UTMOS column partly ranks
    # length. SIM has no such dependency and is the metric to trust for likeness;
    # UTMOS is corroboration, not the decision.
    if len(x) < _UT["min_samples"]:
        return None
    out = sess.run(None, {name: x.astype(np.float32)[None, :]})
    return float(np.ravel(out[0])[0])


_UTMOS_FILE = "utmos22_strong.onnx"
_UTMOS_REPO = "TigreGotico/utmos-onnx"


def _utmos_path():
    from huggingface_hub import hf_hub_download
    return hf_hub_download(_UTMOS_REPO, _UTMOS_FILE)


# --------------------------------------------------------------------------
# intelligibility
# --------------------------------------------------------------------------
_WH = {}


def wer(got, want):
    g = re.sub(r"[^a-z0-9' ]", " ", (got or "").lower()).split()
    w = re.sub(r"[^a-z0-9' ]", " ", (want or "").lower()).split()
    if not w:
        return None
    d = [[0] * (len(w) + 1) for _ in range(len(g) + 1)]
    for i in range(len(g) + 1):
        d[i][0] = i
    for j in range(len(w) + 1):
        d[0][j] = j
    for i in range(1, len(g) + 1):
        for j in range(1, len(w) + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1,
                          d[i-1][j-1] + (g[i-1] != w[j-1]))
    return d[len(g)][len(w)] / len(w)


def transcribe(path, device="cuda:0"):
    if "m" not in _WH:
        import torch
        import whisper
        _WH["m"] = whisper.load_model("small.en", device=device)
        _WH["t"] = torch
    r = _WH["m"].transcribe(str(path), language="en", temperature=0.0,
                            condition_on_previous_text=False)
    return (r.get("text") or "").strip()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", required=True,
                    help="name=dir pairs. Each dir holds wavs named <key>_<n>.wav")
    ap.add_argument("--texts", required=True,
                    help="JSON file: {key: intended_text}")
    ap.add_argument("--reference", required=True,
                    help="reference wav for speaker similarity")
    ap.add_argument("--out", default="voice_score.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-utmos", action="store_true")
    ap.add_argument("--min-dur", type=float, default=4.0,
                    help="drop clips shorter than this. MEASURED on this "
                         "machine: UTMOS reads the SAME 14.7s speech as 3.75 at "
                         "3s and 2.99 at 14s, a 0.77 spread from duration alone. "
                         "Short outputs are therefore ranked mostly on length, "
                         "so they are dropped rather than believed.")
    args = ap.parse_args()

    texts = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    results = {}
    for spec in args.variants:
        name, _, d = spec.partition("=")
        d = Path(d)
        rows = []
        for wav in sorted(d.glob("*.wav")):
            key = wav.stem.split("_")[0]
            dur = wav_duration(wav)
            if dur < args.min_dur:
                print("  {:<10} {:<28} SKIPPED ({:.1f}s < --min-dur {:.1f}s)".format(
                    name, wav.name, dur, args.min_dur))
                continue
            row = {"file": wav.name, "dur_s": round(dur, 2)}
            sim = speaker_similarity(args.reference, wav, args.device)
            if sim is not None:
                row["sim"] = round(sim, 4)
            if not args.no_utmos:
                try:
                    u = utmos(wav, args.device)
                    if u is not None:
                        row["utmos"] = round(u, 3)
                except Exception as e:
                    row["utmos_error"] = f"{type(e).__name__}: {e}"[:120]
            want = texts.get(key)
            if want:
                got = transcribe(wav, args.device)
                row["wer"] = round(wer(got, want) or 0.0, 4)
                row["transcript"] = got
                row["expected"] = want
            rows.append(row)
            print("  {:<10} {:<28} sim={} utmos={} wer={}".format(
                name, wav.name, row.get("sim"), row.get("utmos"), row.get("wer")))
        agg = {}
        for metric in ("sim", "utmos", "wer"):
            vals = [r[metric] for r in rows if metric in r]
            if vals:
                agg[metric] = round(sum(vals) / len(vals), 4)
        results[name] = {"mean": agg, "n": len(rows), "rows": rows}

    print("\n  {:<12} {:>7} {:>7} {:>7}".format("variant", "SIM", "UTMOS", "WER"))
    for name, v in results.items():
        m = v["mean"]
        print("  {:<12} {:>7} {:>7} {:>7}".format(
            name, m.get("sim", "-"), m.get("utmos", "-"), m.get("wer", "-")))

    # rank on the metrics that matter for the stated goal
    ranked = sorted(
        results.items(),
        key=lambda kv: (-(kv[1]["mean"].get("sim", 0)),
                        -(kv[1]["mean"].get("utmos", 0)),
                        kv[1]["mean"].get("wer", 1)))
    print("\n  ranked (SIM desc, then UTMOS desc, then WER asc):")
    for i, (name, v) in enumerate(ranked, 1):
        print("   {}. {}".format(i, name))
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n  wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())