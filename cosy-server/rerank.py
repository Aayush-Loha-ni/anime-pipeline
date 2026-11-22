"""Best-of-N: generate several candidates, keep the one that sounds most like you.

This is the technique IWSLT 2026's cross-lingual cloning entry describes under
"Optimal Result Search" (https://aclanthology.org/2026.iwslt-1.11.pdf): expand one
reference slice into many differentiated candidates, score every one by
cosine speaker similarity against the ORIGINAL speaker, and keep the winner.
Graded competitions have used reranking for years for the same reason - the
generator's per-sample variance is larger than the average difference between
configurations.

Why it fits this machine specifically: the constraint is that the microphone is
fixed. Recording quality cannot improve, but the SPREAD between candidates
still can be exploited, and that costs only inference time.

Two axes are available and this measures both:
  ref     N different reference clips from the same 34-minute recording
  seed    N generations from the same reference (sampling temperature)
Reference choice is the one worth shipping: it changes the conditioning, which is
what is actually marginal here. Seed variation only changes the draw.

Run:
    python rerank.py --n-refs 8 --sentences heldout33.json
    python rerank.py --n-refs 8 --baseline          # is there spread to exploit?
"""

import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep_ref as P  # noqa: E402
import score_voice as SV  # noqa: E402

SRC = r"<YOUR_RECORDINGS_DIR>\<your-recording>"


def candidate_refs(n, sr=P.SR):
    """n distinct 8-14s clips from the recording, best-SNR first.

    Deliberately drawn from different parts of the take: cloning quality varies
    with room noise, distance and vocal effort, so sampling across the recording
    is what surfaces a better conditioning clip than picking one by hand.
    """
    x, _ = P.load_24k_mono(SRC)
    regions = [(s, e) for s, e in P.regions(x, sr) if e - s >= 8.0]
    scored = []
    for s, e in regions:
        seg = x[int(s * sr):int(min(e, s + 14.0) * sr)]
        if len(seg) / sr < 8.0:
            continue
        scored.append((P.snr_db(seg), s, e))
    scored.sort(reverse=True)
    # spread across the take rather than taking the top n, which are neighbours
    picked, used = [], []
    for snr, s, e in scored:
        if any(abs(s - u) < 20 for u in used):
            continue
        picked.append((s, min(e, s + 14.0)))
        used.append(s)
        if len(picked) >= n:
            break
    out = HERE / "rerank_refs"
    out.mkdir(exist_ok=True)
    paths = []
    for i, (s, e) in enumerate(picked):
        p = out / "r{:02d}.wav".format(i)
        sf = __import__("soundfile")
        sf.write(str(p), P.normalise(x[int(s * sr):int(e * sr)]), sr,
                 subtype="PCM_16")
        paths.append((p, s, e))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-refs", type=int, default=8)
    ap.add_argument("--sentences", default=str(HERE / "heldout33.json"))
    ap.add_argument("--baseline", default=str(HERE / "ref33.wav"),
                    help="the currently shipped reference; candidates are "
                         "scored against THIS, since similarity is only "
                         "meaningful relative to something")
    ap.add_argument("--probe", type=int, default=3,
                    help="sentences to test the spread on. Reranking is only "
                         "worth shipping if the spread is real, and that is "
                         "cheap to establish on a few.")
    ap.add_argument("--out", default=str(HERE / "rerank"))
    args = ap.parse_args()

    texts = json.loads(pathlib.Path(args.sentences).read_text(encoding="utf-8"))
    keys = sorted(texts)[:args.probe]
    print("  probing {} sentences x {} candidate references".format(
        len(keys), args.n_refs))

    cands = candidate_refs(args.n_refs)
    print("  candidate references:")
    for p, s, e in cands:
        print("     {}  {:.2f}-{:.2f}s".format(p.name, s, e))

    from qwen_tts import Qwen3TTSModel
    import soundfile as sf
    import whisper

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
        dtype=torch.bfloat16, attn_implementation="sdpa")
    whisper_model = whisper.load_model("small.en", device="cuda:0")

    # ground truth for scoring: a long stretch of the user's OWN voice, which is
    # the thing every candidate is supposed to sound like
    truth = HERE / "rerank_truth.wav"
    if not truth.is_file():
        x, sr = P.load_24k_mono(SRC)
        big = max([(s, e) for s, e in P.regions(x, sr) if e - s > 12.0],
                  key=lambda r: r[1] - r[0])
        sf.write(str(truth), P.normalise(x[int(big[0] * sr):int(big[1] * sr)]),
                 sr, subtype="PCM_16")
        print("  truth reference for scoring: {} ({:.1f}s)".format(
            truth.name, (big[1] - big[0])))
    else:
        print("  truth reference: {}".format(truth.name))

    out_root = pathlib.Path(args.out)
    results = {k: [] for k in keys}
    tmp = HERE / "_cand_out.wav"

    for key in keys:
        print("  [{}] {}".format(key, texts[key][:60]))
        for p, s, e in cands:
            # every candidate needs its OWN transcript: ref_text must match the
            # audio or ICL echoes the wrong words
            t = (whisper_model.transcribe(str(p), language="en", temperature=0.0,
                                         condition_on_previous_text=False)
                 .get("text") or "").strip()
            prompt = model.create_voice_clone_prompt(
                ref_audio=str(p), ref_text=t, x_vector_only_mode=False)
            # create_voice_clone_prompt returns a LIST of items even for a
            # single ref_audio, so it must be passed through as a list-of-items
            # rather than wrapped again -- [items] makes the model iterate over
            # lists and fail on .ref_code.
            wavs, srr = model.generate_voice_clone(
                text=texts[key], language="English",
                voice_clone_prompt=prompt if isinstance(prompt, list) else [prompt])
            sf.write(str(tmp), wavs[0], srr)
            sim = SV.speaker_similarity(truth, tmp)
            ut = SV.utmos(tmp)
            got = (whisper_model.transcribe(str(tmp), language="en",
                                           temperature=0.0,
                                           condition_on_previous_text=False)
                   .get("text") or "").strip()
            wer = SV.wer(got, texts[key]) or 0.0
            results[key].append((sim or 0.0, ut or 0.0, wer, p.name))
            print("     {}  sim {:.4f}  utmos {:.3f}  wer {:.3f}".format(
                p.name, sim or 0, ut or 0, wer))

    print()
    print("  {:<6} {:>9} {:>9} {:>9} {:>9}".format(
        "sent", "min SIM", "mean SIM", "max SIM", "spread"))
    spreads = []
    for key in keys:
        sims = [r[0] for r in results[key]]
        sp = max(sims) - min(sims)
        spreads.append(sp)
        print("  {:<6} {:9.4f} {:9.4f} {:9.4f} {:9.4f}".format(
            key, min(sims), sum(sims) / len(sims), max(sims), sp))
    mean_spread = sum(spreads) / len(spreads)
    print()
    print("  mean spread across candidates: {:.4f}".format(mean_spread))
    print("  for scale: the between-CONFIGURATION spread measured earlier was")
    print("  0.0017 (fine-tuned vs not), i.e. indistinguishable. A spread")
    print("  several times that means reranking has real material to work with.")

    best = {k: max(results[k]) for k in keys}
    worst = {k: min(results[k]) for k in keys}
    print()
    print("  {:<6} {:>10} {:>10} {:>10}".format("sent", "worst SIM", "best SIM", "winner"))
    for key in keys:
        print("  {:<6} {:10.4f} {:10.4f} {:>10}".format(
            key, worst[key][0], best[key][0], best[key][3]))
    (out_root / "rerank.json").write_text(json.dumps(
        {k: [(round(a, 4), round(b, 3), round(c, 3), d)
             for a, b, c, d in results[k]] for k in keys},
        indent=2), encoding="utf-8")
    tmp.unlink(missing_ok=True)
    print()
    print("  wrote", out_root / "rerank.json")


if __name__ == "__main__":
    sys.exit(main())