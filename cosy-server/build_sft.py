"""Build the SFT dataset from recordings, with a held-out split.

This is the ONE dataset builder. It supersedes an earlier make_dataset.py, which
was deleted: that one had no held-out split, no SNR gate, and wrote its own
clips\ref.wav without touching ref.json, so running it recreated exactly the
drift described below. train_voice.bat calls this file.

What it adds over simply transcribing everything:
  * a deterministic held-out fraction, kept OUT of training, so a variant is
    scored on audio it never saw
  * whole-region splits (not random 20ms frames), so a val clip is a real
    utterance
  * per-utterance SNR gating, because one bad take poisons a 3-minute run
    harder than it would a 3-hour one
  * the reference clip is re-picked AND ref.json's wav+text are rewritten in the
    same run, so the dataset and the served config cannot disagree

ponytail: --holdout 0.15 of ~50 regions is ~7 utterances / ~28s. Small, but the
point is that it is audio the model never fit, not that it is statistically
powered.

NOTE on honesty: the eval SENTENCES for a round come from this val split, so
they are text the model never trained on. Vocabulary still overlaps the training
rows because it is all one continuous script - a strict zero-overlap filter over
<your-recording>'s 48 val rows left exactly ONE usable sentence - so WER is a
weak signal and SIM is the one to trust.

MEASURED, twice, do not re-litigate without new data (full numbers in
SFT_FINDINGS.md):
  * Recording (32), 178s train: SFT +0.0031 SIM (t=+0.88) - no gain
  * <your-recording>, 1188s train: SFT -0.0007 SIM (t=-0.21) - no gain, at two
    learning rates, with training loss falling. 6x the data did not move it, so
    volume is not the binding constraint; recording QUALITY is. 16 kHz / 37 kbps
    phone AAC with Whisper small.en transcripts teaches the transcriber's errors
    along with the voice.
  * What works: a better REFERENCE. Both rounds improved the voice by improving
    the 8-second clip, never by training.

Run:
    python build_sft.py --out-dir train_data "rec1.m4a" "rec2.wav"
"""
import argparse
import json
import pathlib
import random
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from prep_ref import SR, load_24k_mono, normalise, regions, snr_db, transcribe  # noqa: E402

HERE = pathlib.Path(__file__).parent
MIN_WORDS = 4
MAX_WORDS = 60


def chop(sig, sr=SR, max_s=12.0):
    """Split a long stretch at its quietest internal point, recursively."""
    if len(sig) / sr <= max_s:
        return [sig]
    hop = int(0.4 * sr)
    n = len(sig) // hop * hop
    if n == 0:
        return [sig]
    env = np.array([np.sqrt((sig[i:i + hop] ** 2).mean() + 1e-12)
                    for i in range(0, n, hop)])
    mid = len(env) // 2
    lo, hi = max(1, mid - len(env) // 4), min(len(env) - 1, mid + len(env) // 4)
    if hi <= lo:
        return [sig]
    cut = lo + int(np.argmin(env[lo:hi]))
    cut = max(1, min(len(sig) - 1, cut * hop))
    return chop(sig[:cut], sr, max_s) + chop(sig[cut:], sr, max_s)


def pick_reference(x, sr=SR):
    """Choose the single shared ref_audio, and return (clip, start, end).

    Deliberately separate from main() so build_sft.py can rebuild the reference
    and the dataset together and the two cannot drift apart. They drifted once:
    ref.json pointed at the old song clip while train_raw.jsonl pointed at
    ref32.wav, and every ICL generation came out prefixed with "It was meant to
    be" - four words of a One Direction lyric, on every single line.

    Selection, in order of what actually matters for a clone reference:
      1. starts at a real pause, so the model is not conditioned on a fragment
         with no opening cadence
      2. 8-20s, per the Qwen3-TTS guidance that longer usually helps
      3. high SNR
      4. a full sentence: capitalised first letter, terminal punctuation
    """
    cands = []
    for s, e in [(s, e) for s, e in regions(x, sr) if e - s > 6.0]:
        # internal pauses >= 0.12s are candidate sentence starts
        hop = 480
        lo, hi = int(s * sr), int(e * sr)
        env = np.array([np.sqrt((x[i:i + hop] ** 2).mean() + 1e-12)
                        for i in range(lo, hi, hop)])
        quiet = env < 0.02
        idx = [i for i, v in enumerate(quiet) if v]
        runs = []
        for i in idx:
            if runs and i - runs[-1][-1] <= 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        starts = [s] + [lo + (r[-1] + 1) * hop for r in runs
                        if len(r) * hop / sr >= 0.12]
        for t0 in starts:
            seg = x[int(t0 * sr):int(e * sr)]
            dur = len(seg) / sr
            if not (8.0 <= dur <= 20.0):
                continue
            cands.append((snr_db(seg), min(dur, 20.0) / 20.0, t0, e, seg))
    if not cands:
        raise SystemExit("no region 8-20s long; record a longer take")
    snr, _, s, e, seg = max(cands)
    return normalise(seg), s, e


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-dir", default=str(HERE / "train_data"))
    # Default to the SHIPPED reference, read from ref.json. It used to hardcode
    # ref32.wav, which is two references out of date: a fresh build silently
    # wrote ref32.wav into every training row while ref.json (and therefore the
    # server) used something else, and the voice-quality tests then skipped
    # because no dataset matched. One source of truth.
    ap.add_argument("--ref", default=None,
                    help="shared ref_audio every row points at. Defaults to "
                         "ref.json's wav, so the dataset cannot drift from what "
                         "the server is actually serving.")
    ap.add_argument("--keep-ref", action="store_true",
                    help="use --ref as given instead of re-picking it. Implied "
                         "when --ref is omitted.")
    ap.add_argument("--reref", action="store_true",
                    help="re-pick the reference from the first input even when "
                         "--ref was omitted. Discards the measured pick.")
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--max-clip", type=float, default=12.0)
    ap.add_argument("--min-snr", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    clips = out_dir / "clips"
    if not args.dry_run:
        clips.mkdir(parents=True, exist_ok=True)

    # ---- enumerate candidate utterances across every input -----------------
    cands = []
    for p in args.inputs:
        x, sr = load_24k_mono(p)
        for cs, ce in [(s, e) for s, e in regions(x, sr) if e - s > 0.6]:
            for i, piece in enumerate(chop(x[int(cs * SR):int(ce * SR)],
                                           max_s=args.max_clip)):
                dur = len(piece) / SR
                if dur < 0.9:
                    continue
                if snr_db(piece) < args.min_snr:
                    continue
                cands.append({"src": pathlib.Path(p).stem, "t0": cs,
                              "idx": i, "piece": piece, "dur": dur,
                              "snr": snr_db(piece)})
    print("  {} candidate utterances from {} file(s)".format(
        len(cands), len(args.inputs)))
    if not cands:
        print("  nothing survived the SNR gate")
        return 1

    # ---- transcribe every candidate, then split ----------------------------
    for n, c in enumerate(cands, 1):
        name = "{}_{:06d}_{}.wav".format(c["src"], int(c["t0"] * 100), c["idx"])
        c["name"] = name
        c["path"] = clips / name
        if not args.dry_run:
            sf.write(str(c["path"]), normalise(c["piece"]), SR, subtype="PCM_16")
            c["text"] = transcribe(c["path"])
        else:
            c["text"] = transcribe(_tmp(c["piece"]))
        if n % 10 == 0:
            print("   transcribed {}/{}".format(n, len(cands)))

    keep = [c for c in cands
            if MIN_WORDS <= len(c["text"].split()) <= MAX_WORDS]
    print("  {} kept, {} dropped on transcript length".format(
        len(keep), len(cands) - len(keep)))
    if not keep:
        print("  nothing trainable")
        return 1

    # ---- (re)build the reference, unless told to keep the existing one -----
    # No --ref means "use the shipped one". Overwriting a reference that
    # pick_ref.py selected by measurement (WavLM similarity + holdout rerank)
    # would trade a vetted clone for an arbitrary clip off the new recording,
    # so the default is to KEEP it and only rebuild when asked with --reref.
    if args.ref:
        ref_path = pathlib.Path(args.ref)
    else:
        ref_path = pathlib.Path(json.loads(
            (HERE / "ref.json").read_text(encoding="utf-8"))["wav"])
        args.keep_ref = not args.reref
        print("  using the shipped reference {} (pass --reref to re-pick)".format(
            ref_path.name))
    if not args.keep_ref:
        rx, _ = load_24k_mono(args.inputs[0])
        clip, rs, re_ = pick_reference(rx)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(ref_path), clip, SR, subtype="PCM_16")
        print("  reference rebuilt: {:.2f}-{:.2f}s ({:.1f}s) SNR {:.1f} dB -> {}".format(
            rs, re_, len(clip) / SR, snr_db(clip), ref_path.name))
        # ref.json's 'text' MUST be a transcript of THIS clip. ICL mode
        # conditions on the reference audio and its text together, so a stale
        # text makes the model continue the wrong utterance - which is how a
        # One Direction lyric ended up prefixed to every generated line.
        if not args.dry_run:
            import json as _json
            rj = pathlib.Path(HERE / "ref.json")
            cfg = _json.loads(rj.read_text(encoding="utf-8")) if rj.is_file() else {}
            cfg["wav"] = str(ref_path.resolve())
            cfg["text"] = transcribe(ref_path)
            rj.write_text(_json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8", newline="")
            print("  ref.json synced: wav + machine transcript of that clip")

    # ---- sync the dataset's ref_audio to whatever ref.json now names -------
    # Order matters: do this AFTER any rebuild, and unconditionally, because the
    # two files drifting apart is the lyric-leak bug. Every row gets the shipped
    # reference so training conditions on the voice that will actually be served.
    ref = str(ref_path.resolve())
    raw_p = out_dir / "train_raw.jsonl"
    if not args.dry_run and raw_p.is_file():
        rows = [json.loads(l) for l in
                raw_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        stale = [r for r in rows if r.get("ref_audio") != ref]
        if stale:
            for r in rows:
                r["ref_audio"] = ref
            with raw_p.open("w", encoding="utf-8", newline="") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print("  retargeted {} stale ref_audio rows -> {}".format(
                len(stale), ref_path.name))

    rng = random.Random(args.seed)
    order = keep[:]
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * args.holdout)))
    val, tr = order[:n_val], order[n_val:]
    print("  split: {} train / {} held-out".format(len(tr), len(val)))

    ref = str(ref_path.resolve())
    tr_secs = sum(c["dur"] for c in tr)
    val_secs = sum(c["dur"] for c in val)
    print("  train {:.1f}s   held-out {:.1f}s   mean SNR {:.1f} dB".format(
        tr_secs, val_secs, sum(c["snr"] for c in tr) / max(len(tr), 1)))
    print("  {}".format("OK for SFT" if tr_secs >= 240 else
                        "under 4 min - expect a modest gain only"))

    if args.dry_run:
        print("\n(dry run)")
        return 0

    for label, group, fn in (("train", tr, "train_raw.jsonl"),
                             ("val", val, "val.jsonl")):
        rows = [{"audio": str(c["path"].resolve()), "text": c["text"],
                 "ref_audio": ref, "dur_s": round(c["dur"], 2),
                 "snr": round(c["snr"], 1)} for c in group]
        p = out_dir / fn
        with p.open("w", encoding="utf-8", newline="") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("  wrote {} ({} rows)".format(p.name, len(rows)))

    print("\n  val utterances for scoring:")
    for c in val:
        print("     {}  {:.1f}s  {}".format(c["name"], c["dur"], c["text"][:60]))
    return 0


def _tmp(piece):
    import tempfile
    p = pathlib.Path(tempfile.gettempdir()) / "sft_probe.wav"
    sf.write(str(p), piece, SR, subtype="PCM_16")
    return p


if __name__ == "__main__":
    sys.exit(main())