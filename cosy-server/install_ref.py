"""Install a chosen reference: copy the clip and rewrite ref.json together.

The wav and the transcript MUST move as a pair. That is the bug that put a pop
lyric on every generated line: ref.json's text described a different clip than
ref.json's wav, and ICL mode conditions on both together, so the model continued
the wrong utterance. Writing one without the other reintroduces it.

So this does not take "new wav, old text". It transcribes the NEW clip itself and
overwrites both fields, then asserts they agree.

Usage:
    python install_ref.py rerank_refs/r01.wav
    python install_ref.py rerank_refs/r01.wav --speaker narrator
"""

import argparse
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
REF = HERE / "ref.json"


def transcribe(p, device="cuda:0"):
    import whisper
    m = whisper.load_model("small.en", device=device)
    return (m.transcribe(str(p), language="en", temperature=0.0,
                         condition_on_previous_text=False).get("text") or "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip")
    ap.add_argument("--name", default="ref.wav",
                    help="filename to install as inside cosy-server")
    ap.add_argument("--speaker", default=None,
                    help="custom_voice speaker_name; keeps ref.json's "
                         "finetuned_speaker in sync when serving a checkpoint")
    ap.add_argument("--keep-mode", action="store_true",
                    help="do not touch x_vector_only (default: leave as-is too)")
    args = ap.parse_args()

    src = pathlib.Path(args.clip)
    if not src.is_absolute():
        src = HERE / src
    if not src.is_file():
        raise SystemExit("no such clip: " + str(src))

    cfg = json.loads(REF.read_text(encoding="utf-8"))
    before = {k: cfg.get(k) for k in ("wav", "text", "x_vector_only", "language")}

    dest = HERE / args.name
    shutil.copy(str(src), str(dest))

    # transcribe the clip we just installed, not the one we came from
    text = transcribe(dest)
    if len(text.split()) < 4:
        raise SystemExit("transcript too short to be usable: {!r}".format(text))

    cfg["wav"] = str(dest.resolve())
    cfg["text"] = text
    if args.speaker:
        cfg["finetuned_speaker"] = args.speaker
    REF.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="")

    after = json.loads(REF.read_text(encoding="utf-8"))
    print("  installed: {}".format(dest))
    print("    wav  {}  ->  {}".format(
        pathlib.Path(before["wav"] or "-").name, pathlib.Path(after["wav"]).name))
    print("    text       ->  {}".format(after["text"][:96]))
    print("    mode       {}  (unchanged)".format(
        "x_vector_only" if after["x_vector_only"] else "ICL"))

    # the invariant that keeps the lyric bug dead: the transcript must be OF this
    # file. Re-transcribe and compare, so a stale field cannot pass silently.
    check = transcribe(dest)
    same = " ".join(check.lower().split()) == " ".join(text.lower().split())
    print("    re-transcribe matches: {}".format(same))
    assert same, "ref.json text does not match the installed audio"

    # and every built dataset row must now point at the new clip
    for d in sorted(HERE.glob("train_data*")):
        raw = d / "train_raw.jsonl"
        if not raw.is_file():
            continue
        rows = [json.loads(l) for l in
                raw.read_text(encoding="utf-8").splitlines() if l.strip()]
        stale = sum(1 for r in rows
                    if pathlib.Path(r["ref_audio"]).resolve() != dest.resolve())
        print("    {:<18} {} rows, {} still point elsewhere{}".format(
            d.name, len(rows), stale,
            "  <- rebuild with build_sft.py if you retrain" if stale else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())