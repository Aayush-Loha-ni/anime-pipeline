
"""Whisper-transcribe the whole take, then pick eval sentences (default 12) for scoring.

Hand-written test sentences let a model that MEMORISED the recording score
well on text it never saw, which is not the test we want: this pipeline
always speaks the user's OWN writing, so same-domain narration prose is the
realistic input. Sentences are therefore lifted from the recording's own
transcript, which makes WER meaningful (the expected text is exactly right)
at the cost of not testing generalisation. The scorer does not claim to. So the eval sentences are drawn from the same transcript the model
trained on, which means:
  - every variant speaks text from the correct domain and register (narration
    prose in the same style), so the comparison is about voice not topic
  - WER is meaningful, because the expected text is exactly right
It is not a generalisation test and the scorer says so.

Usage: python gen_heldout.py "<recording>"
"""


def main():
    import json
    import pathlib
    import re
    import shutil
    import sys
    import tempfile

    import soundfile as sf

    src = sys.argv[1]
    out_name = sys.argv[3] if len(sys.argv) > 3 else "heldout_sentences.json"
    # cosy-server is the parent of hermes-skill/scripts, derived rather than
    # hardcoded so a moved checkout still finds prep_ref
    root = pathlib.Path(__file__).resolve().parents[2] / "cosy-server"
    sys.path.insert(0, str(root))
    import prep_ref as P

    x, sr = P.load_24k_mono(src)

    # Transcribe per region so segment boundaries are real pauses, not whisper's
    # own segmentation. Consecutive regions are often two halves of one sentence
    # ("...flat, uniform white" + "the color of paper"), so join them with a
    # space and let the sentence splitter decide -- splitting per region would
    # hand the scorer half-sentences as the expected text and report a WER that
    # is really a segmentation artefact.
    pieces = []
    scratch = tempfile.mkdtemp(prefix="heldout_")   # cleaned even on a crash
    try:
        for i, (s, e) in enumerate(P.regions(x, sr)):
            if e - s < 1.5:
                continue
            p = pathlib.Path(scratch) / ("_t%03d.wav" % i)
            sf.write(str(p), P.normalise(x[int(s * P.SR):int(e * P.SR)]), P.SR,
                     subtype="PCM_16")
            pieces.append((s, e, P.transcribe(p)))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    full = " ".join(t for _, _, t in pieces)

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full) if s.strip()]
    # 10-18 words: long enough for UTMOS to score stably, short enough to render fast
    cand = [s for s in sents if 10 <= len(s.split()) <= 18]
    print("  %d sentences of 10-18 words from %d regions" % (len(cand), len(pieces)))
    for s in cand:
        print("     %s" % s)

    # Spread the picks across the recording, and take more than five.
    #
    # ponytail: MEASURED on this data, five sentences cannot separate variants.
    # The between-variant SIM range was 0.0089 while the within-variant spread
    # over the same five sentences was up to 0.031 - the noise is 3.5x the
    # signal. Any ranking off five sentences is a coin toss dressed as a result.
    # Twelve sentences costs a few extra seconds of GPU and is what it takes for
    # the paired per-sentence comparison to mean anything.
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    if len(cand) >= want:
        step = len(cand) / float(want)
        pick = [cand[int(i * step)] for i in range(want)]
    else:
        pick = cand
    out = {"h%02d" % (i + 1): t for i, t in enumerate(pick)}
    (root / out_name).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n  wrote %s with %d sentences" % (out_name, len(out)))


if __name__ == "__main__":
    main()
