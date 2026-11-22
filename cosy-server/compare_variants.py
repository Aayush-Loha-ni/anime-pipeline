"""Render the same held-out sentences through every variant, for scoring.

The comparison only means anything if every variant speaks the SAME text, so
each variant is handed the identical sentence list and the output filename is
the sentence key. score_voice.py then reads the key back off the filename.

Variants, all served through the real cosy-server code path where possible:

  zero_shot_xvec   base model, x_vector_only_mode=True     (today's behaviour)
  zero_shot_icl    base model, x_vector_only_mode=False    (best benchmark mode)
  A_upstream       SFT as upstream ships it: custom_voice + speaker embedding
                   only. Included as a CONTROL, because QwenLM/Qwen3-TTS#72
                   reports this path degrades likeness versus zero-shot.
  B_icl_preserved  SFT that keeps tts_model_type="base" and the speaker encoder,
                   then served through generate_voice_clone with ICL. The
                   issue's own prescribed workaround.

Usage:
    python compare_variants.py --out variants --sentences heldout.json
"""
import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

DEFAULT_SENTENCES = [
    "The sky above the city was flat and uniform white.",
    "He had no memory of the transition from one moment to the next.",
    "Light existed, but it had no source at all.",
    "She found the tea house by instinct, which she declined to examine.",
    "Overcast implies clouds, movement, and the suggestion of weather painting.",
]


def sentences_from(src):
    """Accepts either a {key: text} JSON file or a val.jsonl of training rows.

    Two formats because the sentences come from two places: a hand-written
    held-out set, or the val split build_sft.py already carved out of the
    recording. Guessing wrong here means scoring against the wrong text and
    reporting a confident WER for the wrong sentence.
    """
    p = pathlib.Path(src)
    text = p.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except ValueError:
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
        obj = {pathlib.Path(r["audio"]).stem: r["text"] for r in rows}
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items() if len(str(v).split()) >= 4}
    raise SystemExit("{} is neither a {{key: text}} map nor val.jsonl".format(p))


def render(model_path, texts, out_dir, mode, speaker=None, device="cuda:0",
           ref_json=None):
    """mode: 'icl' | 'xvec' | 'custom'"""
    from qwen_tts import Qwen3TTSModel
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(
        str(model_path), device_map=device,
        dtype=torch.bfloat16, attn_implementation="sdpa")

    prompt = None
    if mode in ("icl", "xvec"):
        # the reference is passed in, not re-read per call: a variant trained
        # against one ref must be served with the SAME ref, and silently picking
        # up whatever ref.json happens to say is how that gets broken
        if ref_json is None:
            raise SystemExit("icl/xvec rendering needs --ref-json")
        ref = json.loads(pathlib.Path(ref_json).read_text(encoding="utf-8"))
        wav = pathlib.Path(ref["wav"])
        text = ref["text"]
        prompt = model.create_voice_clone_prompt(
            ref_audio=str(wav), ref_text=text, x_vector_only_mode=(mode == "xvec"))
        print("  prompt built (mode={})".format(mode))
    elif mode == "custom" and not speaker:
        raise SystemExit("custom mode needs --speaker")

    made = []
    for key, text in texts.items():
        if mode == "custom":
            wavs, sr = model.generate_custom_voice(
                text=text, language="English", speaker=speaker)
        else:
            wavs, sr = model.generate_voice_clone(
                text=text, language="English", voice_clone_prompt=prompt)
        p = out_dir / (key + ".wav")
        sf.write(str(p), wavs[0], sr)
        made.append(p)
        print("     {:<34} {:5.2f}s".format(key + ".wav", len(wavs[0]) / sr))
    del model
    torch.cuda.empty_cache()
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "variants"))
    ap.add_argument("--ckpt-root", default=None,
                    help="where the fine-tuned checkpoints live, if not under --out")
    ap.add_argument("--sentences", default=str(HERE / "heldout_sentences.json"),
                    help="JSON {key: text}. Defaults to 5 held-out sentences.")
    ap.add_argument("--speaker", default="narrator")
    ap.add_argument("--ref-json", default=str(HERE / "ref.json"),
                    help="the ref.json whose wav+text built the prompt. "
                         "Must be the one the variant was trained with.")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--variants", nargs="*", default=[
        "zero_shot_xvec", "zero_shot_icl", "A_upstream", "B_icl_preserved"])
    args = ap.parse_args()

    texts = (sentences_from(args.sentences) if args.sentences
             else {"s{:02d}".format(i + 1): t for i, t in enumerate(DEFAULT_SENTENCES)})
    out_root = pathlib.Path(args.out)
    # checkpoints live under variants33/ for this round, not variants/
    ckpt_root = pathlib.Path(args.ckpt_root) if args.ckpt_root else out_root
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sentences.json").write_text(
        json.dumps(texts, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  {} sentences: {}\n".format(len(texts), list(texts)))

    base = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    jobs = {
        "zero_shot_xvec": (base, "xvec", None),
        "zero_shot_icl": (base, "icl", None),
        "A_upstream": (ckpt_root / "A_upstream" / "checkpoint-epoch-1", "custom", args.speaker),
        "B_icl_preserved": (ckpt_root / "B_icl_preserved" / "checkpoint-epoch-1", "icl", None),
        "B_lr1e-4": (ckpt_root / "B_lr1e-4e2" / "checkpoint-epoch-1", "icl", None),
    }
    # two variants have no checkpoint and are served straight from the hub id
    for name in args.variants:
        if name in args.skip:
            print("  -- skip {} (requested)".format(name))
            continue
        path, mode, speaker = jobs[name]
        needs_checkpoint = name not in ("zero_shot_xvec", "zero_shot_icl")
        # any variant other than the two zero-shot ones needs a checkpoint on
        # disk. The old guard was `mode not in (xvec, icl)`, which is only true
        # for "custom", so B_icl_preserved was never checked and fell through
        # to an OSError inside the try instead of being reported as absent.
        if needs_checkpoint and not pathlib.Path(path).is_dir():
            print("  -- skip {}: no checkpoint at {}".format(name, path))
            continue
        print("  [{}] mode={} model={}".format(name, mode, path))
        try:
            render(path, texts, out_root / name, mode, speaker,
                   ref_json=args.ref_json)
        except Exception as e:
            print("     FAILED: {}: {}".format(type(e).__name__, str(e)[:200]))
        print()
    print("  outputs under", out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())