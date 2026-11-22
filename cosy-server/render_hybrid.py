"""Render a variant through a BLENDED prompt: user timbre + narrator delivery.

compare_variants.py only knows x_vector_only / icl / custom. The blend needs a
fourth path: load a saved VoiceClonePromptItem straight from hybrid.json and
hand it to generate_voice_clone, so the narration codes and the speaker
embedding come from different sources.

Usage:
    python render_hybrid.py --prompts hybrid.json --out variants/hybrid_speaker
"""

import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", nargs="+", required=True,
                    help="name=hybrid.json pairs")
    ap.add_argument("--out", default=str(HERE / "variants"))
    ap.add_argument("--sentences", default=str(HERE / "heldout_sentences.json"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
    import soundfile as sf

    texts = json.loads(pathlib.Path(args.sentences).read_text(encoding="utf-8"))
    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    print("  {} sentences: {}".format(len(texts), list(texts)))

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map=args.device,
        dtype=torch.bfloat16, attn_implementation="sdpa")

    for spec in args.prompts:
        name, _, path = spec.partition("=")
        blob = torch.load(path, weights_only=False)
        item = VoiceClonePromptItem(
            ref_code=blob["ref_code"],
            ref_spk_embedding=blob["ref_spk_embedding"],
            x_vector_only_mode=blob["x_vector_only_mode"],
            icl_mode=blob["icl_mode"],
            ref_text=blob["ref_text"])
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        meta = blob.get("_meta", {})
        print("  [{}] strength={} timbre={} delivery={}".format(
            name, meta.get("strength", "?"),
            "user" if meta.get("strength") == "speaker" else "blend",
            pathlib.Path(meta.get("narrator_wav", "?")).name))
        for key, text in texts.items():
            wavs, sr = model.generate_voice_clone(
                text=text, language="English", voice_clone_prompt=[item])
            p = d / (key + ".wav")
            sf.write(str(p), wavs[0], sr)
            print("     {:<12} {:5.2f}s".format(key + ".wav", len(wavs[0]) / sr))
    print("  done")
    return 0


if __name__ == "__main__":
    sys.exit(main())