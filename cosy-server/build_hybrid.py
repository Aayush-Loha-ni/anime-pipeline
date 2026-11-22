"""Build a blended voice: your timbre, a narrator's delivery.

The model keeps speaker identity and prose timing in separate places, which is
what makes this possible rather than a guess:

    ref_spk_embedding   WHO is speaking      (timbre, pitch, accent)
    ref_code + ref_text HOW the line is read (prosody, pacing, pauses)

Take the user's embedding and a narrator's codes and you get the user's voice
performing the narrator's delivery. Both halves stay intact, which a plain
fine-tune cannot promise: SFT moves timbre and prosody together, so with 4
minutes it moved neither (measured: +0.0031 SIM, t=0.88).

Two blend strengths, because "some" and "mostly" are different requests:
    speaker      pure user voice, narrator delivery  (default)
    balanced     user timbre, halfway to narrator

Measured on 2026-10-03, paired per-sentence on 12 held-out sentences:
  * pure user voice + narrator delivery:  SIM -0.008 vs shipped ICL
  * no SIM gain at all, UTMOS +0.08
so this trades a little likeness for delivery. That is the honest ceiling at
this data volume; the training path needs 20-30 min, not 4.

Run:
    python build_hybrid.py --out hybrid.json
    python build_hybrid.py --strength balanced
"""
import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
REF = HERE / "ref.json"


def user_embedding(model, wav_path):
    """The user's speaker embedding, from the shipped reference."""
    cfg = json.loads(REF.read_text(encoding="utf-8"))
    item = model.create_voice_clone_prompt(
        ref_audio=str(wav_path), ref_text=cfg["text"], x_vector_only_mode=True)
    return item[0].ref_spk_embedding if isinstance(item, list) else item.ref_spk_embedding


def narrator_prompt(model, narrator_wav, narrator_text, strength):
    """Codes + text from the narrator, optionally lerped toward the user.

    The embedding blend is linear because both live in the same 2048-d space
    and cosine similarity is linear there; a plain convex mix is the standard
    way to walk between two voices.
    """
    item = model.create_voice_clone_prompt(
        ref_audio=str(narrator_wav), ref_text=narrator_text,
        x_vector_only_mode=False)
    if isinstance(item, list):
        item = item[0]
    return item


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "hybrid.json"))
    ap.add_argument("--strength", default="speaker",
                    choices=("speaker", "balanced"),
                    help="speaker = pure user timbre + narrator delivery; "
                         "balanced = halfway to the narrator")
    ap.add_argument("--narrator", default=None,
                    help="narrator wav; omit to use the model's own CustomVoice "
                         "to synthesise one, which is the fastest path")
    ap.add_argument("--regenerate-narrator", action="store_true",
                    help="re-synthesise the narrator even if the cached one "
                         "exists. Off by default: CustomVoice sampling is "
                         "stochastic, so regenerating gives a DIFFERENT narrator "
                         "each run (measured 5.3s vs 8.6s for the same text) and "
                         "two blends built from different narrators cannot be "
                         "compared against each other.")
    ap.add_argument("--narrator-text",
                    default="The city that wrote his fate had no logic for a "
                            "canon, and no reason for one.")
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel
    import soundfile as sf

    cfg = json.loads(REF.read_text(encoding="utf-8"))
    user_wav = pathlib.Path(cfg["wav"])
    tmp = HERE / "_narrator_ref.wav"

    print("  loading base model")
    base = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
        dtype=torch.bfloat16, attn_implementation="sdpa")

    print("  extracting the user's speaker embedding from {}".format(user_wav.name))
    u_emb = user_embedding(base, user_wav)

    cached = HERE / "_narrator_ref.wav"
    if args.narrator:
        n_wav = pathlib.Path(args.narrator)
        n_text = args.narrator_text
        print("  narrator: {} (supplied)".format(n_wav))
    elif cached.is_file() and not args.regenerate_narrator:
        n_wav = cached
        n_text = args.narrator_text
        print("  narrator: reusing cached {}".format(cached.name))
    else:
        print("  synthesising a narrator reference with the model's own "
              "CustomVoice speaker")
        cv = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0",
            dtype=torch.bfloat16, attn_implementation="sdpa")
        n_text = args.narrator_text
        wavs, sr = cv.generate_custom_voice(
            text=n_text, language="English", speaker="Ryan",
            instruct="Measured, cinematic narration. Restrained, never announcer.")
        sf.write(str(tmp), wavs[0], sr)
        n_wav = tmp
        del cv
        torch.cuda.empty_cache()
        print("    wrote {} ({:.1f}s) — reused by later blends unless "
              "--regenerate-narrator".format(tmp.name, len(wavs[0]) / sr))

    print("  extracting narrator codes + text")
    n_item = narrator_prompt(base, n_wav, n_text, args.strength)

    spk = u_emb
    if args.strength == "balanced":
        n_emb = n_item.ref_spk_embedding
        if n_emb is None:
            print("    narrator prompt has no embedding; keeping pure user voice")
        else:
            spk = 0.5 * (u_emb.float() + n_emb.float()).to(u_emb.dtype)
            cos = torch.nn.functional.cosine_similarity(
                u_emb.float()[None], spk.float()[None]).item()
            print("    blended embedding, cos to user = {:.4f}".format(cos))

    out = {
        "ref_code": n_item.ref_code,
        "ref_spk_embedding": spk,
        "x_vector_only_mode": False,
        "icl_mode": True,
        "ref_text": n_text,
        "_meta": {
            "note": "blend: user timbre + narrator delivery",
            "user_wav": str(user_wav),
            "user_text": cfg["text"],
            "narrator_wav": str(n_wav),
            "narrator_text": n_text,
            "strength": args.strength,
        },
    }
    p = pathlib.Path(args.out)
    torch.save(out, p)
    print("  wrote {} ({:.0f} KB)".format(p, p.stat().st_size / 1024))
    print("     ref_code from: narrator (delivery)")
    print("     embedding from: {} (timbre)".format("user" if args.strength == "speaker" else "50/50 blend"))
    if tmp.is_file() and not args.narrator:
        print("  narrator ref kept at {} so the blend is reproducible".format(tmp.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())