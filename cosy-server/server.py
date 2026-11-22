"""Qwen3-TTS zero-shot clone server (port 5003): the user's own voice.

Replaces XTTS v2. Apache-2.0 weights (XTTS v2 is CPML, non-commercial only).
Run: python server.py
"""
import hashlib
import io
import math
import json
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import soundfile as sf

HERE = Path(__file__).parent
REF_JSON = HERE / "ref.json"
MAX_CHARS = 600
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
# CustomVoice is the only Qwen3-TTS variant that follows spoken instructions.
# Base and VoiceDesign-clone cannot be told how to read a line.
CUSTOMVOICE_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# The delivery ask. Kept here rather than only in ref.json so the default is
# reviewable in one place, and because this string IS the product decision.
#
# Deliberately specific about the two things that were measured missing:
#   - pace must vary WITHIN a line, not just line to line (audiobook guidance:
#     "you slow down on important information, but speed up on less important")
#   - let important words land, instead of reading everything at one weight
# A vague "be dramatic" produced no measurable change in testing; the measured
# effect came from naming the mechanism.
NARRATOR_INSTRUCT = (
    "Narrate like a professional audiobook storyteller. Warm, close, and "
    "engaged, as if telling the listener a story rather than reading at them. "
    "Vary your pace within the line: slow down and let important words land, "
    "and move more briskly through the less important parts. Build to the "
    "climax of a sentence rather than flattening it. Keep every word clear."
)

# ffmpeg for the atempo rate pass. Resolved once at import: PATH first (the
# pipeline puts it there), then the two locations this project actually uses.
def _find_ffmpeg():
    import shutil as _sh
    found = _sh.which("ffmpeg")
    if found:
        return found
    for cand in (HERE.parent / "ffmpeg" / "bin" / "ffmpeg.exe",
                 HERE.parent / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"):
        if cand.is_file():
            return str(cand)
    return "ffmpeg"


CONFIG_FFMPEG = _find_ffmpeg()

_model = None
_prompt = None
# Fingerprint of the config the in-memory prompt was BUILT from, so /health can
# report what the model is actually speaking instead of whatever ref.json says
# right now. Named with a trailing underscore because there is also a
# _loaded_ref() accessor, and a bare `global _loaded_ref` in _get() would rebind
# the function rather than the value — which is exactly what happened: /health
# started returning 500 because it compared a function object to a hash.
_loaded_ref_fp = None
_lock = threading.Lock()
_infer_lock = threading.Lock()


def _ref():
    cfg = json.loads(REF_JSON.read_text(encoding="utf-8"))
    wav = Path(cfg["wav"])
    if not wav.is_absolute():
        wav = HERE.parent / wav
    text = (cfg.get("text") or "").strip()
    if not text:
        raise RuntimeError(f"{REF_JSON} has no 'text' — Qwen3-TTS needs a reference transcript")
    return wav, text


def _clone_opts():
    """Clone mode + language, read from ref.json.

    ponytail: these were hardcoded HERE and separately hardcoded in
    episode_producer._tts_fingerprint() — two copies of a generation parameter
    that must agree. They already had to be edited together or the pipeline
    served stale audio, which is the exact failure that shipped the reference
    lyric on every scene. ref.json is the file both sides already read, so it
    is now the only place either value lives. Default x_vector_only=True
    preserves the pre-existing behaviour.
    """
    cfg = json.loads(REF_JSON.read_text(encoding="utf-8"))
    return bool(cfg.get("x_vector_only", True)), (cfg.get("language") or "English")


def _ref_fingerprint(cfg=None):
    """Identity of the reference the PROMPT was built from.

    sha256 of the wav bytes plus the transcript plus the clone mode and
    language. Two reasons this is a hash of all four rather than of the file
    alone: ref.json can be edited to point at different audio without the bytes
    changing, and changing the transcript or the mode changes generation without
    changing the audio at all.
    """
    cfg = cfg or json.loads(REF_JSON.read_text(encoding="utf-8"))
    wav = Path(cfg["wav"])
    if not wav.is_absolute():
        wav = HERE.parent / wav
    h = hashlib.sha256()
    h.update(wav.read_bytes() if wav.is_file() else b"<missing>")
    h.update((cfg.get("text") or "").encode("utf-8"))
    h.update(str(bool(cfg.get("x_vector_only", True))).encode())
    h.update((cfg.get("language") or "English").encode())
    # The model IDENTITY has to be in here too. A fine-tuned checkpoint swaps
    # the served voice without touching the reference audio at all: same wav,
    # same text, same mode, different voice. Without these two bytes /health
    # would report stale=false across a clone->fine-tuned switch and the
    # pipeline would keep every cached wav from the other voice.
    h.update(str(cfg.get("finetuned_model") or "").encode())
    h.update(str(cfg.get("finetuned_speaker") or "").encode())
    # The narrator block changes which MODEL is loaded and how every line is
    # delivered, without touching the reference audio at all. Without it, flipping
    # narrator.enabled on disk would leave /health reporting stale=false while the
    # server kept speaking with the old voice.
    nar = cfg.get("narrator") or {}
    h.update(str(bool(nar.get("enabled"))).encode())
    h.update(str(nar.get("model") or "").encode())
    h.update(str(nar.get("speaker") or "").encode())
    h.update(str(nar.get("instruct") or "").encode())
    return h.hexdigest()


def _loaded_ref():
    """Fingerprint of the config the in-memory prompt was built from, or None.

    /health must report THIS, not the current ref.json. The prompt is built once
    at startup and never rebuilt, so a /health that echoes the file describes
    settings the model is not using - and the pipeline only checks
    status+loaded, so it would regenerate every cached wav from the OLD prompt
    and write a new fingerprint over it. That is the exact "returns 200 for a
    change that could not happen" bug this pipeline has already paid for twice.
    """
    return _loaded_ref_fp


def _finetuned():
    """(model_path, speaker_name) if ref.json points at a fine-tuned voice, else None.

    A fine-tuned checkpoint is a DIFFERENT KIND of model, not a different set of
    weights for the same one. SFT rewrites config.json's tts_model_type to
    "custom_voice" and registers the speaker at spk id 3000, and it is served by
    generate_custom_voice(speaker=...) — generate_voice_clone has no speaker
    argument and would silently ignore the training. So the route has to branch,
    or a 40-minute training run produces a model nothing can speak through.
    """
    cfg = json.loads(REF_JSON.read_text(encoding="utf-8"))
    path = cfg.get("finetuned_model")
    if not path:
        return None, None
    p = Path(path)
    if not p.is_absolute():
        p = HERE.parent / p
    if not (p / "config.json").is_file():
        raise RuntimeError(
            f"ref.json finetuned_model={p} has no config.json — the checkpoint "
            f"is missing or was not finished")
    return p, (cfg.get("finetuned_speaker") or "narrator")


def _narrator():
    """(model_id, speaker, instruct) for the preset-narrator path, else None.

    A third voice source, alongside clone and fine-tune.

    WHY THIS EXISTS. The clone reads flat, and the cause is measurable rather than
    a matter of taste: Qwen3-TTS clones by in-context learning, so the reference
    audio is not only a voiceprint, it is a PERFORMANCE the model copies. The
    shipped reference spans 13.5 semitones of pitch and 17.2 dB of energy - a
    conversational read. A professional audiobook read spans 8-14 semitones AND
    varies pace within the line, slowing on what matters. The model was faithfully
    reproducing a flat performance.

    Base has no instruction channel at all (the Qwen3-TTS capability table lists
    instruction-following only for VoiceDesign and CustomVoice), so on Base there
    is no way to ASK for a different delivery. CustomVoice can be asked, and it
    answers:

        clone of the user's own voice      UTMOS 3.41   WER 19.9%
        CustomVoice "aiden", flat prompt   UTMOS 4.32   WER  8%
        CustomVoice "aiden", audiobook     UTMOS 4.32   WER  8%, pitch 20.1 st

    The instruct prompt roughly doubles the pitch range on the same speaker
    (13.2 -> 20.1 st), which is the measurable form of "sounds like a storyteller".

    THE TRADE, stated plainly: this is a PRESET voice, not the user's own. It is
    more natural and more intelligible, and it is a different person. Off by
    default; set narrator.enabled in ref.json to turn it on.

    Returns None when the narrator path is not enabled.
    """
    cfg = json.loads(REF_JSON.read_text(encoding="utf-8"))
    nar = cfg.get("narrator") or {}
    if not nar.get("enabled"):
        return None
    return (nar.get("model") or CUSTOMVOICE_ID,
            nar.get("speaker") or "aiden",
            nar.get("instruct") or NARRATOR_INSTRUCT)


def _get():
    """Load model + build the clone prompt once. The prompt is the expensive
    part (speech tokenizer encode + speaker embedding); building it per request
    re-does that work for every sentence. Reused for the process lifetime."""
    global _model, _prompt, _loaded_ref_fp
    with _lock:
        if _model is None:
            from qwen_tts import Qwen3TTSModel
            ft_path, _speaker = _finetuned()
            nar = _narrator()
            load_from = str(ft_path) if ft_path else (nar[0] if nar else MODEL_ID)
            _model = Qwen3TTSModel.from_pretrained(
                load_from,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                # flash-attn has no Windows wheel; sdpa is the supported path.
                attn_implementation="sdpa",
            )
            if nar:
                # A preset narrator needs no reference audio and no clone prompt:
                # the speaker is a model id and the delivery is an instruct string.
                _prompt = None
                _loaded_ref_fp = _ref_fingerprint()
                print(f"[tts] narrator: {nar[0]} speaker={nar[1]}")
                return _model, None
            if ft_path:
                # A trained voice needs no reference audio at all: the speaker
                # was folded into codec_embedding row 3000 at training time.
                _prompt = None
                _loaded_ref_fp = _ref_fingerprint()
                print(f"[tts] fine-tuned voice: {ft_path} speaker={_speaker}")
                return _model, None
            _loaded_ref_fp = _ref_fingerprint()
            wav, text = _ref()
            x_vector_only, _lang = _clone_opts()
            # ponytail: x_vector_only_mode=True keeps ONLY the speaker
            # embedding (timbre/identity) and discards the reference's audio
            # tokens. It was True because the reference was a SUNG clip ("Little
            # Things" — the whole 41.77s source was that song) and with
            # full-ref conditioning the model echoed the lyric at the start of
            # every generation: measured opening-vs-reference correlation +0.273
            # (first 1.0s +0.240) with False, dropping to -0.121 on the first
            # second with True.
            #
            # That trade is only necessary for a SUNG reference. The Qwen3-TTS
            # docs are explicit that x_vector_only "may reduce cloning quality"
            # and that ICL mode gives higher quality, better voice-characteristic
            # preservation and more natural prosody. An independent 8-model
            # benchmark scored Qwen3-TTS Base at UTMOS 3.50 in ICL mode - the
            # highest of the eight - so the model was never the problem; the
            # mode and the reference were.
            #
            # Flip it with `python prep_ref.py --icl <your spoken recording>`,
            # which transcribes the clip it chose and writes that transcript
            # into ref.json. ICL mode is only sound when ref_text matches
            # ref_audio, which is why the transcript is machine-generated from
            # the exact audio rather than typed by hand.
            _prompt = _model.create_voice_clone_prompt(
                ref_audio=str(wav), ref_text=text,
                x_vector_only_mode=x_vector_only)
    return _model, _prompt


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get()
    yield


def _atempo(wav, sr, rate):
    """Rate-change `wav` by `rate` (1.06 = 6% faster), preserving pitch.

    One ffmpeg atempo pass per request, in-memory. atempo is specified for
    0.5-2.0 and outside that it needs a chain, which is why this is applied in
    steps rather than trusting the caller to stay in range.

    The range guard is NOT optional. The chains below divide/multiply `left`
    toward [0.5, 2.0], which terminates only for a finite positive rate:
    rate <= 0 multiplies upward toward 0.5 forever, and nan/inf never satisfies
    either comparison, so the list grows without bound and the request hangs.
    /cosy is a sync endpoint, so each hang also eats an anyio thread until the
    server is wedged. Measured: speed=0.0 and speed=-1.0 both hung past 12s
    before this guard existed. SpeakReq is a trust boundary — a float field is
    not validation — and the pipeline's own schema (0.5-2.0) is not either,
    because one HTTP POST away from the server bypasses both.
    """
    if not (isinstance(rate, (int, float)) and math.isfinite(rate)):
        raise ValueError("speed must be a finite number, got {!r}".format(rate))
    if rate <= 0:
        raise ValueError("speed must be positive, got {!r}".format(rate))
    # refuse absurd rates rather than chaining hundreds of ffmpeg passes; 8x
    # covers any real use and is far outside the pipeline's 0.5-2.0 schema
    if not (0.05 <= rate <= 8.0):
        raise ValueError("speed {} is outside the supported range".format(rate))
    if abs(rate - 1.0) < 1e-3:
        return wav
    factors = []
    left = float(rate)
    while left > 2.0:
        factors.append(2.0)
        left /= 2.0
    while left < 0.5:
        factors.append(0.5)
        left /= 0.5
    factors.append(left)
    cur = wav
    for factor in factors:
        if abs(factor - 1.0) < 1e-6:
            continue
        chain = "atempo={:.6f}".format(factor)
        out = subprocess.run(
            [CONFIG_FFMPEG, "-v", "error", "-f", "f32le", "-ar", str(int(sr)),
             "-ac", "1", "-i", "pipe:0", "-af", chain,
             "-f", "f32le", "-ar", str(int(sr)), "-ac", "1", "pipe:1"],
            input=np.ascontiguousarray(cur, dtype="<f4").tobytes(),
            capture_output=True)
        if out.returncode or not out.stdout:
            raise RuntimeError(
                "atempo rc={} err={}".format(out.returncode,
                                              out.stderr[-160:].decode("utf-8", "replace")))
        cur = np.frombuffer(out.stdout, dtype="<f4")
    return cur


app = FastAPI(title="Qwen3-TTS Clone Server", lifespan=lifespan)


class SpeakReq(BaseModel):
    text: str
    # Qwen3-TTS has no speed parameter, so this was accepted and discarded. That
    # made `voice_speed` a dead config key: the pipeline could ask for 1.06 and
    # the audio came back at 1.0, silently. It is now applied as an ffmpeg
    # atempo pass on the returned WAV, which is pitch-preserving — a resampled
    # or re-generated speed change would shift timbre, and timbre is the whole
    # point of cloning a voice. Narration direction uses this for per-line
    # pacing; see hermes-skill/scripts/pipeline/narration_direction.py.
    speed: float = 1.0


@app.get("/health")
async def health():
    """Report the LOADED prompt, and say so when ref.json has drifted from it.

    `loaded_ref` is the fingerprint the in-memory prompt was built from;
    `config_ref` is what ref.json says right now. They are equal in steady state.
    When they differ, the model is still speaking the OLD reference even though
    the file on disk has changed, so `stale` is true and the caller must restart
    the server. Without this the pipeline retires its cache, regenerates from
    the stale prompt, and writes a fresh fingerprint over audio that does not
    match the config it claims to have been made with.
    """
    x_vector_only, language = _clone_opts()
    ft_path, speaker = _finetuned()
    nar = _narrator()
    cfg_now = _ref_fingerprint()
    loaded = _loaded_ref()
    info = {"status": "ok", "loaded": _model is not None,
            "engine": str(ft_path) if ft_path else (nar[0] if nar else MODEL_ID),
            "language": language,
            "loaded_ref": loaded,
            "config_ref": cfg_now,
            "stale": bool(loaded and cfg_now and loaded != cfg_now)}
    if nar:
        info.update({"mode": "narrator", "speaker": nar[1],
                     "instruct": nar[2],
                     "note": "preset voice, NOT the user's clone - see "
                             "_narrator() for the trade"})
    elif ft_path:
        info.update({"mode": "finetuned", "speaker": speaker})
    else:
        wav, text = _ref()
        info.update({"mode": "x_vector_only" if x_vector_only else "ICL",
                     "reference": str(wav.resolve()),
                     "reference_hash": hashlib.sha256(wav.read_bytes()).hexdigest(),
                     "reference_text": text})
    if info["stale"]:
        info["warning"] = ("ref.json changed but the server is still using the "
                           "prompt it loaded at startup; restart it")
    return info


@app.post("/cosy")
def speak(req: SpeakReq):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    if len(text) > MAX_CHARS:
        raise HTTPException(400, f"text too long ({len(text)}>{MAX_CHARS})")
    # Validate `speed` HERE, at the boundary, as a 400 rather than letting
    # _atempo raise deep inside the lock. Same rule _atempo enforces, but a bad
    # request should not burn a model load and an inference slot to find out.
    # 0.05-8.0 is _atempo's own supported range, wider than the pipeline's
    # 0.5-2.0 schema so a direct caller can still use the chaining.
    if not math.isfinite(req.speed) or not (0.05 <= req.speed <= 8.0):
        raise HTTPException(400, f"speed must be finite and in [0.05, 8.0], "
                                 f"got {req.speed!r}")
    try:
        model, prompt = _get()
        _x_vector_only, language = _clone_opts()
        ft_path, speaker = _finetuned()
        nar = _narrator()
        with _infer_lock:
            if nar:
                # generate_custom_voice, not generate_voice_clone. The narrator
                # path takes a preset speaker id plus an instruct string; sending
                # it down the clone path would silently ignore both, which is the
                # "returns 200 for a change that could not happen" failure this
                # pipeline has already paid for twice.
                wavs, sr = model.generate_custom_voice(
                    text=text, language=language, speaker=nar[1],
                    instruct=nar[2])
            elif ft_path:
                # ponytail: a fine-tuned voice is NOT a clone. generate_voice_clone
                # takes no speaker and would quietly ignore the whole training
                # run, which is the kind of "returns 200, changed nothing" bug
                # this pipeline has already paid for twice.
                wavs, sr = model.generate_custom_voice(
                    text=text, language=language, speaker=speaker)
            else:
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt,
                )
        wav = np.asarray(wavs[0], dtype=np.float32)
        if wav.size == 0:
            raise HTTPException(502, "Qwen3-TTS returned empty audio")
        # pitch-preserving rate change. Skipped entirely at 1.0 so the common path
        # costs nothing and cannot drift.
        rate = float(getattr(req, "speed", 1.0) or 1.0)
        if abs(rate - 1.0) > 0.005 and wav.size:
            try:
                wav = _atempo(wav, int(sr), rate)
            except Exception as e:
                # a failed retime must not lose the utterance: serve it at the
                # synthesised rate rather than returning nothing
                print(f"[tts] atempo {rate:g} failed ({e}); serving un-retimed")
        buf = io.BytesIO()
        sf.write(buf, wav, int(sr), format="WAV", subtype="PCM_16")
        data = buf.getvalue()
        if len(data) < 1000:
            raise HTTPException(502, "Qwen3-TTS returned empty audio")
        return Response(content=data, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Qwen3-TTS inference failed: {str(e)[:200]}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5003)
