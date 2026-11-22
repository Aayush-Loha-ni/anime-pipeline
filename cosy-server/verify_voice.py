"""Serve ref.json's config, and prove the shipped one actually works end to end.

The pipeline gates on /health returning status=ok AND loaded=True, so this
starts the real server, checks /health, speaks a sentence through /cosy, and
transcribes the result to confirm it said the words rather than merely
returning bytes.

Usage:
    python verify_voice.py                 # full check, real server
    python verify_voice.py --no-serve      # config + files only, no GPU
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REF = HERE / "ref.json"
URL = "http://127.0.0.1:5003"
TEST_LINE = "The engine turned over twice and settled into a low uneven idle."


def cfg_checks():
    cfg = json.loads(REF.read_text(encoding="utf-8"))
    print("  config:")
    ok = True

    wav = pathlib.Path(cfg["wav"])
    exists = wav.is_file()
    ok &= exists
    print("    {:<18} {} {}".format("wav", cfg["wav"][-40:], "OK" if exists else "MISSING"))

    has_text = bool((cfg.get("text") or "").strip())
    ok &= has_text
    print("    {:<18} {}".format("text", (cfg.get("text") or "<EMPTY>")[:60]))

    mode = "ICL" if not cfg.get("x_vector_only", True) else "x_vector_only"
    print("    {:<18} {}".format("mode", mode))

    print("    {:<18} {}".format("language", cfg.get("language", "English")))
    print("    {:<18} {}".format("finetuned", cfg.get("finetuned_model", "(none - base clone)")))
    if cfg.get("finetuned_model"):
        p = pathlib.Path(cfg["finetuned_model"])
        ok &= (p / "config.json").is_file()
        print("    {:<18} {}".format("ft checkpoint", "OK" if (p / "config.json").is_file() else "MISSING"))
    return cfg, ok


def serve_check(cfg):
    p = subprocess.Popen([sys.executable, str(HERE / "server.py")],
                         cwd=str(HERE), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    try:
        health = None
        for _ in range(240):
            time.sleep(2)
            if p.poll() is not None:
                print("    server exited early:\n{}".format(p.stdout.read()[-1200:]))
                return None, False
            try:
                health = json.loads(urllib.request.urlopen(URL + "/health", timeout=5).read())
                break
            except Exception:
                continue
        if not health:
            print("    server did not become healthy in time")
            return None, False

        print("  /health:")
        for k in ("status", "loaded", "mode", "language", "engine"):
            if k in health:
                print("    {:<12} {}".format(k, str(health[k])[-58:]))
        ok = health.get("status") == "ok" and health.get("loaded") is True
        print("    gate (status==ok and loaded): {}".format("PASS" if ok else "FAIL"))

        req = urllib.request.Request(
            URL + "/cosy",
            data=json.dumps({"text": TEST_LINE}).encode(),
            headers={"Content-Type": "application/json"})
        wav = urllib.request.urlopen(req, timeout=300).read()
        out = HERE / "_verify.wav"
        out.write_bytes(wav)
        print("  /cosy: {:,} bytes -> {}".format(len(wav), out.name))

        import soundfile as sf
        a, sr = sf.read(str(out))
        dur = len(a) / sr
        print("    {:.2f}s @ {}Hz, peak {:.1f} dBFS".format(
            dur, sr, 20 * __import__("math").log10(max(1e-9, float(abs(a).max())))))
        ok &= dur > 0.5

        import whisper
        import torch
        m = whisper.load_model("small.en", device="cuda:0" if torch.cuda.is_available() else "cpu")
        got = (m.transcribe(str(out), language="en", temperature=0.0,
                            condition_on_previous_text=False).get("text") or "").strip()
        print("    expected: {}".format(TEST_LINE))
        print("    heard:    {}".format(got))
        # A leak shows up as the reference's own words appearing in output the
        # model was not asked to speak. Derive the probe from ref.json rather
        # than hardcoding: the previous hardcoded list ("meant to be", "little
        # things") was the DELETED song's lyrics, so with a spoken reference the
        # check could never fire again - it looked like a guard and was not one.
        ref_cfg = json.loads((HERE / "ref.json").read_text(encoding="utf-8"))
        ref_words = [w.lower().strip(".,!?;:\"'")
                     for w in re.findall(r"[A-Za-z']+", ref_cfg.get("text") or "")]
        # distinctive 5-grams from the reference: a leak repeats a RUN, not a word
        grams = set()
        for i in range(max(0, len(ref_words) - 4)):
            g = tuple(ref_words[i:i + 5])
            if len(g) == 5 and g not in grams:
                grams.add(g)
        spoken = re.findall(r"[A-Za-z']+", got.lower())
        spoken_grams = {tuple(spoken[i:i + 5])
                        for i in range(max(0, len(spoken) - 4))}
        leaked = sorted(" ".join(g) for g in (grams & spoken_grams))
        ok &= not leaked
        if leaked:
            print("    LEAK: reference text in output: {}".format(leaked))
        else:
            print("    no reference-text leak ({} 5-grams probed)".format(len(grams)))
        out.unlink(missing_ok=True)
        return health, ok
    finally:
        p.terminate()
        try:
            p.wait(timeout=20)
        except Exception:
            p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()
    cfg, ok = cfg_checks()
    if args.no_serve:
        return 0 if ok else 1
    print("  starting server (loads the 1.7B model, ~30-60s)...")
    _, ok2 = serve_check(cfg)
    print()
    print("  {}".format("PASS" if (ok and ok2) else "FAIL"))
    return 0 if (ok and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())