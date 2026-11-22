"""Build a voice from a restored reference: user's timbre, restored audio.

VoiceFixer (arXiv:2204.05841) restores what a phone recording actually loses.
It is the right tool for THIS machine's constraint, because the user's mic is
fixed: their recordings are 16 kHz / 37 kbps AAC, so every clone inherits a
bandwidth ceiling that no amount of extra recording minutes can raise.

VoiceFixer is a general speech restoration model - it denoises, dereverbs,
declips and expands bandwidth to 44.1 kHz in one pass. Its super-resolution
branch is precisely the "1 kHz to 22.05 kHz -> full-band" case.

The important caveat, stated up front because it decides how this is used:
restoration is GENERATIVE. It hallucinates plausible high frequencies rather
than recovering the true ones. So it is safe for the REFERENCE CLIP - where we
want the model's best guess at clean timbre - and it is NOT applied to the
synthesised output, which the TTS model produces at full quality already.

Run:
    python restore_ref.py --in ref33.wav --out ref33_restored.wav
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 24000            # what Qwen3-TTS wants
HERE = pathlib.Path(__file__).parent


def bandwidth_report(path, label=""):
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    p = X ** 2
    total = p.sum()
    def pct(level):
        return f[np.searchsorted(np.cumsum(p), level * total)] / 1000.0
    # energy in the top band is the tell: AAC at 37 kbps throws it away
    hi = p[(f >= 8000)].sum() / total
    print("    {:<14} {:.2f}s  99.5%<{:.1f}kHz  energy>8kHz {:6.2f}%  peak {:.1f}dBFS".format(
        label, len(x) / sr, pct(0.995), hi * 100,
        20 * np.log10(max(1e-9, float(np.abs(x).max())))))
    return len(x) / sr, pct(0.995), hi


def restore(src, out, mode="0"):
    """Run VoiceFixer. Its CLI takes --infile/--outfile and numeric --mode."""
    # CUDA by default; voicefixer's flag is --disable-cuda, so it is only added
    # when there is no GPU. Verified: --cuda is NOT a valid flag on this CLI.
    cmd = [sys.executable, "-m", "voicefixer",
           "--infile", str(src), "--outfile", str(out), "--mode", mode]
    try:
        import torch
        if not torch.cuda.is_available():
            cmd.append("--disable-cuda")
    except Exception:
        cmd.append("--disable-cuda")
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=str(HERE / "ref33.wav"))
    ap.add_argument("--out", default=str(HERE / "ref33_restored.wav"))
    ap.add_argument("--mode", default="0", choices=("0", "1", "2", "all"),
                    help="VoiceFixer's own modes: 0 = analysis+synthesis "
                         "(restoration), 1 = analysis only, 2 = synthesis only, "
                         "all = every distortion. 0 is what we want.")
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args()

    src = pathlib.Path(args.inp)
    if not src.is_file():
        raise SystemExit("no such file: " + str(src))

    if not args.skip_report:
        print("  before:")
        bandwidth_report(src, src.name)

    # VoiceFixer wants 44.1 kHz to do its full-band job
    up = pathlib.Path(tempfile.gettempdir()) / "vf_in.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-ar", "44100", "-ac", "1", str(up)], capture_output=True)

    print("  running VoiceFixer (first run downloads ~600MB)...")
    r = restore(up, args.out, args.mode)
    up.unlink(missing_ok=True)
    if r.returncode or not pathlib.Path(args.out).is_file():
        print("  VoiceFixer FAILED:")
        print((r.stdout or "")[-1500:])
        print((r.stderr or "")[-2500:])
        return 1

    print("  after:")
    y, sr = sf.read(args.out, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        down = pathlib.Path(tempfile.gettempdir()) / "vf_out24.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(args.out),
                        "-ar", str(SR), "-ac", "1", str(down)], capture_output=True)
        y, sr = sf.read(str(down), dtype="float32")
        down.unlink(missing_ok=True)
        if y.ndim > 1:
            y = y.mean(axis=1)
    peak = float(np.abs(y).max())
    if peak > 0.0:
        y = y * (10 ** (-3.0 / 20) / peak)      # same -3 dBFS as every other ref
    sf.write(args.out, y, SR, subtype="PCM_16")
    bandwidth_report(args.out, pathlib.Path(args.out).name)
    # Gate it. VoiceFixer is a GENERATIVE vocoder: it invents the missing high
    # frequencies rather than recovering them, which widens the spectrum and
    # can still lose the speaker. MEASURED on ref33.wav: bandwidth +1.42pp
    # (good), UTMOS -0.255 (worse), speaker similarity to the ORIGINAL -0.0528
    # (the voice drifted). So the honest verdict is "wider, not better".
    #
    # Run compare_refs.py to check your own pair before committing; this is not
    # a formality, because the two effects go in opposite directions and only
    # measurement says which won.
    print()
    print("  restored reference: {}".format(args.out))
    print("  CHECK IT before shipping:  python compare_refs.py {} <original>".format(
        pathlib.Path(args.out).name))
    print("  now transcribe it and put THAT text in ref.json - the transcript")
    print("  must match the audio or ICL mode will echo the wrong words.")
    return 0


if __name__ == "__main__":
    sys.exit(main())