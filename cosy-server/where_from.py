"""Where did a clip come from? Match it against candidate source recordings.

Finds the source by cross-correlating the clip's spectrum-free envelope against
each recording's envelope - a fingerprint match, not a guess from filenames.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import prep_ref as P  # noqa: E402

SR = 24000
HOP = 240  # 10 ms


def envelope(x, sr=SR):
    n = len(x) // HOP * HOP
    e = np.array([np.sqrt((x[i:i + HOP] ** 2).mean() + 1e-12)
                  for i in range(0, n, HOP)])
    return e / (e.max() + 1e-12)


def best_offset(clip_env, src_env):
    """Normalised cross-correlation of the clip envelope over the source."""
    c = clip_env - clip_env.mean()
    cn = np.linalg.norm(c)
    if cn < 1e-9:
        return 0.0, 0.0
    best, best_at = -2.0, 0
    step = 4
    for i in range(0, len(src_env) - len(c), step):
        s = src_env[i:i + len(c)] - src_env[i:i + len(c)].mean()
        sn = np.linalg.norm(s)
        if sn < 1e-9:
            continue
        r = float((c * s).sum() / (cn * sn))
        if r > best:
            best, best_at = r, i
    return best, best_at


def main():
    clip_path = Path(sys.argv[1])
    cx, _ = P.load_24k_mono(clip_path)
    ce = envelope(cx)
    print("clip {} ({:.2f}s)".format(clip_path.name, len(cx) / SR))
    for src in sys.argv[2:]:
        sp = Path(src)
        try:
            sx, _ = P.load_24k_mono(sp)
        except Exception as e:
            print("  {}  DECODE FAILED {}".format(sp.name, e))
            continue
        se = envelope(sx)
        r, at = best_offset(ce, se)
        print("  {:<22} corr {:+.4f}  at {:.2f}s".format(sp.name, r, at / SR / 10.0))


if __name__ == "__main__":
    main()
