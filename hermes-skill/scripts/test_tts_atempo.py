"""Tests for the TTS rate-change and the narration gaps it works with.

_atempo is the riskiest new code in the narration work: it shells out to ffmpeg
per request, and a wrong duration or a lost utterance is invisible in a passing
test suite. Everything here is deterministic - a synthetic tone, and arithmetic -
so it needs no TTS server and no GPU.
"""

import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import soundfile as sf

import episode_producer as ep

# server.py lives in cosy-server/, a sibling of hermes-skill. Importing it needs
# torch and the qwen stack, but only the module-level bindings - no model is
# loaded at import time, so this stays fast and GPU-free.
COSY = Path(__file__).resolve().parents[2] / "cosy-server"
if str(COSY) not in sys.path:
    sys.path.insert(0, str(COSY))
try:
    import server as S
except Exception as _e:                       # pragma: no cover
    S = None
    _IMPORT_ERROR = "{}: {}".format(type(_e).__name__, _e)

SR = 24000


def tone(seconds=2.0, freq=220.0, sr=SR, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_hz(x, sr=SR):
    """Dominant frequency by FFT peak. Retiming must not move it."""
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float(f[int(np.argmax(X))])


class AtempoGuardTests(unittest.TestCase):
    """A float field is not validation. These were hangs."""

    def setUp(self):
        if S is None:
            self.skipTest("cosy-server/server.py not importable: " + _IMPORT_ERROR)

    def test_non_finite_rates_are_refused_not_looped(self):
        """The chain that walks `left` into [0.5, 2.0] never terminates for
        rate <= 0 (multiplies toward 0.5 forever) or nan/inf (satisfies neither
        comparison). Each hang also burns an anyio thread on a sync endpoint."""
        w = tone(0.5)
        for bad in (0.0, -1.0, -0.5, float("inf"), float("-inf"), float("nan")):
            with self.assertRaises(ValueError, msg="rate {!r} must be refused".format(bad)):
                S._atempo(w, SR, bad)

    def test_absurd_rates_are_refused_rather_than_chained(self):
        w = tone(0.5)
        for bad in (1e9, 1e-9, 1e6, 100.0):
            with self.assertRaises(ValueError):
                S._atempo(w, SR, bad)

    def test_the_normal_range_is_accepted(self):
        w = tone(0.5)
        for good in (0.5, 0.9, 0.98, 1.0, 1.04, 1.06, 1.5, 2.0):
            S._atempo(w, SR, good)   # must not raise

    def test_rate_one_is_a_noop(self):
        w = tone(0.5)
        self.assertIs(S._atempo(w, SR, 1.0), w)


class AtempoBehaviourTests(unittest.TestCase):
    """Duration and pitch, measured rather than asserted."""

    @classmethod
    def setUpClass(cls):
        if S is None:
            raise unittest.SkipTest("cosy-server/server.py not importable: "
                                   + _IMPORT_ERROR)
        if not Path(S.CONFIG_FFMPEG).is_file() and S.CONFIG_FFMPEG != "ffmpeg":
            raise unittest.SkipTest("ffmpeg not resolvable")

    def test_duration_matches_the_requested_rate(self):
        w = tone(3.0)
        base = len(w) / SR
        for rate in (0.9, 0.98, 1.04, 1.06, 1.1, 1.5):
            out = S._atempo(w, SR, rate)
            got = len(out) / SR
            want = base / rate
            self.assertAlmostEqual(
                got, want, delta=want * 0.03,
                msg="rate {}: got {:.3f}s, expected {:.3f}s".format(rate, got, want))

    def test_pitch_is_preserved(self):
        """atempo is pitch-preserving; a resample-based speed change would move
        this, and timbre is the entire point of cloning a voice."""
        w = tone(2.0, freq=220.0)
        before = dominant_hz(w)
        for rate in (0.9, 1.06, 1.25):
            after = dominant_hz(S._atempo(w, SR, rate))
            self.assertAlmostEqual(
                after, before, delta=2.0,
                msg="rate {} moved pitch {:.1f}Hz -> {:.1f}Hz".format(
                    rate, before, after))

    def test_output_stays_in_range(self):
        """Clipping would be heard as a crack and is not caught by duration."""
        w = tone(2.0, amp=0.45)
        out = S._atempo(w, SR, 1.5)
        self.assertLessEqual(float(np.abs(out).max()), 1.0)
        self.assertGreater(float(np.abs(out).max()), 0.05, "silence in, silence out")


class VariableGapTests(unittest.TestCase):
    """_concat_wavs is shared with the lip-sync and assembly paths, so the
    variable-gap behaviour needs pinning rather than just a signature check."""

    def _seg(self, tmp, name, seconds):
        p = Path(tmp) / name
        sf.write(str(p), tone(seconds), SR, subtype="PCM_16")
        return p

    def test_gaps_are_applied_per_join(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            parts = [self._seg(tmp, "a.wav", 1.0), self._seg(tmp, "b.wav", 1.0)]
            for gap in (0.05, 0.20):
                out = ep._concat_wavs(parts, Path(tmp) / "j.wav", gaps=[gap])
                self.assertIsNotNone(out)
                got = sf.info(str(out)).duration
                self.assertAlmostEqual(got, 2.0 + gap, delta=0.02)

    def test_gaps_differ_between_joins(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            parts = [self._seg(tmp, "a.wav", 1.0), self._seg(tmp, "b.wav", 1.0),
                     self._seg(tmp, "c.wav", 1.0)]
            out = ep._concat_wavs(parts, Path(tmp) / "j.wav", gaps=[0.05, 0.30])
            self.assertAlmostEqual(sf.info(str(out)).duration, 3.35, delta=0.02)

    def test_a_short_gap_list_falls_back_to_gap_s(self):
        """Documented behaviour: gaps shorter than n-1 leave the rest at gap_s."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            parts = [self._seg(tmp, "a.wav", 1.0), self._seg(tmp, "b.wav", 1.0),
                     self._seg(tmp, "c.wav", 1.0)]
            out = ep._concat_wavs(parts, Path(tmp) / "j.wav",
                                  gap_s=0.12, gaps=[0.05])
            self.assertAlmostEqual(sf.info(str(out)).duration, 3.17, delta=0.02)

    def test_omitting_gaps_reproduces_the_old_behaviour(self):
        """Backwards compatibility for every other caller, including the tests
        that already pin a flat gap."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            parts = [self._seg(tmp, "a.wav", 1.0), self._seg(tmp, "b.wav", 1.0)]
            out = ep._concat_wavs(parts, Path(tmp) / "j.wav")
            self.assertAlmostEqual(sf.info(str(out)).duration, 2.12, delta=0.02)

    def test_a_zero_gap_on_the_last_join_is_not_padding_the_scene_end(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            parts = [self._seg(tmp, "a.wav", 1.0), self._seg(tmp, "b.wav", 1.0)]
            out = ep._concat_wavs(parts, Path(tmp) / "j.wav", gaps=[0.0])
            self.assertAlmostEqual(sf.info(str(out)).duration, 2.0, delta=0.02)


class CacheKeyTests(unittest.TestCase):
    """The narration plan decides what is sent to the engine, so it belongs in
    the key. This is the third time a generation parameter was left out of
    _tts_fingerprint and served stale audio as valid."""

    def test_the_key_includes_the_direction_module(self):
        fp = ep._tts_fingerprint("a line", 1.0)
        self.assertIn("direction", fp)

    def test_retuning_the_direction_retires_the_cache(self):
        d = Path(ep.__file__).parent / "pipeline" / "narration_direction.py"
        original = d.read_text(encoding="utf-8")
        before = ep._tts_fingerprint("a line", 1.0)
        try:
            marker = "_GAP_PAYOFF = "
            i = original.index(marker)
            j = original.index("\n", i)
            d.write_text(original[:i] + marker + "0.41" + original[j:],
                         encoding="utf-8")
            after = ep._tts_fingerprint("a line", 1.0)
            self.assertNotEqual(before, after,
                                "retuning narration_direction must retire every "
                                "cached wav, or it serves the old timing as valid")
        finally:
            d.write_text(original, encoding="utf-8", newline="")


class VoiceSpeedValidationTests(unittest.TestCase):
    def test_bad_speeds_are_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "x.wav"
            for bad in (-1.0, float("nan"), float("inf")):
                live = dict(ep.CONFIG, voice_speed=bad)
                with unittest.mock.patch.dict(ep.CONFIG, live, clear=True):
                    with self.assertRaises(ValueError):
                        ep.generate_tts("words", out)


if __name__ == "__main__":
    unittest.main()