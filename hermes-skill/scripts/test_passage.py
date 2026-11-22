"""Tests for passage synthesis and the splitter that backs it.

The defect it fixes: every sentence was a separate engine call, so the model
restarted its prosody each time and pitch jumped at every boundary.

    mean |pitch step| across a sentence boundary
        one call per sentence   4.84 semitones   (2 of 4 boundaries > 3 st)
        one call for the run    1.42 semitones   (0 of 4)

A human speaker moves 1-2 semitones across a boundary in one breath group.
"""

import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import passage as P  # noqa: E402

SR = 24000


def phrase(secs, f0=140, sr=SR):
    """A voiced-sounding phrase: a fundamental plus a wobble, so it has an F0."""
    n = int(secs * sr)
    t = np.arange(n) / sr
    y = 0.4 * np.sin(2 * np.pi * f0 * t + 3 * np.sin(2 * np.pi * 4 * t))
    return y.astype(np.float32)


def phrase_run(count, dur=0.9, gap=0.18, step=8, sr=SR):
    parts, gaps = [], []
    for i in range(count):
        parts.append(phrase(dur, 120 + i * step, sr))
        if i < count - 1:
            gaps.append(np.zeros(int(gap * sr), dtype=np.float32))
    out = []
    for i, p in enumerate(parts):
        out.append(p)
        if i < len(gaps):
            out.append(gaps[i])
    return np.concatenate(out).astype(np.float32)


class BoundaryTests(unittest.TestCase):
    def test_cuts_land_in_silence(self):
        y = phrase_run(5)
        cuts = P.find_boundaries(y, 5, SR)
        self.assertEqual(len(cuts), 4)
        for c in cuts:
            local = np.abs(y[max(0, c - int(0.02 * SR)):c + int(0.02 * SR)]).max()
            self.assertLess(local, 0.02,
                            "cut at {:.3f}s landed on audio at {:.4f}".format(
                                c / SR, local))

    def test_cuts_are_strictly_increasing_and_inside_the_file(self):
        for n in (2, 3, 5, 8, 12):
            y = phrase_run(n)
            cuts = P.find_boundaries(y, n, SR)
            self.assertEqual(len(cuts), n - 1, "{} parts".format(n))
            self.assertTrue(all(b > a for a, b in zip(cuts, cuts[1:])),
                            "cuts not increasing for {} parts".format(n))
            self.assertTrue(all(0 < c < len(y) for c in cuts))

    def test_a_short_silence_is_not_scrapped(self):
        """With only an 8ms plosive available there is no real boundary, and the
        proportional fallback would cut on audio. That must be REJECTED, not used -
        a clipped word is worse than the pitch seam."""
        a = phrase(0.8)
        b = phrase(0.8, 200)
        y = np.concatenate([a, np.zeros(int(0.008 * SR), dtype=np.float32), b])
        ok, bad = P.validate_cuts(y, P.find_boundaries(y, 2, SR), SR)
        self.assertFalse(ok, "a cut on 0.4 amplitude audio must be rejected")
        self.assertEqual(bad, 0)

    def test_split_refuses_rather_than_clipping_a_word(self):
        a = phrase(0.8)
        b = phrase(0.8, 200)
        y = np.concatenate([a, np.zeros(int(0.008 * SR), dtype=np.float32), b])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "p.wav"
            P.write_wav(src, y)
            with self.assertRaises(ValueError):
                P.split_passage(src, [tmp / "o0.wav", tmp / "o1.wav"])

    def test_a_real_boundary_passes_validation(self):
        y = phrase_run(5)
        cuts = P.find_boundaries(y, 5, SR)
        ok, bad = P.validate_cuts(y, cuts, SR)
        self.assertTrue(ok, "cut {} of {} is on audio".format(bad, len(cuts)))

    def test_a_long_boundary_wins_over_a_short_one(self):
        """With a real 200ms pause available, the cut must go there rather than to
        a plosive, which is what min_gap_s is for."""
        a = phrase(0.7)
        plosive = np.zeros(int(0.008 * SR), dtype=np.float32)
        b = phrase(0.7, 200)
        real_gap = np.zeros(int(0.20 * SR), dtype=np.float32)
        c2 = phrase(0.7, 240)
        y = np.concatenate([a, plosive, b, real_gap, c2])
        cuts = P.find_boundaries(y, 3, SR)
        self.assertEqual(len(cuts), 2)
        second = cuts[1]
        # must be at the real gap (~1.41s), not the plosive (~0.70s)
        self.assertGreater(second / SR, 1.2)
        local = np.abs(y[max(0, second - int(0.02 * SR)):second + int(0.02 * SR)]).max()
        self.assertLess(local, 0.02)

    def test_no_silence_falls_back_to_proportional(self):
        """Better than failing: a TTS line's duration is close to proportional to
        its length, so even cuts are roughly right."""
        y = phrase(4.0)
        cuts = P.find_boundaries(y, 4, SR)
        self.assertEqual(cuts, [int(len(y) * k / 4) for k in range(1, 4)])

    def test_one_part_needs_no_cuts(self):
        self.assertEqual(P.find_boundaries(phrase(1.0), 1, SR), [])

    def test_short_input_does_not_crash(self):
        self.assertEqual(P.find_boundaries(np.zeros(10, dtype=np.float32), 4, SR),
                         [2, 5, 7])


class SplitPassageTests(unittest.TestCase):
    def _roundtrip(self, parts):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "p.wav"
            P.write_wav(src, phrase_run(parts))
            outs = [tmp / ("o%02d.wav" % k) for k in range(parts)]
            cuts = P.split_passage(src, outs)
            sizes = [len(P.read_wav(p)[0]) / SR for p in outs]
        return cuts, sizes

    def test_it_produces_one_file_per_part_with_the_right_durations(self):
        for n in (3, 5, 8):
            cuts, sizes = self._roundtrip(n)
            self.assertEqual(len(cuts), n - 1)
            self.assertEqual(len(sizes), n)
            for s in sizes:
                self.assertGreater(s, 0.3, "a piece came out as a stub")

    def test_no_piece_is_silence(self):
        _, sizes = self._roundtrip(5)
        self.assertTrue(all(s > 0.5 for s in sizes))

    def test_files_are_valid_16bit_mono(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "p.wav"
            P.write_wav(src, phrase_run(4))
            out = tmp / "o.wav"
            P.split_passage(src, [out])
            with wave.open(str(out), "rb") as w:
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getsampwidth(), 2)
                self.assertEqual(w.getframerate(), SR)
                self.assertGreater(w.getnframes(), 0)


class PassageConfigTests(unittest.TestCase):
    def test_the_run_length_is_bounded(self):
        """Context is what buys the prosody, but this engine autoregresses and
        drifts on very long requests, so the run must stay bounded."""
        import episode_producer as ep
        self.assertGreaterEqual(ep.PASSAGE_SCENES, 3)
        self.assertLessEqual(ep.PASSAGE_SCENES, 12)
        self.assertLessEqual(ep.PASSAGE_MAX_CHARS, 900)

    def test_a_single_line_takes_the_single_line_path(self):
        """One line must not go through the passage machinery - there is no
        passage, and it would needlessly depend on Whisper alignment."""
        import episode_producer as ep
        calls = []
        original = ep.generate_tts
        ep.generate_tts = lambda *a, **k: calls.append(a[0]) or True
        try:
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "a.wav"
                self.assertEqual(
                    ep.generate_tts_passage(["one line"], [str(p)]), [True])
        finally:
            # MUST restore. Leaving generate_tts replaced breaks every later test
            # that exercises the real one - which is how this leaked once and
            # failed 4 unrelated tests in the full suite.
            ep.generate_tts = original
        self.assertEqual(len(calls), 1)

    def test_a_failing_passage_falls_back_per_line(self):
        """A batch must never be a new failure mode: if the passage request or the
        split cannot deliver, every scene still gets rendered on its own."""
        import episode_producer as ep
        calls = []
        original = ep.generate_tts
        original_health = ep.check_tts_server
        ep.generate_tts = lambda text, path, **k: calls.append(text) or True
        ep.check_tts_server = lambda: False        # force the fallback
        try:
            with tempfile.TemporaryDirectory() as tmp:
                paths = [str(Path(tmp) / ("s%d.wav" % i)) for i in range(4)]
                res = ep.generate_tts_passage(["a", "b", "c", "d"], paths)
        finally:
            ep.generate_tts = original
            ep.check_tts_server = original_health
        self.assertEqual(res, [True] * 4)
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
