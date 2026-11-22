#!/usr/bin/env python3
"""Tests for _loudnorm_dual()'s LRA target selection.

ffmpeg reverts linear=true to dynamic mode unless target LRA >= source LRA
(ffmpeg loudnorm docs). A dynamic fallback pumps the narration against the
music bed, and the downstream alimiter still holds true peak, so every QA gate
stays green while the mix is silently time-varying.

These tests stub the ffmpeg measurement so both the narrow (real behaviour)
and wide (latent) paths are deterministic — gated silence is excluded from
loudness-range measurement, so a synthetic file cannot reach a wide LRA.
"""
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import episode_producer as ep


def _json_block(**kw):
    """Render a loudnorm measurement block the way ffmpeg actually does.

    ffmpeg prints non-finite values as QUOTED strings ("-inf", "inf"),
    never as bare JSON literals, so the stub must match: a bare `inf` would
    be invalid JSON and fail json.loads before the clamp is ever reached.
    """
    def _fmt(v):
        s = repr(float(v))
        if "inf" in s or "nan" in s:
            return f'"{s}"'
        return s
    body = ",".join(f'"{k}": {_fmt(v)}' for k, v in kw.items())
    return "{" + body + "}"


class LoudnormLraTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src = self.root / "in.wav"
        self.dst = self.root / "out.wav"
        self.src.write_bytes(b"RIFF" + b"\0" * 64)  # never decoded; ffmpeg is stubbed
        self.captured = []
        self._orig = ep.subprocess.run

    def _stub(self, input_lra):
        """Run _loudnorm_dual with ffmpeg stubbed to report input_lra."""
        block = _json_block(input_i=-22.7, input_tp=-5.6, input_lra=input_lra,
                            input_thresh=-33.0, target_offset=0.1)

        def fake(cmd, *a, **k):
            self.captured.append(" ".join(map(str, cmd)))
            if "-f" in cmd and "null" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "", block)
            self.dst.write_bytes(b"RIFF" + b"\0" * 128)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch.object(ep.subprocess, "run", fake):
            ok = ep._loudnorm_dual(str(self.src), str(self.dst), -14.0)
        filt = [c for c in self.captured if "linear=true" in c]
        return ok, (filt[0] if filt else None)

    def _target(self, filt):
        return float(re.search(r"LRA=([\d.]+)", filt).group(1))

    def test_narrow_source_keeps_shipped_target(self):
        """Real narration measures LRA ~5.3; behaviour must not change."""
        ok, filt = self._stub(input_lra=5.3)
        self.assertTrue(ok)
        self.assertIsNotNone(filt)
        self.assertEqual(self._target(filt), 11.0,
                         "a narrow source must keep the shipped LRA=11 target")

    def test_wide_source_raises_target_above_source(self):
        """The fix: target LRA must reach the measured source LRA so ffmpeg
        does not silently drop out of linear mode."""
        ok, filt = self._stub(input_lra=23.0)
        self.assertTrue(ok)
        self.assertIsNotNone(filt)
        tgt = self._target(filt)
        self.assertGreaterEqual(tgt, 23.0,
                                f"target {tgt} must be >= source LRA 23.0")
        self.assertIn("linear=true", filt)

    def test_wide_source_still_passes_measured_values(self):
        ok, filt = self._stub(input_lra=30.0)
        self.assertTrue(ok)
        # `:g` drops trailing zeros, so compare numerically not textually.
        self.assertIn("measured_LRA=30:", filt)
        self.assertEqual(float(re.search(r"measured_LRA=([\d.]+)", filt).group(1)), 30.0)
        self.assertEqual(float(re.search(r"measured_I=(-[\d.]+)", filt).group(1)), -22.7)

    def test_out_of_range_measurement_is_clamped_not_fatal(self):
        """A wild measurement must be clamped into ffmpeg's real accepted
        range, not merely survive to the filter string.

        ffmpeg 9.0.1 verified ranges: LRA 1..50, measured_I -99..0,
        measured_LRA 0..99, measured_thresh/offset reject inf/-inf/nan.
        A silent master really does measure input_i=-inf, target_offset=inf.
        """
        cases = [0.0, -5.0, 999.0, 100.0, float("inf"), float("-inf")]
        for bogus in cases:
            with self.subTest(lra=bogus):
                self.captured.clear()
                self.dst.unlink(missing_ok=True)
                ok, filt = self._stub(input_lra=bogus)
                self.assertTrue(ok, f"LRA={bogus} must not fail the pass")
                self.assertIsNotNone(filt)
                # Every numeric value in the filter must be finite and inside
                # its documented range.
                for key, lo, hi in (("LRA", 1.0, 50.0),
                                    ("measured_I", -99.0, 0.0),
                                    ("measured_LRA", 0.0, 99.0),
                                    ("measured_thresh", -99.0, 0.0),
                                    ("offset", -99.0, 99.0)):
                    m = re.search(rf"{key}=(-?[\d.e+-]+)", filt)
                    self.assertIsNotNone(m, f"{key} missing from {filt!r}")
                    val = float(m.group(1))
                    self.assertEqual(val, val, f"{key}={val} is NaN")
                    self.assertTrue(lo <= val <= hi,
                                    f"{key}={val} outside ffmpeg range {lo}..{hi}")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
    def test_silent_master_measurement_still_builds_a_valid_filter(self):
        """End-to-end against the real binary: a silent input measures
        input_i=-inf / target_offset=inf. Those must never reach ffmpeg, or
        the filter is rejected and the publish silently downgrades to an
        un-normalized master."""
        silent = self.root / "silent.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "anullsrc=r=44100:cl=stereo", "-t", "4",
                        "-c:a", "pcm_s16le", str(silent)], check=True, timeout=120)

        block = None
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(silent),
             "-map", "0:a:0", "-af",
             f"loudnorm=I=-14:TP={ep._TP_LIMIT}:LRA=11:print_format=json",
             "-f", "null", "-"], capture_output=True, text=True, timeout=180)
        blocks = re.findall(r'\{[^{}]*"input_i"[^{}]*\}', p.stderr or "", re.DOTALL)
        self.assertTrue(blocks, "expected a loudnorm measurement block")
        block = json.loads(blocks[-1])
        # Confirm the premise: this input really does measure non-finite.
        self.assertIn("-inf", json.dumps(block),
                      "premise changed: silent input no longer measures -inf")

        # ponytail: `orig_run` was undefined here — pyflakes flagged it and the
        # test still passed only because every call this test makes takes the
        # branch below, so the fallback (and its NameError) was unreachable. A
        # latent NameError in a test helper is a trap: the moment a second ffmpeg
        # call appears, this fails with a name error instead of a real
        # assertion. Bind the genuine implementation explicitly.
        orig_run = subprocess.run

        def fake(cmd, *a, **k):
            self.captured.append(" ".join(map(str, cmd)))
            if "-f" in cmd and "null" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "", json.dumps(block))
            return orig_run(cmd, *a, **k)

        with patch.object(ep.subprocess, "run", fake):
            filt = None
            out = self.root / "out.wav"
            ok = ep._loudnorm_dual(str(silent), str(out), -14.0)
            self.assertTrue(ok, "a clamped filter must still produce output")
            p2 = [c for c in self.captured if "linear=true" in c]
            filt = p2[0] if p2 else None
        self.assertIsNotNone(filt)
        # The REAL ffmpeg must accept the string the code built.
        self.assertNotIn("inf", filt, f"non-finite value leaked: {filt}")
        self.assertNotIn("nan", filt.lower(), f"NaN leaked: {filt}")
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(silent),
             "-filter:a", filt.split("-filter:a ")[1].split(" -c:a")[0],
             "-c:a", "pcm_s16le", "-ar", "44100", str(self.root / "probe.wav")],
            capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0,
                         f"real ffmpeg rejected the filter: {r.stderr[-300:]}")

    def test_measured_values_absent_falls_back_to_default(self):
        """A malformed measurement block must not crash the assembler."""
        block = _json_block(input_i=-22.7, input_tp=-5.6, input_thresh=-33.0,
                            target_offset=0.1)  # note: no input_lra

        def fake(cmd, *a, **k):
            self.captured.append(" ".join(map(str, cmd)))
            if "-f" in cmd and "null" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "", block)
            self.dst.write_bytes(b"RIFF" + b"\0" * 128)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch.object(ep.subprocess, "run", fake):
            ok = ep._loudnorm_dual(str(self.src), str(self.dst), -14.0)
        self.assertTrue(ok)

    def test_no_output_reports_failure(self):
        """A zero-return ffmpeg that wrote nothing must not report success."""
        def fake(cmd, *a, **k):
            if "-f" in cmd and "null" in cmd:
                block = _json_block(input_i=-22.7, input_tp=-5.6, input_lra=5.0,
                                    input_thresh=-33.0, target_offset=0.0)
                return subprocess.CompletedProcess(cmd, 0, "", block)
            return subprocess.CompletedProcess(cmd, 0, "", "")  # writes nothing

        with patch.object(ep.subprocess, "run", fake):
            ok = ep._loudnorm_dual(str(self.src), str(self.dst), -14.0)
        self.assertFalse(ok, "must not claim success without an output file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
