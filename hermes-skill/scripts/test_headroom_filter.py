"""Regression tests for the post-synthesis filter chain.

This chain has now been wrong three times, and every failure was invisible in the
output file: the audio still "looked fine" and still measured loud.

1. `silenceremove , areverse , silenceremove , areverse` DELETED SPOKEN AUDIO.
   "He opened it." became "Ope".
2. "Fixed" with a head-only silenceremove at -60dB over 0.1s. Still wrong:
   silenceremove gates on LEVEL, and the soft onset of the first word is quieter
   than -60dB, so it deleted the first word of 13 of 50 renders.
3. A room-tone bed was then added to fix an "abrupt cut", which introduced hiss
   audible in every pause - a worse artefact than the one it fixed. Removed.

What ships now: `areverse,afade=t=in:curve=qsin,areverse`, 90ms. The ramp is
what removes the abrupt stop, and it adds no noise, so nothing new to hear.

Measured on ch001, same wav into each variant, 2 renders x 25 lines:

    chain                                  mean WER   renders losing a word
    silenceremove .1s @-60dB (head trim)     21.0%          13 / 50
    afade + volume                 (now)    12.8%           0 / 50
    silenceremove .35s @-50dB               48.0%          49 / 50

The third row is why no level-gated silence filter belongs in this chain at all:
lowering sensitivity and raising the minimum duration made it dramatically worse,
because a 0.35s window swallows the first word outright.

The original bug, for the record:

The old chain ended every TTS line with
    silenceremove=start_periods=1:start_duration=0.1:start_threshold=-60dB
which deletes the first WORD of short lines, because a TTS word onset is quieter
than -60 dB and silenceremove gates on LEVEL, not on "is this speech".

Measured on ch001 (same wav into every variant, 2 renders x 25 lines):

    chain                                mean WER   renders losing a word
    silenceremove .1s @-60dB (old)         21.0%          13 / 50
    apad + volume only          (current)  12.8%           0 / 50
    silenceremove .35s @-50dB             48.0%          49 / 50

No synthetic sentence can reproduce this reliably - it needs the level structure
of a real TTS onset, which varies per engine. So this pins the CHAIN and pins
the cache key, and the measured table lives above as the reason.
"""

import math
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import episode_producer as ep

SR = 24000

# The chain generate_tts() applies after synthesis. Keep in sync with
# episode_producer.generate_tts.
CHAIN = "areverse,afade=t=in:st=0:d=0.090:curve=qsin,areverse,volume=-3dB"


class NarrationGapTests(unittest.TestCase):
    """The silence between sentences must be NARRATION_GAP and nothing else.

    Measured on ch001 before this was centralised: mean gap 1.24s (min 1.01,
    max 1.46), from three stacked layers - the apad at the end of every TTS file,
    the scene-slot slack, and the next scene's lead-in. Six times the pause a
    narrator takes.

    The cause was padding, not a bad cut point: the engine's output already decays
    to -80dBFS, and every speech-end threshold from -30dB to -60dB finds the
    identical cut on all 12 sampled lines, so no word was being clipped. There was
    nothing to trim - only padding to stop adding.
    """

    def test_the_gap_is_the_documented_value(self):
        self.assertAlmostEqual(ep.NARRATION_GAP, 0.20, places=3)

    def test_the_tts_file_does_not_pad_its_own_tail(self):
        """No apad in the per-line chain.

        The scene slot already ends NARRATION_GAP after the audio, so a file-level
        pad is a SECOND gap on top of it. Measured: 0.40s instead of 0.20s. It was
        invisible while the slot carried 1.24s of slack of its own.
        """
        src = Path(ep.__file__).read_text(encoding="utf-8")
        live = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        # The per-line chain is the ONLY place this matters. episode_producer
        # legitimately uses apad elsewhere - apad[music] for a music bed, and
        # adelay+apad for lip-sync alignment - so grepping the whole module would
        # flag correct code.
        chain = live[live.index('"-af", "areverse'):]
        chain = chain[:chain.index("trimmed)")]
        self.assertNotIn("apad", chain)
        self.assertNotIn("pad_dur", chain)

    def _slot(self, d, fps=24):
        """Replicate compute_scene_durations' arithmetic for a narration of `d`.

        lead is FIXED at NARRATION_LEAD. Making it absorb the frame-snapping slack
        was a bug: the slack then landed in the audible gap, because

            audible gap = (slot - lead - audio) + NEXT lead
                        = tail + lead

        so a lead that grows by up to one frame doubles the error. Fixed lead means
        slot = audio + NARRATION_GAP and the tail is simply whatever is left.
        """
        dur = math.ceil((d + ep.NARRATION_GAP) * fps) / fps
        return dur, ep.NARRATION_LEAD

    def _gap(self, d, fps=24):
        """The audible gap after a narration of `d`, including the next lead-in."""
        slot, lead = self._slot(d, fps)
        _next_slot, next_lead = self._slot(d, fps)
        return (slot - lead - d) + next_lead

    def test_lead_plus_tail_equals_the_gap(self):
        """The invariant, stated once: the audible gap is NARRATION_GAP.

        Tolerance is one frame, because the slot is snapped UP to the frame grid so
        a scene is never a frame short of its own narration. That snapping can only
        lengthen the gap, never shorten it.
        """
        frame = 1.0 / 24
        for d in (0.5, 0.76, 1.2, 2.0, 2.41, 5.0, 12.0):
            g = self._gap(d)
            self.assertGreaterEqual(g, ep.NARRATION_GAP - 1e-9,
                                    "{:.2f}s narration gave a {:.3f}s gap, which is "
                                    "SHORTER than the target".format(d, g))
            self.assertLessEqual(g, ep.NARRATION_GAP + frame + 1e-9,
                                 "{:.2f}s narration gave a {:.3f}s gap".format(d, g))

    def test_the_lead_matches_the_assembler_edge_fade(self):
        """NARRATION_LEAD is not a free parameter. It equals the 0.05s fade the
        assembler applies to every scene slot, so pinning it there costs nothing:
        the voice already could not start on a hard boundary."""
        self.assertAlmostEqual(ep.NARRATION_LEAD, 0.05, places=3)

    def test_a_scene_is_not_shorter_than_the_gap_itself(self):
        """A very short narration still needs somewhere to put the pause."""
        slot, lead = self._slot(0.2)
        self.assertGreaterEqual(slot, ep.NARRATION_GAP)

    def test_the_gap_includes_the_next_scenes_lead(self):
        """The arithmetic that hides the bug.

            audible gap = (slot - lead - audio) + NEXT lead

        A 2.5s minimum on a 1.2s narration forced lead to 0.9s, and that landed
        straight in the silence - measured mean gap 0.95s against a 0.20s target,
        even though the tail was pinned correctly. Pin the TOTAL, not the tail.
        """
        prev = 1.2                                  # a short ch001 line
        slot, lead = self._slot(prev)
        _nxt_slot, nxt_lead = self._slot(1.6)        # the FOLLOWING scene
        gap = (slot - lead - prev) + nxt_lead
        self.assertAlmostEqual(gap, ep.NARRATION_GAP, delta=0.02,
                               msg="total audible gap is {}s because the next "
                                   "scene's lead-in ({:.2f}s) lands in it"
                                   .format(gap, nxt_lead))

    def test_no_scene_is_held_past_its_narration(self):
        """13 of 25 ch001 scenes were longer than their narration plus lead and
        gap. Holding on a finished frame also means the cut lands late."""
        for d in (0.76, 1.08, 1.21, 1.37, 1.45, 1.61, 1.93, 2.01, 2.41):
            self.assertAlmostEqual(self._gap(d), ep.NARRATION_GAP,
                                   delta=1.0 / 24 + 1e-9,
                                   msg="{:.2f}s narration held for {:.3f}s"
                                       .format(d, self._gap(d)))

    def test_the_scene_slot_allows_exactly_one_gap(self):
        """The AUDIBLE gap must be NARRATION_GAP for every line length.

        Measuring the tail alone is what let this hide twice. The tail is only part
        of the gap - the next scene's lead-in is the rest - so a correct tail still
        shipped a 0.95s pause.
        """
        for d in (0.8, 1.2, 2.0, 2.5, 5.0, 12.0):
            gap = self._gap(d)
            self.assertAlmostEqual(gap, ep.NARRATION_GAP, delta=1.0 / 24 + 1e-9,
                                   msg="{:.1f}s narration gives a {:.3f}s gap, "
                                       "expected {:.2f}s".format(d, gap, ep.NARRATION_GAP))

    def test_the_short_lines_of_ch001_get_the_right_gap(self):
        """The actual distribution in the shipped episode, not a synthetic 2.0s."""
        gaps = [self._gap(d) for d in (0.76, 1.08, 1.21, 1.62, 2.03)]
        for g in gaps:
            self.assertAlmostEqual(g, ep.NARRATION_GAP, delta=1.0 / 24 + 1e-9,
                                   msg="ch001-length narration gave {:.3f}s".format(g))

    def test_a_long_line_is_not_truncated_by_the_slot(self):
        """A too-short slot silently cuts the last words, which is the bug this
        area has a history of."""
        for d in (1.0, 2.0, 5.0, 12.0):
            dur, lead = self._slot(d)
            self.assertGreaterEqual(dur - lead, d - 1e-9,
                                    "slot {}s cannot hold {}s of narration".format(dur, d))

    def test_internal_joins_stay_below_the_scene_gap(self):
        """A sentence boundary inside a scene is a smaller beat than a scene
        change. If the two were equal, the pacing would read as one flat series of
        identical beats - the metronome this module was written to avoid."""
        from pipeline import narration_direction as nd
        for segs in (["The canal was black and silent."],
                     ["Nobody moved.", "Everything stayed exactly where it was."]):
            gaps = [nd.gap_after(s, is_last_segment=(i == len(segs) - 1))
                    for i, s in enumerate(segs)]
            for g in gaps:
                self.assertLessEqual(g, ep.NARRATION_GAP,
                                     "internal join {}s exceeds the scene gap {}s"
                                     .format(g, ep.NARRATION_GAP))

    def test_the_last_segment_gets_no_trailing_gap(self):
        """Padding the final segment only pushes the next scene's start, which
        assembly already controls."""
        from pipeline import narration_direction as nd
        self.assertEqual(nd.gap_after("Anything at all.", is_last_segment=True), 0.0)


class LevelMatchTests(unittest.TestCase):
    """Every line must sit at the same level, or the read is not smooth.

    "Smooth pacing" is mostly loudness consistency, not gaps. Measured on ch001
    before this: per-line RMS ran -32.2 to -22.7 dB, a 6.6 dB p10-p90 spread. A
    real narrator holds an episode within ~3 dB. A 6 dB step between consecutive
    lines is heard as a jolt long before the listener registers a gap - which is
    why gap tuning alone never fixed it.
    """

    @staticmethod
    def _speech_rms(pcm):
        """Replicate generate_tts' measurement: int16 scale in, dBFS out."""
        import math
        import numpy as np
        nz = np.where(np.abs(pcm) > 32768 * 10 ** (-45 / 20))[0]
        if not len(nz):
            return None
        seg = pcm[int(nz[0]):int(nz[-1]) + 1]
        return 20 * math.log10(max(
            float(np.sqrt((seg.astype(np.float32) ** 2).mean())) / 32768.0, 1e-9))

    def _correct(self, pcm):
        import numpy as np
        got = self._speech_rms(pcm)
        if got is None:
            return None, None
        gain = min(6.0, max(-6.0, ep.NARRATION_RMS - got))
        out = np.clip(pcm.astype(np.float32) * 10 ** (gain / 20.0),
                      -32767, 32767).astype(np.int16)
        return out, got

    def test_a_corrected_line_lands_on_the_target(self):
        """Amplitudes chosen to sit INSIDE the +/-6 dB clamp.

        A line at -17 dBFS cannot be pulled to -26.5 by a 6 dB ceiling - it lands
        at -23 - and that is correct behaviour, not a failure. So the reach test
        uses amplitudes within the clamp, and the clamp itself is pinned
        separately below.
        """
        import numpy as np
        n = int(0.8 * SR)
        t = np.arange(n) / SR
        for amp in (0.038, 0.055, 0.070):      # -32, -29, -26 dBFS RMS
            pcm = (amp * np.sin(2 * np.pi * 180 * t) * 32767).astype(np.int16)
            out, got = self._correct(pcm)
            self.assertAlmostEqual(self._speech_rms(out), ep.NARRATION_RMS, delta=0.5,
                                   msg="amp {:.3f}: {} dB in, {} dB out".format(
                                       amp, got, self._speech_rms(out)))

    def test_an_out_of_reach_line_lands_on_the_clamp_not_the_target(self):
        """A hot line cannot be pulled 9 dB down with a 6 dB ceiling. Asserting
        it lands exactly on the target would be asserting the clamp does not
        exist."""
        import numpy as np
        n = int(0.8 * SR)
        pcm = (0.4 * np.sin(2 * np.pi * 180 * np.arange(n) / SR) * 32767).astype(np.int16)
        out, got = self._correct(pcm)
        self.assertGreater(got - ep.NARRATION_RMS, 6.0,
                           "fixture must start more than 6 dB above the target")
        self.assertAlmostEqual(self._speech_rms(out), got - 6.0, delta=0.2)

    def test_the_measurement_is_in_dBFS_not_int16_scale(self):
        """The bug this shipped with, pinned.

        The RMS is computed on int16 samples while the target is dBFS (1.0 = full
        scale). Dividing one by the other asks for about -83 dB of gain, which the
        +/-6 dB clamp swallows: every line logged a correction, none applied, the
        spread stayed at 6.6 dB, and the feature looked alive.
        """
        import math
        import numpy as np
        n = int(0.8 * SR)
        pcm = (0.2 * np.sin(2 * np.pi * 180 * np.arange(n) / SR) * 32767).astype(np.int16)
        self.assertGreater(self._speech_rms(pcm), -40.0,
                           "dBFS measurement looks int16-scaled")
        mixed = 20 * math.log10(
            10 ** (ep.NARRATION_RMS / 20.0)
            / float(np.sqrt((pcm.astype(np.float32) ** 2).mean())))
        self.assertLess(mixed, -40.0,
                        "the mixed-scale expression should be nonsense here; if it "
                        "is not, the fixture changed and this guards nothing")

    def test_the_gain_is_clamped_both_ways(self):
        """A very quiet line must not be boosted into its own noise floor, and a
        hot one must not be pushed further into the ceiling."""
        import numpy as np
        n = int(0.8 * SR)
        t = np.arange(n) / SR
        # 0.015 -> -39.5 dBFS (13 dB below target, beyond the +6 clamp);
        # 0.4   -> -11 dBFS   (15 dB above target, beyond the -6 clamp)
        for amp, ceiling in ((0.015, 6.0), (0.4, -6.0)):
            pcm = (amp * np.sin(2 * np.pi * 180 * t) * 32767).astype(np.int16)
            got = self._speech_rms(pcm)
            self.assertIsNotNone(got, "amp {} is below the -45 dBFS floor".format(amp))
            gain = min(6.0, max(-6.0, ep.NARRATION_RMS - got))
            self.assertLessEqual(gain, ceiling,
                                 "amp {} wanted more than {:+} dB".format(amp, ceiling))

    def test_a_silent_line_is_left_alone(self):
        """No speech above -45 dBFS means nothing to measure. Amplifying noise to
        fill a level target would be the wrong fix."""
        import numpy as np
        pcm = np.zeros(SR, dtype=np.int16)
        self.assertIsNone(self._speech_rms(pcm))
        self.assertIsNone(self._correct(pcm)[0])

    def test_the_cache_key_names_the_level_target(self):
        """Changing the target changes the audio, so it belongs in the key."""
        self.assertIn("levelmatch", ep._tts_fingerprint("a line", 1.0)["trim"])


class PostSynthesisChainTests(unittest.TestCase):
    def test_the_shipped_chain_contains_no_silenceremove(self):
        """Pin the finding itself, so a future tidy-up cannot reintroduce it.

        Lowering the threshold sensitivity and raising the minimum duration made
        it dramatically WORSE (.35s @-50dB lost a word in 49 of 50 renders),
        because a 0.35s window swallows the first word outright. There is no
        threshold that separates a TTS word onset from silence here, so the
        filter is the wrong tool rather than a mistuned one.
        """
        self.assertNotIn("silenceremove", CHAIN)

    def test_the_cache_key_names_this_exact_chain(self):
        """A trim change must retire every cached wav, or audio trimmed by the
        old chain keeps being served as valid."""
        fp = ep._tts_fingerprint("a line", 1.0)
        self.assertNotIn("silenceremove", fp["trim"])
        self.assertNotIn("apad", fp["trim"])
        self.assertIn("afade", fp["trim"])

    def test_there_is_no_pad_to_order_against(self):
        """The old test was "fade must precede apad". There is no apad now, and
        the scene slot owns the pause, so the question is gone rather than moved."""
        self.assertNotIn("apad", CHAIN)

    def test_the_fade_is_long_and_quarter_sine(self):
        """90ms quarter-sine, both measured.

            fade            dB lost in the final 20ms      WER
            45ms linear            10.5                     2.9%   <- was shipping
            90ms linear            16.1                     4.0%
            90ms QUARTER-SINE       2.9                     0.0%   <- shipped
            140ms quarter-sine      6.2                     9.5%
            140ms double-exp        3.0                   108.5%

        A linear fade starts at full steepness, so it chops the word instead of
        releasing it. Quarter-sine starts gently. And the dexp row is the trap: it
        wins on abruptness alone and destroys the audio, so the WER column is not
        optional.
        """
        self.assertIn("d=0.090", CHAIN)
        self.assertIn("curve=qsin", CHAIN)
        self.assertNotIn("curve=tri", CHAIN)
        self.assertNotIn("curve=dexp", CHAIN)

    def test_the_fade_targets_the_end_not_the_start(self):
        """`afade=t=out:st=0` is a trap: on t=out, `st` is measured from the
        START of the stream, so it ramps the first 45ms and deletes the quiet
        onset - the same class of bug this chain already committed twice.

        Measured on the fixture below (0.002-amplitude onset at 0.20s):
            afade=t=out:st=0              onset band max 0.000000  destroyed
            areverse,afade=t=in,areverse  onset band max 0.001404  intact
        """
        self.assertTrue(CHAIN.startswith("areverse"))
        self.assertNotIn("t=out", CHAIN)

    def test_the_onset_survives_the_actual_shipped_chain(self):
        """Directly assert the fixture's quiet onset comes through, using the
        real chain rather than a hand-built one."""
        out = self._run(self._leading_silence_then_quiet_onset())
        band = float(np.abs(out[int(0.20 * SR):int(0.32 * SR)]).max())
        self.assertGreater(band, 5e-4,
                           "the quiet onset was destroyed by the filter chain")

    def test_the_chain_adds_no_noise(self):
        """The third failure was a room-tone bed: hiss in every pause. Nothing
        in this chain may introduce a noise source."""
        for forbidden in ("anoisesrc", "aevalsrc", "amix", "room_tone"):
            self.assertNotIn(forbidden, CHAIN)

    def _leading_silence_then_quiet_onset(self):
        """The shape of a real TTS line: silence, then a word whose ONSET is
        quiet (~-54 dBFS), then the vowel. Reproducible, unlike real speech."""
        n = np.arange(SR) / SR
        x = np.zeros_like(n)
        x[(n >= 0.20) & (n < 0.32)] = 0.002 * np.sin(2 * np.pi * 3000 * n[(n >= 0.20) & (n < 0.32)])
        x[n >= 0.32] = 0.5 * np.sin(2 * np.pi * 140 * n[n >= 0.32])
        return x.astype(np.float32)

    def test_the_quiet_onset_reaches_the_output_intact(self):
        """The actual regression.

        Measured on this fixture: the old chain removed 0.31s — the 0.20s of
        leading silence AND ~0.11s of the 0.12s quiet onset, i.e. 92% of the
        first word. The current chain removes nothing from the head.
        """
        x = self._leading_silence_then_quiet_onset()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
            with wave.open(str(src), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                w.writeframes(np.clip(x * 32767, -32768, 32767)
                              .astype("<i2").tobytes())
            r = subprocess.run([ep.CONFIG["ffmpeg"], "-y", "-i", str(src),
                                "-af", CHAIN, "-ar", str(SR), "-ac", "1", str(dst)],
                               capture_output=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr[-300:])
            with wave.open(str(dst), "rb") as w:
                y = np.frombuffer(w.readframes(w.getnframes()),
                                  dtype="<i2").astype(np.float32) / 32768.0
        # No pad, so the length must be unchanged: the fade releases the tail, it
        # does not extend the file.
        self.assertAlmostEqual(len(y) / SR, 1.0, delta=0.02)
        onset_at = float(np.argmax(np.abs(y) > 1e-3)) / SR
        self.assertAlmostEqual(onset_at, 0.20, delta=0.02,
                               msg="the head was trimmed; a quiet first word "
                                   "does not survive a level-gated silence filter")

    def _run(self, x):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
            with wave.open(str(src), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                w.writeframes(np.clip(x * 32767, -32768, 32767)
                              .astype("<i2").tobytes())
            r = subprocess.run([ep.CONFIG["ffmpeg"], "-y", "-i", str(src),
                                "-af", CHAIN, "-ar", str(SR), "-ac", "1", str(dst)],
                               capture_output=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr[-300:])
            with wave.open(str(dst), "rb") as w:
                return np.frombuffer(w.readframes(w.getnframes()),
                                     dtype="<i2").astype(np.float32) / 32768.0

    def test_the_release_is_a_ramp_not_a_step(self):
        """The shipped fade must actually ramp.

        Measured on REAL engine output (5 lines, s016 the worst offender) as dB
        lost inside the final 20ms:

            no fade          9.3 dB
            45ms linear     10.5 dB   <- was shipping: no better than no fade
            90ms linear     16.1 dB
            90ms QUARTER-SINE 2.9 dB   <- shipped
            140ms quarter-sine 6.2 dB, 9.5% WER
            140ms double-exp   3.0 dB, 108.5% WER  <- trap, destroys the audio

        A sine or an exponentially-decaying tone is the WRONG fixture here: both
        have a flat 5ms envelope, so the drop metric reads ~0 either way and
        proves nothing. The real defect is that the engine ends a word while it is
        still sounding at -4 dBFS. So this uses a constant-amplitude signal and
        asserts the applied gain really is a ramp that reaches zero.
        """
        n = int(0.5 * SR)
        x = np.full(n, 0.4, dtype=np.float32)          # constant amplitude
        out = self._run(x)
        fade = int(0.090 * SR)
        tail = out[-fade:]
        self.assertAlmostEqual(float(tail[-1]), 0.0, places=6,
                               msg="the release does not reach silence")
        # quarter-sine: gain rises from 0 to 1 over the ramp, so reading the ramp
        # BACKWARDS from the file end gives a monotonically falling envelope
        gain = np.abs(tail[::-1])
        gain = gain / max(gain.max(), 1e-9)
        steps = np.diff(gain)
        self.assertLess(float(steps.max()), 0.02,
                        "the release is not a smooth ramp - something in it is a step")
        self.assertGreater(float(gain[-1]), 0.5,
                           "a quarter-sine must start GENTLY; if this is high the "
                           "curve is wrong and it will chop the word")

    def test_no_step_anywhere_in_the_tail(self):
        """A step inside one window is a click. The shipped chain produces zero
        of these; the head-trim era produced one in 4% of lines."""
        n = int(1.0 * SR)
        t = np.arange(n) / SR
        y = (0.4 * np.sin(2 * np.pi * 180 * t) * np.exp(-1.5 * t)).astype(np.float32)
        out = self._run(y)
        a = np.abs(out)
        win = int(0.005 * SR)
        env = np.array([a[i:i + win].max() for i in range(0, len(a) - win + 1, win)])
        loud = np.where(env >= 10 ** (-30 / 20))[0]
        for e in loud:
            before = float(env[e])
            after = float(env[e + 1]) if e + 1 < len(env) else 0.0
            db = 20 * np.log10(max(after, 1e-9) / max(before, 1e-9))
            self.assertGreater(db, -20.0,
                               "step of {:.1f} dB inside one 5ms window".format(db))


if __name__ == "__main__":
    unittest.main()