"""Clip length must come off the narration, not a constant.

This is the check that fails if someone hardcodes a duration again: before
this, every clip was built at `duration=6.0` regardless of how long the line
was, and assembly papered over the gap with -stream_loop.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import episode_producer as ep


def _h3_frames(seconds, fps=None):
    """The frames _try_h3_clip would actually build, without loading ComfyUI.

    Copied from the call site plus _try_h3_clip's snap rather than imported:
    the snap is inlined inside a function that builds a ComfyUI workflow, so
    there is nothing to call. If the snap changes, these numbers go stale —
    which is the point, because a silent change to it is what put a seam in
    16 scenes.
    """
    fps = fps or int(ep.CONFIG.get("h3_fps", 24))
    length = math.ceil(seconds * fps)
    return 17 * max(1, math.ceil((length - 5) / 17)) + 5


class TestClipLengthFollowsAudio(unittest.TestCase):
    def test_short_line_gets_a_short_clip(self):
        self.assertLessEqual(ep.scene_clip_seconds({"_dur": 2.4}), 2.4)

    def test_long_line_gets_a_long_clip(self):
        # The old behaviour returned 6.0 for every scene, so a 9s line came
        # back short and assembly looped it twice.
        self.assertAlmostEqual(ep.scene_clip_seconds({"_dur": 9.0}), 9.0, places=6)

    def test_absurdly_long_line_is_capped(self):
        # A 40s narration must not become a 40s diffusion pass.
        cap = float(ep.CONFIG.get("max_scene_clip_sec", 12))
        self.assertLessEqual(ep.scene_clip_seconds({"_dur": 40.0}), cap)

    def test_no_duration_falls_back_to_a_usable_floor(self):
        # A scene with no audio (faceless establishing) still needs a clip
        # that is not 0s long, or ffmpeg writes nothing and the phase gate
        # trips on a missing file instead of a missing narration.
        self.assertGreaterEqual(ep.scene_clip_seconds({}), 1.0)
        self.assertGreaterEqual(ep.scene_clip_seconds({"_dur": 0}), 1.0)

    def test_length_never_exceeds_the_cap(self):
        cap = float(ep.CONFIG.get("max_scene_clip_sec", 12))
        for d in (0.1, 1, 5, 12, 13, 120):
            self.assertLessEqual(ep.scene_clip_seconds({"_dur": d}), cap)

    def test_h3_frame_count_is_on_the_model_grid(self):
        fps = int(ep.CONFIG.get("h3_fps", 24))
        for d in (2.0, 5.17, 9.0, 12.0):
            snapped = _h3_frames(ep.scene_clip_seconds({"_dur": d}), fps)
            # On-grid, and never a different shot length than asked for.
            self.assertEqual((snapped - 5) % 17, 0)
            self.assertLessEqual(abs(snapped / fps - d), 17 / fps + 1e-6)

    def test_h3_clip_always_covers_the_narration_slot(self):
        """The one that actually matters: no seam in the middle of a line.

        The grid is 17k+5, so any off-grid request has to round SOMEWHERE.
        Rounding to nearest made the clip shorter than the audio it carries
        (2.67s slot -> 2.33s clip on 16 of 29 real scenes), and assembly
        covers the shortfall with -stream_loop — a visible restart in the
        middle of a spoken sentence. Round up instead: the tail is trimmed
        silently, a loop is not.
        """
        fps = int(ep.CONFIG.get("h3_fps", 24))
        gap = ep.NARRATION_GAP
        # Every distinct second of narration in the real voice/ dir, plus the
        # range either side of it. Integer frames are the hard case: a slot
        # that lands exactly on a grid step must not round down by one frame.
        audited = [round(v, 2) for v in
                   (1.49, 1.62, 1.77, 1.89, 1.97, 2.09, 2.43, 2.44, 2.5, 2.65,
                    2.67, 2.69, 2.75, 2.86, 2.9, 2.94, 3.02, 3.06, 3.12, 3.15,
                    3.23, 3.51, 7.67, 7.82, 10.87)]
        for audio_sec in audited:
            slot = audio_sec + gap
            frames = _h3_frames(ep.scene_clip_seconds({"_dur": slot}), fps)
            clip_sec = frames / fps
            loops = max(0, math.ceil(slot / clip_sec) - 1)
            self.assertEqual(
                loops, 0,
                "slot %.2fs -> %.2fs clip needs %d loop(s); a seam lands "
                "mid-sentence" % (slot, clip_sec, loops))

    def test_up_to_the_cap_no_looping_at_all(self):
        """Everything under the cap must be a single take, no exceptions."""
        fps = int(ep.CONFIG.get("h3_fps", 24))
        cap = float(ep.CONFIG.get("max_scene_clip_sec", 12))
        for centis in range(100, int(cap * 100), 7):
            slot = centis / 100.0
            clip = _h3_frames(ep.scene_clip_seconds({"_dur": slot}), fps) / fps
            self.assertGreaterEqual(clip, slot - 1e-9,
                                    "%.2fs slot produced a shorter clip" % slot)

    def test_over_the_cap_is_expected_to_loop(self):
        # Past the cap the clip is deliberately one take and assembly loops
        # it. This is the known cost of the cap, not an accident — assert it
        # so the trade stays visible instead of being rediscovered later.
        fps = int(ep.CONFIG.get("h3_fps", 24))
        cap = float(ep.CONFIG.get("max_scene_clip_sec", 12))
        slot = cap + 5.0
        clip = _h3_frames(ep.scene_clip_seconds({"_dur": slot}), fps) / fps
        self.assertLessEqual(clip, cap + 17 / fps + 1e-9)
        self.assertGreater(max(0, math.ceil(slot / clip) - 1), 0)


class TestBeatGateFollowsAudio(unittest.TestCase):
    """The mid-scene still costs a render; it has to earn its place."""

    def _gate(self, beats, dur):
        minimum = float(ep.CONFIG.get("beat_min_sec", 6))
        return len(beats) >= 2 and dur >= minimum

    def test_short_two_sentence_scene_gets_no_second_still(self):
        self.assertFalse(self._gate(["a.", "b."], 3.0))

    def test_long_two_sentence_scene_does(self):
        self.assertTrue(self._gate(["a.", "b."], 9.0))

    def test_single_sentence_never_gets_a_beat(self):
        self.assertFalse(self._gate(["only one."], 30.0))


if __name__ == "__main__":
    unittest.main()