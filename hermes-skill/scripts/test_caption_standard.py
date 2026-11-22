"""Captions must meet a published standard, not a taste.

These are the checks that would have caught the old behaviour: fixed 3-word
chunking emitted duplicate words across a split, no line-length or reading-speed
limit, a 0.2s flash for one-word cues, and a hardcoded 48px that vanished
against the footage's blacks.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import episode_producer as ep


def parse_ass(path):
    """[(start_s, end_s, style, text)] from an ASS the producer wrote."""
    out = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        p = line.split(",", 9)

        def sec(t):
            h, m, s = t.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)

        body = re.sub(r"\{[^}]*\}", "", p[9])
        out.append((sec(p[1]), sec(p[2]), p[3], body.replace("\\N", " ")))
    return out


class TestCaptionStandard(unittest.TestCase):
    def test_cpl_never_exceeded(self):
        for words in (
            "The mill burned for the third night and Edvyn Crael stood watching "
            "the beams fall like dying men",
            "By dawn the mill would be ash and so would the evidence",
            "A beam gave way behind them with a sound like a breaking spine",
        ):
            for line in ep._cap_wrap(words.split()):
                self.assertLessEqual(len(line), ep.CAP_CPL,
                                     "%r is %d chars" % (line, len(line)))

    def test_wrap_respects_the_character_limit_per_line(self):
        # More than two lines is allowed only when the text genuinely cannot
        # fit two; what must never happen is a line over the limit.
        long_text = ("one two three four five six seven eight nine ten "
                     "eleven twelve thirteen fourteen fifteen sixteen")
        for line in ep._cap_wrap(long_text.split()):
            self.assertLessEqual(len(line), ep.CAP_CPL)

    def test_short_text_never_exceeds_two_lines(self):
        for words in (
            "By dawn the mill would be ash",
            "The canal was black and silent",
            "He looked at a stranger at himself",
        ):
            self.assertLessEqual(len(ep._cap_wrap(words.split())),
                                 ep.CAP_MAX_LINES)

    def test_line_never_starts_with_a_function_word(self):
        for words in (
            "the man who came back was never the same",
            "and so the fire spread into the mill",
            "he looked at her and she did not move",
        ):
            for line in ep._cap_wrap(words.split())[1:]:
                first = line.split()[0].lower().strip(",.;:")
                self.assertNotIn(first, ep._CAP_NO_LEAD,
                                 "line starts with %r" % first)

    def test_short_line_is_one_cue_not_a_word_chop(self):
        # The old 3-word chunking turned this into three cues.
        text = "The sky was a flat, dead white,"
        self.assertEqual(ep._cap_split(text, 2.8), [(text, 1.0)])

    def test_split_never_duplicates_or_loses_a_word(self):
        text = ("The mill burned for the third night, and Edvyn Crael stood "
                "watching the beams fall like dying men. Nobody in the "
                "village had come out to help him.")
        for window in (0.5, 1.0, 2.0, 5.0):
            parts = ep._cap_split(text, window)
            joined = " ".join(t for t, _s in parts).split()
            self.assertEqual(joined, text.split(),
                             "window %.1fs lost or duplicated words" % window)

    def test_split_shares_of_the_window_sum_to_one(self):
        text = ("The mill burned for the third night. Nobody in the village "
                "had come out to help him. He watched the beams fall.")
        parts = ep._cap_split(text, 0.5)
        self.assertGreater(len(parts), 1, "should have split")
        self.assertAlmostEqual(sum(s for _t, s in parts), 1.0, places=6)

    def test_cap_font_size_is_bounded_by_frame_height(self):
        for w, h in ((1920, 1080), (1080, 1920), (1280, 720), (3840, 2160)):
            size = ep.cap_font_size(w, h)
            self.assertGreaterEqual(size, 16)
            self.assertLessEqual(size, int(h * 0.08),
                                 "%dx%d gave %dpx" % (w, h, size))

    def test_font_size_grows_with_resolution(self):
        self.assertGreater(ep.cap_font_size(3840, 2160),
                           ep.cap_font_size(1920, 1080))


class TestCueEnforcement(unittest.TestCase):
    """The standard is applied AFTER cues are built, so test it on real output."""

    def setUp(self):
        scenes = [
            {"scene_number": 1, "narration": "The sky was a flat, dead white,",
             "_dur": 3.58, "_cap_start": 0.0, "_cap_dur": 3.58, "_lead": 0.05},
            {"scene_number": 2, "narration": "Edvyn walked the streets like a ghost.",
             "_dur": 3.75, "_cap_start": 3.58, "_cap_dur": 3.75, "_lead": 0.05},
            {"scene_number": 3,
             "narration": "By dawn the mill would be ash, and so would the "
                          "evidence. He walked the riverbank once, counting "
                          "the embers, and made his decision before the first "
                          "grey light touched the water.",
             "_dur": 3.63, "_cap_start": 7.33, "_cap_dur": 3.63, "_lead": 0.05},
            {"scene_number": 4, "narration": "He opened it.",
             "_dur": 2.50, "_cap_start": 10.96, "_cap_dur": 2.50, "_lead": 0.05},
        ]
        out = Path(self.enterContext(__import__("tempfile").TemporaryDirectory())) \
            / "subs.ass"
        ep.create_ass_subtitles(scenes, out)
        self.cues = parse_ass(out)

    def test_every_cue_meets_the_minimum_duration(self):
        for s, e, _st, tx in self.cues:
            self.assertGreaterEqual(e - s, ep.CAP_MIN_DUR - 1e-6,
                                    "%r only %.2fs" % (tx, e - s))

    def test_no_cue_exceeds_the_maximum_duration(self):
        for s, e, _st, tx in self.cues:
            self.assertLessEqual(e - s, ep.CAP_MAX_DUR + 1e-6,
                                 "%r ran %.2fs" % (tx, e - s))

    def test_cps_is_enforced_where_it_can_be(self):
        """Timing can only fix CPS when the text fits its slot.

        Scene 3 in the fixture carries ~170 characters into a 3.63s slot, so
        ~47 CPS is the floor for the scene as a whole and no per-cue timing
        can beat it. Feasibility is therefore a SCENE property, not a cue one:
        a cue inside an over-long scene may still be slow while every cue in a
        scene that fits is brought under the ceiling.
        """
        # group the cues by the scene they came from
        scenes = [(0.0, 3.58), (3.58, 7.33), (7.33, 10.96), (10.96, 13.46)]
        for lo, hi in scenes:
            mine = [c for c in self.cues if lo - 1e-6 <= c[0] < hi]
            if not mine:
                continue
            window = max(c[1] for c in mine) - min(c[0] for c in mine)
            chars = sum(len(c[3]) for c in mine)
            if chars / ep.CAP_CPS_TARGET <= window:
                for _s, e, _st, tx in mine:
                    cps = len(tx) / max(0.01, e - _s)
                    self.assertLessEqual(
                        cps, ep.CAP_CPS_MAX + 0.5,
                        "%r ran %.1f CPS in a scene that could have been fixed"
                        % (tx, cps))

    def test_over_speed_cue_is_reported_not_hidden(self):
        """The scene that cannot meet the ceiling must log a WARN.

        Silently shipping a 60 CPS cue is exactly the failure this standard
        exists to prevent, so the report is the assertion.
        """
        import io as _io
        import contextlib as _ctx
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            ep.log = lambda *a, **k: buf.write(" ".join(str(x) for x in a) + "\n")
            try:
                scenes = [{
                    "scene_number": 1,
                    "narration": "By dawn the mill would be ash, and so would "
                                 "the evidence. He walked the riverbank once, "
                                 "counting the embers, and made his decision "
                                 "before the first grey light touched it.",
                    "_dur": 3.63, "_cap_start": 0.0, "_cap_dur": 3.63,
                    "_lead": 0.05,
                }]
                out = Path(self.enterContext(__import__("tempfile")
                                             .TemporaryDirectory())) / "s.ass"
                ep.create_ass_subtitles(scenes, out)
            finally:
                import importlib
                importlib.reload(ep)
        self.assertIn("CPS", buf.getvalue(),
                      "an unsatisfiable scene was not reported")

    def test_cues_do_not_overlap(self):
        for (s1, e1, _a, t1), (s2, _e2, _b, t2) in zip(self.cues, self.cues[1:]):
            self.assertGreaterEqual(s2, e1 - 1e-6,
                                    "%r overlaps %r" % (t1, t2))

    def test_gaps_are_at_least_two_frames(self):
        fps = int(ep.CONFIG.get("fps", 24))
        for (s1, e1, _a, _t1), (s2, _e2, _b, _t2) in zip(self.cues, self.cues[1:]):
            gap = (s2 - e1) * fps
            self.assertGreaterEqual(gap + 1e-6, ep.CAP_GAP_FRAMES,
                                    "gap of %.2f frames between %r and %r"
                                    % (gap, _t1, _t2))

    def test_no_cue_is_entirely_upper_case(self):
        # Tinker 1955: all-caps costs 9.5-19% reading speed. Sentence case only.
        for _s, _e, _st, tx in self.cues:
            letters = [c for c in tx if c.isalpha()]
            if len(letters) > 3:
                self.assertFalse(all(c.isupper() for c in letters),
                                 "cue is all caps: %r" % tx)

    def test_a_fit_line_produces_exactly_one_cue(self):
        # Scenes 1, 2 and 4 each fit; only scene 3 is over-long.
        by_start = {}
        for s, e, _st, tx in self.cues:
            by_start.setdefault(round(s, 3), []).append(tx)
        self.assertEqual(len(by_start[0.0]), 1)
        self.assertEqual(len(by_start[3.58]), 1)

    def test_no_cue_starts_before_the_previous_one_ends(self):
        starts = [round(s, 6) for s, _e, _st, _t in self.cues]
        self.assertEqual(starts, sorted(starts))


if __name__ == "__main__":
    unittest.main()