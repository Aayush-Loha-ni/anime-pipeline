"""Tests for narration direction.

The module is pure text-in/text-out, so all of this needs no GPU and no TTS
server. That matters: the effects it controls are SMALLER than this engine's
generation noise (measured: +/-0.073 UTMOS between two samples of the same
config), so a behavioural test against synthesised audio would be testing noise.
These pin the DECISIONS instead.
"""

import unittest

from pipeline import narration_direction as ND


class LineRateTests(unittest.TestCase):
    def test_a_hook_runs_faster_than_the_body(self):
        """Highest-impact published rule: the opening line carries more energy."""
        hook = ND.line_rate("The sky here had no sun at all.", is_first_segment=True)
        body = ND.line_rate("The sky here had no sun at all.")
        self.assertGreater(hook, body)
        self.assertEqual(hook, ND._HOOK_FAST)

    def test_the_hook_rule_only_applies_to_a_short_opening(self):
        """A long opening is exposition, not a hook; speeding it up reads as
        breathless rather than urgent.

        It falls through to the long-line rule, so the value is the slow one -
        what matters is that it is not the hook speed.
        """
        long_open = "x " * 80
        self.assertNotEqual(ND.line_rate(long_open, is_first_segment=True),
                            ND._HOOK_FAST)
        self.assertEqual(ND.line_rate(long_open, is_first_segment=True),
                         ND._LONG_SLOW)

    def test_a_long_expository_line_eases_back(self):
        slow = " ".join(["word"] * 40)
        self.assertLess(ND.line_rate(slow), 1.0)

    def test_a_middling_line_is_left_alone(self):
        mid = " ".join(["word"] * 15)
        self.assertEqual(ND.line_rate(mid), 1.0)

    def test_rates_are_bounded(self):
        for text in ("", "a", "x " * 500, "Word " * 30):
            r = ND.line_rate(text)
            self.assertGreaterEqual(r, 0.90)
            self.assertLessEqual(r, 1.10)


class PayoffTests(unittest.TestCase):
    def test_a_strong_payoff_counts_anywhere(self):
        self.assertTrue(ND.has_payoff("Nobody had ever seen anything like it."))
        self.assertTrue(ND.has_payoff("Nothing changed."))
        self.assertTrue(ND.has_payoff("He was never going to forgive her."))

    def test_a_weak_payoff_counts_only_in_the_tail(self):
        # "until" at the end is the point
        self.assertTrue(ND.has_payoff(
            "He waited until the very last second to turn away."))
        # "still" leading is weather, not a beat
        self.assertFalse(ND.has_payoff("It was still raining when he left the city."))

    def test_an_ordinary_line_is_not_a_payoff(self):
        self.assertFalse(ND.has_payoff("The street that received him was narrow."))
        self.assertFalse(ND.has_payoff("This was different."))

    def test_empty_and_tiny_inputs_are_safe(self):
        for t in ("", "   ", "No."):
            self.assertFalse(ND.has_payoff(t))


class EmphasisTests(unittest.TestCase):
    """Emphasis by capitalisation was CUT. These pin the no-op and the reasoning.

    Kept as tests rather than deleted with the feature so a future attempt at
    emphasis re-reads why it failed before spending an afternoon rediscovering it.
    """

    def test_emphasis_is_a_documented_noop(self):
        for t in ("Nobody moved. Everything stayed exactly where it was.",
                  "The door opened and NOTHING came through it.",
                  "Nothing changed.", "", "   "):
            self.assertEqual(ND.emphasise(t), t.strip())

    def test_it_never_reintroduces_capitals(self):
        """The specific regression: clean_tts_text de-shouts ALL-CAPS hook
        words, and capitalising a payoff put them straight back."""
        shouted = "ALWAYS watch the wings, they never stop."
        de_shouted = shouted[0] + shouted[1:].lower()
        self.assertEqual(ND.emphasise(de_shouted).count("NEVER"), 0)

    def test_the_reason_is_recorded_in_the_source(self):
        import inspect
        src = inspect.getsource(ND.emphasise)
        self.assertIn("no-op", src.lower())
        # the two findings that actually justify the cut
        self.assertIn("fragment", src.lower(),
                      "the tokeniser-fragmentation finding must stay recorded")
        self.assertIn("drop", src.lower(),
                      "the dropped-word measurement must stay recorded")


class GapTests(unittest.TestCase):
    def test_a_payoff_gets_a_longer_beat(self):
        self.assertGreater(ND.gap_after("Nobody had ever seen it."),
                           ND.gap_after("The street was narrow."))

    def test_a_short_line_keeps_its_momentum(self):
        short = ND.gap_after("This was different.")
        ordinary = ND.gap_after("The street that received him was narrow.")
        self.assertLess(short, ordinary)

    def test_the_last_segment_gets_no_trailing_gap(self):
        """Padding the end of a scene only moves the next scene's start, which
        assembly already positions."""
        self.assertEqual(ND.gap_after("Nobody moved.", is_last_segment=True), 0.0)

    def test_gaps_are_not_all_the_same(self):
        """The metronome effect is precisely that every join is identical."""
        gaps = {ND.gap_after(t) for t in (
            "This was different.",
            "The street that received him was narrow and the light was gone.",
            "Nobody had ever seen anything like it.")}
        self.assertGreater(len(gaps), 1)


class DirectAllTests(unittest.TestCase):
    def test_a_single_segment_keeps_the_users_speed_exactly(self):
        """There is no rhythm to vary in one line, and changing it would silently
        rewrite an explicit voice_speed."""
        for base in (0.9, 1.0, 1.25):
            plan = ND.direct_all(["Spoken words"], base_rate=base)
            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0]["rate"], base)

    def test_multiple_segments_vary_but_honour_the_base(self):
        plan = ND.direct_all(["The sky here had no sun.", "It was flat."],
                             base_rate=1.25)
        rates = [p["rate"] for p in plan]
        self.assertGreater(len(set(rates)), 1, "no variation at all")
        for r in rates:
            # every line is faster than the base, because every line's per-line
            # rate is >= 1.0; the base must never be silently reduced
            self.assertGreaterEqual(r, 1.24)

    def test_the_first_segment_is_flagged_and_the_last_has_no_gap(self):
        plan = ND.direct_all(["One thing happened here.", "Then another thing."],
                             base_rate=1.0)
        self.assertEqual(plan[0]["index"], 0)
        self.assertEqual(plan[-1]["gap_after"], 0.0)
        self.assertGreater(plan[0]["gap_after"], 0.0)

    def test_it_is_a_pure_function_of_the_text(self):
        segs = ["A line of narration here.", "And a second one after it."]
        a = ND.direct_all(segs, base_rate=1.0)
        b = ND.direct_all(segs, base_rate=1.0)
        self.assertEqual(a, b)


class PipelineWiringTests(unittest.TestCase):
    """The wiring must actually happen, or the module is dead code."""

    def test_the_producer_imports_and_uses_it(self):
        import inspect
        import episode_producer as ep
        src = inspect.getsource(ep.generate_tts)
        self.assertIn("narration_direction", src,
                      "generate_tts must import narration_direction")
        self.assertIn("_narrate.direct_all", src,
                      "generate_tts must build a plan from the segments")
        self.assertIn("gaps=", src,
                      "generate_tts must pass the per-line gaps to _concat_wavs")

    def test_concat_wavs_accepts_per_join_gaps(self):
        import inspect
        import episode_producer as ep
        sig = inspect.signature(ep._concat_wavs)
        self.assertIn("gaps", sig.parameters)

    def test_the_server_no_longer_discards_speed(self):
        import pathlib
        server = pathlib.Path(__file__).resolve().parents[2] / "cosy-server" / "server.py"
        src = server.read_text(encoding="utf-8")
        self.assertIn("_atempo", src,
                      "server.py must apply the requested rate; it used to "
                      "accept `speed` and ignore it, making voice_speed dead")
        self.assertNotIn("IGNORED —\n    # Qwen3-TTS has no speed control",
                         src, "the stale 'ignored' comment is back")


if __name__ == "__main__":
    unittest.main()