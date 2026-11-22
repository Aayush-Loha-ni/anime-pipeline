"""A withheld detail is a re-hook even with no cue word and no punctuation.

The gap this closes: has_open_loop() used to require FORWARD_TAILS (...?!) or
a cue word from a fixed list. So "The Registry never sent cards." - which
withholds *why*, and is exactly the restraint the prompt asks for - scored 0,
while a script that stapled "..." onto 24 of 25 lines scored 96%. The check
could only see punctuation, so it rewarded a tic over real writing.

Kept separate from _CURIOSITY_RE on purpose: that one matches forward momentum
("but", "then", "what if"), this one matches absence ("never", "unnamed",
"without"). Same question - is something withheld? - different evidence.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.viral_narration import has_open_loop


class TestWithheldDetailCountsAsReHook(unittest.TestCase):

    def test_absence_markers_in_the_tail(self):
        # real lines from the regenerated ch002 script, all with no cue word
        # and no trailing punctuation
        for line in ("The Registry never sent cards.",
                     "An unnamed buyer had purchased it.",
                     "The fire was of uncertain origin.",
                     "Her hands were too tight.",
                     "She didn't look up from her ledger."):
            with self.subTest(line=line):
                self.assertTrue(has_open_loop(line),
                                "withheld detail not recognised: %r" % line)

    def test_plain_statement_with_nothing_withheld_is_still_false(self):
        # a line that just states what happened gives the viewer no reason
        # to continue, and should not be laundered into passing
        for line in ("He woke on the floor.",
                     "He went downstairs for coffee.",
                     "She sat across from him.",
                     "The money was ready."):
            with self.subTest(line=line):
                self.assertFalse(has_open_loop(line),
                                 "flat statement wrongly counted as a re-hook")

    def test_punctuation_still_counts(self):
        for line in ("He pulled out a photograph...",
                     "What did the mother know?",
                     "Someone was already inside!"):
            with self.subTest(line=line):
                self.assertTrue(has_open_loop(line))

    def test_absent_markers_outside_the_tail_do_not_count(self):
        # "never" is in the setup, and "sign" is a resolved action: nothing
        # is being withheld at the end, so this is not a re-hook.
        # Needs >8 words before "never" to push it out of the tail window.
        self.assertFalse(has_open_loop(
            "A clerk who had never seen the file asked him to sign today"))

    def test_un_prefix_alone_does_not_count(self):
        # an \bun\w+ pattern would match these, but they are ordinary
        # scene-setting, not withheld detail
        for line in ("He went under the bridge to wait.",
                     "The meeting was unusual for her."):
            with self.subTest(line=line):
                self.assertFalse(has_open_loop(line))

    def test_empty_input(self):
        self.assertFalse(has_open_loop(""))
        self.assertFalse(has_open_loop("   "))


if __name__ == "__main__":
    unittest.main()