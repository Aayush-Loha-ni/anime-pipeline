"""The breakdown LLM must see the whole chapter, and must still fit.

Regression for a quiet, expensive bug: `_chapter_excerpt` capped chapter text
at 12000 chars, so 123 of the 403 chapters in `chapters/` were silently cut
before the scriptwriter ever saw them. ch2 (17492 chars) lost its middle
third; the model then complained about repetition that was an artefact of
the truncation seam, which points the debugging at the wrong layer entirely.

Two failure modes worth guarding separately:
  1. the budget is silently lowered again -> chapters start losing content
  2. the budget is raised past the context window -> every call fails, or
     llama.cpp quietly truncates the tail, which looks identical to (1)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import episode_producer as ep

ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = ROOT / "novels" / "Example Novel" / "chapters"

# The breakdown prompt, measured from a real build (2026-11-13: the template
# is 11234 chars -> 2948 tokens, chapter excluded). If the prompt grows
# materially this needs re-measuring, not guessing: a stale constant here
# would let the context math drift back into overflow. 3.4 chars/token is the
# producer's own conservative ratio (English prose measures ~3.6).
PROMPT_TOKENS = 2948
CHARS_PER_TOKEN = 3.4


class TestChapterBudget(unittest.TestCase):

    def test_no_chapter_in_corpus_is_truncated(self):
        """Every chapter on disk must reach the LLM whole."""
        if not CHAPTERS.is_dir():
            self.skipTest("no chapters/ directory")
        budget = int(ep.CONFIG.get("llm_chapter_chars", 48000))
        chapters = sorted(CHAPTERS.glob("*.txt"))
        if not chapters:
            self.skipTest("chapters/ is empty")
        cut = [p.name for p in chapters if len(p.read_text(encoding="utf-8")) > budget]
        self.assertEqual(
            cut, [],
            "%d chapter(s) exceed the %d-char breakdown budget: %s"
            % (len(cut), budget, ", ".join(cut[:5])))

    def test_worst_case_fits_the_context_window(self):
        """Prompt + longest chapter + output allowance must fit in llm_ctx.

        The output allowance is the CLAMPED budget, not the configured cap:
        the cap is "never ask for more than this", and the clamp is what
        actually fits beside the prompt and the chapter.
        """
        ctx = int(ep.CONFIG.get("llm_ctx", 32768))
        budget = int(ep.CONFIG.get("llm_chapter_chars", 48000))
        out_tok = ep._script_answer_budget(budget)
        worst = PROMPT_TOKENS + budget / CHARS_PER_TOKEN + out_tok
        self.assertLess(
            worst, ctx,
            "budget of %d chars needs ~%d tok but llm_ctx is only %d; "
            "raise llm_ctx before llm_chapter_chars"
            % (budget, worst, ctx))

    def test_headroom_survives_a_bigger_chapter(self):
        """A chapter at the full budget must still leave room to answer.

        Guards the specific case where someone raises the budget right up to
        the context wall and every long chapter starts failing at once.
        """
        ctx = int(ep.CONFIG.get("llm_ctx", 32768))
        budget = int(ep.CONFIG.get("llm_chapter_chars", 48000))
        self.assertGreater(
            ctx - (PROMPT_TOKENS + budget / CHARS_PER_TOKEN), 2048,
            "less than 2k tokens of headroom at the configured budget")

    def test_truncation_still_warns_when_it_happens(self):
        """The fallback must never be silent — that was the original bug."""
        long_chapter = "A" * 200000
        excerpt = ep._chapter_excerpt(long_chapter, limit=1000)
        self.assertLess(len(excerpt), 200000)
        self.assertIn("middle omitted", excerpt)
        # head and tail both survive, so the ending is never the thing lost.
        # Split is 75/25, so with limit=1000: 750 head, 250 tail.
        self.assertTrue(excerpt.startswith("A" * 750), "head was cut")
        self.assertTrue(excerpt.endswith("A" * 250), "tail was cut")

    def test_short_chapter_passes_through_untouched(self):
        text = "short chapter body"
        self.assertEqual(ep._chapter_excerpt(text), text)


if __name__ == "__main__":
    unittest.main()