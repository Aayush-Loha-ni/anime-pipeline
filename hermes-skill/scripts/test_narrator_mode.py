"""Pin the narrator trade-off so it cannot be flipped on by accident."""
import sys, json, pathlib
sys.path.insert(0, ".")
COSY = pathlib.Path(r"<PROJECT_ROOT>\cosy-server")
import unittest
sys.path.insert(0, str(COSY))
import server as S

class NarratorModeTests(unittest.TestCase):
    """A preset narrator voice, opt-in.

    Measured on all 25 ch001 lines, same engine, same post-processing:
        user's own clone      UTMOS 3.41   pitch 10.1 st   WER 19.9%
        narrator "aiden"      UTMOS 4.30   pitch 13.0 st   WER  7.8%
    The audiobook instruct prompt roughly doubles pitch range on the same
    speaker (13.2 -> 20.1 st on the calibration set).

    THE TRADE: it is a preset voice, not the user's own. More natural and more
    intelligible, and a different person. So it must be opt-in.
    """

    def test_it_is_off_by_default(self):
        cfg = json.loads(S.REF_JSON.read_text(encoding="utf-8"))
        self.assertFalse((cfg.get("narrator") or {}).get("enabled"),
                         "the preset narrator must stay opt-in - it is a "
                         "different voice from the user's clone")

    def test_disabled_means_the_clone_path(self):
        self.assertIsNone(S._narrator())

    def test_the_fingerprint_covers_the_narrator_block(self):
        """Flipping narrator.enabled changes the model AND the delivery of every
        line, without touching the reference audio. Without this in the
        fingerprint, /health reports stale=false while the server keeps speaking
        with the old voice - the same failure the reference fingerprint exists to
        prevent."""
        base = json.loads(S.REF_JSON.read_text(encoding="utf-8"))
        off = S._ref_fingerprint(base)
        on = dict(base)
        on["narrator"] = {"enabled": True, "speaker": "aiden",
                          "model": S.CUSTOMVOICE_ID, "instruct": "x"}
        self.assertNotEqual(off, S._ref_fingerprint(on),
                            "narrator block is missing from the fingerprint")
        on2 = dict(on); on2["narrator"] = dict(on["narrator"]); on2["narrator"]["speaker"] = "ryan"
        self.assertNotEqual(S._ref_fingerprint(on), S._ref_fingerprint(on2),
                            "speaker change must invalidate too")
        on3 = dict(on); on3["narrator"] = dict(on["narrator"]); on3["narrator"]["instruct"] = "y"
        self.assertNotEqual(S._ref_fingerprint(on), S._ref_fingerprint(on3),
                            "instruct change must invalidate - it IS the delivery")

    def test_the_instruct_names_the_measured_gap(self):
        """A vague 'be dramatic' produced no measurable change; the effect came
        from naming the mechanism - pace varies WITHIN a line, not just line to
        line."""
        t = S.NARRATOR_INSTRUCT.lower()
        self.assertIn("vary your pace", t)
        self.assertIn("within the line", t)
        self.assertIn("story", t)

    def test_customvoice_is_the_only_variant_that_follows_instructions(self):
        self.assertIn("CustomVoice", S.CUSTOMVOICE_ID)
        self.assertNotIn("CustomVoice", S.MODEL_ID)


if __name__ == "__main__":
    unittest.main()
