"""The head-in-frame gate must judge the HEAD, not the body.

ch018 s4 (Voss, whole head inside the frame) was escalated to a REVIEW in
the release report because the vision model said "the coat is cut off by
the frame" and the gate matched a bare "cut off". One scene's cosmetic
wording held the whole episode's verdict at NEEDS_REVIEW.

The verdict is also a property of the PICTURE, so it is cached beside the
still: re-asking the vision model for every cached still made a one-scene
redo pay 40 minutes to re-grade 39 stills it had already graded.
"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import episode_producer as ep


class FaceGateTests(unittest.TestCase):
    def _grade(self, issues):
        return {"grade": "B", "anime_look": 80, "composition": 70, "issues": issues}

    def _check(self, issues):
        with patch.object(ep, "vlm_available", return_value=True), \
             patch.object(ep, "vlm_grade_image", return_value=self._grade(issues)):
            return ep.face_visibility_check("x.png", 1)

    def test_a_body_crop_is_not_a_head_failure(self):
        ok, _ = self._check(["the coat is cut off by the frame"])
        self.assertTrue(ok, "a cropped coat must not fail the head gate")

    def test_a_cropped_head_still_fails(self):
        ok, _ = self._check(["the head is cut off by the top of the frame"])
        self.assertFalse(ok, "a head that IS cut off must still fail")

    def test_a_cropped_face_still_fails(self):
        ok, _ = self._check(["the face is cropped by the frame edge"])
        self.assertFalse(ok)

    def test_a_clean_head_passes(self):
        ok, _ = self._check([])
        self.assertTrue(ok)

    def test_hair_covering_the_face_is_on_model(self):
        ok, _ = self._check(["hair covering the face"])
        self.assertTrue(ok, "fringe doctrine: hair over the face is styling")


class FaceFlagCacheTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.dir = Path(tempfile.mkdtemp())
        self.still = self.dir / "ch016_s05_abc_00001_.png"
        self.still.write_bytes(b"png")

    def test_the_verdict_is_cached_beside_the_still(self):
        with patch.object(ep, "vlm_available", return_value=True), \
             patch.object(ep, "vlm_grade_image",
                          return_value={"issues": ["head cut off"]}) as grade:
            ok, why = ep._face_flag_for(self.still, 5)
        self.assertFalse(ok)
        self.assertIn("head cut off", why)
        self.assertEqual(grade.call_count, 1)
        # second call: no vision round trip
        with patch.object(ep, "vlm_grade_image",
                          side_effect=AssertionError("must not re-ask")) as grade:
            ok2, why2 = ep._face_flag_for(self.still, 5)
        self.assertFalse(ok2)
        self.assertEqual(why2, why)
        self.assertEqual(grade.call_count, 0)

    def test_the_cached_answer_is_judged_again_not_the_old_verdict(self):
        # The cache holds the model's ANSWER, so a change to the judging
        # rules takes effect at once instead of being frozen in.
        side = self.still.with_suffix(".face.json")
        side.write_text(json.dumps({"issues": ["the coat is cut off by the frame"]}),
                        encoding="utf-8")
        with patch.object(ep, "vlm_grade_image",
                          side_effect=AssertionError("must not re-ask")):
            ok, _ = ep._face_flag_for(self.still, 5)
        self.assertTrue(ok, "a body crop must not fail the head gate")

    def test_the_provenance_sidecar_is_not_touched(self):
        with patch.object(ep, "vlm_available", return_value=True), \
             patch.object(ep, "vlm_grade_image", return_value={"issues": []}):
            ep._face_flag_for(self.still, 5)
        self.assertFalse(self.still.with_suffix(".cache.json").exists(),
                         "the fingerprint sidecar must stay exactly as written")
        self.assertTrue(self.still.with_suffix(".face.json").exists())
        data = json.loads(self.still.with_suffix(".face.json").read_text())
        self.assertIn("issues", data)


if __name__ == "__main__":
    unittest.main()
