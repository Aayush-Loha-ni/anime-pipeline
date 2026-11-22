import unittest

from pipeline.viral_narration import (
    detect_triggers,
    fk_grade,
    has_open_loop,
    scene_grip,
    score_hook,
    score_narration,
    viral_prompt_block,
)


def _scenes(lines):
    return [{"scene_number": i + 1, "type": "action", "narration": t}
            for i, t in enumerate(lines)]


class ViralNarrationTests(unittest.TestCase):
    def test_hook_triggers_detected(self):
        self.assertIn("question", detect_triggers("Can you see it?"))
        self.assertIn("negation", detect_triggers("Not a dream."))
        self.assertIn("specificity", detect_triggers("73% of them never came back."))
        self.assertIn("story", detect_triggers("He ran through the white fog."))
        self.assertIn("curiosity", detect_triggers("The hidden truth about the tower."))
        self.assertEqual(detect_triggers("The sky was white."), [])

    def test_strong_hook_scores_high_weak_low(self):
        good, trig = score_hook("Can you see it? He ran — and the tower kept its secret.")
        self.assertGreaterEqual(good, 70)
        self.assertIsNotNone(trig)
        bad, trig2 = score_hook("The sky was white.")
        self.assertLess(bad, 45)
        self.assertIsNone(trig2)

    def test_forbidden_opener_penalized(self):
        banned, _ = score_hook("In this video we will explore the city.")
        plain, _ = score_hook("We will explore the city.")
        self.assertLess(banned, plain)

    def test_long_scene_and_missing_rehook_flagged(self):
        from pipeline.viral_narration import MAX_WORDS
        # one word over the cap — the pacing warning must fire at the limit,
        # not at some number a test hardcoded and the code forgot.
        over = " ".join(["word"] * (MAX_WORDS + 1))
        scenes = _scenes([
            "Can you see it?",
            over,
            "He walked on.",
        ])
        res = score_narration(scenes)
        by = {(f["scene"], f["check"]): f["severity"] for f in res["findings"]}
        self.assertEqual(by.get((2, "pacing")), "warning")
        self.assertNotIn((1, "retention"), by)  # ends on "?" = open loop
        self.assertLess(res["score"], 100)

    def test_final_scene_must_ask(self):
        res = score_narration(_scenes(["Can you see it?", "He walked home quietly."]))
        self.assertTrue(any(f["check"] == "retention" and f["scene"] == 2
                            for f in res["findings"]))

    def test_clean_script_scores_full(self):
        res = score_narration(_scenes([
            "Can you see it? The tower kept its secret...",
            "He ran through the white fog, but something was wrong...",
            "Who was watching him?",
        ]))
        self.assertEqual(res["score"], 100)

    def test_rehook_proxy_is_one_note_and_does_not_price_the_score(self):
        """The re-hook check is a keyword proxy, so it must not become the score.

        ch020 emitted a "No re-hook at the tail" info on 33 of its 40 lines
        and each one cost 2 points: the review score came out 31/100 on a
        script that holds a name back for thirteen scenes. The rate is still
        reported - as ONE note - but it is no longer priced per scene.
        """
        scenes = _scenes([f"The clerk filed the report number {i}."
                          for i in range(1, 41)])
        res = score_narration(scenes)
        retention = [f for f in res["findings"] if f["check"] == "retention"]
        # scene 0 is the episode-level aggregate; only the final-scene note
        # may name a scene. The old code named 39 of them.
        self.assertLessEqual(len([f for f in retention if f["scene"]]), 1,
                             f"{len(retention)} retention notes for 40 lines")
        self.assertGreater(res["score"], 60,
                           "the review score is the re-hook proxy rate in "
                           "disguise")

    def test_readability_flags_hard_text(self):
        grade = fk_grade("The sky was white.")
        self.assertLess(grade, 6.5)
        hard = fk_grade("The antidisestablishmentarianism controversy incomprehensibly proliferated.")
        self.assertGreater(hard, 6.5)

    def test_open_loop_and_grip(self):
        self.assertTrue(has_open_loop("And then he saw it..."))
        self.assertTrue(has_open_loop("Who was watching?"))
        self.assertFalse(has_open_loop("He walked home quietly."))
        self.assertEqual(scene_grip("Can you see it?", True), "question")
        self.assertEqual(scene_grip("He walked home quietly."), "story")
        self.assertEqual(scene_grip("The canals were quiet."), "weak")

    def test_review_integration_warns_without_invalidating(self):
        from pipeline.script_reviewer import review_script
        scenes = [{"scene_number": 1, "type": "action", "narration": "The sky was white."},
                  {"scene_number": 2, "type": "action", "narration": "He walked on."}]
        result = review_script(scenes, 1)
        self.assertTrue(result.valid)  # viral = warning/info only
        self.assertTrue(any(i.field in ("hook", "retention") for i in result.issues))

    def test_prompt_block_has_menu_and_bans(self):
        from pipeline.viral_narration import MAX_WORDS
        block = viral_prompt_block()
        # The line cap is the viewer's whole information channel: the frame
        # shows ONE image, so the narration carries the beat. The prompt must
        # say so, with the measured number, or the model keeps writing
        # 12-word fragments.
        for token in ("question", "negation", "BANNED",
                      "40-70 words",
                      "voice carries the WHOLE beat"):
            self.assertIn(token, block)

    def test_gate_blocks_structural_failure(self):
        import episode_producer as ep
        with self.assertRaisesRegex(RuntimeError, "script review failed"):
            ep.gate_script_review(None, 1)

    def test_gate_blocks_voiceless_script(self):
        import episode_producer as ep
        scenes = [{"scene_number": 1, "type": "action", "narration": ""},
                  {"scene_number": 2, "type": "action", "dialogue": ""}]
        with self.assertRaisesRegex(RuntimeError, "no scene has narration"):
            ep.gate_script_review(scenes, 1)

    def test_gate_passes_clean_script(self):
        import episode_producer as ep
        score = ep.gate_script_review(_scenes([
            "Can you see it? The tower kept its secret...",
            "He ran through the white fog, but something was wrong...",
            "Who was watching him?",
        ]), 1)
        self.assertEqual(score, 100)

    def test_gate_viral_score_warns_only(self):
        import episode_producer as ep
        score = ep.gate_script_review(_scenes([
            "The sky was white.",
            "He walked on.",
        ]), 1)
        self.assertLess(score, 100)  # no raise: taste never fails a render


if __name__ == "__main__":
    unittest.main()
