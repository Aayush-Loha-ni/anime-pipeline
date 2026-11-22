"""Who a shot is OF, and what identity the still prompt gets for it.

ch016 rendered Lena's close-up (scene 6) as TWO MEN and painted Edvyn's
face on the intruder (scenes 18/19/30). Both came from the same place: the
still prompt concatenated EVERY cast member's danbooru_tags, so a
two-person frame carried two count tokens ("1girl, solo" + "1boy, solo")
and two style tails, and the subject was guessed out of free text instead
of being asked for.

These tests pin the two answers: the SUBJECT, and the tag set.
"""
import unittest

import episode_producer as ep


LENA = {"danbooru_tags": "1girl, solo, young woman, brown eyes, auburn hair, "
                          "bun, parted bangs, pale skin, dark coat, "
                          "dark fantasy, gothic, detailed eyes"}
EDVYN = {"danbooru_tags": "1boy, solo, young man, green eyes, short hair, "
                           "brown hair, messy hair, parted bangs, pale skin, "
                           "stubble, dark coat, white shirt, dark fantasy, "
                           "gothic, detailed eyes"}
UNKNOWN = {"danbooru_tags": "1man, middle-aged man, short hair, dark eyes, "
                            "plain face, dark clothing, dark fantasy, gothic, "
                            "detailed eyes"}


class SubjectTests(unittest.TestCase):
    def setUp(self):
        self.old_bible = ep.CHAR_BIBLE
        ep.CHAR_BIBLE = {"characters": {
            "lena_ovrik": LENA, "edvyn_crael": EDVYN, "unknown_man": UNKNOWN}}

    def tearDown(self):
        ep.CHAR_BIBLE = self.old_bible

    # -- the subject --------------------------------------------------
    def test_focal_character_wins_over_script_order(self):
        # ch016 s18: the intruder in the doorway lists Edvyn FIRST because
        # Edvyn is who he looks AT. The director's plan says the scene is
        # about the intruder.
        scene = {"characters": ["edvyn_crael", "unknown_man"],
                 "image_prompt": "a man in a worn coat standing in the doorway",
                 "focal_character": "unknown_man"}
        self.assertEqual(ep._shot_subject(scene), "unknown_man")
        self.assertEqual(ep._lora_candidate_order(scene), ["unknown_man"])

    def test_focal_character_is_matched_to_the_cast_key(self):
        scene = {"characters": ["edvyn_crael", "unknown_man"],
                 "focal_character": "Edvyn"}
        self.assertEqual(ep._focal_character(scene), "edvyn_crael")

    def test_unknown_focal_character_falls_back(self):
        scene = {"characters": ["edvyn_crael"], "focal_character": "nobody"}
        self.assertEqual(ep._focal_character(scene), "")
        self.assertEqual(ep._shot_subject(scene), "edvyn_crael")

    def test_single_named_character_in_the_prompt_is_the_subject(self):
        scene = {"characters": ["lena_ovrik", "edvyn_crael"],
                 "image_prompt": "dutch tilt, Lena's eyes, sharp and assessing"}
        self.assertEqual(ep._shot_subject(scene), "lena_ovrik")

    # -- the tag set --------------------------------------------------
    def test_solo_scene_is_byte_identical_to_the_old_concatenation(self):
        # Cached solo stills must stay valid: the composed tags for one
        # character are exactly the bible's danbooru_tags.
        scene = {"characters": ["lena_ovrik"]}
        self.assertEqual(ep._character_prompt_tags(scene), LENA["danbooru_tags"])

    def test_two_character_scene_counts_both_and_never_says_solo(self):
        scene = {"characters": ["lena_ovrik", "edvyn_crael"], "shot_size": "medium"}
        tags = ep._character_prompt_tags(scene)
        self.assertEqual(tags.count("1girl"), 1)
        self.assertEqual(tags.count("1boy"), 1)
        self.assertNotIn("solo", tags.lower())

    def test_two_character_scene_carries_one_style_tail(self):
        scene = {"characters": ["lena_ovrik", "edvyn_crael"], "shot_size": "medium"}
        tags = ep._character_prompt_tags(scene).lower()
        self.assertEqual(tags.count("dark fantasy"), 1)
        self.assertEqual(tags.count("detailed eyes"), 1)

    def test_close_up_carries_only_the_subject(self):
        # ch016 s6: Lena's close-up with Edvyn in the room. Two count tokens
        # made the model draw two MEN.
        scene = {"characters": ["lena_ovrik", "edvyn_crael"],
                 "shot_size": "close-up",
                 "image_prompt": "dutch tilt, Lena's eyes, sharp and assessing"}
        tags = ep._character_prompt_tags(scene)
        self.assertIn("1girl", tags)
        self.assertNotIn("1boy", tags)

    def test_a_close_up_two_shot_keeps_the_second_character(self):
        scene = {"characters": ["edvyn_crael", "unknown_man"],
                 "shot_size": "close-up",
                 "image_prompt": "tight two-shot of faces, Edvyn's face"}
        tags = ep._character_prompt_tags(scene)
        self.assertIn("1boy", tags)
        self.assertIn("1man", tags)

    def test_the_subject_keeps_its_own_count_token(self):
        scene = {"characters": ["unknown_man", "edvyn_crael"],
                 "shot_size": "close-up"}
        tags = ep._character_prompt_tags(scene)
        self.assertIn("1man", tags)
        self.assertNotIn("1boy", tags)

    # -- director plan -------------------------------------------------
    def test_apply_director_plan_copies_the_focal_character(self):
        scenes = [{"scene_number": 1}, {"scene_number": 2}, {"scene_number": 3}]
        dp = {"scene_plan": [
            {"scene_number": 1, "focal_character": "unknown_man"},
            {"scene_number": 3, "focal_character": "lena_ovrik"}]}
        n = ep._apply_director_plan(scenes, dp)
        self.assertEqual(n, 2)
        self.assertEqual(scenes[0]["focal_character"], "unknown_man")
        self.assertEqual(scenes[2]["focal_character"], "lena_ovrik")
        self.assertNotIn("focal_character", scenes[1])

    def test_apply_director_plan_survives_a_missing_plan(self):
        scenes = [{"scene_number": 1}]
        self.assertEqual(ep._apply_director_plan(scenes, None), 0)
        self.assertEqual(ep._apply_director_plan(scenes, {}), 0)


if __name__ == "__main__":
    unittest.main()
