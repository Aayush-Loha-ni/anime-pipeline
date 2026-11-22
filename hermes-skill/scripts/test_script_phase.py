"""The script phase must survive a malformed answer, must not start a
second llama-server, and must not flatten the episode's camera language.

ch019 (2026-10-22): three consecutive 40-scene answers were rejected by
json.loads, each cost ~10 minutes on the CPU-only llama-server, and the
render died at "Script generation failed 3x". Separately, the dashboard and
the watchdog each launched a llama-server 25s apart; the loser never served
and stranded 7.2GB of RAM on a 31GB box.

Every assertion here fails without the fix it pins.
"""
import collections
import json
import unittest
from unittest.mock import patch

import episode_producer as ep


def _scene(i):
    return {"scene_number": i, "type": "action", "characters": ["edvyn_crael"],
            "narration": f"Line number {i} says something new.",
            "dialogue": "", "image_prompt": "close-up, dim light, anime",
            "stages": []}


class ScriptSalvageTests(unittest.TestCase):
    def test_valid_array_parses(self):
        good = json.dumps([_scene(i) for i in range(1, 41)])
        self.assertEqual(len(ep._parse_scenes(good)), 40)

    def test_one_broken_object_still_yields_the_rest(self):
        good = json.dumps([_scene(i) for i in range(1, 41)], indent=2)
        broken = good.replace('"Line number 7 says something new."',
                              '"Line number 7 says "something" new."', 1)
        with self.assertRaises(ValueError):
            json.loads(broken)          # the whole blob is unusable
        scenes = ep._parse_scenes(broken)
        self.assertIsNotNone(scenes)
        self.assertEqual(len(scenes), 39)
        self.assertNotIn(7, [s["scene_number"] for s in scenes])

    def test_truncated_array_salvages_the_complete_objects(self):
        full = json.dumps([_scene(i) for i in range(1, 41)], indent=2)
        scenes = ep._parse_scenes(full[:int(len(full) * 0.6)])
        self.assertIsNotNone(scenes)
        self.assertGreaterEqual(len(scenes), 20)

    def test_characters_array_alone_is_not_a_scene_list(self):
        # The old fallback handed back ["edvyn_crael"] - the characters array
        # of scene 1 - and the caller called it "not a JSON array of scenes".
        broken = ('[{"scene_number": 1, "characters": ["edvyn_crael"], '
                  '"narration": "He said it."}, {"scene_number": 2, '
                  '"narration": "She said it."')
        scenes = ep._parse_scenes(broken)
        self.assertEqual([s["scene_number"] for s in scenes], [1])

    def test_salvaged_script_passes_the_probe(self):
        good = json.dumps([_scene(i) for i in range(1, 41)], indent=2)
        broken = good.replace('"Line number 7 says something new."',
                              '"Line number 7 says "something" new."', 1)
        self.assertIsNone(ep._probe_reason(broken, 40))


class LlmLaunchGuardTests(unittest.TestCase):
    def test_no_second_llama_server_while_one_is_loading(self):
        with patch.object(ep, "llm_health", return_value=False), \
                patch.object(ep, "http_get", return_value={"status": "ok",
                                                            "loaded": True}), \
                patch.object(ep, "_llm_server_pids", return_value=[4242]), \
                patch.object(ep, "_launch_detached") as launch:
            ep.ensure_services(auto_start=True, wait_s=0)
        self.assertFalse(launch.called,
                         "a second llama-server was launched while one was "
                         "still loading - it loses the bind race and strands "
                         "its model in RAM")

    def test_llm_is_launched_when_no_server_exists(self):
        with patch.object(ep, "llm_health", return_value=False), \
                patch.object(ep, "http_get", return_value={"status": "ok",
                                                            "loaded": True}), \
                patch.object(ep, "_llm_server_pids", return_value=[]), \
                patch.object(ep, "_launch_detached") as launch:
            ep.ensure_services(auto_start=True, wait_s=0)
        self.assertTrue(launch.called)


class ShotDiversityTests(unittest.TestCase):
    """The diversity enforcer must SPREAD the pool, not fill it front to back.

    ch019 shipped 16 push-in + 16 pull-out and zero static / handheld /
    dutch angles: "the first pool value with room left" filled one value to
    the cap, then the next, so a 40-scene episode used two camera moves.
    """

    @staticmethod
    def _flat_scenes(n=40):
        return [{"scene_number": i, "camera_move": "static", "angle": "eye-level",
                 "mood": "tense", "shot_size": "medium", "animation_hint": "static",
                 "image_prompt": "a room", "characters": [], "narration": "n"}
                for i in range(1, n + 1)]

    def test_camera_moves_use_the_whole_pool(self):
        out = ep._enforce_shot_diversity(self._flat_scenes())
        moves = collections.Counter(s["camera_move"] for s in out)
        self.assertGreaterEqual(len(moves), 5,
                                f"only {len(moves)} camera moves used: {moves}")
        cap = int(len(out) * 0.4)
        self.assertTrue(all(v <= cap for v in moves.values()), moves)

    def test_angles_use_the_whole_pool(self):
        out = ep._enforce_shot_diversity(self._flat_scenes())
        angles = collections.Counter(s["angle"] for s in out)
        self.assertGreaterEqual(len(angles), 5,
                                f"only {len(angles)} angles used: {angles}")


class NewCharacterDetectionTests(unittest.TestCase):
    """A chapter names people it never SHOWS; they must still be learned.

    ch020 put Aldric Grent in 5 scenes and Corvin Hest in 1 - both
    off-screen (a newspaper headline, a spoken name, a death report) - so the
    on-screen cast was all-known and the learning call was skipped entirely.
    Grent recurs through ch044, so the chapter that finally shows him would
    inherit whatever face the model invents that day.
    """

    def test_off_screen_character_is_learned_with_an_all_known_cast(self):
        payload = json.dumps({"aldric_grent": {
            "full_name": "Aldric Grent", "role": "Assemblyman",
            "danbooru_tags": "1boy, middle aged, grey hair, dark suit",
            "prompt_tags": "a middle aged man with grey hair in a dark suit"}})
        with patch.object(ep, "llm_chat", return_value=payload) as chat, \
             patch.object(ep, "dump_json_atomic"):
            added = ep._learn_new_characters(["edvyn_crael"], None, 20)
        self.addCleanup(ep.CHAR_BIBLE.get("characters", {}).pop,
                        "aldric_grent", None)
        self.assertTrue(chat.called,
                        "the learning call was skipped - a chapter whose "
                        "on-screen cast is all-known never asks about anyone "
                        "else")
        self.assertIn("aldric_grent", added)

    def test_a_known_character_is_never_re_added_under_another_key(self):
        payload = json.dumps({"edvyn": {
            "full_name": "Edvyn Crael", "danbooru_tags": "1boy, dark hair",
            "prompt_tags": "a man with dark hair"}})
        with patch.object(ep, "llm_chat", return_value=payload), \
             patch.object(ep, "dump_json_atomic"):
            added = ep._learn_new_characters(["edvyn_crael"], None, 20)
        self.addCleanup(ep.CHAR_BIBLE.get("characters", {}).pop, "edvyn", None)
        self.assertNotIn("edvyn", added,
                         "a known character re-described under another key "
                         "would fork the bible")


if __name__ == "__main__":
    unittest.main()
