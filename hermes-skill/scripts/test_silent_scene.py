"""A scene with no words must never reach the voice phase.

ch017 s4 was an establishing wide with an empty narration AND an empty
dialogue. The episode is voiced scene by scene, so that scene had no audio
and the voice gate aborted the whole render 20 minutes later:
"[FAIL] Voiceless scenes [4] - aborting (never ship silence)".

The script layer now rejects the blank scene, so the generator retries
instead of the render dying, and a cached script that cannot be voiced is
regenerated rather than aborting with no way out.
"""
import unittest

import episode_producer as ep


def _scene(n, narration="", dialogue=""):
    return {"scene_number": n, "narration": narration, "dialogue": dialogue,
            "characters": ["edvyn_crael"], "shot_size": "medium"}


class SilentSceneTests(unittest.TestCase):
    def test_validate_rejects_a_silent_scene(self):
        scenes = [_scene(1, "He burned the note."),
                  _scene(2, "The market was emptying."),
                  _scene(3, "", ""),
                  _scene(4, "He reached the gate.")]
        bad = ep._validate_scenes(scenes, 4)
        self.assertIsNotNone(bad, "a scene with no narration and no dialogue "
                                  "must fail the script gate")
        self.assertIn("3", bad)

    def test_validate_accepts_a_scene_with_dialogue_only(self):
        scenes = [_scene(1, "He burned the note."),
                  _scene(2, "", "Precautionary."),
                  _scene(3, "He reached the gate.")]
        self.assertIsNone(ep._validate_scenes(scenes, 3))

    def test_parse_probe_rejects_a_silent_scene(self):
        import json
        scenes = [_scene(1, "He burned the note."),
                  _scene(2, "The market was emptying."),
                  _scene(3, "", ""),
                  _scene(4, "He reached the gate.")]
        blob = json.dumps(scenes)
        self.assertFalse(ep._parse_probe(blob, 4),
                         "the generator must retry a script with a blank scene")

    def test_parse_probe_accepts_a_fully_voiced_script(self):
        import json
        scenes = [_scene(n, f"Line {n} of the episode.") for n in range(1, 5)]
        self.assertTrue(ep._parse_probe(json.dumps(scenes), 4))


if __name__ == "__main__":
    unittest.main()
