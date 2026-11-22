#!/usr/bin/env python3
"""Verify anime pipeline prerequisites (config, dirs, chapters, characters).

ponytail: this was a `main()` full of bare asserts. `unittest discover`
registered ZERO tests from it, so `python -m unittest discover` never checked
any of it and a failure reported as a shell exit code with no test name. Now a
TestCase — and each check is a real assertion rather than a print.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CONFIG_FILE = Path(__file__).parent / "config.json"


def _cfg_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


REQUIRED_CONFIG_KEYS = ("llm_url", "comfyui_url", "tts_url", "output_dir",
                        "chapters_dir", "images_dir", "scenes_dir")
REQUIRED_DIR_KEYS = ("chapters_dir", "images_dir", "scenes_dir", "output_dir")


class PrerequisiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONFIG_FILE, encoding="utf-8") as f:
            cls.config = json.load(f)

    def test_config_has_every_required_key(self):
        missing = [k for k in REQUIRED_CONFIG_KEYS if k not in self.config]
        self.assertEqual(missing, [], f"missing config keys: {missing}")

    def test_config_declares_no_dead_keys(self):
        """`tts_engine` had exactly one occurrence repo-wide — its own
        definition. A config key nothing reads is a lie about what is live."""
        self.assertNotIn(
            "tts_engine", self.config,
            "tts_engine has no readers; the engine is fixed at Qwen3-TTS")

    def test_output_directories_exist_or_can_be_created(self):
        for key in REQUIRED_DIR_KEYS:
            p = _cfg_path(self.config[key])
            p.mkdir(parents=True, exist_ok=True)
            self.assertTrue(p.is_dir(), f"{key} is not a directory: {p}")

    def test_chapter_files_present(self):
        chapters = sorted(_cfg_path(self.config["chapters_dir"]).glob("*.txt"))
        self.assertTrue(chapters, "no chapter files found")

    def test_character_bible_has_characters(self):
        with open(_cfg_path(self.config["characters_file"]), encoding="utf-8") as f:
            bible = json.load(f)
        self.assertGreater(len(bible.get("characters", {})), 0,
                           "no characters defined")


if __name__ == "__main__":
    unittest.main(verbosity=2)
