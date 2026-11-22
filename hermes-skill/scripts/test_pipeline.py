#!/usr/bin/env python3
"""Pipeline module tests.

ponytail: this file used to be a straight-line script with 17 module-level
`assert`s. `unittest discover` registered ZERO tests from it, so every one of
those assertions ran as an import-time side effect and a failure surfaced as
`ImportError: Failed to import test module` with no test name attached — the
"337 tests pass" headline silently excluded all of them. Now a real TestCase:
each assertion is counted, named, and reported independently.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SKIP_GENAI", "1")  # keep captioner offline in tests
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).parent.parent.parent
BOOK = ROOT / "novels" / "Example Novel"

from pipeline.asset_library import AssetLibrary
from pipeline.task_manager import TaskManager
from pipeline.script_reviewer import review_script
from pipeline.storyboard import generate_storyboard
from pipeline.project_audit import audit_project
from pipeline.beat_planner import beats_summary
from pipeline.camera_presets import all_presets, build_camera_modifiers
from pipeline import config as ini_config
from pipeline import utils as pipe_utils
from pipeline import captioner as captioner_mod
from pipeline import tokens as token_bucket
from reference_selector import ReferenceImageSelector


class PipelineModuleTests(unittest.TestCase):
    """Live-module smoke tests against the real project root."""

    @classmethod
    def setUpClass(cls):
        cls.lib = AssetLibrary(BOOK)
        cls.chapters = cls.lib.load_chapters()
        cls.chars = cls.lib.load_characters()
        scenes_file = BOOK / "scenes" / "ch001.json"
        cls.scenes = (json.loads(scenes_file.read_text("utf-8"))
                      if scenes_file.exists() else None)

    # -- Asset library -------------------------------------------------
    def test_asset_library_loads(self):
        self.assertGreaterEqual(len(self.chars), 0)
        self.assertIsInstance(self.chapters, dict)
        summary = self.lib.summary()
        self.assertIn("total_chapters", summary)

    def test_library_still_count_excludes_beat_and_companion_stills(self):
        """`b_`/`c_` stills are derivatives of one scene, not extra scenes.

        Counting them inflated ch001 to 38 stills for 25 scenes (+52%) and made
        the dashboard disagree with quality_evolve, which excluded them already.
        """
        if 1 not in self.chapters:
            self.skipTest("ch001 not present")
        ch = self.chapters[1]
        # The stills are a CACHE, not the artefact: they can be purged (disk
        # cleanup, a rebuild) while the published videos are intact. The
        # property under test only has meaning when the cache is there.
        if not ch.total_stills:
            self.skipTest("ch001 still cache is empty (purged); nothing to count")
        if self.scenes:
            self.assertEqual(ch.total_stills, len(self.scenes),
                             "still count must match the scene count")

    def test_library_serves_posix_paths(self):
        """Windows backslashes are escape characters in the JSON the browser gets."""
        for ch in self.chapters.values():
            if ch.video_path:
                self.assertNotIn("\\", ch.video_path)
                self.assertTrue(ch.video_path.startswith("videos/"))

    def test_library_never_picks_a_rotated_version_as_the_video(self):
        for ch in self.chapters.values():
            if ch.video_path:
                self.assertNotRegex(ch.video_path, r"\.v\d+\.mp4$")

    # -- Task manager --------------------------------------------------
    def test_task_manager_constructs(self):
        tm = TaskManager()
        self.assertGreaterEqual(tm.max_concurrent, 1)

    # -- Script reviewer ------------------------------------------------
    def test_review_ch001(self):
        if self.scenes is None:
            self.skipTest("no scenes/ch001.json")
        result = review_script(list(self.scenes), 1)
        self.assertIsInstance(result.issues, list)
        self.assertGreaterEqual(result.score, 0)

    def test_review_persists_when_scenes_dir_is_absent(self):
        """A fresh project has no scenes/ — that used to be a FileNotFoundError."""
        from pipeline import script_reviewer
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.object(script_reviewer, "ROOT", root):
                review_script([{"scene_number": None, "characters": ["x"]}],
                              1, known_characters={"x"}, persist=True)
            self.assertTrue((root / "scenes" / "ch001.json").is_file())

    # -- Storyboard / beats --------------------------------------------
    def test_storyboard_and_beats(self):
        if self.scenes is None:
            self.skipTest("no scenes/ch001.json")
        sb = generate_storyboard(1, list(self.scenes))
        self.assertGreaterEqual(len(sb.shots), 0)
        beats = beats_summary(list(self.scenes))
        self.assertIn("assignment", beats)
        self.assertIn("gaps", beats)

    # -- Project audit ---------------------------------------------------
    def test_audit_project(self):
        audit = audit_project(ROOT)
        self.assertIsInstance(audit.findings, list)

    # -- Camera presets --------------------------------------------------
    def test_camera_preset_modifiers(self):
        presets = all_presets()
        self.assertIn("lenses", presets)
        mod = build_camera_modifiers("classic-anamorphic", "35mm", "f1.4")
        self.assertIn("anamorphic", mod)
        self.assertIn("human-eye", mod)
        self.assertIn("shallow", mod)

    # -- INI config ------------------------------------------------------
    def test_ini_config_defaults(self):
        self.assertEqual(ini_config.genai_max_size(), 1024)
        self.assertEqual(ini_config.genai_default_strength(), 0.5)
        self.assertEqual(ini_config.get("Nope", "missing", "fallback"), "fallback")
        self.assertIs(ini_config.tokens_enabled(), False)  # off by default

    # -- Utils -----------------------------------------------------------
    def test_utils_sha1_dedup_and_timestamped_save(self):
        from PIL import Image
        img = Image.new("RGB", (16, 16), "red")
        with tempfile.TemporaryDirectory() as td:
            h1 = pipe_utils.image_sha1(img)
            p1 = pipe_utils.save_image_as_file(img, td)
            p2 = pipe_utils.save_image_as_file(img, td)   # dedup -> same path
            self.assertEqual(p1, p2)
            self.assertTrue(Path(p1).is_file())
            p3 = pipe_utils.save_image_with_timestamp(img, td, reference=h1[:8])
            self.assertTrue(p3 and Path(p3).is_file() and p3.endswith(".png"))
            self.assertEqual(pipe_utils.get_all_local_models(td), [])

    def test_save_image_with_timestamp_sanitises_the_reference(self):
        """`/` used to make the save return None; `:` is illegal on Windows."""
        from PIL import Image
        img = Image.new("RGB", (8, 8), "blue")
        with tempfile.TemporaryDirectory() as td:
            p = pipe_utils.save_image_with_timestamp(img, td, reference="sub/evil")
            self.assertIsNotNone(p, "a slash in the reference must not fail the save")
            self.assertTrue(Path(p).is_file())
            self.assertNotIn(":", Path(p).name)

    # -- Captioner (offline) ----------------------------------------------
    def test_captioner_offline_returns_empty(self):
        self.assertIs(captioner_mod.available(), False)
        self.assertEqual(captioner_mod.describe_image("nonexistent.png"), "")

    def test_skip_genai_honours_the_env_var_every_call(self):
        """skip_genai() must re-read the environment, not latch at import.

        ponytail: it was a module global evaluated at import and OR-ed together
        on each config read, so SKIP_GENAI=1 only took effect if the setting
        existed before pipeline.config was first imported, and the flag could
        never be turned back off. Mutation testing confirmed that reverting
        this to a latch left the whole suite green — it had zero coverage.
        """
        from pipeline import config as cfg
        with patch.dict(os.environ, {"SKIP_GENAI": "1"}):
            self.assertTrue(cfg.skip_genai(),
                            "SKIP_GENAI=1 must be honoured at call time")
            self.assertIs(captioner_mod.available(), False)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SKIP_GENAI", None)
            # No [GenAI] skip=true in the shipped app.config, so this is False —
            # and critically it CHANGED, which the old latch could not do.
            self.assertFalse(cfg.skip_genai(),
                             "clearing SKIP_GENAI must take effect immediately")

    def test_skip_genai_is_never_a_sticky_true(self):
        """The old `SKIP_GENAI = SKIP_GENAI or ...` latch meant one True could
        never be cleared."""
        from pipeline import config as cfg
        with patch.dict(os.environ, {"SKIP_GENAI": "1"}):
            self.assertTrue(cfg.skip_genai())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SKIP_GENAI", None)
            self.assertFalse(cfg.skip_genai(), "the flag latched on")

    def test_debug_enabled_reads_rather_than_latches(self):
        """`DEBUG` was a global with zero readers; it is now a function."""
        from pipeline import config as cfg
        self.assertIsInstance(cfg.debug_enabled(), bool)
        # no [General] debug in the shipped app.config, so this must be False
        # and must not latch to True if something ever sets it
        self.assertFalse(cfg.debug_enabled())

    # -- Tokens -----------------------------------------------------------
    def test_disabled_bucket_allows_spend(self):
        ok, _bal = token_bucket.spend(99)
        self.assertTrue(ok, "disabled bucket must allow spend")
        st = token_bucket.status()
        self.assertIs(st["enabled"], False)
        self.assertEqual(st["render_cost"], 1)

    def test_token_bucket_arithmetic_and_overdraft(self):
        """ponytail: this used to rebind token_bucket._file at module scope and
        never restore it, so EVERY later test in the shared unittest process
        wrote its tokens into a TemporaryDirectory that no longer existed."""
        with tempfile.TemporaryDirectory() as td:
            with patch.object(token_bucket, "_file",
                              lambda: Path(td) / "tokens.json"), \
                 patch.object(ini_config, "tokens_enabled", lambda default=False: True):
                self.assertEqual(token_bucket.balance(), 0)
                self.assertEqual(token_bucket.grant(5, "test"), 5)
                self.assertEqual(token_bucket.spend(2), (True, 3))
                self.assertEqual(token_bucket.spend(99), (False, 3))
        # And the real path is intact for whoever runs next.
        self.assertTrue(callable(token_bucket._file))
        self.assertNotEqual(str(token_bucket._file()), str(Path(td) / "tokens.json"))

    def test_token_spend_is_serialised_across_processes(self):
        """spend()/grant() must hold a CROSS-PROCESS lock, not just a thread lock.

        ponytail: the first version of this test only exercised FileLock
        against itself and PASSED with the cross-process protection deleted
        outright. A second attempt spawned a subprocess that monkeypatched
        module internals and was flaky.

        This version spies on _process_lock(), so deleting the lock from
        _locked_update makes it fail (verified by mutation), while still
        exercising the real spend path end to end.
        """
        calls = []
        real = token_bucket._process_lock

        def spy():
            lock = real()
            calls.append(lock)
            return lock

        with tempfile.TemporaryDirectory() as td:
            tokfile = Path(td) / "tokens.json"
            with patch.object(token_bucket, "_file", lambda: tokfile), \
                 patch.object(ini_config, "tokens_enabled",
                              lambda default=False: True), \
                 patch.object(token_bucket, "_process_lock", spy):
                token_bucket.grant(5, "test")
                token_bucket.spend(2)
        self.assertEqual(len(calls), 2,
                         "every spend/grant must acquire the process lock "
                         "(a threading.Lock alone cannot stop two processes "
                         "double-spending the same balance)")
        self.assertIsNotNone(calls[0].path)
        self.assertEqual(calls[0].path.name, "tokens.json.lock",
                         "the lock must be a sidecar, and must APPEND rather "
                         "than with_suffix a dotted name")

    def test_file_lock_refuses_a_second_holder(self):
        """Same process, nested: must raise BlockingIOError, not deadlock."""
        from json_util import FileLock
        with tempfile.TemporaryDirectory() as td:
            lf = Path(td) / "x.lock"
            with FileLock(lf, blocking=False):
                with self.assertRaises(BlockingIOError):
                    with FileLock(lf, blocking=False):
                        pass
            # released, so it can be taken again
            with FileLock(lf, blocking=False):
                pass

    # -- Reference selector ------------------------------------------------
    def test_reference_selector_autofill_passthrough_when_offline(self):
        sel = ReferenceImageSelector("http://127.0.0.1:8080", "test")
        self.assertEqual(sel._autofill_texts(["a.png"], [""]), [""])

    # -- Gallery / pins (existence smoke) ---------------------------------
    def test_pins_file_is_readable_or_absent(self):
        pins_file = BOOK / "pinned.json"
        if pins_file.exists():
            self.assertIsInstance(json.loads(pins_file.read_text("utf-8")), dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
