#!/usr/bin/env python3
"""Coverage for the extracted pipeline/quality_evolve.py module.

Exists because `rate_episode` had ZERO test coverage, and the one time the
module's helpers were split apart it raised NameError('_newest') on every
render — swallowed by the blanket `except` around Phase 8, leaving
quality_log.json frozen and the auto-tuner running on stale history. 300
green tests did not notice. These tests call the entry points for real.
"""
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import episode_producer as ep
import pipeline.quality_evolve as qe


class DelegationTests(unittest.TestCase):
    """Every helper the moved code borrows must actually resolve."""

    def test_newest_delegates_to_episode_producer(self):
        with patch.object(ep, "_newest", return_value="sentinel") as m:
            self.assertEqual(qe._newest(["a", "b"]), "sentinel")
        m.assert_called_once_with(["a", "b"])

    def test_newest_works_without_a_mock(self):
        """Regression: the bare `_newest` NameError that broke every render."""
        import tempfile
        import time as _t
        with tempfile.TemporaryDirectory() as td:
            old = Path(td) / "old.png"
            new = Path(td) / "new.png"
            old.write_bytes(b"x")
            _t.sleep(0.01)
            new.write_bytes(b"x")
            # episode_producer._newest returns a str path (or None) — match it.
            got = qe._newest([old, new])
            self.assertEqual(Path(got), new)
            self.assertIsNone(qe._newest([]))

    def test_importable_before_episode_producer(self):
        """episode_producer imports this module from its tail, so the import
        of it must stay deferred or `import pipeline.quality_evolve` first
        raises a circular ImportError."""
        code = ("import sys; sys.path.insert(0, r'%s'); "
                "import pipeline.quality_evolve; print('ok')"
                % str(Path(qe.__file__).parent.parent))
        p = subprocess.run(["python", "-c", code], capture_output=True,
                           text=True, timeout=180, cwd=str(Path(qe.__file__).parent.parent))
        self.assertEqual(p.returncode, 0,
                         f"standalone import failed: {p.stderr[-400:]}")
        self.assertIn("ok", p.stdout)

    def test_module_has_no_undefined_names(self):
        """pyflakes in a subprocess: catches any name the split left behind."""
        p = subprocess.run(["python", "-m", "pyflakes", str(qe.__file__)],
                           capture_output=True, text=True, timeout=180)
        undefined = [ln for ln in (p.stdout or "").splitlines()
                     if "undefined name" in ln]
        self.assertEqual(undefined, [],
                         f"undefined names in quality_evolve.py: {undefined}")
        unused = [ln for ln in (p.stdout or "").splitlines()
                  if "imported but unused" in ln]
        self.assertEqual(unused, [], f"stale imports: {unused}")


class RateEpisodeTests(unittest.TestCase):
    """rate_episode must survive a scene with no still and no audio."""

    def test_rates_scenes_without_media(self):
        """rate_episode must survive a scene with no still and no audio.

        Writes a real PNG so the `_newest(refs)` branch is actually taken —
        with an empty images_dir the ternary short-circuits and _newest is
        never called, which is the exact call that NameError'd after the
        split. test_newest_works_without_a_mock pins _newest itself.
        """
        import tempfile
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            imgdir = Path(td) / "images"
            imgdir.mkdir()
            Image.new("L", (256, 256), 128).save(imgdir / "ch900_s01_x.png")
            # a b_ companion that must be EXCLUDED from the A-still pick
            Image.new("L", (256, 256), 250).save(imgdir / "ch900_s01b_y.png")
            cfg = {
                "images_dir": str(imgdir), "output_dir": td, "scenes_dir": td,
                "ffmpeg": "ffmpeg", "auto_tune": {"image_steps": 25},
                "image_cfg": 1.0, "platform": "youtube", "video_engine": "h3",
            }
            scenes = [{"scene_number": 1, "narration": "a dark room",
                       "image_prompt": "dark room"}]
            logs = []
            with patch.object(ep, "CONFIG", cfg), \
                 patch.object(ep, "ROOT", Path(td)), \
                 patch.object(ep, "log", lambda m, i=0: logs.append(m)), \
                 patch.object(qe, "_ep_get", return_value=ep):
                entry = qe.rate_episode(900, scenes, [None], [None])
        self.assertIsNotNone(entry, "rate_episode must not return None")
        self.assertIn("score", entry)
        self.assertIn("chapter", entry)
        self.assertIsInstance(entry["score"], (int, float))


class EvolveGuardTests(unittest.TestCase):
    """The allowlist guard must survive `python -O`, which strips asserts.

    The test below drives the guard through a real evolve_quality() run rather
    than grepping its source: a grep would stay green if the raise were
    swapped for a different exception type or moved into a helper, and an
    earlier version of this test shelled out to `python -O` with an
    `assert` inside — which `-O` strips, so it could never fail.
    """

    def _run_with_allowlist(self, allowlist):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # avg_sharpness 99 > 75 with steps 40 > 25 drives apply().
            (root / "quality_log.json").write_text(json.dumps([{
                "chapter": 900, "timestamp": "2026-09-29T00:00:00",
                "score": 90, "avg_image": 90, "avg_audio": 100,
                "clip_rate": 100, "vlm_score": None,
                "avg_sharpness": 99, "avg_brightness": 50, "scenes": [],
            }]), encoding="utf-8")
            (root / "tuning_ledger.json").write_text("[]", encoding="utf-8")
            cfgfile = root / "config.json"
            cfgfile.write_text(json.dumps({"auto_tune": {"image_steps": 40}}),
                               encoding="utf-8")
            cfg = json.loads(cfgfile.read_text(encoding="utf-8"))
            # Patch episode_producer, NOT quality_evolve: _sync() copies every
            # alias out of ep at the top of evolve_quality, so a qe.* patch
            # would be overwritten before the body reads it.
            with patch.object(ep, "ROOT", root), \
                 patch.object(ep, "CONFIG_FILE", cfgfile), \
                 patch.object(ep, "CONFIG", cfg), \
                 patch.object(ep, "AUTO_ALLOWLIST", allowlist), \
                 patch.object(qe, "_log", lambda *a, **k: None):
                qe.evolve_quality()
            return json.loads(cfgfile.read_text(encoding="utf-8"))

    def test_apply_path_is_allowlisted(self):
        """With image_steps allowlisted, the step-down actually applies:
        40 steps + avg_sharp 99 is the documented trigger, and the result
        must be 35. A vacuous version of this test passed just because the
        key existed in the config to begin with."""
        tune = self._run_with_allowlist(("image_steps",))
        self.assertEqual(tune["auto_tune"]["image_steps"], 35,
                         "the step-down did not apply — the test input does "
                         "not reach apply(), so it proves nothing")

    def test_guard_rejects_non_allowlisted_param(self):
        """Emptying the allowlist must make the SAME input raise, proving the
        guard is a live check on the execution path and not decoration."""
        with self.assertRaises(ValueError):
            self._run_with_allowlist(())

    def test_guard_is_not_a_bare_assert(self):
        """Under -O an assert would be stripped and the param would apply.
        Run the real guard under -O and require it to still raise."""
        import textwrap
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(Path(qe.__file__).parent.parent)!r})
            import json, tempfile
            from pathlib import Path
            from unittest.mock import patch
            import episode_producer as ep
            import pipeline.quality_evolve as q
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root / "quality_log.json").write_text(json.dumps([{{
                    "chapter": 900, "timestamp": "x", "score": 90,
                    "avg_image": 90, "avg_audio": 100, "clip_rate": 100,
                    "vlm_score": None, "avg_sharpness": 99,
                    "avg_brightness": 50, "scenes": []}}]))
                (root / "tuning_ledger.json").write_text("[]")
                cf = root / "config.json"
                cf.write_text(json.dumps({{"auto_tune": {{"image_steps": 40}}}}))
                cfg = json.loads(cf.read_text())
                with patch.object(ep, "ROOT", root), \\
                     patch.object(ep, "CONFIG_FILE", cf), \\
                     patch.object(ep, "CONFIG", cfg), \\
                     patch.object(ep, "AUTO_ALLOWLIST", ()), \\
                     patch.object(q, "_log", lambda *a, **k: None):
                    q.evolve_quality()
            print("NO-RAISE")
        """)
        p = subprocess.run(["python", "-O", "-c", script], capture_output=True,
                           text=True, timeout=240,
                           cwd=str(Path(qe.__file__).parent.parent))
        self.assertNotIn("NO-RAISE", p.stdout,
                         "under -O the non-allowlisted param applied silently")
        self.assertIn("ValueError", p.stderr,
                      f"expected ValueError under -O; stderr: {p.stderr[-400:]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
