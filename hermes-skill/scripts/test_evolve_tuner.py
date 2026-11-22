#!/usr/bin/env python3
"""Tests for evolve_quality() — the closed-loop auto-tuner.

Regression guard for the bright_boost oscillation: the param was a prompt
prefix Qwen-Image ignored, so the controller measured its own noise and
flapped A-B-A across renders (quality_log 37->38->39 read 17.9->15.2->17.9),
burning a full episode render per cycle.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import episode_producer as ep


def _entry(score, bright, sharp=50.0, idx=0):
    return {
        "chapter": 900, "timestamp": f"2026-09-29T{idx:02d}:00:00",
        "score": score, "avg_image": 50.0, "avg_audio": 100.0,
        "clip_rate": 100.0, "vlm_score": None,
        "avg_sharpness": sharp, "avg_brightness": bright,
        "scenes": [], "render_idx": idx,
    }


class EvolveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _run(self, history, ledger, config):
        """Run evolve_quality against temp state; return (config, ledger)."""
        (self.root / "quality_log.json").write_text(
            json.dumps(history), encoding="utf-8")
        (self.root / "tuning_ledger.json").write_text(
            json.dumps(ledger), encoding="utf-8")
        cfgfile = self.root / "config.json"
        cfgfile.write_text(json.dumps(config), encoding="utf-8")
        with patch.object(ep, "ROOT", self.root), \
             patch.object(ep, "CONFIG_FILE", cfgfile), \
             patch.object(ep, "CONFIG", config), \
             patch.object(ep, "log", lambda *a, **k: None):
            ep.evolve_quality()
        return (json.loads(cfgfile.read_text(encoding="utf-8")),
                json.loads((self.root / "tuning_ledger.json").read_text(encoding="utf-8")))

    def test_harness_actually_reaches_apply(self):
        """Everything below is only meaningful if _run reaches apply().
        40 steps + avg_sharp 99 is the documented trigger; a harness that
        silently never fires makes every other assertion in this file vacuous.
        """
        hist = [_entry(70, 50, sharp=99, idx=i) for i in range(5)]
        cfg, ledger = self._run(hist, [], {"auto_tune": {"image_steps": 40}})
        self.assertEqual(cfg["auto_tune"]["image_steps"], 35,
                         "the harness never reached apply(); every other "
                         "assertion in this file would be vacuous")
        self.assertTrue([e for e in ledger if e["outcome"] == "pending"],
                        "no pending row was recorded")

    def test_bright_boost_never_reapplied(self):
        """Dark frames must NOT flip any knob. Uses the exact input that
        drove the real oscillation (avg_bright 8, well under the old
        `< 45` trigger) so restoring the arm fails this test."""
        hist = [_entry(70, 8, idx=i) for i in range(5)]
        cfg, ledger = self._run(hist, [], {"auto_tune": {"image_steps": 25}})
        tune = cfg["auto_tune"]
        self.assertNotIn("bright_boost", tune,
                         "bright_boost must never be set on the live config")
        live = [e for e in ledger if e["param"] == "bright_boost"
                and e.get("outcome") in ("pending", "helped", "reverted")]
        self.assertEqual(live, [],
                         f"bright_boost was acted on again: {live}")
        self.assertFalse(any("bright_boost" in c
                             for h in tune.get("history", []) for c in h["changes"]),
                         "bright_boost must never appear in a new change")

    def test_bright_boost_retired_from_config_and_ledger(self):
        """A config still carrying the dead key gets it dropped, and its
        pending ledger rows are marked retired, not reverted."""
        hist = [_entry(70, 20, idx=i) for i in range(5)]
        ledger = [{"at": "x", "render_idx": 1, "param": "bright_boost",
                   "old": False, "new": True, "target": "avg_bright",
                   "target_before": 10.0, "outcome": "pending",
                   "rationale": "dark frames 10"}]
        cfg, out = self._run(hist, ledger, {"auto_tune": {"image_steps": 25,
                                                           "bright_boost": True}})
        self.assertNotIn("bright_boost", cfg["auto_tune"])
        row = next(e for e in out if e["param"] == "bright_boost")
        self.assertEqual(row["outcome"], "retired")

    def test_allowlist_is_image_steps_only(self):
        """The allowlist is the contract; bright_boost must never return."""
        self.assertNotIn("bright_boost", ep.AUTO_ALLOWLIST)
        self.assertIn("image_steps", ep.AUTO_ALLOWLIST)

    def test_no_flap_across_repeated_renders(self):
        """Simulate the real A-B-A: same dark input twice must converge
        (no oscillation), i.e. the 2nd run must not undo the 1st."""
        hist = [_entry(70, 8, idx=i) for i in range(5)]
        cfg = {"auto_tune": {"image_steps": 25}}
        cfg1, _ = self._run(hist, [], cfg)
        # Feed the post-run config back in, as the next render would.
        hist2 = [_entry(70, 8, idx=i) for i in range(5, 10)]
        cfg2, ledger2 = self._run(hist2, [], cfg1)
        self.assertEqual(cfg1["auto_tune"].get("image_steps"),
                         cfg2["auto_tune"].get("image_steps"),
                         "stable dark input must reach a fixed point, not flap")
        self.assertFalse(any(e["param"] == "bright_boost" for e in ledger2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
