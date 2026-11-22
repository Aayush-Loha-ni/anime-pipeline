#!/usr/bin/env python3
"""Behavioural tests for the dashboard's MUTATING endpoints.

ponytail: these routes had NO tests, and two bugs shipped in them that a
one-line unit test would have caught:

  * `api_settings_set` and `api_set_engine` logged via a bare `logger` that the
    module never defines, so the exception paths raised NameError inside an
    `except` block, masking the real error with a 500.
  * `api_set_engine` called `ep.load_json_file`, which does not exist — that
    name lives in json_util and was never imported into episode_producer. The
    endpoint raised AttributeError on its first line of work.

The old code returned 200 for a change that could not happen, so nothing ever
called these in a test. "It returns 200" was never the property worth pinning;
"the tier actually flips and the caller is told the truth" is.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import episode_producer as ep
import pipeline_server as srv


class MutatingEndpointTests(unittest.TestCase):
    """Every test restores config.json, so run order does not matter."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(srv.app)

    def setUp(self):
        self._cfg_text = ep.CONFIG_FILE.read_text(encoding="utf-8")
        self._cfg = json.loads(self._cfg_text)
        self._live = dict(ep.CONFIG)

    def tearDown(self):
        ep.CONFIG_FILE.write_text(self._cfg_text, encoding="utf-8")
        ep.CONFIG.clear()
        ep.CONFIG.update(self._live)

    def test_a_ledger_write_failure_names_the_cause(self):
        """tokens._save deliberately raises so a lost charge cannot be reported
        as a successful one. That turns /api/tokens/grant into a bare 500 with
        no `detail`, while the two mutating config routes name the cause — a
        grant that silently vanishes is the token equivalent of the
        "returned 200 for a change that could not happen" bug."""
        with patch.object(srv.token_bucket, "grant",
                          side_effect=OSError("ledger is locked by pid 4")):
            r = self.client.post("/api/tokens/grant",
                                 json={"n": 5, "reason": "new image"})
        self.assertEqual(r.status_code, 500, r.text)
        self.assertIn("ledger is locked by pid 4", r.text,
                      "the 500 must name why the ledger could not be written")

    def test_a_successful_grant_reports_the_new_balance(self):
        # against a scratch ledger: a real grant writes the real one
        with tempfile.TemporaryDirectory() as td:
            led = Path(td) / "tokens.json"
            with patch.object(srv.token_bucket, "_file", lambda: led):
                r = self.client.post("/api/tokens/grant",
                                     json={"n": 3, "reason": "new image"})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json(), {"balance": 3, "granted": 3})
                self.assertEqual(json.loads(led.read_text(encoding="utf-8")),
                                 {"balance": 3, "last_grant_reason": "new image"})

    # -- /api/config/engine ------------------------------------------------
    def test_switching_quality_actually_flips_the_render(self):
        """The headline bug: this wrote config.yaml, which config.json beats,
        so the tier never changed and the route still reported 200."""
        target = "final" if ep.CONFIG.get("quality") == "fast" else "fast"
        r = self.client.post("/api/config/engine",
                             json={"engine": "h3", "quality": target})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["applied"],
                        f"reported applied={body.get('applied')} for {target}")
        self.assertEqual(body["quality"], target)
        # 1. the authoritative file on disk
        on_disk = json.loads(ep.CONFIG_FILE.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["quality"], target,
                         "config.json is what the render reads")
        # 2. the live in-process config, for the next render in this process
        self.assertEqual(ep.CONFIG["quality"], target)

    def test_engine_rejects_an_unknown_value(self):
        for engine in ("wan", "", "H3"):
            r = self.client.post("/api/config/engine",
                                 json={"engine": engine, "quality": "fast"})
            self.assertEqual(r.status_code, 400, f"engine={engine!r} accepted")

    def test_quality_rejects_an_unknown_tier(self):
        for q in ("Final", "best", ""):
            r = self.client.post("/api/config/engine",
                                 json={"engine": "h3", "quality": q})
            self.assertEqual(r.status_code, 400, f"quality={q!r} accepted")

    def test_a_failed_disk_write_is_reported_not_swallowed(self):
        """Returning 200 for a change that never reached disk is the exact
        failure mode this endpoint had."""
        with patch.object(ep, "_atomic_write_json",
                          side_effect=OSError("disk locked")):
            r = self.client.post("/api/config/engine",
                                 json={"engine": "h3", "quality": "final"})
        self.assertEqual(r.status_code, 500)
        self.assertIn("not written", r.json()["detail"])

    def test_the_engine_switch_never_rewrites_config_yaml(self):
        """config.yaml carries 46 lines of comments explaining the precedence
        rules. Mirroring into it accomplished nothing (config.json wins) except
        destroying those comments via yaml.dump and leaving a value that
        disagrees with the real one."""
        yaml_path = Path(srv.__file__).parent / "pipeline" / "config.yaml"
        if not yaml_path.exists():
            self.skipTest("no config.yaml in this checkout")
        before = yaml_path.read_text(encoding="utf-8")
        self.client.post("/api/config/engine",
                         json={"engine": "h3", "quality": "final"})
        after = yaml_path.read_text(encoding="utf-8")
        self.assertEqual(before, after,
                         "config.yaml must not be rewritten by an API call")

    # -- /api/settings -----------------------------------------------------
    def test_out_of_range_value_is_clamped_and_reported(self):
        """The dashboard advertised image_cfg 4-8 while the producer caps it at
        4 and ships 1.0 — it could write a value the producer calls broken."""
        r = self.client.post("/api/settings",
                             json={"key": "image_cfg", "value": 99.0})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["clamped"], "a clamped write must say so")
        self.assertEqual(body["effective"], ep._CONFIG_SCHEMA["image_cfg"][2])
        self.assertNotEqual(body["effective"], 99.0)
        self.assertLessEqual(body["effective"],
                             ep._CONFIG_SCHEMA["image_cfg"][2],
                             "the producer schema is the contract")

    def test_the_producer_schema_is_what_actually_clamps(self):
        """Whose clamp runs first — the endpoint's, or the producer's?

        This pins the layering so a future reader does not mistake the
        `ep._validate_config` call in api_settings_set for the thing enforcing
        the bound. Today the ENDPOINT clamps (SETTINGS_SCHEMA is a subset of
        _CONFIG_SCHEMA), and that subset relationship is enforced by
        test_settings_bounds_never_exceed_the_producer_schema. If someone
        widens a dashboard bound past the producer's, this test flips and says
        so instead of the two silently disagreeing.
        """
        spec = next(s for s in srv.SETTINGS_SCHEMA if s["key"] == "image_cfg")
        _typ, plo, phi = ep._CONFIG_SCHEMA["image_cfg"]
        self.assertTrue(spec["max"] <= phi,
                        "dashboard max exceeds the producer's; the producer's "
                        "clamp is now the one doing the work")
        # and with a valid value nothing clamps at all
        r = self.client.post("/api/settings",
                             json={"key": "image_cfg", "value": 2.0})
        self.assertEqual(r.json()["effective"], 2.0)
        self.assertFalse(r.json()["clamped"])

    def test_in_range_value_takes_effect(self):
        r = self.client.post("/api/settings",
                             json={"key": "image_cfg", "value": 2.0})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["effective"], 2.0)
        self.assertEqual(ep.CONFIG["image_cfg"], 2.0)
        self.assertEqual(
            json.loads(ep.CONFIG_FILE.read_text(encoding="utf-8"))["image_cfg"],
            2.0)

    def test_unknown_key_is_rejected(self):
        r = self.client.post("/api/settings",
                             json={"key": "not_a_setting", "value": 1})
        self.assertEqual(r.status_code, 400)

    def test_settings_bounds_never_exceed_the_producer_schema(self):
        """Structural guard: the dashboard's UI bounds are a convenience, the
        producer's schema is the contract. They must not diverge again."""
        for spec in srv.SETTINGS_SCHEMA:
            key = spec["key"]
            if key not in ep._CONFIG_SCHEMA:
                continue
            _typ, plo, phi = ep._CONFIG_SCHEMA[key]
            with self.subTest(key=key):
                dlo, dhi = spec.get("min"), spec.get("max")
                self.assertTrue(dlo is None or plo is None or dlo >= plo,
                                f"{key}: dashboard min {dlo} below producer {plo}")
                self.assertTrue(dhi is None or phi is None or dhi <= phi,
                                f"{key}: dashboard max {dhi} above producer {phi}")

    def test_settings_suggestions_do_not_recommend_a_broken_value(self):
        """The hint used to read 'h3_steps: 8 = PDD distilled (recommended)'
        while the producer's own defaults record that 8 steps produced clips
        which ignored the reference still entirely."""
        for spec in srv.SETTINGS_SCHEMA:
            hint = (spec.get("hint") or "")
            with self.subTest(key=spec["key"]):
                if spec["key"] == "h3_steps":
                    # The verified recipe is 20 steps; the hint used to
                    # recommend 8, which the producer's own defaults record as
                    # producing clips that ignore the reference still. Check the
                    # rule, not a magic digit: a hint that calls any value
                    # "recommended" while the shipped value differs is wrong.
                    recommended = hint.lower()
                    if "recommended" in recommended:
                        shipped = ep.CONFIG.get("h3_steps")
                        self.assertIn(str(shipped), hint,
                                      f"h3_steps hint recommends a value other "
                                      f"than the shipped {shipped}: {hint!r}")
                    self.assertNotIn("8 = ", hint,
                                     "the stale '8 steps is recommended' advice "
                                     "is back")

    # -- misc honesty ------------------------------------------------------
    def test_phase_ladder_puts_a_stopped_run_at_a_terminal_step(self):
        """'stopped' was handled in the staleness check but not the ladder, so
        cur was -1 and every step rendered as todo after a Stop."""
        import re
        src = Path(srv.__file__).read_text(encoding="utf-8")
        block = re.search(r"PHASES = \[.*?\]\s*\n\s*order = .*?\n\s*phases = ",
                          src, re.S)
        self.assertIsNotNone(block)
        body = block.group(0)
        self.assertIn('("stopped", "idle")', body,
                      "the terminal-phase branch is gone")
        self.assertIn("len(PHASES) - 1", body,
                      "a terminal phase must land past the LAST real step")

    def test_caption_endpoint_refuses_paths_outside_images(self):
        """/api/caption joined an unvalidated name onto images/, so `../../..`
        escaped the directory and the 404-vs-200 split was an existence oracle
        for any path on disk."""
        for bad in ("../../../Windows/win.ini",
                    "..\\..\\..\\Windows\\win.ini",
                    "C:/Windows/win.ini"):
            with self.subTest(path=bad):
                r = self.client.post("/api/caption", json={"image": bad})
                self.assertIn(r.status_code, (400, 404),
                              f"{bad} -> {r.status_code} (200 means it read it)")
                self.assertNotEqual(r.status_code, 200)

    def test_cancel_all_is_honest_when_there_is_nothing_to_stop(self):
        r = self.client.post("/api/tasks/cancel-all")
        self.assertEqual(r.status_code, 200)
        self.assertIn("cancelled", r.json())
        if not r.json()["cancelled"]:
            self.assertTrue(r.json().get("reason"),
                            "a False answer needs a reason")


if __name__ == "__main__":
    unittest.main(verbosity=2)
