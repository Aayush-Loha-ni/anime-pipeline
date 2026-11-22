#!/usr/bin/env python3
"""Config precedence: config.json must beat config.yaml.

The YAML merge used to run AFTER the JSON load and overwrite it, so editing
config.json silently did nothing for every h3_* key while SKILL.md called it
authoritative. Also guards the dead `generation.t2i` YAML block, which
contradicted config.json (cfg 7.0 vs image_cfg 1.0) and set nothing.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import episode_producer as ep
from pipeline import config_loader as cl

YAML_PATH = Path(__file__).parent / "pipeline" / "config.yaml"


class ConfigPrecedenceTests(unittest.TestCase):
    def test_json_wins_on_shared_keys(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.json"
            user = json.loads(ep.CONFIG_FILE.read_text(encoding="utf-8"))
            user["h3_steps"] = 33     # YAML declares 20
            user["h3_frames"] = 250   # YAML declares 124
            p.write_text(json.dumps(user), encoding="utf-8")
            err = io.StringIO()
            with patch.object(ep, "CONFIG_FILE", p), \
                 contextlib.redirect_stderr(err):
                cfg = ep.load_config()
        self.assertEqual(cfg["h3_steps"], 33, "config.json must win over config.yaml")
        self.assertEqual(cfg["h3_frames"], 250, "config.json must win over config.yaml")

    def test_drift_is_warned_not_silent(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.json"
            user = json.loads(ep.CONFIG_FILE.read_text(encoding="utf-8"))
            user["h3_steps"] = 33
            p.write_text(json.dumps(user), encoding="utf-8")
            err = io.StringIO()
            with patch.object(ep, "CONFIG_FILE", p), \
                 contextlib.redirect_stderr(err):
                ep.load_config()
        warned = err.getvalue()
        self.assertIn("authoritative", warned,
                      "a config.json/config.yaml disagreement must be reported")

    def test_shipped_configs_do_not_drift(self):
        """The two files currently agree; a silent future edit must fail here."""
        user = json.loads(ep.CONFIG_FILE.read_text(encoding="utf-8"))
        eng = cl.get_active_engine_config(cl.load_config())
        drift = {}
        for k, v in eng.items():
            if str(k).startswith("_"):
                continue
            jk = f"h3_{k}"
            if jk in user and str(user[jk]) != str(v):
                drift[jk] = (user[jk], v)
        self.assertEqual(drift, {},
                         f"config.json and config.yaml disagree: {drift}")

    def test_yaml_has_no_dead_t2i_block(self):
        ycfg = cl.load_config()
        self.assertNotIn("t2i", (ycfg.get("generation") or {}),
                         "generation.t2i was never merged and contradicted "
                         "config.json; T2I params live in config.json only")

    def test_tier_keys_present_in_json(self):
        """Keys the YAML introduced must also live in config.json now."""
        user = json.loads(ep.CONFIG_FILE.read_text(encoding="utf-8"))
        for k in ("quality", "video_engine"):
            self.assertIn(k, user,
                          f"{k} was YAML-only; config.json is authoritative")

    def test_missing_config_file_does_not_crash(self):
        """A missing config.json must not NameError on user_config."""
        with tempfile.TemporaryDirectory() as td:
            absent = Path(td) / "nope.json"
            with patch.object(ep, "CONFIG_FILE", absent):
                cfg = ep.load_config()  # must not raise
        self.assertIsInstance(cfg, dict)
        self.assertIn("lufs_target", cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
