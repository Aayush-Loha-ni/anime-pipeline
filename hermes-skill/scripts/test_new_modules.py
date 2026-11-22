#!/usr/bin/env python3
"""Tests for pipeline modules: preflight, schemas, rights manifest v2.

ponytail: this was a straight-line script with 20 module-level `assert`s, so
`unittest discover` registered ZERO tests from it — the checks ran as an
import-time side effect and a failure reported as `ImportError: Failed to
import test module` with no name attached. Now a real TestCase so each
assertion is counted, named and independently reported.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SKIP_GENAI", "1")
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent.parent

from pipeline.preflight import (check_director_package, check_disk_ram,
                                check_issues, check_rights,
                                format_preflight_report)
from pipeline.schemas import (validate_character_bible, validate_edl_item)
from pipeline.rights_manifest import (add_character_reference, add_lora,
                                      check_safety, load_rights_manifest,
                                      save_rights_manifest, _empty_manifest,
                                      preflight_rights_gate)


class PreflightCheckTests(unittest.TestCase):
    def test_empty_issues_file_is_ok(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "issues.json").write_text("[]", encoding="utf-8")
            self.assertTrue(check_issues(Path(td))["ok"])

    def test_issues_file_of_the_wrong_shape_fails_closed(self):
        """A dict-shaped issues.json used to report ok=True.

        Iterating a dict yields its keys, `isinstance(i, dict)` filtered them
        all out, so `open_crits` came back empty and the render-blocking gate
        silently reported clean. Eight lines below a comment saying it must
        fail closed.
        """
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "issues.json").write_text('{"oops": true}', encoding="utf-8")
            r = check_issues(Path(td))
            self.assertFalse(r["ok"], "a non-list issues.json must fail closed")
            self.assertIn("not a list", r["detail"])

    def test_unreadable_issues_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "issues.json").write_text("{not json", encoding="utf-8")
            self.assertFalse(check_issues(Path(td))["ok"])

    def test_open_critical_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "issues.json").write_text(json.dumps([
                {"severity": "critical", "ts": "2099-01-01 00:00:00",
                 "symptom": "boom", "resolved": False}]), encoding="utf-8")
            self.assertFalse(check_issues(Path(td))["ok"])

    def test_stale_critical_is_triaged_out(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "issues.json").write_text(json.dumps([
                {"severity": "critical", "ts": "2020-01-01 00:00:00",
                 "symptom": "old", "resolved": False}]), encoding="utf-8")
            r = check_issues(Path(td))
            self.assertTrue(r["ok"], "a >7-day critical must age out")
            self.assertEqual(len(r["stale_criticals"]), 1)

    def test_missing_director_package_detected(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "scenes").mkdir()
            r = check_director_package(td, 1)
            self.assertFalse(r["ok"])
            self.assertIn("not found", r["detail"])

    def test_placeholder_director_package_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            scenes = Path(td) / "scenes"
            scenes.mkdir()
            (scenes / "ch001_director.json").write_text(json.dumps(
                {"episode_promise": "TBD", "version": "placeholder",
                 "owner_approved": False}), encoding="utf-8")
            self.assertIn("placeholder", check_director_package(td, 1)["detail"])

    def test_approved_director_package_passes(self):
        with tempfile.TemporaryDirectory() as td:
            scenes = Path(td) / "scenes"
            scenes.mkdir()
            (scenes / "ch001_director.json").write_text(json.dumps(
                {"episode_promise": "A test", "version": "1.0",
                 "owner_approved": True}), encoding="utf-8")
            r = check_director_package(td, 1)
            self.assertTrue(r["ok"])
            self.assertIn("approved: True", r["detail"])

    def test_report_formatting(self):
        text = format_preflight_report({
            "timestamp": "2026-09-22T00:00:00", "chapter": 1, "ok": True,
            "checks": {"services": {"llm": {"ok": True, "detail": "responding"}},
                       "issues": {"ok": True, "detail": "no issues"}}})
        self.assertIn("PREFLIGHT REPORT", text)
        self.assertIn("PASS", text)

    def test_check_rights_survives_a_non_dict_manifest(self):
        """One stray '[' used to raise AttributeError and 500 /api/readiness."""
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "rights_manifest.json").write_text(
                '["a list, not a dict"]', encoding="utf-8")
            r = check_rights(Path(td))
            self.assertFalse(r["ok"])
            self.assertIsInstance(r["detail"], str)

    def test_check_rights_reports_an_empty_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            r = check_rights(Path(td))
            self.assertFalse(r["ok"])
            self.assertIn("empty", r["detail"])

    def test_ram_check_fails_closed_when_it_cannot_measure(self):
        """Not knowing the free RAM is not the same as having RAM.

        This used to report {"ok": True, "detail": "unable to determine"}, so a
        broken RAM probe counted as a PASS — while the disk branch one screen up
        failed closed with an explicit comment saying it should. The wmic
        fallback it degraded into was dead three ways (`"/ value"` had a space
        inside the switch, the parser wanted `Key=Value` that `wmic get` never
        emits, and wmic is absent from modern Windows) and has been deleted, so
        psutil is now the only path and "no psutil" must be a FAILURE.
        """
        import builtins
        real_import = builtins.__import__

        def no_psutil(name, *a, **k):
            if name == "psutil":
                raise ImportError("simulated: psutil absent")
            return real_import(name, *a, **k)

        builtins.__import__ = no_psutil
        try:
            info = check_disk_ram()
        finally:
            builtins.__import__ = real_import
        self.assertFalse(info["ram"]["ok"],
                         "an unmeasurable RAM check must not report a pass")
        self.assertIn("unknown", info["ram"]["detail"])

    def test_ram_check_reports_a_real_measurement_when_psutil_is_present(self):
        info = check_disk_ram()
        self.assertIn("ok", info["ram"])
        self.assertRegex(info["ram"]["detail"], r"available|unknown")


class SchemaTests(unittest.TestCase):
    def test_valid_character_bible_passes(self):
        errors = validate_character_bible({"test": {
            "character_id": "test", "full_name": "Test Character",
            "role": "test", "canonical_description": "A test character",
            "approved_references": [], "approval_state": "approved"}})
        self.assertEqual(errors, [])

    def test_valid_edl_item_passes(self):
        self.assertEqual(validate_edl_item(
            {"scene_number": 1, "shot_number": 1, "in_point": 0.0,
             "out_point": 5.0, "asset_id": "scene_01",
             "asset_type": "still"}), [])

    def test_invalid_edl_item_is_caught(self):
        self.assertGreater(len(validate_edl_item({"scene_number": 1})), 0)


class RightsManifestV2Tests(unittest.TestCase):
    def setUp(self):
        self.m = _empty_manifest()
        self.m["novel_adaptation_rights"]["owner_confirmation"] = True
        self.m["novel_adaptation_rights"]["commercial_use"] = "yes"
        self.m["voice"]["owner_confirmation"] = True
        self.m["voice"]["commercial_use"] = "yes"

    def test_add_character_reference(self):
        add_character_reference(self.m, "test_char", "local", "owner_created",
                                "yes", "test.png", "Test character")
        self.assertEqual(len(self.m["character_references"]), 1)

    def test_add_lora(self):
        add_lora(self.m, "test_lora", "trigger_word", "yes", "local", "open",
                 "test.safetensors")
        self.assertEqual(len(self.m["loras"]), 1)

    def test_safety_checks(self):
        self.assertTrue(check_safety("beautiful anime scene in dark city")[0])
        self.assertFalse(check_safety("in the style of studio ghibli")[0])
        self.assertFalse(check_safety("portrait of real person celebrity")[0])

    def test_save_load_round_trip(self):
        add_character_reference(self.m, "test_char", "local", "owner_created",
                                "yes", "test.png", "Test character")
        add_lora(self.m, "test_lora", "trigger_word", "yes", "local", "open",
                 "test.safetensors")
        with tempfile.TemporaryDirectory() as td:
            save_rights_manifest(self.m, td)
            loaded = load_rights_manifest(td)
            self.assertEqual(loaded["version"], "2.0")
            self.assertEqual(len(loaded["character_references"]), 1)
            self.assertEqual(len(loaded["loras"]), 1)

    def test_save_load_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as td:
            save_rights_manifest(self.m, td)
            self.assertEqual(list(Path(td).glob("*.tmp")), [])

    def test_gate_survives_a_non_dict_row(self):
        """The container was guarded, the row was not — a string in `music`
        raised AttributeError and turned the release gate into a 500."""
        ok, blockers, _warn = preflight_rights_gate(
            {"music": ["oops-a-string"], "entries": []})
        self.assertFalse(ok)
        self.assertIsInstance(blockers, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
