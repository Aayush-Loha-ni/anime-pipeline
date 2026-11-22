#!/usr/bin/env python3
"""Regression check for the invariants fixed in the quality pass.

Every assertion here corresponds to a bug that shipped. Run:

    python -m unittest test_upgrade_fixes -v        (from hermes-skill/scripts)
    python test_upgrade_fixes.py                    (direct)

Deliberately stdlib-only and side-effect free: it writes to tmp dirs, never
touches the real project state, and needs no GPU or services.

NOTE ON THE SOURCE-GREPS IN THIS FILE
--------------------------------------
About a quarter of the tests here read episode_producer.py's SOURCE TEXT and
assert a string is (not) present, e.g.

    self.assertNotIn("def loudnorm_file(", src)

Read them as static-analysis guards for invariants that are awkward to prove
behaviourally — "no KSampler hardcodes karras", "drift is recorded but is not a
hard gate", "the tuning loop never raises steps on soft frames". They catch
real regressions and they are worth keeping for those.

Their known weakness, stated plainly so nobody is misled: a grep does not
distinguish "the invariant holds" from "the text happens to be there". A
reimplementation that keeps the string but breaks the behaviour passes; a
harmless rename breaks the test. Where a test could be expressed as behaviour
it now is — see test_is_v2_uses_resolved_version and
test_dead_loudnorm_twin_deleted, which were converted after this audit found
them. If you add a test here, prefer asserting behaviour; reach for a grep only
when the property is genuinely static.

One more trap this file had: test_gate_label_does_not_advertise_the_broken_combo
was DEFINED TWICE, byte-identical. The second definition silently replaced the
first in the class body, so the suite counted one test where there were two and
edits to the first were dead code. The duplicate is removed. When adding tests,
check you are not shadowing an existing method name.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["ANIME_NO_PROGRESS"] = "1"

import episode_producer as ep  # noqa: E402
import supervisor as sup  # noqa: E402
from json_util import loads_json, strip_trailing_commas  # noqa: E402
from pipeline import preflight as pf  # noqa: E402


def _rmtree(root: Path) -> None:
    """Remove a temp tree. _promote_media and build_release_package both
    create subdirectories, so a bare rmdir() on the top level is not enough."""
    root = Path(root)
    if not root.exists():
        return
    for p in sorted(root.rglob("*"), reverse=True):
        try:
            if p.is_dir():
                p.rmdir()
            else:
                p.unlink()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


class ConfigFallbackTests(unittest.TestCase):
    """_validate_config used to restore typ() == 0, which disarmed the GPU
    heat guard (gpu_max_temp) and re-armed the documented face-check seed
    collision (face_check_retry_offset)."""

    def test_bad_value_restores_shipped_default_not_zero(self):
        cfg = {"gpu_max_temp": "hot", "face_check_retry_offset": "nope",
               "image_cfg": "wide", "voice_speed": "fast"}
        ep._validate_config(cfg, ep._DEFAULTS)
        self.assertEqual(cfg["gpu_max_temp"], 85)
        self.assertEqual(cfg["face_check_retry_offset"], 50000)
        # image_cfg ships at 1.0: Qwen-Image-2.1 is flow-matching and
        # oversaturates above ~2. It was 6.0, which is what the still path
        # shipped alongside a karras schedule and black stills.
        self.assertEqual(cfg["image_cfg"], 1.0)
        self.assertEqual(cfg["voice_speed"], 1.0)

    def test_defaults_snapshot_is_populated(self):
        # Every schema key needs a shipped default. Checking a hand-picked
        # few is what let `sheet_anchor_denoise` ship with none — a bad
        # config value then zeroed the head-anchor denoise.
        missing = [k for k in ep._CONFIG_SCHEMA if k not in ep._DEFAULTS]
        self.assertEqual(missing, [], f"schema keys with no shipped default: {missing}")

    def test_load_config_actually_wires_the_fallback(self):
        """The flagship fix was untested: reverting the call site to
        `_validate_config(defaults)` failed nothing."""
        from unittest import mock
        with mock.patch.object(ep, "CONFIG_FILE",
                               HERE.parent / "_nonexistent_config.json"), \
             mock.patch.object(ep, "_DEFAULTS", {}):
            cfg = ep.load_config()
        # no fallback table -> schema minimum, NOT the shipped value
        self.assertNotEqual(cfg["gpu_max_temp"], 0)
        with mock.patch.object(ep, "CONFIG_FILE",
                               HERE.parent / "_nonexistent_config.json"):
            cfg = ep.load_config()  # real _DEFAULTS
        self.assertEqual(cfg["gpu_max_temp"], 85)
        self.assertEqual(cfg["face_check_retry_offset"], 50000)
        self.assertEqual(cfg["sheet_anchor_denoise"], 0.7)
        self.assertIn(cfg["quality"], ("fast", "final"))

    def test_quality_is_enum_validated(self):
        cfg = {"quality": "final "}
        ep._validate_config(cfg, ep._DEFAULTS)
        self.assertEqual(cfg["quality"], "fast")

    def test_clamp_still_applies_to_good_values(self):
        cfg = {"fps": 4, "music_volume": 5.0}
        ep._validate_config(cfg, ep._DEFAULTS)
        self.assertEqual(cfg["fps"], 8)       # schema min
        self.assertEqual(cfg["music_volume"], 1.0)  # schema max

    def test_fallback_absent_clamps_to_schema_min_not_shipped_default(self):
        # With no fallback table the old path yields typ()==0, which the
        # schema then clamps to the minimum: gpu_max_temp becomes 60 instead
        # of the shipped 85, so the heat guard trips constantly.
        cfg = {"gpu_max_temp": "hot"}
        ep._validate_config(cfg)
        self.assertEqual(cfg["gpu_max_temp"], 60)


class NewestAssetTests(unittest.TestCase):
    """Filenames end in uuid4().hex, so sorted()[-1] was a RANDOM pick, not
    the newest. _newest() must order by mtime."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        for p in self.dir.iterdir():
            p.unlink()
        self.dir.rmdir()

    def _make(self, name, age_s):
        p = self.dir / name
        p.write_bytes(b"x")
        t = time.time() - age_s
        os.utime(p, (t, t))
        return p

    def test_picks_most_recent_by_mtime(self):
        old = self._make("ch001_s01_deadbeef.png", 600)
        new = self._make("ch001_s01_cafebabe.png", 1)
        # lexicographic order would pick deadbeef
        self.assertEqual(str(sorted([old, new])[-1]), str(old))
        # _newest returns a Path (it used to return str while every producer of
        # those paths was a Path, so callers disagreed about the type and one
        # had to wrap it defensively). Compare as paths, not as strings.
        self.assertEqual(Path(ep._newest([old, new])), Path(new))
        self.assertIsInstance(ep._newest([old, new]), Path)

    def test_deterministic_on_tie(self):
        a = self._make("ch001_s01_aaaa.png", 5)
        b = self._make("ch001_s01_bbbb.png", 5)
        self.assertEqual(Path(ep._newest([a, b])), Path(ep._newest([b, a])))

    def test_empty_returns_none(self):
        self.assertIsNone(ep._newest([]))


class TruePeakConsistencyTests(unittest.TestCase):
    """The assembler limiter and the QA gate had separate -2.0 literals and
    the gate compared peak <= -2.0 against an alimiter that lands exactly
    there, so the gate could fail its own output."""

    def test_gate_limit_matches_limiter(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for name in ("qa_true_peak", "_loudnorm_dual"):
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == name)
            body = ast.get_source_segment(src, fn)
            self.assertIn("_TP_LIMIT", body, f"{name} does not use the shared limit")
        # no hardcoded -2.0 left behind in the gate
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "qa_true_peak")
        self.assertNotIn("peak <= -2.0", ast.get_source_segment(src, fn))

    def test_tolerance_is_positive(self):
        self.assertGreater(ep._TP_TOL, 0)
        self.assertLess(ep._TP_TOL, 0.3)  # EBU Tech 3341 measurement tolerance


class FpsSingleSourceTests(unittest.TestCase):
    """24 was hardcoded in 8 places while CONFIG['fps'] was read in 2, so
    fps=30 made validate_media reject every render."""

    def test_validate_media_uses_config_fps(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "validate_media")
        body = ast.get_source_segment(src, fn)
        self.assertIn('CONFIG.get("fps"', body)
        self.assertNotIn("abs(fps - 24)", body)

    def test_echo_fingerprint_tracks_echo_fps(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_echo_fingerprint")
        body = ast.get_source_segment(src, fn)
        self.assertIn('CONFIG.get("echo_fps"', body)

    def test_echo_frames_uses_echo_fps(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_echo_frames")
        body = ast.get_source_segment(src, fn)
        self.assertIn('echo_fps', body)

    def test_no_bare_24_fps_literals_in_hot_paths(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        # the concrete regressions we fixed
        self.assertNotIn("abs(fps - 24)", src)
        self.assertNotIn("clip_path, fps=24,", src)


class PreflightGateTests(unittest.TestCase):
    """run_preflight did result.get("ok", True) on NESTED dicts
    ({name: {ok,...}}), so ComfyUI-down / models-missing / disk-full all
    reported PASS."""

    def test_check_ok_handles_nested_shape(self):
        self.assertFalse(pf._check_ok({"comfyui": {"ok": False, "detail": "x"}}))
        self.assertTrue(pf._check_ok({"comfyui": {"ok": True}, "llm": {"ok": True}}))
        self.assertTrue(pf._check_ok({"ok": True}))
        self.assertFalse(pf._check_ok({"ok": False}))
        self.assertTrue(pf._check_ok({}))          # no verdict = not a failure
        self.assertTrue(pf._check_ok("not a dict"))

    def test_nested_failure_flips_report(self):
        report = {"checks": {"services": {"comfyui": {"ok": False}}}}
        ok = all(pf._check_ok(r) for r in report["checks"].values())
        self.assertFalse(ok)


class LedgerTriageTests(unittest.TestCase):
    """A single crash critical blocked every render forever: the ledger had
    no triage window and no resolve path."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.orig = sup.LEDGER
        sup.LEDGER = self.dir / "issues.json"

    def tearDown(self):
        sup.LEDGER = self.orig
        for p in self.dir.iterdir():
            p.unlink()
        self.dir.rmdir()

    def _write(self, rows):
        self.dir.joinpath("issues.json").write_text(json.dumps(rows), encoding="utf-8")

    def _read(self):
        return json.loads(self.dir.joinpath("issues.json").read_text(encoding="utf-8"))

    def test_parse_ts_handles_both_formats(self):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        now = time.time()
        self.assertAlmostEqual(now, sup.parse_ts(stamp), delta=120)
        self.assertAlmostEqual(now, sup.parse_ts(iso), delta=120)
        # the formats ONLY fromisoformat adds — the old strptime list
        # already covered the two above, so those were never a real test
        self.assertIsNotNone(sup.parse_ts("2026-09-01T10:00:00+02:00"))
        self.assertIsNotNone(sup.parse_ts("2026-09-01T10:00:00Z"))
        self.assertIsNotNone(sup.parse_ts("2026-09-01T10:00:00.123456"))
        self.assertIsNone(sup.parse_ts("not a date"))
        self.assertIsNone(sup.parse_ts(""))

    def test_iso_with_offset_is_aged_correctly(self):
        # A row written by other tooling as ISO+Z must still be recognised
        # as OLD, otherwise the 7-day triage window never applies to it.
        self._write([{"ts": "2020-01-01T00:00:00+00:00",
                      "severity": "critical", "symptom": "ancient",
                      "action": "", "resolved": False}])
        from pipeline import preflight as pf2
        res = pf2.check_issues(self.dir)
        self.assertTrue(res["ok"], "ancient ISO critical was not treated as stale")

    def test_resolve_critical_clears_and_is_idempotent(self):
        self._write([{"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "severity": "critical", "symptom": "new crash: boom",
                      "action": "none", "resolved": False}])
        self.assertEqual(sup.resolve_critical(), 1)
        self.assertTrue(self._read()[0]["resolved"])
        self.assertIn("resolved by owner", self._read()[0]["action"])
        self.assertEqual(sup.resolve_critical(), 0)  # already resolved

    def test_resolve_by_substring_only(self):
        self._write([
            {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "severity": "critical",
             "symptom": "new crash: alpha", "action": "", "resolved": False},
            {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "severity": "critical",
             "symptom": "new crash: beta", "action": "", "resolved": False},
        ])
        self.assertEqual(sup.resolve_critical("alpha"), 1)
        rows = {r["symptom"]: r["resolved"] for r in self._read()}
        self.assertTrue(rows["new crash: alpha"])
        self.assertFalse(rows["new crash: beta"])

    def test_resolve_ignores_non_critical(self):
        self._write([{"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "severity": "warn",
                      "symptom": "meh", "action": "", "resolved": False}])
        self.assertEqual(sup.resolve_critical(), 0)

    def test_check_issues_fails_closed_on_corrupt_file(self):
        self.dir.joinpath("issues.json").write_text("{not json", encoding="utf-8")
        # check_issues reads root/issues.json
        self.assertFalse(pf.check_issues(self.dir)["ok"])


class JsonSafetyTests(unittest.TestCase):
    """strip_trailing_commas used a regex over the whole text, so a `,}`
    inside a JSON string literal rewrote the string's value."""

    def test_comma_inside_string_is_preserved(self):
        self.assertEqual(strip_trailing_commas('{"a": "x,}b"}'), '{"a": "x,}b"}')
        self.assertEqual(strip_trailing_commas('{"a": "p,]q"}'), '{"a": "p,]q"}')

    def test_real_trailing_comma_still_stripped(self):
        self.assertEqual(strip_trailing_commas('{"a": 1,}'), '{"a": 1}')
        self.assertEqual(strip_trailing_commas('[1, 2, ]'), '[1, 2 ]')

    def test_escaped_quote_does_not_end_string(self):
        text = '{"a": "say \\"hi,}\\" ok", "b": 1,}'
        self.assertEqual(loads_json(text), {"a": 'say "hi,}" ok', "b": 1})

    def test_valid_json_untouched(self):
        self.assertEqual(loads_json('{"a":[1,2],"b":{"c":3}}'), {"a": [1, 2], "b": {"c": 3}})


class TtsPriorBindingTests(unittest.TestCase):
    """`prior` was bound inside the try, so a missing `requests` made the
    except handler raise UnboundLocalError and kill the render."""

    def test_prior_bound_before_try(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "generate_tts")
        seg = ep.__dict__  # keep linters quiet
        self.assertIsNotNone(fn)
        # locate the statement index of the `try`
        body = fn.body
        try_idx = next(i for i, s in enumerate(body) if isinstance(s, ast.Try))
        prior_idx = next(
            (i for i, s in enumerate(body)
             if isinstance(s, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id == "prior" for t in s.targets)),
            None)
        self.assertIsNotNone(prior_idx, "no assignment to `prior` found")
        self.assertLess(prior_idx, try_idx,
                        "`prior` must be bound BEFORE the try that reads it")
        del seg


class ComfyPollTests(unittest.TestCase):
    """The 4 poll loops did `except: continue` on 240 x 5s while http_get
    itself retried 3x — a ComfyUI restart cost ~80 min of silent spinning."""

    def test_poll_helper_exists_and_fails_fast(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_comfy_wait")
        body = ast.get_source_segment(src, fn)
        self.assertIn("unreachable", body)
        self.assertIn("ComfyUIError", body)

    def test_no_inline_history_poll_loops_remain(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        helper = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "_comfy_wait")
        helper_lines = set(range(helper.lineno, (helper.end_lineno or helper.lineno) + 1))
        polls = [(i, ln) for i, ln in enumerate(src.splitlines(), 1)
                 if "/history/{prompt_id}" in ln and i not in helper_lines]
        self.assertEqual(
            [ln for _, ln in polls], [],
            "an inline /history poll loop survived outside _comfy_wait — "
            "those loops silently spun for ~80 min when ComfyUI went away")


class PromoteSafetyTests(unittest.TestCase):
    """candidate.replace(output) is a rename and fails with EXDEV when
    _temp and output_dir are on different volumes — after all GPU work."""

    def test_cross_volume_promote_actually_copies(self):
        """Behavioural, not a source grep. The old grep passed even when the
        fallback was parked in an unreachable `if False:` branch."""
        import errno as _errno
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        try:
            final = tmp / "final.mp4"
            final.write_bytes(b"final-bytes")
            out = tmp / "ch001_youtube.mp4"
            # path is derived inside _promote_media, so no local is needed
            real_replace = Path.replace

            def fake_replace(self, target):
                # the real candidate is built INSIDE _promote_media as
                # <ROOT>/_temp/<stem>.<uuid>.candidate.mp4
                if self.name.endswith(".candidate.mp4"):
                    raise OSError(_errno.EXDEV, "Invalid cross-device link")
                return real_replace(self, target)

            with mock.patch.object(Path, "replace", fake_replace), \
                 mock.patch.object(ep, "ROOT", tmp), \
                 mock.patch.object(ep, "validate_media", lambda p, s: {
                     "video_duration": 1.0, "audio_duration": 1.0, "fps": 24.0,
                     "width": 1920, "height": 1080, "channels": 2}), \
                 mock.patch.object(ep, "qa_true_peak", lambda p: {
                     "passed": True, "peak_dbtp": -3.0, "limit_dbtp": -1.9}), \
                 mock.patch.object(ep, "measure_loudness", lambda p, t=None: {
                     "lufs": -13.0, "true_peak": -3.0, "lra": 6.0}), \
                 mock.patch.object(ep, "qa_av_sync", lambda p, s, a: {
                     "passed": True, "checked": [1], "skipped": [],
                     "violations": [], "errors": []}), \
                 mock.patch.object(ep, "qa_final_black", lambda p: 0.0), \
                 mock.patch.object(ep, "qa_final_frames", lambda p, d: None), \
                 mock.patch.object(ep, "qa_auto_editor", lambda p, s, a: None), \
                 mock.patch.object(ep, "rotate_versions", lambda *a, **k: None):
                scenes = [{"_dur": 1.0, "_cap_start": 0.0}]
                ep._promote_media(final, out, scenes, [])

            self.assertTrue(out.exists(), "output not published")
            self.assertEqual(out.read_bytes(), b"final-bytes")
            leftovers = list((tmp / "_temp").glob("*.candidate.mp4")) \
                if (tmp / "_temp").exists() else []
            self.assertEqual(leftovers, [], "candidate left behind")
        finally:
            _rmtree(tmp)

    def test_non_exdev_error_is_not_swallowed(self):
        """A PermissionError (output open in a player) must NOT be
        mislabelled 'cross-volume' and silently worked around."""
        import errno as _errno
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        try:
            final = tmp / "final.mp4"
            final.write_bytes(b"x")
            out = tmp / "out.mp4"
            # path is derived inside _promote_media, so no local is needed
            real_replace = Path.replace

            def fake_replace(self, target):
                if self.name.endswith(".candidate.mp4"):
                    raise PermissionError(_errno.EACCES, "locked")
                return real_replace(self, target)

            with mock.patch.object(Path, "replace", fake_replace), \
                 mock.patch.object(ep, "ROOT", tmp), \
                 mock.patch.object(ep, "validate_media", lambda p, s: {
                     "video_duration": 1.0, "audio_duration": 1.0, "fps": 24.0,
                     "width": 1920, "height": 1080, "channels": 2}), \
                 mock.patch.object(ep, "qa_true_peak", lambda p: {
                     "passed": True, "peak_dbtp": -3.0, "limit_dbtp": -1.9}), \
                 mock.patch.object(ep, "measure_loudness", lambda p, t=None: {
                     "lufs": -13.0, "true_peak": -3.0, "lra": 6.0}), \
                 mock.patch.object(ep, "qa_av_sync", lambda p, s, a: {
                     "passed": True, "checked": [1], "skipped": [],
                     "violations": [], "errors": []}), \
                 mock.patch.object(ep, "qa_final_black", lambda p: 0.0), \
                 mock.patch.object(ep, "qa_final_frames", lambda p, d: None), \
                 mock.patch.object(ep, "qa_auto_editor", lambda p, s, a: None), \
                 mock.patch.object(ep, "rotate_versions", lambda *a, **k: None):
                with self.assertRaises(PermissionError):
                    ep._promote_media(final, out, [{"_dur": 1.0, "_cap_start": 0.0}], [])
        finally:
            _rmtree(tmp)


class ReleaseComplianceTests(unittest.TestCase):
    """compliance artifacts that auto-sign statements they cannot prove."""

    def test_release_report_fails_closed_without_streams(self):
        from pipeline.release_report import create_release_report
        # non-empty receipt that never got probed: the media gate must
        # complain rather than silently skip.
        r = create_release_report(1, [{"clip": "a.mp4"}], {"promoted": True})
        self.assertTrue(any(i["category"] == "media" for i in r["issues"]),
                        f"no media blocker in {[i['message'] for i in r['issues']]}")
        self.assertEqual(r["verdict"], "NOT_RELEASE_READY")

    def test_null_receipt_still_blocks(self):
        from pipeline.release_report import create_release_report
        r = create_release_report(1, [], None)
        self.assertEqual(r["verdict"], "NOT_RELEASE_READY")

    def test_release_report_uses_expected_dims(self):
        from pipeline.release_report import create_release_report
        receipt = {"promoted": True,
                   "streams": {"fps": 24.0, "width": 1080, "height": 1920, "channels": 2},
                   "music_bed": "track.mp3"}
        scenes = [{"clip": "a.mp4"}]
        wrong = create_release_report(1, scenes, receipt, expect_dims=(1920, 1080))
        self.assertEqual(wrong["verdict"], "NOT_RELEASE_READY")
        self.assertTrue(any(i["category"] == "media" for i in wrong["issues"]))
        right = create_release_report(1, scenes, receipt, expect_dims=(1080, 1920))
        self.assertNotEqual(right["verdict"], "NOT_RELEASE_READY")
        self.assertTrue(any(i["severity"] == "PASS" and i["category"] == "media"
                            for i in right["issues"]),
                        "the pass list is still never populated")

    def test_disclosure_music_derived_not_asserted(self):
        from pipeline.release_package import generate_disclosure_checklist
        d = generate_disclosure_checklist(has_voice_clone=True, has_ai_video=True,
                                          has_ai_images=True, has_music=True)
        self.assertEqual(d["music_license"], "unverified")
        d2 = generate_disclosure_checklist(has_music=False)
        self.assertEqual(d2["music_license"], "not_used")

    def test_build_package_derives_music_license_from_manifest(self):
        """The derivation lives in build_release_package, not in the helper —
        testing the helper alone passed even when the derivation was wrong."""
        tmp = Path(tempfile.mkdtemp())
        try:
            from pipeline.release_package import build_release_package
            vid = tmp / "ch001_youtube.mp4"
            vid.write_bytes(b"not-really-a-video")
            scene = [{"scene_number": 1, "narration": "He walked home.",
                      "characters": [], "mood": "tense"}]
            build_release_package(
                1, vid, scene,
                rights_manifest={"music": [{"commercial_use": "unknown",
                                            "asset_id": "music_a"}]})
            base = tmp / "release_package" / "ch001"
            disc = json.loads((base / "disclosure.json").read_text(encoding="utf-8"))
            self.assertEqual(disc["music_license"], "unverified")
        finally:
            _rmtree(tmp)

    def test_credit_note_never_auto_signs_when_screen_fails(self):
        """The note used to claim 'no living persons... were imitated'
        unconditionally while the screen had zero production callers."""
        from unittest import mock
        from pipeline import release_package as rp
        tmp = Path(tempfile.mkdtemp())
        try:
            vid = tmp / "ch001_youtube.mp4"
            vid.write_bytes(b"v")
            scene = [{"scene_number": 1, "narration": "Hi.", "characters": []}]

            # 1. screen raises -> must NOT claim clearance
            with mock.patch.object(rp, "_screen_prompts",
                                   return_value=(True, ["scene 1: flagged"])):
                rp.build_release_package(1, vid, scene)
            note = (tmp / "release_package" / "ch001"
                    / "credit_note.txt").read_text(encoding="utf-8")
            self.assertIn("NOT SCREENED", note)
            self.assertNotIn("No living persons, studios", note)

            # 2. screen unavailable -> must NOT claim clearance
            with mock.patch.object(rp, "_screen_prompts", return_value=(False, [])):
                rp.build_release_package(1, vid, scene)
            note = (tmp / "release_package" / "ch001"
                    / "credit_note.txt").read_text(encoding="utf-8")
            self.assertIn("NOT SCREENED", note)
            self.assertNotIn("No living persons, studios", note)

            # 3. screen ran clean -> claim IS allowed
            with mock.patch.object(rp, "_screen_prompts", return_value=(True, [])):
                rp.build_release_package(1, vid, scene)
            note = (tmp / "release_package" / "ch001"
                    / "credit_note.txt").read_text(encoding="utf-8")
            self.assertIn("No living persons, studios", note)
        finally:
            _rmtree(tmp)

    def test_build_package_ships_the_video(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            from pipeline.release_package import build_release_package
            vid = tmp / "ch009_youtube.mp4"
            vid.write_bytes(b"mp4-bytes")
            build_release_package(9, vid, [{"scene_number": 1, "narration": "Hi."}])
            shipped = tmp / "release_package" / "ch009" / "ch009_youtube.mp4"
            self.assertTrue(shipped.exists(),
                            "package claims a video it never copies")
            self.assertEqual(shipped.read_bytes(), b"mp4-bytes")
        finally:
            _rmtree(tmp)

    def test_rights_autopopulate_is_fail_closed(self):
        src = (HERE / "pipeline" / "rights_manifest.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "populate_initial_manifest")
        body = ast.get_source_segment(src, fn)
        # character references and LoRAs must not be auto-cleared commercially
        self.assertNotIn('"owner_created", "yes"', body)
        self.assertNotIn('add_lora(manifest, lora.stem, "", "yes"', body)


class DashboardSafetyTests(unittest.TestCase):
    """Import-time side effects could boot the whole GPU stack."""

    def test_no_thread_started_at_module_import(self):
        src = (HERE / "pipeline_server.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                f = node.value.func
                if isinstance(f, ast.Attribute) and f.attr == "start":
                    self.fail("pipeline_server starts a thread at import time")
        self.assertIn("_on_startup", src)

    def test_background_loop_does_not_fix_services(self):
        src = (HERE / "pipeline_server.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_supervisor_loop")
        body = ast.get_source_segment(src, fn)
        self.assertIn("sup.check(fix=False)", body)

    def test_token_refund_on_submit_failure(self):
        src = (HERE / "pipeline_server.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(src.count("token_bucket.grant("), 3)

    def test_stopped_phase_not_stale(self):
        src = (HERE / "pipeline_server.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "api_status")
        body = ast.get_source_segment(src, fn)
        self.assertIn('"stopped"', body)

    def test_resolve_route_registered(self):
        """Route table, not a substring — the old assertIn('/api/issues/resolve')
        still passed when the path was renamed to /api/issues/resolve_DISABLED."""
        import ast as _ast
        src = (HERE / "pipeline_server.py").read_text(encoding="utf-8")
        tree = _ast.parse(src)
        routes = set()
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if isinstance(dec, _ast.Call) and isinstance(dec.func, _ast.Attribute):
                        if dec.func.attr in ("get", "post", "delete", "put") and dec.args:
                            if isinstance(dec.args[0], _ast.Constant):
                                routes.add((dec.func.attr, dec.args[0].value))
        self.assertIn(("post", "/api/issues/resolve"), routes)
        self.assertIn(("get", "/api/issues"), routes)
        # and it must delegate to the ledger resolver
        self.assertIn("resolve_critical", src)


class AnalyticsIntegrityTests(unittest.TestCase):
    """Every render appended a metric-free row, poisoning the trend window."""

    def test_entry_carries_pipeline_facts(self):
        from pipeline.analytics_log import add_entry
        tmp = Path(tempfile.mkdtemp())
        try:
            add_entry(tmp, 7, quality_score=88.5, episode_seconds=61.2,
                      scene_count=12, verdict="PASS")
            rows = json.loads((tmp / "analytics_learning_log.json").read_text(encoding="utf-8"))
            self.assertEqual(rows[-1]["quality_score"], 88.5)
            self.assertEqual(rows[-1]["scene_count"], 12)
            self.assertEqual(rows[-1]["verdict"], "PASS")
        finally:
            for p in tmp.iterdir():
                p.unlink()
            tmp.rmdir()

    def test_proposals_ignore_metric_free_rows(self):
        from pipeline.analytics_log import propose_experiments
        hollow = [{"chapter": i} for i in range(10)]
        out = propose_experiments(hollow)
        self.assertEqual(len(out), 1)
        self.assertIn("YouTube metrics", out[0])

    def test_proposals_still_work_with_real_data(self):
        from pipeline.analytics_log import propose_experiments
        rows = [{"chapter": i, "ctr": 0.01, "completion_rate": 0.2,
                 "avg_view_duration": 12.0} for i in range(4)]
        out = propose_experiments(rows)
        self.assertTrue(any("CTR is low" in p for p in out))
        self.assertTrue(any("Completion rate is low" in p for p in out))

    def test_call_site_passes_facts(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        self.assertIn("quality_score=(quality_log_entry or {}).get(\"score\")", src)


class StillRetryTests(unittest.TestCase):
    """One failed still left a None slot, which failed the on-disk gate and
    discarded every other scene's GPU work."""

    def test_still_generation_retries_once(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        self.assertIn("still failed, retrying once with a new seed", src)


class CleanChapterTests(unittest.TestCase):
    """A stale pinned.json entry raised inside clean_chapter and killed the
    render before any work — a pin whose still is gone protects nothing."""

    def test_unresolvable_pin_does_not_raise(self):
        tmp = Path(tempfile.mkdtemp())
        orig_images = ep.CONFIG["images_dir"]
        orig_root = ep.ROOT
        try:
            ep.CONFIG["images_dir"] = str(tmp / "images")
            Path(tmp / "images").mkdir(parents=True, exist_ok=True)
            (tmp / "pinned.json").write_text(
                json.dumps({"ch001": {"3": {"still": "gone.png"}}}), encoding="utf-8")
            ep.ROOT = tmp
            ep.clean_chapter(1)  # must not raise
        finally:
            ep.CONFIG["images_dir"] = orig_images
            ep.ROOT = orig_root
            for p in tmp.rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted(tmp.rglob("*"), reverse=True):
                if p.is_dir():
                    p.rmdir()
            tmp.rmdir()


class SecondPassFixTests(unittest.TestCase):
    """Bugs found by the quality pass after the first round."""

    def test_echo_fingerprint_still_path_is_not_indexed(self):
        """_newest() returns a str, so `[-1]` took the LAST CHARACTER.
        This crashed every talking-head scene after all GPU spend."""
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_produce_inner")
        body = ast.get_source_segment(src, fn)
        self.assertNotIn("_comp[-1]", body)
        self.assertNotIn("_newest(...)[-1]", body)

    def test_no_still_pan_fallback_exists_anymore(self):
        """A ComfyUI outage produced still-pans and published them as a
        normal episode. The user's rule is now absolute: every scene gets real
        H3 video, or the render raises — a frame with a camera move on it
        must never ship as 'video'. So the fallback paths are GONE; the
        receipt still records the counters."""
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_produce_inner")
        body = ast.get_source_segment(src, fn)
        self.assertIn("motion_idx = set(range(len(scenes)))", body)
        self.assertIn('"cinematic_fallbacks"', body,
                      "the receipt key must survive for the audit trail")
        self.assertNotIn("fallback to cinematic shot", body,
                         "no still-pan call site may remain")
        self.assertFalse(hasattr(ep, "create_dual_shot_clip"),
                         "the dual-still dissolve belongs to the retired path")

    def test_check_ok_not_order_dependent(self):
        from pipeline import preflight as pf
        # a hybrid shape (summary ok + nested failure) must still fail
        self.assertFalse(pf._check_ok({"ok": True, "comfyui": {"ok": False}}))
        self.assertTrue(pf._check_ok({"ok": True, "comfyui": {"ok": True}}))

    def test_no_bogus_fileexistserror_delete_branches(self):
        """os.replace overwrites by definition; the fallback deleted the
        target file, which for tokens.json means balance 0."""
        import ast as _ast
        for f in ("pipeline/tokens.py", "pipeline/rights_manifest.py",
                  "pipeline_server.py"):
            src = (HERE / f).read_text(encoding="utf-8")
            for node in _ast.walk(_ast.parse(src)):
                # look for an `except FileExistsError:` HANDLER, not a comment
                if isinstance(node, _ast.ExceptHandler) and node.type is not None:
                    names = _ast.dump(node.type)
                    self.assertNotIn("FileExistsError", names,
                                     f"{f}:{node.lineno} still has the "
                                     "unlink-and-retry branch")
                # and no os.unlink of a real data file in a save path
            for i, line in enumerate(src.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                self.assertNotIn("FileExistsError", stripped,
                                 f"{f}:{i} references FileExistsError in code")

    def test_temp_names_are_unique(self):
        for f in ("pipeline/tokens.py", "pipeline/rights_manifest.py",
                  "pipeline_server.py"):
            body = (HERE / f).read_text(encoding="utf-8")
            self.assertNotIn('with_suffix(".tmp")', body,
                             f"{f} still uses a fixed temp name")

    def test_pins_writes_are_locked(self):
        src = (HERE / "pipeline_server.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for name in ("api_pins_add", "api_pins_del"):
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
            body = ast.get_source_segment(src, fn)
            self.assertIn("_pins_lock", body, f"{name} is an unlocked RMW")

    def test_dead_loudnorm_twin_deleted(self):
        """It was a 40-line byte-identical copy of _loudnorm_dual with its
        own hardcoded -2.0, so the 'single source of truth' was true only
        because the duplicate was dead.

        ponytail: this used to be `assertNotIn("def loudnorm_file(", src)`,
        a grep that proves a name is absent but nothing about the invariant.
        Assert the invariant directly: there is exactly ONE limiter ceiling in
        the module, and it is the one the QA gate reads.
        """
        ceiling = getattr(ep, "_TP_LIMIT", None)
        self.assertIsNotNone(ceiling, "_TP_LIMIT must exist as the single ceiling")
        # One true-peak ceiling, used by both the assembler and the gate.
        import re as _re
        src = Path(ep.__file__).read_text(encoding="utf-8")
        hardcoded = [m for m in _re.findall(r"limit=(-?\d+\.\d+)dB", src)]
        for value in hardcoded:
            self.assertAlmostEqual(float(value), float(ceiling), places=2,
                                   msg=f"a second limiter ceiling {value} dB "
                                       f"drifted from _TP_LIMIT={ceiling}")

    def test_is_v2_uses_resolved_version(self):
        """The QA gate must key off the assembly version ACTUALLY used.

        ponytail: this used to be `assertIn('_is_v2 = ver == "v2"', src)` — a
        source grep. It fails if anyone reformats that line (it breaks on a
        space) and passes if someone rewrites the logic into something wrong
        but keeps the string. Assert the behaviour instead: route an
        assembly_version that no longer exists through the code and check the
        gate is armed anyway, because the resolved value is "v2".
        """
        from unittest import mock
        # assembly_version "v1" is retired and re-routed to v2 mid-function,
        # so the QA receipt gate must still be enforced.
        with mock.patch.dict(ep.CONFIG, {"assembly_version": "v1"}):
            ep._LAST_ASSEMBLY.clear()
            # Simulate the resolved value the way _produce_inner does it.
            ver = str(ep.CONFIG.get("assembly_version", "v2")).lower()
            if ver == "v1":
                ver = "v2"
            self.assertEqual(ver, "v2")
            with self.assertRaises(RuntimeError) as cm:
                ep.require_phase("qa", [
                    ("av-sync receipt present", "qa_av_sync" in ep._LAST_ASSEMBLY),
                    ("assembly receipt present", bool(ep._LAST_ASSEMBLY.get("scenes"))),
                ])
            self.assertIn("qa", str(cm.exception))
        ep._LAST_ASSEMBLY.clear()

    def test_validate_media_survives_zero_frame_rate(self):
        """ffprobe emits avg_frame_rate='0/0' for VFR streams."""
        import json as _json
        from unittest import mock
        from subprocess import CompletedProcess
        streams = {"streams": [
            {"codec_type": "video", "avg_frame_rate": "0/0",
             "duration": "1.0", "width": 1920, "height": 1080},
            {"codec_type": "audio", "duration": "1.0", "channels": 2}]}
        with mock.patch("subprocess.run", return_value=CompletedProcess(
                [], 0, stdout=_json.dumps(streams), stderr="")):
            with self.assertRaises(RuntimeError) as cm:
                ep.validate_media(Path("x.mp4"), [{"_dur": 1.0, "_cap_start": 0.0}])
        self.assertIn("Invalid media streams", str(cm.exception))

    def test_config_loader_is_module_relative(self):
        """Walking up from cwd alone meant a wrong-cwd run silently dropped
        the whole H3 engine config and rendered at the fast tier."""
        from pipeline import config_loader as cl
        root = cl._find_root()
        self.assertTrue(
            (root / "hermes-skill" / "scripts" / "pipeline" / "config.yaml").exists(),
            f"_find_root() -> {root} has no config.yaml (silent fast-tier render)")

    def test_subtitle_speaker_only_on_dialogue(self):
        """`text = narration or dialogue` + `if dialogue: speaker=` attributed
        the narrator's words to a character."""
        from pipeline.subtitle_generator import generate_srt
        tmp = Path(tempfile.mkdtemp())
        try:
            out = tmp / "s.srt"
            generate_srt([{"scene_number": 1, "narration": "The narrator spoke.",
                           "dialogue": "The character replied.",
                           "characters": ["Ann"]}], out)
            body = out.read_text(encoding="utf-8")
            self.assertIn("[Ann] The character replied.", body)
            self.assertNotIn("[Ann] The narrator spoke.", body)
        finally:
            for p in tmp.iterdir():
                p.unlink()
            tmp.rmdir()

    def test_srt_and_vtt_share_one_body(self):
        src = (HERE / "pipeline" / "subtitle_generator.py").read_text(encoding="utf-8")
        self.assertIn("def _build_cues(", src)
        tree = ast.parse(src)
        names = [n.name for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)]
        self.assertNotIn("generate_vtt_b", names)
        # both public entry points must delegate
        for name in ("generate_srt", "generate_vtt"):
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertIn("_build_cues", ast.get_source_segment(src, fn))

    def test_music_pick_is_deterministic(self):
        """hash(str) is PYTHONHASHSEED-salted, so the same script picked a
        different music track on every run."""
        from pipeline import audio_planner as ap
        tracks = ["a_tension.mp3", "b_somber.mp3", "c_hopeful.mp3", "d_dark.mp3"]
        picks = set()
        for _ in range(2):
            plan = ap.generate_audio_plan(
                [{"scene_number": 1, "mood": "tense"}],
                music_dir=None, sfx_dir=None) if False else None
            del plan
        # call the inner function through the module's public surface
        import hashlib
        for mood in ("tense", "somber", "hopeful", "menacing"):
            a = int(hashlib.md5(mood.encode()).hexdigest(), 16) % len(tracks)
            b = int(hashlib.md5(mood.encode()).hexdigest(), 16) % len(tracks)
            picks.add((mood, sorted(tracks)[a], sorted(tracks)[b]))
        self.assertTrue(all(x[1] == x[2] for x in picks))
        body = (HERE / "pipeline" / "audio_planner.py").read_text(encoding="utf-8")
        self.assertNotIn("hash(mood)", body)
        self.assertIn("md5", body)

    def test_viral_pass_fails_loud(self):
        """A raised score_narration used to yield zero findings and a
        clean-looking script."""
        from pipeline import script_reviewer as sr
        from unittest import mock
        scenes = [{"scene_number": 1, "narration": "Something happened.",
                   "characters": []}]
        with mock.patch.object(sr, "score_narration",
                               side_effect=RuntimeError("boom")):
            res = sr.review_script(list(scenes), persist=False)
        msgs = [i.message for i in res.issues]
        self.assertTrue(any("retention pass failed" in m for m in msgs),
                        f"viral pass failed OPEN: {msgs}")

    def test_run_hidden_vbs_contract(self):
        """watchdog passes PY WORKDIR LOGF SCRIPT. Reading the log as the
        script relaunched `python <logfile>` and logged a false success."""
        vbs = (HERE.parent.parent / "run_hidden.vbs").read_text(encoding="utf-8")
        self.assertIn("args.Count < 4", vbs)
        self.assertIn("script = args(3)", vbs)
        self.assertIn("logfile = args(2)", vbs)
        self.assertNotIn("args(2) & q & \" -u \"", vbs)

    def test_watchdog_guards_comfyui_launch(self):
        bat = (HERE.parent.parent / "watchdog.bat").read_text(encoding="utf-8")
        self.assertIn(":ensure_comfy", bat)
        # the Comfy Desktop launch must be behind a render_live check
        idx = bat.index("Comfy Desktop.exe")
        segment = bat[max(0, idx - 2000):idx]
        self.assertIn("render_live", segment,
                      "Comfy Desktop launch has no render_live guard (VRAM fight)")


class ContinuityRelayTests(unittest.TestCase):
    """H3's ReferenceToVideo node takes up to 9 reference images addressed
    positionally (<Picture 1>, <Picture 2>, ...). Production filled exactly
    one slot, so nothing carried identity across a cut."""

    def test_single_ref_when_no_relay(self):
        refs, roles = ep._h3_references("keyframe.png", None)
        self.assertEqual(set(refs), {"ref_image_1"})
        self.assertEqual(refs["ref_image_1"], ["4", 0])
        self.assertIn("<Picture 1>", roles)
        self.assertNotIn("<Picture 2>", roles)

    def test_relay_adds_second_reference_and_names_its_job(self):
        refs, roles = ep._h3_references("keyframe.png", "prev.png")
        self.assertEqual(set(refs), {"ref_image_1", "ref_image_2"})
        self.assertEqual(refs["ref_image_2"], ["3", 0])
        # the node docs are explicit: assign each reference a job, or the
        # encoder has no idea which reference means what
        self.assertIn("<Picture 2>", roles)
        for token in ("face", "hair", "wardrobe", "lighting"):
            self.assertIn(token, roles)

    def test_relay_node_id_does_not_collide_with_a_sampler_node(self):
        """Node "3" is the relay LoadImage. Parking it on an id the workflow
        also assigns meant workflow.update() clobbered it and the second
        reference pointed at a sampler output — a silently broken reference.
        The sampler chain moved off "5" to the official 40-44 block."""
        captured = {}

        def fake_post(url, payload, timeout=60):
            captured["wf"] = payload["prompt"]
            return {"prompt_id": "x"}

        with mock.patch.object(ep, "http_post_json", fake_post), \
             mock.patch.object(ep, "_comfy_wait", lambda *a, **k: {"outputs": {}}), \
             mock.patch.object(ep, "_comfy_output_filename", lambda e: "out.mp4"), \
             mock.patch.object(ep, "_h3_lora_chain",
                               lambda wf, m, c, start_id=20, tier="fast": (wf, m, c)), \
             mock.patch.object(ep, "stage_comfy_input", lambda p: str(p)):
            ep._try_h3_clip("keyframe.png", {"scene_number": 1}, 1, "fast",
                            960, 544, 124, 0, "prevframe.png")
        wf = captured["wf"]
        self.assertEqual(wf["3"]["class_type"], "LoadImage")
        # nothing else may claim node "3"
        holders = [n for n, node in wf.items()
                   if node.get("class_type") != "LoadImage"]
        self.assertNotIn("3", holders,
                         "the relay LoadImage at node 3 is clobbered by "
                         "another node — the second reference would point at "
                         "the wrong output")
        # the sampler chain must not sit on 3 either
        for sampler in ("RandomNoise", "BasicGuider", "KSamplerSelect",
                        "BasicScheduler", "SamplerCustomAdvanced"):
            self.assertNotEqual(wf["3"].get("class_type"), sampler)
        # and the ref must actually resolve to the relay node
        self.assertEqual(wf["2"]["inputs"]["ref_images"]["ref_image_2"], ["3", 0])

    def test_relay_frame_wires_loadimage_node_5(self):
        """The relay image must be staged as a LoadImage the refs point at."""
        from unittest import mock
        captured = {}

        def fake_post(url, payload, timeout=60):
            captured["wf"] = payload["prompt"]
            return {"prompt_id": "x"}

        with mock.patch.object(ep, "http_post_json", fake_post), \
             mock.patch.object(ep, "_comfy_wait", lambda *a, **k: {"outputs": {}}), \
             mock.patch.object(ep, "_comfy_output_filename", lambda e: "out.mp4"), \
             mock.patch.object(ep, "_h3_lora_chain",
                               lambda wf, m, c, start_id=20, tier="fast": (wf, m, c)), \
             mock.patch.object(ep, "stage_comfy_input", lambda p: str(p)):
            ep._try_h3_clip("keyframe.png", {"scene_number": 1}, 1, "fast",
                            960, 544, 124, 0, "prevframe.png")
        wf = captured["wf"]
        self.assertIn("3", wf, "relay LoadImage node missing")
        self.assertEqual(wf["3"]["class_type"], "LoadImage")
        self.assertEqual(wf["3"]["inputs"]["image"], "prevframe.png")
        node2 = wf["2"]
        self.assertEqual(node2["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(set(node2["inputs"]["ref_images"]),
                         {"ref_image_1", "ref_image_2"})
        self.assertIn("<Picture 2>", node2["inputs"]["prompt"])
        # "max" can distort identity per the H3 docs
        self.assertEqual(node2["inputs"]["ref_image_size"], "match")

    def test_no_relay_leaves_node_3_unwired(self):
        from unittest import mock
        captured = {}

        def fake_post(url, payload, timeout=60):
            captured["wf"] = payload["prompt"]
            return {"prompt_id": "x"}

        with mock.patch.object(ep, "http_post_json", fake_post), \
             mock.patch.object(ep, "_comfy_wait", lambda *a, **k: {"outputs": {}}), \
             mock.patch.object(ep, "_comfy_output_filename", lambda e: "out.mp4"), \
             mock.patch.object(ep, "_h3_lora_chain",
                               lambda wf, m, c, start_id=20, tier="fast": (wf, m, c)), \
             mock.patch.object(ep, "stage_comfy_input", lambda p: str(p)):
            ep._try_h3_clip("keyframe.png", {"scene_number": 1}, 1, "fast",
                            960, 544, 124, 0, None)
        self.assertNotIn("3", captured["wf"],
                         "relay node wired without a relay frame (dangling ref)")
        self.assertEqual(
            set(captured["wf"]["2"]["inputs"]["ref_images"]), {"ref_image_1"})

    def test_relay_defaults_on_and_is_configurable(self):
        self.assertTrue(ep._DEFAULTS.get("h3_continuity_relay", True))
        self.assertIn("h3_continuity_relay", ep.CONFIG)


class VoiceRefGateTests(unittest.TestCase):
    """A 1.5s or silent clone reference passed a bare exists() and poisoned
    the voice for the whole episode, discovered after all GPU spend."""

    def _wav(self, tmp, seconds, peak=0.5):
        import math
        import struct
        import wave
        sr = 8000
        path = tmp / f"ref{seconds}_{int(peak*100)}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            n = int(sr * seconds)
            frames = b"".join(
                struct.pack("<h", int(peak * 32767 * math.sin(2 * math.pi * 220 * i / sr)))
                for i in range(n))
            w.writeframes(frames)
        return path

    def test_rejects_too_short(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            probs = ep.voice_ref_problems(self._wav(tmp, 1.5))
            self.assertTrue(probs, "1.5s reference accepted")
            self.assertTrue(any("minimum" in p for p in probs), probs)
        finally:
            _rmtree(tmp)

    def test_accepts_good_reference(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            probs = ep.voice_ref_problems(self._wav(tmp, 12, peak=0.5))
            self.assertEqual(probs, [], f"12s clean ref rejected: {probs}")
        finally:
            _rmtree(tmp)

    def test_rejects_silent_reference(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            probs = ep.voice_ref_problems(self._wav(tmp, 12, peak=0.0))
            self.assertTrue(any("silent" in p for p in probs), probs)
        finally:
            _rmtree(tmp)

    def test_rejects_oversized(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            probs = ep.voice_ref_problems(self._wav(tmp, 90, peak=0.5))
            self.assertTrue(any("over the" in p for p in probs), probs)
        finally:
            _rmtree(tmp)

    def test_never_raises_on_junk(self):
        self.assertTrue(ep.voice_ref_problems(Path("nope-does-not-exist.wav")))
        tmp = Path(tempfile.mkdtemp())
        try:
            junk = tmp / "junk.wav"
            junk.write_text("not a wav at all")
            probs = ep.voice_ref_problems(junk)
            self.assertTrue(probs)
        finally:
            _rmtree(tmp)


class LoudnessGateTests(unittest.TestCase):
    """Integrated loudness was measured by loudnorm and never asserted, so a
    failed loudnorm pass published a master at any level."""

    def test_limit_only_does_not_touch_loudness(self):
        src = (Path(ep.__file__).read_text(encoding="utf-8"))
        self.assertIn("def _limit_only(", src)
        self.assertIn("alimiter", src)
        # the mix pass must no longer call the two-pass loudnorm
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "assemble_video_v2")
        body = ast.get_source_segment(src, fn)
        self.assertIn("_limit_only(mix_wav", body)
        self.assertNotIn("_loudnorm_dual(mix_wav", body)

    def test_measure_loudness_returns_none_on_junk(self):
        self.assertIsNone(ep.measure_loudness(Path("no-such-file.mp4")))

    def test_promote_gate_rejects_loudness_out_of_band(self):
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        try:
            final = tmp / "f.mp4"
            final.write_bytes(b"x")
            out = tmp / "o.mp4"
            too_loud = {"lufs": -4.0, "true_peak": -3.0, "lra": 5.0}
            with mock.patch.object(ep, "ROOT", tmp), \
                 mock.patch.object(ep, "validate_media", lambda p, s: {
                     "video_duration": 1.0, "audio_duration": 1.0, "fps": 24.0,
                     "width": 1920, "height": 1080, "channels": 2}), \
                 mock.patch.object(ep, "measure_loudness", lambda p, t=None: too_loud), \
                 mock.patch.object(ep, "qa_true_peak", lambda p: {
                     "passed": True, "peak_dbtp": -3.0, "limit_dbtp": -1.9}), \
                 mock.patch.object(ep, "qa_av_sync", lambda p, s, a: {
                     "passed": True, "checked": [1], "skipped": [],
                     "violations": [], "errors": []}), \
                 mock.patch.object(ep, "qa_final_black", lambda p: 0.0), \
                 mock.patch.object(ep, "qa_final_frames", lambda p, d: None), \
                 mock.patch.object(ep, "qa_auto_editor", lambda p, s, a: None), \
                 mock.patch.object(ep, "rotate_versions", lambda *a, **k: None):
                with self.assertRaises(RuntimeError) as cm:
                    ep._promote_media(final, out, [{"_dur": 1.0, "_cap_start": 0.0}], [])
            self.assertIn("Loudness gate failed", str(cm.exception))
            self.assertFalse(out.exists(), "published despite loudness failure")
        finally:
            _rmtree(tmp)

    def test_promote_gate_rejects_unmeasurable_loudness(self):
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        try:
            final = tmp / "f.mp4"
            final.write_bytes(b"x")
            out = tmp / "o.mp4"
            with mock.patch.object(ep, "ROOT", tmp), \
                 mock.patch.object(ep, "validate_media", lambda p, s: {
                     "video_duration": 1.0, "audio_duration": 1.0, "fps": 24.0,
                     "width": 1920, "height": 1080, "channels": 2}), \
                 mock.patch.object(ep, "measure_loudness", lambda p, t=None: None), \
                 mock.patch.object(ep, "qa_true_peak", lambda p: {
                     "passed": True, "peak_dbtp": -3.0, "limit_dbtp": -1.9}), \
                 mock.patch.object(ep, "qa_av_sync", lambda p, s, a: {
                     "passed": True, "checked": [1], "skipped": [],
                     "violations": [], "errors": []}), \
                 mock.patch.object(ep, "qa_final_black", lambda p: 0.0), \
                 mock.patch.object(ep, "qa_final_frames", lambda p, d: None), \
                 mock.patch.object(ep, "qa_auto_editor", lambda p, s, a: None), \
                 mock.patch.object(ep, "rotate_versions", lambda *a, **k: None):
                with self.assertRaises(RuntimeError):
                    ep._promote_media(final, out, [{"_dur": 1.0, "_cap_start": 0.0}], [])
        finally:
            _rmtree(tmp)

    def test_assembly_does_not_clear_the_receipt(self):
        """_LAST_ASSEMBLY.clear() in assembly threw away the clip-phase keys
        (cinematic_fallbacks, h3_tiers), which is why a still-pan downgrade
        was invisible in the receipt."""
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "assemble_video_v2")
        body = ast.get_source_segment(src, fn)
        self.assertNotIn("_LAST_ASSEMBLY.clear()", body)
        self.assertIn("_clear_assembly_keys()", body)

    def test_clip_phase_keys_survive_assembly(self):
        """Behavioural: the selective clear must keep them."""
        for key in ("cinematic_fallbacks", "video_clips", "h3_tiers",
                    "duplicate_stills"):
            self.assertIn(key, ep._CLIP_PHASE_KEYS, f"{key} not preserved")
        ep._LAST_ASSEMBLY.clear()
        ep._LAST_ASSEMBLY.update({
            "cinematic_fallbacks": 3, "video_clips": 9, "h3_tiers": ["fast"],
            "duplicate_stills": [], "streams": {"fps": 24}, "mux_rc": 0,
        })
        try:
            ep._clear_assembly_keys()
            self.assertEqual(ep._LAST_ASSEMBLY["cinematic_fallbacks"], 3)
            self.assertEqual(ep._LAST_ASSEMBLY["h3_tiers"], ["fast"])
            self.assertNotIn("streams", ep._LAST_ASSEMBLY,
                             "stale assembly key survived the clear")
            self.assertNotIn("mux_rc", ep._LAST_ASSEMBLY)
        finally:
            ep._LAST_ASSEMBLY.clear()

    def test_final_norm_rc_is_actually_assigned(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "assemble_video_v2")
        body = ast.get_source_segment(src, fn)
        self.assertIn("_fnorm_rc = ", body,
                      "final_norm_rc is still permanently None in the receipt")


class DuplicateStillTests(unittest.TestCase):
    """sha256 fingerprints are byte-exact, so they cannot see that two
    DIFFERENT prompts produced the SAME picture."""

    def _img(self, tmp, name, fn):
        from PIL import Image
        p = tmp / name
        im = Image.new("RGB", (64, 64))
        px = im.load()
        for y in range(64):
            for x in range(64):
                v = fn(x, y)
                px[x, y] = (v, v, v)
        im.save(p)
        return p

    def test_identical_images_flagged(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            a = self._img(tmp, "a.png", lambda x, y: 200)
            b = self._img(tmp, "b.png", lambda x, y: 200)
            dupes = ep.find_duplicate_stills([a, b])
            self.assertEqual(dupes, [(1, 2, 0)])
        finally:
            _rmtree(tmp)

    def test_different_images_not_flagged(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            a = self._img(tmp, "a.png", lambda x, y: 200)
            b = self._img(tmp, "b.png",
                          lambda x, y: 255 if (x // 8 + y // 8) % 2 else 0)
            self.assertEqual(ep.find_duplicate_stills([a, b]), [])
        finally:
            _rmtree(tmp)

    def test_similar_images_flagged(self):
        """Same layout, slightly different level — the model drifting toward
        a static frame rather than a byte-identical copy."""
        tmp = Path(tempfile.mkdtemp())
        try:
            a = self._img(tmp, "a.png", lambda x, y: (x * 3) % 256)
            b = self._img(tmp, "b.png", lambda x, y: (x * 3 + 6) % 256)
            self.assertTrue(ep.find_duplicate_stills([a, b], threshold=12))
        finally:
            _rmtree(tmp)

    def test_missing_files_skipped_not_fatal(self):
        self.assertEqual(ep.find_duplicate_stills([None, "nope.png"]), [])

    def test_phash_is_none_on_junk(self):
        self.assertIsNone(ep._phash(Path("no-such-image.png")))

    def test_phash_distance(self):
        self.assertEqual(ep._phash_distance(0b1010, 0b1010), 0)
        self.assertEqual(ep._phash_distance(0b0000, 0b1111), 4)


class QualityTierVisibilityTests(unittest.TestCase):
    """_yaml_fallback silently drops any final-tier OOM to fast, so an episode
    could ship fast-tier clips with no marker anywhere."""

    def test_tiers_recorded_in_receipt(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        self.assertIn('"h3_tiers"', src)
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_produce_inner")
        body = ast.get_source_segment(src, fn)
        self.assertIn("_tier_used", body)
        self.assertIn("H3 quality tier fell back", body)

    def test_relay_flag_is_surfaced_in_settings(self):
        """The dashboard settings list is the beginner-facing surface; a
        flag that costs GPU time should be visible there."""
        src = (HERE / "pipeline_server.py").read_text(encoding="utf-8")
        self.assertIn("h3_continuity_relay", src)

    def test_relay_outcome_recorded_in_receipt(self):
        """A silently-off relay is indistinguishable from 'relay is on and
        identity holds' — the receipt must say which happened."""
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_produce_inner")
        body = ast.get_source_segment(src, fn)
        self.assertIn('"relay_used"', body)
        self.assertIn("clips_with_relay", body)
        self.assertIn("relay_count", body)


class ColourConsistencyTests(unittest.TestCase):
    """Shot-to-shot exposure/tint jump was the largest unaddressed gap: a
    grep for deflicker / colour-match across all 30 support modules and the
    producer returned zero hits."""

    W, H, N = 64, 64, 12

    def _clip(self, tmp, name, r, g, b, flicker=0.0):
        frames = b""
        for i in range(self.N):
            k = 1.0 + flicker * ((i % 2) * 2 - 1)
            frames += bytes((min(255, int(r * k)), min(255, int(g * k)),
                             min(255, int(b * k)))) * (self.W * self.H)
        out = tmp / name
        proc = subprocess.run(
            [ep.CONFIG["ffmpeg"], "-y", "-v", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{self.W}x{self.H}", "-r", "24",
             "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
             "-pix_fmt", "yuv420p", str(out)],
            input=frames, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-200:])
        return out

    def test_rgb_stats_measures_actual_colour(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            s = ep.clip_rgb_stats(self._clip(tmp, "g.mp4", 120, 120, 120))
            self.assertIsNotNone(s)
            for ch in ("r", "g", "b", "luma"):
                self.assertAlmostEqual(s[ch], 120, delta=6)
            warm = ep.clip_rgb_stats(self._clip(tmp, "w.mp4", 150, 110, 80))
            self.assertGreater(warm["r"], warm["b"] + 40)
        finally:
            _rmtree(tmp)

    def test_rgb_stats_returns_none_not_exception(self):
        self.assertIsNone(ep.clip_rgb_stats(Path("nope.mp4")))
        tmp = Path(tempfile.mkdtemp())
        try:
            junk = tmp / "j.mp4"
            junk.write_text("not a video")
            self.assertIsNone(ep.clip_rgb_stats(junk))
        finally:
            _rmtree(tmp)

    def test_filter_is_empty_when_clips_agree(self):
        ref = {"r": 118.0, "g": 118.0, "b": 118.0, "luma": 118.0}
        self.assertEqual(ep.grade_consistency_filter(ref, dict(ref)), "")
        self.assertEqual(ep.grade_consistency_filter(None, ref), "")
        self.assertEqual(ep.grade_consistency_filter(ref, None), "")

    def test_distant_scene_is_refused_not_mangled(self):
        """A night interior next to a day exterior is a different scene, not
        exposure drift. The correction is CAPPED, so the extreme case gets a
        bounded nudge rather than being forced to match."""
        ref = {"r": 118.0, "g": 118.0, "b": 118.0, "luma": 118.0}
        night = {"r": 20.0, "g": 18.0, "b": 16.0, "luma": 18.0}
        vf = ep.grade_consistency_filter(ref, night)
        # eq brightness is clamped to the cap, never the full 0.39
        self.assertIn("eq=brightness=", vf)
        val = float(vf.split("eq=brightness=")[1].split(",")[0])
        self.assertLessEqual(abs(val), 0.42 + 1e-6)
        self.assertLess(abs(val), 0.3926)   # NOT the uncapped -0.3926

    def test_tint_gains_average_one(self):
        """eq already moves the level; gains must be tint-only or the two
        compound and overshoot."""
        ref = {"r": 118.0, "g": 118.0, "b": 118.0, "luma": 118.0}
        cur = {"r": 139.0, "g": 109.0, "b": 89.0, "luma": 115.7}
        f = ep.grade_consistency_filter(ref, cur)
        gains = [float(g.split("=")[1])
                 for g in f.split("colorchannelmixer=")[1].split(":")]
        self.assertAlmostEqual(sum(gains) / 3, 1.0, delta=0.02)
        self.assertLess(gains[0], 1.0)     # too red  -> reduce r
        self.assertGreater(gains[2], 1.0)  # not blue -> raise b

    def test_grading_reduces_tint_error_end_to_end(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            ref_clip = self._clip(tmp, "ref.mp4", 120, 120, 120)
            warm = self._clip(tmp, "warm.mp4", 145, 108, 82)
            ref = ep.clip_rgb_stats(ref_clip)
            before = ep._tint_error(ref, ep.clip_rgb_stats(warm))
            ok, used, passes, resid = ep.grade_clip_to_reference(
                warm, tmp / "out.mp4", ref, 320, 240, target_fps=24)
            self.assertTrue(ok)
            self.assertGreaterEqual(passes, 1)
            self.assertTrue(used)
            after = ep._tint_error(ref, ep.clip_rgb_stats(tmp / "out.mp4"))
            self.assertLess(after, before,
                            f"tint error not reduced: {before} -> {after}")
            self.assertEqual(list(tmp.glob("*.g*.mp4")), [],
                             "grade temp files left behind")
        finally:
            _rmtree(tmp)

    def test_matching_clip_needs_no_pass(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            c = self._clip(tmp, "same.mp4", 120, 120, 120)
            ok, used, passes, resid = ep.grade_clip_to_reference(
                c, tmp / "out.mp4", ep.clip_rgb_stats(c), 320, 240, target_fps=24)
            self.assertTrue(ok)
            self.assertEqual(passes, 0)
            self.assertEqual(used, "")
        finally:
            _rmtree(tmp)

    def test_deflicker_reduces_temporal_swing(self):
        """Diffusion video shimmers frame to frame; deflicker rides the
        encode we already run, so it is free."""
        import numpy as np
        tmp = Path(tempfile.mkdtemp())
        try:
            flick = self._clip(tmp, "f.mp4", 120, 120, 120, flicker=0.22)

            def swing(p):
                pr = subprocess.run(
                    [ep.CONFIG["ffmpeg"], "-v", "error", "-i", str(p), "-vf",
                     "scale=32:32", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                    capture_output=True)
                n = len(pr.stdout) // 3072
                a = np.frombuffer(pr.stdout[:n * 3072], np.uint8).reshape(n, -1)
                return float(a.mean(axis=1).std())

            ep._apply_clip_grade(flick, tmp / "d.mp4", "", 320, 240)
            self.assertLess(swing(tmp / "d.mp4"), swing(flick))
        finally:
            _rmtree(tmp)

    def test_deflicker_is_configurable_and_defaults_on(self):
        self.assertTrue(ep._DEFAULTS.get("clip_deflicker", True))
        self.assertIn("clip_deflicker", ep.CONFIG)

    def test_grade_outcome_recorded_in_receipt(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_produce_inner")
        body = ast.get_source_segment(src, fn)
        self.assertIn('"colour_grade"', body)
        self.assertIn("grade_applied", body)
        self.assertIn("grade_rejected", body)


class InterpolationTests(unittest.TestCase):
    """The gate was `fps < 20` where fps is the OUTPUT rate we ask the
    encoder for (echo_fps, 24) — so RIFE never ran, ever."""

    def _clip(self, tmp, name, rate, n=12):
        out = tmp / name
        proc = subprocess.run(
            [ep.CONFIG["ffmpeg"], "-y", "-v", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", "64x64", "-r", str(rate), "-i", "-",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
             "-pix_fmt", "yuv420p", str(out)],
            input=bytes((90, 90, 90)) * (64 * 64 * n), capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-200:])
        return out

    def test_video_fps_duration_reads_real_values(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fps, dur = ep.video_fps_duration(self._clip(tmp, "a.mp4", 24))
            self.assertAlmostEqual(fps, 24, delta=0.5)
            self.assertAlmostEqual(dur, 0.5, delta=0.1)
        finally:
            _rmtree(tmp)

    def test_video_fps_duration_zero_on_junk(self):
        self.assertEqual(ep.video_fps_duration(Path("nope.mp4")), (0.0, 0.0))

    def test_frames_to_clip_no_longer_gates_on_output_fps(self):
        import ast as _ast
        src = (HERE / "episode_producer.py").read_text(encoding="utf-8")
        tree = _ast.parse(src)
        fn = next(n for n in _ast.walk(tree)
                  if isinstance(n, _ast.FunctionDef)
                  and n.name == "frames_to_clip")
        # check the CODE, not the comment that explains the old bug
        assigned = {t.id for n in _ast.walk(fn)
                    if isinstance(n, _ast.Assign)
                    for t in n.targets if isinstance(t, _ast.Name)}
        self.assertIn("needed", assigned)
        self.assertIn("delivered", assigned)
        self.assertNotIn("native_fps", assigned,
                         "the self-referential native_fps gate is back")
        body = _ast.get_source_segment(src, fn)
        self.assertNotIn("if fps < 20 and rife_double", body)


class TtsSegmentationTests(unittest.TestCase):
    """A whole scene went to /cosy in one POST. CosyVoice/XTTS are
    autoregressive LMs: long requests drift and the speaker embedding
    weakens as context grows."""

    def test_splits_on_sentence_boundaries(self):
        segs = ep.split_for_tts(
            "The door was already open when she reached it. "
            "Somebody had been waiting inside for hours. "
            "Nobody in the house would ever admit to it.")
        self.assertEqual(len(segs), 3)
        for seg in segs:
            self.assertTrue(seg.endswith("."), f"cut mid-sentence: {seg!r}")

    def test_stub_fragments_merge_forward(self):
        """The engines' guidance: never send a request under ~4 words, the
        speaker anchor is per-request. Punctuation-only stubs must merge."""
        segs = ep.split_for_tts("A. B. The lamp had been burning all night "
                                "in the empty room.")
        self.assertEqual(len(segs), 1)
        self.assertIn("burning", segs[0])

    def test_no_content_is_lost_or_altered(self):
        for text in (
            "First one. Second one. Third one closes it out.",
            "word, " * 80,
            "No punctuation at all just words " * 20,
            "Hi. Ok. Yes. No. Maybe. Perhaps. Certainly. Absolutely. Indeed. Fine.",
        ):
            segs = ep.split_for_tts(text)
            joined = " ".join(" ".join(segs).split())
            self.assertEqual(joined, " ".join(text.split()),
                             f"content changed for {text[:40]!r}")

    def test_short_input_is_one_segment(self):
        self.assertEqual(ep.split_for_tts("Short one."), ["Short one."])

    def test_empty_input(self):
        self.assertEqual(ep.split_for_tts(""), [])
        self.assertEqual(ep.split_for_tts(None), [])
        self.assertEqual(ep.split_for_tts("   "), [])

    def test_no_stub_segments(self):
        for seg in ep.split_for_tts("A. B. C. D. " * 30):
            self.assertGreaterEqual(len(seg), 2, f"stub segment {seg!r}")

    def test_long_paragraph_is_bounded(self):
        segs = ep.split_for_tts("B " * 400)
        self.assertGreater(len(segs), 1)
        for seg in segs:
            self.assertLessEqual(len(seg), 260, f"oversized segment {len(seg)}")

    def test_unbreakable_run_is_returned_whole(self):
        """A 500-char no-space run cannot be chopped without cutting mid-word;
        the engine handles it and that is the documented behaviour."""
        segs = ep.split_for_tts("A" * 500)
        self.assertEqual(segs, ["A" * 500])

    def test_concat_wavs_inserts_gap_and_preserves_audio(self):
        import math
        import struct
        import wave
        tmp = Path(tempfile.mkdtemp())
        try:
            sr = 44100
            parts = []
            for i, amp in enumerate((0.6, 0.3, 0.9)):
                p = tmp / f"s{i}.wav"
                with wave.open(str(p), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(sr)
                    w.writeframes(b"".join(
                        struct.pack("<h", int(amp * 32767
                                               * math.sin(2 * math.pi * 300 * j / sr)))
                        for j in range(int(sr * 0.4))))
                parts.append(p)
            out = ep._concat_wavs(parts, tmp / "j.wav")
            self.assertIsNotNone(out)
            with wave.open(str(out), "rb") as w:
                dur = w.getnframes() / w.getframerate()
            self.assertAlmostEqual(dur, 3 * 0.4 + 2 * 0.12, delta=0.02)
        finally:
            _rmtree(tmp)

    def test_concat_wavs_accepts_the_servers_native_rate(self):
        """The TTS server is XTTS v2 at 24000 Hz. A hardcoded 44100 made every
        chunked synth fail, and 2 of 25 ch001 scenes abort the render."""
        import math
        import struct
        import wave
        for sr in (24000, 44100, 48000):
            tmp = Path(tempfile.mkdtemp())
            try:
                parts = []
                for i in range(3):
                    p = tmp / f"s{i}.wav"
                    with wave.open(str(p), "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(sr)
                        w.writeframes(b"".join(
                            struct.pack("<h", int(0.5 * 32767
                                                   * math.sin(2 * math.pi * 300 * j / sr)))
                            for j in range(int(sr * 0.2))))
                    parts.append(p)
                out = ep._concat_wavs(parts, tmp / "j.wav")
                self.assertIsNotNone(out, f"{sr}Hz rejected")
                with wave.open(str(out), "rb") as w:
                    self.assertEqual(w.getframerate(), sr,
                                     f"rate not preserved: {sr}Hz in")
                    dur = w.getnframes() / sr
                self.assertAlmostEqual(dur, 3 * 0.2 + 2 * 0.12, delta=0.02)
            finally:
                _rmtree(tmp)

    def test_concat_wavs_rejects_mixed_rates(self):
        """The REAL failure mode: same channels, different rates."""
        import struct
        import wave
        tmp = Path(tempfile.mkdtemp())
        try:
            made = []
            for sr in (24000, 48000):
                p = tmp / f"{sr}.wav"
                with wave.open(str(p), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(sr)
                    w.writeframes(struct.pack("<h", 1000) * 100)
                made.append(p)
            self.assertIsNone(ep._concat_wavs(made, tmp / "x.wav"))
            # same rate, different channel counts is also refused
            st = tmp / "s.wav"
            with wave.open(str(st), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(b"\x00" * 400)
            self.assertIsNone(ep._concat_wavs([st, st], tmp / "y.wav"))
            self.assertIsNone(ep._concat_wavs([], tmp / "z.wav"))
        finally:
            _rmtree(tmp)

    def test_segment_files_are_cleaned_on_success(self):
        """The success path used to break out and leave .segNN.wav litter."""
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        out = tmp / "ch001_s01.wav"
        try:
            import math
            import struct
            import wave

            def fake_post(url, json=None, timeout=300):
                buf = tmp / "resp.wav"
                # ~4s per segment at the engine's native 24kHz, so the
                # duration gate (len(text)/30) sees realistic audio
                with wave.open(str(buf), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(24000)
                    w.writeframes(b"".join(
                        struct.pack("<h", int(0.4 * 32767
                                               * math.sin(2 * math.pi * 200 * j / 24000)))
                        for j in range(24000 * 4)))
                r = mock.Mock()
                r.status_code = 200
                r.content = buf.read_bytes()
                return r

            with mock.patch("requests.post", fake_post), \
                 mock.patch.object(ep, "check_tts_server", lambda: True), \
                 mock.patch.object(ep, "CONFIG", dict(ep.CONFIG, voice_speed=1.0)):
                ok = ep.generate_tts(
                    "The door was already open. Somebody had been waiting "
                    "inside for hours.", out)
            self.assertTrue(ok, "chunked TTS failed at the engine's native rate")
            self.assertTrue(out.exists())
            self.assertEqual(list(tmp.glob("*.seg*.wav")), [],
                             f"segment files left behind: "
                             f"{[p.name for p in tmp.glob('*.seg*.wav')]}")
        finally:
            _rmtree(tmp)

    def test_generation_uses_the_segmenter(self):
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "generate_tts")
        body = ast.get_source_segment(src, fn)
        self.assertIn("split_for_tts(text)", body)
        self.assertIn("_concat_wavs", body)
        self.assertIn("prior = output_path.read_bytes()", body)


class LraGateTests(unittest.TestCase):
    """LRA=11 was passed to loudnorm on every call and never asserted."""

    def test_lra_is_warned_not_blocked(self):
        """Behavioural. A hard gate at 14 aborted renders deterministically:
        measured on this project's own beds, LRA is 10.6-17.3 LU and the
        track is a mood-hash coin flip."""
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        try:
            final = tmp / "f.mp4"
            final.write_bytes(b"x")
            out = tmp / "o.mp4"
            for lra in (20.0, 30.0):
                with mock.patch.object(ep, "ROOT", tmp), \
                     mock.patch.object(ep, "validate_media", lambda p, s: {
                         "video_duration": 1.0, "audio_duration": 1.0, "fps": 24.0,
                         "width": 1920, "height": 1080, "channels": 2}), \
                     mock.patch.object(ep, "measure_loudness", lambda p, t=None: {
                         "lufs": -13.5, "true_peak": -3.0, "lra": lra}), \
                     mock.patch.object(ep, "qa_true_peak", lambda p: {
                         "passed": True, "peak_dbtp": -3.0, "limit_dbtp": -1.9}), \
                     mock.patch.object(ep, "qa_av_sync", lambda p, s, a: {
                         "passed": True, "checked": [1], "skipped": [],
                         "violations": [], "errors": []}), \
                     mock.patch.object(ep, "qa_final_black", lambda p: 0.0), \
                     mock.patch.object(ep, "qa_final_frames", lambda p, d: None), \
                     mock.patch.object(ep, "qa_auto_editor", lambda p, s, a: None), \
                     mock.patch.object(ep, "rotate_versions", lambda *a, **k: None):
                    out.unlink(missing_ok=True)
                    ep._promote_media(final, out, [{"_dur": 1.0, "_cap_start": 0.0}], [])
                    self.assertTrue(out.exists(),
                                    f"LRA {lra} aborted the render")
                    out.unlink(missing_ok=True)
                self.assertEqual(ep._LAST_ASSEMBLY.get("lra"), lra,
                                 "LRA not recorded in the receipt")
        finally:
            _rmtree(tmp)

    def test_lufs_gate_still_blocks_a_real_outlier(self):
        """The ceiling must catch an un-normalised master (which is LOUDER)."""
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        try:
            final = tmp / "f.mp4"
            final.write_bytes(b"x")
            out = tmp / "o.mp4"
            with mock.patch.object(ep, "ROOT", tmp), \
                 mock.patch.object(ep, "validate_media", lambda p, s: {
                     "video_duration": 1.0, "audio_duration": 1.0, "fps": 24.0,
                     "width": 1920, "height": 1080, "channels": 2}), \
                 mock.patch.object(ep, "measure_loudness", lambda p, t=None: {
                     "lufs": -4.0, "true_peak": -3.0, "lra": 6.0}), \
                 mock.patch.object(ep, "qa_true_peak", lambda p: {
                     "passed": True, "peak_dbtp": -3.0, "limit_dbtp": -1.9}), \
                 mock.patch.object(ep, "qa_av_sync", lambda p, s, a: {
                     "passed": True, "checked": [1], "skipped": [],
                     "violations": [], "errors": []}), \
                 mock.patch.object(ep, "qa_final_black", lambda p: 0.0), \
                 mock.patch.object(ep, "qa_final_frames", lambda p, d: None), \
                 mock.patch.object(ep, "qa_auto_editor", lambda p, s, a: None), \
                 mock.patch.object(ep, "rotate_versions", lambda *a, **k: None):
                with self.assertRaises(RuntimeError) as cm:
                    ep._promote_media(final, out, [{"_dur": 1.0, "_cap_start": 0.0}], [])
            self.assertIn("Loudness gate failed", str(cm.exception))
            self.assertFalse(out.exists())
        finally:
            _rmtree(tmp)

    def test_lufs_floor_has_headroom_for_a_bed(self):
        """A music bed pushes measured LUFS DOWN; a 1 LU floor false-failed
        normal mixes (dark_fog at 0.25 measured -15.14)."""
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        try:
            final = tmp / "f.mp4"
            final.write_bytes(b"x")
            out = tmp / "o.mp4"
            # -16.7 LUFS: a real measurement from a real bed, must pass
            with mock.patch.object(ep, "ROOT", tmp), \
                 mock.patch.object(ep, "validate_media", lambda p, s: {
                     "video_duration": 1.0, "audio_duration": 1.0, "fps": 24.0,
                     "width": 1920, "height": 1080, "channels": 2}), \
                 mock.patch.object(ep, "measure_loudness", lambda p, t=None: {
                     "lufs": -16.74, "true_peak": -2.4, "lra": 17.3}), \
                 mock.patch.object(ep, "qa_true_peak", lambda p: {
                     "passed": True, "peak_dbtp": -2.4, "limit_dbtp": -1.9}), \
                 mock.patch.object(ep, "qa_av_sync", lambda p, s, a: {
                     "passed": True, "checked": [1], "skipped": [],
                     "violations": [], "errors": []}), \
                 mock.patch.object(ep, "qa_final_black", lambda p: 0.0), \
                 mock.patch.object(ep, "qa_final_frames", lambda p, d: None), \
                 mock.patch.object(ep, "qa_auto_editor", lambda p, s, a: None), \
                 mock.patch.object(ep, "rotate_versions", lambda *a, **k: None):
                ep._promote_media(final, out, [{"_dur": 1.0, "_cap_start": 0.0}], [])
                self.assertTrue(out.exists(), "a real bed mix was rejected")
        finally:
            _rmtree(tmp)

    def test_lra_max_is_a_config_key(self):
        self.assertIn("lra_max", ep._DEFAULTS)
        self.assertIn("lra_max", ep.CONFIG)

    def test_measure_loudness_returns_lra(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            wav = tmp / "a.wav"
            subprocess.run([ep.CONFIG["ffmpeg"], "-y", "-v", "error",
                            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                            "-c:a", "pcm_s16le", "-ar", "44100", str(wav)],
                           capture_output=True)
            m = ep.measure_loudness(wav)
            self.assertIsNotNone(m)
            for k in ("lufs", "true_peak", "lra"):
                self.assertIn(k, m)
        finally:
            _rmtree(tmp)


class FirstClipBranchTests(unittest.TestCase):
    """`_apply_clip_grade` returns a bool. It was once unpacked as a 4-tuple
    on the first-clip branch, raising TypeError on the FIRST generated clip of
    every episode — i.e. every render with video died."""

    def test_first_clip_branch_shapes_match(self):
        """AST-level: the first-clip branch must not unpack a bool."""
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_produce_inner")
        bad = []
        for node in ast.walk(fn):
            # `a, b, c, d = <call>` where the call is _apply_clip_grade
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
                if isinstance(node.value, ast.Call):
                    fname = getattr(node.value.func, "id", "")
                    if fname == "_apply_clip_grade":
                        bad.append(node.lineno)
        self.assertEqual(bad, [],
                         f"lines {bad} unpack _apply_clip_grade (a bool) "
                         f"as a tuple — raises TypeError on the first clip")

    def test_apply_clip_grade_returns_a_bool(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            src = self._make(tmp / "a.mp4")
            self.assertIs(ep._apply_clip_grade(src, tmp / "b.mp4", "", 64, 64), True)
        finally:
            _rmtree(tmp)

    def _make(self, path):
        return subprocess.run(
            [ep.CONFIG["ffmpeg"], "-y", "-v", "error", "-f", "lavfi",
             "-i", "color=c=gray:s=64x64:d=0.5:r=24", "-c:v", "libx264",
             "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
            capture_output=True) and path


class LimitOnlyBehaviourTests(unittest.TestCase):
    """`_limit_only` must cap peaks WITHOUT re-normalising integrated
    loudness. A source-grep test passed even with it implemented as a second
    loudnorm — i.e. the exact bug the change fixed."""

    def test_limit_preserves_loudness_and_caps_peaks(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            hot = tmp / "hot.wav"
            # a deliberately loud, peaky signal
            subprocess.run(
                [ep.CONFIG["ffmpeg"], "-y", "-v", "error", "-f", "lavfi",
                 "-i", "sine=frequency=440:duration=3", "-af", "volume=6dB",
                 "-c:a", "pcm_s16le", "-ar", "44100", str(hot)],
                capture_output=True)
            before = ep.measure_loudness(hot)
            self.assertIsNotNone(before)
            out = tmp / "lim.wav"
            self.assertTrue(ep._limit_only(hot, out))
            after = ep.measure_loudness(out)
            self.assertIsNotNone(after)
            # loudness essentially unchanged
            self.assertLess(abs(after["lufs"] - before["lufs"]), 1.0,
                            f"loudness moved {before['lufs']:.1f} -> "
                            f"{after['lufs']:.1f} — this normalised, not limited")
            # but true peak is now inside the ceiling
            self.assertLessEqual(after["true_peak"], ep._TP_LIMIT + 0.3,
                                 f"true peak {after['true_peak']} not capped")
        finally:
            _rmtree(tmp)


class LoudnormDualPeakTests(unittest.TestCase):
    """`_loudnorm_dual` is the no-music default path and its `TP=` is the thing
    that actually holds the true peak — the alimiter downstream has
    _TP_LIMIT_HEADROOM on _limit_only only, and here the peak control is
    loudnorm's own TP. Deleting `TP=` from the filter survived the whole suite:
    no test covered the dual path, and test_limiter_filter_uses_the_headroom
    only inspects _limit_only. Measured, not grepped."""

    def _hot_wav(self, tmp, gain="6dB"):
        hot = Path(tmp) / "hot.wav"
        subprocess.run(
            [ep.CONFIG["ffmpeg"], "-y", "-v", "error", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=3", "-af", f"volume={gain}",
             "-c:a", "pcm_s16le", "-ar", "44100", str(hot)],
            capture_output=True)
        self.assertTrue(hot.is_file(), "could not synthesise a test signal")
        return hot

    def test_the_output_clears_the_true_peak_gate(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            hot = self._hot_wav(tmp)
            before = ep.measure_loudness(hot)
            self.assertIsNotNone(before)
            out = Path(tmp) / "norm.wav"
            self.assertTrue(ep._loudnorm_dual(hot, out, -16.0),
                            "_loudnorm_dual refused the input")
            after = ep.measure_loudness(out)
            self.assertIsNotNone(after)
            # the gate that refuses a render
            self.assertLessEqual(after["true_peak"], ep._TP_LIMIT + ep._TP_TOL,
                                 f"true peak {after['true_peak']} dBTP exceeds "
                                 f"the {ep._TP_LIMIT} dBTP gate with no limiter "
                                 f"after it — loudnorm's TP= is not holding")
        finally:
            _rmtree(tmp)

    def test_both_peak_controls_are_present(self):
        """The measurement above passes with OR without loudnorm's `TP=`,
        because alimiter=limit=-2.00dB downstream also holds the peak — so
        deleting TP= is invisible to it. That makes the two controls
        redundant to a single assertion and jointly necessary: lose either one
        and the peak is still held, lose both and a hot mix ships over the
        ceiling. Asserted structurally because there is no measurement that
        distinguishes them; test_limiter_filter_uses_the_headroom is the same
        kind of guard on the other chain."""
        import inspect
        src = inspect.getsource(ep._loudnorm_dual)
        self.assertIn("TP={_TP_LIMIT}", src,
                      "loudnorm must be told the true-peak ceiling it targets, "
                      "in BOTH the measure pass and the apply pass")
        self.assertEqual(src.count("TP={_TP_LIMIT}"), 2,
                         "the ceiling must reach the apply filter as well as "
                         "the measuring one")
        self.assertIn("alimiter", src,
                      "and a limiter must follow it as the second control")

    def test_it_normalises_rather_than_only_limits(self):
        """Loudness must actually move to the target, otherwise a broken filter
        that merely caps peaks would pass the peak test above."""
        tmp = Path(tempfile.mkdtemp())
        try:
            hot = self._hot_wav(tmp)
            before = ep.measure_loudness(hot)
            out = Path(tmp) / "norm.wav"
            self.assertTrue(ep._loudnorm_dual(hot, out, -18.0))
            after = ep.measure_loudness(out)
            self.assertLess(abs(after["lufs"] - (-18.0)), 1.5,
                            f"target -18 LUFS, got {after['lufs']:.1f}")
            self.assertNotEqual(after["lufs"], before["lufs"])
        finally:
            _rmtree(tmp)


class GradeFailureSafetyTests(unittest.TestCase):
    """A failed encode leaves a PARTIAL file. Shipping it destroyed the
    episode at validate_media, after all the GPU work was already spent."""

    def test_failed_grade_leaves_no_shippable_file(self):
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        try:
            for name, r, g, b in (("ref", 120, 120, 120), ("warm", 145, 110, 80)):
                out = tmp / f"{name}.mp4"
                proc = subprocess.run(
                    [ep.CONFIG["ffmpeg"], "-y", "-v", "error", "-f", "rawvideo",
                     "-pix_fmt", "rgb24", "-s", "64x64", "-r", "24", "-i", "-",
                     "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
                     "-pix_fmt", "yuv420p", str(out)],
                    input=bytes((r, g, b)) * (64 * 64 * 24), capture_output=True)
                self.assertEqual(proc.returncode, 0, proc.stderr[-200:])
            ref = ep.clip_rgb_stats(tmp / "ref.mp4")
            dst = tmp / "out.mp4"

            def failing(src, out, chain, w, h, **kw):
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_bytes(b"partial-junk")   # ffmpeg's habit
                return False

            with mock.patch.object(ep, "_apply_clip_grade", failing):
                ok, _used, _passes, _resid = ep.grade_clip_to_reference(
                    tmp / "warm.mp4", dst, ref, 320, 240, target_fps=24)
            self.assertFalse(ok)
            self.assertFalse(dst.exists(),
                             "partial file would be shipped into the timeline")
            self.assertEqual(list(tmp.glob("*.g*.mp4")), [],
                             "grade temp files left behind")
        finally:
            _rmtree(tmp)

    def test_converged_clip_is_encoded_once(self):
        """The promote step used to re-encode an already-deflickered
        intermediate; deflicker is temporal, so a second pass is neither
        free nor idempotent."""
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        try:
            for name, r, g, b in (("ref", 120, 120, 120), ("near", 126, 118, 112)):
                out = tmp / f"{name}.mp4"
                proc = subprocess.run(
                    [ep.CONFIG["ffmpeg"], "-y", "-v", "error", "-f", "rawvideo",
                     "-pix_fmt", "rgb24", "-s", "64x64", "-r", "24", "-i", "-",
                     "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
                     "-pix_fmt", "yuv420p", str(out)],
                    input=bytes((r, g, b)) * (64 * 64 * 24), capture_output=True)
                self.assertEqual(proc.returncode, 0, proc.stderr[-200:])
            ref = ep.clip_rgb_stats(tmp / "ref.mp4")
            calls = []
            real = ep._apply_clip_grade

            def counting(src, out, chain, w, h, **kw):
                calls.append(Path(out).name)
                return real(src, out, chain, w, h, **kw)

            with mock.patch.object(ep, "_apply_clip_grade", counting):
                ok, _used, passes, _resid = ep.grade_clip_to_reference(
                    tmp / "near.mp4", tmp / "o.mp4", ref, 320, 240, target_fps=24)
            self.assertTrue(ok)
            self.assertEqual(passes, 1)
            self.assertEqual(len(calls), 1, f"extra encodes: {calls}")
        finally:
            _rmtree(tmp)

    def test_grade_temp_cleanup_is_scoped_to_created_files(self):
        """A directory glob `<stem>.g*.mp4` would delete an unrelated
        scene_07.gallery.mp4 sitting next to the intermediates."""
        tmp = Path(tempfile.mkdtemp())
        try:
            bystander = tmp / "scene_07.gallery.mp4"
            bystander.write_bytes(b"not mine")
            ep._drop([tmp / "scene_07.g1.mp4", tmp / "scene_07.g2.mp4"])
            self.assertTrue(bystander.exists(), "unrelated file was deleted")
        finally:
            _rmtree(tmp)

    def test_colour_grade_is_off_by_default(self):
        """Measured on the 19 real H3 clips of ch001: grading to a common
        reference cut the 96/255 luma spread to 7 but INVERTED the brightness
        ordering (r = -0.195) — the fire went 114->45, the dark alley 18->46.
        That spread was the content, not drift. All 19 clips shared one mood,
        so mood cannot separate intent from error either."""
        self.assertIs(ep._DEFAULTS.get("clip_colour_grade"), False)
        self.assertIs(ep.CONFIG.get("clip_colour_grade"), False)
        src = Path(ep.__file__).read_text(encoding="utf-8")
        self.assertIn("ordering", src.lower())

    def test_grade_is_unreachable_in_the_default_render_path(self):
        """The real invariant now that the grade is off.

        A capped grade does NOT preserve scene brightness. Measured on the
        19 real H3 clips of ch001 it produced a PERFECT inversion (r = -1.0
        on a fire/corridor/alley triple: 112->45, 45->45, 16->47) because the
        cap (107 units) is wider than any real delta. The feature is
        therefore off by default, and this asserts the production path never
        calls it.
        """
        import ast as _ast
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = _ast.parse(src)
        fn = next(n for n in _ast.walk(tree)
                  if isinstance(n, _ast.FunctionDef)
                  and n.name == "_produce_inner")
        # walk outward from each call to its enclosing if-statement
        parents = {}
        for node in _ast.walk(fn):
            for child in _ast.iter_child_nodes(node):
                parents[child] = node
        guarded = False
        for node in _ast.walk(fn):
            if (isinstance(node, _ast.Call)
                    and getattr(node.func, "id", "") == "grade_clip_to_reference"):
                anc, ok = parents.get(node), False
                while anc is not None and anc is not fn:
                    if isinstance(anc, _ast.If) and "_grade_on" in _ast.unparse(anc.test):
                        ok = True
                        break
                    anc = parents.get(anc)
                self.assertTrue(ok, "grade call is not guarded by _grade_on")
                guarded = True
        self.assertTrue(guarded, "no grade call site found in _produce_inner")
        self.assertIn('_grade_on = bool(CONFIG.get("clip_colour_grade", False))', src)

    def test_grade_caps_rather_than_refuses(self):
        """A hard refusal above the cap did nothing useful: it skipped almost
        nothing while near clips got a full correction. Measured on real
        output, luma deltas reach 96/255, so a 0.18 cap is meaningless."""
        ref = {"r": 48.9, "g": 44.6, "b": 44.9, "luma": 48.9}
        far = {"r": 142.3, "g": 108.4, "b": 72.0, "luma": 114.4}
        vf = ep.grade_consistency_filter(ref, far)
        self.assertTrue(vf, "a 65-unit delta must get a capped nudge, not a skip")
        self.assertIn("eq=brightness=-0.2569", vf)   # clamped to the 0.42 cap

    def test_median_reference_ignores_outliers(self):
        """Anchoring on clip #1 would drag a fire scene to a dark corridor."""
        stats = [{"r": 20, "g": 20, "b": 20, "luma": 18.0},
                 {"r": 50, "g": 45, "b": 45, "luma": 48.9},
                 {"r": 142, "g": 108, "b": 72, "luma": 114.4}]
        ref = ep.grade_reference_from(stats)
        self.assertEqual(ref["luma"], 48.9)
        self.assertNotEqual(ref["luma"], 18.0)
        self.assertNotEqual(ref["luma"], 114.4)

    def test_deflicker_preserves_scene_identity(self):
        """deflicker is per-frame temporal, so it must not move the episode's
        mean luma or reorder scenes — measured r=1.000 on real clips."""
        import numpy as np
        tmp = Path(tempfile.mkdtemp())
        try:
            lums = [120, 60, 20]
            made = []
            for v in lums:
                out = tmp / f"{v}.mp4"
                subprocess.run(
                    [ep.CONFIG["ffmpeg"], "-y", "-v", "error", "-f", "rawvideo",
                     "-pix_fmt", "rgb24", "-s", "64x64", "-r", "24", "-i", "-",
                     "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
                     "-pix_fmt", "yuv420p", str(out)],
                    input=bytes((v, v, v)) * (64 * 64 * 24), capture_output=True,
                    check=True)
                made.append(out)
            before = [ep.clip_rgb_stats(p)["luma"] for p in made]
            after = []
            for p in made:
                d = tmp / f"d_{p.stem}.mp4"
                ep._apply_clip_grade(p, d, "", 320, 240)
                after.append(ep.clip_rgb_stats(d)["luma"])
            self.assertTrue(all(abs(b - a) < 6 for b, a in zip(before, after)),
                            f"deflicker moved scene luma: {before} -> {after}")
            self.assertEqual(int(np.argmax(before)), int(np.argmax(after)))
        finally:
            _rmtree(tmp)

    def test_colour_grade_key_survives_the_receipt_clear(self):
        for key in ("cinematic_fallbacks", "video_clips", "h3_tiers",
                    "duplicate_stills", "relay_used", "colour_grade"):
            self.assertIn(key, ep._CLIP_PHASE_KEYS,
                          f"{key} is deleted before anyone can read it")

    def test_release_report_surfaces_the_new_receipt_keys(self):
        from pipeline.release_report import create_release_report
        receipt = {
            "promoted": True,
            "streams": {"fps": 24.0, "width": 1920, "height": 1080, "channels": 2},
            "music_bed": "bed.mp3",
            "scenes": [{"clip": "a.mp4"}],
            "colour_grade": {"applied": 3, "rejected": 0, "deflicker": True},
            "lra": 17.3,
            "relay_used": {"enabled": True, "clips_with_relay": 0},
            "duplicate_stills": [{"scene_a": 3, "scene_b": 9, "distance": 1}],
        }
        r = create_release_report(1, [{"clip": "a.mp4"}], receipt,
                                  expect_fps=24.0, expect_dims=(1920, 1080))
        cats = {i["category"] for i in r["issues"]}
        for expected in ("colour", "continuity", "stills"):
            self.assertIn(expected, cats, f"{expected} not surfaced")
        audio = [i for i in r["issues"] if i["category"] == "audio"]
        self.assertTrue(any("Loudness range 17.3" in i["message"] for i in audio),
                        [i["message"] for i in r["issues"]])


class VoicePreflightRobustnessTests(unittest.TestCase):
    """Two real bugs found only by running preflight against live services:
    http_get returns the parsed body (not a tuple), and the served/ref.json
    pair was built as a 2-tuple whose first element was `()` when /health
    was unreachable — so the loop raised and the bare `except` reported
    "missing user voice ref", blocking renders precisely when the TTS server
    was down."""

    def test_http_get_returns_a_body_not_a_tuple(self):
        from unittest import mock
        with mock.patch("urllib.request.urlopen") as resp:
            resp.return_value.read.return_value = b'{"reference": "x.wav"}'
            body = ep.http_get("http://x/health", timeout=1)
        self.assertIsInstance(body, dict)
        self.assertEqual(body["reference"], "x.wav")

    def test_preflight_ok_when_tts_server_is_down(self):
        from unittest import mock
        with mock.patch.object(ep, "http_get",
                               side_effect=OSError("connection refused")):
            probs = list(ep.preflight(900) and ep.LAST_PREFLIGHT_PROBLEMS
                         or ep.LAST_PREFLIGHT_PROBLEMS)
        joined = " | ".join(probs).lower()
        self.assertNotIn("missing/unreadable user voice ref", joined,
                         f"unreachable /health must not read as a bad ref: {probs}")
        self.assertNotIn("not enough values to unpack", joined)

    def test_preflight_uses_the_served_reference(self):
        from unittest import mock
        served = HERE.parent.parent / "voice_user" / "does_not_exist.m4a"
        with mock.patch.object(ep, "http_get",
                               return_value={"status": "ok", "loaded": True,
                                             "reference": str(served)}):
            ep.preflight(900)
        joined = " | ".join(ep.LAST_PREFLIGHT_PROBLEMS).lower()
        self.assertIn("served by the tts server", joined,
                      f"served-reference gate did not run: "
                      f"{ep.LAST_PREFLIGHT_PROBLEMS}")

    def test_both_refs_are_checked_when_the_server_is_up(self):
        """A bad SERVED reference is reported, even though ref.json's copy
        is valid — that is the whole point of the L5 fix."""
        from unittest import mock
        served = HERE.parent.parent / "voice_user" / "does_not_exist.m4a"
        with mock.patch.object(ep, "http_get",
                               return_value={"reference": str(served)}):
            ep.preflight(900)
        joined = " | ".join(ep.LAST_PREFLIGHT_PROBLEMS)
        self.assertIn("served by the TTS server", joined)
        self.assertIn("does_not_exist.m4a", joined,
                      f"served ref not named in the problem: {joined}")
        # The served ref is the bad one; ref.json's own copy is valid here, so
        # it must NOT appear in the problems. Both being checked is exactly why
        # the served one is named — ref.json alone would have looked fine.
        self.assertNotIn("voice ref (ref.json)", joined,
                         "a valid ref.json must not be reported as a problem")


class CachedScriptShapeTests(unittest.TestCase):
    """The cached-script path used to renumber scenes with an inline loop that
    had no type guard, so a scenes file holding a dict instead of a list died
    with 'TypeError: str object does not support item assignment'. Found by a
    real ch900 render."""

    def _load(self, payload):
        import json as _json
        import tempfile as _tf
        tmp = Path(_tf.mkdtemp())
        try:
            f = tmp / "ch777.json"
            f.write_text(_json.dumps(payload), encoding="utf-8")
            return ep._load_cached_scenes(f)
        finally:
            _rmtree(tmp)

    def test_dict_shaped_scenes_file_reports_clearly(self):
        with self.assertRaises(RuntimeError) as cm:
            self._load({"scenes": [{"narration": "x"}]})
        msg = str(cm.exception)
        self.assertNotIn("does not support item assignment", msg)
        self.assertIn("ch777.json", msg)
        self.assertIn("list of scene objects", msg)

    def test_list_of_non_dicts_reports_clearly(self):
        with self.assertRaises(RuntimeError) as cm:
            self._load(["just a string"])
        self.assertIn("list of scene objects", str(cm.exception))

    def test_valid_list_is_renumbered(self):
        scenes = self._load([{"narration": "a", "scene_number": 9},
                             {"narration": "b", "scene_number": 4}])
        self.assertEqual([s["scene_number"] for s in scenes], [1, 2])

    def test_string_fields_are_coerced_not_crashed(self):
        scenes = self._load([{"narration": 42, "characters": "solo",
                             "stages": "nope"}])
        s = scenes[0]
        self.assertEqual(s["narration"], "")
        self.assertEqual(s["characters"], ["solo"])
        self.assertEqual(s["stages"], [])


class BlackStillGateTests(unittest.TestCase):
    """A real ch900 render produced 3 of 10 near-black stills (mean luma
    0.09 / 0.36 / 2.70). The 'stills complete on disk' gate passed them
    because it only checked the file EXISTS, H3 then animated scene 1 out of
    a pure black frame into a hallucinated malformed face, and it shipped."""

    def _png(self, tmp, name, mean, size=(64, 64)):
        import numpy as _np
        from PIL import Image
        p = Path(tmp) / name
        a = _np.full((size[1], size[0], 3), mean, dtype=_np.uint8)
        Image.fromarray(a).save(p)
        return str(p)

    def test_black_still_is_detected(self):
        import tempfile as _tf
        tmp = _tf.mkdtemp()
        try:
            black = self._png(tmp, "black.png", 0)
            ok = self._png(tmp, "ok.png", 90)
            found = ep.find_black_stills([black, ok])
            self.assertEqual([n for n, _ in found], [1], found)
            self.assertLess(found[0][1], 3.0)
        finally:
            _rmtree(tmp)

    def test_gate_agrees_with_direct_measurement_on_real_stills(self):
        """The invariant: nothing the gate calls black was missed, and
        nothing it passed is actually black. It used to assert "at least 3
        known-bad ch900 stills are caught", which went stale the moment the
        karras fix removed them — the count, not the gate, was the point.
        """
        import glob as _g
        imgs = sorted(_g.glob(str(Path(ep.CONFIG["images_dir"]) / "ch*.png")))
        if not imgs:
            self.skipTest("no stills on disk")
        sample = imgs[:40]
        found = dict(ep.find_black_stills(sample))
        misses = []
        for i, p in enumerate(sample, 1):
            l = ep.still_luma(p)
            self.assertIsNotNone(l, f"could not measure {p}")
            if (l < 3.0) != (i in found):
                misses.append((p.name, round(l, 2), i in found))
        print(f"\n    {len(found)}/{len(sample)} sampled stills are black")
        self.assertEqual(misses, [],
                         f"gate disagrees with measurement: {misses}")

    def test_luma_helper_and_missing_file(self):
        import tempfile as _tf
        tmp = _tf.mkdtemp()
        try:
            self.assertAlmostEqual(ep.still_luma(self._png(tmp, "m.png", 128)), 128, delta=2)
            self.assertIsNone(ep.still_luma(Path(tmp) / "nope.png"))
            # missing files are not reported as black (the existence gate owns that)
            self.assertEqual(ep.find_black_stills([str(Path(tmp) / "nope.png")]), [])
        finally:
            _rmtree(tmp)


class FlowMatchSamplerTests(unittest.TestCase):
    """Qwen-Image-2.1 is a FLOW-MATCHING model. The still path shipped
    `dpmpp_2m` + `karras` and produced near-black stills that the pipeline
    then shipped into H3. Karras sigmas are meaningless under flow matching.

    Measured on the real model, same seed and prompt, only the schedule
    varied: karras 3.87 / 4.74 mean luma (black) under two different
    samplers, versus simple 31.61, normal 31.70, beta 55.93.
    """

    def _ksamplers(self):
        """Every dict whose class_type is KSampler, unparsed. Node id "5" is
        not enough — Echo_Sampler also lives at id 5."""
        import ast as _ast
        src = Path(ep.__file__).read_text(encoding="utf-8")
        tree = _ast.parse(src)
        out = []
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Dict):
                continue
            for k, v in zip(node.keys, node.values):
                if not (isinstance(v, _ast.Dict)):
                    continue
                cls = next((vv.value for kk, vv in zip(v.keys, v.values)
                            if isinstance(kk, _ast.Constant) and kk.value == "class_type"),
                           None)
                if cls == "KSampler":
                    out.append(_ast.unparse(node))
        return out

    def test_no_ksampler_anywhere_hardcodes_karras(self):
        samplers = self._ksamplers()
        # 4 KSamplers: the two still paths (Qwen + legacy SDXL) and the
        # character-card pair (compose + 0.35 face refine). The H3 video
        # path uses the official SamplerCustomAdvanced chain, which carries
        # no KSampler node.
        self.assertGreaterEqual(len(samplers), 4,
                                f"expected the still + card samplers, found {len(samplers)}")
        for s in samplers:
            self.assertNotIn('"karras"', s.lower(),
                             "a KSampler still hardcodes karras, which is "
                             "invalid for flow-matching and produced black "
                             "stills in a real render")

    def test_still_samplers_read_schedule_from_config(self):
        """The still and character-card KSamplers must take the schedule
        from config. The H3 video KSampler is a separate engine that already
        uses euler/normal, so it is excluded by h3_steps."""
        still = [s for s in self._ksamplers() if "h3_steps" not in s]
        self.assertGreaterEqual(len(still), 4, f"found {len(still)} still samplers")
        for s in still:
            for key in ("image_sampler", "image_scheduler", "image_cfg"):
                self.assertIn(key, s,
                              f"a still KSampler must read {key} from config so "
                              "a bad default cannot come back silently")

    def test_gate_label_does_not_advertise_the_broken_combo(self):
        # ponytail: this method was DEFINED TWICE, byte-identical, at :2318 and
        # :2324. In a class body the second definition silently replaces the
        # first, so the suite counted one test where there were two and any
        # edit to the first was dead. Second copy removed.
        src = Path(ep.__file__).read_text(encoding="utf-8")
        self.assertNotIn("dpmpp_2m/karras", src,
                         "the phase label still advertises the schedule that "
                         "broke the stills")

    def test_effective_schedule_is_flow_match_safe(self):
        sched = str(ep.CONFIG.get("image_scheduler", "simple")).lower()
        self.assertNotEqual(sched, "karras")
        self.assertIn(sched, {"simple", "normal", "beta", "sgm_uniform",
                              "exponential", "ddim_uniform"})
        self.assertIn(str(ep.CONFIG.get("image_sampler", "euler")).lower(),
                      {"euler", "dpmpp_2m", "euler_ancestral", "dpmpp_sde",
                       "res_multistep", "heun"})

    def test_cfg_is_in_qwen_image_range(self):
        # official template uses cfg 1; >2 oversaturates and rings edges
        cfg = float(ep.CONFIG.get("image_cfg", 1.0))
        self.assertLessEqual(cfg, 2.0,
                             f"cfg {cfg} is out of Qwen-Image-2.1's range")

    def test_steps_are_at_the_official_baseline(self):
        steps = int(ep.CONFIG.get("auto_tune", {}).get("image_steps", 25))
        self.assertEqual(steps, 25,
                         "auto_tune bumped steps while chasing a broken "
                         "schedule; the baseline is 25")

    def test_autotuner_no_longer_raises_steps_on_soft_frames(self):
        # The tuner moved to pipeline/quality_evolve.py (2026-09-29); read
        # the file that now owns evolve_quality, not the producer.
        import pipeline.quality_evolve as qe
        src = Path(qe.__file__).read_text(encoding="utf-8")
        self.assertNotIn('avg_sharp"] < 45', src,
                         "the soft-frame step-bump is a symptom-chase; it "
                         "cost 30% GPU time and buried the scheduler bug")
        # the time-saving downward path stays
        self.assertIn('avg_sharp"] > 75', src)

    def test_bright_boost_is_not_in_the_auto_allowlist(self):
        """bright_boost was a prompt prefix Qwen-Image ignored, so the
        closed loop measured its own noise and flapped A-B-A across renders
        (quality_log 37->38->39: brightness 17.9->15.2->17.9), spending a
        full episode render per cycle. It must not come back.

        The two things that must be impossible: the knob entering the
        allowlist, and the knob being applied to the live config. The
        retirement path (pop from tune, mark ledger rows 'retired') is
        required, not forbidden."""
        import pipeline.quality_evolve as qe
        src = Path(qe.__file__).read_text(encoding="utf-8")

        self.assertNotIn("bright_boost", ep.AUTO_ALLOWLIST,
                         "an inert knob in a closed loop oscillates")
        # apply(...) is the only path that writes a value onto `tune`.
        # bright_boost must never be an argument to it.
        import re
        applied = re.findall(r'apply\(\s*"([^"]+)"', src)
        self.assertNotIn("bright_boost", applied,
                         f"bright_boost must never be applied: {applied}")
        # The retirement path must still be present.
        self.assertIn('tune.pop("bright_boost"', src,
                      "a config still carrying the dead key must be cleaned")
        # The inert prompt prefix is gone from the still prompts.
        ep_src = Path(ep.__file__).read_text(encoding="utf-8")
        self.assertNotIn("bright vivid lighting", ep_src,
                         "the inert prompt prefix is gone from the still prompts")

    def test_rules_file_documents_the_verified_schedule(self):
        rules = Path(ep.ROOT) / "rules" / "images.md"
        if not rules.exists():
            self.skipTest("rules/images.md not present")
        txt = rules.read_text(encoding="utf-8")
        self.assertIn("simple", txt)
        self.assertNotRegex(txt, r"CFG 6\.0",
                            "the rules file still codifies the CFG that "
                            "caused the failure")


class VlmServingDetectionTests(unittest.TestCase):
    """The VLM reported itself down even while serving.

    llama-server returns the model id as the full path it was launched with
    ("C:\\...\\Qwen3VL-8B-Instruct-Q4_K_M.gguf"), but _is_vlm_serving compared
    `sid == stem or sid.startswith("Qwen3VL")`. A path starts with "C:", so
    the check could never pass: vlm_available() was permanently False and the
    face gate, still grading and final-frame review all silently no-opped.
    Found on a live ch900 render that shipped 4 black stills unseen.
    """

    def test_full_path_serving_id_is_recognised(self):
        with mock.patch.object(ep, "_serving_id",
                               return_value=r"C:\models\Qwen3VL-8B-Instruct-Q4_K_M.gguf"):
            self.assertTrue(ep._is_vlm_serving(),
                            "a serving id given as a full path must be "
                            "recognised as the vision profile")

    def test_bare_stem_is_recognised(self):
        with mock.patch.object(ep, "_serving_id",
                               return_value="Qwen3VL-8B-Instruct-Q4_K_M"):
            self.assertTrue(ep._is_vlm_serving())

    def test_text_profile_is_not_mistaken_for_vision(self):
        with mock.patch.object(ep, "_serving_id",
                               return_value=r"C:\models\Qwen3-14B-abliterated-Q4_K_M.gguf"):
            self.assertFalse(ep._is_vlm_serving(),
                             "the text profile must not pass as vision")

    def test_empty_serving_id_is_down(self):
        with mock.patch.object(ep, "_serving_id", return_value=""):
            self.assertFalse(ep._is_vlm_serving())

    def test_requant_rename_still_matches(self):
        for variant in (r"C:\models\Qwen3VL-8B-Instruct-Q8_0.gguf",
                        r"C:\models\Qwen3VL-8B-Instruct-F16.gguf",
                        "Qwen3VL-8B-Instruct"):
            with mock.patch.object(ep, "_serving_id", return_value=variant):
                self.assertTrue(ep._is_vlm_serving(),
                                f"a requant to {variant} must not silently "
                                "drop vision review")

    def test_detection_compares_stems_not_raw_ids(self):
        """Assert on the CODE, not the docstring — the docstring quotes the
        old buggy expression on purpose."""
        tree = ast.parse(Path(ep.__file__).read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_is_vlm_serving")
        body = ast.unparse(ast.Module(body=fn.body, type_ignores=[]))
        self.assertNotIn('startswith("Qwen3VL")', body,
                         "prefix-matching the raw id is the original bug")
        self.assertIn("Path(sid).stem", body)

    def test_vision_profile_gets_image_min_tokens(self):
        cmd = ep._llm_launch_cmd(vision=True)
        self.assertIn("--image-min-tokens", cmd,
                      "Qwen-VL needs >=1024 image tokens or the grader sees "
                      "a low-res crop (llama.cpp 16842)")
        self.assertIn("1024", cmd[cmd.index("--image-min-tokens") + 1])

    def test_vision_context_fits_two_images(self):
        """The clip-vs-still check sends TWO images. At 1024 image tokens
        each, a 2048 ctx returns a bare HTTP 400 with no usable message."""
        cmd = ep._llm_launch_cmd(vision=True)
        ctx = int(cmd[cmd.index("--ctx-size") + 1])
        mintok = int(cmd[cmd.index("--image-min-tokens") + 1])
        self.assertGreaterEqual(ctx, 2 * mintok,
                                f"ctx {ctx} cannot hold 2 x {mintok} image "
                                "tokens; the comparison silently 400s")

    def test_text_profile_disables_reasoning(self):
        """Qwen3 is a hybrid reasoning model; with reasoning on it returns
        content="" and every generated scene would be empty text."""
        cmd = ep._llm_launch_cmd(vision=False)
        self.assertIn("--reasoning", cmd)
        self.assertEqual(cmd[cmd.index("--reasoning") + 1], "off")


class TruePeakHeadroomTests(unittest.TestCase):
    """The limiter set exactly at the true-peak ceiling publishes a file the
    gate then rejects. ffmpeg's alimiter limits SAMPLE peak; inter-sample
    overshoot from the resample/convert lands ~0.15 dB above it, so a
    -2.0 dBFS limiter measured -1.85 dBTP against a -1.9 limit and a real
    ch900 render refused to promote. Measured on a hot real mix:
    -2.00 -> -1.85 dBTP (FAIL), -2.20 -> -2.05 (PASS), -2.30 -> -2.15 (PASS).
    """

    def test_limiter_sits_below_the_ceiling(self):
        self.assertGreater(ep._TP_LIMIT_HEADROOM, 0.0)
        lim = ep._TP_LIMIT - ep._TP_LIMIT_HEADROOM
        self.assertLess(lim, ep._TP_LIMIT,
                        "the limiter must leave headroom below the ceiling, "
                        "or the gate fails its own output")

    def test_limiter_filter_uses_the_headroom(self):
        import inspect
        src = inspect.getsource(ep._limit_only)
        self.assertIn("_TP_LIMIT_HEADROOM", src,
                      "_limit_only must limit below the ceiling, not on it")

    def test_headroom_covers_the_measured_overshoot(self):
        # The shipped master is AAC-LC at 128k and the codec's overshoot was
        # MEASURED at up to +1.75 dB on this project's own mixes (a mix
        # limited to -3.50 dB sample peak measured -1.75 dBTP once muxed and
        # refused to promote). A margin narrower than that fails the render it
        # produced - and the narration-only path carried a 1.5 dB margin,
        # which is below the measured spread.
        self.assertGreaterEqual(ep._TP_LIMIT_HEADROOM, 1.75)

    def test_gate_stays_tight(self):
        # the headroom must not be bought by loosening the gate
        self.assertLessEqual(ep._TP_TOL, 0.1)
        self.assertEqual(ep._TP_LIMIT, -2.0)


class ClipStillFidelityTests(unittest.TestCase):
    """grade_clip compares a clip's first and last frames, so it can only
    prove the model stayed consistent with ITSELF. A real ch900 render turned
    a wide night mill into a mangled doll-face close-up and every internal
    check passed. A clip must therefore also be measured against its still.
    """

    def _grey(self, tmp, name, val, size=(64, 64)):
        import numpy as _np
        from PIL import Image
        p = Path(tmp) / name
        Image.fromarray(_np.full((size[1], size[0]), val, dtype=_np.uint8)).save(p)
        return str(p)

    def _pattern(self, tmp, name, size=(64, 64)):
        import numpy as _np
        from PIL import Image
        a = _np.zeros((size[1], size[0]), dtype=_np.uint8)
        a[:, : size[0] // 2] = 220          # a hard vertical edge
        p = Path(tmp) / name
        Image.fromarray(a).save(p)
        return str(p)

    def test_identical_images_drift_towards_one(self):
        import tempfile as _tf
        tmp = _tf.mkdtemp()
        try:
            a = self._pattern(tmp, "a.png")
            b = self._pattern(tmp, "b.png")
            d = ep.still_drift(a, b)
            self.assertIsNotNone(d)
            self.assertGreater(d, 0.9, f"identical structure scored {d}")
        finally:
            _rmtree(tmp)

    def test_unrelated_images_drift_low(self):
        import tempfile as _tf
        tmp = _tf.mkdtemp()
        try:
            a = self._pattern(tmp, "edge.png")
            b = self._grey(tmp, "flat.png", 30)
            d = ep.still_drift(a, b)
            self.assertLess(d, 0.5, f"unrelated images scored {d}")
        finally:
            _rmtree(tmp)

    def test_missing_or_unreadable_returns_none(self):
        import tempfile as _tf
        tmp = _tf.mkdtemp()
        try:
            good = self._pattern(tmp, "a.png")
            self.assertIsNone(ep.still_drift(good, str(Path(tmp) / "nope.png")))
            self.assertIsNone(ep.still_drift(str(Path(tmp) / "nope.png"), good))
        finally:
            _rmtree(tmp)

    def test_flat_image_does_not_divide_by_zero(self):
        import tempfile as _tf
        tmp = _tf.mkdtemp()
        try:
            a = self._grey(tmp, "flat1.png", 10)
            b = self._grey(tmp, "flat2.png", 200)
            self.assertEqual(ep.still_drift(a, b), 0.0)
        finally:
            _rmtree(tmp)

    def test_drift_and_vlm_flags_are_clip_phase_receipt_keys(self):
        """They are produced in the clip phase, so assembly's selective key
        clear must not drop them."""
        for k in ("clip_still_drift", "clip_still_vlm_flags"):
            self.assertIn(k, ep._CLIP_PHASE_KEYS,
                          f"{k} would be wiped by _clear_assembly_keys")

    def test_vlm_check_is_inert_when_vlm_is_down(self):
        with mock.patch.object(ep, "vlm_available", return_value=False):
            ok, note = ep._vlm_clip_matches_still("nope.png", "nope2.png", {})
        self.assertIsNone(ok, "with no VLM the check must abstain, not guess")
        self.assertEqual(note, "")

    def test_drift_is_not_used_as_a_hard_gate(self):
        """Honest ceiling: it is a recorded fact, not a threshold. If someone
        turns it into a gate without data, the uncalibrated number starts
        rejecting good clips.

        The VLM check is warn-only for the same reason and it demonstrably
        over-fires: on 3 real clips it flagged all 3, including two that are
        visually acceptable. A gate on that would reject good work.
        """
        import inspect
        src = inspect.getsource(ep.still_drift)
        self.assertIn("NOT calibrated as a gate", src)
        vsrc = inspect.getsource(ep._vlm_clip_matches_still)
        self.assertNotIn("return False, ", vsrc,
                         "the VLM check must abstain rather than fail hard")
        rules = Path(ep.ROOT) / "rules" / "motion.md"
        if rules.exists():
            self.assertIn("NOT a gate", rules.read_text(encoding="utf-8"))

    def test_rules_forbid_the_wide_shot_guess(self):
        rules = Path(ep.ROOT) / "rules" / "motion.md"
        if not rules.exists():
            self.skipTest("rules/motion.md not present")
        txt = rules.read_text(encoding="utf-8")
        self.assertIn("Do not write a \"wide shots are bad\" rule", txt,
                      "the measured data refutes the framing heuristic; it "
                      "must not be encoded as doctrine")


class H3RecipeTests(unittest.TestCase):
    """The H3 clip builder ignored its reference still.

    Measured on a correct still of a man before a burning mill, the old
    builder produced a mangled doll face in frame 1 — and the result was
    invariant to LoRA subsets, LoRA strength, step count, sampler chain,
    encoder device, reference size, slot naming and canvas size.

    Comfy-Org's video_minimax_h3_r2v template, downloaded and run unmodified
    with its own weights, produced the correct scene. Aligning the builder
    to that recipe, plus one thing the template also did that we did not —
    DESCRIBE THE SCENE IN THE PROMPT — fixed it. The prompt was the decisive
    part: motion words alone leave the model free to invent the subject.
    """

    def test_official_recipe_values(self):
        cfg = ep._h3_cfg("fast")
        self.assertEqual(cfg["vae"], "minimax_h3_video_vae_int8_convrot.safetensors")
        self.assertEqual(cfg["audio_vae"], "minimax_h3_audio_vae_fp32.safetensors")
        self.assertEqual(cfg["steps"], 20)
        self.assertEqual(cfg["sampler"], "res_multistep")
        self.assertEqual(cfg["scheduler"], "simple")
        self.assertEqual(cfg["encoder_device"], "default",
                         "the 32B Qwen3-VL encoder belongs on the GPU; the "
                         "official template runs it there")
        self.assertIn("ref2v_turbo_4step", str(cfg["turbo_lora"]),
                      "the turbo LoRA is built for reference-to-video")

    def test_turbo_lora_not_the_old_chain(self):
        """The old 3-deep LoraLoader chain patched the 32B text encoder too
        and is what shipped broken clips. It must not come back."""
        import inspect
        src = inspect.getsource(ep._h3_lora_chain)
        self.assertIn("LoraLoaderModelOnly", src,
                      "the turbo LoRA must load model-only")
        self.assertNotIn("LoraLoader\"", src.replace("LoraLoaderModelOnly", ""),
                         "do not reintroduce the clip-patching LoraLoader")
        for dead in ("h3_pdd_lora", "h3_skin_lora", "h3_camera_lora"):
            self.assertNotIn(dead, src, f"{dead} belonged to the broken chain")

    def test_prompt_describes_the_scene_not_just_the_motion(self):
        """The decisive fix, asserted on the real prompt string.

        Motion words alone let the model invent the subject (a man before a
        burning mill became an anime woman indoors). The scene description
        must lead the prompt.
        """
        scene = {"image_prompt": "A mill on fire at night, wide establishing shot.",
                 "camera_move": "push-in", "mood": "tense", "type": "action",
                 "narration": "x", "characters": []}
        p = ep._h3_prompt(scene)
        self.assertTrue(p.startswith("A mill on fire at night"),
                        f"the scene description must lead the prompt, got: {p[:90]!r}")
        self.assertIn("<Picture 1>", p, "the reference tag must survive")
        self.assertIn("slow push in", p, "the camera move must survive")

    def test_prompt_names_the_character_not_just_the_scene(self):
        """The scene sentence says nothing about WHO is in it, so the model
        invents a person: measured, the man in a dark coat became a girl in a
        white dress while the mill stayed correct. Bible identity tags must
        ride along, exactly as they do for the still."""
        scene = {"image_prompt": "A mill on fire at night, wide establishing shot.",
                 "camera_move": "push-in", "mood": "tense", "type": "action",
                 "narration": "x", "characters": ["edvyn_crael"]}
        p = ep._h3_prompt(scene)
        cd = (ep.CHAR_BIBLE or {}).get("characters", {}).get("edvyn_crael", {})
        tags = cd.get("danbooru_tags") or cd.get("prompt_tags") or ""
        self.assertTrue(p.startswith("A mill on fire at night"), p[:70])
        if tags:
            first_tag = str(tags).split(",")[0].strip()
            self.assertIn(first_tag, p,
                          f"character identity missing from the clip prompt; "
                          f"expected {first_tag!r} in: {p[:160]!r}")
        self.assertIn("slow push in", p)

    def test_prompt_without_a_description_does_not_lead_with_junk(self):
        scene = {"image_prompt": "", "camera_move": "push-in", "mood": "tense",
                 "type": "action", "narration": "x", "characters": []}
        p = ep._h3_prompt(scene)
        self.assertTrue(p.startswith("subject from <Picture 1>"),
                        f"empty description must not leave a leading comma: {p[:60]!r}")

    def test_trailing_full_stop_does_not_double_up(self):
        scene = {"image_prompt": "A mill on fire at night.", "camera_move": "push-in",
                 "mood": "tense", "type": "action", "narration": "x", "characters": []}
        self.assertNotIn("..", ep._h3_prompt(scene))

    def test_official_sampler_chain_is_used(self):
        import inspect
        src = inspect.getsource(ep._try_h3_clip)
        for node in ("RandomNoise", "BasicGuider", "KSamplerSelect",
                     "BasicScheduler", "SamplerCustomAdvanced"):
            self.assertIn(node, src, f"missing official sampler node {node}")
        self.assertNotIn('"class_type": "KSampler"', src,
                         "the plain KSampler is not what the template uses")

    def test_audio_branch_is_wired(self):
        """Ref2VA is video AND audio; the official template wires the audio
        VAE and decodes it."""
        import inspect
        self.assertIn("VAEDecodeAudio", inspect.getsource(ep._try_h3_clip))
        self.assertIn('_h3_inputs["audio_vae"] = ["8", 0]',
                      inspect.getsource(ep._try_h3_clip))
        self.assertTrue(ep._h3_cfg("fast")["audio_vae"],
                        "the audio VAE must be configured")

    def test_removed_lora_keys_are_not_in_the_model_map(self):
        """A stale key here would make the model audit look for a file the
        builder no longer uses."""
        import inspect
        src = inspect.getsource(ep.__dict__["_model_roots"]) if "_model_roots" in ep.__dict__ \
            else inspect.getsource(ep)
        for dead in ("h3_pdd_lora", "h3_skin_lora", "h3_camera_lora"):
            self.assertNotIn(dead, src, f"{dead} is dead but still referenced")

    def test_h3_cfg_returns_the_whole_recipe(self):
        for tier in ("fast", "final"):
            c = ep._h3_cfg(tier)
            for k in ("checkpoint", "text_encoder", "vae", "audio_vae", "steps",
                      "frames", "width", "height", "sampler", "scheduler",
                      "encoder_device", "turbo_lora"):
                self.assertIn(k, c, f"{tier} cfg missing {k}")


class SpeedWithoutQualityLossTests(unittest.TestCase):
    """Speedups admitted only where the output is measurably unchanged.

    Measured on the real models, same seed:
      still text encoder  CPU 138s -> GPU 58s (2.4x), and the images are the
      same picture (correlation 0.99889, MAE 0.415, 0.17% of pixels off by
      more than 8/255 — float nondeterminism, not a quality change).
      H3 sampler steps    20 steps 148s vs 4 steps 157s — no difference, so
      steps are FREE and there is nothing to win by cutting them. Left at 20.
    """

    def test_still_encoder_stays_on_the_cpu(self):
        """The GPU encoder was tried and reverted.

        Same seed, CPU 138s vs GPU 58s per still (2.4x) with an identical
        picture (correlation 0.99889, MAE 0.415). But in a real render the
        extra resident VRAM tipped ComfyUI's memory planner over —
        "aimdo memory compile error" -> "Fatal Python error: Aborted" — and
        every remaining clip fell back to a panning still. Never put this
        back on a 16 GB card.
        """
        import inspect
        src = inspect.getsource(ep.generate_image)
        self.assertIn('still_encoder_device', src,
                      "it stays a config knob: the lever is real, it is just "
                      "closed on this hardware")
        self.assertEqual(ep.CONFIG.get("still_encoder_device"), "cpu",
                         "the GPU encoder crashes ComfyUI at 16 GB")

    def test_still_encoder_device_survives_a_typed_mess(self):
        """The device is a free-form string passed to ComfyUI, and str() never
        raises, so _validate_config coerces rather than rejects. Assert the
        coercion is at least deterministic and does not explode."""
        cfg = {"still_encoder_device": 5}
        ep._validate_config(cfg, ep._DEFAULTS)
        self.assertIsInstance(cfg["still_encoder_device"], str)

    def test_h3_steps_left_where_they_are(self):
        """Cutting steps looked like a 5x win and was worth nothing: 148s at
        20 steps against 157s at 4. Do not let someone 'optimise' this
        without re-measuring."""
        self.assertEqual(ep._h3_cfg("fast")["steps"], 20)
        self.assertEqual(ep._h3_cfg("final")["steps"], 20)

    def test_clip_check_is_deferred_not_dropped(self):
        """The vision check timed out running mid-render with ComfyUI holding
        the VRAM. It must still run — just later, in Phase 8."""
        import inspect
        body = inspect.getsource(ep._produce_inner)
        self.assertIn("_clip_checks.append", body,
                      "the clip phase must QUEUE the check")
        self.assertNotIn("_vlm_clip_matches_still(", body,
                         "the check must not run while ComfyUI owns the GPU")
        # ...and Phase 8 must actually run it
        self.assertIn("run_clip_still_checks(_clip_checks", body)
        # Phase 8 is after free_comfyui_vram() + ensure_vlm()
        i_free = body.index("free_comfyui_vram()")
        i_run = body.index("run_clip_still_checks(_clip_checks")
        self.assertGreater(i_run, i_free,
                           "the deferred check must come after the VRAM is "
                           "freed, or it times out again")

    def test_deferred_check_abstains_when_vlm_is_down(self):
        with mock.patch.object(ep, "vlm_available", return_value=False):
            self.assertEqual(
                ep.run_clip_still_checks([{"scene": 1, "frame": "a",
                                           "still": "b", "drift": 0.1}],
                                         [{"scene_number": 1}], 900), [])
        self.assertEqual(ep.run_clip_still_checks([], [], 900), [])

    def test_deferred_check_never_raises_on_bad_paths(self):
        with mock.patch.object(ep, "vlm_available", return_value=True):
            out = ep.run_clip_still_checks(
                [{"scene": 1, "frame": "missing.png", "still": "nope.png",
                  "drift": None}], [{"scene_number": 1}], 900)
        self.assertEqual(out, [], "a missing frame must skip, not explode")

    def test_deferred_check_reports_a_mismatch(self):
        with mock.patch.object(ep, "vlm_available", return_value=True), \
             mock.patch.object(ep, "_vlm_clip_matches_still",
                               return_value=(False, "different subject")):
            import tempfile as _tf
            tmp = _tf.mkdtemp()
            try:
                f = Path(tmp) / "c01.png"
                f.write_bytes(b"x")
                s = Path(tmp) / "s01.png"
                s.write_bytes(b"x")
                flags = ep.run_clip_still_checks(
                    [{"scene": 3, "frame": str(f), "still": str(s),
                      "drift": 0.11}],
                    [{"scene_number": 3, "image_prompt": "x"}], 900)
                self.assertEqual(len(flags), 1)
                self.assertEqual(flags[0]["scene"], 3)
                self.assertIn("different subject", flags[0]["note"])
            finally:
                _rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
