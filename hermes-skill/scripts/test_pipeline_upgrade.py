#!/usr/bin/env python3
"""Tests for the pipeline upgrade modules.

ponytail: this was a straight-line script with 77 module-level `assert`s.
`unittest discover` registered ZERO tests from it, so none of those checks ran
as part of the suite — they executed as an import-time side effect and a
failure surfaced as `ImportError: Failed to import test module`, with no test
name attached. Now a real TestCase: every assertion is counted, named and
reported independently.
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

from pipeline.schemas import (validate_audio_plan, validate_dict,
                              validate_director_package, validate_rights_manifest,
                              validate_release_package, validate_release_report,
                              validate_style_bible, DIRECTOR_PACKAGE)
from pipeline.director_package import (load_director_package,
                                       save_director_package)
from pipeline.rights_manifest import (add_audio_asset, add_entry,
                                      check_prompt_rights,
                                      load_rights_manifest,
                                      populate_initial_manifest,
                                      preflight_rights_gate,
                                      save_rights_manifest, _empty_manifest)
from pipeline.audio_planner import (generate_audio_plan, load_audio_plan,
                                    save_audio_plan)
from pipeline.subtitle_generator import generate_subtitles
from pipeline.release_report import create_release_report, save_release_report
from pipeline.release_package import (build_release_package,
                                      generate_disclosure_checklist,
                                      generate_metadata,
                                      generate_thumbnail_concepts,
                                      music_license_state)
from pipeline.analytics_log import (add_entry as add_analytics,
                                    load_analytics_log, propose_experiments,
                                    summarize)
from pipeline import config as ini_config

VALID_DP = {
    "episode_promise": "A shocking discovery changes everything",
    "target_audience": "Adult anime fans",
    "target_emotion": "Suspense",
    "cold_open_hook": {"description": "Man finds box", "duration_range": "0-15s"},
    "central_dramatic_question": "What's in the box?",
    "midpoint_reversal": "The box is empty",
    "climax": "He realizes the truth",
    "ending": "Cliffhanger",
    "beat_sheet": [{"beat_number": 1, "description": "Opening",
                    "source_chapter_ref": "ch1"}],
    "scene_plan": [{"scene_number": 1, "purpose": "HOOK",
                    "conflict_or_change": "Discovery", "focal_character": "Edvyn",
                    "location": "City", "time_of_day": "Dawn",
                    "narration_or_dialogue": "narration", "visual_motif": "fog",
                    "intended_shot_type": "wide", "camera_movement": "push-in",
                    "target_duration": "5"}],
    "shot_budget": {"total_shots": 1, "shot_type_limits": {},
                    "composition_rules": "no dupes"},
    "continuity_ledger": {"character_states": {}, "unresolved_promises": [],
                          "prohibited_contradictions": []},
}

VALID_RM = {
    "novel_adaptation_rights": {"asset_id": "novel", "asset_type": "rights",
                                 "source": "owner", "license": "original",
                                 "commercial_use": "yes",
                                 "local_path_or_hash": "abc"},
    "image_model_licenses": [],
    "audio_assets": [],
    "entries": [],
}

VALID_SB = {
    "art_direction": "anime", "palette": "dark", "lighting": "moody",
    "lens_language": "cinematic", "aspect_ratio": "16:9",
    "recurring_motifs": ["fog"], "prohibited_visual_traits": ["3d"],
    "negative_prompts": "none",
}

TEST_SCENES = [
    {"scene_number": 1, "shot_size": "wide", "characters": [], "mood": "tense",
     "type": "establishing", "narration": ""},
    {"scene_number": 2, "shot_size": "close-up", "characters": ["edvyn_crael"],
     "mood": "somber", "type": "action", "narration": "He looked at the box"},
]

SUB_SCENES = [
    {"scene_number": 1, "_cap_start": 0.0, "_cap_dur": 5.0,
     "narration": "The city waited in silence", "dialogue": "", "characters": []},
    {"scene_number": 2, "_cap_start": 5.0, "_cap_dur": 4.0,
     "narration": "He looked up", "dialogue": "What did you find?",
     "characters": ["edvyn_crael"]},
]

TEST_ASSEMBLY = {
    "promoted": True,
    "qa_av_sync": {"passed": True, "violations": [],
                   "checked": [{"scene": 1}, {"scene": 2}]},
    "true_peak": {"passed": True, "peak_dbtp": -2.5},
    "streams": {"fps": 24, "width": 1920, "height": 1080, "channels": 2},
    "scenes": [{"clip": "scene_01.mp4"}, {"clip": "scene_02.mp4"}],
    "music_bed": "Dark Fog.mp3",
}


def _to_ms(ts):
    ts = ts.replace(".", ",")
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


class SchemaValidationTests(unittest.TestCase):
    def test_valid_director_package(self):
        self.assertEqual(validate_director_package(VALID_DP), [])

    def test_invalid_director_package_names_the_missing_key(self):
        errors = validate_director_package({"episode_promise": "test"})
        self.assertTrue(errors)
        self.assertIn("target_audience", errors[0])

    def test_valid_rights_manifest(self):
        self.assertEqual(validate_rights_manifest(VALID_RM), [])

    def test_valid_style_bible(self):
        self.assertEqual(validate_style_bible(VALID_SB), [])

    def test_unknown_keys_are_rejected(self):
        typo = dict(VALID_DP)
        typo["unresolved_promts"] = True
        errors = validate_dict(typo, DIRECTOR_PACKAGE, "director_package")
        self.assertTrue(any("unexpected key" in e for e in errors),
                        "typos must be rejected")

    def test_validate_audio_plan(self):
        self.assertEqual(validate_audio_plan(generate_audio_plan(TEST_SCENES, 1)), [])
        self.assertTrue(validate_audio_plan([{"scene_number": 1}]),
                        "incomplete audio item must fail")


class DirectorPackageTests(unittest.TestCase):
    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            save_director_package(VALID_DP, td, 1, force=True)
            loaded = load_director_package(td, 1)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["episode_promise"], VALID_DP["episode_promise"])

    def test_no_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as td:
            save_director_package(VALID_DP, td, 1, force=True)
            modified = dict(VALID_DP, episode_promise="Modified")
            save_director_package(modified, td, 1, force=False)
            self.assertEqual(load_director_package(td, 1)["episode_promise"],
                             VALID_DP["episode_promise"])

    def test_force_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            save_director_package(VALID_DP, td, 1, force=True)
            save_director_package(dict(VALID_DP, episode_promise="Modified"),
                                  td, 1, force=True)
            self.assertEqual(load_director_package(td, 1)["episode_promise"],
                             "Modified")

    def test_save_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as td:
            save_director_package(VALID_DP, td, 1, force=True)
            self.assertEqual([p.name for p in Path(td).glob("*.tmp")], [])


class RightsManifestTests(unittest.TestCase):
    def _confirmed(self):
        m = _empty_manifest()
        m["novel_adaptation_rights"]["owner_confirmation"] = True
        m["novel_adaptation_rights"]["commercial_use"] = "yes"
        m["voice"]["owner_confirmation"] = True
        m["voice"]["commercial_use"] = "yes"
        return m

    def test_empty_template_is_unconfirmed(self):
        m = _empty_manifest()
        self.assertIs(m["novel_adaptation_rights"]["owner_confirmation"], False)

    def test_blocks_on_unconfirmed_novel_rights(self):
        _ok, blockers, _ = preflight_rights_gate(_empty_manifest())
        self.assertTrue(any("novel" in b.lower() for b in blockers))

    def test_pending_commercial_use_blocks(self):
        m = _empty_manifest()
        m["novel_adaptation_rights"]["owner_confirmation"] = True
        m["novel_adaptation_rights"]["commercial_use"] = "pending_confirmation"
        ok, _b, _ = preflight_rights_gate(m)
        self.assertFalse(ok, "pending_confirmation must block (fail-closed)")

    def test_passes_after_confirmation(self):
        ok, blockers, _ = preflight_rights_gate(self._confirmed())
        self.assertTrue(ok, f"should pass after confirmation: {blockers}")

    def test_add_and_update_entry(self):
        m = self._confirmed()
        self.assertFalse(add_entry(m, {
            "asset_id": "test_model", "asset_type": "model", "source": "local",
            "license": "open", "commercial_use": "yes",
            "local_path_or_hash": "test.safetensors"}))
        self.assertEqual(len(m["entries"]), 1)
        self.assertFalse(add_entry(m, {
            "asset_id": "test_model", "asset_type": "model", "source": "local",
            "license": "open_v2", "commercial_use": "yes",
            "local_path_or_hash": "test_v2.safetensors"}))
        self.assertEqual(len(m["entries"]), 1, "same asset_id must update, not append")
        self.assertEqual(m["entries"][0]["license"], "open_v2")

    def test_add_entry_rejects_invalid_directly(self):
        self.assertTrue(add_entry(self._confirmed(), {"asset_id": "x"}))

    def test_prompt_imitation_check(self):
        m = self._confirmed()
        self.assertTrue(check_prompt_rights(
            "beautiful anime scene in a dark city", m)[0])
        self.assertFalse(check_prompt_rights("in the style of studio ghibli", m)[0])

    def test_save_load_round_trip(self):
        m = self._confirmed()
        add_entry(m, {"asset_id": "a", "asset_type": "model", "source": "local",
                      "license": "open", "commercial_use": "yes",
                      "local_path_or_hash": "a"})
        with tempfile.TemporaryDirectory() as td:
            save_rights_manifest(m, td)
            loaded = load_rights_manifest(td)
            self.assertIs(loaded["novel_adaptation_rights"]["owner_confirmation"], True)
            self.assertEqual(len(loaded["entries"]), 1)

    def test_audio_asset_and_initial_manifest(self):
        m = self._confirmed()
        add_audio_asset(m, "dark_fog.mp3", "music", "local",
                        "unknown", "unknown")
        self.assertTrue(any(a.get("asset_id") == "dark_fog.mp3"
                            for a in m["music"] + m["sfx"]))
        # populate_initial_manifest(root, config) is non-destructive and writes
        # to disk, so exercise it against a temp root.
        with tempfile.TemporaryDirectory() as td:
            save_rights_manifest(self._confirmed(), td)
            before = load_rights_manifest(td)["novel_adaptation_rights"].get(
                "owner_confirmation")
            populate_initial_manifest(td, {})
            after = load_rights_manifest(td)
            self.assertEqual(after["version"], "2.0")
            self.assertEqual(
                after["novel_adaptation_rights"]["owner_confirmation"], before,
                "populate must preserve existing confirmations")

    def test_release_report_and_package_validators_exist(self):
        self.assertTrue(callable(validate_release_report))
        self.assertTrue(callable(validate_release_package))


class AudioPlannerTests(unittest.TestCase):
    def test_generate_plan(self):
        plan = generate_audio_plan(TEST_SCENES, 1)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["scene_number"], 1)
        self.assertEqual(plan[0]["voice_priority"], "none")
        self.assertIs(plan[0]["intended_silence"], True)
        self.assertEqual(plan[1]["voice_priority"], "high")
        self.assertGreaterEqual(plan[0]["emotional_intensity"], 0.5)
        self.assertGreaterEqual(plan[1]["emotional_intensity"], 0.3)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            plan = generate_audio_plan(TEST_SCENES, 1)
            save_audio_plan(plan, 1, td)
            loaded = load_audio_plan(1, td)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded), 2)

    def test_mood_aware_music_selection(self):
        mood_scenes = [
            {"scene_number": 1, "mood": "tense", "shot_size": "wide",
             "characters": [], "type": "", "narration": "x"},
            {"scene_number": 2, "mood": "hopeful", "shot_size": "wide",
             "characters": [], "type": "", "narration": "x"},
        ]
        with tempfile.TemporaryDirectory() as md:
            Path(md, "Dark Fog.mp3").write_bytes(b"a")
            Path(md, "Resolution Dawn.mp3").write_bytes(b"b")
            mp = generate_audio_plan(mood_scenes, 1, music_dir=md)
            self.assertIn("Dark Fog", mp[0]["music_cue"])
            self.assertIn("Resolution", mp[1]["music_cue"])

    def test_null_mood_does_not_crash(self):
        """`scene.get("mood", "neutral")` is the default for a MISSING key,
        not for a null one — and scenes/*.json is hand-edited."""
        plan = generate_audio_plan(
            [{"scene_number": 1, "mood": None, "shot_size": "wide",
              "characters": [], "type": "", "narration": "x"}], 1)
        self.assertEqual(len(plan), 1)


class SubtitleGeneratorTests(unittest.TestCase):
    def test_srt_and_vtt_with_speaker_labels(self):
        with tempfile.TemporaryDirectory() as td:
            srt, vtt = generate_subtitles(SUB_SCENES, 1, td)
            self.assertTrue(srt.exists())
            self.assertTrue(vtt.exists())
            srt_text = srt.read_text(encoding="utf-8")
            self.assertIn("The city waited", srt_text)
            self.assertIn("WEBVTT", vtt.read_text(encoding="utf-8"))
            self.assertIn("[edvyn_crael]", vtt.read_text(encoding="utf-8"))

    def test_missing_timing_does_not_overlap_at_zero(self):
        no_time = [
            {"scene_number": 1, "narration": "First line here", "dialogue": "",
             "characters": []},
            {"scene_number": 2, "narration": "Second line here", "dialogue": "",
             "characters": []},
        ]
        with tempfile.TemporaryDirectory() as sd:
            srt_p, _vtt_p = generate_subtitles(no_time, 99, sd)
            txt = srt_p.read_text(encoding="utf-8")
            self.assertEqual(txt.count("-->"), 2)
            starts, ends = [], []
            for line in txt.splitlines():
                if "-->" in line:
                    a, b = [t.strip() for t in line.split("-->")]
                    starts.append(_to_ms(a))
                    ends.append(_to_ms(b))
            self.assertGreaterEqual(starts[1], ends[0], f"overlapping: {starts} {ends}")

    def test_speaker_label_only_when_unambiguous(self):
        """A three-hander scene where only the SECOND character speaks used to
        caption `[Kael] "..."` — attributed to someone who did not say it."""
        scenes = [{"scene_number": 1, "_cap_start": 0.0, "_cap_dur": 3.0,
                   "narration": "", "dialogue": "I never said that.",
                   "characters": ["kael", "mira", "tomas"]}]
        with tempfile.TemporaryDirectory() as td:
            _srt, vtt = generate_subtitles(scenes, 1, td)
            self.assertNotIn("[kael]", vtt.read_text(encoding="utf-8"))


class ReleaseReportTests(unittest.TestCase):
    def test_pass_on_clean_assembly(self):
        r = create_release_report(1, SUB_SCENES, TEST_ASSEMBLY)
        self.assertIn(r["verdict"], ("PASS", "PASS_WITH_WARNINGS"), r["verdict"])
        self.assertEqual(r["summary"]["blockers"], 0)

    def test_not_release_ready_on_failed_assembly(self):
        r = create_release_report(1, SUB_SCENES,
                                  {"promoted": False, "error": "assembly failed"})
        self.assertEqual(r["verdict"], "NOT_RELEASE_READY")
        self.assertGreater(r["summary"]["blockers"], 0)

    def test_blocks_on_av_sync_failure(self):
        bad = dict(TEST_ASSEMBLY)
        bad["qa_av_sync"] = {"passed": False, "violations": [{"scene": 1}]}
        self.assertEqual(create_release_report(1, SUB_SCENES, bad)["verdict"],
                         "NOT_RELEASE_READY")

    def test_null_receipt_is_handled(self):
        r = create_release_report(1, SUB_SCENES, None)
        self.assertEqual(r["verdict"], "NOT_RELEASE_READY")
        self.assertGreater(r["summary"]["blockers"], 0)

    def test_empty_clip_slots_lower_coverage(self):
        empty = dict(TEST_ASSEMBLY)
        empty["scenes"] = [{}, {}]
        rep = create_release_report(1, SUB_SCENES, empty)
        self.assertTrue(
            any("clip" in c["message"].lower()
                for c in rep["issues"] if c["severity"] == "REVIEW"),
            "empty slots must lower coverage")

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = save_release_report(
                create_release_report(1, SUB_SCENES, None), td, 1)
            self.assertTrue(path.exists())
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["verdict"],
                "NOT_RELEASE_READY")


class ReleasePackageTests(unittest.TestCase):
    def test_metadata_generation(self):
        meta = generate_metadata(1, TEST_SCENES)
        for key in ("title_options", "description", "chapters", "tags"):
            self.assertIn(key, meta)
        self.assertTrue(meta["title_options"])

    def test_metadata_handles_no_scenes(self):
        """`max()` on an empty iterable used to raise ValueError, and the
        caller's bare except turned a whole episode's release package into
        'Release package failed'."""
        meta = generate_metadata(1, [])
        self.assertTrue(meta["description"])

    def test_disclosure_checklist(self):
        d = generate_disclosure_checklist(has_voice_clone=True, has_ai_video=True)
        self.assertIs(d["synthetic_content"], True)
        self.assertIs(d["voice_consent"], True)

    def test_thumbnail_concepts(self):
        thumbs = generate_thumbnail_concepts(1, TEST_SCENES)
        self.assertEqual(len(thumbs), 3)
        self.assertTrue(all("focal_point" in t for t in thumbs))

    def test_full_build(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "ch001_youtube.mp4"
            out.write_bytes(b"fake video")
            pkg = build_release_package(1, out, TEST_SCENES)
            self.assertEqual(pkg["chapter"], 1)
            self.assertTrue(pkg["video_sha256"])
            self.assertTrue(pkg["credit_note"])
            pkg_dir = Path(td) / "release_package" / "ch001"
            self.assertTrue((pkg_dir / "metadata.json").exists())
            self.assertTrue((pkg_dir / "credit_note.txt").exists())

    def test_package_points_at_the_copy_it_ships(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "ch001_youtube.mp4"
            out.write_bytes(b"fake video")
            pkg = build_release_package(1, out, TEST_SCENES)
            pkg_dir = Path(td) / "release_package" / "ch001"
            self.assertTrue(Path(pkg["video_path"]).is_file(),
                            "video_path must resolve inside the package")
            self.assertIn(str(pkg_dir), pkg["video_path"])

    def test_description_never_asserts_an_unverified_licence(self):
        """rules/assembly.md: 'Do not automatically assert Fesliyan licensing'.
        The old description hardcoded it for every episode and pointed at a
        credits section that was never emitted."""
        unverified = {"music": [{"asset_id": "dark_fog.mp3",
                                 "commercial_use": "unknown"}]}
        for rm in ({}, unverified,
                   {"music": [{"asset_id": "x.mp3", "commercial_use": "yes"}]}):
            d = generate_metadata(1, TEST_SCENES, rights_manifest=rm)["description"]
            self.assertNotIn("Fesliyan Studios (see credits below)", d)
            self.assertIn("Credits", d, "credits section must exist if referenced")
        d = generate_metadata(1, TEST_SCENES, rights_manifest=unverified)["description"]
        self.assertIn("NOT YET VERIFIED", d)

    def test_music_license_state_is_derived(self):
        self.assertEqual(music_license_state({}), "not_used")
        self.assertEqual(music_license_state(None), "not_used")
        self.assertEqual(music_license_state({"music": []}), "not_used")
        # a non-list value is not a music list at all
        self.assertEqual(music_license_state({"music": "Licensed/Fesliyan"}),
                         "not_used")
        self.assertEqual(
            music_license_state({"music": [{"commercial_use": "yes"}]}), "verified")
        self.assertEqual(
            music_license_state({"music": [{"commercial_use": "original"}]}),
            "verified")
        self.assertEqual(
            music_license_state({"music": [{"commercial_use": "unknown"}]}),
            "unverified")
        # ponytail: this case is why the function exists. A manifest whose rows
        # are all non-dicts filtered `licenses` down to set(), and
        # `set() <= {"yes","original"}` is True — so it returned "verified"
        # from a manifest that verified nothing. Verified by mutation.
        self.assertEqual(music_license_state({"music": ["oops-a-string"]}),
                         "unverified")
        self.assertEqual(music_license_state({"music": [None, 42]}),
                         "unverified")
        # one good row among junk still verifies
        self.assertEqual(
            music_license_state({"music": [{"commercial_use": "yes"}, "junk"]}),
            "verified")

    def test_description_refuses_to_certify_unverifiable_music(self):
        """End-to-end: the published copy must not claim a cleared licence when
        the manifest cannot prove one."""
        unverifiable = {"music": ["oops-a-string"]}
        d = generate_metadata(1, TEST_SCENES,
                              rights_manifest=unverifiable)["description"]
        self.assertNotIn("cleared for commercial use", d)
        self.assertIn("NOT YET VERIFIED", d)
        self.assertIn("Do not publish", d)


class AnalyticsLogTests(unittest.TestCase):
    def test_add_entry_and_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            entry = add_analytics(root, 1, impressions=1000, ctr=0.05,
                                  avg_view_duration=45, completion_rate=0.35,
                                  subscribers_gained=2)
            self.assertEqual(entry["chapter"], 1)
            self.assertEqual(len(load_analytics_log(root)), 1)

    def test_summarize(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            add_analytics(root, 1, impressions=1000, ctr=0.05,
                          avg_view_duration=45, completion_rate=0.35)
            s = summarize(load_analytics_log(root))
            self.assertEqual(s["entries"], 1)
            self.assertEqual(s["avg_impressions"], 1000)

    def test_experiment_proposals(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            add_analytics(root, 1, impressions=1000, ctr=0.05,
                          avg_view_duration=45, completion_rate=0.35)
            for i in range(5):
                add_analytics(root, i + 2, impressions=1000, ctr=0.02,
                              avg_view_duration=20, completion_rate=0.2)
            proposals = propose_experiments(load_analytics_log(root))
            self.assertTrue(proposals)
            self.assertTrue(any(
                any(k in p for k in ("CTR", "completion", "hook"))
                for p in proposals))

    def test_save_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            add_analytics(root, 1, impressions=1, ctr=0.1)
            self.assertEqual(list(root.glob("*.tmp")), [])


class ShippedConfigTests(unittest.TestCase):
    def test_voice_speed_is_one(self):
        cfg = json.loads((Path(__file__).parent / "config.json").read_text("utf-8"))
        self.assertEqual(cfg.get("voice_speed"), 1.0,
                         f"voice_speed should be 1.0, got {cfg.get('voice_speed')}")

    def test_ini_tokens_disabled_by_default(self):
        self.assertIs(ini_config.tokens_enabled(), False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
