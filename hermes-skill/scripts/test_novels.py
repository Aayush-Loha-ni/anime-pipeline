"""Multi-novel support: one folder per novel, isolated assets, shared engine.

A novel owns its chapters, scenes, images, voice, videos, characters, dataset
and pins. The engine (models, ComfyUI, the voice clone, the rules) stays
shared. The built-in "sinners_world" novel keeps the legacy layout at the
project root, so nothing existing moved.
"""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import episode_producer as ep
from pipeline.novels import (DEFAULT_NOVEL_ID, Novel, create_novel, get_novel,
                              list_novels, slugify)


class NovelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_legacy_novel_is_always_listed_first(self):
        novels = list_novels(self.root)
        self.assertEqual(novels[0].id, DEFAULT_NOVEL_ID)
        self.assertTrue(novels[0].legacy)
        self.assertEqual(novels[0].root, self.root)

    def test_create_novel_makes_the_folder_tree_and_registers_it(self):
        novel = create_novel("My Novel", root=self.root)
        for folder in ("chapters", "scenes", "images", "voice", "videos",
                       "characters", "dataset", "music", "sfx"):
            self.assertTrue((novel.root / folder).is_dir(), folder)
        self.assertTrue((novel.root / "novel.json").is_file())
        self.assertTrue((novel.root / "characters" / "characters.json").is_file())
        self.assertTrue((novel.root / "rights_manifest.json").is_file())
        self.assertEqual([n.id for n in list_novels(self.root)],
                         [DEFAULT_NOVEL_ID, novel.id])
        self.assertEqual(get_novel(novel.id, root=self.root).title, "My Novel")

    def test_two_novels_may_both_ship_chapter_one(self):
        a = create_novel("Alpha", root=self.root)
        b = create_novel("Beta", root=self.root)
        (a.chapters_dir() / "Chapter_1.txt").write_text("alpha", encoding="utf-8")
        (b.chapters_dir() / "Chapter_1.txt").write_text("beta", encoding="utf-8")
        self.assertEqual((a.chapters_dir() / "Chapter_1.txt").read_text(encoding="utf-8"),
                         "alpha")
        self.assertEqual((b.chapters_dir() / "Chapter_1.txt").read_text(encoding="utf-8"),
                         "beta")

    def test_duplicate_titles_get_distinct_folders(self):
        a = create_novel("Same", root=self.root)
        b = create_novel("Same", root=self.root)
        self.assertNotEqual(a.id, b.id)
        self.assertNotEqual(a.root, b.root)

    def test_unknown_novel_is_not_resolved(self):
        self.assertIsNone(get_novel("nope", root=self.root))

    def test_slugify_keeps_ids_filesystem_safe(self):
        self.assertEqual(slugify("Example Novel!"), "sinners_world")
        self.assertEqual(slugify("   "), "novel")

    def test_index_survives_a_corrupt_file(self):
        (self.root / "novels").mkdir()
        (self.root / "novels" / "index.json").write_text("{ not json",
                                                          encoding="utf-8")
        self.assertEqual([n.id for n in list_novels(self.root)], [DEFAULT_NOVEL_ID])


class ProducerNovelContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.novel = Novel(id="alpha", title="Alpha", root=self.root / "alpha")
        for folder in ("chapters", "scenes", "images", "voice", "videos",
                       "characters"):
            (self.novel.root / folder).mkdir(parents=True, exist_ok=True)

    def test_context_rewrites_the_paths_and_restores_them(self):
        old_cfg, old_bible = ep.CONFIG, ep.CHAR_BIBLE
        with patch.object(ep, "get_novel", return_value=self.novel):
            with ep._novel_context("alpha"):
                self.assertEqual(ep.CONFIG["scenes_dir"], str(self.novel.scenes_dir()))
                self.assertEqual(ep.CONFIG["images_dir"], str(self.novel.images_dir()))
                self.assertEqual(ep.CONFIG["chapters_dir"], str(self.novel.chapters_dir()))
                self.assertEqual(ep.CONFIG["output_dir"], str(self.novel.videos_dir()))
                self.assertEqual(ep.CONFIG["characters_file"],
                                 str(self.novel.characters_file()))
                self.assertEqual(ep._asset("voice"), self.novel.root / "voice")
                self.assertEqual(ep._asset("pinned.json"),
                                 self.novel.root / "pinned.json")
                self.assertEqual(ep.current_novel_id(), "alpha")
            self.assertIs(ep.CONFIG, old_cfg)
            self.assertIs(ep.CHAR_BIBLE, old_bible)
            self.assertIsNone(ep.active_novel())

    def test_legacy_novel_leaves_the_config_alone(self):
        old_cfg = ep.CONFIG
        legacy = Novel(id=DEFAULT_NOVEL_ID, title="Example Novel",
                       root=self.root, legacy=True)
        with patch.object(ep, "get_novel", return_value=legacy):
            with ep._novel_context(DEFAULT_NOVEL_ID):
                self.assertIs(ep.CONFIG, old_cfg)
                self.assertEqual(ep._asset("voice"), ep.ROOT / "voice")

    def test_unknown_novel_raises(self):
        with patch.object(ep, "get_novel", return_value=None):
            with self.assertRaises(ValueError):
                with ep._novel_context("nope"):
                    pass

    def test_nested_context_reuses_the_active_novel(self):
        with patch.object(ep, "get_novel", return_value=self.novel):
            with ep._novel_context("alpha"):
                with ep._novel_context(None):
                    self.assertEqual(ep.current_novel_id(), "alpha")
                    self.assertEqual(ep.CONFIG["scenes_dir"],
                                     str(self.novel.scenes_dir()))

    def test_chapter_lookup_is_scoped_to_the_active_novel(self):
        (self.novel.chapters_dir() / "Chapter_1.txt").write_text("alpha one",
                                                                  encoding="utf-8")
        other = self.root / "other"
        (other / "chapters").mkdir(parents=True)
        (other / "chapters" / "Chapter_1.txt").write_text("other one",
                                                           encoding="utf-8")
        with patch.object(ep, "get_novel", return_value=self.novel):
            with ep._novel_context("alpha"):
                found = ep.find_chapter_file(1)
                self.assertEqual(found.read_text(encoding="utf-8"), "alpha one")
                self.assertEqual(
                    ep.find_chapter_file(1, other / "chapters").read_text(encoding="utf-8"),
                    "other one")
                self.assertIsNone(ep.find_chapter_file(99))


class ServerNovelRoutingTests(unittest.TestCase):
    """The server's chapter list must be scoped to one novel."""

    def _server_helpers(self, root):
        path = Path(__file__).with_name("pipeline_server.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body
                 if isinstance(n, ast.FunctionDef)
                 and n.name in {"_chapter_list", "_novel_or_404"}]
        ns = {"re": __import__("re"),
              "HTTPException": __import__("fastapi").HTTPException,
              "get_novel": lambda nid=None: get_novel(nid, root=root)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
        return ns

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_each_novel_reports_only_its_own_chapters(self):
        a = create_novel("Alpha", root=self.root)
        b = create_novel("Beta", root=self.root)
        (a.chapters_dir() / "Chapter_1.txt").write_text("a", encoding="utf-8")
        (b.chapters_dir() / "Chapter_2.txt").write_text("b", encoding="utf-8")
        ns = self._server_helpers(self.root)
        listing = ns["_chapter_list"]
        self.assertEqual([c["chapter"] for c in listing(a)], [1])
        self.assertEqual([c["chapter"] for c in listing(b)], [2])
        self.assertEqual(listing(a)[0]["novel"], a.id)

    def test_unknown_novel_raises(self):
        from fastapi import HTTPException
        ns = self._server_helpers(self.root)
        with self.assertRaises(HTTPException):
            ns["_novel_or_404"]("nope")
        self.assertEqual(ns["_novel_or_404"](DEFAULT_NOVEL_ID).root, self.root)


    def test_concurrent_preflight_does_not_disturb_a_running_render(self):
        """The dashboard may run preflight on an API thread WHILE a render
        runs in another thread. The render's CONFIG must survive it."""
        import threading
        novel_a = Novel(id="alpha", title="Alpha", root=self.root / "alpha")
        seen = {}
        started = threading.Event()
        release = threading.Event()

        def render():
            with patch.object(ep, "get_novel", return_value=novel_a):
                with ep._novel_context("alpha"):
                    seen["before"] = ep.CONFIG["scenes_dir"]
                    started.set()
                    release.wait(5)
                    seen["after"] = ep.CONFIG["scenes_dir"]

        t = threading.Thread(target=render)
        t.start()
        self.assertTrue(started.wait(5))
        try:
            # a preflight for the same novel, on THIS thread
            with patch.object(ep, "get_novel", return_value=novel_a):
                with ep._novel_context("alpha", swap_config=False):
                    self.assertEqual(ep._cfg_path("scenes_dir"),
                                     novel_a.scenes_dir())
        finally:
            release.set()
            t.join(5)
        self.assertEqual(seen["before"], str(novel_a.scenes_dir()))
        self.assertEqual(seen["after"], str(novel_a.scenes_dir()))


class NovelArtDirectionTests(unittest.TestCase):
    """A novel's art direction reaches the prompt (and the still cache key)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.novel = Novel(id="alpha", title="Alpha", root=self.root / "alpha",
                            art_direction={"style": "noir",
                                           "still": "ink and wash, heavy blacks",
                                           "negative": "pastel"})
        self.scene = {"scene_number": 1, "narration": "x"}

    def test_art_direction_is_appended_to_the_style(self):
        with patch.object(ep, "get_novel", return_value=self.novel):
            with ep._novel_context("alpha"):
                spec = ep._style_spec(self.scene)
                self.assertIn("ink and wash", spec["still"])
                self.assertIn("pastel", spec["negative"])
                self.assertEqual(ep.CONFIG["animation_style"], "noir")

    def test_legacy_novel_keeps_its_prompt_and_its_cached_stills(self):
        outside = ep._visual_fingerprint(self.scene)["hash"]
        legacy = Novel(id=DEFAULT_NOVEL_ID, title="Example Novel",
                       root=self.root, legacy=True)
        with patch.object(ep, "get_novel", return_value=legacy):
            with ep._novel_context(DEFAULT_NOVEL_ID):
                self.assertEqual(ep._novel_art(), {})
                self.assertEqual(ep._style_spec(self.scene),
                                 ep.ANIMATION_STYLES[ep._animation_style(self.scene)])
                # the fingerprint must not move: the cached stills stay valid
                self.assertEqual(ep._visual_fingerprint(self.scene)["hash"],
                                 outside)

    def test_art_direction_changes_the_still_cache_key(self):
        with patch.object(ep, "get_novel", return_value=self.novel):
            with ep._novel_context("alpha"):
                with_art = ep._visual_fingerprint(self.scene)["hash"]
        plain = Novel(id="beta", title="Beta", root=self.root / "beta")
        with patch.object(ep, "get_novel", return_value=plain):
            with ep._novel_context("beta"):
                without_art = ep._visual_fingerprint(self.scene)["hash"]
        self.assertNotEqual(with_art, without_art)


class CharacterIdentityTests(unittest.TestCase):
    """The bible's tags are the identity source; a LoRA is a stronger one."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.novel = Novel(id="alpha", title="Alpha", root=self.root / "alpha")
        (self.novel.root / "characters").mkdir(parents=True)

    def _context(self, bible=None):
        """Bind the novel, with its own character bible (the novel's file)."""
        if bible is not None:
            (self.novel.root / "characters" / "characters.json").write_text(
                json.dumps(bible), encoding="utf-8")
        return patch.object(ep, "get_novel", return_value=self.novel)

    def test_bible_tags_satisfy_the_identity_gate(self):
        bible = {"characters": {"hero": {"danbooru_tags": "1boy, dark coat"}}}
        with self._context(bible):
            with ep._novel_context("alpha"):
                self.assertTrue(ep._character_identity_ready("hero"))
                self.assertTrue(ep._character_identity_ready("Hero"))
                self.assertFalse(ep._character_identity_ready("nobody"))
                self.assertFalse(ep._character_identity_ready(""))

    def test_a_bible_entry_with_no_tags_is_not_ready(self):
        bible = {"characters": {"hero": {"role": "merchant"}}}
        with self._context(bible):
            with ep._novel_context("alpha"):
                self.assertFalse(ep._character_identity_ready("hero"))

    def test_prompt_tags_alone_are_enough(self):
        bible = {"characters": {"hero": {"prompt_tags": "tall, dark coat"}}}
        with self._context(bible):
            with ep._novel_context("alpha"):
                self.assertTrue(ep._character_identity_ready("hero"))


    def test_a_novel_can_turn_lora_training_off(self):
        """A new novel's first render must not block for hours on LoRA
        training, so the novel can opt out with one config line."""
        novel = Novel(id="alpha", title="Alpha", root=self.root / "alpha",
                      config={"lora_training": {"enabled": False}})
        with patch.object(ep, "get_novel", return_value=novel):
            with ep._novel_context("alpha"):
                self.assertFalse(ep.CONFIG["lora_training"]["enabled"])
                # the rest of the training settings survive the one-level merge
                self.assertIn("steps", ep.CONFIG["lora_training"])

    def test_a_novel_cannot_redirect_its_own_folders(self):
        novel = Novel(id="alpha", title="Alpha", root=self.root / "alpha",
                      config={"output_dir": str(self.root / "elsewhere")})
        with patch.object(ep, "get_novel", return_value=novel):
            with ep._novel_context("alpha"):
                self.assertEqual(ep.CONFIG["output_dir"],
                                 str(self.root / "alpha" / "videos"))


class CharacterCardColdStartTests(unittest.TestCase):
    """A character with no training images gets a card, which is the LoRA's
    training set. The card is never a render input."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.novel = Novel(id="alpha", title="Alpha", root=self.root / "alpha")
        (self.novel.root / "dataset").mkdir(parents=True)

    def _context(self):
        return patch.object(ep, "get_novel", return_value=self.novel)

    def test_no_training_images_makes_a_card_then_trains(self):
        with self._context(), \
                patch.object(ep, "_lora_installed", return_value=False), \
                patch.object(ep, "_stage_character_card", return_value=True) as stage, \
                patch.object(ep, "train_character_lora") as train:
            with ep._novel_context("alpha"):
                ep.ensure_cast_loras(["new_guy"])
        stage.assert_called_once_with("new_guy")
        train.assert_called_once_with("new_guy")

    def test_staged_images_skip_the_card(self):
        ds = self.novel.root / "dataset" / "known_guy_lora"
        ds.mkdir(parents=True)
        for i in range(3):
            (ds / f"known_guy_{i}.png").write_bytes(b"png")
        with self._context(), \
                patch.object(ep, "_lora_installed", return_value=False), \
                patch.object(ep, "_stage_character_card") as stage, \
                patch.object(ep, "train_character_lora") as train:
            with ep._novel_context("alpha"):
                ep.ensure_cast_loras(["known_guy"])
        stage.assert_not_called()
        train.assert_called_once_with("known_guy")

    def test_an_installed_lora_skips_both(self):
        with self._context(), \
                patch.object(ep, "_lora_installed", return_value=True), \
                patch.object(ep, "_stage_character_card") as stage, \
                patch.object(ep, "train_character_lora") as train:
            with ep._novel_context("alpha"):
                ep.ensure_cast_loras(["known_guy"])
        stage.assert_not_called()
        train.assert_not_called()


class MarkdownChapterTests(unittest.TestCase):
    """Novels arrive as .txt or as a .md download. Both must be found."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.novel = Novel(id="alpha", title="Alpha", root=self.root / "alpha")
        self.novel.chapters_dir().mkdir(parents=True)

    def test_a_markdown_chapter_is_found(self):
        md = self.novel.chapters_dir() / "Example Novel_001_Chapter 1_ The Joker.md"
        md.write_text("Chapter 1: The Joker\n", encoding="utf-8")
        with patch.object(ep, "get_novel", return_value=self.novel):
            with ep._novel_context("alpha"):
                found = ep.find_chapter_file(1)
                self.assertIsNotNone(found)
                self.assertEqual(found.name, md.name)
                self.assertEqual(self.novel.chapter_file(1).name, md.name)

    def test_a_text_chapter_is_still_found(self):
        txt = self.novel.chapters_dir() / "Chapter_1.txt"
        txt.write_text("Chapter one\n", encoding="utf-8")
        with patch.object(ep, "get_novel", return_value=self.novel):
            with ep._novel_context("alpha"):
                self.assertEqual(ep.find_chapter_file(1).name, txt.name)

    def test_the_number_comes_from_the_name_not_the_order(self):
        made = {}
        for i in (1, 2, 3):
            name = f"Vol_ {i:03d} _ Chapter {i}.md"
            (self.novel.chapters_dir() / name).write_text("x", encoding="utf-8")
            made[i] = name
        self.assertEqual(self.novel.chapter_file(3).name, made[3])
        self.assertIsNone(self.novel.chapter_file(9))


class FanWorkRightsTests(unittest.TestCase):
    """A fan work of someone else's novel: it may RENDER, never PUBLISH."""

    def _manifest(self, commercial_use):
        return {
            "novel_adaptation_rights": {
                "asset_id": "novel_adaptation", "asset_type": "adaptation_rights",
                "source": "web novel (fan work)", "license": "fan work",
                "commercial_use": commercial_use, "local_path_or_hash": "",
                "owner_confirmation": True, "reviewer": "owner",
                "reviewed_at": "2026-11-12", "notes": "",
            },
            "voice": {"asset_id": "voice_narrator", "asset_type": "voice_clone",
                      "source": "owner", "license": "original_voice",
                      "commercial_use": "yes", "local_path_or_hash": "",
                      "owner_confirmation": True, "reviewer": "owner",
                      "reviewed_at": "2026-11-12", "notes": ""},
            "image_models": [], "video_models": [], "llm_models": [],
            "vlm_models": [], "loras": [], "music": [], "sfx": [],
            "character_references": [], "entries": [],
        }

    def test_a_non_commercial_fan_work_may_render(self):
        from pipeline.rights_manifest import preflight_rights_gate
        ok, blockers, warnings = preflight_rights_gate(
            self._manifest("non-commercial"), for_publish=False)
        self.assertTrue(ok, blockers)
        self.assertTrue(any("non-commercial" in w for w in warnings),
                        warnings)

    def test_a_non_commercial_fan_work_may_never_publish(self):
        from pipeline.rights_manifest import preflight_rights_gate
        ok, blockers, _ = preflight_rights_gate(
            self._manifest("non-commercial"), for_publish=True)
        self.assertFalse(ok)
        self.assertTrue(any("cannot be published" in b for b in blockers), blockers)

    def test_an_unconfirmed_rights_still_blocks_the_render(self):
        from pipeline.rights_manifest import preflight_rights_gate
        m = self._manifest("non-commercial")
        m["novel_adaptation_rights"]["owner_confirmation"] = False
        ok, blockers, _ = preflight_rights_gate(m, for_publish=False)
        self.assertFalse(ok)
        self.assertTrue(any("not confirmed by owner" in b for b in blockers), blockers)

    def test_an_original_work_still_publishes(self):
        from pipeline.rights_manifest import preflight_rights_gate
        ok, blockers, _ = preflight_rights_gate(
            self._manifest("yes"), for_publish=True)
        self.assertTrue(ok, blockers)

    def test_approving_a_fan_work_does_not_claim_commercial_rights(self):
        import ast
        from pathlib import Path as _P
        src = (_P(__file__).with_name("pipeline_server.py")).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_approve_rights_manifest")
        body = ast.get_source_segment(src, fn)
        self.assertIn("_is_non_commercial", body,
                      "approving a fan work must not flip it to commercial")


class NovelSettingResolutionTests(unittest.TestCase):
    """A novel's still prompts use ITS settings, not the legacy novel's."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.novel = Novel(id="alpha", title="Alpha", root=self.root / "alpha")
        (self.novel.root / "characters").mkdir(parents=True)
        bible = {"characters": {}, "settings": {
            "crimson_moon": {"keywords": ["crimson moon"], "visual_tags": "red moon"},
            "gray_city": {"keywords": ["gray street"], "visual_tags": "gray town"}}}
        (self.novel.root / "characters" / "characters.json").write_text(
            json.dumps(bible), encoding="utf-8")

    def test_a_non_legacy_novel_uses_its_own_settings(self):
        with patch.object(ep, "get_novel", return_value=self.novel):
            with ep._novel_context("alpha"):
                self.assertEqual(ep._setting_key_for("the crimson moon hangs over it"),
                                  "crimson_moon")
                self.assertEqual(ep._setting_key_for("a gray street at dusk"),
                                  "gray_city")
                # the legacy novel's hardcoded cues must not leak in
                self.assertNotEqual(ep._setting_key_for("a wide street"),
                                     "white_sky_city")

    def test_the_legacy_novel_keeps_its_hardcoded_cues(self):
        if "white_sky_city" not in (ep.CHAR_BIBLE.get("settings") or {}):
            self.skipTest("not running against the legacy project bible")
        self.assertEqual(ep._setting_key_for("he walked along the street"),
                         "white_sky_city")


class NovelScopedTaskTests(unittest.TestCase):
    def test_tasks_carry_their_novel(self):
        import threading
        from pipeline.task_manager import TaskManager
        manager = TaskManager()
        release = threading.Event()
        task = manager.submit("render", 1, lambda t: release.wait(3), novel="alpha")
        try:
            self.assertEqual(task.novel, "alpha")
            self.assertEqual(task.to_dict()["novel"], "alpha")
            self.assertIsNone(manager.active_for_chapter(1, "beta"))
            self.assertIsNotNone(manager.active_for_chapter(1, "alpha"))
        finally:
            release.set()
            task._thread.join(3)
