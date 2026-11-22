import ast
import asyncio
import base64
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pipeline.asset_library import AssetLibrary
from pipeline.project_audit import audit_project, _scripted
from pipeline import script_reviewer
from pipeline.storyboard import generate_storyboard
from pipeline.beat_planner import beats_summary, plan_beats
from reference_selector import ReferenceImageSelector
from best_image_selector import BestImageSelector


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ("scenes", "chapters", "characters", "videos"):
            (self.root / folder).mkdir()
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network"))
        self.urlopen = self.network.start()
        self.addCleanup(self.network.stop)

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def novel(self, novel_id="sinners_world"):
        """A legacy-layout Novel rooted at this test's temp dir.

        The server resolves a novel id to a Novel object; the tests inject
        this instead of the real registry (which points at the live project).
        """
        from pipeline.novels import Novel
        return Novel(id=novel_id or "sinners_world", title="Test Novel",
                     root=self.root, legacy=True)

    def test_script_discovery_excludes_storyboards_and_invalid_roots(self):
        self.write_json("characters/characters.json", {"characters": {"hero": {}}})
        self.write_json("scenes/ch001.json", [{"scene_number": 1, "characters": ["hero"]}])
        for name, value in (
            ("ch002_storyboard.json", [{"scene_number": 1, "characters": ["hero"]}]),
            ("ch003.json", {"shots": []}),
            ("ch004.json", [{"scene_number": 1}, "invalid"]),
            ("ch005.json", [{"characters": ["hero"]}]),
            ("ch006.json", [{"scene_number": None}]),
            ("ch007.json", []),
            ("ch08.json", [{"scene_number": 1}]),
            ("ch0009.json", [{"scene_number": 1}]),
            ("ch010.json", [{"scene_number": True}]),
        ):
            self.write_json("scenes/" + name, value)
        (self.root / "scenes/ch011.json").write_text("{", encoding="utf-8")
        library = AssetLibrary(self.root)
        self.assertEqual(_scripted(self.root), {1})
        self.assertEqual(set(library.load_chapters()), {1})
        self.assertEqual(library.get_chapter(1).scene_count, 1)
        self.assertEqual(library.get_character("hero").appearances, [1])
        self.assertEqual(library.summary()["chapters_scripted"], 1)
        for chapter in (3, 4, 5, 6, 7, 10, 11):
            self.assertEqual(library.load_scenes(chapter), [])

    def test_audit_uses_last_header_even_before_long_traceback(self):
        self.write_json("scenes/ch001.json", [{"scene_number": 1}])
        (self.root / "chapters/ch001.txt").write_text("chapter", encoding="utf-8")
        for header in ("=== 9 2026-09-17 10:00:00 ===", "=== redo 9/2 2026-09-17 10:00:00 ==="):
            with self.subTest(header=header):
                (self.root / "render_crash.log").write_text(
                    "=== 1 2026-09-16 10:00:00 ===\nold traceback\n" + header + "\n" + "trace\n" * 80,
                    encoding="utf-8")
                self.write_json("progress_live.json", {"phase": "error"})
                report = audit_project(self.root)
                self.assertEqual(report.next_chapter, 9)
                self.assertIn("crashed", report.next_reason)
                self.write_json("progress_live.json", {"phase": "done"})
                report = audit_project(self.root)
                self.assertEqual(report.next_chapter, 1)
                self.assertFalse(any("crash" in f.message.lower() for f in report.findings))
                self.assertNotIn("crashed", report.next_reason)

    def test_review_rejects_invalid_roots_without_writing(self):
        with patch.object(script_reviewer, "ROOT", self.root):
            for value in (None, {}, "text", 1, [None], [{}, "bad"], []):
                with self.subTest(value=value):
                    result = script_reviewer.review_script(value, 1, persist=True)
                    self.assertFalse(result.valid)
                    self.assertEqual(result.score, 0)
                    self.assertEqual(result.issues[0].field, "root")
                    self.assertEqual(result.auto_fixes, [])
        self.assertEqual(list((self.root / "scenes").iterdir()), [])

    def test_review_preview_repairs_memory_and_persist_saves_explicitly(self):
        scenes = [{"scene_number": number, "type": "still", "narration": "hello",
                   "characters": "hero, guest"}
                  for number in (None, 0, -1, "4", True, 2.5, [], {})]
        path = self.write_json("scenes/ch001.json", scenes)
        original = path.read_bytes()
        with patch.object(script_reviewer, "ROOT", self.root):
            preview = copy.deepcopy(scenes)
            result = script_reviewer.review_script(preview, 1)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual([s["scene_number"] for s in preview], list(range(1, 9)))
            self.assertEqual(preview[0]["characters"], ["hero", "guest"])
            fixes = [f for f in result.auto_fixes if f["field"] == "scene_number"]
            self.assertEqual([f["new"] for f in fixes], list(range(1, 9)))
            result = script_reviewer.review_script(scenes, 1, persist=True)
            self.assertTrue(result.valid)
            self.assertEqual(json.loads(path.read_text("utf-8")), preview)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            repaired = script_reviewer.auto_repair([{"scene_number": None}])
            self.assertEqual(repaired[0]["scene_number"], 1)

    def test_review_save_failure_preserves_original(self):
        scenes = [{"scene_number": None}]
        path = self.write_json("scenes/ch001.json", scenes)
        original = path.read_bytes()
        # Patch the real write path. This used to patch `script_reviewer.os`,
        # which only worked while that module imported os; the persist path now
        # goes through json_util.dump_json_atomic (fsync + bounded replace
        # retry), so patch THAT. It must be the name as bound in
        # script_reviewer (`from json_util import dump_json_atomic`), because
        # patching json_util's attribute would not affect that local binding.
        with patch.object(script_reviewer, "ROOT", self.root), \
                patch.object(script_reviewer, "dump_json_atomic",
                             side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                script_reviewer.review_script(scenes, 1, persist=True)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_dashboard_review_contract_without_server_import(self):
        path = Path(__file__).with_name("pipeline_server.py")
        tree = ast.parse(path.read_text("utf-8"))
        handler = next(node for node in tree.body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "api_review")
        handler.decorator_list = []
        namespace = {"ROOT": self.root, "json": json, "ReviewReq": object,
                     "Request": object, "asyncio": asyncio,
                     "HTTPException": RuntimeError, "tasks": SimpleNamespace(active=lambda: []),
                     "AssetLibrary": lambda root: SimpleNamespace(load_characters=lambda: {}),
                     "DEFAULT_NOVEL_ID": "sinners_world",
                     "_novel_or_404": self.novel,
                     "review_script": script_reviewer.review_script,
                     "ep": SimpleNamespace(render_alive=lambda: False),
                     "beats_summary": beats_summary}
        exec(compile(ast.Module(body=[handler], type_ignores=[]), str(path), "exec"), namespace)
        with patch.object(script_reviewer, "ROOT", self.root):
            for scenes in ([{"scene_number": None}], {"shots": []}):
                path = self.write_json("scenes/ch001.json", scenes)
                result = namespace["api_review"](None, SimpleNamespace(chapter=1))
                if isinstance(handler, ast.AsyncFunctionDef):
                    result = asyncio.run(result)
                if isinstance(scenes, list):
                    self.assertEqual(json.loads(path.read_text("utf-8"))[0]["scene_number"], 1)
                    self.assertEqual(result["beats"]["status"], "unplanned")
                else:
                    self.assertFalse(result["valid"])
                    self.assertEqual(result["beats"]["status"], "invalid")

    def test_storyboard_partitions_duration_and_honors_default(self):
        scenes = [{"narration": text, "duration": duration}
                  for text, duration in (("fight", 8), ("said", 6.5), ("arrived", 4))]
        scenes.append({"narration": "felt tears"})
        original = copy.deepcopy(scenes)
        board = generate_storyboard(1, scenes, default_duration=9)
        self.assertEqual([sum(p["duration"] for p in shot.panels) for shot in board.shots],
                         [8, 6.5, 4, 9])
        self.assertEqual(board.total_duration_s, 27.5)
        self.assertEqual(board.total_panels, 7)
        self.assertEqual(scenes, original)
        self.assertEqual(generate_storyboard(1, [{"narration": "arrived"}], 11).total_duration_s, 11)
        self.assertEqual(generate_storyboard(1, []).total_duration_s, 0)

    def test_beat_order_endpoints_alias_and_missing_purposes(self):
        for purposes, gaps in (
            (["HOOK", "RISE", "RISE", "TURN", "CLIMAX", "RESOLUTION"], []),
            (["RISE", "TURN", "LAND"], [0]),
            (["HOOK", "RISE", "TURN"], [2]),
            (["HOOK", "TURN", "RISE", "CLIMAX", "LAND"], [2]),
            (["HOOK", "LAND", "CLIMAX", "LAND"], [2]),
            (["HOOK", "CLIMAX", "RISE", "TURN", "LAND"], [2, 3]),
        ):
            with self.subTest(purposes=purposes):
                scenes = [{"purpose": p} for p in purposes]
                original = copy.deepcopy(scenes)
                result = beats_summary(scenes)
                self.assertEqual(result["gaps"], gaps)
                self.assertEqual(result["source"], "purpose")
                self.assertEqual(result["status"], "gaps" if gaps else "planned")
                self.assertEqual(scenes, original)
        scenes = [{}, {}, {}]
        result = plan_beats(scenes)
        self.assertEqual(result.status, "unplanned")
        self.assertEqual(result.source, "heuristic")
        self.assertTrue(all(s["source"] == "heuristic" for s in scenes))
        self.assertEqual(scenes[-1]["beat"], "LAND")
        partial = beats_summary([{"purpose": "HOOK"}, {}, {"purpose": "LAND"}])
        self.assertEqual(partial["source"], "mixed")
        self.assertEqual(partial["status"], "unplanned")
        self.assertEqual(partial["gaps"], [1])
        self.assertEqual(beats_summary([])["status"], "unplanned")

    def response(self, content):
        # json_util.oai_chat opens the response as a context manager (it is a
        # long-lived server; a leaked socket per call is a real leak), so the
        # double has to honour the protocol. __enter__ returns SELF so the
        # configured read() is still the one that runs.
        r = Mock(read=Mock(return_value=json.dumps(
            {"choices": [{"message": {"content": content}}]}).encode()))
        r.__enter__ = Mock(return_value=r)
        r.__exit__ = Mock(return_value=False)
        return r

    def test_oai_vision_payload_shape(self):
        selector = ReferenceImageSelector("http://127.0.0.1:8080", "test")
        with patch.object(selector, "_b64", side_effect=lambda path: path):
            self.urlopen.side_effect = None
            self.urlopen.return_value = self.response('{"best_image_index": 0, "reason": "ok"}')
            selector.select("frame", ["r0", "r1"], ["ref0", "ref1"],
                            ["c0"], ["cand0"], 8)
            payload = json.loads(self.urlopen.call_args.args[0].data)
            user = payload["messages"][1]
            kinds = [p["type"] for p in user["content"]]
            self.assertEqual(kinds, ["text", "image_url", "image_url", "image_url"])
            self.assertTrue(user["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"))
            self.assertNotIn("images", user)

    def test_vlm_availability_gating(self):
        import episode_producer as ep
        with patch.dict(ep.CONFIG, {"llm_vision_enabled": False}):
            with patch.object(ep, "_serving_id", return_value=ep._vlm_id()):
                self.assertFalse(ep.vlm_available())
                self.assertIsNone(ep._pick_vlm())
        with patch.dict(ep.CONFIG, {"llm_vision_enabled": True}):
            with patch.object(ep, "_serving_id", return_value="Qwen3-14B-abliterated-Q4_K_M"):
                self.assertFalse(ep.vlm_available())
            with patch.object(ep, "_serving_id", return_value=ep._vlm_id()):
                self.assertTrue(ep.vlm_available())
                self.assertEqual(ep._pick_vlm(), ep._vlm_id())
            with patch.object(ep, "_serving_id", return_value=""):
                self.assertFalse(ep.vlm_available())

    def test_reference_filter_validates_json_indices_and_caps(self):
        selector = ReferenceImageSelector("http://127.0.0.1:8080", "test")
        pairs = [(str(i), str(i)) for i in range(5)]
        for content, expected in (
            ('{"indices": [3, 1, 2]}', [pairs[3], pairs[1]]),
            ('```json\n{"indices": [true, -1, 9, "2", 2, 2, 0]}\n```', [pairs[2], pairs[0]]),
            ('{"indices": "bad"}', pairs[:2]),
            ('{"indices": []}', pairs[:2]),
            ('not JSON', pairs[:2]),
            ('{"indices": [}', pairs[:2]),
        ):
            with self.subTest(content=content):
                self.urlopen.side_effect = None
                self.urlopen.return_value = self.response(content)
                self.assertEqual(selector._filter_by_text(pairs, "frame", 2), expected)
        self.assertEqual(selector._filter_by_text(pairs, "frame", 0), [])

    def test_reference_filter_failure_still_caps_attachments(self):
        selector = ReferenceImageSelector("http://127.0.0.1:8080", "test")
        for cap in (0, 2, 8):
            with self.subTest(cap=cap), \
                    patch.object(selector, "_filter_by_text", side_effect=OSError("offline")), \
                    patch.object(selector, "_b64", side_effect=lambda path: path):
                self.urlopen.side_effect = None
                self.urlopen.return_value = self.response('{"best_image_index": 0}')
                selector.select("frame", ["r"] * 10, ["ref"] * 10, ["candidate"], ["text"], cap)
                payload = json.loads(self.urlopen.call_args.args[0].data)
                n_images = sum(1 for p in payload["messages"][1]["content"]
                               if p["type"] == "image_url")
                self.assertEqual(n_images, cap + 1)

    def test_best_image_single_candidate_skips_io_and_vlm(self):
        selector = BestImageSelector("http://127.0.0.1:8080", "test")
        with patch.object(selector, "_b64") as encode:
            self.assertEqual(selector.select("missing", "ref", "frame", ["missing"], []),
                             (0, "single-candidate", {}))
            self.assertEqual(selector.select("missing", "ref", "frame", [], []),
                             (0, "no candidates", {}))
            encode.assert_not_called()
            self.urlopen.assert_not_called()

    def _server_ns(self, **extra):
        import episode_producer as ep
        path = Path(__file__).with_name("pipeline_server.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        want = {"api_scenes_status", "api_workbench", "api_scene_update",
                "_read_pins", "_live_render", "_canonical_videos"}
        nodes = [n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 and getattr(n, "name", "") in want]
        for n in nodes:
            if hasattr(n, "decorator_list"):
                n.decorator_list = []
        ns = {"ROOT": self.root, "json": json, "os": __import__("os"),
              "re": __import__("re"), "HTTPException": RuntimeError,
              "tasks": SimpleNamespace(active=lambda: []),
              "ep": ep, "SceneUpdateReq": SimpleNamespace,
              "PINS_FILE": self.root / "pinned.json",
              "DEFAULT_NOVEL_ID": "sinners_world",
              "_novel_or_404": self.novel,
              "AssetLibrary": AssetLibrary,
              "SCENE_EDITABLE": {"narration", "dialogue", "image_prompt", "shot_size",
                                 "camera_move", "mood", "angle", "animation_hint",
                                 "animation_style"}}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
        ns.update(extra)
        return ns

    def test_animation_style_presets_wired(self):
        import episode_producer as ep
        self.assertEqual(sorted(ep.ANIMATION_STYLES),
                         ["cinematic", "gothic", "noir", "painterly", "retro-cel", "soft"])
        with patch.dict(ep.CONFIG, {"animation_style": "cinematic"}):
            self.assertEqual(ep._animation_style({}), "cinematic")
        self.assertEqual(ep._animation_style({"animation_style": "noir"}), "noir")
        self.assertEqual(ep._animation_style({"animation_style": "bogus"}), "cinematic")
        src = Path(__file__).with_name("pipeline_server.py").read_text(encoding="utf-8")
        self.assertIn('"animation_style"', src)

    def test_workbench_combines_cards_and_blocks_live_edit(self):
        import episode_producer as ep
        scenes = [{"scene_number": 1, "narration": "hello",
                   "image_prompt": "a street", "shot_size": "wide",
                   "camera_move": "", "mood": "tense", "angle": "eye-level",
                   "animation_hint": "", "characters": []}]
        self.write_json("scenes/ch001.json", scenes)
        (self.root / "images").mkdir(exist_ok=True)
        (self.root / "voice").mkdir(exist_ok=True)
        (self.root / "images" / "ch001_s01_x.png").write_bytes(b"still")
        ns = self._server_ns()
        wb = asyncio.run(ns["api_workbench"](chapter=1))
        self.assertEqual(len(wb["scenes"]), 1)
        self.assertEqual(wb["scenes"][0]["still"], "ch001_s01_x.png")
        self.assertIn("cinematic", wb["styles"])
        self.assertFalse(wb["scenes"][0]["pinned"])
        (self.root / "pinned.json").write_text(
            json.dumps({"ch001": {"1": {"still": "", "note": "t"}}}), encoding="utf-8")
        wb = asyncio.run(ns["api_workbench"](chapter=1))
        self.assertTrue(wb["scenes"][0]["pinned"])
        upd = asyncio.run(ns["api_scene_update"](
            SimpleNamespace(chapter=1, scene=1,
                            fields={"narration": "new line", "mood": "somber"})))
        self.assertTrue(upd["updated"])
        saved = json.loads((self.root / "scenes/ch001.json").read_text(encoding="utf-8"))
        self.assertEqual(saved[0]["narration"], "new line")
        with self.assertRaises(RuntimeError):
            asyncio.run(ns["api_scene_update"](
                SimpleNamespace(chapter=1, scene=1, fields={"checkpoint": "x"})))
        ns_live = self._server_ns()
        ns_live["tasks"] = SimpleNamespace(active=lambda: [object()])
        tree = ast.parse(Path(__file__).with_name("pipeline_server.py").read_text(encoding="utf-8"))
        handler = next(n for n in tree.body if getattr(n, "name", "") == "api_scene_update")
        handler.decorator_list = []
        ns2 = {"ROOT": self.root, "json": json, "os": __import__("os"),
               "re": __import__("re"), "HTTPException": RuntimeError,
               "tasks": ns_live["tasks"], "ep": ep, "SceneUpdateReq": SimpleNamespace,
               "SCENE_EDITABLE": ns["SCENE_EDITABLE"],
               "DEFAULT_NOVEL_ID": "sinners_world",
               "_novel_or_404": self.novel,
               "_live_render": lambda: True}
        exec(compile(ast.Module(body=[handler], type_ignores=[]), "x", "exec"), ns2)
        with self.assertRaises(RuntimeError):
            asyncio.run(ns2["api_scene_update"](
                SimpleNamespace(chapter=1, scene=1, fields={"mood": "tense"})))

    def test_api_approve_rights_and_director(self):
        from pipeline.rights_manifest import (
            load_rights_manifest, save_rights_manifest, preflight_rights_gate, _empty_manifest, _commercial_status, _is_non_commercial)
        from pipeline.director_package import load_director_package, save_director_package
        path = Path(__file__).with_name("pipeline_server.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and getattr(n, "name", "") in {"api_approve", "_approve_rights_manifest"}]
        self.assertTrue(nodes, "api_approve not found")
        for n in nodes:
            n.decorator_list = []
        save_rights_manifest(_empty_manifest(), self.root)
        ns = {"ROOT": self.root, "time": __import__("time"),
              "Request": object, "ApproveReq": SimpleNamespace,
              "_commercial_status": _commercial_status,
              "_is_non_commercial": _is_non_commercial,
              "DEFAULT_NOVEL_ID": "sinners_world",
              "_novel_or_404": self.novel,
              "load_rights_manifest": load_rights_manifest,
              "save_rights_manifest": save_rights_manifest,
              "preflight_rights_gate": preflight_rights_gate,
              "load_director_package": load_director_package,
              "save_director_package": save_director_package}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
        r = asyncio.run(ns["api_approve"](None, SimpleNamespace(item="rights", chapter=1)))
        self.assertTrue(r.get("ok"), r)
        m = load_rights_manifest(self.root)
        self.assertTrue(m["novel_adaptation_rights"]["owner_confirmation"])
        self.assertEqual(m["novel_adaptation_rights"]["commercial_use"], "yes")
        self.assertTrue(m["voice"]["owner_confirmation"])
        self.assertEqual(m["voice"]["commercial_use"], "yes")
        save_director_package({"version": "1.0"}, self.root / "scenes", 1, force=True)
        r2 = asyncio.run(ns["api_approve"](None, SimpleNamespace(item="director", chapter=1)))
        self.assertTrue(r2.get("ok"), r2)
        pkg = load_director_package(self.root / "scenes", 1)
        self.assertTrue(pkg["owner_approved"])
        r3 = asyncio.run(ns["api_approve"](None, SimpleNamespace(item="director", chapter=99)))
        self.assertFalse(r3.get("ok"))

    def test_api_approve_all(self):
        from test_task_manager import server_functions
        from pipeline.preflight import check_rights, check_director_package
        from pipeline.rights_manifest import (save_rights_manifest, _empty_manifest,
                                             load_rights_manifest, _commercial_status,
                                             _is_non_commercial,
                                             load_rights_manifest as _lr)
        from pipeline.director_package import save_director_package, load_director_package as _ldp
        save_rights_manifest(_empty_manifest(), self.root)
        (self.root / "chapters" / "Chapter_1.txt").write_text("Once upon a time.", encoding="utf-8")
        self.write_json("scenes/ch001_director.json", {"version": "1.0", "owner_approved": False})
        self.write_json("scenes/ch002_director.json", {"version": "1.0", "owner_approved": False})
        ns = server_functions(
            "api_approve_all", "_approve_rights_manifest",
            ROOT=self.root,
            time=__import__("time"),
            Request=SimpleNamespace, ApproveAllReq=SimpleNamespace,
            ep=SimpleNamespace(preflight=lambda ch, novel=None: True),
            DEFAULT_NOVEL_ID="sinners_world",
            _novel_or_404=self.novel,
            _commercial_status=_commercial_status,
            _is_non_commercial=_is_non_commercial,
            load_rights_manifest=_lr,
            save_rights_manifest=save_rights_manifest,
            load_director_package=_ldp,
            save_director_package=save_director_package,
            check_rights=check_rights,
            check_director_package=check_director_package)
        r = asyncio.run(ns["api_approve_all"](None, SimpleNamespace(chapters=[1,2])))
        self.assertTrue(r["ok"])
        self.assertEqual(len(r["approved_director"]), 2)
        m = load_rights_manifest(self.root)
        self.assertTrue(m["novel_adaptation_rights"]["owner_confirmation"])
        pkg1 = ns["load_director_package"](self.root / "scenes", 1)
        self.assertTrue(pkg1["owner_approved"])
        pkg2 = ns["load_director_package"](self.root / "scenes", 2)
        self.assertTrue(pkg2["owner_approved"])

    def test_api_preflight_approvals_contract(self):
        from test_task_manager import server_functions
        from pipeline.preflight import check_rights, check_director_package
        from pipeline.rights_manifest import save_rights_manifest, _empty_manifest
        save_rights_manifest(_empty_manifest(), self.root)
        (self.root / "chapters" / "Chapter_1.txt").write_text("Once upon a time.", encoding="utf-8")
        self.write_json("scenes/ch001_director.json",
                        {"version": "1.0", "owner_approved": False})
        ns = server_functions(
            "api_preflight", "_approval_hints", "_preflight_reason",
            ROOT=self.root,
            ep=SimpleNamespace(preflight=lambda ch, novel=None: True),
            DEFAULT_NOVEL_ID="sinners_world",
            _novel_or_404=self.novel,
            check_rights=check_rights,
            check_director_package=check_director_package)
        r = asyncio.run(ns["api_preflight"](SimpleNamespace(), chapter=1))
        self.assertFalse(r["ready"])
        self.assertFalse(r["rights_ok"])
        self.assertFalse(r["director_ok"])
        items = {a["item"]: a for a in r["approvals"]}
        self.assertFalse(items["rights"]["ok"])
        self.assertIn("blockers", items["rights"])
        self.assertIn("exists", items["director"])
        self.assertTrue(r["director_ok"] is False)
        ns["ep"] = SimpleNamespace(preflight=lambda ch, novel=None: False)
        r2 = asyncio.run(ns["api_preflight"](SimpleNamespace(), chapter=1))
        self.assertTrue(any("approve below" in h for h in r2["hints"]), r2["hints"])

    def test_best_image_attaches_and_labels_every_candidate_in_order(self):
        paths = []
        for name in ("reference", "candidate0", "candidate1"):
            path = self.root / name
            path.write_bytes(name.encode())
            paths.append(str(path))
        selector = BestImageSelector("http://127.0.0.1:8080", "test")
        self.urlopen.side_effect = None
        self.urlopen.return_value = self.response('{"best_image_index": 1, "reason": "matched", "composition": 90}')
        result = selector.select(paths[0], "reference", "frame", paths[1:], ["first"])
        self.assertEqual(result, (1, "matched", {"composition": 90}))
        payload = json.loads(self.urlopen.call_args.args[0].data)
        message = payload["messages"][1]
        parts = message["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(
            [base64.b64decode(p["image_url"]["url"].split(",", 1)[1])
             for p in parts[1:]],
            [b"reference", b"candidate0", b"candidate1"])
        labels = parts[0]["text"]
        self.assertLess(labels.index("Reference Image:"), labels.index("Candidate Image 0: first"))
        self.assertLess(labels.index("Candidate Image 0: first"), labels.index("Candidate Image 1: candidate1"))
        self.assertEqual(self.urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
