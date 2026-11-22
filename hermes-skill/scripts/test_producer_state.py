import io
import math
import numpy as np
import json
import wave
import multiprocessing
import os
import re
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import episode_producer as ep


TEMP_ROOT = Path(tempfile.gettempdir())


def attempt_lock(root, queue):
    ep.ROOT = Path(root)
    try:
        with ep._render_ownership():
            queue.put(True)
    except (OSError, RuntimeError):
        queue.put(False)


class ProducerStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(ep.CONFIG)
        for key, folder in (("images_dir", "images"), ("output_dir", "videos"),
                            ("chapters_dir", "chapters"), ("scenes_dir", "scenes")):
            path = self.root / folder
            path.mkdir()
            self.config[key] = str(path)
        for folder in ("voice", "_temp", "cosy-server"):
            (self.root / folder).mkdir()
        self.ref = self.root / "reference.wav"
        self.ref.write_bytes(b"reference one")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b"\x00\x04" * 16000)
        self.mock_wav = buf.getvalue()
        (self.root / "cosy-server" / "ref.json").write_text(
            json.dumps({"wav": str(self.ref), "text": "reference words"}), encoding="utf-8")
        for name, value in (("ROOT", self.root), ("CONFIG", self.config),
                            ("LIVE_PROGRESS", self.root / "progress_live.json"),
                            ("STOP_FILE", self.root / "stop.requested"),
                            ("RENDER_PID", self.root / "render.pid")):
            patcher = patch.object(ep, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ("_keep_awake_on", "_keep_awake_off"):
            patcher = patch.object(ep, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_tts_speed_and_reference_content_invalidate_cache(self):
        output = self.root / "voice" / "ch001_s01.wav"
        response = Mock(status_code=200, content=self.mock_wav)
        with patch("requests.post", return_value=response) as post, \
                patch.object(ep, "check_tts_server", return_value=True), \
                patch.object(ep.subprocess, "run", return_value=Mock(returncode=1)):
            self.assertTrue(ep.generate_tts("Spoken words", output))
            # Not byte-identical to the server's response any more: generate_tts
            # now level-matches each line so the episode holds a constant level
            # (NARRATION_RMS), which scales the samples. The mock is a constant
            # 1024, i.e. -30.1 dBFS, so it gets +3.6 dB. Assert the audio survived
            # rather than that nothing touched it - asserting byte equality here
            # would forbid any post-processing from ever existing.
            with wave.open(str(output), "rb") as got:
                frames = got.readframes(got.getnframes())
                self.assertEqual(got.getframerate(), 16000)
                self.assertEqual(got.getnframes(), 16000)
            src = np.frombuffer(self.mock_wav[44:], dtype="<i2").astype(np.float32)
            dst = np.frombuffer(frames, dtype="<i2").astype(np.float32)
            self.assertEqual(len(src), len(dst))
            ratio = dst[src != 0] / src[src != 0]
            self.assertLess(float(ratio.std()), 0.01,
                            "level match should be a single uniform gain")
            self.assertAlmostEqual(float(ratio.mean()),
                                   10 ** (min(6.0, max(-6.0,
                                        ep.NARRATION_RMS - 20 * math.log10(
                                            float(np.sqrt((src ** 2).mean())) / 32768.0)))
                                    / 20.0), delta=0.01)
            self.assertEqual(output.with_suffix(".txt").read_text(), "Spoken words")
            self.assertTrue(ep.generate_tts("Spoken words", output))
            self.assertEqual(post.call_count, 1)
            self.config["voice_speed"] = 1.25
            self.assertTrue(ep.generate_tts("Spoken words", output))
            self.assertEqual(post.call_count, 2)
            self.assertEqual(post.call_args.kwargs["json"]["speed"], 1.25)
            self.ref.write_bytes(b"reference two")
            self.assertTrue(ep.generate_tts("Spoken words", output))
            self.assertEqual(post.call_count, 3)

    def test_tts_health_accepts_reference_path(self):
        health = {"status": "ok", "loaded": True,
                  "reference": str(self.root / "voice_user" / "rec.m4a")}
        with patch.object(ep, "http_get", return_value=health):
            self.assertTrue(ep.check_tts_server())
            self.assertTrue(ep.check_tts_server())
        with patch.object(ep, "http_get",
                          return_value={"status": "ok", "loaded": False}):
            self.assertFalse(ep.check_tts_server())

    def test_lock_excludes_threads_and_processes(self):
        results = []
        def contender():
            try:
                with ep._render_ownership():
                    results.append(True)
            except (OSError, RuntimeError):
                results.append(False)
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        self.addCleanup(queue.close)
        with ep._render_ownership():
            thread = threading.Thread(target=contender)
            thread.start()
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(results, [False])
            process = ctx.Process(target=attempt_lock, args=(str(self.root), queue))
            process.start()
            process.join(15)
            if process.is_alive():
                process.terminate()
                process.join()
                self.fail("lock contender hung")
            self.assertEqual(process.exitcode, 0)
            self.assertFalse(queue.get(timeout=2))
            self.assertEqual((self.root / "render.lock").read_text(), str(os.getpid()))
        with ep._render_ownership():
            pass
        self.assertTrue((self.root / "render.lock").exists())

    def test_stop_at_entry_is_not_cleared(self):
        ep.STOP_FILE.touch()
        with patch.object(ep, "ensure_services") as services, \
                patch.object(ep, "_produce_inner") as inner:
            self.assertIsNone(ep.produce_episode(1))
        services.assert_not_called()
        inner.assert_not_called()
        self.assertTrue(ep.STOP_FILE.exists())
        self.assertIsNone(ep._PROGRESS_OWNER)


    def test_progress_monitoring_cannot_write_without_ownership(self):
        monitoring = threading.Thread(target=ep.log, args=("poll",))
        monitoring.start()
        monitoring.join(5)
        self.assertFalse(monitoring.is_alive())
        self.assertFalse(ep.LIVE_PROGRESS.exists())
        with ep._render_ownership():
            ep._PROGRESS_OWNER = (os.getpid(), threading.get_ident())
            ep.log("owned")
            self.assertTrue(ep.LIVE_PROGRESS.exists())
            stamp = json.loads(ep.LIVE_PROGRESS.read_text(encoding="utf-8"))
            self.assertIn("owned", stamp["log"][0])
            ep.set_progress("images", 1, 4, "Scene 2")
            stamp = json.loads(ep.LIVE_PROGRESS.read_text(encoding="utf-8"))
            self.assertEqual(stamp["phase"], "images")
            self.assertGreater(stamp["heartbeat"], 0)
            leftovers = list(ep.LIVE_PROGRESS.parent.glob("progress_live.json.*tmp"))
            self.assertEqual(leftovers, [])

    def test_redo_pinned_scene_refused_and_cache_kept(self):
        still = self.root / "images" / "ch001_s02_a.png"
        still.write_bytes(b"pinned frame")
        (self.root / "pinned.json").write_text(
            json.dumps({"ch001": {"2": {"still": str(still), "note": "keep"}}}),
            encoding="utf-8")
        scenes = [{"scene_number": 1}, {"scene_number": 2}]
        with self.assertRaises(ValueError):
            ep._invalidate_scene(1, 2, scenes, ep._chapter_pins(1))
        self.assertTrue(still.exists())
        with patch.object(ep, "produce_episode") as produce:
            ep.redo_scene(1, 2)
        produce.assert_called_once_with(1, redo_scene_num=2)

    def test_redo_unpinned_scene_invalidates_only_that_scene(self):
        images = self.root / "images"
        (images / "ch001_s02_a.png").write_bytes(b"old still")
        (images / "ch001_s03_a.png").write_bytes(b"other still")
        (self.root / "voice" / "ch001_s02.wav").write_bytes(b"old voice")
        scenes = [{"scene_number": 1}, {"scene_number": 2}, {"scene_number": 3}]
        ep._invalidate_scene(1, 2, scenes, {})
        self.assertFalse((images / "ch001_s02_a.png").exists())
        self.assertFalse((self.root / "voice" / "ch001_s02.wav").exists())
        self.assertTrue((images / "ch001_s03_a.png").exists())

    def test_fresh_run_preserves_pinned_and_final_videos(self):
        images = self.root / "images"
        pinned = images / "ch001_s02_a.png"
        pinned.write_bytes(b"pinned")
        unpinned = images / "ch001_s03_a.png"
        unpinned.write_bytes(b"unpinned")
        final = self.root / "videos" / "ch001_youtube.mp4"
        final.write_bytes(b"last good episode")
        versioned = self.root / "videos" / "ch001_youtube.v1.mp4"
        versioned.write_bytes(b"versioned")
        (self.root / "pinned.json").write_text(
            json.dumps({"ch001": {"2": {"still": str(pinned), "note": "keep"}}}),
            encoding="utf-8")
        with patch.object(ep, "_PROGRESS_OWNER", (os.getpid(), threading.get_ident())):
            ep.clean_chapter(1)
        self.assertTrue(pinned.exists())
        self.assertFalse(unpinned.exists())
        self.assertTrue(final.exists())
        self.assertTrue(versioned.exists())

    def test_comfy_argv_rebuilds_relative_main_py(self):
        # restart_comfyui() relaunches ComfyUI from the live process's own
        # command line. Comfy-Desktop launches it as a RELATIVE "main.py", so
        # the cwd has to be recovered or the relaunch dies on a missing file.
        # base = dirname(dirname(exe)), and the real install is <base>/ComfyUI.
        install = self.root / "installs" / "Example Novel"
        comfy = install / "ComfyUI"
        comfy.mkdir(parents=True)
        (comfy / "main.py").write_text("#", encoding="utf-8")
        exe = install / "standalone-env" / "python.exe"
        exe.parent.mkdir(parents=True)
        exe.write_text("", encoding="utf-8")
        cmd = f'"{exe}" main.py --port 8188 --input-directory C:\\out'
        argv, cwd = ep._comfy_argv(cmd)
        self.assertIsNotNone(argv, "relative main.py must resolve a cwd")
        self.assertEqual(argv[0], str(exe))
        self.assertIn("--port", argv)
        self.assertEqual(argv[-1], "C:\\out")
        self.assertEqual(Path(cwd), comfy)
        # Absolute main.py needs no cwd recovery.
        argv2, cwd2 = ep._comfy_argv(f'"{exe}" {comfy / "main.py"} --port 8188')
        self.assertIsNotNone(argv2)
        self.assertIsNone(cwd2)

    def test_comfy_argv_refuses_unresolvable_path(self):
        # Never relaunch into a directory that does not exist.
        argv, cwd = ep._comfy_argv(
            '"C:\\nope\\env\\python.exe" main.py --port 8188')
        self.assertIsNone(argv)
        self.assertIsNone(cwd)

    def test_tts_fingerprint_keys_the_trim_chain(self):
        # The trim used to live outside the cache key, so changing it produced
        # different bytes from identical inputs while every cached wav still
        # read as valid -- the stale-audio bug the x_vector key already taught.
        fp = ep._tts_fingerprint("hello", 1.0)
        self.assertIn("trim", fp)
        self.assertTrue(fp["x_vector_only"])

    def test_fresh_run_invalidates_tts_cache_on_config_change(self):
        output = self.root / "voice" / "ch001_s01.wav"
        response = Mock(status_code=200, content=self.mock_wav)
        with patch("requests.post", return_value=response) as post, \
                patch.object(ep, "check_tts_server", return_value=True), \
                patch.object(ep.subprocess, "run", return_value=Mock(returncode=1)):
            ep.generate_tts("Spoken words", output)
            ep.generate_tts("Different words", output)
            self.assertEqual(post.call_count, 2)
            self.assertEqual(output.with_suffix(".txt").read_text(), "Different words")

    def test_echo_fingerprint_invalidates_on_audio_change(self):
        aud = self.root / "voice" / "ch001_s01.wav"
        aud.write_bytes(b"voice take one")
        still = self.root / "images" / "ch001_s01_c.png"
        still.write_bytes(b"face still")
        echo = self.root / "images" / "ch001_echo01.mp4"
        echo.write_bytes(b"lips")
        fingerprint = {"text": "spoken", "audio": ep._file_fingerprint(aud)}
        echo.with_suffix(".cache.json").write_text(json.dumps(fingerprint), encoding="utf-8")
        self.assertTrue(ep._cache_matches(echo, dict(fingerprint, text="spoken")))
        self.assertFalse(ep._cache_matches(echo, dict(fingerprint, text="revised")))
        changed = self.root / "voice" / "ch001_s01.wav"
        changed.write_bytes(b"voice take two")
        self.assertFalse(ep._cache_matches(echo, dict(fingerprint, audio=ep._file_fingerprint(changed))))


    def test_produce_episode_lifecycle_and_error_recovery(self):
        output = self.root / "videos" / "ch001_youtube.mp4"
        with patch.object(ep, "ensure_services", return_value=(True, {})), \
                patch.object(ep, "_produce_inner") as inner:
            inner.return_value = output
            self.assertEqual(ep.produce_episode(1), output)
            self.assertTrue((self.root / "render.lock").read_text() == "")
            self.assertIsNone(ep._PROGRESS_OWNER)
            self.assertEqual(ep._RUN_T0["t"], 0.0)
            self.assertEqual(ep._PHASE_T0, {})
            inner.assert_called_once_with(1, preserve=False, redo_scene_num=None)
            inner.side_effect = RuntimeError("boom")
            with self.assertRaises(RuntimeError):
                ep.produce_episode(1)
            self.assertEqual(inner.call_count, 2)
        self.assertFalse(ep._run_lock_held())
        self.assertFalse(ep.RENDER_PID.exists())
        self.assertTrue(ep.LIVE_PROGRESS.exists())
        self.assertEqual(json.loads(ep.LIVE_PROGRESS.read_text(encoding="utf-8"))["phase"], "error")

    def test_preflight_failure_loses_no_cache(self):
        (self.root / "chapters" / "ch001.txt").write_text("story", encoding="utf-8")
        still = self.root / "images" / "ch001_s01_a.png"
        still.write_bytes(b"cached frame")
        with patch.object(ep, "ensure_services", return_value=(True, {})), \
                patch.object(ep, "preflight", return_value=False), \
                patch.object(ep, "clean_chapter") as clean:
            self.assertIsNone(ep.produce_episode(1))
        clean.assert_not_called()
        self.assertTrue(still.exists())


    def test_closeup_prompt_bans_torso_and_face_locks(self):
        scene = {"scene_number": 1, "shot_size": "close-up",
                 "image_prompt": "edvyn in street", "characters": ["edvyn_crael"],
                 "negative_prompt": "", "narration": "he steps forward",
                 "dialogue": "", "stages": [], "pose": ""}
        captured = {}

        def fake_post(url, payload, timeout=None):
            captured["wf"] = payload["prompt"]
            raise RuntimeError("capture-only")

        with patch.object(ep, "http_post_json", side_effect=fake_post), \
                patch.object(ep, "props_for_scene", return_value=[]), \
                patch.object(ep, "pose_for_scene", return_value=""):
            try:
                ep.generate_image(scene, 0, 1)
            except RuntimeError:
                pass

        pos = captured["wf"]["2"]["inputs"]["prompt"].lower()
        neg = captured["wf"]["2"]["inputs"]["negative_prompt"].lower()
        self.assertIn("face fills most of frame", pos)
        self.assertNotIn("upper body portrait", pos)
        self.assertIn("torso shot", neg)
        self.assertIn("waist-up crop", neg)

    def test_still_cache_sidecar_records_prompt_not_ownership(self):
        # The PNG is the artefact; the sidecar is only a note recording which
        # prompt produced it. A MISSING note must not condemn the image -- that
        # is what threw away 38 good stills and forced a full re-render.
        # But "unverified" must never be laundered into "verified": only a
        # caller that has DECIDED to use the file may write provenance.
        still = self.root / "images" / "ch001_s01_a.png"
        still.write_bytes(b"still")
        scene = {"image_prompt": "x", "shot_size": "close-up"}
        sidecar = still.with_suffix(".cache.json")

        # purge path: reusable, but no note is invented
        self.assertTrue(ep._still_cache_valid(still, scene))
        self.assertFalse(sidecar.exists(), "purge path must not write provenance")

        # use path: records what we decided to adopt
        self.assertTrue(ep._still_cache_valid(still, scene, adopt=True))
        self.assertTrue(sidecar.is_file())

        # a matching note validates...
        sidecar.write_text(json.dumps(ep._visual_fingerprint(scene, still_path=still)),
                           encoding="utf-8")
        self.assertTrue(ep._still_cache_valid(still, scene))
        # ...and a note recording a DIFFERENT prompt still invalidates
        sidecar.write_text(json.dumps({"image_prompt": "y"}), encoding="utf-8")
        self.assertFalse(ep._still_cache_valid(still, scene))

    def test_fingerprint_covers_every_still_generation_input(self):
        """Every config key generate_image/build_workflow actually READS must
        appear in _STILL_CFG_KEYS.

        ponytail: two versions of this test existed and both were worthless.
        The original was `assertIn("framing", ep._visual_fingerprint(scene))` —
        a tautology checking a hardcoded 2 was a dict key. The replacement
        looped `for key in ep._STILL_CFG_KEYS`, which is tautological in the
        direction that matters: DELETING a key from the list deletes its own
        check, and removing 4 real still-affecting keys (checkpoint,
        image_sampler, image_scheduler, width) left the suite green.

        So the key list is now pinned independently, against what the code
        really reads. Removing any of these fails the suite.
        """
        import inspect
        # The keys the still path genuinely consumes. Keep in sync ONLY by
        # adding a test failure here first.
        required = {
            "checkpoint", "image_cfg", "image_sampler", "image_scheduler",
            "image_seed_base", "lora_strength", "qwen_lora_trigger",
            "qwen_text_encoder", "qwen_vae", "still_encoder_device",
            "sheet_anchor_denoise", "sheet_anchor_head_span",
            "sheet_anchor_head_frac",
            # consumed by get_output_dims(), which generate_image calls for the
            # portrait-native canvas: a platform switch changes the pixels.
            "platform", "width", "height",
        }
        missing = required - set(ep._STILL_CFG_KEYS)
        self.assertFalse(missing,
                         f"these still-affecting keys are absent from "
                         f"_STILL_CFG_KEYS, so editing them would silently "
                         f"serve stale PNGs: {sorted(missing)}")

        # And every key the function reads must move the key.
        src = (inspect.getsource(ep.generate_image)
               + inspect.getsource(ep.get_output_dims))
        read = set(re.findall(r'CONFIG\.get\(\s*["\']([a-z0-9_]+)["\']', src))
        # comfyui_url is read but does not affect pixels (it is the endpoint).
        read -= {"comfyui_url"}
        # auto_tune is a dict: only `auto_tune.image_steps` reaches the
        # sampler, and it is fed under its own dotted key (ep._AUTO_TUNE_KEY)
        # rather than as a bare "auto_tune" entry. Hashing the whole dict would
        # rotate every still on each quality_evolve history append.
        read -= {"auto_tune"}
        seen_keys = set(ep._STILL_CFG_KEYS) | {ep._AUTO_TUNE_KEY}
        unseen = read - seen_keys
        self.assertFalse(unseen,
                         f"the still path reads these but the fingerprint "
                         f"ignores them: {sorted(unseen)}")

        scene = {"image_prompt": "x", "mood": "tense", "characters": ["kael"]}
        base = ep._visual_fingerprint(scene)["hash"]
        for key in ep._STILL_CFG_KEYS:
            with self.subTest(config_key=key):
                original = ep.CONFIG.get(key)
                try:
                    ep.CONFIG[key] = ({"image_steps": 999} if key == "auto_tune"
                                      else "__moved__")
                    self.assertNotEqual(
                        ep._visual_fingerprint(scene)["hash"], base,
                        f"config key {key!r} does not invalidate the still cache")
                finally:
                    ep.CONFIG[key] = original

    def test_fingerprint_tracks_character_bible(self):
        """Editing a character in characters.json must invalidate their stills.

        ponytail: the first version of this test changed BOTH `characters` and
        `settings` in the same breath, so the separately-hashed settings feed
        moved the digest and satisfied both assertions. Removing the character
        feed entirely left the suite GREEN (verified by mutation). Each case now
        holds the other subtree byte-identical, so only the intended input can
        possibly be responsible for a change.
        """
        scene = {"image_prompt": "x", "characters": ["kael"]}
        settings = {"white_sky_city": {"visual_tags": "fog, canals"}}
        original = ep.CHAR_BIBLE
        try:
            # Case 1: character entry changes, settings held IDENTICAL.
            ep.CHAR_BIBLE = {"characters": {"kael": {"hair": "RED"}},
                             "settings": settings}
            with_char = ep._visual_fingerprint(scene)["hash"]

            # Only the character differs from the line above; settings is the
            # same object. So the digests MUST differ.
            ep.CHAR_BIBLE = {"characters": {"kael": {"hair": "BLUE"}},
                             "settings": settings}
            self.assertNotEqual(
                ep._visual_fingerprint(scene)["hash"], with_char,
                "editing a character's hair must invalidate that character's stills")

            # Case 2: settings change, characters held IDENTICAL.
            ep.CHAR_BIBLE = {"characters": {"kael": {"hair": "RED"}},
                             "settings": {"white_sky_city": {"visual_tags": "CHANGED"}}}
            with_setting = ep._visual_fingerprint(scene)["hash"]
            self.assertNotEqual(
                ep._visual_fingerprint(scene)["hash"], with_char,
                "changing a setting's visual_tags must invalidate stills")
            _ = with_setting

            # Case 3: a character NOT in the scene must not move the digest —
            # otherwise unrelated bible edits re-render everything.
            ep.CHAR_BIBLE = {"characters": {"kael": {"hair": "RED"},
                                            "someone_else": {"hair": "X"}},
                             "settings": settings}
            self.assertEqual(ep._visual_fingerprint(scene)["hash"], with_char,
                             "an unrelated character must not invalidate this scene")
        finally:
            ep.CHAR_BIBLE = original

    def test_pick_cached_still_requires_a_valid_sidecar(self):
        """The rule the beat-2 and companion paths both use.

        ponytail: this rule lived inline inside the 1200-line _produce_inner,
        as a bare `if existing:` on a glob, which meant ANY file on disk counted
        as a usable cached still. Mutation testing proved no test could reach
        it from there — so the selection was extracted to _pick_cached_still and
        this test pins the behaviour directly.
        """
        images = self.root / "images"
        images.mkdir(parents=True, exist_ok=True)
        scene = {"image_prompt": "x", "shot_size": "medium", "mood": "tense"}
        pat = "ch001_s01b_*.png"

        # Nothing on disk at all -> must render.
        self.assertIsNone(ep._pick_cached_still(images, pat, scene))

        # A file with NO provenance -> reusable (unverified, not condemned)...
        good = images / "ch001_s01b_aaaa_00001_.png"
        good.write_bytes(b"x" * 64)
        self.assertEqual(ep._pick_cached_still(images, pat, scene), good)

        # ...but once a sidecar exists it is authoritative.
        good.with_suffix(".cache.json").write_text(
            json.dumps({"v": 2, "hash": "WRONG"}), encoding="utf-8")
        self.assertIsNone(ep._pick_cached_still(images, pat, scene),
                          "a sidecar recording a different fingerprint must "
                          "force a regeneration")

        # A 0-byte file is never a keeper, even with no sidecar.
        empty = images / "ch001_s01b_bbbb_00001_.png"
        empty.write_bytes(b"")
        self.assertIsNone(ep._pick_cached_still(images, pat, scene),
                          "a 0-byte still must never be selected as a cache hit")

        # Repair the sidecar -> selected again, and the NEWEST valid one wins.
        self.assertTrue(ep._write_still_sidecar(good, scene))
        empty.unlink()
        self.assertEqual(ep._pick_cached_still(images, pat, scene), good)
        newer = images / "ch001_s01b_zzzz_00001_.png"
        newer.write_bytes(b"y" * 64)
        self.assertTrue(ep._write_still_sidecar(newer, scene))
        os.utime(good, (10, 10))        # older
        os.utime(newer, (20, 20))       # newer  -> deterministic, no clock skew
        self.assertEqual(ep._pick_cached_still(images, pat, scene), newer)

    def test_fingerprint_ignores_auto_tune_history(self):
        """`auto_tune.history` is an append-only changelog, not an input.

        It used to be hashed as part of the auto_tune dict, so every tune
        event rotated every still sidecar and forced a full ~90 min re-render
        for no change to any pixel. Only `auto_tune.image_steps` reaches the
        KSampler, so only that field may move the key.
        """
        scene = {"image_prompt": "x", "mood": "tense"}
        original = ep.CONFIG.get("auto_tune")
        try:
            base = {"image_steps": 25, "history": [{"at": "2026-09-30", "changes": ["a"]}]}
            ep.CONFIG["auto_tune"] = base
            before = ep._visual_fingerprint(scene)["hash"]

            # 1. a pure log append must NOT move the key
            ep.CONFIG["auto_tune"] = dict(
                base, history=base["history"] + [{"at": "2026-10-02",
                                                  "changes": ["bright_boost->True"]}])
            self.assertEqual(
                ep._visual_fingerprint(scene)["hash"], before,
                "appending to auto_tune.history must not invalidate a still")

            # 2. but the field that reaches the sampler MUST
            ep.CONFIG["auto_tune"] = dict(base, image_steps=26)
            self.assertNotEqual(
                ep._visual_fingerprint(scene)["hash"], before,
                "auto_tune.image_steps changes the KSampler and must invalidate")
        finally:
            ep.CONFIG["auto_tune"] = original

    def test_token_spend_refuses_when_the_write_fails(self):
        """A charge that was never persisted must not be reported as success."""
        import json as _json
        import tempfile
        from pipeline import tokens as tk, config as cfg
        with tempfile.TemporaryDirectory() as td:
            tokfile = Path(td) / "tokens.json"
            with patch.object(tk, "_file", lambda: tokfile), \
                 patch.object(cfg, "tokens_enabled",
                              lambda default=False: True):
                tk.grant(10)
                self.assertEqual(tk.balance(), 10)
                with patch.object(tk, "dump_json_atomic",
                                  side_effect=OSError("locked")):
                    with self.assertRaises(OSError):
                        tk.spend(4)
                # the file is untouched, so the caller is not told it was charged
                self.assertEqual(tk.balance(), 10)
                self.assertEqual(_json.loads(tokfile.read_text(encoding="utf-8"))
                                 ["balance"], 10)

    def test_beat_and_companion_stills_are_sidecar_gated(self):
        """beat-2 (`_sNNb_`) and the lip-sync companion (`_sNNc_`) must be
        validated against a provenance sidecar, not merely exist.

        ponytail: glob existence was the ENTIRE cache policy for both, so a
        companion rendered from a different prompt — or against an older
        character bible — was replayed forever as the EchoMimic identity
        source. A 0-byte PNG must also be refused (that is a half-written
        artefact, never a keeper).
        """
        images = self.root / "images"
        images.mkdir(parents=True, exist_ok=True)
        scene = {"image_prompt": "x", "shot_size": "medium", "mood": "tense"}
        still = images / "ch001_s01b_deadbeef_00001_.png"
        still.write_bytes(b"x" * 64)          # non-empty but 0 provenance

        # No sidecar at all -> "unverified", which is still reusable.
        self.assertTrue(ep._still_cache_valid(still, scene),
                        "a missing sidecar must not condemn a real PNG")

        # Wrong sidecar -> invalid.
        still.with_suffix(".cache.json").write_text(
            json.dumps({"v": 1, "image_prompt": "a different prompt"}),
            encoding="utf-8")
        self.assertFalse(ep._still_cache_valid(still, scene),
                         "a sidecar recording a different prompt must invalidate")

        # Correct sidecar -> valid, for ANY companion/beat scene, proving the
        # gate is keyed on content rather than on which files exist.
        self.assertTrue(ep._write_still_sidecar(still, scene))
        self.assertTrue(ep._still_cache_valid(still, scene))

        # 0-byte file is never a keeper even with a valid sidecar.
        empty = images / "ch001_s01c_empty_00001_.png"
        empty.write_bytes(b"")
        self.assertTrue(ep._write_still_sidecar(empty, scene))
        self.assertFalse(ep._still_cache_valid(empty, scene),
                         "a 0-byte still must be refused regardless of its sidecar")

    def test_fingerprint_is_deterministic_and_serialisable(self):
        scene = {"image_prompt": "x", "characters": ["kael"]}
        self.assertEqual(ep._visual_fingerprint(scene), ep._visual_fingerprint(scene))
        json.dumps(ep._visual_fingerprint(scene))  # must survive a round trip

    def test_echo_fingerprint_tracks_the_fields_that_build_the_prompt(self):
        """shot_size and mood compose the EchoMimic prompt; they must invalidate
        a cached lip clip. They used to be absent from the key."""
        audio = self.root / "a.wav"
        audio.write_bytes(b"wav")
        still = self.root / "s.png"
        still.write_bytes(b"png")
        scene = {"narration": "hello", "shot_size": "medium", "mood": "cold",
                 "_dur": 4.0}
        base = ep._echo_fingerprint(still, audio, scene)
        for key, newval in (("shot_size", "close-up"), ("mood", "warm")):
            with self.subTest(field=key):
                moved = dict(scene)
                moved[key] = newval
                self.assertNotEqual(
                    ep._echo_fingerprint(still, audio, moved), base,
                    f"{key} changes the lip-sync prompt but not the cache key")

    def test_still_cache_rejects_half_written_png(self):
        # An interrupted write leaves a 0-byte file. Adopting it would pin a
        # broken image forever behind a sidecar claiming provenance.
        still = self.root / "images" / "ch001_s01_a.png"
        still.parent.mkdir(parents=True, exist_ok=True)
        still.write_bytes(b"")
        scene = {"image_prompt": "x"}
        self.assertFalse(ep._still_cache_valid(still, scene, adopt=True))
        self.assertFalse(still.with_suffix(".cache.json").exists())


    def test_grade_clip_survives_one_odd_final_frame(self):
        # ch001 scene 9 burned two full H3 generations (~11 min) and was
        # discarded for a still-pan anyway, because the grader compared only
        # the first and last frame and the last frame happened to be dark.
        # One odd frame out of five must no longer convict a clip.
        import numpy as np
        from PIL import Image

        d = self.root / "frames"
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(7)

        def frame(i, dark=False, drift=0):
            base = np.full((240, 320, 3), 200, dtype=np.uint8)
            base[60:180, 100:220] = 120# a stable "subject"
            base = np.clip(base.astype(np.int16) + rng.integers(-4, 4, base.shape)
                           + drift, 0, 255).astype(np.uint8)
            if dark:
                base = (base * 0.25).astype(np.uint8)   # simulates a settled tail
            Image.fromarray(base).save(d / f"g{i:02d}.png")
            return d / f"g{i:02d}.png"

        good = [frame(0), frame(1), frame(2), frame(3), frame(4, dark=True)]
        passed, why = ep.grade_clip(good)
        self.assertTrue(passed, f"good clip with one dark tail rejected: {why}")

        # A take that genuinely drifts must still fail.
        bad = [frame(10), frame(11, drift=30), frame(12, drift=90),
               frame(13, drift=150), frame(14, drift=200)]
        passed2, why2 = ep.grade_clip(bad)
        self.assertFalse(passed2, f"drifting clip passed: {why2}")


    def test_grade_clip_face_count_does_not_gate(self):
        # Measured on the real ch001 clips: _count_faces returns 0 on 4 of 5
        # frames of clips that are fine (h301, h311), and 1-on-frame-0 for
        # h302. One spurious detection therefore decided the verdict, which
        # rejected ch001 scene 9 and cost two H3 generations (~11 min) for a
        # good clip. The face count is recorded, never a gate; histogram and
        # sharpness still are.
        import numpy as np
        from PIL import Image
        import episode_producer as ep

        d = self.root / "gateframes"
        d.mkdir(parents=True, exist_ok=True)

        def mk(name, scale=1.0, shift=0):
            a = np.full((240, 320, 3), 200, dtype=np.uint8)
            a[60:180, 100:220] = 120
            a = np.clip(a.astype(np.int16) * scale + shift, 0, 255).astype(np.uint8)
            Image.fromarray(a).save(d / name)
            return d / name

        # detector reports 1 face on the first frame only. iter(callable), not
        # iter(list): the drift grade_clip below runs with the patch still
        # installed, and a spent iterator raises StopIteration, which
        # grade_clip swallows into "grader error" — so the assertion that a
        # drifting clip fails would pass for the wrong reason. Keep answering 0.
        fake_counts = iter([1, 0, 0, 0, 0])
        orig = ep._count_faces
        ep._count_faces = lambda p: next(fake_counts, 0)
        try:
            frames = [mk(f"f{i}.png") for i in range(5)]
            passed, why = ep.grade_clip(frames)
            self.assertTrue(passed, f"face count gated the clip: {why}")
            self.assertEqual(why, [])

            # a real continuity failure still fails
            drift = [mk("g0.png"), mk("g1.png", shift=20), mk("g2.png", shift=90),
                     mk("g3.png", shift=160), mk("g4.png", shift=220)]
            passed2, why2 = ep.grade_clip(drift)
            self.assertFalse(passed2, "genuinely drifting clip passed")
        finally:
            ep._count_faces = orig


    def test_grade_clip_rejects_blank_frames(self):
        # A frame with no image structure is a visible defect on its own and
        # must not be outvoted by four good frames. Continuity is a majority
        # property; a blank frame is not.
        import numpy as np
        from PIL import Image
        d = self.root / "blank"
        d.mkdir(parents=True, exist_ok=True)

        def mk(name, structure=True):
            a = np.full((240, 320, 3), 200, dtype=np.uint8)
            if structure:
                a[60:180, 100:220] = 120
                a[90:150, 130:190] = 30
            Image.fromarray(a).save(d / name)
            return d / name

        good = [mk("g0.png"), mk("g1.png"), mk("g2.png"), mk("g3.png")]
        blanks = [mk("b0.png", structure=False), mk("b1.png", structure=False)]
        passed, why = ep.grade_clip(good + blanks)
        self.assertFalse(passed, f"half-blank clip accepted: {why}")
        self.assertTrue(any("blank" in r for r in why), why)

        # and a uniformly good clip still passes
        ok, why2 = ep.grade_clip([mk(f"c{i}.png") for i in range(5)])
        self.assertTrue(ok, why2)

    def test_shot_plan_pass_populates_purpose_and_tests_real_enum(self):
        # _normalize_scenes runs first in the render path and sets purpose="",
        # so setdefault("purpose", ...) left every scene empty — canon wrote
        # purpose:"?" and beat_planner never took its primary branch. And the
        # branch searched hook_type for TURN/REVEAL/TWIST, none of which is in
        # the prompt enum, so it could never fire on a real model answer.
        scenes = ep._normalize_scenes(
            [{"hook_type": h, "image_prompt": "a man walks", "narration": "x",
              "scene_number": i + 1}
             for i, h in enumerate(["mystery", "shock", "tension", "emotion",
                                    "cliffhanger", "tension", "mystery", "emotion"])])
        out = ep.shot_plan_pass([dict(s) for s in scenes])
        self.assertTrue(all(s["purpose"] for s in out),
                        f"purpose left empty: {[s.get('purpose') for s in out]}")
        self.assertEqual(out[0]["purpose"], "HOOK")
        # shock is a TURN value and lands on a middle index, so it must fire
        self.assertEqual(out[1]["purpose"], "TURN")
        # cliffhanger is a real TURN value the model actually emits
        self.assertEqual(out[4]["purpose"], "TURN")
        self.assertIn(out[-1]["purpose"], ("TURN", "LAND"))

    def test_purpose_is_not_part_of_the_still_fingerprint(self):
        # Populate purpose after _normalize_scenes and the still fingerprint
        # must NOT change — or the shot_plan_pass fix would invalidate every
        # cached still in every episode.
        base = ep._normalize_scenes(
            [{"image_prompt": "a man", "narration": "x", "scene_number": 1}])[0]
        fp0 = ep._visual_fingerprint(base)
        with_purpose = dict(base, purpose="TURN")
        self.assertEqual(fp0, ep._visual_fingerprint(with_purpose),
                         "purpose leaked into the still fingerprint")

    def test_clip_grade_upscales_with_cover_crop_not_squash(self):
        # A bare scale= squashes 960x544 into 1920x1080 by 0.74% (2.000x
        # horizontally vs 1.985x vertically), baked into every generated clip.
        # vertical_fit_filter scales-to-cover and center-crops instead.
        import inspect
        src = inspect.getsource(ep._apply_clip_grade)
        self.assertIn("vertical_fit_filter", src,
                      "_apply_clip_grade must use the cover-crop helper, not a "
                      "bare scale")
        self.assertNotIn('vf = f"scale={out_w}:{out_h}"', src,
                         "a bare scale= survived the fix")
        # and sharpen stays off here because NORM sharpens once at the end
        self.assertIn("sharpen=False", src)

    def test_platform_srt_prefers_the_cps_compliant_sidecar(self):
        # generate_subtitles wraps to nothing and has no CPS concept — its
        # videos/chNNN.srt hit 55 chars/line. The ASS-derived sidecar next to
        # the mp4 enforces 42/17. build_release_package must ship the latter.
        out = self.root / "videos" / "ch002_youtube.mp4"
        out.write_bytes(b"mp4")
        sidecar = out.with_suffix(".srt")
        self.assertFalse(sidecar.exists())
        self.assertIsNone(ep._platform_srt(out))
        sidecar.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n",
                           encoding="utf-8")
        self.assertEqual(ep._platform_srt(out), sidecar)
        # an empty file is not a subtitle track
        sidecar.write_text("", encoding="utf-8")
        self.assertIsNone(ep._platform_srt(out))

    def test_ass_line_break_becomes_a_real_newline_in_the_srt(self):
        # \N is an ASS line-break escape; SRT has no such thing. The old
        # exporter stripped ASS override tags but left \N behind, so a shipped
        # cue read "waking\Nup." verbatim. Importantly the tag strip runs on
        # the same string, so the \N substitution must not be shadowed by it.
        import re as _re
        ass_text = "{\\an8}Outside, the industrial city was waking\\Nup."
        rendered = _re.sub(r"\\N", "\n", _re.sub(r"\{[^}]*\}", "", ass_text)).strip()
        self.assertIn("\n", rendered)
        self.assertNotIn("\\N", rendered)
        self.assertNotIn("{", rendered)

    def test_generate_tts_passage_fallback_does_not_pass_speed(self):
        # The fallback called generate_tts(texts[i], paths[i], speed=speed) but
        # generate_tts has no speed= parameter — a latent TypeError, unreachable
        # only because _PASSAGE_ENABLED is False. The keyword must be gone.
        import inspect
        src = inspect.getsource(ep.generate_tts_passage)
        self.assertNotIn("speed=speed", src,
                         "generate_tts_passage still passes a speed= kwarg "
                         "generate_tts does not accept")

    def test_scene_target_scales_for_essential_adaptation(self):
        # The episode's length is a function of the CHAPTER, not a policy: the
        # user asked for no length cap and for every essential beat to be on
        # screen. ~280 chars/scene is a complete adaptation (a scene per
        # beat); at 560 a 6000-char chapter was compressed into 10 scenes,
        # which is a summary of a chapter, not the chapter. The only ceiling
        # left is PHYSICAL - what the LLM's answer can fit in the context
        # window - so it grows with the chapter instead of stopping at 40.
        self.assertEqual(ep.scene_target_count(500), 10)        # floor
        self.assertEqual(ep.scene_target_count(6208), 22)       # a LOTM chapter
        self.assertGreaterEqual(ep.scene_target_count(17714), 60)  # ch2
        # a huge chapter is bounded by the CONTEXT, not by a hard cap
        self.assertEqual(ep.scene_target_count(10_000_000),
                         ep._scene_budget_ceiling(44000))

    def test_the_scene_ceiling_is_the_context_window_not_a_number(self):
        # A policy cap is how a chapter loses its essential beats: the LLM is
        # told "EXACTLY N scenes" and cuts to fit. The ceiling must come from
        # what the server can actually answer, so raising llm_ctx raises it.
        import inspect
        src = inspect.getsource(ep.scene_target_count)
        self.assertNotIn("min(40", src, "the hard scene cap must be gone")
        self.assertIn("_scene_budget_ceiling", src)
        src2 = inspect.getsource(ep.break_into_scenes)
        self.assertIn("_script_answer_budget", src2,
                      "the request must be budgeted by the same window")

    def test_script_prompt_demands_coverage_and_voices_dialogue(self):
        # The two changes that turn a recap into an adaptation: an explicit
        # coverage directive, and the rule that nothing essential may live only
        # in the (unvoiced) dialogue field.
        import inspect
        src = inspect.getsource(ep)
        self.assertIn("COVERAGE — THE PRIME DIRECTIVE", src)
        self.assertIn("Nothing essential may live ONLY in the dialogue field",
                      src)
        self.assertIn("essential story", src)
        # ...and it must not ask the model to compress the chapter away
        self.assertNotIn("Compress hard", src,
                         "the prompt must not over-cut the chapter's beats")

    def test_script_answer_fits_the_context_window(self):
        # The answer is budgeted by the CONTEXT WINDOW: prompt + chapter +
        # answer must fit in llm_ctx, or the server rejects the request
        # outright ("the request exceeds the available context size"). The
        # config value is a CAP on the ask; the clamp is what must fit.
        import inspect
        src = inspect.getsource(ep.break_into_scenes)
        self.assertIn("_script_answer_budget", src,
                      "the script request must be budgeted by the context window")
        ctx = int(self.config.get("llm_ctx", 32768))
        chapter_chars = int(self.config.get("llm_chapter_chars", 44000))
        budget = ep._script_answer_budget(chapter_chars)
        prompt_tok = ep._PROMPT_CHARS / ep._CHARS_PER_TOKEN
        self.assertLessEqual(prompt_tok + chapter_chars / ep._CHARS_PER_TOKEN + budget,
                             ctx, "prompt + chapter + answer no longer fits llm_ctx")
        self.assertLessEqual(budget,
                             int(self.config.get("llm_script_max_tokens", 32768)),
                             "the configured cap must still bound the ask")

    def test_lora_attaches_to_the_character_the_shot_is_of(self):
        # The LoRA carries ONE face. ch2 scene 16 listed
        # [edvyn_crael, hartwell_principal] and Edvyn's LoRA loaded onto a shot
        # whose image_prompt described Hartwell — the client drawn as the
        # protagonist. The image_prompt names the subject.
        order = ep._lora_candidate_order(
            {"characters": ["edvyn_crael", "hartwell_principal"],
             "image_prompt": "Hartwell standing in the doorway, grey suit"})
        self.assertEqual(order[0], "hartwell_principal")
        order = ep._lora_candidate_order(
            {"characters": ["edvyn_crael", "hartwell_principal"],
             "image_prompt": "Edvyn watches from across the room"})
        self.assertEqual(order[0], "edvyn_crael")
        # shots naming nobody keep script order (stable sort, no reordering)
        order = ep._lora_candidate_order(
            {"characters": ["edvyn_crael", "maren_voss"],
             "image_prompt": "two figures in a dark hallway"})
        self.assertEqual(order, ["edvyn_crael", "maren_voss"])

    def test_lora_does_not_hijack_a_shot_of_another_character(self):
        # ch3 scene 3: the shot is Maret opening the door, Edvyn is merely in
        # the cast list. The old code fell through to Edvyn's LoRA and drew HIS
        # face. Now a prompt-named character with no LoRA means NO LoRA at all,
        # so her bible tags carry the face instead of a hijack.
        order = ep._lora_candidate_order(
            {"characters": ["edvyn_crael", "maret_ovrik"],
             "image_prompt": "Maret Ovrik opening the door, hollow-eyed"})
        self.assertEqual(order, ["maret_ovrik"],
                         "Edvyn must be dropped from a shot that names Maret")

    def test_wide_shots_prompt_for_deep_focus_not_bokeh(self):
        # The wide-shot complaint: a sharp face over a pixel-blurred backdrop.
        # The prompt must (a) swap the face-detail DNA for an environment block
        # and (b) name the ACTUAL blur mechanism in the negative — the old
        # "soft focus" negative never said "depth of field" or "bokeh".
        self.assertTrue(ep._wide_shot({"shot_size": "wide"}))
        self.assertTrue(ep._wide_shot({"shot_size": "extreme wide"}))
        self.assertTrue(ep._wide_shot({"type": "establishing"}))
        self.assertFalse(ep._wide_shot({"shot_size": "close-up"}))
        self.assertFalse(ep._wide_shot({}))
        import inspect
        src = inspect.getsource(ep.generate_image)
        for token in ("deep depth of field", "blurry background", "bokeh",
                      "wide establishing shot", "_wide_shot"):
            self.assertIn(token, src,
                          f"wide-shot prompt handling missing {token!r}")

    def test_the_narration_limiter_leaves_codec_headroom(self):
        # The delivered master is AAC; the DECODED file's true peak sits above
        # the limiter's target, so a limiter AT the ceiling publishes a file
        # its own gate rejects (measured: -2.00 dBTP in, -1.55 out).
        import inspect
        src = inspect.getsource(ep._loudnorm_dual)
        self.assertIn("_TP_LIMIT - _TP_LIMIT_HEADROOM", src,
                      "the narration limiter must leave the SAME headroom the "
                      "music path leaves - two margins for one codec drift "
                      "apart, and the narrow one fails the gate on exactly "
                      "the renders with no music bed")
        self.assertGreater(ep._TP_LIMIT_HEADROOM, 0.0)

    def test_the_learner_records_the_scripts_spelling_as_an_alias(self):
        # The script spelled the character "ammon"; the model answered "amon".
        # The bible lookup is by key, so without an alias that character had NO
        # tags in the still prompt at all.
        bible = {"characters": {}}
        payload = json.dumps({"amon": {
            "full_name": "Amon", "role": "Marauder",
            "danbooru_tags": "1boy, black hair, black eyes, monocle",
            "prompt_tags": "black-haired man with a crystal monocle"}})
        with patch.object(ep, "CHAR_BIBLE", bible), \
                patch.object(ep, "dump_json_atomic"), \
                patch.object(ep, "llm_chat", return_value=payload):
            added = ep._learn_new_characters(["ammon"], None, 2)
        self.assertEqual(added, ["amon"])
        self.assertIn("ammon", bible["characters"]["amon"].get("aliases") or [],
                      "the script's spelling must resolve to the learned entry")

    def test_the_still_cache_key_follows_a_bible_alias(self):
        # A change to an ALIASED bible entry must move the still cache key.
        # The raw lookup found nothing for the alias, so the cache never
        # noticed and the stills carried no identity tags for that character.
        bible = {"characters": {"amon": {
            "full_name": "Amon", "danbooru_tags": "1boy, black hair",
            "aliases": ["ammon"]}}}
        scene = {"scene_number": 1, "characters": ["ammon"], "narration": "x"}
        with patch.object(ep, "CHAR_BIBLE", bible):
            before = ep._visual_fingerprint(scene)["hash"]
            bible["characters"]["amon"]["danbooru_tags"] = "1boy, black hair, monocle"
            after = ep._visual_fingerprint(scene)["hash"]
        self.assertNotEqual(before, after,
                            "a change to the aliased bible entry must move the "
                            "still cache key")

    def test_a_bible_alias_resolves_to_its_character(self):
        # A novel with a masked protagonist has many names for one person.
        # Without aliases the learner invents a SECOND entry - and a second
        # face - for someone the bible already has.
        bible = {"characters": {"klein_moretti": {
            "full_name": "Klein Moretti", "danbooru_tags": "1boy, black hair",
            "aliases": ["mr_fake_it_until_you_make_it"]}}}
        with patch.object(ep, "CHAR_BIBLE", bible):
            self.assertEqual(ep._char_tags("mr_fake_it_until_you_make_it"),
                             "1boy, black hair")
            self.assertEqual(ep._bible_alias_map(),
                             {"mr_fake_it_until_you_make_it": "klein_moretti"})

    def test_the_learner_does_not_add_a_second_entry_for_an_alias(self):
        bible = {"characters": {"klein_moretti": {
            "full_name": "Klein Moretti", "danbooru_tags": "1boy, black hair",
            "aliases": ["mr_fake_it_until_you_make_it"]}}}
        payload = json.dumps({"mr_fake_it_until_you_make_it": {
            "full_name": "Mr. Fake It Until You Make It", "role": "mentioned",
            "danbooru_tags": "1boy, generic", "prompt_tags": "generic face"}})
        with patch.object(ep, "CHAR_BIBLE", bible), \
                patch.object(ep, "llm_chat", return_value=payload):
            added = ep._learn_new_characters(["alaric_zoroast"], None, 1)
        self.assertNotIn("mr_fake_it_until_you_make_it", added,
                         "an alias is the same person - it must not become a "
                         "second bible entry (and a second face)")

    def test_learn_new_characters_writes_and_reloads_the_bible(self):
        from unittest.mock import patch
        import json as _json
        # a chapter naming a character the bible does not know
        new_cast = ["edvyn_crael", "maret_ovrik"]
        payload = _json.dumps({
            "maret_ovrik": {
                "full_name": "Maret Ovrik", "role": "Lena's mother", "age": "50s",
                "build": "frail", "face": "hollow, weary", "hair": "grey",
                "eyes": "pale", "clothing": "dark house dress",
                "distinguishing_features": "trembling hands",
                "personality_visual": "grieving, watchful",
                "danbooru_tags": "1girl, old woman, grey hair, dark dress",
                "prompt_tags": "elderly woman, grey hair, dark house dress",
                "negative_tags": "young, bright colors"}})
        before = set(ep.CHAR_BIBLE.get("characters", {}))
        with patch.object(ep, "llm_chat", return_value=payload), \
             patch.object(ep, "dump_json_atomic") as dump:
            ep._learn_new_characters(new_cast, None, 3)
        after = set(ep.CHAR_BIBLE.get("characters", {}))
        self.assertIn("maret_ovrik", after - before,
                      "a new character must be added to the bible")
        self.assertTrue(dump.called, "the bible must be persisted")
        # cleanup so we do not leak the test character into the real bible
        if "maret_ovrik" not in before:
            ep.CHAR_BIBLE.get("characters", {}).pop("maret_ovrik", None)

    def test_learn_new_characters_never_aborts_on_llm_failure(self):
        from unittest.mock import patch
        before = set(ep.CHAR_BIBLE.get("characters", {}))
        with patch.object(ep, "llm_chat", side_effect=RuntimeError("llm down")):
            ep._learn_new_characters(["ghost_character"], None, 3)  # must not raise
        after = set(ep.CHAR_BIBLE.get("characters", {}))
        self.assertEqual(before, after, "a failed learning call must not touch the bible")

    def test_learn_new_characters_skips_empty_tags(self):
        from unittest.mock import patch
        import json as _json
        payload = _json.dumps({"ghost_character": {"full_name": "Ghost",
                                                   "danbooru_tags": "",
                                                   "prompt_tags": ""}})
        before = set(ep.CHAR_BIBLE.get("characters", {}))
        with patch.object(ep, "llm_chat", return_value=payload), \
             patch.object(ep, "dump_json_atomic"):
            ep._learn_new_characters(["ghost_character"], None, 3)
        after = set(ep.CHAR_BIBLE.get("characters", {}))
        self.assertNotIn("ghost_character", after,
                         "an entry with empty tags must not be written")

    def test_learn_new_characters_returns_added_keys(self):
        import json as _json
        payload = _json.dumps({"brand_new_guy": {
            "full_name": "Brand New", "danbooru_tags": "1boy, tall",
            "prompt_tags": "a tall man"}})
        with patch.object(ep, "llm_chat", return_value=payload), \
             patch.object(ep, "dump_json_atomic"):
            added = ep._learn_new_characters(["brand_new_guy"], None, 12)
        self.assertEqual(added, ["brand_new_guy"],
                         "learning must report the keys it added")
        ep.CHAR_BIBLE.get("characters", {}).pop("brand_new_guy", None)

    def test_train_character_lora_stages_dataset_writes_config_and_installs(self):
        root, key = self.root, "sample_guy"
        (root / "dataset" / f"{key}_lora").mkdir(parents=True)
        for i in range(4):
            (root / "dataset" / f"{key}_lora" / f"{key}_{i:02d}.png").write_bytes(b"png")
        (root / "videos").mkdir(exist_ok=True)
        loras, cfgdir, outdir = root / "loras", root / "cfgs", root / "out"
        loras.mkdir()
        job = outdir / f"{key}_qwen21_v1"
        job.mkdir(parents=True)
        (job / f"{key}_qwen21_v1.safetensors").write_bytes(b"lora")
        old = (ep.ROOT, ep.CONFIG, ep.CHAR_BIBLE)
        ep.ROOT = root
        ep.CONFIG = dict(ep.CONFIG, lora_training={
            "enabled": True, "python": "py.exe", "aitk_dir": str(root),
            "config_dir": str(cfgdir), "output_dir": str(outdir),
            "model": "m.safetensors", "extras": "e", "arch": "qwen_image_2",
            "steps": 10, "lr": 1e-4, "linear": 16, "linear_alpha": 16,
            "timeout_s": 60})
        ep.CHAR_BIBLE = {"characters": {key: {"danbooru_tags": "1boy, tall, dark coat"}}}
        try:
            with patch.object(ep, "_loras_dirs", return_value=[loras]), \
                 patch.object(ep, "free_comfyui_vram"), \
                 patch.object(ep, "restart_comfyui") as restart, \
                 patch.object(ep.subprocess, "run") as run:
                run.return_value = Mock(returncode=0)
                installed = ep.train_character_lora(key)
            ds = root / "dataset" / f"{key}_lora"
            self.assertEqual(len(list(ds.glob("*.png"))), 4,
                             "the staged training images are used as-is")
            self.assertEqual(len(list(ds.glob("*.txt"))), 4,
                             "a missing caption is written from the bible tags")
            self.assertEqual((ds / f"{key}_00.txt").read_text(encoding="utf-8").strip(),
                             "1boy, tall, dark coat")
            self.assertTrue((cfgdir / f"{key}_qwen21_v1.yaml").is_file())
            self.assertTrue((loras / f"{key}_qwen21_v1.safetensors").is_file(),
                            "the trained LoRA must land in the loras dir")
            self.assertEqual(installed, [loras / f"{key}_qwen21_v1.safetensors"])
            self.assertEqual(run.call_args[0][0][-1],
                             str(cfgdir / f"{key}_qwen21_v1.yaml"))
            self.assertTrue(restart.called,
                            "ComfyUI must restart so the new LoRA is listable")
        finally:
            ep.ROOT, ep.CONFIG, ep.CHAR_BIBLE = old

    def test_train_character_lora_fails_on_too_few_images(self):
        root, key = self.root, "thin_guy"
        (root / "dataset" / f"{key}_lora").mkdir(parents=True)
        (root / "dataset" / f"{key}_lora" / f"{key}_00.png").write_bytes(b"png")
        (root / "dataset").mkdir(exist_ok=True)
        old = (ep.ROOT, ep.CONFIG, ep.CHAR_BIBLE)
        ep.ROOT = root
        ep.CONFIG = dict(ep.CONFIG, lora_training={
            "enabled": True, "python": "py.exe", "aitk_dir": str(root),
            "config_dir": str(root / "cfgs"), "output_dir": str(root / "out"),
            "model": "m", "extras": "e"})
        ep.CHAR_BIBLE = {"characters": {key: {"danbooru_tags": "1boy"}}}
        try:
            with patch.object(ep.subprocess, "run") as run:
                with self.assertRaises(RuntimeError):
                    ep.train_character_lora(key)
            run.assert_not_called()
        finally:
            ep.ROOT, ep.CONFIG, ep.CHAR_BIBLE = old

    def test_ensure_cast_loras_skips_installed_disabled_and_never_raises(self):
        with patch.object(ep, "_lora_installed", return_value=True), \
             patch.object(ep, "train_character_lora") as train:
            ep.ensure_cast_loras(["someone"])
            train.assert_not_called()
        old = ep.CONFIG
        ep.CONFIG = dict(old, lora_training={"enabled": False})
        try:
            with patch.object(ep, "_lora_installed", return_value=False), \
                 patch.object(ep, "train_character_lora") as train:
                ep.ensure_cast_loras(["someone"])
                train.assert_not_called()
        finally:
            ep.CONFIG = old
        with patch.object(ep, "_lora_installed", return_value=False), \
             patch.object(ep, "train_character_lora", side_effect=RuntimeError("boom")):
            ep.ensure_cast_loras(["someone"])  # must warn, never abort the render

    def test_prompt_teaches_show_then_tell_and_beat_weight(self):
        # The two adaptation techniques from the LN→anime research: the image
        # carries what can be seen, narration carries what cannot, and the big
        # turns get more scenes than the small beats.
        import inspect
        src = inspect.getsource(ep)
        self.assertIn("SHOW, THEN TELL", src)
        self.assertIn("WEIGHT THE BEATS", src)


if __name__ == "__main__":
    unittest.main()
