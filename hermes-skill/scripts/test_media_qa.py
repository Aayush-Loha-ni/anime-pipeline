import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

import episode_producer as ep


class MediaQATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(ep.CONFIG, output_dir=str(self.root / "videos"),
                           images_dir=str(self.root / "images"), music_dir=str(self.root / "music"),
                           hook_enabled=False, subtitles=False, fps=24, voice_speed=1.5,
                           voice_speeds={"test_character": 0.8}, qa_auto_editor=False)
        for folder in ("videos", "images", "voice", "music", "_temp"):
            (self.root / folder).mkdir()
        for name, value in (("ROOT", self.root), ("CONFIG", self.config), ("_LAST_ASSEMBLY", {})):
            p = patch.object(ep, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.audio = self.root / "voice" / "ch001_s01.wav"
        self.samples = (np.random.default_rng(7).normal(0, 0.05, 32000)).astype(np.float32)
        with wave.open(str(self.audio), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes((self.samples * 32767).astype("<i2").tobytes())

    def test_natural_wav_duration_not_divided_by_speed(self):
        def probe(cmd, **kwargs):
            with wave.open(cmd[-1], "rb") as f:
                return Mock(returncode=0, stdout=str(f.getnframes() / f.getframerate()))
        scenes = [{"characters": ["test character"]}, {}]
        with patch.object(ep.subprocess, "run", side_effect=probe):
            durations = ep.compute_scene_durations(scenes, [self.audio, self.audio])
        # 2s of narration + the 0.15s lead floor + NARRATION_GAP, snapped to the
        # frame grid. The slot is no longer lead==tail: the tail is pinned to
        # NARRATION_GAP so the pause between sentences is that value and nothing
        # else, and the 2.5s scene minimum is absorbed by the lead-in instead.
        # 2.0s of narration + NARRATION_GAP, snapped up to the frame grid:
        # ceil(2.20 * 24) / 24 = 53/24 = 2.2083s.
        #
        # The slot is exactly narration + gap. The lead is FIXED at NARRATION_LEAD
        # rather than centring, because the audible gap is
        # (slot - lead - audio) + NEXT lead - so a lead that absorbs slack lands
        # straight in the pause. Centring plus the old 2.5s floor is what produced
        # a measured 0.95s gap between sentences on ch001.
        self.assertEqual(durations, [53 / 24, 53 / 24], "slot must not depend on speed")
        for s in scenes:
            self.assertGreaterEqual(s["_dur"], s["_lead"] + 2)
            # the audible gap = tail + the NEXT scene's lead-in
            gap = (s["_dur"] - s["_lead"] - 2) + s["_lead"]
            self.assertAlmostEqual(gap, ep.NARRATION_GAP, delta=1.0 / 24 + 1e-9,
                                   msg="the pause between sentences is {}s, "
                                       "expected {}s".format(gap, ep.NARRATION_GAP))
            self.assertGreaterEqual(s["_lead"], ep.NARRATION_LEAD)

    def fake_encode(self, cmd, **kwargs):
        self.assertFalse(any("atempo" in str(arg) for arg in cmd))
        if cmd[0] == "ffprobe":
            if "-show_streams" in cmd:
                w, h = ep.get_output_dims()
                return Mock(returncode=0, stdout=json.dumps({"streams": [
                    {"codec_type": "video", "duration": "3", "avg_frame_rate": "24/1",
                     "width": w, "height": h},
                    {"codec_type": "audio", "duration": "3", "channels": 2}]}))
            return Mock(returncode=0, stdout="2" if "stream=channels" in cmd else "3")
        af = " ".join(str(a) for a in cmd)
        if "loudnorm" in af and "print_format=json" in af:
            return Mock(returncode=0, stdout="",
                        stderr='{"input_i": "-16.0", "input_tp": "-3.0",'
                               ' "input_lra": "7.0", "input_thresh": "-26.0",'
                               ' "target_offset": "0.0"}')
        if "loudnorm" in af and "measured_I" in af:
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out), "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes((np.zeros(44100, dtype="<i2")).tobytes())
            return Mock(returncode=0, stdout=b"", stderr=b"")
        if cmd[-1] == "-":
            return Mock(returncode=0, stdout=self.samples.tobytes(), stderr="")
        Path(cmd[-1]).write_bytes(b"new candidate")
        return Mock(returncode=0, stdout=b"", stderr=b"")

    def test_invalid_config_value_clamped_not_inherited(self):
        # _validate_config class: a bad value must clamp to its own
        # range, not inherit the previous schema key's coerced value
        # (stale `coerced`) or raise UnboundLocalError on first key.
        cfg = {"fps": "bad", "voice_speed": "fast"}
        ep._validate_config(cfg)
        self.assertEqual(cfg["fps"], 8)
        self.assertEqual(cfg["voice_speed"], 0.5)

    def test_rotate_versions_keeps_two_generations(self):
        # preserve path (used by _promote_media): current->v1, old v1->v2.
        # Old order archived first then shifted, so v1 never survived.
        out = self.root / "videos" / "ch001_youtube.mp4"
        out.write_bytes(b"new")
        (self.root / "videos" / "ch001_youtube.v1.mp4").write_bytes(b"old")
        ep.rotate_versions(out, preserve_current=True)
        self.assertEqual(out.read_bytes(), b"new")
        self.assertEqual((self.root / "videos" / "ch001_youtube.v1.mp4").read_bytes(), b"new")
        self.assertEqual((self.root / "videos" / "ch001_youtube.v2.mp4").read_bytes(), b"old")

    def test_av_sync_tolerates_rounded_timeline_edge(self):
        # Scene-25 class: 2-dp _cap_dur end exceeds the mix by ~2ms with
        # zero A/V error. Full-precision _dur + tol must pass, not crash.
        sr = 16000
        dur, lead, vlen = 2.041666666666667, 0.2, 1.5
        rng = np.random.default_rng(25)
        voice = rng.normal(0, 0.3, int(vlen * sr)).astype(np.float64)
        mix = np.zeros(int((dur - 0.004) * sr))
        mix[int(lead * sr):int(lead * sr) + len(voice)] += voice
        vw = self.root / "s25.wav"
        mw = self.root / "mix25.wav"
        for p, s in ((vw, voice), (mw, mix)):
            with wave.open(str(p), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sr)
                f.writeframes((np.clip(s, -1, 1) * 32767).astype("<i2").tobytes())
        scene = {"scene_number": 25, "_cap_start": 0.0, "_cap_dur": 2.04,
                 "_dur": dur, "_lead": lead}
        res = ep.qa_av_sync(mw, [scene], [vw])
        self.assertTrue(res["passed"], res)
        self.assertEqual(res["violations"], [])

    def test_assembly_qa_failure_preserves_deliverable_and_versions(self):
        output = self.root / "videos" / "ch001_youtube.mp4"
        output.write_bytes(b"good")
        version = output.with_name("ch001_youtube.v1.mp4")
        version.write_bytes(b"older")
        clip = self.root / "clip.mp4"
        clip.write_bytes(b"clip")
        with patch.object(ep.subprocess, "run", side_effect=self.fake_encode), \
                patch.object(ep, "qa_av_sync", return_value={"passed": False, "violations": [[1, "tail"]],
                                                            "skipped": [], "errors": [], "checked": [1]}), \
                patch.object(ep, "qa_auto_editor", return_value=[]):
            with self.assertRaises(RuntimeError):
                ep.assemble_video_v2([clip], [{"_dur": 3, "_lead": 0.35}], [self.audio], 1, None)
        self.assertEqual(output.read_bytes(), b"good")
        self.assertEqual(version.read_bytes(), b"older")
        self.assertEqual(list(output.parent.glob("*.candidate.mp4")), [])

    def test_echo_frames_cover_scene_slot(self):
        def echo(cmd, **kwargs):
            self.assertFalse(any(arg == "5000" for arg in cmd), "echo frames must be capped at service max")
            self.assertFalse(any("adelay=0" in str(arg) for arg in cmd))
            self.assertFalse(any("adelay" in str(arg) for arg in cmd), "lead padding must already be in the staged wav")
            self.assertFalse(any("stream_loop" in str(arg) for arg in cmd))
            return self.fake_encode(cmd, **kwargs)
        scene = {"_dur": 4.5, "_lead": 0.4, "characters": ["a"], "narration": "hello"}
        self.assertEqual(ep._echo_frames(scene), 108)
        self.assertEqual(ep._echo_frames({**scene, "_dur": 500}), 5000)
        self.assertEqual(ep._echo_frames({**scene, "_dur": 0.1}), 12)
        with patch.object(ep, "http_post_json", return_value={"prompt_id": "p"}), \
                patch.object(ep, "http_get", return_value={"p": {"outputs": {"6": {"images": [{"filename": "e.png"}]}}}}), \
                patch.object(ep, "find_comfy_file", return_value=self.root / "e.png"), \
                patch.object(ep, "frames_to_clip", side_effect=lambda src, dst, **kw: True), \
                patch.object(ep, "stage_comfy_input", side_effect=lambda p: str(p)), \
                patch.object(ep.subprocess, "run", side_effect=echo):
            self.assertTrue(ep.generate_echomimic_clip("still.png", "voice.wav", scene, 1))

    def test_stop_prevents_publish(self):
        output = self.root / "videos" / "episode.mp4"
        output.write_bytes(b"last good")
        final = self.root / "_temp" / "assembled.mp4"
        final.write_bytes(b"cancelled")
        with patch.object(ep, "should_stop", return_value=True), \
                patch.object(ep, "qa_true_peak", return_value={"passed": True, "peak_dbtp": -3}), \
                patch.object(ep, "qa_av_sync", return_value={"passed": True, "checked": [1]}), \
                patch.object(ep, "qa_auto_editor", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "[Cc]ancel"):
                ep._promote_media(final, output, [{"_dur": 3}], [self.audio])
        self.assertEqual(output.read_bytes(), b"last good")
        self.assertFalse(output.with_name("episode.v1.mp4").exists())
    def test_full_assembly_success_promotes_deliverable(self):
        still = self.root / "images" / "st.png"
        subprocess.run([self.config["ffmpeg"], "-y", "-v", "error", "-f", "lavfi",
                        "-i", "testsrc2=size=320x180:rate=1", "-frames:v", "1", str(still)],
                       check=True, timeout=30)
        clip = self.root / "images" / "ch001_wan01.mp4"
        # a real mp4, built the way the pipeline consumes it — this test
        # pins the assembly contract, not one specific clip generator.
        subprocess.run([self.config["ffmpeg"], "-y", "-v", "error", "-f", "lavfi",
                        "-i", "testsrc2=size=320x180:rate=24:duration=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
                       check=True, timeout=30)
        with patch.object(ep, "qa_auto_editor", return_value=None):
            out = ep.assemble_video_v2(
                [clip, clip], [{"_dur": 3.0, "_lead": 0.35, "scene_number": n,
                                "narration": "line", "characters": []} for n in (1, 2)],
                [self.audio, self.audio], 1, None)
        self.assertTrue(out and Path(out).is_file())
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration:stream=codec_name", "-of", "csv=p=0", str(out)],
                               capture_output=True, text=True, timeout=60)
        self.assertEqual(probe.returncode, 0)
        lines = probe.stdout.strip().splitlines()
        self.assertIn("h264", [line.split(",")[0] for line in lines])
        self.assertIn("aac", [line.split(",")[0] for line in lines])
        durations = [float(line) for line in lines if line.replace(".", "").isdigit()]
        self.assertTrue(durations)
        self.assertAlmostEqual(durations[-1], 6.0, delta=0.1)
        self.assertEqual(self.root / "videos" / "ch001_youtube.mp4", Path(out))
        self.assertTrue(ep._LAST_ASSEMBLY.get("promoted"))
        self.assertTrue(ep._LAST_ASSEMBLY["true_peak"]["passed"])
        self.assertTrue(ep._LAST_ASSEMBLY["qa_av_sync"]["passed"])
        self.assertEqual(ep._LAST_ASSEMBLY["qa_auto_editor"], "skipped")


    def test_a_failed_caption_burn_is_visible(self):
        """Losing the caption re-encode costs a full re-mux to discover, so it
        must not be a silent fallback to the caption-less master.

        It was the only one of four sibling failures in assemble_video_v2 with
        no `else`: the loudnorm failure logs [FAIL] and aborts, the mux failure
        logs and returns None, and the music mix logs AND sets
        _LAST_ASSEMBLY["music_mix_failed"]. The caption burn did neither — rc!=0
        fell through, and the receipt had no captions key at all, so the export
        and the release report looked clean.
        """
        srt = self.root / "_temp" / "ch001.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n",
                       encoding="utf-8")
        self.config["subtitles"] = True
        clip = self.root / "clip.mp4"
        clip.write_bytes(b"clip")

        def fail_burn(cmd, **kwargs):
            if any("subtitles=" in str(a) for a in cmd):
                return Mock(returncode=1, stdout="", stderr="libass: font not found")
            return self.fake_encode(cmd, **kwargs)

        said = []
        with patch.object(ep.subprocess, "run", side_effect=fail_burn), \
                patch.object(ep, "log", side_effect=lambda m, p=0: said.append(str(m))), \
                patch.object(ep, "qa_av_sync", return_value={"passed": True, "checked": [1]}), \
                patch.object(ep, "qa_auto_editor", return_value=[]):
            ep.assemble_video_v2([clip], [{"_dur": 3, "_lead": 0.35}],
                                 [self.audio], 1, str(srt))
        self.assertIs(ep._LAST_ASSEMBLY["captions_burned"], False,
                      "the receipt must record the failed burn; the export "
                      "and release report read this key")
        # the receipt survives into the EDL; this is what a human sees live
        self.assertTrue(any("caption" in m.lower() and "fail" in m.lower()
                            for m in said),
                        f"nothing was logged for the lost caption burn: {said}")

    def test_a_successful_caption_burn_is_recorded(self):
        srt = self.root / "_temp" / "ch001.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n",
                       encoding="utf-8")
        self.config["subtitles"] = True
        clip = self.root / "clip.mp4"
        clip.write_bytes(b"clip")
        with patch.object(ep.subprocess, "run", side_effect=self.fake_encode), \
                patch.object(ep, "qa_av_sync", return_value={"passed": True, "checked": [1]}), \
                patch.object(ep, "qa_auto_editor", return_value=[]):
            ep.assemble_video_v2([clip], [{"_dur": 3, "_lead": 0.35}],
                                 [self.audio], 1, str(srt))
        self.assertIs(ep._LAST_ASSEMBLY["captions_burned"], True)

    def test_no_caption_attempt_is_distinguishable_from_a_failure(self):
        """None, not False — a false receipt would claim a burn was tried."""
        clip = self.root / "clip.mp4"
        clip.write_bytes(b"clip")
        with patch.object(ep.subprocess, "run", side_effect=self.fake_encode), \
                patch.object(ep, "qa_av_sync", return_value={"passed": True, "checked": [1]}), \
                patch.object(ep, "qa_auto_editor", return_value=[]):
            ep.assemble_video_v2([clip], [{"_dur": 3, "_lead": 0.35}],
                                 [self.audio], 1, None)
        self.assertIsNone(ep._LAST_ASSEMBLY["captions_burned"])


if __name__ == "__main__":
    unittest.main()
