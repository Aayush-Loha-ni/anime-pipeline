import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import episode_producer as ep


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args],
                   check=True, timeout=120)


class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _png(self, name="s.png"):
        p = self.root / name
        _ffmpeg("-f", "lavfi", "-i",
                "testsrc=size=640x480:rate=1,noise=alls=30:allf=t",
                "-frames:v", "1", str(p))
        return p

    def _mp4(self, name="c.mp4", dur=2, black=False):
        p = self.root / name
        if black:
            # mostly-black but big: 2s content + 20s black (pure black alone
            # compresses under the tiny-file floor, which is also correct).
            _ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2",
                    "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10:d=20",
                    "-filter_complex", "[0][1]concat=n=2:v=1:a=0",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", str(p))
        else:
            _ffmpeg("-f", "lavfi", "-i",
                    f"testsrc=size=320x240:rate=10:duration={dur}",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", str(p))
        return p

    def test_clip_missing_rejected(self):
        ok, _ = ep.qa_clip_file(self.root / "nope.mp4")
        self.assertFalse(ok)

    def test_clip_good_accepted(self):
        self.assertEqual(ep.qa_clip_file(self._mp4()), (True, "ok"))

    def test_clip_all_black_rejected(self):
        ok, why = ep.qa_clip_file(self._mp4("black.mp4", black=True))
        self.assertFalse(ok)
        self.assertIn("black", why)

    def test_note_visual_review(self):
        with patch.object(ep, "vlm_available", return_value=True):
            self.assertFalse(ep.note_visual_review())
        import supervisor
        with patch.object(ep, "vlm_available", return_value=False), \
                patch.object(supervisor, "record") as rec:
            self.assertTrue(ep.note_visual_review())
            rec.assert_called_once()
            self.assertEqual(rec.call_args.args[0], "warn")

    def test_final_black_gate(self):
        with self.assertRaisesRegex(RuntimeError, "black"):
            ep.qa_final_black(self._mp4("black.mp4", black=True))
        self.assertEqual(ep.qa_final_black(self._mp4()), 0)

    def test_final_frames_skipped_without_vlm(self):
        with patch.object(ep, "vlm_available", return_value=False):
            self.assertIsNone(ep.qa_final_frames(self._mp4(), self.root))

    def test_final_frames_graded_when_vlm_up(self):
        with patch.object(ep, "vlm_available", return_value=True), \
                patch.object(ep, "vlm_grade_image",
                             return_value={"issues": ["torso only"]}) as g:
            res = ep.qa_final_frames(self._mp4(), self.root, n=2)
            self.assertEqual(res["checked"], 2)
            self.assertTrue(any("torso only" in i for i in res["issues"]))
            self.assertEqual(g.call_count, 2)


if __name__ == "__main__":
    unittest.main()
