"""Regression test: the purge must keep a kept still's PROVENANCE.

A still whose sidecar validated was kept by the purge, but its sidecar was
deleted (the keep_valid branch only recognised .png and .mp4). The next pass
then found the PNG with no note, called it "unverified", kept it and
re-stamped it with the CURRENT fingerprint - so the fingerprint could never
invalidate it again and a novel's new art direction silently stopped
reaching the stills.

    python -m unittest test_sheet_provenance
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import episode_producer as ep


class PurgeKeepsProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.imgs = self.root / "images"
        self.imgs.mkdir()
        (self.root / "voice").mkdir()
        (self.root / "_temp").mkdir()
        self.cfg = dict(ep.CONFIG, images_dir=str(self.imgs))
        for name, value in (("CONFIG", self.cfg), ("ROOT", self.root)):
            p = patch.object(ep, name, value)
            p.start()
            self.addCleanup(p.stop)

    def _still(self, scene, name="ch001_s01_a.png"):
        p = self.imgs / name
        p.write_bytes(b"png" * 4000)
        ep._write_still_sidecar(p, scene)
        return p

    def test_a_kept_still_keeps_its_sidecar(self):
        scene = {"scene_number": 1, "characters": ["hero"], "narration": "x"}
        with patch.object(ep, "CHAR_BIBLE", {"characters": {"hero": {}}}):
            still = self._still(scene)
            side = still.with_suffix(".cache.json")
            self.assertTrue(side.is_file())
            ep.clean_chapter(1, keep_valid=True, scenes_cache=[scene])
            self.assertTrue(still.is_file(), "a valid still must be kept")
            self.assertTrue(side.is_file(),
                            "a kept still must keep its provenance - without it "
                            "the next pass launders a stale frame into "
                            "'verified' and the fingerprint can never "
                            "invalidate it again")

    def test_an_invalid_still_is_purged_with_its_sidecar(self):
        scene = {"scene_number": 1, "characters": ["hero"], "narration": "x"}
        with patch.object(ep, "CHAR_BIBLE", {"characters": {"hero": {}}}):
            still = self._still(scene)
            side = still.with_suffix(".cache.json")
            # a changed scene makes the sidecar stale
            ep.clean_chapter(1, keep_valid=True, scenes_cache=[
                {"scene_number": 1, "characters": ["villain"], "narration": "y"}])
            self.assertFalse(still.is_file(), "a stale still must be purged")
            self.assertFalse(side.is_file())

    def test_a_changed_bible_invalidates_a_kept_still(self):
        """The whole point: a novel's look or bible change must reach the
        stills, not be laundered away by the purge."""
        scene = {"scene_number": 1, "characters": ["hero"], "narration": "x"}
        with patch.object(ep, "CHAR_BIBLE", {"characters": {"hero": {}}}):
            still = self._still(scene)
            with patch.object(ep, "CHAR_BIBLE",
                              {"characters": {"hero": {"danbooru_tags": "1boy, red hair"}}}):
                ep.clean_chapter(1, keep_valid=True, scenes_cache=[scene])
            self.assertFalse(still.is_file(),
                             "a changed bible must invalidate the still")

    def test_a_novel_that_owns_its_look_refuses_an_unprovenanced_still(self):
        """A still with no note cannot be shown to carry the book's look.

        Adopting it stamped the note with the CURRENT fingerprint, so the
        frame claimed to have been made under the new art direction and the
        art direction could never reach it again. A novel that owns its look
        must redraw instead of adopting."""
        scene = {"scene_number": 1, "characters": ["hero"], "narration": "x"}
        still = self.imgs / "ch001_s01_a.png"
        still.write_bytes(b"png" * 4000)   # no sidecar at all
        with patch.object(ep, "CHAR_BIBLE", {"characters": {"hero": {}}}), \
                patch.object(ep, "_novel_art",
                             lambda: {"still": "gothic revival victorian"}):
            self.assertFalse(ep._still_cache_valid(still, scene),
                             "an unprovenanced still cannot prove the look")
            self.assertFalse(ep._still_cache_valid(still, scene, adopt=True),
                             "adopting it would launder it into 'verified'")
            self.assertFalse(still.with_suffix(".cache.json").is_file(),
                             "no note may be synthesised for it")

    def test_a_novel_without_its_own_look_still_adopts(self):
        """The legacy library must not be re-rendered over a missing note."""
        scene = {"scene_number": 1, "characters": ["hero"], "narration": "x"}
        still = self.imgs / "ch001_s01_a.png"
        still.write_bytes(b"png" * 4000)
        with patch.object(ep, "CHAR_BIBLE", {"characters": {"hero": {}}}), \
                patch.object(ep, "_novel_art", lambda: {}):
            self.assertTrue(ep._still_cache_valid(still, scene))
            self.assertTrue(ep._still_cache_valid(still, scene, adopt=True))
            self.assertTrue(still.with_suffix(".cache.json").is_file())


if __name__ == "__main__":
    unittest.main()
