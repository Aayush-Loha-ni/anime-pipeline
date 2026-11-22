"""Regression checks for the stills quality fixes (framing + prompt terms).

Each test fails if the defect the VLM reported comes back:
blur/soft focus, blue colour cast, banding, or a tiny un-sized full-body face.
"""
import unittest

import episode_producer as ep


def _prompt_for(scene):
    """Run the real prompt builder and return (positive, negative)."""
    captured = {}

    class _Stop(Exception):
        pass

    def fake_post(url, payload, timeout=None):
        wf = payload["prompt"]
        for node in wf.values():
            if node.get("class_type") == "TextEncodeQwenImage21":
                captured["pos"] = node["inputs"]["prompt"]
                captured["neg"] = node["inputs"].get("negative_prompt", "")
        raise _Stop()  # skip the render/poll loop; we only want the prompt

    real_post = ep.http_post_json
    ep.http_post_json = fake_post
    try:
        ep.generate_image(dict(scene), 0, 1)
    except _Stop:
        pass
    except Exception:
        pass
    finally:
        ep.http_post_json = real_post
    return captured.get("pos", ""), captured.get("neg", "")


BASE = {
    "scene_number": 1,
    "narration": "A boy dances in the rain.",
    "image_prompt": "boy dancing in a neon alley, heavy rain",
    "characters": ["edvyn_crael"],
}


class FramingTests(unittest.TestCase):
    def test_unsized_shot_defaults_to_medium_not_full_body(self):
        # 40% of cached scenes carry no shot_size; those used to fall into the
        # full-body branch and produced a ~90px face.
        pos, _ = _prompt_for(dict(BASE))
        self.assertNotIn("full body", pos.lower(),
                         "un-sized shot must not fall back to full body")
        self.assertIn("medium shot", pos.lower())

    def test_explicit_wide_still_gets_full_body(self):
        pos, _ = _prompt_for(dict(BASE, shot_size="wide"))
        self.assertIn("full body", pos.lower())

    def test_explicit_close_up_still_wins(self):
        pos, _ = _prompt_for(dict(BASE, shot_size="close up"))
        self.assertIn("close-up portrait", pos.lower())


class PromptTermTests(unittest.TestCase):
    def test_positive_asks_for_sharpness_and_neutral_grade(self):
        pos, _ = _prompt_for(dict(BASE, shot_size="close-up"))
        low = pos.lower()
        self.assertIn("sharp focus", low)
        self.assertIn("clean neutral colour grade", low)
        # "selective focus" reads as a blur request and was in the positive block
        self.assertNotIn("selective focus", low)
        self.assertNotIn("film grain", low)

    def test_negative_covers_soft_focus_and_colour_cast(self):
        _, neg = _prompt_for(dict(BASE, shot_size="close-up"))
        low = neg.lower()
        for term in ("blurry", "out of focus", "hazy", "bloom",
                     "colour cast", "underexposed", "banding", "oversaturated"):
            self.assertIn(term, low, f"negative missing {term!r}")


class MotionPromptTests(unittest.TestCase):
    def test_no_default_camera_push_in(self):
        # The old default made every un-hinted scene push in, and H3 read it
        # literally (rain-dance test ended on a face close-up).
        offenders = [ln.strip() for ln in
                     open(ep.__file__, encoding="utf-8").read().splitlines()
                     if "animation_hint" in ln and "slow cinematic push in" in ln]
        self.assertEqual(offenders, [],
                         f"default camera push-in still present: {offenders}")


class AnchorCropTests(unittest.TestCase):
    """REMOVED with the character sheets: there is no anchor crop any more.

    The still path is pure T2I — identity comes from the prompt's tags and
    the character's LoRA. The crop geometry that used to keep the head
    inside an anchored frame guarded code that no longer exists, so the
    tests went with it (a test that guards nothing is worse than no test).
    """


class QwenIdentitySourceTests(unittest.TestCase):
    """Identity comes from the text prompt and the character's LoRA - never I2I.

    Measured on ch001: five consecutive the_man_with_the_box scenes all
    anchored from the same sheet front at denoise 0.7 and came back at pHash
    distance 0-2 - one picture repeated, which is what tripped the
    duplicate-still gate and blocked the render. At that denoise the anchor
    outvotes the prompt (no box in the box scene, no vials in the vials scene)
    and drags the sheet's flat backdrop in. The bible's danbooru_tags already
    carry identity, so pure T2I keeps the character AND obeys the shot.
    The sheet anchor has since been removed outright.
    """

    def _src(self):
        import inspect
        return inspect.getsource(ep.generate_image)

    def test_sheet_anchor_is_scoped_to_sheet_only_characters(self):
        """Owner rule 2026-11-18: LoRA characters stay pure T2I; only a
        low-appearance character with a sheet and NO LoRA gets an I2I anchor.
        """
        src = self._src()
        self.assertIn("_sheet_identity_phrase", src, "sheet identity is gone")
        self.assertNotIn("LoadImage", src, "no image may enter the still workflow")
        self.assertNotIn("ReferenceLatent", src, "no reference panel in frame")
        from unittest.mock import patch
        with patch.object(ep, "_qwen_lora_installed", return_value=True):
            self.assertEqual(
                ep._sheet_identity_phrase({"focal_character": "alaric_zoroast"}), "",
                "a LoRA character must never get sheet identity words")

    def _legacy_anchor_assertions_removed(self):
        src = self._src()
        self.assertNotIn("_anchor", src,
                         "the sheet anchor is gone - no anchor code may remain")
        self.assertNotIn("LoadImage", src,
                         "the still workflow must stay pure T2I")
        self.assertNotIn("VAEEncode", src,
                         "an anchor image must not be encoded into the latent")

    def test_lora_trigger_only_when_a_lora_actually_loaded(self):
        """`qwen_lora_trigger` is ONE character's token (ohwx_edvyn). Appended
        to a scene with no LoRA behind it, it dragged every LoRA-less character
        toward Edvyn's face."""
        lines = [ln for ln in self._src().splitlines()]
        guard = next((i for i, ln in enumerate(lines)
                      if ln.strip() == 'if "20" in _extra:'), None)
        self.assertIsNotNone(guard, "the trigger guard on the LoRA node is gone")
        assigns = [i for i, ln in enumerate(lines)
                   if 'CONFIG.get("qwen_lora_trigger"' in ln]
        self.assertEqual(len(assigns), 1, "expected exactly one trigger assignment")
        self.assertGreater(assigns[0], guard, "trigger assignment moved above its guard")
        self.assertGreater(len(lines[assigns[0]]) - len(lines[assigns[0]].lstrip()),
                           len(lines[guard]) - len(lines[guard].lstrip()),
                           "the trigger assignment must be nested inside the guard, "
                           "or it fires for scenes with no LoRA loaded")


    def test_duplicate_gate_needs_pixel_confirmation_not_just_the_hash(self):
        """The hash alone must not be able to block a render.

        find_duplicate_stills used to report any pair whose 64-bit aHash was
        within 4, on the assumption that random pairs sit ~32 apart. Measured
        across ch001/2/6/7 that is false: two DIFFERENT stills of the same
        character in the same street average 12.3, because the frames are not
        independent samples. So the gate flagged 25 distinct pictures of ch007
        (closest pixel RMS 39.1), re-rolled them, made the count worse, and
        crashed - and it also fired on ch006, which had already shipped.

        It must still catch a REAL repeat, so this asserts both directions:
        an identical copy is flagged, and a similar-looking-but-different
        picture is not.
        """
        import os
        import tempfile
        import numpy as np
        from PIL import Image

        # An 8x8 grid of blocks, so the layout SURVIVES the 8x8 downscale that
        # _phash does. Pure noise does not: averaging kills its variance, the
        # mean threshold lands on numerical noise, and the hash is unstable even
        # under a +6 exposure shift - which would make this test pass or fail
        # for reasons unrelated to the gate.
        rng = np.random.default_rng(7)
        grid = rng.integers(0, 256, size=(8, 8)).astype("uint8")
        plane = np.kron(grid, np.ones((32, 32), dtype="uint8"))
        base = np.repeat(plane[:, :, None], 3, axis=2)

        with tempfile.TemporaryDirectory() as td:
            def save(name, arr):
                p = os.path.join(td, name)
                Image.fromarray(arr).save(p)
                return p

            # 1. a genuine duplicate: same picture twice -> MUST be flagged
            a = save("a.png", base)
            a_copy = save("a_copy.png", base)
            self.assertEqual(
                len(ep.find_duplicate_stills([a, a_copy])), 1,
                "an identical still must still be caught as a duplicate")

            # 2. the same picture re-encoded with a mild exposure shift is still
            #    a duplicate - this is the "25x static" incident the gate exists
            #    for, and the case the fix must not lose. Verified: hash 0,
            #    RMS 5.87.
            lifted = np.clip(base.astype("int16") + 6, 0, 255).astype("uint8")
            b = save("b_lifted.png", lifted)
            self.assertTrue(
                ep.find_duplicate_stills([a, b]),
                "a re-encoded near-copy must still be caught")

            # 3. THE REGRESSION: a close hash with genuinely different pixels.
            #    Inverting two of the 64 blocks leaves the coarse signature
            #    nearly intact (verified hash distance 4) while the picture is
            #    plainly different (RMS 43.6). This is the ch007 shape - and
            #    the old single-stage gate blocked on it.
            other = plane.copy()
            for idx in (13, 33):
                r0, c0 = idx // 8, idx % 8
                other[r0 * 32:(r0 + 1) * 32, c0 * 32:(c0 + 1) * 32] = \
                    255 - other[r0 * 32:(r0 + 1) * 32, c0 * 32:(c0 + 1) * 32]
            d = save("d_inverted.png", np.repeat(other[:, :, None], 3, axis=2))
            self.assertLessEqual(
                ep._phash_distance(ep._phash(a), ep._phash(d)), 4,
                "fixture is not exercising the close-hash case")
            self.assertGreater(
                ep._still_rms(a, d), 12.0,
                "fixture must differ by more than the confirmation threshold")
            self.assertEqual(
                ep.find_duplicate_stills([a, d]), [],
                "a close hash with genuinely different pixels must NOT block")

    def test_framing_clause_is_not_suppressed_by_the_prompt_lead(self):
        """The shot_size framing clause must be applied unconditionally.

        This pins a REVERT. A version of this block treated a disagreement
        between the prompt's framing words and shot_size as "the prompt is
        more specific, so the prompt wins" and dropped the framing clause on
        conflict. That looked like it fixed ch007's shot monotony, but the
        premise was false: _enforce_shot_diversity (episode_producer 3020-3031)
        strips the writer's own framing lead and injects FRAMING_POOL[i % 20]
        into image_prompt, while shot_size rotates independently via
        pools["shot_size"]. Both sides are mechanical and unrelated, so the
        keyword test fired on a random collision - it suppressed the face
        protection on 9 of 40 ch007 scenes and dropped "body shot without
        face", the term guarding the 90px-face bug, with nothing authorial
        behind it.

        So the clause is deterministic: shot_size in {close-up, medium} always
        gets a face-framing clause plus the torso negatives, and wide always
        gets full body, even when the prompt's own words disagree. The real
        contradiction is internal to the pipeline and belongs in
        _enforce_shot_diversity, which should pick a pool entry matching
        shot_size.
        """
        contradicting = dict(BASE)
        contradicting["shot_size"] = "close-up"
        contradicting["image_prompt"] = ("low angle full body shot, figure framed "
                                         "against the ceiling, maren_voss")
        pos, neg = _prompt_for(contradicting)
        self.assertIn("2d anime", pos, "prompt capture failed - assertions below are vacuous")
        self.assertIn("face fills most of frame", pos,
                      "a close-up scene must keep its framing clause even when the "
                      "prompt's own lead disagrees; the lead is a mechanical "
                      "FRAMING_POOL injection, not authorial intent")
        self.assertIn("body shot without face", neg,
                      "the face-protection negative must never be dropped - it guards "
                      "the documented 90px-face regression")

        wide = dict(BASE)
        wide["shot_size"] = "wide"
        wide["image_prompt"] = "tight close-up on the eyes, shallow depth of field"
        pos2, _neg2 = _prompt_for(wide)
        self.assertIn("2d anime", pos2, "prompt capture failed - assertion below is vacuous")
        self.assertIn("head to boots", pos2,
                      "a wide scene must keep its full-body clause for the same reason")

    def test_duplicate_stills_reroll_before_they_block(self):
        """The gate used to fail the render outright. Measured on ch001: five
        scenes drew consecutive seeds and all five landed on one portrait
        (distance 0-2), while the same prompt at seed+7000 was correct and
        varied (distance 27). A re-roll is the fix, so re-roll first."""
        import inspect
        src = inspect.getsource(ep._produce_inner)
        first = src.index("find_duplicate_stills(scene_images)")
        reroll = src.index("_reroll + si")
        gate = src.index("pixel-confirmed")
        self.assertLess(first, reroll, "re-roll must come after the first measurement")
        self.assertLess(reroll, gate, "re-roll must come before the gate that raises")
        self.assertIn("_dupes = find_duplicate_stills(scene_images)", src[reroll:],
                      "the gate must measure the RE-ROLLED stills, not the originals")

    def test_rerolled_still_is_staged_not_a_bare_comfy_filename(self):
        """generate_image returns a name that only exists in ComfyUI's output.

        Storing that return value straight into scene_images[] gives the clip
        phase a path to a file that was never copied into images/, so the still
        phase reports success and H3 staging raises FileNotFoundError an hour
        later. ch2 crashed twice on exactly this: the duplicate re-roll replaced
        scenes 14 and 16 with bare names and the render died on scene 16.

        Every caller that re-rolls must go through _reroll_still, which copies
        the file home and writes its sidecar.
        """
        import inspect
        src = inspect.getsource(ep._produce_inner)
        # no direct generate_image() -> scene_images[] assignment may survive
        self.assertNotIn("scene_images[si] = generate_image", src)
        self.assertEqual(
            src.count("_reroll_still("), 2,
            "expected the black-still regen and the duplicate re-roll to both "
            "stage through _reroll_still")
        # and the helper must actually copy into images_dir and write a sidecar
        helper = inspect.getsource(ep._reroll_still)
        self.assertIn("find_comfy_file", helper)
        self.assertIn("shutil.copy2", helper)
        self.assertIn("_write_still_sidecar", helper)
        self.assertIn("images_dir / name", helper)


    def test_no_clip_is_just_a_still_with_a_camera_move(self):
        """The user asked for clips with real movement — characters or
        background — never "a frame with effects on it". So every scene goes
        through H3 (real video), and if H3 cannot produce a clip the render
        STOPS rather than shipping a Ken Burns still.

        The still-pan path was the one this guards against, and the beat-2
        companion still only existed to feed it."""
        import inspect
        inner = inspect.getsource(ep._produce_inner)
        # every scene is a motion scene; nothing routes to a still-pan
        self.assertIn("motion_idx = set(range(len(scenes)))", inner,
                      "every scene must get real video")
        for gone in ("cinematic dual shot", "cinematic shot...",
                     "fallback to cinematic shot"):
            self.assertNotIn(gone, inner,
                             f"a still-with-camera-move path is back: {gone!r}")
        self.assertFalse(hasattr(ep, "create_dual_shot_clip"))
        self.assertFalse(hasattr(ep, "create_cinematic_shot"))
        self.assertFalse(hasattr(ep, "narration_beats"))

    def test_a_single_cached_still_is_not_put_to_the_vlm(self):
        """The resume tax. Every resume re-walked all 25 scenes and asked the
        VLM selector to choose among the cached stills — normally exactly ONE,
        at ~3 min a scene, ~75 min per resume, to return index 0 of 1. The
        selector's own failure path returns that same file, so the call is
        provably redundant. Real selection must still run for 2+ candidates."""
        import inspect
        src = inspect.getsource(ep._produce_inner)
        guard = src.index("if len(existing) == 1:")
        self.assertIn("scene_images[si] = existing[0]", src[guard:guard + 400],
                      "a lone cached still must be taken directly")
        sel = src.index("sel = _reference_selector()")
        self.assertLess(guard, sel, "the cheap branch must come before the selector")
        guarded = src[guard:sel]
        self.assertIn("elif vlm_available():", guarded,
                      "the selector must still run when 2+ candidates exist")

    def test_reused_clips_count_as_video(self):
        """A resumed render reported '0 video clips' with 14 clips on disk: the
        reuse branch `continue`d before video_count was touched."""
        import inspect
        src = inspect.getsource(ep._produce_inner)
        reuse = src.index('log(f"  Scene {scene_num}: video exists"')
        self.assertIn("video_count += 1", src[:reuse][-400:],
                      "the video-exists branch must count the clip as video")


class WorldLookTests(unittest.TestCase):
    """The bible's world has to reach the model on EVERY scene.

    The setting gate used to apply the environment only when the scene text
    happened to contain "white sky"/"city"/"fog". Every other scene got no
    world tags, and Qwen filled the gap with its own default — bright
    suburban daylight. That is how a flat-white-sky gothic city rendered with
    a blonde schoolgirl under a blue sky.
    """

    def test_world_is_a_default_not_a_keyword_opt_in(self):
        self.assertEqual(ep._setting_key_for("a silent street"), "white_sky_city",
                         "the copy of Vethmore is the default world")
        self.assertEqual(ep._setting_key_for("coal smoke over vethmore"), "vethmore",
                         "Vethmore must stay reachable as the explicit opt-out")

    def test_interior_scenes_get_the_room_tail_not_the_canal(self):
        """The white-sky tail names cobblestones and a canal; an interior
        scene carrying it renders the room out on the street (ch021 s6)."""
        for text in ("petra_vasek in the hallway",
                     "a dim room with a small cot and a cold stove",
                     "edvyn in the study, morning"):
            self.assertEqual(ep._setting_key_for(text), "interior", text)
        self.assertEqual(ep._setting_key_for("a silent street"), "white_sky_city")
        self.assertEqual(ep._setting_key_for("knock at the door"), "interior")
        # A named room beats the district in the director's location.
        self.assertEqual(ep._setting_key_for("", "Crown Quarter records office"),
                         "interior")
        self.assertEqual(ep._setting_key_for("", "Crown Quarter"), "white_sky_city")
        self.assertEqual(ep._setting_key_for("", "Horner Lane docks"), "white_sky_city")
        self.assertTrue(
            ep.CHAR_BIBLE.get("settings", {}).get("interior", {}).get("visual_tags"),
            "settings.interior.visual_tags missing from the bible")

    def test_director_location_reaches_the_scene(self):
        scenes = [{"scene_number": 1}]
        dp = {"scene_plan": [{"scene_number": 1, "focal_character": "edvyn_crael",
                              "location": "Edvyn's house - hallway"}]}
        ep._apply_director_plan(scenes, dp)
        self.assertEqual(scenes[0].get("location"), "Edvyn's house - hallway")

    def test_configured_style_has_a_negative(self):
        """`cinematic` ships an EMPTY negative, so nothing pushed back on bright
        daylight. Guard the property, not the value, so a future style swap
        cannot silently reopen this."""
        style = ep._animation_style({})
        self.assertTrue(ep.ANIMATION_STYLES[style]["negative"].strip(),
                        f"style {style!r} has no negative prompt, so bright/pastel "
                        f"frames are unguarded")

    def test_the_bible_world_tags_are_not_dead(self):
        for key in ("white_sky_city", "vethmore"):
            self.assertTrue(ep.CHAR_BIBLE.get("settings", {}).get(key, {}).get("visual_tags"),
                            f"settings.{key}.visual_tags missing from the bible")


if __name__ == "__main__":
    unittest.main()
