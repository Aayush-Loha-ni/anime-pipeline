"""Tests for the voice-clone tooling.

These guard the failures that actually happened, each of which cost real time:

  * the cache key and the server disagreeing about the clone mode, which ships
    stale audio that looks correct (the pre-existing ponytail note)
  * a fine-tuned checkpoint being served through generate_voice_clone, which
    raises, or through generate_custom_voice with no speaker, which silently
    ignores the whole training run
  * build_sft pointing the dataset at one reference while ref.json names
    another, which is how a song lyric got prefixed to every generated line
  * score_voice's WER being computed against the wrong expected text

Run:  python -m unittest test_voice_tools -v
"""
import inspect
import json
import pathlib
import re
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
COSY = HERE.parent.parent / "cosy-server"
sys.path.insert(0, str(COSY))

import score_voice  # noqa: E402


class RefJsonShapeTests(unittest.TestCase):
    """ref.json is the single source of truth for BOTH server and pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.ref = json.loads((COSY / "ref.json").read_text(encoding="utf-8"))

    def test_it_has_the_three_keys_the_model_needs(self):
        for key in ("wav", "text", "x_vector_only"):
            self.assertIn(key, self.ref, "ref.json is missing " + key)

    def test_the_reference_file_exists(self):
        self.assertTrue(pathlib.Path(self.ref["wav"]).is_file(),
                        "ref.json points at a file that does not exist")

    def test_the_transcript_is_not_empty(self):
        self.assertGreaterEqual(len(self.ref["text"].split()), 4,
                                "ICL mode needs a real transcript of the "
                                "reference audio")

    def test_the_server_and_the_cache_key_agree_on_the_mode(self):
        """The bug this whole refactor exists to prevent.

        Both used to hardcode the clone mode independently. Flipping one and
        not the other left every cached wav 'valid' while the server spoke with
        different settings - audio that is wrong and looks right.
        """
        import episode_producer as ep
        fp = ep._tts_fingerprint("test line", 1.0)
        self.assertEqual(fp["x_vector_only"], bool(self.ref["x_vector_only"]),
                         "episode_producer._tts_fingerprint disagrees with "
                         "ref.json about the clone mode; cached wavs will be "
                         "served with the wrong settings")
        self.assertEqual(fp["language"], self.ref.get("language", "English"),
                         "fingerprint and ref.json disagree on language")

    def test_flipping_the_mode_actually_changes_the_cache_key(self):
        import episode_producer as ep
        before = ep._tts_fingerprint("test line", 1.0)
        ref_path = COSY / "ref.json"
        original = ref_path.read_text(encoding="utf-8")
        try:
            flipped = dict(self.ref, x_vector_only=not self.ref["x_vector_only"])
            ref_path.write_text(json.dumps(flipped), encoding="utf-8", newline="")
            after = ep._tts_fingerprint("test line", 1.0)
            self.assertNotEqual(before, after,
                                "changing the clone mode must retire the cache, "
                                "or every previously-generated wav stays valid")
        finally:
            ref_path.write_text(original, encoding="utf-8", newline="")


class DatasetReferenceTests(unittest.TestCase):
    """Every training row must point at the reference that is actually in use.

    Datasets are NOT at a fixed path: each recording gets its own directory
    (train_data for Recording 32, train_data33 for Recording 33). So these tests
    find the dataset that belongs to the SHIPPED reference instead of hardcoding
    one, which is how the previous version of this test went stale the moment a
    second recording arrived and silently stopped testing anything.
    """

    @classmethod
    def _dataset_for_shipped_ref(cls):
        ref = json.loads((COSY / "ref.json").read_text(encoding="utf-8"))
        want = pathlib.Path(ref["wav"]).resolve()
        for d in sorted(COSY.glob("train_data*")):
            raw = d / "train_raw.jsonl"
            if not raw.is_file():
                continue
            rows = [json.loads(l) for l in
                    raw.read_text(encoding="utf-8").splitlines() if l.strip()]
            if rows and pathlib.Path(rows[0]["ref_audio"]).resolve() == want:
                return raw, rows, ref
        return None, None, ref

    def test_some_dataset_matches_the_shipped_reference(self):
        _, _, ref = self._dataset_for_shipped_ref()
        self.assertTrue(
            pathlib.Path(ref["wav"]).is_file(),
            "ref.json points at a reference that does not exist")
        raw, rows, _ = self._dataset_for_shipped_ref()
        if raw is None:
            # NOT a skip. A dataset that disagrees with the shipped reference is
            # how training silently regresses: build_sft.py, install_ref.py and
            # ref.json each track the reference independently. train_data33 was
            # pointing at ref33.wav while ref.json named ref34.wav, and every
            # voice-quality assertion below quietly stopped running.
            built = sorted(COSY.glob("train_data*"))
            self.fail(
                "no built dataset matches the shipped reference {} - {} exist but "
                "their train_raw.jsonl names a different ref_audio. Retrain rows "
                "or delete the stale dataset; do not skip.".format(
                    ref.get("wav"), [d.name for d in built]))

    def test_train_rows_all_reference_the_same_wav(self):
        raw, rows, _ = self._dataset_for_shipped_ref()
        if raw is None:
            self.skipTest("no dataset for the shipped reference")
        refs = {r["ref_audio"] for r in rows}
        self.assertEqual(len(refs), 1,
                         "upstream: keeping ref_audio identical across the "
                         "dataset improves speaker consistency. Found: "
                         + ", ".join(sorted(refs)))

    def test_train_rows_and_ref_json_agree(self):
        """The drift that put a lyric on every line.

        build_sft.py writes train_raw.jsonl and ref.json separately, so they can
        disagree. When they do, training conditions on one voice and inference
        conditions on another, and with ICL the model continues the WRONG
        utterance - observed as four words of a pop song prefixed to every line.
        """
        raw, rows, ref = self._dataset_for_shipped_ref()
        if raw is None:
            self.skipTest("no dataset for the shipped reference")
        want = pathlib.Path(ref["wav"]).resolve()
        for r in rows:
            self.assertEqual(pathlib.Path(r["ref_audio"]).resolve(), want,
                             "{} and ref.json name different reference audio"
                             .format(raw.parent.name))


class WerTests(unittest.TestCase):
    """WER is reported to the user, so it has to be right."""

    def test_perfect_match_is_zero(self):
        self.assertEqual(score_voice.wer("the sky was white", "the sky was white"), 0.0)

    def test_one_substitution_out_of_five(self):
        self.assertAlmostEqual(
            score_voice.wer("the sky is white", "the sky was white"), 0.25, places=4)

    def test_a_deletion_costs_one(self):
        self.assertAlmostEqual(
            score_voice.wer("the sky white", "the sky was white"), 0.25, places=4)

    def test_punctuation_and_case_do_not_count(self):
        self.assertEqual(score_voice.wer("The sky, was white.", "the sky was white"), 0.0)

    def test_empty_expected_text_is_refused_not_a_crash(self):
        # returning None keeps it out of the mean instead of dividing by zero
        self.assertIsNone(score_voice.wer("anything at all", ""))


class UtmosContractTests(unittest.TestCase):
    """UTMOS is the number a variant gets ranked on, so its contract matters."""

    def test_it_speaks_raw_waveform_not_features(self):
        """A plausible wrong number beats a crash, so pin the input contract.

        The exporter states: input wave, shape [batch, time], float32, 16 kHz
        mono, raw [-1,1], rescaled in-graph. No client-side log-mel. Feeding the
        log-mel this function used to build returns a number in a believable
        range, so nothing would flag it.
        """
        import inspect
        module_src = inspect.getsource(score_voice)
        self.assertNotIn("num_mel_bins=560", module_src,
                         "the client-side log-mel path is back; UTMOS wants raw "
                         "audio and a log-mel returns a plausible wrong score")
        self.assertNotIn("def _utmos_feat", module_src,
                         "_utmos_feat existed to compute a log-mel for UTMOS; "
                         "the ONNX export takes the waveform directly")
        fn_src = inspect.getsource(score_voice.utmos)
        self.assertIn("[None, :]", fn_src,
                      "input must be fed as a raw [batch, time] waveform")

    def test_short_clips_are_refused_by_default(self):
        import inspect
        src = inspect.getsource(score_voice)
        self.assertIn('default=4.0', src,
                      "min-dur guard must default to rejecting short clips")


class LeakProbeTests(unittest.TestCase):
    """A guard that cannot fail is not a guard.

    verify_voice.py's leak check used to hardcode the deleted SONG's lyrics
    ("meant to be", "little things"). Once the reference became spoken prose
    those strings could never appear, so the check passed forever while
    detecting nothing — and it still printed "no reference-lyric leak", which
    reads like evidence. It is now derived from ref.json as 5-grams, so these
    tests pin that it both fires and stays quiet.
    """

    @staticmethod
    def _grams(text):
        words = [w.lower().strip(".,!?;:'\"")
                 for w in re.findall(r"[A-Za-z']+", text or "")]
        return {tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))}

    def _detect(self, ref_text, got):
        g = self._grams(ref_text)
        spoken = re.findall(r"[A-Za-z']+", (got or "").lower())
        sg = {tuple(spoken[i:i + 5]) for i in range(max(0, len(spoken) - 4))}
        return g & sg

    def test_it_fires_on_the_lyric_leak_that_actually_happened(self):
        lyric = ("Your hand fits in mine like it's made just for me. "
                 "But bear this in mind, it was meant to be.")
        leak = "It was meant to be. The engine turned over twice."
        self.assertTrue(self._detect(lyric, leak),
                        "the probe must catch the exact defect it was written "
                        "for: a pop lyric prefixed to every generated line")

    def test_it_stays_quiet_on_clean_output(self):
        ref = json.loads((COSY / "ref.json").read_text(encoding="utf-8"))
        self.assertEqual(
            self._detect(ref["text"],
                         "The engine turned over twice and settled into a low "
                         "uneven idle."),
            set(), "false positive on unrelated speech")

    def test_verify_voice_derives_the_probe_rather_than_hardcoding_it(self):
        import verify_voice
        src = inspect.getsource(verify_voice)
        self.assertIn("ref.json", src,
                      "the leak probe must read ref.json; hardcoded lyrics from "
                      "the deleted song can never match a spoken reference")
        for lyric in ("little things", "your hand fits"):
            self.assertNotIn(lyric, src,
                             "the old song's lyrics are back in the probe")


class StaleReferenceTests(unittest.TestCase):
    """The server builds its prompt once; ref.json can change underneath it.

    If nothing checks, the pipeline retires its cache, regenerates every scene
    from the OLD prompt, and writes a fresh fingerprint over audio that does not
    match the config it claims to have been made with. Correct-looking files,
    wrong voice — the class of bug this repo has already paid for twice.
    """

    @classmethod
    def setUpClass(cls):
        import server
        cls.server = server
        cls.ref = json.loads((COSY / "ref.json").read_text(encoding="utf-8"))

    def test_health_reports_the_loaded_prompt_not_the_file(self):
        import inspect
        src = inspect.getsource(self.server.health)
        self.assertIn("loaded_ref", src,
                      "/health must report the fingerprint the prompt was "
                      "built from")
        self.assertIn("stale", src,
                      "/health must flag a ref.json that has drifted from the "
                      "loaded prompt, or the drift is invisible")

    def test_the_pipeline_refuses_a_stale_server(self):
        import episode_producer as ep
        src = inspect.getsource(ep.check_tts_server)
        self.assertIn("stale", src,
                      "check_tts_server must act on /health's stale flag; "
                      "status+loaded alone cannot see a changed reference")

    def test_the_fingerprint_covers_audio_transcript_and_mode(self):
        import inspect
        src = inspect.getsource(self.server._ref_fingerprint)
        for token, why in (("read_bytes", "the reference audio"),
                           ("text", "the transcript"),
                           ("x_vector_only", "the clone mode")):
            self.assertIn(token, src, "fingerprint ignores " + why)

    def test_changing_any_input_changes_the_fingerprint(self):
        f = self.server._ref_fingerprint
        base = f(self.ref)
        self.assertEqual(f(self.ref), base, "must be stable for identical config")
        self.assertNotEqual(f(dict(self.ref, text="different words")), base)
        self.assertNotEqual(
            f(dict(self.ref, x_vector_only=not self.ref["x_vector_only"])), base)


class DatasetIntegrityTests(unittest.TestCase):
    """The two artefacts that decide the voice must never disagree."""

    def test_the_entry_point_calls_the_builder_that_syncs_ref_json(self):
        bat = (COSY / "train_voice.bat").read_text(encoding="utf-8")
        invoked = [ln for ln in bat.splitlines()
                   if '"%PY%"' in ln and ln.rstrip().endswith("%* || exit /b 1")]
        self.assertTrue(invoked, "no dataset-building step found in train_voice.bat")
        for line in invoked:
            self.assertIn("build_sft.py", line,
                          "train_voice.bat must call build_sft.py, which "
                          "rewrites ref.json alongside the dataset; the deleted "
                          "make_dataset.py did not and re-created the drift")

    def test_the_dead_dataset_builder_is_gone(self):
        self.assertFalse((COSY / "make_dataset.py").exists(),
                         "make_dataset.py was superseded by build_sft.py and "
                         "deleted; if it is back, train_voice.bat may call it")

    def test_the_reference_text_is_not_the_old_song(self):
        ref = json.loads((COSY / "ref.json").read_text(encoding="utf-8"))
        lowered = ref["text"].lower()
        for lyric in ("little things", "meant to be", "fits in mine",
                      "freckles on your cheeks"):
            self.assertNotIn(lyric, lowered,
                             "ref.json still carries the pop lyric that "
                             "prefixed every generated line")


if __name__ == "__main__":
    unittest.main()