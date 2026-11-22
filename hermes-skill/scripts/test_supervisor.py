#!/usr/bin/env python3
"""Tests for supervisor.py — the ledger that BLOCKS renders.

The ledger is the highest-leverage piece of state in the project: ep.preflight
refuses to render while any unresolved critical is newer than its 7-day triage
window. So both what it records and how it fails matter. Mutation testing found
the crash-contention fix untested (reverting it left the suite green), which is
what this file exists to close.
"""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import supervisor as sup


class SupervisorLedgerTests(unittest.TestCase):
    """The ledger is a RENDER BLOCKER, so what lands in it matters enormously.

    ponytail: 'Another render is already running' used to be filed as
    `critical: new crash: ...`. ep.preflight refuses to render while any
    unresolved critical is newer than its 7-day triage window, so one
    double-click on Start during a live render took the whole pipeline down for
    a week. A refusal is not a crash.
    """

    def setUp(self):
        self._root = sup.ROOT
        self._ledger = sup.LEDGER
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        sup.ROOT = Path(self.tmp.name)
        sup.LEDGER = sup.ROOT / "issues.json"

        def _restore():
            sup.ROOT = self._root
            sup.LEDGER = self._ledger
        self.addCleanup(_restore)

        # check() probes three live services over HTTP, retrying 3x with 5s
        # sleeps. Leaving that in made these tests take 40s and their result
        # depend on whether llama-server happened to be up. The crash-tail and
        # ledger behaviour under test has nothing to do with service health.
        self._svc = patch.object(sup.ep, "ensure_services",
                                 return_value=(True, {"llm": True, "comfyui": True,
                                                      "tts": True}))
        self._svc.start()
        self.addCleanup(self._svc.stop)

    def _seed_marker(self):
        (sup.ROOT / ".supervisor_crash_seen").write_text(
            "=== 1 2026-01-01 00:00:00 ===\nRuntimeError: something older\n",
            encoding="utf-8")

    def _check(self):
        # quiet=True collects findings WITHOUT writing the ledger, so the
        # crash-tail branch can be exercised on its own.
        return sup.check(fix=False, quiet=True)

    def test_contention_is_info_never_critical(self):
        self._seed_marker()
        (sup.ROOT / "render_crash.log").write_text(
            "=== 1 2026-01-01 00:00:00 ===\n"
            "RuntimeError: Another render is already running (render.lock is "
            "held by another process)\n", encoding="utf-8")
        found = self._check()
        crashish = [e for e in found if "lock" in e["symptom"].lower()
                    or "crash" in e["symptom"].lower()]
        self.assertTrue(crashish, "expected the crash-tail branch to report")
        for e in crashish:
            self.assertEqual(e["severity"], "info",
                             f"contention filed as {e['severity']}: {e}")
            self.assertNotIn("crash", e["symptom"].lower())

    def test_a_real_crash_is_still_critical(self):
        """The mitigation must not have blunted the actual signal."""
        self._seed_marker()
        (sup.ROOT / "render_crash.log").write_text(
            "=== 7 2026-01-01 00:00:00 ===\n"
            "RuntimeError: ComfyUI died\nTraceback line\n", encoding="utf-8")
        found = self._check()
        crit = [e for e in found if "new crash" in e["symptom"].lower()]
        self.assertTrue(crit, "a genuine crash must still be recorded")
        self.assertEqual(crit[0]["severity"], "critical")

    def test_a_non_ascii_crash_still_reaches_the_ledger(self):
        """The marker was written without encoding= (cp1252 on Windows) and read
        back as utf-8. Real tracebacks carry em-dashes, so the read raised
        UnicodeDecodeError, the bare except swallowed it as "check skipped",
        and a genuine CRITICAL crash reached the ledger zero times across ten
        recorded crashes. The write must be utf-8 AND the read must not abort on
        a marker left behind by the old writer."""
        (sup.ROOT / "render_crash.log").write_text(
            "=== 9 2026-01-01 00:00:00 ===\n"
            "RuntimeError: ComfyUI died \u2014 still warming up\n", encoding="utf-8")
        # a marker left by the OLD cp1252 writer: valid cp1252, invalid utf-8
        (sup.ROOT / ".supervisor_crash_seen").write_bytes(
            "older \u2014 crash\n".encode("cp1252"))
        found = sup.check(fix=False, quiet=False)
        self.assertTrue(any("new crash" in e["symptom"].lower() for e in found),
                        "an unreadable marker must not silence crash detection")
        # and the marker it leaves behind must itself be readable next time
        self.assertEqual(
            (sup.ROOT / ".supervisor_crash_seen").read_text(encoding="utf-8"),
            (sup.ROOT / "render_crash.log").read_text(encoding="utf-8").strip())

    def test_tts_being_down_alone_is_not_a_critical(self):
        """A render evicts the TTS server twice, on purpose, to free ~3GB of
        VRAM before the GPU phases. Reporting that as critical made ep.preflight
        refuse to render for 7 days after every successful episode — the
        pipeline blocked itself. Seven such criticals had accumulated."""
        self._svc.stop()  # replace the all-up mock with tts-only-down
        svc2 = patch.object(sup.ep, "ensure_services",
                            return_value=(False, {"llm": True,
                                                  "comfyui": True,
                                                  "tts": False}))
        svc2.start()
        self.addCleanup(svc2.stop)
        sup.ROOT.joinpath("issues.json").unlink(missing_ok=True)
        found = sup.check(fix=False, quiet=False)
        for e in found:
            self.assertNotEqual(e["severity"], "critical",
                                f"a by-design TTS eviction armed a render "
                                f"block: {e}")

    def test_tts_only_criticals_are_retired(self):
        """The criticals the old policy manufactured must be cleared, or they
        keep blocking renders until their 7-day triage window expires."""
        sup.LEDGER.write_text(json.dumps([
            {"ts": "2026-01-01 00:00:00", "severity": "critical",
             "symptom": "services down: ['tts']", "action": "none", "resolved": False},
            {"ts": "2026-01-01 00:00:01", "severity": "critical",
             "symptom": "services down: ['llm']", "action": "none", "resolved": False},
        ]), encoding="utf-8")
        sup._reconcile_service_outages(except_services=("tts",))
        rows = {r["symptom"]: r for r in sup._load_ledger()}
        self.assertTrue(rows["services down: ['tts']"]["resolved"])
        self.assertFalse(rows["services down: ['llm']"]["resolved"],
                         "a genuine outage must not be retired")

    def test_record_survives_a_ledger_row_without_a_symptom(self):
        """A hand-edited row missing 'symptom' raised KeyError out of record(),
        which aborted a whole check() mid-way."""
        sup.LEDGER.write_text(json.dumps(
            [{"ts": "2026-01-01 00:00:00", "severity": "warn"}]),
            encoding="utf-8")
        entry = sup.record("warn", "a fresh symptom")
        self.assertEqual(entry["symptom"], "a fresh symptom")

    def test_a_non_list_ledger_does_not_silently_disarm_the_gate(self):
        """A torn issues.json read as [] — which reads as 'no open criticals',
        DISARMING the gate rather than reporting a problem."""
        sup.LEDGER.write_text('{"oops": true}', encoding="utf-8")
        self.assertEqual(sup._load_ledger(), [])
        with self.assertLogs("supervisor", level="WARNING"):
            sup._load_ledger()

    def test_ledger_round_trip_is_atomic(self):
        sup.record("warn", "one")
        sup.record("critical", "two")
        rows = sup._load_ledger()
        self.assertEqual([r["symptom"] for r in rows], ["one", "two"])
        self.assertEqual(list(sup.ROOT.glob("*.tmp")), [],
                         "a ledger write must not leave a temp file")

    def test_save_ledger_reports_failure_instead_of_swallowing_it(self):
        """It used to `except: pass`, so a recorded critical could silently
        never land while record() returned an entry implying it had."""
        ok = sup._save_ledger([{"ts": "2026-01-01 00:00:00", "severity": "info",
                                "symptom": "x", "action": "", "resolved": True}])
        self.assertTrue(ok, "a normal save must report success")
        with patch.object(sup.ep, "_atomic_write_json",
                          side_effect=OSError("disk locked")):
            with self.assertLogs("supervisor", level="WARNING"):
                self.assertFalse(sup._save_ledger([]),
                                 "a failed save must report failure")

    # -- the RECORDING path (the only one production uses) -----------------
    # Every test above calls check(quiet=True), which is the DASHBOARD's mode.
    # That left the entire recording path — the one the 120s loop and the ledger
    # gate actually run — untested, which is why a quiet poll silently eating a
    # CRITICAL crash survived three review rounds.

    def _crash_log(self, body="RuntimeError: ComfyUI died\nTraceback line\n"):
        self._seed_marker()
        (sup.ROOT / "render_crash.log").write_text(
            "=== 1 2026-01-01 00:00:00 ===\n"
            "RuntimeError: something older\n"
            "=== 7 2026-01-01 00:00:00 ===\n" + body, encoding="utf-8")

    def test_recording_loop_persists_a_crash_to_the_ledger(self):
        self._crash_log()
        found = sup.check(fix=False, quiet=False)
        rows = sup._load_ledger()
        self.assertTrue(any("new crash" in r.get("symptom", "") for r in rows),
                        f"the crash never reached the ledger; rows={rows}")
        self.assertTrue(any(e["severity"] == "critical" for e in found))

    def test_a_quiet_poll_does_not_consume_the_crash(self):
        """A quiet poll may REPORT a crash, but it must not write the dedupe
        marker — otherwise the recording loop sees nothing new and the crash is
        lost entirely."""
        self._crash_log()
        quiet = sup.check(fix=False, quiet=True)
        self.assertTrue(any("new crash" in e["symptom"] for e in quiet),
                        "quiet mode should still surface the finding")
        self.assertEqual(sup._load_ledger(), [],
                         "quiet mode must not write the ledger")
        # and the recording loop must still see it
        sup.check(fix=False, quiet=False)
        rows = sup._load_ledger()
        self.assertTrue(any("new crash" in r.get("symptom", "") for r in rows),
                        "a quiet poll consumed the crash observation")

    def test_quiet_mode_never_writes_the_ledger(self):
        for i, sev in enumerate(("warn", "critical", "info")):
            self._seed_marker()
            (sup.ROOT / "render_crash.log").write_text(
                f"=== 1 2026-01-01 00:00:00 ===\nRuntimeError: old {i}\n"
                f"=== {i + 2} 2026-01-01 00:00:00 ===\nRuntimeError: crash {i}\n",
                encoding="utf-8")
            sup.check(fix=False, quiet=True)
            self.assertEqual(sup._load_ledger(), [],
                             f"quiet mode wrote the ledger on run {i}")

    def test_the_same_crash_is_recorded_once(self):
        """30-minute dedupe: the supervisor loops every 120s, so without it the
        ledger fills with identical rows."""
        self._crash_log()
        for _ in range(4):
            sup.check(fix=False, quiet=False)
        rows = [r for r in sup._load_ledger() if "new crash" in r.get("symptom", "")]
        self.assertEqual(len(rows), 1,
                         f"the same crash was recorded {len(rows)} times")

    def test_a_crash_row_names_the_exception_not_a_stack_frame(self):
        """The row a human triages must name the FAILURE.

        render_crash.log is appended with traceback.format_exc(), so the last
        line is the exception and the third-from-last is a `File ...` frame.
        Taking the wrong end of the window filed `new crash:   File
        episode_producer.py, line 712, in produce` — and
        resolve_critical(symptom_contains=...) has to be matched against that
        text, so the row was both unreadable and unusable. Pinned here against
        a real format_exc() shape, not a hand-written stub.
        """
        import traceback
        try:
            raise RuntimeError("ComfyUI returned 0 images for scene 4")
        except RuntimeError:
            tb = traceback.format_exc()
        self._seed_marker()
        (sup.ROOT / "render_crash.log").write_text(
            "=== 3 2026-01-01 00:00:00 ===\n" + tb, encoding="utf-8")
        sup.check(fix=False, quiet=False)
        rows = [r for r in sup._load_ledger()
                if "ComfyUI returned 0 images" in r.get("symptom", "")]
        self.assertEqual(len(rows), 1, "the crash was not recorded by its exception")
        self.assertNotIn("File \"", rows[0]["symptom"],
                         "the ledger row quoted a stack frame, not the error")

    def test_record_dedupes_a_repeated_symptom_within_the_window(self):
        """Directly on record(), not via check().

        ponytail: the crash-tail dedupe above is a SECOND mechanism (the marker
        file). This covers the first one — record()'s own 30-minute window.
        Mutation testing showed shortening the window to 0 left the suite
        green, because every other test records each symptom exactly once and
        so never reaches the update branch.
        """
        sup.LEDGER.unlink(missing_ok=True)
        first = sup.record("warn", "services down: ['llm']")
        self.assertFalse(first["resolved"])
        again = sup.record("warn", "services down: ['llm']", action="auto-restarted")
        rows = sup._load_ledger()
        self.assertEqual(len(rows), 1,
                         f"the same symptom was appended {len(rows)} times "
                         f"instead of updating in place")
        self.assertEqual(rows[0]["action"], "auto-restarted",
                         "the dedupe path must update the action")
        # record() returns the MUTATED row, which is the same object held in
        # `entries` — so `again` reflects the new action while `first` (a
        # separate dict from the caller) does not. Identity is not the contract;
        # what must hold is that only one row exists and it carries the update.
        self.assertEqual(again["action"], "auto-restarted")

    def test_a_different_symptom_is_a_separate_row(self):
        sup.LEDGER.unlink(missing_ok=True)
        sup.record("warn", "alpha")
        sup.record("warn", "beta")
        self.assertEqual(len(sup._load_ledger()), 2)

    def test_a_symptom_outside_the_window_appends_a_new_row(self):
        """Past the window it is a new event, not an update."""
        sup.LEDGER.unlink(missing_ok=True)
        sup.record("warn", "old news")
        rows = sup._load_ledger()
        rows[0]["ts"] = time.strftime("%Y-%m-%d %H:%M:%S",
                                      time.localtime(time.time() - 7200))
        sup._save_ledger(rows)
        sup.record("warn", "old news")
        self.assertEqual(len(sup._load_ledger()), 2,
                         "a symptom older than the window must append")

    def test_the_ledger_lock_serialises_concurrent_writers(self):
        """The ledger is read-modify-write by the supervisor thread AND the
        dashboard's /api/issues/resolve, in two processes. Without the lock,
        one write is lost.

        Mutation testing showed deleting _ledger_guard entirely left the suite
        green, because every other test here is single-threaded. Assert the
        lock is HELD, and prove mutual exclusion with real threads.
        """
        import threading

        held = []
        real = sup.FileLock

        def spy(path, **kw):
            lock = real(path, **kw)
            held.append(lock)
            return lock

        # 1. the guard is actually taken on the write path
        with patch.object(sup, "FileLock", spy):
            sup.record("warn", "one")
            sup.record("warn", "two")
        self.assertGreaterEqual(len(held), 2,
                                "record() must take the cross-process lock")

        # 2. two threads writing at once must not lose an entry
        sup.LEDGER.unlink(missing_ok=True)
        errors = []

        def writer(tag):
            try:
                for i in range(6):
                    sup.record("warn", f"{tag}-{i}")
            except Exception as e:      # a TimeoutError here IS the bug
                errors.append(e)

        ts = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=60)
        self.assertEqual(errors, [], f"concurrent record() raised: {errors[:2]}")
        rows = sup._load_ledger()
        symptoms = {r.get("symptom") for r in rows}
        missing = {f"{t}-{i}" for t in range(4) for i in range(6)} - symptoms
        self.assertFalse(missing,
                         f"{len(missing)} concurrent writes were lost, e.g. "
                         f"{sorted(missing)[:5]} — the ledger is not serialised")

    def test_a_dead_pid_in_render_lock_is_reported(self):
        """The empty-file case is the idle state; a lock naming a dead PID is
        the real stale case, and only the former was covered."""
        (sup.ROOT / "render.lock").write_text("999999", encoding="utf-8")
        found = sup.check(fix=False, quiet=False)
        self.assertTrue(any("stale render.lock" in e["symptom"] for e in found),
                        "a dead pid holding render.lock must be reported")

    def test_render_lock_of_zero_bytes_is_not_a_stale_lock(self):
        """A released lock is truncated to 0 bytes, which is the NORMAL idle
        state. int('') raised and the bare except swallowed it, so this check
        never once fired in production."""
        (sup.ROOT / "render.lock").write_bytes(b"")
        (sup.ROOT / "render.lock").touch()
        found = sup.check(fix=False, quiet=True)
        self.assertFalse(any("stale render.lock" in e["symptom"] for e in found),
                         "an empty render.lock is idle, not stale")


if __name__ == "__main__":
    unittest.main(verbosity=2)
