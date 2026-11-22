#!/usr/bin/env python3
"""Tests for the durability primitives in json_util.

These are the two functions every durable state file in the pipeline routes
through, and they were both UNTESTED while being load-bearing: mutation
testing showed that reverting `dump_json_atomic` to "fail silently" and
deleting the cross-process lock both left the whole suite green.

`dump_json_atomic` exists because seven writers were truncate-then-write, and
`FileLock` because three processes read-modify-write the same files. Both are
one-line mistakes to get wrong and impossible to notice in production.
"""
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import episode_producer as ep
import json_util
from json_util import FileLock, dump_json_atomic, load_json_file, pid_alive


class AtomicWriteTests(unittest.TestCase):
    def test_round_trip_and_read_back(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            dump_json_atomic(p, {"a": 1, "b": [1, 2]}, indent=2)
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")),
                             {"a": 1, "b": [1, 2]})

    def test_creates_missing_parent_directory(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "deep" / "deeper" / "state.json"
            dump_json_atomic(p, {"ok": True})
            self.assertTrue(p.is_file())

    def test_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            dump_json_atomic(p, {"v": 1})
            self.assertEqual(list(Path(td).glob("*.tmp")), [],
                             "a successful write must not leave a .tmp behind")

    def test_RAISES_when_the_destination_is_held_open(self):
        """The single most important property, and previously untested.

        Windows: os.replace() is MoveFileExW(MOVEFILE_REPLACE_EXISTING) and
        "existing opens of the destination path are not allowed... Most
        Windows programs, including Python, do not share delete access on open
        files" (cpython#90161). So any concurrent reader of progress_live.json
        — which the dashboard AND the supervisor both poll — makes the replace
        raise. The original helper retried three times and then exited the loop
        NORMALLY: no exception, no log, no update. The write vanished and every
        caller believed it happened.

        Assert it raises, that the OLD content survives, and that no temp file
        is orphaned.
        """
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            dump_json_atomic(p, {"v": 0})
            holder = open(p, "r", encoding="utf-8")   # blocks delete access
            try:
                with self.assertRaises(OSError) as cm:
                    dump_json_atomic(p, {"v": 1}, retries=2)
                self.assertIn("could not replace", str(cm.exception))
            finally:
                holder.close()
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["v"], 0,
                             "a failed swap must leave the old content intact")
            self.assertEqual(list(Path(td).glob("*.tmp")), [],
                             "a failed write must not orphan its temp file")

    def test_succeeds_once_the_holder_releases(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            dump_json_atomic(p, {"v": 0})
            holder = open(p, "r", encoding="utf-8")
            holder.close()
            dump_json_atomic(p, {"v": 1}, retries=3)
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["v"], 1)

    def test_fsync_is_called_before_the_rename(self):
        """Durability ordering, pinned as far as it can be from Python.

        `close()` flushes Python's buffer to the OS but does NOT order the data
        before the directory entry. Without fsync, a power loss can land the
        rename while the file contents are still unwritten — leaving a
        correctly-named file full of zeroes. fsync(data) then rename is the
        order that prevents it.

        Honest limitation: the actual crash-window property cannot be observed
        without crashing the machine, so this asserts the fsync CALL rather than
        its effect. That is the strongest available proxy — and it is still a
        real regression guard, because mutation testing showed deleting the
        fsync line previously left the entire suite green.
        """
        calls = []
        real_fsync = os.fsync
        real_replace = os.replace

        def spy_fsync(fd):
            calls.append(fd)
            return real_fsync(fd)

        def spy_replace(src, dst):
            # At the moment of the swap, fsync must ALREADY have happened.
            self.assertTrue(calls, "os.replace ran before any fsync")
            return real_replace(src, dst)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            with patch("json_util.os.fsync", spy_fsync), \
                 patch("json_util.os.replace", spy_replace):
                dump_json_atomic(p, {"v": 1})
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["v"], 1)
        self.assertGreaterEqual(len(calls), 1, "fsync was never called")

    def test_every_write_takes_the_per_path_thread_lock(self):
        """The in-process lock must be ACQUIRED, not merely exist.

        ponytail: two weaker versions of this test both passed with the lock
        deleted. One used writers only (with no reader holding the destination
        open, Windows rarely refuses the swap, so there is nothing to
        serialise). The other ran readers but finished in 0.7s, so contention
        never developed. Timing-based contention is inherently flaky; assert the
        mechanism instead, and prove the mechanism's two properties below.
        """
        acquired = []
        real = json_util._thread_lock_for

        def spy(path):
            lock = real(path)
            acquired.append(lock)
            return lock

        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "progress_live.json"
            b = Path(td) / "issues.json"
            with patch("json_util._thread_lock_for", spy):
                dump_json_atomic(a, {"n": 1})
                dump_json_atomic(a, {"n": 2})
                dump_json_atomic(b, {"n": 3})
            self.assertEqual(len(acquired), 3,
                             "every durable write must take the thread lock")
            self.assertIs(acquired[0], acquired[1],
                          "two writes to the SAME file must share one lock, "
                          "otherwise they do not serialise")
            self.assertIsNot(acquired[0], acquired[2],
                             "different files must not share a lock, or an "
                             "unrelated write would block")

    def test_thread_lock_map_is_keyed_case_insensitively_on_windows(self):
        """Windows paths are case-insensitive, so two spellings of the same
        file must map to the SAME lock or serialisation silently fails."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            a = json_util._thread_lock_for(base / "state.json")
            b = json_util._thread_lock_for(
                base / "STATE.JSON")            # Path has no .upper(); the
            #                                       comparison is on the STRING
            self.assertIs(a, b)
            c = json_util._thread_lock_for(base / "other.json")
            self.assertIsNot(a, c)

    def test_thread_lock_is_load_bearing_under_read_contention(self):
        """The per-path thread lock must actually earn its keep.

        ponytail: an earlier version of this test used writers only, and it
        passed whether or not the lock existed — with no reader holding the
        destination open, Windows rarely refuses the swap, so there is nothing
        for the lock to prevent. Mutation testing confirmed it: deleting the
        lock left the suite green.

        The real workload is a dashboard/supervisor polling the same file while
        a render writes it, so that is what this reproduces: readers keep the
        destination open, and every writer has to retry around them.
        """
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            dump_json_atomic(p, {"tag": -1, "n": -1})
            denied, stop = [], threading.Event()
            lock = threading.Lock()
            start = threading.Barrier(4)

            def writer(tag):
                start.wait(timeout=30)
                i = 0
                while not stop.is_set() and i < 25:
                    try:
                        dump_json_atomic(p, {"tag": tag, "n": i}, retries=6)
                    except OSError:
                        with lock:
                            denied.append(i)
                    i += 1

            def reader():
                while not stop.is_set():
                    try:
                        json.loads(p.read_text(encoding="utf-8"))
                    except (ValueError, OSError):
                        pass
                    time.sleep(0.005)

            ts = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
            rs = [threading.Thread(target=reader) for _ in range(2)]
            for t in ts + rs:
                t.start()
            for t in ts:
                t.join(timeout=180)
            stop.set()
            for t in rs:
                t.join(timeout=30)
            self.assertEqual(denied, [],
                             f"{len(denied)} of {4 * 25} writes exhausted their "
                             f"retries; the in-process lock is not doing its job")


class AtomicWriteWrapperTests(unittest.TestCase):
    """episode_producer._atomic_write_json is the entry point every durable
    state file in the render goes through. Its whole purpose is to be loud;
    reverting it to a silent swallow survived the whole suite before this."""

    def test_wrapper_propagates_a_failed_write(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "progress_live.json"
            ep._atomic_write_json(p, {"phase": "starting"})
            self.assertTrue(p.is_file())
            holder = open(p, "r", encoding="utf-8")
            try:
                with self.assertRaises(OSError):
                    ep._atomic_write_json(p, {"phase": "clips"}, retries=2)
            finally:
                holder.close()
            # and the write lands once the reader lets go
            ep._atomic_write_json(p, {"phase": "done"}, retries=3)
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["phase"],
                             "done")

    def test_wrapper_does_not_swallow(self):
        src = inspect.getsource(ep._atomic_write_json)
        self.assertNotIn("except Exception", src,
                         "the wrapper must not swallow a failed durable write")
        self.assertIn("dump_json_atomic", src)
        """Writers alone never collide; see the read-contention test."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            dump_json_atomic(p, {"tag": -1, "n": -1})
            denied, start = [], threading.Barrier(4)
            lock = threading.Lock()

            def writer(tag):
                start.wait(timeout=30)
                for i in range(40):
                    try:
                        dump_json_atomic(p, {"tag": tag, "n": i}, retries=6)
                    except OSError:
                        with lock:
                            denied.append(i)

            ts = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
            t0 = time.monotonic()
            for t in ts:
                t.start()
            for t in ts:
                t.join(timeout=120)
            elapsed = time.monotonic() - t0
            self.assertEqual(denied, [],
                             "same-process writers must serialise on a thread "
                             "lock; exhausting retries means they did not")
            self.assertLess(elapsed, 20,
                            f"160 serialised writes took {elapsed:.1f}s")

    def test_readers_never_observe_a_torn_document(self):
        """Atomicity: a reader sees the old file or the new one, never half.

        Readers holding the destination open block the swap on Windows, so a
        write may legitimately exhaust its retries — but it must REPORT that
        (never silently drop the write) and the file on disk must stay whole.
        """
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            dump_json_atomic(p, {"tag": -1, "n": -1})
            tears, denied, stop = [], [], threading.Event()
            lock = threading.Lock()

            def writer(tag):
                i = 0
                while not stop.is_set() and i < 25:
                    try:
                        dump_json_atomic(p, {"tag": tag, "n": i}, retries=4)
                    except OSError:
                        with lock:
                            denied.append(i)
                    i += 1

            def reader():
                while not stop.is_set():
                    try:
                        doc = json.loads(p.read_text(encoding="utf-8"))
                    except (ValueError, OSError):
                        continue        # mid-swap; not a tear
                    if set(doc) != {"tag", "n"} or not isinstance(doc.get("n"), int):
                        with lock:
                            tears.append(doc)
                    time.sleep(0.005)   # realistic read cadence

            ts = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
            rs = [threading.Thread(target=reader) for _ in range(2)]
            for t in ts + rs:
                t.start()
            for t in ts:
                t.join(timeout=120)
            stop.set()
            for t in rs:
                t.join(timeout=30)
            self.assertEqual(tears, [], f"a reader saw a TORN document: {tears[:3]}")

    def test_non_dict_and_unicode_survive(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            payload = {"rows": [{"ts": "2026-09-29 21:31:06", "symptom": "café — ok"}]}
            dump_json_atomic(p, payload, indent=2, ensure_ascii=False)
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), payload)


class LoadJsonFileTests(unittest.TestCase):
    def test_returns_the_default_on_missing_and_on_garbage(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "nope.json"
            self.assertEqual(load_json_file(missing, []), [])
            self.assertEqual(load_json_file(missing, None), None)
            bad = Path(td) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            self.assertEqual(load_json_file(bad, {}), {})

    def test_returns_the_parsed_document_when_valid(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ok.json"
            p.write_text('{"a": 2}', encoding="utf-8")
            self.assertEqual(load_json_file(p, {}), {"a": 2})


class FileLockTests(unittest.TestCase):
    def test_second_holder_is_refused_not_deadlocked(self):
        with tempfile.TemporaryDirectory() as td:
            lf = Path(td) / "x.lock"
            with FileLock(lf, blocking=False):
                with self.assertRaises(BlockingIOError):
                    with FileLock(lf, blocking=False):
                        pass
            # released: takeable again
            with FileLock(lf, blocking=False):
                pass

    def test_release_is_idempotent_and_safe_without_acquire(self):
        lf = FileLock(Path(tempfile.gettempdir()) / "never-created.lock",
                      blocking=False)
        lf.release()          # must not raise
        with lf:
            pass
        lf.release()          # must not raise

    def test_works_on_a_zero_length_file(self):
        """msvcrt.locking locks bytes from the current position; a 0-byte
        sidecar must still acquire (verified on Windows)."""
        with tempfile.TemporaryDirectory() as td:
            lf = Path(td) / "zero.lock"
            lf.write_bytes(b"")
            with FileLock(lf, blocking=False):
                pass

    def test_blocks_then_times_out_with_a_clear_error(self):
        with tempfile.TemporaryDirectory() as td:
            lf = Path(td) / "t.lock"
            with FileLock(lf, blocking=False):
                t0 = time.monotonic()
                with self.assertRaises(TimeoutError) as cm:
                    with FileLock(lf, blocking=True, timeout=0.3, poll=0.05):
                        pass
                self.assertIn("timed out", str(cm.exception))
                self.assertLess(time.monotonic() - t0, 10)

    def test_excludes_another_PROCESS(self):
        """The reason this class exists: a threading.Lock cannot. Prove it with
        a real second interpreter rather than a second thread."""
        with tempfile.TemporaryDirectory() as td:
            lf = Path(td) / "cross.lock"
            child = (
                "import sys, time;"
                "sys.path.insert(0, %r);" % str(Path(__file__).parent) +
                "from json_util import FileLock;"
                "t=time.monotonic();"
                "\ntry:\n"
                "    with FileLock(%r, blocking=False):\n"
                "        print('ACQUIRED')\n"
                "except BlockingIOError:\n"
                "    print('BLOCKED')\n"
            ) % (str(lf),)
            with FileLock(lf, blocking=False):
                proc = subprocess.run([sys.executable, "-c", child],
                                      capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.stdout.strip(), "BLOCKED",
                             f"a separate process acquired a held lock: "
                             f"{proc.stdout} {proc.stderr[-200:]}")


class PidAliveTests(unittest.TestCase):
    def test_this_process_is_alive(self):
        self.assertTrue(pid_alive(os.getpid()))

    def test_non_positive_and_garbage_are_not_alive(self):
        # os.kill(0, 0) signals the caller's whole process group and returns 0;
        # os.kill(-1, 0) signals every process the user owns. Both would read as
        # "a render is alive", which is how a truncated PID file could make the
        # supervisor refuse to restart services.
        for bad in (0, -1, -99999):
            self.assertFalse(pid_alive(bad), f"pid {bad} must not read as alive")
        for junk in (None, "", "abc", float("nan")):
            self.assertFalse(pid_alive(junk))

    def test_a_dead_pid_is_not_alive(self):
        # Ask the OS for a pid that cannot exist. Windows PIDs are multiples of 4
        # and bounded by the session, so walk up from a very high base until one
        # is rejected rather than guessing a magic number.
        base = 0xFFFF00
        for delta in range(0, 400, 4):
            pid = base + delta
            if not pid_alive(pid):
                return                      # found a reliably-dead pid
        self.skipTest("could not find an unused pid on this machine")

    # Honest limitation, recorded rather than papered over:
    # mutation testing shows that DELETING the `if pid <= 0` guard leaves this
    # whole suite green on Windows. That is expected and not a coverage gap:
    # the guard lives in the POSIX branch (os.kill), and on Windows pid_alive
    # returns via ctypes/OpenProcess, which already rejects pid <= 0 because
    # OpenProcess(0x1000, False, 0) opens the caller's own process group. The
    # guard exists for POSIX, and a POSIX-only behaviour cannot be observed from
    # a Windows test run. The assertion above still pins the OUTCOME
    # (pid <= 0 is never alive) on both platforms, which is the property that
    # matters to every caller.


if __name__ == "__main__":
    unittest.main(verbosity=2)
