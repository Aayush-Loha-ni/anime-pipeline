#!/usr/bin/env python3
"""Structural guards on watchdog.bat.

The watchdog was the only one of the five fixes with zero coverage, and its
singleton guard was broken twice before it worked — both times the code
looked correct. These tests parse the script and assert the invariants that
the manual runs confirmed, so a future edit that drops them is caught
without spawning real services.

They assert STRUCTURE, not behaviour: a full behavioural test would have to
launch ComfyUI/llama-server. The behavioural half was verified separately
against a sandboxed replica (4 invocations -> 3 declines -> 3 cycles).
"""
import unittest
from pathlib import Path

WATCHDOG = Path(__file__).resolve().parents[2] / "watchdog.bat"


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(WATCHDOG.exists(), f"{WATCHDOG} not found")
        self.src = WATCHDOG.read_text(encoding="utf-8", errors="replace")

    def test_loops_rather_than_running_once(self):
        """The original bug: four ensures then exit. Services that died
        afterwards stayed dead (53 'services down' rows in issues.json)."""
        self.assertRegex(self.src, r"(?m)^goto\s+watch\s*$",
                         "the four ensures must be inside a loop")

    def test_bare_invocation_spawns_and_returns(self):
        """The documented flow is 'watchdog.bat, THEN Open Dashboard.bat',
        so the bare call must not block."""
        self.assertIn('if /i not "%~1"=="--loop"', self.src)
        self.assertRegex(self.src, r'start\s+"anime-watchdog"\s+/min\s+"%~f0"\s+--loop')
        # The parent must exit rather than fall into the loop.
        guard = self.src.index('if /i not "%~1"=="--loop"')
        block = self.src[guard:guard + 400]
        self.assertIn("exit /b 0", block,
                      "the spawning parent must return immediately")

    def test_is_singleton_guarded(self):
        """N bare runs must not mean N loops — they would race the :8080
        relaunch and kill/relaunch the LLM in a loop."""
        self.assertRegex(self.src, r"(?m)^\s*call :wd_claim\s*$",
                         "the claim must be made before anything is spawned")
        self.assertIn("watchdog.lock", self.src)
        self.assertIn("New-Item -ItemType Directory", self.src,
                      "the lock claim must be the atomic New-Item")

    def test_parent_claims_because_child_cannot(self):
        """`start` gives the CHILD the "anime-watchdog" window title, so a
        claim made by the child matches itself and declines — 4 bare
        invocations produced 4 competing loops. The parent carries no such
        title, so it must be the one that claims, and the child must only
        beat. This is the invariant that makes the singleton work at all."""
        guard = self.src.index('if /i not "%~1"=="--loop"')
        block = self.src[guard:guard + 700]
        self.assertIn("call :wd_claim", block,
                      "the PARENT (bare invocation) must claim the lock")
        claim_at = block.index("call :wd_claim")
        spawn_at = block.index("start ")
        self.assertLess(claim_at, spawn_at,
                        "claim must precede the spawn, or two parents race")
        # the --loop half must NOT claim
        after = self.src[self.src.index("goto watch"):]
        self.assertNotIn("call :wd_claim", after.split(":ensure")[0],
                         "the child must not claim: it would match its own "
                         "window title and decline every time")

    def test_the_claim_verdict_is_stashed_before_the_cleanup_runs(self):
        """`rd` clobbers ERRORLEVEL: a successful rd leaves it alone, but an rd
        on an already-deleted directory sets it to 2. So reading `if errorlevel
        1` after the cleanup loop makes a genuine CAS WINNER look like a loser —
        it prints "watchdog already running" and exits WITHOUT spawning, while
        holding a fresh lock, so the watchdog is dead for WD_STALE_S (900s by
        default) with nothing logged. Measured: the interleaving needs only two
        bare invocations against a stale lock, which is the documented
        four-rapid-clicks case.

        Structural, like the rest of this file: a behavioural test would have to
        race two real cmd.exe processes against a live lock directory.
        """
        claim = self.src[self.src.index("\n:wd_claim"):self.src.index("\n:wd_beat")]
        stash = claim.index("WD_CLAIM_FAILED=1")
        cleanup = claim.index("do rd /s /q")
        decide = claim.index("if defined WD_CLAIM_FAILED")
        self.assertLess(stash, cleanup,
                        "the verdict must be captured BEFORE `rd`, which "
                        "overwrites ERRORLEVEL on an already-deleted directory")
        self.assertLess(cleanup, decide,
                        "the loser's reclaim directory must still be cleaned up "
                        "before returning")
        self.assertNotIn("\nif errorlevel 1 exit /b 1", claim[cleanup:],
                         "the verdict must be read from the saved value, not "
                         "from ERRORLEVEL after the cleanup")

    def test_liveness_is_a_heartbeat_not_an_identity_check(self):
        """Identity-based liveness was tried three times and failed each way:
        CIM COMMANDLINE matched nothing, tasklist /V WINDOWTITLE matched the
        child itself (or nothing at all, for a /min detached window), and a
        `set` inside a parenthesised block never reached the `if defined`.
        A heartbeat the loop refreshes on a file it already owns depends on
        none of that."""
        claim = self.src[self.src.index("\n:wd_claim"):][:2600]
        # Comments deliberately name the approaches that failed, so the
        # assertions below look at executable lines only.
        code = "\n".join(l for l in claim.splitlines()
                         if not l.strip().upper().startswith("REM"))
        self.assertIn("LastWriteTime", code,
                      "liveness must be judged by the lock's timestamp")
        self.assertNotIn("tasklist", code,
                         "do not reintroduce an identity-based liveness test")
        self.assertNotIn("Get-CimInstance", code)
        self.assertRegex(self.src, r"(?m)^\s*call :wd_beat\s*$",
                         "the loop must refresh the heartbeat each cycle")
        self.assertIn("WD_STALE_S", self.src,
                      "staleness threshold must be configurable and generous")

    def test_claim_agrees_with_itself(self):
        """The bug this guards: :wd_claim WROTE the lock directory's
        LastWriteTime but READ a nonexistent 'hb' file inside it, so $age was
        always 999999 and every invocation reclaimed instead of declining."""
        claim = "\n".join(l for l in self.src[self.src.index("\n:wd_claim"):][:2600].splitlines()
                          if not l.strip().upper().startswith("REM"))
        self.assertIn("(Get-Item -LiteralPath $d).LastWriteTime", claim)
        self.assertNotIn("'hb'", claim,
                         "no 'hb' file is written; reading one made the lock "
                         "look permanently stale")
        # and the threshold must be interpolated, not read from the
        # environment: $env:WD_STALE_S was empty inside the spawned child,
        # so [double]$null = 0 and nothing was ever considered fresh.
        self.assertIn("[double]'%WD_STALE_S%'", claim,
                      "the threshold must be passed by expansion, not $env:")

    def test_loop_is_stoppable(self):
        """SKILL.md documents a stop command; it has to match reality. The
        window title set by `start` is what taskkill filters on."""
        self.assertIn('start "anime-watchdog"', self.src)
        for doc in (WATCHDOG.parents[0] / "hermes-skill" / "SKILL.md",
                    WATCHDOG.parents[0] / "rules" / "ops.md"):
            if doc.exists():
                txt = doc.read_text(encoding="utf-8", errors="replace")
                self.assertIn('WINDOWTITLE eq anime-watchdog', txt,
                              f"{doc.name} documents a stop command that "
                              f"does not match the title `start` sets")

    def test_interval_is_guarded(self):
        """A non-numeric WD_INTERVAL makes Start-Sleep throw and the loop
        spins as fast as PowerShell starts, hammering all four services."""
        self.assertIn("WD_INTERVAL", self.src)
        self.assertRegex(self.src, r'delims=0123456789',
                         "a non-numeric interval must be coerced")

    def test_render_guard_still_present(self):
        """Looping makes the render-live guard more important, not less: the
        loop now restarts services every cycle, so it must still refuse to
        touch the GPU while a render holds it.

        Sliced to the next label rather than a magic character count - a fixed
        window silently started failing the moment the subroutine grew a
        comment, which reads as "the guard was removed" when it is present.
        """
        start = self.src.index("\n:render_live")
        rest = self.src[start + 1:]
        # Stop at the next real routine. Slicing to the next bare `:` label is
        # wrong: :render_live has internal goto labels (RL_UNKNOWN,
        # RL_RENDER_CHECK) that are part of the same subroutine.
        for boundary in ("\n:ensure", "\n:render_pid", "\n:wd_"):
            idx = rest.find(boundary)
            if idx != -1:
                rest = rest[:idx]
                break
        body = rest
        self.assertIn("render.pid", body)
        self.assertIn("render.lock", body)
        # Every ensure must consult it before restarting anything. Bound the search by
        # the next ROUTINE, not the next label: batch uses internal goto labels
        # (:comfy_probe, :comfy_recovery, RL_*) inside a routine, so both a fixed
        # character window and "up to the next label" cut the call off.
        routines = (":ensure", ":ensure_comfy", ":ensure_llm", ":render_live",
                    ":wd_claim", ":wd_claim_child", ":recover_pid", ":render_pid")
        for routine in ("\n:ensure_comfy", "\n:ensure_llm"):
            start = self.src.index(routine)
            ends = [self.src.find("\n" + r, start + 1) for r in routines]
            ends = [e for e in ends if e != -1]
            stop = min(ends) if ends else len(self.src)
            self.assertIn("call :render_live", self.src[start:stop],
                          f"{routine} must check for a live render first")

    def test_single_exit_point_per_loop(self):
        """No `goto :watch` outside the loop body, or a restart from inside
        an ensure would double-loop."""
        self.assertEqual(self.src.count("goto watch"), 1,
                         "exactly one back-edge into the loop")
