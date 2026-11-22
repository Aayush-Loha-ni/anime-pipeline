#!/usr/bin/env python3
"""Pipeline supervisor: watches the render, fixes known issues, keeps a ledger.

CPU-only (stdlib json). Never touches the GPU. The text-model
diagnosis only fires when NO render is alive, so it can never fight
ComfyUI/LLM for VRAM.

Ledger: issues.json at repo root — the dashboard (/api/issues) and any
future agent read the same file.

Usage:
  python supervisor.py --once          single check (also callable via check())
  python supervisor.py --loop [secs]   background monitor, default 120s
"""
import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

os.environ["ANIME_NO_PROGRESS"] = "1"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).parent))
import episode_producer as ep
from json_util import FileLock

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
LEDGER = ROOT / "issues.json"
STALL_S = 20 * 60  # progress older than this + alive render = stalled
_LEDGER_LOCK = threading.Lock()


@contextmanager
def _ledger_guard():
    """Serialise read-modify-write of issues.json across threads AND processes.

    ponytail: the old code carried a threading.Lock comment saying "the
    dashboard, a render and the supervisor all write at once" and had no lock
    at all. The supervisor thread records while /api/issues/resolve rewrites
    the same file, so one of the two writes was being lost.
    """
    with _LEDGER_LOCK:
        with FileLock(LEDGER.with_suffix(".lock"), blocking=True, timeout=15):
            yield


def _load_ledger():
    try:
        with open(LEDGER, encoding="utf-8") as f:
            data = json.load(f)
        # A hand-edited or torn file must not become a silent empty ledger —
        # an empty list reads as "no open criticals", which DISARMS the gate.
        if not isinstance(data, list):
            logger.warning("issues.json is not a list (%s); treating as unreadable",
                           type(data).__name__)
            return []
        return data
    except FileNotFoundError:
        return []
    except Exception as e:
        # ponytail: park torn writes so they aren't re-parsed forever, and SAY
        # so. This path used to be a bare except with a comment and no log,
        # so every recorded critical vanished from the gate's point of view.
        # (handle must be closed first — Windows can't rename open files)
        logger.warning("issues.json unreadable (%s); quarantined to %s.corrupt",
                       e, LEDGER.name)
        try:
            LEDGER.replace(LEDGER.with_name(LEDGER.name + ".corrupt"))
        except OSError:
            pass
        return []


def _save_ledger(entries):
    """Persist the ledger. Returns True on success.

    ponytail: this swallowed everything, so a recorded CRITICAL could silently
    never land while record() returned an entry implying it had been written.
    The caller decides what to do; the save itself just stops lying.
    """
    try:
        ep._atomic_write_json(LEDGER, entries[-200:], indent=2)
        return True
    except Exception as e:
        logger.warning("issues.json write FAILED (%s) — the ledger is stale "
                       "and a recorded critical will NOT block a render", e)
        return False


def record(severity, symptom, action="none", resolved=False):
    """Append one ledger entry. severity: info|warn|critical.

    Dedupe: same symptom within 30 min updates the existing entry
    instead of flooding the ledger (dashboard polls every 2s).
    """
    with _ledger_guard():
        entries = _load_ledger()
        now = time.time()
        for e in reversed(entries[-10:]):
            if not isinstance(e, dict):
                continue
            ets = parse_ts(e.get("ts"))
            if ets is None:
                continue
            # .get, not [": a row missing "symptom" used to raise KeyError
            # here, aborting the whole check() mid-way.
            if e.get("symptom") == symptom and now - ets < 1800:
                e["action"] = action
                e["resolved"] = resolved or e.get("resolved", False)
                _save_ledger(entries)
                return e
        entries.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "severity": severity, "symptom": symptom,
                        "action": action, "resolved": resolved})
        _save_ledger(entries)
        return entries[-1]


def _reconcile_service_outages(except_services=()):
    """Mark stale unresolved 'services down' criticals resolved when all
    services verify up. Historical outage records must not block preflight
    forever after recovery (only the recording loop writes here).

    ponytail: except_services also resolves criticals naming a service the
    pipeline deliberately takes down. A render evicts the TTS server for VRAM,
    so a post-render check legitimately finds it down — and ep.preflight blocks
    renders on unresolved criticals for 7 days. Seven such rows accumulated and
    were refusing /api/start. Pass ("tts",) to retire them.
    """
    try:
        with _ledger_guard():
            entries = _load_ledger()
            changed = False
            for e in entries:
                if (isinstance(e, dict) and e.get("severity") == "critical"
                        and not e.get("resolved")
                        and str(e.get("symptom", "")).startswith("services down:")):
                    names = str(e.get("symptom", ""))
                    if except_services:
                        # retiring by-design outages only; leave real ones open
                        if any(f"'{s}'" in names for s in except_services):
                            e["resolved"] = True
                            e["action"] = ("retired: service is taken down on "
                                           "purpose during a render")
                            changed = True
                        continue
                    e["resolved"] = True
                    e["action"] = (str(e.get("action", "")) + " | auto-resolved: "
                                   "services verified up").strip(" |")
                    changed = True
            if changed:
                _save_ledger(entries)
    except Exception as e:
        logger.warning("service-outage reconcile skipped: %s", e)


def _pid_alive(pid):
    from json_util import pid_alive
    return pid_alive(pid)


def _render_alive():
    return ep.render_alive()


def _gpu_temp():
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().split()[0])
    except Exception:
        return None


def check(fix=True, quiet=False):
    """Run all checks. Returns list of findings.

    quiet=True collects findings WITHOUT ledger writes (for 2s dashboard
    polls). The 120s background loop records. Without this split, every
    poll persists a row and parameterized symptoms spam the ledger.
    """
    found = []
    alive = _render_alive()

    def _rec(severity, symptom, action="none", resolved=False):
        e = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
             "severity": severity, "symptom": symptom,
             "action": action, "resolved": resolved}
        found.append(e)
        if not quiet:
            record(severity, symptom, action, resolved)
        return e

    # 1. stale render.lock (dead PID holds the GPU gate)
    try:
        lockf = ROOT / "render.lock"
        # A released lock is truncated to 0 bytes by _render_ownership's
        # finally block, so an empty file is the NORMAL idle state — not a
        # stale lock. int("") raised ValueError, the bare except swallowed it,
        # and this check never once fired in production. Only judge a lock that
        # actually names a PID.
        raw = lockf.read_text(encoding="utf-8").strip() if lockf.exists() else ""
        if raw:
            pid = int(raw)
            if not _pid_alive(pid):
                _rec("warn", "stale render.lock",
                     f"pid {pid} dead — producer reaps the OS lock on next run",
                     resolved=False)
    except Exception as e:
        logger.warning("render.lock check skipped: %s", e)

    # 2. services (read-only when a render is live — never restart mid-render)
    try:
        ok, status = ep.ensure_services(auto_start=False)
        down = [k for k, v in status.items() if not v]
        if not down and not quiet:
            _reconcile_service_outages()
        if down:
            # ponytail: a render evicts the TTS server on purpose, twice, to
            # free ~3GB of VRAM before the GPU phases. Reporting that as a
            # critical was an alert the pipeline manufactured about itself:
            # ep.preflight blocks renders on unresolved criticals for 7 days,
            # so every successful render armed a week-long self-block. The
            # outage is real but expected, so it is a warn with a reason.
            expected = [k for k in down if k == "tts"] if not alive else []
            real = [k for k in down if k not in expected]
            if expected and not real:
                _rec("warn", "tts down",
                     "none (render evicts tts for VRAM; watchdog relaunches)")
                # retire the criticals this policy used to manufacture
                _reconcile_service_outages(except_services=("tts",))
            if real and fix and not alive:
                ep.ensure_services(auto_start=True)
                _, status2 = ep.ensure_services(auto_start=False)
                still = [k for k, v in status2.items() if not v and k in real]
                action = ("auto-restarted, all up" if not still
                          else f"auto-restart attempted, still down: {still}")
                _rec("critical" if still else "warn",
                    f"services down: {real}", action,
                    resolved=not still)
            elif real:
                _rec("warn" if alive else "critical",
                    f"services down: {real}",
                    "none (render live, no restart mid-run)"
                    if alive else "none (fix disabled)")
    except Exception as e:
        _rec("warn", f"service check failed: {e}")

    # 3. disk space
    try:
        import shutil
        free_gb = shutil.disk_usage(ep.CONFIG["output_dir"]).free / 1e9
        if free_gb < 10:
            _rec("critical", f"disk_full ({free_gb:.1f}GB free)",
                "none — free space manually")
    except Exception as e:
        logger.warning("disk check skipped: %s", e)

    # 4. GPU heat (informational — heat guard owns the pause).
    # ponytail: this was the only _rec() outside a try. A non-numeric
    # gpu_max_temp, or the KeyError fixed in record(), escaped and killed the
    # whole check() — the background loop caught it and every check after this
    # point silently stopped running.
    try:
        temp = _gpu_temp()
        limit = int(ep.CONFIG.get("gpu_max_temp", 85))
        if temp is not None and temp >= limit:
            _rec("info", f"GPU hot ({temp}C), heat guard pausing",
                "none — automatic")
    except Exception as e:
        logger.warning("gpu heat check skipped: %s", e)

    # 5. stalled progress (alive render, no update for STALL_S)
    try:
        pl = ROOT / "progress_live.json"
        if alive and pl.exists():
            try:
                age = time.time() - json.loads(pl.read_text(encoding="utf-8")).get("heartbeat", 0)
            except (OSError, ValueError):
                age = time.time() - pl.stat().st_mtime
            if age > STALL_S:
                prog = json.loads(pl.read_text(encoding="utf-8"))
                # done/error = finished run awaiting cleanup, not "stalled"
                if prog.get("phase") not in ("done", "error"):
                    _rec("warn",
                         f"progress stalled at "
                         f"{prog.get('phase')}/{prog.get('detail','')}",
                         f"no update for {age/60:.0f}min — inspect render, "
                         "Stop if hung")
    except Exception as e:
        logger.warning("stall check skipped: %s", e)

    # 6. new crash entries (tail hash to avoid re-logging same crash)
    try:
        crash = ROOT / "render_crash.log"
        if crash.exists():
            tail = crash.read_text(encoding="utf-8").strip().splitlines()[-3:]
            marker = ROOT / ".supervisor_crash_seen"
            # read tolerantly: a marker written before the encoding= fix is cp1252
            try:
                seen = marker.read_text(encoding="utf-8") if marker.exists() else ""
            except UnicodeDecodeError:
                seen = marker.read_text(encoding="cp1252", errors="replace")
            cur = "\n".join(tail)
            # ponytail: "another render is already running" is a REFUSAL, not
            # a crash. It used to be recorded as `critical: new crash: ...
            # Another render is already running`, and ep.preflight blocks on
            # unresolved criticals for 7 days — so one impatient double-click
            # stopped every future render for a week. Treat contention as info.
            # The marker is the dedupe key, so ONLY the recording path may
            # advance it. It used to be written unconditionally, which meant a
            # quiet=True poll (the mode the dashboard's 2s poll is documented
            # to use) consumed the observation and wrote nothing — and the
            # recording loop then saw cur == seen and found nothing. Verified:
            # a genuine CRITICAL crash reached the ledger zero times.
            #
            # Quiet mode therefore reports the finding WITHOUT advancing the
            # marker, so the next recording pass still sees it as new.
            if cur and cur != seen and seen:  # skip the very first observation
                _busy = "Another render is already running" in cur
                _rec("info" if _busy else "critical",
                     ("another render already holds the lock"
                      if _busy else f"new crash: {tail[-1][:160]}"),
                     ("none — expected; one GPU job at a time"
                      if _busy else "none — see render_crash.log"))
                if not quiet:
                    # encoding= is load-bearing: without it Windows writes
                    # cp1252, and the next read is utf-8. render_crash.log
                    # tracebacks contain em-dashes, so the marker failed to
                    # decode, :282 swallowed it as "check skipped", and a
                    # genuine CRITICAL crash reached the ledger zero times
                    # across 10 recorded crashes.
                    marker.write_text(cur, encoding="utf-8")
    except Exception as e:
        logger.warning("crash-tail check skipped: %s", e)

    return found


def diagnose(symptom):
    """Text-model second opinion. Only when NO render is alive (VRAM safety)."""
    if _render_alive():
        return "render live — diagnosis deferred (won't fight ComfyUI for VRAM)"
    try:
        return ep.llm_chat(
            [{"role": "user",
              "content":
                 "You monitor a local anime render pipeline (llama-server/ComfyUI/CosyVoice/FFmpeg, "
                "Windows, 16GB VRAM). Symptom: " + symptom +
                ". Reply in 3 lines: likely cause, one safe fix, when to escalate."}],
            timeout=60, max_tokens=300)
    except Exception as e:
        return f"diagnosis unavailable: {e}"


def open_issues(limit=20):
    """Recent ledger entries plus counts (for dashboard / agents)."""
    # Take the same lock the three writers take. Without it this read can land
    # mid-write and return a torn/short list; the writers already serialise, so
    # the reader being the odd one out was an oversight rather than a decision.
    with _ledger_guard():
        entries = _load_ledger()
    return {"issues": entries[-limit:],
            "counts": {s: sum(1 for e in entries[-50:]
                              if isinstance(e, dict) and e.get("severity") == s)
                       for s in ("info", "warn", "critical")}}


def parse_ts(value):
    """Ledger timestamps to epoch seconds, or None if unparsable.

    Entries are written "%Y-%m-%d %H:%M:%S" but other tooling writes
    ISO-8601 with a "T" — the old strptime-only version silently treated
    every ISO row as "now", which is what made the 7-day triage filter a
    no-op. ponytail: fromisoformat handles both, so try it first.
    """
    import datetime as _dt
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return _dt.datetime.fromisoformat(text).timestamp()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return time.mktime(time.strptime(text, fmt))
        except ValueError:
            continue
    return None


def resolve_critical(symptom_contains=None):
    """Mark unresolved criticals resolved. Returns how many were cleared.

    Without this the ledger was write-only: a single crash critical blocked
    every future render (ep.preflight gates on it) and nothing — not the
    dashboard, not the CLI — could ever clear it.
    """
    try:
        with _ledger_guard():
            entries = _load_ledger()
            needle = (symptom_contains or "").strip().lower()
            cleared = 0
            for e in entries:
                if not isinstance(e, dict) or e.get("resolved"):
                    continue
                if e.get("severity") != "critical":
                    continue
                if needle and needle not in str(e.get("symptom", "")).lower():
                    continue
                e["resolved"] = True
                e["action"] = (str(e.get("action", "")) + " | resolved by owner").strip(" |")
                cleared += 1
            if cleared:
                _save_ledger(entries)
            return cleared
    except Exception as e:
        logger.warning("resolve_critical failed: %s", e)
        return 0


def loop(secs=120):
    print(f"supervisor loop every {secs}s (Ctrl+C to stop)")
    while True:
        # ponytail: check() is not allowed to kill this loop. record() takes a
        # cross-process ledger lock with a 15s timeout, so contention during a
        # slow cycle raises TimeoutError out of check() — and without this
        # guard the background supervisor died PERMANENTLY, taking the whole
        # services-down detection with it.
        try:
            findings = check(fix=True)
            for e in findings:
                print(f"[{e['severity']}] {e['symptom']} -> {e['action']}")
        except Exception as e:
            logger.warning("supervisor check failed: %s", e)
        time.sleep(secs)


if __name__ == "__main__":
    if "--loop" in sys.argv:
        secs = 120
        try:
            secs = int(sys.argv[sys.argv.index("--loop") + 1])
        except Exception:
            pass
        loop(secs)
    else:
        found = check(fix="--no-fix" not in sys.argv)
        for e in found:
            print(f"[{e['severity']}] {e['symptom']} -> {e['action']}")
        if not found:
            print("clear")
