"""Project Audit — "what should I do next?" plain-language status.

Imported from DramaClaw's Xia Director agent: reads project state
(progress, quality, crash logs, scripts, rendered episodes, open issues)
and returns human-readable findings + a suggested next action.
Nothing writes; it only observes.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .asset_library import is_scene_script


@dataclass
class Finding:
    level: str  # "ok", "info", "warn", "error"
    message: str
    action: str = ""

    def to_dict(self) -> dict:
        return {"level": self.level, "message": self.message, "action": self.action}


@dataclass
class AuditReport:
    findings: list[Finding] = field(default_factory=list)
    next_chapter: int | None = None
    next_reason: str = ""
    generated_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "next_chapter": self.next_chapter,
            "next_reason": self.next_reason,
            "generated_at": self.generated_at or time.time(),
        }


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return default


def _chapter_nums(root: Path) -> list[int]:
    nums = set()
    for f in (root / "chapters").glob("*.txt"):
        m = re.search(r"(\d+)", f.stem)
        if m:
            nums.add(int(m.group(1)))
    return sorted(nums)


def _rendered(root: Path) -> set[int]:
    out = set()
    for v in (root / "videos").glob("ch*.mp4"):
        m = re.search(r"ch(\d+)", v.stem)
        if m:
            out.add(int(m.group(1)))
    return out


def _scripted(root: Path) -> set[int]:
    out = set()
    for f in (root / "scenes").glob("ch*.json"):
        m = re.fullmatch(r"ch(\d{3})\.json", f.name)
        if m and is_scene_script(_read_json(f, None)):
            out.add(int(m.group(1)))
    return out


def audit_project(root: Path | None = None) -> AuditReport:
    root = root or Path(__file__).resolve().parent.parent.parent.parent
    rep = AuditReport()
    rep.generated_at = time.time()

    # 1. Open critical issues — these block renders by design.
    issues = _read_json(root / "issues.json", [])
    open_crit = [i for i in issues
                 if isinstance(i, dict) and i.get("severity") == "critical"
                 and not i.get("resolved")]
    if open_crit:
        rep.findings.append(Finding(
            "error",
            f"{len(open_crit)} open CRITICAL issue(s) block rendering",
            "Triage issues.json first — open the System tab / codex",
        ))
    else:
        rep.findings.append(Finding("ok", "No blocking critical issues", ""))

    # 2. Services health
    prog = _read_json(root / "progress_live.json", {})
    phase = prog.get("phase", "idle") if isinstance(prog, dict) else "idle"
    # service status is not in progress_live; rely on crash + features below

    # 3. Last run — crashed or stopped?
    try:
        crash = (root / "render_crash.log").read_text("utf-8", errors="ignore").strip()
    except Exception:
        crash = ""
    crash_lines = crash.splitlines()
    # ponytail: render_crash.log is append-only and nothing truncates it, so
    # this fired FOREVER after any single crash — every dashboard load said
    # "The last render ended in a crash". Gate on the timestamp INSIDE the most
    # recent entry, not the file mtime: any later, unrelated crash appends a
    # line and refreshes mtime, which re-arms the old finding forever (verified).
    headers = list(re.finditer(
        r"^===\s+(?:redo\s+)?(\d+)(?:/\d+)?\s+([^\n]*?)===\s*$",
        crash, flags=re.MULTILINE))
    crash_recent = False
    recent_chapter = None
    if headers:
        last = headers[-1]
        recent_chapter = int(last.group(1))
        # The header carries a wall-clock stamp: "2026-09-29 21:31:06".
        m = re.search(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})", last.group(2))
        if m:
            try:
                import datetime as _dt
                when = _dt.datetime.strptime(
                    f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S").timestamp()
                crash_recent = (time.time() - when) < 7 * 86400
            except ValueError:
                crash_recent = True   # unparsable stamp: assume recent, don't hide
        else:
            crash_recent = True       # no stamp: report rather than hide
    if crash_lines and phase != "done":
        # Which chapter last crashed is a FACT about the log and is useful at
        # any age — "the last render ended in a crash" pointing at chapter 9 is
        # correct even if it was three weeks ago. Only the ALARM level depends
        # on recency. These used to be conflated, and the 2026-09-17 header in
        # test_integrtegration proves the chapter extraction is independent.
        if recent_chapter is not None:
            rep.next_chapter = recent_chapter
            rep.next_reason = "crashed — review before retrying"
        if crash_recent:
            rep.findings.append(Finding(
                "warn", "The last render ended in a crash",
                "Open System tab → Crash log to see the traceback"))
        else:
            rep.findings.append(Finding(
                "info", "An older crash is in the log (nothing recent)",
                "Open System tab → Crash log if you want the old traceback"))

    # 4. State of last progress
    if phase == "error":
        rep.findings.append(Finding("error", "Last run ended in phase 'error'",
                                    "See Live Log for the failing step"))
    elif phase in ("done", "idle"):
        pass
    else:
        rep.findings.append(Finding(
            "info", f"Last run stopped mid-phase ('{phase}')",
            "Use 'Continue' to resume from checkpoint if interrupted"))

    # 5. Episodes: scripted vs rendered
    chapters = _chapter_nums(root)
    rendered = _rendered(root)
    scripted = _scripted(root)
    if not chapters:
        rep.findings.append(Finding("error", "No chapter text files found",
                                    "Drop .txt chapters into the chapters/ folder"))
    else:
        missing_script = [n for n in chapters if n not in scripted]
        missing_render = [n for n in chapters if n in scripted and n not in rendered]
        rep.findings.append(Finding(
            "info",
            f"{len(rendered)}/{len(chapters)} chapters rendered, "
            f"{len(scripted)} have scripts",
            "",))
        if missing_script:
            nxt = min(missing_script)
            if rep.next_chapter is None:
                rep.next_chapter = nxt
                rep.next_reason = "first chapter without a script"
            rep.findings.append(Finding(
                "info", f"{len(missing_script)} chapter(s) have no script yet",
                f"Render Ch {nxt} to write its script first",))
        elif missing_render:
            nxt = min(missing_render)
            if rep.next_chapter is None:
                rep.next_chapter = nxt
                rep.next_reason = "next scripted chapter that isn't rendered yet"
            rep.findings.append(Finding(
                "info", f"{len(missing_render)} scripted chapter(s) not rendered",
                f"Render Ch {nxt}",))

    # 6. Last quality signal
    quality = _read_json(root / "quality_log.json", [])
    if quality and isinstance(quality, list):
        last = quality[-1]
        score = last.get("score") if isinstance(last, dict) else None
        if score is not None:
            if score >= 70:
                rep.findings.append(Finding("ok", f"Last render scored {score}/100", ""))
            elif score >= 50:
                rep.findings.append(Finding(
                    "warn", f"Last render scored only {score}/100",
                    "Check quality_log.json for weak clips to redo"))
            else:
                rep.findings.append(Finding(
                    "warn", f"Last render scored {score}/100 — low",
                    "Redo weak scenes before shipping"))

    # 7. Work in progress gate.
    # ponytail: this was UNREACHABLE. Step 1 unconditionally appends either an
    # error or an "ok" finding, so rep.findings was never empty here. Deleted
    # rather than kept as a decorative second opinion; the "All clear" message
    # is emitted in step 1, where the decision is actually made.
    return rep