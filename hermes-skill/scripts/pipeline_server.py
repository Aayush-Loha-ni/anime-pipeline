#!/usr/bin/env python3
"""Anime Pipeline Dashboard — REST API + Web UI.

Features imported from DramaClaw:
  - Asset Library (characters, chapters, scenes with metadata)
  - Async Task Manager (status/progress/cancel/retry)
  - Script Review & Repair loops
  - Storyboard auto-generation
  - Modern dark-theme dashboard

Run: python pipeline_server.py  (then open http://localhost:8000/)
"""
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import asyncio
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

ROOT = Path(__file__).parent.parent.parent
os.environ["ANIME_NO_PROGRESS"] = "1"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).parent))
import episode_producer as ep
import supervisor as sup

# Pipeline imports
sys.path.insert(0, str(Path(__file__).parent))
from pipeline.asset_library import AssetLibrary
from pipeline.task_manager import TaskManager
from pipeline.script_reviewer import review_script
from pipeline.storyboard import generate_storyboard, save_storyboard, load_storyboard
from pipeline.project_audit import audit_project
from pipeline.beat_planner import beats_summary
from pipeline.camera_presets import all_presets
from pipeline import config as ini_config
from pipeline import utils as pipe_utils
from pipeline import captioner as captioner_mod
from pipeline import tokens as token_bucket
from pipeline import config_loader as yaml_config
# comfy_patch (426 lines) was imported here and never called — the live
# ComfyUI path builds workflows inline. The module is now deleted.
from pipeline.rights_manifest import (load_rights_manifest, save_rights_manifest,
                                     preflight_rights_gate, _commercial_status,
                                     _is_non_commercial)
from pipeline.director_package import load_director_package, save_director_package
from pipeline.preflight import check_rights, check_director_package
from pipeline.viral_narration import scene_grip
from pipeline.novels import get_novel, list_novels, create_novel, DEFAULT_NOVEL_ID

app = FastAPI(title="Sinners World Studio")
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# ponytail: default_limits does NOTHING without the middleware registered —
# slowapi only evaluates them from SlowAPIMiddleware._check_limits (see
# slowapi/middleware.py), so 22 routes were entirely unthrottled while the code
# implied global protection. Registered here so the promise is real.
# Must come after the exception handler, per slowapi's documented setup.
app.add_middleware(SlowAPIMiddleware)

# Load YAML config (cascading: config.yaml → local.yaml → dev.yaml)
YAML_CFG = yaml_config.load_config()
# No global AssetLibrary: the library is per-novel now (each novel has its own
# characters/ and scenes/), so routes build one from the requested novel.
tasks = TaskManager(max_concurrent=1)
STOP_FILE = ROOT / "stop.requested"

def _auto_start_services():
    """Start LLM, ComfyUI, TTS in background on dashboard boot."""
    def _bg():
        try:
            ep.ensure_services(auto_start=True, wait_s=120)
        except Exception:
            pass
    threading.Thread(target=_bg, daemon=True).start()


# ---------------------------------------------------------------------------
# Background supervisor
# ---------------------------------------------------------------------------

def _supervisor_loop():
    while True:
        try:
            time.sleep(120)
            # fix=False here: the watchdog / `supervisor.py --loop` owns
            # recovery. This loop only keeps the ledger honest, and a stray
            # `import pipeline_server` (uvicorn, IDE indexer, a test) must
            # never be able to launch Comfy Desktop into a live render.
            sup.check(fix=False)
        except Exception as e:
            logging.getLogger("supervisor").warning("supervisor check failed: %s", e)


import threading

# Static mounts
# The legacy novel relocated into novels/Example Novel/ when that folder
# exists; the mounts must point at the real folder or the legacy routes 404.
_SW = ROOT / "novels" / "Example Novel"
_LEGACY_ROOT = _SW if (_SW / "characters" / "characters.json").exists() else ROOT
for _mount, _dir in (("/videos", "videos"), ("/images", "images"),
                     ("/voice", "voice"), ("/characters", "characters")):
    try:
        app.mount(_mount, StaticFiles(directory=str(_LEGACY_ROOT / _dir)),
                  name=_mount.strip("/"))
    except Exception:
        pass


# ponytail: started from the startup hook, NOT at import. This used to be a
# bare module-level Thread().start() with fix=True, so importing this module
# booted ComfyUI / llama-server / cosy-server with no way to stop it.
@app.on_event("startup")
def _on_startup():
    _auto_start_services()
    threading.Thread(target=_supervisor_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _novel_or_404(novel_id):
    novel = get_novel(novel_id)
    if novel is None:
        raise HTTPException(404, f"unknown novel: {novel_id}")
    return novel


def _chapter_list(novel=None):
    novel = novel or get_novel(None)
    nums = set()
    for f in novel.chapter_files():
        m = re.search(r"(\d+)", f.stem)
        if m:
            nums.add(int(m.group(1)))
    out = []
    for n in sorted(nums):
        vids = sorted(novel.videos_dir().glob(f"ch{n:03d}_*.mp4"))
        out.append({"chapter": n, "done": bool(vids),
                    "video": vids[-1].name if vids else None,
                    "novel": novel.id, "novel_title": novel.title})
    return out


def _approval_hints(novel, chapter: int):
    """Plain-English rights/director blockers for the given chapter."""
    hints = []
    try:
        r = check_rights(novel.root)
        if not r.get("ok"):
            blockers = r.get("blockers") or []
            if blockers:
                for b in blockers:
                    hints.append(str(b).replace("BLOCKER: ", "") + " — confirm below")
            else:
                hints.append("Rights not confirmed — approve below")
        dp = novel.scenes_dir() / f"ch{chapter:03d}_director.json"
        if dp.exists():
            d = check_director_package(novel.root, chapter)
            if not d.get("ok"):
                hints.append(f"Director package for Ch {chapter} not approved — approve below")
    except Exception:
        pass
    return hints


def _preflight_reason(novel, chapter: int) -> str:
    """Actionable toast text when ep.preflight fails.

    ep.preflight stashes the exact problems in LAST_PREFLIGHT_PROBLEMS;
    naming the blocker beats "go read the log".
    """
    hints = _approval_hints(novel, chapter)
    if hints:
        return "; ".join(hints)
    problems = list(getattr(ep, "LAST_PREFLIGHT_PROBLEMS", []) or [])
    if problems:
        return "; ".join(str(p) for p in problems[:3])
    return "Preflight failed — see System → Live Log for details"


def _last_progress_hint() -> str:
    """One-line 'last: <phase> <detail>' so a dead render's task error
    names the phase without opening the Live Log. Never raises."""
    try:
        lp = ep._read_progress() or {}
        if lp.get("phase"):
            return f" — last: {lp['phase']} {lp.get('detail', '')}".rstrip()
    except Exception:
        pass
    return ""

def _approve_rights_manifest(m, now, root=None):
    """Owner approval: confirm consent and clear commercial use.

    A non-commercial entry (a fan work of someone else's novel) is NOT
    flipped to "yes": approving a RENDER must not silently claim publication
    rights the owner does not hold. It keeps its non-commercial status, and
    the release gate still blocks it.
    """
    for key in ("novel_adaptation_rights", "voice"):
        if key in m:
            m[key]["owner_confirmation"] = True
            if not _is_non_commercial(m[key].get("commercial_use", "")):
                m[key]["commercial_use"] = "yes"
            m[key]["reviewer"] = "owner"
            m[key]["reviewed_at"] = now
    for cat in ("image_models", "video_models", "llm_models", "vlm_models",
                "loras", "music", "sfx", "character_references", "entries"):
        for e in m.get(cat) or []:
            allowed, _ = _commercial_status(e.get("commercial_use", ""))
            if not allowed and not _is_non_commercial(e.get("commercial_use", "")):
                e["commercial_use"] = "yes"
                e["reviewer"] = "owner"
                e["reviewed_at"] = now
    save_rights_manifest(m, root or ROOT)

_svc_cache = {"ts": 0.0, "data": {"ok": False, "services": {"llm": False, "comfyui": False, "tts": False}}}
_svc_lock = threading.Lock()

async def _async_check_one(client, url, key, timeout=1.0):
    """Single service check — 1s timeout, returns (key, bool)."""
    try:
        r = await client.get(url, timeout=timeout)
        if key == "tts":
            data = r.json()
            return key, bool(data.get("status") == "ok" and data.get("loaded", False))
        return key, r.status_code == 200
    except Exception:
        return key, False

async def async_services_check(fast_check=True):
    """Parallel async health checks with 1s TTL cache. Returns {ok, services}."""
    now = time.time()
    with _svc_lock:
        if _svc_cache["data"] is not None and now - _svc_cache["ts"] < 3.0:
            return _svc_cache["data"]
    timeout = 1.0 if fast_check else 3.0
    llm_url = (ep.CONFIG.get("llm_url") or ep.CONFIG.get("ollama_url")
                or "http://127.0.0.1:8080").rstrip("/")
    comfy_url = ep.CONFIG.get("comfyui_url", "http://127.0.0.1:8188")
    tts_url = ep.CONFIG.get("tts_url", "http://127.0.0.1:5003")
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            _async_check_one(client, f"{llm_url}/v1/models", "llm", timeout),
            _async_check_one(client, f"{comfy_url}/system_stats", "comfyui", timeout),
            _async_check_one(client, f"{tts_url}/health", "tts", timeout),
        )
    status = {k: v for k, v in results}
    ok = all(status.values())
    data = {"ok": ok, "services": status}
    with _svc_lock:
        _svc_cache.update(ts=time.time(), data=data)
    # share with ep's 2s cache so supervisor/preflight see this result too
    ep._svc_cache_ep.update(ts=time.time(), data=(ok, status))
    return data

def _services_sync():
    """Sync fallback for non-async contexts (supervisor, background tasks)."""
    now = time.time()
    with _svc_lock:
        if _svc_cache["data"] is not None and now - _svc_cache["ts"] < 2.0:
            return _svc_cache["data"]
    ok, status = ep.ensure_services(auto_start=False)
    data = {"ok": ok, "services": status}
    with _svc_lock:
        _svc_cache.update(ts=time.time(), data=data)
        return _svc_cache["data"]


_gpu_cache = {"ts": 0.0, "data": {}}

def _gpu_stats():
    now = time.time()
    # ponytail: cache the FAILURE too. `if _gpu_cache["data"]` was falsy for
    # {}, so a missing/slow nvidia-smi re-ran a 10s subprocess on every
    # /api/status poll, once per browser tab, in the default executor.
    if now - _gpu_cache["ts"] < 3.0 and "ts" in _gpu_cache:
        return _gpu_cache["data"]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        parts = [x.strip() for x in out.split(",")]
        data = {"gpu_pct": float(parts[0]), "vram_used": float(parts[1]),
                "vram_total": float(parts[2]), "temp": float(parts[3]) if len(parts) > 3 else None}
    except Exception:
        data = {}
    with _svc_lock:
        _gpu_cache.update(ts=now, data=data)
    return data


def _render_alive():
    return ep.render_alive()


def _task_dict(task):
    data = task.to_dict()
    if (task.kind in ("render", "redo") and task.started_at is not None
            and task.finished_at is None and not task.is_cancelled):
        try:
            path = ROOT / "progress_live.json"
            if path.stat().st_mtime >= task.started_at:
                prog = json.loads(path.read_text(encoding="utf-8"))
                data.update(phase=prog.get("phase", data["phase"]),
                            detail=prog.get("detail", data["detail"]),
                            progress=max(0.0, min(100.0, float(prog.get("percent", 0)))))
        except (OSError, ValueError, TypeError, AttributeError) as e:
            logging.getLogger("task_dict").debug("progress read skipped: %s", e)
    if task.is_cancelled and task.finished_at is None:
        data.update(phase="stopping", detail="Stop requested; waiting for worker to exit")
    return data


def _crash_tail(n=8):
    try:
        lines = (ROOT / "render_crash.log").read_text(
            encoding="utf-8").strip().splitlines()
        return lines[-n:]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class StartReq(BaseModel):
    chapter: int
    novel: str = DEFAULT_NOVEL_ID


class RedoReq(BaseModel):
    chapter: int
    scene: int
    novel: str = DEFAULT_NOVEL_ID


class ReviewReq(BaseModel):
    chapter: int
    novel: str = DEFAULT_NOVEL_ID


class StoryboardReq(BaseModel):
    chapter: int
    novel: str = DEFAULT_NOVEL_ID


# ---------------------------------------------------------------------------
# API — Chapters & Status
# ---------------------------------------------------------------------------

@app.get("/api/chapters")
async def api_chapters(novel: str = DEFAULT_NOVEL_ID):
    return {"chapters": _chapter_list(_novel_or_404(novel))}


@app.get("/api/novels")
async def api_novels():
    return {"novels": [n.to_dict() for n in list_novels()]}


# A StaticFiles mount cannot be created for novels/ before the first novel
# exists, so novel media is served by an explicit route instead. `kind` is
# whitelisted and the resolved path must stay inside the novel's folder.
_NOVEL_MEDIA_DIRS = ("videos", "images", "voice", "characters", "scenes",
                     "release_package")


@app.get("/novels/{novel_id}/{kind}/{filename:path}")
async def api_novel_media(novel_id: str, kind: str, filename: str):
    novel = _novel_or_404(novel_id)
    if kind not in _NOVEL_MEDIA_DIRS:
        raise HTTPException(404, f"unknown media kind: {kind}")
    base = (novel.root / kind).resolve()
    path = (base / filename).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path)


class CreateNovelReq(BaseModel):
    title: str
    novel_id: str | None = None


@app.post("/api/novels")
async def api_novels_create(req: CreateNovelReq):
    try:
        novel = create_novel(req.title, req.novel_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"novel": novel.to_dict()}


@app.get("/api/status")
# ponytail: the UI polls this every 5s PER TAB (nicegui_dashboard.py:341).
# At 30/min three open tabs = 36/min and the dashboard showed itself as
# "API offline" — a self-inflicted outage. 240/min is 4/s, far above any
# real poll rate, and this binds 127.0.0.1 only.
@limiter.limit("240/minute")
async def api_status(request: Request, novel: str = DEFAULT_NOVEL_ID):
    novel_obj = _novel_or_404(novel)
    prog = {}
    try:
        with open(ROOT / "progress_live.json", encoding="utf-8") as f:
            prog = json.load(f)
    except Exception:
        pass
    loglines = prog.get("log", [])
    errors = [l for l in loglines
              if re.search(r"ERROR|FAIL|failed|error|WARN", l, re.I)][-8:]
    phase = prog.get("phase", "idle")
    # "stopped" is a clean user-initiated finish, not a stall — it was
    # missing here, so a deliberate Stop made the dashboard report
    # "stale: true" and render the whole step ladder as todo.
    stale = bool(phase not in ("idle", "done", "error", "starting", "stopped")
                 and not _render_alive()
                 and not tasks.active())

    # ponytail: "voice" before "images" because TTS now runs first — every
    # shot length is measured off the narration it will carry. The list was
    # left in the old order, so the dashboard ran the ladder backwards for the
    # whole clip phase.
    PHASES = ["starting", "script", "characters", "voice", "images", "clips",
              "cooling", "talking", "assembly", "done", "error"]
    order = {p: i for i, p in enumerate(PHASES)}
    cur = order.get(phase, -1)
    # ponytail: "stopped" was fixed in `stale` above but NOT here, so cur was
    # -1 and the ladder rendered every step as todo after a Stop. A terminal
    # phase sits past the LAST real step (len-1), never len-2 — that lands on
    # "assembly" and showed the server mid-render while idle.
    if phase in ("stopped", "idle"):
        cur = len(PHASES) - 1
    phases = [{"name": p,
               "state": ("done" if i < cur else "active" if i == cur else "todo")}
              for i, p in enumerate(PHASES)]

    vids = sorted((p for p in novel_obj.videos_dir().glob("ch*.mp4")
                   if ".candidate." not in p.name and not re.search(r"\.v\d+\.mp4$", p.name)),
                  key=lambda p: p.stat().st_mtime)

    # Active tasks
    active_tasks = [_task_dict(t) for t in tasks.active()]

    loop = asyncio.get_running_loop()
    svc, gpu = await asyncio.gather(
        async_services_check(fast_check=True),
        loop.run_in_executor(None, _gpu_stats),
    )

    return {
        "run": {"running": bool(tasks.active()), "alive": _render_alive()},
        "progress": prog,
        "services": svc,
        "videos": [v.name for v in vids[-10:]],
        "errors": errors,
        "crash": _crash_tail(),
        "gpu": gpu,
        "phases": phases,
        "stale": stale,
        "tasks": active_tasks,
        "task_history": tasks.history(10),
    }


# ---------------------------------------------------------------------------
# API — Asset Library
# ---------------------------------------------------------------------------

@app.get("/api/assets/summary")
async def api_assets_summary(novel: str = DEFAULT_NOVEL_ID):
    lib = AssetLibrary(_novel_or_404(novel).root)
    lib.invalidate()
    return lib.summary()


@app.get("/api/assets/chapters")
async def api_assets_chapters(novel: str = DEFAULT_NOVEL_ID):
    lib = AssetLibrary(_novel_or_404(novel).root)
    lib.invalidate()
    chapters = lib.load_chapters()
    return {"novel": novel, "chapters": [ch.to_dict() for ch in chapters.values()]}


@app.get("/api/assets/chapter/{chapter}")
async def api_assets_chapter(chapter: int, novel: str = DEFAULT_NOVEL_ID):
    lib = AssetLibrary(_novel_or_404(novel).root)
    lib.invalidate()
    ch = lib.get_chapter(chapter)
    if not ch:
        raise HTTPException(404, f"Chapter {chapter} not found")
    scenes = lib.load_scenes(chapter)
    return {**ch.to_dict(), "novel": novel,
            "scenes": [s.to_dict() for s in scenes]}


@app.get("/api/assets/characters")
async def api_assets_characters(novel: str = DEFAULT_NOVEL_ID):
    lib = AssetLibrary(_novel_or_404(novel).root)
    lib.invalidate()
    chars = lib.load_characters()
    return {"novel": novel, "characters": [c.to_dict() for c in chars.values()]}


@app.get("/api/assets/character/{key}")
async def api_assets_character(key: str, novel: str = DEFAULT_NOVEL_ID):
    char = AssetLibrary(_novel_or_404(novel).root).get_character(key)
    if not char:
        raise HTTPException(404, f"Character '{key}' not found")
    return char.to_dict()


# ---------------------------------------------------------------------------
# API — Tasks
# ---------------------------------------------------------------------------

@app.get("/api/tasks/active")
async def api_tasks_active():
    return {"tasks": [_task_dict(t) for t in tasks.active()]}


@app.get("/api/tasks/history")
async def api_tasks_history():
    return {"tasks": tasks.history(20)}


@app.get("/api/tasks/{task_id}")
async def api_tasks_get(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, f"Task '{task_id}' not found")
    return _task_dict(task)


@app.post("/api/tasks/{task_id}/cancel")
@limiter.limit("10/minute")
async def api_tasks_cancel(request: Request, task_id: str):
    if tasks.cancel(task_id):
        try:
            STOP_FILE.write_text("stop")
        except OSError as e:
            tasks.uncancel(task_id)
            return {"cancelled": False, "reason": str(e)}
        return {"cancelled": True}
    raise HTTPException(404, "Task not found or not running")


@app.post("/api/tasks/cancel-all")
@limiter.limit("10/minute")
async def api_tasks_cancel_all(request: Request):
    # ponytail: this returned {"cancelled": True} even when cancel_all() was
        # False, i.e. it claimed success for cancelling nothing. /api/stop
        # (below) already does this correctly; match it.
        if not tasks.cancel_all():
            return {"cancelled": False, "reason": "nothing to stop"}
        try:
            STOP_FILE.write_text("stop")
        except OSError as e:
            tasks.uncancel()
            return {"cancelled": False, "reason": str(e)}
        return {"cancelled": True}


# ---------------------------------------------------------------------------
# API — Render / Redo / Stop
# ---------------------------------------------------------------------------

@app.post("/api/start")
@limiter.limit("3/minute")
async def api_start(request: Request, req: StartReq):
    if tasks.active():
        return {"started": False, "reason": "a task is already running"}
    nid = getattr(req, "novel", None) or DEFAULT_NOVEL_ID
    novel = _novel_or_404(nid)
    if not any(c["chapter"] == req.chapter for c in _chapter_list(novel)):
        return {"started": False, "reason": "chapter not found"}
    # M9: server-side preflight gate
    pf = await asyncio.to_thread(ep.preflight, req.chapter, novel=nid)
    if not pf:
        return {"started": False, "reason": _preflight_reason(novel, req.chapter)}
    # Require services to be up before starting render (no auto_start in handler)
    svc = await async_services_check(fast_check=True)
    if not svc["ok"]:
        down = [k for k, v in svc["services"].items() if not v]
        return {"started": False, "reason": f"services down: {', '.join(down)} — press Start first"}
    ok, bal = token_bucket.spend(ini_config.token_cost_render())
    if not ok:
        return {"started": False,
                "reason": f"Not enough tokens (balance {bal}). Ask the admin to grant more."}

    def _run(task):
        try:
            if task.is_cancelled:
                return
            result = ep.produce_episode(req.chapter, novel=nid)
            if result is None:
                if ep.should_stop():
                    return {"ok": False, "reason": "stopped by user"}
                failed_at = getattr(ep, "_LAST_ASSEMBLY", {}).get("failed_at")
                if failed_at:
                    return {"ok": False, "reason": f"assembly failed at {failed_at} — see Live Log"}
                return {"ok": False, "reason": f"produce_episode returned None (gate/preflight failure){_last_progress_hint()} — see Live Log"}
            return result if isinstance(result, dict) else {"ok": True}
        except Exception:
            import traceback
            crash = ROOT / "render_crash.log"
            with open(crash, "a", encoding="utf-8") as f:
                f.write(f"\n=== {nid} ch{req.chapter} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(traceback.format_exc())
            ep.set_progress("error", 0, 1, "crashed, see render_crash.log")
            raise

    try:
        STOP_FILE.unlink(missing_ok=True)
        t = tasks.submit("render", req.chapter, _run, novel=nid)
    except (RuntimeError, OSError) as e:
        # Refund: the charge above happened before the submit, and a
        # capacity/lock failure used to burn the owner's tokens for a
        # render that never started.
        try:
            token_bucket.grant(ini_config.token_cost_render())
        except Exception:
            pass
        return {"started": False, "reason": str(e)}
    return {"started": True, "chapter": req.chapter, "novel": nid,
            "task_id": t.id}


@app.post("/api/continue")
@limiter.limit("3/minute")
async def api_continue(request: Request):
    if tasks.active():
        return {"started": False, "reason": "a task is already running"}
    try:
        raw = json.loads((ROOT / "render_chapter").read_text().strip())
    except Exception:
        raw = None
    if isinstance(raw, dict):
        novel_id = str(raw.get("novel") or DEFAULT_NOVEL_ID)
        chapter = int(raw.get("chapter"))
    elif isinstance(raw, (int, float)):
        novel_id, chapter = DEFAULT_NOVEL_ID, int(raw)
    else:
        return {"started": False, "reason": "no interrupted chapter found"}
    novel = _novel_or_404(novel_id)

    # Preflight gate: don't queue GPU work that will fail preflight again.
    pf = await asyncio.to_thread(ep.preflight, chapter, novel=novel_id)
    if not pf:
        return {"started": False, "reason": _preflight_reason(novel, chapter)}
    svc = await async_services_check(fast_check=True)
    if not svc["ok"]:
        down = [k for k, v in svc["services"].items() if not v]
        return {"started": False, "reason": f"services down: {', '.join(down)} — press Start first"}
    ok, bal = token_bucket.spend(ini_config.token_cost_render())
    if not ok:
        return {"started": False,
                "reason": f"Not enough tokens (balance {bal}). Ask the admin to grant more."}

    def _run(task):
        try:
            if task.is_cancelled:
                return
            result = ep.produce_episode(chapter, preserve=True, novel=novel_id)
            if result is None:
                if ep.should_stop():
                    return {"ok": False, "reason": "stopped by user"}
                failed_at = getattr(ep, "_LAST_ASSEMBLY", {}).get("failed_at")
                if failed_at:
                    return {"ok": False, "reason": f"assembly failed at {failed_at} — see Live Log"}
                return {"ok": False, "reason": f"produce_episode returned None (gate/preflight failure){_last_progress_hint()} — see Live Log"}
            return result if isinstance(result, dict) else {"ok": True}
        except Exception:
            import traceback
            crash = ROOT / "render_crash.log"
            with open(crash, "a", encoding="utf-8") as f:
                f.write(f"\n=== {novel_id} ch{chapter} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(traceback.format_exc())
            ep.set_progress("error", 0, 1, "crashed, see render_crash.log")
            raise

    try:
        STOP_FILE.unlink(missing_ok=True)
        t = tasks.submit("render", chapter, _run, novel=novel_id)
    except (RuntimeError, OSError) as e:
        try:
            token_bucket.grant(ini_config.token_cost_render())
        except Exception:
            pass
        return {"started": False, "reason": str(e)}
    return {"started": True, "chapter": chapter, "novel": novel_id,
            "task_id": t.id}


@app.post("/api/redo")
@limiter.limit("3/minute")
async def api_redo(request: Request, req: RedoReq):
    if tasks.active():
        return {"started": False, "reason": "a task is already running"}
    nid = getattr(req, "novel", None) or DEFAULT_NOVEL_ID
    novel = _novel_or_404(nid)
    if not any(c["chapter"] == req.chapter for c in _chapter_list(novel)):
        return {"started": False, "reason": "chapter not found"}
    try:
        with open(novel.scenes_dir() / f"ch{req.chapter:03d}.json", encoding="utf-8") as f:
            nscenes = len(json.load(f))
    except Exception:
        return {"started": False, "reason": "no script for chapter"}
    if not (1 <= req.scene <= nscenes):
        return {"started": False, "reason": f"scene out of range 1-{nscenes}"}
    # Preflight gate: redo does GPU work too.
    pf = await asyncio.to_thread(ep.preflight, req.chapter, novel=nid)
    if not pf:
        return {"started": False, "reason": _preflight_reason(novel, req.chapter)}
    svc = await async_services_check(fast_check=True)
    if not svc["ok"]:
        down = [k for k, v in svc["services"].items() if not v]
        return {"started": False, "reason": f"services down: {', '.join(down)} — press Start first"}
    ok, bal = token_bucket.spend(ini_config.token_cost_redo())
    if not ok:
        return {"started": False,
                "reason": f"Not enough tokens (balance {bal}). Ask the admin to grant more."}

    def _run(task):
        try:
            if task.is_cancelled:
                return
            result = ep.redo_scene(req.chapter, req.scene, novel=nid)
            if result is None:
                if ep.should_stop():
                    return {"ok": False, "reason": "stopped by user"}
                return {"ok": False, "reason": "redo_scene returned None (gate/preflight failure) — see Live Log"}
            return result if isinstance(result, dict) else {"ok": True}
        except Exception:
            import traceback
            crash = ROOT / "render_crash.log"
            with open(crash, "a", encoding="utf-8") as f:
                f.write(f"\n=== redo {nid} ch{req.chapter}/{req.scene} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(traceback.format_exc())
            ep.set_progress("error", 0, 1, "redo crashed, see render_crash.log")
            raise

    try:
        STOP_FILE.unlink(missing_ok=True)
        t = tasks.submit("redo", req.chapter, _run, scene=req.scene, novel=nid)
    except (RuntimeError, OSError) as e:
        try:
            token_bucket.grant(ini_config.token_cost_redo())
        except Exception:
            pass
        return {"started": False, "reason": str(e)}
    return {"started": True, "chapter": req.chapter, "scene": req.scene,
            "novel": nid, "task_id": t.id}


@app.post("/api/stop")
@limiter.limit("10/minute")
async def api_stop(request: Request):
    cancelled = tasks.cancel_all()
    if not cancelled and not ep.render_alive():
        # nothing running — don't leave a STOP_FILE that aborts the next run
        try:
            STOP_FILE.unlink(missing_ok=True)
        except OSError:
            pass
        return {"stopped": False, "reason": "nothing to stop"}
    try:
        STOP_FILE.write_text("stop")
    except Exception as e:
        return {"stopped": False, "reason": str(e)}
    return {"stopped": True}


# ---------------------------------------------------------------------------
# API — Script Review
# ---------------------------------------------------------------------------

@app.post("/api/review")
@limiter.limit("10/minute")
async def api_review(request: Request, req: ReviewReq):
    novel = _novel_or_404(getattr(req, "novel", None) or DEFAULT_NOVEL_ID)
    scenes_file = novel.scenes_dir() / f"ch{req.chapter:03d}.json"
    if not scenes_file.exists():
        raise HTTPException(404, f"No script for chapter {req.chapter}")
    try:
        scenes = json.loads(scenes_file.read_text("utf-8"))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse script: {e}")

    # Get known characters
    chars = AssetLibrary(novel.root).load_characters()
    known = set(chars.keys())

    live = bool(tasks.active()) or ep.render_alive()
    result = await asyncio.to_thread(review_script, scenes, req.chapter, known,
                                     persist=not live,
                                     scenes_dir=novel.scenes_dir())
    # Beat layer: annotate arcs without mutating the saved script.
    beats = beats_summary(scenes)
    return {**result.to_dict(), "beats": beats}


# ---------------------------------------------------------------------------
# API — Storyboard
# ---------------------------------------------------------------------------

@app.post("/api/storyboard/generate")
@limiter.limit("5/minute")
async def api_storyboard_generate(request: Request, req: StoryboardReq):
    novel = _novel_or_404(getattr(req, "novel", None) or DEFAULT_NOVEL_ID)
    scenes_file = novel.scenes_dir() / f"ch{req.chapter:03d}.json"
    if not scenes_file.exists():
        raise HTTPException(404, f"No script for chapter {req.chapter}")
    try:
        scenes = json.loads(scenes_file.read_text("utf-8"))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse script: {e}")

    sb = generate_storyboard(req.chapter, scenes)
    out = save_storyboard(sb, storyboard_dir=novel.scenes_dir())
    return {**sb.to_dict(), "saved": str(out)}


@app.get("/api/storyboard/{chapter}")
async def api_storyboard_get(chapter: int, novel: str = DEFAULT_NOVEL_ID):
    sb = load_storyboard(chapter, _novel_or_404(novel).scenes_dir())
    if not sb:
        raise HTTPException(404, f"No storyboard for chapter {chapter}")
    return sb.to_dict()


# ---------------------------------------------------------------------------
# API — Pins (commit-to-series: mark good frames so redo skips them)
# ---------------------------------------------------------------------------

PINS_FILE = (_SW if (_SW / "characters" / "characters.json").exists() else ROOT) / "pinned.json"


class PinReq(BaseModel):
    chapter: int
    scene: int
    still: str = ""
    note: str = ""
    novel: str = DEFAULT_NOVEL_ID


class PinDelReq(BaseModel):
    chapter: int
    scene: int
    novel: str = DEFAULT_NOVEL_ID


def _read_pins(novel=None):
    path = novel.pins_file() if novel is not None else PINS_FILE
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_pins_lock = threading.Lock()


def _write_pins(data: dict, novel=None):
    # ponytail: unique temp name (a fixed ".tmp" collides with a concurrent
    # writer) and no FileExistsError branch — os.replace() overwrites by
    # definition, so that branch was unreachable AND would have DELETED
    # pinned.json if it ever fired. Callers hold _pins_lock for the
    # read-modify-write.
    path = novel.pins_file() if novel is not None else PINS_FILE
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


@app.get("/api/pins")
async def api_pins(chapter: int | None = None, novel: str = DEFAULT_NOVEL_ID):
    data = _read_pins(_novel_or_404(novel))
    if chapter is not None:
        return {"pins": data.get(f"ch{chapter:03d}", {})}
    return {"pins": data}


@app.post("/api/pins")
async def api_pins_add(req: PinReq):
    novel = _novel_or_404(getattr(req, "novel", None) or DEFAULT_NOVEL_ID)
    # read-modify-write under one lock — two concurrent pins used to
    # clobber each other's entry.
    with _pins_lock:
        data = _read_pins(novel)
        ch_key = f"ch{req.chapter:03d}"
        cur = data.get(ch_key)
        if not isinstance(cur, dict):
            cur = {}
            data[ch_key] = cur
        cur[str(req.scene)] = {
            "still": req.still,
            "note": req.note,
            "pinned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_pins(data, novel)
    return {"pinned": True, "chapter": req.chapter, "scene": req.scene}


@app.delete("/api/pins")
async def api_pins_del(req: PinDelReq):
    novel = _novel_or_404(getattr(req, "novel", None) or DEFAULT_NOVEL_ID)
    with _pins_lock:
        data = _read_pins(novel)
        ch_key = f"ch{req.chapter:03d}"
        if ch_key in data and str(req.scene) in data[ch_key]:
            del data[ch_key][str(req.scene)]
            if not data[ch_key]:
                del data[ch_key]
            _write_pins(data, novel)
            return {"unpinned": True}
    return {"unpinned": False, "reason": "not pinned"}


# ---------------------------------------------------------------------------
# API — Scenes (legacy compat)
# ---------------------------------------------------------------------------

@app.get("/api/scenes")
async def api_scenes(chapter: int = 1, novel: str = DEFAULT_NOVEL_ID):
    scenes = AssetLibrary(_novel_or_404(novel).root).load_scenes(chapter)
    return {"scenes": [s.to_dict() for s in scenes]}


@app.get("/api/characters")
async def api_characters(novel: str = DEFAULT_NOVEL_ID):
    chars = AssetLibrary(_novel_or_404(novel).root).load_characters()
    return {"characters": [c.to_dict() for c in chars.values()]}


@app.get("/api/rules")
async def api_rules():
    rules = {}
    try:
        for p in sorted((ROOT / "rules").glob("*.md")):
            rules[p.stem] = p.read_text(encoding="utf-8")
    except Exception:
        pass
    return {"rules": rules}


@app.get("/api/quality")
async def api_quality(novel: str = DEFAULT_NOVEL_ID):
    n = _novel_or_404(novel)
    try:
        with open(ROOT / "quality_log.json", encoding="utf-8") as f:
            hist = [e for e in json.load(f) if isinstance(e, dict)
                    and e.get("novel", DEFAULT_NOVEL_ID) == n.id]
        last = hist[-1] if hist else {}
    except Exception:
        last = {}
    try:
        with open(ROOT / "cost_log.json", encoding="utf-8") as f:
            costs = [e for e in json.load(f) if isinstance(e, dict)
                     and e.get("novel", DEFAULT_NOVEL_ID) == n.id]
    except Exception:
        costs = []
    return {"latest": last, "renders": len(costs)}


@app.get("/api/issues")
@limiter.limit("6/minute")
async def api_issues(request: Request):
    # ponytail: the comment here claimed this avoided blocking, but it called a
    # SYNCHRONOUS wrapper (ep.ensure_services -> ep.http_get, which retries 3x
    # with a 5s sleep) straight from an `async def`. Measured on this box with
    # the LLM and TTS down: llm_health 16.1s + /health 22.2s = 38s, worst case
    # ~75s, with the single-threaded event loop blocked the whole time.
    #
    # That is not just latency: watchdog.bat probes /api/chapters with
    # `curl --max-time 5`, fails 3x, then kills and relaunches the dashboard —
    # so one cold /api/issues call could take the server down.
    #
    # asyncio.to_thread is the documented offload for calling blocking code
    # from an async endpoint (FastAPI's own maintainer recommends
    # run_in_threadpool/to_thread); to_thread is 3.9+ and copies contextvars.
    svc = await asyncio.to_thread(_services_sync)
    issues = await asyncio.to_thread(sup.open_issues)
    return {"live": bool(tasks.active()), "services": svc, **issues}


@app.post("/api/issues/resolve")
@limiter.limit("10/minute")
async def api_issues_resolve(request: Request, symptom: str = ""):
    """Clear unresolved criticals so preflight stops blocking renders.

    The ledger was write-only before this route: ep.preflight refuses to
    render while any critical is unresolved, and nothing could ever mark
    one resolved, so a single crash blocked every future render
    permanently. Omit `symptom` to clear all of them.
    """
    cleared = await asyncio.to_thread(sup.resolve_critical, symptom or None)
    return {"resolved": cleared,
            "message": (f"cleared {cleared} critical(s)" if cleared
                        else "no matching unresolved criticals")}


@app.get("/api/audit")
async def api_audit():
    """Plain-language 'what should I do next?' report (Xia Director style)."""
    return audit_project(ROOT).to_dict()


# ---------------------------------------------------------------------------
# API — Camera Presets (cinema lens/focal/aperture → prompt modifiers)
# ---------------------------------------------------------------------------

@app.get("/api/camera-presets")
async def api_camera_presets():
    return all_presets()


# ---------------------------------------------------------------------------
# API — Gallery (all takes per scene: stills + clips, for browsing/picking)
# ---------------------------------------------------------------------------

@app.get("/api/gallery")
async def api_gallery(chapter: int = 1, scene: int = 1,
                       novel: str = DEFAULT_NOVEL_ID):
    """List every still and clip for a scene, for the generations gallery."""
    img_dir = _novel_or_404(novel).images_dir()
    ch = f"ch{chapter:03d}"
    sn = f"s{scene:02d}"

    if not img_dir.is_dir():
        return {"chapter": chapter, "scene": scene, "stills": [], "clips": []}

    # All stills for this scene (including b_ and c_ variants)
    stills = sorted(img_dir.glob(f"{ch}_{sn}_*.png"))
    # All clips
    clips = sorted(img_dir.glob(f"{ch}_{sn}*_an*.mp4"))
    if not clips:
        clips = sorted(img_dir.glob(f"{ch}_echo{scene:02d}.mp4"))
    if not clips:
        clips = sorted(img_dir.glob(f"{ch}_h3{scene:02d}.mp4"))
    if not clips:
        clips = sorted(img_dir.glob(f"{ch}_ltx{scene:02d}.mp4"))
    if not clips:
        clips = sorted(img_dir.glob(f"{ch}_{sn}*.mp4"))

    def _info(p):
        name = p.name
        # Classify variant
        variant = "main"
        if f"{sn}b_" in name:
            variant = "alt-b"
        elif f"{sn}c_" in name:
            variant = "alt-c"
        # Extract seed from name like ch001_s01_00001_.png
        seed = ""
        parts = name.split("_")
        for part in parts:
            if part.isdigit() and len(part) >= 4:
                seed = part
                break
        return {"name": name, "variant": variant, "seed": seed,
                "size": p.stat().st_size, "mtime": p.stat().st_mtime}

    return {
        "chapter": chapter, "scene": scene,
        "stills": [_info(p) for p in stills],
        "clips": [_info(p) for p in clips],
    }


# ---------------------------------------------------------------------------
# API — Caption (BLIP auto-describe), Models (local scan), Tokens (bucket)
# ---------------------------------------------------------------------------

class CaptionReq(BaseModel):
    image: str = ""  # images-dir filename or absolute path
    novel: str = DEFAULT_NOVEL_ID


@app.post("/api/caption")
async def api_caption(req: CaptionReq):
    """Auto-describe a still via local BLIP. "" when captioner unavailable."""
    name = (req.image or "").strip()
    if not name:
        raise HTTPException(400, "image must be a filename")
    # ponytail: path traversal. `ROOT/"images"/name` with an unvalidated name
    # let "../../../anything" resolve outside images/ — verified: a relative
    # escape and an absolute path both returned 200 and handed the path to
    # PIL, and the 404-vs-200 split was an existence oracle for any path on
    # disk even with BLIP unloaded. The endpoint is "caption a still", so the
    # only legitimate input is a file inside images/.
    images_root = _novel_or_404(getattr(req, "novel", None) or DEFAULT_NOVEL_ID).images_dir().resolve()
    candidate = Path(name)
    if candidate.is_absolute():
        p = candidate.resolve()
    else:
        p = (images_root / candidate).resolve()
    if not p.is_relative_to(images_root):
        raise HTTPException(400, "image must be inside the images directory")
    if not p.is_file():
        raise HTTPException(404, f"Image not found: {name}")
    if not captioner_mod.available():
        return {"caption": "", "available": False}
    return {"caption": await asyncio.to_thread(captioner_mod.describe_image, p),
            "available": True}


@app.get("/api/models")
async def api_models():
    """Local .safetensors checkpoints visible to the pipeline."""
    try:
        ckpt = str(ep.CONFIG.get("checkpoint", ""))
    except Exception:
        ckpt = ""
    seen: dict[str, None] = {}
    for folder in (ROOT / "models", Path(ckpt).parent if ckpt else None):
        if not folder:
            continue
        try:
            if Path(folder).is_dir():
                for m in pipe_utils.get_all_local_models(str(folder)):
                    seen[m] = None
        except Exception:
            pass
    return {"models": sorted(seen), "active_checkpoint": ckpt}


@app.get("/api/tokens")
async def api_tokens():
    return token_bucket.status()


class TokenGrantReq(BaseModel):
    n: int = 0
    reason: str = ""


@app.post("/api/tokens/grant")
@limiter.limit("5/minute")
async def api_tokens_grant(request: Request, req: TokenGrantReq):
    n = max(0, int(req.n or 0))
    if not n:
        n = ini_config.token_grant_new_image()
    # tokens._save deliberately raises on a failed write, so an unwrapped grant
    # surfaced as a bare 500 with no detail — while the two mutating config
    # routes name the cause. Granting is a ledger write; report it like one.
    try:
        bal = token_bucket.grant(n, req.reason)
    except Exception as e:
        raise HTTPException(500, f"Could not write token ledger: {e}") from e
    return {"balance": bal, "granted": n}


# ---------------------------------------------------------------------------
# API — Beginner-safe settings (curated subset of config.json)
# ---------------------------------------------------------------------------

SETTINGS_SCHEMA = [
    {"key": "platform", "group": "Video", "label": "Platform",
     "hint": "youtube = widescreen 1920x1080, vertical = 9:16 phone format",
     "type": "select", "options": ["youtube", "vertical"]},
    {"key": "video_engine", "group": "Video", "label": "Motion engine",
     "hint": "h3 = MiniMax H3 Ref2VA int8 (sole motion engine; fast draft / final via Quality below)",
     "type": "select", "options": ["h3"]},
{"key": "h3_steps", "group": "Video", "label": "Motion quality steps",
        # ponytail: this used to say "8 = PDD distilled (recommended)". Both
        # config.yaml and episode_producer's own defaults record that 8 steps
        # produced clips which ignored the reference still entirely, and that
        # the verified Comfy-Org recipe is 20. Following the old hint was a
        # silent quality regression.
        "hint": "20 = the verified recipe (Comfy-Org Ref2VA). LOWER values "
                "produce clips that ignore the reference still — do not.",
        "type": "int", "min": 1, "max": 100},
    {"key": "subtitles", "group": "Video", "label": "Burn in captions",
     "hint": "Captions burned into the picture. OFF ships a .srt sidecar only, "
             "which is invisible unless the viewer turns captions on. "
             "Costs one extra video encode per episode.",
     "type": "bool"},
    {"key": "kinetic_captions", "group": "Video", "label": "Animated (pop) captions",
     "hint": "Pop style for Shorts. Captions are one cue per line by default, "
             "not word-by-word; this only changes the style.",
     "type": "bool"},
    {"key": "caption_outline", "group": "Video", "label": "Caption outline (px)",
     "hint": "5 is the standard for bold sans over dark footage; 3 or less "
             "disappears against deep blacks.",
     "type": "int", "min": 0, "max": 10},
    {"key": "caption_safe_margin_v", "group": "Video", "label": "Vertical caption dead zone (px)",
     "hint": "VERTICAL ONLY — keeps captions clear of the platform's own "
             "buttons. Horizontal always uses 10% of frame height.",
     "type": "int", "min": 0, "max": 800},
    {"key": "shorts_enabled", "group": "Video", "label": "Also make vertical Shorts cut",
     "hint": "Auto-builds a 9:16 version after each episode.",
     "type": "bool"},
    {"key": "voice_speed", "group": "Voice", "label": "Voice speed",
     "hint": "1.0 = natural. Above 1.1 sounds chipmunked — don't.",
     "type": "float", "min": 0.8, "max": 1.2},
    {"key": "narration_only", "group": "Voice", "label": "Single narrator voice",
     "hint": "ON = one consistent narrator (recommended). OFF = per-character dialogue voices.",
     "type": "bool"},
    {"key": "lipsync_engine", "group": "Voice", "label": "Lip-sync engine",
     "hint": "echo = animate still portrait to voice (default). latentsync = sync lips on existing video clip. s2v = speech-to-video fallback.",
     "type": "select", "options": ["echo", "latentsync", "s2v"]},
    {"key": "echo_enabled", "group": "Voice", "label": "Animate talking heads",
     "hint": "Takes a still close-up face + voice audio → talking video. Best for portrait shots. Skips automatically when no face detected.",
     "type": "bool"},
    {"key": "music_volume", "group": "Audio", "label": "Music bed volume",
     "hint": "0 = silent tails (by design). 0.15 = gentle bed under voice.",
     "type": "float", "min": 0.0, "max": 0.5},
    {"key": "animation_style", "group": "Look", "label": "Animation style",
     "hint": "Prompt-level look for stills + motion on the same local model. Changing it regenerates stills.",
     "type": "select", "options": ["cinematic", "retro-cel", "painterly", "gothic", "noir", "soft"]},
    {"key": "image_seed_base", "group": "Look", "label": "Visual take",
     "hint": "Same script + same number = same pictures. Bump for a fresh look (e.g. 2000 -> 3000).",
     "type": "int", "min": 0, "max": 999999},
{"key": "image_cfg", "group": "Look", "label": "Style strength",
        # ponytail: this advertised 4.0-8.0 with the hint "5-6 is the sweet
        # spot", while ep._CONFIG_SCHEMA caps image_cfg at 4.0, the shipped
        # value is 1.0, and the producer's own comment says "cfg above ~2
        # oversaturates and adds edge artefacts; the official workflow does not
        # use CFG at all". The settings surface was pushing the live render
        # into the exact regime the producer calls broken. Bounds and hint now
        # match _CONFIG_SCHEMA, and api_settings_set re-validates after the
        # mutation so this UI can never write an out-of-range value again.
        "hint": "1.0 = verified (flow-matching; the official Qwen-Image "
                "workflow uses no CFG). Above ~2 oversaturates and adds edge "
                "artefacts.",
        "type": "float", "min": 0.0, "max": 4.0},
    {"key": "face_check_enabled", "group": "Look", "label": "Head-in-frame check",
     "hint": "Needs a staged vision model; currently checks are skipped (prompt head-lock is the real net).",
     "type": "bool"},
    {"key": "face_check_max_reframes", "group": "Look", "label": "Max face fixes per episode",
     "hint": "Ceiling on extra picture renders. 4 is plenty.",
     "type": "int", "min": 2, "max": 6},
    {"key": "motion_prompt_extend", "group": "Advanced", "label": "AI motion prompt expansion",
     "hint": "Uses the text LLM to enrich motion detail. Off = faster, still good.",
     "type": "bool"},
    {"key": "h3_continuity_relay", "group": "Look", "label": "Continuity relay",
     "hint": "Feed the previous shot's last frame to H3 as a second reference so "
             "the face, hair, wardrobe and lighting carry across cuts. ON = "
             "fewer identity jumps; OFF = slightly faster.",
     "type": "bool"},
]


def _read_disk_config():
    try:
        with open(ep.CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@app.get("/api/settings")
async def api_settings_get():
    disk = _read_disk_config()
    values = {s["key"]: disk.get(s["key"], ep.CONFIG.get(s["key"])) for s in SETTINGS_SCHEMA}
    return {"schema": SETTINGS_SCHEMA, "values": values}


class SettingsReq(BaseModel):
    key: str
    value: object = None


@app.post("/api/settings")
@limiter.limit("10/minute")
async def api_settings_set(request: Request, req: SettingsReq):
    spec = next((s for s in SETTINGS_SCHEMA if s["key"] == req.key), None)
    if not spec:
        raise HTTPException(400, f"Setting '{req.key}' is not dashboard-editable")
    v = req.value
    try:
        t = spec["type"]
        if t == "bool":
            if isinstance(v, str):
                v = v.lower() in ("1", "true", "yes", "on")
            v = bool(v)
        elif t == "int":
            v = int(v)
            v = max(spec["min"], min(spec["max"], v))
        elif t == "float":
            v = float(v)
            v = max(spec["min"], min(spec["max"], v))
        elif t == "select":
            if v not in spec["options"]:
                raise HTTPException(400, f"Must be one of {spec['options']}")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, f"Bad value for {req.key}")
    cfg_path = ep.CONFIG_FILE
    try:
        disk = _read_disk_config()
        disk[req.key] = v
        ep._atomic_write_json(cfg_path, disk, indent=2)
    except Exception as e:
        raise HTTPException(500, f"Could not save config: {e}")
    live_applied = True
    if _live_render():
        live_applied = False  # disk saved; live render keeps its snapshot
    else:
        try:
            ep.CONFIG[req.key] = v  # live-update the running server
            # Defence in depth: the producer's schema is the contract, and this
            # path used to bypass it entirely (a bounds typo shipped as
            # image_cfg 4.0-8.0 while the producer capped it at 4).
            #
            # It is currently UNREACHABLE for every dashboard key, and that is
            # deliberate rather than redundant: test_settings_bounds_never_
            # exceed_the_producer_schema guarantees each SETTINGS_SCHEMA range
            # is a subset of _CONFIG_SCHEMA, so the endpoint's own clamp at
            # :1186-1191 always lands in range first. Keeping the call means a
            # future key added to _CONFIG_SCHEMA alone cannot be written out of
            # range from this route — the invariant is enforced by a test, not
            # by remembering to add a bounds entry here.
            #
            # No snapshot/rollback: _validate_config coerces and clamps IN
            # PLACE and never raises (verified: a non-numeric image_seed_base
            # is repaired to the shipped default, no exception). An earlier
            # version here took a `before = dict(ep.CONFIG)` snapshot and
            # assigned it to `_`, which read as an unimplemented rollback.
            ep._validate_config(ep.CONFIG, getattr(ep, "_DEFAULTS", {}))
            v = ep.CONFIG.get(req.key, v)
            if v != req.value:
                # clamped or coerced by the producer's schema: persist what
                # actually took effect, not what was asked for.
                disk[req.key] = v
                try:
                    ep._atomic_write_json(cfg_path, disk, indent=2)
                except Exception as e:
                    logging.getLogger("settings").warning(
                        "could not persist clamped %s: %s", req.key, e)
        except Exception as e:
            live_applied = False
            logging.getLogger("settings").warning(
                "live settings update failed for %s: %s", req.key, e)
    effective = ep.CONFIG.get(req.key)
    return {"saved": {req.key: v}, "live_applied": live_applied,
            "effective": effective, "requested": req.value,
            "clamped": effective != req.value}


# ---------------------------------------------------------------------------
# API — YAML Config (two-tier model configs, ComfyUI patching)
# ---------------------------------------------------------------------------

@app.get("/api/config/yaml")
async def api_yaml_config():
    """Full merged YAML config (cascading files + env)."""
    return YAML_CFG


@app.get("/api/config/engine")
async def api_active_engine():
    """Active engine + quality tier with resolved model params."""
    return yaml_config.get_active_engine_config(YAML_CFG)


class EngineConfigReq(BaseModel):
    engine: str = "h3"  # h3 only (MiniMax H3 Ref2VA int8)
    quality: str = "fast"  # fast | final


@app.post("/api/config/engine")
async def api_set_engine(req: EngineConfigReq):
    """Switch active engine / quality tier for next render.

    ponytail: this used to write ONLY config.yaml and return the value it was
    asked for, reporting success for a change that could not happen. Two
    independent reasons why:
      1. config.json already declares video_engine and quality, and
         episode_producer.load_config() makes JSON win over YAML (it prints
         "config.yaml=... ignored - config.json=... is authoritative").
      2. ep.CONFIG is assigned once at import and never reloaded, so even a
         correct disk write would not reach the next render started in this
         process.

    Verified before this fix: config.yaml quality=final, ep.CONFIG['quality']
    still 'fast'. So write the AUTHORITATIVE file, validate with the
    producer's own validator, and update the live CONFIG — then report what is
    actually in effect rather than what was requested.
    """
    if req.engine not in ("h3",):
        raise HTTPException(400, "engine must be h3")
    if req.quality not in ("fast", "final"):
        raise HTTPException(400, "quality must be fast or final")
    YAML_CFG.setdefault("generation", {})["video_engine"] = req.engine
    YAML_CFG["generation"]["quality"] = req.quality

    # ponytail: do NOT mirror into pipeline/config.yaml. config.json is
    # authoritative and wins every merge, so a YAML write accomplishes nothing
    # except damage: yaml.dump() rewrites the whole file and destroys the 46
    # lines of comments that document it, then leaves a value that disagrees
    # with the real one — which is exactly the drift test_config_precedence
    # fails on. Write-only decoration is worse than no decoration.

    # config.json is what the render actually reads.
    disk = ep.read_json_file(ep.CONFIG_FILE, {}) or {}
    disk["video_engine"] = req.engine
    disk["quality"] = req.quality
    try:
        ep._atomic_write_json(ep.CONFIG_FILE, disk, indent=2)
    except Exception as e:
        raise HTTPException(500, f"config.json not written: {e}") from e
    # Live-update this process so the NEXT render here uses it too.
    ep.CONFIG["video_engine"] = req.engine
    ep.CONFIG["quality"] = req.quality
    ep._validate_config(ep.CONFIG, getattr(ep, "_DEFAULTS", {}))
    effective = ep.CONFIG.get("quality")
    return {"video_engine": ep.CONFIG.get("video_engine"), "quality": effective,
            "requested_quality": req.quality,
            "applied": effective == req.quality,
            "authoritative_file": str(ep.CONFIG_FILE)}


# ---------------------------------------------------------------------------
# API — Preflight (beginner-readable render readiness)
# ---------------------------------------------------------------------------

@app.get("/api/preflight")
@limiter.limit("30/minute")
async def api_preflight(request: Request, chapter: int = 1,
                        novel: str = DEFAULT_NOVEL_ID):
    n = _novel_or_404(novel)
    svc = await async_services_check(fast_check=True)
    ok, status = svc["ok"], svc["services"]
    txt = n.chapter_file(chapter)
    scenes_file = n.scenes_dir() / f"ch{chapter:03d}.json"
    hints = []
    if not ok:
        down = [k for k, v in status.items() if not v]
        hints.append(f"These need starting: {', '.join(down)} — press 'Start all services'.")
    if not txt:
        hints.append(f"No chapter text found for Ch {chapter}.")
    if not scenes_file.exists():
        hints.append("No script yet — one will be written on first render (takes a few minutes).")
    pf = (await asyncio.to_thread(ep.preflight, chapter, novel=novel)) if (txt and ok) else False
    if txt and ok and not pf:
        approval = _approval_hints(n, chapter)
        if approval:
            hints.extend(approval)
        else:
            hints.append("Preflight found a problem — see Live Log / issues in System.")
    rights = check_rights(n.root)
    director_path = n.scenes_dir() / f"ch{chapter:03d}_director.json"
    director = check_director_package(n.root, chapter) if director_path.exists() else {"ok": True, "detail": "will be generated on first render"}
    approvals = [
        {"item": "rights", "ok": bool(rights.get("ok")),
         "detail": rights.get("detail", ""),
         "blockers": rights.get("blockers", [])},
        {"item": "director", "ok": bool(director.get("ok")),
         "detail": director.get("detail", ""),
         "exists": director_path.exists()},
    ]
    rights_ok = bool(rights.get("ok"))
    director_ok = bool(director.get("ok"))
    ready = bool(txt and ok and pf and rights_ok and director_ok)
    return {"ready": ready, "services": status, "services_ok": ok,
            "chapter_text": bool(txt), "script_ready": scenes_file.exists(),
            "preflight_ok": pf, "hints": hints, "approvals": approvals,
            "rights_ok": rights_ok, "director_ok": director_ok}


class ApproveReq(BaseModel):
    item: str  # rights | director
    chapter: int = 1
    novel: str = DEFAULT_NOVEL_ID


@app.post("/api/approve")
@limiter.limit("10/minute")
async def api_approve(request: Request, req: ApproveReq):
    n = _novel_or_404(getattr(req, "novel", None) or DEFAULT_NOVEL_ID)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if req.item == "rights":
        m = load_rights_manifest(n.root)
        _approve_rights_manifest(m, now, n.root)
        ok, blockers, _ = preflight_rights_gate(m)
        return {"ok": ok, "item": "rights", "blockers": blockers}
    if req.item == "director":
        pkg = load_director_package(n.scenes_dir(), req.chapter)
        if pkg is None:
            return {"ok": False, "reason": f"No director package for Ch {req.chapter} yet — render once to generate it"}
        if pkg.get("version") == "placeholder":
            return {"ok": False, "reason": "Director package is a placeholder — generate real content first"}
        pkg["owner_approved"] = True
        pkg["approved_by"] = "owner"
        pkg["approved_at"] = now
        save_director_package(pkg, n.scenes_dir(), req.chapter, force=True)
        return {"ok": True, "item": "director", "chapter": req.chapter}
    return {"ok": False, "reason": f"unknown approval item: {req.item}"}


class ApproveAllReq(BaseModel):
    chapters: list[int] = []
    novel: str = DEFAULT_NOVEL_ID

@app.post("/api/approve-all")
@limiter.limit("5/minute")
async def api_approve_all(request: Request, req: ApproveAllReq):
    n = _novel_or_404(getattr(req, "novel", None) or DEFAULT_NOVEL_ID)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    chapters = req.chapters or [d["chapter"] for d in _chapter_list(n)]
    approved_dirs = []
    m = load_rights_manifest(n.root)
    _approve_rights_manifest(m, now)
    for chapter in chapters:
        dp = n.scenes_dir() / f"ch{chapter:03d}_director.json"
        if not dp.exists():
            continue
        pkg = load_director_package(n.scenes_dir(), chapter)
        if pkg is None or pkg.get("version") == "placeholder":
            continue
        if not pkg.get("owner_approved"):
            pkg["owner_approved"] = True
            pkg["approved_by"] = "owner"
            pkg["approved_at"] = now
            save_director_package(pkg, n.scenes_dir(), chapter, force=True)
            approved_dirs.append(chapter)
    return {"ok": True, "approved_director": approved_dirs, "rights": True}

@app.post("/api/services/start")
@limiter.limit("2/minute")
async def api_services_start(request: Request):
    if tasks.active():
        return {"started": False, "reason": "a task is already running"}

    def _run(task):
        if task.is_cancelled:
            return
        ok, status = ep.ensure_services(auto_start=True)
        return {"ok": ok, "reason": "" if ok else "services unavailable",
                "services": status}

    try:
        t = tasks.submit("services", 0, _run)
    except (RuntimeError, OSError) as e:
        return {"started": False, "reason": str(e)}
    return {"started": True, "task_id": t.id}


# ---------------------------------------------------------------------------
# API — Scene status (per-scene artifacts + redo)
# ---------------------------------------------------------------------------

@app.get("/api/scenes/status")
async def api_scenes_status(chapter: int = 1, novel: str = DEFAULT_NOVEL_ID):
    n = _novel_or_404(novel)
    sf = n.scenes_dir() / f"ch{chapter:03d}.json"
    if not sf.exists():
        raise HTTPException(404, f"No script for chapter {chapter}")
    try:
        scenes = json.loads(sf.read_text(encoding="utf-8"))
        if isinstance(scenes, dict):
            scenes = scenes.get("scenes", [])
    except Exception as e:
        raise HTTPException(400, f"Failed to parse script: {e}")
    out = []
    for i, s in enumerate(scenes):
        if not isinstance(s, dict):
            continue
        num = s.get("scene_number", i + 1)
        stills = [p for p in n.images_dir().glob(f"ch{chapter:03d}_s{num:02d}_*.png")
                  if f"s{num:02d}b_" not in p.name and f"s{num:02d}c_" not in p.name]
        clips = [p for p in n.images_dir().glob(f"ch{chapter:03d}_*an{num:02d}.mp4")]
        if not clips:
            clips = [p for p in n.images_dir().glob(f"ch{chapter:03d}_echo{num:02d}.mp4")]
        if not clips:
            clips = [p for p in n.images_dir().glob(f"ch{chapter:03d}_h3{num:02d}.mp4")]
        if not clips:
            clips = [p for p in n.images_dir().glob(f"ch{chapter:03d}_ltx{num:02d}.mp4")]
        voice = n.voice_dir() / f"ch{chapter:03d}_s{num:02d}.wav"
        echofail = (n.images_dir() / f"ch{chapter:03d}_echofail{num:02d}.flag").exists()
        line = s.get("narration") or s.get("dialogue") or ""
        out.append({"scene": num,
                    "has_still": bool(stills), "has_clip": bool(clips),
                    "has_voice": voice.exists(), "no_face": echofail,
                    "grip": scene_grip(str(line), num == 1),
                    "shot": s.get("shot_size", ""), "mood": s.get("mood", ""),
                    "characters": s.get("characters", []),
                    "line": str(line)[:120]})
    return {"chapter": chapter, "scenes": out}


# ---------------------------------------------------------------------------
# API — Workbench (Toonflow-style: plan → storyboard → production → output)
# ---------------------------------------------------------------------------

def _live_render():
    return bool(tasks.active()) or ep.render_alive()


def _canonical_videos(novel, chapter):
    out = []
    for p in sorted(novel.videos_dir().glob(f"ch{chapter:03d}_*.mp4"),
                    key=lambda q: q.stat().st_mtime):
        if ".candidate." in p.name or re.search(r"\.v\d+\.mp4$", p.name):
            continue
        out.append(p.name)
    return out


SCENE_EDITABLE = {"narration", "dialogue", "image_prompt", "shot_size",
                  "camera_move", "mood", "angle", "animation_hint",
                  "animation_style", "camera_lens", "camera_focal", "camera_aperture"}


class SceneUpdateReq(BaseModel):
    chapter: int
    scene: int
    fields: dict = {}
    novel: str = DEFAULT_NOVEL_ID


@app.get("/api/workbench")
async def api_workbench(chapter: int = 1, novel: str = DEFAULT_NOVEL_ID):
    n = _novel_or_404(novel)
    sf = n.scenes_dir() / f"ch{chapter:03d}.json"
    if not sf.exists():
        raise HTTPException(404, f"No script for chapter {chapter}")
    try:
        scenes = json.loads(sf.read_text(encoding="utf-8"))
        if isinstance(scenes, dict):
            scenes = scenes.get("scenes", [])
    except Exception as e:
        raise HTTPException(400, f"Failed to parse script: {e}")
    try:
        status = (await api_scenes_status(chapter, novel=novel))["scenes"]
    except Exception:
        status = []
    stat = {s["scene"]: s for s in status if isinstance(s, dict)}
    pins = _read_pins(n).get(f"ch{chapter:03d}", {})
    cards = []
    for i, s in enumerate(scenes):
        if not isinstance(s, dict):
            continue
        num = s.get("scene_number", i + 1)
        st = stat.get(num, {})
        stills = [p for p in n.images_dir().glob(f"ch{chapter:03d}_s{num:02d}_*.png")
                  if f"s{num:02d}b_" not in p.name and f"s{num:02d}c_" not in p.name]
        # filenames end in uuid4().hex, so sorted()[-1] is a RANDOM still.
        # Newest by mtime, same rule as the producer's _newest().
        newest = max(stills, key=lambda p: (p.stat().st_mtime, p.name)) if stills else None
        cards.append({"scene": num,
                      "narration": s.get("narration", ""), "dialogue": s.get("dialogue", ""),
                      "image_prompt": s.get("image_prompt", ""),
                      "shot_size": s.get("shot_size", ""), "camera_move": s.get("camera_move", ""),
                      "mood": s.get("mood", ""), "angle": s.get("angle", ""),
                      "animation_hint": s.get("animation_hint", ""),
                      "animation_style": s.get("animation_style", ""),
                      "camera_lens": s.get("camera_lens", ""),
                      "camera_focal": s.get("camera_focal", ""),
                      "camera_aperture": s.get("camera_aperture", ""),
                      "characters": s.get("characters", []),
                      "still": newest.name if newest else None,
                      "has_clip": bool(st.get("has_clip")), "has_voice": bool(st.get("has_voice")),
                      "no_face": bool(st.get("no_face")), "pinned": str(num) in pins})
    try:
        with open(ROOT / "quality_log.json", encoding="utf-8") as f:
            hist = json.load(f)
        quality = next((e for e in reversed(hist)
                        if isinstance(e, dict) and e.get("chapter") == chapter
                        and e.get("novel", DEFAULT_NOVEL_ID) == n.id), {})
    except Exception:
        quality = {}
    return {"chapter": chapter, "novel": novel, "scenes": cards,
            "pins": pins, "videos": _canonical_videos(n, chapter),
            "quality": quality, "live": _live_render(),
            "styles": sorted(ep.ANIMATION_STYLES)}


@app.post("/api/scene/update")
async def api_scene_update(req: SceneUpdateReq):
    if not isinstance(req.fields, dict) or not req.fields:
        raise HTTPException(400, "fields must be a non-empty object")
    bad = set(req.fields) - SCENE_EDITABLE
    if bad:
        raise HTTPException(400, f"Not editable: {sorted(bad)}")
    if _live_render():
        raise HTTPException(409, "A render is live — editing is disabled until it finishes")
    sf = _novel_or_404(getattr(req, "novel", None) or DEFAULT_NOVEL_ID).scenes_dir() / f"ch{req.chapter:03d}.json"
    if not sf.exists():
        raise HTTPException(404, f"No script for chapter {req.chapter}")
    try:
        scenes = json.loads(sf.read_text(encoding="utf-8"))
        if isinstance(scenes, dict):
            scenes = scenes.get("scenes", [])
    except Exception as e:
        raise HTTPException(400, f"Failed to parse script: {e}")
    if not isinstance(scenes, list):
        raise HTTPException(400, "Script root must be a scene list")
    target = next((s for s in scenes
                   if isinstance(s, dict) and s.get("scene_number") == req.scene), None)
    if target is None:
        raise HTTPException(404, f"Scene {req.scene} not found")
    for k, v in req.fields.items():
        if not isinstance(v, str) or len(v) > 4000:
            raise HTTPException(400, f"Bad value for {k}: must be a short string")
        if k == "animation_style" and v and v not in ep.ANIMATION_STYLES:
            raise HTTPException(400, f"Unknown animation_style: {v}")
        target[k] = v.strip()
    # ponytail: unique temp name — a fixed ".tmp" collides with a concurrent
    # writer, and os.replace() overwrites by definition so no unlink-retry
    # is needed (and none should exist: it would delete the script).
    tmp = sf.with_name(f"{sf.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(scenes, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, sf)
    except Exception as e:
        raise HTTPException(500, f"Could not save script: {e}")
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return {"updated": True, "chapter": req.chapter, "scene": req.scene}


# ---------------------------------------------------------------------------
# API — Self-improvement review (ledger + proposals)
# ---------------------------------------------------------------------------

def _load_json(name, default):
    try:
        p = ROOT / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return default
    except Exception:
        return default


@app.get("/api/tuning")
async def api_tuning():
    ledger = _load_json("tuning_ledger.json", [])
    props = _load_json("tuning_proposals.json", [])
    hist = _load_json("quality_log.json", [])
    trend = [{"chapter": e.get("chapter"), "score": e.get("score"),
              "image": e.get("avg_image"), "audio": e.get("avg_audio"),
              "at": e.get("timestamp")} for e in hist[-12:] if isinstance(e, dict)]
    pend_ledger = [e for e in ledger if isinstance(e, dict) and e.get("outcome") == "proposed"][-10:]
    recent = [e for e in ledger if isinstance(e, dict)][-10:]
    pend_props = [p for p in props if isinstance(p, dict) and p.get("status") == "pending"]
    return {"ledger_proposed": pend_ledger, "ledger_recent": recent,
            "proposals": pend_props, "trend": trend}


class TuningLedgerReq(BaseModel):
    render_idx: int
    param: str
    action: str = "accept"


@app.post("/api/tuning/ledger")
async def api_tuning_ledger(req: TuningLedgerReq):
    """Human gate for proposed GPU-spend knobs. Only face_budget is executable:
    writes face_check_max_reframes (bounded 2..6) into config + live CONFIG."""
    if req.param != "face_budget" or req.action != "accept":
        raise HTTPException(400, "Only face_budget proposals can be accepted here")
    ledger = _load_json("tuning_ledger.json", [])
    target = next((e for e in ledger
                   if isinstance(e, dict) and e.get("param") == "face_budget"
                   and e.get("render_idx") == req.render_idx
                   and e.get("outcome") == "proposed"), None)
    if not target:
        raise HTTPException(404, "Proposal not found or already handled")
    new = max(2, min(6, int(target.get("new", 4))))
    cfg_path = ep.CONFIG_FILE
    try:
        disk = _read_disk_config()
        disk["face_check_max_reframes"] = new
        ep._atomic_write_json(cfg_path, disk, indent=2)
    except Exception as e:
        raise HTTPException(500, f"Could not save config: {e}")
    try:
        ep.CONFIG["face_check_max_reframes"] = new
    except Exception:
        pass
    target["outcome"] = "accepted"
    try:
        ep._atomic_write_json(ROOT / "tuning_ledger.json", ledger[-200:], indent=2)
    except Exception:
        pass
    return {"accepted": {"face_check_max_reframes": new}}


class TuningProposalReq(BaseModel):
    key: str
    action: str = "dismiss"


@app.post("/api/tuning/proposal")
async def api_tuning_proposal(req: TuningProposalReq):
    """Acknowledge a text proposal (accept = 'I did it', dismiss = 'not now').
    Neither auto-changes the pipeline — that is the point."""
    if req.action not in ("accept", "dismiss"):
        raise HTTPException(400, "action must be accept or dismiss")
    props = _load_json("tuning_proposals.json", [])
    target = next((p for p in props
                   if isinstance(p, dict) and p.get("key") == req.key
                   and p.get("status") == "pending"), None)
    if not target:
        raise HTTPException(404, "Proposal not found or already handled")
    target["status"] = "accepted" if req.action == "accept" else "dismissed"
    try:
        ep._atomic_write_json(ROOT / "tuning_proposals.json", props[-200:], indent=2)
    except Exception as e:
        raise HTTPException(500, f"Could not save: {e}")
    return {"updated": {req.key: target["status"]}}
# ---------------------------------------------------------------------------
# Web UI — v3 only (NiceGUI, mounted in __main__). "/" redirects there.
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return RedirectResponse("/v3/", status_code=307)


if __name__ == "__main__":
    try:
        from nicegui_dashboard import mount_v3
        mount_v3(app)
        print("  [v3] combined dashboard at http://127.0.0.1:8000/v3/")
    except Exception as e:
        print(f"  [WARN] /v3 dashboard unavailable ({e})")
    uvicorn.run(app, host="127.0.0.1", port=8000)
