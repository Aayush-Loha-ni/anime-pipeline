"""Async Task Manager — structured task tracking with status/progress/cancel/retry.

Replaces the simple threading + _run_state dict with a proper task queue.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Callable, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskLog:
    timestamp: float
    level: str
    message: str

    def to_dict(self) -> dict:
        return {"ts": self.timestamp, "level": self.level, "msg": self.message}


@dataclass
class Task:
    id: str
    kind: str  # "render", "redo", "review", "storyboard"
    chapter: int
    novel: str = "sinners_world"
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    phase: str = ""
    detail: str = ""
    scene: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    logs: list[TaskLog] = field(default_factory=list)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = {f.name: deepcopy(getattr(self, f.name)) for f in fields(self)
             if not f.name.startswith("_") and f.name != "logs"}
        d["status"] = self.status.value
        d["logs"] = [l.to_dict() for l in self.logs[-50:]]
        return d

    def log(self, level: str, msg: str):
        self.logs.append(TaskLog(time.time(), level, msg))
        del self.logs[:-100]

    def cancel(self):
        if self.finished_at is None and not self.is_cancelled:
            self._cancel_event.set()
            self.phase = "stopping"
            self.detail = "Stop requested; waiting for worker to exit"
            self.log("INFO", "Cancel requested")

    @property
    def is_active(self) -> bool:
        return self.finished_at is None or bool(self._thread and self._thread.is_alive())

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at


class TaskManager:
    """Manages concurrent pipeline tasks with lifecycle tracking."""

    def __init__(self, max_concurrent: int = 1, max_history: int = 100):
        if max_concurrent < 1 or max_history < 1:
            raise ValueError("Task limits must be positive")
        self.max_concurrent = max_concurrent
        self.max_history = max_history
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._history: list[dict] = []

    def submit(self, kind: str, chapter: int, fn: Callable[[Task], Any],
               scene: int | None = None, novel: str | None = None) -> Task:
        """Submit a new task. Returns the Task object."""
        task = Task(
            id=str(uuid.uuid4())[:8],
            kind=kind,
            chapter=chapter,
            scene=scene,
            novel=novel or "sinners_world",
        )
        task.log("INFO", f"Task created: {kind} ch{chapter:03d}" +
                 (f"/s{scene:02d}" if scene else ""))

        with self._lock:
            self._prune()
            running = sum(t.is_active for t in self._tasks.values())
            if running >= self.max_concurrent:
                raise RuntimeError(f"Max concurrent tasks ({self.max_concurrent}) reached")
            self._tasks[task.id] = task

        def _run():
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            task.log("INFO", "Started")
            try:
                if not task.is_cancelled:
                    result = fn(task)
                    task.result = result if isinstance(result, dict) else {"ok": True}
                    if task.result.get("ok") is False:
                        task.error = str(task.result.get("reason") or
                                         task.result.get("error") or "Task returned ok:false")
                        task.log("ERROR", task.error)
            except Exception as e:
                task.error = traceback.format_exc()
                task.log("ERROR", str(e))
            finally:
                with self._lock:
                    if task.is_cancelled:
                        task.status = TaskStatus.CANCELLED
                        task.phase = "cancelled"
                        task.detail = "Worker exited after stop request"
                        task.log("INFO", "Cancelled")
                    elif task.error is not None:
                        task.status = TaskStatus.FAILED
                    else:
                        task.status = TaskStatus.COMPLETED
                        task.progress = 100.0
                        task.log("INFO", "Completed")
                    task.finished_at = time.time()
                    self._history.append(task.to_dict())
                    del self._history[:-self.max_history]
                    self._prune()

        t = threading.Thread(target=_run, daemon=True)
        task._thread = t
        try:
            t.start()
        except Exception:
            with self._lock:
                del self._tasks[task.id]
            raise
        return task

    def submit_if_idle(self, kind: str, chapter: int, fn: Callable[[Task], Any],
                       scene: int | None = None,
                       novel: str | None = None) -> tuple[Task | None, str]:
        """Atomic check-and-submit. Returns (task, "") on success, (None, reason) on reject."""
        with self._lock:
            self._prune()
            running = sum(t.is_active for t in self._tasks.values())
            if running >= self.max_concurrent:
                return None, f"Max concurrent tasks ({self.max_concurrent}) reached"
        try:
            t = self.submit(kind, chapter, fn, scene=scene, novel=novel)
            return t, ""
        except (RuntimeError, OSError) as e:
            return None, str(e)

    def _prune(self):
        finished = [t for t in self._tasks.values() if not t.is_active]
        finished.sort(key=lambda t: t.finished_at)
        for task in finished[:-self.max_history]:
            del self._tasks[task.id]

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            self._prune()
            return self._tasks.get(task_id)

    def active(self) -> list[Task]:
        with self._lock:
            self._prune()
            return [t for t in self._tasks.values() if t.is_active]

    def active_for_chapter(self, chapter: int, novel: str | None = None) -> Task | None:
        for t in self.active():
            if t.chapter == chapter and (novel is None or t.novel == novel):
                return t
        return None

    def history(self, limit: int = 20) -> list[dict]:
        with self._lock:
            self._prune()
            return deepcopy(self._history[-limit:]) if limit > 0 else []

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.finished_at is None:
                task.cancel()
                return True
            return False

    def cancel_all(self) -> bool:
        with self._lock:
            active = [t for t in self._tasks.values() if t.finished_at is None]
            for task in active:
                task.cancel()
            return bool(active)

    def uncancel(self, task_id: str | None = None) -> None:
        """Roll back a cancel whose STOP_FILE write failed — the worker
        never saw the stop, so leaving the event set would mark a
        successful run CANCELLED / stick phase at 'stopping'."""
        with self._lock:
            targets = ([self._tasks[task_id]] if task_id else
                       [t for t in self._tasks.values() if t.finished_at is None])
            for task in targets:
                if task.finished_at is None and task._cancel_event.is_set():
                    task._cancel_event.clear()
                    if task.phase == "stopping":
                        task.phase = ""
                        task.detail = ""
