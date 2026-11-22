"""File-based token bucket — simplified port of AI-Anime-Maker SessionState.

Single-machine studio: one shared bucket in <project-root>/tokens.json.
Disabled by default ([Token] enabled=false → every spend succeeds).
When enabled: /api/start|continue spend render_cost, /api/redo spends
redo_cost; uploads via /api/tokens/grant earn new_image tokens.

# ponytail: single global bucket, per-user buckets if multi-user matters
"""
from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from pathlib import Path

# json_util is stdlib-only (no pipeline.* imports), so this cannot cycle.
from json_util import FileLock, dump_json_atomic

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


# Tests point the bucket at a temp file by assigning this; None = the real one.
_FILE_OVERRIDE = None


def _file() -> Path:
    # ponytail: the try/except could never fire — Path(...) and .parent do not
    # raise — and the fallback returned a CWD-relative path, which is a WORSE
    # outcome than the primary (it would silently write a second, unrelated
    # tokens.json). A str is accepted so a caller can override it with one.
    if _FILE_OVERRIDE:
        return Path(_FILE_OVERRIDE)
    return Path(__file__).parent.parent.parent.parent / "tokens.json"


def _load() -> dict:
    try:
        p = _file()
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception as e:
        logger.warning("Token load failed: %s", e)
    return {"balance": 0}


def _save(data: dict) -> None:
    """Persist the balance. RAISES if the write did not land.

    ponytail: this used to catch the failure and only log it, so spend()
    returned (True, decremented_balance) while the file on disk still held the
    OLD balance — a token charged and never recorded. Verified: spend(4)
    returned (True, 6) with 10 still on disk. Silent loss is worse than a
    failed request, because the caller has already been told it succeeded.
    """
    p = _file()
    dump_json_atomic(p, data, indent=2)


def enabled() -> bool:
    from pipeline import config as _cfg
    return _cfg.tokens_enabled()


def balance() -> int:
    try:
        return int(_load().get("balance", 0))
    except Exception:
        return 0


def _process_lock():
    """Cross-process advisory lock over tokens.json.

    ponytail: _LOCK is a threading.Lock, so it protects nothing between the
    dashboard, the render CLI and the supervisor — three separate processes
    that all read-modify-write this file. Two of them could read balance=1,
    both pass the `bal < n` check, both write, and one spend vanishes.

    An exclusive byte-range lock on a sidecar file is the stdlib answer on both
    platforms (msvcrt.locking on Windows, fcntl.flock on POSIX) and needs no
    new dependency. msvcrt is the same primitive this repo already uses for the
    render lock in episode_producer._render_ownership.

    with_name, not with_suffix: with_suffix REPLACES the last suffix, so a
    config-ish name like "tokens.prod.json" would become "tokens.prod.lock"
    only by luck, and a dotted name with no extension loses it. Appending is
    unambiguous. Path() first because _file() may hand back a str.
    """
    return FileLock(Path(_file()).with_name(Path(_file()).name + ".lock"))


@contextmanager
def _locked_update():
    """Hold the process lock across a load/modify/save cycle."""
    with _LOCK:            # threads within this process
        with _process_lock():   # other processes
            yield


def grant(n: int, reason: str = "") -> int:
    """Add n tokens. Returns new balance."""
    with _locked_update():
        data = _load()
        data["balance"] = max(0, int(data.get("balance", 0)) + max(0, int(n)))
        if reason:
            data["last_grant_reason"] = str(reason)[:200]
        _save(data)
        return data["balance"]


def spend(n: int) -> tuple[bool, int]:
    """Spend n tokens. No-op success when disabled. Returns (ok, balance).

    Raises if the balance could not be persisted — see _save. The caller must
    NOT treat a failed write as a successful charge.
    """
    from pipeline import config as _cfg
    if not _cfg.tokens_enabled():
        return True, balance()
    n = max(0, int(n))
    with _locked_update():
        data = _load()
        bal = int(data.get("balance", 0))
        if bal < n:
            return False, bal
        data["balance"] = bal - n
        _save(data)
        return True, data["balance"]


def status() -> dict:
    from pipeline import config as _cfg
    return {"enabled": _cfg.tokens_enabled(), "balance": balance(),
            "render_cost": _cfg.token_cost_render(),
            "redo_cost": _cfg.token_cost_redo()}
