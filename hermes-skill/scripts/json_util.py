"""JSON helpers shared by the pipeline: tolerant parsing, ONE atomic writer.

Trailing-comma tolerant JSON parsing exists for LLM responses. Some models
(notably Gemini-flash-lite via OpenAI-compatible endpoints, and occasionally
Ollama) emit a trailing comma before a closing `}`/`]`. Standard json.loads
rejects it; this strips only those commas locally before retrying, avoiding
a wasted LLM call.

The atomic writer lives here because the pipeline writes durable state to a
dozen-plus JSON files at the repo root from three different processes (the
render CLI, the dashboard, the supervisor): issues.json, progress_live.json,
quality_log.json, cost_log.json, perf_log.json, analytics_learning_log.json,
rights_manifest.json, tuning_ledger.json, tuning_proposals.json, tokens.json,
config.json, pinned.json. Seven of those writers were truncate-then-write, so
a crash mid-dump destroyed the whole file and the next reader silently
returned an empty list — the history was gone and nothing said so. One
implementation, one guarantee, imported everywhere.
"""
import json
import os
import tempfile
import threading
import time
from pathlib import Path

# Windows: os.replace() is MoveFileExW(MOVEFILE_REPLACE_EXISTING), and
# "existing opens of the destination path are not allowed, even if they share
# delete access... Most Windows programs, including Python, do not share
# delete access on open files." (cpython#90161). So a reader holding the file
# open makes replace() raise PermissionError even though we only need the
# directory entry to swap. A bounded retry IS the accepted workaround; what is
# NOT acceptable is exiting that retry loop without raising, which silently
# discards the write and leaves the old content in place.
_REPLACE_ATTEMPTS = 6
_REPLACE_BACKOFF_S = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)

# Threads within one process must serialise too. A byte-range file lock is the
# right tool ACROSS processes, but it is a poor fit for same-process threads:
# on Windows every contender must retry, and measured with 4 concurrent writers
# 93 of 120 writes exhausted all six retries (they still raised, so nothing was
# lost, but that is a lot of wasted backoff for a function called on every
# progress tick). One lock per destination path, held for the swap only.
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(path: Path) -> threading.Lock:
    key = str(path).lower()          # Windows paths are case-insensitive
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[key] = lock
        return lock


def strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before a closing brace/bracket, outside strings.

    The naive regex version also matched a `,}` living INSIDE a JSON string
    literal and silently rewrote the string's value — a data-corruption
    path in a module whose whole job is safe parsing. This walks the text
    tracking string/escape state instead.
    """
    out = []
    in_str = False
    escaped = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            # look ahead past whitespace for a closer
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j] in "}]":
                i += 1  # drop the comma, keep the whitespace/closer
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def dump_json_atomic(path, obj, *, retries=_REPLACE_ATTEMPTS, **dump_kw):
    """Write JSON so readers see the old file or the new one, never a torn one.

    Fails LOUD on purpose. The previous episode_producer helper retried
    `tmp.replace(path)` three times and then fell out of the loop normally, so
    a locked destination produced no exception, no log line, and no update —
    the caller believed the write happened. Verified on this box: with any
    concurrent reader of progress_live.json (the dashboard and supervisor both
    poll it), every progress write vanished silently.

    fsync before the rename, not just close(): close() flushes Python's buffer
    to the OS, but on a crash or power loss the rename can land before the
    data does, leaving a correctly-named file full of zeroes. fsync orders the
    data before the metadata.

    Returns the Path written. Raises OSError if the swap never succeeded.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, **dump_kw)
    tmp = None
    try:
        fd, name = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".",
                                    suffix=".tmp")
        tmp = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        last = None
        # Serialise same-process writers; the retry below is for OTHER
        # processes (and for editors/previewers holding the file).
        with _thread_lock_for(p):
            for attempt in range(retries):
                try:
                    os.replace(tmp, p)
                    return p
                except PermissionError as e:
                    # Another process has the destination open without
                    # FILE_SHARE_DELETE. Transient; back off and retry.
                    last = e
                    if attempt < retries - 1:
                        time.sleep(_REPLACE_BACKOFF_S[min(
                            attempt, len(_REPLACE_BACKOFF_S) - 1)])
        raise OSError(
            f"could not replace {p}: {last}. Something holds the file open "
            f"without FILE_SHARE_DELETE (a reader that does not close it, or a "
            f"file left open in an editor/preview). The old contents are "
            f"intact; the new data was NOT written.") from last
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def load_json_file(path, default=None):
    """Read JSON, returning `default` on any read/parse failure.

    Returns the default instead of None so callers can pass [] or {} and get a
    usable value; the caller still decides whether a missing file is benign.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


class FileLock:
    """Cross-process exclusive lock on a sidecar file. Context manager.

    stdlib only: fcntl.flock on POSIX, msvcrt.locking on Windows — the same
    primitive this repo already uses for the render lock. Deliberately NOT
    portalocker: that library wraps these two calls and nothing else, so it
    would be a dependency to avoid eight lines.

    `blocking=False` raises BlockingIOError instead of waiting, which is what
    a render-lock check wants ("is someone else holding it?"). With
    blocking=True a waiter retries for `timeout` seconds and then raises
    TimeoutError, because an unbounded wait inside a request handler is a
    denial of service on yourself.
    """

    def __init__(self, path, blocking=True, timeout=10.0, poll=0.05):
        self.path = Path(path)
        self.blocking = blocking
        self.timeout = float(timeout)
        self.poll = float(poll)
        self._fh = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._fh.seek(0)
                    # LK_NBLCK raises OSError if the byte range is already
                    # locked — documented behaviour, so a translate is exact.
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fh.fileno(),
                                fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as e:
                if not self.blocking:
                    self._fh.close()
                    self._fh = None
                    raise BlockingIOError(
                        f"{self.path} is locked by another process") from e
                if time.monotonic() >= deadline:
                    self._fh.close()
                    self._fh = None
                    raise TimeoutError(
                        f"timed out after {self.timeout:g}s waiting for "
                        f"{self.path}") from e
                time.sleep(self.poll)

    def release(self):
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


def loads_json(text: str) -> object:
    """Parse JSON, tolerating trailing commas (invalid but common)."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        cleaned = strip_trailing_commas(text)
        if cleaned == text:
            raise
        return json.loads(cleaned)


def oai_chat(base_url, messages, timeout=120, max_tokens=300, temperature=0.1):
    """Chat completion against an OpenAI-compatible server (llama-server).
    Returns the assistant text; raises on HTTP/parse failure."""
    import urllib.request
    payload = {"messages": messages, "temperature": float(temperature),
               "max_tokens": int(max_tokens), "stream": False}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    # Context manager: this is a long-lived dashboard process, and a leaked
    # socket per call is a real leak, not a CPython-refcounting detail.
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "replace")
    # ponytail: name the failure. An unguarded ["choices"][0] chain turned a
    # proxy HTML error page or a truncated body into a bare KeyError with no
    # hint about which URL produced it.
    try:
        data = json.loads(body)
        return data["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"oai_chat: malformed reply from {base_url} ({e}); "
            f"first 200 chars: {body[:200]!r}") from e


def oai_image_part(b64_png: str) -> dict:
    return {"type": "image_url",
            "image_url": {"url": "data:image/png;base64," + b64_png}}


def pid_alive(pid) -> bool:
    """Cross-platform PID check. NOTE: os.kill(pid, 0) always raises
    WinError 87 on Windows (signal 0 doesn't exist there), so POSIX-only
    code reports every process dead. Use this everywhere instead."""
    try:
        pid = int(pid)
    except (ValueError, TypeError):
        return False
    # ponytail: reject non-positive pids BEFORE signalling. os.kill(0, 0)
    # signals the caller's entire process group and os.kill(-1, 0) signals
    # every process the user owns — both return 0 and both would read as
    # "alive". A truncated or half-written PID file must never make the
    # supervisor think a render is running. (The Windows branch below already
    # returned False for these; this makes the POSIX branch agree.)
    if pid <= 0:
        return False
    if os.name == "nt":
        # ponytail: ctypes OpenProcess beats tasklist — no subprocess,
        # no 15s hang when tasklist is wedged (that hang made supervisor
        # report render dead mid-run and restart services into the GPU).
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.OpenProcess.restype = ctypes.c_void_p
            k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            k32.GetExitCodeProcess.restype = ctypes.c_int
            k32.CloseHandle.argtypes = [ctypes.c_void_p]
            h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            try:
                code = ctypes.c_ulong()
                return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259
            finally:
                k32.CloseHandle(h)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        # It exists, we just may not signal it. Reading this as "dead" is the
        # exact bug this function exists to prevent.
        return True
    except (ProcessLookupError, OSError):
        return False
