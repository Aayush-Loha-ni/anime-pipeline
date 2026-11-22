# RULE: Ops — Machine Safety (16GB Laptop, No Excuses)

## VRAM order (hard ceiling 16GB — Qwen14B 9GB + Qwen Image 2.1 GGUF + TTS
## can NEVER co reside)

1. Phase 1 (script): the LLM server owns VRAM → `unload_llm()` (scoped process stop).
2. Phase 2–3 (stills/motion): ComfyUI owns VRAM → `free_comfyui_vram()`
   before LLM work, `ensure_comfyui()` before Comfy work.
3. Phase 5 (voice): TTS owns VRAM, Comfy evicted.
4. Never load two engines "just to check" mid render. Supervisor
   diagnosis fires ONLY when no render is alive.

## Heat (laptop chassis, sustained 85 min phases)

  `gpu_heat_guard()`: ≥85°C pauses to 78°C, sleep block released while
  cooling. Normal render band is 46–68°C. AC power + open lid, always.
  `TIME_BUDGET_S = 5400` (~90 min) per episode — over budget switches
   H3 final tier to fast tier single pass, never to silent failure. Observed full
   runs take ~3h (H3+echo); budget gates the final tier path only.

## Services & power (learned the hard way)

  Watchdog guards ONE TTS (:5003 CosyVoice), dashboard :8000,
  Comfy :8188, LLM :8080. All four UP before any render.
  `watchdog.bat` now CYCLES every `WD_INTERVAL` s (default 60, min 5)
  instead of running once and exiting (2026-09-29) — a bare run spawns a
  minimised `--loop` copy and returns, so `watchdog.bat` then
  `Open Dashboard.bat` still works. Previously a service that died 2 min
  after the check stayed dead until someone re-ran the file by hand:
  53 such rows in issues.json. Singleton: one loop only, enforced by the
  `watchdog.lock/` lock dir, because two loops would race the `:8080`
  relaunch and kill/relaunch the LLM in a loop. Stop it with
  `taskkill /FI "WINDOWTITLE eq anime-watchdog" /F`.
  TTS/Comfy die silently on restarts — always health check, never assume.
  :5001 (Chatterbox) and :5002 (F5) are DELETED with their servers,
  venvs, refs and clone branches (2026 09 17).
  Comfy Desktop wrapper can wedge at git bootstrap: backend runs fine
  launched directly (Installs `.venv` python + instance yaml + Shared
  in/out dirs). Verify `:8188`, not the Desktop window.
  **Windows Update reboots kill multi hour jobs** (LoRA died at 69%).
  Pause updates before long runs. Keep awake blocks sleep, NOT reboots.
  Launcher shells intermittently report `ChildProcess.kill` — children
  usually survive. Poll with short (<60s) commands, never long sleeps.
  One GPU job at a time: `render.lock` + `render.pid`. Stale locks from
  dead PIDs auto reap; live locks abort new runs. `stop.requested` stops
  cleanly at scene boundaries — never taskkill mid sampling.
  `preserve=True` renders (Continue / redo scene) skip the Phase 0 purge
  and reuse good caches — a 1 scene fix never costs a 3.5h re render.
  Fresh CLI runs always purge.
  Server tasks report honestly: `produce_episode → None` (gate/preflight/
  mux failure) surfaces as `{"ok": False}`, never "completed".
  `/api/preflight` says ready ONLY when preflight actually passes.
  Dashboard phase chips cover the real phases (talking/cooling/characters/
  error included); supervisor skips stall flags on done/error.
