#!/usr/bin/env python3
"""Preflight — health checks and render readiness gate.

Run without GPU to diagnose services, models, rights, and director package.
Blocks render when any critical issue is unresolved.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def _http_get(url, timeout=5):
    """Quick HTTP GET, returns (status_code, body_str) or (0, error_str)."""
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pipeline-preflight/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return 0, str(e)
    except Exception as e:
        return 0, str(e)


def check_services(config):
    """Check LLM, ComfyUI, TTS health. Returns dict of service -> {ok, detail}."""
    services = {}

    # LLM (llama-server)
    llm_url = config.get("llm_url", "http://127.0.0.1:8080")
    code, body = _http_get(f"{llm_url}/v1/models", timeout=5)
    if code == 200:
        try:
            data = json.loads(body)
            models = [m.get("id", "") for m in data.get("data", [])]
            services["llm"] = {"ok": True, "detail": f"models: {', '.join(models)}"}
        except json.JSONDecodeError:
            services["llm"] = {"ok": True, "detail": "responding (unparseable models)"}
    else:
        services["llm"] = {"ok": False, "detail": f"HTTP {code}: {body[:200]}"}

    # ComfyUI
    comfy_url = config.get("comfyui_url", "http://localhost:8188")
    code, body = _http_get(f"{comfy_url}/system_stats", timeout=5)
    if code == 200:
        try:
            data = json.loads(body)
            vram = data.get("devices", [{}])[0].get("vram_total", 0)
            services["comfyui"] = {"ok": True, "detail": f"VRAM: {vram // (1024**2)}MB"}
        except (json.JSONDecodeError, IndexError):
            services["comfyui"] = {"ok": True, "detail": "responding"}
    else:
        services["comfyui"] = {"ok": False, "detail": f"HTTP {code}: {body[:200]}"}

    # TTS (Qwen3-TTS)
    tts_url = config.get("tts_url", "http://localhost:5003")
    code, body = _http_get(f"{tts_url}/health", timeout=5)
    if code == 200:
        try:
            data = json.loads(body)
            loaded = data.get("loaded", False)
            services["tts"] = {"ok": True, "detail": f"loaded: {loaded}"}
        except json.JSONDecodeError:
            services["tts"] = {"ok": True, "detail": "responding"}
    else:
        services["tts"] = {"ok": False, "detail": f"HTTP {code}: {body[:200]}"}

    return services


def check_models(config):
    """Verify configured model files exist locally. Returns dict of model -> {ok, detail}."""
    models = {}
    # ponytail: search the real ComfyUI model roots. config's comfyui_input is
    # the staging dir, not the weights dir — searching it always said "missing".
    _roots = [Path(os.path.expanduser("~")) / "AppData" / "Local" / "Comfy-Desktop" / "ComfyUI-Shared" / "models",
              Path(os.path.expanduser("~")) / "AppData" / "Local" / "Comfy-Desktop" / "ComfyUI-Installs" / "Example Novel" / "ComfyUI" / "models"]
    checks = [
        ("checkpoint", config.get("checkpoint", "")),
        ("qwen_text_encoder", config.get("qwen_text_encoder", "")),
        ("qwen_vae", config.get("qwen_vae", "")),
        ("h3_checkpoint", config.get("h3_checkpoint", "")),
        ("h3_text_encoder", config.get("h3_text_encoder", "")),
        ("h3_vae", config.get("h3_vae", "")),
        ("h3_audio_vae", config.get("h3_audio_vae", "")),
        # ponytail: the 3-deep pdd/skin/camera LoRA chain was replaced by a
        # single turbo LoRA loaded model-only (config.yaml: it patched the 32B
        # encoder and shipped clips that ignored the still). Checking the dead
        # keys reported "not configured" forever, so this module said BLOCKED
        # on a healthy machine while episode_producer.preflight passed. Mirrors
        # episode_producer._model_roots.
        ("h3_turbo_lora", config.get("h3_turbo_lora", "")),
        ("llm_gguf", config.get("llm_gguf", "")),
        ("llm_vlm_gguf", config.get("llm_vlm_gguf", "")),
        ("llm_vlm_mmproj", config.get("llm_vlm_mmproj", "")),
        ("llm_exe", config.get("llm_exe", "")),
    ]
    for name, filename in checks:
        if not filename:
            models[name] = {"ok": False, "detail": "not configured"}
            continue
        # Check if the file exists as a direct path or in ComfyUI model roots
        p = Path(filename)
        if p.exists():
            size_mb = p.stat().st_size // (1024 * 1024)
            models[name] = {"ok": True, "detail": f"{size_mb}MB"}
            continue
        found = None
        for root in _roots:
            if not root.exists():
                continue
            hits = list(root.rglob(f"*{Path(filename).name}*"))
            if hits:
                found = hits[0]
                break
        if found:
            size_mb = found.stat().st_size // (1024 * 1024)
            models[name] = {"ok": True, "detail": f"{size_mb}MB ({found.parent.name}/)"}
        else:
            models[name] = {"ok": False, "detail": f"not found: {filename}"}
    return models


def check_disk_ram():
    """Check free disk space and system RAM. Returns dict."""
    info = {}
    # Disk
    try:
        import platform
        disk_path = "C:\\" if platform.system() == "Windows" else "/"
        usage = shutil.disk_usage(disk_path)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        info["disk"] = {"ok": free_gb > 10, "detail": f"{free_gb:.1f}GB free / {total_gb:.1f}GB total"}
    except Exception as e:
        info["disk"] = {"ok": False, "detail": str(e)}

    # RAM
    # ponytail: fail CLOSED, matching the disk branch 20 lines above and
    # episode_producer.preflight's explicit rule ("not knowing the free space is
    # not the same as having space"). This used to report ok:True with
    # "unable to determine", so a broken RAM probe counted as a PASS.
    #
    # The wmic fallback it degraded into was dead three ways: `"/ value"` has a
    # space inside the switch, the parser expects "Key=Value" while `wmic get`
    # emits a whitespace-aligned table with no '=', and wmic is removed from
    # modern Windows (verified absent on this machine). Deleted rather than
    # fixed — psutil is the only supported path.
    try:
        import psutil
        mem = psutil.virtual_memory()
        free_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        info["ram"] = {"ok": free_gb > 4, "detail": f"{free_gb:.1f}GB available / {total_gb:.1f}GB total"}
    except ImportError:
        info["ram"] = {"ok": False,
                       "detail": "unknown — install psutil to check RAM"}
    except Exception as e:
        info["ram"] = {"ok": False, "detail": f"unknown ({e})"}

    # GPU (basic check)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 5:
                name = parts[0]
                vram_total = int(parts[1])
                vram_free = int(parts[2])
                temp = int(parts[3])
                power = parts[4]
                info["gpu"] = {
                    "ok": True,
                    "detail": f"{name} | {vram_free}MB/{vram_total}MB free | {temp}C | {power}W"
                }
            else:
                info["gpu"] = {"ok": True, "detail": result.stdout.strip()[:200]}
        else:
            info["gpu"] = {"ok": False, "detail": "nvidia-smi not available or failed"}
    except FileNotFoundError:
        info["gpu"] = {"ok": False, "detail": "nvidia-smi not found"}
    except Exception as e:
        info["gpu"] = {"ok": False, "detail": str(e)}

    return info


def check_rights(root):
    """Check rights manifest status. Returns dict."""
    from pipeline.rights_manifest import load_rights_manifest, preflight_rights_gate
    root = Path(root)
    manifest = load_rights_manifest(root)
    if not manifest.get("entries") and not manifest.get("novel_adaptation_rights", {}).get("owner_confirmation"):
        return {"ok": False, "detail": "rights manifest not initialized or empty"}
    ok, blockers, warnings = preflight_rights_gate(manifest)
    return {
        "ok": ok,
        "detail": f"{len(blockers)} blockers, {len(warnings)} warnings",
        "blockers": blockers,
        "warnings": warnings,
    }


def check_director_package(root, chapter_num):
    """Check director package exists and is approved. Returns dict."""
    from pipeline.director_package import load_director_package
    root = Path(root)
    pkg = load_director_package(root / "scenes", chapter_num)
    if pkg is None:
        return {"ok": False, "detail": f"ch{chapter_num:03d}_director.json not found"}
    if pkg.get("version") == "placeholder":
        return {"ok": False, "detail": "director package is a placeholder — generate real content"}
    approved = pkg.get("owner_approved", False)
    return {
        "ok": approved,
        "detail": f"version {pkg.get('version', '?')}, approved: {approved}"
    }


def check_issues(root):
    """Check issues.json for unresolved criticals. Returns dict."""
    root = Path(root)
    issues_path = root / "issues.json"
    if not issues_path.exists():
        return {"ok": True, "detail": "no issues.json found"}
    try:
        issues = json.loads(issues_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        # Fail CLOSED. This used to return ok=True, so a torn issues.json
        # disabled the very gate it feeds. ep.preflight now reports it too.
        return {"ok": False, "detail": f"issues.json unreadable ({e}) — fix or delete it"}
    # ponytail: also fail CLOSED on the wrong SHAPE. Iterating a dict yields
    # its KEYS, `isinstance(i, dict)` filtered every one of them out, so
    # open_crits came back empty and this returned ok=True — one stray '{' at
    # the top of issues.json silently disarmed the render-blocking gate, eight
    # lines below the comment saying it must fail closed.
    if not isinstance(issues, list):
        return {"ok": False,
                "detail": (f"issues.json is a {type(issues).__name__}, not a list "
                           f"— fix its shape or delete it")}
    open_crits = [
        i for i in issues
        if isinstance(i, dict) and i.get("severity") == "critical" and not i.get("resolved")
    ]
    # Stale threshold: entries older than 7 days are considered stale.
    # ponytail: reuse supervisor.parse_ts so an ISO-8601 "T" row (written
    # by other tooling) is aged correctly instead of always counting as now.
    # Guarded like ep.preflight: an unimportable supervisor must degrade to
    # "no timestamp known", never kill the whole report.
    try:
        from supervisor import parse_ts
    except Exception:
        parse_ts = None
    stale_threshold = time.time() - 7 * 86400
    stale_crits = []
    current_crits = []
    for c in open_crits:
        entry_epoch = parse_ts(c.get("ts")) if parse_ts else None
        if entry_epoch is not None and entry_epoch < stale_threshold:
            stale_crits.append(c)
        else:
            current_crits.append(c)
    return {
        "ok": len(current_crits) == 0,
        "detail": f"{len(current_crits)} current criticals, {len(stale_crits)} stale",
        "current_criticals": current_crits,
        "stale_criticals": stale_crits,
    }


def _check_ok(result):
    """True unless the result reports a failure.

    services/models/system return {name: {ok, detail}} — a NESTED shape with
    no top-level "ok". The old `result.get("ok", True)` therefore returned
    True unconditionally for exactly the three checks that matter most
    (ComfyUI down, models missing, disk full all reported PASS). Handles
    both the flat and nested shapes; mirrors format_preflight_report.
    """
    if not isinstance(result, dict):
        return True
    # Do NOT return early on a truthy top-level "ok": a check that grows a
    # summary verdict later would then hide a nested failure.
    if result.get("ok") is False:
        return False
    for value in result.values():
        if isinstance(value, dict) and value.get("ok") is False:
            return False
    return True


def run_preflight(root, chapter_num, config=None):
    """Run full preflight check. Returns structured report dict."""
    root = Path(root)
    if config is None:
        config_path = root / "hermes-skill" / "scripts" / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            config = {}

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chapter": chapter_num,
        "checks": {},
        "ok": True,
    }

    # Run all checks
    report["checks"]["services"] = check_services(config)
    report["checks"]["models"] = check_models(config)
    report["checks"]["system"] = check_disk_ram()
    report["checks"]["rights"] = check_rights(root)
    report["checks"]["director_package"] = check_director_package(root, chapter_num)
    report["checks"]["issues"] = check_issues(root)

    # Determine overall status
    for name, result in report["checks"].items():
        if not _check_ok(result):
            report["ok"] = False
            report.setdefault("failed_checks", []).append(name)

    return report


def format_preflight_report(report):
    """Format preflight report as human-readable text."""
    lines = [
        f"=== PREFLIGHT REPORT — Chapter {report['chapter']} ===",
        f"Timestamp: {report['timestamp']}",
        f"Overall: {'PASS' if report['ok'] else 'BLOCKED'}",
        "",
    ]

    for name, result in report["checks"].items():
        lines.append(f"--- {name.upper()} ---")
        if isinstance(result, dict):
            if "ok" in result:
                status = "OK" if result["ok"] else "FAIL"
                lines.append(f"  [{status}] {result.get('detail', '')}")
            for k, v in result.items():
                if k in ("ok", "detail"):
                    continue
                if isinstance(v, list) and v:
                    for item in v:
                        if isinstance(item, dict):
                            lines.append(f"  - {item.get('symptom', item.get('message', str(item)))}")
                        else:
                            lines.append(f"  - {item}")
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, dict):
                            status = "OK" if vv.get("ok") else "FAIL"
                            lines.append(f"  [{status}] {kk}: {vv.get('detail', '')}")
                        else:
                            lines.append(f"  {kk}: {vv}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    # ponytail: allow `python pipeline/preflight.py` from scripts/ — without
    # this, `from pipeline...` imports fail and only -m works.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    root = Path(__file__).parent.parent.parent.parent
    chapter = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    report = run_preflight(root, chapter)
    print(format_preflight_report(report))
    sys.exit(0 if report["ok"] else 1)
