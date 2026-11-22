#!/usr/bin/env python3
"""Studio dashboard (v3) — one-page front end for the render pipeline.

Pure HTTP client of the pipeline_server REST API: no server imports, so this
module imports anywhere with zero side effects. Mounted by pipeline_server
via mount_v3(app) -> ui.run_with(app, mount_path='/v3').
"""
import httpx

BASE = "http://127.0.0.1:8000"



def _http_error(status, data):
    if isinstance(data, dict):
        detail = data.get("detail") or data.get("reason") or data.get("error") or data
    else:
        detail = data
    return f"{status}: {detail}"[:200]


async def _api_get(client, base, path, params=None, timeout=15):
    try:
        r = await client.get(base + path, params=params, timeout=timeout)
        data = r.json()
        if r.status_code >= 400:
            return {"_error": _http_error(r.status_code, data)}
        return data if isinstance(data, dict) else {"_error": f"bad response: {data!r}"[:200]}
    except Exception as e:
        return {"_error": str(e)}


async def _api_post(client, base, path, payload=None, timeout=60):
    try:
        r = await client.post(base + path, json=payload or {}, timeout=timeout)
        data = r.json()
        if r.status_code >= 400:
            return {"_error": _http_error(r.status_code, data)}
        return data if isinstance(data, dict) else {"_error": f"bad response: {data!r}"[:200]}
    except Exception as e:
        return {"_error": str(e)}


def _steps_text(phases):
    return " → ".join(
        f"{'✔' if p.get('state') == 'done' else '▶' if p.get('state') == 'active' else '·'} {p.get('name', '?')}"
        for p in phases or [] if isinstance(p, dict))




def mount_v3(app, base=BASE, mount_path="/v3"):
    """Combined one-page dashboard."""
    from nicegui import ui  # lazy: server stays importable without nicegui

    @ui.page("/")
    def index():
        els = {}
        state = {"reloading": False}  # guard: chapter set_options/set_value refires on_change

        def _novel_id():
            return els["novel"].value or "sinners_world"

        def _video_url(name):
            """Absolute URL for the media file. The bare path is rewritten by the
            NiceGUI mount (/v3) and 404s; the full URL resolves as-is."""
            nid = _novel_id()
            if nid == "sinners_world":
                return f"{BASE}/videos/{name}"
            return f"{BASE}/novels/{nid}/videos/{name}"
        with ui.row().classes("items-center gap-4"):
            ui.label("Studio").classes("text-2xl font-bold")
            ui.label("everything on one page").classes("text-sm")
        els["msg"] = ui.label("Pick a chapter and press Start.").classes("text-sm")
        with ui.row().classes("gap-6"):
            els["svc"] = {k: ui.label(f"{k} …") for k in ("llm", "comfyui", "tts")}
            els["gpu"] = ui.label("GPU …")
            els["tokens"] = ui.label("tokens …")

        with ui.card().classes("w-full"):
            ui.label("1 · Make an episode").classes("font-bold")
            with ui.row().classes("items-center gap-2"):
                els["novel"] = ui.select({}, value=None, label="Novel",
                                         on_change=lambda: _on_novel_change()).classes("w-48")
                els["chapter"] = ui.select({}, value=None, label="Chapter",
                                           on_change=lambda: _reload()).classes("w-32")
                els["b_start"] = ui.button("Start", on_click=lambda: _act("start")).props("color=primary")
                els["b_cont"] = ui.button("Continue", on_click=lambda: _act("continue")).props("color=secondary")
                els["b_stop"] = ui.button("Stop", on_click=lambda: _ask_stop()).props("color=negative")
                ui.button("Start all services", on_click=lambda: _act("services")).props("size=sm")
                ui.button("Check readiness", on_click=lambda: _act("preflight")).props("size=sm")
            with ui.row().classes("items-center gap-2"):
                ui.button("New novel", on_click=lambda: new_dlg.open()).props("size=sm")
                ui.label("Each novel is its own folder (chapters, videos, cast).").classes("text-xs")

        with ui.dialog() as new_dlg, ui.card():
            ui.label("New novel").classes("font-bold")
            els["new_title"] = ui.input("Title", placeholder="My Novel")
            with ui.row().classes("gap-2"):
                ui.button("Create", on_click=lambda: _new_novel_create()).props("color=primary")
                ui.button("Cancel", on_click=new_dlg.close)

        with ui.dialog() as dlg, ui.card():
            ui.label("Stop the current render?")
            with ui.row():
                ui.button("Yes, stop", on_click=lambda: _do_stop()).props("color=negative")
                ui.button("Keep going", on_click=dlg.close)

        with ui.card().classes("w-full"):
            ui.label("2 · Progress").classes("font-bold")
            els["phase"] = ui.label("idle").classes("text-lg")
            els["bar"] = ui.linear_progress(value=0).classes("w-full")
            els["detail"] = ui.label("").classes("text-sm")
            els["steps"] = ui.label("").classes("text-xs")
            els["log"] = ui.markdown("")

        with ui.card().classes("w-full"):
            ui.label("3 · Is it ready?").classes("font-bold")
            els["ready"] = ui.label("press “Check readiness”")
            els["hints"] = ui.markdown("")
            els["appr"] = {}
            for item in ("rights", "director"):
                with ui.row().classes("items-center gap-2"):
                    lab = ui.label(f"{item}: unknown").classes("text-sm")
                    btn = ui.button(f"Approve {item}",
                                    on_click=lambda i=item: _act(f"approve:{i}")).props("size=sm")
                    els["appr"][item] = (lab, btn)
            ui.button("Approve everything", on_click=lambda: _act("approve-all")).props("size=sm")

        with ui.card().classes("w-full"):
            ui.label("4 · Scenes (what will render)").classes("font-bold")
            with ui.row().classes("items-center gap-2"):
                els["scene"] = ui.number(label="Scene", value=1, min=1, step=1).classes("w-28")
                els["b_redo"] = ui.button("Redo this scene", on_click=lambda: _act("redo")).props("size=sm")
                ui.button("Reload scenes", on_click=lambda: _reload()).props("size=sm")
            els["stable"] = ui.table(columns=[
                {"name": "scene", "label": "Scene", "field": "scene"},
                {"name": "grip", "label": "Hook", "field": "grip"},
                {"name": "shot", "label": "Shot", "field": "shot"},
                {"name": "still", "label": "Still", "field": "still"},
                {"name": "clip", "label": "Clip", "field": "clip"},
                {"name": "voice", "label": "Voice", "field": "voice"},
                {"name": "line", "label": "Line", "field": "line"},
            ], rows=[]).classes("w-full")

        with ui.card().classes("w-full"):
            ui.label("5 · Watch").classes("font-bold")
            els["videos"] = ui.select({}, value=None, label="Episode file").classes("w-80")
            els["player"] = ui.video("")

        with ui.card().classes("w-full"):
            ui.label("6 · Problems & what to do next").classes("font-bold")
            with ui.row().classes("gap-2"):
                ui.button("Refresh problems", on_click=lambda: _act("problems")).props("size=sm")
            els["errors"] = ui.markdown("_none_")

        async def _say(text):
            els["msg"].set_text(text)
            ui.notify(text)

        async def _act(kind):
            ch = els["chapter"].value or 1
            nid = _novel_id()
            async with httpx.AsyncClient() as client:
                if kind == "start":
                    res = await _api_post(client, base, "/api/start",
                                          {"chapter": ch, "novel": nid})
                    await _say("Started." if res.get("started")
                               else f"Not started: {res.get('reason', res)}")
                elif kind == "continue":
                    res = await _api_post(client, base, "/api/continue")
                    await _say("Continuing." if res.get("started")
                               else f"Not continued: {res.get('reason', res)}")
                elif kind == "services":
                    res = await _api_post(client, base, "/api/services/start", timeout=180)
                    await _say("Services starting — badges turn green as they come up."
                               if res.get("started") else f"{res.get('reason', res)}")
                elif kind == "preflight":
                    await _say("Checking readiness…")
                    res = await _api_get(client, base, "/api/preflight",
                                         {"chapter": ch, "novel": nid}, timeout=180)
                    if "_error" in res:
                        await _say(f"Check failed: {res['_error']}")
                    else:
                        _show_preflight(res)
                        await _say("Ready to render." if res.get("ready")
                                   else "Not ready — see section 3 below.")
                elif kind.startswith("approve:"):
                    res = await _api_post(client, base, "/api/approve",
                                          {"item": kind.split(":")[1], "chapter": ch,
                                           "novel": nid})
                    await _say(str(res.get("ok", res)))
                    await _reload_preflight(client, ch, nid)
                elif kind == "approve-all":
                    res = await _api_post(client, base, "/api/approve-all",
                                          {"chapters": [ch], "novel": nid})
                    await _say("Approved." if res.get("ok") else str(res))
                    await _reload_preflight(client, ch, nid)
                elif kind == "redo":
                    sc = int(els["scene"].value or 1)
                    res = await _api_post(client, base, "/api/redo",
                                          {"chapter": ch, "scene": sc, "novel": nid})
                    await _say(f"Redoing scene {sc}." if res.get("started")
                               else f"Not started: {res.get('reason', res)}")
                elif kind == "problems":
                    await _load_problems(client)
                    await _say("Problems refreshed.")
            await refresh()

        async def _ask_stop():
            dlg.open()

        async def _do_stop():
            dlg.close()
            async with httpx.AsyncClient() as client:
                res = await _api_post(client, base, "/api/stop")
                await _say("Stop requested." if res.get("stopped") else f"{res.get('reason', res)}")
            await refresh()

        async def _reload():
            if state["reloading"]:
                return
            state["reloading"] = True
            try:
                await _load_scenes()
                await refresh()
                # The chapter selector used to reload scenes and progress only,
                # so section 3 kept painting the PREVIOUS chapter's verdict:
                # switch off a blocked ch4 and ch5 shows "Blocked - fix the X
                # rows" while /api/preflight answers ready=true. _show_preflight
                # ran only from the Check-readiness button and the approve
                # paths. Re-check here, where the chapter actually changes.
                # Deliberately not in refresh(): that runs on a 5s timer and
                # preflight can take up to 180s.
                ch = els["chapter"].value
                if ch is not None:
                    async with httpx.AsyncClient() as client:
                        await _reload_preflight(client, ch, _novel_id())
            finally:
                state["reloading"] = False

        async def _on_novel_change():
            """Switching novel re-reads that novel's chapter list."""
            if state["reloading"]:
                return
            state["reloading"] = True
            try:
                els["chapter"].set_value(None)
                await refresh()
            finally:
                state["reloading"] = False
            await _load_scenes()
            ch = els["chapter"].value
            if ch is not None:
                async with httpx.AsyncClient() as client:
                    await _reload_preflight(client, ch, _novel_id())

        async def _new_novel_create():
            title = (els["new_title"].value or "").strip()
            if not title:
                await _say("Give the novel a title first.")
                return
            async with httpx.AsyncClient() as client:
                res = await _api_post(client, base, "/api/novels", {"title": title})
            if "_error" in res:
                await _say(f"Could not create the novel: {res['_error']}")
                return
            new_dlg.close()
            els["new_title"].set_value("")
            nid = (res.get("novel") or {}).get("id")
            await _say(f"Novel '{title}' created.")
            if nid:
                els["novel"].set_value(nid)
            await refresh()

        async def _reload_preflight(client, ch, nid=None):
            res = await _api_get(client, base, "/api/preflight",
                                 {"chapter": ch, "novel": nid or _novel_id()}, timeout=180)
            if "_error" not in res:
                _show_preflight(res)

        def _show_preflight(pf):
            els["ready"].set_text("Ready ✔ — press Start" if pf.get("ready") else "Blocked — fix the ✘ rows")
            els["ready"].style(f"color:{'#22c55e' if pf.get('ready') else '#ef4444'}")
            hints = pf.get("hints") or []
            lines = [f"- {h}" for h in hints] or ["- No blockers found."]
            els["hints"].set_content("\n".join(lines))
            for a in pf.get("approvals") or []:
                if not isinstance(a, dict) or a.get("item") not in els["appr"]:
                    continue
                lab, _ = els["appr"][a["item"]]
                mark = "✔" if a.get("ok") else "✘"
                extra = "; ".join(a.get("blockers") or []) or a.get("detail", "")
                lab.set_text(f"{mark} {a['item']}: {extra}")
                lab.style(f"color:{'#22c55e' if a.get('ok') else '#ef4444'}")

        async def _load_scenes():
            ch = els["chapter"].value or 1
            async with httpx.AsyncClient() as client:
                res = await _api_get(client, base, "/api/scenes/status",
                                     {"chapter": ch, "novel": _novel_id()})
            rows = []
            for s in res.get("scenes") or []:
                if not isinstance(s, dict):
                    continue
                tick = lambda v: "✔" if v else "✘"  # noqa: E731
                rows.append({"scene": s.get("scene"), "grip": s.get("grip", ""),
                             "shot": s.get("shot", ""),
                             "still": tick(s.get("has_still")), "clip": tick(s.get("has_clip")),
                             "voice": tick(s.get("has_voice")),
                             "line": (s.get("line") or "")[:80]})
            els["stable"].rows = rows
            els["stable"].update()
            try:
                els["scene"].max = max(1, len(rows))
            except Exception:
                pass
            if "_error" in res:
                els["msg"].set_text(f"Scenes: {res['_error']}")
            elif not rows:
                extra = f" ({res.get('detail')})" if res.get("detail") else ""
                els["msg"].set_text(f"Scenes: no script for chapter {ch} yet"
                                     f" — render once to generate it{extra}")

        async def _load_problems(client):
            st = await _api_get(client, base, "/api/status")
            if "_error" in st:
                lines = [f"- status feed offline: `{st['_error']}`"]
            else:
                lines = [f"- `{str(e).replace('`', chr(39))}`" for e in (st.get("errors") or [])]
                lines += [f"- crash: `{str(e).replace('`', chr(39))}`" for e in (st.get("crash") or [])[-4:]]
            iss = await _api_get(client, base, "/api/issues")
            if "_error" in iss:
                lines.append(f"- issues feed: `{iss['_error']}` (6/min limit — wait, then retry)")
            else:
                for i in (iss.get("issues") or []):
                    if isinstance(i, dict) and i.get("severity") == "critical" and not i.get("resolved"):
                        lines.append(f"- BLOCKED: {i.get('symptom')} → {i.get('action')}")
            aud = await _api_get(client, base, "/api/audit")
            if "_error" in aud:
                lines.append(f"- audit feed: `{aud['_error']}`")
            else:
                for f in (aud.get("findings") or [])[:8]:
                    if isinstance(f, dict) and f.get("level") in ("error", "warn"):
                        lines.append(f"- next: {f.get('message')} → {f.get('action')}")
            els["errors"].set_content("\n".join(lines) or "_none_")

        async def refresh():
            ch = els["chapter"].value
            nid = _novel_id()
            async with httpx.AsyncClient() as client:
                st = await _api_get(client, base, "/api/status", {"novel": nid})
                chs = await _api_get(client, base, "/api/chapters", {"novel": nid})
                tok = await _api_get(client, base, "/api/tokens")
                nvs = await _api_get(client, base, "/api/novels")
            if "_error" in st:
                els["msg"].set_text(f"Dashboard API offline ({st['_error']}) — is the server running?")
                return
            try:
                running = bool((st.get("run") or {}).get("running"))
                if running:
                    els["b_start"].disable()
                    els["b_cont"].disable()
                    els["b_stop"].enable()
                    els["b_redo"].disable()
                else:
                    els["b_start"].enable()
                    els["b_cont"].enable()
                    els["b_stop"].disable()
                    els["b_redo"].enable()
                prog = st.get("progress") or {}
                els["phase"].set_text(f"{prog.get('phase', 'idle')} — {prog.get('percent', 0)}%")
                els["bar"].set_value(float(prog.get("percent", 0) or 0) / 100)
                els["detail"].set_text(f"{prog.get('detail', '')}  {('ETA ' + prog['eta']) if prog.get('eta') else ''}")
                els["steps"].set_text(_steps_text(st.get("phases")))
                log = (prog.get("log") or [])[-12:]
                els["log"].set_content("```\n" + "\n".join(
                    str(x)[-160:].replace("```", "'''") for x in log) + "\n```" if log else "")
                for k, badge in els["svc"].items():
                    ok = (st.get("services") or {}).get("services", {}).get(k)
                    badge.set_text(f"{k} {'●' if ok else '○'}")
                    badge.style(f"color:{'#22c55e' if ok else '#ef4444'}")
                gpu = st.get("gpu") or {}
                els["gpu"].set_text(f"GPU {gpu.get('temp')}°C" if gpu.get("temp") is not None else "GPU n/a")
                if tok.get("enabled"):
                    els["tokens"].set_text(f"tokens: {tok.get('balance')} (render −{tok.get('render_cost')})")
                else:
                    els["tokens"].set_text("tokens: off")
                novels = nvs.get("novels") or []
                opts = {n.get("id"): (n.get("title") or n.get("id"))
                        for n in novels if isinstance(n, dict) and n.get("id")}
                if opts:
                    try:
                        cur = list(els["novel"].options or [])
                    except Exception:
                        cur = []
                    if sorted(map(str, opts)) != sorted(map(str, cur)):
                        els["novel"].set_options(opts)
                    if els["novel"].value is None or els["novel"].value not in opts:
                        els["novel"].set_value(nid if nid in opts else next(iter(opts)))
                names = [c.get("chapter") for c in (chs.get("chapters") or [])
                         if isinstance(c, dict) and isinstance(c.get("chapter"), int)] or [1]
                try:
                    cur_opts = list(els["chapter"].options or [])
                except Exception:
                    cur_opts = []
                if sorted(map(str, names)) != sorted(map(str, cur_opts)):
                    els["chapter"].set_options(names)
                if ch not in names:
                    els["chapter"].set_value(names[0])
                vids = st.get("videos") or []
                els["videos"].set_options(vids)
                if vids and els["videos"].value not in vids:
                    els["videos"].set_value(vids[-1])
                    els["player"].set_source(_video_url(vids[-1]))
                elif not vids and els["videos"].value:
                    els["videos"].set_value(None)
                    els["player"].set_source("")
                errs = (st.get("errors") or []) + (st.get("crash") or [])[-4:]
                if errs:
                    els["errors"].set_content("\n".join(
                        f"- `{str(e).replace(chr(96), chr(39))}`" for e in errs))
            except Exception as e:
                els["msg"].set_text(f"Dashboard hiccup ({e}) — retrying…")

        async def _on_video_change():
            v = els["videos"].value
            if v:
                els["player"].set_source(_video_url(v))

        els["videos"].on_value_change(lambda _: _on_video_change())
        ui.timer(5.0, refresh)
        ui.timer(0.5, refresh, once=True)
        ui.timer(1.0, _reload, once=True)

    ui.run_with(app, mount_path=mount_path, title="Studio", dark=True)


if __name__ == "__main__":
    # Standalone smoke-run: python nicegui_dashboard.py (API expected on :8000).
    import uvicorn
    from fastapi import FastAPI
    _app = FastAPI()
    mount_v3(_app)
    uvicorn.run(_app, host="127.0.0.1", port=8001)
