#!/usr/bin/env python3
"""Opt-in auto-publish: push a finished release package to YouTube, Instagram
Reels and TikTok.

OFF by default. The render path only calls this when config.json carries
`"publish": {"enabled": true, ...}` (see `publish_package`).

Deliberate choices:

- **httpx only.** It is already a pipeline dependency. google-api-python-client
  would be a ~40MB pin to wrap calls that are each ~5 lines of HTTP here.
- **Credentials live outside this repo** (default `~/.anime_pipeline_publish.json`).
  A git checkout is not a secret store.
- **Privacy defaults to the most restrictive setting on every platform.** An
  unattended pipeline must not be able to publish a public video without an
  explicit `"public"` in config.
- **Synthetic-media disclosure is not automated.** YouTube's altered/synthetic
  content declaration is a Studio UI toggle with no Data API v3 field, and this
  repo's rule is to never assert a disclosure the pipeline cannot prove. The
  upload sets `selfDeclaredMadeForKids: false` and logs a reminder; the owner
  still ticks the disclosure in Studio.
- **Instagram has no upload-bytes call.** The Graph API fetches the video from
  a PUBLIC URL, so `publish.instagram.video_url` must point somewhere Instagram
  can reach (e.g. a `cloudflared tunnel --url` quick tunnel over the package
  directory). Without it that platform reports the reason instead of failing
  the run.
- **TikTok public posting needs an audited app.** An unaudited app can only
  post `SELF_ONLY`; the API returns an error rather than a silent downgrade.

CLI:
    python -m pipeline.publisher auth youtube|instagram|tiktok
    python -m pipeline.publisher publish <package_dir> [--platforms yt,ig,tt]
"""

import base64
import hashlib
import re
import secrets
import time
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from json_util import dump_json_atomic, load_json_file

DEFAULT_CREDS_PATH = Path.home() / ".anime_pipeline_publish.json"
GRAPH = "v21.0"
CHUNK = 8 << 20  # YouTube resumable chunk: 8 MiB

# unified privacy -> (YouTube privacyStatus, TikTok privacy_level)
PRIVACY = {
    "private": ("private", "SELF_ONLY"),
    "unlisted": ("unlisted", "MUTUAL_FOLLOW_FRIENDS"),
    "public": ("public", "PUBLIC_TO_EVERYONE"),
}


class PublishError(RuntimeError):
    """Any failure that should be reported per-platform, never crash a render."""


# --- credentials ------------------------------------------------------

def load_creds(path=None):
    data = load_json_file(path or DEFAULT_CREDS_PATH, {})
    return data if isinstance(data, dict) else {}


def save_creds(creds, path=None):
    p = Path(path or DEFAULT_CREDS_PATH)
    dump_json_atomic(p, creds, indent=2)
    try:
        p.chmod(0o600)  # no-op-ish on Windows, real on POSIX
    except OSError:
        pass
    return p


def _expired(entry, skew=300):
    exp = entry.get("expires_at")
    return exp is None or time.time() + skew >= float(exp)


def _check(r, what):
    if r.status_code >= 400:
        raise PublishError(f"{what}: HTTP {r.status_code} {r.text[:300]}")
    return r


def _youtube_token(creds, path):
    y = creds.get("youtube") or {}
    if not y.get("refresh_token"):
        raise PublishError("youtube: no refresh_token in credentials — run "
                           "`python -m pipeline.publisher auth youtube`")
    if y.get("access_token") and not _expired(y):
        return y["access_token"]
    r = _check(httpx.post("https://oauth2.googleapis.com/token", data={
        "client_id": y["client_id"], "client_secret": y["client_secret"],
        "refresh_token": y["refresh_token"], "grant_type": "refresh_token"}),
        "youtube token refresh")
    d = r.json()
    y["access_token"] = d["access_token"]
    y["expires_at"] = time.time() + int(d.get("expires_in", 3600))
    creds["youtube"] = y
    save_creds(creds, path)
    return y["access_token"]


def _ig_token(creds, path):
    ig = creds.get("instagram") or {}
    tok = ig.get("access_token")
    if not tok:
        raise PublishError("instagram: no access_token — run "
                           "`python -m pipeline.publisher auth instagram`")
    if not _expired(ig):
        return tok
    r = _check(httpx.get(f"https://graph.facebook.com/{GRAPH}/oauth/access_token",
                         params={"grant_type": "fb_exchange_token",
                                 "client_id": ig["client_id"],
                                 "client_secret": ig["client_secret"],
                                 "fb_exchange_token": tok}),
               "instagram token refresh")
    d = r.json()
    ig["access_token"] = d["access_token"]
    ig["expires_at"] = time.time() + int(d.get("expires_in", 5183944))
    creds["instagram"] = ig
    save_creds(creds, path)
    return ig["access_token"]


def _tiktok_token(creds, path):
    tt = creds.get("tiktok") or {}
    if not tt.get("access_token"):
        raise PublishError("tiktok: no access_token — run "
                           "`python -m pipeline.publisher auth tiktok`")
    if not _expired(tt):
        return tt["access_token"]
    r = _check(httpx.post("https://open.tiktokapis.com/v2/oauth/token/", data={
        "client_key": tt["client_key"], "client_secret": tt["client_secret"],
        "grant_type": "refresh_token", "refresh_token": tt["refresh_token"]}),
        "tiktok token refresh")
    d = r.json()
    # TikTok ROTATES the refresh token on every refresh; persisting the new one
    # is mandatory or the next refresh re-authenticates by browser.
    tt["access_token"] = d["access_token"]
    tt["refresh_token"] = d.get("refresh_token") or tt.get("refresh_token")
    tt["expires_at"] = time.time() + int(d.get("expires_in", 86400))
    creds["tiktok"] = tt
    save_creds(creds, path)
    return tt["access_token"]


# --- uploaders --------------------------------------------------------

def _yt_resumable_put(client, session_url, video, total, log=print):
    """Chunked PUT. Restarts a failed chunk from the chunk boundary, not the
    whole file — an episode is 100-500 MB on this pipeline."""
    offset = 0
    while offset < total:
        end = min(offset + CHUNK, total) - 1
        with open(video, "rb") as f:
            f.seek(offset)
            blob = f.read(end - offset + 1)
        headers = {"Content-Type": "video/mp4",
                   "Content-Length": str(len(blob)),
                   "Content-Range": f"bytes {offset}-{end}/{total}"}
        for attempt in range(3):
            try:
                r = client.put(session_url, content=blob, headers=headers)
            except httpx.HTTPError as e:
                if attempt == 2:
                    raise PublishError(f"youtube upload chunk {offset}-{end}: {e}") from e
                time.sleep(2 << attempt)
                continue
            if r.status_code in (200, 201):
                return r.json()
            if r.status_code == 308:  # chunk accepted, more to send
                break
            if r.status_code in (408, 429) or r.status_code >= 500:
                time.sleep(2 << attempt)
                continue
            raise PublishError(f"youtube upload rejected: HTTP {r.status_code} "
                               f"{r.text[:200]}")
        else:
            raise PublishError(f"youtube upload chunk {offset}-{end} kept failing")
        offset = end + 1
    raise PublishError("youtube upload: session closed before the video was complete")


def upload_youtube(video_path, title, description, tags=(), privacy="private",
                   category_id="24", creds=None, creds_path=None, log=print):
    """Resumable upload via Data API v3. Returns {"id": ..., "url": ...}."""
    video = Path(video_path)
    size = video.stat().st_size
    tok = _youtube_token(creds or {}, creds_path)
    yt_privacy = PRIVACY.get(privacy, PRIVACY["private"])[0]
    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": list(tags)[:15],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": yt_privacy,
            "selfDeclaredMadeForKids": False,
        },
    }
    with httpx.Client(timeout=httpx.Timeout(10.0, read=600.0)) as c:
        r = c.post("https://www.googleapis.com/upload/youtube/v3/videos",
                   params={"uploadType": "resumable", "part": "snippet,status"},
                   json=body,
                   headers={"Authorization": f"Bearer {tok}",
                            "X-Upload-Content-Type": "video/mp4",
                            "X-Upload-Content-Length": str(size)})
        _check(r, "youtube session")
        session = r.headers.get("location")
        if not session:
            raise PublishError(f"youtube session: no upload URL in {dict(r.headers)}")
        log(f"  YouTube: uploading {size / 1e6:.0f} MB ({yt_privacy}) ...")
        data = _yt_resumable_put(c, session, video, size, log=log)
    vid = data.get("id")
    log("  YouTube: uploaded — remember the synthetic/altered-content "
        "disclosure toggle in Studio (no API field for it)")
    return {"id": vid, "url": f"https://youtu.be/{vid}" if vid else None}


def upload_instagram(video_path, video_url, caption, privacy="private",
                     creds=None, creds_path=None, log=print):
    """Two-step Reels publish. The video is FETCHED by Meta from `video_url` —
    there is no bytes-upload call in the Graph API."""
    if not video_url:
        raise PublishError("instagram: needs a public video_url — set "
                           "publish.instagram.video_url (a cloudflared quick "
                           "tunnel over the package dir works)")
    creds = creds or {}
    tok = _ig_token(creds, creds_path)
    user_id = (creds.get("instagram") or {}).get("ig_user_id")
    if not user_id:
        raise PublishError("instagram: no ig_user_id in credentials")
    share_feed = "true" if privacy != "private" else "false"
    with httpx.Client(timeout=httpx.Timeout(10.0, read=300.0)) as c:
        base = f"https://graph.facebook.com/{GRAPH}/{user_id}"
        r = _check(c.post(f"{base}/media", data={
            "media_type": "REELS", "video_url": video_url,
            "caption": caption[:2200], "share_to_feed": share_feed,
            "access_token": tok}), "instagram container")
        cid = (r.json() or {}).get("id")
        if not cid:
            raise PublishError(f"instagram container: no id in {r.text[:200]}")
        # Meta needs to fetch + transcode before media_publish is allowed.
        for _ in range(24):  # ~2 min
            st = c.get(f"https://graph.facebook.com/{GRAPH}/{cid}",
                       params={"fields": "status_code", "access_token": tok})
            code = (st.json() or {}).get("status_code") if st.status_code == 200 else None
            if code == "FINISHED":
                break
            if code == "ERROR":
                raise PublishError(f"instagram container failed to process "
                                   f"({st.text[:200]})")
            time.sleep(5)
        else:
            raise PublishError("instagram container never finished processing")
        r = _check(c.post(f"{base}/media_publish", data={
            "creation_id": cid, "access_token": tok}), "instagram publish")
        mid = (r.json() or {}).get("id")
    return {"id": mid, "url": f"https://www.instagram.com/reel/{mid}/" if mid else None}


def upload_tiktok(video_path, title, privacy="private", creds=None,
                  creds_path=None, log=print):
    """FILE_UPLOAD single-chunk publish. Returns {"publish_id": ...}."""
    video = Path(video_path)
    size = video.stat().st_size
    tok = _tiktok_token(creds or {}, creds_path)
    tt_privacy = PRIVACY.get(privacy, PRIVACY["private"])[1]
    with httpx.Client(timeout=httpx.Timeout(10.0, read=600.0)) as c:
        r = c.post("https://open.tiktokapis.com/v2/post/publish/video/init/",
                   headers={"Authorization": f"Bearer {tok}"},
                   json={"post_info": {"title": title[:2200],
                                       "privacy_level": tt_privacy,
                                       "disable_duet": False,
                                       "disable_comment": False,
                                       "disable_stitch": False},
                         "source_info": {"source": "FILE_UPLOAD",
                                         "video_size": size,
                                         "chunk_size": size,
                                         "total_chunk_count": 1}})
        _check(r, "tiktok init")
        err = (r.json() or {}).get("error") or {}
        if err.get("code") not in (None, "ok"):
            raise PublishError(f"tiktok init: {err.get('code')} "
                               f"{str(err.get('message'))[:200]}")
        d = (r.json() or {}).get("data") or {}
        pid, url = d.get("publish_id"), d.get("upload_url")
        if not pid or not url:
            raise PublishError(f"tiktok init: no publish_id/upload_url in {r.text[:200]}")
        log(f"  TikTok: uploading {size / 1e6:.0f} MB ({tt_privacy}) ...")
        with open(video, "rb") as f:
            r = c.put(url, content=f,
                      headers={"Content-Type": "video/mp4",
                               "Content-Length": str(size),
                               "Content-Range": f"bytes 0-{size - 1}/{size}"})
        _check(r, "tiktok upload")
        for _ in range(8):  # TikTok transcodes after upload; poll briefly
            time.sleep(5)
            st = c.post("https://open.tiktokapis.com/v2/post/publish/status/fetch/",
                        headers={"Authorization": f"Bearer {tok}"},
                        json={"publish_id": pid})
            data = (st.json() or {}).get("data") or {}
            if data.get("status") in ("PUBLISH_COMPLETE", "FAILED"):
                if data.get("status") == "FAILED":
                    raise PublishError(f"tiktok publish failed: "
                                       f"{data.get('fail_reason')}")
                log(f"  TikTok: published (post {data.get('publicaly_available_post_id')})")
                break
    return {"publish_id": pid}


# --- orchestrator -----------------------------------------------------

def _resolve_video(pkg, pkg_dir=None):
    """The mp4 to upload: pkg["video_path"] when it still exists, otherwise an
    mp4 inside the package directory.

    release_package stores video_path = the COPY inside the package when the
    source existed at build time, but ch001's package points at a source mp4
    that was moved away afterwards, so the recorded path 404s and the episode
    is unpublishable even though the file ships with the package.
    """
    p = Path(pkg.get("video_path") or "")
    if p.exists():
        return p
    directory = Path(pkg_dir) if pkg_dir else p.parent
    for cand in sorted(directory.glob("*.mp4")):
        if cand.exists():
            return cand
    return p


def publish_package(pkg, publish_cfg, pkg_dir=None, creds_path=None, log=print):
    """Publish one release package (dict as built by release_package).

    Returns {"results": {platform: {"ok": bool, ...}}}. Never raises: a
    network failure at 3am must not turn a finished episode into a crash.
    """
    cfg = publish_cfg or {}
    results = {}
    platforms = [str(p).lower() for p in (cfg.get("platforms") or [])]
    if not platforms:
        log("  [PUBLISH] no platforms configured — nothing to do")
        return {"results": results}

    report = pkg.get("release_report") or {}
    if report.get("verdict") == "NOT_RELEASE_READY":
        log("  [PUBLISH] BLOCKED — episode is NOT release-ready: "
            f"{(report.get('summary') or {}).get('blockers')} blockers")
        return {"blocked": "NOT_RELEASE_READY", "results": results}
    if (pkg.get("disclosure_checklist") or {}).get("music_license") == "unverified":
        log("  [PUBLISH] WARNING: music licence UNVERIFIED — the description "
            "tells viewers not to publish. Confirm rights_manifest.json first.")

    video = _resolve_video(pkg, pkg_dir)
    if not video.exists():
        for p in platforms:
            results[p] = {"ok": False, "error": f"video not found: {video}"}
        return {"results": results}

    meta = pkg.get("metadata") or {}
    title = (meta.get("title_options")
             or [f"Chapter {pkg.get('chapter')}"])[0]
    description = meta.get("description") or ""
    tags = [str(t) for t in (meta.get("tags") or [])]
    caption = (description + "\n\n"
               + " ".join("#" + re.sub(r"\W+", "", t) for t in tags)).strip()

    privacy = str(cfg.get("privacy") or "private").lower()
    if privacy not in PRIVACY:
        log(f"  [PUBLISH] unknown privacy {privacy!r} — using 'private'")
        privacy = "private"
    creds = load_creds(creds_path)
    ig_cfg = cfg.get("instagram") or {}

    for platform in platforms:
        try:
            if platform in ("yt", "youtube"):
                results[platform] = {"ok": True, "result": upload_youtube(
                    video, title, description, tags, privacy,
                    creds=creds, creds_path=creds_path, log=log)}
            elif platform in ("ig", "instagram"):
                results[platform] = {"ok": True, "result": upload_instagram(
                    video, ig_cfg.get("video_url"), caption, privacy,
                    creds=creds, creds_path=creds_path, log=log)}
            elif platform in ("tt", "tiktok"):
                results[platform] = {"ok": True, "result": upload_tiktok(
                    video, title, privacy, creds=creds,
                    creds_path=creds_path, log=log)}
            else:
                raise PublishError(f"unknown platform {platform!r}")
            log(f"  [PUBLISH] {platform}: {results[platform]['result']}")
        except Exception as e:  # per-platform failure never kills the render
            results[platform] = {"ok": False,
                                 "error": f"{type(e).__name__}: {e}"}
            log(f"  [WARN] Publish to {platform} failed: {e}")

    out_dir = Path(pkg_dir) if pkg_dir else video.parent
    try:
        dump_json_atomic(out_dir / "publish_result.json",
                         {"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "privacy": privacy, "results": results}, indent=2)
    except OSError as e:
        log(f"  [WARN] could not write publish_result.json: {e}")
    return {"results": results}


# --- one-time auth CLI ------------------------------------------------

def _paste_param(name, auth_url):
    print("\nOpen this URL in the browser, approve, then paste the full "
          "redirect URL:\n")
    print(auth_url, "\n")
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass
    pasted = input("Redirect URL: ").strip()
    q = parse_qs(urlparse(pasted).query)
    if name not in q:
        raise PublishError(f"redirect URL has no {name!r} parameter: {pasted[:120]}")
    return q[name][0]


def _ask(creds_section, key, prompt):
    if not creds_section.get(key):
        creds_section[key] = input(prompt).strip()
    return creds_section[key]


def auth_youtube(creds_path=None):
    c = load_creds(creds_path)
    y = c.get("youtube") or {}
    _ask(y, "client_id", "YouTube OAuth client_id (Google Cloud console): ")
    _ask(y, "client_secret", "YouTube OAuth client_secret: ")
    redirect = y.get("redirect_uri") or "http://localhost:8971/"
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
        "client_id": y["client_id"], "redirect_uri": redirect,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/youtube.upload",
        "access_type": "offline", "prompt": "consent"})
    code = _paste_param("code", url)
    r = _check(httpx.post("https://oauth2.googleapis.com/token", data={
        "code": code, "client_id": y["client_id"],
        "client_secret": y["client_secret"], "redirect_uri": redirect,
        "grant_type": "authorization_code"}), "youtube auth")
    d = r.json()
    if not d.get("refresh_token"):
        raise PublishError("no refresh_token returned — the OAuth client must "
                           "be a Desktop/Web app and the consent screen must "
                           "have been shown (access_type=offline, prompt=consent)")
    y.update(access_token=d["access_token"],
             refresh_token=d["refresh_token"],
             expires_at=time.time() + int(d.get("expires_in", 3600)),
             redirect_uri=redirect)
    c["youtube"] = y
    save_creds(c, creds_path)
    print(f"\nSaved YouTube credentials to "
          f"{creds_path or DEFAULT_CREDS_PATH}")


def auth_instagram(creds_path=None):
    c = load_creds(creds_path)
    ig = c.get("instagram") or {}
    _ask(ig, "client_id", "Instagram/Facebook App ID: ")
    _ask(ig, "client_secret", "Instagram/Facebook App Secret: ")
    _ask(ig, "ig_user_id", "Instagram Business/Creator user id: ")
    short = input("Short-lived access token with instagram_content_publish "
                  "(Graph API Explorer): ").strip()
    r = _check(httpx.get(f"https://graph.facebook.com/{GRAPH}/oauth/access_token",
                         params={"grant_type": "fb_exchange_token",
                                 "client_id": ig["client_id"],
                                 "client_secret": ig["client_secret"],
                                 "fb_exchange_token": short}),
               "instagram token exchange")
    d = r.json()
    ig.update(access_token=d["access_token"],
              expires_at=time.time() + int(d.get("expires_in", 5183944)))
    c["instagram"] = ig
    save_creds(c, creds_path)
    print(f"\nSaved Instagram credentials to "
          f"{creds_path or DEFAULT_CREDS_PATH} (long-lived, auto-refreshed)")


def auth_tiktok(creds_path=None):
    c = load_creds(creds_path)
    t = c.get("tiktok") or {}
    _ask(t, "client_key", "TikTok client key: ")
    _ask(t, "client_secret", "TikTok client secret: ")
    _ask(t, "redirect_uri", "TikTok redirect URI (must match the app config): ")
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    url = "https://www.tiktokapis.com/v2/auth/authorize/?" + urlencode({
        "client_key": t["client_key"],
        "scope": "user.info.basic,video.publish",
        "response_type": "code", "redirect_uri": t["redirect_uri"],
        "state": secrets.token_hex(8), "code_challenge": challenge,
        "code_challenge_method": "S256"})
    code = _paste_param("code", url)
    r = _check(httpx.post("https://open.tiktokapis.com/v2/oauth/token/", data={
        "client_key": t["client_key"], "client_secret": t["client_secret"],
        "code": code, "code_verifier": verifier,
        "grant_type": "authorization_code", "redirect_uri": t["redirect_uri"]}),
        "tiktok auth")
    d = r.json()
    if not d.get("access_token"):
        raise PublishError(f"tiktok auth: {r.text[:200]}")
    t.update(access_token=d["access_token"],
             refresh_token=d.get("refresh_token"),
             expires_at=time.time() + int(d.get("expires_in", 86400)))
    c["tiktok"] = t
    save_creds(c, creds_path)
    print(f"\nSaved TikTok credentials to {creds_path or DEFAULT_CREDS_PATH}")
    print("NOTE: public posting needs the app audited by TikTok; unaudited "
          "apps can only post SELF_ONLY.")


_AUTH = {"youtube": auth_youtube, "yt": auth_youtube,
         "instagram": auth_instagram, "ig": auth_instagram,
         "tiktok": auth_tiktok, "tt": auth_tiktok}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Anime pipeline auto-publisher")
    ap.add_argument("mode", choices=["auth", "publish"])
    ap.add_argument("target", help="auth: platform; publish: package dir")
    ap.add_argument("--creds", default=None, help="credentials file path")
    ap.add_argument("--platforms", default="youtube",
                    help="publish: comma list (youtube,instagram,tiktok)")
    ap.add_argument("--privacy", default="private",
                    choices=sorted(PRIVACY))
    args = ap.parse_args(argv)

    if args.mode == "auth":
        fn = _AUTH.get(args.target.lower())
        if not fn:
            ap.error(f"unknown platform {args.target!r}")
        fn(args.creds)
        return 0

    pkg_file = Path(args.target) / "package.json"
    pkg = load_json_file(pkg_file, None)
    if pkg is None:
        print(f"no package.json in {args.target}")
        return 1
    cfg = {"platforms": [p.strip() for p in args.platforms.split(",") if p.strip()],
           "privacy": args.privacy,
           "instagram": {}}
    publish_package(pkg, cfg, pkg_dir=args.target, creds_path=args.creds,
                    log=print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
