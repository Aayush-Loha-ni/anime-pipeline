#!/usr/bin/env python3
"""Tests for pipeline.publisher — the opt-in auto-publish module.

ponytail: no live network here. Every httpx entry point is stubbed, so these
tests assert REQUEST SHAPE (methods, paths, privacy mapping, chunked upload)
and the safety gates (disabled config, NOT_RELEASE_READY, per-platform
failure isolation), which is the part that can be wrong and silent.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import publisher as P


class _Resp:
    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


class _FakeClient:
    """Records calls; `handler(request-ish tuple) -> _Resp` decides replies."""

    def __init__(self, handler, calls, **kw):
        self._handler = handler
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, **kw):
        self._calls.append(("GET", url, kw))
        return self._handler(("GET", url, kw))

    def post(self, url, **kw):
        self._calls.append(("POST", url, kw))
        return self._handler(("POST", url, kw))

    def put(self, url, **kw):
        self._calls.append(("PUT", url, kw))
        return self._handler(("PUT", url, kw))


class _FakeModuleHTTP:
    """Stub for the module-level httpx.post/httpx.get (token endpoints)."""

    def __init__(self, handler, calls):
        self._handler = handler
        self._calls = calls

    def post(self, url, **kw):
        self._calls.append(("POST", url, kw))
        return self._handler(("POST", url, kw))

    def get(self, url, **kw):
        self._calls.append(("GET", url, kw))
        return self._handler(("GET", url, kw))


class PublisherTest(unittest.TestCase):

    def setUp(self):
        self.calls = []
        self.tmp = Path(tempfile.mkdtemp())
        self.creds_file = self.tmp / "creds.json"
        self.creds_file.write_text(json.dumps({
            "youtube": {"client_id": "cid", "client_secret": "cs",
                        "refresh_token": "rt", "access_token": "tok",
                        "expires_at": 9e18},
            "tiktok": {"client_key": "tk", "client_secret": "ts",
                       "access_token": "ttok", "refresh_token": "trt",
                       "expires_at": 9e18},
            "instagram": {"client_id": "app", "client_secret": "secret",
                          "ig_user_id": "123", "access_token": "igtok",
                          "expires_at": 9e18},
        }), encoding="utf-8")
        self.video = self.tmp / "ep.mp4"
        self.video.write_bytes(b"x" * 1024)

        def fake_client_factory(*a, **kw):
            return _FakeClient(self.handler, self.calls, **kw)

        self._patches = [
            mock.patch.object(P.httpx, "Client", fake_client_factory),
            mock.patch.object(P.httpx, "post",
                              _FakeModuleHTTP(self.handler, self.calls).post),
            mock.patch.object(P.httpx, "get",
                              _FakeModuleHTTP(self.handler, self.calls).get),
            mock.patch.object(P, "save_creds",
                              lambda creds, path=None: creds),
            mock.patch.object(P.time, "sleep", lambda s: None),
            mock.patch.object(P.webbrowser, "open", lambda url: None),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    def handler(self, call):
        method, url, kw = call
        if url == "https://oauth2.googleapis.com/token":
            return _Resp(200, {"access_token": "fresh", "expires_in": 3599})
        if "googleapis.com/upload" in url and method == "POST":
            return _Resp(200, headers={"location": "https://up.example/sess"})
        if url == "https://up.example/sess" and method == "PUT":
            # 308 until the last byte, then the video resource.
            cr = kw["headers"]["Content-Range"]  # "bytes 0-4095/8388608"
            end = int(cr.split()[1].split("/")[0].split("-")[1])
            total = int(cr.split("/")[1])
            return _Resp(200, {"id": "VID1"}) if end == total - 1 else _Resp(308)
        if "graph.facebook.com" in url and url.endswith("/media") and method == "POST":
            return _Resp(200, {"id": "CONT1"})
        if "graph.facebook.com" in url and url.endswith("/media_publish") and method == "POST":
            return _Resp(200, {"id": "MED1"})
        if "graph.facebook.com" in url and method == "GET":
            return _Resp(200, {"status_code": "FINISHED"})
        if "tiktokapis.com" in url and "init" in url:
            return _Resp(200, {"data": {"publish_id": "PUB1",
                                        "upload_url": "https://tt.example/up"}})
        if "tiktokapis.com" in url and "status" in url:
            return _Resp(200, {"data": {"status": "PUBLISH_COMPLETE"}})
        if url == "https://tt.example/up" and method == "PUT":
            return _Resp(200, {})
        return _Resp(200, {})

    # --- safety gates -------------------------------------------------

    def test_disabled_by_default_config(self):
        pkg = {"video_path": str(self.video), "chapter": 1}
        out = P.publish_package(pkg, {}, log=lambda m: None)
        self.assertEqual(out["results"], {})
        self.assertEqual(self.calls, [])

    def test_not_release_ready_blocks_everything(self):
        pkg = {"video_path": str(self.video), "chapter": 1,
               "release_report": {"verdict": "NOT_RELEASE_READY",
                                  "summary": {"blockers": 2}}}
        out = P.publish_package(
            pkg, {"platforms": ["youtube", "tiktok"], "privacy": "public"},
            creds_path=str(self.creds_file), log=lambda m: None)
        self.assertEqual(out["blocked"], "NOT_RELEASE_READY")
        self.assertEqual(self.calls, [])

    def test_one_platform_failing_does_not_stop_others(self):
        base = self.handler

        def h(call):
            if "tiktokapis.com" in call[1]:
                return _Resp(403, {"error": {"code": "access_token_invalid",
                                             "message": "nope"}})
            return base(call)

        self.handler = h
        pkg = {"video_path": str(self.video), "chapter": 7,
               "metadata": {"title_options": ["T"], "description": "D"}}
        out = P.publish_package(
            pkg, {"platforms": ["tiktok", "youtube"]},
            creds_path=str(self.creds_file), log=lambda m: None)
        self.assertFalse(out["results"]["tiktok"]["ok"])
        self.assertTrue(out["results"]["youtube"]["ok"])
        self.assertEqual(out["results"]["youtube"]["result"]["id"], "VID1")

    def test_unknown_privacy_falls_back_to_private(self):
        pkg = {"video_path": str(self.video), "chapter": 1}
        out = P.publish_package(
            pkg, {"platforms": ["tiktok"], "privacy": "PUBLIC-TO-EVERYONE"},
            creds_path=str(self.creds_file), log=lambda m: None)
        self.assertTrue(out["results"]["tiktok"]["ok"])
        posts = [c for c in self.calls if "init" in c[1]]
        self.assertEqual(posts[0][2]["json"]["post_info"]["privacy_level"],
                         "SELF_ONLY")

    # --- request shape -------------------------------------------------

    def test_youtube_request_shape_and_privacy(self):
        pkg = {"video_path": str(self.video), "chapter": 3,
               "metadata": {"title_options": ["Ep 3"], "description": "desc",
                            "tags": ["anime", "gothic"]}}
        out = P.publish_package(
            pkg, {"platforms": ["youtube"], "privacy": "unlisted"},
            creds_path=str(self.creds_file), log=lambda m: None)
        self.assertTrue(out["results"]["youtube"]["ok"])
        sess = [c for c in self.calls if "upload/youtube/v3/videos" in c[1]][0]
        self.assertEqual(sess[0], "POST")
        self.assertEqual(sess[2]["params"],
                         {"uploadType": "resumable", "part": "snippet,status"})
        self.assertEqual(sess[2]["json"]["status"]["privacyStatus"], "unlisted")
        self.assertFalse(sess[2]["json"]["status"]["selfDeclaredMadeForKids"])
        self.assertEqual(sess[2]["json"]["snippet"]["title"], "Ep 3")
        self.assertEqual(sess[2]["json"]["snippet"]["tags"], ["anime", "gothic"])
        put = [c for c in self.calls if c[1] == "https://up.example/sess"][0]
        self.assertEqual(put[2]["headers"]["Content-Range"],
                         f"bytes 0-{self.video.stat().st_size - 1}/"
                         f"{self.video.stat().st_size}")

    def test_youtube_chunked_upload_walks_the_file(self):
        size = (4 << 10) + 10
        big = self.tmp / "big.mp4"
        big.write_bytes(b"x" * size)
        with mock.patch.object(P, "CHUNK", 4 << 10):
            P.upload_youtube(big, "t", "d", creds=P.load_creds(self.creds_file),
                             creds_path=str(self.creds_file), log=lambda m: None)
        puts = [c for c in self.calls if c[1] == "https://up.example/sess"]
        covered = 0
        for c in puts:
            span, total = c[2]["headers"]["Content-Range"].split()[1].split("/")
            start, end = (int(x) for x in span.split("-"))
            self.assertEqual(start, covered)   # contiguous, no gap/overlap
            self.assertEqual(int(total), size)
            self.assertEqual(len(c[2]["content"]), end - start + 1)
            covered = end + 1
        self.assertEqual(covered, size)         # every byte was sent
        self.assertEqual(puts[-1][2]["content"], b"x" * 10)

    def test_youtube_http_error_raises_publisherror(self):
        base = self.handler

        def h(call):
            if call[1].endswith("/v3/videos"):
                return _Resp(403, text="quota exceeded")
            return base(call)

        self.handler = h
        with self.assertRaises(P.PublishError):
            P.upload_youtube(self.video, "t", "d",
                             creds=P.load_creds(self.creds_file),
                             creds_path=str(self.creds_file),
                             log=lambda m: None)

    def test_tiktok_error_code_in_200_body_raises(self):
        base = self.handler

        def h(call):
            if "init" in call[1]:
                return _Resp(200, {"error": {"code": "spam_risk_too_many_posts",
                                             "message": "rate limited"}})
            return base(call)

        self.handler = h
        with self.assertRaises(P.PublishError):
            P.upload_tiktok(self.video, "t", creds=P.load_creds(self.creds_file),
                            creds_path=str(self.creds_file), log=lambda m: None)

    def test_instagram_requires_public_video_url(self):
        with self.assertRaisesRegex(P.PublishError, "video_url"):
            P.upload_instagram(self.video, "", "cap",
                               creds=P.load_creds(self.creds_file),
                               creds_path=str(self.creds_file),
                               log=lambda m: None)

    def test_instagram_two_step_publish(self):
        out = P.publish_package(
            {"video_path": str(self.video), "chapter": 1,
             "metadata": {"title_options": ["T"], "description": "d",
                          "tags": ["anime"]}},
            {"platforms": ["instagram"], "privacy": "public",
             "instagram": {"video_url": "https://pub.example/ep.mp4"}},
            creds_path=str(self.creds_file), log=lambda m: None)
        self.assertTrue(out["results"]["instagram"]["ok"])
        cont = [c for c in self.calls if c[1].endswith("/media") and c[0] == "POST"][0]
        self.assertEqual(cont[2]["data"]["media_type"], "REELS")
        self.assertEqual(cont[2]["data"]["share_to_feed"], "true")
        self.assertIn("#anime", cont[2]["data"]["caption"])
        pub = [c for c in self.calls if c[1].endswith("/media_publish")][0]
        self.assertEqual(pub[2]["data"]["creation_id"], "CONT1")

    def test_tiktok_single_chunk_file_upload(self):
        out = P.publish_package(
            {"video_path": str(self.video), "chapter": 1,
             "metadata": {"title_options": ["T"], "description": "d"}},
            {"platforms": ["tt"], "privacy": "public"},
            creds_path=str(self.creds_file), log=lambda m: None)
        self.assertTrue(out["results"]["tt"]["ok"])
        init = [c for c in self.calls if "init" in c[1]][0]
        size = self.video.stat().st_size
        self.assertEqual(init[2]["json"]["source_info"],
                         {"source": "FILE_UPLOAD", "video_size": size,
                          "chunk_size": size, "total_chunk_count": 1})
        self.assertEqual(init[2]["json"]["post_info"]["privacy_level"],
                         "PUBLIC_TO_EVERYONE")
        put = [c for c in self.calls if c[1] == "https://tt.example/up"][0]
        self.assertEqual(put[2]["headers"]["Content-Range"],
                         f"bytes 0-{size - 1}/{size}")

    # --- credentials / tokens ------------------------------------------

    def test_expired_youtube_token_refreshes_and_persists(self):
        c = P.load_creds(self.creds_file)
        c["youtube"]["expires_at"] = 0
        c["youtube"]["access_token"] = None
        saved = {}

        def fake_save(creds, path=None):
            saved.update(creds)
            return path

        def h(call):
            if call[1] == "https://oauth2.googleapis.com/token":
                return _Resp(200, {"access_token": "fresh",
                                   "expires_in": 3599})
            return base(call)

        base = self.handler
        self.handler = h
        with mock.patch.object(P, "save_creds", fake_save):
            P.upload_youtube(self.video, "t", "d", creds=c,
                             creds_path=str(self.creds_file), log=lambda m: None)
        self.assertEqual(saved["youtube"]["access_token"], "fresh")
        self.assertGreater(saved["youtube"]["expires_at"], 1.7e9)

    def test_missing_credentials_error_names_the_auth_command(self):
        empty = self.tmp / "empty.json"
        empty.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(P.PublishError, "auth"):
            P.upload_youtube(self.video, "t", "d", creds={}, creds_path=str(empty),
                             log=lambda m: None)
        with self.assertRaisesRegex(P.PublishError, "auth"):
            P.upload_tiktok(self.video, "t", creds={}, creds_path=str(empty),
                            log=lambda m: None)

    def test_publish_result_written_next_to_the_package(self):
        pkg = {"video_path": str(self.video), "chapter": 1,
               "metadata": {"title_options": ["T"], "description": "d"}}
        P.publish_package(pkg, {"platforms": ["youtube"]},
                          creds_path=str(self.creds_file), log=lambda m: None)
        res = json.loads((self.tmp / "publish_result.json").read_text(encoding="utf-8"))
        self.assertTrue(res["results"]["youtube"]["ok"])
        self.assertEqual(res["privacy"], "private")

    def test_stale_video_path_falls_back_to_the_package_copy(self):
        # ch001's package.json records a source mp4 that was moved away after
        # the package was built; the copy inside the package must be used.
        pkgdir = self.tmp / "pkg"
        pkgdir.mkdir()
        (pkgdir / "ch001_youtube.mp4").write_bytes(b"x" * 1024)
        pkg = {"video_path": str(self.tmp / "moved_away.mp4"), "chapter": 1,
               "metadata": {"title_options": ["T"], "description": "d"}}
        out = P.publish_package(pkg, {"platforms": ["youtube"]},
                                pkg_dir=str(pkgdir),
                                creds_path=str(self.creds_file), log=lambda m: None)
        self.assertTrue(out["results"]["youtube"]["ok"])
        puts = [c for c in self.calls if c[1] == "https://up.example/sess"]
        self.assertEqual(len(puts), 1)
        self.assertTrue((pkgdir / "publish_result.json").exists())


if __name__ == "__main__":
    unittest.main()
