import httpx
import unittest


class NiceGUIDashboardTests(unittest.TestCase):
    def test_module_imports_without_side_effects(self):
        import nicegui_dashboard as v3
        self.assertEqual(v3.BASE, "http://127.0.0.1:8000")
        self.assertTrue(callable(v3.mount_v3))
        self.assertFalse(hasattr(v3, "mount_v2"))

    def test_mount_v3_registers_on_plain_fastapi_app(self):
        from fastapi import FastAPI
        import nicegui_dashboard as v3
        app = FastAPI()
        v3.mount_v3(app)  # must not raise
        paths = [getattr(r, "path", "") for r in app.routes]
        self.assertIn("/v3", paths)

    def test_double_mount_does_not_raise(self):
        from fastapi import FastAPI
        import nicegui_dashboard as v3
        v3.mount_v3(FastAPI())
        v3.mount_v3(FastAPI())

    def test_api_helpers_coerce_bad_payloads(self):
        import asyncio
        import nicegui_dashboard as v3

        def located(payload, status=200):
            def handler(request):
                return httpx.Response(status, json=payload)
            return httpx.MockTransport(handler)

        async def run():
            async with httpx.AsyncClient(transport=located([1, 2])) as c:
                r = await v3._api_get(c, "http://x", "/api/status")
                self.assertIn("_error", r)
            async with httpx.AsyncClient(transport=located({"ok": True})) as c:
                r = await v3._api_post(c, "http://x", "/api/stop")
                self.assertEqual(r, {"ok": True})
            async with httpx.AsyncClient(transport=located({"detail": "nope"}, 404)) as c:
                r = await v3._api_get(c, "http://x", "/api/scenes/status")
                self.assertIn("_error", r)
                self.assertIn("404", r["_error"])

            def boom(request):
                raise httpx.ConnectError("down")
            async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as c:
                r = await v3._api_get(c, "http://x", "/api/status")
                self.assertIn("down", r["_error"])
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
