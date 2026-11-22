import ast
import asyncio
import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pipeline.task_manager import TaskManager, TaskStatus


def server_functions(*names, **namespace):
    path = Path(__file__).with_name("pipeline_server.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in names]
    for node in nodes:
        node.decorator_list = []
    namespace.setdefault("Request", SimpleNamespace)
    namespace.setdefault("asyncio", asyncio)
    if "async_services_check" not in namespace:
        async def _default_async_svc(**kw):
            return {"ok": True, "services": {"llm": True, "comfyui": True, "tts": True}}
        namespace["async_services_check"] = _default_async_svc
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class TaskTests(unittest.TestCase):
    def test_cancel_endpoint_serialization_and_capacity(self):
        manager = TaskManager()
        entered = threading.Event()
        release = threading.Event()
        stop_file = MagicMock()
        root = MagicMock()
        root.__truediv__.return_value = stop_file

        def worker(task):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Worker was not released")
            return {"ok": True}

        task = manager.submit("render", 1, worker)
        try:
            self.assertTrue(entered.wait(3))
            data = json.loads(json.dumps(task.to_dict()))
            self.assertEqual(data["status"], "running")
            self.assertNotIn("_cancel_event", data)
            self.assertNotIn("_thread", data)
            api = server_functions("api_tasks_cancel", tasks=manager, ROOT=root,
                                   STOP_FILE=stop_file)
            self.assertEqual(asyncio.run(api["api_tasks_cancel"](SimpleNamespace(), task.id)),
                             {"cancelled": True})
            self.assertTrue(task.is_cancelled)
            self.assertEqual(manager.active(), [task])
            self.assertIsNone(task.finished_at)
            with self.assertRaises(RuntimeError):
                manager.submit("render", 2, worker)
            stop_file.write_text.assert_called_once_with("stop")
        finally:
            release.set()
            task._thread.join(3)
        self.assertFalse(task._thread.is_alive())
        self.assertEqual(task.status, TaskStatus.CANCELLED)
        self.assertEqual(manager.active(), [])
        self.assertEqual(manager.history()[0]["status"], "cancelled")

    def test_render_stop_integration_without_dashboard_import(self):
        manager = TaskManager()
        entered = threading.Event()
        stop = threading.Event()
        stop_file = MagicMock()
        stop_file.write_text.side_effect = lambda text: stop.set()
        stop_file.unlink.side_effect = lambda **kwargs: stop.clear()
        ep = MagicMock()
        ep.preflight.return_value = True
        ep.ensure_services.return_value = (True, {})

        def produce(chapter, novel=None):
            entered.set()
            if not stop.wait(3):
                raise TimeoutError("Stop was not propagated")
            return None

        ep.produce_episode.side_effect = produce
        token_bucket = MagicMock()
        token_bucket.spend.return_value = (True, 0)
        ini_config = MagicMock()
        ini_config.token_cost_render.return_value = 1
        api = server_functions("api_start", "api_stop", tasks=manager, ep=ep,
                               STOP_FILE=stop_file, StartReq=SimpleNamespace,
                               _chapter_list=lambda novel=None: [{"chapter": 1}],
                               _novel_or_404=lambda novel_id: SimpleNamespace(
                                   id=novel_id or "sinners_world"),
                               DEFAULT_NOVEL_ID="sinners_world",
                               token_bucket=token_bucket, ini_config=ini_config)
        response = asyncio.run(api["api_start"](SimpleNamespace(), SimpleNamespace(chapter=1)))
        self.assertTrue(response["started"])
        task = manager.get(response["task_id"])
        try:
            self.assertTrue(entered.wait(3))
            self.assertEqual(asyncio.run(api["api_stop"](SimpleNamespace())), {"stopped": True})
        finally:
            stop.set()
            task._thread.join(3)
        self.assertFalse(task._thread.is_alive())
        self.assertEqual(task.status, TaskStatus.CANCELLED)
        self.assertEqual(manager.history()[0]["status"], "cancelled")
        ep.produce_episode.assert_called_once_with(1, novel="sinners_world")

    def test_redo_cancel_during_task(self):
        manager = TaskManager()
        entered = threading.Event()
        release = threading.Event()
        stop_file = MagicMock()

        def blocking_redo(task):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Redo was not released")
            return {"ok": True}

        task = manager.submit("redo", 1, blocking_redo)
        try:
            self.assertTrue(entered.wait(3))
            self.assertEqual(manager.active(), [task])
            manager.cancel_all()
            stop_file.write_text("stop")
            stop_file.write_text.assert_called_once_with("stop")
        finally:
            release.set()
            task._thread.join(3)
        self.assertEqual(task.status, TaskStatus.CANCELLED)
        self.assertEqual(task.phase, "cancelled")

    def test_pending_cancel_skips_callback(self):
        manager = TaskManager()
        fn = MagicMock()
        with patch("pipeline.task_manager.threading.Thread") as thread:
            task = manager.submit("render", 1, fn)
            self.assertTrue(manager.cancel(task.id))
            self.assertEqual(task.status, TaskStatus.PENDING)
            thread.call_args.kwargs["target"]()
        fn.assert_not_called()
        self.assertEqual(task.status, TaskStatus.CANCELLED)

    def test_results_exceptions_and_retention(self):
        manager = TaskManager(max_history=2)
        for result, status in (({"ok": True}, TaskStatus.COMPLETED),
                               ({"ok": False, "reason": "gate failed"}, TaskStatus.FAILED),
                               ({"ok": False}, TaskStatus.FAILED)):
            task = manager.submit("render", 1, lambda t: result)
            task._thread.join(3)
            self.assertFalse(task._thread.is_alive())
            self.assertEqual(task.status, status)
            self.assertEqual(task.result, result)
            self.assertEqual(task.progress, 100 if status == TaskStatus.COMPLETED else 0)
            if status == TaskStatus.FAILED:
                self.assertEqual(task.error, result.get("reason", "Task returned ok:false"))
            self.assertFalse(manager.cancel(task.id))
        self.assertEqual(len(manager.history()), 2)
        self.assertEqual(manager.history(0), [])
        self.assertEqual(len(manager._tasks), 2)
        snapshot = manager.history()
        snapshot[0]["result"]["reason"] = "changed"
        self.assertEqual(manager.history()[0]["result"]["reason"], "gate failed")
        task = manager.submit("render", 1, MagicMock(side_effect=ValueError("broken")))
        task._thread.join(3)
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("ValueError: broken", task.error)
        for i in range(150):
            task.log("INFO", str(i))
        self.assertEqual(len(task.logs), 100)
        self.assertEqual(len(task.to_dict()["logs"]), 50)

    def test_live_progress_and_stopping(self):
        from pipeline.task_manager import Task
        task = Task("progress", "render", 1, started_at=10)
        root = MagicMock()
        progress = root.__truediv__.return_value
        progress.stat.return_value.st_mtime = 11
        progress.read_text.return_value = json.dumps({"phase": "images", "percent": 42,
                                                     "detail": "Scene 2"})
        api = server_functions("_task_dict", ROOT=root, json=json)
        data = api["_task_dict"](task)
        self.assertEqual((data["phase"], data["progress"], data["detail"]),
                         ("images", 42, "Scene 2"))
        self.assertEqual(task.progress, 0)
        progress.stat.return_value.st_mtime = 9
        self.assertEqual(api["_task_dict"](task)["progress"], 0)
        progress.stat.return_value.st_mtime = 11
        task.cancel()
        self.assertEqual(api["_task_dict"](task)["phase"], "stopping")

    def test_sources_compile_without_importing_dashboard(self):
        scripts = Path(__file__).parent
        for name in ("pipeline_server.py", "pipeline/task_manager.py"):
            path = scripts / name
            compile(path.read_text(encoding="utf-8"), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
