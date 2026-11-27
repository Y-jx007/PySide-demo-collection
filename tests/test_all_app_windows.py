# -*- coding: utf-8 -*-
"""Construct every registered app window in an isolated process."""
import os
import re
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import apps_manifest  # noqa: E402
from tests.tmpdir import temporary_directory  # noqa: E402

HELPER = os.path.join(ROOT, "tools", "smoke_one_window.py")

# 这些应用需要真实 OpenGL 上下文或 CUDA；GitHub 的 windows-latest 没有 GPU，
# 在 cpu 模式下跳过。完整 41 应用 smoke 仍在本地用 RUN_WINDOW_SMOKE=1 跑。
_GPU_SOURCE_MARKERS = (
    re.compile(r"gl\s*=\s*True"),
    re.compile(r"QOpenGLWidget"),
    re.compile(r"common\.taichi_base"),
    re.compile(r"taichi_forge"),
    re.compile(r"common\.gpu_surface"),
    re.compile(r"common\.gpu_fft"),
)


def _needs_gpu(app):
    if "torch" in app.get("requires", ()):
        return True
    path = os.path.join(ROOT, "apps", app["path"])
    try:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
    except OSError:
        return True
    return any(marker.search(source) for marker in _GPU_SOURCE_MARKERS)


def _smoke_mode():
    value = os.environ.get("RUN_WINDOW_SMOKE", "").strip().lower()
    if value in ("cpu", "no-gpu", "nogpu"):
        return "cpu"
    if value in ("1", "true", "full", "all"):
        return "full"
    return ""


class TestAllAppWindows(unittest.TestCase):
    @unittest.skipUnless(
        _smoke_mode(),
        "窗口级全量 smoke 由 RUN_WINDOW_SMOKE=1 启用；CI 用 RUN_WINDOW_SMOKE=cpu",
    )
    def test_every_app_window_constructs_runs_and_saves(self):
        mode = _smoke_mode()
        apps = list(apps_manifest.APPS)
        if mode == "cpu":
            apps = [app for app in apps if not _needs_gpu(app)]
        self.assertGreater(len(apps), 0, "没有可 smoke 的应用")

        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"

        with temporary_directory() as tmp:
            def run(app):
                title = app["title"]
                output = os.path.join(tmp, title.replace("/", "_") + ".png")
                attempts = []
                # GL 窗口并行时偶发 0xC0000409；失败重试一次，两次都失败才算错。
                for _ in range(2):
                    result = subprocess.run(
                        [sys.executable, HELPER, app["path"], output],
                        cwd=ROOT,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=40,
                    )
                    attempts.append(result)
                    if result.returncode == 0 and os.path.exists(output) and os.path.getsize(output) > 0:
                        return title, output, result, attempts
                return title, output, attempts[-1], attempts

            # 2 个并发比 4 个更稳：QOpenGLWidget/Taichi 同时初始化时不容易撞车。
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run, app) for app in apps]
                for future in futures:
                    title, output, result, attempts = future.result()
                    with self.subTest(app=title):
                        detail = (result.stdout + "\n" + result.stderr)[-1200:]
                        if len(attempts) > 1:
                            detail = f"(retried {len(attempts)}x)\n" + detail
                        self.assertEqual(result.returncode, 0, f"{title}: {detail}")
                        self.assertTrue(os.path.getsize(output) > 0, title)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
