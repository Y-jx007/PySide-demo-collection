# -*- coding: utf-8 -*-
"""Construct every registered app window in an isolated process."""
import os
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


class TestAllAppWindows(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("RUN_WINDOW_SMOKE") == "1",
        "窗口级全量 smoke 由 CI 或 RUN_WINDOW_SMOKE=1 启用",
    )
    def test_every_app_window_constructs_runs_and_saves(self):
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        with temporary_directory() as tmp:
            def run(app):
                title = app["title"]
                output = os.path.join(tmp, title.replace("/", "_") + ".png")
                result = subprocess.run(
                    [sys.executable, HELPER, app["path"], output],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=40,
                )
                return title, output, result

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(run, app) for app in apps_manifest.APPS]
                for future in futures:
                    title, output, result = future.result()
                    with self.subTest(app=title):
                        detail = (result.stdout + "\n" + result.stderr)[-1200:]
                        self.assertEqual(result.returncode, 0, f"{title}: {detail}")
                        self.assertTrue(os.path.getsize(output) > 0, title)


if __name__ == "__main__":
    unittest.main()
