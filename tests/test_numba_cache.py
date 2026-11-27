# -*- coding: utf-8 -*-
"""numba 缓存必须落在可写目录，否则只读安装里 5 个应用 import 直接失败。"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import numba_base  # noqa: E402


class NumbaCacheTests(unittest.TestCase):
    def test_cache_dir_is_writable(self):
        cache = Path(os.environ["NUMBA_CACHE_DIR"])
        self.assertTrue(cache.is_dir(), cache)
        probe = cache / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()

    def test_cache_dir_is_not_inside_apps(self):
        cache = Path(os.environ["NUMBA_CACHE_DIR"])
        self.assertNotIn("apps", cache.parts, cache)

    def test_cache_scope_is_per_entrypoint(self):
        """测试/冒烟工具与应用启动不能共用同一份 numba 缓存。"""
        original = sys.argv
        try:
            sys.argv = [r"C:\somewhere\apps\Physics\Maxwell.py"]
            self.assertEqual(numba_base._cache_scope(), "Maxwell")
            sys.argv = [r"C:\somewhere\tools\smoke_one_window.py"]
            self.assertEqual(numba_base._cache_scope(), "smoke_one_window")
            sys.argv = [r"C:\somewhere\app_runner.py"]
            self.assertEqual(numba_base._cache_scope(), "app_runner")
        finally:
            sys.argv = original

    def test_fallback_to_temp_when_runtime_is_read_only(self):
        import common.paths as paths

        original = paths.ensure_runtime

        def boom():
            raise OSError("read-only")

        paths.ensure_runtime = boom
        try:
            cache = numba_base._pick_cache_dir()
        finally:
            paths.ensure_runtime = original
        self.assertEqual(cache.name, "pyside_tools_numba_cache")
        self.assertTrue(cache.is_dir())


if __name__ == "__main__":
    unittest.main()
