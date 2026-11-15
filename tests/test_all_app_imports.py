# -*- coding: utf-8 -*-
"""Import smoke for every app module in the registry."""
import importlib.util
import os
import subprocess
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import apps_manifest  # noqa: E402


# Taichi 在同一个进程里反复初始化 GPU 后端会偶发崩溃，所以带 taichi 的应用
# 单独起子进程导入；其余应用仍然就地导入（快）。
ISOLATED_CODE = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location('probe', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
)


def _uses_taichi(absolute_path):
    with open(absolute_path, encoding="utf-8") as handle:
        return "taichi" in handle.read()


def _import_isolated(absolute_path):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    return subprocess.run(
        [sys.executable, "-c", ISOLATED_CODE, absolute_path],
        capture_output=True, text=True, timeout=300, env=env, cwd=ROOT,
    )


class TestAllAppImports(unittest.TestCase):
    def test_every_app_module_imports(self):
        failed = []
        for app in apps_manifest.APPS:
            relative = app["path"]
            absolute = os.path.join(ROOT, "apps", relative)
            if _uses_taichi(absolute):
                result = _import_isolated(absolute)
                if result.returncode != 0:
                    failed.append((relative, result.stderr[-400:]))
                continue
            spec = importlib.util.spec_from_file_location(
                "import_" + app["title"].replace(" ", "_").replace("-", "_"),
                absolute,
            )
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as exc:
                if exc.name in {"torch", "cplot", "sounddevice"}:
                    continue
                failed.append((relative, repr(exc)))
            except Exception as exc:  # noqa: BLE001 - collect every app failure
                failed.append((relative, repr(exc)))
        self.assertEqual(failed, [], f"应用导入失败: {failed}")


if __name__ == "__main__":
    unittest.main()
