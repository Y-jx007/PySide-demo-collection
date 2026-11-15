# -*- coding: utf-8 -*-
"""Import smoke for every app module in the registry."""
import importlib.util
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import apps_manifest  # noqa: E402


class TestAllAppImports(unittest.TestCase):
    def test_every_app_module_imports(self):
        failed = []
        for app in apps_manifest.APPS:
            relative = app["path"]
            spec = importlib.util.spec_from_file_location(
                "import_" + app["title"].replace(" ", "_").replace("-", "_"),
                os.path.join(ROOT, "apps", relative),
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
