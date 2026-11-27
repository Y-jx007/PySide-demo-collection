# -*- coding: utf-8 -*-
"""Lenia 默认 256² 用不到 torch/scipy，模块导入时不应把它们拉起来。"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROBE = (
    "import importlib.util, sys\n"
    "path = sys.argv[1]\n"
    "spec = importlib.util.spec_from_file_location('lenia_import_probe', path)\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
    "assert module.torch is None, 'torch should stay lazy'\n"
    "assert module.scipy_fft is None, 'scipy should stay lazy'\n"
    "assert 'torch' not in sys.modules, 'torch must not be imported at module import'\n"
    "print('ok')\n"
)


class LeniaLazyImportTests(unittest.TestCase):
    def test_module_import_does_not_pull_torch_or_scipy(self):
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
        result = subprocess.run(
            [sys.executable, "-c", PROBE,
             os.path.join(ROOT, "apps", "Cellular Automata", "Lenia.py")],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-1200:])
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
