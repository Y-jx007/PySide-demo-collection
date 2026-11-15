# -*- coding: utf-8 -*-
"""Headless construction/run/save smoke for the canonical apps."""
import importlib.util
import os
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

CASES = (
    ("apps/Math/Quasicrystal.py", "Quasicrystal"),
    ("apps/Physics/Mass-Spring.py", "Mass-Spring"),
    ("apps/Cellular Automata/Particle Lenia.py", "Particle Lenia"),
)


class TestHeadlessAppSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_apps_construct_run_save_and_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            for relative, label in CASES:
                spec = importlib.util.spec_from_file_location(
                    "smoke_" + label.replace(" ", "_"),
                    os.path.join(ROOT, relative),
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                window = module.MainWindow()
                window.show()
                QTest.qWait(180)
                self.app.processEvents()
                path = os.path.join(tmp, label + ".png")
                self.assertTrue(window.canvas.save_image(path))
                self.assertTrue(os.path.getsize(path) > 0)
                window.close()


if __name__ == "__main__":
    unittest.main()
