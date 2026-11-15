# -*- coding: utf-8 -*-
"""Visual regression for deterministic exemplar canvases."""
import importlib.util
import os
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if ROOT not in os.path.dirname(__file__):
    pass

from PySide6.QtWidgets import QApplication  # noqa: E402

from tests.visual_utils import capture_canvas, image_array  # noqa: E402

CASES = (
    ("apps/Math/Quasicrystal.py", "Quasicrystal.png"),
    ("apps/Physics/Mass-Spring.py", "Mass-Spring.png"),
)


class TestVisualRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _capture(self, relative, name):
        spec = importlib.util.spec_from_file_location(
            "visual_" + name.split(".")[0], os.path.join(ROOT, relative)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        window = module.MainWindow()
        if hasattr(window, "timer"):
            window.timer.stop()
        if name == "Mass-Spring.png":
            window.sim.reset("悬索桥")
        window.show()
        pix = capture_canvas(window, self.app)
        window.close()
        return pix

    def test_quasicrystal_canvas(self):
        pix = self._capture(*CASES[0])
        baseline = self._baseline("Quasicrystal.png")
        current = image_array(pix).astype(np.int16)
        diff = np.abs(current - baseline.astype(np.int16))
        self.assertLess(float(diff.mean()), 2.0)
        self.assertLess(float((diff > 20).mean()), 0.01)

    def test_mass_spring_canvas(self):
        pix = self._capture(*CASES[1])
        baseline = self._baseline("Mass-Spring.png")
        current = image_array(pix).astype(np.int16)
        diff = np.abs(current - baseline.astype(np.int16))
        self.assertLess(float(diff.mean()), 2.0)
        self.assertLess(float((diff > 20).mean()), 0.01)

    def _baseline(self, name):
        from PySide6.QtGui import QImage

        path = os.path.join(ROOT, "tests", "baselines", name)
        image = QImage(path).convertToFormat(QImage.Format_RGB888)
        width, height = image.width(), image.height()
        line = image.bytesPerLine()
        raw = np.frombuffer(image.bits(), dtype=np.uint8).reshape(height, line)
        return raw[:, : width * 3].reshape(height, width, 3).copy()


if __name__ == "__main__":
    unittest.main()
