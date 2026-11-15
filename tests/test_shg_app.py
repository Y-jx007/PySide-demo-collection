# -*- coding: utf-8 -*-
"""The SHG window: sliders, the symmetry switch and the QPM control."""
import importlib.util
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


def load_app():
    path = os.path.join(ROOT, "apps", "Physics", "SHG.py")
    spec = importlib.util.spec_from_file_location("shg_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSHGWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.module = load_app()

    def setUp(self):
        self.window = self.module.MainWindow()

    def tearDown(self):
        self.window.close()

    def test_canvas_is_painted(self):
        self.assertIsNotNone(self.window.canvas.image())

    def test_symmetry_switch_erases_the_efficiency(self):
        self.assertGreater(self.window.canvas.model.peak_efficiency(), 0.0)
        self.window.symmetric_check.setChecked(True)
        self.assertEqual(self.window.canvas.model.peak_efficiency(), 0.0)
        self.assertIn("0.0%", self.window.readout_label.text())

    def test_qpm_enables_the_period_slider_and_changes_the_map(self):
        self.assertTrue(self.window.poling_slider.isHidden() is False)
        self.assertFalse(self.window.poling_slider.isEnabled())
        before = self.window.canvas.model.efficiency().copy()
        self.window.qpm_check.setChecked(True)
        self.assertTrue(self.window.poling_slider.isEnabled())
        self.assertFalse((self.window.canvas.model.efficiency() == before).all())

    def test_length_slider_reaches_the_model(self):
        self.window.length_slider.setValue(150)
        self.assertAlmostEqual(self.window.canvas.model.length, 1.5, places=6)

    def test_equation_panel_states_the_model(self):
        text = self.window.equation_label.text()
        self.assertIn("Δk", text)
        self.assertIn("χ⁽²⁾", text)
        self.assertTrue(self.window.equation_label.toolTip())


if __name__ == "__main__":
    unittest.main()