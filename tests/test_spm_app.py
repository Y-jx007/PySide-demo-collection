# -*- coding: utf-8 -*-
"""The SPM window: waterfall stepping, pause/reset and the spectrum plot."""
import importlib.util
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


def load_app():
    path = os.path.join(ROOT, "apps", "Physics", "SPM.py")
    spec = importlib.util.spec_from_file_location("spm_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSPMWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.module = load_app()

    def setUp(self):
        self.window = self.module.MainWindow()
        self.window.timer.stop()

    def tearDown(self):
        self.window.close()

    def test_tick_advances_and_paints(self):
        steps = self.window.steps_slider.value()
        self.window.tick()
        self.assertEqual(self.window.canvas.sim.steps, steps)
        self.assertIsNotNone(self.window.canvas.image())

    def test_pause_stops_the_stepping(self):
        self.window.toggle_pause()
        self.window.tick()
        self.assertEqual(self.window.canvas.sim.steps, 0)
        self.assertEqual(self.window.pause_button.text(), "继续")

    def test_order_slider_resets_the_pulse(self):
        self.window.tick()
        self.window.order_slider.setValue(300)
        self.assertAlmostEqual(self.window.order(), 3.0, places=6)
        self.assertEqual(self.window.canvas.sim.steps, 0)

    def test_readout_reports_soliton_numbers(self):
        self.window.tick()
        text = self.window.readout_label.text()
        for token in ("能量", "峰值", "时宽", "谱宽"):
            self.assertIn(token, text)

    def test_fundamental_soliton_keeps_its_width(self):
        self.window.order_slider.setValue(100)
        before = self.window.canvas.sim.width()
        for _ in range(40):
            self.window.tick()
        self.assertAlmostEqual(self.window.canvas.sim.width(), before,
                               delta=0.05 * before)

    def test_equation_panel_states_the_model(self):
        text = self.window.equation_label.text()
        self.assertIn("∂A/∂z", text)
        self.assertIn("β₂", text)
        self.assertTrue(self.window.equation_label.toolTip())


if __name__ == "__main__":
    unittest.main()