# -*- coding: utf-8 -*-
"""The Soliton window: timer stepping, pause/reset and the readout."""
import importlib.util
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


def load_app():
    path = os.path.join(ROOT, "apps", "Physics", "Soliton.py")
    spec = importlib.util.spec_from_file_location("soliton_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSolitonWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.module = load_app()

    def setUp(self):
        self.window = self.module.MainWindow()
        self.window.timer.stop()

    def tearDown(self):
        self.window.close()

    def test_tick_advances_the_field_and_the_canvas(self):
        steps = self.window.steps_slider.value()
        self.window.tick()
        self.assertEqual(self.window.canvas.sim.steps, steps)
        self.assertIsNotNone(self.window.canvas.image())

    def test_pause_stops_the_stepping(self):
        self.window.toggle_pause()
        self.window.tick()
        self.assertEqual(self.window.canvas.sim.steps, 0)
        self.assertEqual(self.window.pause_button.text(), "继续")

    def test_reset_returns_to_step_zero(self):
        self.window.tick()
        self.window.tick()
        self.window.toggle_pause()
        self.window.reset()
        self.assertEqual(self.window.canvas.sim.steps, 0)
        self.assertEqual(self.window.pause_button.text(), "暂停")

    def test_preset_switch_rebuilds_the_initial_condition(self):
        self.window.preset_combo.setCurrentText("双束")
        self.assertEqual(self.window.preset(), "双束")
        self.assertEqual(self.window.canvas.sim.steps, 0)
        self.window.tick()
        self.assertGreater(self.window.canvas.sim.steps, 0)

    def test_amplitude_slider_reaches_the_simulation(self):
        self.window.amp_slider.setValue(100)
        self.assertAlmostEqual(self.window.amplitude(), 1.0, places=6)
        self.assertEqual(self.window.canvas.sim.steps, 0)

    def test_readout_shows_numbers(self):
        self.window.tick()
        self.assertIn("功率", self.window.power_label.text())
        self.assertIn("步数", self.window.step_label.text())


if __name__ == "__main__":
    unittest.main()