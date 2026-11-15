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

    def test_equation_panel_states_the_model(self):
        text = self.window.equation_label.text()
        self.assertIn("∇²ψ", text)
        self.assertIn("|ψ|²", text)
        self.assertIn("Kerr", text)
        self.assertIn("饱和", text)
        self.assertEqual(text.count("<div"), text.count("</div>"))
        self.assertTrue(self.window.equation_label.toolTip())

    def test_readout_shows_numbers(self):
        self.window.tick()
        text = self.window.readout_label.text()
        self.assertIn("功率", text)
        self.assertIn("步数", text)

    def test_view_switch_moves_the_canvas_between_field_and_spectrum(self):
        self.window.tick()
        field_image = self.window.canvas.image()
        self.window.view_combo.setCurrentText("动量谱")
        self.assertEqual(self.window.canvas.mode, "动量谱")
        self.assertIsNot(self.window.canvas.image(), field_image)
        self.window.view_combo.setCurrentText("场")
        self.assertEqual(self.window.canvas.mode, "场")

    def test_model_switch_exposes_the_kappa_slider(self):
        # 窗口没 show()，用 isHidden() 判断显式隐藏状态
        self.assertTrue(self.window.kappa_slider.isHidden())
        self.window.model_combo.setCurrentText("高阶 Kerr")
        self.assertEqual(self.window.canvas.sim.model, "高阶 Kerr")
        self.assertFalse(self.window.kappa_slider.isHidden())
        self.window.kappa_slider.setValue(50)
        self.assertAlmostEqual(self.window.canvas.sim.kappa, 0.05, places=6)

    def test_response_plot_follows_the_peak(self):
        self.window.tick()
        self.assertGreater(self.window.response_plot.peak, 0.0)
        self.assertIs(self.window.response_plot.sim, self.window.canvas.sim)


if __name__ == "__main__":
    unittest.main()