# -*- coding: utf-8 -*-
"""The merged nonlinear-optics window: three modes, one canvas, one panel stack."""
import importlib.util
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from PySide6.QtWidgets import QApplication  # noqa: E402


def load_app():
    path = os.path.join(ROOT, "apps", "Physics", "Nonlinear.py")
    spec = importlib.util.spec_from_file_location("nonlinear_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestNonlinearWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.module = load_app()

    def setUp(self):
        self.window = self.module.MainWindow()
        self.window.timer.stop()

    def tearDown(self):
        self.window.close()

    def switch(self, name):
        self.window.mode_combo.setCurrentText(name)
        self.app.processEvents()

    def test_starts_in_space_mode(self):
        self.assertEqual(self.window.mode, "空间自聚焦")
        self.assertEqual(self.window.stack.currentIndex(), 0)
        self.assertIsNotNone(self.window.canvas.image())

    def test_mode_switch_moves_the_stack_and_the_canvas(self):
        self.switch("时间孤子")
        self.assertEqual(self.window.stack.currentIndex(), 1)
        self.assertEqual(self.window.canvas.mode, "时间孤子")
        self.switch("调频")
        self.assertEqual(self.window.stack.currentIndex(), 2)
        self.assertFalse(self.window.pause_button.isEnabled())

    def test_space_mode_ticks_and_pauses(self):
        steps = self.window.steps_slider.value()
        self.window.tick()
        self.assertEqual(self.window.canvas.space.steps, steps)
        self.window.toggle_pause()
        self.window.tick()
        self.assertEqual(self.window.canvas.space.steps, steps)

    def test_space_preset_and_amplitude_reset_the_field(self):
        self.window.tick()
        self.window.preset_combo.setCurrentText("双束")
        self.assertEqual(self.window.canvas.space.steps, 0)
        self.window.amp_slider.setValue(100)
        self.assertAlmostEqual(self.window.amplitude(), 1.0, places=6)
        self.assertEqual(self.window.canvas.space.steps, 0)

    def test_space_view_and_model_switches(self):
        self.window.tick()
        field_image = self.window.canvas.image()
        self.window.view_combo.setCurrentText("动量谱")
        self.assertIsNot(self.window.canvas.image(), field_image)
        self.window.view_combo.setCurrentText("场")
        self.assertTrue(self.window.model_kappa_slider.isHidden())
        self.window.model_combo.setCurrentText("高阶 Kerr")
        self.assertEqual(self.window.canvas.space.model, "高阶 Kerr")
        self.assertFalse(self.window.model_kappa_slider.isHidden())

    def test_time_mode_steps_and_updates_the_spectrum(self):
        self.switch("时间孤子")
        steps = self.window.time_steps_slider.value()
        self.window.tick()
        self.assertEqual(self.window.canvas.time.steps, steps)
        self.assertIn("谱宽", self.window.readout_label.text())

    def test_fundamental_soliton_keeps_its_width(self):
        self.switch("时间孤子")
        self.window.order_slider.setValue(100)
        before = self.window.canvas.time.width()
        for _ in range(40):
            self.window.tick()
        self.assertAlmostEqual(self.window.canvas.time.width(), before,
                               delta=0.05 * before)

    def test_doubling_mode_is_static(self):
        self.switch("调频")
        self.window.tick()
        self.assertEqual(self.window.canvas.doubling.length, 2.0)
        self.assertIn("峰值效率", self.window.readout_label.text())

    def test_symmetry_switch_zeroes_the_conversion(self):
        self.switch("调频")
        self.assertGreater(self.window.canvas.doubling.peak_efficiency(), 0.0)
        self.window.symmetric_check.setChecked(True)
        self.assertEqual(self.window.canvas.doubling.peak_efficiency(), 0.0)

    def test_qpm_toggle_enables_the_period_and_changes_the_map(self):
        self.switch("调频")
        before = self.window.canvas.doubling.efficiency().copy()
        self.assertFalse(self.window.poling_slider.isEnabled())
        self.window.qpm_check.setChecked(True)
        self.assertTrue(self.window.poling_slider.isEnabled())
        self.assertFalse((self.window.canvas.doubling.efficiency() == before).all())

    def test_response_tabs_hold_both_microscopic_views(self):
        self.assertEqual(self.window.response_tabs.count(), 2)
        self.window.tick()
        field = self.window.oscillator_plot.field
        self.assertGreater(field, 0.0)
        self.assertAlmostEqual(field,
                               float(np.sqrt(self.window.canvas.space.peak())),
                               places=6)

    def test_momentum_sketch_tracks_the_qpm_switch(self):
        self.switch("调频")
        self.assertFalse(self.window.momentum_sketch.show_qpm)
        self.window.qpm_check.setChecked(True)
        self.assertTrue(self.window.momentum_sketch.show_qpm)
        self.assertGreater(self.window.momentum_sketch.period, 0.0)

    def test_process_switch_changes_the_model_and_the_readout(self):
        self.switch("调频")
        self.assertEqual(self.window.process_combo.currentText(), "倍频")
        self.window.process_combo.setCurrentText("和频")
        self.assertEqual(self.window.canvas.process, "和频")
        self.assertFalse(self.window.qpm_check.isEnabled())
        self.assertIn("n₃", self.window.readout_label.text())
        self.window.process_combo.setCurrentText("调频")
        self.assertTrue(self.window.qpm_check.isEnabled())

    def test_each_mode_states_its_equation(self):
        self.assertIn("∇²ψ", self.window.space_label.text())
        self.assertIn("β₂", self.window.time_label.text())
        self.assertIn("Δk", self.window.double_label.text())
        for label in (self.window.space_label, self.window.time_label,
                      self.window.double_label):
            self.assertTrue(label.toolTip())


if __name__ == "__main__":
    unittest.main()