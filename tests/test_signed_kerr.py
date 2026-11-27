# -*- coding: utf-8 -*-
"""Negative spin, signed charge, co-rotating disk and the UI interlock."""
import importlib.util
import math
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from support.physics.kerr import (  # noqa: E402
    KN_MAX, KerrConfig, KerrRenderer, _pair, kerr_disk_omega,
    kerr_disk_ut, kerr_horizon, kerr_ibco, kerr_isco, kerr_metric,
    kerr_photon_radius,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestSignedPhysics(unittest.TestCase):
    def test_domain_preserves_signs(self):
        a, q = _pair(-0.8, -0.7)
        self.assertLess(a, 0.0)
        self.assertLess(q, 0.0)
        self.assertLessEqual(a * a + q * q, KN_MAX * KN_MAX + 1e-12)

    def test_charge_sign_does_not_change_metric(self):
        plus = kerr_metric(6.0, math.pi / 2.0, 0.5, 0.3)
        minus = kerr_metric(6.0, math.pi / 2.0, 0.5, -0.3)
        for x, y in zip(plus, minus):
            self.assertAlmostEqual(x, y, places=12)
        self.assertEqual(kerr_horizon(0.5, 0.3), kerr_horizon(0.5, -0.3))

    def test_spin_sign_flips_frame_dragging(self):
        plus = kerr_metric(6.0, math.pi / 2.0, 0.5, 0.3)
        minus = kerr_metric(6.0, math.pi / 2.0, -0.5, 0.3)
        self.assertAlmostEqual(plus[3], -minus[3], places=12)

    def test_disk_co_rotates_with_spin(self):
        self.assertGreater(kerr_disk_omega(6.0, 0.5, 0.0), 0.0)
        self.assertLess(kerr_disk_omega(6.0, -0.5, 0.0), 0.0)
        self.assertAlmostEqual(
            kerr_disk_omega(6.0, -0.5, 0.0),
            -kerr_disk_omega(6.0, 0.5, 0.0), places=12)
        self.assertGreater(kerr_disk_ut(6.0, -0.5, 0.0), 0.0)

    def test_isco_and_photon_ring_are_even(self):
        for a in (0.3, 0.7, 0.9):
            self.assertAlmostEqual(kerr_isco(-a, 0.0),
                                   kerr_isco(a, 0.0), places=10)
            self.assertAlmostEqual(kerr_photon_radius(-a, 0.0),
                                   kerr_photon_radius(a, 0.0), places=10)
        self.assertAlmostEqual(kerr_photon_radius(0.0, 0.0), 3.0, places=9)
        self.assertLess(kerr_photon_radius(0.9, 0.0), 2.0)

    def test_charged_photon_ring_matches_rn_at_zero_spin(self):
        q = 0.5
        expected = 0.5 * (3.0 + math.sqrt(9.0 - 8.0 * q * q))
        self.assertAlmostEqual(kerr_photon_radius(0.0, q), expected, places=6)

    def test_retrograde_orbits_and_ibco(self):
        co = kerr_isco(0.9, 0.0)
        retro = kerr_isco(0.9, 0.0, retrograde=True)
        self.assertGreater(retro, co)
        self.assertGreater(retro, 8.0)
        ph_co = kerr_photon_radius(0.9, 0.0)
        ph_retro = kerr_photon_radius(0.9, 0.0, retrograde=True)
        self.assertGreater(ph_retro, ph_co)
        self.assertGreater(ph_retro, 3.5)
        self.assertAlmostEqual(
            kerr_isco(0.0, 0.0, retrograde=True), 6.0, places=5)
        self.assertAlmostEqual(
            kerr_photon_radius(0.0, 0.0, retrograde=True), 3.0, places=9)
        self.assertAlmostEqual(kerr_ibco(0.0, 0.0), 4.0, delta=0.02)
        self.assertGreater(kerr_ibco(0.9, 0.0),
                           kerr_horizon(0.9, 0.0))
        self.assertLess(kerr_ibco(0.9, 0.0), co)

    def test_negative_spin_charge_renders(self):
        renderer = KerrRenderer(star_size=(64, 32), output_size=48)
        cfg = KerrConfig(
            width=48, height=48, distance=46.0, inclination=84.0, fov=32.0,
            spin=-0.75, charge=-0.45, disk_on=False, max_steps=220,
            step_scale=0.03,
        )
        img = renderer.render(cfg)
        self.assertEqual(img.shape, (48, 48, 3))
        self.assertGreater(int(img.max()), 0)


class TestSignedUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_charge_always_draggable_and_interlocks(self):
        spec = importlib.util.spec_from_file_location(
            "lensing_signed_ui",
            os.path.join(ROOT, "apps", "Unpolished", "Lensing.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        window = module.MainWindow()
        window.show()
        self.app.processEvents()
        old_spin = window.spin_slider.value()
        old_charge = window.charge_slider.value()
        old_rin = window.r_in_slider.value()
        old_disk = window.disk_check.isChecked()
        old_auto = window.auto_orbit_check.isChecked()
        old_mass = window.mass_slider.value()
        old_fov = window.fov_slider.value()
        try:
            self.assertTrue(window.charge_slider.isEnabled())
            self.assertEqual(window.spin_slider.slider.minimum(), -99)
            self.assertEqual(window.charge_slider.slider.minimum(), -99)

            window.charge_slider.setValue(45)
            window.spin_slider.setValue(95)
            self.assertTrue(window.charge_slider.isEnabled())
            a = window.spin_slider.value() / 100.0
            q = window.charge_slider.value() / 100.0
            self.assertLessEqual(a * a + q * q, KN_MAX * KN_MAX + 2e-3)

            window.charge_slider.setValue(90)
            self.assertTrue(window.charge_slider.isEnabled())
            a = window.spin_slider.value() / 100.0
            q = window.charge_slider.value() / 100.0
            self.assertLessEqual(a * a + q * q, KN_MAX * KN_MAX + 2e-3)

            window.spin_slider.setValue(-95)
            window.charge_slider.setValue(-90)
            self.assertLess(window.spin_slider.value(), 0)
            self.assertLess(window.charge_slider.value(), 0)
            self.assertTrue(window.charge_slider.isEnabled())
            self.assertEqual(window.tabs.count(), 2)
            self.assertFalse(window.diagram.grab().toImage().isNull())

            window.d_spin.setValue(-40)
            self.assertEqual(window.spin_slider.value(), -40)
            window.spin_slider.setValue(60)
            self.assertEqual(window.d_spin.value(), 60)
            window.d_rin.setValue(7)
            self.assertEqual(window.r_in_slider.value(), 7)
            self.assertEqual(window.d_rin.value(), 7)
            window.d_disk.setChecked(False)
            self.assertFalse(window.disk_check.isChecked())

            self.assertFalse(hasattr(window, "distance_slider"))
            self.assertEqual(window.fov_slider.slider.minimum(), 5)
            self.assertEqual(window.fov_slider.slider.maximum(), 120)

            window.spin_slider.setValue(-35)
            window.mass_slider.setValue(650)
            window.fov_slider.setValue(90)
            window._time = 3.0
            window._reset()
            self.assertEqual(window._time, 0.0)
            self.assertEqual(window.spin_slider.value(), -35)
            self.assertEqual(window.mass_slider.value(), 650)
            self.assertEqual(window.fov_slider.value(), 90)
            self.assertEqual(window.reset_btn.text(), "重置时间")

            before = window._azimuth
            window.auto_orbit_check.setChecked(True)
            self.assertTrue(window._orbit_timer.isActive())
            window._auto_orbit_tick()
            self.assertNotEqual(window._azimuth, before)
            window.auto_orbit_check.setChecked(False)
            self.assertFalse(window._orbit_timer.isActive())
        finally:
            window.spin_slider.setValue(old_spin)
            window.charge_slider.setValue(old_charge)
            window.r_in_slider.setValue(old_rin)
            window.disk_check.setChecked(old_disk)
            window.auto_orbit_check.setChecked(old_auto)
            window.mass_slider.setValue(old_mass)
            window.fov_slider.setValue(old_fov)
            window.close()


if __name__ == "__main__":
    unittest.main()
