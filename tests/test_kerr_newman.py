# -*- coding: utf-8 -*-
"""Kerr-Newman geometry, charged ray tracer and app checks."""
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics import kerr as K  # noqa: E402


class TestKerrNewmanGeometry(unittest.TestCase):
    def test_horizon_charge(self):
        self.assertAlmostEqual(K.kerr_horizon(0.0, 0.5),
                               1.0 + math.sqrt(0.75), places=12)
        self.assertAlmostEqual(K.kerr_horizon(0.8, 0.4),
                               1.0 + math.sqrt(0.2), places=12)
        # A charge outside the sub-extremal domain is clamped, not NaNs.
        self.assertTrue(math.isfinite(K.kerr_horizon(0.99, 0.9)))

    def test_ergosphere_charge(self):
        self.assertAlmostEqual(K.kerr_ergosphere(0.0, q=0.5),
                               1.0 + math.sqrt(0.75), places=12)
        self.assertAlmostEqual(K.kerr_ergosphere(0.8, math.pi / 2.0, 0.4),
                               1.0 + math.sqrt(1.0 - 0.16), places=12)

    def test_metric_delta_and_gtt(self):
        a, q, r = 0.7, 0.4, 8.0
        sigma, delta, gtt, *_ = K.kerr_metric(r, math.pi / 2.0, a, q)
        self.assertAlmostEqual(delta, r * r - 2 * r + a * a + q * q,
                               places=12)
        self.assertAlmostEqual(gtt, -(1.0 - (2 * r - q * q) / sigma),
                               places=12)

    def test_isco_charge_shifts_inward(self):
        self.assertAlmostEqual(K.kerr_isco(0.8, 0.0), 2.9066, places=3)
        self.assertLess(K.kerr_isco(0.8, 0.4), K.kerr_isco(0.8, 0.0))
        self.assertGreater(K.kerr_isco(0.8, 0.4), K.kerr_horizon(0.8, 0.4))
        self.assertAlmostEqual(K.kerr_isco(0.0, 0.5), 5.6067, places=2)

    def test_disk_orbit_q0_and_charged(self):
        self.assertAlmostEqual(K.kerr_disk_omega(6.0, 0.8),
                               1.0 / (6.0 ** 1.5 + 0.8), places=14)
        for q in (0.0, 0.45):
            omega = K.kerr_disk_omega(6.0, 0.75, q)
            ut = K.kerr_disk_ut(6.0, 0.75, q)
            self.assertGreater(omega, 0.0)
            self.assertGreater(ut, 0.0)


class TestKerrNewmanIntegrator(unittest.TestCase):
    def test_initial_invariants_with_charge(self):
        a, q = 0.75, 0.45
        st = K._camera_initial(46.0, math.radians(84.0), 0.0,
                               a, 0.2, -0.1, 32.0, q)
        sigma, delta, *_ = st["cam"]
        L, Q = st["L"], st["Q"]
        H = 46.0 ** 2 + a * a - a * L
        R = H * H - delta * ((L - a) ** 2 + Q)
        theta = math.radians(84.0)
        Theta = Q + a * a * math.cos(theta) ** 2 - L * L * (
            math.cos(theta) ** 2 / math.sin(theta) ** 2
        )
        self.assertAlmostEqual(R, (sigma * st["vr"]) ** 2, places=2)
        self.assertAlmostEqual(Theta, (sigma * st["vth"]) ** 2, places=8)

    def test_charge_changes_capture(self):
        slow = K.trace_kerr(distance=46, inclination=84, a=0.75, q=0.0,
                            x=0.25, y=0.0, max_steps=1200, step_scale=0.02)
        charged = K.trace_kerr(distance=46, inclination=84, a=0.75, q=0.45,
                               x=0.25, y=0.0, max_steps=1200,
                               step_scale=0.02)
        self.assertEqual(slow["status"], "captured")
        self.assertEqual(charged["status"], "escaped")

    def test_redshift_charged_disk_is_finite(self):
        st = K._camera_initial(46.0, math.radians(84.0), 0.0, 0.75,
                               0.1, 0.0, 32.0, 0.45)
        g = K.kerr_redshift(6.0, st["E_ph"], st["L_ph"], 0.75, 0.45)
        self.assertGreater(g, 0.0)
        self.assertLess(g, 10.0)

    @unittest.skipUnless(K.taichi_available(), "Taichi backend unavailable")
    def test_charged_kernel_renders(self):
        renderer = K.KerrRenderer(star_size=(512, 256), output_size=840)
        cfg = K.KerrConfig(width=48, height=48, spin=0.75, charge=0.45,
                           disk_on=False, max_steps=500, step_scale=0.02)
        image = renderer.render(cfg)
        self.assertEqual(image.shape, (48, 48, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(int(image.max()), 0)


class TestKerrNewmanApp(unittest.TestCase):
    def test_app_exposes_spin_and_charge(self):
        path = os.path.join(ROOT, "apps", "Physics", "Lensing.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("KerrConfig", source)
        self.assertIn("charge_slider", source)
        self.assertIn("电荷 Q", source)
        self.assertIn("kerr_horizon(a, q)", source)

    def test_manifest_and_cover(self):
        import apps_manifest
        entry = next(a for a in apps_manifest.APPS
                     if a["path"] == "Physics/Lensing.py")
        self.assertIn("Kerr", entry["title"])
        self.assertIn("Newman", entry["title"])
        self.assertIn("电荷", entry["description"])
        cover = os.path.join(ROOT, "assets", "thumbnails", "Lensing.png")
        self.assertTrue(os.path.exists(cover))


if __name__ == "__main__":
    unittest.main()
