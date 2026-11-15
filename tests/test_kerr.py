# -*- coding: utf-8 -*-
"""Kerr geometry, ray tracer and Kerr-app checks."""
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics import kerr as K  # noqa: E402


class TestKerrGeometry(unittest.TestCase):
    def test_horizon(self):
        self.assertAlmostEqual(K.kerr_horizon(0.0), 2.0, places=12)
        self.assertAlmostEqual(K.kerr_horizon(0.9), 1.0 + math.sqrt(0.19),
                               places=12)
        self.assertAlmostEqual(K.kerr_horizon(0.99), 1.0 + math.sqrt(0.0199),
                               places=12)

    def test_isco(self):
        self.assertAlmostEqual(K.kerr_isco(0.0), 6.0, places=10)
        self.assertAlmostEqual(K.kerr_isco(0.9), 2.32088304, places=6)
        self.assertAlmostEqual(K.kerr_isco(0.99), 1.4545, places=3)

    def test_metric_at_horizon_has_zero_delta(self):
        for a in (0.0, 0.5, 0.9):
            sigma, delta, gtt, gtphi, gphiphi, grr, gthth = K.kerr_metric(
                K.kerr_horizon(a), 1.1, a
            )
            self.assertAlmostEqual(delta, 0.0, places=10)
            self.assertGreater(sigma, 0.0)
            self.assertAlmostEqual(gtt + 1.0 - 2.0 * K.kerr_horizon(a) / sigma,
                                   0.0, places=10)

    def test_disk_orbit_is_keplerian(self):
        for a in (0.0, 0.5, 0.9):
            for r in (6.0, 10.0):
                omega = K.kerr_disk_omega(r, a)
                self.assertAlmostEqual(omega, 1.0 / (r ** 1.5 + a), places=12)
                self.assertGreater(K.kerr_disk_ut(r, a), 0.0)


class TestKerrInitialConditions(unittest.TestCase):
    def test_invariants_match_initial_velocities(self):
        for a in (0.0, 0.6, 0.9):
            st = K._camera_initial(46.0, math.radians(84.0), 0.0, a,
                                   0.3, -0.2, 32.0)
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
            self.assertGreater(st["E"], 0.0)

    def test_prograde_disk_redshift_is_finite(self):
        st = K._camera_initial(46.0, math.radians(84.0), 0.0, 0.8,
                               0.1, 0.0, 32.0)
        g = K.kerr_redshift(6.0, st["E_ph"], st["L_ph"], 0.8)
        self.assertGreater(g, 0.0)
        self.assertLess(g, 10.0)


class TestKerrRayTracer(unittest.TestCase):
    def test_spin_changes_the_capture_cross_section(self):
        # At this image point the a=0 shadow captures the ray, while a
        # rapidly spinning Kerr hole lets it escape on the prograde side.
        x = 0.25
        slow = K.trace_kerr(distance=46, inclination=84, a=0.0, x=x, y=0,
                            max_steps=1200, step_scale=0.02)
        fast = K.trace_kerr(distance=46, inclination=84, a=0.95, x=x, y=0,
                            max_steps=1200, step_scale=0.02)
        self.assertEqual(slow["status"], "captured")
        self.assertEqual(fast["status"], "escaped")

    def test_central_ray_is_captured(self):
        rec = K.trace_kerr(distance=46, inclination=84, a=0.8, x=0.0, y=0.0,
                           max_steps=1200, step_scale=0.02)
        self.assertEqual(rec["status"], "captured")

    @unittest.skipUnless(K.taichi_available(), "Taichi backend unavailable")
    def test_kerr_kernel_renders_a_frame(self):
        renderer = K.KerrRenderer(star_size=(512, 256), output_size=840)
        cfg = K.KerrConfig(width=48, height=48, spin=0.85,
                           disk_on=False, max_steps=500, step_scale=0.02)
        image = renderer.render(cfg)
        self.assertEqual(image.shape, (48, 48, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(int(image.max()), 0)


class TestKerrApp(unittest.TestCase):
    def test_app_uses_kerr_and_exposes_spin(self):
        path = os.path.join(ROOT, "apps", "Physics", "Lensing.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("KerrConfig", source)
        self.assertIn("KerrRenderer", source)
        self.assertIn("kerr_isco", source)
        self.assertIn("自旋 a", source)
        self.assertIn("spin_slider", source)

    def test_cover_and_manifest(self):
        import apps_manifest
        entry = next(a for a in apps_manifest.APPS
                     if a["path"] == "Physics/Lensing.py")
        self.assertIn("Kerr", entry["title"])
        self.assertIn("Kerr", entry["description"])
        cover = os.path.join(ROOT, "assets", "thumbnails", "Lensing.png")
        self.assertTrue(os.path.exists(cover))


if __name__ == "__main__":
    unittest.main()
