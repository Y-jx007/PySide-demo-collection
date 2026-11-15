# -*- coding: utf-8 -*-
"""Physics, visual-data and UI checks for the Lensing app."""
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.ui_style import check_app_ui  # noqa: E402
from support.physics import lensing as L  # noqa: E402


class TestSchwarzschildGeometry(unittest.TestCase):
    def test_photon_sphere_and_critical_impact_parameter(self):
        self.assertAlmostEqual(L.R_PHOTON, 3.0, places=12)
        self.assertAlmostEqual(L.R_ISCO, 6.0, places=12)
        self.assertAlmostEqual(L.B_CRIT, 3.0 * math.sqrt(3.0), places=12)
        self.assertAlmostEqual(L.critical_impact_parameter(), L.B_CRIT, places=12)

    def test_shadow_angle_is_consistent_with_impact_parameter(self):
        for distance in (12.0, 36.0, 100.0, 1000.0):
            angle = L.shadow_angular_radius(distance)
            f = 1.0 - 2.0 / distance
            recovered = distance * math.sin(angle) / math.sqrt(f)
            self.assertAlmostEqual(recovered, L.B_CRIT, places=8)

    def test_camera_basis_is_orthonormal(self):
        for inc in (0.0, 18.0, 65.0, 84.0):
            frame = L.CameraFrame(36.0, inc, 17.0, 42.0)
            basis = (frame.forward, frame.right, frame.up)
            for vec in basis:
                self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=10)
            for i in range(3):
                for j in range(i + 1, 3):
                    self.assertAlmostEqual(float(np.dot(basis[i], basis[j])), 0.0,
                                           places=10)
            center = frame.ray(0.0, 0.0)
            self.assertAlmostEqual(float(np.linalg.norm(center)), 1.0, places=10)

    def test_look_around_keeps_camera_fixed_and_follows_drag(self):
        base = L.CameraFrame(46.0, 84.0, 0.0, 32.0)
        view = L.CameraFrame(46.0, 84.0, 0.0, 32.0,
                             view_yaw=10.0, view_pitch=5.0)
        # The observer position is unchanged; only the look direction moves.
        np.testing.assert_allclose(view.origin, base.origin, atol=1e-12)
        self.assertGreater(float(np.linalg.norm(view.forward - base.forward)), 0.01)
        for vec in (view.forward, view.right, view.up):
            self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=10)
        self.assertAlmostEqual(float(np.dot(view.forward, view.right)), 0.0,
                               places=10)
        self.assertAlmostEqual(float(np.dot(view.forward, view.up)), 0.0,
                               places=10)

        # A star near the centre moves right for +yaw and down for +pitch:
        # this is the direct-manipulation direction used by the mouse.
        star = L._unit(base.forward + 0.06 * base.right + 0.04 * base.up)

        def project(frame, direction):
            return (
                float(np.dot(direction, frame.right) / np.dot(direction, frame.forward)),
                float(np.dot(direction, frame.up) / np.dot(direction, frame.forward)),
            )

        x0, y0 = project(base, star)
        x1, y1 = project(view, star)
        self.assertGreater(x1, x0)
        self.assertLess(y1, y0)

    def test_initial_velocity_has_impact_parameter_b(self):
        frame = L.CameraFrame(36.0, 78.0, 0.0, 42.0)
        for x, y in ((0.0, 0.0), (0.4, -0.2), (-0.6, 0.5), (1.0, 0.0)):
            n = frame.ray(x, y)
            v = L.initial_velocity(frame.origin, n)
            hp = np.cross(frame.origin, v)
            b = float(np.linalg.norm(hp))
            nr = float(np.dot(n, frame.origin / frame.distance))
            expected = frame.distance * math.sqrt(max(0.0, 1.0 - nr * nr)) / math.sqrt(
                1.0 - 2.0 / frame.distance
            )
            self.assertLess(abs(b - expected), 1e-6)


class TestGeodesicReference(unittest.TestCase):
    def test_weak_field_deflection_matches_post_newtonian_series(self):
        b = 100.0
        got = L.deflection_angle(b, r0=5000.0, max_steps=4000)
        expected = 4.0 / b + 15.0 * math.pi / (4.0 * b * b)
        self.assertLess(abs(got - expected) / expected, 0.005)

    def test_strong_field_deflection_close_to_critical(self):
        # Reference value from scipy.integrate.solve_ivp on the Binet ODE
        # u'' + u = 3u^2 with u(0)=1/5000, u'(0)=sqrt(1/b^2-u0^2+2u0^3).
        got = math.degrees(L.deflection_angle(6.0, r0=5000.0, max_steps=4000))
        self.assertAlmostEqual(got, 98.5138, places=2)

    def test_capture_threshold(self):
        def status(b):
            f = 1.0 - 2.0 / 5000.0
            sinpsi = min(1.0, b * math.sqrt(f) / 5000.0)
            cospsi = math.sqrt(1.0 - sinpsi * sinpsi)
            origin = np.array([5000.0, 0.0, 0.0])
            n = np.array([-cospsi, sinpsi, 0.0])
            return L.trace_ray(origin, n, max_steps=5000)["status"]

        self.assertEqual(status(5.0), "captured")
        self.assertEqual(status(5.3), "escaped")
        self.assertEqual(status(5.2), "escaped")

    def test_binet_equation_has_a_fixed_point_at_the_photon_sphere(self):
        u_photon = 1.0 / L.R_PHOTON
        self.assertAlmostEqual(L.binet_curvature(u_photon), 0.0, places=12)
        self.assertGreater(L.binet_curvature(2.0 * u_photon), 0.0)
        self.assertLess(L.binet_curvature(0.5 * u_photon), 0.0)

    def test_photon_sphere_circular_orbit_has_no_radial_acceleration(self):
        # At r = 3M, a tangent photon with b = 3 sqrt(3) M has r'' = 0.
        r = np.array([3.0, 0.0, 0.0])
        v = np.array([0.0, math.sqrt(3.0), 0.0])
        h2 = float(np.dot(np.cross(r, v), np.cross(r, v)))
        self.assertAlmostEqual(h2, 27.0, places=12)
        accel = L._accel(r, h2)
        # Cartesian acceleration minus the centripetal part is zero; the
        # curvature of the circular path itself has |v|^2/r = 1.
        self.assertAlmostEqual(float(np.linalg.norm(accel)), 1.0, places=12)


class TestDiskPhysics(unittest.TestCase):
    def test_zero_torque_profile_peaks_at_49_over_36_rin(self):
        r_in = 6.0
        peak_r = L.disk_profile_peak_radius(r_in)
        self.assertAlmostEqual(peak_r, 49.0 / 6.0, places=12)
        self.assertLess(abs(float(L.profile_value(r_in, r_in))), 1e-12)
        self.assertAlmostEqual(float(L.profile_value(peak_r, r_in)), 1.0, places=10)
        self.assertGreater(float(L.profile_value(10.0, r_in)), 0.0)
        self.assertGreater(float(L.profile_value(peak_r, r_in)),
                           float(L.profile_value(20.0, r_in)))

    def test_doppler_beaming_is_antisymmetric_in_bz(self):
        approaching = L.doppler_factor(6.0, 5.0, 46.0)
        receding = L.doppler_factor(6.0, -5.0, 46.0)
        self.assertGreater(approaching, receding)
        self.assertGreater(approaching, 1.0)
        self.assertLess(receding, 1.0)
        # At b_z = 0 only the gravitational part remains.
        self.assertLess(L.doppler_factor(10.0, 0.0, 46.0), 1.0)

    def test_blackbody_colours_move_from_red_to_blue(self):
        cool = L.blackbody_rgb(3000.0)
        hot = L.blackbody_rgb(15000.0)
        self.assertGreater(cool[0], cool[2])
        self.assertGreater(hot[2], hot[0])
        lut = L.blackbody_lut(64)
        self.assertEqual(lut.shape, (64, 3))
        self.assertTrue(np.isfinite(lut).all())
        self.assertTrue((lut >= 0.0).all() and (lut <= 1.0 + 1e-6).all())

    def test_disk_temperature_on_the_peak(self):
        t = L.disk_temperature(L.disk_profile_peak_radius(6.0), 6.0, 9000.0)
        self.assertAlmostEqual(float(t), 9000.0, places=6)
        self.assertEqual(float(L.disk_temperature(5.9, 6.0, 9000.0)), 0.0)


class TestSkyAndTextures(unittest.TestCase):
    def test_starfield_is_deterministic_and_finite(self):
        a = L.build_starfield(256, 128, seed=7)
        b = L.build_starfield(256, 128, seed=7)
        self.assertEqual(a.shape, (128, 256, 3))
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.isfinite(a).all())
        self.assertGreater(float(a.max()), 0.0)
        self.assertLess(float(a.mean()), 1.0)

    def test_starfield_different_seed_differs(self):
        a = L.build_starfield(128, 64, seed=7)
        b = L.build_starfield(128, 64, seed=8)
        self.assertFalse(np.array_equal(a, b))

    def test_disk_texture_is_periodic_and_bounded(self):
        tex = L.build_disk_texture(128, 64, seed=17)
        self.assertEqual(tex.shape, (64, 128))
        self.assertTrue(np.isfinite(tex).all())
        self.assertGreaterEqual(float(tex.min()), 0.0)
        self.assertLessEqual(float(tex.max()), 1.0)
        self.assertGreater(float(tex.std()), 0.02)


class TestRendererConfiguration(unittest.TestCase):
    def test_render_config_defaults_are_valid(self):
        cfg = L.RenderConfig()
        self.assertGreater(cfg.width, 0)
        self.assertGreater(cfg.r_out, cfg.r_in)
        self.assertGreaterEqual(cfg.opacity, 0.0)
        self.assertLessEqual(cfg.opacity, 1.0)
        self.assertGreater(cfg.max_steps, 8)

    @unittest.skipUnless(L.taichi_available(), "Taichi backend unavailable")
    def test_taichi_kernel_renders_a_frame(self):
        renderer = L.LensingRenderer(star_size=(512, 256))
        cfg = L.RenderConfig(width=48, height=48, distance=36.0,
                             inclination=72.0, fov=38.0, disk_on=False,
                             max_steps=180, exposure=1.0, star_gain=1.0)
        image = renderer.render(cfg)
        self.assertEqual(image.shape, (48, 48, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(int(image.max()), 0)
        self.assertTrue(np.isfinite(image).all())


class TestLensingUI(unittest.TestCase):
    def test_app_follows_the_shared_ui_contract(self):
        path = os.path.join(ROOT, "apps", "Physics", "Lensing.py")
        issues = check_app_ui(path)
        self.assertEqual(issues, [], issues)

    def test_app_is_in_the_manifest_and_layout(self):
        import apps_manifest
        entries = {app["path"]: app for app in apps_manifest.APPS}
        self.assertIn("Physics/Lensing.py", entries)
        self.assertEqual(entries["Physics/Lensing.py"]["category"], "Physics")
        layout = os.path.join(ROOT, "runtime", "launcher_layout.json")
        if os.path.exists(layout):
            import json
            with open(layout, encoding="utf-8") as handle:
                data = json.load(handle)
            paths = {p for cat in data["categories"] for p in cat["apps"]}
            self.assertIn("Physics/Lensing.py", paths)


if __name__ == "__main__":
    unittest.main()
