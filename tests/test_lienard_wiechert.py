"""Tests for the retarded Liénard-Wiechert field model."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "apps", "Physics"))

from lienard_wiechert import C, Trajectory  # noqa: E402


class TestLienardWiechert(unittest.TestCase):
    def test_circular_state(self):
        tr = Trajectory(beta0=0.6, modulation=0.1, drive=0.7, radius=1.0)
        st = tr.state(2.0)
        pos = st["pos"]
        self.assertAlmostEqual(float(np.linalg.norm(pos)), 1.0, places=6)
        # velocity is tangent to the circle
        self.assertAlmostEqual(float(np.dot(pos, st["vel"])), 0.0, places=6)

    def test_retarded_light_cone(self):
        tr = Trajectory(beta0=0.6, modulation=0.1, drive=0.7, radius=0.72)
        obs = np.array([[1.2, 0.2], [-1.0, 0.8], [0.1, -1.3]])
        r = tr.retarded_state(obs, 3.0, iterations=12)
        pos = tr.state(r["tr"])["pos"]
        dist = np.linalg.norm(obs - pos, axis=1)
        np.testing.assert_allclose(r["tr"] + dist / C, 3.0, atol=1e-3)

    def test_radiation_components_nonnegative(self):
        tr = Trajectory()
        obs = np.array([[1.2, 0.3], [0.0, 1.4], [-1.1, -0.6]])
        f = tr.field(obs, 1.0)
        self.assertTrue(np.all(np.isfinite(f["i_par"])))
        self.assertTrue(np.all(np.isfinite(f["i_perp"])))
        self.assertTrue(np.all(f["i_par"] >= 0))
        self.assertTrue(np.all(f["i_perp"] >= 0))

    def test_plane_components_share_polarization(self):
        # In the orbital plane the radiation E has only one in-plane
        # polarization. Parallel and perpendicular acceleration therefore
        # produce collinear E vectors; their field lines coincide physically.
        tr = Trajectory(beta0=0.65, modulation=0.2, drive=0.7, radius=0.72)
        axis = np.linspace(-2.0, 2.0, 17)
        xx, yy = np.meshgrid(axis, axis)
        f = tr.field(np.stack([xx, yy], axis=-1), 2.0)
        a, b = f["e_par"], f["e_perp"]
        ma = np.linalg.norm(a, axis=-1)
        mb = np.linalg.norm(b, axis=-1)
        mask = (ma > 1e-9) & (mb > 1e-9)
        cross = np.abs(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])
        self.assertLess(float(np.max(cross[mask] / (ma[mask] * mb[mask]))), 1e-7)

    def test_far_field_angles(self):
        tr = Trajectory()
        f = tr.far_field(1.0, angles=16)
        self.assertEqual(f["i_par"].shape, (16,))
        self.assertEqual(f["i_perp"].shape, (16,))
        self.assertEqual(f["e_par"].shape, (16, 2))


if __name__ == "__main__":
    unittest.main()
