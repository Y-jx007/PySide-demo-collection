# -*- coding: utf-8 -*-
"""Logic tests for the KAM standard-map model (no Qt needed)."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.kam_standard_map import (  # noqa: E402
    cyclic_lut,
    frequency_image,
    frequency_map,
    standard_map,
    trace_orbit,
)


class TestStandardMap(unittest.TestCase):
    def test_zero_kick_is_shear(self):
        q, p = 0.3, 0.4
        q2, p2 = standard_map(q, p, 0.0)
        self.assertAlmostEqual(p2, p)
        self.assertAlmostEqual(q2, (q + p) % 1.0)

    def test_kick_changes_momentum(self):
        q, p = 0.25, 0.1
        _, p2 = standard_map(q, p, 2.0)
        self.assertAlmostEqual(p2, (p + 2.0 / (2.0 * np.pi)) % 1.0)

    def test_arrays_and_wrapping(self):
        q = np.array([0.0, 0.25, 0.5])
        p = np.array([0.1, 0.2, 0.3])
        for _ in range(30):
            q, p = standard_map(q, p, 1.7)
        self.assertTrue(np.all(q >= 0.0) and np.all(q < 1.0))
        self.assertTrue(np.all(p >= 0.0) and np.all(p < 1.0))


class TestFrequencyMap(unittest.TestCase):
    def test_shapes_and_ranges(self):
        nu, lyap = frequency_map(0.97, 24, 40, transient=10)
        self.assertEqual(nu.shape, (24, 24))
        self.assertEqual(lyap.shape, (24, 24))
        self.assertTrue(np.all(np.isfinite(nu)))
        self.assertTrue(np.all(np.isfinite(lyap)))
        self.assertGreaterEqual(float(nu.min()), 0.0)
        self.assertLess(float(nu.max()), 1.0)
        self.assertGreater(float(lyap.max()), 0.01)

    def test_zero_kick_rotation_is_momentum(self):
        res = 16
        nu, lyap = frequency_map(0.0, res, 32, transient=8)
        expected = 1.0 - (np.arange(res) + 0.5) / res
        for row in range(res):
            self.assertAlmostEqual(float(nu[row, 0]), float(expected[row]), places=10)
            self.assertTrue(np.allclose(nu[row], nu[row, 0], atol=1e-12))
        self.assertLess(float(np.max(np.abs(lyap))), 1e-10)

    def test_stop_callback(self):
        nu, lyap = frequency_map(
            1.0, 16, 30, transient=5, should_stop=lambda: True
        )
        self.assertIsNone(nu)
        self.assertIsNone(lyap)


class TestOrbitAndImage(unittest.TestCase):
    def test_trace_orbit(self):
        orbit, nu, lyap = trace_orbit(0.31, 0.42, 0.97, steps=240)
        self.assertEqual(orbit.shape, (241, 2))
        self.assertTrue(np.all(np.isfinite(orbit)))
        self.assertTrue(np.all(orbit >= 0.0) and np.all(orbit < 1.0))
        self.assertGreaterEqual(nu, 0.0)
        self.assertLess(nu, 1.0)
        self.assertTrue(np.isfinite(lyap))

    def test_regular_and_chaotic_orbits(self):
        _, nu_regular, lyap_regular = trace_orbit(0.5, 0.55, 0.97, steps=1200)
        _, nu_chaotic, lyap_chaotic = trace_orbit(0.1, 0.9, 0.97, steps=1200)
        self.assertLess(abs(nu_regular - 0.5), 0.01)
        self.assertLess(lyap_regular, 0.01)
        self.assertGreater(lyap_chaotic, 0.05)

    def test_frequency_image(self):
        nu, lyap = frequency_map(0.97, 20, 36, transient=8)
        rgb = frequency_image(nu, lyap)
        self.assertEqual(rgb.shape, (20, 20, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue(np.all(rgb >= 0) and np.all(rgb <= 255))
        self.assertGreater(int(rgb.max()), 80)

    def test_lut_is_cyclic(self):
        lut = cyclic_lut(64)
        self.assertEqual(lut.shape, (64, 3))
        self.assertEqual(lut.dtype, np.uint8)
        self.assertTrue(np.array_equal(lut[0], lut[-1]))


if __name__ == "__main__":
    unittest.main()
