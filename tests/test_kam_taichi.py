# -*- coding: utf-8 -*-
"""Smoke tests for the Taichi KAM backend (skips without Taichi)."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.kam_taichi import (  # noqa: E402
    frequency_map_taichi,
    taichi_available,
    taichi_backend_name,
)


@unittest.skipUnless(taichi_available(), "Taichi backend unavailable")
class TestKAMTaichi(unittest.TestCase):
    def test_small_map(self):
        nu, lyap = frequency_map_taichi(0.97, 16, 20, transient=5)
        self.assertIsNotNone(nu)
        self.assertEqual(nu.shape, (16, 16))
        self.assertEqual(lyap.shape, (16, 16))
        self.assertTrue(np.all(np.isfinite(nu)))
        self.assertTrue(np.all(np.isfinite(lyap)))
        self.assertGreaterEqual(float(nu.min()), 0.0)
        self.assertLess(float(nu.max()), 1.0)
        self.assertGreater(float(lyap.max()), 0.01)
        self.assertTrue(taichi_backend_name().startswith("Taichi"))

    def test_zero_kick(self):
        res = 16
        nu, lyap = frequency_map_taichi(0.0, res, 24, transient=5)
        expected = 1.0 - (np.arange(res) + 0.5) / res
        for row in range(res):
            self.assertAlmostEqual(float(nu[row, 0]), float(expected[row]), places=5)
            self.assertTrue(np.allclose(nu[row], nu[row, 0], atol=1e-5))
        self.assertLess(float(np.max(np.abs(lyap))), 1e-5)


if __name__ == "__main__":
    unittest.main()
