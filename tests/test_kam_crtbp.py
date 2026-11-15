# -*- coding: utf-8 -*-
"""Tests for the Sun–Jupiter CRTBP Poincaré backend."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.kam_crtbp import (  # noqa: E402
    X_MAX,
    X_MIN,
    compute_poincare,
    crtbp_derivative,
    poincare_numpy,
)


class TestCRTBP(unittest.TestCase):
    def test_acceleration_is_finite(self):
        ax, ay = crtbp_derivative(0.8, 0.1, 0.02, -0.03)
        self.assertTrue(np.isfinite(ax))
        self.assertTrue(np.isfinite(ay))

    def test_numpy_section(self):
        points = poincare_numpy(3.0, 10, 500)
        self.assertGreater(len(points), 0)
        self.assertTrue(np.all(np.isfinite(points)))
        self.assertTrue(np.all(np.abs(points[:, 0]) < 100.0))
        self.assertTrue(np.all(np.abs(points[:, 1]) < 10.0))

    def test_compute_section(self):
        points = compute_poincare(3.0, 10, 500)
        self.assertGreater(len(points), 0)
        self.assertTrue(np.all(np.isfinite(points)))


if __name__ == "__main__":
    unittest.main()
