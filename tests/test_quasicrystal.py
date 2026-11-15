# -*- coding: utf-8 -*-
"""Real-space and diffraction quasicrystal tests."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.quasicrystal import (  # noqa: E402
    Quasicrystal,
    apply_palette,
)


class TestQuasicrystal(unittest.TestCase):
    def test_real_space_shape(self):
        sim = Quasicrystal(128)
        field = sim.real_space()
        self.assertEqual(field.shape, (128, 128))
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertGreater(float(field.max() - field.min()), 0.5)

    def test_fold_changes_pattern(self):
        first = Quasicrystal(128)
        first.n = 5
        second = Quasicrystal(128)
        second.n = 7
        self.assertFalse(np.allclose(first.real_space(), second.real_space()))

    def test_diffraction_crop(self):
        sim = Quasicrystal(128)
        sim.diffraction_zoom = 4.0
        spectrum = sim.diffraction(log=True)
        self.assertEqual(spectrum.shape, (32, 32))
        self.assertAlmostEqual(float(spectrum.max()), 1.0, places=5)
        self.assertTrue(np.all(spectrum >= 0.0))

    def test_palette(self):
        values = np.linspace(0.0, 1.0, 16).reshape(4, 4)
        rgb = apply_palette(values, "冰蓝")
        self.assertEqual(rgb.shape, (4, 4, 3))
        self.assertEqual(rgb.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
