# -*- coding: utf-8 -*-
"""Particle Lenia reference-kernel tests."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.particle_lenia import (  # noqa: E402
    WORLD_HALF,
    ParticleLenia,
    calc_norm_coef,
)


class TestParticleLenia(unittest.TestCase):
    def test_normalization_positive(self):
        self.assertGreater(calc_norm_coef(1.27, 1.0), 0.0)
        self.assertLess(calc_norm_coef(1.27, 1.0), 1.0)

    def test_reset_count(self):
        sim = ParticleLenia()
        sim.reset("环", 300)
        self.assertEqual(sim.count(), 300)
        self.assertEqual(sim.positions().shape, (300, 2))

    def test_step_is_finite(self):
        sim = ParticleLenia()
        sim.reset("圆盘", 280)
        for _ in range(5):
            sim.step()
        positions = sim.positions()
        self.assertTrue(np.all(np.isfinite(positions)))
        self.assertTrue(np.all(np.abs(positions) <= WORLD_HALF + 1e-4))

    def test_add_particle(self):
        sim = ParticleLenia()
        sim.reset("圆盘", 50)
        before = sim.count()
        sim.add_particle(8.0, 3.0)
        self.assertEqual(sim.count(), before + 1)
        np.testing.assert_allclose(sim.positions()[-1], [8.0, 3.0], atol=1e-5)

    def test_square_boundary(self):
        sim = ParticleLenia()
        sim.reset("随机", 160)
        for _ in range(5):
            sim.step()
        positions = sim.positions()
        self.assertLessEqual(
            float(np.max(np.abs(positions))), WORLD_HALF + 1e-4
        )


if __name__ == "__main__":
    unittest.main()
