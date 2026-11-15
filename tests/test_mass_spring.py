# -*- coding: utf-8 -*-
"""Mass-spring integration tests."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.mass_spring import MassSpring  # noqa: E402


class TestMassSpring(unittest.TestCase):
    def test_preset_builds_springs(self):
        sim = MassSpring()
        sim.reset("悬索桥")
        count, springs = sim.count()
        self.assertEqual(count, 15)
        self.assertGreater(springs, 10)
        self.assertEqual(sim.fixed_mask().sum(), 2)

    def test_step_is_finite(self):
        sim = MassSpring()
        sim.reset("网")
        for _ in range(3):
            sim.step()
        positions = sim.positions()
        self.assertTrue(np.all(np.isfinite(positions)))
        self.assertTrue(np.all((positions >= 0.0) & (positions <= 1.0)))

    def test_spring_rest_length_is_real(self):
        sim = MassSpring()
        sim.add_particle(0.3, 0.5, connect=False)
        sim.add_particle(0.5, 0.5, connect=False)
        sim._add_spring(0, 1, 0.2)
        _, _, rest = sim.springs()
        self.assertAlmostEqual(float(rest[0]), 0.2, places=5)


if __name__ == "__main__":
    unittest.main()
