# -*- coding: utf-8 -*-
"""MPM2D solver guardrails: capacity and render shape."""
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.mpm2d import (  # noqa: E402
    MATERIAL_ELASTIC, MPM2D, RENDER_RES,
)


class TestMPM2D(unittest.TestCase):
    def test_capacity_is_clamped(self):
        sim = MPM2D(res=32, density=4, max_particles=40)
        sim.add_cube((0.1, 0.1), (0.8, 0.8), MATERIAL_ELASTIC)
        self.assertLessEqual(int(sim.n_particles[None]), 40)
        sim.step(1.0 / 240.0)
        frame = sim.draw()
        self.assertEqual(frame.shape, (RENDER_RES, RENDER_RES, 3))


if __name__ == "__main__":
    unittest.main()
