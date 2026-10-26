"""Checks for the Voronoi-polygon cohesive network."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from cohesive_voronoi_desiccation import solve_voronoi  # noqa: E402


class TestVoronoiDesiccation(unittest.TestCase):
    def test_polygon_network_localises(self):
        u, d, coh, nodes = solve_voronoi(0.003, m=10)
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertGreater(len(coh), 100)
        self.assertGreater(float(d.max()), 0.5)
        self.assertLess(float(d.mean()), 0.9)

    def test_more_drying_damages_more(self):
        _, d1, _, _ = solve_voronoi(0.0015, m=10)
        _, d2, _, _ = solve_voronoi(0.0030, m=10)
        self.assertGreater(float(d2.mean()), float(d1.mean()))


if __name__ == "__main__":
    unittest.main()
