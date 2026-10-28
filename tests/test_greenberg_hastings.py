"""Logic tests for the Greenberg-Hastings excitable-media CA."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from support.cellular_automata.greenberg_hastings_model import GreenbergHastings  # noqa: E402


class TestGreenbergHastings(unittest.TestCase):
    def test_excitation_and_refractory_cycle(self):
        sim = GreenbergHastings(7, refractory=3, threshold=1, neighbors=4)
        sim.state.fill(0)
        sim.state[3, 3] = 1
        sim.step()
        self.assertEqual(int(sim.state[3, 3]), 2)
        self.assertEqual(int(sim.state[3, 4]), 1)
        self.assertEqual(int(sim.state[3, 2]), 1)

        sim.state.fill(0)
        sim.state[3, 3] = 1
        sim.step()
        self.assertEqual(int(sim.state[3, 3]), 2)
        sim.step()
        self.assertEqual(int(sim.state[3, 3]), 3)
        sim.step()
        self.assertEqual(int(sim.state[3, 3]), 0)

    def test_threshold(self):
        sim = GreenbergHastings(7, refractory=3, threshold=2, neighbors=4)
        sim.state.fill(0)
        sim.state[3, 3] = 1
        sim.step()
        self.assertEqual(int(sim.state[3, 4]), 0)

    def test_wrap_neighbour(self):
        sim = GreenbergHastings(7, refractory=3, threshold=1, neighbors=4)
        sim.state.fill(0)
        sim.state[0, 0] = 1
        sim.step()
        self.assertEqual(int(sim.state[6, 0]), 1)

    def test_spiral_and_target_seeds(self):
        spiral = GreenbergHastings(64, refractory=5, seed=1)
        spiral.reset_spiral()
        self.assertGreater(int(np.count_nonzero(spiral.state == 1)), 0)
        for _ in range(20):
            spiral.step()
        self.assertGreater(spiral.metrics()["refractory"], 0)

        target = GreenbergHastings(64, refractory=5, seed=1)
        target.reset_target()
        self.assertTrue(target.pacemaker)
        for _ in range(target.pacemaker_interval + 1):
            target.step()
        self.assertGreater(int(np.count_nonzero(target.state == 1)), 0)


if __name__ == "__main__":
    unittest.main()
