"""Checks for the K_I/K_IC desiccation-crack opening model."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "apps"))
sys.path.insert(0, os.path.join(ROOT, "apps", "Physics"))

from Erosion import ErosionSim  # noqa: E402


class TestFractureOpening(unittest.TestCase):
    def test_dry_soil_opens_cracks_and_wet_soil_closes_them(self):
        sim = ErosionSim(96, seed=4)
        sim.soil.theta1[:] = sim.soil.theta_wp + 0.08 * (sim.soil.theta_fc - sim.soil.theta_wp)
        for _ in range(120):
            sim._update_cracks()
        dry = float(sim.crack.mean())
        self.assertGreater(dry, 0.0)
        sim.soil.theta1[:] = sim.soil.theta_fc
        sim._update_cracks()
        # 吸水膨胀有滞后，不能一帧关死。
        self.assertGreater(float(sim.crack.mean()), 0.0)
        self.assertLess(float(sim.crack.mean()), dry)
        for _ in range(220):
            sim._update_cracks()
        self.assertLess(float(sim.crack.mean()), dry * 0.2)

    def test_new_dry_cycle_can_choose_new_crack_pattern(self):
        sim = ErosionSim(96, seed=6)
        before = sim.crack_edge.copy()
        sim.new_crack_pattern()
        self.assertFalse(np.array_equal(before, sim.crack_edge))

    def test_high_toughness_suppresses_cracking(self):
        sim = ErosionSim(96, seed=5)
        sim.soil.theta1[:] = sim.soil.theta_wp + 0.08 * (sim.soil.theta_fc - sim.soil.theta_wp)
        sim.fracture_kic = 1.0e9
        sim._update_cracks()
        self.assertLess(float(sim.crack.mean()), 1e-6)


if __name__ == "__main__":
    unittest.main()