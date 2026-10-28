"""Tests for the two-layer soil water model used by Erosion."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from support.physics.soil_water import SoilWater  # noqa: E402


class TestSoilWater(unittest.TestCase):
    def _soil(self, fines=0.5, clay=0.30):
        return SoilWater(np.full((20, 20), fines, np.float32), clay)

    def test_infiltration_and_mass_balance(self):
        soil = self._soil()
        before = soil.storage_mm
        soil.step(20.0, 4.0, dt_min=10.0)
        self.assertGreater(soil.storage_mm, before)
        self.assertLess(abs(soil.balance_error), 0.05)

    def test_drying_reduces_water_and_opens_cracks(self):
        soil = self._soil()
        wet = soil.storage_mm
        for _ in range(700):
            soil.step(0.0, 6.0, dt_min=10.0)
        self.assertLess(soil.storage_mm, wet - 10.0)
        self.assertGreater(float(soil.shrinkage.mean()), 0.1)

    def test_rewetting_closes_cracks(self):
        soil = self._soil()
        for _ in range(700):
            soil.step(0.0, 6.0, dt_min=10.0)
        self.assertGreater(float(soil.shrinkage.mean()), 0.1)
        for _ in range(80):
            soil.step(40.0, 6.0, dt_min=10.0)
        self.assertLess(float(soil.shrinkage.mean()), 0.05)

    def test_sand_infiltrates_faster_than_clay(self):
        sand = self._soil(fines=0.2, clay=0.10)
        clay = self._soil(fines=0.8, clay=0.60)
        self.assertGreater(float(sand.ks.mean()), float(clay.ks.mean()) * 3.0)


if __name__ == "__main__":
    unittest.main()
