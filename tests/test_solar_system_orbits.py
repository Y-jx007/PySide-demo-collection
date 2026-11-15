# -*- coding: utf-8 -*-
"""Geometry tests for the compact solar-system view."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.solar_system_orbits import (  # noqa: E402
    ASTEROID_GAPS,
    BELT,
    PLANETS,
    PLANET_ORBITS,
    asteroid_positions,
    planet_position,
)


class TestSolarSystemOrbits(unittest.TestCase):
    def test_eight_planets_and_paths(self):
        self.assertEqual(len(PLANETS), 8)
        for planet in PLANETS:
            points = PLANET_ORBITS[planet["name"]]
            self.assertEqual(points.shape, (260, 2))
            self.assertTrue(np.all(np.isfinite(points)))

    def test_orbit_radius_bounds(self):
        for planet in PLANETS:
            points = PLANET_ORBITS[planet["name"]]
            r = np.hypot(points[:, 0], points[:, 1])
            self.assertGreaterEqual(float(r.min()), planet["a"] * (1 - planet["e"]) - 1e-4)
            self.assertLessEqual(float(r.max()), planet["a"] * (1 + planet["e"]) + 1e-4)

    def test_position_is_periodic(self):
        for planet in PLANETS:
            x0, y0 = planet_position(planet, 0.0)
            x1, y1 = planet_position(planet, planet["period"])
            self.assertAlmostEqual(x0, x1, places=6)
            self.assertAlmostEqual(y0, y1, places=6)

    def test_belt_has_kirkwood_gaps(self):
        a = BELT["a"]
        self.assertGreater(len(a), 300)
        for gap in ASTEROID_GAPS:
            self.assertTrue(np.all(np.abs(a - gap) > 0.034))
        positions = asteroid_positions(BELT, 0.0)
        self.assertEqual(positions.shape, (len(a), 2))
        self.assertTrue(np.all(np.isfinite(positions)))
        self.assertTrue(np.all(positions[:, 0] ** 2 + positions[:, 1] ** 2 >= 2.0 ** 2))


if __name__ == "__main__":
    unittest.main()
