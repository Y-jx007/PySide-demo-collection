# -*- coding: utf-8 -*-
"""Tests for long arcs with didactic apsidal precession."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.solar_system_orbits import (  # noqa: E402
    PLANETS,
    planet_arc,
    planet_orbit_at,
    planet_position_at,
)


class TestPrecessingArcs(unittest.TestCase):
    def test_arc_ends_at_current_planet(self):
        for planet in PLANETS:
            arc = planet_arc(planet, 3.0, 3000.0, 12.0, 240)
            self.assertEqual(arc.shape, (240, 2))
            self.assertTrue(np.all(np.isfinite(arc)))
            x, y = planet_position_at(planet, 3.0, 12.0)
            self.assertAlmostEqual(float(arc[-1, 0]), x, places=6)
            self.assertAlmostEqual(float(arc[-1, 1]), y, places=6)

    def test_precession_changes_long_arc(self):
        planet = PLANETS[2]
        fixed = planet_arc(planet, 3.0, 3000.0, 0.0, 240)
        precessed = planet_arc(planet, 3.0, 3000.0, 12.0, 240)
        self.assertGreater(
            float(np.max(np.abs(fixed - precessed))), 0.05
        )
        self.assertGreater(float(np.ptp(precessed[:, 0])), 1.4)

    def test_current_orbit_is_finite(self):
        for planet in PLANETS:
            orbit = planet_orbit_at(planet, 4.0, 12.0, 120)
            self.assertEqual(orbit.shape, (120, 2))
            self.assertTrue(np.all(np.isfinite(orbit)))


if __name__ == "__main__":
    unittest.main()
