# -*- coding: utf-8 -*-
"""Real apsidal precession rates and time-accelerated orbit paths."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.solar_system_orbits import (  # noqa: E402
    PLANETS,
    PRECESSION_ARCSEC_PER_YEAR,
    planet_orbit_at,
)


class TestRealPrecession(unittest.TestCase):
    def test_mercury_rate_is_realistic(self):
        rate = PRECESSION_ARCSEC_PER_YEAR["水星"] / 3600.0
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 0.002)

    def test_orbit_rotates_with_real_rate(self):
        mercury = next(p for p in PLANETS if p["name"] == "水星")
        rate = PRECESSION_ARCSEC_PER_YEAR["水星"] / 3600.0
        first = planet_orbit_at(
            mercury, 0.0, precession_deg_per_year=rate
        )
        later = planet_orbit_at(
            mercury, 1000.0, precession_deg_per_year=rate
        )
        self.assertFalse(np.allclose(first, later))

    def test_negligible_rate_for_outer_planets(self):
        self.assertEqual(PRECESSION_ARCSEC_PER_YEAR["木星"], 0.0)
        self.assertEqual(PRECESSION_ARCSEC_PER_YEAR["海王星"], 0.0)


if __name__ == "__main__":
    unittest.main()
