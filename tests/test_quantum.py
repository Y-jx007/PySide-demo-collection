# -*- coding: utf-8 -*-
"""Two-mode squeezing: twin beams are quieter than shot noise."""
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.quantum import (  # noqa: E402
    gain_to_squeezing,
    photon_difference_variance,
    shot_noise_variance,
    squeezing_db,
)


class TestTwoModeSqueezing(unittest.TestCase):
    def test_zero_gain_is_zero_squeezing(self):
        self.assertAlmostEqual(gain_to_squeezing(1.0), 0.0, places=9)
        self.assertAlmostEqual(squeezing_db(1.0), 0.0, places=9)

    def test_gain_squeezes_the_quadrature(self):
        self.assertLess(squeezing_db(4.0), -10.0)
        self.assertLess(squeezing_db(20.0), squeezing_db(4.0))

    def test_twin_beams_beat_shot_noise(self):
        difference, mean = photon_difference_variance(4.0)
        shot = shot_noise_variance(4.0)
        self.assertGreater(difference, 0.0)
        self.assertLess(difference, 0.25 * shot)

    def test_no_gain_means_shot_noise(self):
        difference, _ = photon_difference_variance(1.0)
        shot = shot_noise_variance(1.0)
        self.assertAlmostEqual(difference / shot, 1.0, delta=0.15)


if __name__ == "__main__":
    unittest.main()
