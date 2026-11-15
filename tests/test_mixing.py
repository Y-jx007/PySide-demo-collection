# -*- coding: utf-8 -*-
"""Sum-frequency generation: Manley-Rowe invariants and phase matching."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.mixing import (  # noqa: E402
    PulseSHG,
    fwm_conjugate,
    propagate_sfg,
)


class TestSumFrequency(unittest.TestCase):
    def run_case(self, detune, kappa=1.0, length=2.0, samples=400):
        a1, a2, a3, history = propagate_sfg(
            np.array([float(detune)]), length=length, samples=samples, kappa=kappa
        )
        return (float(np.abs(a1[0]) ** 2), float(np.abs(a2[0]) ** 2),
                float(np.abs(a3[0]) ** 2), history)

    def test_photon_flux_invariants(self):
        for detune in (0.0, 7.0):
            n1, n2, n3, _ = self.run_case(detune)
            self.assertAlmostEqual(n1 + n3, 1.0, places=9)
            self.assertAlmostEqual(n2 + n3, 0.36, places=9)

    def test_phase_matching_converts_more(self):
        _, _, matched, _ = self.run_case(0.0)
        _, _, mismatched, _ = self.run_case(7.0)
        self.assertGreater(matched, mismatched)

    def test_conversion_is_bounded_by_the_weaker_pump(self):
        _, _, _, history = self.run_case(0.0, length=4.0, samples=600)
        peak = float(np.max(history[:, 0]))
        self.assertGreater(peak, 0.3)
        self.assertLessEqual(peak, 0.36 + 1e-6)

    def test_zero_coupling_does_nothing(self):
        _, _, n3, _ = self.run_case(0.0, kappa=0.0)
        self.assertEqual(n3, 0.0)




class TestFourWaveMixing(unittest.TestCase):
    def gain(self, detune, pump=0.6, kappa=1.0, length=2.0):
        return float(fwm_conjugate(np.array([float(detune)]), pump=pump,
                                   kappa=kappa, length=length)[0])

    def test_phase_matched_conjugate_is_amplified(self):
        self.assertGreater(self.gain(0.0), 10.0)

    def test_mismatch_suppresses_the_gain(self):
        self.assertGreater(self.gain(0.0), 10.0 * self.gain(30.0))

    def test_gain_grows_with_the_pump(self):
        self.assertGreater(self.gain(0.0, pump=0.6), self.gain(0.0, pump=0.2))

    def test_no_pump_no_gain(self):
        self.assertAlmostEqual(self.gain(0.0, pump=0.0), 1.0, places=6)



class TestPulseSHG(unittest.TestCase):
    def run_case(self, walk, steps=120):
        pulse = PulseSHG(walk=walk)
        pulse.step(steps)
        return pulse.efficiency(), pulse.shift()

    def test_matched_pulse_converts(self):
        efficiency, _ = self.run_case(0.0)
        self.assertGreater(efficiency, 0.1)

    def test_no_walk_means_no_time_shift(self):
        _, shift = self.run_case(0.0)
        self.assertAlmostEqual(shift, 0.0, places=6)

    def test_walk_off_delays_the_harmonic(self):
        _, shift = self.run_case(0.4)
        self.assertLess(shift, -0.05)

    def test_walk_off_costs_efficiency(self):
        matched, _ = self.run_case(0.0)
        walked, _ = self.run_case(0.8)
        self.assertLess(walked, matched)

if __name__ == "__main__":
    unittest.main()
