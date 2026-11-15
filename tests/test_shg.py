# -*- coding: utf-8 -*-
"""SHG: photon conservation, the Maker fringe, QPM and the symmetry switch."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.shg import SHGMap, propagate  # noqa: E402


class TestCoupledWaves(unittest.TestCase):
    def test_photon_flux_is_conserved(self):
        a1, a2, _ = propagate(0.0, length=2.0, samples=400, kappa=1.0)
        flux = float(np.abs(a1) ** 2 + 2.0 * np.abs(a2) ** 2)
        self.assertAlmostEqual(flux, 1.0, places=6)
        # 完全匹配时泵浦确实被耗尽
        self.assertLess(float(np.abs(a1) ** 2), 0.5)

    def test_undepleted_efficiency_matches_the_analytic_curve(self):
        kappa, length, amp = 0.2, 0.4, 1.0
        for delta_k in (0.0, 2.0, 5.0, 7.85):
            a1, a2, _ = propagate(delta_k, length=length, samples=400, kappa=kappa,
                                  amplitude=amp)
            numeric = float(np.abs(a2) ** 2)
            if delta_k == 0.0:
                analytic = (kappa * length / 2.0) ** 2
            else:
                analytic = (kappa ** 2 * amp ** 4
                            * np.sin(delta_k * length / 2.0) ** 2 / delta_k ** 2)
            self.assertAlmostEqual(numeric / analytic, 1.0, places=2)

    def test_first_zero_sits_at_dk_times_l_equals_two_pi(self):
        kappa, length = 0.2, 0.4
        peak_a1, peak_a2, _ = propagate(0.0, length=length, samples=400, kappa=kappa)
        zero_a1, zero_a2, _ = propagate(2.0 * np.pi / length, length=length,
                                        samples=400, kappa=kappa)
        peak = float(np.abs(peak_a2) ** 2)
        zero = float(np.abs(zero_a2) ** 2)
        self.assertGreater(peak, 0.0)
        self.assertLess(zero, 1e-4 * peak)

    def test_centrosymmetric_medium_never_converts(self):
        a1, a2, _ = propagate(0.0, length=2.0, samples=400, kappa=0.0)
        self.assertEqual(float(np.abs(a2) ** 2), 0.0)


class TestQPM(unittest.TestCase):
    def test_periodic_poling_recovers_a_mismatched_crystal(self):
        delta_k, length = 20.0, 2.0
        poling = 2.0 * np.pi / delta_k
        _, plain, _ = propagate(delta_k, length=length, samples=600, kappa=1.0)
        _, poled, _ = propagate(delta_k, length=length, samples=600, kappa=1.0,
                                poling=poling)
        self.assertGreater(float(np.abs(poled) ** 2),
                           20.0 * float(np.abs(plain) ** 2))


class TestMap(unittest.TestCase):
    def test_map_shape_and_render(self):
        model = SHGMap(mismatches=41, samples=60, length=1.0, kappa=1.0, span=10.0)
        self.assertEqual(model.map.shape, (41, 60))
        self.assertEqual(model.efficiency().shape, (41,))
        rgb = model.render()
        self.assertEqual(rgb.shape, (41, 60, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_render_uses_the_wavelength_colours(self):
        model = SHGMap(mismatches=41, samples=60, length=2.0, kappa=1.0, span=8.0)
        rgb = model.render()
        # 转换最弱处偏红（基频），最强处偏绿（倍频）
        weakest = rgb[np.unravel_index(np.argmin(model.map), model.map.shape)]
        strongest = rgb[np.unravel_index(np.argmax(model.map), model.map.shape)]
        self.assertGreater(int(weakest[0]), int(weakest[1]))
        self.assertGreater(int(strongest[1]), int(strongest[0]))

    def test_beam_colours_run_from_pump_to_harmonic(self):
        model = SHGMap(mismatches=41, samples=60, length=2.0, kappa=1.0, span=8.0)
        colours = model.beam_colours()
        self.assertEqual(colours.shape, (60, 3))
        self.assertGreater(int(colours[0, 0]), int(colours[0, 1]))
        self.assertGreater(int(colours[-1, 1]), int(colours[-1, 0]))
        strip = model.beam_strip(strip=12)
        self.assertEqual(strip.shape, (12, 60, 3))

    def test_symmetry_switch_zeroes_the_map(self):
        model = SHGMap(mismatches=21, samples=40, length=1.0, kappa=1.0, span=8.0)
        self.assertGreater(model.peak_efficiency(), 0.0)
        model.configure(symmetric=True)
        self.assertEqual(model.peak_efficiency(), 0.0)


if __name__ == "__main__":
    unittest.main()