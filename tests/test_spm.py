# -*- coding: utf-8 -*-
"""Temporal NLSE: soliton invariance, breathing and self-phase modulation."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.spm import TemporalNLSE  # noqa: E402


class TestSoliton(unittest.TestCase):
    def setUp(self):
        self.sim = TemporalNLSE()

    def test_energy_is_conserved(self):
        self.sim.reset(2.0, 1.0)
        before = self.sim.energy()
        self.sim.step(300)
        self.assertAlmostEqual(self.sim.energy(), before, delta=1e-8 * before)

    def test_fundamental_soliton_does_not_change(self):
        self.sim.reset(1.0, 1.0)
        width, spread = self.sim.width(), self.sim.spectral_width()
        self.sim.step(250)
        self.assertAlmostEqual(self.sim.width(), width, delta=0.02 * width)
        self.assertAlmostEqual(self.sim.spectral_width(), spread,
                               delta=0.02 * spread)

    def test_sub_soliton_pulse_broadens(self):
        self.sim.reset(0.6, 1.0)
        width = self.sim.width()
        self.sim.step(250)
        self.assertGreater(self.sim.width(), 1.5 * width)

    def test_spectral_width_grows_with_the_order(self):
        spreads = []
        for order in (1.0, 1.6, 2.0, 3.0):
            self.sim.reset(order, 1.0)
            self.sim.step(200)
            spreads.append(self.sim.spectral_width())
        self.assertEqual(spreads, sorted(spreads))

    def test_two_soliton_returns_after_one_period(self):
        self.sim.reset(2.0, 1.0)
        start = self.sim.spectral_width()
        steps = int(round((np.pi / 2.0) / self.sim.dz))
        self.sim.step(steps)
        self.assertAlmostEqual(self.sim.spectral_width(), start, delta=0.1 * start)


class TestWaterfall(unittest.TestCase):
    def test_rows_fill_from_the_bottom(self):
        sim = TemporalNLSE(rows=64)
        sim.reset(2.0, 1.0)
        self.assertEqual(sim.filled, 1)
        sim.step(5)
        self.assertEqual(sim.filled, 6)
        image = sim.waterfall()
        self.assertEqual(image.shape, (64, sim.n))
        np.testing.assert_allclose(image[-1], sim.intensity())
        self.assertTrue(np.all(image[:-6] == 0.0))

    def test_buffer_scrolls_when_full(self):
        sim = TemporalNLSE(rows=16)
        sim.reset(2.0, 1.0)
        sim.step(40)
        self.assertEqual(sim.filled, 16)
        image = sim.waterfall()
        np.testing.assert_allclose(image[-1], sim.intensity())
        self.assertGreater(float(image.max()), 0.0)

    def test_render_shape_and_dtype(self):
        sim = TemporalNLSE(rows=32)
        sim.reset(2.0, 1.0)
        sim.step(40)
        rgb = sim.render_waterfall()
        self.assertEqual(rgb.shape, (32, sim.n, 3))
        self.assertEqual(rgb.dtype, np.uint8)


class TestSpectrum(unittest.TestCase):
    def test_spectrum_is_centred_and_normalised(self):
        sim = TemporalNLSE()
        sim.reset(1.0, 1.0)
        db = sim.spectrum_db()
        self.assertEqual(db.shape, (sim.n,))
        self.assertAlmostEqual(float(db.max()), 1.0, places=6)
        centre = db.size // 2
        self.assertAlmostEqual(float(db[centre]), 1.0, places=6)




class TestRamanShift(unittest.TestCase):
    def centroid(self, raman, steps=200):
        sim = TemporalNLSE(raman=raman)
        sim.reset(2.0, 1.0)
        sim.step(steps)
        spec = np.abs(np.fft.fft(sim.field)) ** 2
        omega = np.fft.fftfreq(sim.n, d=sim.dt) * 2.0 * np.pi
        return float(np.sum(omega * spec) / np.sum(spec))

    def test_raman_shifts_the_spectrum_red(self):
        self.assertAlmostEqual(self.centroid(0.0), 0.0, places=9)
        self.assertLess(self.centroid(0.02), -1.0)

    def test_shift_grows_with_the_raman_coefficient(self):
        self.assertLess(self.centroid(0.05), self.centroid(0.02))

if __name__ == "__main__":
    unittest.main()
