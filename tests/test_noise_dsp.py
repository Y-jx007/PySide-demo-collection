"""DSP checks for the Noise app (no audio device needed)."""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "apps", "Physics"))

from noise_dsp import (  # noqa: E402
    analyse_block, a_weighting_db, leq, log_spectrum, noise_colour,
    percentile_levels, spectral_slope, third_octave_density,
    third_octave_energy, third_octave_levels,
)
from noise_history import NoiseHistory  # noqa: E402


class TestNoiseDSP(unittest.TestCase):
    def test_a_weighting_reference(self):
        self.assertAlmostEqual(float(a_weighting_db([1000.0])[0]), 0.0, places=1)
        self.assertLess(float(a_weighting_db([100.0])[0]), -15.0)

    def test_sine_level_and_frequency(self):
        fs = 44100
        n = 4096
        t = np.arange(n) / fs
        for amp, expected in ((0.01, 50.99), (0.1, 70.99), (1.0, 90.99)):
            freqs, spec, z_db, a_db = analyse_block(
                amp * np.sin(2 * np.pi * 1000.0 * t), fs
            )
            self.assertAlmostEqual(z_db, expected, places=1)
            self.assertAlmostEqual(a_db, expected, places=1)
            self.assertAlmostEqual(float(freqs[int(np.argmax(spec))]), 1000.0, delta=25.0)

    def test_leq_and_percentiles(self):
        self.assertAlmostEqual(leq([60.0, 60.0, 60.0]), 60.0, places=5)
        self.assertAlmostEqual(leq([60.0, 70.0]), 67.4, places=1)
        l10, l50, l90 = percentile_levels([50, 60, 70], (10, 50, 90))
        self.assertAlmostEqual(l10, 52.0, places=5)
        self.assertAlmostEqual(l50, 60.0, places=5)
        self.assertAlmostEqual(l90, 68.0, places=5)

    def test_history_roundtrip_and_hourly(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = NoiseHistory(os.path.join(tmp, "noise.sqlite"))
            now = time.time()
            history.add(now, 50.0, 52.0, 70.0, 0.8)
            history.add(now - 3600, 55.0, 57.0, 75.0, 1.2)
            summary = history.summary(1)
            self.assertEqual(summary["count"], 2)
            self.assertAlmostEqual(summary["avg"], 54.5, places=1)
            self.assertAlmostEqual(summary["alpha"], 1.0, places=2)
            profile = history.hourly_profile(1)
            self.assertEqual(profile.shape, (24,))
            self.assertGreater(int(np.sum(np.isfinite(profile))), 0)
            out = os.path.join(tmp, "out.csv")
            self.assertEqual(history.export_csv(out), 2)
            self.assertTrue(os.path.exists(out))
            history.close()

    def test_1f_slope(self):
        freqs = np.geomspace(50.0, 2000.0, 400)
        spec = -10.0 * np.log10(freqs) + 30.0
        alpha, slope, intercept, r2 = spectral_slope(freqs, spec)
        self.assertAlmostEqual(alpha, 1.0, places=3)
        self.assertAlmostEqual(slope, -1.0, places=3)
        self.assertAlmostEqual(r2, 1.0, places=3)
        self.assertEqual(noise_colour(alpha), "接近 1/f")
        self.assertEqual(noise_colour(-1.0), "偏蓝")

    def test_third_octave_levels(self):
        fs = 44100
        n = 8192
        t = np.arange(n) / fs
        freqs, spec, _z, _a = analyse_block(
            0.1 * np.sin(2 * np.pi * 1000.0 * t), fs
        )
        centers, levels = third_octave_levels(freqs, spec)
        self.assertEqual(levels.shape, (30,))
        self.assertEqual(centers[0], 25.0)
        self.assertEqual(centers[-1], 20000.0)
        self.assertEqual(float(centers[int(np.nanargmax(levels))]), 1000.0)

    def test_third_octave_energy(self):
        fs = 44100
        n = 8192
        t = np.arange(n) / fs
        freqs, spec, _z, _a = analyse_block(
            0.1 * np.sin(2 * np.pi * 1000.0 * t), fs
        )
        centers, energy = third_octave_energy(freqs, spec)
        self.assertEqual(energy.shape, (30,))
        self.assertTrue(np.all(np.isfinite(energy)))
        self.assertEqual(float(centers[int(np.argmax(energy))]), 1000.0)
        self.assertGreater(float(np.max(energy)), 0.0)

    def test_third_octave_density(self):
        fs = 44100
        n = 8192
        t = np.arange(n) / fs
        freqs, spec, _z, _a = analyse_block(
            0.1 * np.sin(2 * np.pi * 1000.0 * t), fs
        )
        centers, density = third_octave_density(freqs, spec)
        self.assertEqual(density.shape, (30,))
        self.assertTrue(np.all(np.isfinite(density)))
        self.assertEqual(float(centers[int(np.argmax(density))]), 1000.0)
        self.assertGreater(float(np.max(density)), 0.0)

    def test_log_spectrum_shape(self):
        freqs = np.linspace(0, 22050, 1025)
        values = np.full_like(freqs, -40.0)
        centers, out = log_spectrum(freqs, values, n_bins=128)
        self.assertEqual(out.shape, (128,))
        self.assertEqual(centers.shape, (128,))
        self.assertTrue(np.all(np.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
