"""Data and terrain-model tests for the Circulation app."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "apps", "Physics"))

from circulation_data import (  # noqa: E402
    RealWind, elevation_at, load_elevation, terrain_adjust,
)


class TestCirculationData(unittest.TestCase):
    def test_elevation_grid(self):
        elev, lat, lon = load_elevation()
        self.assertEqual(elev.shape, (181, 361))
        self.assertEqual(lat.shape, (181,))
        self.assertEqual(lon.shape, (361,))
        self.assertGreater(int(elev.max()), 4000)
        grid = elevation_at(360, 180)
        self.assertEqual(grid.shape, (180, 360))
        self.assertTrue(np.isfinite(grid).all())

    def test_real_wind_months_and_shapes(self):
        real = RealWind()
        self.assertEqual(real.year_values.tolist(), list(range(2015, 2025)))
        self.assertEqual(real.month_values.tolist(), list(range(1, 13)))
        self.assertEqual(len(real.years), 120)
        self.assertEqual(len(real.months), 120)
        self.assertEqual(real.u.shape, (120, 71, 144))
        self.assertEqual(real.v.shape, real.u.shape)
        self.assertEqual(real.clim_u.shape, (12, 71, 144))
        self.assertTrue(np.isfinite(real.u.astype(np.float32)).all())
        speed = np.hypot(real.u.astype(np.float32), real.v.astype(np.float32))
        self.assertLess(float(speed.max()), 30.0)
        self.assertGreater(float(speed.mean()), 0.5)

    def test_trend_and_anomaly_are_finite(self):
        real = RealWind()
        self.assertEqual(real.trend_u.shape, (71, 144))
        self.assertTrue(np.isfinite(real.trend_u).all())
        for layer in ("monthly", "anomaly", "trend"):
            field = real.field(0, 0, layer)
            self.assertEqual(field.shape, (71, 144, 2))
            self.assertTrue(np.isfinite(field).all())

    def test_direction_delta_is_bounded(self):
        real = RealWind()
        delta = real.direction_delta(0, 0)
        self.assertEqual(delta.shape, (71, 144))
        self.assertLessEqual(float(np.abs(delta).max()), 180.0)

    def test_terrain_slows_wind_and_stays_finite(self):
        nh, nw = 40, 40
        yy = np.linspace(0.0, 1.0, nh, dtype=np.float32)[:, None]
        elev = np.tile((yy * 3500.0), (1, nw)).astype(np.float32)
        u = np.full((nh, nw), 5.0, np.float32)
        v = np.zeros((nh, nw), np.float32)
        u2, v2 = terrain_adjust(u, v, elev, strength=1.0)
        self.assertTrue(np.isfinite(u2).all())
        self.assertTrue(np.isfinite(v2).all())
        low = slice(0, nh // 3)
        high = slice(2 * nh // 3, nh)
        low_speed = float(np.hypot(u2, v2)[low].mean())
        high_speed = float(np.hypot(u2, v2)[high].mean())
        self.assertLess(high_speed, low_speed)


if __name__ == "__main__":
    unittest.main()
