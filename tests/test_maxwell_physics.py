"""Unit tests for Maxwell's physics kernels (no OpenGL context required)."""
import os
import sys
import unittest

import numpy as np

APPS_DIR = os.path.join(os.path.dirname(__file__), "..", "apps")
PHYSICS_DIR = os.path.join(APPS_DIR, "Physics")
sys.path.insert(0, APPS_DIR)
sys.path.insert(0, PHYSICS_DIR)

import Maxwell  # noqa: E402


def _make_grid(width, height, cell, n):
    grid_w = max(1, int(width // cell))
    grid_h = max(1, int(height // cell))
    ncells = grid_w * grid_h
    return {
        "grid_w": grid_w,
        "grid_h": grid_h,
        "inv_cell_w": 1.0 / (width / grid_w),
        "inv_cell_h": 1.0 / (height / grid_h),
        "cell_ids": np.empty(n, np.int32),
        "counts": np.empty(ncells, np.int32),
        "cursor": np.empty(ncells + 1, np.int32),
        "starts": np.empty(ncells + 1, np.int32),
        "items": np.empty(n, np.int32),
        "occupied": np.empty(min(n, ncells), np.int32),
    }


def _build(x, y, grid):
    return Maxwell.build_cell_grid(
        x, y, grid["grid_w"], grid["grid_h"],
        grid["inv_cell_w"], grid["inv_cell_h"],
        grid["cell_ids"], grid["counts"], grid["cursor"],
        grid["starts"], grid["items"], grid["occupied"])


def _resolve(x, y, vx, vy, radius, grid):
    n_occupied = _build(x, y, grid)
    Maxwell.resolve_grid_collisions(
        x, y, vx, vy, radius, grid["starts"], grid["items"],
        grid["occupied"], n_occupied, grid["grid_w"], grid["grid_h"])


class TestPairCollision(unittest.TestCase):
    def test_head_on_equal_mass_exchange(self):
        x = np.array([10.0, 14.0], np.float32)
        y = np.zeros(2, np.float32)
        vx = np.array([10.0, -10.0], np.float32)
        vy = np.zeros(2, np.float32)
        Maxwell._resolve_pair(x, y, vx, vy, 2.0, 0, 1)
        np.testing.assert_allclose(vx, [-10.0, 10.0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(vy, [0.0, 0.0], rtol=0, atol=1e-6)

    def test_separating_pair_is_untouched(self):
        x = np.array([10.0, 13.0], np.float32)
        y = np.zeros(2, np.float32)
        vx = np.array([-5.0, 5.0], np.float32)
        vy = np.zeros(2, np.float32)
        Maxwell._resolve_pair(x, y, vx, vy, 2.0, 0, 1)
        np.testing.assert_allclose(vx, [-5.0, 5.0], rtol=0, atol=1e-6)

    def test_overlap_is_pushed_apart(self):
        x = np.array([10.0, 12.0], np.float32)
        y = np.zeros(2, np.float32)
        vx = np.zeros(2, np.float32)
        vy = np.zeros(2, np.float32)
        Maxwell._resolve_pair(x, y, vx, vy, 2.0, 0, 1)
        self.assertAlmostEqual(abs(float(x[0] - x[1])), 4.0, places=4)

    def test_energy_is_conserved(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(5, 45, 2).astype(np.float32)
        y = rng.uniform(5, 45, 2).astype(np.float32)
        vx = rng.uniform(-10, 10, 2).astype(np.float32)
        vy = rng.uniform(-10, 10, 2).astype(np.float32)
        before = float(np.sum(vx * vx + vy * vy))
        Maxwell._resolve_pair(x, y, vx, vy, 2.0, 0, 1)
        after = float(np.sum(vx * vx + vy * vy))
        self.assertAlmostEqual(after, before, places=4)


class TestNeighbourCells(unittest.TestCase):
    """Pairs straddling every neighbour direction must be found by the grid."""

    WIDTH = 40.0
    HEIGHT = 20.0
    CELL = 4.0
    RADIUS = 1.0

    def _run_pair(self, p1, p2, v1, v2):
        x = np.array([p1[0], p2[0]], np.float32)
        y = np.array([p1[1], p2[1]], np.float32)
        vx = np.array([v1[0], v2[0]], np.float32)
        vy = np.array([v1[1], v2[1]], np.float32)
        _resolve(x, y, vx, vy, self.RADIUS, _make_grid(self.WIDTH, self.HEIGHT,
                                                       self.CELL, 2))
        return vx, vy

    def test_same_cell(self):
        vx, vy = self._run_pair((5.0, 5.0), (5.5, 5.0), (5, 0), (-5, 0))
        np.testing.assert_allclose(vx, [-5, 5], atol=1e-5)

    def test_right_cell(self):
        vx, vy = self._run_pair((3.9, 5.0), (4.1, 5.0), (5, 0), (-5, 0))
        np.testing.assert_allclose(vx, [-5, 5], atol=1e-5)

    def test_up_cell(self):
        vx, vy = self._run_pair((5.0, 3.9), (5.0, 4.1), (0, 5), (0, -5))
        np.testing.assert_allclose(vy, [-5, 5], atol=1e-5)

    def test_diagonal_up_right_cell(self):
        vx, vy = self._run_pair((3.9, 3.9), (4.1, 4.1), (5, 5), (-5, -5))
        self.assertLess(vx[0], 0.0)
        self.assertLess(vy[0], 0.0)
        self.assertGreater(vx[1], 0.0)
        self.assertGreater(vy[1], 0.0)

    def test_diagonal_up_left_cell(self):
        vx, vy = self._run_pair((4.1, 3.9), (3.9, 4.1), (-5, 5), (5, -5))
        self.assertGreater(vx[0], 0.0)
        self.assertLess(vy[0], 0.0)
        self.assertLess(vx[1], 0.0)
        self.assertGreater(vy[1], 0.0)


class TestGridBuild(unittest.TestCase):
    def test_every_particle_is_bucketed_once(self):
        rng = np.random.default_rng(3)
        n = 500
        x = rng.uniform(0, 100, n).astype(np.float32)
        y = rng.uniform(0, 100, n).astype(np.float32)
        grid = _make_grid(100, 100, 5, n)
        n_occupied = _build(x, y, grid)
        self.assertEqual(n_occupied, len(np.unique(grid["cell_ids"])))
        np.testing.assert_array_equal(np.sort(grid["items"]), np.arange(n))
        self.assertEqual(int(grid["starts"][grid["grid_w"] * grid["grid_h"]]), n)

    def test_dense_random_cluster_keeps_energy(self):
        rng = np.random.default_rng(11)
        n = 150
        width = height = 120.0
        x = rng.uniform(2, width - 2, n).astype(np.float32)
        y = rng.uniform(2, height - 2, n).astype(np.float32)
        vx = rng.uniform(-20, 20, n).astype(np.float32)
        vy = rng.uniform(-20, 20, n).astype(np.float32)
        before = float(np.sum(vx * vx + vy * vy))
        _resolve(x, y, vx, vy, 2.0, _make_grid(width, height, 4, n))
        after = float(np.sum(vx * vx + vy * vy))
        self.assertTrue(np.all(np.isfinite(x)) and np.all(np.isfinite(y)))
        self.assertAlmostEqual(after / before, 1.0, places=4)


class TestIntegration(unittest.TestCase):
    def test_wall_bounce_preserves_speed(self):
        x = np.array([1.0], np.float32)
        y = np.array([50.0], np.float32)
        vx = np.array([-10.0], np.float32)
        vy = np.array([0.0], np.float32)
        Maxwell.integrate_positions(x, y, vx, vy, 2.0, 0.05, 100.0, 100.0)
        self.assertGreaterEqual(float(x[0]), 2.0)
        self.assertGreater(float(vx[0]), 0.0)

    def test_positions_stay_in_bounds(self):
        rng = np.random.default_rng(5)
        n = 300
        x = rng.uniform(0, 100, n).astype(np.float32)
        y = rng.uniform(0, 60, n).astype(np.float32)
        vx = rng.uniform(-100, 100, n).astype(np.float32)
        vy = rng.uniform(-100, 100, n).astype(np.float32)
        for _ in range(100):
            Maxwell.integrate_positions(x, y, vx, vy, 2.0, 0.05, 100.0, 60.0)
        self.assertTrue(np.all(x >= 2.0 - 1e-3) and np.all(x <= 98.0 + 1e-3))
        self.assertTrue(np.all(y >= 2.0 - 1e-3) and np.all(y <= 58.0 + 1e-3))


class TestCreation(unittest.TestCase):
    def test_fills_canvas(self):
        width, height = 625, 927
        x, y, vx, vy = Maxwell.create_grid_balls(2000, 3.0, 20.0, width, height)
        self.assertEqual(x.shape, (2000,))
        self.assertEqual(x.dtype, np.float32)
        self.assertGreater(float(x.max() - x.min()), 0.85 * width)
        self.assertGreater(float(y.max() - y.min()), 0.85 * height)
        self.assertGreaterEqual(float(x.min()), 0.0)
        self.assertLessEqual(float(x.max()), width)
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), height)

    def test_initial_speeds_match_request(self):
        x, y, vx, vy = Maxwell.create_grid_balls(500, 3.0, 20.0, 625, 927)
        speeds = np.sqrt(vx * vx + vy * vy)
        np.testing.assert_allclose(speeds, 20.0, rtol=0, atol=1e-4)

    def test_auto_radius_layout_has_no_initial_overlap(self):
        n = 500
        radius = 3.0 * (2000.0 / n) ** 0.5
        width, height = 625, 927
        x, y, _, _ = Maxwell.create_grid_balls(n, radius, 20.0, width, height)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx * dx + dy * dy
        np.fill_diagonal(dist_sq, np.inf)
        self.assertGreaterEqual(float(np.sqrt(dist_sq.min())), 2.0 * radius * 0.99)

    def test_zero_speed_is_allowed(self):
        x, y, vx, vy = Maxwell.create_grid_balls(10, 1.0, 0.0, 100, 100)
        np.testing.assert_array_equal(vx, np.zeros(10, np.float32))


if __name__ == "__main__":
    unittest.main()
