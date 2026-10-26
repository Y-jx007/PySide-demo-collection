"""Unit tests for Lloyd relaxation geometry and kernels (no OpenGL context)."""
import os
import sys
import unittest

import numpy as np

APPS_DIR = os.path.join(os.path.dirname(__file__), "..", "apps")
MATH_DIR = os.path.join(APPS_DIR, "Math")
sys.path.insert(0, APPS_DIR)
sys.path.insert(0, MATH_DIR)

import Lloyd  # noqa: E402


class TestCentroidKernel(unittest.TestCase):
    def test_square_centroid(self):
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], np.float32)
        centroids = np.zeros((1, 2), np.float32)
        Lloyd.compute_cell_centroids(
            polygon, np.array([0], np.int32), np.array([5], np.int32), centroids)
        np.testing.assert_allclose(centroids[0], [0.5, 0.5], atol=1e-5)

    def test_degenerate_cell_falls_back_to_vertex_average(self):
        polygon = np.array([[0, 0], [1, 0], [2, 0], [0, 0]], np.float32)
        centroids = np.zeros((1, 2), np.float32)
        Lloyd.compute_cell_centroids(
            polygon, np.array([0], np.int32), np.array([4], np.int32), centroids)
        # area == 0 -> average of the three non-closing vertices
        np.testing.assert_allclose(centroids[0], [1.0, 0.0], atol=1e-5)

    def test_invalid_cell_is_zeroed(self):
        polygon = np.zeros((4, 2), np.float32)
        centroids = np.full((1, 2), 9.0, np.float32)
        Lloyd.compute_cell_centroids(
            polygon, np.array([0], np.int32), np.array([2], np.int32), centroids)
        np.testing.assert_allclose(centroids[0], [0.0, 0.0])

    def test_monte_carlo_square(self):
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], np.float32)
        centroids = np.zeros((1, 2), np.float32)
        Lloyd.compute_monte_carlo_centroids(
            polygon, np.array([0], np.int32), np.array([5], np.int32),
            centroids, 2000, 1.0)
        self.assertAlmostEqual(float(centroids[0, 0]), 0.5, delta=0.03)
        self.assertAlmostEqual(float(centroids[0, 1]), 0.5, delta=0.03)


class TestMoveKernel(unittest.TestCase):
    def test_mobility_one_reaches_centroid(self):
        points = np.array([[0.0, 0.0], [0.9, 0.9]], np.float32)
        centroids = np.array([[0.5, 0.5], [0.5, 0.5]], np.float32)
        Lloyd.move_points(points, centroids, 1.0, 0.0, 1.0, 0.0, 1.0)
        np.testing.assert_allclose(points, [[0.5, 0.5], [0.5, 0.5]], atol=1e-6)

    def test_positions_are_clamped(self):
        points = np.array([[-5.0, 5.0]], np.float32)
        centroids = np.array([[-5.0, 5.0]], np.float32)
        Lloyd.move_points(points, centroids, 1.0, 0.0, 1.0, 0.0, 1.0)
        self.assertGreaterEqual(float(points[0, 0]), 0.0)
        self.assertLessEqual(float(points[0, 0]), 1.0)


class TestCellBuilder(unittest.TestCase):
    def setUp(self):
        self.sim = Lloyd.HoneycombCVT(n_points=120, mobility=1.5)
        self.sim.step()

    def test_flat_polygons_are_packed_and_closed(self):
        sizes = self.sim._sizes_np
        offsets = self.sim._offsets_np
        flat = self.sim._polygons_np
        self.assertGreaterEqual(int(sizes.min()), 3)
        self.assertEqual(int(sizes.sum()), len(flat))
        self.assertTrue(np.array_equal(offsets, np.concatenate(([0], np.cumsum(sizes)[:-1]))))
        for offset, size in zip(offsets, sizes):
            self.assertTrue(np.allclose(flat[offset], flat[offset + size - 1]))

    def test_cells_cover_the_plane_around_their_point(self):
        # The polygon must surround its seed point (convex Voronoi cell).
        flat = self.sim._polygons_np
        for i in range(0, len(self.sim._sizes_np), 17):
            start = self.sim._offsets_np[i]
            poly = flat[start:start + self.sim._sizes_np[i]]
            x0, y0 = poly.min(axis=0)
            x1, y1 = poly.max(axis=0)
            px, py = self.sim.points[i]
            self.assertGreaterEqual(px, x0 - 1e-5)
            self.assertLessEqual(px, x1 + 1e-5)
            self.assertGreaterEqual(py, y0 - 1e-5)
            self.assertLessEqual(py, y1 + 1e-5)

    def test_triangle_indices_stay_inside_one_cell(self):
        sizes = self.sim._sizes_np
        offsets = self.sim._offsets_np
        sides = self.sim._last_sides
        owner = np.full(len(self.sim._polygons_np), -1, np.int32)
        for i in range(len(sizes)):
            owner[offsets[i]:offsets[i] + sizes[i]] = i

        hex_mask = sides == 6
        other_mask = (sides != 6) & (sides > 0)
        for mask in (hex_mask, other_mask):
            idx = Lloyd.VoronoiWidget._fan_indices(
                offsets[mask].astype(np.int32), sizes[mask].astype(np.int32))
            cells = owner[idx]
            cells = cells.reshape(-1, 3)
            self.assertTrue(np.all(cells[:, 0] == cells[:, 1]))
            self.assertTrue(np.all(cells[:, 0] == cells[:, 2]))

    def test_line_indices_are_adjacent_vertices(self):
        sizes = self.sim._sizes_np
        offsets = self.sim._offsets_np
        idx = Lloyd.VoronoiWidget._line_indices(
            offsets.astype(np.int32), sizes.astype(np.int32))
        pairs = idx.reshape(-1, 2)
        self.assertTrue(np.all(pairs[:, 1] == pairs[:, 0] + 1))


class TestSimulation(unittest.TestCase):
    def test_points_stay_in_bounds_and_are_finite(self):
        sim = Lloyd.HoneycombCVT(n_points=200, mobility=1.5)
        for _ in range(40):
            sim.step()
        self.assertTrue(np.isfinite(sim.points).all())
        self.assertTrue(np.all(sim.points >= 0.0))
        self.assertTrue(np.all(sim.points <= 1.0))

    def test_relaxation_increases_hexagon_fraction(self):
        sim = Lloyd.HoneycombCVT(n_points=300, mobility=1.5)
        for _ in range(50):
            sim.step()
        hex_count, total, fraction, avg_sides = sim.get_stats()
        self.assertEqual(total, 300)
        self.assertGreater(fraction, 0.6)
        self.assertGreater(avg_sides, 5.7)
        self.assertLess(avg_sides, 6.1)

    def test_zero_mobility_reuses_cached_geometry(self):
        sim = Lloyd.HoneycombCVT(n_points=80, mobility=0.0)
        initial = sim.points.copy()
        sim.step()
        cached = sim._polygons_np
        sim.step()
        self.assertIs(sim._polygons_np, cached)
        np.testing.assert_allclose(sim.points, initial, atol=1e-7)

    def test_multi_step_matches_step(self):
        a = Lloyd.HoneycombCVT(n_points=150, mobility=1.5)
        b = Lloyd.HoneycombCVT(n_points=150, mobility=1.5)
        a.step()
        b.step()
        a.multi_step(3)
        for _ in range(3):
            b.step()
        np.testing.assert_allclose(a.points, b.points, atol=1e-5)

    def test_legacy_class_name(self):
        self.assertIs(Lloyd.TaichiHoneycombCVT, Lloyd.HoneycombCVT)


if __name__ == "__main__":
    unittest.main()
