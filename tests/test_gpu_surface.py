# -*- coding: utf-8 -*-
"""The shared 3D helpers: camera maths and the vertex batching."""
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from common.gpu_surface import (  # noqa: E402
    Camera3D, GpuSurface, quad_vertex_batch, triangle_vertex_batch, unit,
)

APP = QApplication.instance() or QApplication([])


class TestUnit(unittest.TestCase):
    def test_normalises_and_keeps_zero(self):
        out = unit([3.0, 0.0, 4.0])
        np.testing.assert_allclose(out, [0.6, 0.0, 0.8])
        np.testing.assert_allclose(unit([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0])


class TestCamera(unittest.TestCase):
    def setUp(self):
        self.camera = Camera3D()

    def test_basis_is_orthonormal(self):
        for axis in (self.camera.right, self.camera.up, self.camera.forward):
            self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=9)
        self.assertAlmostEqual(float(self.camera.right @ self.camera.up), 0.0, places=9)
        self.assertAlmostEqual(float(self.camera.right @ self.camera.forward), 0.0, places=9)
        self.assertAlmostEqual(float(self.camera.up @ self.camera.forward), 0.0, places=9)

    def test_target_projects_to_the_canvas_centre(self):
        point, depth = self.camera.project(self.camera.target, 600, 600)
        self.assertAlmostEqual(point.x(), 300.0, places=6)
        self.assertAlmostEqual(point.y(), 300.0, places=6)
        self.assertGreater(depth, 0.0)

    def test_ray_inverts_projection(self):
        world = np.array([0.4, 0.2, -0.3])
        point, _ = self.camera.project(world, 600, 600)
        origin, direction = self.camera.ray(point.x(), point.y(), 600, 600)
        # the point must lie on the ray
        offset = world - origin
        along = float(offset @ direction)
        closest = origin + direction * along
        self.assertLess(float(np.linalg.norm(world - closest)), 1e-9)

    def test_zoom_scales_the_offset_from_the_centre(self):
        world = np.array([0.5, 0.0, 0.0])
        near, _ = self.camera.project(world, 600, 600)
        self.camera.zoom = 2.0
        far, _ = self.camera.project(world, 600, 600)
        self.assertAlmostEqual((far.x() - 300.0) / (near.x() - 300.0), 2.0, places=6)

    def test_pitch_and_yaw_move_the_eye(self):
        start = self.camera.eye.copy()
        self.camera.yaw += 30.0
        self.camera.update()
        moved = self.camera.eye.copy()
        self.assertGreater(float(np.linalg.norm(moved - start)), 0.1)
        self.assertAlmostEqual(
            float(np.linalg.norm(moved - self.camera.target)),
            self.camera.distance,
            places=6,
        )

    def test_world_per_pixel_matches_the_projection_scale(self):
        step = self.camera.world_per_pixel(600, 600)
        a, _ = self.camera.project(self.camera.target, 600, 600)
        # a step along the view plane, not along the world axis
        offset = self.camera.target + self.camera.up * step
        b, _ = self.camera.project(offset, 600, 600)
        self.assertAlmostEqual(abs(a.y() - b.y()), 1.0, places=4)


class TestVertexBatches(unittest.TestCase):
    def test_quads_repeat_colours_per_corner(self):
        corners = np.array([[[0, 0], [4, 0], [4, 4], [0, 4]],
                            [[5, 5], [6, 5], [6, 6], [5, 6]]], dtype=float)
        colours = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        vertices, colours_out = quad_vertex_batch(corners, colours)
        self.assertEqual(vertices.shape, (8, 2))
        self.assertEqual(colours_out.shape, (8, 3))
        np.testing.assert_allclose(vertices[:4], corners[0])
        np.testing.assert_allclose(colours_out[:4], [[1.0, 0.0, 0.0]] * 4)
        np.testing.assert_allclose(colours_out[4:], [[0.0, 1.0, 0.0]] * 4)

    def test_triangles_repeat_colours_per_corner(self):
        corners = np.array([[[0, 0], [3, 0], [0, 3]]], dtype=float)
        vertices, colours_out = triangle_vertex_batch(corners, [[0.2, 0.4, 0.6]])
        self.assertEqual(vertices.shape, (3, 2))
        self.assertEqual(colours_out.shape, (3, 3))

    def test_shape_mismatch_is_reported(self):
        with self.assertRaises(ValueError):
            quad_vertex_batch(np.zeros((1, 3, 2)), np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            triangle_vertex_batch(np.zeros((2, 3, 2)), np.zeros((1, 3)))


class TestGpuFallback(unittest.TestCase):
    def test_headless_platform_reports_no_gpu(self):
        if QApplication.platformName() not in ("offscreen", "minimal", "vnc"):
            self.skipTest("GL may exist on this platform")
        self.assertIsNone(GpuSurface.create())

    def test_render_without_context_returns_none(self):
        gpu = GpuSurface()
        self.assertFalse(gpu.ok)
        self.assertIsNone(gpu.render(100, 100, 1.0, lambda w, h: None))


if __name__ == "__main__":
    unittest.main()