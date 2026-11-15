# -*- coding: utf-8 -*-
"""Differential-geometry checks for the Connection app."""
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.math.connection import (  # noqa: E402
    PlaneSurface, SphereSurface, oriented_angle, parallel_transport,
)


class TestConnectionMath(unittest.TestCase):
    def test_plane_christoffel_zero(self):
        surf = PlaneSurface()
        gamma = surf.christoffel(0.2, 0.3)
        self.assertLess(float(np.max(np.abs(gamma))), 1e-8)

    def test_plane_transport_is_parallel(self):
        surf = PlaneSurface()
        v = np.array([[0.3, -0.4]])
        out = parallel_transport(surf, (0.0, 0.0), (0.4, 0.5), v, steps=5)
        np.testing.assert_allclose(out[0], v[0], atol=1e-6)

    def test_sphere_transport_preserves_length(self):
        surf = SphereSurface()
        uv = surf.initial()
        comps, vecs = surf.cluster(uv[0], uv[1], 8)
        out = parallel_transport(surf, uv, (uv[0] + 0.8, uv[1] + 0.15), comps, steps=12)
        self.assertTrue(np.all(np.isfinite(out)))
        for c in out:
            vec = surf.components_to_vector(uv[0] + 0.8, uv[1] + 0.15, c)
            self.assertAlmostEqual(float(np.linalg.norm(vec)), 0.24, places=2)

    def test_sphere_latitude_holonomy(self):
        surf = SphereSurface()
        v = 1.10
        u0 = 0.25
        comps, _ = surf.cluster(u0, v, 1, 0.24)
        start = comps.copy()
        n = 48
        prev = u0
        for k in range(1, n + 1):
            u = u0 + 2.0 * np.pi * k / n
            comps = parallel_transport(surf, (prev, v), (u, v), comps, steps=4)
            prev = u
        normal = surf.normal(u0, v)
        first = surf.components_to_vector(u0, v, start[0])
        last = surf.components_to_vector(u0, v, comps[0])
        angle = oriented_angle(normal, first, last)
        self.assertGreater(abs(angle), 1.0)
        self.assertLess(abs(angle), 360.0)

    def test_sphere_latitude_holonomy_matches_solid_angle(self):
        """Gauss-Bonnet: going around a latitude line rotates by the cap area."""
        surf = SphereSurface()
        u0 = 0.3
        for v in (0.8, 1.0):
            comps, _ = surf.cluster(u0, v, 1, 0.24)
            start = comps.copy()
            prev = u0
            n = 120
            for k in range(1, n + 1):
                u = u0 + 2.0 * math.pi * k / n
                comps = parallel_transport(surf, (prev, v), (u, v), comps, steps=4)
                prev = u
            angle = abs(oriented_angle(
                surf.normal(u0, v),
                surf.components_to_vector(u0, v, start[0]),
                surf.components_to_vector(u0, v, comps[0]),
            ))
            cap = math.degrees(2.0 * math.pi * (1.0 - math.cos(v)))
            wrapped = abs(((angle - cap) % 360.0 + 180.0) % 360.0 - 180.0)
            self.assertLess(wrapped, 0.5, (v, angle, cap))


if __name__ == "__main__":
    unittest.main()
