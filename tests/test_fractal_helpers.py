"""Unit tests for the shared complex-fractal helpers (no GPU work)."""
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from common.dynamic_fractal import ViewTransform


class TestViewTransform(unittest.TestCase):
    def setUp(self):
        self.transform = ViewTransform(
            width=900,
            height=600,
            view_scale=1.5,
            center_x=-0.5,
            center_y=0.2,
        )

    def test_affine_coefficients_match_screen_mapping(self):
        t = self.transform
        for i, j in ((0, 0), (450, 300), (899, 599), (123, 456)):
            expected_x = i * t.scale_x + t.offset_x
            expected_y = j * t.scale_y + t.offset_y
            actual_x, actual_y = t.screen_to_world(i, j)
            self.assertAlmostEqual(actual_x, expected_x, places=12)
            self.assertAlmostEqual(actual_y, expected_y, places=12)

    def test_screen_world_roundtrip(self):
        t = self.transform
        for i, j in ((0, 0), (450, 300), (899, 599)):
            wx, wy = t.screen_to_world(i, j)
            sx, sy = t.world_to_screen(wx, wy)
            self.assertAlmostEqual(sx, i, places=9)
            self.assertAlmostEqual(sy, j, places=9)

    def test_aspect(self):
        self.assertAlmostEqual(self.transform.aspect, 1.5, places=12)


class TestCustomExpression(unittest.TestCase):
    def test_compile_returns_callable_with_requested_name(self):
        from support.fractals.fractal_common import compile_complex_function

        func = compile_complex_function("csqr(z) + csin(z)", "custom_func")
        self.assertTrue(callable(func))
        self.assertEqual(func.__name__, "custom_func")

    def test_syntax_error_is_reported(self):
        from support.fractals.fractal_common import compile_complex_function

        with self.assertRaises(SyntaxError):
            compile_complex_function("csqr(z", "custom_func")


if __name__ == "__main__":
    unittest.main()
