# -*- coding: utf-8 -*-
"""扰动法深度缩放的参考轨道（高精度 → f64）。"""
import sys
import unittest
from decimal import Decimal, getcontext, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from support.fractals.reference_orbit import reference_orbit


class ReferenceOrbitTests(unittest.TestCase):
    def test_mandelbrot_orbit(self):
        # c = i 的临界轨道：0, i, -1+i, -i, -1+i, -i, ...
        re, im = reference_orbit(0.0, 0.0, 1.0, 0.0, 6)
        expected = [(0, 0), (0, 1), (-1, 1), (0, -1), (-1, 1), (0, -1), (-1, 1)]
        for n, (er, ei) in enumerate(expected):
            self.assertAlmostEqual(float(re[n]), er, places=12)
            self.assertAlmostEqual(float(im[n]), ei, places=12)

    def test_julia_orbit_starts_at_center(self):
        # Julia z²（c = 0），z0 = i → -1 → 1
        re, im = reference_orbit(0.0, 0.0, 1.0, 0.0, 2,
                                 julia_re=0.0, julia_im=0.0)
        self.assertAlmostEqual(float(re[0]), 0.0, places=12)
        self.assertAlmostEqual(float(im[0]), 1.0, places=12)
        self.assertAlmostEqual(float(re[1]), -1.0, places=12)
        self.assertAlmostEqual(float(im[1]), 0.0, places=12)

    def test_deep_center_keeps_the_low_word(self):
        hi = -0.7436438870371587
        lo = -3.628952515063387e-17
        # 低位比 f64 的 ulp 小，float 相加会把它吃掉；参考轨道必须用 (hi, lo)。
        self.assertEqual(hi + lo, hi)
        plain_re, _ = reference_orbit(hi, 0.0, 0.13182590420531198, 0.0, 60)
        deep_re, _ = reference_orbit(hi, lo, 0.13182590420531198, 0.0, 60)
        self.assertFalse(np.array_equal(plain_re, deep_re))

    def test_rounding_matches_an_independent_80_digit_orbit(self):
        getcontext().prec = 60
        cx = Decimal("-0.743643887037158704752191506114774")
        cy = Decimal("0.131825904205311970493132056385139")
        hi_x = float(cx)
        lo_x = float(cx - Decimal(hi_x))
        hi_y = float(cy)
        lo_y = float(cy - Decimal(hi_y))

        re, im = reference_orbit(hi_x, lo_x, hi_y, lo_y, 40)

        with localcontext() as ctx:
            ctx.prec = 80
            zr = zi = Decimal(0)
            for n in range(41):
                self.assertAlmostEqual(float(re[n]), float(zr), places=15)
                self.assertAlmostEqual(float(im[n]), float(zi), places=15)
                zr, zi = zr * zr - zi * zi + cx, 2 * zr * zi + cy


if __name__ == "__main__":
    unittest.main()
