# -*- coding: utf-8 -*-
"""Jones polynomial values for the two standard prime knots."""
import importlib.util
import os
import sys
from fractions import Fraction
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_jones():
    path = os.path.join(ROOT, "apps", "Math", "Jones.py")
    spec = importlib.util.spec_from_file_location("jones_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestJonesPolynomial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jones = load_jones()

    def test_unknot_is_one(self):
        self.assertEqual(self.jones.jones_polynomial((), 0), {Fraction(0): Fraction(1)})

    def test_trefoil(self):
        got = self.jones.jones_polynomial(self.jones.TREFOIL_PD, -3)
        self.assertEqual(got, {
            Fraction(-4): Fraction(-1),
            Fraction(-3): Fraction(1),
            Fraction(-1): Fraction(1),
        })

    def test_figure_eight(self):
        got = self.jones.jones_polynomial(self.jones.FIGURE_EIGHT_PD, 0)
        self.assertEqual(got, {
            Fraction(-2): Fraction(1),
            Fraction(-1): Fraction(-1),
            Fraction(0): Fraction(1),
            Fraction(1): Fraction(-1),
            Fraction(2): Fraction(1),
        })

    def test_mirror_inverts_exponents(self):
        base = self.jones.jones_polynomial(self.jones.TREFOIL_PD, -3)
        mirror = {(-e): c for e, c in base.items()}
        self.assertEqual(mirror, {
            Fraction(4): Fraction(-1),
            Fraction(3): Fraction(1),
            Fraction(1): Fraction(1),
        })


class TestBraidEngine(unittest.TestCase):
    """Names -> braid word -> generated geometry and Jones polynomial."""

    @classmethod
    def setUpClass(cls):
        from support.math import knots
        cls.knots = knots

    def test_trefoil_from_braid(self):
        got = self.knots.jones_from_braid((1, 1, 1), 2)
        self.assertEqual(got, {Fraction(1): Fraction(1), Fraction(3): Fraction(1),
                               Fraction(4): Fraction(-1)})
        self.assertEqual(self.knots.determinant_from_jones(got), 3)

    def test_figure_eight_from_braid(self):
        got = self.knots.jones_from_braid((1, -2, 1, -2), 3)
        self.assertEqual(got, {Fraction(-2): Fraction(1), Fraction(-1): Fraction(-1),
                               Fraction(0): Fraction(1), Fraction(1): Fraction(-1),
                               Fraction(2): Fraction(1)})
        self.assertEqual(self.knots.determinant_from_jones(got), 5)

    def test_every_catalogue_knot_generates(self):
        self.assertGreaterEqual(len(self.knots.KNOT_TABLE), 30)
        for key, entry in self.knots.KNOT_TABLE.items():
            poly = self.knots.jones_from_braid(entry["braid"], entry["strands"])
            self.assertTrue(poly, key)
            self.assertGreater(self.knots.determinant_from_jones(poly), 0, key)

    def test_name_aliases(self):
        for text in ("3_1", "3 1", "3-1", "3₁", "31", "Trefoil", "三叶结", "三叶"):
            self.assertEqual(self.knots.resolve(text)["key"], "3_1", text)
        self.assertEqual(self.knots.resolve("5₂")["key"], "5_2")
        self.assertEqual(self.knots.resolve("figure-eight")["key"], "4_1")
        self.assertEqual(self.knots.resolve("八字结")["key"], "4_1")

    def test_torus_notation(self):
        self.assertEqual(self.knots.resolve("T(2,5)")["key"], "5_1")
        self.assertEqual(self.knots.resolve("torus 3 4")["key"], "8_19")
        spec = self.knots.resolve("T(4,5)")
        self.assertEqual(spec["crossings"], 15)
        with self.assertRaises(self.knots.KnotError):
            self.knots.resolve("T(2,2)")   # Hopf link, not a knot

    def test_unknown_name_raises(self):
        with self.assertRaises(self.knots.KnotError):
            self.knots.resolve("definitely-not-a-knot")

    def test_curve_generation(self):
        for key in ("3_1", "5_2", "8_19", "T(3,5)"):
            spec = self.knots.resolve(key)
            curves = self.knots.braid_curves(spec["braid"], spec["strands"])
            self.assertEqual(len(curves), 1, key)
            self.assertGreater(len(curves[0]), 60, key)
            for x, y, z in curves[0]:
                self.assertLess(abs(x) + abs(y) + abs(z), 40.0, key)


if __name__ == "__main__":
    unittest.main()
