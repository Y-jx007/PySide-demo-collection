# -*- coding: utf-8 -*-
"""Arbitrary-precision deep zoom renderer."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from support.fractals.deep_zoom import DeepZoomRequest, render_deep  # noqa: E402


class DeepZoomTests(unittest.TestCase):
    def test_render_shape_and_variation(self):
        data = render_deep(
            DeepZoomRequest("0", "0", "1.5", size=32, max_iter=60,
                            precision=30, workers=1)
        )
        self.assertEqual(len(data), 32 * 32 * 3)
        self.assertGreater(max(data), min(data))

    def test_deep_scale_is_still_finite(self):
        data = render_deep(
            DeepZoomRequest("-0.7436438870371587", "0.1318259042053119",
                            "1e-20", size=32, max_iter=120,
                            precision=40, workers=1)
        )
        self.assertEqual(len(data), 32 * 32 * 3)
        self.assertTrue(all(0 <= value <= 255 for value in data))


if __name__ == "__main__":
    unittest.main()
