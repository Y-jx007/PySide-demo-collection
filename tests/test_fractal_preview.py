# -*- coding: utf-8 -*-
"""分形画布的交互预览：交互中 1x，停手后按用户选择的 AA 精修。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dynamic_fractal import (  # noqa: E402
    PREVIEW_ANTIALIAS,
    PREVIEW_REFINE_MS,
    render_dimensions,
)


class RenderDimensionsTests(unittest.TestCase):
    def test_preview_uses_single_sample(self):
        self.assertEqual(PREVIEW_ANTIALIAS, 1)
        self.assertEqual(render_dimensions(600, 400, 4, preview=True), (600, 400, 600, 400))

    def test_full_quality_uses_selected_factor(self):
        self.assertEqual(render_dimensions(600, 400, 4), (600, 400, 2400, 1600))
        self.assertEqual(render_dimensions(585, 506, 2), (585, 506, 1170, 1012))

    def test_degenerate_sizes_are_clamped(self):
        self.assertEqual(render_dimensions(0, 0, 0), (1, 1, 1, 1))
        self.assertEqual(render_dimensions(-5, 10, 4, preview=True), (1, 10, 1, 10))

    def test_refine_delay_is_short_but_visible(self):
        self.assertGreaterEqual(PREVIEW_REFINE_MS, 120)
        self.assertLessEqual(PREVIEW_REFINE_MS, 400)


if __name__ == "__main__":
    unittest.main()
