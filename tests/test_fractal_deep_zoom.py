"""深度缩放的采样网格：越过 f64 精度后，色块必须仍然是正方形。

f64 在实轴/虚轴的量化步长不同（取决于数值量级），所以放大到约 1e14 倍后
原始采样会退化成"宽条"（一列 3~30 像素宽、1 像素高）。``sample_quantum``
把两轴对齐到同一张网格，下面的测试就是这条保证的回归检查。
"""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from common.dynamic_fractal import ViewTransform, sample_quantum

WIDTH, HEIGHT = 585, 506
CENTERS = (
    (-0.75, 0.0),
    (-0.77568377, 0.13646737),
    (-1.25, 0.02),
    (-0.743643887037151, 0.13182590420533),
)


def _kernel_samples(width, height, view_scale, center_x, center_y):
    """Replicate what the Taichi kernel evaluates for every pixel."""
    transform = ViewTransform(width, height, view_scale, center_x, center_y)
    columns = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    xs = columns * transform.scale_x + transform.offset_x
    ys = rows * transform.scale_y + transform.offset_y
    quantum = sample_quantum(width, height, view_scale, center_x, center_y)
    if quantum > 0.0:
        inv_q = 1.0 / quantum
        xs = (np.round(transform.offset_x * inv_q)
              + np.round(columns * (transform.scale_x * inv_q))) * quantum
        ys = (np.round(transform.offset_y * inv_q)
              + np.round(rows * (transform.scale_y * inv_q))) * quantum
    return xs, ys, quantum


def _mean_run(values):
    """Average length of runs of identical consecutive samples."""
    if len(values) < 2:
        return 1.0
    changes = np.count_nonzero(values[1:] != values[:-1]) + 1
    return len(values) / changes


class SampleQuantumTests(unittest.TestCase):
    def test_normal_zoom_keeps_exact_affine_samples(self):
        self.assertEqual(sample_quantum(WIDTH, HEIGHT, 1.0, -0.5, 0.3), 0.0)
        self.assertEqual(sample_quantum(WIDTH, HEIGHT, 1e-9, -0.75, 0.1), 0.0)
        self.assertEqual(sample_quantum(WIDTH, HEIGHT, 1e-13, -0.75, 0.1), 0.0)

    def test_quantum_starts_at_the_precision_limit(self):
        for cx, cy in CENTERS:
            self.assertEqual(sample_quantum(WIDTH, HEIGHT, 1e-13, cx, cy), 0.0)
            self.assertGreater(sample_quantum(WIDTH, HEIGHT, 1e-15, cx, cy), 0.0)

    def test_quantum_is_a_power_of_two_and_covers_the_view(self):
        for cx, cy in CENTERS:
            quantum = sample_quantum(WIDTH, HEIGHT, 1e-15, cx, cy)
            self.assertGreater(quantum, 0.0)
            log2 = math.log2(quantum)
            self.assertEqual(log2, math.floor(log2))

            half_w = (WIDTH / HEIGHT) * 1e-15
            half_h = 1e-15
            corner_ulp = max(
                math.ulp(cx - half_w), math.ulp(cx + half_w),
                math.ulp(cy - half_h), math.ulp(cy + half_h),
            )
            self.assertGreaterEqual(quantum, corner_ulp)

    def test_deep_zoom_blocks_are_square(self):
        # 1e-16 的视野只剩两三个网格，均值被边缘块主导，单独检查（见下）。
        for cx, cy in CENTERS:
            for view_scale in (1e-14, 1e-15):
                xs, ys, quantum = _kernel_samples(
                    WIDTH, HEIGHT, view_scale, cx, cy)
                self.assertGreater(quantum, 0.0)
                run_x = _mean_run(xs)
                run_y = _mean_run(ys)
                self.assertGreater(run_x, 1.5, "blocks should exist")
                self.assertLess(
                    abs(run_x - run_y) / max(run_x, run_y), 0.05,
                    f"block aspect {run_x / run_y:.2f} at {view_scale:.0e} "
                    f"c=({cx},{cy})")

    def test_extreme_zoom_samples_share_one_lattice(self):
        for cx, cy in CENTERS:
            xs, ys, quantum = _kernel_samples(WIDTH, HEIGHT, 1e-16, cx, cy)
            self.assertGreater(quantum, 0.0)
            # 每个采样值都落在同一张 q 网格上，两轴之间没有各用各的 ulp
            for values in (xs, ys):
                grid_index = values / quantum
                np.testing.assert_allclose(
                    grid_index, np.round(grid_index), rtol=0.0, atol=1e-9)

    def test_raw_samples_would_be_rectangles(self):
        """The un-quantized grid really is anisotropic - the bug being fixed."""
        transform = ViewTransform(WIDTH, HEIGHT, 1e-15, -0.75, 0.0)
        xs = np.arange(WIDTH) * transform.scale_x + transform.offset_x
        ys = np.arange(HEIGHT) * transform.scale_y + transform.offset_y
        self.assertGreater(_mean_run(xs) / _mean_run(ys), 5.0)

    def test_degenerate_inputs_are_safe(self):
        self.assertEqual(sample_quantum(0, 100, 1e-15, 0.0, 0.0), 0.0)
        self.assertEqual(sample_quantum(WIDTH, -1, 1e-15, 0.0, 0.0), 0.0)
        self.assertEqual(sample_quantum(WIDTH, HEIGHT, 0.0, 0.0, 0.0), 0.0)
        self.assertEqual(sample_quantum(WIDTH, HEIGHT, -1e-15, 0.0, 0.0), 0.0)
        self.assertEqual(
            sample_quantum(WIDTH, HEIGHT, float("inf"), 0.0, 0.0), 0.0)
        self.assertEqual(
            sample_quantum(WIDTH, HEIGHT, 1e-15, float("nan"), 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
