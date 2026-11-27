# -*- coding: utf-8 -*-
"""Newton 修复的回归检查：临界点附近不能再出现黑圆盘。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KERNELS = ROOT / "support" / "fractals" / "fractal_kernels.py"


def _converges(z, max_iter=300, threshold=1e-6):
    """Pure-python mirror of the kernel's Newton loop for z^4 - 1."""
    for _ in range(max_iter):
        f = z ** 4 - 1.0
        df = 4.0 * z ** 3
        denom = df.real * df.real + df.imag * df.imag
        if denom <= 0.0:
            return False
        if abs(f) < threshold:
            return True
        z = z - f / df
    return False


class NewtonStepTests(unittest.TestCase):
    def test_points_inside_the_old_black_disk_converge(self):
        # 旧实现把 |df|² < 1e-12 的除法钳成 0，|z| < 6.3e-3 的整片区域
        # 原地打转、最后画成黑圆盘；这些点其实会跳远再收敛到某个根。
        for z in (0.005, 0.005 + 0.002j, 0.002 - 0.004j, -0.004 + 0.001j):
            self.assertTrue(_converges(z), f"{z} should converge")

    def test_critical_point_is_handled_without_a_crash(self):
        # z = 0 处 df = 0，除法产生 inf/nan，但循环必须安全结束而不是抛异常。
        self.assertFalse(_converges(0j))

    def test_newton_step_uses_the_unclamped_division(self):
        source = KERNELS.read_text(encoding="utf-8")
        newton = source[source.index("def render_newton("):]
        self.assertIn("cdiv_exact(f_z, df_z)", newton)
        self.assertNotIn("cdiv(f_z, df_z)", newton)
        self.assertNotIn("< 1e-16", newton)

    def test_edge_supersampling_uses_the_same_step(self):
        source = KERNELS.read_text(encoding="utf-8")
        gray_loop = source[source.index("def _newton_gray("):]
        self.assertIn("cdiv_exact(f_z, df_z)", gray_loop)
        self.assertNotIn("cdiv(f_z, df_z)", gray_loop)


if __name__ == "__main__":
    unittest.main()
