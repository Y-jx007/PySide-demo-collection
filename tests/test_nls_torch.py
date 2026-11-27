# -*- coding: utf-8 -*-
"""Nonlinear 的 torch FFT 后端：与 NumPy 结果一致，且没有 torch 时自动回退。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from support.physics.nls import NLSE  # noqa: E402
from support.physics.spm import TemporalNLSE  # noqa: E402


@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TorchBackendTests(unittest.TestCase):
    def test_nlse_torch_matches_numpy(self):
        reference = NLSE(size=64, seed=3)
        gpu = NLSE(size=64, seed=3, backend="torch")
        self.assertTrue(gpu.enable_torch("cpu"))
        self.assertTrue(gpu.warm_torch())
        for _ in range(2):
            reference.step(3)
            gpu.step(3)
        np.testing.assert_allclose(gpu.psi, reference.psi, atol=1e-10)
        self.assertAlmostEqual(gpu.power(), reference.power(), places=8)
        self.assertEqual(gpu.steps, reference.steps)

    def test_temporal_torch_matches_numpy(self):
        reference = TemporalNLSE(points=256, rows=64)
        gpu = TemporalNLSE(points=256, rows=64, backend="torch")
        self.assertTrue(gpu.enable_torch("cpu"))
        self.assertTrue(gpu.warm_torch())
        for _ in range(2):
            reference.step(2)
            gpu.step(2)
        np.testing.assert_allclose(gpu.field, reference.field, atol=1e-10)
        self.assertEqual(gpu.filled, reference.filled)

    def test_numpy_backend_is_default(self):
        sim = NLSE(size=32)
        self.assertFalse(sim.torch_ready)
        sim.step(1)
        self.assertEqual(sim.steps, 1)


if __name__ == "__main__":
    unittest.main()
