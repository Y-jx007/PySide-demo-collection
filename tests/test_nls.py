# -*- coding: utf-8 -*-
"""Numbers for the nonlinear Schrodinger demo: conservation and regimes."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.nls import (  # noqa: E402
    NLSE, PALETTE, PALETTES, PRESETS, apply_palette, palette_lut,
)


class TestPresets(unittest.TestCase):
    def test_preset_names_are_stable(self):
        self.assertEqual(PRESETS, ("单束", "涡旋", "双束", "噪声海"))

    def test_unknown_preset_is_rejected(self):
        sim = NLSE(size=48)
        with self.assertRaises(ValueError):
            sim.reset("不存在的预设")

    def test_reset_clears_the_step_counter(self):
        sim = NLSE(size=48)
        sim.step(5)
        self.assertEqual(sim.steps, 5)
        sim.reset()
        self.assertEqual(sim.steps, 0)

    def test_every_preset_starts_around_the_requested_amplitude(self):
        sim = NLSE(size=64)
        for preset in PRESETS:
            sim.reset(preset, 0.6, 0.0)
            self.assertAlmostEqual(float(np.max(np.abs(sim.psi))), 0.6, places=6)


class TestSplitStep(unittest.TestCase):
    def setUp(self):
        self.sim = NLSE(size=96)

    def test_power_is_conserved(self):
        self.sim.reset("单束", 0.8, 0.02)
        before = self.sim.power()
        self.sim.step(120)
        self.assertAlmostEqual(self.sim.power(), before, delta=1e-8 * before)

    def test_weak_beam_spreads(self):
        self.sim.reset("单束", 0.3, 0.0)
        peak, width = self.sim.peak(), self.sim.width()
        self.sim.step(300)
        self.assertLess(self.sim.peak(), peak)
        self.assertGreater(self.sim.width(), width * 1.4)

    def test_strong_beam_collapses(self):
        self.sim.reset("单束", 1.1, 0.0)
        peak = self.sim.peak()
        self.sim.step(300)
        self.assertGreater(self.sim.peak(), peak * 2.0)

    def test_saturation_keeps_the_peak_finite(self):
        self.sim.reset("单束", 1.15, 0.0)
        self.sim.step(900)
        self.assertLess(self.sim.peak(), 1e4)


class TestVortex(unittest.TestCase):
    def test_vortex_carries_a_hole_at_the_centre(self):
        sim = NLSE(size=96)
        sim.reset("涡旋", 0.6, 0.0)
        intensity = sim.intensity()
        centre = sim.n // 2
        self.assertLess(intensity[centre, centre], 1e-6 * intensity.max())
        profile = intensity[centre, centre:]
        radius = float(np.argmax(profile)) * sim.dx
        self.assertGreater(radius, 1.0)

    def test_charge_is_the_winding_number_around_the_core(self):
        sim = NLSE(size=96)
        sim.reset("涡旋", 0.6, 0.0)
        centre = sim.n // 2
        theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        ys = np.rint(centre + 10 * np.sin(theta)).astype(int)
        xs = np.rint(centre + 10 * np.cos(theta)).astype(int)
        ring = np.angle(sim.psi[ys, xs])
        steps = np.diff(np.append(ring, ring[0]))
        winding = float(np.sum(np.angle(np.exp(1j * steps)))) / (2.0 * np.pi)
        self.assertAlmostEqual(winding, 3.0, places=3)


class TestSeeding(unittest.TestCase):
    def test_adding_a_seed_raises_the_power(self):
        sim = NLSE(size=64)
        sim.reset("单束", 0.4, 0.0)
        before = sim.power()
        sim.add_seed(6.0, -4.0, 0.8)
        self.assertGreater(sim.power(), before)


class TestPalette(unittest.TestCase):
    def test_palette_lut_shape_and_endpoints(self):
        table = palette_lut(PALETTE)
        self.assertEqual(table.shape, (256, 3))
        stops = PALETTES[PALETTE]
        light, dark = stops[0][1], stops[-1][1]
        np.testing.assert_allclose(table[0].astype(int), light, atol=1)
        np.testing.assert_allclose(table[-1].astype(int), dark, atol=1)

    def test_render_shape_dtype_and_range(self):
        sim = NLSE(size=64)
        sim.reset("单束", 0.8, 0.01)
        sim.step(20)
        rgb = apply_palette(sim.intensity(), log=True)
        self.assertEqual(rgb.shape, (64, 64, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_percentile_scaling_keeps_contrast_with_a_spike(self):
        field = np.full((64, 64), 0.1)
        field[32, 32] = 1.0e6
        rgb = apply_palette(field, gamma=1.0)
        # 绝大多数像素仍在色阶中段，而不是被单个尖峰压成白纸
        self.assertLess(float(rgb.mean()), 250.0)

    def test_empty_field_falls_back_to_paper(self):
        rgb = apply_palette(np.zeros((8, 8)))
        self.assertEqual(rgb.shape, (8, 8, 3))
        np.testing.assert_allclose(rgb[0, 0].astype(int), PALETTES[PALETTE][0][1], atol=1)


if __name__ == "__main__":
    unittest.main()