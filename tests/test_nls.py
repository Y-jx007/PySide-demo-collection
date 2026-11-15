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


class TestModels(unittest.TestCase):
    def test_response_is_one_at_weak_light(self):
        sim = NLSE(size=64)
        for model in ("饱和", "高阶 Kerr"):
            sim.model = model
            self.assertAlmostEqual(float(sim.response(0.0)), 1.0, places=9)

    def test_saturable_response_falls_as_one_over_one_plus_s(self):
        sim = NLSE(size=64)
        sim.model = "饱和"
        value = float(sim.response(np.array([sim.saturation]))[0])
        self.assertAlmostEqual(value, 0.5, places=9)

    def test_kerr5_response_shuts_off_beyond_one_over_kappa(self):
        sim = NLSE(size=64)
        sim.model = "高阶 Kerr"
        sim.kappa = 0.02
        self.assertAlmostEqual(float(sim.response(np.array([25.0]))[0]), 0.5, places=9)
        self.assertEqual(float(sim.response(np.array([80.0]))[0]), 0.0)

    def test_kerr5_defocuses_where_the_saturable_model_collapses(self):
        sim = NLSE(size=96)
        sim.model = "高阶 Kerr"
        sim.kappa = 0.03
        sim.reset("单束", 1.0, 0.0)
        peak = sim.peak()
        sim.step(300)
        kerr5_peak = sim.peak()
        sim.model = "饱和"
        sim.reset("单束", 1.0, 0.0)
        sim.step(300)
        self.assertLess(kerr5_peak, 2.5 * peak)
        self.assertGreater(sim.peak(), 2.5 * peak)


class TestSpectrum(unittest.TestCase):
    def test_spectrum_shape_and_dtype(self):
        sim = NLSE(size=64)
        spec = sim.render_spectrum()
        self.assertEqual(spec.shape, (64, 64, 3))
        self.assertEqual(spec.dtype, np.uint8)
        norm = sim.spectrum()
        self.assertGreaterEqual(float(norm.min()), 0.0)
        self.assertLessEqual(float(norm.max()), 1.0)

    def test_spectrum_of_a_standing_wave_has_two_peaks(self):
        sim = NLSE(size=64, extent=32.0)
        k = 2.0 * np.pi * 4 / sim.extent
        sim.psi = (np.exp(1j * k * sim.X) + np.exp(-1j * k * sim.X)).astype(complex)
        norm = sim.spectrum(floor_db=60.0)
        # 能量集中在两行谱线上：中心行之外几乎全暗
        centre = sim.n // 2
        bright = np.argwhere(norm > 0.85)
        self.assertGreater(len(bright), 0)
        rows = set(int(row) for row, _ in bright)
        self.assertLessEqual(len(rows), 4)
        self.assertIn(centre, rows)


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

    def test_log_view_lifts_faint_structure(self):
        # 一个很弱的像素在线性档几乎看不见，在 dB 档要被抬进色阶
        field = np.zeros((64, 64))
        field[32, 32] = 1.0
        field[10, 10] = 3.0e-2          # −15 dB：线性档几乎白，dB 档落到中段
        linear = apply_palette(field, log=False)
        log = apply_palette(field, log=True)
        self.assertGreater(int(linear[10, 10].mean()) - int(log[10, 10].mean()), 60)
        # 峰值两端仍要区分：中心比外围深
        self.assertLess(int(log[32, 32].mean()), int(log[10, 10].mean()))

    def test_empty_field_falls_back_to_paper(self):
        rgb = apply_palette(np.zeros((8, 8)))
        self.assertEqual(rgb.shape, (8, 8, 3))
        np.testing.assert_allclose(rgb[0, 0].astype(int), PALETTES[PALETTE][0][1], atol=1)


class TestSaturableAbsorption(unittest.TestCase):
    def loss(self, amplitude, coefficient=0.08, steps=60):
        sim = NLSE()
        sim.reset("单束", amplitude, 0.0)
        before = sim.power()
        sim.saturable_loss = coefficient
        sim.step(steps)
        return 1.0 - sim.power() / before

    def test_absorption_removes_power(self):
        self.assertGreater(self.loss(0.6), 0.0)

    def test_brighter_beam_is_more_transparent(self):
        weak = self.loss(0.3)
        strong = self.loss(0.9)
        self.assertLess(strong, weak)


class TestTwoPhoton(unittest.TestCase):
    def simulate(self, amplitude, coefficient, steps=120):
        sim = NLSE()
        sim.reset("单束", amplitude, 0.0)
        sim.two_photon = coefficient
        sim.step(steps)
        return sim.power()

    def test_two_photon_absorption_removes_power(self):
        plain = self.simulate(0.8, 0.0)
        lossy = self.simulate(0.8, 0.05)
        self.assertLess(lossy, plain)

    def test_loss_grows_with_intensity(self):
        weak = 1.0 - self.simulate(0.4, 0.05) / self.simulate(0.4, 0.0)
        strong = 1.0 - self.simulate(0.8, 0.05) / self.simulate(0.8, 0.0)
        self.assertGreater(strong, weak)


if __name__ == "__main__":
    unittest.main()