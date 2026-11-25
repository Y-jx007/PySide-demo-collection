# -*- coding: utf-8 -*-
"""Extended-DLA growth-model invariants (no Qt window needed)."""
import importlib.util
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP = os.path.join(ROOT, "apps", "Fractals", "DLA.py")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location("dla_app", APP)
dla = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dla)


class TestDLA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = dla.DLASimulation(512, 600)
        cls.sim.set_walkers(1024)

    def setUp(self):
        self.sim.reset(512, 1, 0)
        self.sim.set_walkers(1024)

    def _grow(self, steps=30, substeps=128, stick=1.0, drift=0.0,
              aniso=0.0, kfold=5.0, relax=0.0, bias=0.0):
        for _ in range(steps):
            self.sim.step(512, substeps, stick, drift, aniso, kfold, relax, bias)

    def _count_radius(self):
        count, radius, _, _ = self.sim.stats()
        return count, radius

    def _density(self):
        count, radius = self._count_radius()
        return count / max(1.0, radius) ** 2

    def test_reset_state(self):
        occ = self.sim.occ.to_numpy()[:512, :512]
        self.assertEqual(int(np.count_nonzero(occ)), 1)
        self.assertEqual(occ[256, 256], 1)
        self.assertEqual(int(self.sim.steps[None]), 0)
        self.assertEqual(int(self.sim.max_r[None]), 0)

    def test_multi_nuclei_reset(self):
        self.sim.reset(512, 6, 16)
        occ = self.sim.occ.to_numpy()[:512, :512]
        self.assertGreaterEqual(int(np.count_nonzero(occ)), 6)
        self.assertEqual(int(self.sim.max_r[None]), 16)

    def test_growth_is_monotonic_and_count_exact(self):
        self._grow(20)
        first, _ = self._count_radius()
        self.assertGreater(first, 1)
        self._grow(20)
        second, _ = self._count_radius()
        self.assertGreater(second, first)
        occ = self.sim.occ.to_numpy()[:512, :512]
        self.assertEqual(second, int(np.count_nonzero(occ)))

    def test_max_radius_matches_occupied_cells(self):
        self._grow(30)
        occ = self.sim.occ.to_numpy()[:512, :512]
        ys, xs = np.nonzero(occ)
        radius = int(np.max(np.hypot(ys - 256, xs - 256)))
        self.assertEqual(int(self.sim.max_r[None]), radius)
        self.assertLessEqual(radius, 254)

    def test_low_stick_grows_denser(self):
        self._grow(40, substeps=128, stick=1.0)
        open_density = self._density()
        self.sim.reset(512, 1, 0)
        self._grow(40, substeps=128, stick=0.2)
        self.assertGreater(self._density(), open_density)

    def test_relaxation_fills_the_cluster(self):
        self._grow(40, substeps=128)
        rough_density = self._density()
        self.sim.reset(512, 1, 0)
        self._grow(40, substeps=128, relax=1.0)
        self.assertGreater(self._density(), rough_density)

    def test_inward_bias_compacts_the_cluster(self):
        self._grow(40, substeps=128)
        free_density = self._density()
        self.sim.reset(512, 1, 0)
        self._grow(40, substeps=128, bias=0.5)
        self.assertGreater(self._density(), free_density)

    def test_anisotropy_breaks_the_round_envelope(self):
        def envelope_cv(sim):
            occ = sim.occ.to_numpy()[:512, :512]
            ys, xs = np.nonzero(occ)
            ang = np.arctan2(ys - 256, xs - 256)
            rad = np.hypot(ys - 256, xs - 256)
            idx = np.digitize(ang, np.linspace(-np.pi, np.pi, 49)) - 1
            prof = np.zeros(48)
            for b in range(48):
                mask = idx == b
                if mask.any():
                    prof[b] = rad[mask].max()
            prof = prof[prof > 0]
            return float(prof.std() / max(1e-6, prof.mean()))

        self._grow(60, substeps=128, aniso=0.0)
        round_cv = envelope_cv(self.sim)
        self.sim.reset(512, 1, 0)
        self._grow(60, substeps=128, aniso=0.9, kfold=6.0)
        self.assertGreater(envelope_cv(self.sim), round_cv)

    def test_grid_switch_leaves_no_stale_cells(self):
        self._grow(15)
        self.sim.reset(2048, 1, 0)
        self.assertEqual(int(self.sim.stats()[0]), 1)
        self.sim.reset(512, 1, 0)
        occ = self.sim.occ.to_numpy()[:512, :512]
        self.assertEqual(int(np.count_nonzero(occ)), 1)

    def test_render_is_white_outside_and_coloured_inside(self):
        self._grow(30)
        radius = int(self.sim.max_r[None])
        self.sim.render(512, 600, 256.0, 256.0, radius * 1.16 + 8.0, 2, 0.5, 0.04, 0.28)
        img = self.sim.pix.to_numpy()
        self.assertEqual(img.shape, (600, 600, 3))
        self.assertTrue(np.all(img[0, 0] == 255))
        inner = img[300 - 40:300 + 40, 300 - 40:300 + 40]
        self.assertLess(int(inner.min()), 200)

    def test_palette_has_no_banding(self):
        lut = dla.palette_lut()
        self.assertEqual(lut.shape, (256, 3))
        diff = np.abs(np.diff(lut.astype(np.int16), axis=0))
        self.assertLessEqual(int(diff.max()), 8)
        self.assertGreater(len(np.unique(lut, axis=0)), 64)

    def test_presets_are_complete_and_distinct(self):
        self.assertEqual(len(dla.PRESETS), 8)
        keys = {"walkers", "speed", "stick", "aniso", "kfold", "relax",
                "bias", "drift", "seeds", "seed_r", "view"}
        for name, cfg in dla.PRESETS.items():
            self.assertEqual(set(cfg), keys, name)
            self.assertIn(cfg["kfold"], dla.ORDERS, name)
            self.assertGreaterEqual(cfg["seeds"], 1, name)
            self.assertGreater(cfg["walkers"], 0, name)
            self.assertGreater(cfg["view"], 0, name)
        # the family must not collapse into the same physics
        signatures = {tuple(sorted(cfg.items())) for cfg in dla.PRESETS.values()}
        self.assertEqual(len(signatures), 8)
        self.assertNotEqual(dla.PRESETS["珊瑚"]["aniso"], dla.PRESETS["地衣"]["aniso"])
        self.assertNotEqual(dla.PRESETS["水晶"]["kfold"], dla.PRESETS["海扇"]["kfold"])
        self.assertGreater(dla.PRESETS["玛瑙"]["bias"], 0)
        self.assertGreater(dla.PRESETS["海藻"]["drift"], 0)


if __name__ == "__main__":
    unittest.main()
