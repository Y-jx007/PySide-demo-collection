# -*- coding: utf-8 -*-
"""DLA growth-model invariants (no Qt window needed)."""
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
              aniso=0.0, kfold=5.0):
        for _ in range(steps):
            self.sim.step(512, substeps, stick, drift, aniso, kfold)

    def _count_radius(self):
        count, radius, _, _ = self.sim.stats()
        return count, radius

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
        n_open, r_open = self._count_radius()
        self.sim.reset(512, 1, 0)
        self._grow(40, substeps=128, stick=0.2)
        n_dense, r_dense = self._count_radius()
        # Non-sticking walkers slide along the surface instead of waiting on
        # the same site, so the cluster fills in and its fractal dimension
        # rises even though fewer particles arrive.
        open_density = n_open / max(1.0, r_open) ** 2
        dense_density = n_dense / max(1.0, r_dense) ** 2
        self.assertGreater(dense_density, open_density)

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
        lobed_cv = envelope_cv(self.sim)
        self.assertGreater(lobed_cv, round_cv)

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

    def test_presets_use_different_physics(self):
        coral = dla.PRESETS["珊瑚"]
        lichen = dla.PRESETS["地衣"]
        dendrite = dla.PRESETS["枝晶"]
        self.assertNotEqual(coral["stick"], lichen["stick"])
        self.assertNotEqual(coral["aniso"], dendrite["aniso"])
        self.assertGreater(lichen["seeds"], 1)
        self.assertGreater(dendrite["aniso"], coral["aniso"])


if __name__ == "__main__":
    unittest.main()
