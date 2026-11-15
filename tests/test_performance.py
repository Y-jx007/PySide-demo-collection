# -*- coding: utf-8 -*-
"""Loose performance gates for core models.

These are guardrails against accidental complexity regressions, not
micro-benchmarks; CI machines are slower than the development GPU.
"""
import os
import sys
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from support.physics.mass_spring import MassSpring  # noqa: E402
from support.physics.particle_lenia import ParticleLenia  # noqa: E402
from support.physics.quasicrystal import Quasicrystal  # noqa: E402


class TestPerformance(unittest.TestCase):
    def test_particle_lenia_step_budget(self):
        sim = ParticleLenia()
        sim.reset("圆盘", 300)
        for _ in range(3):
            sim.step()
        start = time.perf_counter()
        for _ in range(5):
            sim.step()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0, f"particle 5 steps took {elapsed:.2f}s")

    def test_quasicrystal_render_budget(self):
        sim = Quasicrystal(256)
        start = time.perf_counter()
        sim.real_space()
        sim.diffraction(log=True)
        sim.render_real("冰蓝")
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 3.0, f"quasicrystal render took {elapsed:.2f}s")

    def test_mass_spring_step_budget(self):
        sim = MassSpring()
        sim.reset("悬索桥")
        start = time.perf_counter()
        for _ in range(100):
            sim.step()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0, f"100 mass-spring steps took {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
