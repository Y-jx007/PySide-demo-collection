"""Checks for the pre-temperature Circulation model.

The synthetic pressure field is intentionally the old analytic one: the
Antarctic land surface is a low, not the temperature-driven polar high.
"""
import importlib.util
import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _app_paths import add_all_app_paths, app_dir  # noqa: E402

add_all_app_paths()


def _load_app():
    # Circulation 可能被拖到任意分类，路径从清单里取
    path = os.path.join(app_dir("Circulation", "Physics"), "Circulation.py")
    spec = importlib.util.spec_from_file_location("circulation_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSyntheticCirculation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_app()
        cls.wind = cls.mod.WindSim()

    def test_antarctic_land_is_low_pressure(self):
        self.wind.update(7.0, 1.5)
        pressure = self.wind.pressure()
        south_pole = float(pressure[0].mean())       # row 0 = -90
        southern_ocean = float(pressure[10:20].mean())
        self.assertLess(south_pole, southern_ocean - 5.0)


if __name__ == "__main__":
    unittest.main()