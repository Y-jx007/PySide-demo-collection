# -*- coding: utf-8 -*-
"""Regenerate deterministic canvas baselines for visual regression tests."""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "apps")):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from tests.visual_utils import capture_canvas

CASES = (
    ("apps/Math/Quasicrystal.py", "Quasicrystal.png"),
    ("apps/Physics/Mass-Spring.py", "Mass-Spring.png"),
)


def main():
    app = QApplication.instance() or QApplication([])
    out = os.path.join(ROOT, "tests", "baselines")
    os.makedirs(out, exist_ok=True)
    for relative, name in CASES:
        spec = importlib.util.spec_from_file_location(
            "baseline_" + name.split(".")[0], os.path.join(ROOT, relative)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        window = module.MainWindow()
        if hasattr(window, "timer"):
            window.timer.stop()
        if name == "Mass-Spring.png":
            window.sim.reset("悬索桥")
        window.show()
        app.processEvents()
        pix = capture_canvas(window, app)
        pix.save(os.path.join(out, name))
        window.close()
        print("saved", name, pix.width(), pix.height())


if __name__ == "__main__":
    main()
