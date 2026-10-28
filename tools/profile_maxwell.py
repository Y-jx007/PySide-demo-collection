"""Measure live frame costs of the Maxwell app with a real window.

Usage (from the project root):

    python tools/profile_maxwell.py 2000 5
    python tools/profile_maxwell.py 200000 4

The app is shown normally, so this measures the real paint path (including
OpenGL buffer uploads).  The first second is treated as JIT warm-up and is not
included in the statistics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
APPS = ROOT / "apps"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APPS))

# 应用可能被启动器拖到别的分类，目录从 apps_manifest 里查
from apps_manifest import APPS as APP_ENTRIES  # noqa: E402

for _entry in APP_ENTRIES:
    if Path(_entry["path"]).stem == "Maxwell":
        sys.path.insert(0, str(APPS / Path(_entry["path"]).parent))
        break
else:
    sys.path.insert(0, str(APPS / "Physics"))

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import Maxwell  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", type=int, nargs="?", default=2000)
    parser.add_argument("seconds", type=float, nargs="?", default=5.0)
    parser.add_argument("--size", type=int, nargs=2, default=[1280, 760])
    args = parser.parse_args()

    fmt = Maxwell.QSurfaceFormat()
    fmt.setRenderableType(Maxwell.QSurfaceFormat.OpenGL)
    fmt.setProfile(Maxwell.QSurfaceFormat.CompatibilityProfile)
    Maxwell.QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = Maxwell.MaxwellBoltzmannApp()
    window.resize(*args.size)
    window.show()
    window.count_spin.setValue(args.count)

    simulation = window.simulation_widget
    stats = {"update": [], "paint": [], "histogram": []}
    started = time.perf_counter()

    def record(name, fn):
        def wrapper(*fn_args):
            begin = time.perf_counter()
            result = fn(*fn_args)
            if time.perf_counter() - started > 1.0:
                stats[name].append((time.perf_counter() - begin) * 1000.0)
            return result
        return wrapper

    simulation.update_simulation = record("update", simulation.update_simulation)
    simulation.paintGL = record("paint", simulation.paintGL)
    window.histogram_widget.paintEvent = record(
        "histogram", window.histogram_widget.paintEvent)

    loop = QEventLoop()
    QTimer.singleShot(int(args.seconds * 1000), loop.quit)
    loop.exec()
    app.processEvents()

    elapsed = args.seconds
    for name, values in stats.items():
        if not values:
            print(f"{name:>9}: no samples")
            continue
        array = np.array(values)
        print(f"{name:>9}: n={len(array):4d} mean={array.mean():7.2f}ms "
              f"p50={np.percentile(array, 50):7.2f} "
              f"p95={np.percentile(array, 95):7.2f} max={array.max():7.2f}")
    if stats["update"]:
        total = sum(stats["update"]) + sum(stats["paint"])
        print(f"CPU update+paint: {100 * total / (elapsed * 1000):.0f}% of wall time, "
              f"{len(stats['update']) / (elapsed - 1.0):.0f} updates/s")
    window.close()


if __name__ == "__main__":
    main()
