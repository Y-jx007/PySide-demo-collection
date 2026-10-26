"""Measure live Lloyd frame costs (step + paint + display refresh).

Usage (from the project root):

    python tools/profile_lloyd.py 2000 5
    python tools/profile_lloyd.py 10000 4

The first 1.5 seconds are treated as JIT/scipy warm-up and excluded.
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
sys.path.insert(0, str(APPS))
sys.path.insert(0, str(APPS / "Math"))

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import Lloyd  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", type=int, nargs="?", default=2000)
    parser.add_argument("seconds", type=float, nargs="?", default=5.0)
    parser.add_argument("--size", type=int, nargs=2, default=[1280, 760])
    args = parser.parse_args()

    Lloyd.setup_gl_format()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = Lloyd.VoronoiSimulation()
    window.resize(*args.size)
    window.show()

    # The constructor queues a reset with singleShot(1); let it run first.
    pre = QEventLoop()
    QTimer.singleShot(300, pre.quit)
    pre.exec()

    window.points_spin.setValue(args.count)
    window.reset_simulation()
    window.start_btn.click()

    stats = {"step": [], "paint": [], "refresh": []}
    frames = [0]
    started = time.perf_counter()

    def wrap(name, fn):
        def wrapper(*fn_args, **kwargs):
            begin = time.perf_counter()
            result = fn(*fn_args, **kwargs)
            if time.perf_counter() - started > 1.5:
                stats[name].append((time.perf_counter() - begin) * 1000.0)
            return result
        return wrapper

    window.sim.step = wrap("step", window.sim.step)
    window.voronoi_widget.paintGL = wrap("paint", window.voronoi_widget.paintGL)
    window._refresh_display = wrap("refresh", window._refresh_display)
    window.voronoi_widget.frameSwapped.connect(lambda: frames.__setitem__(0, frames[0] + 1))

    loop = QEventLoop()
    QTimer.singleShot(int(args.seconds * 1000), loop.quit)
    loop.exec()
    app.processEvents()

    for name, values in stats.items():
        if not values:
            print(f"{name:>8}: no samples")
            continue
        array = np.array(values)
        print(f"{name:>8}: n={len(array):4d} mean={array.mean():7.2f}ms "
              f"p50={np.percentile(array, 50):7.2f} "
              f"p95={np.percentile(array, 95):7.2f} max={array.max():7.2f}")
    if stats["step"]:
        print(f"steps/s: {len(stats['step']) / (args.seconds - 1.5):.1f}, "
              f"painted frames/s: {frames[0] / args.seconds:.1f}, "
              f"step+paint CPU: {100 * (sum(stats['step']) + sum(stats['paint'])) / (args.seconds * 1000):.0f}%")
    window.close()


if __name__ == "__main__":
    main()
