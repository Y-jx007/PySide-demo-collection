"""Measure Quaternion Julia render cost and convergence/idle behaviour.

Usage (from the project root):

    python tools/profile_quaternion_julia.py
    python tools/profile_quaternion_julia.py --samples 1 --accum 24 --save shot.png

Requires a working Taichi GPU backend and a desktop session (QOpenGLWidget).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
APPS = ROOT / "apps"
APP_PATH = APPS / "Fractals" / "Quaternion Julia.py"
sys.path.insert(0, str(APPS))
sys.path.insert(0, str(APPS / "Fractals"))

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

spec = importlib.util.spec_from_file_location("QuaternionJulia", str(APP_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _wait(ms):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--accum", type=int, default=48)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    app = QApplication([])
    win = mod.MainWindow()
    win.resize(1100, 780)
    win.show()
    _wait(300)

    # Warm up Taichi/JIT before timing (first paint can take seconds).
    warm_start = time.perf_counter()
    while mod.acc_cnt[None] < 2 and time.perf_counter() - warm_start < 10.0:
        _wait(50)

    win.samp_s.setValue(args.samples)
    win.accum_s.setValue(args.accum)
    gl = win.gl
    gl.kick()

    times = []
    orig_paint = gl.paintGL
    start = time.perf_counter()

    def paint(*a):
        t = time.perf_counter()
        orig_paint(*a)
        times.append((time.perf_counter() - t) * 1000.0)

    gl.paintGL = paint
    while mod.acc_cnt[None] < gl.max_accum and time.perf_counter() - start < args.seconds:
        _wait(25)
    converged_at = time.perf_counter() - start
    gl.paintGL = orig_paint

    # Count paints while converged: should be 0.
    idle = [0]
    orig_paint = gl.paintGL

    def idle_paint(*a):
        idle[0] += 1
        orig_paint(*a)

    gl.paintGL = idle_paint
    _wait(1000)
    gl.paintGL = orig_paint

    if times:
        print(f"frames={len(times)} mean={sum(times) / len(times):.2f}ms "
              f"min={min(times):.2f} max={max(times):.2f}")
        print(f"converged at {converged_at:.2f}s with acc={mod.acc_cnt[None]}/"
              f"{gl.max_accum}, samples/frame={gl.samples}")
    else:
        print("no frames rendered")
    print(f"paints during 1s idle: {idle[0]}")
    if args.save:
        win.grab().save(args.save)
        print("saved", args.save)
    win.close()


if __name__ == "__main__":
    main()
