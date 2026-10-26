"""Benchmark the Maxwell-Boltzmann app's hot paths.

Run from the project root:

    python tools/bench_maxwell.py
    python tools/bench_maxwell.py --counts 2000 20000 200000 --repeat 5

Qt runs on the offscreen platform and only CPU-side simulation / plotting work
is timed.  The app keeps its render buffers in plain NumPy arrays, so the
simulation numbers include position packing and speed updates.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
APPS = ROOT / "apps"
sys.path.insert(0, str(APPS))
sys.path.insert(0, str(APPS / "Physics"))

import numpy as np  # noqa: E402
from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import Maxwell  # noqa: E402


def _median(fn, repeat):
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def benchmark_count(n, repeat, width, height):
    radius = max(0.1, 3.0 * math.sqrt(2000.0 / n))
    speed = 20.0 * math.sqrt(n / 2000.0)
    widget = Maxwell.GLSimulationWidget()
    widget.physical_width = width
    widget.physical_height = height
    widget.set_balls(n, radius, speed, width, height)

    create_ms = _median(lambda: widget.set_balls(n, radius, speed, width, height),
                        max(3, repeat // 2))
    widget.set_balls(n, radius, speed, width, height)
    widget.update_simulation()
    frame_ms = _median(widget.update_simulation, repeat)

    def collisions():
        n_occupied = Maxwell.build_cell_grid(
            widget.x, widget.y, widget.grid_w, widget.grid_h,
            widget.inv_cell_w, widget.inv_cell_h,
            widget.cell_ids, widget.cell_counts, widget.cell_cursor,
            widget.cell_starts, widget.cell_items, widget.cell_occupied)
        Maxwell.resolve_grid_collisions(
            widget.x, widget.y, widget.vx, widget.vy, widget.radius,
            widget.cell_starts, widget.cell_items, widget.cell_occupied,
            n_occupied, widget.grid_w, widget.grid_h)

    grid_build = _median(lambda: Maxwell.build_cell_grid(
        widget.x, widget.y, widget.grid_w, widget.grid_h,
        widget.inv_cell_w, widget.inv_cell_h,
        widget.cell_ids, widget.cell_counts, widget.cell_cursor,
        widget.cell_starts, widget.cell_items, widget.cell_occupied), repeat)
    collision_ms = _median(collisions, repeat) - grid_build

    def colors():
        widget._update_colors()

    color_ms = _median(colors, repeat)

    histogram = Maxwell.HistogramWidget()
    histogram.resize(520, 520)
    histogram.update_data(widget.speed_array)
    pixmap = QPixmap(histogram.size())
    hist_ms = _median(lambda: histogram.render(pixmap), max(3, repeat // 2))

    return {
        "n": n,
        "radius": radius,
        "speed": speed,
        "cells": widget.grid_w * widget.grid_h,
        "create": create_ms,
        "frame": frame_ms,
        "build": grid_build,
        "collisions": collision_ms,
        "colors": color_ms,
        "histogram": hist_ms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+",
                        default=[2000, 20000, 100000, 200000])
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    app = QApplication.instance() or QApplication([])
    print(f"window {args.width}x{args.height} | repeat {args.repeat}")
    header = (f"{'balls':>8} {'radius':>7} {'speed':>7} {'cells':>9} "
              f"{'create':>9} {'frame':>9} {'build':>9} {'collide':>9} "
              f"{'colors':>9} {'hist':>9}")
    print(header)
    print("-" * len(header))
    for n in args.counts:
        row = benchmark_count(n, args.repeat, args.width, args.height)
        print(f"{row['n']:>8} {row['radius']:>7.3f} {row['speed']:>7.1f} "
              f"{row['cells']:>9} {row['create']:>8.2f}ms {row['frame']:>8.3f}ms "
              f"{row['build']:>8.3f}ms {row['collisions']:>8.3f}ms "
              f"{row['colors']:>8.3f}ms {row['histogram']:>8.2f}ms")


if __name__ == "__main__":
    main()
