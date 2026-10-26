"""Benchmark the Lloyd / Voronoi relaxation app.

Run from the project root:

    python tools/bench_lloyd.py
    python tools/bench_lloyd.py --counts 500 2000 5000 10000

The app's window/GL is not created; only the simulation hot paths are timed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
APPS = ROOT / "apps"
sys.path.insert(0, str(APPS))
sys.path.insert(0, str(APPS / "Math"))

import numpy as np  # noqa: E402
from scipy.spatial import Voronoi  # noqa: E402

import Lloyd  # noqa: E402


def _median(fn, repeat):
    fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def benchmark(n, repeat):
    sim = Lloyd.HoneycombCVT(n_points=n, mobility=1.5)
    sim.step()  # warm Numba / scipy

    vor = Voronoi(sim.points)
    vor_ms = _median(lambda: Voronoi(sim.points), repeat)
    build_ms = _median(lambda: sim._build_cells(vor), repeat)
    centroid_ms = _median(lambda: Lloyd.compute_cell_centroids(
        sim._polygons_np, sim._offsets_np, sim._sizes_np, sim.centroids), repeat)
    move_ms = _median(lambda: Lloyd.move_points(
        sim.points, sim.centroids, float(sim.mobility),
        float(sim.xmin), float(sim.xmax), float(sim.ymin), float(sim.ymax)), repeat)
    sides_ms = _median(sim._compute_sides_numpy, repeat)
    step_ms = _median(sim.step, repeat)

    # Geometry that the renderer must build each frame.
    sides = sim._last_sides
    hex_mask = sides == 6
    other_mask = (sides != 6) & (sides > 0)
    hex_idx = Lloyd.VoronoiWidget._fan_indices(
        sim._offsets_np[hex_mask].astype(np.int32),
        sim._sizes_np[hex_mask].astype(np.int32))
    other_idx = Lloyd.VoronoiWidget._fan_indices(
        sim._offsets_np[other_mask].astype(np.int32),
        sim._sizes_np[other_mask].astype(np.int32))
    line_idx = Lloyd.VoronoiWidget._line_indices(
        np.ascontiguousarray(sim._offsets_np, dtype=np.int32),
        np.ascontiguousarray(sim._sizes_np, dtype=np.int32))
    idx_ms = _median(lambda: (
        Lloyd.VoronoiWidget._fan_indices(
            sim._offsets_np[hex_mask].astype(np.int32),
            sim._sizes_np[hex_mask].astype(np.int32)),
        Lloyd.VoronoiWidget._fan_indices(
            sim._offsets_np[other_mask].astype(np.int32),
            sim._sizes_np[other_mask].astype(np.int32)),
        Lloyd.VoronoiWidget._line_indices(
            np.ascontiguousarray(sim._offsets_np, dtype=np.int32),
            np.ascontiguousarray(sim._sizes_np, dtype=np.int32)),
    ), repeat)

    return {
        "n": n,
        "verts": len(sim._polygons_np),
        "voronoi": vor_ms,
        "build": build_ms,
        "centroid": centroid_ms,
        "move": move_ms,
        "sides": sides_ms,
        "step": step_ms,
        "indices": idx_ms,
        "triangles": (len(hex_idx) + len(other_idx)) // 3,
        "lines": len(line_idx) // 2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+",
                        default=[500, 2000, 5000, 10000])
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()

    header = (f"{'points':>7} {'verts':>7} {'voronoi':>9} {'build':>8} "
              f"{'centroid':>9} {'move':>7} {'sides':>7} {'step':>8} "
              f"{'indices':>8} {'tris':>8}")
    print(header)
    print("-" * len(header))
    for n in args.counts:
        row = benchmark(n, args.repeat)
        print(f"{row['n']:>7} {row['verts']:>7} {row['voronoi']:>8.2f}ms "
              f"{row['build']:>7.2f}ms {row['centroid']:>8.2f}ms "
              f"{row['move']:>6.2f}ms {row['sides']:>6.2f}ms "
              f"{row['step']:>7.2f}ms {row['indices']:>7.2f}ms "
              f"{row['triangles']:>8}")


if __name__ == "__main__":
    main()
