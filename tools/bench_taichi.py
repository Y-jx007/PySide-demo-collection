# -*- coding: utf-8 -*-
"""Small reproducible Taichi Forge benchmark for the project's CPU kernels.

Run before/after upgrading taichi-forge:

    python tools/bench_taichi.py --out runtime/bench_taichi_before.json

Measures first-launch compile time and steady-state kernel time.  The Lorenz
kernel mirrors apps/Math/Lorenz.py; the Mandelbrot kernel mirrors the fractal
renderer's 2-D workload.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import taichi_forge as ti

ROOT = Path(__file__).resolve().parents[1]
_TIER = os.environ.get("TAICHI_COMPILE_TIER", "fast")
_ADVANCED = os.environ.get("TAICHI_ADVANCED", "0") == "1"
ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, debug=False,
        compile_tier=_TIER, advanced_optimization=_ADVANCED)


@ti.func
def deriv_lorenz(x, y, z, p0, p1, p2, p3, p4, p5):
    sigma, rho, beta = p0, p1, p2
    return ti.math.vec3(
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    )


@ti.kernel
def lorenz_steps(
    x_in: ti.f64, y_in: ti.f64, z_in: ti.f64,
    p0: ti.f64, p1: ti.f64, p2: ti.f64,
    p3: ti.f64, p4: ti.f64, p5: ti.f64,
    dt: ti.f64, n: ti.i32,
    out: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    x, y, z = x_in, y_in, z_in
    for i in range(n):
        k1 = deriv_lorenz(x, y, z, p0, p1, p2, p3, p4, p5)
        k2 = deriv_lorenz(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1],
                          z + 0.5 * dt * k1[2], p0, p1, p2, p3, p4, p5)
        k3 = deriv_lorenz(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1],
                          z + 0.5 * dt * k2[2], p0, p1, p2, p3, p4, p5)
        k4 = deriv_lorenz(x + dt * k3[0], y + dt * k3[1],
                          z + dt * k3[2], p0, p1, p2, p3, p4, p5)
        x = x + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y = y + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z = z + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = z


@ti.kernel
def mandelbrot(
    width: ti.i32, height: ti.i32, max_iter: ti.i32,
    out: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    for i, j in ti.ndrange(width, height):
        cr = (ti.cast(i, ti.f64) / width - 0.75) * 2.6
        ci = (ti.cast(j, ti.f64) / height - 0.5) * 2.6
        zr = 0.0
        zi = 0.0
        it = 0
        while it < max_iter and zr * zr + zi * zi <= 4.0:
            zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
            it += 1
        out[i, j] = ti.cast(it, ti.f32) / ti.cast(max_iter, ti.f32)


def timed(fn, args, reps=5):
    fn(*args)
    ti.sync()
    times = []
    for _ in range(reps):
        ti.sync()
        t0 = time.perf_counter()
        fn(*args)
        ti.sync()
        times.append(time.perf_counter() - t0)
    return times


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    lorenz_out = np.zeros((250_000, 3), dtype=np.float64)
    lorenz_args = (0.1, 0.0, 0.0, 10.0, 28.0, 8.0 / 3.0,
                   0.0, 0.0, 0.0, 0.005, 250_000, lorenz_out)
    t0 = time.perf_counter()
    lorenz_steps(*lorenz_args)
    ti.sync()
    lorenz_compile = time.perf_counter() - t0
    lorenz_times = timed(lorenz_steps, lorenz_args, args.reps)

    fractal_out = np.zeros((512, 512), dtype=np.float32)
    fractal_args = (512, 512, 256, fractal_out)
    t0 = time.perf_counter()
    mandelbrot(*fractal_args)
    ti.sync()
    fractal_compile = time.perf_counter() - t0
    fractal_times = timed(mandelbrot, fractal_args, args.reps)

    result = {
        "taichi_forge": getattr(ti, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "arch": "cpu",
        "compile_tier": _TIER,
        "advanced_optimization": _ADVANCED,
        "workloads": {
            "lorenz_250k_rk4": {
                "compile_s": lorenz_compile,
                "runtime_s": lorenz_times,
                "median_s": statistics.median(lorenz_times),
            },
            "mandelbrot_512x512_256": {
                "compile_s": fractal_compile,
                "runtime_s": fractal_times,
                "median_s": statistics.median(fractal_times),
            },
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
