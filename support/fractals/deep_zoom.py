"""Arbitrary-precision deep zoom for the Mandelbrot set.

This is deliberately a CPU renderer: the GPU path uses f64 and loses
precision after roughly 1e14.  Here every pixel is iterated with
``decimal`` at a precision chosen by the caller, so the image stays valid
far past the double-precision limit.  It is slow, hence the low default
resolution and the progress callback.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class DeepZoomRequest:
    center_x: str
    center_y: str
    scale: str
    size: int = 200
    max_iter: int = 250
    precision: int = 40
    workers: int = 0


_ROOT = Path(__file__).resolve().parents[2]


def _render_rows(args):
    """Worker: render a contiguous block of scanlines and return bytes."""
    cx_s, cy_s, scale_s, size, max_iter, precision, y0, y1 = args
    from decimal import Decimal, localcontext

    with localcontext() as ctx:
        ctx.prec = max(20, int(precision))
        cx = Decimal(cx_s)
        cy = Decimal(cy_s)
        scale = Decimal(scale_s)
        two = Decimal(2)
        dx = two * scale / Decimal(size)
        half = Decimal(size) / two
        half_px = Decimal("0.5")
        rows = bytearray()
        for y in range(y0, y1):
            imag = cy - (Decimal(y) - half + half_px) * dx
            for x in range(size):
                real = cx + (Decimal(x) - half + half_px) * dx
                zr = Decimal(0)
                zi = Decimal(0)
                escaped = False
                for n in range(max_iter):
                    zr2 = zr * zr
                    zi2 = zi * zi
                    if zr2 + zi2 > 4:
                        escaped = True
                        break
                    zi = two * zr * zi + imag
                    zr = zr2 - zi2 + real
                value = int(255.0 * (n + 1) / max(1, max_iter)) if escaped else 255
                rows.extend((value, value, value))
        return y0, y1, bytes(rows)


def _run_chunk(job):
    payload = {
        "cx": job[0], "cy": job[1], "scale": job[2], "size": job[3],
        "max_iter": job[4], "precision": job[5], "y0": job[6], "y1": job[7],
    }
    proc = subprocess.run(
        [sys.executable, "-m", "support.fractals.deep_zoom_worker"],
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", "replace")[-800:]
        raise RuntimeError(f"deep zoom worker failed: {message}")
    expected = (job[7] - job[6]) * job[3] * 3
    if len(proc.stdout) != expected:
        raise RuntimeError(
            f"deep zoom worker returned {len(proc.stdout)} bytes, expected {expected}"
        )
    return job[6], job[7], proc.stdout


def render_deep(
    request: DeepZoomRequest,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> bytearray:
    """Render an RGB image at ``request.size`` square pixels.

    Work is split across a small pool of independent Python subprocesses.
    ``mpmath`` is CPU-bound and does not release the GIL, so threads would
    not help; subprocesses also avoid re-importing the Qt app on Windows.
    """
    size = max(32, int(request.size))
    max_iter = max(1, int(request.max_iter))
    precision = max(20, int(request.precision))
    workers = int(request.workers) or min(8, max(1, (os.cpu_count() or 2)))
    workers = max(1, min(workers, size))

    # Chunky rows keep process startup and IPC overhead small.
    chunk = max(1, size // (workers * 3))
    jobs = []
    for y0 in range(0, size, chunk):
        y1 = min(size, y0 + chunk)
        jobs.append((request.center_x, request.center_y, request.scale,
                     size, max_iter, precision, y0, y1))

    out = bytearray(size * size * 3)
    done = 0
    if workers == 1:
        for job in jobs:
            if cancelled and cancelled():
                break
            y0, y1, data = _render_rows(job)
            out[y0 * size * 3:y1 * size * 3] = data
            done += y1 - y0
            if progress:
                progress(done, size)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_chunk, job) for job in jobs]
        for future in as_completed(futures):
            if cancelled and cancelled():
                for f in futures:
                    f.cancel()
                break
            y0, y1, data = future.result()
            out[y0 * size * 3:y1 * size * 3] = data
            done += y1 - y0
            if progress:
                progress(done, size)
    return out
