# -*- coding: utf-8 -*-
"""Planar Sun–Jupiter circular restricted three-body problem.

Coordinates rotate with the two primaries.  The Sun is at x = -mu and
Jupiter at x = 1 - mu, with mu = 0.000953875.  The Poincaré section is
recorded at y = 0 crossings with vy > 0, in the (x, vx) plane.
"""
import numpy as np

try:
    from support.physics.kam_taichi import taichi_available, ti
except Exception:
    taichi_available = lambda: False
    ti = None

MU = 0.000953875
DT = 0.1
X_MIN = 0.60
X_MAX = 1.40
C_DEFAULT = 3.0
_TAI = bool(taichi_available())
_FIELDS = {}
_KERNEL = None


def crtbp_derivative(x, y, vx, vy):
    """Rotating-frame acceleration (planar CRTBP)."""
    r1 = np.sqrt((x + MU) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1.0 + MU) ** 2 + y ** 2)
    ax = (
        2.0 * vy + x
        - (1.0 - MU) * (x + MU) / r1 ** 3
        - MU * (x - 1.0 + MU) / r2 ** 3
    )
    ay = (
        -2.0 * vx + y
        - (1.0 - MU) * y / r1 ** 3
        - MU * y / r2 ** 3
    )
    return ax, ay


def _rk4_step(x, y, vx, vy, dt=DT):
    ax1, ay1 = crtbp_derivative(x, y, vx, vy)
    ax2, ay2 = crtbp_derivative(
        x + 0.5 * dt * vx, y + 0.5 * dt * vy,
        vx + 0.5 * dt * ax1, vy + 0.5 * dt * ay1,
    )
    ax3, ay3 = crtbp_derivative(
        x + 0.5 * dt * (vx + 0.5 * dt * ax1),
        y + 0.5 * dt * (vy + 0.5 * dt * ay1),
        vx + 0.5 * dt * ax2, vy + 0.5 * dt * ay2,
    )
    ax4, ay4 = crtbp_derivative(
        x + dt * (vx + 0.5 * dt * ax2),
        y + dt * (vy + 0.5 * dt * ay2),
        vx + dt * ax3, vy + dt * ay3,
    )
    nx = x + dt / 6.0 * (vx + 2.0 * (vx + 0.5 * dt * ax1)
                         + 2.0 * (vx + 0.5 * dt * ax2)
                         + (vx + dt * ax3))
    # Recompute positions from the k values consistently.
    k1x = vx
    k1y = vy
    k1vx = ax1
    k1vy = ay1
    k2x = vx + 0.5 * dt * k1vx
    k2y = vy + 0.5 * dt * k1vy
    k2vx = ax2
    k2vy = ay2
    k3x = vx + 0.5 * dt * k2vx
    k3y = vy + 0.5 * dt * k2vy
    k3vx = ax3
    k3vy = ay3
    k4x = vx + dt * k3vx
    k4y = vy + dt * k3vy
    k4vx = ax4
    k4vy = ay4
    nx = x + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    ny = y + dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    nvx = vx + dt / 6.0 * (k1vx + 2.0 * k2vx + 2.0 * k3vx + k4vx)
    nvy = vy + dt / 6.0 * (k1vy + 2.0 * k2vy + 2.0 * k3vy + k4vy)
    return nx, ny, nvx, nvy


def poincare_numpy(C=C_DEFAULT, n_orbits=60, steps=3000):
    """Reference NumPy implementation of the Poincaré section."""
    points = []
    for i in range(int(n_orbits)):
        x0 = X_MIN + (X_MAX - X_MIN) * i / max(1, int(n_orbits) - 1)
        y0 = -0.001
        vx0 = 0.0
        x, y, vx, vy = x0, y0, vx0, 0.0
        r1 = np.hypot(x + MU, y)
        r2 = np.hypot(x - 1.0 + MU, y)
        omega = 0.5 * (x * x + y * y) + (1.0 - MU) / r1 + MU / r2
        vsq = 2.0 * omega - C
        if vsq <= 0.0:
            continue
        vy = float(np.sqrt(vsq))
        prev_x, prev_y, prev_vx = x, y, vx
        for _ in range(int(steps)):
            x, y, vx, vy = _rk4_step(x, y, vx, vy)
            if prev_y < 0.0 and y >= 0.0 and vy > 0.0:
                frac = -prev_y / (y - prev_y) if y != prev_y else 0.0
                points.append((
                    prev_x + frac * (x - prev_x),
                    prev_vx + frac * (vx - prev_vx),
                ))
            prev_x, prev_y, prev_vx = x, y, vx
    return np.asarray(points, dtype=np.float64)


def _ensure_kernel():
    global _KERNEL
    if not _TAI or ti is None:
        return False
    if _KERNEL is not None:
        return True
    try:
        @ti.func
        def accel(x, y, vx, vy):
            r1 = ti.sqrt((x + MU) * (x + MU) + y * y)
            r2 = ti.sqrt((x - 1.0 + MU) * (x - 1.0 + MU) + y * y)
            ax = (2.0 * vy + x
                  - (1.0 - MU) * (x + MU) / (r1 * r1 * r1)
                  - MU * (x - 1.0 + MU) / (r2 * r2 * r2))
            ay = (-2.0 * vx + y
                  - (1.0 - MU) * y / (r1 * r1 * r1)
                  - MU * y / (r2 * r2 * r2))
            return ax, ay

        @ti.kernel
        def kernel(C: ti.f32, n: ti.i32, steps: ti.i32,
                   xs: ti.template(), vxs: ti.template(),
                   counts: ti.template()):
            for i in range(n):
                x = X_MIN + (X_MAX - X_MIN) * i / (n - 1)
                y = -0.001
                vx = 0.0
                vy = 0.0
                r1 = ti.sqrt((x + MU) * (x + MU) + y * y)
                r2 = ti.sqrt((x - 1.0 + MU) * (x - 1.0 + MU) + y * y)
                omega = 0.5 * (x * x + y * y) + (1.0 - MU) / r1 + MU / r2
                vsq = 2.0 * omega - C
                count = 0
                if vsq > 0.0:
                    vy = ti.sqrt(vsq)
                    prev_x, prev_y, prev_vx = x, y, vx
                    for _ in range(steps):
                        ax1, ay1 = accel(x, y, vx, vy)
                        ax2, ay2 = accel(
                            x + 0.5 * DT * vx,
                            y + 0.5 * DT * vy,
                            vx + 0.5 * DT * ax1,
                            vy + 0.5 * DT * ay1,
                        )
                        ax3, ay3 = accel(
                            x + 0.5 * DT * (vx + 0.5 * DT * ax1),
                            y + 0.5 * DT * (vy + 0.5 * DT * ay1),
                            vx + 0.5 * DT * ax2,
                            vy + 0.5 * DT * ay2,
                        )
                        ax4, ay4 = accel(
                            x + DT * (vx + 0.5 * DT * ax2),
                            y + DT * (vy + 0.5 * DT * ay2),
                            vx + DT * ax3,
                            vy + DT * ay3,
                        )
                        k1x, k1y = vx, vy
                        k2x = vx + 0.5 * DT * ax1
                        k2y = vy + 0.5 * DT * ay1
                        k3x = vx + 0.5 * DT * ax2
                        k3y = vy + 0.5 * DT * ay2
                        k4x = vx + DT * ax3
                        k4y = vy + DT * ay3
                        nx = x + DT / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
                        ny = y + DT / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
                        nvx = vx + DT / 6.0 * (
                            ax1 + 2.0 * ax2 + 2.0 * ax3 + ax4
                        )
                        nvy = vy + DT / 6.0 * (
                            ay1 + 2.0 * ay2 + 2.0 * ay3 + ay4
                        )
                        if prev_y < 0.0 and ny >= 0.0 and nvy > 0.0:
                            frac = (-prev_y / (ny - prev_y)
                                    if ny != prev_y else 0.0)
                            xs[i, count] = prev_x + frac * (nx - prev_x)
                            vxs[i, count] = prev_vx + frac * (nvx - prev_vx)
                            count += 1
                            if count >= steps:
                                break
                        prev_x, prev_y, prev_vx = nx, ny, nvx
                        x, y, vx, vy = nx, ny, nvx, nvy
                counts[i] = count

        _KERNEL = kernel
        return True
    except Exception:
        _KERNEL = None
        return False


def poincare_taichi(C=C_DEFAULT, n_orbits=60, steps=3000):
    """Taichi parallel section; returns None when unavailable."""
    if not _TAI or not _ensure_kernel():
        return None
    n = max(8, int(n_orbits))
    steps = max(100, int(steps))
    key = (n, steps)
    fields = _FIELDS.get(key)
    try:
        if fields is None:
            fields = (
                ti.field(ti.f32, shape=(n, steps)),
                ti.field(ti.f32, shape=(n, steps)),
                ti.field(ti.i32, shape=n),
            )
            _FIELDS[key] = fields
        xs, vxs, counts = fields
        _KERNEL(float(C), n, steps, xs, vxs, counts)
        ti.sync()
        xs_np = xs.to_numpy()
        vxs_np = vxs.to_numpy()
        counts_np = counts.to_numpy()
        points = []
        for i in range(n):
            for j in range(int(counts_np[i])):
                points.append((float(xs_np[i, j]), float(vxs_np[i, j])))
        return np.asarray(points, dtype=np.float64)
    except Exception:
        return None


def compute_poincare(C=C_DEFAULT, n_orbits=60, steps=3000):
    points = poincare_taichi(C, n_orbits, steps)
    if points is None:
        points = poincare_numpy(C, n_orbits, steps)
    if len(points):
        mask = (np.abs(points[:, 0]) < 2.0) & (np.abs(points[:, 1]) < 1.2)
        points = points[mask]
    return points
