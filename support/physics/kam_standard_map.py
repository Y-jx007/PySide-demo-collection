# -*- coding: utf-8 -*-
"""Chirikov–Taylor standard map helpers for the KAM app.

The map is

    p' = p + (K / 2π) sin(2π q)   (mod 1)
    q' = q + p'                   (mod 1)

The app colours a grid of initial conditions by the rotation number
ν = <p> (mod 1), and dims cells whose finite-time Lyapunov exponent is
positive.  Regular KAM tori have λ ≈ 0; chaotic initial conditions have
λ > 0, so they fade into a quiet dark sea while tori and islands stay
luminous.  A large p-amplitude is not treated as chaos.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi

# Cyclic palette for the rotation number (0 -> 1 -> 0). Ends share a colour.
_PALETTE_STOPS = (
    (0.00, (32, 46, 94)),
    (0.16, (33, 110, 137)),
    (0.34, (62, 176, 158)),
    (0.50, (208, 190, 100)),
    (0.66, (228, 145, 107)),
    (0.82, (178, 98, 147)),
    (1.00, (32, 46, 94)),
)

_CHAOS_COLOR = np.array((17, 21, 28), dtype=np.float32)
_LUT_CACHE = None

# Image: λ below this is fully regular; above it the cell fades to chaos.
_FTLE_FLOOR = 0.004
_FTLE_SPAN = 0.12


def cyclic_lut(size=256):
    """Return the cyclic palette as a (size, 3) uint8 array."""
    global _LUT_CACHE
    if _LUT_CACHE is None or len(_LUT_CACHE) != size:
        xs = np.linspace(0.0, 1.0, size)
        out = np.empty((size, 3), dtype=np.float32)
        for i in range(len(_PALETTE_STOPS) - 1):
            x0, c0 = _PALETTE_STOPS[i]
            x1, c1 = _PALETTE_STOPS[i + 1]
            mask = (xs >= x0) & (xs <= x1)
            t = ((xs[mask] - x0) / (x1 - x0))[:, None]
            out[mask] = np.array(c0, np.float32) * (1.0 - t) + np.array(c1, np.float32) * t
        _LUT_CACHE = np.clip(out, 0, 255).astype(np.uint8)
    return _LUT_CACHE


def standard_map(q, p, K):
    """One standard-map step (works for scalars and numpy arrays)."""
    p_next = (p + (K / TWO_PI) * np.sin(TWO_PI * q)) % 1.0
    q_next = (q + p_next) % 1.0
    return q_next, p_next


def frequency_map(K, resolution=420, iterations=240, transient=None,
                  progress=None, should_stop=None):
    """Rotation number and finite-time Lyapunov exponent on a grid.

    Row 0 corresponds to p ≈ 1 and the last row to p ≈ 0, matching a plot
    whose p axis grows upwards.  nu is in [0, 1); lyap is the mean of
    log|J v| along the orbit after the transient, with v renormalised each
    step.  Regular tori have lyap close to zero.
    """
    res = max(8, int(resolution))
    iters = max(2, int(iterations))
    if transient is None:
        transient = max(10, iters // 4)
    transient = max(0, min(int(transient), iters - 1))
    span = max(1, iters - transient)

    q0 = (np.arange(res, dtype=np.float64) + 0.5) / res
    p0 = 1.0 - (np.arange(res, dtype=np.float64) + 0.5) / res
    q, p = np.meshgrid(q0, p0)
    q = q.ravel()
    p = p.ravel()
    total = np.zeros_like(p)
    lyap = np.zeros_like(p)
    tangent_q = np.ones_like(p)
    tangent_p = np.zeros_like(p)
    count = 0
    report_every = max(1, iters // 24)

    for i in range(iters):
        if should_stop is not None and should_stop():
            return None, None
        angle = TWO_PI * q
        kick = (K / TWO_PI) * np.sin(angle)
        shear = K * np.cos(angle)

        next_q = (1.0 + shear) * tangent_q + tangent_p
        next_p = shear * tangent_q + tangent_p
        norm = np.hypot(next_q, next_p)
        tangent_q = next_q / norm
        tangent_p = next_p / norm

        p = (p + kick) % 1.0
        q = (q + p) % 1.0
        if i >= transient:
            total += p
            lyap += np.log(norm)
            count += 1
        if progress is not None and (i % report_every == 0 or i == iters - 1):
            progress((i + 1) / iters)

    mean = total / max(count, 1)
    nu = mean % 1.0
    lyap = lyap / max(count, 1)
    return nu.reshape(res, res), lyap.reshape(res, res)


def trace_orbit(q0, p0, K, steps=1600, transient=None):
    """Trace one orbit; return points, rotation number and Lyapunov exponent."""
    steps = max(2, int(steps))
    if transient is None:
        transient = min(160, steps // 5)
    transient = max(0, min(int(transient), steps - 1))
    q = np.empty(steps + 1, dtype=np.float64)
    p = np.empty(steps + 1, dtype=np.float64)
    q[0] = float(q0) % 1.0
    p[0] = float(p0) % 1.0
    tangent_q = 1.0
    tangent_p = 0.0
    lyap = 0.0
    count = 0
    for i in range(steps):
        angle = TWO_PI * q[i]
        kick = (K / TWO_PI) * np.sin(angle)
        shear = K * np.cos(angle)
        next_q = (1.0 + shear) * tangent_q + tangent_p
        next_p = shear * tangent_q + tangent_p
        norm = float(np.hypot(next_q, next_p))
        tangent_q = next_q / norm
        tangent_p = next_p / norm
        p[i + 1] = (p[i] + kick) % 1.0
        q[i + 1] = (q[i] + p[i + 1]) % 1.0
        if i >= transient:
            lyap += float(np.log(norm))
            count += 1
    tail = p[1 + transient:]
    nu = float(np.mean(tail) % 1.0) if tail.size else 0.0
    if count:
        lyap /= count
    return np.column_stack((q, p)), nu, float(lyap)


def frequency_image(nu, lyap):
    """Render a rotation-number map as an RGB uint8 image.

    Stable cells keep the cyclic ν colour; cells with positive finite-time
    Lyapunov exponent fade towards a near-black chaos colour.
    """
    nu = np.asarray(nu, dtype=np.float64)
    lyap = np.asarray(lyap, dtype=np.float64)
    lut = cyclic_lut()
    idx = np.clip((nu % 1.0) * (len(lut) - 1), 0, len(lut) - 1).astype(np.int32)
    rgb = lut[idx].astype(np.float32)
    chaos = np.clip((lyap - _FTLE_FLOOR) / _FTLE_SPAN, 0.0, 1.0)
    rgb = rgb * (1.0 - chaos[..., None]) + _CHAOS_COLOR * chaos[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)
