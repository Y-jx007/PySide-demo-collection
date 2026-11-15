# -*- coding: utf-8 -*-
"""Taichi parallel backend for the KAM standard-map frequency map.

The kernel integrates all initial conditions in parallel (CUDA when
available, otherwise the Taichi CPU backend) and returns rotation numbers
and finite-time Lyapunov exponents as NumPy arrays.  The pure NumPy model
in kam_standard_map.py remains the fallback and the reference for tests.
"""
try:
    from common.taichi_base import ensure_taichi, ti
    _TI_IMPORTED = True
except Exception:
    ensure_taichi = None
    ti = None
    _TI_IMPORTED = False

TWO_PI = 6.283185307179586

_ARCH = ""
_AVAILABLE = False
_BROKEN = False

if _TI_IMPORTED:
    try:
        ensure_taichi()
        _ARCH = "GPU"
        _AVAILABLE = True
    except Exception:
        _AVAILABLE = False


if _AVAILABLE:
    @ti.kernel
    def _kam_kernel(K: ti.f32, res: ti.i32, iters: ti.i32, transient: ti.i32,
                    nu: ti.template(), lyap: ti.template()):
        for row, col in ti.ndrange(res, res):
            q = (ti.cast(col, ti.f32) + 0.5) / res
            p = 1.0 - (ti.cast(row, ti.f32) + 0.5) / res
            tq = 1.0
            tp = 0.0
            total = 0.0
            lsum = 0.0
            count = 0
            for i in range(iters):
                angle = TWO_PI * q
                kick = K / TWO_PI * ti.sin(angle)
                shear = K * ti.cos(angle)
                nq = (1.0 + shear) * tq + tp
                ntp = shear * tq + tp
                norm = ti.sqrt(nq * nq + ntp * ntp)
                tq = nq / norm
                tp = ntp / norm
                p = p + kick
                p = p - ti.floor(p)
                q = q + p
                q = q - ti.floor(q)
                if i >= transient:
                    total += p
                    lsum += ti.log(norm)
                    count += 1
            nuv = total / count
            nuv = nuv - ti.floor(nuv)
            nu[row, col] = nuv
            lyap[row, col] = lsum / count
else:
    _kam_kernel = None

_FIELDS = {}


def taichi_available():
    return bool(_AVAILABLE and not _BROKEN)


def taichi_backend_name():
    if taichi_available():
        return f"Taichi {_ARCH}"
    return "NumPy"


def _field_pair(res):
    pair = _FIELDS.get(res)
    if pair is None:
        pair = (
            ti.field(ti.f32, shape=(res, res)),
            ti.field(ti.f32, shape=(res, res)),
        )
        _FIELDS[res] = pair
    return pair


def frequency_map_taichi(K, resolution=360, iterations=220, transient=None):
    """Compute (nu, lyap) with the Taichi parallel kernel.

    Returns None, None when the backend is unavailable or has failed.
    """
    global _BROKEN
    if not taichi_available():
        return None, None
    res = max(8, int(resolution))
    iters = max(2, int(iterations))
    if transient is None:
        transient = max(10, iters // 4)
    transient = max(0, min(int(transient), iters - 1))
    try:
        nu, lyap = _field_pair(res)
        _kam_kernel(float(K), res, iters, transient, nu, lyap)
        ti.sync()
        return nu.to_numpy(), lyap.to_numpy()
    except Exception:
        _BROKEN = True
        return None, None


def compute_frequency_map(K, resolution=360, iterations=220, transient=None,
                          progress=None, should_stop=None):
    """Try Taichi first; fall back to the NumPy reference implementation."""
    if taichi_available() and not (should_stop is not None and should_stop()):
        if progress is not None:
            progress(0.02)
        nu, lyap = frequency_map_taichi(K, resolution, iterations, transient)
        if nu is not None and lyap is not None:
            if progress is not None:
                progress(1.0)
            return nu, lyap, f"Taichi {_ARCH}"

    from support.physics.kam_standard_map import frequency_map

    nu, lyap = frequency_map(
        K, resolution, iterations, transient=transient,
        progress=progress, should_stop=should_stop,
    )
    return nu, lyap, "NumPy"
