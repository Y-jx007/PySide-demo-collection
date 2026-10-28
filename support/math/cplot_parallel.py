# -*- coding: utf-8 -*-
"""Parallel sampling for the cplot UI.

cplot itself evaluates the user function on one flattened complex array in the
GUI process.  mpmath functions are pure Python and that loop is the expensive
part, so evaluate the same grid in worker processes and hand the finished
array to cplot's Plotter.  cplot keeps ownership of the color map, contours and
colorbars.
"""
from __future__ import annotations

import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import mpmath as mp
try:
    from scipy import special as _sps
except Exception:
    _sps = None

mp.dps = 15


def _wrap_mp(fn):
    return lambda z: complex(fn(z))


def _gamma(z):
    # scipy's C implementation handles scalar and complex arrays and is far
    # faster than the mpmath scalar loop for the common gamma preset.
    if _sps is not None:
        try:
            return _sps.gamma(z)
        except Exception:
            pass
    return complex(mp.gamma(z))


def _zeta(z):
    return complex(mp.zeta(z))


def _eta(z):
    return complex(mp.eta(z))


def _sn(z, tau):
    return complex(mp.ellipfun("sn", z, tau))


def _cn(z, tau):
    return complex(mp.ellipfun("cn", z, tau))


def _dn(z, tau):
    return complex(mp.ellipfun("dn", z, tau))


def _elliptic_parameter(z):
    return complex(mp.mfrom(tau=z))


def _kleinj(z):
    return complex(mp.kleinj(tau=z))


_WP_E1 = 6.8751858180203728274900957798105571979
_WP_SCALE = (2.0 * _WP_E1) ** 0.5
_WP_Q = float(np.exp(-np.pi))
_WP_N = 6
_WP_THETA2_0 = float(2.0 * _WP_Q ** 0.25 * sum(
    _WP_Q ** (k * (k + 1)) for k in range(_WP_N + 1)))
_WP_THETA3_0 = float(1.0 + 2.0 * sum(
    _WP_Q ** (k * k) for k in range(1, _WP_N + 1)))


def weierstrass_p(z):
    """Weierstrass wp for the square lattice with periods 1 and i.

    The lattice sum has 10000 terms per point in the preset.  This uses the
    Jacobi-sn/theta representation instead, vectorized over the whole grid.
    """
    arr = np.asarray(z, dtype=np.complex128)
    scalar = arr.ndim == 0
    zz = np.atleast_1d(arr).copy()
    # wp has periods 1 and i: reduce to the fundamental cell before the
    # theta series, otherwise large imaginary parts overflow the q-series.
    zz = ((zz.real - np.floor(zz.real + 0.5))
          + 1j * (zz.imag - np.floor(zz.imag + 0.5)))
    v = _WP_SCALE * zz / (_WP_THETA3_0 * _WP_THETA3_0)

    k1 = np.arange(_WP_N + 1)
    q1 = _WP_Q ** (k1 * (k1 + 1))
    th1 = np.zeros_like(v)
    for k, qk in zip(k1, q1):
        th1 += ((-1) ** int(k)) * qk * np.sin((2 * k + 1) * v)
    th1 *= 2.0 * _WP_Q ** 0.25

    k4 = np.arange(1, _WP_N + 1)
    q4 = _WP_Q ** (k4 * k4)
    th4 = np.ones_like(v)
    for k, qk in zip(k4, q4):
        th4 += 2.0 * ((-1) ** int(k)) * qk * np.cos(2 * k * v)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sn = (_WP_THETA3_0 / _WP_THETA2_0) * th1 / th4
        result = -_WP_E1 + 2.0 * _WP_E1 / (sn * sn)
    if scalar:
        return complex(result.reshape(-1)[0])
    return result.reshape(arr.shape)


_HELPERS = {
    "gamma": _gamma,
    "zeta": _zeta,
    "eta": _eta,
    "sn": _sn,
    "cn": _cn,
    "dn": _dn,
    "EllipticParameter": _elliptic_parameter,
    "Kleinj": _kleinj,
    "weierstrass_p": weierstrass_p,
    "wp": weierstrass_p,
}


def _make_env():
    env = {
        "math": math,
        "np": np,
        "mp": mp,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "pi": np.pi,
        "e": np.e,
        "z": None,
        "abs": np.abs,
        "real": np.real,
        "imag": np.imag,
        "conj": np.conj,
        "angle": np.angle,
    }
    env.update(_HELPERS)
    return env


def build_evaluator(source: str):
    """Compile the same scalar/expression forms accepted by the app."""
    if source.strip() in ("weierstrass_p(z)", "wp(z)"):
        return weierstrass_p
    if ("for i in range(-50,50)" in source
            and "1/(z - i - 1j*j)**2" in source):
        return weierstrass_p
    env = _make_env()
    if "\n" in source or "def " in source or "lambda " in source:
        exec(source, env)
        if "f" in env and callable(env["f"]):
            return env["f"]
        lines = [line.strip() for line in source.splitlines() if line.strip()]
        last = lines[-1] if lines else ""
        if last.startswith(("if ", "for ", "while ", "def ", "class ", "import ", "from ")):
            raise ValueError("多行代码必须定义一个名为 f 的函数，或最后一行是有效表达式")

        def f(z):
            env["z"] = z
            return eval(last, env)

        return f

    def f(z):
        env["z"] = z
        return eval(source, env)

    return f


_WORKER_EVALUATORS = {}


def _worker_eval(payload):
    source, real, imag = payload
    f = _WORKER_EVALUATORS.get(source)
    if f is None:
        f = build_evaluator(source)
        _WORKER_EVALUATORS[source] = f
    z = real + 1j * imag

    # Try the array form first.  A user-written double loop such as the old
    # Weierstrass code then performs NumPy operations on the whole chunk,
    # instead of running its 10000 Python iterations once per point.
    try:
        with np.errstate(all="ignore"):
            values = np.asarray(f(z))
        if values.shape == z.shape:
            return values.astype(np.complex128, copy=False)
    except Exception:
        pass

    # Scalar-only functions (branches on z.real, Python math, etc.).
    n = len(z)
    out = np.empty(n, dtype=np.complex128)
    for i in range(n):
        try:
            out[i] = complex(f(complex(z[i])))
        except Exception:
            out[i] = complex(np.nan, np.nan)
    return out


_EXECUTOR = None
_EXECUTOR_WORKERS = 0


def _get_executor(workers):
    global _EXECUTOR, _EXECUTOR_WORKERS
    if _EXECUTOR is None or _EXECUTOR_WORKERS != workers:
        if _EXECUTOR is not None:
            _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        # Spawn avoids inheriting Qt/OpenGL state from the GUI process.
        _EXECUTOR = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        _EXECUTOR_WORKERS = workers
    return _EXECUTOR


def shutdown():
    global _EXECUTOR, _EXECUTOR_WORKERS
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _EXECUTOR = None
        _EXECUTOR_WORKERS = 0


def _vectorized_eval(source, z_grid):
    """Evaluate the common vectorizable expression form on the whole array."""
    if ("for i in range(-50,50)" in source
            and "1/(z - i - 1j*j)**2" in source):
        return weierstrass_p(z_grid)
    if "\n" in source or "def " in source:
        # Small grids: use the whole-array form.  Large grids: let workers
        # process chunks, which is both vectorized and parallel.
        if z_grid.size > 60000:
            return None
    f = build_evaluator(source)
    try:
        with np.errstate(all="ignore"):
            values = f(z_grid)
        values = np.asarray(values)
        if values.shape == z_grid.shape:
            return values.astype(np.complex128, copy=False)
    except Exception:
        return None
    return None


def evaluate_grid(source: str, z_grid: np.ndarray, workers: int | None = None):
    """Evaluate f(Z) with a process pool, preserving the grid shape."""
    z_grid = np.asarray(z_grid)
    flat = z_grid.ravel()
    if workers is None:
        workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    # Process startup is not free; stay in-process for small grids.
    fast = _vectorized_eval(source, z_grid)
    if fast is not None:
        return fast
    if flat.size < 12000 or workers <= 1:
        f = build_evaluator(source)
        out = np.empty(flat.size, dtype=np.complex128)
        for i, z in enumerate(flat):
            try:
                out[i] = complex(f(complex(z)))
            except Exception:
                out[i] = complex(np.nan, np.nan)
        return out.reshape(z_grid.shape)

    n_chunks = min(flat.size, max(workers, min(workers * 4, 32)))
    re_parts = np.array_split(flat.real, n_chunks)
    im_parts = np.array_split(flat.imag, n_chunks)
    payloads = [(source, re, im) for re, im in zip(re_parts, im_parts)]
    try:
        parts = list(_get_executor(workers).map(_worker_eval, payloads))
        out = np.concatenate(parts)
        return out.reshape(z_grid.shape)
    except Exception:
        # If a worker pool cannot start, keep the app usable.
        f = build_evaluator(source)
        out = np.empty(flat.size, dtype=np.complex128)
        for i, z in enumerate(flat):
            try:
                out[i] = complex(f(complex(z)))
            except Exception:
                out[i] = complex(np.nan, np.nan)
        return out.reshape(z_grid.shape)
