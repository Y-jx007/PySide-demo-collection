# -*- coding: utf-8 -*-
"""contourpy-backed replacement for matplotx.contour."""
from __future__ import annotations

import os

import contourpy
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def _grid_axes(X, Y):
    x = np.asarray(X)
    y = np.asarray(Y)
    if x.ndim == 2:
        x = x[0, :]
    if y.ndim == 2:
        y = y[:, 0]
    return np.ascontiguousarray(x, dtype=float), np.ascontiguousarray(y, dtype=float)


def _jump_mask(Z, min_jump, max_jump):
    if min_jump is None and max_jump is None:
        return None
    mask = np.zeros(Z.shape, dtype=bool)

    def apply(diff, dst_slice, src_slice):
        bad = np.zeros_like(diff, dtype=bool)
        if min_jump is not None:
            bad |= np.abs(diff) < min_jump
        if max_jump is not None:
            bad |= np.abs(diff) > max_jump
        mask[dst_slice] |= bad
        mask[src_slice] |= bad

    # horizontal and vertical neighbour jumps
    with np.errstate(invalid="ignore"):
        apply(Z[:, 1:] - Z[:, :-1], (slice(None), slice(0, -1)),
              (slice(None), slice(1, None)))
        apply(Z[1:, :] - Z[:-1, :], (slice(0, -1), slice(None)),
              (slice(1, None), slice(None)))
    return mask


def _filter_lengths(paths, min_length):
    if min_length is None:
        return paths
    out = []
    for path in paths:
        if len(path) < 2:
            continue
        diff = np.diff(path, axis=0)
        length = float(np.sum(np.hypot(diff[:, 0], diff[:, 1])))
        if length >= min_length:
            out.append(path)
    return out


def contour(X, Y, Z, levels, min_jump=None, max_jump=None,
            min_contour_length=None, colors=None, alpha=None,
            linestyles="solid", linewidths=None, ax=None):
    x, y = _grid_axes(X, Y)
    z = np.asarray(Z)
    if z.shape != (len(y), len(x)):
        z = np.asarray(Z).reshape(len(y), len(x))

    levels = list(levels)
    if not levels:
        return
    if isinstance(colors, str) or colors is None:
        colors = [colors] * len(levels)
    if len(colors) == 1:
        colors = colors * len(levels)

    mask = _jump_mask(z, min_jump, max_jump)
    zz = np.ma.array(z, mask=mask) if mask is not None else z

    workers = max(1, min(8, os.cpu_count() or 1))
    try:
        cg = contourpy.contour_generator(
            x, y, zz, name="threaded", thread_count=workers,
            corner_mask=True,
        )
    except Exception:
        cg = contourpy.contour_generator(x, y, zz, corner_mask=True)

    cmap = plt.get_cmap()
    norm = mcolors.Normalize(vmin=min(levels), vmax=max(levels))
    all_lines = []
    all_colors = []
    for level, color in zip(levels, colors):
        paths = cg.lines(float(level))
        paths = _filter_lengths(paths, min_contour_length)
        if not paths:
            continue
        if color is None:
            color = cmap(norm(level))
        all_lines.extend(paths)
        all_colors.extend([color] * len(paths))

    if not all_lines:
        return
    lc = LineCollection(
        all_lines,
        colors=all_colors,
        alpha=alpha,
        linestyle=linestyles,
        linewidths=linewidths,
    )
    ax = ax or plt.gca()
    ax.add_collection(lc)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    return plt


def install_fast_contours():
    import cplot._main as cplot_main
    cplot_main.matplotx.contour = contour
