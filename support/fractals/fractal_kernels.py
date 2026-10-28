"""Taichi kernels and preset formulas for the Mandelbrot/Julia and Newton apps.

The public kernels take the iteration function as a ``ti.template()``
argument. Taichi therefore compiles one specialised version per preset
function (or per user-supplied expression) and removes all runtime function
dispatch from the inner pixel loop.
"""

# Do not add `from __future__ import annotations` here: Taichi resolves the
# ``c64`` annotations when the module is imported, and postponed annotation
# strings are rejected by @ti.func.

import taichi_forge as ti

from common.custom_function import (
    c64,
    cconj,
    cdiv,
    cexp,
    cmul,
    ccos,
    csin,
    csqr,
    pack_color,
)

MANDELBROT_FUNCTION_NAMES = (
    "z²", "z³", "z⁴", "z⁵", "z²+z", "1/z",
    "sin(z)", "cos(z)", "exp(z)", "conj(z²)", "(|Re|+i|Im|)²",
)

NEWTON_FUNCTION_NAMES = (
    "z³-1", "z⁴-1", "z⁵-1", "z³-z", "z⁴+z²+1", "sin(z)", "cos(z)",
)


# ---------------------------------------------------------------------------
# Mandelbrot / Julia
# ---------------------------------------------------------------------------
@ti.func
def mandelbrot_preset(z: c64, func_type: ti.template()) -> c64:
    result = z
    if func_type == 0:
        result = csqr(z)
    elif func_type == 1:
        result = cmul(z, csqr(z))
    elif func_type == 2:
        result = csqr(csqr(z))
    elif func_type == 3:
        z2 = csqr(z)
        z3 = cmul(z, z2)
        result = cmul(z2, z3)
    elif func_type == 4:
        result = csqr(z) + z
    elif func_type == 5:
        result = cdiv(c64(1.0, 0.0), z)
    elif func_type == 6:
        result = csin(z)
    elif func_type == 7:
        result = ccos(z)
    elif func_type == 8:
        result = cexp(z)
    elif func_type == 9:
        result = cconj(csqr(z))
    elif func_type == 10:
        result = csqr(c64(ti.abs(z.x), ti.abs(z.y)))
    return result


@ti.kernel
def render_mandelbrot(
    pixels: ti.template(),
    width: ti.i32, height: ti.i32,
    max_iter: ti.i32, escape_radius: ti.f64,
    c_real: ti.f64, c_imag: ti.f64,
    display_mode: ti.i32,
    scale_x: ti.f64, offset_x: ti.f64,
    scale_y: ti.f64, offset_y: ti.f64,
    func: ti.template(), use_custom: ti.template(), func_type: ti.template(),
):
    escape_sq = escape_radius * escape_radius
    inv_log_radius = 1.0 / ti.log(escape_radius)
    inv_log2 = 1.0 / ti.log(2.0)
    inv_max_iter = 1.0 / ti.cast(max_iter, ti.f64)

    ti.loop_config(block_dim=256)
    for i, j in ti.ndrange(width, height):
        uv = c64(i * scale_x + offset_x, j * scale_y + offset_y)
        z = c64(0.0, 0.0)
        c = c64(0.0, 0.0)
        if display_mode == 0:       # Mandelbrot
            c = uv
        else:                       # Julia
            c = c64(c_real, c_imag)
            z = uv

        iterations = 0
        escaped = False
        for _ in range(max_iter):
            if z.x * z.x + z.y * z.y > escape_sq:
                escaped = True
                break
            if ti.static(use_custom):
                z = func(z) + c
            else:
                z = mandelbrot_preset(z, func_type) + c
            iterations += 1

        gray = 1.0
        if escaped:
            log_zn = ti.log(z.x * z.x + z.y * z.y) * 0.5
            nu = ti.log(log_zn * inv_log_radius) * inv_log2
            smooth_iter = ti.cast(iterations, ti.f64) + 1.0 - nu
            gray = ti.cast(smooth_iter * inv_max_iter, ti.f32)
        pixels[j, i] = pack_color(gray, gray, gray)


@ti.kernel
def compute_orbit(
    points: ti.template(), length: ti.template(),
    max_iter: ti.i32,
    c_real: ti.f64, c_imag: ti.f64, escape_radius: ti.f64,
    func: ti.template(), use_custom: ti.template(), func_type: ti.template(),
):
    """Sample the critical orbit once; the canvas draws it as a polyline."""
    z = c64(0.0, 0.0)
    c = c64(c_real, c_imag)
    escape_sq = escape_radius * escape_radius
    points[0] = z
    count = 1
    ti.loop_config(serialize=True)
    for _ in range(max_iter):
        if ti.static(use_custom):
            z = func(z) + c
        else:
            z = mandelbrot_preset(z, func_type) + c
        points[count] = z
        count += 1
        if z.x * z.x + z.y * z.y > escape_sq:
            break
    length[None] = count


# ---------------------------------------------------------------------------
# Newton / Nova
# ---------------------------------------------------------------------------
@ti.func
def newton_preset_f(z: c64, func_type: ti.template()) -> c64:
    """Preset functions, using algebraic powers instead of polar cpow."""
    result = c64(0.0, 0.0)
    if func_type == 0:          # z³ - 1
        result = cmul(z, csqr(z)) - c64(1.0, 0.0)
    elif func_type == 1:        # z⁴ - 1
        result = csqr(csqr(z)) - c64(1.0, 0.0)
    elif func_type == 2:        # z⁵ - 1
        result = cmul(z, csqr(csqr(z))) - c64(1.0, 0.0)
    elif func_type == 3:        # z³ - z
        result = cmul(z, csqr(z)) - z
    elif func_type == 4:        # z⁴ + z² + 1
        z2 = csqr(z)
        result = csqr(z2) + z2 + c64(1.0, 0.0)
    elif func_type == 5:        # sin(z)
        result = csin(z)
    elif func_type == 6:        # cos(z)
        result = ccos(z)
    else:
        result = cmul(z, csqr(z)) - c64(1.0, 0.0)
    return result


@ti.func
def newton_preset_df(z: c64, func_type: ti.template()) -> c64:
    result = c64(0.0, 0.0)
    if func_type == 0:
        result = cmul(c64(3.0, 0.0), csqr(z))
    elif func_type == 1:
        result = cmul(c64(4.0, 0.0), cmul(z, csqr(z)))
    elif func_type == 2:
        result = cmul(c64(5.0, 0.0), csqr(csqr(z)))
    elif func_type == 3:
        result = cmul(c64(3.0, 0.0), csqr(z)) - c64(1.0, 0.0)
    elif func_type == 4:
        result = cmul(c64(4.0, 0.0), cmul(z, csqr(z))) + cmul(c64(2.0, 0.0), z)
    elif func_type == 5:
        result = ccos(z)
    elif func_type == 6:
        result = -csin(z)
    else:
        result = cmul(c64(3.0, 0.0), csqr(z))
    return result


# Root tables kept in the same order as the original implementation so the
# grayscale palette is unchanged.
_ROOTS_Z3 = ((1.0, 0.0), (-0.5, 0.8660254), (-0.5, -0.8660254))
_ROOTS_Z4 = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
_ROOTS_Z5 = (
    (1.0, 0.0),
    (0.309016994, 0.951056516),
    (-0.809016994, 0.587785252),
    (-0.809016994, -0.587785252),
    (0.309016994, -0.951056516),
)
_ROOTS_Z3_MINUS_Z = ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0))
_ROOTS_Z4_PLUS_Z2_PLUS_1 = (
    (0.5, 0.8660254),
    (0.5, -0.8660254),
    (-0.5, 0.8660254),
    (-0.5, -0.8660254),
)

# Derived from the table lengths so the grayscale ramp stays in sync.
_GRAY_SCALE_Z3 = 1.0 / (len(_ROOTS_Z3) - 1)
_GRAY_SCALE_Z4 = 1.0 / (len(_ROOTS_Z4) - 1)
_GRAY_SCALE_Z5 = 1.0 / (len(_ROOTS_Z5) - 1)
_GRAY_SCALE_Z3_MINUS_Z = 1.0 / (len(_ROOTS_Z3_MINUS_Z) - 1)
_GRAY_SCALE_Z4_PLUS_Z2_PLUS_1 = 1.0 / (len(_ROOTS_Z4_PLUS_Z2_PLUS_1) - 1)


@ti.func
def nearest_root(z: c64, roots: ti.template()) -> ti.i32:
    best = 0
    best_dist = ti.cast(1e30, ti.f64)
    for k in ti.static(range(len(roots))):
        dx = z.x - roots[k][0]
        dy = z.y - roots[k][1]
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best = k
    return best


@ti.func
def newton_root_gray(z: c64, func_type: ti.template()) -> ti.f32:
    """Map the nearest root to the original grayscale ramp.

    The divisor comes from the table length, so changing a root table cannot
    silently desynchronise the palette.
    """
    gray = 1.0
    if func_type == 0:
        gray = ti.cast(nearest_root(z, _ROOTS_Z3), ti.f32) * _GRAY_SCALE_Z3
    elif func_type == 1:
        gray = ti.cast(nearest_root(z, _ROOTS_Z4), ti.f32) * _GRAY_SCALE_Z4
    elif func_type == 2:
        gray = ti.cast(nearest_root(z, _ROOTS_Z5), ti.f32) * _GRAY_SCALE_Z5
    elif func_type == 3:
        gray = (
            ti.cast(nearest_root(z, _ROOTS_Z3_MINUS_Z), ti.f32)
            * _GRAY_SCALE_Z3_MINUS_Z
        )
    elif func_type == 4:
        gray = (
            ti.cast(nearest_root(z, _ROOTS_Z4_PLUS_Z2_PLUS_1), ti.f32)
            * _GRAY_SCALE_Z4_PLUS_Z2_PLUS_1
        )
    return gray


@ti.kernel
def render_newton(
    pixels: ti.template(),
    width: ti.i32, height: ti.i32,
    max_iter: ti.i32, threshold: ti.f64,
    c_real: ti.f64, c_imag: ti.f64,
    display_mode: ti.i32,
    scale_x: ti.f64, offset_x: ti.f64,
    scale_y: ti.f64, offset_y: ti.f64,
    num_roots: ti.i32,
    func: ti.template(), use_custom: ti.template(), func_type: ti.template(),
):
    threshold_sq = threshold * threshold
    inv_max_iter = 1.0 / ti.cast(max_iter, ti.f64)
    h = 1e-8
    two_h = 2.0 * h

    ti.loop_config(block_dim=256)
    for i, j in ti.ndrange(width, height):
        z = c64(i * scale_x + offset_x, j * scale_y + offset_y)
        c = c64(c_real, c_imag)
        iterations = 0
        converged = False
        root_idx = 0

        for _ in range(max_iter):
            f_z = c64(0.0, 0.0)
            df_z = c64(0.0, 0.0)
            if ti.static(use_custom):
                # User expressions are arbitrary Python/Taichi expressions, so
                # keep the original central-difference derivative.
                f_z = func(z)
                f_plus = func(z + c64(h, 0.0))
                f_minus = func(z - c64(h, 0.0))
                df_z = c64(
                    (f_plus.x - f_minus.x) / two_h,
                    (f_plus.y - f_minus.y) / two_h,
                )
            else:
                f_z = newton_preset_f(z, func_type)
                df_z = newton_preset_df(z, func_type)

            if df_z.x * df_z.x + df_z.y * df_z.y < 1e-16:
                break

            z_new = z - cdiv(f_z, df_z)
            if display_mode == 1:   # Nova
                z_new = z_new + c

            if display_mode == 0:   # Newton
                if f_z.x * f_z.x + f_z.y * f_z.y < threshold_sq:
                    converged = True
                    if ti.static(use_custom):
                        angle = ti.atan2(z.y, z.x)
                        root_idx = ti.cast(
                            (angle + ti.math.pi) / (2.0 * ti.math.pi) * num_roots,
                            ti.i32,
                        ) % ti.max(num_roots, 1)
                    break
            else:                   # Nova
                diff = z_new - z
                if diff.x * diff.x + diff.y * diff.y < threshold_sq:
                    converged = True
                    break

            z = z_new
            iterations += 1

        gray = 0.0
        if converged:
            if display_mode == 0:
                if ti.static(use_custom):
                    gray = ti.select(
                        num_roots <= 1,
                        0.5,
                        ti.cast(root_idx, ti.f32)
                        / ti.cast(ti.max(num_roots - 1, 1), ti.f32),
                    )
                else:
                    gray = newton_root_gray(z, func_type)
            else:
                smooth = ti.cast(iterations, ti.f64) + 1.0
                gray = ti.cast(
                    3.0 * ti.log(1.0 + ti.log(1.0 + smooth * inv_max_iter)),
                    ti.f32,
                )
            pixels[j, i] = pack_color(gray, gray, gray)
        else:
            pixels[j, i] = pack_color(0.0, 0.0, 0.0)
