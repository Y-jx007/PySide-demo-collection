"""High-precision reference orbits for perturbation deep zoom.

The GPU kernel iterates f64 *offsets* from a reference orbit; the reference
itself must be known to far more than f64 precision, so it is computed here
with ``decimal`` (50 digits) and then rounded to f64 arrays for the kernel.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np

DIGITS = 50
# 参考轨道一旦逃逸就没必要继续平方（Decimal 会溢出，而且之后的像素也早
# 已经逃逸了）；留一个很大的有限值，内核仍然会把它判成逃逸。
_ESCAPE_LIMIT = Decimal(1e12)


def reference_orbit(
    center_re_hi: float,
    center_re_lo: float,
    center_im_hi: float,
    center_im_lo: float,
    max_iter: int,
    *,
    julia_re: float | None = None,
    julia_im: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``Z_0 .. Z_max_iter`` as float64 arrays.

    Mandelbrot: ``Z_{n+1} = Z_n² + c0`` with ``c0`` = the view centre.
    Julia: ``Z_0`` = the view centre, ``c`` fixed.
    """
    count = max(1, int(max_iter)) + 1
    with localcontext() as ctx:
        ctx.prec = DIGITS
        center_re = Decimal(center_re_hi) + Decimal(center_re_lo)
        center_im = Decimal(center_im_hi) + Decimal(center_im_lo)
        if julia_re is None:
            c_re, c_im = center_re, center_im
            zr = zi = Decimal(0)
        else:
            c_re, c_im = Decimal(julia_re), Decimal(julia_im)
            zr, zi = center_re, center_im

        re = np.empty(count, dtype=np.float64)
        im = np.empty(count, dtype=np.float64)
        escaped = False
        for n in range(count):
            re[n] = float(zr)
            im[n] = float(zi)
            if escaped:
                continue
            zr, zi = zr * zr - zi * zi + c_re, 2 * zr * zi + c_im
            if zr * zr + zi * zi > _ESCAPE_LIMIT:
                escaped = True
    return re, im
