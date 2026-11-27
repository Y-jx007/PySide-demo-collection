# -*- coding: utf-8 -*-
"""Taichi GPU field evaluation for the Liénard-Wiechert app.

Field-line traces use adaptive Dormand-Prince 5(4) with position-error control;
the image-field kernels remain fixed-iteration retarded-time evaluators.
"""
import numpy as np
import taichi_forge as ti

N = 512
try:
    ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=False, advanced_optimization=True)
except Exception:
    ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False, advanced_optimization=True)

i_par = ti.field(dtype=ti.f32, shape=(N, N))
i_perp = ti.field(dtype=ti.f32, shape=(N, N))
i_total = ti.field(dtype=ti.f32, shape=(N, N))
i_b = ti.field(dtype=ti.f32, shape=(N, N))
i_s = ti.field(dtype=ti.f32, shape=(N, N))


@ti.func
def retarded_guess(x, y, t, beta0, modulation, drive, radius):
    """Better initial retarded time: source position at the observation time.

    The old t-r guess (distance from the origin) can be far from the root when
    the observer is close to the moving charge.  This costs one sin/cos but
    removes the worst near-source Newton start.
    """
    theta_t = (beta0 * t / radius
               + modulation * (1.0 - ti.cos(drive * t))
               / (drive * radius))
    px_t = radius * ti.cos(theta_t)
    py_t = radius * ti.sin(theta_t)
    return t - ti.sqrt((x - px_t) ** 2 + (y - py_t) ** 2)



@ti.kernel
def evaluate(
    t: ti.f32,
    beta0: ti.f32,
    modulation: ti.f32,
    drive: ti.f32,
    radius: ti.f32,
    extent: ti.f32,
    coulomb: ti.i32,
):
    for i, j in ti.ndrange(N, N):
        x = (-1.0 + 2.0 * i / (N - 1)) * extent
        y = (-1.0 + 2.0 * j / (N - 1)) * extent
        tr = retarded_guess(x, y, t, beta0, modulation, drive, radius)

        # Newton solve for the retarded time.
        for _ in ti.static(range(6)):
            raw = beta0 + modulation * ti.sin(drive * tr)
            beta = ti.max(0.02, ti.min(0.985, raw))
            theta = (beta0 * tr / radius
                     + modulation * (1.0 - ti.cos(drive * tr))
                     / (drive * radius))
            px = radius * ti.cos(theta)
            py = radius * ti.sin(theta)
            dx = x - px
            dy = y - py
            dist = ti.sqrt(dx * dx + dy * dy)
            nx = dx / ti.max(dist, 1e-9)
            ny = dy / ti.max(dist, 1e-9)
            vx = beta * (-ti.sin(theta))
            vy = beta * ti.cos(theta)
            df = 1.0 - (nx * vx + ny * vy)
            tr = tr - (tr + dist - t) / ti.max(df, 1e-4)

        raw = beta0 + modulation * ti.sin(drive * tr)
        beta = ti.max(0.02, ti.min(0.985, raw))
        theta = (beta0 * tr / radius
                 + modulation * (1.0 - ti.cos(drive * tr))
                 / (drive * radius))
        ct = ti.cos(theta)
        st = ti.sin(theta)
        px = radius * ct
        py = radius * st
        dx = x - px
        dy = y - py
        dist = ti.max(ti.sqrt(dx * dx + dy * dy), 1e-6)
        nx = dx / dist
        ny = dy / dist
        vx = -beta * st
        vy = beta * ct
        bdot = modulation * drive * ti.cos(drive * tr)
        omega = beta / radius
        ax = bdot * (-st) - beta * omega * ct
        ay = bdot * ct - beta * omega * st

        vnorm = ti.max(ti.sqrt(vx * vx + vy * vy), 1e-9)
        ux = vx / vnorm
        uy = vy / vnorm
        adot = ax * ux + ay * uy
        apx = adot * ux
        apy = adot * uy
        aqx = ax - apx
        aqy = ay - apy

        den = ti.max(1.0 - (nx * vx + ny * vy), 1e-4)
        denom = den * den * den * dist

        # E_par
        ndot = nx * apx + ny * apy
        epx = ((nx - vx) * ndot - apx * den) / denom
        epy = ((ny - vy) * ndot - apy * den) / denom
        i_par[i, j] = epx * epx + epy * epy

        # E_perp
        ndot = nx * aqx + ny * aqy
        eqx = ((nx - vx) * ndot - aqx * den) / denom
        eqy = ((ny - vy) * ndot - aqy * den) / denom
        i_perp[i, j] = eqx * eqx + eqy * eqy

        # Total radiation E (plus the velocity/Coulomb field if requested).
        etx = epx + eqx
        ety = epy + eqy
        if coulomb == 1:
            gamma2 = 1.0 / ti.max(1.0 - beta * beta, 1e-6)
            evden = gamma2 * den * den * den * dist * dist
            etx += (nx - vx) / ti.max(evden, 1e-30)
            ety += (ny - vy) / ti.max(evden, 1e-30)
        i_total[i, j] = etx * etx + ety * ety
        bz = nx * ety - ny * etx
        i_b[i, j] = bz * bz
        sx = ety * bz
        sy = -etx * bz
        i_s[i, j] = sx * sx + sy * sy


def evaluate_np(t, beta0, modulation, drive, radius, extent, show_coulomb=False):
    evaluate(
        float(t), float(beta0), float(modulation),
        float(drive), float(radius), float(extent),
        1 if show_coulomb else 0,
    )
    return (i_par.to_numpy(), i_perp.to_numpy(), i_total.to_numpy(),
            i_b.to_numpy(), i_s.to_numpy())


V = 192
v_par = ti.Vector.field(2, dtype=ti.f32, shape=(V, V))
v_perp = ti.Vector.field(2, dtype=ti.f32, shape=(V, V))
v_total = ti.Vector.field(2, dtype=ti.f32, shape=(V, V))
b_z = ti.field(dtype=ti.f32, shape=(V, V))


@ti.kernel
def evaluate_vectors(
    t: ti.f32,
    beta0: ti.f32,
    modulation: ti.f32,
    drive: ti.f32,
    radius: ti.f32,
    extent: ti.f32,
    coulomb: ti.i32,
):
    for i, j in ti.ndrange(V, V):
        x = (-1.0 + 2.0 * i / (V - 1)) * extent
        y = (-1.0 + 2.0 * j / (V - 1)) * extent
        tr = retarded_guess(x, y, t, beta0, modulation, drive, radius)
        for _ in ti.static(range(6)):
            raw = beta0 + modulation * ti.sin(drive * tr)
            beta = ti.max(0.02, ti.min(0.985, raw))
            theta = (beta0 * tr / radius
                     + modulation * (1.0 - ti.cos(drive * tr))
                     / (drive * radius))
            px = radius * ti.cos(theta)
            py = radius * ti.sin(theta)
            dx = x - px
            dy = y - py
            dist = ti.sqrt(dx * dx + dy * dy)
            nx = dx / ti.max(dist, 1e-9)
            ny = dy / ti.max(dist, 1e-9)
            vx = beta * (-ti.sin(theta))
            vy = beta * ti.cos(theta)
            df = 1.0 - (nx * vx + ny * vy)
            tr = tr - (tr + dist - t) / ti.max(df, 1e-4)

        raw = beta0 + modulation * ti.sin(drive * tr)
        beta = ti.max(0.02, ti.min(0.985, raw))
        theta = (beta0 * tr / radius
                 + modulation * (1.0 - ti.cos(drive * tr))
                 / (drive * radius))
        ct = ti.cos(theta)
        st = ti.sin(theta)
        px = radius * ct
        py = radius * st
        dx = x - px
        dy = y - py
        dist = ti.max(ti.sqrt(dx * dx + dy * dy), 1e-6)
        nx = dx / dist
        ny = dy / dist
        vx = -beta * st
        vy = beta * ct
        bdot = modulation * drive * ti.cos(drive * tr)
        omega = beta / radius
        ax = bdot * (-st) - beta * omega * ct
        ay = bdot * ct - beta * omega * st

        vnorm = ti.max(ti.sqrt(vx * vx + vy * vy), 1e-9)
        ux, uy = vx / vnorm, vy / vnorm
        adot = ax * ux + ay * uy
        apx, apy = adot * ux, adot * uy
        aqx, aqy = ax - apx, ay - apy
        den = ti.max(1.0 - (nx * vx + ny * vy), 1e-4)
        denom = den * den * den * dist

        ndot = nx * apx + ny * apy
        epx = ((nx - vx) * ndot - apx * den) / denom
        epy = ((ny - vy) * ndot - apy * den) / denom
        ndot = nx * aqx + ny * aqy
        eqx = ((nx - vx) * ndot - aqx * den) / denom
        eqy = ((ny - vy) * ndot - aqy * den) / denom
        v_par[i, j] = ti.Vector([epx, epy])
        v_perp[i, j] = ti.Vector([eqx, eqy])
        etx = epx + eqx
        ety = epy + eqy
        if coulomb == 1:
            gamma2 = 1.0 / ti.max(1.0 - beta * beta, 1e-6)
            evden = gamma2 * den * den * den * dist * dist
            etx += (nx - vx) / ti.max(evden, 1e-30)
            ety += (ny - vy) / ti.max(evden, 1e-30)
        v_total[i, j] = ti.Vector([etx, ety])
        b_z[i, j] = nx * ety - ny * etx


def vector_field_np(t, beta0, modulation, drive, radius, extent,
                    show_coulomb=False):
    evaluate_vectors(
        float(t), float(beta0), float(modulation),
        float(drive), float(radius), float(extent),
        1 if show_coulomb else 0,
    )
    # Taichi writes [x, y]; the line tracer expects [y, x].
    return {
        "e_par": v_par.to_numpy().transpose(1, 0, 2),
        "e_perp": v_perp.to_numpy().transpose(1, 0, 2),
        "e_total": v_total.to_numpy().transpose(1, 0, 2),
        "bz": b_z.to_numpy().T,
    }


LIC_N = 768
LIC_STEPS = 28
lic_rgb = ti.Vector.field(3, dtype=ti.f32, shape=(LIC_N, LIC_N))
_LIC_NOISE = np.random.default_rng(20261118).random(
    (LIC_N, LIC_N)).astype(np.float32)
lic_noise_tex = ti.field(dtype=ti.f32, shape=(LIC_N, LIC_N))
lic_noise_tex.from_numpy(_LIC_NOISE)


@ti.func
def lic_sample(x, y, extent, field_mode):
    u = (x / extent + 1.0) * 0.5 * (V - 1)
    v = (y / extent + 1.0) * 0.5 * (V - 1)
    u = ti.max(0.0, ti.min(u, V - 1.001))
    v = ti.max(0.0, ti.min(v, V - 1.001))
    i0 = ti.cast(ti.floor(u), ti.i32)
    j0 = ti.cast(ti.floor(v), ti.i32)
    i1 = ti.min(i0 + 1, V - 1)
    j1 = ti.min(j0 + 1, V - 1)
    fx = u - i0
    fy = v - j0
    a = ti.Vector([0.0, 0.0])
    b = ti.Vector([0.0, 0.0])
    if field_mode == 1:
        a = v_par[i0, j0] * (1.0 - fx) + v_par[i1, j0] * fx
        b = v_par[i0, j1] * (1.0 - fx) + v_par[i1, j1] * fx
    elif field_mode == 2:
        a = v_perp[i0, j0] * (1.0 - fx) + v_perp[i1, j0] * fx
        b = v_perp[i0, j1] * (1.0 - fx) + v_perp[i1, j1] * fx
    else:
        a = v_total[i0, j0] * (1.0 - fx) + v_total[i1, j0] * fx
        b = v_total[i0, j1] * (1.0 - fx) + v_total[i1, j1] * fx
    return a * (1.0 - fy) + b * fy


@ti.func
def lic_sample_bz(x, y, extent):
    u = (x / extent + 1.0) * 0.5 * (V - 1)
    v = (y / extent + 1.0) * 0.5 * (V - 1)
    u = ti.max(0.0, ti.min(u, V - 1.001))
    v = ti.max(0.0, ti.min(v, V - 1.001))
    i0 = ti.cast(ti.floor(u), ti.i32)
    j0 = ti.cast(ti.floor(v), ti.i32)
    i1 = ti.min(i0 + 1, V - 1)
    j1 = ti.min(j0 + 1, V - 1)
    fx = u - i0
    fy = v - j0
    a = b_z[i0, j0] * (1.0 - fx) + b_z[i1, j0] * fx
    b = b_z[i0, j1] * (1.0 - fx) + b_z[i1, j1] * fx
    return a * (1.0 - fy) + b * fy


@ti.func
def lic_sample_poynting(x, y, extent):
    """In-plane Poynting vector S = E × B (B along z)."""
    e = lic_sample(x, y, extent, 0)
    bz = lic_sample_bz(x, y, extent)
    return ti.Vector([e[1] * bz, -e[0] * bz])


@ti.func
def lic_sample_noise(x, y, extent):
    u = (x / extent + 1.0) * 0.5 * (LIC_N - 1)
    v = (y / extent + 1.0) * 0.5 * (LIC_N - 1)
    u = ti.max(0.0, ti.min(u, LIC_N - 1.001))
    v = ti.max(0.0, ti.min(v, LIC_N - 1.001))
    i0 = ti.cast(ti.floor(u), ti.i32)
    j0 = ti.cast(ti.floor(v), ti.i32)
    i1 = ti.min(i0 + 1, LIC_N - 1)
    j1 = ti.min(j0 + 1, LIC_N - 1)
    fx = u - i0
    fy = v - j0
    a = lic_noise_tex[i0, j0] * (1.0 - fx) + lic_noise_tex[i1, j0] * fx
    b = lic_noise_tex[i0, j1] * (1.0 - fx) + lic_noise_tex[i1, j1] * fx
    return a * (1.0 - fy) + b * fy


@ti.kernel
def lic_kernel(
    extent: ti.f32, step: ti.f32,
    scale_par: ti.f32, scale_perp: ti.f32,
    scale_total: ti.f32, scale_poynting: ti.f32,
    show_e: ti.i32, show_coulomb: ti.i32, show_brems: ti.i32,
    show_sync: ti.i32, show_total: ti.i32,
    field_mode: ti.i32,
):
    for i, j in ti.ndrange(LIC_N, LIC_N):
        x = (-1.0 + 2.0 * i / (LIC_N - 1)) * extent
        y = (-1.0 + 2.0 * j / (LIC_N - 1)) * extent
        acc_n = 0.0
        acc_r = 0.0
        acc_g = 0.0
        acc_b = 0.0

        for direction in ti.static((-1.0, 1.0)):
            px, py = x, y
            for _ in range(LIC_STEPS):
                vec = lic_sample(px, py, extent, field_mode)
                if field_mode == 3:
                    vec = lic_sample_poynting(px, py, extent)
                mag = ti.sqrt(vec.dot(vec))
                if mag > 1e-9:
                    px += direction * step * vec[0] / mag
                    py += direction * step * vec[1] / mag

                # Colour is accumulated along the LIC path, not copied from a
                # separate intensity image at the pixel centre.
                vp = lic_sample(px, py, extent, 1)
                vq = lic_sample(px, py, extent, 2)
                vt = lic_sample(px, py, extent, 0)
                vs = lic_sample_poynting(px, py, extent)
                mp = vp.dot(vp)
                mq = vq.dot(vq)
                mt = vt.dot(vt)
                ms = vs.dot(vs)
                n_par = ti.min(1.0, ti.log(1.0 + mp / ti.max(scale_par, 1e-30))
                               / 0.6931471805599453)
                n_perp = ti.min(1.0, ti.log(1.0 + mq / ti.max(scale_perp, 1e-30))
                                / 0.6931471805599453)
                n_total = ti.min(1.0, ti.log(1.0 + mt / ti.max(scale_total, 1e-30))
                                 / 0.6931471805599453)
                n_poynting = ti.min(1.0, ti.log(1.0 + ms / ti.max(scale_poynting, 1e-30))
                                    / 0.6931471805599453)

                # Component colour weights: brems red, sync blue,
                # E_rad teal, Poynting/total yellow.
                e_active = show_e + show_coulomb
                wr = (show_sync * 0.90 * n_perp
                      + e_active * 0.75 * n_total
                      + show_total * 0.05 * n_poynting
                      + show_brems * 0.10 * n_par)
                wg = (show_brems * 0.90 * n_par
                      + show_sync * 0.70 * n_perp
                      + e_active * 0.35 * n_total
                      + show_total * 0.25 * n_poynting)
                wb = (show_brems * 0.90 * n_par
                      + show_sync * 0.10 * n_perp
                      + e_active * 0.45 * n_total
                      + show_total * 0.95 * n_poynting)
                noise = lic_sample_noise(px, py, extent)
                acc_n += noise
                acc_r += noise * wr
                acc_g += noise * wg
                acc_b += noise * wb

        active = show_e + show_coulomb + show_brems + show_sync + show_total
        if active == 0 or acc_n <= 1e-9:
            lic_rgb[i, j] = ti.Vector([1.0, 1.0, 1.0])
        else:
            inv = 1.0 / (2.0 * LIC_STEPS)
            line = acc_n * inv
            line = ti.max(0.0, ti.min(1.0, (line - 0.5) * 80.0 + 0.5))
            wr = ti.min(1.0, 2.8 * acc_r / ti.max(acc_n, 1e-9))
            wg = ti.min(1.0, 2.8 * acc_g / ti.max(acc_n, 1e-9))
            wb = ti.min(1.0, 2.8 * acc_b / ti.max(acc_n, 1e-9))
            lic_rgb[i, j] = ti.Vector([
                1.0 - line * ti.min(1.0, wr),
                1.0 - line * ti.min(1.0, wg),
                1.0 - line * ti.min(1.0, wb),
            ])


def lic_np(t, beta0, modulation, drive, radius, extent,
           show_e=True, show_coulomb=False,
           show_brems=True, show_sync=True, show_total=False):
    """Coloured LIC: colours are accumulated along each LIC path."""
    evaluate_vectors(
        float(t), float(beta0), float(modulation),
        float(drive), float(radius), float(extent),
        1 if show_coulomb else 0,
    )
    par = v_par.to_numpy()
    perp = v_perp.to_numpy()
    total = v_total.to_numpy()
    bz = b_z.to_numpy()
    poynting = np.stack((total[..., 1] * bz, -total[..., 0] * bz), axis=-1)

    def scale(arr):
        mag2 = np.sum(arr * arr, axis=-1)
        positive = mag2[mag2 > 0]
        if positive.size == 0:
            return 1e-30
        return max(float(np.percentile(positive, 99.3)) * 0.22, 1e-30)

    if show_total:
        field_mode = 3
    elif show_e or show_coulomb:
        field_mode = 0
    elif show_brems and show_sync:
        field_mode = 0
    elif show_brems:
        field_mode = 1
    elif show_sync:
        field_mode = 2
    else:
        field_mode = 0

    step = 1.5 * (2.0 * float(extent)) / (LIC_N - 1)
    lic_kernel(
        float(extent), step,
        scale(par), scale(perp), scale(total), scale(poynting),
        1 if show_e else 0, 1 if show_coulomb else 0,
        1 if show_brems else 0, 1 if show_sync else 0,
        1 if show_total else 0, field_mode,
    )
    return np.ascontiguousarray(
        np.clip(lic_rgb.to_numpy().transpose(1, 0, 2) * 255.0, 0, 255).astype(np.uint8)
    )


color_img = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
prev_img = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))


@ti.kernel
def colorize(
    sr: ti.f32, sb: ti.f32, st: ti.f32, sbz: ti.f32, ss: ti.f32,
    show_e: ti.i32, show_b: ti.i32,
    show_brems: ti.i32, show_sync: ti.i32, show_total: ti.i32,
):
    for i, j in ti.ndrange(N, N):
        nr = ti.min(1.0, ti.log(1.0 + i_par[i, j] / ti.max(sr, 1e-30)) / 0.6931471805599453)
        nb = ti.min(1.0, ti.log(1.0 + i_perp[i, j] / ti.max(sb, 1e-30)) / 0.6931471805599453)
        nt = ti.min(1.0, ti.log(1.0 + i_total[i, j] / ti.max(st, 1e-30)) / 0.6931471805599453)
        nz = ti.min(1.0, ti.log(1.0 + i_b[i, j] / ti.max(sbz, 1e-30)) / 0.6931471805599453)
        ns = ti.min(1.0, ti.log(1.0 + i_s[i, j] / ti.max(ss, 1e-30)) / 0.6931471805599453)
        rgb = ti.Vector([1.0, 1.0, 1.0])
        if show_brems == 1:
            a = nr * 0.62
            rgb -= a * ti.Vector([1.0 - 215.0 / 255.0, 1.0 - 55.0 / 255.0, 1.0 - 43.0 / 255.0])
        if show_sync == 1:
            a = nb * 0.62
            rgb -= a * ti.Vector([1.0 - 38.0 / 255.0, 1.0 - 90.0 / 255.0, 1.0 - 205.0 / 255.0])
        if show_e == 1:
            a = nt * 0.30
            rgb -= a * ti.Vector([1.0 - 28.0 / 255.0, 1.0 - 112.0 / 255.0, 1.0 - 132.0 / 255.0])
        if show_b == 1:
            a = nz * 0.62
            rgb -= a * ti.Vector([1.0 - 26.0 / 255.0, 1.0 - 145.0 / 255.0, 1.0 - 82.0 / 255.0])
        if show_total == 1:
            a = ns * 0.62
            rgb -= a * ti.Vector([1.0 - 255.0 / 255.0, 1.0 - 215.0 / 255.0, 1.0 - 0.0 / 255.0])
        rgb = ti.max(rgb, 0.0)
        rgb = ti.min(rgb, 1.0)
        prev_img[i, j] = 0.65 * prev_img[i, j] + 0.35 * rgb
        color_img[i, j] = prev_img[i, j]


def colorize_np(scales, show_e, show_b, show_brems, show_sync, show_total):
    sr, sb, st, sbz, ss = [
        max(float(s), 1e-30) if s is not None else 1e-30 for s in scales
    ]
    colorize(
        sr, sb, st, sbz, ss,
        1 if show_e else 0, 1 if show_b else 0,
        1 if show_brems else 0, 1 if show_sync else 0, 1 if show_total else 0,
    )
    rgb = color_img.to_numpy().transpose(1, 0, 2)
    return np.ascontiguousarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))

prev_img.fill(1.0)
