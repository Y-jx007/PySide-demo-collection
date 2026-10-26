# -*- coding: utf-8 -*-
"""Taichi GPU field evaluation for the Liénard-Wiechert app."""
import numpy as np
import taichi_forge as ti

N = 1024
try:
    ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=False)
except Exception:
    ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False)

i_par = ti.field(dtype=ti.f32, shape=(N, N))
i_perp = ti.field(dtype=ti.f32, shape=(N, N))
i_total = ti.field(dtype=ti.f32, shape=(N, N))
i_b = ti.field(dtype=ti.f32, shape=(N, N))


@ti.kernel
def evaluate(
    t: ti.f32,
    beta0: ti.f32,
    modulation: ti.f32,
    drive: ti.f32,
    radius: ti.f32,
    extent: ti.f32,
):
    for i, j in ti.ndrange(N, N):
        x = (-1.0 + 2.0 * i / (N - 1)) * extent
        y = (-1.0 + 2.0 * j / (N - 1)) * extent
        tr = t - ti.sqrt(x * x + y * y)

        # Newton solve for the retarded time.
        for _ in ti.static(range(5)):
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

        # Total radiation E and B = n x E (c = 1).
        etx = epx + eqx
        ety = epy + eqy
        i_total[i, j] = etx * etx + ety * ety
        bz = nx * ety - ny * etx
        i_b[i, j] = bz * bz


def evaluate_np(t, beta0, modulation, drive, radius, extent):
    evaluate(
        float(t), float(beta0), float(modulation),
        float(drive), float(radius), float(extent),
    )
    return i_par.to_numpy(), i_perp.to_numpy(), i_total.to_numpy(), i_b.to_numpy()


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
):
    for i, j in ti.ndrange(V, V):
        x = (-1.0 + 2.0 * i / (V - 1)) * extent
        y = (-1.0 + 2.0 * j / (V - 1)) * extent
        tr = t - ti.sqrt(x * x + y * y)
        for _ in ti.static(range(5)):
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
        v_total[i, j] = ti.Vector([epx + eqx, epy + eqy])
        b_z[i, j] = nx * (epy + eqy) - ny * (epx + eqx)


def vector_field_np(t, beta0, modulation, drive, radius, extent):
    evaluate_vectors(
        float(t), float(beta0), float(modulation),
        float(drive), float(radius), float(extent),
    )
    # Taichi writes [x, y]; the line tracer expects [y, x].
    return {
        "e_par": v_par.to_numpy().transpose(1, 0, 2),
        "e_perp": v_perp.to_numpy().transpose(1, 0, 2),
        "e_total": v_total.to_numpy().transpose(1, 0, 2),
        "bz": b_z.to_numpy().T,
    }


TRACE_STEPS = 220
TRACE_SEEDS = 49
trace_pts = ti.Vector.field(2, dtype=ti.f32, shape=(TRACE_STEPS + 1, TRACE_SEEDS))
trace_alpha = ti.field(dtype=ti.f32, shape=(TRACE_STEPS + 1, TRACE_SEEDS))
trace_count = ti.field(dtype=ti.i32, shape=(TRACE_SEEDS,))


@ti.func
def lw_component(x, y, t, beta0, modulation, drive, radius, mode):
    tr = t - ti.sqrt(x * x + y * y)
    for _ in ti.static(range(5)):
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
    etx = epx + eqx
    ety = epy + eqy
    bz = nx * ety - ny * etx
    fx = 0.0
    fy = 0.0
    if mode == 0:
        fx, fy = etx, ety
    elif mode == 1:
        fx, fy = epx, epy
    elif mode == 2:
        fx, fy = eqx, eqy
    else:
        fx, fy = ety * bz, -etx * bz
    return ti.Vector([fx, fy])


@ti.kernel
def trace_component(
    t: ti.f32,
    beta0: ti.f32,
    modulation: ti.f32,
    drive: ti.f32,
    radius: ti.f32,
    extent: ti.f32,
    mode: ti.i32,
    direction: ti.f32,
    scale: ti.f32,
):
    for s in range(TRACE_SEEDS):
        gx = s % 7
        gy = s // 7
        x = (-0.92 + 1.84 * gx / 6.0) * extent
        y = (-0.92 + 1.84 * gy / 6.0) * extent
        trace_pts[0, s] = ti.Vector([x, y])
        trace_alpha[0, s] = 0.0
        count = 0
        active = 1
        h = 0.004 * extent
        for k in range(TRACE_STEPS):
            if active == 1:
                v1 = lw_component(x, y, t, beta0, modulation, drive, radius, mode)
                m1 = ti.sqrt(v1.dot(v1))
                norm = ti.min(1.0, ti.log(1.0 + m1 * m1 / ti.max(scale, 1e-30)) / 0.6931471805599453)
                if k == 0:
                    trace_alpha[0, s] = norm
                if m1 > 1e-12:
                    k1 = v1 / m1
                    v2 = lw_component(x + direction * 0.5 * h * k1[0],
                                      y + direction * 0.5 * h * k1[1],
                                      t, beta0, modulation, drive, radius, mode)
                    m2 = ti.sqrt(v2.dot(v2))
                    if m2 > 1e-12:
                        k2 = v2 / m2
                        v3 = lw_component(x + direction * 0.5 * h * k2[0],
                                          y + direction * 0.5 * h * k2[1],
                                          t, beta0, modulation, drive, radius, mode)
                        m3 = ti.sqrt(v3.dot(v3))
                        if m3 > 1e-12:
                            k3 = v3 / m3
                            v4 = lw_component(x + direction * h * k3[0],
                                              y + direction * h * k3[1],
                                              t, beta0, modulation, drive, radius, mode)
                            m4 = ti.sqrt(v4.dot(v4))
                            if m4 > 1e-12:
                                k4 = v4 / m4
                                nxstep = x + direction * h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
                                nystep = y + direction * h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
                                if ti.abs(nxstep) <= extent and ti.abs(nystep) <= extent:
                                    x, y = nxstep, nystep
                                    count += 1
                                    trace_pts[count, s] = ti.Vector([x, y])
                                    trace_alpha[count, s] = norm
                                else:
                                    active = 0
                            else:
                                active = 0
                        else:
                            active = 0
                    else:
                        active = 0
                else:
                    active = 0
        trace_count[s] = count + 1



def trace_component_np(t, beta0, modulation, drive, radius, extent,
                       mode, direction, scale):
    trace_component(
        float(t), float(beta0), float(modulation), float(drive),
        float(radius), float(extent), int(mode), float(direction), float(scale),
    )
    return (
        trace_pts.to_numpy(),
        trace_alpha.to_numpy(),
        trace_count.to_numpy(),
    )


color_img = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
prev_img = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))


@ti.kernel
def colorize(
    sr: ti.f32, sb: ti.f32, st: ti.f32, sbz: ti.f32,
    show_e: ti.i32, show_b: ti.i32,
    show_brems: ti.i32, show_sync: ti.i32, show_total: ti.i32,
):
    for i, j in ti.ndrange(N, N):
        nr = ti.min(1.0, ti.log(1.0 + i_par[i, j] / ti.max(sr, 1e-30)) / 0.6931471805599453)
        nb = ti.min(1.0, ti.log(1.0 + i_perp[i, j] / ti.max(sb, 1e-30)) / 0.6931471805599453)
        nt = ti.min(1.0, ti.log(1.0 + i_total[i, j] / ti.max(st, 1e-30)) / 0.6931471805599453)
        nz = ti.min(1.0, ti.log(1.0 + i_b[i, j] / ti.max(sbz, 1e-30)) / 0.6931471805599453)
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
            a = nt * 0.62
            rgb -= a * ti.Vector([1.0 - 255.0 / 255.0, 1.0 - 215.0 / 255.0, 1.0 - 0.0 / 255.0])
        rgb = ti.max(rgb, 0.0)
        rgb = ti.min(rgb, 1.0)
        prev_img[i, j] = 0.65 * prev_img[i, j] + 0.35 * rgb
        color_img[i, j] = prev_img[i, j]


def colorize_np(scales, show_e, show_b, show_brems, show_sync, show_total):
    sr, sb, st, sbz = [max(float(s), 1e-30) if s is not None else 1e-30 for s in scales]
    colorize(
        sr, sb, st, sbz,
        1 if show_e else 0, 1 if show_b else 0,
        1 if show_brems else 0, 1 if show_sync else 0, 1 if show_total else 0,
    )
    rgb = color_img.to_numpy().transpose(1, 0, 2)
    return np.ascontiguousarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))

prev_img.fill(1.0)
