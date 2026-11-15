# -*- coding: utf-8 -*-
"""2D MLS-MPM material solver, after taichi_elements mpm_solver.

Taichi is initialised once by common.taichi_base.  The particle count is
clamped to the allocated capacity after every seed batch.
"""
import math

import numpy as np

from common.taichi_base import ensure_taichi, ti

ensure_taichi()

real = ti.f32
dim = 2

RES_DEFAULT = 128
RES_OPTIONS = (128, 192, 256, 384)
DENSITY_OPTIONS = (4, 8, 16, 24, 36, 48)
RES_MAX = max(RES_OPTIONS)
PADDING = 3
GRAVITY_Y = -9.8
SIZE = 1.0

MATERIAL_WATER = 0
MATERIAL_ELASTIC = 1
MATERIAL_SNOW = 2
MATERIAL_SAND = 3
MATERIAL_STATIONARY = 4

SUPPORT_PLASTICITY = True
WATER_DENSITY = 1.0
RENDER_RES = 600
MAX_PARTICLES = 600000
MAX_SUBSTEPS = 40             # 单帧子步上限，避免局部爆炸时卡死


@ti.data_oriented
class MPM2D:
    """2D MLS-MPM。网格分辨率与采样密度都是运行时参数，可随时重建。"""

    def __init__(self, res=RES_DEFAULT, density=16, max_particles=MAX_PARTICLES,
                 arch=None):

        self.max_particles = max_particles
        self.padding = PADDING
        self.gsize = RES_MAX + 2 * PADDING

        n = max_particles
        self.n_particles = ti.field(ti.i32, shape=())
        self.x = ti.Vector.field(dim, real, n)
        self.v = ti.Vector.field(dim, real, n)
        self.F = ti.Matrix.field(dim, dim, real, n)
        self.C = ti.Matrix.field(dim, dim, real, n)
        self.Jp = ti.field(real, n)
        self.material = ti.field(ti.i32, n)

        self.grid_v = ti.Vector.field(dim, real, (self.gsize, self.gsize))
        self.grid_m = ti.field(real, (self.gsize, self.gsize))
        self.gravity = ti.Vector.field(dim, real, shape=())
        self.attractor_pos = ti.Vector.field(dim, real, shape=())
        self.attractor_strength = ti.field(real, shape=())
        self.gravity[None] = [0.0, GRAVITY_Y]
        self.attractor_strength[None] = 0.0

        self.img = ti.Vector.field(3, ti.u8, (RENDER_RES, RENDER_RES))
        self.blur_tmp = ti.Vector.field(3, ti.u8, (RENDER_RES, RENDER_RES))

        self.par = ti.Vector.field(4, real, shape=())   # dx, inv_dx, p_vol, p_mass
        self.res_cur = ti.field(ti.i32, shape=())

        self.E = 1e6 * SIZE
        self.nu = 0.2
        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        import math
        sin_phi = math.sin(math.radians(45))
        self.alpha = math.sqrt(2.0 / 3.0) * 2 * sin_phi / (3 - sin_phi)

        self.rres = RENDER_RES
        self.dt_scale = 1.0
        self.allowed_cfl = 0.9
        self.configure(res, density)
        self._compile()
        self._clear_grid_all()

    def configure(self, res, density):
        self.res = int(res)
        self.density = int(density)
        self._vmax_age = 99
        self.dx = SIZE / res
        self.inv_dx = 1.0 / self.dx
        self.default_dt = 2e-2 * self.dx / SIZE
        self.p_vol = self.dx ** dim
        self.p_rho = 1000.0
        self.p_mass = self.p_vol * self.p_rho
        self.par[None] = [self.dx, self.inv_dx, self.p_vol, self.p_mass]
        self.res_cur[None] = self.res
        self.n_particles[None] = 0
        self.attractor_strength[None] = 0.0
        self.gravity[None] = [0.0, GRAVITY_Y]

    def _compile(self):
        gsize = self.gsize
        padding = self.padding

        @ti.kernel
        def p2g(dt: real):
            par = self.par[None]
            dx = par[0]
            inv_dx = par[1]
            p_vol = par[2]
            p_mass = par[3]
            mu_0 = self.mu_0
            la_0 = self.lambda_0
            for p in range(self.n_particles[None]):
                base = ti.cast(ti.floor(self.x[p] * inv_dx - 0.5), ti.i32)
                fx = self.x[p] * inv_dx - ti.cast(base, real)
                w = [0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1.0) ** 2,
                     0.5 * (fx - 0.5) ** 2]
                Fp = self.F[p]
                if self.material[p] == MATERIAL_WATER:
                    Fp = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
                    Fp[0, 0] = self.Jp[p]
                Fp = (ti.Matrix([[1.0, 0.0], [0.0, 1.0]]) + dt * self.C[p]) @ Fp

                h = 1.0
                if self.material[p] != MATERIAL_WATER:
                    h = ti.exp(10.0 * (1.0 - self.Jp[p]))
                if self.material[p] == MATERIAL_ELASTIC:
                    h = 0.3
                mu = mu_0 * h
                la = la_0 * h
                if self.material[p] == MATERIAL_WATER:
                    mu = 0.0

                U, sig, V = ti.svd(Fp)
                J = 1.0
                if self.material[p] != MATERIAL_SAND:
                    for d in ti.static(range(dim)):
                        new_sig = sig[d, d]
                        if self.material[p] == MATERIAL_SNOW:
                            new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
                        self.Jp[p] *= sig[d, d] / new_sig
                        sig[d, d] = new_sig
                        J *= new_sig

                if self.material[p] == MATERIAL_WATER:
                    Fp = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
                    Fp[0, 0] = J
                    self.Jp[p] = J
                elif self.material[p] == MATERIAL_SNOW:
                    Fp = U @ sig @ V.transpose()

                stress = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                if self.material[p] != MATERIAL_SAND:
                    stress = 2 * mu * (Fp - U @ V.transpose()) @ Fp.transpose() \
                        + ti.Matrix([[1.0, 0.0], [0.0, 1.0]]) * la * J * (J - 1)
                self.F[p] = Fp

                stress = (-dt * p_vol * 4 * inv_dx ** 2) * stress
                mass = p_mass
                if self.material[p] == MATERIAL_WATER:
                    mass *= WATER_DENSITY
                affine = stress + mass * self.C[p]

                for i, j in ti.static(ti.ndrange(3, 3)):
                    dpos = (ti.Vector([i, j]) - fx) * dx
                    weight = w[i][0] * w[j][1]
                    nb = base + ti.Vector([i, j]) + padding
                    if 0 <= nb[0] < gsize and 0 <= nb[1] < gsize:
                        ti.atomic_add(self.grid_v[nb],
                                      weight * (mass * self.v[p] + affine @ dpos))
                        ti.atomic_add(self.grid_m[nb], weight * mass)

        @ti.kernel
        def grid_op(dt: real):
            res = self.res_cur[None]
            dx = self.par[None][0]
            ext = res + 2 * padding
            for i, j in ti.ndrange(ext, ext):
                if self.grid_m[i, j] > 0.0:
                    self.grid_v[i, j] = (1.0 / self.grid_m[i, j]) * self.grid_v[i, j]
                    self.grid_v[i, j] += dt * self.gravity[None]
                    strength = self.attractor_strength[None]
                    if strength != 0.0:
                        wx = (i - padding) * dx
                        wy = (j - padding) * dx
                        delta = self.attractor_pos[None] - ti.Vector([wx, wy])
                        dist = delta.norm()
                        self.grid_v[i, j] += (
                            delta / (0.01 + dist) * strength * dt * 100.0
                        )
                xi = i - padding
                yj = j - padding
                if xi < padding and self.grid_v[i, j][0] < 0.0:
                    self.grid_v[i, j][0] = 0.0
                if xi >= res - padding and self.grid_v[i, j][0] > 0.0:
                    self.grid_v[i, j][0] = 0.0
                if yj < padding and self.grid_v[i, j][1] < 0.0:
                    self.grid_v[i, j][1] = 0.0
                if yj >= res - padding and self.grid_v[i, j][1] > 0.0:
                    self.grid_v[i, j][1] = 0.0

        @ti.kernel
        def g2p(dt: real):
            par = self.par[None]
            dx = par[0]
            inv_dx = par[1]
            res = self.res_cur[None]
            for p in range(self.n_particles[None]):
                base = ti.cast(ti.floor(self.x[p] * inv_dx - 0.5), ti.i32)
                fx = self.x[p] * inv_dx - ti.cast(base, real)
                w = [0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1.0) ** 2,
                     0.5 * (fx - 0.5) ** 2]
                new_v = ti.Vector([0.0, 0.0])
                new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                for i, j in ti.static(ti.ndrange(3, 3)):
                    dpos = ti.Vector([i, j]) - fx
                    nb = base + ti.Vector([i, j]) + padding
                    if 0 <= nb[0] < gsize and 0 <= nb[1] < gsize:
                        # grid_op already normalised momentum and applied
                        # gravity/boundaries; g2p only interpolates velocity.
                        g_v = self.grid_v[nb]
                        weight = w[i][0] * w[j][1]
                        new_v += weight * g_v
                        new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
                if self.material[p] != MATERIAL_STATIONARY:
                    self.v[p] = new_v
                    self.C[p] = new_C
                    self.x[p] += dt * self.v[p]
            # 顺手清空网格，省掉一个独立的 clear kernel
            ext = self.res_cur[None] + 2 * padding
            for i, j in ti.ndrange(ext, ext):
                self.grid_v[i, j] = ti.Vector([0.0, 0.0])
                self.grid_m[i, j] = 0.0

        @ti.kernel
        def clear_grid_all():
            for i, j in self.grid_v:
                self.grid_v[i, j] = ti.Vector([0.0, 0.0])
                self.grid_m[i, j] = 0.0

        @ti.kernel
        def max_velocity() -> real:
            vm = 0.0
            for p in range(self.n_particles[None]):
                vm = ti.max(vm, ti.abs(self.v[p][0]))
                vm = ti.max(vm, ti.abs(self.v[p][1]))
            return vm

        @ti.kernel
        def render(radius: ti.i32):
            rres = RENDER_RES
            for i, j in self.img:
                self.img[i, j] = ti.Vector([255, 255, 255], ti.u8)
            for p in range(self.n_particles[None]):
                px = self.x[p][0] * rres
                py = (1.0 - self.x[p][1]) * rres
                col = ti.Vector([92, 104, 116], ti.u8)
                if self.material[p] == MATERIAL_WATER:
                    col = ti.Vector([40, 105, 205], ti.u8)
                elif self.material[p] == MATERIAL_ELASTIC:
                    col = ti.Vector([195, 55, 45], ti.u8)
                elif self.material[p] == MATERIAL_SNOW:
                    col = ti.Vector([175, 195, 220], ti.u8)
                elif self.material[p] == MATERIAL_SAND:
                    col = ti.Vector([190, 145, 70], ti.u8)
                ix = ti.cast(px, ti.i32)
                iy = ti.cast(py, ti.i32)
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        if di * di + dj * dj <= radius * radius + 1:
                            ii = ix + di
                            jj = iy + dj
                            if 0 <= ii < rres and 0 <= jj < rres:
                                self.img[jj, ii] = col

        @ti.kernel
        def dilate(level: ti.i32):
            # 按需要填缝：level 0 = 不处理，1 = 十字 5 抽样，2 = 3x3
            # 取最大值不会改变颜色（3x3 平均会把孤立点压成 4/9 亮度）
            rres = RENDER_RES
            for i, j in self.img:
                c = self.img[i, j]
                for di, dj in ti.static(ti.ndrange(-1, 2)):
                    if level == 2 or di == 0 or dj == 0:
                        ii = ti.min(ti.max(i + di, 0), rres - 1)
                        jj = ti.min(ti.max(j + dj, 0), rres - 1)
                        o = self.img[ii, jj]
                        c = ti.Vector([ti.max(c[0], o[0]), ti.max(c[1], o[1]),
                                       ti.max(c[2], o[2])], ti.u8)
                self.blur_tmp[i, j] = c
            for i, j in self.img:
                self.img[i, j] = self.blur_tmp[i, j]

        self._p2g = p2g
        self._grid_op = grid_op
        self._g2p = g2p
        self._clear_grid_all = clear_grid_all
        self._max_velocity = max_velocity
        self._render = render
        self._blur = dilate

    @ti.kernel
    def _seed_batch(self, count: ti.i32, new_material: ti.i32,
                    lx: real, ly: real, ux: real, uy: real,
                    vx: real, vy: real):
        # 一次 atomic 拿一整段下标，Python 侧不回读，避免每倒一次就同步一次
        base = ti.atomic_add(self.n_particles[None], count)
        for k in range(count):
            p = base + k
            if p < self.max_particles:
                rx = ti.random()
                ry = ti.random()
                self.x[p] = ti.Vector([lx + rx * (ux - lx), ly + ry * (uy - ly)])
                self.v[p] = ti.Vector([vx, vy])
                self.F[p] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
                self.C[p] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                self.material[p] = new_material
                # A zero initial Jp makes the hardening coefficient explode;
                # sand must start at the same reference volume as snow/water.
                self.Jp[p] = 1.0

    def add_cube(self, lower, size, material, velocity=(0.0, 0.0),
                 sample_density=None):
        if sample_density is None:
            sample_density = self.density
        vol = size[0] * size[1]
        num_new = int(sample_density * vol / self.dx ** dim + 1)
        self._seed_batch(num_new, int(material),
                         float(lower[0]), float(lower[1]),
                         float(lower[0] + size[0]), float(lower[1] + size[1]),
                         float(velocity[0]), float(velocity[1]))
        if int(self.n_particles[None]) > self.max_particles:
            self.n_particles[None] = self.max_particles
        return num_new

    def set_attractor(self, x=None, y=None, strength=0.0):
        if x is None or y is None or strength == 0.0:
            self.attractor_strength[None] = 0.0
            return
        self.attractor_pos[None] = [float(x), float(y)]
        self.attractor_strength[None] = float(strength)

    def set_gravity(self, gx, gy):
        self.gravity[None] = [float(gx), float(gy)]

    def step(self, frame_dt):
        self._vmax_age = getattr(self, "_vmax_age", 99) + 1
        if self._vmax_age >= 6:                      # 每 6 帧才回读一次，省掉每帧同步
            self._vmax = float(self._max_velocity())
            self._vmax_age = 0
        vmax = getattr(self, "_vmax", 0.0)
        cfl_dt = self.allowed_cfl * self.dx / (vmax + 1e-6)
        dt = min(self.default_dt * self.dt_scale, cfl_dt)
        # 子步数受上限约束，但 dt 绝不能再被放大（那样会越过 CFL 直接炸）
        substeps = max(1, min(MAX_SUBSTEPS, int(frame_dt / dt) + 1))
        dt = min(dt, frame_dt / substeps)
        for _ in range(substeps):
            self._p2g(dt)
            self._grid_op(dt)
            self._g2p(dt)
        self._sim_time = substeps * dt
        return substeps

    def particle_radius(self, grain=1.0):
        """点半径按粒子间距自适应；grain 是颗粒度倍率。"""
        spacing_px = (self.dx / max(1.0, (self.density ** 0.5))) * RENDER_RES
        return max(1, int(round(spacing_px * 0.55 * grain)))

    def draw(self, radius=None, grain=1.0):
        if radius is None:
            radius = self.particle_radius(grain)
        self._render(radius)
        # 点足够大时不填缝；点很小时用十字，中等用 3x3
        if radius < 2:
            self._blur(1)
        return self.img.to_numpy()

    def particle_info(self):
        n = int(self.n_particles[None])
        return {
            "position": self.x.to_numpy()[:n],
            "velocity": self.v.to_numpy()[:n],
            "material": self.material.to_numpy()[:n],
        }


