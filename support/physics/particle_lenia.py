# -*- coding: utf-8 -*-
"""Particle Lenia after Mordvintsev/Niklasson/Randazzo.

Two-stage ring kernel:
    field_i = w1 * sum_j K1(|x_j-x_i|, m1, s1)
    g_i     = w1 * sum_j K1'(|x_j-x_i|, m1, s1) * direction
    drive   = repulsion * short_range_repulsion - K2'(field_i, m2, s2) * g_i

The particle update keeps half of the previous velocity, matching the
reference shader (demos at znah.net/lenia).  The simulation fills a square
canvas/world instead of a circular dish.

Neighbour search uses a fixed-capacity uniform grid.  The kernel support is
at most 8 world units, so a 10-unit cell and a 3x3 stencil are sufficient.
"""
import math

import numpy as np

from common.taichi_base import ensure_taichi, ti

ensure_taichi()

N_MAX = 5000
WORLD_HALF = 75.0
DISH_R = WORLD_HALF  # backwards-compatible name for older callers
CELL_SIZE = 10.0
GRID_W = 16
CELL_CAP = 64


def calc_norm_coef(m, s):
    """Discrete normalization of K1, exactly as the reference demo."""
    dr = 0.1 * float(s)
    acc = 0.0
    prev = None
    r = max(float(m) - 3.0 * float(s), 0.0)
    while r < float(m) + 3.0 * float(s):
        y = (r - float(m)) / float(s)
        v = r * math.exp(-y * y)
        if prev is not None:
            acc += 0.5 * (prev + v)
        prev = v
        r += dr
    return 1.0 / max(acc * dr * 2.0 * math.pi, 1e-12)


@ti.data_oriented
class ParticleLenia:
    def __init__(self):
        self.pos = ti.Vector.field(2, ti.f32, shape=N_MAX)
        self.vel = ti.Vector.field(2, ti.f32, shape=N_MAX)
        self.n = ti.field(ti.i32, shape=())
        self.cell_count = ti.field(ti.i32, shape=GRID_W * GRID_W)
        self.cell_items = ti.field(ti.i32, shape=(GRID_W * GRID_W, CELL_CAP))
        self.field_value = ti.field(ti.f32, shape=N_MAX)
        self.m1 = 1.27
        self.s1 = 1.0
        self.w1 = calc_norm_coef(self.m1, self.s1)
        self.m2 = 0.6
        self.s2 = 0.2
        self.repulsion = 1.0
        self.dt = 0.1
        self.substeps = 16
        self.world_half = WORLD_HALF
        self.pointer = None

    def set_kernel(self, m1, s1):
        self.m1 = float(m1)
        self.s1 = float(s1)
        self.w1 = calc_norm_coef(self.m1, self.s1)

    @ti.kernel
    def _add_one(self, x: ti.f32, y: ti.f32):
        i = self.n[None]
        if i < N_MAX:
            self.pos[i] = ti.Vector([x, y])
            self.vel[i] = ti.Vector([0.0, 0.0])
            self.n[None] = i + 1

    @ti.kernel
    def _init(self, count: ti.i32, mode: ti.i32):
        for i in range(count):
            x = 0.0
            y = 0.0
            angle = ti.random() * 6.2831853
            if mode == 0:      # random across the square
                x = -70.0 + 140.0 * ti.random()
                y = -70.0 + 140.0 * ti.random()
            elif mode == 1:    # disk
                radius = ti.sqrt(ti.random()) * 54.0
                x = radius * ti.cos(angle)
                y = radius * ti.sin(angle)
            elif mode == 2:    # ring
                radius = 40.0 + 18.0 * ti.random()
                x = radius * ti.cos(angle)
                y = radius * ti.sin(angle)
            else:              # chain
                x = -50.0 + 100.0 * ti.random()
                y = 11.0 * ti.sin(x * 0.12)
            self.pos[i] = ti.Vector([x, y])
            self.vel[i] = ti.Vector([0.0, 0.0])
        self.n[None] = count

    @ti.kernel
    def _clear_grid(self):
        for c in range(GRID_W * GRID_W):
            self.cell_count[c] = 0

    @ti.kernel
    def _fill_grid(self):
        for i in range(self.n[None]):
            cx = ti.cast((self.pos[i][0] + DISH_R) / CELL_SIZE, ti.i32)
            cy = ti.cast((self.pos[i][1] + DISH_R) / CELL_SIZE, ti.i32)
            if cx < 0:
                cx = 0
            elif cx >= GRID_W:
                cx = GRID_W - 1
            if cy < 0:
                cy = 0
            elif cy >= GRID_W:
                cy = GRID_W - 1
            c = cy * GRID_W + cx
            slot = ti.atomic_add(self.cell_count[c], 1)
            if slot < CELL_CAP:
                self.cell_items[c, slot] = i

    @ti.kernel
    def _apply_pointer(
        self, x: ti.f32, y: ti.f32, radius: ti.f32,
        strength: ti.f32, sign: ti.f32,
    ):
        for i in range(self.n[None]):
            dx = self.pos[i][0] - x
            dy = self.pos[i][1] - y
            dist = ti.sqrt(dx * dx + dy * dy)
            if dist > 1e-5 and dist < radius:
                q = ti.max(0.0, 1.0 - dist / radius)
                falloff = q * q * (3.0 - 2.0 * q)
                f = sign * strength * falloff / dist
                self.vel[i] += ti.Vector([dx * f, dy * f])

    @ti.kernel
    def _step(
        self, dt: ti.f32, m1: ti.f32, s1: ti.f32, w1: ti.f32,
        m2: ti.f32, s2: ti.f32, repulsion: ti.f32,
    ):
        count = self.n[None]
        dmin = m1 - 3.0 * s1
        dmax = m1 + 3.0 * s1
        for i in range(count):
            pi = self.pos[i]
            rep_dir = ti.Vector([0.0, 0.0])
            grad = ti.Vector([0.0, 0.0])
            field = 0.0
            rep = 0.5
            cx = ti.cast((pi[0] + DISH_R) / CELL_SIZE, ti.i32)
            cy = ti.cast((pi[1] + DISH_R) / CELL_SIZE, ti.i32)
            if cx < 0:
                cx = 0
            elif cx >= GRID_W:
                cx = GRID_W - 1
            if cy < 0:
                cy = 0
            elif cy >= GRID_W:
                cy = GRID_W - 1
            for oy in range(-1, 2):
                ny = cy + oy
                if ny >= 0 and ny < GRID_W:
                    for ox in range(-1, 2):
                        nx = cx + ox
                        if nx >= 0 and nx < GRID_W:
                            c = ny * GRID_W + nx
                            cnt = self.cell_count[c]
                            if cnt > CELL_CAP:
                                cnt = CELL_CAP
                            for k in range(cnt):
                                j = self.cell_items[c, k]
                                if j != i:
                                    delta = self.pos[j] - pi
                                    dist = delta.norm()
                                    if dist > 1e-8:
                                        direction = delta / dist
                                        if dist < 1.0:
                                            f = 1.0 - dist
                                            rep_dir -= f * direction
                                            rep += 0.5 * f * f
                                        if dist > dmin and dist < dmax:
                                            c1 = (dist - m1) / s1
                                            value = ti.exp(-c1 * c1)
                                            field += value
                                            grad += (-2.0 * c1 * value / s1) * direction
            field *= w1
            grad *= w1
            self.field_value[i] = field
            c2 = (field - m2) / s2
            k2 = ti.exp(-c2 * c2)
            k2_prime = -2.0 * c2 * k2 / s2
            drive = rep_dir * repulsion - k2_prime * grad
            velocity = 0.5 * self.vel[i] + 0.5 * drive * dt
            point = pi + velocity
            px = ti.max(-WORLD_HALF, ti.min(WORLD_HALF, point[0]))
            py = ti.max(-WORLD_HALF, ti.min(WORLD_HALF, point[1]))
            if px != point[0] or py != point[1]:
                velocity = ti.Vector([0.0, 0.0])
            self.pos[i] = ti.Vector([px, py])
            self.vel[i] = velocity

    def add_particle(self, x, y):
        if int(self.n[None]) < N_MAX:
            x = max(-WORLD_HALF, min(WORLD_HALF, float(x)))
            y = max(-WORLD_HALF, min(WORLD_HALF, float(y)))
            self._add_one(x, y)

    def set_pointer(self, x=None, y=None, mode=None, radius=35.0, strength=0.05):
        if x is None or y is None or mode is None:
            self.pointer = None
            return
        sign = -1.0 if mode == "attract" else 1.0
        self.pointer = (float(x), float(y), float(radius), float(strength), sign)

    def reset(self, pattern="随机", count=900):
        modes = {"随机": 0, "圆盘": 1, "环": 2, "链": 3}
        count = max(50, min(N_MAX, int(count)))
        self.pointer = None
        self._init(count, modes.get(pattern, 0))

    def step(self):
        pointer = self.pointer
        for _ in range(max(1, int(self.substeps))):
            if pointer is not None:
                self._apply_pointer(
                    pointer[0], pointer[1], pointer[2], pointer[3], pointer[4]
                )
            self._clear_grid()
            self._fill_grid()
            self._step(
                float(self.dt), float(self.m1), float(self.s1), float(self.w1),
                float(self.m2), float(self.s2), float(self.repulsion),
            )

    def positions(self):
        return self.pos.to_numpy()[: int(self.n[None])].copy()

    def velocities(self):
        return self.vel.to_numpy()[: int(self.n[None])].copy()

    def count(self):
        return int(self.n[None])

    def field_values(self):
        return self.field_value.to_numpy()[: int(self.n[None])].copy()
