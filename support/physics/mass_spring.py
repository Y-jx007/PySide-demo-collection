# -*- coding: utf-8 -*-
"""Mass-spring network with Taichi integration.

Particles live in the unit square.  Springs use their natural length at
creation (the original demo hard-coded 0.1, which made every network
contract visibly).  Forces are semi-implicit Euler + dashpot damping.
"""
import numpy as np

from common.taichi_base import ensure_taichi, ti

ensure_taichi()

N_MAX = 600
SPRING_MAX = 5000
WORLD = 1.0


@ti.data_oriented
class MassSpring:
    def __init__(self):
        self.pos = ti.Vector.field(2, ti.f32, shape=N_MAX)
        self.vel = ti.Vector.field(2, ti.f32, shape=N_MAX)
        self.force = ti.Vector.field(2, ti.f32, shape=N_MAX)
        self.fixed = ti.field(ti.i32, shape=N_MAX)
        self.spring_a = ti.field(ti.i32, shape=SPRING_MAX)
        self.spring_b = ti.field(ti.i32, shape=SPRING_MAX)
        self.spring_rest = ti.field(ti.f32, shape=SPRING_MAX)
        self.n = ti.field(ti.i32, shape=())
        self.spring_count = ti.field(ti.i32, shape=())

        self.gravity = 9.8
        self.stiffness = 1800.0
        self.dashpot = 22.0
        self.drag = 0.35
        self.dt = 0.0012
        self.substeps = 8
        self.connect_radius = 0.13
        self._n = 0
        self._springs = 0

    # -- construction -------------------------------------------------
    @ti.kernel
    def _add(self, x: ti.f32, y: ti.f32, fixed: ti.i32):
        i = self.n[None]
        self.pos[i] = ti.Vector([x, y])
        self.vel[i] = ti.Vector([0.0, 0.0])
        self.fixed[i] = fixed
        self.n[None] = i + 1

    @ti.kernel
    def _add_spring(self, i: ti.i32, j: ti.i32, rest: ti.f32):
        k = self.spring_count[None]
        self.spring_a[k] = i
        self.spring_b[k] = j
        self.spring_rest[k] = rest
        self.spring_count[None] = k + 1

    @ti.kernel
    def _set_fixed(self, i: ti.i32, value: ti.i32):
        self.fixed[i] = value
        if value != 0:
            self.vel[i] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def _clear(self):
        self.n[None] = 0
        self.spring_count[None] = 0

    def clear(self):
        self._clear()
        self._n = 0
        self._springs = 0

    def add_particle(self, x, y, fixed=False, connect=True):
        if self.n[None] >= N_MAX:
            return -1
        x = float(min(max(x, 0.005), WORLD - 0.005))
        y = float(min(max(y, 0.005), WORLD - 0.005))
        idx = int(self.n[None])
        self._add(x, y, int(fixed))
        if connect and idx > 0:
            positions = self.pos.to_numpy()[:idx]
            delta = positions - np.array([x, y], dtype=np.float64)
            distances = np.linalg.norm(delta, axis=1)
            radius = float(self.connect_radius)
            for other in np.where(
                (distances > 1e-5) & (distances < radius)
            )[0]:
                if self.spring_count[None] >= SPRING_MAX:
                    break
                self._add_spring(int(other), idx, float(distances[other]))
        return idx

    def set_fixed(self, index, value=True):
        if 0 <= int(index) < int(self.n[None]):
            self._set_fixed(int(index), int(bool(value)))

    # -- simulation ---------------------------------------------------
    @ti.kernel
    def _step(
        self, gravity: ti.f32, stiffness: ti.f32,
        dashpot: ti.f32, drag: ti.f32, dt: ti.f32,
    ):
        n = self.n[None]
        for i in range(n):
            self.force[i] = ti.Vector([0.0, -gravity])
        for s in range(self.spring_count[None]):
            i = self.spring_a[s]
            j = self.spring_b[s]
            delta = self.pos[i] - self.pos[j]
            dist = delta.norm()
            if dist > 1e-5:
                direction = delta / dist
                stretch = dist - self.spring_rest[s]
                spring_force = -stiffness * stretch * direction
                self.force[i] += spring_force
                self.force[j] -= spring_force
                relative = (self.vel[i] - self.vel[j]).dot(direction)
                damping_force = -dashpot * relative * direction
                self.force[i] += damping_force
                self.force[j] -= damping_force
        for i in range(n):
            if self.fixed[i] != 0:
                self.vel[i] = ti.Vector([0.0, 0.0])
                continue
            self.vel[i] += dt * self.force[i]
            self.vel[i] *= ti.exp(-dt * drag)
            p = self.pos[i] + dt * self.vel[i]
            for d in ti.static(range(2)):
                if p[d] < 0.0:
                    p[d] = 0.0
                    if self.vel[i][d] < 0.0:
                        self.vel[i][d] = 0.0
                elif p[d] > WORLD:
                    p[d] = WORLD
                    if self.vel[i][d] > 0.0:
                        self.vel[i][d] = 0.0
            self.pos[i] = p

    @ti.kernel
    def _attract(self, x: ti.f32, y: ti.f32, strength: ti.f32, dt: ti.f32):
        target = ti.Vector([x, y])
        for i in range(self.n[None]):
            if self.fixed[i] == 0:
                delta = target - self.pos[i]
                self.vel[i] += strength * dt * delta

    def step(self):
        for _ in range(int(self.substeps)):
            self._step(
                float(self.gravity), float(self.stiffness),
                float(self.dashpot), float(self.drag), float(self.dt),
            )

    def attract(self, x, y):
        self._attract(
            float(x), float(y), 22.0, float(self.dt) * self.substeps
        )

    # -- view data ----------------------------------------------------
    def positions(self):
        return self.pos.to_numpy()[: int(self.n[None])].copy()

    def velocities(self):
        return self.vel.to_numpy()[: int(self.n[None])].copy()

    def springs(self):
        count = int(self.spring_count[None])
        return (
            self.spring_a.to_numpy()[:count].copy(),
            self.spring_b.to_numpy()[:count].copy(),
            self.spring_rest.to_numpy()[:count].copy(),
        )

    def fixed_mask(self):
        return self.fixed.to_numpy()[: int(self.n[None])].astype(bool)

    def count(self):
        return int(self.n[None]), int(self.spring_count[None])

    # -- presets ------------------------------------------------------
    def reset(self, kind="悬索桥"):
        self.clear()
        if kind == "悬索桥":
            count = 15
            for i in range(count):
                x = 0.12 + 0.76 * i / (count - 1)
                self.add_particle(x, 0.70, fixed=(i in (0, count - 1)),
                                  connect=True)
        elif kind == "网":
            cols, rows = 8, 6
            for row in range(rows):
                for col in range(cols):
                    x = 0.18 + 0.64 * col / (cols - 1)
                    y = 0.22 + 0.62 * row / (rows - 1)
                    self.add_particle(x, y, fixed=(row == rows - 1),
                                      connect=True)
        elif kind == "链":
            count = 12
            for i in range(count):
                y = 0.18 + 0.70 * i / (count - 1)
                self.add_particle(0.5, y, fixed=(i == count - 1),
                                  connect=True)
        else:  # 随机
            rng = np.random.default_rng(4)
            for _ in range(38):
                self.add_particle(
                    float(rng.uniform(0.15, 0.85)),
                    float(rng.uniform(0.15, 0.85)),
                    fixed=False, connect=True,
                )
            for index in range(min(3, int(self.n[None]))):
                self.set_fixed(index, True)
