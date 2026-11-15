# -*- coding: utf-8 -*-
"""Differential geometry for the Connection app.

A point moves on a parametric surface S(u, v).  Tangent vectors are stored
as chart components X = X^u S_u + X^v S_v.  Parallel transport solves

    dX^k/dt = -Gamma^k_ij X^i du^j/dt

with RK4.  The Christoffel symbols are obtained from the induced metric and
the second derivatives of S, so the same code works for every surface.
"""
from __future__ import annotations

import math

import numpy as np

TAU = 2.0 * math.pi


def unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3, dtype=float)
    return np.asarray(v, dtype=float) / n


class Surface:
    """Base class for a parametric surface embedded in R^3."""

    name = "surface"
    normal_flip = 1.0     # -1 when cross(S_u, S_v) points into the body
    u_range = (-1.0, 1.0)
    v_range = (-1.0, 1.0)
    periodic_u = False
    periodic_v = False
    initial_uv = (0.0, 0.0)

    def point(self, u, v):
        raise NotImplementedError

    # ---------- differential geometry ----------
    def first(self, u, v, h=1e-4):
        su = (self.point(u + h, v) - self.point(u - h, v)) / (2.0 * h)
        sv = (self.point(u, v + h) - self.point(u, v - h)) / (2.0 * h)
        return su, sv

    def second(self, u, v, h=2e-3):
        suu = (self.point(u + h, v) - 2.0 * self.point(u, v) + self.point(u - h, v)) / (h * h)
        svv = (self.point(u, v + h) - 2.0 * self.point(u, v) + self.point(u, v - h)) / (h * h)
        suv = (
            self.point(u + h, v + h) - self.point(u + h, v - h)
            - self.point(u - h, v + h) + self.point(u - h, v - h)
        ) / (4.0 * h * h)
        return suu, suv, svv

    def metric(self, u, v):
        su, sv = self.first(u, v)
        return np.array(
            [[float(su @ su), float(su @ sv)],
             [float(su @ sv), float(sv @ sv)]],
            dtype=float,
        )

    def normal(self, u, v):
        su, sv = self.first(u, v)
        return unit(np.cross(su, sv))

    def faces_camera(self, u, v, eye):
        """True when the surface at (u, v) is turned towards the eye."""
        p = self.point(u, v)
        n = self.normal(u, v) * self.normal_flip
        return float(n @ (np.asarray(eye, dtype=float) - p)) > 0.0

    def christoffel(self, u, v):
        su, sv = self.first(u, v)
        suu, suv, svv = self.second(u, v)
        g = np.array(
            [[float(su @ su), float(su @ sv)],
             [float(su @ sv), float(sv @ sv)]],
            dtype=float,
        )
        try:
            ginv = np.linalg.inv(g)
        except np.linalg.LinAlgError:
            ginv = np.eye(2)
        table = {(0, 0): suu, (0, 1): suv, (1, 0): suv, (1, 1): svv}
        basis = (su, sv)
        gamma = np.zeros((2, 2, 2), dtype=float)
        for i in range(2):
            for j in range(2):
                hij = table[(i, j)]
                for k in range(2):
                    gamma[k, i, j] = sum(
                        ginv[k, l] * float(hij @ basis[l]) for l in range(2)
                    )
        return gamma

    # ---------- chart helpers ----------
    def wrap_uv(self, u, v):
        if self.periodic_u:
            u = (float(u) - self.u_range[0]) % TAU + self.u_range[0]
        else:
            u = float(np.clip(u, self.u_range[0], self.u_range[1]))
        if self.periodic_v:
            v = (float(v) - self.v_range[0]) % TAU + self.v_range[0]
        else:
            v = float(np.clip(v, self.v_range[0], self.v_range[1]))
        return u, v

    def delta_uv(self, u0, v0, u1, v1):
        du = float(u1) - float(u0)
        dv = float(v1) - float(v0)
        if self.periodic_u:
            du = (du + math.pi) % TAU - math.pi
        if self.periodic_v:
            dv = (dv + math.pi) % TAU - math.pi
        return du, dv

    def initial(self):
        return self.wrap_uv(*self.initial_uv)

    # ---------- tangent vectors ----------
    def tangent_basis(self, u, v):
        su, sv = self.first(u, v)
        e1 = unit(su)
        e2 = unit(sv - float(sv @ e1) * e1)
        return e1, e2

    def vector_to_components(self, u, v, vec):
        su, sv = self.first(u, v)
        g = np.array(
            [[float(su @ su), float(su @ sv)],
             [float(su @ sv), float(sv @ sv)]],
            dtype=float,
        )
        rhs = np.array([float(su @ vec), float(sv @ vec)], dtype=float)
        try:
            return np.linalg.solve(g, rhs)
        except np.linalg.LinAlgError:
            return np.zeros(2)

    def components_to_vector(self, u, v, comp):
        su, sv = self.first(u, v)
        return float(comp[0]) * su + float(comp[1]) * sv

    def cluster(self, u, v, count=12, length=0.24):
        e1, e2 = self.tangent_basis(u, v)
        comps = []
        vecs = []
        for k in range(int(count)):
            angle = TAU * k / max(1, int(count))
            vec = length * (math.cos(angle) * e1 + math.sin(angle) * e2)
            vecs.append(vec)
            comps.append(self.vector_to_components(u, v, vec))
        return np.asarray(comps, dtype=float), np.asarray(vecs, dtype=float)

    # ---------- nearest chart point under a screen ray ----------
    def _ray_hit(self, origin, direction, u, v):
        """Squared distance from the ray to S(u, v), plus the ray parameter."""
        p = self.point(u, v)
        w = p - origin
        t = float(w @ direction)
        if t <= 0.0:
            return None
        return float(w @ w - t * t), t

    def _refine_ray(self, origin, direction, seed, step_u, step_v, rounds=12):
        """Move the chart point onto the ray with damped Newton steps.

        Each step solves the 2x2 metric system for the tangent displacement
        that cancels the offset between S(u, v) and the ray, which lands on the
        exact pixel under the cursor in a handful of iterations (a plain grid
        descent stalls in the shallow valley near the silhouette).
        """
        best = self._ray_hit(origin, direction, *seed)
        if best is None:
            return self.wrap_uv(*seed), (1e100, 1e100)
        u, v = self.wrap_uv(*seed)
        limit = 1.6 * max(float(step_u), float(step_v))
        for _ in range(int(rounds)):
            p = self.point(u, v)
            w = p - origin
            t = float(w @ direction)
            if t <= 0.0:
                break
            offset = w - t * direction
            if float(offset @ offset) <= 1e-22:
                break
            delta = self.vector_to_components(u, v, -offset)
            span = float(np.hypot(delta[0], delta[1]))
            if span > limit:
                delta = delta * (limit / span)
            nu, nv = self.wrap_uv(u + delta[0], v + delta[1])
            hit = self._ray_hit(origin, direction, nu, nv)
            if hit is None or hit[0] > best[0] + 1e-9:
                break
            u, v = nu, nv
            best = hit
        return (u, v), best

    def nearest_uv_to_ray(self, origin, direction, samples=44, hint=None,
                          tolerance=None):
        """Chart point under a ray; a hint turns this into a cheap local search.

        With ``tolerance`` (world units) the call returns None when the ray
        misses the surface, so callers can keep the probe where it is.
        """
        direction = unit(direction)
        origin = np.asarray(origin, dtype=float)
        span_u = (self.u_range[1] - self.u_range[0]) / max(1, samples)
        span_v = (self.v_range[1] - self.v_range[0]) / max(1, samples)
        tol2 = 0.02 ** 2 if tolerance is None else float(tolerance) ** 2
        if hint is not None:
            local, hit = self._refine_ray(
                origin, direction, hint, span_u, span_v, rounds=16
            )
            if hit[0] <= tol2:
                return self.wrap_uv(*local)
        # Seed the search on the visible branch: among the grid nodes that are
        # about as close to the ray, keep the one nearest the eye, so a sphere
        # never hands back the hidden back face.
        hits = []
        us = np.linspace(self.u_range[0], self.u_range[1], samples)
        vs = np.linspace(self.v_range[0], self.v_range[1], samples)
        for u in us:
            for v in vs:
                hit = self._ray_hit(origin, direction, float(u), float(v))
                if hit is not None:
                    hits.append((hit[0], hit[1], float(u), float(v)))
        if not hits:
            return None
        hits.sort(key=lambda c: c[0])
        depth_cut = hits[0][0] + 0.05 * (1.0 + hits[0][0])
        best = min((c for c in hits if c[0] <= depth_cut), key=lambda c: c[1])
        local, hit = self._refine_ray(
            origin, direction, (best[2], best[3]), span_u, span_v, rounds=16
        )
        if tolerance is not None and hit[0] > tol2:
            return None
        return self.wrap_uv(*local)

    # ---------- mesh ----------
    def mesh(self, nu=42, nv=34):
        us = np.linspace(self.u_range[0], self.u_range[1], nu + 1)
        vs = np.linspace(self.v_range[0], self.v_range[1], nv + 1)
        verts = np.empty((nu + 1, nv + 1, 3), dtype=float)
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                verts[i, j] = self.point(float(u), float(v))
        faces = []
        for i in range(nu):
            for j in range(nv):
                faces.append(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))
        return verts, faces


class SphereSurface(Surface):
    name = "球面"
    normal_flip = -1.0    # this parameterisation gives an inward normal
    u_range = (0.0, TAU)
    v_range = (0.01, math.pi - 0.01)
    periodic_u = True
    initial_uv = (0.60, 2.49)

    def point(self, u, v):
        return np.array(
            [math.sin(v) * math.cos(u), math.sin(v) * math.sin(u), math.cos(v)],
            dtype=float,
        )


class SaddleSurface(Surface):
    name = "鞍面"
    u_range = (-1.25, 1.25)
    v_range = (-1.25, 1.25)
    initial_uv = (0.35, 0.25)

    def point(self, u, v):
        return np.array([u, v, 0.55 * (u * u - v * v)], dtype=float)


class TorusSurface(Surface):
    name = "环面"
    u_range = (0.0, TAU)
    v_range = (0.0, TAU)
    periodic_u = True
    periodic_v = True
    initial_uv = (0.8, 1.0)
    R = 0.72
    r = 0.28

    def point(self, u, v):
        w = self.R + self.r * math.cos(v)
        return np.array([w * math.cos(u), w * math.sin(u), self.r * math.sin(v)], dtype=float)


class PlaneSurface(Surface):
    name = "平面"
    u_range = (-1.30, 1.30)
    v_range = (-1.30, 1.30)
    initial_uv = (0.0, 0.0)

    def point(self, u, v):
        return np.array([u, v, 0.0], dtype=float)


SURFACES = ("球面", "鞍面", "环面", "平面")


def make_surface(name):
    return {
        "球面": SphereSurface,
        "鞍面": SaddleSurface,
        "环面": TorusSurface,
        "平面": PlaneSurface,
    }.get(name, SphereSurface)()


# ---------- ambient field ----------
def ambient_directions(count, tilt=0.32):
    """Unit directions of the ambient fan that is projected onto the surface."""
    count = max(1, int(count))
    return np.asarray([
        unit([math.cos(TAU * k / count), math.sin(TAU * k / count), float(tilt)])
        for k in range(count)
    ], dtype=float)


def field_envelope(point, amplitude=0.55):
    """Smooth positive envelope that gives the ambient field a varying length.

    The argument is the surface point in R^3, so the result is continuous on
    every chart, including the seams of the sphere and the torus.  With
    amplitude = 0 the field keeps a constant length of 1 everywhere.
    """
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    wave = (
        0.55 * math.sin(1.70 * x + 1.15 * y + 0.55 * z + 0.35)
        + 0.28 * math.sin(1.30 * y - 0.95 * z + 1.75)
        + 0.17 * math.sin(0.85 * z + 1.05 * x - 2.20)
    )
    amp = min(max(float(amplitude), 0.0), 0.85)
    return max(0.05, 1.0 + amp * wave)


# ---------- transport and connection ----------
def parallel_transport(surface, start, end, comps, steps=10):
    """Transport an array of chart vectors (N,2) from start to end."""
    u0, v0 = surface.wrap_uv(*start)
    u1, v1 = surface.wrap_uv(*end)
    du, dv = surface.delta_uv(u0, v0, u1, v1)
    x = np.array(comps, dtype=float, copy=True)
    g0 = surface.metric(u0, v0)
    target_len = np.array([
        math.sqrt(max(
            1e-16,
            g0[0, 0] * c[0] ** 2 + 2.0 * g0[0, 1] * c[0] * c[1] + g0[1, 1] * c[1] ** 2
        ))
        for c in x
    ])

    def rhs(t, xx):
        u, v = surface.wrap_uv(u0 + du * t, v0 + dv * t)
        gamma = surface.christoffel(u, v)
        out = np.empty_like(xx)
        # dX^k = -Gamma^k_ij X^i du^j
        out[:, 0] = -(
            gamma[0, 0, 0] * xx[:, 0] * du
            + gamma[0, 0, 1] * xx[:, 0] * dv
            + gamma[0, 1, 0] * xx[:, 1] * du
            + gamma[0, 1, 1] * xx[:, 1] * dv
        )
        out[:, 1] = -(
            gamma[1, 0, 0] * xx[:, 0] * du
            + gamma[1, 0, 1] * xx[:, 0] * dv
            + gamma[1, 1, 0] * xx[:, 1] * du
            + gamma[1, 1, 1] * xx[:, 1] * dv
        )
        return out

    dt = 1.0 / max(1, int(steps))
    for k in range(int(steps)):
        t = k * dt
        k1 = rhs(t, x)
        k2 = rhs(t + dt * 0.5, x + dt * 0.5 * k1)
        k3 = rhs(t + dt * 0.5, x + dt * 0.5 * k2)
        k4 = rhs(t + dt, x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        u, v = surface.wrap_uv(u0 + du * (k + 1) * dt, v0 + dv * (k + 1) * dt)
        g = surface.metric(u, v)
        for i in range(x.shape[0]):
            n2 = g[0, 0] * x[i, 0] ** 2 + 2.0 * g[0, 1] * x[i, 0] * x[i, 1] + g[1, 1] * x[i, 1] ** 2
            if n2 > 1e-16:
                x[i] *= target_len[i] / math.sqrt(n2)
    return x


def covariant_derivative(surface, uv, comps, du, dv):
    """Return chart components of ∇_w X for constant chart components."""
    u, v = surface.wrap_uv(*uv)
    gamma = surface.christoffel(u, v)
    x = np.asarray(comps, dtype=float)
    out = np.zeros_like(x)
    out[:, 0] = (
        gamma[0, 0, 0] * x[:, 0] * du
        + gamma[0, 0, 1] * x[:, 0] * dv
        + gamma[0, 1, 0] * x[:, 1] * du
        + gamma[0, 1, 1] * x[:, 1] * dv
    )
    out[:, 1] = (
        gamma[1, 0, 0] * x[:, 0] * du
        + gamma[1, 0, 1] * x[:, 0] * dv
        + gamma[1, 1, 0] * x[:, 1] * du
        + gamma[1, 1, 1] * x[:, 1] * dv
    )
    return out


def oriented_angle(normal, a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = unit(normal)
    return math.degrees(math.atan2(float(n @ np.cross(a, b)), float(a @ b)))
