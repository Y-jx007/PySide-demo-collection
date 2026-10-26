# -*- coding: utf-8 -*-
"""Retarded Liénard-Wiechert radiation field for an electron.

Units: c = 1, q/(4 pi eps0) = 1, radiation field only.
"""
from __future__ import annotations

import numpy as np

C = 1.0


class Trajectory:
    """Electron on a circle with a modulated tangential speed.

    beta(t) = beta0 + modulation * sin(drive * t)
    theta(t) is the exact integral for the unclipped beta.
    """

    def __init__(self, beta0=0.65, modulation=0.25, drive=0.7, radius=1.0):
        self.beta0 = float(beta0)
        self.modulation = float(modulation)
        self.drive = max(0.02, float(drive))
        self.radius = max(0.15, float(radius))

    def state(self, t):
        t = np.asarray(t, dtype=np.float64)
        beta_raw = self.beta0 + self.modulation * np.sin(self.drive * t)
        beta = np.clip(beta_raw, 0.02, 0.985)
        beta_dot = self.modulation * self.drive * np.cos(self.drive * t)
        # Integral of beta0 + modulation*sin(drive*t).
        theta = (self.beta0 * t / self.radius
                 + self.modulation * (1.0 - np.cos(self.drive * t))
                 / (self.drive * self.radius))
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        pos = np.stack([self.radius * cos_t, self.radius * sin_t], axis=-1)
        tangent = np.stack([-sin_t, cos_t], axis=-1)
        radial = np.stack([cos_t, sin_t], axis=-1)
        vel = beta[..., None] * tangent
        omega = beta / self.radius
        acc = beta_dot[..., None] * tangent - (beta * omega)[..., None] * radial
        return {
            "t": t, "beta": beta, "beta_dot": beta_dot, "theta": theta,
            "pos": pos, "vel": vel, "acc": acc,
        }

    def retarded_state(self, observers, t, iterations=4):
        """Solve t_ret = t - |r_obs - r_e(t_ret)| / c by Newton iteration."""
        observers = np.asarray(observers, dtype=np.float64)
        obs_shape = observers.shape
        obs = observers.reshape(-1, 2)
        t = np.asarray(t, dtype=np.float64)
        # Expansion around the origin is only an initial guess; Newton then
        # converges quadratically even for beta close to one.
        tr = np.full(len(obs), float(t) - np.linalg.norm(obs, axis=1) / C)
        for _ in range(int(iterations)):
            st = self.state(tr)
            rvec = obs - st["pos"]
            dist = np.linalg.norm(rvec, axis=1)
            n = rvec / np.maximum(dist[..., None], 1e-9)
            beta = st["vel"] / C
            f = tr + dist / C - float(t)
            df = 1.0 - np.sum(n * beta, axis=1) / C
            tr = tr - f / np.maximum(df, 1e-4)
        st = self.state(tr)
        rvec = obs - st["pos"]
        dist = np.linalg.norm(rvec, axis=1)
        return {
            "tr": tr.reshape(obs_shape[:-1]),
            "pos": st["pos"].reshape(obs_shape),
            "vel": st["vel"].reshape(obs_shape),
            "acc": st["acc"].reshape(obs_shape),
            "dist": dist.reshape(obs_shape[:-1]),
            "beta": st["beta"].reshape(obs_shape[:-1]),
        }

    def field(self, observers, t, iterations=4):
        """Retarded radiation E field, split by acceleration components."""
        r = self.retarded_state(observers, t, iterations=iterations)
        obs = np.asarray(observers, dtype=np.float64)
        pos = r["pos"]
        dist = np.maximum(r["dist"], 1e-6)
        n = (obs - pos) / dist[..., None]
        beta = r["vel"] / C
        acc = r["acc"]
        den = np.maximum(1.0 - np.sum(n * beta, axis=-1), 1e-4)

        v_norm = np.linalg.norm(beta, axis=-1)
        vhat = beta / np.maximum(v_norm[..., None], 1e-9)
        a_parallel = np.sum(acc * vhat, axis=-1)[..., None] * vhat
        a_perp = acc - a_parallel

        def e_of(a):
            n_dot_a = np.sum(n * a, axis=-1)
            return (n - beta) * (n_dot_a[..., None]) - a * den[..., None]

        denom = (den ** 3) * dist
        e_par = np.nan_to_num(e_of(a_parallel) / denom[..., None], nan=0.0, posinf=0.0, neginf=0.0)
        e_perp = np.nan_to_num(e_of(a_perp) / denom[..., None], nan=0.0, posinf=0.0, neginf=0.0)
        e_total = e_par + e_perp
        b_z = n[..., 0] * e_total[..., 1] - n[..., 1] * e_total[..., 0]
        return {
            "e_par": e_par, "e_perp": e_perp, "e_total": e_total,
            "i_par": np.sum(e_par * e_par, axis=-1),
            "i_perp": np.sum(e_perp * e_perp, axis=-1),
            "i_total": np.sum(e_total * e_total, axis=-1),
            "i_b": b_z * b_z,
            "dist": dist,
            "n": n,
            "tr": r["tr"],
        }

    def far_field(self, t, angles=16, distance=None):
        if distance is None:
            distance = 10.0 * self.radius
        ang = np.linspace(0.0, 2.0 * np.pi, int(angles), endpoint=False)
        obs = distance * np.stack([np.cos(ang), np.sin(ang)], axis=-1)
        f = self.field(obs, t)
        return {
            "i_par": f["i_par"], "i_perp": f["i_perp"],
            "e_par": f["e_par"], "e_perp": f["e_perp"],
        }
