# -*- coding: utf-8 -*-
"""Temporal nonlinear Schrodinger equation: self-phase modulation and solitons.

    i dA/dz + 1/2 beta2 d^2A/dt^2 + gamma |A|^2 A = 0

Sign convention: beta2 > 0 in this equation means anomalous dispersion in the
usual fibre convention (where the equation is i A_z = (beta2/2) A_tt - gamma
|A|^2 A).  With beta2 = +1, gamma = 1 and width T0 = 1, a sech pulse of
amplitude N is exactly the N-soliton: N = 1 propagates unchanged, N = 2
breathes with period pi/2, and the spectrum broadens through self-phase
modulation as N grows.

Spatial and temporal NLSE are the same equation with the transverse coordinate
replaced by (retarded) time, so this module reuses the palette and split-step
ideas of nls.py.
"""
from __future__ import annotations

import numpy as np

from support.physics.nls import PALETTE, palette_lut


class TemporalNLSE:
    """One transverse dimension is time; the marching direction is distance z."""

    def __init__(self, points=512, window=16.0, beta2=1.0, gamma=1.0,
                 dz=0.02, rows=512):
        self.n = int(points)
        self.window = float(window)
        self.dt = self.window / self.n
        self.beta2 = float(beta2)
        self.gamma = float(gamma)
        self.dz = float(dz)
        self.rows = int(rows)
        self.t = (np.arange(self.n) - self.n // 2) * self.dt
        omega = 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.dt)
        self.omega = omega
        self._dispersion = np.exp(-0.5j * self.beta2 * omega ** 2 * self.dz)
        self.field = np.zeros(self.n, dtype=np.complex128)
        # 预分配的水落图缓冲：新行从底部进，旧行整体上移（比每帧重建便宜）
        self._buffer = np.zeros((self.rows, self.n))
        self.filled = 0
        self.steps = 0
        self.reset(2.0, 1.0)

    # ---------- 初始条件 ----------
    def reset(self, order=2.0, width=1.0):
        """sech pulse of amplitude `order`: N = 1 is a fundamental soliton."""
        width = max(float(width), 0.1)
        self.field = (float(order) / np.cosh(self.t / width)).astype(np.complex128)
        self.steps = 0
        self._buffer[:] = 0.0
        self.filled = 0
        self._push(self.intensity())
        return self.field

    # ---------- 推进 ----------
    def step(self, count=1):
        field = self.field
        half = 0.5 * self.gamma * self.dz
        for _ in range(int(count)):
            intensity = np.abs(field) ** 2
            theta = half * intensity
            field = field * (np.cos(theta) + 1j * np.sin(theta))
            field = np.fft.ifft(np.fft.fft(field) * self._dispersion)
            intensity = np.abs(field) ** 2
            theta = half * intensity
            field = field * (np.cos(theta) + 1j * np.sin(theta))
            self._push(intensity)
        self.field = field
        self.steps += int(count)
        return field

    # ---------- 观测量 ----------
    def intensity(self):
        return np.abs(self.field) ** 2

    def energy(self):
        return float(np.sum(self.intensity()) * self.dt)

    def peak(self):
        return float(np.max(self.intensity()))

    def width(self):
        """RMS width in time."""
        intensity = self.intensity()
        total = float(intensity.sum())
        if total <= 0.0:
            return 0.0
        return float(np.sqrt(np.sum(intensity * self.t ** 2) / total))

    def spectrum_power(self):
        """|FFT|^2, fftshift'ed so omega = 0 sits in the middle."""
        return np.abs(np.fft.fftshift(np.fft.fft(self.field))) ** 2

    def spectral_width(self):
        """RMS width in frequency: the number SPM pushes up."""
        power = self.spectrum_power()
        shifted = np.fft.fftshift(self.omega)
        total = float(power.sum())
        if total <= 0.0:
            return 0.0
        return float(np.sqrt(np.sum(power * shifted ** 2) / total))

    def spectrum_db(self, floor_db=40.0):
        """Normalised dB spectrum in [0, 1], ready to plot."""
        power = self.spectrum_power()
        reference = float(power.max())
        if reference <= 0.0:
            return np.zeros_like(power)
        db = 10.0 * np.log10(np.clip(power / reference, 1e-12, 1.0))
        return np.clip((db + floor_db) / floor_db, 0.0, 1.0)

    # ---------- 渲染 ----------
    def _push(self, row):
        if self.filled < self.rows:
            self._buffer[self.filled] = row
            self.filled += 1
        else:
            self._buffer[:-1] = self._buffer[1:]
            self._buffer[-1] = row

    def waterfall(self):
        """(rows, points) image of |A|^2(t) while z advances, oldest on top."""
        if self.filled < self.rows:
            buffer = np.zeros((self.rows, self.n))
            buffer[self.rows - self.filled:] = self._buffer[:self.filled]
            return buffer
        return self._buffer

    def render_waterfall(self, palette=PALETTE, percentile=99.0):
        field = self.waterfall()
        reference = float(np.percentile(field[::4, ::4], percentile))
        if reference <= 0.0:
            reference = float(field.max()) or 1.0
        ratio = np.clip(field / reference, 0.0, 1.0)
        index = (ratio * 255.0).astype(np.uint8)
        return palette_lut(palette, gamma=0.75)[index]