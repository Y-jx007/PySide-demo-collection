# -*- coding: utf-8 -*-
"""Focusing 2D nonlinear Schrodinger equation (cubic NLSE).

    i psi_t = -1/2 laplacian(psi) - |psi|^2 psi

Solved with the split-step Fourier method: the cubic nonlinearity is a half
step in real space, the dispersion is a full step in k space.  Splitting keeps
the total power exactly (the tests check it), which matters because the
whole point of the demo is the competition between diffraction and focusing.

Above a critical power the beam stops spreading and collapses; a wide noisy
beam instead develops a field of filaments (modulational instability), and a
vortex of charge m breaks into m filaments.  All three are the same equation.
"""
from __future__ import annotations

import numpy as np

# 只有一个色阶：激光本身的颜色（532 nm 的绿光），从纸白到深绿。
# 强度越高越深，和 Connection 的纸感取色一致。
PALETTE = "绿光"
PALETTES = {
    PALETTE: (
        (0.00, (250, 250, 244)),
        (0.16, (214, 236, 204)),
        (0.40, (138, 205, 140)),
        (0.66, (52, 158, 96)),
        (0.86, (16, 104, 62)),
        (1.00, (6, 46, 28)),
    ),
}

PRESETS = ("单束", "涡旋", "双束", "噪声海")


_LUTS = {}


def palette_lut(palette=PALETTE):
    """256-entry RGB lookup table for a palette (built once per palette)."""
    table = _LUTS.get(palette)
    if table is None:
        stops = PALETTES.get(palette, PALETTES[PALETTE])
        positions = np.array([p for p, _ in stops], dtype=float)
        colours = np.array([c for _, c in stops], dtype=float) / 255.0
        ramp = np.linspace(0.0, 1.0, 256)
        table = np.stack(
            [np.interp(ramp, positions, colours[:, channel]) for channel in range(3)],
            axis=1,
        )
        table = np.clip(table * 255.0, 0, 255).astype(np.uint8)
        _LUTS[palette] = table
    return table


def apply_palette(values, palette=PALETTE, log=False, gamma=0.7, percentile=99.5):
    """Map a non-negative field to uint8 RGB through a paper-ish ramp.

    Scaling uses a high percentile rather than the maximum: a collapsing
    spike is orders of magnitude brighter than the filaments around it and
    would otherwise flatten everything else to paper white.
    """
    field = np.asarray(values, dtype=float)
    field = np.clip(field, 0.0, None)
    if log:
        field = np.log1p(field)
    if field.size == 0:
        return np.zeros(field.shape + (3,), dtype=np.uint8)
    # 参考量取高分位（抽样后再算，省掉一次全量排序）
    sample = field[::4, ::4]
    reference = float(np.percentile(sample, percentile))
    if reference <= 0.0:
        reference = float(field.max())
    if reference <= 0.0:
        height, width = field.shape
        return np.full((height, width, 3), PALETTES[palette][0][1], dtype=np.uint8)
    norm = np.clip(field / reference, 0.0, 1.0) ** gamma
    index = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    return palette_lut(palette)[index]


class NLSE:
    """Periodic box, complex field, split-step Fourier time stepping."""

    def __init__(self, size=192, extent=32.0, dt=0.02, seed=7, saturation=60.0):
        self.n = int(size)
        self.extent = float(extent)
        self.dx = self.extent / self.n
        self.dt = float(dt)
        self.seed = int(seed)
        # 可饱和非线性：二维立方 NLSE 的坍缩在数学上是奇异的，纯 |psi|^2
        # 项会在网格尺度上炸开；带饱和项既保持物理图像，又让亮点停在可分辨
        # 的尺度上（对应非线性介质里的饱和效应）。
        self.saturation = float(saturation)
        self.x = (np.arange(self.n) - self.n // 2) * self.dx
        self.X, self.Y = np.meshgrid(self.x, self.x, indexing="xy")
        wave = 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.dx)
        kx, ky = np.meshgrid(wave, wave, indexing="xy")
        self.K2 = kx ** 2 + ky ** 2
        self.psi = np.zeros((self.n, self.n), dtype=np.complex128)
        self._dispersion = None
        self._dispersion_dt = None
        self.steps = 0
        self.reset("单束", 0.6, 0.01)

    # ---------- 初始条件 ----------
    def reset(self, preset="单束", amplitude=0.6, noise=0.01):
        r2 = self.X ** 2 + self.Y ** 2
        rng = np.random.default_rng(self.seed)
        amp = float(amplitude)
        if preset == "单束":
            width = 2.0
            field = amp * np.exp(-r2 / (2.0 * width ** 2))
        elif preset == "涡旋":
            width, charge = 2.6, 3
            radius = np.sqrt(r2) + 1e-12
            theta = np.arctan2(self.Y, self.X)
            field = (amp * (radius / width) ** charge
                     * np.exp(-r2 / (2.0 * width ** 2))
                     * np.exp(1j * charge * theta))
        elif preset == "双束":
            width, kick = 1.5, 0.9
            left = amp * np.exp(-((self.X + 5.0) ** 2 + self.Y ** 2) / (2 * width ** 2))
            right = amp * np.exp(-((self.X - 5.0) ** 2 + self.Y ** 2) / (2 * width ** 2))
            field = (left * np.exp(1j * kick * self.X)
                     + right * np.exp(-1j * kick * self.X))
        elif preset == "噪声海":
            disc = np.exp(-r2 / (2.0 * 11.0 ** 2))
            field = amp * disc * (1.0 + 0.0j)
        else:
            raise ValueError("unknown preset: " + str(preset))
        peak = float(np.max(np.abs(field)))
        if peak > 0.0:
            field = field * (amp / peak)
        perturb = (rng.normal(size=self.psi.shape) + 1j * rng.normal(size=self.psi.shape))
        field = field * (1.0 + float(noise) * perturb)
        self.psi = field.astype(np.complex128)
        self.steps = 0
        return self.psi

    # ---------- 时间推进 ----------
    def dispersion_operator(self):
        """e^{-i k^2 dt / 2}, cached: it only changes when dt does."""
        if self._dispersion is None or self._dispersion_dt != self.dt:
            self._dispersion = np.exp(-0.5j * self.K2 * self.dt)
            self._dispersion_dt = self.dt
        return self._dispersion

    def step(self, count=1):
        """One Strang step: half of the cubic term, all of the dispersion.

        The nonlinear half step is written as cos + i sin instead of
        exp(i theta): numpy spends ~1.4x longer on the complex exponential,
        and the complex absolute value is faster than real^2 + imag^2.
        """
        psi = self.psi
        half = 0.5 * self.dt
        dispersion = self.dispersion_operator()
        sat = self.saturation
        for _ in range(int(count)):
            intensity = np.abs(psi) ** 2
            theta = half * intensity / (1.0 + intensity / sat)
            psi = psi * (np.cos(theta) + 1j * np.sin(theta))
            psi = np.fft.ifft2(np.fft.fft2(psi) * dispersion)
            intensity = np.abs(psi) ** 2
            theta = half * intensity / (1.0 + intensity / sat)
            psi = psi * (np.cos(theta) + 1j * np.sin(theta))
        self.psi = psi
        self.steps += int(count)
        return psi

    def add_seed(self, x, y, amplitude=0.9, width=1.6, phase=0.0):
        """Drop one more Gaussian beam into the running field."""
        envelope = np.exp(-((self.X - x) ** 2 + (self.Y - y) ** 2) / (2.0 * width ** 2))
        self.psi = self.psi + (amplitude * envelope * np.exp(1j * phase))
        return self.psi

    # ---------- 观测量 ----------
    def intensity(self):
        return np.abs(self.psi) ** 2

    def power(self):
        return float(np.sum(self.intensity()) * self.dx * self.dx)

    def peak(self):
        return float(np.max(self.intensity()))

    def width(self):
        """Second-moment beam radius, a scale-free measure of spreading."""
        intensity = self.intensity()
        total = float(intensity.sum())
        if total <= 0.0:
            return 0.0
        r2 = self.X ** 2 + self.Y ** 2
        return float(np.sqrt(np.sum(intensity * r2) / total))

    def render(self, palette=PALETTE, log=True):
        return apply_palette(self.intensity(), palette=palette, log=log)