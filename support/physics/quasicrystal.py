# -*- coding: utf-8 -*-
"""N-fold quasicrystal: real-space plane-wave sum and its diffraction.

The pattern is the interference of N plane waves with equal wave number and
angles 2*pi*k/N.  Its Fourier transform has 2N Bragg peaks; displaying that
spectrum is the point of a quasicrystal/diffraction demo, not just the
real-space cosine field.
"""
from __future__ import annotations

import numpy as np

TAU = 2.0 * np.pi

PALETTES = {
    "冰蓝": (
        (0.00, (3, 7, 18)),
        (0.30, (14, 45, 92)),
        (0.60, (44, 138, 196)),
        (0.82, (140, 220, 238)),
        (1.00, (242, 252, 255)),
    ),
    "火焰": (
        (0.00, (5, 3, 5)),
        (0.28, (55, 12, 45)),
        (0.55, (150, 40, 52)),
        (0.78, (238, 116, 40)),
        (1.00, (255, 246, 196)),
    ),
    "热力": (
        (0.00, (4, 3, 4)),
        (0.30, (70, 16, 22)),
        (0.60, (190, 64, 22)),
        (0.84, (246, 168, 52)),
        (1.00, (255, 252, 225)),
    ),
    "灰度": (
        (0.00, (0, 0, 0)),
        (1.00, (255, 255, 255)),
    ),
}


def apply_palette(values, palette="冰蓝"):
    """Map a float image in [0, 1] to uint8 RGB with a small LUT."""
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    stops = PALETTES.get(palette, PALETTES["冰蓝"])
    positions = np.array([s[0] for s in stops], dtype=np.float64)
    colors = np.array([s[1] for s in stops], dtype=np.float64)
    out = np.empty(values.shape + (3,), dtype=np.uint8)
    for channel in range(3):
        out[..., channel] = np.interp(
            values, positions, colors[:, channel]
        ).astype(np.uint8)
    return out


class Quasicrystal:
    """Plane-wave quasicrystal and its Fourier diffraction pattern."""

    def __init__(self, size=512):
        self.size = int(size)
        self.n = 7
        self.wave_number = 5.0
        self.zoom = 12.0
        self.rotation = 0.0
        self.phase = 0.0
        self.diffraction_zoom = 4.0

    def _axes(self):
        coords = np.linspace(
            -float(self.zoom), float(self.zoom), self.size
        )
        return np.meshgrid(coords, coords)

    def real_space(self):
        """Return the normalized [-1, 1] interference field."""
        xx, yy = self._axes()
        field = np.zeros_like(xx, dtype=np.float64)
        for k in range(int(self.n)):
            theta = float(self.rotation) + TAU * k / max(1, int(self.n))
            kx = np.cos(theta) * float(self.wave_number)
            ky = np.sin(theta) * float(self.wave_number)
            field += np.cos(kx * xx + ky * yy + float(self.phase))
        return field / max(1, int(self.n))

    def diffraction(self, log=True, zoom=None):
        """Return the normalized |FFT|^2 spectrum of the real-space pattern."""
        field = self.real_space()
        window = np.outer(
            np.hanning(self.size), np.hanning(self.size)
        )
        spectrum = np.abs(
            np.fft.fftshift(np.fft.fft2(field * window))
        ) ** 2
        if log:
            spectrum = np.log1p(spectrum)
        zoom = self.diffraction_zoom if zoom is None else float(zoom)
        if zoom > 1.0:
            half = max(16, int(self.size / (2.0 * zoom)))
            center = self.size // 2
            spectrum = spectrum[
                center - half:center + half,
                center - half:center + half,
            ]
        peak = float(spectrum.max())
        if peak > 0.0:
            spectrum = spectrum / peak
        return spectrum

    def render_real(self, palette="冰蓝"):
        field = (np.clip(self.real_space(), -1.0, 1.0) + 1.0) * 0.5
        return apply_palette(field, palette)

    def render_diffraction(self, palette="火焰", log=True, zoom=None):
        return apply_palette(
            self.diffraction(log=log, zoom=zoom), palette
        )
