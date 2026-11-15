# -*- coding: utf-8 -*-
"""A beam, not a plane wave: (x, z) through a badly phase-matched chi-2 crystal.

The harmonic never builds up, but the crystal still imprints a phase on the
fundamental that grows like the local intensity:

    dA/dz = (i/2k) d2A/dx2 - i (kappa^2 / dk) |A|^2 A

which is exactly a Kerr medium with n2_eff = -kappa^2/(dk k).  A Gaussian beam
gets a parabolic phase across its profile (a lens), and the picture shows that
phase growing with z: the cascaded Kerr lens.
"""
import numpy as np


class CascadedBeam:
    def __init__(self, points=192, window=12.0, length=2.0, steps=160,
                 kappa=1.0, detune=40.0, width=1.6, amplitude=1.0):
        self.n = int(points)
        self.window = float(window)
        self.dx = self.window / self.n
        self.x = (np.arange(self.n) - self.n // 2) * self.dx
        self.length = float(length)
        self.steps = int(steps)
        self.dz = self.length / self.steps
        self.kappa = float(kappa)
        self.detune = float(detune)
        self.width = float(width)
        self.amplitude = float(amplitude)
        k = 2.0 * np.pi / 0.5
        self.k = k
        self.kx = 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.dx)
        self.reset()

    def reset(self):
        self.field = self.amplitude * np.exp(-self.x ** 2 / (2.0 * self.width ** 2))
        self.z = 0.0
        self.phase_history = [np.zeros(self.n)]
        self.intensity_history = [np.abs(self.field) ** 2]

    def step(self, count=1):
        for _ in range(int(count)):
            intensity = np.abs(self.field) ** 2
            phase = -self.kappa ** 2 / max(abs(self.detune), 1e-6) * intensity * self.dz
            self.field = self.field * np.exp(1j * phase)
            spectrum = np.fft.fft(self.field) * np.exp(-0.5j * self.kx ** 2 / self.k * self.dz)
            self.field = np.fft.ifft(spectrum)
            self.z += self.dz
            self.phase_history.append(np.angle(self.field))
            self.intensity_history.append(np.abs(self.field) ** 2)

    def phase_map(self):
        """(z, x) phase, unwrapped along x so the lens shape is visible."""
        data = np.array(self.phase_history)
        centre = data.shape[1] // 2
        aligned = data - data[:, centre][:, None]
        return np.unwrap(aligned, axis=1)

class TwinBeamField:
    """Two speckled beams: the idler copies the signal up to a shrinking error."""

    def __init__(self, points=96, gain=4.0, seed=7, smoothing=3):
        self.n = int(points)
        self.gain = float(gain)
        self.seed = int(seed)
        self.smoothing = int(smoothing)

    def _speckle(self, rng):
        raw = rng.normal(size=(self.n, self.n)) + 1j * rng.normal(size=(self.n, self.n))
        k = self.smoothing
        kernel = np.ones((k, k)) / (k * k)
        from numpy.fft import fft2, ifft2
        padded = np.pad(np.real(raw), k, mode="wrap")
        smoke = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                smoke[i, j] = (padded[i:i + k, j:j + k] * kernel).sum()
        return np.abs(raw) ** 2 * (0.5 + 2.0 * smoke)

    def images(self):
        rng = np.random.default_rng(self.seed)
        signal = self._speckle(rng) * self.gain
        # 独立噪声按 1/sqrt(gain) 衰减：增益越高，两臂越像
        error = self._speckle(rng) * np.sqrt(self.gain)
        idler = signal + (error - error.mean())
        difference = signal - idler
        return signal, idler, difference
