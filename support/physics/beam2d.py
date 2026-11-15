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
    """两束光滑高斯光斑，逐帧亮度一起起伏——孪生光束的样子。"""

    def __init__(self, points=64, frames=90, gain=4.0, seed=7, width=0.9):
        self.n = int(points)
        self.frames = int(frames)
        self.gain = float(gain)
        self.seed = int(seed)
        self.width = float(width)
        self.x = np.linspace(-3.0, 3.0, self.n)

    def place(self):
        """(x, 帧) 两张强度图：信号、闲频；差值由两者逐点相减。"""
        rng = np.random.default_rng(self.seed)
        mode = np.exp(-self.x ** 2 / (2.0 * self.width ** 2))
        # 每一帧一个强度值：信号大起伏，闲频 = 同一起伏 + 很小独立噪声
        loud = rng.gamma(shape=2.0, scale=self.gain / 2.0, size=self.frames)
        quiet = rng.normal(scale=np.sqrt(self.gain) * 0.5, size=self.frames)
        signal = mode[:, None] * loud[None, :]
        idler = mode[:, None] * (loud + quiet)[None, :]
        return signal, idler, signal - idler
