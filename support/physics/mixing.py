# -*- coding: utf-8 -*-
"""Three-wave mixing with photon-flux amplitudes (SFG; DFG is the same system
run with the idler seeded instead of the signal).

With a_i normalised so that n_i = |a_i|^2 is a photon flux, one event eats one
w1 photon and one w2 photon and makes one w3 photon, so the couplings are all
the same kappa and the invariants are n1 + n3 and n2 + n3:

    da1/dz = i kappa a3 a2* e^{+i dk z}
    da2/dz = i kappa a3 a1* e^{+i dk z}
    da3/dz = i kappa a1 a2  e^{-i dk z}        dk = k3 - k1 - k2
"""
import numpy as np

from support.physics.shg import propagate


def _rhs(a1, a2, a3, kappa, detune, z):
    up = 1j * kappa * np.exp(1j * detune * z)
    down = 1j * kappa * np.exp(-1j * detune * z)
    return up * a3 * np.conj(a2), up * a3 * np.conj(a1), down * a1 * a2


def propagate_sfg(delta_k, length=2.0, samples=400, kappa=1.0,
                  seed1=1.0, seed2=0.6, seed3=0.0):
    """Integrate the three envelopes over z; returns (a1, a2, a3, history)."""
    detune = np.asarray(delta_k, dtype=float)
    dz = length / float(samples)
    a1 = np.full(detune.shape, complex(seed1))
    a2 = np.full(detune.shape, complex(seed2))
    a3 = np.full(detune.shape, complex(seed3))
    history = [np.abs(a3) ** 2]
    z = 0.0
    for _ in range(int(samples)):
        b1, b2, b3 = _rhs(a1, a2, a3, kappa, detune, z)
        c1, c2, c3 = _rhs(a1 + 0.5 * dz * b1, a2 + 0.5 * dz * b2, a3 + 0.5 * dz * b3,
                          kappa, detune, z + 0.5 * dz)
        d1, d2, d3 = _rhs(a1 + 0.5 * dz * c1, a2 + 0.5 * dz * c2, a3 + 0.5 * dz * c3,
                          kappa, detune, z + 0.5 * dz)
        e1, e2, e3 = _rhs(a1 + dz * d1, a2 + dz * d2, a3 + dz * d3,
                          kappa, detune, z + dz)
        a1 = a1 + dz * (b1 + 2.0 * c1 + 2.0 * d1 + e1) / 6.0
        a2 = a2 + dz * (b2 + 2.0 * c2 + 2.0 * d2 + e2) / 6.0
        a3 = a3 + dz * (b3 + 2.0 * c3 + 2.0 * d3 + e3) / 6.0
        z += dz
        history.append(np.abs(a3) ** 2)
    return a1, a2, a3, np.array(history)

class SFGMap:
    """n3(z) for a fan of mismatches: the same diagram, for sum frequency."""

    def __init__(self, mismatches=241, samples=360, length=2.0, kappa=1.0,
                 span=20.0, seed2=0.6, branch="和频"):
        self.mismatches = int(mismatches)
        self.samples = int(samples)
        self.length = float(length)
        self.kappa = float(kappa)
        self.span = float(span)
        self.seed2 = float(seed2)
        self.branch = branch
        self.delta_k = np.linspace(-self.span, self.span, self.mismatches)
        self.configure()

    def configure(self, length=None, kappa=None, span=None, seed2=None):
        if length is not None:
            self.length = float(length)
        if kappa is not None:
            self.kappa = float(kappa)
        if span is not None:
            self.span = float(span)
            self.delta_k = np.linspace(-self.span, self.span, self.mismatches)
        if seed2 is not None:
            self.seed2 = float(seed2)
        if self.branch == "四波混频":
            _, a4, history = propagate_fwm(
                self.delta_k, length=self.length, samples=self.samples,
                kappa=self.kappa, pump=self.seed2,
            )
            self.map = history.T
            self._exit = np.abs(a4) ** 2
            return self.map
        if self.branch == "差频":
            a1, a2, a3, history = propagate_dfg(
                self.delta_k, length=self.length, samples=self.samples,
                kappa=self.kappa, seed3=self.seed2,
            )
        else:
            a1, a2, a3, history = propagate_sfg(
                self.delta_k, length=self.length, samples=self.samples,
                kappa=self.kappa, seed2=self.seed2,
            )
        self.map = history.T                      # (dk, z), like SHGMap
        # 和频看的是 ω₃，差频看的是闲频 ω₂（不能都画 a3，否则差频会显示
        # 信号从 0.25 长到 1.25，看起来像"效率 175%"）
        self._exit = np.abs(a2) ** 2 if self.branch == "差频" else np.abs(a3) ** 2
        return self.map

    def efficiency(self):
        """Signal photon flux at the exit face, one value per mismatch."""
        return self._exit

    def peak(self):
        return float(np.max(self._exit))

    def peak_efficiency(self):
        return self.peak()

    def render(self, floor_db=30.0):
        from support.physics.nls import PALETTE, palette_lut
        field = np.clip(self.map, 0.0, None)
        reference = float(field.max())
        if reference <= 0.0:
            return np.full(self.map.shape + (3,), 250, dtype=np.uint8)
        db = 10.0 * np.log10(np.clip(field / reference, 1e-12, 1.0))
        norm = np.clip((db + floor_db) / floor_db, 0.0, 1.0)
        return palette_lut(PALETTE, gamma=1.0)[(norm * 255.0).astype(np.uint8)]

def _rhs_dfg(a1, a2, a3, kappa, detune, z):
    """Difference frequency: one w1 photon splits into w2 + w3 (w1 = w2 + w3)."""
    up = 1j * kappa * np.exp(1j * detune * z)
    down = 1j * kappa * np.exp(-1j * detune * z)
    return up * a2 * a3, down * a1 * np.conj(a3), down * a1 * np.conj(a2)


def propagate_dfg(delta_k, length=2.0, samples=400, kappa=1.0,
                  seed1=1.0, seed3=0.5):
    """Integrate the DFG branch; returns (a1, a2, a3, history of |a2|^2)."""
    detune = np.asarray(delta_k, dtype=float)
    dz = length / float(samples)
    a1 = np.full(detune.shape, complex(seed1))
    a2 = np.zeros(detune.shape, dtype=np.complex128)
    a3 = np.full(detune.shape, complex(seed3))
    history = [np.abs(a2) ** 2]
    z = 0.0
    for _ in range(int(samples)):
        b = _rhs_dfg(a1, a2, a3, kappa, detune, z)
        c = _rhs_dfg(a1 + 0.5 * dz * b[0], a2 + 0.5 * dz * b[1], a3 + 0.5 * dz * b[2],
                     kappa, detune, z + 0.5 * dz)
        d = _rhs_dfg(a1 + 0.5 * dz * c[0], a2 + 0.5 * dz * c[1], a3 + 0.5 * dz * c[2],
                     kappa, detune, z + 0.5 * dz)
        e = _rhs_dfg(a1 + dz * d[0], a2 + dz * d[1], a3 + dz * d[2],
                     kappa, detune, z + dz)
        a1 = a1 + dz * (b[0] + 2.0 * c[0] + 2.0 * d[0] + e[0]) / 6.0
        a2 = a2 + dz * (b[1] + 2.0 * c[1] + 2.0 * d[1] + e[1]) / 6.0
        a3 = a3 + dz * (b[2] + 2.0 * c[2] + 2.0 * d[2] + e[2]) / 6.0
        z += dz
        history.append(np.abs(a2) ** 2)
    return a1, a2, a3, np.array(history)

def _rhs_fwm(a3, a4, kappa, pump, detune, z):
    """Two pumps (undepleted) plus probe a3 and conjugate a4.

    dA3/dz = i kappa (|A1|^2 + |A2|^2) A3 + i kappa A1 A2 A4* e^{+i dk z}
    dA4/dz = i kappa (|A1|^2 + |A2|^2) A4 + i kappa A1 A2 A3* e^{+i dk z}
    """
    shift = 1j * kappa * (2.0 * pump * pump)
    coupling = 1j * kappa * pump * pump * np.exp(1j * detune * z)
    return shift * a3 + coupling * np.conj(a4), shift * a4 + coupling * np.conj(a3)


def fwm_conjugate(delta_k, length=2.0, samples=200, kappa=1.0, pump=0.6,
                  seed=1e-6):
    """Amplitude gain of the conjugate wave when only the probe is seeded."""
    detune = np.asarray(delta_k, dtype=float)
    dz = length / float(samples)
    a3 = np.full(detune.shape, complex(seed))
    a4 = np.full(detune.shape, complex(seed * 1e-3))
    z = 0.0
    for _ in range(int(samples)):
        b3, b4 = _rhs_fwm(a3, a4, kappa, pump, detune, z)
        c3, c4 = _rhs_fwm(a3 + 0.5 * dz * b3, a4 + 0.5 * dz * b4, kappa, pump,
                          detune, z + 0.5 * dz)
        d3, d4 = _rhs_fwm(a3 + 0.5 * dz * c3, a4 + 0.5 * dz * c4, kappa, pump,
                          detune, z + 0.5 * dz)
        e3, e4 = _rhs_fwm(a3 + dz * d3, a4 + dz * d4, kappa, pump, detune, z + dz)
        a3 = a3 + dz * (b3 + 2.0 * c3 + 2.0 * d3 + e3) / 6.0
        a4 = a4 + dz * (b4 + 2.0 * c4 + 2.0 * d4 + e4) / 6.0
        z += dz
    return np.abs(a4) / (seed * 1e-3)

def propagate_fwm(delta_k, length=2.0, samples=200, kappa=1.0, pump=0.6,
                  seed=1e-6):
    """Same as fwm_conjugate but keeps |A4|^2 at every step (for the map)."""
    detune = np.asarray(delta_k, dtype=float)
    dz = length / float(samples)
    a3 = np.full(detune.shape, complex(seed))
    a4 = np.full(detune.shape, complex(seed * 1e-3))
    history = [np.abs(a4) ** 2]
    z = 0.0
    for _ in range(int(samples)):
        b3, b4 = _rhs_fwm(a3, a4, kappa, pump, detune, z)
        c3, c4 = _rhs_fwm(a3 + 0.5 * dz * b3, a4 + 0.5 * dz * b4, kappa, pump,
                          detune, z + 0.5 * dz)
        d3, d4 = _rhs_fwm(a3 + 0.5 * dz * c3, a4 + 0.5 * dz * c4, kappa, pump,
                          detune, z + 0.5 * dz)
        e3, e4 = _rhs_fwm(a3 + dz * d3, a4 + dz * d4, kappa, pump, detune, z + dz)
        a3 = a3 + dz * (b3 + 2.0 * c3 + 2.0 * d3 + e3) / 6.0
        a4 = a4 + dz * (b4 + 2.0 * c4 + 2.0 * d4 + e4) / 6.0
        z += dz
        history.append(np.abs(a4) ** 2)
    return a3, a4, np.array(history)

class PulseSHG:
    """SHG of a pulse: dispersion plus group-velocity walk-off.

    A1_t = ... in the lab frame is tedious, so the pump frame is used and the
    harmonic walks away from it:

        dA1/dz = -i (b1/2) w^2 A1 + i kappa A2 A1* e^{-i dk z}
        dA2/dz = -i (b2/2) w^2 A2 + i (kappa/2) A1^2 e^{+i dk z}
                 + i delta w A2              (walk-off)

    delta = 1/v_g(2w) - 1/v_g(w) separates the two pulses in time, which is
    what limits the bandwidth of a real frequency-doubling crystal.
    """

    def __init__(self, points=256, window=8.0, length=1.0, steps=120,
                 kappa=1.0, delta_k=0.0, walk=0.0, beta1=0.05, beta2=0.10):
        self.n = int(points)
        self.window = float(window)
        self.dt = self.window / self.n
        self.length = float(length)
        self.steps = int(steps)
        self.kappa = float(kappa)
        self.delta_k = float(delta_k)
        self.walk = float(walk)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.dz = self.length / self.steps
        self.t = (np.arange(self.n) - self.n // 2) * self.dt
        self.omega = 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.dt)
        self.reset()

    def reset(self, width=1.2, amplitude=1.0):
        self.a1 = amplitude * np.exp(-self.t ** 2 / (2.0 * width ** 2))
        self.a2 = np.zeros(self.n, dtype=np.complex128)
        self.z = 0.0
        self.history = [np.abs(self.a2) ** 2]

    def _dispersion(self, beta):
        return np.exp(-0.5j * beta * self.omega ** 2 * self.dz)

    def step(self, count=1):
        lin1 = self._dispersion(self.beta1)
        lin2 = self._dispersion(self.beta2) * np.exp(1j * self.walk * self.omega * self.dz)
        for _ in range(int(count)):
            phase = np.exp(-1j * self.delta_k * (self.z + 0.5 * self.dz))
            src1 = 1j * self.kappa * self.a2 * np.conj(self.a1) * np.conj(phase)
            src2 = 0.5j * self.kappa * self.a1 * self.a1 * phase
            half = 0.5 * self.dz
            self.a1 = self.a1 + half * src1
            self.a2 = self.a2 + half * src2
            self.a1 = np.fft.ifft(np.fft.fft(self.a1) * lin1)
            self.a2 = np.fft.ifft(np.fft.fft(self.a2) * lin2)
            phase = np.exp(-1j * self.delta_k * (self.z + self.dz))
            src1 = 1j * self.kappa * self.a2 * np.conj(self.a1) * np.conj(phase)
            src2 = 0.5j * self.kappa * self.a1 * self.a1 * phase
            self.a1 = self.a1 + half * src1
            self.a2 = self.a2 + half * src2
            self.z += self.dz
            self.history.append(np.abs(self.a2) ** 2)
        return self.a2

    def map(self):
        """|A2(t)|^2 at every step: rows are z, columns are time."""
        return np.array(self.history)

    def efficiency(self):
        return float(np.sum(np.abs(self.a2) ** 2) / max(np.sum(np.abs(self.a1) ** 2), 1e-30))

    def shift(self):
        """Mean time offset of the harmonic relative to the pump (walk-off)."""
        i2 = np.abs(self.a2) ** 2
        total = float(np.sum(i2))
        return float(np.sum(i2 * self.t) / total) if total > 0 else 0.0

def cascaded_phase_shift(kappa=1.0, detune=40.0, length=2.0, pump=1.0,
                         samples=400):
    """Effective chi-3 from a badly phase-matched chi-2: the pump gets a phase.

    For large dk the harmonic never builds up, but it still acts back on the
    fundamental, leaving a Kerr-like phase shift dphi = -(kappa^2/dk) * I * L.
    This is the second face of the multi-mode story: the same crystal is chi-2
    locally and chi-3 effectively.
    """
    a1, a2, _ = propagate(np.array([float(detune)]), length=length,
                          samples=samples, kappa=kappa, amplitude=pump)
    return float(-np.angle(a1[0])), float(np.abs(a2[0]) ** 2)
