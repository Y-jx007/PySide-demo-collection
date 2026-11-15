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
                 span=20.0, seed2=0.6):
        self.mismatches = int(mismatches)
        self.samples = int(samples)
        self.length = float(length)
        self.kappa = float(kappa)
        self.span = float(span)
        self.seed2 = float(seed2)
        self.delta_k = np.linspace(-self.span, self.span, self.mismatches)
        self.configure()

    def configure(self, length=None, kappa=None, span=None):
        if length is not None:
            self.length = float(length)
        if kappa is not None:
            self.kappa = float(kappa)
        if span is not None:
            self.span = float(span)
            self.delta_k = np.linspace(-self.span, self.span, self.mismatches)
        a1, a2, a3, history = propagate_sfg(
            self.delta_k, length=self.length, samples=self.samples,
            kappa=self.kappa, seed2=self.seed2,
        )
        self.map = history.T                      # (dk, z), like SHGMap
        self._exit = np.abs(a3) ** 2
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
