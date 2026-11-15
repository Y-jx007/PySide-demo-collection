# -*- coding: utf-8 -*-
"""Second-harmonic generation: the coupled-wave equations and phase matching.

    dA1/dz = i kappa A2 A1* exp(-i dk z)
    dA2/dz = i (kappa/2) A1^2 exp(+i dk z)

Two pump photons make one second-harmonic photon, so the conserved quantity is
the photon flux  N = |A1|^2 + 2|A2|^2  (two at omega, one at 2 omega).

Absorbing the mismatch into the second harmonic, A2 = a2 exp(+i dk z), leaves
a system with no explicit z dependence that can be integrated with RK4 for
every phase mismatch at once (|a2| = |A2|, so powers are unaffected):

    da1/dz = i kappa a2 a1*
    da2/dz = i (kappa/2) a1^2 - i dk a2

With kappa = 0 (a centrosymmetric medium) nothing happens at all: chi-2 is the
term that symmetry can switch off, and phase matching decides how efficiently
the allowed term runs.
"""
from __future__ import annotations

import numpy as np

from support.physics.nls import PALETTE, PALETTES, palette_lut


def _derivative(a1, a2, kappa, detune):
    """Right-hand side of the rotating-frame equations (arrays over dk)."""
    return (
        1j * kappa * a2 * np.conj(a1),
        0.5j * kappa * a1 * a1 - 1j * detune * a2,
    )


def propagate(delta_k, length=2.0, samples=400, kappa=1.0, amplitude=1.0,
              poling=None):
    """Integrate over the crystal.  `delta_k` may be a scalar or an array.

    `poling` is the domain-inversion period Lambda: the sign of kappa flips
    every Lambda/2, which quasi-phase-matches a finite dk.
    Returns (a1, a2) at the exit and the |A2|^2(z) history when `samples` > 1.
    """
    detune = np.asarray(delta_k, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    a1 = np.full_like(detune, float(amplitude), dtype=np.complex128)
    a2 = np.zeros_like(a1)
    history = [np.abs(a2) ** 2]
    if samples <= 1:
        return a1, a2, np.array(history)
    dz = float(length) / (samples - 1)
    for step in range(1, samples):
        if poling:
            z = step * dz
            sign = 1.0 if np.cos(2.0 * np.pi * z / float(poling)) >= 0.0 else -1.0
        else:
            sign = 1.0
        k = kappa * sign
        k1a, k1b = _derivative(a1, a2, k, detune)
        k2a, k2b = _derivative(a1 + 0.5 * dz * k1a, a2 + 0.5 * dz * k1b, k, detune)
        k3a, k3b = _derivative(a1 + 0.5 * dz * k2a, a2 + 0.5 * dz * k2b, k, detune)
        k4a, k4b = _derivative(a1 + dz * k3a, a2 + dz * k3b, k, detune)
        a1 = a1 + (dz / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
        a2 = a2 + (dz / 6.0) * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)
        history.append(np.abs(a2) ** 2)
    return a1, a2, np.array(history)


class SHGMap:
    """|A2|^2(z) for a fan of phase mismatches: the phase-matching diagram."""

    def __init__(self, mismatches=241, samples=360, length=2.0, kappa=1.0,
                 amplitude=1.0, span=20.0):
        self.mismatches = int(mismatches)
        self.samples = int(samples)
        self.length = float(length)
        self.kappa = float(kappa)
        self.amplitude = float(amplitude)
        self.span = float(span)
        self.delta_k = np.linspace(-self.span, self.span, self.mismatches)
        self.poling = None
        self.symmetric = False
        self.map = None
        self.compute()

    def configure(self, length=None, kappa=None, amplitude=None, span=None,
                  poling=None, symmetric=None):
        if length is not None:
            self.length = float(length)
        if kappa is not None:
            self.kappa = float(kappa)
        if amplitude is not None:
            self.amplitude = float(amplitude)
        if span is not None:
            self.span = float(span)
            self.delta_k = np.linspace(-self.span, self.span, self.mismatches)
        if poling is not None:
            self.poling = float(poling) if poling else None
        if symmetric is not None:
            self.symmetric = bool(symmetric)
        self.compute()

    def compute(self):
        """Integrate every mismatch at once (the state is a vector over dk)."""
        kappa = 0.0 if self.symmetric else self.kappa
        _, _, history = propagate(
            self.delta_k, length=self.length, samples=self.samples,
            kappa=kappa, amplitude=self.amplitude, poling=self.poling,
        )
        self.history = history
        peak = float(self.amplitude) ** 2 or 1.0
        # 转置成 (dk, z)：横轴传播距离，纵轴相位失配
        self.map = (history.T / peak)
        return self.map

    def efficiency(self):
        """Conversion efficiency at the exit face, one value per mismatch."""
        return self.map[:, -1]

    def peak_efficiency(self):
        return float(np.max(self.efficiency()))

    def render(self, palette=PALETTE, floor_db=30.0):
        """dB image of the efficiency map.

        Rows are phase mismatch (from -span to +span) and columns are the
        propagation distance, so a horizontal cut is one crystal and a
        vertical cut is one exit face.
        """
        field = np.clip(self.map, 0.0, None)
        reference = float(field.max())
        if reference <= 0.0:
            return np.full(self.map.shape + (3,),
                           PALETTES[palette][0][1], dtype=np.uint8)
        db = 10.0 * np.log10(np.clip(field / reference, 1e-12, 1.0))
        norm = np.clip((db + floor_db) / floor_db, 0.0, 1.0)
        index = (norm * 255.0).astype(np.uint8)
        return palette_lut(palette, gamma=1.0)[index]