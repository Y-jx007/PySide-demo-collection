# -*- coding: utf-8 -*-
"""Quantum side of four-wave mixing: two-mode squeezing and twin beams.

Undepleted pumps turn the classical gain g into a squeezing parameter
r = arccosh(sqrt(g)), and the field transformation is the textbook one:

    a_s(out) = a_s cosh r + a_i^dag sinh r
    a_i(out) = a_i cosh r + a_s^dag sinh r

Signal and idler come out photon-number correlated, so the *difference* of the
counts is quiet even though each beam alone is noisy.  Sampling vacuum input
gives the counting statistics without any operator algebra.
"""
import numpy as np


def gain_to_squeezing(gain):
    """r from the classical amplitude gain: r = arccosh(sqrt(gain))."""
    return float(np.arccosh(np.sqrt(max(float(gain), 1.0))))


def squeezing_db(gain):
    """Quadrature noise change in dB (negative = squeezed)."""
    return -20.0 / np.log(10.0) * gain_to_squeezing(gain)


def twin_beams(gain, runs=2000, seed=7):
    """Vacuum input through the two-mode squeezer; returns (n_s, n_i)."""
    rng = np.random.default_rng(seed)
    r = gain_to_squeezing(gain)
    ch, sh = np.cosh(r), np.sinh(r)

    def vacuum():
        # 真空正交分量方差 1/2
        return (rng.normal(size=runs) + 1j * rng.normal(size=runs)) / np.sqrt(2.0)

    a_s, a_i = vacuum(), vacuum()
    out_s = a_s * ch + np.conj(a_i) * sh
    out_i = a_i * ch + np.conj(a_s) * sh
    return np.abs(out_s) ** 2, np.abs(out_i) ** 2


def photon_difference_variance(gain, runs=2000, seed=7):
    n_s, n_i = twin_beams(gain, runs=runs, seed=seed)
    return float(np.var(n_s - n_i)), float(np.mean(n_s))


def shot_noise_variance(gain, runs=2000, seed=7):
    """Same mean photon number, but two independent coherent beams."""
    rng = np.random.default_rng(seed + 1)
    _, n_i = twin_beams(gain, runs=runs, seed=seed)
    mean = float(np.mean(n_i))
    return float(np.var(rng.poisson(mean, runs) - rng.poisson(mean, runs)))
