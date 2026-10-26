# -*- coding: utf-8 -*-
"""Small, testable DSP helpers for the Noise app."""
from __future__ import annotations

import numpy as np

A_WEIGHT_OFFSET = 2.0
REFERENCE_SPL = 94.0
EPS = 1e-12


def a_weighting_db(freqs):
    """IEC 61672 A-weighting in dB for an array of frequencies in Hz."""
    f = np.asarray(freqs, dtype=np.float64)
    f2 = f * f
    ra = (12194.0 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2)
        * (f2 + 12194.0 ** 2)
        * np.sqrt(f2 + 107.7 ** 2)
        * np.sqrt(f2 + 737.9 ** 2)
    )
    return 20.0 * np.log10(ra + EPS) + A_WEIGHT_OFFSET


def _windowed_power(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    if n < 16:
        raise ValueError("block too short")
    window = np.hanning(n)
    spectrum = np.fft.rfft(x * window)
    power = np.abs(spectrum) ** 2
    if len(power) > 2:
        power[1:-1] *= 2.0
    # Normalize so that sum(power) estimates mean-square pressure.
    power /= max(n * np.sum(window * window), EPS)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    return freqs, power


def spectral_power(block, fs):
    """Return FFT frequencies and normalized raw power."""
    return _windowed_power(block, fs)


def analyse_power(freqs, power, calibration_db=0.0):
    """Turn one raw power spectrum into A/Z levels and A-weighted spectrum."""
    weights = 10.0 ** (a_weighting_db(freqs) / 10.0)
    power_a = power * weights
    z_db = 10.0 * np.log10(np.sum(power) + EPS) + REFERENCE_SPL + calibration_db
    a_db = 10.0 * np.log10(np.sum(power_a) + EPS) + REFERENCE_SPL + calibration_db
    spec_db = 10.0 * np.log10(power_a + EPS) + REFERENCE_SPL + calibration_db
    return spec_db, z_db, a_db


def subtract_noise(power, noise_power, strength=1.0):
    """Spectral subtraction with a small residual floor.

    strength=0 keeps the raw spectrum, strength=1 removes the estimated
    stationary noise floor completely (except for a 2% residual).
    """
    amount = float(np.clip(strength, 0.0, 1.0))
    clean = power - amount * noise_power
    return np.maximum(clean, power * 0.02)


def analyse_block(block, fs, calibration_db=0.0):
    freqs, power = spectral_power(block, fs)
    spec_db, z_db, a_db = analyse_power(freqs, power, calibration_db)
    return freqs, spec_db, z_db, a_db


def leq(levels_db):
    levels = np.asarray(levels_db, dtype=np.float64)
    if levels.size == 0:
        return float("nan")
    return float(10.0 * np.log10(np.mean(10.0 ** (levels / 10.0)) + EPS))


def percentile_levels(levels_db, percentiles=(10, 50, 90)):
    levels = np.asarray(levels_db, dtype=np.float64)
    if levels.size == 0:
        return tuple(float("nan") for _ in percentiles)
    return tuple(float(v) for v in np.percentile(levels, percentiles))


def log_spectrum(freqs, spec_db, n_bins=128, fmin=20.0, fmax=20000.0):
    """Resample a linear FFT spectrum onto log-spaced display bins."""
    f = np.asarray(freqs, dtype=np.float64)
    s = np.asarray(spec_db, dtype=np.float64)
    mask = (f >= fmin) & (f <= fmax) & np.isfinite(s)
    if not np.any(mask):
        return np.zeros(n_bins), np.full(n_bins, -120.0)
    f = f[mask]
    s = s[mask]
    center = np.geomspace(fmin, fmax, n_bins)
    # Interpolate in log-frequency domain.
    values = np.interp(np.log(center), np.log(f), s)
    return center, values


def spectrogram_row(freqs, spec_db, n_bins=128):
    _, values = log_spectrum(freqs, spec_db, n_bins=n_bins)
    return values


def spectral_slope(freqs, spec_db, fmin=50.0, fmax=2000.0):
    """Fit PSD = C * f^-alpha over a log-log band.

    Returns (alpha, slope, intercept, r2).  For pink/1/f noise alpha is ~1,
    white noise is ~0, brown/red noise is ~2.  A-weighted spectra are used;
    the 50-2000 Hz band avoids most of the A-weighting roll-off.
    """
    f = np.asarray(freqs, dtype=np.float64)
    s = np.asarray(spec_db, dtype=np.float64)
    mask = (f >= fmin) & (f <= fmax) & np.isfinite(s)
    if int(np.sum(mask)) < 8:
        return float("nan"), float("nan"), float("nan"), float("nan")
    x = np.log10(f[mask])
    y = np.log10(10.0 ** (s[mask] / 10.0) + EPS)
    slope, intercept = np.polyfit(x, y, 1)
    alpha = -float(slope)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return alpha, float(slope), float(intercept), float(r2)


def fit_curve_db(freqs, slope, intercept):
    f = np.asarray(freqs, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * (slope * np.log10(f) + intercept)


def reference_1f_db(freqs, spec_db, f0=100.0):
    """A 1/f reference line through the measured level at f0."""
    f = np.asarray(freqs, dtype=np.float64)
    s = np.asarray(spec_db, dtype=np.float64)
    if len(f) == 0:
        return np.full_like(f, np.nan)
    i0 = int(np.argmin(np.abs(f - f0)))
    level0 = s[i0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return level0 - 10.0 * np.log10(f / max(f0, EPS))


def noise_colour(alpha):
    if not np.isfinite(alpha):
        return "–"
    if alpha < 0.0:
        return "偏蓝"
    if alpha < 0.35:
        return "偏白"
    if alpha < 0.75:
        return "粉红倾向"
    if alpha < 1.3:
        return "接近 1/f"
    if alpha < 1.8:
        return "红/棕"
    return "强 1/f 倾斜"


THIRD_OCTAVE_CENTERS = np.array([
    25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200,
    250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
    2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
], dtype=np.float64)


def third_octave_levels(freqs, spec_db, centers=None):
    """Accumulate FFT power into standard 1/3-octave bands."""
    f = np.asarray(freqs, dtype=np.float64)
    s = np.asarray(spec_db, dtype=np.float64)
    c = THIRD_OCTAVE_CENTERS if centers is None else np.asarray(centers, dtype=np.float64)
    out = np.full(len(c), np.nan, dtype=np.float64)
    for i, center in enumerate(c):
        lo = center * 2.0 ** (-1.0 / 6.0)
        hi = center * 2.0 ** (1.0 / 6.0)
        mask = (f >= lo) & (f < hi) & np.isfinite(s)
        if np.any(mask):
            out[i] = 10.0 * np.log10(np.sum(10.0 ** (s[mask] / 10.0)) + EPS)
    return c, out


def third_octave_energy(freqs, spec_db, centers=None, calibration_db=0.0):
    """Linear relative energy per 1/3-octave band, in FS^2.

    spec_db is the A-weighted spectrum expressed on the 94 dB FS convention.
    Subtracting the reference (and the user calibration) recovers relative
    mean-square energy rather than a decibel level.
    """
    f = np.asarray(freqs, dtype=np.float64)
    s = np.asarray(spec_db, dtype=np.float64) - REFERENCE_SPL - float(calibration_db)
    c = THIRD_OCTAVE_CENTERS if centers is None else np.asarray(centers, dtype=np.float64)
    out = np.full(len(c), np.nan, dtype=np.float64)
    for i, center in enumerate(c):
        lo = center * 2.0 ** (-1.0 / 6.0)
        hi = center * 2.0 ** (1.0 / 6.0)
        mask = (f >= lo) & (f < hi) & np.isfinite(s)
        if np.any(mask):
            out[i] = np.sum(10.0 ** (s[mask] / 10.0))
    return c, out


def third_octave_density(freqs, spec_db, centers=None, calibration_db=0.0):
    """Relative energy density in FS^2/Hz for each 1/3-octave band."""
    c, energy = third_octave_energy(
        freqs, spec_db, centers=centers, calibration_db=calibration_db
    )
    width = c * (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))
    return c, energy / width
