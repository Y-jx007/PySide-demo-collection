# -*- coding: utf-8 -*-
"""CPU FFT helpers shared by visualization apps.

These are intentionally NumPy-only: small/one-shot transforms should not pay
GPU initialization or host/device transfer costs.
"""
from __future__ import annotations

import numpy as np


def fft2_magnitude_db(image):
    """Return ``20*log10(|fftshift(fft2(image))| + 1)`` as float32."""
    arr = np.asarray(image, dtype=np.float32)
    fshift = np.fft.fftshift(np.fft.fft2(arr))
    return (20.0 * np.log(np.abs(fshift) + 1.0)).astype(np.float32)


def rfft(array, *, axis=-1):
    """Real FFT on the CPU."""
    return np.fft.rfft(array, axis=axis)


def rfftfreq(n, d=1.0):
    """Sample frequencies for ``rfft``."""
    return np.fft.rfftfreq(n, d=d)
