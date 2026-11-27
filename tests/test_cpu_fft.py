# -*- coding: utf-8 -*-
import numpy as np

from common.cpu_fft import fft2_magnitude_db, rfft


def test_cpu_fft_matches_numpy():
    rng = np.random.default_rng(3)
    image = rng.random((96, 96), dtype=np.float32)
    got = fft2_magnitude_db(image)
    ref = 20.0 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(image))) + 1.0)
    assert got.dtype == np.float32
    assert np.allclose(got, ref, atol=1e-4)


def test_cpu_rfft():
    x = np.arange(32, dtype=np.float32)
    assert np.allclose(rfft(x), np.fft.rfft(x))
