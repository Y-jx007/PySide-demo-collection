# -*- coding: utf-8 -*-
import numpy as np

from common.gpu_fft import (
    GPU_2D_MIN_SAMPLES,
    TorchFFTConvolver2D,
    fft2_magnitude_db,
    torch_available,
)


def test_fft2_matches_numpy():
    rng = np.random.default_rng(7)
    small = rng.random((64, 64), dtype=np.float32)
    got = fft2_magnitude_db(small, use_gpu=False)
    ref = 20.0 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(small))) + 1.0)
    assert np.allclose(got, ref, atol=1e-4)

    if torch_available():
        big = rng.random((256, 256), dtype=np.float32)
        got = fft2_magnitude_db(big, use_gpu=True)
        ref = 20.0 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(big))) + 1.0)
        assert big.size >= GPU_2D_MIN_SAMPLES
        assert np.allclose(got, ref, atol=2e-3)


def test_torch_convolver_matches_direct_fft():
    if not torch_available():
        return
    import torch

    rng = np.random.default_rng(11)
    n = 256
    r = 7
    kernel = rng.random((2 * r + 1, 2 * r + 1), dtype=np.float32)
    kernel /= kernel.sum()
    state = rng.random((n, n), dtype=np.float32)

    conv = TorchFFTConvolver2D()
    conv.set_kernel(kernel, r, n, version=1)
    got = conv.convolve(torch.as_tensor(state, device="cuda")).cpu().numpy()

    # Same circular convolution via NumPy.
    pad = np.zeros((n, n), dtype=np.float32)
    offsets = np.arange(-r, r + 1)
    idx = offsets % n
    pad[np.ix_(idx, idx)] = kernel
    want = np.fft.irfft2(
        np.fft.rfft2(state) * np.fft.rfft2(pad), s=(n, n)
    ).astype(np.float32)
    assert np.allclose(got, want, atol=2e-4)
