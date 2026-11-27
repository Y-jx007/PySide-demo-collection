# -*- coding: utf-8 -*-
"""Small GPU FFT helpers shared by visualization apps.

Torch CUDA is optional and lazily imported, so apps that never use the GPU
FFT path do not pay torch's import cost.  Small transforms stay on the CPU;
GPU is used only when the caller asks and the input is large enough.
"""
from __future__ import annotations

import numpy as np

from common.cpu_fft import fft2_magnitude_db as cpu_fft2_magnitude_db
from common.cpu_fft import rfft as cpu_rfft

_torch = None
_torch_import_attempted = False

# A 2-D transform below this many samples is normally faster on the CPU.
GPU_2D_MIN_SAMPLES = 128 * 128


def _torch_module():
    global _torch, _torch_import_attempted
    if not _torch_import_attempted:
        _torch_import_attempted = True
        try:
            import torch as _t
            _torch = _t
        except ImportError:
            _torch = None
    return _torch


def torch_available():
    mod = _torch_module()
    return bool(mod is not None and mod.cuda.is_available())


def _device(device=None):
    mod = _torch_module()
    if mod is None or not mod.cuda.is_available():
        return None
    if device is None:
        return mod.device("cuda")
    return mod.device(device)


def fft2_magnitude_db(image, *, device=None, use_gpu=None):
    """Return ``20*log10(|fftshift(fft2(image))| + 1)`` as float32."""
    arr = np.asarray(image, dtype=np.float32)
    use_gpu = torch_available() if use_gpu is None else bool(use_gpu)
    use_gpu = use_gpu and arr.size >= GPU_2D_MIN_SAMPLES
    if not use_gpu:
        return cpu_fft2_magnitude_db(arr)

    mod = _torch_module()
    dev = _device(device)
    tensor = mod.as_tensor(arr, dtype=mod.float32, device=dev)
    spectrum = mod.fft.fftshift(mod.fft.fft2(tensor))
    result = 20.0 * mod.log(mod.abs(spectrum) + 1.0)
    return result.cpu().numpy().astype(np.float32, copy=False)


def rfft(array, *, axis=-1, device=None, use_gpu=None):
    """Real FFT with an optional GPU path; returns a NumPy array."""
    arr = np.asarray(array)
    use_gpu = torch_available() if use_gpu is None else bool(use_gpu)
    if not use_gpu:
        return cpu_rfft(arr, axis=axis)
    mod = _torch_module()
    dev = _device(device)
    tensor = mod.as_tensor(arr, device=dev)
    return mod.fft.rfft(tensor, dim=axis).cpu().numpy()


class TorchFFTConvolver2D:
    """Cached circular (toroidal) convolution with one fixed 2-D kernel."""

    def __init__(self, device=None):
        self._torch = _torch_module()
        if self._torch is None:
            raise RuntimeError("Torch is not installed")
        self.device = _device(device)
        if self.device is None:
            raise RuntimeError("Torch CUDA is not available")
        self._key = None
        self._shape = None
        self._kernel_fft = None

    def set_kernel(self, kernel, radius, field_size, version=None):
        """Set a normalized ``(2*radius+1, 2*radius+1)`` kernel."""
        key = (int(field_size), int(radius), version)
        if key == self._key and self._kernel_fft is not None:
            return
        mod = self._torch
        n = int(field_size)
        r = int(radius)
        k = mod.as_tensor(
            np.ascontiguousarray(kernel, dtype=np.float32),
            dtype=mod.float32, device=self.device,
        )
        pad = mod.zeros((n, n), dtype=mod.float32, device=self.device)
        offsets = mod.arange(-r, r + 1, device=self.device)
        idx = offsets % n
        pad[idx[:, None], idx[None, :]] = k
        self._kernel_fft = mod.fft.rfft2(pad)
        self._shape = (n, n)
        self._key = key

    def convolve(self, state):
        if self._kernel_fft is None:
            raise RuntimeError("set_kernel() must be called first")
        mod = self._torch
        return mod.fft.irfft2(
            mod.fft.rfft2(state) * self._kernel_fft,
            s=self._shape,
        )
