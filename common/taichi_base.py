# -*- coding: utf-8 -*-
import taichi_forge as ti
import taichi_forge.math as tm

_TAICHI_READY = False


def ensure_taichi():
    """Initialize Taichi once per process and reuse the runtime.

    Several support modules define Taichi fields; calling ti.init() in each
    of them resets the runtime and invalidates earlier fields.  All new
    modules call this guard instead.
    """
    global _TAICHI_READY
    if _TAICHI_READY:
        return ti
    try:
        ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=False, advanced_optimization=True)
    except Exception:
        ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False, advanced_optimization=True)
    _TAICHI_READY = True
    return ti
