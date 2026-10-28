# -*- coding: utf-8 -*-
"""Numba base: jit / njit / prange, with a pure-Python fallback.

原本只有 Traveler 自己做了 try/except ImportError 的降级；现在统一放在这里，
其他 app 也可以直接依赖 NUMBA_AVAILABLE 决定是否展示“未启用加速”的提示。
"""
from __future__ import annotations

try:
    from numba import jit, njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于运行环境
    NUMBA_AVAILABLE = False
    prange = range

    def _decorator(*args, **kwargs):
        """同时支持 @jit 与 @jit(...) 两种写法。"""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(func):
            return func

        return wrap

    jit = njit = _decorator
