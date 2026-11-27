# -*- coding: utf-8 -*-
"""Numba base: jit / njit / prange, with a pure-Python fallback.

原本只有 Traveler 自己做了 try/except ImportError 的降级；现在统一放在这里，
其他 app 也可以直接依赖 NUMBA_AVAILABLE 决定是否展示“未启用加速”的提示。

另外：``@njit(cache=True)`` 需要一个可写的缓存目录。默认位置是源码旁边的
``__pycache__``，在只读安装（Program Files、只读挂载、打包后的 onedir）里会直接
抛 ``no locator available for file ...``。这里在导入 numba 之前把
``NUMBA_CACHE_DIR`` 指到 ``runtime/``（不可写时退回系统临时目录），让这些应用
在只读环境里也能启动。

缓存目录按"入口脚本"分域：启动器（app_runner）用应用名，测试/冒烟工具用各自的
脚本名。共用一个目录时，测试用 ``spec_from_file_location`` 以别的模块名编译出的
缓存，会被真实启动时的 numba 当成同一份缓存加载，进而在 ``_rebuild_env`` 里报
``No module named '<dynamic>'``（Maxwell 就踩过）。分域之后各入口互不污染。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def _cache_scope() -> str:
    """按当前入口脚本生成缓存子目录名（app_runner 已把 sys.argv[0] 设成目标应用）。"""
    raw = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "python"
    stem = os.path.splitext(raw)[0] or "python"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)
    return (safe or "python")[:48]


def _pick_cache_dir() -> Path:
    """返回一个可写的 numba 缓存目录（按入口脚本分域）。"""
    try:
        from common.paths import ensure_runtime

        candidate = ensure_runtime() / "numba_cache" / _cache_scope()
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "pyside_tools_numba_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# numba 在 import 时读取该环境变量，所以必须在 import numba 之前设置。
# 用户显式指定的 NUMBA_CACHE_DIR 优先，不覆盖。
os.environ.setdefault("NUMBA_CACHE_DIR", str(_pick_cache_dir()))

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
