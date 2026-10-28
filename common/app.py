# -*- coding: utf-8 -*-
"""所有独立 app 共用的启动流程。

每个 app 的 __main__ 原先都是同一段样板：

    app = QApplication(sys.argv)
    ... 读 style.qss ...
    window = XxxWindow(); window.show(); sys.exit(app.exec())

OpenGL app 还各自写一遍 QSurfaceFormat。这里收敛成两行：

    if __name__ == "__main__":
        raise SystemExit(run_app(MainWindow, gl=True))
"""
from __future__ import annotations

import os
import sys

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication, QWidget

# 路径约定集中在 common.paths；这里保留同名量，兼容既有导入
from common.paths import PROJECT_ROOT, STYLE_PATH  # noqa: F401,E402


def setup_gl_format(version=(2, 1), *, profile=None, renderable_type=None,
                    swap_interval: int | None = None) -> None:
    """设置默认 OpenGL 表面格式，必须在创建 QApplication 之前调用。"""
    fmt = QSurfaceFormat()
    fmt.setVersion(*version)
    fmt.setProfile(profile or QSurfaceFormat.CompatibilityProfile)
    if renderable_type is not None:
        fmt.setRenderableType(renderable_type)
    if swap_interval is not None:
        fmt.setSwapInterval(int(swap_interval))
    QSurfaceFormat.setDefaultFormat(fmt)


def load_stylesheet(path: str | None = None) -> str:
    """读取 QSS；文件不存在时返回空字符串。"""
    path = path or STYLE_PATH
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


def apply_stylesheet(app: QApplication, path: str | None = None) -> bool:
    """把 QSS 应用到 app，返回是否成功读到样式。"""
    text = load_stylesheet(path)
    if text:
        app.setStyleSheet(text)
    return bool(text)


def create_application(argv=None, *, stylesheet: bool = False,
                       name: str | None = None) -> QApplication:
    """取回（或创建）唯一的 QApplication，并按需套用全局 QSS。"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(list(sys.argv if argv is None else argv))
    if name:
        app.setApplicationName(name)
    if stylesheet:
        apply_stylesheet(app)
    return app


def run_app(window, *, gl: bool = False, gl_options: dict | None = None,
            stylesheet: bool = False, argv=None, name: str | None = None,
            before_exec=None) -> int:
    """创建应用与窗口并进入事件循环，返回进程退出码。

    window 可以是窗口实例、窗口类，或任何零参工厂；这样 __main__ 里
    不必再关心 QApplication / QSurfaceFormat / QSS 的顺序问题。

    stylesheet 默认关闭：各 app 的配色并不统一，全局 QSS 需要显式开启，
    以免统一组件的同时顺手改掉谁的外观。
    """
    if gl:
        setup_gl_format(**(gl_options or {}))
    app = create_application(argv, stylesheet=stylesheet, name=name)
    if not isinstance(window, QWidget):
        window = window()
    window.show()
    if before_exec is not None:
        # 先让窗口/状态栏画出来，再进入事件循环（首帧要 JIT 的 app 用得上）
        before_exec(app)
    return app.exec()
