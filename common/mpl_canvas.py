# -*- coding: utf-8 -*-
"""matplotlib 画布的公共封装（Fourier / Cplot / Noise 共用）。

这三处原先各自写着「new Figure + FigureCanvas + 可选工具栏」，Cplot 还要
手动把画布登记进 pyplot 的 Gcf，NavigationToolbar 的缩放/平移才生效。
这里统一成 MplCanvas：

    canvas = MplCanvas(6.4, 4.8, fixed_size=True, register=True, toolbar=True)
    canvas.figure.add_subplot(111)
    layout.addWidget(canvas)

register=True 时会把画布登记到 Gcf，并在关闭时自动注销。
"""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


def register_figure_manager(canvas):
    """把画布登记到 pyplot 的 Gcf（NavigationToolbar 的缩放/平移依赖它）。

    返回 FigureManagerBase；关闭窗口时需要传给 unregister_figure_manager。
    """
    from matplotlib._pylab_helpers import Gcf
    from matplotlib.backend_bases import FigureManagerBase

    manager = FigureManagerBase(canvas, 0)
    # FigureManagerBase 自己不建立这个连接，而 Gcf.destroy_all() 期待它存在
    manager._cidgcf = canvas.mpl_connect("close_event", lambda event: None)
    Gcf.set_active(manager)
    return manager


def unregister_figure_manager(manager) -> None:
    """注销登记（重复调用安全）。"""
    if manager is None:
        return
    try:
        from matplotlib._pylab_helpers import Gcf

        Gcf.destroy(manager)
    except Exception:      # noqa: BLE001 - 关闭阶段不该再抛异常
        pass


class MplCanvas(FigureCanvasQTAgg):
    """可直接放进 Qt 布局的 matplotlib 画布。

    参数
    ----
    width/height : Figure 尺寸（英寸）
    dpi          : 分辨率，fixed_size 时用来换算像素
    facecolor    : 画布底色，None 表示沿用 matplotlib 默认
    tight_layout : 交给 Figure 的 tight_layout（Noise 用）
    fixed_size   : 按 width/height/dpi 固定像素大小（Fourier / Cplot 用）
    toolbar      : 顺带创建一个 NavigationToolbar2QT，放在 canvas.toolbar
    toolbar_parent: 工具栏的父控件（默认挂在画布自己身上）
    register     : 登记到 Gcf，见 register_figure_manager
    """

    def __init__(self, width: float = 4.0, height: float = 3.0, dpi: int = 100,
                 *, facecolor: str | None = "white", tight_layout: bool = False,
                 fixed_size: bool = False, toolbar: bool = False,
                 toolbar_parent=None, register: bool = False, parent=None):
        figure = Figure(figsize=(width, height), dpi=dpi, tight_layout=tight_layout)
        if facecolor:
            figure.patch.set_facecolor(facecolor)
        super().__init__(figure)
        if parent is not None:
            self.setParent(parent)

        if fixed_size:
            self.setFixedSize(int(round(width * dpi)), int(round(height * dpi)))

        self.manager = register_figure_manager(self) if register else None
        self.toolbar = (NavigationToolbar2QT(self, toolbar_parent or self)
                        if toolbar else None)

    # -- 便捷方法 ----------------------------------------------------
    def add_axes(self, *args, **kwargs):
        """等价于 figure.add_subplot(...)。"""
        return self.figure.add_subplot(*args, **kwargs)

    def save(self, path: str, **kwargs) -> None:
        self.figure.savefig(path, **kwargs)

    def closeEvent(self, event):
        unregister_figure_manager(self.manager)
        self.manager = None
        super().closeEvent(event)
