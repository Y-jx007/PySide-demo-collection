# -*- coding: utf-8 -*-
"""Qt 控件与布局的小型积木，供所有 app 复用。

这里只放“每个 app 都要写一遍”的东西：

* LabeledSlider —— 滑块 + 数值标签（可选标题），替代各 app 里的
  _create_slider_pair / _add_slider 之类的手写重复代码；
* spinbox / double_spinbox —— 带范围、步长、回调的数值输入框；
* caption —— 标题在上、控件在下的紧凑排版；
* group_box / form_group —— 统一风格的分组框；
* hbox / vbox —— 少写几行 layout 样板代码；
* separator —— 垂直分隔线（QSS 里用 #separator 选择）；
* SquareContainer —— 让内部 OpenGL 画布保持正方形并居中。

只依赖 PySide6，不依赖项目里的其他模块，避免循环导入。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------- 布局积木
def hbox(*items, spacing=None, margins=None) -> QHBoxLayout:
    """水平布局；None 表示加一个弹簧，Layout 会被嵌套，其余当作控件。"""
    layout = QHBoxLayout()
    _fill(layout, items, spacing, margins)
    return layout


def vbox(*items, spacing=None, margins=None) -> QVBoxLayout:
    """垂直布局，规则同 hbox。"""
    layout = QVBoxLayout()
    _fill(layout, items, spacing, margins)
    return layout


def _fill(layout, items, spacing, margins) -> None:
    if margins is not None:
        layout.setContentsMargins(*margins)
    if spacing is not None:
        layout.setSpacing(spacing)
    for item in items:
        if item is None:
            layout.addStretch(1)
        elif isinstance(item, QLayout):
            layout.addLayout(item)
        else:
            layout.addWidget(item)


def group_box(title: str, *items, spacing=None, margins=None) -> QGroupBox:
    """带标题的分组框，内部默认是一个垂直布局。"""
    box = QGroupBox(title)
    box.setLayout(vbox(*items, spacing=spacing, margins=margins))
    return box


def form_group(title: str) -> tuple[QGroupBox, QFormLayout]:
    """带标题的分组框 + QFormLayout，返回 (group, form)。"""
    box = QGroupBox(title)
    form = QFormLayout(box)
    form.setSpacing(4)
    return box, form


def caption(text: str, widget: QWidget, *, width: int = 0) -> QWidget:
    """标题在上、控件在下的紧凑组合（原 Arnold._create_spinbox 的形态）。"""
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    label = QLabel(text)
    if width:
        label.setFixedWidth(width)
    layout.addWidget(label)
    layout.addWidget(widget)
    return box


def separator() -> QFrame:
    """垂直分隔线；样式通过 #separator 选择器在 style.qss 中定义。"""
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setObjectName("separator")
    return line


# ---------------------------------------------------------------- 数值输入
def spinbox(minimum: int, maximum: int, value: int, step: int = 1, *,
            callback=None, suffix: str = "", width: int = 0,
            buttons: bool = True) -> QSpinBox:
    """整数输入框；buttons=False 用于紧凑的参数行。"""
    box = QSpinBox()
    box.setRange(int(minimum), int(maximum))
    box.setSingleStep(int(step))
    box.setValue(int(value))
    if suffix:
        box.setSuffix(suffix)
    if width:
        box.setMinimumWidth(width)
    if not buttons:
        box.setButtonSymbols(QSpinBox.NoButtons)
    if callback is not None:
        box.valueChanged.connect(callback)
    return box


def double_spinbox(minimum: float, maximum: float, value: float, step: float = 0.1, *,
                   decimals: int = 3, callback=None, suffix: str = "",
                   width: int = 0, buttons: bool = True) -> QDoubleSpinBox:
    """浮点输入框，默认三位小数、步长 0.1。"""
    box = QDoubleSpinBox()
    box.setDecimals(int(decimals))
    box.setRange(float(minimum), float(maximum))
    box.setSingleStep(float(step))
    box.setValue(float(value))
    if suffix:
        box.setSuffix(suffix)
    if width:
        box.setMinimumWidth(width)
    if not buttons:
        box.setButtonSymbols(QDoubleSpinBox.NoButtons)
    if callback is not None:
        box.valueChanged.connect(callback)
    return box


# ---------------------------------------------------------------- 滑块
class LabeledSlider(QWidget):
    """滑块 + 数值标签（可选前缀标题）。

    与各 app 里手写的 _create_slider_pair 等价，但：

    * 数值标签的格式化可以自定义（formatter 参数）；
    * bind() 一行完成信号连接并同步一次初始值；
    * 通过 .slider 仍可直接拿到 QSlider，通过 .value_label 拿数值标签。

    用法::

        speed = LabeledSlider("移动速度", 20, 600, 100,
                              formatter=lambda v: f"{v / 100:.2f}")
        speed.bind(self._set_speed)
        form.addRow(speed)
    """

    valueChanged = Signal(int)

    def __init__(self, label: str = "", minimum: int = 0, maximum: int = 100,
                 value: int = 0, *, formatter=None, label_width: int = 0,
                 value_width: int = 44, slider_width: int = 0,
                 spacing: int = 4, parent=None):
        super().__init__(parent)
        self._formatter = formatter or str

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(spacing)

        self.label = QLabel(label)
        if label_width:
            self.label.setFixedWidth(label_width)
        if not label:
            # 隐藏用 hide()；此时控件还没有父窗口，setVisible(True) 会把它
            # 变成独立顶层窗口闪一下（带标题的滑块一多就是"小窗口乱闪"）
            self.label.hide()
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(minimum), int(maximum))
        self.slider.setValue(int(value))
        if slider_width:
            self.slider.setFixedWidth(slider_width)
        layout.addWidget(self.slider, 1)

        self.value_label = QLabel()
        self.value_label.setFixedWidth(value_width)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.value_label)

        # 让整行在布局里像 QSlider 一样横向吃掉多余空间（QGridLayout 尤其明显）
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._update_text(self.slider.value())
        self.slider.valueChanged.connect(self._on_slider_changed)

    # -- 取值 / 赋值 -------------------------------------------------
    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:
        self.slider.setValue(int(value))

    def set_range(self, minimum: int, maximum: int) -> None:
        self.slider.setRange(int(minimum), int(maximum))

    def set_formatter(self, formatter) -> None:
        self._formatter = formatter or str
        self._update_text(self.slider.value())

    # -- 信号 --------------------------------------------------------
    def bind(self, callback, emit_now: bool = True) -> "LabeledSlider":
        """连接回调；emit_now 为真时立刻用当前值回调一次。"""
        self.valueChanged.connect(callback)
        if emit_now:
            callback(self.value())
        return self

    def _on_slider_changed(self, value: int) -> None:
        self._update_text(value)
        self.valueChanged.emit(value)

    def _update_text(self, value: int) -> None:
        self.value_label.setText(self._formatter(value))


# ---------------------------------------------------------------- 正方形容器
class SquareContainer(QWidget):
    """让内部唯一控件保持正方形并居中，四周留出描边背景。

    3D 吸引子、Voronoi 等 OpenGL app 都需要这块“画布框”。
    """

    def __init__(self, child: QWidget, *, background: str = "#cccccc",
                 border: str = "2px solid black", parent=None):
        super().__init__(parent)
        self.child = child
        child.setParent(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(child, 0, Qt.AlignCenter)
        self.setStyleSheet(f"background-color: {background}; border: {border};")

    def resizeEvent(self, event):
        size = min(self.width(), self.height())
        self.child.setFixedSize(size, size)
        super().resizeEvent(event)
