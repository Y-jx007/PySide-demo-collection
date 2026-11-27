"""Shared Qt scaffolding for the Mandelbrot-Julia and Newton/Nova apps.

Keeping the window skeleton, custom-expression compiler and orbit drawing in
one place means the two apps only contain their own state, controls and one
``compute_image`` method.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import taichi_forge as ti
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from common.custom_function import (
    c64,
    cconj,
    ccos,
    cdiv,
    cexp,
    clog,
    cmul,
    cpow,
    csin,
    csqr,
)
from common.dynamic_fractal import ViewTransform

ORBIT_COLOR = QColor(242, 193, 17)
ORBIT_WIDTH = 1.0

# Names available to user expressions in both apps.
_COMPLEX_SCOPE = {
    "ti": ti,
    "c64": c64,
    "csqr": csqr,
    "cconj": cconj,
    "cmul": cmul,
    "cdiv": cdiv,
    "csin": csin,
    "ccos": ccos,
    "cexp": cexp,
    "clog": clog,
    "cpow": cpow,
}


def compile_complex_function(expression: str, func_name: str = "custom_func") -> Callable:
    """Compile a trusted local expression into a Taichi-inlinable callable.

    The expression is evaluated in a restricted namespace, but this is not a
    security sandbox: only paste expressions you wrote yourself.
    """
    source = f"def {func_name}(z: c64) -> c64:\n    return {expression}\n"
    code = compile(source, f"<{func_name}>", "exec")
    scope = dict(_COMPLEX_SCOPE)
    exec(code, scope)
    return scope[func_name]


def draw_orbit(
    painter: QPainter,
    points: np.ndarray,
    transform: ViewTransform,
) -> None:
    """Draw the critical orbit as a polyline over the already rendered image."""
    if points is None or len(points) < 2:
        return
    if transform.scale_x == 0.0 or transform.scale_y == 0.0:
        return

    inv_scale_x = 1.0 / transform.scale_x
    inv_scale_y = 1.0 / transform.scale_y
    half_w = transform.width * 0.5
    half_h = transform.height * 0.5
    center_x = transform.center_x
    center_y = transform.center_y

    polygon = QPolygonF()
    for point in points:
        wx, wy = float(point[0]), float(point[1])
        if not (math.isfinite(wx) and math.isfinite(wy)):
            break
        sx = half_w + (wx - center_x) * inv_scale_x
        sy = half_h + (wy - center_y) * inv_scale_y
        polygon.append(QPointF(sx, sy))
    if len(polygon) < 2:
        return

    painter.save()
    # The original in-kernel overlay was an aliased 1-pixel trace. Keeping
    # antialiasing off also makes QPainter roughly 20x faster here.
    painter.setRenderHint(QPainter.Antialiasing, False)
    pen = QPen(ORBIT_COLOR)
    pen.setWidthF(ORBIT_WIDTH)
    painter.setPen(pen)
    painter.drawPolyline(polygon)
    painter.restore()


class SquarePairArea(QWidget):
    """Two square canvases side by side, centred with a 6 px gutter.

    The complex plane must not be letterboxed, so both canvases keep a square
    shape: the container recomputes the largest square that fits whenever the
    window is resized, instead of stretching a splitter.
    """

    GAP = 6
    MIN_SIDE = 260

    def __init__(self, left, right, parent=None):
        super().__init__(parent)
        self.left_canvas = left
        self.right_canvas = right
        for canvas in (left, right):
            canvas.setParent(self)
        self.setMinimumHeight(self.MIN_SIDE)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _side(self) -> int:
        available = max(0, self.width() - self.GAP)
        return max(self.MIN_SIDE, min(available // 2, self.height()))

    def _place(self):
        side = self._side()
        x = (self.width() - (2 * side + self.GAP)) // 2
        y = (self.height() - side) // 2
        self.left_canvas.setGeometry(x, y, side, side)
        self.right_canvas.setGeometry(x + side + self.GAP, y, side, side)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place()

    def showEvent(self, event):
        super().showEvent(event)
        self._place()


class DualCanvasFractalWindow(QMainWindow):
    """Shared window frame: slim toolbar, two square canvases, status readouts."""

    #: 自定义表达式输入框的占位提示（子类覆盖）
    custom_placeholder = "输入迭代表达式"

    def __init__(self, title: str, info_text: str, default_view_scale: float = 1.0):
        super().__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(1000, 700)
        self.resize(1080, 730)
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.default_view_scale = float(default_view_scale)
        self.info_text = info_text

        self.antialias_factor = 1
        self.mouse_pos = [-1.0, -1.0]
        self.realtime_update = True
        self.left_widget = None
        self.right_widget = None
        self.readout_widget = None
        self.custom_panel = None

        # These are filled by the subclass before setup_ui(); defaults keep the
        # shared readouts and custom-function helper usable.
        self.c_real = 0.0
        self.c_imag = 0.0
        self.using_custom_function = False
        self.function_revision = 0

    # ------------------------------------------------------------------
    # 界面骨架：上方左右两块（参数 / 自定义函数）+ 方形画布 + 底部读数
    # ------------------------------------------------------------------
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        layout.addWidget(self.create_top_area())
        layout.addWidget(self.create_display_area(), 1)
        self.create_status_bar()
        self.refresh_readouts()

    def create_top_area(self) -> QWidget:
        area = QWidget()
        area.setFixedHeight(142)
        layout = QHBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self.create_parameter_box())
        layout.addWidget(self.create_custom_box(), 1)
        return area

    def create_parameter_box(self) -> QGroupBox:
        box = QGroupBox("参数与控制")
        box.setFixedWidth(470)
        outer = QHBoxLayout(box)
        outer.setContentsMargins(12, 18, 12, 10)
        outer.setSpacing(18)

        rows = QVBoxLayout()
        rows.setSpacing(10)
        primary = QHBoxLayout()
        primary.setSpacing(16)
        secondary = QHBoxLayout()
        secondary.setSpacing(16)
        self.build_parameter_controls(primary, secondary)
        rows.addLayout(primary)
        rows.addLayout(secondary)
        rows.addStretch(1)
        outer.addLayout(rows)
        outer.addStretch(1)

        actions = QVBoxLayout()
        actions.setSpacing(10)
        self.realtime_btn = QPushButton()
        self.realtime_btn.setFixedSize(104, 28)
        self.realtime_btn.setToolTip("切换实时更新（也可左键点击画布）")
        self.realtime_btn.clicked.connect(self.toggle_realtime_update)
        actions.addWidget(self.realtime_btn)

        reset_btn = QPushButton("重置视图")
        reset_btn.setFixedSize(104, 28)
        reset_btn.setToolTip("回到默认缩放和视图中心")
        reset_btn.clicked.connect(self.reset_views)
        actions.addWidget(reset_btn)
        actions.addStretch(1)
        outer.addLayout(actions)

        self._sync_realtime_button()
        return box

    def create_custom_box(self) -> QGroupBox:
        box = QGroupBox("自定义函数")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 18, 12, 10)
        layout.setSpacing(8)

        self.custom_edit = QPlainTextEdit()
        self.custom_edit.setPlaceholderText(self.custom_placeholder)
        self.custom_edit.setMinimumHeight(56)
        self.custom_edit.setToolTip("仅执行你信任的本地表达式")
        layout.addWidget(self.custom_edit, 1)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        self.build_custom_controls(footer)
        footer.addStretch(1)
        apply_btn = QPushButton("应用")
        apply_btn.setFixedSize(84, 28)
        apply_btn.setToolTip("编译并启用当前表达式")
        apply_btn.clicked.connect(self.apply_custom_function)
        footer.addWidget(apply_btn)
        layout.addLayout(footer)
        return box

    def labeled_control(self, text: str, control: QWidget) -> QWidget:
        """Small gray label + control."""
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        label = QLabel(text)
        label.setStyleSheet("color: #777777;")
        row.addWidget(label)
        row.addWidget(control)
        return box

    def create_status_bar(self):
        status = self.statusBar()
        status.setSizeGripEnabled(False)
        status.setStyleSheet(
            "QStatusBar { border-top: 1px solid #ececea; background: #ffffff; }"
            "QStatusBar::item { border: none; }")

        fixed = QFontDatabase.systemFont(QFontDatabase.FixedFont)

        self.c_label = QLabel(self._c_text())
        self.c_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.c_label.setToolTip("当前 Julia / Nova 参数 c（可选中复制）")
        c_font = QFont(fixed)
        c_font.setBold(True)
        c_font.setPointSizeF(c_font.pointSizeF() + 1.0)
        self.c_label.setFont(c_font)
        status.addWidget(self.c_label)

        self.hint_label = QLabel(" · ".join(self.info_text.splitlines()))
        self.hint_label.setStyleSheet(
            "color: #8a8a86; padding-left: 18px;")
        status.addWidget(self.hint_label, 1)

        self.zoom_label = QLabel("缩放 1.00×")
        self.zoom_label.setTextFormat(Qt.RichText)
        zoom_font = QFont(fixed)
        zoom_font.setPointSizeF(zoom_font.pointSizeF() + 0.5)
        self.zoom_label.setFont(zoom_font)
        self.zoom_label.setStyleSheet("color: #444444; padding-right: 6px;")
        self.zoom_label.setToolTip("当前画布的放大倍数（相对默认视图）")
        status.addPermanentWidget(self.zoom_label)

    def create_display_area(self) -> QWidget:
        self.left_widget = self.create_canvas(is_left=True)
        self.right_widget = self.create_canvas(is_left=False)
        self.display_area = SquarePairArea(self.left_widget, self.right_widget)
        self.readout_widget = self.left_widget
        return self.display_area

    # ------------------------------------------------------------------
    # 读数：c 的有效位数跟随当前缩放，放大倍数单独显示
    # ------------------------------------------------------------------
    _MAX_C_DECIMALS = 15

    def _c_decimals(self) -> int:
        """小数位数 = 当前视图一个像素对应的复数步长。"""
        widget = self.left_widget
        if widget is None or widget.height() <= 0 or widget.view_scale <= 0.0:
            return 4
        pixel = 2.0 * widget.view_scale / widget.height()
        if not math.isfinite(pixel) or pixel <= 0.0:
            return 4
        return max(4, min(self._MAX_C_DECIMALS, int(math.ceil(-math.log10(pixel)))))

    def _zoom_text(self, widget) -> str:
        if widget is None or widget.view_scale <= 0.0 or widget.default_view_scale <= 0.0:
            return "缩放 —"
        zoom = widget.default_view_scale / widget.view_scale
        if zoom < 0.01:
            return "缩放 <0.01×"
        if zoom < 10.0:
            return f"缩放 {zoom:.2f}×"
        if zoom < 1000.0:
            return f"缩放 {zoom:.0f}×"
        exponent = int(math.floor(math.log10(zoom)))
        mantissa = zoom / (10.0 ** exponent)
        # 富文本上标：等宽字体里 Unicode 上标数字会掉字形，看起来像 10^1^4。
        return f"缩放 {mantissa:.1f}×10<sup>{exponent}</sup>"

    def _c_text(self) -> str:
        decimals = self._c_decimals()
        sign = "+" if self.c_imag >= 0 else "-"
        return f"c = {self.c_real:.{decimals}f} {sign} {abs(self.c_imag):.{decimals}f}i"

    def refresh_readouts(self, widget=None):
        if widget is not None:
            self.readout_widget = widget
        widget = self.readout_widget or self.left_widget
        if getattr(self, "zoom_label", None) is not None:
            self.zoom_label.setText(self._zoom_text(widget))
        if getattr(self, "c_label", None) is not None:
            self.c_label.setText(self._c_text())

    def view_changed(self, widget):
        """Canvas hook: the user zoomed/panned, update the readouts."""
        self.refresh_readouts(widget)

    def update_c_label(self):
        self.c_label.setText(self._c_text())

    # ------------------------------------------------------------------
    # Shared state changes
    # ------------------------------------------------------------------
    def mark_all_dirty(self, preview: bool = False):
        """Mark both canvases dirty.

        ``preview=True`` is for live parameter edits: draw a 1x preview now and
        let each canvas refine at the selected AA factor after the user stops.
        Structural changes (AA switch, reset, custom function) keep full quality
        immediately.
        """
        for widget in (self.left_widget, self.right_widget):
            if widget is None:
                continue
            if preview:
                widget.request_preview()
            else:
                widget.needs_recompute = True
                widget.update()

    def set_c_value(self, real: float, imag: float):
        self.c_real = float(real)
        self.c_imag = float(imag)
        self.update_c_label()
        # c only changes the right canvas (Julia/Nova). The left canvas may
        # still need a repaint for its orbit overlay, but not a full rerender.
        if self.right_widget is not None:
            self.right_widget.request_preview()
        if self.left_widget is not None:
            self.left_widget.update()

    def on_aa_changed(self, idx: int):
        self.antialias_factor = 2 ** int(idx)
        self.mark_all_dirty()

    def _sync_realtime_button(self):
        self.realtime_btn.setText(
            f"实时更新: {'开' if self.realtime_update else '关'}"
        )

    def toggle_realtime_update(self):
        self.realtime_update = not self.realtime_update
        self._sync_realtime_button()
        if not self.realtime_update:
            self.set_fixed_positions()
        self.mark_all_dirty()

    def reset_views(self):
        for widget in (self.left_widget, self.right_widget):
            if widget is None:
                continue
            widget.view_center = list(widget.default_center)
            widget.view_scale = widget.default_view_scale
            widget.needs_recompute = True
            widget.update()
        self.readout_widget = self.left_widget
        self.refresh_readouts()

    def create_antialias_combo(self) -> QComboBox:
        self.aa_combo = QComboBox()
        self.aa_combo.addItems(["无", "2×", "4×"])
        self.aa_combo.setFixedSize(76, 26)
        self.aa_combo.setToolTip("超采样倍率：无 = 1x；4× 最清晰，也更耗 GPU")
        self.aa_combo.currentIndexChanged.connect(self.on_aa_changed)
        return self.aa_combo

    def apply_custom_expression(
        self, expression: str, build: Callable[[str], None], success_message: str
    ) -> bool:
        """Run ``build`` and activate the custom function on success."""
        expression = expression.strip()
        if not expression:
            QMessageBox.warning(self, "警告", "请输入自定义函数表达式")
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            build(expression)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "错误", f"编译失败:\n{exc}")
            return False
        QApplication.restoreOverrideCursor()

        self.using_custom_function = True
        self.function_revision += 1
        self.mark_all_dirty()
        QMessageBox.information(self, "成功", success_message)
        return True

    # ------------------------------------------------------------------
    # Hooks implemented by each app
    # ------------------------------------------------------------------
    def build_parameter_controls(self, primary_row, secondary_row):
        raise NotImplementedError

    def build_custom_controls(self, layout):
        """Extra controls for the custom-function panel (e.g. root count)."""

    def create_canvas(self, is_left: bool):
        raise NotImplementedError

    def apply_custom_function(self):
        raise NotImplementedError

    def set_fixed_positions(self):
        """Called when realtime updates are switched off."""

    def closeEvent(self, event):
        for widget in (self.left_widget, self.right_widget):
            if widget is not None:
                widget.shutdown()
        super().closeEvent(event)
