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
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
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


def hint_label(text: str) -> QLabel:
    label = QLabel(text)
    font = label.font()
    font.setPointSizeF(max(7.5, font.pointSizeF() - 1.0))
    label.setFont(font)
    return label


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
    """Common window frame for the two-canvas complex fractal apps."""

    def __init__(self, title: str, info_text: str, default_view_scale: float = 1.0):
        super().__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(1000, 700)
        self.resize(1080, 740)
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.default_view_scale = float(default_view_scale)
        self.info_text = info_text

        self.antialias_factor = 1
        self.mouse_pos = [-1.0, -1.0]
        self.realtime_update = True
        self.left_widget = None
        self.right_widget = None

        # These are filled by the subclass before setup_ui(); defaults keep the
        # shared status label and custom-function helper usable.
        self.c_real = 0.0
        self.c_imag = 0.0
        self.using_custom_function = False
        self.function_revision = 0

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self.create_control_panel())
        layout.addWidget(self.create_display_area(), 1)

    def create_control_panel(self) -> QWidget:
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        widget.setFixedHeight(176)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        param_group = self.create_parameter_group()
        param_group.setFixedWidth(250)
        status_group = self.create_status_group()
        status_group.setFixedWidth(330)
        custom_group = self.create_custom_group()
        custom_group.setMinimumWidth(330)

        layout.addWidget(param_group)
        layout.addWidget(status_group)
        layout.addWidget(custom_group, 1)
        return widget

    def create_status_group(self) -> QGroupBox:
        group = QGroupBox("状态与控制")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 18, 12, 10)
        layout.setSpacing(8)

        self.c_label = QLabel(self._c_text())
        self.c_label.setAlignment(Qt.AlignCenter)
        self.c_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.c_label.setToolTip("当前 Julia / Nova 参数 c（可选中复制）")
        label_font = self.c_label.font()
        label_font.setBold(True)
        label_font.setPointSizeF(label_font.pointSizeF() + 0.5)
        self.c_label.setFont(label_font)
        layout.addWidget(self.c_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.realtime_btn = QPushButton()
        self.realtime_btn.setFixedSize(112, 28)
        self.realtime_btn.setToolTip("切换实时更新（也可左键点击画布）")
        self.realtime_btn.clicked.connect(self.toggle_realtime_update)
        buttons.addWidget(self.realtime_btn)

        reset_btn = QPushButton("重置视图")
        reset_btn.setFixedSize(112, 28)
        reset_btn.setToolTip("回到默认缩放和视图中心")
        reset_btn.clicked.connect(self.reset_views)
        buttons.addWidget(reset_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        # 操作提示按行拆成独立标签，行距由布局控制；之前用 wordWrap 的
        # 单个标签在 320 px 宽里被压成一团。
        hints = QVBoxLayout()
        hints.setContentsMargins(2, 4, 0, 0)
        hints.setSpacing(5)
        for line in self.info_text.splitlines():
            hint = QLabel(line)
            hint.setStyleSheet("color: #555555;")
            hints.addWidget(hint)
        layout.addLayout(hints)
        layout.addStretch(1)

        self._sync_realtime_button()
        return group

    def create_display_area(self) -> QWidget:
        self.left_widget = self.create_canvas(is_left=True)
        self.right_widget = self.create_canvas(is_left=False)
        self.display_area = SquarePairArea(self.left_widget, self.right_widget)
        return self.display_area

    def create_antialias_combo(self) -> QComboBox:
        self.aa_combo = QComboBox()
        self.aa_combo.addItems(["1x (无)", "2x", "4x"])
        self.aa_combo.setFixedSize(132, 26)
        self.aa_combo.setToolTip("超采样倍率；4x 最清晰，也更耗 GPU")
        self.aa_combo.currentIndexChanged.connect(self.on_aa_changed)
        return self.aa_combo

    def create_custom_edit(self, placeholder: str, height: int) -> QPlainTextEdit:
        self.custom_edit = QPlainTextEdit()
        self.custom_edit.setFixedHeight(height)
        self.custom_edit.setPlaceholderText(placeholder)
        self.custom_edit.setToolTip("仅执行你信任的本地表达式")
        return self.custom_edit

    def create_apply_button(self) -> QPushButton:
        button = QPushButton("应用自定义函数")
        button.setFixedHeight(28)
        button.setMinimumWidth(128)
        button.setToolTip("编译并启用当前表达式")
        button.clicked.connect(self.apply_custom_function)
        return button

    # ------------------------------------------------------------------
    # Shared state changes
    # ------------------------------------------------------------------
    def _c_text(self) -> str:
        sign = "+" if self.c_imag >= 0 else "-"
        return f"c = {self.c_real:.4f} {sign} {abs(self.c_imag):.4f}i"

    def update_c_label(self):
        self.c_label.setText(self._c_text())

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
    def create_parameter_group(self) -> QGroupBox:
        raise NotImplementedError

    def create_custom_group(self) -> QGroupBox:
        raise NotImplementedError

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
