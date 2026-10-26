from common.taichi_base import (ti)
from common.custom_base import (BaseFractalWidget)
from common.qt_base import (QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, Qt, sys)

from fractal_common import (
    DualCanvasFractalWindow,
    compile_complex_function,
)
from fractal_kernels import NEWTON_FUNCTION_NAMES, newton_preset_f, render_newton

ti.init(arch=ti.gpu, offline_cache=True)

INFO_TEXT = (
    "左键 切换实时更新 · 中键 拖拽 · 滚轮 缩放\n"
    "左 牛顿分形 · 右 Nova 分形"
)
CUSTOM_PLACEHOLDER = (
    "输入 f(z) 表达式，如: cpow(z, 4.0) - c64(1.0, 0.0)\n"
    "可用: csqr, cconj, cmul, cdiv, csin,\n"
    "      ccos, cexp, clog, cpow(z, n)"
)


# -------------------- 显示组件 --------------------
class NewtonNovaWidget(BaseFractalWidget):
    def __init__(self, parent, is_left):
        super().__init__(parent, is_left)
        self.view_scale = 1.2

    def compute_image(self):
        w, h, rw, rh = self.get_render_dims()
        if w is None:
            return
        self.ensure_taichi_field(rw, rh)
        app = self.parent_app
        transform = self.view_transform(rw, rh)

        custom = bool(app.using_custom_function)
        func = app.custom_function if custom else newton_preset_f
        # func_type is ignored when a custom function is active; pin it to 0
        # so the warmed kernel and the render call share one specialization.
        func_type = 0 if custom else app.function_type
        render_newton(
            self.taichi_field, rw, rh,
            app.max_iterations, float(app.convergence_threshold),
            float(app.c_real), float(app.c_imag),
            0 if self.is_left else 1,
            transform.scale_x, transform.offset_x,
            transform.scale_y, transform.offset_y,
            int(app.custom_num_roots),
            func, custom, func_type,
        )
        self.field_to_image(rw, rh, w, h)


# -------------------- 主窗口 --------------------
class NewtonNovaApp(DualCanvasFractalWindow):
    def __init__(self):
        super().__init__(
            "牛顿 / Nova 分形 — 双精度 Taichi GPU",
            INFO_TEXT,
            default_view_scale=1.2,
        )
        self.max_iterations = 300
        self.convergence_threshold = 1e-6
        self.function_type = 0
        self.custom_function = None
        self.custom_num_roots = 3
        self._warmup_field = None
        self.setup_ui()

    # ------------------------------------------------------------------
    # 控件
    # ------------------------------------------------------------------
    def create_parameter_group(self):
        group = QGroupBox("参数设置")
        form = QFormLayout(group)
        form.setContentsMargins(12, 16, 12, 8)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(4)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        self.function_combo = QComboBox()
        self.function_combo.addItems(NEWTON_FUNCTION_NAMES)
        self.function_combo.setFixedSize(132, 26)
        self.function_combo.setToolTip("选择迭代函数；首次切换会编译专用内核")
        self.function_combo.currentIndexChanged.connect(self.on_function_changed)
        form.addRow("函数:", self.function_combo)

        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 1000)
        self.iter_spin.setValue(self.max_iterations)
        self.iter_spin.setFixedSize(132, 26)
        self.iter_spin.setToolTip("最大迭代次数")
        self.iter_spin.valueChanged.connect(self.on_parameters_changed)
        form.addRow("迭代:", self.iter_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(-12, -1)
        self.threshold_spin.setDecimals(0)
        self.threshold_spin.setValue(-6)
        self.threshold_spin.setFixedSize(132, 26)
        self.threshold_spin.setToolTip("收敛阈值（10 的幂）")
        self.threshold_spin.valueChanged.connect(self.on_threshold_changed)
        form.addRow("阈值:", self.threshold_spin)

        form.addRow("抗锯齿:", self.create_antialias_combo())
        return group

    def create_custom_group(self):
        group = QGroupBox("自定义函数 (Taichi 表达式)")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 16, 12, 10)
        layout.setSpacing(6)
        layout.addWidget(self.create_custom_edit(CUSTOM_PLACEHOLDER, 80))

        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addWidget(QLabel("根数量:"))
        self.roots_spin = QSpinBox()
        self.roots_spin.setRange(1, 20)
        self.roots_spin.setValue(self.custom_num_roots)
        self.roots_spin.setFixedSize(64, 26)
        self.roots_spin.setToolTip("自定义函数的根数量（用于角度着色）")
        self.roots_spin.valueChanged.connect(self.on_custom_roots_changed)
        footer.addWidget(self.roots_spin)
        footer.addStretch(1)
        footer.addWidget(self.create_apply_button())
        layout.addLayout(footer)
        layout.addStretch(1)
        return group

    def create_canvas(self, is_left):
        return NewtonNovaWidget(self, is_left)

    # ------------------------------------------------------------------
    # 参数与交互
    # ------------------------------------------------------------------
    def on_function_changed(self, idx):
        self.using_custom_function = False
        self.function_type = int(idx)
        self.function_revision += 1
        self.mark_all_dirty()

    def on_parameters_changed(self):
        self.max_iterations = self.iter_spin.value()
        self.mark_all_dirty()

    def on_threshold_changed(self, value):
        self.convergence_threshold = 10.0 ** value
        self.mark_all_dirty()

    def on_custom_roots_changed(self, value):
        self.custom_num_roots = int(value)
        if self.using_custom_function:
            self.mark_all_dirty()

    # ------------------------------------------------------------------
    # 自定义函数
    # ------------------------------------------------------------------
    def _warm_up_custom(self, func, num_roots):
        """Compile the custom kernel now so errors appear in the dialog."""
        if self._warmup_field is None:
            self._warmup_field = ti.field(dtype=ti.u32, shape=(1, 1))
        render_newton(
            self._warmup_field, 1, 1, 2, float(self.convergence_threshold),
            float(self.c_real), float(self.c_imag), 0,
            2.0, 0.0, -2.0, 0.0, int(num_roots),
            func, True, 0,
        )
        ti.sync()

    def _build_custom(self, expression):
        num_roots = int(self.roots_spin.value())
        func = compile_complex_function(expression, "custom_f_func")
        self._warm_up_custom(func, num_roots)
        self.custom_function = func
        self.custom_num_roots = num_roots

    def apply_custom_function(self):
        num_roots = int(self.roots_spin.value())
        self.apply_custom_expression(
            self.custom_edit.toPlainText(),
            self._build_custom,
            f"自定义函数已应用，根数量 = {num_roots}",
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NewtonNovaApp()
    win.show()
    sys.exit(app.exec())
