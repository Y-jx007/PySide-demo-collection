from common.taichi_base import (ti)
from common.custom_base import (BaseFractalWidget, c64)
from common.qt_base import (QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QPainter, QSpinBox, QVBoxLayout, Qt, sys)

from fractal_common import (
    DualCanvasFractalWindow,
    compile_complex_function,
    draw_orbit,
    hint_label,
)
from fractal_kernels import (
    MANDELBROT_FUNCTION_NAMES,
    compute_orbit,
    mandelbrot_preset,
    render_mandelbrot,
)

ti.init(arch=ti.gpu, offline_cache=True)

INFO_TEXT = (
    "左键 切换实时更新 · 中键 拖拽 · 滚轮 缩放\n"
    "左 Mandelbrot · 右 Julia（鼠标选取 c）"
)
CUSTOM_PLACEHOLDER = (
    "输入表达式，如: csqr(z) + csin(z)\n"
    "可用: csqr, cconj, cmul, cdiv, csin,\n"
    "      ccos, cexp, clog, cpow(z, n)"
)


# -------------------- 显示组件 --------------------
class MandelJuliaWidget(BaseFractalWidget):
    def __init__(self, parent, is_left):
        super().__init__(parent, is_left)
        self.view_scale = 1.0
        self._orbit_key = None
        self._orbit_points = None

    def compute_image(self):
        w, h, rw, rh = self.get_render_dims()
        if w is None:
            return
        self.ensure_taichi_field(rw, rh)
        app = self.parent_app
        transform = self.view_transform(rw, rh)

        custom = bool(app.using_custom_function)
        func = app.custom_function if custom else mandelbrot_preset
        # func_type is ignored when a custom function is active; pin it to 0
        # so the warmed kernel and the render call share one specialization.
        func_type = 0 if custom else app.function_type
        render_mandelbrot(
            self.taichi_field, rw, rh,
            app.max_iterations, float(app.escape_radius),
            float(app.c_real), float(app.c_imag),
            0 if self.is_left else 1,
            transform.scale_x, transform.offset_x,
            transform.scale_y, transform.offset_y,
            func, custom, func_type,
        )
        self.field_to_image(rw, rh, w, h)
        app.ensure_orbit_kernel_warm()

    def _orbit_position(self):
        """Return the world-space orbit seed, or None when no mouse is set."""
        app = self.parent_app
        if app.mouse_pos[0] < 0:
            return None
        if app.realtime_update:
            return self.screen_to_world(app.mouse_pos[0], app.mouse_pos[1])
        return float(app.fixed_orbit_pos[0]), float(app.fixed_orbit_pos[1])

    def _current_orbit(self):
        position = self._orbit_position()
        if position is None:
            return None
        app = self.parent_app
        key = (
            app.function_revision,
            bool(app.using_custom_function),
            app.function_type,
            float(position[0]), float(position[1]),
            int(app.max_iterations),
            float(app.escape_radius),
        )
        if key != self._orbit_key:
            self._orbit_points = app.compute_orbit(position[0], position[1])
            self._orbit_key = key
        return self._orbit_points

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.is_left:
            return
        points = self._current_orbit()
        if points is None:
            return
        painter = QPainter(self)
        draw_orbit(painter, points, self.view_transform())
        painter.end()


# -------------------- 主窗口 --------------------
class ComplexDynamicsApp(DualCanvasFractalWindow):
    def __init__(self):
        super().__init__(
            "Mandelbrot / Julia 集 — 双精度 Taichi GPU",
            INFO_TEXT,
            default_view_scale=1.0,
        )
        self.max_iterations = 250
        self.escape_radius = 4.0
        self.c_real = -0.7
        self.c_imag = 0.27015
        self.function_type = 0
        self.custom_function = None
        self.fixed_orbit_pos = [0.0, 0.0]

        self._orbit_points_field = None
        self._orbit_length_field = None
        self._orbit_capacity = 0
        self._warmup_field = None
        self._warmup_orbit_field = None
        self._warmup_orbit_length = None
        self._orbit_warm_revision = -1
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
        self.function_combo.addItems(MANDELBROT_FUNCTION_NAMES)
        self.function_combo.setFixedSize(132, 26)
        self.function_combo.setToolTip("选择迭代函数；首次切换会编译专用内核")
        self.function_combo.currentIndexChanged.connect(self.on_function_changed)
        form.addRow("函数:", self.function_combo)

        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(50, 50000)
        self.iter_spin.setValue(self.max_iterations)
        self.iter_spin.setFixedSize(132, 26)
        self.iter_spin.setToolTip("最大迭代次数")
        self.iter_spin.valueChanged.connect(self.on_parameters_changed)
        form.addRow("迭代:", self.iter_spin)

        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(2.0, 20.0)
        self.radius_spin.setDecimals(2)
        self.radius_spin.setValue(self.escape_radius)
        self.radius_spin.setFixedSize(132, 26)
        self.radius_spin.setToolTip("逃逸半径")
        self.radius_spin.valueChanged.connect(self.on_parameters_changed)
        form.addRow("半径:", self.radius_spin)

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
        footer.addWidget(hint_label("仅执行本地可信表达式"))
        footer.addStretch(1)
        footer.addWidget(self.create_apply_button())
        layout.addLayout(footer)
        layout.addStretch(1)
        return group

    def create_canvas(self, is_left):
        return MandelJuliaWidget(self, is_left)

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
        self.escape_radius = self.radius_spin.value()
        self.mark_all_dirty()

    def set_fixed_positions(self):
        if self.mouse_pos[0] >= 0 and self.left_widget is not None:
            real, imag = self.left_widget.screen_to_world(
                self.mouse_pos[0], self.mouse_pos[1]
            )
            self.fixed_orbit_pos[0] = real
            self.fixed_orbit_pos[1] = imag

    # ------------------------------------------------------------------
    # 自定义函数
    # ------------------------------------------------------------------
    def _warm_up_custom(self, func):
        """Compile the custom kernels now so errors appear in the dialog."""
        if self._warmup_field is None:
            self._warmup_field = ti.field(dtype=ti.u32, shape=(1, 1))
        render_mandelbrot(
            self._warmup_field, 1, 1, 2, float(self.escape_radius),
            float(self.c_real), float(self.c_imag), 0,
            2.0, 0.0, -2.0, 0.0,
            func, True, 0,
        )

        if self._warmup_orbit_field is None:
            self._warmup_orbit_field = ti.field(dtype=c64, shape=(3,))
            self._warmup_orbit_length = ti.field(dtype=ti.i32, shape=())
        compute_orbit(
            self._warmup_orbit_field, self._warmup_orbit_length,
            2, 0.0, 0.0, float(self.escape_radius),
            func, True, 0,
        )
        ti.sync()

    def _build_custom(self, expression):
        func = compile_complex_function(expression, "custom_func")
        self._warm_up_custom(func)
        self.custom_function = func

    def apply_custom_function(self):
        self.apply_custom_expression(
            self.custom_edit.toPlainText(),
            self._build_custom,
            "自定义函数已应用！",
        )

    # ------------------------------------------------------------------
    # 轨道叠加
    # ------------------------------------------------------------------
    def ensure_orbit_kernel_warm(self):
        """Compile the overlay kernel with the first render of a function."""
        if self._orbit_warm_revision == self.function_revision:
            return
        # A seed just outside the escape radius keeps this throw-away orbit
        # short even when the user selected a very high iteration count.
        self.compute_orbit(float(self.escape_radius) * 2.0, 0.0)
        self._orbit_warm_revision = self.function_revision

    def compute_orbit(self, c_real, c_imag):
        max_iter = max(1, int(self.max_iterations))
        capacity = max_iter + 1
        if self._orbit_points_field is None or self._orbit_capacity != capacity:
            self._orbit_points_field = ti.field(dtype=c64, shape=(capacity,))
            self._orbit_length_field = ti.field(dtype=ti.i32, shape=())
            self._orbit_capacity = capacity

        custom = bool(self.using_custom_function)
        func = self.custom_function if custom else mandelbrot_preset
        func_type = 0 if custom else self.function_type
        compute_orbit(
            self._orbit_points_field, self._orbit_length_field,
            max_iter, float(c_real), float(c_imag), float(self.escape_radius),
            func, custom, func_type,
        )
        count = int(self._orbit_length_field[None])
        return self._orbit_points_field.to_numpy()[:count].copy()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ComplexDynamicsApp()
    win.show()
    sys.exit(app.exec())
