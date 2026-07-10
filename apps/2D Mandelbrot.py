from custom_import import *

ti.init(arch=ti.gpu)

# -------------------- 预设函数表 --------------------
@ti.func
def preset_func(z: c64, func_type: ti.i32) -> c64:
    result = z
    if func_type == 0:
        result = csqr(z)
    elif func_type == 1:
        result = cmul(z, csqr(z))
    elif func_type == 2:
        result = csqr(csqr(z))
    elif func_type == 3:
        z2 = csqr(z)
        z3 = cmul(z, z2)
        result = cmul(z2, z3)
    elif func_type == 4:
        result = csqr(z) + z
    elif func_type == 5:
        result = cdiv(c64(1.0, 0.0), z)
    elif func_type == 6:
        result = csin(z)
    elif func_type == 7:
        result = ccos(z)
    elif func_type == 8:
        result = cexp(z)
    elif func_type == 9:
        result = cconj(csqr(z))
    elif func_type == 10:
        result = csqr(c64(ti.abs(z.x), ti.abs(z.y)))
    else:
        result = csqr(z)
    return result

# 自定义函数默认占位
custom_func = csqr

# -------------------- 分形内核（支持自定义函数和轨道绘制） --------------------
fractal_kernel = None

def build_kernel():
    global fractal_kernel
    @ti.kernel
    def render(
        pixels: ti.template(),
        width: ti.i32, height: ti.i32,
        max_iter: ti.i32, escape_radius: ti.f64,
        c_real: ti.f64, c_imag: ti.f64,
        display_mode: ti.i32,
        view_scale: ti.f64, view_center_x: ti.f64, view_center_y: ti.f64,
        func_type: ti.i32,
        use_custom: ti.i32,
        mouse_wx: ti.f64, mouse_wy: ti.f64, mouse_valid: ti.i32
    ):
        ti.loop_config(block_dim=256)
        for i, j in ti.ndrange(width, height):
            aspect = ti.cast(width, ti.f64) / ti.cast(height, ti.f64)
            uv_x = (ti.cast(i, ti.f64) / ti.cast(width, ti.f64) - 0.5) * 2.0 * aspect
            uv_y = (0.5 - ti.cast(j, ti.f64) / ti.cast(height, ti.f64)) * 2.0
            uv_x = uv_x * view_scale + view_center_x
            uv_y = uv_y * view_scale + view_center_y

            z = c64(0.0, 0.0)
            c = c64(0.0, 0.0)
            if display_mode == 0:       # Mandelbrot
                c = c64(uv_x, uv_y)
            else:                       # Julia
                c = c64(c_real, c_imag)
                z = c64(uv_x, uv_y)

            iterations = 0
            escaped = False
            for _ in range(max_iter):
                if (z.x * z.x + z.y * z.y) > escape_radius * escape_radius:
                    escaped = True
                    break
                if use_custom:
                    z = custom_func(z) + c
                else:
                    z = preset_func(z, func_type) + c
                iterations += 1

            color_r, color_g, color_b = 1.0, 1.0, 1.0
            if escaped:
                log_zn = ti.log(z.x * z.x + z.y * z.y) / 2.0
                nu = ti.log(log_zn / ti.log(escape_radius)) / ti.log(2.0)
                smooth_iter = ti.cast(iterations, ti.f64) + 1.0 - nu
                cval = ti.cast(smooth_iter / ti.cast(max_iter, ti.f64), ti.f32)
                color_r = color_g = color_b = cval

            # 轨道绘制（仅 Mandelbrot 模式且鼠标有效时）
            if display_mode == 0 and mouse_valid:
                orbit_z = c64(0.0, 0.0)
                orbit_c = c64(mouse_wx, mouse_wy)
                for _ in range(max_iter):
                    next_z = custom_func(orbit_z) + orbit_c if use_custom else preset_func(orbit_z, func_type) + orbit_c
                    ab = next_z - orbit_z
                    ap = c64(uv_x, uv_y) - orbit_z
                    dot_ab_ab = ab.x * ab.x + ab.y * ab.y
                    t = 0.0
                    if dot_ab_ab > 1e-12:
                        t = ti.math.clamp((ap.x * ab.x + ap.y * ab.y) / dot_ab_ab, 0.0, 1.0)
                    closest = orbit_z + t * ab
                    dx = uv_x - closest.x
                    dy = uv_y - closest.y
                    dist_sq = dx * dx + dy * dy
                    threshold = 0.001 * view_scale
                    if dist_sq < threshold * threshold:
                        color_r, color_g, color_b = 0.95, 0.76, 0.07
                        break
                    orbit_z = next_z
                    if orbit_z.x * orbit_z.x + orbit_z.y * orbit_z.y > escape_radius * escape_radius:
                        break

            pixels[j, i] = pack_color(color_r, color_g, color_b)
    fractal_kernel = render

build_kernel()

# -------------------- 显示组件 --------------------
class MandelJuliaWidget(BaseFractalWidget):
    def __init__(self, parent, is_left):
        super().__init__(parent, is_left)
        self.view_scale = 1.0

    def compute_image(self):
        w, h, rw, rh = self.get_render_dims()
        if w is None:
            return
        self.ensure_taichi_field(rw, rh)
        app = self.parent_app

        # 计算鼠标世界坐标（用于轨道）
        mouse_wx, mouse_wy = 0.0, 0.0
        mouse_valid = 0
        if app.mouse_pos[0] >= 0:
            aspect = w / h if h > 0 else 1.0
            norm_x = (app.mouse_pos[0] / w - 0.5) * 2.0 * aspect
            norm_y = (0.5 - app.mouse_pos[1] / h) * 2.0
            mouse_wx = norm_x * self.view_scale + self.view_center[0]
            mouse_wy = norm_y * self.view_scale + self.view_center[1]
            if self.is_left:
                if app.realtime_update:
                    mouse_valid = 1
                else:
                    mouse_wx, mouse_wy = app.fixed_orbit_pos
                    mouse_valid = 1
            else:
                if not app.realtime_update:
                    mouse_wx, mouse_wy = app.fixed_orbit_pos
                    mouse_valid = 1

        fractal_kernel(
            self.taichi_field, rw, rh,
            app.max_iterations, float(app.escape_radius),
            app.c_real, app.c_imag,
            0 if self.is_left else 1,
            float(self.view_scale), float(self.view_center[0]), float(self.view_center[1]),
            app.function_type, 1 if app.using_custom_function else 0,
            mouse_wx, mouse_wy, mouse_valid
        )
        self.field_to_image(rw, rh, w, h)

# -------------------- 主窗口 --------------------
class ComplexDynamicsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mandelbrot / Julia 集 — 双精度 Taichi GPU")
        self.setGeometry(100, 100, 1200, 700)

        self.max_iterations = 250
        self.escape_radius = 4.0
        self.c_real = -0.7
        self.c_imag = 0.27015
        self.mouse_pos = [-1.0, -1.0]
        self.function_type = 0
        self.realtime_update = True
        self.fixed_orbit_pos = [0.0, 0.0]
        self.fixed_c_value = [-0.7, 0.27015]
        self.using_custom_function = False
        self.antialias_factor = 1

        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.addWidget(self.create_control_panel())
        main_layout.addWidget(self.create_display_area())

    def create_control_panel(self):
        widget = QWidget()
        widget.setMaximumHeight(150)
        layout = QHBoxLayout(widget)
        layout.setSpacing(6)

        # 参数组
        param_group = QGroupBox("参数设置")
        param_group.setFixedWidth(200)
        param_layout = QVBoxLayout(param_group)

        # 函数
        func_layout = QHBoxLayout()
        func_layout.addWidget(QLabel("函数:"))
        self.function_combo = QComboBox()
        self.function_combo.addItems([
            "z²", "z³", "z⁴", "z⁵", "z²+z", "1/z",
            "sin(z)", "cos(z)", "exp(z)", "conj(z²)", "(|Re|+i|Im|)²"
        ])
        self.function_combo.currentIndexChanged.connect(self.on_function_changed)
        self.function_combo.setFixedWidth(120)
        func_layout.addWidget(self.function_combo)
        param_layout.addLayout(func_layout)

        # 迭代次数
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("迭代:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(50, 50000)
        self.iter_spin.setValue(self.max_iterations)
        self.iter_spin.valueChanged.connect(self.on_parameters_changed)
        self.iter_spin.setFixedWidth(120)
        iter_layout.addWidget(self.iter_spin)
        param_layout.addLayout(iter_layout)

        # 逃逸半径
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("半径:"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(2.0, 20.0)
        self.radius_spin.setValue(self.escape_radius)
        self.radius_spin.valueChanged.connect(self.on_parameters_changed)
        self.radius_spin.setFixedWidth(120)
        radius_layout.addWidget(self.radius_spin)
        param_layout.addLayout(radius_layout)

        # 抗锯齿
        aa_layout = QHBoxLayout()
        aa_layout.addWidget(QLabel("抗锯齿:"))
        self.aa_combo = QComboBox()
        self.aa_combo.addItems(["1x (无)", "2x", "4x"])
        self.aa_combo.currentIndexChanged.connect(self.on_aa_changed)
        self.aa_combo.setFixedWidth(120)
        aa_layout.addWidget(self.aa_combo)
        param_layout.addLayout(aa_layout)

        layout.addWidget(param_group)

        # 状态与控制组
        status_group = QGroupBox("状态与控制")
        status_group.setFixedWidth(380)
        status_layout = QVBoxLayout(status_group)

        self.c_label = QLabel(f"c = {self.c_real:.4f} + {self.c_imag:.4f}i")
        self.c_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.c_label)

        btn_layout = QHBoxLayout()
        self.realtime_btn = QPushButton("实时更新: 开")
        self.realtime_btn.setFixedWidth(90)
        self.realtime_btn.clicked.connect(self.toggle_realtime_update)
        btn_layout.addWidget(self.realtime_btn)

        reset_btn = QPushButton("重置视图")
        reset_btn.setFixedWidth(90)
        reset_btn.clicked.connect(self.reset_views)
        btn_layout.addWidget(reset_btn)

        apply_btn = QPushButton("应用自定义函数")
        apply_btn.setFixedWidth(120)
        apply_btn.clicked.connect(self.apply_custom_function)
        btn_layout.addWidget(apply_btn)
        status_layout.addLayout(btn_layout)

        info = QLabel("左键:切换实时更新  中键:拖拽  滚轮:缩放\n左:Mandelbrot  右:Julia (鼠标选取c)")
        info.setWordWrap(True)
        status_layout.addWidget(info)

        layout.addWidget(status_group)

        # 自定义函数输入
        custom_group = QGroupBox("自定义函数 (Taichi 表达式)")
        custom_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        custom_layout = QVBoxLayout(custom_group)
        self.custom_edit = QPlainTextEdit()
        self.custom_edit.setMaximumHeight(80)
        self.custom_edit.setPlaceholderText(
            "输入表达式，如: csqr(z) + csin(z)\n"
            "可用: csqr, cconj, cmul, cdiv, csin, ccos, cexp, clog, cpow(z, n)"
        )
        custom_layout.addWidget(self.custom_edit)
        layout.addWidget(custom_group)

        return widget

    def create_display_area(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        splitter = QSplitter(Qt.Horizontal)
        self.left_widget = MandelJuliaWidget(self, is_left=True)
        self.right_widget = MandelJuliaWidget(self, is_left=False)
        splitter.addWidget(self.left_widget)
        splitter.addWidget(self.right_widget)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter)
        return widget

    def set_c_value(self, real, imag, update_spinners=True):
        self.c_real = real
        self.c_imag = imag
        self.c_label.setText(f"c = {real:.4f} + {imag:.4f}i")
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def set_fixed_positions(self):
        self.fixed_c_value = [self.c_real, self.c_imag]
        if self.mouse_pos[0] >= 0:
            w = self.left_widget.width()
            h = self.left_widget.height()
            aspect = w / h if h > 0 else 1.0
            norm_x = (self.mouse_pos[0] / w - 0.5) * 2.0 * aspect
            norm_y = (0.5 - self.mouse_pos[1] / h) * 2.0
            self.fixed_orbit_pos[0] = norm_x * self.left_widget.view_scale + self.left_widget.view_center[0]
            self.fixed_orbit_pos[1] = norm_y * self.left_widget.view_scale + self.left_widget.view_center[1]
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def on_function_changed(self, idx):
        self.using_custom_function = False
        self.function_type = idx
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def on_aa_changed(self, idx):
        self.antialias_factor = 2 ** idx
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def on_parameters_changed(self):
        self.max_iterations = self.iter_spin.value()
        self.escape_radius = self.radius_spin.value()
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def toggle_realtime_update(self):
        self.realtime_update = not self.realtime_update
        self.realtime_btn.setText(f"实时更新: {'开' if self.realtime_update else '关'}")
        if not self.realtime_update:
            self.set_fixed_positions()
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def reset_views(self):
        for w in (self.left_widget, self.right_widget):
            w.view_center = [0.0, 0.0]
            w.view_scale = 1.0
            w.needs_recompute = True

    def apply_custom_function(self):
        code = self.custom_edit.toPlainText().strip()
        if not code:
            QMessageBox.warning(self, "警告", "请输入自定义函数表达式")
            return

        scope = {
            'ti': ti,
            'c64': c64,
            'csqr': csqr, 'cconj': cconj, 'cmul': cmul,
            'cdiv': cdiv, 'csin': csin, 'ccos': ccos,
            'cexp': cexp, 'clog': clog, 'cpow': cpow,
        }
        try:
            func_def = f"def custom_func(z: c64) -> c64:\n    return {code}\n"
            exec(func_def, scope)
            global custom_func
            custom_func = scope['custom_func']
            build_kernel()
            self.using_custom_function = True
            self.left_widget.needs_recompute = True
            self.right_widget.needs_recompute = True
            QMessageBox.information(self, "成功", "自定义函数已应用！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"编译失败:\n{str(e)}")

    def closeEvent(self, event):
        self.left_widget.timer.stop()
        self.right_widget.timer.stop()
        ti.reset()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ComplexDynamicsApp()
    win.show()
    sys.exit(app.exec())