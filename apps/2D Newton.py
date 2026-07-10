from custom_import import *

ti.init(arch=ti.gpu)

# -------------------- 预设函数及其导数 --------------------
@ti.func
def preset_f(z: c64, func_type: ti.i32) -> c64:
    result = c64(0.0, 0.0)
    if func_type == 0:
        result = cpow(z, 3.0) - c64(1.0, 0.0)
    elif func_type == 1:
        result = cpow(z, 4.0) - c64(1.0, 0.0)
    elif func_type == 2:
        result = cpow(z, 5.0) - c64(1.0, 0.0)
    elif func_type == 3:
        result = cpow(z, 3.0) - z
    elif func_type == 4:
        result = cpow(z, 4.0) + cpow(z, 2.0) + c64(1.0, 0.0)
    elif func_type == 5:
        result = csin(z)
    elif func_type == 6:
        result = ccos(z)
    else:
        result = cpow(z, 3.0) - c64(1.0, 0.0)
    return result

@ti.func
def preset_df(z: c64, func_type: ti.i32) -> c64:
    result = c64(0.0, 0.0)
    if func_type == 0:
        result = cmul(c64(3.0, 0.0), csqr(z))
    elif func_type == 1:
        result = cmul(c64(4.0, 0.0), cpow(z, 3.0))
    elif func_type == 2:
        result = cmul(c64(5.0, 0.0), cpow(z, 4.0))
    elif func_type == 3:
        result = cmul(c64(3.0, 0.0), csqr(z)) - c64(1.0, 0.0)
    elif func_type == 4:
        result = cmul(c64(4.0, 0.0), cpow(z, 3.0)) + cmul(c64(2.0, 0.0), z)
    elif func_type == 5:
        result = ccos(z)
    elif func_type == 6:
        result = -csin(z)
    else:
        result = cmul(c64(3.0, 0.0), csqr(z))
    return result

# -------------------- 根判定与灰度（修复非静态 return 问题） --------------------
@ti.func
def get_root_index(z: c64, func_type: ti.i32) -> ti.i32:
    """
    计算 z 最接近哪个预设根。
    为避免 Taichi 的 "return inside non-static if" 限制，所有分支均通过赋值实现，
    最后统一返回 best_idx。
    """
    min_dist = ti.cast(1e30, ti.f64)
    best_idx = 0

    if func_type == 0:   # z³ - 1
        d0 = (z.x - 1.0) * (z.x - 1.0) + z.y * z.y
        d1 = (z.x + 0.5) * (z.x + 0.5) + (z.y - 0.8660254) * (z.y - 0.8660254)
        d2 = (z.x + 0.5) * (z.x + 0.5) + (z.y + 0.8660254) * (z.y + 0.8660254)
        min_dist = d0
        if d1 < min_dist:
            min_dist = d1
            best_idx = 1
        if d2 < min_dist:
            min_dist = d2
            best_idx = 2
    elif func_type == 1: # z⁴ - 1
        d0 = (z.x - 1.0) * (z.x - 1.0) + z.y * z.y
        d1 = (z.x + 1.0) * (z.x + 1.0) + z.y * z.y
        d2 = z.x * z.x + (z.y - 1.0) * (z.y - 1.0)
        d3 = z.x * z.x + (z.y + 1.0) * (z.y + 1.0)
        min_dist = d0
        if d1 < min_dist:
            min_dist = d1
            best_idx = 1
        if d2 < min_dist:
            min_dist = d2
            best_idx = 2
        if d3 < min_dist:
            min_dist = d3
            best_idx = 3
    elif func_type == 2: # z⁵ - 1
        d0 = (z.x - 1.0) * (z.x - 1.0) + z.y * z.y
        d1 = (z.x - 0.309016994) * (z.x - 0.309016994) + (z.y - 0.951056516) * (z.y - 0.951056516)
        d2 = (z.x + 0.809016994) * (z.x + 0.809016994) + (z.y - 0.587785252) * (z.y - 0.587785252)
        d3 = (z.x + 0.809016994) * (z.x + 0.809016994) + (z.y + 0.587785252) * (z.y + 0.587785252)
        d4 = (z.x - 0.309016994) * (z.x - 0.309016994) + (z.y + 0.951056516) * (z.y + 0.951056516)
        min_dist = d0
        if d1 < min_dist:
            min_dist = d1
            best_idx = 1
        if d2 < min_dist:
            min_dist = d2
            best_idx = 2
        if d3 < min_dist:
            min_dist = d3
            best_idx = 3
        if d4 < min_dist:
            min_dist = d4
            best_idx = 4
    elif func_type == 3: # z³ - z
        d0 = z.x * z.x + z.y * z.y
        d1 = (z.x - 1.0) * (z.x - 1.0) + z.y * z.y
        d2 = (z.x + 1.0) * (z.x + 1.0) + z.y * z.y
        min_dist = d0
        if d1 < min_dist:
            min_dist = d1
            best_idx = 1
        if d2 < min_dist:
            min_dist = d2
            best_idx = 2
    elif func_type == 4: # z⁴ + z² + 1
        d0 = (z.x - 0.5) * (z.x - 0.5) + (z.y - 0.8660254) * (z.y - 0.8660254)
        d1 = (z.x - 0.5) * (z.x - 0.5) + (z.y + 0.8660254) * (z.y + 0.8660254)
        d2 = (z.x + 0.5) * (z.x + 0.5) + (z.y - 0.8660254) * (z.y - 0.8660254)
        d3 = (z.x + 0.5) * (z.x + 0.5) + (z.y + 0.8660254) * (z.y + 0.8660254)
        min_dist = d0
        if d1 < min_dist:
            min_dist = d1
            best_idx = 1
        if d2 < min_dist:
            min_dist = d2
            best_idx = 2
        if d3 < min_dist:
            min_dist = d3
            best_idx = 3
    # 对于 sin(z), cos(z) 等没有预设根的函数，best_idx 保持 0
    return best_idx

@ti.func
def root_to_gray(root_idx: ti.i32, func_type: ti.i32) -> ti.f32:
    """
    将根索引映射为灰度值。
    修复了非静态 return 问题。
    """
    gray = 0.0
    if func_type == 5 or func_type == 6:   # sin, cos
        gray = 1.0
    else:
        count = 3
        if func_type == 1: count = 4
        elif func_type == 2: count = 5
        elif func_type == 3: count = 3
        elif func_type == 4: count = 4
        gray = ti.cast(root_idx, ti.f32) / ti.cast(count - 1, ti.f32)
    return gray

# -------------------- 预设函数静态内核 --------------------
@ti.kernel
def render_preset(
    pixels: ti.template(),
    width: ti.i32, height: ti.i32,
    max_iter: ti.i32, threshold: ti.f64,
    c_real: ti.f64, c_imag: ti.f64,
    display_mode: ti.i32,
    view_scale: ti.f64, view_center_x: ti.f64, view_center_y: ti.f64,
    func_type: ti.i32
):
    ti.loop_config(block_dim=256)
    for i, j in ti.ndrange(width, height):
        aspect = ti.cast(width, ti.f64) / ti.cast(height, ti.f64)
        uv_x = (ti.cast(i, ti.f64) / ti.cast(width, ti.f64) - 0.5) * 2.0 * aspect
        uv_y = (0.5 - ti.cast(j, ti.f64) / ti.cast(height, ti.f64)) * 2.0
        uv_x = uv_x * view_scale + view_center_x
        uv_y = uv_y * view_scale + view_center_y

        z = c64(uv_x, uv_y)
        c = c64(c_real, c_imag)
        iterations = 0
        converged = False
        root_idx = -1

        for _ in range(max_iter):
            f_z = preset_f(z, func_type)
            df_z = preset_df(z, func_type)
            if (df_z.x * df_z.x + df_z.y * df_z.y) < 1e-16:
                break
            z_new = z - cdiv(f_z, df_z)
            if display_mode == 1:   # Nova
                z_new = z_new + c
            if display_mode == 0:   # Newton
                if (f_z.x * f_z.x + f_z.y * f_z.y) < threshold * threshold:
                    converged = True
                    root_idx = get_root_index(z, func_type)
                    break
            else:  # Nova
                diff = z_new - z
                if (diff.x * diff.x + diff.y * diff.y) < threshold * threshold:
                    converged = True
                    break
            z = z_new
            iterations += 1

        if converged:
            if display_mode == 0:
                gray = root_to_gray(root_idx, func_type)
                pixels[j, i] = pack_color(gray, gray, gray)
            else:
                smooth = ti.cast(iterations, ti.f64) + 1.0
                gray = 3.0 * ti.log(1.0 + ti.log(1.0 + smooth / ti.cast(max_iter, ti.f64)))
                pixels[j, i] = pack_color(ti.cast(gray, ti.f32), ti.cast(gray, ti.f32), ti.cast(gray, ti.f32))
        else:
            pixels[j, i] = pack_color(0.0, 0.0, 0.0)

# -------------------- 自定义函数动态内核 --------------------
custom_f_func = None
custom_kernel = None

def build_custom_kernel():
    global custom_kernel
    @ti.kernel
    def render_custom(
        pixels: ti.template(),
        width: ti.i32, height: ti.i32,
        max_iter: ti.i32, threshold: ti.f64,
        c_real: ti.f64, c_imag: ti.f64,
        display_mode: ti.i32,
        view_scale: ti.f64, view_center_x: ti.f64, view_center_y: ti.f64,
        num_roots: ti.i32
    ):
        ti.loop_config(block_dim=256)
        for i, j in ti.ndrange(width, height):
            aspect = ti.cast(width, ti.f64) / ti.cast(height, ti.f64)
            uv_x = (ti.cast(i, ti.f64) / ti.cast(width, ti.f64) - 0.5) * 2.0 * aspect
            uv_y = (0.5 - ti.cast(j, ti.f64) / ti.cast(height, ti.f64)) * 2.0
            uv_x = uv_x * view_scale + view_center_x
            uv_y = uv_y * view_scale + view_center_y

            z = c64(uv_x, uv_y)
            c = c64(c_real, c_imag)
            iterations = 0
            converged = False
            h = 1e-8

            for _ in range(max_iter):
                f_z = custom_f_func(z)
                f_plus  = custom_f_func(z + c64(h, 0.0))
                f_minus = custom_f_func(z - c64(h, 0.0))
                df_z = c64((f_plus.x - f_minus.x) / (2.0 * h),
                           (f_plus.y - f_minus.y) / (2.0 * h))
                if (df_z.x * df_z.x + df_z.y * df_z.y) < 1e-16:
                    break
                z_new = z - cdiv(f_z, df_z)
                if display_mode == 1:
                    z_new = z_new + c
                if display_mode == 0:
                    if (f_z.x * f_z.x + f_z.y * f_z.y) < threshold * threshold:
                        converged = True
                        break
                else:
                    diff = z_new - z
                    if (diff.x * diff.x + diff.y * diff.y) < threshold * threshold:
                        converged = True
                        break
                z = z_new
                iterations += 1

            if converged:
                if display_mode == 0:
                    angle = ti.atan2(z.y, z.x)
                    root_idx = ti.cast((angle + ti.math.pi) / (2.0 * ti.math.pi) * ti.cast(num_roots, ti.f64), ti.i32) % num_roots
                    gray = ti.select(num_roots <= 1, 0.5, ti.cast(root_idx, ti.f32) / ti.cast(num_roots - 1, ti.f32))
                    pixels[j, i] = pack_color(gray, gray, gray)
                else:
                    smooth = ti.cast(iterations, ti.f64) + 1.0
                    gray = 3.0 * ti.log(1.0 + ti.log(1.0 + smooth / ti.cast(max_iter, ti.f64)))
                    pixels[j, i] = pack_color(ti.cast(gray, ti.f32), ti.cast(gray, ti.f32), ti.cast(gray, ti.f32))
            else:
                pixels[j, i] = pack_color(0.0, 0.0, 0.0)
    custom_kernel = render_custom

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

        if app.using_custom_function and custom_kernel is not None:
            custom_kernel(
                self.taichi_field, rw, rh,
                app.max_iterations, app.convergence_threshold,
                app.c_real, app.c_imag,
                0 if self.is_left else 1,
                float(self.view_scale), float(self.view_center[0]), float(self.view_center[1]),
                app.custom_num_roots
            )
        else:
            render_preset(
                self.taichi_field, rw, rh,
                app.max_iterations, app.convergence_threshold,
                app.c_real, app.c_imag,
                0 if self.is_left else 1,
                float(self.view_scale), float(self.view_center[0]), float(self.view_center[1]),
                app.function_type
            )
        self.field_to_image(rw, rh, w, h)

# -------------------- 主窗口 --------------------
class NewtonNovaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("牛顿 / Nova 分形 — 双精度 Taichi GPU")
        self.setGeometry(100, 100, 1200, 700)

        self.max_iterations = 300
        self.convergence_threshold = 1e-6
        self.c_real = 0.0
        self.c_imag = 0.0
        self.mouse_pos = [-1.0, -1.0]
        self.function_type = 0
        self.realtime_update = True
        self.fixed_c_value = [0.0, 0.0]
        self.using_custom_function = False
        self.custom_num_roots = 3
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
            "z³-1", "z⁴-1", "z⁵-1", "z³-z",
            "z⁴+z²+1", "sin(z)", "cos(z)"
        ])
        self.function_combo.currentIndexChanged.connect(self.on_function_changed)
        self.function_combo.setFixedWidth(120)
        func_layout.addWidget(self.function_combo)
        param_layout.addLayout(func_layout)

        # 迭代次数
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("迭代:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 1000)
        self.iter_spin.setValue(self.max_iterations)
        self.iter_spin.valueChanged.connect(self.on_parameters_changed)
        self.iter_spin.setFixedWidth(120)
        iter_layout.addWidget(self.iter_spin)
        param_layout.addLayout(iter_layout)

        # 收敛阈值 (指数)
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("阈值:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(-12, -1)
        self.threshold_spin.setDecimals(0)
        self.threshold_spin.setValue(-6)
        self.threshold_spin.valueChanged.connect(self.on_threshold_changed)
        self.threshold_spin.setFixedWidth(120)
        thresh_layout.addWidget(self.threshold_spin)
        param_layout.addLayout(thresh_layout)

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

        info = QLabel("左键:切换实时更新  中键:拖拽  滚轮:缩放\n左:牛顿分形  右:Nova分形")
        status_layout.addWidget(info)

        layout.addWidget(status_group)

        # 自定义函数输入组 (含根数量)
        custom_group = QGroupBox("自定义函数 (Taichi 表达式)")
        custom_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        custom_layout = QVBoxLayout(custom_group)
        self.custom_edit = QPlainTextEdit()
        self.custom_edit.setMaximumHeight(60)
        self.custom_edit.setPlaceholderText(
            "输入 f(z) 表达式，如: cpow(z, 4.0) - c64(1.0, 0.0)\n"
            "可用: csqr, cconj, cmul, cdiv, csin, ccos, cexp, clog, cpow(z, n)"
        )
        custom_layout.addWidget(self.custom_edit)

        roots_layout = QHBoxLayout()
        roots_layout.addWidget(QLabel("根数量:"))
        self.roots_spin = QSpinBox()
        self.roots_spin.setRange(1, 20)
        self.roots_spin.setValue(self.custom_num_roots)
        roots_layout.addWidget(self.roots_spin)
        roots_layout.addStretch()
        custom_layout.addLayout(roots_layout)

        layout.addWidget(custom_group)
        return widget

    def create_display_area(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        splitter = QSplitter(Qt.Horizontal)
        self.left_widget = NewtonNovaWidget(self, is_left=True)
        self.right_widget = NewtonNovaWidget(self, is_left=False)
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
        # 牛顿分形中无需额外操作，接口与基类兼容

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
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def on_threshold_changed(self, value):
        self.convergence_threshold = 10.0 ** value
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
            w.view_scale = 1.2
            w.needs_recompute = True

    def apply_custom_function(self):
        expr = self.custom_edit.toPlainText().strip()
        if not expr:
            QMessageBox.warning(self, "警告", "请输入自定义函数表达式")
            return
        num_roots = self.roots_spin.value()
        self.custom_num_roots = num_roots

        scope = {
            'ti': ti,
            'c64': c64,
            'csqr': csqr, 'cconj': cconj, 'cmul': cmul,
            'cdiv': cdiv, 'csin': csin, 'ccos': ccos,
            'cexp': cexp, 'clog': clog, 'cpow': cpow,
        }
        try:
            func_def = f"def custom_f_func(z: c64) -> c64:\n    return {expr}\n"
            exec(func_def, scope)
            global custom_f_func
            custom_f_func = scope['custom_f_func']
            build_custom_kernel()
            self.using_custom_function = True
            self.left_widget.needs_recompute = True
            self.right_widget.needs_recompute = True
            QMessageBox.information(self, "成功", f"自定义函数已应用，根数量 = {num_roots}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"编译失败:\n{str(e)}")

        def closeEvent(self, event):
            self.left_widget._render_deferred.stop()
            self.right_widget._render_deferred.stop()
            ti.reset()
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NewtonNovaApp()
    win.show()
    sys.exit(app.exec())