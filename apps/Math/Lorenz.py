from custom_import import *

# ---------- Taichi 初始化 ----------
ti.init(arch=ti.cpu, debug=False)

# ---------- 通用 RK4 步进器与泛型积分内核 ----------
@ti.func
def rk4_step(x: ti.f64, y: ti.f64, z: ti.f64,
             dt: ti.f64,
             deriv: ti.template(),
             p0: ti.f64, p1: ti.f64, p2: ti.f64,
             p3: ti.f64, p4: ti.f64, p5: ti.f64):
    """执行一步 RK4 积分，返回新的 (x, y, z) 三个标量"""
    # k1
    k1 = deriv(x, y, z, p0, p1, p2, p3, p4, p5)
    # k2
    k2 = deriv(x + 0.5 * dt * k1[0],
               y + 0.5 * dt * k1[1],
               z + 0.5 * dt * k1[2],
               p0, p1, p2, p3, p4, p5)
    # k3
    k3 = deriv(x + 0.5 * dt * k2[0],
               y + 0.5 * dt * k2[1],
               z + 0.5 * dt * k2[2],
               p0, p1, p2, p3, p4, p5)
    # k4
    k4 = deriv(x + dt * k3[0],
               y + dt * k3[1],
               z + dt * k3[2],
               p0, p1, p2, p3, p4, p5)

    x_new = x + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    y_new = y + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    z_new = z + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return x_new, y_new, z_new


@ti.kernel
def generic_integrate(
    x_in: ti.f64, y_in: ti.f64, z_in: ti.f64,
    p0: ti.f64, p1: ti.f64, p2: ti.f64,
    p3: ti.f64, p4: ti.f64, p5: ti.f64,
    dt: ti.f64, n: ti.i32,
    out: ti.types.ndarray(dtype=ti.f64, ndim=2),
    start: ti.i32,
    deriv: ti.template()
):
    """泛型积分内核，通过模板参数 deriv 指定导数函数"""
    x, y, z = x_in, y_in, z_in
    for i in range(n):
        x, y, z = rk4_step(x, y, z, dt, deriv, p0, p1, p2, p3, p4, p5)
        out[start + i, 0] = x
        out[start + i, 1] = y
        out[start + i, 2] = z


# ---------- 各吸引子的导数函数（统一的 6 参数接口）----------
@ti.func
def deriv_lorenz(x, y, z, p0, p1, p2, p3, p4, p5):
    sigma, rho, beta = p0, p1, p2
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return ti.math.vec3(dx, dy, dz)


@ti.func
def deriv_rossler(x, y, z, p0, p1, p2, p3, p4, p5):
    a, b, c = p0, p1, p2
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return ti.math.vec3(dx, dy, dz)


@ti.func
def deriv_chen(x, y, z, p0, p1, p2, p3, p4, p5):
    a, b, c = p0, p1, p2
    dx = a * (y - x)
    dy = (c - a) * x - x * z + c * y
    dz = x * y - b * z
    return ti.math.vec3(dx, dy, dz)


@ti.func
def deriv_aizawa(x, y, z, p0, p1, p2, p3, p4, p5):
    a, b, c, d, e, f = p0, p1, p2, p3, p4, p5
    dx = (z - b) * x - d * y
    dy = d * x + (z - b) * y
    dz = c + a * z - (z ** 3) / 3.0 - (x ** 2 + y ** 2) * (1.0 + e * z) + f * z * x ** 3
    return ti.math.vec3(dx, dy, dz)


# ---------- 预设配置 ----------
PRESETS = {
    "Lorenz": (deriv_lorenz, ["σ", "ρ", "β", "", "", ""],
               [10.0, 28.0, 8.0 / 3, 0, 0, 0]),
    "Rössler": (deriv_rossler, ["a", "b", "c", "", "", ""],
                [0.2, 0.2, 5.7, 0, 0, 0]),
    "Chen": (deriv_chen, ["a", "b", "c", "", "", ""],
             [35.0, 3.0, 28.0, 0, 0, 0]),
    "Aizawa": (deriv_aizawa, ["a", "b", "c", "d", "e", "f"],
               [0.95, 0.7, 0.6, 3.5, 0.25, 0.1]),
    "Custom": (None, ["p0", "p1", "p2", "p3", "p4", "p5"], [1.0] * 6)
}

FORMULAS = {
    "Lorenz": "dx/dt = σ(y - x)\ndy/dt = x(ρ - z) - y\ndz/dt = xy - βz",
    "Rössler": "dx/dt = -y - z\ndy/dt = x + ay\ndz/dt = b + z(x - c)",
    "Chen": "dx/dt = a(y - x)\ndy/dt = (c - a)x - xz + cy\ndz/dt = xy - bz",
    "Aizawa": "dx/dt = (z - b)x - dy\ndy/dt = dx + (z - b)y\ndz/dt = c + az - z³/3 - (x²+y²)(1+ez) + fzx³",
    "Custom": "自定义方程：在下方输入 dx/dt, dy/dt, dz/dt"
}


# ---------- 3D 渲染组件 ----------
class AttractorGLWidget(QOpenGLWidget):
    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.trail = np.empty((0, 3), dtype=np.float32)
        self.line_color = QColor(0, 0, 0)
        self.bg_color = QColor(230, 230, 230)
        self.last_mouse = None

    def initializeGL(self):
        glClearColor(*self.bg_color.getRgbF()[:3], 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glDisable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT)
        if self.trail.shape[0] < 2:
            return

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width() / max(self.height(), 1)
        proj = QMatrix4x4()
        proj.perspective(45.0, aspect, 0.1, 1000.0)
        glLoadMatrixf(proj.data())

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        mv = self.camera.view_matrix() * self.camera.model_matrix()
        glLoadMatrixf(mv.data())

        glColor3f(*self.line_color.getRgbF()[:3])
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.trail.tobytes())
        glDrawArrays(GL_LINE_STRIP, 0, self.trail.shape[0])
        glDisableClientState(GL_VERTEX_ARRAY)

    def update_trail(self, points, max_len=50000):
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        self.trail = np.concatenate([self.trail, points.astype(np.float32)])
        if self.trail.shape[0] > max_len:
            self.trail = self.trail[-max_len:]

    def clear_trail(self):
        self.trail = np.empty((0, 3), dtype=np.float32)

    def mousePressEvent(self, event):
        self.last_mouse = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse is None:
            return
        dx = event.pos().x() - self.last_mouse.x()
        dy = event.pos().y() - self.last_mouse.y()
        btn = event.buttons()
        if btn & Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier:
                self.camera.rotate_model(0, 0, dx * 0.5)
            else:
                self.camera.rotate_model(dy * 0.5, dx * 0.5, 0)
        elif btn & Qt.MiddleButton:
            self.camera.pan(-dx * 0.1, dy * 0.1)
        self.last_mouse = event.pos()
        self.update()

    def wheelEvent(self, event):
        self.camera.zoom(event.angleDelta().y() * 0.05)
        self.update()


class SquareGLContainer(QWidget):
    """强制内部 AttractorGLWidget 保持正方形，并居中显示"""
    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.gl_widget = AttractorGLWidget(camera, parent=self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.gl_widget, 0, Qt.AlignCenter)
        self.setStyleSheet("background-color: #cccccc; border: 2px solid black;")

    def resizeEvent(self, event):
        size = min(self.width(), self.height())
        self.gl_widget.setFixedSize(size, size)
        super().resizeEvent(event)


# ---------- 主窗口 ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D 吸引子 (Taichi)")
        self.resize(1100, 680)

        self.running = False
        self.current_preset = "Lorenz"
        self.deriv_func = PRESETS["Lorenz"][0]          # 当前导数函数
        self.custom_code = [None, None, None]
        self.x, self.y, self.z = 0.1, 0.0, 0.0
        self.params = list(PRESETS["Lorenz"][2])
        self.dt = 0.005
        self.steps_per_frame = 80

        self.camera = OrbitCamera()
        self._init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(16)

        self.on_preset_changed("Lorenz")
        self.reset_sim()

    def _init_ui(self):
        central = QSplitter(Qt.Horizontal)
        self.setCentralWidget(central)

        # ---- 左侧控制面板 ----
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)

        row = 0

        # 预设选择
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("预设:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS.keys())
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        grid.addLayout(preset_layout, row, 0, 1, 2)
        row += 1

        # 参数组
        param_group = QGroupBox("参数")
        param_grid = QGridLayout(param_group)
        param_grid.setSpacing(4)
        self.param_spins = []
        self.param_labels = []
        for i in range(6):
            label = QLabel("")
            spin = QDoubleSpinBox()
            spin.setRange(-100, 100)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            self.param_labels.append(label)
            self.param_spins.append(spin)
            col = i // 3
            row_in = i % 3
            hb = QHBoxLayout()
            hb.addWidget(label)
            hb.addWidget(spin)
            hb.addStretch()
            param_grid.addLayout(hb, row_in, col)
        grid.addWidget(param_group, row, 0, 1, 2)
        row += 1

        # 初始条件 + 模拟设置
        init_group = QGroupBox("初始条件")
        init_form = QFormLayout(init_group)
        self.x0_spin = QDoubleSpinBox(); self.x0_spin.setRange(-100,100); self.x0_spin.setValue(0.1)
        self.y0_spin = QDoubleSpinBox(); self.y0_spin.setRange(-100,100); self.y0_spin.setValue(0.0)
        self.z0_spin = QDoubleSpinBox(); self.z0_spin.setRange(-100,100); self.z0_spin.setValue(0.0)
        init_form.addRow("x₀:", self.x0_spin)
        init_form.addRow("y₀:", self.y0_spin)
        init_form.addRow("z₀:", self.z0_spin)
        grid.addWidget(init_group, row, 0)

        sim_group = QGroupBox("模拟设置")
        sim_form = QFormLayout(sim_group)
        self.dt_spin = QDoubleSpinBox(); self.dt_spin.setRange(0.0001,0.1); self.dt_spin.setValue(0.005); self.dt_spin.setDecimals(5)
        self.steps_spin = QSpinBox(); self.steps_spin.setRange(1,500); self.steps_spin.setValue(80)
        self.trail_len_spin = QSpinBox(); self.trail_len_spin.setRange(1000,200000); self.trail_len_spin.setValue(50000); self.trail_len_spin.setSingleStep(5000)
        sim_form.addRow("dt:", self.dt_spin)
        sim_form.addRow("每帧步数:", self.steps_spin)
        sim_form.addRow("最大点数:", self.trail_len_spin)

        zoom_container, self.zoom_slider, self.zoom_label = self._create_slider_pair(
            5, 200, 100, "视距:", lambda v: setattr(self.camera, 'distance', float(v)))
        sim_form.addRow(zoom_container)
        grid.addWidget(sim_group, row, 1)
        row += 1

        # 变换滑块组
        trans_group = QGroupBox("模型旋转 / 平移")
        trans_grid = QGridLayout(trans_group)
        trans_grid.setSpacing(4)

        rx_c, self.rot_x_slider, _ = self._create_slider_pair(0, 360, 0, "绕X:", lambda v: setattr(self.camera, 'rot_x', float(v)))
        ry_c, self.rot_y_slider, _ = self._create_slider_pair(0, 360, 0, "绕Y:", lambda v: setattr(self.camera, 'rot_y', float(v)))
        rz_c, self.rot_z_slider, _ = self._create_slider_pair(0, 360, 0, "绕Z:", lambda v: setattr(self.camera, 'rot_z', float(v)))
        px_c, self.pan_x_slider, _ = self._create_slider_pair(-200, 200, 0, "平移X:", lambda v: self.camera.target.setX(float(v)))
        py_c, self.pan_y_slider, _ = self._create_slider_pair(-200, 200, 0, "平移Y:", lambda v: self.camera.target.setY(float(v)))
        pz_c, self.pan_z_slider, _ = self._create_slider_pair(-200, 200, 0, "平移Z:", lambda v: self.camera.target.setZ(float(v)))

        trans_grid.addWidget(rx_c, 0, 0); trans_grid.addWidget(ry_c, 0, 1)
        trans_grid.addWidget(rz_c, 1, 0); trans_grid.addWidget(px_c, 1, 1)
        trans_grid.addWidget(py_c, 2, 0); trans_grid.addWidget(pz_c, 2, 1)
        grid.addWidget(trans_group, row, 0, 1, 2)
        row += 1

        # 自定义方程
        self.custom_group = QGroupBox("自定义方程")
        cust_layout = QFormLayout(self.custom_group)
        self.dx_edit = QLineEdit("p0*(y - x)")
        self.dy_edit = QLineEdit("x*(p1 - z) - y")
        self.dz_edit = QLineEdit("x*y - p2*z")
        self.apply_custom_btn = QPushButton("编译并应用")
        self.apply_custom_btn.clicked.connect(self.compile_custom)
        cust_layout.addRow("dx/dt:", self.dx_edit)
        cust_layout.addRow("dy/dt:", self.dy_edit)
        cust_layout.addRow("dz/dt:", self.dz_edit)
        cust_layout.addRow(self.apply_custom_btn)
        self.custom_group.setVisible(False)
        grid.addWidget(self.custom_group, row, 0, 1, 2)
        row += 1

        # 公式显示
        formula_group = QGroupBox("公式")
        formula_layout = QVBoxLayout(formula_group)
        self.formula_label = QLabel(FORMULAS["Lorenz"])
        self.formula_label.setStyleSheet("font-family: monospace; font-size: 9pt;")
        formula_layout.addWidget(self.formula_label)
        grid.addWidget(formula_group, row, 0, 1, 2)
        row += 1

        # 按钮行
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始")
        self.pause_btn = QPushButton("暂停")
        self.reset_btn = QPushButton("重置")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addStretch()
        grid.addLayout(btn_layout, row, 0, 1, 2)

        self.start_btn.clicked.connect(self.start_sim)
        self.pause_btn.clicked.connect(self.pause_sim)
        self.reset_btn.clicked.connect(self.reset_sim)

        central.addWidget(panel)

        # 右侧正方形 OpenGL 画布
        self.right_container = SquareGLContainer(self.camera)
        central.addWidget(self.right_container)

        central.setSizes([280, 700])
        self.gl_widget = self.right_container.gl_widget

    def _create_slider_pair(self, min_val, max_val, init, label_text, callback):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init)
        val_label = QLabel(str(init))
        val_label.setMinimumWidth(30)
        slider.valueChanged.connect(lambda v: (callback(v), val_label.setText(str(v)), self.gl_widget.update()))
        layout.addWidget(slider)
        layout.addWidget(val_label)
        return container, slider, val_label

    # ---------- 预设与自定义逻辑 ----------
    def on_preset_changed(self, name):
        self.current_preset = name
        _, labels, defaults = PRESETS[name]
        for i, (spin, def_val) in enumerate(zip(self.param_spins, defaults)):
            label_text = labels[i] if labels[i] else f"p{i}"
            self.param_labels[i].setText(label_text)
            spin.setValue(def_val)
        self.deriv_func = PRESETS[name][0]
        self.custom_group.setVisible(name == "Custom")
        self.formula_label.setText(FORMULAS.get(name, ""))
        if name != "Custom":
            self.custom_code = [None, None, None]
        self.running = False

    def compile_custom(self):
        try:
            for i, edit in enumerate([self.dx_edit, self.dy_edit, self.dz_edit]):
                expr = make_safe_expression(edit.text())
                ns = {'x':0,'y':0,'z':0,'p0':0,'p1':0,'p2':0,'p3':0,'p4':0,'p5':0,'math':math}
                eval(compile(expr, f'<d{i}>', 'eval'), ns)
                self.custom_code[i] = compile(expr, f'<d{i}>', 'eval')
            QMessageBox.information(self, "成功", "自定义方程编译成功！")
        except Exception as e:
            QMessageBox.critical(self, "编译错误", str(e))
            self.custom_code = [None, None, None]

    def start_sim(self):
        if self.current_preset == "Custom" and any(c is None for c in self.custom_code):
            QMessageBox.warning(self, "提示", "请先编译自定义方程！")
            return
        self.running = True

    def pause_sim(self):
        self.running = False

    def reset_sim(self):
        self.running = False
        self.x, self.y, self.z = self.x0_spin.value(), self.y0_spin.value(), self.z0_spin.value()
        self.gl_widget.clear_trail()
        self.camera.reset()
        self.zoom_slider.setValue(100)
        for slider in [self.rot_x_slider, self.rot_y_slider, self.rot_z_slider,
                       self.pan_x_slider, self.pan_y_slider, self.pan_z_slider]:
            slider.setValue(0)
        self.gl_widget.update()

    def update_simulation(self):
        if not self.running:
            return
        params = [s.value() for s in self.param_spins]
        dt = self.dt_spin.value()
        n = self.steps_spin.value()
        if n <= 0:
            return

        if self.current_preset == "Custom":
            if any(c is None for c in self.custom_code):
                return
            out, self.x, self.y, self.z = integrate_custom_python(
                self.x, self.y, self.z, params, dt, n, *self.custom_code
            )
        else:
            out = np.zeros((n, 3), dtype=np.float64)
            # 调用泛型 Taichi 内核，传入当前导数函数
            generic_integrate(self.x, self.y, self.z, *params,
                              dt, n, out, 0, self.deriv_func)
            self.x, self.y, self.z = out[-1, 0], out[-1, 1], out[-1, 2]

        self.gl_widget.update_trail(out, self.trail_len_spin.value())
        self.gl_widget.update()

    def closeEvent(self, event):
        self.running = False
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())