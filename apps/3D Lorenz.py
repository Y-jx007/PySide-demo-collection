from custom_import import *
import re

# ---------- Taichi 初始化 ----------
ti.init(arch=ti.cpu, debug=False)

# ---------- 预设积分内核（RK4）----------
@ti.kernel
def lorenz_integrate(
    x_in: ti.f64, y_in: ti.f64, z_in: ti.f64,
    p0: ti.f64, p1: ti.f64, p2: ti.f64,
    p3: ti.f64, p4: ti.f64, p5: ti.f64,
    dt: ti.f64, n: ti.i32,
    out: ti.types.ndarray(dtype=ti.f64, ndim=2),
    start: ti.i32
):
    x, y, z = x_in, y_in, z_in
    sigma, rho, beta = p0, p1, p2
    for i in range(n):
        dx1 = sigma * (y - x)
        dy1 = x * (rho - z) - y
        dz1 = x * y - beta * z
        x2 = x + 0.5 * dt * dx1
        y2 = y + 0.5 * dt * dy1
        z2 = z + 0.5 * dt * dz1
        dx2 = sigma * (y2 - x2)
        dy2 = x2 * (rho - z2) - y2
        dz2 = x2 * y2 - beta * z2
        x3 = x + 0.5 * dt * dx2
        y3 = y + 0.5 * dt * dy2
        z3 = z + 0.5 * dt * dz2
        dx3 = sigma * (y3 - x3)
        dy3 = x3 * (rho - z3) - y3
        dz3 = x3 * y3 - beta * z3
        x4 = x + dt * dx3
        y4 = y + dt * dy3
        z4 = z + dt * dz3
        dx4 = sigma * (y4 - x4)
        dy4 = x4 * (rho - z4) - y4
        dz4 = x4 * y4 - beta * z4
        x += dt / 6.0 * (dx1 + 2*dx2 + 2*dx3 + dx4)
        y += dt / 6.0 * (dy1 + 2*dy2 + 2*dy3 + dy4)
        z += dt / 6.0 * (dz1 + 2*dz2 + 2*dz3 + dz4)
        out[start + i, 0] = x
        out[start + i, 1] = y
        out[start + i, 2] = z

@ti.kernel
def rossler_integrate(
    x_in: ti.f64, y_in: ti.f64, z_in: ti.f64,
    p0: ti.f64, p1: ti.f64, p2: ti.f64,
    p3: ti.f64, p4: ti.f64, p5: ti.f64,
    dt: ti.f64, n: ti.i32,
    out: ti.types.ndarray(dtype=ti.f64, ndim=2),
    start: ti.i32
):
    x, y, z = x_in, y_in, z_in
    a, b, c = p0, p1, p2
    for i in range(n):
        dx1 = -y - z
        dy1 = x + a * y
        dz1 = b + z * (x - c)
        x2 = x + 0.5*dt*dx1; y2 = y + 0.5*dt*dy1; z2 = z + 0.5*dt*dz1
        dx2 = -y2 - z2
        dy2 = x2 + a * y2
        dz2 = b + z2 * (x2 - c)
        x3 = x + 0.5*dt*dx2; y3 = y + 0.5*dt*dy2; z3 = z + 0.5*dt*dz2
        dx3 = -y3 - z3
        dy3 = x3 + a * y3
        dz3 = b + z3 * (x3 - c)
        x4 = x + dt*dx3; y4 = y + dt*dy3; z4 = z + dt*dz3
        dx4 = -y4 - z4
        dy4 = x4 + a * y4
        dz4 = b + z4 * (x4 - c)
        x += dt/6.0 * (dx1 + 2*dx2 + 2*dx3 + dx4)
        y += dt/6.0 * (dy1 + 2*dy2 + 2*dy3 + dy4)
        z += dt/6.0 * (dz1 + 2*dz2 + 2*dz3 + dz4)
        out[start+i,0]=x; out[start+i,1]=y; out[start+i,2]=z

@ti.kernel
def chen_integrate(
    x_in: ti.f64, y_in: ti.f64, z_in: ti.f64,
    p0: ti.f64, p1: ti.f64, p2: ti.f64,
    p3: ti.f64, p4: ti.f64, p5: ti.f64,
    dt: ti.f64, n: ti.i32,
    out: ti.types.ndarray(dtype=ti.f64, ndim=2),
    start: ti.i32
):
    x, y, z = x_in, y_in, z_in
    a, b, c = p0, p1, p2
    for i in range(n):
        dx1 = a*(y - x)
        dy1 = (c - a)*x - x*z + c*y
        dz1 = x*y - b*z
        x2=x+0.5*dt*dx1; y2=y+0.5*dt*dy1; z2=z+0.5*dt*dz1
        dx2=a*(y2-x2); dy2=(c-a)*x2 - x2*z2 + c*y2; dz2=x2*y2 - b*z2
        x3=x+0.5*dt*dx2; y3=y+0.5*dt*dy2; z3=z+0.5*dt*dz2
        dx3=a*(y3-x3); dy3=(c-a)*x3 - x3*z3 + c*y3; dz3=x3*y3 - b*z3
        x4=x+dt*dx3; y4=y+dt*dy3; z4=z+dt*dz3
        dx4=a*(y4-x4); dy4=(c-a)*x4 - x4*z4 + c*y4; dz4=x4*y4 - b*z4
        x += dt/6.0*(dx1+2*dx2+2*dx3+dx4)
        y += dt/6.0*(dy1+2*dy2+2*dy3+dy4)
        z += dt/6.0*(dz1+2*dz2+2*dz3+dz4)
        out[start+i,0]=x; out[start+i,1]=y; out[start+i,2]=z

@ti.kernel
def aizawa_integrate(
    x_in: ti.f64, y_in: ti.f64, z_in: ti.f64,
    p0: ti.f64, p1: ti.f64, p2: ti.f64, p3: ti.f64, p4: ti.f64, p5: ti.f64,
    dt: ti.f64, n: ti.i32,
    out: ti.types.ndarray(dtype=ti.f64, ndim=2),
    start: ti.i32
):
    x, y, z = x_in, y_in, z_in
    a, b, c, d, e, f = p0, p1, p2, p3, p4, p5
    for i in range(n):
        dx1 = (z - b)*x - d*y
        dy1 = d*x + (z - b)*y
        dz1 = c + a*z - (z**3)/3.0 - (x**2 + y**2)*(1.0 + e*z) + f*z*x**3
        x2=x+0.5*dt*dx1; y2=y+0.5*dt*dy1; z2=z+0.5*dt*dz1
        dx2=(z2-b)*x2 - d*y2
        dy2=d*x2 + (z2-b)*y2
        dz2=c + a*z2 - (z2**3)/3.0 - (x2**2+y2**2)*(1.0+e*z2) + f*z2*x2**3
        x3=x+0.5*dt*dx2; y3=y+0.5*dt*dy2; z3=z+0.5*dt*dz2
        dx3=(z3-b)*x3 - d*y3
        dy3=d*x3 + (z3-b)*y3
        dz3=c + a*z3 - (z3**3)/3.0 - (x3**2+y3**2)*(1.0+e*z3) + f*z3*x3**3
        x4=x+dt*dx3; y4=y+dt*dy3; z4=z+dt*dz3
        dx4=(z4-b)*x4 - d*y4
        dy4=d*x4 + (z4-b)*y4
        dz4=c + a*z4 - (z4**3)/3.0 - (x4**2+y4**2)*(1.0+e*z4) + f*z4*x4**3
        x += dt/6.0*(dx1+2*dx2+2*dx3+dx4)
        y += dt/6.0*(dy1+2*dy2+2*dy3+dy4)
        z += dt/6.0*(dz1+2*dz2+2*dz3+dz4)
        out[start+i,0]=x; out[start+i,1]=y; out[start+i,2]=z

PRESETS = {
    "Lorenz": (lorenz_integrate, ["σ", "ρ", "β", "", "", ""], [10.0, 28.0, 8.0/3, 0, 0, 0]),
    "Rössler": (rossler_integrate, ["a", "b", "c", "", "", ""], [0.2, 0.2, 5.7, 0, 0, 0]),
    "Chen": (chen_integrate, ["a", "b", "c", "", "", ""], [35.0, 3.0, 28.0, 0, 0, 0]),
    "Aizawa": (aizawa_integrate, ["a", "b", "c", "d", "e", "f"],
              [0.95, 0.7, 0.6, 3.5, 0.25, 0.1]),
    "Custom": (None, ["p0", "p1", "p2", "p3", "p4", "p5"], [1.0]*6)
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.trail = np.empty((0, 3), dtype=np.float32)
        self.line_color = QColor(0, 0, 0)
        self.bg_color = QColor(230, 230, 230)
        self.camera = OrbitCamera()
        self.last_mouse = None

    def reset_view(self):
        self.camera.reset()

    def initializeGL(self):
        glClearColor(self.bg_color.redF(), self.bg_color.greenF(),
                     self.bg_color.blueF(), 1.0)

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
        glLoadMatrixf(self.camera.view_matrix().data())

        glColor3f(self.line_color.redF(), self.line_color.greenF(), self.line_color.blueF())
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.trail.tobytes())
        glDrawArrays(GL_LINE_STRIP, 0, self.trail.shape[0])
        glDisableClientState(GL_VERTEX_ARRAY)

    def update_trail(self, points: np.ndarray, max_len=50000):
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        self.trail = np.concatenate([self.trail, points.astype(np.float32)])
        if self.trail.shape[0] > max_len:
            self.trail = self.trail[-max_len:]

    def clear_trail(self):
        self.trail = np.empty((0, 3), dtype=np.float32)

    def mousePressEvent(self, event):
        self.last_mouse = event.position()

    def mouseMoveEvent(self, event):
        if self.last_mouse is None:
            return
        dx = event.position().x() - self.last_mouse.x()
        dy = event.position().y() - self.last_mouse.y()
        if event.buttons() & Qt.LeftButton:
            self.camera.rotate(dx * 0.5, dy * 0.5)
        elif event.buttons() & Qt.MiddleButton:
            self.camera.pan(dx * 0.01, -dy * 0.01)
        self.last_mouse = event.position()
        self.update()

    def wheelEvent(self, event):
        self.camera.zoom(event.angleDelta().y() * 0.05)
        self.update()

# ---------- 主窗口 ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D 吸引子 (Taichi)")
        self.resize(900, 680)

        self.running = False
        self.current_preset = "Lorenz"
        self.integrate_kernel = PRESETS["Lorenz"][0]
        self.custom_dx_code = None
        self.custom_dy_code = None
        self.custom_dz_code = None
        self.x, self.y, self.z = 0.1, 0.0, 0.0
        self.params = list(PRESETS["Lorenz"][2])
        self.dt = 0.005
        self.steps_per_frame = 50

        central = QSplitter(Qt.Horizontal)
        self.setCentralWidget(central)

        # ---- 左侧面板 ----
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(6, 6, 6, 6)
        panel_layout.setSpacing(4)

        # 预设选择
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(4)
        preset_layout.addWidget(QLabel("预设:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS.keys())
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        panel_layout.addLayout(preset_layout)

        # 参数组
        param_group = QGroupBox("参数")
        param_group.setContentsMargins(4, 12, 4, 4)
        self.param_form = QFormLayout()
        self.param_form.setSpacing(3)
        self.param_form.setContentsMargins(2, 2, 2, 2)
        self.param_spins = []
        self.param_labels = []
        for i in range(6):
            spin = QDoubleSpinBox()
            spin.setRange(-100, 100)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            spin.setMaximumWidth(100)
            self.param_spins.append(spin)
            label = QLabel("")
            self.param_labels.append(label)
            self.param_form.addRow(label, spin)
        param_group.setLayout(self.param_form)
        panel_layout.addWidget(param_group)

        # 初始条件
        init_group = QGroupBox("初始条件")
        init_group.setContentsMargins(4, 12, 4, 4)
        init_layout = QFormLayout()
        init_layout.setSpacing(3)
        init_layout.setContentsMargins(2, 2, 2, 2)
        self.x0_spin = QDoubleSpinBox(); self.x0_spin.setRange(-100,100); self.x0_spin.setValue(0.1); self.x0_spin.setMaximumWidth(100)
        self.y0_spin = QDoubleSpinBox(); self.y0_spin.setRange(-100,100); self.y0_spin.setValue(0.0); self.y0_spin.setMaximumWidth(100)
        self.z0_spin = QDoubleSpinBox(); self.z0_spin.setRange(-100,100); self.z0_spin.setValue(0.0); self.z0_spin.setMaximumWidth(100)
        init_layout.addRow("x₀:", self.x0_spin)
        init_layout.addRow("y₀:", self.y0_spin)
        init_layout.addRow("z₀:", self.z0_spin)
        init_group.setLayout(init_layout)
        panel_layout.addWidget(init_group)

        # 模拟设置
        sim_group = QGroupBox("模拟设置")
        sim_group.setContentsMargins(4, 12, 4, 4)
        sim_layout = QFormLayout()
        sim_layout.setSpacing(3)
        sim_layout.setContentsMargins(2, 2, 2, 2)
        self.dt_spin = QDoubleSpinBox(); self.dt_spin.setRange(0.0001,0.1); self.dt_spin.setValue(0.005); self.dt_spin.setDecimals(5); self.dt_spin.setMaximumWidth(100)
        self.steps_spin = QDoubleSpinBox(); self.steps_spin.setRange(1,500); self.steps_spin.setDecimals(0); self.steps_spin.setValue(self.steps_per_frame); self.steps_spin.setMaximumWidth(100)
        sim_layout.addRow("dt:", self.dt_spin)
        sim_layout.addRow("每帧步数:", self.steps_spin)

        zoom_layout = QHBoxLayout()
        zoom_layout.setSpacing(4)
        zoom_label = QLabel("视距:")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(5, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMaximumWidth(100)
        self.zoom_value_label = QLabel("100")
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_value_label)
        sim_layout.addRow(zoom_layout)
        sim_group.setLayout(sim_layout)
        panel_layout.addWidget(sim_group)

        # 自定义方程
        self.custom_group = QGroupBox("自定义方程")
        self.custom_group.setContentsMargins(4, 12, 4, 4)
        custom_layout = QFormLayout()
        custom_layout.setSpacing(3)
        custom_layout.setContentsMargins(2, 2, 2, 2)
        self.dx_edit = QLineEdit("p0*(y - x)"); self.dx_edit.setMaximumWidth(120)
        self.dy_edit = QLineEdit("x*(p1 - z) - y"); self.dy_edit.setMaximumWidth(120)
        self.dz_edit = QLineEdit("x*y - p2*z"); self.dz_edit.setMaximumWidth(120)
        custom_layout.addRow("dx/dt:", self.dx_edit)
        custom_layout.addRow("dy/dt:", self.dy_edit)
        custom_layout.addRow("dz/dt:", self.dz_edit)
        self.apply_custom_btn = QPushButton("编译并应用"); self.apply_custom_btn.setMaximumWidth(100)
        custom_layout.addRow(self.apply_custom_btn)
        self.custom_group.setLayout(custom_layout)
        self.custom_group.setVisible(False)
        panel_layout.addWidget(self.custom_group)

        # 公式显示
        formula_group = QGroupBox("公式")
        formula_group.setContentsMargins(4, 12, 4, 4)
        self.formula_label = QLabel(FORMULAS["Lorenz"])
        self.formula_label.setStyleSheet("font-family: monospace; font-size: 9pt;")
        formula_layout = QVBoxLayout()
        formula_layout.addWidget(self.formula_label)
        formula_group.setLayout(formula_layout)
        panel_layout.addWidget(formula_group)

        # 按钮行
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        self.start_btn = QPushButton("开始"); self.start_btn.setMaximumWidth(60)
        self.pause_btn = QPushButton("暂停"); self.pause_btn.setMaximumWidth(60)
        self.reset_btn = QPushButton("重置"); self.reset_btn.setMaximumWidth(60)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.reset_btn)
        panel_layout.addLayout(btn_layout)
        panel_layout.addStretch()

        central.addWidget(panel)

        # OpenGL 窗口
        self.gl_widget = AttractorGLWidget()
        central.addWidget(self.gl_widget)
        central.setSizes([250, 650])

        # 信号连接
        self.start_btn.clicked.connect(self.start_sim)
        self.pause_btn.clicked.connect(self.pause_sim)
        self.reset_btn.clicked.connect(self.reset_sim)
        self.apply_custom_btn.clicked.connect(self.compile_custom)

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.setInterval(16)

        self.on_preset_changed("Lorenz")
        self.reset_sim()

    # ---------- 控制逻辑 ----------
    def on_preset_changed(self, name):
        self.current_preset = name
        _, labels, defaults = PRESETS[name]
        for i, spin in enumerate(self.param_spins):
            label_text = labels[i] if labels[i] else f"p{i}"
            self.param_labels[i].setText(label_text)
            spin.setValue(defaults[i])
        if name == "Custom":
            self.custom_group.setVisible(True)
            self.integrate_kernel = None
        else:
            self.custom_group.setVisible(False)
            self.integrate_kernel = PRESETS[name][0]
        self.formula_label.setText(FORMULAS.get(name, ""))
        self.running = False

    def on_zoom_changed(self, value):
        self.gl_widget.camera.distance = float(value)
        self.zoom_value_label.setText(str(value))
        self.gl_widget.update()

    def compile_custom(self):
        try:
            import math
            dx_expr = make_safe_expression(self.dx_edit.text())
            dy_expr = make_safe_expression(self.dy_edit.text())
            dz_expr = make_safe_expression(self.dz_edit.text())

            test_ns = {'x': 0.0, 'y': 0.0, 'z': 0.0,
                       'p0': 0.0, 'p1': 0.0, 'p2': 0.0,
                       'p3': 0.0, 'p4': 0.0, 'p5': 0.0,
                       'math': math}
            eval(compile(dx_expr, '<dx>', 'eval'), test_ns)
            eval(compile(dy_expr, '<dy>', 'eval'), test_ns)
            eval(compile(dz_expr, '<dz>', 'eval'), test_ns)

            self.custom_dx_code = compile(dx_expr, '<dx>', 'eval')
            self.custom_dy_code = compile(dy_expr, '<dy>', 'eval')
            self.custom_dz_code = compile(dz_expr, '<dz>', 'eval')

            QMessageBox.information(self, "成功", "自定义方程编译成功！")
        except Exception as e:
            QMessageBox.critical(self, "编译错误", f"自定义方程编译失败:\n{str(e)}")
            self.custom_dx_code = self.custom_dy_code = self.custom_dz_code = None

    def start_sim(self):
        if self.current_preset == "Custom" and (self.custom_dx_code is None):
            QMessageBox.warning(self, "提示", "请先编译自定义方程！")
            return
        self.running = True

    def pause_sim(self):
        self.running = False

    def reset_sim(self):
        self.running = False
        self.x = self.x0_spin.value()
        self.y = self.y0_spin.value()
        self.z = self.z0_spin.value()
        self.gl_widget.clear_trail()
        self.gl_widget.reset_view()
        self.zoom_slider.setValue(100)
        self.zoom_value_label.setText("100")
        self.gl_widget.update()

    def update_simulation(self):
        if not self.running:
            return
        self.params = [spin.value() for spin in self.param_spins]
        self.dt = self.dt_spin.value()
        n = int(self.steps_spin.value())
        if n <= 0:
            return

        if self.current_preset == "Custom":
            if self.custom_dx_code is None:
                return
            out, new_x, new_y, new_z = integrate_custom_python(
                self.x, self.y, self.z, self.params, self.dt, n,
                self.custom_dx_code, self.custom_dy_code, self.custom_dz_code
            )
            self.x, self.y, self.z = new_x, new_y, new_z
        else:
            out = np.zeros((n, 3), dtype=np.float64)
            kernel = self.integrate_kernel
            if kernel is None:
                return
            kernel(self.x, self.y, self.z, *self.params, self.dt, n, out, 0)
            self.x, self.y, self.z = out[-1, 0], out[-1, 1], out[-1, 2]

        self.gl_widget.update_trail(out)
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
    window.timer.start()
    sys.exit(app.exec())