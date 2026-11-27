import math
import time

from common.qt_base import (LabeledSlider, OrbitCamera, QColor, QComboBox, QDoubleSpinBox, QFormLayout,
                            QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                            QMatrix4x4, QMessageBox, QOpenGLWidget, QPushButton, QSpinBox, QSplitter,
                            QTimer, QVBoxLayout, QWidget, Qt, SquareContainer, math, run_app,
                            setup_gl_format)
from common.numpy_base import (np)
from common.taichi_base import (ti)
from common.opengl_base import (GL_COLOR_BUFFER_BIT, GL_DEPTH_TEST, GL_FLOAT, GL_LINE_STRIP, GL_MODELVIEW, GL_PROJECTION, GL_VERTEX_ARRAY, glClear, glClearColor, glColor3f, glDisable, glDisableClientState, glDrawArrays, glEnableClientState, glLineWidth, glLoadIdentity, glLoadMatrixf, glMatrixMode, glVertexPointer, glViewport)
from common.custom_base import (integrate_custom_python, make_safe_expression)
from common.lab_window import LabWindow
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import QDialog, QDialogButtonBox

# ---------- Taichi 初始化 ----------
ti.init(arch=ti.cpu, debug=False, offline_cache=True, advanced_optimization=True)

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
        self.line_color = QColor(30, 70, 84)
        self.bg_color = QColor(255, 255, 255)
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

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.trail.tobytes())
        n = self.trail.shape[0]
        # Older plume in a softer ink, recent trail darker.  Two strips keep
        # the depth cue without a per-vertex colour buffer.
        split = max(0, n - max(1500, int(n * 0.18)))
        glLineWidth(1.6)
        glColor3f(0.47, 0.61, 0.66)
        if split > 1:
            glDrawArrays(GL_LINE_STRIP, 0, split)
        glColor3f(0.07, 0.24, 0.30)
        start = max(0, split - 1)
        if n - start >= 2:
            glDrawArrays(GL_LINE_STRIP, start, n - start)
        glDisableClientState(GL_VERTEX_ARRAY)

    def bounds(self):
        """Axis-aligned bounds of the current trail, or None."""
        if self.trail.shape[0] < 2:
            return None
        return (self.trail.min(axis=0).astype(float),
                self.trail.max(axis=0).astype(float))

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


# ---------- 主窗口 ----------
class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lorenz 吸引子")
        self.resize(1000, 720)
        self.setMinimumSize(1000, 720)

        self.running = False
        self.current_preset = "Lorenz"
        self.deriv_func = PRESETS["Lorenz"][0]
        self.custom_code = [None, None, None]
        self.x, self.y, self.z = 0.1, 0.0, 0.0
        self.params = list(PRESETS["Lorenz"][2])
        self.dt = 0.005
        self.steps_per_frame = 80
        self._pending_fit = True
        self._custom_dialog = None
        self._fps_t0 = time.perf_counter()
        self._fps_n = 0

        self.camera = OrbitCamera()
        self._init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(16)

        self.on_preset_changed("Lorenz")
        self.start_sim()

    # ---------- UI ----------
    def _init_ui(self):
        central = QWidget()
        central.setStyleSheet("background-color: #ffffff;")
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(330)
        panel.setStyleSheet("background-color: #ffffff;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        preset_group = QGroupBox("吸引子")
        preset_form = QFormLayout(preset_group)
        preset_form.setContentsMargins(6, 6, 6, 6)
        preset_form.setSpacing(4)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS.keys())
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_form.addRow("系统", self.preset_combo)
        self.custom_btn = QPushButton("编辑自定义方程…")
        self.custom_btn.clicked.connect(self.edit_custom)
        self.custom_btn.setVisible(False)
        preset_form.addRow(self.custom_btn)
        layout.addWidget(preset_group)

        param_group = QGroupBox("参数")
        self.param_grid = QGridLayout(param_group)
        self.param_grid.setContentsMargins(6, 6, 6, 6)
        self.param_grid.setSpacing(4)
        self.param_grid.setColumnStretch(0, 1)
        self.param_grid.setColumnStretch(1, 1)
        self.param_spins = []
        self.param_labels = []
        self.param_boxes = []
        for i in range(6):
            box = QWidget()
            row = QHBoxLayout(box)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(3)
            label = QLabel("")
            label.setFixedWidth(24)
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            spin.setFixedWidth(96)
            spin.setKeyboardTracking(False)
            row.addWidget(label)
            row.addWidget(spin)
            row.addStretch(1)
            self.param_labels.append(label)
            self.param_spins.append(spin)
            self.param_boxes.append(box)
            self.param_grid.addWidget(box, i // 2, i % 2)
        layout.addWidget(param_group)

        init_group = QGroupBox("初始条件")
        init_grid = QGridLayout(init_group)
        init_grid.setContentsMargins(6, 6, 6, 6)
        init_grid.setSpacing(3)
        self.x0_spin = QDoubleSpinBox()
        self.y0_spin = QDoubleSpinBox()
        self.z0_spin = QDoubleSpinBox()
        for col, (name, spin, value) in enumerate((
            ("x₀", self.x0_spin, 0.1),
            ("y₀", self.y0_spin, 0.0),
            ("z₀", self.z0_spin, 0.0),
        )):
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setValue(value)
            spin.setFixedWidth(88)
            init_grid.addWidget(QLabel(name), 0, col)
            init_grid.addWidget(spin, 1, col)
        layout.addWidget(init_group)

        sim_group = QGroupBox("积分")
        sim_form = QFormLayout(sim_group)
        sim_form.setContentsMargins(6, 6, 6, 6)
        sim_form.setSpacing(3)
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.0001, 0.1)
        self.dt_spin.setDecimals(5)
        self.dt_spin.setValue(0.005)
        self.dt_spin.setFixedWidth(102)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 500)
        self.steps_spin.setValue(80)
        self.steps_spin.setFixedWidth(102)
        self.trail_len_spin = QSpinBox()
        self.trail_len_spin.setRange(1000, 200000)
        self.trail_len_spin.setSingleStep(5000)
        self.trail_len_spin.setValue(50000)
        self.trail_len_spin.setFixedWidth(102)
        sim_form.addRow("步长 dt", self.dt_spin)
        sim_form.addRow("每帧步数", self.steps_spin)
        sim_form.addRow("轨迹点数", self.trail_len_spin)
        layout.addWidget(sim_group)

        view_group = QGroupBox("视图")
        view_grid = QGridLayout(view_group)
        view_grid.setContentsMargins(6, 6, 6, 6)
        view_grid.setSpacing(4)
        self.zoom_slider = self._camera_slider(
            "视距", 5, 200, 100, lambda v: setattr(self.camera, "distance", float(v)))
        self.rot_x_slider = self._camera_slider(
            "绕X", 0, 360, 0, lambda v: setattr(self.camera, "rot_x", float(v)))
        self.rot_y_slider = self._camera_slider(
            "绕Y", 0, 360, 0, lambda v: setattr(self.camera, "rot_y", float(v)))
        self.rot_z_slider = self._camera_slider(
            "绕Z", 0, 360, 0, lambda v: setattr(self.camera, "rot_z", float(v)))
        self.fit_btn = QPushButton("适配视图")
        self.fit_btn.setToolTip("按当前轨迹包围盒重新居中并缩放")
        self.fit_btn.clicked.connect(self._fit_view)
        view_grid.addWidget(self.zoom_slider, 0, 0, 1, 2)
        view_grid.addWidget(self.rot_x_slider, 1, 0)
        view_grid.addWidget(self.rot_y_slider, 1, 1)
        view_grid.addWidget(self.rot_z_slider, 2, 0)
        view_grid.addWidget(self.fit_btn, 2, 1)
        layout.addWidget(view_group)

        formula_group = QGroupBox("方程")
        formula_layout = QVBoxLayout(formula_group)
        formula_layout.setContentsMargins(6, 5, 6, 5)
        formula_layout.setSpacing(2)
        self.formula_label = QLabel(FORMULAS["Lorenz"])
        self.formula_label.setWordWrap(True)
        self.formula_label.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace;"
            "font-size: 9pt; color: #56616c;"
        )
        formula_layout.addWidget(self.formula_label)
        layout.addWidget(formula_group)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.start_btn = QPushButton("开始")
        self.reset_btn = QPushButton("重置")
        self.start_btn.clicked.connect(self.toggle_sim)
        self.reset_btn.clicked.connect(self.reset_sim)
        controls.addWidget(self.start_btn, 1)
        controls.addWidget(self.reset_btn, 1)
        layout.addLayout(controls)
        layout.addStretch(1)

        root.addWidget(panel)

        self.gl_widget = AttractorGLWidget(self.camera)
        self.gl_widget.setFixedSize(640, 640)
        self.canvas = SquareContainer(
            self.gl_widget, background="#ffffff", border="0")
        self.canvas.setFixedSize(640, 640)
        root.addWidget(self.canvas, 1)

        self.setCentralWidget(central)

        self.lbl_state = QLabel("已暂停")
        self.lbl_points = QLabel("轨迹 0")
        self.lbl_fps = QLabel("– fps")
        for label in (self.lbl_points, self.lbl_fps):
            label.setStyleSheet("color: #66717a; padding-left: 10px;")
        self.statusBar().addPermanentWidget(self.lbl_state)
        self.statusBar().addPermanentWidget(self.lbl_points)
        self.statusBar().addPermanentWidget(self.lbl_fps)
        self.statusBar().showMessage("左键拖动旋转 · 中键平移 · 滚轮缩放")

    def _camera_slider(self, label_text, lo, hi, init, setter):
        widget = LabeledSlider(label_text, lo, hi, init)
        widget.valueChanged.connect(lambda v: (setter(float(v)), self._refresh_view()))
        return widget

    def _refresh_view(self):
        widget = getattr(self, "gl_widget", None)
        if widget is not None:
            widget.update()

    def _sync_camera_sliders(self):
        pairs = (
            (self.zoom_slider, self.camera.distance),
            (self.rot_x_slider, self.camera.rot_x),
            (self.rot_y_slider, self.camera.rot_y),
            (self.rot_z_slider, self.camera.rot_z),
        )
        for slider, value in pairs:
            slider.blockSignals(True)
            slider.setValue(int(round(value)))
            slider.blockSignals(False)

    def _fit_view(self):
        bounds = self.gl_widget.bounds()
        if bounds is None:
            return
        lo, hi = bounds
        center = (lo + hi) * 0.5
        extent = float(np.max(hi - lo))
        if extent <= 1e-6:
            return
        self.camera.target = QVector3D(float(center[0]), float(center[1]), float(center[2]))
        self.camera.distance = float(np.clip(extent * 1.65, 5.0, 200.0))
        self._sync_camera_sliders()
        self.gl_widget.update()

    # ---------- 运行状态 ----------
    def _set_running_ui(self):
        self.start_btn.setText("暂停" if self.running else "开始")
        self.lbl_state.setText("运行中" if self.running else "已暂停")

    def toggle_sim(self):
        if self.running:
            self.pause_sim()
        else:
            self.start_sim()

    def start_sim(self):
        if self.current_preset == "Custom" and any(c is None for c in self.custom_code):
            QMessageBox.warning(self, "提示", "请先编译自定义方程！")
            return
        self.running = True
        self._fps_t0 = time.perf_counter()
        self._fps_n = 0
        self._set_running_ui()

    def pause_sim(self):
        self.running = False
        self.lbl_fps.setText("– fps")
        self._set_running_ui()

    def reset_sim(self):
        self.running = False
        self.x, self.y, self.z = self.x0_spin.value(), self.y0_spin.value(), self.z0_spin.value()
        self.gl_widget.clear_trail()
        self.camera.reset()
        self.camera.rot_x = self.camera.rot_y = self.camera.rot_z = 0.0
        self._sync_camera_sliders()
        self._pending_fit = True
        self.lbl_points.setText("轨迹 0")
        self.lbl_fps.setText("– fps")
        self._set_running_ui()
        self.gl_widget.update()

    # ---------- 预设 / 自定义 ----------
    def on_preset_changed(self, name):
        was_running = self.running
        self.current_preset = name
        _, labels, defaults = PRESETS[name]
        active = sum(1 for label in labels if label)
        for i, (spin, value) in enumerate(zip(self.param_spins, defaults)):
            visible = i < active
            self.param_boxes[i].setVisible(visible)
            self.param_labels[i].setText(labels[i] if labels[i] else f"p{i}")
            spin.setValue(value)
        self.deriv_func = PRESETS[name][0]
        self.custom_btn.setVisible(name == "Custom")
        if name != "Custom":
            self.custom_code = [None, None, None]
        self.formula_label.setText(FORMULAS.get(name, ""))
        self.reset_sim()
        if was_running and not (
            name == "Custom" and any(c is None for c in self.custom_code)
        ):
            self.start_sim()

    def _build_custom_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("自定义方程")
        form = QFormLayout(dialog)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(6)
        self.dx_edit = QLineEdit("p0*(y - x)")
        self.dy_edit = QLineEdit("x*(p1 - z) - y")
        self.dz_edit = QLineEdit("x*y - p2*z")
        form.addRow("dx/dt", self.dx_edit)
        form.addRow("dy/dt", self.dy_edit)
        form.addRow("dz/dt", self.dz_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.compile_custom)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        self.custom_dialog = dialog

    def edit_custom(self):
        if self.custom_dialog is None:
            self._build_custom_dialog()
        self.custom_dialog.exec()

    def compile_custom(self):
        try:
            for i, edit in enumerate([self.dx_edit, self.dy_edit, self.dz_edit]):
                expr = make_safe_expression(edit.text())
                ns = {'x': 0, 'y': 0, 'z': 0, 'p0': 0, 'p1': 0, 'p2': 0, 'p3': 0, 'p4': 0, 'p5': 0, 'math': math}
                eval(compile(expr, f'<d{i}>', 'eval'), ns)
                self.custom_code[i] = compile(expr, f'<d{i}>', 'eval')
        except Exception as exc:
            QMessageBox.critical(self, "编译错误", str(exc))
            self.custom_code = [None, None, None]
        else:
            self.lbl_state.setText("自定义方程已编译")
            if self.custom_dialog is not None:
                self.custom_dialog.accept()
            if self.current_preset == "Custom":
                self.start_sim()

    # ---------- 循环 ----------
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
            generic_integrate(self.x, self.y, self.z, *params,
                              dt, n, out, 0, self.deriv_func)
            self.x, self.y, self.z = out[-1, 0], out[-1, 1], out[-1, 2]

        self.gl_widget.update_trail(out, self.trail_len_spin.value())
        if self._pending_fit and self.gl_widget.trail.shape[0] >= 120:
            self._fit_view()
            self._pending_fit = False
        self.gl_widget.update()

        self.lbl_points.setText(f"轨迹 {self.gl_widget.trail.shape[0]:,}")
        self._fps_n += 1
        now = time.perf_counter()
        elapsed = now - self._fps_t0
        if elapsed >= 0.5:
            self.lbl_fps.setText(f"{self._fps_n / elapsed:.0f} fps")
            self._fps_t0 = now
            self._fps_n = 0

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_sim()
            return
        if event.key() == Qt.Key_R:
            self.reset_sim()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.running = False
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    setup_gl_format()
    raise SystemExit(run_app(MainWindow))
