from custom_import import *

ti.init(arch=ti.gpu, default_fp=ti.f32, kernel_profiler=False, offline_cache=True)

# ── 结构体定义 ─────────────────────────
PotentialParams = ti.types.struct(
    strength=ti.f32,
    preset=ti.i32,          # 0:双缝, 1:单缝, 2:圆形势, 3:自定义
    barrier_thickness=ti.f32,
    single_slit_width=ti.f32,
    double_slit_width=ti.f32,
    double_slit_separation=ti.f32,
    radius=ti.f32
)
WavePacketParams = ti.types.struct(
    momentum=ti.f32,
    width=ti.f32,
    center_x=ti.f32,
    center_y=ti.f32,
    direction=ti.f32
)
SimulationParams = ti.types.struct(
    time_step=ti.f32,
    steps_per_frame=ti.i32
)

potential_params = PotentialParams.field(shape=())
wave_packet_params = WavePacketParams.field(shape=())
simulation_params = SimulationParams.field(shape=())

resolution = (1080, 540)
lut_size = 720

custom_potential_field = ti.field(dtype=ti.f32, shape=resolution)

# ── 自定义势能编译器 ──────────────────────
class CustomPotentialCompiler:
    def __init__(self):
        self.globals_dict = {
            'math': math, 'np': np, 'ti': ti, 'tm': tm,
            'sin': math.sin, 'cos': math.cos, 'exp': math.exp,
            'sqrt': math.sqrt, 'abs': abs, 'pi': math.pi, 'e': math.e
        }

    def compile_potential(self, code_str, width, height):
        try:
            for key in list(self.globals_dict.keys()):
                if key not in ['math', 'np', 'ti', 'tm', 'sin', 'cos',
                               'exp', 'sqrt', 'abs', 'pi', 'e']:
                    del self.globals_dict[key]

            func_code = f"""
def potential_func(x, y, width={width}, height={height}):
    {code_str}
    return result
"""
            exec(func_code, self.globals_dict)
            potential_func = self.globals_dict['potential_func']

            x_coords = np.linspace(0, width - 1, width)
            y_coords = np.linspace(0, height - 1, height)
            X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
            potential_values = np.zeros((width, height), dtype=np.float32)
            for i in range(width):
                for j in range(height):
                    try:
                        potential_values[i, j] = float(potential_func(X[i, j], Y[i, j]))
                    except:
                        potential_values[i, j] = 0.0
            return potential_values, None
        except Exception as e:
            return None, str(e)

potential_compiler = CustomPotentialCompiler()

# ── OKLAB 矩阵 ──────────────────────────
lms_matrix_inv = ti.Matrix([
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480]
])
rgb_matrix_inv = ti.Matrix([
    [4.0767416621, -3.3077115913, 0.2309699292],
    [-1.2684380046, 2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, 1.7076147010]
])

# ── 场定义 ──────────────────────────────
psi = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
k1 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
k2 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
k3 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
k4 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
y_temp1 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
y_temp2 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
y_temp3 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
potential_temp = ti.field(dtype=ti.f32, shape=resolution)
output = ti.Vector.field(4, dtype=ti.f32, shape=resolution)
hue_lut = ti.Vector.field(3, dtype=ti.f32, shape=lut_size)

# ── 参数初始化 ──────────────────────────
@ti.kernel
def init_parameters():
    potential_params[None].strength = 1.0
    potential_params[None].preset = 0
    potential_params[None].barrier_thickness = 5.0
    potential_params[None].single_slit_width = 50.0
    potential_params[None].double_slit_width = 10.0
    potential_params[None].double_slit_separation = 50.0
    potential_params[None].radius = 50.0

    wave_packet_params[None].momentum = 250.0
    wave_packet_params[None].width = 0.5
    wave_packet_params[None].center_x = 0.3
    wave_packet_params[None].center_y = 0.5
    wave_packet_params[None].direction = 0.0

    simulation_params[None].time_step = 0.1
    simulation_params[None].steps_per_frame = 30

# ── 势能函数 (强度可为负) ────────────────
@ti.func
def potential(p):
    result = 0.0
    preset = potential_params[None].preset

    if preset == 0:                     # 双缝
        barrier_center_x = resolution[0] / 2
        barrier_thickness = potential_params[None].barrier_thickness
        if ti.abs(p[0] - barrier_center_x) < barrier_thickness / 2:
            slit_sep = potential_params[None].double_slit_separation
            slit_w = potential_params[None].double_slit_width
            c1 = resolution[1] / 2 - slit_sep / 2
            c2 = resolution[1] / 2 + slit_sep / 2
            in_slit1 = ti.abs(p[1] - c1) < slit_w / 2
            in_slit2 = ti.abs(p[1] - c2) < slit_w / 2
            if not (in_slit1 or in_slit2):
                result = 1.0

    elif preset == 1:                   # 单缝
        barrier_center_x = resolution[0] / 2
        barrier_thickness = potential_params[None].barrier_thickness
        if ti.abs(p[0] - barrier_center_x) < barrier_thickness / 2:
            slit_w = potential_params[None].single_slit_width
            if ti.abs(p[1] - resolution[1] / 2) >= slit_w / 2:
                result = 1.0

    elif preset == 2:                   # 圆形势 (强度决定正负)
        center = ti.Vector([resolution[0] / 2, resolution[1] / 2])
        rsqr = (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2
        if rsqr < potential_params[None].radius ** 2:
            result = 1.0

    elif preset == 3:                   # 自定义势能
        result = custom_potential_field[int(p[0]), int(p[1])]

    return result * potential_params[None].strength

# ── 初始波函数 ──────────────────────────
@ti.func
def psi0(p):
    x = (p[0] - resolution[0] * wave_packet_params[None].center_x) / resolution[1]
    y = (p[1] - resolution[1] * wave_packet_params[None].center_y) / resolution[1]
    sigma = wave_packet_params[None].width * 0.1
    r2 = x * x + y * y
    gaussian = 0.1 * ti.exp(-r2 / (2.0 * sigma * sigma)) / (sigma * ti.sqrt(math.pi))
    angle = wave_packet_params[None].direction * math.pi / 180.0
    kx = wave_packet_params[None].momentum * ti.cos(angle)
    ky = wave_packet_params[None].momentum * ti.sin(angle)
    plane_wave = tm.cexp(tm.vec2(0, kx * x + ky * y))
    return gaussian * plane_wave

# ── RK4 子步 ──────────────────────────
@ti.func
def compute_k(i, j, field, potential):
    laplacian = field[i+1, j] + field[i-1, j] + field[i, j+1] + field[i, j-1] - 4.0 * field[i, j]
    return tm.cdiv(-laplacian + potential[i, j] * field[i, j], tm.vec2(0, 1))

@ti.kernel
def update_psi():
    for i, j in ti.ndrange((1, resolution[0]-1), (1, resolution[1]-1)):
        p = ti.Vector([i, j])
        potential_temp[i, j] = potential(p)
        k1[i, j] = compute_k(i, j, psi, potential_temp)
        y_temp1[i, j] = psi[i, j] + 0.5 * simulation_params[None].time_step * k1[i, j]
    for i, j in ti.ndrange((1, resolution[0]-1), (1, resolution[1]-1)):
        k2[i, j] = compute_k(i, j, y_temp1, potential_temp)
        y_temp2[i, j] = psi[i, j] + 0.5 * simulation_params[None].time_step * k2[i, j]
    for i, j in ti.ndrange((1, resolution[0]-1), (1, resolution[1]-1)):
        k3[i, j] = compute_k(i, j, y_temp2, potential_temp)
        y_temp3[i, j] = psi[i, j] + simulation_params[None].time_step * k3[i, j]
    for i, j in ti.ndrange((1, resolution[0]-1), (1, resolution[1]-1)):
        k4[i, j] = compute_k(i, j, y_temp3, potential_temp)
        psi[i, j] = psi[i, j] + simulation_params[None].time_step * (
            k1[i, j] + 2.0 * k2[i, j] + 2.0 * k3[i, j] + k4[i, j]) / 6.0

# ── 色相查找表 ──────────────────────────
@ti.func
def srgb_from_linear_srgb(x):
    xlo = 12.92 * x
    xhi = 1.055 * ti.pow(x, 1.0/2.4) - 0.055
    result = ti.Vector([0.0, 0.0, 0.0])
    for k in ti.static(range(3)):
        result[k] = xlo[k] if x[k] <= 0.0031308 else xhi[k]
    return result

@ti.func
def linear_srgb_from_oklab(c):
    lms_nonlinear = lms_matrix_inv @ c
    lms_cubed = ti.Vector([lms_nonlinear[0]**3, lms_nonlinear[1]**3, lms_nonlinear[2]**3])
    return rgb_matrix_inv @ lms_cubed

@ti.kernel
def init_hue_lut():
    for i in range(lut_size):
        h = i / lut_size
        angle = h * 2 * math.pi
        oklab_color = ti.Vector([0.8, 0.4 * ti.cos(angle), 0.4 * ti.sin(angle)])
        linear_rgb = linear_srgb_from_oklab(oklab_color)
        srgb_color = srgb_from_linear_srgb(linear_rgb)
        for k in ti.static(range(3)):
            srgb_color[k] = max(0.0, min(1.0, srgb_color[k]))
        hue_lut[i] = srgb_color

@ti.kernel
def visualize():
    for i, j in output:
        v = psi[i, j]
        mag = ti.sqrt(v[0]**2 + v[1]**2)
        phase = ti.atan2(v[1], v[0])
        hue = phase / (2.0 * math.pi)
        idx = ti.cast(hue * lut_size, ti.i32) % lut_size
        rgb = hue_lut[idx]
        pot = potential_temp[i, j]
        color = 1.5 * mag * rgb + 0.25 * ti.abs(pot)
        output[i, j] = ti.Vector([color[0], color[1], color[2], 1.0])

@ti.kernel
def initialize():
    for i, j in psi:
        psi[i, j] = psi0(ti.Vector([i, j]))

@ti.kernel
def update_custom_potential_field(potential_array: ti.types.ndarray()):
    for i, j in custom_potential_field:
        custom_potential_field[i, j] = potential_array[i, j]

# ── OpenGL 渲染画布 ───────────────────────
class TaichiCanvas(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(resolution[0], resolution[1])
        self.texture_id = None
        init_hue_lut()

    def initializeGL(self):
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def update_simulation(self):
        steps = simulation_params[None].steps_per_frame
        for _ in range(steps):
            update_psi()
        visualize()
        self.update()

    def paintGL(self):
        if self.texture_id is None:
            return
        data = output.to_numpy()
        data = np.ascontiguousarray(data)
        data = np.clip(data, 0.0, 1.0) * 255
        data = data.astype(np.uint8)
        data = np.transpose(data, (1, 0, 2))
        data = np.ascontiguousarray(data)

        glClear(GL_COLOR_BUFFER_BIT)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, resolution[0], resolution[1],
                     0, GL_RGBA, GL_UNSIGNED_BYTE, data)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(-1, -1)
        glTexCoord2f(1, 1); glVertex2f( 1, -1)
        glTexCoord2f(1, 0); glVertex2f( 1,  1)
        glTexCoord2f(0, 0); glVertex2f(-1,  1)
        glEnd()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)


# ── 控制面板 ─────────────────────────────
class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.simulation_running = True
        self.custom_potential_code = (
            'result = 1 if ((x+33.75) % 135 < 67.5 and (y-33.75) % 135 > 67.5) '
            'or ((x+33.75) % 135 > 67.5 and (y-33.75) % 135 < 67.5) else -1'
        )
        self.sliders = {}
        self.labels = {}
        self._init_ui()

    def _init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(450)
        scroll.setMinimumWidth(380)
        content = QWidget()
        main = QVBoxLayout(content)

        grid = QGridLayout()

        grid.addWidget(QLabel("势能类型:"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["双缝", "单缝", "圆势", "自定义势能"])
        self.preset_combo.setCurrentIndex(0)
        grid.addWidget(self.preset_combo, 0, 1)

        slider_defs = [
            ("strength",             -100, 100, 10),
            ("single_slit_width",    0,    100, 50),
            ("double_slit_width",    0,    100, 10),
            ("double_slit_separation", 0,  200, 50),
            ("radius",               0,    200, 50),
            ("barrier_thickness",    1,    50,  5),
            ("momentum",             0,    1000,250),
            ("width",                10,   200, 50),
            ("center_x",             0,    100, 30),
            ("center_y",             0,    100, 50),
            ("direction",            0,    360, 0),
            ("time_step",            1,    30,  10),
            ("steps_per_frame",      1,    180, 30),
        ]

        row = 1
        left_col = 0
        right_col = 3
        for name, minv, maxv, defv in slider_defs:
            if name == "momentum":
                row = 1
            col_offset = right_col if name in ["momentum","width","center_x","center_y",
                                               "direction","time_step","steps_per_frame"] else left_col
            grid.addWidget(QLabel(f"{self._param_label(name)}:"), row, col_offset)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(minv, maxv)
            slider.setValue(defv)
            grid.addWidget(slider, row, col_offset+1)
            label = QLabel()
            grid.addWidget(label, row, col_offset+2)
            self.sliders[name] = slider
            self.labels[name] = label
            slider.valueChanged.connect(lambda val, n=name: self._on_param_changed(n, val))
            self._update_label(name, defv)
            row += 1

        main.addLayout(grid)
        main.addWidget(QFrame(frameShape=QFrame.HLine))   # 已去除凹陷阴影

        custom_grp = QGroupBox("自定义势能函数")
        custom_layout = QVBoxLayout(custom_grp)
        custom_layout.addWidget(QLabel("编辑自定义势能函数，必须赋值变量 'result'。"))
        self.code_editor = QTextEdit()
        self.code_editor.setPlainText(self.custom_potential_code)
        # 字体设置已删除
        self.code_editor.setMinimumHeight(100)
        custom_layout.addWidget(self.code_editor)
        btn_row = QHBoxLayout()
        self.apply_custom_btn = QPushButton("应用自定义势能")
        self.reset_custom_btn = QPushButton("重置为示例")
        btn_row.addWidget(self.apply_custom_btn)
        btn_row.addWidget(self.reset_custom_btn)
        custom_layout.addLayout(btn_row)
        main.addWidget(custom_grp)
        main.addWidget(QFrame(frameShape=QFrame.HLine))   # 已去除凹陷阴影

        btn_row2 = QHBoxLayout()
        self.start_pause_btn = QPushButton("暂停")
        self.start_pause_btn.setCheckable(True)
        self.start_pause_btn.setChecked(True)
        self.reset_btn = QPushButton("重置波函数")
        btn_row2.addWidget(self.start_pause_btn)
        btn_row2.addWidget(self.reset_btn)
        main.addLayout(btn_row2)
        main.addStretch()

        scroll.setWidget(content)
        panel_layout = QVBoxLayout(self)
        panel_layout.addWidget(scroll)
        self.setLayout(panel_layout)

        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.start_pause_btn.clicked.connect(self._toggle_simulation)
        self.reset_btn.clicked.connect(lambda: initialize())
        self.apply_custom_btn.clicked.connect(self._apply_custom_potential)
        self.reset_custom_btn.clicked.connect(lambda: self.code_editor.setPlainText(self.custom_potential_code))

    def _param_label(self, name):
        return {
            "strength": "势能强度", "single_slit_width": "单缝宽度",
            "double_slit_width": "双缝宽度", "double_slit_separation": "双缝间距",
            "radius": "圆半径", "barrier_thickness": "壁垒厚度",
            "momentum": "初始动量", "width": "波包半径",
            "center_x": "中心X", "center_y": "中心Y",
            "direction": "传播方向", "time_step": "时间步长",
            "steps_per_frame": "每帧步数"
        }.get(name, name)

    def _on_param_changed(self, name, value):
        self._update_label(name, value)
        if name == "strength":
            potential_params[None].strength = value / 10.0
        elif name == "momentum":
            wave_packet_params[None].momentum = float(value)
        elif name == "width":
            wave_packet_params[None].width = value / 100.0
        elif name == "center_x":
            wave_packet_params[None].center_x = value / 100.0
        elif name == "center_y":
            wave_packet_params[None].center_y = value / 100.0
        elif name == "direction":
            wave_packet_params[None].direction = float(value)
        elif name == "time_step":
            simulation_params[None].time_step = value / 100.0
        elif name == "steps_per_frame":
            simulation_params[None].steps_per_frame = value
        elif name == "single_slit_width":
            potential_params[None].single_slit_width = float(value)
        elif name == "double_slit_width":
            potential_params[None].double_slit_width = float(value)
        elif name == "double_slit_separation":
            potential_params[None].double_slit_separation = float(value)
        elif name == "radius":
            potential_params[None].radius = float(value)
        elif name == "barrier_thickness":
            potential_params[None].barrier_thickness = float(value)

    def _update_label(self, name, value):
        if name == "strength":
            self.labels[name].setText(f"{value/10.0:.1f}")
        elif name in ("momentum", "direction", "single_slit_width", "double_slit_width",
                      "double_slit_separation", "radius", "barrier_thickness", "steps_per_frame"):
            self.labels[name].setText(str(value))
        elif name in ("width", "center_x", "center_y", "time_step"):
            self.labels[name].setText(f"{value/100.0:.2f}")

    def _on_preset_changed(self, idx):
        potential_params[None].preset = idx
        if idx == 3:
            self._apply_custom_potential()

    def _apply_custom_potential(self):
        code = self.code_editor.toPlainText()
        potential_values, error = potential_compiler.compile_potential(code, resolution[0], resolution[1])
        if error:
            QMessageBox.warning(self, "编译错误", f"自定义势能函数编译失败:\n{error}")
            return
        update_custom_potential_field(potential_values)
        potential_params[None].preset = 3
        self.preset_combo.setCurrentIndex(3)

    def _toggle_simulation(self):
        self.simulation_running = not self.simulation_running
        self.start_pause_btn.setText("暂停" if self.simulation_running else "开始")


# ── 主窗口 ─────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("薛定谔方程模拟 – 自定义势能版")
        self.setGeometry(100, 100, 1200, 600)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        self.control_panel = ControlPanel()
        self.canvas = TaichiCanvas()

        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.canvas, 1)

        init_parameters()
        initialize()
        custom_potential_field.fill(0.0)

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_simulation)
        self.timer.start(0)

    def _update_simulation(self):
        if self.control_panel.simulation_running:
            self.canvas.update_simulation()

    def closeEvent(self, event):
        self.timer.stop()
        ti.reset()
        event.accept()

def main():
    app = QApplication(sys.argv)
    # 不再设置全局字体
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()