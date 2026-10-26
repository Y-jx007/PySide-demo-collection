from common.qt_base import (QApplication, QComboBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QOpenGLWidget, QPushButton, QScrollArea, QSizePolicy, QSlider, QTextEdit, QTimer, QVBoxLayout, QWidget, Qt, math, sys, time)
from common.numpy_base import (np)
from common.taichi_base import (ti, tm)
from common.opengl_base import (GL_CLAMP_TO_EDGE, GL_COLOR_BUFFER_BIT, GL_LINEAR, GL_NEAREST, GL_QUADS, GL_RGBA, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_UNSIGNED_BYTE, glBegin, glBindTexture, glClear, glEnable, glEnd, glGenTextures, glTexCoord2f, glTexImage2D, glTexSubImage2D, glTexParameteri, glVertex2f, glViewport)
from PySide6.QtWidgets import QDialog

ti.init(arch=ti.gpu, default_fp=ti.f32, kernel_profiler=False, offline_cache=False)

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
            ns = dict(self.globals_dict)
            body = "\n".join("    " + line if line.strip() else "" for line in code_str.splitlines())
            func_code = f"""
def potential_func(x, y, width={width}, height={height}):
{body}
    return result
"""
            exec(func_code, ns)
            potential_func = ns['potential_func']

            x_coords = np.linspace(0, width - 1, width)
            y_coords = np.linspace(0, height - 1, height)
            X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

            # 快路径：整段表达式能直接吃 numpy 数组（用 np.where 之类的写法）
            try:
                values = np.asarray(potential_func(X, Y), dtype=np.float32)
                if values.shape == (width, height) and np.all(np.isfinite(values)):
                    return np.ascontiguousarray(values), None
            except Exception:
                pass

            # 慢路径：逐点求值（用户写的是普通 Python 标量运算时走这里）
            potential_values = np.zeros((width, height), dtype=np.float32)
            for i in range(width):
                for j in range(height):
                    try:
                        potential_values[i, j] = float(potential_func(X[i, j], Y[i, j]))
                    except Exception:
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
texture_output = ti.Vector.field(4, dtype=ti.u8, shape=(resolution[1], resolution[0]))
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

    simulation_params[None].time_step = 0.2
    simulation_params[None].steps_per_frame = 12

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
            half = slit_w / 2
            # 半开区间 [lo, hi)：两条缝落在相同数量的像素上，缝宽 0 时完全不透
            in_slit1 = (p[1] >= c1 - half) and (p[1] < c1 + half)
            in_slit2 = (p[1] >= c2 - half) and (p[1] < c2 + half)
            if not (in_slit1 or in_slit2):
                result = 1.0

    elif preset == 1:                   # 单缝
        barrier_center_x = resolution[0] / 2
        barrier_thickness = potential_params[None].barrier_thickness
        if ti.abs(p[0] - barrier_center_x) < barrier_thickness / 2:
            slit_w = potential_params[None].single_slit_width
            half = slit_w / 2
            cy = resolution[1] / 2
            in_slit = (p[1] >= cy - half) and (p[1] < cy + half)
            if not in_slit:
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
    # Write the final RGBA texture directly in image row order.  This avoids
    # a 9 MB float32 copy plus clip/transpose on the CPU every frame.
    for i, j in psi:
        v = psi[i, j]
        mag = ti.sqrt(v[0]**2 + v[1]**2)
        phase = ti.atan2(v[1], v[0])
        hue = phase / (2.0 * math.pi)
        idx = ti.cast(hue * lut_size, ti.i32) % lut_size
        rgb = hue_lut[idx]
        pot = potential_temp[i, j]
        color = 1.5 * mag * rgb + 0.25 * ti.abs(pot)
        texture_output[j, i] = ti.Vector([
            ti.cast(ti.min(1.0, ti.max(0.0, color[0])) * 255.0, ti.u8),
            ti.cast(ti.min(1.0, ti.max(0.0, color[1])) * 255.0, ti.u8),
            ti.cast(ti.min(1.0, ti.max(0.0, color[2])) * 255.0, ti.u8),
            ti.u8(255),
        ])

@ti.kernel
def initialize():
    for i, j in psi:
        psi[i, j] = psi0(ti.Vector([i, j]))

@ti.kernel
def update_custom_potential_field(potential_array: ti.types.ndarray()):
    for i, j in custom_potential_field:
        custom_potential_field[i, j] = potential_array[i, j]

# ── OpenGL 渲染画布 ───────────────────────
def cover_texcoords(w, h, dpr, target):
    """把纹理按「填满视口」的方式裁切，返回 (u0, v0, u1, v1)。

    视口比目标比例更宽就裁上下，更窄就裁左右——不拉伸、也不留黑边。
    w/h 是逻辑像素，计算前换算成设备像素。
    """
    W = max(1, int(round(w * dpr)))
    H = max(1, int(round(h * dpr)))
    if W / H > target:
        th = target / (W / H)      # 高度方向保留的比例
        tw = 1.0
    else:
        tw = (W / H) / target      # 宽度方向保留的比例
        th = 1.0
    u0 = (1.0 - tw) / 2.0
    v0 = (1.0 - th) / 2.0
    return u0, v0, 1.0 - u0, 1.0 - v0


class TaichiCanvas(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.texture_id = None
        self._texture_allocated = False
        self._texture_dirty = True
        init_hue_lut()

    def initializeGL(self):
        self._texture_allocated = False
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def update_simulation(self):
        steps = simulation_params[None].steps_per_frame
        for _ in range(steps):
            update_psi()
        visualize()
        self._texture_dirty = True
        self.update()

    def paintGL(self):
        if self.texture_id is None:
            return
        if self._texture_dirty:
            data = np.ascontiguousarray(texture_output.to_numpy())
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            if not self._texture_allocated:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, resolution[0], resolution[1],
                             0, GL_RGBA, GL_UNSIGNED_BYTE, data)
                self._texture_allocated = True
            else:
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, resolution[0], resolution[1],
                                GL_RGBA, GL_UNSIGNED_BYTE, data)
            self._texture_dirty = False

        # 视口铺满整个画布（设备像素），由 UV 裁剪保证不拉伸
        dpr = self.devicePixelRatioF()
        glViewport(0, 0, max(1, int(round(self.width() * dpr))),
                   max(1, int(round(self.height() * dpr))))
        u0, v0, u1, v1 = cover_texcoords(self.width(), self.height(), dpr,
                                         resolution[0] / resolution[1])

        glClear(GL_COLOR_BUFFER_BIT)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        glBegin(GL_QUADS)
        glTexCoord2f(u0, v1); glVertex2f(-1, -1)
        glTexCoord2f(u1, v1); glVertex2f( 1, -1)
        glTexCoord2f(u1, v0); glVertex2f( 1,  1)
        glTexCoord2f(u0, v0); glVertex2f(-1,  1)
        glEnd()

    def resizeGL(self, w, h):
        dpr = self.devicePixelRatioF()
        glViewport(0, 0, max(1, int(w * dpr)), max(1, int(h * dpr)))


# ── 控制面板 ─────────────────────────────
class ControlPanel(QWidget):
    """左侧参数面板：按「势能 / 波包 / 求解器」分组，只显示当前类型相关的参数。"""

    # key: (标签, 最小, 最大, 默认, 格式化)
    SLIDERS = {
        'strength':               ('势能强度', -100, 100, 10, lambda v: f"{v / 10:.1f}"),
        'barrier_thickness':      ('壁垒厚度', 1, 50, 5, str),
        'double_slit_width':      ('缝宽', 0, 80, 10, str),
        'double_slit_separation': ('缝间距', 4, 400, 50, str),
        'single_slit_width':      ('缝宽', 0, 400, 50, str),
        'radius':                 ('圆半径', 5, 500, 50, str),
        'momentum':               ('初始动量', 0, 1000, 250, str),
        'width':                  ('波包半径', 10, 200, 50, lambda v: f"{v / 100:.2f}"),
        'center_x':               ('中心 X', 0, 100, 30, lambda v: f"{v / 100:.2f}"),
        'center_y':               ('中心 Y', 0, 100, 50, lambda v: f"{v / 100:.2f}"),
        'direction':              ('传播方向', 0, 360, 0, lambda v: f"{v}°"),
        'time_step':              ('时间步长', 5, 25, 20, lambda v: f"{v / 100:.2f}"),
        'steps_per_frame':        ('每帧步数', 1, 200, 12, str),
    }

    DEFAULT_CODE = (
        "a = ((x + 33.75) % 135 < 67.5) & ((y - 33.75) % 135 > 67.5)\n"
        "b = ((x + 33.75) % 135 > 67.5) & ((y - 33.75) % 135 < 67.5)\n"
        "result = np.where(a | b, 1, -1)"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.simulation_running = True
        self.sliders = {}
        self.labels = {}
        self._custom_code = self.DEFAULT_CODE
        self._custom_ready = False
        self._init_ui()

    # ---------- UI ----------
    def _init_ui(self):
        self.setFixedWidth(330)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        main = QVBoxLayout(content)
        main.setSpacing(4)
        main.setContentsMargins(2, 0, 2, 0)

        main.addWidget(self._build_control_group())
        main.addWidget(self._build_potential_group())
        main.addWidget(self._build_packet_group())
        main.addWidget(self._build_solver_group())
        main.addStretch()

        scroll.setWidget(content)
        panel_layout = QVBoxLayout(self)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)

        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.start_pause_btn.clicked.connect(self._toggle_simulation)
        self.reset_btn.clicked.connect(self._on_reset)
        self.edit_potential_btn.clicked.connect(self._open_potential_editor)

        # 先把默认的自定义势能编译好，切到「自定义」时就能直接看
        self._compile_and_upload(self._custom_code)
        self._on_preset_changed(0)

    def _build_control_group(self):
        group = QGroupBox("控制")
        layout = QHBoxLayout()
        layout.setSpacing(6)
        self.start_pause_btn = QPushButton("暂停")
        self.start_pause_btn.setCheckable(True)
        self.start_pause_btn.setChecked(True)
        self.reset_btn = QPushButton("重置波函数")
        layout.addWidget(self.start_pause_btn)
        layout.addWidget(self.reset_btn)
        group.setLayout(layout)
        return group

    def _build_potential_group(self):
        group = QGroupBox("势能")
        layout = QVBoxLayout()
        layout.setSpacing(6)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("类型:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["双缝", "单缝", "圆势", "自定义"])
        h.addWidget(self.preset_combo, 1)
        layout.addWidget(row)

        form = QFormLayout()
        form.setSpacing(3)
        self._add_slider(form, 'strength')
        layout.addLayout(form)

        # 只与某种势能相关的参数，按类型显示/隐藏
        self.shape_groups = {}
        self.shape_groups[0] = self._make_shape_group(
            "双缝参数", ['barrier_thickness', 'double_slit_width', 'double_slit_separation'])
        self.shape_groups[1] = self._make_shape_group(
            "单缝参数", ['barrier_thickness', 'single_slit_width'])
        self.shape_groups[2] = self._make_shape_group("圆势参数", ['radius'])
        for g in self.shape_groups.values():
            layout.addWidget(g)

        self.edit_potential_btn = QPushButton("编辑自定义势能…")
        layout.addWidget(self.edit_potential_btn)

        group.setLayout(layout)
        return group

    def _make_shape_group(self, title, keys):
        box = QGroupBox(title)
        form = QFormLayout()
        form.setSpacing(3)
        for k in keys:
            self._add_slider(form, k)
        box.setLayout(form)
        return box

    def _build_packet_group(self):
        group = QGroupBox("初始波包")
        form = QFormLayout()
        form.setSpacing(3)
        for k in ('momentum', 'width', 'center_x', 'center_y', 'direction'):
            self._add_slider(form, k)
        group.setLayout(form)
        return group

    def _build_solver_group(self):
        group = QGroupBox("求解器")
        form = QFormLayout()
        form.setSpacing(3)
        for k in ('time_step', 'steps_per_frame'):
            self._add_slider(form, k)
        group.setLayout(form)
        return group

    def _open_potential_editor(self):
        dlg = PotentialEditor(self._custom_code, self)
        if dlg.exec() == QDialog.Accepted:
            error = self._compile_and_upload(dlg.code())
            if error:
                QMessageBox.warning(self, "编译错误", f"自定义势能函数编译失败:\n{error}")
                return
            potential_params[None].preset = 3
            if self.preset_combo.currentIndex() != 3:
                self.preset_combo.setCurrentIndex(3)

    def _compile_and_upload(self, code):
        values, error = potential_compiler.compile_potential(code, resolution[0], resolution[1])
        if error:
            return error
        update_custom_potential_field(values)
        self._custom_code = code
        self._custom_ready = True
        return None

    def _build_stats_group(self):
        # 统计信息放在窗口状态栏，面板只保留参数
        return None

    def _add_slider(self, form, key):
        label, lo, hi, default, fmt = self.SLIDERS[key]
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(lo), int(hi))
        slider.setValue(int(default))
        value_label = QLabel(fmt(default))
        value_label.setFixedWidth(44)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def _on_change(v):
            value_label.setText(fmt(v))
            self._apply_param(key, v)

        slider.valueChanged.connect(_on_change)
        h.addWidget(slider, 1)
        h.addWidget(value_label)
        form.addRow(label, row)
        self.sliders[key] = slider
        self.labels[key] = value_label
        self._apply_param(key, default)
        return slider

    # ---------- 参数写入 ----------
    def _apply_param(self, key, value):
        if key == "strength":
            potential_params[None].strength = value / 10.0
        elif key == "momentum":
            wave_packet_params[None].momentum = float(value)
        elif key == "width":
            wave_packet_params[None].width = value / 100.0
        elif key == "center_x":
            wave_packet_params[None].center_x = value / 100.0
        elif key == "center_y":
            wave_packet_params[None].center_y = value / 100.0
        elif key == "direction":
            wave_packet_params[None].direction = float(value)
        elif key == "time_step":
            simulation_params[None].time_step = value / 100.0
        elif key == "steps_per_frame":
            simulation_params[None].steps_per_frame = value
        elif key == "single_slit_width":
            potential_params[None].single_slit_width = float(value)
        elif key == "double_slit_width":
            potential_params[None].double_slit_width = float(value)
        elif key == "double_slit_separation":
            potential_params[None].double_slit_separation = float(value)
        elif key == "radius":
            potential_params[None].radius = float(value)
        elif key == "barrier_thickness":
            potential_params[None].barrier_thickness = float(value)

    # ---------- 事件 ----------
    def _on_preset_changed(self, idx):
        potential_params[None].preset = idx
        for k, box in self.shape_groups.items():
            box.setVisible(k == idx)
        if idx == 3 and not self._custom_ready:
            self._compile_and_upload(self._custom_code)

    def _on_reset(self):
        initialize()
        lbl = getattr(self, "lbl_time", None)
        if lbl is not None:
            lbl.setText("模拟时间 0.0")

    def _toggle_simulation(self):
        self.simulation_running = not self.simulation_running
        self.start_pause_btn.setText("暂停" if self.simulation_running else "开始")


# ── 自定义势能编辑对话框 ──────────────────
class PotentialEditor(QDialog):
    def __init__(self, code, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自定义势能函数")
        self.resize(720, 460)

        layout = QVBoxLayout(self)
        hint = QLabel("给 result 赋值。x、y 在向量化路径下是 numpy 数组，"
                      "可以直接写 np.where(...)；普通标量写法也能用，只是会慢一些。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888;")
        layout.addWidget(hint)

        self.editor = QTextEdit()
        self.editor.setPlainText(code)
        self.editor.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self.editor, 1)

        row = QHBoxLayout()
        reset_btn = QPushButton("恢复示例")
        reset_btn.clicked.connect(lambda: self.editor.setPlainText(ControlPanel.DEFAULT_CODE))
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("应用")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        row.addWidget(reset_btn)
        row.addStretch()
        row.addWidget(cancel_btn)
        row.addWidget(ok_btn)
        layout.addLayout(row)

    def code(self):
        return self.editor.toPlainText()


# ── 主窗口 ─────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("薛定谔方程模拟 – 自定义势能版")
        # 面板 330 + 左右边距 12；画布随窗口伸缩（默认接近 2:1）
        self.setGeometry(100, 100, 330 + 12 + 960, 505)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # 先初始化 Taichi 侧参数，再建面板：面板滑条的默认值最终生效
        init_parameters()
        initialize()
        custom_potential_field.fill(0.0)

        self.control_panel = ControlPanel()
        self.canvas = TaichiCanvas()

        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.canvas, 1)

        # 统计放在状态栏，面板就不用为了三行数字挤压参数区
        bar = self.statusBar()
        bar.setSizeGripEnabled(False)
        self.lbl_time = QLabel("模拟时间 0.0")
        self.lbl_norm = QLabel("归一化 –")
        self.lbl_fps = QLabel("帧率 –")
        for w in (self.lbl_time, self.lbl_norm, self.lbl_fps):
            bar.addWidget(w)
            bar.addWidget(QLabel("  "))
        self.control_panel.lbl_time = self.lbl_time

        self._sim_time = 0.0
        self._frames = 0
        self._last_stat = time.perf_counter()

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_simulation)
        self.timer.start(16)

    def _update_simulation(self):
        panel = self.control_panel
        if panel.simulation_running:
            self.canvas.update_simulation()
            self._sim_time += (simulation_params[None].steps_per_frame
                               * simulation_params[None].time_step)
            self._frames += 1

        now = time.perf_counter()
        if now - self._last_stat > 0.5:
            fps = self._frames / (now - self._last_stat)
            self._frames = 0
            self._last_stat = now
            self.lbl_time.setText(f"模拟时间 {self._sim_time:.1f}")
            self.lbl_fps.setText(f"帧率 {fps:.1f}")
            self.lbl_norm.setText(f"积分 |ψ|² {compute_norm():.3e}")

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


@ti.kernel
def compute_norm() -> ti.f32:
    s = 0.0
    for i, j in psi:
        v = psi[i, j]
        s += v[0] * v[0] + v[1] * v[1]
    # Grid spacing is 1/resolution[1] in both directions.
    return s / (resolution[1] * resolution[1])


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
