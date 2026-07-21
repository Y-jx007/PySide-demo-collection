from custom_import import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import cm

# ================== 常量 ==================
MAX_KERNEL_RADIUS = 50
KERNEL_WINDOW = 2 * MAX_KERNEL_RADIUS + 1
MAX_KERNEL_ELEMENTS = KERNEL_WINDOW * KERNEL_WINDOW

PRESET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Lenia presets.npz")

# ================== 参数元数据 ==================
PARAM_META = {
    'R':         {'display': '半径 R',   'slider_range': (1, 50),   'scale': 1.0,  'decimals': 0, 'default': 15},
    'num_rings': {'display': '环数',     'slider_range': (1, 6),    'scale': 1.0,  'decimals': 0, 'default': 3},
    'rho':       {'display': 'ρ (中心)', 'slider_range': (0, 100),  'scale': 0.01, 'decimals': 3, 'default': 0.5},
    'omega':     {'display': 'ω (宽度)', 'slider_range': (1, 100),  'scale': 0.01, 'decimals': 3, 'default': 0.15},
    'mu':        {'display': 'μ',       'slider_range': (0, 500),  'scale': 0.001,'decimals': 3, 'default': 0.156},
    'sigma':     {'display': 'σ',       'slider_range': (1, 1000), 'scale': 0.0001,'decimals': 4, 'default': 0.0224},
    'dt':        {'display': 'Δt',      'slider_range': (1, 100),  'scale': 0.01, 'decimals': 2, 'default': 0.1},
}
BETA_META = {'display': '振幅 β', 'slider_range': (0, 100), 'scale': 0.01, 'decimals': 2, 'default': 1.0, 'count': 6}

# ================== 模块级 Taichi 核心（接收字段作为参数） ==================
@ti.func
def growth_func(u: ti.f32, mu: ti.f32, sigma: ti.f32) -> ti.f32:
    diff = u - mu
    return 2.0 * ti.exp(-diff * diff / (2.0 * sigma * sigma)) - 1.0

@ti.kernel
def _clear_state_kernel(state: ti.template()):
    for i, j in state:
        state[i, j] = 0.0

@ti.kernel
def _random_state_kernel(state: ti.template()):
    for i, j in state:
        state[i, j] = ti.random()

@ti.kernel
def _draw_points_batch_kernel(state: ti.template(),
                              points: ti.types.ndarray(dtype=ti.f32, ndim=2),
                              brush_size: ti.i32):
    for idx in range(points.shape[0]):
        px = ti.cast(points[idx, 0], ti.i32)
        py = ti.cast(points[idx, 1], ti.i32)
        value = points[idx, 2]
        for i, j in ti.ndrange(
            (ti.max(0, py - brush_size), ti.min(state.shape[0], py + brush_size + 1)),
            (ti.max(0, px - brush_size), ti.min(state.shape[1], px + brush_size + 1))
        ):
            if (j - px) ** 2 + (i - py) ** 2 <= brush_size ** 2:
                state[i, j] = value

@ti.kernel
def _step_kernel(state: ti.template(), new_state: ti.template(),
                 kernel_vals: ti.template(), kernel_dx: ti.template(),
                 kernel_dy: ti.template(), kernel_count: ti.template(),
                 mu: ti.f32, sigma: ti.f32, dt: ti.f32):
    cnt = kernel_count[None]
    field_size = state.shape[0]
    for i, j in state:
        conv = 0.0
        for k in range(cnt):
            dx = kernel_dx[k]
            dy = kernel_dy[k]
            si = (i + dx) % field_size
            sj = (j + dy) % field_size
            conv += kernel_vals[k] * state[si, sj]
        new_state[i, j] = state[i, j] + dt * growth_func(conv, mu, sigma)
        if new_state[i, j] < 0.0:
            new_state[i, j] = 0.0
        if new_state[i, j] > 1.0:
            new_state[i, j] = 1.0
    for i, j in state:
        state[i, j] = new_state[i, j]

# ================== Lenia 模拟封装类 ==================
class LeniaSimulation:
    def __init__(self, field_size=256):
        # 每个 Simulation 实例创建前重置 Taichi 上下文（单实例模式）
        ti.reset()
        ti.init(arch=ti.gpu, default_fp=ti.f32)

        self.field_size = field_size
        # 创建所有 Taichi 字段
        self.state = ti.field(dtype=ti.f32, shape=(field_size, field_size))
        self.new_state = ti.field(dtype=ti.f32, shape=(field_size, field_size))
        self.kernel_field = ti.field(dtype=ti.f32, shape=(KERNEL_WINDOW, KERNEL_WINDOW))
        self.kernel_vals = ti.field(dtype=ti.f32, shape=MAX_KERNEL_ELEMENTS)
        self.kernel_dx   = ti.field(dtype=ti.i32, shape=MAX_KERNEL_ELEMENTS)
        self.kernel_dy   = ti.field(dtype=ti.i32, shape=MAX_KERNEL_ELEMENTS)
        self.kernel_count = ti.field(dtype=ti.i32, shape=())

    def random_state(self):
        _random_state_kernel(self.state)

    def clear_state(self):
        _clear_state_kernel(self.state)

    def draw_points_batch(self, points: np.ndarray, brush_size: int):
        # points: (N, 3) float32 array
        _draw_points_batch_kernel(self.state, points, brush_size)

    def step(self, mu: float, sigma: float, dt: float):
        _step_kernel(self.state, self.new_state,
                     self.kernel_vals, self.kernel_dx, self.kernel_dy,
                     self.kernel_count, mu, sigma, dt)

    def update_kernel(self, R: int, num_rings: int, rho: float, omega: float, betas: list):
        """根据参数构建核并更新 Taichi 字段"""
        R = int(R)
        num_rings = int(num_rings)
        center = MAX_KERNEL_RADIUS
        dy, dx = np.ogrid[-center:center+1, -center:center+1]
        r = np.sqrt(dx*dx + dy*dy) / R
        valid = r <= 1.0
        ring_idx = np.floor(r * num_rings).astype(int)
        frac = r * num_rings - ring_idx
        K = np.zeros((KERNEL_WINDOW, KERNEL_WINDOW), dtype=np.float32)
        for i in range(num_rings):
            mask = valid & (ring_idx == i)
            if mask.any():
                K[mask] = betas[i] * np.exp(-(frac[mask] - rho)**2 / (2.0 * omega**2))
        total = K.sum()
        if total < 1e-6:
            total = 1.0
        self.kernel_field.from_numpy(K)
        nonzeros = np.nonzero(K)
        vals = K[nonzeros] / total
        dxs = nonzeros[1] - center
        dys = nonzeros[0] - center
        count = min(len(vals), MAX_KERNEL_ELEMENTS)

        def _pad(arr, length, dtype):
            if len(arr) == length:
                return arr.astype(dtype)
            padded = np.zeros(length, dtype=dtype)
            padded[:len(arr)] = arr[:length]
            return padded

        self.kernel_vals.from_numpy(_pad(vals, MAX_KERNEL_ELEMENTS, np.float32))
        self.kernel_dx.from_numpy(_pad(dxs, MAX_KERNEL_ELEMENTS, np.int32))
        self.kernel_dy.from_numpy(_pad(dys, MAX_KERNEL_ELEMENTS, np.int32))
        self.kernel_count[None] = count

    def get_state_numpy(self) -> np.ndarray:
        return self.state.to_numpy()

    def set_state_numpy(self, arr: np.ndarray):
        self.state.from_numpy(arr)

    def center_on_activity(self, threshold=0.1):
        """周期性平移使活跃区域质心位于图像中心"""
        arr = self.state.to_numpy()
        active = arr > threshold
        if not active.any():
            return
        ys, xs = np.nonzero(active)
        angles_y = 2 * np.pi * ys / self.field_size
        angles_x = 2 * np.pi * xs / self.field_size
        sum_cos_y = np.sum(np.cos(angles_y))
        sum_sin_y = np.sum(np.sin(angles_y))
        sum_cos_x = np.sum(np.cos(angles_x))
        sum_sin_x = np.sum(np.sin(angles_x))
        mean_angle_y = np.arctan2(sum_sin_y, sum_cos_y)
        mean_angle_x = np.arctan2(sum_sin_x, sum_cos_x)
        cy = (mean_angle_y / (2 * np.pi)) % 1.0 * self.field_size
        cx = (mean_angle_x / (2 * np.pi)) % 1.0 * self.field_size
        shift_y = int(round(self.field_size / 2 - cy))
        shift_x = int(round(self.field_size / 2 - cx))
        arr_shifted = np.roll(arr, shift=(shift_y, shift_x), axis=(0, 1))
        self.state.from_numpy(arr_shifted)

    def get_kernel_preview(self) -> np.ndarray:
        """返回核预览数组"""
        return self.kernel_field.to_numpy()

# ================== 画布（使用 sim 实例） ==================
class LeniaCanvas(QOpenGLWidget):
    def __init__(self, sim: LeniaSimulation, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.field_size = sim.field_size
        self.canvas_size = 640
        self.setFixedSize(self.canvas_size, self.canvas_size)
        self.drawing = False
        self.brush_size = 2
        self.draw_value = 1.0
        self.last_point = None

        self._lut = (cm.cubehelix(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        self._display_img = QImage(self.canvas_size, self.canvas_size, QImage.Format_RGB888)
        self._display_img.fill(Qt.black)

        self._draw_queue = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._flush_draw_queue)

        self.update_display()

    def initializeGL(self):
        pass

    def paintGL(self):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self._display_img)
        painter.end()

    def _make_display_image(self):
        arr = self.sim.get_state_numpy()
        idx = np.clip((arr * 255).astype(np.uint8), 0, 255)
        rgb = self._lut[idx]
        img = np.ascontiguousarray(rgb)
        fs = int(self.field_size)
        src = QImage(img.data, fs, fs, fs * 3, QImage.Format_RGB888)
        self._display_img = src.scaled(self.canvas_size, self.canvas_size,
                                       Qt.IgnoreAspectRatio, Qt.FastTransformation)

    def update_display(self):
        self._make_display_image()
        self.update()

    # ----- 鼠标交互（队列+定时合并） -----
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.draw_value = 1.0
        elif event.button() == Qt.RightButton:
            self.drawing = True
            self.draw_value = 0.0
        if self.drawing:
            self.last_point = event.position().toPoint()
            self._enqueue_brush(self.last_point)

    def mouseMoveEvent(self, event):
        if not self.drawing:
            return
        cur = event.position().toPoint()
        if self.last_point:
            self._enqueue_line(self.last_point, cur)
        else:
            self._enqueue_brush(cur)
        self.last_point = cur

    def mouseReleaseEvent(self, event):
        self.drawing = False
        self.last_point = None
        self._flush_timer.stop()
        self._flush_draw_queue()

    def _enqueue_brush(self, pos: QPoint):
        px = int(pos.x() / self.canvas_size * self.field_size)
        py = int(pos.y() / self.canvas_size * self.field_size)
        px = max(0, min(self.field_size - 1, px))
        py = max(0, min(self.field_size - 1, py))
        self._draw_queue.append((px, py, self.draw_value))
        if not self._flush_timer.isActive():
            self._flush_timer.start(5)

    def _enqueue_line(self, p1, p2):
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        while True:
            self._enqueue_brush(QPoint(x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def _flush_draw_queue(self):
        if not self._draw_queue:
            return
        pts = np.array(self._draw_queue, dtype=np.float32)
        self.sim.draw_points_batch(pts, self.brush_size)
        self._draw_queue.clear()
        self.update_display()

    def center_on_activity(self, threshold=0.1):
        self.sim.center_on_activity(threshold)
        self.update_display()

    def set_field_size(self, size):
        self.field_size = size


# ================== 核预览 ==================
class KernelPreview(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(2.2, 2.2), dpi=80)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.axis('off')
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.setFixedSize(180, 180)

    def update_preview(self, arr):
        self.ax.clear()
        self.ax.imshow(arr, cmap='cubehelix',
                       extent=[-MAX_KERNEL_RADIUS, MAX_KERNEL_RADIUS,
                               -MAX_KERNEL_RADIUS, MAX_KERNEL_RADIUS])
        self.ax.axis('off')
        self.draw()


# ================== 主窗口 ==================
class LeniaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.running = False

        # 当前参数值（真实值）
        self.R = PARAM_META['R']['default']
        self.num_rings = PARAM_META['num_rings']['default']
        self.rho = PARAM_META['rho']['default']
        self.omega = PARAM_META['omega']['default']
        self.betas = [BETA_META['default']] * BETA_META['count']
        self.mu = PARAM_META['mu']['default']
        self.sigma = PARAM_META['sigma']['default']
        self.dt = PARAM_META['dt']['default']

        self._batch_update = False
        self.sliders = {}
        self.inputs = {}

        # 创建模拟实例
        self.sim = LeniaSimulation(256)
        self.canvas = LeniaCanvas(self.sim)
        self.init_ui()

        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.run_simulation)

        self.update_kernel()
        self.sim.random_state()
        self.canvas.update_display()
        self.kernel_preview.update_preview(self.sim.get_kernel_preview())
        self.update_beta_visibility()

    def init_ui(self):
        self.setWindowTitle("2D Lenia")
        self.setGeometry(100, 100, 1060, 720)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # 顶部工具栏
        tool_bar = QHBoxLayout()
        for text, slot in [("开始", self.toggle_simulation),
                           ("随机重置", self.reset_simulation),
                           ("清空", self.clear_field),
                           ("居中", self.center_view)]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            tool_bar.addWidget(btn)
        tool_bar.addSpacing(10)
        self.gallery_btn = QPushButton("预设")
        self.gallery_btn.clicked.connect(self.open_gallery)
        tool_bar.addWidget(self.gallery_btn)
        tool_bar.addStretch()
        main_layout.addLayout(tool_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)

        # 左侧控制面板
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(310)
        scroll.setMaximumWidth(330)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(4)
        left_layout.setContentsMargins(2, 2, 2, 2)

        # 画笔 + 分辨率 同行
        tools_row = QHBoxLayout()
        tools_row.setSpacing(6)

        brush_widget = QWidget()
        brush_layout = QHBoxLayout(brush_widget)
        brush_layout.setContentsMargins(0, 0, 0, 0)
        brush_layout.setSpacing(2)
        lbl_brush = QLabel("画笔:")
        lbl_brush.setFixedWidth(30)
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 20)
        self.brush_slider.setValue(self.canvas.brush_size)
        self.brush_slider.valueChanged.connect(self.on_brush_change)
        self.brush_input = QLineEdit()
        self.brush_input.setFixedWidth(30)
        self.brush_input.setText(str(self.canvas.brush_size))
        self.brush_input.editingFinished.connect(lambda: self._set_brush_from_input())
        brush_layout.addWidget(lbl_brush)
        brush_layout.addWidget(self.brush_slider, 1)
        brush_layout.addWidget(self.brush_input)
        tools_row.addWidget(brush_widget)

        res_widget = QWidget()
        res_layout = QHBoxLayout(res_widget)
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.setSpacing(2)
        lbl_res = QLabel("网格:")
        lbl_res.setFixedWidth(30)
        self.res_combo = QComboBox()
        self.res_combo.addItems(["128", "256", "512", "1024"])
        self.res_combo.setCurrentText("256")
        self.res_combo.currentTextChanged.connect(self.change_resolution)
        res_layout.addWidget(lbl_res)
        res_layout.addWidget(self.res_combo, 1)
        tools_row.addWidget(res_widget)

        left_layout.addLayout(tools_row)

        # 核参数
        kernel_group = QGroupBox("核参数")
        kf = QFormLayout(kernel_group)
        kf.setSpacing(4)
        kf.setContentsMargins(4, 4, 4, 4)

        # 利用 PARAM_META 动态生成控件
        for key in ['R', 'num_rings', 'rho', 'omega']:
            meta = PARAM_META[key]
            slider_val = int(round(getattr(self, key) / meta['scale']))
            slider, input_widget, row = self._add_param_row(key, meta, slider_val,
                                                            callback=self.on_param)
            self.sliders[key] = slider
            self.inputs[key] = input_widget
            kf.addRow(row)

        beta_container = self._create_beta_controls()
        kf.addRow("振幅 β₁-β₆:", beta_container)

        self.kernel_preview = KernelPreview()
        kf.addRow("核预览:", self.kernel_preview)
        left_layout.addWidget(kernel_group)

        # 生长参数
        growth_group = QGroupBox("生长参数")
        gf = QFormLayout(growth_group)
        gf.setSpacing(4)
        gf.setContentsMargins(4, 4, 4, 4)

        for key in ['mu', 'sigma', 'dt']:
            meta = PARAM_META[key]
            slider_val = int(round(getattr(self, key) / meta['scale']))
            slider, input_widget, row = self._add_param_row(key, meta, slider_val,
                                                            callback=self.on_growth)
            self.sliders[key] = slider
            self.inputs[key] = input_widget
            gf.addRow(row)

        left_layout.addWidget(growth_group)

        # 公式
        formula_group = QGroupBox("公式")
        fl = QVBoxLayout(formula_group)
        fl.setContentsMargins(4, 4, 4, 4)
        fl.addWidget(QLabel(
            "<b>卷积：</b> U(x) = Σ K(y-x) A(y) / Σ K(y-x)<br>"
            "<b>生长：</b> G(u) = 2·exp(-(u-μ)²/(2σ²)) - 1<br>"
            "<b>更新：</b> A<sub>t+Δt</sub>(x) = clamp(A<sub>t</sub>(x)+Δt·G(U<sub>t</sub>), 0,1)<br>"
            "<b>核：</b> bell<sub>i</sub> = β<sub>i</sub> · exp(-(frac-ρ)²/(2ω²))"
        ))
        left_layout.addWidget(formula_group)

        left_layout.addStretch()
        scroll.setWidget(left)
        splitter.addWidget(scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setSizes([330, 640])
        main_layout.addWidget(splitter, 1)

    def _add_param_row(self, key, meta, slider_value, callback):
        """根据元数据创建滑块+文本框行"""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        label = QLabel(meta['display'])
        label.setFixedWidth(65)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(*meta['slider_range'])
        slider.setValue(slider_value)
        slider.valueChanged.connect(callback)
        input_widget = QLineEdit()
        input_widget.setFixedWidth(65)
        precision = meta['decimals']
        scale = meta['scale']
        input_widget.setText(f"{slider_value * scale:.{precision}f}")
        slider.valueChanged.connect(lambda v: self._update_input_from_slider(slider, input_widget, scale, precision))
        input_widget.editingFinished.connect(lambda: self._update_slider_from_input(slider, input_widget, scale, precision, callback))
        layout.addWidget(label)
        layout.addWidget(slider, 1)
        layout.addWidget(input_widget)
        return slider, input_widget, row

    def _update_input_from_slider(self, slider, input_widget, scale, precision):
        if input_widget.hasFocus():
            return
        val = slider.value()
        input_widget.setText(f"{val * scale:.{precision}f}")

    def _update_slider_from_input(self, slider, input_widget, scale, precision, callback):
        try:
            val = float(input_widget.text())
            new_val = val / scale
            new_val = max(slider.minimum(), min(slider.maximum(), new_val))
            slider.blockSignals(True)
            slider.setValue(int(round(new_val)))
            slider.blockSignals(False)
            callback()
        except ValueError:
            self._update_input_from_slider(slider, input_widget, scale, precision)

    def _set_brush_from_input(self):
        try:
            val = int(self.brush_input.text())
            val = max(1, min(20, val))
            self.brush_slider.setValue(val)
            self.canvas.brush_size = val
        except ValueError:
            self.brush_input.setText(str(self.canvas.brush_size))

    def _create_beta_controls(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.beta_sliders = []
        self.beta_labels = []
        self.beta_rows = []
        for i in range(BETA_META['count']):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(2)
            label = QLabel(f"β{i+1}")
            label.setFixedWidth(20)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(*BETA_META['slider_range'])
            slider.setValue(int(self.betas[i] / BETA_META['scale']))
            slider.valueChanged.connect(self.on_param)
            value_label = QLabel(f"{self.betas[i]:.2f}")
            value_label.setFixedWidth(40)
            row_layout.addWidget(label)
            row_layout.addWidget(slider, 1)
            row_layout.addWidget(value_label)
            self.beta_sliders.append(slider)
            self.beta_labels.append(value_label)
            self.beta_rows.append(row)
            layout.addWidget(row)
        return container

    def _refresh_labels(self):
        for key, inp in self.inputs.items():
            if inp.hasFocus():
                continue
            meta = PARAM_META[key]
            val = getattr(self, key)
            inp.setText(f"{val:.{meta['decimals']}f}")
        for i, lbl in enumerate(self.beta_labels):
            lbl.setText(f"{self.betas[i]:.2f}")

    def toggle_simulation(self):
        if self.running:
            self.sim_timer.stop()
            self.sender().setText("开始")
            self.running = False
        else:
            self.sim_timer.start(30)
            self.sender().setText("暂停")
            self.running = True

    def reset_simulation(self):
        self.sim.random_state()
        self.canvas.update_display()

    def clear_field(self):
        self.sim.clear_state()
        self.canvas.update_display()

    def run_simulation(self):
        self.sim.step(self.mu, self.sigma, self.dt)
        self.canvas.update_display()

    def center_view(self):
        self.canvas.center_on_activity()

    def change_resolution(self, new_size_text):
        new_size = int(new_size_text)
        if new_size == self.sim.field_size:
            return
        was_running = self.running
        if self.running:
            self.sim_timer.stop()
        # 重新创建模拟实例
        self.sim = LeniaSimulation(new_size)
        self.canvas.sim = self.sim
        self.canvas.set_field_size(new_size)
        self.canvas._lut = (cm.cubehelix(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        self.update_kernel()
        self.sim.random_state()
        self.canvas.update_display()
        self.kernel_preview.update_preview(self.sim.get_kernel_preview())
        if was_running:
            self.sim_timer.start(30)

    def update_kernel(self):
        self.sim.update_kernel(self.R, self.num_rings, self.rho, self.omega, self.betas)
        self.kernel_preview.update_preview(self.sim.get_kernel_preview())

    def update_beta_visibility(self):
        n = self.num_rings
        for i in range(BETA_META['count']):
            self.beta_rows[i].setVisible(i < n)

    def _read_all_params(self):
        # 使用元数据将滑块值转换为真实值
        for key in PARAM_META:
            meta = PARAM_META[key]
            setattr(self, key, self.sliders[key].value() * meta['scale'])
        for i in range(BETA_META['count']):
            self.betas[i] = self.beta_sliders[i].value() * BETA_META['scale']

    def on_param(self):
        if self._batch_update:
            return
        self._read_all_params()
        self._refresh_labels()
        self.update_beta_visibility()
        self.update_kernel()

    def on_growth(self):
        if self._batch_update:
            return
        self._read_all_params()
        self._refresh_labels()

    def on_brush_change(self):
        size = self.brush_slider.value()
        self.canvas.brush_size = size
        self.brush_input.setText(str(size))

    def closeEvent(self, event):
        self.sim_timer.stop()
        # 清理 Taichi 资源（全局）
        ti.reset()
        event.accept()

    # ================== 预设画廊 ==================
    def open_gallery(self):
        if hasattr(self, '_gallery_popup') and self._gallery_popup.isVisible():
            self._gallery_popup.hide()
            return
        gallery = PresetGallery(self)
        btn_rect = self.gallery_btn.rect()
        bottom_left = self.gallery_btn.mapToGlobal(QPoint(0, btn_rect.height()))
        gallery.move(bottom_left)
        gallery.show()
        self._gallery_popup = gallery

    def _apply_preset_data(self, data):
        params = data['params']
        defaults = [PARAM_META['R']['default'], PARAM_META['num_rings']['default'],
                    PARAM_META['rho']['default'], PARAM_META['omega']['default']] + \
                   [BETA_META['default']]*6 + \
                   [PARAM_META['mu']['default'], PARAM_META['sigma']['default'],
                    PARAM_META['dt']['default'], 256]
        if len(params) < 14:
            params = list(params) + defaults[len(params):]
        else:
            params = list(params[:14])
        R, rings, rho, omega, *rest = params
        betas = rest[:6]
        mu, sigma, dt, field_size = rest[6], rest[7], rest[8], int(rest[9])

        was_running = self.running
        if self.running:
            self.sim_timer.stop()

        if field_size != self.sim.field_size:
            self.sim = LeniaSimulation(field_size)
            self.canvas.sim = self.sim
            self.canvas.set_field_size(field_size)
            self.canvas._lut = (cm.cubehelix(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
            self.res_combo.blockSignals(True)
            self.res_combo.setCurrentText(str(field_size))
            self.res_combo.blockSignals(False)

        self.R = int(R)
        self.num_rings = int(rings)
        self.rho = float(rho)
        self.omega = float(omega)
        self.betas = [float(b) for b in betas]
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.dt = float(dt)

        self._batch_update = True
        for key in PARAM_META:
            meta = PARAM_META[key]
            self.sliders[key].setValue(int(round(getattr(self, key) / meta['scale'])))
        for i in range(BETA_META['count']):
            self.beta_sliders[i].setValue(int(round(self.betas[i] / BETA_META['scale'])))
        self._batch_update = False

        self._refresh_labels()
        self.update_beta_visibility()

        self.sim.set_state_numpy(data['state'])
        self.update_kernel()
        self.canvas.update_display()

        if was_running:
            self.sim_timer.start(30)


class PresetGallery(QWidget):
    """下拉画廊：网格排列预设，双击加载但保持打开，点击外部关闭"""
    class Thumbnail(QWidget):
        clicked = Signal(str)

        def __init__(self, name, state_arr, parent=None):
            super().__init__(parent)
            self.name = name
            self._state = state_arr
            self._selected = False
            self._cached_scaled = None
            self.setCursor(Qt.PointingHandCursor)
            self.setToolTip(name)

        def paintEvent(self, event):
            s = self.width()
            if self._cached_scaled is None or self._cached_scaled.width() != s:
                arr = self._state
                fs = arr.shape[0]
                active = arr > 0.05
                if active.any():
                    ys, xs = np.nonzero(active)
                    cy = np.arctan2(np.mean(np.sin(2*np.pi*ys/fs)), np.mean(np.cos(2*np.pi*ys/fs))) / (2*np.pi) * fs
                    cx = np.arctan2(np.mean(np.sin(2*np.pi*xs/fs)), np.mean(np.cos(2*np.pi*xs/fs))) / (2*np.pi) * fs
                    arr = np.roll(arr, shift=(int(round(fs/2 - cy)), int(round(fs/2 - cx))), axis=(0, 1))
                arr_clip = np.clip(arr * 255, 0, 255).astype(np.uint8)
                qimg = QImage(arr_clip.tobytes(), fs, fs, fs, QImage.Format_Grayscale8)
                scaled = qimg.scaled(s-4, s-4, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._cached_scaled = scaled
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.drawImage((s - self._cached_scaled.width())//2, (s - self._cached_scaled.height())//2, self._cached_scaled)
            if self._selected:
                painter.setPen(QPen(QColor(0xFF, 0xD7, 0x00), 2))
                painter.drawRect(1, 1, s - 3, s - 3)
            painter.setPen(Qt.white)
            f = painter.font()
            f.setPointSize(8)
            painter.setFont(f)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(self.name)
            th = fm.height()
            painter.fillRect(2, 2, min(tw+8, s-4), th+2, QColor(0,0,0,160))
            painter.drawText(QRect(4, 2, min(tw+8, s-4)-4, th), Qt.AlignLeft|Qt.AlignVCenter, self.name)

        def mousePressEvent(self, event):
            self.clicked.emit(self.name)

        def mouseDoubleClickEvent(self, event):
            self.clicked.emit('\0' + self.name)

    class GridWidget(QWidget):
        def __init__(self, cell_size, parent=None):
            super().__init__(parent)
            self.cell_size = cell_size
            self.cols = 1
            self._thumbnails = []

        def set_layout_data(self, cols, thumb_data):
            self.cols = cols
            for w, _, _ in self._thumbnails:
                w.setParent(None)
            self._thumbnails = thumb_data
            for w, r, c in self._thumbnails:
                w.setParent(self)
                w.setFixedSize(self.cell_size, self.cell_size)
                w.move(c * self.cell_size, r * self.cell_size)
                w.show()
            rows = max((r for _, r, _ in self._thumbnails), default=0) + 1
            self.setMinimumSize(cols * self.cell_size, rows * self.cell_size)
            self.updateGeometry()

        def paintEvent(self, event):
            super().paintEvent(event)
            if self.cell_size <= 0:
                return
            painter = QPainter(self)
            painter.setPen(QPen(QColor(0x55, 0x55, 0x55), 1))
            w = self.width()
            h = self.height()
            for col in range(self.cols + 1):
                x = col * self.cell_size
                painter.drawLine(x, 0, x, h)
            rows = max(1, (h + self.cell_size - 1) // self.cell_size)
            for row in range(rows + 1):
                y = row * self.cell_size
                painter.drawLine(0, y, w, y)

    def __init__(self, app_window):
        super().__init__()
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.app = app_window
        self._presets = {}
        self._selected = None
        self._cell_size = 100

        self.resize(626, 480)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(8, 8, 8, 8)

        tb = QHBoxLayout()
        tb.setContentsMargins(0, 0, 0, 4)
        for txt, slot in [("保存当前", self._save_current),
                          ("重命名", self._rename_selected),
                          ("删除", self._delete_selected)]:
            btn = QPushButton(txt)
            btn.clicked.connect(slot)
            tb.addWidget(btn)
        tb.addStretch()
        main_layout.addLayout(tb)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self._refresh_content()

    def _cols(self):
        w = self.tabs.contentsRect().width()
        return max(2, w // self._cell_size)

    def _refresh_content(self):
        self.tabs.clear()
        self._presets = PresetGallery.load_preset_dict()
        by_rings = {}
        for n, d in self._presets.items():
            by_rings.setdefault(int(d['params'][1]), []).append((n, d))
        for rings in sorted(by_rings.keys()):
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            cols = self._cols()
            grid = self.GridWidget(self._cell_size)
            items = sorted(by_rings[rings], key=lambda x: x[0])
            thumb_data = []
            for i, (n, d) in enumerate(items):
                t = self.Thumbnail(n, d['state'])
                t.clicked.connect(self._on_thumbnail_clicked)
                if self._selected == n:
                    t._selected = True
                row, col = i // cols, i % cols
                thumb_data.append((t, row, col))
            grid.set_layout_data(cols, thumb_data)
            sc.setWidget(grid)
            self.tabs.addTab(sc, f"{rings} 环")
        if not self._presets:
            lab = QLabel("暂无预设，请先保存")
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet("color:#888; font-size:16px;")
            self.tabs.addTab(lab, "空")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        for i in range(self.tabs.count()):
            sc = self.tabs.widget(i)
            if sc and sc.widget():
                grid = sc.widget()
                if isinstance(grid, self.GridWidget):
                    cols = self._cols()
                    thumbs = grid._thumbnails[:]
                    new_data = []
                    for idx, (w, _, _) in enumerate(thumbs):
                        row, col = idx // cols, idx % cols
                        new_data.append((w, row, col))
                    grid.set_layout_data(cols, new_data)

    def _on_thumbnail_clicked(self, name):
        if name.startswith('\0'):
            real = name[1:]
            if real in self._presets:
                self.app._apply_preset_data(self._presets[real])
            return
        self._selected = name
        for i in range(self.tabs.count()):
            sc = self.tabs.widget(i)
            if sc and sc.widget():
                grid = sc.widget()
                if isinstance(grid, self.GridWidget):
                    for w, _, _ in grid._thumbnails:
                        if isinstance(w, self.Thumbnail):
                            w._selected = (w.name == name)
                            w._cached_scaled = None
                            w.update()

    def _save_current(self):
        name, ok = QInputDialog.getText(self, "保存预设", "名称：")
        if not ok or not name.strip():
            return
        name = name.strip()
        ps = PresetGallery.load_preset_dict()
        if name in ps:
            r = QMessageBox.question(self, "覆盖", f"'{name}'已存在，覆盖？",
                                     QMessageBox.Yes | QMessageBox.No)
            if r == QMessageBox.No:
                return
        ps[name] = {
            'state': np.copy(self.app.sim.get_state_numpy()),
            'params': np.array([
                self.app.R, self.app.num_rings,
                self.app.rho, self.app.omega,
                *self.app.betas,
                self.app.mu, self.app.sigma,
                self.app.dt, self.app.sim.field_size
            ], dtype=np.float32)
        }
        PresetGallery.save_preset_dict(ps)
        self._refresh_content()

    def _delete_selected(self):
        if not self._selected:
            return
        r = QMessageBox.question(self, "确认", f"删除'{self._selected}'?",
                                 QMessageBox.Yes | QMessageBox.No)
        if r == QMessageBox.No:
            return
        ps = PresetGallery.load_preset_dict()
        ps.pop(self._selected, None)
        PresetGallery.save_preset_dict(ps)
        self._selected = None
        self._refresh_content()

    def _rename_selected(self):
        if not self._selected:
            return
        ps = PresetGallery.load_preset_dict()
        old = self._selected
        new, ok = QInputDialog.getText(self, "重命名", "新名称：", text=old)
        if not ok or not new.strip() or new.strip() == old:
            return
        new = new.strip()
        if new in ps:
            QMessageBox.warning(self, "失败", f"'{new}'已存在")
            return
        ps[new] = ps.pop(old)
        PresetGallery.save_preset_dict(ps)
        self._selected = new
        self._refresh_content()

    @staticmethod
    def load_preset_dict():
        presets = {}
        try:
            data = np.load(PRESET_FILE, allow_pickle=False)
        except FileNotFoundError:
            return presets
        keys = list(data.keys())
        names = set()
        for k in keys:
            if k.endswith('_state') or k.endswith('_params'):
                names.add(k.rsplit('_', 1)[0])
        for name in names:
            state_key = f"{name}_state"
            params_key = f"{name}_params"
            if state_key in data and params_key in data:
                presets[name] = {
                    'state': data[state_key],
                    'params': data[params_key]
                }
        return presets

    @staticmethod
    def save_preset_dict(preset_dict):
        save_dict = {}
        for n, d in preset_dict.items():
            save_dict[f"{n}_state"] = d['state']
            save_dict[f"{n}_params"] = d['params']
        np.savez_compressed(PRESET_FILE, **save_dict)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LeniaApp()
    window.show()
    sys.exit(app.exec())