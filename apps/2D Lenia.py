from custom_import import *           # 假定包含 numpy, taichi 等基础导入
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import cm

# ================== 全局 Taichi 资源 ==================
state = None
new_state = None
kernel_field = None
kernel_vals = None
kernel_dx = None
kernel_dy = None
kernel_count = None

FIELD_SIZE = 256
MAX_KERNEL_RADIUS = 50
KERNEL_WINDOW = 2 * MAX_KERNEL_RADIUS + 1
MAX_KERNEL_ELEMENTS = KERNEL_WINDOW * KERNEL_WINDOW

clear_state = None
random_state = None
draw_points_batch = None   # 新的批量绘制 kernel
step = None

PRESET_FILE = "2D Lenia presets.npz"


def init_taichi(field_size=256):
    global state, new_state, kernel_field, kernel_vals, kernel_dx, kernel_dy, kernel_count
    global clear_state, random_state, draw_points_batch, step, FIELD_SIZE

    ti.reset()
    ti.init(arch=ti.gpu, default_fp=ti.f32)

    FIELD_SIZE = field_size
    state = ti.field(dtype=ti.f32, shape=(FIELD_SIZE, FIELD_SIZE))
    new_state = ti.field(dtype=ti.f32, shape=(FIELD_SIZE, FIELD_SIZE))
    kernel_field = ti.field(dtype=ti.f32, shape=(KERNEL_WINDOW, KERNEL_WINDOW))
    kernel_vals = ti.field(dtype=ti.f32, shape=MAX_KERNEL_ELEMENTS)
    kernel_dx   = ti.field(dtype=ti.i32, shape=MAX_KERNEL_ELEMENTS)
    kernel_dy   = ti.field(dtype=ti.i32, shape=MAX_KERNEL_ELEMENTS)
    kernel_count = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def clear_state():
        for i, j in state:
            state[i, j] = 0.0

    @ti.kernel
    def random_state():
        for i, j in state:
            state[i, j] = ti.random()

    # ---- 批量绘制 kernel（一次处理所有笔触） ----
    @ti.kernel
    def draw_points_batch(points: ti.types.ndarray(dtype=ti.f32, ndim=2),
                          brush_size: ti.i32):
        for idx in range(points.shape[0]):
            px = ti.cast(points[idx, 0], ti.i32)
            py = ti.cast(points[idx, 1], ti.i32)
            value = points[idx, 2]
            for i, j in ti.ndrange(
                (ti.max(0, py - brush_size), ti.min(FIELD_SIZE, py + brush_size + 1)),
                (ti.max(0, px - brush_size), ti.min(FIELD_SIZE, px + brush_size + 1))
            ):
                if (j - px) ** 2 + (i - py) ** 2 <= brush_size ** 2:
                    state[i, j] = value

    @ti.func
    def growth_func(u: ti.f32, mu: ti.f32, sigma: ti.f32) -> ti.f32:
        diff = u - mu
        return 2.0 * ti.exp(-diff * diff / (2.0 * sigma * sigma)) - 1.0

    @ti.kernel
    def step(mu: ti.f32, sigma: ti.f32, dt: ti.f32):
        cnt = kernel_count[None]
        for i, j in state:
            conv = 0.0
            for k in range(cnt):
                dx = kernel_dx[k]
                dy = kernel_dy[k]
                si = (i + dx) % FIELD_SIZE
                sj = (j + dy) % FIELD_SIZE
                conv += kernel_vals[k] * state[si, sj]
            new_state[i, j] = state[i, j] + dt * growth_func(conv, mu, sigma)
            if new_state[i, j] < 0.0:
                new_state[i, j] = 0.0
            if new_state[i, j] > 1.0:
                new_state[i, j] = 1.0

        for i, j in state:
            state[i, j] = new_state[i, j]


# ================== 画布（QOpenGLWidget + 批处理 + 颜色LUT） ==================
class LeniaCanvas(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.field_size = FIELD_SIZE
        self.canvas_size = 640
        self.setFixedSize(self.canvas_size, self.canvas_size)
        self.drawing = False
        self.brush_size = 2
        self.draw_value = 1.0
        self.last_point = None

        # 预计算颜色查找表（LUT），加速颜色映射
        self._lut = (cm.cubehelix(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        # 当前显示的 QImage，在 paintGL 中使用（复用对象，减少内存分配）
        self._display_img = QImage(self.canvas_size, self.canvas_size, QImage.Format_RGB888)
        self._display_img.fill(Qt.black)

        # 鼠标绘制批处理队列：每项为 (px, py, value)
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
        """从 GPU 状态生成缩放后的 QImage，存入 self._display_img"""
        arr = state.to_numpy()
        idx = np.clip((arr * 255).astype(np.uint8), 0, 255)
        rgb = self._lut[idx]                     # shape (field_size, field_size, 3)
        img = np.ascontiguousarray(rgb)
        fs = int(self.field_size)
        src = QImage(img.data, fs, fs, fs * 3, QImage.Format_RGB888)
        self._display_img = src.scaled(self.canvas_size, self.canvas_size,
                                       Qt.IgnoreAspectRatio, Qt.FastTransformation)

    def update_display(self):
        self._make_display_image()
        self.update()

    # ----- 鼠标交互优化：队列 + 定时合并 -----
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.draw_value = 1.0
        elif event.button() == Qt.RightButton:
            self.drawing = True
            self.draw_value = 0.0
        if self.drawing:
            self.last_point = event.position().toPoint()
            self._enqueue_brush(self.last_point)

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.drawing:
            return
        cur = event.position().toPoint()
        if self.last_point:
            self._enqueue_line(self.last_point, cur)
        else:
            self._enqueue_brush(cur)
        self.last_point = cur

    def mouseReleaseEvent(self, event: QMouseEvent):
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
        """Bresenham 直线，将线段上的点批量加入队列"""
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
        """将队列中的所有画笔操作一次性提交给 Taichi 核，并触发显示更新"""
        if not self._draw_queue:
            return
        # 构造 numpy 数组 shape (N, 3)：px, py, value
        pts = np.array(self._draw_queue, dtype=np.float32)
        draw_points_batch(pts, self.brush_size)
        self._draw_queue.clear()
        self.update_display()

    def center_on_activity(self, threshold=0.1):
        """将状态数组周期平移，使活跃区域的质心移动到图像中心"""
        arr = state.to_numpy()
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
        state.from_numpy(arr_shifted)
        self.update_display()


# ================== 核预览（保持不变） ==================
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
        self.R = 15
        self.num_rings = 3
        self.rho = 0.5
        self.omega = 0.15
        self.betas = [1.0] * 6
        self.mu = 0.156
        self.sigma = 0.0224
        self.dt = 0.1

        self._batch_update = False
        self.sliders = {}
        self.inputs = {}

        init_taichi(256)
        self.canvas = LeniaCanvas()
        self.init_ui()

        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.run_simulation)

        self.update_kernel()
        random_state()
        self.canvas.update_display()
        self.kernel_preview.update_preview(kernel_field.to_numpy())
        self.update_beta_visibility()

    def init_ui(self):
        self.setWindowTitle("SmoothLife 模拟器 (cubehelix + β振幅)")
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
        for text, slot in [("保存预设", self.save_preset),
                           ("加载预设", self.load_preset),
                           ("重命名", self.rename_preset),
                           ("删除", self.delete_preset)]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            tool_bar.addWidget(btn)
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

        kernel_config = [
            ("半径 R:", "R", (1, 50), 1, self.on_param, self.R, True, 0),
            ("环数:", "num_rings", (1, 6), 1, self.on_param, self.num_rings, True, 0),
            ("ρ (中心):", "rho", (0, 100), 0.01, self.on_param, int(self.rho*100), False, 3),
            ("ω (宽度):", "omega", (1, 100), 0.01, self.on_param, int(self.omega*100), False, 3),
        ]
        for label, key, (min_v, max_v), scale, callback, init_val, is_int, prec in kernel_config:
            slider, input_widget, row = self._add_input_row(label, min_v, max_v, init_val, callback, scale, is_int, prec)
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

        growth_config = [
            ("μ:", "mu", (0, 500), 0.001, self.on_growth, int(self.mu*1000), False, 3),
            ("σ:", "sigma", (1, 1000), 0.0001, self.on_growth, int(self.sigma*10000), False, 4),
            ("Δt:", "dt", (1, 100), 0.01, self.on_growth, int(self.dt*100), False, 2),
        ]
        for label, key, (min_v, max_v), scale, callback, init_val, is_int, prec in growth_config:
            slider, input_widget, row = self._add_input_row(label, min_v, max_v, init_val, callback, scale, is_int, prec)
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

    def _add_input_row(self, title, min_v, max_v, init_v, callback, scale=1.0, is_int=False, precision=2):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        label = QLabel(title)
        label.setFixedWidth(55)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(init_v)
        slider.valueChanged.connect(callback)
        input_widget = QLineEdit()
        input_widget.setFixedWidth(65)
        if is_int:
            input_widget.setText(str(init_v))
        else:
            input_widget.setText(f"{init_v * scale:.{precision}f}")
        slider.valueChanged.connect(lambda v: self._update_input_from_slider(slider, input_widget, scale, is_int, precision))
        input_widget.editingFinished.connect(lambda: self._update_slider_from_input(slider, input_widget, scale, is_int, precision, callback))
        layout.addWidget(label)
        layout.addWidget(slider, 1)
        layout.addWidget(input_widget)
        return slider, input_widget, row

    def _update_input_from_slider(self, slider, input_widget, scale, is_int, precision):
        if input_widget.hasFocus():
            return
        val = slider.value()
        if is_int:
            input_widget.setText(str(val))
        else:
            input_widget.setText(f"{val * scale:.{precision}f}")

    def _update_slider_from_input(self, slider, input_widget, scale, is_int, precision, callback):
        try:
            val = float(input_widget.text())
            if is_int:
                new_val = int(val)
            else:
                new_val = val / scale
            new_val = max(slider.minimum(), min(slider.maximum(), new_val))
            slider.blockSignals(True)
            slider.setValue(int(round(new_val)))
            slider.blockSignals(False)
            callback()
        except ValueError:
            self._update_input_from_slider(slider, input_widget, scale, is_int, precision)

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
        for i in range(6):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(2)
            label = QLabel(f"β{i+1}")
            label.setFixedWidth(20)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(self.betas[i] * 100))
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
        fmt = {
            'R': ('d', 0), 'num_rings': ('d', 0),
            'mu': ('.3f', 1), 'sigma': ('.4f', 1),
            'rho': ('.3f', 1), 'omega': ('.3f', 1), 'dt': ('.2f', 1)
        }
        for key, inp in self.inputs.items():
            if inp.hasFocus():
                continue
            val = getattr(self, key)
            fspec, _ = fmt.get(key, ('d', 0))
            inp.setText(f"{val:{fspec}}")
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
        random_state()
        self.canvas.update_display()

    def clear_field(self):
        clear_state()
        self.canvas.update_display()

    def run_simulation(self):
        step(self.mu, self.sigma, self.dt)
        self.canvas.update_display()

    def center_view(self):
        self.canvas.center_on_activity()

    def change_resolution(self, new_size_text):
        new_size = int(new_size_text)
        if new_size == FIELD_SIZE:
            return
        was_running = self.running
        if self.running:
            self.sim_timer.stop()
        init_taichi(new_size)
        self.canvas.field_size = new_size
        self.canvas._lut = (cm.cubehelix(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        self.update_kernel()
        random_state()
        self.canvas.update_display()
        self.kernel_preview.update_preview(kernel_field.to_numpy())
        if was_running:
            self.sim_timer.start(30)

    def update_kernel(self):
        center = MAX_KERNEL_RADIUS
        dy, dx = np.ogrid[-center:center+1, -center:center+1]
        r = np.sqrt(dx*dx + dy*dy) / self.R
        valid = r <= 1.0
        ring_idx = np.floor(r * self.num_rings).astype(int)
        frac = r * self.num_rings - ring_idx
        K = np.zeros((KERNEL_WINDOW, KERNEL_WINDOW), dtype=np.float32)
        for i in range(self.num_rings):
            mask = valid & (ring_idx == i)
            if mask.any():
                K[mask] = self.betas[i] * np.exp(-(frac[mask] - self.rho)**2 / (2.0 * self.omega**2))
        total = K.sum()
        if total < 1e-6:
            total = 1.0
        kernel_field.from_numpy(K)
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
        kernel_vals.from_numpy(_pad(vals, MAX_KERNEL_ELEMENTS, np.float32))
        kernel_dx.from_numpy(_pad(dxs, MAX_KERNEL_ELEMENTS, np.int32))
        kernel_dy.from_numpy(_pad(dys, MAX_KERNEL_ELEMENTS, np.int32))
        kernel_count[None] = count

    def update_beta_visibility(self):
        n = self.num_rings
        for i in range(6):
            self.beta_rows[i].setVisible(i < n)

    def _read_all_params(self):
        self.R = self.sliders["R"].value()
        self.num_rings = self.sliders["num_rings"].value()
        self.rho = self.sliders["rho"].value() * 0.01
        self.omega = self.sliders["omega"].value() * 0.01
        for i in range(6):
            self.betas[i] = self.beta_sliders[i].value() * 0.01
        self.mu = self.sliders["mu"].value() * 0.001
        self.sigma = self.sliders["sigma"].value() * 0.0001
        self.dt = self.sliders["dt"].value() * 0.01

    def on_param(self):
        if self._batch_update:
            return
        self._read_all_params()
        self._refresh_labels()
        self.update_beta_visibility()
        self.update_kernel()
        self.kernel_preview.update_preview(kernel_field.to_numpy())

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
        ti.reset()
        event.accept()

    # ================== 预设管理 ==================
    def _get_preset_dict(self):
        presets = {}
        try:
            data = np.load(PRESET_FILE, allow_pickle=False)
        except FileNotFoundError:
            return presets
        keys = list(data.keys())
        names = set()
        for k in keys:
            if k.endswith('_state') or k.endswith('_params'):
                name = k.rsplit('_', 1)[0]
                names.add(name)
        for name in names:
            state_key = f"{name}_state"
            params_key = f"{name}_params"
            if state_key in data and params_key in data:
                presets[name] = {
                    'state': data[state_key],
                    'params': data[params_key]
                }
        return presets

    def _save_preset_dict(self, preset_dict):
        save_dict = {}
        for n, d in preset_dict.items():
            save_dict[f"{n}_state"] = d['state']
            save_dict[f"{n}_params"] = d['params']
        np.savez_compressed(PRESET_FILE, **save_dict)

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "保存预设", "输入预设名称：")
        if not ok or not name.strip():
            return
        name = name.strip()
        presets = self._get_preset_dict()
        if name in presets:
            reply = QMessageBox.question(
                self, "覆盖确认",
                f"预设 “{name}” 已存在，是否覆盖？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        params = np.array([
            self.R, self.num_rings, self.rho, self.omega,
            *self.betas, self.mu, self.sigma, self.dt, FIELD_SIZE
        ], dtype=np.float32)
        state_arr = np.copy(state.to_numpy())
        presets[name] = {'state': state_arr, 'params': params}
        self._save_preset_dict(presets)

    def load_preset(self):
        presets = self._get_preset_dict()
        if not presets:
            QMessageBox.information(self, "提示", "没有找到任何预设。")
            return
        names = sorted(presets.keys())
        name, ok = QInputDialog.getItem(self, "加载预设", "选择预设：", names, 0, False)
        if not ok or not name:
            return
        self._apply_preset_data(presets[name])

    def _apply_preset_data(self, data):
        params = data['params']
        defaults = [15, 3, 0.5, 0.15, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    0.156, 0.0224, 0.1, 256]
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

        if field_size != FIELD_SIZE:
            init_taichi(field_size)
            self.canvas.field_size = field_size
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
        self.sliders["R"].setValue(self.R)
        self.sliders["num_rings"].setValue(self.num_rings)
        self.sliders["rho"].setValue(int(self.rho * 100))
        self.sliders["omega"].setValue(int(self.omega * 100))
        for i in range(6):
            self.beta_sliders[i].setValue(int(self.betas[i] * 100))
        self.sliders["mu"].setValue(int(round(self.mu * 1000)))
        self.sliders["sigma"].setValue(int(round(self.sigma * 10000)))
        self.sliders["dt"].setValue(int(round(self.dt * 100)))
        self._batch_update = False

        self._refresh_labels()
        self.update_beta_visibility()

        state.from_numpy(data['state'])
        self.update_kernel()
        self.kernel_preview.update_preview(kernel_field.to_numpy())
        self.canvas.update_display()

        if was_running:
            self.sim_timer.start(30)

    def rename_preset(self):
        presets = self._get_preset_dict()
        if not presets:
            QMessageBox.information(self, "提示", "没有预设可以重命名。")
            return
        names = sorted(presets.keys())
        old_name, ok = QInputDialog.getItem(self, "重命名预设", "选择预设：", names, 0, False)
        if not ok or not old_name:
            return
        new_name, ok = QInputDialog.getText(self, "重命名", "新名称：", text=old_name)
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        new_name = new_name.strip()
        if new_name in presets:
            reply = QMessageBox.question(
                self, "覆盖确认",
                f"预设 “{new_name}” 已存在，是否覆盖？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        presets[new_name] = presets.pop(old_name)
        self._save_preset_dict(presets)

    def delete_preset(self):
        presets = self._get_preset_dict()
        if not presets:
            QMessageBox.information(self, "提示", "没有预设可以删除。")
            return
        names = sorted(presets.keys())
        name, ok = QInputDialog.getItem(self, "删除预设", "选择预设：", names, 0, False)
        if not ok or not name:
            return
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除预设 “{name}” 吗？此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return
        del presets[name]
        self._save_preset_dict(presets)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LeniaApp()
    window.show()
    sys.exit(app.exec())