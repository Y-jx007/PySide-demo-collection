import sys
import numpy as np
import taichi_forge as ti
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QSlider, QPushButton, QSplitter,
    QFileDialog, QComboBox, QScrollArea, QFormLayout,
    QInputDialog, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QPainter, QImage, QPixmap, QMouseEvent
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
draw_point = None
step = None

PRESET_FILE = "2D Lenia presets.npz"


def init_taichi(field_size=256):
    global state, new_state, kernel_field, kernel_vals, kernel_dx, kernel_dy, kernel_count
    global clear_state, random_state, draw_point, step, FIELD_SIZE

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

    @ti.kernel
    def draw_point(px: int, py: int, brush: int, value: ti.f32):
        for i, j in ti.ndrange((max(0, py - brush), min(FIELD_SIZE, py + brush + 1)),
                               (max(0, px - brush), min(FIELD_SIZE, px + brush + 1))):
            if (j - px) ** 2 + (i - py) ** 2 <= brush ** 2:
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


# ================== 画布 ==================
class LeniaCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.field_size = FIELD_SIZE
        self.canvas_size = 640
        self.setFixedSize(self.canvas_size, self.canvas_size)
        self.drawing = False
        self.brush_size = 2
        self.draw_value = 1.0
        self._pixmap = QPixmap(self.canvas_size, self.canvas_size)
        self._pixmap.fill(Qt.black)
        self.last_point = None
        self.update_display()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)

    def update_display(self):
        arr = state.to_numpy()
        rgba = cm.cubehelix(arr)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        img = np.ascontiguousarray(rgb)
        qimg = QImage(img.data, self.field_size, self.field_size,
                      self.field_size * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg).scaled(
            self.canvas_size, self.canvas_size,
            Qt.IgnoreAspectRatio, Qt.FastTransformation
        )
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.draw_value = 1.0
            self.last_point = event.position().toPoint()
            self.apply_brush(self.last_point)
        elif event.button() == Qt.RightButton:
            self.drawing = True
            self.draw_value = 0.0
            self.last_point = event.position().toPoint()
            self.apply_brush(self.last_point)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing:
            cur = event.position().toPoint()
            if self.last_point:
                self.draw_line(self.last_point, cur)
            else:
                self.apply_brush(cur)
            self.last_point = cur

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.drawing = False
        self.last_point = None

    def apply_brush(self, pos: QPoint):
        px = int(pos.x() / self.canvas_size * self.field_size)
        py = int(pos.y() / self.canvas_size * self.field_size)
        px = max(0, min(self.field_size - 1, px))
        py = max(0, min(self.field_size - 1, py))
        draw_point(px, py, self.brush_size, self.draw_value)
        self.update_display()

    def draw_line(self, p1, p2):
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        while True:
            self.apply_brush(QPoint(x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy


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
        # 参数初始值（与滑块默认值保持一致）
        self.R = 15
        self.num_rings = 3
        self.rho = 0.5
        self.omega = 0.15
        self.betas = [1.0] * 6
        self.mu = 0.26
        self.sigma = 0.027
        self.dt = 0.1

        self._batch_update = False
        self.sliders = {}          # 统一存储所有滑块控件

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

    # ================== UI 构建（精简版） ==================
    def init_ui(self):
        self.setWindowTitle("SmoothLife 模拟器 (cubehelix + β振幅)")
        self.setGeometry(100, 100, 1060, 720)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # 顶部工具栏
        tool_bar = QHBoxLayout()
        buttons = [
            ("开始", self.toggle_simulation),
            ("随机重置", self.reset_simulation),
            ("清空", self.clear_field),
        ]
        for text, slot in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            tool_bar.addWidget(btn)
        tool_bar.addSpacing(20)

        preset_buttons = [
            ("保存预设", self.save_preset),
            ("加载预设", self.load_preset),
            ("重命名", self.rename_preset),
            ("删除", self.delete_preset),
        ]
        for text, slot in preset_buttons:
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
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(320)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(6)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # ----- 画笔 -----
        brush_group = QGroupBox("画笔")
        bf = QFormLayout(brush_group)
        self.brush_slider, self.brush_label, brush_row = self._add_slider_row(
            "半径:", 1, 20, self.canvas.brush_size, self.on_brush_change
        )
        bf.addRow(brush_row)
        left_layout.addWidget(brush_group)

        # ----- 分辨率 -----
        res_group = QGroupBox("分辨率")
        res_form = QFormLayout(res_group)
        self.res_combo = QComboBox()
        self.res_combo.addItems(["128", "256", "512", "1024"])
        self.res_combo.setCurrentText("256")
        self.res_combo.currentTextChanged.connect(self.change_resolution)
        res_form.addRow("网格:", self.res_combo)
        left_layout.addWidget(res_group)

        # ----- 核参数（配置驱动） -----
        kernel_group = QGroupBox("核参数")
        kf = QFormLayout(kernel_group)

        # 配置列表：每个元素 (名称, 属性键, 范围, 缩放, 回调, 初始值来源)
        kernel_config = [
            ("半径 R:", "R", (1, 50), 1, self.on_param, self.R),
            ("环数:", "num_rings", (1, 6), 1, self.on_param, self.num_rings),
            ("ρ (中心):", "rho", (0, 100), 0.01, self.on_param, int(self.rho*100)),
            ("ω (宽度):", "omega", (1, 100), 0.01, self.on_param, int(self.omega*100)),
        ]
        for label, key, (min_v, max_v), scale, callback, init_val in kernel_config:
            slider, val_label, row = self._add_slider_row(label, min_v, max_v, init_val, callback, scale)
            self.sliders[key] = slider
            self.sliders[key + "_label"] = val_label
            kf.addRow(row)

        # β 振幅（动态创建）
        beta_container = self._create_beta_controls()
        kf.addRow("振幅 β₁-β₆:", beta_container)

        self.kernel_preview = KernelPreview()
        kf.addRow("核预览:", self.kernel_preview)
        left_layout.addWidget(kernel_group)

        # ----- 生长参数 -----
        growth_group = QGroupBox("生长参数")
        gf = QFormLayout(growth_group)

        growth_config = [
            ("μ:", "mu", (1, 50), 0.01, self.on_growth, int(self.mu*100)),
            ("σ:", "sigma", (1, 100), 0.001, self.on_growth, int(self.sigma*1000)),
            ("Δt:", "dt", (1, 100), 0.01, self.on_growth, int(self.dt*100)),
        ]
        for label, key, (min_v, max_v), scale, callback, init_val in growth_config:
            slider, val_label, row = self._add_slider_row(label, min_v, max_v, init_val, callback, scale)
            self.sliders[key] = slider
            self.sliders[key + "_label"] = val_label
            gf.addRow(row)

        left_layout.addWidget(growth_group)

        # ----- 公式 -----
        formula_group = QGroupBox("公式")
        fl = QVBoxLayout(formula_group)
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

        # 右侧画布
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setSizes([320, 640])
        main_layout.addWidget(splitter, 1)

    def _add_slider_row(self, title, min_v, max_v, init_v, callback, scale=1.0):
        """创建一行：标签 + 滑块 + 数值标签，返回 (slider, value_label, row)"""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label = QLabel(title)
        label.setFixedWidth(60)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(init_v)
        slider.valueChanged.connect(callback)
        if scale < 1:
            value_label = QLabel(f"{init_v * scale:.3f}")
        else:
            value_label = QLabel(str(init_v))
        value_label.setFixedWidth(40)

        layout.addWidget(label)
        layout.addWidget(slider, 1)
        layout.addWidget(value_label)
        return slider, value_label, row

    def _create_beta_controls(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        self.beta_sliders = []
        self.beta_labels = []
        self.beta_rows = []

        for i in range(6):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            label = QLabel(f"β{i+1}")
            label.setFixedWidth(20)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(self.betas[i] * 100))
            slider.valueChanged.connect(self.on_param)
            value_label = QLabel(f"{self.betas[i]:.2f}")
            value_label.setFixedWidth(35)

            row_layout.addWidget(label)
            row_layout.addWidget(slider, 1)
            row_layout.addWidget(value_label)

            self.beta_sliders.append(slider)
            self.beta_labels.append(value_label)
            self.beta_rows.append(row)
            layout.addWidget(row)

        return container

    # ================== 参数同步与槽函数 ==================
    def _refresh_labels(self):
        """根据当前参数统一刷新所有标签"""
        self.sliders["R_label"].setText(str(self.R))
        self.sliders["num_rings_label"].setText(str(self.num_rings))
        self.sliders["rho_label"].setText(f"{self.rho:.2f}")
        self.sliders["omega_label"].setText(f"{self.omega:.2f}")
        for i, lbl in enumerate(self.beta_labels):
            lbl.setText(f"{self.betas[i]:.2f}")
        self.sliders["mu_label"].setText(f"{self.mu:.2f}")
        self.sliders["sigma_label"].setText(f"{self.sigma:.3f}")
        self.sliders["dt_label"].setText(f"{self.dt:.2f}")

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

    def change_resolution(self, new_size_text):
        new_size = int(new_size_text)
        if new_size == FIELD_SIZE:
            return
        was_running = self.running
        if self.running:
            self.sim_timer.stop()
        init_taichi(new_size)
        self.canvas.field_size = new_size
        self.update_kernel()
        random_state()
        self.canvas.update_display()
        self.kernel_preview.update_preview(kernel_field.to_numpy())
        if was_running:
            self.sim_timer.start(30)

    def update_kernel(self):
        center = MAX_KERNEL_RADIUS
        K = np.zeros((KERNEL_WINDOW, KERNEL_WINDOW), dtype=np.float32)
        total = 0.0
        for di in range(KERNEL_WINDOW):
            for dj in range(KERNEL_WINDOW):
                dx = di - center
                dy = dj - center
                r = np.sqrt(dx*dx + dy*dy) / self.R
                if r <= 1.0:
                    Br = r * self.num_rings
                    ring_idx = int(np.floor(Br))
                    frac = Br - ring_idx
                    if ring_idx < self.num_rings:
                        base = np.exp(-(frac - self.rho)**2 / (2.0 * self.omega**2))
                        val = self.betas[ring_idx] * base
                        K[di, dj] = val
                        total += val

        kernel_field.from_numpy(K)
        if total < 1e-6:
            total = 1.0
        idx = 0
        vals = np.zeros(MAX_KERNEL_ELEMENTS, dtype=np.float32)
        dxs = np.zeros(MAX_KERNEL_ELEMENTS, dtype=np.int32)
        dys = np.zeros(MAX_KERNEL_ELEMENTS, dtype=np.int32)
        for di in range(KERNEL_WINDOW):
            for dj in range(KERNEL_WINDOW):
                if K[di, dj] != 0.0:
                    vals[idx] = K[di, dj] / total
                    dxs[idx] = dj - center
                    dys[idx] = di - center
                    idx += 1

        kernel_vals.from_numpy(vals)
        kernel_dx.from_numpy(dxs)
        kernel_dy.from_numpy(dys)
        kernel_count[None] = idx

    def update_beta_visibility(self):
        n = self.num_rings
        for i in range(6):
            self.beta_rows[i].setVisible(i < n)

    def on_param(self):
        if self._batch_update:
            return
        self._read_params_from_ui()
        self._refresh_labels()
        self.update_beta_visibility()
        self.update_kernel()
        self.kernel_preview.update_preview(kernel_field.to_numpy())

    def on_growth(self):
        if self._batch_update:
            return
        self._read_growth_from_ui()
        self._refresh_labels()

    def on_brush_change(self):
        size = self.brush_slider.value()
        self.canvas.brush_size = size
        self.brush_label.setText(str(size))

    def _read_params_from_ui(self):
        self.R = self.sliders["R"].value()
        self.num_rings = self.sliders["num_rings"].value()
        self.rho = self.sliders["rho"].value() * 0.01
        self.omega = self.sliders["omega"].value() * 0.01
        for i in range(6):
            self.betas[i] = self.beta_sliders[i].value() * 0.01

    def _read_growth_from_ui(self):
        self.mu = self.sliders["mu"].value() * 0.01
        self.sigma = self.sliders["sigma"].value() * 0.001
        self.dt = self.sliders["dt"].value() * 0.01

    # ================== 预设管理（修复参数映射） ==================
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
        # 参数版本兼容
        if len(params) == 14:
            # 新格式：R, rings, rho, omega, b1..b6, mu, sigma, dt, size
            R, rings, rho, omega, *rest = params
            b1,b2,b3,b4,b5,b6, mu, sigma, dt, field_size = rest
            betas = [b1,b2,b3,b4,b5,b6]
        elif len(params) == 8:
            # 旧格式（无 β）：R, rings, rho, omega, mu, sigma, dt, size
            R, rings, rho, omega, mu, sigma, dt, field_size = params
            betas = [1.0] * 6
        else:
            # 回退：尽可能填充，缺失部分用默认值
            defaults = [15,3,0.5,0.15, 1.0,1.0,1.0,1.0,1.0,1.0, 0.26,0.027,0.1,256]
            full = defaults.copy()
            full[:len(params)] = params
            R, rings, rho, omega, *rest = full
            betas = rest[:6]
            mu, sigma, dt, field_size = rest[6], rest[7], rest[8], rest[9] if len(rest)>=10 else 256

        R = int(R)
        rings = int(rings)
        field_size = int(field_size)
        mu = float(mu)
        sigma = float(sigma)
        dt = float(dt)
        betas = [float(b) for b in betas]

        was_running = self.running
        if self.running:
            self.sim_timer.stop()

        if field_size != FIELD_SIZE:
            init_taichi(field_size)
            self.canvas.field_size = field_size
            self.res_combo.blockSignals(True)
            self.res_combo.setCurrentText(str(field_size))
            self.res_combo.blockSignals(False)

        self.R = R
        self.num_rings = rings
        self.rho = float(rho)
        self.omega = float(omega)
        self.betas = betas
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

        # 批量更新滑块（阻止重复刷新）
        self._batch_update = True
        self.sliders["R"].setValue(R)
        self.sliders["num_rings"].setValue(rings)
        self.sliders["rho"].setValue(int(rho * 100))
        self.sliders["omega"].setValue(int(omega * 100))
        for i in range(6):
            self.beta_sliders[i].setValue(int(betas[i] * 100))
        self.sliders["mu"].setValue(int(mu * 100))
        self.sliders["sigma"].setValue(int(sigma * 1000))
        self.sliders["dt"].setValue(int(dt * 100))
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