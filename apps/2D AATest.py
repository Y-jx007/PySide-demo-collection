from custom_import import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import cm

# ================== 全局 Taichi 资源 ==================
FIELD_SIZE = 256
MAX_KERNEL_RADIUS = 50
KERNEL_WINDOW = 2 * MAX_KERNEL_RADIUS + 1
MAX_KERNEL_ELEMENTS = KERNEL_WINDOW * KERNEL_WINDOW
MAX_PARTICLES = 5000

# 连续场
state = None
new_state = None
kernel_field = None
kernel_vals = None
kernel_dx = None
kernel_dy = None
kernel_count = None
clear_state = None
random_state = None
draw_points_batch = None
step_field_kernel = None

# 粒子
particle_pos = None
particle_num = None
particle_colors = None
particle_world_size = 20.0
init_particles_random = None
step_particles = None
compute_particle_colors = None

PRESET_FILE = "2D_Lenia_presets.npz"


def trapz(y, x):
    """简易梯形积分"""
    return np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0


def init_taichi(field_size=256, world_size=20.0):
    global FIELD_SIZE, particle_world_size
    global state, new_state, kernel_field, kernel_vals, kernel_dx, kernel_dy, kernel_count
    global clear_state, random_state, draw_points_batch, step_field_kernel
    global particle_pos, particle_num, particle_colors
    global init_particles_random, step_particles, compute_particle_colors

    ti.reset()
    ti.init(arch=ti.gpu, default_fp=ti.f32)

    FIELD_SIZE = field_size
    particle_world_size = world_size

    # 连续场
    state = ti.field(dtype=ti.f32, shape=(FIELD_SIZE, FIELD_SIZE))
    new_state = ti.field(dtype=ti.f32, shape=(FIELD_SIZE, FIELD_SIZE))
    kernel_field = ti.field(dtype=ti.f32, shape=(KERNEL_WINDOW, KERNEL_WINDOW))
    kernel_vals = ti.field(dtype=ti.f32, shape=MAX_KERNEL_ELEMENTS)
    kernel_dx = ti.field(dtype=ti.i32, shape=MAX_KERNEL_ELEMENTS)
    kernel_dy = ti.field(dtype=ti.i32, shape=MAX_KERNEL_ELEMENTS)
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
    def step_field_kernel(mu: ti.f32, sigma: ti.f32, dt: ti.f32):
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

    # ---------- 粒子 ----------
    particle_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PARTICLES)
    particle_num = ti.field(dtype=ti.i32, shape=())
    particle_colors = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)

    @ti.kernel
    def init_particles_random(count: ti.i32, L: ti.f32):
        particle_num[None] = count
        for i in range(count):
            particle_pos[i] = ti.Vector([ti.random() * L, ti.random() * L])

    @ti.kernel
    def compute_particle_colors(L: ti.f32, mu_k: ti.f32, sigma_k: ti.f32, w_k: ti.f32,
                                mu_g: ti.f32, sigma_g: ti.f32):
        n = particle_num[None]
        for i in range(n):
            U = 0.0
            pi = particle_pos[i]
            for j in range(n):
                pj = particle_pos[j]
                dx = pi[0] - pj[0]
                dy = pi[1] - pj[1]
                dx = dx - L * ti.round(dx / L)
                dy = dy - L * ti.round(dy / L)
                r = ti.sqrt(dx*dx + dy*dy)
                Kval = w_k * ti.exp(-((r - mu_k) / sigma_k) ** 2)
                U += Kval
            G = ti.exp(-((U - mu_g) / sigma_g) ** 2)
            particle_colors[i] = G

    @ti.kernel
    def step_particles(dt: ti.f32, L: ti.f32,
                      mu_k: ti.f32, sigma_k: ti.f32, w_k: ti.f32,
                      mu_g: ti.f32, sigma_g: ti.f32,
                      c_rep: ti.f32, speed: ti.f32):
        n = particle_num[None]
        # 先为每个粒子计算当前 U 并暂存到 colors（稍后重算，这里只用于下一步的梯度计算）
        for i in range(n):
            U_i = 0.0
            pi = particle_pos[i]
            for j in range(n):
                pj = particle_pos[j]
                dx = pi[0] - pj[0]
                dy = pi[1] - pj[1]
                dx = dx - L * ti.round(dx / L)
                dy = dy - L * ti.round(dy / L)
                r = ti.sqrt(dx*dx + dy*dy)
                U_i += w_k * ti.exp(-((r - mu_k) / sigma_k) ** 2)
            G_i = ti.exp(-((U_i - mu_g) / sigma_g) ** 2)
            # 计算梯度 ∇R 和 ∇G
            grad_R = ti.Vector([0.0, 0.0])
            grad_G = ti.Vector([0.0, 0.0])
            for j in range(n):
                if i == j:
                    continue
                pj = particle_pos[j]
                dx = pi[0] - pj[0]
                dy = pi[1] - pj[1]
                dx = dx - L * ti.round(dx / L)
                dy = dy - L * ti.round(dy / L)
                r = ti.sqrt(dx*dx + dy*dy)
                if r < 1e-6:
                    continue
                vec = ti.Vector([dx, dy]) / r

                # 排斥梯度
                if r < 1.0:
                    grad_R += c_rep * (1.0 - r) * vec

                # 核梯度对 ∇G 的贡献
                Kval = w_k * ti.exp(-((r - mu_k) / sigma_k) ** 2)
                Kprime = -2.0 * ((r - mu_k) / (sigma_k*sigma_k)) * Kval
                grad_G += Kprime * vec  # 这里积累的是 ∇U 的贡献（不含 G'）

            # 乘以 G'(U)
            Gprime = -2.0 * ((U_i - mu_g) / (sigma_g*sigma_g)) * G_i
            grad_G *= Gprime

            velocity = -grad_R + grad_G
            new_pos = pi + speed * dt * velocity
            new_pos[0] = new_pos[0] % L
            new_pos[1] = new_pos[1] % L
            particle_pos[i] = new_pos

        # 更新显示颜色
        for i in range(n):
            U = 0.0
            pi = particle_pos[i]
            for j in range(n):
                pj = particle_pos[j]
                dx = pi[0] - pj[0]
                dy = pi[1] - pj[1]
                dx = dx - L * ti.round(dx / L)
                dy = dy - L * ti.round(dy / L)
                r = ti.sqrt(dx*dx + dy*dy)
                U += w_k * ti.exp(-((r - mu_k) / sigma_k) ** 2)
            particle_colors[i] = ti.exp(-((U - mu_g) / sigma_g) ** 2)


# ================== 画布 ==================
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
        self.mode = 0  # 0=连续场, 1=粒子

        self._lut = (cm.cubehelix(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        self._display_img = QImage(self.canvas_size, self.canvas_size, QImage.Format_RGB888)
        self._display_img.fill(Qt.black)
        self._draw_queue = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._flush_draw_queue)
        self.update_display()

    def paintGL(self):
        if self.mode == 0:
            painter = QPainter(self)
            painter.drawImage(self.rect(), self._display_img)
            painter.end()
        else:
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.black)
            n = particle_num[None]
            if n == 0:
                painter.end()
                return
            pos_np = particle_pos.to_numpy()[:n]
            col_np = particle_colors.to_numpy()[:n]
            scale = self.canvas_size / particle_world_size
            for i in range(n):
                px = pos_np[i][0] * scale
                py = pos_np[i][1] * scale
                c = np.clip(int(col_np[i] * 255), 0, 255)
                color = QColor(*self._lut[c])
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(color))
                radius = 6.0  # 增大半径以便看到
                painter.drawEllipse(QPointF(px, py), radius, radius)
            painter.end()

    def _make_display_image(self):
        arr = state.to_numpy()
        idx = np.clip((arr * 255).astype(np.uint8), 0, 255)
        rgb = self._lut[idx]
        img = np.ascontiguousarray(rgb)
        fs = self.field_size
        src = QImage(img.data, fs, fs, fs * 3, QImage.Format_RGB888)
        self._display_img = src.scaled(self.canvas_size, self.canvas_size,
                                       Qt.IgnoreAspectRatio, Qt.FastTransformation)

    def update_display(self):
        if self.mode == 0:
            self._make_display_image()
        self.update()

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
        if self.mode == 0:
            px = int(pos.x() / self.canvas_size * self.field_size)
            py = int(pos.y() / self.canvas_size * self.field_size)
            px = max(0, min(self.field_size - 1, px))
            py = max(0, min(self.field_size - 1, py))
        else:
            px = pos.x() / self.canvas_size * particle_world_size
            py = pos.y() / self.canvas_size * particle_world_size
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
        if self.mode == 0:
            pts = np.array(self._draw_queue, dtype=np.float32)
            draw_points_batch(pts, self.brush_size)
        else:
            n = particle_num[None]
            for px, py, _ in self._draw_queue:
                if n < MAX_PARTICLES:
                    particle_pos[n] = ti.Vector([px, py])
                    n += 1
                else:
                    break
            particle_num[None] = n
            # 添加粒子后更新颜色
            self._update_particle_colors()
        self._draw_queue.clear()
        self.update_display()

    def _update_particle_colors(self):
        """更新粒子颜色，供添加粒子后调用"""
        # 计算当前参数下的 w_k
        from types import SimpleNamespace
        # 这里需要访问主窗口的参数，有点耦合，简单起见我们直接从主窗口传递，或者在此处使用全局参数
        # 为简化，我们在此处使用主窗口的实例来获取参数，但画布不应依赖主窗口细节。
        # 解决方案：在 LeniaApp 中主动调用 compute_particle_colors 并调用 update_display
        pass  # 实际更新由 LeniaApp 负责


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
        # 连续场参数
        self.R = 15
        self.num_rings = 3
        self.rho = 0.5
        self.omega = 0.15
        self.betas = [1.0] * 6
        self.mu_field = 0.156
        self.sigma_field = 0.0224
        self.dt_field = 0.1

        # 粒子参数
        self.world_size = 20.0
        self.mu_k = 4.0
        self.sigma_k = 1.0
        self.mu_g = 0.6
        self.sigma_g = 0.15
        self.c_rep = 1.0
        self.particle_speed = 1.0
        self.num_particles = 200

        self._batch_update = False
        self.sliders = {}  # 连续场滑块
        self.inputs = {}   # 连续场输入框

        init_taichi(FIELD_SIZE, self.world_size)
        self.canvas = LeniaCanvas()
        self.init_ui()
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.run_simulation)

        self.update_continuous_kernel()
        random_state()
        init_particles_random(self.num_particles, self.world_size)
        self.update_particle_colors()  # 初始化粒子颜色
        self.canvas.update_display()
        self.kernel_preview.update_preview(kernel_field.to_numpy())
        self.update_panel_visibility()
        self.update_beta_visibility()

    # ---------- UI ----------
    def init_ui(self):
        self.setWindowTitle("Lenia 模拟器 (场 + 粒子)")
        self.setGeometry(100, 100, 1160, 820)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 工具栏
        toolbar = QHBoxLayout()
        for text, slot in [("开始", self.toggle_sim), ("随机", self.reset),
                           ("清空", self.clear), ("居中", self.center_view)]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            toolbar.addWidget(btn)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)

        # 左侧控制
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        left = QWidget()
        left_layout = QVBoxLayout(left)

        # 模式切换
        mode_group = QGroupBox("模拟模式")
        mode_layout = QHBoxLayout(mode_group)
        self.btn_field = QRadioButton("连续场")
        self.btn_particle = QRadioButton("粒子")
        self.btn_field.setChecked(True)
        self.mode_buttons = QButtonGroup(self)
        self.mode_buttons.addButton(self.btn_field, 0)
        self.mode_buttons.addButton(self.btn_particle, 1)
        self.mode_buttons.buttonClicked.connect(self.switch_mode)
        mode_layout.addWidget(self.btn_field)
        mode_layout.addWidget(self.btn_particle)
        left_layout.addWidget(mode_group)

        # ---------- 连续场面板 ----------
        self.field_panel = QWidget()
        fl = QVBoxLayout(self.field_panel)

        # 画笔 + 分辨率
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        lbl_brush = QLabel("画笔:")
        lbl_brush.setFixedWidth(30)
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 20)
        self.brush_slider.setValue(2)
        self.brush_slider.valueChanged.connect(self.on_brush_change)
        self.brush_input = QLineEdit()
        self.brush_input.setFixedWidth(30)
        self.brush_input.setText("2")
        self.brush_input.editingFinished.connect(self.set_brush_from_input)
        row1.addWidget(lbl_brush)
        row1.addWidget(self.brush_slider, 1)
        row1.addWidget(self.brush_input)

        lbl_res = QLabel("网格:")
        lbl_res.setFixedWidth(30)
        self.res_combo = QComboBox()
        self.res_combo.addItems(["128", "256", "512", "1024"])
        self.res_combo.setCurrentText("256")
        self.res_combo.currentTextChanged.connect(self.change_resolution)
        row1.addWidget(lbl_res)
        row1.addWidget(self.res_combo, 1)
        fl.addLayout(row1)

        # 核参数
        kernel_group = QGroupBox("核参数")
        kf = QFormLayout(kernel_group)
        self._add_field_slider(kf, "半径 R:", "R", 1, 50, self.R, is_int=True)
        self._add_field_slider(kf, "环数:", "num_rings", 1, 6, self.num_rings, is_int=True)
        self._add_field_slider(kf, "ρ (中心):", "rho", 0, 100, int(self.rho*100), scale=0.01, fmt=".3f")
        self._add_field_slider(kf, "ω (宽度):", "omega", 1, 100, int(self.omega*100), scale=0.01, fmt=".3f")
        # beta 控件
        beta_container = QWidget()
        bl = QVBoxLayout(beta_container)
        self.beta_sliders = []
        self.beta_labels = []
        self.beta_rows = []
        for i in range(6):
            rw = QWidget()
            rl = QHBoxLayout(rw)
            rl.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(f"β{i+1}")
            lbl.setFixedWidth(20)
            s = QSlider(Qt.Horizontal)
            s.setRange(0, 100)
            s.setValue(int(self.betas[i]*100))
            s.valueChanged.connect(self.on_field_param)
            vl = QLabel(f"{self.betas[i]:.2f}")
            vl.setFixedWidth(40)
            rl.addWidget(lbl)
            rl.addWidget(s, 1)
            rl.addWidget(vl)
            self.beta_sliders.append(s)
            self.beta_labels.append(vl)
            self.beta_rows.append(rw)
            bl.addWidget(rw)
        kf.addRow("振幅 β₁-β₆:", beta_container)
        self.kernel_preview = KernelPreview()
        kf.addRow("核预览:", self.kernel_preview)
        fl.addWidget(kernel_group)

        # 生长参数
        growth_group = QGroupBox("生长参数")
        gf = QFormLayout(growth_group)
        self._add_field_slider(gf, "μ:", "mu_field", 0, 500, int(self.mu_field*1000), scale=0.001, fmt=".3f")
        self._add_field_slider(gf, "σ:", "sigma_field", 1, 1000, int(self.sigma_field*10000), scale=0.0001, fmt=".4f")
        self._add_field_slider(gf, "Δt:", "dt_field", 1, 100, int(self.dt_field*100), scale=0.01, fmt=".2f")
        fl.addWidget(growth_group)

        # 公式显示
        formula = QGroupBox("公式")
        flg = QVBoxLayout(formula)
        flg.addWidget(QLabel("连续场 Lenia 公式 (略...)"))
        fl.addWidget(formula)
        left_layout.addWidget(self.field_panel)

        # ---------- 粒子面板 ----------
        self.particle_panel = QWidget()
        pl = QFormLayout(self.particle_panel)
        self.particle_sliders = {}

        def add_pslider(title, key, minv, maxv, step, init, fmt=".2f"):
            s = QSlider(Qt.Horizontal)
            s.setRange(int(minv/step), int(maxv/step))
            s.setValue(int(init/step))
            lab = QLabel()
            lab.setFixedWidth(50)
            s.valueChanged.connect(lambda v: lab.setText(f"{v*step:{fmt}}"))
            s.valueChanged.connect(self.on_particle_param)
            lab.setText(f"{init:{fmt}}")
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(title)
            lbl.setFixedWidth(70)
            rl.addWidget(lbl)
            rl.addWidget(s, 1)
            rl.addWidget(lab)
            self.particle_sliders[key] = (s, lab, step)
            return row

        pl.addRow("世界尺寸 L:", add_pslider("L:", "L", 5, 50, 0.5, self.world_size))
        pl.addRow("μ_k:", add_pslider("μ_k:", "mu_k", 1, 10, 0.1, self.mu_k))
        pl.addRow("σ_k:", add_pslider("σ_k:", "sigma_k", 0.1, 5, 0.1, self.sigma_k))
        pl.addRow("μ_g:", add_pslider("μ_g:", "mu_g", 0.01, 2, 0.01, self.mu_g))
        pl.addRow("σ_g:", add_pslider("σ_g:", "sigma_g", 0.01, 2, 0.01, self.sigma_g))
        pl.addRow("c_rep:", add_pslider("c_rep:", "c_rep", 0.1, 5, 0.1, self.c_rep))
        pl.addRow("速度系数:", add_pslider("speed:", "speed", 0.1, 5, 0.1, self.particle_speed))
        pl.addRow("粒子数:", add_pslider("N:", "N", 10, MAX_PARTICLES, 10, self.num_particles, "d"))

        # 预设按钮（共用）
        btn_save = QPushButton("保存预设")
        btn_load = QPushButton("加载预设")
        btn_save.clicked.connect(self.save_preset)
        btn_load.clicked.connect(self.load_preset)
        pl.addRow(btn_save, btn_load)
        left_layout.addWidget(self.particle_panel)

        left_layout.addStretch()
        scroll.setWidget(left)
        splitter.addWidget(scroll)

        # 右侧画布
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setSizes([350, 700])
        main_layout.addWidget(splitter)

    def _add_field_slider(self, form, label, key, minv, maxv, init_val, scale=1.0, fmt=".2f", is_int=False):
        s = QSlider(Qt.Horizontal)
        s.setRange(minv, maxv)
        s.setValue(init_val)
        inp = QLineEdit()
        inp.setFixedWidth(65)
        if is_int:
            inp.setText(str(init_val))
        else:
            inp.setText(f"{init_val*scale:{fmt}}")
        s.valueChanged.connect(lambda v: self._field_slider_changed(key, v, inp, scale, fmt, is_int))
        inp.editingFinished.connect(lambda: self._field_input_changed(key, s, inp, scale, fmt, is_int))
        self.sliders[key] = s
        self.inputs[key] = inp
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        lb = QLabel(label)
        lb.setFixedWidth(70)
        rl.addWidget(lb)
        rl.addWidget(s, 1)
        rl.addWidget(inp)
        form.addRow(row)

    def _field_slider_changed(self, key, v, inp, scale, fmt, is_int):
        if inp.hasFocus():
            return
        if is_int:
            inp.setText(str(v))
        else:
            inp.setText(f"{v*scale:{fmt}}")
        self.on_field_param()

    def _field_input_changed(self, key, s, inp, scale, fmt, is_int):
        try:
            val = float(inp.text())
            if is_int:
                new_val = int(val)
            else:
                new_val = val / scale
            new_val = max(s.minimum(), min(s.maximum(), new_val))
            s.blockSignals(True)
            s.setValue(int(round(new_val)))
            s.blockSignals(False)
            self.on_field_param()
        except ValueError:
            v = s.value()
            if is_int:
                inp.setText(str(v))
            else:
                inp.setText(f"{v*scale:{fmt}}")

    def on_field_param(self):
        if self._batch_update:
            return
        self.R = self.sliders["R"].value()
        self.num_rings = self.sliders["num_rings"].value()
        self.rho = self.sliders["rho"].value() * 0.01
        self.omega = self.sliders["omega"].value() * 0.01
        for i in range(6):
            self.betas[i] = self.beta_sliders[i].value() * 0.01
            self.beta_labels[i].setText(f"{self.betas[i]:.2f}")
        self.mu_field = self.sliders["mu_field"].value() * 0.001
        self.sigma_field = self.sliders["sigma_field"].value() * 0.0001
        self.dt_field = self.sliders["dt_field"].value() * 0.01
        self.update_beta_visibility()
        self.update_continuous_kernel()
        self.kernel_preview.update_preview(kernel_field.to_numpy())

    def on_particle_param(self):
        if self._batch_update:
            return
        for key, (s, lab, step) in self.particle_sliders.items():
            val = s.value() * step
            if key == "L":
                self.world_size = val
                global particle_world_size
                particle_world_size = val
            elif key == "mu_k":
                self.mu_k = val
            elif key == "sigma_k":
                self.sigma_k = val
            elif key == "mu_g":
                self.mu_g = val
            elif key == "sigma_g":
                self.sigma_g = val
            elif key == "c_rep":
                self.c_rep = val
            elif key == "speed":
                self.particle_speed = val
            elif key == "N":
                self.num_particles = int(val)
        if self.canvas.mode == 1:
            if particle_num[None] != self.num_particles:
                init_particles_random(self.num_particles, self.world_size)
            self.update_particle_colors()
            self.canvas.update_display()

    def update_continuous_kernel(self):
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
        kernel_vals.from_numpy(np.pad(vals, (0, MAX_KERNEL_ELEMENTS - len(vals)))[:MAX_KERNEL_ELEMENTS])
        kernel_dx.from_numpy(np.pad(dxs, (0, MAX_KERNEL_ELEMENTS - len(dxs)))[:MAX_KERNEL_ELEMENTS].astype(np.int32))
        kernel_dy.from_numpy(np.pad(dys, (0, MAX_KERNEL_ELEMENTS - len(dys)))[:MAX_KERNEL_ELEMENTS].astype(np.int32))
        kernel_count[None] = count

    def update_beta_visibility(self):
        for i in range(6):
            self.beta_rows[i].setVisible(i < self.num_rings)

    def switch_mode(self, btn):
        mode = self.mode_buttons.id(btn)
        self.canvas.mode = mode
        self.update_panel_visibility()
        if mode == 1:
            # 确保粒子颜色是最新的
            self.update_particle_colors()
        self.canvas.update_display()

    def update_panel_visibility(self):
        if self.canvas.mode == 0:
            self.field_panel.setVisible(True)
            self.particle_panel.setVisible(False)
        else:
            self.field_panel.setVisible(False)
            self.particle_panel.setVisible(True)

    def toggle_sim(self):
        if self.running:
            self.sim_timer.stop()
            self.running = False
        else:
            self.sim_timer.start(30)
            self.running = True

    def reset(self):
        if self.canvas.mode == 0:
            random_state()
        else:
            init_particles_random(self.num_particles, self.world_size)
            self.update_particle_colors()
        self.canvas.update_display()

    def clear(self):
        if self.canvas.mode == 0:
            clear_state()
        else:
            particle_num[None] = 0
        self.canvas.update_display()

    def center_view(self):
        if self.canvas.mode == 0:
            arr = state.to_numpy()
            active = arr > 0.1
            if not active.any():
                return
            ys, xs = np.nonzero(active)
            ang_y = 2*np.pi*ys/self.canvas.field_size
            ang_x = 2*np.pi*xs/self.canvas.field_size
            cy = (np.arctan2(np.sum(np.sin(ang_y)), np.sum(np.cos(ang_y))) % (2*np.pi)) / (2*np.pi) * self.canvas.field_size
            cx = (np.arctan2(np.sum(np.sin(ang_x)), np.sum(np.cos(ang_x))) % (2*np.pi)) / (2*np.pi) * self.canvas.field_size
            shift_y = int(round(self.canvas.field_size/2 - cy))
            shift_x = int(round(self.canvas.field_size/2 - cx))
            state.from_numpy(np.roll(arr, (shift_y, shift_x), axis=(0,1)))
            self.canvas.update_display()

    def get_w_k(self):
        """计算归一化权重 w_k"""
        rs = np.linspace(0, self.world_size/2, 1000)
        Kr = np.exp(-((rs - self.mu_k) / self.sigma_k) ** 2)
        integral = trapz(Kr * rs, rs) * 2 * np.pi
        return 1.0 / integral if integral > 1e-8 else 1.0

    def update_particle_colors(self):
        w_k = self.get_w_k()
        compute_particle_colors(self.world_size, self.mu_k, self.sigma_k, w_k,
                                self.mu_g, self.sigma_g)

    def run_simulation(self):
        if self.canvas.mode == 0:
            step_field_kernel(self.mu_field, self.sigma_field, self.dt_field)
        else:
            w_k = self.get_w_k()
            step_particles(self.dt_field, self.world_size,
                           self.mu_k, self.sigma_k, w_k,
                           self.mu_g, self.sigma_g,
                           self.c_rep, self.particle_speed)
            compute_particle_colors(self.world_size, self.mu_k, self.sigma_k, w_k,
                                    self.mu_g, self.sigma_g)
        self.canvas.update_display()

    def on_brush_change(self):
        size = self.brush_slider.value()
        self.canvas.brush_size = size
        self.brush_input.setText(str(size))

    def set_brush_from_input(self):
        try:
            v = int(self.brush_input.text())
            v = max(1, min(20, v))
            self.brush_slider.setValue(v)
            self.canvas.brush_size = v
        except:
            self.brush_input.setText(str(self.canvas.brush_size))

    def change_resolution(self, new_size_text):
        new_size = int(new_size_text)
        if new_size == FIELD_SIZE:
            return
        was_running = self.running
        if self.running:
            self.sim_timer.stop()
        init_taichi(new_size, self.world_size)
        self.canvas.field_size = new_size
        self.canvas._lut = (cm.cubehelix(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        self.update_continuous_kernel()
        random_state()
        init_particles_random(self.num_particles, self.world_size)
        self.update_particle_colors()
        self.canvas.update_display()
        self.kernel_preview.update_preview(kernel_field.to_numpy())
        if was_running:
            self.sim_timer.start(30)

    # 预设保存/加载 (支持两种模式)
    def save_preset(self):
        name, ok = QInputDialog.getText(self, "保存预设", "名称:")
        if not ok or not name.strip():
            return
        name = name.strip()
        presets = self._get_presets()
        if name in presets:
            if QMessageBox.question(self, "覆盖?", "已存在，覆盖？") != QMessageBox.Yes:
                return
        data = {}
        params = np.array([
            self.R, self.num_rings, self.rho, self.omega, *self.betas,
            self.mu_field, self.sigma_field, self.dt_field, FIELD_SIZE,
            self.canvas.mode, self.world_size, self.mu_k, self.sigma_k,
            self.mu_g, self.sigma_g, self.c_rep, self.particle_speed, self.num_particles
        ], dtype=np.float32)
        data['params'] = params
        if self.canvas.mode == 0:
            data['state'] = state.to_numpy().copy()
        else:
            n = particle_num[None]
            data['particles'] = {
                'pos': particle_pos.to_numpy()[:n].copy(),
                'n': n
            }
        presets[name] = data
        self._save_presets(presets)

    def load_preset(self):
        presets = self._get_presets()
        if not presets:
            QMessageBox.information(self, "提示", "无预设")
            return
        name, ok = QInputDialog.getItem(self, "加载预设", "选择:", sorted(presets.keys()), 0, False)
        if not ok:
            return
        self._apply_preset(presets[name])

    def _get_presets(self):
        try:
            return np.load(PRESET_FILE, allow_pickle=True).item()
        except:
            return {}

    def _save_presets(self, d):
        np.save(PRESET_FILE, d, allow_pickle=True)

    def _apply_preset(self, d):
        params = d['params']
        defaults = [15,3,0.5,0.15,1,1,1,1,1,1,0.156,0.0224,0.1,256,0,20,4,1,0.6,0.15,1,1,200]
        params = list(params) + list(defaults[len(params):])
        (self.R, self.num_rings, self.rho, self.omega,
         *b, self.mu_field, self.sigma_field, self.dt_field, fs,
         mode, ws, muk, sigk, mug, sigg, crep, speed, np_) = params[:len(defaults)]
        self.betas = b[:6]
        self.world_size = ws
        self.mu_k = muk
        self.sigma_k = sigk
        self.mu_g = mug
        self.sigma_g = sigg
        self.c_rep = crep
        self.particle_speed = speed
        self.num_particles = int(np_)
        was_running = self.running
        if self.running:
            self.sim_timer.stop()
        if int(fs) != FIELD_SIZE:
            init_taichi(int(fs), self.world_size)
            self.canvas.field_size = int(fs)
            self.canvas._lut = (cm.cubehelix(np.linspace(0,1,256))[:,:3]*255).astype(np.uint8)
            self.res_combo.setCurrentText(str(int(fs)))
        # 更新滑块
        self._batch_update = True
        self.sliders["R"].setValue(int(self.R))
        self.sliders["num_rings"].setValue(int(self.num_rings))
        self.sliders["rho"].setValue(int(self.rho*100))
        self.sliders["omega"].setValue(int(self.omega*100))
        for i in range(6):
            self.beta_sliders[i].setValue(int(self.betas[i]*100))
        self.sliders["mu_field"].setValue(int(self.mu_field*1000))
        self.sliders["sigma_field"].setValue(int(self.sigma_field*10000))
        self.sliders["dt_field"].setValue(int(self.dt_field*100))
        # 粒子滑块
        for key, (s, lab, step) in self.particle_sliders.items():
            if key == "L":
                s.setValue(int(self.world_size/step))
            elif key == "mu_k":
                s.setValue(int(self.mu_k/step))
            elif key == "sigma_k":
                s.setValue(int(self.sigma_k/step))
            elif key == "mu_g":
                s.setValue(int(self.mu_g/step))
            elif key == "sigma_g":
                s.setValue(int(self.sigma_g/step))
            elif key == "c_rep":
                s.setValue(int(self.c_rep/step))
            elif key == "speed":
                s.setValue(int(self.particle_speed/step))
            elif key == "N":
                s.setValue(int(self.num_particles/step))
        self._batch_update = False
        self.canvas.mode = int(mode)
        if int(mode) == 0:
            self.btn_field.setChecked(True)
            if 'state' in d:
                state.from_numpy(d['state'])
            else:
                random_state()
        else:
            self.btn_particle.setChecked(True)
            if 'particles' in d:
                p = d['particles']
                particle_num[None] = p['n']
                if p['n'] > 0:
                    particle_pos.from_numpy(p['pos'])
                self.update_particle_colors()
            else:
                init_particles_random(self.num_particles, self.world_size)
                self.update_particle_colors()
        self.update_continuous_kernel()
        self.kernel_preview.update_preview(kernel_field.to_numpy())
        self.update_panel_visibility()
        self.canvas.update_display()
        if was_running:
            self.sim_timer.start(30)

    def closeEvent(self, event):
        self.sim_timer.stop()
        ti.reset()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LeniaApp()
    window.show()
    sys.exit(app.exec())