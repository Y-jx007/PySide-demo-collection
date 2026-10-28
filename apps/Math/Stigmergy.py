from common.qt_base import (FPSCounter, LabeledSlider, QCheckBox, QColor, QComboBox, QFormLayout,
                            QGroupBox, QHBoxLayout, QImage, QLabel, QMainWindow, QPainter, QPushButton,
                            QSpinBox, QTimer, QVBoxLayout, QWidget, Qt, math, run_app)
from common.numpy_base import (np)
from common.taichi_base import (ti)

ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=True)

# ================== 常量 ==================
FIELD_SIZE = 512
MAX_PARTICLES = 200000


# ================== Taichi 核 ==================
@ti.data_oriented
class StigmergySimulation:
    """黏菌 / 蚁群的信息素寻路（stigmergy）模型。

    个体只做三件事：嗅前方三个点、朝气味更浓的方向转、在脚下留下气味；
    气味场则扩散 + 衰减。宏观的输运网络完全由这两条局部规则涌现。
    """

    def __init__(self, size=FIELD_SIZE, n_particles=30000):
        self.size = size
        self.n_max = MAX_PARTICLES
        self.n = int(min(max(n_particles, 1000), MAX_PARTICLES))

        # 个体状态
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PARTICLES)
        self.ang = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)
        self.species = ti.field(dtype=ti.i32, shape=MAX_PARTICLES)
        # 蚁群模式：0 = 觅食（离巢），1 = 返程（携带食物）
        self.ant_state = ti.field(dtype=ti.i32, shape=MAX_PARTICLES)
        self.nest = ti.Vector.field(2, dtype=ti.f32, shape=())

        # 两种信息素（双缓冲），外加一张静态食物图
        self.trail_a = ti.field(dtype=ti.f32, shape=(size, size))
        self.trail_b = ti.field(dtype=ti.f32, shape=(size, size))
        self.trail_a_buf = ti.field(dtype=ti.f32, shape=(size, size))
        self.trail_b_buf = ti.field(dtype=ti.f32, shape=(size, size))
        self.food = ti.field(dtype=ti.f32, shape=(size, size))

        # 参数（默认值取自能自发形成静脉状网络的区间）
        self.sense_dist = 30.0
        self.sense_angle = math.radians(45.0)
        self.turn_angle = math.radians(30.0)
        self.speed = 1.0
        self.deposit = 4.0
        self.decay = 0.96
        self.diffuse = 0.12
        self.food_weight = 0.6
        self.dual_species = True
        self.mode = 0                    # 0 = 黏菌，1 = 蚁群
        self.nest[None] = ti.Vector([size * 0.5, size * 0.5])

        self.seed()
        self.clear()

    # ---------- 初始化 ----------
    @ti.kernel
    def _seed_kernel(self, n: ti.i32, size: ti.i32):
        for i in range(n):
            self.pos[i] = ti.Vector([ti.random() * size, ti.random() * size])
            self.ang[i] = ti.random() * 6.2831853
            self.species[i] = i % 2
            self.ant_state[i] = 0

    @ti.kernel
    def _clear_kernel(self):
        for i, j in self.trail_a:
            self.trail_a[i, j] = 0.0
            self.trail_b[i, j] = 0.0
            self.food[i, j] = 0.0

    def seed(self):
        self._seed_kernel(self.n, self.size)

    def clear(self):
        self._clear_kernel()

    def clear_trails(self):
        for f in (self.trail_a, self.trail_b):
            f.fill(0.0)

    def clear_food(self):
        self.food.fill(0.0)

    # ---------- 单步演化 ----------
    @ti.func
    def _sense(self, x: ti.f32, y: ti.f32, sp: ti.i32, size: ti.i32, fw: ti.f32) -> ti.f32:
        """在 (x, y) 处采样气味（含食物加成）。场按 [行=y, 列=x] 存储。"""
        xi = ti.cast(x, ti.i32) % size
        yi = ti.cast(y, ti.i32) % size
        val = 0.0
        if sp == 0:
            val = self.trail_a[yi, xi]
        else:
            val = self.trail_b[yi, xi]
        return val + fw * self.food[yi, xi]

    @ti.kernel
    def step(self, n: ti.i32, size: ti.i32, sense_dist: ti.f32, sense_angle: ti.f32,
             turn_angle: ti.f32, speed: ti.f32, deposit: ti.f32,
             food_weight: ti.f32, dual: ti.i32):
        for i in range(n):
            p = self.pos[i]
            a = self.ang[i]
            sp = self.species[i] if dual == 1 else 0

            f_center = self._sense(p.x + ti.cos(a) * sense_dist,
                                   p.y + ti.sin(a) * sense_dist, sp, size, food_weight)
            f_left = self._sense(p.x + ti.cos(a - sense_angle) * sense_dist,
                                 p.y + ti.sin(a - sense_angle) * sense_dist, sp, size, food_weight)
            f_right = self._sense(p.x + ti.cos(a + sense_angle) * sense_dist,
                                  p.y + ti.sin(a + sense_angle) * sense_dist, sp, size, food_weight)

            if f_center > f_left and f_center > f_right:
                pass                                              # 前方最浓，直行
            elif f_center < f_left and f_center < f_right:
                a += (ti.random() - 0.5) * 2.0 * turn_angle       # 两侧都更浓，随机转
            elif f_left > f_right:
                a += turn_angle                                   # 左前方更浓
            else:
                a -= turn_angle                                   # 右前方更浓

            p.x = (p.x + ti.cos(a) * speed) % size
            p.y = (p.y + ti.sin(a) * speed) % size

            self.pos[i] = p
            self.ang[i] = a

            xi = ti.cast(p.x, ti.i32) % size
            yi = ti.cast(p.y, ti.i32) % size
            if sp == 0:
                ti.atomic_add(self.trail_a[yi, xi], deposit)
            else:
                ti.atomic_add(self.trail_b[yi, xi], deposit)

    @ti.func
    def _sample(self, x: ti.f32, y: ti.f32, size: ti.i32, which: ti.i32) -> ti.f32:
        xi = ti.cast(x, ti.i32) % size
        yi = ti.cast(y, ti.i32) % size
        v = 0.0
        if which == 0:
            v = self.trail_a[yi, xi]      # 食物信息素
        else:
            v = self.trail_b[yi, xi]      # 归巢信息素
        return v

    @ti.kernel
    def ant_step(self, n: ti.i32, size: ti.i32, sense_dist: ti.f32, sense_angle: ti.f32,
                 turn_angle: ti.f32, speed: ti.f32, deposit: ti.f32, food_hit: ti.f32):
        nest = self.nest[None]
        for i in range(n):
            p = self.pos[i]
            a = self.ang[i]
            if self.ant_state[i] == 0:
                # ---- 觅食：跟随食物信息素，留下归巢信息素 ----
                f_c = self._sample(p.x + ti.cos(a) * sense_dist,
                                   p.y + ti.sin(a) * sense_dist, size, 0)
                f_l = self._sample(p.x + ti.cos(a - sense_angle) * sense_dist,
                                   p.y + ti.sin(a - sense_angle) * sense_dist, size, 0)
                f_r = self._sample(p.x + ti.cos(a + sense_angle) * sense_dist,
                                   p.y + ti.sin(a + sense_angle) * sense_dist, size, 0)
                if f_c > 1e-3 or f_l > 1e-3 or f_r > 1e-3:
                    # 闻到信息素：按梯度走
                    if f_c > f_l and f_c > f_r:
                        a += (ti.random() - 0.5) * turn_angle * 0.3
                    elif f_l > f_r:
                        a += turn_angle
                    else:
                        a -= turn_angle
                else:
                    # 场上没有信息素：保持大致直线地扫场，靠这个走出巢穴
                    a += (ti.random() - 0.5) * turn_angle * 0.25

                p.x = (p.x + ti.cos(a) * speed) % size
                p.y = (p.y + ti.sin(a) * speed) % size
                xi = ti.cast(p.x, ti.i32) % size
                yi = ti.cast(p.y, ti.i32) % size
                ti.atomic_add(self.trail_b[yi, xi], deposit * 0.06)

                if self.food[yi, xi] > food_hit:      # 找到食物 -> 返程
                    self.ant_state[i] = 1
                    a += 3.14159265
            else:
                # ---- 返程：朝巢穴走，沿途留下食物信息素 ----
                dx = (nest.x - p.x + size * 0.5) % size - size * 0.5
                dy = (nest.y - p.y + size * 0.5) % size - size * 0.5
                d = ti.atan2(dy, dx) - a
                d = (d + 3.14159265) % 6.2831853 - 3.14159265
                a += d * 0.35 + (ti.random() - 0.5) * turn_angle * 0.3

                p.x = (p.x + ti.cos(a) * speed) % size
                p.y = (p.y + ti.sin(a) * speed) % size
                xi = ti.cast(p.x, ti.i32) % size
                yi = ti.cast(p.y, ti.i32) % size
                ti.atomic_add(self.trail_a[yi, xi], deposit)

                if ti.abs(dx) < 7.0 and ti.abs(dy) < 7.0:   # 回到巢穴 -> 再出发
                    self.ant_state[i] = 0
                    a = ti.random() * 6.2831853            # 随机方向离巢
                    p.x = (nest.x + ti.cos(a) * 8.0) % size
                    p.y = (nest.y + ti.sin(a) * 8.0) % size

            self.pos[i] = p
            self.ang[i] = a

    @ti.kernel
    def diffuse_decay(self, size: ti.i32, decay: ti.f32, diffuse: ti.f32):
        for i, j in self.trail_a:
            ip = (i + 1) % size
            im = (i - 1) % size
            jp = (j + 1) % size
            jm = (j - 1) % size
            for ch in ti.static(range(2)):
                if ch == 0:
                    center = self.trail_a[i, j]
                    near = (self.trail_a[im, j] + self.trail_a[ip, j] +
                            self.trail_a[i, jm] + self.trail_a[i, jp]) * 0.25
                    self.trail_a_buf[i, j] = (center * (1.0 - diffuse) + near * diffuse) * decay
                else:
                    center = self.trail_b[i, j]
                    near = (self.trail_b[im, j] + self.trail_b[ip, j] +
                            self.trail_b[i, jm] + self.trail_b[i, jp]) * 0.25
                    self.trail_b_buf[i, j] = (center * (1.0 - diffuse) + near * diffuse) * decay

    @ti.kernel
    def _swap_kernel(self):
        for i, j in self.trail_a:
            self.trail_a[i, j] = self.trail_a_buf[i, j]
            self.trail_b[i, j] = self.trail_b_buf[i, j]

    @ti.kernel
    def _add_food_kernel(self, cx: ti.f32, cy: ti.f32, radius: ti.f32,
                         amount: ti.f32, size: ti.i32):
        r = ti.cast(radius, ti.i32) + 1
        cx_i = ti.cast(cx, ti.i32)
        cy_i = ti.cast(cy, ti.i32)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                d2 = di * di + dj * dj
                if d2 <= r * r:
                    xi = (cx_i + di) % size
                    yi = (cy_i + dj) % size
                    falloff = 1.0 - ti.sqrt(ti.cast(d2, ti.f32)) / (radius + 1.0)
                    v = self.food[yi, xi] + amount * ti.max(falloff, 0.0)
                    self.food[yi, xi] = ti.max(v, 0.0)

    @ti.kernel
    def _total_kernel(self) -> ti.f32:
        s = 0.0
        for i, j in self.trail_a:
            s += self.trail_a[i, j] + self.trail_b[i, j]
        return s

    def update(self):
        if self.mode == 0:
            self.step(self.n, self.size, self.sense_dist, self.sense_angle,
                      self.turn_angle, self.speed, self.deposit,
                      self.food_weight, 1 if self.dual_species else 0)
            self.diffuse_decay(self.size, self.decay, self.diffuse)
        else:
            # 蚁群的信息素要留得久一些，否则路径来不及被强化
            self.ant_step(self.n, self.size, self.sense_dist, self.sense_angle,
                          self.turn_angle, max(0.2, self.speed), max(0.3, self.deposit * 0.6),
                          0.4)
            ant_decay = 1.0 - (1.0 - self.decay) * 0.12
            self.diffuse_decay(self.size, min(0.9995, ant_decay), self.diffuse * 0.5)
        self._swap_kernel()

    def set_mode(self, mode):
        self.mode = int(mode)
        self.clear_trails()
        self.seed()

    def set_nest(self, x, y):
        self.nest[None] = ti.Vector([float(x) % self.size, float(y) % self.size])

    def get_nest(self):
        n = self.nest[None]
        return float(n[0]), float(n[1])

    def add_food(self, cx, cy, radius, amount=6.0):
        self._add_food_kernel(float(cx) % self.size, float(cy) % self.size,
                              float(radius), float(amount), self.size)

    def total_mass(self):
        return float(self._total_kernel())

    # ---------- 显示 ----------
    def get_rgb(self, gain=1.0):
        a = self.trail_a.to_numpy()
        b = self.trail_b.to_numpy()
        food = self.food.to_numpy()

        # 分位数归一 + gamma：既不会过曝，也能保住静脉状结构的对比度
        # 分位数在子采样上算，省掉一大半时间
        if self.mode == 1:
            # 蚁群：两种信息素量级差很多，各归一各的
            sa = 1.0 / (float(np.percentile(a[::4, ::4], 99.5)) + 1e-4) * gain
            sb = 1.0 / (float(np.percentile(b[::4, ::4], 99.5)) + 1e-4) * gain
            cool = 0.45
        else:
            sa = sb = 1.0 / (float(np.percentile((a + b)[::4, ::4], 99.0)) + 1e-4) * gain
            cool = 1.0
        a = np.sqrt(np.clip(a * sa, 0.0, 1.0))
        b = np.sqrt(np.clip(b * sb, 0.0, 1.0))

        img = np.zeros((self.size, self.size, 3), dtype=np.float32)
        img[..., 0] += a * 255.0
        img[..., 1] += a * 165.0
        img[..., 2] += a * 50.0
        img[..., 0] += b * 60.0 * cool
        img[..., 1] += b * 185.0 * cool
        img[..., 2] += b * 255.0 * cool
        img += np.clip(food, 0.0, 1.0)[..., None] * np.array([12.0, 22.0, 20.0])

        if self.mode == 1:      # 蚁群：标出巢穴
            nx, ny = self.get_nest()
            yy, xx = np.ogrid[:self.size, :self.size]
            dist = np.sqrt((xx - nx) ** 2 + (yy - ny) ** 2)
            ring = (dist > 5.0) & (dist < 8.0)
            img[ring] = [255.0, 245.0, 205.0]

        return np.clip(img, 0, 255).astype(np.uint8)


# ================== 画布 ==================
class StigmergyCanvas(QWidget):
    def __init__(self, sim: StigmergySimulation, parent=None):
        super().__init__(parent)
        self.sim = sim
        self._image = QImage(sim.size, sim.size, QImage.Format_RGB888)
        self._image.fill(Qt.black)
        self.brush_radius = 18.0
        self.brush_amount = 6.0
        self._drawing = False
        self._erase = False
        self.setMinimumSize(320, 320)

    def set_sim(self, sim):
        self.sim = sim
        self._image = QImage(sim.size, sim.size, QImage.Format_RGB888)
        self.update_display()

    def update_display(self):
        rgb = self.sim.get_rgb()
        self._image = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                             rgb.shape[1] * 3, QImage.Format_RGB888).copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 12))
        # 直接铺满画布：略有一点非等比拉伸，但不会出现上下/左右黑边
        painter.drawImage(self.rect(), self._image)

    def _to_sim(self, pos):
        w = max(1, self.width())
        h = max(1, self.height())
        return pos.x() / w * self.sim.size, pos.y() / h * self.sim.size

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and (event.modifiers() & Qt.ShiftModifier):
            x, y = self._to_sim(event.position())
            self.sim.set_nest(x, y)
            self.update_display()
            return
        self._drawing = True
        self._erase = event.button() == Qt.RightButton
        self._paint_food(event.position())

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._paint_food(event.position())

    def mouseReleaseEvent(self, event):
        self._drawing = False
        self.update_display()

    def _paint_food(self, pos):
        x, y = self._to_sim(pos)
        amount = -40.0 if self._erase else self.brush_amount
        self.sim.add_food(x, y, self.brush_radius, amount)
        self.update_display()


# ================== 主窗口 ==================
class StigmergyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sim = StigmergySimulation(FIELD_SIZE, 30000)
        self.running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_tick)
        self.fps_counter = FPSCounter()
        self.init_ui()
        self.canvas.update_display()

    # ---------- UI ----------
    def init_ui(self):
        self.setWindowTitle("Stigmergy 黏菌 / 蚁群网络")
        # 面板 300 + 左右边距 12 + 间距 8；高度只留上下边距，让画布尽量是正方形
        self.setGeometry(100, 100, 320 + 640, 12 + 640)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(300)
        panel_layout = QVBoxLayout()
        panel.setLayout(panel_layout)
        panel_layout.setSpacing(6)

        panel_layout.addWidget(self._create_control_group())
        panel_layout.addWidget(self._create_param_group())
        panel_layout.addWidget(self._create_display_group())
        panel_layout.addWidget(self._create_stats_group())
        panel_layout.addStretch()

        self.canvas = StigmergyCanvas(self.sim)
        main_layout.addWidget(panel)
        main_layout.addWidget(self.canvas, 1)

    def _create_control_group(self):
        group = QGroupBox("控制")
        layout = QHBoxLayout()
        self.start_btn = QPushButton("开始")
        step_btn = QPushButton("单步")
        clear_btn = QPushButton("清空痕迹")
        reset_btn = QPushButton("重置")
        self.start_btn.clicked.connect(self.toggle_running)
        step_btn.clicked.connect(self.single_step)
        clear_btn.clicked.connect(self.clear_trails)
        reset_btn.clicked.connect(self.reset_simulation)
        for b in (self.start_btn, step_btn, clear_btn, reset_btn):
            layout.addWidget(b)
        group.setLayout(layout)
        return group

    def _add_slider(self, form, label, lo, hi, value, fmt, callback):
        """一行滑条 + 数值标签，具体外观交给公共的 LabeledSlider。"""
        slider = LabeledSlider(label, lo, hi, value, formatter=fmt, value_width=52)
        slider.bind(callback)          # 连接回调，并立刻同步一次初始值
        form.addRow(slider)
        return slider

    def _create_param_group(self):
        group = QGroupBox("参数")
        form = QFormLayout()
        form.setSpacing(4)

        pct = lambda v: f"{v / 100:.2f}"
        mil = lambda v: f"{v / 100:.3f}"
        deg = lambda v: f"{v}°"

        self._add_slider(form, "感知距离", 1, 60, 30, str, self._set_sense_dist)
        self._add_slider(form, "感知角度", 5, 90, 45, deg, self._set_sense_angle)
        self._add_slider(form, "转向角", 5, 90, 30, deg, self._set_turn_angle)
        self._add_slider(form, "移动速度", 20, 600, 100, pct, self._set_speed)
        self._add_slider(form, "沉积量", 20, 2000, 400, pct, self._set_deposit)
        self._add_slider(form, "衰减", 70, 99, 96, mil, self._set_decay)
        self._add_slider(form, "扩散", 0, 100, 12, pct, self._set_diffuse)
        self._add_slider(form, "食物权重", 0, 500, 60, pct, self._set_food_weight)

        self.dual_check = QCheckBox("双色（两种信息素）")
        self.dual_check.setChecked(True)
        self.dual_check.toggled.connect(self._set_dual)
        form.addRow("", self.dual_check)

        group.setLayout(form)
        return group

    def _create_display_group(self):
        group = QGroupBox("显示 / 交互")
        layout = QVBoxLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["黏菌 Physarum", "蚁群 Ant colony"])
        self.mode_combo.currentIndexChanged.connect(self._set_mode)
        row0 = QWidget()
        h0 = QHBoxLayout(row0)
        h0.setContentsMargins(0, 0, 0, 0)
        h0.addWidget(QLabel("模式:"))
        h0.addWidget(self.mode_combo, 1)
        layout.addWidget(row0)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 8)
        self.steps_spin.setValue(1)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("每帧步数:"))
        h.addWidget(self.steps_spin)
        layout.addWidget(row)

        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(2, 80)
        self.brush_spin.setValue(18)
        self.brush_spin.valueChanged.connect(lambda v: setattr(self.canvas, "brush_radius", float(v)))
        row2 = QWidget()
        h2 = QHBoxLayout(row2)
        h2.setContentsMargins(0, 0, 0, 0)
        h2.addWidget(QLabel("画笔(食物):"))
        h2.addWidget(self.brush_spin)
        layout.addWidget(row2)

        tip = QLabel("左键拖拽放食物，右键擦除\n蚁群模式：Shift+左键设置巢穴")
        tip.setStyleSheet("color:#888;")
        layout.addWidget(tip)
        group.setLayout(layout)
        return group

    def _set_mode(self, idx):
        self.sim.set_mode(idx)
        self.canvas.update_display()
        self.lbl_mass.setText("0")

    def _create_stats_group(self):
        group = QGroupBox("统计")
        layout = QFormLayout()
        self.lbl_particles = QLabel(str(self.sim.n))
        self.lbl_mass = QLabel("0")
        self.lbl_fps = QLabel("0")
        layout.addRow("个体数:", self.lbl_particles)
        layout.addRow("信息素总量:", self.lbl_mass)
        layout.addRow("帧率:", self.lbl_fps)
        group.setLayout(layout)
        return group

    # ---------- 参数回调 ----------
    def _set_sense_dist(self, v):
        self.sim.sense_dist = float(max(1, v))

    def _set_sense_angle(self, v):
        self.sim.sense_angle = math.radians(float(max(1, v)))

    def _set_turn_angle(self, v):
        self.sim.turn_angle = math.radians(float(max(1, v)))

    def _set_speed(self, v):
        self.sim.speed = max(0.05, v / 100.0)

    def _set_deposit(self, v):
        self.sim.deposit = max(0.05, v / 100.0)

    def _set_decay(self, v):
        self.sim.decay = min(0.999, max(0.5, v / 100.0))

    def _set_diffuse(self, v):
        self.sim.diffuse = min(1.0, max(0.0, v / 100.0))

    def _set_food_weight(self, v):
        self.sim.food_weight = max(0.0, v / 100.0)

    def _set_dual(self, checked):
        self.sim.dual_species = bool(checked)

    # ---------- 运行 ----------
    def toggle_running(self):
        if self.running:
            self.timer.stop()
            self.start_btn.setText("开始")
            self.running = False
        else:
            self.timer.start(30)
            self.start_btn.setText("暂停")
            self.running = True

    def on_tick(self):
        for _ in range(self.steps_spin.value()):
            self.sim.update()
        self.canvas.update_display()
        fps = self.fps_counter.tick()
        if fps is not None:
            self.lbl_fps.setText(f"{fps:.1f}")
            self.lbl_mass.setText(f"{self.sim.total_mass():.3e}")

    def single_step(self):
        self.sim.update()
        self.canvas.update_display()
        self.lbl_mass.setText(f"{self.sim.total_mass():.3e}")

    def clear_trails(self):
        self.sim.clear_trails()
        self.canvas.update_display()

    def reset_simulation(self):
        was_running = self.running
        if self.running:
            self.timer.stop()
            self.running = False
        self.sim.clear()
        self.sim.seed()
        self.canvas.update_display()
        self.lbl_mass.setText("0")
        if was_running:
            self.toggle_running()


if __name__ == "__main__":
    raise SystemExit(run_app(StigmergyApp))
