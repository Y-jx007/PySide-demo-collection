from custom_import import *
from PySide6.QtGui import QImage

SQ3 = math.sqrt(3.0)

# 六角格 6 个邻居方向（行偏移, 列偏移）
HEX_DIRS = ((0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0))

PRESETS = {
    "枝晶 Dendrite": (0.90, 0.40, 0.0008),
    "羽毛 Feather": (0.45, 0.38, 0.0003),
    "宽枝 Wide": (0.60, 0.42, 0.0015),
    "密实 Dense": (0.70, 0.50, 0.0040),
    "板状 Plate": (1.00, 0.45, 0.0000),
}


class SnowCrystal:
    """Reiter 雪花模型：每格一个含水量 s，冻结(s>=1) 与边界格为"可接收"区。

    s 拆成参与扩散的 u 和只能累积的 v：
        u = 0, v = s            （可接收格）
        u = s, v = 0            （其余格）
    每步  v += γ（仅可接收格）、u += α/2·(⟨u⟩ - u)、边缘格 u = β，
    最后 s = u + v。可接收格不断吸水直到 s>=1 冻结，扩散供水的快慢决定
    是长成枝晶还是长成实心六边形。
    """

    def __init__(self, radius=110, alpha=0.9, beta=0.40, gamma=0.0008, cluster=False):
        self.radius = radius
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n = 2 * radius + 1
        self.steps = 0

        n = self.n
        rr, cc = np.mgrid[0:n, 0:n]
        q = cc - radius
        r = rr - radius
        self.mask = (np.abs(q) <= radius) & (np.abs(r) <= radius) & (np.abs(q + r) <= radius)
        self.edge = self.mask & (
            (np.abs(q) >= radius - 1) | (np.abs(r) >= radius - 1) | (np.abs(q + r) >= radius - 1)
        )

        ones = self.mask.astype(np.float32)
        self.nb_cnt = np.maximum(self._shift_sum(ones), 1.0)

        self.s = np.where(self.mask, beta, 0.0).astype(np.float32)
        centre = radius * n + radius
        if cluster:
            d = np.sqrt(q.astype(np.float32) ** 2 + r.astype(np.float32) ** 2
                        + (q + r).astype(np.float32) ** 2)
            self.s[(d <= 2.5) & self.mask] = 1.0
        else:
            self.s.flat[centre] = 1.0

    # ---------- 计算 ----------
    def _shift_sum(self, a):
        """六邻域求和（越界部分为 0）。"""
        acc = np.zeros_like(a)
        n = self.n
        for dr, dc in HEX_DIRS:
            r0 = max(0, -dr)
            r1 = n - max(0, dr)
            c0 = max(0, -dc)
            c1 = n - max(0, dc)
            acc[r0:r1, c0:c1] += a[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        return acc

    @property
    def frozen(self):
        return self.s >= 1.0

    @property
    def ice_count(self):
        return int((self.s >= 1.0).sum())

    @property
    def extent(self):
        """已生长半径占可用半径的比例。"""
        ys, xs = np.nonzero(self.frozen)
        if ys.size == 0:
            return 0.0
        return float(np.max(np.hypot(xs - self.radius, ys - self.radius)) / self.radius)

    def step(self, n=1):
        for _ in range(n):
            frozen = self.s >= 1.0
            near_frozen = self._shift_sum(frozen.astype(np.float32)) > 0.5
            receptive = frozen | near_frozen

            u = np.where(receptive, 0.0, self.s).astype(np.float32)
            v = np.where(receptive, self.s, 0.0).astype(np.float32)

            u = u + 0.5 * self.alpha * (self._shift_sum(u) / self.nb_cnt - u)
            u *= self.mask          # 六角域外不参与扩散，否则会被 1 格邻域放大
            v = v + np.where(receptive, self.gamma, 0.0)

            u[self.edge] = self.beta
            v[self.edge] = 0.0

            self.s = np.clip(u + v, 0.0, 4.0)
            self.steps += 1


class SnowCanvas(QWidget):
    """把六角格映射到像素：每个像素查表得到格子，再套一层调色板。"""

    def __init__(self, crystal, main_window=None):
        super().__init__()
        self.crystal = crystal
        self.main_window = main_window
        self.ice_color = QColor(70, 130, 200)
        self.bg_color = QColor(252, 252, 252)
        self.show_vapor = True
        self._map = None
        self._image = None
        self._palette = None
        self._dirty = True
        self.setMinimumSize(420, 420)

    def set_crystal(self, crystal):
        self.crystal = crystal
        self._map = None
        self._dirty = True
        self.update()

    def invalidate(self):
        self._dirty = True
        self.update()

    # ---------- 映射与调色板 ----------
    def _build_map(self):
        n = self.crystal.n
        R = self.crystal.radius
        w, h = max(1, self.width()), max(1, self.height())
        side = min(w, h)
        scale = (side / 2.0 - 4) / R
        xs = (np.arange(w, dtype=np.float32) - w / 2.0 + 0.5) / scale
        ys = (np.arange(h, dtype=np.float32) - h / 2.0 + 0.5) / scale
        r2 = ys[:, None] / (SQ3 / 2.0)
        q2 = xs[None, :] - r2 / 2.0
        ri = np.rint(r2).astype(np.int32) + R
        qi = np.rint(q2).astype(np.int32) + R
        inside = (ri >= 0) & (ri < n) & (qi >= 0) & (qi < n)
        ri = np.clip(ri, 0, n - 1)
        qi = np.clip(qi, 0, n - 1)
        idx = np.where(inside & self.crystal.mask[ri, qi], ri * n + qi, -1)
        self._map = idx
        self._image = None

    def _build_palette(self):
        # 0 号是背景，1..VAPOR_LEVELS 是水汽浓度，最后一个是冰
        levels = 24
        bg = np.array([self.bg_color.red(), self.bg_color.green(), self.bg_color.blue()], np.float32)
        ice = np.array([self.ice_color.red(), self.ice_color.green(), self.ice_color.blue()], np.float32)
        pal = np.empty((levels + 2, 3), dtype=np.uint8)
        pal[0] = bg
        for i in range(levels):
            t = (i + 1) / levels
            pal[1 + i] = (bg * (1 - 0.35 * t) + ice * (0.35 * t)).astype(np.uint8)
        pal[-1] = ice
        self._palette = pal

    def _render_image(self):
        if self._map is None:
            self._build_map()
        if self._palette is None:
            self._build_palette()

        crystal = self.crystal
        frozen = crystal.frozen
        levels = self._palette.shape[0] - 2
        if self.show_vapor:
            t = np.clip(crystal.s, 0.0, 1.0)
            code = 1 + np.minimum((t * levels).astype(np.uint8), levels - 1)
            code *= crystal.mask.astype(np.uint8)
        else:
            code = np.zeros((crystal.n, crystal.n), dtype=np.uint8)
        code[frozen] = self._palette.shape[0] - 1
        flat = np.concatenate([code.ravel(), np.array([0], dtype=np.uint8)])

        idx = self._map
        idx = np.where(idx < 0, crystal.n * crystal.n, idx)
        rgb = self._palette[flat[idx]]
        h, w = idx.shape
        self._image = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._dirty = False

    def paintEvent(self, event):
        if self._dirty or self._image is None:
            self._render_image()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.fillRect(self.rect(), self.bg_color)
        painter.drawImage(self.rect(), self._image)
        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._map = None
        self._dirty = True


class SnowApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("雪花生长 Reiter Snowflake")
        self.setGeometry(140, 90, 1080, 760)

        self.radius = 110
        self.alpha, self.beta, self.gamma = PRESETS["枝晶 Dendrite"]
        self.speed = 8
        self.auto_stop = True

        self.crystal = SnowCrystal(self.radius, self.alpha, self.beta, self.gamma)
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)

        self.update_stats()

    # ---------- UI ----------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(270)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(6)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = SnowCanvas(self.crystal, self)

        self._build_preset_group(panel_layout)
        self._build_param_group(panel_layout)
        self._build_run_group(panel_layout)
        self._build_display_group(panel_layout)
        panel_layout.addStretch()

        root.addWidget(panel)
        root.addWidget(self.canvas, 1)

        self.lbl_status = QLabel("就绪")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_ice = QLabel("冰格 0")
        self.lbl_steps = QLabel("步数 0")
        self.statusBar().addPermanentWidget(self.lbl_ice)
        self.statusBar().addPermanentWidget(self.lbl_steps)

    def _build_preset_group(self, layout):
        group = QGroupBox("形态预设")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        vbox.addWidget(self.preset_combo)

        row = QHBoxLayout()
        row.setSpacing(4)
        btn_reset = QPushButton("重新生长")
        btn_reset.clicked.connect(self.reset)
        row.addWidget(btn_reset, 1)
        vbox.addLayout(row)
        layout.addWidget(group)

    def _build_param_group(self, layout):
        group = QGroupBox("参数")
        form = QFormLayout(group)
        form.setSpacing(4)
        form.setContentsMargins(6, 6, 6, 6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.alpha_slider, self.alpha_label = self._add_slider(form, "扩散 α", 10, 100, int(self.alpha * 100),
                                                               lambda v: v / 100.0)
        self.beta_slider, self.beta_label = self._add_slider(form, "湿度 β", 10, 90, int(self.beta * 100),
                                                             lambda v: v / 100.0)
        self.gamma_slider, self.gamma_label = self._add_slider(form, "沉积 γ", 0, 100, int(self.gamma * 10000),
                                                               lambda v: v / 10000.0)
        layout.addWidget(group)

    def _add_slider(self, form, text, lo, hi, value, transform):
        row = QHBoxLayout()
        row.setSpacing(4)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        label = QLabel(f"{transform(value):.4g}")
        label.setFixedWidth(44)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(slider, 1)
        row.addWidget(label)

        def on_change(v):
            label.setText(f"{transform(v):.4g}")
            self.apply_params()

        slider.valueChanged.connect(on_change)
        form.addRow(text, row)
        return slider, label

    def _build_run_group(self, layout):
        group = QGroupBox("生长")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.run_btn = QPushButton("开始")
        self.run_btn.clicked.connect(self.toggle_run)
        row.addWidget(self.run_btn, 1)
        step_btn = QPushButton("单步")
        step_btn.clicked.connect(self.single_step)
        row.addWidget(step_btn, 1)
        vbox.addLayout(row)

        form = QFormLayout()
        form.setSpacing(4)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 60)
        self.speed_slider.setValue(self.speed)
        self.speed_slider.valueChanged.connect(self.on_speed)
        self.speed_label = QLabel(f"{self.speed} 步/帧")
        self.speed_label.setFixedWidth(56)
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        row2.addWidget(self.speed_slider, 1)
        row2.addWidget(self.speed_label)
        form.addRow("速度", row2)

        self.auto_check = QCheckBox("长满自动停")
        self.auto_check.setChecked(self.auto_stop)
        self.auto_check.toggled.connect(lambda c: setattr(self, "auto_stop", bool(c)))
        form.addRow("", self.auto_check)
        vbox.addLayout(form)
        layout.addWidget(group)

    def _build_display_group(self, layout):
        group = QGroupBox("显示")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)

        self.vapor_check = QCheckBox("显示水汽浓度")
        self.vapor_check.setChecked(True)
        self.vapor_check.toggled.connect(self.toggle_vapor)
        vbox.addWidget(self.vapor_check)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("冰色"))
        self.ice_btn = ColorButton(self.canvas.ice_color)
        self.ice_btn.colorChanged.connect(self.on_ice_color)
        row.addWidget(self.ice_btn)
        row.addWidget(QLabel("底色"))
        self.bg_btn = ColorButton(self.canvas.bg_color)
        self.bg_btn.colorChanged.connect(self.on_bg_color)
        row.addWidget(self.bg_btn)
        row.addStretch()
        vbox.addLayout(row)
        layout.addWidget(group)

    # ---------- 逻辑 ----------
    def apply_preset(self, name):
        if name not in PRESETS:
            return
        self.alpha, self.beta, self.gamma = PRESETS[name]
        for slider, value in ((self.alpha_slider, int(self.alpha * 100)),
                              (self.beta_slider, int(self.beta * 100)),
                              (self.gamma_slider, int(self.gamma * 10000))):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        self.alpha_label.setText(f"{self.alpha:.4g}")
        self.beta_label.setText(f"{self.beta:.4g}")
        self.gamma_label.setText(f"{self.gamma:.4g}")
        self.reset()

    def apply_params(self):
        self.alpha = self.alpha_slider.value() / 100.0
        self.beta = self.beta_slider.value() / 100.0
        self.gamma = self.gamma_slider.value() / 10000.0
        if self.crystal is not None:
            self.crystal.alpha = self.alpha
            self.crystal.beta = self.beta
            self.crystal.gamma = self.gamma

    def reset(self):
        self.timer.stop()
        self.run_btn.setText("开始")
        self.crystal = SnowCrystal(self.radius, self.alpha, self.beta, self.gamma)
        self.canvas.set_crystal(self.crystal)
        self.update_stats()
        self.lbl_status.setText("已重置")

    def toggle_run(self):
        if self.timer.isActive():
            self.timer.stop()
            self.run_btn.setText("开始")
            self.lbl_status.setText("已暂停")
        else:
            self.timer.start(33)
            self.run_btn.setText("暂停")
            self.lbl_status.setText("生长中…")

    def single_step(self):
        self.crystal.step(8)
        self.canvas.invalidate()
        self.update_stats()

    def tick(self):
        self.crystal.step(self.speed)
        self.canvas.invalidate()
        self.update_stats()
        if self.auto_stop and self.crystal.extent > 0.86:
            self.timer.stop()
            self.run_btn.setText("开始")
            self.lbl_status.setText("已长满，自动停止")
            if hasattr(self, "on_growth_done"):
                self.on_growth_done()

    def on_speed(self, value):
        self.speed = value
        self.speed_label.setText(f"{value} 步/帧")

    def toggle_vapor(self, checked):
        self.canvas.show_vapor = bool(checked)
        self.canvas.invalidate()

    def on_ice_color(self, color):
        self.canvas.ice_color = color
        self.canvas._palette = None
        self.canvas.invalidate()

    def on_bg_color(self, color):
        self.canvas.bg_color = color
        self.canvas._palette = None
        self.canvas.invalidate()

    def update_stats(self):
        self.lbl_ice.setText(f"冰格 {self.crystal.ice_count:,}")
        self.lbl_steps.setText(f"步数 {self.crystal.steps:,}")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SnowApp()
    window.show()
    sys.exit(app.exec())
