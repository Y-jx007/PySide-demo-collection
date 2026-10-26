from common.qt_base import (ColorButton, QApplication, QCheckBox, QColor, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QImage, QLabel, QMainWindow, QPainter, QPushButton, QSlider, QSpinBox, QTimer, QVBoxLayout, QWidget, Qt, math, sys)
from common.numpy_base import (np)
from PySide6.QtGui import QImage

SQ3 = math.sqrt(3.0)

# 六角格 6 个邻居方向（行偏移=Δr, 列偏移=Δq）
HEX_DIRS = ((0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0))

PRESETS = {
    "枝晶 Dendrite": (0.90, 0.40, 0.0008),
    "羽毛 Feather": (0.45, 0.38, 0.0003),
    "宽枝 Wide": (0.60, 0.42, 0.0015),
    "密实 Dense": (0.70, 0.50, 0.0040),
    "板状 Plate": (1.00, 0.45, 0.0000),
}


def _blur3(a):
    """3×3 盒式模糊（软边用）。"""
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


class SnowCrystal:
    """Reiter 雪花模型，方形域 + 六角格。

    每格含水量 s 按类别拆成参与扩散的 u 与只累积的 v：
        u = 0, v = s                （可接收格：已冻结或与冻结格相邻）
        u = s, v = 0                （其余格）
    每步 v += γ（仅可接收格）、u += α/2·(⟨u⟩−u)、方形边界格 u = β，
    s = u + v，s ≥ 1 即冻结。扩散供水的快慢决定长成枝晶还是实心六边形。
    """

    def __init__(self, size=420, alpha=0.9, beta=0.40, gamma=0.0008, cluster=False):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.steps = 0

        # 方形域半边长（格），r 是行、q 是列
        S = size / 2.0
        rr_max = int(S / (SQ3 / 2.0)) + 1
        qq_max = int(S + rr_max * 0.5) + 2
        rows = np.arange(-rr_max, rr_max + 1)
        cols = np.arange(-qq_max - 1, qq_max + 2)
        R, Q = np.meshgrid(rows, cols, indexing="ij")
        X = Q + R * 0.5
        Y = R * (SQ3 / 2.0)
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.mask = (np.abs(X) <= S) & (np.abs(Y) <= S)
        self.nrows, self.ncols = self.mask.shape

        # 方形边界（离边界 1 格以内）作为水汽入口
        self.edge = self.mask & ((np.abs(X) >= S - 1.5) | (np.abs(Y) >= S - 1.5))

        self.nb_cnt = np.maximum(self._shift_sum(self.mask.astype(np.float32)), 1.0)
        self._inv_cnt = (1.0 / self.nb_cnt).astype(np.float32)
        self._mask_f = self.mask.astype(np.float32)

        self.s = np.where(self.mask, beta, 0.0).astype(np.float32)
        centre = (self.nrows // 2, self.ncols // 2)
        if cluster:
            d = np.sqrt(X.astype(np.float32) ** 2 + Y.astype(np.float32) ** 2)
            self.s[(d <= 2.5) & self.mask] = 1.0
        else:
            self.s[centre] = 1.0

    # ---------- 计算 ----------
    def _shift_sum(self, a):
        acc = np.zeros_like(a)
        nr, nc = a.shape
        for dr, dc in HEX_DIRS:
            r0 = max(0, -dr)
            r1 = nr - max(0, dr)
            c0 = max(0, -dc)
            c1 = nc - max(0, dc)
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
        """已生长半径 / 可用半径。"""
        ys, xs = np.nonzero(self.frozen)
        if ys.size == 0:
            return 0.0
        d = np.hypot(self.X[ys, xs], self.Y[ys, xs])
        return float(d.max() / (self.size / 2.0))

    MARGIN = 22          # 只在晶体周围的窗口内演化，窗口外恒为 β

    def step(self, n=1):
        s = self.s
        nr, nc = s.shape
        gamma = self.gamma
        half_alpha = 0.5 * self.alpha
        beta = self.beta
        for _ in range(n):
            frozen = s >= 1.0
            rows = np.flatnonzero(frozen.any(axis=1))
            cols = np.flatnonzero(frozen.any(axis=0))
            if rows.size:
                r0 = max(0, rows[0] - self.MARGIN)
                r1 = min(nr, rows[-1] + self.MARGIN + 1)
                c0 = max(0, cols[0] - self.MARGIN)
                c1 = min(nc, cols[-1] + self.MARGIN + 1)
            else:
                r0, r1, c0, c1 = 0, nr, 0, nc

            # 外扩 1 格做缓冲，保证写回区域内的邻域和是完整的
            rr0, cc0 = max(0, r0 - 1), max(0, c0 - 1)
            rr1, cc1 = min(nr, r1 + 1), min(nc, c1 + 1)
            sub = s[rr0:rr1, cc0:cc1]

            fz = sub >= 1.0
            near = self._shift_sum(fz.astype(np.float32))
            rmask = (fz | (near > 0.5)).astype(np.float32)
            u = sub * (1.0 - rmask)
            u = u + half_alpha * (self._shift_sum(u) * self._inv_cnt[rr0:rr1, cc0:cc1] - u)
            u *= self._mask_f[rr0:rr1, cc0:cc1]

            new = u + (sub + gamma) * rmask
            new[self.edge[rr0:rr1, cc0:cc1]] = beta
            np.clip(new, 0.0, 4.0, out=new)

            s[r0:r1, c0:c1] = new[r0 - rr0:r0 - rr0 + (r1 - r0),
                                  c0 - cc0:c0 - cc0 + (c1 - c0)]
            self.steps += 1


class SnowCanvas(QWidget):
    """像素→格子查表 + 冰晶着色（软边、假光照、辉光）。"""

    def __init__(self, crystal, main_window=None):
        super().__init__()
        self.crystal = crystal
        self.main_window = main_window
        self.ice_color = QColor(150, 205, 255)
        self.bg_color = QColor(238, 244, 250)
        self.vapor_color = QColor(96, 132, 190)
        self.glow_color = QColor(120, 180, 255)
        self.show_vapor = True
        self._map = None
        self._image = None
        self._lut = None
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

    # ---------- 像素映射 ----------
    RENDER_CAP = 640

    def _build_map(self):
        c = self.crystal
        w, h = max(1, self.width()), max(1, self.height())
        cap = self.RENDER_CAP
        if max(w, h) > cap:
            f = cap / float(max(w, h))
            w, h = max(1, int(w * f)), max(1, int(h * f))
        side = min(w, h)
        scale = (side / 2.0 - 4) / (c.size / 2.0)
        xs = (np.arange(w, dtype=np.float32) - w / 2.0 + 0.5) / scale
        ys = (np.arange(h, dtype=np.float32) - h / 2.0 + 0.5) / scale
        r2 = ys[:, None] / (SQ3 / 2.0)
        q2 = xs[None, :] - r2 / 2.0
        ri = np.rint(r2).astype(np.int32) + c.nrows // 2
        ci = np.rint(q2).astype(np.int32) + c.ncols // 2
        inside = (ri >= 0) & (ri < c.nrows) & (ci >= 0) & (ci < c.ncols)
        ri = np.clip(ri, 0, c.nrows - 1)
        ci = np.clip(ci, 0, c.ncols - 1)
        idx = np.where(inside & c.mask[ri, ci], ri * c.ncols + ci, c.mask.size)
        self._map = idx.astype(np.int32)

    # ---------- 着色 ----------
    def _build_lut(self):
        """256 级渐变：底色 → 水汽 → 冰 → 高光。"""
        stops = [
            (0.00, np.array([self.bg_color.red(), self.bg_color.green(), self.bg_color.blue()], np.float32)),
            (0.30, np.array([self.vapor_color.red(), self.vapor_color.green(), self.vapor_color.blue()], np.float32) * 0.55
             + np.array([self.bg_color.red(), self.bg_color.green(), self.bg_color.blue()], np.float32) * 0.45),
            (0.62, np.array([self.ice_color.red(), self.ice_color.green(), self.ice_color.blue()], np.float32) * 0.82),
            (0.88, np.array([self.ice_color.red(), self.ice_color.green(), self.ice_color.blue()], np.float32)),
            (1.00, np.array([252, 253, 255], np.float32)),
        ]
        xs = np.linspace(0.0, 1.0, 256)
        lut = np.empty((256, 3), dtype=np.uint8)
        for ch in range(3):
            xs_stop = [s[0] for s in stops]
            ys_stop = [s[1][ch] for s in stops]
            lut[:, ch] = np.interp(xs, xs_stop, ys_stop)
        self._lut = lut

    def _render_image(self):
        c = self.crystal
        if self._map is None:
            self._build_map()
        if self._lut is None:
            self._build_lut()

        frozen = (c.s >= 1.0).astype(np.float32)
        soft = _blur3(frozen)                  # 冰边缘：一次模糊，保留细枝
        glow = _blur3(_blur3(soft))            # 外层辉光
        thick = np.clip(c.s - 1.0, 0.0, 1.2) * (1.0 / 1.2)

        # 软边梯度 → 假光照（光来自左上）
        gy, gx = np.gradient(soft)
        light = np.clip(1.0 - 1.5 * gx + 1.5 * gy, 0.15, 1.9)

        if self.show_vapor:
            val = np.clip(c.s, 0.0, 1.0) * 0.30 * (1.0 - soft)
        else:
            val = np.zeros_like(soft)
        val += soft * 0.74
        val += soft * (light - 1.0) * 0.22 + soft * thick * 0.18
        val += glow * (1.0 - soft) * 0.22
        val = np.clip(val, 0.0, 1.0)

        code = (val * 255).astype(np.uint8)
        cells = self._lut[code.ravel()]
        cells = np.vstack([cells, self._lut[0][None, :]])

        out = cells[self._map]
        h, w = self._map.shape
        self._image = QImage(out.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._dirty = False

    def paintEvent(self, event):
        if self._dirty or self._image is None:
            self._render_image()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
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
        self.setGeometry(120, 70, 1120, 800)

        self.grid_size = 420
        self.alpha, self.beta, self.gamma = PRESETS["枝晶 Dendrite"]
        self.speed = 4
        self.auto_stop = True

        self.crystal = SnowCrystal(self.grid_size, self.alpha, self.beta, self.gamma)
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

        btn_reset = QPushButton("重新生长")
        btn_reset.clicked.connect(self.reset)
        vbox.addWidget(btn_reset)
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
        self.speed_slider.setRange(1, 30)
        self.speed_slider.setValue(self.speed)
        self.speed_slider.valueChanged.connect(self.on_speed)
        self.speed_label = QLabel(f"{self.speed} 步/帧")
        self.speed_label.setFixedWidth(56)
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        row2.addWidget(self.speed_slider, 1)
        row2.addWidget(self.speed_label)
        form.addRow("速度", row2)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(240, 640)
        self.size_spin.setSingleStep(60)
        self.size_spin.setValue(self.grid_size)
        self.size_spin.editingFinished.connect(self.apply_grid_size)
        form.addRow("网格", self.size_spin)

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
        self.ice_btn.colorChanged.connect(lambda c: self._set_color("ice_color", c))
        row.addWidget(self.ice_btn)
        row.addWidget(QLabel("底色"))
        self.bg_btn = ColorButton(self.canvas.bg_color)
        self.bg_btn.colorChanged.connect(lambda c: self._set_color("bg_color", c))
        row.addWidget(self.bg_btn)
        row.addStretch()
        vbox.addLayout(row)
        layout.addWidget(group)

    # ---------- 逻辑 ----------
    def _set_color(self, attr, color):
        setattr(self.canvas, attr, color)
        self.canvas._lut = None
        self.canvas.invalidate()

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

    def apply_grid_size(self):
        size = self.size_spin.value()
        if size == self.grid_size:
            return
        self.grid_size = size
        self.reset()
        self.lbl_status.setText(f"网格已改为 {size}×{size}")

    def reset(self):
        self.timer.stop()
        self.run_btn.setText("开始")
        self.crystal = SnowCrystal(self.grid_size, self.alpha, self.beta, self.gamma)
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
        self.crystal.step(max(1, self.speed))
        self.canvas.invalidate()
        self.update_stats()

    def tick(self):
        self.crystal.step(self.speed)
        self.canvas.invalidate()
        self.update_stats()
        if self.auto_stop and self.crystal.extent > 0.92:
            self.timer.stop()
            self.run_btn.setText("开始")
            self.lbl_status.setText("已长满，自动停止")

    def on_speed(self, value):
        self.speed = value
        self.speed_label.setText(f"{value} 步/帧")

    def toggle_vapor(self, checked):
        self.canvas.show_vapor = bool(checked)
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
