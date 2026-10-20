from custom_import import *
from PySide6.QtGui import QImage

# ==================== 地形与侵蚀 ====================
def _blur3(a):
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


def _upsample(g, n):
    """双线性 + smoothstep 插值放大到 n×n。"""
    f = g.shape[0]
    x = np.linspace(0, f - 1, n)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, f - 1)
    t = x - i0
    t = t * t * (3 - 2 * t)                      # smoothstep
    rows = g[i0, :] * (1 - t)[:, None] + g[i1, :] * t[:, None]
    return rows[:, i0] * (1 - t)[None, :] + rows[:, i1] * t[None, :]


def value_noise(n, rng, octaves=6, base=3):
    """多倍频值噪声，用来生成山地初始地形。"""
    total = np.zeros((n, n), dtype=np.float32)
    amp = 1.0
    for i in range(octaves):
        f = base * (2 ** i)
        if f > n:
            break
        g = rng.random((f, f)).astype(np.float32)
        total += amp * _upsample(g, n)
        amp *= 0.45
    total -= total.min()
    total /= max(1e-6, total.max())
    for _ in range(2):
        total = _blur3(total)
    return total


class ErosionSim:
    """雨滴侵蚀 + 干燥龟裂。

    侵蚀：每个雨滴沿坡向下流，按"携沙容量"侵蚀或沉积，容量与坡度、速度、水量成正比。
    龟裂：泥板收缩时沿 Voronoi 板块边界开裂，随干燥程度逐渐张开，只出现在平缓处。
    """

    MAX_DROPS = 6000

    def __init__(self, size=320, seed=1):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.h = self._make_terrain()
        self.h0 = self.h.copy()
        self.wet = np.zeros((size, size), dtype=np.float32)
        self.moisture = np.full((size, size), 0.85, dtype=np.float32)
        self.dry = 0.0
        self.raining = False
        self.drying = False
        self.rain_rate = 320

        # 侵蚀参数
        self.inertia = 0.35
        self.erode = 0.12
        self.deposit = 0.20
        self.capacity = 4.0
        self.min_slope = 0.008
        self.evap = 0.02
        self.gravity = 4.0
        self.max_life = 34

        # 雨滴状态
        self.px = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.py = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.dx = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.dy = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.spd = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.wat = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.sed = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.life = np.zeros(self.MAX_DROPS, dtype=np.int32)
        self.count = 0
        self.total_drops = 0

        # 龟裂场
        self.crack_line = None
        self.crack_noise = None
        self._build_cracks()

    # ---------- 地形 ----------
    def _make_terrain(self):
        n = self.size
        rng = self.rng
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        h = np.zeros((n, n), dtype=np.float32)

        # 几条随机走向的山脊叠成山脉，侵蚀出的沟谷才能在起伏上看出来
        for _ in range(5):
            theta = rng.random() * np.pi
            cx = n * rng.uniform(0.3, 0.7)
            cy = n * rng.uniform(0.3, 0.7)
            along = (xx - cx) * -np.sin(theta) + (yy - cy) * np.cos(theta)
            across = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
            across = np.abs(across) - rng.uniform(0.0, 0.12) * n
            ridge = np.exp(-(across ** 2) / (2 * (n * 0.11) ** 2)) \
                * np.exp(-(along ** 2) / (2 * (n * 0.5) ** 2))
            h += ridge * rng.uniform(0.6, 1.2)

        h /= max(1e-6, h.max())
        h += 0.30 * value_noise(n, rng, octaves=4, base=3)
        h -= h.min()
        h /= max(1e-6, h.max())
        return (h * 1.6).astype(np.float32)

    def reset(self):
        self.h = self._make_terrain()
        self.h0 = self.h.copy()
        self.wet[:] = 0.0
        self.moisture[:] = 0.85
        self.dry = 0.0
        self.count = 0
        self.total_drops = 0
        self._build_cracks()

    # ---------- 龟裂 ----------
    def _build_cracks(self):
        n = self.size
        rng = self.rng
        n_sites = int(n / 3) + rng.integers(0, n // 6)
        sites = rng.random((n_sites, 2)) * (n - 1)

        ys = np.arange(n, dtype=np.float32)[:, None]
        xs = np.arange(n, dtype=np.float32)[None, :]
        d1 = np.full((n, n), 1e9, dtype=np.float32)
        d2 = np.full((n, n), 1e9, dtype=np.float32)
        for sx, sy in sites:
            d = np.hypot(xs - sx, ys - sy)
            closer = d < d1
            d2 = np.where(closer, d1, np.minimum(d2, d))
            d1 = np.where(closer, d, d1)

        width = 2.2
        self.crack_line = np.clip(1.0 - (d2 - d1) / width, 0.0, 1.0).astype(np.float32)

        # 开裂的"传播噪声"：低频率噪声，让裂纹一片片出现
        noise = value_noise(n, rng, octaves=4, base=5)
        self.crack_noise = (0.55 * noise + 0.45 * _blur3(_blur3(noise))).astype(np.float32)

    # ---------- 雨滴 ----------
    def add_rain(self, count, cx=None, cy=None, radius=0):
        if count <= 0:
            return
        free = self.MAX_DROPS - self.count
        count = int(min(count, free))
        if count <= 0:
            return
        i0 = self.count
        i1 = i0 + count
        n = self.size - 2
        if cx is None:
            self.px[i0:i1] = self.rng.random(count) * n + 1.0
            self.py[i0:i1] = self.rng.random(count) * n + 1.0
        else:
            ang = self.rng.random(count) * (2 * np.pi)
            rad = np.sqrt(self.rng.random(count)) * max(1.0, radius)
            self.px[i0:i1] = np.clip(cx + np.cos(ang) * rad, 1, n)
            self.py[i0:i1] = np.clip(cy + np.sin(ang) * rad, 1, n)
        self.dx[i0:i1] = 0.0
        self.dy[i0:i1] = 0.0
        self.spd[i0:i1] = 1.0
        self.wat[i0:i1] = 1.0
        self.sed[i0:i1] = 0.0
        self.life[i0:i1] = self.max_life
        self.count = i1
        self.total_drops += count

    def _sample(self, x, y):
        n = self.size
        xi = np.clip(x.astype(np.int32), 0, n - 2)
        yi = np.clip(y.astype(np.int32), 0, n - 2)
        u = np.clip(x - xi, 0.0, 1.0)
        v = np.clip(y - yi, 0.0, 1.0)
        h = self.h
        h00 = h[yi, xi]
        h10 = h[yi, xi + 1]
        h01 = h[yi + 1, xi]
        h11 = h[yi + 1, xi + 1]
        height = h00 * (1 - u) * (1 - v) + h10 * u * (1 - v) + h01 * (1 - u) * v + h11 * u * v
        gx = (h00 - h10) * (1 - v) + (h01 - h11) * v
        gy = (h00 - h01) * (1 - u) + (h10 - h11) * u
        return height, gx, gy, yi, xi

    def step_drops(self):
        if self.count == 0:
            return
        n = self.count
        x = self.px[:n]
        y = self.py[:n]
        h_old, gx, gy, yi, xi = self._sample(x, y)

        # 方向：惯性 + 下坡
        inv = 1.0 - self.inertia
        self.dx[:n] = self.dx[:n] * self.inertia - gx * inv
        self.dy[:n] = self.dy[:n] * self.inertia - gy * inv
        norm = np.hypot(self.dx[:n], self.dy[:n])
        norm[norm < 1e-6] = 1.0
        self.dx[:n] /= norm
        self.dy[:n] /= norm

        nx = x + self.dx[:n]
        ny = y + self.dy[:n]
        h_new, _, _, _, _ = self._sample(nx, ny)

        dh = h_new - h_old
        cap = np.maximum(-dh, self.min_slope) * self.spd[:n] * self.capacity * self.wat[:n]
        dep = np.maximum(self.sed[:n] - cap, 0.0) * self.deposit
        ero = np.minimum((cap - self.sed[:n]) * self.erode, np.maximum(-dh, 0.0))

        delta = dep - ero
        np.add.at(self.h, (yi, xi), delta)
        self.sed[:n] = np.clip(self.sed[:n] + ero - dep, 0.0, 4.0)

        self.spd[:n] = np.sqrt(np.maximum(0.0, self.spd[:n] ** 2 + dh * self.gravity))
        self.wat[:n] = self.wat[:n] * (1.0 - self.evap)
        self.px[:n] = nx
        self.py[:n] = ny
        self.life[:n] -= 1

        # 记录湿润痕迹
        self.wet[yi, xi] += 0.05
        np.clip(self.wet, 0.0, 1.0, out=self.wet)

        dead = (self.life[:n] <= 0) | (self.wat[:n] < 0.02) | (nx < 1) | (ny < 1) \
            | (nx > self.size - 2) | (ny > self.size - 2)
        if dead.any():
            keep = ~dead
            m = int(keep.sum())
            for arr in (self.px, self.py, self.dx, self.dy, self.spd, self.wat, self.sed, self.life):
                arr[:m] = arr[:n][keep]
            self.count = m

    # ---------- 每帧 ----------
    def step(self):
        if self.raining:
            self.add_rain(self.rain_rate)
            self.wet *= 0.995
            self.dry = max(0.0, self.dry - 0.006)
        else:
            self.wet *= 0.988
            if self.drying:
                self.dry = min(1.0, self.dry + 0.005)
            else:
                self.dry = max(0.0, self.dry - 0.002)
        self.step_drops()
        np.clip(self.h, 0.0, 3.0, out=self.h)


# ==================== 画布 ====================
class TerrainCanvas(QWidget):
    def __init__(self, sim, app=None):
        super().__init__()
        self.sim = sim
        self.app = app
        self.show_water = True
        self.show_cracks = True
        self.show_shade = True
        self._soil_lut = None
        self._image = None
        self._dirty = True
        self.setMinimumSize(420, 420)
        self._drawing = False

    def set_sim(self, sim):
        self.sim = sim
        self._dirty = True

    def invalidate(self):
        self._dirty = True
        self.update()

    # ---------- 着色 ----------
    def _build_soil_lut(self):
        """256×256 查表：高度 × 湿度 → 土壤颜色（含高处露岩）。"""
        hi = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None]
        wi = np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :]
        dry = np.array([188, 162, 118], np.float32)
        wetc = np.array([118, 92, 66], np.float32)
        lut = dry[None, None, :] * (1.0 - wi[..., None]) + wetc[None, None, :] * wi[..., None]
        lut = np.broadcast_to(lut, (256, 256, 3)).copy()
        rock = np.clip((hi - 0.66) / 0.34, 0.0, 1.0)[..., None] * 0.55
        lut = lut * (1.0 - rock) + np.array([146, 142, 138], np.float32) * rock
        self._soil_lut = np.clip(lut, 0, 255).astype(np.uint8).reshape(-1, 3)

    def _render_image(self):
        sim = self.sim
        n = sim.size
        h = _blur3(sim.h)     # 显示用平滑：抹掉单格开凿痕迹

        hmin, hmax = float(h.min()), float(h.max())
        hn = (h - hmin) / max(1e-6, hmax - hmin)

        # 光影：梯度按格子尺寸放大成真实坡度，再用法线点乘光向
        gy, gx = np.gradient(h)
        exag = n * 0.40
        nx, ny, nz = -gx * exag, -gy * exag, np.ones_like(hn)
        inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
        light = (nx * -0.48 + ny * -0.48 + nz * 0.73) * inv
        shade = np.clip(0.62 + 0.75 * light, 0.30, 1.45)

        # 干土 / 湿泥 + 高处露岩：查表（以高度和湿度为索引），省掉多次浮点混合
        if self._soil_lut is None:
            self._build_soil_lut()
        wetv = np.clip(_blur3(sim.wet) * 1.25, 0.0, 1.0)
        idx = (hn * 255.0).astype(np.int32) * 256 + (wetv * 255.0).astype(np.int32)
        rgb = self._soil_lut[idx].astype(np.float32)

        if self.show_shade:
            rgb *= shade[..., None]

        # 积水 / 湿润：低洼处偏蓝
        if self.show_water:
            pond = np.clip((0.16 - hn) / 0.16, 0.0, 1.0) * np.clip((sim.wet - 0.10) * 2.5, 0, 1)
            water = np.clip(pond, 0.0, 1.0)[..., None]
            rgb = rgb * (1 - water * 0.75) + np.array([70, 118, 165], np.float32) * (water * 0.75)

        # 龟裂：平缓、且已经干透的地方张开
        if self.show_cracks and sim.dry > 0.02:
            slope = np.hypot(gx, gy) * exag
            flat = np.clip(1.0 - slope * 0.55, 0.0, 1.0)
            reveal = np.clip((sim.dry * 1.35 - sim.crack_noise) * 2.8, 0.0, 1.0)
            crack = sim.crack_line * reveal * flat * np.clip((sim.dry - 0.15) * 1.8, 0.0, 1.0)
            crack = _blur3(crack)[..., None]
            rgb = rgb * (1 - crack * 0.88) + np.array([58, 46, 38], np.float32) * (crack * 0.88)

        out = np.clip(rgb, 0, 255).astype(np.uint8)
        self._image = QImage(out.data, n, n, n * 3, QImage.Format_RGB888).copy()
        self._dirty = False

    def paintEvent(self, event):
        if self._dirty or self._image is None:
            self._render_image()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        side = min(self.width(), self.height())
        x0 = (self.width() - side) // 2
        y0 = (self.height() - side) // 2
        painter.drawImage(QRect(x0, y0, side, side), self._image)
        painter.end()

    # ---------- 鼠标：局部降雨 ----------
    def _to_sim(self, pos):
        side = min(self.width(), self.height())
        x0 = (self.width() - side) // 2
        y0 = (self.height() - side) // 2
        if side <= 0:
            return None
        sx = (pos.x() - x0) / side * self.sim.size
        sy = (pos.y() - y0) / side * self.sim.size
        if 0 <= sx < self.sim.size and 0 <= sy < self.sim.size:
            return sx, sy
        return None

    def mousePressEvent(self, event):
        p = self._to_sim(event.position().toPoint())
        if p is None:
            return
        self._drawing = True
        self.sim.add_rain(160, p[0], p[1], radius=self.sim.size * 0.05)

    def mouseMoveEvent(self, event):
        if not self._drawing:
            return
        p = self._to_sim(event.position().toPoint())
        if p is not None:
            self.sim.add_rain(60, p[0], p[1], radius=self.sim.size * 0.04)

    def mouseReleaseEvent(self, event):
        self._drawing = False


# ==================== 主窗口 ====================
class ErosionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水力侵蚀 · 干燥龟裂")
        self.setGeometry(120, 70, 1080, 800)

        self.size = 320
        self.sim = ErosionSim(self.size, seed=1)
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(45)
        self.update_stats()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(272)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(6)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = TerrainCanvas(self.sim, self)

        self._build_weather_group(panel_layout)
        self._build_erosion_group(panel_layout)
        self._build_display_group(panel_layout)
        panel_layout.addStretch()

        root.addWidget(panel)
        root.addWidget(self.canvas, 1)

        self.lbl_status = QLabel("点「降雨」开始冲刷，点「干燥」看泥面开裂")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_dry = QLabel("干燥 0%")
        self.lbl_drops = QLabel("雨滴 0")
        self.lbl_fps = QLabel("帧率 –")
        for w in (self.lbl_dry, self.lbl_drops, self.lbl_fps):
            self.statusBar().addPermanentWidget(w)
        self._frames = 0
        self._fps_t = time.perf_counter()

    def _build_weather_group(self, layout):
        group = QGroupBox("天气")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.rain_btn = QPushButton("降雨")
        self.rain_btn.clicked.connect(self.toggle_rain)
        row.addWidget(self.rain_btn, 1)
        self.dry_btn = QPushButton("干燥")
        self.dry_btn.clicked.connect(self.toggle_dry)
        row.addWidget(self.dry_btn, 1)
        vbox.addLayout(row)

        form = QFormLayout()
        form.setSpacing(4)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.rain_slider = QSlider(Qt.Horizontal)
        self.rain_slider.setRange(40, 1200)
        self.rain_slider.setValue(self.sim.rain_rate)
        self.rain_slider.valueChanged.connect(self.on_rain_rate)
        self.rain_label = QLabel(str(self.sim.rain_rate))
        self.rain_label.setFixedWidth(44)
        self.rain_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        r = QHBoxLayout()
        r.setSpacing(4)
        r.addWidget(self.rain_slider, 1)
        r.addWidget(self.rain_label)
        form.addRow("雨量", r)

        reset_btn = QPushButton("重置地形")
        reset_btn.clicked.connect(self.reset)
        form.addRow("", reset_btn)
        vbox.addLayout(form)
        layout.addWidget(group)

    def _build_erosion_group(self, layout):
        group = QGroupBox("侵蚀参数")
        form = QFormLayout(group)
        form.setSpacing(4)
        form.setContentsMargins(6, 6, 6, 6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.sl_erode, _ = self._slider(form, "侵蚀", 5, 100, int(self.sim.erode * 100),
                                        lambda v: setattr(self.sim, "erode", v / 100.0))
        self.sl_deposit, _ = self._slider(form, "沉积", 5, 100, int(self.sim.deposit * 100),
                                          lambda v: setattr(self.sim, "deposit", v / 100.0))
        self.sl_inertia, _ = self._slider(form, "惯性", 0, 95, int(self.sim.inertia * 100),
                                          lambda v: setattr(self.sim, "inertia", v / 100.0))
        self.sl_capacity, _ = self._slider(form, "携沙", 10, 100, int(self.sim.capacity * 10),
                                           lambda v: setattr(self.sim, "capacity", v / 10.0))
        layout.addWidget(group)

    def _slider(self, form, text, lo, hi, value, setter):
        row = QHBoxLayout()
        row.setSpacing(4)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        label = QLabel(str(value))
        label.setFixedWidth(38)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def on_change(v):
            label.setText(str(v))
            setter(v)

        slider.valueChanged.connect(on_change)
        row.addWidget(slider, 1)
        row.addWidget(label)
        form.addRow(text, row)
        return slider, label

    def _build_display_group(self, layout):
        group = QGroupBox("显示")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(2)
        for text, attr, default in (("光影", "show_shade", True),
                                    ("积水 / 湿润", "show_water", True),
                                    ("干燥龟裂", "show_cracks", True)):
            cb = QCheckBox(text)
            cb.setChecked(default)
            cb.toggled.connect(lambda checked, a=attr: self._toggle(a, checked))
            vbox.addWidget(cb)
        layout.addWidget(group)

    def _toggle(self, attr, checked):
        setattr(self.canvas, attr, bool(checked))
        self.canvas.invalidate()

    # ---------- 交互 ----------
    def toggle_rain(self):
        self.sim.raining = not self.sim.raining
        self.rain_btn.setText("停雨" if self.sim.raining else "降雨")
        if self.sim.raining:
            self.sim.drying = False
            self.dry_btn.setText("干燥")
            self.lbl_status.setText("正在降雨冲刷…")

    def toggle_dry(self):
        self.sim.drying = not self.sim.drying
        self.dry_btn.setText("停止干燥" if self.sim.drying else "干燥")
        if self.sim.drying:
            self.sim.raining = False
            self.rain_btn.setText("降雨")
            self.lbl_status.setText("泥土干燥中，裂纹逐渐张开…")

    def reset(self):
        self.sim.reset()
        self.canvas.invalidate()
        self.lbl_status.setText("地形已重置")

    def on_rain_rate(self, v):
        self.sim.rain_rate = v
        self.rain_label.setText(str(v))

    def tick(self):
        sim = self.sim
        busy = sim.raining or sim.drying or sim.count > 0
        if not busy:
            return
        sim.step()
        if sim.raining:
            sim.step()          # 下雨时每帧推进两步，视觉速度不变但重绘次数少一半
        self.canvas.invalidate()
        self._frames += 1

    def update_stats(self):
        self.lbl_dry.setText(f"干燥 {self.sim.dry * 100:.0f}%")
        self.lbl_drops.setText(f"雨滴 {self.sim.count:,}")
        now = time.perf_counter()
        if now - self._fps_t >= 0.5:
            self.lbl_fps.setText(f"帧率 {self._frames / (now - self._fps_t):.0f}")
            self._frames = 0
            self._fps_t = now

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ErosionApp()
    window.show()
    sys.exit(app.exec())
