# -*- coding: utf-8 -*-
"""Diffusion-limited aggregation: coral-like growth from random walkers.

The whole simulation lives on the GPU: walkers, occupancy and attachment time.
A single render kernel maps the occupancy grid straight to canvas pixels
(bilinear footprint, supersampled, continuous 256-level palette), so a frame
costs one small device-to-host RGB copy instead of a full grid read-back plus a
Python scatter.  Grids are allocated once at MAX_GRID and the active size is a
kernel argument, so switching 512/1024/2048 never recompiles a kernel.
"""
import numpy as np

from common.lab_window import LabWindow
from common.taichi_base import ensure_taichi
from common.qt_base import (
    QApplication, QCheckBox, QColor, QComboBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QImage, QLabel, QMessageBox, QPainter, QPen,
    QPushButton, QTimer, QVBoxLayout, QWidget, Qt, run_app, LabeledSlider,
)
from common.utils import FPSCounter

ti = ensure_taichi()

MAX_GRID = 2048
MAX_WALKERS = 32768
MAX_RES = 1200
GRID_SIZES = (512, 1024, 2048)
CANVAS_SIZE = 600

# Coral-reef palette: deep water -> lagoon teal -> coral red -> coral pink
# -> pale sand.  Anchors sampled from the reef photographs used as reference.
PALETTE_STOPS = np.array([0.0, 0.22, 0.45, 0.68, 0.86, 1.0])
PALETTE_COLS = np.array([
    [20, 46, 58], [42, 152, 192], [198, 54, 56],
    [236, 118, 144], [231, 156, 194], [220, 188, 152],
], dtype=float)


def palette_lut(levels=256):
    """Continuous 256-level colour ramp (>= 64 levels, no banding)."""
    t = np.linspace(0.0, 1.0, levels)
    lut = np.empty((levels, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.interp(t, PALETTE_STOPS, PALETTE_COLS[:, ch]).astype(np.uint8)
    return lut


@ti.data_oriented
class DLASimulation:
    def __init__(self, size=1024, res=CANVAS_SIZE):
        self.size = int(size)
        self.res = int(min(MAX_RES, max(CANVAS_SIZE, res)))
        self.occ = ti.field(ti.u8, shape=(MAX_GRID, MAX_GRID))
        self.age = ti.field(ti.i32, shape=(MAX_GRID, MAX_GRID))
        self.grain = ti.field(ti.u8, shape=(MAX_GRID, MAX_GRID))
        self.wx = ti.field(ti.i32, shape=MAX_WALKERS)
        self.wy = ti.field(ti.i32, shape=MAX_WALKERS)
        self.alive = ti.field(ti.u8, shape=MAX_WALKERS)
        self.max_r = ti.field(ti.i32, shape=())
        self.steps = ti.field(ti.i32, shape=())
        self.n_walkers = ti.field(ti.i32, shape=())
        self.hist = ti.field(ti.i32, shape=MAX_GRID // 2 + 4)
        self.lut = ti.Vector.field(3, ti.u8, shape=256)
        self.pix = ti.Vector.field(3, ti.u8, shape=(self.res, self.res))
        self.lut.from_numpy(palette_lut())
        self._init_grain()
        self.reset()

    # ------------------------------------------------------------------ setup
    @ti.kernel
    def _init_grain(self):
        for i, j in ti.ndrange(MAX_GRID, MAX_GRID):
            h = ti.sin(ti.cast(i, ti.f32) * 12.9898 +
                       ti.cast(j, ti.f32) * 78.233) * 43758.5453
            h = h - ti.floor(h)
            self.grain[i, j] = ti.cast(h * 255.0, ti.u8)

    @ti.kernel
    def _reset_kernel(self, size: ti.i32):
        for i, j in ti.ndrange(size, size):
            self.occ[i, j] = ti.cast(0, ti.u8)
            self.age[i, j] = 0
        for i in range(MAX_WALKERS):
            self.alive[i] = ti.cast(0, ti.u8)
        c = size // 2
        self.occ[c, c] = ti.cast(1, ti.u8)
        self.age[c, c] = 1
        self.max_r[None] = 0
        self.steps[None] = 0

    def reset(self, size=None):
        if size is not None:
            self.size = int(size)
        self._reset_kernel(self.size)

    @ti.kernel
    def set_walkers_kernel(self, n: ti.i32):
        self.n_walkers[None] = ti.max(1, ti.min(n, MAX_WALKERS))
        for i in range(MAX_WALKERS):
            if i >= self.n_walkers[None]:
                self.alive[i] = ti.cast(0, ti.u8)

    def set_walkers(self, n):
        self.set_walkers_kernel(int(n))

    # ----------------------------------------------------------------- growth
    @ti.func
    def _spawn(self, i: ti.i32, size: ti.i32):
        c = size // 2
        limit = size // 2 - 2
        gap = ti.max(4, size // 160)
        rad = ti.min(limit - 3, self.max_r[None] + gap)
        ang = ti.random() * 6.283185307
        x = ti.cast(ti.cast(c, ti.f32) + ti.cos(ang) * ti.cast(rad, ti.f32), ti.i32)
        y = ti.cast(ti.cast(c, ti.f32) + ti.sin(ang) * ti.cast(rad, ti.f32), ti.i32)
        x = ti.max(1, ti.min(size - 2, x))
        y = ti.max(1, ti.min(size - 2, y))
        if self.occ[x, y] == 0:
            self.wx[i] = x
            self.wy[i] = y
            self.alive[i] = ti.cast(1, ti.u8)

    @ti.kernel
    def step(self, size: ti.i32, substeps: ti.i32, stick: ti.f32, drift: ti.f32):
        c = size // 2
        limit = size // 2 - 2
        limit2 = limit * limit
        for i in range(MAX_WALKERS):
            if i >= self.n_walkers[None]:
                self.alive[i] = ti.cast(0, ti.u8)
            elif self.alive[i] == 0:
                self._spawn(i, size)
            else:
                for _ in range(substeps):
                    if self.alive[i] == 0:
                        break
                    x = self.wx[i]
                    y = self.wy[i]
                    r = ti.random()
                    dx = 0
                    dy = 0
                    if r < 0.25:
                        dx = 1
                    elif r < 0.5:
                        dx = -1
                    elif r < 0.75:
                        dy = 1
                    else:
                        dy = -1
                    if ti.random() < ti.abs(drift):
                        dx = 1 if drift > 0 else -1
                        dy = 0
                    nx = x + dx
                    ny = y + dy
                    if nx < 1 or nx > size - 2 or ny < 1 or ny > size - 2:
                        self.alive[i] = ti.cast(0, ti.u8)
                        break
                    if self.occ[nx, ny] != 0:
                        if ti.random() < stick:
                            if self.occ[x, y] == 0:
                                self.occ[x, y] = ti.cast(1, ti.u8)
                                self.age[x, y] = self.steps[None]
                                rr = ti.cast(ti.sqrt((x - c) * (x - c) +
                                                     (y - c) * (y - c)), ti.i32)
                                ti.atomic_max(self.max_r[None], rr)
                            self.alive[i] = ti.cast(0, ti.u8)
                            break
                    else:
                        self.wx[i] = nx
                        self.wy[i] = ny
                        ux = nx - c
                        uy = ny - c
                        if ux * ux + uy * uy > limit2:
                            self.alive[i] = ti.cast(0, ti.u8)
                            break
        self.steps[None] += 1

    # ------------------------------------------------------------------ stats
    @ti.kernel
    def _clear_hist(self):
        for i in range(self.hist.shape[0]):
            self.hist[i] = 0

    @ti.kernel
    def radial_hist(self, size: ti.i32):
        c = size // 2
        for i, j in ti.ndrange(size, size):
            if self.occ[i, j] != 0:
                rr = ti.cast(ti.sqrt(ti.cast((i - c) * (i - c) +
                                             (j - c) * (j - c), ti.f32)), ti.i32)
                if rr < self.hist.shape[0]:
                    ti.atomic_add(self.hist[rr], 1)

    def stats(self):
        self._clear_hist()
        self.radial_hist(self.size)
        h = self.hist.to_numpy()
        count = int(h.sum())
        cs = np.cumsum(h)
        rs = np.arange(len(cs), dtype=np.float64)
        radius = int(self.max_r[None])
        sel = (rs > 5) & (rs <= radius) & (cs > 20)
        dim = float("nan")
        if sel.sum() >= 5:
            dim = float(np.polyfit(np.log(rs[sel]), np.log(cs[sel]), 1)[0])
        return count, radius, int(self.steps[None]), dim

    # ----------------------------------------------------------------- render
    @ti.func
    def _sample(self, size: ti.i32, total: ti.f32, gx: ti.f32, gy: ti.f32,
                gamma: ti.f32, gamp: ti.f32, shade: ti.f32):
        """Bilinear footprint -> (premultiplied rgb, coverage)."""
        fx = gx - 0.5
        fy = gy - 0.5
        x0 = ti.floor(fx)
        y0 = ti.floor(fy)
        tx = fx - x0
        ty = fy - y0
        sumw = 0.0
        suma = 0.0
        sumg = 0.0
        for dy in ti.static(range(2)):
            wy = ty if dy == 1 else 1.0 - ty
            yi = ti.cast(y0, ti.i32) + dy
            for dx in ti.static(range(2)):
                wx = tx if dx == 1 else 1.0 - tx
                xi = ti.cast(x0, ti.i32) + dx
                if xi >= 0 and xi < size and yi >= 0 and yi < size:
                    if self.occ[xi, yi] != 0:
                        w = wx * wy
                        sumw += w
                        suma += w * ti.cast(self.age[xi, yi], ti.f32)
                        sumg += w * ti.cast(self.grain[xi, yi], ti.f32)
        col = ti.Vector([0.0, 0.0, 0.0])
        if sumw > 1e-4:
            t = ti.min(1.0, ti.max(0.0, suma / sumw / total))
            t = ti.pow(t, gamma)
            idx = ti.cast(t * 255.0, ti.i32)
            g = 1.0 - gamp + 2.0 * gamp * (sumg / sumw) / 255.0
            g *= 1.0 - shade + shade * t
            col = ti.Vector([
                ti.cast(self.lut[idx][0], ti.f32) * g,
                ti.cast(self.lut[idx][1], ti.f32) * g,
                ti.cast(self.lut[idx][2], ti.f32) * g,
            ])
        return ti.Vector([col[0] * sumw, col[1] * sumw, col[2] * sumw, sumw])

    @ti.kernel
    def render(self, size: ti.i32, res: ti.i32, cx: ti.f32, cy: ti.f32,
               half: ti.f32, ss: ti.i32, gamma: ti.f32, gamp: ti.f32,
               shade: ti.f32):
        scale = 2.0 * half / ti.cast(res, ti.f32)
        total = ti.max(1.0, ti.cast(self.steps[None], ti.f32))
        inv = 1.0 / ti.cast(ss, ti.f32)
        for i, j in ti.ndrange(res, res):
            ar = 0.0
            ag = 0.0
            ab = 0.0
            aa = 0.0
            for sy in range(ss):
                for sx in range(ss):
                    gx = cx + (ti.cast(j, ti.f32) + (ti.cast(sx, ti.f32) + 0.5) * inv
                               - ti.cast(res, ti.f32) * 0.5) * scale
                    gy = cy + (ti.cast(i, ti.f32) + (ti.cast(sy, ti.f32) + 0.5) * inv
                               - ti.cast(res, ti.f32) * 0.5) * scale
                    s = self._sample(size, total, gx, gy, gamma, gamp, shade)
                    ar += s[0]
                    ag += s[1]
                    ab += s[2]
                    aa += s[3]
            a = aa * inv * inv
            if a > 1e-4:
                r = ar / aa
                g = ag / aa
                b = ab / aa
                self.pix[i, j] = ti.Vector([
                    ti.cast(255.0 + (r - 255.0) * a, ti.u8),
                    ti.cast(255.0 + (g - 255.0) * a, ti.u8),
                    ti.cast(255.0 + (b - 255.0) * a, ti.u8),
                ])
            else:
                self.pix[i, j] = ti.Vector([ti.cast(255, ti.u8),
                                            ti.cast(255, ti.u8),
                                            ti.cast(255, ti.u8)])

    def render_numpy(self, cx, cy, half, ss=2, gamma=0.50, gamp=0.04, shade=0.28):
        self.render(self.size, self.res, float(cx), float(cy), float(half), int(ss),
                    float(gamma), float(gamp), float(shade))
        return self.pix.to_numpy()


def screen_resolution():
    """Physical canvas resolution for the current screen scale factor."""
    app = QApplication.instance()
    dpr = 1.0
    if app is not None:
        screen = app.primaryScreen()
        if screen is not None:
            dpr = float(screen.devicePixelRatio())
    return int(min(MAX_RES, round(CANVAS_SIZE * max(1.0, dpr))))


class DLACanvas(QWidget):
    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.setFixedSize(600, 600)
        self._image = None
        self._buffer = None
        self.dpr = sim.res / float(CANVAS_SIZE)
        self.auto_fit = True
        self.view_cx = sim.size * 0.5
        self.view_cy = sim.size * 0.5
        self.view_half = 0.0
        self.supersample = 2

    # ------------------------------------------------------------- view / draw
    def update_view(self):
        sim = self.sim
        radius = float(sim.max_r[None])
        if self.auto_fit:
            target = max(40.0, radius * 1.16 + 8.0)
        else:
            target = sim.size * 0.5 + 1.0
        if self.view_half <= 0.0 or self.view_half > sim.size:
            self.view_half = target
        else:
            self.view_half += (target - self.view_half) * 0.10
        self.view_cx = sim.size * 0.5
        self.view_cy = sim.size * 0.5
        self.supersample = 1
        if sim.size >= 900:
            self.supersample = 2
        if sim.size >= 1600:
            self.supersample = 3

    def render_frame(self):
        self.update_view()
        data = self.sim.render_numpy(
            self.view_cx, self.view_cy, self.view_half, self.supersample
        )
        self._buffer = data
        h, w = data.shape[:2]
        image = QImage(data.data, w, h, w * 3, QImage.Format_RGB888)
        image.setDevicePixelRatio(self.dpr)
        self._image = image
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        if self._image is not None:
            p.drawImage(0, 0, self._image)
        p.setPen(QPen(QColor(198, 209, 211), 1))
        p.drawRect(1, 1, self.width() - 2, self.height() - 2)
        p.end()

    def save(self, path):
        if self._image is None:
            self.render_frame()
        return self._image.save(path)


class DLAApp(LabWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLA · 扩散限制凝聚")
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.resize(960, 640)
        self.setMinimumSize(960, 640)
        self.sim = DLASimulation(1024, screen_resolution())
        self.sim.set_walkers(8192)
        self.canvas = DLACanvas(self.sim)
        self._build_ui()
        self.fps = FPSCounter()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self.frame = 0

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(330)
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        run = QGroupBox("模拟")
        rh = QHBoxLayout(run)
        rh.setContentsMargins(6, 6, 6, 6)
        rh.setSpacing(4)
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.toggle_running)
        self.step_btn = QPushButton("单步")
        self.step_btn.clicked.connect(self.single_step)
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset)
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_image)
        for b in (self.pause_btn, self.step_btn, self.reset_btn, self.save_btn):
            rh.addWidget(b)
        col.addWidget(run)

        params = QGroupBox("生长参数")
        form = QFormLayout(params)
        form.setContentsMargins(8, 6, 8, 6)
        form.setSpacing(5)
        self.grid_combo = QComboBox()
        self.grid_combo.addItems([str(n) for n in GRID_SIZES])
        self.grid_combo.setMinimumWidth(110)
        self.grid_combo.setCurrentText("1024")
        self.grid_combo.currentTextChanged.connect(self.change_grid)
        form.addRow("网格", self.grid_combo)

        self.walkers_slider = LabeledSlider(
            "", 512, MAX_WALKERS, 8192, formatter=lambda v: f"{v}", value_width=56
        )
        self.walkers_slider.valueChanged.connect(self.change_walkers)
        form.addRow("游走粒子", self.walkers_slider)

        self.speed_slider = LabeledSlider(
            "", 1, 32, 12, formatter=lambda v: f"{v}×", value_width=44
        )
        self.speed_slider.valueChanged.connect(self.change_speed)
        form.addRow("每帧步数", self.speed_slider)

        self.stick_slider = LabeledSlider(
            "", 5, 100, 100, formatter=lambda v: f"{v}%", value_width=46
        )
        self.stick_slider.valueChanged.connect(self.change_stick)
        form.addRow("黏附概率", self.stick_slider)

        self.drift_slider = LabeledSlider(
            "", -100, 100, 0, formatter=lambda v: f"{v}%", value_width=46
        )
        self.drift_slider.valueChanged.connect(self.change_drift)
        form.addRow("风向漂移", self.drift_slider)

        self.fit_box = QCheckBox("自动缩放")
        self.fit_box.setChecked(True)
        self.fit_box.toggled.connect(self.change_fit)
        form.addRow("", self.fit_box)
        col.addWidget(params)

        presets = QGroupBox("预设")
        ph = QHBoxLayout(presets)
        ph.setContentsMargins(6, 6, 6, 6)
        for text, values in (
            ("珊瑚", (100, 0, 8192)),
            ("地衣", (55, 0, 12288)),
            ("枝晶", (100, 35, 6144)),
        ):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, v=values: self.apply_preset(*v))
            ph.addWidget(b)
        col.addWidget(presets)

        stats = QGroupBox("读数")
        sv = QVBoxLayout(stats)
        sv.setContentsMargins(8, 6, 8, 6)
        sv.setSpacing(4)
        self.lbl_count = QLabel("颗粒  1")
        self.lbl_radius = QLabel("半径  0")
        self.lbl_cov = QLabel("覆盖  0.00%")
        self.lbl_dim = QLabel("维数  —")
        self.lbl_steps = QLabel("步数  0")
        for w in (self.lbl_count, self.lbl_radius, self.lbl_cov,
                  self.lbl_dim, self.lbl_steps):
            w.setStyleSheet("font-family: Consolas, monospace;")
            sv.addWidget(w)
        col.addWidget(stats)

        info = QGroupBox("机制")
        iv = QVBoxLayout(info)
        iv.setContentsMargins(8, 6, 8, 6)
        txt = QLabel(
            "粒子从团簇外缘随机游走，碰到已有颗粒后按概率黏附。\n"
            "外层枝杈先接住粒子、屏蔽内部，形成珊瑚状屏蔽生长；"
            "维数由质量–半径斜率估计。"
        )
        txt.setWordWrap(True)
        txt.setStyleSheet("color:#52606d; font-size:12px;")
        iv.addWidget(txt)
        col.addWidget(info)
        col.addStretch(1)

        root.addWidget(panel)
        root.addWidget(self.canvas)
        self.status_left = QLabel("运行中")
        self.status_right = QLabel("")
        self.statusBar().addWidget(self.status_left, 0)
        self.statusBar().addPermanentWidget(self.status_right, 0)

        self.steps_per_frame = 12
        self.stick = 1.0
        self.drift = 0.0

    # ------------------------------------------------------------------- loop
    def tick(self):
        if self.sim is None:
            return
        self.sim.step(self.sim.size, self.steps_per_frame, self.stick, self.drift)
        self.canvas.render_frame()
        self.frame += 1
        if self.frame % 10 == 0:
            self.update_stats()
        fps = self.fps.tick()
        if fps is not None:
            self._update_status()

    def single_step(self):
        self.tick()
        self.update_stats()

    def toggle_running(self):
        if self.timer.isActive():
            self.timer.stop()
            self.pause_btn.setText("开始")
            self.status_left.setText("已暂停")
        else:
            self.timer.start()
            self.pause_btn.setText("暂停")
            self.status_left.setText("运行中")

    # --------------------------------------------------------------- commands
    def reset(self):
        self.sim.reset(self.sim.size)
        self.sim.set_walkers(self.walkers_slider.value())
        self.canvas.view_half = 0.0
        self.canvas.render_frame()
        self.frame = 0
        self.fps.reset()
        self.update_stats()

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存画布", "dla.png", "PNG 图片 (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if not self.canvas.save(path):
            QMessageBox.warning(self, "保存失败", f"无法写入：{path}")
            return
        self.status_left.setText("已保存")

    def apply_preset(self, stick, drift, walkers):
        self.stick_slider.setValue(int(stick))
        self.drift_slider.setValue(int(drift))
        self.walkers_slider.setValue(int(min(walkers, MAX_WALKERS)))
        self.reset()

    def change_grid(self, text):
        n = int(text)
        self.sim.reset(n)
        self.sim.set_walkers(self.walkers_slider.value())
        self.canvas.view_half = 0.0
        self.canvas.render_frame()
        self.frame = 0
        self.update_stats()

    def change_walkers(self, value):
        self.sim.set_walkers(int(value))
        self._update_status()

    def change_speed(self, value):
        self.steps_per_frame = int(value)

    def change_stick(self, value):
        self.stick = max(0.05, float(value) / 100.0)

    def change_drift(self, value):
        self.drift = float(value) / 100.0

    def change_fit(self, checked):
        self.canvas.auto_fit = bool(checked)

    # ------------------------------------------------------------------ stats
    def update_stats(self):
        count, radius, steps, dim = self.sim.stats()
        n = self.sim.size
        cov = count / float(n * n) * 100.0
        self.lbl_count.setText(f"颗粒  {count:,}")
        self.lbl_radius.setText(f"半径  {radius}")
        self.lbl_cov.setText(f"覆盖  {cov:.2f}%")
        self.lbl_dim.setText(f"维数  {dim:.2f}" if dim == dim else "维数  —")
        self.lbl_steps.setText(f"步数  {steps:,}")
        self._update_status()

    def _update_status(self):
        self.status_right.setText(
            f"{self.sim.size}² · {self.walkers_slider.value()} 粒子 · "
            f"{self.fps.fps:.0f} fps"
        )

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


MainWindow = DLAApp

if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
