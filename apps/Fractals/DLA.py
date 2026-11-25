# -*- coding: utf-8 -*-
"""Extended diffusion-limited aggregation: coral, lichen, dendrite, crystal...

Growth model
------------
Walkers diffuse in a thin shell around the cluster and stick with a probability
that can be biased by direction (anisotropy), pulled toward the centre (bias)
or refused and slid along the surface (stick < 100%).  A freshly stuck particle
may relax to the neighbouring free site with the most occupied neighbours
(surface relaxation), which fills fjords and turns open DLA into compact
crystal-like growth.  Several nuclei can grow at once and merge.

Rendering stays on the GPU: one kernel maps the grid to canvas pixels
(bilinear, supersampled, continuous 256-level palette), so a frame costs a
single small device-to-host RGB copy.
"""
import numpy as np

from common.lab_window import LabWindow
from common.taichi_base import ensure_taichi
from common.qt_base import (
    QApplication, QColor, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QImage, QLabel, QMessageBox, QPainter, QPen, QPushButton,
    QSpinBox, QTimer, QVBoxLayout, QWidget, Qt, run_app, LabeledSlider,
)
from common.utils import FPSCounter

ti = ensure_taichi()

MAX_GRID = 2048
MAX_WALKERS = 32768
MAX_RES = 1200
GRID_SIZES = (512, 1024, 2048)
ORDERS = (3, 4, 5, 6, 8)
CANVAS_SIZE = 600

# walkers, substeps, stick%, anisotropy%, order, relax%, bias%, drift%,
# nuclei, nucleus radius, view half-size in cells (for a 1024 grid)
PRESETS = {
    "珊瑚": dict(walkers=1024, speed=512, stick=100, aniso=30, kfold=5,
                 relax=0, bias=0, drift=0, seeds=1, seed_r=0, view=260),
    "地衣": dict(walkers=4096, speed=128, stick=12, aniso=0, kfold=5,
                 relax=50, bias=0, drift=0, seeds=6, seed_r=16, view=180),
    "枝晶": dict(walkers=768, speed=512, stick=100, aniso=85, kfold=6,
                 relax=0, bias=0, drift=0, seeds=1, seed_r=0, view=250),
    "雪花": dict(walkers=1024, speed=384, stick=85, aniso=75, kfold=6,
                 relax=55, bias=8, drift=0, seeds=1, seed_r=0, view=400),
    "水晶": dict(walkers=1536, speed=256, stick=90, aniso=65, kfold=4,
                 relax=60, bias=20, drift=0, seeds=1, seed_r=0, view=505),
    "海扇": dict(walkers=1024, speed=384, stick=100, aniso=50, kfold=3,
                 relax=50, bias=0, drift=0, seeds=1, seed_r=0, view=215),
    "玛瑙": dict(walkers=2048, speed=192, stick=100, aniso=0, kfold=5,
                 relax=55, bias=30, drift=0, seeds=1, seed_r=0, view=514),
    "海藻": dict(walkers=512, speed=640, stick=100, aniso=0, kfold=5,
                 relax=20, bias=0, drift=35, seeds=1, seed_r=0, view=360),
}
PRESET_NAMES = tuple(PRESETS)

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
        self.seeds = 1
        self.seed_r = 0
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
    def _reset_kernel(self, size: ti.i32, seeds: ti.i32, seed_r: ti.i32):
        for i, j in ti.ndrange(size, size):
            self.occ[i, j] = ti.cast(0, ti.u8)
            self.age[i, j] = 0
        for i in range(MAX_WALKERS):
            self.alive[i] = ti.cast(0, ti.u8)
        c = size // 2
        for _ in range(seeds):
            ang = ti.random() * 6.283185307
            rad = ti.sqrt(ti.random()) * ti.cast(seed_r, ti.f32)
            x = ti.max(1, ti.min(size - 2, ti.cast(ti.cast(c, ti.f32) +
                                                  ti.cos(ang) * rad, ti.i32)))
            y = ti.max(1, ti.min(size - 2, ti.cast(ti.cast(c, ti.f32) +
                                                  ti.sin(ang) * rad, ti.i32)))
            self.occ[x, y] = ti.cast(1, ti.u8)
            self.age[x, y] = 1
        self.occ[c, c] = ti.cast(1, ti.u8)
        self.age[c, c] = 1
        self.max_r[None] = seed_r
        self.steps[None] = 0

    def reset(self, size=None, seeds=None, seed_r=None):
        if size is not None:
            self.size = int(size)
        if seeds is not None:
            self.seeds = int(seeds)
        if seed_r is not None:
            self.seed_r = int(seed_r)
        self._reset_kernel(self.size, self.seeds, self.seed_r)

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
    def _spawn(self, i: ti.i32, size: ti.i32) -> ti.i32:
        c = size // 2
        limit = size // 2 - 3
        gap = ti.max(4, size // 160)
        rad = ti.min(limit, self.max_r[None] + gap)
        ang = ti.random() * 6.283185307
        x = ti.cast(ti.cast(c, ti.f32) + ti.cos(ang) * ti.cast(rad, ti.f32), ti.i32)
        y = ti.cast(ti.cast(c, ti.f32) + ti.sin(ang) * ti.cast(rad, ti.f32), ti.i32)
        x = ti.max(1, ti.min(size - 2, x))
        y = ti.max(1, ti.min(size - 2, y))
        ok = 0
        if self.occ[x, y] == 0:
            self.wx[i] = x
            self.wy[i] = y
            ok = 1
        return ok

    @ti.func
    def _neighbours(self, x: ti.i32, y: ti.i32, size: ti.i32) -> ti.i32:
        n = 0
        for dy in ti.static(range(-1, 2)):
            for dx in ti.static(range(-1, 2)):
                if dx != 0 or dy != 0:
                    xx = x + dx
                    yy = y + dy
                    if xx >= 0 and xx < size and yy >= 0 and yy < size:
                        if self.occ[xx, yy] != 0:
                            n += 1
        return n

    @ti.kernel
    def step(self, size: ti.i32, substeps: ti.i32, stick: ti.f32, drift: ti.f32,
             aniso: ti.f32, kfold: ti.f32, relax: ti.f32, bias: ti.f32):
        c = ti.cast(size // 2, ti.f32)
        killpad = ti.max(24, size // 24)
        for i in range(MAX_WALKERS):
            if i >= self.n_walkers[None]:
                self.alive[i] = ti.cast(0, ti.u8)
            else:
                state = ti.cast(self.alive[i], ti.i32)
                for _ in range(substeps):
                    if state == 0:
                        state = self._spawn(i, size)
                    if state == 0:
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
                    if ti.random() < bias:
                        # Inward step, axis chosen by distance so the bias does
                        # not lock onto the x/y axes.
                        ax = ti.abs(ti.cast(x, ti.f32) - c)
                        ay = ti.abs(ti.cast(y, ti.f32) - c)
                        if ti.random() * (ax + ay + 1e-6) < ax:
                            dx = -1 if ti.cast(x, ti.f32) > c else 1
                            dy = 0
                        else:
                            dx = 0
                            dy = -1 if ti.cast(y, ti.f32) > c else 1
                    nx = x + dx
                    ny = y + dy
                    if nx < 1 or nx > size - 2 or ny < 1 or ny > size - 2:
                        state = 0
                        continue
                    if self.occ[nx, ny] != 0:
                        ang = ti.atan2(ti.cast(y, ti.f32) - c, ti.cast(x, ti.f32) - c)
                        p = stick * (1.0 + aniso * ti.cos(kfold * ang))
                        if ti.random() < ti.min(1.0, ti.max(0.0, p)):
                            px = x
                            py = y
                            if ti.random() < relax:
                                # Surface relaxation: move to the free neighbour
                                # with the most occupied neighbours (fills
                                # fjords instead of extending tips).
                                best = self._neighbours(x, y, size)
                                bx = x
                                by = y
                                for ddy in ti.static(range(-1, 2)):
                                    for ddx in ti.static(range(-1, 2)):
                                        if ddx != 0 or ddy != 0:
                                            xx = x + ddx
                                            yy = y + ddy
                                            if xx >= 1 and xx <= size - 2 and yy >= 1 and yy <= size - 2:
                                                if self.occ[xx, yy] == 0:
                                                    cnt = self._neighbours(xx, yy, size)
                                                    if cnt > best:
                                                        best = cnt
                                                        bx = xx
                                                        by = yy
                                px = bx
                                py = by
                            if self.occ[px, py] == 0:
                                self.occ[px, py] = ti.cast(1, ti.u8)
                                self.age[px, py] = self.steps[None]
                                rr = ti.cast(ti.sqrt((px - c) * (px - c) +
                                                     (py - c) * (py - c)), ti.i32)
                                ti.atomic_max(self.max_r[None], rr)
                            state = 0
                        else:
                            # Non-sticking: slide away instead of waiting on the
                            # same site, so the probability changes morphology.
                            bx = x - dx
                            by = y - dy
                            if bx >= 1 and bx <= size - 2 and by >= 1 and by <= size - 2:
                                if self.occ[bx, by] == 0:
                                    self.wx[i] = bx
                                    self.wy[i] = by
                    else:
                        self.wx[i] = nx
                        self.wy[i] = ny
                        ux = nx - size // 2
                        uy = ny - size // 2
                        shell = self.max_r[None] + killpad
                        if ux * ux + uy * uy > shell * shell:
                            state = 0
                self.alive[i] = ti.cast(state, ti.u8)
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
    """Canvas with manual camera: wheel zooms at cursor, drag pans."""

    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.setFixedSize(600, 600)
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip("滚轮缩放 · 左键拖动平移 · 双击适应团簇")
        self._image = None
        self._buffer = None
        self.dpr = sim.res / float(CANVAS_SIZE)
        self.view_cx = sim.size * 0.5
        self.view_cy = sim.size * 0.5
        self.view_half = sim.size * 0.5 + 2.0
        self.supersample = 2
        self._press = None
        self._press_center = None

    # ----------------------------------------------------------------- camera
    def set_view(self, half):
        """Frame the scene once (used on reset / preset load, never per frame)."""
        self.view_half = float(half)
        self.view_cx = self.sim.size * 0.5
        self.view_cy = self.sim.size * 0.5
        self._clamp_view()

    def reset_view(self, fit=False):
        sim = self.sim
        if fit:
            radius = float(sim.max_r[None])
            self.view_half = max(24.0, radius * 1.12 + 8.0)
        else:
            self.view_half = sim.size * 0.5 + 2.0
        self.view_cx = sim.size * 0.5
        self.view_cy = sim.size * 0.5
        self._clamp_view()

    def _clamp_view(self):
        size = float(self.sim.size)
        half = min(max(self.view_half, 16.0), size * 0.75)
        self.view_half = half
        self.view_cx = min(max(self.view_cx, half * 0.5), size - half * 0.5)
        self.view_cy = min(max(self.view_cy, half * 0.5), size - half * 0.5)

    def zoom_factor(self):
        return (self.sim.size * 0.5) / max(1.0, self.view_half)

    def _zoom_at(self, px, py, steps):
        w = float(self.width())
        fx = px / w - 0.5
        fy = py / w - 0.5
        wx = self.view_cx + fx * 2.0 * self.view_half
        wy = self.view_cy + fy * 2.0 * self.view_half
        self.view_half *= 0.84 ** steps
        self._clamp_view()
        self.view_cx = wx - fx * 2.0 * self.view_half
        self.view_cy = wy - fy * 2.0 * self.view_half
        self._clamp_view()
        self.render_frame()

    def wheelEvent(self, event):
        self._zoom_at(event.position().x(), event.position().y(),
                      event.angleDelta().y() / 120.0)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press = event.position()
            self._press_center = (self.view_cx, self.view_cy)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self._press is None:
            return
        w = float(self.width())
        dx = event.position().x() - self._press.x()
        dy = event.position().y() - self._press.y()
        scale = 2.0 * self.view_half / w
        self.view_cx = self._press_center[0] - dx * scale
        self.view_cy = self._press_center[1] - dy * scale
        self._clamp_view()
        self.render_frame()

    def mouseReleaseEvent(self, event):
        self._press = None
        self.setCursor(Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, event):
        self.reset_view(fit=True)
        self.render_frame()

    # ------------------------------------------------------------- view / draw
    def render_frame(self):
        sim = self.sim
        self.supersample = 1
        if sim.size >= 900:
            self.supersample = 2
        if sim.size >= 1600:
            self.supersample = 3
        data = sim.render_numpy(self.view_cx, self.view_cy, self.view_half,
                                self.supersample)
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
        self.sim.set_walkers(PRESETS["珊瑚"]["walkers"])
        self.canvas = DLACanvas(self.sim)
        self.view_half_cells = float(PRESETS["珊瑚"]["view"])
        self._build_ui()
        self.fps = FPSCounter()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)
        self.reset()
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
        col.setSpacing(5)

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
        form.setContentsMargins(8, 5, 8, 5)
        form.setSpacing(3)
        self.grid_combo = QComboBox()
        self.grid_combo.addItems([str(n) for n in GRID_SIZES])
        self.grid_combo.setMinimumWidth(110)
        self.grid_combo.setCurrentText("1024")
        self.grid_combo.currentTextChanged.connect(self.change_grid)
        form.addRow("网格", self.grid_combo)

        self.walkers_slider = LabeledSlider(
            "", 128, MAX_WALKERS, 1024, formatter=lambda v: f"{v}", value_width=56
        )
        self.walkers_slider.valueChanged.connect(self.change_walkers)
        form.addRow("游走粒子", self.walkers_slider)

        self.speed_slider = LabeledSlider(
            "", 8, 768, 512, formatter=lambda v: f"{v}×", value_width=44
        )
        self.speed_slider.valueChanged.connect(self.change_speed)
        form.addRow("每帧步数", self.speed_slider)

        self.stick_slider = LabeledSlider(
            "", 5, 100, 100, formatter=lambda v: f"{v}%", value_width=46
        )
        self.stick_slider.valueChanged.connect(self.change_stick)
        form.addRow("黏附概率", self.stick_slider)

        self.aniso_slider = LabeledSlider(
            "", 0, 100, 30, formatter=lambda v: f"{v}%", value_width=46
        )
        self.aniso_slider.valueChanged.connect(self.change_aniso)
        form.addRow("各向异性", self.aniso_slider)

        self.relax_slider = LabeledSlider(
            "", 0, 100, 0, formatter=lambda v: f"{v}%", value_width=46
        )
        self.relax_slider.valueChanged.connect(self.change_relax)
        form.addRow("表面弛豫", self.relax_slider)

        self.bias_slider = LabeledSlider(
            "", 0, 100, 0, formatter=lambda v: f"{v}%", value_width=46
        )
        self.bias_slider.valueChanged.connect(self.change_bias)
        form.addRow("趋心偏置", self.bias_slider)

        self.drift_slider = LabeledSlider(
            "", -100, 100, 0, formatter=lambda v: f"{v}%", value_width=46
        )
        self.drift_slider.valueChanged.connect(self.change_drift)
        form.addRow("风向漂移", self.drift_slider)

        order_row = QWidget()
        order_layout = QHBoxLayout(order_row)
        order_layout.setContentsMargins(0, 0, 0, 0)
        order_layout.setSpacing(6)
        self.order_combo = QComboBox()
        self.order_combo.addItems([str(k) for k in ORDERS])
        self.order_combo.setCurrentText("5")
        self.order_combo.currentTextChanged.connect(self.change_order)
        self.nuclei_spin = QSpinBox()
        self.nuclei_spin.setRange(1, 12)
        self.nuclei_spin.setValue(1)
        self.nuclei_spin.setSuffix(" 核")
        self.nuclei_spin.valueChanged.connect(self.change_nuclei)
        order_layout.addWidget(self.order_combo, 1)
        order_layout.addWidget(self.nuclei_spin, 1)
        form.addRow("取向 / 核数", order_row)
        col.addWidget(params)

        presets = QGroupBox("预设")
        pv = QVBoxLayout(presets)
        pv.setContentsMargins(6, 5, 6, 5)
        pv.setSpacing(4)
        rows = (PRESET_NAMES[:4], PRESET_NAMES[4:])
        for names in rows:
            h = QHBoxLayout()
            h.setSpacing(4)
            for name in names:
                b = QPushButton(name)
                b.setMinimumWidth(0)
                b.clicked.connect(lambda _=False, n=name: self.apply_preset(n))
                h.addWidget(b)
            pv.addLayout(h)
        col.addWidget(presets)

        stats = QGroupBox("读数")
        sv = QVBoxLayout(stats)
        sv.setContentsMargins(8, 5, 8, 4)
        sv.setSpacing(2)
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
        iv.setContentsMargins(8, 5, 8, 5)
        txt = QLabel(
            "弛豫＝填谷致密，趋心＝向内压缩，取向＝晶向对称，核数＝多核并合；"
            "滚轮缩放，拖动平移，双击适应。"
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

        self.steps_per_frame = 512
        self.stick = 1.0
        self.drift = 0.0
        self.aniso = 0.30
        self.kfold = 5.0
        self.relax = 0.0
        self.bias = 0.0

    # ------------------------------------------------------------------- loop
    def tick(self):
        if self.sim is None:
            return
        self.sim.step(self.sim.size, self.steps_per_frame, self.stick,
                      self.drift, self.aniso, self.kfold, self.relax, self.bias)
        self.canvas.render_frame()
        self.frame += 1
        if self.frame % 8 == 0:
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
        self.sim.reset(self.sim.size, self.sim.seeds, self.sim.seed_r)
        self.sim.set_walkers(self.walkers_slider.value())
        self.canvas.set_view(self.view_half_cells * self.sim.size / 1024.0)
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

    def apply_preset(self, name):
        cfg = PRESETS[name]
        self.sim.seeds = cfg["seeds"]
        self.sim.seed_r = cfg["seed_r"]
        self.view_half_cells = float(cfg["view"])
        self.kfold = float(cfg["kfold"])
        self.steps_per_frame = int(cfg["speed"])
        self.stick = max(0.05, cfg["stick"] / 100.0)
        self.aniso = cfg["aniso"] / 100.0
        self.relax = cfg["relax"] / 100.0
        self.bias = cfg["bias"] / 100.0
        self.drift = cfg["drift"] / 100.0
        self.walkers_slider.setValue(cfg["walkers"])
        self.speed_slider.setValue(cfg["speed"])
        self.stick_slider.setValue(cfg["stick"])
        self.aniso_slider.setValue(cfg["aniso"])
        self.relax_slider.setValue(cfg["relax"])
        self.bias_slider.setValue(cfg["bias"])
        self.drift_slider.setValue(cfg["drift"])
        self.order_combo.setCurrentText(str(cfg["kfold"]))
        self.nuclei_spin.setValue(cfg["seeds"])
        self.reset()

    def change_grid(self, text):
        n = int(text)
        self.sim.reset(n, self.sim.seeds, self.sim.seed_r)
        self.sim.set_walkers(self.walkers_slider.value())
        self.canvas.set_view(self.view_half_cells * self.sim.size / 1024.0)
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

    def change_aniso(self, value):
        self.aniso = float(value) / 100.0

    def change_relax(self, value):
        self.relax = float(value) / 100.0

    def change_bias(self, value):
        self.bias = float(value) / 100.0

    def change_drift(self, value):
        self.drift = float(value) / 100.0

    def change_order(self, text):
        self.kfold = float(text)

    def change_nuclei(self, value):
        self.sim.seeds = int(value)

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
            f"×{self.canvas.zoom_factor():.1f} · {self.fps.fps:.0f} fps"
        )

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


MainWindow = DLAApp

if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
