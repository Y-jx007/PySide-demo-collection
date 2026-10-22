from custom_import import *
from PySide6.QtCore import QPointF, QRectF

import numpy as np

real = ti.f32
dim = 2

# ==================== 参数（本构照 taichi_elements engine/mpm_solver.py） ====================
RES_DEFAULT = 128
RES_OPTIONS = (128, 192, 256, 384)
DENSITY_OPTIONS = (4, 8, 16, 24)
RES_MAX = max(RES_OPTIONS)
PADDING = 3
GRAVITY_Y = -9.8
SIZE = 1.0

MATERIAL_WATER = 0
MATERIAL_ELASTIC = 1
MATERIAL_SNOW = 2
MATERIAL_SAND = 3
MATERIAL_STATIONARY = 4

SUPPORT_PLASTICITY = True
WATER_DENSITY = 1.0
RENDER_RES = 768
MAX_PARTICLES = 600000
MAX_SUBSTEPS = 40             # 单帧子步上限，避免局部爆炸时卡死


@ti.data_oriented
class MPM2D:
    """2D MLS-MPM。网格分辨率与采样密度都是运行时参数，可随时重建。"""

    def __init__(self, res=RES_DEFAULT, density=4, max_particles=MAX_PARTICLES,
                 arch=None):
        ti.reset()
        ti.init(arch=arch if arch is not None else ti.gpu, default_fp=real)

        self.max_particles = max_particles
        self.padding = PADDING
        self.gsize = RES_MAX + 2 * PADDING

        n = max_particles
        self.n_particles = ti.field(ti.i32, shape=())
        self.x = ti.Vector.field(dim, real, n)
        self.v = ti.Vector.field(dim, real, n)
        self.F = ti.Matrix.field(dim, dim, real, n)
        self.C = ti.Matrix.field(dim, dim, real, n)
        self.Jp = ti.field(real, n)
        self.material = ti.field(ti.i32, n)

        self.grid_v = ti.Vector.field(dim, real, (self.gsize, self.gsize))
        self.grid_m = ti.field(real, (self.gsize, self.gsize))

        self.img = ti.Vector.field(3, real, (RENDER_RES, RENDER_RES))
        self.blur_tmp = ti.Vector.field(3, real, (RENDER_RES, RENDER_RES))

        self.par = ti.Vector.field(4, real, shape=())   # dx, inv_dx, p_vol, p_mass
        self.res_cur = ti.field(ti.i32, shape=())

        self.E = 1e6 * SIZE
        self.nu = 0.2
        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        import math
        sin_phi = math.sin(math.radians(45))
        self.alpha = math.sqrt(2.0 / 3.0) * 2 * sin_phi / (3 - sin_phi)

        self.rres = RENDER_RES
        self.dt_scale = 1.0
        self.allowed_cfl = 0.9
        self.configure(res, density)
        self._compile()

    def configure(self, res, density):
        self.res = int(res)
        self.density = int(density)
        self.dx = SIZE / res
        self.inv_dx = 1.0 / self.dx
        self.default_dt = 2e-2 * self.dx / SIZE
        self.p_vol = self.dx ** dim
        self.p_rho = 1000.0
        self.p_mass = self.p_vol * self.p_rho
        self.par[None] = [self.dx, self.inv_dx, self.p_vol, self.p_mass]
        self.res_cur[None] = self.res
        self.n_particles[None] = 0

    def _compile(self):
        gsize = self.gsize
        padding = self.padding

        @ti.kernel
        def clear_grid():
            for i, j in self.grid_v:
                self.grid_v[i, j] = ti.Vector([0.0, 0.0])
                self.grid_m[i, j] = 0.0

        @ti.kernel
        def p2g(dt: real):
            par = self.par[None]
            dx = par[0]
            inv_dx = par[1]
            p_vol = par[2]
            p_mass = par[3]
            mu_0 = self.mu_0
            la_0 = self.lambda_0
            for p in range(self.n_particles[None]):
                base = ti.cast(ti.floor(self.x[p] * inv_dx - 0.5), ti.i32)
                fx = self.x[p] * inv_dx - ti.cast(base, real)
                w = [0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1.0) ** 2,
                     0.5 * (fx - 0.5) ** 2]
                Fp = self.F[p]
                if self.material[p] == MATERIAL_WATER:
                    Fp = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
                    Fp[0, 0] = self.Jp[p]
                Fp = (ti.Matrix([[1.0, 0.0], [0.0, 1.0]]) + dt * self.C[p]) @ Fp

                h = 1.0
                if self.material[p] != MATERIAL_WATER:
                    h = ti.exp(10.0 * (1.0 - self.Jp[p]))
                if self.material[p] == MATERIAL_ELASTIC:
                    h = 0.3
                mu = mu_0 * h
                la = la_0 * h
                if self.material[p] == MATERIAL_WATER:
                    mu = 0.0

                U, sig, V = ti.svd(Fp)
                J = 1.0
                if self.material[p] != MATERIAL_SAND:
                    for d in ti.static(range(dim)):
                        new_sig = sig[d, d]
                        if self.material[p] == MATERIAL_SNOW:
                            new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
                        self.Jp[p] *= sig[d, d] / new_sig
                        sig[d, d] = new_sig
                        J *= new_sig

                if self.material[p] == MATERIAL_WATER:
                    Fp = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
                    Fp[0, 0] = J
                    self.Jp[p] = J
                elif self.material[p] == MATERIAL_SNOW:
                    Fp = U @ sig @ V.transpose()

                stress = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                if self.material[p] != MATERIAL_SAND:
                    stress = 2 * mu * (Fp - U @ V.transpose()) @ Fp.transpose() \
                        + ti.Matrix([[1.0, 0.0], [0.0, 1.0]]) * la * J * (J - 1)
                self.F[p] = Fp

                stress = (-dt * p_vol * 4 * inv_dx ** 2) * stress
                mass = p_mass
                if self.material[p] == MATERIAL_WATER:
                    mass *= WATER_DENSITY
                affine = stress + mass * self.C[p]

                for i, j in ti.static(ti.ndrange(3, 3)):
                    dpos = (ti.Vector([i, j]) - fx) * dx
                    weight = w[i][0] * w[j][1]
                    nb = base + ti.Vector([i, j]) + padding
                    if 0 <= nb[0] < gsize and 0 <= nb[1] < gsize:
                        ti.atomic_add(self.grid_v[nb],
                                      weight * (mass * self.v[p] + affine @ dpos))
                        ti.atomic_add(self.grid_m[nb], weight * mass)

        @ti.kernel
        def grid_op(dt: real):
            res = self.res_cur[None]
            for i, j in self.grid_m:
                if self.grid_m[i, j] > 0.0:
                    self.grid_v[i, j] = (1.0 / self.grid_m[i, j]) * self.grid_v[i, j]
                    self.grid_v[i, j] += dt * ti.Vector([0.0, GRAVITY_Y])
                xi = i - padding
                yj = j - padding
                if xi < padding and self.grid_v[i, j][0] < 0.0:
                    self.grid_v[i, j][0] = 0.0
                if xi >= res - padding and self.grid_v[i, j][0] > 0.0:
                    self.grid_v[i, j][0] = 0.0
                if yj < padding and self.grid_v[i, j][1] < 0.0:
                    self.grid_v[i, j][1] = 0.0
                if yj >= res - padding and self.grid_v[i, j][1] > 0.0:
                    self.grid_v[i, j][1] = 0.0

        @ti.kernel
        def g2p(dt: real):
            par = self.par[None]
            dx = par[0]
            inv_dx = par[1]
            for p in range(self.n_particles[None]):
                base = ti.cast(ti.floor(self.x[p] * inv_dx - 0.5), ti.i32)
                fx = self.x[p] * inv_dx - ti.cast(base, real)
                w = [0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1.0) ** 2,
                     0.5 * (fx - 0.5) ** 2]
                new_v = ti.Vector([0.0, 0.0])
                new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                for i, j in ti.static(ti.ndrange(3, 3)):
                    dpos = ti.Vector([i, j]) - fx
                    nb = base + ti.Vector([i, j]) + padding
                    if 0 <= nb[0] < gsize and 0 <= nb[1] < gsize:
                        g_v = self.grid_v[nb]
                        weight = w[i][0] * w[j][1]
                        new_v += weight * g_v
                        new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
                if self.material[p] != MATERIAL_STATIONARY:
                    self.v[p] = new_v
                    self.C[p] = new_C
                    self.x[p] += dt * self.v[p]

        @ti.kernel
        def max_velocity() -> real:
            vm = 0.0
            for p in range(self.n_particles[None]):
                vm = ti.max(vm, ti.abs(self.v[p][0]))
                vm = ti.max(vm, ti.abs(self.v[p][1]))
            return vm

        @ti.kernel
        def render(radius: ti.i32):
            rres = RENDER_RES
            for i, j in self.img:
                self.img[i, j] = ti.Vector([14.0, 17.0, 24.0])
            for p in range(self.n_particles[None]):
                px = self.x[p][0] * rres
                py = (1.0 - self.x[p][1]) * rres
                col = ti.Vector([235.0, 235.0, 240.0])
                if self.material[p] == MATERIAL_WATER:
                    col = ti.Vector([70.0, 140.0, 240.0])
                elif self.material[p] == MATERIAL_ELASTIC:
                    col = ti.Vector([240.0, 90.0, 70.0])
                elif self.material[p] == MATERIAL_SNOW:
                    col = ti.Vector([240.0, 248.0, 255.0])
                elif self.material[p] == MATERIAL_SAND:
                    col = ti.Vector([220.0, 190.0, 110.0])
                ix = ti.cast(px, ti.i32)
                iy = ti.cast(py, ti.i32)
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        if di * di + dj * dj <= radius * radius + 1:
                            ii = ix + di
                            jj = iy + dj
                            if 0 <= ii < rres and 0 <= jj < rres:
                                self.img[jj, ii] = col

        @ti.kernel
        def blur():
            rres = RENDER_RES
            for i, j in self.img:
                acc = self.img[i, j] * 4.0
                cnt = 4.0
                for di, dj in ti.static(ti.ndrange(-1, 2)):
                    ii = ti.min(ti.max(i + di, 0), rres - 1)
                    jj = ti.min(ti.max(j + dj, 0), rres - 1)
                    acc += self.img[ii, jj]
                    cnt += 1.0
                self.blur_tmp[i, j] = acc / cnt
            for i, j in self.img:
                self.img[i, j] = self.blur_tmp[i, j]

        self._clear_grid = clear_grid
        self._p2g = p2g
        self._grid_op = grid_op
        self._g2p = g2p
        self._max_velocity = max_velocity
        self._render = render
        self._blur = blur

    @ti.kernel
    def _seed_batch(self, count: ti.i32, new_material: ti.i32,
                    lx: real, ly: real, ux: real, uy: real,
                    vx: real, vy: real):
        # 一次 atomic 拿一整段下标，Python 侧不回读，避免每倒一次就同步一次
        base = ti.atomic_add(self.n_particles[None], count)
        for k in range(count):
            p = base + k
            if p < self.max_particles:
                rx = ti.random()
                ry = ti.random()
                self.x[p] = ti.Vector([lx + rx * (ux - lx), ly + ry * (uy - ly)])
                self.v[p] = ti.Vector([vx, vy])
                self.F[p] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
                self.C[p] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                self.material[p] = new_material
                if new_material == MATERIAL_SAND:
                    self.Jp[p] = 0.0
                else:
                    self.Jp[p] = 1.0

    def add_cube(self, lower, size, material, velocity=(0.0, 0.0),
                 sample_density=None):
        if sample_density is None:
            sample_density = self.density
        vol = size[0] * size[1]
        num_new = int(sample_density * vol / self.dx ** dim + 1)
        self._seed_batch(num_new, int(material),
                         float(lower[0]), float(lower[1]),
                         float(lower[0] + size[0]), float(lower[1] + size[1]),
                         float(velocity[0]), float(velocity[1]))
        return num_new

    def step(self, frame_dt):
        vmax = float(self._max_velocity())
        cfl_dt = self.allowed_cfl * self.dx / (vmax + 1e-6)
        dt = min(self.default_dt * self.dt_scale, cfl_dt)
        substeps = max(1, min(MAX_SUBSTEPS, int(frame_dt / dt) + 1))
        dt = frame_dt / substeps
        for _ in range(substeps):
            self._clear_grid()
            self._p2g(dt)
            self._grid_op(dt)
            self._g2p(dt)
        return substeps

    def particle_radius(self):
        """点半径按粒子间距自适应，避免大块圆点。"""
        spacing_px = (self.dx / max(1.0, (self.density ** 0.5))) * RENDER_RES
        return max(1, int(round(spacing_px * 0.35)))

    def draw(self, radius=None):
        if radius is None:
            radius = self.particle_radius()
        self._render(radius)
        self._blur()
        return np.clip(self.img.to_numpy(), 0, 255).astype(np.uint8)

    def particle_info(self):
        n = int(self.n_particles[None])
        return {
            "position": self.x.to_numpy()[:n],
            "velocity": self.v.to_numpy()[:n],
            "material": self.material.to_numpy()[:n],
        }


# ==================== 界面 ====================
MATERIAL_IDS = {
    "固体": MATERIAL_ELASTIC,
    "雪": MATERIAL_SNOW,
    "液体": MATERIAL_WATER,
    "沙子": MATERIAL_SAND,
}


class MPMCanvas(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumSize(520, 520)
        self.dragging = False

    def _to_world(self, pos):
        w, h = self.width(), self.height()
        side = min(w, h)
        ox = (w - side) / 2.0
        oy = (h - side) / 2.0
        return (pos.x() - ox) / side, 1.0 - (pos.y() - oy) / side

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(10, 12, 18))
        img = self.app.frame
        if img is None:
            p.end()
            return
        from PySide6.QtGui import QImage
        hr, wr = img.shape[:2]
        qimg = QImage(np.ascontiguousarray(img).data, wr, hr, wr * 3,
                      QImage.Format_RGB888)
        side = min(w, h)
        ox = (w - side) / 2.0
        oy = (h - side) / 2.0
        p.drawImage(QRectF(ox, oy, side, side), qimg)
        p.setPen(QPen(QColor(90, 100, 120, 120), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(ox, oy, side, side))
        p.end()

    def mousePressEvent(self, e):
        self.dragging = True
        self.app.pour(*self._to_world(e.position()))

    def mouseReleaseEvent(self, e):
        self.dragging = False

    def mouseMoveEvent(self, e):
        if self.dragging:
            self.app.pour(*self._to_world(e.position()))


class MPMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPM 2D · 固体 / 雪 / 液体")
        self.setGeometry(110, 70, 1120, 800)

        self.sim = MPM2D()
        self.material = "固体"
        self.paused = False
        self.frame = None
        self.frame_dt = 1.0 / 240.0
        self.fps = 0.0
        self._last_t = time.perf_counter()
        self._last_pour = 0.0
        self._stat_tick = 0
        self.substeps = 0

        self.build_scene()
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def build_scene(self):
        self.sim.configure(self.sim.res, self.sim.density)
        self.sim.add_cube(lower=(0.18, 0.62), size=(0.14, 0.14),
                          material=MATERIAL_ELASTIC)
        self.sim.add_cube(lower=(0.42, 0.62), size=(0.14, 0.14),
                          material=MATERIAL_SNOW)
        self.sim.add_cube(lower=(0.30, 0.22), size=(0.40, 0.20),
                          material=MATERIAL_WATER)
        self.frame = self.sim.draw()

    def pour(self, x, y):
        now = time.perf_counter()
        if now - self._last_pour < 0.10:
            return
        self._last_pour = now
        if not (0.05 < x < 0.95 and 0.05 < y < 0.95):
            return
        r = max(0.008, self.brush / 100.0 * 0.045)
        self.sim.add_cube(lower=(x - r, y - r), size=(2 * r, 2 * r),
                          material=MATERIAL_IDS[self.material],
                          velocity=(0.0, 0.0))

    def tick(self):
        if not self.paused:
            self.substeps = self.sim.step(self.frame_dt)
            self.frame = self.sim.draw()
            self.canvas.update()
            now = time.perf_counter()
            dt = now - self._last_t
            self._last_t = now
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, 1e-6))
        self._stat_tick += 1
        if self._stat_tick % 8 == 0:
            self.lbl_stat.setText(
                "网格 %d² · 采样 %d/格 · 粒子 %d · %d 子步/帧 · %.0f fps" % (
                    self.sim.res, self.sim.density, int(self.sim.n_particles[None]),
                    self.substeps, self.fps))

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        panel = QWidget()
        panel.setFixedWidth(252)
        pv = QVBoxLayout(panel)
        pv.setSpacing(8)
        pv.setContentsMargins(0, 0, 0, 0)

        title = QLabel("MPM 2D 材料")
        f = title.font()
        f.setPointSize(13)
        f.setBold(True)
        title.setFont(f)
        pv.addWidget(title)
        sub = QLabel("MLS-MPM / APIC，本构照 taichi_elements" + chr(10)
                     + "拖拽倾倒 · 分辨率和采样可调")
        sub.setStyleSheet("color: #7a7a80;")
        sf = sub.font()
        sf.setPointSize(8)
        sub.setFont(sf)
        pv.addWidget(sub)

        self.canvas = MPMCanvas(self)

        g0 = QGroupBox("精度")
        f0 = QFormLayout(g0)
        f0.setSpacing(6)
        f0.setContentsMargins(8, 8, 8, 8)
        self.res_box = QComboBox()
        for r in RES_OPTIONS:
            self.res_box.addItem("%d²" % r, r)
        self.res_box.setCurrentIndex(RES_OPTIONS.index(RES_DEFAULT))
        self.res_box.currentIndexChanged.connect(self.on_res)
        f0.addRow("网格", self.res_box)
        self.den_box = QComboBox()
        for d in DENSITY_OPTIONS:
            self.den_box.addItem("%d / 格" % d, d)
        self.den_box.setCurrentIndex(0)
        self.den_box.currentIndexChanged.connect(self.on_density)
        f0.addRow("采样", self.den_box)
        self.spd_box = QComboBox()
        for name, val in (("0.5×", 1 / 480), ("1×", 1 / 240), ("2×", 1 / 120)):
            self.spd_box.addItem(name, val)
        self.spd_box.setCurrentIndex(1)
        self.spd_box.currentIndexChanged.connect(self.on_speed)
        f0.addRow("时间步", self.spd_box)
        pv.addWidget(g0)

        g1 = QGroupBox("材料")
        v1 = QVBoxLayout(g1)
        v1.setSpacing(6)
        v1.setContentsMargins(8, 8, 8, 8)
        self.btns = {}
        for name in ("固体", "雪", "液体", "沙子"):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setChecked(name == "固体")
            b.clicked.connect(lambda _=False, n=name: self.select(n))
            self.btns[name] = b
            v1.addWidget(b)
        pv.addWidget(g1)

        g2 = QGroupBox("画笔")
        v2 = QVBoxLayout(g2)
        v2.setSpacing(6)
        v2.setContentsMargins(8, 8, 8, 8)
        self.brush = 45
        sl = QSlider(Qt.Horizontal)
        sl.setRange(5, 100)
        sl.setValue(45)
        sl.valueChanged.connect(lambda v: setattr(self, "brush", v))
        v2.addWidget(sl)
        row = QHBoxLayout()
        row.setSpacing(6)
        b_reset = QPushButton("重置场景")
        b_reset.clicked.connect(self.reset)
        row.addWidget(b_reset)
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.clicked.connect(self.toggle)
        row.addWidget(self.btn_pause)
        v2.addLayout(row)
        pv.addWidget(g2)
        pv.addStretch()

        root.addWidget(panel)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(0)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.canvas, 1)
        root.addWidget(right, 1)

        self.lbl_stat = QLabel("")
        self.statusBar().addWidget(self.lbl_stat, 1)

    # ---------- 选项 ----------
    def on_res(self, idx):
        self.sim.configure(RES_OPTIONS[idx], self.sim.density)
        self.build_scene()
        self.canvas.update()

    def on_density(self, idx):
        self.sim.configure(self.sim.res, DENSITY_OPTIONS[idx])
        self.build_scene()
        self.canvas.update()

    def on_speed(self, idx):
        self.frame_dt = self.spd_box.itemData(idx)

    def select(self, name):
        self.material = name
        for k, b in self.btns.items():
            b.setChecked(k == name)

    def reset(self):
        self.build_scene()
        self.canvas.update()

    def toggle(self):
        self.paused = not self.paused
        self.btn_pause.setText("继续" if self.paused else "暂停")

    def closeEvent(self, event):
        self.timer.stop()
        try:
            ti.reset()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MPMApp()
    window.show()
    sys.exit(app.exec())
