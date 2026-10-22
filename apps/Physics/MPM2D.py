from custom_import import *
from PySide6.QtCore import QPointF, QRectF

import numpy as np

real = ti.f32
dim = 2

# ==================== 参数（严格照 taichi_elements engine/mpm_solver.py 的 2D 路径） ====================
RES = 128
SIZE = 1.0
MAX_PARTICLES = 120000
PADDING = 3
GRAVITY_Y = -9.8

# 材料编号，和参考一致
MATERIAL_WATER = 0
MATERIAL_ELASTIC = 1
MATERIAL_SNOW = 2
MATERIAL_SAND = 3
MATERIAL_STATIONARY = 4

WATER_DENSITY = 1.0
SUPPORT_PLASTICITY = True
RENDER_RES = 512             # 渲染缓冲分辨率（和物理解析度无关）
SAMPLE_DENSITY = 4           # 每格采样数，必须保持参考的 2^dim=4，否则初始过压会炸开


@ti.data_oriented
class MPM2D:
    """2D MLS-MPM，材料模型照搬 taichi_elements。"""

    def __init__(self, res=RES, size=SIZE, max_particles=MAX_PARTICLES,
                 arch=None, dt_scale=1.0, allowed_cfl=0.9):
        ti.reset()
        ti.init(arch=arch if arch is not None else ti.gpu, default_fp=real)

        self.res = res
        self.dx = size / res
        self.inv_dx = 1.0 / self.dx
        self.size = size
        self.default_dt = 2e-2 * self.dx / size
        self.p_vol = self.dx ** dim
        self.p_rho = 1000.0
        self.p_mass = self.p_vol * self.p_rho
        self.max_particles = max_particles
        self.padding = PADDING
        self.dt_scale = dt_scale
        self.allowed_cfl = allowed_cfl

        # 弹性参数
        self.E = 1e6 * size
        self.nu = 0.2
        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        # 沙子参数（这里不用，但保留给沙材料）
        import math
        sin_phi = math.sin(math.radians(45))
        self.alpha = math.sqrt(2.0 / 3.0) * 2 * sin_phi / (3 - sin_phi)

        n = max_particles
        self.n_particles = ti.field(ti.i32, shape=())
        self.x = ti.Vector.field(dim, real, n)
        self.v = ti.Vector.field(dim, real, n)
        self.F = ti.Matrix.field(dim, dim, real, n)
        self.C = ti.Matrix.field(dim, dim, real, n)
        self.Jp = ti.field(real, n)
        self.material = ti.field(ti.i32, n)

        # 网格比物理区域大 PADDING 圈，避免边界处模板越界丢质量
        g = res + 2 * self.padding
        self.gsize = g
        self.grid_v = ti.Vector.field(dim, real, (g, g))
        self.grid_m = ti.field(real, (g, g))

        # 图像缓冲：分辨率和物理解析度解耦，画得细
        self.rres = RENDER_RES
        self.img = ti.Vector.field(3, real, (RENDER_RES, RENDER_RES))
        self.blur_tmp = ti.Vector.field(3, real, (RENDER_RES, RENDER_RES))

        self._compile()

    # ---------- 内核 ----------
    def _compile(self):
        res, dx, inv_dx, p_vol, p_mass = self.res, self.dx, self.inv_dx, self.p_vol, self.p_mass
        mu_0, lambda_0, alpha = self.mu_0, self.lambda_0, self.alpha
        n_part = self.n_particles
        x, v, F, C, Jp, mat = self.x, self.v, self.F, self.C, self.Jp, self.material
        gv, gm = self.grid_v, self.grid_m
        padding = self.padding
        gsize = self.gsize
        rres = self.rres
        img = self.img
        blur_tmp = self.blur_tmp
        water_density = WATER_DENSITY

        @ti.kernel
        def clear_grid():
            for i, j in gv:
                gv[i, j] = ti.Vector([0.0, 0.0])
                gm[i, j] = 0.0

        @ti.kernel
        def p2g(dt: real):
            for p in range(n_part[None]):
                base = ti.cast(ti.floor(x[p] * inv_dx - 0.5), ti.i32)
                fx = x[p] * inv_dx - ti.cast(base, real)
                w = [0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1.0) ** 2,
                     0.5 * (fx - 0.5) ** 2]
                Fp = F[p]
                if mat[p] == MATERIAL_WATER:
                    Fp = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
                    Fp[0, 0] = Jp[p]
                Fp = (ti.Matrix([[1.0, 0.0], [0.0, 1.0]]) + dt * C[p]) @ Fp

                h = 1.0
                if ti.static(SUPPORT_PLASTICITY):
                    if mat[p] != MATERIAL_WATER:
                        h = ti.exp(10.0 * (1.0 - Jp[p]))
                if mat[p] == MATERIAL_ELASTIC:
                    h = 0.3
                mu, la = mu_0 * h, lambda_0 * h
                if mat[p] == MATERIAL_WATER:
                    mu = 0.0

                U, sig, V = ti.svd(Fp)
                J = 1.0
                if mat[p] != MATERIAL_SAND:
                    for d in ti.static(range(dim)):
                        new_sig = sig[d, d]
                        if mat[p] == MATERIAL_SNOW:
                            new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
                        if ti.static(SUPPORT_PLASTICITY):
                            Jp[p] *= sig[d, d] / new_sig
                        sig[d, d] = new_sig
                        J *= new_sig

                if mat[p] == MATERIAL_WATER:
                    Fp = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
                    Fp[0, 0] = J
                    if ti.static(SUPPORT_PLASTICITY):
                        Jp[p] = J
                elif mat[p] == MATERIAL_SNOW:
                    Fp = U @ sig @ V.transpose()

                stress = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                if mat[p] != MATERIAL_SAND:
                    stress = 2 * mu * (Fp - U @ V.transpose()) @ Fp.transpose() \
                        + ti.Matrix([[1.0, 0.0], [0.0, 1.0]]) * la * J * (J - 1)
                F[p] = Fp

                stress = (-dt * p_vol * 4 * inv_dx ** 2) * stress
                mass = p_mass
                if mat[p] == MATERIAL_WATER:
                    mass *= water_density
                affine = stress + mass * C[p]

                for i, j in ti.static(ti.ndrange(3, 3)):
                    dpos = (ti.Vector([i, j]) - fx) * dx
                    weight = w[i][0] * w[j][1]
                    nb = base + ti.Vector([i, j]) + self.padding
                    if 0 <= nb[0] < gsize and 0 <= nb[1] < gsize:
                        ti.atomic_add(gv[nb], weight * (mass * v[p] + affine @ dpos))
                        ti.atomic_add(gm[nb], weight * mass)

        @ti.kernel
        def grid_op(dt: real):
            for i, j in gm:
                if gm[i, j] > 0.0:
                    gv[i, j] = (1.0 / gm[i, j]) * gv[i, j]
                    gv[i, j] += dt * ti.Vector([0.0, GRAVITY_Y])
                xi = i - padding
                yj = j - padding
                if xi < padding and gv[i, j][0] < 0.0:
                    gv[i, j][0] = 0.0
                if xi >= res - padding and gv[i, j][0] > 0.0:
                    gv[i, j][0] = 0.0
                if yj < padding and gv[i, j][1] < 0.0:
                    gv[i, j][1] = 0.0
                if yj >= res - padding and gv[i, j][1] > 0.0:
                    gv[i, j][1] = 0.0

        @ti.kernel
        def g2p(dt: real):
            for p in range(n_part[None]):
                base = ti.cast(ti.floor(x[p] * inv_dx - 0.5), ti.i32)
                fx = x[p] * inv_dx - ti.cast(base, real)
                w = [0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1.0) ** 2,
                     0.5 * (fx - 0.5) ** 2]
                new_v = ti.Vector([0.0, 0.0])
                new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                for i, j in ti.static(ti.ndrange(3, 3)):
                    dpos = ti.Vector([i, j]) - fx
                    nb = base + ti.Vector([i, j]) + self.padding
                    if 0 <= nb[0] < gsize and 0 <= nb[1] < gsize:
                        g_v = gv[nb]
                        weight = w[i][0] * w[j][1]
                        new_v += weight * g_v
                        new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
                if mat[p] != MATERIAL_STATIONARY:
                    v[p] = new_v
                    C[p] = new_C
                    x[p] += dt * v[p]

        # ---------- 渲染：512² 缓冲，粒子画成小圆盘再模糊 ----------
        @ti.kernel
        def render(radius: ti.i32):
            for i, j in img:
                img[i, j] = ti.Vector([14.0, 17.0, 24.0])
            for p in range(n_part[None]):
                px = x[p][0] * rres
                py = (1.0 - x[p][1]) * rres
                col = ti.Vector([235.0, 235.0, 240.0])
                if mat[p] == MATERIAL_WATER:
                    col = ti.Vector([70.0, 140.0, 240.0])
                elif mat[p] == MATERIAL_ELASTIC:
                    col = ti.Vector([240.0, 90.0, 70.0])
                elif mat[p] == MATERIAL_SNOW:
                    col = ti.Vector([240.0, 248.0, 255.0])
                elif mat[p] == MATERIAL_SAND:
                    col = ti.Vector([220.0, 190.0, 110.0])
                ix = ti.cast(px, ti.i32)
                iy = ti.cast(py, ti.i32)
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        if di * di + dj * dj <= radius * radius + 1:
                            ii = ix + di
                            jj = iy + dj
                            if 0 <= ii < rres and 0 <= jj < rres:
                                img[jj, ii] = col

        @ti.kernel
        def blur():
            for i, j in img:
                acc = img[i, j] * 4.0
                cnt = 4.0
                for di, dj in ti.static(ti.ndrange(-1, 2)):
                    ii = ti.min(ti.max(i + di, 0), rres - 1)
                    jj = ti.min(ti.max(j + dj, 0), rres - 1)
                    acc += img[ii, jj]
                    cnt += 1.0
                blur_tmp[i, j] = acc / cnt
            for i, j in img:
                img[i, j] = blur_tmp[i, j]

        @ti.kernel
        def max_velocity() -> real:
            vm = 0.0
            for p in range(n_part[None]):
                vm = ti.max(vm, ti.abs(v[p][0]))
                vm = ti.max(vm, ti.abs(v[p][1]))
            return vm

        self._max_velocity = max_velocity
        self._clear_grid = clear_grid
        self._p2g = p2g
        self._grid_op = grid_op
        self._g2p = g2p
        self._render = render
        self._blur = blur

    # ---------- 加粒子（照参考的 add_cube） ----------
    @ti.kernel
    def _seed(self, start: ti.i32, count: ti.i32, new_material: ti.i32,
              lower: ti.template(), upper: ti.template(), vel: ti.template(),
              sample_density: ti.i32):
        for k in range(count):
            p = start + k
            rx = ti.random()
            ry = ti.random()
            self.x[p] = ti.Vector([lower[0] + rx * (upper[0] - lower[0]),
                                   lower[1] + ry * (upper[1] - lower[1])])
            self.v[p] = ti.Vector([vel[0], vel[1]])
            self.F[p] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            self.C[p] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            self.material[p] = new_material
            if ti.static(SUPPORT_PLASTICITY):
                if new_material == MATERIAL_SAND:
                    self.Jp[p] = 0.0
                else:
                    self.Jp[p] = 1.0

    def add_cube(self, lower, size, material, velocity=(0.0, 0.0),
                 sample_density=None):
        if sample_density is None:
            sample_density = SAMPLE_DENSITY
        vol = size[0] * size[1]
        num_new = int(sample_density * vol / self.dx ** dim + 1)
        start = int(self.n_particles[None])
        if start + num_new > self.max_particles:
            return 0
        lower_v = ti.Vector([float(lower[0]), float(lower[1])])
        upper_v = ti.Vector([float(lower[0] + size[0]), float(lower[1] + size[1])])
        vel_v = ti.Vector([float(velocity[0]), float(velocity[1])])
        self._seed(start, num_new, int(material), lower_v, upper_v, vel_v,
                   int(sample_density))
        self.n_particles[None] = start + num_new
        return num_new

    # ---------- 推进（照参考的 step） ----------
    def step(self, frame_dt):
        # 自适应 dt：CFL 上限（参考里的 use_adaptive_dt）
        vmax = float(self._max_velocity())
        cfl_dt = self.allowed_cfl * self.dx / (vmax + 1e-6)
        dt = min(self.default_dt * self.dt_scale, cfl_dt)
        substeps = max(1, int(frame_dt / dt) + 1)
        dt = frame_dt / substeps
        for _ in range(substeps):
            self._clear_grid()
            self._p2g(dt)
            self._grid_op(dt)
            self._g2p(dt)
        return substeps

    def draw(self, radius=2):
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
MATERIAL_COLORS = {
    "固体": (240, 90, 70),
    "雪": (235, 245, 255),
    "液体": (70, 140, 240),
    "沙子": (220, 190, 110),
}
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
        self.setMouseTracking(True)
        self.dragging = False

    def _to_world(self, pos):
        w = self.width()
        h = self.height()
        side = min(w, h)
        ox = (w - side) / 2.0
        oy = (h - side) / 2.0
        x = (pos.x() - ox) / side
        y = 1.0 - (pos.y() - oy) / side
        return x, y

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
        qimg = QImage(np.ascontiguousarray(img).data, wr, hr, wr * 3, QImage.Format_RGB888)
        side = min(w, h)
        ox = (w - side) / 2.0
        oy = (h - side) / 2.0
        p.drawImage(QRectF(ox, oy, side, side), qimg)
        # 边框
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
        self.setGeometry(110, 70, 1100, 780)

        self.sim = MPM2D()
        self.material = "固体"
        self.paused = False
        self.frame = None
        self.frame_dt = 1.0 / 480.0
        self.fps = 0.0
        self._last_t = time.perf_counter()
        self._last_pour = 0.0
        self._stat_tick = 0

        self.build_scene()
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    # ---------- 场景 ----------
    def build_scene(self):
        self.sim = MPM2D()
        self.sim.add_cube(lower=(0.18, 0.62), size=(0.14, 0.14),
                          material=MATERIAL_ELASTIC)
        self.sim.add_cube(lower=(0.42, 0.62), size=(0.14, 0.14),
                          material=MATERIAL_SNOW)
        self.sim.add_cube(lower=(0.30, 0.22), size=(0.40, 0.20),
                          material=MATERIAL_WATER)
        self.frame = self.sim.draw(radius=2)

    def pour(self, x, y):
        now = time.perf_counter()
        if now - self._last_pour < 0.05:      # 最多 20 次/秒，否则一下子灌太多粒子
            return
        self._last_pour = now
        if not (0.05 < x < 0.95 and 0.05 < y < 0.95):
            return
        r = self.brush / 100.0 * 0.06
        r = max(0.01, r)
        self.sim.add_cube(lower=(x - r, y - r), size=(2 * r, 2 * r),
                          material=MATERIAL_IDS[self.material],
                          velocity=(0.0, -0.5))

    # ---------- 主循环 ----------
    def tick(self):
        if not self.paused:
            self.sim.step(self.frame_dt)
            self.frame = self.sim.draw(radius=2)
            self.canvas.update()
            now = time.perf_counter()
            dt = now - self._last_t
            self._last_t = now
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, 1e-6))
        self._stat_tick += 1
        if self._stat_tick % 10 == 0:          # 每 10 帧才回读粒子数，避免每帧同步 GPU
            self.lbl_stat.setText(
                "粒子 %d · %.0f fps · 每帧 %.4f s 仿真" % (
                    int(self.sim.n_particles[None]), self.fps, self.frame_dt))

    # ---------- UI ----------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        panel = QWidget()
        panel.setFixedWidth(250)
        pv = QVBoxLayout(panel)
        pv.setSpacing(8)
        pv.setContentsMargins(0, 0, 0, 0)

        title = QLabel("MPM 2D 材料")
        f = title.font()
        f.setPointSize(13)
        f.setBold(True)
        title.setFont(f)
        pv.addWidget(title)

        sub = QLabel("MLS-MPM / APIC，材料模型照 taichi_elements\n"
                     "在画布上按住拖拽即可倾倒")
        sub.setStyleSheet("color: #7a7a80;")
        sf = sub.font()
        sf.setPointSize(8)
        sub.setFont(sf)
        pv.addWidget(sub)

        self.canvas = MPMCanvas(self)

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
        sl.valueChanged.connect(self.set_brush)
        v2.addWidget(sl)
        row = QHBoxLayout()
        row.setSpacing(6)
        b_reset = QPushButton("重置场景")
        b_reset.clicked.connect(self.reset)
        row.addWidget(b_reset)
        b_pause = QPushButton("暂停")
        b_pause.clicked.connect(self.toggle)
        self.btn_pause = b_pause
        row.addWidget(b_pause)
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

    def select(self, name):
        self.material = name
        for k, b in self.btns.items():
            b.setChecked(k == name)

    def set_brush(self, v):
        self.brush = v

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
