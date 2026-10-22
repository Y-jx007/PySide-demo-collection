from custom_import import *
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QRadialGradient

import numpy as np

real = ti.f32
dim = 3

# ==================== 参数 ====================
N_PARTICLES = 2600
R_CUP = 0.42                 # 杯内半径（世界单位）
H_CUP = 0.80                 # 杯高
FILL = 0.55                  # 初始水柱高度
H = 0.082                    # 光滑核半径
DX = 0.046                   # 初始粒子间距
RHO0 = 1.0
REP = 0.55                # 排斥力强度
VISC = 0.9                # 粘性
KP = 5.0                 # 超压刚度（维持体积）
RHO0 = 6.2                # 静止时的粒子数密度
DT = 2e-3
SUBSTEPS = 6
GRAVITY = -2.2
GRID = 14                    # 邻居网格边长（覆盖 [-0.5, 0.5]）
CELL = 1.0 / GRID          # ≈ H，保证 27 格邻域刚好覆盖核半径
MAX_PER_CELL = 48


class CupFluid:
    def __init__(self, n=N_PARTICLES):
        ti.reset()
        ti.init(arch=ti.cpu, default_fp=real)

        pts = self._make_points(n)
        self.n = n = len(pts)
        self.pos = ti.Vector.field(dim, real, n)
        self.vel = ti.Vector.field(dim, real, n)
        self.acc = ti.Vector.field(dim, real, n)
        self.dens = ti.field(real, n)
        self.pres = ti.field(real, n)
        self.cell_cnt = ti.field(ti.i32, GRID * GRID * GRID)
        self.cell_idx = ti.field(ti.i32, (GRID * GRID * GRID, MAX_PER_CELL))

        # 杯子状态
        self.cup_c = ti.Vector.field(dim, real, shape=())
        self.cup_v = ti.Vector.field(dim, real, shape=())
        self.cup_w = ti.field(real, shape=())        # 绕竖轴角速度
        self.cup_c[None] = [0.0, 0.0, 0.0]
        self.cup_v[None] = [0.0, 0.0, 0.0]
        self.cup_w[None] = 0.0

        self.pos.from_numpy(pts)
        self.vel.fill(0.0)
        self.home = pts.copy()
        self.r_max = R_CUP - 0.035
        self._compile()

    # ---------- 初始水柱 ----------
    def _make_points(self, n):
        pts = []
        step = DX
        r_max = R_CUP - 0.035
        y = 0.03
        while y < 0.03 + FILL * H_CUP and len(pts) < n:
            r = 0.0
            while r < r_max and len(pts) < n:
                n_ring = max(1, int(2 * np.pi * max(r, step * 0.5) / step))
                for k in range(n_ring):
                    a = 2 * np.pi * k / n_ring
                    pts.append([r * np.cos(a), y, r * np.sin(a)])
                r += step
            y += step
        return np.array(pts[:n], np.float32)

    # ---------- kernel ----------
    def _compile(self):
        n = self.n
        pos, vel, acc = self.pos, self.vel, self.acc
        dens, pres = self.dens, self.pres
        cnt, idx = self.cell_cnt, self.cell_idx
        cup_c, cup_v, cup_w = self.cup_c, self.cup_v, self.cup_w

        @ti.kernel
        def clear_cells():
            for c in range(GRID * GRID * GRID):
                cnt[c] = 0

        @ti.kernel
        def bin_particles():
            for i in range(n):
                p = pos[i] + 0.5
                gx = ti.cast(p[0] / CELL, ti.i32)
                gy = ti.cast(p[1] / CELL, ti.i32)
                gz = ti.cast(p[2] / CELL, ti.i32)
                gx = ti.min(ti.max(gx, 0), GRID - 1)
                gy = ti.min(ti.max(gy, 0), GRID - 1)
                gz = ti.min(ti.max(gz, 0), GRID - 1)
                c = gx * GRID * GRID + gz * GRID + gy
                k = ti.atomic_add(cnt[c], 1)
                if k < MAX_PER_CELL:
                    idx[c, k] = i

        @ti.kernel
        def compute_density():
            for i in range(n):
                gx = ti.min(ti.max(ti.cast((pos[i][0] + 0.5) / CELL, ti.i32), 0), GRID - 1)
                gy = ti.min(ti.max(ti.cast((pos[i][1] + 0.5) / CELL, ti.i32), 0), GRID - 1)
                gz = ti.min(ti.max(ti.cast((pos[i][2] + 0.5) / CELL, ti.i32), 0), GRID - 1)
                d = 0.0
                for dx in ti.static(range(-1, 2)):
                    for dy in ti.static(range(-1, 2)):
                        for dz in ti.static(range(-1, 2)):
                            cx, cy, cz = gx + dx, gy + dy, gz + dz
                            if 0 <= cx < GRID and 0 <= cy < GRID and 0 <= cz < GRID:
                                c = cx * GRID * GRID + cz * GRID + cy
                                m = ti.min(cnt[c], MAX_PER_CELL)
                                for k in range(m):
                                    j = idx[c, k]
                                    r2 = (pos[i] - pos[j]).norm_sqr()
                                    if r2 < H * H:
                                        q = 1.0 - r2 / (H * H)
                                        d += q * q * q
                dens[i] = d

        @ti.kernel
        def compute_forces():
            for i in range(n):
                gx = ti.min(ti.max(ti.cast((pos[i][0] + 0.5) / CELL, ti.i32), 0), GRID - 1)
                gy = ti.min(ti.max(ti.cast((pos[i][1] + 0.5) / CELL, ti.i32), 0), GRID - 1)
                gz = ti.min(ti.max(ti.cast((pos[i][2] + 0.5) / CELL, ti.i32), 0), GRID - 1)
                f = ti.Vector([0.0, 0.0, 0.0])
                for dx in ti.static(range(-1, 2)):
                    for dy in ti.static(range(-1, 2)):
                        for dz in ti.static(range(-1, 2)):
                            cx, cy, cz = gx + dx, gy + dy, gz + dz
                            if 0 <= cx < GRID and 0 <= cy < GRID and 0 <= cz < GRID:
                                c = cx * GRID * GRID + cz * GRID + cy
                                m = ti.min(cnt[c], MAX_PER_CELL)
                                for k in range(m):
                                    j = idx[c, k]
                                    r = pos[i] - pos[j]
                                    r2 = r.norm_sqr()
                                    if 1e-9 < r2 < H * H:
                                        d = ti.sqrt(r2)
                                        q = 1.0 - d / H
                                        # 排斥：基础项 + 超压项（近似不可压）
                                        over = dens[i] - RHO0
                                        if over < 0.0:
                                            over = 0.0
                                        f += (REP + KP * over) * q * q * (r / d)
                                        # 粘性：向邻域平均速度靠拢
                                        f += VISC * (vel[j] - vel[i]) * q
                f[1] += GRAVITY
                acc[i] = f

        @ti.kernel
        def integrate():
            c = cup_c[None]
            cv = cup_v[None]
            w = cup_w[None]
            for i in range(n):
                vel[i] += DT * acc[i]
                p = pos[i] + DT * vel[i]
                # 杯底
                if p[1] < 0.01:
                    p[1] = 0.01
                    vel[i][1] = ti.max(vel[i][1], 0.0)
                # 侧壁：圆柱
                rel = ti.Vector([p[0] - c[0], 0.0, p[2] - c[2]])
                rr = rel.norm() + 1e-8
                # 杯壁附近：向壁面切向速度靠拢（否则水根本碰不到壁，转不起来）
                if rr > R_CUP - 0.10:
                    tang_w = ti.Vector([-rel[2], 0.0, rel[0]]) / rr * w * R_CUP
                    blend = ti.min(1.0, (rr - (R_CUP - 0.10)) / 0.10) * 0.35
                    vel[i] = vel[i] * (1.0 - blend) + tang_w * blend
                if rr > R_CUP - 0.02:
                    nrm = rel / rr
                    p = ti.Vector([c[0] + nrm[0] * (R_CUP - 0.02), p[1],
                                   c[2] + nrm[2] * (R_CUP - 0.02)])
                    # 壁面速度 = 杯子平移 + 旋转
                    tang = ti.Vector([-nrm[2], 0.0, nrm[0]]) * w * (R_CUP - 0.02)
                    vel[i] = cv + tang * 0.6
                    vel[i][1] *= 0.5
                if p[1] > H_CUP:
                    p[1] = H_CUP
                    vel[i][1] = ti.min(vel[i][1], 0.0)
                pos[i] = p

        self._clear_cells = clear_cells
        self._bin = bin_particles
        self._density = compute_density
        self._forces = compute_forces
        self._integrate = integrate

    # ---------- 推进 ----------
    def step(self, sub=SUBSTEPS):
        for _ in range(sub):
            self._clear_cells()
            self._bin()
            self._density()
            self._forces()
            self._integrate()

    def snapshot(self):
        return self.pos.to_numpy()

    def speeds(self):
        return np.linalg.norm(self.vel.to_numpy(), axis=1)


# ==================== 渲染与界面 ====================
TILT = np.deg2rad(34.0)          # 俯视角度
SCALE_BASE = 1.0


class CupCanvas(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumSize(560, 560)
        self.setMouseTracking(True)
        self.dragging = False
        self.last = None
        self._buf = None

    def _buffer(self, w, h):
        if self._buf is None or self._buf.shape[0] != h or self._buf.shape[1] != w:
            self._buf = np.full((h, w, 3), (250, 249, 246), np.uint8)
        return self._buf

    # ---------- 3D → 2D ----------
    def project(self, pts, w, h):
        cx, cy = self.app.sim.cup_c[None][0], self.app.sim.cup_c[None][2]
        scale = min(w, h) * 0.78 / 1.25
        sx = w / 2 + (pts[:, 0] - cx) * scale
        sy = h * 0.62 - (pts[:, 1] * np.cos(TILT) + (pts[:, 2] - cy) * np.sin(TILT)) * scale
        depth = pts[:, 1] * np.sin(TILT) - (pts[:, 2] - cy) * np.cos(TILT)
        return sx, sy, depth, scale

    def _proj1(self, x, y, z, w, h, scale, cx, base_y):
        c = self.app.sim.cup_c[None]
        sx = cx + (x) * scale
        sy = base_y - (y * np.cos(TILT) + (z) * np.sin(TILT)) * scale
        return sx, sy

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(10, 16, 30))
        w, h = self.width(), self.height()
        sim = self.app.sim

        # 杯子轮廓（两条竖边 + 上下椭圆）
        c = sim.cup_c[None]
        scale = min(w, h) * 0.78 / 1.25
        cx = w / 2
        base_y = h * 0.62
        cup_r = R_CUP * scale
        top_y = base_y - H_CUP * np.cos(TILT) * scale
        ell_h = R_CUP * np.sin(TILT) * scale

        p.setPen(QPen(QColor(120, 160, 210, 70), 2))
        p.setBrush(QColor(40, 70, 110, 30))
        p.drawEllipse(QRectF(cx - cup_r, base_y - ell_h, cup_r * 2, ell_h * 2))
        p.drawEllipse(QRectF(cx - cup_r, top_y - ell_h, cup_r * 2, ell_h * 2))
        p.drawLine(QPointF(cx - cup_r, base_y), QPointF(cx - cup_r, top_y))
        p.drawLine(QPointF(cx + cup_r, base_y), QPointF(cx + cup_r, top_y))
        # 杯壁竖条纹，体现转动
        ang = self.app.visual_spin
        for k in range(12):
            a = ang + k * np.pi / 6
            px = cx + np.cos(a) * cup_r
            py = base_y - (0.5 + 0.5 * np.sin(a)) * (base_y - top_y)
            p.setPen(QPen(QColor(90, 130, 180, 35), 1))
            p.drawLine(QPointF(px, py - 26), QPointF(px, py + 26))

        # 水粒子：用 numpy 直接画进缓冲，再贴成一张图（比 5000 次 drawEllipse 快两个数量级）
        pts = sim.pos.to_numpy()
        spd = sim.speeds()
        sx, sy, depth, scale = self.project(pts, w, h)

        img = self._buffer(w, h)
        hh, ww = img.shape[:2]
        # 每帧清空（残影会让画面糊成网格）
        img[:] = np.array([10, 16, 30], np.uint8)
        vmax = max(0.6, float(np.percentile(spd, 96)))
        t = np.clip(spd / vmax, 0.0, 1.0)
        r = (52 + 130 * t).astype(np.float32)
        g = (104 + 140 * t).astype(np.float32)
        b = (178 + 70 * t).astype(np.float32)
        # 深度越近越不透明（远端的粒子淡一些）
        dnorm = (depth - depth.min()) / max(1e-6, depth.max() - depth.min())
        a = np.clip(0.45 + 0.45 * (1.0 - dnorm), 0.30, 0.92).astype(np.float32)
        ix = sx.astype(np.int32)
        iy = sy.astype(np.int32)
        rad = max(2, int(round(0.022 * scale)))
        # 3x3 小方块刷点：9 次向量化赋值，比逐粒子画圆快两个数量级
        for dy in (-rad, 0, rad):
            for dx in (-rad, 0, rad):
                xs = ix + dx
                ys = iy + dy
                ok = (xs >= 0) & (xs < ww) & (ys >= 0) & (ys < hh)
                if not ok.any():
                    continue
                wgt = (a * (0.75 if (dx or dy) else 1.0))[ok, None]
                col = np.stack([r[ok], g[ok], b[ok]], axis=1) * wgt
                old = img[ys[ok], xs[ok]].astype(np.float32)
                np.minimum(old + col, 255.0, out=old)
                img[ys[ok], xs[ok]] = old.astype(np.uint8)
        from PySide6.QtGui import QImage
        qimg = QImage(img.data, ww, hh, ww * 3, QImage.Format_RGB888)
        p.drawImage(0, 0, qimg)

        # 玻璃杯轮廓（画在水的上面）
        p.setPen(QPen(QColor(120, 175, 225, 90), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(cx - cup_r, top_y - ell_h, cup_r * 2, ell_h * 2))
        p.drawLine(QPointF(cx - cup_r, base_y), QPointF(cx - cup_r, top_y))
        p.drawLine(QPointF(cx + cup_r, base_y), QPointF(cx + cup_r, top_y))
        p.drawEllipse(QRectF(cx - cup_r, base_y - ell_h, cup_r * 2, ell_h * 2))

        # 涡心标记：速度环流最强处
        if self.app.show_axis:
            m = np.abs(pts[:, 0] + pts[:, 2]) < 0.02
            if m.any():
                cxm = sx[m].mean()
                cym = sy[m].mean()
                p.setPen(QPen(QColor(255, 255, 255, 160), 1.2, Qt.DashLine))
                p.drawEllipse(QPointF(cxm, cym), 0.10 * scale, 0.04 * scale)
        p.end()

    # ---------- 鼠标摇杯子 ----------
    def mousePressEvent(self, e):
        self.dragging = True
        self.last = e.position()

    def mouseReleaseEvent(self, e):
        self.dragging = False

    def mouseMoveEvent(self, e):
        if not self.dragging or self.last is None:
            return
        d = e.position() - self.last
        self.last = e.position()
        self.app.drag_cup(float(d.x()) * 0.0016, float(d.y()) * 0.0016)


class CupVortexApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("杯中水 · 3D SPH 涡旋")
        self.setGeometry(120, 80, 1100, 760)

        self.sim = CupFluid()
        self.visual_spin = 0.0
        self.show_axis = False
        self.shaking = False
        self.shake_phase = 0.0
        self.steps_per_frame = 6

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    # ---------- 交互 ----------
    def drag_cup(self, dx, dz):
        c = self.sim.cup_c[None]
        c = [max(-0.12, min(0.12, c[0] + dx)), 0.0,
             max(-0.12, min(0.12, c[2] + dz))]
        self.sim.cup_c[None] = c
        self.sim.cup_v[None] = [dx * 30.0, 0.0, dz * 30.0]

    def set_shake(self, amp):
        self.shake_amp = amp * 0.010

    def tick(self):
        # 自动摇晃
        if self.shaking:
            self.shake_phase += 0.16
            c = self.sim.cup_c[None]
            nx = self.shake_amp * np.sin(self.shake_phase)
            nz = self.shake_amp * np.sin(self.shake_phase * 0.5 + 1.0)
            self.sim.cup_v[None] = [(nx - c[0]) * 18.0, 0.0, (nz - c[2]) * 18.0]
            self.sim.cup_c[None] = [nx, 0.0, nz]
        else:
            self.sim.cup_v[None] = [0.0, 0.0, 0.0]

        self.sim.cup_w[None] = float(self.spin_slider.value()) / 100.0 * 9.0
        self.visual_spin += self.sim.cup_w[None] * 0.02

        self.sim.step(self.steps_per_frame)
        self.canvas.update()
        p = self.sim.snapshot()
        v = self.sim.vel.to_numpy()
        L = float(np.sum(p[:, 2] * v[:, 0] - p[:, 0] * v[:, 2]))
        self.lbl_l.setText(f"环流 {L:+.0f}")
        self.lbl_v.setText(f"最大速度 {self.sim.speeds().max():.2f}")
        self.lbl_status.setText(
            f"{self.sim.n} 粒子 · 每帧 {self.steps_per_frame} 步 · dt 1e-3 · 拖动鼠标摇杯子")

    # ---------- UI ----------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        panel = QWidget()
        panel.setFixedWidth(258)
        pv = QVBoxLayout(panel)
        pv.setSpacing(8)
        pv.setContentsMargins(0, 0, 0, 0)

        title = QLabel("杯中水 · 涡旋")
        tf = title.font()
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        pv.addWidget(title)
        sub = QLabel("5200 个 SPH 粒子在圆柱杯里" + chr(10) + "拖动鼠标摇杯子，或转杯壁制造涡旋")
        sub.setStyleSheet("color: #7a7a80;")
        sf = sub.font()
        sf.setPointSize(8)
        sub.setFont(sf)
        pv.addWidget(sub)

        self.canvas = CupCanvas(self)

        g1 = QGroupBox("驱动")
        v1 = QVBoxLayout(g1)
        v1.setSpacing(6)
        v1.setContentsMargins(8, 8, 8, 8)
        self.btn_shake = QPushButton("开始摇晃")
        self.btn_shake.clicked.connect(self.toggle_shake)
        v1.addWidget(self.btn_shake)
        self.spin_slider = QSlider(Qt.Horizontal)
        self.spin_slider.setRange(0, 100)
        self.spin_slider.setValue(0)
        self.spin_lbl = QLabel("0")
        self.spin_lbl.setFixedWidth(30)
        self.spin_slider.valueChanged.connect(lambda v: self.spin_lbl.setText(str(v)))
        row = QHBoxLayout()
        row.addWidget(QLabel("转壁"))
        row.addWidget(self.spin_slider, 1)
        row.addWidget(self.spin_lbl)
        v1.addLayout(row)
        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(0, 100)
        self.amp_slider.setValue(60)
        self.amp_slider.valueChanged.connect(lambda v: self.set_shake(v))
        self.shake_amp = 0.006
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("幅度"))
        row2.addWidget(self.amp_slider, 1)
        v1.addLayout(row2)
        btn_reset = QPushButton("重置水面")
        btn_reset.clicked.connect(self.reset)
        v1.addWidget(btn_reset)
        pv.addWidget(g1)

        g2 = QGroupBox("状态")
        v2 = QVBoxLayout(g2)
        v2.setSpacing(4)
        v2.setContentsMargins(8, 8, 8, 8)
        self.lbl_l = QLabel("环流 +0")
        self.lbl_v = QLabel("最大速度 0.00")
        v2.addWidget(self.lbl_l)
        v2.addWidget(self.lbl_v)
        pv.addWidget(g2)
        pv.addStretch()

        root.addWidget(panel)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(0)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.canvas, 1)
        root.addWidget(right, 1)

        self.lbl_status = QLabel("拖动鼠标摇杯子")
        self.statusBar().addWidget(self.lbl_status, 1)

    def toggle_shake(self):
        self.shaking = not self.shaking
        self.btn_shake.setText("停止摇晃" if self.shaking else "开始摇晃")

    def reset(self):
        self.sim = CupFluid()
        self.shaking = False
        self.btn_shake.setText("开始摇晃")

    def closeEvent(self, event):
        self.timer.stop()
        try:
            ti.reset()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CupVortexApp()
    window.show()
    sys.exit(app.exec())
