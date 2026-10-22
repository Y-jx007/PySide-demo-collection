from custom_import import *
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPolygonF

import numpy as np

real = ti.f32
dim = 3

# ==================== 参数 ====================
N_PARTICLES = 90000
R_CUP = 0.30                 # 世界单位；整只杯子当作 0.6 m 宽、1.2 m 高
H_CUP = 1.20
FILL = 0.86
DX = 0.016                   # 粒子间距
H = 0.032                    # 光滑核半径 = 2*DX
GRAVITY = -9.81              # ← 真实重力，之前是 -2.2
DT = 5e-4                    # 每个子步（CFL: 0.4*h/c_sound）
SUBSTEPS_PER_FRAME = 16      # 每帧推进 8 ms 仿真时间
RHO0 = 1000.0                # 水的密度
MASS = RHO0 * DX ** 3        # 粒子质量
STIFF_B = 2.0e4              # Tait 方程系数 p = B((rho/rho0)^7 - 1)
MU = 0.06                    # 动力粘性
CELL = 1.0 / 33.0
GXZ = 33
GY = int(1.25 / CELL) + 2
MAX_PER_CELL = 32

# 核函数常数
K_POLY6 = 315.0 / (64.0 * np.pi * H ** 9)
K_SPIKY = -45.0 / (np.pi * H ** 6)
K_LAP = 45.0 / (np.pi * H ** 6)

RW, RH = 640, 620            # 渲染缓冲尺寸


class CupFluid:
    def __init__(self, n=N_PARTICLES):
        ti.reset()
        ti.init(arch=ti.gpu, default_fp=real)

        pts = self._make_points(n)
        self.n = n = len(pts)

        self.pos = ti.Vector.field(dim, real, n)
        self.vel = ti.Vector.field(dim, real, n)
        self.acc = ti.Vector.field(dim, real, n)
        self.dens = ti.field(real, n)
        self.pres = ti.field(real, n)
        self.cell_cnt = ti.field(ti.i32, GXZ * GXZ * GY)
        self.cell_idx = ti.field(ti.i32, (GXZ * GXZ * GY, MAX_PER_CELL))
        self.img = ti.Vector.field(3, real, (RH, RW))
        self.dens_img = ti.field(real, (RH, RW))
        self.blur_tmp = ti.field(real, (RH, RW))
        self.col_img = ti.Vector.field(3, real, (RH, RW))
        self.blur_col_tmp = ti.Vector.field(3, real, (RH, RW))
        self.dens_blur = ti.field(real, (RH, RW))
        self.blur_col = ti.Vector.field(3, real, (RH, RW))

        self.cup_c = ti.Vector.field(dim, real, shape=())
        self.cup_v = ti.Vector.field(dim, real, shape=())
        self.cup_w = ti.field(real, shape=())
        self.cup_c[None] = [0.0, 0.0, 0.0]
        self.cup_v[None] = [0.0, 0.0, 0.0]
        self.cup_w[None] = 0.0

        self.pos.from_numpy(pts)
        self.vel.fill(0.0)
        self._compile()

        self._clear_cells()
        self._bin()
        self._density()
        info = self.dens.to_numpy()
        print(f"[sph] 初始密度 mean {info.mean():.1f} min {info.min():.1f} max {info.max():.1f}", flush=True)

    # ---------- 初始水柱 ----------
    def _make_points(self, n):
        pts = []
        r_max = R_CUP - 0.02
        y = 0.02
        while y < 0.02 + FILL * H_CUP and len(pts) < n:
            r = 0.0
            while r < r_max and len(pts) < n:
                ring = max(1, int(2 * np.pi * max(r, DX * 0.5) / DX))
                for k in range(ring):
                    a = 2 * np.pi * k / ring
                    pts.append([r * np.cos(a), y, r * np.sin(a)])
                r += DX
            y += DX
        arr = np.array(pts[:n], np.float32)
        arr += np.random.default_rng(7).normal(0.0, 0.3 * DX, arr.shape).astype(np.float32)
        return arr

    # ---------- kernel ----------
    def _compile(self):
        n = self.n
        pos, vel, acc = self.pos, self.vel, self.acc
        dens = self.dens
        pres = self.pres
        cnt, idx = self.cell_cnt, self.cell_idx
        cup_c, cup_v, cup_w = self.cup_c, self.cup_v, self.cup_w
        img = self.img
        dens_img = self.dens_img
        blur_tmp = self.blur_tmp
        col_img = self.col_img
        blur_col_tmp = self.blur_col_tmp
        dens_blur = self.dens_blur
        blur_col = self.blur_col

        @ti.kernel
        def clear_cells():
            for c in range(GXZ * GXZ * GY):
                cnt[c] = 0

        @ti.kernel
        def bin_particles():
            for i in range(n):
                gx = ti.min(ti.max(ti.cast((pos[i][0] + 0.5) / CELL, ti.i32), 0), GXZ - 1)
                gz = ti.min(ti.max(ti.cast((pos[i][2] + 0.5) / CELL, ti.i32), 0), GXZ - 1)
                gy = ti.min(ti.max(ti.cast(pos[i][1] / CELL, ti.i32), 0), GY - 1)
                c = (gx * GXZ + gz) * GY + gy
                k = ti.atomic_add(cnt[c], 1)
                if k < MAX_PER_CELL:
                    idx[c, k] = i

        @ti.kernel
        def compute_density():
            for i in range(n):
                gx = ti.min(ti.max(ti.cast((pos[i][0] + 0.5) / CELL, ti.i32), 0), GXZ - 1)
                gz = ti.min(ti.max(ti.cast((pos[i][2] + 0.5) / CELL, ti.i32), 0), GXZ - 1)
                gy = ti.min(ti.max(ti.cast(pos[i][1] / CELL, ti.i32), 0), GY - 1)
                d = 0.0
                for dx in ti.static(range(-1, 2)):
                    for dz in ti.static(range(-1, 2)):
                        for dy in ti.static(range(-1, 2)):
                            cx, cz, cy = gx + dx, gz + dz, gy + dy
                            if 0 <= cx < GXZ and 0 <= cz < GXZ and 0 <= cy < GY:
                                c = (cx * GXZ + cz) * GY + cy
                                m = ti.min(cnt[c], MAX_PER_CELL)
                                for k in range(m):
                                    j = idx[c, k]
                                    r2 = (pos[i] - pos[j]).norm_sqr()
                                    if r2 < H * H:
                                        q = H * H - r2
                                        d += q * q * q
                dens[i] = MASS * K_POLY6 * d
                # Tait 方程，负压截断（避免拉在一起）
                ratio = dens[i] / RHO0
                p = STIFF_B * (ratio ** 7 - 1.0)
                pres[i] = ti.max(p, 0.0)

        @ti.kernel
        def compute_forces():
            for i in range(n):
                gx = ti.min(ti.max(ti.cast((pos[i][0] + 0.5) / CELL, ti.i32), 0), GXZ - 1)
                gz = ti.min(ti.max(ti.cast((pos[i][2] + 0.5) / CELL, ti.i32), 0), GXZ - 1)
                gy = ti.min(ti.max(ti.cast(pos[i][1] / CELL, ti.i32), 0), GY - 1)
                f = ti.Vector([0.0, 0.0, 0.0])
                pi_ = pres[i]
                ri = dens[i]
                for dx in ti.static(range(-1, 2)):
                    for dz in ti.static(range(-1, 2)):
                        for dy in ti.static(range(-1, 2)):
                            cx, cz, cy = gx + dx, gz + dz, gy + dy
                            if 0 <= cx < GXZ and 0 <= cz < GXZ and 0 <= cy < GY:
                                c = (cx * GXZ + cz) * GY + cy
                                m = ti.min(cnt[c], MAX_PER_CELL)
                                for k in range(m):
                                    j = idx[c, k]
                                    r_vec = pos[i] - pos[j]
                                    r2 = r_vec.norm_sqr()
                                    if 1e-12 < r2 < H * H:
                                        r = ti.sqrt(r2)
                                        # 压力：spiky 梯度的标准对称形式
                                        hq = H - r
                                        wp = K_SPIKY * hq * hq / r * r_vec
                                        coeff = -MASS * (pi_ / (ri * ri) + pres[j] / (dens[j] * dens[j]))
                                        f += coeff * wp
                                        # 粘性：拉普拉斯核
                                        f += MU * MASS * (vel[j] - vel[i]) / dens[j] * (K_LAP * hq)
                f[1] += GRAVITY
                acc[i] = f

        @ti.kernel
        def integrate():
            c = cup_c[None]
            cv_raw = cup_v[None]
            sp = cv_raw.norm()
            cv = cv_raw * (4.0 / sp) if sp > 4.0 else cv_raw
            w = cup_w[None]
            for i in range(n):
                vel[i] += DT * acc[i]
                p = pos[i] + DT * vel[i]
                if p[1] < 0.01:
                    p[1] = 0.01
                    vel[i][1] = ti.max(vel[i][1], 0.0)
                    vel[i][0] += cv[0] * 0.30
                    vel[i][2] += cv[2] * 0.30
                rel = ti.Vector([p[0] - c[0], 0.0, p[2] - c[2]])
                rr = rel.norm() + 1e-8
                # 只有贴壁最后 0.05 的一圈受壁面拖曳
                if rr > R_CUP - 0.05:
                    tang = ti.Vector([-rel[2], 0.0, rel[0]]) / rr * w * R_CUP
                    blend = ti.min(1.0, (rr - (R_CUP - 0.05)) / 0.05) * 0.12
                    vel[i] = vel[i] * (1.0 - blend) + (cv + tang) * blend
                if rr > R_CUP - 0.01:
                    nrm = rel / rr
                    p[0] = c[0] + nrm[0] * (R_CUP - 0.01)
                    p[2] = c[2] + nrm[2] * (R_CUP - 0.01)
                if p[1] > H_CUP:
                    p[1] = H_CUP
                    vel[i][1] = ti.min(vel[i][1], 0.0)
                # 速度限幅，防止数值爆炸
                spd = vel[i].norm()
                if spd > 10.0:
                    vel[i] *= 10.0 / spd
                pos[i] = p

        # ---------- 屏幕空间流体渲染 ----------
        @ti.kernel
        def splat(m00: real, m01: real, m02: real,
                  m10: real, m11: real, m12: real,
                  m20: real, m21: real, m22: real,
                  scale: real, cx: real, cy: real,
                  cx3: real, cy3: real, cz3: real):
            for i, j in img:
                dens_img[i, j] = 0.0
                col_img[i, j] = ti.Vector([0.0, 0.0, 0.0])
            for p in range(n):
                v = pos[p] - ti.Vector([cx3, cy3, cz3])
                X = m00 * v[0] + m01 * v[1] + m02 * v[2]
                Y = m10 * v[0] + m11 * v[1] + m12 * v[2]
                Z = m20 * v[0] + m21 * v[1] + m22 * v[2]
                sx = cx + X * scale
                sy = cy - Y * scale
                spd = vel[p].norm()
                t = ti.min(spd / 1.5, 1.0)
                depth = ti.min(ti.max(Z * 2.0 + 0.5, 0.0), 1.0)
                w = 0.55 + 0.45 * (1.0 - depth)
                col = ti.Vector([30.0 + 130.0 * t,
                                 90.0 + 150.0 * t,
                                 190.0 + 65.0 * t])
                ix = ti.cast(sx, ti.i32)
                iy = ti.cast(sy, ti.i32)
                for ox, oy in ti.static(ti.ndrange(2, 2)):
                    px = ix + ox
                    py = iy + oy
                    if 0 <= px < RW and 0 <= py < RH:
                        ti.atomic_add(dens_img[py, px], w)
                        ti.atomic_add(col_img[py, px], col * w)

        @ti.kernel
        def blur_h(rad: ti.i32):
            for i, j in dens_img:
                s_ = 0.0
                c_ = ti.Vector([0.0, 0.0, 0.0])
                cnt = 0.0
                for k in range(-rad, rad + 1):
                    x = ti.min(ti.max(j + k, 0), RW - 1)
                    wt = 1.0 - ti.abs(ti.cast(k, real)) / (rad + 1.0)
                    s_ += dens_img[i, x] * wt
                    c_ += col_img[i, x] * wt
                    cnt += wt
                blur_tmp[i, j] = s_ / cnt
                blur_col_tmp[i, j] = c_ / cnt

        @ti.kernel
        def blur_v(rad: ti.i32):
            for i, j in dens_img:
                s_ = 0.0
                c_ = ti.Vector([0.0, 0.0, 0.0])
                cnt = 0.0
                for k in range(-rad, rad + 1):
                    y = ti.min(ti.max(i + k, 0), RH - 1)
                    wt = 1.0 - ti.abs(ti.cast(k, real)) / (rad + 1.0)
                    s_ += blur_tmp[y, j] * wt
                    c_ += blur_col_tmp[y, j] * wt
                    cnt += wt
                blur_col[i, j] = c_ / cnt
                dens_blur[i, j] = s_ / cnt

        @ti.kernel
        def shade(gain: real):
            for i, j in img:
                d = ti.min(dens_blur[i, j] * gain, 2.0)
                # 薄处透亮、厚处深沉，看起来像有体积的液体
                alpha = ti.min(d * 1.15, 1.0)
                thin = ti.Vector([150.0, 225.0, 255.0])
                thick = ti.Vector([18.0, 62.0, 150.0])
                k = ti.min(d, 1.0)
                body = thin * (1.0 - k) + thick * k
                # 上下一点点明暗变化，避免整块死板
                vert = 0.85 + 0.35 * (1.0 - ti.cast(i, real) / RH)
                # 速度着色：流得快的地方偏青白
                spd_c = col_img[i, j] / ti.max(dens_blur[i, j] * 3.0, 1e-4)
                col = body * vert + spd_c * 0.25 * alpha
                bg = ti.Vector([8.0, 12.0, 24.0])
                img[i, j] = bg * (1.0 - alpha) + col * alpha

        self._clear_cells = clear_cells
        self._bin = bin_particles
        self._density = compute_density
        self._forces = compute_forces
        self._integrate = integrate
        self._splat = splat
        self._blur_h = blur_h
        self._blur_v = blur_v
        self._shade = shade

    def step(self, sub=SUBSTEPS_PER_FRAME):
        for _ in range(sub):
            self._clear_cells()
            self._bin()
            self._density()
            self._forces()
            self._integrate()

    def draw(self, m, scale, cx, cy, center):
        self._splat(m[0][0], m[0][1], m[0][2],
                    m[1][0], m[1][1], m[1][2],
                    m[2][0], m[2][1], m[2][2],
                    scale, cx, cy, center[0], center[1], center[2])
        self._blur_h(7)
        self._blur_v(7)
        self._shade(0.55)
        return np.clip(self.img.to_numpy(), 0, 255).astype(np.uint8)

    def speeds(self):
        return np.linalg.norm(self.vel.to_numpy(), axis=1)

    def snapshot(self):
        return self.pos.to_numpy()


# ==================== 界面 ====================
class CupCanvas(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumSize(620, 560)
        self.yaw = -0.6
        self.pitch = 0.35
        self.zoom = 1.0
        self.dragging = False
        self.orbiting = False
        self.last = None

    # ---------- 相机 ----------
    def matrix(self):
        cy_, sy_ = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]], np.float32)
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
        return (rx @ ry).astype(np.float32)

    def cam(self):
        scale = min(RW, RH) * 0.80 * self.zoom / H_CUP
        cx = RW * 0.5
        cy = RH * 0.5 + H_CUP * 0.5 * scale
        return scale, cx, cy

    def project_points(self, pts):
        m = self.matrix()
        c = self.app.sim.cup_c[None]
        v = pts - np.array([c[0], 0.0, c[2]], np.float32)
        p = v @ m.T
        scale, cx, cy = self.cam()
        return cx + p[:, 0] * scale, cy - p[:, 1] * scale

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(8, 12, 24))

        sim = self.app.sim
        m = self.matrix()
        scale, cx, cy = self.cam()
        c = sim.cup_c[None]
        img = sim.draw(m, scale, cx, cy, (c[0], 0.0, c[2]))

        from PySide6.QtGui import QImage
        img = np.ascontiguousarray(img)
        hr, wr = img.shape[:2]
        qimg = QImage(img.data, wr, hr, wr * 3, QImage.Format_RGB888)

        k = min(w / wr, h / hr)
        ox = (w - wr * k) * 0.5
        oy = (h - hr * k) * 0.5
        p.drawImage(QRectF(ox, oy, wr * k, hr * k), qimg)

        # 跟随杯子运动的玻璃轮廓（图像坐标 → 画布坐标）
        def to_canvas(xs, ys):
            return ox + xs * k, oy + ys * k

        p.setBrush(Qt.NoBrush)
        for yy, alpha in ((0.0, 120), (H_CUP, 70)):
            a = np.linspace(0, 2 * np.pi, 72)
            ring = np.stack([c[0] + R_CUP * np.cos(a), np.full(72, yy),
                             c[2] + R_CUP * np.sin(a)], axis=1).astype(np.float32)
            rx, ry = self.project_points(ring)
            rx, ry = to_canvas(rx, ry)
            p.setPen(QPen(QColor(130, 180, 230, alpha), 1.4))
            p.drawPolyline(QPolygonF([QPointF(float(u), float(v_)) for u, v_ in zip(rx, ry)]))
        for kk in range(6):
            a = kk * np.pi / 3
            seg = np.array([[c[0] + R_CUP * np.cos(a), 0.0, c[2] + R_CUP * np.sin(a)],
                            [c[0] + R_CUP * np.cos(a), H_CUP, c[2] + R_CUP * np.sin(a)]],
                           np.float32)
            rx, ry = self.project_points(seg)
            rx, ry = to_canvas(rx, ry)
            p.setPen(QPen(QColor(110, 160, 210, 45), 1.1))
            p.drawLine(QPointF(float(rx[0]), float(ry[0])),
                       QPointF(float(rx[1]), float(ry[1])))
        p.end()

    # ---------- 鼠标 ----------
    def mousePressEvent(self, e):
        self.dragging = True
        self.orbiting = bool(e.buttons() & Qt.RightButton) or bool(e.modifiers() & Qt.ControlModifier)
        self.last = e.position()

    def mouseReleaseEvent(self, e):
        self.dragging = False

    def mouseMoveEvent(self, e):
        if not self.dragging or self.last is None:
            return
        d = e.position() - self.last
        self.last = e.position()
        if self.orbiting:
            self.yaw += float(d.x()) * 0.01
            self.pitch = max(-0.3, min(1.4, self.pitch + float(d.y()) * 0.01))
        else:
            self.app.drag_cup(float(d.x()) * 0.0012, float(d.y()) * 0.0012)

    def wheelEvent(self, e):
        self.zoom = max(0.4, min(3.0, self.zoom * (1.0015 ** e.angleDelta().y())))


class CupVortexApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("杯中水 · 3D SPH 涡旋")
        self.setGeometry(120, 80, 1120, 800)

        self.sim = CupFluid()
        self.shaking = False
        self.shake_phase = 0.0
        self.shake_amp = 0.010
        self.steps_per_frame = SUBSTEPS_PER_FRAME

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    def drag_cup(self, dx, dz):
        c = self.sim.cup_c[None]
        nc = [max(-0.10, min(0.10, c[0] + dx)), 0.0,
              max(-0.10, min(0.10, c[2] + dz))]
        self.sim.cup_c[None] = nc
        self.sim.cup_v[None] = [dx * 20.0, 0.0, dz * 20.0]

    def tick(self):
        if self.shaking:
            self.shake_phase += 0.16
            c = self.sim.cup_c[None]
            nx = self.shake_amp * np.sin(self.shake_phase)
            nz = self.shake_amp * np.sin(self.shake_phase * 0.5 + 1.0)
            self.sim.cup_v[None] = [(nx - c[0]) * 16.0, 0.0, (nz - c[2]) * 16.0]
            self.sim.cup_c[None] = [nx, 0.0, nz]
        else:
            self.sim.cup_v[None] = [0.0, 0.0, 0.0]

        self.sim.cup_w[None] = float(self.spin_slider.value()) / 100.0 * 9.0
        self.sim.step(self.steps_per_frame)
        self.canvas.update()

        p = self.sim.snapshot()
        v = self.sim.vel.to_numpy()
        L = float(np.sum(p[:, 2] * v[:, 0] - p[:, 0] * v[:, 2]))
        self.lbl_l.setText(f"环流 {L:+.0f}")
        self.lbl_v.setText(f"最大速度 {self.sim.speeds().max():.2f}")

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

        title = QLabel("杯中水 · 涡旋")
        tf = title.font()
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        pv.addWidget(title)
        sub = QLabel("36k SPH 粒子 · Taichi GPU" + chr(10) + "左键摇杯子 · 右键/滚轮转视角")
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
        self.amp_slider.setValue(65)
        self.amp_slider.valueChanged.connect(lambda v: setattr(self, "shake_amp", v * 0.00016))
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

        self.lbl_status = QLabel("左键拖拽摇杯子 · 右键拖拽转视角 · 滚轮缩放")
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
