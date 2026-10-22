from custom_import import *
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPolygonF

import numpy as np

real = ti.f32

# ==================== 参数（MLS-MPM / APIC，参考 taichi mpm128） ====================
N_GRID = 80
DX = 1.0 / N_GRID
INV_DX = float(N_GRID)
DT = 3e-4
SUBSTEPS = 20
GRAVITY = -9.81

R_CUP = 0.30
H_CUP = 1.00
FILL = 0.72

SPACING = DX * 0.80        # 实际粒子间距
P_VOL = SPACING ** 3      # 每个粒子代表的体积（必须和间距一致，否则水位不对）
P_RHO = 1000.0
P_MASS = P_VOL * P_RHO
E_FLUID = 1.0e7           # 体积模量：实测 1e7 时水面稳定在 0.67（灌装 0.72）
VISC_DAMP = 0.9995

RW, RH = 640, 620


class CupFluid:
    """3D MLS-MPM 流体：APIC 传输 + J 驱动的压力项，杯子是解析圆柱边界。"""

    def __init__(self):
        ti.reset()
        ti.init(arch=ti.gpu, default_fp=real)

        pts = self._make_points()
        n = self.n = len(pts)

        self.x = ti.Vector.field(3, real, n)
        self.v = ti.Vector.field(3, real, n)
        self.C = ti.Matrix.field(3, 3, real, n)
        self.J = ti.field(real, n)
        self.x.from_numpy(pts)
        self.v.fill(0.0)
        self.C.fill(0.0)
        self.J.fill(1.0)

        ng = N_GRID
        self.gv = ti.Vector.field(3, real, (ng, ng, ng))
        self.gm = ti.field(real, (ng, ng, ng))
        self.gv.fill(0.0)
        self.gm.fill(0.0)

        self.img = ti.Vector.field(3, real, (RH, RW))
        self.dens_img = ti.field(real, (RH, RW))
        self.blur_tmp = ti.field(real, (RH, RW))
        self.col_img = ti.Vector.field(3, real, (RH, RW))
        self.blur_col_tmp = ti.Vector.field(3, real, (RH, RW))
        self.dens_blur = ti.field(real, (RH, RW))

        self.cup_c = ti.Vector.field(3, real, shape=())
        self.cup_v = ti.Vector.field(3, real, shape=())
        self.cup_w = ti.field(real, shape=())
        self.cup_c[None] = [0.5, 0.0, 0.5]
        self.cup_v[None] = [0.0, 0.0, 0.0]
        self.cup_w[None] = 0.0

        self._compile()

    # ---------- 初始水体 ----------
    def _make_points(self):
        pts = []
        r_max = R_CUP - 0.02
        y = 0.03
        spacing = SPACING
        while y < 0.03 + FILL * H_CUP:
            r = 0.0
            while r < r_max:
                ring = max(1, int(2 * np.pi * max(r, spacing * 0.5) / spacing))
                for k in range(ring):
                    a = 2 * np.pi * k / ring
                    pts.append([0.5 + r * np.cos(a), y, 0.5 + r * np.sin(a)])
                r += spacing
            y += spacing
        return np.array(pts, np.float32)

    # ---------- 内核 ----------
    def _compile(self):
        n = self.n
        x, v, C, J = self.x, self.v, self.C, self.J
        gv, gm = self.gv, self.gm
        cup_c, cup_v, cup_w = self.cup_c, self.cup_v, self.cup_w
        img = self.img
        dens_img, blur_tmp, col_img = self.dens_img, self.blur_tmp, self.col_img
        blur_col_tmp, dens_blur = self.blur_col_tmp, self.dens_blur
        ng = N_GRID

        @ti.kernel
        def clear_grid():
            for i, j, k in gv:
                gv[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                gm[i, j, k] = 0.0

        @ti.kernel
        def p2g():
            for p in range(n):
                base = ti.cast(x[p] * INV_DX - 0.5, ti.i32)
                fx = x[p] * INV_DX - ti.cast(base, real)
                w = [0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1.0) ** 2,
                     0.5 * (fx - 0.5) ** 2]
                new_C = C[p]
                new_J = (1.0 + DT * (new_C[0, 0] + new_C[1, 1] + new_C[2, 2])) * J[p]
                J[p] = new_J
                pressure = -DT * 4.0 * INV_DX * INV_DX * E_FLUID * P_VOL * (new_J - 1.0)
                stress = ti.Matrix([[pressure, 0.0, 0.0],
                                    [0.0, pressure, 0.0],
                                    [0.0, 0.0, pressure]])
                affine = stress + P_MASS * new_C
                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    dpos = (ti.Vector([i, j, k]) - fx) * DX
                    wt = w[i][0] * w[j][1] * w[k][2]
                    nb = base + ti.Vector([i, j, k])
                    if 0 <= nb[0] < ng and 0 <= nb[1] < ng and 0 <= nb[2] < ng:
                        ti.atomic_add(gv[nb], wt * (P_MASS * v[p] + affine @ dpos))
                        ti.atomic_add(gm[nb], wt * P_MASS)

        @ti.kernel
        def grid_update():
            c = cup_c[None]
            cv = cup_v[None]
            w_ = cup_w[None]
            for i, j, k in gm:
                if gm[i, j, k] > 1e-8:
                    vout = gv[i, j, k] / gm[i, j, k]
                    vout[1] += DT * GRAVITY
                    vout *= VISC_DAMP
                    if j <= 1 and vout[1] < 0.0:
                        vout = ti.Vector([0.0, 0.0, 0.0])
                    px = (ti.cast(i, real) + 0.5) * DX - c[0]
                    pz = (ti.cast(k, real) + 0.5) * DX - c[2]
                    rr = ti.sqrt(px * px + pz * pz) + 1e-8
                    if rr > R_CUP - DX:
                        nx = px / rr
                        nz = pz / rr
                        vx = cv[0] + (-nz) * w_ * R_CUP
                        vz = cv[2] + (nx) * w_ * R_CUP
                        vn = vout[0] * nx + vout[2] * nz
                        if vn > 0.0:
                            vout[0] -= vn * nx
                            vout[2] -= vn * nz
                        vout[0] = vout[0] * 0.3 + vx * 0.7
                        vout[2] = vout[2] * 0.3 + vz * 0.7
                    if (ti.cast(j, real) + 0.5) * DX > H_CUP and vout[1] > 0.0:
                        vout[1] = 0.0
                    gv[i, j, k] = vout

        @ti.kernel
        def g2p():
            for p in range(n):
                base = ti.cast(x[p] * INV_DX - 0.5, ti.i32)
                fx = x[p] * INV_DX - ti.cast(base, real)
                w = [0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1.0) ** 2,
                     0.5 * (fx - 0.5) ** 2]
                nv = ti.Vector([0.0, 0.0, 0.0])
                nC = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    nb = base + ti.Vector([i, j, k])
                    if 0 <= nb[0] < ng and 0 <= nb[1] < ng and 0 <= nb[2] < ng:
                        dpos = (ti.Vector([i, j, k]) - fx) * DX
                        wt = w[i][0] * w[j][1] * w[k][2]
                        nv += wt * gv[nb]
                        nC += 4.0 * INV_DX * wt * gv[nb].outer_product(dpos)
                v[p] = nv
                x[p] += DT * nv
                C[p] = nC

        # ---------- 屏幕空间渲染 ----------
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
                vv = x[p] - ti.Vector([cx3, cy3, cz3])
                X = m00 * vv[0] + m01 * vv[1] + m02 * vv[2]
                Y = m10 * vv[0] + m11 * vv[1] + m12 * vv[2]
                Z = m20 * vv[0] + m21 * vv[1] + m22 * vv[2]
                sx = cx + X * scale
                sy = cy - Y * scale
                spd = v[p].norm()
                t = ti.min(spd / 2.5, 1.0)
                depth = ti.min(ti.max(Z * 2.0 + 0.5, 0.0), 1.0)
                w = 0.55 + 0.45 * (1.0 - depth)
                col = ti.Vector([30.0 + 140.0 * t,
                                 95.0 + 150.0 * t,
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
                    xx = ti.min(ti.max(j + k, 0), RW - 1)
                    wt = 1.0 - ti.abs(ti.cast(k, real)) / (rad + 1.0)
                    s_ += dens_img[i, xx] * wt
                    c_ += col_img[i, xx] * wt
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
                    yy = ti.min(ti.max(i + k, 0), RH - 1)
                    wt = 1.0 - ti.abs(ti.cast(k, real)) / (rad + 1.0)
                    s_ += blur_tmp[yy, j] * wt
                    c_ += blur_col_tmp[yy, j] * wt
                    cnt += wt
                dens_blur[i, j] = s_ / cnt
                img[i, j] = c_ / cnt

        @ti.kernel
        def shade(gain: real):
            for i, j in img:
                d = ti.min(dens_blur[i, j] * gain, 2.0)
                alpha = ti.min(d * 1.15, 1.0)
                thin = ti.Vector([150.0, 225.0, 255.0])
                thick = ti.Vector([18.0, 62.0, 150.0])
                k = ti.min(d, 1.0)
                body = thin * (1.0 - k) + thick * k
                vert = 0.85 + 0.35 * (1.0 - ti.cast(i, real) / RH)
                bg = ti.Vector([8.0, 12.0, 24.0])
                img[i, j] = bg * (1.0 - alpha) + body * vert * alpha

        self._clear_grid = clear_grid
        self._p2g = p2g
        self._grid = grid_update
        self._g2p = g2p
        self._splat = splat
        self._blur_h = blur_h
        self._blur_v = blur_v
        self._shade = shade

    def step(self, sub=SUBSTEPS):
        for _ in range(sub):
            self._clear_grid()
            self._p2g()
            self._grid()
            self._g2p()

    def draw(self, m, scale, cx, cy, center):
        self._splat(m[0][0], m[0][1], m[0][2],
                    m[1][0], m[1][1], m[1][2],
                    m[2][0], m[2][1], m[2][2],
                    scale, cx, cy, center[0], center[1], center[2])
        self._blur_h(7)
        self._blur_v(7)
        self._shade(0.35)
        return np.clip(self.img.to_numpy(), 0, 255).astype(np.uint8)

    def speeds(self):
        return np.linalg.norm(self.v.to_numpy(), axis=1)

    def snapshot(self):
        return self.x.to_numpy()


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

    def matrix(self):
        cy_, sy_ = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]], np.float32)
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
        return (rx @ ry).astype(np.float32)

    def cam(self):
        scale = min(RW, RH) * 0.80 * self.zoom / H_CUP
        return scale, RW * 0.5, RH * 0.5 + H_CUP * 0.5 * scale

    def project_points(self, pts):
        m = self.matrix()
        c = self.app.sim.cup_c[None]
        origin = np.array([c[0], 0.0, c[2]], np.float32)
        p = (pts - origin) @ m.T
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

        p.setBrush(Qt.NoBrush)
        for yy, alpha in ((0.0, 120), (H_CUP, 70)):
            a = np.linspace(0, 2 * np.pi, 72)
            ring = np.stack([c[0] + R_CUP * np.cos(a), np.full(72, yy),
                             c[2] + R_CUP * np.sin(a)], axis=1).astype(np.float32)
            rx, ry = self.project_points(ring)
            rx = ox + rx * k
            ry = oy + ry * k
            p.setPen(QPen(QColor(130, 180, 230, alpha), 1.4))
            p.drawPolyline(QPolygonF([QPointF(float(u), float(v_)) for u, v_ in zip(rx, ry)]))
        for kk in range(6):
            a = kk * np.pi / 3
            seg = np.array([[c[0] + R_CUP * np.cos(a), 0.0, c[2] + R_CUP * np.sin(a)],
                            [c[0] + R_CUP * np.cos(a), H_CUP, c[2] + R_CUP * np.sin(a)]],
                           np.float32)
            rx, ry = self.project_points(seg)
            rx = ox + rx * k
            ry = oy + ry * k
            p.setPen(QPen(QColor(110, 160, 210, 45), 1.1))
            p.drawLine(QPointF(float(rx[0]), float(ry[0])),
                       QPointF(float(rx[1]), float(ry[1])))
        p.end()

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
            self.app.drag_cup(float(d.x()) * 0.0008, float(d.y()) * 0.0008)

    def wheelEvent(self, e):
        self.zoom = max(0.4, min(3.0, self.zoom * (1.0015 ** e.angleDelta().y())))


class CupVortexApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("杯中水 · 3D MLS-MPM")
        self.setGeometry(120, 80, 1120, 800)

        self.sim = CupFluid()
        self.shaking = False
        self.shake_phase = 0.0
        self.shake_amp = 0.010
        self.steps_per_frame = SUBSTEPS

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    def drag_cup(self, dx, dz):
        c = self.sim.cup_c[None]
        nc = [max(0.42, min(0.58, c[0] + dx)), 0.0,
              max(0.42, min(0.58, c[2] + dz))]
        self.sim.cup_c[None] = nc
        self.sim.cup_v[None] = [dx * 12.0, 0.0, dz * 12.0]

    def tick(self):
        if self.shaking:
            self.shake_phase += 0.16
            c = self.sim.cup_c[None]
            nx = 0.5 + self.shake_amp * np.sin(self.shake_phase)
            nz = 0.5 + self.shake_amp * np.sin(self.shake_phase * 0.5 + 1.0)
            self.sim.cup_v[None] = [(nx - c[0]) * 12.0, 0.0, (nz - c[2]) * 12.0]
            self.sim.cup_c[None] = [nx, 0.0, nz]
        else:
            self.sim.cup_v[None] = [0.0, 0.0, 0.0]

        self.sim.cup_w[None] = float(self.spin_slider.value()) / 100.0 * 6.0
        self.sim.step(self.steps_per_frame)
        self.canvas.update()

        p = self.sim.snapshot()
        v = self.sim.v.to_numpy()
        L = float(np.sum((p[:, 2] - 0.5) * v[:, 0] - (p[:, 0] - 0.5) * v[:, 2]))
        self.lbl_l.setText(f"环流 {L:+.1f}")
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

        title = QLabel("杯中水 · MLS-MPM")
        tf = title.font()
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        pv.addWidget(title)
        sub = QLabel(f"{self.sim.n // 1000}k 粒子 · {N_GRID}³ 网格" + chr(10)
                     + "APIC 传输 + 状态方程压力")
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
