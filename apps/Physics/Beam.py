from custom_import import *
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPolygonF, QBrush

# ==================== 材料 ====================
G = 9.81
B_WIDTH = 0.06          # 梁宽（出平面方向）

MATERIALS = {
    "松木": (10e9, 40e6, 0.30, 600.0),
    "橡木": (12e9, 55e6, 0.30, 700.0),
    "钢材": (200e9, 250e6, 0.28, 7800.0),
    "混凝土": (30e9, 3e6, 0.20, 2400.0),
}

MASS_SCALE = 8.0       # 质量放大：准静态加载时显式时间步能大很多


# ==================== 静态有限元（测极限载荷 / 扫描） ====================
class StaticBeam:
    """平面应力 Q4 + Cholesky；用于快速求极限载荷和扫描。"""

    def __init__(self, length, height, E, sigma_ult, support="simple", nx=36, ny=6, nu=0.3):
        self.L, self.h, self.E, self.sigma_ult = length, height, E, sigma_ult
        self.nu, self.support, self.nx, self.ny = nu, support, nx, ny
        self._build()
        self._assemble()
        self._apply_bc()
        self._factor()

    def _build(self):
        nx, ny = self.nx, self.ny
        xs = np.linspace(0.0, self.L, nx + 1)
        ys = np.linspace(-self.h / 2.0, self.h / 2.0, ny + 1)
        self.nodes = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
        conn = []
        for j in range(ny):
            for i in range(nx):
                n0 = j * (nx + 1) + i
                conn.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
        self.conn = np.array(conn, dtype=np.int32)

    def _D(self):
        E, nu = self.E, self.nu
        return E / (1.0 - nu * nu) * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0],
                                               [0.0, 0.0, (1.0 - nu) / 2.0]])

    def _B(self, xi, eta):
        dx, dy = self.L / self.nx, self.h / self.ny
        dNx = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) * 2.0 / dx
        dNy = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) * 2.0 / dy
        B = np.zeros((3, 8))
        B[0, 0::2] = dNx
        B[1, 1::2] = dNy
        B[2, 0::2] = dNy
        B[2, 1::2] = dNx
        return B

    def _assemble(self):
        D = self._D()
        g = 1.0 / math.sqrt(3.0)
        dx, dy = self.L / self.nx, self.h / self.ny
        ke = np.zeros((8, 8))
        for xi in (-g, g):
            for eta in (-g, g):
                B = self._B(xi, eta)
                ke += B.T @ D @ B * (dx * dy / 4.0) * B_WIDTH
        K = np.zeros((2 * len(self.nodes), 2 * len(self.nodes)))
        for conn in self.conn:
            idx = np.empty(8, dtype=np.int32)
            idx[0::2] = 2 * conn
            idx[1::2] = 2 * conn + 1
            K[np.ix_(idx, idx)] += ke
        self.K = K
        self.ndof = 2 * len(self.nodes)
        self.corner_S = [D @ self._B(xi, eta) for xi, eta in
                         ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))]
        self.elem_dof = np.empty((len(self.conn), 8), dtype=np.int32)
        self.elem_dof[:, 0::2] = 2 * self.conn
        self.elem_dof[:, 1::2] = 2 * self.conn + 1

    def _apply_bc(self):
        nx = self.nx
        fixed = [0, 1] + ([2 * nx, 2 * nx + 1] if self.support == "fixed" else [2 * nx + 1])
        self.fixed = np.array(sorted(set(fixed)), dtype=np.int32)

    def _factor(self):
        mask = np.ones(self.ndof, dtype=bool)
        mask[self.fixed] = False
        self.free = np.nonzero(mask)[0]
        Kff = self.K[np.ix_(self.free, self.free)]
        try:
            from scipy.linalg import cho_factor, cho_solve
            self._cho = cho_factor(Kff, lower=False, check_finite=False)
            self._solve = lambda b: cho_solve(self._cho, b, check_finite=False)
        except Exception:
            self._cho = None
            self.Kinv = np.linalg.inv(Kff)
            self._solve = lambda b: self.Kinv @ b

    def solve(self, load, load_pos=None):
        if load_pos is None:
            load_pos = self.L / 2.0
        F = np.zeros(self.ndof)
        nx = self.nx
        t = float(np.clip(load_pos / self.L, 0.0, 1.0)) * nx
        i0 = int(np.floor(t))
        i1 = min(nx, i0 + 1)
        w1 = t - i0
        base = self.ny * (nx + 1)
        F[2 * (base + i0) + 1] -= load * (1.0 - w1)
        F[2 * (base + i1) + 1] -= load * w1
        u = np.zeros(self.ndof)
        u[self.free] = self._solve(F[self.free])
        ue = u[self.elem_dof].T
        best = None
        for S in self.corner_S:
            s = S @ ue
            val = 0.5 * (s[0] + s[1]) + np.hypot(0.5 * (s[0] - s[1]), s[2])
            best = val if best is None else np.maximum(best, val)
        return u, best

    def capacity(self, load_pos=None):
        lo, hi = 0.0, 1.0
        while self.solve(hi, load_pos)[1].max() < self.sigma_ult and hi < 1e12:
            hi *= 2.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.solve(mid, load_pos)[1].max() >= self.sigma_ult:
                hi = mid
            else:
                lo = mid
        return hi


# ==================== Taichi 显式动力学 + 单元失效 ====================
class DynamicBeam:
    """显式动力学：单元内力回代到节点，应力超限就把该单元删掉。

    裂缝不是画出来的——单元失效后刚度消失，相邻单元随即超载，裂纹自己往前跑，
    整根梁最终塌下去，全过程都是算出来的。
    """

    def __init__(self, length, height, E, sigma_ult, rho, support="simple", nx=36, ny=6, nu=0.3):
        self.L, self.h, self.nx, self.ny = length, height, nx, ny
        self.sigma_ult = sigma_ult
        self.support = support
        xs = np.linspace(0.0, length, nx + 1)
        ys = np.linspace(-height / 2.0, height / 2.0, ny + 1)
        nodes = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
        conn = []
        for j in range(ny):
            for i in range(nx):
                n0 = j * (nx + 1) + i
                conn.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
        conn = np.array(conn, dtype=np.int32)
        self.nodes, self.conn = nodes, conn
        n_nodes, n_elem = len(nodes), len(conn)

        dx, dy = length / nx, height / ny
        D = E / (1 - nu * nu) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

        def Bmat(xi, eta):
            dNx = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) * 2.0 / dx
            dNy = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) * 2.0 / dy
            B = np.zeros((3, 8))
            B[0, 0::2] = dNx
            B[1, 1::2] = dNy
            B[2, 0::2] = dNy
            B[2, 1::2] = dNx
            return B

        g = 1.0 / math.sqrt(3.0)
        KE = np.zeros((8, 8))
        for xi in (-g, g):
            for eta in (-g, g):
                B = Bmat(xi, eta)
                KE += B.T @ D @ B * (dx * dy / 4.0) * B_WIDTH
        S = np.stack([D @ Bmat(xi, eta) for xi, eta in
                      ((-1, -1), (1, -1), (1, 1), (-1, 1))]).astype(np.float32)
        mass = np.full(n_nodes, rho * MASS_SCALE * dx * dy * B_WIDTH / 4.0, dtype=np.float32)

        is_fixed = np.zeros((n_nodes, 2), dtype=np.int32)
        is_fixed[0] = (1, 1)
        is_fixed[nx] = (1, 1) if support == "fixed" else (0, 1)

        c = math.sqrt(E / (rho * MASS_SCALE))
        self.dt = 0.30 * min(dx, dy) / c
        self.damp = 0.995

        ti.reset()
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        self.pos = ti.Vector.field(2, ti.f32, n_nodes)
        self.vel = ti.Vector.field(2, ti.f32, n_nodes)
        self.force = ti.Vector.field(2, ti.f32, n_nodes)
        self.rest = ti.Vector.field(2, ti.f32, n_nodes)
        self.broken = ti.field(ti.i32, n_elem)
        self.stress = ti.field(ti.f32, n_elem)
        self.ke_f = ti.field(ti.f32, (8, 8))
        self.s_f = ti.field(ti.f32, (4, 3, 8))
        self.mass_f = ti.field(ti.f32, n_nodes)
        self.conn_f = ti.field(ti.i32, (n_elem, 4))
        self.fix_f = ti.field(ti.i32, (n_nodes, 2))

        self.pos.from_numpy(nodes)
        self.rest.from_numpy(nodes)
        self.ke_f.from_numpy(KE.astype(np.float32))
        self.s_f.from_numpy(S)
        self.mass_f.from_numpy(mass)
        self.conn_f.from_numpy(conn)
        self.fix_f.from_numpy(is_fixed)
        self.n_nodes, self.n_elem = n_nodes, n_elem
        self.load_node = self.ny * (self.nx + 1) + self.nx // 2
        self._compile()

    def _compile(self):
        pos, vel, force, broken, stress = self.pos, self.vel, self.force, self.broken, self.stress
        rest = self.rest
        ke_f, s_f, mass_f, conn_f, fix_f = self.ke_f, self.s_f, self.mass_f, self.conn_f, self.fix_f
        n_nodes, n_elem, load_node = self.n_nodes, self.n_elem, self.load_node

        ground = -0.30 - self.h  # 地面高度（米），碎块落到这里停住

        rest_area = (self.L / self.nx) * (self.h / self.ny)
        rest_dx = self.L / self.nx
        rest_dy = self.h / self.ny

        @ti.kernel
        def step_kernel(load: ti.f32, sigma: ti.f32, dt: ti.f32, damp: ti.f32):
            for i in range(n_nodes):
                force[i] = ti.Vector([0.0, 0.0])
            force[load_node] += ti.Vector([0.0, -load])
            for e in range(n_elem):
                ue = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                for k in ti.static(range(4)):
                    n = conn_f[e, k]
                    ue[2 * k] = pos[n][0] - rest[n][0]
                    ue[2 * k + 1] = pos[n][1] - rest[n][1]
                if broken[e] == 0:
                    fe = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    for i in ti.static(range(8)):
                        acc = 0.0
                        for j in ti.static(range(8)):
                            acc += ke_f[i, j] * ue[j]
                        fe[i] = acc
                    for k in ti.static(range(4)):
                        ti.atomic_add(force[conn_f[e, k]][0], -fe[2 * k])
                        ti.atomic_add(force[conn_f[e, k]][1], -fe[2 * k + 1])
                    smax = -1e30
                    smin_all = 1e30
                    for c in ti.static(range(4)):
                        sx = 0.0
                        sy = 0.0
                        sxy = 0.0
                        for j in ti.static(range(8)):
                            sx += s_f[c, 0, j] * ue[j]
                            sy += s_f[c, 1, j] * ue[j]
                            sxy += s_f[c, 2, j] * ue[j]
                        rad = ti.sqrt(0.25 * (sx - sy) ** 2 + sxy * sxy)
                        val = 0.5 * (sx + sy) + rad
                        smin = 0.5 * (sx + sy) - rad
                        smax = ti.max(smax, val)
                        smin_all = ti.min(smin_all, smin)
                    stress[e] = smax
                    # 单元被拉到严重畸变也删掉：避免裂口拉出一条橡皮筋似的长条
                    area = 0.0
                    cx = 0.0
                    cy = 0.0
                    for k in ti.static(range(4)):
                        n0 = conn_f[e, k]
                        n1 = conn_f[e, (k + 1) % 4]
                        area += pos[n0][0] * pos[n1][1] - pos[n1][0] * pos[n0][1]
                        cx += pos[n0][0]
                        cy += pos[n0][1]
                    area *= 0.5
                    # 边被拉到原长 4 倍以上也删（否则裂口会拉出一条细长的“橡皮筋”）
                    e0 = (pos[conn_f[e, 1]] - pos[conn_f[e, 0]]).norm()
                    e1 = (pos[conn_f[e, 2]] - pos[conn_f[e, 1]]).norm()
                    stretched = e0 > 4.0 * rest_dx or e1 > 4.0 * rest_dy
                    # 拉断、压碎、面积塌缩/膨胀、或严重拉长都让单元失效
                    if (smax > sigma or smin_all < -1.6 * sigma
                            or area < 0.35 * rest_area or area > 3.5 * rest_area or stretched):
                        broken[e] = 1
            for i in range(n_nodes):
                if fix_f[i, 0] == 0:
                    vel[i][0] += force[i][0] / mass_f[i] * dt
                    vel[i][0] *= damp
                    pos[i][0] += vel[i][0] * dt
                if fix_f[i, 1] == 0:
                    vel[i][1] += force[i][1] / mass_f[i] * dt
                    vel[i][1] *= damp
                    pos[i][1] += vel[i][1] * dt
                    if pos[i][1] < ground:
                        pos[i][1] = ground
                        vel[i][1] = 0.0
        self._step_kernel = step_kernel

    def step(self, load, substeps=6):
        for k in range(substeps):
            damp = self.damp if k == 0 else 1.0
            self._step_kernel(load, self.sigma_ult, self.dt, damp)

    def set_state(self, disp, broken_idx=None):
        """把静态解作为初始状态装进来，并可以预先弄断指定单元。"""
        self.pos.from_numpy((self.nodes + disp.reshape(-1, 2)).astype(np.float32))
        self.vel.from_numpy(np.zeros((self.n_nodes, 2), dtype=np.float32))
        if broken_idx is not None:
            m = np.zeros(self.n_elem, dtype=np.int32)
            m[broken_idx] = 1
            self.broken.from_numpy(m)

    def positions(self):
        return self.pos.to_numpy()

    def stresses(self):
        return self.stress.to_numpy()

    def broken_mask(self):
        return self.broken.to_numpy() > 0

    def broken_count(self):
        return int((self.broken.to_numpy() > 0).sum())


# ==================== 画布 ====================
class BeamCanvas(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumSize(560, 320)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(248, 248, 246))
        app = self.app
        dyn = app.dyn
        if dyn is None or app.pos is None:
            p.end()
            return

        w, hh = self.width(), self.height()
        margin_x = 90
        span = max(1e-6, dyn.L)
        scale_x = (w - 2 * margin_x) / span
        base_y = hh * 0.50

        pos = app.pos
        px = margin_x + pos[:, 0] * scale_x
        py = base_y - pos[:, 1] * scale_x
        stress = app.elem_stress
        broken = app.broken_mask
        limit = dyn.sigma_ult

        for node in (0, dyn.nx):
            sx = margin_x + dyn.nodes[node, 0] * scale_x
            sy = base_y - dyn.nodes[node, 1] * scale_x + 6
            p.setBrush(QColor(90, 90, 96))
            p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygonF([QPointF(sx, sy), QPointF(sx - 15, sy + 22),
                                     QPointF(sx + 15, sy + 22)]))
            p.setPen(QPen(QColor(150, 150, 155), 2))
            p.drawLine(QPointF(sx - 22, sy + 22), QPointF(sx + 22, sy + 22))

        ground_y = base_y + (0.30 + dyn.h) * scale_x
        p.setPen(QPen(QColor(200, 196, 188), 2))
        p.drawLine(QPointF(margin_x - 40, ground_y), QPointF(margin_x + span * scale_x + 40, ground_y))

        for e, conn in enumerate(dyn.conn):
            if broken[e]:
                continue        # 失效单元不再参与绘制，裂口是真的缺口
            else:
                t = float(np.clip(abs(stress[e]) / limit, 0.0, 1.0))
                if stress[e] >= 0:
                    col = QColor(int(150 + 100 * t), int(175 - 120 * t), int(210 - 170 * t))
                else:
                    col = QColor(int(120 - 25 * t), int(155 + 35 * t), int(215))
                p.setBrush(col)
                p.setPen(QPen(QColor(110, 110, 120, 80), 0.5))
            p.drawPolygon(QPolygonF([QPointF(px[i], py[i]) for i in conn]))

        if app.load > 0:
            nid = dyn.load_node
            sx = float(px[nid])
            top = float(py[nid]) - 86
            frac = min(1.0, app.load / max(1.0, app.capacity_ref or app.load * 1.2))
            bw, bh = 36 + 44 * frac, 20 + 22 * frac
            p.setBrush(QColor(70, 78, 92))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(sx - bw / 2, top, bw, bh), 4, 4)
            p.setPen(QColor(255, 255, 255))
            f = p.font()
            f.setPointSize(8)
            p.setFont(f)
            p.drawText(QRectF(sx - bw / 2, top, bw, bh), Qt.AlignCenter, f"{app.load / G:.0f}kg")

        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        p.setPen(QColor(90, 90, 96))
        p.drawText(QRectF(16, hh - 26, 620, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   f"Taichi 显式动力学 · 失效单元 {int(broken.sum())}/{len(broken)} · 主应力上色（红＝受拉）")
        p.end()


# ==================== 数据图 ====================
class SweepChart(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumHeight(210)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(255, 255, 253))
        w, h = self.width(), self.height()
        left, right, top, bottom = 76, 24, 32, 48
        pw, ph = w - left - right, h - top - bottom

        f = p.font()
        f.setPointSize(10)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(45, 45, 50))
        mode = self.app.sweep_mode
        title = {"height": "梁高对极限承载力的影响", "length": "梁长对极限承载力的影响"}.get(
            mode, "运行一次扫描，得到承重与尺寸的关系")
        p.drawText(QRectF(left, 4, pw, 20), Qt.AlignCenter, title)
        f.setBold(False)
        f.setPointSize(8)
        p.setFont(f)

        runs = self.app.runs
        p.setPen(QColor(70, 70, 76))
        p.drawLine(QPointF(left, top), QPointF(left, top + ph))
        p.drawLine(QPointF(left, top + ph), QPointF(left + pw, top + ph))
        if not runs:
            p.end()
            return

        xs = np.array([r["x"] for r in runs])
        ys = np.array([r["P"] for r in runs]) / 1000.0
        x_lo, x_hi = float(xs.min()), float(xs.max())
        pad = (x_hi - x_lo) * 0.06 + 1e-6
        x_lo -= pad
        x_hi += pad
        y_hi = float(ys.max()) * 1.16

        def X(v):
            return left + (v - x_lo) / max(1e-9, x_hi - x_lo) * pw

        def Y(v):
            return top + ph - v / max(1e-9, y_hi) * ph

        p.setPen(QPen(QColor(228, 228, 232), 1))
        for i in range(5):
            yy = top + ph * i / 4.0
            p.drawLine(QPointF(left, yy), QPointF(left + pw, yy))
        p.setPen(QColor(90, 90, 96))
        for i in range(5):
            v = y_hi * (1 - i / 4.0)
            yy = top + ph * i / 4.0
            p.drawText(QRectF(8, yy - 8, left - 16, 16), Qt.AlignRight | Qt.AlignVCenter, f"{v:.0f}")
            p.drawLine(QPointF(left - 4, yy), QPointF(left, yy))
        for i in range(5):
            v = x_lo + (x_hi - x_lo) * i / 4.0
            xx = X(v)
            p.drawText(QRectF(xx - 32, top + ph + 8, 64, 14), Qt.AlignCenter, f"{v:.2f}")
            p.drawLine(QPointF(xx, top + ph), QPointF(xx, top + ph + 4))

        xlabel = "梁高 h (cm)" if mode == "height" else "梁长 L (m)"
        p.drawText(QRectF(left, h - 20, pw, 16), Qt.AlignCenter, xlabel)
        p.save()
        p.translate(16, top + ph / 2)
        p.rotate(-90)
        p.drawText(QRectF(-ph / 2, -14, ph, 16), Qt.AlignCenter, "极限载荷 P (kN)")
        p.restore()

        if len(xs) > 2:
            b, ln_a = np.polyfit(np.log(xs), np.log(ys), 1)
            pred = np.exp(ln_a) * xs ** b
            ss_res = float(((ys - pred) ** 2).sum())
            ss_tot = float(((ys - ys.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / max(1e-12, ss_tot)
            gx = np.linspace(x_lo + pad, x_hi - pad, 60)
            gy = np.exp(ln_a) * gx ** b
            p.setPen(QPen(QColor(70, 110, 190), 2))
            p.drawPolyline(QPolygonF([QPointF(X(a), Y(bv)) for a, bv in zip(gx, gy)]))
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(50, 80, 160))
            p.drawText(QRectF(left + 10, top + 4, pw - 20, 16), Qt.AlignLeft | Qt.AlignVCenter,
                       f"拟合  P ∝ {'h' if mode == 'height' else 'L'} ^{b:.2f}    R² = {r2:.4f}")
            f.setBold(False)
            p.setFont(f)

        p.setPen(QPen(QColor(215, 85, 55), 1.2))
        p.setBrush(QColor(215, 85, 55))
        for x, y in zip(xs, ys):
            px_, py_ = X(x), Y(y)
            p.drawEllipse(QPointF(px_, py_), 3.6, 3.6)
            p.setPen(QColor(120, 60, 45))
            p.drawText(QRectF(px_ - 34, py_ - 20, 68, 14), Qt.AlignCenter, f"{y:.1f}")
            p.setPen(QPen(QColor(215, 85, 55), 1.2))
        p.end()


# ==================== 主窗口 ====================
class BeamApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("横梁承重 · Taichi 实时破坏模拟")
        self.setGeometry(110, 60, 1200, 840)

        self.material = "松木"
        self.support = "simple"
        self.beam_length = 1.6
        self.beam_height = 0.12
        self.load = 0.0
        self.broken = False
        self.running = False
        self.runs = []
        self.sweep_mode = None
        self.dyn = None
        self.pos = None
        self.broken_mask = None
        self.elem_stress = np.zeros(1)
        self._pending_rebuild = False
        self.mode = "quasi"
        self.substeps = 12
        self.collapse_frames = 0

        self._rebuild_dyn()
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

        self.rebuild_timer = QTimer(self)
        self.rebuild_timer.setSingleShot(True)
        self.rebuild_timer.timeout.connect(self.apply_pending)

    # ---------- 模型 ----------
    def _rebuild_dyn(self):
        E, sig, nu, rho = MATERIALS[self.material]
        self.dyn = DynamicBeam(self.beam_length, self.beam_height, E, sig, rho,
                               support=self.support, nu=nu)
        self.static = StaticBeam(self.beam_length, self.beam_height, E, sig,
                                 support=self.support, nu=nu)
        self.capacity_ref = self.static.capacity()
        self.load = 0.0
        self.broken = False
        self.mode = "quasi"
        self.pos = self.dyn.positions()
        self.broken_mask = self.dyn.broken_mask()
        self.elem_stress = self.dyn.stresses()
        if hasattr(self, "lbl_cap"):
            self.lbl_cap.setText(f"理论极限 {self.capacity_ref / 1000:.1f} kN")

    def reset_beam(self):
        self._rebuild_dyn()
        self.mode = "quasi"
        self.running = False
        self.run_btn.setText("开始加载")
        self.canvas.update()
        self.update_stats()
        self.lbl_status.setText("已复位")

    def tick(self):
        if self.dyn is None:
            return
        if self.mode == "quasi":
            if not self.running:
                return
            # 准静态：直接解平衡方程，形变与应力都是精确的
            frac = self.load / max(1.0, self.capacity_ref)
            step = max(8.0, self.capacity_ref * 0.010) * (1.0 if frac < 0.75 else 0.4)
            self.load += step
            u, sig = self.static.solve(self.load)
            self.pos = (self.static.nodes + u.reshape(-1, 2)).astype(np.float32)
            self.elem_stress = sig
            self.broken_mask = np.zeros(len(sig), dtype=bool)
            if sig.max() >= self.static.sigma_ult:
                self.start_collapse(sig)
        else:
            # 动力学塌落：单元失效后刚度消失，裂纹自己扩展
            self.dyn.step(self.load, substeps=self.substeps)
            self.pos = self.dyn.positions()
            self.elem_stress = self.dyn.stresses()
            self.broken_mask = self.dyn.broken_mask()
            self.collapse_frames -= 1
            if self.collapse_frames <= 0:
                self.mode = "done"
                self.lbl_status.setText(
                    f"塌落结束：{int(self.broken_mask.sum())}/{len(self.broken_mask)} 个单元失效，"
                    f"载荷 {self.load / 1000:.2f} kN（{self.load / G:.0f} kg）")
        self.canvas.update()
        self.update_stats()

    def start_collapse(self, sig):
        """应力首次超过强度：把最大应力单元弄断，切到显式动力学算塌落。"""
        k = int(np.argmax(sig))
        u, _ = self.static.solve(self.load)
        self.dyn.set_state(u, broken_idx=[k])
        self.mode = "dynamic"
        self.collapse_frames = 90
        self.running = False
        self.run_btn.setText("开始加载")
        self.runs.append({"L": self.beam_length, "h": self.beam_height, "P": self.load,
                          "x": self.beam_height * 100 if self.sweep_mode == "height"
                          else self.beam_length})
        self.chart.update()
        self.broken = True
        self.lbl_status.setText(
            f"跨中单元被拉断！载荷 {self.load / 1000:.2f} kN（{self.load / G:.0f} kg），进入塌落模拟")

    # ---------- UI ----------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(272)
        pv = QVBoxLayout(panel)
        pv.setSpacing(6)
        pv.setContentsMargins(0, 0, 0, 0)
        self.canvas = BeamCanvas(self)
        self._build_beam_group(pv)
        self._build_load_group(pv)
        self._build_sweep_group(pv)
        pv.addStretch()
        root.addWidget(panel)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(6)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.canvas, 1)
        self.chart = SweepChart(self)
        rv.addWidget(self.chart)
        root.addWidget(right, 1)

        self.lbl_status = QLabel("点「开始加载」：载荷实时上升，单元逐个被撕开")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_load = QLabel("载荷 0 kg")
        self.lbl_stress = QLabel("应力 –")
        self.lbl_defl = QLabel("挠度 –")
        for wdg in (self.lbl_load, self.lbl_stress, self.lbl_defl):
            self.statusBar().addPermanentWidget(wdg)

    def _build_beam_group(self, parent):
        group = QGroupBox("梁")
        form = QFormLayout(group)
        form.setSpacing(4)
        form.setContentsMargins(6, 6, 6, 6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.mat_combo = QComboBox()
        self.mat_combo.addItems(list(MATERIALS.keys()))
        self.mat_combo.currentTextChanged.connect(lambda _n: self.schedule_rebuild())
        form.addRow("材料", self.mat_combo)
        self.sup_combo = QComboBox()
        self.sup_combo.addItems(["简支（钉子可转）", "固支（完全固定）"])
        self.sup_combo.currentIndexChanged.connect(lambda _i: self.schedule_rebuild())
        form.addRow("支承", self.sup_combo)
        self.len_slider, self.len_label = self._slider(form, "长度 L", 30, 300,
                                                       int(self.beam_length * 100),
                                                       lambda v: v / 100.0, " m")
        self.hgt_slider, self.hgt_label = self._slider(form, "高度 h", 2, 30,
                                                       int(self.beam_height * 100),
                                                       lambda v: v / 100.0, " m")
        parent.addWidget(group)

    def _build_load_group(self, parent):
        group = QGroupBox("加载")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)
        self.run_btn = QPushButton("开始加载")
        self.run_btn.clicked.connect(self.toggle_run)
        vbox.addWidget(self.run_btn)
        row = QHBoxLayout()
        row.setSpacing(4)
        btn_reset = QPushButton("复位")
        btn_reset.clicked.connect(self.reset_beam)
        row.addWidget(btn_reset)
        btn_cap = QPushButton("理论极限")
        btn_cap.clicked.connect(self.show_capacity)
        row.addWidget(btn_cap)
        vbox.addLayout(row)
        self.lbl_cap = QLabel(f"理论极限 {self.capacity_ref / 1000:.1f} kN")
        vbox.addWidget(self.lbl_cap)
        parent.addWidget(group)

    def _build_sweep_group(self, parent):
        group = QGroupBox("实验：承重与尺寸的关系")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)
        b1 = QPushButton("扫描梁高（固定长度）")
        b1.clicked.connect(lambda: self.sweep("height"))
        vbox.addWidget(b1)
        b2 = QPushButton("扫描梁长（固定高度）")
        b2.clicked.connect(lambda: self.sweep("length"))
        vbox.addWidget(b2)
        b3 = QPushButton("清空数据")
        b3.clicked.connect(self.clear_runs)
        vbox.addWidget(b3)
        parent.addWidget(group)

    def _slider(self, form, text, lo, hi, value, transform, suffix=""):
        row = QHBoxLayout()
        row.setSpacing(4)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        label = QLabel(f"{transform(value):.2f}{suffix}")
        label.setFixedWidth(56)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def on_change(v):
            label.setText(f"{transform(v):.2f}{suffix}")
            self.schedule_rebuild()

        slider.valueChanged.connect(on_change)
        row.addWidget(slider, 1)
        row.addWidget(label)
        form.addRow(text, row)
        return slider, label

    # ---------- 交互 ----------
    def schedule_rebuild(self):
        self._pending_rebuild = True
        self.rebuild_timer.start(220)

    def apply_pending(self):
        if not self._pending_rebuild:
            return
        self._pending_rebuild = False
        self.material = self.mat_combo.currentText()
        self.support = "simple" if self.sup_combo.currentIndex() == 0 else "fixed"
        self.beam_length = self.len_slider.value() / 100.0
        self.beam_height = self.hgt_slider.value() / 100.0
        self._rebuild_dyn()
        self.running = False
        self.run_btn.setText("开始加载")
        self.canvas.update()
        self.update_stats()

    def toggle_run(self):
        if self.dyn is None:
            return
        if self.broken or self.mode == "done":
            self.reset_beam()
            return
        self.running = not self.running
        self.run_btn.setText("暂停" if self.running else "开始加载")
        self.lbl_status.setText("实时加载中…" if self.running else "已暂停")

    def show_capacity(self):
        self.lbl_cap.setText(f"理论极限 {self.capacity_ref / 1000:.1f} kN（{self.capacity_ref / G:.0f} kg）")
        self.lbl_status.setText(f"静态有限元给出的极限：{self.capacity_ref / G:.0f} kg")

    def sweep(self, mode):
        self.sweep_mode = mode
        self.runs = []
        E, sig, nu, _rho = MATERIALS[self.material]
        if mode == "height":
            for v in np.linspace(0.02, 0.30, 9):
                sim = StaticBeam(self.beam_length, v, E, sig, self.support, nu=nu)
                self.runs.append({"L": self.beam_length, "h": v, "P": sim.capacity(), "x": v * 100})
        else:
            for v in np.linspace(0.5, 3.0, 9):
                sim = StaticBeam(v, self.beam_height, E, sig, self.support, nu=nu)
                self.runs.append({"L": v, "h": self.beam_height, "P": sim.capacity(), "x": v})
        self.chart.update()
        self.lbl_status.setText(f"已扫描 {len(self.runs)} 组数据")

    def clear_runs(self):
        self.runs = []
        self.chart.update()
        self.lbl_status.setText("数据已清空")

    def update_stats(self):
        self.lbl_load.setText(f"载荷 {self.load / G:.0f} kg")
        if self.dyn is not None and self.pos is not None:
            defl = float(np.abs(self.pos[:, 1] - self.dyn.nodes[:, 1]).max())
            self.lbl_defl.setText(f"挠度 {defl * 1000:.2f} mm")
            smax = float(np.abs(self.elem_stress).max())
            self.lbl_stress.setText(
                f"主应力 {smax / 1e6:.1f} MPa（{smax / self.dyn.sigma_ult * 100:.0f}%）")

    def closeEvent(self, event):
        self.timer.stop()
        try:
            ti.reset()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BeamApp()
    window.show()
    sys.exit(app.exec())
