from custom_import import *
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPolygonF

import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ==================== 材料 ====================
G = 9.81
B_WIDTH = 0.06          # 梁宽（出平面方向）

N_X = 64                # 梁长方向单元数
N_Y = 10                # 梁高方向单元数

MATERIALS = {
    "松木": (10e9, 40e6, 0.30, 600.0),
    "橡木": (12e9, 55e6, 0.30, 700.0),
    "钢材": (200e9, 250e6, 0.28, 7800.0),
    "混凝土": (30e9, 3e6, 0.20, 2400.0),
}


# ==================== 静态有限元（测极限载荷 / 扫描） ====================
class StaticBeam:
    """平面应力 Q4 + Cholesky；用来快速求极限载荷。"""

    def __init__(self, length, height, E, sigma_ult, support="simple", nx=N_X, ny=N_Y, nu=0.3):
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
        # 钉子钉在梁端面的中高处（不是底角），更像“两端钉住”
        nx = self.nx
        mid = self.ny // 2
        left = mid * (nx + 1)
        right = left + nx
        fixed = [2 * left, 2 * left + 1]
        if self.support == "fixed":
            fixed += [2 * right, 2 * right + 1]
        else:
            fixed += [2 * right + 1]
        self.left_node, self.right_node = left, right
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


# ==================== 隐式动力学 + 单元失效 ====================
class ImplicitBeam:
    """后向差分隐式动力学 + 稀疏 LU，单元应力/失效判定交给 Taichi 并行。

    这是 Taichi 官方 FEM 例子的做法：显式时间步受 CFL 限制，这里只能取到
    ~9 μs，而梁的一阶周期约 70 ms，纯显式根本跑不到准静态；换成隐式积分后
    时间步可以取毫秒级，一次分解、每步只做回代，才可能实时。
    """

    def __init__(self, length, height, E, sigma_ult, rho, support="simple", nx=N_X, ny=N_Y,
                 nu=0.3, dt=1.2e-3):
        self.L, self.h, self.nx, self.ny = length, height, nx, ny
        self.sigma_ult, self.rho, self.dt = sigma_ult, rho, dt
        self.support = support
        xs = np.linspace(0.0, length, nx + 1)
        ys = np.linspace(-height / 2.0, height / 2.0, ny + 1)
        self.nodes = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
        conn = []
        for j in range(ny):
            for i in range(nx):
                n0 = j * (nx + 1) + i
                conn.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
        self.conn = np.array(conn, dtype=np.int32)
        self.n_nodes, self.n_elem = len(self.nodes), len(self.conn)
        self.ndof = 2 * self.n_nodes

        dx, dy = length / nx, height / ny
        self.dx, self.dy = dx, dy
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
        ke = np.zeros((8, 8))
        for xi in (-g, g):
            for eta in (-g, g):
                B = Bmat(xi, eta)
                ke += B.T @ D @ B * (dx * dy / 4.0) * B_WIDTH
        self.ke = ke
        self.corner_S = np.stack([D @ Bmat(xi, eta) for xi, eta in
                                  ((-1, -1), (1, -1), (1, 1), (-1, 1))]).astype(np.float32)
        self.elem_dof = np.empty((self.n_elem, 8), dtype=np.int32)
        self.elem_dof[:, 0::2] = 2 * self.conn
        self.elem_dof[:, 1::2] = 2 * self.conn + 1

        node_mass = rho * dx * dy * B_WIDTH / 4.0
        self.mass = np.full(self.ndof, node_mass)

        mid = ny // 2
        left = mid * (nx + 1)
        right = left + nx
        fixed = [2 * left, 2 * left + 1]
        fixed += ([2 * right, 2 * right + 1] if support == "fixed" else [2 * right + 1])
        self.left_node, self.right_node = left, right
        self.fixed = np.array(sorted(set(fixed)), dtype=np.int32)

        self.broken = np.zeros(self.n_elem, dtype=bool)
        self.u = np.zeros(self.ndof)
        self.u_prev = np.zeros(self.ndof)
        self.alpha = 18.0      # 质量阻尼
        self.load_node = self.ny * (nx + 1) + nx // 2
        self._build_matrices()
        self._init_taichi()

    # ---------- 稀疏矩阵 ----------
    def _build_matrices(self):
        rows, cols, vals = [], [], []
        for e in range(self.n_elem):
            if self.broken[e]:
                continue
            dofs = self.elem_dof[e]
            for i in range(8):
                for j in range(8):
                    rows.append(dofs[i])
                    cols.append(dofs[j])
                    vals.append(self.ke[i, j])
        K = sp.coo_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof)).tocsc()
        self.K = K

        # 找出孤立节点（周围没有完好单元）并临时固定，否则它们只有质量没有刚度，
        # 方程会解出天文数字的位移
        used = np.zeros(self.n_nodes, dtype=bool)
        for e in range(self.n_elem):
            if not self.broken[e]:
                used[self.conn[e]] = True
        orphan = np.nonzero(~used)[0]
        self.orphan_dofs = np.concatenate([2 * orphan, 2 * orphan + 1]) if orphan.size else np.array([], dtype=np.int32)

        dt = self.dt
        M = sp.diags(self.mass, format="csc")
        A = (M * (1.0 / dt ** 2) + M * (self.alpha / dt) + K).tolil()
        self.all_fixed = np.unique(np.concatenate([self.fixed, self.orphan_dofs]))
        for d in self.all_fixed:
            A.rows[d] = [d]
            A.data[d] = [1.0]
        self.A = A.tocsc()
        self._lu = spla.splu(self.A)

    # ---------- Taichi：单元应力与失效 ----------
    def _init_taichi(self):
        ti.reset()
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        self.t_dis = ti.field(ti.f32, 2 * self.n_nodes)
        self.t_stress = ti.field(ti.f32, self.n_elem)
        self.t_broken = ti.field(ti.i32, self.n_elem)
        self.t_s = ti.field(ti.f32, (4, 3, 8))
        self.t_conn = ti.field(ti.i32, (self.n_elem, 4))
        self.t_s.from_numpy(self.corner_S)
        self.t_conn.from_numpy(self.conn)
        dis, stress, broken, s_f, conn_f = (self.t_dis, self.t_stress, self.t_broken,
                                            self.t_s, self.t_conn)
        n_elem = self.n_elem
        sigma = self.sigma_ult

        @ti.kernel
        def kern():
            for e in range(n_elem):
                if broken[e] != 0:
                    continue
                ue = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                for k in ti.static(range(4)):
                    n = conn_f[e, k]
                    ue[2 * k] = dis[2 * n]
                    ue[2 * k + 1] = dis[2 * n + 1]
                smax = -1e30
                smin = 1e30
                for c in ti.static(range(4)):
                    sx = 0.0
                    sy = 0.0
                    sxy = 0.0
                    for j in ti.static(range(8)):
                        sx += s_f[c, 0, j] * ue[j]
                        sy += s_f[c, 1, j] * ue[j]
                        sxy += s_f[c, 2, j] * ue[j]
                    rad = ti.sqrt(0.25 * (sx - sy) ** 2 + sxy * sxy)
                    smax = ti.max(smax, 0.5 * (sx + sy) + rad)
                    smin = ti.min(smin, 0.5 * (sx + sy) - rad)
                stress[e] = smax
                if smax > sigma or smin < -1.6 * sigma:
                    broken[e] = 1
        self._kern = kern

    # ---------- 推进 ----------
    def step(self, load, substeps=3):
        dt = self.dt
        M = self.mass
        alpha = self.alpha
        new_break = False
        for _ in range(substeps):
            rhs = M * (2.0 * self.u - self.u_prev) / dt ** 2
            rhs = rhs + M * (alpha / dt) * self.u
            rhs[2 * self.load_node + 1] -= load
            rhs[self.all_fixed] = 0.0
            u_new = self._lu.solve(rhs)
            self.u_prev, self.u = self.u, u_new
            self.t_dis.from_numpy(self.u.astype(np.float32))
            self._kern()
            if int((self.t_broken.to_numpy() > 0).sum()) > int(self.broken.sum()):
                self.broken = self.t_broken.to_numpy() > 0
                self._relocate_load()
                self._build_matrices()
                new_break = True
                break
        return new_break

    def _relocate_load(self):
        """断裂后把重物挪到最近的、仍然完好的顶面节点上。

        否则载荷挂在已经失效的节点上，力等于凭空消失，梁会自己弹回去。
        """
        nx = self.nx
        top0 = self.ny * (nx + 1)
        alive = ~self.broken
        best, best_cost = self.load_node, 1e30
        for i in range(nx + 1):
            # 该顶面节点周围的单元是否还在
            near = []
            if i > 0:
                near.append(top0 - 1 + i - 1)
            if i < nx:
                near.append(top0 - 1 + i)
            cnt = sum(1 for e in near if 0 <= e < self.n_elem and alive[e])
            if cnt == 0:
                continue
            cost = abs(i - nx / 2.0) - 10.0 * cnt
            if cost < best_cost:
                best_cost, best = cost, top0 + i
        self.load_node = best

    # ---------- 读回 ----------
    def positions(self):
        return (self.nodes + self.u.reshape(-1, 2)).astype(np.float32)

    def stresses(self):
        return self.t_stress.to_numpy()

    def broken_mask(self):
        return self.broken.copy()

    def broken_count(self):
        return int(self.broken.sum())


# ==================== 画布 ====================
class BeamCanvas(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumSize(560, 320)

    def _top_surface_y(self, sim, x_pos, mx, sx, base_y):
        """梁顶面在给定 x 处的像素 y（跟着变形走）。"""
        top0 = sim.ny * (sim.nx + 1)
        pos = self.app.pos
        i = int(np.clip(round(x_pos / sim.L * sim.nx), 0, sim.nx))
        x0, y0 = float(pos[top0 + i, 0]), float(pos[top0 + i, 1])
        if i < sim.nx:
            x1, y1 = float(pos[top0 + i + 1, 0]), float(pos[top0 + i + 1, 1])
            if abs(x1 - x0) > 1e-9:
                frac = float(np.clip((x_pos - x0) / (x1 - x0), 0.0, 1.0))
                y0 = y0 + (y1 - y0) * frac
        return base_y - y0 * sx

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(248, 248, 246))
        app = self.app
        sim = app.sim
        if sim is None or app.pos is None:
            p.end()
            return

        w, hh = self.width(), self.height()
        margin_x = 90
        span = max(1e-6, sim.L)
        scale_x = (w - 2 * margin_x) / span
        base_y = hh * 0.46

        pos = app.pos
        px = margin_x + pos[:, 0] * scale_x
        py = base_y - pos[:, 1] * scale_x
        stress = app.elem_stress
        broken = app.broken_mask
        limit = sim.sigma_ult

        # 两侧木条支座（钉子钉在端面中高处）
        post_w = 15.0
        ground_y = base_y + (sim.h * 0.5 + 0.30) * scale_x
        for node in (sim.left_node, sim.right_node):
            sx = margin_x + float(pos[node, 0]) * scale_x
            top_y = base_y - (float(pos[node, 1]) - sim.h * 0.5) * scale_x
            wood = QColor(176, 136, 92)
            p.setBrush(wood)
            p.setPen(QPen(QColor(122, 92, 58), 1.2))
            p.drawRect(QRectF(sx - post_w / 2, top_y, post_w, ground_y - top_y))
            # 木纹
            p.setPen(QPen(QColor(150, 114, 74), 1))
            for k in (-4, 0, 4):
                p.drawLine(QPointF(sx + k, top_y + 3), QPointF(sx + k, ground_y - 3))
            # 钉子
            p.setBrush(QColor(70, 70, 75))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(sx, base_y - float(pos[node, 1]) * scale_x), 4.5, 4.5)

        for e, conn in enumerate(sim.conn):
            if broken[e]:
                continue
            t = float(np.clip(abs(stress[e]) / limit, 0.0, 1.0))
            if stress[e] >= 0:
                col = QColor(int(150 + 100 * t), int(175 - 120 * t), int(210 - 170 * t))
            else:
                col = QColor(int(120 - 25 * t), int(155 + 35 * t), int(215))
            p.setBrush(col)
            p.setPen(QPen(QColor(110, 110, 120, 70), 0.5))
            p.drawPolygon(QPolygonF([QPointF(px[i], py[i]) for i in conn]))

        if app.load > 0:
            x_pos = app.load_pos if app.load_pos is not None else span / 2.0
            sx = margin_x + x_pos * scale_x
            surf_y = self._top_surface_y(sim, x_pos, margin_x, scale_x, base_y)
            frac = min(1.0, app.load / max(1.0, app.capacity_ref or app.load * 1.2))
            bw, bh = 40 + 46 * frac, 24 + 26 * frac
            top = surf_y - bh - 1
            p.setBrush(QColor(66, 74, 90))
            p.setPen(QPen(QColor(40, 46, 58), 1))
            p.drawRoundedRect(QRectF(sx - bw / 2, top, bw, bh), 4, 4)
            p.setPen(QColor(255, 255, 255))
            f = p.font()
            f.setPointSize(8)
            p.setFont(f)
            p.drawText(QRectF(sx - bw / 2, top, bw, bh), Qt.AlignCenter, f"{app.load / G:.0f}kg")
            p.setPen(QPen(QColor(160, 60, 50), 2))
            p.drawLine(QPointF(sx, top + bh), QPointF(sx, surf_y - 1))

        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        p.setPen(QColor(90, 90, 96))
        p.drawText(QRectF(16, hh - 26, 660, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   f"隐式动力学实时求解（Δt = {sim.dt * 1000:.1f} ms）· "
                   f"失效单元 {int(broken.sum())}/{len(broken)} · 红＝受拉")
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
        self.load_pos = None
        self.running = False
        self.broken = False
        self.break_load = 0.0
        self.runs = []
        self.sweep_mode = None
        self.sim = None
        self.pos = None
        self.broken_mask = None
        self.elem_stress = np.zeros(1)
        self._pending_rebuild = False
        self.substeps = 3

        self._rebuild_sim()
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

        self.rebuild_timer = QTimer(self)
        self.rebuild_timer.setSingleShot(True)
        self.rebuild_timer.timeout.connect(self.apply_pending)

    # ---------- 模型 ----------
    def _rebuild_sim(self):
        E, sig, nu, rho = MATERIALS[self.material]
        self.sim = ImplicitBeam(self.beam_length, self.beam_height, E, sig, rho,
                                support=self.support, nu=nu)
        self.static = StaticBeam(self.beam_length, self.beam_height, E, sig,
                                 support=self.support, nu=nu)
        self.capacity_ref = self.static.capacity()
        self.load = 0.0
        self.broken = False
        self.pos = self.sim.positions()
        self.broken_mask = self.sim.broken_mask()
        self.elem_stress = self.sim.stresses()
        if hasattr(self, "lbl_cap"):
            self.lbl_cap.setText(f"理论极限 {self.capacity_ref / 1000:.1f} kN")

    def reset_beam(self):
        self._rebuild_sim()
        self.running = False
        self.run_btn.setText("开始加载")
        self.canvas.update()
        self.update_stats()
        self.lbl_status.setText("已复位")

    def tick(self):
        if not self.running or self.sim is None:
            return
        if self.broken:
            # 已经断了：不再加重物，只按断裂时的载荷继续算塌落
            new_break = self.sim.step(self.break_load, substeps=self.substeps)
        else:
            frac = self.load / max(1.0, self.capacity_ref)
            step = max(6.0, self.capacity_ref * 0.008) * (1.0 if frac < 0.8 else 0.35)
            self.load += step
            new_break = self.sim.step(self.load, substeps=self.substeps)
        self.pos = self.sim.positions()
        self.elem_stress = self.sim.stresses()
        self.broken_mask = self.sim.broken_mask()
        if new_break and not self.broken:
            self.broken = True
            self.break_load = self.load
            self.runs.append({"L": self.beam_length, "h": self.beam_height, "P": self.load,
                              "x": self.beam_height * 100 if self.sweep_mode == "height"
                              else self.beam_length})
            self.chart.update()
            self.lbl_status.setText(
                f"开始断裂！载荷 {self.load / 1000:.2f} kN（{self.load / G:.0f} kg）")
        self.canvas.update()
        self.update_stats()

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
        self._rebuild_sim()
        self.running = False
        self.run_btn.setText("开始加载")
        self.canvas.update()
        self.update_stats()

    def toggle_run(self):
        if self.sim is None:
            return
        if self.broken:
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
        if self.sim is not None and self.pos is not None:
            defl = float(np.abs(self.pos[:, 1] - self.sim.nodes[:, 1]).max())
            self.lbl_defl.setText(f"挠度 {defl * 1000:.2f} mm")
            smax = float(np.abs(self.elem_stress).max())
            self.lbl_stress.setText(
                f"主应力 {smax / 1e6:.1f} MPa（{smax / self.sim.sigma_ult * 100:.0f}%）")

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
