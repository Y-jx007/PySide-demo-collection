from custom_import import *
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPolygonF, QLinearGradient, QBrush

# ==================== 材料 ====================
G = 9.81
B_WIDTH = 0.06          # 梁宽（出平面方向）

MATERIALS = {
    "松木": (10e9, 40e6, 0.30),
    "橡木": (12e9, 55e6, 0.30),
    "钢材": (200e9, 250e6, 0.28),
    "混凝土": (30e9, 3e6, 0.20),
}


# ==================== 二维平面应力有限元 ====================
class BeamSolver:
    """平面应力 Q4 单元，梁用矩形网格离散。

    每个单元 4 节点 × 2 自由度，2×2 高斯积分组装刚度矩阵；约束自由度后
    求一次逆（几何不变时复用），之后每改一次载荷只做一次矩阵乘。
    """

    def __init__(self, length, height, E, sigma_ult, support="simple", nx=36, ny=6, nu=0.3):
        self.L = length
        self.h = height
        self.E = E
        self.sigma_ult = sigma_ult
        self.nu = nu
        self.support = support
        self.nx = nx
        self.ny = ny
        self._build()
        self._assemble()
        self._apply_bc()
        self._factor()

    # ---------- 网格 ----------
    def _build(self):
        nx, ny = self.nx, self.ny
        xs = np.linspace(0.0, self.L, nx + 1)
        ys = np.linspace(-self.h / 2.0, self.h / 2.0, ny + 1)
        self.xs, self.ys = xs, ys
        self.nodes = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
        self.n_nodes = self.nodes.shape[0]
        conn = []
        for j in range(ny):
            for i in range(nx):
                n0 = j * (nx + 1) + i
                conn.append([n0, n0 + 1, n0 + (nx + 1) + 1, n0 + (nx + 1)])
        self.conn = np.array(conn, dtype=np.int32)

    def _material_D(self):
        E, nu = self.E, self.nu
        return E / (1.0 - nu * nu) * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])

    # ---------- 组装 ----------
    def _B_matrix(self, xi, eta):
        """矩形单元在自然坐标 (xi, eta) 处的应变矩阵；所有单元形状相同，只需算一次。"""
        dx, dy = self.L / self.nx, self.h / self.ny
        dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        dNx = dN_dxi * (2.0 / dx)
        dNy = dN_deta * (2.0 / dy)
        B = np.zeros((3, 8))
        B[0, 0::2] = dNx
        B[1, 1::2] = dNy
        B[2, 0::2] = dNy
        B[2, 1::2] = dNx
        return B

    def _assemble(self):
        D = self._material_D()
        g = 1.0 / np.sqrt(3.0)
        dx, dy = self.L / self.nx, self.h / self.ny
        ke = np.zeros((8, 8))
        for xi in (-g, g):
            for eta in (-g, g):
                B = self._B_matrix(xi, eta)
                ke += B.T @ D @ B * (dx * dy / 4.0) * B_WIDTH
        self.ke = ke

        K = np.zeros((2 * self.n_nodes, 2 * self.n_nodes))
        for conn in self.conn:
            idx = np.empty(8, dtype=np.int32)
            idx[0::2] = 2 * conn
            idx[1::2] = 2 * conn + 1
            K[np.ix_(idx, idx)] += ke
        self.K = K
        self.ndof = 2 * self.n_nodes

        # 四角的应力矩阵也只需要算一次
        corners = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
        self.corner_S = [D @ self._B_matrix(xi, eta) for xi, eta in corners]

    def _apply_bc(self):
        nx = self.nx
        left, right = 0, nx
        fixed = [2 * left, 2 * left + 1]        # 左端下角：铰接
        if self.support == fixed:
            fixed += [2 * right, 2 * right + 1]
        else:
            fixed += [2 * right + 1]            # 简支：右端仅约束竖向
        self.fixed = np.array(sorted(set(fixed)), dtype=np.int32)

    def _factor(self):
        mask = np.ones(self.ndof, dtype=bool)
        mask[self.fixed] = False
        self.free = np.nonzero(mask)[0]
        Kff = self.K[np.ix_(self.free, self.free)]
        # 对称正定，用 Cholesky 分解；每个载荷步只做两次三角回代
        # （numpy 的 inv 在这台机器上异常慢，scipy 的 potrf 只要 20ms）
        try:
            from scipy.linalg import cho_factor, cho_solve
            self._cho = cho_factor(Kff, lower=False, check_finite=False)
            self._solve_free = lambda b: cho_solve(self._cho, b, check_finite=False)
        except Exception:
            self._cho = None
            self.Kinv = np.linalg.inv(Kff)
            self._solve_free = lambda b: self.Kinv @ b
        # 单元自由度索引，便于一次性取 u_e
        self.elem_dof = np.empty((len(self.conn), 8), dtype=np.int32)
        self.elem_dof[:, 0::2] = 2 * self.conn
        self.elem_dof[:, 1::2] = 2 * self.conn + 1

    # ---------- 求解 ----------
    def _stress(self, u):
        ue = u[self.elem_dof].T                     # (8, n_elem)
        best = None
        for S in self.corner_S:
            s = S @ ue                              # (3, n_elem)
            val = 0.5 * (s[0] + s[1]) + np.hypot(0.5 * (s[0] - s[1]), s[2])
            best = val if best is None else np.maximum(best, val)
        return best

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
        u[self.free] = self._solve_free(F[self.free])
        sig1 = self._stress(u)
        k = int(np.argmax(sig1))
        dmax = float(np.abs(u[1::2]).max())
        cx = float(np.mean(self.nodes[self.conn[k], 0]))
        return u, float(sig1[k]), dmax, cx, sig1

    def capacity(self, load_pos=None):
        if load_pos is None:
            load_pos = self.L / 2.0
        lo, hi = 0.0, 1.0
        while self.solve(hi, load_pos)[1] < self.sigma_ult and hi < 1e12:
            hi *= 2.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.solve(mid, load_pos)[1] >= self.sigma_ult:
                hi = mid
            else:
                lo = mid
        return hi


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
        sim = app.solver

        w, hh = self.width(), self.height()
        margin_x = 90
        span = max(1e-6, sim.L)
        scale_x = (w - 2 * margin_x) / span
        base_y = hh * 0.52
        exag = app.exag

        sup_y = base_y + sim.h * scale_x * 0.5 + 6
        for x in (0.0, span):
            sx = margin_x + x * scale_x
            p.setBrush(QColor(90, 90, 96))
            p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygonF([QPointF(sx, sup_y),
                                     QPointF(sx - 15, sup_y + 22),
                                     QPointF(sx + 15, sup_y + 22)]))
            p.setPen(QPen(QColor(150, 150, 155), 2))
            p.drawLine(QPointF(sx - 22, sup_y + 22), QPointF(sx + 22, sup_y + 22))

        if app.broken:
            self._draw_broken(p, margin_x, scale_x, base_y, sim)
        else:
            self._draw_mesh(p, margin_x, scale_x, base_y, sim, exag)

        if not app.broken and app.load > 0:
            pos = app.load_pos if app.load_pos is not None else span / 2.0
            sx = margin_x + pos * scale_x
            top_y = self._top_y(sim, pos, margin_x, scale_x, base_y, exag)
            frac = min(1.0, app.load / max(1.0, app.break_load or app.load * 1.4))
            bw = 34 + 44 * frac
            bh = 20 + 24 * frac
            p.setBrush(QColor(70, 78, 92))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(sx - bw / 2, top_y - bh - 2, bw, bh), 4, 4)
            p.setPen(QColor(255, 255, 255))
            f = p.font()
            f.setPointSize(8)
            p.setFont(f)
            p.drawText(QRectF(sx - bw / 2, top_y - bh - 2, bw, bh), Qt.AlignCenter,
                       f"{app.load / G:.0f}kg")

        self._draw_legend(p, w, hh)
        p.end()

    def _node_pixels(self, sim, mx, sx, base_y, exag):
        u = self.app.u
        nodes = sim.nodes
        if u is None:
            px = mx + nodes[:, 0] * sx
            py = base_y - nodes[:, 1] * sx
        else:
            # 只有位移放大，梁本身的厚度和位置不放大
            px = mx + nodes[:, 0] * sx + u[0::2] * exag * sx
            py = base_y - nodes[:, 1] * sx - u[1::2] * exag * sx
        return px, py

    def _top_y(self, sim, x_pos, mx, sx, base_y, exag):
        px, py = self._node_pixels(sim, mx, sx, base_y, exag)
        i = int(np.clip(round(x_pos / sim.L * sim.nx), 0, sim.nx))
        return float(py[sim.ny * (sim.nx + 1) + i])

    def _draw_mesh(self, p, mx, sx, base_y, sim, exag):
        px, py = self._node_pixels(sim, mx, sx, base_y, exag)
        limit = sim.sigma_ult
        for e, conn in enumerate(sim.conn):
            sig = self.app.sigma_elem[e]
            t = float(np.clip(abs(sig) / limit, 0.0, 1.0))
            if sig >= 0:      # 拉应力：越接近强度越红
                col = QColor(int(150 + 95 * t), int(170 - 110 * t), int(205 - 160 * t))
            else:             # 压应力：蓝
                col = QColor(int(120 - 25 * t), int(150 + 35 * t), int(210))
            p.setBrush(col)
            p.setPen(QPen(QColor(110, 110, 120, 90), 0.5))
            p.drawPolygon(QPolygonF([QPointF(px[i], py[i]) for i in conn]))

        p.setPen(QPen(QColor(40, 40, 45), 1.5))
        p.setBrush(Qt.NoBrush)
        top = [sim.ny * (sim.nx + 1) + i for i in range(sim.nx + 1)]
        bottom = list(range(sim.nx + 1))
        p.drawPolyline(QPolygonF([QPointF(px[i], py[i]) for i in bottom]))
        p.drawPolyline(QPolygonF([QPointF(px[i], py[i]) for i in top]))

    def _draw_broken(self, p, mx, sx, base_y, sim):
        span = sim.L
        h_px = max(6.0, sim.h * sx)
        mid = mx + span * sx / 2
        p.setPen(Qt.NoPen)
        for sign in (-1, 1):
            x0 = mx if sign < 0 else mid
            x1 = mid if sign < 0 else mx + span * sx
            pivot = mx if sign < 0 else mx + span * sx
            ang = 13.0 if sign < 0 else -13.0   # 断口向下垂
            p.save()
            p.translate(pivot, base_y)
            p.rotate(ang)
            p.translate(-pivot, -base_y)
            grad = QLinearGradient(0, base_y - h_px / 2, 0, base_y + h_px / 2)
            grad.setColorAt(0.0, QColor(170, 190, 235))
            grad.setColorAt(1.0, QColor(220, 160, 140))
            p.setBrush(QBrush(grad))
            p.drawRect(QRectF(x0, base_y - h_px / 2, x1 - x0, h_px))
            p.restore()
        p.setPen(QPen(QColor(60, 60, 66), 2))
        p.drawLine(QPointF(mid - 6, base_y - h_px / 2 - 8), QPointF(mid + 4, base_y + h_px / 2 + 8))
        f = p.font()
        f.setPointSize(13)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(190, 60, 50))
        p.drawText(QRectF(mid - 120, base_y - h_px / 2 - 60, 240, 30), Qt.AlignCenter, "断裂")

    def _draw_legend(self, p, w, h):
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        p.setPen(QColor(90, 90, 96))
        p.drawText(QRectF(16, h - 28, 560, 20), Qt.AlignLeft | Qt.AlignVCenter,
                   "单元按主应力上色：偏红＝受拉（危险），偏蓝＝受压；形变已放大显示")


# ==================== 数据图 ====================
class SweepChart(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumHeight(170)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(252, 252, 250))
        runs = self.app.runs
        w, h = self.width(), self.height()
        left, right, top, bottom = 62, 16, 18, 34
        p.setPen(QColor(120, 120, 126))
        p.drawRect(QRectF(left, top, w - left - right, h - top - bottom))

        mode = self.app.sweep_mode
        f = p.font()
        f.setPointSize(9)
        p.setFont(f)
        p.setPen(QColor(70, 70, 76))
        title = {"height": "承重 vs 梁高（理论 ∝ h²）",
                 "length": "承重 vs 梁长（理论 ∝ 1/L）"}.get(mode, "还没有数据 —— 运行一次扫描")
        p.drawText(QRectF(left, 2, w - left - right, 16), Qt.AlignCenter, title)

        if not runs:
            p.end()
            return

        xs = np.array([r["x"] for r in runs])
        ys = np.array([r["P"] for r in runs])
        x_lo, x_hi = float(xs.min()), float(xs.max())
        y_lo, y_hi = 0.0, float(max(ys)) * 1.15
        pw, ph = w - left - right, h - top - bottom

        def to_px(x, y):
            fx = (x - x_lo) / max(1e-9, x_hi - x_lo)
            fy = (y - y_lo) / max(1e-9, y_hi - y_lo)
            return left + fx * pw, top + (1 - fy) * ph

        if mode in ("height", "length"):
            L = runs[0]["L"]
            hh = runs[0]["h"]
            E, sig, nu = MATERIALS[self.app.material]
            curve_x = np.linspace(x_lo, x_hi, 32)
            curve_y = []
            for v in curve_x:
                if mode == "height":
                    sim = BeamSolver(L, v / 100.0, E, sig, self.app.support, nu=nu)
                else:
                    sim = BeamSolver(v, hh, E, sig, self.app.support, nu=nu)
                curve_y.append(sim.capacity())
            pts = [QPointF(*to_px(x, y)) for x, y in zip(curve_x, curve_y)]
            p.setPen(QPen(QColor(120, 150, 210), 2))
            p.drawPolyline(QPolygonF(pts))

        p.setPen(QPen(QColor(220, 90, 60), 1))
        p.setBrush(QColor(220, 90, 60))
        for x, y in zip(xs, ys):
            px, py = to_px(x, y)
            p.drawEllipse(QPointF(px, py), 3.5, 3.5)

        p.setPen(QColor(90, 90, 96))
        xlabel = "梁高 h (cm)" if mode == "height" else ("梁长 L (m)" if mode == "length" else "")
        p.drawText(QRectF(left, h - 22, pw, 16), Qt.AlignCenter, xlabel)
        p.drawText(QRectF(4, top, 56, 16), Qt.AlignRight | Qt.AlignVCenter, f"{y_hi / 1000:.0f}kN")
        p.drawText(QRectF(4, top + ph - 12, 56, 16), Qt.AlignRight | Qt.AlignVCenter, "0")
        p.end()


# ==================== 主窗口 ====================
class BeamApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("横梁承重 · 二维有限元")
        self.setGeometry(110, 60, 1180, 820)

        self.material = "松木"
        self.support = "simple"
        self.beam_length = 1.6
        self.beam_height = 0.12
        self.load = 0.0
        self.load_pos = None
        self.broken = False
        self.auto_loading = False
        self.break_load = None
        self.runs = []
        self.sweep_mode = None
        self.u = None
        self.sigma_elem = None
        self.max_stress = 0.0
        self.max_defl = 0.0
        self._rebuild()
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(40)

    # ---------- 模型 ----------
    def _rebuild(self):
        E, sig, nu = MATERIALS[self.material]
        self.solver = BeamSolver(self.beam_length, self.beam_height, E, sig, self.support, nu=nu)
        self.u = None
        self.sigma_elem = np.zeros(len(self.solver.conn))
        self.max_stress = 0.0
        self.max_defl = 0.0

    def update_solution(self):
        if self.broken:
            return
        u, smax, dmax, cx, sig = self.solver.solve(self.load, self.load_pos)
        self.u = u
        self.max_stress = smax
        self.max_defl = dmax
        self.sigma_elem = sig
        if smax >= self.solver.sigma_ult and self.load > 0:
            self.fail()

    def fail(self):
        self.broken = True
        self.auto_loading = False
        self.break_load = self.load
        self.runs.append({
            "L": self.beam_length, "h": self.beam_height, "P": self.load,
            "x": self.beam_height * 100 if self.sweep_mode == "height" else self.beam_length,
        })
        self.chart.update()
        self.update_stats()

    def tick(self):
        if self.auto_loading and not self.broken:
            step = max(15.0, self.load * 0.05)
            self.load += step
            self.update_solution()
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

        self.lbl_status = QLabel("调好尺寸，点「自动加载」看梁什么时候断")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_load = QLabel("载荷 0 kg")
        self.lbl_stress = QLabel("应力 –")
        self.lbl_defl = QLabel("挠度 –")
        for wdg in (self.lbl_load, self.lbl_stress, self.lbl_defl):
            self.statusBar().addPermanentWidget(wdg)

    def _build_beam_group(self, parent):
        group = QGroupBox("梁（二维网格）")
        form = QFormLayout(group)
        form.setSpacing(4)
        form.setContentsMargins(6, 6, 6, 6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.mat_combo = QComboBox()
        self.mat_combo.addItems(list(MATERIALS.keys()))
        self.mat_combo.currentTextChanged.connect(self.on_material)
        form.addRow("材料", self.mat_combo)

        self.sup_combo = QComboBox()
        self.sup_combo.addItems(["简支（钉子可转）", "固支（完全固定）"])
        self.sup_combo.currentIndexChanged.connect(self.on_support)
        form.addRow("支承", self.sup_combo)

        self.len_slider, self.len_label = self._slider(form, "长度 L", 30, 300, int(self.beam_length * 100),
                                                       lambda v: v / 100.0, " m")
        self.hgt_slider, self.hgt_label = self._slider(form, "高度 h", 2, 30, int(self.beam_height * 100),
                                                       lambda v: v / 100.0, " m")
        parent.addWidget(group)

    def _build_load_group(self, parent):
        group = QGroupBox("加载")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)
        self.load_btn = QPushButton("自动加载")
        self.load_btn.clicked.connect(self.toggle_load)
        vbox.addWidget(self.load_btn)
        row = QHBoxLayout()
        row.setSpacing(4)
        btn_reset = QPushButton("复位")
        btn_reset.clicked.connect(self.reset_beam)
        row.addWidget(btn_reset)
        self.btn_cap = QPushButton("测极限")
        self.btn_cap.clicked.connect(self.show_capacity)
        row.addWidget(self.btn_cap)
        vbox.addLayout(row)
        self.lbl_cap = QLabel("极限载荷 –")
        vbox.addWidget(self.lbl_cap)
        parent.addWidget(group)

    def _build_sweep_group(self, parent):
        group = QGroupBox("实验：承重与尺寸的关系")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)
        btn_h = QPushButton("扫描梁高（固定长度）")
        btn_h.clicked.connect(lambda: self.sweep("height"))
        vbox.addWidget(btn_h)
        btn_l = QPushButton("扫描梁长（固定高度）")
        btn_l.clicked.connect(lambda: self.sweep("length"))
        vbox.addWidget(btn_l)
        btn_clear = QPushButton("清空数据")
        btn_clear.clicked.connect(self.clear_runs)
        vbox.addWidget(btn_clear)
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
            self.on_geometry()

        slider.valueChanged.connect(on_change)
        row.addWidget(slider, 1)
        row.addWidget(label)
        form.addRow(text, row)
        return slider, label

    # ---------- 交互 ----------
    def on_material(self, name):
        self.material = name
        self._rebuild()
        self.canvas.update()
        self.update_stats()

    def on_support(self, index):
        self.support = "simple" if index == 0 else "fixed"
        self._rebuild()
        self.canvas.update()
        self.update_stats()

    def on_geometry(self):
        self.beam_length = self.len_slider.value() / 100.0
        self.beam_height = self.hgt_slider.value() / 100.0
        self._rebuild()
        self.lbl_cap.setText("极限载荷 –")
        self.canvas.update()
        self.update_stats()

    def toggle_load(self):
        if self.broken:
            self.reset_beam()
            return
        self.auto_loading = not self.auto_loading
        self.load_btn.setText("停止加载" if self.auto_loading else "自动加载")
        self.lbl_status.setText("加载中…" if self.auto_loading else "已暂停")

    def reset_beam(self):
        self.broken = False
        self.load = 0.0
        self.break_load = None
        self.auto_loading = False
        self.load_btn.setText("自动加载")
        self._rebuild()
        self.canvas.update()
        self.update_stats()
        self.lbl_status.setText("已复位")

    def show_capacity(self):
        cap = self.solver.capacity(self.load_pos)
        self.lbl_cap.setText(f"极限载荷 {cap / 1000:.1f} kN（{cap / G:.0f} kg）")
        self.lbl_status.setText(f"理论上这根梁能撑 {cap / G:.0f} kg")

    def sweep(self, mode):
        self.sweep_mode = mode
        self.runs = []
        self.auto_loading = False
        E, sig, nu = MATERIALS[self.material]
        if mode == "height":
            for v in np.linspace(0.02, 0.30, 9):
                sim = BeamSolver(self.beam_length, v, E, sig, self.support, nu=nu)
                self.runs.append({"L": self.beam_length, "h": v, "P": sim.capacity(), "x": v * 100})
        else:
            for v in np.linspace(0.5, 3.0, 9):
                sim = BeamSolver(v, self.beam_height, E, sig, self.support, nu=nu)
                self.runs.append({"L": v, "h": self.beam_height, "P": sim.capacity(), "x": v})
        self.chart.update()
        self.lbl_status.setText(f"已扫描 {len(self.runs)} 组：" +
                                ("梁越高越抗弯（∝ h²）" if mode == "height" else "梁越长越弱（∝ 1/L）"))

    def clear_runs(self):
        self.runs = []
        self.chart.update()
        self.lbl_status.setText("数据已清空")

    def update_stats(self):
        self.lbl_load.setText(f"载荷 {self.load / G:.0f} kg")
        if self.max_stress > 0:
            ratio = self.max_stress / self.solver.sigma_ult
            self.lbl_stress.setText(f"主应力 {self.max_stress / 1e6:.1f} MPa（{ratio * 100:.0f}%）")
            self.lbl_defl.setText(f"挠度 {self.max_defl * 1000:.2f} mm")
        else:
            self.lbl_stress.setText("应力 –")
            self.lbl_defl.setText("挠度 –")
        if self.broken:
            self.lbl_status.setText(
                f"断裂！极限载荷 {self.break_load / 1000:.2f} kN（{self.break_load / G:.0f} kg）")

    @property
    def exag(self):
        d = self.max_defl
        if d <= 1e-12 or self.beam_length <= 0:
            return 1.0
        return float(np.clip(self.beam_length * 0.12 / d, 1.0, 4000.0))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BeamApp()
    window.show()
    sys.exit(app.exec())
