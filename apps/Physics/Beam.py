from custom_import import *
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPolygonF, QLinearGradient, QBrush

# ==================== 材料与梁 ====================
G = 9.81
B_WIDTH = 0.06          # 梁的宽度（出平面方向），固定

MATERIALS = {
    "松木": (10e9, 40e6),
    "橡木": (12e9, 55e6),
    "钢材": (200e9, 250e6),
    "混凝土": (30e9, 3e6),
}


class BeamSolver:
    """欧拉-伯努利梁的一维有限元（每节点挠度+转角两个自由度）。"""

    def __init__(self, length, height, E, sigma_ult, support="simple", n_elem=64):
        self.L = length
        self.h = height
        self.E = E
        self.sigma_ult = sigma_ult
        self.support = support
        self.n = n_elem
        self.le = length / n_elem

    def solve(self, load, load_pos=None):
        """返回 (挠度数组, 最大弯曲应力, 最大挠度, 断裂位置x)。"""
        n = self.n
        le = self.le
        EI = self.E * B_WIDTH * self.h ** 3 / 12.0
        ndof = 2 * (n + 1)
        if load_pos is None:
            load_pos = self.L / 2.0

        ke = EI / le ** 3 * np.array([
            [12.0, 6 * le, -12.0, 6 * le],
            [6 * le, 4 * le * le, -6 * le, 2 * le * le],
            [-12.0, -6 * le, 12.0, -6 * le],
            [6 * le, 2 * le * le, -6 * le, 4 * le * le],
        ])
        K = np.zeros((ndof, ndof))
        for e in range(n):
            idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            K[np.ix_(idx, idx)] += ke

        F = np.zeros(ndof)
        j = int(np.clip(round(load_pos / le), 0, n))
        F[2 * j] -= load

        fixed = [0, 2 * n]
        if self.support == "fixed":
            fixed += [1, 2 * n + 1]
        # 注意：不要用 np.setdiff1d —— 它在这套 numpy 里会触发 numpy.ma 的惰性
        # 导入并递归，直接栈溢出把进程打崩
        mask = np.ones(ndof, dtype=bool)
        mask[fixed] = False
        free = np.nonzero(mask)[0]
        u = np.zeros(ndof)
        u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

        w = u[0::2]
        # 每单元两端弯矩 → 弯曲应力 σ = 6M / (b h²)
        i2 = 6.0 * EI / (B_WIDTH * self.h ** 2)
        m1 = i2 / le ** 2 * (u[0:-3:2] * 6.0 + u[1:-2:2] * 4.0 * le
                             - u[2:-1:2] * 6.0 + u[3::2] * 2.0 * le)
        m2 = i2 / le ** 2 * (-u[0:-3:2] * 6.0 - u[1:-2:2] * 2.0 * le
                             + u[2:-1:2] * 6.0 - u[3::2] * 4.0 * le)
        stress = np.maximum(np.abs(m1), np.abs(m2))
        k = int(np.argmax(stress))
        return w, float(stress[k]), float(np.abs(w).max()), (k + 0.5) * le, stress

    def capacity(self, load_pos=None):
        """二分求断裂载荷。"""
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
        self.setMinimumSize(560, 340)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(248, 248, 246))
        app = self.app
        sim = app.solver

        w, hh = self.width(), self.height()
        margin_x = 80
        span = max(1e-6, sim.L)
        scale_x = (w - 2 * margin_x) / span
        base_y = hh * 0.62
        h_px = max(6.0, min(90.0, sim.h * scale_x * 2.2))
        exag = app.exag

        w_def = app.deflection          # 当前挠度分布
        xs = np.linspace(0, span, len(w_def))

        # 支座
        sup_color = QColor(90, 90, 96)
        for x in (0.0, span):
            sx = margin_x + x * scale_x
            path = QPolygonF([
                QPointF(sx, base_y + h_px / 2),
                QPointF(sx - 14, base_y + h_px / 2 + 24),
                QPointF(sx + 14, base_y + h_px / 2 + 24),
            ])
            p.setBrush(sup_color)
            p.setPen(Qt.NoPen)
            p.drawPolygon(path)
            p.setPen(QPen(QColor(150, 150, 155), 2))
            p.drawLine(QPointF(sx - 22, base_y + h_px / 2 + 24),
                       QPointF(sx + 22, base_y + h_px / 2 + 24))
            # 钉子
            p.setBrush(QColor(70, 70, 75))
            p.drawEllipse(QPointF(sx, base_y), 4, 4)

        if app.broken:
            self._draw_broken(p, margin_x, scale_x, base_y, h_px)
        else:
            self._draw_beam(p, margin_x, scale_x, base_y, h_px, xs, w_def, exag)

        # 重物
        if not app.broken and app.load > 0:
            pos = app.load_pos if app.load_pos is not None else span / 2.0
            sx = margin_x + pos * scale_x
            bw = 30 + 40 * min(1.0, app.load / max(1.0, app.break_load or 1e9))
            bh = 18 + 26 * min(1.0, app.load / max(1.0, app.break_load or 1e9))
            top = base_y - h_px / 2 + w_def[len(w_def) // 2] * exag - bh
            p.setBrush(QColor(70, 78, 92))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(sx - bw / 2, top, bw, bh), 4, 4)
            p.setPen(QColor(255, 255, 255))
            f = p.font()
            f.setPointSize(8)
            p.setFont(f)
            p.drawText(QRectF(sx - bw / 2, top, bw, bh), Qt.AlignCenter,
                       f"{app.load / G:.0f}kg")

        # 应力颜色条
        self._draw_legend(p, w, hh)
        p.end()

    def _draw_beam(self, p, mx, sx, base_y, h_px, xs, w_def, exag):
        sim = self.app.solver
        n = len(w_def) - 1
        # 逐段绘制：上缘受压（蓝）、下缘受拉（红），颜色强度 ∝ 应力
        for i in range(n):
            x0 = mx + xs[i] * sx
            x1 = mx + xs[i + 1] * sx
            y0 = base_y + w_def[i] * exag
            y1 = base_y + w_def[i + 1] * exag
            frac = float(self.app.stress_profile[i]) if i < len(self.app.stress_profile) else 0.0
            t = float(np.clip(frac, 0.0, 1.0))
            top = QColor(int(120 + 60 * t), int(150 + 40 * t), 235)
            bot = QColor(235, int(150 - 80 * t), int(120 - 60 * t))
            grad = QLinearGradient(x0, y0 - h_px / 2, x0, y0 + h_px / 2)
            grad.setColorAt(0.0, top)
            grad.setColorAt(1.0, bot)
            p.setBrush(QBrush(grad))
            p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygonF([
                QPointF(x0, y0 - h_px / 2), QPointF(x1, y1 - h_px / 2),
                QPointF(x1, y1 + h_px / 2), QPointF(x0, y0 + h_px / 2),
            ]))
        p.setPen(QPen(QColor(40, 40, 45), 1.2))
        poly_top = [QPointF(mx + xs[i] * sx, base_y + w_def[i] * exag - h_px / 2) for i in range(n + 1)]
        poly_bot = [QPointF(mx + xs[i] * sx, base_y + w_def[i] * exag + h_px / 2) for i in range(n)]
        p.drawPolyline(QPolygonF(poly_top))
        p.drawPolyline(QPolygonF(poly_bot))

    def _draw_broken(self, p, mx, sx, base_y, h_px):
        app = self.app
        span = app.solver.L
        mid = mx + span * sx / 2
        drop = min(120.0, h_px * 1.5 + 40)
        p.setPen(Qt.NoPen)
        for sign in (-1, 1):
            x0 = mx if sign < 0 else mid
            x1 = mid if sign < 0 else mx + span * sx
            # 绕着外侧支座转，断口自然垂下来
            pivot = mx if sign < 0 else mx + span * sx
            ang = -14.0 if sign < 0 else 14.0
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
        p.drawText(QRectF(16, h - 34, 420, 20), Qt.AlignLeft | Qt.AlignVCenter,
                   "上缘受压（蓝）  下缘受拉（红）  颜色越深越接近材料强度")


# ==================== 数据图 ====================
class SweepChart(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumHeight(180)

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
        p.setPen(QColor(70, 70, 76))
        f = p.font()
        f.setPointSize(9)
        p.setFont(f)
        title = {"height": "承重 vs 梁高（理论 ∝ h²）",
                 "length": "承重 vs 梁长（理论 ∝ 1/L）"}.get(mode, "还没有数据 —— 运行一次扫描")
        p.drawText(QRectF(left, 2, w - left - right, 16), Qt.AlignCenter, title)

        if not runs:
            p.end()
            return

        xs = np.array([r["x"] for r in runs])
        ys = np.array([r["P"] for r in runs])
        x_lo, x_hi = xs.min(), xs.max()
        y_lo, y_hi = 0.0, max(ys) * 1.15
        pw, ph = w - left - right, h - top - bottom

        def to_px(x, y):
            fx = (x - x_lo) / max(1e-9, x_hi - x_lo)
            fy = (y - y_lo) / max(1e-9, y_hi - y_lo)
            return left + fx * pw, top + (1 - fy) * ph

        # 理论曲线
        if mode in ("height", "length"):
            L = runs[0]["L"]
            hh = runs[0]["h"]
            E, sig = MATERIALS[self.app.material]
            n_pts = 60
            curve_x = np.linspace(x_lo, x_hi, n_pts)
            curve_y = []
            for v in curve_x:
                if mode == "height":
                    sim = BeamSolver(L, v / 100.0, E, sig, self.app.support)
                else:
                    sim = BeamSolver(v, hh, E, sig, self.app.support)
                curve_y.append(sim.capacity())
            pts = [QPointF(*to_px(x, y)) for x, y in zip(curve_x, curve_y)]
            p.setPen(QPen(QColor(120, 150, 210), 2))
            p.drawPolyline(QPolygonF(pts))

        # 实测点
        p.setPen(QPen(QColor(220, 90, 60), 1))
        p.setBrush(QColor(220, 90, 60))
        for x, y in zip(xs, ys):
            px, py = to_px(x, y)
            p.drawEllipse(QPointF(px, py), 3.5, 3.5)

        p.setPen(QColor(90, 90, 96))
        xlabel = "梁高 h (cm)" if mode == "height" else ("梁长 L (m)" if mode == "length" else "")
        p.drawText(QRectF(left, h - 22, pw, 16), Qt.AlignCenter, xlabel)
        p.drawText(QRectF(4, top, 56, 16), Qt.AlignRight | Qt.AlignVCenter,
                   f"{y_hi / 1000:.0f}kN")
        p.drawText(QRectF(4, top + ph - 12, 56, 16), Qt.AlignRight | Qt.AlignVCenter, "0")
        p.end()


# ==================== 主窗口 ====================
class BeamApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("横梁承重 · 断裂实验")
        self.setGeometry(120, 70, 1180, 800)

        self.material = "松木"
        self.support = "simple"
        self.beam_length = 1.6          # m
        self.beam_height = 0.12         # m
        self.load = 0.0            # N
        self.load_pos = None       # 默认跨中
        self.broken = False
        self.auto_loading = False
        self.break_load = None
        self.runs = []
        self.sweep_mode = None
        self._rebuild()
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(40)

    # ---------- 模型 ----------
    def _rebuild(self):
        E, sig = MATERIALS[self.material]
        self.solver = BeamSolver(self.beam_length, self.beam_height, E, sig, self.support)
        self.deflection = np.zeros(self.solver.n + 1)
        self.stress_profile = np.zeros(self.solver.n)
        self.max_stress = 0.0
        self.max_defl = 0.0
        self.break_x = self.beam_length / 2

    def update_solution(self):
        if self.broken:
            return
        load = self.load
        w, smax, dmax, bx, stress = self.solver.solve(load, self.load_pos)
        self.deflection = w
        self.max_stress = smax
        self.max_defl = dmax
        self.break_x = bx
        # 每单元应力占比（用于沿梁长上色）
        self.stress_profile = np.clip(stress / self.solver.sigma_ult, 0.0, 1.0)
        if smax >= self.solver.sigma_ult:
            self.fail()

    def fail(self):
        self.broken = True
        self.auto_loading = False
        self.break_load = self.load
        self.runs.append({
            "L": self.beam_length, "h": self.beam_height, "P": self.load,
            "x": self.beam_height * 100 if self.sweep_mode == "height" else self.beam_length,
            "mode": self.sweep_mode, "material": self.material, "support": self.support,
        })
        if hasattr(self, "chart"):
            self.chart.update()
        self.update_stats()

    def tick(self):
        if self.auto_loading and not self.broken:
            # 逐步加载：越接近断裂走得越慢，看得清
            step = max(20.0, self.load * 0.06)
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

        self.lbl_status = QLabel("调好梁的尺寸，点「自动加载」看它什么时候断")
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
        if hasattr(self, "lbl_cap"):
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
        E, sig = MATERIALS[self.material]
        if mode == "height":
            values = np.linspace(0.02, 0.30, 9)
            for v in values:
                sim = BeamSolver(self.beam_length, v, E, sig, self.support)
                cap = sim.capacity()
                self.runs.append({"L": self.beam_length, "h": v, "P": cap, "x": v * 100, "mode": mode})
        else:
            values = np.linspace(0.5, 3.0, 9)
            for v in values:
                sim = BeamSolver(v, self.beam_height, E, sig, self.support)
                cap = sim.capacity()
                self.runs.append({"L": v, "h": self.beam_height, "P": cap, "x": v, "mode": mode})
        self.chart.update()
        n = len(self.runs)
        self.lbl_status.setText(f"已扫描 {n} 组：" + ("梁越高越抗弯（∝ h²）" if mode == "height" else "梁越长越弱（∝ 1/L）"))

    def clear_runs(self):
        self.runs = []
        self.chart.update()
        self.lbl_status.setText("数据已清空")

    def update_stats(self):
        self.lbl_load.setText(f"载荷 {self.load / G:.0f} kg")
        if self.max_stress > 0:
            ratio = self.max_stress / self.solver.sigma_ult
            self.lbl_stress.setText(f"应力 {self.max_stress / 1e6:.1f} MPa（{ratio * 100:.0f}%）")
            self.lbl_defl.setText(f"挠度 {self.max_defl * 1000:.2f} mm")
        else:
            self.lbl_stress.setText("应力 –")
            self.lbl_defl.setText("挠度 –")
        if self.broken:
            self.lbl_status.setText(
                f"断裂！极限载荷 {self.break_load / 1000:.2f} kN（{self.break_load / G:.0f} kg）")

    # 挠度显示放大系数
    @property
    def exag(self):
        d = self.max_defl
        if d <= 1e-12:
            return 0.0
        target = self.beam_height * 2.2          # 目标显示偏移（米）
        return min(4000.0, target / d)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BeamApp()
    window.show()
    sys.exit(app.exec())
