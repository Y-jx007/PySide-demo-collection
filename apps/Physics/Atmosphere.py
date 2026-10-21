from custom_import import *
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainterPath, QPolygonF

# ==================== 对流求解器 ====================
N = 200                      # 网格边长
ITERS = 20                   # 压力泊松迭代次数（每步一次 kernel 启动）


class Atmosphere:
    """二维不可压 Boussinesq 流体（瑞利-贝纳德对流）。

        ∂u/∂t + (u·∇)u = -∇p + ν∇²u + α(T-T0)·ŷ
        ∇·u = 0
        ∂T/∂t + (u·∇)T = κ∇²T

    底部加热、顶部冷却。投影步里解出的压力 p 就是气压，直接拿来画等压线。
    """

    def __init__(self, n=N):
        self.n = n
        ti.reset()
        ti.init(arch=ti.cpu, default_fp=ti.f32)

        self.u = ti.field(ti.f32, (n, n))
        self.v = ti.field(ti.f32, (n, n))
        self.u0 = ti.field(ti.f32, (n, n))
        self.v0 = ti.field(ti.f32, (n, n))
        self.T = ti.field(ti.f32, (n, n))
        self.T0 = ti.field(ti.f32, (n, n))
        self.p = ti.field(ti.f32, (n, n))
        self.div = ti.field(ti.f32, (n, n))

        # 风粒子
        self.px = ti.field(ti.f32, 1400)
        self.py = ti.field(ti.f32, 1400)
        self.ppx = ti.field(ti.f32, 1400)
        self.ppy = ti.field(ti.f32, 1400)
        self.plife = ti.field(ti.f32, 1400)
        self.n_part = 1400

        # 参数
        self.dt = 0.12
        self.visc = 0.00016
        self.kappa = 0.00018
        self.alpha = 0.9
        self.t_bottom = 1.0
        self.t_top = 0.0

        self._seed()
        self._compile()

    # ---------- 初始场 ----------
    def _seed(self):
        n = self.n
        rng = np.random.default_rng(7)
        T = np.zeros((n, n), np.float32)
        yy = np.linspace(0.0, 1.0, n)[:, None]
        T += (1.0 - yy) * 1.0                       # 底部热、顶部冷
        k = 8          # 必须整除 n
        noise = rng.random((k, k)).astype(np.float32) - 0.5
        big = np.kron(noise, np.ones((n // k, n // k), np.float32))[:n, :n]
        for _ in range(4):
            big = _blur(big)
        T += 0.06 * big
        self.T.from_numpy(T.astype(np.float32))
        self.u.from_numpy(np.zeros((n, n), np.float32))
        self.v.from_numpy(np.zeros((n, n), np.float32))
        rng2 = np.random.default_rng(3)
        self.px.from_numpy(rng2.random(self.n_part).astype(np.float32) * (n - 1))
        self.py.from_numpy(rng2.random(self.n_part).astype(np.float32) * (n - 1))
        self.ppx.from_numpy(self.px.to_numpy())
        self.ppy.from_numpy(self.py.to_numpy())
        self.plife.from_numpy(rng2.random(self.n_part).astype(np.float32))

    def reset(self):
        self._seed()

    # ---------- kernel ----------
    def _compile(self):
        n = self.n
        u, v, u0, v0 = self.u, self.v, self.u0, self.v0
        T, T0, p, div = self.T, self.T0, self.p, self.div
        px, py, ppx, ppy, plife = self.px, self.py, self.ppx, self.ppy, self.plife
        visc, kappa, alpha = self.visc, self.kappa, self.alpha
        dt = self.dt
        n_part = self.n_part
        t_bottom = float(self.t_bottom)
        t_top = float(self.t_top)

        @ti.func
        def sample(f: ti.template(), x, y):
            x = ti.max(0.0, ti.min(x, n - 1.001))
            y = ti.max(0.0, ti.min(y, n - 1.001))
            i = ti.cast(x, ti.i32)
            j = ti.cast(y, ti.i32)
            fx = x - i
            fy = y - j
            return (f[i, j] * (1 - fx) * (1 - fy) + f[i + 1, j] * fx * (1 - fy)
                    + f[i, j + 1] * (1 - fx) * fy + f[i + 1, j + 1] * fx * fy)

        @ti.kernel
        def advect():
            for i, j in u:
                x = i - u[i, j] * dt
                y = j - v[i, j] * dt
                u0[i, j] = sample(u, x, y)
                v0[i, j] = sample(v, x, y)
                T0[i, j] = sample(T, i - u[i, j] * dt, j - v[i, j] * dt)

        @ti.kernel
        def forces():
            for i, j in u:
                if 1 <= i < n - 1 and 1 <= j < n - 1:
                    # 浮力 + 粘性
                    u[i, j] = u0[i, j] + visc * dt * (u0[i + 1, j] + u0[i - 1, j]
                                                      + u0[i, j + 1] + u0[i, j - 1] - 4.0 * u0[i, j])
                    v[i, j] = (v0[i, j]
                               + visc * dt * (v0[i + 1, j] + v0[i - 1, j]
                                              + v0[i, j + 1] + v0[i, j - 1] - 4.0 * v0[i, j])
                               + alpha * dt * (T0[i, j] - 0.5))
                    T[i, j] = T0[i, j] + kappa * dt * (T0[i + 1, j] + T0[i - 1, j]
                                                       + T0[i, j + 1] + T0[i, j - 1] - 4.0 * T0[i, j])
                else:
                    u[i, j] = 0.0
                    v[i, j] = 0.0
                    T[i, j] = T0[i, j]

        @ti.kernel
        def compute_div():
            for i, j in u:
                p[i, j] = 0.0
                if 1 <= i < n - 1 and 1 <= j < n - 1:
                    div[i, j] = 0.5 * (u[i + 1, j] - u[i - 1, j] + v[i, j + 1] - v[i, j - 1])
                else:
                    div[i, j] = 0.0

        @ti.kernel
        def jacobi():
            # 这个版本的 Taichi 不允许在 kernel 里嵌套 field 循环，
            # 所以雅可比迭代由外面的 Python 循环逐步推动
            for i, j in u:
                if 1 <= i < n - 1 and 1 <= j < n - 1:
                    # SOR 超松弛，比纯雅可比收敛快得多
                    p[i, j] = p[i, j] + 1.8 * (
                        0.25 * (p[i + 1, j] + p[i - 1, j] + p[i, j + 1] + p[i, j - 1]
                                - div[i, j]) - p[i, j])

        @ti.kernel
        def subtract_grad():
            for i, j in u:
                if 1 <= i < n - 1 and 1 <= j < n - 1:
                    u[i, j] -= 0.5 * (p[i + 1, j] - p[i - 1, j])
                    v[i, j] -= 0.5 * (p[i, j + 1] - p[i, j - 1])

        @ti.kernel
        def boundary():
            for j in range(n):
                T[0, j] = t_bottom
                T[n - 1, j] = t_top
                u[0, j] = 0.0
                u[n - 1, j] = 0.0
                v[0, j] = 0.0
                v[n - 1, j] = 0.0
            for i in range(n):
                u[i, 0] = 0.0
                u[i, n - 1] = 0.0
                v[i, 0] = 0.0
                v[i, n - 1] = 0.0
                T[i, 0] = t_bottom
                T[i, n - 1] = t_top

        @ti.kernel
        def move_particles(dt_p: ti.f32):
            for k in range(n_part):
                x = px[k]
                y = py[k]
                uu = sample(u, x, y)
                vv = sample(v, x, y)
                ppx[k] = x
                ppy[k] = y
                x += uu * dt_p
                y += vv * dt_p
                plife[k] += 0.02
                if x < 1 or x > n - 2 or y < 1 or y > n - 2 or plife[k] > 1.0:
                    # 重生
                    x = 1.0 + ti.random() * (n - 2)
                    y = 1.0 + ti.random() * (n - 2)
                    ppx[k] = x
                    ppy[k] = y
                    plife[k] = 0.0
                px[k] = x
                py[k] = y

        self._advect = advect
        self._forces = forces
        self._compute_div = compute_div
        self._jacobi = jacobi
        self._subtract_grad = subtract_grad
        self._boundary = boundary
        self._move_particles = move_particles

    # ---------- 推进 ----------
    def step(self, substeps=2):
        for _ in range(substeps):
            self._advect()
            self._forces()
            self._compute_div()
            for _ in range(ITERS):
                self._jacobi()
            self._subtract_grad()
            self._boundary()
        self._move_particles(1.0)

    # ---------- 读回 ----------
    def temperature(self):
        return self.T.to_numpy()

    def pressure(self):
        return self.p.to_numpy()

    def velocity(self):
        return self.u.to_numpy(), self.v.to_numpy()

    def particles(self):
        return (self.px.to_numpy(), self.py.to_numpy(),
                self.ppx.to_numpy(), self.ppy.to_numpy(), self.plife.to_numpy())


def _blur(a):
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] + p[1:-1, :-2] + p[1:-1, 1:-1]
            + p[1:-1, 2:] + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


# ==================== 画布 ====================
class AtmosCanvas(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumSize(560, 560)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(250, 250, 252))
        app = self.app
        sim = app.sim
        if sim is None or app.image is None:
            p.end()
            return

        side = min(self.width(), self.height())
        x0 = (self.width() - side) // 2
        y0 = (self.height() - side) // 2
        rect = QRectF(x0, y0, side, side)
        p.drawImage(rect, app.image)

        if app.show_wind:
            scale = side / sim.n
            px, py, ppx, ppy, plife = app.particles
            # 所有流线攒成一条 path 一次画完，逐条 drawLine 太慢
            path = QPainterPath()
            for k in range(len(px)):
                if float(plife[k]) >= 1.0:
                    continue
                path.moveTo(x0 + ppx[k] * scale, y0 + ppy[k] * scale)
                path.lineTo(x0 + px[k] * scale, y0 + py[k] * scale)
            p.setPen(QPen(QColor(255, 255, 255, 120), 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        p.setPen(QPen(QColor(90, 90, 96, 120), 1))
        p.drawRect(rect)
        p.end()


# ==================== 主窗口 ====================
class AtmosApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("大气对流 · 气流 / 气温 / 气压")
        self.setGeometry(120, 70, 1080, 780)

        self.sim = Atmosphere()
        self.image = None
        self.show_wind = True
        self.show_isobars = True
        self.show_temp = True
        self.particles = self.sim.particles()
        self._view = None

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(40)
        self._render()

    # ---------- 渲染 ----------
    def _temp_lut(self):
        """气温色标：冷（蓝）→ 中性（米白）→ 热（橙红）。"""
        stops = [(0.0, (36, 74, 140)), (0.30, (86, 148, 196)), (0.50, (236, 232, 210)),
                 (0.70, (228, 160, 86)), (1.0, (176, 48, 36))]
        lut = np.zeros((256, 3), np.float32)
        xs = np.linspace(0, 1, 256)
        for k in range(3):
            lut[:, k] = np.interp(xs, [s[0] for s in stops], [s[1][k] for s in stops])
        return lut.astype(np.uint8)

    def _render(self):
        sim = self.sim
        n = sim.n
        T = sim.temperature()
        pr = sim.pressure()

        T_img = np.flipud(T)            # 行 0 是地面（热），翻到图像底部
        pr_img = np.flipud(pr)

        if self.show_temp:
            lut = self._temp_lut()
            idx = np.clip(T_img, 0, 1) * 255.0
            rgb = lut[idx.astype(np.uint8)]
        else:
            rgb = np.full((n, n, 3), 245, np.uint8)

        if self.show_isobars:
            # 等压线：按分位数取等值线，用亚像素窄带画 1px 细线
            ps = _blur(_blur(pr_img))          # 压力场先平滑，等压线才不会呈锯齿
            lo, hi = np.percentile(ps, 2), np.percentile(ps, 98)
            if hi - lo < 1e-9:
                hi = lo + 1e-9
            levels = np.linspace(lo, hi, 9)
            band = (hi - lo) / 110.0
            line = np.zeros((n, n), np.float32)
            for lv in levels:
                line = np.maximum(line, np.clip(1.0 - np.abs(ps - lv) / band, 0.0, 1.0))
            line = _blur(line)
            dark = (line * 0.55)[..., None]
            rgb = (rgb * (1 - dark) + np.array([26, 28, 34], np.float32) * dark).astype(np.uint8)

        self.image = QImage(np.ascontiguousarray(rgb).data, n, n, n * 3,
                            QImage.Format_RGB888).copy()
        self._view = (T, pr)

    def tick(self):
        self.sim.step(substeps=2)
        self.particles = self.sim.particles()
        self._render()
        self.canvas.update()
        self.update_stats()

    # ---------- UI ----------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        panel = QWidget()
        panel.setFixedWidth(280)
        pv = QVBoxLayout(panel)
        pv.setSpacing(8)
        pv.setContentsMargins(0, 0, 0, 0)

        title = QLabel("大气对流")
        tf = title.font()
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        pv.addWidget(title)
        sub = QLabel("底部受热、顶部冷却，热空气上升形成对流\n"
                     "气温＝底色，气流＝白色流线，气压＝等压线")
        sub.setStyleSheet("color: #7a7a80;")
        sf = sub.font()
        sf.setPointSize(8)
        sub.setFont(sf)
        pv.addWidget(sub)

        self.canvas = AtmosCanvas(self)

        group = QGroupBox("参数")
        form = QFormLayout(group)
        form.setSpacing(6)
        form.setContentsMargins(8, 8, 8, 8)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.alpha_s, _ = self._slider(form, "加热强度", 0, 200, 60
                                       , lambda v: setattr(self.sim, "alpha", float(v)))
        self.visc_s, _ = self._slider(form, "粘性", 1, 60, 1,
                                      lambda v: setattr(self.sim, "visc", v * 1e-6))
        self.kappa_s, _ = self._slider(form, "热扩散", 1, 60, 4,
                                       lambda v: setattr(self.sim, "kappa", v * 1e-5))
        pv.addWidget(group)

        group2 = QGroupBox("显示")
        v2 = QVBoxLayout(group2)
        v2.setSpacing(4)
        v2.setContentsMargins(8, 8, 8, 8)
        for text, attr in (("气温底色", "show_temp"), ("气压等值线", "show_isobars"),
                           ("气流流线", "show_wind")):
            cb = QCheckBox(text)
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, a=attr: self._toggle(a, checked))
            v2.addWidget(cb)
        pv.addWidget(group2)

        btn = QPushButton("重新播撒扰动")
        btn.clicked.connect(self.reset)
        pv.addWidget(btn)
        pv.addStretch()

        root.addWidget(panel)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(8)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.canvas, 1)
        root.addWidget(right, 1)

        self.lbl_status = QLabel("瑞利-贝纳德对流：热量从底部向上输运")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_t = QLabel("气温 –")
        self.lbl_p = QLabel("气压 –")
        self.lbl_fps = QLabel("帧率 –")
        for wdg in (self.lbl_t, self.lbl_p, self.lbl_fps):
            wdg.setMinimumWidth(110)
            wdg.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.statusBar().addPermanentWidget(wdg)
        self._frames = 0
        self._t0 = time.perf_counter()

    def _slider(self, form, text, lo, hi, value, setter):
        row = QHBoxLayout()
        row.setSpacing(6)
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(value)
        lbl = QLabel(str(value))
        lbl.setFixedWidth(40)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def on_change(v):
            lbl.setText(str(v))
            setter(v)

        s.valueChanged.connect(on_change)
        setter(value)
        row.addWidget(s, 1)
        row.addWidget(lbl)
        form.addRow(text, row)
        return s, lbl

    def _toggle(self, attr, checked):
        setattr(self, attr, bool(checked))
        self._render()
        self.canvas.update()

    def reset(self):
        self.sim.reset()
        self.particles = self.sim.particles()
        self._render()
        self.canvas.update()
        self.lbl_status.setText("已重新播撒扰动")

    def update_stats(self):
        self._frames += 1
        if self._view is not None:
            T, pr = self._view
            self.lbl_t.setText(f"气温 {T.min():.2f}–{T.max():.2f}")
            self.lbl_p.setText(f"气压 ±{np.abs(pr).max():.1f}")
        now = time.perf_counter()
        if now - self._t0 >= 0.5:
            self.lbl_fps.setText(f"帧率 {self._frames / (now - self._t0):.0f}")
            self._frames = 0
            self._t0 = now

    def closeEvent(self, event):
        self.timer.stop()
        try:
            ti.reset()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AtmosApp()
    window.show()
    sys.exit(app.exec())
