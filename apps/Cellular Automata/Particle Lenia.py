import taichi_forge as ti

from common.qt_base import (LabeledSlider, QColor, QHBoxLayout, QMainWindow, QPainter, QPen, QPointF, QPolygonF,
                            QPushButton, QTimer, QVBoxLayout, QWidget, Qt, run_app)

# ---- Taichi 初始化（无 GPU 时自动回退 CPU）----
try:
    ti.init(arch=ti.gpu, offline_cache=True)
except Exception:
    ti.init(arch=ti.cpu, offline_cache=True)

N = 1500
WORLD_SIZE = 512
DT = 0.2
MAX_SPEED = 15.0

# 粒子状态
pos = ti.Vector.field(2, ti.f32, shape=N)
vel = ti.Vector.field(2, ti.f32, shape=N)

# ---- 可调参数：0 维 field，实时生效 ----
p_mu    = ti.field(ti.f32, shape=())   # 环半径 μ
p_sigma = ti.field(ti.f32, shape=())   # 环宽度 σ
p_force = ti.field(ti.f32, shape=())   # 力强度 k
p_patch = ti.field(ti.f32, shape=())   # 相互作用截断半径
p_damp  = ti.field(ti.f32, shape=())   # 阻尼

# 默认参数
p_mu[None]    = 12.0
p_sigma[None] = 4.0
p_force[None] = 2.0
p_patch[None] = 30.0
p_damp[None]  = 0.08


@ti.func
def pair_force(r):
    """环状势阱力。r<μ 为正(排斥)，r>μ 为负(吸引)，r=μ 为零。"""
    x = (r - p_mu[None]) / p_sigma[None]
    return -p_force[None] * x * ti.exp(-x * x)


@ti.kernel
def step(dt: ti.f32):
    patch   = p_patch[None]
    damping = p_damp[None]
    for i in range(N):
        pi = pos[i]
        f = ti.Vector([0.0, 0.0])
        for j in range(N):
            if i != j:
                diff = pi - pos[j]
                r = diff.norm()
                if r > 1e-4 and r < patch:
                    f += (diff / r) * pair_force(r)
        v = (vel[i] + f * dt) * (1.0 - damping)
        # 限速防止爆炸
        sp = v.norm()
        if sp > MAX_SPEED:
            v *= MAX_SPEED / sp
        # 边界反弹
        p = pi + v * dt
        if p[0] < 0.0:
            p[0] = 0.0; v[0] = -v[0] * 0.5
        elif p[0] > WORLD_SIZE:
            p[0] = WORLD_SIZE; v[0] = -v[0] * 0.5
        if p[1] < 0.0:
            p[1] = 0.0; v[1] = -v[1] * 0.5
        elif p[1] > WORLD_SIZE:
            p[1] = WORLD_SIZE; v[1] = -v[1] * 0.5
        pos[i] = p
        vel[i] = v


@ti.kernel
def init_particles():
    for i in range(N):
        cx = WORLD_SIZE * 0.5
        cy = WORLD_SIZE * 0.5
        a  = ti.random() * 6.2831853
        rr = ti.sqrt(ti.random()) * WORLD_SIZE * 0.35
        pos[i] = ti.Vector([cx + rr * ti.cos(a), cy + rr * ti.sin(a)])
        vel[i] = ti.Vector([0.0, 0.0])


# ---------- UI ----------
class ParticleWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(WORLD_SIZE, WORLD_SIZE)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 20))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        positions = pos.to_numpy()
        pts = [QPointF(float(p[0]), float(p[1])) for p in positions]
        poly = QPolygonF(pts)

        pen = QPen(QColor(120, 200, 255, 200))
        pen.setWidth(3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawPoints(poly)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Particle Lenia — PySide6 + Taichi")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.view = ParticleWidget()
        layout.addWidget(self.view)

        panel = QWidget()
        panel.setFixedWidth(240)
        panel_layout = QVBoxLayout(panel)

        # 参数滑块 —— 直接写入 Taichi field，实时生效
        self.add_slider(panel_layout, "环半径 μ",   p_mu,    4.0,  30.0, 0.5)
        self.add_slider(panel_layout, "环宽度 σ",   p_sigma, 1.0,  15.0, 0.25)
        self.add_slider(panel_layout, "力强度 k",   p_force, 0.0,  8.0,  0.1)
        self.add_slider(panel_layout, "截断半径",   p_patch, 10.0, 80.0, 1.0)
        self.add_slider(panel_layout, "阻尼",       p_damp,  0.01, 0.3,  0.005)

        reset_btn = QPushButton("重置粒子")
        reset_btn.clicked.connect(init_particles)
        panel_layout.addWidget(reset_btn)
        panel_layout.addStretch()

        layout.addWidget(panel)

        # 启动
        init_particles()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def add_slider(self, layout, label, field, lo, hi, step):
        """滑块值 = 实际值 / step；写入 Taichi field 立即生效。"""
        initial = float(field[None])
        widget = LabeledSlider(label, int(lo / step), int(hi / step),
                               int(round(initial / step)),
                               formatter=lambda v: f"{v * step:.2f}")
        widget.valueChanged.connect(lambda v: field.__setitem__(None, v * step))
        layout.addWidget(widget)
        return widget

    def tick(self):
        step(DT)
        self.view.update()


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
