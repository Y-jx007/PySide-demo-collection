from common.qt_base import (QComboBox, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget, run_app)
from common.taichi_base import (ti)
from common.reaction_diffusion import (
    SimulationBase, SimulationViewer,
    render_reaction_image, render_gray_image,
    create_equation_group, create_param_group,
    create_init_group, create_control_group,
)

ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=False)

# ---------- f–k 地图（200×200 参数，可缩放，点击选点） ----------
from pathlib import Path
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QImageReader, QPainter, QPen
from PySide6.QtWidgets import QGroupBox

from common.paths import PRESETS_DIR

_PRESET_DIR = PRESETS_DIR / "turing"

F_MIN, F_MAX = 0.010, 0.060
K_MIN, K_MAX = 0.040, 0.070
MAP_N = 200          # 地图是 200x200 个格子


class FkMapView(QWidget):
    """可缩放/平移的 f–k 参数地图，单击取点。"""

    picked = Signal(float, float)

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        QImageReader.setAllocationLimit(1024)
        self._img = QImage(str(image_path)).convertToFormat(QImage.Format_Grayscale8)
        self._ok = not self._img.isNull()
        self.setFixedSize(288, 288)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self._min_scale = self.width() / max(self._img.width(), 1)
        self._max_scale = max(self._min_scale, 1.2)
        self._scale = self._min_scale
        self._center = QPointF(self._img.width() / 2.0, self._img.height() / 2.0)
        self._f, self._k = 0.040, 0.060
        self._press = None
        self._moved = False

    # --- 坐标换算 ---
    def _src_rect(self):
        w = self.width() / self._scale
        h = self.height() / self._scale
        cx = min(max(self._center.x(), w / 2), max(self._img.width() - w / 2, w / 2))
        cy = min(max(self._center.y(), h / 2), max(self._img.height() - h / 2, h / 2))
        return QRectF(cx - w / 2, cy - h / 2, w, h)

    def _to_image(self, pos):
        r = self._src_rect()
        return (r.left() + pos.x() / self._scale, r.top() + pos.y() / self._scale)

    def set_params(self, f, k):
        self._f, self._k = f, k
        self.update()

    def reset_view(self):
        self._scale = self._min_scale
        self._center = QPointF(self._img.width() / 2.0, self._img.height() / 2.0)
        self.update()

    # --- 绘制 ---
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(12, 12, 16))
        if not self._ok:
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, self._scale < 1.0)
        p.drawImage(QRectF(self.rect()), self._img, self._src_rect())

        # 当前参数点
        w, h = self._src_rect().width(), self._src_rect().height()
        x = (self._k - K_MIN) / (K_MAX - K_MIN) * self._img.width()
        y = (F_MAX - self._f) / (F_MAX - F_MIN) * self._img.height()
        r = self._src_rect()
        sx = (x - r.left()) * self.width() / w
        sy = (y - r.top()) * self.height() / h
        if 0 <= sx <= self.width() and 0 <= sy <= self.height():
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(QColor(255, 70, 70), 2))
            p.drawEllipse(QPointF(sx, sy), 5, 5)

    # --- 交互 ---
    def wheelEvent(self, event):
        if not self._ok:
            return
        factor = 1.22 ** (event.angleDelta().y() / 120.0)
        new_scale = min(max(self._scale * factor, self._min_scale), self._max_scale)
        if new_scale == self._scale:
            return
        pos = event.position()
        ix, iy = self._to_image(pos)
        self._scale = new_scale
        self._center = QPointF(
            ix - (pos.x() - self.width() / 2) / self._scale,
            iy - (pos.y() - self.height() / 2) / self._scale,
        )
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press = event.position()
            self._moved = False

    def mouseMoveEvent(self, event):
        if self._press is None:
            return
        delta = event.position() - self._press
        if abs(delta.x()) + abs(delta.y()) > 2:
            self._moved = True
        if self._moved:
            self._center = QPointF(
                self._center.x() - delta.x() / self._scale,
                self._center.y() - delta.y() / self._scale,
            )
            self._press = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self._press is None:
            return
        was_drag = self._moved
        self._press = None
        if was_drag or not self._ok:
            return
        ix, iy = self._to_image(event.position())
        col = int(ix / self._img.width() * MAP_N)
        row = int(iy / self._img.height() * MAP_N)
        col = min(max(col, 0), MAP_N - 1)
        row = min(max(row, 0), MAP_N - 1)
        f = F_MAX - (row + 0.5) / MAP_N * (F_MAX - F_MIN)
        k = K_MIN + (col + 0.5) / MAP_N * (K_MAX - K_MIN)
        self.picked.emit(f, k)

    def mouseDoubleClickEvent(self, event):
        self.reset_view()


def create_fk_map(sim, viewer, sliders, get_pattern=None):
    image = _PRESET_DIR / "fk_map.png"
    if not image.exists():
        return QWidget()

    group = QGroupBox("f–k 地图")
    layout = QVBoxLayout()
    layout.setContentsMargins(4, 4, 4, 4)
    widget = FkMapView(image)

    def current_pattern():
        return get_pattern() if get_pattern else 'random'

    def on_pick(f, k):
        sim.f = f
        sim.k = k
        sliders['f'].setValue(int(round(f * 10000)))
        sliders['k'].setValue(int(round(k * 10000)))
        sim.reset(current_pattern())
        viewer.gl_widget.update()
        widget.set_params(f, k)

    widget.picked.connect(on_pick)
    widget.set_params(sim.f, sim.k)
    layout.addWidget(widget, 0, Qt.AlignHCenter)
    hint = QLabel("滚轮缩放 · 拖动平移 · 双击复位　|　横轴 k　纵轴 f")
    hint.setStyleSheet("color: #8a8a94;")
    layout.addWidget(hint, 0, Qt.AlignHCenter)
    group.setLayout(layout)
    return group

# ---------- 通用五点拉普拉斯 ----------
@ti.func
def compute_laplacian(field, i, j):
    return field[i+1, j] + field[i-1, j] + field[i, j+1] + field[i, j-1] - 4.0 * field[i, j]

# ---------- Gray‑Scott 图灵斑模拟 ----------
@ti.data_oriented
class TuringSimulation(SimulationBase):
    def __init__(self, size=512):
        self.size = size
        self.U = ti.field(ti.f32, shape=(size, size))
        self.V = ti.field(ti.f32, shape=(size, size))
        self.Du = 0.16
        self.Dv = 0.08
        self.f = 0.04
        self.k = 0.06
        self.dt = 0.5
        self._init_patterns = {
            'center': self._init_center,
            'random': self._init_random,
            'edges': self._init_edges,
            'spots': self._init_spots
        }
        self.reset('random')

    @ti.kernel
    def update_step(self, Du: ti.f32, Dv: ti.f32, f: ti.f32, k: ti.f32, dt: ti.f32):
        for i, j in ti.ndrange((1, self.size-1), (1, self.size-1)):
            lap_u = compute_laplacian(self.U, i, j)
            lap_v = compute_laplacian(self.V, i, j)
            reaction = self.U[i, j] * self.V[i, j] * self.V[i, j]
            U_new = self.U[i, j] + dt * (Du * lap_u - reaction + f * (1.0 - self.U[i, j]))
            V_new = self.V[i, j] + dt * (Dv * lap_v + reaction - (f + k) * self.V[i, j])
            self.U[i, j] = ti.math.clamp(U_new, 0.0, 1.0)
            self.V[i, j] = ti.math.clamp(V_new, 0.0, 1.0)

    def step(self):
        self.update_step(self.Du, self.Dv, self.f, self.k, self.dt)

    def get_image(self, mode=None):
        u_np, v_np = self.U.to_numpy(), self.V.to_numpy()
        return render_reaction_image(u_np, v_np)

    def reset(self, pattern='random'):
        self.U.fill(1.0)
        self.V.fill(0.0)
        self._init_patterns.get(pattern, self._init_random)()

    @ti.kernel
    def _init_center(self):
        center = self.size // 2
        s = self.size // 20
        for i, j in ti.ndrange(self.size, self.size):
            if ti.abs(i - center) < s and ti.abs(j - center) < s:
                self.V[i, j] = 1.0

    @ti.kernel
    def _init_random(self):
        border = self.size // 20
        density = 0.001 * (512 / self.size)
        for i, j in ti.ndrange(self.size, self.size):
            if (border <= i < self.size - border and
                border <= j < self.size - border and ti.random() < density):
                s = max(2, self.size // 256)
                for di, dj in ti.ndrange((-s, s+1), (-s, s+1)):
                    ni, nj = i+di, j+dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        self.V[ni, nj] = 1.0

    @ti.kernel
    def _init_edges(self):
        b = max(3, self.size // 170)
        for i, j in ti.ndrange(self.size, self.size):
            if i < b or i >= self.size - b or j < b or j >= self.size - b:
                self.V[i, j] = 1.0

    @ti.kernel
    def _init_spots(self):
        spacing = max(15, self.size // 17)
        for i, j in ti.ndrange(self.size, self.size):
            if i % spacing == spacing//2 and j % spacing == spacing//2:
                s = max(2, self.size // 256)
                for di, dj in ti.ndrange((-s, s+1), (-s, s+1)):
                    ni, nj = i+di, j+dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        self.V[ni, nj] = 1.0


# ---------- Oregonator (B‑Z) 化学波模拟 ----------
@ti.data_oriented
class BZSimulation(SimulationBase):
    def __init__(self, size=512):
        self.size = size
        self.u = ti.field(ti.f32, shape=(size, size))
        self.v = ti.field(ti.f32, shape=(size, size))
        self.epsilon = 0.05
        self.f_param = 1.6
        self.q_param = 0.002
        self.Du = 1.0
        self.Dv = 0.6
        self.dt = 0.01
        self._init_patterns = {
            'random': self.init_random,
            'center': self.init_center,
            'edges': self.init_edges,
            'spots': self.init_spots,
            'spiral': self.init_spiral
        }
        self.reset('spiral')

    @ti.kernel
    def step_kernel(self, epsilon: ti.f32, f: ti.f32, q: ti.f32,
                    Du: ti.f32, Dv: ti.f32, dt: ti.f32):
        for i, j in ti.ndrange((1, self.size - 1), (1, self.size - 1)):
            u_center = self.u[i, j]
            v_center = self.v[i, j]
            lap_u = compute_laplacian(self.u, i, j)
            lap_v = compute_laplacian(self.v, i, j)
            reaction = (u_center - u_center*u_center -
                        f * v_center * (u_center - q) / (u_center + q + 1e-10))
            du = (1.0 / epsilon) * reaction + Du * lap_u
            dv = u_center - v_center + Dv * lap_v
            self.u[i, j] = ti.math.clamp(u_center + du * dt, 0.0, 1.0)
            self.v[i, j] = ti.math.clamp(v_center + dv * dt, 0.0, 1.0)

    def step(self):
        self.step_kernel(self.epsilon, self.f_param, self.q_param,
                         self.Du, self.Dv, self.dt)

    def get_image(self, mode=None):
        u_np, v_np = self.u.to_numpy(), self.v.to_numpy()
        return render_reaction_image(u_np, v_np)

    def reset(self, pattern='spiral'):
        self.u.fill(0.0)
        self.v.fill(0.0)
        self._init_patterns.get(pattern, self.init_spiral)()

    @ti.kernel
    def init_center(self):
        center = self.size // 2
        r = self.size // 15
        for i, j in ti.ndrange(self.size, self.size):
            if (i - center)**2 + (j - center)**2 < r * r:
                self.v[i, j] = 0.8

    @ti.kernel
    def init_random(self):
        border = self.size // 15
        for i, j in ti.ndrange(self.size, self.size):
            if (border <= i < self.size - border and
                border <= j < self.size - border):
                self.v[i, j] = ti.random() * 0.5

    @ti.kernel
    def init_edges(self):
        b = max(3, self.size // 170)
        for i, j in ti.ndrange(self.size, self.size):
            if i < b or i >= self.size - b or j < b or j >= self.size - b:
                self.v[i, j] = 0.8

    @ti.kernel
    def init_spots(self):
        spacing = max(15, self.size // 17)
        for i, j in ti.ndrange(self.size, self.size):
            if i % spacing == spacing//2 and j % spacing == spacing//2:
                s = max(2, self.size // 256)
                for di, dj in ti.ndrange((-s, s+1), (-s, s+1)):
                    ni, nj = i+di, j+dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        self.v[ni, nj] = 0.8

    @ti.kernel
    def init_spiral(self):
        center = self.size // 2
        for i, j in ti.ndrange(self.size, self.size):
            dx = ti.cast(i, ti.f32) - center
            dy = ti.cast(j, ti.f32) - center
            angle = ti.atan2(dy, dx) + 3.14159
            self.u[i, j] = 0.5 + 0.5 * ti.sin(angle)
            self.v[i, j] = 0.1


# ---------- Cahn‑Hilliard 油水分离模拟 ----------
@ti.data_oriented
class CahnHilliardSimulation(SimulationBase):
    def __init__(self, size=512):
        self.size = size
        self.phi = ti.field(ti.f32, shape=(size, size))
        self.mu = ti.field(ti.f32, shape=(size, size))
        self.gamma = 1.0
        self.M = 1.0
        self.dt = 0.01
        self._init_patterns = {
            'random': self._init_random,
            'spots': self._init_spots,
            'center': self._init_center,
            'layers': self._init_layers
        }
        self.reset('random')

    @ti.kernel
    def compute_mu(self, gamma: ti.f32):
        for i, j in ti.ndrange((1, self.size-1), (1, self.size-1)):
            phi_c = self.phi[i, j]
            lap_phi = compute_laplacian(self.phi, i, j)
            self.mu[i, j] = phi_c*phi_c*phi_c - phi_c - gamma * lap_phi

    @ti.kernel
    def update_phi(self, M: ti.f32, dt: ti.f32):
        for i, j in ti.ndrange((2, self.size-2), (2, self.size-2)):
            lap_mu = compute_laplacian(self.mu, i, j)
            new_phi = self.phi[i, j] + dt * M * lap_mu
            self.phi[i, j] = ti.math.clamp(new_phi, -1.0, 1.0)

    def step(self):
        self.compute_mu(self.gamma)
        self.update_phi(self.M, self.dt)

    def get_image(self, mode=None):
        phi_np = self.phi.to_numpy()
        return render_gray_image((phi_np + 1.0) * 0.5)

    def reset(self, pattern='random'):
        self.phi.fill(0.0)
        self._init_patterns.get(pattern, self._init_random)()

    @ti.kernel
    def _init_random(self):
        for i, j in ti.ndrange(self.size, self.size):
            self.phi[i, j] = (ti.random() - 0.5) * 0.05

    @ti.kernel
    def _init_spots(self):
        num_spots = max(5, self.size // 50)
        for _ in range(num_spots):
            cx = ti.random(int) * (self.size - 20) + 10
            cy = ti.random(int) * (self.size - 20) + 10
            r = ti.random() * 10 + 2
            for i, j in ti.ndrange(self.size, self.size):
                if (i-cx)**2 + (j-cy)**2 < r*r:
                    self.phi[i, j] = 1.0

    @ti.kernel
    def _init_center(self):
        center = self.size // 2
        r = self.size // 10
        for i, j in ti.ndrange(self.size, self.size):
            if (i-center)**2 + (j-center)**2 < r*r:
                self.phi[i, j] = 1.0

    @ti.kernel
    def _init_layers(self):
        w = self.size // 10
        for i, j in ti.ndrange(self.size, self.size):
            if (j // w) % 2 == 0:
                self.phi[i, j] = 0.8
            else:
                self.phi[i, j] = -0.8


# ---------- 面板构建 ----------
def build_panel(sim, viewer, sim_type):
    panel = QWidget()
    layout = QVBoxLayout()

    if sim_type == 'turing':
        eq_html = ("<b>Gray‑Scott 模型</b><br>"
                   "∂u/∂t = D<sub>u</sub>∇²u − uv² + f(1−u)<br>"
                   "∂v/∂t = D<sub>v</sub>∇²v + uv² − (f + k)v")
        param_defs = [
            ("f:", 0.01, 0.08, sim.f, 10000, 'f'),
            ("k:", 0.04, 0.08, sim.k, 10000, 'k'),
            ("D<sub>u</sub>:", 0.1, 0.3, sim.Du, 1000, 'Du'),
            ("D<sub>v</sub>:", 0.04, 0.12, sim.Dv, 1000, 'Dv'),
            ("步长 dt:", 0.05, 2.0, sim.dt, 100, 'dt'),
        ]
        reset_mapping = {"随机": "random", "中心": "center", "边缘": "edges", "点阵": "spots"}
        presets = ["迷宫", "条纹", "斑点", "蜂巢", "云雾"]
        preset_vals = {
            "迷宫": (0.04, 0.06), "条纹": (0.055, 0.065),
            "斑点": (0.04, 0.065), "蜂巢": (0.03, 0.055),
            "云雾": (0.016, 0.045)
        }
        def make_preset_callback(sliders, get_pattern=None):
            def apply_preset(idx):
                name = presets[idx]
                if name in preset_vals:
                    fv, kv = preset_vals[name]
                    sim.f, sim.k = fv, kv
                    sliders['f'].setValue(int(fv * 10000))
                    sliders['k'].setValue(int(kv * 10000))
                    sim.reset(get_pattern() if get_pattern else 'random')
                    viewer.gl_widget.update()
            return apply_preset

    elif sim_type == 'oregonator':
        eq_html = ("<b>Oregonator 模型</b><br>"
                   "ε ∂u/∂t = u − u² − f·v·(u−q)/(u+q) + D<sub>u</sub>∇²u<br>"
                   "∂v/∂t = u − v + D<sub>v</sub>∇²v")
        param_defs = [
            ("ε:", 0.01, 0.2, sim.epsilon, 1000, 'epsilon'),
            ("f:", 0.5, 3.0, sim.f_param, 1000, 'f_param'),
            ("q:", 0.001, 0.02, sim.q_param, 100000, 'q_param'),
            ("D<sub>u</sub>:", 0.5, 2.0, sim.Du, 100, 'Du'),
            ("D<sub>v</sub>:", 0.2, 1.5, sim.Dv, 100, 'Dv'),
            ("步长 dt:", 0.005, 0.1, sim.dt, 1000, 'dt'),
        ]
        reset_mapping = {"随机": "random", "中心": "center", "边缘": "edges", "点阵": "spots"}
        presets = ["螺旋", "靶波", "混沌"]
        preset_vals = {
            "螺旋":  {"ε":0.05, "f":1.6, "q":0.002, "Du":1.0, "Dv":0.6, "init":"spiral"},
            "靶波":  {"ε":0.05, "f":1.6, "q":0.002, "Du":1.0, "Dv":0.6, "init":"center"},
            "混沌":  {"ε":0.02, "f":2.8, "q":0.005, "Du":1.2, "Dv":0.5, "init":"random"},
        }
        def make_preset_callback(sliders, get_pattern=None):
            def apply_preset(idx):
                p = preset_vals[presets[idx]]
                sim.epsilon = p["ε"]
                sim.f_param = p["f"]
                sim.q_param = p["q"]
                sim.Du = p["Du"]
                sim.Dv = p["Dv"]
                sliders['epsilon'].setValue(int(p["ε"] * 1000))
                sliders['f_param'].setValue(int(p["f"] * 1000))
                sliders['q_param'].setValue(int(p["q"] * 100000))
                sliders['Du'].setValue(int(p["Du"] * 100))
                sliders['Dv'].setValue(int(p["Dv"] * 100))
                sim.u.fill(0.0)
                sim.v.fill(0.0)
                {'spiral': sim.init_spiral, 'center': sim.init_center, 'random': sim.init_random}[p["init"]]()
                viewer.gl_widget.update()
            return apply_preset

    else:   # cahn_hilliard
        eq_html = ("<b>Cahn‑Hilliard 方程</b><br>"
                   "∂φ/∂t = M ∇² (φ³ − φ − γ ∇²φ)")
        param_defs = [
            ("γ (界面能):", 0.1, 3.0, sim.gamma, 1000, 'gamma'),
            ("M (迁移率):", 0.1, 5.0, sim.M, 1000, 'M'),
            ("步长 dt:", 0.001, 0.1, sim.dt, 10000, 'dt'),
        ]
        reset_mapping = {"随机": "random", "点阵": "spots", "中心": "center", "层状": "layers"}
        presets = ["细密相畴", "粗大相畴", "快速分离"]
        preset_vals = {
            "细密相畴": {"gamma": 1.5, "M": 1.0, "dt": 0.01},
            "粗大相畴": {"gamma": 0.3, "M": 1.0, "dt": 0.02},
            "快速分离": {"gamma": 1.0, "M": 3.0, "dt": 0.005},
        }
        def make_preset_callback(sliders, get_pattern=None):
            def apply_preset(idx):
                name = presets[idx]
                p = preset_vals[name]
                sim.gamma = p["gamma"]
                sim.M = p["M"]
                sim.dt = p["dt"]
                sliders['gamma'].setValue(int(p["gamma"] * 1000))
                sliders['M'].setValue(int(p["M"] * 1000))
                sliders['dt'].setValue(int(p["dt"] * 10000))
                viewer.gl_widget.update()
            return apply_preset

    layout.addWidget(create_equation_group("方程", eq_html, compact=True))
    param_group, sliders = create_param_group(sim, param_defs, compact=True)
    layout.addWidget(param_group)

    pattern_state = {'name': '随机'}
    def get_pattern():
        return reset_mapping.get(pattern_state['name'], 'random')

    preset_callback = make_preset_callback(sliders, get_pattern)
    init_group, pattern_combo, _ = create_init_group(sim, viewer,
                                         list(reset_mapping.keys()),
                                         presets,
                                         lambda idx: preset_callback(idx),
                                         reset_mapping)
    pattern_state['name'] = pattern_combo.currentText()
    pattern_combo.currentTextChanged.connect(lambda t: pattern_state.__setitem__('name', t))
    if sim_type == 'turing':
        layout.addWidget(create_fk_map(sim, viewer, sliders, get_pattern))
    layout.addWidget(init_group)
    layout.addWidget(create_control_group(viewer))
    layout.addStretch()
    panel.setLayout(layout)
    return panel


# ---------- 主窗口 ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("反应扩散模拟系统")
        self.current_sim_type = "turing"
        self.current_size = 512

        self.sim = TuringSimulation(self.current_size)
        self.viewer = SimulationViewer(self.sim)
        self.setCentralWidget(self.viewer)

        self.viewer.set_display_modes(['混合'])
        self.viewer.param_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.top_selector = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(5)
        top_layout.addWidget(QLabel("模拟:"))

        self.sim_combo = QComboBox()
        self.sim_combo.addItems(["Gray‑Scott", "Oregonator", "Cahn‑Hilliard"])
        self._fit_combo(self.sim_combo)
        self.sim_combo.currentTextChanged.connect(self._on_sim_type_changed)
        top_layout.addWidget(self.sim_combo)

        top_layout.addWidget(QLabel("尺寸:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["128x128", "256x256", "512x512", "1024x1024"])
        self.size_combo.setCurrentText("512x512")
        self._fit_combo(self.size_combo)
        self.size_combo.currentTextChanged.connect(self._on_size_changed)
        top_layout.addWidget(self.size_combo)
        top_layout.addStretch(1)

        self.top_selector.setLayout(top_layout)

        self.custom_container = QWidget()
        self.custom_container_layout = QVBoxLayout()
        self.custom_container.setLayout(self.custom_container_layout)

        self.viewer.param_panel_layout.insertWidget(0, self.top_selector)
        self.viewer.param_panel_layout.insertWidget(1, self.custom_container)

        self._rebuild_custom_panel()
        self.setMinimumSize(900, 620)
        self.setGeometry(100, 100, 1180, 780)

    @staticmethod
    def _fit_combo(combo):
        """按最长选项文字 + 箭头留白定宽，避免不同字体/DPI 下截断。"""
        fm = combo.fontMetrics()
        widest = max((fm.horizontalAdvance(combo.itemText(i)) for i in range(combo.count())), default=0)
        combo.setFixedWidth(widest + 42)

    def _on_sim_type_changed(self, text):
        mapping = {
            "Gray‑Scott": "turing",
            "Oregonator": "oregonator",
            "Cahn‑Hilliard": "cahn_hilliard"
        }
        self.current_sim_type = mapping[text]
        self._rebuild_simulation(keep_params=False)

    def _on_size_changed(self, text):
        sizes = {"128x128":128, "256x256":256, "512x512":512, "1024x1024":1024}
        new_size = sizes[text]
        if new_size != self.current_size:
            self.current_size = new_size
            self._rebuild_simulation(keep_params=True)

    def _rebuild_simulation(self, keep_params=False):
        old_sim = self.viewer.sim
        was_playing = self.viewer.is_playing
        self.viewer.timer.stop()

        if self.current_sim_type == "turing":
            new_sim = TuringSimulation(self.current_size)
            if keep_params and isinstance(old_sim, TuringSimulation):
                new_sim.f, new_sim.k = old_sim.f, old_sim.k
                new_sim.Du, new_sim.Dv = old_sim.Du, old_sim.Dv
                new_sim.dt = old_sim.dt
        elif self.current_sim_type == "oregonator":
            new_sim = BZSimulation(self.current_size)
            if keep_params and isinstance(old_sim, BZSimulation):
                new_sim.epsilon = old_sim.epsilon
                new_sim.f_param = old_sim.f_param
                new_sim.q_param = old_sim.q_param
                new_sim.Du = old_sim.Du
                new_sim.Dv = old_sim.Dv
                new_sim.dt = old_sim.dt
        else:  # cahn_hilliard
            new_sim = CahnHilliardSimulation(self.current_size)
            if keep_params and isinstance(old_sim, CahnHilliardSimulation):
                new_sim.gamma = old_sim.gamma
                new_sim.M = old_sim.M
                new_sim.dt = old_sim.dt

        self.viewer.sim = new_sim
        self._rebuild_custom_panel()
        if was_playing:
            self.viewer.timer.start()

    def _rebuild_custom_panel(self):
        while self.custom_container_layout.count():
            child = self.custom_container_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
        panel = build_panel(self.viewer.sim, self.viewer, self.current_sim_type)
        self.custom_container_layout.addWidget(panel)

    def closeEvent(self, event):
        self.viewer.timer.stop()
        event.accept()

if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow, gl=True))
