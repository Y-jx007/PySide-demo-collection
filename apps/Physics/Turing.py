from custom_import import *
from reaction_diffusion import *

ti.init(arch=ti.gpu, default_fp=ti.f32)

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
        gray = np.clip((phi_np + 1.0) * 0.5 * 255, 0, 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

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
        def make_preset_callback(sliders):
            def apply_preset(idx):
                name = presets[idx]
                if name in preset_vals:
                    fv, kv = preset_vals[name]
                    sim.f, sim.k = fv, kv
                    sliders['f'].setValue(int(fv * 10000))
                    sliders['k'].setValue(int(kv * 10000))
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
        def make_preset_callback(sliders):
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
        def make_preset_callback(sliders):
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

    layout.addWidget(create_equation_group("方程", eq_html))
    param_group, sliders = create_param_group(sim, param_defs)
    layout.addWidget(param_group)

    preset_callback = make_preset_callback(sliders)
    init_group, _, _ = create_init_group(sim, viewer,
                                         list(reset_mapping.keys()),
                                         presets,
                                         lambda idx: preset_callback(idx),
                                         reset_mapping)
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

        self.top_selector = QWidget()
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("模拟:"))

        self.sim_combo = QComboBox()
        self.sim_combo.addItems(["Gray‑Scott", "Oregonator", "Cahn‑Hilliard"])
        self.sim_combo.setMinimumWidth(120)
        self.sim_combo.currentTextChanged.connect(self._on_sim_type_changed)
        top_layout.addWidget(self.sim_combo)

        top_layout.addWidget(QLabel("尺寸:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["128x128", "256x256", "512x512", "1024x1024"])
        self.size_combo.setCurrentText("512x512")
        self.size_combo.currentTextChanged.connect(self._on_size_changed)
        top_layout.addWidget(self.size_combo)

        self.top_selector.setLayout(top_layout)

        self.custom_container = QWidget()
        self.custom_container_layout = QVBoxLayout()
        self.custom_container.setLayout(self.custom_container_layout)

        self.viewer.param_panel_layout.insertWidget(0, self.top_selector)
        self.viewer.param_panel_layout.insertWidget(1, self.custom_container)

        self._rebuild_custom_panel()
        self.setGeometry(100, 100, 960, 650)

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
        ti.reset()
        event.accept()


if __name__ == "__main__":
    setup_gl_format()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())