from custom_import import *

ti.init(arch=ti.gpu, default_fp=ti.f32)

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
        self.reset('spiral')

    @ti.kernel
    def step_kernel(self, epsilon: ti.f32, f: ti.f32, q: ti.f32,
                    Du: ti.f32, Dv: ti.f32, dt: ti.f32):
        for i, j in ti.ndrange((1, self.size - 1), (1, self.size - 1)):
            u_center = self.u[i, j]
            v_center = self.v[i, j]
            lap_u = (self.u[i-1, j] + self.u[i+1, j] + self.u[i, j-1] + self.u[i, j+1] - 4.0 * u_center)
            lap_v = (self.v[i-1, j] + self.v[i+1, j] + self.v[i, j-1] + self.v[i, j+1] - 4.0 * v_center)
            reaction = u_center - u_center*u_center - f * v_center * (u_center - q) / (u_center + q + 1e-10)
            du = (1.0 / epsilon) * reaction + Du * lap_u
            dv = u_center - v_center + Dv * lap_v
            self.u[i, j] = max(0.0, min(1.0, u_center + du * dt))
            self.v[i, j] = max(0.0, min(1.0, v_center + dv * dt))

    def step(self):
        self.step_kernel(self.epsilon, self.f_param, self.q_param,
                         self.Du, self.Dv, self.dt)

    def get_image(self, mode='u'):
        if mode == 'u':
            field = self.u.to_numpy()
            img = np.clip(field * 255, 0, 255).astype(np.uint8)
            return np.stack([img, img, img], axis=-1)
        elif mode == 'v':
            field = self.v.to_numpy()
            img = np.clip(field * 255, 0, 255).astype(np.uint8)
            return np.stack([img, img, img], axis=-1)
        elif mode == 'phase':
            u_np = self.u.to_numpy()
            v_np = self.v.to_numpy()
            combined = np.stack([u_np, v_np, np.zeros_like(u_np)], axis=-1)
            return (np.clip(combined * 255, 0, 255)).astype(np.uint8)
        return self.get_image('u')

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
            if border <= i < self.size - border and border <= j < self.size - border:
                self.v[i, j] = ti.random() * 0.5

    @ti.kernel
    def init_spiral(self):
        center = self.size // 2
        for i, j in ti.ndrange(self.size, self.size):
            dx = ti.cast(i, ti.f32) - center
            dy = ti.cast(j, ti.f32) - center
            angle = ti.atan2(dy, dx) + 3.14159
            self.u[i, j] = 0.5 + 0.5 * ti.sin(angle)
            self.v[i, j] = 0.1

    @ti.kernel
    def init_target(self):
        center = self.size // 2
        for i, j in ti.ndrange(self.size, self.size):
            if (i - center)**2 + (j - center)**2 < 4.0:
                self.u[i, j] = 0.9
                self.v[i, j] = 0.1

    def reset(self, pattern='spiral'):
        self.u.fill(0.0)
        self.v.fill(0.0)
        if pattern == 'center':
            self.init_center()
        elif pattern == 'random':
            self.init_random()
        elif pattern == 'target':
            self.init_target()
        else:
            self.init_spiral()

class BZWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BZ 化学波 · 螺旋波/靶波/混沌")
        self.sim = BZSimulation(512)
        self.viewer = SimulationViewer(self.sim)
        self.viewer.set_display_modes(['u', 'v', 'phase'])
        self.setCentralWidget(self.viewer)
        self._add_bz_params()
        self.setGeometry(100, 100, 920, 650)

    def _add_bz_params(self):
        panel = QWidget()
        layout = QVBoxLayout()
        eq_label = QLabel(
            "<b>Oregonator 模型</b><br>"
            "ε ∂u/∂t = u − u² − f·v·(u−q)/(u+q) + D<sub>u</sub>∇²u<br>"
            "∂v/∂t = u − v + D<sub>v</sub>∇²v"
        )
        eq_label.setWordWrap(True)
        eq_label.setStyleSheet("QLabel { background: #f0f0f0; padding: 6px; border-radius: 4px; }")
        layout.addWidget(eq_label)

        param_grid = QGridLayout()
        params = [
            ("ε", 0.01, 0.2, 0.05, 1000),
            ("f", 0.5, 3.0, 1.6, 1000),
            ("q", 0.001, 0.02, 0.002, 100000),
            ("Du", 0.5, 2.0, 1.0, 100),
            ("Dv", 0.2, 1.5, 0.6, 100),
            ("dt", 0.001, 0.05, 0.01, 10000),
        ]
        self.sliders = {}
        for row, (name, vmin, vmax, vdef, scale) in enumerate(params):
            label = QLabel(f"{name}: "); label.setMinimumWidth(40)
            val_label = QLabel(f"{vdef:.4f}"); val_label.setMinimumWidth(50)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(int(vmin*scale), int(vmax*scale))
            slider.setValue(int(vdef*scale))
            def make_cb(n, lbl, scl):
                def cb(val):
                    scaled = val / scl
                    lbl.setText(f"{scaled:.4f}")
                    sim = self.sim
                    if   n == "ε": sim.epsilon = scaled
                    elif n == "f": sim.f_param = scaled
                    elif n == "q": sim.q_param = scaled
                    elif n == "Du": sim.Du = scaled
                    elif n == "Dv": sim.Dv = scaled
                    elif n == "dt": sim.dt = scaled
                return cb
            slider.valueChanged.connect(make_cb(name, val_label, scale))
            self.sliders[name] = (slider, val_label)
            param_grid.addWidget(label, row, 0)
            param_grid.addWidget(slider, row, 1)
            param_grid.addWidget(val_label, row, 2)
        layout.addLayout(param_grid)

        preset_layout = QGridLayout()
        preset_layout.addWidget(QLabel("预设:"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["经典螺旋波", "靶波 (中心激发)", "混沌 (高 f)"])
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_combo, 0, 1, 1, 2)
        preset_layout.addWidget(QLabel("初始图案:"), 1, 0)
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["螺旋位相", "中心斑", "随机扰动"])
        self.pattern_combo.currentIndexChanged.connect(self.change_init_pattern)
        preset_layout.addWidget(self.pattern_combo, 1, 1, 1, 2)
        layout.addLayout(preset_layout)

        panel.setLayout(layout)
        self.viewer.param_panel.insertWidget(0, panel)

        # 重写重置函数以使用当前初始图案
        original_reset = self.viewer.reset_sim
        def custom_reset():
            mapping = {"螺旋位相": "spiral", "中心斑": "center", "随机扰动": "random"}
            self.sim.reset(mapping[self.pattern_combo.currentText()])
            self.viewer.gl_widget.update()
        self.viewer.reset_sim = custom_reset

    def apply_preset(self, idx):
        presets = {
            0: {"ε":0.05, "f":1.6, "q":0.002, "Du":1.0, "Dv":0.6, "dt":0.01, "init":"spiral"},
            1: {"ε":0.05, "f":1.6, "q":0.002, "Du":1.0, "Dv":0.6, "dt":0.01, "init":"target"},
            2: {"ε":0.02, "f":2.8, "q":0.005, "Du":1.2, "Dv":0.5, "dt":0.005, "init":"random"},
        }
        p = presets[idx]
        sim = self.sim
        sim.epsilon, sim.f_param, sim.q_param = p["ε"], p["f"], p["q"]
        sim.Du, sim.Dv, sim.dt = p["Du"], p["Dv"], p["dt"]
        def set_slider(name, val, scale):
            sl, lbl = self.sliders[name]
            sl.blockSignals(True)
            sl.setValue(int(val*scale))
            sl.blockSignals(False)
            lbl.setText(f"{val:.4f}")
        set_slider("ε", p["ε"], 1000)
        set_slider("f", p["f"], 1000)
        set_slider("q", p["q"], 100000)
        set_slider("Du", p["Du"], 100)
        set_slider("Dv", p["Dv"], 100)
        set_slider("dt", p["dt"], 10000)
        sim.reset(p["init"])
        self.viewer.gl_widget.update()

    def change_init_pattern(self):
        mapping = {"螺旋位相":"spiral", "中心斑":"center", "随机扰动":"random"}
        self.sim.reset(mapping[self.pattern_combo.currentText()])
        self.viewer.gl_widget.update()

    def closeEvent(self, event):
        self.viewer.timer.stop()
        ti.reset()
        event.accept()

if __name__ == "__main__":
    setup_gl_format()
    app = QApplication(sys.argv)
    window = BZWindow()
    window.show()
    sys.exit(app.exec())