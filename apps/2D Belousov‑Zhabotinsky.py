from custom_import import *

ti.init(arch=ti.gpu, default_fp=ti.f32)


def setup_gl_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(fmt)


@ti.data_oriented
class BZSimulation:
    """
    BZ 反应扩散系统 (Oregonator 简化模型)
        ε ∂u/∂t = u - u² - f·v·(u - q)/(u + q) + D_u ∇²u
        ∂v/∂t  = u - v + D_v ∇²v

    合并 kernel，同时计算拉普拉斯和反应，无中间场。
    """
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

        self.reset()

    @ti.kernel
    def step(self, epsilon: ti.f32, f: ti.f32, q: ti.f32,
             Du: ti.f32, Dv: ti.f32, dt: ti.f32):
        for i, j in ti.ndrange((1, self.size - 1), (1, self.size - 1)):
            u_center = self.u[i, j]
            v_center = self.v[i, j]

            lap_u = (self.u[i - 1, j] + self.u[i + 1, j] +
                     self.u[i, j - 1] + self.u[i, j + 1] - 4.0 * u_center)
            lap_v = (self.v[i - 1, j] + self.v[i + 1, j] +
                     self.v[i, j - 1] + self.v[i, j + 1] - 4.0 * v_center)

            reaction = u_center - u_center * u_center - f * v_center * (u_center - q) / (u_center + q + 1e-10)

            du = (1.0 / epsilon) * reaction + Du * lap_u
            dv = u_center - v_center + Dv * lap_v

            self.u[i, j] = max(0.0, min(1.0, u_center + du * dt))
            self.v[i, j] = max(0.0, min(1.0, v_center + dv * dt))

    def get_visualization(self, mode='u'):
        if mode == 'u':
            field = self.u.to_numpy()
            img = np.clip(field * 255, 0, 255).astype(np.uint8)
            return np.stack([img, img, img], axis=-1)
        elif mode == 'v':
            field = self.v.to_numpy()
            img = np.clip(field * 255, 0, 255).astype(np.uint8)
            return np.stack([img, img, img], axis=-1)
        else:  # phase
            u_np = self.u.to_numpy()
            v_np = self.v.to_numpy()
            combined = np.stack([u_np, v_np, np.zeros_like(u_np)], axis=-1)
            return (np.clip(combined * 255, 0, 255)).astype(np.uint8)

    @ti.kernel
    def init_center(self):
        center = self.size // 2
        r = self.size // 15
        for i, j in ti.ndrange(self.size, self.size):
            if (i - center) ** 2 + (j - center) ** 2 < r * r:
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
            if (i - center) ** 2 + (j - center) ** 2 < 4.0:
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


# ================== OpenGL 组件 ==================
class BZOpenGLWidget(QOpenGLWidget):
    def __init__(self, simulation, parent=None):
        super().__init__(parent)
        self.sim = simulation
        self.vis_mode = 'u'
        self.texture_id = None
        self.texture_size = (0, 0)
        self.setMinimumSize(600, 600)

    def set_visualization(self, mode):
        self.vis_mode = mode
        self.update()

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)

    def paintGL(self):
        img = self.sim.get_visualization(self.vis_mode)
        h, w, _ = img.shape

        glClear(GL_COLOR_BUFFER_BIT)
        ratio = self.devicePixelRatio()
        w_view, h_view = int(self.width() * ratio), int(self.height() * ratio)
        glViewport(0, 0, w_view, h_view)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, w, 0, h, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        if (w, h) != self.texture_size:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
            self.texture_size = (w, h)
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, img)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(w, 0)
        glTexCoord2f(1, 1); glVertex2f(w, h)
        glTexCoord2f(0, 1); glVertex2f(0, h)
        glEnd()


# ================== 主界面 ==================
class BZWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.simulation = BZSimulation(512)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.is_playing = True
        self.vis_mode = 'u'
        self.steps_per_frame = 20
        self.init_ui()
        self.timer.start(30)

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(6)

        # 公式展示
        eq_label = QLabel(
            "<b>Oregonator 模型（反应扩散方程）</b><br>"
            "ε ∂u/∂t = u − u² − f·v·(u−q)/(u+q) + D<sub>u</sub>∇²u<br>"
            "∂v/∂t = u − v + D<sub>v</sub>∇²v"
        )
        eq_label.setWordWrap(True)
        eq_label.setStyleSheet("QLabel { background: #f0f0f0; padding: 6px; border-radius: 4px; }")
        left_panel.addWidget(eq_label)

        # 参数网格
        param_grid = QGridLayout()
        param_grid.setHorizontalSpacing(8)
        param_grid.setVerticalSpacing(6)

        params = [
            ("ε", "时间分离比 ε", 0.01, 0.2, 0.05, 1000),
            ("f", "化学计量系数 f", 0.5, 3.0, 1.6, 1000),
            ("q", "阈值常数 q", 0.001, 0.02, 0.002, 100000),
            ("Du", "u 扩散系数 Du", 0.5, 2.0, 1.0, 100),
            ("Dv", "v 扩散系数 Dv", 0.2, 1.5, 0.6, 100),
            ("dt", "时间步长 dt", 0.001, 0.05, 0.01, 10000),
        ]

        self.sliders = {}
        row = 0
        for short, full, vmin, vmax, vdef, scale in params:
            label = QLabel(f"{short}: ")
            label.setMinimumWidth(40)
            value_label = QLabel(f"{vdef:.4f}")
            value_label.setMinimumWidth(50)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(int(vmin * scale), int(vmax * scale))
            slider.setValue(int(vdef * scale))

            def make_callback(name, lbl, scl):
                def cb(val):
                    scaled_val = val / scl
                    lbl.setText(f"{scaled_val:.4f}")
                    if name == "ε": self.simulation.epsilon = scaled_val
                    elif name == "f": self.simulation.f_param = scaled_val
                    elif name == "q": self.simulation.q_param = scaled_val
                    elif name == "Du": self.simulation.Du = scaled_val
                    elif name == "Dv": self.simulation.Dv = scaled_val
                    elif name == "dt": self.simulation.dt = scaled_val
                return cb

            slider.valueChanged.connect(make_callback(short, value_label, scale))
            self.sliders[short] = (slider, value_label)

            param_grid.addWidget(label, row, 0)
            param_grid.addWidget(slider, row, 1)
            param_grid.addWidget(value_label, row, 2)
            row += 1

        left_panel.addLayout(param_grid)

        # 预设与初始条件
        preset_layout = QGridLayout()
        preset_layout.addWidget(QLabel("预设:"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["经典螺旋波", "靶波 (中心激发)", "混沌 (高 f)"])
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_combo, 0, 1, 1, 2)

        preset_layout.addWidget(QLabel("初始图案:"), 1, 0)
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["螺旋位相", "中心斑", "随机扰动"])
        preset_layout.addWidget(self.pattern_combo, 1, 1, 1, 2)

        left_panel.addLayout(preset_layout)

        # 显示与控制
        disp_ctrl_grid = QGridLayout()
        disp_ctrl_grid.addWidget(QLabel("显示:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["u 浓度", "v 浓度", "相位 (u/R v/G)"])
        self.mode_combo.currentIndexChanged.connect(self.change_vis_mode)
        disp_ctrl_grid.addWidget(self.mode_combo, 0, 1)

        self.play_button = QPushButton("暂停")
        self.play_button.clicked.connect(self.toggle_play)
        disp_ctrl_grid.addWidget(self.play_button, 1, 0)

        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_sim)
        disp_ctrl_grid.addWidget(self.reset_button, 1, 1)

        left_panel.addLayout(disp_ctrl_grid)

        # 每帧步数
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("每帧步数:"))
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setRange(1, 100)
        self.steps_slider.setValue(20)
        self.steps_slider.valueChanged.connect(lambda v: setattr(self, 'steps_per_frame', v))
        self.steps_label = QLabel("20")
        self.steps_label.setFixedWidth(30)
        steps_layout.addWidget(self.steps_slider)
        steps_layout.addWidget(self.steps_label)
        left_panel.addLayout(steps_layout)

        self.save_button = QPushButton("保存图像")
        self.save_button.clicked.connect(self.save_image)
        left_panel.addWidget(self.save_button)

        left_panel.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(320)
        main_layout.addWidget(left_widget)

        # 右侧画布
        self.gl_widget = BZOpenGLWidget(self.simulation)
        self.gl_widget.setFixedSize(600, 600)
        main_layout.addWidget(self.gl_widget)

        self.setLayout(main_layout)

    def apply_preset(self, idx):
        presets = {
            0: {"ε": 0.05, "f": 1.6, "q": 0.002, "Du": 1.0, "Dv": 0.6, "dt": 0.01, "init": "spiral"},
            1: {"ε": 0.05, "f": 1.6, "q": 0.002, "Du": 1.0, "Dv": 0.6, "dt": 0.01, "init": "target"},
            2: {"ε": 0.02, "f": 2.8, "q": 0.005, "Du": 1.2, "Dv": 0.5, "dt": 0.005, "init": "random"},
        }
        p = presets[idx]
        self.simulation.epsilon = p["ε"]
        self.simulation.f_param = p["f"]
        self.simulation.q_param = p["q"]
        self.simulation.Du = p["Du"]
        self.simulation.Dv = p["Dv"]
        self.simulation.dt = p["dt"]

        def set_slider(name, val, scale):
            sl, lbl = self.sliders[name]
            sl.blockSignals(True)
            sl.setValue(int(val * scale))
            sl.blockSignals(False)
            lbl.setText(f"{val:.4f}")

        set_slider("ε", p["ε"], 1000)
        set_slider("f", p["f"], 1000)
        set_slider("q", p["q"], 100000)
        set_slider("Du", p["Du"], 100)
        set_slider("Dv", p["Dv"], 100)
        set_slider("dt", p["dt"], 10000)

        self.simulation.reset(p["init"])
        self.gl_widget.update()

    def reset_sim(self):
        init_map = {"螺旋位相": "spiral", "中心斑": "center", "随机扰动": "random"}
        self.simulation.reset(init_map[self.pattern_combo.currentText()])
        self.gl_widget.update()

    def change_vis_mode(self, idx):
        modes = ["u", "v", "phase"]
        self.vis_mode = modes[idx]
        self.gl_widget.set_visualization(self.vis_mode)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_button.setText("播放" if not self.is_playing else "暂停")

    def save_image(self):
        img = self.simulation.get_visualization(self.vis_mode)
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        fname = f"BZ_wave_f{self.simulation.f_param:.2f}_e{self.simulation.epsilon:.3f}.png"
        path, _ = QFileDialog.getSaveFileName(self, "保存图像", fname, "PNG (*.png)")
        if path and qimg.save(path):
            QMessageBox.information(self, "保存成功", f"已保存至:\n{path}")

    def update_simulation(self):
        if self.is_playing:
            for _ in range(self.steps_per_frame):
                self.simulation.step(self.simulation.epsilon,
                                     self.simulation.f_param,
                                     self.simulation.q_param,
                                     self.simulation.Du,
                                     self.simulation.Dv,
                                     self.simulation.dt)
            self.gl_widget.update()


class BZWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BZ 化学波 · 螺旋波/靶波/混沌 (Taichi + OpenGL)")
        self.setGeometry(100, 100, 920, 650)
        self.central = BZWidget()
        self.setCentralWidget(self.central)

    def closeEvent(self, event):
        self.central.timer.stop()
        ti.reset()
        event.accept()


if __name__ == "__main__":
    setup_gl_format()
    app = QApplication(sys.argv)
    window = BZWindow()
    window.show()
    sys.exit(app.exec())