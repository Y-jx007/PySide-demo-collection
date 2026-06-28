from custom_import import *   # 保留你的自定义导入

ti.init(arch=ti.gpu, default_fp=ti.f32)

def setup_gl_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(fmt)

@ti.data_oriented
class ReactionDiffusionSimulation:
    def __init__(self, size=512):
        self.size = size
        self.width = size
        self.height = size

        self.U = ti.field(dtype=ti.f32, shape=(size, size))
        self.V = ti.field(dtype=ti.f32, shape=(size, size))
        self.U_temp = ti.field(dtype=ti.f32, shape=(size, size))
        self.V_temp = ti.field(dtype=ti.f32, shape=(size, size))

        self.Du = 0.16
        self.Dv = 0.08
        self.f = 0.04
        self.k = 0.06

        self.reset()

    @ti.kernel
    def compute_laplacian(self, field: ti.template(), result: ti.template()):
        for i, j in ti.ndrange((1, self.size - 1), (1, self.size - 1)):
            result[i, j] = (field[i - 1, j] + field[i + 1, j] +
                            field[i, j - 1] + field[i, j + 1] -
                            4 * field[i, j])

    @ti.kernel
    def update_step(self, Du: ti.f32, Dv: ti.f32, f: ti.f32, k: ti.f32):
        for i, j in ti.ndrange((1, self.size - 1), (1, self.size - 1)):
            reaction = self.U[i, j] * self.V[i, j] * self.V[i, j]
            U_new = self.U[i, j] + Du * self.U_temp[i, j] - reaction + f * (1 - self.U[i, j])
            V_new = self.V[i, j] + Dv * self.V_temp[i, j] + reaction - (f + k) * self.V[i, j]
            self.U[i, j] = max(0.0, min(1.0, U_new))
            self.V[i, j] = max(0.0, min(1.0, V_new))

    def step(self):
        self.compute_laplacian(self.U, self.U_temp)
        self.compute_laplacian(self.V, self.V_temp)
        self.update_step(self.Du, self.Dv, self.f, self.k)

    def get_visualization(self, mode='V', color_scheme='default'):
        # 返回 RGB uint8 数组 (height, width, 3)
        if mode == 'V':
            field_np = self.V.to_numpy()
        elif mode == 'U':
            field_np = self.U.to_numpy()
        else:  # UV混合
            U_np = self.U.to_numpy()
            V_np = self.V.to_numpy()
            field_np = np.stack([U_np, V_np, np.zeros_like(U_np)], axis=-1) * 255
            return np.clip(field_np, 0, 255).astype(np.uint8)

        field_visual = np.clip(field_np * 255, 0, 255).astype(np.uint8)
        if color_scheme == 'inverse':
            field_visual = 255 - field_visual
        return np.stack([field_visual, field_visual, field_visual], axis=-1)

    def reset(self, pattern='random'):
        self.U.fill(1.0)
        self.V.fill(0.0)
        if pattern == 'center':
            self._init_center_pattern()
        elif pattern == 'random':
            self._init_random_pattern()
        elif pattern == 'edges':
            self._init_edges_pattern()
        elif pattern == 'spots':
            self._init_spots_pattern()

    @ti.kernel
    def _init_center_pattern(self):
        center = self.size // 2
        size = self.size // 20
        for i, j in ti.ndrange(self.size, self.size):
            if ti.abs(i - center) < size and ti.abs(j - center) < size:
                self.V[i, j] = 1.0

    @ti.kernel
    def _init_random_pattern(self):
        border = self.size // 20
        density = 0.001 * (512 / self.size)
        for i, j in ti.ndrange(self.size, self.size):
            if (border <= i < self.size - border and border <= j < self.size - border and
                    ti.random() < density):
                size = max(2, self.size // 256)
                for di, dj in ti.ndrange((-size, size + 1), (-size, size + 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        self.V[ni, nj] = 1.0

    @ti.kernel
    def _init_edges_pattern(self):
        border_size = max(3, self.size // 170)
        for i, j in ti.ndrange(self.size, self.size):
            if (i < border_size or i >= self.size - border_size or
                    j < border_size or j >= self.size - border_size):
                self.V[i, j] = 1.0

    @ti.kernel
    def _init_spots_pattern(self):
        spacing = max(15, self.size // 17)
        for i, j in ti.ndrange(self.size, self.size):
            if i % spacing == spacing // 2 and j % spacing == spacing // 2:
                size = max(2, self.size // 256)
                for di, dj in ti.ndrange((-size, size + 1), (-size, size + 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        self.V[ni, nj] = 1.0


# ================== OpenGL 渲染组件 ==================
class ReactionDiffusionGLWidget(QOpenGLWidget):
    def __init__(self, simulation, parent=None):
        super().__init__(parent)
        self.sim = simulation
        self.vis_mode = 'V'
        self.color_scheme = 'default'
        self.texture_id = None
        self.texture_size = (0, 0)
        self.setMinimumSize(600, 600)

    def set_visualization(self, mode, color_scheme):
        self.vis_mode = mode
        self.color_scheme = color_scheme
        self.update()

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)

    def resizeGL(self, w, h):
        pass

    def paintGL(self):
        # 获取当前模拟图像
        img = self.sim.get_visualization(self.vis_mode, self.color_scheme)
        h, w, _ = img.shape

        glClear(GL_COLOR_BUFFER_BIT)

        # 设置正交投影，使图像正好覆盖视口
        ratio = self.devicePixelRatio()
        w_view = int(self.width() * ratio)
        h_view = int(self.height() * ratio)
        glViewport(0, 0, w_view, h_view)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, w, 0, h, -1, 1)          # 世界坐标与图像像素一一对应
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 更新纹理（若尺寸变化则重新分配，否则仅替换内容）
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        if (w, h) != self.texture_size:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                            GL_RGB, GL_UNSIGNED_BYTE, img)
            self.texture_size = (w, h)
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
                               GL_RGB, GL_UNSIGNED_BYTE, img)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # 绘制全屏四边形
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex2f(0, 0)
        glTexCoord2f(1.0, 0.0); glVertex2f(w, 0)
        glTexCoord2f(1.0, 1.0); glVertex2f(w, h)
        glTexCoord2f(0.0, 1.0); glVertex2f(0, h)
        glEnd()


# ================== 主界面（与原来布局相同）==================
class ReactionDiffusionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.simulation = ReactionDiffusionSimulation(512)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.is_playing = True
        self.visualization_mode = 'V'
        self.color_scheme = 'default'
        self.steps_per_frame = 60

        self.init_ui()
        self.timer.start(30)

    def init_ui(self):
        main_layout = QHBoxLayout()

        # ----- 左侧参数面板（与原代码完全一致）-----
        left_panel = QVBoxLayout()

        # 参数控制组
        params_group = QGroupBox("模拟参数")
        params_layout = QVBoxLayout()
        self.f_slider, self.f_label = self.create_slider_with_label("供给率 (f):", 0.01, 0.08, 0.04, 10000)
        self.k_slider, self.k_label = self.create_slider_with_label("去除率 (k):", 0.04, 0.08, 0.06, 10000)
        self.Du_slider, self.Du_label = self.create_slider_with_label("U扩散系数:", 0.1, 0.3, 0.16, 1000)
        self.Dv_slider, self.Dv_label = self.create_slider_with_label("V扩散系数:", 0.04, 0.12, 0.08, 1000)
        params_layout.addWidget(self.f_slider)
        params_layout.addWidget(self.k_slider)
        params_layout.addWidget(self.Du_slider)
        params_layout.addWidget(self.Dv_slider)

        # 手动输入
        manual_input_layout = QHBoxLayout()
        manual_input_layout.addWidget(QLabel("手动输入:"))
        self.manual_f_input = QLineEdit(); self.manual_f_input.setPlaceholderText("f值"); self.manual_f_input.setMaximumWidth(60)
        self.manual_k_input = QLineEdit(); self.manual_k_input.setPlaceholderText("k值"); self.manual_k_input.setMaximumWidth(60)
        self.apply_manual_button = QPushButton("应用"); self.apply_manual_button.setMaximumWidth(60); self.apply_manual_button.clicked.connect(self.apply_manual_parameters)
        manual_input_layout.addWidget(self.manual_f_input)
        manual_input_layout.addWidget(self.manual_k_input)
        manual_input_layout.addWidget(self.apply_manual_button)
        params_layout.addLayout(manual_input_layout)
        params_group.setLayout(params_layout)
        left_panel.addWidget(params_group)

        # 显示设置组
        display_group = QGroupBox("显示设置")
        display_layout = QVBoxLayout()
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("显示模式:"))
        self.mode_combo = QComboBox(); self.mode_combo.addItems(["物质V", "物质U", "UV混合"]); self.mode_combo.currentTextChanged.connect(self.change_visualization_mode)
        mode_layout.addWidget(self.mode_combo)
        display_layout.addLayout(mode_layout)
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("颜色:"))
        self.color_combo = QComboBox(); self.color_combo.addItems(["默认", "反色"]); self.color_combo.currentTextChanged.connect(self.change_color_scheme)
        color_layout.addWidget(self.color_combo)
        display_layout.addLayout(color_layout)
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("图像尺寸:"))
        self.size_combo = QComboBox(); self.size_combo.addItems(["128x128", "256x256", "512x512", "1024x1024"]); self.size_combo.setCurrentText("512x512"); self.size_combo.currentTextChanged.connect(self.change_size)
        size_layout.addWidget(self.size_combo)
        display_layout.addLayout(size_layout)
        display_group.setLayout(display_layout)
        left_panel.addWidget(display_group)

        # 初始条件组
        init_group = QGroupBox("初始条件")
        init_layout = QVBoxLayout()
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("初始模式:"))
        self.pattern_combo = QComboBox(); self.pattern_combo.addItems(["随机", "中心", "边缘", "点阵"])
        pattern_layout.addWidget(self.pattern_combo)
        init_layout.addLayout(pattern_layout)
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("预设:"))
        self.preset_combo = QComboBox(); self.preset_combo.addItems(["迷宫", "条纹", "斑点", "蜂巢", "云雾"]); self.preset_combo.currentTextChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_combo)
        init_layout.addLayout(preset_layout)
        init_group.setLayout(init_layout)
        left_panel.addWidget(init_group)

        # 控制组
        control_group = QGroupBox("控制")
        control_layout = QVBoxLayout()
        button_row1 = QHBoxLayout()
        self.play_button = QPushButton("暂停"); self.play_button.clicked.connect(self.toggle_play)
        self.reset_button = QPushButton("重置"); self.reset_button.clicked.connect(self.reset_simulation)
        button_row1.addWidget(self.play_button); button_row1.addWidget(self.reset_button)
        control_layout.addLayout(button_row1)
        button_row2 = QHBoxLayout()
        self.step_button = QPushButton("单步"); self.step_button.clicked.connect(self.step_simulation)
        self.save_button = QPushButton("保存"); self.save_button.clicked.connect(self.save_image)
        button_row2.addWidget(self.step_button); button_row2.addWidget(self.save_button)
        control_layout.addLayout(button_row2)
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("每帧步数:"))
        self.steps_slider = QSlider(Qt.Horizontal); self.steps_slider.setMinimum(1); self.steps_slider.setMaximum(100); self.steps_slider.setValue(60)
        self.steps_slider.valueChanged.connect(self.change_steps_per_frame)
        self.steps_label = QLabel("60")
        steps_layout.addWidget(self.steps_slider); steps_layout.addWidget(self.steps_label)
        control_layout.addLayout(steps_layout)
        control_group.setLayout(control_layout)
        left_panel.addWidget(control_group)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(300)
        left_widget.setFixedHeight(630)
        main_layout.addWidget(left_widget)

        # ----- 右侧：使用 OpenGL 组件替代 QLabel -----
        self.gl_widget = ReactionDiffusionGLWidget(self.simulation)
        self.gl_widget.setFixedSize(600, 600)
        main_layout.addWidget(self.gl_widget)

        self.setLayout(main_layout)

    # ---------- 以下 UI 控制函数与原版相同，只对显示刷新做适配 ----------
    def create_slider_with_label(self, label_text, min_val, max_val, default_val, scale=100):
        layout = QHBoxLayout()
        label = QLabel(f"{label_text} {default_val:.4f}")
        label.setMinimumWidth(120)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * scale))
        slider.setMaximum(int(max_val * scale))
        slider.setValue(int(default_val * scale))
        def update_value(value):
            scaled = value / scale
            label.setText(f"{label_text} {scaled:.4f}")
            self.update_parameter(label_text.split()[0].lower(), scaled)
        slider.valueChanged.connect(update_value)
        layout.addWidget(label); layout.addWidget(slider)
        container = QWidget(); container.setLayout(layout)
        return container, label

    def update_parameter(self, param_name, value):
        if "供给率" in param_name: self.simulation.f = value
        elif "去除率" in param_name: self.simulation.k = value
        elif "u扩散" in param_name: self.simulation.Du = value
        elif "v扩散" in param_name: self.simulation.Dv = value

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_button.setText("暂停" if self.is_playing else "开始")

    def reset_simulation(self):
        pattern_map = {"随机": "random", "中心": "center", "边缘": "edges", "点阵": "spots"}
        self.simulation.reset(pattern_map[self.pattern_combo.currentText()])
        self.gl_widget.update()

    def step_simulation(self):
        self.simulation.step()
        self.gl_widget.update()

    def change_visualization_mode(self, text):
        mode_map = {"物质V": "V", "物质U": "U", "UV混合": "UV"}
        self.visualization_mode = mode_map[text]
        self.gl_widget.set_visualization(self.visualization_mode, self.color_scheme)

    def change_color_scheme(self, text):
        scheme_map = {"默认": "default", "反色": "inverse"}
        self.color_scheme = scheme_map[text]
        self.gl_widget.set_visualization(self.visualization_mode, self.color_scheme)

    def change_size(self, text):
        size_map = {"128x128": 128, "256x256": 256, "512x512": 512, "1024x1024": 1024}
        if text in size_map:
            new_size = size_map[text]
            f, k, Du, Dv = self.simulation.f, self.simulation.k, self.simulation.Du, self.simulation.Dv
            self.simulation = ReactionDiffusionSimulation(new_size)
            self.simulation.f, self.simulation.k, self.simulation.Du, self.simulation.Dv = f, k, Du, Dv
            self.gl_widget.sim = self.simulation   # 更新 widget 中的引用
            self.reset_simulation()

    def change_steps_per_frame(self, value):
        self.steps_per_frame = value
        self.steps_label.setText(str(value))

    def apply_preset(self, name):
        presets = {
            "迷宫": {"f": 0.0400, "k": 0.0600},
            "条纹": {"f": 0.0550, "k": 0.0650},
            "斑点": {"f": 0.0400, "k": 0.0650},
            "蜂巢": {"f": 0.0300, "k": 0.0550},
            "云雾": {"f": 0.0160, "k": 0.0450},
        }
        if name in presets:
            p = presets[name]
            self.simulation.f, self.simulation.k = p["f"], p["k"]
            self.f_slider.layout().itemAt(1).widget().setValue(int(p["f"] * 10000))
            self.k_slider.layout().itemAt(1).widget().setValue(int(p["k"] * 10000))
            self.f_label.setText(f"供给率 (f): {p['f']:.4f}")
            self.k_label.setText(f"去除率 (k): {p['k']:.4f}")
            self.reset_simulation()

    def apply_manual_parameters(self):
        try:
            f_text = self.manual_f_input.text().strip()
            k_text = self.manual_k_input.text().strip()
            if f_text:
                f_val = float(f_text)
                if 0.01 <= f_val <= 0.08:
                    self.simulation.f = f_val
                    self.f_slider.layout().itemAt(1).widget().setValue(int(f_val * 10000))
                    self.f_label.setText(f"供给率 (f): {f_val:.4f}")
                else:
                    QMessageBox.warning(self, "输入错误", "f值必须在0.01到0.08之间")
                    return
            if k_text:
                k_val = float(k_text)
                if 0.04 <= k_val <= 0.08:
                    self.simulation.k = k_val
                    self.k_slider.layout().itemAt(1).widget().setValue(int(k_val * 10000))
                    self.k_label.setText(f"去除率 (k): {k_val:.4f}")
                else:
                    QMessageBox.warning(self, "输入错误", "k值必须在0.04到0.08之间")
                    return
            self.reset_simulation()
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字")

    def save_image(self):
        img = self.simulation.get_visualization(self.visualization_mode, self.color_scheme)
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        default_name = f"turing_pattern_f{self.simulation.f:.4f}_k{self.simulation.k:.4f}.png"
        path, _ = QFileDialog.getSaveFileName(self, "保存图像", default_name, "PNG (*.png);;JPEG (*.jpg)")
        if path and qimg.save(path):
            QMessageBox.information(self, "保存成功", f"已保存至:\n{path}")

    def update_simulation(self):
        if self.is_playing:
            for _ in range(self.steps_per_frame):
                self.simulation.step()
            self.gl_widget.update()   # 触发 OpenGL 重绘

class ReactionDiffusionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("反应扩散方程模拟 - 图灵斑图生成器 (Taichi GPU + OpenGL)")
        self.setGeometry(100, 100, 900, 650)
        self.central = ReactionDiffusionWidget()
        self.setCentralWidget(self.central)

    def closeEvent(self, event):
        self.central.timer.stop()
        ti.reset()
        event.accept()

if __name__ == "__main__":
    setup_gl_format()
    app = QApplication(sys.argv)
    window = ReactionDiffusionWindow()
    window.show()
    sys.exit(app.exec())