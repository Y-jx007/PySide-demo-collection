from custom_import import *

ti.init(arch=ti.gpu, default_fp=ti.f32)

@ti.data_oriented
class TuringSimulation(SimulationBase):
    def __init__(self, size=512):
        self.size = size
        self.U = ti.field(ti.f32, shape=(size, size))
        self.V = ti.field(ti.f32, shape=(size, size))
        self.U_temp = ti.field(ti.f32, shape=(size, size))
        self.V_temp = ti.field(ti.f32, shape=(size, size))
        self.Du = 0.16
        self.Dv = 0.08
        self.f = 0.04
        self.k = 0.06
        self.color_scheme = 'default'
        self.reset('random')

    @ti.kernel
    def compute_laplacian(self, field: ti.template(), result: ti.template()):
        for i, j in ti.ndrange((1, self.size-1), (1, self.size-1)):
            result[i, j] = (field[i-1, j] + field[i+1, j] +
                            field[i, j-1] + field[i, j+1] - 4*field[i, j])

    @ti.kernel
    def update_step(self, Du: ti.f32, Dv: ti.f32, f: ti.f32, k: ti.f32):
        for i, j in ti.ndrange((1, self.size-1), (1, self.size-1)):
            reaction = self.U[i, j] * self.V[i, j] * self.V[i, j]
            U_new = self.U[i, j] + Du*self.U_temp[i, j] - reaction + f*(1 - self.U[i, j])
            V_new = self.V[i, j] + Dv*self.V_temp[i, j] + reaction - (f + k)*self.V[i, j]
            self.U[i, j] = max(0.0, min(1.0, U_new))
            self.V[i, j] = max(0.0, min(1.0, V_new))

    def step(self):
        self.compute_laplacian(self.U, self.U_temp)
        self.compute_laplacian(self.V, self.V_temp)
        self.update_step(self.Du, self.Dv, self.f, self.k)

    def get_image(self, mode='V'):
        if mode == 'V':
            field = self.V.to_numpy()
        elif mode == 'U':
            field = self.U.to_numpy()
        else:  # UV混合
            u = self.U.to_numpy()
            v = self.V.to_numpy()
            if self.color_scheme == 'inverse':
                u, v = 1-u, 1-v
            combined = np.stack([u, v, np.zeros_like(u)], axis=-1)
            return (np.clip(combined*255, 0, 255)).astype(np.uint8)

        if self.color_scheme == 'inverse':
            field = 1 - field
        img = np.clip(field*255, 0, 255).astype(np.uint8)
        return np.stack([img, img, img], axis=-1)

    def reset(self, pattern='random'):
        self.U.fill(1.0)
        self.V.fill(0.0)
        if pattern == 'center':
            self._init_center()
        elif pattern == 'random':
            self._init_random()
        elif pattern == 'edges':
            self._init_edges()
        elif pattern == 'spots':
            self._init_spots()
        else:
            self._init_random()

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
            if border <= i < self.size - border and border <= j < self.size - border and ti.random() < density:
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

class TuringWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图灵斑图生成器")
        self.sim = TuringSimulation(512)
        self.viewer = SimulationViewer(self.sim)
        self.viewer.set_display_modes(['V', 'U', 'UV混合'])
        self.setCentralWidget(self.viewer)
        self._add_turing_params()
        self.setGeometry(100, 100, 920, 650)

    def _add_turing_params(self):
        panel = QWidget()
        layout = QVBoxLayout()

        # 参数控制组
        params_group = QGroupBox("模拟参数")
        p_layout = QVBoxLayout()
        self.f_slider, self.f_label = self._make_slider("供给率 (f):", 0.01, 0.08, 0.04, 10000, 'f')
        self.k_slider, self.k_label = self._make_slider("去除率 (k):", 0.04, 0.08, 0.06, 10000, 'k')
        self.Du_slider, self.Du_label = self._make_slider("U扩散系数:", 0.1, 0.3, 0.16, 1000, 'Du')
        self.Dv_slider, self.Dv_label = self._make_slider("V扩散系数:", 0.04, 0.12, 0.08, 1000, 'Dv')
        p_layout.addWidget(self.f_slider)
        p_layout.addWidget(self.k_slider)
        p_layout.addWidget(self.Du_slider)
        p_layout.addWidget(self.Dv_slider)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("手动输入:"))
        self.manual_f = QLineEdit(); self.manual_f.setPlaceholderText("f值"); self.manual_f.setMaximumWidth(60)
        self.manual_k = QLineEdit(); self.manual_k.setPlaceholderText("k值"); self.manual_k.setMaximumWidth(60)
        apply_btn = QPushButton("应用"); apply_btn.setMaximumWidth(60); apply_btn.clicked.connect(self._apply_manual)
        input_layout.addWidget(self.manual_f); input_layout.addWidget(self.manual_k); input_layout.addWidget(apply_btn)
        p_layout.addLayout(input_layout)
        params_group.setLayout(p_layout)
        layout.addWidget(params_group)

        # 显示设置
        disp_group = QGroupBox("显示设置")
        d_layout = QVBoxLayout()
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("颜色:"))
        self.color_combo = QComboBox(); self.color_combo.addItems(["默认", "反色"])
        self.color_combo.currentTextChanged.connect(self._change_color)
        color_layout.addWidget(self.color_combo)
        d_layout.addLayout(color_layout)
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("图像尺寸:"))
        self.size_combo = QComboBox(); self.size_combo.addItems(["128x128","256x256","512x512","1024x1024"])
        self.size_combo.setCurrentText("512x512")
        self.size_combo.currentTextChanged.connect(self._change_size)
        size_layout.addWidget(self.size_combo)
        d_layout.addLayout(size_layout)
        disp_group.setLayout(d_layout)
        layout.addWidget(disp_group)

        # 初始条件
        init_group = QGroupBox("初始条件")
        i_layout = QVBoxLayout()
        pat_layout = QHBoxLayout()
        pat_layout.addWidget(QLabel("初始模式:"))
        self.pattern_combo = QComboBox(); self.pattern_combo.addItems(["随机","中心","边缘","点阵"])
        pat_layout.addWidget(self.pattern_combo)
        i_layout.addLayout(pat_layout)
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("预设:"))
        self.preset_combo = QComboBox(); self.preset_combo.addItems(["迷宫","条纹","斑点","蜂巢","云雾"])
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_layout.addWidget(self.preset_combo)
        i_layout.addLayout(preset_layout)
        init_group.setLayout(i_layout)
        layout.addWidget(init_group)

        panel.setLayout(layout)
        self.viewer.param_panel.insertWidget(0, panel)

        # 自定义重置
        original_reset = self.viewer.reset_sim
        def custom_reset():
            mapping = {"随机":"random","中心":"center","边缘":"edges","点阵":"spots"}
            self.sim.reset(mapping[self.pattern_combo.currentText()])
            self.viewer.gl_widget.update()
        self.viewer.reset_sim = custom_reset

    def _make_slider(self, label, vmin, vmax, vdef, scale, attr):
        w = QWidget()
        lay = QHBoxLayout()
        lbl = QLabel(f"{label} {vdef:.4f}")
        lbl.setMinimumWidth(120)
        s = QSlider(Qt.Horizontal)
        s.setMinimum(int(vmin*scale)); s.setMaximum(int(vmax*scale)); s.setValue(int(vdef*scale))
        def update(val):
            scaled = val/scale
            lbl.setText(f"{label} {scaled:.4f}")
            setattr(self.sim, attr, scaled)
        s.valueChanged.connect(update)
        lay.addWidget(lbl); lay.addWidget(s)
        w.setLayout(lay)
        return w, lbl

    def _apply_manual(self):
        try:
            ft, kt = self.manual_f.text().strip(), self.manual_k.text().strip()
            if ft:
                fv = float(ft)
                if 0.01 <= fv <= 0.08:
                    self.sim.f = fv
                    self.f_slider.layout().itemAt(1).widget().setValue(int(fv*10000))
                    self.f_label.setText(f"供给率 (f): {fv:.4f}")
                else: QMessageBox.warning(self,"错误","f值需在0.01~0.08")
            if kt:
                kv = float(kt)
                if 0.04 <= kv <= 0.08:
                    self.sim.k = kv
                    self.k_slider.layout().itemAt(1).widget().setValue(int(kv*10000))
                    self.k_label.setText(f"去除率 (k): {kv:.4f}")
                else: QMessageBox.warning(self,"错误","k值需在0.04~0.08")
        except ValueError:
            QMessageBox.warning(self,"错误","请输入有效数字")

    def _change_color(self, text):
        self.sim.color_scheme = 'default' if text == '默认' else 'inverse'
        self.viewer.gl_widget.update()

    def _change_size(self, text):
        sizes = {"128x128":128,"256x256":256,"512x512":512,"1024x1024":1024}
        if text in sizes:
            ns = sizes[text]
            f,k,Du,Dv,cs = self.sim.f, self.sim.k, self.sim.Du, self.sim.Dv, self.sim.color_scheme
            self.sim = TuringSimulation(ns)
            self.sim.f, self.sim.k, self.sim.Du, self.sim.Dv = f,k,Du,Dv
            self.sim.color_scheme = cs
            self.viewer.sim = self.sim
            self.viewer.reset_sim()

    def _apply_preset(self, name):
        presets = {"迷宫":(0.04,0.06),"条纹":(0.055,0.065),"斑点":(0.04,0.065),
                   "蜂巢":(0.03,0.055),"云雾":(0.016,0.045)}
        if name in presets:
            fv, kv = presets[name]
            self.sim.f, self.sim.k = fv, kv
            self.f_slider.layout().itemAt(1).widget().setValue(int(fv*10000))
            self.k_slider.layout().itemAt(1).widget().setValue(int(kv*10000))
            self.f_label.setText(f"供给率 (f): {fv:.4f}")
            self.k_label.setText(f"去除率 (k): {kv:.4f}")
            self.viewer.reset_sim()

    def closeEvent(self, event):
        self.viewer.timer.stop()
        ti.reset()
        event.accept()

if __name__ == "__main__":
    setup_gl_format()
    app = QApplication(sys.argv)
    window = TuringWindow()
    window.show()
    sys.exit(app.exec())