from custom_import import *

ti.init(arch=ti.gpu, default_fp=ti.f32)

@ti.data_oriented
class RDLeniaSimulation(SimulationBase):
    def __init__(self, size=512):
        self.size = size
        # 主场 U
        self.U = ti.field(ti.f32, shape=(size, size))
        # 辅助变量场 Vj，论文使用 M=40
        self.M = 40
        self.V = ti.field(ti.f32, shape=(self.M, size, size))
        # 拉普拉斯算子的临时存储
        self.U_temp = ti.field(ti.f32, shape=(size, size))
        self.V_temp = ti.field(ti.f32, shape=(self.M, size, size))

        # ---------- 论文核心参数 (Section 2.2.1) ----------
        self.D = 0.0                 # 主场的扩散系数，论文设为 0[reference:5]
        self.mu = 0.1               # μ[reference:6]
        self.epsilon = 0.005        # ε[reference:7]
        self.Dj_base = 1.0          # Dj = j[reference:8]
        # 生长函数参数 (来自原版Lenia)
        self.m = 0.15               # 生长函数均值
        self.s = 0.015              # 生长函数宽度
        # ------------------------------------------------

        # α0 和 αj 系数 (通过线性回归得到，此处为论文示例值)
        # 注意：这些值需要根据具体 kernel 预先计算，这里是占位演示
        self.alpha0 = -0.5
        self.alphas = ti.field(ti.f32, shape=self.M)
        self._init_alphas()  # 初始化 alphas

        self.color_scheme = 'default'
        self.reset('random')

    @ti.kernel
    def _init_alphas(self):
        # 初始化 alpha 系数 (示例，实际应通过回归计算)
        for j in range(self.M):
            self.alphas[j] = 0.1 / (j + 1)

    @ti.kernel
    def compute_laplacian_2d(self, field: ti.template(), result: ti.template()):
        """计算 2D 标量场的拉普拉斯算子 (5点差分)"""
        for i, j in ti.ndrange((1, self.size-1), (1, self.size-1)):
            result[i, j] = (field[i-1, j] + field[i+1, j] +
                            field[i, j-1] + field[i, j+1] - 4.0 * field[i, j])

    @ti.kernel
    def compute_laplacian_3d(self, field: ti.template(), result: ti.template()):
        """计算 3D 标量场 (M, size, size) 的拉普拉斯算子"""
        for idx, i, j in ti.ndrange((1, self.M-1), (1, self.size-1), (1, self.size-1)):
            # 注意：这里只对空间维度 (i, j) 计算拉普拉斯，不对 M 维度计算
            result[idx, i, j] = (field[idx, i-1, j] + field[idx, i+1, j] +
                                 field[idx, i, j-1] + field[idx, i, j+1] - 4.0 * field[idx, i, j])

    @ti.kernel
    def update_step(self):
        """执行一步 RD Lenia 更新 (论文 Eq. 8)[reference:9]"""
        # 先更新辅助变量 Vj (快变量)
        for idx, i, j in ti.ndrange(self.M, self.size, self.size):
            Dj = self.Dj_base * (idx + 1)  # Dj = j[reference:10]
            v_new = self.V[idx, i, j] + (1.0 / self.epsilon) * (
                Dj * self.V_temp[idx, i, j] + self.mu * self.U[i, j] - self.V[idx, i, j]
            )
            self.V[idx, i, j] = max(0.0, min(1.0, v_new))

        # 再更新主场 U (慢变量)
        for i, j in ti.ndrange(self.size, self.size):
            # 计算组合输入: alpha0 + sum(alpha_j * v_j)
            combined = self.alpha0
            for idx in range(self.M):
                combined += self.alphas[idx] * self.V[idx, i, j]

            # 生长函数 T(z) = 2 * exp(-((z - m) / s)^2) - 1
            z = (combined - self.m) / self.s
            growth = 2.0 * ti.exp(-z * z) - 1.0

            # 反应项 F(u, K*u) = T(K*u) - u[reference:11]
            reaction = growth - self.U[i, j]

            # 更新 U: du/dt = D * laplacian(U) + reaction[reference:12]
            u_new = self.U[i, j] + self.D * self.U_temp[i, j] + reaction

            # 裁剪到 [0, 1]
            self.U[i, j] = max(0.0, min(1.0, u_new))

    def step(self):
        """执行一个完整的模拟步骤"""
        # 计算所有场的拉普拉斯算子
        self.compute_laplacian_2d(self.U, self.U_temp)
        self.compute_laplacian_3d(self.V, self.V_temp)
        # 执行更新
        self.update_step()

    def get_image(self, mode='U'):
        """根据模式返回用于显示的图像"""
        if mode == 'U':
            field = self.U.to_numpy()
        elif mode == 'V_sum':
            # 显示所有 Vj 的和，以观察辅助变量的整体效果
            v_sum = np.sum(self.V.to_numpy(), axis=0)
            field = np.clip(v_sum, 0, 1)
        else:
            field = self.U.to_numpy()

        if self.color_scheme == 'inverse':
            field = 1 - field
        img = np.clip(field * 255, 0, 255).astype(np.uint8)
        return np.stack([img, img, img], axis=-1)

    def reset(self, pattern='random'):
        """重置模拟状态"""
        self.U.fill(0.0)
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
                self.U[i, j] = 1.0

    @ti.kernel
    def _init_random(self):
        border = self.size // 20
        density = 0.001 * (512 / self.size)
        for i, j in ti.ndrange(self.size, self.size):
            if border <= i < self.size - border and border <= j < self.size - border:
                if ti.random() < density:
                    s = max(2, self.size // 256)
                    for di, dj in ti.ndrange((-s, s+1), (-s, s+1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            self.U[ni, nj] = 1.0

    @ti.kernel
    def _init_edges(self):
        b = max(3, self.size // 170)
        for i, j in ti.ndrange(self.size, self.size):
            if i < b or i >= self.size - b or j < b or j >= self.size - b:
                self.U[i, j] = 1.0

    @ti.kernel
    def _init_spots(self):
        spacing = max(15, self.size // 17)
        for i, j in ti.ndrange(self.size, self.size):
            if i % spacing == spacing // 2 and j % spacing == spacing // 2:
                s = max(2, self.size // 256)
                for di, dj in ti.ndrange((-s, s+1), (-s, s+1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        self.U[ni, nj] = 1.0


class RDLeniaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RD Lenia (论文严格实现)")
        self.sim = RDLeniaSimulation(512)
        self.viewer = SimulationViewer(self.sim)
        self.viewer.set_display_modes(['U', 'V_sum'])
        self.setCentralWidget(self.viewer)
        self._add_rd_lenia_params()
        self.setGeometry(100, 100, 920, 650)

    def _add_rd_lenia_params(self):
        panel = QWidget()
        layout = QVBoxLayout()

        params_group = QGroupBox("RD Lenia 参数 (论文)")
        p_layout = QVBoxLayout()

        self.m_slider, self.m_label = self._make_slider(
            "生长均值 (m):", 0.05, 0.35, 0.15, 10000, 'm'
        )
        self.s_slider, self.s_label = self._make_slider(
            "生长宽度 (s):", 0.005, 0.05, 0.015, 100000, 's'
        )
        self.eps_slider, self.eps_label = self._make_slider(
            "时间尺度 (ε):", 0.001, 0.02, 0.005, 100000, 'epsilon'
        )
        self.mu_slider, self.mu_label = self._make_slider(
            "耦合强度 (μ):", 0.01, 0.3, 0.1, 10000, 'mu'
        )

        p_layout.addWidget(self.m_slider)
        p_layout.addWidget(self.s_slider)
        p_layout.addWidget(self.eps_slider)
        p_layout.addWidget(self.mu_slider)

        params_group.setLayout(p_layout)
        layout.addWidget(params_group)

        # 显示设置
        disp_group = QGroupBox("显示设置")
        d_layout = QVBoxLayout()
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("颜色:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["默认", "反色"])
        self.color_combo.currentTextChanged.connect(self._change_color)
        color_layout.addWidget(self.color_combo)
        d_layout.addLayout(color_layout)

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("图像尺寸:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["128x128", "256x256", "512x512", "1024x1024"])
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
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["随机", "中心", "边缘", "点阵"])
        pat_layout.addWidget(self.pattern_combo)
        i_layout.addLayout(pat_layout)

        init_group.setLayout(i_layout)
        layout.addWidget(init_group)

        panel.setLayout(layout)
        self.viewer.param_panel.insertWidget(0, panel)

        # 自定义重置
        def custom_reset():
            mapping = {"随机": "random", "中心": "center", "边缘": "edges", "点阵": "spots"}
            self.sim.reset(mapping[self.pattern_combo.currentText()])
            self.viewer.gl_widget.update()
        self.viewer.reset_sim = custom_reset

    def _make_slider(self, label, vmin, vmax, vdef, scale, attr):
        w = QWidget()
        lay = QHBoxLayout()
        lbl = QLabel(f"{label} {vdef:.4f}")
        lbl.setMinimumWidth(120)
        s = QSlider(Qt.Horizontal)
        s.setMinimum(int(vmin * scale))
        s.setMaximum(int(vmax * scale))
        s.setValue(int(vdef * scale))

        def update(val):
            scaled = val / scale
            lbl.setText(f"{label} {scaled:.4f}")
            setattr(self.sim, attr, scaled)

        s.valueChanged.connect(update)
        lay.addWidget(lbl)
        lay.addWidget(s)
        w.setLayout(lay)
        return w, lbl

    def _change_color(self, text):
        self.sim.color_scheme = 'default' if text == '默认' else 'inverse'
        self.viewer.gl_widget.update()

    def _change_size(self, text):
        sizes = {"128x128": 128, "256x256": 256, "512x512": 512, "1024x1024": 1024}
        if text in sizes:
            ns = sizes[text]
            m, s, eps, mu, cs = self.sim.m, self.sim.s, self.sim.epsilon, self.sim.mu, self.sim.color_scheme
            self.sim = RDLeniaSimulation(ns)
            self.sim.m, self.sim.s, self.sim.epsilon, self.sim.mu = m, s, eps, mu
            self.sim.color_scheme = cs
            self.viewer.sim = self.sim
            self.viewer.reset_sim()

    def closeEvent(self, event):
        self.viewer.timer.stop()
        ti.reset()
        event.accept()


if __name__ == "__main__":
    setup_gl_format()
    app = QApplication(sys.argv)
    window = RDLeniaWindow()
    window.show()
    sys.exit(app.exec())