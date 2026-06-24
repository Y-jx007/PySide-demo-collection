from custom_import import *

# 初始化Taichi，使用GPU后端
ti.init(arch=ti.gpu, default_fp=ti.f32)


@ti.data_oriented
class ReactionDiffusionSimulation:
    def __init__(self, size=1024):
        self.size = size
        self.width = size
        self.height = size
        
        # 使用Taichi字段替代numpy数组
        self.U = ti.field(dtype=ti.f32, shape=(size, size))
        self.V = ti.field(dtype=ti.f32, shape=(size, size))
        self.U_temp = ti.field(dtype=ti.f32, shape=(size, size))
        self.V_temp = ti.field(dtype=ti.f32, shape=(size, size))
        
        # 参数（使用更适合图灵斑图的默认值）
        self.Du = 0.16
        self.Dv = 0.08
        self.f = 0.04
        self.k = 0.06
        
        self.reset()
    
    @ti.kernel
    def compute_laplacian(self, field: ti.template(), result: ti.template()):
        """使用5点离散拉普拉斯算子 - Taichi GPU并行版本"""
        for i, j in ti.ndrange((1, self.size-1), (1, self.size-1)):
            result[i, j] = (field[i-1, j] + field[i+1, j] + 
                            field[i, j-1] + field[i, j+1] - 
                            4 * field[i, j])
    
    @ti.kernel
    def update_step(self, Du: ti.f32, Dv: ti.f32, f: ti.f32, k: ti.f32):
        """更新反应扩散方程 - Taichi GPU并行版本"""
        for i, j in ti.ndrange((1, self.size-1), (1, self.size-1)):
            # Gray-Scott模型
            reaction = self.U[i, j] * self.V[i, j] * self.V[i, j]
            
            # 更新方程
            U_new = self.U[i, j] + Du * self.U_temp[i, j] - reaction + f * (1 - self.U[i, j])
            V_new = self.V[i, j] + Dv * self.V_temp[i, j] + reaction - (f + k) * self.V[i, j]
            
            # 限制值范围
            self.U[i, j] = max(0.0, min(1.0, U_new))
            self.V[i, j] = max(0.0, min(1.0, V_new))
    
    def step(self):
        """执行一步模拟"""
        # 计算拉普拉斯
        self.compute_laplacian(self.U, self.U_temp)
        self.compute_laplacian(self.V, self.V_temp)
        
        # 更新浓度场
        self.update_step(self.Du, self.Dv, self.f, self.k)
    
    def get_visualization(self, mode='V', color_scheme='default'):
        """获取可视化图像"""
        # 将Taichi字段转换为numpy数组进行可视化
        if mode == 'V':
            field_np = self.V.to_numpy()
        elif mode == 'U':
            field_np = self.U.to_numpy()
        else:  # UV混合
            U_np = self.U.to_numpy()
            V_np = self.V.to_numpy()
            field_np = np.stack([U_np, V_np, np.zeros_like(U_np)], axis=-1)
            field_np = field_np * 255
            field_np = np.clip(field_np, 0, 255).astype(np.uint8)
            return field_np
        
        # 转换为图像
        field_visual = field_np * 255
        field_visual = np.clip(field_visual, 0, 255).astype(np.uint8)
        
        if color_scheme == 'inverse':
            field_visual = 255 - field_visual
        
        field_rgb = np.stack([field_visual, field_visual, field_visual], axis=-1)
        return field_rgb
    
    def reset(self, pattern='random'):
        """重置模拟"""
        # 初始化字段
        self.U.fill(1.0)
        self.V.fill(0.0)
        
        # 创建初始模式
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
        """初始化中心模式"""
        center = self.size // 2
        size = self.size // 20  # 根据尺寸调整初始模式大小
        for i, j in ti.ndrange(self.size, self.size):
            if (ti.abs(i - center) < size and 
                ti.abs(j - center) < size):
                self.V[i, j] = 1.0
    
    @ti.kernel
    def _init_random_pattern(self):
        """初始化随机模式"""
        for i, j in ti.ndrange(self.size, self.size):
            # 根据尺寸调整随机点密度
            border = self.size // 20
            density = 0.001 * (512 / self.size)  # 根据尺寸调整密度
            if (border <= i < self.size - border and 
                border <= j < self.size - border and 
                ti.random() < density):
                size = max(2, self.size // 256)  # 根据尺寸调整点大小
                for di, dj in ti.ndrange((-size, size+1), (-size, size+1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        self.V[ni, nj] = 1.0
    
    @ti.kernel
    def _init_edges_pattern(self):
        """初始化边缘模式"""
        border_size = max(3, self.size // 170)  # 根据尺寸调整边框大小
        for i, j in ti.ndrange(self.size, self.size):
            if (i < border_size or i >= self.size - border_size or
                j < border_size or j >= self.size - border_size):
                self.V[i, j] = 1.0
    
    @ti.kernel
    def _init_spots_pattern(self):
        """初始化点阵模式"""
        spacing = max(15, self.size // 17)  # 根据尺寸调整间距
        for i, j in ti.ndrange(self.size, self.size):
            if (i % spacing == spacing//2 and 
                j % spacing == spacing//2):
                size = max(2, self.size // 256)  # 根据尺寸调整点大小
                for di, dj in ti.ndrange((-size, size+1), (-size, size+1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        self.V[ni, nj] = 1.0


class ReactionDiffusionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.simulation = ReactionDiffusionSimulation(512)  # 默认512x512
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.is_playing = True
        self.visualization_mode = 'V'
        self.color_scheme = 'default'
        self.steps_per_frame = 60
        
        self.init_ui()
    
    def init_ui(self):
        main_layout = QHBoxLayout()
        
        # 左侧参数面板
        left_panel = QVBoxLayout()
        
        # 参数控制组
        params_group = QGroupBox("模拟参数")
        params_layout = QVBoxLayout()
        
        # 参数滑块
        self.f_slider, self.f_label = self.create_slider_with_label("供给率 (f):", 0.01, 0.08, 0.04, 10000)
        self.k_slider, self.k_label = self.create_slider_with_label("去除率 (k):", 0.04, 0.08, 0.06, 10000)
        self.Du_slider, self.Du_label = self.create_slider_with_label("U扩散系数:", 0.1, 0.3, 0.16, 1000)
        self.Dv_slider, self.Dv_label = self.create_slider_with_label("V扩散系数:", 0.04, 0.12, 0.08, 1000)
        
        params_layout.addWidget(self.f_slider)
        params_layout.addWidget(self.k_slider)
        params_layout.addWidget(self.Du_slider)
        params_layout.addWidget(self.Dv_slider)
        
        # 手动输入参数
        manual_input_layout = QHBoxLayout()
        manual_input_layout.addWidget(QLabel("手动输入:"))
        
        self.manual_f_input = QLineEdit()
        self.manual_f_input.setPlaceholderText("f值")
        self.manual_f_input.setMaximumWidth(60)
        manual_input_layout.addWidget(self.manual_f_input)
        
        self.manual_k_input = QLineEdit()
        self.manual_k_input.setPlaceholderText("k值")
        self.manual_k_input.setMaximumWidth(60)
        manual_input_layout.addWidget(self.manual_k_input)
        
        self.apply_manual_button = QPushButton("应用")
        self.apply_manual_button.setMaximumWidth(60)
        self.apply_manual_button.clicked.connect(self.apply_manual_parameters)
        manual_input_layout.addWidget(self.apply_manual_button)
        
        params_layout.addLayout(manual_input_layout)
        
        params_group.setLayout(params_layout)
        left_panel.addWidget(params_group)
        
        # 显示设置组
        display_group = QGroupBox("显示设置")
        display_layout = QVBoxLayout()
        
        # 可视化模式选择
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("显示模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["物质V", "物质U", "UV混合"])
        self.mode_combo.currentTextChanged.connect(self.change_visualization_mode)
        mode_layout.addWidget(self.mode_combo)
        display_layout.addLayout(mode_layout)
        
        # 颜色方案
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("颜色:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["默认", "反色"])
        self.color_combo.currentTextChanged.connect(self.change_color_scheme)
        color_layout.addWidget(self.color_combo)
        display_layout.addLayout(color_layout)
        
        # 图像尺寸选择
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("图像尺寸:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["128x128", "256x256", "512x512", "1024x1024"])
        self.size_combo.setCurrentText("1024x1024")
        self.size_combo.currentTextChanged.connect(self.change_size)
        size_layout.addWidget(self.size_combo)
        display_layout.addLayout(size_layout)
        
        display_group.setLayout(display_layout)
        left_panel.addWidget(display_group)
        
        # 初始条件组
        init_group = QGroupBox("初始条件")
        init_layout = QVBoxLayout()
        
        # 初始模式选择
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("初始模式:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["随机", "中心", "边缘", "点阵"])
        pattern_layout.addWidget(self.pattern_combo)
        init_layout.addLayout(pattern_layout)
        
        # 预设参数
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("预设:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["迷宫","条纹","斑点","蜂巢","云雾"])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_combo)
        init_layout.addLayout(preset_layout)
        
        init_group.setLayout(init_layout)
        left_panel.addWidget(init_group)
        
        # 控制组
        control_group = QGroupBox("控制")
        control_layout = QVBoxLayout()
        
        # 第一行按钮
        button_row1 = QHBoxLayout()
        
        self.play_button = QPushButton("暂停")
        self.play_button.clicked.connect(self.toggle_play)
        button_row1.addWidget(self.play_button)
        
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_simulation)
        button_row1.addWidget(self.reset_button)
        
        control_layout.addLayout(button_row1)
        
        # 第二行按钮
        button_row2 = QHBoxLayout()
        
        self.step_button = QPushButton("单步")
        self.step_button.clicked.connect(self.step_simulation)
        button_row2.addWidget(self.step_button)
        
        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.save_image)
        button_row2.addWidget(self.save_button)
        
        control_layout.addLayout(button_row2)
        
        # 每帧步数设置
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("每帧步数:"))
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setMinimum(1)
        self.steps_slider.setMaximum(100)
        self.steps_slider.setValue(60)
        self.steps_slider.valueChanged.connect(self.change_steps_per_frame)
        self.steps_label = QLabel("60")
        steps_layout.addWidget(self.steps_slider)
        steps_layout.addWidget(self.steps_label)
        control_layout.addLayout(steps_layout)
        
        control_group.setLayout(control_layout)
        left_panel.addWidget(control_group)
        
        # 添加左侧面板到主布局
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(300)
        left_widget.setFixedHeight(630)
        main_layout.addWidget(left_widget)
        
        # 右侧图像显示区域
        self.image_label = QLabel()
        self.image_label.setFixedSize(600, 600)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.update_display()
        main_layout.addWidget(self.image_label)
        
        self.setLayout(main_layout)
        
        # 启动定时器
        self.timer.start(30)  # 约33 FPS
    
    def create_slider_with_label(self, label_text, min_val, max_val, default_val, scale=100):
        """创建参数滑块和标签"""
        layout = QHBoxLayout()
        
        label = QLabel(f"{label_text} {default_val:.4f}")
        label.setMinimumWidth(120)
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * scale))
        slider.setMaximum(int(max_val * scale))
        slider.setValue(int(default_val * scale))
        
        def update_value(value):
            scaled_value = value / scale
            label.setText(f"{label_text} {scaled_value:.4f}")
            self.update_parameter(label_text.split()[0].lower(), scaled_value)
        
        slider.valueChanged.connect(update_value)
        
        layout.addWidget(label)
        layout.addWidget(slider)
        
        container = QWidget()
        container.setLayout(layout)
        
        return container, label
    
    def update_parameter(self, param_name, value):
        """更新模拟参数"""
        if param_name == "供给率":
            self.simulation.f = value
        elif param_name == "去除率":
            self.simulation.k = value
        elif param_name == "u扩散系数":
            self.simulation.Du = value
        elif param_name == "v扩散系数":
            self.simulation.Dv = value
    
    def toggle_play(self):
        """切换播放状态"""
        self.is_playing = not self.is_playing
        self.play_button.setText("暂停" if self.is_playing else "开始")
    
    def reset_simulation(self):
        """重置模拟"""
        pattern_map = {"随机": "random", "中心": "center", "边缘": "edges", "点阵": "spots"}
        pattern = pattern_map[self.pattern_combo.currentText()]
        self.simulation.reset(pattern)
        self.update_display()
    
    def step_simulation(self):
        """单步模拟"""
        self.simulation.step()
        self.update_display()
    
    def change_visualization_mode(self, mode_text):
        """更改可视化模式"""
        mode_map = {"物质V": "V", "物质U": "U", "UV混合": "UV"}
        self.visualization_mode = mode_map[mode_text]
        self.update_display()
    
    def change_color_scheme(self, scheme_text):
        """更改颜色方案"""
        scheme_map = {"默认": "default", "反色": "inverse"}
        self.color_scheme = scheme_map[scheme_text]
        self.update_display()
    
    def change_size(self, size_text):
        """更改图像尺寸"""
        size_map = {
            "128x128": 128,
            "256x256": 256,
            "512x512": 512,
            "1024x1024": 1024
        }
        if size_text in size_map:
            new_size = size_map[size_text]
            current_f = self.simulation.f
            current_k = self.simulation.k
            current_Du = self.simulation.Du
            current_Dv = self.simulation.Dv
            
            self.simulation = ReactionDiffusionSimulation(new_size)
            
            self.simulation.f = current_f
            self.simulation.k = current_k
            self.simulation.Du = current_Du
            self.simulation.Dv = current_Dv
            
            self.reset_simulation()
    
    def change_steps_per_frame(self, value):
        """更改每帧计算的步数"""
        self.steps_per_frame = value
        self.steps_label.setText(str(value))
    
    def apply_preset(self, preset_name):
        """应用预设参数"""
        presets = {
            "迷宫": {"f": 0.0400, "k": 0.0600},
            "条纹": {"f": 0.0550, "k": 0.0650},
            "斑点": {"f": 0.0400, "k": 0.0650},
            "蜂巢": {"f": 0.0300, "k": 0.0550},  
            "云雾": {"f": 0.0160, "k": 0.0450},  
        }
        
        if preset_name in presets:
            preset = presets[preset_name]
            self.simulation.f = preset["f"]
            self.simulation.k = preset["k"]
            
            self.f_slider.layout().itemAt(1).widget().setValue(int(preset["f"] * 10000))
            self.k_slider.layout().itemAt(1).widget().setValue(int(preset["k"] * 10000))
            self.f_label.setText(f"供给率 (f): {preset['f']:.4f}")
            self.k_label.setText(f"去除率 (k): {preset['k']:.4f}")
            
            self.reset_simulation()
    
    def apply_manual_parameters(self):
        """应用手动输入的参数"""
        try:
            f_text = self.manual_f_input.text().strip()
            k_text = self.manual_k_input.text().strip()
            
            if f_text:
                f_value = float(f_text)
                if 0.01 <= f_value <= 0.08:
                    self.simulation.f = f_value
                    self.f_slider.layout().itemAt(1).widget().setValue(int(f_value * 10000))
                    self.f_label.setText(f"供给率 (f): {f_value:.4f}")
                else:
                    QMessageBox.warning(self, "输入错误", "f值必须在0.01到0.08之间")
                    return
            
            if k_text:
                k_value = float(k_text)
                if 0.04 <= k_value <= 0.08:
                    self.simulation.k = k_value
                    self.k_slider.layout().itemAt(1).widget().setValue(int(k_value * 10000))
                    self.k_label.setText(f"去除率 (k): {k_value:.4f}")
                else:
                    QMessageBox.warning(self, "输入错误", "k值必须在0.04到0.08之间")
                    return
            
            self.reset_simulation()
            
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字")
    
    def save_image(self):
        """保存当前图像到文件"""
        default_filename = f"turing_pattern_f{self.simulation.f:.4f}_k{self.simulation.k:.4f}.png"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", default_filename, "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)"
        )
        
        if file_path:
            image_data = self.simulation.get_visualization(self.visualization_mode, self.color_scheme)
            height, width, channel = image_data.shape
            
            q_image = QImage(image_data.data, width, height, 3 * width, QImage.Format_RGB888)
            
            if q_image.save(file_path):
                QMessageBox.information(self, "保存成功", f"图像已保存到:\n{file_path}")
            else:
                QMessageBox.warning(self, "保存失败", "无法保存图像，请检查文件路径和权限")
    
    def update_simulation(self):
        """定时器更新模拟"""
        if self.is_playing:
            for _ in range(self.steps_per_frame):
                self.simulation.step()
            self.update_display()
    
    def update_display(self):
        """更新显示"""
        image_data = self.simulation.get_visualization(self.visualization_mode, self.color_scheme)
        height, width, channel = image_data.shape
        
        q_image = QImage(image_data.data, width, height, 3 * width, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

class ReactionDiffusionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("反应扩散方程模拟 - 图灵斑图生成器 (Taichi GPU加速)")
        self.setGeometry(100, 100, 900, 650)
        
        central_widget = ReactionDiffusionWidget()
        self.setCentralWidget(central_widget)

    def closeEvent(self, event):
                self.centralWidget().timer.stop()
                ti.reset()
                event.accept()

def main():
    app = QApplication(sys.argv)
    window = ReactionDiffusionWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()