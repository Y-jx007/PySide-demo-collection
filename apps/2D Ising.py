from custom_import import *

# 初始化Taichi
ti.init(arch=ti.gpu)  # 使用GPU加速
class OrderParameterWidget(QWidget):
    """序参量（磁化强度）和能量演化曲线显示组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setMaximumHeight(250)
        self.magnetization_history = []
        self.energy_history = []
        self.max_history_length = 300
        
    def add_data(self, magnetization, energy):
        self.magnetization_history.append(magnetization)
        self.energy_history.append(energy)
        
        if len(self.magnetization_history) > self.max_history_length:
            self.magnetization_history.pop(0)
            self.energy_history.pop(0)
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(240, 240, 240))
        
        if len(self.magnetization_history) < 2:
            return
            
        margin = 30
        plot_rect = self.rect().adjusted(margin, margin, -margin, -margin)
        
        # 绘制坐标轴
        painter.setPen(QPen(Qt.black, 1))
        painter.drawLine(plot_rect.left(), plot_rect.center().y(), 
                        plot_rect.right(), plot_rect.center().y())
        painter.drawLine(plot_rect.left(), plot_rect.top(),
                        plot_rect.left(), plot_rect.bottom())
        
        # 绘制磁化强度曲线（蓝色）
        painter.setPen(QPen(QColor(0, 100, 200), 2))
        x_step = plot_rect.width() / (len(self.magnetization_history) - 1)
        
        for i in range(len(self.magnetization_history) - 1):
            x1 = plot_rect.left() + i * x_step
            y1 = plot_rect.center().y() - self.magnetization_history[i] * (plot_rect.height() / 4)
            x2 = plot_rect.left() + (i + 1) * x_step
            y2 = plot_rect.center().y() - self.magnetization_history[i + 1] * (plot_rect.height() / 4)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # 绘制能量曲线（红色）
        painter.setPen(QPen(QColor(200, 0, 0), 2))
        for i in range(len(self.energy_history) - 1):
            x1 = plot_rect.left() + i * x_step
            y1 = plot_rect.center().y() - self.energy_history[i] * (plot_rect.height() / 4)
            x2 = plot_rect.left() + (i + 1) * x_step
            y2 = plot_rect.center().y() - self.energy_history[i + 1] * (plot_rect.height() / 4)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # 显示当前值
        current_mag = self.magnetization_history[-1]
        current_energy = self.energy_history[-1]
        painter.setPen(QPen(Qt.black))
        painter.drawText(60, 25, f"M: {current_mag:.4f}")
        painter.drawText(180, 25, f"E: {current_energy:.4f}")
@ti.data_oriented
class IsingModel:
    def __init__(self, size=1024):
        self.size = size
        self.temperature = 2.269
        self.steps_per_frame = 1
        self.is_paused = False
        self.use_wolff = False
        self.use_checkerboard = True
        self.cluster_probability = 1.0 - np.exp(-2.0 / self.temperature)
        
        # 每次尝试次数默认为size²/4
        total_spins = self.size * self.size
        self.updates_per_step = total_spins // 4
        
        # Taichi字段
        self.spins = ti.field(dtype=ti.i8, shape=(size, size))
        self.temperature_field = ti.field(dtype=ti.f32, shape=())
        self.updates_per_step_field = ti.field(dtype=ti.i32, shape=())
        
        # 纹理字段 - 用于OpenGL渲染，改为RGB格式
        self.texture_field = ti.field(dtype=ti.f32, shape=(size, size, 3))
        
        # 初始化字段
        self.temperature_field[None] = self.temperature
        self.updates_per_step_field[None] = self.updates_per_step
        
        self.reset_spins()
    
    @ti.kernel
    def update_texture(self):
        """将自旋数据转换为RGB纹理数据"""
        for i, j in self.spins:
            # 将-1,1映射为灰度值 (0.0为黑色，1.0为白色)
            value = (ti.cast(self.spins[i, j], ti.f32) + 1.0) * 0.5
            # 设置为RGB相同的值，形成灰度图像
            self.texture_field[i, j, 0] = value  # R
            self.texture_field[i, j, 1] = value  # G  
            self.texture_field[i, j, 2] = value  # B
    
    @ti.kernel
    def taichi_checkerboard_step(self, mask_type: ti.i32): # type: ignore
        """Taichi优化的棋盘格更新"""
        grid_size = self.spins.shape[0]
        temperature = self.temperature_field[None]
        updates_per_step=self.updates_per_step_field[None]
        half_cells = (grid_size * grid_size + 1) // 2  # 近似一半
    
        # 每个棋盘格位置以概率p被选中更新
        p = ti.min(1,updates_per_step / half_cells)
        
        for i, j in ti.ndrange(grid_size, grid_size):
            # 根据mask_type选择棋盘格
            if (mask_type == 0 and (i + j) % 2 == 0) or (mask_type == 1 and (i + j) % 2 == 1):
                if ti.random()<p:
                    # 计算能量变化
                    up = self.spins[(i - 1) % grid_size, j]
                    down = self.spins[(i + 1) % grid_size, j]
                    left = self.spins[i, (j - 1) % grid_size]
                    right = self.spins[i, (j + 1) % grid_size]
                    
                    delta_E = 2.0 * self.spins[i, j] * (up + down + left + right)
                    
                    # Metropolis准则
                    if delta_E <= 0.0 or ti.random() < ti.exp(-delta_E / temperature):
                        self.spins[i, j] = -self.spins[i, j]
    
    @ti.kernel
    def calculate_magnetization_taichi(self) -> ti.f32: # type: ignore
        total = 0.0
        for i, j in self.spins:
            total += ti.cast(self.spins[i, j], ti.f32)
        return total / (self.spins.shape[0] * self.spins.shape[1])
    
    @ti.kernel
    def calculate_energy_taichi(self) -> ti.f32: # type: ignore
        grid_size = self.spins.shape[0]
        total_energy = 0.0
        
        for i, j in ti.ndrange(grid_size, grid_size):
            up = self.spins[(i - 1) % grid_size, j]
            down = self.spins[(i + 1) % grid_size, j]
            left = self.spins[i, (j - 1) % grid_size]
            right = self.spins[i, (j + 1) % grid_size]
            
            energy_per_spin = -self.spins[i, j] * (up + down + left + right) / 4.0
            total_energy += energy_per_spin
            
        return total_energy / (grid_size * grid_size) + 1.0
    @ti.kernel
    def taichi_monte_carlo_step(self):
        grid_size = self.spins.shape[0]
        temperature = self.temperature_field[None]
        updates_per_step = self.updates_per_step_field[None]
        
        ti.loop_config(serialize=True)
        for k in range(updates_per_step):
            i = ti.cast(ti.floor(ti.random() * grid_size), ti.i32)
            j = ti.cast(ti.floor(ti.random() * grid_size), ti.i32)
            
            up = self.spins[(i - 1) % grid_size, j]
            down = self.spins[(i + 1) % grid_size, j]
            left = self.spins[i, (j - 1) % grid_size]
            right = self.spins[i, (j + 1) % grid_size]
            
            delta_E = 2.0 * self.spins[i, j] * (up + down + left + right)
            
            if delta_E <= 0.0 or ti.random() < ti.exp(-delta_E / temperature):
                self.spins[i, j] = -self.spins[i, j]
    
    def cpu_wolff_step(self):
        """在CPU上执行Wolff算法，避免GPU递归问题"""
        i_start = np.random.randint(0, self.size)
        j_start = np.random.randint(0, self.size)
        cluster_probability = 1.0 - np.exp(-2.0 / self.temperature)
        
        # 从GPU获取当前自旋状态
        spins_np = self.spins.to_numpy()
        
        cluster = set()
        stack = [(i_start, j_start)]
        cluster_sign = spins_np[i_start, j_start]
        
        while stack:
            i, j = stack.pop()
            if (i, j) in cluster:
                continue
            cluster.add((i, j))
            
            neighbors = [
                ((i-1) % self.size, j),
                ((i+1) % self.size, j),
                (i, (j-1) % self.size),
                (i, (j+1) % self.size)
            ]
            
            for ni, nj in neighbors:
                if (ni, nj) not in cluster:
                    neighbor_spin = spins_np[ni, nj]
                    if neighbor_spin == cluster_sign and np.random.random() < cluster_probability:
                        stack.append((ni, nj))
        
        # 翻转整个集群
        for i, j in cluster:
            spins_np[i, j] = -spins_np[i, j]
        
        # 将结果同步回GPU
        self.spins.from_numpy(spins_np)
            
    def set_temperature(self, temperature):
        self.temperature = temperature
        self.temperature_field[None] = temperature
        
    def set_steps_per_frame(self, steps):
        self.steps_per_frame = steps
        
    def set_updates_per_step(self, updates):
        self.updates_per_step = updates
        self.updates_per_step_field[None] = updates
        
    def reset_spins(self):
        spins_np = np.random.choice([-1, 1], size=(self.size, self.size)).astype(np.int8)
        self.spins.from_numpy(spins_np)
class IsingModelGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_size = 1024
        self.simulation = IsingModel(self.grid_size)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_model)
        self.timer.start(0)
        self.setMinimumSize(540, 540)
        
        self.last_checkerboard = 0
        self.texture_id = None
        
    def initializeGL(self):
        """初始化OpenGL"""
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
    def resizeGL(self, w, h):
        """调整OpenGL视口"""
        glViewport(0, 0, w, h)
        
    def paintGL(self):
        """绘制OpenGL场景"""
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        
        # 更新纹理数据
        self.simulation.update_texture()
        texture_data = self.simulation.texture_field.to_numpy()
        
        # 上传RGB纹理到GPU
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.grid_size, self.grid_size, 0, 
                     GL_RGB, GL_FLOAT, texture_data)
        
        # 绘制全屏四边形
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(-1, 1)
        glEnd()
        
    def set_grid_size(self, size):
        """更改网格尺寸"""
        if size != self.grid_size:
            # 保存当前参数
            current_temperature = self.simulation.temperature
            current_is_paused = self.simulation.is_paused
            current_use_wolff = self.simulation.use_wolff
            current_use_checkerboard = self.simulation.use_checkerboard
            current_steps_per_frame = self.simulation.steps_per_frame
            
            # 创建新的模拟实例
            self.simulation = IsingModel(size)
            
            # 恢复参数
            self.simulation.temperature = current_temperature
            self.simulation.temperature_field[None] = current_temperature
            self.simulation.is_paused = current_is_paused
            self.simulation.use_wolff = current_use_wolff
            self.simulation.use_checkerboard = current_use_checkerboard
            self.simulation.steps_per_frame = current_steps_per_frame
            # updates_per_step使用新实例的默认值（size²/4）
            
            self.grid_size = size
            
            # 通知主窗口更新spinbox显示
            if hasattr(self, 'updates_callback'):
                self.updates_callback(self.simulation.updates_per_step)
            
            self.update()
        
    def calculate_magnetization(self):
        return self.simulation.calculate_magnetization_taichi()
    
    def calculate_energy(self):
        return self.simulation.calculate_energy_taichi()
    def update_model(self):
        if not self.simulation.is_paused:
            for _ in range(self.simulation.steps_per_frame):
                if self.simulation.use_wolff:
                    # 使用CPU版本的Wolff算法
                    self.simulation.cpu_wolff_step()
                else:
                    if self.simulation.use_checkerboard:
                        # 交替使用两种棋盘格掩码
                        current_mask = 1 - self.last_checkerboard
                        self.last_checkerboard = current_mask
                        self.simulation.taichi_checkerboard_step(current_mask)
                    else:
                        self.simulation.taichi_monte_carlo_step()
            
            magnetization = self.calculate_magnetization()
            energy = self.calculate_energy()
            if hasattr(self, 'data_callback'):
                self.data_callback(magnetization, energy)
            self.update()
    def set_data_callback(self, callback):
        self.data_callback = callback
            
    def set_temperature(self, temperature):
        self.simulation.set_temperature(temperature)
        
    def set_steps_per_frame(self, steps):
        self.simulation.set_steps_per_frame(steps)
        
    def set_updates_per_step(self, updates):
        self.simulation.set_updates_per_step(updates)
        
    def set_use_checkerboard(self, use_checkerboard):
        self.simulation.use_checkerboard = use_checkerboard
        
    def reset_spins(self):
        self.simulation.reset_spins()
        
    def toggle_pause(self):
        self.simulation.is_paused = not self.simulation.is_paused
        
    def set_use_wolff(self, use_wolff):
        self.simulation.use_wolff = use_wolff
        
    def set_updates_callback(self, callback):
        self.updates_callback = callback
        
    @property
    def is_paused(self):
        return self.simulation.is_paused
        
    @property
    def updates_per_step(self):
        return self.simulation.updates_per_step
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("二维伊辛模型模拟 (Taichi加速 + OpenGL渲染)")
        self.setGeometry(50, 50, 800, 540)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧参数面板
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QGridLayout(left_panel)
        
        # 分辨率控制组
        res_group = QGroupBox("网格分辨率")
        res_layout = QVBoxLayout(res_group)
        self.grid_size_spinbox = QSpinBox()
        self.grid_size_spinbox.setRange(64, 2048)
        self.grid_size_spinbox.setValue(1024)
        self.grid_size_spinbox.valueChanged.connect(self.on_grid_size_changed)
        res_layout.addWidget(self.grid_size_spinbox)
        
        preset_layout = QHBoxLayout()
        for size in [512, 1024, 2048]:
            btn = QPushButton(str(size))
            btn.clicked.connect(lambda checked, s=size: self.set_preset_grid_size(s))
            preset_layout.addWidget(btn)
        res_layout.addLayout(preset_layout)
        left_layout.addWidget(res_group, 0, 0)
        
        # 温度控制组
        temp_group = QGroupBox("温度控制")
        temp_layout = QVBoxLayout(temp_group)
        self.temp_spinbox = QDoubleSpinBox()
        self.temp_spinbox.setRange(0.0001, 1000.0)
        self.temp_spinbox.setSingleStep(0.0001)
        self.temp_spinbox.setDecimals(4)
        self.temp_spinbox.setValue(2.269)
        self.temp_spinbox.setKeyboardTracking(False)
        self.temp_spinbox.valueChanged.connect(self.on_temperature_changed)
        temp_layout.addWidget(self.temp_spinbox)
        
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(1)
        self.temp_slider.setMaximum(100000)
        self.temp_slider.setValue(int(2.269 * 10000))
        self.temp_slider.valueChanged.connect(self.on_temperature_slider_changed)
        temp_layout.addWidget(self.temp_slider)
        
        self.temp_value_label = QLabel(f"当前: {2.269:.4f}")
        temp_layout.addWidget(self.temp_value_label)
        left_layout.addWidget(temp_group, 1, 0)
        
        # 算法控制组
        algo_group = QGroupBox("算法设置")
        algo_layout = QVBoxLayout(algo_group)
        self.wolff_checkbox = QCheckBox("Wolff算法")
        self.wolff_checkbox.toggled.connect(self.on_wolff_changed)
        algo_layout.addWidget(self.wolff_checkbox)
        
        self.checkerboard_checkbox = QCheckBox("棋盘格优化")
        self.checkerboard_checkbox.setChecked(True)
        self.checkerboard_checkbox.toggled.connect(self.on_checkerboard_changed)
        algo_layout.addWidget(self.checkerboard_checkbox)
        
        self.performance_label = QLabel("性能: 棋盘格优化")
        algo_layout.addWidget(self.performance_label)
        left_layout.addWidget(algo_group, 0, 1)
        
        # 性能控制组
        perf_group = QGroupBox("性能控制")
        perf_layout = QVBoxLayout(perf_group)
        
        perf_layout.addWidget(QLabel("每帧更新次数:"))
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setRange(1, 1000)
        self.steps_spinbox.setValue(1)
        self.steps_spinbox.valueChanged.connect(self.on_steps_changed)
        perf_layout.addWidget(self.steps_spinbox)
        
        perf_layout.addWidget(QLabel("每次尝试次数:"))
        self.updates_spinbox = QSpinBox()
        self.updates_spinbox.setRange(1, 20000000)
        self.widget = IsingModelGLWidget()
        self.updates_spinbox.setValue(self.widget.updates_per_step)
        self.updates_spinbox.valueChanged.connect(self.on_updates_changed)
        perf_layout.addWidget(self.updates_spinbox)
        left_layout.addWidget(perf_group, 1, 1)
        
        # 控制按钮组
        control_group = QGroupBox("状态控制")
        control_layout = QVBoxLayout(control_group)
        
        button_layout = QHBoxLayout()
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.pause_button)
        
        self.reset_button = QPushButton("随机重置")
        self.reset_button.clicked.connect(self.reset_spins)
        button_layout.addWidget(self.reset_button)
        control_layout.addLayout(button_layout)
        
        button_layout2 = QHBoxLayout()
        self.ferro_button = QPushButton("铁磁")
        self.ferro_button.clicked.connect(self.set_ferromagnetic)
        button_layout2.addWidget(self.ferro_button)
        
        self.antiferro_button = QPushButton("反铁磁")
        self.antiferro_button.clicked.connect(self.set_antiferromagnetic)
        button_layout2.addWidget(self.antiferro_button)
        control_layout.addLayout(button_layout2)
        left_layout.addWidget(control_group, 2, 1)
        
        # 序参量曲线显示
        self.order_parameter_widget = OrderParameterWidget()
        left_layout.addWidget(self.order_parameter_widget, 3, 0, 1, 2)
        
        # 右侧窗口
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.widget.set_updates_callback(self.on_updates_changed_by_widget)
        self.widget.set_data_callback(self.order_parameter_widget.add_data)
        right_layout.addWidget(self.widget)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)
        
        self.current_temperature = 2.269
        
    def set_preset_grid_size(self, size):
        self.grid_size_spinbox.setValue(size)
        
    def on_grid_size_changed(self):
        self.widget.set_grid_size(self.grid_size_spinbox.value())
        # 重置序参量曲线
        self.order_parameter_widget.magnetization_history.clear()
        self.order_parameter_widget.energy_history.clear()
        self.order_parameter_widget.update()
        
    def on_temperature_changed(self, value):
        self.current_temperature = value
        self.temp_value_label.setText(f"当前: {value:.4f}")
        self.widget.set_temperature(value)
        self.temp_slider.setValue(int(value * 10000))
        
    def on_temperature_slider_changed(self, value):
        temperature = value / 10000.0
        self.current_temperature = temperature
        self.temp_spinbox.setValue(temperature)
        self.temp_value_label.setText(f"当前: {temperature:.4f}")
        self.widget.set_temperature(temperature)
        
    def on_steps_changed(self, value):
        self.widget.set_steps_per_frame(value)
        
    def on_updates_changed(self, value):
        self.widget.set_updates_per_step(value)
        
    def on_updates_changed_by_widget(self, value):
        self.updates_spinbox.setValue(value)
        
    def on_wolff_changed(self, checked):
        self.widget.set_use_wolff(checked)
        # 更新性能标签
        if checked:
            text = "性能: Wolff算法"
        else:
            text = "性能: 棋盘格优化"
        self.performance_label.setText(text)
        
    def on_checkerboard_changed(self, checked):
        self.widget.set_use_checkerboard(checked)
        if not self.wolff_checkbox.isChecked():
            text = "性能: " + ("棋盘格优化" if checked else "标准蒙特卡洛")
            self.performance_label.setText(text)
            
    def toggle_pause(self):
        self.widget.toggle_pause()
        self.pause_button.setText("继续" if self.widget.is_paused else "暂停")
            
    def reset_spins(self):
        self.widget.reset_spins()
        self.order_parameter_widget.magnetization_history.clear()
        self.order_parameter_widget.energy_history.clear()
        self.order_parameter_widget.update()
        
    def set_ferromagnetic(self):
        spins_np = np.ones((self.widget.grid_size, self.widget.grid_size), dtype=np.int8)
        self.widget.simulation.spins.from_numpy(spins_np)
        self.widget.update()
        
    def set_antiferromagnetic(self):
        spins = np.ones((self.widget.grid_size, self.widget.grid_size), dtype=np.int8)
        for i in range(self.widget.grid_size):
            for j in range(self.widget.grid_size):
                if (i + j) % 2 == 1:
                    spins[i, j] = -1
        self.widget.simulation.spins.from_numpy(spins)
        self.widget.update()
    def closeEvent(self, event):
        self.widget.timer.stop()  # 停止 Taichi kernel 调用
        ti.reset()
        event.accept()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

