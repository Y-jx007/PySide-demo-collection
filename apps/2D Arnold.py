from custom_import import *

@njit(parallel=True, nogil=True, cache=True)
def cat_map_transform(image_array, p, q):
    """使用Numba加速并行的猫变换
    
    参数:
        image_array: 输入图像数组 (H, W, C) 或 (H, W)
        p: 变换参数p
        q: 变换参数q
    
    返回:
        变换后的图像数组
    """
    # 确保处理的是副本，不修改原图
    if image_array.ndim == 3:
        # 彩色图像 (H, W, C)
        current_array = image_array.copy()
        height, width, channels = current_array.shape
        N = min(height, width)
        
        # 创建输出数组
        new_array = np.zeros_like(current_array)
        
        # 对每个通道进行变换
        for c in range(channels):
            for x in prange(N):  # 使用prange进行并行循环
                for y in range(N):
                    new_x = (x + p * y) % N
                    new_y = (q * x + (p * q + 1) * y) % N
                    
                    if new_x < width and new_y < height:
                        new_array[new_y, new_x, c] = current_array[y, x, c]
    else:
        # 灰度图像 (H, W)
        current_array = image_array.copy()
        height, width = current_array.shape
        N = min(height, width)
        new_array = np.zeros_like(current_array)
        
        for x in prange(N):  # 使用prange进行并行循环
            for y in range(N):
                new_x = (x + p * y) % N
                new_y = (q * x + (p * q + 1) * y) % N
                
                if new_x < width and new_y < height:
                    new_array[new_y, new_x] = current_array[y, x]
    
    return new_array


@njit(parallel=True, nogil=True, cache=True)
def inverse_cat_map_transform(image_array, p, q):
    """使用Numba加速并行的猫逆变换"""
    if image_array.ndim == 3:
        current_array = image_array.copy()
        height, width, channels = current_array.shape
        N = min(height, width)
        new_array = np.zeros_like(current_array)
        
        for c in range(channels):
            for x in prange(N):
                for y in range(N):
                    orig_x = ((p * q + 1) * x - p * y) % N
                    orig_y = (-q * x + y) % N
                    
                    if 0 <= orig_x < width and 0 <= orig_y < height:
                        new_array[orig_y, orig_x, c] = current_array[y, x, c]
    else:
        current_array = image_array.copy()
        height, width = current_array.shape
        N = min(height, width)
        new_array = np.zeros_like(current_array)
        
        for x in prange(N):
            for y in range(N):
                orig_x = ((p * q + 1) * x - p * y) % N
                orig_y = (-q * x + y) % N
                
                if 0 <= orig_x < width and 0 <= orig_y < height:
                    new_array[orig_y, orig_x] = current_array[y, x]
    
    return new_array


class CatMapWorker(QThread):
    """工作线程，用于执行猫变换计算"""
    progress_updated = Signal(int, int)  # 进度, 迭代次数
    transformation_finished = Signal(np.ndarray, int, int)  # 结果, 迭代次数, 周期
    frame_ready = Signal(np.ndarray, int)  # 每帧完成时发射信号
    period_found = Signal(int)  # 找到周期时发射信号
    
    def __init__(self, image_array, original_hash, p, q, max_iterations, update_interval=1):
        super().__init__()
        self.image_array = image_array
        self.original_hash = original_hash
        self.p = p
        self.q = q
        self.max_iterations = max_iterations
        self.update_interval = update_interval  # 更新间隔，每N次变换更新一次图像
        self._is_running = True
        self.current_iteration = 0
        self.period = 0
        self.image_history = {}  # 记录图像哈希和对应的迭代次数
    
    def run(self):
        """执行猫变换"""
        if self.image_array is None:
            return
            
        current_array = self.image_array.copy()
        height, width = current_array.shape[:2]
        N = min(height, width)
        
        # 记录初始状态
        current_hash = self.image_hash(current_array)
        self.image_history[current_hash] = 0
        
        for i in range(self.max_iterations):
            if not self._is_running:
                break
                
            # 使用Numba加速的变换
            current_array = cat_map_transform(current_array, self.p, self.q)
            self.current_iteration = i + 1
            
            # 计算当前图像哈希
            current_hash = self.image_hash(current_array)
            
            # 检查是否回到原始状态（找到周期）
            if current_hash == self.original_hash and self.current_iteration > 0:
                self.period = self.current_iteration
                self.period_found.emit(self.period)
                # 确保在找到周期时更新图像
                self.frame_ready.emit(current_array.copy(), self.current_iteration)
                break
            
            # 检查是否出现重复状态（找到周期）
            if current_hash in self.image_history and self.current_iteration > 0:
                prev_iteration = self.image_history[current_hash]
                self.period = self.current_iteration - prev_iteration
                self.period_found.emit(self.period)
                # 确保在找到周期时更新图像
                self.frame_ready.emit(current_array.copy(), self.current_iteration)
                break
            else:
                self.image_history[current_hash] = self.current_iteration
            
            # 根据更新间隔决定是否发射帧
            if self.current_iteration % self.update_interval == 0:
                self.frame_ready.emit(current_array.copy(), self.current_iteration)
            
            progress = int((i + 1) / self.max_iterations * 100)
            self.progress_updated.emit(progress, self.current_iteration)
            
            # 短暂暂停以便观察动画效果
            time.sleep(0.001)  # 1ms延迟，确保UI能够响应
        
        if self._is_running:
            # 确保在结束时更新图像
            self.frame_ready.emit(current_array.copy(), self.current_iteration)
            self.transformation_finished.emit(current_array, self.current_iteration, self.period)
    
    def image_hash(self, image_array):
        """计算图像的哈希值，用于检测周期"""
        # 使用图像数据的哈希来快速比较图像是否相同
        return hashlib.md5(image_array.tobytes()).hexdigest()
    
    def stop(self):
        """停止变换"""
        self._is_running = False


class CatMapWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.current_image = None
        self.original_hash = None
        self.worker = None
        self.iteration_count = 0
        self.detected_period = 0
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("Arnold's Cat Map 图像置乱工具 - 增强版")
        self.setGeometry(100, 100, 800, 600)
        
        # 设置字体
        self.setFont(QFont("Arial", 10))
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局 - 垂直分割
        main_layout = QVBoxLayout(central_widget)
        
        # 上部控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 下部图像显示区域
        image_panel = self.create_image_panel()
        main_layout.addWidget(image_panel, 1)  # 1表示可扩展
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("控制面板")
        panel.setMaximumHeight(280)
        layout = QHBoxLayout(panel)
        
        # 左侧文件操作
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)
        
        self.load_btn = QPushButton("加载图像")
        self.load_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        file_layout.addWidget(self.save_btn)
        
        self.reset_btn = QPushButton("重置图像")
        self.reset_btn.clicked.connect(self.reset_image)
        self.reset_btn.setEnabled(False)
        file_layout.addWidget(self.reset_btn)
        
        layout.addWidget(file_group)
        
        # 中间参数设置
        param_group = QGroupBox("变换参数")
        param_layout = QVBoxLayout(param_group)
        
        # 第一行参数
        param_row1 = QHBoxLayout()
        
        # N 参数
        n_layout = QVBoxLayout()
        n_layout.addWidget(QLabel("图像尺寸 (N)"))
        self.n_spinbox = QSpinBox()
        self.n_spinbox.setRange(1, 10000)
        self.n_spinbox.setValue(256)
        self.n_spinbox.valueChanged.connect(self.on_n_changed)
        n_layout.addWidget(self.n_spinbox)
        param_row1.addLayout(n_layout)
        
        # p 参数
        p_layout = QVBoxLayout()
        p_layout.addWidget(QLabel("p 参数"))
        self.p_spinbox = QSpinBox()
        self.p_spinbox.setRange(1, 50)
        self.p_spinbox.setValue(1)
        p_layout.addWidget(self.p_spinbox)
        param_row1.addLayout(p_layout)
        
        # q 参数
        q_layout = QVBoxLayout()
        q_layout.addWidget(QLabel("q 参数"))
        self.q_spinbox = QSpinBox()
        self.q_spinbox.setRange(1, 50)
        self.q_spinbox.setValue(1)
        q_layout.addWidget(self.q_spinbox)
        param_row1.addLayout(q_layout)
        
        param_layout.addLayout(param_row1)
        
        # 第二行参数
        param_row2 = QHBoxLayout()
        
        # 最大迭代次数
        iter_layout = QVBoxLayout()
        iter_layout.addWidget(QLabel("最大迭代次数"))
        self.iter_spinbox = QSpinBox()
        self.iter_spinbox.setRange(100, 1000000)
        self.iter_spinbox.setValue(10000)
        self.iter_spinbox.setSingleStep(100)
        iter_layout.addWidget(self.iter_spinbox)
        param_row2.addLayout(iter_layout)
        
        # 更新间隔
        interval_layout = QVBoxLayout()
        interval_layout.addWidget(QLabel("更新间隔"))
        interval_control = QHBoxLayout()
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setRange(1, 1000)
        self.interval_slider.setValue(1)
        self.interval_slider.valueChanged.connect(self.update_interval_label)
        interval_control.addWidget(self.interval_slider)
        self.interval_label = QLabel("1")
        interval_control.addWidget(self.interval_label)
        interval_layout.addLayout(interval_control)
        param_row2.addLayout(interval_layout)
        
        # 保持宽高比
        self.aspect_ratio_check = QCheckBox("保持宽高比")
        self.aspect_ratio_check.setChecked(True)
        param_row2.addWidget(self.aspect_ratio_check)
        
        param_layout.addLayout(param_row2)
        
        # 变换公式显示
        formula_layout = QVBoxLayout()
        self.formula_label = QLabel()
        self.formula_label.setAlignment(Qt.AlignCenter)
        self.formula_label.setStyleSheet("font-size: 12px; color: #555;")
        self.update_formula_display()
        formula_layout.addWidget(self.formula_label)
        param_layout.addLayout(formula_layout)
        
        layout.addWidget(param_group)
        
        # 右侧变换控制
        control_group = QGroupBox("变换控制")
        control_layout = QVBoxLayout(control_group)
        
        self.transform_btn = QPushButton("开始变换")
        self.transform_btn.clicked.connect(self.start_transformation)
        self.transform_btn.setEnabled(False)
        control_layout.addWidget(self.transform_btn)
        
        self.stop_btn = QPushButton("停止变换")
        self.stop_btn.clicked.connect(self.stop_transformation)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        # 单步执行
        step_layout = QHBoxLayout()
        self.step_btn = QPushButton("单步执行")
        self.step_btn.clicked.connect(self.step_transformation)
        self.step_btn.setEnabled(False)
        step_layout.addWidget(self.step_btn)
        
        self.inverse_step_btn = QPushButton("单步逆执行")
        self.inverse_step_btn.clicked.connect(self.inverse_step_transformation)
        self.inverse_step_btn.setEnabled(False)
        step_layout.addWidget(self.inverse_step_btn)
        control_layout.addLayout(step_layout)
        
        layout.addWidget(control_group)
        
        # 最右侧信息显示
        info_group = QGroupBox("变换信息")
        info_group.setFixedWidth(180)
        info_layout = QVBoxLayout(info_group)
        
        self.iteration_label = QLabel("迭代次数: 0")
        info_layout.addWidget(self.iteration_label)
        
        self.period_label = QLabel("周期: 未检测")
        info_layout.addWidget(self.period_label)
        
        self.status_label = QLabel("状态: 等待开始")
        info_layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        info_layout.addWidget(self.progress_bar)
        
        layout.addWidget(info_group)
        
        return panel
    
    def update_formula_display(self):
        """更新变换公式显示"""
        formula_text = "正变换: x' = (x + p y) mod N, y' = (q x + (p q + 1) y) mod N\n"
        formula_text += "逆变换: x = ((p q + 1) x' - p y') mod N, y = (-q x' + y') mod N"
        
        self.formula_label.setText(formula_text)
    
    def update_interval_label(self, value):
        """更新间隔标签"""
        self.interval_label.setText(str(value))
    
    def create_image_panel(self):
        """创建图像显示面板"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # 使用分割器，使左右面板可以调整大小
        splitter = QSplitter(Qt.Horizontal)
        
        # 原始图像面板
        original_group = QGroupBox("原始图像")
        original_layout = QVBoxLayout(original_group)
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        self.original_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.original_label.setText("原始图像将显示在这里")
        original_layout.addWidget(self.original_label)
        
        # 原始图像信息
        self.original_info = QLabel("尺寸: - x -")
        original_layout.addWidget(self.original_info)
        
        splitter.addWidget(original_group)
        
        # 变换后图像面板
        transformed_group = QGroupBox("变换后图像")
        transformed_layout = QVBoxLayout(transformed_group)
        self.transformed_label = QLabel()
        self.transformed_label.setAlignment(Qt.AlignCenter)
        self.transformed_label.setMinimumSize(400, 400)
        self.transformed_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.transformed_label.setText("变换结果将显示在这里")
        transformed_layout.addWidget(self.transformed_label)
        
        # 变换后图像信息
        self.transformed_info = QLabel("迭代: 0, 周期: -")
        transformed_layout.addWidget(self.transformed_info)
        
        splitter.addWidget(transformed_group)
        
        # 设置分割器比例
        splitter.setSizes([500, 500])
        
        layout.addWidget(splitter)
        
        return panel
    
    def resize_image_to_n(self, image_array, n):
        """将图像调整为N×N大小"""
        if image_array is None:
            return None
            
        # 使用PIL进行高质量缩放
        if len(image_array.shape) == 3:
            height, width, channels = image_array.shape
            if channels == 3:
                pil_image = Image.fromarray(image_array, 'RGB')
            elif channels == 4:
                pil_image = Image.fromarray(image_array, 'RGBA')
        else:
            height, width = image_array.shape
            pil_image = Image.fromarray(image_array, 'L')
        
        # 计算缩放比例，保持宽高比
        if self.aspect_ratio_check.isChecked():
            ratio = min(n / width, n / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 创建N×N的画布，将图像放在中央
            if pil_image.mode == 'RGB':
                new_image = Image.new('RGB', (n, n), (255, 255, 255))
            elif pil_image.mode == 'RGBA':
                new_image = Image.new('RGBA', (n, n), (255, 255, 255, 0))
            else:  # 灰度图像
                new_image = Image.new('L', (n, n), 255)
                
            offset = ((n - new_width) // 2, (n - new_height) // 2)
            new_image.paste(pil_image, offset)
            pil_image = new_image
        else:
            # 直接缩放为N×N
            pil_image = pil_image.resize((n, n), Image.Resampling.LANCZOS)
        
        return np.array(pil_image)
    
    def load_image(self):
        """加载图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", 
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            try:
                # 使用PIL加载图像
                pil_image = Image.open(file_path)
                # 转换为numpy数组
                image_array = np.array(pil_image)
                
                self.original_image = image_array
                
                # 调整图像大小为N×N
                n = self.n_spinbox.value()
                resized_image = self.resize_image_to_n(self.original_image, n)
                self.current_image = resized_image.copy()
                
                # 计算原始图像的哈希值，用于周期检测
                self.original_hash = self.image_hash(resized_image)
                
                # 显示原始图像
                self.display_image(resized_image, self.original_label)
                self.display_image(self.current_image, self.transformed_label)
                
                # 更新界面状态
                self.transform_btn.setEnabled(True)
                self.reset_btn.setEnabled(True)
                self.save_btn.setEnabled(True)
                self.step_btn.setEnabled(True)
                self.inverse_step_btn.setEnabled(True)
                
                # 重置迭代计数和周期
                self.iteration_count = 0
                self.detected_period = 0
                self.update_info_display()
                
                # 显示图像信息
                height, width = resized_image.shape[:2]
                channels = resized_image.shape[2] if len(resized_image.shape) > 2 else 1
                self.original_info.setText(f"尺寸: {width} × {height}, 通道: {channels}")
                self.transformed_info.setText(f"迭代: 0, 周期: -")
                
                self.status_label.setText(f"像素: {n}×{n}")
                
            except Exception as e:
                self.status_label.setText(f"加载图像错误: {str(e)}")
    
    def image_hash(self, image_array):
        """计算图像的哈希值"""
        return hashlib.md5(image_array.tobytes()).hexdigest()
    
    def on_n_changed(self):
        """当N值改变时重新调整图像大小"""
        if self.original_image is not None:
            n = self.n_spinbox.value()
            resized_image = self.resize_image_to_n(self.original_image, n)
            self.current_image = resized_image.copy()
            self.original_hash = self.image_hash(resized_image)
            self.display_image(resized_image, self.original_label)
            self.display_image(self.current_image, self.transformed_label)
            
            # 重置迭代计数和周期
            self.iteration_count = 0
            self.detected_period = 0
            self.update_info_display()
            
            height, width = resized_image.shape[:2]
            channels = resized_image.shape[2] if len(resized_image.shape) > 2 else 1
            self.original_info.setText(f"尺寸: {width} × {height}, 通道: {channels}")
            self.transformed_info.setText(f"迭代: 0, 周期: -")
            
            self.status_label.setText(f"像素: {n}×{n}")
    
    def display_image(self, image_array, label):
        """在QLabel中显示numpy数组图像"""
        if image_array is None:
            return
            
        try:
            # 转换numpy数组为QImage
            if len(image_array.shape) == 3:
                height, width, channel = image_array.shape
                if channel == 3:
                    q_image = QImage(
                        image_array.data, 
                        width, height, 
                        3 * width, 
                        QImage.Format_RGB888
                    )
                elif channel == 4:
                    q_image = QImage(
                        image_array.data, 
                        width, height, 
                        4 * width, 
                        QImage.Format_RGBA8888
                    )
            else:
                height, width = image_array.shape
                q_image = QImage(
                    image_array.data, 
                    width, height, 
                    width, 
                    QImage.Format_Grayscale8
                )
            
            # 缩放图像以适应标签大小
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                label.width() - 20, 
                label.height() - 20,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            label.setText(f"显示图像错误: {str(e)}")
    
    def start_transformation(self):
        """开始猫变换"""
        if self.current_image is None or self.worker is not None:
            return
        
        p = self.p_spinbox.value()
        q = self.q_spinbox.value()
        max_iterations = self.iter_spinbox.value()
        
        # 获取更新间隔
        update_interval = self.interval_slider.value()
        
        # 禁用控制按钮
        self.transform_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.load_btn.setEnabled(False)
        self.step_btn.setEnabled(False)
        self.inverse_step_btn.setEnabled(False)
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建工作线程
        self.worker = CatMapWorker(
            self.current_image, 
            self.original_hash,
            p, q, 
            max_iterations, 
            update_interval
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.period_found.connect(self.period_detected)
        self.worker.transformation_finished.connect(self.transformation_complete)
        self.worker.start()
        
        self.status_label.setText("正在进行猫变换...")
    
    def step_transformation(self):
        """单步执行猫变换（正变换）"""
        if self.current_image is None:
            return
            
        p = self.p_spinbox.value()
        q = self.q_spinbox.value()
        
        # 使用Numba加速的变换
        self.current_image = cat_map_transform(self.current_image, p, q)
        self.iteration_count += 1
        
        # 检查是否回到原始状态
        current_hash = self.image_hash(self.current_image)
        if current_hash == self.original_hash and self.iteration_count > 0:
            self.detected_period = self.iteration_count
            self.period_label.setText(f"周期: {self.detected_period}")
        
        self.update_info_display()
        self.display_image(self.current_image, self.transformed_label)
    
    def inverse_step_transformation(self):
        """单步执行猫变换（逆变换）"""
        if self.current_image is None:
            return
            
        p = self.p_spinbox.value()
        q = self.q_spinbox.value()
        
        # 使用Numba加速的逆变换
        self.current_image = inverse_cat_map_transform(self.current_image, p, q)
        self.iteration_count += -1
        
        # 检查是否回到原始状态
        current_hash = self.image_hash(self.current_image)
        if current_hash == self.original_hash and self.iteration_count > 0:
            self.detected_period = self.iteration_count
            self.period_label.setText(f"周期: {self.detected_period}")
        
        self.update_info_display()
        self.display_image(self.current_image, self.transformed_label)
    
    def update_frame(self, image_array, iteration):
        """更新动画帧"""
        self.current_image = image_array
        self.iteration_count = iteration
        self.update_info_display()
        self.display_image(self.current_image, self.transformed_label)
    
    def update_progress(self, value, iteration):
        """更新进度条和迭代次数"""
        self.progress_bar.setValue(value)
        self.iteration_count = iteration
        self.iteration_label.setText(f"迭代次数: {iteration}")
    
    def update_info_display(self):
        """更新信息显示"""
        self.iteration_label.setText(f"迭代次数: {self.iteration_count}")
        self.transformed_info.setText(f"迭代: {self.iteration_count}, 周期: {self.detected_period if self.detected_period > 0 else '-'}")
    
    def period_detected(self, period):
        """周期检测回调"""
        self.detected_period = period
        self.period_label.setText(f"周期: {period}")
        self.transformed_info.setText(f"迭代: {self.iteration_count}, 周期: {period}")
        self.status_label.setText(f"找到周期: {period}")
    
    def stop_transformation(self):
        """停止变换"""
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
            
        self.transform_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.load_btn.setEnabled(True)
        self.step_btn.setEnabled(True)
        self.inverse_step_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.status_label.setText("变换已停止")
    
    def transformation_complete(self, result_array, final_iteration, period):
        """变换完成处理"""
        self.current_image = result_array
        self.iteration_count = final_iteration
        self.detected_period = period
        
        self.update_info_display()
        self.display_image(self.current_image, self.transformed_label)
        
        # 重置界面状态
        self.worker = None
        self.transform_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.load_btn.setEnabled(True)
        self.step_btn.setEnabled(True)
        self.inverse_step_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def reset_image(self):
        """重置图像到原始状态"""
        if self.original_image is not None:
            n = self.n_spinbox.value()
            resized_image = self.resize_image_to_n(self.original_image, n)
            self.current_image = resized_image.copy()
            self.original_hash = self.image_hash(resized_image)
            self.display_image(self.current_image, self.transformed_label)
            
            # 重置迭代计数和周期
            self.iteration_count = 0
            self.detected_period = 0
            self.update_info_display()
            
            self.transformed_info.setText(f"迭代: 0, 周期: -")
            self.status_label.setText("图像已重置")
    
    def save_image(self):
        """保存变换后的图像"""
        if self.current_image is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "", 
            "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)"
        )
        
        if file_path:
            try:
                # 转换numpy数组为PIL图像并保存
                pil_image = Image.fromarray(self.current_image)
                pil_image.save(file_path)
                self.status_label.setText(f"图像已保存: {file_path}")
            except Exception as e:
                self.status_label.setText(f"保存图像错误: {str(e)}")
    
    def resizeEvent(self, event):
        """窗口大小改变时调整图像显示"""
        super().resizeEvent(event)
        if self.original_image is not None:
            self.display_image(self.resize_image_to_n(self.original_image, self.n_spinbox.value()), self.original_label)
        if self.current_image is not None:
            self.display_image(self.current_image, self.transformed_label)


def main():
    app = QApplication(sys.argv)
    window = CatMapWidget()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 