from custom_import import *

# ========== 核心变换（不变）==========
@njit(parallel=True, nogil=True, cache=True)
def _cat_map_core(image_array, p, q, inverse=False):
    if image_array.ndim == 3:
        H, W, C = image_array.shape
        output = np.empty_like(image_array)
        total = H * W
        for idx in prange(total):
            x = idx // W
            y = idx % W
            if inverse:
                nx = ((p * q + 1) * x - p * y) % H
                ny = (-q * x + y) % W
            else:
                nx = (x + p * y) % H
                ny = (q * x + (p * q + 1) * y) % W
            for c in range(C):
                output[ny, nx, c] = image_array[y, x, c]
    else:
        H, W = image_array.shape
        output = np.empty_like(image_array)
        total = H * W
        for idx in prange(total):
            x = idx // W
            y = idx % W
            if inverse:
                nx = ((p * q + 1) * x - p * y) % H
                ny = (-q * x + y) % W
            else:
                nx = (x + p * y) % H
                ny = (q * x + (p * q + 1) * y) % W
            output[ny, nx] = image_array[y, x]
    return output

def cat_map_transform(img, p, q):
    return _cat_map_core(img, p, q, inverse=False)

def inverse_cat_map_transform(img, p, q):
    return _cat_map_core(img, p, q, inverse=True)


# ========== 工作线程（重构）==========
class CatMapWorker(QThread):
    progress_updated = Signal(int, int)          # 进度百分比, 当前全局迭代次数
    transformation_finished = Signal(np.ndarray, int, int)  # 结果, 最终全局迭代次数, 检测到的周期
    frame_ready = Signal(np.ndarray, int)        # 帧, 当前全局迭代次数
    period_found = Signal(int)                   # 周期

    def __init__(self, image_array, original_hash, p, q, max_iterations,
                 start_iteration=0, update_interval=20):
        super().__init__()
        self.image_array = image_array.copy()
        self.original_hash = original_hash
        self.p = p
        self.q = q
        self.max_iterations = max_iterations
        self.update_interval = max(1, update_interval)
        self._is_running = True
        self.start_iteration = start_iteration  # 起始全局计数（可负）

    def run(self):
        if self.image_array is None:
            return
        current = self.image_array
        iter_count = self.start_iteration

        for i in range(self.max_iterations):
            if not self._is_running:
                break

            current = cat_map_transform(current, self.p, self.q)
            iter_count += 1

            # 只需检查是否回到原始图像
            if hashlib.md5(current.tobytes()).hexdigest() == self.original_hash:
                period = abs(iter_count) if iter_count != 0 else 0
                if period > 0:
                    self.period_found.emit(period)
                self.frame_ready.emit(current.copy(), iter_count)
                self.transformation_finished.emit(current, iter_count, period)
                return

            # 按间隔更新
            if (i + 1) % self.update_interval == 0:
                self.frame_ready.emit(current.copy(), iter_count)
                progress = int((i + 1) / self.max_iterations * 100)
                self.progress_updated.emit(progress, iter_count)

        # 达到最大次数仍未回到原始图
        if self._is_running:
            self.frame_ready.emit(current.copy(), iter_count)
            self.transformation_finished.emit(current, iter_count, 0)

    def stop(self):
        self._is_running = False


# ========== 主窗口（重构状态管理）==========
class CatMapWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.current_image = None
        self.original_hash = None
        self.worker = None
        self.global_iteration = 0         # 带符号的全局迭代计数
        self.detected_period = 0
        self.period_candidates = []
        self.init_ui()

    # ------- UI 构建（与之前相似，仅修改状态相关部分）------
    def _create_spinbox(self, label, minv, maxv, default, callback=None, step=1):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(label))
        spin = QSpinBox()
        spin.setRange(minv, maxv)
        spin.setValue(default)
        spin.setSingleStep(step)
        if callback:
            spin.valueChanged.connect(callback)
        layout.addWidget(spin)
        return layout, spin

    def _fast_hash(self, img):
        return hashlib.md5(img.tobytes()).hexdigest()

    def _update_all_info(self):
        self.iteration_label.setText(f"迭代次数: {self.global_iteration}")
        p_text = str(self.detected_period) if self.detected_period > 0 else "未检测"
        self.period_label.setText(f"周期: {p_text}")
        self.transformed_info.setText(f"迭代: {self.global_iteration}, 周期: {p_text}")

    def _record_period(self, period):
        if period > 0:
            self.period_candidates.append(period)
            self.detected_period = min(self.period_candidates)

    def _set_controls_enabled(self, running):
        img_ok = self.current_image is not None
        self.transform_btn.setEnabled(not running and img_ok)
        self.stop_btn.setEnabled(running)
        self.load_btn.setEnabled(not running)
        self.step_btn.setEnabled(not running and img_ok)
        self.inverse_step_btn.setEnabled(not running and img_ok)
        self.progress_bar.setVisible(running)

    def _full_reset_state(self, image):
        self.current_image = image.copy()
        self.original_hash = self._fast_hash(image)
        self.global_iteration = 0
        self.detected_period = 0
        self.period_candidates = []
        self._update_all_info()
        self.display_image(self.current_image, self.transformed_label)
        self._set_controls_enabled(False)

    def init_ui(self):
        self.setWindowTitle("Arnold's Cat Map - 优雅迭代计数")
        self.setGeometry(100, 100, 800, 600)
        self.setFont(QFont("Arial", 10))
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self._create_image_panel(), 1)

    def _create_control_panel(self):
        panel = QGroupBox("控制面板")
        panel.setMaximumHeight(280)
        panel.setContentsMargins(6, 6, 6, 6)
        outer = QHBoxLayout(panel)
        outer.setSpacing(8)

        # 文件操作
        file_group = QGroupBox("文件操作")
        fl = QVBoxLayout(file_group)
        self.load_btn = QPushButton("加载图像")
        self.load_btn.clicked.connect(self.load_image)
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        self.reset_btn = QPushButton("重置图像")
        self.reset_btn.clicked.connect(self.reset_image)
        self.reset_btn.setEnabled(False)
        fl.addWidget(self.load_btn)
        fl.addWidget(self.save_btn)
        fl.addWidget(self.reset_btn)
        outer.addWidget(file_group)

        # 变换参数
        param_group = QGroupBox("变换参数")
        pl = QVBoxLayout(param_group)
        row1 = QHBoxLayout()
        _, self.n_spinbox = self._create_spinbox("图像尺寸 (N)", 1, 10000, 256, self.on_n_changed)
        row1.addLayout(_)
        _, self.p_spinbox = self._create_spinbox("p 参数", 1, 50, 1)
        row1.addLayout(_)
        _, self.q_spinbox = self._create_spinbox("q 参数", 1, 50, 1)
        row1.addLayout(_)
        pl.addLayout(row1)

        row2 = QHBoxLayout()
        _, self.iter_spinbox = self._create_spinbox("最大迭代次数", 100, 1000000, 10000, step=100)
        row2.addLayout(_)

        interval_layout = QVBoxLayout()
        interval_layout.addWidget(QLabel("更新间隔"))
        int_ctl = QHBoxLayout()
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setRange(1, 1000)
        self.interval_slider.setValue(20)
        self.interval_slider.valueChanged.connect(lambda v: self.interval_label.setText(str(v)))
        int_ctl.addWidget(self.interval_slider)
        self.interval_label = QLabel("20")
        int_ctl.addWidget(self.interval_label)
        interval_layout.addLayout(int_ctl)
        row2.addLayout(interval_layout)

        self.aspect_ratio_check = QCheckBox("保持宽高比")
        self.aspect_ratio_check.setChecked(True)
        row2.addWidget(self.aspect_ratio_check)
        pl.addLayout(row2)

        self.formula_label = QLabel(
            "正变换: x' = (x + py) mod N, y' = (qx + (pq+1)y) mod N\n"
            "逆变换: x = ((pq+1)x' - py') mod N, y = (-qx' + y') mod N"
        )
        self.formula_label.setAlignment(Qt.AlignCenter)
        self.formula_label.setStyleSheet("font-size: 12px; color: #555;")
        pl.addWidget(self.formula_label)
        outer.addWidget(param_group)

        # 变换控制
        ctrl_group = QGroupBox("变换控制")
        cl = QVBoxLayout(ctrl_group)
        self.transform_btn = QPushButton("开始变换")
        self.transform_btn.clicked.connect(self.start_transformation)
        self.transform_btn.setEnabled(False)
        self.stop_btn = QPushButton("停止变换")
        self.stop_btn.clicked.connect(self.stop_transformation)
        self.stop_btn.setEnabled(False)
        cl.addWidget(self.transform_btn)
        cl.addWidget(self.stop_btn)

        step_ly = QHBoxLayout()
        self.step_btn = QPushButton("单步执行")
        self.step_btn.clicked.connect(self.step_transformation)
        self.step_btn.setEnabled(False)
        self.inverse_step_btn = QPushButton("单步逆执行")
        self.inverse_step_btn.clicked.connect(self.inverse_step_transformation)
        self.inverse_step_btn.setEnabled(False)
        step_ly.addWidget(self.step_btn)
        step_ly.addWidget(self.inverse_step_btn)
        cl.addLayout(step_ly)
        outer.addWidget(ctrl_group)

        # 信息显示
        info_group = QGroupBox("变换信息")
        info_group.setFixedWidth(180)
        il = QVBoxLayout(info_group)
        self.iteration_label = QLabel("迭代次数: 0")
        self.period_label = QLabel("周期: 未检测")
        self.status_label = QLabel("状态: 等待开始")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        il.addWidget(self.iteration_label)
        il.addWidget(self.period_label)
        il.addWidget(self.status_label)
        il.addWidget(self.progress_bar)
        outer.addWidget(info_group)

        return panel

    def _create_image_panel(self):
        panel = QWidget()
        layout = QHBoxLayout(panel)
        splitter = QSplitter(Qt.Horizontal)

        og = QGroupBox("原始图像")
        ogl = QVBoxLayout(og)
        self.original_label = QLabel("原始图像将显示在这里")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        self.original_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.original_info = QLabel("尺寸: - x -")
        ogl.addWidget(self.original_label)
        ogl.addWidget(self.original_info)
        splitter.addWidget(og)

        tg = QGroupBox("变换后图像")
        tgl = QVBoxLayout(tg)
        self.transformed_label = QLabel("变换结果将显示在这里")
        self.transformed_label.setAlignment(Qt.AlignCenter)
        self.transformed_label.setMinimumSize(400, 400)
        self.transformed_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.transformed_info = QLabel("迭代: 0, 周期: -")
        tgl.addWidget(self.transformed_label)
        tgl.addWidget(self.transformed_info)
        splitter.addWidget(tg)

        splitter.setSizes([500, 500])
        layout.addWidget(splitter)
        return panel

    # ------- 图像处理 -------
    def resize_image_to_n(self, image_array, n):
        if image_array is None:
            return None
        pil_img = Image.fromarray(image_array)
        if self.aspect_ratio_check.isChecked():
            w, h = pil_img.size
            ratio = min(n / w, n / h)
            new_size = (int(w * ratio), int(h * ratio))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            canvas = Image.new(pil_img.mode, (n, n), 255 if pil_img.mode == 'L' else (255,) * len(pil_img.mode))
            offset = ((n - new_size[0]) // 2, (n - new_size[1]) // 2)
            canvas.paste(pil_img, offset)
            return np.array(canvas)
        return np.array(pil_img.resize((n, n), Image.Resampling.LANCZOS))

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "",
                                              "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not path:
            return
        try:
            img = Image.open(path)
            self.original_image = np.array(img)
            self._apply_resize_and_reset()
            self.save_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"加载错误: {e}")

    def on_n_changed(self):
        if self.original_image is not None:
            self._apply_resize_and_reset()

    def _apply_resize_and_reset(self):
        n = self.n_spinbox.value()
        resized = self.resize_image_to_n(self.original_image, n)
        self._full_reset_state(resized)
        self.display_image(resized, self.original_label)
        self.original_info.setText(f"尺寸: {resized.shape[1]} × {resized.shape[0]}")
        self.status_label.setText(f"像素: {n}×{n}")

    def display_image(self, img_array, label):
        if img_array is None:
            return
        try:
            if not img_array.flags['C_CONTIGUOUS']:
                img_array = np.ascontiguousarray(img_array)
            h, w = img_array.shape[:2]
            if img_array.ndim == 3:
                ch = img_array.shape[2]
                fmt = QImage.Format_RGB888 if ch == 3 else QImage.Format_RGBA8888
                qimg = QImage(img_array.data, w, h, ch * w, fmt)
            else:
                qimg = QImage(img_array.data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(label.size() - QSize(20, 20), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)
        except Exception as e:
            label.setText(f"显示错误: {e}")

    # ------- 单步操作（支持负迭代）-------
    def step_transformation(self):
        if self.current_image is None:
            return
        p, q = self.p_spinbox.value(), self.q_spinbox.value()
        self.current_image = cat_map_transform(self.current_image, p, q)
        self.global_iteration += 1
        self._check_period()
        self._update_all_info()
        self.display_image(self.current_image, self.transformed_label)

    def inverse_step_transformation(self):
        if self.current_image is None:
            return
        p, q = self.p_spinbox.value(), self.q_spinbox.value()
        self.current_image = inverse_cat_map_transform(self.current_image, p, q)
        self.global_iteration -= 1
        self._check_period()
        self._update_all_info()
        self.display_image(self.current_image, self.transformed_label)

    def _check_period(self):
        """简单检测：当前图像等于原始图像 且 迭代次数≠0"""
        if self.global_iteration != 0 and self._fast_hash(self.current_image) == self.original_hash:
            self._record_period(abs(self.global_iteration))
            self.status_label.setText(f"找到周期: {self.detected_period}")

    # ------- 自动变换（基于当前状态继续）-------
    def start_transformation(self):
        if self.current_image is None or self.worker is not None:
            return
        self._set_controls_enabled(True)
        self.progress_bar.setValue(0)

        self.worker = CatMapWorker(
            self.current_image,
            self.original_hash,
            self.p_spinbox.value(),
            self.q_spinbox.value(),
            self.iter_spinbox.value(),
            start_iteration=self.global_iteration,  # 继承当前全局计数
            update_interval=self.interval_slider.value()
        )
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.period_found.connect(self._on_period)
        self.worker.transformation_finished.connect(self._on_finished)
        self.worker.start()
        self.status_label.setText("正在进行猫变换...")

    def stop_transformation(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        self._set_controls_enabled(False)
        self.status_label.setText("变换已停止")

    # ------- 回调 -------
    def _on_frame(self, img, iter_num):
        self.current_image = img
        self.global_iteration = iter_num
        self._update_all_info()
        self.display_image(img, self.transformed_label)

    def _on_progress(self, val, iter_num):
        self.progress_bar.setValue(val)
        self.global_iteration = iter_num
        self._update_all_info()

    def _on_period(self, period):
        self._record_period(period)
        self._update_all_info()
        self.status_label.setText(f"找到周期: {self.detected_period}")

    def _on_finished(self, result, final_iter, period):
        self.current_image = result
        self.global_iteration = final_iter
        if period > 0:
            self._record_period(period)
        self._update_all_info()
        self.display_image(result, self.transformed_label)
        self.worker = None
        self._set_controls_enabled(False)
        if period == 0:
            self.status_label.setText("达到最大迭代次数，未发现周期")

    def reset_image(self):
        if self.original_image is not None:
            self._apply_resize_and_reset()
            self.status_label.setText("图像已重置")

    def save_image(self):
        if self.current_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            try:
                Image.fromarray(self.current_image).save(path)
                self.status_label.setText(f"已保存: {path}")
            except Exception as e:
                self.status_label.setText(f"保存失败: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_image is not None:
            self.display_image(
                self.resize_image_to_n(self.original_image, self.n_spinbox.value()),
                self.original_label
            )
        if self.current_image is not None:
            self.display_image(self.current_image, self.transformed_label)


# ========== 入口 ==========
def main():
    app = QApplication(sys.argv)
    import os
    qss_path = os.path.join(os.path.dirname(__file__), "style.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    window = CatMapWidget()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()