from custom_import import *

# 经典 IFS 分形数据
IFS_FRACTALS = {
    "蕨类植物": {
        "transforms": [
            (0.0, 0.0, 0.0, 0.16, 0.0, 0.0),
            (0.85, 0.04, -0.04, 0.85, 0.0, 1.6),
            (0.2, -0.26, 0.23, 0.22, 0.0, 1.6),
            (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44)
        ],
        "probabilities": [0.01, 0.85, 0.07, 0.07],
        "iterations": 50000
    },
    "谢尔宾斯基三角形": {
        "transforms": [
            (0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
            (0.5, 0.0, 0.0, 0.5, 0.5, 0.0),
            (0.5, 0.0, 0.0, 0.5, 0.25, 0.5)
        ],
        "probabilities": [0.333, 0.333, 0.333],
        "iterations": 50000
    },
    "龙形曲线": {
        "transforms": [
            (0.5, -0.5, 0.5, 0.5, 0.0, 0.0),
            (-0.5, -0.5, 0.5, -0.5, 1.0, 0.0)
        ],
        "probabilities": [0.5, 0.5],
        "iterations": 50000
    },
    "分形树": {
        "transforms": [
            (0.0, 0.0, 0.0, 0.5, 0.0, 0.0),
            (0.42, -0.42, 0.42, 0.42, 0.0, 0.2),
            (0.42, 0.42, -0.42, 0.42, 0.0, 0.2),
            (0.1, 0.0, 0.0, 0.1, 0.0, 0.2)
        ],
        "probabilities": [0.05, 0.4, 0.4, 0.15],
        "iterations": 50000
    }
}


@jit(nopython=True)
def generate_ifs_points(transforms, probabilities, iterations):
    points = np.zeros((iterations, 2), dtype=np.float64)
    x, y = 0.0, 0.0
    cum_probs = np.zeros(len(probabilities) + 1, dtype=np.float64)
    for i in range(len(probabilities)):
        cum_probs[i+1] = cum_probs[i] + probabilities[i]

    for i in range(iterations):
        r = np.random.random()          # 修复点：使用 NumPy 随机数
        idx = 0
        for j in range(len(probabilities)):
            if r < cum_probs[j+1]:
                idx = j
                break
        a, b, c, d, e, f = transforms[idx]
        x_new = a * x + b * y + e
        y_new = c * x + d * y + f
        x, y = x_new, y_new
        points[i, 0] = x
        points[i, 1] = y
    return points


class FractalWorker(QThread):
    finished = Signal(np.ndarray)

    def __init__(self, transforms, probabilities, iterations):
        super().__init__()
        self.transforms = transforms
        self.probabilities = probabilities
        self.iterations = iterations

    def run(self):
        transforms_np = np.array(self.transforms, dtype=np.float64)
        probs_np = np.array(self.probabilities, dtype=np.float64)
        points = generate_ifs_points(transforms_np, probs_np, self.iterations)
        self.finished.emit(points)


class FractalWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.points = None
        self._cached_image = None
        self.color_start = QColor(218, 247, 166)
        self.color_end = QColor(34, 139, 34)
        self.setMinimumSize(400, 400)

    def set_points(self, points):
        self.points = points
        self._cached_image = None
        self.update()

    def set_colors(self, start, end):
        self.color_start = start
        self.color_end = end
        self._cached_image = None
        if self.points is not None:
            self.update()

    def _render_to_image(self):
        """将点集渲染为 QImage（使用 numpy 批量计算替代逐点 Qt 绘制）"""
        if self.points is None:
            return None

        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return None

        x_coords = self.points[:, 0]
        y_coords = self.points[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        scale_x = w / (x_max - x_min) if x_max != x_min else 1
        scale_y = h / (y_max - y_min) if y_max != y_min else 1
        scale = min(scale_x, scale_y) * 0.9
        offset_x = (w - (x_max - x_min) * scale) / 2
        offset_y = (h - (y_max - y_min) * scale) / 2

        # 批量计算像素坐标
        px = ((x_coords - x_min) * scale + offset_x).astype(np.int32)
        py = ((y_coords - y_min) * scale + offset_y).astype(np.int32)
        py = h - py  # 翻转 Y 轴

        # 裁剪到有效范围
        mask = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        px, py = px[mask], py[mask]

        # 计算每个点的颜色（基于迭代索引的比例→梯度）
        total = len(self.points)
        indices = np.where(mask)[0]
        t = np.sqrt(indices / total) if total > 0 else np.zeros_like(indices, dtype=float)

        sr, sg, sb = self.color_start.red(), self.color_start.green(), self.color_start.blue()
        er, eg, eb = self.color_end.red(), self.color_end.green(), self.color_end.blue()

        r = np.clip(sr + (er - sr) * t, 0, 255).astype(np.uint8)
        g = np.clip(sg + (eg - sg) * t, 0, 255).astype(np.uint8)
        b = np.clip(sb + (eb - sb) * t, 0, 255).astype(np.uint8)

        # 创建 RGB 图像
        img = np.full((h, w, 3), 255, dtype=np.uint8)  # 白色背景
        img[py, px, 0] = r
        img[py, px, 1] = g
        img[py, px, 2] = b

        return QImage(img.data, w, h, w * 3, QImage.Format_RGB888)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.points is None:
            painter.fillRect(self.rect(), Qt.white)
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignCenter, "选择分形类型并点击生成")
            painter.end()
            return

        if self._cached_image is None:
            self._cached_image = self._render_to_image()

        if self._cached_image:
            painter.drawImage(self.rect(), self._cached_image)
        else:
            painter.fillRect(self.rect(), Qt.white)
        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._cached_image = None  # 尺寸变化后重新渲染

    def get_image(self):
        if self.points is None:
            return None
        img = self._render_to_image()
        if img is None:
            return None
        pixmap = QPixmap.fromImage(img)
        return pixmap


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IFS 分形生成器")
        self.setGeometry(100, 100, 900, 600)
        self.worker = None

        self.color_start = QColor(218, 247, 166)
        self.color_end = QColor(34, 139, 34)

        self.init_ui()
        self.on_fractal_change("蕨类植物")
        self.generate_fractal()

    def init_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        panel = QWidget()
        panel.setMaximumWidth(280)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(10)
        panel_layout.setContentsMargins(5, 5, 5, 5)

        self.create_fractal_selection(panel_layout)
        self.create_parameters_section(panel_layout)
        self.create_transform_section(panel_layout)
        self.create_buttons_section(panel_layout)
        panel_layout.addStretch()

        self.fractal_widget = FractalWidget()
        splitter.addWidget(panel)
        splitter.addWidget(self.fractal_widget)
        splitter.setSizes([280, 620])

    def create_fractal_selection(self, layout):
        group = QGroupBox("分形选择")
        vbox = QVBoxLayout(group)
        vbox.addWidget(QLabel("选择预设分形:"))
        self.fractal_combo = QComboBox()
        self.fractal_combo.addItems(IFS_FRACTALS.keys())
        self.fractal_combo.currentTextChanged.connect(self.on_fractal_change)
        vbox.addWidget(self.fractal_combo)
        layout.addWidget(group)

    def create_parameters_section(self, layout):
        group = QGroupBox("参数设置")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("迭代次数:"))
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1000, 1000000)
        self.iterations_spin.setValue(50000)
        self.iterations_spin.setSingleStep(10000)
        iter_row.addWidget(self.iterations_spin)
        vbox.addLayout(iter_row)

        vbox.addWidget(QLabel("颜色渐变:"))
        colors_row = QHBoxLayout()
        colors_row.addWidget(QLabel("起始:"))
        self.start_color_btn = ColorButton(self.color_start)
        self.start_color_btn.colorChanged.connect(self.on_start_color_changed)
        colors_row.addWidget(self.start_color_btn)
        colors_row.addSpacing(10)
        colors_row.addWidget(QLabel("结束:"))
        self.end_color_btn = ColorButton(self.color_end)
        self.end_color_btn.colorChanged.connect(self.on_end_color_changed)
        colors_row.addWidget(self.end_color_btn)
        colors_row.addStretch()
        vbox.addLayout(colors_row)

        layout.addWidget(group)

    def create_transform_section(self, layout):
        group = QGroupBox("变换参数")
        vbox = QVBoxLayout(group)
        vbox.addWidget(QLabel("IFS变换参数 (每行一个变换，格式: a,b,c,d,e,f,概率):"))
        self.transform_edit = QTextEdit()
        self.transform_edit.setMaximumHeight(120)
        vbox.addWidget(self.transform_edit)
        layout.addWidget(group)

    def create_buttons_section(self, layout):
        group = QGroupBox("操作")
        hbox = QHBoxLayout(group)
        hbox.setSpacing(6)
        self.generate_btn = QPushButton("生成")
        self.generate_btn.clicked.connect(self.generate_fractal)
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_fractal)
        hbox.addWidget(self.generate_btn)
        hbox.addWidget(self.save_btn)
        layout.addWidget(group)

    def on_fractal_change(self, name):
        if name not in IFS_FRACTALS:
            return
        data = IFS_FRACTALS[name]
        self.iterations_spin.setValue(data["iterations"])
        lines = []
        for i, trans in enumerate(data["transforms"]):
            a, b, c, d, e, f = trans
            prob = data["probabilities"][i]
            lines.append(f"{a:.3f},{b:.3f},{c:.3f},{d:.3f},{e:.3f},{f:.3f},{prob:.3f}")
        self.transform_edit.setPlainText("\n".join(lines))

    def on_start_color_changed(self, color):
        self.color_start = color
        self.fractal_widget.set_colors(self.color_start, self.color_end)

    def on_end_color_changed(self, color):
        self.color_end = color
        self.fractal_widget.set_colors(self.color_start, self.color_end)

    def generate_fractal(self):
        if self.worker and self.worker.isRunning():
            return
        text = self.transform_edit.toPlainText().strip()
        transforms = []
        probabilities = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 7:
                continue
            try:
                a, b, c, d, e, f, prob = map(float, parts)
                transforms.append((a, b, c, d, e, f))
                probabilities.append(prob)
            except ValueError:
                continue
        if not transforms:
            return
        self.worker = FractalWorker(transforms, probabilities, self.iterations_spin.value())
        self.worker.finished.connect(self.on_fractal_generated)
        self.generate_btn.setEnabled(False)
        self.worker.start()

    def on_fractal_generated(self, points):
        self.fractal_widget.set_points(points)
        self.fractal_widget.set_colors(self.color_start, self.color_end)
        self.generate_btn.setEnabled(True)

    def save_fractal(self):
        if self.fractal_widget.points is None:
            QMessageBox.warning(self, "警告", "没有可保存的分形，请先生成")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "ifs_fractal.png",
            "PNG Images (*.png);;All Files (*)"
        )
        if filename:
            img = self.fractal_widget.get_image()
            if img and img.save(filename):
                QMessageBox.information(self, "成功", f"图像已保存到: {filename}")
            else:
                QMessageBox.warning(self, "错误", "保存失败")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())