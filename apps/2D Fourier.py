from custom_import import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ── 数学辅助 ────────────────────────
def preprocess_expression(expr):
    return expr.replace('^', '**').replace(' and ', ' & ').replace(' or ', ' | ')

def safe_eval(expr, x, y):
    safe_dict = {
        "x": x, "y": y,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "abs": np.abs, "pi": np.pi, "atan2": np.arctan2
    }
    return eval(expr, {"__builtins__": None}, safe_dict)

def calculate_fourier(image_array):
    if np.all(image_array == 0):
        return None
    f = np.fft.fft2(image_array)
    fshift = np.fft.fftshift(f)
    return 20 * np.log(np.abs(fshift) + 1)

# ── 绘图画布（原逻辑完全不变）────
class SimpleDrawingCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.canvas_size = 400
        self.setFixedSize(self.canvas_size, self.canvas_size)

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)

        self._pixmap_cache = None
        self._dirty = True

        self.last_point = None
        self.drawing = False
        self.mode = "draw"
        self.shape_mode = None
        self.shape_start = None
        self.temp_shape_points = []

        self._brush_size = 5
        self._fill_shape = False
        # 样式已全部由全局样式表控制

    def update_brush_settings(self):
        if self.parent_app:
            self._brush_size = self.parent_app.brush_size.value()
            self._fill_shape = self.parent_app.fill_shape.isChecked()

    def _update_cache(self):
        if self._dirty:
            data = np.ascontiguousarray(self.image)
            h, w = data.shape
            qimg = QImage(data.data, w, h, w, QImage.Format_Grayscale8)
            self._pixmap_cache = QPixmap.fromImage(qimg)
            self._dirty = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        self._update_cache()
        if self._pixmap_cache:
            painter.drawPixmap(0, 0, self._pixmap_cache)
        self._draw_temp_shape(painter)

    def _draw_temp_shape(self, painter):
        if not (self.shape_mode and self.shape_start and self.temp_shape_points):
            return
        self.update_brush_settings()
        pen = QPen(QColor(255, 255, 255), self._brush_size)
        painter.setPen(pen)
        if self._fill_shape:
            brush = QBrush(QColor(255, 255, 255, 100))
            painter.setBrush(brush)
        else:
            painter.setBrush(Qt.NoBrush)
        if len(self.temp_shape_points) > 1:
            if self.shape_mode == "直线":
                painter.setRenderHint(QPainter.Antialiasing, False)
                painter.drawLine(self.shape_start, self.temp_shape_points[-1])
            elif self.shape_mode == "矩形":
                rect = self._normalize_rect(self.shape_start, self.temp_shape_points[-1])
                painter.drawRect(rect)
            elif self.shape_mode == "圆形":
                center = self.shape_start
                radius = self._calculate_radius(center, self.temp_shape_points[-1])
                painter.drawEllipse(center, radius, radius)
            elif self.shape_mode in ["三角形", "五边形", "六边形", "五角星", "六角星"]:
                painter.drawPolygon(self.temp_shape_points)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            self.drawing = True
            self.last_point = pos
            if self.mode == "shape" and self.shape_mode:
                self.shape_start = pos
                self.temp_shape_points = [pos]
            else:
                self._draw_point(pos)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing and event.buttons() & Qt.LeftButton:
            pos = event.pos()
            if self.mode == "shape" and self.shape_mode and self.shape_start:
                self._update_temp_shape(pos)
            else:
                self._draw_line(pos)
            self.last_point = pos
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self.mode == "shape" and self.shape_mode and self.shape_start:
                self._finalize_shape()
            self.drawing = False
            self.last_point = None
            self.parent_app.calculate_fourier()

    def _draw_point(self, pos):
        self.update_brush_settings()
        x, y = pos.x(), pos.y()
        color = 255 if self.mode == "draw" else 0
        half_size = self._brush_size
        self.draw.ellipse(
            [x - half_size, y - half_size, x + half_size, y + half_size],
            fill=color
        )
        self._dirty = True
        self.update()

    def _draw_line(self, pos):
        if self.last_point is None:
            return
        self.update_brush_settings()
        start_x, start_y = self.last_point.x(), self.last_point.y()
        end_x, end_y = pos.x(), pos.y()
        color = 255 if self.mode == "draw" else 0
        self._draw_bresenham_line(start_x, start_y, end_x, end_y, self._brush_size, color)
        self._dirty = True
        self.update()

    def _draw_bresenham_line(self, x0, y0, x1, y1, brush_size, color):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        half_size = brush_size
        while True:
            self.draw.ellipse(
                [x0 - half_size, y0 - half_size, x0 + half_size, y0 + half_size],
                fill=color
            )
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _update_temp_shape(self, pos):
        self.temp_shape_points = [self.shape_start, pos]
        if self.shape_mode in ["五边形", "六边形", "五角星", "六角星"]:
            center_x = (self.shape_start.x() + pos.x()) / 2
            center_y = (self.shape_start.y() + pos.y()) / 2
            radius = min(abs(pos.x() - self.shape_start.x()),
                        abs(pos.y() - self.shape_start.y())) / 2
            if self.shape_mode == "五边形":
                self.temp_shape_points = self._calculate_regular_polygon(center_x, center_y, radius, 5)
            elif self.shape_mode == "六边形":
                self.temp_shape_points = self._calculate_regular_polygon(center_x, center_y, radius, 6)
            elif self.shape_mode == "五角星":
                self.temp_shape_points = self._calculate_star(center_x, center_y, radius, 5)
            elif self.shape_mode == "六角星":
                self.temp_shape_points = self._calculate_star(center_x, center_y, radius, 6)
        elif self.shape_mode == "三角形":
            x1, y1 = self.shape_start.x(), self.shape_start.y()
            x2, y2 = pos.x(), pos.y()
            x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
            self.temp_shape_points = [
                QPoint(x1, y2),
                QPoint(x2, y2),
                QPoint((x1 + x2) // 2, y1)
            ]

    def _finalize_shape(self):
        if not self.shape_start or not self.temp_shape_points:
            return
        self.update_brush_settings()
        if self.shape_mode == "直线":
            start, end = self.shape_start, self.temp_shape_points[-1]
            self.draw.line([start.x(), start.y(), end.x(), end.y()],
                          fill=255, width=self._brush_size)
        elif self.shape_mode == "矩形":
            rect = self._normalize_rect(self.shape_start, self.temp_shape_points[-1])
            if self._fill_shape:
                self.draw.rectangle([rect.left(), rect.top(), rect.right(), rect.bottom()],
                                   fill=255)
            else:
                self.draw.rectangle([rect.left(), rect.top(), rect.right(), rect.bottom()],
                                   outline=255, width=self._brush_size)
        elif self.shape_mode == "圆形":
            center = self.shape_start
            radius = self._calculate_radius(center, self.temp_shape_points[-1])
            bbox = [center.x()-radius, center.y()-radius, center.x()+radius, center.y()+radius]
            if self._fill_shape:
                self.draw.ellipse(bbox, fill=255)
            else:
                self.draw.ellipse(bbox, outline=255, width=self._brush_size)
        elif self.shape_mode in ["三角形", "五边形", "六边形", "五角星", "六角星"]:
            points = [(p.x(), p.y()) for p in self.temp_shape_points]
            if self._fill_shape:
                self.draw.polygon(points, fill=255)
            else:
                self.draw.polygon(points, outline=255, width=self._brush_size)
        self.shape_start = None
        self.temp_shape_points = []
        self._dirty = True
        self.update()

    def _normalize_rect(self, p1, p2):
        return QRect(min(p1.x(), p2.x()), min(p1.y(), p2.y()),
                     abs(p1.x() - p2.x()), abs(p1.y() - p2.y()))

    def _calculate_radius(self, center, point):
        return max(abs(point.x() - center.x()), abs(point.y() - center.y()))

    def _calculate_regular_polygon(self, center_x, center_y, radius, sides):
        points = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides - math.pi / 2
            points.append(QPoint(int(center_x + radius * math.cos(angle)),
                                int(center_y + radius * math.sin(angle))))
        return points

    def _calculate_star(self, center_x, center_y, radius, points_count):
        star_points = []
        outer_radius = radius
        inner_radius = radius * 0.4
        for i in range(points_count * 2):
            angle = math.pi * i / points_count - math.pi / 2
            r = outer_radius if i % 2 == 0 else inner_radius
            star_points.append(QPoint(int(center_x + r * math.cos(angle)),
                                     int(center_y + r * math.sin(angle))))
        return star_points

    def reset_canvas(self):
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self._dirty = True
        self.shape_start = None
        self.temp_shape_points = []
        self.update()

    def get_image_array(self):
        return np.array(self.image)

    def set_image_from_array(self, array):
        self.image = Image.fromarray(array)
        self.draw = ImageDraw.Draw(self.image)
        self._dirty = True
        self.update()

    def save_image(self, file_path):
        self.image.save(file_path)


# ── 主窗口 ────────────────────────
class FourierDrawApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("傅里叶变换绘图工具")
        self.setGeometry(100, 100, 800, 600)

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)

        self.create_tool_panel(main_layout)
        self.create_drawing_area(main_layout)
        self.create_function_panel(main_layout)

    def _make_separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setObjectName("separator")   # 样式通过 ID 选择器定义在 qss 中
        return sep

    def _set_entry_from_example(self, text, entry_widget):
        if text != "选择示例函数...":
            entry_widget.setText(text.split("  (")[0])

    def create_tool_panel(self, parent_layout):
        tool_group = QGroupBox("绘图工具")
        tool_layout = QHBoxLayout(tool_group)
        tool_layout.setSpacing(6)
        tool_layout.setContentsMargins(8, 16, 8, 8)

        # 基本操作
        basic_box = QVBoxLayout()
        basic_box.addWidget(QLabel("基本操作"))
        row = QHBoxLayout()
        self.clear_btn = QPushButton("清空画布")
        self.clear_btn.clicked.connect(self.reset_canvas)
        row.addWidget(self.clear_btn)
        self.invert_btn = QPushButton("反转灰度")
        self.invert_btn.clicked.connect(self.invert_grayscale)
        row.addWidget(self.invert_btn)
        basic_box.addLayout(row)
        tool_layout.addLayout(basic_box)
        tool_layout.addWidget(self._make_separator())

        # 绘图模式
        mode_box = QVBoxLayout()
        mode_box.addWidget(QLabel("绘图模式"))
        row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.draw_btn = QPushButton("画笔")
        self.draw_btn.setCheckable(True)
        self.draw_btn.setChecked(True)
        self.draw_btn.clicked.connect(lambda: self.set_mode("draw"))
        row.addWidget(self.draw_btn)
        self.mode_group.addButton(self.draw_btn)
        self.erase_btn = QPushButton("橡皮擦")
        self.erase_btn.setCheckable(True)
        self.erase_btn.clicked.connect(lambda: self.set_mode("erase"))
        row.addWidget(self.erase_btn)
        self.mode_group.addButton(self.erase_btn)
        mode_box.addLayout(row)
        tool_layout.addLayout(mode_box)
        tool_layout.addWidget(self._make_separator())

        # 形状工具
        shape_box = QVBoxLayout()
        shape_box.addWidget(QLabel("形状工具"))
        row = QHBoxLayout()
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(
            ["选择形状...", "直线", "矩形", "圆形", "三角形", "五边形", "六边形", "五角星", "六角星"]
        )
        self.shape_combo.currentTextChanged.connect(self.activate_shape_mode)
        self.shape_combo.setFixedWidth(90)
        row.addWidget(self.shape_combo)
        self.fill_shape = QCheckBox("填充")
        row.addWidget(self.fill_shape)
        shape_box.addLayout(row)
        tool_layout.addLayout(shape_box)
        tool_layout.addWidget(self._make_separator())

        # 画笔设置
        brush_box = QVBoxLayout()
        brush_box.addWidget(QLabel("画笔设置"))
        row = QHBoxLayout()
        row.addWidget(QLabel("大小:"))
        self.brush_size = QSlider(Qt.Horizontal)
        self.brush_size.setRange(1, 20)
        self.brush_size.setValue(5)
        self.brush_size.setFixedWidth(70)
        row.addWidget(self.brush_size)
        size_label = QLabel("5")
        size_label.setFixedWidth(15)
        self.brush_size.valueChanged.connect(lambda v: size_label.setText(str(v)))
        row.addWidget(size_label)
        brush_box.addLayout(row)
        tool_layout.addLayout(brush_box)
        tool_layout.addWidget(self._make_separator())

        # 保存
        save_box = QVBoxLayout()
        save_box.addWidget(QLabel("保存选项"))
        row = QHBoxLayout()
        self.save_drawing_btn = QPushButton("保存绘图")
        self.save_drawing_btn.clicked.connect(lambda: self.save_image("drawing"))
        row.addWidget(self.save_drawing_btn)
        self.save_spectrum_btn = QPushButton("保存频谱")
        self.save_spectrum_btn.clicked.connect(lambda: self.save_image("spectrum"))
        row.addWidget(self.save_spectrum_btn)
        save_box.addLayout(row)
        tool_layout.addLayout(save_box)

        parent_layout.addWidget(tool_group)

    def create_drawing_area(self, parent_layout):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        draw_group = QGroupBox("绘图区域")
        inner = QVBoxLayout(draw_group)
        inner.setAlignment(Qt.AlignCenter)
        self.drawing_canvas = SimpleDrawingCanvas(self)
        inner.addWidget(self.drawing_canvas)
        left_layout.addWidget(draw_group)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        spec_group = QGroupBox("傅里叶变换频谱")
        inner = QVBoxLayout(spec_group)
        inner.setAlignment(Qt.AlignCenter)
        self.fig = plt.Figure(figsize=(4, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.canvas_fig = FigureCanvas(self.fig)
        self.canvas_fig.setFixedSize(400, 400)
        inner.addWidget(self.canvas_fig)
        right_layout.addWidget(spec_group)
        splitter.addWidget(right)

        splitter.setSizes([350, 350])
        parent_layout.addWidget(splitter, 1)

    def create_function_panel(self, parent_layout):
        func_group = QGroupBox("函数绘图")
        layout = QVBoxLayout(func_group)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 16, 8, 8)

        # 坐标范围
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("坐标范围:"))
        range_row.addWidget(QLabel("X:"))
        self.x_min = QLineEdit("-200"); self.x_min.setFixedWidth(50)
        range_row.addWidget(self.x_min)
        range_row.addWidget(QLabel("到"))
        self.x_max = QLineEdit("200"); self.x_max.setFixedWidth(50)
        range_row.addWidget(self.x_max)
        range_row.addWidget(QLabel("Y:"))
        self.y_min = QLineEdit("-200"); self.y_min.setFixedWidth(50)
        range_row.addWidget(self.y_min)
        range_row.addWidget(QLabel("到"))
        self.y_max = QLineEdit("200"); self.y_max.setFixedWidth(50)
        range_row.addWidget(self.y_max)
        range_row.addWidget(QLabel("分辨率:"))
        self.resolution = QLineEdit("0.5"); self.resolution.setFixedWidth(40)
        range_row.addWidget(self.resolution)
        range_row.addStretch()
        layout.addLayout(range_row)

        tabs = QTabWidget()
        tabs.setMaximumHeight(120)

        # 形状函数选项卡
        shape_tab = QWidget()
        sl = QVBoxLayout(shape_tab)
        sl.setSpacing(6)
        row = QHBoxLayout()
        row.addWidget(QLabel("形状函数:"))
        self.func_entry = QLineEdit()
        self.func_entry.setPlaceholderText("例如: sqrt(x**2+y**2) <= 180*cos(3*atan2(y,x))")
        self.func_entry.returnPressed.connect(self.draw_function)
        row.addWidget(self.func_entry, 1)
        self.draw_func_btn = QPushButton("绘制形状")
        self.draw_func_btn.clicked.connect(self.draw_function)
        row.addWidget(self.draw_func_btn)
        sl.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("示例函数:"))
        self.example_combo = QComboBox()
        self.example_combo.addItems([
            "选择示例函数...",
            "sqrt(x**2+y**2) <= 180*cos(3*atan2(y,x))  (花瓣)"
        ])
        self.example_combo.currentTextChanged.connect(
            lambda t: self._set_entry_from_example(t, self.func_entry)
        )
        row2.addWidget(self.example_combo, 1)
        sl.addLayout(row2)
        tabs.addTab(shape_tab, "形状函数")

        # 灰度函数选项卡
        gray_tab = QWidget()
        gl = QVBoxLayout(gray_tab)
        gl.setSpacing(6)
        row = QHBoxLayout()
        row.addWidget(QLabel("灰度函数:"))
        self.gray_entry = QLineEdit()
        self.gray_entry.setPlaceholderText("例如: 255 * (1 - (x**2 + y**2)/400)")
        self.gray_entry.returnPressed.connect(self.apply_grayscale_function)
        row.addWidget(self.gray_entry, 1)
        self.apply_gray_btn = QPushButton("应用灰度")
        self.apply_gray_btn.clicked.connect(self.apply_grayscale_function)
        row.addWidget(self.apply_gray_btn)
        gl.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("示例函数:"))
        self.gray_example_combo = QComboBox()
        self.gray_example_combo.addItems([
            "选择示例函数...",
            "255 * (1 - (x**2 + y**2)/400)  (径向渐变)",
            "128 + 127 * sin(x/30)  (水平正弦波)",
            "128 + 127 * sin(y/30)  (垂直正弦波)",
            "128 + 127 * sin(sqrt(x**2+y**2)/20)  (径向正弦波)",
            "255 * (0.5 + 0.5*sin(x/25)*cos(y/25))  (二维正弦波)",
            "255 * abs(sin(5*atan2(y,x)))  (角度条纹)",
            "255 * (1 - (abs(x)+abs(y))/100)  (菱形渐变)"
        ])
        self.gray_example_combo.currentTextChanged.connect(
            lambda t: self._set_entry_from_example(t, self.gray_entry)
        )
        row2.addWidget(self.gray_example_combo, 1)
        gl.addLayout(row2)
        tabs.addTab(gray_tab, "灰度函数")

        layout.addWidget(tabs)
        parent_layout.addWidget(func_group)

    # ── 模式切换 ────────────────────
    def set_mode(self, mode):
        self.drawing_canvas.mode = mode
        self.draw_btn.setChecked(mode == "draw")
        self.erase_btn.setChecked(mode == "erase")
        self.shape_combo.setCurrentText("选择形状...")
        self.drawing_canvas.shape_mode = None

    def activate_shape_mode(self, shape):
        if shape != "选择形状...":
            self.drawing_canvas.mode = "shape"
            self.drawing_canvas.shape_mode = shape

    # ── 画布操作 ────────────────────
    def reset_canvas(self):
        self.drawing_canvas.reset_canvas()
        self.ax.clear()
        self.ax.axis('off')
        self.canvas_fig.draw()

    def calculate_fourier(self):
        img = self.drawing_canvas.get_image_array()
        spectrum = calculate_fourier(img)
        self.ax.clear()
        if spectrum is not None:
            self.ax.imshow(spectrum, cmap='gray')
        self.ax.axis('off')
        self.canvas_fig.draw()

    def invert_grayscale(self):
        arr = self.drawing_canvas.get_image_array()
        self.drawing_canvas.set_image_from_array(255 - arr)
        self.calculate_fourier()

    def save_image(self, img_type):
        if img_type == "drawing":
            path, _ = QFileDialog.getSaveFileName(self, "保存绘图", "", "PNG (*.png)")
            if path:
                if not path.lower().endswith('.png'):
                    path += '.png'
                self.drawing_canvas.save_image(path)
        else:
            path, _ = QFileDialog.getSaveFileName(self, "保存频谱", "", "PNG (*.png)")
            if path:
                if not path.lower().endswith('.png'):
                    path += '.png'
                self.fig.savefig(path, bbox_inches='tight', pad_inches=0, dpi=100)

    def _read_ranges(self):
        return (float(self.x_min.text()), float(self.x_max.text()),
                float(self.y_min.text()), float(self.y_max.text()),
                float(self.resolution.text()))

    def _render_function_to_canvas(self, expr, is_binary):
        size = self.drawing_canvas.canvas_size
        x_min, x_max, y_min, y_max, _ = self._read_ranges()

        cx = np.arange(size)
        cy = np.arange(size)
        x_coords = x_min + (cx + 0.5) * (x_max - x_min) / size
        y_coords = y_max - (cy + 0.5) * (y_max - y_min) / size
        X, Y = np.meshgrid(x_coords, y_coords)

        expr_pp = preprocess_expression(expr)
        result = safe_eval(expr_pp, X, Y)

        if is_binary:
            img = np.where(result, 255, 0).astype(np.uint8)
        else:
            img = np.clip(result, 0, 255).astype(np.uint8)
        return img

    def draw_function(self):
        expr = self.func_entry.text().strip()
        if not expr:
            QMessageBox.warning(self, "警告", "请输入函数表达式")
            return
        try:
            img = self._render_function_to_canvas(expr, is_binary=True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"表达式求值失败: {e}")
            return
        self.drawing_canvas.set_image_from_array(img)
        self.calculate_fourier()

    def apply_grayscale_function(self):
        expr = self.gray_entry.text().strip()
        if not expr:
            QMessageBox.warning(self, "警告", "请输入灰度函数表达式")
            return
        try:
            img = self._render_function_to_canvas(expr, is_binary=False)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"表达式求值失败: {e}")
            return
        self.drawing_canvas.set_image_from_array(img)
        self.calculate_fourier()


# ── 入口 ──────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FourierDrawApp()
    window.show()
    sys.exit(app.exec())