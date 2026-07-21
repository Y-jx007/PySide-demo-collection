from custom_import import *

# ========== L-system 核心 ==========
class LSystem:
    """优化后的 L-system，字符串生成使用列表拼接，O(n) 复杂度"""
    def __init__(self):
        self.axiom = ""
        self.rules = {}
        self.angle = 25.0
        self.distance = 10.0
        self.generations = 4

    def set_axiom(self, axiom):
        self.axiom = axiom

    def add_rule(self, predecessor, successor):
        self.rules[predecessor] = successor

    def generate(self, generations=None):
        if generations is None:
            generations = self.generations
        result = self.axiom
        for _ in range(generations):
            parts = []
            rules = self.rules  # 本地引用加速
            for char in result:
                parts.append(rules.get(char, char))
            result = ''.join(parts)
        return result

    def set_parameters(self, angle, distance, generations):
        self.angle = angle
        self.distance = distance
        self.generations = generations

# ========== 后台生成线程 ==========
class LSystemGenerator(QThread):
    """在后台计算 L-system 路径，避免阻塞 GUI"""
    pathReady = Signal(list, float, float, float, float)  # path, min_x, min_y, max_x, max_y

    def __init__(self, lsystem, start_x, start_y, parent=None):
        super().__init__(parent)
        self.lsystem = lsystem
        self.start_x = start_x
        self.start_y = start_y

    def run(self):
        string = self.lsystem.generate()
        path = []
        stack = []
        x, y = self.start_x, self.start_y
        angle = -90.0
        distance = self.lsystem.distance
        angle_rad = math.radians(self.lsystem.angle)

        cos_a = math.cos
        sin_a = math.sin
        append = path.append

        min_x = max_x = x
        min_y = max_y = y

        for char in string:
            if char == 'F':
                new_x = x + distance * cos_a(math.radians(angle))
                new_y = y + distance * sin_a(math.radians(angle))
                # 更新包围盒
                if new_x < min_x: min_x = new_x
                if new_x > max_x: max_x = new_x
                if new_y < min_y: min_y = new_y
                if new_y > max_y: max_y = new_y
                append(( (x, y), (new_x, new_y) ))
                x, y = new_x, new_y
            elif char == 'f':
                x += distance * cos_a(math.radians(angle))
                y += distance * sin_a(math.radians(angle))
                # 移动也可能影响包围盒（虽然不画线，但影响后续起点，安全起见也更新）
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y
            elif char == '+':
                angle += self.lsystem.angle
            elif char == '-':
                angle -= self.lsystem.angle
            elif char == '[':
                stack.append((x, y, angle))
            elif char == ']':
                if stack:
                    x, y, angle = stack.pop()
        self.pathReady.emit(path, min_x, min_y, max_x, max_y)

# ========== L-system 绘图部件 ==========
class LSystemWidget(QWidget):
    """优化后的绘图部件：缓存路径、包围盒、缩放因子；支持后台生成"""
    def __init__(self):
        super().__init__()
        self.lsystem = None
        self.path = []              # [( (x1,y1), (x2,y2) ), ...]
        self.bounding_rect = QRectF()  # 包围盒
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        self.start_color = QColor(139, 69, 19)
        self.end_color = QColor(34, 139, 34)
        self.bg_color = QColor(240, 240, 240)

        self.setMinimumSize(400, 400)
        self._generator = None          # 当前后台线程

    def set_lsystem(self, lsystem):
        """设置新的 L-system 并启动后台生成"""
        # 如果有正在运行的线程，先终止
        if self._generator and self._generator.isRunning():
            self._generator.terminate()
            self._generator.wait()
        self.lsystem = lsystem
        self._start_background_generation()

    def _start_background_generation(self):
        """启动后台线程计算路径"""
        # 起点设置为画布中心底部（与原始逻辑一致）
        start_x = self.width() / 2.0
        start_y = self.height() - 50.0
        self._generator = LSystemGenerator(self.lsystem, start_x, start_y)
        self._generator.pathReady.connect(self._on_path_ready)
        self._generator.start()

    def _on_path_ready(self, path, min_x, min_y, max_x, max_y):
        """后台计算完成，更新路径并重绘"""
        self.path = path
        # 转换为 QRectF 方便计算
        self.bounding_rect = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)
        self._update_transform()
        self.update()

    def _update_transform(self):
        """根据当前包围盒和窗口大小计算缩放和平移"""
        if not self.path:
            return
        rect = self.bounding_rect
        if rect.width() < 0.01 or rect.height() < 0.01:
            return
        margin = 50
        scale_x = (self.width() - 2 * margin) / rect.width()
        scale_y = (self.height() - 2 * margin) / rect.height()
        self.scale = min(scale_x, scale_y)
        self.offset_x = margin - rect.left() * self.scale
        self.offset_y = margin - rect.top() * self.scale

    def set_colors(self, start, end, bg):
        self.start_color = start
        self.end_color = end
        self.bg_color = bg
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)
        if not self.path:
            return
        self._draw_path(painter)

    def _draw_path(self, painter):
        """公共绘制方法，供 paintEvent 和 save_image 复用"""
        if not self.path:
            return
        scale = self.scale
        off_x = self.offset_x
        off_y = self.offset_y
        path_len = len(self.path)
        # 颜色渐变计算
        sr, sg, sb = self.start_color.red(), self.start_color.green(), self.start_color.blue()
        er, eg, eb = self.end_color.red(), self.end_color.green(), self.end_color.blue()
        dr, dg, db = er - sr, eg - sg, eb - sb

        pen = QPen()
        pen_width = max(1, int(2 * scale))
        pen.setWidth(pen_width)

        for i, ((x1, y1), (x2, y2)) in enumerate(self.path):
            # 根据进度计算颜色
            t = i / (path_len - 1) if path_len > 1 else 0
            r = int(sr + dr * t)
            g = int(sg + dg * t)
            b = int(sb + db * t)
            pen.setColor(QColor(r, g, b))
            painter.setPen(pen)
            painter.drawLine(
                int(x1 * scale + off_x), int(y1 * scale + off_y),
                int(x2 * scale + off_x), int(y2 * scale + off_y)
            )

    def save_image(self, filename):
        """将当前图形保存为图片文件"""
        if not self.path:
            return False
        image = QImage(self.size(), QImage.Format_ARGB32)
        image.fill(self.bg_color)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_path(painter)
        painter.end()
        return image.save(filename)

    def resizeEvent(self, event):
        """窗口大小变化时重新计算变换"""
        super().resizeEvent(event)
        if self.path:
            self._update_transform()

# ========== 主窗口 ==========
class LSystemWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("L-system 分形生成器 (优化版)")
        self.setGeometry(100, 100, 900, 600)

        self.canvas = LSystemWidget()
        self.init_ui()
        self.setup_default_systems()

    def init_ui(self):
        central = QSplitter(Qt.Horizontal)
        self.setCentralWidget(central)

        # 左侧控制面板
        panel = QWidget()
        panel.setMaximumWidth(280)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(10)
        panel_layout.setContentsMargins(5, 5, 5, 5)

        self.create_system_selection(panel_layout)
        self.create_parameters_and_colors_section(panel_layout)
        self.create_rules_section(panel_layout)
        self.create_buttons_section(panel_layout)
        panel_layout.addStretch()

        central.addWidget(panel)
        central.addWidget(self.canvas)
        central.setSizes([280, 620])

    def create_system_selection(self, layout):
        group = QGroupBox("预设系统")
        vbox = QVBoxLayout(group)
        self.system_combo = QComboBox()
        self.system_combo.currentTextChanged.connect(self.on_system_changed)
        vbox.addWidget(self.system_combo)
        layout.addWidget(group)

    def create_parameters_and_colors_section(self, layout):
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)

        # 参数组
        params_group = QGroupBox("参数设置")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(6)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("迭代:"))
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 20)
        self.iterations_spin.setValue(4)
        iter_row.addWidget(self.iterations_spin)
        params_layout.addLayout(iter_row)

        ang_row = QHBoxLayout()
        ang_row.addWidget(QLabel("角度:"))
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(1, 180)
        self.angle_spin.setValue(25)
        self.angle_spin.setSuffix("°")
        ang_row.addWidget(self.angle_spin)
        params_layout.addLayout(ang_row)

        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel("步长:"))
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(1, 50)
        self.distance_spin.setValue(10)
        dist_row.addWidget(self.distance_spin)
        params_layout.addLayout(dist_row)

        hbox.addWidget(params_group)

        # 颜色组
        colors_group = QGroupBox("颜色设置")
        colors_layout = QVBoxLayout(colors_group)
        colors_layout.setSpacing(6)

        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("起始:"))
        self.start_color_btn = ColorButton(self.canvas.start_color)
        self.start_color_btn.colorChanged.connect(self.on_start_color_changed)
        start_row.addWidget(self.start_color_btn)
        colors_layout.addLayout(start_row)

        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("结束:"))
        self.end_color_btn = ColorButton(self.canvas.end_color)
        self.end_color_btn.colorChanged.connect(self.on_end_color_changed)
        end_row.addWidget(self.end_color_btn)
        colors_layout.addLayout(end_row)

        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("背景:"))
        self.bg_color_btn = ColorButton(self.canvas.bg_color)
        self.bg_color_btn.colorChanged.connect(self.on_bg_color_changed)
        bg_row.addWidget(self.bg_color_btn)
        colors_layout.addLayout(bg_row)

        hbox.addWidget(colors_group)
        layout.addWidget(container)

    def create_rules_section(self, layout):
        group = QGroupBox("L-system 规则")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)

        vbox.addWidget(QLabel("公理:"))
        self.axiom_edit = QTextEdit()
        self.axiom_edit.setMaximumHeight(35)
        self.axiom_edit.setPlaceholderText("例如: F")
        vbox.addWidget(self.axiom_edit)

        vbox.addWidget(QLabel("规则 (每行一条，格式: 前驱->后继):"))
        self.rules_edit = QTextEdit()
        self.rules_edit.setMaximumHeight(120)
        self.rules_edit.setPlaceholderText("例如: F->F+F-F-F+F")
        vbox.addWidget(self.rules_edit)
        layout.addWidget(group)

    def create_buttons_section(self, layout):
        group = QGroupBox("操作")
        hbox = QHBoxLayout(group)
        hbox.setSpacing(6)

        self.generate_btn = QPushButton("生成")
        self.generate_btn.clicked.connect(self.generate_system)
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_image)
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_parameters)

        hbox.addWidget(self.generate_btn)
        hbox.addWidget(self.save_btn)
        hbox.addWidget(self.reset_btn)
        layout.addWidget(group)

    def setup_default_systems(self):
        self.systems = {
            "Koch曲线": {"axiom": "F", "rules": "F->F+F-F-F+F", "angle": 90, "distance": 10, "iterations": 4},
            "分形树1": {"axiom": "F", "rules": "F->F[+F]F[-F]F", "angle": 25.7, "distance": 10, "iterations": 4},
            "分形树2": {"axiom": "F", "rules": "F->F[+F]F[-F][F]", "angle": 20, "distance": 10, "iterations": 5},
            "Sierpinski三角": {"axiom": "F", "rules": "F->G-F-G\nG->F+G+F", "angle": 60, "distance": 10, "iterations": 6},
            "分形植物": {"axiom": "X", "rules": "X->F+[[X]-X]-F[-FX]+X\nF->FF", "angle": 25, "distance": 5, "iterations": 5},
            "龙形曲线": {"axiom": "FX", "rules": "X->X+YF+\nY->-FX-Y", "angle": 90, "distance": 5, "iterations": 8},
            "Hilbert曲线": {"axiom": "A", "rules": "A->-BF+AFA+FB-\nB->+AF-BFB-FA+", "angle": 90, "distance": 10, "iterations": 4}
        }
        for name in self.systems:
            self.system_combo.addItem(name)
        self.system_combo.setCurrentIndex(0)

    def on_system_changed(self, name):
        if name not in self.systems:
            return
        sys = self.systems[name]
        self.axiom_edit.setText(sys["axiom"])
        self.rules_edit.setText(sys["rules"])
        self.angle_spin.setValue(sys["angle"])
        self.distance_spin.setValue(sys["distance"])
        self.iterations_spin.setValue(sys["iterations"])
        self.generate_system()

    def generate_system(self):
        """构建 L-system 并传递给绘图部件（后台生成）"""
        lsys = LSystem()
        lsys.set_axiom(self.axiom_edit.toPlainText().strip())
        rules_text = self.rules_edit.toPlainText().strip()
        for line in rules_text.split('\n'):
            if '->' in line:
                pred, succ = line.split('->', 1)
                lsys.add_rule(pred.strip(), succ.strip())
        lsys.set_parameters(
            self.angle_spin.value(),
            self.distance_spin.value(),
            self.iterations_spin.value()
        )
        self.canvas.set_lsystem(lsys)

    def reset_parameters(self):
        self.axiom_edit.clear()
        self.rules_edit.clear()
        self.angle_spin.setValue(25)
        self.distance_spin.setValue(10)
        self.iterations_spin.setValue(4)
        self.canvas.path.clear()
        self.canvas.update()

    def on_start_color_changed(self, color):
        self.canvas.start_color = color
        self.canvas.update()

    def on_end_color_changed(self, color):
        self.canvas.end_color = color
        self.canvas.update()

    def on_bg_color_changed(self, color):
        self.canvas.bg_color = color
        self.canvas.update()

    def save_image(self):
        if not self.canvas.path:
            QMessageBox.warning(self, "警告", "没有可保存的图形，请先生成图形")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存图像", f"lsystem_{self.system_combo.currentText()}.png",
            "PNG Images (*.png);;All Files (*)"
        )
        if filename:
            if self.canvas.save_image(filename):
                QMessageBox.information(self, "成功", f"图像已保存到: {filename}")
            else:
                QMessageBox.warning(self, "错误", "保存图像失败")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LSystemWindow()
    window.show()
    sys.exit(app.exec())