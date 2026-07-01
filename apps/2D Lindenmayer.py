from custom_import import *

class LSystem:
    """L-system 实现类"""
    def __init__(self):
        self.axiom = ""
        self.rules = {}
        self.angle = 25
        self.distance = 10
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
            new_result = ""
            for char in result:
                new_result += self.rules.get(char, char)
            result = new_result
        return result

    def set_parameters(self, angle, distance, generations):
        self.angle = angle
        self.distance = distance
        self.generations = generations


class LSystemWidget(QWidget):
    """L-system 绘图部件"""
    def __init__(self):
        super().__init__()
        self.lsystem = LSystem()
        self.path = []
        self.start_color = QColor(139, 69, 19)
        self.end_color = QColor(34, 139, 34)
        self.bg_color = QColor(240, 240, 240)
        self.setMinimumSize(400, 400)

    def set_lsystem(self, lsystem):
        self.lsystem = lsystem
        self.generate_path()

    def set_colors(self, start_color, end_color, bg_color):
        self.start_color = start_color
        self.end_color = end_color
        self.bg_color = bg_color
        self.update()

    def generate_path(self):
        self.path = []
        string = self.lsystem.generate()
        stack = []
        x, y = self.width() / 2, self.height() - 50
        angle = -90
        distance = self.lsystem.distance

        for char in string:
            if char == 'F':
                new_x = x + distance * math.cos(math.radians(angle))
                new_y = y + distance * math.sin(math.radians(angle))
                self.path.append((QPoint(int(x), int(y)), QPoint(int(new_x), int(new_y))))
                x, y = new_x, new_y
            elif char == 'f':
                x += distance * math.cos(math.radians(angle))
                y += distance * math.sin(math.radians(angle))
            elif char == '+':
                angle += self.lsystem.angle
            elif char == '-':
                angle -= self.lsystem.angle
            elif char == '[':
                stack.append((x, y, angle))
            elif char == ']':
                x, y, angle = stack.pop()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)

        if not self.path:
            return

        min_x = min(min(s.x(), e.x()) for s, e in self.path)
        max_x = max(max(s.x(), e.x()) for s, e in self.path)
        min_y = min(min(s.y(), e.y()) for s, e in self.path)
        max_y = max(max(s.y(), e.y()) for s, e in self.path)
        width = max_x - min_x
        height = max_y - min_y
        if width == 0 or height == 0:
            return

        scale_x = (self.width() - 100) / width
        scale_y = (self.height() - 100) / height
        scale = min(scale_x, scale_y)
        offset_x = (self.width() - width * scale) / 2 - min_x * scale
        offset_y = (self.height() - height * scale) / 2 - min_y * scale

        for i, (start, end) in enumerate(self.path):
            progress = i / len(self.path) if self.path else 0
            r = int(self.start_color.red() + (self.end_color.red() - self.start_color.red()) * progress)
            g = int(self.start_color.green() + (self.end_color.green() - self.start_color.green()) * progress)
            b = int(self.start_color.blue() + (self.end_color.blue() - self.start_color.blue()) * progress)
            pen = QPen(QColor(r, g, b))
            pen.setWidth(max(1, int(2 * scale)))
            painter.setPen(pen)
            sx = int(start.x() * scale + offset_x)
            sy = int(start.y() * scale + offset_y)
            ex = int(end.x() * scale + offset_x)
            ey = int(end.y() * scale + offset_y)
            painter.drawLine(sx, sy, ex, ey)

    def save_image(self, filename):
        if not self.path:
            QMessageBox.warning(self, "警告", "没有可保存的图形")
            return False
        image = QImage(self.size(), QImage.Format_ARGB32)
        image.fill(self.bg_color)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        # 复用 paint_path 方法
        self.paint_path(painter)
        painter.end()
        return image.save(filename)

    def paint_path(self, painter):
        """在指定 painter 上绘制（用于保存）"""
        if not self.path:
            return
        # 与 paintEvent 相同的坐标计算逻辑
        min_x = min(min(s.x(), e.x()) for s, e in self.path)
        max_x = max(max(s.x(), e.x()) for s, e in self.path)
        min_y = min(min(s.y(), e.y()) for s, e in self.path)
        max_y = max(max(s.y(), e.y()) for s, e in self.path)
        width = max_x - min_x
        height = max_y - min_y
        if width == 0 or height == 0:
            return
        scale_x = (self.width() - 100) / width
        scale_y = (self.height() - 100) / height
        scale = min(scale_x, scale_y)
        offset_x = (self.width() - width * scale) / 2 - min_x * scale
        offset_y = (self.height() - height * scale) / 2 - min_y * scale

        for i, (start, end) in enumerate(self.path):
            progress = i / len(self.path) if self.path else 0
            r = int(self.start_color.red() + (self.end_color.red() - self.start_color.red()) * progress)
            g = int(self.start_color.green() + (self.end_color.green() - self.start_color.green()) * progress)
            b = int(self.start_color.blue() + (self.end_color.blue() - self.start_color.blue()) * progress)
            pen = QPen(QColor(r, g, b))
            pen.setWidth(max(1, int(2 * scale)))
            painter.setPen(pen)
            sx = int(start.x() * scale + offset_x)
            sy = int(start.y() * scale + offset_y)
            ex = int(end.x() * scale + offset_x)
            ey = int(end.y() * scale + offset_y)
            painter.drawLine(sx, sy, ex, ey)


class LSystemWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("L-system 分形生成器")
        self.setGeometry(100, 100, 900, 600)

        # 初始化画布
        self.canvas = LSystemWidget()

        # 初始化 UI
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
        vbox.addWidget(QLabel("选择预设系统:"))
        self.system_combo = QComboBox()
        self.system_combo.currentTextChanged.connect(self.on_system_changed)
        vbox.addWidget(self.system_combo)
        layout.addWidget(group)

    def create_parameters_and_colors_section(self, layout):
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(10)

        # 参数组
        params_group = QGroupBox("参数设置")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(6)

        # 迭代次数
        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("迭代:"))
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 20)
        self.iterations_spin.setValue(4)
        iter_row.addWidget(self.iterations_spin)
        params_layout.addLayout(iter_row)

        # 角度
        ang_row = QHBoxLayout()
        ang_row.addWidget(QLabel("角度:"))
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(1, 180)
        self.angle_spin.setValue(25)
        self.angle_spin.setSuffix("°")
        ang_row.addWidget(self.angle_spin)
        params_layout.addLayout(ang_row)

        # 步长
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

        # 起始颜色
        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("起始:"))
        self.start_color_btn = ColorButton(self.canvas.start_color)
        self.start_color_btn.colorChanged.connect(self.on_start_color_changed)
        start_row.addWidget(self.start_color_btn)
        colors_layout.addLayout(start_row)

        # 结束颜色
        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("结束:"))
        self.end_color_btn = ColorButton(self.canvas.end_color)
        self.end_color_btn.colorChanged.connect(self.on_end_color_changed)
        end_row.addWidget(self.end_color_btn)
        colors_layout.addLayout(end_row)

        # 背景颜色
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("背景:"))
        self.bg_color_btn = ColorButton(self.canvas.bg_color)
        self.bg_color_btn.colorChanged.connect(self.on_bg_color_changed)
        bg_row.addWidget(self.bg_color_btn)
        colors_layout.addLayout(bg_row)

        hbox.addWidget(colors_group)
        layout.addWidget(container)

    def create_rules_section(self, layout):
        group = QGroupBox("L-system规则")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)
        vbox.addWidget(QLabel("公理:"))
        self.axiom_edit = QTextEdit()
        self.axiom_edit.setMaximumHeight(35)
        self.axiom_edit.setPlaceholderText("输入初始字符串，如: F")
        vbox.addWidget(self.axiom_edit)

        vbox.addWidget(QLabel("规则 (每行一个，格式: 前驱->后继):"))
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
            "Sierpinski三角": {"axiom": "F", "rules": "F->G-F-G\nG->F+G+F", "angle": 60, "distance": 10, "iterations": 10},
            "分形植物": {"axiom": "X", "rules": "X->F+[[X]-X]-F[-FX]+X\nF->FF", "angle": 25, "distance": 5, "iterations": 6},
            "龙形曲线": {"axiom": "FX", "rules": "X->X+YF+\nY->-FX-Y", "angle": 90, "distance": 5, "iterations": 10},
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
        lsys = LSystem()
        lsys.set_axiom(self.axiom_edit.toPlainText())
        for line in self.rules_edit.toPlainText().split('\n'):
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
        self.canvas.path = []
        self.canvas.update()

    # 颜色改变槽
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