from custom_import import *
from PySide6.QtCore import QLineF

# ========== L-system 核心 ==========
class LSystem:
    """优化后的 L-system，字符串生成使用列表拼接，O(n) 复杂度"""
    MAX_CHARS = 1_200_000      # 超过就截断，避免指数爆炸把进程拖死

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

    def generate(self, generations=None, max_chars=None):
        """返回 (字符串, 是否被截断)。"""
        if generations is None:
            generations = self.generations
        cap = max_chars or self.MAX_CHARS
        result = self.axiom
        truncated = False
        rules = self.rules
        for _ in range(generations):
            parts = []
            total = 0
            for char in result:
                s = rules.get(char, char)
                parts.append(s)
                total += len(s)
                if total > cap:
                    truncated = True
                    break
            result = ''.join(parts)
            if truncated:
                break
        return result, truncated

    def set_parameters(self, angle, distance, generations):
        self.angle = angle
        self.distance = distance
        self.generations = generations

# ========== 后台生成线程 ==========
class LSystemGenerator(QThread):
    """在后台计算 L-system 路径，避免阻塞 GUI"""
    pathReady = Signal(object)   # dict: path/bbox/chars/truncated/elapsed


    def __init__(self, lsystem, start_x, start_y, parent=None):
        super().__init__(parent)
        self.lsystem = lsystem
        self.start_x = start_x
        self.start_y = start_y
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        t0 = time.perf_counter()
        string, truncated = self.lsystem.generate()
        if self._cancelled:
            return
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

        for index, char in enumerate(string):
            if index % 4096 == 0 and self._cancelled:
                return
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
        self.pathReady.emit({
            "path": path,
            "bbox": (min_x, min_y, max_x, max_y),
            "chars": len(string),
            "truncated": truncated,
            "elapsed": time.perf_counter() - t0,
        })

# ========== L-system 绘图部件 ==========
class LSystemWidget(QWidget):
    """优化后的绘图部件：缓存路径、包围盒、缩放因子；支持后台生成"""
    statsReady = Signal(int, int, bool, float)   # 线段数, 字符串长度, 是否截断, 耗时(秒)

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
        if self._generator and self._generator.isRunning():
            self._generator.cancel()
            self._generator.wait(2000)
        self.lsystem = lsystem
        self._start_background_generation()

    def shutdown(self):
        """窗口关闭时停止后台生成线程。"""
        if self._generator and self._generator.isRunning():
            self._generator.cancel()
            self._generator.wait(2000)

    def _start_background_generation(self):
        """启动后台线程计算路径"""
        # 起点设置为画布中心底部（与原始逻辑一致）
        start_x = self.width() / 2.0
        start_y = self.height() - 50.0
        self._generator = LSystemGenerator(self.lsystem, start_x, start_y)
        self._generator.pathReady.connect(self._on_path_ready)
        self._generator.start()

    def _on_path_ready(self, result):
        """后台计算完成，更新路径并重绘"""
        path = result["path"]
        min_x, min_y, max_x, max_y = result["bbox"]
        self.path = path
        self.bounding_rect = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)
        self._update_transform()
        self.update()
        self.statsReady.emit(len(path), int(result["chars"]),
                             bool(result["truncated"]), float(result["elapsed"]))

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
        """公共绘制方法，供 paintEvent 和 save_image 复用

        颜色按进度分成若干档，同档线段一次性 drawLines —— 几十万条线段也不会卡。
        """
        if not self.path:
            return
        scale = self.scale
        off_x = self.offset_x
        off_y = self.offset_y
        path_len = len(self.path)
        sr, sg, sb = self.start_color.red(), self.start_color.green(), self.start_color.blue()
        er, eg, eb = self.end_color.red(), self.end_color.green(), self.end_color.blue()

        pen = QPen()
        pen.setWidth(max(1, min(3, int(round(2 * scale)))))
        buckets = 48
        batch = []
        current = -1

        for i, ((x1, y1), (x2, y2)) in enumerate(self.path):
            t = i / (path_len - 1) if path_len > 1 else 0.0
            b = int(t * (buckets - 1))
            if b != current:
                if batch:
                    painter.setPen(pen)
                    painter.drawLines(batch)
                    batch = []
                k = b / (buckets - 1)
                pen.setColor(QColor(int(sr + (er - sr) * k),
                                    int(sg + (eg - sg) * k),
                                    int(sb + (eb - sb) * k)))
                current = b
            batch.append(QLineF(x1 * scale + off_x, y1 * scale + off_y,
                                x2 * scale + off_x, y2 * scale + off_y))

        if batch:
            painter.setPen(pen)
            painter.drawLines(batch)

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
        self.setWindowTitle("L-system 分形生成器")
        self.setGeometry(100, 100, 1000, 680)

        self.canvas = LSystemWidget()
        self.canvas.statsReady.connect(self.on_stats)
        self.init_ui()
        self.setup_default_systems()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(310)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(6)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        self.create_system_selection(panel_layout)
        self.create_parameters_section(panel_layout)
        self.create_colors_section(panel_layout)
        self.create_rules_section(panel_layout)
        self.create_buttons_section(panel_layout)
        self.create_stats_section(panel_layout)
        panel_layout.addStretch()

        main_layout.addWidget(panel)
        main_layout.addWidget(self.canvas, 1)

        bar = self.statusBar()
        bar.setSizeGripEnabled(False)
        self.lbl_status = QLabel("就绪")
        bar.addWidget(self.lbl_status)

    def create_system_selection(self, layout):
        group = QGroupBox("预设系统")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)
        self.system_combo = QComboBox()
        self.system_combo.currentTextChanged.connect(self.on_system_changed)
        vbox.addWidget(self.system_combo)
        layout.addWidget(group)

    def create_parameters_section(self, layout):
        group = QGroupBox("参数")
        form = QFormLayout()
        form.setSpacing(3)

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 12)
        self.iterations_spin.setValue(4)
        form.addRow("迭代次数:", self.iterations_spin)

        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(1, 180)
        self.angle_spin.setValue(25)
        self.angle_spin.setSuffix("°")
        self.angle_spin.setDecimals(1)
        form.addRow("转角:", self.angle_spin)

        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(1, 50)
        self.distance_spin.setValue(10)
        form.addRow("步长:", self.distance_spin)

        group.setLayout(form)
        layout.addWidget(group)

    def create_colors_section(self, layout):
        group = QGroupBox("颜色")
        hbox = QHBoxLayout(group)
        hbox.setSpacing(6)
        hbox.addWidget(QLabel("起始"))
        self.start_color_btn = ColorButton(self.canvas.start_color)
        self.start_color_btn.colorChanged.connect(self.on_start_color_changed)
        hbox.addWidget(self.start_color_btn)
        hbox.addWidget(QLabel("结束"))
        self.end_color_btn = ColorButton(self.canvas.end_color)
        self.end_color_btn.colorChanged.connect(self.on_end_color_changed)
        hbox.addWidget(self.end_color_btn)
        hbox.addWidget(QLabel("背景"))
        self.bg_color_btn = ColorButton(self.canvas.bg_color)
        self.bg_color_btn.colorChanged.connect(self.on_bg_color_changed)
        hbox.addWidget(self.bg_color_btn)
        hbox.addStretch()
        layout.addWidget(group)

    def create_rules_section(self, layout):
        group = QGroupBox("L-system 规则")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)

        vbox.addWidget(QLabel("公理:"))
        self.axiom_edit = QTextEdit()
        self.axiom_edit.setMaximumHeight(32)
        self.axiom_edit.setPlaceholderText("例如: F")
        vbox.addWidget(self.axiom_edit)

        hint = QLabel("规则每行一条：前驱->后继；F 前进、+/- 转向、[ ] 入栈出栈")
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        vbox.addWidget(hint)
        self.rules_edit = QTextEdit()
        self.rules_edit.setPlaceholderText("例如: F->F+F-F-F+F")
        vbox.addWidget(self.rules_edit, 1)
        layout.addWidget(group, 1)

    def create_buttons_section(self, layout):
        group = QGroupBox("操作")
        hbox = QHBoxLayout(group)
        hbox.setSpacing(6)

        self.generate_btn = QPushButton("生成")
        self.generate_btn.clicked.connect(self.generate_system)
        self.reset_btn = QPushButton("恢复预设")
        self.reset_btn.clicked.connect(
            lambda: self.on_system_changed(self.system_combo.currentText()))
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_image)

        for b in (self.generate_btn, self.reset_btn, self.save_btn):
            hbox.addWidget(b)
        layout.addWidget(group)

    def create_stats_section(self, layout):
        group = QGroupBox("统计")
        form = QFormLayout()
        form.setSpacing(3)
        self.lbl_length = QLabel("–")
        self.lbl_segments = QLabel("–")
        self.lbl_time = QLabel("–")
        form.addRow("字符串长度:", self.lbl_length)
        form.addRow("线段数:", self.lbl_segments)
        form.addRow("生成耗时:", self.lbl_time)
        group.setLayout(form)
        layout.addWidget(group)

    def on_stats(self, segments, chars, truncated, elapsed):
        self.lbl_length.setText(f"{chars:,}")
        self.lbl_segments.setText(f"{segments:,}")
        self.lbl_time.setText(f"{elapsed * 1000:.0f} ms")
        self.generate_btn.setEnabled(True)
        if truncated:
            self.lbl_status.setText(
                f"已完成，但字符串超过 {LSystem.MAX_CHARS:,} 字符被截断——降低迭代次数或简化规则")
        else:
            self.lbl_status.setText(f"完成：{segments:,} 条线段，{elapsed * 1000:.0f} ms")

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
        axiom = self.axiom_edit.toPlainText().strip()
        if not axiom:
            self.lbl_status.setText("公理为空，什么都不画")
            return
        lsys.set_axiom(axiom)
        rules_text = self.rules_edit.toPlainText().strip()
        n_rules = 0
        for line in rules_text.split('\n'):
            if '->' in line:
                pred, succ = line.split('->', 1)
                if pred.strip():
                    lsys.add_rule(pred.strip(), succ.strip())
                    n_rules += 1
        lsys.set_parameters(
            self.angle_spin.value(),
            self.distance_spin.value(),
            self.iterations_spin.value()
        )
        self.generate_btn.setEnabled(False)
        self.lbl_status.setText(
            f"生成中…（{self.iterations_spin.value()} 代，{n_rules} 条规则）")
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

    def closeEvent(self, event):
        self.canvas.shutdown()
        super().closeEvent(event)

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