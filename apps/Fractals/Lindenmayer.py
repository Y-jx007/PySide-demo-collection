from custom_import import *
from array import array
from PySide6.QtCore import QLineF

# ========== L-system 核心 ==========
class LSystem:
    """优化后的 L-system，字符串生成使用列表拼接，O(n) 复杂度"""
    MAX_CHARS = 6_000_000      # 超过就截断：路径用 numpy 存，600 万字符约 200MB 以内

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

        # 用 array('f') 累积坐标，最后零拷贝转成 numpy（比逐行写数组快得多）
        buf = array('f')
        seg = 0
        stack = []
        x, y = self.start_x, self.start_y

        # 用方向向量代替每步求 sin/cos：转角固定，只需一次三角函数
        step_angle = math.radians(self.lsystem.angle)
        cs = math.cos(step_angle)
        sn = math.sin(step_angle)
        distance = self.lsystem.distance
        dx, dy = 0.0, -1.0          # 初始朝上（屏幕坐标 y 向下）

        min_x = max_x = x
        min_y = max_y = y

        for index, char in enumerate(string):
            if index % 4096 == 0 and self._cancelled:
                return
            if char == 'F':
                new_x = x + dx * distance
                new_y = y + dy * distance
                if new_x < min_x: min_x = new_x
                if new_x > max_x: max_x = new_x
                if new_y < min_y: min_y = new_y
                if new_y > max_y: max_y = new_y
                buf.extend((x, y, new_x, new_y))
                x, y = new_x, new_y
            elif char == 'f':
                x += dx * distance
                y += dy * distance
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y
            elif char == '+':
                dx, dy = dx * cs - dy * sn, dx * sn + dy * cs
            elif char == '-':
                dx, dy = dx * cs + dy * sn, -dx * sn + dy * cs
            elif char == '[':
                stack.append((x, y, dx, dy))
            elif char == ']':
                if stack:
                    x, y, dx, dy = stack.pop()

        path = np.frombuffer(buf, dtype=np.float32).reshape(-1, 4).copy()
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
        self.path = np.empty((0, 4), dtype=np.float32)   # 每行 (x1,y1,x2,y2)
        self.bounding_rect = QRectF()  # 包围盒
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._pixmap = None             # 渲染缓存

        self.start_color = QColor(139, 69, 19)
        self.end_color = QColor(34, 139, 34)
        self.bg_color = QColor(255, 255, 255)

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
        self._pixmap = None
        self.update()
        self.statsReady.emit(int(path.shape[0]), int(result["chars"]),
                             bool(result["truncated"]), float(result["elapsed"]))

    def _update_transform(self):
        """根据当前包围盒和窗口大小计算缩放和平移（居中）"""
        if self.path.shape[0] == 0:
            return
        rect = self.bounding_rect
        if rect.width() < 0.01 or rect.height() < 0.01:
            return
        margin = 24
        avail_w = max(1.0, self.width() - 2 * margin)
        avail_h = max(1.0, self.height() - 2 * margin)
        scale_x = avail_w / rect.width() if rect.width() > 0 else 1.0
        scale_y = avail_h / rect.height() if rect.height() > 0 else 1.0
        self.scale = min(scale_x, scale_y)
        draw_w = rect.width() * self.scale
        draw_h = rect.height() * self.scale
        self.offset_x = (self.width() - draw_w) / 2.0 - rect.left() * self.scale
        self.offset_y = (self.height() - draw_h) / 2.0 - rect.top() * self.scale

    def set_colors(self, start, end, bg):
        self.start_color = start
        self.end_color = end
        self.bg_color = bg
        self._pixmap = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.bg_color)
        if self.path.shape[0] == 0:
            return
        if self._pixmap is None or self._pixmap.size() != self.size():
            self._pixmap = self._render_pixmap()
        painter.drawPixmap(0, 0, self._pixmap)

    def _render_pixmap(self):
        """线段很多时用 numpy 直接打点（快两个数量级），否则走矢量线（更细腻）。"""
        n = self.path.shape[0]
        if n > 300_000:
            image = self._render_scatter()
            if image is not None:
                return QPixmap.fromImage(image)

        pm = QPixmap(self.size())
        pm.fill(self.bg_color)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_path(painter)
        painter.end()
        return pm

    def _render_scatter(self):
        """把每个线段终点画成一个像素；几十万段以上时肉眼与矢量线几乎无差别。"""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return None
        arr = self.path
        n = arr.shape[0]

        xs = arr[:, 2] * self.scale + self.offset_x
        ys = arr[:, 3] * self.scale + self.offset_y
        px = np.rint(xs).astype(np.int32)
        py = np.rint(ys).astype(np.int32)
        mask = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if not mask.any():
            return None
        idx = np.nonzero(mask)[0]

        t = idx / max(1, n - 1)
        sr, sg, sb = self.start_color.red(), self.start_color.green(), self.start_color.blue()
        er, eg, eb = self.end_color.red(), self.end_color.green(), self.end_color.blue()
        cols = np.empty((idx.size, 3), dtype=np.uint8)
        cols[:, 0] = np.clip(sr + (er - sr) * t, 0, 255)
        cols[:, 1] = np.clip(sg + (eg - sg) * t, 0, 255)
        cols[:, 2] = np.clip(sb + (eb - sb) * t, 0, 255)

        img = np.empty((h, w, 3), dtype=np.uint8)
        img[:, :] = (self.bg_color.red(), self.bg_color.green(), self.bg_color.blue())
        img[py[mask], px[mask]] = cols

        # 起点单独补一个像素，避免整条路径缺头
        sx = int(round(arr[0, 0] * self.scale + self.offset_x))
        sy = int(round(arr[0, 1] * self.scale + self.offset_y))
        if 0 <= sx < w and 0 <= sy < h:
            img[sy, sx] = (sr, sg, sb)

        return QImage(img.data, w, h, w * 3, QImage.Format_RGB888).copy()

    def _draw_path(self, painter):
        """按进度分 48 档颜色，逐档 drawLines；路径来自 numpy 数组，内存很小。"""
        n = self.path.shape[0]
        if n == 0:
            return
        scale = self.scale
        off_x = self.offset_x
        off_y = self.offset_y
        sr, sg, sb = self.start_color.red(), self.start_color.green(), self.start_color.blue()
        er, eg, eb = self.end_color.red(), self.end_color.green(), self.end_color.blue()

        pen = QPen()
        pen.setWidth(max(1, min(3, int(round(2 * scale)))))
        buckets = 48

        for b in range(buckets):
            i0 = b * n // buckets
            i1 = (b + 1) * n // buckets
            if i1 <= i0:
                continue
            chunk = self.path[i0:i1]
            xs = (chunk[:, 0] * scale + off_x).tolist()
            ys = (chunk[:, 1] * scale + off_y).tolist()
            xe = (chunk[:, 2] * scale + off_x).tolist()
            ye = (chunk[:, 3] * scale + off_y).tolist()
            lines = [QLineF(a, c, d, e) for a, c, d, e in zip(xs, ys, xe, ye)]
            k = b / (buckets - 1)
            pen.setColor(QColor(int(sr + (er - sr) * k),
                                int(sg + (eg - sg) * k),
                                int(sb + (eb - sb) * k)))
            painter.setPen(pen)
            painter.drawLines(lines)

    def save_image(self, filename):
        """将当前图形保存为图片文件"""
        if self.path.shape[0] == 0:
            return False
        return self._render_pixmap().save(filename)

    def resizeEvent(self, event):
        """窗口大小变化时重新计算变换"""
        super().resizeEvent(event)
        self._pixmap = None
        if self.path.shape[0] > 0:
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
        self.canvas.path = np.empty((0, 4), dtype=np.float32)
        self.canvas._pixmap = None
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