from custom_import import *
from PySide6.QtCore import QLineF

# ==================== 经典预设图案 ====================
# '.' 或 ' ' 表示死亡，其余字符表示存活
PRESETS = {
    "滑翔机 Glider": [
        ".O.",
        "..O",
        "OOO",
    ],
    "轻型飞船 LWSS": [
        ".O..O",
        "....O",
        "O...O",
        ".OOOO",
    ],
    "高斯帕滑翔机枪 Gosper Gun": [
        "........................O...........",
        "......................O.O...........",
        "............OO......OO............OO",
        "...........O...O....OO............OO",
        "OO........O.....O...OO..............",
        "OO........O...O.OO....O.O...........",
        "..........O.....O.......O...........",
        "...........O...O....................",
        "............OO......................",
    ],
    "脉冲星 Pulsar": [
        "..OOO...OOO..",
        ".............",
        "O....O.O....O",
        "O....O.O....O",
        "O....O.O....O",
        "..OOO...OOO..",
        ".............",
        "..OOO...OOO..",
        "O....O.O....O",
        "O....O.O....O",
        "O....O.O....O",
        ".............",
        "..OOO...OOO..",
    ],
    "R 五连 R-pentomino": [
        ".OO",
        "OO.",
        ".O.",
    ],
    "橡子 Acorn": [
        ".O.....",
        "...O...",
        "OO..OOO",
    ],
    "顽固分子 Diehard": [
        "......O.",
        "OO......",
        ".O...OOO",
    ],
    "无限增长 Infinite Growth": [
        "OOO.O",
        "O....",
        "...OO",
        ".OO.O",
        "O.O.O",
    ],
    "蜂巢 Beehive": [
        ".OO.",
        "O..O",
        ".OO.",
    ],
    "面包 Loaf": [
        ".OO.",
        "O..O",
        ".O.O",
        "..O.",
    ],
    "闪光灯 Blinker": [
        "OOO",
    ],
    "蟾蜍 Toad": [
        ".OOO",
        "OOO.",
    ],
    "方块 Block": [
        "OO",
        "OO",
    ],
    "飞船队 Fleet": [
        ".O..O......",
        "....O......",
        "O...O......",
        ".OOOO......",
        "...........",
        "......O..O.",
        ".........O.",
        "......O...O",
        "......OOOO.",
    ],
}


class LifeAutomaton:
    """稠密 numpy 网格上的生命游戏，边界环绕。"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=bool)
        self.generation = 0
        self.trail = np.zeros((height, width), dtype=np.float32)
        self.trail_decay = 0.90

    # ---------- 状态操作 ----------
    def clear(self):
        self.grid[:] = False
        self.trail[:] = 0.0
        self.generation = 0

    def set_cell(self, x, y, state):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = bool(state)

    def get_cell(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return 1 if self.grid[y, x] else 0
        return 0

    def toggle_cell(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = not self.grid[y, x]

    def resize(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=bool)
        self.trail = np.zeros((height, width), dtype=np.float32)
        self.generation = 0

    def randomize(self, density=0.28, margin=0):
        h, w = self.grid.shape
        rnd = np.random.random((h, w)) < density
        mask = np.zeros((h, w), dtype=bool)
        mask[margin:h - margin, margin:w - margin] = True
        self.grid[:] = rnd & mask
        self.trail[self.grid] = 1.0
        self.generation = 0

    def place_pattern(self, rows, cx, cy):
        """把字符图案以 (cx, cy) 为中心放置，返回实际放置的细胞数。"""
        pat_h = len(rows)
        pat_w = max(len(r) for r in rows)
        x0 = cx - pat_w // 2
        y0 = cy - pat_h // 2
        count = 0
        for j, row in enumerate(rows):
            for i, ch in enumerate(row):
                if ch in ". ":
                    continue
                x, y = x0 + i, y0 + j
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.grid[y, x] = True
                    count += 1
        self.trail[self.grid] = 1.0
        return count

    # ---------- 演化 ----------
    def step(self):
        g = self.grid
        h, w = g.shape
        padded = np.pad(g, 1, mode="wrap")
        n = padded[0:h, 0:w].astype(np.uint8)
        n += padded[0:h, 1:w + 1]
        n += padded[0:h, 2:w + 2]
        n += padded[1:h + 1, 0:w]
        n += padded[1:h + 1, 2:w + 2]
        n += padded[2:h + 2, 0:w]
        n += padded[2:h + 2, 1:w + 1]
        n += padded[2:h + 2, 2:w + 2]
        alive = n == 3
        alive |= g & (n == 2)
        self.grid = alive
        self.generation += 1
        self.trail *= self.trail_decay
        self.trail[self.grid] = 1.0

    @property
    def population(self):
        return int(self.grid.sum())


# ==================== 画布 ====================
class LifeCanvas(QWidget):
    """整块画布用一个缩放 QImage 绘制，单帧开销与细胞数无关。"""

    def __init__(self, automaton, cell_size=2, main_window=None):
        super().__init__()
        self.automaton = automaton
        self.cell_size = cell_size
        self.main_window = main_window
        self.setMouseTracking(True)
        self.draw_mode = 1
        self.drawing = False
        self.dragging = False
        self.last_drag_pos = None
        self.last_cell = None

        self.alive_color = QColor(0, 128, 255)
        self.dead_color = QColor(248, 248, 248)
        self.grid_color = QColor(210, 210, 210)
        self.show_grid = True
        self.show_trail = False
        self._lut = None
        self._image = None
        self._dirty = True
        self._update_size()

    # ---------- 尺寸与图像 ----------
    def _update_size(self):
        self.setFixedSize(self.automaton.width * self.cell_size,
                          self.automaton.height * self.cell_size)

    def invalidate(self):
        self._dirty = True
        self.update()

    def set_colors(self, alive, dead):
        self.alive_color = alive
        self.dead_color = dead
        self._lut = None
        self._dirty = True
        self.update()

    def _build_lut(self):
        """256 级 LUT：把轨迹强度映射成颜色，避免每帧做浮点混合。"""
        lut = np.empty((256, 3), dtype=np.uint8)
        ar, ag, ab = self.alive_color.red(), self.alive_color.green(), self.alive_color.blue()
        dr, dg, db = self.dead_color.red(), self.dead_color.green(), self.dead_color.blue()
        t = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None]
        lut[:, 0] = np.clip(dr + (ar - dr) * t[:, 0], 0, 255)
        lut[:, 1] = np.clip(dg + (ag - dg) * t[:, 0], 0, 255)
        lut[:, 2] = np.clip(db + (ab - db) * t[:, 0], 0, 255)
        self._lut = lut

    def _render_image(self):
        if self._lut is None:
            self._build_lut()
        if self.show_trail:
            idx = (np.clip(self.automaton.trail, 0, 1) * 255).astype(np.uint8)
        else:
            idx = np.where(self.automaton.grid, 255, 0).astype(np.uint8)
        rgb = self._lut[idx]
        h, w = idx.shape
        img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self._image = img.copy()
        self._dirty = False

    def paintEvent(self, event):
        if self._dirty or self._image is None:
            self._render_image()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(self.rect(), self._image)

        if self.show_grid and self.cell_size >= 6:
            self._draw_grid(painter, event.rect())
        painter.end()

    def _draw_grid(self, painter, rect):
        cs = self.cell_size
        x0 = max(0, rect.left() // cs)
        x1 = min(self.automaton.width, rect.right() // cs + 1)
        y0 = max(0, rect.top() // cs)
        y1 = min(self.automaton.height, rect.bottom() // cs + 1)
        painter.setPen(QPen(self.grid_color, 1))
        lines = []
        for x in range(x0, x1 + 1):
            lines.append(QLineF(x * cs, y0 * cs, x * cs, y1 * cs))
        for y in range(y0, y1 + 1):
            lines.append(QLineF(x0 * cs, y * cs, x1 * cs, y * cs))
        painter.drawLines(lines)

    # ---------- 鼠标 ----------
    def _cell_at(self, pos):
        return pos.x() // self.cell_size, pos.y() // self.cell_size

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            x, y = self._cell_at(event.position().toPoint())
            self._paint_cell(x, y)
            self.last_cell = (x, y)
        elif event.button() == Qt.MiddleButton:
            self.dragging = True
            self.last_drag_pos = event.globalPosition().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self.drawing:
            x, y = self._cell_at(event.position().toPoint())
            if self.last_cell and (x, y) != self.last_cell:
                self._paint_line(self.last_cell, (x, y))
                self.last_cell = (x, y)
            elif not self.last_cell:
                self._paint_cell(x, y)
                self.last_cell = (x, y)
        elif self.dragging and self.last_drag_pos:
            pos = event.globalPosition().toPoint()
            delta = pos - self.last_drag_pos
            if self.main_window:
                self.main_window.pan_view(delta.x(), delta.y())
            self.last_drag_pos = pos
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_cell = None
        elif event.button() == Qt.MiddleButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event):
        if self.main_window:
            if event.angleDelta().y() > 0:
                self.main_window.zoom_by(1)
            else:
                self.main_window.zoom_by(-1)
        event.accept()

    def _paint_cell(self, x, y):
        if self.draw_mode == 1:
            self.automaton.set_cell(x, y, True)
        else:
            self.automaton.set_cell(x, y, False)
        self.invalidate()

    def _paint_line(self, start, end):
        x0, y0 = start
        x1, y1 = end
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            self._paint_cell(x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("生命游戏 Conway's Game of Life")
        self.setGeometry(120, 60, 1120, 860)

        self.grid_width = 400
        self.grid_height = 400
        self.cell_size = 2
        self.fps = 30
        self.automaton = LifeAutomaton(self.grid_width, self.grid_height)

        self._fps_frames = 0
        self._fps_time = time.perf_counter()

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_automaton)

        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(500)

        # 默认放一个高斯帕滑翔机枪，打开就能看到东西
        self.automaton.place_pattern(PRESETS["高斯帕滑翔机枪 Gosper Gun"],
                                     self.grid_width // 2, self.grid_height // 2)
        self.canvas.invalidate()
        self.update_stats()

    # ---------- UI ----------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(270)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(6)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = LifeCanvas(self.automaton, self.cell_size, self)

        self._build_preset_group(panel_layout)
        self._build_sim_group(panel_layout)
        self._build_display_group(panel_layout)
        self._build_grid_group(panel_layout)
        panel_layout.addStretch()

        root.addWidget(panel)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.canvas)
        root.addWidget(self.scroll_area, 1)

        # 统计放到状态栏，面板留给操作
        self.lbl_status = QLabel("就绪")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_generation = QLabel("代数 0")
        self.lbl_population = QLabel("存活 0")
        self.lbl_fps = QLabel("帧率 –")
        for lbl in (self.lbl_generation, self.lbl_population, self.lbl_fps):
            self.statusBar().addPermanentWidget(lbl)

    def _build_preset_group(self, layout):
        group = QGroupBox("预设图案")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        vbox.addWidget(self.preset_combo)

        row = QHBoxLayout()
        row.setSpacing(4)
        place_btn = QPushButton("放置到中心")
        place_btn.clicked.connect(self.place_preset)
        row.addWidget(place_btn, 1)
        vbox.addLayout(row)

        row2 = QHBoxLayout()
        row2.setSpacing(4)
        rand_btn = QPushButton("随机")
        rand_btn.clicked.connect(self.random_fill)
        row2.addWidget(rand_btn)
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.clear_grid)
        row2.addWidget(clear_btn)
        vbox.addLayout(row2)

        layout.addWidget(group)

    def _build_sim_group(self, layout):
        group = QGroupBox("模拟")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.run_btn = QPushButton("开始")
        self.run_btn.clicked.connect(self.toggle_simulation)
        row.addWidget(self.run_btn, 1)
        step_btn = QPushButton("单步")
        step_btn.clicked.connect(self.single_step)
        row.addWidget(step_btn, 1)
        vbox.addLayout(row)

        form = QFormLayout()
        form.setSpacing(4)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 120)
        self.speed_slider.setValue(self.fps)
        self.speed_slider.valueChanged.connect(self.set_fps)
        self.speed_label = QLabel(f"{self.fps}")
        self.speed_label.setFixedWidth(34)
        self.speed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        speed_row = QHBoxLayout()
        speed_row.setSpacing(4)
        speed_row.addWidget(self.speed_slider, 1)
        speed_row.addWidget(self.speed_label)
        form.addRow("速度", speed_row)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 20)
        self.zoom_slider.setValue(self.cell_size)
        self.zoom_slider.valueChanged.connect(self.set_zoom)
        self.zoom_label = QLabel(f"{self.cell_size}px")
        self.zoom_label.setFixedWidth(34)
        self.zoom_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(4)
        zoom_row.addWidget(self.zoom_slider, 1)
        zoom_row.addWidget(self.zoom_label)
        form.addRow("缩放", zoom_row)

        vbox.addLayout(form)
        layout.addWidget(group)

    def _build_display_group(self, layout):
        group = QGroupBox("显示")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)

        self.grid_check = QCheckBox("网格线（缩放 ≥ 6px 时显示）")
        self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(self.toggle_grid)
        vbox.addWidget(self.grid_check)

        self.trail_check = QCheckBox("残影轨迹")
        self.trail_check.setChecked(False)
        self.trail_check.toggled.connect(self.toggle_trail)
        vbox.addWidget(self.trail_check)

        color_row = QHBoxLayout()
        color_row.setSpacing(6)
        color_row.addWidget(QLabel("颜色"))
        self.alive_btn = ColorButton(self.canvas.alive_color)
        self.alive_btn.colorChanged.connect(self.on_alive_color)
        color_row.addWidget(self.alive_btn)
        self.dead_btn = ColorButton(self.canvas.dead_color)
        self.dead_btn.colorChanged.connect(self.on_dead_color)
        color_row.addWidget(self.dead_btn)
        color_row.addStretch()
        vbox.addLayout(color_row)

        layout.addWidget(group)

    def _build_grid_group(self, layout):
        group = QGroupBox("网格")
        form = QFormLayout(group)
        form.setSpacing(4)
        form.setContentsMargins(6, 6, 6, 6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(50, 1000)
        self.size_spin.setSingleStep(50)
        self.size_spin.setValue(self.grid_width)
        self.size_spin.editingFinished.connect(self.apply_grid_size)
        form.addRow("边长", self.size_spin)

        hint = QLabel("滚轮缩放，中键拖动")
        hint.setStyleSheet("color: #888;")
        form.addRow("", hint)

        layout.addWidget(group)

    # ---------- 操作 ----------
    def place_preset(self):
        name = self.preset_combo.currentText()
        rows = PRESETS.get(name)
        if not rows:
            return
        self.automaton.place_pattern(rows, self.grid_width // 2, self.grid_height // 2)
        self.canvas.invalidate()
        self.update_stats()
        self.lbl_status.setText(f"已放置：{name}")

    def random_fill(self):
        self.automaton.randomize(0.28)
        self.canvas.invalidate()
        self.update_stats()

    def clear_grid(self):
        self.automaton.clear()
        self.canvas.invalidate()
        self.update_stats()
        self.lbl_status.setText("网格已清除")

    def toggle_simulation(self):
        if self.timer.isActive():
            self.timer.stop()
            self.run_btn.setText("开始")
            self.lbl_status.setText("已暂停")
        else:
            self.timer.start(max(1, 1000 // self.fps))
            self.run_btn.setText("暂停")
            self.lbl_status.setText("运行中…")

    def single_step(self):
        self.automaton.step()
        self.canvas.invalidate()
        self.update_stats()

    def update_automaton(self):
        self.automaton.step()
        self.canvas.invalidate()
        self._fps_frames += 1

    def set_fps(self, value):
        self.fps = value
        self.speed_label.setText(str(value))
        if self.timer.isActive():
            self.timer.start(max(1, 1000 // value))

    def set_zoom(self, value):
        self.cell_size = value
        self.zoom_label.setText(f"{value}px")
        self.zoom_by(0)

    def zoom_by(self, delta):
        new_size = max(1, min(20, self.cell_size + delta))
        if new_size != self.cell_size:
            self.cell_size = new_size
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(new_size)
            self.zoom_slider.blockSignals(False)
            self.zoom_label.setText(f"{new_size}px")
        self._apply_canvas_size()

    def _apply_canvas_size(self):
        # 保持视口中心不动
        viewport = self.scroll_area.viewport()
        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()
        cx = (hbar.value() + viewport.width() / 2) / max(1, self.canvas.cell_size)
        cy = (vbar.value() + viewport.height() / 2) / max(1, self.canvas.cell_size)
        self.canvas.cell_size = self.cell_size
        self.canvas._update_size()
        self.canvas.invalidate()
        hbar.setValue(int(cx * self.cell_size - viewport.width() / 2))
        vbar.setValue(int(cy * self.cell_size - viewport.height() / 2))

    def toggle_grid(self, checked):
        self.canvas.show_grid = bool(checked)
        self.canvas.invalidate()

    def toggle_trail(self, checked):
        self.canvas.show_trail = bool(checked)
        if not checked:
            self.automaton.trail[:] = 0.0
        self.canvas.invalidate()

    def on_alive_color(self, color):
        self.canvas.set_colors(color, self.canvas.dead_color)

    def on_dead_color(self, color):
        self.canvas.set_colors(self.canvas.alive_color, color)

    def apply_grid_size(self):
        size = self.size_spin.value()
        if size == self.grid_width:
            return
        self.grid_width = self.grid_height = size
        self.automaton.resize(size, size)
        self.canvas._update_size()
        self.canvas.invalidate()
        self.update_stats()
        self.lbl_status.setText(f"网格已重置为 {size}×{size}")

    def pan_view(self, dx, dy):
        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()
        hbar.setValue(hbar.value() - dx)
        vbar.setValue(vbar.value() - dy)

    def update_stats(self):
        self.lbl_generation.setText(f"{self.automaton.generation:,}")
        self.lbl_population.setText(f"{self.automaton.population:,}")
        now = time.perf_counter()
        if now - self._fps_time >= 0.5:
            self.lbl_fps.setText(f"{self._fps_frames / (now - self._fps_time):.0f}")
            self._fps_frames = 0
            self._fps_time = now

    def closeEvent(self, event):
        self.timer.stop()
        self.stats_timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
