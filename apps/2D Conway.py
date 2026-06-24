from custom_import import *
from collections import defaultdict

class SparseCellAutomaton:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.live_cells = set()
        self._render_cache = None
        self._cache_dirty = True
        
    def get_cell(self, x, y):
        return 1 if (y, x) in self.live_cells else 0
    
    def set_cell(self, x, y, state):
        if state == 1:
            self.live_cells.add((y, x))
        else:
            self.live_cells.discard((y, x))
        self._cache_dirty = True
        
    def update(self):
        neighbor_count = defaultdict(int)
        for y, x in self.live_cells:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = (x + dx) % self.width, (y + dy) % self.height
                    neighbor_count[(ny, nx)] += 1
        
        new_live_cells = set()
        for (y, x), count in neighbor_count.items():
            if (y, x) in self.live_cells:
                if 2 <= count <= 3:
                    new_live_cells.add((y, x))
            else:
                if count == 3:
                    new_live_cells.add((y, x))
        
        self.live_cells = new_live_cells
        self._cache_dirty = True
    
    def get_render_grid(self):
        if self._render_cache is None or self._cache_dirty:
            self._render_cache = np.zeros((self.height, self.width), dtype=np.int8)
            for y, x in self.live_cells:
                if 0 <= y < self.height and 0 <= x < self.width:
                    self._render_cache[y, x] = 1
            self._cache_dirty = False
        return self._render_cache
    
    def clear(self):
        self.live_cells.clear()
        self._cache_dirty = True


class CellWidget(QWidget):
    def __init__(self, automaton, cell_size=8, main_window=None):
        super().__init__()
        self.automaton = automaton
        self.cell_size = cell_size
        self.main_window = main_window
        self.setMouseTracking(True)
        self.drawing = False
        self.draw_mode = 1
        self.dragging = False
        self.last_drag_pos = None
        self.last_cell_pos = None
        
        self.update_size()
        
        self.alive_color = QColor(0, 128, 255)
        self.dead_color = QColor(240, 240, 240)
        self.grid_color = QColor(200, 200, 200)
        
    def update_size(self):
        self.setFixedSize(
            self.automaton.width * self.cell_size,
            self.automaton.height * self.cell_size
        )
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(event.rect(), self.dead_color)
        
        grid = self.automaton.get_render_grid()
        x_start = max(0, event.rect().x() // self.cell_size)
        y_start = max(0, event.rect().y() // self.cell_size)
        x_end = min(self.automaton.width, (event.rect().right() // self.cell_size) + 1)
        y_end = min(self.automaton.height, (event.rect().bottom() // self.cell_size) + 1)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.alive_color)
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                if grid[y, x] == 1:
                    painter.drawRect(
                        x * self.cell_size, 
                        y * self.cell_size,
                        self.cell_size, 
                        self.cell_size
                    )
        
        painter.setPen(QPen(self.grid_color, 1))
        for x in range(x_start, x_end + 1):
            painter.drawLine(
                x * self.cell_size, y_start * self.cell_size,
                x * self.cell_size, y_end * self.cell_size
            )
        for y in range(y_start, y_end + 1):
            painter.drawLine(
                x_start * self.cell_size, y * self.cell_size,
                x_end * self.cell_size, y * self.cell_size
            )
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            pos = event.position().toPoint()
            self.toggle_cell(pos)
            self.last_cell_pos = (pos.x() // self.cell_size, pos.y() // self.cell_size)
        elif event.button() == Qt.MiddleButton:
            self.dragging = True
            self.last_drag_pos = event.globalPos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
    
    def mouseMoveEvent(self, event):
        if self.drawing:
            pos = event.position().toPoint()
            current_cell_pos = (pos.x() // self.cell_size, pos.y() // self.cell_size)
            if self.last_cell_pos and current_cell_pos != self.last_cell_pos:
                self.draw_line(self.last_cell_pos, current_cell_pos)
                self.last_cell_pos = current_cell_pos
            else:
                self.toggle_cell(pos)
        elif self.dragging and self.last_drag_pos:
            current_pos = event.globalPos()
            delta = current_pos - self.last_drag_pos
            if self.main_window:
                self.main_window.pan_view(delta.x(), delta.y())
            self.last_drag_pos = current_pos
            event.accept()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_cell_pos = None
        elif event.button() == Qt.MiddleButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
    
    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        if self.main_window:
            if zoom_in:
                self.main_window.zoom_in()
            else:
                self.main_window.zoom_out()
        event.accept()
    
    def toggle_cell(self, pos):
        x = pos.x() // self.cell_size
        y = pos.y() // self.cell_size
        if 0 <= x < self.automaton.width and 0 <= y < self.automaton.height:
            self.automaton.set_cell(x, y, self.draw_mode)
            self.update(
                x * self.cell_size, 
                y * self.cell_size,
                self.cell_size, 
                self.cell_size
            )
    
    def draw_line(self, start, end):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        updated_cells = set()
        
        while True:
            if 0 <= x0 < self.automaton.width and 0 <= y0 < self.automaton.height:
                self.automaton.set_cell(x0, y0, self.draw_mode)
                updated_cells.add((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        for x, y in updated_cells:
            self.update(
                x * self.cell_size, 
                y * self.cell_size,
                self.cell_size, 
                self.cell_size
            )
    
    def update_cell_size(self, cell_size):
        self.cell_size = max(1, cell_size)
        self.update_size()
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("元胞自动机 - 生命游戏 (大网格)")
        
        self.width, self.height = 500, 500
        self.automaton = SparseCellAutomaton(self.width, self.height)
        self.fps = 60
        self.update_interval = 1000 // self.fps
        self.cell_size = 6
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_automaton)
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # 控制面板
        control_layout = QHBoxLayout()
        control_layout.setSpacing(6)
        
        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.toggle_simulation)
        control_layout.addWidget(self.start_button)
        
        self.step_button = QPushButton("单步")
        self.step_button.clicked.connect(self.step_automaton)
        control_layout.addWidget(self.step_button)
        
        self.clear_button = QPushButton("清除")
        self.clear_button.clicked.connect(self.clear_grid)
        control_layout.addWidget(self.clear_button)
        
        control_layout.addWidget(QLabel("绘制模式:"))
        self.draw_mode_combo = QComboBox()
        self.draw_mode_combo.addItem("绘制", 1)
        self.draw_mode_combo.addItem("擦除", 0)
        self.draw_mode_combo.currentIndexChanged.connect(self.change_draw_mode)
        control_layout.addWidget(self.draw_mode_combo)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 缩放和速度控制
        zoom_speed_layout = QHBoxLayout()
        zoom_speed_layout.setSpacing(10)
        
        zoom_layout = QHBoxLayout()
        zoom_layout.setSpacing(4)
        zoom_layout.addWidget(QLabel("缩放:"))
        
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_out_button.setFixedWidth(30)
        zoom_layout.addWidget(self.zoom_out_button)
        
        self.zoom_label = QLabel(f"{self.cell_size}px")
        self.zoom_label.setFixedWidth(40)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        zoom_layout.addWidget(self.zoom_label)
        
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_in_button.setFixedWidth(30)
        zoom_layout.addWidget(self.zoom_in_button)
        
        zoom_speed_layout.addLayout(zoom_layout)
        
        speed_layout = QHBoxLayout()
        speed_layout.setSpacing(6)
        speed_layout.addWidget(QLabel("速度:"))
        
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(1, 120)
        self.fps_slider.setValue(self.fps)
        self.fps_slider.valueChanged.connect(self.set_fps)
        self.fps_slider.setFixedWidth(120)
        speed_layout.addWidget(self.fps_slider)
        
        self.fps_label = QLabel(f"{self.fps} FPS")
        self.fps_label.setFixedWidth(50)
        speed_layout.addWidget(self.fps_label)
        
        zoom_speed_layout.addLayout(speed_layout)
        zoom_speed_layout.addStretch()
        layout.addLayout(zoom_speed_layout)
        
        # 滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.cell_widget = CellWidget(self.automaton, self.cell_size, self)
        self.scroll_area.setWidget(self.cell_widget)
        layout.addWidget(self.scroll_area)
        
        # 状态栏
        self.status_label = QLabel(f"就绪 - 网格: {self.width}x{self.height}, 细胞大小: {self.cell_size}px, FPS: {self.fps}")
        layout.addWidget(self.status_label)
        
        self.resize(800, 600)
        
        # 样式已通过外部 QSS 文件加载，此处不再内嵌
    
    def showEvent(self, event):
        super().showEvent(event)
        self.center_view()
        
    def center_view(self):
        h_scrollbar = self.scroll_area.horizontalScrollBar()
        v_scrollbar = self.scroll_area.verticalScrollBar()
        h_center = (self.cell_widget.width() - self.scroll_area.viewport().width()) // 2
        v_center = (self.cell_widget.height() - self.scroll_area.viewport().height()) // 2
        h_scrollbar.setValue(max(0, h_center))
        v_scrollbar.setValue(max(0, v_center))
        
    def pan_view(self, dx, dy):
        h_scrollbar = self.scroll_area.horizontalScrollBar()
        v_scrollbar = self.scroll_area.verticalScrollBar()
        h_scrollbar.setValue(h_scrollbar.value() - dx)
        v_scrollbar.setValue(v_scrollbar.value() - dy)
        
    def toggle_simulation(self):
        if self.timer.isActive():
            self.timer.stop()
            self.start_button.setText("开始")
            self.status_label.setText("已暂停")
        else:
            self.timer.start(self.update_interval)
            self.start_button.setText("暂停")
            self.status_label.setText("运行中...")
    
    def step_automaton(self):
        self.automaton.update()
        self.cell_widget.update()
    
    def update_automaton(self):
        self.automaton.update()
        self.cell_widget.update()
    
    def clear_grid(self):
        self.automaton.clear()
        self.cell_widget.update()
        self.status_label.setText("网格已清除")
    
    def set_fps(self, value):
        self.fps = value
        self.update_interval = 1000 // self.fps
        self.fps_label.setText(f"{self.fps} FPS")
        self.status_label.setText(f"网格: {self.width}x{self.height}, 细胞大小: {self.cell_size}px, FPS: {self.fps}")
        if self.timer.isActive():
            self.timer.start(self.update_interval)
    
    def change_draw_mode(self, index):
        self.cell_widget.draw_mode = self.draw_mode_combo.itemData(index)
    
    def zoom_in(self):
        old_cell_size = self.cell_size
        self.cell_size = min(50, self.cell_size + 1)
        if old_cell_size != self.cell_size:
            self.update_zoom(old_cell_size)
    
    def zoom_out(self):
        old_cell_size = self.cell_size
        self.cell_size = max(1, self.cell_size - 1)
        if old_cell_size != self.cell_size:
            self.update_zoom(old_cell_size)
    
    def update_zoom(self, old_cell_size):
        h_scrollbar = self.scroll_area.horizontalScrollBar()
        v_scrollbar = self.scroll_area.verticalScrollBar()
        viewport = self.scroll_area.viewport()
        view_center_x = h_scrollbar.value() + viewport.width() / 2
        view_center_y = v_scrollbar.value() + viewport.height() / 2
        world_x = view_center_x / old_cell_size
        world_y = view_center_y / old_cell_size
        
        self.cell_widget.update_cell_size(self.cell_size)
        
        new_view_center_x = world_x * self.cell_size
        new_view_center_y = world_y * self.cell_size
        new_h_scroll = new_view_center_x - viewport.width() / 2
        new_v_scroll = new_view_center_y - viewport.height() / 2
        
        h_scrollbar.setValue(int(new_h_scroll))
        v_scrollbar.setValue(int(new_v_scroll))
        
        self.zoom_label.setText(f"{self.cell_size}px")
        self.status_label.setText(f"网格: {self.width}x{self.height}, 细胞大小: {self.cell_size}px, FPS: {self.fps}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())