import sys
import math
import numpy as np
from numba import njit
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QGroupBox, QLabel, QPushButton,
                               QSpinBox, QDoubleSpinBox, QSlider, QSplitter)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


# ========== Numba 加速的物理核心 ==========

@njit(nogil=True, fastmath=True, cache=True)
def update_positions_vectorized(balls_data, dt, width, height):
    n = balls_data.shape[0]
    if n == 0:
        return
    balls_data[:, 0] += balls_data[:, 2] * dt
    balls_data[:, 1] += balls_data[:, 3] * dt
    radii = balls_data[:, 4]
    # 边界反弹
    mask_left = balls_data[:, 0] - radii < 0
    balls_data[mask_left, 0] = radii[mask_left]
    balls_data[mask_left, 2] = -balls_data[mask_left, 2]
    mask_right = balls_data[:, 0] + radii > width
    balls_data[mask_right, 0] = width - radii[mask_right]
    balls_data[mask_right, 2] = -balls_data[mask_right, 2]
    mask_top = balls_data[:, 1] - radii < 0
    balls_data[mask_top, 1] = radii[mask_top]
    balls_data[mask_top, 3] = -balls_data[mask_top, 3]
    mask_bottom = balls_data[:, 1] + radii > height
    balls_data[mask_bottom, 1] = height - radii[mask_bottom]
    balls_data[mask_bottom, 3] = -balls_data[mask_bottom, 3]


@njit(nogil=True, fastmath=True, cache=True)
def resolve_collision_vectorized(balls_data, i, j):
    x1, y1, vx1, vy1, r1, m1 = balls_data[i, 0:6]
    x2, y2, vx2, vy2, r2, m2 = balls_data[j, 0:6]
    dx = x1 - x2
    dy = y1 - y2
    dist_sq = dx * dx + dy * dy
    min_dist = r1 + r2
    min_dist_sq = min_dist * min_dist
    if dist_sq <= min_dist_sq and dist_sq > 0:
        dist = math.sqrt(dist_sq)
        nx = dx / dist
        ny = dy / dist
        dvx = vx1 - vx2
        dvy = vy1 - vy2
        vn = dvx * nx + dvy * ny
        if vn > 0:
            return
        impulse = 2 * vn / (m1 + m2)
        balls_data[i, 2] = vx1 - impulse * m2 * nx
        balls_data[i, 3] = vy1 - impulse * m2 * ny
        balls_data[j, 2] = vx2 + impulse * m1 * nx
        balls_data[j, 3] = vy2 + impulse * m1 * ny
        overlap = min_dist - dist
        if overlap > 0:
            total_m = m1 + m2
            balls_data[i, 0] = x1 + overlap * nx * (m2 / total_m)
            balls_data[i, 1] = y1 + overlap * ny * (m2 / total_m)
            balls_data[j, 0] = x2 - overlap * nx * (m1 / total_m)
            balls_data[j, 1] = y2 - overlap * ny * (m1 / total_m)


@njit(nogil=True, fastmath=True, cache=True)
def check_collisions_with_dynamic_grid(balls_data, width, height,
                                       grid, grid_counts, max_capacity):
    n = balls_data.shape[0]
    if n < 2:
        return
    avg_per_cell = 8
    grid_size = int(math.sqrt(n / avg_per_cell))
    grid_size = max(5, min(50, grid_size))
    cell_w = width / grid_size
    cell_h = height / grid_size
    grid_counts.fill(0)
    xs = balls_data[:, 0]
    ys = balls_data[:, 1]
    for i in range(n):
        cx = min(max(0, int(xs[i] / cell_w)), grid_size - 1)
        cy = min(max(0, int(ys[i] / cell_h)), grid_size - 1)
        cnt = grid_counts[cy, cx]
        if cnt < max_capacity:
            grid[cy, cx, cnt] = i
            grid_counts[cy, cx] = cnt + 1
    for cy in range(grid_size):
        for cx in range(grid_size):
            count = grid_counts[cy, cx]
            for i_idx in range(count):
                i = grid[cy, cx, i_idx]
                for j_idx in range(i_idx + 1, count):
                    j = grid[cy, cx, j_idx]
                    resolve_collision_vectorized(balls_data, i, j)
            for dy in range(0, 2):
                for dx in range(-1 if dy == 0 else 0, 2):
                    if dx == 0 and dy == 0:
                        continue
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < grid_size and 0 <= nx < grid_size:
                        nbr_count = grid_counts[ny, nx]
                        for i_idx in range(count):
                            i = grid[cy, cx, i_idx]
                            for j_idx in range(nbr_count):
                                j = grid[ny, nx, j_idx]
                                if i < j:
                                    resolve_collision_vectorized(balls_data, i, j)


@njit(nogil=True, fastmath=True, cache=True)
def calculate_speeds_vectorized(balls_data):
    vx = balls_data[:, 2]
    vy = balls_data[:, 3]
    return np.sqrt(vx * vx + vy * vy)


@njit(fastmath=True, cache=True)
def maxwell_distribution(v, v_p):
    if v < 0 or v_p <= 0:
        return 0.0
    return (v / (v_p * v_p)) * math.exp(-v * v / (2 * v_p * v_p))


# ========== OpenGL 仿真控件 ==========

class GLSimulationWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.balls_data = np.zeros((0, 9), dtype=np.float32)
        self.speed_array = np.zeros(0, dtype=np.float32)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.dt = 0.05
        self.setMinimumSize(500, 500)
        self.frame_counter = 0
        self.collision_check_interval = 5
        self.color_update_counter = 0
        self.color_update_interval = 5
        self.shader_program = None
        self.vbo_positions = None
        self.vbo_attributes = None
        self.attributes_dirty = True
        self.grid = None
        self.grid_counts = None
        self.max_cell_capacity = 128
        self.current_dpr = 1.0
        self.physical_width = 500
        self.physical_height = 500
        self.reset_callback = None

    def set_reset_callback(self, callback):
        self.reset_callback = callback

    def initializeGL(self):
        glClearColor(0.95, 0.95, 0.95, 1.0)
        vertex_src = """
        #version 330 core
        layout (location = 0) in vec2 position;
        layout (location = 1) in float radius;
        layout (location = 2) in vec3 color;
        out vec3 frag_color;
        uniform vec2 screen_size;
        void main() {
            vec2 ndc = vec2(2.0 * position.x / screen_size.x - 1.0,
                            1.0 - 2.0 * position.y / screen_size.y);
            gl_Position = vec4(ndc, 0.0, 1.0);
            gl_PointSize = radius * 2.0;
            frag_color = color;
        }
        """
        fragment_src = """
        #version 330 core
        in vec3 frag_color;
        out vec4 color;
        void main() {
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float dist = length(coord);
            if (dist > 1.0) discard;
            if (dist > 0.92) {
                color = vec4(0.0, 0.0, 0.0, 1.0);
            } else {
                color = vec4(frag_color, 1.0);
            }
        }
        """
        vs = compileShader(vertex_src, GL_VERTEX_SHADER)
        fs = compileShader(fragment_src, GL_FRAGMENT_SHADER)
        self.shader_program = compileProgram(vs, fs)
        self.vbo_positions = glGenBuffers(1)
        self.vbo_attributes = glGenBuffers(1)

    def resizeGL(self, w, h):
        dpr = self.window().devicePixelRatioF() if self.window() else 1.0
        phys_w = int(w * dpr)
        phys_h = int(h * dpr)
        glViewport(0, 0, phys_w, phys_h)
        self.physical_width = phys_w
        self.physical_height = phys_h
        self.current_dpr = dpr
        if self.reset_callback and self.balls_data.shape[0] > 0:
            self.reset_callback()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        if self.shader_program is None:
            return
        # 黑色边框
        glUseProgram(0)
        glLineWidth(2.0)
        glColor4f(0.0, 0.0, 0.0, 1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(-1.0, -1.0)
        glVertex2f(1.0, -1.0)
        glVertex2f(1.0, 1.0)
        glVertex2f(-1.0, 1.0)
        glEnd()

        n = self.balls_data.shape[0]
        if n == 0:
            return
        glUseProgram(self.shader_program)
        screen_size_loc = glGetUniformLocation(self.shader_program, "screen_size")
        glUniform2f(screen_size_loc, self.physical_width, self.physical_height)
        # 上传位置
        positions = self.balls_data[:, 0:2].copy().ravel()
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_positions)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        # 上传属性（仅当脏时）
        if self.attributes_dirty:
            attr = np.zeros((n, 4), dtype=np.float32)
            attr[:, 0] = self.balls_data[:, 4]          # 半径
            attr[:, 1:4] = self.balls_data[:, 6:9] / 255.0   # 颜色
            flat = attr.ravel()
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_attributes)
            glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STATIC_DRAW)
            self.attributes_dirty = False
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_attributes)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(1 * 4))
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glDrawArrays(GL_POINTS, 0, n)
        # 清理
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)

    def set_balls(self, n_balls, radius, initial_speed, phys_width, phys_height):
        if n_balls <= 0:
            self.balls_data = np.zeros((0, 9), dtype=np.float32)
            self.speed_array = np.zeros(0, dtype=np.float32)
            return
        cols = max(1, int(phys_width / (2 * radius)))
        rows = max(1, int(phys_height / (2 * radius)))
        self.balls_data = self._create_grid_balls(n_balls, radius, initial_speed,
                                                  phys_width, phys_height, rows, cols)
        self.speed_array = np.zeros(len(self.balls_data), dtype=np.float32)
        self._update_speeds_cache()
        self._update_colors_cache()
        self.attributes_dirty = True
        self._ensure_grid_capacity()

    def _create_grid_balls(self, n, radius, speed, width, height, rows, cols):
        balls = np.zeros((n, 9), dtype=np.float32)
        placed = 0
        for row in range(rows):
            for col in range(cols):
                if placed >= n:
                    break
                x = col * (2 * radius) + radius
                y = row * (2 * radius) + radius
                x = min(x, width - radius)
                y = min(y, height - radius)
                angle = np.random.uniform(0, 2 * np.pi)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                balls[placed] = [x, y, vx, vy, radius, 1.0, 100, 200, 100]
                placed += 1
        return balls[:placed]

    def _update_speeds_cache(self):
        if self.balls_data.shape[0] > 0:
            self.speed_array = calculate_speeds_vectorized(self.balls_data)

    def _update_colors_cache(self):
        if self.balls_data.shape[0] == 0 or len(self.speed_array) == 0:
            return
        speeds = self.speed_array
        max_s = speeds.max()
        min_s = speeds.min()
        if max_s > min_s:
            norm = (speeds - min_s) / (max_s - min_s)
        else:
            norm = np.full_like(speeds, 0.5)
        self.balls_data[:, 6] = 100
        self.balls_data[:, 7] = 200 - 50 * norm
        self.balls_data[:, 8] = 100 + 155 * norm
        self.attributes_dirty = True

    def _ensure_grid_capacity(self):
        n = self.balls_data.shape[0]
        if n < 2:
            self.grid = None
            self.grid_counts = None
            return
        gs = max(5, min(50, int(math.sqrt(n / 8))))
        if self.grid is None or self.grid.shape[0] != gs:
            self.grid = np.zeros((gs, gs, self.max_cell_capacity), dtype=np.int32)
            self.grid_counts = np.zeros((gs, gs), dtype=np.int32)

    def start_simulation(self):
        self.timer.start(0)

    def stop_simulation(self):
        self.timer.stop()

    def update_simulation(self):
        if self.balls_data.shape[0] == 0:
            return
        width, height = self.physical_width, self.physical_height
        update_positions_vectorized(self.balls_data, self.dt, width, height)
        self.frame_counter += 1
        if self.frame_counter >= self.collision_check_interval:
            self.frame_counter = 0
            n = self.balls_data.shape[0]
            if n > 100 and self.grid is not None:
                check_collisions_with_dynamic_grid(
                    self.balls_data, width, height,
                    self.grid, self.grid_counts, self.max_cell_capacity)
            elif n >= 2:
                self._simple_collision_check()
        self.color_update_counter += 1
        if self.color_update_counter >= self.color_update_interval:
            self.color_update_counter = 0
            self._update_speeds_cache()
            self._update_colors_cache()
        self.update()

    def _simple_collision_check(self):
        n = self.balls_data.shape[0]
        if n < 2:
            return
        # 使用 sort 的扫描法（未使用 njit，但仅小规模使用，性能足够）
        indices = np.argsort(self.balls_data[:, 0])
        for i_idx in range(n):
            i = indices[i_idx]
            xi, yi, ri = self.balls_data[i, 0], self.balls_data[i, 1], self.balls_data[i, 4]
            max_x = xi + 2 * ri
            for j_idx in range(i_idx + 1, n):
                j = indices[j_idx]
                if self.balls_data[j, 0] > max_x:
                    break
                dx = xi - self.balls_data[j, 0]
                dy = yi - self.balls_data[j, 1]
                if dx * dx + dy * dy <= (ri + self.balls_data[j, 4]) ** 2:
                    resolve_collision_vectorized(self.balls_data, i, j)


# ========== 直方图控件 ==========

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.speed_data = None
        self.bin_count = 20
        self.max_speed = 10.0
        self.bar_spacing = 0
        self.initial_speed = 20.0
        self.setMinimumSize(500, 500)
        self.setStyleSheet("background-color: #f0f0f0;")

    def update_data(self, speed_array):
        self.speed_data = speed_array if speed_array is not None and len(speed_array) > 0 else None
        if self.speed_data is not None:
            self.max_speed = np.max(self.speed_data)
        self.update()

    def set_initial_speed(self, speed):
        self.initial_speed = speed
        self.update()

    def set_bin_count(self, count):
        self.bin_count = max(1, count)
        self.update()

    def set_bar_spacing(self, spacing):
        self.bar_spacing = max(0, spacing)
        self.update()

    def get_color_for_bin(self, idx, total):
        ratio = idx / total
        return QColor(100, int(200 - 50 * ratio), int(100 + 155 * ratio))

    def paintEvent(self, event):
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.Antialiasing)
            if self.speed_data is None:
                painter.setPen(QPen(Qt.black))
                painter.drawText(self.rect(), Qt.AlignCenter, "等待数据...")
                return
            margin = 50
            graph_w = self.width() - 2 * margin
            graph_h = self.height() - 2 * margin
            # 坐标轴
            painter.setPen(QPen(Qt.black, 2))
            painter.drawLine(margin, self.height() - margin, self.width() - margin, self.height() - margin)
            painter.drawLine(margin, margin, margin, self.height() - margin)
            # 刻度
            painter.setFont(QFont("Arial", 8))
            for i in range(6):
                x = margin + i * (graph_w / 5)
                val = i * (self.max_speed / 5)
                painter.drawLine(int(x), self.height() - margin - 5, int(x), self.height() - margin + 5)
                painter.drawText(int(x - 10), self.height() - margin + 20, f"{val:.1f}")
            bins = np.histogram(self.speed_data, bins=self.bin_count, range=(0, self.max_speed))[0]
            max_count = np.max(bins) if len(bins) > 0 else 1
            for i in range(6):
                y = self.height() - margin - i * (graph_h / 5)
                val = i * (max_count / 5)
                painter.drawLine(margin - 5, int(y), margin + 5, int(y))
                painter.drawText(margin - 40, int(y + 5), f"{val:.0f}")
            # 直方图柱
            bar_w = max(1, (graph_w / self.bin_count) - self.bar_spacing)
            for i, count in enumerate(bins):
                if count == 0:
                    continue
                x = margin + i * (graph_w / self.bin_count) + self.bar_spacing / 2
                bar_h = (count / max_count) * graph_h
                y = self.height() - margin - bar_h
                painter.setPen(QPen(Qt.black, 1))
                painter.setBrush(QBrush(self.get_color_for_bin(i, self.bin_count)))
                painter.drawRect(int(x), int(y), int(bar_w), int(bar_h))
                if bar_h > 15:
                    painter.setPen(QPen(Qt.black))
                    painter.drawText(int(x), int(y - 5), f"{count}")
            # 麦克斯韦理论曲线
            v_p = self.initial_speed / math.sqrt(2)
            max_v = self.max_speed
            peak = maxwell_distribution(v_p, v_p)
            painter.setPen(QPen(Qt.black, 2))
            points = []
            for v in np.linspace(0, max_v, 100):
                x = margin + (v / max_v) * graph_w
                y_val = maxwell_distribution(v, v_p)
                y = self.height() - margin - (y_val / peak) * graph_h
                points.append((x, y))
            for k in range(len(points) - 1):
                painter.drawLine(int(points[k][0]), int(points[k][1]),
                                 int(points[k+1][0]), int(points[k+1][1]))
            # 标签
            painter.rotate(-90)
            painter.drawText(-self.height()//2-15, 20, "数量")
            painter.rotate(90)
            painter.drawText(self.width()//2-15, self.height() - 20, "速率")


# ========== 主窗口 ==========

class MaxwellBoltzmannApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("麦克斯韦-玻尔兹曼分布模拟")
        self.setGeometry(100, 100, 1200, 500)
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(splitter)
        self.simulation_widget = GLSimulationWidget()
        self.simulation_widget.set_reset_callback(self.reset_simulation)
        splitter.addWidget(self.simulation_widget)
        self.histogram_widget = HistogramWidget()
        splitter.addWidget(self.histogram_widget)
        splitter.setSizes([600, 600])
        self.setup_control_panel()
        self.ball_count = 2000
        self.initial_speed = 20.0
        self.ball_radius = 3.0
        self.create_balls()
        self.simulation_widget.start_simulation()
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_histogram)
        self.update_timer.start(100)

    def setup_control_panel(self):
        control = QWidget()
        control.setMaximumWidth(300)
        layout = QVBoxLayout(control)
        # 模拟组
        sim = QGroupBox("模拟控制")
        sly = QVBoxLayout(sim)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 1000000)
        self.count_spin.setValue(2000)
        self.count_spin.valueChanged.connect(self.on_ball_count_changed)
        sly.addWidget(QLabel("小球数量(1-1000000):"))
        sly.addWidget(self.count_spin)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.0, 1000.0)
        self.speed_spin.setValue(20.0)
        self.speed_spin.setSingleStep(0.5)
        self.speed_spin.valueChanged.connect(self.on_initial_speed_changed)
        sly.addWidget(QLabel("初始速率(0.0-1000.0):"))
        sly.addWidget(self.speed_spin)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 20.0)
        self.radius_spin.setValue(3.0)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.valueChanged.connect(self.on_ball_radius_changed)
        sly.addWidget(QLabel("小球半径(0.1-20.0):"))
        sly.addWidget(self.radius_spin)
        btns = QHBoxLayout()
        self.btn_start = QPushButton("开始")
        self.btn_start.clicked.connect(self.simulation_widget.start_simulation)
        self.btn_stop = QPushButton("暂停")
        self.btn_stop.clicked.connect(self.simulation_widget.stop_simulation)
        self.btn_reset = QPushButton("重置")
        self.btn_reset.clicked.connect(self.reset_simulation)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_reset)
        sly.addLayout(btns)
        layout.addWidget(sim)
        # 直方图组
        hist = QGroupBox("直方图设置")
        hly = QVBoxLayout(hist)
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(1, 100)
        self.bins_spin.setValue(20)
        self.bins_spin.valueChanged.connect(self.histogram_widget.set_bin_count)
        hly.addWidget(QLabel("区间数量(1-100):"))
        hly.addWidget(self.bins_spin)
        self.spacing_slider = QSlider(Qt.Horizontal)
        self.spacing_slider.setRange(0, 20)
        self.spacing_slider.valueChanged.connect(self.histogram_widget.set_bar_spacing)
        hly.addWidget(QLabel("柱子间距(0-20):"))
        hly.addWidget(self.spacing_slider)
        layout.addWidget(hist)
        layout.addStretch()
        self.main_layout.addWidget(control)

    def get_physical_size(self):
        dpr = self.simulation_widget.window().devicePixelRatioF() if self.simulation_widget.window() else 1.0
        w = self.simulation_widget.width()
        h = self.simulation_widget.height()
        return int(w * dpr), int(h * dpr)

    def create_balls(self):
        pw, ph = self.get_physical_size()
        if pw <= 0 or ph <= 0:
            pw, ph = 800, 600
        self.simulation_widget.set_balls(self.ball_count, self.ball_radius,
                                         self.initial_speed, pw, ph)
        self.update_histogram()

    def reset_simulation(self):
        self.create_balls()
        self.simulation_widget.start_simulation()

    def on_ball_count_changed(self, val):
        self.ball_count = val
        new_radius = max(0.1, min(20.0, 3 * math.sqrt(2000 / val)))
        self.radius_spin.blockSignals(True)
        self.radius_spin.setValue(new_radius)
        self.radius_spin.blockSignals(False)
        self.ball_radius = new_radius
        new_speed = max(0.0, min(1000.0, 20 * math.sqrt(val / 2000)))
        self.speed_spin.blockSignals(True)
        self.speed_spin.setValue(new_speed)
        self.speed_spin.blockSignals(False)
        self.initial_speed = new_speed
        self.histogram_widget.set_initial_speed(new_speed)
        self.reset_simulation()

    def on_initial_speed_changed(self, val):
        self.initial_speed = val
        self.histogram_widget.set_initial_speed(val)
        self.reset_simulation()

    def on_ball_radius_changed(self, val):
        self.ball_radius = val
        self.reset_simulation()

    def update_histogram(self):
        self.histogram_widget.update_data(self.simulation_widget.speed_array)


def main():
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    window = MaxwellBoltzmannApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()