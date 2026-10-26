from common.qt_base import (QApplication, QBrush, QColor, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QOpenGLWidget, QPainter, QPen, QPointF, QPolygonF, QPushButton, QRect, QSlider, QSpinBox, QSplitter, QSurfaceFormat, QTimer, QVBoxLayout, QWidget, Qt, ctypes, math, sys)
from common.numpy_base import (np)
from common.numba_base import (njit)
from common.opengl_base import (GL_ARRAY_BUFFER, GL_COLOR_BUFFER_BIT, GL_DYNAMIC_DRAW, GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER, GL_LINE_LOOP, GL_POINTS, GL_POINT_SPRITE, GL_PROGRAM_POINT_SIZE, GL_STATIC_DRAW, GL_VERTEX_SHADER, glBegin, glBindBuffer, glBufferData, glBufferSubData, glClear, glClearColor, glColor4f, glDeleteBuffers, glDeleteProgram, glDisableVertexAttribArray, glDrawArrays, glEnable, glEnableVertexAttribArray, glEnd, glGenBuffers, glGetUniformLocation, glLineWidth, glUniform2f, glUseProgram, glVertex2f, glVertexAttribPointer, glViewport)
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPolygonF

# ========== Numba 加速的物理核心 ==========
#
# 状态用 5 条连续的 float32 数组保存（结构数组 → 数组结构），
# 避免 (N, 6) 二维数组在 Numba 中的跨步访问，也避免逐帧产生掩码/临时数组。
#
# 碰撞检测使用计数排序的空间网格：
#   * 格子边长 >= 2 * 半径，因此只需检查自身 + 4 个邻居格；
#   * 用 starts/items 取代固定容量的 (格, 格, 128) 网格，不再静默丢球；
#   * 只遍历“非空格子”，低密度下比扫描全部格子快一个数量级。

CELL_TARGET_OCCUPANCY = 0.35   # 每格平均粒子数：格子数约为 3N，候选对最少
MIN_CELL_SIZE = 0.25           # 物理像素；小半径+大数量时不要强行放大格子
MAX_GRID_CELLS = 2_000_000     # 极端数量下限制网格内存（约 40–80 MB）


@njit(nogil=True, fastmath=True, cache=True)
def integrate_positions(x, y, vx, vy, radius, dt, width, height):
    """位置积分 + 四壁反弹，单趟完成，不分配临时数组。"""
    n = x.shape[0]
    for i in range(n):
        xi = x[i] + vx[i] * dt
        yi = y[i] + vy[i] * dt
        if xi < radius:
            xi = radius
            vx[i] = -vx[i]
        elif xi > width - radius:
            xi = width - radius
            vx[i] = -vx[i]
        if yi < radius:
            yi = radius
            vy[i] = -vy[i]
        elif yi > height - radius:
            yi = height - radius
            vy[i] = -vy[i]
        x[i] = xi
        y[i] = yi


@njit(inline="always")
def _resolve_pair(x, y, vx, vy, radius, i, j):
    """等质量弹性碰撞：先做包围盒剔除，再算冲量与位置修正。"""
    sr = radius + radius
    dx = x[i] - x[j]
    if dx > sr or dx < -sr:
        return
    dy = y[i] - y[j]
    if dy > sr or dy < -sr:
        return
    dist_sq = dx * dx + dy * dy
    if dist_sq > sr * sr or dist_sq <= 0.0:
        return
    dist = math.sqrt(dist_sq)
    inv = 1.0 / dist
    nx = dx * inv
    ny = dy * inv
    dvx = vx[i] - vx[j]
    dvy = vy[i] - vy[j]
    vn = dvx * nx + dvy * ny
    if vn > 0.0:            # 已经在分离
        return
    # 等质量：冲量强度就是 vn，速度沿法线交换
    vx[i] -= vn * nx
    vy[i] -= vn * ny
    vx[j] += vn * nx
    vy[j] += vn * ny
    half = (sr - dist) * 0.5
    x[i] += half * nx
    y[i] += half * ny
    x[j] -= half * nx
    y[j] -= half * ny


@njit(nogil=True, fastmath=True, cache=True)
def build_cell_grid(x, y, grid_w, grid_h, inv_cell_w, inv_cell_h,
                    cell_ids, counts, cursor, starts, items, occupied):
    """把粒子按格子做计数排序，返回非空格子数。"""
    ncells = grid_w * grid_h
    counts.fill(0)
    n = x.shape[0]
    for i in range(n):
        cx = int(x[i] * inv_cell_w)
        if cx < 0:
            cx = 0
        elif cx >= grid_w:
            cx = grid_w - 1
        cy = int(y[i] * inv_cell_h)
        if cy < 0:
            cy = 0
        elif cy >= grid_h:
            cy = grid_h - 1
        cell = cy * grid_w + cx
        cell_ids[i] = cell
        counts[cell] += 1

    total = 0
    n_occupied = 0
    for cell in range(ncells):
        cnt = counts[cell]
        if cnt:
            occupied[n_occupied] = cell
            n_occupied += 1
        starts[cell] = total
        cursor[cell] = total
        total += cnt
    starts[ncells] = total

    for i in range(n):
        cell = cell_ids[i]
        pos = cursor[cell]
        items[pos] = i
        cursor[cell] = pos + 1
    return n_occupied


@njit(nogil=True, fastmath=True, cache=True)
def resolve_grid_collisions(x, y, vx, vy, radius, starts, items,
                            occupied, n_occupied, grid_w, grid_h):
    """遍历非空格子：自身配对 + 右 / 左上 / 上 / 右上，每对只处理一次。"""
    row = 0
    next_row = grid_w
    for oc in range(n_occupied):
        cell = occupied[oc]
        while cell >= next_row:
            next_row += grid_w
            row += 1
        cx = cell - (next_row - grid_w)
        cy = row
        start = starts[cell]
        end = starts[cell + 1]
        for a in range(start, end):
            i = items[a]
            for b in range(a + 1, end):
                _resolve_pair(x, y, vx, vy, radius, i, items[b])
            if cx + 1 < grid_w:
                s2 = starts[cell + 1]
                e2 = starts[cell + 2]
                for b in range(s2, e2):
                    _resolve_pair(x, y, vx, vy, radius, i, items[b])
            if cy + 1 < grid_h:
                up = cell + grid_w
                if cx > 0:
                    s2 = starts[up - 1]
                    e2 = starts[up]
                    for b in range(s2, e2):
                        _resolve_pair(x, y, vx, vy, radius, i, items[b])
                s2 = starts[up]
                e2 = starts[up + 1]
                for b in range(s2, e2):
                    _resolve_pair(x, y, vx, vy, radius, i, items[b])
                if cx + 1 < grid_w:
                    s2 = starts[up + 1]
                    e2 = starts[up + 2]
                    for b in range(s2, e2):
                        _resolve_pair(x, y, vx, vy, radius, i, items[b])


@njit(nogil=True, fastmath=True, cache=True)
def compute_speeds(vx, vy, out):
    for i in range(out.shape[0]):
        out[i] = math.sqrt(vx[i] * vx[i] + vy[i] * vy[i])


@njit(nogil=True, fastmath=True, cache=True)
def write_colors(speeds, min_speed, max_speed, attributes):
    """把速率映射到颜色，直接写入 GPU 属性缓冲，不产生中间数组。"""
    n = speeds.shape[0]
    inv = 0.0
    if max_speed > min_speed:
        inv = 1.0 / (max_speed - min_speed)
    for i in range(n):
        if inv > 0.0:
            norm = (speeds[i] - min_speed) * inv
            if norm < 0.0:
                norm = 0.0
            elif norm > 1.0:
                norm = 1.0
        else:
            norm = 0.5
        attributes[i, 1] = 100.0 / 255.0
        attributes[i, 2] = (200.0 - 50.0 * norm) / 255.0
        attributes[i, 3] = (100.0 + 155.0 * norm) / 255.0


@njit(nogil=True, fastmath=True, cache=True)
def pack_positions(x, y, out):
    """把 x / y 打包成渲染用的连续 (N, 2) 缓冲。"""
    for i in range(out.shape[0]):
        out[i, 0] = x[i]
        out[i, 1] = y[i]


@njit(fastmath=True, cache=True)
def maxwell_distribution(v, v_p):
    if v < 0 or v_p <= 0:
        return 0.0
    return (v / (v_p * v_p)) * math.exp(-v * v / (2 * v_p * v_p))


def create_grid_balls(n, radius, speed, width, height):
    """在整块画布上均匀撒球（抖动网格）。

    旧实现按 2r 间距从左上角逐行填充，球数远小于网格容量时会挤在顶部
    一条几十像素的带里，局部密度约为平均值的 15 倍，碰撞开销成倍增加。
    这里按画布宽高比铺满整个窗口，并用 ±0.3 格抖动避免规则点阵感。
    """
    n = int(n)
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    radius = max(float(radius), 1e-3)
    speed = max(float(speed), 0.0)

    cols = max(1, int(math.ceil(math.sqrt(n * width / height))))
    rows = max(1, int(math.ceil(n / cols)))
    sx = width / cols
    sy = height / rows

    idx = np.arange(n, dtype=np.int64)
    col = idx % cols
    row = idx // cols
    rnd = np.random.random((2, n))
    x = (col + 0.2 + 0.6 * rnd[0]) * sx
    y = (row + 0.2 + 0.6 * rnd[1]) * sy
    lo_x, hi_x = min(radius, width * 0.5), max(width - radius, width * 0.5)
    lo_y, hi_y = min(radius, height * 0.5), max(height - radius, height * 0.5)
    np.clip(x, lo_x, hi_x, out=x)
    np.clip(y, lo_y, hi_y, out=y)

    theta = np.random.uniform(0.0, 2.0 * math.pi, n)
    vx = np.cos(theta) * speed
    vy = np.sin(theta) * speed

    return (x.astype(np.float32), y.astype(np.float32),
            vx.astype(np.float32), vy.astype(np.float32))


# ========== OpenGL 仿真控件 ==========

class GLSimulationWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 物理状态：结构数组拆成 5 条连续数组
        self.x = np.zeros(0, dtype=np.float32)
        self.y = np.zeros(0, dtype=np.float32)
        self.vx = np.zeros(0, dtype=np.float32)
        self.vy = np.zeros(0, dtype=np.float32)
        self.radius = 3.0
        self.speed_array = np.zeros(0, dtype=np.float32)

        # 渲染缓冲（预分配，逐帧原地填充）
        self.pos_buffer = np.zeros((0, 2), dtype=np.float32)
        self.attr_buffer = np.zeros((0, 4), dtype=np.float32)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_simulation)
        self.dt = 0.05        # 固定物理步长
        self.setMinimumSize(500, 500)

        self.shader_program = None
        self.uniform_screen = -1
        self.vbo_positions = None
        self.vbo_attributes = None
        # 记录 VBO 内当前元素数量，用于 glBufferSubData 优化
        self.vbo_pos_capacity = 0
        self.vbo_attr_capacity = 0
        # 当颜色或半径变化时，需要重新上传属性缓冲
        self.attributes_dirty = True

        # 空间网格（尺寸随球数/窗口/半径变化时重建）
        self.grid_w = 0
        self.grid_h = 0
        self.cell_w = 1.0
        self.cell_h = 1.0
        self.inv_cell_w = 1.0
        self.inv_cell_h = 1.0
        self.cell_ids = None
        self.cell_counts = None
        self.cell_cursor = None
        self.cell_starts = None
        self.cell_items = None
        self.cell_occupied = None
        self._grid_ready = False
        self._grid_dirty = True
        self._grid_key = None

        self._color_tick = 0
        self._color_interval = 5

        self.current_dpr = 1.0
        self.physical_width = 500
        self.physical_height = 500
        self.reset_callback = None

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._on_resize_timeout)

    def closeEvent(self, event):
        # 停止物理更新与延迟重置定时器
        self.timer.stop()
        self._resize_timer.stop()
        # 释放物理数组，释放 CPU 内存
        self.x = np.zeros(0, dtype=np.float32)
        self.y = np.zeros(0, dtype=np.float32)
        self.vx = np.zeros(0, dtype=np.float32)
        self.vy = np.zeros(0, dtype=np.float32)
        self.speed_array = np.zeros(0, dtype=np.float32)
        self.pos_buffer = np.zeros((0, 2), dtype=np.float32)
        self.attr_buffer = np.zeros((0, 4), dtype=np.float32)
        self.cell_ids = self.cell_counts = self.cell_cursor = None
        self.cell_starts = self.cell_items = self.cell_occupied = None
        self._grid_ready = False
        # 清理 OpenGL 资源（避免少量 GPU 负担）
        self.makeCurrent()
        if self.vbo_positions:
            glDeleteBuffers(1, [self.vbo_positions])
            self.vbo_positions = None
        if self.vbo_attributes:
            glDeleteBuffers(1, [self.vbo_attributes])
            self.vbo_attributes = None
        if self.shader_program:
            glDeleteProgram(self.shader_program)
            self.shader_program = None
        self.doneCurrent()
        super().closeEvent(event)

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
        self.uniform_screen = glGetUniformLocation(self.shader_program, "screen_size")
        self.vbo_positions = glGenBuffers(1)
        self.vbo_attributes = glGenBuffers(1)

    def resizeGL(self, w, h):
        dpr = self.window().devicePixelRatioF() if self.window() else 1.0
        phys_w = int(w * dpr)
        phys_h = int(h * dpr)
        glViewport(0, 0, phys_w, phys_h)
        self.physical_width = max(1, phys_w)
        self.physical_height = max(1, phys_h)
        self.current_dpr = dpr
        if self.reset_callback is not None and self.x.size > 0:
            # 拖拽改变窗口大小时每帧重建几千/几十万个球会严重卡顿，
            # 停手 150 ms 后再按新尺寸重建。
            self._resize_timer.start(150)
        else:
            self._grid_dirty = True

    def _on_resize_timeout(self):
        if self.reset_callback is not None:
            self.reset_callback()
        else:
            self._grid_dirty = True

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        if self.shader_program is None:
            return
        # 黑色边框（固定管线）
        glUseProgram(0)
        glLineWidth(2.0)
        glColor4f(0.0, 0.0, 0.0, 1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(-1.0, -1.0)
        glVertex2f(1.0, -1.0)
        glVertex2f(1.0, 1.0)
        glVertex2f(-1.0, 1.0)
        glEnd()

        n = self.x.size
        if n == 0:
            return

        glUseProgram(self.shader_program)
        glUniform2f(self.uniform_screen, self.physical_width, self.physical_height)

        # ---- 上传位置数据 ----
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_positions)
        pos_bytes = n * 2 * 4
        if n != self.vbo_pos_capacity:
            glBufferData(GL_ARRAY_BUFFER, pos_bytes, self.pos_buffer, GL_DYNAMIC_DRAW)
            self.vbo_pos_capacity = n
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, pos_bytes, self.pos_buffer)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # ---- 上传属性数据（半径 + 颜色），只在颜色变化时上传 ----
        if self.attributes_dirty and self.attr_buffer.shape[0] == n:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_attributes)
            attr_bytes = n * 4 * 4
            if n != self.vbo_attr_capacity:
                glBufferData(GL_ARRAY_BUFFER, attr_bytes, self.attr_buffer, GL_STATIC_DRAW)
                self.vbo_attr_capacity = n
            else:
                glBufferSubData(GL_ARRAY_BUFFER, 0, attr_bytes, self.attr_buffer)
            self.attributes_dirty = False

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_attributes)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(1 * 4))

        glEnable(GL_POINT_SPRITE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glDrawArrays(GL_POINTS, 0, n)

        # 清理状态
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)

    def set_balls(self, n_balls, radius, initial_speed, phys_width, phys_height):
        n = max(0, int(n_balls))
        self.physical_width = max(1, int(phys_width))
        self.physical_height = max(1, int(phys_height))

        if n == 0:
            self.x = np.zeros(0, dtype=np.float32)
            self.y = np.zeros(0, dtype=np.float32)
            self.vx = np.zeros(0, dtype=np.float32)
            self.vy = np.zeros(0, dtype=np.float32)
            self.speed_array = np.zeros(0, dtype=np.float32)
            self.pos_buffer = np.zeros((0, 2), dtype=np.float32)
            self.attr_buffer = np.zeros((0, 4), dtype=np.float32)
            self._grid_ready = False
            self._grid_key = None
            self.attributes_dirty = True
            self.update()
            return

        self.radius = max(float(radius), 1e-3)
        self.x, self.y, self.vx, self.vy = create_grid_balls(
            n, self.radius, initial_speed,
            self.physical_width, self.physical_height)

        self.speed_array = np.empty(n, dtype=np.float32)
        compute_speeds(self.vx, self.vy, self.speed_array)

        self.pos_buffer = np.empty((n, 2), dtype=np.float32)
        pack_positions(self.x, self.y, self.pos_buffer)

        self.attr_buffer = np.empty((n, 4), dtype=np.float32)
        self.attr_buffer[:, 0] = self.radius
        self._update_colors()

        self._color_tick = 0
        # 球越多，颜色刷新/属性上传越稀疏，避免每帧 16N 字节的带宽
        self._color_interval = max(5, min(60, 5 + n // 25000))

        self._grid_dirty = True
        self._ensure_grid()
        self.update()

    def _ensure_grid(self):
        """按当前球数、窗口尺寸和半径分配网格（尺寸不变时复用）。"""
        n = self.x.size
        if n < 2:
            self._grid_ready = False
            self._grid_key = None
            return

        width = max(1, int(self.physical_width))
        height = max(1, int(self.physical_height))
        target = math.sqrt(width * height * CELL_TARGET_OCCUPANCY / n)
        cell = max(2.0 * self.radius, target, MIN_CELL_SIZE)
        grid_w = max(1, int(width // cell))
        grid_h = max(1, int(height // cell))
        # 极端参数下限制格子数，换取可控内存
        while grid_w * grid_h > MAX_GRID_CELLS:
            cell *= 1.3
            grid_w = max(1, int(width // cell))
            grid_h = max(1, int(height // cell))

        key = (n, width, height, grid_w, grid_h)
        if key == self._grid_key:
            self._grid_dirty = False
            self._grid_ready = True
            return

        # 向下取整保证实际格宽 >= 2 * 半径，邻居只需看一圈
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_w = width / grid_w
        self.cell_h = height / grid_h
        self.inv_cell_w = 1.0 / self.cell_w
        self.inv_cell_h = 1.0 / self.cell_h

        ncells = grid_w * grid_h
        self.cell_ids = np.empty(n, dtype=np.int32)
        self.cell_counts = np.empty(ncells, dtype=np.int32)
        self.cell_cursor = np.empty(ncells + 1, dtype=np.int32)
        self.cell_starts = np.empty(ncells + 1, dtype=np.int32)
        self.cell_items = np.empty(n, dtype=np.int32)
        self.cell_occupied = np.empty(min(n, ncells), dtype=np.int32)
        self._grid_key = key
        self._grid_dirty = False
        self._grid_ready = True

    def start_simulation(self):
        self.timer.start(16)          # 约 60 fps 的物理更新

    def stop_simulation(self):
        self.timer.stop()

    def _update_speeds_cache(self):
        n = self.x.size
        if n == 0:
            self.speed_array = np.zeros(0, dtype=np.float32)
            return
        if self.speed_array.shape[0] != n:
            self.speed_array = np.empty(n, dtype=np.float32)
        compute_speeds(self.vx, self.vy, self.speed_array)

    def _update_colors(self):
        n = self.speed_array.shape[0]
        if n == 0 or self.attr_buffer.shape[0] != n:
            return
        min_s = float(self.speed_array.min())
        max_s = float(self.speed_array.max())
        write_colors(self.speed_array, min_s, max_s, self.attr_buffer)
        self.attributes_dirty = True

    def update_simulation(self):
        n = self.x.size
        if n == 0:
            return
        if self._grid_dirty or self._grid_key is None:
            self._ensure_grid()

        update_width = max(1, self.physical_width)
        update_height = max(1, self.physical_height)
        integrate_positions(self.x, self.y, self.vx, self.vy, self.radius,
                            self.dt, update_width, update_height)

        if n >= 2 and self._grid_ready:
            n_occupied = build_cell_grid(
                self.x, self.y, self.grid_w, self.grid_h,
                self.inv_cell_w, self.inv_cell_h,
                self.cell_ids, self.cell_counts, self.cell_cursor,
                self.cell_starts, self.cell_items, self.cell_occupied)
            if n_occupied:
                resolve_grid_collisions(
                    self.x, self.y, self.vx, self.vy, self.radius,
                    self.cell_starts, self.cell_items, self.cell_occupied,
                    n_occupied, self.grid_w, self.grid_h)

        pack_positions(self.x, self.y, self.pos_buffer)
        compute_speeds(self.vx, self.vy, self.speed_array)

        self._color_tick += 1
        if self._color_tick >= self._color_interval:
            self._color_tick = 0
            self._update_colors()

        self.update()


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
        self._bins = None
        self._brushes = []
        self._brush_bins = -1

    def update_data(self, speed_array):
        self.speed_data = speed_array if speed_array is not None and len(speed_array) > 0 else None
        if self.speed_data is not None:
            self.max_speed = max(1e-6, float(np.max(self.speed_data)))
        self._recompute_bins()
        self.update()

    def _recompute_bins(self):
        if self.speed_data is None or self.max_speed <= 0:
            self._bins = None
            return
        self._bins = np.histogram(self.speed_data, bins=self.bin_count,
                                  range=(0, self.max_speed))[0]

    def set_initial_speed(self, speed):
        self.initial_speed = speed
        self.update()

    def set_bin_count(self, count):
        self.bin_count = max(1, count)
        self._brush_bins = -1
        self._recompute_bins()
        self.update()

    def set_bar_spacing(self, spacing):
        self.bar_spacing = max(0, spacing)
        self.update()

    def get_color_for_bin(self, idx, total):
        ratio = idx / total
        return QColor(100, int(200 - 50 * ratio), int(100 + 155 * ratio))

    def _brushes_for_bins(self):
        if self._brush_bins != self.bin_count:
            self._brushes = [QBrush(self.get_color_for_bin(i, self.bin_count))
                             for i in range(self.bin_count)]
            self._brush_bins = self.bin_count
        return self._brushes

    def paintEvent(self, event):
        with QPainter(self) as painter:
            # 坐标轴和柱子都是横平竖直的，关掉抗锯齿后填充/描边更快；
            # 理论曲线再单独打开，保持平滑。
            painter.setRenderHint(QPainter.Antialiasing, False)
            if self.speed_data is None or self._bins is None:
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
            # X 轴刻度
            for i in range(6):
                x = margin + i * (graph_w / 5)
                val = i * (self.max_speed / 5)
                painter.drawLine(int(x), self.height() - margin - 5, int(x), self.height() - margin + 5)
                painter.drawText(int(x - 10), self.height() - margin + 20, f"{val:.1f}")
            # 直方图柱
            bins = self._bins
            max_count = int(np.max(bins)) if len(bins) > 0 else 1
            if max_count <= 0:
                max_count = 1
            for i in range(6):
                y = self.height() - margin - i * (graph_h / 5)
                val = i * (max_count / 5)
                painter.drawLine(margin - 5, int(y), margin + 5, int(y))
                painter.drawText(margin - 40, int(y + 5), f"{val:.0f}")
            bar_w = max(1, (graph_w / self.bin_count) - self.bar_spacing)
            brushes = self._brushes_for_bins()
            rects = []
            for i, count in enumerate(bins):
                if count == 0:
                    continue
                x = int(margin + i * (graph_w / self.bin_count) + self.bar_spacing / 2)
                bar_h = (count / max_count) * graph_h
                y = int(self.height() - margin - bar_h)
                rects.append(QRect(x, y, int(bar_w), int(bar_h)))
                # 先无边框填充，最后一次性描边，避免每个柱子两次绘制
                painter.fillRect(rects[-1], brushes[i])
                if bar_h > 15:
                    painter.setPen(QPen(Qt.black))
                    painter.drawText(x, y - 5, f"{count}")
            if rects:
                painter.setPen(QPen(Qt.black, 1))
                painter.setBrush(Qt.NoBrush)
                painter.drawRects(rects)
            # 麦克斯韦理论曲线：一次性画折线，避免 100 次 drawLine
            v_p = self.initial_speed / math.sqrt(2)
            max_v = self.max_speed
            if v_p > 0 and max_v > 0:
                painter.setRenderHint(QPainter.Antialiasing, True)
                peak = (1.0 / v_p) * math.exp(-0.5)
                vs = np.linspace(0, max_v, 100)
                ys = (vs / (v_p * v_p)) * np.exp(-vs * vs / (2 * v_p * v_p))
                poly = QPolygonF()
                for v, y_val in zip(vs, ys):
                    poly.append(QPointF(margin + (v / max_v) * graph_w,
                                        self.height() - margin - (y_val / peak) * graph_h))
                painter.setPen(QPen(Qt.black, 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawPolyline(poly)
            # 标签
            painter.rotate(-90)
            painter.drawText(-self.height()//2-15, 20, "数量")
            painter.rotate(90)
            painter.drawText(self.width()//2-15, self.height() - 20, "速率")


# ========== 主窗口（控件绑定微调） ==========

class MaxwellBoltzmannApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("麦克斯韦-玻尔兹曼分布模拟（高性能版）")
        self.setGeometry(100, 100, 1200, 500)
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(splitter)

        self.simulation_widget = GLSimulationWidget()
        self.simulation_widget.set_reset_callback(self._reset_after_resize)
        splitter.addWidget(self.simulation_widget)

        self.histogram_widget = HistogramWidget()
        splitter.addWidget(self.histogram_widget)
        splitter.setSizes([600, 600])

        # 参数拖动/连续输入时合并重置请求，避免每个键击都重建全部小球
        self._reset_timer = QTimer(self)
        self._reset_timer.setSingleShot(True)
        self._reset_timer.timeout.connect(self.reset_simulation)

        self.setup_control_panel()
        self.ball_count = 2000
        self.initial_speed = 20.0
        self.ball_radius = 3.0
        self.create_balls()
        self.simulation_widget.start_simulation()

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_histogram)
        self.update_timer.start(100)

    def closeEvent(self, event):
        # 停止直方图更新与延迟重置定时器
        self.update_timer.stop()
        self._reset_timer.stop()
        # 强制关闭仿真控件（会触发上面的 closeEvent）
        self.simulation_widget.close()
        event.accept()

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

    def _schedule_reset(self, delay_ms=120):
        self._reset_timer.start(delay_ms)

    def _reset_after_resize(self):
        """窗口尺寸变化后仅按新尺寸重排小球，不改变运行/暂停状态。"""
        self.create_balls()

    def reset_simulation(self):
        self._reset_timer.stop()
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
        self._schedule_reset()

    def on_initial_speed_changed(self, val):
        self.initial_speed = val
        self.histogram_widget.set_initial_speed(val)
        self._schedule_reset()

    def on_ball_radius_changed(self, val):
        self.ball_radius = val
        self._schedule_reset()

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
