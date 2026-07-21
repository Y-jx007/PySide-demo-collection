import sys
import math
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider, QLabel, QHBoxLayout
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer, QPointF
from OpenGL.GL import *
import taichi_forge as ti

ti.init(arch=ti.gpu, random_seed=42)

# ========== 全局常量与默认值 ==========
MAX_PARTICLES = 2048
DISH_R = 75.0
DT = 0.1

# 可调节参数（初始值）
repulsion_val = 1.0
field_ge_val = -1.0
particle_scale_val = 1.0

# 视图控制
view_center = np.array([0.0, 0.0], dtype=np.float32)
view_extent = 80.0

# 窗口大小（动态调整）
W, H = 800, 600

# ========== Taichi 数据 ==========
particles = ti.Vector.field(4, dtype=ti.f32, shape=MAX_PARTICLES)
particle_states = ti.Vector.field(4, dtype=ti.f32, shape=MAX_PARTICLES)
num_particles = ti.field(ti.i32, shape=())

field_tex_w, field_tex_h = 400, 300
field_tex = ti.Vector.field(3, dtype=ti.f32, shape=(field_tex_w, field_tex_h))

rendered_img = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))


# ========== 数学工具 ==========
@ti.func
def kernel_func(r, m, s):
    c = (r - m) / s
    y = ti.exp(-c * c)
    dy_dr = -2.0 * c * y / s
    return ti.Vector([y, dy_dr])

@ti.func
def is_alive(p):
    return p.x > -10000.0

@ti.func
def mix(x, y, a):
    return x * (1.0 - a) + y * a

@ti.func
def smoothstep(edge0, edge1, x):
    t = ti.max(0.0, ti.min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)

def calc_norm_coef(m, s):
    dr = 0.1 * s
    acc = 0.0
    prev = None
    r = max(m - s * 3.0, 0.0)
    while r < m + s * 3.0:
        y = (r - m) / s
        v = r * math.exp(-y * y)
        if prev is not None:
            acc += (prev + v) * 0.5
        prev = v
        r += dr
    return 1.0 / (acc * dr * 2.0 * math.pi)

M1, S1 = 1.27, 1.2
M2, S2 = 0.6, 0.2
w1 = calc_norm_coef(M1, S1)


# ========== 模拟核心 ==========
@ti.kernel
def init_particles(n: ti.i32):
    num_particles[None] = n
    for idx in range(n):
        r = ti.sqrt(idx * 0.5 + 0.25)
        a = 2.4 * idx
        p = ti.Vector([ti.sin(a) * r, ti.cos(a) * r])
        particles[idx] = ti.Vector([p.x, p.y, p.x, p.y])
        particle_states[idx] = ti.Vector([0.5, 0.0, 0.0, 0.0])

@ti.kernel
def add_particle(x: ti.f32, y: ti.f32):
    n = num_particles[None]
    if n < MAX_PARTICLES:
        particles[n] = ti.Vector([x, y, x, y])
        particle_states[n] = ti.Vector([0.5, 0.0, 0.0, 0.0])
        num_particles[None] = n + 1

@ti.kernel
def remove_particle(index: ti.i32):
    n = num_particles[None]
    if n > 0 and 0 <= index < n:
        last = n - 1
        particles[index] = particles[last]
        particle_states[index] = particle_states[last]
        particles[last] = ti.Vector([-20000.0, 0.0, 0.0, 0.0])
        num_particles[None] = last

@ti.kernel
def simulation_step(dt: ti.f32, repulsion: ti.f32):
    dmin = M1 - 3.0 * S1
    dmax = M1 + 3.0 * S1
    n = num_particles[None]
    for idx in range(n):
        p = particles[idx]
        pos = ti.Vector([p.x, p.y])
        prev_pos = ti.Vector([p.z, p.w])
        state = particle_states[idx]
        rep = state.x
        energy = state.y
        force = state.z

        rep_dir = ti.Vector([0.0, 0.0])
        field = 0.0
        gv = ti.Vector([0.0, 0.0])
        rep_val = 0.5

        for j in range(n):
            if j == idx:
                continue
            other = particles[j]
            r_vec = ti.Vector([other.x, other.y]) - pos
            d = r_vec.norm()
            if d < 1e-8:
                pos += ti.Vector([ti.random(), ti.random()]) * 0.001
                continue
            r = r_vec / d
            if d < 1.0:
                f = 1.0 - d
                rep_dir -= f * r
                rep_val += 0.5 * f * f
            if d > dmin and d < dmax:
                v_dv = kernel_func(d, M1, S1)
                field += v_dv.x
                gv += v_dv.y * r

        field *= w1
        gv *= w1
        a_da = kernel_func(field, M2, S2)
        dpos = rep_dir * repulsion - a_da.y * gv
        pos = pos + (pos - prev_pos) * 0.5 + dpos * 0.5 * dt

        pos_len = pos.norm()
        if pos_len > DISH_R:
            pos = pos / pos_len * DISH_R

        energy_new = rep_val * repulsion - field
        force_new = dpos.norm()

        particles[idx] = ti.Vector([pos.x, pos.y, p.x, p.y])
        particle_states[idx] = ti.Vector([rep_val, energy_new, force_new, 0.0])

@ti.kernel
def render_field_texture(
    view_center_x: ti.f32, view_center_y: ti.f32,
    view_extent: ti.f32, field_tex_aspect: ti.f32,
    field_ge: ti.f32, repulsion: ti.f32
):
    extent_x = view_extent
    extent_y = view_extent / field_tex_aspect
    n = num_particles[None]

    for i, j in field_tex:
        uv = ti.Vector([(i + 0.5) / field_tex_w, (j + 0.5) / field_tex_h])
        scr = uv * 2.0 - 1.0
        wpos = ti.Vector([view_center_x, view_center_y]) + 0.5 * scr * ti.Vector([extent_x, extent_y])

        U = 0.0
        gradU = ti.Vector([0.0, 0.0])
        R = 0.0
        gradR = ti.Vector([0.0, 0.0])
        dmin = M1 - 3.0 * S1
        dmax = M1 + 3.0 * S1

        for idx in range(n):
            p = particles[idx]
            ppos = ti.Vector([p.x, p.y])
            rvec = wpos - ppos
            d = rvec.norm()
            if d < 1e-8:
                d = 1e-8
            dr = rvec / d

            if d < 1.0:
                rep = 1.0 - d
                R += 0.5 * repulsion * rep * rep
                gradR += -repulsion * rep * dr

            if d > dmin and d < dmax:
                kv = kernel_func(d, M1, S1)
                U += kv.x
                gradU += kv.y * dr

        U *= w1
        gradU *= w1
        G_dG = kernel_func(U, M2, S2)
        E = R - G_dG.x
        gradE = gradR - G_dG.y * gradU

        g = gradE * field_ge if field_ge > 0.0 else -gradU * field_ge
        normal = ti.Vector([-g.x, -g.y, 1.0]).normalized()
        light_dir = ti.Vector([0.6, 0.6, 0.6]).normalized()
        light = 0.5 + 0.5 * max(normal.dot(light_dir), 0.0)

        c = ti.sqrt(U)
        colorUG = mix(ti.Vector([0.1, 0.1, 0.3]), ti.Vector([0.2, 0.7, 1.0]), c)
        colorUG = mix(colorUG, ti.Vector([0.9, 0.7, 0.1]), G_dG.x * min(c * 2.0, 1.0))
        bg = ti.Vector([1.0, 1.0, 1.0])

        base_color = bg
        if field_ge > 0.0:
            Es = E + kernel_func(0.0, M2, S2).x
            colorE = mix(ti.Vector([1.0, 0.0, 0.0]), ti.Vector([0.2, 0.8, 0.2]), abs(Es))
            base_color = mix(bg, colorE, field_ge)
        elif field_ge < 0.0:
            base_color = mix(bg, colorUG, -field_ge)

        color = light * base_color
        field_tex[i, j] = ti.max(ti.min(color, 1.0), 0.0)


# ========== OpenGL 部件 ==========
class LeniaWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.dragging = False
        self.last_mouse_pos = QPointF()
        self.paused = False
        self.texture_field = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.step_simulation)
        self.timer.start(16)

    def step_simulation(self):
        if not self.paused:
            for _ in range(3):
                simulation_step(DT, repulsion_val)
            self.update()

    def initializeGL(self):
        self.texture_field = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_field)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, field_tex_w, field_tex_h,
                     0, GL_RGB, GL_UNSIGNED_BYTE, None)

        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def paintGL(self):
        global W, H, view_center, view_extent

        ratio = self.devicePixelRatioF()
        vp_w = int(self.width() * ratio)
        vp_h = int(self.height() * ratio)

        # 场纹理渲染
        field_aspect = field_tex_w / field_tex_h
        render_field_texture(
            view_center[0], view_center[1], view_extent, field_aspect,
            field_ge_val, repulsion_val
        )
        arr_field = field_tex.to_numpy()
        arr_field = np.transpose(arr_field, (1, 0, 2))
        arr_field = np.clip(arr_field, 0.0, 1.0)
        arr_field = (arr_field * 255).astype(np.uint8)

        glBindTexture(GL_TEXTURE_2D, self.texture_field)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, field_tex_w, field_tex_h,
                        GL_RGB, GL_UNSIGNED_BYTE, arr_field.tobytes())

        glViewport(0, 0, vp_w, vp_h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, vp_w, 0, vp_h, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 绘制背景四边形
        glEnable(GL_TEXTURE_2D)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(vp_w, 0)
        glTexCoord2f(1, 1); glVertex2f(vp_w, vp_h)
        glTexCoord2f(0, 1); glVertex2f(0, vp_h)
        glEnd()

        # 粒子绘制（修复点大小设置位置）
        n = num_particles.to_numpy().item()
        if n > 0:
            pos_arr = particles.to_numpy()[:n, :2]
            states_arr = particle_states.to_numpy()[:n, 0]

            window_aspect = W / H
            extent_y = view_extent / window_aspect
            screen_x = (pos_arr[:, 0] - (view_center[0] - view_extent * 0.5)) / view_extent * vp_w
            screen_y = (pos_arr[:, 1] - (view_center[1] - extent_y * 0.5)) / extent_y * vp_h

            max_radius = 8.0 * particle_scale_val
            radius_raw = 0.5 + 1.5 / (states_arr + 0.2)
            radius = np.clip(radius_raw * particle_scale_val, 1.0, max_radius)

            visible = (screen_x >= -radius) & (screen_x <= vp_w + radius) & \
                      (screen_y >= -radius) & (screen_y <= vp_h + radius)
            screen_x = screen_x[visible]
            screen_y = screen_y[visible]
            radius = radius[visible]
            rep_vals = states_arr[visible]

            glDisable(GL_TEXTURE_2D)
            # 每个粒子单独一个 begin/end 对，确保 glPointSize 设置合法
            for i in range(len(screen_x)):
                glPointSize(radius[i] * 2)
                glBegin(GL_POINTS)
                r = rep_vals[i]
                color_val = np.clip((r - 0.2) * 1.5, 0.0, 1.0)
                glColor4f(1.0 - color_val, 0.6, color_val, 0.8)
                glVertex2f(screen_x[i], screen_y[i])
                glEnd()
            glEnable(GL_TEXTURE_2D)

        glFlush()

    def resizeGL(self, w, h):
        global W, H, rendered_img
        ratio = self.devicePixelRatioF()
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        if new_w != W or new_h != H:
            W, H = new_w, new_h
            rendered_img = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))

    # ---------- 交互 ----------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier:
                self.dragging = True
                self.last_mouse_pos = event.position()
            else:
                self.add_particle_at(event.position())
        elif event.button() == Qt.MiddleButton:
            self.dragging = True
            self.last_mouse_pos = event.position()
        elif event.button() == Qt.RightButton:
            self.remove_nearest_particle(event.position())

    def mouseMoveEvent(self, event):
        if self.dragging:
            pos = event.position()
            dx = pos.x() - self.last_mouse_pos.x()
            dy = pos.y() - self.last_mouse_pos.y()
            extent_y = view_extent / (W / H)
            self.last_mouse_pos = pos
            view_center[0] -= dx / W * view_extent
            view_center[1] += dy / H * extent_y

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self.dragging = False

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        view_extent *= 1.0 - delta * 0.001
        view_extent = max(5.0, min(200.0, view_extent))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.paused = not self.paused
            self.window().setWindowTitle(
                f"Particle Lenia {'[PAUSED]' if self.paused else ''}"
            )

    def add_particle_at(self, screen_pos):
        extent_y = view_extent / (W / H)
        sx = (screen_pos.x() / W) * 2.0 - 1.0
        sy = -((screen_pos.y() / H) * 2.0 - 1.0)
        wx = view_center[0] + 0.5 * sx * view_extent
        wy = view_center[1] + 0.5 * sy * extent_y
        add_particle(wx, wy)

    def remove_nearest_particle(self, screen_pos):
        extent_y = view_extent / (W / H)
        sx = (screen_pos.x() / W) * 2.0 - 1.0
        sy = -((screen_pos.y() / H) * 2.0 - 1.0)
        wx = view_center[0] + 0.5 * sx * view_extent
        wy = view_center[1] + 0.5 * sy * extent_y

        n = num_particles.to_numpy().item()
        if n == 0:
            return
        pos = particles.to_numpy()[:n, :2]
        diff = pos - np.array([wx, wy])
        dist = np.sqrt(np.sum(diff**2, axis=1))
        idx = np.argmin(dist)
        if dist[idx] < view_extent * 0.1:
            remove_particle(idx)


# ========== 参数控制面板 ==========
class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        rep_row = self._make_slider("Repulsion", 0.0, 3.0, repulsion_val, 0.1)
        rep_row.slider.valueChanged.connect(
            lambda v: self._set_param('repulsion', v * 0.1)
        )
        layout.addLayout(rep_row)

        ge_row = self._make_slider("Field GE", -2.0, 2.0, field_ge_val, 0.1)
        ge_row.slider.valueChanged.connect(
            lambda v: self._set_param('field_ge', v * 0.1)
        )
        layout.addLayout(ge_row)

        scale_row = self._make_slider("Particle Scale", 0.2, 3.0, particle_scale_val, 0.1)
        scale_row.slider.valueChanged.connect(
            lambda v: self._set_param('particle_scale', v * 0.1)
        )
        layout.addLayout(scale_row)

        self.setLayout(layout)

    def _make_slider(self, name, min_val, max_val, init_val, step):
        row = QHBoxLayout()
        label = QLabel(f"{name}: {init_val:.1f}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(min_val / step), int(max_val / step))
        slider.setValue(int(init_val / step))
        slider.setSingleStep(1)
        slider.valueChanged.connect(
            lambda v, l=label, n=name, s=step: l.setText(f"{n}: {v * s:.1f}")
        )
        row.addWidget(label)
        row.addWidget(slider)
        row.slider = slider
        return row

    def _set_param(self, name, value):
        global repulsion_val, field_ge_val, particle_scale_val
        if name == 'repulsion':
            repulsion_val = value
        elif name == 'field_ge':
            field_ge_val = value
        elif name == 'particle_scale':
            particle_scale_val = value


# ========== 主窗口 ==========
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Particle Lenia – 左键添加 | 右键移除 | 中键/Ctrl+左键拖拽 | 空格暂停 | 滚轮缩放")

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.gl_widget = LeniaWidget()
        main_layout.addWidget(self.gl_widget, 1)

        self.panel = ControlPanel()
        main_layout.addWidget(self.panel)

        self.setCentralWidget(central)
        self.resize(800, 660)


if __name__ == "__main__":
    init_particles(512)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())