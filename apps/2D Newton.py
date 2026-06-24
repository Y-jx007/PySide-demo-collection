from custom_import import *
import sys
import numpy as np
import taichi_forge as ti
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox,
    QSpinBox, QComboBox, QGroupBox, QSplitter,
    QMessageBox, QPlainTextEdit, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter

ti.init(arch=ti.gpu)

# ==================== Taichi 基础类型与辅助函数 ====================
c64 = ti.types.vector(2, ti.f64)

@ti.func
def csqr(z: c64) -> c64:
    return c64(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y)

@ti.func
def cconj(z: c64) -> c64:
    return c64(z.x, -z.y)

@ti.func
def cmul(a: c64, b: c64) -> c64:
    return c64(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)

@ti.func
def cdiv(a: c64, b: c64) -> c64:
    denom = b.x * b.x + b.y * b.y
    safe = denom > 1e-12
    inv_denom = 1.0 / ti.max(denom, 1e-12)
    real = (a.x * b.x + a.y * b.y) * inv_denom
    imag = (a.y * b.x - a.x * b.y) * inv_denom
    return c64(ti.select(safe, real, 0.0), ti.select(safe, imag, 0.0))

@ti.func
def csin(z: c64) -> c64:
    ey = ti.exp(z.y)
    e_neg_y = ti.exp(-z.y)
    cosh_y = (ey + e_neg_y) * 0.5
    sinh_y = (ey - e_neg_y) * 0.5
    return c64(ti.sin(z.x) * cosh_y, ti.cos(z.x) * sinh_y)

@ti.func
def ccos(z: c64) -> c64:
    ey = ti.exp(z.y)
    e_neg_y = ti.exp(-z.y)
    cosh_y = (ey + e_neg_y) * 0.5
    sinh_y = (ey - e_neg_y) * 0.5
    return c64(ti.cos(z.x) * cosh_y, -ti.sin(z.x) * sinh_y)

@ti.func
def cexp(z: c64) -> c64:
    r = ti.exp(z.x)
    return c64(r * ti.cos(z.y), r * ti.sin(z.y))

@ti.func
def clog(z: c64) -> c64:
    return c64(ti.log(ti.sqrt(z.x * z.x + z.y * z.y)),
               ti.atan2(z.y, z.x))

@ti.func
def cpow(z: c64, n: ti.f64) -> c64:
    r = ti.sqrt(z.x * z.x + z.y * z.y)
    theta = ti.atan2(z.y, z.x)
    new_r = ti.pow(r, n)
    new_theta = theta * n
    return c64(new_r * ti.cos(new_theta), new_r * ti.sin(new_theta))

# ==================== 预设牛顿分形函数及其导数 ====================
@ti.func
def preset_f(z: c64, func_type: ti.i32) -> c64:
    result = c64(0.0, 0.0)
    if func_type == 0:
        result = cpow(z, 3.0) - c64(1.0, 0.0)
    elif func_type == 1:
        result = cpow(z, 4.0) - c64(1.0, 0.0)
    elif func_type == 2:
        result = cpow(z, 5.0) - c64(1.0, 0.0)
    elif func_type == 3:
        result = cpow(z, 3.0) - z
    elif func_type == 4:
        result = cpow(z, 4.0) + cpow(z, 2.0) + c64(1.0, 0.0)
    elif func_type == 5:
        result = csin(z)
    elif func_type == 6:
        result = ccos(z)
    else:
        result = cpow(z, 3.0) - c64(1.0, 0.0)
    return result

@ti.func
def preset_df(z: c64, func_type: ti.i32) -> c64:
    result = c64(0.0, 0.0)
    if func_type == 0:
        result = cmul(c64(3.0, 0.0), csqr(z))
    elif func_type == 1:
        result = cmul(c64(4.0, 0.0), cpow(z, 3.0))
    elif func_type == 2:
        result = cmul(c64(5.0, 0.0), cpow(z, 4.0))
    elif func_type == 3:
        result = cmul(c64(3.0, 0.0), csqr(z)) - c64(1.0, 0.0)
    elif func_type == 4:
        result = cmul(c64(4.0, 0.0), cpow(z, 3.0)) + cmul(c64(2.0, 0.0), z)
    elif func_type == 5:
        result = ccos(z)
    elif func_type == 6:
        result = -csin(z)
    else:
        result = cmul(c64(3.0, 0.0), csqr(z))
    return result

# ==================== 颜色与根判定（仅预设用） ====================
@ti.func
def pack_color(r: ti.f32, g: ti.f32, b: ti.f32) -> ti.u32:
    ri = ti.cast(ti.min(255, ti.max(0, r * 255.0)), ti.u32)
    gi = ti.cast(ti.min(255, ti.max(0, g * 255.0)), ti.u32)
    bi = ti.cast(ti.min(255, ti.max(0, b * 255.0)), ti.u32)
    return (ti.u32(0xff) << 24) | (ri << 16) | (gi << 8) | bi

@ti.func
def get_root_index_taichi(z: c64, func_type: ti.i32) -> ti.i32:
    min_dist = 1e30
    best_idx = 0
    if func_type == 0:
        d0 = (z.x - 1.0)*(z.x - 1.0) + z.y*z.y
        d1 = (z.x + 0.5)*(z.x + 0.5) + (z.y - 0.8660254)*(z.y - 0.8660254)
        d2 = (z.x + 0.5)*(z.x + 0.5) + (z.y + 0.8660254)*(z.y + 0.8660254)
        min_dist = d0; best_idx = 0
        if d1 < min_dist: min_dist = d1; best_idx = 1
        if d2 < min_dist: min_dist = d2; best_idx = 2
    elif func_type == 1:
        d0 = (z.x - 1.0)*(z.x - 1.0) + z.y*z.y
        d1 = (z.x + 1.0)*(z.x + 1.0) + z.y*z.y
        d2 = z.x*z.x + (z.y - 1.0)*(z.y - 1.0)
        d3 = z.x*z.x + (z.y + 1.0)*(z.y + 1.0)
        min_dist = d0; best_idx = 0
        if d1 < min_dist: min_dist = d1; best_idx = 1
        if d2 < min_dist: min_dist = d2; best_idx = 2
        if d3 < min_dist: min_dist = d3; best_idx = 3
    elif func_type == 2:
        d0 = (z.x - 1.0)*(z.x - 1.0) + z.y*z.y
        d1 = (z.x - 0.309016994)*(z.x - 0.309016994) + (z.y - 0.951056516)*(z.y - 0.951056516)
        d2 = (z.x + 0.809016994)*(z.x + 0.809016994) + (z.y - 0.587785252)*(z.y - 0.587785252)
        d3 = (z.x + 0.809016994)*(z.x + 0.809016994) + (z.y + 0.587785252)*(z.y + 0.587785252)
        d4 = (z.x - 0.309016994)*(z.x - 0.309016994) + (z.y + 0.951056516)*(z.y + 0.951056516)
        min_dist = d0; best_idx = 0
        if d1 < min_dist: min_dist = d1; best_idx = 1
        if d2 < min_dist: min_dist = d2; best_idx = 2
        if d3 < min_dist: min_dist = d3; best_idx = 3
        if d4 < min_dist: min_dist = d4; best_idx = 4
    elif func_type == 3:
        d0 = z.x*z.x + z.y*z.y
        d1 = (z.x - 1.0)*(z.x - 1.0) + z.y*z.y
        d2 = (z.x + 1.0)*(z.x + 1.0) + z.y*z.y
        min_dist = d0; best_idx = 0
        if d1 < min_dist: min_dist = d1; best_idx = 1
        if d2 < min_dist: min_dist = d2; best_idx = 2
    elif func_type == 4:
        d0 = (z.x - 0.5)*(z.x - 0.5) + (z.y - 0.8660254)*(z.y - 0.8660254)
        d1 = (z.x - 0.5)*(z.x - 0.5) + (z.y + 0.8660254)*(z.y + 0.8660254)
        d2 = (z.x + 0.5)*(z.x + 0.5) + (z.y - 0.8660254)*(z.y - 0.8660254)
        d3 = (z.x + 0.5)*(z.x + 0.5) + (z.y + 0.8660254)*(z.y + 0.8660254)
        min_dist = d0; best_idx = 0
        if d1 < min_dist: min_dist = d1; best_idx = 1
        if d2 < min_dist: min_dist = d2; best_idx = 2
        if d3 < min_dist: min_dist = d3; best_idx = 3
    elif func_type == 5 or func_type == 6:
        best_idx = 0
    return best_idx

@ti.func
def get_root_gray_taichi(root_index: ti.i32, func_type: ti.i32) -> ti.f32:
    count = 3
    if func_type == 0: count = 3
    elif func_type == 1: count = 4
    elif func_type == 2: count = 5
    elif func_type == 3: count = 3
    elif func_type == 4: count = 4
    elif func_type == 5: count = 1
    elif func_type == 6: count = 1
    return ti.select(count <= 1, 1.0, ti.cast(root_index, ti.f32) / ti.cast(count - 1, ti.f32))

# ==================== 预设函数的静态 Taichi 内核 ====================
@ti.kernel
def render_preset(
    pixels: ti.template(),
    width: ti.i32, height: ti.i32,
    max_iter: ti.i32, threshold: ti.f64,
    c_real: ti.f64, c_imag: ti.f64,
    display_mode: ti.i32,
    view_scale: ti.f64, view_center_x: ti.f64, view_center_y: ti.f64,
    func_type: ti.i32
):
    ti.loop_config(block_dim=256)
    for i, j in ti.ndrange(width, height):
        aspect = ti.cast(width, ti.f64) / ti.cast(height, ti.f64)
        uv_x = (ti.cast(i, ti.f64) / ti.cast(width, ti.f64) - 0.5) * 2.0 * aspect
        uv_y = (0.5 - ti.cast(j, ti.f64) / ti.cast(height, ti.f64)) * 2.0
        uv_x = uv_x * view_scale + view_center_x
        uv_y = uv_y * view_scale + view_center_y

        z = c64(uv_x, uv_y)
        c = c64(c_real, c_imag)
        iterations = 0
        converged = False
        root_index = -1

        for _ in range(max_iter):
            f_z = preset_f(z, func_type)
            df_z = preset_df(z, func_type)

            if (df_z.x * df_z.x + df_z.y * df_z.y) < 1e-16:
                break

            z_new = z - cdiv(f_z, df_z)
            if display_mode == 1:   # Nova
                z_new = z_new + c

            if display_mode == 0:
                if (f_z.x * f_z.x + f_z.y * f_z.y) < threshold * threshold:
                    converged = True
                    root_index = get_root_index_taichi(z, func_type)
                    break
            else:
                diff = z_new - z
                if (diff.x * diff.x + diff.y * diff.y) < threshold * threshold:
                    converged = True
                    break

            z = z_new
            iterations += 1

        if converged:
            if display_mode == 0:
                gray = get_root_gray_taichi(root_index, func_type)
                pixels[j, i] = pack_color(gray, gray, gray)
            else:
                smooth = ti.cast(iterations, ti.f64) + 1.0
                gray = 3.0 * ti.log(1.0 + ti.log(1.0 + smooth / ti.cast(max_iter, ti.f64)))
                pixels[j, i] = pack_color(ti.cast(gray, ti.f32), ti.cast(gray, ti.f32), ti.cast(gray, ti.f32))
        else:
            pixels[j, i] = pack_color(0.0, 0.0, 0.0)

# ==================== 自定义函数的动态编译基础设施 ====================
def _custom_f_placeholder(z: c64) -> c64:
    return z

custom_f_func = _custom_f_placeholder
custom_kernel = None

def build_custom_kernel():
    """使用当前的 custom_f_func 动态编译渲染内核（数值导数）"""
    global custom_kernel

    @ti.kernel
    def render_custom(
        pixels: ti.template(),
        width: ti.i32, height: ti.i32,
        max_iter: ti.i32, threshold: ti.f64,
        c_real: ti.f64, c_imag: ti.f64,
        display_mode: ti.i32,
        view_scale: ti.f64, view_center_x: ti.f64, view_center_y: ti.f64,
        num_roots: ti.i32   # 用户指定的根数量
    ):
        ti.loop_config(block_dim=256)
        for i, j in ti.ndrange(width, height):
            aspect = ti.cast(width, ti.f64) / ti.cast(height, ti.f64)
            uv_x = (ti.cast(i, ti.f64) / ti.cast(width, ti.f64) - 0.5) * 2.0 * aspect
            uv_y = (0.5 - ti.cast(j, ti.f64) / ti.cast(height, ti.f64)) * 2.0
            uv_x = uv_x * view_scale + view_center_x
            uv_y = uv_y * view_scale + view_center_y

            z = c64(uv_x, uv_y)
            c = c64(c_real, c_imag)
            iterations = 0
            converged = False

            h = 1e-8

            for _ in range(max_iter):
                f_z = custom_f_func(z)

                z_plus = z + c64(h, 0.0)
                z_minus = z - c64(h, 0.0)
                f_plus = custom_f_func(z_plus)
                f_minus = custom_f_func(z_minus)
                df_z = c64((f_plus.x - f_minus.x) / (2.0 * h),
                           (f_plus.y - f_minus.y) / (2.0 * h))

                if (df_z.x * df_z.x + df_z.y * df_z.y) < 1e-16:
                    break

                z_new = z - cdiv(f_z, df_z)
                if display_mode == 1:   # Nova
                    z_new = z_new + c

                if display_mode == 0:
                    if (f_z.x * f_z.x + f_z.y * f_z.y) < threshold * threshold:
                        converged = True
                        break
                else:
                    diff = z_new - z
                    if (diff.x * diff.x + diff.y * diff.y) < threshold * threshold:
                        converged = True
                        break

                z = z_new
                iterations += 1

            if converged:
                if display_mode == 0:
                    # 根据用户指定的根数量进行角度分区着色
                    angle = ti.atan2(z.y, z.x)
                    root_idx = ti.cast(
                        (angle + ti.math.pi) / (2.0 * ti.math.pi) * ti.cast(num_roots, ti.f64), ti.i32
                    ) % num_roots
                    gray = ti.select(num_roots <= 1, 0.5,
                                     ti.cast(root_idx, ti.f32) / ti.cast(num_roots - 1, ti.f32))
                    pixels[j, i] = pack_color(gray, gray, gray)
                else:
                    # Nova 保持平滑着色
                    smooth = ti.cast(iterations, ti.f64) + 1.0
                    gray = 3.0 * ti.log(1.0 + ti.log(1.0 + smooth / ti.cast(max_iter, ti.f64)))
                    pixels[j, i] = pack_color(ti.cast(gray, ti.f32), ti.cast(gray, ti.f32), ti.cast(gray, ti.f32))
            else:
                pixels[j, i] = pack_color(0.0, 0.0, 0.0)

    custom_kernel = render_custom

# ==================== 显示组件 ====================
class FractalWidget(QWidget):
    def __init__(self, parent=None, is_left=True):
        super().__init__(parent)
        self.parent_app = parent
        self.is_left = is_left
        self.view_center = [0.0, 0.0]
        self.view_scale = 1.2
        self.is_dragging = False
        self.last_drag_pos = None
        self.taichi_field = None
        self.image = None
        self.needs_recompute = True

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(300, 300)

    def compute_image(self):
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return

        app = self.parent_app
        aa = app.antialias_factor
        render_w = w * aa
        render_h = h * aa

        if self.taichi_field is None or self.taichi_field.shape != (render_h, render_w):
            self.taichi_field = ti.field(dtype=ti.u32, shape=(render_h, render_w))

        if app.using_custom_function and custom_kernel is not None:
            custom_kernel(
                self.taichi_field, render_w, render_h,
                app.max_iterations, app.convergence_threshold,
                app.c_real, app.c_imag,
                0 if self.is_left else 1,
                float(self.view_scale), float(self.view_center[0]), float(self.view_center[1]),
                app.custom_num_roots
            )
        else:
            render_preset(
                self.taichi_field, render_w, render_h,
                app.max_iterations, app.convergence_threshold,
                app.c_real, app.c_imag,
                0 if self.is_left else 1,
                float(self.view_scale), float(self.view_center[0]), float(self.view_center[1]),
                app.function_type
            )

        arr = self.taichi_field.to_numpy()
        qimg = QImage(arr.tobytes(), render_w, render_h, render_w * 4, QImage.Format_RGB32)
        self.image = qimg.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.needs_recompute = False

    def paintEvent(self, event):
        if self.needs_recompute:
            self.compute_image()
        if self.image:
            painter = QPainter(self)
            painter.drawImage(self.rect(), self.image)
            painter.end()
        else:
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.black)
            painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_dragging = True
            self.last_drag_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            self.parent_app.toggle_realtime_update()
            if not self.parent_app.realtime_update:
                self.parent_app.set_fixed_c_value()

    def mouseMoveEvent(self, event):
        self.update_mouse_position(event)
        if self.parent_app.realtime_update and self.is_left and not self.is_dragging:
            self.handle_mouse_move()
        if self.is_dragging and event.buttons() & Qt.MiddleButton:
            current_pos = event.position()
            dx = current_pos.x() - self.last_drag_pos.x()
            dy = current_pos.y() - self.last_drag_pos.y()
            aspect = self.width() / self.height()
            dx_world = -dx / self.width() * 2.0 * aspect * self.view_scale
            dy_world = dy / self.height() * 2.0 * self.view_scale
            self.view_center[0] += dx_world
            self.view_center[1] += dy_world
            self.last_drag_pos = current_pos
            self.needs_recompute = True

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_dragging = False
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event):
        mouse_x = event.position().x()
        mouse_y = event.position().y()
        aspect = self.width() / self.height()
        norm_x = (mouse_x / self.width() - 0.5) * 2.0 * aspect
        norm_y = (0.5 - mouse_y / self.height()) * 2.0
        mouse_real = norm_x * self.view_scale + self.view_center[0]
        mouse_imag = norm_y * self.view_scale + self.view_center[1]
        zoom_factor = 1.1 if event.angleDelta().y() <= 0 else 1.0 / 1.1
        self.view_scale *= zoom_factor
        self.view_center[0] = mouse_real - (mouse_real - self.view_center[0]) * zoom_factor
        self.view_center[1] = mouse_imag - (mouse_imag - self.view_center[1]) * zoom_factor
        self.needs_recompute = True

    def update_mouse_position(self, event):
        if self.parent_app:
            self.parent_app.mouse_pos[0] = event.position().x()
            self.parent_app.mouse_pos[1] = event.position().y()

    def handle_mouse_move(self):
        if not self.parent_app:
            return
        x_ratio = self.parent_app.mouse_pos[0] / self.width()
        y_ratio = self.parent_app.mouse_pos[1] / self.height()
        aspect = self.width() / self.height()
        norm_x = (x_ratio - 0.5) * 2.0 * aspect
        norm_y = (0.5 - y_ratio) * 2.0
        real = norm_x * self.view_scale + self.view_center[0]
        imag = norm_y * self.view_scale + self.view_center[1]
        self.parent_app.set_c_value(real, imag)

# ==================== 主窗口 ====================
class NewtonNovaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("牛顿分形与Nova分形 (Taichi 预设 + 动态自定义)")
        self.setGeometry(100, 100, 1200, 700)

        self.max_iterations = 300
        self.convergence_threshold = 1e-6
        self.c_real = 0.0
        self.c_imag = 0.0
        self.mouse_pos = [-1.0, -1.0]
        self.function_type = 0
        self.realtime_update = True
        self.fixed_c_value = [0.0, 0.0]
        self.using_custom_function = False
        self.custom_num_roots = 3   # 默认根数量
        self.antialias_factor = 1

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(self.create_control_panel())
        main_layout.addWidget(self.create_display_area())

    def create_control_panel(self):
        control_widget = QWidget()
        control_widget.setMaximumHeight(150)
        control_layout = QHBoxLayout(control_widget)
        control_layout.setSpacing(6)

        # 参数组
        param_group = QGroupBox("参数设置")
        param_group.setFixedWidth(200)
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(6)

        func_layout = QHBoxLayout()
        func_label = QLabel("函数:")
        func_label.setFixedWidth(30)
        func_layout.addWidget(func_label)
        self.function_combo = QComboBox()
        self.function_combo.addItems([
            "z³ - 1", "z⁴ - 1", "z⁵ - 1", "z³ - z",
            "z⁴ + z² + 1", "sin(z)", "cos(z)"
        ])
        self.function_combo.currentIndexChanged.connect(self.on_function_changed)
        self.function_combo.setFixedWidth(120)
        func_layout.addWidget(self.function_combo)
        param_layout.addLayout(func_layout)

        iter_layout = QHBoxLayout()
        iter_label = QLabel("迭代:")
        iter_label.setFixedWidth(30)
        iter_layout.addWidget(iter_label)
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 1000)
        self.iter_spin.setValue(self.max_iterations)
        self.iter_spin.valueChanged.connect(self.on_parameters_changed)
        self.iter_spin.setFixedWidth(120)
        iter_layout.addWidget(self.iter_spin)
        param_layout.addLayout(iter_layout)

        thresh_layout = QHBoxLayout()
        thresh_label = QLabel("阈值:")
        thresh_label.setFixedWidth(30)
        thresh_layout.addWidget(thresh_label)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(-12, -1)
        self.threshold_spin.setDecimals(0)
        self.threshold_spin.setValue(-6)
        self.threshold_spin.valueChanged.connect(self.on_threshold_changed)
        self.threshold_spin.setFixedWidth(120)
        thresh_layout.addWidget(self.threshold_spin)
        param_layout.addLayout(thresh_layout)

        aa_layout = QHBoxLayout()
        aa_label = QLabel("抗锯齿:")
        aa_label.setFixedWidth(50)
        aa_layout.addWidget(aa_label)
        self.aa_combo = QComboBox()
        self.aa_combo.addItems(["1x (无)", "2x", "4x"])
        self.aa_combo.currentIndexChanged.connect(self.on_aa_changed)
        self.aa_combo.setFixedWidth(120)
        aa_layout.addWidget(self.aa_combo)
        param_layout.addLayout(aa_layout)

        control_layout.addWidget(param_group)

        # 状态与控制组
        control_status_group = QGroupBox("状态与控制")
        control_status_group.setFixedWidth(380)
        status_layout = QVBoxLayout(control_status_group)
        status_layout.setSpacing(6)

        self.c_label = QLabel(f"c = {self.c_real:.4f} + {self.c_imag:.4f}i")
        self.c_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.c_label)

        btn_layout = QHBoxLayout()
        self.realtime_button = QPushButton("实时更新: 开")
        self.realtime_button.setFixedWidth(90)
        self.realtime_button.clicked.connect(self.toggle_realtime_update)
        btn_layout.addWidget(self.realtime_button)

        reset_btn = QPushButton("重置视图")
        reset_btn.setFixedWidth(90)
        reset_btn.clicked.connect(self.reset_views)
        btn_layout.addWidget(reset_btn)

        apply_btn = QPushButton("应用自定义函数")
        apply_btn.setFixedWidth(120)
        apply_btn.clicked.connect(self.apply_custom_function)
        btn_layout.addWidget(apply_btn)
        status_layout.addLayout(btn_layout)

        info_label = QLabel("左键:切换实时更新 中键:拖拽 滚轮:缩放 左:牛顿 右:Nova")
        status_layout.addWidget(info_label)

        control_layout.addWidget(control_status_group)

        # 自定义函数输入区域（增加根数量输入）
        custom_group = QGroupBox("自定义函数 (Taichi 表达式，需指定根数量)")
        custom_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        custom_layout = QVBoxLayout(custom_group)
        self.custom_code_edit = QPlainTextEdit()
        self.custom_code_edit.setMaximumHeight(70)
        self.custom_code_edit.setPlaceholderText(
            "输入 f(z) 的 Taichi 表达式，例如 cpow(z, 4.0) - c64(1.0, 0.0)"
        )
        custom_layout.addWidget(self.custom_code_edit)

        # 根数量行
        roots_layout = QHBoxLayout()
        roots_label = QLabel("根数量:")
        roots_label.setFixedWidth(50)
        roots_layout.addWidget(roots_label)
        self.roots_spin = QSpinBox()
        self.roots_spin.setRange(1, 20)
        self.roots_spin.setValue(self.custom_num_roots)
        roots_layout.addWidget(self.roots_spin)
        roots_layout.addStretch()
        custom_layout.addLayout(roots_layout)

        control_layout.addWidget(custom_group)

        return control_widget

    def create_display_area(self):
        display_widget = QWidget()
        display_layout = QHBoxLayout(display_widget)
        splitter = QSplitter(Qt.Horizontal)
        self.left_widget = FractalWidget(self, is_left=True)
        self.right_widget = FractalWidget(self, is_left=False)
        splitter.addWidget(self.left_widget)
        splitter.addWidget(self.right_widget)
        splitter.setSizes([500, 500])
        display_layout.addWidget(splitter)
        return display_widget

    def set_c_value(self, real, imag):
        self.c_real = real
        self.c_imag = imag
        self.c_label.setText(f"c = {real:.4f} + {imag:.4f}i")
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def set_fixed_c_value(self):
        self.fixed_c_value = [self.c_real, self.c_imag]

    def on_function_changed(self, index):
        self.using_custom_function = False
        self.function_type = index
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def on_aa_changed(self, idx):
        self.antialias_factor = 2 ** idx
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def apply_custom_function(self):
        expr = self.custom_code_edit.toPlainText().strip()
        if not expr:
            QMessageBox.warning(self, "警告", "请输入自定义函数表达式")
            return

        num_roots = self.roots_spin.value()
        self.custom_num_roots = num_roots

        scope = {
            'ti': ti,
            'c64': c64,
            'csqr': csqr,
            'cconj': cconj,
            'cmul': cmul,
            'cdiv': cdiv,
            'csin': csin,
            'ccos': ccos,
            'cexp': cexp,
            'clog': clog,
            'cpow': cpow,
        }
        try:
            func_def = f"def custom_f_func(z: c64) -> c64:\n    return {expr}\n"
            exec(func_def, scope)
            global custom_f_func
            custom_f_func = scope['custom_f_func']
            build_custom_kernel()
            self.using_custom_function = True
            self.left_widget.needs_recompute = True
            self.right_widget.needs_recompute = True
            QMessageBox.information(self, "成功",
                f"自定义函数已应用，根数量 = {num_roots}，牛顿分形将使用 {num_roots} 级灰度")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"自定义函数编译失败:\n{str(e)}")

    def on_parameters_changed(self):
        self.max_iterations = self.iter_spin.value()
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def on_threshold_changed(self, value):
        self.convergence_threshold = 10.0 ** value
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def toggle_realtime_update(self):
        self.realtime_update = not self.realtime_update
        self.realtime_button.setText(f"实时更新: {'开' if self.realtime_update else '关'}")
        if not self.realtime_update:
            self.set_fixed_c_value()
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def reset_views(self):
        self.left_widget.view_center = [0.0, 0.0]
        self.left_widget.view_scale = 1.2
        self.right_widget.view_center = [0.0, 0.0]
        self.right_widget.view_scale = 1.2
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True

    def closeEvent(self, event):
            self.left_widget.timer.stop()
            self.right_widget.timer.stop()
            ti.reset()
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NewtonNovaApp()
    window.show()
    sys.exit(app.exec())