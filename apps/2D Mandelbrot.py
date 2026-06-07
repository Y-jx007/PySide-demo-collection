from custom_import import *
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox,
    QSpinBox, QComboBox, QGroupBox, QSplitter,
    QMessageBox, QPlainTextEdit, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QFontDatabase, QImage, QPainter
ti.init(arch=ti.gpu)
# 定义双精度复数向量类型
c64 = ti.types.vector(2, ti.f64)
# -------------------- 复数运算函数（全部双精度） --------------------
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
    return c64(
        ti.select(safe, real, 0.0),
        ti.select(safe, imag, 0.0)
    )
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
# 预设函数表
@ti.func
def preset_func(z: c64, func_type: ti.i32) -> c64:
    result = z
    if func_type == 0:
        result = csqr(z)
    elif func_type == 1:
        result = cmul(z, csqr(z))
    elif func_type == 2:
        result = csqr(csqr(z))
    elif func_type == 3:
        z2 = csqr(z)
        z3 = cmul(z, z2)
        result = cmul(z2, z3)
    elif func_type == 4:
        result = csqr(z) + z
    elif func_type == 5:
        result = cdiv(c64(1.0, 0.0), z)
    elif func_type == 6:
        result = csin(z)
    elif func_type == 7:
        result = ccos(z)
    elif func_type == 8:
        result = cexp(z)
    elif func_type == 9:
        result = cconj(csqr(z))
    elif func_type == 10:
        result = csqr(c64(ti.abs(z.x), ti.abs(z.y)))
    else:
        result = csqr(z)
    return result
# 自定义函数占位
@ti.func
def custom_func(z: c64) -> c64:
    return csqr(z)
# -------------------- 分形计算 Kernel --------------------
@ti.func
def pack_color(r: ti.f32, g: ti.f32, b: ti.f32) -> ti.u32:
    ri = ti.cast(ti.min(255, ti.max(0, r * 255.0)), ti.u32)
    gi = ti.cast(ti.min(255, ti.max(0, g * 255.0)), ti.u32)
    bi = ti.cast(ti.min(255, ti.max(0, b * 255.0)), ti.u32)
    return (ti.u32(0xff) << 24) | (ri << 16) | (gi << 8) | bi
fractal_kernel = None
def build_kernel():
    global fractal_kernel
    @ti.kernel
    def render(
        pixels: ti.template(),
        width: ti.i32, height: ti.i32,
        max_iter: ti.i32, escape_radius: ti.f64,
        c_real: ti.f64, c_imag: ti.f64,
        display_mode: ti.i32,
        view_scale: ti.f64, view_center_x: ti.f64, view_center_y: ti.f64,
        func_type: ti.i32,
        use_custom: ti.i32,
        mouse_wx: ti.f64, mouse_wy: ti.f64, mouse_valid: ti.i32
    ):
        # 优化块大小，提升 GPU 占用率
        ti.loop_config(block_dim=256)
        for i, j in ti.ndrange(width, height):
            # 屏幕坐标 -> 复平面坐标（基于当前渲染的分辨率）
            aspect = ti.cast(width, ti.f64) / ti.cast(height, ti.f64)
            uv_x = (ti.cast(i, ti.f64) / ti.cast(width, ti.f64) - 0.5) * 2.0 * aspect
            uv_y = (0.5 - ti.cast(j, ti.f64) / ti.cast(height, ti.f64)) * 2.0
            uv_x = uv_x * view_scale + view_center_x
            uv_y = uv_y * view_scale + view_center_y
            z = c64(0.0, 0.0)
            c = c64(0.0, 0.0)
            if display_mode == 0:       # Mandelbrot 模式
                c = c64(uv_x, uv_y)
            else:                       # Julia 模式
                c = c64(c_real, c_imag)
                z = c64(uv_x, uv_y)
            iterations = 0
            escaped = False
            for _ in range(max_iter):
                if (z.x * z.x + z.y * z.y) > escape_radius * escape_radius:
                    escaped = True
                    break
                if use_custom:
                    z = custom_func(z) + c
                else:
                    z = preset_func(z, func_type) + c
                iterations += 1
            # 默认集内颜色（白色）
            color_r = 1.0
            color_g = 1.0
            color_b = 1.0
            if escaped:
                # 平滑着色
                log_zn = ti.log(z.x * z.x + z.y * z.y) / 2.0
                nu = ti.log(log_zn / ti.log(escape_radius)) / ti.log(2.0)
                smooth_iter = ti.cast(iterations, ti.f64) + 1.0 - nu
                cval = ti.cast(smooth_iter / ti.cast(max_iter, ti.f64), ti.f32)
                color_r = cval
                color_g = cval
                color_b = cval
            # 绘制轨道（仅 Mandelbrot 模式且鼠标有效时）
            if display_mode == 0 and mouse_valid:
                orbit_z = c64(0.0, 0.0)
                orbit_c = c64(mouse_wx, mouse_wy)
                for _ in range(max_iter):
                    next_z = c64(0.0, 0.0)
                    if use_custom:
                        next_z = custom_func(orbit_z) + orbit_c
                    else:
                        next_z = preset_func(orbit_z, func_type) + orbit_c
                    ab = next_z - orbit_z
                    ap = c64(uv_x, uv_y) - orbit_z
                    dot_ab_ab = ab.x * ab.x + ab.y * ab.y
                    t = 0.0
                    if dot_ab_ab > 1e-12:
                        t = ti.math.clamp((ap.x * ab.x + ap.y * ab.y) / dot_ab_ab, 0.0, 1.0)
                    closest = orbit_z + t * ab
                    dx = uv_x - closest.x
                    dy = uv_y - closest.y
                    dist_sq = dx * dx + dy * dy
                    threshold = 0.001 * view_scale
                    if dist_sq < threshold * threshold:
                        color_r = 0.95
                        color_g = 0.76
                        color_b = 0.07
                        break
                    orbit_z = next_z
                    if orbit_z.x * orbit_z.x + orbit_z.y * orbit_z.y > escape_radius * escape_radius:
                        break
            pixels[j, i] = pack_color(color_r, color_g, color_b)
    fractal_kernel = render
build_kernel()
# -------------------- 中文字体设置 --------------------
def setup_chinese_font():
    families = QFontDatabase().families()
    font_families = ["Microsoft YaHei", "SimHei", "SimSun", "NSimSun"]
    for font_family in font_families:
        if font_family in families:
            app_font = QFont(font_family, 9)
            QApplication.setFont(app_font)
            break
# -------------------- 分形显示组件 --------------------
class FractalWidget(QWidget):
    def __init__(self, parent=None, is_left=True):
        super().__init__(parent)
        self.parent_app = parent
        self.is_left = is_left
        self.view_center = [0.0, 0.0]
        self.view_scale = 1.0
        self.is_dragging = False
        self.last_drag_pos = None
        self.pixels_field = None
        self.image = None
        self.needs_recompute = True
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(300, 300)
    def compute_image(self):
        if not fractal_kernel:
            return
        # 原始窗口尺寸
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return
        app = self.parent_app
        # 获取抗锯齿因子
        aa = app.antialias_factor
        # 渲染分辨率
        render_w = w * aa
        render_h = h * aa
        if self.pixels_field is None or self.pixels_field.shape != (render_h, render_w):
            self.pixels_field = ti.field(dtype=ti.u32, shape=(render_h, render_w))
        max_iter = app.max_iterations
        escape_r = float(app.escape_radius)
        c_r = float(app.c_real)
        c_i = float(app.c_imag)
        display_mode = 0 if self.is_left else 1
        func_type = app.function_type
        use_custom = 1 if app.using_custom_function else 0
        # 鼠标世界坐标计算（始终基于原始窗口尺寸，保证轨道位置准确）
        mouse_wx = 0.0
        mouse_wy = 0.0
        mouse_valid = 0
        if app.mouse_pos[0] >= 0:
            aspect = w / h
            norm_x = (app.mouse_pos[0] / w - 0.5) * 2.0 * aspect
            norm_y = (0.5 - app.mouse_pos[1] / h) * 2.0
            mouse_wx = norm_x * self.view_scale + self.view_center[0]
            mouse_wy = norm_y * self.view_scale + self.view_center[1]
            if self.is_left:
                if app.realtime_update:
                    mouse_valid = 1
                else:
                    mouse_valid = 1
                    mouse_wx = app.fixed_orbit_pos[0]
                    mouse_wy = app.fixed_orbit_pos[1]
            else:
                if app.realtime_update:
                    mouse_valid = 0
                else:
                    mouse_valid = 1
                    mouse_wx = app.fixed_orbit_pos[0]
                    mouse_wy = app.fixed_orbit_pos[1]
        fractal_kernel(
            self.pixels_field, render_w, render_h,
            max_iter, escape_r,
            c_r, c_i,
            display_mode,
            float(self.view_scale), float(self.view_center[0]), float(self.view_center[1]),
            func_type, use_custom,
            mouse_wx, mouse_wy, mouse_valid
        )
        # 将计算结果转为 QImage
        arr = self.pixels_field.to_numpy()
        # 创建与渲染分辨率一致的 QImage
        full_img = QImage(arr.tobytes(), render_w, render_h, render_w * 4, QImage.Format_RGB32)
        # 缩放到窗口大小，使用平滑变换
        self.image = full_img.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.needs_recompute = False
    def paintEvent(self, event):
        if self.needs_recompute:
            self.compute_image()
        if self.image is not None:
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
                self.update_mouse_position(event)
                self.parent_app.set_fixed_positions()
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
        zoom_factor = 1.1
        if event.angleDelta().y() > 0:
            zoom_factor = 1.0 / zoom_factor
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
        self.parent_app.set_c_value(real, imag, update_spinners=False)
# -------------------- 主窗口 --------------------
class ComplexDynamicsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("复动力系统可视化 (双精度 Taichi GPU + 抗锯齿)")
        self.setGeometry(100, 100, 1200, 700)
        self.max_iterations = 250
        self.escape_radius = 4.0
        self.c_real = -0.7
        self.c_imag = 0.27015
        self.mouse_pos = [-1.0, -1.0]
        self.function_type = 0
        self.realtime_update = True
        self.fixed_orbit_pos = [0.0, 0.0]
        self.fixed_c_value = [-0.7, 0.27015]
        self.using_custom_function = False
        self.antialias_factor = 1  # 新增：抗锯齿因子
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
        control_layout.setContentsMargins(6, 6, 6, 6)
        # 参数组（函数、迭代、半径、抗锯齿）
        param_group = QGroupBox("参数设置")
        param_group.setFixedWidth(200)
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(6)
        param_layout.setContentsMargins(6, 6, 6, 6)
        input_w = 120
        combo_w = 120
        # 函数选择
        func_layout = QHBoxLayout()
        func_label = QLabel("函数:")
        func_label.setFixedWidth(30)
        func_layout.addWidget(func_label)
        self.function_combo = QComboBox()
        self.function_combo.addItems([
            "z²", "z³", "z⁴", "z⁵", "z² + z", "1/z", "sin(z)", "cos(z)", "exp(z)",
            "z²共轭", "(|Re z|+i|Im z|)²"
        ])
        self.function_combo.setFixedWidth(combo_w)
        self.function_combo.currentIndexChanged.connect(self.on_function_changed)
        func_layout.addWidget(self.function_combo)
        param_layout.addLayout(func_layout)
        # 迭代次数
        iter_layout = QHBoxLayout()
        iter_label = QLabel("迭代:")
        iter_label.setFixedWidth(30)
        iter_layout.addWidget(iter_label)
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(50, 50000)
        self.iter_spin.setValue(self.max_iterations)
        self.iter_spin.valueChanged.connect(self.on_parameters_changed)
        self.iter_spin.setFixedWidth(input_w)
        iter_layout.addWidget(self.iter_spin)
        param_layout.addLayout(iter_layout)
        # 逃逸半径
        radius_layout = QHBoxLayout()
        radius_label = QLabel("半径:")
        radius_label.setFixedWidth(30)
        radius_layout.addWidget(radius_label)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(2.0, 20.0)
        self.radius_spin.setValue(self.escape_radius)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.valueChanged.connect(self.on_parameters_changed)
        self.radius_spin.setFixedWidth(input_w)
        radius_layout.addWidget(self.radius_spin)
        param_layout.addLayout(radius_layout)
        # 抗锯齿设置
        aa_layout = QHBoxLayout()
        aa_label = QLabel("抗锯齿:")
        aa_label.setFixedWidth(50)
        aa_layout.addWidget(aa_label)
        self.aa_combo = QComboBox()
        self.aa_combo.addItems(["1x (无)", "2x", "4x"])
        self.aa_combo.setCurrentIndex(0)
        self.aa_combo.currentIndexChanged.connect(self.on_aa_changed)
        self.aa_combo.setFixedWidth(input_w)
        aa_layout.addWidget(self.aa_combo)
        param_layout.addLayout(aa_layout)
        control_layout.addWidget(param_group)
        # 状态与控制组
        control_status_group = QGroupBox("状态与控制")
        control_status_group.setFixedWidth(380)
        control_status_layout = QVBoxLayout(control_status_group)
        control_status_layout.setSpacing(6)
        control_status_layout.setContentsMargins(6, 6, 6, 6)
        self.c_label = QLabel(f"c = {self.c_real:.4f} + {self.c_imag:.4f}i")
        self.c_label.setAlignment(Qt.AlignCenter)
        control_status_layout.addWidget(self.c_label)
        button_layout = QHBoxLayout()
        self.realtime_button = QPushButton("实时更新: 开")
        self.realtime_button.setFixedWidth(90)
        self.realtime_button.clicked.connect(self.toggle_realtime_update)
        button_layout.addWidget(self.realtime_button)
        reset_button = QPushButton("重置视图")
        reset_button.setFixedWidth(90)
        reset_button.clicked.connect(self.reset_views)
        button_layout.addWidget(reset_button)
        apply_button = QPushButton("应用自定义函数")
        apply_button.setFixedWidth(120)
        apply_button.clicked.connect(self.apply_custom_function)
        button_layout.addWidget(apply_button)
        control_status_layout.addLayout(button_layout)
        info_label = QLabel("左键:切换实时更新 中键:拖拽 滚轮:缩放 左图:Mandelbrot集 右图:Julia集")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size:10px; color:#666;")
        control_status_layout.addWidget(info_label)
        control_layout.addWidget(control_status_group)
        # 自定义函数输入组
        custom_group = QGroupBox("自定义函数 (Taichi 表达式)")
        custom_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        custom_layout = QVBoxLayout(custom_group)
        custom_layout.setSpacing(6)
        custom_layout.setContentsMargins(6, 6, 6, 6)
        self.custom_code_edit = QPlainTextEdit()
        self.custom_code_edit.setMaximumHeight(90)
        self.custom_code_edit.setPlaceholderText(
            "输入 Taichi 表达式，如: csqr(z) + csin(z)\n"
            "可用函数: csqr, cconj, cmul, cdiv, csin, ccos, cexp, clog, cpow(z, n)"
        )
        custom_layout.addWidget(self.custom_code_edit)
        control_layout.addWidget(custom_group)
        control_layout.setStretch(0, 0)
        control_layout.setStretch(1, 0)
        control_layout.setStretch(2, 1)
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
    def set_c_value(self, real, imag, update_spinners=True):
        self.c_real = real
        self.c_imag = imag
        self.c_label.setText(f"c = {real:.4f} + {imag:.4f}i")
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True
    def set_fixed_positions(self):
        self.fixed_c_value = [self.c_real, self.c_imag]
        if self.mouse_pos[0] >= 0:
            w = self.left_widget.width()
            h = self.left_widget.height()
            aspect = w / h
            norm_x = (self.mouse_pos[0] / w - 0.5) * 2.0 * aspect
            norm_y = (0.5 - self.mouse_pos[1] / h) * 2.0
            self.fixed_orbit_pos[0] = norm_x * self.left_widget.view_scale + self.left_widget.view_center[0]
            self.fixed_orbit_pos[1] = norm_y * self.left_widget.view_scale + self.left_widget.view_center[1]
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True
    def on_function_changed(self, index):
        if self.using_custom_function:
            self.using_custom_function = False
        self.function_type = index
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True
    def on_aa_changed(self, idx):
        self.antialias_factor = 2 ** idx
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True
    def apply_custom_function(self):
        custom_code = self.custom_code_edit.toPlainText().strip()
        if not custom_code:
            QMessageBox.warning(self, "警告", "请输入自定义函数表达式")
            return
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
            func_def = f"def custom_func(z: c64) -> c64:\n    return {custom_code}\n"
            exec(func_def, scope)
            global custom_func
            custom_func = scope['custom_func']
            build_kernel()
            self.using_custom_function = True
            self.left_widget.needs_recompute = True
            self.right_widget.needs_recompute = True
            QMessageBox.information(self, "成功", "自定义函数应用成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"自定义函数编译失败:\n{str(e)}")
    def on_parameters_changed(self):
        self.max_iterations = self.iter_spin.value()
        self.escape_radius = self.radius_spin.value()
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True
    def toggle_realtime_update(self):
        self.realtime_update = not self.realtime_update
        self.realtime_button.setText(f"实时更新: {'开' if self.realtime_update else '关'}")
        if not self.realtime_update:
            self.set_fixed_positions()
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True
    def reset_views(self):
        self.left_widget.view_center = [0.0, 0.0]
        self.left_widget.view_scale = 1.0
        self.right_widget.view_center = [0.0, 0.0]
        self.right_widget.view_scale = 1.0
        self.left_widget.needs_recompute = True
        self.right_widget.needs_recompute = True
    def closeEvent(self, event):
                ti.reset()
                event.accept()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    setup_chinese_font()
    window = ComplexDynamicsApp()
    window.show()
    sys.exit(app.exec())

