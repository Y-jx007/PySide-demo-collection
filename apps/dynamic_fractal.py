import numpy as np
import taichi_forge as ti
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QImage, QColor, QPixmap
from PySide6.QtWidgets import QWidget

class BaseFractalWidget(QWidget):
    
    def __init__(self, parent, is_left: bool):
        super().__init__(parent)
        self.parent_app = parent
        self.is_left = is_left
        self.view_center = [0.0, 0.0]
        self.view_scale = 1.0          # 子类可在构造后修改默认值
        self.is_dragging = False
        self.last_drag_pos = None
        self.taichi_field = None
        self.image = None
        self.needs_recompute = True

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(300, 300)

    # ---------- 子类必须实现的方法 ----------
    def compute_image(self):
        raise NotImplementedError

    # ---------- 公共工具方法 ----------
    def get_render_dims(self):
        """返回 (显示宽, 显示高, 渲染宽, 渲染高)"""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return None, None, None, None
        aa = self.parent_app.antialias_factor
        render_w, render_h = w * aa, h * aa
        return w, h, render_w, render_h

    def ensure_taichi_field(self, render_w, render_h):
        if self.taichi_field is None or self.taichi_field.shape != (render_h, render_w):
            self.taichi_field = ti.field(dtype=ti.u32, shape=(render_h, render_w))

    def field_to_image(self, render_w, render_h, target_w, target_h):
        arr = self.taichi_field.to_numpy()
        qimg = QImage(arr.tobytes(), render_w, render_h, render_w * 4, QImage.Format_RGB32)
        self.image = qimg.scaled(target_w, target_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.needs_recompute = False

    # ---------- 鼠标与视图交互 ----------
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
            aspect = self.width() / self.height() if self.height() > 0 else 1.0
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
        aspect = self.width() / self.height() if self.height() > 0 else 1.0
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
        """将鼠标位置映射为复平面坐标，并通知父窗口更新参数 c"""
        if not self.parent_app:
            return
        x_ratio = self.parent_app.mouse_pos[0] / self.width()
        y_ratio = self.parent_app.mouse_pos[1] / self.height()
        aspect = self.width() / self.height() if self.height() > 0 else 1.0
        norm_x = (x_ratio - 0.5) * 2.0 * aspect
        norm_y = (0.5 - y_ratio) * 2.0
        real = norm_x * self.view_scale + self.view_center[0]
        imag = norm_y * self.view_scale + self.view_center[1]
        self.parent_app.set_c_value(real, imag, update_spinners=False)

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