"""Shared canvas base class for the complex-plane fractal apps."""

from dataclasses import dataclass

import taichi_forge as ti
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget


@dataclass(frozen=True)
class ViewTransform:
    """Affine mapping between widget pixels and the complex plane.

    The kernels use the same affine form (``world = pixel * scale + offset``)
    so host-side and device-side coordinates stay in sync.
    """

    width: int
    height: int
    view_scale: float
    center_x: float
    center_y: float

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height > 0 else 1.0

    @property
    def scale_x(self) -> float:
        return 2.0 * self.aspect * self.view_scale / self.width if self.width > 0 else 0.0

    @property
    def offset_x(self) -> float:
        return -self.aspect * self.view_scale + self.center_x

    @property
    def scale_y(self) -> float:
        return -2.0 * self.view_scale / self.height if self.height > 0 else 0.0

    @property
    def offset_y(self) -> float:
        return self.view_scale + self.center_y

    def screen_to_world(self, x: float, y: float) -> tuple[float, float]:
        if self.width <= 0 or self.height <= 0:
            return self.center_x, self.center_y
        norm_x = (x / self.width - 0.5) * 2.0 * self.aspect
        norm_y = (0.5 - y / self.height) * 2.0
        return (
            norm_x * self.view_scale + self.center_x,
            norm_y * self.view_scale + self.center_y,
        )

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        if self.width <= 0 or self.height <= 0:
            return 0.0, 0.0
        if self.scale_x == 0.0 or self.scale_y == 0.0:
            return self.width * 0.5, self.height * 0.5
        screen_x = self.width * 0.5 + (x - self.center_x) / self.scale_x
        screen_y = self.height * 0.5 + (y - self.center_y) / self.scale_y
        return screen_x, screen_y


class BaseFractalWidget(QWidget):
    """Double-buffered Taichi canvas with pan/zoom and dirty tracking."""

    def __init__(self, parent, is_left: bool):
        super().__init__(parent)
        self.parent_app = parent
        self.is_left = is_left
        self.view_center = [0.0, 0.0]
        self.view_scale = 1.0          # subclasses may change the default
        self.is_dragging = False
        self.last_drag_pos = None
        self.taichi_field = None
        self.image = None
        self._image_buffer = None
        self.needs_recompute = True

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(30)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(300, 300)

    def _on_timer(self):
        # Only repaint when something actually changed, so an idle canvas
        # does not keep waking up the GUI thread.
        if self.needs_recompute:
            self.update()

    def shutdown(self):
        """Stop the repaint timer; called by the owning window on close."""
        self.timer.stop()

    # ---------- subclass hook ----------
    def compute_image(self):
        raise NotImplementedError

    # ---------- shared helpers ----------
    def view_transform(
        self, width: int | None = None, height: int | None = None
    ) -> ViewTransform:
        return ViewTransform(
            self.width() if width is None else width,
            self.height() if height is None else height,
            self.view_scale,
            self.view_center[0],
            self.view_center[1],
        )

    def screen_to_world(self, x: float, y: float) -> tuple[float, float]:
        return self.view_transform().screen_to_world(x, y)

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        return self.view_transform().world_to_screen(x, y)

    def get_render_dims(self):
        """Return (display_w, display_h, render_w, render_h)."""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return None, None, None, None
        aa = max(1, int(self.parent_app.antialias_factor))
        return w, h, w * aa, h * aa

    def ensure_taichi_field(self, render_w, render_h):
        if self.taichi_field is None or self.taichi_field.shape != (render_h, render_w):
            self.taichi_field = ti.field(dtype=ti.u32, shape=(render_h, render_w))

    def field_to_image(self, render_w, render_h, target_w, target_h):
        arr = self.taichi_field.to_numpy()
        buffer = arr.tobytes()
        image = QImage(buffer, render_w, render_h, render_w * 4, QImage.Format_RGB32)
        if (render_w, render_h) != (target_w, target_h):
            image = image.scaled(
                target_w, target_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            new_buffer = None
        else:
            # QImage does not copy the buffer; keep it alive with the widget.
            new_buffer = buffer
        self.image = image
        self._image_buffer = new_buffer
        self.needs_recompute = False

    # ---------- mouse and view interaction ----------
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
            transform = self.view_transform()
            self.view_center[0] -= dx * transform.scale_x
            self.view_center[1] -= dy * transform.scale_y
            self.last_drag_pos = current_pos
            self.needs_recompute = True

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_dragging = False
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event):
        mouse_real, mouse_imag = self.screen_to_world(
            event.position().x(), event.position().y()
        )
        zoom_factor = 1.1 if event.angleDelta().y() <= 0 else 1.0 / 1.1
        self.view_scale *= zoom_factor
        self.view_center[0] = (
            mouse_real - (mouse_real - self.view_center[0]) * zoom_factor
        )
        self.view_center[1] = (
            mouse_imag - (mouse_imag - self.view_center[1]) * zoom_factor
        )
        self.needs_recompute = True

    def update_mouse_position(self, event):
        if self.parent_app:
            self.parent_app.mouse_pos[0] = event.position().x()
            self.parent_app.mouse_pos[1] = event.position().y()

    def handle_mouse_move(self):
        """Map the mouse position to the complex plane and update c."""
        if not self.parent_app:
            return
        real, imag = self.screen_to_world(
            self.parent_app.mouse_pos[0], self.parent_app.mouse_pos[1]
        )
        self.parent_app.set_c_value(real, imag)

    def paintEvent(self, event):
        if self.needs_recompute:
            self.compute_image()
        painter = QPainter(self)
        if self.image is not None and not self.image.isNull():
            painter.drawImage(self.rect(), self.image)
        else:
            painter.fillRect(self.rect(), Qt.black)
        painter.end()
