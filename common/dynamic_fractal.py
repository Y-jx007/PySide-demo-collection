"""Shared canvas base class for the complex-plane fractal apps."""

from dataclasses import dataclass

import taichi_forge as ti
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

# 交互进行中先用 1x 采样出预览；停手后由 QTimer 触发一次全质量精修。
PREVIEW_ANTIALIAS = 1
PREVIEW_REFINE_MS = 180


def render_dimensions(width, height, antialias, *, preview=False,
                       preview_antialias=PREVIEW_ANTIALIAS):
    """Return (display_w, display_h, render_w, render_h).

    During interaction the canvas renders one sample per pixel, which is
    roughly an order of magnitude cheaper than the selected AA factor; the
    refine pass restores full quality once the user stops.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    aa = preview_antialias if preview else max(1, int(antialias))
    return width, height, width * aa, height * aa


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
        self._field_cache = {}
        self.image = None
        self._image_buffer = None
        self.needs_recompute = True

        # 交互预览：拖拽/缩放/拖参数时先画 1x，停手 PREVIEW_REFINE_MS 后精修。
        self._interactive = False
        self._refine_timer = QTimer(self)
        self._refine_timer.setSingleShot(True)
        self._refine_timer.setInterval(PREVIEW_REFINE_MS)
        self._refine_timer.timeout.connect(self._end_preview)

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
        """Stop the repaint/refine timers; called by the owning window on close."""
        self._refine_timer.stop()
        self.timer.stop()

    # ---------- interaction preview ----------
    @property
    def interactive(self) -> bool:
        return self._interactive

    def request_preview(self):
        """Draw a cheap 1x preview now; refine at full quality when idle."""
        self._interactive = True
        self.needs_recompute = True
        self._refine_timer.start()
        self.update()

    def _end_preview(self):
        if not self._interactive:
            return
        self._interactive = False
        self.needs_recompute = True
        self.update()

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
        if getattr(self.parent_app, "high_dpi_render", False):
            dpr = max(1.0, float(self.devicePixelRatioF()))
            w = max(1, int(round(w * dpr)))
            h = max(1, int(round(h * dpr)))
        return render_dimensions(
            w, h, self.parent_app.antialias_factor, preview=self._interactive,
            preview_antialias=getattr(
                self.parent_app, "preview_antialias", PREVIEW_ANTIALIAS),
        )

    def resizeEvent(self, event):
        # A widget resize invalidates the cached bitmap.  Without this, the
        # old image is stretched to the new rect, which is why one canvas
        # could show square pixels while the other showed rectangles.
        super().resizeEvent(event)
        self.needs_recompute = True
        self.update()

    def showEvent(self, event):
        # The initial QSplitter layout can settle after the first paint.
        super().showEvent(event)
        self.needs_recompute = True
        self.update()

    def ensure_taichi_field(self, render_w, render_h):
        # 预览（1x）与精修（AA）尺寸不同；缓存最近两个尺寸，避免每次开始
        # 交互都重新分配几十 MB 的 field（分配 + 首次写入约几十毫秒）。
        key = (render_h, render_w)
        field = self._field_cache.get(key)
        if field is None:
            field = ti.field(dtype=ti.u32, shape=key)
            if len(self._field_cache) >= 2:
                self._field_cache.pop(next(iter(self._field_cache)))
            self._field_cache[key] = field
        self.taichi_field = field

    def field_to_image(self, render_w, render_h, target_w, target_h):
        array = self.taichi_field.to_numpy()
        buffer = None
        try:
            # QImage 只是一层视图，必须把缓冲区保活到下一次绘制；这里直接持有
            # numpy 数组，省掉一遍 tobytes()。
            image = QImage(
                array.data, render_w, render_h, render_w * 4, QImage.Format_RGB32
            )
        except TypeError:                      # 极少数 PySide 版本不接受 buffer
            buffer = array.tobytes()
            image = QImage(
                buffer, render_w, render_h, render_w * 4, QImage.Format_RGB32
            )
        if (render_w, render_h) != (target_w, target_h):
            image = image.scaled(
                target_w, target_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            buffer = None
        elif buffer is None:
            buffer = array
        if getattr(self.parent_app, "high_dpi_render", False):
            image.setDevicePixelRatio(max(1.0, float(self.devicePixelRatioF())))
        self.image = image
        self._image_buffer = buffer
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
            self.request_preview()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_dragging = False
            self.setCursor(Qt.ArrowCursor)
            self._refine_timer.start()

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
        self.request_preview()

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
            # Exact-size bitmap; avoid a second smooth resample.
            painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
            if getattr(self.parent_app, "high_dpi_render", False):
                # Draw at the origin so the physical-pixel bitmap is never
                # scaled by a fractional logical-size difference.
                painter.fillRect(self.rect(), Qt.black)
                painter.drawImage(QPointF(0.0, 0.0), self.image)
            else:
                painter.drawImage(self.rect(), self.image)
        else:
            painter.fillRect(self.rect(), Qt.black)
        painter.end()
