# -*- coding: utf-8 -*-
"""用 numpy 图像当画面的画布基类（Erosion / MPM2D 共用）。

两个 app 原先各自写着同一段：

    side = min(w, h)
    ox, oy = (w - side) / 2, (h - side) / 2
    painter.drawImage(QRectF(ox, oy, side, side), qimg)

外加各写一份「鼠标 → 网格坐标」的换算。ArrayCanvas 把这几件事收拢：

* 等比居中绘制（默认 1:1 正方形，可改成任意宽高比或跟随图像）；
* set_array() 直接给一帧，set_renderer() 走懒渲染（paint 时才生成）；
* image_rect() / map_normalized() / map_pos() 提供坐标映射；
* 可选的背景色与描边。
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from common.imaging import ndarray_to_qimage


class ArrayCanvas(QWidget):
    """把 numpy 图像按比例居中画出来的画布。

    renderer 是一个返回 numpy 数组的回调：调用 invalidate() 之后的下一次
    paint 才会真正生成图像，适合「数据每帧变、但没必要每帧重算颜色」的场景。
    已经算好的图像则用 set_array() 直接塞进来。
    """

    def __init__(self, *, aspect: float = 1.0, background=None, border=None,
                 smooth: bool = True, renderer=None, minimum=(320, 320),
                 padding: int = 0, parent=None):
        super().__init__(parent)
        self._aspect = float(aspect)
        self._padding = max(0, int(padding))
        self._background = QColor(background) if background is not None else None
        self._border = QColor(border) if border is not None else None
        self._smooth = bool(smooth)
        self._renderer = renderer
        self._array = None
        self._image: QImage | None = None
        self._dirty = True
        if minimum:
            self.setMinimumSize(*minimum)

    # -- 数据 --------------------------------------------------------
    def set_renderer(self, renderer) -> None:
        """设置懒渲染回调（返回 numpy 数组）；传 None 表示只接受 set_array。"""
        self._renderer = renderer
        self.invalidate()

    def set_array(self, array) -> None:
        """直接设置要显示的数组（或 QImage）；传 None 清空。

        同一个对象不会重复转换——调用方每次生成新的 QImage/数组即可。
        """
        if array is self._array:
            return
        self._array = array
        if array is None:
            self._image = None
        elif isinstance(array, QImage):
            self._image = array
        else:
            self._image = ndarray_to_qimage(array)
        self._dirty = False
        self.update()

    def array(self):
        return self._array

    def image(self) -> QImage | None:
        return self._image

    def invalidate(self) -> None:
        """标记内容过期，下一次 paint 重新生成。"""
        self._dirty = True
        self.update()

    # -- 几何 --------------------------------------------------------
    def image_rect(self) -> QRectF:
        """当前图像在控件里的绘制区域（等比、居中，已扣除 padding）。"""
        width, height = self.width(), self.height()
        avail_w = width - 2 * self._padding
        avail_h = height - 2 * self._padding
        if avail_w <= 0 or avail_h <= 0:
            return QRectF()
        aspect = self._aspect
        if aspect <= 0:                       # 跟随图像自身比例
            source = self._image
            aspect = (source.width() / source.height()) if source and source.height() else 1.0
        if avail_w / avail_h > aspect:
            draw_w, draw_h = avail_h * aspect, avail_h
        else:
            draw_w, draw_h = avail_w, avail_w / aspect
        return QRectF((width - draw_w) / 2.0, (height - draw_h) / 2.0, draw_w, draw_h)

    def map_normalized(self, pos):
        """控件坐标 → 图像归一化坐标 (u, v)，超出边界也照常返回。"""
        rect = self.image_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return None
        return ((pos.x() - rect.x()) / rect.width(),
                (pos.y() - rect.y()) / rect.height())

    def map_pos(self, pos):
        """控件坐标 → 归一化坐标；落在图像之外时返回 None。"""
        point = self.map_normalized(pos)
        if point is None:
            return None
        u, v = point
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            return u, v
        return None

    # -- 绘制 --------------------------------------------------------
    def paintEvent(self, event):
        if self._dirty and self._renderer is not None:
            self._renderer_result(self._renderer())
        painter = QPainter(self)
        if self._smooth:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if self._background is not None:
            painter.fillRect(self.rect(), self._background)
        if self._image is not None:
            rect = self.image_rect()
            painter.drawImage(rect, self._image)
            if self._border is not None:
                painter.setPen(QPen(self._border, 1))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect)
            self.paint_overlay(painter, rect)
        painter.end()

    def paint_overlay(self, painter: QPainter, rect: QRectF) -> None:
        """子类可覆盖：在图像上方画标注（rect 是图像的绘制区域）。"""

    def _renderer_result(self, array) -> None:
        self._array = array
        self._image = None if array is None else ndarray_to_qimage(array)
        self._dirty = False
