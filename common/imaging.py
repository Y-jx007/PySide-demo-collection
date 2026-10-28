# -*- coding: utf-8 -*-
"""numpy 图像 <-> Qt 图像，以及统一的打开/保存对话框。

各 app 原先各写一份 ndarray -> QImage/QPixmap 的转换、各写一遍
“选路径 + 补后缀 + 保存”的样板；这里统一掉：

* ndarray_to_qimage / ndarray_to_qpixmap：支持灰度、RGB、RGBA；
* ImageLabel：直接吃 numpy 数组的 QLabel，窗口缩放时自动重排；
* ask_open_image / ask_save_path：文件对话框（自动补后缀）；
* save_ndarray：用 Pillow 落盘。

依赖 numpy + PySide6，Pillow 只在真正保存时导入。
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QLabel

IMAGE_FILTERS = "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
PNG_FILTERS = "PNG (*.png)"
JPEG_FILTERS = "JPEG (*.jpg)"


def ndarray_to_qimage(array) -> QImage:
    """把 HxW / HxWx3 / HxWx4 的 numpy 数组转成独立持有内存的 QImage。

    直接 QImage(buf, ...) 只是一层视图，buf 被回收后就会花屏，所以这里
    统一 .copy() 一次。调用方不需要再保证数组连续。
    """
    array = np.ascontiguousarray(array)
    if array.ndim == 2:
        height, width = array.shape
        fmt = QImage.Format_Grayscale8
        stride = width
    elif array.ndim == 3:
        height, width, channels = array.shape
        if channels == 3:
            fmt, stride = QImage.Format_RGB888, width * 3
        elif channels == 4:
            fmt, stride = QImage.Format_RGBA8888, width * 4
        else:
            raise ValueError(f"不支持的通道数: {channels}")
    else:
        raise ValueError(f"不支持的数组形状: {array.shape}")
    image = QImage(array.data, width, height, stride, fmt)
    return image.copy()


def ndarray_to_qpixmap(array) -> QPixmap:
    """ndarray -> QPixmap（内部走 ndarray_to_qimage）。"""
    return QPixmap.fromImage(ndarray_to_qimage(array))


class ImageLabel(QLabel):
    """显示 numpy 图像并保持宽高比的 QLabel。

    原先 Arnold 里的 display_image() + resizeEvent() 组合就是这件事：
    记下原始 QPixmap，按当前控件尺寸（留出 margin）做平滑缩放。
    """

    def __init__(self, placeholder: str = "", *, margin: int = 20,
                 framed: bool = True, parent=None):
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignCenter)
        if framed:
            self.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self._margin = max(0, int(margin))
        self._source: QPixmap | None = None

    def set_array(self, array) -> bool:
        """设置要显示的数组；失败时把异常信息写进 label 并返回 False。"""
        try:
            self._source = ndarray_to_qpixmap(array)
        except Exception as exc:  # noqa: BLE001 - 图像种类太多，直接反馈给用户
            self._source = None
            self.setText(f"显示错误: {exc}")
            return False
        self._rescale()
        return True

    def clear_array(self) -> None:
        self._source = None
        self.clear()
        self.setText(self.placeholderText())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source is None:
            return
        target = self.size() - QSize(2 * self._margin, 2 * self._margin)
        if target.width() <= 0 or target.height() <= 0:
            return
        self.setPixmap(self._source.scaled(
            target, Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ---------------------------------------------------------------- 文件对话框
def ask_open_image(parent, caption: str = "选择图像", directory: str = "") -> str:
    """选择一张图片，取消时返回空字符串。"""
    path, _ = QFileDialog.getOpenFileName(parent, caption, directory, IMAGE_FILTERS)
    return path


def ask_save_path(parent, caption: str = "保存图像", *, directory: str = "",
                  suffix: str = ".png", filters: str = PNG_FILTERS) -> str:
    """选择保存路径；用户没写后缀时自动补上，取消时返回空字符串。"""
    path, _ = QFileDialog.getSaveFileName(parent, caption, directory, filters)
    if not path:
        return ""
    if suffix and not path.lower().endswith(suffix.lower()):
        path += suffix
    return path


def save_ndarray(array, path: str) -> None:
    """用 Pillow 保存数组；后缀决定格式。"""
    from PIL import Image

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    Image.fromarray(np.asarray(array)).save(path)
