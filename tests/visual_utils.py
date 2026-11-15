# -*- coding: utf-8 -*-
"""Shared helpers for deterministic canvas captures."""
import numpy as np
from PySide6.QtGui import QImage


def capture_canvas(window, app):
    app.processEvents()
    return window.canvas.grab()


def image_array(pixmap):
    image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
    width, height = image.width(), image.height()
    line = image.bytesPerLine()
    raw = np.frombuffer(image.bits(), dtype=np.uint8).reshape(height, line)
    return raw[:, : width * 3].reshape(height, width, 3).copy()
