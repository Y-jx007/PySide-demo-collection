# -*- coding: utf-8 -*-
"""NumPy + Pillow base for all apps."""
import numpy as np
from PIL import Image, ImageDraw

# numpy 图像 <-> Qt 图像，以及统一的打开/保存对话框
from common.imaging import (
    ImageLabel,
    ask_open_image,
    ask_save_path,
    ndarray_to_qimage,
    ndarray_to_qpixmap,
    save_ndarray,
)
