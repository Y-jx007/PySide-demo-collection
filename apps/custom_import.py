# ── 标准库 ──
import os
import sys
import math
import re
import time
import logging
import io
import hashlib
import ctypes
from abc import ABC, abstractmethod

# ── 数值计算 ──
import numpy as np
import taichi_forge as ti
import taichi_forge.math as tm
from mpmath import mp
import matplotlib.pyplot as plt

# ── 图像处理 ──
from PIL import Image, ImageDraw

# ── JIT 编译 ──
from numba import jit, njit, prange

# ── PySide6 ──
from PySide6.QtCore import (
    Qt, QTimer, QRect, QSize, QPoint, QThread, Signal, QObject
)
from PySide6.QtGui import (
    QFont, QPainter, QPixmap, QImage, QColor, QPen, QBrush,
    QMouseEvent, QSurfaceFormat, QVector3D, QMatrix4x4, QTransform,
    QFontDatabase, QAction
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFormLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QMessageBox,
    QLayout, QLayoutItem, QGroupBox, QComboBox, QLineEdit,
    QSpinBox, QDoubleSpinBox, QSlider, QCheckBox, QRadioButton,
    QSplitter, QTabWidget, QPlainTextEdit, QTextEdit, QProgressBar,
    QFileDialog, QColorDialog, QInputDialog, QButtonGroup,
    QListWidget, QListWidgetItem, QStackedWidget, QSizePolicy
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

# ── OpenGL ──
from OpenGL.GL import *
from OpenGL.GLUT import *

# ── 项目内部模块（在各 .py 文件中按需导入）──
# 注意：这些模块不再从 custom_import 导入，消除了循环依赖
# 应用文件通过 from custom_import import * 获得所有常用符号
# 下面显式导入项目中各功能模块的符号，供所有 app 文件使用
from utils import (
    OrbitCamera, ColorButton
)
from custom_function import (
    vectorize_func, make_safe_expression, integrate_custom_python,
    c64, csqr, cconj, cmul, cdiv, csin, ccos, cexp, clog, cpow, pack_color
)
from reaction_diffusion import (
    setup_gl_format, TextureGLWidget, SimulationBase, SimulationViewer
)
from dynamic_fractal import BaseFractalWidget

