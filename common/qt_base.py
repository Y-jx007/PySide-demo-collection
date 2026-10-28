# -*- coding: utf-8 -*-
"""Qt + standard-library base for all apps."""
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

from PySide6.QtCore import (
    Qt, QTimer, QRect, QRectF, QSize, QPoint, QPointF, QLineF, QSettings, QThread, Signal, QObject
)
from PySide6.QtGui import (
    QFont, QPainter, QPixmap, QImage, QColor, QPen, QBrush, QPolygonF,
    QMouseEvent, QSurfaceFormat, QVector3D, QMatrix4x4, QQuaternion, QTransform,
    QFontDatabase, QAction
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFormLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QMessageBox,
    QLayout, QLayoutItem, QGroupBox, QComboBox, QLineEdit,
    QSpinBox, QDoubleSpinBox, QSlider, QCheckBox, QRadioButton,
    QSplitter, QTabWidget, QPlainTextEdit, QTextEdit, QProgressBar,
    QFileDialog, QColorDialog, QInputDialog, QDialog, QButtonGroup,
    QListWidget, QListWidgetItem, QStackedWidget, QSizePolicy, QStyle, QStyleOptionSlider
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from common.utils import OrbitCamera, ColorButton, FPSCounter

# ---- 公共 UI 积木 / 启动流程 / 后台线程 ----
from common.widgets import (
    LabeledSlider,
    SquareContainer,
    caption,
    double_spinbox,
    form_group,
    group_box,
    hbox,
    separator,
    spinbox,
    vbox,
)
from common.workers import StoppableThread
from common.app import (
    apply_stylesheet,
    create_application,
    load_stylesheet,
    run_app,
    setup_gl_format,
)
