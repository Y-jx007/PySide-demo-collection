import sys
import math
import numpy as np
from PIL import Image, ImageDraw

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QCheckBox, QSlider, QLineEdit,
    QFrame, QMessageBox, QSplitter, QGroupBox, QSizePolicy,
    QTabWidget, QFileDialog, QButtonGroup, QGridLayout
)
from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QImage, QPixmap, QMouseEvent, QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class DrawingCache:
    """绘图缓存类，优化图像更新性能"""
    def __init__(self, canvas_size):
        self.canvas_size = canvas_size
        self._image = None
        self._qimage_cache = None
        self._dirty = True
        
    @property
    def image(self):
        return self._image
    
    @image.setter
    def image(self, value):
        self._image = value
        self._dirty = True
        
    def get_qimage(self):
        """获取缓存的QImage，如果图像已更新则重新生成缓存"""
        if self._dirty or self._qimage_cache is None:
            if self._image is not None:
                img_data = self._image.tobytes("raw", "L")
                self._qimage_cache = QImage(img_data, self.canvas_size, 
                                          self.canvas_size, QImage.Format_Grayscale8)
            else:
                self._qimage_cache = QImage(self.canvas_size, self.canvas_size, 
                                          QImage.Format_Grayscale8)
                self._qimage_cache.fill(0)
            self._dirty = False
        return self._qimage_cache


class SimpleDrawingCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.canvas_size = 400  # 缩小画布尺寸
        self.setFixedSize(self.canvas_size, self.canvas_size)
        
        # 初始化图像和缓存
        self.cache = DrawingCache(self.canvas_size)
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.cache.image = self.image
        
        # 绘图相关变量
        self.last_point = None
        self.drawing = False
        self.mode = "draw"
        self.shape_mode = None
        self.shape_start = None
        self.temp_shape_points = []
        
        # 预计算常用值
        self._brush_size = 5
        self._fill_shape = False
        
        # 设置样式
        self.setStyleSheet("""
            SimpleDrawingCanvas {
                background-color: white;
                border: 1px solid #cccccc;
            }
        """)
    
    def update_brush_settings(self):
        """更新画笔设置，减少重复计算"""
        if self.parent_app:
            self._brush_size = self.parent_app.brush_size.value()
            self._fill_shape = self.parent_app.fill_shape.isChecked()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景和图像
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        painter.drawPixmap(0, 0, QPixmap.fromImage(self.cache.get_qimage()))
        
        # 绘制临时形状预览
        self._draw_temp_shape(painter)
    
    def _draw_temp_shape(self, painter):
        """绘制临时形状预览"""
        if not (self.shape_mode and self.shape_start and self.temp_shape_points):
            return
            
        self.update_brush_settings()
        pen = QPen(QColor(255, 255, 255), self._brush_size)
        painter.setPen(pen)
        
        if self._fill_shape:
            brush = QBrush(QColor(255, 255, 255, 100))
            painter.setBrush(brush)
        else:
            painter.setBrush(Qt.NoBrush)
            
        if len(self.temp_shape_points) > 1:
            if self.shape_mode == "直线":
                painter.setRenderHint(QPainter.Antialiasing, False)
                painter.drawLine(self.shape_start, self.temp_shape_points[-1])
            elif self.shape_mode == "矩形":
                rect = self._normalize_rect(self.shape_start, self.temp_shape_points[-1])
                painter.drawRect(rect)
            elif self.shape_mode == "圆形":
                center = self.shape_start
                radius = self._calculate_radius(center, self.temp_shape_points[-1])
                painter.drawEllipse(center, radius, radius)
            elif self.shape_mode in ["三角形", "五边形", "六边形", "五角星", "六角星"]:
                painter.drawPolygon(self.temp_shape_points)
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            self.drawing = True
            self.last_point = pos
            
            if self.mode == "shape" and self.shape_mode:
                self.shape_start = pos
                self.temp_shape_points = [pos]
            else:
                self._draw_point(pos)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing and event.buttons() & Qt.LeftButton:
            pos = event.pos()
            
            if self.mode == "shape" and self.shape_mode and self.shape_start:
                self._update_temp_shape(pos)
            else:
                self._draw_line(pos)
            
            self.last_point = pos
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self.mode == "shape" and self.shape_mode and self.shape_start:
                self._finalize_shape()
            
            self.drawing = False
            self.last_point = None
            self.parent_app.calculate_fourier()
    
    def _draw_point(self, pos):
        """绘制单个点"""
        self.update_brush_settings()
        x, y = pos.x(), pos.y()
        color = 255 if self.mode == "draw" else 0
        
        # 预计算坐标
        half_size = self._brush_size
        x1, y1 = x - half_size, y - half_size
        x2, y2 = x + half_size, y + half_size
        
        self.draw.ellipse([x1, y1, x2, y2], fill=color)
        self.cache.image = self.image  # 标记缓存为脏
        self.update()
    
    def _draw_line(self, pos):
        """绘制直线"""
        if self.last_point is None:
            return
            
        self.update_brush_settings()
        start_x, start_y = self.last_point.x(), self.last_point.y()
        end_x, end_y = pos.x(), pos.y()
        
        color = 255 if self.mode == "draw" else 0
        self._draw_bresenham_line(start_x, start_y, end_x, end_y, self._brush_size, color)
        
        self.cache.image = self.image  # 标记缓存为脏
        self.update()
    
    def _draw_bresenham_line(self, x0, y0, x1, y1, brush_size, color):
        """使用优化的Bresenham算法绘制直线"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        half_size = brush_size
        
        while True:
            # 绘制点
            x1_pt, y1_pt = x0 - half_size, y0 - half_size
            x2_pt, y2_pt = x0 + half_size, y0 + half_size
            self.draw.ellipse([x1_pt, y1_pt, x2_pt, y2_pt], fill=color)
            
            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    def _update_temp_shape(self, pos):
        """更新临时形状"""
        if not self.shape_start:
            return
            
        self.temp_shape_points = [self.shape_start, pos]
        
        if self.shape_mode in ["五边形", "六边形", "五角星", "六角星"]:
            center_x = (self.shape_start.x() + pos.x()) / 2
            center_y = (self.shape_start.y() + pos.y()) / 2
            radius = min(abs(pos.x() - self.shape_start.x()), 
                        abs(pos.y() - self.shape_start.y())) / 2
            
            if self.shape_mode == "五边形":
                self.temp_shape_points = self._calculate_regular_polygon(center_x, center_y, radius, 5)
            elif self.shape_mode == "六边形":
                self.temp_shape_points = self._calculate_regular_polygon(center_x, center_y, radius, 6)
            elif self.shape_mode == "五角星":
                self.temp_shape_points = self._calculate_star(center_x, center_y, radius, 5)
            elif self.shape_mode == "六角星":
                self.temp_shape_points = self._calculate_star(center_x, center_y, radius, 6)
        elif self.shape_mode == "三角形":
            x1, y1 = self.shape_start.x(), self.shape_start.y()
            x2, y2 = pos.x(), pos.y()
            x1, y1, x2, y2 = self._normalize_coords(x1, y1, x2, y2)
            self.temp_shape_points = [
                QPoint(x1, y2),
                QPoint(x2, y2),
                QPoint((x1 + x2) // 2, y1)
            ]
    
    def _finalize_shape(self):
        """完成形状绘制"""
        if not self.shape_start or not self.temp_shape_points:
            return
            
        self.update_brush_settings()
        
        if self.shape_mode == "直线":
            start, end = self.shape_start, self.temp_shape_points[-1]
            self.draw.line([start.x(), start.y(), end.x(), end.y()], 
                          fill=255, width=self._brush_size)
        elif self.shape_mode == "矩形":
            rect = self._normalize_rect(self.shape_start, self.temp_shape_points[-1])
            if self._fill_shape:
                self.draw.rectangle([rect.left(), rect.top(), rect.right(), rect.bottom()], 
                                   fill=255)
            else:
                self.draw.rectangle([rect.left(), rect.top(), rect.right(), rect.bottom()], 
                                   outline=255, width=self._brush_size)
        elif self.shape_mode == "圆形":
            center = self.shape_start
            radius = self._calculate_radius(center, self.temp_shape_points[-1])
            bbox = [center.x()-radius, center.y()-radius, center.x()+radius, center.y()+radius]
            if self._fill_shape:
                self.draw.ellipse(bbox, fill=255)
            else:
                self.draw.ellipse(bbox, outline=255, width=self._brush_size)
        elif self.shape_mode in ["三角形", "五边形", "六边形", "五角星", "六角星"]:
            points = [(p.x(), p.y()) for p in self.temp_shape_points]
            if self._fill_shape:
                self.draw.polygon(points, fill=255)
            else:
                self.draw.polygon(points, outline=255, width=self._brush_size)
        
        self.shape_start = None
        self.temp_shape_points = []
        self.cache.image = self.image  # 标记缓存为脏
        self.update()
    
    def _normalize_rect(self, p1, p2):
        return QRect(
            min(p1.x(), p2.x()), min(p1.y(), p2.y()),
            abs(p1.x() - p2.x()), abs(p1.y() - p2.y())
        )
    
    def _normalize_coords(self, x1, y1, x2, y2):
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    
    def _calculate_radius(self, center, point):
        dx = point.x() - center.x()
        dy = point.y() - center.y()
        return max(abs(dx), abs(dy))
    
    def _calculate_regular_polygon(self, center_x, center_y, radius, sides):
        """计算正多边形顶点"""
        points = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append(QPoint(int(x), int(y)))
        return points
    
    def _calculate_star(self, center_x, center_y, radius, points_count):
        """计算星形顶点"""
        star_points = []
        outer_radius = radius
        inner_radius = radius * 0.4
        
        for i in range(points_count * 2):
            angle = math.pi * i / points_count - math.pi / 2
            r = outer_radius if i % 2 == 0 else inner_radius
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            star_points.append(QPoint(int(x), int(y)))
        
        return star_points
    
    def reset_canvas(self):
        """重置画布"""
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.cache.image = self.image
        self.shape_start = None
        self.temp_shape_points = []
        self.update()
    
    def get_image_array(self):
        return np.array(self.image)
    
    def set_image_from_array(self, array):
        self.image = Image.fromarray(array)
        self.draw = ImageDraw.Draw(self.image)
        self.cache.image = self.image
        self.update()
    
    def save_image(self, file_path):
        self.image.save(file_path)


class FourierCalculator:
    """傅里叶变换计算器，分离计算逻辑"""
    @staticmethod
    def calculate_fourier_transform(image_array):
        """计算傅里叶变换"""
        if np.all(image_array == 0):
            return None
        
        f = np.fft.fft2(image_array)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        return magnitude_spectrum


class FunctionEvaluator:
    """函数求值器，优化数学表达式计算"""
    
    SAFE_DICT = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "abs": np.abs, "pi": np.pi, "atan2": np.arctan2
    }
    
    @classmethod
    def preprocess_expression(cls, expr):
        """预处理表达式"""
        return expr.replace('^', '**').replace(' and ', ' & ').replace(' or ', ' | ')
    
    @classmethod
    def evaluate_expression(cls, expr, x, y):
        """安全地求值表达式"""
        safe_dict = {"x": x, "y": y, **cls.SAFE_DICT}
        return eval(expr, {"__builtins__": None}, safe_dict)


class FourierDrawApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.fourier_calculator = FourierCalculator()
        self.function_evaluator = FunctionEvaluator()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("傅里叶变换绘图工具")
        self.setGeometry(100, 100, 800, 600)  # 缩小窗口尺寸
        
        # 设置应用程序样式
        self.set_app_style()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # 创建各个UI部分
        self.create_tool_panel(main_layout)
        self.create_drawing_area(main_layout)
        self.create_function_panel(main_layout)
    
    def set_app_style(self):
        """设置应用程序样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
                color: #333333;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                margin-top: 1ex;
                padding-top: 8px;
                background-color: white;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
                color: #495057;
            }
            QLabel {
                color: #495057;
                font-weight: 500;
                font-size: 11px;
            }
            QLineEdit, QComboBox {
                background-color: white;
                color: #495057;
                border: 1px solid #ced4da;
                border-radius: 3px;
                padding: 4px 6px;
                font-size: 11px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #80bdff;
                outline: none;
            }
            QPushButton {
                background-color: #ffffff;
                color: #495057;
                border: 1px solid #ced4da;
                padding: 6px 12px;
                border-radius: 0px;
                font-weight: 500;
                font-size: 11px;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
            QPushButton:pressed {
                background-color: #dee2e6;
            }
            QPushButton:checked {
                background-color: #6c757d;
                color: white;
                border-color: #6c757d;
            }
            QCheckBox {
                color: #495057;
                spacing: 4px;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #ced4da;
                border-radius: 2px;
                background: white;
            }
            QCheckBox::indicator:checked {
                background: #6c757d;
                border: 1px solid #6c757d;
            }
            QSlider::groove:horizontal {
                border: 1px solid #ced4da;
                height: 3px;
                background: #e9ecef;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #6c757d;
                border: 1px solid #495057;
                width: 14px;
                height: 14px;
                border-radius: 7px;
                margin: -5px 0;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f8f9fa;
                color: #6c757d;
                padding: 6px 12px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                font-weight: 500;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #495057;
                border-bottom: 2px solid #6c757d;
            }
            QTabBar::tab:hover {
                background-color: #e9ecef;
            }
        """)
    
    def create_tool_panel(self, parent_layout):
        """创建工具面板"""
        tool_group = QGroupBox("绘图工具")
        tool_layout = QHBoxLayout(tool_group)
        tool_layout.setSpacing(6)
        tool_layout.setContentsMargins(8, 16, 8, 8)
        
        # 基本操作
        basic_tools_layout = QVBoxLayout()
        basic_tools_layout.setSpacing(4)
        
        basic_label = QLabel("基本操作")
        basic_label.setStyleSheet("color: #6c757d; font-weight: bold; font-size: 10px;")
        basic_tools_layout.addWidget(basic_label)
        
        basic_buttons_layout = QHBoxLayout()
        self.clear_btn = QPushButton("清空画布")
        self.clear_btn.clicked.connect(self.reset_canvas)
        basic_buttons_layout.addWidget(self.clear_btn)
        
        self.invert_btn = QPushButton("反转灰度")
        self.invert_btn.clicked.connect(self.invert_grayscale)
        basic_buttons_layout.addWidget(self.invert_btn)
        
        basic_tools_layout.addLayout(basic_buttons_layout)
        tool_layout.addLayout(basic_tools_layout)
        
        # 分隔线
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.VLine)
        separator1.setFrameShadow(QFrame.Sunken)
        separator1.setStyleSheet("color: #dee2e6;")
        tool_layout.addWidget(separator1)
        
        # 绘图模式
        mode_layout = QVBoxLayout()
        mode_layout.setSpacing(4)
        
        mode_label = QLabel("绘图模式")
        mode_label.setStyleSheet("color: #6c757d; font-weight: bold; font-size: 10px;")
        mode_layout.addWidget(mode_label)
        
        mode_buttons_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        
        self.draw_btn = QPushButton("画笔")
        self.draw_btn.setCheckable(True)
        self.draw_btn.setChecked(True)
        self.draw_btn.clicked.connect(lambda: self.set_mode("draw"))
        mode_buttons_layout.addWidget(self.draw_btn)
        self.mode_group.addButton(self.draw_btn)
        
        self.erase_btn = QPushButton("橡皮擦")
        self.erase_btn.setCheckable(True)
        self.erase_btn.clicked.connect(lambda: self.set_mode("erase"))
        mode_buttons_layout.addWidget(self.erase_btn)
        self.mode_group.addButton(self.erase_btn)
        
        mode_layout.addLayout(mode_buttons_layout)
        tool_layout.addLayout(mode_layout)
        
        # 分隔线
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setFrameShadow(QFrame.Sunken)
        separator2.setStyleSheet("color: #dee2e6;")
        tool_layout.addWidget(separator2)
        
        # 形状工具
        shape_layout = QVBoxLayout()
        shape_layout.setSpacing(4)
        
        shape_label = QLabel("形状工具")
        shape_label.setStyleSheet("color: #6c757d; font-weight: bold; font-size: 10px;")
        shape_layout.addWidget(shape_label)
        
        shape_controls_layout = QHBoxLayout()
        self.shape_combo = QComboBox()
        shapes = ["选择形状...", "直线", "矩形", "圆形", "三角形", "五边形", "六边形", "五角星", "六角星"]
        self.shape_combo.addItems(shapes)
        self.shape_combo.currentTextChanged.connect(self.activate_shape_mode)
        self.shape_combo.setMinimumWidth(90)
        self.shape_combo.setMaximumWidth(90)
        shape_controls_layout.addWidget(self.shape_combo)
        
        self.fill_shape = QCheckBox("填充")
        shape_controls_layout.addWidget(self.fill_shape)
        
        shape_layout.addLayout(shape_controls_layout)
        tool_layout.addLayout(shape_layout)
        
        # 分隔线
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.VLine)
        separator3.setFrameShadow(QFrame.Sunken)
        separator3.setStyleSheet("color: #dee2e6;")
        tool_layout.addWidget(separator3)
        
        # 画笔设置
        brush_layout = QVBoxLayout()
        brush_layout.setSpacing(4)
        
        brush_label = QLabel("画笔设置")
        brush_label.setStyleSheet("color: #6c757d; font-weight: bold; font-size: 10px;")
        brush_layout.addWidget(brush_label)
        
        brush_controls_layout = QHBoxLayout()
        brush_controls_layout.addWidget(QLabel("大小:"))
        self.brush_size = QSlider(Qt.Horizontal)
        self.brush_size.setMinimum(1)
        self.brush_size.setMaximum(20)
        self.brush_size.setValue(5)
        self.brush_size.setFixedWidth(70)
        brush_controls_layout.addWidget(self.brush_size)
        
        brush_size_value = QLabel("5")
        brush_size_value.setFixedWidth(15)
        brush_size_value.setStyleSheet("font-size: 10px;")
        self.brush_size.valueChanged.connect(lambda v: brush_size_value.setText(str(v)))
        brush_controls_layout.addWidget(brush_size_value)
        
        brush_layout.addLayout(brush_controls_layout)
        tool_layout.addLayout(brush_layout)
        
        # 分隔线
        separator4 = QFrame()
        separator4.setFrameShape(QFrame.VLine)
        separator4.setFrameShadow(QFrame.Sunken)
        separator4.setStyleSheet("color: #dee2e6;")
        tool_layout.addWidget(separator4)
        
        # 保存选项
        save_layout = QVBoxLayout()
        save_layout.setSpacing(4)
        
        save_label = QLabel("保存选项")
        save_label.setStyleSheet("color: #6c757d; font-weight: bold; font-size: 10px;")
        save_layout.addWidget(save_label)
        
        save_buttons_layout = QHBoxLayout()
        self.save_drawing_btn = QPushButton("保存绘图")
        self.save_drawing_btn.clicked.connect(lambda: self.save_image("drawing"))
        save_buttons_layout.addWidget(self.save_drawing_btn)
        
        self.save_spectrum_btn = QPushButton("保存频谱")
        self.save_spectrum_btn.clicked.connect(lambda: self.save_image("spectrum"))
        save_buttons_layout.addWidget(self.save_spectrum_btn)
        
        save_layout.addLayout(save_buttons_layout)
        tool_layout.addLayout(save_layout)
        
        parent_layout.addWidget(tool_group)
    
    def create_drawing_area(self, parent_layout):
        """创建绘图和显示区域"""
        # 创建水平分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(4)
        
        # 左侧绘图区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        
        drawing_group = QGroupBox("绘图区域")
        drawing_layout = QVBoxLayout(drawing_group)
        drawing_layout.setAlignment(Qt.AlignCenter)
        drawing_layout.setContentsMargins(4, 4, 4, 9)
        
        self.drawing_canvas = SimpleDrawingCanvas(self)
        drawing_layout.addWidget(self.drawing_canvas)
        
        left_layout.addWidget(drawing_group)
        splitter.addWidget(left_widget)
        
        # 右侧频谱显示区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        
        spectrum_group = QGroupBox("傅里叶变换频谱")
        spectrum_layout = QVBoxLayout(spectrum_group)
        spectrum_layout.setAlignment(Qt.AlignCenter)
        spectrum_layout.setContentsMargins(4, 4, 4, 9)
        
        # 创建matplotlib图形
        self.fig = plt.Figure(figsize=(3, 3), dpi=100)  # 缩小图形尺寸
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        
        self.canvas_fig = FigureCanvas(self.fig)
        self.canvas_fig.setFixedSize(400, 400)  # 缩小画布尺寸
        self.canvas_fig.setStyleSheet("background-color: white; border: 1px solid #dee2e6;")
        spectrum_layout.addWidget(self.canvas_fig)
        
        right_layout.addWidget(spectrum_group)
        splitter.addWidget(right_widget)
        
        # 设置分割比例
        splitter.setSizes([350, 350])
        parent_layout.addWidget(splitter, 1)  # 1表示可拉伸
    
    def create_function_panel(self, parent_layout):
        """创建函数控制面板"""
        function_group = QGroupBox("函数绘图")
        function_layout = QVBoxLayout(function_group)
        function_layout.setSpacing(8)
        function_layout.setContentsMargins(8, 16, 8, 8)
        
        # 坐标范围设置
        range_layout = QHBoxLayout()
        range_layout.setSpacing(6)
        
        range_label = QLabel("坐标范围:")
        range_label.setStyleSheet("font-weight: bold;")
        range_layout.addWidget(range_label)
        
        # X范围
        x_range_layout = QHBoxLayout()
        x_range_layout.setSpacing(2)
        x_range_layout.addWidget(QLabel("X:"))
        self.x_min_entry = QLineEdit("-200")
        self.x_min_entry.setFixedWidth(50)
        x_range_layout.addWidget(self.x_min_entry)
        x_range_layout.addWidget(QLabel("到"))
        self.x_max_entry = QLineEdit("200")
        self.x_max_entry.setFixedWidth(50)
        x_range_layout.addWidget(self.x_max_entry)
        range_layout.addLayout(x_range_layout)
        
        # Y范围
        y_range_layout = QHBoxLayout()
        y_range_layout.setSpacing(2)
        y_range_layout.addWidget(QLabel("Y:"))
        self.y_min_entry = QLineEdit("-200")
        self.y_min_entry.setFixedWidth(50)
        y_range_layout.addWidget(self.y_min_entry)
        y_range_layout.addWidget(QLabel("到"))
        self.y_max_entry = QLineEdit("200")
        self.y_max_entry.setFixedWidth(50)
        y_range_layout.addWidget(self.y_max_entry)
        range_layout.addLayout(y_range_layout)
        
        # 分辨率
        resolution_layout = QHBoxLayout()
        resolution_layout.setSpacing(2)
        resolution_layout.addWidget(QLabel("分辨率:"))
        self.resolution_entry = QLineEdit("0.5")
        self.resolution_entry.setFixedWidth(40)
        resolution_layout.addWidget(self.resolution_entry)
        range_layout.addLayout(resolution_layout)
        
        range_layout.addStretch(1)
        function_layout.addLayout(range_layout)
        
        # 函数输入区域
        tab_widget = QTabWidget()
        tab_widget.setMaximumHeight(120)  # 限制高度

        # 形状函数选项卡
        shape_tab = QWidget()
        shape_layout = QVBoxLayout(shape_tab)
        shape_layout.setSpacing(6)
        shape_layout.setContentsMargins(4, 4, 4, 4)

        # 函数输入
        shape_input_layout = QHBoxLayout()
        shape_input_layout.setSpacing(8)  # 减少间距
        shape_label = QLabel("形状函数:")
        shape_label.setMinimumWidth(60)  # 固定标签宽度
        shape_input_layout.addWidget(shape_label)

        self.function_entry = QLineEdit()
        self.function_entry.setPlaceholderText("例如: sqrt(x**2+y**2) <= 180*cos(3*atan2(y,x))")
        self.function_entry.returnPressed.connect(self.draw_function)
        shape_input_layout.addWidget(self.function_entry, 1)  # 添加拉伸因子让输入框占据剩余空间

        self.draw_function_btn = QPushButton("绘制形状")
        self.draw_function_btn.clicked.connect(self.draw_function)
        self.draw_function_btn.setMinimumWidth(80)  # 设置按钮正常宽度
        shape_input_layout.addWidget(self.draw_function_btn)

        shape_layout.addLayout(shape_input_layout)

        # 示例选择
        example_layout = QHBoxLayout()
        example_layout.setSpacing(8)  # 减少间距
        example_label = QLabel("示例函数:")
        example_label.setMinimumWidth(60)  # 固定标签宽度
        example_layout.addWidget(example_label)

        self.example_combo = QComboBox()
        examples = [
            "选择示例函数...",
            "sqrt(x**2+y**2) <= 180*cos(3*atan2(y,x))  (花瓣)"
        ]
        self.example_combo.addItems(examples)
        self.example_combo.currentTextChanged.connect(self.select_example)
        example_layout.addWidget(self.example_combo, 1)  # 添加拉伸因子让下拉框占据剩余空间

        shape_layout.addLayout(example_layout)

        # 灰度函数选项卡
        grayscale_tab = QWidget()
        grayscale_layout = QVBoxLayout(grayscale_tab)
        grayscale_layout.setSpacing(6)
        grayscale_layout.setContentsMargins(4, 4, 4, 4)

        # 灰度函数输入
        grayscale_input_layout = QHBoxLayout()
        grayscale_input_layout.setSpacing(8)  # 减少间距
        grayscale_label = QLabel("灰度函数:")
        grayscale_label.setMinimumWidth(60)  # 固定标签宽度
        grayscale_input_layout.addWidget(grayscale_label)

        self.grayscale_entry = QLineEdit()
        self.grayscale_entry.setPlaceholderText("例如: 255 * (1 - (x**2 + y**2)/400)")
        self.grayscale_entry.returnPressed.connect(self.apply_grayscale_function)
        grayscale_input_layout.addWidget(self.grayscale_entry, 1)  # 添加拉伸因子让输入框占据剩余空间

        self.apply_grayscale_btn = QPushButton("应用灰度")
        self.apply_grayscale_btn.clicked.connect(self.apply_grayscale_function)
        self.apply_grayscale_btn.setMinimumWidth(80)  # 设置按钮正常宽度
        grayscale_input_layout.addWidget(self.apply_grayscale_btn)

        grayscale_layout.addLayout(grayscale_input_layout)

        # 灰度示例选择
        grayscale_example_layout = QHBoxLayout()
        grayscale_example_layout.setSpacing(8)  # 减少间距
        grayscale_example_label = QLabel("示例函数:")
        grayscale_example_label.setMinimumWidth(60)  # 固定标签宽度
        grayscale_example_layout.addWidget(grayscale_example_label)

        self.grayscale_example_combo = QComboBox()
        grayscale_examples = [
            "选择示例函数...",
            "255 * (1 - (x**2 + y**2)/400)  (径向渐变)",
            "128 + 127 * sin(x/30)  (水平正弦波)",
            "128 + 127 * sin(y/30)  (垂直正弦波)",
            "128 + 127 * sin(sqrt(x**2+y**2)/20)  (径向正弦波)",
            "255 * (0.5 + 0.5*sin(x/25)*cos(y/25))  (二维正弦波)",
            "255 * abs(sin(5*atan2(y,x)))  (角度条纹)",
            "255 * (1 - (abs(x)+abs(y))/100)  (菱形渐变)"
        ]
        self.grayscale_example_combo.addItems(grayscale_examples)
        self.grayscale_example_combo.currentTextChanged.connect(self.select_grayscale_example)
        grayscale_example_layout.addWidget(self.grayscale_example_combo, 1)  # 添加拉伸因子让下拉框占据剩余空间

        grayscale_layout.addLayout(grayscale_example_layout)

        tab_widget.addTab(shape_tab, "形状函数")
        tab_widget.addTab(grayscale_tab, "灰度函数")

        function_layout.addWidget(tab_widget)
        parent_layout.addWidget(function_group)
    
    def set_mode(self, mode):
        self.drawing_canvas.mode = mode
        if mode == "draw":
            self.draw_btn.setChecked(True)
            self.erase_btn.setChecked(False)
        else:
            self.draw_btn.setChecked(False)
            self.erase_btn.setChecked(True)
        
        self.shape_combo.setCurrentText("选择形状...")
        self.drawing_canvas.shape_mode = None
    
    def activate_shape_mode(self, shape):
        if shape != "选择形状...":
            self.drawing_canvas.mode = "shape"
            self.drawing_canvas.shape_mode = shape
    
    def reset_canvas(self):
        self.drawing_canvas.reset_canvas()
        self.ax.clear()
        self.ax.axis('off')
        self.canvas_fig.draw()
    
    def calculate_fourier(self):
        img_array = self.drawing_canvas.get_image_array()
        magnitude_spectrum = self.fourier_calculator.calculate_fourier_transform(img_array)
        
        if magnitude_spectrum is None:
            self.ax.clear()
            self.ax.axis('off')
        else:
            self.ax.clear()
            self.ax.imshow(magnitude_spectrum, cmap='gray')
            self.ax.axis('off')
        
        self.canvas_fig.draw()
    
    def invert_grayscale(self):
        img_array = self.drawing_canvas.get_image_array()
        inverted_array = 255 - img_array
        self.drawing_canvas.set_image_from_array(inverted_array)
        self.calculate_fourier()
    
    def save_image(self, image_type):
        if image_type == "drawing":
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存绘图图像", "", "PNG图像 (*.png);;所有文件 (*)"
            )
            
            if file_path:
                if not file_path.lower().endswith('.png'):
                    file_path += '.png'
                
                try:
                    self.drawing_canvas.save_image(file_path)
                    QMessageBox.information(self, "保存成功", f"绘图图像已保存到:\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "保存失败", f"保存绘图图像时出错:\n{str(e)}")
        
        elif image_type == "spectrum":
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存频谱图像", "", "PNG图像 (*.png);;所有文件 (*)"
            )
            
            if file_path:
                if not file_path.lower().endswith('.png'):
                    file_path += '.png'
                
                try:
                    self.fig.savefig(file_path, bbox_inches='tight', pad_inches=0, dpi=100)
                    QMessageBox.information(self, "保存成功", f"频谱图像已保存到:\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "保存失败", f"保存频谱图像时出错:\n{str(e)}")
    
    def select_example(self, example):
        if example != "选择示例函数...":
            func_expr = example.split("  (")[0]
            self.function_entry.setText(func_expr)
    
    def select_grayscale_example(self, example):
        if example != "选择示例函数...":
            func_expr = example.split("  (")[0]
            self.grayscale_entry.setText(func_expr)
    
    def draw_function(self):
        func_expr = self.function_entry.text().strip()
        if not func_expr:
            QMessageBox.warning(self, "警告", "请输入函数表达式")
            return
        
        try:
            x_min, x_max, y_min, y_max, resolution = self._get_range_values()
        except ValueError:
            QMessageBox.critical(self, "错误", "请输入有效的数值范围")
            return
        
        self.reset_canvas()
        func_expr = self.function_evaluator.preprocess_expression(func_expr)
        
        try:
            x_values = np.arange(x_min, x_max, resolution)
            y_values = np.arange(y_min, y_max, resolution)
            
            X, Y = np.meshgrid(x_values, y_values)
            result = self.function_evaluator.evaluate_expression(func_expr, X, Y)
            
            true_points = np.where(result)
            temp_img = self._create_function_image(x_min, x_max, y_min, y_max, 
                                                 x_values, y_values, true_points)
            
            img_array = self.drawing_canvas.get_image_array()
            img_array = np.maximum(img_array, temp_img)
            self.drawing_canvas.set_image_from_array(img_array)
            self.calculate_fourier()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法解析函数表达式: {str(e)}")
    
    def apply_grayscale_function(self):
        func_expr = self.grayscale_entry.text().strip()
        if not func_expr:
            QMessageBox.warning(self, "警告", "请输入灰度函数表达式")
            return
        
        try:
            x_min, x_max, y_min, y_max, resolution = self._get_range_values()
        except ValueError:
            QMessageBox.critical(self, "错误", "请输入有效的数值范围")
            return
        
        self.reset_canvas()
        func_expr = self.function_evaluator.preprocess_expression(func_expr)
        
        try:
            x_values = np.arange(x_min, x_max, resolution)
            y_values = np.arange(y_min, y_max, resolution)
            
            X, Y = np.meshgrid(x_values, y_values)
            gray_values = self.function_evaluator.evaluate_expression(func_expr, X, Y)
            gray_values = np.clip(gray_values, 0, 255)
            
            temp_img = self._create_grayscale_image(x_min, x_max, y_min, y_max,
                                                  x_values, y_values, gray_values)
            
            self.drawing_canvas.set_image_from_array(temp_img)
            self.calculate_fourier()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法解析灰度函数表达式: {str(e)}")
    
    def _get_range_values(self):
        """获取范围值"""
        x_min = float(self.x_min_entry.text())
        x_max = float(self.x_max_entry.text())
        y_min = float(self.y_min_entry.text())
        y_max = float(self.y_max_entry.text())
        resolution = float(self.resolution_entry.text())
        return x_min, x_max, y_min, y_max, resolution
    
    def _create_function_image(self, x_min, x_max, y_min, y_max, x_values, y_values, true_points):
        """创建函数图像"""
        temp_img = np.zeros((self.drawing_canvas.canvas_size, self.drawing_canvas.canvas_size), dtype=np.uint8)
        
        for i in range(len(true_points[0])):
            x_idx = true_points[0][i]
            y_idx = true_points[1][i]
            
            x = x_values[y_idx]
            y = y_values[x_idx]
            
            cx = self._canvas_x(x, x_min, x_max)
            cy = self._canvas_y(y, y_min, y_max)
            
            if 0 <= cx < self.drawing_canvas.canvas_size and 0 <= cy < self.drawing_canvas.canvas_size:
                temp_img[cy, cx] = 255
        
        return temp_img
    
    def _create_grayscale_image(self, x_min, x_max, y_min, y_max, x_values, y_values, gray_values):
        """创建灰度图像"""
        temp_img = np.zeros((self.drawing_canvas.canvas_size, self.drawing_canvas.canvas_size), dtype=np.uint8)
        
        for i in range(len(x_values)):
            for j in range(len(y_values)):
                x = x_values[i]
                y = y_values[j]
                
                cx = self._canvas_x(x, x_min, x_max)
                cy = self._canvas_y(y, y_min, y_max)
                
                if 0 <= cx < self.drawing_canvas.canvas_size and 0 <= cy < self.drawing_canvas.canvas_size:
                    gray_value = int(gray_values[j, i])
                    temp_img[cy, cx] = gray_value
        
        return temp_img
    
    def _canvas_x(self, x, x_min, x_max):
        return int((x - x_min) / (x_max - x_min) * self.drawing_canvas.canvas_size)
    
    def _canvas_y(self, y, y_min, y_max):
        return int(self.drawing_canvas.canvas_size - (y - y_min) / (y_max - y_min) * self.drawing_canvas.canvas_size)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序字体
    font = QFont("Microsoft YaHei", 9)  # 缩小字体
    app.setFont(font)
    
    window = FourierDrawApp()
    window.show()
    sys.exit(app.exec_()) 