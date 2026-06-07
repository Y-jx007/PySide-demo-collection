import sys
import random
import math
import numpy as np
from numba import jit
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, 
                               QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
                               QFrame, QColorDialog, QTextEdit, QFileDialog)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QPixmap, QImage

# 定义一些经典的IFS分形
IFS_FRACTALS = {
    "蕨类植物": {
        "transforms": [
            (0.0, 0.0, 0.0, 0.16, 0.0, 0.0),
            (0.85, 0.04, -0.04, 0.85, 0.0, 1.6),
            (0.2, -0.26, 0.23, 0.22, 0.0, 1.6),
            (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44)
        ],
        "probabilities": [0.01, 0.85, 0.07, 0.07],
        "iterations": 50000
    },
    "谢尔宾斯基三角形": {
        "transforms": [
            (0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
            (0.5, 0.0, 0.0, 0.5, 0.5, 0.0),
            (0.5, 0.0, 0.0, 0.5, 0.25, 0.5)
        ],
        "probabilities": [0.333, 0.333, 0.333],
        "iterations": 50000
    },
    "龙形曲线": {
        "transforms": [
            (0.5, -0.5, 0.5, 0.5, 0.0, 0.0),
            (-0.5, -0.5, 0.5, -0.5, 1.0, 0.0)
        ],
        "probabilities": [0.5, 0.5],
        "iterations": 50000
    },
    "分形树": {
        "transforms": [
            (0.0, 0.0, 0.0, 0.5, 0.0, 0.0),
            (0.42, -0.42, 0.42, 0.42, 0.0, 0.2),
            (0.42, 0.42, -0.42, 0.42, 0.0, 0.2),
            (0.1, 0.0, 0.0, 0.1, 0.0, 0.2)
        ],
        "probabilities": [0.05, 0.4, 0.4, 0.15],
        "iterations": 50000
    }
}

# 使用Numba优化的IFS迭代函数
@jit(nopython=True)
def generate_ifs_points(transforms, probabilities, iterations):
    """
    生成IFS分形点
    
    参数:
    transforms: 变换矩阵列表，每个变换是(a, b, c, d, e, f)
    probabilities: 每个变换的概率
    iterations: 迭代次数
    """
    # 初始化点数组
    points = np.zeros((iterations, 2), dtype=np.float64)
    
    # 初始点
    x, y = 0.0, 0.0
    
    # 累积概率，用于随机选择变换
    cum_probs = np.zeros(len(probabilities) + 1, dtype=np.float64)
    for i in range(len(probabilities)):
        cum_probs[i+1] = cum_probs[i] + probabilities[i]
    
    # 生成点
    for i in range(iterations):
        # 随机选择变换
        r = random.random()
        idx = 0
        for j in range(len(probabilities)):
            if r < cum_probs[j+1]:
                idx = j
                break
        
        # 应用变换
        a, b, c, d, e, f = transforms[idx]
        x_new = a * x + b * y + e
        y_new = c * x + d * y + f
        x, y = x_new, y_new
        
        points[i, 0] = x
        points[i, 1] = y
    
    return points

class FractalWorker(QThread):
    finished = Signal(np.ndarray)
    
    def __init__(self, transforms, probabilities, iterations):
        super().__init__()
        self.transforms = transforms
        self.probabilities = probabilities
        self.iterations = iterations
    
    def run(self):
        # 转换为numpy数组以便Numba处理
        transforms_np = np.array(self.transforms, dtype=np.float64)
        probabilities_np = np.array(self.probabilities, dtype=np.float64)
        
        # 使用Numba优化的函数生成点
        points = generate_ifs_points(
            transforms_np, 
            probabilities_np, 
            self.iterations
        )
        
        self.finished.emit(points)

class ColorButton(QWidget):
    colorChanged = Signal(QColor)  # 定义颜色更改信号
    
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(25, 25)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制正方形色块
        painter.fillRect(0, 0, self.width(), self.height(), self.color)
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(0, 0, self.width()-1, self.height()-1)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            color = QColorDialog.getColor(self.color, self)
            if color.isValid():
                self.color = color
                self.update()
                # 发射颜色更改信号
                self.colorChanged.emit(color)

class FractalWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.points = None
        # 蕨类孢子色（浅黄绿色）和蕨类叶子色（深绿色）
        self.color_start = QColor(218, 247, 166)  # 蕨类孢子色 - 浅黄绿色
        self.color_end = QColor(34, 139, 34)      # 蕨类叶子色 - 森林绿
        self.setMinimumSize(600, 600)
    
    def set_points(self, points):
        self.points = points
        self.update()
    
    def set_colors(self, color_start, color_end):
        self.color_start = color_start
        self.color_end = color_end
        if self.points is not None:
            self.update()
    
    def get_image(self):
        """获取当前分形的图像"""
        if self.points is None:
            return None
            
        # 创建一个与显示区域相同大小的图像
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.white)
        
        painter = QPainter(pixmap)
        self.draw_fractal(painter)
        painter.end()
        
        return pixmap
    
    def draw_fractal(self, painter):
        """绘制分形到指定的painter"""
        if self.points is None:
            return
        
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 计算点云的实际范围
        x_coords = self.points[:, 0]
        y_coords = self.points[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # 将点映射到窗口坐标
        width = self.width()
        height = self.height()
        
        # 计算缩放和偏移 - 确保分形居中
        scale_x = width / (x_max - x_min) if x_max != x_min else 1
        scale_y = height / (y_max - y_min) if y_max != y_min else 1
        scale = min(scale_x, scale_y) * 0.9  # 留一些边距
        
        # 计算居中偏移
        offset_x = (width - (x_max - x_min) * scale) / 2
        offset_y = (height - (y_max - y_min) * scale) / 2
        
        # 获取总点数
        total_points = len(self.points)
        
        # 绘制点 - 使用渐变色
        if total_points > 0:
            # 使用渐变色，根据迭代进度进行颜色插值
            for i in range(total_points):
                # 计算渐变因子（0到1之间）
                # 使用平方根函数使颜色变化更平滑
                t = math.sqrt(i / total_points) if total_points > 0 else 0
                
                # 线性插值计算颜色
                r = int(self.color_start.red() + (self.color_end.red() - self.color_start.red()) * t)
                g = int(self.color_start.green() + (self.color_end.green() - self.color_start.green()) * t)
                b = int(self.color_start.blue() + (self.color_end.blue() - self.color_start.blue()) * t)
                
                # 确保颜色值在有效范围内
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                
                color = QColor(r, g, b)
                painter.setPen(QPen(color, 1))
                
                x = self.points[i, 0]
                y = self.points[i, 1]
                # 映射到窗口坐标
                px = offset_x + (x - x_min) * scale
                py = height - (offset_y + (y - y_min) * scale)  # 反转Y轴
                painter.drawPoint(int(px), int(py))
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 填充背景
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        
        if self.points is None:
            # 绘制默认提示
            painter.setPen(QColor(180, 180, 180))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "选择分形类型并点击生成")
            return
        
        self.draw_fractal(painter)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IFS 分形生成器")
        self.setGeometry(100, 100, 800, 600)
        
        self.worker = None
        self.current_fractal = "蕨类植物"
        # 蕨类孢子色（浅黄绿色）和蕨类叶子色（深绿色）
        self.color_start = QColor(218, 247, 166)  # 蕨类孢子色 - 浅黄绿色
        self.color_end = QColor(34, 139, 34)      # 蕨类叶子色 - 森林绿
        
        # 设置朴素简约的窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                background-color: #f5f5f5;
                color: #333333;
            }
            QComboBox, QSpinBox, QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox:focus, QSpinBox:focus, QTextEdit:focus {
                border-color: #666666;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 6px 10px;
                min-width: 70px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
            QPushButton:disabled {
                background-color: #f0f0f0;
                color: #999999;
            }
            QLabel {
                color: #333333;
            }
        """)
        
        self.init_ui()
        # 初始化显示蕨类植物参数并生成分形
        self.on_fractal_change("蕨类植物")
        self.generate_fractal()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局 - 水平分割
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 右侧显示区域
        display_area = self.create_display_area()
        main_layout.addWidget(display_area)
        
        # 设置比例
        main_layout.setStretchFactor(control_panel, 1)
        main_layout.setStretchFactor(display_area, 3)
    
    def create_control_panel(self):
        # 创建左侧控制面板
        panel = QWidget()
        panel.setFixedWidth(320)
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标题
        title = QLabel("IFS 分形生成器")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # 分形选择
        fractal_section = self.create_section("分形选择")
        layout.addWidget(fractal_section)
        
        # 参数设置
        params_section = self.create_section("参数设置")
        layout.addWidget(params_section)
        
        # 分形参数面板
        custom_section = self.create_custom_section()
        layout.addWidget(custom_section)
        
        # 控制按钮
        control_section = self.create_control_section()
        layout.addWidget(control_section)
        
        # 添加弹性空间
        layout.addStretch()
        
        return panel
    
    def create_section(self, title):
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 分区标题
        section_title = QLabel(title)
        section_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(section_title)
        
        # 分区内容
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(8)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        if title == "分形选择":
            # 分形选择
            self.fractal_combo = QComboBox()
            self.fractal_combo.addItems(IFS_FRACTALS.keys())
            self.fractal_combo.currentTextChanged.connect(self.on_fractal_change)
            content_layout.addWidget(self.fractal_combo)
            
        elif title == "参数设置":
            # 迭代次数 - 紧凑布局
            iterations_layout = QHBoxLayout()
            iterations_layout.setSpacing(5)
            iterations_label = QLabel("迭代次数:")
            iterations_label.setFixedWidth(70)
            iterations_layout.addWidget(iterations_label)
            self.iterations_spin = QSpinBox()
            self.iterations_spin.setRange(1000, 1000000)
            self.iterations_spin.setValue(50000)
            # 移除上下按钮
            self.iterations_spin.setButtonSymbols(QSpinBox.NoButtons)
            iterations_layout.addWidget(self.iterations_spin)
            iterations_layout.addStretch()
            content_layout.addLayout(iterations_layout)
            
            # 颜色设置标题
            color_title = QLabel("颜色渐变设置:")
            color_title.setStyleSheet("margin-top: 8px;")
            content_layout.addWidget(color_title)
            
            # 颜色选择 - 在同一行显示起始和结束颜色
            colors_layout = QHBoxLayout()
            colors_layout.setSpacing(5)
            
            # 起始颜色
            start_color_label = QLabel("起始颜色:")
            colors_layout.addWidget(start_color_label)
            self.color_start_button = ColorButton(self.color_start)
            # 连接颜色更改信号
            self.color_start_button.colorChanged.connect(self.on_start_color_changed)
            colors_layout.addWidget(self.color_start_button)
            
            # 添加一些间距
            colors_layout.addSpacing(10)
            
            # 结束颜色
            end_color_label = QLabel("结束颜色:")
            colors_layout.addWidget(end_color_label)
            self.color_end_button = ColorButton(self.color_end)
            # 连接颜色更改信号
            self.color_end_button.colorChanged.connect(self.on_end_color_changed)
            colors_layout.addWidget(self.color_end_button)
            
            colors_layout.addStretch()
            content_layout.addLayout(colors_layout)
            
            # 颜色说明
            color_info = QLabel("提示: 分形点将按迭代顺序从起始颜色渐变到结束颜色")
            color_info.setWordWrap(True)
            color_info.setStyleSheet("font-size: 11px; color: #666666; margin-top: 5px;")
            content_layout.addWidget(color_info)
        
        layout.addWidget(content)
        
        return section
    
    def create_custom_section(self):
        # 创建分形参数面板
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标题
        custom_title = QLabel("分形参数")
        custom_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(custom_title)
        
        # 说明文本
        info_label = QLabel("IFS变换参数（每行一个变换，格式: a,b,c,d,e,f,概率）:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 文本编辑区域
        self.custom_text = QTextEdit()
        self.custom_text.setMaximumHeight(120)
        layout.addWidget(self.custom_text)
        
        return section
    
    def create_control_section(self):
        # 创建控制按钮区域
        section = QWidget()
        layout = QHBoxLayout(section)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 生成按钮
        self.generate_btn = QPushButton("生成")
        self.generate_btn.clicked.connect(self.generate_fractal)
        layout.addWidget(self.generate_btn)
        
        # 保存按钮
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_fractal)
        layout.addWidget(self.save_btn)
        
        return section
    
    def create_display_area(self):
        # 创建右侧显示区域
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 显示区域标题
        display_title = QLabel("分形预览")
        display_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(display_title)
        
        # 分形显示区域
        self.fractal_widget = FractalWidget()
        layout.addWidget(self.fractal_widget)
        
        return area
    
    def on_fractal_change(self, name):
        self.current_fractal = name
        
        # 更新迭代次数默认值
        if name in IFS_FRACTALS:
            self.iterations_spin.setValue(IFS_FRACTALS[name]["iterations"])
            
            # 在自定义输入框中显示变换参数
            fractal_data = IFS_FRACTALS[name]
            text = ""
            for i, transform in enumerate(fractal_data["transforms"]):
                a, b, c, d, e, f = transform
                prob = fractal_data["probabilities"][i]
                # 格式化数字，避免长串小数
                a_str = f"{a:.3f}".rstrip('0').rstrip('.') if a != int(a) else str(int(a))
                b_str = f"{b:.3f}".rstrip('0').rstrip('.') if b != int(b) else str(int(b))
                c_str = f"{c:.3f}".rstrip('0').rstrip('.') if c != int(c) else str(int(c))
                d_str = f"{d:.3f}".rstrip('0').rstrip('.') if d != int(d) else str(int(d))
                e_str = f"{e:.3f}".rstrip('0').rstrip('.') if e != int(e) else str(int(e))
                f_str = f"{f:.3f}".rstrip('0').rstrip('.') if f != int(f) else str(int(f))
                prob_str = f"{prob:.3f}".rstrip('0').rstrip('.') if prob != int(prob) else str(int(prob))
                
                text += f"{a_str},{b_str},{c_str},{d_str},{e_str},{f_str},{prob_str}\n"
            self.custom_text.setPlainText(text.strip())
    
    def on_start_color_changed(self, color):
        """处理起始颜色更改"""
        self.color_start = color
        # 立即更新分形颜色
        self.fractal_widget.set_colors(self.color_start, self.color_end)
    
    def on_end_color_changed(self, color):
        """处理结束颜色更改"""
        self.color_end = color
        # 立即更新分形颜色
        self.fractal_widget.set_colors(self.color_start, self.color_end)
    
    def generate_fractal(self):
        if self.worker and self.worker.isRunning():
            return
        
        # 解析输入框中的参数
        text = self.custom_text.toPlainText()
        lines = text.strip().split('\n')
        
        transforms = []
        probabilities = []
        
        for line in lines:
            if not line.strip():
                continue
                
            parts = line.split(',')
            if len(parts) != 7:
                # 错误处理
                continue
                
            try:
                a, b, c, d, e, f, prob = map(float, parts)
                transforms.append((a, b, c, d, e, f))
                probabilities.append(prob)
            except ValueError:
                # 错误处理
                continue
        
        if transforms and probabilities:
            # 创建并启动工作线程
            self.worker = FractalWorker(
                transforms,
                probabilities,
                self.iterations_spin.value()
            )
            
            self.worker.finished.connect(self.on_fractal_generated)
            self.generate_btn.setEnabled(False)
            self.worker.start()
    
    def on_fractal_generated(self, points):
        # 使用当前颜色设置点
        self.fractal_widget.set_points(points)
        self.fractal_widget.set_colors(self.color_start, self.color_end)
        self.generate_btn.setEnabled(True)
    
    def save_fractal(self):
        if self.fractal_widget.points is None:
            return
            
        # 获取保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存分形图像", "", "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)"
        )
        
        if file_path:
            # 获取分形图像并保存
            image = self.fractal_widget.get_image()
            if image:
                image.save(file_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序字体
    font = QFont("Arial", 9)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())