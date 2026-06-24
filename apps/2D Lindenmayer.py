import sys
import math
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
                               QTextEdit, QGroupBox, QComboBox, QColorDialog, QSplitter,
                               QFrame, QSizePolicy, QFileDialog, QMessageBox, QGridLayout)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QPen, QColor, QLinearGradient, QFont, QPixmap, QImage


class LSystem:
    """L-system 实现类"""
    def __init__(self):
        self.axiom = ""  # 初始字符串
        self.rules = {}  # 规则字典
        self.angle = 25  # 角度
        self.distance = 10  # 步长
        self.generations = 4  # 迭代次数
        
    def set_axiom(self, axiom):
        self.axiom = axiom
        
    def add_rule(self, predecessor, successor):
        self.rules[predecessor] = successor
        
    def generate(self, generations=None):
        if generations is None:
            generations = self.generations
            
        result = self.axiom
        for _ in range(generations):
            new_result = ""
            for char in result:
                if char in self.rules:
                    new_result += self.rules[char]
                else:
                    new_result += char
            result = new_result
        return result
    
    def set_parameters(self, angle, distance, generations):
        self.angle = angle
        self.distance = distance
        self.generations = generations


class LSystemWidget(QWidget):
    """L-system 绘图部件"""
    def __init__(self):
        super().__init__()
        self.lsystem = LSystem()
        self.path = []
        self.start_color = QColor(139, 69, 19)   # 棕色
        self.end_color = QColor(34, 139, 34)     # 绿色
        self.bg_color = QColor(240, 240, 240)    # 浅灰色背景
        self.setMinimumSize(400, 400)
        
    def set_lsystem(self, lsystem):
        self.lsystem = lsystem
        self.generate_path()
        
    def set_colors(self, start_color, end_color, bg_color):
        self.start_color = start_color
        self.end_color = end_color
        self.bg_color = bg_color
        self.update()
        
    def generate_path(self):
        """根据L-system字符串生成绘图路径"""
        self.path = []
        string = self.lsystem.generate()
        
        # 初始化状态
        stack = []
        x, y = self.width() / 2, self.height() - 50
        angle = -90  # 初始角度向上
        distance = self.lsystem.distance
        
        # 解析字符串
        for char in string:
            if char == 'F':  # 向前移动并画线
                new_x = x + distance * math.cos(math.radians(angle))
                new_y = y + distance * math.sin(math.radians(angle))
                self.path.append((QPoint(int(x), int(y)), QPoint(int(new_x), int(new_y))))
                x, y = new_x, new_y
            elif char == 'f':  # 向前移动但不画线
                x += distance * math.cos(math.radians(angle))
                y += distance * math.sin(math.radians(angle))
            elif char == '+':  # 左转
                angle += self.lsystem.angle
            elif char == '-':  # 右转
                angle -= self.lsystem.angle
            elif char == '[':  # 保存状态
                stack.append((x, y, angle))
            elif char == ']':  # 恢复状态
                x, y, angle = stack.pop()
                
        self.update()
        
    def paintEvent(self, event):
        """绘制L-system图形"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), self.bg_color)
        
        # 如果没有路径，不绘制
        if not self.path:
            return
            
        # 计算路径的边界框，用于居中显示
        min_x = min(min(start.x(), end.x()) for start, end in self.path)
        max_x = max(max(start.x(), end.x()) for start, end in self.path)
        min_y = min(min(start.y(), end.y()) for start, end in self.path)
        max_y = max(max(start.y(), end.y()) for start, end in self.path)
        
        # 计算缩放比例和偏移量
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            return
            
        scale_x = (self.width() - 100) / width
        scale_y = (self.height() - 100) / height
        scale = min(scale_x, scale_y)
        
        offset_x = (self.width() - width * scale) / 2 - min_x * scale
        offset_y = (self.height() - height * scale) / 2 - min_y * scale
        
        # 绘制路径，使用从起始颜色到结束颜色的渐变
        for i, (start, end) in enumerate(self.path):
            # 计算颜色渐变
            progress = i / len(self.path) if len(self.path) > 0 else 0
            r = int(self.start_color.red() + (self.end_color.red() - self.start_color.red()) * progress)
            g = int(self.start_color.green() + (self.end_color.green() - self.start_color.green()) * progress)
            b = int(self.start_color.blue() + (self.end_color.blue() - self.start_color.blue()) * progress)
            
            pen = QPen(QColor(r, g, b))
            pen.setWidth(max(1, int(2 * scale)))  # 根据缩放调整线宽
            painter.setPen(pen)
            
            # 应用缩放和偏移
            scaled_start = QPoint(int(start.x() * scale + offset_x), 
                                 int(start.y() * scale + offset_y))
            scaled_end = QPoint(int(end.x() * scale + offset_x), 
                               int(end.y() * scale + offset_y))
            
            painter.drawLine(scaled_start, scaled_end)
    
    def save_image(self, filename):
        """保存当前图形为图片"""
        if not self.path:
            QMessageBox.warning(self, "警告", "没有可保存的图形")
            return False
            
        # 创建与当前显示相同大小的图像
        image = QImage(self.size(), QImage.Format_ARGB32)
        image.fill(self.bg_color)
        
        # 在图像上绘制
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        self.paint_path(painter)
        painter.end()
        
        # 保存图像
        if image.save(filename):
            return True
        else:
            return False
    
    def paint_path(self, painter):
        """在指定的painter上绘制路径（用于保存）"""
        if not self.path:
            return
            
        # 计算路径的边界框，用于居中显示
        min_x = min(min(start.x(), end.x()) for start, end in self.path)
        max_x = max(max(start.x(), end.x()) for start, end in self.path)
        min_y = min(min(start.y(), end.y()) for start, end in self.path)
        max_y = max(max(start.y(), end.y()) for start, end in self.path)
        
        # 计算缩放比例和偏移量
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            return
            
        scale_x = (self.width() - 100) / width
        scale_y = (self.height() - 100) / height
        scale = min(scale_x, scale_y)
        
        offset_x = (self.width() - width * scale) / 2 - min_x * scale
        offset_y = (self.height() - height * scale) / 2 - min_y * scale
        
        # 绘制路径
        for i, (start, end) in enumerate(self.path):
            progress = i / len(self.path) if len(self.path) > 0 else 0
            r = int(self.start_color.red() + (self.end_color.red() - self.start_color.red()) * progress)
            g = int(self.start_color.green() + (self.end_color.green() - self.start_color.green()) * progress)
            b = int(self.start_color.blue() + (self.end_color.blue() - self.start_color.blue()) * progress)
            
            pen = QPen(QColor(r, g, b))
            pen.setWidth(max(1, int(2 * scale)))
            painter.setPen(pen)
            
            scaled_start = QPoint(int(start.x() * scale + offset_x), 
                                 int(start.y() * scale + offset_y))
            scaled_end = QPoint(int(end.x() * scale + offset_x), 
                               int(end.y() * scale + offset_y))
            
            painter.drawLine(scaled_start, scaled_end)


class ClickableColorLabel(QLabel):
    """可点击的颜色预览标签"""
    def __init__(self, color_type, main_window, parent=None):
        super().__init__(parent)
        self.color_type = color_type
        self.main_window = main_window
        self.setFixedSize(30, 30)
        self.setCursor(Qt.PointingHandCursor)
        
    def mousePressEvent(self, event):
        self.main_window.choose_color(self.color_type)


class LSystemWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_default_systems()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("L-system 分形生成器")
        self.setGeometry(100, 100, 900, 600)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(5)
        
        # 创建左右分割
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        control_widget = QWidget()
        control_widget.setMaximumWidth(280)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(5, 5, 5, 5)
        splitter.addWidget(control_widget)
        
        # 右侧绘图区域
        self.canvas = LSystemWidget()
        splitter.addWidget(self.canvas)
        
        # 设置分割比例
        splitter.setSizes([280, 620])
        
        # 创建界面组件
        self.create_system_selection(control_layout)
        self.create_parameters_and_colors_section(control_layout)
        self.create_rules_section(control_layout)
        self.create_buttons_section(control_layout)
        
    def create_system_selection(self, layout):
        """创建系统选择部分"""
        group_box = QGroupBox("预设系统")
        group_layout = QVBoxLayout(group_box)
        group_layout.setSpacing(8)
        group_layout.setContentsMargins(8, 0, 8, 8)
        
        self.system_combo = QComboBox()
        self.system_combo.setMinimumHeight(25)
        self.system_combo.currentTextChanged.connect(self.on_system_changed)
        
        group_layout.addWidget(QLabel("选择预设系统:"))
        group_layout.addWidget(self.system_combo)
        layout.addWidget(group_box)
    
    def create_parameters_and_colors_section(self, layout):
        """创建参数和颜色设置部分 - 两列布局"""
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setSpacing(10)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # 左侧 - 参数设置
        params_group = QGroupBox("参数设置")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(8)
        params_layout.setContentsMargins(8, 0, 8, 8)
        
        # 迭代次数
        iterations_layout = QHBoxLayout()
        iterations_layout.addWidget(QLabel("迭代:"))
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 20)
        self.iterations_spin.setValue(4)
        self.iterations_spin.setMinimumHeight(25)
        iterations_layout.addWidget(self.iterations_spin)
        iterations_layout.addStretch()
        params_layout.addLayout(iterations_layout)
        
        # 角度
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("角度:"))
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(1, 180)
        self.angle_spin.setValue(25)
        self.angle_spin.setSuffix("°")
        self.angle_spin.setMinimumHeight(25)
        angle_layout.addWidget(self.angle_spin)
        angle_layout.addStretch()
        params_layout.addLayout(angle_layout)
        
        # 步长
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("步长:"))
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(1, 50)
        self.distance_spin.setValue(10)
        self.distance_spin.setMinimumHeight(25)
        distance_layout.addWidget(self.distance_spin)
        distance_layout.addStretch()
        params_layout.addLayout(distance_layout)
        
        container_layout.addWidget(params_group)
        
        # 右侧 - 颜色设置
        colors_group = QGroupBox("颜色设置")
        colors_layout = QVBoxLayout(colors_group)
        colors_layout.setSpacing(8)
        colors_layout.setContentsMargins(8, 0, 8, 8)
        
        # 渐变起始颜色（恢复色块显示）
        start_color_layout = QHBoxLayout()
        start_color_layout.addWidget(QLabel("起始:"))
        start_color_layout.addStretch()
        self.start_color_preview = ClickableColorLabel("start", self)
        self.start_color_preview.setStyleSheet(
            f"background-color: rgb({self.canvas.start_color.red()}, {self.canvas.start_color.green()}, {self.canvas.start_color.blue()}); "
            "border: 2px solid #666; border-radius: 3px;"
        )
        start_color_layout.addWidget(self.start_color_preview)
        colors_layout.addLayout(start_color_layout)
        
        # 渐变结束颜色
        end_color_layout = QHBoxLayout()
        end_color_layout.addWidget(QLabel("结束:"))
        end_color_layout.addStretch()
        self.end_color_preview = ClickableColorLabel("end", self)
        self.end_color_preview.setStyleSheet(
            f"background-color: rgb({self.canvas.end_color.red()}, {self.canvas.end_color.green()}, {self.canvas.end_color.blue()}); "
            "border: 2px solid #666; border-radius: 3px;"
        )
        end_color_layout.addWidget(self.end_color_preview)
        colors_layout.addLayout(end_color_layout)
        
        # 背景颜色
        bg_color_layout = QHBoxLayout()
        bg_color_layout.addWidget(QLabel("背景:"))
        bg_color_layout.addStretch()
        self.bg_color_preview = ClickableColorLabel("bg", self)
        self.bg_color_preview.setStyleSheet(
            f"background-color: rgb({self.canvas.bg_color.red()}, {self.canvas.bg_color.green()}, {self.canvas.bg_color.blue()}); "
            "border: 2px solid #666; border-radius: 3px;"
        )
        bg_color_layout.addWidget(self.bg_color_preview)
        colors_layout.addLayout(bg_color_layout)
        
        container_layout.addWidget(colors_group)
        
        layout.addWidget(container)
    
    def create_rules_section(self, layout):
        """创建规则设置部分"""
        group_box = QGroupBox("L-system规则")
        group_layout = QVBoxLayout(group_box)
        group_layout.setSpacing(8)
        group_layout.setContentsMargins(8, 0, 8, 8)
        
        # 公理
        axiom_layout = QVBoxLayout()
        axiom_layout.setSpacing(4)
        axiom_layout.addWidget(QLabel("公理:"))
        self.axiom_edit = QTextEdit()
        self.axiom_edit.setMaximumHeight(35)
        self.axiom_edit.setPlaceholderText("输入初始字符串，如: F")
        axiom_layout.addWidget(self.axiom_edit)
        group_layout.addLayout(axiom_layout)
        
        # 规则
        rules_layout = QVBoxLayout()
        rules_layout.setSpacing(4)
        rules_layout.addWidget(QLabel("规则 (每行一个，格式: 前驱->后继):"))
        self.rules_edit = QTextEdit()
        self.rules_edit.setMaximumHeight(120)
        self.rules_edit.setPlaceholderText("输入规则，每行一个\n例如: F->F+F-F-F+F")
        rules_layout.addWidget(self.rules_edit)
        group_layout.addLayout(rules_layout)
        
        layout.addWidget(group_box)
    
    def create_buttons_section(self, layout):
        """创建按钮部分"""
        group_box = QGroupBox("操作")
        group_layout = QHBoxLayout(group_box)
        group_layout.setSpacing(6)
        group_layout.setContentsMargins(8, 0, 8, 8)
        
        self.generate_btn = QPushButton("生成")
        self.generate_btn.setMinimumHeight(30)
        self.generate_btn.clicked.connect(self.generate_system)
        
        self.save_btn = QPushButton("保存")
        self.save_btn.setMinimumHeight(30)
        self.save_btn.clicked.connect(self.save_image)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.setMinimumHeight(30)
        self.reset_btn.clicked.connect(self.reset_parameters)
        
        group_layout.addWidget(self.generate_btn)
        group_layout.addWidget(self.save_btn)
        group_layout.addWidget(self.reset_btn)
        
        layout.addWidget(group_box)
        
        # 添加弹性空间
        layout.addStretch()
    
    def setup_default_systems(self):
        """设置默认的L-system系统"""
        self.systems = {
            "Koch曲线": {
                "axiom": "F",
                "rules": "F->F+F-F-F+F",
                "angle": 90,
                "distance": 10,
                "iterations": 4
            },
            "分形树1": {
                "axiom": "F",
                "rules": "F->F[+F]F[-F]F",
                "angle": 25.7,
                "distance": 10,
                "iterations": 4
            },
            "分形树2": {
                "axiom": "F",
                "rules": "F->F[+F]F[-F][F]",
                "angle": 20,
                "distance": 10,
                "iterations": 5
            },
            "Sierpinski三角": {
                "axiom": "F",
                "rules": "F->G-F-G\nG->F+G+F",
                "angle": 60,
                "distance": 10,
                "iterations": 10
            },
            "分形植物": {
                "axiom": "X",
                "rules": "X->F+[[X]-X]-F[-FX]+X\nF->FF",
                "angle": 25,
                "distance": 5,
                "iterations": 6
            },
            "龙形曲线": {
                "axiom": "FX",
                "rules": "X->X+YF+\nY->-FX-Y",
                "angle": 90,
                "distance": 5,
                "iterations": 10
            },
            "Hilbert曲线": {
                "axiom": "A",
                "rules": "A->-BF+AFA+FB-\nB->+AF-BFB-FA+",
                "angle": 90,
                "distance": 10,
                "iterations": 4
            }
        }
        
        # 添加到下拉框
        for name in self.systems.keys():
            self.system_combo.addItem(name)
        
        # 选择第一个系统
        self.system_combo.setCurrentIndex(0)
    
    def on_system_changed(self, system_name):
        """当选择系统改变时"""
        if system_name in self.systems:
            system = self.systems[system_name]
            self.axiom_edit.setText(system["axiom"])
            self.rules_edit.setText(system["rules"])
            self.angle_spin.setValue(system["angle"])
            self.distance_spin.setValue(system["distance"])
            self.iterations_spin.setValue(system["iterations"])
            
            # 自动生成图形
            self.generate_system()
    
    def generate_system(self):
        """生成L-system图形"""
        # 创建L-system对象
        lsystem = LSystem()
        
        # 设置公理和规则
        lsystem.set_axiom(self.axiom_edit.toPlainText())
        
        rules_text = self.rules_edit.toPlainText()
        for line in rules_text.split('\n'):
            if '->' in line:
                predecessor, successor = line.split('->', 1)
                lsystem.add_rule(predecessor.strip(), successor.strip())
        
        # 设置参数
        lsystem.set_parameters(
            self.angle_spin.value(),
            self.distance_spin.value(),
            self.iterations_spin.value()
        )
        
        # 更新画布
        self.canvas.set_lsystem(lsystem)
    
    def reset_parameters(self):
        """重置参数"""
        self.axiom_edit.clear()
        self.rules_edit.clear()
        self.angle_spin.setValue(25)
        self.distance_spin.setValue(10)
        self.iterations_spin.setValue(4)
        self.canvas.path = []
        self.canvas.update()
    
    def choose_color(self, color_type):
        """选择颜色并更新色块显示"""
        if color_type == "start":
            current_color = self.canvas.start_color
            title = "选择渐变起始颜色"
        elif color_type == "end":
            current_color = self.canvas.end_color
            title = "选择渐变结束颜色"
        else:  # bg
            current_color = self.canvas.bg_color
            title = "选择背景颜色"
            
        color = QColorDialog.getColor(current_color, self, title)
        if color.isValid():
            if color_type == "start":
                self.canvas.start_color = color
                self.start_color_preview.setStyleSheet(
                    f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
                    "border: 2px solid #666; border-radius: 3px;"
                )
            elif color_type == "end":
                self.canvas.end_color = color
                self.end_color_preview.setStyleSheet(
                    f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
                    "border: 2px solid #666; border-radius: 3px;"
                )
            else:  # bg
                self.canvas.bg_color = color
                self.bg_color_preview.setStyleSheet(
                    f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
                    "border: 2px solid #666; border-radius: 3px;"
                )
            
            self.canvas.update()
    
    def save_image(self):
        """保存当前图像"""
        if not self.canvas.path:
            QMessageBox.warning(self, "警告", "没有可保存的图形，请先生成图形")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "保存图像", 
            f"lsystem_{self.system_combo.currentText()}.png", 
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;All Files (*)"
        )
        
        if filename:
            if self.canvas.save_image(filename):
                QMessageBox.information(self, "成功", f"图像已保存到: {filename}")
            else:
                QMessageBox.warning(self, "错误", "保存图像失败")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = LSystemWindow()
    window.show()
    sys.exit(app.exec())