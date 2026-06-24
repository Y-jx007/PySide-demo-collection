import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                               QComboBox, QSpinBox, QScrollArea, 
                               QGroupBox, QMessageBox, QTabWidget,
                               QGridLayout, QSplitter, QProgressBar)
from PySide6.QtCore import QThread, Signal, QRectF, QSize, Qt
from PySide6.QtGui import QPainter, QPen, QColor, QImage
import time
import numba
from numba import jit, float32, int32, prange

print("使用CPU计算")

# =============================================================================
# 映射函数定义
# =============================================================================

def logistic_map(x, r, n):
    """逻辑斯蒂映射"""
    for _ in range(n):
        x = r * x * (1 - x)
    return x

def logistic_map_derivative(x, r, n):
    """逻辑斯蒂映射的导数"""
    return r * (1 - 2 * x)

def sine_map(x, r, n):
    """正弦映射"""
    for _ in range(n):
        x = r * np.sin(np.pi * x) / 4
    return x

def sine_map_derivative(x, r, n):
    """正弦映射的导数"""
    return (r * np.pi * np.cos(np.pi * x)) / 4

def tent_map(x, r, n):
    """帐篷映射 - 修复版本"""
    for _ in range(n):
        if x < 0.5:
            x = 2 * r * x
        else:
            x = 2 * r * (1 - x)
        # 限制在合理范围内
        x = max(0.0, min(x, 1.0))
    return x

def tent_map_derivative(x, r, n):
    """帐篷映射的导数 - 修复版本"""
    if x < 0.5:
        return 2 * r
    else:
        return -2 * r

def cubic_map(x, r, n):
    """三次映射"""
    for _ in range(n):
        x = r * x * (1 - x * x)
    return x

def cubic_map_derivative(x, r, n):
    """三次映射的导数"""
    return r * (1 - 3 * x * x)

def exponential_map(x, r, n):
    """指数映射"""
    for _ in range(n):
        x = x * np.exp(r * (1 - x))
    return x

def exponential_map_derivative(x, r, n):
    """指数映射的导数"""
    return np.exp(r * (1 - x)) * (1 - r * x)

# =============================================================================
# 核心计算函数
# =============================================================================

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def calculate_lyapunov_fractal_cpu(width, height, x_min, x_max, y_min, y_max, 
                                  pattern_array, iterations, skip_iterations):
    """CPU并行计算Lyapunov分形 - 修复版本"""
    fractal = np.zeros((height, width), dtype=np.float32)
    seq_len = len(pattern_array)
    
    # 预计算序列索引
    total_iterations = iterations + skip_iterations
    pattern_indices = np.zeros(total_iterations, dtype=np.int32)
    for k in range(total_iterations):
        pattern_indices[k] = k % seq_len
    
    # 并行计算每个像素
    for i in prange(height):
        y_val = y_min + (y_max - y_min) * i / max(height - 1, 1)
        
        for j in range(width):
            x_val = x_min + (x_max - x_min) * j / max(width - 1, 1)
            
            # 初始化状态
            x_state = 0.5
            lyapunov_sum = 0.0
            valid_iterations = 0
            
            # 跳过瞬态
            for k in range(skip_iterations):
                idx = pattern_indices[k]
                r_val = x_val if pattern_array[idx] == 1 else y_val
                x_state = r_val * x_state * (1.0 - x_state)
                
                if x_state <= 1e-10 or x_state >= 1.0 - 1e-10:
                    break
            
            # 计算Lyapunov指数
            for k in range(skip_iterations, total_iterations):
                idx = pattern_indices[k]
                r_val = x_val if pattern_array[idx] == 1 else y_val
                
                # 计算导数
                derivative = abs(r_val * (1.0 - 2.0 * x_state))
                if derivative > 1e-10:
                    lyapunov_sum += np.log(derivative)
                    valid_iterations += 1
                
                # 更新状态
                x_state = r_val * x_state * (1.0 - x_state)
                
                if x_state <= 1e-10 or x_state >= 1.0 - 1e-10:
                    break
            
            if valid_iterations > 0:
                fractal[i, j] = lyapunov_sum / valid_iterations
            else:
                fractal[i, j] = -10.0
    
    return fractal

# =============================================================================
# 数据处理类
# =============================================================================

class CustomFunctionHandler:
    """自定义函数处理器"""
    
    def __init__(self):
        self.function_str = "r * x * (1 - x)"
        self.derivative_str = "r * (1 - 2*x)"
    
    def set_functions(self, func_str, deriv_str):
        self.function_str = func_str
        self.derivative_str = deriv_str
    
    def evaluate_function(self, x, r, n):
        for _ in range(n):
            try:
                x = eval(self.function_str, {"x": x, "r": r, "np": np, "sin": np.sin, "cos": np.cos, 
                                           "exp": np.exp, "log": np.log, "abs": abs})
            except:
                x = 0.5
        return x
    
    def evaluate_derivative(self, x, r, n):
        try:
            return eval(self.derivative_str, {"x": x, "r": r, "np": np, "sin": np.sin, "cos": np.cos, 
                                            "exp": np.exp, "log": np.log, "abs": abs})
        except:
            return 1.0

class LyapunovCalculator:
    """Lyapunov指数计算器"""
    
    @staticmethod
    def calculate_lyapunov_fractal(width, height, x_min, x_max, y_min, y_max, 
                                  sequence, iterations, skip_iterations, 
                                  use_cpu_parallel=True,
                                  map_func=logistic_map, derivative_func=logistic_map_derivative,
                                  custom_handler=None, progress_callback=None, cancel_flag=None):
        """计算Lyapunov分形 - 支持多种加速方式"""
        
        # 将序列转换为数字数组（A=1, B=2）
        pattern_array = np.array([1 if c == 'A' else 2 for c in sequence], dtype=np.int32)
        
        # 检查是否可以使用CPU并行 - 所有内置映射函数都使用并行，自定义函数除外
        if use_cpu_parallel and custom_handler is None and map_func.__name__ in ["logistic_map", "tent_map", "sine_map", "cubic_map", "exponential_map"]:
            try:
                print("使用CPU并行计算...")
                fractal = calculate_lyapunov_fractal_cpu(
                    width, height, x_min, x_max, y_min, y_max,
                    pattern_array, iterations, skip_iterations
                )
                if progress_callback:
                    progress_callback(100)
                return fractal
            except Exception as e:
                print(f"CPU并行计算失败，使用串行计算: {e}")
        
        # 回退到串行计算
        print("使用串行计算...")
        fractal = np.zeros((height, width), dtype=np.float32)
        seq_len = len(sequence)
        
        for i in range(height):
            if cancel_flag and cancel_flag():
                return None
                
            y = y_min + (y_max - y_min) * i / max(height - 1, 1)
            for j in range(width):
                if cancel_flag and cancel_flag():
                    return None
                    
                x = x_min + (x_max - x_min) * j / max(width - 1, 1)
                
                # 计算单个点的Lyapunov指数
                x_state = 0.5
                lyapunov_sum = 0.0
                valid_iterations = 0
                pattern = [x if c == 'A' else y for c in sequence]
                
                # 跳过瞬态
                for k in range(skip_iterations):
                    r = pattern[k % seq_len]
                    if custom_handler:
                        x_state = custom_handler.evaluate_function(x_state, r, 1)
                    else:
                        x_state = map_func(x_state, r, 1)
                    if x_state <= 1e-10 or x_state >= 1.0 - 1e-10:
                        break
                
                # 计算Lyapunov指数
                for k in range(iterations):
                    r = pattern[k % seq_len]
                    
                    if custom_handler:
                        derivative = abs(custom_handler.evaluate_derivative(x_state, r, 1))
                    else:
                        derivative = abs(derivative_func(x_state, r, 1))
                        
                    if derivative > 1e-10:
                        lyapunov_sum += np.log(derivative)
                        valid_iterations += 1
                    
                    if custom_handler:
                        x_state = custom_handler.evaluate_function(x_state, r, 1)
                    else:
                        x_state = map_func(x_state, r, 1)
                        
                    if x_state <= 1e-10 or x_state >= 1.0 - 1e-10:
                        break
                
                fractal[i, j] = lyapunov_sum / valid_iterations if valid_iterations > 0 else -10.0
            
            # 更新进度 - 更频繁地更新以提高响应性
            if progress_callback:
                progress = int(100 * (i + 1) / height)
                progress_callback(progress)
        
        if progress_callback:
            progress_callback(100)
            
        return fractal

class BifurcationCalculator:
    """分叉图计算器"""
    
    @staticmethod
    def calculate_bifurcation(r_min, r_max, num_points, iterations, skip_iterations, 
                             map_func, custom_handler=None, cancel_flag=None):
        """计算分叉图数据"""
        r_values = np.linspace(r_min, r_max, num_points)
        bifurcation_data = []
        
        for idx, r in enumerate(r_values):
            if cancel_flag and cancel_flag():
                return None
                
            x = 0.5
            
            # 跳过瞬态
            for _ in range(skip_iterations):
                if cancel_flag and cancel_flag():
                    return None
                    
                if custom_handler:
                    x = custom_handler.evaluate_function(x, r, 1)
                else:
                    x = map_func(x, r, 1)
            
            # 收集吸引子点
            attractor = set()
            collect_start = max(0, iterations - 100)
            for i in range(iterations):
                if cancel_flag and cancel_flag():
                    return None
                    
                if custom_handler:
                    x = custom_handler.evaluate_function(x, r, 1)
                else:
                    x = map_func(x, r, 1)
                if i >= collect_start:
                    attractor.add(round(x, 4))
            
            bifurcation_data.append((r, list(attractor)))
        
        return bifurcation_data

# =============================================================================
# 显示组件
# =============================================================================

class FractalWidget(QWidget):
    """分形显示组件 - 优化性能版本"""
    
    def __init__(self):
        super().__init__()
        self.fractal_data = None
        self.colormap = "bw"
        self.cached_image = None
        self.last_size = QSize(0, 0)
        self.setMinimumSize(400, 400)
        
        # 预定义颜色映射
        self.colormaps = {
            "viridis": self.viridis_colormap,
            "hot": self.hot_colormap,
            "cool": self.cool_colormap,
            "rainbow": self.rainbow_colormap,
            "bw": self.bw_colormap,
            "spectral": self.spectral_colormap
        }
        
    def set_fractal_data(self, data):
        self.fractal_data = data
        self.cached_image = None  # 清除缓存
        self.update()
        
    def set_colormap(self, colormap):
        if self.colormap != colormap:
            self.colormap = colormap
            self.cached_image = None  # 清除缓存
            self.update()
    
    def sizeHint(self):
        return QSize(800, 800)
    
    def paintEvent(self, event):
        if self.fractal_data is None:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # 计算绘制区域，保持图像比例
        data_height, data_width = self.fractal_data.shape
        widget_width = self.width()
        widget_height = self.height()
        
        # 计算保持比例的尺寸
        scale = min(widget_width / data_width, widget_height / data_height)
        draw_width = int(data_width * scale)
        draw_height = int(data_height * scale)
        
        # 居中绘制
        x_offset = (widget_width - draw_width) // 2
        y_offset = (widget_height - draw_height) // 2
        draw_rect = QRectF(x_offset, y_offset, draw_width, draw_height)
        
        # 检查是否需要重新生成缓存
        current_size = QSize(draw_width, draw_height)
        if (self.cached_image is None or 
            self.cached_image.size() != current_size or
            self.last_size != current_size):
            
            # 重新生成缓存 - 使用更快的缩放方法
            self.cached_image = self.fractal_data_to_image().scaled(
                draw_width, draw_height, Qt.KeepAspectRatio, Qt.FastTransformation
            )
            self.last_size = current_size
        
        # 绘制缓存的图像
        painter.drawImage(draw_rect, self.cached_image)
    
    def fractal_data_to_image(self):
        """将分形数据转换为QImage - 优化版本"""
        if self.fractal_data is None:
            return QImage()
            
        data_min = np.min(self.fractal_data)
        data_max = np.max(self.fractal_data)
        data_height, data_width = self.fractal_data.shape
        
        # 使用更快的图像创建方式
        image = QImage(data_width, data_height, QImage.Format_RGB32)
        
        # 预计算颜色映射
        colormap_func = self.colormaps.get(self.colormap, self.bw_colormap)
        
        # 优化像素设置
        for i in range(data_height):
            for j in range(data_width):
                value = self.fractal_data[i, j]
                
                if data_max > data_min:
                    normalized = (value - data_min) / (data_max - data_min)
                else:
                    normalized = 0.0
                
                color = colormap_func(normalized)
                image.setPixel(j, i, color.rgb())
        
        return image
    
    def get_color(self, value):
        """根据值和颜色映射获取颜色"""
        if value < 0:
            value = 0
        elif value > 1:
            value = 1
            
        colormap_func = self.colormaps.get(self.colormap, self.bw_colormap)
        return colormap_func(value)
    
    # 颜色映射函数
    def viridis_colormap(self, t):
        if t < 0.33:
            r = int(68 * (t / 0.33))
            g = int(1 * (t / 0.33))
            b = int(84 * (t / 0.33))
        elif t < 0.66:
            r = int(68 + (58 - 68) * ((t - 0.33) / 0.33))
            g = int(1 + (82 - 1) * ((t - 0.33) / 0.33))
            b = int(84 + (139 - 84) * ((t - 0.33) / 0.33))
        else:
            r = int(58 + (253 - 58) * ((t - 0.66) / 0.34))
            g = int(82 + (231 - 82) * ((t - 0.66) / 0.34))
            b = int(139 + (36 - 139) * ((t - 0.66) / 0.34))
        return QColor(r, g, b)
    
    def hot_colormap(self, t):
        if t < 0.33:
            r = int(255 * (t / 0.33))
            g = 0
            b = 0
        elif t < 0.66:
            r = 255
            g = int(255 * ((t - 0.33) / 0.33))
            b = 0
        else:
            r = 255
            g = 255
            b = int(255 * ((t - 0.66) / 0.34))
        return QColor(r, g, b)
    
    def cool_colormap(self, t):
        r = int(255 * t)
        g = int(255 * (1 - t))
        b = 255
        return QColor(r, g, b)
    
    def rainbow_colormap(self, t):
        r = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * t)))
        g = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.33))))
        b = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.66))))
        return QColor(r, g, b)
    
    def bw_colormap(self, t):
        intensity = int(255 * t)
        return QColor(intensity, intensity, intensity)
    
    def spectral_colormap(self, t):
        if t < 0.2:
            r = int(255 * (t / 0.2))
            g = 0
            b = int(255 * (1 - t / 0.2))
        elif t < 0.4:
            r = 255
            g = int(255 * ((t - 0.2) / 0.2))
            b = 0
        elif t < 0.6:
            r = int(255 * (1 - (t - 0.4) / 0.2))
            g = 255
            b = int(255 * ((t - 0.4) / 0.2))
        elif t < 0.8:
            r = 0
            g = int(255 * (1 - (t - 0.6) / 0.2))
            b = 255
        else:
            r = int(255 * ((t - 0.8) / 0.2))
            g = 0
            b = 255
        return QColor(r, g, b)

class BifurcationWidget(QWidget):
    """分叉图显示组件"""
    
    def __init__(self):
        super().__init__()
        self.bifurcation_data = None
        self.cached_image = None
        self.setMinimumSize(600, 600)
        
    def set_bifurcation_data(self, data):
        self.bifurcation_data = data
        self.cached_image = None  # 清除缓存
        self.update()
    
    def sizeHint(self):
        return QSize(800, 800)
    
    def paintEvent(self, event):
        if self.bifurcation_data is None:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        
        # 如果缓存不存在，则创建
        if self.cached_image is None or self.cached_image.size() != self.size():
            self.cached_image = self.render_bifurcation_image()
        
        # 绘制缓存的图像
        painter.drawImage(0, 0, self.cached_image)
    
    def render_bifurcation_image(self):
        """渲染分叉图到缓存"""
        image = QImage(self.size(), QImage.Format_RGB32)
        image.fill(QColor(255, 255, 255))
        
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, False)
        
        width = self.width()
        height = self.height()
        
        # 绘制坐标轴
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        margin = 40
        painter.drawLine(margin, height - margin, width - margin, height - margin)  # x轴
        painter.drawLine(margin, margin, margin, height - margin)  # y轴
        
        # 绘制标签
        painter.drawText(margin // 2, height // 2, "x")
        painter.drawText(width // 2, height - margin // 2, "r")
        
        if not self.bifurcation_data:
            painter.end()
            return image
            
        # 找到数据范围
        r_min = min(r for r, _ in self.bifurcation_data)
        r_max = max(r for r, _ in self.bifurcation_data)
        x_min = 0
        x_max = 1
        
        # 计算绘制区域
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin
        
        # 绘制分叉图
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        
        for r, x_values in self.bifurcation_data:
            # 映射到屏幕坐标 - 添加边界检查
            try:
                screen_r = margin + (r - r_min) / (r_max - r_min) * plot_width
                
                for x in x_values:
                    screen_x = int(screen_r)
                    screen_y = int(margin + (1 - x) * plot_height)
                    
                    # 检查坐标是否在有效范围内
                    if (0 <= screen_x < width and 0 <= screen_y < height):
                        painter.drawPoint(screen_x, screen_y)
            except (OverflowError, ValueError):
                continue
        
        painter.end()
        return image

# =============================================================================
# 工作线程
# =============================================================================

class LyapunovFractalWorker(QThread):
    """在后台线程中计算Lyapunov分形的worker"""
    
    progress_updated = Signal(int)
    calculation_finished = Signal(np.ndarray)
    calculation_cancelled = Signal()
    calculation_error = Signal(str)
    
    def __init__(self, width, height, x_range, y_range, sequence, iterations, skip_iterations, 
                 map_func, derivative_func, custom_handler=None, use_cpu_parallel=True):
        super().__init__()
        self.width = width
        self.height = height
        self.x_range = x_range
        self.y_range = y_range
        self.sequence = sequence
        self.iterations = iterations
        self.skip_iterations = skip_iterations
        self.map_func = map_func
        self.derivative_func = derivative_func
        self.custom_handler = custom_handler
        self.use_cpu_parallel = use_cpu_parallel
        self._is_cancelled = False
        
    def cancel(self):
        self._is_cancelled = True
        
    def run(self):
        try:
            start_time = time.time()
            
            def progress_callback(value):
                if not self._is_cancelled:
                    self.progress_updated.emit(value)
            
            def cancel_flag():
                return self._is_cancelled
            
            fractal = LyapunovCalculator.calculate_lyapunov_fractal(
                self.width, self.height, 
                self.x_range[0], self.x_range[1],
                self.y_range[0], self.y_range[1],
                self.sequence, self.iterations, self.skip_iterations,
                self.use_cpu_parallel,
                self.map_func, self.derivative_func, self.custom_handler,
                progress_callback, cancel_flag
            )
            
            end_time = time.time()
            print(f"计算完成，耗时: {end_time - start_time:.2f}秒")
            
            if self._is_cancelled:
                self.calculation_cancelled.emit()
            elif fractal is not None:
                self.calculation_finished.emit(fractal)
                
        except Exception as e:
            print(f"计算错误: {e}")
            self.calculation_error.emit(str(e))

class BifurcationWorker(QThread):
    """在后台线程中计算分叉图的worker"""
    
    calculation_finished = Signal(list)
    calculation_cancelled = Signal()
    
    def __init__(self, r_min, r_max, num_points, iterations, skip_iterations, 
                 map_func, custom_handler=None):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.num_points = num_points
        self.iterations = iterations
        self.skip_iterations = skip_iterations
        self.map_func = map_func
        self.custom_handler = custom_handler
        self._is_cancelled = False
        
    def cancel(self):
        self._is_cancelled = True
        
    def run(self):
        try:
            start_time = time.time()
            
            def cancel_flag():
                return self._is_cancelled
            
            bifurcation_data = BifurcationCalculator.calculate_bifurcation(
                self.r_min, self.r_max, self.num_points, 
                self.iterations, self.skip_iterations, self.map_func, self.custom_handler,
                cancel_flag
            )
            
            end_time = time.time()
            print(f"分叉图计算完成，耗时: {end_time - start_time:.2f}秒")
            
            if self._is_cancelled:
                self.calculation_cancelled.emit()
            elif bifurcation_data is not None:
                self.calculation_finished.emit(bifurcation_data)
        except Exception as e:
            print(f"分叉图计算错误: {e}")

# =============================================================================
# 主应用程序
# =============================================================================

class LyapunovFractalApp(QMainWindow):
    """主应用程序窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lyapunov分形与分叉图生成器")
        self.setGeometry(100, 100, 1100, 700)
        
        # 默认参数
        self.width = 800
        self.height = 800
        self.x_range = (2.5, 4.0)
        self.y_range = (2.5, 4.0)
        self.sequence = "AB"
        self.iterations = 200
        self.skip_iterations = 100
        
        # 分叉图参数
        self.bifurcation_range = (2.5, 4.0)
        
        # 优化选项
        self.use_cpu_parallel = True
        
        # 映射函数
        self.map_functions = {
            "逻辑斯蒂映射": (logistic_map, logistic_map_derivative, (2.5, 4.0)),
            "正弦映射": (sine_map, sine_map_derivative, (0.5, 4.0)),
            "帐篷映射": (tent_map, tent_map_derivative, (0.5, 2.0)),
            "三次映射": (cubic_map, cubic_map_derivative, (0.5, 3.0)),
            "指数映射": (exponential_map, exponential_map_derivative, (0.5, 3.0)),
            "自定义映射": (logistic_map, logistic_map_derivative, (0.5, 4.0))
        }
        
        self.current_map = "逻辑斯蒂映射"
        self.current_map_func, self.current_derivative_func, self.default_range = self.map_functions[self.current_map]
        
        # 自定义函数处理器
        self.custom_handler = CustomFunctionHandler()
        
        # 计算线程
        self.calculation_thread = None
        self.bifurcation_thread = None
        
        # 初始化UI
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # 右侧显示区域
        display_panel = self.create_display_panel()
        splitter.addWidget(display_panel)
        
        # 设置分割比例
        splitter.setSizes([350, 1050])
        
        main_layout.addWidget(splitter)
    
    def create_control_panel(self):
        """创建控制面板 - 使用两列布局"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)
        
        # 创建两列布局
        columns_layout = QHBoxLayout()
        left_column = QVBoxLayout()
        right_column = QVBoxLayout()
        
        # ==================== 左列 ====================
        
        # 映射函数设置组
        function_group = QGroupBox("映射函数设置")
        function_layout = QVBoxLayout(function_group)
        
        # 函数选择
        func_layout = QHBoxLayout()
        func_layout.addWidget(QLabel("映射函数:"))
        self.function_combo = QComboBox()
        functions = list(self.map_functions.keys())
        self.function_combo.addItems(functions)
        self.function_combo.currentTextChanged.connect(self.select_function)
        func_layout.addWidget(self.function_combo)
        function_layout.addLayout(func_layout)
        
        # 自定义函数输入
        custom_layout = QVBoxLayout()
        custom_layout.addWidget(QLabel("自定义函数:"))
        self.custom_func_edit = QLineEdit()
        self.custom_func_edit.setPlaceholderText("例如: r * x * (1 - x)")
        self.custom_func_edit.setText("r * x * (1 - x)")
        custom_layout.addWidget(self.custom_func_edit)
        
        custom_deriv_layout = QHBoxLayout()
        custom_deriv_layout.addWidget(QLabel("导数:"))
        self.custom_deriv_edit = QLineEdit()
        self.custom_deriv_edit.setPlaceholderText("例如: r * (1 - 2*x)")
        self.custom_deriv_edit.setText("r * (1 - 2*x)")
        custom_deriv_layout.addWidget(self.custom_deriv_edit)
        custom_layout.addLayout(custom_deriv_layout)
        
        self.apply_custom_btn = QPushButton("应用自定义函数")
        self.apply_custom_btn.clicked.connect(self.apply_custom_function)
        custom_layout.addWidget(self.apply_custom_btn)
        
        function_layout.addLayout(custom_layout)
        
        left_column.addWidget(function_group)
        
        # 序列设置组
        sequence_group = QGroupBox("序列设置")
        sequence_layout = QVBoxLayout(sequence_group)
        
        # 序列输入
        seq_layout = QHBoxLayout()
        seq_layout.addWidget(QLabel("AB序列:"))
        self.sequence_edit = QLineEdit(self.sequence)
        self.sequence_edit.setPlaceholderText("例如: AB, AAB, ABBA等")
        seq_layout.addWidget(self.sequence_edit)
        sequence_layout.addLayout(seq_layout)
        
        # 示例序列
        examples_layout = QHBoxLayout()
        examples_layout.addWidget(QLabel("示例:"))
        self.examples_combo = QComboBox()
        examples = [
            "AB (标准)",
            "AAB", 
            "ABB",
            "ABBA",
            "AAAB",
            "ABAB"
        ]
        self.examples_combo.addItems(examples)
        self.examples_combo.currentTextChanged.connect(self.select_example)
        examples_layout.addWidget(self.examples_combo)
        sequence_layout.addLayout(examples_layout)
        
        left_column.addWidget(sequence_group)
        
        # ==================== 右列 ====================
        
        # 参数设置组
        params_group = QGroupBox("计算参数")
        params_layout = QGridLayout(params_group)
        
        # 分辨率
        params_layout.addWidget(QLabel("分辨率:"), 0, 0)
        self.resolution_combo = QComboBox()
        resolutions = ["100x100", "200x200", "400x400", "600x600", "800x800", "1000x1000", "1200x1200", "1500x1500", "2000x2000"]
        self.resolution_combo.addItems(resolutions)
        self.resolution_combo.setCurrentText("800x800")
        self.resolution_combo.currentTextChanged.connect(self.update_resolution)
        params_layout.addWidget(self.resolution_combo, 0, 1)
        
        # 迭代次数
        params_layout.addWidget(QLabel("迭代次数:"), 1, 0)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(50, 5000)
        self.iterations_spin.setValue(self.iterations)
        self.iterations_spin.setSuffix(" 次")
        params_layout.addWidget(self.iterations_spin, 1, 1)
        
        # 跳过迭代
        params_layout.addWidget(QLabel("跳过迭代:"), 2, 0)
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(10, 2000)
        self.skip_spin.setValue(self.skip_iterations)
        self.skip_spin.setSuffix(" 次")
        params_layout.addWidget(self.skip_spin, 2, 1)
        
        right_column.addWidget(params_group)
        
        # 范围设置组
        range_group = QGroupBox("参数范围")
        range_layout = QGridLayout(range_group)
        
        # X范围
        range_layout.addWidget(QLabel("X范围 (A):"), 0, 0)
        self.x_min_edit = QLineEdit(f"{self.x_range[0]:.2f}")
        self.x_min_edit.setFixedWidth(60)
        range_layout.addWidget(self.x_min_edit, 0, 1)
        range_layout.addWidget(QLabel("到"), 0, 2)
        self.x_max_edit = QLineEdit(f"{self.x_range[1]:.2f}")
        self.x_max_edit.setFixedWidth(60)
        range_layout.addWidget(self.x_max_edit, 0, 3)
        
        # Y范围
        range_layout.addWidget(QLabel("Y范围 (B):"), 1, 0)
        self.y_min_edit = QLineEdit(f"{self.y_range[0]:.2f}")
        self.y_min_edit.setFixedWidth(60)
        range_layout.addWidget(self.y_min_edit, 1, 1)
        range_layout.addWidget(QLabel("到"), 1, 2)
        self.y_max_edit = QLineEdit(f"{self.y_range[1]:.2f}")
        self.y_max_edit.setFixedWidth(60)
        range_layout.addWidget(self.y_max_edit, 1, 3)
        
        # 分叉图范围
        range_layout.addWidget(QLabel("分叉图范围:"), 2, 0)
        self.bifurcation_min_edit = QLineEdit(f"{self.bifurcation_range[0]:.2f}")
        self.bifurcation_min_edit.setFixedWidth(60)
        range_layout.addWidget(self.bifurcation_min_edit, 2, 1)
        range_layout.addWidget(QLabel("到"), 2, 2)
        self.bifurcation_max_edit = QLineEdit(f"{self.bifurcation_range[1]:.2f}")
        self.bifurcation_max_edit.setFixedWidth(60)
        range_layout.addWidget(self.bifurcation_max_edit, 2, 3)
        
        # 预设范围
        range_layout.addWidget(QLabel("预设:"), 3, 0)
        self.range_combo = QComboBox()
        ranges = [
            "标准 (2.5-4.0)",
            "经典 (3.0-4.0)",
            "放大中心 (3.4-3.9)",
            "放大左上 (2.5-3.5)",
            "完整范围 (0-4.0)",
            "自定义"
        ]
        self.range_combo.addItems(ranges)
        self.range_combo.currentTextChanged.connect(self.select_range_preset)
        range_layout.addWidget(self.range_combo, 3, 1, 1, 3)
        
        right_column.addWidget(range_group)
        
        # 颜色设置组
        color_group = QGroupBox("颜色设置")
        color_layout = QHBoxLayout(color_group)
        
        # 颜色映射
        color_layout.addWidget(QLabel("颜色映射:"))
        self.colormap_combo = QComboBox()
        colormaps = ["bw","viridis", "hot", "cool", "rainbow", "spectral"]
        self.colormap_combo.addItems(colormaps)
        self.colormap_combo.setCurrentText("bw")
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        color_layout.addWidget(self.colormap_combo)
        
        right_column.addWidget(color_group)
        
        # 添加弹性空间使右列与左列高度一致
        right_column.addStretch()
        
        # ==================== 合并列 ====================
        
        columns_layout.addLayout(left_column)
        columns_layout.addLayout(right_column)
        layout.addLayout(columns_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.calculate_btn = QPushButton("Lyapunov分形")
        self.calculate_btn.clicked.connect(self.calculate_fractal)
        button_layout.addWidget(self.calculate_btn)

        self.cancel_btn = QPushButton("取消Lyapunov")
        self.cancel_btn.clicked.connect(self.cancel_calculation)
        self.cancel_btn.setEnabled(False)
        button_layout.addWidget(self.cancel_btn)
        
        self.calculate_bifurcation_btn = QPushButton("分叉图")
        self.calculate_bifurcation_btn.clicked.connect(self.calculate_bifurcation)
        button_layout.addWidget(self.calculate_bifurcation_btn)
        
        self.cancel_bifurcation_btn = QPushButton("取消分叉图")
        self.cancel_bifurcation_btn.clicked.connect(self.cancel_bifurcation_calculation)
        self.cancel_bifurcation_btn.setEnabled(False)
        button_layout.addWidget(self.cancel_bifurcation_btn)
        
        layout.addLayout(button_layout)
        
        # 信息显示
        self.info_label = QLabel("准备就绪")
        self.info_label.setWordWrap(True)
        self.info_label.setMaximumHeight(40)
        layout.addWidget(self.info_label)
        
        layout.addStretch()
        
        return panel
    
    def create_display_panel(self):
        """创建显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        
        # Lyapunov分形标签页
        fractal_tab = QWidget()
        fractal_layout = QVBoxLayout(fractal_tab)
        fractal_layout.setContentsMargins(0, 0, 0, 0)
        self.fractal_widget = FractalWidget()
        
        # 创建滚动区域
        self.fractal_scroll = QScrollArea()
        self.fractal_scroll.setWidget(self.fractal_widget)
        self.fractal_scroll.setWidgetResizable(True)
        fractal_layout.addWidget(self.fractal_scroll)
        self.tab_widget.addTab(fractal_tab, "Lyapunov分形")
        
        # 分叉图标签页
        bifurcation_tab = QWidget()
        bifurcation_layout = QVBoxLayout(bifurcation_tab)
        bifurcation_layout.setContentsMargins(0, 0, 0, 0)
        self.bifurcation_widget = BifurcationWidget()
        
        # 创建滚动区域
        self.bifurcation_scroll = QScrollArea()
        self.bifurcation_scroll.setWidget(self.bifurcation_widget)
        self.bifurcation_scroll.setWidgetResizable(True)
        bifurcation_layout.addWidget(self.bifurcation_scroll)
        self.tab_widget.addTab(bifurcation_tab, "分叉图")
        
        layout.addWidget(self.tab_widget)
        
        return panel
    
    # =========================================================================
    # 事件处理函数
    # =========================================================================
    
    def select_function(self, function_name):
        """选择映射函数"""
        self.current_map = function_name
        self.current_map_func, self.current_derivative_func, self.default_range = self.map_functions[function_name]
        
        # 更新默认范围
        if function_name != "自定义映射":
            self.x_min_edit.setText(f"{self.default_range[0]:.2f}")
            self.x_max_edit.setText(f"{self.default_range[1]:.2f}")
            self.y_min_edit.setText(f"{self.default_range[0]:.2f}")
            self.y_max_edit.setText(f"{self.default_range[1]:.2f}")
            self.bifurcation_min_edit.setText(f"{self.default_range[0]:.2f}")
            self.bifurcation_max_edit.setText(f"{self.default_range[1]:.2f}")
        
        self.info_label.setText(f"已选择映射函数: {function_name}")
    
    def apply_custom_function(self):
        """应用自定义函数"""
        func_str = self.custom_func_edit.text().strip()
        deriv_str = self.custom_deriv_edit.text().strip()
        
        if not func_str or not deriv_str:
            QMessageBox.warning(self, "错误", "请输入自定义函数和导数")
            return
        
        try:
            # 测试函数
            x, r = 0.5, 2.0
            test_result = eval(func_str, {"x": x, "r": r, "np": np, "sin": np.sin, "cos": np.cos, 
                                         "exp": np.exp, "log": np.log, "abs": abs})
            test_deriv = eval(deriv_str, {"x": x, "r": r, "np": np, "sin": np.sin, "cos": np.cos, 
                                         "exp": np.exp, "log": np.log, "abs": abs})
            
            # 设置自定义函数
            self.custom_handler.set_functions(func_str, deriv_str)
            self.select_function("自定义映射")
            
            self.info_label.setText("自定义函数应用成功")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"自定义函数无效: {e}")
    
    def select_example(self, example):
        """选择示例序列"""
        if example:
            sequence = example.split(" ")[0]
            self.sequence_edit.setText(sequence)
    
    def update_resolution(self, resolution):
        """更新分辨率"""
        if resolution:
            size = int(resolution.split("x")[0])
            self.width = size
            self.height = size
    
    def select_range_preset(self, preset):
        """选择范围预设"""
        if preset == "标准 (2.5-4.0)":
            self.x_min_edit.setText("2.5")
            self.x_max_edit.setText("4.0")
            self.y_min_edit.setText("2.5")
            self.y_max_edit.setText("4.0")
            self.bifurcation_min_edit.setText("2.5")
            self.bifurcation_max_edit.setText("4.0")
        elif preset == "经典 (3.0-4.0)":
            self.x_min_edit.setText("3.0")
            self.x_max_edit.setText("4.0")
            self.y_min_edit.setText("3.0")
            self.y_max_edit.setText("4.0")
            self.bifurcation_min_edit.setText("3.0")
            self.bifurcation_max_edit.setText("4.0")
        elif preset == "放大中心 (3.4-3.9)":
            self.x_min_edit.setText("3.4")
            self.x_max_edit.setText("3.9")
            self.y_min_edit.setText("3.4")
            self.y_max_edit.setText("3.9")
            self.bifurcation_min_edit.setText("3.4")
            self.bifurcation_max_edit.setText("3.9")
        elif preset == "放大左上 (2.5-3.5)":
            self.x_min_edit.setText("2.5")
            self.x_max_edit.setText("3.5")
            self.y_min_edit.setText("2.5")
            self.y_max_edit.setText("3.5")
            self.bifurcation_min_edit.setText("2.5")
            self.bifurcation_max_edit.setText("3.5")
        elif preset == "完整范围 (0-4.0)":
            self.x_min_edit.setText("0.0")
            self.x_max_edit.setText("4.0")
            self.y_min_edit.setText("0.0")
            self.y_max_edit.setText("4.0")
            self.bifurcation_min_edit.setText("0.0")
            self.bifurcation_max_edit.setText("4.0")
    
    def update_colormap(self, colormap):
        """更新颜色映射"""
        self.fractal_widget.set_colormap(colormap)
    
    def calculate_fractal(self):
        """开始计算分形"""
        # 获取参数
        try:
            self.sequence = self.sequence_edit.text().strip().upper()
            if not self.sequence or not all(c in 'AB' for c in self.sequence):
                QMessageBox.warning(self, "错误", "序列只能包含字母A和B")
                return
            
            self.iterations = self.iterations_spin.value()
            self.skip_iterations = self.skip_spin.value()
            
            x_min = float(self.x_min_edit.text())
            x_max = float(self.x_max_edit.text())
            y_min = float(self.y_min_edit.text())
            y_max = float(self.y_max_edit.text())
            
            self.x_range = (x_min, x_max)
            self.y_range = (y_min, y_max)
            
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的数值")
            return
        
        # 更新UI状态
        self.calculate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 确定计算模式
        custom_handler = self.custom_handler if self.current_map == "自定义映射" else None
        
        # 检查加速选项的兼容性 - 所有内置映射函数都使用并行，自定义函数除外
        use_cpu_parallel = self.use_cpu_parallel and (self.current_map != "自定义映射")
            
        mode_info = "CPU多核并行" if use_cpu_parallel else "串行计算"
            
        self.info_label.setText(f"正在计算Lyapunov分形 - 函数: {self.current_map}, 序列: {self.sequence}, 模式: {mode_info}")
        
        # 启动计算线程
        self.calculation_thread = LyapunovFractalWorker(
            self.width, self.height, self.x_range, self.y_range,
            self.sequence, self.iterations, self.skip_iterations,
            self.current_map_func, self.current_derivative_func, custom_handler,
            use_cpu_parallel
        )
        self.calculation_thread.progress_updated.connect(self.update_progress)
        self.calculation_thread.calculation_finished.connect(self.display_fractal)
        self.calculation_thread.calculation_cancelled.connect(self.on_calculation_cancelled)
        self.calculation_thread.calculation_error.connect(self.on_calculation_error)
        self.calculation_thread.start()
    
    def calculate_bifurcation(self):
        """开始计算分叉图"""
        try:
            r_min = float(self.bifurcation_min_edit.text())
            r_max = float(self.bifurcation_max_edit.text())
            
            if r_min >= r_max:
                QMessageBox.warning(self, "错误", "分叉图范围最小值必须小于最大值")
                return
                
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的数值")
            return
        
        # 更新UI状态
        self.calculate_bifurcation_btn.setEnabled(False)
        self.cancel_bifurcation_btn.setEnabled(True)
        self.info_label.setText(f"正在计算分叉图 - 函数: {self.current_map}")
        
        # 启动分叉图计算线程
        custom_handler = self.custom_handler if self.current_map == "自定义映射" else None
        
        self.bifurcation_thread = BifurcationWorker(
            r_min, r_max, 800,
            self.iterations_spin.value(),
            self.skip_spin.value(),
            self.current_map_func,
            custom_handler
        )
        self.bifurcation_thread.calculation_finished.connect(self.display_bifurcation)
        self.bifurcation_thread.calculation_cancelled.connect(self.on_bifurcation_cancelled)
        self.bifurcation_thread.start()
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def cancel_calculation(self):
        """取消分形计算"""
        if self.calculation_thread and self.calculation_thread.isRunning():
            self.calculation_thread.cancel()
    
    def cancel_bifurcation_calculation(self):
        """取消分叉图计算"""
        if self.bifurcation_thread and self.bifurcation_thread.isRunning():
            self.bifurcation_thread.cancel()
    
    def on_calculation_cancelled(self):
        """分形计算取消回调"""
        self.reset_fractal_ui_state()
        self.info_label.setText("分形计算已取消")
    
    def on_calculation_error(self, error_msg):
        """分形计算错误回调"""
        self.reset_fractal_ui_state()
        self.info_label.setText(f"计算错误: {error_msg}")
        QMessageBox.warning(self, "计算错误", f"分形计算过程中发生错误:\n{error_msg}")
    
    def on_bifurcation_cancelled(self):
        """分叉图计算取消回调"""
        self.reset_bifurcation_ui_state()
        self.info_label.setText("分叉图计算已取消")
    
    def display_fractal(self, fractal_data):
        """显示分形图像"""
        # 重置UI状态
        self.reset_fractal_ui_state()
        
        # 显示分形
        self.fractal_widget.set_fractal_data(fractal_data)
        self.tab_widget.setCurrentIndex(0)
        
        self.info_label.setText(f"Lyapunov分形计算完成 - 大小: {self.width}x{self.height}")
    
    def display_bifurcation(self, bifurcation_data):
        """显示分叉图"""
        # 重置UI状态
        self.reset_bifurcation_ui_state()
        
        # 显示分叉图
        self.bifurcation_widget.set_bifurcation_data(bifurcation_data)
        self.tab_widget.setCurrentIndex(1)
        
        self.info_label.setText(f"分叉图计算完成 - 函数: {self.current_map}")
    
    def reset_fractal_ui_state(self):
        """重置分形UI状态"""
        self.calculate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
    
    def reset_bifurcation_ui_state(self):
        """重置分叉图UI状态"""
        self.calculate_bifurcation_btn.setEnabled(True)
        self.cancel_bifurcation_btn.setEnabled(False)

# =============================================================================
# 应用程序入口
# =============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = LyapunovFractalApp()
    window.show()
    
    sys.exit(app.exec())