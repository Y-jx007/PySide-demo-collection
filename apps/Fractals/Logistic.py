import sys
import math
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QLineEdit,
                               QComboBox, QSpinBox, QScrollArea,
                               QGroupBox, QMessageBox, QTabWidget,
                               QGridLayout, QSplitter, QProgressBar)
from PySide6.QtCore import QThread, Signal, QRectF, QSize, Qt
from PySide6.QtGui import QPainter, QPen, QColor, QImage
import time
from numba import jit, prange

print("使用CPU计算")

# =============================================================================
# 映射函数定义
# =============================================================================

def logistic_map(x, r, n):
    for _ in range(n):
        x = r * x * (1 - x)
    return x

def logistic_map_derivative(x, r, n):
    return r * (1 - 2 * x)

def sine_map(x, r, n):
    for _ in range(n):
        x = r * np.sin(np.pi * x) / 4
    return x

def sine_map_derivative(x, r, n):
    return (r * np.pi * np.cos(np.pi * x)) / 4

def tent_map(x, r, n):
    for _ in range(n):
        x = 2 * r * x if x < 0.5 else 2 * r * (1 - x)
        x = max(0.0, min(x, 1.0))
    return x

def tent_map_derivative(x, r, n):
    return 2 * r if x < 0.5 else -2 * r

def cubic_map(x, r, n):
    for _ in range(n):
        x = r * x * (1 - x * x)
    return x

def cubic_map_derivative(x, r, n):
    return r * (1 - 3 * x * x)

def exponential_map(x, r, n):
    for _ in range(n):
        x = x * np.exp(r * (1 - x))
    return x

def exponential_map_derivative(x, r, n):
    return np.exp(r * (1 - x)) * (1 - r * x)

# =============================================================================
# 核心计算函数 (Numba 加速)
# =============================================================================

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def calculate_lyapunov_fractal_cpu(width, height, x_min, x_max, y_min, y_max,
                                  pattern_array, iterations, skip_iterations):
    fractal = np.zeros((height, width), dtype=np.float32)
    seq_len = len(pattern_array)
    total_iterations = iterations + skip_iterations
    pattern_indices = np.zeros(total_iterations, dtype=np.int32)
    for k in range(total_iterations):
        pattern_indices[k] = k % seq_len

    for i in prange(height):
        y_val = y_min + (y_max - y_min) * i / max(height - 1, 1)
        for j in range(width):
            x_val = x_min + (x_max - x_min) * j / max(width - 1, 1)
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

            # 计算 Lyapunov 指数
            for k in range(skip_iterations, total_iterations):
                idx = pattern_indices[k]
                r_val = x_val if pattern_array[idx] == 1 else y_val
                derivative = abs(r_val * (1.0 - 2.0 * x_state))
                if derivative > 1e-10:
                    lyapunov_sum += math.log(derivative)  # Numba 支持 math.log
                    valid_iterations += 1
                x_state = r_val * x_state * (1.0 - x_state)
                if x_state <= 1e-10 or x_state >= 1.0 - 1e-10:
                    break

            fractal[i, j] = lyapunov_sum / valid_iterations if valid_iterations > 0 else -10.0
    return fractal

# =============================================================================
# 数据处理类
# =============================================================================

class CustomFunctionHandler:
    def __init__(self):
        self.function_str = "r * x * (1 - x)"
        self.derivative_str = "r * (1 - 2*x)"

    def set_functions(self, func_str, deriv_str):
        self.function_str = func_str
        self.derivative_str = deriv_str

    def evaluate_function(self, x, r, n):
        for _ in range(n):
            try:
                x = eval(self.function_str, {"x": x, "r": r, "np": np, "sin": math.sin, "cos": math.cos,
                                            "exp": math.exp, "log": math.log, "abs": abs})
            except:
                x = 0.5
        return x

    def evaluate_derivative(self, x, r, n):
        try:
            return eval(self.derivative_str, {"x": x, "r": r, "np": np, "sin": math.sin, "cos": math.cos,
                                             "exp": math.exp, "log": math.log, "abs": abs})
        except:
            return 1.0

class LyapunovCalculator:
    @staticmethod
    def calculate_lyapunov_fractal(width, height, x_min, x_max, y_min, y_max,
                                  sequence, iterations, skip_iterations,
                                  use_cpu_parallel=True,
                                  map_func=logistic_map, derivative_func=logistic_map_derivative,
                                  custom_handler=None, progress_callback=None, cancel_flag=None):
        pattern_array = np.array([1 if c == 'A' else 2 for c in sequence], dtype=np.int32)

        # 尝试使用 Numba 并行计算（内置函数）
        if use_cpu_parallel and custom_handler is None and map_func.__name__ in {
            "logistic_map", "tent_map", "sine_map", "cubic_map", "exponential_map"
        }:
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

        # 串行回退
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
                x_state = 0.5
                lyapunov_sum = 0.0
                valid_iterations = 0
                pattern = [x if c == 'A' else y for c in sequence]

                for k in range(skip_iterations):
                    r = pattern[k % seq_len]
                    if custom_handler:
                        x_state = custom_handler.evaluate_function(x_state, r, 1)
                    else:
                        x_state = map_func(x_state, r, 1)
                    if x_state <= 1e-10 or x_state >= 1.0 - 1e-10:
                        break

                for k in range(iterations):
                    r = pattern[k % seq_len]
                    if custom_handler:
                        derivative = abs(custom_handler.evaluate_derivative(x_state, r, 1))
                    else:
                        derivative = abs(derivative_func(x_state, r, 1))
                    if derivative > 1e-10:
                        lyapunov_sum += math.log(derivative)
                        valid_iterations += 1
                    if custom_handler:
                        x_state = custom_handler.evaluate_function(x_state, r, 1)
                    else:
                        x_state = map_func(x_state, r, 1)
                    if x_state <= 1e-10 or x_state >= 1.0 - 1e-10:
                        break

                fractal[i, j] = lyapunov_sum / valid_iterations if valid_iterations > 0 else -10.0

            if progress_callback:
                progress_callback(int(100 * (i + 1) / height))

        if progress_callback:
            progress_callback(100)
        return fractal

class BifurcationCalculator:
    @staticmethod
    def calculate_bifurcation(r_min, r_max, num_points, iterations, skip_iterations,
                             map_func, custom_handler=None, cancel_flag=None):
        r_values = np.linspace(r_min, r_max, num_points)
        bifurcation_data = []
        for idx, r in enumerate(r_values):
            if cancel_flag and cancel_flag():
                return None
            x = 0.5
            for _ in range(skip_iterations):
                if cancel_flag and cancel_flag():
                    return None
                if custom_handler:
                    x = custom_handler.evaluate_function(x, r, 1)
                else:
                    x = map_func(x, r, 1)
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
# 显示组件（性能优化重点）
# =============================================================================

class FractalWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.fractal_data = None
        self.colormap = "bw"
        self.cached_image = None
        self.setMinimumSize(400, 400)
        # 预计算颜色查找表 (256级)
        self._colormap_lut = None
        self._build_lut()

    def _build_lut(self):
        """构建当前颜色映射的256级查找表"""
        if self.colormap == "bw":
            lut = np.zeros((256, 3), dtype=np.uint8)
            for i in range(256):
                lut[i] = [i, i, i]
            self._colormap_lut = lut
        elif self.colormap == "hot":
            lut = np.zeros((256, 3), dtype=np.uint8)
            for i in range(256):
                t = i / 255.0
                if t < 0.33:
                    r = int(255 * (t / 0.33))
                    g, b = 0, 0
                elif t < 0.66:
                    r = 255
                    g = int(255 * ((t - 0.33) / 0.33))
                    b = 0
                else:
                    r, g = 255, 255
                    b = int(255 * ((t - 0.66) / 0.34))
                lut[i] = [min(r,255), min(g,255), min(b,255)]
            self._colormap_lut = lut
        elif self.colormap == "cool":
            lut = np.zeros((256, 3), dtype=np.uint8)
            for i in range(256):
                t = i / 255.0
                lut[i] = [int(255*t), int(255*(1-t)), 255]
            self._colormap_lut = lut
        elif self.colormap == "rainbow":
            lut = np.zeros((256, 3), dtype=np.uint8)
            for i in range(256):
                t = i / 255.0
                r = int(255 * (0.5 + 0.5 * math.cos(2 * math.pi * t)))
                g = int(255 * (0.5 + 0.5 * math.cos(2 * math.pi * (t + 0.33))))
                b = int(255 * (0.5 + 0.5 * math.cos(2 * math.pi * (t + 0.66))))
                lut[i] = [r, g, b]
            self._colormap_lut = lut
        elif self.colormap == "spectral":
            lut = np.zeros((256, 3), dtype=np.uint8)
            for i in range(256):
                t = i / 255.0
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
                lut[i] = [r, g, b]
            self._colormap_lut = lut
        else:  # viridis 及其他默认
            lut = np.zeros((256, 3), dtype=np.uint8)
            for i in range(256):
                t = i / 255.0
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
                lut[i] = [r, g, b]
            self._colormap_lut = lut

    def set_fractal_data(self, data):
        self.fractal_data = data
        self.cached_image = None
        self.update()

    def set_colormap(self, colormap):
        if self.colormap != colormap:
            self.colormap = colormap
            self.cached_image = None
            self._build_lut()
            self.update()

    def sizeHint(self):
        return QSize(800, 800)

    def paintEvent(self, event):
        if self.fractal_data is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        data_h, data_w = self.fractal_data.shape
        w = self.width()
        h = self.height()
        scale = min(w / data_w, h / data_h)
        draw_w = int(data_w * scale)
        draw_h = int(data_h * scale)
        x_off = (w - draw_w) // 2
        y_off = (h - draw_h) // 2
        rect = QRectF(x_off, y_off, draw_w, draw_h)
        if self.cached_image is None or self.cached_image.size() != QSize(draw_w, draw_h):
            img = self._fractal_to_qimage()
            self.cached_image = img.scaled(draw_w, draw_h, Qt.KeepAspectRatio, Qt.FastTransformation)
        painter.drawImage(rect, self.cached_image)

    def _fractal_to_qimage(self):
        """将分形数据转换为 QImage（使用内存缓冲区，极快）"""
        if self.fractal_data is None:
            return QImage()
        data = self.fractal_data
        dmin, dmax = np.min(data), np.max(data)
        if dmax == dmin:
            dmax = dmin + 1e-10
        h, w = data.shape
        # 归一化到 0-255
        norm = ((data - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
        # 通过LUT获得RGB图像 (h, w, 3)
        rgb = self._colormap_lut[norm]
        img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        # 深拷贝以避免数据被垃圾回收
        return img.copy()

class BifurcationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.bifurcation_data = None
        self.cached_image = None
        self.setMinimumSize(600, 600)

    def set_bifurcation_data(self, data):
        self.bifurcation_data = data
        self.cached_image = None
        self.update()

    def sizeHint(self):
        return QSize(800, 800)

    def paintEvent(self, event):
        if self.bifurcation_data is None:
            return
        if self.cached_image is None or self.cached_image.size() != self.size():
            self.cached_image = self.render_bifurcation_image()
        painter = QPainter(self)
        painter.drawImage(0, 0, self.cached_image)

    def render_bifurcation_image(self):
        image = QImage(self.size(), QImage.Format_RGB32)
        image.fill(QColor(255, 255, 255))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, False)
        width, height = self.width(), self.height()
        margin = 40
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawLine(margin, height - margin, width - margin, height - margin)
        painter.drawLine(margin, margin, margin, height - margin)
        painter.drawText(margin // 2, height // 2, "x")
        painter.drawText(width // 2, height - margin // 2, "r")
        if not self.bifurcation_data:
            painter.end()
            return image
        r_min = min(r for r, _ in self.bifurcation_data)
        r_max = max(r for r, _ in self.bifurcation_data)
        plot_w = width - 2 * margin
        plot_h = height - 2 * margin
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        for r, x_vals in self.bifurcation_data:
            try:
                screen_r = int(margin + (r - r_min) / (r_max - r_min) * plot_w)
                for x in x_vals:
                    screen_y = int(margin + (1 - x) * plot_h)
                    if 0 <= screen_r < width and 0 <= screen_y < height:
                        painter.drawPoint(screen_r, screen_y)
            except (OverflowError, ValueError):
                continue
        painter.end()
        return image

# =============================================================================
# 工作线程（不变）
# =============================================================================

class LyapunovFractalWorker(QThread):
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
            start = time.time()
            def progress_cb(val):
                if not self._is_cancelled:
                    self.progress_updated.emit(val)
            def cancel_flag():
                return self._is_cancelled
            fractal = LyapunovCalculator.calculate_lyapunov_fractal(
                self.width, self.height, self.x_range[0], self.x_range[1],
                self.y_range[0], self.y_range[1], self.sequence,
                self.iterations, self.skip_iterations, self.use_cpu_parallel,
                self.map_func, self.derivative_func, self.custom_handler,
                progress_cb, cancel_flag
            )
            print(f"计算完成，耗时: {time.time() - start:.2f}秒")
            if self._is_cancelled:
                self.calculation_cancelled.emit()
            elif fractal is not None:
                self.calculation_finished.emit(fractal)
        except Exception as e:
            print(f"计算错误: {e}")
            self.calculation_error.emit(str(e))

class BifurcationWorker(QThread):
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
            start = time.time()
            def cancel_flag():
                return self._is_cancelled
            data = BifurcationCalculator.calculate_bifurcation(
                self.r_min, self.r_max, self.num_points,
                self.iterations, self.skip_iterations, self.map_func,
                self.custom_handler, cancel_flag
            )
            print(f"分叉图计算完成，耗时: {time.time() - start:.2f}秒")
            if self._is_cancelled:
                self.calculation_cancelled.emit()
            elif data is not None:
                self.calculation_finished.emit(data)
        except Exception as e:
            print(f"分叉图计算错误: {e}")

# =============================================================================
# 主应用程序（UI完全保持不变，仅精简部分内部代码）
# =============================================================================

class LyapunovFractalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lyapunov分形与分叉图生成器")
        self.setGeometry(100, 100, 1100, 700)
        self.width = 800
        self.height = 800
        self.x_range = (2.5, 4.0)
        self.y_range = (2.5, 4.0)
        self.sequence = "AB"
        self.iterations = 200
        self.skip_iterations = 100
        self.bifurcation_range = (2.5, 4.0)
        self.use_cpu_parallel = True
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
        self.custom_handler = CustomFunctionHandler()
        self.calculation_thread = None
        self.bifurcation_thread = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        display_panel = self.create_display_panel()
        splitter.addWidget(display_panel)
        splitter.setSizes([350, 1050])
        main_layout.addWidget(splitter)

    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)
        cols = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()

        # 左列 - 映射函数 + 序列
        func_group = QGroupBox("映射函数设置")
        func_layout = QVBoxLayout(func_group)
        fl = QHBoxLayout()
        fl.addWidget(QLabel("映射函数:"))
        self.function_combo = QComboBox()
        self.function_combo.addItems(list(self.map_functions.keys()))
        self.function_combo.currentTextChanged.connect(self.select_function)
        fl.addWidget(self.function_combo)
        func_layout.addLayout(fl)

        cl = QVBoxLayout()
        cl.addWidget(QLabel("自定义函数:"))
        self.custom_func_edit = QLineEdit()
        self.custom_func_edit.setPlaceholderText("例如: r * x * (1 - x)")
        self.custom_func_edit.setText("r * x * (1 - x)")
        cl.addWidget(self.custom_func_edit)
        cdl = QHBoxLayout()
        cdl.addWidget(QLabel("导数:"))
        self.custom_deriv_edit = QLineEdit()
        self.custom_deriv_edit.setPlaceholderText("例如: r * (1 - 2*x)")
        self.custom_deriv_edit.setText("r * (1 - 2*x)")
        cdl.addWidget(self.custom_deriv_edit)
        cl.addLayout(cdl)
        self.apply_custom_btn = QPushButton("应用自定义函数")
        self.apply_custom_btn.clicked.connect(self.apply_custom_function)
        cl.addWidget(self.apply_custom_btn)
        func_layout.addLayout(cl)
        left.addWidget(func_group)

        seq_group = QGroupBox("序列设置")
        seq_layout = QVBoxLayout(seq_group)
        sl = QHBoxLayout()
        sl.addWidget(QLabel("AB序列:"))
        self.sequence_edit = QLineEdit(self.sequence)
        self.sequence_edit.setPlaceholderText("例如: AB, AAB, ABBA等")
        sl.addWidget(self.sequence_edit)
        seq_layout.addLayout(sl)
        el = QHBoxLayout()
        el.addWidget(QLabel("示例:"))
        self.examples_combo = QComboBox()
        self.examples_combo.addItems(["AB (标准)", "AAB", "ABB", "ABBA", "AAAB", "ABAB"])
        self.examples_combo.currentTextChanged.connect(self.select_example)
        el.addWidget(self.examples_combo)
        seq_layout.addLayout(el)
        left.addWidget(seq_group)

        # 右列 - 参数 + 范围 + 颜色
        params_group = QGroupBox("计算参数")
        params_layout = QGridLayout(params_group)
        params_layout.addWidget(QLabel("分辨率:"), 0, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["100x100", "200x200", "400x400", "600x600", "800x800", "1000x1000", "1200x1200", "1500x1500", "2000x2000"])
        self.resolution_combo.setCurrentText("800x800")
        self.resolution_combo.currentTextChanged.connect(self.update_resolution)
        params_layout.addWidget(self.resolution_combo, 0, 1)
        params_layout.addWidget(QLabel("迭代次数:"), 1, 0)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(50, 5000)
        self.iterations_spin.setValue(self.iterations)
        self.iterations_spin.setSuffix(" 次")
        params_layout.addWidget(self.iterations_spin, 1, 1)
        params_layout.addWidget(QLabel("跳过迭代:"), 2, 0)
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(10, 2000)
        self.skip_spin.setValue(self.skip_iterations)
        self.skip_spin.setSuffix(" 次")
        params_layout.addWidget(self.skip_spin, 2, 1)
        right.addWidget(params_group)

        range_group = QGroupBox("参数范围")
        range_layout = QGridLayout(range_group)
        range_layout.addWidget(QLabel("X范围 (A):"), 0, 0)
        self.x_min_edit = QLineEdit(f"{self.x_range[0]:.2f}")
        self.x_min_edit.setFixedWidth(60)
        range_layout.addWidget(self.x_min_edit, 0, 1)
        range_layout.addWidget(QLabel("到"), 0, 2)
        self.x_max_edit = QLineEdit(f"{self.x_range[1]:.2f}")
        self.x_max_edit.setFixedWidth(60)
        range_layout.addWidget(self.x_max_edit, 0, 3)
        range_layout.addWidget(QLabel("Y范围 (B):"), 1, 0)
        self.y_min_edit = QLineEdit(f"{self.y_range[0]:.2f}")
        self.y_min_edit.setFixedWidth(60)
        range_layout.addWidget(self.y_min_edit, 1, 1)
        range_layout.addWidget(QLabel("到"), 1, 2)
        self.y_max_edit = QLineEdit(f"{self.y_range[1]:.2f}")
        self.y_max_edit.setFixedWidth(60)
        range_layout.addWidget(self.y_max_edit, 1, 3)
        range_layout.addWidget(QLabel("分叉图范围:"), 2, 0)
        self.bifurcation_min_edit = QLineEdit(f"{self.bifurcation_range[0]:.2f}")
        self.bifurcation_min_edit.setFixedWidth(60)
        range_layout.addWidget(self.bifurcation_min_edit, 2, 1)
        range_layout.addWidget(QLabel("到"), 2, 2)
        self.bifurcation_max_edit = QLineEdit(f"{self.bifurcation_range[1]:.2f}")
        self.bifurcation_max_edit.setFixedWidth(60)
        range_layout.addWidget(self.bifurcation_max_edit, 2, 3)
        range_layout.addWidget(QLabel("预设:"), 3, 0)
        self.range_combo = QComboBox()
        self.range_combo.addItems(["标准 (2.5-4.0)", "经典 (3.0-4.0)", "放大中心 (3.4-3.9)", "放大左上 (2.5-3.5)", "完整范围 (0-4.0)", "自定义"])
        self.range_combo.currentTextChanged.connect(self.select_range_preset)
        range_layout.addWidget(self.range_combo, 3, 1, 1, 3)
        right.addWidget(range_group)

        color_group = QGroupBox("颜色设置")
        color_layout = QHBoxLayout(color_group)
        color_layout.addWidget(QLabel("颜色映射:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["bw", "viridis", "hot", "cool", "rainbow", "spectral"])
        self.colormap_combo.setCurrentText("bw")
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        color_layout.addWidget(self.colormap_combo)
        right.addWidget(color_group)
        right.addStretch()

        cols.addLayout(left)
        cols.addLayout(right)
        layout.addLayout(cols)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        self.calculate_btn = QPushButton("Lyapunov分形")
        self.calculate_btn.clicked.connect(self.calculate_fractal)
        btn_layout.addWidget(self.calculate_btn)
        self.cancel_btn = QPushButton("取消Lyapunov")
        self.cancel_btn.clicked.connect(self.cancel_calculation)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)
        self.calculate_bifurcation_btn = QPushButton("分叉图")
        self.calculate_bifurcation_btn.clicked.connect(self.calculate_bifurcation)
        btn_layout.addWidget(self.calculate_bifurcation_btn)
        self.cancel_bifurcation_btn = QPushButton("取消分叉图")
        self.cancel_bifurcation_btn.clicked.connect(self.cancel_bifurcation_calculation)
        self.cancel_bifurcation_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_bifurcation_btn)
        layout.addLayout(btn_layout)

        self.info_label = QLabel("准备就绪")
        self.info_label.setWordWrap(True)
        self.info_label.setMaximumHeight(40)
        layout.addWidget(self.info_label)
        layout.addStretch()
        return panel

    def create_display_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tab_widget = QTabWidget()

        fractal_tab = QWidget()
        fractal_layout = QVBoxLayout(fractal_tab)
        fractal_layout.setContentsMargins(0, 0, 0, 0)
        self.fractal_widget = FractalWidget()
        self.fractal_scroll = QScrollArea()
        self.fractal_scroll.setWidget(self.fractal_widget)
        self.fractal_scroll.setWidgetResizable(True)
        fractal_layout.addWidget(self.fractal_scroll)
        self.tab_widget.addTab(fractal_tab, "Lyapunov分形")

        bifurcation_tab = QWidget()
        bifurcation_layout = QVBoxLayout(bifurcation_tab)
        bifurcation_layout.setContentsMargins(0, 0, 0, 0)
        self.bifurcation_widget = BifurcationWidget()
        self.bifurcation_scroll = QScrollArea()
        self.bifurcation_scroll.setWidget(self.bifurcation_widget)
        self.bifurcation_scroll.setWidgetResizable(True)
        bifurcation_layout.addWidget(self.bifurcation_scroll)
        self.tab_widget.addTab(bifurcation_tab, "分叉图")

        layout.addWidget(self.tab_widget)
        return panel

    # ---- 事件处理 ----
    def select_function(self, name):
        self.current_map = name
        self.current_map_func, self.current_derivative_func, self.default_range = self.map_functions[name]
        if name != "自定义映射":
            self.x_min_edit.setText(f"{self.default_range[0]:.2f}")
            self.x_max_edit.setText(f"{self.default_range[1]:.2f}")
            self.y_min_edit.setText(f"{self.default_range[0]:.2f}")
            self.y_max_edit.setText(f"{self.default_range[1]:.2f}")
            self.bifurcation_min_edit.setText(f"{self.default_range[0]:.2f}")
            self.bifurcation_max_edit.setText(f"{self.default_range[1]:.2f}")
        self.info_label.setText(f"已选择映射函数: {name}")

    def apply_custom_function(self):
        fs = self.custom_func_edit.text().strip()
        ds = self.custom_deriv_edit.text().strip()
        if not fs or not ds:
            QMessageBox.warning(self, "错误", "请输入自定义函数和导数")
            return
        try:
            x, r = 0.5, 2.0
            eval(fs, {"x": x, "r": r, "np": np, "sin": math.sin, "cos": math.cos, "exp": math.exp, "log": math.log, "abs": abs})
            eval(ds, {"x": x, "r": r, "np": np, "sin": math.sin, "cos": math.cos, "exp": math.exp, "log": math.log, "abs": abs})
            self.custom_handler.set_functions(fs, ds)
            self.select_function("自定义映射")
            self.info_label.setText("自定义函数应用成功")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"自定义函数无效: {e}")

    def select_example(self, ex):
        if ex:
            self.sequence_edit.setText(ex.split(" ")[0])

    def update_resolution(self, res):
        if res:
            s = int(res.split("x")[0])
            self.width = self.height = s

    def select_range_preset(self, preset):
        presets = {
            "标准 (2.5-4.0)": (2.5, 4.0),
            "经典 (3.0-4.0)": (3.0, 4.0),
            "放大中心 (3.4-3.9)": (3.4, 3.9),
            "放大左上 (2.5-3.5)": (2.5, 3.5),
            "完整范围 (0-4.0)": (0.0, 4.0),
        }
        if preset in presets:
            a, b = presets[preset]
            self.x_min_edit.setText(f"{a:.2f}")
            self.x_max_edit.setText(f"{b:.2f}")
            self.y_min_edit.setText(f"{a:.2f}")
            self.y_max_edit.setText(f"{b:.2f}")
            self.bifurcation_min_edit.setText(f"{a:.2f}")
            self.bifurcation_max_edit.setText(f"{b:.2f}")

    def update_colormap(self, cmap):
        self.fractal_widget.set_colormap(cmap)

    def calculate_fractal(self):
        seq = self.sequence_edit.text().strip().upper()
        if not seq or any(c not in 'AB' for c in seq):
            QMessageBox.warning(self, "错误", "序列只能包含字母A和B")
            return
        try:
            self.sequence = seq
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

        self.calculate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        custom = self.custom_handler if self.current_map == "自定义映射" else None
        use_parallel = self.use_cpu_parallel and self.current_map != "自定义映射"
        mode = "CPU多核并行" if use_parallel else "串行计算"
        self.info_label.setText(f"正在计算Lyapunov分形 - 函数: {self.current_map}, 序列: {self.sequence}, 模式: {mode}")

        self.calculation_thread = LyapunovFractalWorker(
            self.width, self.height, self.x_range, self.y_range,
            self.sequence, self.iterations, self.skip_iterations,
            self.current_map_func, self.current_derivative_func, custom, use_parallel
        )
        self.calculation_thread.progress_updated.connect(self.update_progress)
        self.calculation_thread.calculation_finished.connect(self.display_fractal)
        self.calculation_thread.calculation_cancelled.connect(self.on_calculation_cancelled)
        self.calculation_thread.calculation_error.connect(self.on_calculation_error)
        self.calculation_thread.start()

    def calculate_bifurcation(self):
        try:
            r_min = float(self.bifurcation_min_edit.text())
            r_max = float(self.bifurcation_max_edit.text())
            if r_min >= r_max:
                QMessageBox.warning(self, "错误", "分叉图范围最小值必须小于最大值")
                return
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的数值")
            return
        self.calculate_bifurcation_btn.setEnabled(False)
        self.cancel_bifurcation_btn.setEnabled(True)
        self.info_label.setText(f"正在计算分叉图 - 函数: {self.current_map}")
        custom = self.custom_handler if self.current_map == "自定义映射" else None
        self.bifurcation_thread = BifurcationWorker(
            r_min, r_max, 800,
            self.iterations_spin.value(), self.skip_spin.value(),
            self.current_map_func, custom
        )
        self.bifurcation_thread.calculation_finished.connect(self.display_bifurcation)
        self.bifurcation_thread.calculation_cancelled.connect(self.on_bifurcation_cancelled)
        self.bifurcation_thread.start()

    def update_progress(self, val):
        self.progress_bar.setValue(val)

    def cancel_calculation(self):
        if self.calculation_thread and self.calculation_thread.isRunning():
            self.calculation_thread.cancel()

    def cancel_bifurcation_calculation(self):
        if self.bifurcation_thread and self.bifurcation_thread.isRunning():
            self.bifurcation_thread.cancel()

    def on_calculation_cancelled(self):
        self.reset_fractal_ui()
        self.info_label.setText("分形计算已取消")

    def on_calculation_error(self, msg):
        self.reset_fractal_ui()
        self.info_label.setText(f"计算错误: {msg}")
        QMessageBox.warning(self, "计算错误", f"分形计算过程中发生错误:\n{msg}")

    def on_bifurcation_cancelled(self):
        self.reset_bifurcation_ui()
        self.info_label.setText("分叉图计算已取消")

    def display_fractal(self, data):
        self.reset_fractal_ui()
        self.fractal_widget.set_fractal_data(data)
        self.tab_widget.setCurrentIndex(0)
        self.info_label.setText(f"Lyapunov分形计算完成 - 大小: {self.width}x{self.height}")

    def display_bifurcation(self, data):
        self.reset_bifurcation_ui()
        self.bifurcation_widget.set_bifurcation_data(data)
        self.tab_widget.setCurrentIndex(1)
        self.info_label.setText(f"分叉图计算完成 - 函数: {self.current_map}")

    def reset_fractal_ui(self):
        self.calculate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)

    def reset_bifurcation_ui(self):
        self.calculate_bifurcation_btn.setEnabled(True)
        self.cancel_bifurcation_btn.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LyapunovFractalApp()
    window.show()
    sys.exit(app.exec())