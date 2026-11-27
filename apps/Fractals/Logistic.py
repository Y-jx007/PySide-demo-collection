import math
import logging
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QLineEdit,
                               QComboBox, QSpinBox,
                               QGroupBox, QMessageBox, QTabWidget,
                               QGridLayout, QProgressBar, QDialog, QDialogButtonBox)
from PySide6.QtCore import QThread, Signal, QRectF, QSize, Qt
from PySide6.QtGui import QPainter, QPen, QColor, QImage
import time
from numba import jit, prange

from common.app import run_app
from common.taichi_base import ensure_taichi, ti

logging.debug("Lyapunov 分形使用 CPU/Numba 计算")

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

MAP_FORMULAS = {
    "逻辑斯蒂映射": "xₙ₊₁ = r·xₙ(1−xₙ)    f′ = r(1−2xₙ)",
    "正弦映射": "xₙ₊₁ = r·sin(πxₙ)/4    f′ = rπ·cos(πxₙ)/4",
    "帐篷映射": "xₙ₊₁ = 2r·xₙ (x<½) / 2r(1−xₙ) (x≥½)    f′ = ±2r",
    "三次映射": "xₙ₊₁ = r·xₙ(1−xₙ²)    f′ = r(1−3xₙ²)",
    "指数映射": "xₙ₊₁ = xₙ·exp(r(1−xₙ))    f′ = exp(r(1−xₙ))(1−r·xₙ)",
    "自定义映射": "自定义 f(x) 与 f′(x)，按输入的表达式计算",
}

# =============================================================================
# Taichi GPU 内核
# =============================================================================

_TI = ensure_taichi()


@ti.func
def _ti_map_step(x, r, map_code):
    y = 0.0
    if map_code == 1:
        y = r * ti.sin(math.pi * x) / 4.0
    elif map_code == 2:
        y = 2.0 * r * x if x < 0.5 else 2.0 * r * (1.0 - x)
        y = ti.min(1.0, ti.max(0.0, y))
    elif map_code == 3:
        y = r * x * (1.0 - x * x)
    elif map_code == 4:
        y = x * ti.exp(r * (1.0 - x))
    else:
        y = r * x * (1.0 - x)
    return y


@ti.func
def _ti_map_derivative(x, r, map_code):
    y = 1.0
    if map_code == 1:
        y = r * math.pi * ti.cos(math.pi * x) / 4.0
    elif map_code == 2:
        y = 2.0 * r if x < 0.5 else -2.0 * r
    elif map_code == 3:
        y = r * (1.0 - 3.0 * x * x)
    elif map_code == 4:
        y = ti.exp(r * (1.0 - x)) * (1.0 - r * x)
    else:
        y = r * (1.0 - 2.0 * x)
    return y


@ti.kernel
def _ti_lyapunov_kernel(out: ti.template(), pattern: ti.template(),
                        x_min: ti.f32, x_max: ti.f32,
                        y_min: ti.f32, y_max: ti.f32,
                        iterations: ti.i32, skip_iterations: ti.i32,
                        map_code: ti.i32):
    height, width = out.shape
    seq_len = pattern.shape[0]
    for i, j in ti.ndrange(height, width):
        y_val = y_min + (y_max - y_min) * i / max(height - 1, 1)
        x_val = x_min + (x_max - x_min) * j / max(width - 1, 1)
        x_state = 0.5
        lyapunov_sum = 0.0
        valid_iterations = 0
        for k in range(skip_iterations):
            r_val = x_val if pattern[k % seq_len] == 1 else y_val
            x_state = _ti_map_step(x_state, r_val, map_code)
            if x_state <= 1e-10 or x_state >= 1.0 - 1e-10:
                break
        for k in range(skip_iterations, skip_iterations + iterations):
            r_val = x_val if pattern[k % seq_len] == 1 else y_val
            derivative = ti.abs(_ti_map_derivative(x_state, r_val, map_code))
            if derivative > 1e-10:
                lyapunov_sum += ti.log(derivative)
                valid_iterations += 1
            x_state = _ti_map_step(x_state, r_val, map_code)
            if x_state <= 1e-10 or x_state >= 1.0 - 1e-10:
                break
        out[i, j] = lyapunov_sum / valid_iterations if valid_iterations > 0 else -10.0


@ti.kernel
def _ti_bifurcation_kernel(out: ti.template(), r_min: ti.f32, r_max: ti.f32,
                           iterations: ti.i32, skip_iterations: ti.i32,
                           map_code: ti.i32, collect: ti.i32):
    n = out.shape[0]
    for p in range(n):
        r = r_min + (r_max - r_min) * p / max(n - 1, 1)
        x_state = 0.5
        for k in range(skip_iterations):
            x_state = _ti_map_step(x_state, r, map_code)
            if x_state <= 1e-10 or x_state >= 1.0 - 1e-10:
                break
        for k in range(iterations):
            x_state = _ti_map_step(x_state, r, map_code)
            if x_state <= 1e-10 or x_state >= 1.0 - 1e-10:
                break
            slot = k - (iterations - collect)
            if slot >= 0:
                out[p, slot] = x_state


class MayTaichiBackend:
    """GPU Lyapunov kernel; falls back to Numba if Taichi fails."""
    _out = None
    _shape = None
    _pattern = None
    _pattern_key = None
    _bif_out = None
    _bif_shape = None
    _warmed = False

    @classmethod
    def warm_up(cls):
        """Compile in the main thread; Taichi cannot compile in QThread."""
        if cls._warmed:
            return
        cls.calculate(2, 2, 2.5, 4.0, 2.5, 4.0,
                      np.array([1, 2], dtype=np.int32), 2, 1, 0)
        cls._warmed = True

    @classmethod
    def calculate(cls, width, height, x_min, x_max, y_min, y_max,
                  pattern_array, iterations, skip_iterations, map_code):
        if cls._out is None or cls._shape != (height, width):
            cls._out = ti.field(ti.f32, shape=(height, width))
            cls._shape = (height, width)
        key = tuple(int(x) for x in pattern_array)
        if cls._pattern is None or cls._pattern_key != key:
            cls._pattern = ti.field(ti.i32, shape=len(key))
            cls._pattern.from_numpy(np.asarray(key, dtype=np.int32))
            cls._pattern_key = key
        _ti_lyapunov_kernel(
            cls._out, cls._pattern,
            float(x_min), float(x_max), float(y_min), float(y_max),
            int(iterations), int(skip_iterations), int(map_code),
        )
        return cls._out.to_numpy()

    @classmethod
    def calculate_bifurcation(cls, r_min, r_max, num_points, iterations,
                              skip_iterations, map_code):
        collect = min(100, iterations)
        if (cls._bif_out is None or cls._bif_shape != (num_points, collect)):
            cls._bif_out = ti.field(ti.f32, shape=(num_points, collect))
            cls._bif_shape = (num_points, collect)
        cls._bif_out.fill(-1.0)
        _ti_bifurcation_kernel(
            cls._bif_out, float(r_min), float(r_max),
            int(iterations), int(skip_iterations), int(map_code), collect,
        )
        values = cls._bif_out.to_numpy()
        r_values = np.linspace(r_min, r_max, num_points)
        return [
            (float(r), [float(x) for x in row if x >= 0.0])
            for r, row in zip(r_values, values)
        ]


# =============================================================================
# 核心计算函数 (Numba 加速)
# =============================================================================

@jit(nopython=True, cache=True, inline="always")
def _map_step(x, r, map_code):
    if map_code == 1:          # sine
        return r * math.sin(math.pi * x) / 4.0
    if map_code == 2:          # tent
        x = 2.0 * r * x if x < 0.5 else 2.0 * r * (1.0 - x)
        return min(1.0, max(0.0, x))
    if map_code == 3:          # cubic
        return r * x * (1.0 - x * x)
    if map_code == 4:          # exponential
        return x * math.exp(r * (1.0 - x))
    return r * x * (1.0 - x)   # logistic


@jit(nopython=True, cache=True, inline="always")
def _map_derivative(x, r, map_code):
    if map_code == 1:
        return r * math.pi * math.cos(math.pi * x) / 4.0
    if map_code == 2:
        return 2.0 * r if x < 0.5 else -2.0 * r
    if map_code == 3:
        return r * (1.0 - 3.0 * x * x)
    if map_code == 4:
        return math.exp(r * (1.0 - x)) * (1.0 - r * x)
    return r * (1.0 - 2.0 * x)


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def calculate_lyapunov_fractal_cpu(width, height, x_min, x_max, y_min, y_max,
                                  pattern_array, iterations, skip_iterations,
                                  map_code):
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
                x_state = _map_step(x_state, r_val, map_code)
                if x_state <= 1e-10 or x_state >= 1.0 - 1e-10:
                    break

            # 计算 Lyapunov 指数
            for k in range(skip_iterations, total_iterations):
                idx = pattern_indices[k]
                r_val = x_val if pattern_array[idx] == 1 else y_val
                derivative = abs(_map_derivative(x_state, r_val, map_code))
                if derivative > 1e-10:
                    lyapunov_sum += math.log(derivative)  # Numba 支持 math.log
                    valid_iterations += 1
                x_state = _map_step(x_state, r_val, map_code)
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
                                            "tan": math.tan, "sqrt": math.sqrt, "exp": math.exp,
                                            "log": math.log, "abs": abs, "pi": math.pi})
            except Exception:
                x = 0.5
        return x

    def evaluate_derivative(self, x, r, n):
        try:
            return eval(self.derivative_str, {"x": x, "r": r, "np": np, "sin": math.sin, "cos": math.cos,
                                             "tan": math.tan, "sqrt": math.sqrt, "exp": math.exp,
                                             "log": math.log, "abs": abs, "pi": math.pi})
        except Exception:
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
        map_codes = {
            "logistic_map": 0, "sine_map": 1, "tent_map": 2,
            "cubic_map": 3, "exponential_map": 4,
        }
        map_code = map_codes.get(map_func.__name__)
        if use_cpu_parallel and custom_handler is None and map_code is not None:
            try:
                fractal = MayTaichiBackend.calculate(
                    width, height, x_min, x_max, y_min, y_max,
                    pattern_array, iterations, skip_iterations, map_code,
                )
                if progress_callback:
                    progress_callback(100)
                return fractal
            except Exception as e:
                print(f"Taichi 计算失败，尝试 Numba: {e}")

        if use_cpu_parallel and custom_handler is None and map_code is not None:
            try:
                print("使用CPU并行计算...")
                fractal = calculate_lyapunov_fractal_cpu(
                    width, height, x_min, x_max, y_min, y_max,
                    pattern_array, iterations, skip_iterations,
                    map_code,
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
    regionSelected = Signal(float, float, float, float)  # x0,x1,y0,y1
    zoomRequested = Signal(float, float, float)          # center x, center y, factor
    resetViewRequested = Signal()

    def __init__(self):
        super().__init__()
        self.fractal_data = None
        self.colormap = "bw"
        self.cached_image = None
        self.setMinimumSize(400, 400)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self._x_range = (2.5, 4.0)
        self._y_range = (2.5, 4.0)
        self._image_rect = QRectF()
        self._drag_start = None
        self._drag_current = None
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

    def set_ranges(self, x_range, y_range):
        self._x_range = tuple(float(v) for v in x_range)
        self._y_range = tuple(float(v) for v in y_range)

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
        self._image_rect = rect
        if self.cached_image is None or self.cached_image.size() != QSize(draw_w, draw_h):
            img = self._fractal_to_qimage()
            self.cached_image = img.scaled(draw_w, draw_h, Qt.KeepAspectRatio, Qt.FastTransformation)
        painter.drawImage(rect, self.cached_image)

        if self._drag_start is not None and self._drag_current is not None:
            selection = QRectF(self._drag_start, self._drag_current).normalized()
            painter.fillRect(selection, QColor(0, 120, 255, 45))
            painter.setPen(QPen(QColor(0, 90, 220), 1, Qt.DashLine))
            painter.drawRect(selection)

    def _screen_to_value(self, pos):
        rect = self._image_rect
        if rect.width() <= 0 or rect.height() <= 0:
            return None
        u = min(1.0, max(0.0, (pos.x() - rect.left()) / rect.width()))
        v = min(1.0, max(0.0, (pos.y() - rect.top()) / rect.height()))
        x = self._x_range[0] + u * (self._x_range[1] - self._x_range[0])
        # 图像第 0 行对应 y_min，因此顶部是 y_min
        y = self._y_range[0] + v * (self._y_range[1] - self._y_range[0])
        return x, y

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.fractal_data is not None:
            self._drag_start = event.position()
            self._drag_current = event.position()
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            self._drag_current = event.position()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_start is not None:
            start, end = self._drag_start, event.position()
            self._drag_start = None
            self._drag_current = None
            self.update()
            if abs(end.x() - start.x()) >= 8 and abs(end.y() - start.y()) >= 8:
                a = self._screen_to_value(start)
                b = self._screen_to_value(end)
                if a and b:
                    x0, x1 = sorted((a[0], b[0]))
                    y0, y1 = sorted((a[1], b[1]))
                    self.regionSelected.emit(x0, x1, y0, y1)
        elif event.button() == Qt.RightButton:
            self.resetViewRequested.emit()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.resetViewRequested.emit()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if self.fractal_data is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 ** (delta / 120.0)
        value = self._screen_to_value(event.position())
        if value is not None:
            self.zoomRequested.emit(value[0], value[1], factor)
        event.accept()

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
    regionSelected = Signal(float, float, float, float)  # r0,r1,x0,x1
    zoomRequested = Signal(float, float, float)          # r, x, factor
    resetViewRequested = Signal()

    def __init__(self):
        super().__init__()
        self.bifurcation_data = None
        self.cached_image = None
        self.setMinimumSize(600, 600)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self._r_range = (2.5, 4.0)
        self._x_range = (0.0, 1.0)
        self._plot_rect = QRectF()
        self._image_rect = QRectF()
        self._drag_start = None
        self._drag_current = None

    def set_bifurcation_data(self, data):
        self.bifurcation_data = data
        self.cached_image = None
        self.update()

    def set_ranges(self, r_range, x_range):
        self._r_range = tuple(float(v) for v in r_range)
        self._x_range = tuple(float(v) for v in x_range)
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
        if self._drag_start is not None and self._drag_current is not None:
            selection = QRectF(self._drag_start, self._drag_current).normalized()
            selection = selection.intersected(self._plot_rect)
            painter.fillRect(selection, QColor(0, 120, 255, 45))
            painter.setPen(QPen(QColor(0, 90, 220), 1, Qt.DashLine))
            painter.drawRect(selection)

    def _screen_to_value(self, pos):
        rect = self._plot_rect
        if rect.width() <= 0 or rect.height() <= 0:
            return None
        u = min(1.0, max(0.0, (pos.x() - rect.left()) / rect.width()))
        v = min(1.0, max(0.0, (pos.y() - rect.top()) / rect.height()))
        r = self._r_range[0] + u * (self._r_range[1] - self._r_range[0])
        x = self._x_range[1] - v * (self._x_range[1] - self._x_range[0])
        return r, x

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.bifurcation_data is not None:
            self._drag_start = event.position()
            self._drag_current = event.position()
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            self._drag_current = event.position()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_start is not None:
            start, end = self._drag_start, event.position()
            self._drag_start = None
            self._drag_current = None
            self.update()
            if abs(end.x() - start.x()) >= 8 and abs(end.y() - start.y()) >= 8:
                a = self._screen_to_value(start)
                b = self._screen_to_value(end)
                if a and b:
                    self.regionSelected.emit(min(a[0], b[0]), max(a[0], b[0]),
                                             min(a[1], b[1]), max(a[1], b[1]))
        elif event.button() == Qt.RightButton:
            self.resetViewRequested.emit()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.resetViewRequested.emit()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if self.bifurcation_data is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        value = self._screen_to_value(event.position())
        if value is not None:
            self.zoomRequested.emit(value[0], value[1], 1.25 ** (delta / 120.0))
        event.accept()

    def render_bifurcation_image(self):
        image = QImage(self.size(), QImage.Format_RGB32)
        image.fill(QColor(255, 255, 255))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, False)
        width, height = self.width(), self.height()
        margin = 40
        self._plot_rect = QRectF(margin, margin, width - 2 * margin, height - 2 * margin)
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawLine(margin, height - margin, width - margin, height - margin)
        painter.drawLine(margin, margin, margin, height - margin)
        painter.drawText(margin // 2, height // 2, "x")
        painter.drawText(width // 2, height - margin // 2, "r")
        if not self.bifurcation_data:
            painter.end()
            return image
        r_min, r_max = self._r_range
        x_min, x_max = self._x_range
        if r_max <= r_min:
            r_max = r_min + 1e-9
        if x_max <= x_min:
            x_max = x_min + 1e-9
        plot_w = self._plot_rect.width()
        plot_h = self._plot_rect.height()
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        for r, x_vals in self.bifurcation_data:
            if not (r_min <= r <= r_max):
                continue
            screen_r = int(margin + (r - r_min) / (r_max - r_min) * plot_w)
            for x in x_vals:
                if not (x_min <= x <= x_max):
                    continue
                screen_y = int(margin + (x_max - x) / (x_max - x_min) * plot_h)
                if 0 <= screen_r < width and 0 <= screen_y < height:
                    painter.drawPoint(screen_r, screen_y)
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
        self.setWindowTitle("May · 逻辑映射与分岔")
        # 640 canvas + tab bar + margins: exact fit, no white strip/crop
        self.setMinimumSize(1004, 689)
        self.setGeometry(100, 100, 1004, 689)
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
        self._view_history = []
        self._zoom_in_progress = False
        self._home_x_range = (2.5, 4.0)
        self._home_y_range = (2.5, 4.0)
        self._bif_history = []
        self._bif_zoom_in_progress = False
        self._bif_r_range = (2.5, 4.0)
        self._bif_x_range = (0.0, 1.0)
        self._bif_home_r = (2.5, 4.0)
        self._bif_home_x = (0.0, 1.0)
        try:
            MayTaichiBackend.warm_up()
        except Exception as exc:
            print(f"Taichi 预热失败，将使用其他后端: {exc}")
        self.init_ui()
        self.calculate_fractal()


    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        panel = self.create_control_panel()
        panel.setFixedWidth(330)
        root.addWidget(panel, 0)
        root.addWidget(self.create_display_panel(), 1)

    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)

        # 映射函数
        func_group = QGroupBox("映射函数")
        func_layout = QVBoxLayout(func_group)
        func_layout.setContentsMargins(8, 6, 8, 6)
        func_layout.setSpacing(5)
        fl = QHBoxLayout()
        fl.addWidget(QLabel("映射:"))
        self.function_combo = QComboBox()
        self.function_combo.addItems(list(self.map_functions.keys()))
        self.function_combo.currentTextChanged.connect(self.select_function)
        fl.addWidget(self.function_combo, 1)
        func_layout.addLayout(fl)
        self.formula_label = QLabel()
        self.formula_label.setWordWrap(True)
        self.formula_label.setStyleSheet("color:#555; font-size:11px;")
        self.formula_label.setMaximumHeight(38)
        self.formula_label.setText(MAP_FORMULAS.get(self.current_map, ""))
        func_layout.addWidget(self.formula_label)

        self.custom_func_edit = QLineEdit("r * x * (1 - x)")
        self.custom_deriv_edit = QLineEdit("r * (1 - 2*x)")
        self.custom_func_edit.setVisible(False)
        self.custom_deriv_edit.setVisible(False)
        self.custom_edit_btn = QPushButton("编辑自定义公式…")
        self.custom_edit_btn.clicked.connect(self.edit_custom_formula)
        self.custom_edit_btn.setVisible(False)
        func_layout.addWidget(self.custom_edit_btn)
        layout.addWidget(func_group)

        # 序列
        seq_group = QGroupBox("序列")
        seq_layout = QVBoxLayout(seq_group)
        seq_layout.setContentsMargins(8, 6, 8, 6)
        seq_layout.setSpacing(5)
        sl = QHBoxLayout()
        sl.addWidget(QLabel("AB 序列:"))
        self.sequence_edit = QLineEdit(self.sequence)
        self.sequence_edit.setPlaceholderText("AB / AAB / ABBA")
        sl.addWidget(self.sequence_edit, 1)
        seq_layout.addLayout(sl)
        el = QHBoxLayout()
        el.addWidget(QLabel("示例:"))
        self.examples_combo = QComboBox()
        self.examples_combo.addItems(["AB (标准)", "AAB", "ABB", "ABBA", "AAAB", "ABAB"])
        self.examples_combo.currentTextChanged.connect(self.select_example)
        el.addWidget(self.examples_combo, 1)
        seq_layout.addLayout(el)
        layout.addWidget(seq_group)

        # 计算参数
        params_group = QGroupBox("计算参数")
        params_layout = QGridLayout(params_group)
        params_layout.setContentsMargins(8, 6, 8, 6)
        params_layout.setSpacing(5)
        params_layout.addWidget(QLabel("分辨率:"), 0, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["100x100", "200x200", "400x400", "600x600", "800x800", "1000x1000", "1200x1200", "1500x1500", "2000x2000"])
        self.resolution_combo.setCurrentText("800x800")
        self.resolution_combo.currentTextChanged.connect(self.update_resolution)
        params_layout.addWidget(self.resolution_combo, 0, 1)
        params_layout.addWidget(QLabel("迭代:"), 1, 0)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(50, 5000)
        self.iterations_spin.setValue(self.iterations)
        self.iterations_spin.setSuffix(" 次")
        params_layout.addWidget(self.iterations_spin, 1, 1)
        params_layout.addWidget(QLabel("跳过:"), 2, 0)
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(10, 2000)
        self.skip_spin.setValue(self.skip_iterations)
        self.skip_spin.setSuffix(" 次")
        params_layout.addWidget(self.skip_spin, 2, 1)
        layout.addWidget(params_group)

        # 范围
        range_group = QGroupBox("参数范围")
        range_layout = QGridLayout(range_group)
        range_layout.setContentsMargins(8, 6, 8, 6)
        range_layout.setSpacing(5)
        range_layout.setColumnStretch(1, 1)
        range_layout.setColumnStretch(3, 1)
        range_layout.setColumnMinimumWidth(2, 26)

        def centered_arrow():
            label = QLabel("→")
            label.setAlignment(Qt.AlignCenter)
            label.setFixedWidth(26)
            return label

        range_layout.addWidget(QLabel("X:"), 0, 0)
        self.x_min_edit = QLineEdit(f"{self.x_range[0]:.2f}")
        self.x_min_edit.setMinimumWidth(92)
        range_layout.addWidget(self.x_min_edit, 0, 1)
        range_layout.addWidget(centered_arrow(), 0, 2)
        self.x_max_edit = QLineEdit(f"{self.x_range[1]:.2f}")
        self.x_max_edit.setMinimumWidth(92)
        range_layout.addWidget(self.x_max_edit, 0, 3)
        range_layout.addWidget(QLabel("Y:"), 1, 0)
        self.y_min_edit = QLineEdit(f"{self.y_range[0]:.2f}")
        self.y_min_edit.setMinimumWidth(92)
        range_layout.addWidget(self.y_min_edit, 1, 1)
        range_layout.addWidget(centered_arrow(), 1, 2)
        self.y_max_edit = QLineEdit(f"{self.y_range[1]:.2f}")
        self.y_max_edit.setMinimumWidth(92)
        range_layout.addWidget(self.y_max_edit, 1, 3)
        range_layout.addWidget(QLabel("分叉图:"), 2, 0)
        self.bifurcation_min_edit = QLineEdit(f"{self.bifurcation_range[0]:.2f}")
        self.bifurcation_min_edit.setMinimumWidth(92)
        range_layout.addWidget(self.bifurcation_min_edit, 2, 1)
        range_layout.addWidget(centered_arrow(), 2, 2)
        self.bifurcation_max_edit = QLineEdit(f"{self.bifurcation_range[1]:.2f}")
        self.bifurcation_max_edit.setMinimumWidth(92)
        range_layout.addWidget(self.bifurcation_max_edit, 2, 3)
        range_layout.addWidget(QLabel("预设:"), 3, 0)
        self.range_combo = QComboBox()
        self.range_combo.addItems(["标准 (2.5-4.0)", "经典 (3.0-4.0)", "放大中心 (3.4-3.9)", "放大左上 (2.5-3.5)", "完整范围 (0-4.0)", "自定义"])
        self.range_combo.currentTextChanged.connect(self.select_range_preset)
        range_layout.addWidget(self.range_combo, 3, 1, 1, 3)
        view_row = QHBoxLayout()
        view_row.setSpacing(6)
        self.undo_view_btn = QPushButton("上一步")
        self.undo_view_btn.clicked.connect(self.undo_current_view)
        self.reset_view_btn = QPushButton("重置")
        self.reset_view_btn.clicked.connect(self.reset_current_view)
        view_row.addWidget(self.undo_view_btn, 1)
        view_row.addWidget(self.reset_view_btn, 1)
        range_layout.addLayout(view_row, 4, 0, 1, 4)
        layout.addWidget(range_group)

        # 颜色
        color_group = QGroupBox("颜色")
        color_layout = QHBoxLayout(color_group)
        color_layout.setContentsMargins(8, 6, 8, 6)
        color_layout.addWidget(QLabel("映射:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["bw", "viridis", "hot", "cool", "rainbow", "spectral"])
        self.colormap_combo.setCurrentText("bw")
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        color_layout.addWidget(self.colormap_combo, 1)
        layout.addWidget(color_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self.calculate_btn = QPushButton("Lyapunov 分形")
        self.calculate_btn.clicked.connect(self.calculate_fractal)
        self.calculate_bifurcation_btn = QPushButton("分叉图")
        self.calculate_bifurcation_btn.clicked.connect(self.calculate_bifurcation)
        action_row.addWidget(self.calculate_btn, 1)
        action_row.addWidget(self.calculate_bifurcation_btn, 1)
        layout.addLayout(action_row)

        self.info_label = QLabel("准备就绪")
        self.info_label.setWordWrap(True)
        self.info_label.setMaximumHeight(32)
        layout.addWidget(self.info_label)
        layout.addStretch(1)
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
        self.fractal_widget.setFixedSize(640, 640)
        self.fractal_widget.regionSelected.connect(self.zoom_fractal)
        self.fractal_widget.zoomRequested.connect(self.wheel_zoom_fractal)
        self.fractal_widget.resetViewRequested.connect(self.reset_fractal_view)
        fractal_layout.addWidget(self.fractal_widget, 0, Qt.AlignHCenter | Qt.AlignTop)
        self.tab_widget.addTab(fractal_tab, "Lyapunov 分形")

        bifurcation_tab = QWidget()
        bifurcation_layout = QVBoxLayout(bifurcation_tab)
        bifurcation_layout.setContentsMargins(0, 0, 0, 0)
        self.bifurcation_widget = BifurcationWidget()
        self.bifurcation_widget.setFixedSize(640, 640)
        self.bifurcation_widget.regionSelected.connect(self.zoom_bifurcation)
        self.bifurcation_widget.zoomRequested.connect(self.wheel_zoom_bifurcation)
        self.bifurcation_widget.resetViewRequested.connect(self.reset_bifurcation_view)
        bifurcation_layout.addWidget(self.bifurcation_widget, 0, Qt.AlignHCenter | Qt.AlignTop)
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
        self.formula_label.setText(MAP_FORMULAS.get(name, ""))
        self._view_history.clear()
        self.custom_edit_btn.setVisible(name == "自定义映射")
        self.info_label.setText(f"已选择映射函数: {name}")

    def edit_custom_formula(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("自定义映射公式")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(QLabel("f(x)"))
        func_edit = QLineEdit(self.custom_func_edit.text())
        func_edit.setMinimumHeight(32)
        layout.addWidget(func_edit)
        layout.addWidget(QLabel("f′(x)"))
        deriv_edit = QLineEdit(self.custom_deriv_edit.text())
        deriv_edit.setMinimumHeight(32)
        layout.addWidget(deriv_edit)
        hint = QLabel("可用：r, x, sin, cos, tan, sqrt, exp, log, abs, pi, np")
        hint.setStyleSheet("color:#666; font-size:11px;")
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(480, dialog.sizeHint().height())
        if dialog.exec() == QDialog.Accepted:
            self.custom_func_edit.setText(func_edit.text())
            self.custom_deriv_edit.setText(deriv_edit.text())
            self.apply_custom_function()

    def apply_custom_function(self):
        fs = self.custom_func_edit.text().strip()
        ds = self.custom_deriv_edit.text().strip()
        if not fs or not ds:
            QMessageBox.warning(self, "错误", "请输入自定义函数和导数")
            return
        try:
            x, r = 0.5, 2.0
            env = {"x": x, "r": r, "np": np, "sin": math.sin, "cos": math.cos,
                   "tan": math.tan, "sqrt": math.sqrt, "exp": math.exp,
                   "log": math.log, "abs": abs, "pi": math.pi}
            eval(fs, env)
            eval(ds, env)
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
            self.fractal_widget.set_ranges(self.x_range, self.y_range)
            if not self._zoom_in_progress:
                self._view_history.clear()
                self._home_x_range = self.x_range
                self._home_y_range = self.y_range
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的数值")
            return

        self.calculate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        custom = self.custom_handler if self.current_map == "自定义映射" else None
        use_parallel = self.use_cpu_parallel and self.current_map != "自定义映射"

        # Taichi kernel must run on the main thread.  At 800² it is only
        # a few ms, so run built-in maps synchronously and keep the UI
        # responsive even for rectangle-zoom recalculation.
        map_codes = {
            "logistic_map": 0, "sine_map": 1, "tent_map": 2,
            "cubic_map": 3, "exponential_map": 4,
        }
        map_code = map_codes.get(self.current_map_func.__name__)
        if custom is None and map_code is not None:
            self.info_label.setText(
                f"正在计算Lyapunov分形 - 函数: {self.current_map}, 序列: {self.sequence}, 模式: GPU/Taichi"
            )
            try:
                pattern = np.array([1 if c == 'A' else 2 for c in self.sequence], dtype=np.int32)
                fractal = MayTaichiBackend.calculate(
                    self.width, self.height, self.x_range[0], self.x_range[1],
                    self.y_range[0], self.y_range[1], pattern,
                    self.iterations, self.skip_iterations, map_code,
                )
                self.display_fractal(fractal)
                return
            except Exception as exc:
                print(f"Taichi 计算失败，回退 worker: {exc}")

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

    def zoom_fractal(self, x0, x1, y0, y1):
        """Rectangle zoom: use the selected box as the new parameter range."""
        if abs(x1 - x0) < 1e-9 or abs(y1 - y0) < 1e-9:
            return
        # Keep x/y spans equal so one data cell stays square on screen.
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        span = max(x1 - x0, y1 - y0)
        x0, x1 = cx - span / 2.0, cx + span / 2.0
        y0, y1 = cy - span / 2.0, cy + span / 2.0
        self._view_history.append((self.x_range, self.y_range))
        self._zoom_in_progress = True
        try:
            self.x_min_edit.setText(f"{x0:.6f}")
            self.x_max_edit.setText(f"{x1:.6f}")
            self.y_min_edit.setText(f"{y0:.6f}")
            self.y_max_edit.setText(f"{y1:.6f}")
            self.calculate_fractal()
        finally:
            self._zoom_in_progress = False

    def wheel_zoom_fractal(self, cx, cy, factor):
        if factor <= 0:
            return
        factor = min(8.0, max(0.125, factor))
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        width = (x1 - x0) / factor
        height = (y1 - y0) / factor
        u = (cx - x0) / (x1 - x0)
        v = (cy - y0) / (y1 - y0)
        new_x0 = cx - u * width
        new_x1 = new_x0 + width
        new_y0 = cy - v * height
        new_y1 = new_y0 + height
        self._view_history.append((self.x_range, self.y_range))
        self._zoom_in_progress = True
        try:
            self.x_min_edit.setText(f"{new_x0:.6f}")
            self.x_max_edit.setText(f"{new_x1:.6f}")
            self.y_min_edit.setText(f"{new_y0:.6f}")
            self.y_max_edit.setText(f"{new_y1:.6f}")
            self.calculate_fractal()
        finally:
            self._zoom_in_progress = False

    def undo_current_view(self):
        if self.tab_widget.currentIndex() == 1:
            self.undo_bifurcation_view()
        else:
            self.undo_fractal_view()

    def reset_current_view(self):
        if self.tab_widget.currentIndex() == 1:
            self.reset_bifurcation_view()
        else:
            self.reset_fractal_view()

    def undo_bifurcation_view(self):
        if not self._bif_history:
            return
        r_range, x_range = self._bif_history.pop()
        self._bif_zoom_in_progress = True
        try:
            self._bif_x_range = x_range
            self.bifurcation_min_edit.setText(f"{r_range[0]:.6f}")
            self.bifurcation_max_edit.setText(f"{r_range[1]:.6f}")
            self.calculate_bifurcation()
        finally:
            self._bif_zoom_in_progress = False

    def undo_fractal_view(self):
        if not self._view_history:
            return
        x_range, y_range = self._view_history.pop()
        self._zoom_in_progress = True
        try:
            self.x_min_edit.setText(f"{x_range[0]:.6f}")
            self.x_max_edit.setText(f"{x_range[1]:.6f}")
            self.y_min_edit.setText(f"{y_range[0]:.6f}")
            self.y_max_edit.setText(f"{y_range[1]:.6f}")
            self.calculate_fractal()
        finally:
            self._zoom_in_progress = False

    def reset_fractal_view(self):
        self._view_history.clear()
        self._zoom_in_progress = True
        try:
            self.x_min_edit.setText(f"{self._home_x_range[0]:.6f}")
            self.x_max_edit.setText(f"{self._home_x_range[1]:.6f}")
            self.y_min_edit.setText(f"{self._home_y_range[0]:.6f}")
            self.y_max_edit.setText(f"{self._home_y_range[1]:.6f}")
            self.calculate_fractal()
        finally:
            self._zoom_in_progress = False

    def zoom_bifurcation(self, r0, r1, x0, x1):
        if abs(r1 - r0) < 1e-9 or abs(x1 - x0) < 1e-9:
            return
        self._bif_history.append((self._bif_r_range, self._bif_x_range))
        self._bif_zoom_in_progress = True
        try:
            self._bif_x_range = (x0, x1)
            self.bifurcation_min_edit.setText(f"{r0:.6f}")
            self.bifurcation_max_edit.setText(f"{r1:.6f}")
            self.calculate_bifurcation()
        finally:
            self._bif_zoom_in_progress = False

    def wheel_zoom_bifurcation(self, cr, cx, factor):
        if factor <= 0:
            return
        factor = min(8.0, max(0.125, factor))
        r0, r1 = self._bif_r_range
        x0, x1 = self._bif_x_range
        width = (r1 - r0) / factor
        height = (x1 - x0) / factor
        u = (cr - r0) / (r1 - r0)
        v = (cx - x0) / (x1 - x0)
        new_r0 = cr - u * width
        new_r1 = new_r0 + width
        new_x0 = cx - v * height
        new_x1 = new_x0 + height
        self._bif_history.append((self._bif_r_range, self._bif_x_range))
        self._bif_zoom_in_progress = True
        try:
            self._bif_x_range = (new_x0, new_x1)
            self.bifurcation_min_edit.setText(f"{new_r0:.6f}")
            self.bifurcation_max_edit.setText(f"{new_r1:.6f}")
            self.calculate_bifurcation()
        finally:
            self._bif_zoom_in_progress = False

    def reset_bifurcation_view(self):
        self._bif_history.clear()
        self._bif_zoom_in_progress = True
        try:
            self._bif_x_range = self._bif_home_x
            self.bifurcation_min_edit.setText(f"{self._bif_home_r[0]:.6f}")
            self.bifurcation_max_edit.setText(f"{self._bif_home_r[1]:.6f}")
            self.calculate_bifurcation()
        finally:
            self._bif_zoom_in_progress = False

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
        if not self._bif_zoom_in_progress:
            self._bif_x_range = (0.0, 1.0)
            self._bif_history.clear()
            self._bif_home_r = (r_min, r_max)
            self._bif_home_x = self._bif_x_range
        self._bif_r_range = (r_min, r_max)
        self.bifurcation_widget.set_ranges((r_min, r_max), self._bif_x_range)
        self.info_label.setText(f"正在计算分叉图 - 函数: {self.current_map}")

        custom = self.custom_handler if self.current_map == "自定义映射" else None
        map_codes = {
            "logistic_map": 0, "sine_map": 1, "tent_map": 2,
            "cubic_map": 3, "exponential_map": 4,
        }
        map_code = map_codes.get(self.current_map_func.__name__)
        if custom is None and map_code is not None:
            try:
                data = MayTaichiBackend.calculate_bifurcation(
                    r_min, r_max, 800,
                    self.iterations_spin.value(), self.skip_spin.value(),
                    map_code,
                )
                self.display_bifurcation(data)
                return
            except Exception as exc:
                print(f"Taichi 分叉图失败，回退 CPU: {exc}")

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
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)

    def closeEvent(self, event):
        threads = (self.calculation_thread, self.bifurcation_thread)
        for thread in threads:
            if thread and thread.isRunning():
                thread.cancel()
        for thread in threads:
            if thread and thread.isRunning():
                if not thread.wait(3000):
                    thread.terminate()
                    thread.wait(1000)
        super().closeEvent(event)

    def reset_bifurcation_ui(self):
        self.calculate_bifurcation_btn.setEnabled(True)


if __name__ == "__main__":
    raise SystemExit(run_app(LyapunovFractalApp))