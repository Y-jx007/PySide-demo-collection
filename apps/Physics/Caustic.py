import sys
import numpy as np
import taichi_forge as ti
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSlider, QLabel, QGroupBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush

# -------------------- 初始化 Taichi --------------------
ti.init(arch=ti.gpu, debug=False)

# 图像尺寸 2:1
W, H = 800, 400

# 三通道图像 [H, W, 3]
img_field = ti.field(dtype=ti.f32, shape=(H, W, 3))

PI = 3.14159265

# 基础波长 (红、绿、蓝)
LAMBDA_R = 0.70
LAMBDA_G = 0.55
LAMBDA_B = 0.45

# 世界坐标映射：图像水平映射 x ∈ [-1.5, 2.5]，垂直 y ∈ [-1.0, 1.0]
WORLD_X_MIN = -1.5
WORLD_X_MAX =  2.5
WORLD_Y_MIN = -1.0
WORLD_Y_MAX =  1.0
WORLD_W = WORLD_X_MAX - WORLD_X_MIN
WORLD_H = WORLD_Y_MAX - WORLD_Y_MIN

# 圆形焦散物体参数：中心在右半部 (1.0, 0)，半径 1.0
CIRCLE_CX = 1.0
CIRCLE_CY = 0.0
CIRCLE_R  = 1.0

@ti.kernel
def render_caustics(
    light_x: ti.f32, light_y: ti.f32,
    wavelength_scale: ti.f32,
    n_samples: ti.i32
):
    """
    蒙特卡洛路径积分：对圆内每个像素，在左半圆弧上均匀采样反射点，
    累加红绿蓝三通道复振幅，得到彩色焦散。
    """
    for i, j in ti.ndrange(H, W):
        # 像素 -> 世界坐标
        x = ti.cast(j, ti.f32) / ti.cast(W, ti.f32) * WORLD_W + WORLD_X_MIN
        y = ti.cast(i, ti.f32) / ti.cast(H, ti.f32) * WORLD_H + WORLD_Y_MIN

        dx_c = x - CIRCLE_CX
        dy_c = y - CIRCLE_CY
        if dx_c * dx_c + dy_c * dy_c >= CIRCLE_R * CIRCLE_R:
            img_field[i, j, 0] = 0.0
            img_field[i, j, 1] = 0.0
            img_field[i, j, 2] = 0.0
            continue

        # 复振幅累加器
        rx, ry = 0.0, 0.0
        gx, gy = 0.0, 0.0
        bx, by = 0.0, 0.0

        lam_r = LAMBDA_R * wavelength_scale
        lam_g = LAMBDA_G * wavelength_scale
        lam_b = LAMBDA_B * wavelength_scale

        ti.loop_config(serialize=True)  # 串行累加，避免原子操作
        for k in range(n_samples):
            # 左半圆弧：θ ∈ [-π/2, π/2]
            theta = -0.5 * PI + PI * (ti.cast(k, ti.f32) + ti.random()) / ti.cast(n_samples, ti.f32)
            ct = ti.cos(theta)
            st = ti.sin(theta)

            # 反射点世界坐标
            Px = CIRCLE_CX + CIRCLE_R * ct
            Py = CIRCLE_CY + CIRCLE_R * st

            # 光源到反射点的距离和方向
            dx1 = Px - light_x
            dy1 = Py - light_y
            d1 = ti.sqrt(dx1 * dx1 + dy1 * dy1)
            if d1 < 1e-6:
                continue
            in_dir_x = dx1 / d1
            in_dir_y = dy1 / d1

            # 内法线（指向圆心）：(-ct, -st)
            Nx = -ct
            Ny = -st
            cos_inc = in_dir_x * Nx + in_dir_y * Ny   # 入射角余弦
            if cos_inc <= 0.0:   # 背面照射，无贡献
                continue

            # 反射点到接收点的距离
            dx2 = x - Px
            dy2 = y - Py
            d2 = ti.sqrt(dx2 * dx2 + dy2 * dy2)
            if d2 < 1e-6:
                continue

            # 几何因子：入射余弦 / d2（与原始 GLSL 的 cosθ/d 对应）
            form = cos_inc / d2

            # 总光程
            path = d1 + d2

            # 三波长相位
            s_r = 2.0 * PI / lam_r * path
            s_g = 2.0 * PI / lam_g * path
            s_b = 2.0 * PI / lam_b * path

            rx += ti.cos(s_r) * form
            ry += ti.sin(s_r) * form
            gx += ti.cos(s_g) * form
            gy += ti.sin(s_g) * form
            bx += ti.cos(s_b) * form
            by += ti.sin(s_b) * form

        gain = 2.0
        norm = 1.0 / ti.cast(n_samples, ti.f32)
        ir = ti.sqrt(rx * rx + ry * ry) * norm * gain
        ig = ti.sqrt(gx * gx + gy * gy) * norm * gain
        ib = ti.sqrt(bx * bx + by * by) * norm * gain

        img_field[i, j, 0] = ir
        img_field[i, j, 1] = ig
        img_field[i, j, 2] = ib


# -------------------- 主窗口 --------------------
class CausticsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("焦散可视化 - 点光源 · 光线追踪")
        self.setFixedSize(W + 260, H + 20)

        # 可调参数默认值
        self.light_x = -0.5
        self.light_y = 0.0
        self.wavelength_scale = 0.03
        self.n_samples = 1000

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 左侧控制面板
        panel = QVBoxLayout()

        # ---- 光源位置 ----
        group_light = QGroupBox("点光源位置")
        light_layout = QVBoxLayout()
        light_layout.addWidget(QLabel("X 坐标"))
        self.slider_lx = QSlider(Qt.Horizontal)
        self.slider_lx.setRange(-150, 150)
        self.slider_lx.setValue(int(self.light_x * 100))
        self.slider_lx.valueChanged.connect(self.on_light_x_changed)
        light_layout.addWidget(self.slider_lx)
        self.label_lx = QLabel(f"{self.light_x:.2f}")
        light_layout.addWidget(self.label_lx)

        light_layout.addWidget(QLabel("Y 坐标"))
        self.slider_ly = QSlider(Qt.Horizontal)
        self.slider_ly.setRange(-100, 100)
        self.slider_ly.setValue(int(self.light_y * 100))
        self.slider_ly.valueChanged.connect(self.on_light_y_changed)
        light_layout.addWidget(self.slider_ly)
        self.label_ly = QLabel(f"{self.light_y:.2f}")
        light_layout.addWidget(self.label_ly)
        group_light.setLayout(light_layout)
        panel.addWidget(group_light)

        # ---- 波长缩放 ----
        group_wl = QGroupBox("波长缩放")
        wl_layout = QVBoxLayout()
        self.slider_wl = QSlider(Qt.Horizontal)
        self.slider_wl.setRange(1, 100)
        self.slider_wl.setValue(int(self.wavelength_scale * 1000))
        self.slider_wl.valueChanged.connect(self.on_wavelength_changed)
        wl_layout.addWidget(self.slider_wl)
        self.label_wl = QLabel(f"{self.wavelength_scale:.3f}")
        wl_layout.addWidget(self.label_wl)
        group_wl.setLayout(wl_layout)
        panel.addWidget(group_wl)

        # ---- 采样数 ----
        group_sp = QGroupBox("采样数")
        sp_layout = QVBoxLayout()
        self.slider_sp = QSlider(Qt.Horizontal)
        self.slider_sp.setRange(50, 2000)
        self.slider_sp.setValue(self.n_samples)
        self.slider_sp.valueChanged.connect(self.on_samples_changed)
        sp_layout.addWidget(self.slider_sp)
        self.label_sp = QLabel(str(self.n_samples))
        sp_layout.addWidget(self.label_sp)
        group_sp.setLayout(sp_layout)
        panel.addWidget(group_sp)

        panel.addStretch()
        main_layout.addLayout(panel)

        # 右侧图像显示
        self.image_label = QLabel()
        self.image_label.setFixedSize(W, H)
        main_layout.addWidget(self.image_label)

        # 定时刷新
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(33)  # ~30 FPS

        self.update_display()

    def on_light_x_changed(self, val):
        self.light_x = val / 100.0
        self.label_lx.setText(f"{self.light_x:.2f}")

    def on_light_y_changed(self, val):
        self.light_y = val / 100.0
        self.label_ly.setText(f"{self.light_y:.2f}")

    def on_wavelength_changed(self, val):
        self.wavelength_scale = val / 1000.0
        self.label_wl.setText(f"{self.wavelength_scale:.3f}")

    def on_samples_changed(self, val):
        self.n_samples = val
        self.label_sp.setText(str(val))

    def update_display(self):
        # GPU 计算焦散
        render_caustics(self.light_x, self.light_y, self.wavelength_scale, self.n_samples)
        ti.sync()

        # 转换为 QImage
        img_np = img_field.to_numpy()
        img_np = np.clip(img_np, 0.0, 1.0) * 255.0
        img_np = img_np.astype(np.uint8)
        qimg = QImage(img_np.data, W, H, W * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # 在图像上绘制点光源标记
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # 世界坐标 -> 像素坐标
        px = int((self.light_x - WORLD_X_MIN) / WORLD_W * W)
        py = int((self.light_y - WORLD_Y_MIN) / WORLD_H * H)
        # 限制在画面内
        px = max(0, min(W-1, px))
        py = max(0, min(H-1, py))

        # 绘制光晕
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 200, 80))
        painter.drawEllipse(px - 8, py - 8, 16, 16)
        painter.setBrush(QColor(255, 255, 255, 200))
        painter.drawEllipse(px - 3, py - 3, 6, 6)
        painter.end()

        self.image_label.setPixmap(pixmap)


# -------------------- 启动 --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CausticsWindow()
    window.show()
    sys.exit(app.exec())