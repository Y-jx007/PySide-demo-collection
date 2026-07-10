import numpy as np
import taichi_forge as ti
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QImage, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QGroupBox, QFileDialog, QMessageBox
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from abc import ABC, abstractmethod

def setup_gl_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(fmt)

class TextureGLWidget(QOpenGLWidget):
    """通用的 OpenGL 纹理显示组件，接受一个返回 (H,W,3) uint8 图像的回调"""
    def __init__(self, image_provider, parent=None):
        super().__init__(parent)
        self.image_provider = image_provider
        self.texture_id = None
        self.tex_size = (0, 0)
        self.setMinimumSize(600, 600)

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)

    def paintGL(self):
        img = self.image_provider()
        h, w, _ = img.shape
        glClear(GL_COLOR_BUFFER_BIT)
        ratio = self.devicePixelRatio()
        w_view = int(self.width() * ratio)
        h_view = int(self.height() * ratio)
        glViewport(0, 0, w_view, h_view)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, w, 0, h, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        if (w, h) != self.tex_size:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
            self.tex_size = (w, h)
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, img)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glColor4f(1,1,1,1)
        glBegin(GL_QUADS)
        glTexCoord2f(0,0); glVertex2f(0,0)
        glTexCoord2f(1,0); glVertex2f(w,0)
        glTexCoord2f(1,1); glVertex2f(w,h)
        glTexCoord2f(0,1); glVertex2f(0,h)
        glEnd()

class SimulationBase(ABC):
    """反应扩散模拟抽象基类"""
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reset(self, pattern='default'):
        pass

    @abstractmethod
    def get_image(self, mode) -> np.ndarray:
        """返回 (H, W, 3) uint8 RGB 图像"""
        pass

class SimulationViewer(QWidget):
    """
    通用模拟查看器：
      - 左侧：param_panel（可插入自定义参数控件）
      - 右侧：OpenGL 渲染
      - 底部：播放/暂停、步数、保存、显示模式切换
    """
    def __init__(self, simulation: SimulationBase, parent=None):
        super().__init__(parent)
        self.sim = simulation
        self.is_playing = True
        self.steps_per_frame = 20
        self.display_mode = ''

        self.gl_widget = TextureGLWidget(image_provider=lambda: self.sim.get_image(self.display_mode))
        self.gl_widget.setFixedSize(600, 600)

        main_layout = QHBoxLayout()
        # 左侧参数区域
        self.param_panel = QVBoxLayout()
        param_widget = QWidget()
        param_widget.setLayout(self.param_panel)
        param_widget.setFixedWidth(300)
        main_layout.addWidget(param_widget)
        main_layout.addWidget(self.gl_widget)
        self.setLayout(main_layout)

        self._add_common_controls()

        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(30)

    def _add_common_controls(self):
        group = QGroupBox("控制")
        layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("显示:"))
        self.mode_combo = QComboBox()
        mode_layout.addWidget(self.mode_combo)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        layout.addLayout(mode_layout)

        btn_layout = QHBoxLayout()
        self.play_btn = QPushButton("暂停")
        self.play_btn.clicked.connect(self.toggle_play)
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_sim)
        btn_layout.addWidget(self.play_btn)
        btn_layout.addWidget(self.reset_btn)
        layout.addLayout(btn_layout)

        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("每帧步数:"))
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setRange(1, 100)
        self.steps_slider.setValue(20)
        self.steps_slider.valueChanged.connect(self._on_steps_changed)
        self.steps_label = QLabel("20")
        steps_layout.addWidget(self.steps_slider)
        steps_layout.addWidget(self.steps_label)
        layout.addLayout(steps_layout)

        self.save_btn = QPushButton("保存图像")
        self.save_btn.clicked.connect(self.save_image)
        layout.addWidget(self.save_btn)

        group.setLayout(layout)
        self.param_panel.addWidget(group)

    def set_display_modes(self, modes: list):
        self.mode_combo.clear()
        self.mode_combo.addItems(modes)
        if modes:
            self.display_mode = modes[0]

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_btn.setText("暂停" if self.is_playing else "播放")

    def reset_sim(self):
        self.sim.reset('default')
        self.gl_widget.update()

    def _on_timer(self):
        if self.is_playing:
            for _ in range(self.steps_per_frame):
                self.sim.step()
            self.gl_widget.update()

    def _on_steps_changed(self, val):
        self.steps_per_frame = val
        self.steps_label.setText(str(val))

    def _on_mode_changed(self, text):
        self.display_mode = text
        self.gl_widget.update()

    def save_image(self):
        img = self.sim.get_image(self.display_mode)
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3*w, QImage.Format_RGB888)
        path, _ = QFileDialog.getSaveFileName(self, "保存图像", "pattern.png", "PNG (*.png)")
        if path and qimg.save(path):
            QMessageBox.information(self, "保存成功", f"已保存至:\n{path}")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)