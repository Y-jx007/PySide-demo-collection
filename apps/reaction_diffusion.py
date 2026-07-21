import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QGroupBox, QFileDialog, QMessageBox,
    QLineEdit, QComboBox, QSizePolicy
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from abc import ABC, abstractmethod


# ---------- OpenGL 设置 ----------
def setup_gl_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(fmt)


# ---------- OpenGL 纹理显示组件 ----------
class TextureGLWidget(QOpenGLWidget):
    """通用的 OpenGL 纹理显示组件"""
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

        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(w, 0)
        glTexCoord2f(1, 1); glVertex2f(w, h)
        glTexCoord2f(0, 1); glVertex2f(0, h)
        glEnd()


# ---------- 模拟基类 ----------
class SimulationBase(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reset(self, pattern='default'):
        pass

    @abstractmethod
    def get_image(self, mode) -> np.ndarray:
        pass


# ---------- 模拟查看器 ----------
class SimulationViewer(QWidget):
    def __init__(self, simulation: SimulationBase, parent=None):
        super().__init__(parent)
        self.sim = simulation
        self.is_playing = True
        self.steps_per_frame = 20
        self.display_mode = ''

        self.gl_widget = TextureGLWidget(
            image_provider=lambda: self.sim.get_image(self.display_mode)
        )
        self.gl_widget.setFixedSize(600, 600)

        main_layout = QHBoxLayout()

        # 左侧面板宽度 320
        self.param_panel_layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(self.param_panel_layout)
        left_widget.setFixedWidth(320)
        main_layout.addWidget(left_widget)

        main_layout.addWidget(self.gl_widget)
        self.setLayout(main_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(30)

    # 提供给面板调用的方法
    def set_display_modes(self, modes: list):
        if modes:
            self.display_mode = modes[0]

    def toggle_play(self):
        self.is_playing = not self.is_playing

    def reset_sim(self):
        self.sim.reset('default')
        self.gl_widget.update()

    def set_steps_per_frame(self, val):
        self.steps_per_frame = val

    def save_image(self):
        img = self.sim.get_image(self.display_mode)
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        path, _ = QFileDialog.getSaveFileName(self, "保存图像", "pattern.png", "PNG (*.png)")
        if path and qimg.save(path):
            QMessageBox.information(self, "保存成功", f"已保存至:\n{path}")

    def _on_timer(self):
        if self.is_playing:
            for _ in range(self.steps_per_frame):
                self.sim.step()
            self.gl_widget.update()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


# ---------- 图像边框工具 ----------
def add_border(img, border_width=2, color=(0, 0, 0)):
    """在 numpy 图像数组四周添加实色边框"""
    if border_width <= 0:
        return img
    h, w, _ = img.shape
    img[:border_width, :, :] = color
    img[-border_width:, :, :] = color
    img[:, :border_width, :] = color
    img[:, -border_width:, :] = color
    return img


# ---------- 反应扩散专用渲染（黑白灰度 + 边框）----------
def render_reaction_image(u_np, v_np):
    """
    黑白灰度：取 U、V 中较大者作为灰度值，高浓度 → 黑，低浓度 → 白
    """
    gray = 1.0 - np.maximum(u_np, v_np)
    gray = np.clip(gray, 0, 1)
    img_gray = (gray * 255).astype(np.uint8)
    img = np.stack([img_gray, img_gray, img_gray], axis=-1)
    img = add_border(img, border_width=2, color=(0, 0, 0))
    return img


# ---------- UI 控件辅助函数 ----------
def make_slider_row(label, vmin, vmax, vdef, scale, sim, attr):
    """滑块 + 右侧可编辑框"""
    row = QHBoxLayout()
    row.addWidget(QLabel(label))
    slider = QSlider(Qt.Horizontal)
    slider.setRange(int(vmin * scale), int(vmax * scale))
    slider.setValue(int(vdef * scale))
    slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    edit = QLineEdit(f"{vdef:.4f}")
    edit.setFixedWidth(55)
    edit.setAlignment(Qt.AlignCenter)
    row.addWidget(slider)
    row.addWidget(edit)

    def sync_from_slider(val):
        scaled = val / scale
        edit.blockSignals(True)
        edit.setText(f"{scaled:.4f}")
        edit.blockSignals(False)
        setattr(sim, attr, scaled)

    def sync_from_edit():
        try:
            val = float(edit.text())
            if vmin <= val <= vmax:
                slider.blockSignals(True)
                slider.setValue(int(val * scale))
                slider.blockSignals(False)
                setattr(sim, attr, val)
            else:
                raise ValueError
        except ValueError:
            current = getattr(sim, attr)
            edit.blockSignals(True)
            edit.setText(f"{current:.4f}")
            edit.blockSignals(False)

    slider.valueChanged.connect(sync_from_slider)
    edit.editingFinished.connect(sync_from_edit)
    w = QWidget()
    w.setLayout(row)
    return w, slider, edit


def create_equation_group(title, html):
    """方程分组框"""
    group = QGroupBox(title)
    lb = QLabel(html)
    lb.setWordWrap(True)
    lb.setTextFormat(Qt.RichText)
    layout = QVBoxLayout()
    layout.addWidget(lb)
    group.setLayout(layout)
    return group


def create_param_group(sim, param_defs):
    """参数分组框，param_defs: [(label, vmin, vmax, vdef, scale, attr), ...]"""
    group = QGroupBox("模拟参数")
    layout = QVBoxLayout()
    layout.setSpacing(1)
    sliders = {}
    for label, vmin, vmax, vdef, scale, attr in param_defs:
        w, sld, edt = make_slider_row(label, vmin, vmax, vdef, scale, sim, attr)
        layout.addWidget(w)
        sliders[attr] = sld
    group.setLayout(layout)
    return group, sliders


def create_init_group(sim, viewer, pattern_options, presets_list, preset_callback, reset_mapping):
    """初始条件分组框"""
    group = QGroupBox("初始条件")
    layout = QVBoxLayout()
    pat_row = QHBoxLayout()
    pat_row.addWidget(QLabel("初始模式:"))
    pattern_combo = QComboBox()
    pattern_combo.addItems(pattern_options)
    pat_row.addWidget(pattern_combo)
    layout.addLayout(pat_row)

    if presets_list:
        pres_row = QHBoxLayout()
        pres_row.addWidget(QLabel("预设:"))
        preset_combo = QComboBox()
        preset_combo.addItems(presets_list)
        preset_combo.currentIndexChanged.connect(preset_callback)
        pres_row.addWidget(preset_combo)
        layout.addLayout(pres_row)

    group.setLayout(layout)

    def custom_reset():
        pattern = pattern_combo.currentText()
        sim.reset(reset_mapping.get(pattern, 'random'))
        viewer.gl_widget.update()
    viewer.reset_sim = custom_reset

    return group, pattern_combo, preset_combo if presets_list else None


def create_control_group(viewer):
    """播放/暂停、步数、保存"""
    group = QGroupBox("控制")
    layout = QVBoxLayout()
    steps_row = QHBoxLayout()
    steps_row.addWidget(QLabel("每帧步数:"))
    steps_slider = QSlider(Qt.Horizontal)
    steps_slider.setRange(1, 100)
    steps_slider.setValue(viewer.steps_per_frame)
    steps_label = QLabel(str(viewer.steps_per_frame))
    steps_slider.valueChanged.connect(
        lambda v: (viewer.set_steps_per_frame(v), steps_label.setText(str(v)))
    )
    steps_row.addWidget(steps_slider)
    steps_row.addWidget(steps_label)
    layout.addLayout(steps_row)

    btn_row = QHBoxLayout()
    play_btn = QPushButton("暂停" if viewer.is_playing else "播放")
    reset_btn = QPushButton("重置")
    def toggle_play():
        viewer.toggle_play()
        play_btn.setText("暂停" if viewer.is_playing else "播放")
    play_btn.clicked.connect(toggle_play)
    reset_btn.clicked.connect(viewer.reset_sim)
    btn_row.addWidget(play_btn)
    btn_row.addWidget(reset_btn)
    layout.addLayout(btn_row)

    save_btn = QPushButton("保存图像")
    save_btn.clicked.connect(viewer.save_image)
    layout.addWidget(save_btn)
    group.setLayout(layout)
    return group