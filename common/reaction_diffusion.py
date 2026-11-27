import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QGroupBox, QFileDialog, QMessageBox,
    QLineEdit, QComboBox, QSizePolicy, QScrollArea, QFrame
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from abc import ABC, abstractmethod


# ---------- OpenGL 设置 ----------
def setup_gl_format():
    """兼容旧导入路径；实现已统一到 common.app。"""
    from common.app import setup_gl_format as _setup

    _setup(swap_interval=0)


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
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

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
        self.gl_widget.setFixedSize(640, 640)
        self.gl_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)
        self.main_layout = main_layout

        self.param_panel_layout = QVBoxLayout()
        self.param_panel_layout.setContentsMargins(8, 4, 8, 0)
        self.param_panel_layout.setSpacing(8)
        left_widget = QWidget()
        left_widget.setLayout(self.param_panel_layout)

        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.param_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.param_scroll.setFrameShape(QFrame.NoFrame)
        self.param_scroll.setFixedWidth(340)
        self.param_scroll.setWidget(left_widget)
        # 不要给 QScrollArea 设样式表：Qt 会给它换上 QStyleSheetStyle，
        # 竖直滚动条就不再按 Windows 原生外观绘制。透明背景用 palette 做。
        self.param_scroll.viewport().setAutoFillBackground(False)
        left_widget.setAutoFillBackground(False)
        main_layout.addWidget(self.param_scroll, 0)
        main_layout.addWidget(self.gl_widget, 1)

        # 连续循环：每帧渲染完成后由 frameSwapped 驱动下一步。
        # 没有固定 30 ms 定时器；节奏由显示刷新/交换链决定。
        self.gl_widget.frameSwapped.connect(self._on_frame_swapped)
        self.gl_widget.update()

    # 提供给面板调用的方法
    def set_display_modes(self, modes: list):
        if modes:
            self.display_mode = modes[0]

    def set_top_widget(self, widget):
        """Place a mode/size toolbar above the square canvas."""
        self.main_layout.removeWidget(self.gl_widget)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(widget)
        right_layout.addWidget(self.gl_widget, 0, Qt.AlignHCenter | Qt.AlignTop)
        self.main_layout.addWidget(right, 1)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.gl_widget.update()

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

    def _on_frame_swapped(self):
        # 最小化/隐藏时不要继续烧 GPU；重新显示后由 showEvent 恢复。
        if not self.is_playing or not self.isVisible() or self.isMinimized():
            return
        for _ in range(self.steps_per_frame):
            self.sim.step()
        self.gl_widget.update()

    def showEvent(self, event):
        super().showEvent(event)
        if self.is_playing:
            self.gl_widget.update()

    def closeEvent(self, event):
        self.is_playing = False
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
_render_cache = {}


def _render_buffers(shape):
    """Reuse the large grayscale/RGB buffers instead of allocating every frame."""
    key = tuple(shape)
    buf = _render_cache.get(key)
    if buf is None:
        tmp = np.empty(shape, dtype=np.float32)
        gray = np.empty(shape, dtype=np.uint8)
        img = np.empty((shape[0], shape[1], 3), dtype=np.uint8)
        buf = (tmp, gray, img)
        _render_cache[key] = buf
    return buf


def _finish_gray(tmp, gray, img):
    np.clip(tmp, 0.0, 1.0, out=tmp)
    np.multiply(tmp, 255.0, out=tmp)
    gray[...] = tmp
    img[..., 0] = gray
    img[..., 1] = gray
    img[..., 2] = gray
    return add_border(img, border_width=2, color=(0, 0, 0))


def render_reaction_image(u_np, v_np):
    """黑白灰度：取 U、V 中较大者作为灰度值，高浓度 → 黑，低浓度 → 白。"""
    tmp, gray, img = _render_buffers(u_np.shape)
    np.maximum(u_np, v_np, out=tmp)
    np.subtract(1.0, tmp, out=tmp)
    return _finish_gray(tmp, gray, img)


def render_gray_image(gray_np):
    """Cahn-Hilliard 的单通道灰度图，同样复用缓冲区。"""
    tmp, gray, img = _render_buffers(gray_np.shape)
    np.copyto(tmp, gray_np, casting="unsafe")
    return _finish_gray(tmp, gray, img)


# ---------- UI 控件辅助函数 ----------
def make_slider_row(label, vmin, vmax, vdef, scale, sim, attr, compact=False):
    """滑块 + 右侧可编辑框"""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4 if compact else 6)
    lab = QLabel(label)
    if compact:
        lab.setFixedWidth(46)
        lab.setStyleSheet("font-size: 11px;")
    row.addWidget(lab)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(int(vmin * scale), int(vmax * scale))
    slider.setValue(int(vdef * scale))
    slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    if compact:
        slider.setFixedHeight(16)
    edit = QLineEdit(f"{vdef:.4f}")
    edit.setFixedWidth(46 if compact else 55)
    edit.setAlignment(Qt.AlignCenter)
    if compact:
        edit.setFixedHeight(18)
        edit.setStyleSheet("font-size: 11px;")
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
    if compact:
        w.setFixedHeight(22)
    return w, slider, edit


def create_equation_group(title, html, compact=False):
    """方程分组框"""
    group = QGroupBox(title)
    lb = QLabel(html)
    lb.setWordWrap(True)
    lb.setTextFormat(Qt.RichText)
    layout = QVBoxLayout()
    if compact:
        layout.setContentsMargins(6, 3, 6, 3)
        lb.setStyleSheet("font-size: 11px;")
    else:
        layout.setContentsMargins(8, 6, 8, 6)
    layout.addWidget(lb)
    group.setLayout(layout)
    return group


def create_param_group(sim, param_defs, compact=False):
    """参数分组框，param_defs: [(label, vmin, vmax, vdef, scale, attr), ...]"""
    group = QGroupBox("模拟参数")
    layout = QVBoxLayout()
    layout.setSpacing(1 if compact else 3)
    if compact:
        layout.setContentsMargins(6, 3, 6, 3)
    else:
        layout.setContentsMargins(8, 6, 8, 6)
    sliders = {}
    for label, vmin, vmax, vdef, scale, attr in param_defs:
        w, sld, edt = make_slider_row(label, vmin, vmax, vdef, scale, sim, attr, compact=compact)
        layout.addWidget(w)
        sliders[attr] = sld
    group.setLayout(layout)
    return group, sliders


def _play_control_rows(viewer):
    steps_row = QHBoxLayout()
    steps_row.addWidget(QLabel("每帧步数:"))
    steps_slider = QSlider(Qt.Horizontal)
    steps_slider.setRange(1, 100)
    steps_slider.setValue(viewer.steps_per_frame)
    steps_label = QLabel(str(viewer.steps_per_frame))
    steps_slider.valueChanged.connect(
        lambda v: (viewer.set_steps_per_frame(v), steps_label.setText(str(v)))
    )
    steps_row.addWidget(steps_slider, 1)
    steps_row.addWidget(steps_label)

    btn_row = QHBoxLayout()
    btn_row.setSpacing(6)
    play_btn = QPushButton("暂停" if viewer.is_playing else "开始")
    reset_btn = QPushButton("重置")
    save_btn = QPushButton("保存")
    def toggle_play():
        viewer.toggle_play()
        play_btn.setText("暂停" if viewer.is_playing else "开始")
    play_btn.clicked.connect(toggle_play)
    reset_btn.clicked.connect(viewer.reset_sim)
    save_btn.clicked.connect(viewer.save_image)
    for button in (play_btn, reset_btn, save_btn):
        btn_row.addWidget(button, 1)
    return steps_row, btn_row


def create_init_group(sim, viewer, pattern_options, presets_list,
                      preset_callback, reset_mapping, merge_control=False):
    """初始条件分组框；merge_control=True 时与播放控制合并。"""
    group = QGroupBox("初始化与控制" if merge_control else "初始条件")
    layout = QVBoxLayout()
    layout.setContentsMargins(8, 6, 8, 6)
    layout.setSpacing(6)

    pattern_combo = QComboBox()
    pattern_combo.addItems(pattern_options)
    preset_combo = None
    if merge_control:
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(QLabel("初始模式:"))
        row.addWidget(pattern_combo, 1)
        if presets_list:
            row.addSpacing(4)
            row.addWidget(QLabel("预设:"))
            preset_combo = QComboBox()
            preset_combo.addItems(presets_list)
            preset_combo.currentIndexChanged.connect(preset_callback)
            row.addWidget(preset_combo, 1)
        layout.addLayout(row)
    else:
        pat_row = QHBoxLayout()
        pat_row.addWidget(QLabel("初始模式:"))
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

    if merge_control:
        steps_row, btn_row = _play_control_rows(viewer)
        layout.addLayout(steps_row)
        layout.addLayout(btn_row)

    return group, pattern_combo, preset_combo if presets_list else None


def create_control_group(viewer):
    """播放/暂停、步数、保存"""
    group = QGroupBox("控制")
    layout = QVBoxLayout()
    layout.setContentsMargins(8, 6, 8, 6)
    layout.setSpacing(6)
    steps_row, btn_row = _play_control_rows(viewer)
    layout.addLayout(steps_row)
    layout.addLayout(btn_row)
    group.setLayout(layout)
    return group

    btn_row = QHBoxLayout()
    btn_row.setSpacing(6)
    play_btn = QPushButton("暂停" if viewer.is_playing else "开始")
    reset_btn = QPushButton("重置")
    save_btn = QPushButton("保存")
    def toggle_play():
        viewer.toggle_play()
        play_btn.setText("暂停" if viewer.is_playing else "开始")
    play_btn.clicked.connect(toggle_play)
    reset_btn.clicked.connect(viewer.reset_sim)
    save_btn.clicked.connect(viewer.save_image)
    for button in (play_btn, reset_btn, save_btn):
        btn_row.addWidget(button, 1)
    layout.addLayout(btn_row)
    group.setLayout(layout)
    return group