# -*- coding: utf-8 -*-
"""MPM 2D: MLS-MPM materials on a clean white lab canvas."""
import random
import time

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QGridLayout

from common.lab_window import LabWindow, SquareCanvas
from common.qt_base import (
    LabeledSlider, QColor, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QPainter, QPen, QPointF, QPushButton, QTimer, QVBoxLayout,
    QWidget, Qt, run_app,
)
from support.physics.mpm2d import (
    DENSITY_OPTIONS,
    MATERIAL_ELASTIC,
    MATERIAL_SAND,
    MATERIAL_SNOW,
    MATERIAL_WATER,
    MPM2D,
    RES_DEFAULT,
    RES_OPTIONS,
)

MATERIAL_IDS = {
    "固体": MATERIAL_ELASTIC,
    "雪": MATERIAL_SNOW,
    "液体": MATERIAL_WATER,
    "沙子": MATERIAL_SAND,
}


class MPMCanvas(SquareCanvas):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self._image = None
        self.dragging = False
        self._attract_sign = 0
        self._pointer = None
        self.setCursor(Qt.CrossCursor)

    def set_frame(self, rgb):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        image = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
            QImage.Format_RGB888,
        )
        self._image = image.copy()
        self.update()

    def _to_world(self, pos):
        x = pos.x() / max(1.0, float(self.width()))
        y = 1.0 - pos.y() / max(1.0, float(self.height()))
        return x, y

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        if self._image is not None:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawImage(self.rect(), self._image)
        if self._pointer is not None and self._attract_sign != 0:
            color = QColor(44, 116, 214) if self._attract_sign > 0 else QColor(210, 74, 58)
            painter.setPen(QPen(color, 1.5, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self._pointer, 24.0, 24.0)
        painter.end()

    def mousePressEvent(self, event):
        point = event.position()
        x, y = self._to_world(point)
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ShiftModifier:
                self.dragging = True
                self.app.pour(x, y)
            else:
                self._attract_sign = 1
                self._pointer = point
                self.app.set_attractor(x, y, 1.0)
        elif event.button() == Qt.RightButton:
            self._attract_sign = -1
            self._pointer = point
            self.app.set_attractor(x, y, -1.0)
        else:
            super().mousePressEvent(event)
            return
        self.update()

    def mouseMoveEvent(self, event):
        point = event.position()
        if self.dragging and event.buttons() & Qt.LeftButton:
            self.app.pour(*self._to_world(point))
        elif self._attract_sign != 0:
            x, y = self._to_world(point)
            self._pointer = point
            self.app.set_attractor(x, y, 1.0 if self._attract_sign > 0 else -1.0)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
        if event.button() in (Qt.LeftButton, Qt.RightButton) and self._attract_sign != 0:
            self._attract_sign = 0
            self._pointer = None
            self.app.clear_attractor()
        self.update()


class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPM 2D · 材料")
        self.sim = MPM2D()
        self.material = "固体"
        self.paused = False
        self.frame = None
        self.frame_dt = 1.0 / 60.0
        self.fps = 0.0
        self._last_t = time.perf_counter()
        self._last_pour = 0.0
        self.grain = 1.0
        self.brush = 45
        self._stat_tick = 0
        self.substeps = 0
        self.canvas = MPMCanvas(self)
        self.setFocusPolicy(Qt.StrongFocus)
        self.init_ui()
        self.build_scene()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)
        self._last_t = time.perf_counter()

    def init_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        panel = QWidget()
        panel.setFixedWidth(330)
        panel.setStyleSheet("background-color: #ffffff;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 4, 4)
        layout.setSpacing(6)

        simulation = QGroupBox("模拟")
        sl = QVBoxLayout(simulation)
        sl.setContentsMargins(6, 6, 6, 6)
        sl.setSpacing(5)
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("网格"))
        self.res_box = QComboBox()
        for value in RES_OPTIONS:
            self.res_box.addItem(f"{value}²", value)
        self.res_box.setCurrentIndex(RES_OPTIONS.index(RES_DEFAULT))
        res_row.addWidget(self.res_box, 1)
        sl.addLayout(res_row)
        den_row = QHBoxLayout()
        den_row.addWidget(QLabel("采样"))
        self.den_box = QComboBox()
        for value in DENSITY_OPTIONS:
            self.den_box.addItem(f"{value} / 格", value)
        self.den_box.setCurrentIndex(2)
        den_row.addWidget(self.den_box, 1)
        sl.addLayout(den_row)
        self.grain_slider = LabeledSlider(
            "颗粒度", 1, 800, 100, formatter=lambda v: f"{v / 100:.2f}",
            label_width=56, value_width=44,
        )
        sl.addWidget(self.grain_slider)
        self.dts_box = QComboBox()
        for name, value in (("1×", 1.0), ("2×", 2.0), ("4×", 4.0)):
            self.dts_box.addItem(name, value)
        dts_row = QHBoxLayout()
        dts_row.addWidget(QLabel("步长"))
        dts_row.addWidget(self.dts_box, 1)
        sl.addLayout(dts_row)
        self.spd_box = QComboBox()
        for name, value in (("0.5×", 1 / 120), ("1×", 1 / 60), ("2×", 1 / 30)):
            self.spd_box.addItem(name, value)
        self.spd_box.setCurrentIndex(1)
        spd_row = QHBoxLayout()
        spd_row.addWidget(QLabel("时间"))
        spd_row.addWidget(self.spd_box, 1)
        sl.addLayout(spd_row)
        layout.addWidget(simulation)

        materials = QGroupBox("材料")
        ml = QGridLayout(materials)
        ml.setContentsMargins(6, 6, 6, 6)
        ml.setSpacing(4)
        self.buttons = {}
        for index, name in enumerate(("固体", "雪", "液体", "沙子")):
            button = QPushButton(name)
            button.setCheckable(True)
            button.setChecked(name == "固体")
            button.clicked.connect(lambda _=False, n=name: self.select(n))
            self.buttons[name] = button
            ml.addWidget(button, index // 2, index % 2)
        layout.addWidget(materials)

        brush = QGroupBox("画笔")
        bl = QVBoxLayout(brush)
        bl.setContentsMargins(6, 6, 6, 6)
        bl.setSpacing(4)
        self.brush_slider = LabeledSlider(
            "大小", 5, 100, 45, formatter=lambda v: f"{v}",
            label_width=44, value_width=40,
        )
        bl.addWidget(self.brush_slider)
        controls = QHBoxLayout()
        self.pause_btn = QPushButton("暂停")
        self.reset_btn = QPushButton("重置")
        self.save_btn = QPushButton("保存")
        for button in (self.pause_btn, self.reset_btn, self.save_btn):
            controls.addWidget(button)
        bl.addLayout(controls)
        note = QLabel(
            "左键拖拽：吸引　右键拖拽：排斥\n"
            "Shift + 左键：倾倒材料\n"
            "WASD / 方向键：改变重力"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#6b7280;")
        bl.addWidget(note)
        layout.addWidget(brush)
        layout.addStretch(1)

        root.addWidget(panel, 0)
        self.canvas.setFixedSize(600, 600)
        root.addWidget(self.canvas, 1, Qt.AlignCenter)
        self.setCentralWidget(central)
        central.setStyleSheet("background-color: #ffffff;")
        self.canvas.setFocusPolicy(Qt.NoFocus)
        self.setFocus()
        self.setMinimumSize(960, 640)
        self.resize(960, 640)

        self.status = self.statusBar()
        self.status.setSizeGripEnabled(False)
        self.state_label = QLabel("运行中")
        self.params_label = QLabel("N 0 · 0 子步 · 0 fps")
        self.status.addWidget(self.state_label, 1)
        self.status.addPermanentWidget(self.params_label)

        self.res_box.currentIndexChanged.connect(self.on_res)
        self.den_box.currentIndexChanged.connect(self.on_density)
        self.grain_slider.valueChanged.connect(self.on_grain)
        self.dts_box.currentIndexChanged.connect(
            lambda index: setattr(self.sim, "dt_scale", self.dts_box.itemData(index))
        )
        self.spd_box.currentIndexChanged.connect(self.on_speed)
        self.brush_slider.valueChanged.connect(
            lambda value: setattr(self, "brush", value)
        )
        self.pause_btn.clicked.connect(self.toggle)
        self.reset_btn.clicked.connect(self.reset)
        self.save_btn.clicked.connect(self.save)

    def build_scene(self):
        res = self.res_box.currentData() or self.sim.res
        density = self.den_box.currentData() or self.sim.density
        self.sim.configure(res, density)
        self.sim.add_cube(
            lower=(0.70, 0.06), size=(0.15, 0.15), material=MATERIAL_ELASTIC
        )
        self.sim.add_cube(
            lower=(0.70, 0.28), size=(0.15, 0.15), material=MATERIAL_SNOW
        )
        self.sim.add_cube(
            lower=(0.06, 0.05), size=(0.20, 0.32), material=MATERIAL_WATER
        )
        self.frame = self.sim.draw(grain=self.grain)
        self.canvas.set_frame(self.frame)
        self.substeps = 0
        self._update_status()

    def on_res(self, index):
        self.sim.configure(RES_OPTIONS[index], self.sim.density)
        self.build_scene()

    def on_density(self, index):
        self.sim.configure(self.sim.res, DENSITY_OPTIONS[index])
        self.build_scene()

    def on_grain(self, value):
        self.grain = value / 100.0
        self.frame = self.sim.draw(grain=self.grain)
        self.canvas.set_frame(self.frame)

    def on_speed(self, index):
        self.frame_dt = self.spd_box.itemData(index)

    def select(self, name):
        self.material = name
        for key, button in self.buttons.items():
            button.setChecked(key == name)

    def pour(self, x, y):
        now = time.perf_counter()
        if now - self._last_pour < 0.10:
            return
        self._last_pour = now
        if not (0.05 < x < 0.95 and 0.05 < y < 0.95):
            return
        radius = max(0.008, self.brush / 100.0 * 0.045)
        self.sim.add_cube(
            lower=(x - radius, y - radius),
            size=(2 * radius, 2 * radius),
            material=MATERIAL_IDS[self.material],
        )

    def set_attractor(self, x, y, strength):
        self.sim.set_attractor(x, y, strength)

    def clear_attractor(self):
        self.sim.set_attractor()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_Up):
            self.sim.set_gravity(0.0, 9.8)
            self.state_label.setText("重力 ↑")
        elif key in (Qt.Key_S, Qt.Key_Down):
            self.sim.set_gravity(0.0, -9.8)
            self.state_label.setText("重力 ↓")
        elif key in (Qt.Key_A, Qt.Key_Left):
            self.sim.set_gravity(-9.8, 0.0)
            self.state_label.setText("重力 ←")
        elif key in (Qt.Key_D, Qt.Key_Right):
            self.sim.set_gravity(9.8, 0.0)
            self.state_label.setText("重力 →")
        elif key == Qt.Key_R:
            self.reset()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def tick(self):
        if not self.paused:
            self.substeps = self.sim.step(self.frame_dt)
            self.frame = self.sim.draw(grain=self.grain)
            self.canvas.set_frame(self.frame)
            now = time.perf_counter()
            interval = now - self._last_t
            self._last_t = now
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(interval, 1e-6))
            self._update_status()

    def _update_status(self):
        self.params_label.setText(
            f"N {int(self.sim.n_particles[None])} · "
            f"{self.substeps} 子步 · {self.fps:.0f} fps"
        )

    def toggle(self):
        self.paused = not self.paused
        self.pause_btn.setText("继续" if self.paused else "暂停")
        self.state_label.setText("暂停" if self.paused else "运行中")

    def reset(self):
        self.build_scene()
        self.canvas.update()

    def save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 MPM 2D", "mpm2d.png", "PNG (*.png)"
        )
        if not path:
            return
        if self.canvas.save_image(path):
            self.state_label.setText("已保存")
        else:
            QMessageBox.warning(self, "保存失败", "无法写入所选路径。")

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
