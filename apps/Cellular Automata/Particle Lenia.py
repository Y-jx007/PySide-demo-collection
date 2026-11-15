# -*- coding: utf-8 -*-
"""Particle Lenia: reference two-ring kernel model with a clean white view."""
import math
import random
import time

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QImage, QPen, QPolygonF

from common.lab_window import LabWindow, SquareCanvas
from common.qt_base import (
    LabeledSlider, QCheckBox, QColor, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPainter, QPointF, QPushButton,
    QRectF, QSizePolicy, QTimer, QVBoxLayout, QWidget, Qt, run_app,
)
from support.physics.particle_lenia import WORLD_HALF, ParticleLenia


class ParticleCanvas(SquareCanvas):
    particlesAdded = Signal()

    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim = sim
        self._left_down = False
        self._right_down = False
        self._press_pos = None
        self._last_add_pos = None
        self._dragged = False
        self._show_field = False
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setMinimumSize(520, 520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _rect(self):
        return QRectF(0, 0, self.width(), self.height())

    def _to_screen(self, x, y):
        rect = self._rect()
        return QPointF(
            rect.left() + (0.5 + float(x) / (2.0 * WORLD_HALF)) * rect.width(),
            rect.top() + (0.5 - float(y) / (2.0 * WORLD_HALF)) * rect.height(),
        )

    def _to_world(self, point):
        rect = self._rect()
        nx = (point.x() - rect.left()) / max(1.0, rect.width())
        ny = (point.y() - rect.top()) / max(1.0, rect.height())
        return (nx - 0.5) * 2.0 * WORLD_HALF, (0.5 - ny) * 2.0 * WORLD_HALF

    def set_show_field(self, enabled):
        self._show_field = bool(enabled)
        self.update()

    def _draw_field(self, painter):
        positions = self.sim.positions()
        values = self.sim.field_values()
        if len(positions) == 0 or len(values) == 0:
            return
        grid = 120
        xs = np.clip((positions[:, 0] + WORLD_HALF) / (2.0 * WORLD_HALF) * (grid - 1), 0, grid - 1)
        ys = np.clip((WORLD_HALF - positions[:, 1]) / (2.0 * WORLD_HALF) * (grid - 1), 0, grid - 1)
        density, _, _ = np.histogram2d(
            ys, xs, bins=grid, range=[[0, grid], [0, grid]], weights=values
        )
        kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32)
        kernel /= kernel.sum()
        density = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, density)
        density = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 1, density)
        positive = density[density > 0]
        if positive.size == 0:
            return
        scale = float(np.percentile(positive, 98))
        if scale <= 1e-9:
            return
        t = np.clip(density / scale, 0.0, 1.0) ** 0.65
        rgba = np.empty((grid, grid, 4), dtype=np.uint8)
        rgba[..., 0] = np.uint8(40 + 210 * t)
        rgba[..., 1] = np.uint8(85 + 120 * t)
        rgba[..., 2] = np.uint8(185 - 90 * t)
        rgba[..., 3] = np.uint8(120 * t)
        rgba = np.ascontiguousarray(rgba)
        image = QImage(rgba.data, grid, grid, 4 * grid, QImage.Format_RGBA8888).copy()
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(self.rect(), image)
        painter.restore()

    def _set_pointer(self, point, mode):
        x, y = self._to_world(point)
        if abs(x) <= WORLD_HALF and abs(y) <= WORLD_HALF:
            self.sim.set_pointer(x, y, mode)
        else:
            self._clear_pointer()
        self.update()

    def _clear_pointer(self):
        self.sim.set_pointer(None)

    def _add_burst(self, point, count=16, radius=3.0):
        x, y = self._to_world(point)
        added = 0
        for _ in range(count):
            px = x + random.gauss(0.0, radius * 0.55)
            py = y + random.gauss(0.0, radius * 0.55)
            if abs(px) <= WORLD_HALF and abs(py) <= WORLD_HALF:
                self.sim.add_particle(px, py)
                added += 1
        if added:
            self.particlesAdded.emit()
            self.update()

    def _add_stroke(self, start, end):
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        distance = math.hypot(dx, dy)
        steps = max(1, int(distance / 3.5))
        added = 0
        for index in range(1, steps + 1):
            t = index / steps
            point = QPointF(start.x() + dx * t, start.y() + dy * t)
            x, y = self._to_world(point)
            for _ in range(2):
                px = x + random.gauss(0.0, 0.9)
                py = y + random.gauss(0.0, 0.9)
                if abs(px) <= WORLD_HALF and abs(py) <= WORLD_HALF:
                    self.sim.add_particle(px, py)
                    added += 1
        if added:
            self.particlesAdded.emit()
            self.update()

    def mousePressEvent(self, event):
        point = event.position()
        if event.button() == Qt.LeftButton:
            self._left_down = True
            self._press_pos = point
            self._last_add_pos = point
            self._dragged = False
            if event.modifiers() & Qt.ShiftModifier:
                self._clear_pointer()
                self._add_burst(point, 8, 2.0)
            else:
                self._set_pointer(point, "attract")
        elif event.button() == Qt.RightButton:
            self._right_down = True
            self._set_pointer(point, "repel")
        else:
            super().mousePressEvent(event)
            return
        event.accept()

    def mouseMoveEvent(self, event):
        point = event.position()
        if self._left_down:
            if self._press_pos is not None:
                if (point - self._press_pos).manhattanLength() > 3:
                    self._dragged = True
            if event.modifiers() & Qt.ShiftModifier:
                self._clear_pointer()
                if self._dragged and self._last_add_pos is not None:
                    self._add_stroke(self._last_add_pos, point)
                self._last_add_pos = point
            elif not self._right_down:
                self._set_pointer(point, "attract")
        elif self._right_down:
            self._set_pointer(point, "repel")
        event.accept()

    def mouseReleaseEvent(self, event):
        point = event.position()
        if event.button() == Qt.LeftButton:
            self._left_down = False
            if not self._right_down:
                self._clear_pointer()
        elif event.button() == Qt.RightButton:
            self._right_down = False
            if not self._left_down:
                self._clear_pointer()
        else:
            super().mouseReleaseEvent(event)
            return
        self.update()
        event.accept()

    def leaveEvent(self, event):
        if not self._left_down and not self._right_down:
            self._clear_pointer()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        if self._show_field:
            self._draw_field(painter)
        positions = self.sim.positions()
        velocities = self.sim.velocities()
        buckets = [[] for _ in range(12)]
        for i in range(len(positions)):
            speed = min(1.0, float(np.linalg.norm(velocities[i])) / 0.6)
            buckets[min(11, int(speed * 12))].append(
                self._to_screen(*positions[i])
            )
        for index, points in enumerate(buckets):
            if not points:
                continue
            speed = index / 11.0
            color = QColor(
                int(30 + 170 * speed),
                int(75 + 95 * speed),
                int(150 - 70 * speed),
            )
            glow = QColor(color)
            glow.setAlpha(int(16 + 42 * speed))
            polygon = QPolygonF(points)
            painter.setPen(QPen(glow, 5.0 + 2.0 * speed, Qt.SolidLine, Qt.RoundCap))
            painter.drawPoints(polygon)
            painter.setPen(QPen(color, 2.2 + 0.9 * speed, Qt.SolidLine, Qt.RoundCap))
            painter.drawPoints(polygon)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(150, 155, 165), 1.2))
        painter.drawRect(QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0))
        painter.end()

    def save_image(self, path):
        return self.grab().save(path)


class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.setWindowTitle("粒子 Lenia")
        self.sim = ParticleLenia()
        self.sim.reset("随机", 900)
        self.canvas = ParticleCanvas(self.sim)
        self.paused = False
        self._frames = 0
        self._fps = 0.0
        self._last_fps_time = time.perf_counter()

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        panel = QWidget()
        panel.setObjectName("particlePanel")
        panel.setFixedWidth(330)
        panel.setStyleSheet("QWidget#particlePanel { background-color: #ffffff; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 4, 4)
        layout.setSpacing(6)

        physics = QGroupBox("物理")
        pl = QVBoxLayout(physics)
        pl.setContentsMargins(6, 6, 6, 6)
        pl.setSpacing(5)
        self.m1_slider = LabeledSlider(
            "核半径 m₁", 80, 200, 127,
            formatter=lambda v: f"{v / 100:.2f}",
            label_width=64, value_width=44,
        )
        self.s1_slider = LabeledSlider(
            "核宽 s₁", 50, 200, 100,
            formatter=lambda v: f"{v / 100:.2f}",
            label_width=64, value_width=44,
        )
        self.m2_slider = LabeledSlider(
            "生长 m₂", 20, 120, 60,
            formatter=lambda v: f"{v / 100:.2f}",
            label_width=64, value_width=44,
        )
        self.s2_slider = LabeledSlider(
            "生长宽 s₂", 5, 40, 20,
            formatter=lambda v: f"{v / 100:.2f}",
            label_width=64, value_width=44,
        )
        self.repulsion_slider = LabeledSlider(
            "排斥", 0, 300, 100,
            formatter=lambda v: f"{v / 100:.2f}",
            label_width=64, value_width=44,
        )
        self.dt_slider = LabeledSlider(
            "步长 dt", 2, 20, 10,
            formatter=lambda v: f"{v / 100:.2f}",
            label_width=64, value_width=44,
        )
        for slider in (self.m1_slider, self.s1_slider, self.m2_slider,
                       self.s2_slider, self.repulsion_slider, self.dt_slider):
            pl.addWidget(slider)
        layout.addWidget(physics)

        system = QGroupBox("系统")
        sl = QVBoxLayout(system)
        sl.setContentsMargins(6, 6, 6, 6)
        sl.setSpacing(5)
        pattern_row = QHBoxLayout()
        pattern_row.addWidget(QLabel("初始"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(("随机", "圆盘", "环", "链"))
        pattern_row.addWidget(self.pattern_combo, 1)
        sl.addLayout(pattern_row)
        self.count_slider = LabeledSlider(
            "粒子数", 200, 5000, 900,
            formatter=str, label_width=64, value_width=44,
        )
        sl.addWidget(self.count_slider)
        self.substeps_slider = LabeledSlider(
            "每帧步数", 1, 128, 16,
            formatter=str, label_width=64, value_width=44,
        )
        sl.addWidget(self.substeps_slider)
        add_note = QLabel(
            "左键拖拽：吸引　右键拖拽：排斥\n"
            "Shift + 左键拖拽：添加粒子\n"
            "粒子上限 5000"
        )
        add_note.setWordWrap(True)
        add_note.setStyleSheet("color:#6b7280;")
        sl.addWidget(add_note)
        self.field_check = QCheckBox("显示场浓度")
        self.field_check.toggled.connect(self.canvas.set_show_field)
        sl.addWidget(self.field_check, 0, Qt.AlignLeft)
        layout.addWidget(system)

        control = QGroupBox("控制")
        cl = QHBoxLayout(control)
        cl.setContentsMargins(6, 6, 6, 6)
        cl.setSpacing(4)
        self.pause_btn = QPushButton("暂停")
        self.reset_btn = QPushButton("重置")
        self.save_btn = QPushButton("保存")
        for button in (self.pause_btn, self.reset_btn, self.save_btn):
            cl.addWidget(button)
        layout.addWidget(control)
        layout.addStretch(1)

        root.addWidget(panel, 0)
        self.canvas.setFixedSize(600, 600)
        root.addWidget(self.canvas, 1, Qt.AlignCenter)
        central.setObjectName("particleCentral")
        self.setCentralWidget(central)
        central.setStyleSheet("QWidget#particleCentral { background-color: #ffffff; }")
        self.setMinimumSize(960, 640)
        self.resize(960, 640)

        self.status = self.statusBar()
        self.status.setSizeGripEnabled(False)
        self.state_label = QLabel("运行中")
        self.params_label = QLabel("N 900 · 0 fps")
        self.status.addWidget(self.state_label, 1)
        self.status.addPermanentWidget(self.params_label)

        self.m1_slider.valueChanged.connect(self._apply_params)
        self.s1_slider.valueChanged.connect(self._apply_params)
        self.m2_slider.valueChanged.connect(self._apply_params)
        self.s2_slider.valueChanged.connect(self._apply_params)
        self.repulsion_slider.valueChanged.connect(self._apply_params)
        self.dt_slider.valueChanged.connect(self._apply_params)
        self.substeps_slider.valueChanged.connect(self._apply_params)
        self.count_slider.valueChanged.connect(self._reset)
        self.pattern_combo.currentTextChanged.connect(self._reset)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.reset_btn.clicked.connect(self._reset)
        self.save_btn.clicked.connect(self._save)
        self.canvas.particlesAdded.connect(self._update_status)

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._apply_params()

    def _apply_params(self, *args):
        self.sim.set_kernel(
            self.m1_slider.value() / 100.0,
            self.s1_slider.value() / 100.0,
        )
        self.sim.m2 = self.m2_slider.value() / 100.0
        self.sim.s2 = self.s2_slider.value() / 100.0
        self.sim.repulsion = self.repulsion_slider.value() / 100.0
        self.sim.dt = self.dt_slider.value() / 100.0
        self.sim.substeps = self.substeps_slider.value()

    def _reset(self, *args):
        self.sim.reset(
            self.pattern_combo.currentText(),
            self.count_slider.value(),
        )
        self.canvas.update()
        self._update_status()

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.setText("继续" if self.paused else "暂停")
        self.state_label.setText("暂停" if self.paused else "运行中")

    def _tick(self):
        if not self.paused:
            self.sim.step()
            self.canvas.update()
        self._frames += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 0.6:
            self._fps = self._frames / elapsed
            self._frames = 0
            self._last_fps_time = now
            self._update_status()

    def _update_status(self):
        self.params_label.setText(
            f"N {self.sim.count()} · {self._fps:.0f} fps"
        )

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存粒子 Lenia", "particle-lenia.png", "PNG (*.png)"
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
