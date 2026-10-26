# -*- coding: utf-8 -*-
"""Greenberg-Hastings excitable-media cellular automaton."""
from __future__ import annotations

import sys

import numpy as np

from common.qt_base import (
    QApplication, QComboBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
    QImage, QLabel, QMainWindow, QMouseEvent, QPainter, QPushButton, QRect,
    QScrollArea, QSizePolicy, QSlider, QTimer, QVBoxLayout, QWidget,
    Qt, Signal,
)

from greenberg_hastings_model import GreenbergHastings


SIZE_CHOICES = (256, 384, 512, 768, 1024)
PRESETS = (
    ("螺旋波", "spiral"),
    ("随机激发", "random"),
    ("靶波", "target"),
    ("空白", "empty"),
)


def build_lut(refractory):
    """RGB lookup: resting dark, excited white, refractory warm-to-teal."""
    lut = np.zeros((refractory + 1, 3), dtype=np.uint8)
    lut[0] = (5, 13, 21)
    lut[1] = (225, 255, 248)
    for state in range(2, refractory + 1):
        t = (state - 2) / max(1, refractory - 2)
        r = int(245 * (1 - t) + 10 * t)
        g = int(145 * (1 - t) + 54 * t)
        b = int(65 * (1 - t) + 67 * t)
        lut[state] = (r, g, b)
    return lut


class GHCanvas(QWidget):
    stimulated = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image = None
        self._grid = 0
        self._rect = None

    def set_frame(self, rgb, grid_size):
        self._grid = int(grid_size)
        self._image = QImage(
            rgb.data, self._grid, self._grid, 3 * self._grid,
            QImage.Format_RGB888,
        ).copy()
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._image is None:
            painter.end()
            return
        side = min(self.width(), self.height()) - 28
        if side < 20:
            painter.end()
            return
        left = (self.width() - side) // 2
        top = (self.height() - side) // 2
        self._rect = (left, top, side, side)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(QRect(*self._rect), self._image)
        painter.end()

    def _map_event(self, event: QMouseEvent):
        if self._rect is None or self._grid <= 0:
            return None
        left, top, side, _ = self._rect
        x = (event.position().x() - left) / max(1, side)
        y = (event.position().y() - top) / max(1, side)
        if x < 0 or x >= 1 or y < 0 or y >= 1:
            return None
        return int(y * self._grid), int(x * self._grid)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self._map_event(event)
            if pos is not None:
                self.stimulated.emit(pos[0], pos[1])

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            pos = self._map_event(event)
            if pos is not None:
                self.stimulated.emit(pos[0], pos[1])


class GHWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Greenberg-Hastings — 激发介质元胞自动机")
        self.resize(1180, 820)
        self.setMinimumSize(900, 620)

        self.sim = GreenbergHastings(512, refractory=6, threshold=1, neighbors=8)
        self.sim.reset("spiral")
        self.lut = build_lut(self.sim.refractory)
        self.running = True
        self._steps_per_frame = 1

        self._build_ui()
        self.canvas.stimulated.connect(self._on_stimulate)
        self._render()
        self._update_stats()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left = QWidget()
        left.setFixedWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        panel = QVBoxLayout(content)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.setSpacing(8)
        panel.addWidget(self._build_model_group())
        panel.addWidget(self._build_seed_group())
        panel.addWidget(self._build_control_group())
        panel.addWidget(self._build_stats_group())
        panel.addStretch(1)
        scroll.setWidget(content)
        left_layout.addWidget(scroll)
        root.addWidget(left, 0)

        self.canvas = GHCanvas()
        root.addWidget(self.canvas, 1)
        self.statusBar().showMessage("左键在画布上刺激细胞。")

    def _build_model_group(self):
        group = QGroupBox("模型")
        form = QFormLayout(group)
        form.setSpacing(6)

        self.size_combo = QComboBox()
        for size in SIZE_CHOICES:
            self.size_combo.addItem(f"{size} × {size}", size)
        self.size_combo.setCurrentIndex(2)
        self.size_combo.currentIndexChanged.connect(self._rebuild)
        form.addRow("网格", self.size_combo)

        self.refractory_slider = QSlider(Qt.Horizontal)
        self.refractory_slider.setRange(2, 12)
        self.refractory_slider.setValue(6)
        self.refractory_label = QLabel("6")
        self.refractory_slider.valueChanged.connect(self._refractory_changed)
        refractory_row = QHBoxLayout()
        refractory_row.addWidget(self.refractory_slider, 1)
        refractory_row.addWidget(self.refractory_label)
        form.addRow("不应期", refractory_row)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 4)
        self.threshold_slider.setValue(1)
        self.threshold_label = QLabel("1")
        self.threshold_slider.valueChanged.connect(self._threshold_changed)
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self.threshold_slider, 1)
        threshold_row.addWidget(self.threshold_label)
        form.addRow("激发阈值", threshold_row)

        self.neighbour_combo = QComboBox()
        self.neighbour_combo.addItem("Moore 8", 8)
        self.neighbour_combo.addItem("von Neumann 4", 4)
        self.neighbour_combo.currentIndexChanged.connect(self._rebuild)
        form.addRow("邻域", self.neighbour_combo)
        return group

    def _build_seed_group(self):
        group = QGroupBox("初始条件")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.preset_combo = QComboBox()
        for label, key in PRESETS:
            self.preset_combo.addItem(label, key)
        self.preset_combo.currentIndexChanged.connect(self._reset)
        layout.addWidget(self.preset_combo)

        density_row = QHBoxLayout()
        density_row.addWidget(QLabel("随机密度"))
        self.density_slider = QSlider(Qt.Horizontal)
        self.density_slider.setRange(1, 50)
        self.density_slider.setValue(20)
        self.density_label = QLabel("2.0%")
        self.density_slider.valueChanged.connect(self._density_changed)
        density_row.addWidget(self.density_slider, 1)
        density_row.addWidget(self.density_label)
        layout.addLayout(density_row)

        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self._reset)
        layout.addWidget(self.reset_btn)
        return group

    def _build_control_group(self):
        group = QGroupBox("控制")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        row = QHBoxLayout()
        self.start_btn = QPushButton("暂停")
        self.start_btn.clicked.connect(self._toggle_run)
        self.step_btn = QPushButton("单步")
        self.step_btn.clicked.connect(self._single_step)
        row.addWidget(self.start_btn)
        row.addWidget(self.step_btn)
        layout.addLayout(row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("每帧步数"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 4)
        self.speed_slider.setValue(1)
        self.speed_slider.valueChanged.connect(self._speed_changed)
        self.speed_label = QLabel("1")
        speed_row.addWidget(self.speed_slider, 1)
        speed_row.addWidget(self.speed_label)
        layout.addLayout(speed_row)

        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self._clear)
        layout.addWidget(self.clear_btn)
        return group

    def _build_stats_group(self):
        group = QGroupBox("统计")
        form = QFormLayout(group)
        form.setSpacing(4)
        self.stats_labels = {}
        for key, text in (
            ("tick", "代数"),
            ("excited", "激发比例"),
            ("refractory", "不应期比例"),
            ("wavefront", "波前细胞"),
        ):
            value = QLabel("–")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.addRow(text, value)
            self.stats_labels[key] = value
        return group

    # ------------------------------------------------------------- controls
    def _refractory_changed(self, value):
        self.refractory_label.setText(str(value))
        self._rebuild()

    def _threshold_changed(self, value):
        self.threshold_label.setText(str(value))
        self._rebuild()

    def _density_changed(self, value):
        self.density_label.setText(f"{value / 10.0:.1f}%")

    def _speed_changed(self, value):
        self._steps_per_frame = int(value)
        self.speed_label.setText(str(value))

    def _rebuild(self):
        self.sim = GreenbergHastings(
            int(self.size_combo.currentData()),
            refractory=int(self.refractory_slider.value()),
            threshold=int(self.threshold_slider.value()),
            neighbors=int(self.neighbour_combo.currentData()),
        )
        self.lut = build_lut(self.sim.refractory)
        self._reset()

    def _reset(self):
        mode = self.preset_combo.currentData()
        density = self.density_slider.value() / 1000.0
        self.sim.reset(mode, density=density)
        self._render()
        self._update_stats()
        self.statusBar().showMessage(
            f"{self.preset_combo.currentText()}：{self.sim.size}×{self.sim.size}，"
            f"不应期 {self.sim.refractory}，阈值 {self.sim.threshold}"
        )

    def _clear(self):
        self.sim.clear()
        self._render()
        self._update_stats()
        self.statusBar().showMessage("已清空；左键点击画布重新刺激。")

    def _toggle_run(self):
        self.running = not self.running
        if self.running:
            self.timer.start(40)
            self.start_btn.setText("暂停")
        else:
            self.timer.stop()
            self.start_btn.setText("开始")

    def _single_step(self):
        if self.running:
            self.running = False
            self.timer.stop()
            self.start_btn.setText("开始")
        self.sim.step()
        self._render()
        self._update_stats()

    def _on_stimulate(self, y, x):
        radius = max(2, self.sim.size // 128)
        self.sim.stimulate(y, x, radius)
        self._render()
        self._update_stats()

    # --------------------------------------------------------------- render
    def _tick(self):
        for _ in range(self._steps_per_frame):
            self.sim.step()
        self._render()
        self._update_stats()

    def _render(self):
        rgb = self.lut[self.sim.state]
        rgb = np.ascontiguousarray(rgb)
        self.canvas.set_frame(rgb, self.sim.size)

    def _update_stats(self):
        m = self.sim.metrics()
        total = self.sim.size * self.sim.size
        self.stats_labels["tick"].setText(str(m["tick"]))
        self.stats_labels["excited"].setText(f"{100.0 * m['excited'] / total:5.2f}%")
        self.stats_labels["refractory"].setText(f"{100.0 * m['refractory'] / total:5.2f}%")
        self.stats_labels["wavefront"].setText(f"{m['excited']:,}")

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = GHWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
