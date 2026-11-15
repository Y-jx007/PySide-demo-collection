# -*- coding: utf-8 -*-
"""Quasicrystal: real-space plane waves and their FFT diffraction."""
import numpy as np
from PySide6.QtGui import QImage

from common.lab_window import LabWindow, SquareCanvas
from common.qt_base import (
    LabeledSlider, QCheckBox, QColor, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPainter, QPushButton,
    QSizePolicy, QTimer, QVBoxLayout, QWidget, Qt, run_app,
)
from support.physics.quasicrystal import Quasicrystal


class QuasicrystalCanvas(SquareCanvas):
    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.mode = "real"
        self.palette = "冰蓝"
        self.log = True
        self._image = None
        self.setMinimumSize(520, 520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def refresh(self):
        if self.mode == "diffraction":
            rgb = self.sim.render_diffraction(
                self.palette, log=self.log, zoom=self.sim.diffraction_zoom
            )
        else:
            rgb = self.sim.render_real(self.palette)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        image = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
            QImage.Format_RGB888,
        )
        self._image = image.copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        if self._image is not None:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawImage(self.rect(), self._image)
        painter.end()

    def save_image(self, path):
        return self.grab().save(path)


class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #ffffff;")
        self.setWindowTitle("二维准晶")
        self.sim = Quasicrystal(512)
        self.canvas = QuasicrystalCanvas(self.sim)
        self.playing = False
        self._last_fps_time = 0.0

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

        structure = QGroupBox("结构")
        sl = QVBoxLayout(structure)
        sl.setContentsMargins(6, 6, 6, 6)
        sl.setSpacing(5)
        self.n_slider = LabeledSlider(
            "对称 N", 3, 20, 7, formatter=str, label_width=56, value_width=40
        )
        self.wave_slider = LabeledSlider(
            "波矢", 50, 1000, 500, formatter=lambda v: f"{v / 100:.2f}",
            label_width=56, value_width=44,
        )
        self.zoom_slider = LabeledSlider(
            "视野", 400, 2400, 1200, formatter=lambda v: f"{v / 100:.1f}",
            label_width=56, value_width=44,
        )
        self.rotation_slider = LabeledSlider(
            "旋转", -180, 180, 0, formatter=lambda v: f"{v}°",
            label_width=56, value_width=44,
        )
        self.diff_zoom_slider = LabeledSlider(
            "衍射缩放", 1, 8, 4, formatter=lambda v: f"×{v}",
            label_width=56, value_width=40,
        )
        for slider in (self.n_slider, self.wave_slider, self.zoom_slider,
                       self.rotation_slider, self.diff_zoom_slider):
            sl.addWidget(slider)
        layout.addWidget(structure)

        display = QGroupBox("显示")
        dl = QVBoxLayout(display)
        dl.setContentsMargins(6, 6, 6, 6)
        dl.setSpacing(5)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("视图"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(("实空间", "衍射"))
        mode_row.addWidget(self.mode_combo, 1)
        dl.addLayout(mode_row)
        palette_row = QHBoxLayout()
        palette_row.addWidget(QLabel("配色"))
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(("冰蓝", "火焰", "热力", "灰度"))
        palette_row.addWidget(self.palette_combo, 1)
        dl.addLayout(palette_row)
        self.log_check = QCheckBox("对数强度")
        self.log_check.setChecked(True)
        dl.addWidget(self.log_check)
        layout.addWidget(display)

        animation = QGroupBox("动画")
        al = QHBoxLayout(animation)
        al.setContentsMargins(6, 6, 6, 6)
        al.setSpacing(4)
        self.play_btn = QPushButton("动画")
        self.play_btn.setCheckable(True)
        al.addWidget(self.play_btn)
        self.speed_slider = LabeledSlider(
            "速度", 1, 100, 20, formatter=lambda v: f"{v}%",
            label_width=44, value_width=40,
        )
        al.addWidget(self.speed_slider, 1)
        layout.addWidget(animation)

        control = QGroupBox("控制")
        cl = QHBoxLayout(control)
        cl.setContentsMargins(6, 6, 6, 6)
        cl.setSpacing(4)
        self.reset_btn = QPushButton("重置")
        self.save_btn = QPushButton("保存")
        cl.addWidget(self.reset_btn)
        cl.addWidget(self.save_btn)
        layout.addWidget(control)
        layout.addStretch(1)

        root.addWidget(panel, 0)
        self.canvas.setFixedSize(600, 600)
        root.addWidget(self.canvas, 1, Qt.AlignCenter)
        self.setCentralWidget(central)
        central.setStyleSheet("background-color: #ffffff;")
        self.setMinimumSize(960, 640)
        self.resize(960, 640)

        self.status = self.statusBar()
        self.status.setSizeGripEnabled(False)
        self.state_label = QLabel("就绪")
        self.params_label = QLabel("N 7 · 实空间 · 14 个衍射峰")
        self.status.addWidget(self.state_label, 1)
        self.status.addPermanentWidget(self.params_label)

        self.n_slider.valueChanged.connect(self._apply)
        self.wave_slider.valueChanged.connect(self._apply)
        self.zoom_slider.valueChanged.connect(self._apply)
        self.rotation_slider.valueChanged.connect(self._apply)
        self.diff_zoom_slider.valueChanged.connect(self._apply)
        self.mode_combo.currentTextChanged.connect(self._apply)
        self.palette_combo.currentTextChanged.connect(self._apply)
        self.log_check.toggled.connect(self._apply)
        self.play_btn.toggled.connect(self._toggle_play)
        self.reset_btn.clicked.connect(self._reset)
        self.save_btn.clicked.connect(self._save)

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._apply()

    def _apply(self, *args):
        self.sim.n = self.n_slider.value()
        self.sim.wave_number = self.wave_slider.value() / 100.0
        self.sim.zoom = self.zoom_slider.value() / 100.0
        self.sim.rotation = np.deg2rad(self.rotation_slider.value())
        self.sim.diffraction_zoom = float(self.diff_zoom_slider.value())
        self.canvas.mode = "diffraction" if self.mode_combo.currentText() == "衍射" else "real"
        self.canvas.palette = self.palette_combo.currentText()
        self.canvas.log = self.log_check.isChecked()
        self.canvas.refresh()
        self._update_status()

    def _update_status(self):
        mode = "衍射" if self.canvas.mode == "diffraction" else "实空间"
        self.params_label.setText(
            f"N {self.sim.n} · {mode} · {2 * int(self.sim.n)} 个衍射峰"
        )

    def _toggle_play(self, checked):
        self.playing = bool(checked)
        self.play_btn.setText("暂停" if checked else "动画")
        self.state_label.setText("运行中" if checked else "暂停")

    def _tick(self):
        if self.playing:
            speed = self.speed_slider.value() / 100.0
            self.sim.phase = (self.sim.phase + 0.12 * speed) % (2.0 * np.pi)
            self.canvas.refresh()

    def _reset(self):
        self.play_btn.setChecked(False)
        self.n_slider.setValue(7)
        self.wave_slider.setValue(500)
        self.zoom_slider.setValue(1200)
        self.rotation_slider.setValue(0)
        self.diff_zoom_slider.setValue(4)
        self.mode_combo.setCurrentText("实空间")
        self.palette_combo.setCurrentText("冰蓝")
        self.log_check.setChecked(True)
        self.sim.phase = 0.0
        self._apply()
        self.state_label.setText("就绪")

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存准晶", "quasicrystal.png", "PNG (*.png)"
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
