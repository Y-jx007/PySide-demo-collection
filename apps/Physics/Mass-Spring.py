# -*- coding: utf-8 -*-
"""Mass-spring network: explicit integration on a white canvas."""
import time

from PySide6.QtCore import Signal

from common.lab_window import LabWindow, SquareCanvas
from common.qt_base import (
    LabeledSlider, QColor, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPainter, QPen, QPointF, QPushButton,
    QRectF, QSizePolicy, QTimer, QVBoxLayout, QWidget, Qt, run_app,
)
from support.physics.mass_spring import MassSpring


class SpringCanvas(SquareCanvas):
    changed = Signal()

    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim = sim
        self._attract = None
        self.setMinimumSize(520, 520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _rect(self):
        return QRectF(0, 0, self.width(), self.height())

    def _to_sim(self, point):
        rect = self._rect()
        x = (point.x() - rect.left()) / max(1.0, rect.width())
        y = 1.0 - (point.y() - rect.top()) / max(1.0, rect.height())
        return float(min(max(x, 0.005), 0.995)), float(min(max(y, 0.005), 0.995))

    def _to_screen(self, x, y):
        rect = self._rect()
        return QPointF(
            rect.left() + float(x) * rect.width(),
            rect.bottom() - float(y) * rect.height(),
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        rect = self._rect()
        painter.setPen(QPen(QColor(226, 229, 234), 1))
        for i in range(1, 10):
            x = rect.left() + rect.width() * i / 10.0
            y = rect.top() + rect.height() * i / 10.0
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        positions = self.sim.positions()
        a, b, rest = self.sim.springs()
        for i in range(len(a)):
            pa = self._to_screen(*positions[a[i]])
            pb = self._to_screen(*positions[b[i]])
            dist = float(((positions[a[i]] - positions[b[i]]) ** 2).sum() ** 0.5)
            tension = min(2.0, abs(dist - float(rest[i])) / max(float(rest[i]), 1e-6))
            tension /= 2.0
            painter.setPen(QPen(QColor(
                int(35 + 120 * tension),
                int(75 + 65 * tension),
                int(155 - 80 * tension),
                200,
            ), 1.2))
            painter.drawLine(pa, pb)

        fixed = self.sim.fixed_mask()
        for i, (x, y) in enumerate(positions):
            p = self._to_screen(x, y)
            color = QColor("#c0392b") if fixed[i] else QColor("#23384f")
            glow = QColor(color)
            glow.setAlpha(28)
            painter.setPen(Qt.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(p, 7.0, 7.0)
            painter.setBrush(color)
            painter.drawEllipse(p, 4.0, 4.0)
            painter.setPen(QPen(QColor(255, 255, 255, 220), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(p, 4.0, 4.0)

        if self._attract is not None:
            p = self._to_screen(*self._attract)
            painter.setPen(QPen(QColor(29, 78, 216, 210), 1.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(p, 14.0, 14.0)
        painter.end()

    def mousePressEvent(self, event):
        x, y = self._to_sim(event.position())
        if event.button() == Qt.LeftButton:
            fixed = bool(event.modifiers() & Qt.ShiftModifier)
            self.sim.add_particle(x, y, fixed=fixed, connect=True)
            self.changed.emit()
            self.update()
        elif event.button() == Qt.RightButton:
            self._attract = (x, y)
            self.sim.attract(x, y)
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.RightButton:
            x, y = self._to_sim(event.position())
            self._attract = (x, y)
            self.sim.attract(x, y)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._attract = None
            self.update()

    def save_image(self, path):
        return self.grab().save(path)


class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #ffffff;")
        self.setWindowTitle("Hooke · 弹簧网络")
        self.sim = MassSpring()
        self.sim.reset("悬索桥")
        self.canvas = SpringCanvas(self.sim)
        self.paused = False
        self._frames = 0
        self._fps = 0.0
        self._fps_since = time.perf_counter()

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

        physics = QGroupBox("物理")
        pl = QVBoxLayout(physics)
        pl.setContentsMargins(6, 6, 6, 6)
        pl.setSpacing(5)
        self.gravity_slider = LabeledSlider(
            "重力", 0, 300, 98, formatter=lambda v: f"{v / 10:.1f}",
            label_width=52, value_width=44,
        )
        self.stiffness_slider = LabeledSlider(
            "刚度", 100, 6000, 1800, formatter=str,
            label_width=52, value_width=44,
        )
        self.dashpot_slider = LabeledSlider(
            "阻尼", 0, 100, 22, formatter=str,
            label_width=52, value_width=44,
        )
        self.drag_slider = LabeledSlider(
            "空气", 0, 500, 35, formatter=lambda v: f"{v / 100:.2f}",
            label_width=52, value_width=44,
        )
        for slider in (self.gravity_slider, self.stiffness_slider,
                       self.dashpot_slider, self.drag_slider):
            pl.addWidget(slider)
        layout.addWidget(physics)

        structure = QGroupBox("结构")
        sl = QVBoxLayout(structure)
        sl.setContentsMargins(6, 6, 6, 6)
        sl.setSpacing(5)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("预设"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(("悬索桥", "网", "链", "随机"))
        preset_row.addWidget(self.preset_combo, 1)
        sl.addLayout(preset_row)
        self.radius_slider = LabeledSlider(
            "连接半径", 50, 250, 130, formatter=lambda v: f"{v / 1000:.2f}",
            label_width=64, value_width=44,
        )
        sl.addWidget(self.radius_slider)
        layout.addWidget(structure)

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
        self.setCentralWidget(central)
        central.setStyleSheet("background-color: #ffffff;")
        self.setMinimumSize(960, 640)
        self.resize(960, 640)

        self.status = self.statusBar()
        self.status.setSizeGripEnabled(False)
        self.state_label = QLabel("运行中")
        self.params_label = QLabel("N 15 · 弹簧 27 · 0 fps")
        self.status.addWidget(self.state_label, 1)
        self.status.addPermanentWidget(self.params_label)

        self.gravity_slider.valueChanged.connect(self._apply_physics)
        self.stiffness_slider.valueChanged.connect(self._apply_physics)
        self.dashpot_slider.valueChanged.connect(self._apply_physics)
        self.drag_slider.valueChanged.connect(self._apply_physics)
        self.radius_slider.valueChanged.connect(self._apply_physics)
        self.preset_combo.currentTextChanged.connect(self._reset)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.reset_btn.clicked.connect(self._reset)
        self.save_btn.clicked.connect(self._save)
        self.canvas.changed.connect(self._update_status)

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._apply_physics()

    def _apply_physics(self, *args):
        self.sim.gravity = self.gravity_slider.value() / 10.0
        self.sim.stiffness = float(self.stiffness_slider.value())
        self.sim.dashpot = float(self.dashpot_slider.value())
        self.sim.drag = self.drag_slider.value() / 100.0
        self.sim.connect_radius = self.radius_slider.value() / 1000.0

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
        elapsed = now - self._fps_since
        if elapsed >= 0.6:
            self._fps = self._frames / elapsed
            self._frames = 0
            self._fps_since = now
            self._update_status()

    def _update_status(self):
        count, springs = self.sim.count()
        self.params_label.setText(f"N {count} · 弹簧 {springs} · {self._fps:.0f} fps")

    def _reset(self, *args):
        self.sim.reset(self.preset_combo.currentText())
        self.canvas.update()
        self._update_status()

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存Hooke · 弹簧网络", "mass-spring.png", "PNG (*.png)"
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
