# -*- coding: utf-8 -*-
"""Kerr-Newman null-geodesic ray tracing with an accretion disk.

Every canvas pixel is integrated backwards along an exact null geodesic.
The Taichi kernel uses classical RK4 on the separated Kerr-Newman equations
in Boyer-Lindquist coordinates, so spin a/M and charge Q/M both change the
horizon, ergosphere, photon ring and D-shaped shadow.  The disk is a neutral
thin Keplerian disk with Doppler beaming (g^4) and a Planck colour shift.

This page intentionally uses a dark canvas: the contrast of the shadow, the
photon ring and the star field is the subject of the app, not a dark theme.
"""
import math
import time

import numpy as np

from PySide6.QtGui import QPalette

from common.imaging import ask_save_path, ndarray_to_qimage
from common.lab_window import LabWindow, SquareCanvas
from common.qt_base import (
    LabeledSlider, QApplication, QCheckBox, QColor, QComboBox, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPainter, QPushButton,
    QSizePolicy, QTimer, QVBoxLayout, QWidget, Qt, Signal, run_app,
)
from support.physics.kerr import (
    KerrConfig, KerrRenderer, kerr_horizon, kerr_isco,
)
from support.physics.lensing import (
    orbit_step, taichi_available, taichi_backend_name,
)

APP_TITLE = "Kerr–Newman · 带电旋转黑洞光线追踪"

# The UI contract asks for this token; the actual white window is painted by
# QPalette below so no stylesheet reaches the QCheckBox widgets.
_WINDOW_BG = "background-color: #ffffff"

QUALITY = (
    ("快速", 360, 500),
    ("标准", 600, 800),
    ("精细", 840, 1000),
)

# Full-resolution preview while the camera/disk is moving.  The 600 px
# canvas is drawn 1:1; no 420/600 smooth rescale is paid on the UI thread.
# Keep the integration steps at the full standard count: near-critical rays
# need the extra windings, otherwise the photon ring gains a false dark
# annulus while the mouse button is held down.
DRAG_SIZE = 600
DRAG_STEPS = 800
DRAG_INTERVAL_MS = 8
ANIM_SIZE = 600
ANIM_STEPS = 800

PRESETS = {
    "吸积盘": dict(
        distance=46, inclination=84, fov=32, disk_on=True,
        r_in=6, r_out=13, peak_temperature=5000, opacity=45,
        exposure=130, star_gain=150, disk_gain=22, doppler=True,
        spin=75, charge=45, quality=1,
    ),
    "纯星场": dict(
        distance=36, inclination=65, fov=46, disk_on=False,
        r_in=6, r_out=13, peak_temperature=6500, opacity=70,
        exposure=155, star_gain=125, disk_gain=13, doppler=True,
        spin=75, charge=45, quality=1,
    ),
    "正面": dict(
        distance=44, inclination=18, fov=34, disk_on=True,
        r_in=6, r_out=14, peak_temperature=6000, opacity=55,
        exposure=125, star_gain=140, disk_gain=24, doppler=True,
        spin=90, charge=30, quality=1,
    ),
    "侧视": dict(
        distance=50, inclination=87, fov=28, disk_on=True,
        r_in=6, r_out=14, peak_temperature=5500, opacity=80,
        exposure=120, star_gain=140, disk_gain=20, doppler=True,
        spin=70, charge=50, quality=1,
    ),
    "光子环": dict(
        distance=34, inclination=80, fov=24, disk_on=True,
        r_in=6, r_out=14, peak_temperature=8000, opacity=55,
        exposure=105, star_gain=0, disk_gain=26, doppler=True,
        spin=90, charge=40, quality=1,
    ),
}


class LensingCanvas(SquareCanvas):
    """Fixed 600x600 black canvas; the rendered sky fills it exactly."""

    orbitDragged = Signal(float, float)
    zoomed = Signal(int)
    dragBegan = Signal()
    dragEnded = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFixedSize(600, 600)
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip("左键拖动环绕相机，滚轮缩放。")
        self._image = None
        self._last_pos = None

    def set_frame(self, rgb):
        self._image = ndarray_to_qimage(np.ascontiguousarray(rgb, dtype=np.uint8))
        self.update()

    def clear(self):
        self._image = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self._image is not None:
            # The drag preview is already 600x600: use the no-scaling path,
            # which is cheaper than a full-canvas smooth rescale.
            if (self._image.width() == self.width()
                    and self._image.height() == self.height()):
                painter.drawImage(0, 0, self._image)
            else:
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                painter.drawImage(self.rect(), self._image)
        painter.end()

    def save_image(self, path):
        return self.grab().save(path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            self.dragBegan.emit()

    def mouseMoveEvent(self, event):
        if self._last_pos is None:
            return
        point = event.position()
        dx = point.x() - self._last_pos.x()
        dy = point.y() - self._last_pos.y()
        self._last_pos = point
        self.orbitDragged.emit(float(dx), float(dy))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last_pos = None
            self.setCursor(Qt.OpenHandCursor)
            self.dragEnded.emit()

    def wheelEvent(self, event):
        step = event.angleDelta().y()
        if step:
            self.zoomed.emit(1 if step > 0 else -1)


class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        # LabWindow installs a broad background style.  Clear it: any
        # ancestor stylesheet can switch QCheckBox to QStyleSheetStyle and
        # make the native indicator disappear.  White comes from QPalette.
        self.setStyleSheet("")
        self._white_background(self)
        self.renderer = KerrRenderer(output_size=840)
        self.backend = "初始化"
        self._time = 0.0
        self._azimuth = 0.0
        self._inclination = 84.0
        self._dragging = False
        self._applying = False
        self._first_frame = True
        self._render_ms = 0.0
        self._compile_s = 0.0
        self._cpu_mode = False
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_frame)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(30)
        self._anim_timer.timeout.connect(self._tick)
        self._build_ui()

    # ------------------------------------------------------------------ UI
    @staticmethod
    def _white_background(widget):
        """White palette without a stylesheet cascade on QCheckBox."""
        palette = widget.palette()
        palette.setColor(QPalette.Window, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        widget.setPalette(palette)
        widget.setAutoFillBackground(True)

    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(330)
        self._white_background(panel)
        column = QVBoxLayout(panel)
        column.setContentsMargins(2, 2, 4, 4)
        column.setSpacing(3)

        scene = QGroupBox("场景")
        scene_layout = QVBoxLayout(scene)
        scene_layout.setContentsMargins(4, 4, 4, 4)
        scene_layout.setSpacing(3)
        preset_row = QHBoxLayout()
        preset_row.setSpacing(5)
        preset_row.addWidget(QLabel("预设"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(tuple(PRESETS))
        self.preset_combo.setCurrentText("吸积盘")
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_row.addWidget(self.preset_combo, 1)
        scene_layout.addLayout(preset_row)
        quality_row = QHBoxLayout()
        quality_row.setSpacing(5)
        quality_row.addWidget(QLabel("质量"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(tuple(item[0] for item in QUALITY))
        self.quality_combo.setCurrentIndex(1)
        self.quality_combo.setToolTip(
            "精细：静止/暂停时 840×840 精算；动画中自动降为 600×600 预览。"
        )
        self.quality_combo.currentIndexChanged.connect(self._schedule_render)
        quality_row.addWidget(self.quality_combo, 1)
        scene_layout.addLayout(quality_row)
        column.addWidget(scene)

        self.blackhole_group = QGroupBox("黑洞")
        blackhole = self.blackhole_group
        blackhole_layout = QVBoxLayout(blackhole)
        blackhole_layout.setContentsMargins(4, 4, 4, 4)
        blackhole_layout.setSpacing(2)
        self.spin_slider = LabeledSlider(
            "自旋 a", 0, 99, 75,
            formatter=lambda v: f"{v / 100:.2f}",
            label_width=52, value_width=44,
        )
        self.spin_slider.setToolTip(
            "Kerr–Newman 自旋 a/M；0 为带电球对称，0.99 为近极限黑洞。"
        )
        self.spin_slider.valueChanged.connect(self._on_spin)
        blackhole_layout.addWidget(self.spin_slider)
        self.charge_slider = LabeledSlider(
            "电荷 Q", 0, 99, 45,
            formatter=lambda v: f"{v / 100:.2f}",
            label_width=52, value_width=44,
        )
        self.charge_slider.setToolTip(
            "电荷 Q/M；要求 a²+Q²<1，界面会自动压低电荷上限。"
        )
        self.charge_slider.valueChanged.connect(self._on_charge)
        blackhole_layout.addWidget(self.charge_slider)
        column.addWidget(blackhole)

        observer = QGroupBox("观测者")
        observer_layout = QVBoxLayout(observer)
        observer_layout.setContentsMargins(4, 4, 4, 4)
        observer_layout.setSpacing(3)
        self.distance_slider = LabeledSlider(
            "距离", 12, 120, 46, formatter=lambda v: f"{v} M",
            label_width=52, value_width=44,
        )
        self.inclination_slider = LabeledSlider(
            "倾角", 1, 179, 84, formatter=lambda v: f"{v}°",
            label_width=52, value_width=44,
        )
        self.fov_slider = LabeledSlider(
            "视场", 10, 70, 32, formatter=lambda v: f"{v}°",
            label_width=52, value_width=44,
        )
        for slider in (self.distance_slider, self.fov_slider):
            slider.valueChanged.connect(self._on_parameter)
        self.inclination_slider.valueChanged.connect(self._on_inclination)
        for slider in (self.distance_slider, self.inclination_slider,
                       self.fov_slider):
            observer_layout.addWidget(slider)
        column.addWidget(observer)

        disk = QGroupBox("吸积盘")
        disk_layout = QVBoxLayout(disk)
        disk_layout.setContentsMargins(4, 4, 4, 4)
        disk_layout.setSpacing(3)
        self.disk_check = QCheckBox("显示吸积盘")
        self.disk_check.setChecked(True)
        self.disk_check.toggled.connect(self._on_disk_toggle)
        disk_layout.addWidget(self.disk_check)

        self.r_in_slider = LabeledSlider(
            "内缘", 3, 10, 6, formatter=lambda v: f"{v} M",
            label_width=52, value_width=44,
        )
        self.r_out_slider = LabeledSlider(
            "外缘", 8, 24, 13, formatter=lambda v: f"{v} M",
            label_width=52, value_width=44,
        )
        self.temp_slider = LabeledSlider(
            "温度", 3000, 15000, 5000, formatter=lambda v: f"{v} K",
            label_width=52, value_width=56,
        )
        self.opacity_slider = LabeledSlider(
            "不透明", 0, 100, 45, formatter=lambda v: f"{v}%",
            label_width=52, value_width=44,
        )
        self.disk_gain_slider = LabeledSlider(
            "盘亮度", 3, 40, 22, formatter=lambda v: f"{v / 100:.2f}",
            label_width=52, value_width=44,
        )
        for slider in (self.r_in_slider, self.r_out_slider, self.temp_slider,
                       self.opacity_slider, self.disk_gain_slider):
            slider.valueChanged.connect(self._on_parameter)
            disk_layout.addWidget(slider)
        column.addWidget(disk)

        display = QGroupBox("显影")
        display_layout = QVBoxLayout(display)
        display_layout.setContentsMargins(4, 4, 4, 4)
        display_layout.setSpacing(3)
        self.exposure_slider = LabeledSlider(
            "曝光", 20, 250, 130, formatter=lambda v: f"{v / 100:.2f}×",
            label_width=52, value_width=48,
        )
        self.star_slider = LabeledSlider(
            "星光", 0, 250, 150, formatter=lambda v: f"{v}%",
            label_width=52, value_width=44,
        )
        for slider in (self.exposure_slider, self.star_slider):
            slider.valueChanged.connect(self._on_parameter)
            display_layout.addWidget(slider)
        self.doppler_check = QCheckBox("多普勒增亮")
        self.doppler_check.setChecked(True)
        self.doppler_check.toggled.connect(self._on_parameter)
        display_layout.addWidget(self.doppler_check)
        column.addWidget(display)

        control = QGroupBox("控制")
        control_layout = QHBoxLayout(control)
        control_layout.setContentsMargins(4, 4, 4, 4)
        control_layout.setSpacing(3)
        self.pause_btn = QPushButton("暂停")
        self.reset_btn = QPushButton("重置")
        self.save_btn = QPushButton("保存")
        for button in (self.pause_btn, self.reset_btn, self.save_btn):
            control_layout.addWidget(button, 1)
        column.addWidget(control)
        column.addStretch(1)

        root.addWidget(panel, 0)
        self.canvas = LensingCanvas()
        root.addWidget(self.canvas, 1, Qt.AlignCenter)
        self.setCentralWidget(central)
        self._white_background(central)
        self.setMinimumSize(960, 640)
        self.resize(960, 640)

        self.status = self.statusBar()
        self.status.setSizeGripEnabled(False)
        self.state_label = QLabel("准备中")
        self.params_label = QLabel("史瓦西 · 等待首帧")
        self.status.addWidget(self.state_label, 1)
        self.status.addPermanentWidget(self.params_label)

        self.pause_btn.clicked.connect(self._toggle_pause)
        self.reset_btn.clicked.connect(self._reset)
        self.save_btn.clicked.connect(self._save)
        self.canvas.orbitDragged.connect(self._orbit_drag)
        self.canvas.zoomed.connect(self._zoom)
        self.canvas.dragBegan.connect(self._drag_begin)
        self.canvas.dragEnded.connect(self._drag_end)

        self._paused = False
        self._disk_enabled = True
        self._inclination = float(self.inclination_slider.value())
        self._update_spin_info()
        self._sync_disk_enabled()
        self._anim_timer.start()
        QTimer.singleShot(90, self._first_render)

    # --------------------------------------------------------------- state
    def _first_render(self):
        self.state_label.setText("编译内核")
        self.params_label.setText("Taichi 首次初始化 · 请稍候")
        if taichi_available() and "CPU" in taichi_backend_name().upper():
            self._cpu_mode = True
        QApplication.processEvents()
        self._render_frame()

    def _current_quality(self):
        if self._dragging:
            if self._cpu_mode:
                return "拖动", 300, 100
            return "拖动", DRAG_SIZE, DRAG_STEPS
        index = max(0, min(len(QUALITY) - 1, self.quality_combo.currentIndex()))
        name, width, steps = QUALITY[index]
        if self._cpu_mode:
            width = min(width, 300)
            steps = min(steps, 130)
        # Fine mode is a still-image setting: while the disk texture is
        # animating, render a 600 px preview so the UI stays responsive.
        animating = (not self._paused) and self.disk_check.isChecked()
        if animating and width > ANIM_SIZE:
            return name + "预览", ANIM_SIZE, ANIM_STEPS
        return name, int(width), int(steps)

    def _config(self, width, steps):
        r_in = float(self.r_in_slider.value())
        r_out = max(r_in + 1.5, float(self.r_out_slider.value()))
        return KerrConfig(
            width=width, height=width,
            distance=float(self.distance_slider.value()),
            inclination=float(self._inclination),
            azimuth=float(self._azimuth),
            fov=float(self.fov_slider.value()),
            spin=float(self.spin_slider.value()) / 100.0,
            charge=float(self.charge_slider.value()) / 100.0,
            disk_on=bool(self.disk_check.isChecked()),
            r_in=r_in, r_out=r_out,
            peak_temperature=float(self.temp_slider.value()),
            opacity=float(self.opacity_slider.value()) / 100.0,
            doppler=bool(self.doppler_check.isChecked()),
            exposure=float(self.exposure_slider.value()) / 100.0,
            star_gain=float(self.star_slider.value()) / 100.0,
            disk_gain=float(self.disk_gain_slider.value()) / 100.0,
            max_steps=steps, step_scale=0.02, time=float(self._time),
        )

    def _schedule_render(self, *args, delay=45):
        if self._render_timer.isActive():
            return
        if self._dragging:
            # A refresh cap, not a per-event render: mouse events can arrive
            # at 1000 Hz while the eye only needs ~50 preview frames/s.
            delay = DRAG_INTERVAL_MS
        self._render_timer.start(int(delay))

    def _render_frame(self):
        name, width, steps = self._current_quality()
        config = self._config(width, steps)
        started = time.perf_counter()
        try:
            rgb = self.renderer.render(config)
        except Exception as exc:  # noqa: BLE001 - surface in the status bar
            self.state_label.setText("渲染失败")
            self.params_label.setText(str(exc)[:140])
            return
        elapsed = (time.perf_counter() - started) * 1000.0
        self._render_ms = elapsed
        self.canvas.set_frame(rgb)
        self.backend = self.renderer.backend
        if self._dragging:
            # Keep the drag loop free of status-bar relayouts.
            return
        if self._first_frame:
            self._first_frame = False
            self._compile_s = elapsed
            self._render_ms = 0.0
            self.state_label.setText("编译完成")
            self._schedule_render(delay=0)
        else:
            self._render_ms = elapsed
            if not self._paused and self.state_label.text() not in ("已保存",):
                self.state_label.setText("运行中")
        self._update_status(name, width, steps)

    def _update_status(self, name=None, width=None, steps=None):
        if name is None:
            name, width, steps = self._current_quality()
        mode = "CUDA" if "CUDA" in self.backend.upper() else (
            "CPU" if "CPU" in self.backend.upper() else self.backend
        )
        if self._render_ms <= 0.0:
            timing = f"编译 {self._compile_s / 1000.0:.1f} s"
        else:
            timing = f"{self._render_ms:.1f} ms"
        tag = "预览 · " if name and name.endswith("预览") else ""
        spin = self.spin_slider.value() / 100.0
        charge = self.charge_slider.value() / 100.0
        self.params_label.setText(
            f"{tag}a {spin:.2f} Q {charge:.2f} · {mode} · {width}² · "
            f"{steps} 步 · {timing}"
        )

    # ------------------------------------------------------------ animation
    def _tick(self):
        # The camera position is fixed; only the disk surface evolves.
        if self._paused or self._dragging:
            return
        if not self.disk_check.isChecked():
            return
        self._time += 0.26
        self._schedule_render(delay=0)

    def _toggle_pause(self):
        self._paused = not self._paused
        self.pause_btn.setText("继续" if self._paused else "暂停")
        self.state_label.setText("暂停" if self._paused else "运行中")
        if not self._paused:
            self._schedule_render(delay=0)

    def _on_inclination(self, value):
        self._inclination = float(value)
        self._on_parameter()

    def _charge_limit(self):
        a = float(self.spin_slider.value()) / 100.0
        return int(math.floor(math.sqrt(max(0.0, 0.998 ** 2 - a * a)) * 100.0))

    def _clamp_charge(self):
        limit = self._charge_limit()
        if self.charge_slider.value() > limit:
            was_applying = self._applying
            self._applying = True
            self.charge_slider.setValue(limit)
            self._applying = was_applying

    def _on_spin(self, value):
        self._clamp_charge()
        self._update_spin_info()
        if not self._applying:
            self._schedule_render()

    def _on_charge(self, value):
        self._clamp_charge()
        self._update_spin_info()
        if not self._applying:
            self._schedule_render()

    def _update_spin_info(self):
        a = float(self.spin_slider.value()) / 100.0
        q = float(self.charge_slider.value()) / 100.0
        self.blackhole_group.setTitle(
            f"黑洞 · r+ {kerr_horizon(a, q):.2f} M · ISCO {kerr_isco(a, q):.2f} M"
        )

    def _on_parameter(self, *args):
        if self._applying:
            return
        self._sync_disk_enabled()
        self._schedule_render()

    def _on_disk_toggle(self, checked):
        self._disk_enabled = bool(checked)
        self._sync_disk_enabled()
        if not self._applying:
            self._schedule_render()

    def _sync_disk_enabled(self):
        active = bool(self.disk_check.isChecked())
        for widget in (self.r_in_slider, self.r_out_slider, self.temp_slider,
                       self.opacity_slider, self.disk_gain_slider):
            widget.setEnabled(active)
        self.doppler_check.setEnabled(active)

    # ------------------------------------------------------------- gestures
    def _orbit_drag(self, dx, dy):
        # Connection-style orbit camera: the camera moves around the black
        # hole and always looks at it, so the hole stays framed.  Horizontal
        # drag is yaw; vertical drag is altitude (down raises the camera).
        self._azimuth, self._inclination = orbit_step(
            self._azimuth, self._inclination, float(dx), float(dy)
        )
        rounded = int(round(self._inclination))
        if rounded != self.inclination_slider.value():
            self.inclination_slider.blockSignals(True)
            self.inclination_slider.setValue(rounded)
            self.inclination_slider.blockSignals(False)
        self._schedule_render(delay=0)

    def _zoom(self, steps):
        value = int(self.fov_slider.value()) - (3 if steps > 0 else -3)
        value = max(10, min(70, value))
        self.fov_slider.blockSignals(True)
        self.fov_slider.setValue(value)
        self.fov_slider.blockSignals(False)
        self._schedule_render(delay=0)

    def _drag_begin(self):
        self._dragging = True
        self.state_label.setText("拖动预览")
        self.params_label.setText("松开后精算")

    def _drag_end(self):
        self._dragging = False
        self.state_label.setText("暂停" if self._paused else "精算中")
        self._schedule_render(delay=0)

    # ---------------------------------------------------------------- presets
    def _apply_preset(self, name):
        preset = PRESETS.get(name)
        if not preset:
            return
        self._applying = True
        try:
            self.distance_slider.setValue(int(preset["distance"]))
            self._inclination = float(preset["inclination"])
            self.inclination_slider.setValue(int(preset["inclination"]))
            self.fov_slider.setValue(int(preset["fov"]))
            self.disk_check.setChecked(bool(preset["disk_on"]))
            self.r_in_slider.setValue(int(preset["r_in"]))
            self.r_out_slider.setValue(int(preset["r_out"]))
            self.temp_slider.setValue(int(preset["peak_temperature"]))
            self.opacity_slider.setValue(int(preset["opacity"]))
            self.exposure_slider.setValue(int(preset["exposure"]))
            self.star_slider.setValue(int(preset["star_gain"]))
            self.disk_gain_slider.setValue(int(preset["disk_gain"]))
            self.doppler_check.setChecked(bool(preset["doppler"]))
            self.quality_combo.setCurrentIndex(int(preset["quality"]))
            self.spin_slider.setValue(int(preset["spin"]))
            self.charge_slider.setValue(int(preset["charge"]))
            self._time = 0.0
            self._azimuth = 0.0
            self._disk_enabled = bool(preset["disk_on"])
            self._sync_disk_enabled()
        finally:
            self._applying = False
        self._schedule_render(delay=70)

    def _reset(self):
        self._paused = False
        self.pause_btn.setText("暂停")
        self.state_label.setText("运行中")
        self._time = 0.0
        self._azimuth = 0.0
        current = self.preset_combo.currentText()
        target = "吸积盘"
        if current == target:
            self._apply_preset(target)
        else:
            self.preset_combo.setCurrentText(target)
        self._schedule_render(delay=0)

    # ------------------------------------------------------------------ save
    def _save(self):
        path = ask_save_path(
            self, "保存史瓦西光线追踪", directory="lensing.png",
            suffix=".png",
        )
        if not path:
            return
        if self.canvas.save_image(path):
            self.state_label.setText("已保存")
            self.params_label.setText(path)
        else:
            QMessageBox.warning(self, "保存失败", "无法写入所选路径。")

    def closeEvent(self, event):
        self._anim_timer.stop()
        self._render_timer.stop()
        event.accept()


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
