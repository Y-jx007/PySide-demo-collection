# -*- coding: utf-8 -*-
"""Kerr-Newman null-geodesic ray tracing with an accretion disk.

Every canvas pixel is integrated backwards along a null geodesic.
The Taichi kernel integrates the Kerr-Newman equations in Kerr-Schild
Cartesian coordinates with classical RK4.  Kerr-Schild is regular on the polar
axis, so the renderer has no 1/sin^2(theta) coordinate singularity (the old
Boyer-Lindquist kernel painted a false vertical column at the image centre).
Spin a/M and charge Q/M change the horizon, ergosphere, photon ring and
D-shaped shadow.  The disk is a neutral
thin Keplerian disk with Doppler beaming (g^4) and a Planck colour shift.

This page intentionally uses a dark canvas: the contrast of the shadow, the
photon ring and the star field is the subject of the app, not a dark theme.
"""
import math
import time

import numpy as np

from PySide6.QtCore import QPointF
from PySide6.QtGui import (
    QBrush, QFont, QKeySequence, QPainterPath, QPalette, QPen, QShortcut,
)
from PySide6.QtWidgets import QTabWidget

from common.imaging import ask_save_path, ndarray_to_qimage
from common.lab_window import LabWindow, SquareCanvas
from common.qt_base import (
    LabeledSlider, QApplication, QCheckBox, QColor, QComboBox, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPainter, QPushButton, QSettings,
    QSizePolicy, QTimer, QVBoxLayout, QWidget, Qt, Signal, run_app,
)
from support.physics.kerr import (
    KN_MAX, KerrConfig, KerrRenderer, kerr_horizon, kerr_ibco,
    kerr_isco, kerr_photon_radius,
)
from support.physics.lensing import (
    orbit_step, taichi_available, taichi_backend_name,
)

APP_TITLE = "Kerr–Newman · 带电旋转黑洞光线追踪"

# This page is a documented dark-canvas exception (see docs/UI_GUIDELINES.md);
# the marker lets common/ui_style.py skip the white-window token.  The white
# panels themselves are painted with QPalette so no stylesheet reaches the
# QCheckBox widgets (a QStyleSheetStyle would hide their native indicator).
UI_THEME = "dark_canvas"

QUALITY = (
    ("快速", 360, 800),
    ("标准", 600, 800),
    ("精细", 840, 1600),
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
        inclination=84, fov=32, disk_on=True,
        r_in=6, r_out=13, peak_temperature=5000, opacity=45,
        exposure=130, star_gain=150, disk_gain=22, doppler=True,
        spin=75, charge=45, quality=1,
    ),
    "逆行": dict(
        inclination=84, fov=32, disk_on=True,
        r_in=6, r_out=13, peak_temperature=5000, opacity=45,
        exposure=130, star_gain=150, disk_gain=22, doppler=True,
        spin=-75, charge=45, quality=1,
    ),
    "负电荷": dict(
        inclination=84, fov=32, disk_on=True,
        r_in=6, r_out=13, peak_temperature=5000, opacity=45,
        exposure=130, star_gain=150, disk_gain=22, doppler=True,
        spin=75, charge=-45, quality=1,
    ),
    "纯星场": dict(
        inclination=65, fov=46, disk_on=False,
        r_in=6, r_out=13, peak_temperature=6500, opacity=70,
        exposure=155, star_gain=125, disk_gain=13, doppler=True,
        spin=75, charge=45, quality=1,
    ),
    "正面": dict(
        inclination=18, fov=34, disk_on=True,
        r_in=6, r_out=14, peak_temperature=6000, opacity=55,
        exposure=125, star_gain=140, disk_gain=24, doppler=True,
        spin=90, charge=30, quality=1,
    ),
    "侧视": dict(
        inclination=87, fov=28, disk_on=True,
        r_in=6, r_out=14, peak_temperature=5500, opacity=80,
        exposure=120, star_gain=140, disk_gain=20, doppler=True,
        spin=70, charge=50, quality=1,
    ),
    "光子环": dict(
        inclination=80, fov=24, disk_on=True,
        r_in=6, r_out=14, peak_temperature=8000, opacity=55,
        exposure=105, star_gain=0, disk_gain=26, doppler=True,
        spin=90, charge=40, quality=1,
    ),
}


class LensingCanvas(SquareCanvas):
    """Fixed 640x640 black canvas; the rendered sky fills it exactly.

    640 is the documented expanded size: the control panel needs more height
    than the 600/960x640 default, so the window grows instead of scrolling.
    """

    orbitDragged = Signal(float, float)
    zoomed = Signal(int)
    dragBegan = Signal()
    dragEnded = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFixedSize(640, 640)
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip("左键拖动环绕相机，滚轮缩放。")
        self._image = None
        self._last_pos = None

    def set_frame(self, rgb):
        image = ndarray_to_qimage(np.ascontiguousarray(rgb, dtype=np.uint8))
        # A 840 px fine frame is painted into the 600 px canvas at 1.4x DPR,
        # so the widget keeps the native pixels on a HiDPI screen instead of
        # upscaling a 600 px image.
        if self.width() > 0:
            image.setDevicePixelRatio(image.width() / float(self.width()))
        self._image = image
        self.update()

    def clear(self):
        self._image = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self._image is not None:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            # The image's devicePixelRatio maps its native pixels onto the
            # 600 px logical canvas; no extra rect scaling is needed.
            painter.drawImage(0, 0, self._image)
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



class PlaneDiagram(QWidget):
    """Top-down equatorial-plane schematic.

    Shows the event horizon, the co-rotating photon ring, the ISCO, and the
    inner/outer disk edges.  It is a schematic (not a ray-traced image), so it
    can display radii that are hidden inside the shadow.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 286)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._a = 0.75
        self._q = 0.45
        self._rin = 6.0
        self._rout = 13.0
        self._disk = True

    def set_params(self, a, q, r_in, r_out, disk_on):
        values = (float(a), float(q), float(r_in), float(r_out), bool(disk_on))
        if values == (self._a, self._q, self._rin, self._rout, self._disk):
            return
        self._a, self._q, self._rin, self._rout, self._disk = values
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#0b0e14"))
        w, h = self.width(), self.height()
        a, q = self._a, self._q
        rh = kerr_horizon(a, q)
        isco = kerr_isco(a, q)
        isco_r = kerr_isco(a, q, retrograde=True)
        rph = kerr_photon_radius(a, q)
        rph_r = kerr_photon_radius(a, q, retrograde=True)
        ibco = kerr_ibco(a, q)
        rmax = max(16.0, self._rout + 3.0, isco_r + 2.0,
                   rph_r + 2.0, ibco + 1.0, 12.0)
        bottom = 5 * 13 + 10
        avail = max(80.0, h - bottom - 28.0)
        cx = w / 2.0
        cy = 28.0 + avail * 0.5
        radius = max(60.0, min(w * 0.5 - 14.0, avail * 0.5))
        scale = radius / rmax

        # radial grid
        p.setPen(QPen(QColor(255, 255, 255, 16), 1.0))
        for rr in (5.0, 10.0, 15.0, 20.0):
            if rr < rmax:
                p.drawEllipse(QPointF(cx, cy), rr * scale, rr * scale)

        # disk annulus
        if self._disk:
            path = QPainterPath()
            path.setFillRule(Qt.OddEvenFill)
            path.addEllipse(QPointF(cx, cy), self._rout * scale,
                            self._rout * scale)
            path.addEllipse(QPointF(cx, cy), self._rin * scale,
                            self._rin * scale)
            p.fillPath(path, QColor(232, 162, 90, 62))
            p.setPen(QPen(QColor(232, 162, 90, 190), 1.2))
            p.drawEllipse(QPointF(cx, cy), self._rout * scale,
                          self._rout * scale)
            p.drawEllipse(QPointF(cx, cy), self._rin * scale,
                          self._rin * scale)

        # counter-rotating orbits (thin), then co-rotating (thick)
        p.setPen(QPen(QColor("#4f9fbf"), 1.0, Qt.DotLine))
        p.drawEllipse(QPointF(cx, cy), rph_r * scale, rph_r * scale)
        p.setPen(QPen(QColor("#b9782f"), 1.0, Qt.DashLine))
        p.drawEllipse(QPointF(cx, cy), isco_r * scale, isco_r * scale)
        p.setPen(QPen(QColor("#7fd4ff"), 1.5, Qt.DotLine))
        p.drawEllipse(QPointF(cx, cy), rph * scale, rph * scale)
        p.setPen(QPen(QColor("#ffd166"), 1.5, Qt.DashLine))
        p.drawEllipse(QPointF(cx, cy), isco * scale, isco * scale)
        p.setPen(QPen(QColor("#7ee0a0"), 1.2, Qt.DashDotLine))
        p.drawEllipse(QPointF(cx, cy), ibco * scale, ibco * scale)

        # horizon
        p.setBrush(QBrush(QColor("#000000")))
        p.setPen(QPen(QColor("#9aa3b5"), 1.5))
        p.drawEllipse(QPointF(cx, cy), rh * scale, rh * scale)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#3a4150"), 1.0))
        p.drawEllipse(QPointF(cx, cy), 1.5, 1.5)

        p.setFont(QFont("Microsoft YaHei", 8))
        p.setPen(QColor("#e6e8ee"))
        p.drawText(10, 18, f"赤道面   a={a:+.2f}   Q={q:+.2f}")
        if abs(a) < 5e-3:
            spin_txt = "自旋：无"
        elif a > 0.0:
            spin_txt = "自旋：+φ（逆时针）"
        else:
            spin_txt = "自旋：−φ（顺时针）"
        p.setPen(QColor("#aeb4c2"))
        p.drawText(10, 32, spin_txt)

        # legend
        y = h - bottom + 10
        p.setFont(QFont("Microsoft YaHei", 8))
        p.setPen(QPen(QColor("#9aa3b5"), 1.4))
        p.drawLine(10, y - 4, 26, y - 4)
        p.setPen(QColor("#cfd3dc"))
        p.drawText(32, y, f"视界 r+ = {rh:.2f} M")
        p.setPen(QPen(QColor("#7fd4ff"), 1.4, Qt.DotLine))
        p.drawLine(10, y + 9, 26, y + 9)
        p.setPen(QColor("#cfd3dc"))
        p.drawText(32, y + 13,
                   f"光子环 顺 {rph:.2f} / 逆 {rph_r:.2f} M")
        p.setPen(QPen(QColor("#ffd166"), 1.4, Qt.DashLine))
        p.drawLine(10, y + 22, 26, y + 22)
        p.setPen(QColor("#cfd3dc"))
        p.drawText(32, y + 26,
                   f"ISCO 顺 {isco:.2f} / 逆 {isco_r:.2f} M")
        p.setPen(QPen(QColor("#7ee0a0"), 1.4, Qt.DashDotLine))
        p.drawLine(10, y + 35, 26, y + 35)
        p.setPen(QColor("#cfd3dc"))
        p.drawText(32, y + 39, f"IBCO {ibco:.2f} M")
        p.setPen(QColor("#8f96a6"))
        p.drawText(10, y + 53, "OSCO：Kerr–Newman 无有限值（稳定轨道 → ∞）")


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
        self._syncing = False
        self._last_rgb = None
        self._save_msg_until = 0.0
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
        self._orbit_timer = QTimer(self)
        self._orbit_timer.setInterval(50)
        self._orbit_timer.timeout.connect(self._auto_orbit_tick)
        self._build_ui()
        self._load_settings()

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
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        self.tabs = QTabWidget()
        controls = QWidget()
        self._white_background(controls)
        column = QVBoxLayout(controls)
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
            "自旋 a", -99, 99, 75,
            formatter=lambda v: f"{v / 100:.2f}",
            label_width=52, value_width=44,
        )
        self.spin_slider.setToolTip(
            "Kerr–Newman 自旋 a/M：正值 +φ，负值 −φ（反向旋转），"
            "0 为不旋转；0.99 为近极限黑洞。"
        )
        self.spin_slider.valueChanged.connect(self._on_spin)
        blackhole_layout.addWidget(self.spin_slider)
        self.charge_slider = LabeledSlider(
            "电荷 Q", -99, 99, 45,
            formatter=lambda v: f"{v / 100:.2f}",
            label_width=52, value_width=44,
        )
        self.charge_slider.setToolTip(
            "电荷 Q/M，可正可负；始终可拖动。a²+Q² 超限时会自动调整"
            "另一个滑块，而不是把电荷滑块卡住。"
        )
        self.charge_slider.valueChanged.connect(self._on_charge)
        blackhole_layout.addWidget(self.charge_slider)
        self.mass_slider = LabeledSlider(
            "质量 M", 0, 1000, 500,
            formatter=lambda v: (
                f"{10.0 ** (6.0 + 4.0 * v / 1000.0):.1e} M☉"),
            label_width=52, value_width=64,
        )
        self.mass_slider.setToolTip(
            "质量连续可调（10^6–10^10 M☉）。M 同时改变视在大小"
            "（等效 FOV 缩放）和固定吸积率下的盘温 T∝M^-1/4；"
            "几何形状本身仍以 M 为单位。"
        )
        self.mass_slider.valueChanged.connect(self._on_mass)
        blackhole_layout.addWidget(self.mass_slider)
        self.mass_info = QLabel("r_g —")
        self.mass_info.setStyleSheet("color:#555; font-size:10px;")
        blackhole_layout.addWidget(self.mass_info)
        column.addWidget(blackhole)

        observer = QGroupBox("观测者")
        observer_layout = QVBoxLayout(observer)
        observer_layout.setContentsMargins(4, 4, 4, 4)
        observer_layout.setSpacing(3)
        self.inclination_slider = LabeledSlider(
            "倾角", 1, 179, 84, formatter=lambda v: f"{v}°",
            label_width=52, value_width=44,
        )
        self.fov_slider = LabeledSlider(
            "视场", 5, 120, 32, formatter=lambda v: f"{v}°",
            label_width=52, value_width=44,
        )
        self.fov_slider.valueChanged.connect(self._on_parameter)
        self.inclination_slider.valueChanged.connect(self._on_inclination)
        for slider in (self.inclination_slider, self.fov_slider):
            observer_layout.addWidget(slider)
        self.auto_orbit_check = QCheckBox("自动环绕")
        self.auto_orbit_check.setToolTip(
            "相机绕黑洞自动旋转；与吸积盘动画的暂停/播放相互独立。"
        )
        self.auto_orbit_check.toggled.connect(self._on_auto_orbit)
        observer_layout.addWidget(self.auto_orbit_check)
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
            "温度", 1500, 30000, 5000, formatter=lambda v: f"{v} K",
            label_width=52, value_width=56,
        )
        self.opacity_slider = LabeledSlider(
            "不透明", 0, 100, 45, formatter=lambda v: f"{v}%",
            label_width=52, value_width=44,
        )
        self.disk_gain_slider = LabeledSlider(
            "盘亮度", 1, 200, 22, formatter=lambda v: f"{v / 100:.2f}",
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
            "曝光", 5, 400, 130, formatter=lambda v: f"{v / 100:.2f}×",
            label_width=52, value_width=48,
        )
        self.star_slider = LabeledSlider(
            "星光", 0, 400, 150, formatter=lambda v: f"{v}%",
            label_width=52, value_width=44,
        )
        self.star_density_slider = LabeledSlider(
            "星密", 50, 300, 200, formatter=lambda v: f"{v / 100:.1f}×",
            label_width=52, value_width=44,
        )
        self.star_density_slider.setToolTip(
            "星场密度倍数。近黑洞的透镜拉伸在密度高时更明显。"
        )
        for slider in (self.exposure_slider, self.star_slider,
                       self.star_density_slider):
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
        self.reset_btn = QPushButton("重置时间")
        self.reset_btn.setToolTip("只重置吸积盘的时间相位，不改变任何参数。")
        self.save_btn = QPushButton("保存")
        for button in (self.pause_btn, self.reset_btn, self.save_btn):
            control_layout.addWidget(button, 1)
        column.addWidget(control)
        column.addStretch(1)
        self.tabs.addTab(controls, "控制")

        diagram_page = QWidget()
        self._white_background(diagram_page)
        diagram_layout = QVBoxLayout(diagram_page)
        diagram_layout.setContentsMargins(4, 4, 4, 4)
        self.diagram = PlaneDiagram()
        diagram_layout.addWidget(self.diagram, 1)

        orbit_box = QGroupBox("轨道参数")
        orbit_layout = QVBoxLayout(orbit_box)
        orbit_layout.setContentsMargins(4, 4, 4, 4)
        orbit_layout.setSpacing(2)
        self.d_spin = LabeledSlider(
            "自旋 a", -99, 99, 75,
            formatter=lambda v: f"{v / 100:.2f}",
            label_width=52, value_width=44,
        )
        self.d_charge = LabeledSlider(
            "电荷 Q", -99, 99, 45,
            formatter=lambda v: f"{v / 100:.2f}",
            label_width=52, value_width=44,
        )
        self.d_rin = LabeledSlider(
            "内缘", 3, 10, 6, formatter=lambda v: f"{v} M",
            label_width=52, value_width=44,
        )
        self.d_rout = LabeledSlider(
            "外缘", 8, 24, 13, formatter=lambda v: f"{v} M",
            label_width=52, value_width=44,
        )
        self.d_disk = QCheckBox("显示吸积盘")
        for widget in (self.d_spin, self.d_charge, self.d_rin, self.d_rout,
                       self.d_disk):
            orbit_layout.addWidget(widget)
        self.d_spin.valueChanged.connect(
            lambda v: self._diagram_to_main("spin", v))
        self.d_charge.valueChanged.connect(
            lambda v: self._diagram_to_main("charge", v))
        self.d_rin.valueChanged.connect(
            lambda v: self._diagram_to_main("rin", v))
        self.d_rout.valueChanged.connect(
            lambda v: self._diagram_to_main("rout", v))
        self.d_disk.toggled.connect(
            lambda c: self._diagram_to_main("disk", c))
        diagram_layout.addWidget(orbit_box)

        diagram_note = QLabel(
            "视界、顺/逆行光子环、顺/逆行 ISCO、IBCO、盘内外缘。"
            "OSCO 在纯 Kerr–Newman 中没有有限值（稳定圆轨道延伸到无穷）。"
        )
        diagram_note.setWordWrap(True)
        diagram_note.setStyleSheet("color:#444; font-size:11px;")
        diagram_layout.addWidget(diagram_note)
        self.tabs.addTab(diagram_page, "平面图")
        panel_layout.addWidget(self.tabs)

        root.addWidget(panel, 0)
        self.canvas = LensingCanvas()
        root.addWidget(self.canvas, 1, Qt.AlignCenter)
        self.setCentralWidget(central)
        self._white_background(central)
        self.setMinimumSize(1000, 720)
        self.resize(1000, 720)

        self.status = self.statusBar()
        self.status.setSizeGripEnabled(False)
        self.state_label = QLabel("准备中")
        self.params_label = QLabel("Kerr–Newman · 等待首帧")
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
        for key, slot in (
            (Qt.Key_Space, self._toggle_pause),
            ("Ctrl+S", self._save),
            ("Ctrl+R", self._reset),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(slot)
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
                # CPU-only machines: ~50 ms/frame instead of the old 210 ms.
                return "拖动", 200, 50
            return "拖动", DRAG_SIZE, DRAG_STEPS
        index = max(0, min(len(QUALITY) - 1, self.quality_combo.currentIndex()))
        name, width, steps = QUALITY[index]
        if self._cpu_mode:
            width = min(width, 220)
            steps = min(steps, 70)
        # Fine mode is a still-image setting: while the disk texture is
        # animating, render a 600 px preview so the UI stays responsive.
        animating = ((not self._paused) and self.disk_check.isChecked()) \
            or self.auto_orbit_check.isChecked()
        if animating and width > ANIM_SIZE:
            return name + "预览", ANIM_SIZE, ANIM_STEPS
        return name, int(width), int(steps)

    def _config(self, width, steps):
        r_in = float(self.r_in_slider.value())
        r_out = max(r_in + 1.5, float(self.r_out_slider.value()))
        return KerrConfig(
            width=width, height=width,
            distance=46.0,
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
            star_density=float(self.star_density_slider.value()) / 100.0,
            disk_gain=float(self.disk_gain_slider.value()) / 100.0,
            mass=10.0 ** (
                6.0 + 4.0 * float(self.mass_slider.value()) / 1000.0),
            max_steps=steps,
            # The fine still uses a smaller affine step: near the photon
            # sphere the fixed-step error grows like h^4 and the extra
            # 1600 steps buy roughly 8x lower direction error for ~45 ms.
            step_scale=0.012 if width >= 840 else 0.02,
            time=float(self._time),
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
        self._last_rgb = rgb
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
            if not self._paused and time.time() >= self._save_msg_until:
                self.state_label.setText("运行中")
        self._update_status(name, width, steps)

    def _update_status(self, name=None, width=None, steps=None):
        # Keep the "saved <path>" message visible for a moment instead of
        # letting the next animation tick overwrite it immediately.
        if time.time() < self._save_msg_until:
            return
        if name is None:
            name, width, steps = self._current_quality()
        mode = "CUDA" if "CUDA" in self.backend.upper() else (
            "CPU" if "CPU" in self.backend.upper() else self.backend
        )
        if self._render_ms <= 0.0:
            timing = f"编译 {self._compile_s / 1000.0:.1f} s"
        else:
            timing = f"{1000.0 / max(self._render_ms, 1e-6):.0f} fps"
        tag = "预览 · " if name and name.endswith("预览") else ""
        spin = self.spin_slider.value() / 100.0
        charge = self.charge_slider.value() / 100.0
        mass = 10.0 ** (
            6.0 + 4.0 * float(self.mass_slider.value()) / 1000.0)
        self.params_label.setText(
            f"{tag}M {mass:.0e} · a {spin:+.2f} Q {charge:+.2f} · "
            f"{mode} · {width}² · {steps} 步 · {timing}"
        )

    # ------------------------------------------------------------ animation
    def _tick(self):
        # The camera position is fixed; only the disk surface evolves.
        if self._paused or self._dragging:
            return
        if not self.disk_check.isChecked():
            return
        self._time += 0.05
        self._schedule_render(delay=0)

    def _toggle_pause(self):
        self._paused = not self._paused
        self.pause_btn.setText("继续" if self._paused else "暂停")
        self.state_label.setText("暂停" if self._paused else "运行中")
        if not self._paused:
            self._schedule_render(delay=0)

    def _on_auto_orbit(self, checked):
        if checked:
            self._orbit_timer.start()
        else:
            self._orbit_timer.stop()
        self._schedule_render(delay=0)

    def _auto_orbit_tick(self):
        if self._dragging:
            return
        self._azimuth = (self._azimuth + math.radians(0.8)) % (2.0 * math.pi)
        self._schedule_render(delay=0)

    def _on_inclination(self, value):
        self._inclination = float(value)
        self._on_parameter()

    def _clamp_pair(self, source):
        """Keep a²+Q² < KN_MAX² by moving the *other* slider.

        Both sliders stay enabled and draggable: if the user pushes the
        charge up, the spin gives way, and vice versa.  The old code only
        clamped the charge downward, which looked like a stuck slider.
        """
        a = float(self.spin_slider.value()) / 100.0
        q = float(self.charge_slider.value()) / 100.0
        limit = KN_MAX * KN_MAX
        if a * a + q * q <= limit:
            return
        if source == "charge":
            sign = 1.0 if a >= 0.0 else -1.0
            new_a = math.copysign(
                math.sqrt(max(0.0, limit - q * q)), sign)
            mag = int(abs(new_a) * 100.0)
            self._applying = True
            try:
                self.spin_slider.setValue(-mag if sign < 0.0 else mag)
            finally:
                self._applying = False
        else:
            sign = 1.0 if q >= 0.0 else -1.0
            new_q = math.copysign(
                math.sqrt(max(0.0, limit - a * a)), sign)
            mag = int(abs(new_q) * 100.0)
            self._applying = True
            try:
                self.charge_slider.setValue(-mag if sign < 0.0 else mag)
            finally:
                self._applying = False

    def _on_spin(self, value):
        self._clamp_pair("spin")
        self._update_spin_info()
        if not self._applying:
            self._schedule_render()

    def _on_charge(self, value):
        self._clamp_pair("charge")
        self._update_spin_info()
        if not self._applying:
            self._schedule_render()

    def _update_mass_info(self):
        if not hasattr(self, "mass_info"):
            return
        mass = 10.0 ** (
            6.0 + 4.0 * float(self.mass_slider.value()) / 1000.0)
        a = float(self.spin_slider.value()) / 100.0
        q = float(self.charge_slider.value()) / 100.0
        rg_au = 1.4766 * mass / 1.495978707e8
        rh = kerr_horizon(a, q)
        self.mass_info.setText(
            f"r_g {rg_au:.2f} AU · r+ {rh * rg_au:.2f} AU"
        )

    def _on_mass(self, value):
        self._update_mass_info()
        self._on_parameter()

    def _update_spin_info(self):
        a = float(self.spin_slider.value()) / 100.0
        q = float(self.charge_slider.value()) / 100.0
        isco = kerr_isco(a, q)
        self.blackhole_group.setTitle(
            f"黑洞 · r+ {kerr_horizon(a, q):.2f} M · ISCO {isco:.2f} M"
        )
        self._clamp_disk_inner(isco)
        self._sync_disk_ranges()
        self._update_mass_info()
        self._update_diagram()

    def _clamp_disk_inner(self, isco):
        """Keep the disk outside the ISCO (a disk at r=3 M is a black ring)."""
        min_r = max(3, min(10, int(math.ceil(isco - 1e-9))))
        if self.r_in_slider.slider.minimum() == min_r:
            return
        self.r_in_slider.blockSignals(True)
        self.r_in_slider.set_range(min_r, 10)
        if self.r_in_slider.value() < min_r:
            self.r_in_slider.setValue(min_r)
        self.r_in_slider.blockSignals(False)

    def _sync_disk_ranges(self):
        """Outer edge must stay at least 1.5 M outside the inner edge."""
        r_in = float(self.r_in_slider.value())
        min_out = max(8, min(24, int(math.ceil(r_in + 1.5))))
        if self.r_out_slider.slider.minimum() != min_out:
            self.r_out_slider.blockSignals(True)
            self.r_out_slider.set_range(min_out, 24)
            if self.r_out_slider.value() < min_out:
                self.r_out_slider.setValue(min_out)
            self.r_out_slider.blockSignals(False)

    def _sync_diagram_controls(self):
        if not hasattr(self, "d_spin"):
            return
        for dup, main in (
            (self.d_spin, self.spin_slider),
            (self.d_charge, self.charge_slider),
            (self.d_rin, self.r_in_slider),
            (self.d_rout, self.r_out_slider),
        ):
            dup.blockSignals(True)
            try:
                dup.set_range(main.slider.minimum(), main.slider.maximum())
                dup.setValue(main.value())
            finally:
                dup.blockSignals(False)
        self.d_disk.blockSignals(True)
        try:
            self.d_disk.setChecked(self.disk_check.isChecked())
        finally:
            self.d_disk.blockSignals(False)

    def _diagram_to_main(self, kind, value):
        if self._syncing:
            return
        self._syncing = True
        try:
            if kind == "spin":
                self.spin_slider.setValue(int(value))
            elif kind == "charge":
                self.charge_slider.setValue(int(value))
            elif kind == "rin":
                self.r_in_slider.setValue(int(value))
            elif kind == "rout":
                self.r_out_slider.setValue(int(value))
            elif kind == "disk":
                self.disk_check.setChecked(bool(value))
        finally:
            self._syncing = False

    def _update_diagram(self):
        if not hasattr(self, "diagram"):
            return
        self._sync_diagram_controls()
        self.diagram.set_params(
            float(self.spin_slider.value()) / 100.0,
            float(self.charge_slider.value()) / 100.0,
            float(self.r_in_slider.value()),
            float(self.r_out_slider.value()),
            bool(self.disk_check.isChecked()),
        )

    def _on_parameter(self, *args):
        if self._applying:
            return
        self._sync_disk_enabled()
        self._sync_disk_ranges()
        self._update_diagram()
        self._schedule_render()

    def _on_disk_toggle(self, checked):
        self._disk_enabled = bool(checked)
        self._sync_disk_enabled()
        self._update_diagram()
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
        value = max(5, min(120, value))
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
            self._inclination = float(preset["inclination"])
            self.inclination_slider.setValue(int(preset["inclination"]))
            self.fov_slider.setValue(int(preset["fov"]))
            self.disk_check.setChecked(bool(preset["disk_on"]))
            self.r_in_slider.setValue(int(preset["r_in"]))
            self._sync_disk_ranges()
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
            self.mass_slider.setValue(int(preset.get("mass", 500)))
            self.star_density_slider.setValue(
                int(preset.get("star_density", 200)))
            self._time = 0.0
            self._azimuth = 0.0
            self._disk_enabled = bool(preset["disk_on"])
            self._sync_disk_enabled()
        finally:
            self._applying = False
        self._update_diagram()
        self._schedule_render(delay=70)

    def _reset(self):
        # Only the disk evolution phase is reset.  Parameters, camera angle,
        # preset and pause state are deliberately left untouched.
        self._time = 0.0
        self._save_msg_until = 0.0
        self.state_label.setText("暂停" if self._paused else "运行中")
        self._schedule_render(delay=0)

    # ------------------------------------------------------------------ save
    # ------------------------------------------------------------- settings
    _SETTINGS_SLIDERS = (
        ("spin", "spin_slider"),
        ("charge", "charge_slider"),
        ("mass_v2", "mass_slider"),
        ("star_density", "star_density_slider"),
        ("inclination", "inclination_slider"),
        ("fov", "fov_slider"),
        ("r_in", "r_in_slider"),
        ("r_out", "r_out_slider"),
        ("temperature", "temp_slider"),
        ("opacity", "opacity_slider"),
        ("disk_gain", "disk_gain_slider"),
        ("exposure", "exposure_slider"),
        ("star_gain", "star_slider"),
    )

    def _load_settings(self):
        settings = QSettings("PySide绘图工具", "KerrNewmanLensing")
        self._applying = True
        try:
            for key, attr in self._SETTINGS_SLIDERS:
                value = settings.value(key)
                if value is None:
                    continue
                try:
                    getattr(self, attr).setValue(int(value))
                except (TypeError, ValueError):
                    pass
            for key, attr in (("disk", "disk_check"), ("doppler", "doppler_check")):
                value = settings.value(key)
                if value is not None:
                    getattr(self, attr).setChecked(
                        str(value).lower() in ("true", "1"))
            auto = settings.value("auto_orbit")
            if auto is not None:
                self.auto_orbit_check.setChecked(
                    str(auto).lower() in ("true", "1"))
            quality = settings.value("quality")
            if quality is not None:
                try:
                    self.quality_combo.blockSignals(True)
                    self.quality_combo.setCurrentIndex(int(quality))
                except (TypeError, ValueError):
                    pass
                finally:
                    self.quality_combo.blockSignals(False)
            geometry = settings.value("geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)
            self._inclination = float(self.inclination_slider.value())
        finally:
            self._applying = False
        self._update_spin_info()
        self._sync_disk_enabled()
        self._sync_disk_ranges()
        self._update_diagram()
        settings.sync()

    def _save_settings(self):
        settings = QSettings("PySide绘图工具", "KerrNewmanLensing")
        for key, attr in self._SETTINGS_SLIDERS:
            settings.setValue(key, getattr(self, attr).value())
        settings.setValue("disk", bool(self.disk_check.isChecked()))
        settings.setValue("doppler", bool(self.doppler_check.isChecked()))
        settings.setValue(
            "auto_orbit", bool(self.auto_orbit_check.isChecked()))
        settings.setValue("quality", int(self.quality_combo.currentIndex()))
        settings.setValue("geometry", self.saveGeometry())
        settings.sync()

    def _still_quality(self):
        """Quality used for a still frame regardless of drag/animation."""
        index = max(0, min(len(QUALITY) - 1, self.quality_combo.currentIndex()))
        name, width, steps = QUALITY[index]
        if self._cpu_mode:
            width = min(width, 220)
            steps = min(steps, 70)
        return name, int(width), int(steps)

    def _save(self):
        path = ask_save_path(
            self, "保存 Kerr–Newman 光线追踪", directory="lensing.png",
            suffix=".png",
        )
        if not path:
            return
        # Save the rendered ndarray, not canvas.grab(): the canvas is 600 px
        # while the fine mode renders 840 px, and grab() would silently throw
        # that resolution (and the HiDPI scale) away.
        _name, width, steps = self._still_quality()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rgb = self.renderer.render(self._config(width, steps))
        except Exception as exc:  # noqa: BLE001 - surface in a dialog
            QMessageBox.warning(self, "保存失败", str(exc)[:300])
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._last_rgb = rgb
        self.canvas.set_frame(rgb)
        image = ndarray_to_qimage(np.ascontiguousarray(rgb, dtype=np.uint8))
        if image.save(path):
            self.state_label.setText(f"已保存 {rgb.shape[1]}×{rgb.shape[0]}")
            self._save_msg_until = time.time() + 2.0
            self.params_label.setText(path)
        else:
            QMessageBox.warning(self, "保存失败", "无法写入所选路径。")

    def closeEvent(self, event):
        self._save_settings()
        self._anim_timer.stop()
        self._render_timer.stop()
        event.accept()


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
