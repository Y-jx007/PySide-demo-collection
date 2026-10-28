# -*- coding: utf-8 -*-
"""Retarded Liénard-Wiechert radiation from a variable-speed electron."""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from PySide6.QtGui import QPainterPath, QPixmap

from common.imaging import ndarray_to_qimage
from common.qt_base import (
    QColor, QFont, QComboBox, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QPainter, QPen, QPointF,
    QCheckBox, QPushButton, QScrollArea, QSizePolicy, QTimer, QVBoxLayout,
    QWidget, Qt, QRect, LabeledSlider, run_app,
)

from support.physics.lienard_wiechert import Trajectory
from support.physics.lw_taichi import N as LW_N, evaluate_np, vector_field_np, trace_component_np, colorize_np

FIELD_N = LW_N
EXTENT_FACTOR = 5.0
RED = QColor(255, 70, 56)
BLUE = QColor(68, 142, 255)


def _norm_field(values, scale=None):
    sample = values[::4, ::4]
    positive = sample[np.isfinite(sample) & (sample > 0)]
    if positive.size == 0:
        return np.zeros_like(values, dtype=np.float32), 1e-30
    raw = max(float(np.percentile(positive, 99.3)) * 0.22, 1e-30)
    if scale is None:
        scale = raw
    else:
        scale = 0.84 * float(scale) + 0.16 * raw
    return np.clip(
        np.log1p(values / max(scale, 1e-30)) / math.log(2.0), 0.0, 1.0
    ).astype(np.float32), scale


def _norm_scale(values, scale=None):
    sample = values[::4, ::4]
    positive = sample[np.isfinite(sample) & (sample > 0)]
    if positive.size == 0:
        return 1e-30
    raw = max(float(np.percentile(positive, 99.3)) * 0.22, 1e-30)
    if scale is None:
        return raw
    return 0.84 * float(scale) + 0.16 * raw


def render_field(motion, t, scales=None, show_e=True, show_b=False,
                 show_brems=True, show_sync=True, show_total=False):
    extent = EXTENT_FACTOR * motion.radius
    raw_par, raw_perp, raw_total, raw_b = evaluate_np(
        t, motion.beta0, motion.modulation, motion.drive,
        motion.radius, extent,
    )
    old = scales if scales is not None else (None, None, None, None)
    scale_r = _norm_scale(raw_par, old[0])
    scale_b = _norm_scale(raw_perp, old[1])
    scale_t = _norm_scale(raw_total, old[2])
    scale_bz = _norm_scale(raw_b, old[3])
    rgb = colorize_np(
        (scale_r, scale_b, scale_t, scale_bz),
        show_e, show_b, show_brems, show_sync, show_total,
    )
    current = motion.state(t)
    return rgb, current["pos"], extent, (scale_r, scale_b, scale_t, scale_bz)


class FieldCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 440)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image = None
        self._extent = 1.0
        self._pos = (0.0, 0.0)
        self._radius = 1.0
        self._vectors = []
        self._b_vectors = []
        self._vector_mode = False
        self._line_pixmap = None
        self._b_pixmap = None
        self._line_dirty = True
        self._line_size = None

    def set_frame(self, rgb, pos, extent, radius, vectors=None,
                  b_vectors=None, vector_mode=False):
        self._image = ndarray_to_qimage(rgb)
        self._extent = float(extent)
        self._pos = (float(pos[0]), float(pos[1]))
        self._radius = float(radius)
        new_vectors = vectors or []
        new_b = b_vectors or []
        if new_vectors is not self._vectors or new_b is not self._b_vectors:
            self._vectors = new_vectors
            self._b_vectors = new_b
            self._line_dirty = True
        self._vector_mode = bool(vector_mode)
        self.update()

    def _build_line_pixmap(self, left, top, side):
        pm = QPixmap(max(1, self.width() // 2), max(1, self.height() // 2))
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.scale(0.5, 0.5)
        p.setRenderHint(QPainter.Antialiasing, True)

        def map_xy(x, y):
            ex = self._extent
            return (left + (x + ex) / (2 * ex) * side,
                    top + (y + ex) / (2 * ex) * side)

        for points, colour, alphas, _dashed in self._vectors:
            if len(points) < 2:
                continue
            c = QColor(colour)
            c.setAlpha(min(255, int(1.7 * np.percentile(alphas, 92))))
            pen = QPen(c, 1.45)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            pts = [map_xy(*pt) for pt in points]
            path = QPainterPath()
            path.moveTo(*pts[0])
            if len(pts) == 2:
                path.lineTo(*pts[1])
            else:
                for i in range(1, len(pts) - 1):
                    p0 = pts[max(i - 1, 0)]
                    p1 = pts[i]
                    p2 = pts[i + 1]
                    p3 = pts[min(i + 2, len(pts) - 1)]
                    c1 = (p1[0] + (p2[0] - p0[0]) / 6.0,
                          p1[1] + (p2[1] - p0[1]) / 6.0)
                    c2 = (p2[0] - (p3[0] - p1[0]) / 6.0,
                          p2[1] - (p3[1] - p1[1]) / 6.0)
                    path.cubicTo(*c1, *c2, *p2)
            p.drawPath(path)
        p.end()
        self._line_pixmap = pm

        # B is perpendicular to the plane: draw its ⊙/⊗ symbols at full
        # resolution, not in the half-resolution field-line texture.
        if self._b_vectors:
            bpm = QPixmap(self.width(), self.height())
            bpm.fill(Qt.transparent)
            bp = QPainter(bpm)
            bp.setRenderHint(QPainter.Antialiasing, True)
            for bx, by, bz, alpha in self._b_vectors:
                x1, y1 = map_xy(bx, by)
                c = QColor(18, 135, 72)
                c.setAlpha(int(100 + 155 * alpha))
                if bz >= 0:
                    # out of page: solid dot
                    bp.setPen(Qt.NoPen)
                    bp.setBrush(c)
                    bp.drawEllipse(QPointF(x1, y1), 5.0, 5.0)
                else:
                    # into page: hollow circle with a cross
                    bp.setPen(QPen(c, 2.0))
                    bp.setBrush(Qt.NoBrush)
                    bp.drawEllipse(QPointF(x1, y1), 7.0, 7.0)
                    bp.drawLine(QPointF(x1 - 5, y1 - 5), QPointF(x1 + 5, y1 + 5))
                    bp.drawLine(QPointF(x1 - 5, y1 + 5), QPointF(x1 + 5, y1 - 5))
            bp.end()
            self._b_pixmap = bpm
        else:
            self._b_pixmap = None
        self._line_dirty = False
        self._line_size = (self.width(), self.height())

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        if self._image is None:
            p.end()
            return
        side = min(self.width(), self.height())
        left = (self.width() - side) // 2
        top = (self.height() - side) // 2
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(QRect(left, top, side, side), self._image)

        def map_xy(x, y):
            ex = self._extent
            return (left + (x + ex) / (2 * ex) * side,
                    top + (y + ex) / (2 * ex) * side)

        if self._vector_mode:
            if (self._line_pixmap is None or self._line_dirty
                    or self._line_size != (self.width(), self.height())):
                self._build_line_pixmap(left, top, side)
            if self._line_pixmap is not None:
                p.setRenderHint(QPainter.SmoothPixmapTransform, True)
                p.drawPixmap(self.rect(), self._line_pixmap)
            if self._b_pixmap is not None:
                p.drawPixmap(0, 0, self._b_pixmap)

        cx, cy = map_xy(0.0, 0.0)
        rr = self._radius / self._extent * side / 2
        p.setPen(QPen(QColor(80, 125, 135, 105), 1, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), rr, rr)

        ex, ey = map_xy(*self._pos)
        # Literally one pixel: no black dot, no halo.
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(20, 32, 38))
        p.drawRect(int(round(ex)), int(round(ey)), 1, 1)


        p.end()


class SpectrumCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMaximumHeight(170)
        self.freq = None
        self.red = None
        self.blue = None

    def set_data(self, freq, red, blue):
        self.freq, self.red, self.blue = freq, red, blue
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        left, right, top, bottom = 48, 18, 24, 27
        w = max(1, self.width() - left - right)
        h = max(1, self.height() - top - bottom)
        p.setPen(QColor(45, 65, 70))
        p.setFont(QFont("Microsoft YaHei", 9))
        p.drawText(left, 16, "远场功率谱（16 方向平均，FFT；共同归一化）")
        if self.freq is None or len(self.freq) < 8:
            p.drawText(left + 10, top + h // 2, "积累频谱数据…")
            p.end()
            return
        mask = self.freq > 0
        f = self.freq[mask]
        red = self.red[mask]
        blue = self.blue[mask]
        xmin, xmax = -2.0, 1.2  # fixed Nyquist range; do not rescale per frame
        def xmap(freq):
            return left + (math.log10(max(freq, 1e-9)) - xmin) / (xmax - xmin) * w
        def ymap(v):
            return top + h - float(np.clip(v, 0, 1)) * h
        p.setPen(QPen(QColor(100, 125, 130, 80), 1))
        for tick in range(-2, int(math.ceil(xmax)) + 1):
            x = left + (tick - xmin) / (xmax - xmin) * w
            p.drawLine(x, top, x, top + h)
            p.setPen(QColor(80, 105, 110, 210))
            p.drawText(int(x - 8), self.height() - 9, f"10^{tick}")
            p.setPen(QPen(QColor(100, 125, 130, 80), 1))
        for frac in (0, 0.5, 1):
            y = top + h - frac * h
            p.drawLine(left, y, left + w, y)
            p.setPen(QColor(80, 105, 110, 210))
            p.drawText(10, int(y + 4), f"{frac:.1f}")
            p.setPen(QPen(QColor(100, 125, 130, 80), 1))

        def path_for(values):
            path = QPainterPath()
            path.moveTo(left, top + h)
            for ff, vv in zip(f, values):
                path.lineTo(xmap(ff), ymap(vv))
            path.lineTo(left + w, top + h)
            path.closeSubpath()
            return path

        p.setCompositionMode(QPainter.CompositionMode_Multiply)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(210, 55, 45, 70))
        p.drawPath(path_for(red))
        p.setBrush(QColor(40, 90, 200, 70))
        p.drawPath(path_for(blue))
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        for values, colour in ((red, QColor(180, 42, 35)), (blue, QColor(25, 65, 170))):
            path = QPainterPath()
            for i, (ff, vv) in enumerate(zip(f, values)):
                x, y = xmap(ff), ymap(vv)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.setPen(QPen(colour, 1.4))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
        p.setPen(QColor(80, 105, 110, 210))
        p.drawText(left + w - 92, self.height() - 9, "频率 (1/模拟时间)")
        p.end()


class FarFieldSpectrum:
    def __init__(self, n_fft=512, angles=16, dt=0.033):
        self.n_fft = n_fft
        self.angles = angles
        self.dt = dt
        self.red = deque(maxlen=n_fft)
        self.blue = deque(maxlen=n_fft)

    def reset(self):
        self.red.clear()
        self.blue.clear()

    def update(self, motion, t):
        f = motion.far_field(t, angles=self.angles)
        # Store the vector field, not |E|: the FFT must be applied to the
        # field components, otherwise rectification creates false harmonics.
        self.red.append(np.asarray(f["e_par"], dtype=np.float64))
        self.blue.append(np.asarray(f["e_perp"], dtype=np.float64))

    def data(self):
        n = len(self.red)
        if n < 96:
            return None
        arr_r = np.asarray(self.red) - np.mean(self.red, axis=0, keepdims=True)
        arr_b = np.asarray(self.blue) - np.mean(self.blue, axis=0, keepdims=True)
        window = np.hanning(n)[:, None, None]
        spec_r = np.abs(np.fft.rfft(arr_r * window, axis=0)) ** 2
        spec_b = np.abs(np.fft.rfft(arr_b * window, axis=0)) ** 2
        spec_r = spec_r.sum(axis=(1, 2)) / max(1, self.angles)
        spec_b = spec_b.sum(axis=(1, 2)) / max(1, self.angles)
        freq = np.fft.rfftfreq(n, d=self.dt)
        # One common normalisation for both spectra.  Normalising each curve
        # by its own maximum would destroy the bremsstrahlung/synchrotron
        # relative intensity.
        peak = max(float(np.max(spec_r)), float(np.max(spec_b)), 1e-30)
        return freq, spec_r / peak, spec_b / peak


def sample_field_lines(motion, t, extent, show_e, show_b, show_brems,
                       show_sync, show_poynting, scales=None):
    """Trace one shared E-line geometry on GPU, then colour each component."""
    f = vector_field_np(
        t, motion.beta0, motion.modulation, motion.drive,
        motion.radius, extent,
    )
    lines, b_vectors = [], []
    scales = scales or (None, None, None, None)

    def own_scale(vec):
        mag2 = np.sum(vec * vec, axis=-1)
        pos = mag2[mag2 > 0]
        if pos.size == 0:
            return 1e-30
        return max(float(np.percentile(pos, 99.3)) * 0.22, 1e-30)

    def profile(field, points, scale):
        if scale is None:
            scale = own_scale(field)
        n = f["e_total"].shape[0]
        ux = (points[:, 0] + extent) / (2 * extent) * (n - 1)
        uy = (points[:, 1] + extent) / (2 * extent) * (n - 1)
        ix = np.clip(np.floor(ux).astype(np.int32), 0, n - 2)
        iy = np.clip(np.floor(uy).astype(np.int32), 0, n - 2)
        fx, fy = ux - ix, uy - iy
        a = field[iy, ix] * (1 - fx)[:, None] + field[iy, ix + 1] * fx[:, None]
        b = field[iy + 1, ix] * (1 - fx)[:, None] + field[iy + 1, ix + 1] * fx[:, None]
        vec = a * (1 - fy)[:, None] + b * fy[:, None]
        m2 = vec[:, 0] ** 2 + vec[:, 1] ** 2
        t = np.minimum(1.0, np.log1p(m2 / max(scale, 1e-30)) / math.log(2.0))
        return (8 + 247 * t ** 1.25).astype(np.int32).tolist()

    def geometry(mode):
        scale = scales[2] if mode == 0 else own_scale(f["e_total"])
        forward = trace_component_np(
            t, motion.beta0, motion.modulation, motion.drive,
            motion.radius, extent, mode, 1.0, scale or own_scale(f["e_total"]),
        )
        backward = trace_component_np(
            t, motion.beta0, motion.modulation, motion.drive,
            motion.radius, extent, mode, -1.0, scale or own_scale(f["e_total"]),
        )
        pf, af, cf = forward
        pb, ab, cb = backward
        out = []
        for n in range(pf.shape[1]):
            nf, nb = int(cf[n]), int(cb[n])
            if nf + nb < 4:
                continue
            points = np.vstack([pb[:nb, n][::-1], pf[1:nf, n]])
            alphas = list(np.asarray(ab[:nb, n][::-1]) * 255.0) + list(np.asarray(af[1:nf, n]) * 255.0)
            alphas = [int(v) for v in np.clip(alphas, 0.0, 255.0)]
            if len(points) > 16:
                idx = np.unique(np.linspace(0, len(points) - 1, 16).astype(int))
                points = points[idx]
                alphas = [alphas[i] for i in idx]
            out.append((points, alphas))
        return out

    shared = []
    if show_e or show_brems or show_sync:
        shared = geometry(0)
    if show_e:
        for points, alphas in shared:
            lines.append((points, QColor(30, 110, 130), alphas, False))
    if show_brems:
        for points, _alphas in shared:
            lines.append((points, QColor(205, 55, 45),
                          profile(f["e_par"], points, scales[0]), True))
    if show_sync:
        for points, _alphas in shared:
            lines.append((points, QColor(35, 85, 195),
                          profile(f["e_perp"], points, scales[1]), False))

    bz = f["bz"]
    if show_b:
        bscale = scales[3]
        if bscale is None:
            bscale = own_scale(np.stack([bz, np.zeros_like(bz)], axis=-1))
        for i in range(0, bz.shape[0], 10):
            for j in range(0, bz.shape[1], 10):
                alpha = min(1.0, math.log1p(float(bz[i, j]) ** 2 / max(bscale, 1e-30)) / math.log(2.0))
                if alpha > 0.08:
                    b_vectors.append((
                        float(-extent + 2 * extent * j / (bz.shape[1] - 1)),
                        float(-extent + 2 * extent * i / (bz.shape[0] - 1)),
                        float(bz[i, j]), alpha,
                    ))
    if show_poynting:
        e = f["e_total"]
        svec = np.stack([e[..., 1] * bz, -e[..., 0] * bz], axis=-1)
        sscale = own_scale(svec)
        for points, _alphas in geometry(3):
            lines.append((points, QColor(255, 215, 0),
                          profile(svec, points, sscale), False))
    return lines, b_vectors


class LWWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bremsstrahlung — 推迟 Liénard-Wiechert 辐射")
        self.setFixedSize(830, 710)
        self.setMinimumSize(830, 710)
        self.motion = Trajectory(beta0=0.65, modulation=0.20, drive=0.70, radius=0.72)
        self.t = 0.0
        self.dt = 1.0 / 30.0
        self.running = True
        self.field_scales = None
        self._field_rgb = None
        self.a_parallel_peak = 0.0
        self.a_perp_peak = 0.0
        self._line_cache = None
        self._line_phase = 0
        self.spectrum_buf = FarFieldSpectrum(dt=self.dt)
        self._build_ui()
        self._reset()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(38)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        left = QWidget()
        left.setFixedWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        panel_widget = QWidget()
        panel = QVBoxLayout(panel_widget)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.setSpacing(8)
        panel.addWidget(self._build_motion_group())
        panel.addWidget(self._build_display_group())
        panel.addWidget(self._build_control_group())
        panel.addWidget(self._build_readout_group())
        panel.addStretch(1)
        scroll.setWidget(panel_widget)
        left_layout.addWidget(scroll)
        root.addWidget(left, 0)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        self.canvas = FieldCanvas()
        self.canvas.setFixedSize(520, 520)
        self.spectrum = SpectrumCanvas()
        self.spectrum.setFixedWidth(520)
        right.setFixedWidth(520)
        right_layout.addWidget(self.canvas, 0, Qt.AlignHCenter)
        right_layout.addWidget(self.spectrum, 0)
        root.addWidget(right, 0)
        self.statusBar().showMessage(
            "推迟 Liénard-Wiechert 辐射场，c=1；轨迹为预设圆周运动，未含辐射反作用；颜色为视觉归一化。"
        )

    def _build_motion_group(self):
        group = QGroupBox("运动")
        form = QFormLayout(group)
        form.setSpacing(6)
        # 滑块 + 数值标签统一用 LabeledSlider（百分数显示）
        self.beta_slider = LabeledSlider("", 5, 95, 65, formatter=lambda v: f"{v / 100:.2f}")
        self.mod_slider = LabeledSlider("", 0, 40, 20, formatter=lambda v: f"{v / 100:.2f}")
        self.drive_slider = LabeledSlider("", 1, 300, 70, formatter=lambda v: f"{v / 100:.2f}")
        self.radius_slider = LabeledSlider("", 20, 200, 72, formatter=lambda v: f"{v / 100:.2f}")
        for widget, caption in ((self.beta_slider, "基础 β = v/c"),
                                (self.mod_slider, "速度调制"),
                                (self.drive_slider, "调制频率"),
                                (self.radius_slider, "轨道半径")):
            widget.valueChanged.connect(self._params_changed)
            form.addRow(caption, widget)
        return group

    def _build_display_group(self):
        group = QGroupBox("显示")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        self.e_check = QCheckBox("电场 E（青）")
        self.b_check = QCheckBox("磁场 B（绿）")
        self.brem_check = QCheckBox("轫致辐射（红）")
        self.brem_check.setChecked(True)
        self.sync_check = QCheckBox("同步辐射（蓝）")
        self.sync_check.setChecked(True)
        self.total_check = QCheckBox("总辐射强度 / 坡印廷（亮黄）")
        for check in (self.e_check, self.b_check, self.brem_check,
                      self.sync_check, self.total_check):
            layout.addWidget(check)
            check.stateChanged.connect(self._display_changed)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("显示模式"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("强度", "intensity")
        self.mode_combo.addItem("场线", "lines")
        self.mode_combo.currentIndexChanged.connect(self._display_changed)
        mode_row.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_row)
        return group

    def _build_control_group(self):
        group = QGroupBox("控制")
        layout = QHBoxLayout(group)
        self.start_btn = QPushButton("暂停")
        self.start_btn.clicked.connect(self._toggle_run)
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self._reset)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.reset_btn)
        return group

    def _build_readout_group(self):
        group = QGroupBox("读数")
        form = QFormLayout(group)
        form.setSpacing(4)
        self.labels = {}
        for key, text in (
            ("beta", "β"),
            ("gamma", "γ"),
            ("apar", "a∥ = dβ/dt"),
            ("aperp", "a⊥ = β²/R"),
            ("acc", "|a|"),
            ("apar_max", "a∥ 峰值"),
            ("aperp_max", "a⊥ 峰值"),
            ("ratio", "Σ|E∥|² / Σ|E⊥|²"),
            ("fmax", "峰值频率"),
        ):
            value = QLabel("–")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.addRow(text, value)
            self.labels[key] = value
        return group

    def _display_changed(self, _state=None):
        self._field_rgb = None
        self._line_cache = None
        self._line_phase = 0
        self._tick()

    def _params_changed(self, _value=None):
        self._line_cache = None
        self._line_phase = 0
        beta = self.beta_slider.value() / 100.0
        mod = self.mod_slider.value() / 100.0
        drive = self.drive_slider.value() / 100.0
        radius = self.radius_slider.value() / 100.0
        # The analytic trajectory is valid only for beta < 1.
        if beta + mod > 0.98:
            mod = max(0.0, 0.98 - beta)
            self.mod_slider.blockSignals(True)
            self.mod_slider.setValue(int(round(mod * 100)))
            self.mod_slider.blockSignals(False)
        self.motion = Trajectory(beta0=beta, modulation=mod, drive=drive, radius=radius)
        self._reset()

    def _toggle_run(self):
        self.running = not self.running
        if self.running:
            self.timer.start(38)
            self.start_btn.setText("暂停")
        else:
            self.timer.stop()
            self.start_btn.setText("开始")

    def _reset(self):
        self.t = 0.0
        self.field_scales = None
        self._field_rgb = None
        self.a_parallel_peak = 0.0
        self.a_perp_peak = 0.0
        self._line_cache = None
        self._line_phase = 0
        self.spectrum_buf.reset()
        self._tick()

    def _tick(self):
        self.t += self.dt
        vector_mode = self.mode_combo.currentData() == "lines"
        extent = EXTENT_FACTOR * self.motion.radius
        if vector_mode:
            # Field-line mode: total radiation is shown as Poynting lines,
            # never as a yellow scalar background.
            pos = self.motion.state(self.t)["pos"]
            frame = np.full((FIELD_N, FIELD_N, 3), 255, dtype=np.uint8)
            self._line_phase += 1
            if self._line_cache is None or self._line_phase % 3 == 0:
                vectors, b_vectors = sample_field_lines(
                    self.motion, self.t, extent,
                    self.e_check.isChecked(), self.b_check.isChecked(),
                    self.brem_check.isChecked(), self.sync_check.isChecked(),
                    self.total_check.isChecked(), scales=self.field_scales,
                )
                self._line_cache = (vectors, b_vectors)
            else:
                vectors, b_vectors = self._line_cache
        else:
            rgb, pos, extent, scales = render_field(
                self.motion, self.t, scales=self.field_scales,
                show_e=self.e_check.isChecked(),
                show_b=self.b_check.isChecked(),
                show_brems=self.brem_check.isChecked(),
                show_sync=self.sync_check.isChecked(),
                show_total=self.total_check.isChecked(),
            )
            self.field_scales = scales
            frame = rgb
            vectors, b_vectors = [], []
        self.canvas.set_frame(
            frame, pos, extent, self.motion.radius,
            vectors=vectors, b_vectors=b_vectors, vector_mode=vector_mode,
        )
        self.spectrum_buf.update(self.motion, self.t)
        data = self.spectrum_buf.data()
        if data is not None:
            freq, red, blue = data
            self.spectrum.set_data(freq, red, blue)
            mask = (freq > 0.01) & (freq < 5.0)
            if np.any(mask):
                total = red + blue
                i = int(np.argmax(total[mask]))
                self.labels["fmax"].setText(f"{freq[mask][i]:.2f} Hz")
        state = self.motion.state(self.t)
        beta = float(state["beta"])
        a_parallel = abs(float(state["beta_dot"]))
        a_perp = beta * beta / max(self.motion.radius, 1e-9)
        self.labels["beta"].setText(f"{beta:.3f}")
        self.labels["gamma"].setText(f"{1.0 / math.sqrt(max(1.0 - beta * beta, 1e-9)):.2f}")
        self.a_parallel_peak = max(self.a_parallel_peak, a_parallel)
        self.a_perp_peak = max(self.a_perp_peak, a_perp)
        self.labels["apar"].setText(f"{a_parallel:.3f}")
        self.labels["aperp"].setText(f"{a_perp:.3f}")
        self.labels["acc"].setText(f"{math.hypot(a_parallel, a_perp):.3f}")
        self.labels["apar_max"].setText(f"{self.a_parallel_peak:.3f}")
        self.labels["aperp_max"].setText(f"{self.a_perp_peak:.3f}")
        if len(self.spectrum_buf.red) > 4:
            rr = np.asarray(self.spectrum_buf.red)
            bb = np.asarray(self.spectrum_buf.blue)
            ratio = float(np.sum(rr * rr) / max(np.sum(bb * bb), 1e-30))
            self.labels["ratio"].setText(f"{ratio:.2e}")

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


def main():
    raise SystemExit(run_app(LWWindow))


if __name__ == "__main__":
    main()
