# -*- coding: utf-8 -*-
"""KAM tori and the Chirikov standard map.

The app renders a rotation-number frequency map.  Coloured bands are
regular KAM tori and resonance island chains; dark areas are chaotic
initial conditions whose rotation number is not stable over the sampled
window.  Click the map to trace an orbit and read its rotation number.
"""
from __future__ import annotations

import logging
import random
import time

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from common.app import run_app
from common.imaging import ask_save_path, ndarray_to_qimage
from common.widgets import LabeledSlider
from common.workers import StoppableThread
from support.physics.kam_standard_map import (
    cyclic_lut,
    frequency_image,
    frequency_map,
    trace_orbit,
)
from support.physics.kam_taichi import (
    frequency_map_taichi,
    taichi_available,
)
from support.physics.kam_crtbp import compute_poincare
from support.physics.solar_system_orbits import (
    BELT,
    PLANETS,
    PRECESSION_ARCSEC_PER_YEAR,
    asteroid_positions,
    planet_arc,
    planet_orbit_at,
    planet_position_at,
)

APP_TITLE = "KAM · 标准映射与行星轨道"
FAMILY = "Microsoft YaHei"
CHAOS_LYAP = 0.02
PRECESSION_PER_ORBIT = 12.0

CANVAS_TOP = QColor("#07090d")
CANVAS_BOTTOM = QColor("#0b0f14")
PLOT_EMPTY = QColor("#0c1015")
INK = QColor("#ece7da")
MUTED = QColor("#8f9691")
FAINT = QColor(255, 255, 255, 12)
BORDER = QColor(255, 255, 255, 38)
ACCENT = QColor("#f0c879")


class FrequencyWorker(StoppableThread):
    """Compute the rotation-number map off the UI thread."""

    progress = Signal(int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, K, resolution, iterations, token, parent=None, fast=False):
        super().__init__(parent)
        self.K = float(K)
        self.resolution = int(resolution)
        self.iterations = int(iterations)
        self.token = int(token)
        self.fast = bool(fast)

    def run(self):
        started = time.perf_counter()
        try:
            def report(frac):
                self.progress.emit(int(max(0.0, min(1.0, float(frac))) * 100))

            nu, lyap = frequency_map(
                self.K,
                self.resolution,
                self.iterations,
                progress=report,
                should_stop=lambda: self.stopping,
            )
            if nu is None or self.stopping:
                return
            image = frequency_image(nu, lyap)
            self.done.emit({
                "token": self.token,
                "K": self.K,
                "res": self.resolution,
                "iters": self.iterations,
                "nu": nu,
                "lyap": lyap,
                "image": image,
                "elapsed": time.perf_counter() - started,
                "fast": self.fast,
            })
        except Exception as exc:  # noqa: BLE001 - report to the window
            logging.exception("KAM frequency map failed")
            self.failed.emit(str(exc))


class KAMCanvas(QWidget):
    """Dark phase-space canvas: frequency image, axes, orbit probe."""

    probed = Signal(float, float)
    cursorMoved = Signal(float, float)
    cursorLeft = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

        self._image = None
        self._payload = None
        self._loading = False
        self._progress = 0
        self._error = None
        self._orbit = None
        self._orbit_nu = 0.0
        self._orbit_lyap = 0.0
        self._hover = None

        self._font_title = QFont(FAMILY, 10, QFont.Bold)
        self._font_sub = QFont(FAMILY, 8)
        self._font_tick = QFont("Segoe UI", 7)
        self._font_chip = QFont(FAMILY, 8)

        lut = cyclic_lut()
        bar = np.ascontiguousarray(lut[::-1].reshape(len(lut), 1, 3))
        self._colorbar = ndarray_to_qimage(bar)

    # -- data ---------------------------------------------------------
    def set_image(self, image, payload):
        self._image = image
        self._payload = payload
        self._loading = False
        self._error = None
        self.update()

    def begin_loading(self):
        self._loading = True
        self._progress = 0
        self._error = None
        self.update()

    def set_progress(self, value):
        self._progress = max(0, min(100, int(value)))
        if self._loading:
            self.update()

    def end_loading(self):
        self._loading = False
        self.update()

    def set_error(self, message):
        self._loading = False
        self._error = str(message) or "计算失败"
        self.update()

    def set_orbit(self, orbit, nu, lyap):
        self._orbit = orbit
        self._orbit_nu = float(nu)
        self._orbit_lyap = float(lyap)
        self.update()

    def clear_orbit(self):
        self._orbit = None
        self.update()

    def sample_frequency(self, q, p):
        payload = self._payload
        if not payload:
            return None
        res = int(payload["res"])
        col = min(res - 1, max(0, int(q * res)))
        row = min(res - 1, max(0, int((1.0 - p) * res)))
        return (
            float(payload["nu"][row, col]),
            float(payload["lyap"][row, col]),
        )

    # -- geometry -----------------------------------------------------
    def _plot_rect(self):
        left, right, top, bottom = 54, 78, 54, 48
        avail_w = self.width() - left - right
        avail_h = self.height() - top - bottom
        side = min(avail_w, avail_h)
        if side < 120:
            return QRectF()
        x = left + max(0.0, (avail_w - side) / 2.0)
        y = top + max(0.0, (avail_h - side) / 2.0)
        return QRectF(x, y, side, side)

    def _to_qp(self, pos, plot):
        q = (pos.x() - plot.left()) / plot.width()
        p = 1.0 - (pos.y() - plot.top()) / plot.height()
        return min(1.0, max(0.0, q)), min(1.0, max(0.0, p))

    # -- interaction --------------------------------------------------
    def mouseMoveEvent(self, event):
        plot = self._plot_rect()
        if plot.isEmpty() or not plot.contains(event.position()):
            if self._hover is not None:
                self._hover = None
                self.cursorLeft.emit()
                self.update()
            return
        q, p = self._to_qp(event.position(), plot)
        self._hover = (event.position(), q, p)
        self.cursorMoved.emit(q, p)
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        plot = self._plot_rect()
        if plot.isEmpty() or not plot.contains(event.position()):
            return
        q, p = self._to_qp(event.position(), plot)
        self.probed.emit(q, p)

    def leaveEvent(self, event):
        if self._hover is not None:
            self._hover = None
            self.cursorLeft.emit()
            self.update()
        super().leaveEvent(event)

    # -- painting -----------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        background = QLinearGradient(0, 0, 0, self.height())
        background.setColorAt(0.0, CANVAS_TOP)
        background.setColorAt(1.0, CANVAS_BOTTOM)
        painter.fillRect(self.rect(), background)

        plot = self._plot_rect()
        if plot.isEmpty():
            painter.end()
            return

        self._paint_header(painter, plot)
        painter.fillRect(plot, PLOT_EMPTY)
        if self._image is not None:
            painter.drawImage(plot, self._image)
        self._paint_grid(painter, plot)
        if self._orbit is not None:
            self._paint_orbit(painter, plot)
        painter.setPen(QPen(BORDER, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(plot)
        self._paint_axes(painter, plot)
        self._paint_colorbar(painter, plot)

        if self._hover is not None:
            self._paint_hover(painter, plot)
        if self._loading:
            self._paint_loading(painter, plot)
        elif self._error:
            self._paint_error(painter, plot)
        painter.end()

    def _paint_header(self, painter, plot):
        painter.setFont(self._font_title)
        painter.setPen(INK)
        painter.drawText(QPointF(plot.left(), 22), "KAM 频率图")

        payload = self._payload
        painter.setFont(self._font_sub)
        painter.setPen(MUTED)
        if payload:
            prefix = "实时 · " if payload.get("fast") else ""
            text = (
                f"{prefix}K = {payload['K']:.3f}   ·   {payload['res']}^2 采样   ·   "
                f"{payload['iters']} 步   ·   {payload['elapsed']:.2f} s"
            )
        else:
            text = "Chirikov 标准映射 · 旋转数频率图"
        painter.drawText(QPointF(plot.left(), 38), text)

    def _paint_grid(self, painter, plot):
        ticks = (0.0, 0.25, 0.5, 0.75, 1.0)
        painter.setPen(QPen(FAINT, 1))
        for t in ticks:
            x = plot.left() + t * plot.width()
            y = plot.bottom() - t * plot.height()
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

    def _paint_axes(self, painter, plot):
        ticks = (0.0, 0.25, 0.5, 0.75, 1.0)
        labels = ("0", "0.25", "0.5", "0.75", "1")
        painter.setFont(self._font_tick)
        metrics = QFontMetrics(self._font_tick)
        painter.setPen(QColor("#8d938f"))
        for t, label in zip(ticks, labels):
            x = plot.left() + t * plot.width()
            y = plot.bottom() - t * plot.height()
            painter.drawText(
                QPointF(x - metrics.horizontalAdvance(label) / 2.0, plot.bottom() + 15),
                label,
            )
            painter.drawText(
                QPointF(plot.left() - 8 - metrics.horizontalAdvance(label),
                        y + metrics.ascent() / 2.0 - 2),
                label,
            )

        axis_font = QFont("Segoe UI", 11)
        axis_font.setItalic(True)
        painter.setFont(axis_font)
        painter.setPen(QColor("#d2cdc0"))
        painter.drawText(
            QRectF(plot.left(), plot.bottom() + 23, plot.width(), 20),
            Qt.AlignCenter,
            "q",
        )
        painter.save()
        painter.translate(plot.left() - 30, plot.center().y())
        painter.rotate(-90)
        painter.drawText(
            QRectF(-plot.height() / 2.0, -11, plot.height(), 22),
            Qt.AlignCenter,
            "p",
        )
        painter.restore()

    def _paint_colorbar(self, painter, plot):
        bar = QRectF(plot.right() + 16, plot.top(), 10, plot.height())
        painter.drawImage(bar, self._colorbar)
        painter.setPen(QPen(BORDER, 1))
        painter.drawRect(bar)

        painter.setFont(self._font_tick)
        painter.setPen(QColor("#9aa09b"))
        painter.drawText(QPointF(bar.left(), plot.top() - 8), "ν")
        metrics = QFontMetrics(self._font_tick)
        for value, label in ((1.0, "1"), (0.5, "0.5"), (0.0, "0")):
            y = plot.bottom() - value * plot.height()
            painter.drawLine(QPointF(bar.right(), y), QPointF(bar.right() + 4, y))
            painter.drawText(QPointF(bar.right() + 7, y + metrics.ascent() / 2.0 - 2), label)

    def _paint_orbit(self, painter, plot):
        points = QPolygonF([
            QPointF(
                plot.left() + float(q) * plot.width(),
                plot.bottom() - float(p) * plot.height(),
            )
            for q, p in self._orbit[::2]
        ])
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 246, 224, 24), 2.8))
        painter.drawPoints(points)
        painter.setPen(QPen(QColor(255, 248, 232, 180), 1.0))
        painter.drawPoints(points)

        q0, p0 = self._orbit[0]
        x = plot.left() + float(q0) * plot.width()
        y = plot.bottom() - float(p0) * plot.height()
        painter.setPen(QPen(QColor(5, 7, 10, 210), 2))
        painter.setBrush(QColor("#fff4d8"))
        painter.drawEllipse(QPointF(x, y), 3.1, 3.1)

        tag = "混沌" if self._orbit_lyap >= CHAOS_LYAP else "规则"
        text = f"轨道   ν {self._orbit_nu:.4f}   λ {self._orbit_lyap:.4f}   {tag}"
        font = QFont(FAMILY, 8)
        metrics = QFontMetrics(font)
        box = QRectF(
            plot.left() + 10,
            plot.top() + 10,
            metrics.horizontalAdvance(text) + 18,
            metrics.height() + 10,
        )
        chip = QPainterPath()
        chip.addRoundedRect(box, 4, 4)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(7, 9, 12, 230))
        painter.drawPath(chip)
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(chip)
        painter.setFont(font)
        painter.setPen(QColor("#f3eddc"))
        painter.drawText(box, Qt.AlignCenter, text)

    def _paint_hover(self, painter, plot):
        pos, q, p = self._hover
        painter.setPen(QPen(QColor(255, 255, 255, 48), 1, Qt.DashLine))
        painter.drawLine(QPointF(pos.x(), plot.top()), QPointF(pos.x(), plot.bottom()))
        painter.drawLine(QPointF(plot.left(), pos.y()), QPointF(plot.right(), pos.y()))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 248, 232, 220))
        painter.drawEllipse(pos, 2.4, 2.4)

        lines = [f"q {q:.3f}   p {p:.3f}"]
        sample = self.sample_frequency(q, p)
        if sample is not None:
            nu, lyap = sample
            tag = "混沌" if lyap >= CHAOS_LYAP else "规则"
            lines.append(f"ν {nu:.3f} · λ {lyap:.3f} · {tag}")

        metrics = QFontMetrics(self._font_chip)
        width = max(metrics.horizontalAdvance(line) for line in lines) + 16
        height = len(lines) * (metrics.height() + 2) + 10
        x = pos.x() + 14
        y = pos.y() - height - 12
        if x + width > self.width() - 8:
            x = pos.x() - width - 14
        if x < 8:
            x = 8
        if y < 8:
            y = pos.y() + 14
        if y + height > self.height() - 8:
            y = self.height() - height - 8

        box = QRectF(x, y, width, height)
        path = QPainterPath()
        path.addRoundedRect(box, 4, 4)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(7, 9, 12, 228))
        painter.drawPath(path)
        painter.setPen(QPen(QColor(255, 255, 255, 44), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        painter.setFont(self._font_chip)
        painter.setPen(QColor("#ece8dd"))
        for i, line in enumerate(lines):
            painter.drawText(
                QRectF(x + 8, y + 5 + i * (metrics.height() + 2), width - 16, metrics.height()),
                Qt.AlignLeft | Qt.AlignVCenter,
                line,
            )

    def _paint_loading(self, painter, plot):
        painter.fillRect(plot, QColor(5, 7, 10, 190))
        painter.setFont(QFont(FAMILY, 10, QFont.Bold))
        painter.setPen(QColor("#fff8e8"))
        painter.drawText(plot, Qt.AlignCenter, f"计算中  {self._progress}%")

        track = QRectF(plot.left() + 24, plot.bottom() - 18, plot.width() - 48, 2)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 32))
        painter.drawRoundedRect(track, 1, 1)
        filled = QRectF(track.left(), track.top(), track.width() * self._progress / 100.0, 2)
        painter.setBrush(ACCENT)
        painter.drawRoundedRect(filled, 1, 1)

    def _paint_error(self, painter, plot):
        painter.fillRect(plot, QColor(7, 9, 12, 210))
        painter.setFont(QFont(FAMILY, 9))
        painter.setPen(QColor("#e8b48a"))
        painter.drawText(
            plot.adjusted(24, 24, -24, -24),
            Qt.AlignCenter | Qt.TextWordWrap,
            self._error,
        )


class PlanetCanvas(QWidget):
    """Top-down solar-system view with real orbital elements."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._view = "full"
        self._speed = 200.0
        self._trail_deg = 180
        self._precession = True
        self._time_scale = 10.0
        self._show_belt = True
        self._show_labels = True
        self._paused = False
        self._years = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

        rng = random.Random(17)
        self._stars = [
            (rng.random(), rng.random(), rng.random()) for _ in range(180)
        ]
        self._background = None

        self._font_title = QFont(FAMILY, 10, QFont.Bold)
        self._font_sub = QFont(FAMILY, 8)
        self._font_tiny = QFont(FAMILY, 7)

    # -- animation ----------------------------------------------------
    def start(self):
        if not self._timer.isActive():
            self._timer.start()

    def stop(self):
        self._timer.stop()

    def is_paused(self):
        return self._paused

    def set_view(self, view):
        self._view = view
        self.update()

    def set_speed(self, value):
        self._speed = float(value)

    def set_trail(self, value):
        self._trail_deg = max(0, min(36000, int(value)))
        self.update()

    def set_precession(self, show):
        self._precession = bool(show)
        self.update()

    def set_time_scale(self, value):
        self._time_scale = max(1.0, min(1000.0, float(value)))

    def set_show_belt(self, show):
        self._show_belt = bool(show)
        self.update()

    def set_show_labels(self, show):
        self._show_labels = bool(show)
        self.update()

    def set_paused(self, paused):
        self._paused = bool(paused)
        self.update()

    def reset(self):
        self._years = 0.0
        self.update()

    def resizeEvent(self, event):
        self._background = None
        super().resizeEvent(event)

    def _rebuild_background(self):
        w = self.width()
        h = self.height()
        pixmap = QPixmap(self.size())
        pixmap.fill(QColor("#030509"))
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing, True)
        background = QRadialGradient(
            QPointF(w * 0.5, h * 0.47), max(w, h) * 0.72
        )
        background.setColorAt(0.0, QColor("#0b1424"))
        background.setColorAt(0.55, QColor("#060b13"))
        background.setColorAt(1.0, QColor("#030509"))
        p.fillRect(pixmap.rect(), QBrush(background))
        for nx, ny, glow in self._stars:
            alpha = 30 + int(120 * glow)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(205, 220, 255, alpha))
            p.drawEllipse(
                QPointF(nx * w, ny * h), 0.8 + glow, 0.8 + glow
            )
        p.end()
        self._background = pixmap

    def _tick(self):
        if self._paused:
            return
        self._years += 0.15 * self._speed / 100.0 * self._time_scale
        self.update()

    def _precession_rate(self, planet):
        if not self._precession:
            return 0.0
        return PRECESSION_ARCSEC_PER_YEAR.get(planet["name"], 0.0) / 3600.0

    def _trail_points(self, planet):
        if self._trail_deg <= 0:
            return None
        rate = self._precession_rate(planet)
        if self._view == "inner" or planet["a"] < 2.0:
            samples = 7200
        else:
            samples = 1100
        samples = max(240, min(samples, int(self._trail_deg / 1.25)))
        return planet_arc(
            planet, self._years, self._trail_deg,
            precession_deg_per_year=rate, samples=samples,
        )

    # -- projection ---------------------------------------------------
    def _max_au(self):
        return 2.35 if self._view == "inner" else 32.0

    def _screen(self, x, y, cx, cy, radius):
        r = float(np.hypot(x, y))
        max_au = self._max_au()
        if self._view == "full":
            rr = np.sqrt(min(r, max_au) / max_au) * radius
        else:
            rr = min(r, max_au) / max_au * radius
        factor = rr / r if r > 1e-12 else 0.0
        return QPointF(cx + x * factor, cy - y * factor)

    def _view_label(self):
        return {
            "full": "全系统 · 径向 √ 压缩",
            "inner": "内行星 · 线性 AU",
            "outer": "外行星 · 线性 AU",
        }[self._view]

    @staticmethod
    def _smooth_path(points, closed=False):
        path = QPainterPath()
        if len(points) < 2:
            return path
        path.moveTo(points[0])
        for i in range(1, len(points) - 1):
            p = points[i]
            n = points[i + 1]
            mid = QPointF(
                (p.x() + n.x()) * 0.5,
                (p.y() + n.y()) * 0.5,
            )
            path.quadTo(p, mid)
        path.lineTo(points[-1])
        if closed:
            path.closeSubpath()
        return path

    # -- painting -----------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()
        if self._background is None or self._background.size() != self.size():
            self._rebuild_background()
        painter.drawPixmap(0, 0, self._background)

        cx = w * 0.5
        cy = h * 0.5
        radius = min(w, h) * 0.405
        max_au = self._max_au()

        # Orbits
        for planet in PLANETS:
            if self._view == "inner" and planet["a"] > max_au:
                continue
            rate = self._precession_rate(planet)
            points = planet_orbit_at(
                planet, self._years, precession_deg_per_year=rate
            )
            screen_points = [
                self._screen(float(x), float(y), cx, cy, radius)
                for x, y in points
            ]
            color = QColor(planet["color"])
            color.setAlpha(100 if planet["a"] > 2.0 else 118)
            painter.setPen(QPen(color, 1.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(self._smooth_path(screen_points, closed=True))

        # Asteroid belt with Kirkwood gaps
        if self._show_belt:
            positions = asteroid_positions(BELT, self._years)
            belt_color = QColor(198, 178, 142, 95)
            painter.setPen(QPen(belt_color, 0.9))
            painter.setBrush(Qt.NoBrush)
            dots = QPolygonF()
            for x, y in positions:
                if np.hypot(x, y) > max_au:
                    continue
                dots.append(self._screen(float(x), float(y), cx, cy, radius))
            painter.drawPoints(dots)

        # Sun glow
        glow = QRadialGradient(QPointF(cx, cy), 38)
        glow.setColorAt(0.0, QColor(255, 247, 198, 235))
        glow.setColorAt(0.18, QColor(255, 221, 133, 170))
        glow.setColorAt(0.55, QColor(255, 180, 80, 45))
        glow.setColorAt(1.0, QColor(255, 150, 40, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(QPointF(cx, cy), 38, 38)
        painter.setBrush(QColor("#fff2bd"))
        painter.drawEllipse(QPointF(cx, cy), 6.0, 6.0)
        painter.setFont(self._font_tiny)
        painter.setPen(QColor(255, 234, 176, 190))
        painter.drawText(QPointF(cx + 9, cy + 13), "太阳")

        # Planets, trails, labels
        for planet in PLANETS:
            rate = self._precession_rate(planet)
            x, y = planet_position_at(
                planet, self._years, precession_deg_per_year=rate
            )
            r = float(np.hypot(x, y))
            if r > max_au:
                continue
            pos = self._screen(x, y, cx, cy, radius)
            color = QColor(planet["color"])

            trail = self._trail_points(planet)
            if trail is not None and len(trail) > 2:
                screen_trail = [
                    self._screen(float(px), float(py), cx, cy, radius)
                    for px, py in trail
                ]
                count = len(screen_trail)
                laps = max(1, int(np.ceil(self._trail_deg / 360.0)))
                chunk = max(2, int(np.ceil((count - 1) / laps)))
                for start in range(0, count - 1, chunk):
                    end = min(count, start + chunk + 1)
                    segment = screen_trail[start:end]
                    if len(segment) < 2:
                        continue
                    frac = end / count
                    trail_color = QColor(color)
                    trail_color.setAlpha(int(25 + 200 * frac))
                    painter.setPen(QPen(trail_color, 0.9 + 0.9 * frac))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(self._smooth_path(segment))

            halo = QColor(color)
            halo.setAlpha(45)
            painter.setPen(Qt.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(pos, planet["radius"] * 2.4, planet["radius"] * 2.4)
            painter.setBrush(color)
            painter.drawEllipse(pos, planet["radius"], planet["radius"])

            if planet.get("ring"):
                ring = QColor(color)
                ring.setAlpha(170)
                painter.setPen(QPen(ring, 1.0))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(
                    pos, planet["radius"] * 1.95, planet["radius"] * 0.62
                )

            if self._show_labels and not (self._view == "full" and r < 1.6):
                painter.setFont(self._font_tiny)
                painter.setPen(QColor(planet["color"]).lighter(120))
                painter.drawText(
                    QPointF(
                        pos.x() + planet["radius"] + 4,
                        pos.y() - planet["radius"] - 3,
                    ),
                    planet["name"],
                )

        # Current orbit on top of the historical trail web
        for planet in PLANETS:
            if self._view == "inner" and planet["a"] > max_au:
                continue
            points = planet_orbit_at(
                planet, self._years,
                precession_deg_per_year=self._precession_rate(planet),
            )
            screen_points = [
                self._screen(float(x), float(y), cx, cy, radius)
                for x, y in points
            ]
            color = QColor(248, 242, 226, 205)
            painter.setPen(QPen(color, 1.35))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(self._smooth_path(screen_points, closed=True))

        # HUD
        painter.setFont(self._font_title)
        painter.setPen(QColor("#f1ebdd"))
        painter.drawText(QPointF(24, 27), "太阳系 · 行星轨道")
        painter.setFont(self._font_sub)
        painter.setPen(QColor("#8793a1"))
        painter.drawText(QPointF(24, 43), "J2000 轨道要素 · 顶视投影")

        painter.setFont(self._font_sub)
        painter.setPen(QColor("#dfe4ea"))
        painter.drawText(
            QRectF(w - 330, 15, 306, 18),
            Qt.AlignRight | Qt.AlignVCenter,
            f"t = {self._years:7.1f} 年",
        )
        painter.setPen(QColor("#8793a1"))
        painter.drawText(
            QRectF(w - 330, 33, 306, 18),
            Qt.AlignRight | Qt.AlignVCenter,
            self._view_label(),
        )
        if self._paused:
            painter.setPen(QColor("#f0c879"))
            painter.drawText(
                QRectF(w - 330, 50, 306, 18),
                Qt.AlignRight | Qt.AlignVCenter,
                "暂停",
            )

        painter.setFont(self._font_tiny)
        painter.setPen(QColor("#788492"))
        painter.drawText(
            QPointF(24, h - 38),
            "小行星带 2.1–3.3 AU · Kirkwood gaps = 共振",
        )

        x0 = 24
        for planet in PLANETS:
            color = QColor(planet["color"])
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x0, h - 19), 2.6, 2.6)
            painter.setPen(QColor(190, 198, 210))
            painter.drawText(QPointF(x0 + 7, h - 15), planet["name"])
            x0 += 55

        painter.end()


class PoincareCanvas(QWidget):
    """Dark Poincaré section for the Sun–Jupiter CRTBP."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._points = None
        self._C = 3.0
        self._loading = False
        self._error = None

    def set_points(self, points, C):
        self._points = points
        self._C = float(C)
        self._loading = False
        self._error = None
        self.update()

    def begin_loading(self):
        self._loading = True
        self._error = None
        self.update()

    def set_error(self, message):
        self._loading = False
        self._error = str(message)
        self.update()

    def _plot_rect(self):
        left, right, top, bottom = 58, 30, 54, 50
        side = min(self.width() - left - right, self.height() - top - bottom)
        if side < 120:
            return QRectF()
        x = left + max(0.0, (self.width() - left - right - side) / 2.0)
        y = top + max(0.0, (self.height() - top - bottom - side) / 2.0)
        return QRectF(x, y, side, side)

    @staticmethod
    def _to_screen(x, vx, rect):
        return QPointF(
            rect.left() + (x - 0.60) / 0.80 * rect.width(),
            rect.bottom() - (vx + 0.80) / 1.60 * rect.height(),
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        background = QLinearGradient(0, 0, 0, self.height())
        background.setColorAt(0.0, QColor("#07090d"))
        background.setColorAt(1.0, QColor("#0b0f14"))
        painter.fillRect(self.rect(), background)

        plot = self._plot_rect()
        if plot.isEmpty():
            painter.end()
            return

        painter.setFont(QFont(FAMILY, 10, QFont.Bold))
        painter.setPen(QColor("#ece7da"))
        painter.drawText(QPointF(plot.left(), 22), "太阳–木星 CRTBP · Poincaré 截面")
        painter.setFont(QFont(FAMILY, 8))
        painter.setPen(QColor("#8f9691"))
        painter.drawText(
            QPointF(plot.left(), 38),
            f"y = 0, vy > 0   ·   C = {self._C:.4f}",
        )

        painter.fillRect(plot, QColor("#0c1015"))
        painter.setPen(QPen(QColor(255, 255, 255, 38), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(plot)

        ticks = (0.0, 0.25, 0.5, 0.75, 1.0)
        tick_font = QFont("Segoe UI", 7)
        painter.setFont(tick_font)
        fm = QFontMetrics(tick_font)
        painter.setPen(QPen(QColor(255, 255, 255, 12), 1))
        for t in ticks:
            x = plot.left() + t * plot.width()
            y = plot.bottom() - t * plot.height()
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        painter.setPen(QColor("#8d938f"))
        for t, label in zip(ticks, ("0.60", "0.80", "1.00", "1.20", "1.40")):
            x = plot.left() + t * plot.width()
            painter.drawText(
                QPointF(x - fm.horizontalAdvance(label) / 2.0, plot.bottom() + 15),
                label,
            )
        for t, label in zip(ticks, ("-0.8", "-0.4", "0.0", "0.4", "0.8")):
            y = plot.bottom() - t * plot.height()
            painter.drawText(
                QPointF(plot.left() - 8 - fm.horizontalAdvance(label),
                        y + fm.ascent() / 2.0 - 2),
                label,
            )
        painter.setFont(QFont(FAMILY, 9))
        painter.setPen(QColor("#c8c3b6"))
        painter.drawText(
            QRectF(plot.left(), plot.bottom() + 25, plot.width(), 20),
            Qt.AlignCenter,
            "x",
        )
        painter.save()
        painter.translate(plot.left() - 36, plot.center().y())
        painter.rotate(-90)
        painter.drawText(
            QRectF(-plot.height() / 2.0, -11, plot.height(), 22),
            Qt.AlignCenter,
            "vx",
        )
        painter.restore()

        if self._points is not None and len(self._points) > 0:
            points = QPolygonF([
                self._to_screen(float(x), float(vx), plot)
                for x, vx in self._points
            ])
            painter.setPen(QPen(QColor(240, 200, 121, 175), 1.15))
            painter.setBrush(Qt.NoBrush)
            painter.drawPoints(points)
        else:
            painter.setFont(QFont(FAMILY, 9))
            painter.setPen(QColor("#65707c"))
            painter.drawText(plot, Qt.AlignCenter, "调节 C 后点击计算")

        if self._loading:
            painter.fillRect(plot, QColor(5, 7, 10, 170))
            painter.setFont(QFont(FAMILY, 10, QFont.Bold))
            painter.setPen(QColor("#fff8e8"))
            painter.drawText(plot, Qt.AlignCenter, "计算截面中…")
        elif self._error:
            painter.fillRect(plot, QColor(5, 7, 10, 190))
            painter.setFont(QFont(FAMILY, 9))
            painter.setPen(QColor("#e8b48a"))
            painter.drawText(
                plot.adjusted(24, 24, -24, -24),
                Qt.AlignCenter | Qt.TextWordWrap,
                self._error,
            )
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1180, 800)
        self.setMinimumSize(1020, 660)

        self._worker = None
        self._generation = 0
        self._current_payload = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(320)
        self._debounce.timeout.connect(self._start_compute)

        self._fast_timer = QTimer(self)
        self._fast_timer.setInterval(30)
        self._fast_timer.timeout.connect(self._tick_fast)
        self._fast_single = QTimer(self)
        self._fast_single.setSingleShot(True)
        self._fast_single.setInterval(30)
        self._fast_single.timeout.connect(lambda: self._start_compute(fast=True))
        self._k_dragging = False
        self._fast_k = None

        self._mode = 0
        self._build_ui()
        self._build_status()
        self._set_mode(0)
        QTimer.singleShot(140, self._start_compute)

    # -- UI -----------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(326)
        panel.setAutoFillBackground(False)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 4, 4)
        layout.setSpacing(6)

        title = QLabel("KAM")
        title.setFont(QFont(FAMILY, 18, QFont.Bold))
        layout.addWidget(title)

        self._control_tabs = QTabWidget()
        self._control_tabs.setUsesScrollButtons(False)
        self.standard_page = self._build_standard_page()
        self.planet_page = self._build_planet_page()
        self.crtbp_page = self._build_crtbp_page()
        self._control_tabs.addTab(self.standard_page, "标准映射")
        self._control_tabs.addTab(self.planet_page, "行星轨道")
        self._control_tabs.addTab(self.crtbp_page, "小行星带截面")
        self._control_tabs.currentChanged.connect(self._set_mode)
        layout.addWidget(self._control_tabs, 1)
        root.addWidget(panel, 0)

        self.canvas = KAMCanvas()
        self.canvas.probed.connect(self._trace_orbit)
        self.canvas.cursorMoved.connect(self._on_cursor_moved)
        self.canvas.cursorLeft.connect(self._on_cursor_left)
        self.planet_canvas = PlanetCanvas()
        self.crtbp_canvas = PoincareCanvas()

        self._canvas_stack = QStackedWidget()
        self._canvas_stack.addWidget(self.canvas)
        self._canvas_stack.addWidget(self.planet_canvas)
        self._canvas_stack.addWidget(self.crtbp_canvas)
        root.addWidget(self._canvas_stack, 1)

    def _build_standard_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(self._muted(
            "p′ = p + (K/2π)·sin(2πq)    q′ = q + p′", 9
        ))

        params = QGroupBox("参数")
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(6, 6, 6, 6)
        params_layout.setSpacing(5)

        self.k_slider = LabeledSlider(
            "K", 0, 300, 97,
            formatter=lambda value: f"{value / 100.0:.2f}",
            label_width=16, value_width=38,
        )
        self.k_slider.valueChanged.connect(self._on_k_changed)
        self.k_slider.slider.sliderPressed.connect(self._begin_k_drag)
        self.k_slider.slider.sliderReleased.connect(self._end_k_drag)
        params_layout.addWidget(self.k_slider)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        preset_label = QLabel("预设")
        preset_label.setStyleSheet("color:#6b7280;")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            ("自定义", "0.00", "0.50", "0.97", "1.50", "2.50")
        )
        self.preset_combo.setCurrentText("0.97")
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        preset_row.addWidget(preset_label)
        preset_row.addWidget(self.preset_combo, 1)
        params_layout.addLayout(preset_row)

        params_layout.addWidget(self._muted(
            "K_c ≈ 0.9716 · 最后一条 KAM 环面", 9
        ))

        self.res_slider = LabeledSlider(
            "采样", 220, 620, 360,
            formatter=lambda value: f"{value}²",
            label_width=30, value_width=44,
        )
        self.iter_slider = LabeledSlider(
            "迭代", 80, 360, 220,
            formatter=str,
            label_width=30, value_width=44,
        )
        self.res_slider.valueChanged.connect(self._on_parameter_changed)
        self.iter_slider.valueChanged.connect(self._on_parameter_changed)
        params_layout.addWidget(self.res_slider)
        params_layout.addWidget(self.iter_slider)
        layout.addWidget(params)

        control = QGroupBox("控制")
        control_layout = QVBoxLayout(control)
        control_layout.setContentsMargins(6, 6, 6, 6)
        control_layout.setSpacing(5)
        run_row = QHBoxLayout()
        run_row.setSpacing(4)
        self.compute_btn = QPushButton("重算")
        self.stop_btn = QPushButton("停止")
        self.save_btn = QPushButton("保存")
        for button in (self.compute_btn, self.stop_btn, self.save_btn):
            run_row.addWidget(button, 1)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.compute_btn.clicked.connect(lambda: self._start_compute(fast=False))
        self.stop_btn.clicked.connect(self._stop_compute)
        self.save_btn.clicked.connect(self._save_image)
        control_layout.addLayout(run_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(6)
        self.progress.setVisible(False)
        control_layout.addWidget(self.progress)
        layout.addWidget(control)

        probe = QGroupBox("轨道探测")
        probe_layout = QVBoxLayout(probe)
        probe_layout.setContentsMargins(6, 6, 6, 6)
        probe_layout.setSpacing(5)
        self.orbit_info = QLabel("点击相图取初值")
        self.orbit_info.setWordWrap(True)
        self.clear_orbit_btn = QPushButton("清除轨道")
        self.clear_orbit_btn.setEnabled(False)
        self.clear_orbit_btn.clicked.connect(self._clear_orbit)
        probe_layout.addWidget(self.orbit_info)
        probe_layout.addWidget(self.clear_orbit_btn)
        layout.addWidget(probe)

        layout.addWidget(self._muted(
            "颜色 ν · 暗区 λ>0 混沌 · 拖动 K 实时", 9
        ))
        layout.addStretch(1)
        return page

    def _build_planet_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(self._muted("J2000 轨道要素 · 真实周期", 9))

        display = QGroupBox("视图")
        display_layout = QVBoxLayout(display)
        display_layout.setContentsMargins(6, 6, 6, 6)
        display_layout.setSpacing(5)

        view_row = QHBoxLayout()
        view_row.setSpacing(4)
        view_label = QLabel("视野")
        self.view_combo = QComboBox()
        self.view_combo.addItems(("全系统（√）", "内行星", "外行星"))
        self.view_combo.currentIndexChanged.connect(self._planet_view_changed)
        view_row.addWidget(view_label)
        view_row.addWidget(self.view_combo, 1)
        display_layout.addLayout(view_row)

        self.speed_slider = LabeledSlider(
            "速度", 10, 1000, 200,
            formatter=lambda value: f"{value}%",
            label_width=30, value_width=48,
        )
        self.trail_slider = LabeledSlider(
            "拖尾", 0, 360, 180,
            formatter=lambda value: f"{value}°",
            label_width=30, value_width=48,
        )
        self.time_slider = LabeledSlider(
            "时间 ×", 1, 1000, 10,
            formatter=lambda value: f"×{value}",
            label_width=42, value_width=46,
        )
        self.speed_slider.valueChanged.connect(self._planet_speed_changed)
        self.trail_slider.valueChanged.connect(self._planet_trail_changed)
        self.time_slider.valueChanged.connect(self._planet_time_scale_changed)
        display_layout.addWidget(self.speed_slider)
        display_layout.addWidget(self.time_slider)
        display_layout.addWidget(self.trail_slider)

        checks = QHBoxLayout()
        checks.setSpacing(8)
        self.belt_check = QCheckBox("小行星带")
        self.labels_check = QCheckBox("名称")
        self.precession_check = QCheckBox("真实进动")
        self.belt_check.setChecked(True)
        self.labels_check.setChecked(True)
        self.precession_check.setChecked(True)
        self.belt_check.toggled.connect(self._planet_toggle_belt)
        self.labels_check.toggled.connect(self._planet_toggle_labels)
        self.precession_check.toggled.connect(self._planet_toggle_precession)
        checks.addWidget(self.belt_check)
        checks.addWidget(self.labels_check)
        checks.addWidget(self.precession_check)
        checks.addStretch(1)
        display_layout.addLayout(checks)
        layout.addWidget(display)

        control = QGroupBox("控制")
        control_layout = QHBoxLayout(control)
        control_layout.setContentsMargins(6, 6, 6, 6)
        control_layout.setSpacing(4)
        self.pause_btn = QPushButton("暂停")
        self.reset_btn = QPushButton("重置")
        control_layout.addWidget(self.pause_btn, 1)
        control_layout.addWidget(self.reset_btn, 1)
        self.pause_btn.clicked.connect(self._planet_toggle_pause)
        self.reset_btn.clicked.connect(self._planet_reset)
        layout.addWidget(control)

        layout.addWidget(self._muted(
            "真实进动（水星 5.75″/年）· 时间加速 ×10", 9
        ))
        layout.addStretch(1)
        return page

    def _build_crtbp_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(self._muted(
            "太阳–木星 CRTBP · y=0 截面；KAM 环面 / 岛链 / 混沌海", 9
        ))

        params = QGroupBox("参数")
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(6, 6, 6, 6)
        params_layout.setSpacing(5)
        self.c_slider = LabeledSlider(
            "C", 2800, 3200, 3000,
            formatter=lambda value: f"{value / 1000.0:.3f}",
            label_width=18, value_width=52,
        )
        self.orbits_slider = LabeledSlider(
            "轨道", 20, 200, 120,
            formatter=str,
            label_width=30, value_width=44,
        )
        self.steps_slider = LabeledSlider(
            "步数", 1000, 6000, 3000,
            formatter=str,
            label_width=30, value_width=48,
        )
        params_layout.addWidget(self.c_slider)
        params_layout.addWidget(self.orbits_slider)
        params_layout.addWidget(self.steps_slider)
        layout.addWidget(params)

        control = QGroupBox("控制")
        control_layout = QVBoxLayout(control)
        control_layout.setContentsMargins(6, 6, 6, 6)
        control_layout.setSpacing(5)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.crtbp_compute_btn = QPushButton("计算截面")
        self.crtbp_save_btn = QPushButton("保存")
        self.crtbp_save_btn.setEnabled(False)
        self.crtbp_compute_btn.clicked.connect(self._compute_crtbp)
        self.crtbp_save_btn.clicked.connect(self._save_crtbp)
        row.addWidget(self.crtbp_compute_btn, 1)
        row.addWidget(self.crtbp_save_btn, 1)
        control_layout.addLayout(row)
        self.crtbp_progress = QProgressBar()
        self.crtbp_progress.setRange(0, 100)
        self.crtbp_progress.setValue(0)
        self.crtbp_progress.setTextVisible(False)
        self.crtbp_progress.setMaximumHeight(6)
        self.crtbp_progress.setVisible(False)
        control_layout.addWidget(self.crtbp_progress)
        layout.addWidget(control)

        layout.addWidget(self._muted(
            "截面点来自真实三体积分；C 控制 Jacobi 能量。", 9
        ))
        layout.addStretch(1)
        return page

    @staticmethod
    def _spacer(height):
        widget = QWidget()
        widget.setFixedHeight(int(height))
        return widget

    @staticmethod
    def _muted(text, size=9):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setMinimumHeight(18)
        label.setStyleSheet(f"color: #6b7280; font-size: {int(max(size, 9))}px;")
        return label

    def _build_equation_group(self):
        group = QGroupBox("标准映射")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        equation = QLabel(
            "p′ = p + (K / 2π) sin(2πq)   (mod 1)<br>"
            "q′ = q + p′   (mod 1)"
        )
        equation.setTextFormat(Qt.RichText)
        equation.setWordWrap(True)
        equation.setFont(QFont(FAMILY, 10))
        layout.addWidget(equation)
        legend = QLabel(
            "颜色 = ν　亮带 = KAM 环面 / 岛链<br>暗区 = 混沌（λ &gt; 0）"
        )
        legend.setTextFormat(Qt.RichText)
        legend.setWordWrap(True)
        legend.setMinimumHeight(30)
        legend.setStyleSheet("color: #6b7280; font-size: 9px;")
        layout.addWidget(legend)
        return group

    def _build_parameter_group(self):
        group = QGroupBox("参数")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.k_slider = LabeledSlider(
            "扰动 K", 0, 300, 97,
            formatter=lambda value: f"{value / 100.0:.2f}",
            value_width=44,
        )
        self.res_slider = LabeledSlider(
            "采样", 220, 620, 360,
            formatter=lambda value: f"{value}²",
            value_width=50,
        )
        self.iter_slider = LabeledSlider(
            "迭代", 80, 360, 220,
            formatter=str,
            value_width=44,
        )
        self.k_slider.valueChanged.connect(self._on_k_changed)
        self.k_slider.slider.sliderPressed.connect(self._begin_k_drag)
        self.k_slider.slider.sliderReleased.connect(self._end_k_drag)
        self.res_slider.valueChanged.connect(self._on_parameter_changed)
        self.iter_slider.valueChanged.connect(self._on_parameter_changed)
        layout.addWidget(self.k_slider)
        kc_hint = self._muted("K_c ≈ 0.9716：最后一条 KAM 环面")
        kc_hint.setMinimumHeight(16)
        layout.addWidget(kc_hint)
        layout.addWidget(self.res_slider)
        layout.addWidget(self.iter_slider)
        sample_hint = self._muted("拖动 K 实时刷新；采样越大，岛链与暗区边界越细。")
        sample_hint.setMinimumHeight(30)
        layout.addWidget(sample_hint)
        return group

    def _build_run_group(self):
        group = QGroupBox("运行")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.compute_btn = QPushButton("重新计算")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.compute_btn.clicked.connect(lambda: self._start_compute(fast=False))
        self.stop_btn.clicked.connect(self._stop_compute)
        row.addWidget(self.compute_btn, 1)
        row.addWidget(self.stop_btn, 1)
        layout.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.save_btn = QPushButton("保存图像")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_image)
        layout.addWidget(self.save_btn)
        return group

    def _build_preset_group(self):
        group = QGroupBox("预设 K")
        grid = QGridLayout(group)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(4)
        presets = (("0.00", 0.0), ("0.50", 0.5), ("0.97", 0.97),
                   ("1.50", 1.5), ("2.50", 2.5))
        for index, (text, value) in enumerate(presets):
            button = QPushButton(text)
            button.setFixedHeight(24)
            button.clicked.connect(lambda checked=False, v=value: self._set_preset(v))
            grid.addWidget(button, index // 3, index % 3)
        return group

    def _build_probe_group(self):
        group = QGroupBox("探测轨道")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        probe_hint = self._muted("点击相图任意初值 · 白色轨迹 + ν / λ 读数")
        probe_hint.setMinimumHeight(18)
        layout.addWidget(probe_hint)
        self.orbit_info = QLabel("尚未选择初值")
        self.orbit_info.setFixedHeight(18)
        self.orbit_info.setWordWrap(True)
        self.orbit_info.setStyleSheet("color: #6b7280; font-size: 9px;")
        layout.addWidget(self.orbit_info)
        self.clear_orbit_btn = QPushButton("清除轨道")
        self.clear_orbit_btn.setEnabled(False)
        self.clear_orbit_btn.clicked.connect(self._clear_orbit)
        layout.addWidget(self.clear_orbit_btn)
        group.setMinimumHeight(150)
        return group

    def _build_status(self):
        bar = self.statusBar()
        bar.setSizeGripEnabled(False)
        self.state_label = QLabel("就绪")
        self.state_label.setStyleSheet("color: #4b5563;")
        self.params_label = QLabel("K 0.97 · 360² · 220 步")
        self.params_label.setStyleSheet("color: #6b7280;")
        self.cursor_label = QLabel("")
        self.cursor_label.setMinimumWidth(260)
        self.cursor_label.setStyleSheet("color: #374151;")
        bar.addWidget(self.state_label, 1)
        bar.addPermanentWidget(self.params_label)
        bar.addPermanentWidget(self.cursor_label)

    # -- CRTBP section -------------------------------------------------
    def _compute_crtbp(self):
        C = self.c_slider.value() / 1000.0
        orbits = self.orbits_slider.value()
        steps = self.steps_slider.value()
        self.crtbp_compute_btn.setEnabled(False)
        self.crtbp_save_btn.setEnabled(False)
        self.crtbp_progress.setValue(0)
        self.crtbp_progress.setVisible(True)
        self.crtbp_canvas.begin_loading()
        self.state_label.setText("计算 CRTBP 截面中…")
        QApplication.processEvents()
        try:
            started = time.perf_counter()
            points = compute_poincare(C, orbits, steps)
            elapsed = time.perf_counter() - started
            self.crtbp_canvas.set_points(points, C)
            self.crtbp_progress.setValue(100)
            self.crtbp_progress.setVisible(False)
            self.crtbp_compute_btn.setEnabled(True)
            self.crtbp_save_btn.setEnabled(True)
            self.state_label.setText(f"截面完成 · {elapsed:.1f} s")
            self.params_label.setText(
                f"C {C:.3f} · {orbits} 轨道 · {steps} 步"
            )
            self.cursor_label.setText(f"{len(points)} 个截面点")
        except Exception as exc:  # noqa: BLE001
            self.crtbp_canvas.set_error(str(exc))
            self.crtbp_progress.setVisible(False)
            self.crtbp_compute_btn.setEnabled(True)
            self.state_label.setText("截面计算失败")
            QMessageBox.critical(self, "CRTBP 截面失败", str(exc))

    def _save_crtbp(self):
        path = ask_save_path(self, "保存 CRTBP 截面", suffix=".png")
        if not path:
            return
        if self.crtbp_canvas.grab().save(path):
            self.state_label.setText("已保存截面")
        else:
            QMessageBox.warning(self, "保存失败", "无法写入所选路径。")

    # -- modes --------------------------------------------------------
    def _set_mode(self, index):
        index = max(0, min(2, int(index)))
        self._mode = index
        if hasattr(self, "_control_tabs"):
            if self._control_tabs.currentIndex() != index:
                self._control_tabs.blockSignals(True)
                self._control_tabs.setCurrentIndex(index)
                self._control_tabs.blockSignals(False)
        self._canvas_stack.setCurrentIndex(index)
        if index == 1:
            self.planet_canvas.start()
            self._update_planet_status()
            return
        self.planet_canvas.stop()
        if index == 0:
            self.state_label.setText("标准映射 · 就绪")
            self._update_params_label()
            self.cursor_label.setText("")
        else:
            self.state_label.setText("小行星带截面 · 就绪")
            self.params_label.setText(
                f"太阳–木星 CRTBP · C={self.c_slider.value() / 1000.0:.3f}"
            )
            self.cursor_label.setText("截面：KAM 环面 / 岛链 / 混沌")

    def _update_planet_status(self):
        if getattr(self, "_mode", 0) != 1:
            return
        view = ("全系统", "内行星", "外行星")[
            max(0, min(2, self.view_combo.currentIndex()))
        ]
        state = "暂停" if self.planet_canvas.is_paused() else "运行中"
        self.state_label.setText(f"行星轨道 · {state}")
        self.params_label.setText(
            f"{view} · 速度 {self.speed_slider.value()}% · "
            f"时间 ×{self.time_slider.value()} · "
            f"拖尾 {self.trail_slider.value()}°"
        )
        self.cursor_label.setText("轨道 ≈ KAM 环面 · 小行星带 = 共振")

    def _planet_view_changed(self, index):
        self.planet_canvas.set_view(("full", "inner", "outer")[
            max(0, min(2, int(index)))
        ])
        self._update_planet_status()

    def _planet_speed_changed(self, value):
        self.planet_canvas.set_speed(value)
        self._update_planet_status()

    def _planet_trail_changed(self, value):
        self.planet_canvas.set_trail(value)
        self._update_planet_status()

    def _planet_time_scale_changed(self, value):
        self.planet_canvas.set_time_scale(value)
        self._update_planet_status()

    def _planet_toggle_belt(self, checked):
        self.planet_canvas.set_show_belt(checked)

    def _planet_toggle_labels(self, checked):
        self.planet_canvas.set_show_labels(checked)

    def _planet_toggle_precession(self, checked):
        self.planet_canvas.set_precession(checked)

    def _planet_toggle_pause(self):
        paused = not self.planet_canvas.is_paused()
        self.planet_canvas.set_paused(paused)
        self.pause_btn.setText("继续" if paused else "暂停")
        self._update_planet_status()

    def _planet_reset(self):
        self.planet_canvas.reset()
        self.pause_btn.setText("继续" if self.planet_canvas.is_paused() else "暂停")
        self._update_planet_status()

    # -- parameters / presets ----------------------------------------
    def _set_preset(self, value):
        self.k_slider.setValue(int(round(float(value) * 100)))

    def _on_preset_selected(self, index):
        if index <= 0:
            return
        try:
            value = float(self.preset_combo.currentText())
        except ValueError:
            return
        self._set_preset(value)

    def _reset_probe(self):
        self.canvas.clear_orbit()
        self.clear_orbit_btn.setEnabled(False)
        self.orbit_info.setText("尚未选择初值")

    def _on_parameter_changed(self, *args):
        self._fast_timer.stop()
        self._fast_single.stop()
        self._debounce.start()
        self._reset_probe()
        self.state_label.setText("等待计算…")
        self._update_params_label()

    def _on_k_changed(self, *args):
        self._reset_probe()
        self._update_params_label()
        self.state_label.setText("实时计算…")
        value = self.k_slider.value() / 100.0
        match = self.preset_combo.findText(f"{value:.2f}")
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(max(0, match))
        self.preset_combo.blockSignals(False)
        if not self._k_dragging:
            self._fast_single.start()

    def _begin_k_drag(self):
        self._k_dragging = True
        self._fast_single.stop()
        self._fast_k = None
        self._fast_timer.start()
        self._tick_fast()

    def _end_k_drag(self):
        self._k_dragging = False
        self._fast_timer.stop()
        self._fast_single.stop()
        self._fast_k = self.k_slider.value()
        self._start_compute(fast=True)

    def _tick_fast(self):
        if not self._k_dragging:
            self._fast_timer.stop()
            return
        k = self.k_slider.value()
        if k == self._fast_k:
            return
        self._fast_k = k
        self._start_compute(fast=True)

    def _update_params_label(self):
        K = self.k_slider.value() / 100.0
        res = self.res_slider.value()
        iters = self.iter_slider.value()
        self.params_label.setText(f"K {K:.2f} · {res}² · {iters} 步")

    # -- compute ------------------------------------------------------
    def _start_compute(self, fast=False):
        self._debounce.stop()
        self._fast_single.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(500)

        self._generation += 1
        token = self._generation
        K = self.k_slider.value() / 100.0
        res = self.res_slider.value()
        iters = self.iter_slider.value()

        self._reset_probe()
        if fast:
            self.canvas.end_loading()
            self.progress.setVisible(False)
            self.state_label.setText("实时计算…")
        else:
            self.canvas.begin_loading()
            self.progress.setValue(0)
            self.progress.setVisible(True)
            self.state_label.setText("计算中…")
        self._update_params_label()
        self.compute_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.clear_orbit_btn.setEnabled(False)

        if taichi_available():
            self.stop_btn.setEnabled(False)
            started = time.perf_counter()
            nu, lyap = frequency_map_taichi(K, res, iters)
            if nu is None or token != self._generation:
                self._start_numpy_worker(K, res, iters, token, fast)
                return
            image = frequency_image(nu, lyap)
            self._on_done({
                "token": token,
                "K": K,
                "res": res,
                "iters": iters,
                "nu": nu,
                "lyap": lyap,
                "image": image,
                "elapsed": time.perf_counter() - started,
                "fast": fast,
            }, token)
        else:
            self._start_numpy_worker(K, res, iters, token, fast)

    def _start_numpy_worker(self, K, res, iters, token, fast):
        self.stop_btn.setEnabled(True)
        worker = FrequencyWorker(K, res, iters, token, self, fast=fast)
        worker.progress.connect(lambda value, g=token, f=fast: self._on_progress(value, g, f))
        worker.done.connect(lambda payload, g=token: self._on_done(payload, g))
        worker.failed.connect(lambda message, g=token: self._on_failed(message, g))
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_progress(self, value, token, fast=False):
        if token != self._generation or fast:
            return
        self.progress.setValue(int(value))
        self.canvas.set_progress(value)
        self.state_label.setText(f"计算中… {int(value)}%")

    def _on_done(self, payload, token):
        if token != self._generation:
            return
        self._worker = None
        image = ndarray_to_qimage(payload["image"])
        payload = dict(payload)
        payload["image"] = None
        self._current_payload = payload
        self.canvas.set_image(image, payload)
        self.progress.setValue(100)
        self.progress.setVisible(False)
        if payload.get("fast"):
            self.state_label.setText(
                f"实时 · {payload['elapsed']:.2f} s"
            )
        else:
            self.state_label.setText(
                f"完成 · {payload['elapsed']:.1f} s"
            )
        self.params_label.setText(
            f"K {payload['K']:.2f} · {payload['res']}² · {payload['iters']} 步"
        )
        self.compute_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)

    def _on_failed(self, message, token):
        if token != self._generation:
            return
        self._worker = None
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.canvas.set_error(message)
        self.state_label.setText("计算失败")
        self.compute_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "KAM 计算失败", str(message))

    def _stop_compute(self):
        self._fast_timer.stop()
        self._fast_single.stop()
        if self._worker is None:
            return
        self._generation += 1
        self._worker.stop()
        self._worker.wait(400)
        self._worker = None
        self.canvas.end_loading()
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.state_label.setText("已停止")
        self.compute_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(self._current_payload is not None)

    # -- probe / cursor ----------------------------------------------
    def _on_cursor_moved(self, q, p):
        sample = self.canvas.sample_frequency(q, p)
        if sample is None:
            self.cursor_label.setText(f"q {q:.3f}   p {p:.3f}")
            return
        nu, lyap = sample
        tag = "混沌" if lyap >= CHAOS_LYAP else "规则"
        self.cursor_label.setText(
            f"q {q:.3f}   p {p:.3f}   ν {nu:.3f} · λ {lyap:.3f} · {tag}"
        )

    def _on_cursor_left(self):
        self.cursor_label.setText("")

    def _trace_orbit(self, q, p):
        payload = self._current_payload
        if payload is None:
            self.state_label.setText("等待频率图计算完成")
            return
        orbit, nu, lyap = trace_orbit(q, p, payload["K"], steps=1800)
        self.canvas.set_orbit(orbit, nu, lyap)
        tag = "混沌" if lyap >= CHAOS_LYAP else "规则"
        self.orbit_info.setText(f"初值  q {q:.3f} · p {p:.3f}")
        self.clear_orbit_btn.setEnabled(True)
        self.state_label.setText(f"已追踪 · ν {nu:.4f} · λ {lyap:.4f} · {tag}")

    def _clear_orbit(self):
        self.canvas.clear_orbit()
        self.clear_orbit_btn.setEnabled(False)
        self.orbit_info.setText("尚未选择初值")
        self.state_label.setText("就绪")

    # -- save / close -------------------------------------------------
    def _save_image(self):
        path = ask_save_path(self, "保存 KAM 频率图", suffix=".png")
        if not path:
            return
        if self.canvas.grab().save(path):
            self.state_label.setText("已保存图像")
        else:
            QMessageBox.warning(self, "保存失败", "无法写入所选路径。")

    def closeEvent(self, event):
        self._debounce.stop()
        self._fast_timer.stop()
        self._fast_single.stop()
        if hasattr(self, "planet_canvas"):
            self.planet_canvas.stop()
        if self._worker is not None:
            self._worker.stop_and_wait(2000)
            self._worker = None
        super().closeEvent(event)


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
