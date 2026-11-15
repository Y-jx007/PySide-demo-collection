# -*- coding: utf-8 -*-
"""Nonlinear optics in one window: three regimes of the same idea.

    space  i psi_t = -1/2 lap(psi) - |psi|^2 psi     (self-focusing, filaments)
    time   i A_z   = (beta2/2) A_tt - |A|^2 A        (SPM, solitons)
    double dA1/dz  = i kappa A2 A1* exp(-i dk z)     (chi-2, phase matching)

The first two are cubic nonlinearities (chi-3, present in every medium); the
third is quadratic and only exists where the crystal lacks inversion symmetry.
The mode selector swaps the model, the panel and the canvas; the solvers live in
support/physics/nls.py, spm.py and shg.py.
"""
import time

import numpy as np

from PySide6.QtGui import QImage

from common.lab_window import LabWindow
from common.qt_base import (
    LabeledSlider, QCheckBox, QColor, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPainter, QPen, QPointF, QPushButton,
    QRectF, QSizePolicy, QStackedWidget, QTabWidget, QTimer, QVBoxLayout, QWidget,
    Qt, run_app,
)
from common.field_canvas import ArrayCanvas
from support.physics.nls import MODELS, NLSE, PALETTE, PRESETS, palette_lut
from support.physics.mixing import SFGMap
from support.physics.shg import SHGMap, beam_lut
from support.physics.spm import TemporalNLSE

MODES = ("空间自聚焦", "时间孤子", "倍频")

SPACE_HTML = (
    '<div style="font-size:13px; color:#1b1b1b;">i&thinsp;∂ψ/∂t = −½∇²ψ − |ψ|²ψ</div>'
    '<div style="font-size:10px; color:#4a4a4a; margin-top:4px;">'
    "Kerr：强光抬高折射率 n = n₀ + n₂I，亮处势更低 → 光束自透镜。</div>"
    '<div style="font-size:10px; color:#4a4a4a; margin-top:2px;">'
    "衍射想摊开、自聚焦想收紧：低于阈值散开，超过则坍缩。</div>"
    '<div style="font-size:10px; color:#5c5c5c; margin-top:2px;">'
    "饱和 |ψ|²/(1+|ψ|²/I<sub>s</sub>)：强光时响应封顶，坍缩停在有限尺度。</div>"
)

TIME_HTML = (
    '<div style="font-size:13px; color:#1b1b1b;">i&thinsp;∂A/∂z + ½β₂&thinsp;∂²A/∂t² + |A|²A = 0</div>'
    '<div style="font-size:10px; color:#4a4a4a; margin-top:4px;">'
    "色散让脉冲展宽、非线性让它自相位调制；两者平衡就是孤子。</div>"
    '<div style="font-size:10px; color:#4a4a4a; margin-top:2px;">'
    "阶数 N = √(γP₀T₀²/|β₂|)：1 不变、2 呼吸、≥3 分裂。</div>"
)

DOUBLE_HTML = (
    '<div style="font-size:12px; color:#1b1b1b;">dA₁/dz = iκ·A₂A₁*&thinsp;e<sup>−iΔkz</sup></div>'
    '<div style="font-size:12px; color:#1b1b1b;">dA₂/dz = i(κ/2)·A₁²&thinsp;e<sup>+iΔkz</sup></div>'
    '<div style="font-size:10px; color:#4a4a4a; margin-top:4px;">'
    "κ ∝ χ⁽²⁾ 只在非中心对称介质里；小图就是 Δk 被补平的过程。</div>"
)

TIPS = {
    "空间自聚焦": (
        "左键可在该处再种一束光；动量谱是场的二维 FFT（dB），"
        "调制不稳定性选的波矢与坍缩辐射的环都会显示出来。"
    ),
    "时间孤子": (
        "横轴是延迟时间 t = t_lab − z/v_g，竖轴是传播距离 z；"
        "下方小图是当前频谱，自相位调制让它对称展宽——超连续谱的起点。"
    ),
    "倍频": (
        "横轴是晶体位置 z，竖轴是 Δk，颜色和另外两个模式同一套墨青色阶："
        "越深 = 倍频功率越强；扇形条纹就是 Maker 条纹。"
        "左侧色带是同一套色阶的图例（浅=800 nm 基频，深=400 nm 倍频），"
        "游标标出当前转换率。"
    ),
}


class ResponsePlot(QWidget):
    """空间模式：响应因子与两能级布居随光强的变化。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(96)
        self.sim = None
        self.peak = 0.0
        self.setToolTip(
            "横轴光强，纵轴响应因子与能级布居：ΔN/N = 1/(1+I/Iₛ) 是还能响应的比例，"
            "N₁、N₂ 在强光下趋于均分（漂白），竖线是当前峰值。"
        )

    def update_state(self, sim, peak):
        self.sim = sim
        self.peak = float(peak)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.fillRect(rect, QColor(255, 255, 255))
        painter.setPen(QPen(QColor(214, 214, 214), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        sim = self.sim
        if sim is None:
            painter.end()
            return
        left, top, right, bottom = 24.0, 14.0, 8.0, 18.0
        plot = QRectF(left, top, max(10.0, rect.width() - left - right),
                      max(10.0, rect.height() - top - bottom))
        saturable = sim.model == "饱和"
        x_max = 2.0 * sim.saturation if saturable else max(1.0 / max(sim.kappa, 1e-6), 1.0)
        x = np.linspace(0.0, x_max, 140)

        def sx(value):
            return plot.left() + plot.width() * min(max(value / x_max, 0.0), 1.0)

        def sy(value):
            return plot.bottom() - plot.height() * min(max(value, 0.0), 1.0)

        painter.setPen(QPen(QColor(238, 238, 238), 1))
        for fraction in (0.25, 0.5, 0.75):
            y = sy(fraction)
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        painter.setPen(QPen(QColor(190, 190, 190), 1))
        painter.drawLine(QPointF(plot.left(), plot.bottom()),
                         QPointF(plot.right(), plot.bottom()))
        painter.drawLine(QPointF(plot.left(), plot.top()),
                         QPointF(plot.left(), plot.bottom()))
        curve = sim.response(x)
        painter.setPen(QPen(QColor(24, 96, 150), 1.8))
        painter.drawPolyline([QPointF(sx(xi), sy(yi)) for xi, yi in zip(x, curve)])
        if saturable:
            s = x / sim.saturation
            painter.setPen(QPen(QColor(150, 150, 150), 1.0, Qt.DashLine))
            for share in ((s + 2.0) / (2.0 * (1.0 + s)), s / (2.0 * (1.0 + s))):
                painter.drawPolyline(
                    [QPointF(sx(xi), sy(yi)) for xi, yi in zip(x, share)]
                )
        painter.setPen(QPen(QColor(200, 120, 40), 1.2, Qt.DashLine))
        cursor = sx(self.peak)
        painter.drawLine(QPointF(cursor, plot.top()), QPointF(cursor, plot.bottom()))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(200, 120, 40))
        painter.drawEllipse(QPointF(cursor, sy(float(sim.response(self.peak)))), 3.0, 3.0)
        font = painter.font()
        font.setPointSizeF(7.5)
        painter.setFont(font)
        painter.setPen(QColor(110, 110, 110))
        # 和振子 P(E) 一致的轴标注：纵轴名在轴上方，横轴名在轴右端
        ratio = (1.0 / (1.0 + self.peak / sim.saturation) if saturable
                 else max(1.0 - sim.kappa * self.peak, 0.0))
        painter.drawText(QRectF(plot.left() + 2, plot.top() - 14, 90, 12),
                         Qt.AlignLeft, "响应 ↑ %.2f" % ratio)
        painter.drawText(QRectF(plot.right() - 40, plot.bottom() + 2, 38, 12),
                         Qt.AlignRight, "I →")
        painter.end()


class SpectrumPlot(QWidget):
    """时间模式：当前脉冲的频谱（dB）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(88)
        self.sim = None
        self.setToolTip("脉冲的频谱（对数）：自相位调制越强，谱越宽。")

    def update_state(self, sim):
        self.sim = sim
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.fillRect(rect, QColor(255, 255, 255))
        painter.setPen(QPen(QColor(214, 214, 214), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        sim = self.sim
        if sim is None:
            painter.end()
            return
        left, top, right, bottom = 6.0, 6.0, 6.0, 16.0
        plot = QRectF(left, top, max(10.0, rect.width() - left - right),
                      max(10.0, rect.height() - top - bottom))
        curve = sim.spectrum_db()
        x = np.linspace(0.0, 1.0, curve.size)
        painter.setPen(QPen(QColor(24, 96, 150), 1.5))
        painter.drawPolyline(
            [QPointF(plot.left() + plot.width() * xi,
                     plot.bottom() - plot.height() * yi)
             for xi, yi in zip(x, curve)]
        )
        font = painter.font()
        font.setPointSizeF(7.5)
        painter.setFont(font)
        painter.setPen(QColor(110, 110, 110))
        painter.drawText(QRectF(plot.left(), rect.bottom() - 14, 90, 12),
                         Qt.AlignLeft, "−ω ← 0 → +ω")
        painter.drawText(QRectF(rect.right() - 110, rect.bottom() - 14, 104, 12),
                         Qt.AlignRight, "Δω = %.2f" % sim.spectral_width())
        painter.end()


class FringePlot(QWidget):
    """倍频模式：出射面效率 vs Δk。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(58)
        self.model = None
        self.setToolTip(
            "出射面的倍频效率随相位失配的变化（扫描看到的 Maker 曲线）。\n"
            "按能量算：相位匹配且晶体够长时泵浦被耗尽，效率可达 100%。"
        )

    def update_model(self, model):
        self.model = model
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.fillRect(rect, QColor(255, 255, 255))
        painter.setPen(QPen(QColor(214, 214, 214), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        model = self.model
        if model is None:
            painter.end()
            return
        left, top, right, bottom = 6.0, 6.0, 6.0, 16.0
        plot = QRectF(left, top, max(10.0, rect.width() - left - right),
                      max(10.0, rect.height() - top - bottom))
        curve = model.efficiency()
        peak = float(curve.max()) or 1.0
        x = np.linspace(0.0, 1.0, curve.size)
        painter.setPen(QPen(QColor(24, 96, 150), 1.4))
        painter.drawPolyline(
            [QPointF(plot.left() + plot.width() * xi,
                     plot.bottom() - plot.height() * (yi / peak))
             for xi, yi in zip(x, curve)]
        )
        font = painter.font()
        font.setPointSizeF(7.5)
        painter.setFont(font)
        painter.setPen(QColor(110, 110, 110))
        painter.drawText(QRectF(plot.left(), rect.bottom() - 14, 120, 12),
                         Qt.AlignLeft, "−Δk ← 0 → +Δk")
        painter.drawText(QRectF(rect.right() - 120, rect.bottom() - 14, 114, 12),
                         Qt.AlignRight, "峰值 %.1f%%" % (model.peak_efficiency() * 100.0))
        painter.end()


class BeamStrip(QWidget):
    """倍频模式：完整光谱色标 + 当前转换率游标。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self.model = None
        self.setToolTip(
            "与图像同一套墨青色阶的图例：浅端是 800 nm 基频，深端是 400 nm 倍频，"
            "游标标出当前转换率。"
        )

    def update_model(self, model):
        self.model = model
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        rect = QRectF(self.rect())
        painter.fillRect(rect, QColor(255, 255, 255))
        model = self.model
        if model is None:
            painter.end()
            return
        table = np.ascontiguousarray(beam_lut()[None, :, :])
        width = table.shape[1]
        image = QImage(table.data, width, 1, 3 * width, QImage.Format_RGB888).copy()
        band = QRectF(4.0, 3.0, rect.width() - 8.0, 12.0)
        painter.drawImage(band, image)
        painter.setPen(QPen(QColor(214, 214, 214), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(band)
        converted = float(model.conversions()[-1])
        marker = band.left() + band.width() * min(max(converted, 0.0), 1.0)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 2.6))
        painter.drawLine(QPointF(marker, band.top() - 1.0),
                         QPointF(marker, band.bottom() + 1.0))
        painter.setPen(QPen(QColor(40, 40, 40, 200), 1.2))
        painter.drawLine(QPointF(marker, band.top() - 1.0),
                         QPointF(marker, band.bottom() + 1.0))
        font = painter.font()
        font.setPointSizeF(7.5)
        painter.setFont(font)
        painter.setPen(QColor(110, 110, 110))
        painter.drawText(QRectF(4, 16, 80, 14), Qt.AlignLeft, "800 nm")
        painter.drawText(QRectF(rect.width() - 84, 16, 80, 14),
                         Qt.AlignRight, "400 nm")
        painter.drawText(QRectF(4, 16, rect.width() - 8, 14), Qt.AlignCenter,
                         "已转换 %.0f%%" % (converted * 100.0))
        painter.end()


class OscillatorPlot(QWidget):
    """P–E 曲线：线性振子 vs 非简谐振子，游标在当前峰值场强上。"""

    BETA = 0.02          # 示意用的非谐系数（不是物理单位）
    E_MAX = 8.0          # 与场上峰值量级对齐，游标才不会一直贴边

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(82)
        self.field = 0.0
        self.setToolTip(
            "横轴场强 E、纵轴极化 P：直线是简谐（线性）响应，"
            "弯上去这条是非简谐——高次项就是从这里来的。\n"
            "弯曲越早出现，同样场强下非线性越明显；竖线是当前的峰值场强。"
        )

    def update_state(self, field):
        self.field = float(field)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.fillRect(rect, QColor(255, 255, 255))
        painter.setPen(QPen(QColor(214, 214, 214), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        left, top, right, bottom = 20.0, 16.0, 8.0, 22.0
        plot = QRectF(left, top, max(10.0, rect.width() - left - right),
                      max(10.0, rect.height() - top - bottom))
        e = np.linspace(0.0, self.E_MAX, 120)
        cubic = e + self.BETA * e ** 3
        y_max = float(cubic[-1])

        def sx(value):
            return plot.left() + plot.width() * min(max(value / self.E_MAX, 0.0), 1.0)

        def sy(value):
            return plot.bottom() - plot.height() * min(max(value / y_max, 0.0), 1.0)

        painter.setPen(QPen(QColor(190, 190, 190), 1))
        painter.drawLine(QPointF(plot.left(), plot.bottom()),
                         QPointF(plot.right(), plot.bottom()))
        painter.drawLine(QPointF(plot.left(), plot.top()),
                         QPointF(plot.left(), plot.bottom()))
        painter.setPen(QPen(QColor(160, 160, 160), 1.0, Qt.DashLine))
        painter.drawPolyline([QPointF(sx(xi), sy(xi / y_max * y_max)) for xi in e])
        painter.setPen(QPen(QColor(24, 96, 150), 1.8))
        painter.drawPolyline([QPointF(sx(xi), sy(yi)) for xi, yi in zip(e, cubic)])
        cursor = sx(self.field)
        painter.setPen(QPen(QColor(200, 120, 40), 1.2, Qt.DashLine))
        painter.drawLine(QPointF(cursor, plot.top()), QPointF(cursor, plot.bottom()))
        peak = self.field + self.BETA * self.field ** 3
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(200, 120, 40))
        painter.drawEllipse(QPointF(cursor, sy(peak)), 3.0, 3.0)
        font = painter.font()
        font.setPointSizeF(7.5)
        painter.setFont(font)
        painter.setPen(QColor(110, 110, 110))
        painter.drawText(QRectF(plot.left() + 2, plot.top() - 14, 40, 12),
                         Qt.AlignLeft, "P ↑")
        painter.drawText(QRectF(plot.right() - 40, plot.bottom() + 2, 38, 12),
                         Qt.AlignRight, "E →")
        painter.drawText(QRectF(plot.left() + 2, plot.top() + 1, 120, 12),
                         Qt.AlignLeft, "实线 非简谐 / 虚线 线性")
        painter.end()


class MomentumSketch(QWidget):
    """2k₁ 与 k₂ 的动量示意；开周期极化时补上 K = 2π/Λ。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.show_qpm = False
        self.period = 0.0
        self.setToolTip(
            "相位匹配就是动量守恒：倍频波的波矢 k₂ 要等于两个基频光子之和 2k₁。"
            "\n差出来的那一段就是 Δk；周期极化用 K = 2π/Λ 把它补平。"
        )

    def update_state(self, show_qpm, period):
        self.show_qpm = bool(show_qpm)
        self.period = float(period)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        painter.fillRect(rect, QColor(255, 255, 255))
        x0 = 26.0
        y1 = 9.0
        y2 = 24.0
        base = 120.0
        gap = 34.0
        ink = QColor(24, 96, 150)
        painter.setPen(QPen(ink, 1.6))
        painter.drawLine(QPointF(x0, y1), QPointF(x0 + base, y1))
        painter.drawLine(QPointF(x0 + base, y1 - 4), QPointF(x0 + base, y1 + 4))
        painter.drawLine(QPointF(x0, y2), QPointF(x0 + base + gap, y2))
        painter.drawLine(QPointF(x0 + base + gap, y2 - 4),
                         QPointF(x0 + base + gap, y2 + 4))
        font = painter.font()
        font.setPointSizeF(7.5)
        painter.setFont(font)
        painter.setPen(QColor(110, 110, 110))
        painter.drawText(QRectF(2, y1 - 6, 24, 12), Qt.AlignLeft, "2k₁")
        painter.drawText(QRectF(2, y2 - 6, 24, 12), Qt.AlignLeft, "k₂")
        if self.show_qpm:
            painter.setPen(QPen(QColor(200, 120, 40), 1.6, Qt.DashLine))
            painter.drawLine(QPointF(x0 + base, y2), QPointF(x0 + base + gap, y2))
            painter.setPen(QColor(200, 120, 40))
            painter.drawText(QRectF(x0 + base - 6, y2 + 2, 90, 12), Qt.AlignLeft,
                             "K = 2π/Λ")
        else:
            painter.setPen(QPen(QColor(200, 120, 40), 1.2, Qt.DashLine))
            painter.drawLine(QPointF(x0 + base, y1), QPointF(x0 + base + gap, y1))
            painter.setPen(QColor(200, 120, 40))
            painter.drawText(QRectF(x0 + base - 4, y1 - 14, 90, 12), Qt.AlignLeft,
                             "Δk")
        painter.end()


class OpticsCanvas(ArrayCanvas):
    """一个画布，三种模式：场 / 水落图 / 相位匹配图。"""

    def __init__(self, window, parent=None):
        super().__init__(aspect=1.0, minimum=(480, 480), parent=parent)
        self.setFixedSize(640, 640)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)
        self.window = window
        self.mode = MODES[0]
        self.space = NLSE()
        self.time = TemporalNLSE()
        self.doubling = SHGMap()
        self.sfg = SFGMap()
        self.process = "倍频"
        self.log_view = False
        self.space_view = "场"

    def render_frame(self):
        if self.mode == "时间孤子":
            self.set_array(self.time.render_waterfall())
        elif self.mode == "倍频":
            model = self.doubling if self.process == "倍频" else self.sfg
            self.set_array(np.ascontiguousarray(np.flipud(model.render())))
        elif self.space_view == "动量谱":
            self.set_array(self.space.render_spectrum(floor_db=35.0))
        else:
            self.set_array(self.space.render(PALETTE, log=self.log_view))

    def world_point(self, pos):
        mapped = self.map_pos(pos)
        if mapped is None:
            return None
        u, v = mapped
        half = self.space.extent / 2.0
        return (u - 0.5) * 2.0 * half, (v - 0.5) * 2.0 * half

    def mousePressEvent(self, event):
        if self.mode == "空间自聚焦" and event.button() == Qt.LeftButton:
            point = self.world_point(event.position())
            if point is not None:
                self.space.add_seed(point[0], point[1], self.window.amplitude())
                self.render_frame()
        super().mousePressEvent(event)


class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("非线性光学 · 自聚焦 / 孤子 / 倍频")
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.canvas = OpticsCanvas(self)
        self.mode = MODES[0]
        self.paused = False
        self._frames = 0
        self._fps_clock = time.perf_counter()
        self.init_ui()
        self.setMinimumSize(960, 680)
        self.resize(960, 680)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)
        self.reset()
        self.timer.start()
        self.setFocusPolicy(Qt.StrongFocus)

    # ---------- 界面 ----------
    def init_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(330)
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        mode_box = QGroupBox("模式")
        mode_layout = QVBoxLayout(mode_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODES)
        self.mode_combo.currentTextChanged.connect(self.on_mode)
        mode_layout.addWidget(self.mode_combo)
        column.addWidget(mode_box)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.build_space_page())
        self.stack.addWidget(self.build_time_page())
        self.stack.addWidget(self.build_doubling_page())
        column.addWidget(self.stack, 1)

        buttons = QHBoxLayout()
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self.toggle_pause)
        reset_button = QPushButton("重置")
        reset_button.clicked.connect(self.reset)
        save_button = QPushButton("保存")
        save_button.clicked.connect(self.save)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(reset_button)
        buttons.addWidget(save_button)
        column.addLayout(buttons)

        root.addWidget(panel)
        root.addWidget(self.canvas, 1)
        self.setCentralWidget(central)

        self.readout_label = QLabel("")
        self.fps_label = QLabel("")
        self.fps_label.setStyleSheet("padding-left: 14px;")
        self.statusBar().addPermanentWidget(self.readout_label)
        self.statusBar().addPermanentWidget(self.fps_label)
        self.statusBar().showMessage(TIPS[self.mode])

    def build_space_page(self):
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        equation_box = QGroupBox("方程")
        equation_layout = QVBoxLayout(equation_box)
        self.space_label = QLabel(SPACE_HTML)
        self.space_label.setTextFormat(Qt.RichText)
        self.space_label.setWordWrap(True)
        self.space_label.setToolTip(TIPS["空间自聚焦"])
        equation_layout.addWidget(self.space_label)
        column.addWidget(equation_box)

        init_box = QGroupBox("初始条件")
        init_layout = QVBoxLayout(init_box)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS)
        self.preset_combo.setCurrentText("涡旋")
        self.preset_combo.currentTextChanged.connect(lambda _: self.reset_space())
        init_layout.addWidget(self.preset_combo)
        self.amp_slider = LabeledSlider(
            "振幅", 20, 120, 85, formatter=lambda v: "%.2f" % (v / 100.0)
        )
        self.amp_slider.valueChanged.connect(lambda _: self.reset_space())
        init_layout.addWidget(self.amp_slider)
        self.noise_slider = LabeledSlider(
            "噪声", 0, 100, 40, formatter=lambda v: "%.3f" % (v / 1000.0)
        )
        self.noise_slider.valueChanged.connect(lambda _: self.reset_space())
        init_layout.addWidget(self.noise_slider)
        column.addWidget(init_box)

        run_box = QGroupBox("演化")
        run_layout = QVBoxLayout(run_box)
        self.steps_slider = LabeledSlider("每帧步数", 1, 24, 3)
        run_layout.addWidget(self.steps_slider)
        self.log_check = QCheckBox("对数强度（−30 dB）")
        self.log_check.setChecked(False)
        self.log_check.setToolTip("线性档只看主瓣；对数档把 30 dB 内的辐射环压进色阶。")
        self.log_check.toggled.connect(self.on_log)
        run_layout.addWidget(self.log_check)
        selects = QHBoxLayout()
        selects.setSpacing(6)
        self.view_combo = QComboBox()
        self.view_combo.addItems(("场", "动量谱"))
        self.view_combo.currentTextChanged.connect(self.on_view)
        self.view_combo.setToolTip("动量谱是场的二维 FFT（dB），显示被选中的波矢。")
        selects.addWidget(self.view_combo, 1)
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODELS)
        self.model_combo.currentTextChanged.connect(self.on_model)
        self.model_combo.setToolTip(
            "饱和：1/(1+I/Iₛ)；高阶 Kerr：n₂I + n₄I²，因子 1−κI；"
            "双光子吸收：折射之外还会吸收，dI/dt = −βI²，越亮耗得越快。"
        )
        selects.addWidget(self.model_combo, 1)
        run_layout.addLayout(selects)
        self.model_kappa_slider = LabeledSlider(
            "κ", 1, 80, 20, formatter=lambda v: "%.3f" % (v / 1000.0)
        )
        self.model_kappa_slider.valueChanged.connect(self.on_model_kappa)
        self.model_kappa_slider.setVisible(False)
        run_layout.addWidget(self.model_kappa_slider)
        column.addWidget(run_box)

        micro_box = QGroupBox("微观响应")
        micro_layout = QVBoxLayout(micro_box)
        self.response_tabs = QTabWidget()
        self.response_plot = ResponsePlot()
        self.oscillator_plot = OscillatorPlot()
        self.response_tabs.addTab(self.response_plot, "两能级布居")
        self.response_tabs.addTab(self.oscillator_plot, "振子 P(E)")
        micro_layout.addWidget(self.response_tabs)
        column.addWidget(micro_box)
        column.addStretch(1)
        return page

    def build_time_page(self):
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        equation_box = QGroupBox("方程")
        equation_layout = QVBoxLayout(equation_box)
        self.time_label = QLabel(TIME_HTML)
        self.time_label.setTextFormat(Qt.RichText)
        self.time_label.setWordWrap(True)
        self.time_label.setToolTip(TIPS["时间孤子"])
        equation_layout.addWidget(self.time_label)
        column.addWidget(equation_box)

        pulse_box = QGroupBox("输入脉冲")
        pulse_layout = QVBoxLayout(pulse_box)
        self.order_slider = LabeledSlider(
            "阶数 N", 40, 340, 200, formatter=lambda v: "%.1f" % (v / 100.0)
        )
        self.order_slider.valueChanged.connect(lambda _: self.reset_time())
        pulse_layout.addWidget(self.order_slider)
        self.width_slider = LabeledSlider(
            "脉宽 T₀", 60, 200, 100, formatter=lambda v: "%.2f" % (v / 100.0)
        )
        self.width_slider.valueChanged.connect(lambda _: self.reset_time())
        pulse_layout.addWidget(self.width_slider)
        column.addWidget(pulse_box)

        run_box = QGroupBox("演化")
        run_layout = QVBoxLayout(run_box)
        self.time_steps_slider = LabeledSlider("每帧步数", 1, 8, 2)
        run_layout.addWidget(self.time_steps_slider)
        column.addWidget(run_box)

        spectrum_box = QGroupBox("当前频谱")
        spectrum_layout = QVBoxLayout(spectrum_box)
        self.spectrum_plot = SpectrumPlot()
        spectrum_layout.addWidget(self.spectrum_plot)
        column.addWidget(spectrum_box)
        column.addStretch(1)
        return page

    def build_doubling_page(self):
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        equation_box = QGroupBox("方程")
        equation_layout = QVBoxLayout(equation_box)
        self.double_label = QLabel(DOUBLE_HTML)
        self.double_label.setTextFormat(Qt.RichText)
        self.double_label.setWordWrap(True)
        self.double_label.setToolTip(TIPS["倍频"])
        equation_layout.addWidget(self.double_label)
        self.momentum_sketch = MomentumSketch()
        equation_layout.addWidget(self.momentum_sketch)
        column.addWidget(equation_box)

        crystal_box = QGroupBox("晶体")
        crystal_layout = QVBoxLayout(crystal_box)
        self.process_combo = QComboBox()
        self.process_combo.addItems(("倍频", "和频", "差频"))
        self.process_combo.currentTextChanged.connect(self.on_process)
        self.process_combo.setToolTip(
            "倍频：ω+ω→2ω；和频：ω₁+ω₂→ω₃（两个输入都消耗，就是 OPA 的正向过程）。"
        )
        crystal_layout.addWidget(self.process_combo)
        self.length_slider = LabeledSlider(
            "长度 L", 50, 300, 200, formatter=lambda v: "%.2f" % (v / 100.0)
        )
        self.length_slider.valueChanged.connect(self.refresh_doubling)
        crystal_layout.addWidget(self.length_slider)
        self.crystal_kappa_slider = LabeledSlider(
            "非线性 κ", 20, 200, 100, formatter=lambda v: "%.2f" % (v / 100.0)
        )
        self.crystal_kappa_slider.valueChanged.connect(self.refresh_doubling)
        crystal_layout.addWidget(self.crystal_kappa_slider)
        self.symmetric_check = QCheckBox("中心对称介质（χ⁽²⁾ = 0）")
        self.symmetric_check.toggled.connect(self.refresh_doubling)
        self.symmetric_check.setToolTip("玻璃、水、气体都是中心对称的，不会倍频。")
        crystal_layout.addWidget(self.symmetric_check)
        column.addWidget(crystal_box)

        qpm_box = QGroupBox("周期极化（QPM）")
        qpm_layout = QVBoxLayout(qpm_box)
        self.qpm_check = QCheckBox("周期反转畴")
        self.qpm_check.toggled.connect(self.on_qpm)
        qpm_layout.addWidget(self.qpm_check)
        self.poling_slider = LabeledSlider(
            "周期 Λ", 30, 200, 63, formatter=lambda v: "%.2f" % (v / 10.0)
        )
        self.poling_slider.setEnabled(False)
        self.poling_slider.valueChanged.connect(self.on_poling)
        qpm_layout.addWidget(self.poling_slider)
        column.addWidget(qpm_box)

        self.beam_box = QGroupBox("光束颜色")
        beam_box = self.beam_box
        beam_layout = QVBoxLayout(beam_box)
        self.beam_strip = BeamStrip()
        beam_layout.addWidget(self.beam_strip)
        column.addWidget(beam_box)

        fringe_box = QGroupBox("出射面效率")
        fringe_layout = QVBoxLayout(fringe_box)
        self.fringe_plot = FringePlot()
        fringe_layout.addWidget(self.fringe_plot)
        column.addWidget(fringe_box)
        column.addStretch(1)
        return page

    # ---------- 参数 ----------
    def amplitude(self):
        return self.amp_slider.value() / 100.0

    def noise(self):
        return self.noise_slider.value() / 1000.0

    def preset(self):
        return self.preset_combo.currentText()

    def order(self):
        return self.order_slider.value() / 100.0

    def pulse_width(self):
        return self.width_slider.value() / 100.0

    # ---------- 回调 ----------
    def on_mode(self, name):
        if name not in MODES:
            return
        self.mode = name
        self.stack.setCurrentIndex(MODES.index(name))
        self.canvas.mode = name
        self.paused = False
        self.pause_button.setText("暂停")
        self.pause_button.setEnabled(name != "倍频")
        self.statusBar().showMessage(TIPS[name])
        if name == "倍频":
            self.refresh_doubling()
        else:
            self.canvas.render_frame()
            self.update_readout()

    def on_log(self, checked):
        self.canvas.log_view = bool(checked)
        if self.mode == "空间自聚焦":
            self.canvas.render_frame()

    def on_view(self, name):
        self.canvas.space_view = name
        if self.mode == "空间自聚焦":
            self.canvas.render_frame()

    def on_model(self, name):
        sim = self.canvas.space
        sim.model = name
        self.model_kappa_slider.setVisible(name == "高阶 Kerr")
        sim.kappa = self.model_kappa_slider.value() / 1000.0
        # 双光子吸收：吸收型的 χ⁽³⁾，强度越高耗得越快（系数固定在模型里）
        sim.two_photon = 0.05 if name == "双光子吸收" else 0.0
        self.reset_space()

    def on_model_kappa(self, value):
        self.canvas.space.kappa = value / 1000.0
        self.reset_space()

    def on_process(self, name):
        self.canvas.process = name
        # 色带解释的是 800→400 nm 的倍频过程，和频时隐藏免得误导
        self.beam_box.setVisible(name == "倍频")
        self.qpm_check.setEnabled(name == "倍频")
        self.canvas.process = name
        self.qpm_check.setToolTip("" if name == "倍频" else "和频时用两个输入直接匹配，不涉及周期极化")
        self.refresh_doubling()

    def on_qpm(self, checked):
        self.poling_slider.setEnabled(bool(checked))
        self.refresh_doubling()

    def on_poling(self, _value):
        if self.qpm_check.isChecked():
            self.refresh_doubling()

    # ---------- 重置 / 刷新 ----------
    def reset_space(self):
        sim = self.canvas.space
        sim.reset(self.preset(), self.amplitude(), self.noise())
        if self.mode == "空间自聚焦":
            self.canvas.render_frame()
            self.update_readout()

    def reset_time(self):
        self.canvas.time.reset(self.order(), self.pulse_width())
        if self.mode == "时间孤子":
            self.canvas.render_frame()
            self.update_readout()

    def poling(self):
        return self.poling_slider.value() / 10.0 if self.qpm_check.isChecked() else None

    def refresh_doubling(self):
        model = self.canvas.doubling
        model.configure(
            length=self.length_slider.value() / 100.0,
            kappa=self.crystal_kappa_slider.value() / 100.0,
            symmetric=self.symmetric_check.isChecked(),
            poling=self.poling(),
        )
        sfg = self.canvas.sfg
        sfg.branch = self.process_combo.currentText()
        sfg.configure(length=self.length_slider.value() / 100.0,
                      kappa=0.0 if self.symmetric_check.isChecked()
                      else self.crystal_kappa_slider.value() / 100.0)
        if self.mode == "倍频":
            self.canvas.render_frame()
            active = model if self.canvas.process == "倍频" else self.canvas.sfg
            self.fringe_plot.update_model(active)
            self.beam_strip.update_model(model)
            self.momentum_sketch.update_state(
                self.qpm_check.isChecked(), self.poling() or 0.0
            )
            self.update_readout()

    def reset(self):
        self.paused = False
        self.pause_button.setText("暂停")
        if self.mode == "空间自聚焦":
            self.reset_space()
        elif self.mode == "时间孤子":
            self.reset_time()
        else:
            self.refresh_doubling()

    # ---------- 循环 ----------
    def toggle_pause(self):
        if self.mode == "倍频":
            return
        self.paused = not self.paused
        self.pause_button.setText("继续" if self.paused else "暂停")

    def tick(self):
        if self.paused or self.mode == "倍频":
            return
        if self.mode == "空间自聚焦":
            self.canvas.space.step(self.steps_slider.value())
        else:
            self.canvas.time.step(self.time_steps_slider.value())
        self.canvas.render_frame()
        self._frames += 1
        now = time.perf_counter()
        elapsed = now - self._fps_clock
        if elapsed >= 1.0:
            self.fps_label.setText("%.0f fps" % (self._frames / elapsed))
            self._frames = 0
            self._fps_clock = now
        self.update_readout()

    def update_readout(self):
        if self.mode == "空间自聚焦":
            sim = self.canvas.space
            peak = sim.peak()
            self.readout_label.setText(
                "功率 %.2f · 峰值 %.2f · 束宽 %.2f · 步数 %d"
                % (sim.power(), peak, sim.width(), sim.steps)
            )
            self.response_plot.update_state(sim, peak)
            self.oscillator_plot.update_state(np.sqrt(max(peak, 0.0)))
        elif self.mode == "时间孤子":
            sim = self.canvas.time
            self.readout_label.setText(
                "能量 %.2f · 峰值 %.2f · 时宽 %.2f · 谱宽 %.2f · z %.2f"
                % (sim.energy(), sim.peak(), sim.width(), sim.spectral_width(),
                   sim.steps * sim.dz)
            )
            self.spectrum_plot.update_state(sim)
        else:
            if self.canvas.process != "倍频":
                sfg = self.canvas.sfg
                symbol = "n₃" if self.canvas.process == "和频" else "n₂"
                self.readout_label.setText(
                    "%s · L %.2f · κ %.2f · 峰值 %s %.2f · Δk ±%.0f"
                    % (self.canvas.process, sfg.length, sfg.kappa, symbol,
                       sfg.peak(), sfg.span)
                )
            else:
                model = self.canvas.doubling
                self.readout_label.setText(
                    "L %.2f · κ %.2f · 峰值效率 %.1f%% · Δk ±%.0f"
                    % (model.length, model.kappa, model.peak_efficiency() * 100.0,
                       model.span)
                )

    # ---------- 保存 ----------
    def save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存当前画面", "Nonlinear.png", "PNG (*.png)"
        )
        if not path:
            return
        if not self.canvas.grab().save(path):
            QMessageBox.warning(self, "保存失败", "无法写入 " + path)


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
