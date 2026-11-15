# -*- coding: utf-8 -*-
"""SPM: self-phase modulation and solitons in the temporal nonlinear Schrodinger equation.

    i dA/dz + 1/2 beta2 d^2A/dt^2 + |A|^2 A = 0

The canvas is a waterfall: time runs across, propagation distance runs down, so
the same equation that made filaments in the spatial case now shows a pulse
breathe (N = 2), split and re-form (N = 3), and radiate; the panel plots the
spectrum, which broadens because the pulse writes a time-dependent phase on
itself.  N = 1 is a fixed point: nothing changes at all.
"""
import time

import numpy as np

from common.lab_window import LabWindow
from common.qt_base import (
    LabeledSlider, QColor, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPainter, QPen, QPointF, QPushButton, QRectF, QSizePolicy, QTimer, QVBoxLayout,
    QWidget, Qt, run_app,
)
from common.field_canvas import ArrayCanvas
from support.physics.spm import TemporalNLSE

EQUATION_HTML = (
    '<div style="font-size:13px; color:#1b1b1b;">i&thinsp;∂A/∂z + ½β₂&thinsp;∂²A/∂t² + |A|²A = 0</div>'
    '<div style="font-size:10px; color:#4a4a4a; margin-top:5px;">'
    "色散项让脉冲展宽，非线性项让它自己变慢/变快（自相位调制），"
    "两者平衡就是孤子。</div>"
    '<div style="font-size:10px; color:#4a4a4a; margin-top:3px;">'
    "阶数 N = √(γP₀T₀²/|β₂|)：1 不变、2 呼吸、≥3 分裂。</div>"
)

EQUATION_TIP = (
    "横轴是时间（脉冲形状），竖轴是传播距离 z：每一行是同一时刻的脉冲剖面。\n\n"
    "· N < 1：色散赢，脉冲展宽；\n"
    "· N = 1：基孤子，形状与谱都不变；\n"
    "· N = 2：一个周期 π/2 后回到原样（压缩—展宽—复原）；\n"
    "· N = 3：先分裂成两个峰再合拢，辐射出线性波。\n\n"
    "右边的小图是当前的频谱：自相位调制让谱对称展宽，展宽量随 N 增大。\n"
    "这正是超连续谱的起点。"
)


class SpectrumPlot(QWidget):
    """当前脉冲的频谱（dB），范围固定在 ±ωmax。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(96)
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
        left, top, right, bottom = 6.0, 8.0, 6.0, 16.0
        plot = QRectF(left, top, max(10.0, rect.width() - left - right),
                      max(10.0, rect.height() - top - bottom))
        curve = sim.spectrum_db()
        x = np.linspace(0.0, 1.0, curve.size)
        points = [QPointF(plot.left() + plot.width() * xi,
                          plot.bottom() - plot.height() * yi)
                  for xi, yi in zip(x, curve)]
        painter.setPen(QPen(QColor(24, 96, 150), 1.5))
        painter.drawPolyline(points)
        font = painter.font()
        font.setPointSizeF(7.5)
        painter.setFont(font)
        painter.setPen(QColor(110, 110, 110))
        painter.drawText(QRectF(plot.left(), rect.bottom() - 14, 90, 12),
                         Qt.AlignLeft, "−ω ← 0 → +ω")
        painter.drawText(QRectF(rect.right() - 110, rect.bottom() - 14, 104, 12),
                         Qt.AlignRight, "Δω = %.2f" % sim.spectral_width())
        painter.end()


class SPMCanvas(ArrayCanvas):
    """水落图画布：每帧把累积的 |A|²(t) 历史画出来。"""

    def __init__(self, window, parent=None):
        super().__init__(aspect=1.0, minimum=(480, 480), parent=parent)
        self.setFixedSize(600, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.window = window
        self.sim = TemporalNLSE()

    def render_frame(self):
        self.set_array(self.sim.render_waterfall())


class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPM · 自相位调制与孤子")
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.canvas = SPMCanvas(self)
        self.paused = False
        self._frames = 0
        self._fps_clock = time.perf_counter()
        self.init_ui()
        self.setMinimumSize(960, 640)
        self.resize(960, 640)
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

        equation_box = QGroupBox("方程")
        equation_layout = QVBoxLayout(equation_box)
        self.equation_label = QLabel(EQUATION_HTML)
        self.equation_label.setTextFormat(Qt.RichText)
        self.equation_label.setWordWrap(True)
        self.equation_label.setToolTip(EQUATION_TIP)
        equation_layout.addWidget(self.equation_label)
        column.addWidget(equation_box)

        init_box = QGroupBox("输入脉冲")
        init_layout = QVBoxLayout(init_box)
        self.order_slider = LabeledSlider(
            "阶数 N", 40, 340, 200, formatter=lambda v: "%.1f" % (v / 100.0)
        )
        self.order_slider.valueChanged.connect(lambda _: self.reset())
        self.order_slider.setToolTip(
            "sech 脉冲的振幅：N=1 是基孤子（不变），N=2 呼吸，N≥3 分裂。"
        )
        init_layout.addWidget(self.order_slider)
        self.width_slider = LabeledSlider(
            "脉宽 T₀", 60, 200, 100, formatter=lambda v: "%.2f" % (v / 100.0)
        )
        self.width_slider.valueChanged.connect(lambda _: self.reset())
        init_layout.addWidget(self.width_slider)
        column.addWidget(init_box)

        run_box = QGroupBox("演化")
        run_layout = QVBoxLayout(run_box)
        self.steps_slider = LabeledSlider("每帧步数", 1, 8, 2)
        run_layout.addWidget(self.steps_slider)
        column.addWidget(run_box)

        spectrum_box = QGroupBox("当前频谱")
        spectrum_layout = QVBoxLayout(spectrum_box)
        self.spectrum_plot = SpectrumPlot()
        spectrum_layout.addWidget(self.spectrum_plot)
        column.addWidget(spectrum_box)
        column.addStretch(1)

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
        self.statusBar().showMessage("时间域的 NLSE：孤子呼吸与自相位调制展宽", 6000)

    # ---------- 参数 ----------
    def order(self):
        return self.order_slider.value() / 100.0

    def pulse_width(self):
        return self.width_slider.value() / 100.0

    def reset(self):
        self.paused = False
        self.pause_button.setText("暂停")
        self.canvas.sim.reset(self.order(), self.pulse_width())
        self.canvas.render_frame()
        self.update_readout()

    # ---------- 循环 ----------
    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("继续" if self.paused else "暂停")

    def tick(self):
        if self.paused:
            return
        sim = self.canvas.sim
        sim.step(self.steps_slider.value())
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
        sim = self.canvas.sim
        self.readout_label.setText(
            "能量 %.2f · 峰值 %.2f · 时宽 %.2f · 谱宽 %.2f · z %.2f"
            % (sim.energy(), sim.peak(), sim.width(), sim.spectral_width(),
               sim.steps * sim.dz)
        )
        self.spectrum_plot.update_state(sim)

    # ---------- 保存 ----------
    def save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存当前画面", "SPM.png", "PNG (*.png)"
        )
        if not path:
            return
        if not self.canvas.grab().save(path):
            QMessageBox.warning(self, "保存失败", "无法写入 " + path)


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))