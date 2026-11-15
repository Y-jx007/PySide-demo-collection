# -*- coding: utf-8 -*-
"""SHG: second-harmonic generation -- symmetry, phase matching and QPM.

Two pump photons make one photon at twice the frequency, but only where the
medium is non-centrosymmetric (chi-2 must not vanish) and only while the
shifted waves stay in step.  The canvas is the phase-matching diagram:
propagation distance across, phase mismatch up, second-harmonic power as
colour.  Stripes fanning out from the centre are the Maker fringes; turning on
periodic poling folds them into two sharp resonances, and making the medium
centrosymmetric erases everything.
"""
import time

import numpy as np

from common.lab_window import LabWindow
from common.qt_base import (
    LabeledSlider, QCheckBox, QColor, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPainter, QPen, QPointF, QPushButton, QRectF, QSizePolicy, QTimer,
    QVBoxLayout, QWidget, Qt, run_app,
)
from common.field_canvas import ArrayCanvas
from support.physics.shg import SHGMap

EQUATION_HTML = (
    '<div style="font-size:12px; color:#1b1b1b;">dA₁/dz = iκ·A₂A₁*&thinsp;e<sup>−iΔkz</sup></div>'
    '<div style="font-size:12px; color:#1b1b1b;">dA₂/dz = i(κ/2)·A₁²&thinsp;e<sup>+iΔkz</sup></div>'
    '<div style="font-size:10px; color:#4a4a4a; margin-top:5px;">'
    "κ ∝ χ⁽²⁾：只有非中心对称介质才不为零（关闭对称性 → 整幅图归零）。</div>"
    '<div style="font-size:10px; color:#4a4a4a; margin-top:3px;">'
    "Δk = k₂ − 2k₁：失配让能量在基频与倍频之间来回倒，相干长度 L_c = π/Δk。</div>"
    '<div style="font-size:10px; color:#4a4a4a; margin-top:3px;">'
    "周期极化每 Λ/2 翻转一次 κ，相当于补给 Δk = 2π/Λ 的动量。</div>"
)

EQUATION_TIP = (
    "横轴是晶体里的位置 z，竖轴是相位失配 Δk，颜色是倍频功率（dB）。\n\n"
    "· 中央暗带（Δk≈0）：倍频功率沿 z 单调增长，深处因泵浦耗尽而回落；\n"
    "· 两侧扇形条纹：Maker 条纹，Δk 越大振荡越快，第一个零点在 Δk·L = 2π；\n"
    "· 开周期极化：条纹折成两条共振（Δk = ±2π/Λ），效率回升；\n"
    "· 关对称性（κ=0）：χ⁽²⁾ 不存在，整幅图空白。\n\n"
    "下面小图是出射面的效率随 Δk 的变化，就是实验里扫波长看到的 Maker 曲线。\n"
    "图下那条光带是同一束光在 Δk=0 时的颜色变化：基频（红）逐渐转成倍频（绿），"
    "中途两束光重叠的地方看起来偏黄——实验室里就是这样。"
)


class FringePlot(QWidget):
    """出射面效率 vs Δk：实验室里量到的 Maker 曲线。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(96)
        self.model = None
        self.setToolTip("出射面的倍频效率随相位失配的变化。")

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
        left, top, right, bottom = 6.0, 8.0, 6.0, 16.0
        plot = QRectF(left, top, max(10.0, rect.width() - left - right),
                      max(10.0, rect.height() - top - bottom))
        curve = model.efficiency()
        peak = float(curve.max()) or 1.0
        x = np.linspace(0.0, 1.0, curve.size)
        points = [QPointF(plot.left() + plot.width() * xi,
                          plot.bottom() - plot.height() * (yi / peak))
                  for xi, yi in zip(x, curve)]
        painter.setPen(QPen(QColor(24, 96, 150), 1.4))
        painter.drawPolyline(points)
        font = painter.font()
        font.setPointSizeF(7.5)
        painter.setFont(font)
        painter.setPen(QColor(110, 110, 110))
        painter.drawText(QRectF(plot.left(), rect.bottom() - 14, 120, 12),
                         Qt.AlignLeft, "−Δk ← 0 → +Δk")
        painter.drawText(QRectF(rect.right() - 120, rect.bottom() - 14, 114, 12),
                         Qt.AlignRight, "峰值 %.1f%%" % (model.peak_efficiency() * 100.0))
        painter.end()


class SHGCanvas(ArrayCanvas):
    """相位匹配图：横轴传播距离、竖轴失配。"""

    def __init__(self, window, parent=None):
        # 上图是 (Δk, z) 相位匹配图，下面再接一条光束示意，宽高比约 1.15
        super().__init__(aspect=1.15, minimum=(480, 320), parent=parent)
        self.setFixedSize(600, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.window = window
        self.model = SHGMap()

    def render_frame(self):
        # 上图：行是 Δk、列是 z（光从左往右）；下图：同一束光的颜色随转换变化
        self.set_array(np.ascontiguousarray(self.model.render_with_beams()))


class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SHG · 倍频与相位匹配")
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.canvas = SHGCanvas(self)
        self._frames = 0
        self._fps_clock = time.perf_counter()
        self.init_ui()
        self.setMinimumSize(960, 640)
        self.resize(960, 640)
        self.setFocusPolicy(Qt.StrongFocus)
        self.refresh()

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

        crystal_box = QGroupBox("晶体")
        crystal_layout = QVBoxLayout(crystal_box)
        self.length_slider = LabeledSlider(
            "长度 L", 50, 300, 200, formatter=lambda v: "%.2f" % (v / 100.0)
        )
        self.length_slider.valueChanged.connect(self.on_length)
        crystal_layout.addWidget(self.length_slider)
        self.kappa_slider = LabeledSlider(
            "非线性 κ", 20, 200, 100, formatter=lambda v: "%.2f" % (v / 100.0)
        )
        self.kappa_slider.valueChanged.connect(self.on_kappa)
        crystal_layout.addWidget(self.kappa_slider)
        self.symmetric_check = QCheckBox("中心对称介质（χ⁽²⁾ = 0）")
        self.symmetric_check.toggled.connect(self.on_symmetry)
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

        fringe_box = QGroupBox("出射面效率")
        fringe_layout = QVBoxLayout(fringe_box)
        self.fringe_plot = FringePlot()
        fringe_layout.addWidget(self.fringe_plot)
        column.addWidget(fringe_box)
        column.addStretch(1)

        buttons = QHBoxLayout()
        reset_button = QPushButton("重置")
        reset_button.clicked.connect(self.reset)
        save_button = QPushButton("保存")
        save_button.clicked.connect(self.save)
        buttons.addWidget(reset_button)
        buttons.addWidget(save_button)
        column.addLayout(buttons)

        root.addWidget(panel)
        root.addWidget(self.canvas, 1)
        self.setCentralWidget(central)

        self.readout_label = QLabel("")
        self.statusBar().addPermanentWidget(self.readout_label)
        self.statusBar().showMessage(
            "χ⁽²⁾ 与相位匹配：Δk 决定能量倒向哪边，周期极化把动量补回来", 6000
        )

    # ---------- 参数 ----------
    def length(self):
        return self.length_slider.value() / 100.0

    def kappa(self):
        return self.kappa_slider.value() / 100.0

    def poling(self):
        return self.poling_slider.value() / 10.0 if self.qpm_check.isChecked() else None

    def on_length(self, _value):
        self.refresh()

    def on_kappa(self, _value):
        self.refresh()

    def on_symmetry(self, checked):
        self.canvas.model.configure(symmetric=bool(checked))
        self.after_change()

    def on_qpm(self, checked):
        self.poling_slider.setEnabled(bool(checked))
        self.canvas.model.configure(poling=self.poling() if checked else 0.0)
        self.after_change()

    def on_poling(self, _value):
        if self.qpm_check.isChecked():
            self.canvas.model.configure(poling=self.poling())
            self.after_change()

    def refresh(self):
        self.canvas.model.configure(
            length=self.length(), kappa=self.kappa(),
            symmetric=self.symmetric_check.isChecked(),
            poling=self.poling(),
        )
        self.after_change()

    def after_change(self):
        model = self.canvas.model
        self.canvas.render_frame()
        self.fringe_plot.update_model(model)
        self.readout_label.setText(
            "L %.2f · κ %.2f · 峰值效率 %.1f%% · Δk 范围 ±%.0f"
            % (model.length, model.kappa, model.peak_efficiency() * 100.0, model.span)
        )

    def reset(self):
        self.length_slider.setValue(200)
        self.kappa_slider.setValue(100)
        self.symmetric_check.setChecked(False)
        self.qpm_check.setChecked(False)
        self.poling_slider.setValue(63)
        self.refresh()

    # ---------- 保存 ----------
    def save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存当前画面", "SHG.png", "PNG (*.png)"
        )
        if not path:
            return
        if not self.canvas.grab().save(path):
            QMessageBox.warning(self, "保存失败", "无法写入 " + path)


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))