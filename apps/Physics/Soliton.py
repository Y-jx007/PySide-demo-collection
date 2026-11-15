# -*- coding: utf-8 -*-
"""Soliton: the focusing two-dimensional nonlinear Schrodinger equation.

    i psi_t = -1/2 laplacian(psi) - |psi|^2 psi

Every preset is the same equation with a different initial condition: a single
beam either diffracts away or collapses, a vortex ring breaks into filaments,
two beams collide, and a wide noisy disc sprouts a lattice of filaments through
modulational instability.  Left click drops another beam into the running field.
"""
import time

import numpy as np
from common.lab_window import LabWindow
from common.qt_base import (
    LabeledSlider, QCheckBox, QColor, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPainter, QPen, QPointF, QPushButton,
    QRectF, QSizePolicy, QTimer, QVBoxLayout, QWidget, Qt, run_app,
)
from common.field_canvas import ArrayCanvas
from support.physics.nls import MODELS, NLSE, PALETTE, PRESETS


def nb(text):
    """Word-join a formula so it never wraps in the middle (|psi|^2 stays whole)."""
    return "\u2060".join(text)


SQUARED_PSI = nb("|ψ|²")

EQUATION_HTML = (
    '<div style="font-size:13px; color:#1b1b1b;">i&thinsp;∂ψ/∂t = −½∇²ψ − |ψ|²ψ</div>'
    '<div style="font-size:10px; color:#4a4a4a; margin-top:5px;">'
    "Kerr 效应：强光抬高介质折射率 n = n₀ + n₂I，而 I ∝ " + SQUARED_PSI +
    "——亮处势更低，光束给自己磨一枚透镜。</div>"
    '<div style="font-size:10px; color:#4a4a4a; margin-top:3px;">'
    "衍射想摊开，自聚焦想收紧：低于阈值散开，超过则坍缩。</div>"
    '<div style="font-size:11px; color:#1b1b1b; margin-top:5px;">'
    "饱和 " + nb("|ψ|² → |ψ|²/(1 + |ψ|²/I") + "<sub>s</sub>)</div>"
    '<div style="font-size:10px; color:#5c5c5c; margin-top:2px;">'
    "布居差 ΔN = 1/(1 + I/I<sub>s</sub>)：泵浦 ∝ I 与自发弛豫竞争的结果；"
    "非线性响应 ∝ I·ΔN，所以 |ψ|² 要乘上这个因子。</div>"
    '<div style="font-size:10px; color:#5c5c5c; margin-top:2px;">'
    "弱光回到 Kerr，强光下介质被漂白、响应封顶，坍缩停在有限尺度。</div>"
)

EQUATION_TIP = (
    "同一个方程，四种初始条件：\n"
    "· 单束：窄高斯，看衍射与自聚焦谁赢；\n"
    "· 涡旋：相位绕核一圈转 3·2π（电荷 m=3），环本身不稳定，"
    "裂成一圈光丝；\n"
    "· 双束：两束对撞、干涉后合并；\n"
    "· 噪声海：均匀背景 + 噪声，最快的空间频率被放大，"
    "长出六角状光丝网络。\n\n"
    "左边是主方程，非线性项与色散项不可交换，所以用 Strang 分步："
    "各切一半把线性部分放进 k 空间精确求解。\n\n"
    "为什么非线性项是 |ψ|²ψ：它不是外势，而是波自己造出来的势——"
    "介质折射率随光强变化（Kerr 效应），光强正比于 |ψ|²，"
    "于是有效势 V = −|ψ|²；再往高阶展开就是饱和项。"
)


class ResponsePlot(QWidget):
    """面板里的小图：响应因子与两能级布居随光强的变化。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(104)
        self.sim = None
        self.peak = 0.0
        self.setToolTip(
            "横轴是光强，纵轴是响应因子与能级布居：\n"
            "· ΔN/N = 1/(1+I/Iₛ)：还能响应的比例；\n"
            "· N₁/N、N₂/N：基态与激发态的占比，强光下趋于均分（漂白）；\n"
            "· 竖线是当前峰值强度所在的位置。"
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

        left, top, right, bottom = 28.0, 10.0, 8.0, 20.0
        plot = QRectF(left, top, max(10.0, rect.width() - left - right),
                      max(10.0, rect.height() - top - bottom))
        saturable = sim.model == "饱和"
        x_max = 2.0 * sim.saturation if saturable else max(1.0 / max(sim.kappa, 1e-6), 1.0)
        if saturable:
            x = np.linspace(0.0, x_max, 160)
        else:
            x = np.linspace(0.0, x_max, 160)

        def sx(value):
            return plot.left() + plot.width() * min(max(value / x_max, 0.0), 1.0)

        def sy(value):
            return plot.bottom() - plot.height() * min(max(value, 0.0), 1.0)

        grid = QColor(238, 238, 238)
        painter.setPen(QPen(grid, 1))
        for fraction in (0.25, 0.5, 0.75):
            y = sy(fraction)
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        painter.setPen(QPen(QColor(190, 190, 190), 1))
        painter.drawLine(QPointF(plot.left(), plot.bottom()),
                         QPointF(plot.right(), plot.bottom()))
        painter.drawLine(QPointF(plot.left(), plot.top()),
                         QPointF(plot.left(), plot.bottom()))

        # 响应因子
        curve = sim.response(x)
        painter.setPen(QPen(QColor(24, 96, 150), 1.8))
        points = [QPointF(sx(xi), sy(yi)) for xi, yi in zip(x, curve)]
        painter.drawPolyline(points)

        # 两能级布居（只有饱和模型才有这个解释）
        if saturable:
            s = x / sim.saturation
            n1 = (s + 2.0) / (2.0 * (1.0 + s))
            n2 = s / (2.0 * (1.0 + s))
            painter.setPen(QPen(QColor(150, 150, 150), 1.0, Qt.DashLine))
            painter.drawPolyline([QPointF(sx(xi), sy(yi)) for xi, yi in zip(x, n1)])
            painter.drawPolyline([QPointF(sx(xi), sy(yi)) for xi, yi in zip(x, n2)])

        # 当前峰值位置的竖线
        painter.setPen(QPen(QColor(200, 120, 40), 1.2, Qt.DashLine))
        cursor = sx(self.peak)
        painter.drawLine(QPointF(cursor, plot.top()), QPointF(cursor, plot.bottom()))
        marker = float(sim.response(self.peak))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(200, 120, 40))
        painter.drawEllipse(QPointF(cursor, sy(marker)), 3.0, 3.0)

        font = painter.font()
        font.setPointSizeF(7.5)
        painter.setFont(font)
        painter.setPen(QColor(110, 110, 110))
        painter.drawText(QRectF(rect.left() + 4, rect.top() + 2, 24, 10),
                         Qt.AlignLeft, "1.0")
        painter.drawText(QRectF(rect.left() + 4, rect.bottom() - 24, 24, 10),
                         Qt.AlignLeft, "0")
        painter.drawText(QRectF(rect.left() + 26, rect.bottom() - 15, 120, 12),
                         Qt.AlignLeft, "光强 I")
        if saturable:
            ratio = 1.0 / (1.0 + self.peak / sim.saturation)
            painter.drawText(QRectF(rect.right() - 118, rect.bottom() - 15, 112, 12),
                             Qt.AlignRight, "ΔN/N = %.2f" % ratio)
        else:
            ratio = max(1.0 - sim.kappa * self.peak, 0.0)
            painter.drawText(QRectF(rect.right() - 118, rect.bottom() - 15, 112, 12),
                             Qt.AlignRight, "1−κI = %.2f" % ratio)
        painter.end()


class SolitonCanvas(ArrayCanvas):
    """Field canvas: the app pushes one RGB frame per timer tick."""

    def __init__(self, window, parent=None):
        super().__init__(aspect=1.0, minimum=(480, 480), parent=parent)
        self.setFixedSize(600, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)
        self.window = window
        self.sim = NLSE()
        self.log_view = False
        self.palette = PALETTE
        self.mode = "场"

    def render_frame(self):
        if self.mode == "动量谱":
            self.set_array(self.sim.render_spectrum(floor_db=35.0))
        else:
            self.set_array(self.sim.render(self.palette, log=self.log_view))

    def world_point(self, pos):
        """Canvas position -> simulation coordinates (None when outside)."""
        mapped = self.map_pos(pos)
        if mapped is None:
            return None
        u, v = mapped
        half = self.sim.extent / 2.0
        return (u - 0.5) * 2.0 * half, (v - 0.5) * 2.0 * half

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            point = self.world_point(event.position())
            if point is not None:
                self.sim.add_seed(point[0], point[1], self.window.amplitude())
                self.render_frame()
        super().mousePressEvent(event)


class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Soliton · 非线性薛定谔")
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.canvas = SolitonCanvas(self)
        self.paused = False
        self._frames = 0
        self._fps_clock = time.perf_counter()
        self._fps = 0.0
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

        init_box = QGroupBox("初始条件")
        init_layout = QVBoxLayout(init_box)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS)
        self.preset_combo.setCurrentText("涡旋")
        self.preset_combo.setToolTip(EQUATION_TIP)
        self.preset_combo.currentTextChanged.connect(lambda _: self.reset())
        init_layout.addWidget(self.preset_combo)
        self.amp_slider = LabeledSlider(
            "振幅", 20, 120, 85, formatter=lambda v: "%.2f" % (v / 100.0)
        )
        self.amp_slider.valueChanged.connect(self.on_amplitude)
        init_layout.addWidget(self.amp_slider)
        self.noise_slider = LabeledSlider(
            "噪声", 0, 100, 40, formatter=lambda v: "%.3f" % (v / 1000.0)
        )
        self.noise_slider.valueChanged.connect(lambda _: self.soft_reset())
        init_layout.addWidget(self.noise_slider)
        column.addWidget(init_box)

        run_box = QGroupBox("演化")
        run_layout = QVBoxLayout(run_box)
        self.steps_slider = LabeledSlider("每帧步数", 1, 24, 3)
        run_layout.addWidget(self.steps_slider)
        self.log_check = QCheckBox("对数强度（−30 dB）")
        self.log_check.setChecked(False)
        self.log_check.setToolTip(
            "线性档只看得到主瓣；对数档把 30 dB 内的辐射环一起压进色阶。"
        )
        self.log_check.toggled.connect(self.on_log)
        run_layout.addWidget(self.log_check)
        selects = QHBoxLayout()
        selects.setSpacing(6)
        self.view_combo = QComboBox()
        self.view_combo.addItems(("场", "动量谱"))
        self.view_combo.currentTextChanged.connect(self.on_view)
        self.view_combo.setToolTip(
            "动量谱是场的二维傅里叶谱（dB）：调制不稳定性选的波矢、"
            "坍缩辐射的环、涡旋的环形谱都会直接显示出来。"
        )
        selects.addWidget(self.view_combo, 1)
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODELS)
        self.model_combo.currentTextChanged.connect(self.on_model)
        self.model_combo.setToolTip(
            "饱和：共振介质的两能级响应 1/(1+I/Iₛ)；\n"
            "高阶 Kerr：χ⁽⁵⁾ 修正 1−κI，这才是真实玻璃里停止坍缩的机制。"
        )
        selects.addWidget(self.model_combo, 1)
        run_layout.addLayout(selects)
        self.kappa_slider = LabeledSlider(
            "κ", 1, 80, 20, formatter=lambda v: "%.3f" % (v / 1000.0)
        )
        self.kappa_slider.valueChanged.connect(self.on_kappa)
        self.kappa_slider.setVisible(False)
        run_layout.addWidget(self.kappa_slider)
        column.addWidget(run_box)

        micro_box = QGroupBox("微观响应")
        micro_layout = QVBoxLayout(micro_box)
        self.response_plot = ResponsePlot()
        micro_layout.addWidget(self.response_plot)
        column.addWidget(micro_box)
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
        self.statusBar().addPermanentWidget(self.readout_label)
        self.fps_label = QLabel("")
        self.fps_label.setStyleSheet("padding-left: 14px;")
        self.statusBar().addPermanentWidget(self.fps_label)
        self.statusBar().showMessage(
            "非线性薛定谔：自聚焦、涡旋破裂与光丝；左键点击可在该处再种一束光", 6000
        )

    # ---------- 参数 ----------
    def amplitude(self):
        return self.amp_slider.value() / 100.0

    def noise(self):
        return self.noise_slider.value() / 1000.0

    def preset(self):
        return self.preset_combo.currentText()

    def on_amplitude(self, _value):
        self.soft_reset()

    def on_log(self, checked):
        self.canvas.log_view = bool(checked)
        self.canvas.render_frame()

    def on_view(self, name):
        self.canvas.mode = name
        self.canvas.render_frame()

    def on_model(self, name):
        sim = self.canvas.sim
        sim.model = name
        self.kappa_slider.setVisible(name == "高阶 Kerr")
        sim.kappa = self.kappa_slider.value() / 1000.0
        self.soft_reset()

    def on_kappa(self, value):
        self.canvas.sim.kappa = value / 1000.0
        self.soft_reset()

    def soft_reset(self):
        """Restart from the same preset with the current sliders."""
        self.canvas.sim.reset(self.preset(), self.amplitude(), self.noise())
        self.canvas.render_frame()
        self.update_readout()

    def reset(self):
        self.paused = False
        self.pause_button.setText("暂停")
        self.soft_reset()

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
            self._fps = self._frames / elapsed
            self._frames = 0
            self._fps_clock = now
            self.fps_label.setText("%.0f fps" % self._fps)
        self.update_readout()

    def update_readout(self):
        sim = self.canvas.sim
        peak = sim.peak()
        self.readout_label.setText(
            "功率 %.2f · 峰值 %.2f · 束宽 %.2f · 步数 %d"
            % (sim.power(), peak, sim.width(), sim.steps)
        )
        self.response_plot.update_state(sim, peak)

    # ---------- 保存 ----------
    def save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存当前画面", "Soliton.png", "PNG (*.png)"
        )
        if not path:
            return
        if not self.canvas.grab().save(path):
            QMessageBox.warning(self, "保存失败", "无法写入 " + path)


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))