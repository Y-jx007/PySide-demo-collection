# -*- coding: utf-8 -*-
"""Soliton: the focusing two-dimensional nonlinear Schrodinger equation.

    i psi_t = -1/2 laplacian(psi) - |psi|^2 psi

Every preset is the same equation with a different initial condition: a single
beam either diffracts away or collapses, a vortex ring breaks into filaments,
two beams collide, and a wide noisy disc sprouts a lattice of filaments through
modulational instability.  Left click drops another beam into the running field.
"""
import time
from common.lab_window import LabWindow
from common.qt_base import (
    LabeledSlider, QCheckBox, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSizePolicy, QTimer,
    QVBoxLayout, QWidget, Qt, run_app,
)
from common.field_canvas import ArrayCanvas
from support.physics.nls import NLSE, PALETTES, PRESETS

EQUATION_HTML = (
    '<div style="font-size:13px; color:#1b1b1b;">i&thinsp;∂ψ/∂t = −½∇²ψ − |ψ|²ψ</div>'
    '<div style="font-size:11px; color:#5c5c5c; margin-top:4px;">'
    "衍射项 −½∇²ψ 让波包散开，非线性项 −|ψ|²ψ 把亮处越聚越紧；"
    "功率低时衍射赢（散开），高时自聚焦赢（坍缩）。</div>"
    '<div style="font-size:11px; color:#1b1b1b; margin-top:6px;">'
    "饱和：|ψ|² → |ψ|²/(1+|ψ|²/I<sub>s</sub>)</div>"
    '<div style="font-size:11px; color:#1b1b1b;">'
    "分步：ψ ← e<sup>iθ/2</sup>ψ → FFT → e<sup>−ik²Δt/2</sup> → IFFT → e<sup>iθ/2</sup></div>"
    '<div style="font-size:11px; color:#5c5c5c; margin-top:6px;">'
    "P = ∫|ψ|² dxdy 守恒；θ = Δt·|ψ|²/(1+|ψ|²/I<sub>s</sub>)/2。</div>"
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


class SolitonCanvas(ArrayCanvas):
    """Field canvas: the app pushes one RGB frame per timer tick."""

    def __init__(self, window, parent=None):
        super().__init__(aspect=1.0, minimum=(480, 480), parent=parent)
        self.setFixedSize(600, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)
        self.window = window
        self.sim = NLSE()
        self.log_view = True
        self.palette = "墨青"

    def render_frame(self):
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
        column.setSpacing(6)

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
        self.steps_slider = LabeledSlider("每帧步数", 1, 6, 3)
        run_layout.addWidget(self.steps_slider)
        self.log_check = QCheckBox("对数强度")
        self.log_check.setChecked(True)
        self.log_check.toggled.connect(self.on_log)
        run_layout.addWidget(self.log_check)
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(list(PALETTES.keys()))
        self.palette_combo.setCurrentText("墨青")
        self.palette_combo.currentTextChanged.connect(self.on_palette)
        run_layout.addWidget(self.palette_combo)
        column.addWidget(run_box)

        read_box = QGroupBox("读数")
        read_layout = QVBoxLayout(read_box)
        self.power_label = QLabel("功率 —")
        self.peak_label = QLabel("峰值 —")
        self.width_label = QLabel("束宽 —")
        self.step_label = QLabel("步数 —")
        self.hint_label = QLabel("左键点击：在该处再种一束光")
        self.hint_label.setStyleSheet("color: #6b6b6b;")
        for widget in (self.power_label, self.peak_label, self.width_label,
                       self.step_label, self.hint_label):
            read_layout.addWidget(widget)
        column.addWidget(read_box)
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

        self.fps_label = QLabel("")
        self.statusBar().addPermanentWidget(self.fps_label)
        self.statusBar().showMessage("非线性薛定谔：自聚焦、涡旋破裂与光丝", 4000)

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

    def on_palette(self, name):
        self.canvas.palette = name
        self.canvas.render_frame()

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
        self.power_label.setText("功率 %.2f" % sim.power())
        self.peak_label.setText("峰值 %.2f" % sim.peak())
        self.width_label.setText("束宽 %.2f" % sim.width())
        self.step_label.setText("步数 %d" % sim.steps)

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