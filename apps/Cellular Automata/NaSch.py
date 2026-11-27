# -*- coding: utf-8 -*-
"""Nagel–Schreckenberg traffic flow on a periodic ring.

Rules per vehicle (parallel update):
  1. accelerate: v <- min(v+1, vmax)
  2. brake:      v <- min(v, gap-1)
  3. random braking with probability p
  4. move:       x <- x+v

The canvas shows a live lane at the top and a space-time diagram below.
Red diagonals/bands are stop-and-go waves moving backwards through traffic.
"""
import numpy as np

from common.lab_window import LabWindow
from common.taichi_base import ensure_taichi
from common.qt_base import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QImage, QLabel,
    QPainter, QPushButton, QRect, QColor, QTimer, QVBoxLayout, QWidget, Qt, run_app,
    LabeledSlider,
)

ti = ensure_taichi()

SPEED_COLORS = np.array([
    (235, 55, 55),    # v=0 stopped
    (255, 145, 45),   # v=1
    (250, 210, 70),   # v=2
    (115, 215, 130),  # v=3
    (60, 190, 240),   # v=4
    (130, 145, 255),  # v=5
    (205, 125, 255),  # v=6/7
], dtype=np.uint8)
EMPTY = np.array((12, 16, 24), dtype=np.uint8)


@ti.data_oriented
class NaSchSimulation:
    def __init__(self, length=512, density=0.22, vmax=5, brake=0.25, bottleneck=True):
        self.length = int(length)
        self.density = float(density)
        self.vmax = int(vmax)
        self.brake = float(brake)
        self.bottleneck = bool(bottleneck)
        self.history_h = 512
        self.car = ti.field(ti.i8, shape=(self.length,))
        self.vel = ti.field(ti.i8, shape=(self.length,))
        self.next_car = ti.field(ti.i8, shape=(self.length,))
        self.next_vel = ti.field(ti.i8, shape=(self.length,))
        self.live_rgb = np.zeros((1, self.length, 3), dtype=np.uint8)
        self.history_rgb = np.zeros((self.history_h, self.length, 3), dtype=np.uint8)
        self.car_np = np.zeros(self.length, dtype=np.int8)
        self.vel_np = np.zeros(self.length, dtype=np.int8)
        self._reset_kernel()
        self.history_rgb[:] = EMPTY
        self.warm_history(self.history_h)

    @ti.kernel
    def _reset_kernel(self):
        for x in range(self.length):
            self.car[x] = ti.cast(0, ti.i8)
            self.vel[x] = ti.cast(0, ti.i8)
        for x in range(self.length):
            if ti.random() < self.density:
                self.car[x] = ti.cast(1, ti.i8)
                self.vel[x] = ti.cast(ti.random() * (self.vmax + 1), ti.i8)

    def reset(self, density=None):
        if density is not None:
            self.density = float(density)
        self._reset_kernel()
        self.history_rgb[:] = EMPTY
        self.warm_history(self.history_h)

    def warm_history(self, steps: int):
        """Fill the space-time panel so jams are visible immediately."""
        for _ in range(max(0, int(steps))):
            self.step()
            self.snapshot()

    @ti.kernel
    def _step_kernel(self, car: ti.template(), vel: ti.template(),
                     next_car: ti.template(), next_vel: ti.template(),
                     vmax: ti.i32, brake: ti.f32,
                     bottleneck: ti.i32, slow_start: ti.i32, slow_end: ti.i32):
        for x in range(self.length):
            next_car[x] = ti.cast(0, ti.i8)
            next_vel[x] = ti.cast(0, ti.i8)
        for x in range(self.length):
            if car[x]:
                v = ti.cast(vel[x], ti.i32)
                gap = self.length
                for d in range(1, vmax + 2):
                    if car[(x + d) % self.length]:
                        gap = d
                        break
                v = ti.min(v + 1, vmax)
                v = ti.min(v, gap - 1)
                if ti.random() < brake:
                    v = ti.max(v - 1, 0)
                if bottleneck and slow_start <= x <= slow_end:
                    v = ti.min(v, 1)
                nx = (x + v) % self.length
                next_car[nx] = ti.cast(1, ti.i8)
                next_vel[nx] = ti.cast(v, ti.i8)

    def step(self):
        self._step_kernel(self.car, self.vel, self.next_car, self.next_vel,
                         self.vmax, self.brake,
                         1 if self.bottleneck else 0,
                         self.length // 2 - 3, self.length // 2 + 3)
        # The kernel is asynchronous; swapping the Python field references
        # before it finishes can make multi-step frames read/write the wrong
        # buffers and appear completely frozen.
        ti.sync()
        self.car, self.next_car = self.next_car, self.car
        self.vel, self.next_vel = self.next_vel, self.vel

    def snapshot(self):
        self.car_np = self.car.to_numpy()
        self.vel_np = self.vel.to_numpy()
        occupied = self.car_np != 0
        row = np.full((self.length, 3), EMPTY, dtype=np.uint8)
        if occupied.any():
            vv = np.clip(self.vel_np[occupied], 0, len(SPEED_COLORS) - 1)
            row[occupied] = SPEED_COLORS[vv]
        # Space-time: local density, not individual car pixels.  High-density
        # bands are traffic jams and move backwards through the lane.
        occ = occupied.astype(np.float32)
        kernel = np.ones(15, dtype=np.float32) / 15.0
        density = np.convolve(occ, kernel, mode="same")
        t = np.clip(density / 0.75, 0.0, 1.0)
        hist_row = np.empty((self.length, 3), dtype=np.uint8)
        hist_row[:, 0] = (24 + 220 * t).astype(np.uint8)
        hist_row[:, 1] = (46 + 40 * (1.0 - t)).astype(np.uint8)
        hist_row[:, 2] = (78 * (1.0 - t) + 28).astype(np.uint8)
        self.live_rgb[0] = row
        self.history_rgb[:-1] = self.history_rgb[1:]
        self.history_rgb[-1] = hist_row
        return self.live_rgb, self.history_rgb

    def stats(self):
        occupied = self.car_np != 0
        cars = int(occupied.sum())
        if cars == 0:
            return 0, 0.0, 0.0, 0.0
        avg_v = float(self.vel_np[occupied].mean())
        flow = float(self.vel_np[occupied].sum()) / self.length
        stopped = float(np.count_nonzero(self.vel_np[occupied] == 0)) / cars
        return cars, avg_v, flow, stopped


class NaSchCanvas(QWidget):
    def __init__(self, sim: NaSchSimulation, parent=None):
        super().__init__(parent)
        self.sim = sim
        self._live_q = None
        self._hist_q = None
        self.setFixedSize(640, 640)
        self.set_sim(sim)

    def set_sim(self, sim: NaSchSimulation):
        self.sim = sim
        self._live_q = QImage(sim.live_rgb.data, sim.length, 1,
                              sim.length * 3, QImage.Format_RGB888)
        self._hist_q = QImage(sim.history_rgb.data, sim.length, sim.history_h,
                              sim.length * 3, QImage.Format_RGB888)
        self.update()

    def update_display(self):
        self._live_q = QImage(self.sim.live_rgb.data, self.sim.length, 1,
                              self.sim.length * 3, QImage.Format_RGB888)
        self._hist_q = QImage(self.sim.history_rgb.data, self.sim.length,
                              self.sim.history_h, self.sim.length * 3,
                              QImage.Format_RGB888)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(12, 16, 24))
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        # live lane
        p.drawImage(QRect(0, 0, 640, 72), self._live_q)
        p.setPen(QColor(220, 226, 235))
        p.drawText(QRect(8, 4, 200, 18), Qt.AlignLeft, "LIVE  ·  车辆颜色 = 速度")
        # space-time diagram
        p.drawImage(QRect(0, 76, 640, 560), self._hist_q)
        p.setPen(QColor(160, 170, 185))
        p.drawText(QRect(8, 78, 430, 18), Qt.AlignLeft,
                   "SPACE–TIME · 局部密度 · 下 = 现在")
        p.drawText(QRect(8, 612, 620, 18), Qt.AlignLeft,
                   "横向：道路位置    纵向：时间    红色高密度带 = 堵车波")


class NaSchApp(LabWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.setWindowTitle("Nagel–Schreckenberg · 交通流")
        self.resize(1000, 720)
        self.setMinimumSize(1000, 720)
        self.length = 512
        self.density = 0.35
        self.vmax = 5
        self.brake = 0.35
        self.steps_per_frame = 2
        self.running = True
        self.sim = NaSchSimulation(self.length, self.density, self.vmax, self.brake, False)
        self.canvas = NaSchCanvas(self.sim)
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)
        self.update_stats()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(330)
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        ctrl = QGroupBox("模拟")
        c = QHBoxLayout(ctrl)
        c.setContentsMargins(8, 6, 8, 6)
        self.run_btn = QPushButton("暂停")
        self.run_btn.clicked.connect(self.toggle_running)
        step_btn = QPushButton("单步")
        step_btn.clicked.connect(self.single_step)
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset)
        for b in (self.run_btn, step_btn, reset_btn):
            c.addWidget(b, 1)
        col.addWidget(ctrl)

        params = QGroupBox("交通参数")
        form = QFormLayout(params)
        form.setContentsMargins(8, 6, 8, 6)
        form.setSpacing(5)
        self.size_combo = QComboBox()
        self.size_combo.addItems(["256", "512", "1024"])
        self.size_combo.setCurrentText(str(self.length))
        self.size_combo.currentTextChanged.connect(self.change_size)
        form.addRow("道路长度", self.size_combo)

        self.density_slider = LabeledSlider("", 5, 60, int(self.density * 100),
                                            formatter=lambda v: f"{v}%", value_width=42)
        self.density_slider.valueChanged.connect(self.change_density)
        form.addRow("密度", self.density_slider)

        self.vmax_slider = LabeledSlider("", 1, 7, self.vmax,
                                         formatter=lambda v: f"{v}", value_width=42)
        self.vmax_slider.valueChanged.connect(self.change_vmax)
        form.addRow("最高速度", self.vmax_slider)

        self.brake_slider = LabeledSlider("", 0, 60, int(self.brake * 100),
                                          formatter=lambda v: f"{v}%", value_width=42)
        self.brake_slider.valueChanged.connect(self.change_brake)
        form.addRow("随机刹车", self.brake_slider)

        self.speed_slider = LabeledSlider("", 1, 12, self.steps_per_frame,
                                          formatter=lambda v: f"{v}×", value_width=42)
        self.speed_slider.valueChanged.connect(self.change_speed)
        form.addRow("每帧步数", self.speed_slider)
        self.bottleneck_check = QCheckBox("中间瓶颈（慢速区）")
        self.bottleneck_check.setChecked(False)
        self.bottleneck_check.toggled.connect(self.change_bottleneck)
        form.addRow("", self.bottleneck_check)
        col.addWidget(params)

        presets = QGroupBox("预设")
        p = QHBoxLayout(presets)
        p.setContentsMargins(8, 6, 8, 6)
        for label, d, br in (("自由流", 0.15, 0.12),
                             ("走走停停", 0.24, 0.28),
                             ("拥堵", 0.38, 0.35)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, dd=d, bb=br: self.apply_preset(dd, bb))
            p.addWidget(b, 1)
        col.addWidget(presets)

        stats = QGroupBox("读数")
        s = QVBoxLayout(stats)
        s.setContentsMargins(8, 6, 8, 6)
        self.lbl_cars = QLabel("车辆  –")
        self.lbl_speed = QLabel("平均速度  –")
        self.lbl_flow = QLabel("流量  –")
        self.lbl_stopped = QLabel("停车比例  –")
        for lbl in (self.lbl_cars, self.lbl_speed, self.lbl_flow, self.lbl_stopped):
            lbl.setStyleSheet("font-family: Consolas, monospace;")
            s.addWidget(lbl)
        s.addSpacing(4)
        s.addWidget(QLabel("颜色 = 速度"))
        legend = QLabel(
            '<span style="color:#eb3737">■</span> 0  '
            '<span style="color:#ff912d">■</span> 1  '
            '<span style="color:#fad246">■</span> 2  '
            '<span style="color:#73d782">■</span> 3  '
            '<span style="color:#3cbef0">■</span> 4  '
            '<span style="color:#8291ff">■</span> 5+'
        )
        legend.setStyleSheet("font-size:11px;")
        s.addWidget(legend)
        density_key = QLabel("空间–时间：蓝 → 红 = 局部密度升高")
        density_key.setStyleSheet("color:#667085; font-size:11px;")
        s.addWidget(density_key)
        rule = QLabel("规则：加速 → 按车距减速 → 随机刹车 → 移动")
        rule.setWordWrap(True)
        rule.setStyleSheet("color:#667085; font-size:11px;")
        s.addWidget(rule)
        col.addWidget(stats)
        col.addStretch(1)

        root.addWidget(panel)
        root.addWidget(self.canvas)

        self.lbl_status = QLabel("就绪")
        self.statusBar().addWidget(self.lbl_status, 1)

    def tick(self):
        if not self.running:
            return
        for _ in range(self.steps_per_frame):
            self.sim.step()
        self.sim.snapshot()
        self.canvas.update_display()
        self.update_stats()

    def update_stats(self):
        cars, avg_v, flow, stopped = self.sim.stats()
        self.lbl_cars.setText(f"车辆  {cars:,}")
        self.lbl_speed.setText(f"平均速度  {avg_v:.2f} cells/step")
        self.lbl_flow.setText(f"流量  {flow:.3f} cars·cells/step")
        self.lbl_stopped.setText(f"停车比例  {stopped:.1%}")
        self.lbl_status.setText(
            f"NaSch · L={self.length} · 密度 {self.density:.2f} · "
            f"vmax {self.vmax} · p={self.brake:.2f}"
        )

    def toggle_running(self):
        self.running = not self.running
        self.run_btn.setText("暂停" if self.running else "开始")

    def single_step(self):
        self.sim.step()
        self.sim.snapshot()
        self.canvas.update_display()
        self.update_stats()

    def reset(self):
        self.sim.reset(self.density)
        self.sim.warm_history(self.sim.history_h)
        self.canvas.update_display()
        self.update_stats()

    def apply_preset(self, density, brake):
        self.density = float(density)
        self.brake = float(brake)
        self.density_slider.setValue(int(density * 100))
        self.brake_slider.setValue(int(brake * 100))
        self.sim.density = self.density
        self.sim.brake = self.brake
        self.sim.reset(self.density)
        self.sim.warm_history(self.sim.history_h)
        self.canvas.update_display()
        self.update_stats()

    def change_density(self, value):
        self.density = value / 100.0

    def change_vmax(self, value):
        self.vmax = int(value)
        self.sim.vmax = self.vmax

    def change_brake(self, value):
        self.brake = value / 100.0
        self.sim.brake = self.brake

    def change_speed(self, value):
        self.steps_per_frame = int(value)

    def change_bottleneck(self, checked):
        self.sim.bottleneck = bool(checked)
        self.sim.reset(self.density)
        self.sim.warm_history(self.sim.history_h)
        self.canvas.update_display()
        self.update_stats()

    def change_size(self, text):
        new_size = int(text)
        if new_size == self.length:
            return
        self.length = new_size
        self.sim = NaSchSimulation(self.length, self.density, self.vmax, self.brake, False)
        self.canvas.set_sim(self.sim)
        self.update_stats()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


MainWindow = NaSchApp


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
