# -*- coding: utf-8 -*-
"""Bak–Tang–Wiesenfeld (BTW) sandpile.

Parallel toppling waves on a square lattice.  Grains leave the domain at
the open boundary, which keeps the pile bounded while preserving avalanches.
"""
import numpy as np

from common.lab_window import LabWindow
from common.taichi_base import ensure_taichi
from common.qt_base import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QImage, QLabel,
    QPainter, QPen, QColor, QFont, QPointF, QRect, QPushButton, QTimer,
    QVBoxLayout, QWidget, Qt, run_app, LabeledSlider,
)

ti = ensure_taichi()

THRESHOLD = 4


@ti.data_oriented
class SandpileSimulation:
    def __init__(self, size: int = 256):
        self.size = int(size)
        self.height = ti.field(ti.i32, shape=(self.size, self.size))
        self.next_height = ti.field(ti.i32, shape=(self.size, self.size))
        self.rgb = ti.field(ti.u8, shape=(self.size, self.size, 3))
        self.topples = ti.field(ti.i32, shape=())
        self.stats = ti.field(ti.i32, shape=2)  # total grains, max height
        self.reset_random()

    @ti.kernel
    def _reset_random_kernel(self):
        for i, j in ti.ndrange(self.size, self.size):
            self.height[i, j] = ti.cast(ti.random() * THRESHOLD, ti.i32)

    @ti.kernel
    def _reset_center_kernel(self):
        for i, j in ti.ndrange(self.size, self.size):
            self.height[i, j] = 0
        c = self.size // 2
        radius = max(6, self.size // 10)
        for i, j in ti.ndrange(self.size, self.size):
            di = i - c
            dj = j - c
            if di * di + dj * dj <= radius * radius:
                self.height[i, j] = 3

    @ti.kernel
    def _clear_kernel(self):
        for i, j in ti.ndrange(self.size, self.size):
            self.height[i, j] = 0

    def reset_random(self):
        self._reset_random_kernel()

    def reset_center(self):
        self._reset_center_kernel()

    def clear(self):
        self._clear_kernel()

    @ti.kernel
    def _drop_kernel(self, x: ti.i32, y: ti.i32, amount: ti.i32):
        if 0 <= x < self.size and 0 <= y < self.size:
            ti.atomic_add(self.height[y, x], amount)

    def drop(self, x: float, y: float, amount: int = 1):
        self._drop_kernel(int(x), int(y), int(amount))

    @ti.kernel
    def _drop_random_kernel(self, count: ti.i32):
        for _ in range(count):
            x = ti.cast(ti.random() * self.size, ti.i32)
            y = ti.cast(ti.random() * self.size, ti.i32)
            ti.atomic_add(self.height[y, x], 1)

    def drop_random(self, count: int):
        if count > 0:
            self._drop_random_kernel(int(count))

    @ti.kernel
    def _topple_kernel(self, src: ti.template(), dst: ti.template(),
                       topples: ti.template()):
        for i, j in ti.ndrange(self.size, self.size):
            h = src[i, j]
            q = h // THRESHOLD
            if q > 0:
                dst[i, j] -= q * THRESHOLD
                if i > 0:
                    ti.atomic_add(dst[i - 1, j], q)
                if i + 1 < self.size:
                    ti.atomic_add(dst[i + 1, j], q)
                if j > 0:
                    ti.atomic_add(dst[i, j - 1], q)
                if j + 1 < self.size:
                    ti.atomic_add(dst[i, j + 1], q)
                ti.atomic_add(topples[None], q)

    def step(self, waves: int = 8) -> int:
        total = 0
        self.stable = False
        for _ in range(max(1, int(waves))):
            self.next_height.copy_from(self.height)
            self.topples[None] = 0
            self._topple_kernel(self.height, self.next_height, self.topples)
            self.height, self.next_height = self.next_height, self.height
            n = int(self.topples[None])
            total += n
            if n == 0:
                self.stable = True
                break
        return total

    def relax(self, max_waves: int = 2000) -> int:
        """Run toppling waves until stable; return the number of topplings."""
        total = 0
        for _ in range(max(1, int(max_waves))):
            n = self.step(1)
            total += n
            if n == 0:
                break
        return total

    def measure_avalanche(self, x: float, y: float, max_waves: int = 2000) -> int:
        """Drop one grain and measure a single avalanche event."""
        self.drop(x, y, 1)
        return self.relax(max_waves)

    @ti.kernel
    def _stats_kernel(self):
        self.stats[0] = 0
        self.stats[1] = 0
        for i, j in ti.ndrange(self.size, self.size):
            h = self.height[i, j]
            ti.atomic_add(self.stats[0], h)
            ti.atomic_max(self.stats[1], h)

    def stats_now(self) -> tuple[int, int]:
        self._stats_kernel()
        return int(self.stats[0]), int(self.stats[1])

    @ti.kernel
    def _render_kernel(self):
        for i, j in ti.ndrange(self.size, self.size):
            h = self.height[i, j]
            r, g, b = 15, 20, 27
            if h == 1:
                r, g, b = 55, 180, 170
            elif h == 2:
                r, g, b = 70, 130, 235
            elif h == 3:
                r, g, b = 245, 190, 70
            elif h >= THRESHOLD:
                r, g, b = 255, 245, 225
            self.rgb[i, j, 0] = ti.cast(r, ti.u8)
            self.rgb[i, j, 1] = ti.cast(g, ti.u8)
            self.rgb[i, j, 2] = ti.cast(b, ti.u8)

    def render(self) -> np.ndarray:
        self._render_kernel()
        return self.rgb.to_numpy()

    @ti.kernel
    def _erase_kernel(self, x: ti.i32, y: ti.i32):
        if 0 <= x < self.size and 0 <= y < self.size:
            self.height[y, x] = 0

    def erase(self, x: float, y: float):
        self._erase_kernel(int(x), int(y))


class SandpileCanvas(QWidget):
    def __init__(self, sim: SandpileSimulation, parent=None):
        super().__init__(parent)
        self.sim = sim
        self._rgb = None
        self._image = QImage(sim.size, sim.size, QImage.Format_RGB888)
        self._image.fill(Qt.black)
        self._drawing = False
        self._erase = False
        self.setFixedSize(640, 640)

    def set_sim(self, sim: SandpileSimulation):
        self.sim = sim
        self._image = QImage(sim.size, sim.size, QImage.Format_RGB888)
        self.update_display()

    def update_display(self):
        self._rgb = self.sim.render()
        self._image = QImage(self._rgb.data, self.sim.size, self.sim.size,
                             self.sim.size * 3, QImage.Format_RGB888)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(self.rect(), self._image)

    def _cell(self, pos):
        return (pos.x() / self.width() * self.sim.size,
                pos.y() / self.height() * self.sim.size)

    def mousePressEvent(self, event):
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        self._drawing = True
        self._erase = event.button() == Qt.RightButton
        x, y = self._cell(event.position())
        if self._erase:
            self.sim.erase(x, y)
        else:
            self.sim.drop(x, y, 32)
        self.update_display()

    def mouseMoveEvent(self, event):
        if not self._drawing:
            return
        x, y = self._cell(event.position())
        if self._erase:
            self.sim.erase(x, y)
        else:
            self.sim.drop(x, y, 8)
        self.update_display()

    def mouseReleaseEvent(self, event):
        self._drawing = False


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _sup_label(exp):
    return "10" + str(int(exp)).translate(_SUP)


class AvalanchePlot(QWidget):
    """Log-log avalanche-size histogram with a power-law fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sizes = []
        self.setFixedHeight(180)
        self.setMinimumWidth(280)
        self.setStyleSheet("background:#fbfcfe; border:1px solid #e4e7ec;")

    def set_sizes(self, sizes):
        self.sizes = list(sizes[-5000:])
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#fbfcfe"))
        p.setPen(QPen(QColor("#e4e7ec"), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        p.setPen(QColor("#344054"))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRect(8, 4, self.width() - 16, 18), Qt.AlignLeft,
                   "雪崩规模分布  log–log")

        left, top, right, bottom = 46, 42, self.width() - 14, self.height() - 30
        p.setPen(QPen(QColor("#98a2b3"), 1))
        p.drawLine(left, top, left, bottom)
        p.drawLine(left, bottom, right, bottom)
        p.setFont(QFont("Segoe UI", 7))
        p.setPen(QColor("#667085"))
        p.drawText(QRect(4, top - 11, 38, 12), Qt.AlignRight, "P(s)")
        p.drawText(QRect(right - 20, bottom + 4, 18, 12), Qt.AlignRight, "s")

        if len(self.sizes) < 8:
            p.setPen(QColor("#98a2b3"))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(QRect(left, top, right - left, bottom - top),
                       Qt.AlignCenter, "采集中…")
            return
        values = np.asarray([v for v in self.sizes if v > 0], dtype=np.float64)
        if values.size < 8:
            return
        lo = max(1.0, float(values.min()))
        hi = max(lo + 1.0, float(values.max()))
        # Log bins; normalize by bin width so the vertical axis is a PDF,
        # not raw counts (otherwise increasing bin widths bias the slope).
        bins = np.logspace(np.log10(lo), np.log10(hi + 1.0), 14)
        counts, edges = np.histogram(values, bins=bins)
        widths = np.diff(edges)
        centers = np.sqrt(edges[:-1] * edges[1:])
        pdf = counts / (widths * max(1, values.size))
        good = (counts > 0) & (pdf > 0) & (centers > 0)
        if good.sum() < 3:
            return
        xs = np.log10(centers[good])
        ys = np.log10(pdf[good])
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        if xmax <= xmin:
            xmax = xmin + 0.5
        if ymax <= ymin:
            ymax = ymin + 0.5
        px_pad = max(0.15, (xmax - xmin) * 0.10)
        py_pad = max(0.15, (ymax - ymin) * 0.12)
        xmin -= px_pad; xmax += px_pad
        ymin -= py_pad; ymax += py_pad

        def px(x):
            return left + (x - xmin) / (xmax - xmin) * (right - left)

        def py(y):
            return bottom - (y - ymin) / (ymax - ymin) * (bottom - top)

        # log ticks: 10^n
        p.setFont(QFont("Segoe UI", 7))
        for exp in range(int(np.floor(xmin)), int(np.ceil(xmax)) + 1):
            xx = px(float(exp))
            if left <= xx <= right:
                p.setPen(QPen(QColor("#e4e7ec"), 1))
                p.drawLine(QPointF(xx, top), QPointF(xx, bottom))
                p.setPen(QColor("#667085"))
                p.drawText(QRect(int(xx) - 18, bottom + 3, 36, 12),
                           Qt.AlignCenter, _sup_label(exp))
        for exp in range(int(np.floor(ymin)), int(np.ceil(ymax)) + 1):
            yy = py(float(exp))
            if top <= yy <= bottom:
                p.setPen(QPen(QColor("#e4e7ec"), 1))
                p.drawLine(QPointF(left, yy), QPointF(right, yy))
                p.setPen(QColor("#667085"))
                p.drawText(QRect(2, int(yy) - 6, left - 7, 12),
                           Qt.AlignRight, _sup_label(exp))

        p.setPen(QPen(QColor("#175cd3"), 1.5))
        for k in range(len(xs) - 1):
            p.drawLine(QPointF(px(xs[k]), py(ys[k])),
                       QPointF(px(xs[k + 1]), py(ys[k + 1])))
        p.setBrush(QColor("#175cd3"))
        p.setPen(Qt.NoPen)
        for x, y in zip(xs, ys):
            p.drawEllipse(QPointF(px(x), py(y)), 2.2, 2.2)
        if len(xs) >= 4:
            slope, intercept = np.polyfit(xs, ys, 1)
            xx = np.array([xmin, xmax])
            yy = slope * xx + intercept
            p.setPen(QPen(QColor("#e04f5f"), 1, Qt.DashLine))
            p.drawLine(QPointF(px(xx[0]), py(yy[0])),
                       QPointF(px(xx[1]), py(yy[1])))
            p.setPen(QColor("#b42318"))
            p.setFont(QFont("Segoe UI", 8, QFont.Bold))
            p.drawText(QRect(left + 5, top + 1, 190, 14), Qt.AlignLeft,
                       f"拟合斜率 {slope:.2f}   n={values.size}")


class SandpileApp(LabWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.setWindowTitle("Bak–Tang–Wiesenfeld · 沙堆")
        self.resize(1000, 720)
        self.setMinimumSize(1000, 720)
        self.grid_size = 256
        self.drop_rate = 16
        self.waves = 8
        self.source = "center"
        self.running = True
        self.last_topples = 0
        self.total_topples = 0
        self.avalanche_sizes = []
        self.pending_active = False
        self.pending_total = 0
        self.sim = SandpileSimulation(self.grid_size)
        self.stats_size = 128
        self.stats_sim = SandpileSimulation(self.stats_size)
        self.stats_sim.reset_random()
        self.sim.reset_center()
        self.canvas = SandpileCanvas(self.sim)
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)
        self.canvas.update_display()
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

        params = QGroupBox("沙堆参数")
        form = QFormLayout(params)
        form.setContentsMargins(8, 6, 8, 6)
        form.setSpacing(5)
        self.size_combo = QComboBox()
        self.size_combo.addItems(["128", "256", "512"])
        self.size_combo.setCurrentText(str(self.grid_size))
        self.size_combo.currentTextChanged.connect(self.change_size)
        form.addRow("网格", self.size_combo)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["中心", "随机"])
        self.source_combo.currentTextChanged.connect(self.change_source)
        form.addRow("落点", self.source_combo)

        self.rate_slider = LabeledSlider(
            "", 1, 120, self.drop_rate,
            formatter=lambda v: f"{v}/帧", value_width=50,
        )
        self.rate_slider.valueChanged.connect(self.change_rate)
        form.addRow("落沙", self.rate_slider)

        self.wave_slider = LabeledSlider(
            "", 1, 128, self.waves,
            formatter=lambda v: f"{v}", value_width=42,
        )
        self.wave_slider.valueChanged.connect(self.change_waves)
        form.addRow("每帧波数", self.wave_slider)
        col.addWidget(params)

        presets = QGroupBox("预设")
        p = QHBoxLayout(presets)
        p.setContentsMargins(8, 6, 8, 6)
        for label, fn in (("随机", self.reset_random),
                          ("中央堆", self.reset_center),
                          ("清空", self.reset_clear)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            p.addWidget(b, 1)
        col.addWidget(presets)

        stats = QGroupBox("读数")
        s = QVBoxLayout(stats)
        s.setContentsMargins(8, 6, 8, 6)
        self.lbl_grains = QLabel("颗粒  –")
        self.lbl_avalanche = QLabel("上次雪崩  –")
        self.lbl_total = QLabel("累计崩塌  –")
        self.lbl_height = QLabel("最大高度  –")
        for lbl in (self.lbl_grains, self.lbl_avalanche, self.lbl_total, self.lbl_height):
            lbl.setStyleSheet("font-family: Consolas, monospace;")
            s.addWidget(lbl)
        s.addSpacing(4)
        s.addWidget(QLabel("高度颜色"))
        legend = QLabel(
            '0 空　　'
            '<span style="color:#37b4aa">■</span> 1 青　'
            '<span style="color:#4682eb">■</span> 2 蓝<br>'
            '<span style="color:#f5be46">■</span> 3 金　　'
            '<span style="color:#fff5e1">■</span> ≥4 白'
        )
        legend.setStyleSheet("font-size:11px;")
        s.addWidget(legend)
        col.addWidget(stats)
        self.avalanche_plot = AvalanchePlot()
        col.addWidget(self.avalanche_plot)
        col.addStretch(1)

        root.addWidget(panel)
        root.addWidget(self.canvas)

        self.lbl_status = QLabel("就绪")
        self.statusBar().addWidget(self.lbl_status, 1)

    def tick(self):
        if not self.running:
            return
        # Fast visual pile: restore the original grain throughput, but only
        # advance a bounded number of toppling waves per frame.
        if self.source == "center":
            c = self.grid_size / 2.0
            self.sim.drop(c, c, self.drop_rate)
        else:
            self.sim.drop_random(self.drop_rate)
        self.total_topples += self.sim.step(self.waves)

        # Independent 128² pile for single-avalanche power-law statistics.
        if self.source == "center":
            x = y = self.stats_size / 2.0
        else:
            x = float(np.random.randint(0, self.stats_size))
            y = float(np.random.randint(0, self.stats_size))
        self.last_topples = self.stats_sim.measure_avalanche(x, y, 2000)
        if self.last_topples > 0:
            self.avalanche_sizes.append(self.last_topples)
            self.avalanche_sizes = self.avalanche_sizes[-5000:]

        self.pending_active = False
        self.canvas.update_display()
        self.update_stats()

    def update_stats(self):
        grains, max_h = self.sim.stats_now()
        self.lbl_grains.setText(f"颗粒  {grains:,}")
        if self.last_topples:
            self.lbl_avalanche.setText(
                f"本次雪崩  {self.last_topples:,} 次  (采样 {len(self.avalanche_sizes)})"
            )
        else:
            self.lbl_avalanche.setText(
                f"本次无雪崩  (采样 {len(self.avalanche_sizes)})"
            )
        self.lbl_total.setText(f"累计崩塌  {self.total_topples:,} 次")
        self.lbl_height.setText(f"最大高度  {max_h}")
        if hasattr(self, "avalanche_plot"):
            self.avalanche_plot.set_sizes(self.avalanche_sizes)
        self.lbl_status.setText(
            f"BTW · {self.grid_size}² · "
            f"{"中心" if self.source == "center" else "随机"} · "
            f"{self.drop_rate}/帧 · {self.waves} 波"
        )

    def toggle_running(self):
        self.running = not self.running
        self.run_btn.setText("暂停" if self.running else "开始")

    def single_step(self):
        if self.source == "center":
            x = y = self.grid_size / 2.0
        else:
            x = float(np.random.randint(0, self.grid_size))
            y = float(np.random.randint(0, self.grid_size))
        self.last_topples = self.sim.measure_avalanche(x, y, self.waves)
        self.total_topples += self.last_topples
        if self.last_topples > 0:
            self.avalanche_sizes.append(self.last_topples)
            self.avalanche_sizes = self.avalanche_sizes[-5000:]
        self.canvas.update_display()
        self.update_stats()

    def _reset_stats(self):
        self.stats_sim.reset_random()
        self.last_topples = 0
        self.total_topples = 0
        self.avalanche_sizes = []
        self.pending_active = False
        self.pending_total = 0
        self.avalanche_plot.set_sizes([])

    def _refresh_after_reset(self):
        self.canvas.update_display()
        self.update_stats()

    def reset(self):
        # Reset to the application's default state: settled central pile.
        self.sim.reset_center()
        self._reset_stats()
        self._refresh_after_reset()

    def reset_random(self):
        self.sim.reset_random()
        self._reset_stats()
        self._refresh_after_reset()

    def reset_center(self):
        self.sim.reset_center()
        self._reset_stats()
        self._refresh_after_reset()

    def reset_clear(self):
        self.sim.clear()
        self._reset_stats()
        self._refresh_after_reset()

    def change_source(self, text):
        self.source = "center" if text == "中心" else "random"

    def change_rate(self, value):
        self.drop_rate = int(value)

    def change_waves(self, value):
        self.waves = int(value)

    def change_size(self, text):
        new_size = int(text)
        if new_size == self.grid_size:
            return
        self.grid_size = new_size
        self.sim = SandpileSimulation(self.grid_size)
        self.sim.reset_center()
        self.canvas.set_sim(self.sim)
        self._reset_stats()
        self._refresh_after_reset()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


MainWindow = SandpileApp


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
