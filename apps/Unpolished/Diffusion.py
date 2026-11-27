# -*- coding: utf-8 -*-
"""Two-dimensional diffusion model laboratory.

The model is deliberately small: a time-conditioned MLP predicts the noise
added to a 2-D point cloud.  Training is fast enough to run interactively on
the GPU, while DDPM/DDIM sampling shows the generated distribution and the
denoising trajectories.
"""
import math
import threading

import numpy as np

from common.lab_window import LabWindow, SquareCanvas
from common.qt_base import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPainter, QColor, QPen, QFont, QPointF, QPushButton, QTimer, QThread, Signal,
    QVBoxLayout, QWidget, Qt, run_app, LabeledSlider,
)


class DiffusionEngine:
    """Small time-conditioned noise predictor plus DDPM/DDIM samplers."""

    T = 200
    BATCH = 512

    def __init__(self, device=None):
        import torch
        from torch import nn

        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device_name = str(self.device)
        self.steps = 0
        self.last_loss = float("nan")
        self._build_schedule()
        self._build_model()
        self.reset_data("双月")

    def _build_schedule(self):
        torch = self.torch
        self.beta = torch.linspace(1e-4, 0.02, self.T, device=self.device)
        self.alpha = 1.0 - self.beta
        self.abar = torch.cumprod(self.alpha, dim=0)
        self.freq = torch.arange(8, device=self.device, dtype=torch.float32)

    def _build_model(self):
        torch = self.torch
        nn = torch.nn
        self.net = nn.Sequential(
            nn.Linear(2 + 16, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 2),
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=2e-3)

    def reset_model(self):
        self._build_model()
        self.steps = 0
        self.last_loss = float("nan")

    def _time_embedding(self, t):
        torch = self.torch
        ang = (t.float()[:, None] / self.T) * (2.0 * math.pi) * (2.0 ** self.freq)[None, :]
        return torch.cat((torch.sin(ang), torch.cos(ang)), dim=1)

    def _predict(self, x, t):
        return self.net(self.torch.cat((x, self._time_embedding(t)), dim=1))

    def reset_data(self, name):
        torch = self.torch
        n = 2048
        if name == "螺旋":
            r = torch.rand(n, device=self.device) ** 0.65
            a = r * (3.4 * math.pi)
            x = torch.stack((r * torch.cos(a), r * torch.sin(a)), dim=1)
        elif name == "棋盘":
            cells = torch.randint(0, 2, (n, 2), device=self.device).float()
            x = cells * 2.0 - 1.0
            x = x * 0.72 + (torch.rand_like(x) - 0.5) * 0.55
        elif name == "同心环":
            r = torch.where(
                torch.rand(n, device=self.device) < 0.5,
                torch.tensor(0.38, device=self.device),
                torch.tensor(0.86, device=self.device),
            )
            a = torch.rand(n, device=self.device) * 2.0 * math.pi
            x = torch.stack((r * torch.cos(a), r * torch.sin(a)), dim=1)
        else:
            t = torch.rand(n, device=self.device) * math.pi
            first = torch.stack((torch.cos(t), torch.sin(t)), dim=1)
            second = torch.stack((1.0 - torch.cos(t), 0.35 - torch.sin(t)), dim=1)
            choose = torch.rand(n, device=self.device) < 0.5
            x = torch.where(choose[:, None], first, second)
            x = x * 0.80
        x = x + torch.randn_like(x) * 0.025
        self.target = x
        self.target_np = x.detach().cpu().numpy()
        self.data_name = name

    def train(self, steps):
        torch = self.torch
        if len(self.target) == 0:
            return
        for _ in range(max(1, int(steps))):
            idx = torch.randint(0, len(self.target), (self.BATCH,), device=self.device)
            x0 = self.target[idx]
            t = torch.randint(0, self.T, (self.BATCH,), device=self.device)
            eps = torch.randn_like(x0)
            ab = self.abar[t][:, None]
            xt = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps
            pred = self._predict(xt, t)
            loss = ((pred - eps) ** 2).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            self.steps += 1
            self.last_loss = float(loss.detach().cpu())

    def sample(self, n=128, steps=80, sampler="DDIM"):
        torch = self.torch
        with torch.no_grad():
            x = torch.randn(n, 2, device=self.device)
            indices = torch.linspace(self.T - 1, 0, int(steps), device=self.device).long()
            frames = []
            save_every = max(1, len(indices) // 24)
            for k, ti in enumerate(indices):
                t = torch.full((n,), int(ti), device=self.device, dtype=torch.long)
                eps = self._predict(x, t)
                ab = self.abar[ti]
                if sampler == "DDIM":
                    x0 = (x - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)
                    if k + 1 < len(indices):
                        ab_prev = self.abar[indices[k + 1]]
                        x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1.0 - ab_prev) * eps
                    else:
                        x = x0
                else:
                    beta = self.beta[ti]
                    mean = (x - beta / torch.sqrt(1.0 - ab) * eps) / torch.sqrt(1.0 - beta)
                    if k + 1 < len(indices):
                        x = mean + torch.sqrt(beta) * torch.randn_like(x)
                    else:
                        x = mean
                if k % save_every == 0 or k == len(indices) - 1:
                    frames.append(x.detach().cpu().numpy().copy())
            return x.detach().cpu().numpy(), frames


class DiffusionCanvas(SquareCanvas):
    def __init__(self, parent=None):
        super().__init__(parent, 600)
        self.setFixedSize(600, 600)
        self.target = None
        self.frames = []
        self.frame = 0
        self.samples = None
        self.show_target = True
        self.show_trails = True
        self.show_label = True

    def set_target(self, target):
        self.target = target
        self.frames = []
        self.samples = None
        self.update()

    def set_sampling(self, frames):
        self.frames = frames
        self.frame = 0
        self.samples = frames[-1] if frames else None
        self.update()

    def set_frame(self, index):
        self.frame = max(0, min(index, len(self.frames) - 1)) if self.frames else 0
        self.samples = self.frames[self.frame] if self.frames else None
        self.update()

    def _map(self, pts):
        scale = (self.width() - 90) / 2.25
        return (self.width() * 0.5 + pts[:, 0] * scale,
                self.height() * 0.5 - pts[:, 1] * scale)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        p.setPen(QPen(QColor(198, 209, 211), 1.2))
        p.drawRect(1, 1, self.width() - 2, self.height() - 2)

        if self.show_target and self.target is not None:
            tx, ty = self._map(self.target)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(160, 178, 176, 95))
            for x, y in zip(tx, ty):
                p.drawEllipse(QPointF(x, y), 1.5, 1.5)

        if self.show_trails and self.frames:
            p.setBrush(Qt.NoBrush)
            for i in range(min(len(self.frames[0]), 96)):
                path = [self._map(f[i:i + 1]) for f in self.frames]
                p.setPen(QPen(QColor(238, 116, 88, 38), 1.0))
                last = None
                for xx, yy in path:
                    q = QPointF(xx[0], yy[0])
                    if last is not None:
                        p.drawLine(last, q)
                    last = q

        if self.samples is not None:
            sx, sy = self._map(self.samples)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(224, 82, 62, 220))
            for x, y in zip(sx, sy):
                p.drawEllipse(QPointF(x, y), 2.6, 2.6)

        if self.show_label:
            p.setPen(QColor(92, 108, 110))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(14, 22, "2D diffusion  ·  target / generated / trajectories")
        p.end()


class DiffusionApp(LabWindow):
    engine_ready = Signal(object)
    engine_failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diffusion · 二维扩散模型")
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.resize(1000, 720)
        self.setMinimumSize(1000, 720)
        self.engine = None
        self.error = None
        self._engine_thread = None
        self._loading = True
        self.canvas = DiffusionCanvas()
        self.canvas.set_target(np.zeros((0, 2)))
        self._build_ui()
        self._build_timers()
        self._set_controls_enabled(False)
        self._refresh_stats()
        self.engine_ready.connect(self._on_engine_ready)
        self.engine_failed.connect(self._on_engine_failed)
        # 窗口先出来；torch/CUDA/模型初始化（约 4 s）放到后台线程，
        # 完成前禁用训练/采样按钮并显示提示。
        QTimer.singleShot(150, self._start_engine_load)

    # ------------------------------------------------------------------
    # 引擎后台初始化
    # ------------------------------------------------------------------
    def _start_engine_load(self):
        if self._engine_thread is not None:
            return
        self._engine_thread = threading.Thread(
            target=self._load_engine_worker,
            name="diffusion-engine",
            daemon=True,
        )
        self._engine_thread.start()

    def _load_engine_worker(self):
        try:
            engine = DiffusionEngine()
        except Exception as exc:  # noqa: BLE001 - 把失败原因反馈到界面
            try:
                self.engine_failed.emit(str(exc))
            except RuntimeError:
                pass
            return
        try:
            self.engine_ready.emit(engine)
        except RuntimeError:
            pass

    def _on_engine_ready(self, engine):
        self.engine = engine
        self._loading = False
        self.canvas.set_target(engine.target_np)
        self._set_controls_enabled(True)
        self._refresh_stats()

    def _on_engine_failed(self, message):
        self.engine = None
        self._loading = False
        self.error = str(message)
        self._set_controls_enabled(False)
        self._refresh_stats()

    def _set_controls_enabled(self, enabled):
        for widget in (self.train_btn, self.reset_btn, self.sample_btn,
                       self.data_combo, self.sampler_combo,
                       self.speed_slider, self.step_slider, self.count_slider):
            widget.setEnabled(bool(enabled))

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

        data = QGroupBox("数据分布")
        df = QFormLayout(data)
        df.setContentsMargins(8, 6, 8, 6)
        df.setSpacing(5)
        self.data_combo = QComboBox()
        self.data_combo.addItems(("双月", "螺旋", "棋盘", "同心环"))
        self.data_combo.currentTextChanged.connect(self.change_data)
        df.addRow("分布", self.data_combo)
        col.addWidget(data)

        train = QGroupBox("训练")
        tv = QVBoxLayout(train)
        tv.setContentsMargins(8, 6, 8, 6)
        tv.setSpacing(5)
        buttons = QHBoxLayout()
        self.train_btn = QPushButton("训练")
        self.train_btn.clicked.connect(self.toggle_training)
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_model)
        buttons.addWidget(self.train_btn)
        buttons.addWidget(self.reset_btn)
        tv.addLayout(buttons)
        self.speed_slider = LabeledSlider(
            "每帧步数", 4, 64, 16, formatter=str, label_width=64, value_width=40
        )
        tv.addWidget(self.speed_slider)
        self.lbl_steps = QLabel("训练步数  0")
        self.lbl_loss = QLabel("loss  —")
        self.lbl_device = QLabel("设备  —")
        for w in (self.lbl_steps, self.lbl_loss, self.lbl_device):
            w.setStyleSheet("font-family: Consolas, monospace;")
            tv.addWidget(w)
        col.addWidget(train)

        sampling = QGroupBox("采样")
        sv = QFormLayout(sampling)
        sv.setContentsMargins(8, 6, 8, 6)
        sv.setSpacing(5)
        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(("DDIM", "DDPM"))
        sv.addRow("采样器", self.sampler_combo)
        self.step_slider = LabeledSlider(
            "步数", 10, 200, 80, formatter=str, label_width=40, value_width=40
        )
        sv.addRow("", self.step_slider)
        self.count_slider = LabeledSlider(
            "样本", 32, 256, 128, formatter=str, label_width=40, value_width=40
        )
        sv.addRow("", self.count_slider)
        self.sample_btn = QPushButton("采样")
        self.sample_btn.clicked.connect(self.sample)
        sv.addRow("", self.sample_btn)
        col.addWidget(sampling)

        display = QGroupBox("显示")
        dv = QVBoxLayout(display)
        dv.setContentsMargins(8, 6, 8, 6)
        dv.setSpacing(4)
        self.target_check = QCheckBox("目标分布")
        self.target_check.setChecked(True)
        self.target_check.toggled.connect(self.on_view)
        dv.addWidget(self.target_check)
        self.trail_check = QCheckBox("采样轨迹")
        self.trail_check.setChecked(True)
        self.trail_check.toggled.connect(self.on_view)
        dv.addWidget(self.trail_check)
        col.addWidget(display)

        note = QGroupBox("机制")
        nv = QVBoxLayout(note)
        nv.setContentsMargins(8, 6, 8, 6)
        txt = QLabel(
            "时间条件 MLP 预测噪声 εθ(x,t)，训练目标为真实噪声。\n"
            "DDIM 从高斯噪声逐步去噪；轨迹显示每个样本的生成路径。"
        )
        txt.setWordWrap(True)
        txt.setStyleSheet("color:#52606d; font-size:12px;")
        nv.addWidget(txt)
        col.addWidget(note)
        col.addStretch(1)

        root.addWidget(panel)
        root.addWidget(self.canvas)
        self.status = QLabel("就绪")
        self.statusBar().addWidget(self.status, 1)

    def _build_timers(self):
        self.train_timer = QTimer(self)
        self.train_timer.setInterval(16)
        self.train_timer.timeout.connect(self.train_tick)
        self.sample_timer = QTimer(self)
        self.sample_timer.setInterval(45)
        self.sample_timer.timeout.connect(self.sample_tick)

    def toggle_training(self):
        if self.engine is None:
            return
        if self.train_timer.isActive():
            self.train_timer.stop()
            self.train_btn.setText("训练")
        else:
            self.train_timer.start()
            self.train_btn.setText("暂停")

    def train_tick(self):
        if self.engine is None:
            self.train_timer.stop()
            return
        self.engine.train(self.speed_slider.value())
        self._refresh_stats()

    def reset_model(self):
        self.train_timer.stop()
        self.sample_timer.stop()
        self.train_btn.setText("训练")
        if self.engine is not None:
            self.engine.reset_model()
            self.canvas.set_target(self.engine.target_np)
        self._refresh_stats()

    def change_data(self, name):
        self.train_timer.stop()
        self.sample_timer.stop()
        self.train_btn.setText("训练")
        if self.engine is not None:
            self.engine.reset_data(name)
            self.engine.reset_model()
            self.canvas.set_target(self.engine.target_np)
        self._refresh_stats()

    def sample(self):
        if self.engine is None:
            return
        if self.train_timer.isActive():
            self.train_timer.stop()
            self.train_btn.setText("训练")
        self.sample_timer.stop()
        frames = self.engine.sample(
            self.count_slider.value(),
            self.step_slider.value(),
            self.sampler_combo.currentText(),
        )[1]
        self.canvas.set_sampling(frames)
        self.sample_timer.start()
        self.status.setText("采样中…")

    def sample_tick(self):
        next_frame = self.canvas.frame + 1
        if next_frame >= len(self.canvas.frames):
            self.sample_timer.stop()
            self.status.setText("采样完成")
            return
        self.canvas.set_frame(next_frame)

    def on_view(self):
        self.canvas.show_target = self.target_check.isChecked()
        self.canvas.show_trails = self.trail_check.isChecked()
        self.canvas.update()

    def _refresh_stats(self):
        if self.engine is None:
            self.lbl_device.setText("设备  —")
            self.lbl_steps.setText("训练步数  0")
            self.lbl_loss.setText("loss  —")
            if self.error:
                self.status.setText("PyTorch 不可用: " + self.error)
            else:
                self.status.setText("正在初始化模型（首次约 4 秒）…")
            return
        self.lbl_steps.setText(f"训练步数  {self.engine.steps:,}")
        self.lbl_loss.setText(f"loss  {self.engine.last_loss:.4f}"
                              if self.engine.steps else "loss  —")
        self.lbl_device.setText(
            "设备  " + ("CUDA" if "cuda" in self.engine.device_name else "CPU")
        )
        self.status.setText(
            f"2D diffusion · {self.engine.data_name} · "
            f"{self.engine.steps:,} steps · {self.engine.device_name}"
        )

    def closeEvent(self, event):
        self.train_timer.stop()
        self.sample_timer.stop()
        super().closeEvent(event)


MainWindow = DiffusionApp

if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
