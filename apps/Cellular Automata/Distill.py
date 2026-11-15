import sys
import os
import glob
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from common.paths import MODELS_DIR
from common.qt_base import LabeledSlider, run_app
from common.lab_window import LabWindow, SquareCanvas
from common.workers import StoppableThread

UI_LAYOUT = "two_view"

# ==================== 神经网络模型（支持两种边界） ====================
class CAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super().__init__()
        self.device = device
        self.channel_n = channel_n
        self.fc0 = nn.Linear(channel_n * 3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()
        self.fire_rate = fire_rate

        # 预计算 Sobel 核并注册为 buffer
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0
        dy = dx.T
        sobel_x = torch.tensor(dx, dtype=torch.float32, device=device).view(1,1,3,3).repeat(channel_n, 1, 1, 1)
        sobel_y = torch.tensor(dy, dtype=torch.float32, device=device).view(1,1,3,3).repeat(channel_n, 1, 1, 1)
        self.register_buffer('sobel_x_kernel', sobel_x)
        self.register_buffer('sobel_y_kernel', sobel_y)

        self.to(self.device)

    def alive(self, x, use_circular=True):
        a = x[:, 3:4, :, :]
        if use_circular:
            a = F.pad(a, (1, 1, 1, 1), mode='circular')
            return F.max_pool2d(a, kernel_size=3, stride=1) > 0.1
        return F.max_pool2d(a, kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle=0.0, use_circular=True):
        # 选择边界模式
        pad_mode = 'circular' if use_circular else 'replicate'
        if angle == 0.0:
            y1 = F.conv2d(F.pad(x, (1,1,1,1), mode=pad_mode), self.sobel_x_kernel, groups=self.channel_n)
            y2 = F.conv2d(F.pad(x, (1,1,1,1), mode=pad_mode), self.sobel_y_kernel, groups=self.channel_n)
        else:
            c = np.cos(angle * np.pi / 180)
            s = np.sin(angle * np.pi / 180)
            w1 = c * self.sobel_x_kernel - s * self.sobel_y_kernel
            w2 = s * self.sobel_x_kernel + c * self.sobel_y_kernel
            y1 = F.conv2d(F.pad(x, (1,1,1,1), mode=pad_mode), w1, groups=self.channel_n)
            y2 = F.conv2d(F.pad(x, (1,1,1,1), mode=pad_mode), w2, groups=self.channel_n)
        return torch.cat((x, y1, y2), 1)

    def update(self, x, fire_rate=None, angle=0.0, use_circular=True):
        x = x.transpose(1, 3)
        pre_life_mask = self.alive(x, use_circular)
        dx = self.perceive(x, angle, use_circular)
        dx = dx.transpose(1, 3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)
        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1], device=self.device) > fire_rate
        dx = dx * stochastic.float()
        x = x + dx.transpose(1, 3)
        post_life_mask = self.alive(x, use_circular)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1, 3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0, use_circular=True):
        for _ in range(steps):
            x = self.update(x, fire_rate, angle, use_circular)
        return x

# ==================== 样本池 ====================
class SamplePool:
    def __init__(self, x, pool_size=1024):
        self.pool = np.copy(x)
        self.pool_size = pool_size

    def sample(self, batch_size):
        idxs = np.random.choice(self.pool_size, batch_size, replace=False)
        return idxs, self.pool[idxs]

    def update(self, idxs, new_x):
        self.pool[idxs] = new_x

# ==================== 训练线程 ====================
class TrainThread(StoppableThread):
    loss_signal = Signal(float)
    state_signal = Signal(np.ndarray)
    progress_signal = Signal(int)
    epoch_signal = Signal(int, int)
    lr_signal = Signal(float)
    finished_signal = Signal()

    def __init__(self, model, target_img, grid_size, channel_n, steps_range, epochs, lr,
                 use_pool, use_damage, pool_size, batch_size, device, use_circular,
                 optimizer=None, scheduler=None, manual_lr=None):
        super().__init__()
        self.model = model
        self.target = target_img
        self.grid_size = grid_size
        self.channel_n = channel_n
        self.steps_range = steps_range
        self.epochs = epochs
        self.lr = lr
        self.use_pool = use_pool
        self.use_damage = use_damage
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.device = device
        self.use_circular = use_circular
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.manual_lr = manual_lr

    def run(self):
        H, W = self.grid_size, self.grid_size
        C = self.channel_n

        seed = np.zeros((1, H, W, C), dtype=np.float32)
        seed[0, H // 2, W // 2, 3:] = 1.0

        if self.use_pool:
            pool = SamplePool(x=np.repeat(seed, self.pool_size, axis=0), pool_size=self.pool_size)

        optimizer = self.optimizer
        scheduler = self.scheduler
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[2000], gamma=0.1
            )

        target = self.target.to(self.device)
        target_batch = target.permute(0, 2, 3, 1).expand(self.batch_size, -1, -1, -1)

        seed_torch = torch.from_numpy(seed).to(self.device)

        # 损伤掩膜用到的坐标网格只算一次
        YY, XX = torch.meshgrid(torch.arange(H, device=self.device),
                                torch.arange(W, device=self.device), indexing='ij')

        for epoch in range(1, self.epochs + 1):
            if self.stopping:
                break

            if self.use_pool:
                idxs, batch_x = pool.sample(self.batch_size)
                x0 = torch.from_numpy(batch_x.astype(np.float32)).to(self.device)
                with torch.no_grad():
                    loss_per_sample = torch.mean(
                        torch.pow(x0[:, :, :, :4] - target_batch, 2), dim=[1,2,3])
                    loss_rank = loss_per_sample.argsort(descending=True).cpu().numpy()
                x0 = x0[loss_rank]
                x0[0:1] = seed_torch

                if self.use_damage and self.batch_size > 1:
                    damage_n = min(3, self.batch_size - 1)
                    damage_masks = torch.ones(damage_n, 1, H, W, device=self.device)
                    for i in range(damage_n):
                        cx, cy = np.random.randint(10, H-10), np.random.randint(10, W-10)
                        r = np.random.randint(5, 15)
                        mask = ((XX - cx)**2 + (YY - cy)**2) <= r**2
                        damage_masks[i, 0] = (~mask).float()
                    x0[-damage_n:] *= damage_masks.permute(0, 2, 3, 1)
            else:
                x0 = seed_torch.repeat(self.batch_size, 1, 1, 1)

            steps = np.random.randint(self.steps_range[0], self.steps_range[1] + 1)
            x = self.model(x0, steps=steps, angle=0.0, use_circular=self.use_circular)

            loss = F.mse_loss(x[:, :, :, :4], target_batch)

            optimizer.zero_grad()
            loss.backward()
            # Official Distill training normalises each parameter gradient
            # tensor before Adam; without it some layers dominate and
            # convergence is much slower/noisier.
            with torch.no_grad():
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(parameter.grad.norm() + 1e-8)
            optimizer.step()
            scheduler.step()
            if self.manual_lr is not None:
                for group in optimizer.param_groups:
                    group["lr"] = float(self.manual_lr)
            self.lr_signal.emit(float(optimizer.param_groups[0]["lr"]))

            self.loss_signal.emit(loss.item())
            progress = int(epoch / self.epochs * 100)
            self.progress_signal.emit(progress)
            self.epoch_signal.emit(epoch, self.epochs)

            if self.use_pool:
                pool.update(idxs, x.detach().cpu().numpy())

            if epoch % 10 == 0 or epoch == self.epochs:
                with torch.no_grad():
                    diff = torch.pow(x[:, :, :, :4] - target_batch, 2)
                    alpha = target_batch[..., 3:4]
                    fg_loss = (diff * alpha).mean().item()
                    bg_loss = (diff * (1 - alpha)).mean().item()
                    print(f"Epoch {epoch}: 总Loss={loss.item():.4f}, 前景={fg_loss:.4f}, 背景={bg_loss:.4f}")

                    vis = x[0, :, :, :4].cpu().numpy()
                    vis = np.clip(vis, 0, 1)
                    rgba_uint8 = (vis * 255).astype(np.uint8)
                    self.state_signal.emit(rgba_uint8)

        # Keep the actual final weights.  EMA-of-batch-loss snapshots can have
        # a low short-horizon loss yet be unstable over a long rollout, which
        # made a converged preview suddenly diverge after training ended.
        print("训练结束：保留最后一轮权重")
        self.finished_signal.emit()

class TwoViewCanvas(QWidget):
    """Wide two-view canvas: current state on the left, target on the right."""

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.state_img = None
        self.target_img = None
        self.setMinimumHeight(420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)

    def set_state_image(self, qimg):
        self.state_img = qimg
        self.update()

    def set_target_image(self, qimg):
        self.target_img = qimg
        self.update()

    def _rects(self):
        side = min((self.width() - 6) // 2, self.height())
        side = max(10, side)
        x = (self.width() - (2 * side + 6)) // 2
        y = (self.height() - side) // 2
        return QRect(x, y, side, side), QRect(x + side + 6, y, side, side)

    def grid_coords(self, pos):
        left, _ = self._rects()
        content = left.adjusted(2, 2, -2, -2)
        point = pos.toPoint()
        if not content.contains(point):
            return -1, -1
        gx = int((point.x() - content.x()) / max(1, content.width()) * self.app.grid_size)
        gy = int((point.y() - content.y()) / max(1, content.height()) * self.app.grid_size)
        gx = min(max(gx, 0), self.app.grid_size - 1)
        gy = min(max(gy, 0), self.app.grid_size - 1)
        return gx, gy

    def mousePressEvent(self, event):
        self.app.mouse_press_event(event)

    def mouseMoveEvent(self, event):
        self.app.mouse_move_event(event)

    def mouseReleaseEvent(self, event):
        self.app.mouse_release_event(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        # Neural CA is a pixel grid: upscale with nearest-neighbour, not
        # bilinear smoothing.
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.fillRect(self.rect(), Qt.white)
        left, right = self._rects()
        left_content = left.adjusted(2, 2, -2, -2)
        right_content = right.adjusted(2, 2, -2, -2)
        if self.state_img is not None and not self.state_img.isNull():
            painter.drawImage(left_content, self.state_img)
        if self.target_img is not None and not self.target_img.isNull():
            painter.drawImage(right_content, self.target_img)
        # 双视图必须用可见边框把状态/目标分开；这是 two_view 规则的例外。
        painter.setPen(QPen(QColor(140, 132, 120), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(left.adjusted(0, 0, -1, -1))
        painter.drawRect(right.adjusted(0, 0, -1, -1))
        painter.end()

# ==================== 主窗口 ====================
class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural CA · 自生长")
        self.model_dir = str(MODELS_DIR)
        os.makedirs(self.model_dir, exist_ok=True)

        # 论文配方约 8000 轮；这里默认 4000 轮以缩短等待，可在轮数框调回 8000
        self.grid_size = 40
        self.channel_n = 16
        self.fire_rate = 0.5
        self.steps_range = (64, 96)
        self.epochs = 4000
        self.lr = 2e-3
        self.pool_size = 1024
        self.batch_size = 8

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CAModel(self.channel_n, self.fire_rate, self.device).to(self.device)
        self.target_tensor = None
        self.current_state = None
        self.original_image = None

        self.train_thread = None
        self.optimizer = None
        self.scheduler = None
        self.manual_lr = None
        self.sim_timer = QTimer(self)
        self.sim_timer.setInterval(25)
        self.sim_timer.timeout.connect(self.simulate_step)
        self.sim_running = False

        self.brush_size = 1
        self._brush_active = False
        self._brush_erase = False
        self._brush_gx = -1
        self._brush_gy = -1
        self.sim_steps_per_tick = 16
        self.current_angle = 0.0
        self.target_train_scale = 1.0
        self.use_circular = False
        self.reset_model = True
        self._updating_model_list = False
        self._last_frame = time.perf_counter()
        self._fps = 0.0
        self._status_tick = 0

        self.canvas = TwoViewCanvas(self)
        self.display_widget = self.canvas
        self.init_ui()
        self.reset_state()

    def init_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        data_grp = QGroupBox("数据与模型")
        data_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        data_layout = QVBoxLayout(data_grp)
        data_layout.setContentsMargins(4, 4, 4, 4)
        data_layout.setSpacing(3)
        load_row = QHBoxLayout()
        self.btn_load_img = QPushButton("加载图片")
        self.btn_load_img.clicked.connect(self.load_image)
        self.lbl_path = QLabel("未加载")
        self.lbl_path.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        load_row.addWidget(self.btn_load_img)
        load_row.addWidget(self.lbl_path, 1)
        data_layout.addLayout(load_row)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型"))
        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.refresh_model_list()
        model_row.addWidget(self.model_combo, 1)
        self.btn_save_model = QPushButton("保存")
        self.btn_save_model.clicked.connect(self.save_new_model)
        model_row.addWidget(self.btn_save_model)
        data_layout.addLayout(model_row)
        name_row = QHBoxLayout()
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("模型名称")
        name_row.addWidget(self.model_name_input, 1)
        self.btn_rename_model = QPushButton("重命名")
        self.btn_rename_model.clicked.connect(self.rename_model)
        name_row.addWidget(self.btn_rename_model)
        self.btn_delete_model = QPushButton("删除")
        self.btn_delete_model.clicked.connect(self.delete_model)
        name_row.addWidget(self.btn_delete_model)
        data_layout.addLayout(name_row)
        self.train_scale_slider = LabeledSlider(
            "训练缩放", 10, 100, 100,
            formatter=lambda v: f"{v / 100:.2f}", value_width=40,
        )
        self.train_scale_slider.valueChanged.connect(self.on_train_scale_changed)
        data_layout.addWidget(self.train_scale_slider)
        top_layout.addWidget(data_grp, 1, Qt.AlignTop)

        train_grp = QGroupBox("训练")
        train_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        train_layout = QGridLayout(train_grp)
        train_layout.setContentsMargins(4, 4, 4, 4)
        train_layout.setSpacing(3)
        train_layout.addWidget(QLabel("轮数"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(100, 50000)
        self.epochs_spin.setValue(self.epochs)
        train_layout.addWidget(self.epochs_spin, 0, 1)
        train_layout.addWidget(QLabel("学习率"), 0, 2)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.01)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(self.lr)
        train_layout.addWidget(self.lr_spin, 0, 3)
        train_layout.addWidget(QLabel("网格"), 1, 0)
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(20, 512)
        self.grid_spin.setValue(self.grid_size)
        self.grid_spin.valueChanged.connect(self.change_grid_size)
        train_layout.addWidget(self.grid_spin, 1, 1)
        chk_row = QHBoxLayout()
        self.chk_pool = QCheckBox("样本池")
        self.chk_pool.setChecked(True)
        self.chk_damage = QCheckBox("损伤")
        self.chk_damage.setChecked(True)
        self.chk_reset = QCheckBox("重新训练")
        self.chk_reset.setChecked(True)
        self.chk_reset.stateChanged.connect(
            lambda state: setattr(self, "reset_model", state == Qt.Checked)
        )
        chk_row.addWidget(self.chk_pool)
        chk_row.addWidget(self.chk_damage)
        chk_row.addWidget(self.chk_reset)
        train_layout.addLayout(chk_row, 2, 0, 1, 4)
        train_btn_row = QHBoxLayout()
        self.btn_train = QPushButton("训练")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_stop_train = QPushButton("停止")
        self.btn_stop_train.clicked.connect(self.stop_training)
        self.btn_stop_train.setEnabled(False)
        train_btn_row.addWidget(self.btn_train)
        train_btn_row.addWidget(self.btn_stop_train)
        train_layout.addLayout(train_btn_row, 3, 0, 1, 4)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(14)
        self.progress_bar.setMinimumWidth(120)
        self.progress_bar.setTextVisible(False)
        self.progress_label = QLabel("0%")
        self.epoch_label = QLabel(f"0 / {self.epochs}")
        self.loss_label = QLabel("Loss: -")
        progress_row = QHBoxLayout()
        progress_row.setSpacing(4)
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.progress_label)
        progress_row.addWidget(self.epoch_label)
        progress_row.addWidget(self.loss_label)
        train_layout.addLayout(progress_row, 4, 0, 1, 4)
        top_layout.addWidget(train_grp, 1, Qt.AlignTop)

        run_grp = QGroupBox("运行")
        run_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        run_layout = QVBoxLayout(run_grp)
        run_layout.setContentsMargins(4, 4, 4, 4)
        run_layout.setSpacing(3)
        self.speed_slider = LabeledSlider(
            "速度", 1, 100, self.sim_steps_per_tick,
            formatter=lambda v: f"{v}/帧", value_width=40,
        )
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        run_layout.addWidget(self.speed_slider)
        self.angle_slider = LabeledSlider(
            "角度", 0, 360, 0,
            formatter=lambda v: f"{v}°", value_width=40,
        )
        self.angle_slider.valueChanged.connect(self.on_angle_changed)
        run_layout.addWidget(self.angle_slider)
        self.brush_slider = LabeledSlider(
            "笔刷", 1, 20, self.brush_size, value_width=40,
        )
        self.brush_slider.valueChanged.connect(self.on_brush_changed)
        run_layout.addWidget(self.brush_slider)
        run_btn_row = QHBoxLayout()
        self.btn_run = QPushButton("运行")
        self.btn_run.clicked.connect(self.toggle_simulation)
        self.btn_reset = QPushButton("重置")
        self.btn_reset.clicked.connect(self.reset_state)
        self.btn_save_canvas = QPushButton("保存")
        self.btn_save_canvas.clicked.connect(self.save_canvas)
        self.chk_circular = QCheckBox("周期")
        self.chk_circular.setMinimumWidth(72)
        self.chk_circular.setChecked(self.use_circular)
        self.chk_circular.toggled.connect(self.on_circular_toggled)
        run_btn_row.addWidget(self.btn_run)
        run_btn_row.addWidget(self.btn_reset)
        run_btn_row.addWidget(self.btn_save_canvas)
        run_btn_row.addWidget(self.chk_circular)
        run_layout.addLayout(run_btn_row)
        top_layout.addWidget(run_grp, 1, Qt.AlignTop)

        root.addWidget(top)
        preview_note = QLabel("左：当前状态（左键种植 / 右键擦除）　右：目标")
        preview_note.setStyleSheet("color: #7a7266; font-size: 9px;")
        root.addWidget(preview_note)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.canvas, 1)
        self.setCentralWidget(central)
        self.setStyleSheet("QMainWindow { background-color: #ffffff; }")
        self.setMinimumSize(1000, 700)
        self.resize(1040, 740)

        status = self.statusBar()
        status.setSizeGripEnabled(False)
        self.state_label = QLabel("就绪")
        self.perf_label = QLabel("0 fps")
        status.addWidget(self.state_label, 1)
        status.addPermanentWidget(self.perf_label)

        self.model_combo.currentIndexChanged.connect(self.on_model_selected)
        self.lr_spin.valueChanged.connect(self.on_lr_input)

        self.update_state_display()
        self.update_target_display()

    # ---------- 模型管理 ----------
    def delete_model(self):
        current_name = self.model_combo.currentText()
        if current_name == "空模型":
            QMessageBox.warning(self, "警告", "不能删除空模型")
            return

        reply = QMessageBox.question(self, "确认删除", f"确定要永久删除模型 {current_name} 吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        path = os.path.join(self.model_dir, current_name)
        try:
            os.remove(path)
            QMessageBox.information(self, "提示", f"模型 {current_name} 已删除")
            self.refresh_model_list()
            self.model_combo.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除失败:\n{e}")

    def rename_model(self):
        current_name = self.model_combo.currentText()
        if current_name == "空模型":
            QMessageBox.warning(self, "警告", "不能重命名空模型")
            return

        new_name, ok = QInputDialog.getText(self, "重命名模型", "请输入新名称（不含扩展名）:", text=current_name.replace(".pth", ""))
        if not ok or not new_name.strip():
            return

        new_name = new_name.strip()
        if not new_name.endswith(".pth"):
            new_name += ".pth"

        if new_name == current_name:
            return

        old_path = os.path.join(self.model_dir, current_name)
        new_path = os.path.join(self.model_dir, new_name)
        if os.path.exists(new_path):
            QMessageBox.warning(self, "警告", f"文件 {new_name} 已存在，请使用其他名称")
            return

        try:
            os.rename(old_path, new_path)
            QMessageBox.information(self, "提示", f"模型已重命名为 {new_name}")
            self.refresh_model_list()
            idx = self.model_combo.findText(new_name)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重命名失败:\n{e}")

    def refresh_model_list(self):
        self._updating_model_list = True
        self.model_combo.clear()
        self.model_combo.addItem("空模型")
        files = glob.glob(os.path.join(self.model_dir, "*.pth"))
        names = [os.path.basename(f) for f in files]
        if names:
            self.model_combo.addItems(names)
        self._updating_model_list = False

    def on_model_selected(self, index):
        if self._updating_model_list:
            return
        name = self.model_combo.currentText()
        if name == "空模型":
            return
        self.load_selected_model(name)

    def load_selected_model(self, name):
        path = os.path.join(self.model_dir, name)
        try:
            self.model = CAModel(self.channel_n, self.fire_rate, self.device).to(self.device)
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            self.optimizer = None
            self.scheduler = None
            self.manual_lr = None
            QMessageBox.information(self, "提示", f"模型已从 {name} 加载")
            self.reset_state()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败:\n{e}")

    def save_new_model(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "没有可保存的模型")
            return
        name = self.model_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入名称")
            return
        if not name.endswith(".pth"):
            name += ".pth"
        path = os.path.join(self.model_dir, name)
        try:
            torch.save(self.model.state_dict(), path)
            QMessageBox.information(self, "提示", f"模型已保存为 {name}")
            self.refresh_model_list()
            self.model_combo.setCurrentIndex(0)
            self.model_name_input.clear()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败:\n{e}")

    # ---------- 滑块回调 ----------
    def on_speed_changed(self, value):
        self.sim_steps_per_tick = value

    def on_angle_changed(self, value):
        self.current_angle = float(value)

    def on_brush_changed(self, value):
        self.brush_size = value

    def on_circular_toggled(self, checked):
        self.use_circular = bool(checked)
        self.state_label.setText("周期边界" if checked else "就绪")

    def on_train_scale_changed(self, value):
        self.target_train_scale = value / 100.0
        if self.original_image is not None:
            self.recompute_target()

    def recompute_target(self):
        if self.original_image is None:
            return
        target_pixels = max(1, int(self.grid_size * self.target_train_scale))
        img_resized = self.original_image.resize((target_pixels, target_pixels), Image.LANCZOS)
        full_img = Image.new("RGBA", (self.grid_size, self.grid_size), (0, 0, 0, 0))
        offset_x = (self.grid_size - target_pixels) // 2
        offset_y = (self.grid_size - target_pixels) // 2
        full_img.paste(img_resized, (offset_x, offset_y))
        rgba = np.array(full_img).astype(np.float32) / 255.0
        if (rgba[:, :, 3] == 1.0).all():
            gray = np.mean(rgba[:, :, :3], axis=-1)
            rgba[:, :, 3] = np.where(gray > 0.9, 0.0, 1.0).astype(np.float32)
        target = rgba.transpose(2, 0, 1)[np.newaxis, ...]
        self.target_tensor = torch.from_numpy(target.astype(np.float32)).to(self.device)
        self.update_target_display()

    def change_grid_size(self, new_size):
        if new_size == self.grid_size:
            return
        if new_size > 200:
            reply = QMessageBox.question(self, "注意", "大网格可能消耗大量显存，确定继续？",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                self.grid_spin.setValue(self.grid_size)
                return
        self.grid_size = new_size
        # The CAModel is fully convolutional, so weights survive a grid
        # change; however a fresh module is needed to discard any stale
        # training/cache state.  Clone the weights first, then rebuild the
        # module exactly like the "switch model" workaround did.
        saved_state = None
        if self.model is not None:
            saved_state = {
                key: value.detach().clone()
                for key, value in self.model.state_dict().items()
            }
        self.model = CAModel(self.channel_n, self.fire_rate, self.device).to(self.device)
        if saved_state is not None:
            self.model.load_state_dict(saved_state)
        # Optimizer/scheduler state is tied to the previous training run, but
        # the network weights are not.  Drop only the optimizer state.
        self.optimizer = None
        self.scheduler = None
        self.manual_lr = None
        self.reset_state()
        if self.original_image is not None:
            self.recompute_target()
        else:
            self.state_label.setText("网格已更改 · 模型权重保留")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开图片", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            try:
                self.original_image = Image.open(path).convert("RGBA")
                self.lbl_path.setText("已加载")
                self.recompute_target()
                self.reset_state()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图片失败:\n{e}")

    def reset_state(self):
        H, W = self.grid_size, self.grid_size
        C = self.channel_n
        state = np.zeros((1, H, W, C), dtype=np.float32)
        state[0, H // 2, W // 2, 3:] = 1.0
        self.current_state = torch.from_numpy(state).to(self.device)
        self.update_state_display()

    def simulate_step(self):
        if self.current_state is None:
            return
        # Use the live training model so the running simulation reflects
        # weights learned during training.  inference_mode never builds a
        # graph, and the 40x40 update is safe against the tiny optimizer step.
        with torch.inference_mode():
            self.current_state = self.model(
                self.current_state,
                steps=self.sim_steps_per_tick,
                angle=self.current_angle,
                use_circular=self.use_circular,
            )
            if not torch.isfinite(self.current_state).all():
                self.reset_state()
                self.state_label.setText("状态异常，已重置")
                return
            # Periodic boundaries can amplify the visible RGBA channels past
            # 1.0; without this the display clips to pure white.
            self.current_state[..., :4].clamp_(0.0, 1.0)
            # Re-apply the brush after the NCA update while the button is
            # held.  Without this a loaded model overwrites the seed every
            # tick, which looked like a wrong brush position/size.
            self._apply_brush_now()
            visible = self.current_state[..., :4]
            if float(visible.mean()) > 0.985 and float(visible.std()) < 0.015:
                self.reset_state()
                if self.use_circular:
                    self.state_label.setText("周期边界饱和，已重置状态（周期仍开启）")
                else:
                    self.state_label.setText("状态饱和，已重置")
                return
        self.update_state_display()
        now = time.perf_counter()
        interval = now - self._last_frame
        self._last_frame = now
        if interval > 0:
            self._fps = 0.85 * self._fps + 0.15 / max(interval, 1e-4)
        self._status_tick += 1
        if self._status_tick % 3 == 0:
            self.perf_label.setText(
                f"{self._fps:.0f} fps · {self.sim_steps_per_tick} 步/帧"
            )

    def toggle_simulation(self):
        if self.sim_running:
            self.sim_timer.stop()
            self.sim_running = False
            self.btn_run.setText("运行")
            self.state_label.setText("暂停")
        else:
            self._last_frame = time.perf_counter()
            self.sim_timer.start(25)
            self.sim_running = True
            self.btn_run.setText("暂停")
            self.state_label.setText("运行中")

    # ---------- 鼠标交互 ----------
    def save_canvas(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 Distill 画面", "distill.png", "PNG (*.png)"
        )
        if not path:
            return
        if self.canvas.save_image(path):
            self.state_label.setText("已保存画面")

    def get_grid_coords(self, event: QMouseEvent):
        return self.canvas.grid_coords(event.position())

    def mouse_press_event(self, e):
        gx, gy = self.get_grid_coords(e)
        if gx < 0 or gy < 0:
            return
        if e.button() == Qt.LeftButton:
            self._brush_active = True
            self._brush_erase = False
            self._brush_gx, self._brush_gy = gx, gy
            self.plant_seed(gx, gy)
        elif e.button() == Qt.RightButton:
            self._brush_active = True
            self._brush_erase = True
            self._brush_gx, self._brush_gy = gx, gy
            self.erase_at(gx, gy)
        self.update_state_display()

    def mouse_move_event(self, e):
        if not self._brush_active:
            return
        gx, gy = self.get_grid_coords(e)
        if gx < 0 or gy < 0:
            return
        self._brush_gx, self._brush_gy = gx, gy
        if self._brush_erase:
            self.erase_at(gx, gy)
        else:
            self.plant_seed(gx, gy)
        self.update_state_display()

    def mouse_release_event(self, e):
        self._brush_active = False

    def _apply_brush_now(self):
        if not self._brush_active or self._brush_gx < 0:
            return
        if self._brush_erase:
            self.erase_at(self._brush_gx, self._brush_gy)
        else:
            self.plant_seed(self._brush_gx, self._brush_gy)

    def plant_seed(self, cx, cy):
        if self.current_state is None:
            return
        r = max(0, self.brush_size - 1)
        H, W = self.grid_size, self.grid_size
        with torch.no_grad():
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H and (dx*dx + dy*dy) <= r*r:
                        self.current_state[0, ny, nx, 3:] = 1.0
                        self.current_state[0, ny, nx, :3] = 0.0

    def erase_at(self, cx, cy):
        if self.current_state is None:
            return
        r = max(0, self.brush_size - 1)
        H, W = self.grid_size, self.grid_size
        with torch.no_grad():
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H and (dx*dx + dy*dy) <= r*r:
                        self.current_state[0, ny, nx, :] = 0.0

    # ---------- 显示接口 ----------
    def rgba_to_qimage(self, rgba_uint8):
        h, w, _ = rgba_uint8.shape
        rgb = rgba_uint8[:, :, :3].astype(np.float32) / 255.0
        alpha = rgba_uint8[:, :, 3:].astype(np.float32) / 255.0
        bg = np.ones_like(rgb)
        comp = rgb * alpha + bg * (1 - alpha)
        comp_uint8 = (np.clip(comp, 0, 1) * 255).astype(np.uint8)
        qimg = QImage(comp_uint8.data.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        return qimg.copy()

    def update_state_display(self):
        if self.current_state is None:
            return
        rgba = self.current_state[0, :, :, :4].cpu().numpy()
        rgba = np.clip(rgba, 0, 1)
        rgba_uint8 = (rgba * 255).astype(np.uint8)
        qimg = self.rgba_to_qimage(rgba_uint8)
        self.display_widget.set_state_image(qimg)

    def update_target_display(self):
        if self.target_tensor is None:
            self.display_widget.set_target_image(QImage())
            return
        rgba = self.target_tensor[0].permute(1, 2, 0).cpu().numpy()
        rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        qimg = self.rgba_to_qimage(rgba_uint8)
        self.display_widget.set_target_image(qimg)

    # ---------- 训练控制 ----------
    def start_training(self):
        if self.target_tensor is None:
            QMessageBox.warning(self, "警告", "请先加载图片或调整目标缩放")
            return
        if self.train_thread and self.train_thread.isRunning():
            QMessageBox.warning(self, "警告", "训练正在进行中")
            return

        self.epochs = self.epochs_spin.value()
        self.lr = self.lr_spin.value()
        use_pool = self.chk_pool.isChecked()
        use_damage = self.chk_damage.isChecked()

        continuing = (
            not self.reset_model
            and self.model is not None
            and self.optimizer is not None
        )
        if self.reset_model or self.model is None:
            self.model = CAModel(self.channel_n, self.fire_rate, self.device).to(self.device)
            self.manual_lr = None
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[2000], gamma=0.1
            )
        elif self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[2000], gamma=0.1
            )
        elif continuing:
            # Continue from the decayed LR.  The spin box may lower it, but
            # never reset it back to the original 2e-3.
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            for group in self.optimizer.param_groups:
                group["lr"] = min(current_lr, float(self.lr))
            self.state_label.setText("继续训练 · 沿用优化器")
        elif self.scheduler is None:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[2000], gamma=0.1
            )

        self.train_thread = TrainThread(
            self.model, self.target_tensor, self.grid_size, self.channel_n,
            self.steps_range, self.epochs, self.lr,
            use_pool, use_damage, self.pool_size, self.batch_size, self.device,
            self.use_circular,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            manual_lr=self.manual_lr,
        )
        self.train_thread.loss_signal.connect(self.on_loss)
        self.train_thread.state_signal.connect(self.on_train_state)
        self.train_thread.progress_signal.connect(self.on_progress)
        self.train_thread.epoch_signal.connect(self.on_epoch)
        self.train_thread.lr_signal.connect(self.on_lr)
        self.train_thread.finished_signal.connect(self.on_train_done)

        self.btn_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0%")
        self.epoch_label.setText(f"0 / {self.epochs}")
        self.lr = float(self.optimizer.param_groups[0]["lr"])
        blocker = QSignalBlocker(self.lr_spin)
        self.lr_spin.setValue(self.lr)
        del blocker
        self.state_label.setText("训练中 · 可运行")
        self.train_thread.start()

    def stop_training(self):
        if self.train_thread:
            self.train_thread.stop()
            self.btn_stop_train.setEnabled(False)

    def closeEvent(self, event):
        self.sim_timer.stop()
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.stop()
            if not self.train_thread.wait(3000):
                self.train_thread.terminate()
                self.train_thread.wait(1000)
        event.accept()
        super().closeEvent(event)
        app = QApplication.instance()
        if app is not None:
            app.quit()
        # If a CUDA training step refuses to terminate, make sure the child
        # process really dies so the launcher stops reporting "already running".
        if self.train_thread and self.train_thread.isRunning():
            os._exit(0)

    def on_loss(self, loss):
        self.loss_label.setText(f"Loss: {loss:.4f}")

    def on_train_state(self, rgba_uint8):
        if self.sim_running:
            return
        qimg = self.rgba_to_qimage(rgba_uint8)
        self.display_widget.set_state_image(qimg)

    def on_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"{value}%")

    def on_epoch(self, epoch, total):
        self.epoch_label.setText(f"{epoch} / {total}")

    def on_lr(self, value):
        self.lr = float(value)
        blocker = QSignalBlocker(self.lr_spin)
        self.lr_spin.setValue(self.lr)
        del blocker

    def on_lr_input(self, value):
        self.lr = float(value)
        self.manual_lr = self.lr
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr
        if self.train_thread is not None and self.train_thread.isRunning():
            self.train_thread.manual_lr = self.lr

    def on_train_done(self):
        self.btn_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("100%")
        self.epoch_label.setText(f"{self.epochs} / {self.epochs}")
        self.state_label.setText("训练完成 · 保留最终权重")
        self.reset_state()
        QMessageBox.information(
            self, "提示",
            "训练完成！已保留最后一轮权重。画布已重置为单粒种子，"
            "点「运行」即可观察生长。",
        )

if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))