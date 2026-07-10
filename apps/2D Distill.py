import sys
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtOpenGLWidgets import *

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

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

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
        pre_life_mask = self.alive(x)
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
        post_life_mask = self.alive(x)
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
class TrainThread(QThread):
    loss_signal = Signal(float)
    state_signal = Signal(np.ndarray)
    progress_signal = Signal(int)
    finished_signal = Signal()

    def __init__(self, model, target_img, grid_size, channel_n, steps_range, epochs, lr,
                 use_pool, use_damage, pool_size, batch_size, device, use_circular):
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
        self._is_running = True

    def run(self):
        H, W = self.grid_size, self.grid_size
        C = self.channel_n

        seed = np.zeros((1, H, W, C), dtype=np.float32)
        seed[0, H // 2, W // 2, 3:] = 1.0

        if self.use_pool:
            pool = SamplePool(x=np.repeat(seed, self.pool_size, axis=0), pool_size=self.pool_size)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.5))
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

        target = self.target.to(self.device)
        target_batch = target.permute(0, 2, 3, 1).expand(self.batch_size, -1, -1, -1)

        seed_torch = torch.from_numpy(seed).to(self.device)

        for epoch in range(1, self.epochs + 1):
            if not self._is_running:
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
                        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                        mask = ((X - cx)**2 + (Y - cy)**2) <= r**2
                        damage_masks[i, 0] = ~mask
                    x0[-damage_n:] *= damage_masks.permute(0, 2, 3, 1)
            else:
                x0 = seed_torch.repeat(self.batch_size, 1, 1, 1)

            steps = np.random.randint(self.steps_range[0], self.steps_range[1] + 1)
            x = self.model(x0, steps=steps, angle=0.0, use_circular=self.use_circular)

            loss = F.mse_loss(x[:, :, :, :4], target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.loss_signal.emit(loss.item())
            progress = int(epoch / self.epochs * 100)
            self.progress_signal.emit(progress)

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

        self.finished_signal.emit()

    def stop(self):
        self._is_running = False

# ==================== OpenGL 显示组件（1px 边框，始终显示） ====================
class GLDisplayWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state_img = None
        self.target_img = None
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        fmt = QSurfaceFormat()
        fmt.setSamples(0)
        self.setFormat(fmt)

    def set_state_image(self, qimg):
        self.state_img = qimg
        self.update()

    def set_target_image(self, qimg):
        self.target_img = qimg
        self.update()

    def paintGL(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        total_w = self.width()
        total_h = self.height()
        spacing = 6
        avail_w = (total_w - spacing) // 2
        side = min(avail_w, total_h)
        if side < 10:
            side = 10
        left_x = (avail_w - side) // 2
        left_y = (total_h - side) // 2
        left_rect = QRect(left_x, left_y, side, side)
        right_x = avail_w + spacing + (avail_w - side) // 2
        right_y = left_y
        right_rect = QRect(right_x, right_y, side, side)

        # 白色背景
        painter.fillRect(self.rect(), Qt.white)

        # 绘制图像
        if self.state_img and not self.state_img.isNull():
            painter.drawImage(left_rect, self.state_img)
        if self.target_img and not self.target_img.isNull():
            painter.drawImage(right_rect, self.target_img)

        # 绘制 1px 黑色边框
        pen = QPen(Qt.black, 1)
        painter.setPen(pen)
        painter.drawRect(left_rect)
        painter.drawRect(right_rect)

        painter.end()

# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural CA - 自生长修复 (OpenGL)")
        self.setMinimumSize(1000, 700)

        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

        # 恢复默认网格大小为60，训练更稳定
        self.grid_size = 60
        self.channel_n = 16
        self.fire_rate = 0.5
        self.steps_range = (64, 96)
        self.epochs = 2000
        self.lr = 2e-3
        self.pool_size = 1024
        self.batch_size = 8

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CAModel(self.channel_n, self.fire_rate, self.device).to(self.device)
        self.target_tensor = None
        self.current_state = None
        self.original_image = None

        self.train_thread = None
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.simulate_step)
        self.sim_running = False

        self.brush_size = 1
        self.sim_steps_per_tick = 24
        self.current_angle = 0.0
        self.target_train_scale = 1.0
        self.use_circular = True    # 默认使用周期边界

        self.reset_model = True
        self._updating_model_list = False

        self.init_ui()
        self.reset_state()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        # 数据与模型组
        data_grp = QGroupBox("数据与模型")
        data_layout = QVBoxLayout(data_grp)
        data_layout.setSpacing(2)

        img_row = QHBoxLayout()
        self.btn_load_img = QPushButton("加载图片")
        self.btn_load_img.clicked.connect(self.load_image)
        self.lbl_path = QLabel("未加载")
        self.lbl_path.setFixedWidth(50)
        self.lbl_path.setAlignment(Qt.AlignCenter)
        img_row.addWidget(self.btn_load_img)
        img_row.addWidget(self.lbl_path)
        img_row.addStretch(0)
        data_layout.addLayout(img_row)

        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.refresh_model_list()
        self.model_combo.currentIndexChanged.connect(self.on_model_selected)
        model_row.addWidget(QLabel("模型:"))
        model_row.addWidget(self.model_combo, 1)

        self.btn_rename_model = QPushButton("重命名")
        self.btn_rename_model.clicked.connect(self.rename_model)
        model_row.addWidget(self.btn_rename_model)

        self.btn_delete_model = QPushButton("删除")
        self.btn_delete_model.clicked.connect(self.delete_model)
        model_row.addWidget(self.btn_delete_model)

        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("名称")
        self.model_name_input.setMinimumWidth(50)
        model_row.addWidget(self.model_name_input)

        self.btn_save_model = QPushButton("保存")
        self.btn_save_model.clicked.connect(self.save_new_model)
        model_row.addWidget(self.btn_save_model)
        data_layout.addLayout(model_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("训练缩放:"))
        self.train_scale_slider = QSlider(Qt.Horizontal)
        self.train_scale_slider.setRange(10, 100)
        self.train_scale_slider.setValue(100)
        self.train_scale_label = QLabel("1.00")
        scale_row.addWidget(self.train_scale_slider)
        scale_row.addWidget(self.train_scale_label)
        data_layout.addLayout(scale_row)
        self.train_scale_slider.valueChanged.connect(self.on_train_scale_changed)

        top_layout.addWidget(data_grp, 1)

        # 训练组
        train_grp = QGroupBox("训练")
        train_layout = QGridLayout(train_grp)
        train_layout.setSpacing(2)

        train_layout.addWidget(QLabel("训练轮数:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(100, 50000)
        self.epochs_spin.setValue(self.epochs)
        train_layout.addWidget(self.epochs_spin, 0, 1)

        train_layout.addWidget(QLabel("学习率:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.01)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(self.lr)
        train_layout.addWidget(self.lr_spin, 1, 1)

        train_layout.addWidget(QLabel("网格像素:"), 2, 0)
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(20, 512)
        self.grid_spin.setValue(self.grid_size)
        self.grid_spin.valueChanged.connect(self.change_grid_size)
        train_layout.addWidget(self.grid_spin, 2, 1)

        chk_row = QHBoxLayout()
        self.chk_pool = QCheckBox("样本池")
        self.chk_pool.setChecked(True)
        self.chk_damage = QCheckBox("损伤")
        self.chk_damage.setChecked(True)
        self.chk_reset = QCheckBox("重新训练")
        self.chk_reset.setChecked(True)
        self.chk_reset.stateChanged.connect(lambda state: setattr(self, 'reset_model', state == Qt.Checked))
        chk_row.addWidget(self.chk_pool)
        chk_row.addWidget(self.chk_damage)
        chk_row.addWidget(self.chk_reset)
        train_layout.addLayout(chk_row, 3, 0, 1, 2)

        btn_train_box = QHBoxLayout()
        self.btn_train = QPushButton("训练")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_stop_train = QPushButton("停止")
        self.btn_stop_train.clicked.connect(self.stop_training)
        self.btn_stop_train.setEnabled(False)
        btn_train_box.addWidget(self.btn_train)
        btn_train_box.addWidget(self.btn_stop_train)
        train_layout.addLayout(btn_train_box, 4, 0, 1, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(14)
        self.loss_label = QLabel("Loss: -")
        status_row = QHBoxLayout()
        status_row.addWidget(self.progress_bar)
        status_row.addWidget(self.loss_label)
        train_layout.addLayout(status_row, 5, 0, 1, 2)

        top_layout.addWidget(train_grp, 2)

        # 运行组
        run_grp = QGroupBox("运行")
        run_layout = QGridLayout(run_grp)
        run_layout.setSpacing(2)

        run_layout.addWidget(QLabel("速度(步/帧):"), 0, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(self.sim_steps_per_tick)
        self.speed_label = QLabel(str(self.sim_steps_per_tick))
        speed_row = QHBoxLayout()
        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(self.speed_label)
        run_layout.addLayout(speed_row, 0, 1)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)

        run_layout.addWidget(QLabel("旋转角度:"), 1, 0)
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(0, 360)
        self.angle_slider.setValue(0)
        self.angle_label = QLabel("0°")
        angle_row = QHBoxLayout()
        angle_row.addWidget(self.angle_slider)
        angle_row.addWidget(self.angle_label)
        run_layout.addLayout(angle_row, 1, 1)
        self.angle_slider.valueChanged.connect(self.on_angle_changed)

        run_layout.addWidget(QLabel("笔刷:"), 2, 0)
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 10)
        self.brush_slider.setValue(self.brush_size)
        self.brush_label = QLabel(str(self.brush_size))
        brush_row = QHBoxLayout()
        brush_row.addWidget(self.brush_slider)
        brush_row.addWidget(self.brush_label)
        run_layout.addLayout(brush_row, 2, 1)
        self.brush_slider.valueChanged.connect(self.on_brush_changed)

        # 添加周期边界复选框
        self.chk_circular = QCheckBox("周期边界")
        self.chk_circular.setChecked(self.use_circular)
        self.chk_circular.stateChanged.connect(
            lambda state: setattr(self, 'use_circular', state == Qt.Checked))
        btn_run_box = QHBoxLayout()
        self.btn_run = QPushButton("运行")
        self.btn_run.clicked.connect(self.toggle_simulation)
        self.btn_reset = QPushButton("重置")
        self.btn_reset.clicked.connect(self.reset_state)
        btn_run_box.addWidget(self.btn_run)
        btn_run_box.addWidget(self.btn_reset)
        btn_run_box.addWidget(self.chk_circular)
        run_layout.addLayout(btn_run_box, 3, 0, 1, 2)

        top_layout.addWidget(run_grp, 2)
        main_layout.addWidget(top_widget)

        # OpenGL 显示组件
        self.display_widget = GLDisplayWidget()
        self.display_widget.mousePressEvent = self.mouse_press_event
        self.display_widget.mouseMoveEvent = self.mouse_move_event
        self.display_widget.mouseReleaseEvent = self.mouse_release_event
        main_layout.addWidget(self.display_widget, 1)

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
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
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
        self.speed_label.setText(str(value))

    def on_angle_changed(self, value):
        self.current_angle = float(value)
        self.angle_label.setText(f"{value}°")

    def on_brush_changed(self, value):
        self.brush_size = value
        self.brush_label.setText(str(value))

    def on_train_scale_changed(self, value):
        self.target_train_scale = value / 100.0
        self.train_scale_label.setText(f"{self.target_train_scale:.2f}")
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
        self.model = CAModel(self.channel_n, self.fire_rate, self.device).to(self.device)
        self.reset_state()
        if self.original_image is not None:
            self.recompute_target()
        else:
            QMessageBox.information(self, "提示", "网格大小已更改，模型已重置。")

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
        with torch.no_grad():
            for _ in range(self.sim_steps_per_tick):
                self.current_state = self.model.update(
                    self.current_state,
                    angle=self.current_angle,
                    use_circular=self.use_circular
                )
        self.update_state_display()

    def toggle_simulation(self):
        if self.sim_running:
            self.sim_timer.stop()
            self.sim_running = False
            self.btn_run.setText("运行")
        else:
            self.sim_timer.start(80)
            self.sim_running = True
            self.btn_run.setText("暂停")

    # ---------- 鼠标交互 ----------
    def get_grid_coords(self, event: QMouseEvent):
        dw = self.display_widget
        total_w = dw.width()
        total_h = dw.height()
        spacing = 6
        avail_w = (total_w - spacing) // 2
        side = min(avail_w, total_h)
        left_x = (avail_w - side) // 2
        left_y = (total_h - side) // 2
        mx = event.position().x()
        my = event.position().y()
        rel_x = mx - left_x
        rel_y = my - left_y
        if rel_x < 0 or rel_x >= side or rel_y < 0 or rel_y >= side:
            return -1, -1
        gx = int(rel_x / side * self.grid_size)
        gy = int(rel_y / side * self.grid_size)
        return gx, gy

    def mouse_press_event(self, e):
        gx, gy = self.get_grid_coords(e)
        if gx < 0 or gy < 0:
            return
        if e.button() == Qt.LeftButton:
            self.plant_seed(gx, gy)
        elif e.button() == Qt.RightButton:
            self.erase_at(gx, gy)
        self.update_state_display()

    def mouse_move_event(self, e):
        gx, gy = self.get_grid_coords(e)
        if gx < 0 or gy < 0:
            return
        if e.buttons() & Qt.LeftButton:
            self.plant_seed(gx, gy)
        elif e.buttons() & Qt.RightButton:
            self.erase_at(gx, gy)
        self.update_state_display()

    def mouse_release_event(self, e):
        pass

    def plant_seed(self, cx, cy):
        if self.current_state is None:
            return
        r = self.brush_size // 2
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
        r = self.brush_size // 2
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

        if self.reset_model or self.model is None:
            self.model = CAModel(self.channel_n, self.fire_rate, self.device).to(self.device)

        self.train_thread = TrainThread(
            self.model, self.target_tensor, self.grid_size, self.channel_n,
            self.steps_range, self.epochs, self.lr,
            use_pool, use_damage, self.pool_size, self.batch_size, self.device,
            self.use_circular
        )
        self.train_thread.loss_signal.connect(self.on_loss)
        self.train_thread.state_signal.connect(self.on_train_state)
        self.train_thread.progress_signal.connect(self.progress_bar.setValue)
        self.train_thread.finished_signal.connect(self.on_train_done)

        self.btn_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.progress_bar.setValue(0)
        self.train_thread.start()

    def stop_training(self):
        if self.train_thread:
            self.train_thread.stop()
            self.btn_stop_train.setEnabled(False)

    def on_loss(self, loss):
        self.loss_label.setText(f"Loss: {loss:.4f}")

    def on_train_state(self, rgba_uint8):
        qimg = self.rgba_to_qimage(rgba_uint8)
        self.display_widget.set_state_image(qimg)

    def on_train_done(self):
        self.btn_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "提示", "训练完成！")

if __name__ == "__main__":
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())