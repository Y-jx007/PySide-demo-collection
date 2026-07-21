import sys
import math
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QProgressBar, QTabWidget
)
from PySide6.QtCore import Qt, QTimer, QRectF, QThread, Signal
from PySide6.QtGui import QPainter, QPen, QColor

# ==================== 太阳-木星 CRTBP 模型 (旋转坐标系) ====================
mu = 0.000953875   # 木星真实质量比

def acceleration_crtbp(x, y, vx, vy):
    r1 = math.sqrt((x + mu)**2 + y**2)
    r2 = math.sqrt((x - 1 + mu)**2 + y**2)
    ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
    return ax, ay

def rk4_step(x, y, vx, vy, dt):
    def f(state):
        x, y, vx, vy = state
        ax, ay = acceleration_crtbp(x, y, vx, vy)
        return np.array([vx, vy, ax, ay])
    state = np.array([x, y, vx, vy])
    k1 = f(state)
    k2 = f(state + 0.5*dt*k1)
    k3 = f(state + 0.5*dt*k2)
    k4 = f(state + dt*k3)
    new_state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return new_state[0], new_state[1], new_state[2], new_state[3]

# ==================== 标准映射 (Chirikov-Taylor) ====================
def standard_map(q, p, K):
    p_new = (p + (K/(2*math.pi)) * np.sin(2*math.pi*q)) % 1.0
    q_new = (q + p_new) % 1.0
    return q_new, p_new

# ==================== Poincaré 截面计算线程 (修复版) ====================
class PoincareThread(QThread):
    new_points = Signal(list)
    progress = Signal(int)

    def __init__(self, C, x_range, num_orbits=30, dt=0.1, max_steps=5000):
        super().__init__()
        self.C = C
        self.x_range = x_range
        self.num_orbits = num_orbits
        self.dt = dt
        self.max_steps = max_steps
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        xs = np.linspace(self.x_range[0], self.x_range[1], self.num_orbits)
        for i, x0 in enumerate(xs):
            if not self._is_running:
                break
            # 雅可比积分计算 vy0
            r1 = abs(x0 + mu)
            r2 = abs(x0 - 1 + mu)
            Omega0 = 0.5*x0**2 + (1-mu)/r1 + mu/r2
            v_sq = 2*Omega0 - self.C
            if v_sq <= 0:
                continue
            vy0 = math.sqrt(v_sq)

            # 关键修复：初始 y 设为微小负值，确保能产生从负到正的穿越
            y0 = -0.001
            x, y, vx, vy = x0, y0, 0.0, vy0
            prev_y = y0
            prev_x, prev_vx = x0, 0.0

            points = []
            for step in range(self.max_steps):
                x, y, vx, vy = rk4_step(x, y, vx, vy, self.dt)
                # 检测 y 从负到正且 vy>0（保证向上穿过）
                if prev_y < 0 and y >= 0 and vy > 0:
                    if y - prev_y != 0:
                        frac = -prev_y / (y - prev_y)
                    else:
                        frac = 0.0
                    x_sec = prev_x + frac * (x - prev_x)
                    vx_sec = prev_vx + frac * (vx - prev_vx)
                    points.append((x_sec, vx_sec))
                prev_x, prev_y, prev_vx = x, y, vx

            if points:
                self.new_points.emit(points)
            self.progress.emit(int((i+1)/len(xs)*100))
        self.progress.emit(100)

# ==================== 标准映射线程 ====================
class StandardMapThread(QThread):
    new_points = Signal(list)
    progress = Signal(int)

    def __init__(self, K, num_orbits=30, steps=500):
        super().__init__()
        self.K = K
        self.num_orbits = num_orbits
        self.steps = steps
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        q0 = np.linspace(0, 1, self.num_orbits, endpoint=False)
        p0 = np.zeros_like(q0)
        for i in range(self.num_orbits):
            if not self._is_running:
                break
            q, p = q0[i], p0[i]
            traj = [(q, p)]
            for _ in range(self.steps):
                q, p = standard_map(q, p, self.K)
                traj.append((q, p))
            self.new_points.emit(traj)
            self.progress.emit(int((i+1)/self.num_orbits*100))
        self.progress.emit(100)

# ==================== 截面画布 ====================
class SectionCanvas(QWidget):
    def __init__(self, parent=None, title="", xlabel="x", ylabel="vx"):
        super().__init__(parent)
        self.setMinimumSize(500, 500)
        self.setSizePolicy(self.sizePolicy().horizontalPolicy().Expanding,
                           self.sizePolicy().verticalPolicy().Expanding)
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.points = []            # 存储所有点
        self.x_range = [0.5, 1.5]
        self.y_range = [-1.0, 1.0]
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

    def clear(self):
        self.points = []
        self.update()

    def add_points(self, pts):
        self.points.extend(pts)
        self.update()

    def set_ranges(self, x_min, x_max, y_min, y_max):
        self.x_range = [x_min, x_max]
        self.y_range = [y_min, y_max]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        w = self.width()
        h = self.height()
        margin = 40
        plot_rect = QRectF(margin, margin, w - 2*margin, h - 2*margin)

        # 绘制坐标轴框
        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(plot_rect)

        if not self.points:
            painter.drawText(plot_rect, Qt.AlignCenter, "暂无数据")
            painter.end()
            return

        # 坐标映射
        sx = plot_rect.width() / (self.x_range[1] - self.x_range[0])
        sy = plot_rect.height() / (self.y_range[1] - self.y_range[0])
        ox = plot_rect.left() - self.x_range[0] * sx
        oy = plot_rect.top() - self.y_range[1] * sy   # y轴翻转：大值在上方

        # 绘制数据点
        painter.setPen(QPen(QColor(0, 0, 180, 150), 1))
        for x, y in self.points:
            px = ox + x * sx
            py = oy + y * sy
            painter.drawPoint(int(px), int(py))

        # 轴标签
        painter.setPen(Qt.black)
        painter.drawText(plot_rect.right()+5, plot_rect.bottom(), self.xlabel)
        painter.drawText(plot_rect.left()-30, plot_rect.top()-5, self.ylabel)
        painter.drawText(int(plot_rect.center().x()-50), int(plot_rect.top()-15), self.title)

        painter.end()

# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("太阳系 KAM 稳定性可视化")
        self.setMinimumSize(800, 700)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # ---------- Tab 1: CRTBP Poincaré 截面 ----------
        self.crtbp_tab = QWidget()
        self.init_crtbp_tab()
        self.tabs.addTab(self.crtbp_tab, "太阳-木星 (旋转坐标系)")

        # ---------- Tab 2: 标准映射 ----------
        self.stdmap_tab = QWidget()
        self.init_stdmap_tab()
        self.tabs.addTab(self.stdmap_tab, "简化共振模型 (标准映射)")

        # 启动后自动运行 CRTBP
        QTimer.singleShot(100, self.auto_run_crtbp)

    def init_crtbp_tab(self):
        layout = QVBoxLayout(self.crtbp_tab)

        self.crtbp_canvas = SectionCanvas(title="CRTBP Poincaré 截面 (y=0, vy>0)",
                                          xlabel="x", ylabel="vx")
        self.crtbp_canvas.set_ranges(0.6, 1.4, -0.8, 0.8)
        layout.addWidget(self.crtbp_canvas)

        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)

        c_row = QHBoxLayout()
        c_row.addWidget(QLabel("雅可比常数 C:"))
        self.c_slider = QSlider(Qt.Horizontal)
        self.c_slider.setRange(250, 305)   # 2.50 ~ 3.05
        self.c_slider.setValue(300)
        c_row.addWidget(self.c_slider)
        self.c_label = QLabel("3.00")
        c_row.addWidget(self.c_label)
        self.c_slider.valueChanged.connect(self.on_c_changed)
        ctrl_layout.addLayout(c_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("轨道数:"))
        self.orbits_slider = QSlider(Qt.Horizontal)
        self.orbits_slider.setRange(10, 80)
        self.orbits_slider.setValue(30)
        param_row.addWidget(self.orbits_slider)
        self.orbits_label = QLabel("30")
        self.orbits_slider.valueChanged.connect(lambda v: self.orbits_label.setText(str(v)))
        param_row.addWidget(self.orbits_label)

        param_row.addWidget(QLabel("最大步数:"))
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setRange(1000, 10000)
        self.steps_slider.setValue(4000)
        self.steps_slider.setSingleStep(500)
        param_row.addWidget(self.steps_slider)
        self.steps_label = QLabel("4000")
        self.steps_slider.valueChanged.connect(lambda v: self.steps_label.setText(str(v)))
        param_row.addWidget(self.steps_label)
        ctrl_layout.addLayout(param_row)

        btn_row = QHBoxLayout()
        self.btn_run_crtbp = QPushButton("开始计算截面")
        self.btn_run_crtbp.clicked.connect(self.start_crtbp)
        btn_row.addWidget(self.btn_run_crtbp)
        self.btn_stop_crtbp = QPushButton("停止")
        self.btn_stop_crtbp.clicked.connect(self.stop_crtbp)
        self.btn_stop_crtbp.setEnabled(False)
        btn_row.addWidget(self.btn_stop_crtbp)
        self.btn_clear_crtbp = QPushButton("清除")
        self.btn_clear_crtbp.clicked.connect(self.crtbp_canvas.clear)
        btn_row.addWidget(self.btn_clear_crtbp)
        ctrl_layout.addLayout(btn_row)

        self.progress_crtbp = QProgressBar()
        ctrl_layout.addWidget(self.progress_crtbp)

        layout.addWidget(ctrl)
        self.thread_crtbp = None

    def init_stdmap_tab(self):
        layout = QVBoxLayout(self.stdmap_tab)

        self.stdmap_canvas = SectionCanvas(title="标准映射 (KAM 环面与混沌)",
                                           xlabel="q (角度)", ylabel="p (作用量)")
        self.stdmap_canvas.set_ranges(0, 1, 0, 1)
        layout.addWidget(self.stdmap_canvas)

        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)

        k_row = QHBoxLayout()
        k_row.addWidget(QLabel("扰动强度 K:"))
        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setRange(0, 200)   # 0.00 ~ 2.00
        self.k_slider.setValue(50)
        k_row.addWidget(self.k_slider)
        self.k_label = QLabel("0.50")
        k_row.addWidget(self.k_label)
        self.k_slider.valueChanged.connect(self.on_k_changed)
        ctrl_layout.addLayout(k_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("轨道数:"))
        self.norb_spin = QSlider(Qt.Horizontal)
        self.norb_spin.setRange(10, 60)
        self.norb_spin.setValue(30)
        param_row.addWidget(self.norb_spin)
        self.norb_label = QLabel("30")
        self.norb_spin.valueChanged.connect(lambda v: self.norb_label.setText(str(v)))
        param_row.addWidget(self.norb_label)

        param_row.addWidget(QLabel("步数:"))
        self.step_spin = QSlider(Qt.Horizontal)
        self.step_spin.setRange(100, 2000)
        self.step_spin.setValue(500)
        param_row.addWidget(self.step_spin)
        self.step_label = QLabel("500")
        self.step_spin.valueChanged.connect(lambda v: self.step_label.setText(str(v)))
        param_row.addWidget(self.step_label)
        ctrl_layout.addLayout(param_row)

        btn_row = QHBoxLayout()
        self.btn_run_std = QPushButton("绘制轨道")
        self.btn_run_std.clicked.connect(self.start_stdmap)
        btn_row.addWidget(self.btn_run_std)
        self.btn_stop_std = QPushButton("停止")
        self.btn_stop_std.clicked.connect(self.stop_stdmap)
        self.btn_stop_std.setEnabled(False)
        btn_row.addWidget(self.btn_stop_std)
        self.btn_clear_std = QPushButton("清除")
        self.btn_clear_std.clicked.connect(self.stdmap_canvas.clear)
        btn_row.addWidget(self.btn_clear_std)
        ctrl_layout.addLayout(btn_row)

        self.progress_std = QProgressBar()
        ctrl_layout.addWidget(self.progress_std)

        layout.addWidget(ctrl)
        self.thread_std = None

    def auto_run_crtbp(self):
        """启动后自动进行一次轻量积分，确保画布有内容"""
        self.c_slider.setValue(300)          # C=3.0
        self.orbits_slider.setValue(20)      # 20条轨道，快速出图
        self.steps_slider.setValue(2000)
        self.start_crtbp()

    def on_c_changed(self, val):
        self.c_label.setText(f"{val/100.0:.2f}")

    def start_crtbp(self):
        if self.thread_crtbp and self.thread_crtbp.isRunning():
            return
        self.crtbp_canvas.clear()
        self.btn_run_crtbp.setEnabled(False)
        self.btn_stop_crtbp.setEnabled(True)
        self.progress_crtbp.setValue(0)

        C = self.c_slider.value() / 100.0
        num_orbits = self.orbits_slider.value()
        max_steps = self.steps_slider.value()
        self.thread_crtbp = PoincareThread(C=C, x_range=(0.6, 1.4),
                                           num_orbits=num_orbits,
                                           dt=0.1, max_steps=max_steps)
        self.thread_crtbp.new_points.connect(self.crtbp_canvas.add_points)
        self.thread_crtbp.progress.connect(self.progress_crtbp.setValue)
        self.thread_crtbp.finished.connect(self.crtbp_finished)
        self.thread_crtbp.start()

    def stop_crtbp(self):
        if self.thread_crtbp:
            self.thread_crtbp.stop()
            self.crtbp_finished()

    def crtbp_finished(self):
        self.btn_run_crtbp.setEnabled(True)
        self.btn_stop_crtbp.setEnabled(False)
        self.progress_crtbp.setValue(100)

    # --- 标准映射 ---
    def on_k_changed(self, val):
        self.k_label.setText(f"{val/100.0:.2f}")

    def start_stdmap(self):
        if self.thread_std and self.thread_std.isRunning():
            return
        self.stdmap_canvas.clear()
        self.btn_run_std.setEnabled(False)
        self.btn_stop_std.setEnabled(True)
        self.progress_std.setValue(0)

        K = self.k_slider.value() / 100.0
        num_orbits = self.norb_spin.value()
        steps = self.step_spin.value()
        self.thread_std = StandardMapThread(K=K, num_orbits=num_orbits, steps=steps)
        self.thread_std.new_points.connect(self.stdmap_canvas.add_points)
        self.thread_std.progress.connect(self.progress_std.setValue)
        self.thread_std.finished.connect(self.stdmap_finished)
        self.thread_std.start()

    def stop_stdmap(self):
        if self.thread_std:
            self.thread_std.stop()
            self.stdmap_finished()

    def stdmap_finished(self):
        self.btn_run_std.setEnabled(True)
        self.btn_stop_std.setEnabled(False)
        self.progress_std.setValue(100)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())