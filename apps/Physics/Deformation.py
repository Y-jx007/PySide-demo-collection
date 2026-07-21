import sys
import numpy as np
import taichi_forge as ti
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QPushButton
)
from PySide6.QtCore import QTimer, Qt, QPointF, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPolygonF, QFont

ti.init(arch=ti.gpu)

# ========== 固定参数 ==========
thickness = 0.02                # 平面外厚度 (m)
nx, ny = 80, 20                 # 网格 80x20
critical_strain = 0.02          # 断裂应变
dt = 2e-5
substeps = 50
rho = 1000.0
E = 5e5
nu = 0.3
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
lam_ps = 2 * mu * lam / (lam + 2 * mu)  # 平面应力
damping_mass = 2.0              # 质量比例阻尼系数
weight_contact_stiff = 1e7
weight_contact_damp = 500.0
weight_width = 0.2
weight_height = 0.05
weight_start_y_offset = 0.02

# ========== 可调参数（存储在 field 中） ==========
Lx_field = ti.field(ti.f32, shape=())
Ly_field = ti.field(ti.f32, shape=())
weight_mass_field = ti.field(ti.f32, shape=())

num_vertices = (nx + 1) * (ny + 1)
num_triangles = 2 * nx * ny

# ========== Taichi 数据场 ==========
pos = ti.Vector.field(2, ti.f32, shape=num_vertices)
vel = ti.Vector.field(2, ti.f32, shape=num_vertices)
mass = ti.field(ti.f32, shape=num_vertices)
force = ti.Vector.field(2, ti.f32, shape=num_vertices)

tri = ti.Vector.field(3, ti.i32, shape=num_triangles)
tri_active = ti.field(ti.i32, shape=num_triangles)
Dm_inv = ti.Matrix.field(2, 2, ti.f32, shape=num_triangles)
A0 = ti.field(ti.f32, shape=num_triangles)
strain_max = ti.field(ti.f32, shape=())

weight_pos = ti.Vector.field(2, ti.f32, shape=())
weight_vel = ti.Vector.field(2, ti.f32, shape=())
first_fracture_recorded = ti.field(ti.i32, shape=())
first_fracture_height = ti.field(ti.f32, shape=())

tri_vertices = ti.Vector.field(2, ti.f32, shape=num_triangles * 3)
tri_colors   = ti.Vector.field(3, ti.f32, shape=num_triangles)

# ========== 构建网格（恢复原始初始化，保持几何均匀） ==========
@ti.kernel
def build_mesh_kernel():
    Lx = Lx_field[None]
    Ly = Ly_field[None]
    dx = Lx / nx
    dy = Ly / ny

    for i, j in ti.ndrange(nx + 1, ny + 1):
        idx = i * (ny + 1) + j
        # 所有节点均匀分布，包括后续将被约束的支撑节点
        pos[idx] = ti.Vector([i * dx, j * dy])
        vel[idx] = ti.Vector([0.0, 0.0])
        mass[idx] = 0.0

    for i, j in ti.ndrange(nx, ny):
        quad_idx = i * ny + j
        bl = i * (ny + 1) + j
        br = (i + 1) * (ny + 1) + j
        tl = i * (ny + 1) + j + 1
        tr = (i + 1) * (ny + 1) + j + 1
        t0 = 2 * quad_idx
        t1 = t0 + 1
        tri[t0] = ti.Vector([bl, br, tr])
        tri[t1] = ti.Vector([bl, tr, tl])
        tri_active[t0] = 1
        tri_active[t1] = 1

        p0 = pos[bl]; p1 = pos[br]; p2 = pos[tr]
        Dm = ti.Matrix.cols([p1 - p0, p2 - p0])
        Dm_inv[t0] = Dm.inverse()
        A0[t0] = 0.5 * ti.abs(Dm.determinant())

        p0 = pos[bl]; p1 = pos[tr]; p2 = pos[tl]
        Dm = ti.Matrix.cols([p1 - p0, p2 - p0])
        Dm_inv[t1] = Dm.inverse()
        A0[t1] = 0.5 * ti.abs(Dm.determinant())

    for t in range(num_triangles):
        a, b, c = tri[t]
        tri_mass = rho * A0[t] * thickness
        mass[a] += tri_mass / 3.0
        mass[b] += tri_mass / 3.0
        mass[c] += tri_mass / 3.0

    # 重物初始位置（板顶 Ly + 偏移）
    weight_pos[None] = ti.Vector([Lx / 2, Ly + weight_start_y_offset + weight_height / 2])
    weight_vel[None] = ti.Vector([0.0, 0.0])
    first_fracture_recorded[None] = 0
    first_fracture_height[None] = 0.0

# ========== 内力 ==========
@ti.kernel
def compute_internal_forces():
    for v in range(num_vertices):
        force[v] = ti.Vector([0.0, 0.0])

    for t in range(num_triangles):
        if tri_active[t] == 0:
            continue
        a, b, c = tri[t]
        p0 = pos[a]; p1 = pos[b]; p2 = pos[c]
        Ds = ti.Matrix.cols([p1 - p0, p2 - p0])
        F = Ds @ Dm_inv[t]
        J = F.determinant()
        if J < 1e-4:
            J = 1e-4
        FinvT = F.inverse().transpose()
        P = mu * (F - FinvT) + lam_ps * ti.log(J) * FinvT
        H = -A0[t] * thickness * P @ Dm_inv[t].transpose()
        force[a] += -H[:, 0] - H[:, 1]
        force[b] += H[:, 0]
        force[c] += H[:, 1]

# ========== 外力（重力 + 重物接触） ==========
@ti.kernel
def apply_external_forces():
    g = ti.Vector([0.0, -9.8])
    for v in range(num_vertices):
        force[v] += mass[v] * g

    w_mass = weight_mass_field[None]
    weight_force = ti.Vector([0.0, -w_mass * 9.8])
    w_bottom = weight_pos[None].y - weight_height / 2.0
    w_left = weight_pos[None].x - weight_width / 2.0
    w_right = weight_pos[None].x + weight_width / 2.0

    contact_force_total = ti.Vector([0.0, 0.0])
    for i in range(nx + 1):
        v = i * (ny + 1) + ny   # 顶部节点
        if pos[v].x >= w_left and pos[v].x <= w_right and pos[v].y > w_bottom:
            penetration = pos[v].y - w_bottom
            n = ti.Vector([0.0, -1.0])
            v_rel = vel[v] - weight_vel[None]
            f = weight_contact_stiff * penetration * n + weight_contact_damp * v_rel.dot(n) * n
            force[v] += f
            contact_force_total -= f

    weight_force += contact_force_total
    weight_vel[None] += weight_force / w_mass * dt
    weight_pos[None] += weight_vel[None] * dt

# ========== 时间积分 + 刚性支撑 ==========
@ti.kernel
def integrate():
    for v in range(num_vertices):
        i = v // (ny + 1)
        j = v % (ny + 1)

        # 支撑节点：固定在桌面上（y=0）
        if (i == 0 or i == nx) and j <= 2:
            vel[v] = ti.Vector([0.0, 0.0])
            pos[v].y = 0.0   # 强制约束
            continue

        # 非支撑节点正常更新
        damping = -damping_mass * mass[v] * vel[v]
        acc = force[v] / mass[v] + damping / mass[v]
        vel[v] += acc * dt
        pos[v] += vel[v] * dt

# ========== 断裂检测 ==========
@ti.kernel
def detect_fracture():
    strain_max[None] = 0.0
    for t in range(num_triangles):
        if tri_active[t] == 0:
            continue
        a, b, c = tri[t]
        p0 = pos[a]; p1 = pos[b]; p2 = pos[c]
        Ds = ti.Matrix.cols([p1 - p0, p2 - p0])
        F = Ds @ Dm_inv[t]
        C = F.transpose() @ F
        trace = C[0, 0] + C[1, 1]
        detC = C[0, 0]*C[1, 1] - C[0, 1]*C[1, 0]
        sqrt_disc = ti.sqrt(ti.max(trace*trace - 4.0*detC, 0.0))
        lambda_max = 0.5 * (trace + sqrt_disc)
        max_strain = ti.sqrt(ti.max(lambda_max, 1e-12)) - 1.0
        ti.atomic_max(strain_max[None], max_strain)
        if max_strain > critical_strain:
            tri_active[t] = 0
            if first_fracture_recorded[None] == 0:
                first_fracture_recorded[None] = 1
                first_fracture_height[None] = weight_pos[None].y

# ========== 导出绘图数据 ==========
@ti.kernel
def export_drawing_data():
    for t in range(num_triangles):
        a, b, c = tri[t]
        base = 3 * t
        if tri_active[t] == 1:
            tri_vertices[base]   = pos[a]
            tri_vertices[base+1] = pos[b]
            tri_vertices[base+2] = pos[c]

            p0 = pos[a]; p1 = pos[b]; p2 = pos[c]
            Ds = ti.Matrix.cols([p1 - p0, p2 - p0])
            F = Ds @ Dm_inv[t]
            C = F.transpose() @ F
            trace = C[0, 0] + C[1, 1]
            detC = C[0, 0]*C[1, 1] - C[0, 1]*C[1, 0]
            sqrt_disc = ti.sqrt(ti.max(trace*trace - 4.0*detC, 0.0))
            lambda_max = 0.5 * (trace + sqrt_disc)
            strain = ti.sqrt(ti.max(lambda_max, 1e-12)) - 1.0
            ratio = ti.min(strain / critical_strain, 1.0)
            tri_colors[t] = ti.Vector([ratio, 1.0 - ratio, 0.0])
        else:
            tri_vertices[base]   = ti.Vector([-1000.0, -1000.0])
            tri_vertices[base+1] = ti.Vector([-1000.0, -1000.0])
            tri_vertices[base+2] = ti.Vector([-1000.0, -1000.0])
            tri_colors[t] = ti.Vector([0.0, 0.0, 0.0])

# ========== PySide6 界面 ==========
class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.margin = 40
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.tri_verts_np = None
        self.tri_colors_np = None
        self.tri_active_np = None
        self.weight_pos_np = None
        self.Lx = 2.0
        self.Ly = 0.2

    def resizeEvent(self, event):
        w = self.width()
        h = self.height()
        scale_x = (w - 2 * self.margin) / self.Lx
        scale_y = (h - 2 * self.margin) / self.Ly
        self.scale = min(scale_x, scale_y) * 0.9
        self.offset_x = (w - self.Lx * self.scale) / 2
        self.offset_y = (h + self.Ly * self.scale) / 2

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor(40, 40, 45))

            if self.tri_verts_np is None or self.tri_active_np is None:
                return

            for t in range(num_triangles):
                if self.tri_active_np[t] == 0:
                    continue
                base = 3 * t
                if self.tri_verts_np[base][0] < -100:
                    continue
                x0 = self.tri_verts_np[base][0] * self.scale + self.offset_x
                y0 = self.offset_y - self.tri_verts_np[base][1] * self.scale
                x1 = self.tri_verts_np[base+1][0] * self.scale + self.offset_x
                y1 = self.offset_y - self.tri_verts_np[base+1][1] * self.scale
                x2 = self.tri_verts_np[base+2][0] * self.scale + self.offset_x
                y2 = self.offset_y - self.tri_verts_np[base+2][1] * self.scale

                color = self.tri_colors_np[t]
                r = min(255, int(color[0] * 255))
                g = min(255, int(color[1] * 255))
                b = min(255, int(color[2] * 255))
                painter.setBrush(QColor(r, g, b, 220))
                painter.setPen(Qt.NoPen)
                triangle = QPolygonF([QPointF(x0, y0), QPointF(x1, y1), QPointF(x2, y2)])
                painter.drawConvexPolygon(triangle)

            # 支撑点（红点）—— 物理坐标 y=0
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 80, 80))
            dx_val = self.Lx / nx
            for side in [0, nx]:
                x_screen = side * dx_val * self.scale + self.offset_x
                y_screen = self.offset_y  # y=0 映射到屏幕的 offset_y
                painter.drawRect(QRectF(x_screen - 3, y_screen - 3, 6, 6))

            # 重物
            if self.weight_pos_np is not None:
                wx = self.weight_pos_np[0] - weight_width / 2
                wy = self.weight_pos_np[1] - weight_height / 2
                x = wx * self.scale + self.offset_x
                y = self.offset_y - (wy + weight_height) * self.scale
                w = weight_width * self.scale
                h = weight_height * self.scale
                painter.setBrush(QColor(180, 180, 180, 200))
                painter.setPen(QPen(QColor(100, 100, 100), 2))
                painter.drawRect(QRectF(x, y, w, h))

            font = QFont("Consolas", 10)
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255))
            active_count = int(np.sum(self.tri_active_np))
            max_s = strain_max[None]
            painter.drawText(10, 20, f"重物中心高度: {weight_pos[None].y:.3f} m")
            painter.drawText(10, 35, f"断裂: {num_triangles - active_count}/{num_triangles}  最大应变: {max_s:.4f}")
            if first_fracture_recorded[None]:
                painter.drawText(10, 50, f"首次断裂时重物中心高度: {first_fracture_height[None]:.3f} m")
        finally:
            painter.end()


class ControlPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        layout = QVBoxLayout(self)

        mass_layout = QHBoxLayout()
        mass_layout.addWidget(QLabel("重物质量 (kg):"))
        self.mass_slider = QSlider(Qt.Horizontal)
        self.mass_slider.setRange(1, 10)
        self.mass_slider.setValue(2)
        self.mass_label = QLabel("2.0")
        self.mass_slider.valueChanged.connect(lambda v: self.mass_label.setText(f"{v:.1f}"))
        mass_layout.addWidget(self.mass_slider)
        mass_layout.addWidget(self.mass_label)
        layout.addLayout(mass_layout)

        len_layout = QHBoxLayout()
        len_layout.addWidget(QLabel("板长 (m):"))
        self.len_slider = QSlider(Qt.Horizontal)
        self.len_slider.setRange(10, 40)
        self.len_slider.setValue(20)
        self.len_label = QLabel("2.0")
        self.len_slider.valueChanged.connect(lambda v: self.len_label.setText(f"{v/10:.1f}"))
        len_layout.addWidget(self.len_slider)
        len_layout.addWidget(self.len_label)
        layout.addLayout(len_layout)

        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("板高 (m):"))
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(10, 50)
        self.height_slider.setValue(20)
        self.height_label = QLabel("0.20")
        self.height_slider.valueChanged.connect(lambda v: self.height_label.setText(f"{v/100:.2f}"))
        height_layout.addWidget(self.height_slider)
        height_layout.addWidget(self.height_label)
        layout.addLayout(height_layout)

        self.reset_btn = QPushButton("重置模拟")
        self.reset_btn.clicked.connect(self.main.reset_simulation)
        layout.addWidget(self.reset_btn)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("板弯曲断裂模拟")
        self.resize(1000, 650)
        self.canvas = Canvas(self)
        self.control = ControlPanel(self)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.control)
        self.setCentralWidget(central)

        self.update_params_and_build()
        self.timer = QTimer()
        self.timer.timeout.connect(self.step)
        self.timer.start(10)

    def update_params_and_build(self):
        Lx = self.control.len_slider.value() / 10.0
        Ly = self.control.height_slider.value() / 100.0
        wm = self.control.mass_slider.value()
        Lx_field[None] = Lx
        Ly_field[None] = Ly
        weight_mass_field[None] = float(wm)
        self.canvas.Lx = Lx
        self.canvas.Ly = Ly
        build_mesh_kernel()

    def reset_simulation(self):
        self.timer.stop()
        self.update_params_and_build()
        self.timer.start(10)

    def step(self):
        for _ in range(substeps):
            compute_internal_forces()
            apply_external_forces()
            integrate()
            detect_fracture()

        ti.sync()
        export_drawing_data()
        ti.sync()

        self.canvas.tri_verts_np = np.copy(tri_vertices.to_numpy())
        self.canvas.tri_colors_np = np.copy(tri_colors.to_numpy())
        self.canvas.tri_active_np = np.copy(tri_active.to_numpy())
        self.canvas.weight_pos_np = np.copy(weight_pos.to_numpy())
        self.canvas.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())