import math
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QVector3D, QMatrix4x4, QQuaternion
from PySide6.QtWidgets import QPushButton, QColorDialog

class OrbitCamera:
    def __init__(self, target=QVector3D(0, 0, 0), distance=100.0):
        self.target = QVector3D(target)
        self.distance = float(distance)
        self.yaw = -90.0
        self.pitch = 0.0

        # ---------- 原有欧拉角（吸引子滑块继续用）----------
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0

        # ---------- 模型旋转四元数（分形用）----------
        self.model_rotation = QQuaternion()

        # ---------- Arcball 内部状态 ----------
        self.arc_start = None   # 存储拖拽起始点（球面坐标）

        self._update_vectors()

    # ---------- 核心向量更新 ----------
    def _update_vectors(self):
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        self.front = QVector3D(
            math.cos(pitch_rad) * math.cos(yaw_rad),
            math.sin(pitch_rad),
            math.cos(pitch_rad) * math.sin(yaw_rad)
        ).normalized()
        self.position = self.target - self.front * self.distance
        self.right = QVector3D.crossProduct(self.front, QVector3D(0, 1, 0)).normalized()
        self.up = QVector3D.crossProduct(self.right, self.front).normalized()

    # ---------- 矩阵生成 ----------
    def view_matrix(self) -> QMatrix4x4:
        self._update_vectors()
        view = QMatrix4x4()
        view.lookAt(self.position, self.target, self.up)
        return view

    def model_matrix(self) -> QMatrix4x4:
        model = QMatrix4x4()
        model.translate(self.target)

        # 1. 旧欧拉角旋转（吸引子滑块控制）
        model.rotate(self.rot_z, 0, 0, 1)
        model.rotate(self.rot_x, 1, 0, 0)
        model.rotate(self.rot_y, 0, 1, 0)

        # 2. 四元数旋转（Arcball / roll 产生）
        model.rotate(self.model_rotation)

        model.translate(-self.target)
        return model

    # ---------- 视角控制（不变）----------
    def rotate_view(self, dyaw, dpitch):
        self.yaw += dyaw
        self.pitch += dpitch
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def rotate_model(self, dx, dy, dz):
        self.rot_x += dx
        self.rot_y += dy
        self.rot_z += dz

    def pan(self, dx, dy):
        offset = self.right * dx + self.up * dy
        self.target += offset

    def zoom(self, delta):
        self.distance -= delta
        self.distance = max(2.0, min(self.distance, 500.0))

    # ---------- 屏幕坐标 → 单位半球（静态方法）----------
    @staticmethod
    def screen_to_sphere(x, y, w, h):
        """
        将屏幕坐标映射到半球表面。
        x, y : 鼠标在窗口中的像素坐标
        w, h : 窗口的宽高
        返回单位半球上的点（Z≥0），若在球外则投影到边缘。
        """
        fx = 1.0 - 2.0 * x / w
        fy = 2.0 * y / h - 1.0          # 不翻转 Y，与屏幕坐标一致
        length2 = fx*fx + fy*fy
        if length2 > 1.0:
            norm = 1.0 / math.sqrt(length2)
            return QVector3D(fx * norm, fy * norm, 0.0)
        else:
            return QVector3D(fx, fy, math.sqrt(1.0 - length2))

    # ---------- Arcball 接口 ----------
    def start_arcball(self, x, y, w, h):
        """鼠标按下时调用，记录起始球面点"""
        self.arc_start = self.screen_to_sphere(x, y, w, h)

    def update_arcball(self, x, y, w, h):
        """
        鼠标拖动时调用，根据起始点与当前点计算旋转四元数，
        并右乘到 model_rotation 上（实现本地轴旋转）。
        """
        if self.arc_start is None:
            return
        curr = self.screen_to_sphere(x, y, w, h)
        # 旋转轴 = 起始点 × 当前点（叉积）
        axis = QVector3D.crossProduct(self.arc_start, curr)
        if axis.length() < 1e-6:
            return
        axis.normalize()
        dot = QVector3D.dotProduct(self.arc_start, curr)
        dot = max(-1.0, min(1.0, dot))
        angle = math.acos(dot)          # 弧度
        q_rot = QQuaternion.fromAxisAndAngle(axis, math.degrees(angle))
        # 右乘：新旋转叠加在当前旋转之后（本地坐标系）
        self.model_rotation = self.model_rotation * q_rot
        self.arc_start = curr

    def end_arcball(self):
        """鼠标释放时可调用（非必须，仅清理状态）"""
        self.arc_start = None

    # ---------- 绕视线滚动（Ctrl+左键）----------
    def roll_model(self, angle_deg):
        q_roll = QQuaternion.fromAxisAndAngle(self.front, angle_deg)
        self.model_rotation = q_roll * self.model_rotation

    # ---------- 重置 ----------
    def reset(self):
        self.target = QVector3D(0, 0, 0)
        self.yaw = -90.0
        self.pitch = -30.0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.model_rotation = QQuaternion()
        self.arc_start = None

class ColorButton(QPushButton):
    """带边框的颜色预览按钮，点击弹出系统原生选色界面（无任何多余边框）"""
    colorChanged = Signal(QColor)

    def __init__(self, color=QColor(255, 255, 255), parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(30, 30)

        # 去除焦点框，保留透明背景（边框在paintEvent中手绘，确保永远可见）
        self.setStyleSheet(
            "QPushButton { background-color: transparent; }"
            "QPushButton:focus { outline: none; }"
        )
        self.setFocusPolicy(Qt.NoFocus)
        self.clicked.connect(self._on_click)

    def _on_click(self):
        # 创建系统原生颜色对话框，完全不设置样式表，避免任何边框异常
        dialog = QColorDialog(self._color, self)
        dialog.setWindowTitle("选择颜色")
        # 关键：不调用 dialog.setStyleSheet()，让对话框保持原生无干扰状态
        if dialog.exec() == QColorDialog.Accepted:
            self.set_color(dialog.currentColor())

    def set_color(self, color: QColor):
        if color != self._color:
            self._color = color
            self.update()
            self.colorChanged.emit(color)

    def color(self) -> QColor:
        return self._color

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制填充颜色（留出2px边框空间）
        painter.fillRect(2, 2, self.width()-4, self.height()-4, self._color)

        # 手绘灰色圆角边框（永不消失）
        pen = QPen(QColor("#888888"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRoundedRect(1, 1, self.width()-2, self.height()-2, 3, 3)