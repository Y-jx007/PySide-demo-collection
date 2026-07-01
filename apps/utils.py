from custom_import import *

class OrbitCamera:
    """轨道相机：绕目标点旋转，支持平移和缩放"""

    def __init__(self, target=QVector3D(0, 0, 0), distance=100.0):
        self.target = QVector3D(target)
        self.distance = float(distance)
        self.yaw = -90.0          # 水平角度（度）
        self.pitch = 0.0          # 俯仰角度（度）
        self._update_vectors()

    def _update_vectors(self):
        # 根据 yaw/pitch 计算相机位置和方向向量
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        # 视线方向（从相机指向目标）
        self.front = QVector3D(
            math.cos(pitch_rad) * math.cos(yaw_rad),
            math.sin(pitch_rad),
            math.cos(pitch_rad) * math.sin(yaw_rad)
        ).normalized()
        self.position = self.target - self.front * self.distance
        self.right = QVector3D.crossProduct(self.front, QVector3D(0, 1, 0)).normalized()
        self.up = QVector3D.crossProduct(self.right, self.front).normalized()

    def view_matrix(self) -> QMatrix4x4:
        """返回当前视图矩阵"""
        self._update_vectors()
        view = QMatrix4x4()
        view.lookAt(self.position, self.target, self.up)
        return view

    def rotate(self, dyaw: float, dpitch: float):
        """旋转相机（角度制）"""
        self.yaw += dyaw
        self.pitch += dpitch
        self.pitch = max(-89.0, min(89.0, self.pitch))  # 防止翻转

    def pan(self, dx: float, dy: float):
        """在垂直于视线方向的平面内平移目标（同时移动相机）"""
        offset = self.right * dx + self.up * dy
        self.target += offset

    def zoom(self, delta: float):
        """缩放（改变距离）"""
        self.distance -= delta
        self.distance = max(2.0, min(self.distance, 500.0))

    def reset(self, target=QVector3D(0, 0, 0), distance=100.0):
        self.target = QVector3D(target)
        self.distance = float(distance)
        self.yaw = -90.0
        self.pitch = 0.0

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