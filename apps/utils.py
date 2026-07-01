from custom_import import *

class OrbitCamera:
    def __init__(self, target=QVector3D(0, 0, 0), distance=100.0):
        self.target = QVector3D(target)         # 观察目标
        self.distance = float(distance)         # 相机距离
        self.yaw = -90.0                        # 方位角
        self.pitch = 0.0                        # 俯仰角

        # 模型变换参数（绕目标点旋转）
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0

        self._update_vectors()

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

    def view_matrix(self) -> QMatrix4x4:
        """返回视图矩阵（相机位置不变，lookAt 目标）"""
        self._update_vectors()
        view = QMatrix4x4()
        view.lookAt(self.position, self.target, self.up)
        return view

    def model_matrix(self) -> QMatrix4x4:
        """返回模型变换矩阵：围绕目标点旋转"""
        model = QMatrix4x4()
        model.translate(self.target)
        model.rotate(self.rot_z, 0, 0, 1)
        model.rotate(self.rot_x, 1, 0, 0)
        model.rotate(self.rot_y, 0, 1, 0)
        model.translate(-self.target)
        return model

    # ----- 控制方法 -----
    def rotate_view(self, dyaw: float, dpitch: float):
        """旋转相机视角（绕目标）"""
        self.yaw += dyaw
        self.pitch += dpitch
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def rotate_model(self, dx: float, dy: float, dz: float):
        """增量旋转模型（绕 X/Y/Z 轴，角度制）"""
        self.rot_x += dx
        self.rot_y += dy
        self.rot_z += dz

    def pan(self, dx: float, dy: float):
        """平移相机和目标（屏幕平面内）"""
        offset = self.right * dx + self.up * dy
        self.target += offset

    def pan_model(self, dx: float, dy: float, dz: float):
        """直接设置模型平移（世界坐标系）"""
        self.target = QVector3D(dx, dy, dz)

    def zoom(self, delta: float):
        """缩放距离"""
        self.distance -= delta
        self.distance = max(2.0, min(self.distance, 500.0))

    def reset(self):
        self.target = QVector3D(0, 0, 0)
        self.distance = 100.0
        self.yaw = -90.0
        self.pitch = 0.0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0

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