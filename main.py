import os
import sys
import importlib
import logging

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize, QRect
from PySide6.QtGui import QFont, QCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QMessageBox,
    QLayout, QLayoutItem
)

# ---------- 流式布局 ----------
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self._items = []

    def __del__(self):
        while self._items:
            item = self._items.pop()
            if item.widget():
                item.widget().setParent(None)
            del item

    def addItem(self, item: QLayoutItem):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._doLayout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _doLayout(self, rect, testOnly):
        margins = self.contentsMargins()
        available_width = rect.width() - margins.left() - margins.right()
        x = margins.left()
        y = margins.top()
        line_height = 0

        for item in self._items:
            size = item.sizeHint()
            if x + size.width() > available_width and line_height > 0:
                x = margins.left()
                y += line_height + self.spacing()
                line_height = 0
            if not testOnly:
                item.setGeometry(QRect(x, y, size.width(), size.height()))
            x += size.width() + self.spacing()
            line_height = max(line_height, size.height())

        total_height = y + line_height + margins.bottom()
        return total_height


# ---------- 信号 ----------
class AppSignal(QObject):
    error_occurred = Signal(str)
    app_launched = Signal(object)


# ---------- 应用卡片（正方形）----------
class AppCard(QFrame):
    def __init__(self, title, description, command, disabled=False, parent=None):
        super().__init__(parent)
        self.title = title
        self.description = description
        self.command = command
        self.disabled = disabled
        self.setup_ui()
        self.setup_styles()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(12, 12, 12, 12)

        # 标题
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title_label.setStyleSheet("color: #333;")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # 描述
        desc_label = QLabel(self.description)
        desc_label.setFont(QFont("Segoe UI", 8))
        desc_label.setStyleSheet("color: #666;")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        layout.addStretch()

        # 按钮
        if self.disabled:
            self.button = QPushButton("开发中")
            self.button.setEnabled(False)
            self.button.setStyleSheet("""
                QPushButton {
                    background-color: #e9ecef; color: #6c757d;
                    border: none; padding: 5px 12px; border-radius: 4px;
                    font-size: 10px;
                }
            """)
        else:
            self.button = QPushButton("启动")
            self.button.setCursor(QCursor(Qt.PointingHandCursor))
            self.button.clicked.connect(self.command)
            self.button.setStyleSheet("""
                QPushButton {
                    background-color: #007bff; color: white;
                    border: none; padding: 5px 12px; border-radius: 4px;
                    font-size: 10px;
                }
                QPushButton:hover { background-color: #0056b3; }
                QPushButton:pressed { background-color: #004085; }
            """)

        self.button.setFixedSize(100, 28)
        layout.addWidget(self.button, 0, Qt.AlignCenter)

        # ★ 关键修复：固定正方形尺寸 ★
        self.setFixedSize(160, 160)   # 恢复原尺寸，确保卡片为正方形

    def setup_styles(self):
        self.setStyleSheet("""
            AppCard {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 6px;
            }
            AppCard:hover { border-color: #007bff; }
        """)


# ---------- 主窗口 ----------
class ApplicationLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.active_windows = []
        self.cards = []

        self.signal = AppSignal()
        self.signal.error_occurred.connect(self.show_error)
        self.signal.app_launched.connect(self.handle_app_launch)

        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

        self.setup_ui()
        self.setup_apps()

    def setup_ui(self):
        self.setWindowTitle("绘图工具集")
        self.setMinimumSize(640, 480)
        self.resize(1100, 600)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move((screen.width() - self.width()) // 2,
                  (screen.height() - self.height()) // 2)

        self.setStyleSheet("""
            QMainWindow { background-color: #f8f9fa; }
            QScrollArea {
                border: 1px solid #dee2e6; border-radius: 6px;
                background-color: #f8f9fa;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(24, 16, 24, 12)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scroll_content = QWidget()
        self.flow_layout = FlowLayout(self.scroll_content, margin=16, spacing=16)
        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area)

        info = QLabel("选择工具开始使用 | 绘图工具集 v1.0")
        info.setFont(QFont("Segoe UI", 8))
        info.setStyleSheet("color: #888; margin-top: 8px;")
        info.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info)

    def _make_launcher(self, module_filename, friendly_name):
        def launcher():
            apps_dir = os.path.join(os.path.dirname(__file__), 'apps')
            module_path = os.path.join(apps_dir, module_filename)
            QTimer.singleShot(0, lambda: self.load_app_module(module_path, friendly_name))
        return launcher

    def setup_apps(self):
        app_definitions = [
            ("Fourier",      "绘制图形并查看其傅里叶变换",              "2D Fourier.py", "二维傅里叶变换"),
            ("Arnold",       "可视化图像猫变换",                       "2D Arnold.py", "猫变换"),
            ("Schrödinger",  "Runge-Kutta模拟二维量子波包演化过程",      "2D Schrödinger.py", "薛定谔方程"),
            ("Turing",       "模拟Gray-Scott反应扩散系统生成图灵斑的过程", "2D Turing.py", "反应扩散模型"),
            ("Ising",        "模拟二维伊辛模型",                       "2D Ising.py", "二维伊辛模型"),
            ("Maxwell",      "模拟气体分子速率分布",                    "2D Maxwell.py", "麦克斯韦分布律"),
            ("Mandelbrot-Julia", "复动力系统生成分形",                 "2D Mandelbrot.py", "复动力系统"),
            ("Newton",       "牛顿迭代法生成分形",                     "2D Newton.py", "牛顿迭代法分形"),
            ("Lindenmayer",  "林氏系统分形",                          "2D Lindenmayer.py", "林氏系统"),
            ("IFS分形",      "迭代函数系统生成分形图案",                "2D LFS.py", "LFS分形"),
            ("Logistic分形", "逻辑映射生成分形图案",                    "2D Logistic.py", "逻辑映射分形"),
            ("Lloyd",        "用Voronoi图的Lloyd松弛过程模拟蜂巢结构的形成", "2D Lloyd.py", "Lloyd松弛"),
            ("幅角图",       "在复平面上可视化复变函数",                "2D Cplot.py", "辐角图"),
            ("旅商问题",     "使用遗传算法解决旅行商问题",              "2D Traveler.py", "旅行商问题遗传算法"),
            ("Conway",       "模拟和可视化生命游戏",                    "2D Conway.py", "元胞自动机"),
        ]

        for title, desc, filename, friendly in app_definitions:
            command = self._make_launcher(filename, friendly)
            card = AppCard(title, desc, command)
            self.cards.append(card)
            self.flow_layout.addWidget(card)

    def load_app_module(self, module_path, friendly_name):
        try:
            if not os.path.exists(module_path):
                raise FileNotFoundError(f"找不到文件: {module_path}")

            module_name = os.path.splitext(os.path.basename(module_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError("无法创建模块规格")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            app_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, QMainWindow) and
                    attr is not QMainWindow):
                    app_class = attr
                    break

            if app_class is None:
                raise ImportError("模块中未找到 QMainWindow 子类")

            self.signal.app_launched.emit(app_class)
        except Exception as e:
            logging.exception(f"加载模块失败: {friendly_name}")
            self.signal.error_occurred.emit(f"无法启动 {friendly_name}: {str(e)}")

    def handle_app_launch(self, app_class):
        try:
            window = app_class()
            window.show()
            self.active_windows.append(window)
            window.destroyed.connect(lambda w=window: self.active_windows.remove(w))
        except Exception as e:
            logging.exception("创建应用窗口出错")
            self.show_error(f"创建应用窗口时出错: {str(e)}")

    def show_error(self, message):
        QMessageBox.critical(self, "错误", message)

    def closeEvent(self, event):
        for win in self.active_windows[:]:
            win.close()
        event.accept()


if __name__ == "__main__":
    apps_dir = os.path.join(os.path.dirname(__file__), 'apps')
    if apps_dir not in sys.path:
        sys.path.append(apps_dir)

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))

    launcher = ApplicationLauncher()
    launcher.show()

    sys.exit(app.exec())