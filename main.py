import os
import sys
import importlib
import importlib.util
import inspect
import logging
import io
import hashlib

from PIL import Image, ImageOps

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QCursor, QPainter, QPixmap, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QMessageBox,
    QSizePolicy, QGraphicsDropShadowEffect
)

# ===================== 全局常量 =====================
CARD_WIDTH = 150
CARD_HEIGHT = 150
CARD_BUTTON_WIDTH = 100
CARD_BUTTON_HEIGHT = 28

CARD_TITLE_FONT_SIZE = 11
CARD_DESC_FONT_SIZE = 8

CARD_SHADOW_BLUR = 8
CARD_SHADOW_OFFSET_X = 2
CARD_SHADOW_OFFSET_Y = 2

CARD_RADIUS = 8
CARD_CONTAINER_RADIUS = 6
CARD_BUTTON_RADIUS = 4

CARD_LAYOUT_SPACING = 6
CARD_LAYOUT_MARGINS = (12, 12, 12, 12)
CARD_TEXT_LAYOUT_SPACING = 2
CARD_TEXT_LAYOUT_MARGINS = (6, 4, 6, 4)

CARD_BORDER_WIDTH = 2

CATEGORY_HEIGHT = 170  # 增加高度避免阴影裁剪
CATEGORY_LABEL_WIDTH = 95
CATEGORY_LAYOUT_MARGINS = (12, 4, 10, 4)
CATEGORY_LAYOUT_SPACING = 12
CATEGORY_CARDS_SPACING = 16
CATEGORY_CARDS_MARGINS = (4, 4, 4, 4)
CATEGORY_TITLE_FONT_SIZE = 12
CATEGORY_RADIUS = 8

LAUNCHER_MIN_WIDTH = 640
LAUNCHER_MIN_HEIGHT = 480
LAUNCHER_DEFAULT_WIDTH = 1200
LAUNCHER_DEFAULT_HEIGHT = 800
LAUNCHER_MAIN_MARGINS = (24, 4, 24, 4)

INFO_FONT_SIZE = 8

FAMILY_TITLE = "Segoe UI"
FAMILY_DEFAULT = "Microsoft YaHei"


# ===================== 应用卡片 =====================
class AppCard(QFrame):
    def __init__(self, title, description, command, bg_image_path=None, disabled=False, parent=None):
        super().__init__(parent)
        self.title = title
        self.description = description
        self.command = command
        self.disabled = disabled
        self._color_pixmap = None
        self._gray_pixmap = None
        self._hover = False
        self._load_background(bg_image_path)
        self.setup_ui()
        self.setup_styles()
        self._apply_styles(hovered=False)

    def _load_background(self, image_path):
        inner_w = CARD_WIDTH - 2 * CARD_BORDER_WIDTH
        inner_h = CARD_HEIGHT - 2 * CARD_BORDER_WIDTH

        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                img = ImageOps.fit(img, (inner_w, inner_h),
                                   Image.Resampling.LANCZOS, centering=(0.5, 0.5))
                buffer_color = io.BytesIO()
                img.save(buffer_color, format='PNG')
                self._color_pixmap = QPixmap()
                self._color_pixmap.loadFromData(buffer_color.getvalue())

                gray_img = img.convert('L').convert('RGB')
                buffer_gray = io.BytesIO()
                gray_img.save(buffer_gray, format='PNG')
                self._gray_pixmap = QPixmap()
                self._gray_pixmap.loadFromData(buffer_gray.getvalue())
            except Exception as e:
                logging.warning(f"无法加载背景图片 {image_path}: {e}")

    def paintEvent(self, event):
        # 先绘制样式表背景和边框
        super().paintEvent(event)

        # 再绘制图片，不覆盖边框
        painter = QPainter(self)
        inner_rect = self.rect().adjusted(
            CARD_BORDER_WIDTH, CARD_BORDER_WIDTH,
            -CARD_BORDER_WIDTH, -CARD_BORDER_WIDTH
        )
        pix = self._color_pixmap if self._hover else self._gray_pixmap
        if pix and not pix.isNull():
            painter.drawPixmap(inner_rect, pix)
        painter.end()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(CARD_LAYOUT_SPACING)
        lm = CARD_LAYOUT_MARGINS
        layout.setContentsMargins(lm[0], lm[1], lm[2], lm[3])

        self.text_container = QWidget()
        self.text_container.setAttribute(Qt.WA_StyledBackground, True)
        self.text_container.setStyleSheet(f"""
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: {CARD_CONTAINER_RADIUS}px;
            padding: 4px;
        """)
        text_layout = QVBoxLayout(self.text_container)
        text_layout.setSpacing(CARD_TEXT_LAYOUT_SPACING)
        tm = CARD_TEXT_LAYOUT_MARGINS
        text_layout.setContentsMargins(tm[0], tm[1], tm[2], tm[3])

        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont(FAMILY_TITLE, CARD_TITLE_FONT_SIZE, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("color: #222; background: transparent; border: none;")

        self.desc_label = QLabel(self.description)
        self.desc_label.setFont(QFont(FAMILY_TITLE, CARD_DESC_FONT_SIZE))
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.desc_label.setStyleSheet("color: #444; background: transparent; border: none;")

        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.desc_label)
        layout.addWidget(self.text_container)
        layout.addStretch()

        if self.disabled:
            self.button = QPushButton("开发中")
            self.button.setEnabled(False)
            self.button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #e9ecef; color: #6c757d;
                    border: none; padding: 5px 12px; border-radius: {CARD_BUTTON_RADIUS}px;
                    font-size: 10px;
                }}
            """)
        else:
            self.button = QPushButton("启动")
            self.button.setCursor(QCursor(Qt.PointingHandCursor))
            self.button.clicked.connect(self.command)

        self.button.setFixedSize(CARD_BUTTON_WIDTH, CARD_BUTTON_HEIGHT)
        layout.addWidget(self.button, 0, Qt.AlignCenter)
        self.setFixedSize(CARD_WIDTH, CARD_HEIGHT)

    def setup_styles(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(CARD_SHADOW_BLUR)
        shadow.setOffset(CARD_SHADOW_OFFSET_X, CARD_SHADOW_OFFSET_Y)
        shadow.setColor(QColor(0, 0, 0, 30))
        self.setGraphicsEffect(shadow)

    def _apply_styles(self, hovered=False):
        if self.disabled:
            return
        bw = CARD_BORDER_WIDTH
        cr = CARD_RADIUS
        br = CARD_BUTTON_RADIUS

        if hovered:
            self.setStyleSheet(f"""
                AppCard {{
                    background-color: #ffffff;
                    border: {bw}px solid #495057;
                    border-radius: {cr}px;
                }}
            """)
            self.button.setStyleSheet(f"""
                QPushButton {{
                    background-color: white;
                    color: #222;
                    border: 1.5px solid #222;
                    padding: 5px 12px;
                    border-radius: {br}px;
                    font-size: 10px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #222;
                    color: white;
                }}
                QPushButton:pressed {{
                    background-color: #000;
                    border-color: #000;
                    color: white;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                AppCard {{
                    background-color: #f0f0f0;
                    border: {bw}px solid #ced4da;
                    border-radius: {cr}px;
                }}
            """)
            self.button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #e9ecef;
                    color: #222;
                    border: none;
                    padding: 5px 12px;
                    border-radius: {br}px;
                    font-size: 10px;
                }}
                QPushButton:hover {{
                    background-color: #dee2e6;
                }}
            """)

    def enterEvent(self, event):
        if not self.disabled:
            self._hover = True
            self.update()
            self._apply_styles(hovered=True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self.disabled:
            self._hover = False
            self.update()
            self._apply_styles(hovered=False)
        super().leaveEvent(event)


# ===================== 分类槽 =====================
class CategoryWidget(QFrame):
    def __init__(self, category_name, parent=None):
        super().__init__(parent)
        self.category_name = category_name
        self.setup_ui()
        radius = CATEGORY_RADIUS
        self.setStyleSheet(f"""
            CategoryWidget {{
                background-color: #e9ecef;
                border-top: 2px solid #adb5bd;
                border-left: 2px solid #adb5bd;
                border-bottom: 2px solid #ffffff;
                border-right: 2px solid #ffffff;
                border-radius: {radius}px;
            }}
        """)

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        lm = CATEGORY_LAYOUT_MARGINS
        main_layout.setContentsMargins(lm[0], lm[1], lm[2], lm[3])
        main_layout.setSpacing(CATEGORY_LAYOUT_SPACING)

        title_label = QLabel(self.category_name)
        title_label.setFont(QFont(FAMILY_TITLE, CATEGORY_TITLE_FONT_SIZE, QFont.Bold))
        title_label.setStyleSheet("color: #333; background: transparent; border: none;")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        title_label.setFixedWidth(CATEGORY_LABEL_WIDTH)
        title_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        main_layout.addWidget(title_label, 0, Qt.AlignVCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFixedHeight(CATEGORY_HEIGHT)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        # 移除触摸手势，避免桌面端按钮点击冲突
        # QScroller.grabGesture(...)

        self.cards_container = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_container)
        cm = CATEGORY_CARDS_MARGINS
        self.cards_layout.setContentsMargins(cm[0], cm[1], cm[2], cm[3])
        self.cards_layout.setSpacing(CATEGORY_CARDS_SPACING)
        self.cards_layout.addStretch()

        self.scroll_area.setWidget(self.cards_container)
        main_layout.addWidget(self.scroll_area, 1)

    def add_card(self, card: AppCard):
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)


# ===================== 主窗口 =====================
class ApplicationLauncher(QMainWindow):
    error_occurred = Signal(str)
    app_launched = Signal(object)

    def __init__(self):
        super().__init__()
        self.active_windows = []
        self.cards = []
        self._module_class_cache = {}   # 缓存已加载的类

        self.error_occurred.connect(self.show_error)
        self.app_launched.connect(self.handle_app_launch)

        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

        self.setup_ui()
        self.setup_apps()

    def setup_ui(self):
        self.setWindowTitle("绘图工具集")
        self.setMinimumSize(LAUNCHER_MIN_WIDTH, LAUNCHER_MIN_HEIGHT)
        self.resize(LAUNCHER_DEFAULT_WIDTH, LAUNCHER_DEFAULT_HEIGHT)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move((screen.width() - self.width()) // 2,
                  (screen.height() - self.height()) // 2)

        self.setStyleSheet("QMainWindow { background-color: #f8f9fa; }")

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        mm = LAUNCHER_MAIN_MARGINS
        main_layout.setContentsMargins(mm[0], mm[1], mm[2], mm[3])

        self.outer_scroll = QScrollArea()
        self.outer_scroll.setWidgetResizable(True)
        self.outer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.outer_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.scroll_content = QWidget()
        self.categories_layout = QVBoxLayout(self.scroll_content)
        self.categories_layout.setContentsMargins(0, 0, 0, 0)
        self.categories_layout.setSpacing(4)

        self.outer_scroll.setWidget(self.scroll_content)
        main_layout.addWidget(self.outer_scroll)

        info = QLabel("选择工具开始使用 | 绘图工具集 v1.0")
        info.setFont(QFont(FAMILY_TITLE, INFO_FONT_SIZE))
        info.setStyleSheet("color: #888;")
        info.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info)

    def _make_launcher(self, module_subpath, friendly_name):
        apps_dir = os.path.join(os.path.dirname(__file__), 'apps')
        module_path = os.path.join(apps_dir, module_subpath.replace('/', os.sep))
        return lambda: self.load_app_module(module_path, friendly_name)

    def _create_category(self, category_name, items):
        """创建分类槽并添加卡片"""
        cat_widget = CategoryWidget(category_name)
        self.categories_layout.addWidget(cat_widget)

        for title, desc, subpath, friendly in items:
            command = self._make_launcher(subpath, friendly)
            module_filename = os.path.basename(subpath)
            image_name = os.path.splitext(module_filename)[0] + ".png"
            image_path = os.path.join(os.path.dirname(__file__), "images", image_name)
            card = AppCard(title, desc, command, bg_image_path=image_path)
            self.cards.append(card)
            cat_widget.add_card(card)

    def setup_apps(self):
        app_definitions = [
            ("Fourier",      "绘制图形并查看其傅里叶变换",              "Math/Fourier.py", "二维傅里叶变换"),
            ("Arnold",       "可视化图像猫变换",                       "Math/Arnold.py", "猫变换"),
            ("Lloyd",        "用Voronoi图的Lloyd松弛过程模拟蜂巢结构的形成", "Math/Lloyd.py", "Lloyd松弛"),
            ("幅角图",       "在复平面上可视化复变函数",                "Math/Cplot.py", "辐角图"),
            ("旅商问题",     "使用遗传算法解决旅行商问题",              "Math/Traveler.py", "旅行商问题遗传算法"),
            ("Lorenz",       "三维奇异吸引子可视化",                    "Math/Lorenz.py", "洛伦兹吸引子"),
            ("Schrödinger",  "Runge-Kutta模拟二维量子波包演化过程",      "Physics/Schrödinger.py", "薛定谔方程"),
            ("Turing",       "反应扩散系统模拟",            "Physics/Turing.py", "反应扩散模型"),
            ("Ising",        "模拟二维伊辛模型",                       "Physics/Ising.py", "二维伊辛模型"),
            ("Maxwell",      "模拟气体分子速率分布",                    "Physics/Maxwell.py", "麦克斯韦分布律"),
            ("Navier-Stokes", "2DQ9格子Boltzmann模拟Karmen涡街",       "Physics/Navier-Stokes.py", "Navier-Stokes方程"),
            ("Quaternion Julia",     "三维Julia集可视化",              "Fractals/Quaternion Julia.py", "三维Julia集"),
            ("Mandelbrot-Julia", "复动力系统生成分形",                 "Fractals/Mandelbrot.py", "复动力系统"),
            ("Newton",       "牛顿迭代法生成分形",                     "Fractals/Newton.py", "牛顿迭代法分形"),
            ("Lindenmayer",  "林氏系统分形",                          "Fractals/Lindenmayer.py", "林氏系统"),
            ("IFS分形",      "迭代函数系统生成分形图案",                "Fractals/IFS.py", "IFS分形"),
            ("Logistic分形", "逻辑映射生成分形图案",                    "Fractals/Logistic.py", "逻辑映射分形"),
            ("Lenia",       "连续化元胞自动机",                         "Cellular Automata/Lenia.py", "连续生命游戏"),
            ("Conway",       "模拟和可视化生命游戏",                    "Cellular Automata/Conway.py", "元胞自动机"),
            ("Distill",    "神经网络元胞自动机",                        "Cellular Automata/Distill.py", "神经网络元胞自动机"),
        ]

        categories = {}
        for title, desc, subpath, friendly in app_definitions:
            category = subpath.split('/')[0]
            categories.setdefault(category, []).append((title, desc, subpath, friendly))

        # 优先按指定顺序显示
        category_order = ["Math", "Physics", "Fractals", "Cellular Automata"]
        for cat in category_order:
            if cat in categories:
                self._create_category(cat, categories[cat])

        # 其余分类按字母顺序排列
        for cat in sorted(categories.keys()):
            if cat not in category_order:
                self._create_category(cat, categories[cat])

        self.categories_layout.addStretch()

    def load_app_module(self, module_path, friendly_name):
        # 检查缓存
        if module_path in self._module_class_cache:
            self.app_launched.emit(self._module_class_cache[module_path])
            return

        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            if not os.path.exists(module_path):
                raise FileNotFoundError(f"找不到文件: {module_path}")

            # 生成安全的模块名
            module_hash = hashlib.md5(module_path.encode()).hexdigest()[:8]
            module_name = f"launcher_{module_hash}"

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError("无法创建模块规格")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 查找属于该模块的 QMainWindow 子类
            app_class = None
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, QMainWindow) and
                    obj is not QMainWindow and
                    obj.__module__ == module.__name__):
                    app_class = obj
                    break

            if app_class is None:
                raise ImportError("模块中未找到 QMainWindow 子类")

            self._module_class_cache[module_path] = app_class
            self.app_launched.emit(app_class)
        except Exception as e:
            logging.exception(f"加载模块失败: {friendly_name}")
            self.error_occurred.emit(f"无法启动 {friendly_name}: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def handle_app_launch(self, app_class):
        try:
            window = app_class()
            window.show()
            self.active_windows.append(window)
            window.destroyed.connect(lambda w=window: self._remove_active_window(w))
        except Exception as e:
            logging.exception("创建应用窗口出错")
            self.show_error(f"创建应用窗口时出错: {str(e)}")

    def _remove_active_window(self, window):
        if window in self.active_windows:
            self.active_windows.remove(window)

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
    for root, dirs, files in os.walk(apps_dir):
        for d in dirs:
            full_path = os.path.join(root, d)
            if full_path not in sys.path:
                sys.path.append(full_path)

    app = QApplication(sys.argv)
    app.setFont(QFont(FAMILY_DEFAULT, 9))
    launcher = ApplicationLauncher()
    launcher.show()
    sys.exit(app.exec())