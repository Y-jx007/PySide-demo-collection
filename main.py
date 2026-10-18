import os
import sys
import time
import subprocess
import importlib.util
import logging
import io

from PIL import Image, ImageOps

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QCursor, QPainter, QPixmap, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QMessageBox,
    QSizePolicy, QGraphicsDropShadowEffect
)

from apps_manifest import APPS, CATEGORY_ORDER, INSTALL_HINTS

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

CATEGORY_HEIGHT = CARD_HEIGHT + 8  # 卡片 + 上下 4px 内边距；滚动条高度另算
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
INFO_DEFAULT = "选择工具开始使用 | 绘图工具集 v1.1"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = os.path.join(PROJECT_DIR, "apps")

FAMILY_TITLE = "Segoe UI"
FAMILY_DEFAULT = "Microsoft YaHei"


class HScrollArea(QScrollArea):
    """横向滚动区域：窗口变窄时用滚轮/滚动条仍可访问全部卡片。"""

    def wheelEvent(self, event):
        bar = self.horizontalScrollBar()
        wants_horizontal = bool(event.modifiers() & Qt.ShiftModifier) or (
            abs(event.angleDelta().x()) > abs(event.angleDelta().y())
        )
        if wants_horizontal and bar.maximum() > bar.minimum():
            delta = event.angleDelta().y() or event.angleDelta().x()
            if delta:
                bar.setValue(bar.value() - delta)
                event.accept()
                return
        super().wheelEvent(event)


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

        self.scroll_area = HScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
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

    def _content_height(self):
        """分类行只占卡片实际高度；有横向滚动条时再补上滚动条高度。"""
        height = CARD_HEIGHT + CATEGORY_CARDS_MARGINS[1] + CATEGORY_CARDS_MARGINS[3]
        bar = self.scroll_area.horizontalScrollBar()
        if bar.maximum() > bar.minimum():
            height += max(bar.sizeHint().height(), 16)
        return height

    def _update_height(self):
        self.scroll_area.setFixedHeight(self._content_height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_height()

    def add_card(self, card: AppCard):
        # 顶部对齐，避免 QHBoxLayout 把 150px 卡片在 188px 行里垂直居中
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card, 0, Qt.AlignTop)
        QTimer.singleShot(0, self._update_height)


# ===================== 主窗口 =====================
class ApplicationLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.active_processes = []   # 每个 app 一个独立子进程
        self.cards = []
        self._closing = False

        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._poll_processes)
        self._poll_timer.start()

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

        info = QLabel(INFO_DEFAULT)
        info.setFont(QFont(FAMILY_TITLE, INFO_FONT_SIZE))
        info.setStyleSheet("color: #888;")
        info.setAlignment(Qt.AlignCenter)
        self.info_label = info
        main_layout.addWidget(info)

    def _make_launcher(self, module_subpath, friendly_name):
        module_path = os.path.join(APPS_DIR, module_subpath.replace('/', os.sep))
        return lambda: self.launch_app(module_path, friendly_name)

    def _create_category(self, category_name, items):
        """创建分类槽并添加卡片"""
        cat_widget = CategoryWidget(category_name)
        self.categories_layout.addWidget(cat_widget)

        for app in items:
            command = self._make_launcher(app["path"], app["friendly"])
            module_filename = os.path.basename(app["path"])
            image_name = os.path.splitext(module_filename)[0] + ".png"
            image_path = os.path.join(PROJECT_DIR, "images", image_name)
            card = AppCard(app["title"], app["description"], command, bg_image_path=image_path)
            self.cards.append(card)
            cat_widget.add_card(card)

    def setup_apps(self):
        categories = {}
        for app in APPS:
            categories.setdefault(app["category"], []).append(app)

        # 优先按指定顺序显示
        for cat in CATEGORY_ORDER:
            if cat in categories:
                self._create_category(cat, categories[cat])

        # 其余分类按名称排序
        for cat in sorted(categories.keys()):
            if cat not in CATEGORY_ORDER:
                self._create_category(cat, categories[cat])

        self.categories_layout.addStretch()

    def _manifest_for_path(self, module_path):
        target = os.path.normcase(os.path.abspath(module_path))
        for app in APPS:
            candidate = os.path.normcase(os.path.abspath(
                os.path.join(APPS_DIR, app["path"].replace("/", os.sep))))
            if candidate == target:
                return app
        return None

    def _missing_dependencies(self, app):
        missing = []
        for dep in app.get("requires", []):
            try:
                if importlib.util.find_spec(dep) is None:
                    missing.append(dep)
            except (ImportError, ValueError):
                missing.append(dep)
        return missing

    def launch_app(self, module_path, friendly_name):
        if not os.path.exists(module_path):
            self.show_error(f"找不到应用文件:\n{module_path}")
            return

        app = self._manifest_for_path(module_path)
        missing = self._missing_dependencies(app) if app else []
        if missing:
            hints = "\n".join(f"{dep}: {INSTALL_HINTS.get(dep, 'pip install ' + dep)}" for dep in missing)
            QMessageBox.warning(
                self, "缺少依赖",
                f"{friendly_name} 需要以下依赖：\n{hints}\n\n安装后再启动。"
            )
            return

        for entry in self.active_processes:
            if entry["path"] == module_path and entry["process"].poll() is None:
                QMessageBox.information(self, "已在运行", f"{friendly_name} 已经在运行。")
                return

        runner = os.path.join(PROJECT_DIR, "app_runner.py")
        if not os.path.exists(runner):
            self.show_error(f"找不到启动器入口:\n{runner}")
            return

        # 子进程输出写入 logs/：比管道更稳（打包后无控制台时也能查错）
        log_dir = os.path.join(PROJECT_DIR, "logs")
        base = os.path.splitext(os.path.basename(module_path))[0]
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in base)
        out_log = os.path.join(log_dir, f"{safe}.out.log")
        error_log = os.path.join(log_dir, f"{safe}.err.log")
        handles = []
        try:
            os.makedirs(log_dir, exist_ok=True)
            out_handle = open(out_log, "w", encoding="utf-8", errors="replace")
            handles.append(out_handle)
            err_handle = open(error_log, "w", encoding="utf-8", errors="replace")
            handles.append(err_handle)
        except OSError:
            for handle in handles:
                handle.close()
            handles = []
            out_handle = err_handle = subprocess.DEVNULL
            out_log = error_log = None

        creationflags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        try:
            proc = subprocess.Popen(
                [sys.executable, runner, module_path],
                cwd=PROJECT_DIR,
                stdout=out_handle,
                stderr=err_handle,
                creationflags=creationflags,
            )
        except OSError as exc:
            self.show_error(f"无法启动 {friendly_name}:\n{exc}")
            return
        finally:
            for handle in handles:
                handle.close()

        self.active_processes.append({
            "process": proc,
            "path": module_path,
            "friendly": friendly_name,
            "error_log": error_log,
        })
        self.info_label.setText(f"正在启动 {friendly_name}…（首次启动可能需要几秒）")
        QTimer.singleShot(6000, self._restore_info)

    def _poll_processes(self):
        changed = False
        for entry in list(self.active_processes):
            code = entry["process"].poll()
            if code is None:
                continue
            self.active_processes.remove(entry)
            changed = True
            if self._closing:
                continue
            if code != 0:
                hint = f"\n\n错误日志:\n{entry['error_log']}" if entry.get("error_log") else ""
                self.show_error(f"{entry['friendly']} 异常退出（代码 {code}）。{hint}")
        if changed:
            self._restore_info()

    def _restore_info(self):
        if self._closing:
            return
        if self.active_processes:
            self.info_label.setText(
                f"已启动 {len(self.active_processes)} 个工具 | 关闭启动器会一并关闭它们"
            )
        else:
            self.info_label.setText(INFO_DEFAULT)

    def show_error(self, message):
        QMessageBox.critical(self, "错误", message)

    def closeEvent(self, event):
        self._closing = True
        self._poll_timer.stop()
        entries = list(self.active_processes)
        for entry in entries:
            proc = entry["process"]
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass

        deadline = time.monotonic() + 3.0
        for entry in entries:
            proc = entry["process"]
            try:
                proc.wait(timeout=max(0.05, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass
        self.active_processes.clear()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont(FAMILY_DEFAULT, 9))
    launcher = ApplicationLauncher()
    launcher.show()
    sys.exit(app.exec())