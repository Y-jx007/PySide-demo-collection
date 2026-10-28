import os
import sys
import time
import shutil
import subprocess
import importlib.util
import logging
import json

from PySide6.QtCore import Qt, QTimer, QSize, Signal, QMimeData, QPoint, QEvent, QVariantAnimation
from PySide6.QtGui import QFont, QFontMetrics, QCursor, QPainter, QPixmap, QImage, QImageReader, QDrag
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QMessageBox,
    QSizePolicy
)

from apps_manifest import APPS, CATEGORY_ORDER, INSTALL_HINTS
from manifest_edit import rewrite_app_entry

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
CARD_LAYOUT_MARGINS = (8, 8, 8, 8)
CARD_TEXT_LAYOUT_SPACING = 2
CARD_TEXT_LAYOUT_MARGINS = (4, 4, 4, 4)

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
INFO_DEFAULT = "选择工具开始使用 | 拖动卡片可排序或换分类 | 绘图工具集 v1.1"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = os.path.join(PROJECT_DIR, "apps")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
THUMBNAIL_DIR = os.path.join(ASSETS_DIR, "thumbnails")
RUNTIME_DIR = os.path.join(PROJECT_DIR, "runtime")

FAMILY_TITLE = "Segoe UI"
FAMILY_DEFAULT = "Microsoft YaHei"
APP_CARD_MIME = "application/x-pyside-app-card"
LAYOUT_PATH = os.path.join(RUNTIME_DIR, "launcher_layout.json")


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
    def __init__(self, title, description, command, bg_image_path=None,
                 disabled=False, parent=None, app_path=None):
        super().__init__(parent)
        self.title = title
        self.description = description
        self.command = command
        self.disabled = disabled
        self.app_path = app_path
        self.category_widget = None
        self._drag_start = None
        self.setCursor(Qt.OpenHandCursor)
        self.setAcceptDrops(True)
        self._color_pixmap = None
        self._gray_pixmap = None
        self._bg_image_path = bg_image_path
        self._image_loaded = False
        self._hover = False
        self.setup_ui()
        self.setup_styles()
        self._apply_styles(hovered=False)

    def _load_background(self, image_path=None):
        """延迟到首次绘制再读图；启动时不再解码 32 张缩略图。"""
        if self._image_loaded:
            return
        self._image_loaded = True
        image_path = image_path or self._bg_image_path
        if not image_path or not os.path.exists(image_path):
            return
        inner_w = CARD_WIDTH - 2 * CARD_BORDER_WIDTH
        inner_h = CARD_HEIGHT - 2 * CARD_BORDER_WIDTH
        try:
            reader = QImageReader(image_path)
            reader.setAutoTransform(True)
            img = reader.read()
            if img.isNull():
                logging.warning(f"无法加载背景图片 {image_path}: {reader.errorString()}")
                return
            img = img.scaled(inner_w, inner_h, Qt.KeepAspectRatioByExpanding,
                             Qt.SmoothTransformation)
            x = max(0, (img.width() - inner_w) // 2)
            y = max(0, (img.height() - inner_h) // 2)
            img = img.copy(x, y, inner_w, inner_h)
            self._color_pixmap = QPixmap.fromImage(img)
            gray = img.convertToFormat(QImage.Format_Grayscale8)
            self._gray_pixmap = QPixmap.fromImage(gray)
        except Exception as e:
            logging.warning(f"无法加载背景图片 {image_path}: {e}")

    def paintEvent(self, event):
        if not self._image_loaded:
            self._load_background()
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
        self.text_container.setAttribute(Qt.WA_TransparentForMouseEvents, True)
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
        self.title_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self.desc_label = QLabel(self.description)
        self.desc_label.setFont(QFont(FAMILY_TITLE, CARD_DESC_FONT_SIZE))
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.desc_label.setStyleSheet("color: #444; background: transparent; border: none;")
        self.desc_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

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
        # 每张卡片一个 QGraphicsDropShadowEffect 会显著拖慢首屏；
        # 边框已经足够区分卡片，阴影改为不创建。
        self.setAttribute(Qt.WA_StyledBackground, True)

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

    def dragEnterEvent(self, event):
        if self.category_widget is not None:
            self.category_widget.dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if self.category_widget is not None:
            self.category_widget._drop_event_obj = self
            try:
                self.category_widget.dragMoveEvent(event)
            finally:
                self.category_widget._drop_event_obj = None

    def dragLeaveEvent(self, event):
        if self.category_widget is not None:
            self.category_widget.dragLeaveEvent(event)

    def dropEvent(self, event):
        if self.category_widget is None:
            event.ignore()
            return
        self.category_widget._drop_event_obj = self
        try:
            self.category_widget.dropEvent(event)
        finally:
            self.category_widget._drop_event_obj = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is None or not (event.buttons() & Qt.LeftButton):
            return
        if (event.position().toPoint() - self._drag_start).manhattanLength() < QApplication.startDragDistance():
            return
        self._drag_start = None
        if not self.app_path:
            return
        pix = self.grab()
        launcher = self.window()
        if not hasattr(launcher, "begin_card_drag") or not launcher.begin_card_drag(self):
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(APP_CARD_MIME, self.app_path.encode("utf-8"))
        drag.setMimeData(mime)
        drag.setPixmap(pix)
        drag.setHotSpot(QPoint(pix.width() // 2, pix.height() // 2))
        drag.exec(Qt.MoveAction)
        if not launcher.card_drag_finished():
            launcher.cancel_card_drag()

    def mouseReleaseEvent(self, event):
        self._drag_start = None
        super().mouseReleaseEvent(event)

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
    card_dropped = Signal(object, int)

    def __init__(self, category_name, parent=None):
        super().__init__(parent)
        self.category_name = category_name
        self.setAcceptDrops(True)
        self.setup_ui()
        self._apply_style(False)

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
        # 透明背景只作用在 viewport 上：给 QScrollArea 设样式表会让滚动条
        # 改用 QStyleSheetStyle 绘制，失去系统默认外观
        self.scroll_area.viewport().setAutoFillBackground(False)

        # 移除触摸手势，避免桌面端按钮点击冲突
        # QScroller.grabGesture(...)

        self.cards_container = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_container)
        cm = CATEGORY_CARDS_MARGINS
        self.cards_layout.setContentsMargins(cm[0], cm[1], cm[2], cm[3])
        self.cards_layout.setSpacing(CATEGORY_CARDS_SPACING)
        self.cards_layout.addStretch()

        self.scroll_area.setWidget(self.cards_container)
        for drop_target in (self.scroll_area, self.scroll_area.viewport(), self.cards_container):
            drop_target.setAcceptDrops(True)
            drop_target.installEventFilter(self)
        main_layout.addWidget(self.scroll_area, 1)

    def _apply_style(self, active):
        radius = CATEGORY_RADIUS
        border = "#495057" if active else "#adb5bd"
        background = "#f1f3f5" if active else "#e9ecef"
        self.setStyleSheet(f"""
            CategoryWidget {{
                background-color: {background};
                border-top: 2px solid {border};
                border-left: 2px solid {border};
                border-bottom: 2px solid #ffffff;
                border-right: 2px solid #ffffff;
                border-radius: {radius}px;
            }}
        """)

    def card_count(self):
        return sum(
            1 for i in range(self.cards_layout.count())
            if isinstance(self.cards_layout.itemAt(i).widget(), AppCard)
        )

    def remove_card(self, card):
        self.cards_layout.removeWidget(card)
        card.category_widget = None
        QTimer.singleShot(0, self._update_height)

    def _insert_position(self, card_index):
        """Layout index before the card_index-th AppCard."""
        seen = 0
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            w = item.widget() if item else None
            if isinstance(w, AppCard):
                if seen == card_index:
                    return i
                seen += 1
        return max(0, self.cards_layout.count() - 1)

    def insert_card(self, card, index):
        index = max(0, min(int(index), self.card_count()))
        pos = self._insert_position(index)
        self.cards_layout.insertWidget(pos, card, 0, Qt.AlignTop)
        card.category_widget = self
        QTimer.singleShot(0, self._update_height)

    def insert_placeholder(self, placeholder, index):
        index = max(0, min(int(index), self.card_count()))
        pos = self._insert_position(index)
        self.cards_layout.insertWidget(pos, placeholder, 0, Qt.AlignTop)
        QTimer.singleShot(0, self._update_height)

    def remove_placeholder(self, placeholder):
        if self.cards_layout.indexOf(placeholder) >= 0:
            self.cards_layout.removeWidget(placeholder)
            placeholder.setParent(None)
            QTimer.singleShot(0, self._update_height)

    def placeholder_card_index(self, placeholder):
        seen = 0
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            w = item.widget() if item else None
            if w is placeholder:
                return seen
            if isinstance(w, AppCard):
                seen += 1
        return seen

    def _drop_index(self, event, source=None):
        source = source or self
        global_pos = source.mapToGlobal(event.position().toPoint())
        local = self.cards_container.mapFromGlobal(global_pos)
        x = local.x()
        index = 0
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            card = item.widget() if item else None
            if isinstance(card, AppCard) and x > card.geometry().center().x():
                index += 1
        return index

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(APP_CARD_MIME):
            event.acceptProposedAction()
            self._apply_style(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(APP_CARD_MIME):
            event.acceptProposedAction()
            handler = getattr(self.window(), "update_drag_target", None)
            if handler is not None:
                handler(self, event)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._apply_style(False)
        if not self.rect().contains(self.mapFromGlobal(QCursor.pos())):
            handler = getattr(self.window(), "leave_drag_target", None)
            if handler is not None:
                handler(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self._apply_style(False)
        if not event.mimeData().hasFormat(APP_CARD_MIME):
            event.ignore()
            return
        path = bytes(event.mimeData().data(APP_CARD_MIME)).decode("utf-8")
        card = self.window().card_for_path(path)
        if card is None:
            event.ignore()
            return
        self.card_dropped.emit(card, self._drop_index(event, getattr(self, "_drop_event_obj", None)))
        event.acceptProposedAction()

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
        card.category_widget = self
        QTimer.singleShot(0, self._update_height)


# ===================== 主窗口 =====================
class ApplicationLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.active_processes = []   # 每个 app 一个独立子进程
        self.cards = []
        self.cards_by_path = {}
        self.category_widgets = []
        self.category_by_name = {}
        self._drag_card = None
        self._drag_source_cat = None
        self._drag_source_index = -1
        self._placeholder = None
        self._placeholder_cat = None
        self._placeholder_anim = None
        self._drag_done = False
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
        self.outer_scroll.setFrameShape(QFrame.NoFrame)
        self.outer_scroll.viewport().setAutoFillBackground(False)

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

    def _new_category(self, category_name):
        existing = self.category_by_name.get(category_name)
        if existing is not None:
            return existing
        cat_widget = CategoryWidget(category_name)
        cat_widget.card_dropped.connect(self._on_card_dropped)
        self.categories_layout.addWidget(cat_widget)
        self.category_widgets.append(cat_widget)
        self.category_by_name[category_name] = cat_widget
        return cat_widget

    def _add_card(self, cat_widget, app):
        if app["path"] in self.cards_by_path:
            return
        command = self._make_launcher(app["path"], app["friendly"])
        module_filename = os.path.basename(app["path"])
        image_name = os.path.splitext(module_filename)[0] + ".png"
        image_path = os.path.join(THUMBNAIL_DIR, image_name)
        card = AppCard(
            app["title"], app["description"], command,
            bg_image_path=image_path, app_path=app["path"],
        )
        self.cards.append(card)
        self.cards_by_path[app["path"]] = card
        cat_widget.add_card(card)

    def _create_category(self, category_name, items):
        """创建分类槽并添加卡片"""
        cat_widget = self._new_category(category_name)
        for app in items:
            self._add_card(cat_widget, app)
        return cat_widget

    def card_for_path(self, path):
        return self.cards_by_path.get(path)

    def _load_layout(self):
        try:
            with open(LAYOUT_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            categories = data.get("categories", [])
            if not isinstance(categories, list):
                return None
            out = []
            for entry in categories:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                apps = entry.get("apps", [])
                if isinstance(name, str) and isinstance(apps, list):
                    out.append((name, [p for p in apps if isinstance(p, str)]))
            return out or None
        except (OSError, ValueError):
            return None

    def _save_layout(self):
        data = {"version": 1, "categories": []}
        for cat in self.category_widgets:
            paths = []
            for i in range(cat.card_count()):
                item = cat.cards_layout.itemAt(i)
                card = item.widget() if item else None
                if card is not None and card.app_path:
                    paths.append(card.app_path)
            data["categories"].append({"name": cat.category_name, "apps": paths})
        tmp = LAYOUT_PATH + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, LAYOUT_PATH)
        except OSError as exc:
            logging.warning(f"无法保存启动器布局: {exc}")
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _make_placeholder(self):
        ph = QFrame()
        ph.setFixedSize(CARD_WIDTH, CARD_HEIGHT)
        ph.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        ph.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(255, 255, 255, 170);
                border: 2px dashed #868e96;
                border-radius: {CARD_RADIUS}px;
            }}
        """)
        return ph

    def _animate_placeholder(self):
        if self._placeholder is None:
            return
        ph = self._placeholder
        if self._placeholder_anim is not None:
            self._placeholder_anim.stop()
        ph.setFixedWidth(0)
        anim = QVariantAnimation(self)
        anim.setDuration(130)
        anim.setStartValue(0)
        anim.setEndValue(CARD_WIDTH)
        anim.valueChanged.connect(lambda value: ph.setFixedWidth(int(value)))
        anim.finished.connect(lambda: ph.setFixedWidth(CARD_WIDTH))
        self._placeholder_anim = anim
        anim.start()

    def _remove_placeholder(self):
        if self._placeholder_anim is not None:
            self._placeholder_anim.stop()
            self._placeholder_anim = None
        if self._placeholder_cat is not None and self._placeholder is not None:
            self._placeholder_cat.remove_placeholder(self._placeholder)
            self._placeholder_cat = None

    def begin_card_drag(self, card):
        if self._drag_card is not None or card.category_widget is None:
            return False
        source = card.category_widget
        idx = source.cards_layout.indexOf(card)
        self._drag_card = card
        self._drag_source_cat = source
        self._drag_source_index = idx
        self._drag_done = False
        self._placeholder = self._make_placeholder()
        self._placeholder_cat = None
        card.hide()
        source.remove_card(card)
        self.info_label.setText(f"拖动 {card.title}…")
        return True

    def update_drag_target(self, category, event):
        if self._drag_card is None or self._placeholder is None:
            return
        index = category._drop_index(event, getattr(category, "_drop_event_obj", None))
        if self._placeholder_cat is not category:
            if self._placeholder_cat is not None:
                self._placeholder_cat.remove_placeholder(self._placeholder)
            category.insert_placeholder(self._placeholder, index)
            self._placeholder_cat = category
            self._placeholder.show()
            self._animate_placeholder()
        else:
            if category.placeholder_card_index(self._placeholder) != index:
                category.remove_placeholder(self._placeholder)
                category.insert_placeholder(self._placeholder, index)
                self._placeholder.show()

    def leave_drag_target(self, category):
        if self._placeholder_cat is category:
            self._remove_placeholder()

    def _clear_card_drag(self):
        if self._placeholder_anim is not None:
            self._placeholder_anim.stop()
            self._placeholder_anim = None
        self._drag_card = None
        self._drag_source_cat = None
        self._drag_source_index = -1
        self._placeholder = None
        self._placeholder_cat = None
        self._drag_done = True

    def card_drag_finished(self):
        return self._drag_done

    def cancel_card_drag(self):
        if self._drag_card is None:
            return
        card = self._drag_card
        source = self._drag_source_cat
        index = self._drag_source_index
        self._remove_placeholder()
        if source is not None:
            source.insert_card(card, index)
        card.show()
        self._clear_card_drag()
        self._restore_info()

    def _relocate_app_file(self, card, target):
        """跨分类拖卡片时，把对应的 .py 挪到目标分类目录并同步清单。

        返回 False 表示这次拖动要放弃（文件动不了 / 正在运行 / 同名冲突）。
        """
        if not card.app_path or target is None:
            return True
        old_rel = card.app_path
        category = (target.category_name or "").strip()
        old_dir = old_rel.split("/")[0] if "/" in old_rel else ""
        if category == old_dir:
            return True
        if not category or category in (".", "..") or any(
                ch in category for ch in '\\/:*?"<>|'):
            self.show_error(f"分类名「{category}」不能用作文件夹名，文件保持原位。")
            return False

        name = os.path.basename(old_rel)
        new_rel = f"{category}/{name}"
        old_abs = os.path.join(APPS_DIR, old_rel.replace("/", os.sep))
        new_abs = os.path.join(APPS_DIR, new_rel.replace("/", os.sep))
        if not os.path.isfile(old_abs):
            self.show_error(f"找不到应用文件，卡片先留在原分类：\n{old_abs}")
            return False
        if os.path.exists(new_abs):
            self.show_error(f"目标分类里已经有同名文件：\n{new_abs}")
            return False
        for entry in self.active_processes:
            if (os.path.normcase(entry["path"]) == os.path.normcase(old_abs)
                    and entry["process"].poll() is None):
                self.show_error(f"{card.title} 正在运行，先关掉它再换分类。")
                return False

        try:
            os.makedirs(os.path.dirname(new_abs), exist_ok=True)
            shutil.move(old_abs, new_abs)
        except OSError as exc:
            self.show_error(f"移动文件失败：{exc}")
            return False
        if not rewrite_app_entry(old_rel, new_rel, category):
            try:
                shutil.move(new_abs, old_abs)
            except OSError:
                pass
            self.show_error("写回 apps_manifest.py 失败，文件已移回原处。")
            return False

        app = None
        for entry in APPS:
            if entry.get("path") == old_rel:
                entry["path"] = new_rel
                entry["category"] = category
                app = entry
                break
        for entry in self.active_processes:
            if os.path.normcase(entry["path"]) == os.path.normcase(old_abs):
                entry["path"] = new_abs
        self.cards_by_path.pop(old_rel, None)
        card.app_path = new_rel
        self.cards_by_path[new_rel] = card
        try:
            card.button.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        card.command = self._make_launcher(new_rel, (app or {}).get("friendly", card.title))
        card.button.clicked.connect(card.command)
        return True

    def _finish_card_drag(self, target, index=None):
        if self._drag_card is None or target is None:
            self.cancel_card_drag()
            return
        card = self._drag_card
        if self._placeholder is not None and self._placeholder_cat is target:
            index = target.placeholder_card_index(self._placeholder)
        self._remove_placeholder()
        if not self._relocate_app_file(card, target):
            self.cancel_card_drag()
            return
        index = max(0, min(int(index or 0), target.card_count()))
        target.insert_card(card, index)
        card.show()
        self._save_layout()
        self.info_label.setText(f"已移动 {card.title} 到「{target.category_name}」")
        self._clear_card_drag()
        QTimer.singleShot(2200, self._restore_info)

    def _on_card_dropped(self, card, index):
        target = self.sender()
        if isinstance(target, CategoryWidget):
            self._finish_card_drag(target, index)

    def _move_card(self, card, target, index):
        # kept for programmatic moves/tests
        source = card.category_widget
        if source is None or target is None:
            return
        if not self._relocate_app_file(card, target):
            return
        source_index = source.cards_layout.indexOf(card)
        if source is target and 0 <= source_index < index:
            index -= 1
        source.remove_card(card)
        target.insert_card(card, index)
        card.category_widget = target
        self._save_layout()

    def setup_apps(self):
        app_map = {app["path"]: app for app in APPS}
        custom = self._load_layout()
        seen = set()
        if custom:
            for name, paths in custom:
                cat = self._new_category(name)
                for path in paths:
                    app = app_map.get(path)
                    if app is not None and path not in seen:
                        self._add_card(cat, app)
                        seen.add(path)

        remaining = [app for app in APPS if app["path"] not in seen]
        categories = {}
        for app in remaining:
            categories.setdefault(app["category"], []).append(app)

        # 优先按指定顺序显示；自定义布局里的分类保持原有顺序
        for cat_name in CATEGORY_ORDER:
            if cat_name in categories:
                cat = self._new_category(cat_name)
                for app in categories[cat_name]:
                    self._add_card(cat, app)

        for cat_name in sorted(categories.keys()):
            if cat_name in CATEGORY_ORDER:
                continue
            cat = self._new_category(cat_name)
            for app in categories[cat_name]:
                self._add_card(cat, app)

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

        # 子进程输出写入 runtime/logs：比管道更稳（打包后无控制台时也能查错）
        log_dir = os.path.join(RUNTIME_DIR, "logs")
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