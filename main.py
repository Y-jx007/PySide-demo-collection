import os
import sys
import time
import shutil
import subprocess
import importlib.util
import logging
import json
import math
import random

# PyInstaller one-folder: the launcher is the exe itself, so a child app cannot
# be started as `python app_runner.py`; re-enter this exe in child mode instead.
if len(sys.argv) >= 3 and sys.argv[1] == "--child":
    from app_runner import main as _run_child_app

    sys.argv = [sys.argv[0], sys.argv[2], *sys.argv[3:]]
    _run_child_app()
    raise SystemExit(0)

from PySide6.QtCore import (
    Qt, QTimer, Signal, QMimeData, QPoint, QPointF, QEvent, QSize, QUrl,
    QVariantAnimation, QRectF, QPropertyAnimation, QEasingCurve,
)
from PySide6.QtGui import (
    QFont, QFontMetrics, QCursor, QPainter, QPixmap, QImage, QImageReader, QDrag,
    QColor, QPainterPath, QLinearGradient, QRadialGradient, QPen, QBrush, QIcon,
    QDesktopServices,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QScrollBar, QFrame, QMessageBox,
)

from apps_manifest import APPS, CATEGORY_ORDER, INSTALL_HINTS, category_display
from common.app import READY_ENV
from manifest_edit import rewrite_app_entry

# ===================== 全局常量 =====================
CARD_WIDTH = 150
CARD_HEIGHT = 150
CARD_BUTTON_WIDTH = 100
CARD_BUTTON_HEIGHT = 26

CARD_TITLE_FONT_SIZE = 11
CARD_DESC_FONT_SIZE = 8
CARD_REST_EXPOSURE = 0.18

CARD_RADIUS = 3
CARD_CONTAINER_RADIUS = 2
CARD_BUTTON_RADIUS = 2

CARD_LAYOUT_SPACING = 4
CARD_LAYOUT_MARGINS = (5, 5, 5, 5)
CARD_TEXT_LAYOUT_SPACING = 2
CARD_TEXT_LAYOUT_MARGINS = (6, 4, 6, 4)

CARD_BORDER_WIDTH = 1
CARD_MOUNT = 2                      # 卡片周围露出的浅木框宽度

# ---- 纸质陈列柜外观 ----
# 每个分类是一只横向陈列柜：木框 + 黄铜分类牌 + 深色展柜内衬 + 玻璃反光 + 搁板。
CABINET_RAIL = 2                    # 纸片上下留白（很薄）
CABINET_LEFT = 36                   # 左侧牛皮纸标签宽度
CABINET_RADIUS = 3                  # 纸片圆角
CABINET_HEIGHT = CARD_HEIGHT + 2 * CABINET_RAIL   # 154
CABINET_CARD_GAP = 8
CABINET_CARD_MARGINS = (4, 0, 4, 0)
CABINET_LABEL_SIZE = 9
CABINET_NAV_SIZE = 24

# 纸张：米白纸片、牛皮纸标签、墨色字、朱砂印章
PAPER_ROW = "#f8f2e7"
PAPER_TAB = "#e6d3a8"
PAPER_INK = QColor("#3b352b")
PAPER_STAMP = QColor("#a55240")

# 页面配色：暖石色纸面，顶部有一点漫射光
PAGE_BG = "#e9e2d4"
PAGE_TOP = "#f3ede1"
PAGE_BOTTOM = "#e5ddcd"
PAGE_MUTED = "#6f6658"

# 卡片：暖灰片格 + 象牙纸标签
CARD_BG = "#e4d8c4"
CARD_BG_HOVER = "#efe4d2"
CARD_FRAME = "rgba(70, 60, 45, 0.28)"
CARD_FRAME_HOVER = "rgba(70, 60, 45, 0.50)"
CARD_INK = "#2f2a22"
CARD_MUTED = "#5f5748"
CARD_CAPTION = "rgba(250, 246, 238, 0.94)"
CARD_CAPTION_HOVER = "rgba(253, 250, 244, 0.98)"
CARD_CAPTION_BORDER = "rgba(70, 60, 45, 0.20)"
CARD_CAPTION_BORDER_HOVER = "rgba(70, 60, 45, 0.36)"
CARD_BUTTON_BG = "rgba(250, 246, 238, 0.96)"
CARD_BUTTON_INK = "#2f2a22"
CARD_BUTTON_BORDER = "rgba(70, 60, 45, 0.28)"

# 兼容旧名（分类面板现在固定为陈列柜高度）
CATEGORY_HEIGHT = CABINET_HEIGHT
CATEGORY_LABEL_WIDTH = CABINET_LEFT
CATEGORY_LAYOUT_MARGINS = (CABINET_LEFT, CABINET_RAIL, 8, CABINET_RAIL)
CATEGORY_LAYOUT_SPACING = 0
CATEGORY_CARDS_SPACING = CABINET_CARD_GAP
CATEGORY_CARDS_MARGINS = CABINET_CARD_MARGINS
CATEGORY_TITLE_FONT_SIZE = CABINET_LABEL_SIZE
CATEGORY_RADIUS = CABINET_RADIUS

LAUNCHER_MIN_WIDTH = 640
LAUNCHER_MIN_HEIGHT = 480
LAUNCHER_DEFAULT_WIDTH = 1200
LAUNCHER_DEFAULT_HEIGHT = 800
LAUNCHER_MAIN_MARGINS = (24, 10, 24, 4)
LAUNCHER_CATEGORY_SPACING = 3

APP_TITLE = "绘图工具集"
APP_VERSION = "v1.1"
INFO_DEFAULT = "左键拖动浏览陈列柜  ·  右键拖动卡片换位"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = os.path.join(PROJECT_DIR, "apps")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
APP_ICON_PATH = os.path.join(ASSETS_DIR, "app_icon.ico")
THUMBNAIL_DIR = os.path.join(ASSETS_DIR, "thumbnails")
RUNTIME_DIR = os.path.join(PROJECT_DIR, "runtime")

FAMILY_TITLE = "Segoe UI"
FAMILY_DEFAULT = "Microsoft YaHei"
APP_CARD_MIME = "application/x-pyside-app-card"
LAYOUT_PATH = os.path.join(RUNTIME_DIR, "launcher_layout.json")
READY_DIR = os.path.join(RUNTIME_DIR, "ready")

# 子进程日志：超过这个大小就保留一份 .prev，避免无限增长，也不至于覆盖上次崩溃记录。
LOG_ROTATE_BYTES = 2 * 1024 * 1024
# 超过这个时间还没收到就绪信号，就在状态条里说明“正在编译内核”。
READY_SLOW_SECONDS = 8.0
# 超过这个时间仍未就绪且进程还活着：提示去看日志（不再假装已启动）。
READY_TIMEOUT_SECONDS = 180.0


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

        # 竖直滚轮留给外层页面：否则鼠标停在卡片上时整页滚不动。
        delta = event.pixelDelta().y() or event.angleDelta().y()
        outer = self._outer_scroll_area()
        if delta and outer is not None:
            outer_bar = outer.verticalScrollBar()
            if outer_bar.maximum() > outer_bar.minimum():
                outer_bar.setValue(outer_bar.value() - delta)
                event.accept()
                return
        super().wheelEvent(event)

    def _outer_scroll_area(self):
        parent = self.parent()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parent()
        return parent


# ===================== 纸张纹理 =====================
_PAPER_CACHE = {}


def paper_texture(base, seed=11, size=448):
    """Generate one subtle paper tile: base colour + mottles + fibers + speckles."""
    key = (base, seed, size)
    cached = _PAPER_CACHE.get(key)
    if cached is not None:
        return cached
    rnd = random.Random(seed)
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(QColor(base))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing, True)

    # soft cloudy mottles
    for _ in range(36):
        cx = rnd.uniform(0, size)
        cy = rnd.uniform(0, size)
        radius = rnd.uniform(size * 0.06, size * 0.22)
        grad = QRadialGradient(cx, cy, radius)
        if rnd.random() < 0.55:
            tone = QColor(120, 100, 72)
        else:
            tone = QColor(255, 252, 242)
        alpha = rnd.randint(3, 9)
        grad.setColorAt(0.0, QColor(tone.red(), tone.green(), tone.blue(), alpha))
        grad.setColorAt(1.0, QColor(tone.red(), tone.green(), tone.blue(), 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(grad)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

    # horizontal laid grain
    painter.setPen(QPen(QColor(120, 100, 72, 12), 0.5))
    for y in range(0, size, 4):
        painter.drawLine(QPointF(0, y), QPointF(size, y))
    painter.setPen(QPen(QColor(120, 100, 72, 20), 0.65))
    for y in range(0, size, 17):
        painter.drawLine(QPointF(0, y + 1), QPointF(size, y + 1))

    # short paper fibers
    for _ in range(3000):
        x = rnd.uniform(0, size)
        y = rnd.uniform(0, size)
        length = rnd.uniform(1.5, 10.0)
        if rnd.random() < 0.18:
            angle = rnd.uniform(-0.2, 0.2) + math.pi / 2
        else:
            angle = rnd.uniform(-0.2, 0.2)
        dx = math.cos(angle) * length
        dy = math.sin(angle) * length
        if rnd.random() < 0.55:
            color = QColor(120, 100, 72, rnd.randint(6, 22))
        else:
            color = QColor(255, 252, 244, rnd.randint(6, 20))
        painter.setPen(QPen(color, rnd.choice([0.5, 0.7, 1.0])))
        painter.drawLine(QPointF(x, y), QPointF(x + dx, y + dy))

    # a few long fibres
    for _ in range(70):
        x = rnd.uniform(0, size)
        y = rnd.uniform(0, size)
        length = rnd.uniform(14.0, 52.0)
        angle = rnd.uniform(-0.16, 0.16)
        dx = math.cos(angle) * length
        dy = math.sin(angle) * length
        if rnd.random() < 0.55:
            color = QColor(120, 100, 72, rnd.randint(4, 11))
        else:
            color = QColor(255, 252, 244, rnd.randint(4, 10))
        painter.setPen(QPen(color, 0.8))
        painter.drawLine(QPointF(x, y), QPointF(x + dx, y + dy))

    # tiny specks / foxing
    for _ in range(3200):
        x = rnd.uniform(0, size)
        y = rnd.uniform(0, size)
        color = QColor(120, 100, 72, rnd.randint(5, 18))
        painter.setPen(QPen(color, 0.7))
        painter.drawPoint(QPointF(x, y))

    painter.end()
    pixmap = QPixmap.fromImage(img)
    _PAPER_CACHE[key] = pixmap
    return pixmap


_PAGE_MOTTLE_CACHE = {}


def _page_mottle_pixmap(width, height, seed):
    key = (width, height, seed)
    cached = _PAGE_MOTTLE_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_PAGE_MOTTLE_CACHE) > 6:
        _PAGE_MOTTLE_CACHE.clear()
    rnd = random.Random(seed)
    img = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing, True)

    for _ in range(11):
        cx = rnd.uniform(0, width)
        cy = rnd.uniform(0, height)
        radius = rnd.uniform(width * 0.08, width * 0.30)
        grad = QRadialGradient(cx, cy, radius)
        if rnd.random() < 0.55:
            tone = QColor(120, 100, 72)
        else:
            tone = QColor(255, 252, 244)
        alpha = rnd.randint(3, 8)
        grad.setColorAt(0.0, QColor(tone.red(), tone.green(), tone.blue(), alpha))
        grad.setColorAt(1.0, QColor(tone.red(), tone.green(), tone.blue(), 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(grad)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

    for _ in range(90):
        x = rnd.uniform(0, width)
        y = rnd.uniform(0, height)
        length = rnd.uniform(6.0, 34.0)
        angle = rnd.uniform(-0.22, 0.22)
        dx = math.cos(angle) * length
        dy = math.sin(angle) * length
        if rnd.random() < 0.55:
            color = QColor(120, 100, 72, rnd.randint(4, 11))
        else:
            color = QColor(255, 252, 244, rnd.randint(4, 9))
        painter.setPen(QPen(color, 0.7))
        painter.drawLine(QPointF(x, y), QPointF(x + dx, y + dy))

    painter.end()
    pixmap = QPixmap.fromImage(img)
    _PAGE_MOTTLE_CACHE[key] = pixmap
    return pixmap


def paint_paper_mottle(painter, rect, seed=17):
    """Blit the cached large paper clouds/fibres instead of regenerating them."""
    width = int(rect.width())
    height = int(rect.height())
    if width <= 0 or height <= 0:
        return
    painter.drawPixmap(rect.topLeft(), _page_mottle_pixmap(width, height, seed))


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
        self._gray_pixmap = None
        self._color_pixmap = None
        self._bg_image_path = bg_image_path
        self._image_loaded = False
        self._hover = False
        self._panning = False
        self._exposure = CARD_REST_EXPOSURE
        self._bottom_color = QColor(120, 100, 80)
        self._glass_alpha = 0.46
        self._launch_state = "idle"
        self._launch_dots = 0
        self._launch_started_at = 0.0
        self._launch_timer = QTimer(self)
        self._launch_timer.setInterval(280)
        self._launch_timer.timeout.connect(self._tick_launch_dots)
        self._launch_reset_timer = QTimer(self)
        self._launch_reset_timer.setSingleShot(True)
        self._launch_reset_timer.timeout.connect(self._reset_launch)
        self._exposure_anim = QVariantAnimation(self)
        self._exposure_anim.setDuration(320)
        self._exposure_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._exposure_anim.valueChanged.connect(self._set_exposure)
        self.setContextMenuPolicy(Qt.NoContextMenu)
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
        inner_w = CARD_WIDTH - 2 * CARD_MOUNT
        inner_h = CARD_HEIGHT - 2 * CARD_MOUNT
        try:
            reader = QImageReader(image_path)
            reader.setAutoTransform(True)
            source = reader.size()
            if source.isValid() and source.width() > 0 and source.height() > 0:
                scale = max(inner_w / source.width(), inner_h / source.height())
                reader.setScaledSize(QSize(
                    max(1, int(source.width() * scale)),
                    max(1, int(source.height() * scale)),
                ))
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
            self._bottom_color, self._glass_alpha = self._derive_glass(img)
        except Exception as e:
            logging.warning(f"无法加载背景图片 {image_path}: {e}")

    @staticmethod
    def _relative_luminance(color):
        def channel(value):
            value = value / 255.0
            return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        return (0.2126 * channel(color.red())
                + 0.7152 * channel(color.green())
                + 0.0722 * channel(color.blue()))

    def _derive_glass(self, img):
        """按缩略图下半部的亮度，算一层刚好能透出颜色、又压得住文字的磨砂玻璃。"""
        w, h = img.width(), img.height()
        step = max(1, min(w, h) // 28)
        r = g = b = n = 0
        for y in range(int(h * 0.45), h, step):
            for x in range(0, w, step):
                color = img.pixelColor(x, y)
                r += color.red()
                g += color.green()
                b += color.blue()
                n += 1
        if n == 0:
            return QColor(120, 100, 80), 0.46

        bottom = QColor(r // n, g // n, b // n)
        target = 0.66
        glass_lum = self._relative_luminance(QColor(250, 246, 236))
        bottom_lum = self._relative_luminance(bottom)
        if bottom_lum >= target:
            alpha = 0.28
        else:
            alpha = (target - bottom_lum) / max(0.05, glass_lum - bottom_lum)
            alpha = max(0.28, min(0.64, alpha))
        return bottom, alpha

    def _set_exposure(self, value):
        self._exposure = max(0.0, min(1.0, float(value)))
        self.update()

    def _animate_exposure(self, target):
        self._exposure_anim.stop()
        if abs(self._exposure - target) < 0.01:
            self._set_exposure(target)
            return
        self._exposure_anim.setDuration(320 if target > self._exposure else 240)
        self._exposure_anim.setStartValue(self._exposure)
        self._exposure_anim.setEndValue(target)
        self._exposure_anim.start()

    def clear_hover(self):
        """拖动/滑动结束后清掉悬停态，避免卡片离开鼠标后还停在彩色曝光。"""
        if self._hover:
            self._hover = False
            self._apply_styles(hovered=False)
        if self._launch_state == "idle":
            self._animate_exposure(CARD_REST_EXPOSURE)

    def hideEvent(self, event):
        self._exposure_anim.stop()
        self._set_exposure(CARD_REST_EXPOSURE)
        if self._hover:
            self._hover = False
            self._apply_styles(hovered=False)
        super().hideEvent(event)

    def paintEvent(self, event):
        if not self._image_loaded:
            self._load_background()
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        inner_rect = self.rect().adjusted(
            CARD_MOUNT, CARD_MOUNT, -CARD_MOUNT, -CARD_MOUNT
        )
        gray = self._gray_pixmap
        color = self._color_pixmap
        if gray and not gray.isNull():
            clip = QPainterPath()
            clip.addRoundedRect(QRectF(inner_rect), 1.5, 1.5)
            painter.save()
            painter.setClipPath(clip)
            painter.drawPixmap(inner_rect, gray)
            if color and not color.isNull() and self._exposure > 0.0:
                # 光打上去，黑白底片逐渐曝光显色
                painter.setOpacity(self._exposure)
                painter.drawPixmap(inner_rect, color)
                painter.setOpacity(1.0)
            painter.restore()

            # 画面内缘的暗线；曝光时内侧再亮起一道受光边
            painter.setPen(QPen(QColor(62, 56, 44, 120), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(
                QRectF(inner_rect).adjusted(0.5, 0.5, -0.5, -0.5), 1.5, 1.5
            )
            if self._exposure > 0.01:
                painter.setPen(QPen(QColor(255, 247, 230, int(90 * self._exposure)), 1))
                painter.drawRoundedRect(
                    QRectF(inner_rect).adjusted(1.5, 1.5, -1.5, -1.5), 1.5, 1.5
                )

        painter.setPen(QPen(QColor(255, 248, 235, 34), 1))
        painter.drawLine(QPoint(CARD_MOUNT + 2, 2), QPoint(self.width() - CARD_MOUNT - 2, 2))
        painter.setPen(QPen(QColor(0, 0, 0, 45), 1))
        painter.drawLine(
            QPoint(CARD_MOUNT + 2, self.height() - 3),
            QPoint(self.width() - CARD_MOUNT - 2, self.height() - 3),
        )
        if self._exposure > 0.01:
            # 受光的一圈暖白边，随曝光渐显
            painter.setPen(QPen(QColor(255, 247, 230, int(175 * self._exposure)), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(
                QRectF(self.rect()).adjusted(1.2, 1.2, -1.2, -1.2),
                CARD_RADIUS, CARD_RADIUS,
            )
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
            background-color: {CARD_CAPTION};
            border: 1px solid {CARD_CAPTION_BORDER};
            border-radius: {CARD_CONTAINER_RADIUS}px;
        """)
        text_layout = QVBoxLayout(self.text_container)
        text_layout.setSpacing(CARD_TEXT_LAYOUT_SPACING)
        tm = CARD_TEXT_LAYOUT_MARGINS
        text_layout.setContentsMargins(tm[0], tm[1], tm[2], tm[3])

        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont(FAMILY_TITLE, CARD_TITLE_FONT_SIZE, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet(f"color: {CARD_INK}; background: transparent; border: none;")
        self.title_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self.title_rule = QFrame()
        self.title_rule.setFixedHeight(1)
        self.title_rule.setStyleSheet("background-color: rgba(38, 33, 26, 0.18); border: none;")

        self.desc_label = QLabel(self.description)
        self.desc_label.setFont(QFont(FAMILY_TITLE, CARD_DESC_FONT_SIZE))
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.desc_label.setStyleSheet(f"color: {CARD_MUTED}; background: transparent; border: none;")
        self.desc_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.title_rule)
        text_layout.addWidget(self.desc_label)
        layout.addWidget(self.text_container)
        layout.addStretch()

        if self.disabled:
            self.button = QPushButton("开发中")
            self.button.setEnabled(False)
            self.button.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba(236, 230, 218, 0.55);
                    color: #8a8275;
                    border: 1px solid rgba(255, 248, 235, 0.22);
                    padding: 4px 12px;
                    border-radius: {CARD_BUTTON_RADIUS}px;
                    font-size: 10px;
                }}
            """)
        else:
            self.button = QPushButton("启动")
            self.button.setCursor(QCursor(Qt.PointingHandCursor))
            self.button.clicked.connect(self._on_launch_clicked)

        self.button.setFixedSize(CARD_BUTTON_WIDTH, CARD_BUTTON_HEIGHT)
        layout.addWidget(self.button, 0, Qt.AlignCenter)
        self.setFixedSize(CARD_WIDTH, CARD_HEIGHT)
        self.setToolTip("左键拖动：浏览陈列柜  |  右键拖动：卡片换位")

    def setup_styles(self):
        # 卡片不额外创建阴影，避免滚动拖影和首屏开销。
        self.setAttribute(Qt.WA_StyledBackground, True)

    def _style_caption(self, hovered=False):
        if self.disabled:
            fill, border = "rgba(238, 232, 220, 0.60)", "rgba(38, 33, 26, 0.16)"
        elif hovered:
            fill, border = CARD_CAPTION_HOVER, CARD_CAPTION_BORDER_HOVER
        else:
            fill, border = CARD_CAPTION, CARD_CAPTION_BORDER
        self.text_container.setStyleSheet(f"""
            background-color: {fill};
            border: 1px solid {border};
            border-radius: {CARD_CONTAINER_RADIUS}px;
        """)

    @staticmethod
    def _css_color(color):
        return color.name() if isinstance(color, QColor) else color

    def _button_style(self, bg, fg, border, bold=False, hover_bg=None, pressed_bg=None):
        hover_bg = hover_bg if hover_bg is not None else bg
        pressed_bg = pressed_bg if pressed_bg is not None else bg
        weight = "font-weight: bold;" if bold else ""
        return f"""
            QPushButton {{
                background-color: {self._css_color(bg)};
                color: {fg};
                border: 1px solid {border};
                padding: 4px 12px;
                border-radius: {CARD_BUTTON_RADIUS}px;
                font-size: 10px;
                {weight}
            }}
            QPushButton:hover {{
                background-color: {self._css_color(hover_bg)};
                border-color: {border};
                color: {fg};
            }}
            QPushButton:pressed {{
                background-color: {self._css_color(pressed_bg)};
                border-color: {border};
                color: {fg};
            }}
        """

    def _glass_button_style(self, alpha, bold=False):
        alpha = max(0.24, min(0.66, float(alpha)))
        top = min(0.92, alpha + 0.12)
        bottom = min(0.88, alpha)
        # 直接指到按钮上时让它更透一点，底色更明显；按下只加亮边，不加白
        hover_top = max(0.20, alpha + 0.01)
        hover_bottom = max(0.18, alpha - 0.03)
        pressed_top = max(0.18, alpha - 0.02)
        pressed_bottom = max(0.16, alpha - 0.05)
        weight = "font-weight: bold;" if bold else ""

        def gradient(t, b):
            return (
                "qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                f"stop:0 rgba(255, 250, 240, {t:.3f}), "
                f"stop:1 rgba(250, 244, 232, {b:.3f}))"
            )

        return f"""
            QPushButton {{
                background: {gradient(top, bottom)};
                color: #261f17;
                border: 1px solid rgba(38, 33, 26, 0.42);
                padding: 4px 12px;
                border-radius: {CARD_BUTTON_RADIUS}px;
                font-size: 10px;
                {weight}
            }}
            QPushButton:hover {{
                background: {gradient(hover_top, hover_bottom)};
                border-color: rgba(38, 33, 26, 0.62);
                color: #261f17;
            }}
            QPushButton:pressed {{
                background: {gradient(pressed_top, pressed_bottom)};
                border-color: rgba(255, 248, 235, 0.92);
                color: #261f17;
            }}
        """

    def _apply_button_style(self, hovered):
        if self.disabled:
            return
        if self._launch_state != "idle":
            alpha = self._glass_alpha + (0.02 if self._launch_state == "started" else 0.0)
            self.button.setStyleSheet(self._glass_button_style(alpha, bold=True))
            return
        if hovered:
            self.button.setStyleSheet(self._glass_button_style(self._glass_alpha, bold=True))
        else:
            self.button.setStyleSheet(self._button_style(
                CARD_BUTTON_BG, CARD_BUTTON_INK, CARD_BUTTON_BORDER,
                hover_bg="#fbf6e9", pressed_bg="#ded5c4",
            ))

    def _apply_styles(self, hovered=False):
        bw = CARD_BORDER_WIDTH
        cr = CARD_RADIUS

        if self.disabled:
            self.setStyleSheet(f"""
                AppCard {{
                    background-color: #4a443a;
                    border: {bw}px solid rgba(255, 248, 235, 0.10);
                    border-radius: {cr}px;
                }}
            """)
            self._style_caption(False)
            return

        if hovered:
            self.setStyleSheet(f"""
                AppCard {{
                    background-color: {CARD_BG_HOVER};
                    border: {bw}px solid {CARD_FRAME_HOVER};
                    border-radius: {cr}px;
                }}
            """)
            self._style_caption(True)
        else:
            self.setStyleSheet(f"""
                AppCard {{
                    background-color: {CARD_BG};
                    border: {bw}px solid {CARD_FRAME};
                    border-radius: {cr}px;
                }}
            """)
            self._style_caption(False)
        self._apply_button_style(hovered)

    def _on_launch_clicked(self):
        if self.disabled or self._launch_state != "idle":
            return
        self._launch_state = "starting"
        self._launch_dots = 0
        self._launch_started_at = time.monotonic()
        self.button.setText("启动中")
        self._apply_button_style(self._hover)
        self._launch_timer.start()
        QTimer.singleShot(60, self._run_launch)

    def _run_launch(self):
        if self._launch_state != "starting":
            return
        result = None
        try:
            result = self.command()
        except Exception:
            logging.exception("启动应用失败")
            result = False
        elapsed_ms = (time.monotonic() - self._launch_started_at) * 1000.0
        delay = max(0, int(520 - elapsed_ms))
        QTimer.singleShot(delay, lambda: self._finish_launch(result))

    def _finish_launch(self, result):
        if self._launch_state != "starting":
            return
        self._launch_timer.stop()
        if result is False:
            self._launch_state = "failed"
            self.button.setText("启动失败")
        elif result == "already":
            self._launch_state = "already"
            self.button.setText("运行中")
        else:
            self._launch_state = "started"
            self.button.setText("已启动")
        self._launch_dots = 0
        self._apply_button_style(self._hover)
        self._launch_reset_timer.start(1900)

    def _reset_launch(self):
        if self._launch_state == "idle":
            return
        self._launch_timer.stop()
        self._launch_state = "idle"
        self._launch_dots = 0
        self.button.setText("启动")
        self._apply_button_style(self._hover)
        if not self._hover:
            self._animate_exposure(CARD_REST_EXPOSURE)

    def _tick_launch_dots(self):
        if self._launch_state != "starting":
            self._launch_timer.stop()
            return
        self._launch_dots = (self._launch_dots + 1) % 4
        self.button.setText("启动中" + "·" * self._launch_dots)

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
        # 左键：拖动整条陈列柜；右键：拿起卡片换位。
        if event.button() == Qt.LeftButton:
            self._drag_start = None
            self._panning = bool(
                self.category_widget is not None
                and self.category_widget.begin_pan(self, event.globalPosition().toPoint())
            )
            event.accept()
            return
        if event.button() == Qt.RightButton:
            self._drag_start = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and (event.buttons() & Qt.LeftButton):
            if self.category_widget is not None:
                self.category_widget.pan_to(event.globalPosition().toPoint())
            event.accept()
            return
        if self._drag_start is None or not (event.buttons() & Qt.RightButton):
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
        if event.button() == Qt.LeftButton and self._panning:
            self._panning = False
            if self.category_widget is not None:
                self.category_widget.end_pan()
            event.accept()
            return
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._drag_start = None
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        if self.category_widget is not None and getattr(self.category_widget, "_pan_active", False):
            super().enterEvent(event)
            return
        if not self.disabled:
            if not self._image_loaded:
                self._load_background()
            self._hover = True
            self.update()
            self._apply_styles(hovered=True)
            self._animate_exposure(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self.disabled:
            self._hover = False
            self._apply_styles(hovered=False)
            if self._launch_state == "idle":
                self._animate_exposure(CARD_REST_EXPOSURE)
        super().leaveEvent(event)


class CategoryWidget(QFrame):
    card_dropped = Signal(object, int)

    def __init__(self, category_name, parent=None):
        super().__init__(parent)
        self.category_name = category_name
        self.category_display = category_display(category_name)
        self.setAcceptDrops(True)
        self.setup_ui()
        self._apply_style(False)

    def setup_ui(self):
        self._drop_active = False
        self._nav_anim = None
        self._pan_active = False
        self._pan_source = None
        self._pan_last = None
        self._pan_override = False
        self.setFixedHeight(CABINET_HEIGHT)
        main_layout = QHBoxLayout(self)
        lm = CATEGORY_LAYOUT_MARGINS
        main_layout.setContentsMargins(lm[0], lm[1], lm[2], lm[3])
        main_layout.setSpacing(CATEGORY_LAYOUT_SPACING)

        self.scroll_area = HScrollArea()
        self.scroll_area.setWidgetResizable(True)
        # 陈列柜不显示系统滚动条：改用左右“换卷”按钮 + 边缘渐隐 + Shift 滚轮
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setFixedHeight(CARD_HEIGHT)
        # viewport 透明，陈列柜底纹由 CategoryWidget.paintEvent 统一绘制
        self.scroll_area.setStyleSheet(
            "QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }"
        )
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
            drop_target.setCursor(QCursor(Qt.OpenHandCursor))
        self.setCursor(QCursor(Qt.OpenHandCursor))
        main_layout.addWidget(self.scroll_area, 1)


        self._nav_left = self._make_nav_button("‹", -1, "查看前面的工具")
        self._nav_right = self._make_nav_button("›", 1, "查看后面的工具")
        bar = self.scroll_area.horizontalScrollBar()
        bar.valueChanged.connect(self._sync_navigation)
        bar.rangeChanged.connect(self._sync_navigation)
        QTimer.singleShot(0, self._sync_navigation)

    def _make_nav_button(self, text, direction, tip):
        button = QPushButton(text, self)
        button.setObjectName("cabinetNav")
        button.setFixedSize(CABINET_NAV_SIZE, CABINET_NAV_SIZE)
        button.setCursor(QCursor(Qt.PointingHandCursor))
        button.setFocusPolicy(Qt.NoFocus)
        button.setToolTip(tip)
        button.setStyleSheet(f"""
            QPushButton#cabinetNav {{
                background: rgba(248, 243, 233, 0.72);
                color: #4a4235;
                border: none;
                border-radius: 12px;
                font-family: "{FAMILY_TITLE}";
                font-size: 18px;
                font-weight: 400;
                padding-bottom: 4px;
            }}
            QPushButton#cabinetNav:hover {{
                background: rgba(252, 248, 240, 0.92);
                color: #363028;
            }}
            QPushButton#cabinetNav:pressed {{
                background: #e0d2bd;
                color: #363028;
            }}
        """)
        button.clicked.connect(lambda: self._scroll_strip(direction))
        button.hide()
        return button

    def _scroll_strip(self, direction):
        bar = self.scroll_area.horizontalScrollBar()
        if bar.maximum() <= bar.minimum():
            return
        page = max(CARD_WIDTH, self.scroll_area.viewport().width() - CARD_WIDTH)
        target = max(
            bar.minimum(),
            min(bar.maximum(), bar.value() + direction * page),
        )
        if self._nav_anim is None:
            self._nav_anim = QPropertyAnimation(bar, b"value", self)
            self._nav_anim.setDuration(190)
            self._nav_anim.setEasingCurve(QEasingCurve.OutCubic)
        else:
            self._nav_anim.stop()
        self._nav_anim.setStartValue(bar.value())
        self._nav_anim.setEndValue(target)
        self._nav_anim.start()

    def begin_pan(self, source, global_pos):
        """左键按住陈列柜开始平移；返回 False 表示这条陈列柜没有可滚动空间。"""
        if self._pan_active:
            self.end_pan()
        bar = self.scroll_area.horizontalScrollBar()
        if bar.maximum() <= bar.minimum():
            return False
        if self._nav_anim is not None:
            self._nav_anim.stop()
        self._pan_active = True
        self._pan_source = source
        self._pan_last = QPoint(global_pos)
        QApplication.setOverrideCursor(QCursor(Qt.ClosedHandCursor))
        self._pan_override = True
        return True

    def pan_to(self, global_pos):
        if not self._pan_active or self._pan_last is None:
            return
        dx = int(global_pos.x()) - self._pan_last.x()
        self._pan_last = QPoint(int(global_pos.x()), int(global_pos.y()))
        if dx:
            bar = self.scroll_area.horizontalScrollBar()
            bar.setValue(bar.value() - dx)

    def end_pan(self):
        self._pan_active = False
        self._pan_source = None
        self._pan_last = None
        if self._pan_override:
            QApplication.restoreOverrideCursor()
            self._pan_override = False
        for i in range(self.cards_layout.count()):
            card = self.cards_layout.itemAt(i).widget()
            if isinstance(card, AppCard):
                card.clear_hover()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.begin_pan(self, event.globalPosition().toPoint()):
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_active and (event.buttons() & Qt.LeftButton):
            self.pan_to(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._pan_active:
            self.end_pan()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def eventFilter(self, obj, event):
        # 画格之间的空隙、空陈列区域也要能左键拖动滑动。
        if obj in (self.scroll_area, self.scroll_area.viewport(), self.cards_container):
            etype = event.type()
            if etype == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                if self.begin_pan(obj, event.globalPosition().toPoint()):
                    event.accept()
                    return True
            elif etype == QEvent.MouseMove and self._pan_active and (event.buttons() & Qt.LeftButton):
                self.pan_to(event.globalPosition().toPoint())
                event.accept()
                return True
            elif etype == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self._pan_active:
                self.end_pan()
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def hideEvent(self, event):
        self.end_pan()
        super().hideEvent(event)

    def _sync_navigation(self, *args):
        if not hasattr(self, "_nav_left"):
            return
        bar = self.scroll_area.horizontalScrollBar()
        self._nav_left.setVisible(bar.value() > bar.minimum())
        self._nav_right.setVisible(bar.value() < bar.maximum())
        self._nav_left.raise_()
        self._nav_right.raise_()

    def _layout_navigation(self):
        if not hasattr(self, "_nav_left"):
            return
        y = CABINET_RAIL + (CARD_HEIGHT - CABINET_NAV_SIZE) // 2
        self._nav_left.move(CABINET_LEFT + 5, y)
        self._nav_right.move(max(CABINET_LEFT + 5, self.width() - CABINET_NAV_SIZE - 9), y)
        self._nav_left.raise_()
        self._nav_right.raise_()

    def _apply_style(self, active):
        self._drop_active = bool(active)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, CABINET_RADIUS, CABINET_RADIUS)

        # the sheet itself
        painter.fillPath(path, QBrush(paper_texture(PAPER_ROW, seed=21)))
        light = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        light.setColorAt(0.0, QColor(255, 252, 245, 30))
        light.setColorAt(0.35, QColor(255, 252, 245, 0))
        light.setColorAt(1.0, QColor(120, 100, 72, 12))
        painter.fillPath(path, light)

        painter.save()
        painter.setClipPath(path)
        rnd = random.Random((sum(ord(c) for c in self.category_name) + 19) & 0xffff)
        # a few longer fibers on top of the tiled texture
        for _ in range(36):
            x = rnd.uniform(4, max(5, self.width() - 4))
            y = rnd.uniform(4, max(5, self.height() - 4))
            length = rnd.uniform(2.0, 11.0)
            angle = rnd.uniform(-0.22, 0.22)
            dx = math.cos(angle) * length
            dy = math.sin(angle) * length
            if rnd.random() < 0.5:
                color = QColor(120, 100, 72, rnd.randint(5, 14))
            else:
                color = QColor(255, 252, 244, rnd.randint(5, 14))
            painter.setPen(QPen(color, rnd.choice([0.5, 0.7, 1.0])))
            painter.drawLine(QPointF(x, y), QPointF(x + dx, y + dy))
        # deeper horizontal grain, broken into irregular segments
        for y in range(3, int(self.height()) - 3, 4):
            alpha = 10 + (14 if (y // 4) % 7 == 0 else 0)
            painter.setPen(QPen(QColor(120, 100, 72, alpha), 0.6))
            x = 0.0
            width = float(self.width())
            while x < width:
                seg = rnd.uniform(22.0, 130.0)
                end = min(width, x + seg)
                painter.drawLine(
                    QPointF(x, y + rnd.uniform(-0.35, 0.35)),
                    QPointF(end, y + rnd.uniform(-0.35, 0.35)),
                )
                x = end + rnd.uniform(5.0, 26.0)
        painter.restore()

        # one plain aged-yellow label
        tab = QRectF(5, 4, CABINET_LEFT - 10, self.height() - 8)
        tab_path = QPainterPath()
        tab_path.addRoundedRect(tab, 3, 3)
        painter.fillPath(tab_path, QBrush(paper_texture(PAPER_TAB, seed=43)))
        # rough paper fibres on the right edge
        painter.setPen(QPen(QColor(120, 100, 72, 48), 0.8))
        for _ in range(12):
            fy = tab.top() + rnd.uniform(4, tab.height() - 4)
            fx = tab.right() - rnd.uniform(0, 2)
            painter.drawLine(QPointF(fx, fy), QPointF(fx + rnd.uniform(1, 3), fy + rnd.uniform(-1, 1)))
        shade = QLinearGradient(tab.right(), 0, tab.right() + 4, 0)
        shade.setColorAt(0.0, QColor(90, 76, 56, 26))
        shade.setColorAt(1.0, QColor(90, 76, 56, 0))
        painter.fillRect(QRectF(tab.right(), tab.top() + 2, 4, tab.height() - 4), shade)

        if self._drop_active:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(165, 82, 64, 160), 2))
            painter.drawRoundedRect(rect.adjusted(3, 3, -3, -3),
                                    CABINET_RADIUS, CABINET_RADIUS)

        self._paint_label(painter)

    def _paint_label(self, painter):
        tab = QRectF(5, 4, CABINET_LEFT - 10, self.height() - 8)
        count = self.card_count()

        cx = tab.center().x()
        stamp_cy = tab.top() + 15
        radius = 8
        stamp = QColor(PAPER_STAMP)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(stamp, 1.2))
        painter.drawEllipse(QPointF(cx, stamp_cy), radius, radius)
        stamp_font = QFont(FAMILY_TITLE, 7, QFont.Bold)
        painter.setFont(stamp_font)
        painter.setPen(stamp)
        painter.drawText(
            QRectF(cx - radius, stamp_cy - radius, radius * 2, radius * 2),
            Qt.AlignCenter,
            f"{count:02d}",
        )

        text = self.category_display
        chars = list(text)
        # Couplet layout: upright glyphs, one per line, at the previous
        # rotated-label font size (9 pt) rather than oversized display type.
        size = 9
        font = QFont(FAMILY_TITLE, size, QFont.Bold)
        metrics = QFontMetrics(font)
        line_h = metrics.height() + 1
        top = stamp_cy + radius + 7
        bottom = tab.bottom() - 5
        available = max(1, bottom - top)
        while size > 7 and line_h * len(chars) > available:
            size -= 1
            font = QFont(FAMILY_TITLE, size, QFont.Bold)
            metrics = QFontMetrics(font)
            line_h = metrics.height() + 1
        total_h = line_h * len(chars)
        start_y = top + max(0, (available - total_h) / 2.0)

        painter.setFont(font)
        painter.setPen(PAPER_INK)
        for index, char in enumerate(chars):
            painter.drawText(
                QRectF(tab.left(), start_y + index * line_h, tab.width(), line_h),
                Qt.AlignCenter,
                char,
            )

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
        return CABINET_HEIGHT

    def _update_height(self):
        self.scroll_area.setFixedHeight(CARD_HEIGHT)
        if self.height() != CABINET_HEIGHT:
            self.setFixedHeight(CABINET_HEIGHT)
        self._layout_navigation()
        self._sync_navigation()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_navigation()
        self._sync_navigation()

    def showEvent(self, event):
        super().showEvent(event)
        self._update_height()

    def add_card(self, card: AppCard):
        # 顶部对齐，避免 QHBoxLayout 把卡片在陈列柜行里垂直居中
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card, 0, Qt.AlignTop)
        card.category_widget = self
        QTimer.singleShot(0, self._update_height)


class PaperPage(QWidget):
    """Opaque paper page that scrolls with the cards; avoids transparent-scroll trails."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QBrush(paper_texture(PAGE_TOP, seed=7, size=512)))
        page = QLinearGradient(0, 0, 0, self.height())
        page.setColorAt(0.0, QColor(255, 252, 245, 26))
        page.setColorAt(0.45, QColor(255, 252, 245, 0))
        page.setColorAt(1.0, QColor(120, 100, 72, 12))
        painter.fillRect(self.rect(), page)
        paint_paper_mottle(painter, self.rect(), seed=31)


class LauncherBackdrop(QWidget):
    """整页是一张暖白纸：纸纹 + 很淡的顶光，让一张张纸片浮着。"""

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QBrush(paper_texture(PAGE_TOP, seed=7, size=512)))
        page = QLinearGradient(0, 0, 0, self.height())
        page.setColorAt(0.0, QColor(255, 252, 245, 26))
        page.setColorAt(0.45, QColor(255, 252, 245, 0))
        page.setColorAt(1.0, QColor(120, 100, 72, 12))
        painter.fillRect(self.rect(), page)

        glow = QRadialGradient(self.width() * 0.5, -self.height() * 0.18,
                               self.width() * 0.8)
        glow.setColorAt(0.0, QColor(255, 250, 240, 46))
        glow.setColorAt(0.55, QColor(255, 250, 240, 10))
        glow.setColorAt(1.0, QColor(255, 250, 240, 0))
        painter.fillRect(self.rect(), glow)
        paint_paper_mottle(painter, self.rect(), seed=17)


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

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.WindowDeactivate, QEvent.ApplicationDeactivate):
            for category in self.category_widgets:
                category.end_pan()
            return super().eventFilter(obj, event)
        # 卡片铺满整行，滚轮很容易落在卡片而不是页面滚动区上。
        # 统一在这里处理：竖滚 = 翻页，Shift/横滚 = 横向浏览这一排展柜。
        if event.type() == QEvent.Wheel and isinstance(obj, QWidget):
            if (obj.window() is self
                    and QApplication.activeModalWidget() is None
                    and not isinstance(obj, QScrollBar)):
                if self._scroll_from_wheel(obj, event):
                    return True
        return super().eventFilter(obj, event)

    def _scroll_from_wheel(self, obj, event):
        delta_y = event.pixelDelta().y() or event.angleDelta().y()
        delta_x = event.pixelDelta().x() or event.angleDelta().x()
        want_horizontal = bool(event.modifiers() & Qt.ShiftModifier) or (
            abs(delta_x) > abs(delta_y)
        )
        if want_horizontal:
            category = self._category_for_widget(obj)
            if category is not None:
                bar = category.scroll_area.horizontalScrollBar()
                delta = delta_y or delta_x
                if delta and bar.maximum() > bar.minimum():
                    bar.setValue(bar.value() - delta)
                    event.accept()
                    return True
        bar = self.outer_scroll.verticalScrollBar()
        if delta_y and bar.maximum() > bar.minimum():
            bar.setValue(bar.value() - delta_y)
            event.accept()
            return True
        return False

    def _category_for_widget(self, widget):
        parent = widget
        while parent is not None and not isinstance(parent, CategoryWidget):
            parent = parent.parent()
        return parent if isinstance(parent, CategoryWidget) else None

    def setup_ui(self):
        self.setWindowTitle(f"{APP_TITLE} {APP_VERSION}  ·  {INFO_DEFAULT}")
        icon = QIcon(APP_ICON_PATH)
        if not icon.isNull():
            self.setWindowIcon(icon)
        self.setMinimumSize(LAUNCHER_MIN_WIDTH, LAUNCHER_MIN_HEIGHT)
        self.resize(LAUNCHER_DEFAULT_WIDTH, LAUNCHER_DEFAULT_HEIGHT)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move((screen.width() - self.width()) // 2,
                  (screen.height() - self.height()) // 2)

        self.setStyleSheet(f"QMainWindow {{ background-color: {PAGE_BG}; }}")

        central = LauncherBackdrop()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(0)
        mm = LAUNCHER_MAIN_MARGINS
        main_layout.setContentsMargins(mm[0], mm[1], mm[2], mm[3])

        self.outer_scroll = QScrollArea()
        self.outer_scroll.setWidgetResizable(True)
        self.outer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.outer_scroll.setFrameShape(QFrame.NoFrame)
        self.outer_scroll.viewport().setAutoFillBackground(False)
        self.outer_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._style_outer_scrollbar()

        self.scroll_content = PaperPage()
        self.categories_layout = QVBoxLayout(self.scroll_content)
        self.categories_layout.setContentsMargins(0, 0, 0, 0)
        self.categories_layout.setSpacing(LAUNCHER_CATEGORY_SPACING)

        self.outer_scroll.setWidget(self.scroll_content)
        main_layout.addWidget(self.outer_scroll, 1)

        # status toast: overlay only, takes no layout space
        self.info_label = QLabel(INFO_DEFAULT, central)
        self.info_label.setFont(QFont(FAMILY_TITLE, 8))
        self.info_label.setStyleSheet(
            f"color: {PAGE_MUTED}; background: rgba(248, 242, 231, 0.95);"
            "border: 1px solid rgba(120, 100, 72, 0.22);"
            "border-radius: 3px; padding: 4px 10px;"
        )
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.info_label.hide()
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(self.info_label.hide)

    def _set_status(self, text):
        if not text or text == INFO_DEFAULT:
            self.info_label.setText(INFO_DEFAULT)
            self.info_label.hide()
            self._status_timer.stop()
            return
        self.info_label.setText(text)
        self.info_label.adjustSize()
        self._position_status()
        self.info_label.show()
        self.info_label.raise_()
        self._status_timer.start(4200)

    def _position_status(self):
        if not self.info_label.isVisible():
            return
        central = self.centralWidget()
        if central is None:
            return
        self.info_label.adjustSize()
        x = max(8, (central.width() - self.info_label.width()) // 2)
        y = max(8, central.height() - self.info_label.height() - 16)
        self.info_label.move(x, y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "info_label"):
            self._position_status()

    def _style_outer_scrollbar(self):
        bar = self.outer_scroll.verticalScrollBar()
        bar.setStyleSheet("""
            QScrollBar:vertical {
                background: transparent;
                width: 4px;
                margin: 0;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: rgba(160, 145, 120, 0.38);
                min-height: 32px;
                border-radius: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(140, 125, 100, 0.58);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
                width: 0;
                border: none;
                background: transparent;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)

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
                background-color: rgba(255, 248, 235, 20);
                border: 2px dashed rgba(255, 248, 235, 170);
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
        self._set_status(f"移动 {card.title}…")
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

    def _relocate_app_file(self, card, target, source=None):
        """跨分类拖卡片时，把对应的 .py 挪到目标分类目录并同步 apps_manifest.py。

        只有“真的换了分类”才搬文件：同一条陈列柜里重排（source 与 target 是同一
        个分类槽，例如 Noise/KAM 的 manifest 分类与所在目录本来就不一致）只改顺序。
        返回 False 表示这次拖动要放弃（文件动不了 / 正在运行 / 同名冲突）。
        """
        if not card.app_path or target is None:
            return True
        old_rel = card.app_path
        category = (target.category_name or "").strip()
        source_category = ""
        if source is not None:
            source_category = (getattr(source, "category_name", "") or "").strip()
        if source_category and source_category == category:
            return True
        if not category or category in (".", "..") or any(
                ch in category for ch in '\\/:*?"<>|'):
            self.show_error(f"分类名「{category}」不能用作文件夹名，文件保持原位。")
            return False

        name = os.path.basename(old_rel)
        new_rel = f"{category}/{name}"
        if new_rel == old_rel:
            return True
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
        card.button.clicked.connect(card._on_launch_clicked)
        return True

    def _finish_card_drag(self, target, index=None):
        if self._drag_card is None or target is None:
            self.cancel_card_drag()
            return
        card = self._drag_card
        if self._placeholder is not None and self._placeholder_cat is target:
            index = target.placeholder_card_index(self._placeholder)
        self._remove_placeholder()
        if not self._relocate_app_file(card, target, source=self._drag_source_cat):
            self.cancel_card_drag()
            return
        index = max(0, min(int(index or 0), target.card_count()))
        target.insert_card(card, index)
        card.show()
        self._save_layout()
        self._set_status(f"已移动 {card.title} 到「{target.category_name}」")
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
        if not self._relocate_app_file(card, target, source=source):
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
            return False

        app = self._manifest_for_path(module_path)
        missing = self._missing_dependencies(app) if app else []
        if missing:
            hints = "\n".join(f"{dep}: {INSTALL_HINTS.get(dep, 'pip install ' + dep)}" for dep in missing)
            QMessageBox.warning(
                self, "缺少依赖",
                f"{friendly_name} 需要以下依赖：\n{hints}\n\n安装后再启动。"
            )
            return False

        for entry in self.active_processes:
            if entry["path"] == module_path and entry["process"].poll() is None:
                QMessageBox.information(self, "已在运行", f"{friendly_name} 已经在运行。")
                return "already"

        runner = os.path.join(PROJECT_DIR, "app_runner.py")
        if not os.path.exists(runner):
            self.show_error(f"找不到启动器入口:\n{runner}")
            return False

        # 子进程输出写入 runtime/logs：比管道更稳（打包后无控制台时也能查错）。
        # 追加写 + 单份轮转，崩溃后还能看到上一次启动的记录。
        log_dir = os.path.join(RUNTIME_DIR, "logs")
        base = os.path.splitext(os.path.basename(module_path))[0]
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in base)
        out_log = os.path.join(log_dir, f"{safe}.out.log")
        error_log = os.path.join(log_dir, f"{safe}.err.log")
        ready_path = os.path.join(READY_DIR, f"{safe}.ready")
        handles = []
        try:
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(READY_DIR, exist_ok=True)
            self._remove_ready(ready_path)
            out_handle = self._open_log(out_log)
            handles.append(out_handle)
            err_handle = self._open_log(error_log)
            handles.append(err_handle)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            for handle in (out_handle, err_handle):
                handle.write(f"\n===== {stamp} launch {module_path} =====\n")
        except OSError:
            for handle in handles:
                handle.close()
            handles = []
            out_handle = err_handle = subprocess.DEVNULL
            out_log = error_log = None

        creationflags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        env = dict(os.environ)
        env[READY_ENV] = ready_path
        env["PYSIDE_TOOLS_SPLASH"] = "1"
        try:
            if getattr(sys, "frozen", False):
                child_cmd = [sys.executable, "--child", module_path]
            else:
                child_cmd = [sys.executable, runner, module_path]
            proc = subprocess.Popen(
                child_cmd,
                cwd=PROJECT_DIR,
                stdout=out_handle,
                stderr=err_handle,
                env=env,
                creationflags=creationflags,
            )
        except OSError as exc:
            self.show_error(f"无法启动 {friendly_name}:\n{exc}")
            self._remove_ready(ready_path)
            return False
        finally:
            for handle in handles:
                handle.close()

        self.active_processes.append({
            "process": proc,
            "path": module_path,
            "friendly": friendly_name,
            "out_log": out_log,
            "error_log": error_log,
            "ready": ready_path,
            "ready_seen": False,
            "started_at": time.monotonic(),
            "slow_warned": False,
            "timeout_warned": False,
        })
        self._set_status(f"正在启动 {friendly_name}…（首次编译内核可能要 10 秒以上）")
        return True

    def _poll_processes(self):
        changed = False
        now = time.monotonic()
        for entry in list(self.active_processes):
            code = entry["process"].poll()
            if code is None:
                if entry.get("ready_seen"):
                    continue
                elapsed = now - entry["started_at"]
                if self._ready_matches(entry):
                    entry["ready_seen"] = True
                    self._remove_ready(entry.get("ready"))
                    if not self._closing:
                        ready = sum(
                            1 for item in self.active_processes if item.get("ready_seen")
                        )
                        self._set_status(
                            f"{entry['friendly']} 已就绪 | 已启动 {ready} 个工具"
                        )
                elif elapsed > READY_TIMEOUT_SECONDS and not entry.get("timeout_warned"):
                    entry["timeout_warned"] = True
                    if not self._closing:
                        self._set_status(
                            f"{entry['friendly']} 仍未就绪，可能卡在编译；详见 runtime/logs"
                        )
                elif elapsed > READY_SLOW_SECONDS and not entry.get("slow_warned"):
                    entry["slow_warned"] = True
                    if not self._closing:
                        self._set_status(f"{entry['friendly']} 正在编译内核…")
                continue
            self.active_processes.remove(entry)
            self._remove_ready(entry.get("ready"))
            changed = True
            if self._closing:
                continue
            if code != 0:
                self._show_app_crash(entry, code)
        if changed:
            self._restore_info()

    # ---------- 子进程日志与就绪握手 ----------
    def _open_log(self, path):
        """Append to a log file, keeping one rotated copy when it gets large."""
        try:
            if os.path.exists(path) and os.path.getsize(path) > LOG_ROTATE_BYTES:
                os.replace(path, path + ".prev")
        except OSError:
            pass
        return open(path, "a", encoding="utf-8", errors="replace")

    def _remove_ready(self, path):
        if not path:
            return
        try:
            os.remove(path)
        except OSError:
            pass

    def _ready_matches(self, entry):
        """True when the child wrote its pid to the marker file (no marker: legacy app)."""
        path = entry.get("ready")
        if not path:
            return True
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read().strip() == str(entry["process"].pid)
        except OSError:
            return False

    def _restore_info(self):
        if self._closing:
            return
        total = len(self.active_processes)
        ready = sum(1 for entry in self.active_processes if entry.get("ready_seen"))
        starting = total - ready
        if starting:
            self._set_status(f"正在启动 {starting} 个工具 | 已就绪 {ready} 个")
        elif ready:
            self._set_status(f"已启动 {ready} 个工具 | 关闭启动器会一并关闭它们")
        else:
            self._set_status(INFO_DEFAULT)

    def _show_app_crash(self, entry, code):
        log_path = entry.get("error_log")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("错误")
        box.setText(f"{entry['friendly']} 异常退出（代码 {code}）。")
        open_button = None
        if log_path:
            box.setInformativeText(f"错误日志：{log_path}")
            open_button = box.addButton("打开错误日志", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if open_button is not None and box.clickedButton() is open_button:
            self._open_log_file(log_path)

    def _open_log_file(self, path):
        if not path or not os.path.exists(path):
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def show_error(self, message):
        QMessageBox.critical(self, "错误", message)

    def closeEvent(self, event):
        self._closing = True
        self._poll_timer.stop()
        for category in self.category_widgets:
            category.end_pan()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        entries = list(self.active_processes)
        for entry in entries:
            self._remove_ready(entry.get("ready"))
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
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "PySideDrawingTools.Launcher.1"
            )
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app_icon = QIcon(APP_ICON_PATH)
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    app.setFont(QFont(FAMILY_DEFAULT, 9))
    launcher = ApplicationLauncher()
    launcher.show()
    sys.exit(app.exec())