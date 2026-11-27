import os
import sys
import time
import shutil
import subprocess
import importlib.util
import logging
import json
import math

from PySide6.QtCore import (
    Qt, QTimer, Signal, QMimeData, QPoint, QEvent, QVariantAnimation,
    QRectF, QPropertyAnimation, QEasingCurve,
)
from PySide6.QtGui import (
    QFont, QFontMetrics, QCursor, QPainter, QPixmap, QImage, QImageReader, QDrag,
    QColor, QPainterPath, QLinearGradient, QRadialGradient, QPen, QBrush,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QScrollBar, QFrame, QMessageBox,
    QGraphicsDropShadowEffect,
)

from apps_manifest import APPS, CATEGORY_ORDER, INSTALL_HINTS, category_note
from manifest_edit import rewrite_app_entry

# ===================== 全局常量 =====================
CARD_WIDTH = 150
CARD_HEIGHT = 150
CARD_BUTTON_WIDTH = 100
CARD_BUTTON_HEIGHT = 26

CARD_TITLE_FONT_SIZE = 11
CARD_DESC_FONT_SIZE = 8

CARD_RADIUS = 3
CARD_CONTAINER_RADIUS = 2
CARD_BUTTON_RADIUS = 2

CARD_LAYOUT_SPACING = 6
CARD_LAYOUT_MARGINS = (7, 7, 7, 7)
CARD_TEXT_LAYOUT_SPACING = 2
CARD_TEXT_LAYOUT_MARGINS = (7, 5, 7, 5)

CARD_BORDER_WIDTH = 1
CARD_MOUNT = 4                      # 卡片周围的灰色片格宽度

# ---- 胶卷（film strip）外观 ----
# 每个分类是一整条横向胶片：上下齿孔带 + 左侧竖排分类字 + 中间的画格（卡片）。
FILM_BAND = 13                      # 上下齿孔带高度
FILM_LEADER = 40                    # 左侧竖排分类名所占宽度（原来的 95px 灰色分类栏）
FILM_RADIUS = 7                     # 胶片圆角
FILM_STRIP_HEIGHT = CARD_HEIGHT + 2 * FILM_BAND   # 176
FILM_HOLE_WIDTH = 9
FILM_HOLE_HEIGHT = 7
FILM_HOLE_GAP = 12
FILM_HOLE_START = 34                # 齿孔从分类标签右侧开始
FILM_HOLE_END_PAD = 12
FILM_CARD_GAP = 12
FILM_CARD_MARGINS = (6, 0, 6, 0)
FILM_LABEL_SIZE = 10
FILM_COUNT_SIZE = 7
FILM_NAV_SIZE = 28
FILM_EDGE_WIDTH = 38

# 胶片配色：偏暖的深褐灰，接近真实胶片的片基与齿孔
FILM_TOP = QColor("#3d362e")
FILM_MID = QColor("#2b2620")
FILM_BOTTOM = QColor("#1d1a16")
FILM_EDGE = QColor(14, 11, 8, 170)
FILM_SHEEN = QColor(255, 246, 228, 22)
FILM_HOLE = QColor(233, 226, 212, 158)
FILM_HOLE_SHADE = QColor(0, 0, 0, 85)
FILM_LABEL = QColor("#efe8d9")
FILM_LIGHT = QColor("#fff7e8")
FILM_GATE_TOP = QColor("#4c453b")
FILM_GATE_BOTTOM = QColor("#3a342c")

# 页面配色：暖石色纸面，顶部有一点漫射光
PAGE_BG = "#e6e1d6"
PAGE_TOP = "#ece7dc"
PAGE_BOTTOM = "#ddd7ca"
PAGE_MUTED = "#7f786b"

# 卡片：暖灰片格 + 象牙纸标签
CARD_BG = "#5b5448"
CARD_BG_HOVER = "#6b6355"
CARD_FRAME = "rgba(255, 248, 235, 0.20)"
CARD_FRAME_HOVER = "rgba(201, 128, 91, 0.92)"
CARD_INK = "#26211a"
CARD_MUTED = "#60584b"
CARD_CAPTION = "rgba(247, 241, 229, 0.93)"
CARD_CAPTION_HOVER = "rgba(252, 246, 234, 0.98)"
CARD_CAPTION_BORDER = "rgba(38, 33, 26, 0.24)"
CARD_CAPTION_BORDER_HOVER = "rgba(180, 105, 70, 0.58)"
CARD_IDLE_EXPOSURE = 0.18
CARD_BUTTON_BG = "rgba(240, 234, 221, 0.94)"
CARD_BUTTON_INK = "#26211a"
CARD_BUTTON_BORDER = "rgba(38, 33, 26, 0.34)"

# 兼容旧名（分类面板现在固定为胶片条高度）
CATEGORY_HEIGHT = FILM_STRIP_HEIGHT
CATEGORY_LABEL_WIDTH = FILM_LEADER
CATEGORY_LAYOUT_MARGINS = (FILM_LEADER, FILM_BAND, 12, FILM_BAND)
CATEGORY_LAYOUT_SPACING = 0
CATEGORY_CARDS_SPACING = FILM_CARD_GAP
CATEGORY_CARDS_MARGINS = FILM_CARD_MARGINS
CATEGORY_TITLE_FONT_SIZE = FILM_LABEL_SIZE
CATEGORY_RADIUS = FILM_RADIUS

LAUNCHER_MIN_WIDTH = 640
LAUNCHER_MIN_HEIGHT = 480
LAUNCHER_DEFAULT_WIDTH = 1200
LAUNCHER_DEFAULT_HEIGHT = 800
LAUNCHER_MAIN_MARGINS = (24, 20, 24, 4)
LAUNCHER_CATEGORY_SPACING = 18

INFO_FONT_SIZE = 9
INFO_DEFAULT = "左键拖动滑动胶片  ·  右键拖动卡片换位  ·  绘图工具集 v1.1"

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
        self._exposure = 0.0
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
            self._animate_exposure(0.0)

    def hideEvent(self, event):
        self._exposure_anim.stop()
        self._set_exposure(0.0)
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
            if color and not color.isNull():
                # 静止时保留一层很轻的色偏；悬停时像底片曝光一样逐渐显色
                exposure = (
                    CARD_IDLE_EXPOSURE
                    + (1.0 - CARD_IDLE_EXPOSURE) * self._exposure
                )
                painter.setOpacity(exposure)
                painter.drawPixmap(inner_rect, color)
                painter.setOpacity(1.0)
                if self._exposure > 0.01 and self._exposure < 0.999:
                    flash = int(math.sin(math.pi * self._exposure) * 22)
                    if flash > 0:
                        painter.fillRect(inner_rect, QColor(255, 248, 235, flash))
            painter.restore()

            # 画面内缘的暗线；曝光时内侧再亮起一道受光边
            painter.setPen(QPen(QColor(0, 0, 0, 95), 1))
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
        self.setToolTip("左键拖动：滑动胶片  |  右键拖动：卡片换位")

    def setup_styles(self):
        # 每张卡片一个 QGraphicsDropShadowEffect 会显著拖慢首屏；
        # 边框已经足够区分卡片，阴影改为不创建。
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
            self.title_rule.setStyleSheet(
                "background-color: rgba(180, 105, 70, 0.62); border: none;"
            )
        else:
            self.setStyleSheet(f"""
                AppCard {{
                    background-color: {CARD_BG};
                    border: {bw}px solid {CARD_FRAME};
                    border-radius: {cr}px;
                }}
            """)
            self._style_caption(False)
            self.title_rule.setStyleSheet(
                "background-color: rgba(38, 33, 26, 0.18); border: none;"
            )
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
            self._animate_exposure(0.0)

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
        # 左键：拖动整条胶片；右键：拿起卡片换位。
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
                self._animate_exposure(0.0)
        super().leaveEvent(event)


class FilmEdge(QWidget):
    """画格被裁切的一侧渐隐为胶片底色，让横向滚动看起来像胶片延伸到画外。"""

    def __init__(self, side, parent=None):
        super().__init__(parent)
        self.side = side
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), 0)
        dark = QColor(74, 69, 60, 235)
        clear = QColor(74, 69, 60, 0)
        if self.side == "right":
            gradient.setColorAt(0.0, clear)
            gradient.setColorAt(1.0, dark)
        else:
            gradient.setColorAt(0.0, dark)
            gradient.setColorAt(1.0, clear)
        painter.fillRect(self.rect(), gradient)


# ===================== 分类槽 =====================
class CategoryWidget(QFrame):
    card_dropped = Signal(object, int)

    def __init__(self, category_name, parent=None):
        super().__init__(parent)
        self.category_name = category_name
        self.category_marker = "实验" if category_note(category_name) else ""
        if self.category_marker:
            # 实验的说明放在整条胶卷的 tooltip 上，竖排标签保持短
            self.setToolTip(f"实验：{category_note(category_name)}")
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
        self.setFixedHeight(FILM_STRIP_HEIGHT)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(32, 26, 20, 42))
        self.setGraphicsEffect(shadow)

        main_layout = QHBoxLayout(self)
        lm = CATEGORY_LAYOUT_MARGINS
        main_layout.setContentsMargins(lm[0], lm[1], lm[2], lm[3])
        main_layout.setSpacing(CATEGORY_LAYOUT_SPACING)

        self.scroll_area = HScrollArea()
        self.scroll_area.setWidgetResizable(True)
        # 胶卷不显示系统滚动条：改用左右“换卷”按钮 + 边缘渐隐 + Shift 滚轮
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setFixedHeight(CARD_HEIGHT)
        # viewport 透明，胶片底纹由 CategoryWidget.paintEvent 统一绘制
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

        self._edge_left = FilmEdge("left", self)
        self._edge_right = FilmEdge("right", self)
        self._edge_left.hide()
        self._edge_right.hide()

        self._nav_left = self._make_nav_button("‹", -1, "查看前面的工具")
        self._nav_right = self._make_nav_button("›", 1, "查看后面的工具")
        bar = self.scroll_area.horizontalScrollBar()
        bar.valueChanged.connect(self._sync_navigation)
        bar.rangeChanged.connect(self._sync_navigation)
        QTimer.singleShot(0, self._sync_navigation)

    def _make_nav_button(self, text, direction, tip):
        button = QPushButton(text, self)
        button.setObjectName("filmNav")
        button.setFixedSize(FILM_NAV_SIZE, FILM_NAV_SIZE)
        button.setCursor(QCursor(Qt.PointingHandCursor))
        button.setFocusPolicy(Qt.NoFocus)
        button.setToolTip(tip)
        button.setStyleSheet(f"""
            QPushButton#filmNav {{
                background: transparent;
                color: rgba(255, 248, 235, 0.80);
                border: none;
                font-family: "{FAMILY_TITLE}";
                font-size: 24px;
                font-weight: 400;
                padding-bottom: 6px;
            }}
            QPushButton#filmNav:hover {{
                color: {FILM_LIGHT.name()};
            }}
            QPushButton#filmNav:pressed {{
                color: #c78f43;
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
        """左键按住胶片开始平移；返回 False 表示这条胶片没有可滚动空间。"""
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
        # 画格之间的空隙、空胶片区域也要能左键拖动滑动。
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
        has_overflow = bar.maximum() > bar.minimum()
        self._edge_left.setVisible(has_overflow and bar.value() > bar.minimum())
        self._edge_right.setVisible(has_overflow and bar.value() < bar.maximum())
        self._nav_left.setVisible(bar.value() > bar.minimum())
        self._nav_right.setVisible(bar.value() < bar.maximum())
        self._edge_left.raise_()
        self._edge_right.raise_()
        self._nav_left.raise_()
        self._nav_right.raise_()

    def _layout_navigation(self):
        if not hasattr(self, "_nav_left"):
            return
        y = FILM_BAND + (CARD_HEIGHT - FILM_NAV_SIZE) // 2
        self._nav_left.move(FILM_LEADER + 5, y)
        self._nav_right.move(max(FILM_LEADER + 5, self.width() - FILM_NAV_SIZE - 10), y)
        self._edge_left.setGeometry(FILM_LEADER, FILM_BAND, FILM_EDGE_WIDTH, CARD_HEIGHT)
        self._edge_right.setGeometry(
            self.width() - FILM_HOLE_END_PAD - FILM_EDGE_WIDTH,
            FILM_BAND, FILM_EDGE_WIDTH, CARD_HEIGHT,
        )
        self._edge_left.raise_()
        self._edge_right.raise_()
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
        path.addRoundedRect(rect, FILM_RADIUS, FILM_RADIUS)

        gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0.0, FILM_TOP)
        gradient.setColorAt(0.52, FILM_MID)
        gradient.setColorAt(1.0, FILM_BOTTOM)
        painter.fillPath(path, QBrush(gradient))

        painter.save()
        painter.setClipPath(path)

        # 顶部有一束柔和的光，胶片表面再斜着扫过一道高光
        top_light = QLinearGradient(0, 0, 0, FILM_BAND + 30)
        top_light.setColorAt(0.0, QColor(255, 248, 235, 34))
        top_light.setColorAt(1.0, QColor(255, 248, 235, 0))
        painter.fillRect(QRectF(0, 0, self.width(), FILM_BAND + 30), top_light)

        sheen = QLinearGradient(0, 0, self.width(), self.height())
        sheen.setColorAt(0.00, QColor(255, 248, 235, 0))
        sheen.setColorAt(0.38, QColor(255, 248, 235, 0))
        sheen.setColorAt(0.50, QColor(255, 248, 235, 14))
        sheen.setColorAt(0.62, QColor(255, 248, 235, 0))
        sheen.setColorAt(1.00, QColor(255, 248, 235, 0))
        painter.fillRect(self.rect(), sheen)

        # 画格后面的灰色片格底，卡片之间的空隙也因此是灰的
        gate = QRectF(
            FILM_LEADER,
            FILM_BAND,
            self.width() - FILM_LEADER - FILM_HOLE_END_PAD,
            CARD_HEIGHT,
        )
        gate_gradient = QLinearGradient(gate.topLeft(), gate.bottomLeft())
        gate_gradient.setColorAt(0.0, FILM_GATE_TOP)
        gate_gradient.setColorAt(1.0, FILM_GATE_BOTTOM)
        painter.fillRect(gate, gate_gradient)

        # 灰格上同样有一道斜向反光，让空隙不是死板的灰色
        gate_sheen = QLinearGradient(gate.topLeft(), gate.bottomRight())
        gate_sheen.setColorAt(0.00, QColor(255, 248, 235, 0))
        gate_sheen.setColorAt(0.34, QColor(255, 248, 235, 0))
        gate_sheen.setColorAt(0.47, QColor(255, 248, 235, 44))
        gate_sheen.setColorAt(0.60, QColor(255, 248, 235, 0))
        gate_sheen.setColorAt(1.00, QColor(255, 248, 235, 0))
        painter.fillRect(gate, gate_sheen)

        painter.setPen(QPen(QColor(255, 248, 235, 26), 1))
        painter.drawLine(gate.topLeft(), gate.topRight())
        painter.setPen(QPen(QColor(0, 0, 0, 40), 1))
        painter.drawLine(gate.bottomLeft(), gate.bottomRight())

        self._paint_perforations(painter)
        self._paint_empty_frames(painter)
        painter.restore()

        # 顶部一道细高光，像光落在胶片边缘
        painter.setPen(QPen(QColor(255, 248, 235, 52), 1))
        painter.drawLine(
            QPoint(FILM_RADIUS + 6, 1),
            QPoint(self.width() - FILM_RADIUS - 6, 1),
        )
        # 分类标签与画格之间的分隔线
        painter.setPen(QPen(QColor(255, 248, 235, 18), 1))
        painter.drawLine(
            QPoint(FILM_LEADER - 1, FILM_BAND + 8),
            QPoint(FILM_LEADER - 1, self.height() - FILM_BAND - 8),
        )

        self._paint_label(painter)

        if self._drop_active:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 248, 235, 60), 4))
            painter.drawRoundedRect(rect.adjusted(2.5, 2.5, -2.5, -2.5),
                                    FILM_RADIUS, FILM_RADIUS)
            painter.setPen(QPen(FILM_LIGHT, 1.4))
            painter.drawRoundedRect(rect.adjusted(1.5, 1.5, -1.5, -1.5),
                                    FILM_RADIUS - 1, FILM_RADIUS - 1)
        else:
            painter.setPen(QPen(FILM_EDGE, 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect, FILM_RADIUS, FILM_RADIUS)

    def _paint_perforations(self, painter):
        painter.setPen(Qt.NoPen)
        offset = (FILM_BAND - FILM_HOLE_HEIGHT) / 2.0
        row_ys = (
            offset,
            self.height() - FILM_BAND + offset,
        )
        x = FILM_HOLE_START
        limit = self.width() - FILM_HOLE_END_PAD - FILM_HOLE_WIDTH
        while x <= limit:
            for y in row_ys:
                hole = QRectF(x, y, FILM_HOLE_WIDTH, FILM_HOLE_HEIGHT)
                painter.setBrush(FILM_HOLE)
                painter.drawRoundedRect(hole, 1.6, 1.6)
                painter.setPen(QPen(FILM_HOLE_SHADE, 1))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(hole.adjusted(0.5, 0.5, -0.5, -0.5), 1.4, 1.4)
                painter.setPen(Qt.NoPen)
            x += FILM_HOLE_WIDTH + FILM_HOLE_GAP

    def _paint_empty_frames(self, painter):
        """内容不满一行时，补几格极淡的空画格，让胶片自然延伸到末端。"""
        count = self.card_count()
        x = FILM_LEADER + FILM_CARD_MARGINS[0] + count * (CARD_WIDTH + FILM_CARD_GAP)
        limit = self.width() - FILM_HOLE_END_PAD
        if x + CARD_WIDTH > limit:
            return
        painter.setPen(QPen(QColor(255, 248, 235, 22), 1))
        painter.setBrush(QColor(255, 248, 235, 5))
        while x + CARD_WIDTH <= limit:
            painter.drawRoundedRect(
                QRectF(x, FILM_BAND, CARD_WIDTH, CARD_HEIGHT),
                CARD_RADIUS, CARD_RADIUS,
            )
            x += CARD_WIDTH + FILM_CARD_GAP

    def _paint_label(self, painter):
        count = self.card_count()
        count_font = QFont(FAMILY_TITLE, FILM_COUNT_SIZE, QFont.Bold)
        count_font.setLetterSpacing(QFont.AbsoluteSpacing, 0.8)
        painter.setFont(count_font)
        count_color = QColor(FILM_LIGHT)
        count_color.setAlpha(175)
        painter.setPen(count_color)
        painter.drawText(
            QRectF(9, FILM_BAND + 2, FILM_LEADER - 14, 13),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"{count:02d}",
        )
        underline = QColor(FILM_LIGHT)
        underline.setAlpha(80)
        painter.setPen(QPen(underline, 1))
        painter.drawLine(QPoint(10, FILM_BAND + 17), QPoint(24, FILM_BAND + 17))

        text = self.category_name.upper() if self.category_name.isascii() else self.category_name
        if self.category_marker:
            text = f"{text} · {self.category_marker}"
        size = FILM_LABEL_SIZE
        available = self.height() - 2 * FILM_BAND - 30
        font = QFont(FAMILY_TITLE, size, QFont.Bold)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1.0)
        metrics = QFontMetrics(font)
        while size > 7 and metrics.horizontalAdvance(text) > available:
            size -= 1
            font.setPointSize(size)
            metrics = QFontMetrics(font)
        elided = metrics.elidedText(text, Qt.ElideRight, available)

        painter.save()
        painter.translate(FILM_LEADER / 2.0, self.height() / 2.0 + 4)
        painter.rotate(-90)
        painter.setFont(font)
        painter.setPen(FILM_LABEL)
        painter.drawText(
            QRectF(-available / 2.0, -metrics.height() / 2.0, available, metrics.height()),
            Qt.AlignCenter,
            elided,
        )
        painter.restore()

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
        return FILM_STRIP_HEIGHT

    def _update_height(self):
        self.scroll_area.setFixedHeight(CARD_HEIGHT)
        if self.height() != FILM_STRIP_HEIGHT:
            self.setFixedHeight(FILM_STRIP_HEIGHT)
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
        # 顶部对齐，避免 QHBoxLayout 把卡片在胶片行里垂直居中
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card, 0, Qt.AlignTop)
        card.category_widget = self
        QTimer.singleShot(0, self._update_height)


class LauncherBackdrop(QWidget):
    """浅石灰纸面 + 顶部一层很淡的漫射光，让胶片浮在光里。"""

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        page = QLinearGradient(0, 0, 0, self.height())
        page.setColorAt(0.0, QColor(PAGE_TOP))
        page.setColorAt(1.0, QColor(PAGE_BOTTOM))
        painter.fillRect(self.rect(), page)

        glow = QRadialGradient(self.width() * 0.5, -self.height() * 0.18,
                               self.width() * 0.8)
        glow.setColorAt(0.0, QColor(255, 248, 235, 58))
        glow.setColorAt(0.55, QColor(255, 248, 235, 14))
        glow.setColorAt(1.0, QColor(255, 248, 235, 0))
        painter.fillRect(self.rect(), glow)


# ===================== 主窗口 =====================
