# -*- coding: utf-8 -*-
"""Structural checks for the project UI standard."""
from pathlib import Path

from PySide6.QtGui import QImage

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "setFixedWidth(330)",
    "setFixedSize(600, 600)",
    "setMinimumSize(960, 640)",
    "resize(960, 640)",
    "background-color: #ffffff",
    "QGroupBox",
    "LabeledSlider",
    "common.lab_window",
    "run_app(MainWindow)",
)

FORBIDDEN = (
    "title = QLabel",
    "hint = QLabel",
    "▶",
    "⏸",
    "setWindowIcon(",
    "setGeometry(120, 100, 1180, 780)",
)

TWO_VIEW_MARKER = 'UI_LAYOUT = "two_view"'
TWO_VIEW_REQUIRED = (
    "setMinimumSize(",
    "resize(",
    "background-color: #ffffff",
    "QGroupBox",
    "LabeledSlider",
    "common.lab_window",
    "TwoViewCanvas(",
    "drawRect(",
    "SmoothPixmapTransform, False",
    "run_app(MainWindow)",
)
TWO_VIEW_FORBIDDEN = (
    "setFixedWidth(330)",
    "setFixedSize(600, 600)",
    "setMinimumSize(960, 640)",
    "resize(960, 640)",
    "title = QLabel",
    "hint = QLabel",
    "▶",
    "⏸",
    "setWindowIcon(",
    "setGeometry(120, 100, 1180, 780)",
)


def check_app_ui(app_path):
    """Return UI standard violations.

    Two-view apps opt in with the marker UI_LAYOUT = "two_view": controls on
    top, a wide two-view canvas below, and no 330 left column / 600 square.
    """
    path = Path(app_path)
    if not path.is_absolute():
        path = ROOT / path
    issues = []
    if not path.exists():
        return [f"应用文件不存在: {path}"]
    source = path.read_text(encoding="utf-8")
    two_view = TWO_VIEW_MARKER in source
    required = TWO_VIEW_REQUIRED if two_view else REQUIRED
    forbidden = TWO_VIEW_FORBIDDEN if two_view else FORBIDDEN
    for token in required:
        if token not in source:
            issues.append(f"缺少约定: {token}")
    for token in forbidden:
        if token in source:
            issues.append(f"禁止出现: {token}")
    thumbnail = ROOT / "assets" / "thumbnails" / (path.stem + ".png")
    if not thumbnail.exists():
        issues.append(f"缺少封面: {thumbnail}")
    else:
        image = QImage(str(thumbnail))
        if image.isNull():
            issues.append(f"封面无法读取: {thumbnail}")
        elif (image.width(), image.height()) != (520, 520):
            issues.append(
                f"封面尺寸必须为 520x520，当前为 "
                f"{image.width()}x{image.height()}: {thumbnail}"
            )
    return issues
