# -*- coding: utf-8 -*-
"""Scaffold a new app: skeleton, manifest entry, cover and generated docs.

    python tools/new_app.py Math/Matrix --title Matrix --description "2x2 线性变换"

The generated file already satisfies tools/check_ui_style.py (960x640 window,
330 left panel, 600x600 white canvas, QGroupBox + LabeledSlider sections, the
standard 重置/保存 buttons and a status bar), the manifest entry is inserted at
the top of its category, docs/APP_MATRIX.md is regenerated and a 520x520 cover
is rendered from the fresh canvas.

After it runs, replace the placeholder drawing with the real one and add tests
under tests/.
"""
import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import apps_manifest  # noqa: E402

TEMPLATE = '''# -*- coding: utf-8 -*-
"""{title}: {description}."""
import math

from common.lab_window import LabWindow, SquareCanvas
from common.qt_base import (
    LabeledSlider, QColor, QGroupBox, QHBoxLayout, QLabel, QPainter, QPen,
    QPointF, QPushButton, QVBoxLayout, QWidget, Qt, run_app,
)


class {cls}Canvas(SquareCanvas):
    """Placeholder canvas: swap the drawing for the real one."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.setFixedSize(600, 600)
        self.window = window
        self.phase = 0.35

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        step = 40
        painter.setPen(QPen(QColor(232, 232, 232), 1))
        for x in range(0, self.width(), step):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            painter.drawLine(0, y, self.width(), y)
        centre = QPointF(self.width() / 2.0, self.height() / 2.0)
        radius = 120.0 * self.phase
        painter.setPen(QPen(QColor(24, 96, 150), 2.2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, radius, radius)
        painter.drawLine(
            QPointF(centre.x() - 150, centre.y()),
            QPointF(centre.x() + 150, centre.y()),
        )
        painter.end()


class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("{window_title}")
        self.setStyleSheet("QMainWindow {{ background-color: #ffffff; }}")
        self.canvas = {cls}Canvas(self)
        self.init_ui()
        self.setMinimumSize(960, 640)
        self.resize(960, 640)
        self.setFocusPolicy(Qt.StrongFocus)

    def init_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(330)
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        shape_box = QGroupBox("参数")
        shape_layout = QVBoxLayout(shape_box)
        self.phase_slider = LabeledSlider(
            "尺度", 10, 100, 35, formatter=lambda value: str(value) + "%"
        )
        self.phase_slider.valueChanged.connect(self.on_phase)
        shape_layout.addWidget(self.phase_slider)
        column.addWidget(shape_box)
        column.addStretch(1)

        buttons = QHBoxLayout()
        reset_button = QPushButton("重置")
        reset_button.clicked.connect(self.reset)
        save_button = QPushButton("保存")
        save_button.clicked.connect(self.save)
        buttons.addWidget(reset_button)
        buttons.addWidget(save_button)
        column.addLayout(buttons)

        root.addWidget(panel)
        root.addWidget(self.canvas, 1)
        self.setCentralWidget(central)

        self.phase_label = QLabel("")
        self.statusBar().addWidget(self.phase_label)
        self.update_readout()

    def on_phase(self, value):
        self.canvas.phase = value / 100.0
        self.update_readout()
        self.canvas.update()

    def update_readout(self):
        self.phase_label.setText("尺度 %d%%" % int(self.canvas.phase * 100))

    def reset(self):
        self.phase_slider.setValue(35)
        self.on_phase(35)

    def save(self):
        path = "{cls}.png"
        self.canvas.save_image(path)


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
'''


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="生成一个新应用的骨架并登记到清单")
    parser.add_argument("path", help="应用路径，例如 Math/Matrix 或 Math/Matrix.py")
    parser.add_argument("--title", required=True, help="清单里显示的标题")
    parser.add_argument("--description", required=True, help="一句话说明")
    parser.add_argument("--friendly", default="", help="启动器搜索用的名字")
    parser.add_argument("--category", default="", help="分类，默认取路径的第一段")
    parser.add_argument("--no-cover", action="store_true", help="先不渲染封面")
    return parser.parse_args(argv)


def normalise_path(raw):
    text = raw.replace("\\", "/").strip("/")
    if text.endswith(".py"):
        text = text[:-3]
    parts = [p for p in text.split("/") if p]
    if len(parts) != 2:
        raise SystemExit("路径要写成 <分类>/<名称>，例如 Math/Matrix")
    return parts[0], parts[1]


def insert_manifest_entry(category, entry):
    """Insert the entry at the top of its category block, then verify it parses."""
    path = ROOT / "apps_manifest.py"
    source = path.read_text(encoding="utf-8")
    marker = "# ---------------- %s ----------------" % category
    if marker not in source:
        raise SystemExit("apps_manifest.py 里找不到分类块: " + category)
    block = (
        "    {\n"
        '        "title": "%s",\n'
        '        "description": "%s",\n'
        '        "path": "%s",\n'
        '        "friendly": "%s",\n'
        '        "category": "%s",\n'
        "    },\n"
    ) % (
        entry["title"], entry["description"], entry["path"],
        entry["friendly"], entry["category"],
    )
    head, tail = source.split(marker, 1)
    updated = head + marker + "\n" + block + tail.lstrip("\n")
    ast.parse(updated)
    path.write_text(updated, encoding="utf-8")
    return path


def render_cover(app_path, cover_path):
    """Run the new app offscreen and save its canvas as the 520x520 cover."""
    script = (
        "import importlib.util, os, sys\n"
        "os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
        "sys.path.insert(0, %r)\n"
        "from PySide6.QtTest import QTest\n"
        "from PySide6.QtWidgets import QApplication\n"
        "from PySide6.QtCore import Qt\n"
        "app = QApplication([])\n"
        "spec = importlib.util.spec_from_file_location('scaffold_app', %r)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "window = module.MainWindow()\n"
        "window.show()\n"
        "QTest.qWait(220)\n"
        "app.processEvents()\n"
        "pixmap = window.canvas.grab()\n"
        "image = pixmap.toImage().scaled(520, 520, Qt.IgnoreAspectRatio,\n"
        "                                Qt.SmoothTransformation)\n"
        "image.save(%r)\n"
        "window.close()\n"
    ) % (str(ROOT), str(app_path), str(cover_path))
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=str(ROOT),
        capture_output=True, text=True, timeout=120,
    )
    return result.returncode == 0 and Path(cover_path).exists(), result


def main(argv=None):
    args = parse_args(argv)
    category, name = normalise_path(args.path)
    category = args.category or category
    if category not in apps_manifest.CATEGORY_ORDER:
        raise SystemExit(
            "未知分类 %s，可选: %s" % (category, ", ".join(apps_manifest.CATEGORY_ORDER))
        )
    relative = "%s/%s.py" % (category, name)
    app_path = ROOT / "apps" / relative
    if app_path.exists():
        raise SystemExit("应用已存在: " + str(app_path))
    if any(app["path"] == relative for app in apps_manifest.APPS):
        raise SystemExit("清单里已经有这个路径: " + relative)

    cls = "".join(ch for ch in name if ch.isalnum())
    if not cls or not cls[0].isalpha():
        raise SystemExit("应用文件名要以字母开头、只含字母数字: " + name)
    app_path.parent.mkdir(parents=True, exist_ok=True)
    app_path.write_text(
        TEMPLATE.format(
            title=name,
            description=args.description,
            cls=cls,
            window_title=args.title,
        ),
        encoding="utf-8",
    )
    print("应用骨架: %s" % app_path.relative_to(ROOT))

    insert_manifest_entry(category, {
        "title": args.title,
        "description": args.description,
        "path": relative,
        "friendly": args.friendly or args.title,
        "category": category,
    })
    print("清单条目: %s (%s)" % (relative, category))

    subprocess.run([sys.executable, str(ROOT / "tools" / "sync_registry.py")],
                   cwd=str(ROOT), check=False)
    print("已同步 docs/APP_MATRIX.md")

    cover = ROOT / "assets" / "thumbnails" / (name + ".png")
    if args.no_cover:
        print("跳过封面: 需要 520x520 的 %s" % cover.relative_to(ROOT))
    else:
        cover.parent.mkdir(parents=True, exist_ok=True)
        ok, result = render_cover(app_path, cover)
        if ok:
            print("封面: %s" % cover.relative_to(ROOT))
        else:
            print("封面渲染失败，先留空：\n" + (result.stdout + result.stderr)[-800:])

    from common.ui_style import check_app_ui
    issues = check_app_ui(app_path)
    if issues:
        print("UI 检查:")
        for issue in issues:
            print("  - " + issue)
    else:
        print("UI 检查: [OK]")
    print("\n下一步: 把占位绘制换成真实图形，补 tests/test_%s.py，"
          "然后跑 python -m unittest discover -s tests" % name.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
