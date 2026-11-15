# -*- coding: utf-8 -*-
"""Smoke one app window in an isolated process."""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for path in (ROOT, os.path.join(ROOT, "apps")):
    if path not in sys.path:
        sys.path.insert(0, path)

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget


def _classes(module):
    out = []
    for name, obj in vars(module).items():
        if not isinstance(obj, type) or obj.__module__ != module.__name__:
            continue
        try:
            if issubclass(obj, QMainWindow):
                score = 0
            elif issubclass(obj, QWidget) and (
                name.endswith("App") or name.endswith("Window")
            ):
                score = 1
            elif issubclass(obj, QWidget) and name.endswith("Widget"):
                score = 2
            else:
                continue
        except TypeError:
            continue
        out.append((score, name, obj))
    return sorted(out)


def main():
    if len(sys.argv) != 3:
        print("usage: smoke_one_window.py <app-rel-path> <out.png>")
        return 2
    relative, output = sys.argv[1], sys.argv[2]
    app = QApplication.instance() or QApplication([])
    spec = importlib.util.spec_from_file_location(
        "smoke_one", os.path.join(ROOT, "apps", relative)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    window = None
    errors = []
    for _, _, cls in _classes(module):
        try:
            window = cls()
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
    if window is None:
        print("no window class: " + " | ".join(errors[:2]))
        return 3
    window.show()
    app.processEvents()
    QTest.qWait(120)
    app.processEvents()
    canvas = getattr(window, "canvas", None)
    if canvas is None:
        canvas = window.findChild(QWidget) or window
    pix = canvas.grab()
    if pix.isNull() or pix.width() <= 0:
        print("blank canvas")
        return 4
    if not pix.save(output):
        print("cannot save canvas")
        return 5
    window.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
