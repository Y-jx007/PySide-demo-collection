"""Run one standalone app with the project's import paths configured.

The launcher starts this script in a child process so that every app gets its
own Qt/Taichi/OpenGL/PyTorch runtime.  It can also be used directly:

    python app_runner.py "apps/Fractals/Mandelbrot.py"

When the launcher sets ``PYSIDE_TOOLS_SPLASH=1`` this module shows a small
"splash" window while the app module imports and compiles its kernels, then
closes it as soon as the real window is visible.  GL apps are skipped: they
need to install their QSurfaceFormat before any QApplication exists.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APPS = ROOT / "apps"

_SPLASH = None


def _configure_paths() -> None:
    os.chdir(ROOT)
    candidates = [ROOT, APPS]
    if APPS.is_dir():
        candidates.extend(sorted(p for p in APPS.iterdir() if p.is_dir() and p.name != "__pycache__"))
    for path in candidates:
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _wants_splash(target: Path) -> bool:
    if os.environ.get("PYSIDE_TOOLS_SPLASH") != "1":
        return False
    try:
        source = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    # gl=True 的应用要先设置 QSurfaceFormat，不能提前建 QApplication。
    return "gl=True" not in source


def _show_splash(title: str) -> None:
    """Show a frameless hint before the app module is imported."""
    global _SPLASH
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import QApplication, QLabel
    except Exception:
        return
    try:
        app = QApplication.instance() or QApplication([])
    except Exception:
        return
    label = QLabel(f"正在启动 {title}…\n首次编译内核可能需要几秒")
    label.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
    label.setAlignment(Qt.AlignCenter)
    label.setFixedSize(380, 92)
    label.setStyleSheet(
        "background:#f8f2e7; color:#3b352b; border:1px solid #cbbf9f;"
        "border-radius:6px; font-size:13px;"
    )
    screen = app.primaryScreen()
    if screen is not None:
        area = screen.availableGeometry()
        label.move(area.center().x() - label.width() // 2,
                   area.center().y() - label.height() // 2)
    label.show()
    app.processEvents()
    _SPLASH = label

    timer = QTimer(label)

    def close_when_window_appears():
        for widget in app.topLevelWidgets():
            if widget is label or not widget.isVisible():
                continue
            if widget.width() >= 240 and widget.height() >= 160:
                label.close()
                timer.stop()
                return
        if not label.isVisible():
            timer.stop()

    timer.timeout.connect(close_when_window_appears)
    timer.start(150)
    # 应用一直不出窗口也别让 splash 永久停留（异常时由进程退出收尾）。
    QTimer.singleShot(30000, label.close)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python app_runner.py <path-to-app.py>")
    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        raise SystemExit(f"app not found: {target}")
    _configure_paths()
    if _wants_splash(target):
        _show_splash(target.stem)
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
