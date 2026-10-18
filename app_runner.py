"""Run one standalone app with the project's import paths configured.

The launcher starts this script in a child process so that every app gets its
own Qt/Taichi/OpenGL/PyTorch runtime.  It can also be used directly:

    python app_runner.py "apps/Fractals/Mandelbrot.py"
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APPS = ROOT / "apps"


def _configure_paths() -> None:
    os.chdir(ROOT)
    candidates = [APPS]
    if APPS.is_dir():
        candidates.extend(sorted(p for p in APPS.iterdir() if p.is_dir() and p.name != "__pycache__"))
    for path in candidates:
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python app_runner.py <path-to-app.py>")
    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        raise SystemExit(f"app not found: {target}")
    _configure_paths()
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
