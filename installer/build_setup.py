# -*- coding: utf-8 -*-
"""Build the custom PySide Toolbox installer exe.

Creates a compact payload zip (source + assets, ~40 MB), then wraps
installer/installer.py with PyInstaller.  The app itself is not bundled; the
installer creates a venv and installs requirements, with PyTorch optional.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "installer"
PAYLOAD = BUILD / "payload.zip"
DIST = ROOT / "dist-installer"

INCLUDE = [
    "main.py", "app_runner.py", "apps_manifest.py", "manifest_edit.py",
    "launcher_widgets.py", "requirements.txt", "pyproject.toml", "README.md",
    "apps", "assets", "common", "support", "docs", "tools",
]
EXCLUDE_PARTS = {
    "__pycache__", ".git", ".venv", "runtime", "build", "dist", "dist-installer",
    "installer", "_audit", "tests",
}


def _iter_files(root: Path):
    for name in INCLUDE:
        path = root / name
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and not (set(child.parts) & EXCLUDE_PARTS):
                    yield child


def build_payload():
    BUILD.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PAYLOAD, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in _iter_files(ROOT):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"payload: {PAYLOAD} ({PAYLOAD.stat().st_size / 1024 / 1024:.1f} MB)")


def build_setup():
    DIST.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--windowed", "--onefile", "--name", "PySideToolbox-Setup",
        "--icon", str(ROOT / "assets" / "app_icon.ico"),
        "--distpath", str(DIST), "--workpath", str(BUILD / "pyinstaller"),
        "--specpath", str(BUILD),
        "--add-data", f"{PAYLOAD};.",
        str(ROOT / "installer" / "installer.py"),
    ]
    subprocess.run(args, check=True)
    exe = DIST / "PySideToolbox-Setup.exe"
    print(f"installer: {exe} ({exe.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    build_payload()
    build_setup()
