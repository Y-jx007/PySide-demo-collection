# -*- coding: utf-8 -*-
"""PySide Toolbox installer.

Small custom installer instead of a 900 MB one-folder PyInstaller bundle:

* copies the application payload;
* creates a per-install venv;
* installs core requirements;
* optionally installs PyTorch (CPU / CUDA 12.4);
* writes launcher + uninstaller + optional desktop shortcut.

The installer itself is a small PyInstaller exe; it needs a system Python
(3.10+) to create the venv, and says so clearly if none is found.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

APP_NAME = "PySide 可视化工具箱"
DEFAULT_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "PySideToolbox"
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
TORCH_CHOICES = {
    "none": "不安装（默认）",
    "auto": "pip 默认 PyTorch",
    "cpu": "PyTorch CPU",
    "cu124": "PyTorch CUDA 12.4",
}
TORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu124": "https://download.pytorch.org/whl/cu124",
}


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def _payload_path() -> Path | None:
    candidate = _base_dir() / "payload.zip"
    return candidate if candidate.is_file() else None


def _dev_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run(cmd, cwd=None, log=None):
    """Run a child process, streaming its output into the installer log."""
    proc = subprocess.Popen(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=CREATE_NO_WINDOW,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line and log:
            log(line)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"命令失败（exit {code}）: {' '.join(map(str, cmd))}")


def _python_candidates():
    if not getattr(sys, "frozen", False):
        yield [sys.executable]
        return
    for name, prefix in (("py", ["-3.14"]), ("py", ["-3"]), ("python", []), ("python3", [])):
        exe = shutil.which(name)
        if exe:
            yield [exe, *prefix]


def find_python(log=None):
    for cmd in _python_candidates():
        try:
            out = subprocess.run(
                [*cmd, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                capture_output=True, text=True, timeout=20,
                creationflags=CREATE_NO_WINDOW,
            )
        except Exception:
            continue
        if out.returncode != 0:
            continue
        version = out.stdout.strip()
        try:
            major, minor = (int(part) for part in version.split(".")[:2])
        except ValueError:
            continue
        if (major, minor) >= (3, 10):
            if log:
                log(f"使用 Python {version}: {' '.join(cmd)}")
            return cmd, version
    raise RuntimeError(
        "没有找到 Python 3.10+。\n\n"
        "请先安装 Python 3.14（勾选 Add python.exe to PATH）：\n"
        "https://www.python.org/downloads/windows/"
    )


def _extract_payload(target: Path, log=None):
    target.mkdir(parents=True, exist_ok=True)
    payload = _payload_path()
    if payload is not None:
        if log:
            log("解压应用文件…")
        with zipfile.ZipFile(payload) as archive:
            archive.extractall(target)
        return
    # Running from the source tree: copy the project, skipping generated dirs.
    root = _dev_root()
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", ".venv", "build", "dist", "dist-*",
        "runtime", "_audit", "installer", "tests",
    )
    if log:
        log(f"从源码目录复制：{root}")
    shutil.copytree(root, target, dirs_exist_ok=True, ignore=ignore)


def _write_launchers(target: Path):
    launcher = target / "启动.bat"
    launcher.write_text(
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        "\".venv\\Scripts\\pythonw.exe\" main.py\r\n",
        encoding="utf-8",
    )
    diagnose = target / "诊断启动.bat"
    diagnose.write_text(
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        "\".venv\\Scripts\\python.exe\" main.py\r\n"
        "pause\r\n",
        encoding="utf-8",
    )
    uninstall_ps1 = target / "uninstall.ps1"
    uninstall_ps1.write_text(
        "$target = Split-Path -Parent $MyInvocation.MyCommand.Path\r\n"
        "$shortcut = Join-Path ([Environment]::GetFolderPath('Desktop')) 'PySide 可视化工具箱.lnk'\r\n"
        "Remove-Item -LiteralPath $shortcut -Force -ErrorAction SilentlyContinue\r\n"
        "Start-Sleep -Seconds 1\r\n"
        "Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue\r\n",
        encoding="utf-8",
    )
    (target / "卸载.bat").write_text(
        "@echo off\r\n"
        "powershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0uninstall.ps1\"\r\n",
        encoding="utf-8",
    )


def _desktop_shortcut(target: Path):
    if sys.platform != "win32":
        return
    icon = target / "assets" / "app_icon.ico"
    script = target / "_shortcut.ps1"
    script.write_text(
        "$ws = New-Object -ComObject WScript.Shell\r\n"
        "$desktop = [Environment]::GetFolderPath('Desktop')\r\n"
        "$sc = $ws.CreateShortcut((Join-Path $desktop 'PySide 可视化工具箱.lnk'))\r\n"
        f"$sc.TargetPath = '{target / '启动.bat'}'\r\n"
        f"$sc.WorkingDirectory = '{target}'\r\n"
        f"$sc.IconLocation = '{icon}'\r\n"
        "$sc.Save()\r\n",
        encoding="utf-8",
    )
    try:
        _run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script])
    finally:
        script.unlink(missing_ok=True)


class InstallerWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(bool, str)

    def __init__(self, target, torch_choice, shortcut, parent=None):
        super().__init__(parent)
        self.target = Path(target)
        self.torch_choice = torch_choice
        self.shortcut = shortcut

    def run(self):
        try:
            self._install()
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.finished_ok.emit(False, str(exc))
        else:
            self.finished_ok.emit(True, str(self.target))

    def _install(self):
        target = self.target
        self.log.emit(f"安装目录：{target}")
        _extract_payload(target, self.log.emit)
        python, version = find_python(self.log.emit)
        (target / "install.json").write_text(
            json.dumps({"python": version, "torch": self.torch_choice},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        venv = target / ".venv"
        if not (venv / "Scripts" / "python.exe").is_file():
            self.log.emit("创建虚拟环境…")
            _run([*python, "-m", "venv", venv], log=self.log.emit)
        vpy = venv / "Scripts" / "python.exe"

        self.log.emit("安装核心依赖（PySide6 / numpy / taichi-forge / …）…")
        _run([vpy, "-m", "pip", "install", "--disable-pip-version-check",
              "-r", target / "requirements.txt"], cwd=target, log=self.log.emit)

        if self.torch_choice != "none":
            args = [vpy, "-m", "pip", "install", "--disable-pip-version-check", "torch"]
            index = TORCH_INDEX.get(self.torch_choice)
            if index:
                args += ["--index-url", index]
            self.log.emit(f"安装 PyTorch（{TORCH_CHOICES[self.torch_choice]}）…")
            _run(args, cwd=target, log=self.log.emit)

        _write_launchers(target)
        if self.shortcut:
            self.log.emit("创建桌面快捷方式…")
            _desktop_shortcut(target)
        self.log.emit("安装完成。")


class InstallerWindow(QWidget):
    def __init__(self, silent=False):
        super().__init__()
        self.silent = silent
        self.worker = None
        self.setWindowTitle(APP_NAME + " 安装器")
        icon = Path(__file__).resolve().parent.parent / "assets" / "app_icon.ico"
        if icon.is_file():
            self.setWindowIcon(QIcon(str(icon)))
        self.setMinimumWidth(620)

        root = QVBoxLayout(self)
        title = QLabel(f"<b>{APP_NAME}</b> · 自定义安装")
        root.addWidget(title)

        form = QFormLayout()
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(str(DEFAULT_DIR))
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        form.addRow("安装目录", path_row)

        self.torch_combo = QComboBox()
        for key, label in TORCH_CHOICES.items():
            self.torch_combo.addItem(label, key)
        form.addRow("PyTorch（可选）", self.torch_combo)

        self.shortcut = QCheckBox("创建桌面快捷方式")
        self.shortcut.setChecked(True)
        form.addRow("", self.shortcut)
        root.addLayout(form)

        note = QLabel(
            "PyTorch 仅 Diffusion / Mordvintsev 必需；Nonlinear 没有 torch 会自动回退 NumPy。"
        )
        note.setWordWrap(True)
        root.addWidget(note)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(240)
        root.addWidget(self.log)

        row = QHBoxLayout()
        self.install_btn = QPushButton("开始安装")
        self.install_btn.clicked.connect(self._install)
        self.close_btn = QPushButton("退出")
        self.close_btn.clicked.connect(self.close)
        row.addStretch(1)
        row.addWidget(self.install_btn)
        row.addWidget(self.close_btn)
        root.addLayout(row)

    def _browse(self):
        chosen = QFileDialog.getExistingDirectory(self, "选择安装目录", self.path_edit.text())
        if chosen:
            self.path_edit.setText(chosen)

    def _append(self, text):
        self.log.append(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _install(self):
        target = Path(self.path_edit.text().strip() or str(DEFAULT_DIR))
        if target.exists() and any(target.iterdir()):
            answer = QMessageBox.question(
                self, "目录非空", f"{target}\n目录已有文件，继续会覆盖应用文件（保留 .venv）。继续？"
            )
            if answer != QMessageBox.Yes:
                return
        self.install_btn.setEnabled(False)
        self.close_btn.setEnabled(False)
        self.worker = InstallerWorker(
            target, self.torch_combo.currentData(), self.shortcut.isChecked(), self
        )
        self.worker.log.connect(self._append)
        self.worker.finished_ok.connect(self._done)
        self.worker.start()

    def _done(self, ok, message):
        self.install_btn.setEnabled(True)
        self.close_btn.setEnabled(True)
        if not ok:
            self._append("安装失败：" + message)
            QMessageBox.critical(self, "安装失败", message)
            return
        self._append("完成：" + message)
        answer = QMessageBox.question(
            self, "安装完成", "是否现在启动？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(message) / "启动.bat")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--torch", choices=list(TORCH_CHOICES), default="none")
    parser.add_argument("--no-shortcut", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        payload = _payload_path()
        print("payload:", payload if payload else "source tree")
        print("python:", find_python(print))
        return 0

    if args.silent:
        worker = InstallerWorker(Path(args.dir), args.torch, not args.no_shortcut)
        worker.log.connect(print)
        try:
            worker._install()
        except Exception as exc:  # noqa: BLE001
            print("FAILED:", exc)
            return 1
        print("OK:", args.dir)
        return 0

    app = QApplication.instance() or QApplication(sys.argv)
    window = InstallerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
