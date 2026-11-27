# PySide 可视化工具箱安装器

一个自定义安装器，避免把 PySide6 + Taichi + 全部应用打成一个 900 MB 的
PyInstaller 目录。安装器负责：

- 复制应用源码和资源；
- 用系统 Python 创建独立的 `.venv`；
- 安装 `requirements.txt`；
- PyTorch 作为**可选组件**（不装 / pip 默认 / CPU / CUDA 12.4）；
- 生成 `启动.bat`、`诊断启动.bat`、`卸载.bat`，可选桌面快捷方式。

## 使用

需要 Windows + Python 3.10+（推荐 3.14）和一个能联网的环境。安装器本身是
一个 80 MB 左右的单文件 exe，不需要预装 PySide6/Taichi。

运行 `PySideToolbox-Setup.exe`，选择安装目录和 PyTorch 选项即可。安装完成后
双击 `启动.bat` 或桌面快捷方式。PyTorch 不装时，Diffusion/Mordvintsev 不可用；
Nonlinear 会自动回退 NumPy FFT。

## 构建

先装 PyInstaller，再运行：

```powershell
python -m pip install pyinstaller
python installer/build_setup.py
```

输出：`dist-installer/PySideToolbox-Setup.exe`。

## 说明

- 没有把 Python 解释器打进安装器；找不到 3.10+ 时会给出 python.org 链接。
- 核心依赖使用 `PySide6-Essentials`，不安装 WebEngine/Addons/Pdf，安装体积明显更小。
- 卸载直接运行安装目录里的 `卸载.bat`；它会删除桌面快捷方式和整个安装目录。
