# -*- coding: utf-8 -*-
"""项目目录约定：源码、资源、运行期产物的唯一入口。

    <root>/apps/<分类>/<应用>.py      应用入口（apps_manifest 指向这里）
    <root>/assets/data/               数值数据（.npz）
    <root>/assets/presets/            预设库（Turing 参数图、雪花 alpha/terrain 图集）
    <root>/assets/models/             Distill 训练好的 .pth
    <root>/assets/thumbnails/         启动器卡片缩略图
    <root>/assets/style.qss           全局样式表
    <root>/runtime/                   日志、启动器布局、imgui.ini（不入库）

app 里不要再写 parents[2] / "models" 之类的相对路径，统一从这里取。
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = PROJECT_ROOT / "apps"
ASSETS_DIR = PROJECT_ROOT / "assets"

DATA_DIR = ASSETS_DIR / "data"
PRESETS_DIR = ASSETS_DIR / "presets"
MODELS_DIR = ASSETS_DIR / "models"
THUMBNAIL_DIR = ASSETS_DIR / "thumbnails"
STYLE_PATH = ASSETS_DIR / "style.qss"

RUNTIME_DIR = PROJECT_ROOT / "runtime"
LOG_DIR = RUNTIME_DIR / "logs"
LAYOUT_PATH = RUNTIME_DIR / "launcher_layout.json"


def ensure_runtime() -> Path:
    """确保 runtime/ 与 runtime/logs 存在，返回 runtime 目录。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DIR
