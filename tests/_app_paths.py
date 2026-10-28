# -*- coding: utf-8 -*-
"""测试用的应用路径解析。

启动器允许把卡片拖到别的分类，应用文件会跟着挪到 apps/<分类>/；
所以测试不要写死 apps/Physics 之类的目录，统一从 apps_manifest 里查。
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APPS_DIR = os.path.join(ROOT, "apps")

for _path in (ROOT, APPS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from apps_manifest import APPS
except Exception:      # 清单导入失败时退回旧行为，不让测试直接崩掉
    APPS = []


def app_dir(module_name, fallback=None):
    """返回应用当前所在目录（绝对路径）。"""
    for app in APPS:
        path = app.get("path", "")
        if os.path.splitext(os.path.basename(path))[0] == module_name:
            return os.path.join(APPS_DIR, os.path.dirname(path).replace("/", os.sep))
    if fallback:
        return os.path.join(APPS_DIR, fallback)
    return APPS_DIR


def add_app_path(module_name, fallback=None):
    """把应用目录插进 sys.path，返回该目录。"""
    directory = app_dir(module_name, fallback)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return directory


def add_all_app_paths():
    """把所有应用目录都加进 sys.path（连同它们的助手模块）。"""
    directories = {os.path.dirname(app.get("path", "")) for app in APPS}
    for rel in sorted(directories):
        directory = os.path.join(APPS_DIR, rel.replace("/", os.sep))
        if os.path.isdir(directory) and directory not in sys.path:
            sys.path.insert(0, directory)
