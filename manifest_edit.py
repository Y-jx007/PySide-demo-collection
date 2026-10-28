# -*- coding: utf-8 -*-
"""改写 apps_manifest.py 里某个应用的 path / category。

启动器允许把卡片拖到别的分类；文件跟着搬家之后，清单也得同步，
否则 test_manifest 和下次启动都会指向旧路径。这里用 ast 定位那两个
字符串字面量，只改它们，其余源码（注释、排版）原样保留。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parent / "apps_manifest.py"


def _line_starts(source: str) -> list[int]:
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def _offset(starts: list[int], node: ast.AST, *, end: bool = False) -> int:
    line = (node.end_lineno if end else node.lineno) - 1
    col = node.end_col_offset if end else node.col_offset
    return starts[line] + col


def _find_app_node(tree: ast.Module, old_path: str) -> dict[str, ast.Constant] | None:
    for node in tree.body:
        if not (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "APPS" for t in node.targets)):
            continue
        if not isinstance(node.value, ast.List):
            return None
        for item in node.value.elts:
            if not isinstance(item, ast.Dict):
                continue
            fields: dict[str, ast.Constant] = {}
            for key, value in zip(item.keys, item.values):
                if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                    fields[key.value] = value
            path = fields.get("path")
            if path is not None and path.value == old_path:
                return fields
        return None
    return None


def rewrite_app_entry(old_path: str, new_path: str, new_category: str,
                      manifest: str | Path = MANIFEST_PATH) -> bool:
    """把 path == old_path 的那条记录改成 new_path / new_category。

    返回是否成功（找不到条目、清单格式异常时返回 False，文件不动）。
    """
    manifest = Path(manifest)
    try:
        source = manifest.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return False

    fields = _find_app_node(tree, old_path)
    if fields is None:
        return False

    replacements: list[tuple[ast.Constant, str]] = []
    path_node = fields.get("path")
    category_node = fields.get("category")
    if path_node is not None and path_node.value != new_path:
        replacements.append((path_node, new_path))
    if category_node is not None and category_node.value != new_category:
        replacements.append((category_node, new_category))
    if not replacements:
        return True

    starts = _line_starts(source)
    # 从后往前替换，前面的偏移量才不会失效
    for node, value in sorted(replacements, key=lambda item: _offset(starts, item[0]),
                              reverse=True):
        begin = _offset(starts, node)
        finish = _offset(starts, node, end=True)
        source = source[:begin] + json.dumps(value, ensure_ascii=False) + source[finish:]

    try:
        manifest.write_text(source, encoding="utf-8")
    except OSError:
        return False
    return True
