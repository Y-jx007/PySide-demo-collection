# -*- coding: utf-8 -*-
"""Regenerate the app matrix from apps_manifest.py.

runtime/launcher_layout.json is owned by the launcher (drag order/categories)
and is deliberately not overwritten here.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps_manifest import generated_payloads, write_generated  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="只检查生成文件是否最新，不写入",
    )
    args = parser.parse_args()
    if args.check:
        stale = [
            str(path)
            for path, content in generated_payloads(ROOT).items()
            if not path.exists() or path.read_text(encoding="utf-8") != content
        ]
        if stale:
            print("注册表生成文件已过期，请运行 python tools/sync_registry.py")
            for path in stale:
                print("  - " + path)
            return 1
        print("注册表生成文件已同步")
        return 0
    changed = write_generated(ROOT)
    if not changed:
        print("注册表已是最新")
    else:
        for path in changed:
            print("已更新 " + str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
