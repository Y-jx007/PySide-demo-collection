# -*- coding: utf-8 -*-
"""CLI for checking app UI against docs/UI_GUIDELINES.md."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ui_style import check_app_ui  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="检查新应用的 UI 规范")
    parser.add_argument("apps", nargs="+", help="应用文件路径，例如 apps/Physics/Quasicrystal.py")
    args = parser.parse_args()
    failed = False
    for app in args.apps:
        issues = check_app_ui(app)
        if issues:
            failed = True
            print(f"[FAIL] {app}")
            for issue in issues:
                print("  - " + issue)
        else:
            print(f"[OK] {app}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
