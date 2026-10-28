"""manifest_edit：跨分类拖卡片时改写 apps_manifest.py 的单元测试。"""
import ast
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from manifest_edit import rewrite_app_entry  # noqa: E402


def load_apps(path):
    namespace = {}
    exec(compile(Path(path).read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace["APPS"]


class ManifestEditTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.manifest = self.tmpdir / "apps_manifest.py"
        shutil.copy2(ROOT / "apps_manifest.py", self.manifest)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rewrite_path_and_category(self):
        before = load_apps(self.manifest)
        self.assertTrue(rewrite_app_entry("Math/Traveler.py", "Physics/Traveler.py",
                                          "Physics", self.manifest))
        after = load_apps(self.manifest)
        moved = [app for app in after if app["path"] == "Physics/Traveler.py"]
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved[0]["category"], "Physics")
        self.assertFalse([app for app in after if app["path"] == "Math/Traveler.py"])
        self.assertEqual(len(after), len(before))
        # 其余条目保持原值
        untouched = [app for app in after if app["title"] == moved[0]["title"]][0]
        self.assertEqual(untouched["friendly"], "旅行商问题遗传算法")
        ast.parse(self.manifest.read_text(encoding="utf-8"))

    def test_unknown_path_returns_false(self):
        self.assertFalse(rewrite_app_entry("Nope/X.py", "Math/X.py", "Math", self.manifest))

    def test_idempotent(self):
        self.assertTrue(rewrite_app_entry("Math/Traveler.py", "Physics/Traveler.py",
                                          "Physics", self.manifest))
        self.assertTrue(rewrite_app_entry("Physics/Traveler.py", "Physics/Traveler.py",
                                          "Physics", self.manifest))
        apps = load_apps(self.manifest)
        self.assertEqual(len([a for a in apps if a["path"].endswith("Traveler.py")]), 1)
