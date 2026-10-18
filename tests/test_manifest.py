"""Manifest consistency tests: every registered app must exist and be unique."""
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from apps_manifest import APPS, CATEGORY_ORDER  # noqa: E402

APPS_DIR = os.path.join(ROOT, "apps")
RUNNER = os.path.join(ROOT, "app_runner.py")


class TestAppManifest(unittest.TestCase):
    def test_runner_exists(self):
        self.assertTrue(os.path.isfile(RUNNER), RUNNER)

    def test_paths_exist(self):
        for app in APPS:
            path = os.path.join(APPS_DIR, app["path"].replace("/", os.sep))
            self.assertTrue(os.path.isfile(path), f"missing app file: {app['path']}")

    def test_unique_paths_and_titles(self):
        paths = [app["path"] for app in APPS]
        titles = [app["title"] for app in APPS]
        self.assertEqual(len(paths), len(set(paths)), "duplicate app paths")
        self.assertEqual(len(titles), len(set(titles)), "duplicate app titles")

    def test_categories_are_known(self):
        for app in APPS:
            self.assertIn(app["category"], CATEGORY_ORDER)
            self.assertTrue(app["description"])
            self.assertTrue(app["friendly"])

    def test_thumbnails_are_optional(self):
        for app in APPS:
            name = os.path.splitext(os.path.basename(app["path"]))[0] + ".png"
            path = os.path.join(ROOT, "images", name)
            # A missing thumbnail is allowed (the card falls back to a plain card),
            # but an existing one must be a file.
            if os.path.exists(path):
                self.assertTrue(os.path.isfile(path))


if __name__ == "__main__":
    unittest.main()
