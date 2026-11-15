# -*- coding: utf-8 -*-
"""Validate the generated registry, paths, categories and covers."""
import json
import os
import sys
import unittest

from PySide6.QtGui import QImage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import apps_manifest  # noqa: E402


class TestRegistry(unittest.TestCase):
    def test_layout_covers_registry_without_overwriting_user_order(self):
        path = os.path.join(ROOT, "runtime", "launcher_layout.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                on_disk = json.load(handle)
        else:
            # runtime/ is generated state: a fresh clone (CI) has no layout
            # yet, so fall back to the default one the launcher would write.
            on_disk = apps_manifest.build_launcher_layout()
        layout_paths = []
        for category in on_disk["categories"]:
            layout_paths.extend(category["apps"])
        self.assertEqual(len(layout_paths), len(set(layout_paths)))
        self.assertEqual(set(layout_paths), {app["path"] for app in apps_manifest.APPS})
        for relative in layout_paths:
            self.assertTrue(os.path.isfile(os.path.join(ROOT, "apps", relative)), relative)

    def test_app_paths_and_categories(self):
        for app in apps_manifest.APPS:
            self.assertTrue(
                os.path.isfile(os.path.join(ROOT, "apps", app["path"])),
                app["path"],
            )
            self.assertIn(app["category"], apps_manifest.CATEGORY_ORDER)
        self.assertEqual(len({a["path"] for a in apps_manifest.APPS}), len(apps_manifest.APPS))

    def test_covers_exist_and_are_valid(self):
        exemplars = {"Quasicrystal", "Mass Spring", "Particle Lenia"}
        for app in apps_manifest.APPS:
            path = apps_manifest.thumbnail_path(app["path"], ROOT)
            self.assertTrue(path.exists(), str(path))
            image = QImage(str(path))
            self.assertFalse(image.isNull(), str(path))
            if app["title"] in exemplars:
                self.assertEqual((image.width(), image.height()), (520, 520), str(path))

    def test_exemplars_are_polished(self):
        entries = {app["title"]: app for app in apps_manifest.APPS}
        self.assertEqual(entries["Quasicrystal"]["category"], "Math")
        self.assertEqual(entries["Mass Spring"]["category"], "Physics")
        for title in ("Quasicrystal", "Mass Spring"):
            self.assertNotIn("Unpolished/", entries[title]["path"])
        self.assertEqual(entries["Particle Lenia"]["category"], "Cellular Automata")

    def test_matrix_is_generated(self):
        path = os.path.join(ROOT, "docs", "APP_MATRIX.md")
        self.assertTrue(os.path.isfile(path))
        text = open(path, encoding="utf-8").read()
        self.assertIn("应用矩阵", text)
        for app in apps_manifest.APPS:
            self.assertIn(f"**{app['title']}**", text)


if __name__ == "__main__":
    unittest.main()
