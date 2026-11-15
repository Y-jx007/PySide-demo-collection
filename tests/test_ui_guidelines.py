# -*- coding: utf-8 -*-
"""Enforce the UI standard on the canonical rebuilt apps."""
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.ui_style import check_app_ui  # noqa: E402

EXEMPLARS = (
    "apps/Math/Quasicrystal.py",
    "apps/Physics/Mass-Spring.py",
    "apps/Cellular Automata/Particle Lenia.py",
    "apps/Cellular Automata/Distill.py",
)


class TestUIGuidelines(unittest.TestCase):
    def test_exemplars_follow_ui_standard(self):
        for relative in EXEMPLARS:
            issues = check_app_ui(os.path.join(ROOT, relative))
            self.assertEqual(issues, [], f"{relative} 不符合 UI 规范: {issues}")

    def test_exemplars_are_registered_in_polished_categories(self):
        import apps_manifest

        expected = {
            "Quasicrystal": ("Math/Quasicrystal.py", "Math"),
            "Mass Spring": ("Physics/Mass-Spring.py", "Physics"),
            "Particle Lenia": ("Cellular Automata/Particle Lenia.py", "Cellular Automata"),
            "Distill": ("Cellular Automata/Distill.py", "Cellular Automata"),
        }
        entries = {app["title"]: app for app in apps_manifest.APPS}
        for title, (path, category) in expected.items():
            self.assertIn(title, entries)
            self.assertEqual(entries[title]["path"], path)
            self.assertEqual(entries[title]["category"], category)

    def test_distill_uses_two_view_exception(self):
        path = os.path.join(ROOT, "apps", "Cellular Automata", "Distill.py")
        source = open(path, encoding="utf-8").read()
        self.assertIn('UI_LAYOUT = "two_view"', source)
        self.assertNotIn("setFixedWidth(330)", source)
        self.assertNotIn("setFixedSize(600, 600)", source)
        self.assertIn("TwoViewCanvas(", source)
        self.assertEqual(check_app_ui(path), [])

    def test_guidelines_are_linked_from_readme(self):
        path = os.path.join(ROOT, "docs", "UI_GUIDELINES.md")
        self.assertTrue(os.path.exists(path))
        readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
        self.assertIn("docs/UI_GUIDELINES.md", readme)


if __name__ == "__main__":
    unittest.main()
