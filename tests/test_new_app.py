# -*- coding: utf-8 -*-
"""The app scaffolder: path parsing and a template that passes the UI checks."""
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.ui_style import FORBIDDEN, REQUIRED  # noqa: E402
from tools import new_app  # noqa: E402


class TestNormalisePath(unittest.TestCase):
    def test_accepts_folder_and_file_forms(self):
        for raw in ("Math/Matrix", "Math/Matrix.py", "Math\\Matrix.py", "/Math/Matrix"):
            self.assertEqual(new_app.normalise_path(raw), ("Math", "Matrix"))

    def test_rejects_paths_without_a_category(self):
        for raw in ("Matrix", "Math/Sub/Matrix", ""):
            with self.assertRaises(SystemExit):
                new_app.normalise_path(raw)


class TestTemplate(unittest.TestCase):
    def setUp(self):
        self.source = new_app.TEMPLATE.format(
            title="Demo", description="一句话说明", cls="Demo", window_title="Demo",
        )

    def test_has_every_required_ui_token(self):
        for token in REQUIRED:
            self.assertIn(token, self.source, token)

    def test_has_no_forbidden_ui_token(self):
        for token in FORBIDDEN:
            self.assertNotIn(token, self.source, token)

    def test_compiles_as_python(self):
        compile(self.source, "<template>", "exec")

    def test_placeholder_braces_are_escaped(self):
        # a stray {v} in the template would raise KeyError while formatting
        self.assertEqual(self.source.count("{"), self.source.count("}"))


if __name__ == "__main__":
    unittest.main()