# -*- coding: utf-8 -*-
"""Smoke the split launcher and its generated card set."""
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import apps_manifest  # noqa: E402
import main as launcher_module  # noqa: E402


class TestLauncherSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_launcher_builds_all_cards(self):
        window = launcher_module.ApplicationLauncher()
        window.show()
        QTest.qWait(100)
        self.app.processEvents()
        self.assertEqual(len(window.cards_by_path), len(apps_manifest.APPS))
        window.close()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
