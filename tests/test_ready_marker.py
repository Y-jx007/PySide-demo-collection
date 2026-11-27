# -*- coding: utf-8 -*-
"""启动器就绪握手：子进程在事件循环跑起来后写 pid 标记。"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from common.app import READY_ENV, announce_ready  # noqa: E402


class ReadyMarkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._saved_env = os.environ.get(READY_ENV)

    def tearDown(self):
        if self._saved_env is None:
            os.environ.pop(READY_ENV, None)
        else:
            os.environ[READY_ENV] = self._saved_env

    def test_writes_pid_when_env_is_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "app.ready")
            os.environ[READY_ENV] = marker
            self.assertTrue(announce_ready(0))
            QTest.qWait(50)
            self.assertTrue(os.path.exists(marker))
            with open(marker, encoding="utf-8") as handle:
                self.assertEqual(handle.read().strip(), str(os.getpid()))

    def test_no_marker_without_env(self):
        os.environ.pop(READY_ENV, None)
        self.assertFalse(announce_ready(0))

    def test_missing_directory_is_ignored(self):
        os.environ[READY_ENV] = os.path.join(tempfile.gettempdir(), "no-such-dir-xyz", "a.ready")
        self.assertTrue(announce_ready(0))
        QTest.qWait(30)          # 不该抛异常；目录不存在就静默放弃


if __name__ == "__main__":
    unittest.main()
