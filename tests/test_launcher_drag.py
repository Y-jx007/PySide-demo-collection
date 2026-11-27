# -*- coding: utf-8 -*-
"""拖拽语义：同分类重排不搬文件，跨分类拖动才搬文件。

Noise / KAM 的 manifest category 是 ``Unpolished``，文件却在 ``apps/Physics/``；
旧逻辑用“文件所在目录”判断旧分类，在这两个应用上会把同排重排误判成换分类。
"""
import copy
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PySide6.QtWidgets import QApplication  # noqa: E402

import main as launcher_module  # noqa: E402


class LauncherDragTests(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.moves = []
        self.rewrites = []

        def fake_move(src, dst):
            self.moves.append((src, dst))

        def fake_rewrite(old, new, category):
            self.rewrites.append((old, new, category))
            return True

        self._real_move = launcher_module.shutil.move
        self._real_rewrite = launcher_module.rewrite_app_entry
        self._real_apps = launcher_module.APPS
        launcher_module.shutil.move = fake_move
        launcher_module.rewrite_app_entry = fake_rewrite
        # 清单条目会被 _relocate_app_file 就地改写；用副本避免影响同进程里的其他测试。
        launcher_module.APPS = copy.deepcopy(self._real_apps)

        self.launcher = launcher_module.ApplicationLauncher()
        self.launcher._save_layout = lambda: None       # 不写用户的 runtime 布局

    def tearDown(self):
        self.launcher.active_processes.clear()          # 测试没有真的起子进程
        self.launcher.close()
        launcher_module.shutil.move = self._real_move
        launcher_module.rewrite_app_entry = self._real_rewrite
        launcher_module.APPS = self._real_apps
        self.app.processEvents()

    def test_reorder_inside_one_category_never_moves(self):
        for path, card in self.launcher.cards_by_path.items():
            category = card.category_widget
            self.assertIsNotNone(category, path)
            self.assertTrue(
                self.launcher._relocate_app_file(card, category, source=category), path
            )
        self.assertEqual(self.moves, [])
        self.assertEqual(self.rewrites, [])

    def test_kam_and_noise_reorder_never_moves(self):
        target = self.launcher.category_by_name["Unpolished"]
        for path in ("Physics/KAM.py", "Physics/Noise.py"):
            card = self.launcher.cards_by_path[path]
            self.assertTrue(self.launcher._relocate_app_file(card, target, source=target))
        self.assertEqual(self.moves, [])
        self.assertEqual(self.rewrites, [])

    def test_finish_drag_inside_one_category_keeps_file(self):
        target = self.launcher.category_by_name["Unpolished"]
        card = self.launcher.cards_by_path["Physics/KAM.py"]
        self.assertTrue(self.launcher.begin_card_drag(card))
        self.launcher._finish_card_drag(target, index=0)
        self.assertEqual(self.moves, [])
        self.assertEqual(self.rewrites, [])

    def test_cross_category_drag_moves_file(self):
        card = self.launcher.cards_by_path["Math/Fourier.py"]
        source = card.category_widget
        target = self.launcher.category_by_name["Physics"]
        self.assertTrue(self.launcher._relocate_app_file(card, target, source=source))
        self.assertEqual(len(self.moves), 1)
        source_path, destination = self.moves[0]
        self.assertTrue(source_path.endswith(os.path.join("apps", "Math", "Fourier.py")))
        self.assertTrue(destination.endswith(os.path.join("apps", "Physics", "Fourier.py")))
        self.assertEqual(
            self.rewrites, [("Math/Fourier.py", "Physics/Fourier.py", "Physics")]
        )
        self.assertEqual(card.app_path, "Physics/Fourier.py")
        self.assertIs(
            self.launcher.cards_by_path["Physics/Fourier.py"], card
        )

    def test_ready_marker_must_match_child_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "app.ready")
            entry = {"ready": marker, "process": type("Proc", (), {"pid": os.getpid()})()}
            self.assertFalse(self.launcher._ready_matches(entry))
            with open(marker, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            self.assertTrue(self.launcher._ready_matches(entry))
            with open(marker, "w", encoding="utf-8") as handle:
                handle.write("999999")
            self.assertFalse(self.launcher._ready_matches(entry))

    def test_restore_info_reports_starting_tools(self):
        self.launcher.active_processes = [
            {"ready_seen": True, "friendly": "A", "process": None},
            {"ready_seen": False, "friendly": "B", "process": None},
        ]
        self.launcher._restore_info()
        self.assertIn("正在启动 1 个工具", self.launcher.info_label.text())
        self.assertIn("已就绪 1 个", self.launcher.info_label.text())

    def test_log_rotation_keeps_previous_file(self):
        original = launcher_module.LOG_ROTATE_BYTES
        launcher_module.LOG_ROTATE_BYTES = 1
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "app.log")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("old")
                handle = self.launcher._open_log(path)
                handle.write("new")
                handle.close()
                self.assertTrue(os.path.exists(path + ".prev"))
                with open(path + ".prev", encoding="utf-8") as prev:
                    self.assertEqual(prev.read(), "old")
                with open(path, encoding="utf-8") as current:
                    self.assertEqual(current.read(), "new")
        finally:
            launcher_module.LOG_ROTATE_BYTES = original


if __name__ == "__main__":
    unittest.main()
