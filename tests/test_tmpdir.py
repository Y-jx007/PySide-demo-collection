# -*- coding: utf-8 -*-
"""The repo-local scratch helper used by the other tests."""
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests import tmpdir  # noqa: E402


class TestTmpDir(unittest.TestCase):
    def test_directory_is_writable_and_cleaned_up(self):
        with tmpdir.temporary_directory() as tmp:
            target = os.path.join(tmp, "probe.txt")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("ok")
            self.assertTrue(os.path.isfile(target))
            created = tmp
        self.assertFalse(os.path.exists(created))

    def test_mkdtemp_makes_unique_directories(self):
        first = tmpdir.mkdtemp()
        second = tmpdir.mkdtemp()
        try:
            self.assertNotEqual(first, second)
            self.assertTrue(os.path.isdir(first))
        finally:
            for path in (first, second):
                import shutil
                shutil.rmtree(path, ignore_errors=True)

    def test_root_is_under_the_repository_when_writable(self):
        root = tmpdir._ensure_root()
        if root is None:
            self.skipTest("checkout is read-only, falls back to the system temp")
        self.assertEqual(os.path.dirname(root), os.path.join(ROOT, "tests"))


if __name__ == "__main__":
    unittest.main()