# -*- coding: utf-8 -*-
"""Temporary directories inside the repository.

`tempfile` is avoided on purpose: `TemporaryDirectory` chmods its directory
on cleanup and %TEMP% is not writable in every sandbox (the restricted
runners used during development deny it).  Tests therefore keep their scratch
space in `tests/tmp/`, which is disposable and ignored by git.
"""
import os
import shutil
import tempfile
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")


def _ensure_root():
    """Scratch root, or None when this checkout is read-only."""
    try:
        os.makedirs(TMP_ROOT, exist_ok=True)
        probe = os.path.join(TMP_ROOT, "probe.tmp")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe)
    except OSError:
        return None
    return TMP_ROOT


def mkdtemp(prefix="case-"):
    """Create a unique directory under `tests/tmp/` (or the system temp)."""
    root = _ensure_root() or tempfile.gettempdir()
    path = os.path.join(root, prefix + uuid.uuid4().hex[:8])
    os.makedirs(path)
    return path


class TemporaryDirectory:
    """Minimal stand-in for `tempfile.TemporaryDirectory` (no chmod)."""

    def __init__(self, prefix="case-"):
        self.name = mkdtemp(prefix)

    def __enter__(self):
        return self.name

    def __exit__(self, *exc_info):
        shutil.rmtree(self.name, ignore_errors=True)
        return False


def temporary_directory(prefix="case-"):
    """Context manager version, rooted in `tests/tmp/`."""
    return TemporaryDirectory(prefix)