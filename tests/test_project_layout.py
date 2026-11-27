# -*- coding: utf-8 -*-
"""Guard rails for the project directory conventions in AGENTS.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPS = ROOT / "apps"
DATA = ROOT / "assets" / "data"


def test_no_bak_files_in_tree():
    assert not list(ROOT.glob("*.bak"))
    assert not list(ROOT.glob("*.bak-*"))
    assert not list(APPS.rglob("*.bak"))
    assert not list(APPS.rglob("*.bak-*"))


def test_conway_presets_live_in_assets_data():
    assert (DATA / "conway_presets.npz").is_file()
    assert not (APPS / "Cellular Automata" / "conway_presets").exists()


def test_no_app_local_directories():
    """Apps are single files; no per-app packages/data/build folders."""
    for category in APPS.iterdir():
        if not category.is_dir():
            continue
        for child in category.iterdir():
            if child.is_dir() and child.name != "__pycache__":
                raise AssertionError(
                    f"unexpected app-local directory: {child.relative_to(ROOT)}"
                )
