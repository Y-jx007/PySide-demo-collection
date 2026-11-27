# -*- coding: utf-8 -*-
"""Integrity tests for assets/data/conway_presets.npz (no UI required)."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import DATA_DIR

NPZ = DATA_DIR / "conway_presets.npz"


def load():
    with np.load(NPZ, allow_pickle=False) as data:
        return {
            "keys": [str(v) for v in data["keys"]],
            "shorts": [str(v) for v in data["shorts"]],
            "names": [str(v) for v in data["names"]],
            "categories": [str(v) for v in data["categories"]],
            "periods": [str(v) for v in data["periods"]],
            "sources": [str(v) for v in data["sources"]],
            "category_order": [str(v) for v in data["category_order"]],
            "default": str(data["default"][0]),
            "grids": {
                key: np.asarray(data[f"preset_{key}"], dtype=bool)
                for key in [str(v) for v in data["keys"]]
            },
        }


def test_archive_metadata():
    d = load()
    assert NPZ.is_file()
    assert len(d["keys"]) == 32
    assert d["default"] == "gosper-glider-gun"
    assert d["category_order"] == [
        "Still life", "Oscillator", "Spaceship", "Methuselah", "Gun", "Puffer",
    ]
    counts = {c: d["categories"].count(c) for c in d["category_order"]}
    assert counts == {
        "Still life": 7,
        "Oscillator": 8,
        "Spaceship": 8,
        "Methuselah": 5,
        "Gun": 3,
        "Puffer": 1,
    }


def test_grids_are_valid():
    d = load()
    for key, grid in d["grids"].items():
        assert grid.ndim == 2 and grid.size > 0 and grid.any(), key
        assert grid.dtype == np.bool_, key
        assert len(grid) == grid.shape[0] and grid.shape[1] > 0, key


def test_known_bboxes_and_cells():
    d = load()
    expected = {
        "block": (2, 2, 4),
        "glider": (3, 3, 5),
        "gosper-glider-gun": (36, 9, 36),
        "simkin-glider-gun": (33, 21, 36),
        "copperhead": (8, 12, 28),
        "weekender": (16, 11, 36),
        "pulsar": (13, 13, 48),
        "pentadecathlon": (10, 3, 12),
        "acorn": (7, 3, 7),
    }
    for key, (width, height, cells) in expected.items():
        grid = d["grids"][key]
        assert (grid.shape[1], grid.shape[0], int(grid.sum())) == (width, height, cells), key


def test_metadata_fields():
    d = load()
    for i, key in enumerate(d["keys"]):
        assert d["shorts"][i] and d["names"][i] and d["periods"][i], key
        assert d["sources"][i].startswith("https://conwaylife.com/wiki/"), key
