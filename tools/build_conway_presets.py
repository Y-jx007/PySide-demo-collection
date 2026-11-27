# -*- coding: utf-8 -*-
"""Build the runtime Conway preset archive from the RLE source library.

Source data lives in tools/conway_rle/ (manifest.json + rle/*.rle).
Runtime data is one compressed NumPy archive:

    apps/Cellular Automata/conway_presets/conway_presets.npz

The archive stores each grid as a boolean array plus string metadata arrays.
No pickle is used; the app loads it with allow_pickle=False.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tools" / "conway_rle"
OUT = ROOT / "assets" / "data" / "conway_presets.npz"


def parse_rle(text: str) -> np.ndarray:
    """Parse a Life RLE string into a boolean grid (no external deps)."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("x") and "=" in line:
            continue
        lines.append(line)
    body = "".join(lines).split("!", 1)[0]

    rows: list[list[int]] = []
    row: list[int] = []
    count = 0
    for ch in body:
        if ch.isdigit():
            count = count * 10 + int(ch)
            continue
        n = count or 1
        count = 0
        if ch in "bo":
            row.extend([1 if ch == "o" else 0] * n)
        elif ch == "$":
            rows.append(row)
            row = []
            for _ in range(n - 1):
                rows.append([])
        elif ch == "!":
            break
    if row:
        rows.append(row)
    width = max((len(r) for r in rows), default=0)
    grid = np.zeros((len(rows), width), dtype=np.bool_)
    for y, values in enumerate(rows):
        if values:
            grid[y, :len(values)] = np.asarray(values, dtype=np.bool_)
    return grid


def main() -> int:
    manifest = json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["presets"]
    metadata = {
        "keys": [],
        "shorts": [],
        "names": [],
        "categories": [],
        "periods": [],
        "authors": [],
        "descriptions": [],
        "sources": [],
    }
    grids = {}
    for entry in entries:
        rle_path = SRC / entry["file"]
        grid = parse_rle(rle_path.read_text(encoding="utf-8-sig"))
        if grid.size == 0 or not grid.any():
            raise SystemExit(f"empty preset: {entry['key']} ({rle_path})")
        key = entry["key"]
        grids[f"preset_{key}"] = grid
        metadata["keys"].append(key)
        metadata["shorts"].append(entry["short"])
        metadata["names"].append(entry["name"])
        metadata["categories"].append(entry["category"])
        metadata["periods"].append(entry["period"])
        metadata["authors"].append(entry["author"])
        metadata["descriptions"].append(entry["description"])
        metadata["sources"].append(entry["source"])

    arrays = {
        name: np.asarray(values, dtype=str)
        for name, values in metadata.items()
    }
    arrays.update({
        "category_order": np.asarray(manifest["category_order"], dtype=str),
        "category_labels_keys": np.asarray(
            list(manifest["category_labels"].keys()), dtype=str),
        "category_labels_values": np.asarray(
            list(manifest["category_labels"].values()), dtype=str),
        "default": np.asarray([manifest["default"]], dtype=str),
        "source_library": np.asarray([manifest["source_library"]], dtype=str),
        "source_wiki": np.asarray([manifest["source_wiki"]], dtype=str),
    })
    arrays.update(grids)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes), {len(entries)} presets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
