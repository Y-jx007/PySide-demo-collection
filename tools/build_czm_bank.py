# -*- coding: utf-8 -*-
"""Build a bank of independent Voronoi CZM crack fields.

Each seed produces a different polygon network, so the app can choose a new
pattern per session / per new dry cycle instead of replaying one fixed image.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cohesive_voronoi_desiccation import solve_voronoi, rasterize


def main():
    seeds = [7, 17, 27, 37]
    levels = np.linspace(0.0, 1.0, 9)
    eps_max = 0.0045
    all_frames = []
    for seed in seeds:
        frames = []
        for s in levels:
            if s <= 0:
                frames.append(np.zeros((320, 320), np.uint8))
                continue
            eps = eps_max * (s ** 0.60)
            u, d, coh, nodes = solve_voronoi(eps, m=10, seed=seed)
            frames.append(rasterize(d, coh))
        all_frames.append(np.stack(frames))
        print("seed", seed, "done", all_frames[-1][-1].mean())
    out = Path("apps/Physics/czm_fields.npz")
    np.savez_compressed(out, frames=np.stack(all_frames),
                        levels=levels, seeds=np.array(seeds, np.int32),
                        n=np.int32(10), eps_max=np.float32(eps_max))
    print("saved", out, np.stack(all_frames).shape)


if __name__ == "__main__":
    main()
