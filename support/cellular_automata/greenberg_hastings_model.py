# -*- coding: utf-8 -*-
"""Greenberg-Hastings excitable-media cellular automaton."""
from __future__ import annotations

import numpy as np


class GreenbergHastings:
    """Synchronous Greenberg-Hastings CA.

    States:
        0                  resting
        1                  excited
        2 .. refractory    refractory (R is the last refractory state)
    """

    def __init__(self, size=512, refractory=6, threshold=1, neighbors=8, seed=None):
        self.size = int(size)
        self.refractory = max(2, int(refractory))
        self.threshold = max(1, int(threshold))
        self.neighbors = 8 if int(neighbors) == 8 else 4
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros((self.size, self.size), dtype=np.uint8)
        self.tick = 0
        self.pacemaker = False
        self.pacemaker_interval = self.refractory + 3
        self.pacemaker_radius = max(2, self.size // 80)

    # ------------------------------------------------------------- neighbours
    def _neighbour_count(self, mask):
        count = np.zeros_like(mask, dtype=np.uint8)
        if self.neighbors == 4:
            offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
        else:
            offsets = (
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            )
        for dy, dx in offsets:
            count += np.roll(mask, (dy, dx), axis=(0, 1))
        return count

    # ------------------------------------------------------------------ step
    def step(self):
        if self.pacemaker and self.tick % self.pacemaker_interval == 0:
            center = self.size // 2
            self.stimulate(center, center, self.pacemaker_radius)

        old = self.state
        excited = old == 1
        count = self._neighbour_count(excited)
        new = old.copy()

        resting = old == 0
        new[resting & (count >= self.threshold)] = 1
        new[old == 1] = 2

        middle = (old >= 2) & (old < self.refractory)
        new[middle] = old[middle] + 1
        new[old >= self.refractory] = 0

        self.state = new
        self.tick += 1

    # ---------------------------------------------------------------- seeding
    def stimulate(self, cy, cx, radius=3):
        yy, xx = np.ogrid[:self.size, :self.size]
        mask = (yy - int(cy)) ** 2 + (xx - int(cx)) ** 2 <= int(radius) ** 2
        self.state[mask] = 1

    def clear(self):
        self.state.fill(0)
        self.tick = 0
        self.pacemaker = False

    def reset_random(self, density=0.004):
        self.state.fill(0)
        self.state[self.rng.random((self.size, self.size)) < float(density)] = 1
        self.tick = 0
        self.pacemaker = False

    def reset_spiral(self):
        # A broken planar wave with a refractory tail: the free end curls
        # into a spiral (square-lattice anisotropy makes it rectilinear).
        self.state.fill(0)
        cy = cx = self.size // 2
        self.state[cy - 2:cy + 2, :cx] = 1
        for k in range(1, self.refractory):
            value = min(self.refractory, k + 1)
            self.state[cy - 2 - k:cy - 1 - k, :cx] = value
            self.state[cy + 2 + k:cy + 3 + k, :cx] = value
        self.tick = 0
        self.pacemaker = False

    def reset_target(self):
        self.state.fill(0)
        self.tick = 0
        self.pacemaker = True
        center = self.size // 2
        self.stimulate(center, center, self.pacemaker_radius)

    def reset_empty(self):
        self.state.fill(0)
        self.tick = 0
        self.pacemaker = False

    def reset(self, mode="random", density=0.004):
        if mode == "spiral":
            self.reset_spiral()
        elif mode == "target":
            self.reset_target()
        elif mode == "empty":
            self.reset_empty()
        else:
            self.reset_random(density)

    # ---------------------------------------------------------------- metrics
    def metrics(self):
        excited = int(np.count_nonzero(self.state == 1))
        refractory = int(np.count_nonzero((self.state >= 2) & (self.state <= self.refractory)))
        return {
            "tick": self.tick,
            "excited": excited,
            "refractory": refractory,
            "excited_fraction": excited / float(self.size * self.size),
        }
