"""A two-layer soil water balance for the Erosion app.

This is deliberately a physically based *bucket* model, not a full 3-D
Richards solver.  It keeps the parts that matter for wetting/drying at the
land surface:

* infiltration capacity (Green-Ampt-like, texture dependent);
* surface ponding and runoff;
* percolation from a 10 cm surface layer to a 40 cm root zone;
* deep drainage below field capacity;
* Penman-Monteith potential ET, reduced by soil-water stress;
* shrinkage below the shrinkage limit, used to open the desiccation cracks.

All depths are millimetres of water; theta is volumetric (m3/m3).
"""

from __future__ import annotations

import numpy as np


class SoilWater:
    DEPTH1_MM = 100.0          # 0-10 cm
    DEPTH2_MM = 400.0          # 10-50 cm

    def __init__(self, fines, clay_bias=0.35, init_wetness=0.95):
        self.depth1 = self.DEPTH1_MM
        self.depth2 = self.DEPTH2_MM
        self.clay_bias = float(np.clip(clay_bias, 0.05, 0.75))
        self.update_texture(fines)
        self.theta1 = (self.theta_fc * init_wetness).astype(np.float32)
        self.theta2 = (self.theta_fc * min(1.0, init_wetness + 0.03)).astype(np.float32)
        self.pond = np.zeros_like(self.theta1, dtype=np.float32)
        self.runoff = 0.0
        self.deep_drainage = 0.0
        self.et_loss = 0.0
        self.rain_input = 0.0
        self.balance_error = 0.0

    def update_texture(self, fines):
        """Map the local fine-particle fraction + a user clay bias to soil props."""
        self.fines = np.asarray(fines, dtype=np.float32).copy()
        self.clay = np.clip(self.clay_bias + 0.25 * (self.fines - 0.5),
                            0.04, 0.80)
        # Sand 0.40 / clay 0.52 porosity; loam near 0.45.
        self.theta_s = (0.40 + 0.12 * self.clay).astype(np.float32)
        self.theta_fc = (0.12 + 0.22 * self.clay).astype(np.float32)
        self.theta_wp = (0.03 + 0.12 * self.clay).astype(np.float32)
        # Shrinkage limit: clay-rich soils crack while still fairly wet.
        self.theta_shrink = (self.theta_wp + 0.72 * (self.theta_fc - self.theta_wp)).astype(np.float32)
        # Saturated conductivity (mm/h): sand ~100, clay ~1.
        self.ks = (120.0 * np.power(1.0 - self.clay, 2.5) + 0.6).astype(np.float32)
        # Keep water contents inside the new texture limits.
        if hasattr(self, "theta1"):
            self.theta1 = np.clip(self.theta1, self.theta_wp, self.theta_s).astype(np.float32)
            self.theta2 = np.clip(self.theta2, self.theta_wp, self.theta_s).astype(np.float32)

    # ---------- diagnostics ----------
    @property
    def wetness(self):
        """0 = wilting point, 1 = saturation (surface layer)."""
        return np.clip((self.theta1 - self.theta_wp) /
                       np.maximum(self.theta_s - self.theta_wp, 1e-6), 0.0, 1.0)

    @property
    def root_wetness(self):
        return np.clip((self.theta2 - self.theta_wp) /
                       np.maximum(self.theta_fc - self.theta_wp, 1e-6), 0.0, 1.0)

    @property
    def shrinkage(self):
        """0 = no shrinkage, 1 = at/above the cracking threshold."""
        return np.clip((self.theta_shrink - self.theta1) /
                       np.maximum(self.theta_shrink - self.theta_wp, 1e-6), 0.0, 1.0)

    @property
    def dryness(self):
        return float(1.0 - np.clip(self.root_wetness.mean(), 0.0, 1.0))

    @property
    def storage_mm(self):
        return float(self.theta1.mean() * self.depth1 +
                     self.theta2.mean() * self.depth2 + self.pond.mean())

    # ---------- time step ----------
    def step(self, rain_mm_h, et0_mm_day, dt_min=10.0):
        """Advance the water balance.  Returns the ET actually removed (mm)."""
        dt_h = dt_min / 60.0
        rain = max(0.0, float(rain_mm_h)) * dt_h
        before = self.storage_mm
        runoff0 = self.runoff
        drain0 = self.deep_drainage
        et0 = self.et_loss

        # Infiltration capacity falls as the surface layer fills.
        sat1 = np.clip((self.theta1 - self.theta_wp) /
                       np.maximum(self.theta_s - self.theta_wp, 1e-6), 0.0, 1.0)
        f_cap = self.ks * (0.20 + 0.80 * (1.0 - sat1) ** 2)   # mm/h
        inf = np.minimum(rain + self.pond, f_cap * dt_h)
        self.pond = self.pond + rain - inf

        self.theta1 = self.theta1 + inf / self.depth1

        # Percolation out of the surface layer, then deep drainage.
        excess1 = np.maximum(0.0, self.theta1 - self.theta_fc) * self.depth1
        perc = np.minimum(excess1, self.ks * 0.55 * dt_h)
        self.theta1 -= perc / self.depth1
        self.theta2 += perc / self.depth2

        excess2 = np.maximum(0.0, self.theta2 - self.theta_fc) * self.depth2
        drain = np.minimum(excess2, self.ks * 0.30 * dt_h)
        self.theta2 -= drain / self.depth2
        self.deep_drainage += float(drain.mean())

        # Penman-Monteith potential ET, with FAO-56 style water stress.
        et_pot = max(0.0, float(et0_mm_day)) * dt_min / 1440.0
        ks1 = np.clip((self.theta1 - self.theta_wp) /
                      np.maximum(self.theta_fc - self.theta_wp, 1e-6), 0.0, 1.0)
        ks2 = np.clip((self.theta2 - self.theta_wp) /
                      np.maximum(self.theta_fc - self.theta_wp, 1e-6), 0.0, 1.0)
        et_actual = et_pot * 0.85 * (0.35 * ks1 + 0.65 * ks2)
        take1 = np.minimum(et_actual * 0.35, np.maximum(self.theta1 - self.theta_wp, 0.0) * self.depth1)
        take2 = np.minimum(et_actual * 0.65, np.maximum(self.theta2 - self.theta_wp, 0.0) * self.depth2)
        self.theta1 -= take1 / self.depth1
        self.theta2 -= take2 / self.depth2
        self.et_loss += float((take1 + take2).mean())

        # Surface runoff: ponding above ~12 mm is routed away (not solved).
        over = np.maximum(0.0, self.pond - 12.0)
        self.runoff += float(over.mean())
        self.pond -= over
        self.rain_input += rain

        np.clip(self.theta1, self.theta_wp, self.theta_s, out=self.theta1)
        np.clip(self.theta2, self.theta_wp, self.theta_s, out=self.theta2)
        self.pond = np.clip(self.pond, 0.0, 12.0)

        # Mass balance diagnostic (mm).
        after = self.storage_mm
        self.balance_error = float(after - before - (
            rain - (self.runoff - runoff0) - (self.deep_drainage - drain0)
            - (self.et_loss - et0)))
        return float((take1 + take2).mean())

    def reset(self, fines=None, clay_bias=None):
        if fines is not None:
            self.update_texture(fines)
        if clay_bias is not None:
            self.clay_bias = float(np.clip(clay_bias, 0.05, 0.75))
            self.update_texture(self.fines)
        self.theta1 = (self.theta_fc * 0.95).astype(np.float32)
        self.theta2 = (self.theta_fc * 0.98).astype(np.float32)
        self.pond = np.zeros_like(self.theta1, dtype=np.float32)
        self.runoff = self.deep_drainage = self.et_loss = self.rain_input = 0.0
        self.balance_error = 0.0
