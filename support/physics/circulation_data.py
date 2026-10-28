"""Static data helpers for the Circulation app.

Bundled files live in assets/data/:

* worldmask.npz  720x360 boolean land mask (Natural Earth coastlines).
* worldelev.npz  ETOPO1 1-degree global relief, metres, latitude ascending.
* wind_real.npz  NCEP/NCAR Reanalysis 1, monthly mean 10 m wind aggregated
  to annual means for 2015-2024 on a 2.5-degree grid.

See README.md and THIRD_PARTY_NOTICES.md for sources and attribution.
"""

from __future__ import annotations

import numpy as np

from common.paths import DATA_DIR

ELEV_FILE = str(DATA_DIR / "worldelev.npz")
WIND_FILE = str(DATA_DIR / "wind_real.npz")


def load_elevation():
    """Return (elev, lat, lon) with latitude/longitude ascending.

    `elev` is an int16 array of metres with shape (181, 361); row 0 is -90
    degrees latitude.  Ocean depths are negative.
    """
    data = np.load(ELEV_FILE)
    return data["elev"], data["lat"], data["lon"]


def elevation_at(nw, nh):
    """Sample the 1-degree relief onto an (nh, nw) grid.

    Row 0 is south, matching WindSim's convention.  Missing data falls back to
    zeros so the app still runs if the optional relief file is absent.
    """
    try:
        elev, _, _ = load_elevation()
    except Exception:
        return np.zeros((nh, nw), np.float32)
    ys = np.linspace(0, elev.shape[0] - 1, nh).round().astype(np.int32)
    xs = np.linspace(0, elev.shape[1] - 1, nw).round().astype(np.int32)
    return elev[np.ix_(ys, xs)].astype(np.float32)


def terrain_adjust(u, v, elev, strength=1.0):
    """Apply a simple terrain downscaling to a gridded wind field.

    The correction is deliberately conservative and physically motivated:

    * high ground increases surface drag (u, v are reduced);
    * an upslope wind component is blocked in proportion to the local slope;
    * the blocked component is deflected along the terrain contour, so air
      tends to flow around hills instead of through them.

    Parameters are on the same (nh, nw) grid, row 0 south.  `strength` is a
    user-facing multiplier (1.0 = the default amount).
    """
    if strength <= 0.0 or elev is None:
        return u, v

    h = np.maximum(elev.astype(np.float32), 0.0)
    gy, gx = np.gradient(h)
    slope = np.hypot(gx, gy)
    hn = np.clip(h / 3000.0, 0.0, 1.0)

    mag = np.hypot(u, v) + 1e-6
    # -1 .. 1: is the wind blowing uphill?  Positive means blocked.
    upslope = (u * gx + v * gy) / (mag * (slope + 1e-3))
    block = np.clip(upslope, 0.0, 1.0) * np.clip(slope / 180.0, 0.0, 1.0)
    land = (h > 0.0).astype(np.float32)

    drag = strength * (0.06 * land + 0.28 * hn + 0.38 * block)
    scale = 1.0 - np.clip(drag, 0.0, 0.72)

    # Tangent to the terrain contour; the blocked component is rotated onto it.
    cx, cy = -gy, gx
    cn = np.hypot(cx, cy) + 1e-6
    deflect = strength * 0.55 * block
    u2 = u * scale + (cx / cn) * mag * deflect
    v2 = v * scale + (cy / cn) * mag * deflect
    return u2.astype(np.float32), v2.astype(np.float32)


class RealWind:
    """Monthly mean 10 m wind from NCEP/NCAR Reanalysis 1 (2015-2024).

    `u`/`v` have shape (120, 71, 144) on a 2.5-degree grid, one entry per
    month.  `clim_u`/`clim_v` are the 12 calendar-month climatologies
    (2015-2024); `trend_u`/`trend_v` are least-squares slopes of the annual
    means, in m/s per decade.
    """

    def __init__(self, path=WIND_FILE):
        data = np.load(path)
        self.years = data["years"].astype(np.int32)          # 120 months
        self.months = data["months"].astype(np.int32)
        self.year_values = data["year_values"].astype(np.int32)
        self.month_values = data["month_values"].astype(np.int32)
        self.lat = data["lat"].astype(np.float32)
        self.lon = data["lon"].astype(np.float32)
        self.u = data["u"]                                    # float16, 120
        self.v = data["v"]
        self.clim_u = data["clim_u"].astype(np.float32)       # (12, 71, 144)
        self.clim_v = data["clim_v"].astype(np.float32)
        self.annual_u = data["annual_u"].astype(np.float32)   # (10, 71, 144)
        self.annual_v = data["annual_v"].astype(np.float32)
        self.trend_u = data["trend_u"].astype(np.float32)
        self.trend_v = data["trend_v"].astype(np.float32)
        self.source = str(data["source"]) if "source" in data else "NCEP/NCAR Reanalysis 1"

    def field(self, year_index, month_index, layer="monthly"):
        """Return an (nh, nw, 2) float32 wind field.

        `monthly` is the selected month, `anomaly` is the selected month minus
        the 2015-2024 mean for that calendar month, and `trend` is the annual
        linear trend (same for every month).
        """
        if layer == "trend":
            u, v = self.trend_u, self.trend_v
        else:
            index = int(year_index) * 12 + int(month_index)
            u = self.u[index].astype(np.float32)
            v = self.v[index].astype(np.float32)
            if layer == "anomaly":
                u = u - self.clim_u[int(month_index)]
                v = v - self.clim_v[int(month_index)]
        return np.stack([u, v], axis=-1).astype(np.float32)

    def annual_field(self, year_index):
        """Ten-year annual mean field (used by callers that want a summary)."""
        return np.stack([self.annual_u[year_index], self.annual_v[year_index]],
                        axis=-1).astype(np.float32)

    def direction_delta(self, year_index, month_index):
        """Signed direction change vs the same-month climatology, in degrees.

        Positive = counter-clockwise, negative = clockwise.  Useful as a
        background shading layer for the anomaly view.
        """
        index = int(year_index) * 12 + int(month_index)
        u = self.u[index].astype(np.float32)
        v = self.v[index].astype(np.float32)
        cu = self.clim_u[int(month_index)]
        cv = self.clim_v[int(month_index)]
        dot = u * cu + v * cv
        cross = cu * v - cv * u
        return np.degrees(np.arctan2(cross, dot)).astype(np.float32)
