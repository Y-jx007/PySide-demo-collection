"""Bundled data for the Circulation app.

Run from the project root:

    python tools/build_circulation_data.py

Downloads (small subsets, no credentials):

* ETOPO1 1-degree relief from NOAA ERDDAP.
* NCEP/NCAR Reanalysis 1 monthly mean 10 m wind, 2015-2024 (120 months), via
  the NOAA PSL NetCDF Subset Service.

The outputs are assets/data/worldelev.npz and assets/data/wind_real.npz.
"""

from __future__ import annotations

import io
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import netcdf_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "data"
UA = {"User-Agent": "PySide-demo-collection/0.2"}

ETOPO_URL = (
    "https://cwcgom.aoml.noaa.gov/erddap/griddap/etopo180.csv"
    "?altitude%5B(-90):60:(90)%5D%5B(-180):60:(180)%5D"
)
NCSS = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep.reanalysis/Monthlies/surface_gauss/{}.10m.mon.mean.nc"


def fetch(url: str, params=None) -> bytes:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def build_elevation() -> None:
    print("ETOPO1 ...")
    text = fetch(ETOPO_URL).decode("ascii", "replace")
    rows = np.genfromtxt(io.StringIO(text), delimiter=",", skip_header=2)
    lat = np.unique(rows[:, 0])
    lon = np.unique(rows[:, 1])
    elev = rows[:, 2].reshape(lat.size, lon.size)
    elev = np.where(elev == 32767, 0, elev).astype(np.int16)
    path = OUT / "worldelev.npz"
    np.savez_compressed(path, elev=elev, lat=lat.astype(np.float32),
                        lon=lon.astype(np.float32))
    print("  saved", path, elev.shape, path.stat().st_size)


def _load_wind(name: str):
    params = {
        "var": name,
        "time_start": "2015-01-01T00:00:00Z",
        "time_end": "2024-12-31T23:59:59Z",
        "accept": "netcdf",
    }
    raw = fetch(NCSS.format(name), params)
    tmp = OUT / f"_{name}_2015_2024.nc"
    tmp.write_bytes(raw)
    try:
        nc = netcdf_file(str(tmp), "r", mmap=False)
        var = [k for k in nc.variables if k not in ("time", "lat", "lon")][0]
        data = nc.variables[var][:].astype(np.float64)
        lat = nc.variables["lat"][:].astype(np.float64)
        lon = nc.variables["lon"][:].astype(np.float64)
        time = nc.variables["time"][:].astype(np.float64)
        nc.close()
    finally:
        tmp.unlink(missing_ok=True)
    dates = [datetime(1800, 1, 1) + timedelta(hours=float(h)) for h in time]
    selected = [i for i, d in enumerate(dates) if 2015 <= d.year <= 2024]
    assert len(selected) == 120, "expected 120 months (2015-2024)"
    return data[selected], lat, lon


def build_wind() -> None:
    print("NCEP 10 m wind ...")
    u, lat, lon = _load_wind("uwnd")
    v, _, _ = _load_wind("vwnd")
    years = np.arange(2015, 2025)
    u = u[:, ::-1, :]                      # latitude ascending
    v = v[:, ::-1, :]
    lat = lat[::-1]

    lon_ext = np.concatenate([lon, [360.0]])
    u_ext = np.concatenate([u, u[:, :, :1]], axis=2)
    v_ext = np.concatenate([v, v[:, :, :1]], axis=2)
    tlat = np.arange(-87.5, 87.51, 2.5)
    tlon = np.arange(-178.75, 178.76, 2.5)
    pts = np.stack(np.meshgrid(tlat, tlon % 360.0, indexing="ij"), axis=-1)

    uo = np.empty((120, tlat.size, tlon.size), np.float32)
    vo = np.empty_like(uo)
    for k in range(120):
        uo[k] = RegularGridInterpolator((lat, lon_ext), u_ext[k],
                                        bounds_error=False, fill_value=None)(pts)
        vo[k] = RegularGridInterpolator((lat, lon_ext), v_ext[k],
                                        bounds_error=False, fill_value=None)(pts)

    by_month = uo.reshape(10, 12, tlat.size, tlon.size)
    by_month_v = vo.reshape(10, 12, tlat.size, tlon.size)
    clim_u = by_month.mean(axis=0)
    clim_v = by_month_v.mean(axis=0)
    annual_u = by_month.mean(axis=1)
    annual_v = by_month_v.mean(axis=1)
    years = np.arange(2015, 2025)
    x = years - years.mean()
    den = (x * x).sum()
    trend_u = ((annual_u - annual_u.mean(axis=0)) * x[:, None, None]).sum(axis=0) / den * 10.0
    trend_v = ((annual_v - annual_v.mean(axis=0)) * x[:, None, None]).sum(axis=0) / den * 10.0

    path = OUT / "wind_real.npz"
    np.savez_compressed(
        path,
        years=np.repeat(years, 12).astype(np.int16),
        months=np.tile(np.arange(1, 13), 10).astype(np.int8),
        year_values=years.astype(np.int16),
        month_values=np.arange(1, 13).astype(np.int8),
        lat=tlat.astype(np.float32), lon=tlon.astype(np.float32),
        u=uo.reshape(120, tlat.size, tlon.size).astype(np.float16),
        v=vo.reshape(120, tlat.size, tlon.size).astype(np.float16),
        clim_u=clim_u.astype(np.float32), clim_v=clim_v.astype(np.float32),
        annual_u=annual_u.astype(np.float32), annual_v=annual_v.astype(np.float32),
        trend_u=trend_u.astype(np.float32), trend_v=trend_v.astype(np.float32),
        source=np.array("NCEP/NCAR Reanalysis 1 monthly mean 10m wind, 2015-2024"),
    )
    print("  saved", path, uo.shape, path.stat().st_size)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_elevation()
    build_wind()
    print("done")
