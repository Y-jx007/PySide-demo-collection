# -*- coding: utf-8 -*-
"""Real orbital elements for a compact top-down solar-system view.

Elements are J2000 mean values (semi-major axis in AU, period in years,
longitude of perihelion and mean anomaly in degrees).  Inclinations are
ignored: this is a didactic top-down projection for the KAM app.
"""
from __future__ import annotations

import numpy as np

TAU = 2.0 * np.pi

PLANETS = (
    {"name": "水星", "a": 0.38710, "e": 0.20563, "period": 0.24085,
     "varpi": 77.46, "M0": 174.796, "color": "#b8b2a7", "radius": 2.2},
    {"name": "金星", "a": 0.72333, "e": 0.00677, "period": 0.61520,
     "varpi": 131.60, "M0": 50.115, "color": "#e0b77a", "radius": 3.2},
    {"name": "地球", "a": 1.00000, "e": 0.01671, "period": 1.00000,
     "varpi": 102.94, "M0": 357.517, "color": "#63a9e8", "radius": 3.4},
    {"name": "火星", "a": 1.52368, "e": 0.09340, "period": 1.88085,
     "varpi": 336.06, "M0": 19.373, "color": "#d47755", "radius": 2.7},
    {"name": "木星", "a": 5.20260, "e": 0.04839, "period": 11.862,
     "varpi": 14.75, "M0": 20.020, "color": "#d9b58b", "radius": 6.4},
    {"name": "土星", "a": 9.55491, "e": 0.05386, "period": 29.457,
     "varpi": 92.43, "M0": 317.020, "color": "#e6d09a", "radius": 5.4, "ring": True},
    {"name": "天王星", "a": 19.2184, "e": 0.04726, "period": 84.017,
     "varpi": 170.96, "M0": 142.238, "color": "#9edbd7", "radius": 4.4},
    {"name": "海王星", "a": 30.1104, "e": 0.00859, "period": 164.79,
     "varpi": 44.97, "M0": 259.883, "color": "#6f8de0", "radius": 4.2},
)

ASTEROID_GAPS = (2.50, 2.82, 2.95, 3.27)

# Real observed apsidal (perihelion) precession rates, arcseconds/year.
# Mercury includes the 43"/century general-relativistic contribution.
PRECESSION_ARCSEC_PER_YEAR = {
    "水星": 5.75,
    "金星": 0.06,
    "地球": 0.017,
    "火星": 0.00,
    "木星": 0.00,
    "土星": 0.00,
    "天王星": 0.00,
    "海王星": 0.00,
}


def solve_kepler(mean_anomaly, eccentricity, iterations=6):
    """Solve E - e sin E = M with Newton iterations."""
    M = np.asarray(mean_anomaly, dtype=np.float64)
    e = float(eccentricity)
    E = np.fmod(M + np.pi, TAU) - np.pi
    for _ in range(iterations):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def planet_position(planet, years):
    """Top-down heliocentric position in AU at a given time in years."""
    M = np.radians(planet["M0"] + 360.0 * float(years) / planet["period"])
    E = solve_kepler(M, planet["e"])
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + planet["e"]) * np.sin(E / 2.0),
        np.sqrt(1.0 - planet["e"]) * np.cos(E / 2.0),
    )
    r = planet["a"] * (1.0 - planet["e"] * np.cos(E))
    theta = nu + np.radians(planet["varpi"])
    return float(r * np.cos(theta)), float(r * np.sin(theta))


def planet_orbit_points(planet, samples=260):
    """Closed orbit path in heliocentric AU coordinates."""
    M = np.linspace(0.0, TAU, int(samples), endpoint=False)
    E = solve_kepler(M, planet["e"])
    r = planet["a"] * (1.0 - planet["e"] * np.cos(E))
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + planet["e"]) * np.sin(E / 2.0),
        np.sqrt(1.0 - planet["e"]) * np.cos(E / 2.0),
    )
    theta = nu + np.radians(planet["varpi"])
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def planet_arc(planet, years, span_deg=3000.0, precession_per_orbit=12.0,
               samples=750, precession_deg_per_year=None):
    """Recent orbital arc, oldest point first.

    span_deg is measured in mean anomaly.  A nonzero precession rotates the
    perihelion by that many degrees per orbit, which makes a long arc a
    visible rosette instead of the same closed ellipse drawn repeatedly.
    """
    samples = max(8, int(samples))
    span_deg = max(0.0, min(3600.0, float(span_deg)))
    ago = np.linspace(1.0, 0.0, samples) * (span_deg / 360.0)

    mean_anomaly = np.radians(
        planet["M0"] + 360.0 * float(years) / planet["period"] - 360.0 * ago
    )
    if precession_deg_per_year is not None:
        varpi = (
            planet["varpi"]
            + float(precession_deg_per_year) * float(years)
            - float(precession_deg_per_year) * ago * planet["period"]
        )
    else:
        varpi = (
            planet["varpi"]
            + precession_per_orbit * float(years) / planet["period"]
            - precession_per_orbit * ago
        )
    E = solve_kepler(mean_anomaly, planet["e"])
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + planet["e"]) * np.sin(E / 2.0),
        np.sqrt(1.0 - planet["e"]) * np.cos(E / 2.0),
    )
    r = planet["a"] * (1.0 - planet["e"] * np.cos(E))
    theta = nu + np.radians(varpi)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def planet_orbit_at(planet, years, precession_per_orbit=12.0, samples=260,
                    precession_deg_per_year=None):
    """Closed orbit with the current precessed perihelion."""
    M = np.linspace(0.0, TAU, int(samples), endpoint=False)
    E = solve_kepler(M, planet["e"])
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + planet["e"]) * np.sin(E / 2.0),
        np.sqrt(1.0 - planet["e"]) * np.cos(E / 2.0),
    )
    r = planet["a"] * (1.0 - planet["e"] * np.cos(E))
    if precession_deg_per_year is not None:
        varpi = planet["varpi"] + float(precession_deg_per_year) * float(years)
    else:
        varpi = (
            planet["varpi"]
            + precession_per_orbit * float(years) / planet["period"]
        )
    theta = nu + np.radians(varpi)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def planet_position_at(planet, years, precession_per_orbit=12.0,
                       precession_deg_per_year=None):
    """Current position with the precessed perihelion."""
    if precession_deg_per_year is not None:
        varpi = planet["varpi"] + float(precession_deg_per_year) * float(years)
    else:
        varpi = (
            planet["varpi"]
            + precession_per_orbit * float(years) / planet["period"]
        )
    current = dict(planet)
    current["varpi"] = varpi
    return planet_position(current, years)


def asteroid_belt(count=560, seed=7):
    """Asteroid belt sample with Kirkwood gaps removed (circular orbits)."""
    rng = np.random.default_rng(seed)
    a = 2.10 + 1.30 * rng.random(int(count))
    for gap in ASTEROID_GAPS:
        keep = np.abs(a - gap) > 0.035
        a = a[keep]
    n = len(a)
    e = 0.16 * rng.random(n)
    phase = TAU * rng.random(n)
    return {
        "a": a,
        "phase": phase,
        "period": a ** 1.5,
        "e": e,
    }


def asteroid_positions(belt, years):
    """Belt positions in AU for a given time (circular approximation)."""
    M = belt["phase"] + TAU * float(years) / belt["period"]
    r = belt["a"]
    return np.column_stack((r * np.cos(M), r * np.sin(M)))


PLANET_ORBITS = {planet["name"]: planet_orbit_points(planet) for planet in PLANETS}
BELT = asteroid_belt()
