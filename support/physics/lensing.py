# -*- coding: utf-8 -*-
"""Schwarzschild null-geodesic ray tracer for the Lensing app.

The geometry is exact Schwarzschild (G = c = 1, M = 1).  A photon orbit obeys

    u'' + u = 3 M u^2,      u = 1 / r,

in the orbital plane.  Instead of integrating that second-order equation in
phi (which is singular for radial rays), the Taichi kernel integrates the
equivalent Cartesian null geodesic

    d^2 x / dlambda^2 = -3 M h^2 x / r^5,     h = |x cross dx/dlambda|,

with classical RK4.  The two forms are algebraically identical; the Cartesian
form remains well behaved for every pixel direction and gives the disk
intersection in 3D directly.  The reference tracer below uses the same ODE,
so the app and the tests solve the same system.

The accretion disk is a geometrically thin, optically thin/thick (adjustable)
Keplerian disk.  Its observed shift factor is

    g = sqrt(1 - 2M/r_cam) / [ u^t (1 - Omega b_z) ],
    u^t = 1 / sqrt(1 - 3M/r_emit),  Omega = sqrt(M / r_emit^3),

and the emitted bolometric intensity is boosted by g^4.  Disk colour follows a
Planck blackbody at the observed temperature g T_emit.
"""
import math

import numpy as np

M = 1.0
R_S = 2.0 * M
R_PHOTON = 3.0 * M
R_ISCO = 6.0 * M
B_CRIT = 3.0 * math.sqrt(3.0) * M
TAU = 2.0 * math.pi

BB_N = 512
BB_TMIN = 800.0
BB_TMAX = 40000.0

_DEFAULT_STAR_SIZE = (2048, 1024)
_STAR_CACHE = {}
_LUT_CACHE = {}
_DISK_CACHE = {}


# --------------------------------------------------------------------------- #
# Basic geometry and exact geodesics
# --------------------------------------------------------------------------- #
def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= 1e-15:
        return np.zeros(3, dtype=np.float64)
    return v / n


def critical_impact_parameter():
    """Critical impact parameter of the Schwarzschild photon sphere."""
    return float(B_CRIT)


def shadow_angular_radius(distance, M=1.0):
    """Angular radius of the shadow for a static observer at a distance."""
    r = float(distance)
    if r <= 2.0 * M:
        return math.pi / 2.0
    f = 1.0 - 2.0 * M / r
    arg = min(1.0, B_CRIT * math.sqrt(f) / r)
    return math.asin(arg)


class CameraFrame:
    """Local static-observer camera at a finite Schwarzschild radius."""

    __slots__ = ("origin", "forward", "right", "up", "tan_half",
                 "distance", "inclination", "azimuth", "fov",
                 "view_yaw", "view_pitch")

    def __init__(self, distance=36.0, inclination=78.0, azimuth=0.0,
                 fov=42.0, view_yaw=0.0, view_pitch=0.0):
        self.distance = max(2.01, float(distance))
        self.inclination = float(np.clip(inclination, 0.0, 89.0))
        self.azimuth = float(azimuth)
        self.fov = float(np.clip(fov, 2.0, 120.0))
        self.view_yaw = float(view_yaw)
        self.view_pitch = float(view_pitch)
        inc = math.radians(self.inclination)
        azi = math.radians(self.azimuth)
        self.origin = self.distance * np.array([
            math.sin(inc) * math.cos(azi),
            math.sin(inc) * math.sin(azi),
            math.cos(inc),
        ], dtype=np.float64)
        self.forward = _unit(-self.origin)
        z = np.array([0.0, 0.0, 1.0])
        up_hint = z - float(np.dot(z, self.forward)) * self.forward
        if float(np.linalg.norm(up_hint)) < 1e-8:
            up_hint = np.array([0.0, 1.0, 0.0])
        base_up = _unit(up_hint)
        base_right = _unit(np.cross(self.forward, base_up))
        base_up = _unit(np.cross(base_right, self.forward))

        # The camera position stays fixed while the observer looks around.
        # Positive yaw looks left (so a fixed star moves right); positive
        # pitch looks up (so a fixed star moves down).  This is direct
        # manipulation: the scene follows the drag.
        yaw = math.radians(self.view_yaw)
        cy, sy = math.cos(yaw), math.sin(yaw)
        forward = cy * self.forward - sy * base_right
        right = cy * base_right + sy * self.forward
        up = base_up

        pitch = math.radians(self.view_pitch)
        cp, sp = math.cos(pitch), math.sin(pitch)
        self.forward = _unit(cp * forward + sp * up)
        self.up = _unit(cp * up - sp * forward)
        self.right = _unit(right)
        self.tan_half = math.tan(0.5 * math.radians(self.fov))

    def ray(self, x, y):
        """Direction for normalized image coordinates x,y in [-1, 1].

        x is right, y is up.  The returned vector is a local unit vector in
        the static observer's orthonormal frame.
        """
        d = (self.forward
             + float(x) * self.tan_half * self.right
             + float(y) * self.tan_half * self.up)
        return _unit(d)

    def basis(self):
        return (self.origin.copy(), self.forward.copy(), self.right.copy(),
                self.up.copy())


def initial_velocity(origin, direction, M=1.0):
    """Coordinate velocity for a backward-traced null ray (E = 1).

    direction is a local unit vector of the static camera.  The radial
    component is unchanged; the tangential component gains 1/sqrt(f), which
    makes |r cross v| equal to the impact parameter b = L/E.
    """
    p = np.asarray(origin, dtype=np.float64)
    r = float(np.linalg.norm(p))
    f = 1.0 - 2.0 * M / r
    if f <= 0.0:
        raise ValueError("observer is inside the horizon")
    er = p / r
    n = _unit(direction)
    nr = float(np.dot(n, er))
    nt = n - nr * er
    v = nr * er + nt / math.sqrt(f)
    return v


def binet_curvature(u, M=1.0):
    """Right-hand side of the photon-orbit Binet equation.

    u'' = 3 M u^2 - u, so u'' + u = 3 M u^2.  The Taichi kernel integrates
    the algebraically equivalent Cartesian null geodesic below.
    """
    u = float(u)
    return 3.0 * M * u * u - u


def _accel(r, h2, M=1.0):
    r2 = float(np.dot(r, r))
    return -3.0 * M * h2 * r / (r2 * r2 * math.sqrt(r2))


def _rk4_step(r, v, h2, step, M=1.0):
    k1r = v
    k1v = _accel(r, h2, M)
    k2r = v + 0.5 * step * k1v
    k2v = _accel(r + 0.5 * step * k1r, h2, M)
    k3r = v + 0.5 * step * k2v
    k3v = _accel(r + 0.5 * step * k2r, h2, M)
    k4r = v + step * k3v
    k4v = _accel(r + step * k3r, h2, M)
    rn = r + (step / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
    vn = v + (step / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return rn, vn


def trace_ray(origin, direction, max_steps=2000, step_scale=0.06,
              r_escape=None, M=1.0):
    """Trace one null ray and return a small diagnostic dictionary.

    This is the NumPy/Python reference used by the tests; the app uses the
    identical ODE in the Taichi kernel.
    """
    p = np.asarray(origin, dtype=np.float64)
    r0 = float(np.linalg.norm(p))
    v = initial_velocity(p, direction, M)
    hp = np.cross(p, v)
    h2 = float(np.dot(hp, hp))
    b = math.sqrt(max(0.0, h2))
    if r_escape is None:
        r_escape = max(600.0, 18.0 * r0)
    r = p.copy()
    status = "max"
    steps = 0
    for steps in range(1, int(max_steps) + 1):
        rn = float(np.linalg.norm(r))
        if rn <= 2.002 * M:
            status = "captured"
            break
        if rn >= r_escape:
            status = "escaped"
            break
        step = float(step_scale) * max(rn, 0.5 * M)
        r, v = _rk4_step(r, v, h2, step, M)
    speed = float(np.linalg.norm(v))
    return {
        "status": status,
        "steps": steps,
        "position": r.copy(),
        "velocity": v.copy(),
        "direction": v / speed if speed > 0.0 else v,
        "radius": float(np.linalg.norm(r)),
        "impact": b,
        "h2": h2,
    }


def deflection_angle(b, r0=5000.0, **kwargs):
    """Total weak/strong deflection for an incoming ray with impact b."""
    f = 1.0 - 2.0 * M / r0
    sinpsi = min(1.0, float(b) * math.sqrt(f) / float(r0))
    cospsi = math.sqrt(max(0.0, 1.0 - sinpsi * sinpsi))
    origin = np.array([r0, 0.0, 0.0])
    er = np.array([1.0, 0.0, 0.0])
    n = -cospsi * er + sinpsi * np.array([0.0, 1.0, 0.0])
    record = trace_ray(origin, n, **kwargs)
    if record["status"] == "captured":
        return math.inf
    out = record["direction"]
    value = float(np.clip(np.dot(n, out), -1.0, 1.0))
    return math.acos(value)


# --------------------------------------------------------------------------- #
# Thin-disk physics
# --------------------------------------------------------------------------- #
def disk_profile_peak_radius(r_in):
    """Radius of the zero-torque thin-disk temperature maximum."""
    return (49.0 / 36.0) * float(r_in)


def _profile_raw(r, r_in):
    r = np.asarray(r, dtype=np.float64)
    x = np.clip(float(r_in) / np.maximum(r, 1e-12), 0.0, 1.0)
    value = np.power(x, 0.75) * np.power(np.maximum(0.0, 1.0 - np.sqrt(x)), 0.25)
    return value


_PROFILE_PEAK = {}

def _profile_peak(r_in):
    key = round(float(r_in), 8)
    value = _PROFILE_PEAK.get(key)
    if value is None:
        value = float(_profile_raw(disk_profile_peak_radius(r_in), r_in))
        _PROFILE_PEAK[key] = value
    return max(value, 1e-12)


def disk_temperature(r, r_in, peak_temperature=9000.0):
    """Zero-torque thin-disk temperature profile, normalized at its peak."""
    r = np.asarray(r, dtype=np.float64)
    raw = _profile_raw(r, r_in) / _profile_peak(r_in)
    return np.where(
        (r > float(r_in)) & (r < 1e9),
        raw * float(peak_temperature),
        0.0,
    )


def profile_value(r, r_in):
    """Normalized T/T_peak at radius r (NumPy scalar/array)."""
    return _profile_raw(r, r_in) / _profile_peak(r_in)


def doppler_factor(r_emit, b_z, distance, M=1.0):
    """Observed/emitted frequency ratio for a Keplerian thin disk.

    b_z is the photon's signed axial impact parameter seen by the camera.
    The convention used by the app is b_z = -[origin x v]_z for the
    backward-traced ray, i.e. the physical photon's value.
    """
    r = float(r_emit)
    if r <= 3.0 * M:
        return 0.0
    f_cam = 1.0 - 2.0 * M / float(distance)
    if f_cam <= 0.0:
        return 0.0
    omega = math.sqrt(M / (r * r * r))
    ut = 1.0 / math.sqrt(max(1e-12, 1.0 - 3.0 * M / r))
    denom = ut * (1.0 - omega * float(b_z))
    if denom <= 1e-9:
        return 0.0
    return math.sqrt(f_cam) / denom


# --------------------------------------------------------------------------- #
# Planck blackbody colours
# --------------------------------------------------------------------------- #
def _gauss_piecewise(wave, mu, sigma_left, sigma_right):
    sigma = np.where(wave < mu, sigma_left, sigma_right)
    return np.exp(-0.5 * ((wave - mu) / sigma) ** 2)


def _cie_xyz(wave):
    """Wyman, Sloan & Shirley (2013) analytic CIE 1931 fits."""
    x = (1.056 * _gauss_piecewise(wave, 599.8, 37.9, 31.0)
         + 0.362 * _gauss_piecewise(wave, 442.0, 16.0, 26.7)
         - 0.065 * _gauss_piecewise(wave, 501.1, 20.4, 26.2))
    y = (0.821 * _gauss_piecewise(wave, 568.8, 46.9, 40.5)
         + 0.286 * _gauss_piecewise(wave, 530.9, 16.3, 31.1))
    z = (1.217 * _gauss_piecewise(wave, 437.0, 11.8, 36.0)
         + 0.681 * _gauss_piecewise(wave, 459.0, 26.0, 13.8))
    return x, y, z


_WAVE = np.arange(380.0, 781.0, 2.0)
_CIE_X, _CIE_Y, _CIE_Z = _cie_xyz(_WAVE)


def blackbody_rgb(temperature):
    """Linear sRGB chromaticity of a Planck spectrum, normalised to max=1."""
    t = float(np.clip(temperature, 400.0, 200000.0))
    wave_m = _WAVE * 1e-9
    h = 6.62607015e-34
    c = 2.99792458e8
    kb = 1.380649e-23
    radiance = 1.0 / (wave_m ** 5) / np.expm1(h * c / (wave_m * kb * t))
    X = float(np.sum(radiance * _CIE_X))
    Y = float(np.sum(radiance * _CIE_Y))
    Z = float(np.sum(radiance * _CIE_Z))
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    rgb = np.maximum(np.array([r, g, b], dtype=np.float64), 0.0)
    peak = float(rgb.max())
    if peak <= 1e-15:
        return np.ones(3, dtype=np.float64)
    return rgb / peak


def blackbody_lut(n=BB_N, t_min=BB_TMIN, t_max=BB_TMAX):
    key = (int(n), float(t_min), float(t_max))
    lut = _LUT_CACHE.get(key)
    if lut is not None:
        return lut
    temps = np.geomspace(float(t_min), float(t_max), int(n))
    lut = np.stack([blackbody_rgb(t) for t in temps]).astype(np.float32)
    _LUT_CACHE[key] = lut
    return lut


# --------------------------------------------------------------------------- #
# Procedural background sky
# --------------------------------------------------------------------------- #
def _smoothstep(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_starfield(width=2048, height=1024, seed=7):
    """Return a linear-HDR equirectangular star field texture.

    The texture is deterministic for a given seed and contains a faint
    Milky-Way band, dust lanes and about 24k stars with Planck colours.
    """
    width, height = int(width), int(height)
    key = (width, height, int(seed))
    cached = _STAR_CACHE.get(key)
    if cached is not None:
        return cached

    rng = np.random.default_rng(int(seed))
    vv, uu = np.mgrid[0:height, 0:width]
    lon = (uu + 0.5) / width * TAU - math.pi
    lat = math.pi / 2.0 - (vv + 0.5) / height * math.pi
    clat = np.cos(lat)
    dx = clat * np.cos(lon)
    dy = clat * np.sin(lon)
    dz = np.sin(lat)
    dirs = np.stack([dx, dy, dz], axis=-1)

    mw_n = _unit([0.36, 0.58, 0.73])
    band = np.exp(-0.5 * (np.sum(dirs * mw_n, axis=-1) / 0.20) ** 2)
    waves = []
    for _ in range(7):
        k = _unit(rng.normal(size=3))
        waves.append((
            float(rng.uniform(0.25, 1.0)),
            float(rng.uniform(1.2, 5.0)),
            k,
            float(rng.uniform(0.0, TAU)),
        ))
    cloud = np.zeros((height, width), dtype=np.float64)
    for amp, freq, k, phase in waves:
        cloud += amp * np.sin(freq * np.sum(dirs * k, axis=-1) + phase)
    cloud = 0.5 + 0.5 * np.tanh(cloud * 0.85)

    bulge_dir = _unit([-0.50, 0.62, 0.16])
    bulge = np.exp(-0.5 * (np.sum(dirs * bulge_dir, axis=-1) / 0.24) ** 2)
    dust = np.clip(0.5 + 0.5 * np.sin(3.1 * np.sum(dirs * _unit([0.2, -0.7, 0.5]), axis=-1)
                                     + 2.3 * cloud), 0.0, 1.0)
    dust = _smoothstep(0.55, 0.95, dust) * band

    cold = np.array([0.075, 0.105, 0.175])
    warm = np.array([0.145, 0.085, 0.040])
    core = np.array([0.26, 0.18, 0.09])
    base = (band[..., None] * cold * (0.65 + 0.55 * cloud[..., None])
            + bulge[..., None] * core * 0.50
            + warm * (band * bulge)[..., None] * 1.1)
    base *= (1.0 - 0.52 * dust)[..., None]
    base *= 0.070

    # A very small ambient background keeps empty sky from being pure #000.
    image = np.broadcast_to(np.array([0.0018, 0.0022, 0.0032]),
                            (height, width, 3)).copy()
    image += base

    lut = blackbody_lut()
    count = 11000
    # A third of the stars follow the Galactic band, the rest are isotropic.
    n_band = count * 2 // 3
    z = rng.uniform(-1.0, 1.0, size=count)
    phi = rng.uniform(0.0, TAU, size=count)
    root = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    sx = root * np.cos(phi)
    sy = root * np.sin(phi)
    sz = z
    helper = _unit([0.0, 0.0, 1.0]) if abs(mw_n[2]) < 0.9 else _unit([1.0, 0.0, 0.0])
    e1 = _unit(np.cross(mw_n, helper))
    e2 = np.cross(mw_n, e1)
    alpha = rng.uniform(0.0, TAU, size=n_band)
    offset = rng.normal(0.0, 0.16, size=n_band)
    band_dirs = (np.cos(alpha)[:, None] * e1
                 + np.sin(alpha)[:, None] * e2
                 + offset[:, None] * mw_n)
    band_dirs /= np.linalg.norm(band_dirs, axis=1, keepdims=True)
    sx[:n_band] = band_dirs[:, 0]
    sy[:n_band] = band_dirs[:, 1]
    sz[:n_band] = band_dirs[:, 2]

    lon_s = np.arctan2(sy, sx)
    lat_s = np.arcsin(np.clip(sz, -1.0, 1.0))
    px = (lon_s / TAU + 0.5) * width - 0.5
    py = (0.5 - lat_s / math.pi) * height - 0.5

    brightness = 0.008 + 1.45 * rng.random(count) ** 8.0
    temperature = np.exp(rng.normal(math.log(6500.0), 0.55, size=count))
    temperature = np.clip(temperature, 2200.0, 24000.0)
    lti = np.clip((np.log(temperature / BB_TMIN) /
                   math.log(BB_TMAX / BB_TMIN)) * (BB_N - 1), 0, BB_N - 1)
    i0 = np.floor(lti).astype(np.int32)
    i1 = np.minimum(BB_N - 1, i0 + 1)
    f = (lti - i0)[:, None]
    colours = lut[i0] * (1.0 - f) + lut[i1] * f
    sigma = 0.42 + 0.38 * rng.random(count)
    for idx in range(count):
        cx, cy = px[idx], py[idx]
        ix, iy = int(math.floor(cx)), int(math.floor(cy))
        amp = float(brightness[idx])
        sig = float(sigma[idx])
        col = colours[idx]
        for oy in range(-2, 3):
            yy = iy + oy
            if yy < 0 or yy >= height:
                continue
            wy = math.exp(-0.5 * (yy - cy) ** 2 / (sig * sig))
            for ox in range(-2, 3):
                xx = (ix + ox) % width
                wx = math.exp(-0.5 * (ix + ox - cx) ** 2 / (sig * sig))
                w = wx * wy
                if w > 1e-4:
                    image[yy, xx] += col * (amp * w)

    image = np.clip(image, 0.0, 50.0).astype(np.float32)
    _STAR_CACHE[key] = image
    return image


# --------------------------------------------------------------------------- #
# Disk turbulence
# --------------------------------------------------------------------------- #
def _value_noise(width, height, grid_u, grid_v, rng):
    """Periodic bilinear value noise sampled on a (v, u) grid."""
    base = rng.random((int(grid_v), int(grid_u)))
    u = (np.arange(width) + 0.5) / width * grid_u
    v = (np.arange(height) + 0.5) / height * grid_v
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    fx = (u - x0)[None, :]
    fy = (v - y0)[:, None]
    x1 = (x0 + 1) % grid_u
    y1 = (y0 + 1) % grid_v
    a = base[np.ix_(y0, x0)]
    b = base[np.ix_(y0, x1)]
    c = base[np.ix_(y1, x0)]
    d = base[np.ix_(y1, x1)]
    return ((a * (1.0 - fx) + b * fx) * (1.0 - fy)
            + (c * (1.0 - fx) + d * fx) * fy)


def build_disk_texture(width=512, height=256, seed=17):
    """Multi-octave, azimuthally stretched turbulence for the thin disk."""
    width, height = int(width), int(height)
    key = (width, height, int(seed))
    cached = _DISK_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(int(seed))
    tex = np.zeros((height, width), dtype=np.float64)
    # Few radial cells, many azimuthal cells: elongated sheared streaks.
    for grid_u, grid_v, amp in ((6, 24, 0.38), (12, 48, 0.26),
                                (24, 96, 0.18), (48, 192, 0.12),
                                (80, 300, 0.06)):
        tex += amp * _value_noise(width, height, grid_u, grid_v, rng)
    # Warp the coordinate system a little to break the grid alignment.
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    warp = 0.10 * np.sin(TAU * (1.5 * u[None, :] + 2.0 * v[:, None]))
    tex = np.clip(tex / 1.04 + warp, 0.0, 1.0)
    tex = _smoothstep(0.14, 0.90, tex)
    tex = 0.22 + 0.78 * tex
    tex = tex.astype(np.float32)
    _DISK_CACHE[key] = tex
    return tex


# --------------------------------------------------------------------------- #
# Taichi renderer
# --------------------------------------------------------------------------- #
try:
    from common.taichi_base import ensure_taichi, ti  # noqa: F401
    _TI_IMPORTED = True
except Exception:  # pragma: no cover - optional at import time
    ensure_taichi = None
    ti = None
    _TI_IMPORTED = False


_TI_READY = False
_TI_BROKEN = False
_TI_BACKEND = "unavailable"
_TI_ERROR = ""


def taichi_available():
    """Whether the shared Taichi runtime can be initialised in this process."""
    global _TI_READY, _TI_BROKEN, _TI_BACKEND, _TI_ERROR
    if not _TI_IMPORTED:
        return False
    if _TI_BROKEN:
        return False
    if _TI_READY:
        return True
    try:
        ensure_taichi()
        arch = str(getattr(ti.cfg, "arch", ""))
        _TI_BACKEND = "CUDA" if "cuda" in arch.lower() else "CPU"
        _TI_READY = True
        return True
    except Exception as exc:  # pragma: no cover - depends on the machine
        _TI_BROKEN = True
        _TI_ERROR = str(exc)
        return False


def taichi_backend_name():
    return _TI_BACKEND if _TI_READY else "unavailable"


if _TI_IMPORTED:
    @ti.func
    def _cross3(a, b):
        return ti.Vector([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])

    @ti.func
    def _accel3(r, h2):
        # Cartesian form of the Binet equation u'' + u = 3 M u^2.
        r2 = r.dot(r)
        return -3.0 * h2 * r / (r2 * r2 * ti.sqrt(r2))

    @ti.func
    def _smoothstep3(a, b, x):
        t = ti.min(1.0, ti.max(0.0, (x - a) / (b - a)))
        return t * t * (3.0 - 2.0 * t)

    @ti.func
    def _sample_lut(temp, lut: ti.template()):
        t = ti.min(BB_TMAX, ti.max(BB_TMIN, temp))
        pos = (ti.log(t / BB_TMIN) / ti.log(BB_TMAX / BB_TMIN)) * (BB_N - 1)
        i0 = ti.cast(ti.floor(pos), ti.i32)
        i0 = ti.min(BB_N - 2, ti.max(0, i0))
        f = pos - i0
        return lut[i0] * (1.0 - f) + lut[i0 + 1] * f

    @ti.func
    def _sample_disk_tex(tex: ti.template(), tw: ti.i32, th: ti.i32, u, v):
        u = ti.min(1.0, ti.max(0.0, u))
        v = v - ti.floor(v)
        px = u * (tw - 1)
        py = v * th
        ix = ti.cast(ti.floor(px), ti.i32)
        iy = ti.cast(ti.floor(py), ti.i32)
        fx = px - ix
        fy = py - iy
        ix1 = ti.min(tw - 1, ix + 1)
        iy0 = iy % th
        iy1 = (iy + 1) % th
        a = tex[iy0, ix] * (1.0 - fx) + tex[iy0, ix1] * fx
        b = tex[iy1, ix] * (1.0 - fx) + tex[iy1, ix1] * fx
        return a * (1.0 - fy) + b * fy

    @ti.func
    def _sample_star(d, star: ti.template(), sw: ti.i32, sh: ti.i32):
        length = ti.sqrt(d.dot(d))
        n = d / ti.max(length, 1e-12)
        u = ti.atan2(n[1], n[0]) / TAU + 0.5
        u = u - ti.floor(u)
        horiz = ti.sqrt(n[0] * n[0] + n[1] * n[1])
        v = 0.5 - ti.atan2(n[2], horiz) / 3.141592653589793
        px = u * sw - 0.5
        py = ti.min(ti.cast(sh, ti.f32) - 1.001, ti.max(0.0, v * sh - 0.5))
        ix = ti.cast(ti.floor(px), ti.i32)
        iy = ti.cast(ti.floor(py), ti.i32)
        fx = px - ix
        fy = py - iy
        ix0 = ix % sw
        ix1 = (ix + 1) % sw
        iy1 = ti.min(sh - 1, iy + 1)
        a = star[iy, ix0] * (1.0 - fx) + star[iy, ix1] * fx
        b = star[iy1, ix0] * (1.0 - fx) + star[iy1, ix1] * fx
        return a * (1.0 - fy) + b * fy

    @ti.func
    def _aces(x):
        y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
        return ti.min(1.0, ti.max(0.0, y))

    @ti.kernel
    def _lensing_kernel(
        out: ti.template(), star: ti.template(), lut: ti.template(),
        disk_tex: ti.template(), tw: ti.i32, th: ti.i32,
        sw: ti.i32, sh: ti.i32,
        width: ti.i32, height: ti.i32,
        ox: ti.f32, oy: ti.f32, oz: ti.f32,
        fx: ti.f32, fy: ti.f32, fz: ti.f32,
        rx: ti.f32, ry: ti.f32, rz: ti.f32,
        ux: ti.f32, uy: ti.f32, uz: ti.f32,
        tan_half: ti.f32, distance: ti.f32,
        r_in: ti.f32, r_out: ti.f32, t_peak: ti.f32,
        disk_on: ti.i32, opacity: ti.f32, doppler_on: ti.i32,
        exposure: ti.f32, star_gain: ti.f32, disk_gain: ti.f32,
        max_steps: ti.i32, step_scale: ti.f32, r_escape: ti.f32,
        profile_peak: ti.f32, time: ti.f32, jitter: ti.f32,
    ):
        for j, i in ti.ndrange(height, width):
            sx = (2.0 * (i + 0.5) / width - 1.0) * tan_half
            sy = (1.0 - 2.0 * (j + 0.5) / height) * tan_half
            n = ti.Vector([
                fx + sx * rx + sy * ux,
                fy + sx * ry + sy * uy,
                fz + sx * rz + sy * uz,
            ])
            n = n / ti.sqrt(n.dot(n))

            p = ti.Vector([ox, oy, oz])
            er = p / distance
            nr = n.dot(er)
            nt = n - nr * er
            f_obs = 1.0 - 2.0 / distance
            v = nr * er + nt / ti.sqrt(f_obs)
            hv = _cross3(p, v)
            h2 = hv.dot(hv)
            bz = -hv[2]

            r = p
            rad = ti.Vector([0.0, 0.0, 0.0])
            trans = 1.0
            alive = 1
            captured = 0
            if disk_on == 0 and h2 < 27.0:
                # The capture cross-section is exact: b < 3 sqrt(3) M.
                captured = 1
                alive = 0

            for step in range(max_steps):
                if alive == 1:
                    rn = ti.sqrt(r.dot(r))
                    if rn <= 2.002:
                        captured = 1
                        alive = 0
                    elif rn >= r_escape:
                        alive = 0
                    else:
                        hstep = step_scale * ti.max(rn, 0.5)
                        k1r = v
                        k1v = _accel3(r, h2)
                        k2r = v + 0.5 * hstep * k1v
                        k2v = _accel3(r + 0.5 * hstep * k1r, h2)
                        k3r = v + 0.5 * hstep * k2v
                        k3v = _accel3(r + 0.5 * hstep * k2r, h2)
                        k4r = v + hstep * k3v
                        k4v = _accel3(r + hstep * k3r, h2)
                        r_new = r + (hstep / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
                        v_new = v + (hstep / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

                        if disk_on == 1 and r[2] * r_new[2] < 0.0:
                            denom = r[2] - r_new[2]
                            frac = 0.0 if ti.abs(denom) < 1e-9 else r[2] / denom
                            frac = ti.min(1.0, ti.max(0.0, frac))
                            hit = r + (r_new - r) * frac
                            rh = ti.sqrt(hit.dot(hit))
                            if rh >= r_in and rh <= r_out:
                                raw = ti.pow(r_in / rh, 0.75) * ti.pow(
                                    ti.max(0.0, 1.0 - ti.sqrt(r_in / rh)), 0.25
                                ) / profile_peak
                                edge = _smoothstep3(r_in, r_in + 0.55, rh) * (
                                    1.0 - _smoothstep3(r_out - 1.2, r_out, rh)
                                )
                                u_tex = ti.log(rh / r_in) / ti.log(r_out / r_in)
                                phase = (ti.atan2(hit[1], hit[0]) / TAU
                                         - 0.18 * time * ti.pow(r_in / rh, 1.5))
                                grain = _sample_disk_tex(disk_tex, tw, th, u_tex, phase)
                                alpha = ti.min(1.0, ti.max(0.0, opacity * edge * (0.55 + 0.55 * grain)))
                                g = 1.0
                                if doppler_on == 1:
                                    omega = ti.sqrt(1.0 / (rh * rh * rh))
                                    ut = 1.0 / ti.sqrt(ti.max(1e-6, 1.0 - 3.0 / rh))
                                    den = ut * (1.0 - omega * bz)
                                    if den > 1e-6:
                                        g = ti.sqrt(f_obs) / den
                                temp = g * t_peak * raw
                                col = _sample_lut(temp, lut)
                                intensity = disk_gain * ti.pow(g, 4.0) * ti.pow(raw, 4.0)
                                intensity *= (0.55 + 1.05 * grain)
                                rad = rad * (1.0 - alpha) + col * (intensity * alpha)
                                trans *= (1.0 - alpha)
                                if trans < 0.002:
                                    alive = 0
                        r = r_new
                        v = v_new

            if alive == 1:
                # Ran out of steps while still winding near the photon sphere.
                # It cannot escape for b < b_crit; treating unresolved windings
                # as captured keeps the shadow edge clean.
                captured = 1
            if captured == 0 and trans > 0.002:
                length = ti.sqrt(v.dot(v))
                sv = v / ti.max(length, 1e-12)
                rad += _sample_star(sv, star, sw, sh) * (star_gain * trans)

            c = rad * exposure
            c = ti.Vector([_aces(c[0]), _aces(c[1]), _aces(c[2])])
            c = ti.Vector([
                ti.pow(c[0], 1.0 / 2.2),
                ti.pow(c[1], 1.0 / 2.2),
                ti.pow(c[2], 1.0 / 2.2),
            ])
            h = ti.sin(i * 12.9898 + j * 78.233) * 43758.5453
            d = (h - ti.floor(h) - 0.5) * jitter
            out[j, i] = ti.Vector([
                ti.min(1.0, ti.max(0.0, c[0] + d)),
                ti.min(1.0, ti.max(0.0, c[1] + d)),
                ti.min(1.0, ti.max(0.0, c[2] + d)),
            ])
else:  # pragma: no cover
    _lensing_kernel = None


class LensingRenderer:
    """GPU Schwarzschild ray tracer with a lazy Taichi initialisation."""

    def __init__(self, star_size=_DEFAULT_STAR_SIZE, seed=7, output_size=None):
        self.star_size = (int(star_size[0]), int(star_size[1]))
        self.seed = int(seed)
        self._out_fields = {}
        # A fixed output field means Taichi compiles one kernel variant for
        # the lifespan of the process.  Without it, switching between a
        # 600 px preview and an 840 px still costs a ~0.8 s recompile every
        # time the field shape changes on this Windows/CUDA build.
        self._fixed_output_size = int(output_size) if output_size else None
        self._fixed_output_field = None
        self._star_field = None
        self._lut_field = None
        self._tex_field = None
        self._star_np = None
        self._lut_np = None
        self._tex_np = None
        self._ready = False

    @property
    def backend(self):
        if not self._ready:
            return "unavailable"
        return taichi_backend_name()

    def prepare(self):
        if self._ready:
            return
        if not taichi_available():
            raise RuntimeError(
                "Taichi 初始化失败：" + (_TI_ERROR or "未知错误")
            )
        self._star_np = build_starfield(*self.star_size, seed=self.seed)
        self._lut_np = blackbody_lut()
        sw, sh = self.star_size
        self._star_field = ti.Vector.field(3, ti.f32, shape=(sh, sw))
        self._star_field.from_numpy(self._star_np)
        self._lut_field = ti.Vector.field(3, ti.f32, shape=BB_N)
        self._lut_field.from_numpy(self._lut_np)
        self._tex_np = build_disk_texture(512, 256, seed=17)
        self._tex_field = ti.field(ti.f32, shape=(256, 512))
        self._tex_field.from_numpy(self._tex_np)
        self._ready = True

    def _ensure_output(self, width, height):
        if self._fixed_output_size:
            size = max(self._fixed_output_size, int(width), int(height))
            if self._fixed_output_field is None:
                self._fixed_output_field = ti.Vector.field(
                    3, ti.f32, shape=(size, size)
                )
            return self._fixed_output_field
        # Fallback for generic users/tests: keep one field per size alive.
        shape = (int(height), int(width))
        out = self._out_fields.get(shape)
        if out is None:
            out = ti.Vector.field(3, ti.f32, shape=shape)
            self._out_fields[shape] = out
        return out

    def render(self, config):
        """Render config and return uint8 RGB (height, width, 3)."""
        self.prepare()
        width = int(config.width)
        height = int(config.height)
        out = self._ensure_output(width, height)
        frame = CameraFrame(
            config.distance, config.inclination, config.azimuth, config.fov,
            getattr(config, "view_yaw", 0.0),
            getattr(config, "view_pitch", 0.0),
        )
        origin, forward, right, up = frame.basis()
        r_escape = float(config.r_escape)
        if r_escape <= 0.0:
            r_escape = max(600.0, 18.0 * frame.distance)
        r_in = float(config.r_in)
        r_out = max(r_in + 1.0, float(config.r_out))
        _lensing_kernel(
            out, self._star_field, self._lut_field,
            self._tex_field, 512, 256,
            self.star_size[0], self.star_size[1],
            width, height,
            float(origin[0]), float(origin[1]), float(origin[2]),
            float(forward[0]), float(forward[1]), float(forward[2]),
            float(right[0]), float(right[1]), float(right[2]),
            float(up[0]), float(up[1]), float(up[2]),
            float(frame.tan_half), float(frame.distance),
            r_in, r_out, float(config.peak_temperature),
            1 if config.disk_on else 0,
            float(config.opacity),
            1 if config.doppler else 0,
            float(config.exposure), float(config.star_gain),
            float(config.disk_gain),
            int(config.max_steps), float(config.step_scale), r_escape,
            float(_profile_peak(r_in)), float(config.time), 0.75 / 255.0,
        )
        ti.sync()
        rgb = out.to_numpy()
        if rgb.shape[0] != height or rgb.shape[1] != width:
            rgb = rgb[:height, :width]
        return np.clip(rgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)

    def warm_up(self, config):
        """Compile the kernel and prime the star texture without a frame."""
        self.prepare()
        self._ensure_output(int(config.width), int(config.height))


class RenderConfig:
    """Plain render parameters shared by the app, cover generator and tests."""

    __slots__ = (
        "width", "height", "distance", "inclination", "azimuth", "fov",
        "disk_on", "r_in", "r_out", "peak_temperature", "opacity",
        "doppler", "exposure", "star_gain", "disk_gain",
        "max_steps", "step_scale", "r_escape", "time",
        "view_yaw", "view_pitch",
    )

    def __init__(self, width=600, height=600, distance=36.0,
                 inclination=78.0, azimuth=0.0, fov=42.0,
                 disk_on=True, r_in=6.0, r_out=22.0,
                 peak_temperature=9000.0, opacity=0.82, doppler=True,
                 exposure=1.0, star_gain=1.0, disk_gain=1.4,
                 max_steps=220, step_scale=0.06, r_escape=0.0, time=0.0,
                 view_yaw=0.0, view_pitch=0.0):
        self.width = int(width)
        self.height = int(height)
        self.distance = float(distance)
        self.inclination = float(inclination)
        self.azimuth = float(azimuth)
        self.fov = float(fov)
        self.disk_on = bool(disk_on)
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.peak_temperature = float(peak_temperature)
        self.opacity = float(opacity)
        self.doppler = bool(doppler)
        self.exposure = float(exposure)
        self.star_gain = float(star_gain)
        self.disk_gain = float(disk_gain)
        self.max_steps = int(max_steps)
        self.step_scale = float(step_scale)
        self.r_escape = float(r_escape)
        self.time = float(time)
        self.view_yaw = float(view_yaw)
        self.view_pitch = float(view_pitch)
