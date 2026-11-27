# -*- coding: utf-8 -*-
"""Kerr-Newman null-geodesic ray tracer (G = c = M = 1).

The GPU renderer integrates the null geodesics in Kerr-Schild Cartesian
coordinates with RK4.  Kerr-Schild is regular on the polar axis, so there is no
1/sin^2(theta) coordinate singularity and no false vertical column at the
image centre.  The separated Boyer-Lindquist equations below are kept as the
CPU reference (trace_kerr) used by the tests and for validation.

For a photon with conserved energy E = -p_t, axial angular momentum
L = p_phi and Carter constant Q, define

    Sigma = r^2 + a^2 cos^2(theta)
    Delta = r^2 - 2r + a^2
    H     = r^2 + a^2 - a L
    R(r)  = H^2 - Delta [(L - aE)^2 + Q]
    Theta = Q + a^2 E^2 cos^2(theta) - L^2 cot^2(theta)

and integrate (with E = 1)

    dr/dlambda = v_r
    dtheta/dlambda = v_theta
    dphi/dlambda = [aH/Delta - a + L/sin^2(theta)] / Sigma
    dv_r/dlambda = R'/(2 Sigma^2) - v_r (dSigma/dlambda)/Sigma
    dv_theta/dlambda = Theta'/(2 Sigma^2) - v_theta (dSigma/dlambda)/Sigma.

The camera is a ZAMO (zero angular momentum observer); its local sky frame
is orthonormal, and dragging in the app orbits this camera while keeping the
hole centred.  The disk is a geometrically thin Keplerian disk in the
equatorial plane; its shift factor is evaluated from the physical photon
direction (-n) so that frame dragging and Doppler beaming are both included.
"""
import math

import numpy as np

from support.physics.lensing import (
    BB_N, BB_TMIN, BB_TMAX,
    LensingRenderer, _aces, _sample_disk_tex, _sample_lut, _sample_star,
    _smoothstep3, blackbody_lut, build_disk_texture, build_starfield,
    taichi_available, taichi_backend_name,
)

try:
    from common.taichi_base import ensure_taichi, ti  # noqa: F401
    _TI_IMPORTED = True
except Exception:  # pragma: no cover
    ensure_taichi = None
    ti = None
    _TI_IMPORTED = False


TAU = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Kerr-Newman geometry
# --------------------------------------------------------------------------- #
# One sub-extremal margin for _pair, KerrConfig and the app's charge
# limit; 0.998 leaves the same safety gap the UI sliders assume.
KN_MAX = 0.998


def _pair(a, q):
    """Clamp (a, Q) to a^2+Q^2 < KN_MAX^2, preserving both signs.

    a < 0 is a black hole spinning in the -phi direction; Q < 0 is a negative
    charge.  The metric only uses Q^2, but the sign is shown in the UI and
    flips the frame dragging / disk direction through a.
    """
    a = max(-KN_MAX, min(KN_MAX, float(a)))
    qmax = math.sqrt(max(0.0, KN_MAX * KN_MAX - a * a))
    q = max(-qmax, min(qmax, float(q)))
    return a, q


def kerr_horizon(a, q=0.0):
    """Outer event horizon r_+ for spin a and charge Q."""
    a, q = _pair(a, q)
    return 1.0 + math.sqrt(max(0.0, 1.0 - a * a - q * q))


def _kerr_isco_zeroq(a):
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0)
    )
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    return 3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def _kn_equatorial(r, a, q):
    """Equatorial metric and r-derivatives for a neutral circular orbit."""
    a2 = a * a
    q2 = q * q
    A = 2.0 * r - q2
    r2 = r * r
    r3 = r2 * r
    gtt = -1.0 + 2.0 / r - q2 / r2
    gtphi = -a * A / r2
    gphiphi = r2 + a2 + a2 * A / r2
    gtt_r = -2.0 / r2 + 2.0 * q2 / r3
    gtphi_r = 2.0 * a / r2 - 2.0 * a * q2 / r3
    gphiphi_r = 2.0 * r - 2.0 * a2 / r2 + 2.0 * a2 * q2 / r3
    return gtt, gtphi, gphiphi, gtt_r, gtphi_r, gphiphi_r


def kerr_disk_omega(r, a, q=0.0, retrograde=False):
    """Circular-orbit angular velocity on the equator.

    The default branch co-rotates with the spin (a < 0 gives Omega < 0).
    ``retrograde=True`` returns the counter-rotating circular orbit, used only
    for the characteristic-orbit diagram, not for the accretion disk.
    """
    a, q = _pair(a, q)
    s_co = 1.0 if a >= 0.0 else -1.0
    s = -s_co if retrograde else s_co
    if abs(q) <= 1e-14:
        if retrograde:
            return -s_co / (r ** 1.5 - abs(a))
        return s_co / (r ** 1.5 + abs(a))
    gtt, gtphi, gphiphi, gtt_r, gtphi_r, gphiphi_r = _kn_equatorial(
        r, a, q)
    if gphiphi_r <= 1e-14:
        return 0.0
    disc = gtphi_r * gtphi_r - gtt_r * gphiphi_r
    if disc < 0.0:
        return 0.0
    return (-gtphi_r + s * math.sqrt(disc)) / gphiphi_r


def kerr_disk_ut(r, a, q=0.0, retrograde=False):
    """t-component of the circular-orbit 4-velocity."""
    a, q = _pair(a, q)
    omega = kerr_disk_omega(r, a, q, retrograde)
    if abs(omega) <= 1e-14:
        return 0.0
    gtt, gtphi, gphiphi, *_ = _kn_equatorial(r, a, q)
    den = -gtt - 2.0 * omega * gtphi - omega * omega * gphiphi
    if den <= 1e-12:
        return 0.0
    return 1.0 / math.sqrt(den)


_KN_ISCO_CACHE = {}
_KN_IBCO_CACHE = {}


def _isco_energy(r, a, q, retrograde=False):
    """Circular-orbit energy E(r) for a neutral test particle."""
    omega = kerr_disk_omega(r, a, q, retrograde)
    ut = kerr_disk_ut(r, a, q, retrograde)
    if abs(omega) <= 1e-14 or ut <= 0.0:
        return None
    gtt, gtphi, gphiphi, *_ = _kn_equatorial(r, a, q)
    return -(gtt + omega * gtphi) * ut


def _isco_slope(r, a, q, retrograde=False):
    """dE/dr by central difference; E(r) is smooth above the horizon."""
    h = max(1e-7, 1e-6 * r)
    e1 = _isco_energy(r - h, a, q, retrograde)
    e2 = _isco_energy(r + h, a, q, retrograde)
    if e1 is None or e2 is None:
        return None
    return (e2 - e1) / (2.0 * h)


def kerr_isco(a, q=0.0, retrograde=False):
    """ISCO radius for a neutral test particle.

    The default is the co-rotating (relative to spin) orbit; Q=0 uses the
    exact Bardeen-Press-Teukolsky expression.  ``retrograde=True`` gives the
    counter-rotating ISCO, used for the orbit diagram.  For Q>0 the circular
    orbit energy E(r) is bracketed and dE/dr=0 is solved by bisection.
    """
    a, q = _pair(a, q)
    if abs(q) <= 1e-14 and not retrograde:
        return _kerr_isco_zeroq(a)
    key = (round(a, 6), round(q, 6), bool(retrograde))
    cached = _KN_ISCO_CACHE.get(key)
    if cached is not None:
        return cached
    r_h = kerr_horizon(a, q)
    lo = r_h + 0.01
    hi = 40.0
    n = 320
    bracket = None
    prev_r = lo
    prev_s = _isco_slope(lo, a, q, retrograde)
    for i in range(1, n + 1):
        r = lo + (hi - lo) * i / n
        slope = _isco_slope(r, a, q, retrograde)
        if prev_s is not None and slope is not None and prev_s < 0.0 <= slope:
            bracket = (prev_r, r)
            break
        prev_r, prev_s = r, slope
    if bracket is None:
        value = max(r_h + 0.1, 6.0)
    else:
        left, right = bracket
        for _ in range(60):
            mid = 0.5 * (left + right)
            slope = _isco_slope(mid, a, q, retrograde)
            if slope is None or slope < 0.0:
                left = mid
            else:
                right = mid
            if right - left < 1e-8:
                break
        value = max(r_h + 0.02, min(hi, 0.5 * (left + right)))
    _KN_ISCO_CACHE[key] = value
    return value


_KN_PHOTON_CACHE = {}


def kerr_ibco(a, q=0.0):
    """Innermost bound circular orbit: the co-rotating circular orbit with E=1.

    For Schwarzschild this is r=4M, between the horizon and the ISCO.
    """
    a, q = _pair(a, q)
    key = (round(a, 6), round(q, 6))
    cached = _KN_IBCO_CACHE.get(key)
    if cached is not None:
        return cached
    isco = kerr_isco(a, q)
    r_h = kerr_horizon(a, q)
    e_isco = _isco_energy(isco, a, q)
    if e_isco is None or e_isco >= 1.0:
        return isco
    n = 240
    prev_r = isco
    prev = e_isco - 1.0
    bracket = None
    for i in range(1, n + 1):
        r = isco - (isco - r_h - 0.01) * i / n
        e = _isco_energy(r, a, q)
        if e is None:
            continue
        cur = e - 1.0
        if prev < 0.0 <= cur:
            bracket = (prev_r, r)
            break
        prev_r, prev = r, cur
    if bracket is None:
        value = max(r_h + 0.05, isco - 0.5)
    else:
        left, right = bracket
        for _ in range(60):
            mid = 0.5 * (left + right)
            e = _isco_energy(mid, a, q)
            if e is None:
                break
            if e < 1.0:
                left = mid
            else:
                right = mid
            if right - left < 1e-8:
                break
        value = max(r_h + 0.02, 0.5 * (left + right))
    _KN_IBCO_CACHE[key] = value
    return value


def kerr_photon_radius(a, q=0.0, retrograde=False):
    """Equatorial circular-photon-orbit radius (the photon ring).

    The default is co-rotating with the spin; ``retrograde=True`` gives the
    counter-rotating photon orbit.  Q=0 uses the exact Kerr expression; Q>0
    solves R=R'=0 numerically and picks the smaller/larger root.
    """
    a, q = _pair(a, q)
    if abs(q) <= 1e-12:
        s = 1.0 if retrograde else -1.0
        return 2.0 * (1.0 + math.cos(
            2.0 / 3.0 * math.acos(s * abs(a))))
    r_h = kerr_horizon(a, q)
    if abs(a) < 1e-7:
        disc = 9.0 - 8.0 * q * q
        return 0.5 * (3.0 + math.sqrt(max(0.0, disc)))
    key = (round(a, 6), round(q, 6), bool(retrograde))
    cached = _KN_PHOTON_CACHE.get(key)
    if cached is not None:
        return cached

    def f(r):
        delta = r * r - 2.0 * r + a * a + q * q
        h = 2.0 * r * delta / (r - 1.0)
        b = (r * r + a * a - h) / a
        c = b - a
        return c * c - 2.0 * r * h / (r - 1.0)

    lo = r_h + 1e-3
    hi = 8.0
    n = 400
    prev_r = lo
    prev = f(lo)
    roots = []
    for i in range(1, n + 1):
        r = lo + (hi - lo) * i / n
        cur = f(r)
        if prev == 0.0:
            roots.append(prev_r)
        elif prev * cur < 0.0:
            left, right = prev_r, r
            for _ in range(60):
                mid = 0.5 * (left + right)
                if f(left) * f(mid) <= 0.0:
                    right = mid
                else:
                    left = mid
            roots.append(0.5 * (left + right))
        prev_r, prev = r, cur
    if roots:
        value = max(r_h + 0.01,
                    max(roots) if retrograde else min(roots))
    else:
        value = max(r_h + 0.2, 3.0)
    _KN_PHOTON_CACHE[key] = value
    return value


def kerr_ergosphere(a, theta=math.pi / 2.0, q=0.0):
    """Outer ergosphere radius at polar angle theta."""
    a, q = _pair(a, q)
    c = math.cos(theta)
    return 1.0 + math.sqrt(max(0.0, 1.0 - q * q - a * a * c * c))


def kerr_metric(r, theta, a, q=0.0):
    """Return (Sigma, Delta, g_tt, g_tphi, g_phiphi, g_rr, g_thetatheta)."""
    a, q = _pair(a, q)
    s = math.sin(theta)
    c = math.cos(theta)
    s2, c2 = s * s, c * c
    sigma = r * r + a * a * c2
    delta = r * r - 2.0 * r + a * a + q * q
    gtt = -(1.0 - (2.0 * r - q * q) / sigma)
    gtphi = -a * (2.0 * r - q * q) * s2 / sigma
    gphiphi = (r * r + a * a
               + a * a * (2.0 * r - q * q) * s2 / sigma) * s2
    grr = sigma / delta if abs(delta) > 1e-15 else 1e15
    return sigma, delta, gtt, gtphi, gphiphi, grr, sigma


def disk_profile_raw(r, r_in):
    x = min(1.0, max(0.0, float(r_in) / max(float(r), 1e-12)))
    return (x ** 0.75) * max(0.0, 1.0 - math.sqrt(x)) ** 0.25


_PROFILE_PEAK = {}

def disk_profile_peak(r_in):
    key = round(float(r_in), 8)
    if key not in _PROFILE_PEAK:
        r_peak = (49.0 / 36.0) * float(r_in)
        _PROFILE_PEAK[key] = max(disk_profile_raw(r_peak, r_in), 1e-12)
    return _PROFILE_PEAK[key]


def kerr_redshift(r_emit, E_phys, L_phys, a, q=0.0):
    """nu_camera / nu_emitter for a neutral Keplerian disk.

    The camera's local energy is normalised to 1.  E_phys and L_phys are
    conserved quantities of the physical (disk -> camera) photon, obtained
    from the opposite local direction -n at the camera.
    """
    a, q = _pair(a, q)
    if r_emit <= kerr_horizon(a, q) + 1e-5:
        return 0.0
    ut = kerr_disk_ut(r_emit, a, q)
    if ut <= 0.0:
        return 0.0
    omega = kerr_disk_omega(r_emit, a, q)
    denom = ut * (E_phys - omega * L_phys)
    if denom <= 1e-9:
        return 0.0
    return 1.0 / denom


# --------------------------------------------------------------------------- #
# Reference integrator (NumPy/Python, used by tests)
# --------------------------------------------------------------------------- #
def _camera_initial(r0, theta0, phi0, a, x, y, fov, q=0.0):
    a, q = _pair(a, q)
    sigma, delta, gtt, gtphi, gphiphi, grr, gthth = kerr_metric(
        r0, theta0, a, q)
    omega_cam = -gtphi / gphiphi
    det = gtphi * gtphi - gtt * gphiphi
    ut = math.sqrt(gphiphi / max(det, 1e-30))
    s0, c0 = math.sin(theta0), math.cos(theta0)
    tan_half = math.tan(0.5 * math.radians(fov))
    nr = -1.0
    nphi = float(x) * tan_half
    nth = -float(y) * tan_half
    norm = math.sqrt(nr * nr + nphi * nphi + nth * nth)
    nr, nphi, nth = nr / norm, nphi / norm, nth / norm

    pt = ut
    pr = nr / math.sqrt(grr)
    pth = nth / math.sqrt(gthth)
    pphi = omega_cam * ut + nphi / math.sqrt(gphiphi)
    E = -(gtt * pt + gtphi * pphi)
    L = gtphi * pt + gphiphi * pphi
    ptheta = math.sqrt(gthth) * nth
    Q = ptheta * ptheta + c0 * c0 * (
        L * L / max(s0 * s0, 1e-12) - a * a * E * E
    )
    if E <= 1e-9:
        raise ValueError("camera photon energy is not positive")
    L /= E
    Q = max(0.0, Q / (E * E))
    vr = pr / E
    vth = pth / E

    # Physical photon = opposite local direction at the same camera.
    pphi_ph = omega_cam * ut - nphi / math.sqrt(gphiphi)
    E_ph = -(gtt * ut + gtphi * pphi_ph)
    # The conserved axial momentum is p_phi, not p^phi.  Contracting the
    # ZAMO frame gives p_phi = -n_phi sqrt(g_phiphi) (the drag terms cancel);
    # with p^phi the Omega*L term would be ~sqrt(g_phiphi) too small and the
    # disk would lose its Doppler asymmetry.
    L_ph = gtphi * ut + gphiphi * pphi_ph
    return {
        "r": float(r0), "theta": float(theta0), "phi": float(phi0),
        "vr": float(vr), "vth": float(vth),
        "L": float(L), "Q": float(Q), "E": float(E),
        "E_ph": float(E_ph), "L_ph": float(L_ph),
        "cam": (sigma, delta, gtt, gtphi, gphiphi, grr, gthth,
                omega_cam, ut, s0, c0),
    }


def _rhs(r, theta, phi, vr, vth, a, L, Q, q=0.0):
    s, c = math.sin(theta), math.cos(theta)
    s2, c2 = s * s, c * c
    sigma = r * r + a * a * c2
    delta = r * r - 2.0 * r + a * a + q * q
    H = r * r + a * a - a * L
    C = (L - a) * (L - a) + Q
    Rp = 4.0 * r * H - (2.0 * r - 2.0) * C
    s_safe = max(abs(s), 1e-9)
    Tp = -2.0 * a * a * c * s + 2.0 * L * L * c / (s_safe ** 3)
    sigma_dot = 2.0 * r * vr - 2.0 * a * a * c * s * vth
    dr = vr
    dth = vth
    dvr = Rp / (2.0 * sigma * sigma) - vr * sigma_dot / sigma
    dvth = Tp / (2.0 * sigma * sigma) - vth * sigma_dot / sigma
    dph = (a * H / max(delta, 1e-12) - a + L / max(s2, 1e-12)) / sigma
    return dr, dth, dph, dvr, dvth


def _rk4(state, h, a, L, Q, q=0.0):
    r, th, ph, vr, vth = state
    k1 = _rhs(r, th, ph, vr, vth, a, L, Q, q)
    k2 = _rhs(r + 0.5 * h * k1[0], th + 0.5 * h * k1[1],
              ph + 0.5 * h * k1[2], vr + 0.5 * h * k1[3],
              vth + 0.5 * h * k1[4], a, L, Q, q)
    k3 = _rhs(r + 0.5 * h * k2[0], th + 0.5 * h * k2[1],
              ph + 0.5 * h * k2[2], vr + 0.5 * h * k2[3],
              vth + 0.5 * h * k2[4], a, L, Q, q)
    k4 = _rhs(r + h * k3[0], th + h * k3[1], ph + h * k3[2],
              vr + h * k3[3], vth + h * k3[4], a, L, Q, q)
    return (
        r + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
        th + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
        ph + h / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
        vr + h / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]),
        vth + h / 6.0 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]),
    )


def kerr_sky_direction(r, theta, phi, vr, vth, vphi):
    """Convert BL coordinate velocities at large r to a world direction."""
    st, ct = math.sin(theta), math.cos(theta)
    sp, cp = math.sin(phi), math.cos(phi)
    er = (st * cp, st * sp, ct)
    et = (ct * cp, ct * sp, -st)
    ep = (-sp, cp, 0.0)
    scale = r
    x = vr * er[0] + scale * vth * et[0] + scale * st * vphi * ep[0]
    y = vr * er[1] + scale * vth * et[1] + scale * st * vphi * ep[1]
    z = vr * er[2] + scale * vth * et[2]
    n = math.sqrt(x * x + y * y + z * z)
    return (x / n, y / n, z / n) if n > 1e-30 else (0.0, 0.0, 1.0)


def trace_kerr(distance=46.0, inclination=84.0, azimuth=0.0, fov=32.0,
               x=0.0, y=0.0, a=0.7, q=0.0, max_steps=320,
               step_scale=0.055, r_escape=None):
    """Reference trace of one pixel; returns a diagnostic dictionary."""
    a, q = _pair(a, q)
    st = _camera_initial(distance, math.radians(inclination),
                         math.radians(azimuth), a, x, y, fov, q)
    state = [st["r"], st["theta"], st["phi"], st["vr"], st["vth"]]
    r_h = kerr_horizon(a, q)
    r_escape = float(r_escape or max(600.0, 18.0 * distance))
    status = "max"
    steps = 0
    for steps in range(1, int(max_steps) + 1):
        r = state[0]
        if r <= r_h + 0.02:
            status = "captured"
            break
        if r >= r_escape:
            status = "escaped"
            break
        h = float(step_scale) * max(r, 0.5)
        state = _rk4(state, h, a, st["L"], st["Q"], q)
    r, th, ph, vr, vth = state
    dph = _rhs(r, th, ph, vr, vth, a, st["L"], st["Q"], q)[2]
    return {
        "status": status, "steps": steps, "state": state,
        "direction": kerr_sky_direction(r, th, ph, vr, vth, dph),
        "L": st["L"], "Q": st["Q"], "E_ph": st["E_ph"],
        "L_ph": st["L_ph"], "radius": r,
    }


class KerrConfig:
    """Render parameters for the Kerr app and cover generator."""

    __slots__ = (
        "width", "height", "distance", "inclination", "azimuth", "fov",
        "spin", "charge", "disk_on", "r_in", "r_out",
        "peak_temperature", "opacity",
        "doppler", "exposure", "star_gain", "disk_gain",
        "mass", "star_density",
        "max_steps", "step_scale", "r_escape", "time",
    )

    def __init__(self, width=600, height=600, distance=46.0,
                 inclination=84.0, azimuth=0.0, fov=32.0, spin=0.8,
                 charge=0.0,
                 disk_on=True, r_in=6.0, r_out=13.0,
                 peak_temperature=5000.0, opacity=0.45, doppler=True,
                 exposure=1.3, star_gain=1.5, disk_gain=0.22,
                 mass=1.0e8, star_density=2.0,
                 max_steps=800, step_scale=0.02, r_escape=0.0, time=0.0):
        self.width = int(width)
        self.height = int(height)
        self.distance = float(distance)
        self.inclination = float(inclination)
        self.azimuth = float(azimuth)
        self.fov = float(fov)
        self.spin = max(-0.99, min(0.99, float(spin)))
        qmax = math.sqrt(max(
            0.0, KN_MAX * KN_MAX - self.spin * self.spin))
        self.charge = max(-qmax, min(qmax, float(charge)))
        self.disk_on = bool(disk_on)
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.peak_temperature = float(peak_temperature)
        self.opacity = float(opacity)
        self.doppler = bool(doppler)
        self.exposure = float(exposure)
        self.star_gain = float(star_gain)
        self.disk_gain = float(disk_gain)
        self.mass = max(1.0e5, float(mass))
        self.star_density = max(0.0, min(5.0, float(star_density)))
        self.max_steps = int(max_steps)
        self.step_scale = float(step_scale)
        self.r_escape = float(r_escape)
        self.time = float(time)
# --------------------------------------------------------------------------- #
# Taichi kernel (shared textures and Taichi state live in lensing.py)
# --------------------------------------------------------------------------- #
if _TI_IMPORTED:
    @ti.func
    def _kerr_disk_omega3(r, a, q):
        # General equatorial circular orbit, with the exact Q=0 branch kept
        # for numerical stability near the Kerr ISCO.
        a2 = a * a
        q2 = q * q
        r2 = r * r
        r3 = r2 * r
        gtt_r = -2.0 / r2 + 2.0 * q2 / r3
        gt_r = 2.0 * a / r2 - 2.0 * a * q2 / r3
        gp_r = 2.0 * r - 2.0 * a2 / r2 + 2.0 * a2 * q2 / r3
        disc = ti.max(0.0, gt_r * gt_r - gtt_r * gp_r)
        s = ti.select(a >= 0.0, 1.0, -1.0)
        omega_kn = (-gt_r + s * ti.sqrt(disc)) / ti.max(gp_r, 1e-9)
        omega_k = s / (ti.pow(r, 1.5) + ti.abs(a))
        return ti.select(ti.abs(q) <= 1e-7, omega_k, omega_kn)

    @ti.func
    def _kerr_disk_ut3(r, a, q):
        omega = _kerr_disk_omega3(r, a, q)
        A = 2.0 * r - q * q
        r2 = r * r
        gtt = -1.0 + A / r2
        gtphi = -a * A / r2
        gphiphi = r2 + a * a + a * a * A / r2
        den = -gtt - 2.0 * omega * gtphi - omega * omega * gphiphi
        return 1.0 / ti.sqrt(ti.max(den, 1e-8))

    @ti.func
    def _hash31(p):
        q = p * 0.1031
        q = q - ti.floor(q)
        d = q.dot(ti.Vector([q[1] + 33.33, q[2] + 33.33, q[0] + 33.33]))
        q = q + ti.Vector([d, d, d])
        v = (q[0] + q[1]) * q[2]
        return v - ti.floor(v)

    @ti.func
    def _sample_sky(d, neb: ti.template(), sw: ti.i32, sh: ti.i32,
                    density: ti.f32):
        """Smooth nebula + two isotropic 3D star-cell layers.

        The cells live in direction space, so there is no equirectangular
        meridian: the symmetry-plane column no longer repeats a texture
        column (which was the visible vertical 'axis').
        """
        n = d / ti.max(ti.sqrt(d.dot(d)), 1e-12)
        col = _sample_star(n, neb, sw, sh)

        # Bright near layer: one star per 210-cell direction cell.
        g = n * 210.0
        cell = ti.Vector([ti.floor(g[0]), ti.floor(g[1]), ti.floor(g[2])])
        s = _hash31(cell)
        if s > 1.0 - 0.009 * density:
            f = g - cell - ti.Vector([0.5, 0.5, 0.5])
            o = ti.Vector([
                _hash31(cell + ti.Vector([3.7, 3.7, 3.7])),
                _hash31(cell + ti.Vector([9.1, 9.1, 9.1])),
                _hash31(cell + ti.Vector([21.3, 21.3, 21.3])),
            ]) - ti.Vector([0.5, 0.5, 0.5])
            q = f - o * 0.35
            star = 1.0 - _smoothstep3(0.0, 0.33, ti.sqrt(q.dot(q)))
            temp = _hash31(cell + ti.Vector([19.3, 19.3, 19.3]))
            cool = ti.Vector([1.0, 0.78, 0.58])
            hot = ti.Vector([0.72, 0.86, 1.0])
            star_col = cool * (1.0 - temp) + hot * temp
            amp = 1.2 + 1.6 * (s * 137.0 - ti.floor(s * 137.0))
            col += star_col * (star * amp)

        # Faint far layer: denser, smaller, cooler points.
        g2 = n * 380.0
        cell2 = ti.Vector([ti.floor(g2[0]), ti.floor(g2[1]), ti.floor(g2[2])])
        s2 = _hash31(cell2 + ti.Vector([71.1, 71.1, 71.1]))
        if s2 > 1.0 - 0.006 * density:
            f2 = g2 - cell2 - ti.Vector([0.5, 0.5, 0.5])
            o2 = ti.Vector([
                _hash31(cell2 + ti.Vector([5.3, 5.3, 5.3])),
                _hash31(cell2 + ti.Vector([11.9, 11.9, 11.9])),
                _hash31(cell2 + ti.Vector([27.1, 27.1, 27.1])),
            ]) - ti.Vector([0.5, 0.5, 0.5])
            q2 = f2 - o2 * 0.35
            star2 = 1.0 - _smoothstep3(0.0, 0.27, ti.sqrt(q2.dot(q2)))
            col += ti.Vector([0.85, 0.9, 1.0]) * (star2 * 0.7)
        return col

    # ------------------------------------------------------------------ #
    # Kerr-Schild Cartesian integrator (regular on the polar axis)
    # ------------------------------------------------------------------ #
    @ti.func
    def _ks_r(x, y, z, a):
        rho2 = x * x + y * y + z * z
        a2 = a * a
        disc = ti.sqrt(ti.max(0.0, (rho2 - a2) ** 2 + 4.0 * a2 * z * z))
        return ti.sqrt(ti.max(0.0, 0.5 * (rho2 - a2 + disc)))

    @ti.func
    def _ks_rd(x, y, z, a, r):
        a2 = a * a
        d = r * (x * x + y * y) / (r * r + a2) ** 2 + z * z / (r ** 3)
        d = ti.max(d, 1e-12)
        return ti.Vector([x / (r * r + a2) / d,
                          y / (r * r + a2) / d,
                          z / (r * r) / d])

    @ti.func
    def _ks_rhs(state, a, q):
        x, y, z = state[0], state[1], state[2]
        px, py, pz = state[3], state[4], state[5]
        r = _ks_r(x, y, z, a)
        rd = _ks_rd(x, y, z, a, r)
        a2 = a * a
        D = r ** 4 + a2 * z * z
        N = (2.0 * r - q * q) * r * r
        Nr = 6.0 * r * r - 2.0 * q * q * r
        Dr = 4.0 * r ** 3
        Dz = 2.0 * a2 * z
        f = N / D
        fr = (Nr * D - N * Dr) / (D * D)
        fz = -N * Dz / (D * D)
        fx = fr * rd[0]
        fy = fr * rd[1]
        fzz = fr * rd[2] + fz
        Dk = r * r + a2
        Dk2 = Dk * Dk
        kx = (r * x + a * y) / Dk
        ky = (r * y - a * x) / Dk
        kz = z / r
        Nx = r * x + a * y
        Ny = r * y - a * x
        kxx = ((rd[0] * x + r) * Dk - Nx * 2.0 * r * rd[0]) / Dk2
        kxy = ((rd[1] * x + a) * Dk - Nx * 2.0 * r * rd[1]) / Dk2
        kxz = ((rd[2] * x) * Dk - Nx * 2.0 * r * rd[2]) / Dk2
        kyx = ((rd[0] * y - a) * Dk - Ny * 2.0 * r * rd[0]) / Dk2
        kyy = ((rd[1] * y + r) * Dk - Ny * 2.0 * r * rd[1]) / Dk2
        kyz = ((rd[2] * y) * Dk - Ny * 2.0 * r * rd[2]) / Dk2
        kzx = -z * rd[0] / (r * r)
        kzy = -z * rd[1] / (r * r)
        kzz = (r - z * rd[2]) / (r * r)
        S = 1.0 + kx * px + ky * py + kz * pz
        vx = px - f * kx * S
        vy = py - f * ky * S
        vz = pz - f * kz * S
        kpx = kxx * px + kyx * py + kzx * pz
        kpy = kxy * px + kyy * py + kzy * pz
        kpz = kxz * px + kyz * py + kzz * pz
        return ti.Vector([
            vx, vy, vz,
            0.5 * S * S * fx + f * S * kpx,
            0.5 * S * S * fy + f * S * kpy,
            0.5 * S * S * fzz + f * S * kpz,
        ])

    @ti.func
    def _ks_rk4(state, h, a, q):
        k1 = _ks_rhs(state, a, q)
        k2 = _ks_rhs(state + 0.5 * h * k1, a, q)
        k3 = _ks_rhs(state + 0.5 * h * k2, a, q)
        k4 = _ks_rhs(state + h * k3, a, q)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @ti.func
    def _ks_phi(x, y, a, sdisc, rp, rm):
        rd = ti.sqrt(ti.max(x * x + y * y - a * a, 1e-9))
        psi = ti.atan2(y, x) - ti.atan2(a, rd)
        shift = 0.0
        if ti.abs(a) > 1e-12:
            shift = a / (2.0 * sdisc) * ti.log(
                ti.max(ti.abs((rd - rp) / ti.max(rd - rm, 1e-9)), 1e-12))
        return rd, psi - shift

    @ti.kernel
    def _ks_kernel(
        out: ti.template(), star: ti.template(), lut: ti.template(),
        disk_tex: ti.template(), tw: ti.i32, th_tex: ti.i32,
        sw: ti.i32, sh: ti.i32,
        width: ti.i32, height: ti.i32,
        cx: ti.f32, cy: ti.f32, cz: ti.f32,
        u0: ti.f32, u1: ti.f32, u2: ti.f32, u3: ti.f32,
        er0: ti.f32, er1: ti.f32, er2: ti.f32, er3: ti.f32,
        et0: ti.f32, et1: ti.f32, et2: ti.f32, et3: ti.f32,
        ep0: ti.f32, ep1: ti.f32, ep2: ti.f32, ep3: ti.f32,
        tan_half: ti.f32, a: ti.f32, q: ti.f32,
        sdisc: ti.f32, rp: ti.f32, rm: ti.f32,
        r_in: ti.f32, r_out: ti.f32, t_peak: ti.f32,
        profile_peak: ti.f32, disk_on: ti.i32, opacity: ti.f32,
        doppler_on: ti.i32, exposure: ti.f32, star_gain: ti.f32,
        star_density: ti.f32,
        disk_gain: ti.f32, max_steps: ti.i32, step_scale: ti.f32,
        r_escape: ti.f32, r_horizon: ti.f32, time: ti.f32,
        jitter: ti.f32,
    ):
        for j, i in ti.ndrange(height, width):
            x = (2.0 * (i + 0.5) / width - 1.0) * tan_half
            y = (1.0 - 2.0 * (j + 0.5) / height) * tan_half
            # p = u + n, n = -e_r + x tan e_phi - y tan e_theta (covariant).
            # Normalise n like the BL camera did, otherwise E_ph/L_ph (and
            # the disk Doppler) are off by the local direction norm.
            # x, y are already multiplied by tan_half in the loop above.
            nlen = ti.sqrt(1.0 + x * x + y * y)
            p0 = u0 + (-er0 + x * ep0 - y * et0) / nlen
            p1 = u1 + (-er1 + x * ep1 - y * et1) / nlen
            p2 = u2 + (-er2 + x * ep2 - y * et2) / nlen
            p3 = u3 + (-er3 + x * ep3 - y * et3) / nlen
            # physical photon = u - n
            q0 = u0 - (-er0 + x * ep0 - y * et0) / nlen
            q1 = u1 - (-er1 + x * ep1 - y * et1) / nlen
            q2 = u2 - (-er2 + x * ep2 - y * et2) / nlen
            q3 = u3 - (-er3 + x * ep3 - y * et3) / nlen
            E_ph = -q0
            L_ph = -cy * q1 + cx * q2
            scale = -1.0 / p0
            state = ti.Vector([cx, cy, cz,
                               p1 * scale, p2 * scale, p3 * scale])
            rad = ti.Vector([0.0, 0.0, 0.0])
            trans = 1.0
            alive = 1
            captured = 0
            old_z = cz
            for step in range(max_steps):
                if alive == 1:
                    r = _ks_r(state[0], state[1], state[2], a)
                    if r <= r_horizon + 0.02:
                        captured = 1
                        alive = 0
                    elif r >= r_escape:
                        alive = 0
                    else:
                        h = step_scale * ti.max(r, 0.5)
                        new_state = _ks_rk4(state, h, a, q)
                        new_z = new_state[2]
                        if disk_on == 1 and old_z * new_z <= 0.0:
                            dz = old_z - new_z
                            frac = 0.0 if ti.abs(dz) < 1e-9 else old_z / dz
                            frac = ti.min(1.0, ti.max(0.0, frac))
                            hx = state[0] + (new_state[0] - state[0]) * frac
                            hy = state[1] + (new_state[1] - state[1]) * frac
                            rd, phi = _ks_phi(hx, hy, a, sdisc, rp, rm)
                            if rd >= r_in and rd <= r_out:
                                raw = ti.pow(r_in / rd, 0.75) * ti.pow(
                                    ti.max(0.0, 1.0 - ti.sqrt(r_in / rd)),
                                    0.25) / profile_peak
                                edge = _smoothstep3(r_in, r_in + 0.55, rd) * (
                                    1.0 - _smoothstep3(r_out - 1.2, r_out, rd))
                                u_tex = ti.log(rd / r_in) / ti.log(
                                    r_out / r_in)
                                sdir = ti.select(a >= 0.0, 1.0, -1.0)
                                phase = phi / TAU - sdir * 0.18 * time * ti.pow(
                                    r_in / rd, 1.5)
                                grain = _sample_disk_tex(
                                    disk_tex, tw, th_tex, u_tex, phase)
                                alpha = ti.min(1.0, ti.max(
                                    0.0, opacity * edge *
                                    (0.55 + 0.55 * grain)))
                                ut_emit = _kerr_disk_ut3(rd, a, q)
                                omega = _kerr_disk_omega3(rd, a, q)
                                den_dop = ut_emit * (E_ph - omega * L_ph)
                                den_grav = ut_emit * E_ph
                                den = ti.select(doppler_on == 1, den_dop,
                                                den_grav)
                                g = 1.0
                                if den > 1e-6:
                                    g = 1.0 / den
                                g = ti.min(5.0, ti.max(0.05, g))
                                temp = g * t_peak * raw
                                col = _sample_lut(temp, lut)
                                intensity = disk_gain * ti.pow(g, 4.0) * ti.pow(
                                    raw, 4.0) * (0.55 + 1.05 * grain)
                                rad = rad * (1.0 - alpha) + col * (
                                    intensity * alpha)
                                trans *= (1.0 - alpha)
                                if trans < 0.002:
                                    alive = 0
                        state = new_state
                        old_z = new_z
            if alive == 1:
                captured = 1
            if captured == 0 and trans > 0.002:
                v = _ks_rhs(state, a, q)
                d = ti.Vector([v[0], v[1], v[2]])
                direction = d / ti.max(ti.sqrt(d.dot(d)), 1e-12)
                rad += _sample_sky(
                    direction, star, sw, sh, star_density) * (
                    star_gain * trans)
            c = rad * exposure
            c = ti.Vector([_aces(c[0]), _aces(c[1]), _aces(c[2])])
            c = ti.Vector([ti.pow(c[0], 1.0 / 2.2),
                           ti.pow(c[1], 1.0 / 2.2),
                           ti.pow(c[2], 1.0 / 2.2)])
            hsh = ti.sin(i * 12.9898 + j * 78.233) * 43758.5453
            dth = (hsh - ti.floor(hsh) - 0.5) * jitter
            out[j, i] = ti.Vector([
                ti.min(1.0, ti.max(0.0, c[0] + dth)),
                ti.min(1.0, ti.max(0.0, c[1] + dth)),
                ti.min(1.0, ti.max(0.0, c[2] + dth)),
            ])


def _ks_radius(x, y, z, a):
    rho2 = x * x + y * y + z * z
    a2 = a * a
    disc = math.sqrt(max(0.0, (rho2 - a2) ** 2 + 4.0 * a2 * z * z))
    return math.sqrt(max(0.0, 0.5 * (rho2 - a2 + disc)))


def _ks_metric(x, y, z, a, q):
    r = _ks_radius(x, y, z, a)
    a2 = a * a
    D = r ** 4 + a2 * z * z
    f = (2.0 * r - q * q) * r * r / D
    kx = (r * x + a * y) / (r * r + a2)
    ky = (r * y - a * x) / (r * r + a2)
    kz = z / r
    k = np.array([1.0, kx, ky, kz])
    g = np.zeros((4, 4))
    g[0, 0] = -1.0 + f
    g[0, 1:] = f * k[1:]
    g[1:, 0] = f * k[1:]
    g[1:, 1:] = np.eye(3) + f * np.outer(k[1:], k[1:])
    return g


def _ks_phi_shift(r, a, q):
    if abs(a) < 1e-12:
        return 0.0
    s = math.sqrt(max(1.0 - a * a - q * q, 1e-12))
    rp, rm = 1.0 + s, 1.0 - s
    return a / (2.0 * s) * math.log(abs((r - rp) / (r - rm)))


def _ks_camera_data(r0, theta0, phi0, a, q):
    st = _camera_initial(r0, theta0, phi0, a, 0.0, 0.0, 32.0, q)
    sigma, delta, gtt, gtphi, gphiphi, grr, gthth, omega, ut, s0, c0 = st["cam"]
    G = (2.0 * r0 - q * q) / delta
    H = a / delta
    psi0 = phi0 + _ks_phi_shift(r0, a, q)
    s, c = math.sin(theta0), math.cos(theta0)
    sp, cp = math.sin(psi0), math.cos(psi0)
    px = (r0 * cp - a * sp) * s
    py = (r0 * sp + a * cp) * s
    pz = r0 * c
    xr, yr, zr = s * cp, s * sp, c
    xt, yt, zt = (r0 * cp - a * sp) * c, (r0 * sp + a * cp) * c, -r0 * s
    xp, yp, zp = -(r0 * sp + a * cp) * s, (r0 * cp - a * sp) * s, 0.0
    # BL d/dr in KS: G d_T + d_r + H d_psi; d_psi = (0, -y, x, 0)
    er = np.array([G, xr - H * py, yr + H * px, zr]) / math.sqrt(grr)
    et = np.array([0.0, xt, yt, zt]) / math.sqrt(gthth)
    ep = np.array([0.0, -py, px, 0.0]) / math.sqrt(gphiphi)
    u = np.array([ut, -omega * ut * py, omega * ut * px, 0.0])
    g = _ks_metric(px, py, pz, a, q)
    sdisc = math.sqrt(max(1.0 - a * a - q * q, 1e-12))
    return {
        "pos": (px, py, pz),
        "u": g @ u, "er": g @ er, "et": g @ et, "ep": g @ ep,
        "s": sdisc, "rp": 1.0 + sdisc, "rm": 1.0 - sdisc,
    }


class KerrRenderer(LensingRenderer):
    """GPU Kerr ray tracer; reuses the shared sky/LUT/disk textures."""

    def __init__(self, star_size=(2048, 1024), seed=7, output_size=840):
        super().__init__(star_size=star_size, seed=seed, output_size=output_size)
        # The point stars are procedural 3D cells in the kernel; the uploaded
        # texture is only the smooth nebula, so it has no star-column seam.
        self._sky_stars = False

    def render(self, config):
        self.prepare()
        width = int(config.width)
        height = int(config.height)
        out = self._ensure_output(width, height)
        a = max(-0.99, min(0.99, float(config.spin)))
        qmax = math.sqrt(max(0.0, KN_MAX * KN_MAX - a * a))
        q = max(-qmax, min(qmax, float(config.charge)))
        r0 = float(config.distance)
        theta0 = math.radians(float(config.inclination))
        phi0 = math.radians(float(config.azimuth))
        cam = _ks_camera_data(r0, theta0, phi0, a, q)
        u, er, et, ep = cam["u"], cam["er"], cam["et"], cam["ep"]
        r_horizon = kerr_horizon(a, q)
        r_in = max(float(config.r_in), r_horizon + 0.1)
        r_out = max(r_in + 1.0, float(config.r_out))
        r_escape = float(config.r_escape)
        if r_escape <= 0.0:
            r_escape = max(600.0, 18.0 * r0)
        # The metric is scale-free: M changes the physical scale and the disk
        # temperature at fixed accretion rate, not the image geometry.
        mass = max(1.0e5, float(config.mass))
        t_peak = float(config.peak_temperature) * (1.0e8 / mass) ** 0.25
        t_peak = max(1000.0, min(40000.0, t_peak))
        # Mass also changes the apparent size: at fixed physical distance the
        # angular scale goes as M.  A full linear scaling over 4 decades would
        # push the disk out of frame, so use a 1/4-power apparent-size zoom.
        fov_eff = float(config.fov) * (1.0e8 / mass) ** 0.25
        fov_eff = max(5.0, min(120.0, fov_eff))
        _ks_kernel(
            out, self._star_field, self._lut_field, self._tex_field,
            512, 256,
            self.star_size[0], self.star_size[1],
            width, height,
            cam["pos"][0], cam["pos"][1], cam["pos"][2],
            u[0], u[1], u[2], u[3],
            er[0], er[1], er[2], er[3],
            et[0], et[1], et[2], et[3],
            ep[0], ep[1], ep[2], ep[3],
            math.tan(0.5 * math.radians(fov_eff)),
            a, q,
            cam["s"], cam["rp"], cam["rm"],
            r_in, r_out, t_peak,
            float(disk_profile_peak(r_in)),
            1 if config.disk_on else 0,
            float(config.opacity),
            1 if config.doppler else 0,
            float(config.exposure), float(config.star_gain),
            float(config.star_density),
            float(config.disk_gain),
            int(config.max_steps), float(config.step_scale), r_escape,
            r_horizon, float(config.time), 0.75 / 255.0,
        )
        ti.sync()
        rgb = out.to_numpy()
        if rgb.shape[0] != height or rgb.shape[1] != width:
            rgb = rgb[:height, :width]
        rgb = np.clip(rgb * 255.0 + 0.5, 0.0, 255.0)
        return rgb.astype(np.uint8)
