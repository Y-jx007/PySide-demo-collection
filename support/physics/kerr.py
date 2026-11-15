# -*- coding: utf-8 -*-
"""Kerr null-geodesic ray tracer (Boyer-Lindquist, G = c = M = 1).

The app integrates the separated Kerr equations with RK4 in affine parameter
lambda.  For a photon with conserved energy E = -p_t, axial angular momentum
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
    blackbody_lut, build_disk_texture, build_starfield,
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
# Kerr geometry
# --------------------------------------------------------------------------- #
def kerr_horizon(a):
    """Outer event horizon r_+ for spin a in [0, 1)."""
    a = max(0.0, min(0.999999, float(a)))
    return 1.0 + math.sqrt(max(0.0, 1.0 - a * a))


def kerr_isco(a):
    """Prograde ISCO radius (Bardeen, Press & Teukolsky)."""
    a = max(0.0, min(0.999999, float(a)))
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0)
    )
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    return 3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def kerr_ergosphere(a, theta=math.pi / 2.0):
    """Outer ergosphere radius at polar angle theta."""
    a = max(0.0, min(0.999999, float(a)))
    c = math.cos(theta)
    return 1.0 + math.sqrt(max(0.0, 1.0 - a * a * c * c))


def kerr_metric(r, theta, a):
    """Return (Sigma, Delta, g_tt, g_tphi, g_phiphi, g_rr, g_thetatheta)."""
    s = math.sin(theta)
    c = math.cos(theta)
    s2, c2 = s * s, c * c
    sigma = r * r + a * a * c2
    delta = r * r - 2.0 * r + a * a
    gtt = -(1.0 - 2.0 * r / sigma)
    gtphi = -2.0 * a * r * s2 / sigma
    gphiphi = (r * r + a * a + 2.0 * a * a * r * s2 / sigma) * s2
    grr = sigma / delta if abs(delta) > 1e-15 else 1e15
    return sigma, delta, gtt, gtphi, gphiphi, grr, sigma


def kerr_disk_omega(r, a):
    """Prograde Keplerian angular velocity on the equatorial plane."""
    return 1.0 / (r ** 1.5 + a)


def kerr_disk_ut(r, a):
    """t-component of the prograde circular-orbit 4-velocity (E=1 units)."""
    den = r ** 1.5 - 3.0 * math.sqrt(r) + 2.0 * a
    if den <= 1e-12:
        return 0.0
    return (r ** 1.5 + a) / (r ** 0.75 * math.sqrt(den))


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


def kerr_redshift(r_emit, E_phys, L_phys, a):
    """nu_camera / nu_emitter for a Keplerian disk.

    The camera's local energy is normalised to 1.  E_phys and L_phys are
    conserved quantities of the physical (disk -> camera) photon, obtained
    from the opposite local direction -n at the camera.
    """
    if r_emit <= kerr_horizon(a) + 1e-5:
        return 0.0
    ut = kerr_disk_ut(r_emit, a)
    if ut <= 0.0:
        return 0.0
    omega = kerr_disk_omega(r_emit, a)
    denom = ut * (E_phys - omega * L_phys)
    if denom <= 1e-9:
        return 0.0
    return 1.0 / denom


# --------------------------------------------------------------------------- #
# Reference integrator (NumPy/Python, used by tests)
# --------------------------------------------------------------------------- #
def _camera_initial(r0, theta0, phi0, a, x, y, fov):
    sigma, delta, gtt, gtphi, gphiphi, grr, gthth = kerr_metric(r0, theta0, a)
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
    L_ph = pphi_ph
    return {
        "r": float(r0), "theta": float(theta0), "phi": float(phi0),
        "vr": float(vr), "vth": float(vth),
        "L": float(L), "Q": float(Q), "E": float(E),
        "E_ph": float(E_ph), "L_ph": float(L_ph),
        "cam": (sigma, delta, gtt, gtphi, gphiphi, grr, gthth,
                omega_cam, ut, s0, c0),
    }


def _rhs(r, theta, phi, vr, vth, a, L, Q):
    s, c = math.sin(theta), math.cos(theta)
    s2, c2 = s * s, c * c
    sigma = r * r + a * a * c2
    delta = r * r - 2.0 * r + a * a
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


def _rk4(state, h, a, L, Q):
    r, th, ph, vr, vth = state
    k1 = _rhs(r, th, ph, vr, vth, a, L, Q)
    k2 = _rhs(r + 0.5 * h * k1[0], th + 0.5 * h * k1[1],
              ph + 0.5 * h * k1[2], vr + 0.5 * h * k1[3],
              vth + 0.5 * h * k1[4], a, L, Q)
    k3 = _rhs(r + 0.5 * h * k2[0], th + 0.5 * h * k2[1],
              ph + 0.5 * h * k2[2], vr + 0.5 * h * k2[3],
              vth + 0.5 * h * k2[4], a, L, Q)
    k4 = _rhs(r + h * k3[0], th + h * k3[1], ph + h * k3[2],
              vr + h * k3[3], vth + h * k3[4], a, L, Q)
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
               x=0.0, y=0.0, a=0.7, max_steps=320, step_scale=0.055,
               r_escape=None):
    """Reference trace of one pixel; returns a diagnostic dictionary."""
    st = _camera_initial(distance, math.radians(inclination),
                         math.radians(azimuth), a, x, y, fov)
    state = [st["r"], st["theta"], st["phi"], st["vr"], st["vth"]]
    r_h = kerr_horizon(a)
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
        state = _rk4(state, h, a, st["L"], st["Q"])
    r, th, ph, vr, vth = state
    dph = _rhs(r, th, ph, vr, vth, a, st["L"], st["Q"])[2]
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
        "spin", "disk_on", "r_in", "r_out", "peak_temperature", "opacity",
        "doppler", "exposure", "star_gain", "disk_gain",
        "max_steps", "step_scale", "r_escape", "time",
    )

    def __init__(self, width=600, height=600, distance=46.0,
                 inclination=84.0, azimuth=0.0, fov=32.0, spin=0.8,
                 disk_on=True, r_in=6.0, r_out=13.0,
                 peak_temperature=5000.0, opacity=0.45, doppler=True,
                 exposure=1.3, star_gain=1.5, disk_gain=0.22,
                 max_steps=800, step_scale=0.02, r_escape=0.0, time=0.0):
        self.width = int(width)
        self.height = int(height)
        self.distance = float(distance)
        self.inclination = float(inclination)
        self.azimuth = float(azimuth)
        self.fov = float(fov)
        self.spin = max(0.0, min(0.99, float(spin)))
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
# -*- coding: utf-8 -*-
"""Taichi Kerr backend (continued from the pure functions above)."""
from support.physics.lensing import (
    LensingRenderer, _aces, _sample_disk_tex, _sample_lut, _sample_star,
    _smoothstep3, TAU, BB_N, BB_TMIN, BB_TMAX,
)

_TI_READY = False
_TI_BROKEN = False
_TI_BACKEND = "unavailable"
_TI_ERROR = ""


def taichi_available():
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
    except Exception as exc:  # pragma: no cover
        _TI_BROKEN = True
        _TI_ERROR = str(exc)
        return False


def taichi_backend_name():
    return _TI_BACKEND if _TI_READY else "unavailable"


if _TI_IMPORTED:
    @ti.func
    def _kerr_disk_ut3(r, a):
        den = ti.pow(r, 1.5) - 3.0 * ti.sqrt(r) + 2.0 * a
        return (ti.pow(r, 1.5) + a) / (
            ti.pow(r, 0.75) * ti.sqrt(ti.max(den, 1e-6))
        )

    @ti.func
    def _kerr_rhs3(r, th, ph, vr, vth, a, L, Q):
        s = ti.sin(th)
        c = ti.cos(th)
        s2 = s * s
        c2 = c * c
        sigma = r * r + a * a * c2
        delta = r * r - 2.0 * r + a * a
        H = r * r + a * a - a * L
        C = (L - a) * (L - a) + Q
        Rp = 4.0 * r * H - (2.0 * r - 2.0) * C
        ss = ti.max(ti.abs(s), 1e-9)
        Tp = -2.0 * a * a * c * s + 2.0 * L * L * c / (ss * ss * ss)
        sdot = 2.0 * r * vr - 2.0 * a * a * c * s * vth
        dvr = Rp / (2.0 * sigma * sigma) - vr * sdot / sigma
        dvth = Tp / (2.0 * sigma * sigma) - vth * sdot / sigma
        dph = (a * H / ti.max(delta, 1e-9) - a + L / ti.max(s2, 1e-9)) / sigma
        return ti.Vector([vr, vth, dph, dvr, dvth])

    @ti.func
    def _kerr_rk43(state, h, a, L, Q):
        k1 = _kerr_rhs3(state[0], state[1], state[2], state[3], state[4], a, L, Q)
        k2 = _kerr_rhs3(
            state[0] + 0.5 * h * k1[0], state[1] + 0.5 * h * k1[1],
            state[2] + 0.5 * h * k1[2], state[3] + 0.5 * h * k1[3],
            state[4] + 0.5 * h * k1[4], a, L, Q,
        )
        k3 = _kerr_rhs3(
            state[0] + 0.5 * h * k2[0], state[1] + 0.5 * h * k2[1],
            state[2] + 0.5 * h * k2[2], state[3] + 0.5 * h * k2[3],
            state[4] + 0.5 * h * k2[4], a, L, Q,
        )
        k4 = _kerr_rhs3(
            state[0] + h * k3[0], state[1] + h * k3[1],
            state[2] + h * k3[2], state[3] + h * k3[3],
            state[4] + h * k3[4], a, L, Q,
        )
        return state + (h / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )

    @ti.func
    def _kerr_star_dir(r, th, ph, vr, vth, vph):
        st = ti.sin(th)
        ct = ti.cos(th)
        sp = ti.sin(ph)
        cp = ti.cos(ph)
        er = ti.Vector([st * cp, st * sp, ct])
        et = ti.Vector([ct * cp, ct * sp, -st])
        ep = ti.Vector([-sp, cp, 0.0])
        d = vr * er + r * vth * et + r * st * vph * ep
        n = ti.sqrt(d.dot(d))
        d = d / ti.max(n, 1e-12)
        # The sky is an arbitrary fixed background; a small three-axis tilt
        # avoids a one-texture-column alignment along the camera's symmetry
        # meridian (which otherwise shows up as a false star line).
        cx = 0.541052  # 31 degrees
        sx = 0.840828
        cz = 0.309017  # 18 degrees
        sz = 0.951057
        y1 = cx * d[1] - sx * d[2]
        z1 = sx * d[1] + cx * d[2]
        x2 = cz * d[0] - sz * y1
        y2 = sz * d[0] + cz * y1
        return ti.Vector([x2, y2, z1])

    @ti.kernel
    def _kerr_kernel(
        out: ti.template(), star: ti.template(), lut: ti.template(),
        disk_tex: ti.template(), tw: ti.i32, th_tex: ti.i32,
        sw: ti.i32, sh: ti.i32,
        width: ti.i32, height: ti.i32,
        r0: ti.f32, theta0: ti.f32, phi0: ti.f32, tan_half: ti.f32,
        a: ti.f32,
        gtt_cam: ti.f32, gt_cam: ti.f32, gp_cam: ti.f32,
        ut_cam: ti.f32, omega_cam: ti.f32,
        sqrt_grr: ti.f32, sqrt_gth: ti.f32, sqrt_gphi: ti.f32,
        sin_th0: ti.f32, cos_th0: ti.f32,
        r_in: ti.f32, r_out: ti.f32, t_peak: ti.f32, profile_peak: ti.f32,
        disk_on: ti.i32, opacity: ti.f32, doppler_on: ti.i32,
        exposure: ti.f32, star_gain: ti.f32, disk_gain: ti.f32,
        max_steps: ti.i32, step_scale: ti.f32, r_escape: ti.f32,
        r_horizon: ti.f32, time: ti.f32, jitter: ti.f32,
    ):
        for j, i in ti.ndrange(height, width):
            x = (2.0 * (i + 0.5) / width - 1.0) * tan_half
            y = (1.0 - 2.0 * (j + 0.5) / height) * tan_half
            nr = -1.0
            nphi = x
            nth = -y
            # No random per-pixel jitter: it turns point stars into noise.
            nlen = ti.sqrt(1.0 + nphi * nphi + nth * nth)
            nr /= nlen
            nphi /= nlen
            nth /= nlen

            pt = ut_cam
            pr = nr / sqrt_grr
            pth = nth / sqrt_gth
            pphi = omega_cam * ut_cam + nphi / sqrt_gphi
            E = -(gtt_cam * pt + gt_cam * pphi)
            Lc = gt_cam * pt + gp_cam * pphi
            ptheta = sqrt_gth * nth
            Q = ptheta * ptheta + cos_th0 * cos_th0 * (
                Lc * Lc / ti.max(sin_th0 * sin_th0, 1e-12)
                - a * a * E * E
            )
            Q = ti.max(Q, 0.0)
            L = Lc / ti.max(E, 1e-9)
            Q = Q / ti.max(E * E, 1e-12)
            vr = pr / ti.max(E, 1e-9)
            vth = pth / ti.max(E, 1e-9)

            # Physical photon direction = opposite local direction at camera.
            pphi_ph = omega_cam * ut_cam - nphi / sqrt_gphi
            E_ph = -(gtt_cam * ut_cam + gt_cam * pphi_ph)
            L_ph = pphi_ph

            state = ti.Vector([r0, theta0, phi0, vr, vth])
            rad = ti.Vector([0.0, 0.0, 0.0])
            trans = 1.0
            alive = 1
            captured = 0

            for step in range(max_steps):
                if alive == 1:
                    r = state[0]
                    if r <= r_horizon + 0.02:
                        captured = 1
                        alive = 0
                    elif r >= r_escape:
                        alive = 0
                    else:
                        h = step_scale * ti.max(r, 0.5)
                        old_cos = ti.cos(state[1])
                        new_state = _kerr_rk43(state, h, a, L, Q)
                        new_cos = ti.cos(new_state[1])

                        if disk_on == 1 and old_cos * new_cos <= 0.0:
                            denom = old_cos - new_cos
                            frac = 0.0 if ti.abs(denom) < 1e-9 else old_cos / denom
                            frac = ti.min(1.0, ti.max(0.0, frac))
                            hit_r = state[0] + (new_state[0] - state[0]) * frac
                            hit_phi = state[2] + (new_state[2] - state[2]) * frac
                            if hit_r >= r_in and hit_r <= r_out:
                                raw = ti.pow(r_in / hit_r, 0.75) * ti.pow(
                                    ti.max(0.0, 1.0 - ti.sqrt(r_in / hit_r)), 0.25
                                ) / profile_peak
                                edge = _smoothstep3(r_in, r_in + 0.55, hit_r) * (
                                    1.0 - _smoothstep3(r_out - 1.2, r_out, hit_r)
                                )
                                u_tex = ti.log(hit_r / r_in) / ti.log(r_out / r_in)
                                phase = (hit_phi / TAU
                                         - 0.18 * time * ti.pow(r_in / hit_r, 1.5))
                                grain = _sample_disk_tex(disk_tex, tw, th_tex, u_tex, phase)
                                alpha = ti.min(1.0, ti.max(0.0, opacity * edge * (0.55 + 0.55 * grain)))
                                g = 1.0
                                if doppler_on == 1:
                                    ut_emit = _kerr_disk_ut3(hit_r, a)
                                    omega = 1.0 / (ti.pow(hit_r, 1.5) + a)
                                    den = ut_emit * (E_ph - omega * L_ph)
                                    if den > 1e-6:
                                        g = 1.0 / den
                                    g = ti.min(5.0, ti.max(0.05, g))
                                temp = g * t_peak * raw
                                col = _sample_lut(temp, lut)
                                intensity = disk_gain * ti.pow(g, 4.0) * ti.pow(raw, 4.0)
                                intensity *= (0.55 + 1.05 * grain)
                                rad = rad * (1.0 - alpha) + col * (intensity * alpha)
                                trans *= (1.0 - alpha)
                                if trans < 0.002:
                                    alive = 0
                        state = new_state

            if alive == 1:
                captured = 1
            if captured == 0 and trans > 0.002:
                vph = _kerr_rhs3(
                    state[0], state[1], state[2], state[3], state[4], a, L, Q
                )[2]
                direction = _kerr_star_dir(
                    state[0], state[1], state[2], state[3], state[4], vph
                )
                # Keep point stars crisp; the caustic column cleanup is done
                # in screen space after the kernel.
                rad += _sample_star(direction, star, sw, sh) * (star_gain * trans)

            c = rad * exposure
            c = ti.Vector([_aces(c[0]), _aces(c[1]), _aces(c[2])])
            c = ti.Vector([
                ti.pow(c[0], 1.0 / 2.2),
                ti.pow(c[1], 1.0 / 2.2),
                ti.pow(c[2], 1.0 / 2.2),
            ])
            hsh = ti.sin(i * 12.9898 + j * 78.233) * 43758.5453
            dth = (hsh - ti.floor(hsh) - 0.5) * jitter
            out[j, i] = ti.Vector([
                ti.min(1.0, ti.max(0.0, c[0] + dth)),
                ti.min(1.0, ti.max(0.0, c[1] + dth)),
                ti.min(1.0, ti.max(0.0, c[2] + dth)),
            ])


class KerrRenderer(LensingRenderer):
    """GPU Kerr ray tracer; reuses the shared sky/LUT/disk textures."""

    def __init__(self, star_size=(2048, 1024), seed=7, output_size=840):
        super().__init__(star_size=star_size, seed=seed, output_size=output_size)

    def render(self, config):
        self.prepare()
        width = int(config.width)
        height = int(config.height)
        out = self._ensure_output(width, height)
        a = max(0.0, min(0.99, float(config.spin)))
        r0 = float(config.distance)
        theta0 = math.radians(float(config.inclination))
        phi0 = math.radians(float(config.azimuth))
        sigma, delta, gtt, gtphi, gphiphi, grr, gthth = kerr_metric(
            r0, theta0, a
        )
        omega_cam = -gtphi / gphiphi
        det = gtphi * gtphi - gtt * gphiphi
        ut_cam = math.sqrt(gphiphi / max(det, 1e-30))
        tan_half = math.tan(0.5 * math.radians(config.fov))
        sin_th0 = math.sin(theta0)
        cos_th0 = math.cos(theta0)
        r_horizon = kerr_horizon(a)
        r_in = max(float(config.r_in), r_horizon + 0.1)
        r_out = max(r_in + 1.0, float(config.r_out))
        r_escape = float(config.r_escape)
        if r_escape <= 0.0:
            r_escape = max(600.0, 18.0 * r0)
        _kerr_kernel(
            out, self._star_field, self._lut_field, self._tex_field,
            512, 256,
            self.star_size[0], self.star_size[1],
            width, height,
            r0, theta0, phi0, tan_half,
            a,
            gtt, gtphi, gphiphi,
            ut_cam, omega_cam,
            math.sqrt(grr), math.sqrt(gthth), math.sqrt(gphiphi),
            sin_th0, cos_th0,
            r_in, r_out, float(config.peak_temperature),
            float(disk_profile_peak(r_in)),
            1 if config.disk_on else 0,
            float(config.opacity),
            1 if config.doppler else 0,
            float(config.exposure), float(config.star_gain),
            float(config.disk_gain),
            int(config.max_steps), float(config.step_scale), r_escape,
            r_horizon, float(config.time), 0.75 / 255.0,
        )
        ti.sync()
        rgb = out.to_numpy()
        if rgb.shape[0] != height or rgb.shape[1] != width:
            rgb = rgb[:height, :width]
        rgb = np.clip(rgb * 255.0 + 0.5, 0.0, 255.0)
        # The exact camera/BH symmetry plane is a measure-zero caustic.  With
        # one point sample per pixel a finite star texture collapses a whole
        # sky meridian into a 1-2 px bright column.  Replace that narrow strip
        # with a horizontal blend of its neighbours; it is a sampling artefact,
        # not a feature of the continuous lensed sky.
        if width >= 20:
            cx = width // 2
            half = max(3, int(round(width * 0.006)))
            left = max(0, cx - half - 1)
            right = min(width - 1, cx + half + 1)
            a_col = rgb[:, left, :].copy()
            b_col = rgb[:, right, :].copy()
            for k in range(left + 1, right):
                t = (k - left) / float(right - left)
                rgb[:, k, :] = a_col * (1.0 - t) + b_col * t
        return rgb.astype(np.uint8)
